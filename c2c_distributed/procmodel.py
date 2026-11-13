"""A ComfyUI model on worker PROCESSES, for sequence parallelism over procrun ranks (A27 GPU-009).

This is raylight's "one actor per GPU, each with its own model" idea, done with ComfyUI's own mechanisms:

* Recipe. Core already knows how to rebuild a model it loaded: ``ModelPatcher.cached_patcher_init`` = (loader
  function, its arguments), e.g. ``(comfy.sd.load_diffusion_model, (path, model_options))``. Core uses the same
  factory for its multi-GPU clones (model_patcher.deepclone_multigpu). The recipe is that factory plus the
  patcher's weight patches (LoRA etc.: ``ModelPatcher.patches``). Both are picklable data. No model code is copied.
* Load. Each worker rebuilds the model from the recipe, applies the weight patches and loads it with
  ``comfy.model_management.load_models_gpu``, on its own GPU (``cuda:rank``; CPU for gloo test ranks).
* Call. Every model call of the sampler (rank 0 = the ComfyUI process) is sent to the workers: the call's tensors
  (moved to CPU and pickled) plus the plain-data part of transformer_options. All ranks then run the model's own
  forward with usp's block patch and attention override, and the token split / Ulysses / Ring exchange runs over
  torch.distributed (``comm.TorchComm``). Rank 0 returns the gathered output to the sampler.

Refused rather than run wrong. These conditions are checked, and the run stops with the reason:
- a loader without a reload factory;
- hook patches;
- object patches on the diffusion model;
- weight patches or transformer_options entries that cannot be pickled;
- other DIFFUSION_MODEL wrappers after the Split node, or other patch kinds: they would run on rank 0 only;
- an attention override set before the Split node (it cannot be sent to another process);
- models without a sequence adapter (batch split stays on the thread path).

Limits: the call's tensors travel through the command store every step, which is correct but not the fastest
transport; a broadcast over NCCL is the next step once real GPUs can measure it.
Status: SIMULATED (gloo CPU ranks in WSL, core 0.39 sampler, tiny Wan / H3); NCCL on GPUs
CODE_COMPLETE_HARDWARE_UNVERIFIED.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch

_SKIP_OPTIONS = ("wrappers", "callbacks", "patches", "patches_replace", "optimized_attention_override",
                 "c2c_usp", "c2c_usp_ranks", "c2c_usp_pool", "c2c_tile_dispatch")


class NotDistributable(ValueError):
    """The model or the call cannot be reproduced exactly on another process."""


# ── recipe (parent side) ───────────────────────────────────────────────────────────────────────────────────────

def _fn_ref(fn) -> Tuple[str, str]:
    mod, name = getattr(fn, "__module__", None), getattr(fn, "__qualname__", None)
    if not mod or not name or "<locals>" in name:
        raise NotDistributable(f"the model's loader {fn!r} is not a module-level function, so another process "
                               f"cannot call it")
    return mod, name


def recipe_of(patcher) -> dict:
    """Everything a worker needs to rebuild this exact model (weights + weight patches), as picklable data."""
    init = getattr(patcher, "cached_patcher_init", None)
    if init is None:
        raise NotDistributable("this model's loader cannot reload it (no cached_patcher_init): use a core loader "
                               "(UNETLoader / CheckpointLoaderSimple)")
    if getattr(patcher, "hook_patches", None):
        raise NotDistributable("hook patches (scheduled LoRA hooks) are on this model: not supported across processes")
    bad = [k for k in (getattr(patcher, "object_patches", None) or {}) if str(k).startswith("diffusion_model")]
    if bad:
        raise NotDistributable(f"object patches on the diffusion model ({', '.join(map(str, bad))}) cannot be "
                               f"reproduced in another process")
    mod, name = _fn_ref(init[0])
    import comfy.sd                                 # "comfy" is a namespace package: no __file__ of its own
    rec = {"init": (mod, name, tuple(init[1]), init[2] if len(init) > 2 else None),
           "patches": dict(getattr(patcher, "patches", {}) or {}),
           "core_dir": os.path.dirname(os.path.dirname(os.path.abspath(comfy.sd.__file__))),
           "cpu": _cli_cpu()}
    try:
        blob = pickle.dumps(rec)
    except Exception as exc:  # noqa: BLE001
        raise NotDistributable(f"the model's loader arguments or weight patches cannot be sent to another process "
                               f"({type(exc).__name__}: {exc})") from exc
    rec["key"] = hashlib.sha256(blob).hexdigest()[:20]
    return rec


def _cli_cpu() -> bool:
    try:
        import comfy.cli_args
        return bool(comfy.cli_args.args.cpu)
    except Exception:  # noqa: BLE001
        return False


def call_options(to: dict) -> dict:
    """The plain-data part of transformer_options for the workers; anything else is refused, never dropped."""
    out, bad = {}, []
    for k, v in to.items():
        if k in _SKIP_OPTIONS:
            continue
        try:
            pickle.dumps(v)
        except Exception:  # noqa: BLE001
            bad.append(str(k))
            continue
        out[k] = _to_cpu(v)
    if bad:
        raise NotDistributable(f"transformer_options entries that cannot be sent to another process: "
                               f"{', '.join(sorted(bad))} (a node in the model chain set them)")
    extra = sorted(kind for kind, v in (to.get("patches") or {}).items() if v)
    if extra:
        raise NotDistributable(f"model patches '{', '.join(extra)}' would run on the first GPU only")
    return out


def _to_cpu(obj):
    from .usp import _move
    return _move(obj, torch.device("cpu"))


# ── worker side ────────────────────────────────────────────────────────────────────────────────────────────────

_MODELS: Dict[str, Any] = {}


def _core(rec: dict) -> None:
    import sys
    if rec["core_dir"] not in sys.path:
        sys.path.insert(0, rec["core_dir"])
    import comfy.cli_args
    if rec.get("cpu"):
        comfy.cli_args.args.cpu = True


def rank_load(c, rec: dict):
    """Rank > 0: rebuild the model from the recipe and load it on this rank's device. Rank 0 already has it."""
    import importlib
    from . import procrun
    if c.rank == 0:
        return {"rank": 0}
    _core(rec)
    import comfy.model_management as mm
    mod, name, args, index = rec["init"]
    fn = importlib.import_module(mod)
    for part in name.split("."):
        fn = getattr(fn, part)
    patcher = fn(*args)
    if index is not None:
        patcher = patcher[index]
    if rec["patches"]:
        patcher.patches = {k: list(v) for k, v in rec["patches"].items()}
        import uuid
        patcher.patches_uuid = uuid.uuid4()
    dev = procrun.device()
    patcher.load_device = dev
    mm.load_models_gpu([patcher], force_full_load=True)
    _MODELS.clear()                                 # one model per worker group
    _MODELS[rec["key"]] = patcher
    dm = patcher.model.diffusion_model
    return {"rank": c.rank, "device": str(dev), "class": type(dm).__name__,
            "parameters": int(sum(p.numel() for p in dm.parameters())), "weight_patches": len(rec["patches"])}


def rank_call(c, payload: dict):
    """Rank > 0: the same model call as rank 0, on this rank's copy, with the token split."""
    from . import attention as A
    from . import procrun, usp
    patcher = _MODELS.get(payload["key"])
    if patcher is None:
        raise RuntimeError("this worker has no model loaded (the group was restarted?)")
    dm = patcher.model.diffusion_model
    dev = procrun.device()
    ad = usp.adapter_for(dm)
    to = usp._move(payload["options"], dev)
    to["patches_replace"] = {"dit": {("double_block", i): usp.block_patch for i in range(len(dm.blocks))}}
    to["optimized_attention_override"] = usp.attention_override
    args = [usp._move(a, dev) for a in payload["args"]]
    kwargs = {k: usp._move(v, dev) for k, v in payload["kwargs"].items()}
    if payload["pos"] is None:
        kwargs["transformer_options"] = to
    else:
        args[payload["pos"]] = to
    u, r = payload["plan"]
    usp._tls.ctx = usp.RankCtx(comm=c, mesh=A.Mesh.build(c, u, r), adapter=ad, n_blocks=len(dm.blocks))
    try:
        with torch.no_grad():
            getattr(dm, payload["method"])(*args, **kwargs)
    finally:
        usp._tls.ctx = None
    return None


# ── parent side: a worker group per model ──────────────────────────────────────────────────────────────────────

class ModelGroup:
    """The worker processes holding one model. Restarted (and the model reloaded) when a run breaks the group."""

    def __init__(self, rec: dict, world: int, backend: Optional[str] = None, device_kind: Optional[str] = None,
                 timeout_s: float = 600.0):
        self.rec, self.world = rec, int(world)
        self.backend, self.device_kind, self.timeout_s = backend, device_kind, float(timeout_s)
        self.pr = None
        self.info: List[dict] = []
        self.starts = 0

    def ensure(self):
        from . import procrun
        if self.pr is not None and not self.pr.broken and not self.pr.closed:
            return self.pr
        if self.pr is not None:
            self.pr.close()
        self.pr = procrun.ProcessRanks(self.world, self.backend, self.device_kind, timeout_s=self.timeout_s).start()
        self.starts += 1
        self.info = self.pr.run(rank_load, self.rec)
        return self.pr

    @property
    def reason(self) -> str:
        from . import procrun
        return procrun.choose(self.world, self.backend, self.device_kind)[2]

    def close(self):
        if self.pr is not None:
            self.pr.close()
            self.pr = None


_GROUPS: Dict[str, ModelGroup] = {}


def group_for(rec: dict, world: int, backend=None, device_kind=None, timeout_s: float = 600.0) -> ModelGroup:
    """One live group at a time (each holds a full model per GPU): a different model closes the previous group."""
    key = f"{rec['key']}|{world}|{backend}|{device_kind}"
    g = _GROUPS.get(key)
    if g is None:
        for other in list(_GROUPS.values()):
            other.close()
        _GROUPS.clear()
        g = _GROUPS[key] = ModelGroup(rec, world, backend, device_kind, timeout_s)
    return g


def close_all() -> None:
    for g in list(_GROUPS.values()):
        g.close()
    _GROUPS.clear()
