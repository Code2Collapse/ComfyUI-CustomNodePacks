"""
error_introspector.py — Track D.3: Runtime tensor-introspection translator.

This is the **non-LLM, fact-based** layer of the 3-tier error explainer.
Where ``error_assistant`` matches the exception text against a regex pack
(Tier 1 rule-pack), this module looks at the **actual runtime state** when
the error fired:

    * the deepest traceback frame's ``f_locals`` — sampling every live
      ``torch.Tensor`` / ``torch.nn.Module`` (shape, dtype, device);
    * the owning module's ``__class__.__module__`` to identify the model
      family (``comfy.ldm.flux``, ``comfy.ldm.wan``, ``comfy.ldm.modules
      .diffusionmodules.openaimodel`` → SD1.5/SDXL UNet, etc.);
    * the inputs the upstream node had emitted (from
      ``tensor_inspector._STORE``) for the node that just failed and any
      directly-named LATENT/IMAGE/MASK arguments.

It synthesises a structured ``IntrospectionReport`` describing the failure
in concrete numbers — for example::

    Conv2D shape mismatch at comfy.ldm.flux.model.Flux:
        x:      [1, 16, 128, 128] fp16 cuda:0
        weight: [320, 4, 3, 3]    fp16 cuda:0
    Input has 16 channels, weight expects 4.
    Likely cause: a Flux/SDXL latent was fed into an SD1.5 UNet — swap the
    checkpoint or VAE so the latent channel count matches.

The module is **non-invasive**:

    * Pure analysis: never raises, never mutates the exception.
    * Always returns *something* — even a minimal ``{frames: [...]}``
      dump is better than the raw Python traceback for a newbie.
    * Memory-bounded: the inspector never holds a reference to a tensor
      beyond the function call; only scalar reductions (shape/dtype/device,
      no raw data) escape.

Public surface
--------------
:func:`introspect_exception(exc, tb=None, node_id=None, node_class=None,
                            prompt_id=None) -> IntrospectionReport`
    The main entry point. ``tb`` defaults to ``exc.__traceback__``.

:func:`format_report(report) -> dict`
    Convert the dataclass into a JSON-serialisable explainer envelope that
    matches what ``error_assistant.explain`` returns (``tier``, ``headline``,
    ``cause``, ``fixes``, …) so the route layer can return it directly.

:func:`register_routes(server)`
    Mount ``POST /mec/introspect_error`` for the JS toast / Doctor.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("MEC.error_introspector")


# ─────────────────────────────────────────────────────────────────────────
# Limits — keep the inspector cheap. Worst-case per call: a few hundred
# lightweight dict allocations and a handful of ``int()`` casts.
# ─────────────────────────────────────────────────────────────────────────
_MAX_FRAMES_SCANNED = 8            # walk the deepest N frames of the tb
_MAX_TENSORS_PER_FRAME = 24        # cap names per frame
_MAX_MODULES_PER_FRAME = 8
_MAX_REPR_LEN = 120
_SHAPE_PREVIEW_LEN = 8             # cap on dim count we print verbatim


# ─────────────────────────────────────────────────────────────────────────
# Model-family fingerprints — purely by ``module.__class__.__module__``
# prefix. Order matters: most-specific first.
# ─────────────────────────────────────────────────────────────────────────
_FAMILY_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("comfy.ldm.wan.",                 "Wan (Wan 2.1/2.2/Fun/Animate)"),
    ("comfy.ldm.flux.",                "Flux"),
    ("comfy.ldm.sd3.",                 "SD3 / SD3.5"),
    ("comfy.ldm.hidream.",             "HiDream"),
    ("comfy.ldm.hunyuan_video.",       "HunyuanVideo"),
    ("comfy.ldm.cosmos.",              "Cosmos"),
    ("comfy.ldm.mochi.",               "Mochi"),
    ("comfy.ldm.lightricks.",          "LTX-Video"),
    ("comfy.ldm.aura.",                "AuraFlow"),
    ("comfy.ldm.cascade.",             "StableCascade"),
    ("comfy.ldm.audio.",               "StableAudio"),
    ("comfy.ldm.qwen_image.",          "Qwen-Image / Qwen-Image-Edit"),
    ("comfy.ldm.pixart.",              "PixArt"),
    ("comfy.ldm.modules.diffusionmodules.openaimodel",
                                       "SD1.5 / SDXL UNet"),
    ("comfy.ldm.modules.encoders.noise_aug_modules", "SD-CLIP encoder"),
    ("comfy.sd",                       "Comfy core (SD loader/VAE)"),
    ("comfy.ldm.models.autoencoder",   "VAE"),
    ("comfy.text_encoders.",           "Text encoder (T5/CLIP/UMT5/Llama)"),
)


def _identify_family(qualname: str) -> Optional[str]:
    if not qualname:
        return None
    for prefix, label in _FAMILY_PREFIXES:
        if qualname.startswith(prefix):
            return label
    return None


# ─────────────────────────────────────────────────────────────────────────
# Tensor / module summarisation — borrowed in spirit from
# ``tensor_inspector._summarize_value`` but stripped to only the cheap
# metadata (no min/max/std reductions; we are post-mortem on a failing
# tensor and ``isnan`` could itself raise).
# ─────────────────────────────────────────────────────────────────────────
def _safe_shape(t: Any) -> Optional[List[int]]:
    try:
        return list(t.shape)[:_SHAPE_PREVIEW_LEN]
    except Exception:
        return None


def _safe_dtype(t: Any) -> Optional[str]:
    try:
        return str(t.dtype).replace("torch.", "")
    except Exception:
        return None


def _safe_device(t: Any) -> Optional[str]:
    try:
        return str(t.device)
    except Exception:
        return None


def _summarise_tensor(name: str, t: Any) -> Dict[str, Any]:
    return {
        "name":   name,
        "kind":   "tensor",
        "shape":  _safe_shape(t),
        "dtype":  _safe_dtype(t),
        "device": _safe_device(t),
    }


def _summarise_module(name: str, m: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "name":   name,
        "kind":   "module",
        "class":  type(m).__name__,
        "module": getattr(type(m), "__module__", ""),
    }
    family = _identify_family(info["module"])
    if family:
        info["family"] = family
    # Pull the weight tensor if present — covers Conv2d / Linear / etc.
    weight = getattr(m, "weight", None)
    if weight is not None and _safe_shape(weight) is not None:
        info["weight_shape"] = _safe_shape(weight)
        info["weight_dtype"] = _safe_dtype(weight)
    # In_channels / out_channels is the most user-facing fact for Conv2d.
    for attr in ("in_channels", "out_channels", "in_features", "out_features",
                 "num_heads", "head_dim", "embed_dim"):
        v = getattr(m, attr, None)
        if isinstance(v, int):
            info[attr] = v
    return info


def _collect_from_locals(local_dict: Dict[str, Any]) -> Tuple[List[Dict[str, Any]],
                                                              List[Dict[str, Any]]]:
    """Return (tensors, modules) summaries from one frame's f_locals.

    Robust: never raises; silently skips anything weird.
    """
    tensors: List[Dict[str, Any]] = []
    modules: List[Dict[str, Any]] = []
    try:
        import torch  # type: ignore
    except Exception:
        return tensors, modules

    tensor_cls = torch.Tensor
    module_cls = torch.nn.Module

    for name, val in list(local_dict.items())[:200]:
        if name.startswith("__"):
            continue
        try:
            if isinstance(val, tensor_cls):
                if len(tensors) < _MAX_TENSORS_PER_FRAME:
                    tensors.append(_summarise_tensor(name, val))
            elif isinstance(val, module_cls):
                if len(modules) < _MAX_MODULES_PER_FRAME:
                    modules.append(_summarise_module(name, val))
            elif isinstance(val, dict) and "samples" in val and isinstance(
                    val.get("samples"), tensor_cls):
                # LATENT envelope — surface the inner tensor.
                if len(tensors) < _MAX_TENSORS_PER_FRAME:
                    tensors.append(_summarise_tensor(
                        f"{name}['samples']", val["samples"]))
            elif isinstance(val, (list, tuple)) and val and isinstance(
                    val[0], tensor_cls):
                if len(tensors) < _MAX_TENSORS_PER_FRAME:
                    tensors.append(_summarise_tensor(f"{name}[0]", val[0]))
        except Exception:
            continue
    return tensors, modules


# ─────────────────────────────────────────────────────────────────────────
# Heuristic interpreter — turn the raw frame dump into one English sentence.
# Each heuristic returns ``(cause: str, fixes: List[str]) | None``. The
# first match wins. Heuristics are intentionally narrow; if none fire we
# still emit the structured frames so the user has actionable facts.
# ─────────────────────────────────────────────────────────────────────────
_MSG_CONV_CHANNEL_RE = re.compile(
    r"expected input\[[^\]]+\]\s*to have\s*(\d+)\s*channels?,\s*but got\s*(\d+)\s*channels?",
    re.IGNORECASE)
_MSG_MAT1_MAT2_RE = re.compile(
    r"mat1\s*and\s*mat2\s*shapes\s*cannot\s*be\s*multiplied\s*"
    r"\(?(\d+)x(\d+)\s*and\s*(\d+)x(\d+)\)?",
    re.IGNORECASE)
_MSG_SIZE_MISMATCH_RE = re.compile(
    r"size of tensor a\s*\((\d+)\)\s*must match the size of tensor b\s*\((\d+)\)"
    r".*?at non-singleton dimension\s*(\d+)",
    re.IGNORECASE | re.DOTALL)
_MSG_DTYPE_RE = re.compile(
    r"expected\s+(\S+)\s+but got\s+(\S+)",
    re.IGNORECASE)
# PyTorch ≥2.x form: "expected m1 and m2 to have the same dtype, but got: float != struct c10::Half"
_MSG_DTYPE_MISMATCH_RE = re.compile(
    r"(?:same\s+dtype|same\s+scalar\s*type).*?but got[:\s]+([A-Za-z0-9:_]+)\s*!=\s*([A-Za-z0-9:_]+)",
    re.IGNORECASE | re.DOTALL)
_MSG_DEVICE_RE = re.compile(
    r"expected (?:all tensors to be on the same device|device\s*=?\s*[\"']?(\w+)[\"']?)"
    r".*?(?:on\s+(cuda(?::\d+)?|cpu|meta|mps))",
    re.IGNORECASE | re.DOTALL)
_MSG_OOM_RE = re.compile(r"out of memory|cuda out of memory", re.IGNORECASE)


def _format_shape(shape: Optional[List[int]]) -> str:
    if not shape:
        return "?"
    return "[" + ", ".join(str(d) for d in shape) + "]"


def _find_tensor(frames: List["FrameInfo"], name_substr: str) -> Optional[Dict[str, Any]]:
    """Return the first tensor whose name contains ``name_substr`` (case-insensitive)."""
    needle = name_substr.lower()
    for fr in frames:
        for t in fr.tensors:
            if needle in t["name"].lower():
                return t
    return None


def _find_module_with_weight_channels(frames: List["FrameInfo"], in_ch: int
                                      ) -> Optional[Dict[str, Any]]:
    for fr in frames:
        for m in fr.modules:
            if m.get("in_channels") == in_ch:
                return m
    return None


def _h_conv_channels(exc_msg: str, frames: List["FrameInfo"]
                     ) -> Optional[Tuple[str, List[str]]]:
    m = _MSG_CONV_CHANNEL_RE.search(exc_msg)
    if not m:
        return None
    expected, got = int(m.group(1)), int(m.group(2))
    mod = _find_module_with_weight_channels(frames, expected)
    family = mod.get("family") if mod else None
    family_str = f" ({family})" if family else ""
    cause = (
        f"Conv2D channel mismatch{family_str}: this layer expects "
        f"{expected} input channels but received a tensor with {got}."
    )
    fixes: List[str] = []
    # Latent-channel cheat-sheet: SD1.5/SDXL=4, Flux/SD3=16, Wan=16, Cascade=16, HunyuanVideo=16.
    if expected == 4 and got == 16:
        fixes.append(
            "The downstream model is SD1.5/SDXL (4-channel latents), "
            "but you fed it a Flux/SD3/Wan/HunyuanVideo latent (16-channel). "
            "Either switch the checkpoint to Flux/SD3/Wan or pass the latent "
            "through the matching VAE first.")
    elif expected == 16 and got == 4:
        fixes.append(
            "The downstream model is Flux/SD3/Wan (16-channel latents), "
            "but you fed it an SD1.5/SDXL latent (4-channel). Re-encode "
            "the image with the matching VAE (e.g. FluxVAE / SD3VAE).")
    elif expected == 3 and got in (1, 4):
        fixes.append(
            "Conv expects a 3-channel image; got "
            f"{got}. If {got}==1 your tensor is grayscale — duplicate it to "
            "3 channels. If 4, you have an RGBA — drop the alpha.")
    fixes.append("Insert a Tensor Inspector before this node to confirm the live shape.")
    fixes.append("Verify the loader you used produced a latent for the right model family.")
    return cause, fixes


def _h_mat1_mat2(exc_msg: str, frames: List["FrameInfo"]
                 ) -> Optional[Tuple[str, List[str]]]:
    m = _MSG_MAT1_MAT2_RE.search(exc_msg)
    if not m:
        return None
    a1, a2, b1, b2 = (int(x) for x in m.groups())
    cause = (
        f"Linear/MatMul shape mismatch: tried to multiply a [{a1}×{a2}] tensor "
        f"by a [{b1}×{b2}] tensor — the inner dimensions ({a2} vs {b1}) must agree."
    )
    fixes: List[str] = []
    # Famous case: CLIP-L vs CLIP-G/OpenCLIP dim mix (768 vs 1280 vs 1024).
    clip_dims = {768: "CLIP-L (SD1.5/SDXL)",
                 1024: "OpenCLIP-H (SD2.x)",
                 1280: "CLIP-G (SDXL/SD3)",
                 4096: "T5-XXL"}
    if a2 in clip_dims and b1 in clip_dims and a2 != b1:
        fixes.append(
            f"Text-encoder dim swap: input is {clip_dims[a2]} ({a2}) but "
            f"model wants {clip_dims[b1]} ({b1}). Re-check which CLIP / T5 the "
            "checkpoint expects.")
    fixes.append(
        "Most often: a CLIP-Text-Encode using one CLIP model was wired into a "
        "model loaded with a different CLIP. Use the matching CLIPLoader.")
    fixes.append("If you stacked LoRAs, one may target the wrong base model — disable to confirm.")
    return cause, fixes


def _h_size_mismatch(exc_msg: str, frames: List["FrameInfo"]
                     ) -> Optional[Tuple[str, List[str]]]:
    m = _MSG_SIZE_MISMATCH_RE.search(exc_msg)
    if not m:
        return None
    a, b, dim = int(m.group(1)), int(m.group(2)), int(m.group(3))
    cause = (
        f"Two tensors disagree at dimension {dim}: {a} vs {b}. "
        "Broadcasting only works when one side is 1."
    )
    fixes: List[str] = []
    if dim in (1, 2, 3) and a % 8 != 0:
        fixes.append(f"Dim {dim}={a} is not a multiple of 8 — many UNets/VAEs "
                     "round spatial dims to /8 or /16. Match resolution.")
    fixes.append("Likely cause: an upstream node resized image/mask differently than the model expects.")
    fixes.append("Insert a Tensor Inspector on both inputs to compare shapes side-by-side.")
    return cause, fixes


def _h_dtype(exc_msg: str, frames: List["FrameInfo"]
             ) -> Optional[Tuple[str, List[str]]]:
    m2 = _MSG_DTYPE_MISMATCH_RE.search(exc_msg)
    if m2:
        a, b = m2.group(1), m2.group(2)
        cause = (
            f"dtype mismatch — the two operand tensors had different element "
            f"types ({a} vs {b}); PyTorch will not auto-promote across "
            "Linear/MatMul/Conv."
        )
        fixes = [
            f"Cast one side to match: `tensor.to(dtype=torch.{a.split('::')[-1].lower()})` upstream.",
            "If you toggled `--force-fp16` or `--bf16`, your model weights changed dtype but the input is still float32 — re-encode through the matching VAE/CLIP.",
            "Quantised loaders (FP8/GGUF/torchao) can produce mixed-dtype outputs — verify with a Tensor Inspector.",
        ]
        return cause, fixes

    m = _MSG_DTYPE_RE.search(exc_msg)
    if not m:
        return None
    expected, got = m.group(1), m.group(2)
    # Only fire when both look like torch dtypes / scalar types.
    if "torch" not in expected.lower() and "float" not in expected.lower() \
            and "half" not in expected.lower() and "bfloat" not in expected.lower():
        return None
    cause = f"dtype mismatch — operation expected {expected} but received {got}."
    fixes = [
        "Cast the offending tensor with `.to(dtype=…)` upstream.",
        f"If you forced fp8/bf16 on the model, the input tensor is still {got}; "
        "either remove `--force-fp16/--bf16` flags or re-encode through a matching VAE.",
        "ModelSamplingDiscreteFlow / ModelMergeWeighted can silently change dtypes — verify with a Tensor Inspector.",
    ]
    return cause, fixes


def _h_device(exc_msg: str, frames: List["FrameInfo"]
              ) -> Optional[Tuple[str, List[str]]]:
    if "device" not in exc_msg.lower():
        return None
    if not _MSG_DEVICE_RE.search(exc_msg) and "expected all tensors" not in exc_msg.lower():
        return None
    cuda_tensors = sum(1 for fr in frames for t in fr.tensors
                       if (t.get("device") or "").startswith("cuda"))
    cpu_tensors  = sum(1 for fr in frames for t in fr.tensors
                       if (t.get("device") or "") == "cpu")
    cause = (
        f"Cross-device call: at the failure point there were {cuda_tensors} "
        f"CUDA tensor(s) and {cpu_tensors} CPU tensor(s) in scope, but the op "
        "needs them all on the same device."
    )
    fixes = [
        "A model loaded with `--cpu` cannot consume CUDA tensors. "
        "Either move the offending node off CPU or pre-move inputs with a `.to('cpu')`.",
        "If you used `--lowvram` / `--novram`, an encoder may have been off-loaded; "
        "force-reload the affected model.",
        "Check for a stray `.cuda()` / `.cpu()` in a custom node you recently installed.",
    ]
    return cause, fixes


def _h_oom(exc_msg: str, frames: List["FrameInfo"]
           ) -> Optional[Tuple[str, List[str]]]:
    if not _MSG_OOM_RE.search(exc_msg):
        return None
    biggest = None
    biggest_numel = -1
    for fr in frames:
        for t in fr.tensors:
            sh = t.get("shape") or []
            n = 1
            for d in sh:
                n *= max(1, int(d))
            if n > biggest_numel:
                biggest_numel = n
                biggest = t
    largest_str = (
        f" The largest live tensor at failure was `{biggest['name']}` "
        f"shape={_format_shape(biggest.get('shape'))} dtype={biggest.get('dtype')}."
        if biggest else "")
    free_str = ""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            free_str = (f" Free VRAM at the moment of the error: "
                        f"{free / 1e9:.2f} GB / {total / 1e9:.2f} GB.")
    except Exception:
        pass
    cause = "GPU ran out of VRAM." + largest_str + free_str
    fixes = [
        "Drop resolution / batch size by one step (e.g. 1024→768, batch 2→1).",
        "Restart ComfyUI with `--lowvram` (or `--novram` for tiny GPUs).",
        "Use `tiled` VAE Decode / Encode for ≥1024px images.",
        "Insert an Insight Status node before this node to run `torch.cuda.empty_cache()`.",
    ]
    return cause, fixes


_HEURISTICS = (
    _h_conv_channels,
    _h_mat1_mat2,
    _h_size_mismatch,
    _h_dtype,
    _h_device,
    _h_oom,
)


# ─────────────────────────────────────────────────────────────────────────
# Report dataclasses
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class FrameInfo:
    file:    str
    line:    int
    func:    str
    self_class: Optional[str] = None
    self_module: Optional[str] = None
    family:  Optional[str] = None
    tensors: List[Dict[str, Any]] = field(default_factory=list)
    modules: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "file": self.file,
            "line": self.line,
            "func": self.func,
            "tensors": self.tensors,
            "modules": self.modules,
        }
        if self.self_class:
            out["self_class"] = self.self_class
        if self.self_module:
            out["self_module"] = self.self_module
        if self.family:
            out["family"] = self.family
        return out


@dataclass
class IntrospectionReport:
    exc_type:  str
    exc_msg:   str
    node_id:   Optional[str]
    node_class: Optional[str]
    prompt_id: Optional[str]
    family:    Optional[str]
    frames:    List[FrameInfo]
    cause:     Optional[str] = None
    fixes:     List[str] = field(default_factory=list)
    matched_heuristic: Optional[str] = None
    upstream_snapshot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exc_type": self.exc_type,
            "exc_msg":  self.exc_msg,
            "node_id":  self.node_id,
            "node_class": self.node_class,
            "prompt_id": self.prompt_id,
            "family":    self.family,
            "frames":    [f.to_dict() for f in self.frames],
            "cause":     self.cause,
            "fixes":     list(self.fixes),
            "matched_heuristic": self.matched_heuristic,
            "upstream_snapshot": self.upstream_snapshot,
        }


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────
_LOCK = threading.Lock()


def _project_root_short(path: str) -> str:
    """Trim file paths to something readable; never leak full user homedir."""
    try:
        p = os.path.normpath(path)
        # Show the last 3 segments
        parts = p.split(os.sep)
        if len(parts) > 3:
            return os.sep.join(parts[-3:])
        return p
    except Exception:
        return path


def _walk_frames(tb: "traceback.TracebackException | Any") -> List[FrameInfo]:
    """Walk a real traceback (``types.TracebackType``) and harvest frames.

    We need the *live* tb, not a ``TracebackException`` summary, because the
    summary discards ``f_locals``.
    """
    frames: List[FrameInfo] = []
    cur = tb
    # Walk to the deepest frame and collect upwards.
    chain: List[Any] = []
    while cur is not None:
        chain.append(cur)
        cur = getattr(cur, "tb_next", None)
    # Deepest last; we want deepest first in the report.
    for tb_frame in reversed(chain[-_MAX_FRAMES_SCANNED:]):
        try:
            frame = tb_frame.tb_frame
            code  = frame.f_code
            fi = FrameInfo(
                file=_project_root_short(code.co_filename),
                line=tb_frame.tb_lineno or code.co_firstlineno,
                func=code.co_name,
            )
            local_dict = dict(frame.f_locals)  # copy: do not pin the live frame
            # `self.__class__` identifies the owning module / model family.
            self_obj = local_dict.get("self")
            if self_obj is not None:
                try:
                    cls = type(self_obj)
                    fi.self_class = cls.__name__
                    fi.self_module = getattr(cls, "__module__", "")
                    fi.family = _identify_family(fi.self_module or "")
                except Exception:
                    pass
            fi.tensors, fi.modules = _collect_from_locals(local_dict)
            frames.append(fi)
        except Exception as e:
            log.debug("[error_introspector] frame walk skipped: %s", e)
    return frames


def _fetch_upstream_snapshot(node_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Look up the most-recent recorded output for ``node_id``.

    Uses the live ``tensor_inspector._STORE`` if the module is loaded; safe
    no-op otherwise.
    """
    if node_id is None:
        return None
    try:
        from . import tensor_inspector  # type: ignore
    except Exception:
        return None
    try:
        store = getattr(tensor_inspector, "_STORE", None)
        lock = getattr(tensor_inspector, "_STORE_LOCK", None)
        if store is None or lock is None:
            return None
        with lock:
            snap = store.get(str(node_id))
            return dict(snap) if snap else None
    except Exception:
        return None


def introspect_exception(exc: BaseException,
                          tb: Any = None,
                          *,
                          node_id: Optional[str] = None,
                          node_class: Optional[str] = None,
                          prompt_id: Optional[str] = None,
                          ) -> IntrospectionReport:
    """Analyse a live exception and return a structured report.

    Never raises — on any internal error we still return a minimal report
    so the caller can fall through to the rule pack.
    """
    if tb is None:
        tb = getattr(exc, "__traceback__", None)
    exc_type = type(exc).__name__
    exc_msg  = str(exc) or repr(exc)

    with _LOCK:
        try:
            frames = _walk_frames(tb) if tb is not None else []
        except Exception as e:
            log.debug("[error_introspector] walk failed: %s", e)
            frames = []

    # Identify family from any frame
    family: Optional[str] = None
    for fr in frames:
        if fr.family:
            family = fr.family
            break

    report = IntrospectionReport(
        exc_type=exc_type, exc_msg=exc_msg,
        node_id=str(node_id) if node_id is not None else None,
        node_class=node_class,
        prompt_id=str(prompt_id) if prompt_id is not None else None,
        family=family, frames=frames,
    )
    report.upstream_snapshot = _fetch_upstream_snapshot(report.node_id)

    # Apply heuristics; first one to fire wins.
    for h in _HEURISTICS:
        try:
            res = h(exc_msg, frames)
        except Exception as e:
            log.debug("[error_introspector] heuristic %s raised: %s",
                      h.__name__, e)
            res = None
        if res is not None:
            report.cause, report.fixes = res
            report.matched_heuristic = h.__name__
            break

    return report


# ─────────────────────────────────────────────────────────────────────────
# Format → explainer envelope
# ─────────────────────────────────────────────────────────────────────────
def _headline_from_report(r: IntrospectionReport) -> str:
    head = f"{r.exc_type}: {r.exc_msg.strip().splitlines()[0][:140]}"
    if r.family:
        head = f"[{r.family}] " + head
    return head


def _fact_summary(r: IntrospectionReport) -> str:
    """One-paragraph plain-English dump of the most useful runtime facts."""
    if not r.frames:
        return "No live tensor information was available at the failure point."
    fr = r.frames[0]
    parts: List[str] = []
    if fr.self_class:
        loc = f"in {fr.self_class}"
        if fr.family:
            loc += f" ({fr.family})"
        parts.append(loc)
    parts.append(f"at {fr.file}:{fr.line} ({fr.func})")
    head = "Failure " + " ".join(parts) + "."
    bits: List[str] = [head]
    for t in fr.tensors[:6]:
        bits.append(
            f"  • {t['name']}: shape={_format_shape(t.get('shape'))} "
            f"dtype={t.get('dtype')} device={t.get('device')}")
    for m in fr.modules[:3]:
        line = f"  • module `{m['name']}` = {m.get('class')}"
        if m.get("in_channels") is not None or m.get("in_features") is not None:
            in_ = m.get("in_channels", m.get("in_features"))
            out_ = m.get("out_channels", m.get("out_features"))
            line += f" (in={in_}, out={out_})"
        if m.get("weight_shape"):
            line += f"  weight={_format_shape(m['weight_shape'])}"
        bits.append(line)
    return "\n".join(bits)


def format_report(report: IntrospectionReport) -> Dict[str, Any]:
    """Convert a report into the same envelope shape as ``error_assistant.explain``.

    Tier label is ``"introspector"`` so the UI can badge it differently from
    the regex pack (which uses ``tier: 1``).
    """
    cause = report.cause or _fact_summary(report)
    fixes = list(report.fixes) if report.fixes else [
        "Open the 'Live Tensor Inspector' panel for the failing node to see the exact shape/dtype it produced.",
        "Check that the loaders feeding this node belong to the same model family "
        f"({report.family or 'see family badge in INT'}).",
        "Insert a Tensor Inspector node between the loader and this node to confirm what arrived.",
    ]
    out: Dict[str, Any] = {
        "tier": "introspector",
        "headline": _headline_from_report(report),
        "cause": cause,
        "fixes": fixes,
        "pattern_id": report.matched_heuristic or "introspector_facts_only",
        "category":   "runtime_introspection",
        "confidence": 0.9 if report.matched_heuristic else 0.5,
        "provenance": {
            "pack":   "ComfyUI-CustomNodePacks",
            "source": "error_introspector",
        },
        "introspection": report.to_dict(),
    }
    if report.family:
        out["model_family"] = report.family
    return out


# ─────────────────────────────────────────────────────────────────────────
# REST route
# ─────────────────────────────────────────────────────────────────────────
def register_routes(server: Any) -> None:
    """Register ``POST /mec/introspect_error``.

    Body shape (all optional except ``exc_type`` + ``message``)::

        {
          "exc_type":      "RuntimeError",
          "message":       "Expected input[1, 16, …] to have 4 channels …",
          "node_id":       "23",
          "node_class":    "KSampler",
          "prompt_id":     "abc-123",
          "traceback":     "<full text>",
        }

    Returns the formatted envelope. Note: when invoked **after** the fact
    (from the JS toast), the live traceback / f_locals is gone, so the
    report is heuristic-only — based purely on ``message`` parsing plus
    any snapshot in ``tensor_inspector._STORE``. The live path is
    :func:`introspect_exception` called *during* exception handling.
    """
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[error_introspector] aiohttp unavailable: %s", e)
        return

    routes = server.routes

    @routes.post("/mec/introspect_error")
    async def _introspect(req):  # noqa: ANN001
        try:
            body = await req.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        exc_type   = str(body.get("exc_type") or "Exception")
        message    = str(body.get("message")  or "").strip()
        node_id    = body.get("node_id")
        node_class = body.get("node_class")
        prompt_id  = body.get("prompt_id")
        if not message:
            return web.json_response(
                {"success": False, "error": "missing_message"}, status=400)
        # Build a synthetic exception so we exercise the same code path.
        # The traceback is lost at this point — heuristics will run on the
        # message text alone, augmented by upstream snapshot.
        cls = type(exc_type, (Exception,), {})
        exc = cls(message)
        report = introspect_exception(
            exc,
            tb=None,
            node_id=str(node_id) if node_id is not None else None,
            node_class=str(node_class) if node_class else None,
            prompt_id=str(prompt_id) if prompt_id else None,
        )
        envelope = format_report(report)
        return web.json_response({"success": True, "data": envelope})

    log.info("[error_introspector] Routes registered: POST /mec/introspect_error")
