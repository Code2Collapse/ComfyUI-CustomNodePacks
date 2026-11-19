"""C2C Distributed nodes: GPU Inventory, Sequence Parallel."""
from __future__ import annotations

import json
import time

from . import enabled, kill_switch_reason
from . import inventory as inv
from . import usp
from . import scheduling as sched
try:                                                       # A28 layout declarations (display only)
    from ..nodes._layout_decl import with_layout
except ImportError:  # loaded as a plain file (tests): the same module by path, never an absolute import
    try:
        import importlib.util as _c2c_ilu
        import os as _c2c_os
        _c2c_spec = _c2c_ilu.spec_from_file_location("c2c_layout_by_path", _c2c_os.path.join(
            _c2c_os.path.dirname(_c2c_os.path.abspath(__file__)), '..', 'nodes', '_layout_decl.py'))
        _c2c_mod = _c2c_ilu.module_from_spec(_c2c_spec)
        _c2c_spec.loader.exec_module(_c2c_mod)
        with_layout = _c2c_mod.with_layout
    except Exception:  # pragma: no cover - layout keys are display only: never take the node down
        def with_layout(spec, layout):
            return spec

CATEGORY = "C2C/Distributed"
RANK_BACKENDS = ("threads", "processes (Linux, NCCL)")


def _refuse_if_off():
    if not enabled():
        raise RuntimeError(kill_switch_reason())


class C2CGPUInventory:
    """What this machine can do with several GPUs: models, architecture, links, measured backends, usable modes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "probe_backends": ("BOOLEAN", {"default": True, "tooltip": "Really create a NCCL / gloo process group in "
                                           "a subprocess to see whether it works (a few seconds, cached until "
                                           "ComfyUI restarts). Off: report only what the hardware says."}),
        }, "optional": {
            "qualify": ("BOOLEAN", {"default": False, "tooltip": "Run the real qualification: one process per "
                                    "physical GPU, all-reduce / all-gather / reduce-scatter checked, latency and "
                                    "bandwidth, a hung rank detected within the timeout, the worker mechanism on "
                                    "every GPU. Tens of seconds. Result: MULTI_GPU_VERIFIED only for two or more "
                                    "distinct GPUs that passed; otherwise SINGLE_GPU_ONLY / FAILED."}),
            # L9.174 (A25), appended: defaults = the behaviour of every saved workflow
            "live_monitor": ("BOOLEAN", {"default": False, "tooltip": "Also read the live state: free / total "
                             "memory per GPU, this process's allocated / reserved memory, utilisation (needs "
                             "pynvml, else null), the models ComfyUI has loaded on each GPU and the ComfyUI queue "
                             "(running / pending prompts). On: the node re-reads on every queue. Result in live_json."}),
            "workload_vram_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4096.0, "step": 0.5, "tooltip":
                                 "Workload suitability: memory one GPU needs for the job, in GB (weights + "
                                 "activations). 0 = off. Each GPU is checked with 10% headroom against its total "
                                 "memory (and its free memory when live_monitor is on). Result in suitability."}),
            "workload_precision": (list(sched.PRECISIONS), {"default": "any", "tooltip": "Workload suitability: "
                                   "the precision the job runs in. bf16 needs cc >= 8.0, fp8 tensor cores cc >= 8.9, "
                                   "fp4 tensor cores cc >= 10.0; older GPUs are reported as emulated (slower)."}),
            "workload_gpus": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Workload suitability: how "
                              "many suitable GPUs the job needs (e.g. 2 for a two-GPU split)."}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("summary", "report_json", "live_json", "suitability")
    OUTPUT_TOOLTIPS = ("Readable summary: GPUs, links, working backends and usable multi-GPU modes.",
                       "The full inventory (and qualification result when run) as JSON.",
                       "live_monitor on: per-GPU free / used memory, utilisation, loaded models and the ComfyUI "
                       "queue, as JSON. Empty when off.",
                       "workload_vram_gb > 0: per GPU whether the workload fits (total / free memory, precision "
                       "support, kernels) and a verdict, as JSON. An estimate, not a measured run. Empty when off.")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("Lists the GPUs (architecture Turing to Blackwell, memory, BF16 / FP8 / FP4 support, whether this "
                   "torch build has kernels for them), how they are linked (P2P, NVLink), Grace / aarch64 hosts, "
                   "whether NCCL / gloo really work here, and which multi-GPU modes this box can run.")

    @classmethod
    def IS_CHANGED(cls, probe_backends=True, **kwargs):
        base = f"{probe_backends}|{kwargs.get('qualify', False)}"
        extra = (kwargs.get("live_monitor", False), kwargs.get("workload_vram_gb", 0.0),
                 kwargs.get("workload_precision", "any"), kwargs.get("workload_gpus", 1))
        if extra == (False, 0.0, "any", 1):
            return base                                      # the saved-workflow fingerprint, unchanged
        live = int(time.time()) if kwargs.get("live_monitor", False) else 0   # live state: re-read every queue
        return f"{base}|{extra}|{live}"

    def run(self, probe_backends, qualify=False, live_monitor=False, workload_vram_gb=0.0,
            workload_precision="any", workload_gpus=1, **_ignored):
        _refuse_if_off()
        rep = inv.inventory(probe_backends=bool(probe_backends))
        if qualify:
            rep["qualification"] = inv.qualify()
        text = inv.summary(rep)
        live = sched.live_snapshot(rep.get("gpus", []) or []) if live_monitor else None
        fit = (sched.suitability(rep, float(workload_vram_gb), workload_precision, int(workload_gpus), live)
               if float(workload_vram_gb) > 0 else None)
        live_s = json.dumps(live, indent=1, default=str) if live is not None else ""
        fit_s = json.dumps(fit, indent=1, default=str) if fit is not None else ""
        ui = [text] + ([fit["verdict"]] if fit is not None else [])
        return {"ui": {"text": ui}, "result": (text, json.dumps(rep, indent=1, default=str), live_s, fit_s)}


# A28 (docs/audit/A28/PARAM_CNP_B.md, C2CSequenceParallel): the token-split tuning sits in a section. Display only.
_SP_LAYOUT = {k: {"c2c_section": "Token split"} for k in ("min_tokens_per_gpu", "fast_link_gpus", "rank_backend")}
_SP_LAYOUT.update({k: {"c2c_section": "Scheduling"} for k in ("memory_policy", "memory_headroom_gb",
                                                                 "preflight_workers")})


class C2CSequenceParallel:
    """Split every model call over several GPUs: batch items for any model, tokens for Wan / MiniMax H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return with_layout({"required": {
            "model": ("MODEL", {"tooltip": "The diffusion model to split; put this node last in the model chain (after LoRA / attention nodes)."}),
            "gpus": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "How many GPUs to use; 0 = all that "
                             "ComfyUI sees. Each GPU holds a full copy of the model."}),
            "strategy": (["auto", "ulysses", "ring"], {"default": "auto", "tooltip": "auto: Ulysses as far as the "
                         "head count and fast links allow, Ring for the rest (USP). ulysses: prefer head-parallel "
                         "all-to-all (best on NVLink). ring: K/V blocks passed around (best across slow links)."}),
            "min_tokens_per_gpu": ("INT", {"default": 1024, "min": 1, "max": 1 << 20, "tooltip": "Calls with fewer "
                                   "tokens per GPU run on one GPU: below this the exchange costs more than it saves."}),
        }, "optional": {
            "fast_link_gpus": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "How many GPUs share fast "
                               "links (NVLink domain). 0 = all of them. Ulysses stays inside this many GPUs."}),
            # A27 GPU-009 (appended; default = the behaviour of every saved workflow)
            "rank_backend": (list(RANK_BACKENDS), {"default": RANK_BACKENDS[0], "tooltip": "threads: every GPU "
                             "is driven by a thread of this ComfyUI process (Windows and Linux, no NCCL; any "
                             "model). processes: one worker process per GPU with its own model copy, talking "
                             "through NCCL (Linux server GPUs; Wan and MiniMax H3 only). Verified on CPU test "
                             "ranks only so far - not yet on real GPUs."}),
            # L9.174 (A25), appended: defaults = the behaviour of every saved workflow
            "memory_policy": (["off", "report", "fit"], {"default": "off", "tooltip": "Memory-aware GPU "
                              "selection. off: as before. report: list each GPU's free memory against one model "
                              "copy (weights + headroom) in the plan. fit: also use only the GPUs that have room "
                              "(in core's device order: the model's GPU first, then the others; the count stops at "
                              "the first GPU without room)."}),
            "memory_headroom_gb": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 256.0, "step": 0.25, "tooltip":
                                   "Memory kept free per GPU on top of the model weights (activations, latents), "
                                   "for memory_policy report / fit."}),
            "preflight_workers": ("BOOLEAN", {"default": False, "tooltip": "Before splitting, run a small real "
                                  "matrix multiply on every GPU that would be used and compare it with the CPU. A "
                                  "failed GPU (no kernels, driver error, wrong result) is left out together with the "
                                  "GPUs after it; a failed model GPU stops with an error. Takes well under a second."}),
        }}, _SP_LAYOUT)

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "plan")
    OUTPUT_TOOLTIPS = ("The model with every call split over the GPUs (unchanged with one GPU).",
                       "The chosen mode, devices, memory per GPU and communication path, as text.")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Splits every model call over the GPUs of this machine. Any model: the call's batch items (cond "
                   "and uncond, an image batch, Tile Split's tiles) go to different GPUs, with no exchange at all. "
                   "Wan 2.1 / 2.2 and MiniMax H3 also split the tokens of one item (Ulysses all-to-all / Ring, exact). "
                   "One FULL model copy per GPU, made by ComfyUI's own multi-GPU clones: weight memory is not pooled "
                   "(this is not model sharding / FSDP). Works on Windows and Linux; with one GPU the model passes "
                   "through unchanged. Put it last in the model chain (after LoRA / attention nodes). Replaces core's "
                   "'MultiGPU CFG Split' for this model.\n\nLimits: speed-up on real multi-GPU hardware is not "
                   "measured yet; ControlNet and self-attention masks are refused on the token-split path; other "
                   "models are batch-split only. The 'plan' output lists the mode, devices, memory per GPU and the "
                   "communication path before sampling starts.")

    def run(self, model, gpus, strategy, min_tokens_per_gpu, fast_link_gpus=0, rank_backend=None,
            memory_policy="off", memory_headroom_gb=1.0, preflight_workers=False):
        _refuse_if_off()
        sched_lines = []
        if memory_policy != "off" or preflight_workers:
            gpus, sched_lines = sched.plan_devices(model, gpus, memory_policy, memory_headroom_gb,
                                                   bool(preflight_workers))
        if rank_backend == RANK_BACKENDS[1]:
            m, report, _state = usp.apply_processes(model, gpus=gpus, prefer=strategy,
                                                    min_tokens_per_gpu=min_tokens_per_gpu, fast_domain=fast_link_gpus)
        else:
            m, report, _state = usp.apply(model, gpus=gpus, prefer=strategy, min_tokens_per_gpu=min_tokens_per_gpu,
                                          fast_domain=fast_link_gpus)
        text = "\n".join(list(report) + sched_lines)
        return {"ui": {"text": [text]}, "result": (m, text)}


NODE_CLASS_MAPPINGS = {
    "C2CGPUInventory": C2CGPUInventory,
    "C2CSequenceParallel": C2CSequenceParallel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "C2CGPUInventory": "C2C GPU Inventory",
    "C2CSequenceParallel": "C2C Multi-GPU Split",
}
