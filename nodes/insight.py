# FILE: nodes/insight.py
# FEATURE: F6 — Insight Diagnostic (executor wrap, per-node VRAM delta, exception → human text)
# INTEGRATES WITH: web/extensions/nukenodemax/insight_overlay.js
"""
Wraps `comfy.graph_execution.GraphExecution.execute` (or the closest available
analogue across ComfyUI versions) so we capture, per-node:

    - torch.cuda.memory_allocated() before / after  (delta MB)
    - elapsed wall-clock ms
    - exception type + human-readable hint via lookup dict

Events are pushed to the JS frontend through ComfyUI's PromptServer socket
under event name "nukenodemax.insight". The JS side renders DOM heatmap
overlays — strictly NO console.log fallback for telemetry.
"""
from __future__ import annotations

import functools
import logging
import time
import traceback
from typing import Any, Dict

import torch

log = logging.getLogger("MEC.insight")


# =====================================================================
# Exception → human hint
# =====================================================================
_HINTS = {
    "ModuleNotFoundError": (
        "A Python package is missing. Run `pip install <package>` "
        "in the ComfyUI venv."
    ),
    "FileNotFoundError": (
        "A model / weight file is missing. Check models/ and any custom path "
        "in extra_model_paths.yaml."
    ),
    "RuntimeError_cuda_oom": (
        "CUDA ran out of VRAM. Lower batch size, enable --lowvram, or use "
        "ProPainterTemporalMEC `use_half=True`."
    ),
    "RuntimeError_size_mismatch": (
        "Tensor shape mismatch. Check that previous nodes output the resolution "
        "the consumer expects."
    ),
    "AttributeError_NoneType": (
        "An upstream node returned None. Most often a model loader failed "
        "silently — re-check filename, path, or download integrity."
    ),
    "TypeError": (
        "Wrong argument type passed. Often a wire from IMAGE→MASK or vice-versa."
    ),
    "KeyError": (
        "A required dict key is missing. If it's `stitch_data`, an upstream "
        "InpaintCropProMEC didn't run."
    ),
    "ValueError": "Argument out of allowed range — re-check widget bounds.",
}


def explain_exception(exc: BaseException) -> str:
    name = type(exc).__name__
    msg = str(exc)
    if name == "RuntimeError" and "out of memory" in msg.lower():
        return _HINTS["RuntimeError_cuda_oom"]
    if name == "RuntimeError" and "size" in msg.lower() and "match" in msg.lower():
        return _HINTS["RuntimeError_size_mismatch"]
    if name == "AttributeError" and "NoneType" in msg:
        return _HINTS["AttributeError_NoneType"]
    return _HINTS.get(name, "No specific hint — read traceback for details.")


# =====================================================================
# Telemetry sink (PromptServer socket)
# =====================================================================
def _emit(event: Dict[str, Any]) -> None:
    try:
        import server as _comfy_server  # type: ignore
        ps = _comfy_server.PromptServer.instance
        ps.send_sync("nukenodemax.insight", event)
    except Exception as e:
        log.debug("[insight] socket emit failed: %s", e)


# =====================================================================
# Executor wrapper
# =====================================================================
_INSTALLED = False


def install() -> bool:
    """Idempotent install. Returns True on success."""
    global _INSTALLED
    if _INSTALLED:
        return True

    try:
        import execution as _exec  # ComfyUI top-level execution module
    except Exception as e:
        log.warning("[insight] cannot import ComfyUI `execution` module: %s", e)
        return False

    target_cls = None
    target_attr = None
    for name in ("PromptExecutor", "GraphExecution"):
        cls = getattr(_exec, name, None)
        if cls is not None:
            target_cls = cls
            break
    if target_cls is None:
        log.warning("[insight] no PromptExecutor / GraphExecution class found")
        return False

    # The actual per-node call goes through `_map_node_over_list` or
    # `recursive_execute`. We wrap whichever exists.
    candidates = [
        ("recursive_execute", getattr(_exec, "recursive_execute", None)),
        ("_map_node_over_list", getattr(_exec, "_map_node_over_list", None)),
    ]
    target_fn_name = None
    target_fn = None
    for n, fn in candidates:
        if callable(fn):
            target_fn_name = n
            target_fn = fn
            break
    if target_fn is None:
        log.warning("[insight] no recursive_execute / _map_node_over_list found")
        return False

    @functools.wraps(target_fn)
    def wrapped(*args, **kwargs):
        node_id = None
        # Heuristic: ComfyUI passes `unique_id` either positionally or as kwarg.
        for key in ("unique_id", "node_id"):
            if key in kwargs:
                node_id = kwargs[key]
                break
        if node_id is None and len(args) >= 4:
            node_id = args[3]

        torch_dev_avail = torch.cuda.is_available()
        if torch_dev_avail:
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            peak_before = torch.cuda.max_memory_allocated()
        else:
            mem_before = peak_before = 0
        t0 = time.time()
        try:
            result = target_fn(*args, **kwargs)
            elapsed = (time.time() - t0) * 1000.0
            if torch_dev_avail:
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated()
                peak_after = torch.cuda.max_memory_allocated()
            else:
                mem_after = peak_after = 0
            _emit({
                "type": "node_done",
                "node_id": node_id,
                "elapsed_ms": elapsed,
                "vram_delta_mb": (mem_after - mem_before) / (1 << 20),
                "vram_peak_mb": (peak_after - peak_before) / (1 << 20),
            })
            return result
        except BaseException as e:
            elapsed = (time.time() - t0) * 1000.0
            _emit({
                "type": "node_error",
                "node_id": node_id,
                "elapsed_ms": elapsed,
                "exc_type": type(e).__name__,
                "exc_msg": str(e),
                "hint": explain_exception(e),
                "trace": traceback.format_exc(limit=12),
            })
            raise

    setattr(_exec, target_fn_name, wrapped)
    log.info("[insight] wrapped execution.%s", target_fn_name)
    _INSTALLED = True
    return True


# =====================================================================
# Diagnostic node — exposes status into the graph
# =====================================================================
class InsightStatusMEC:
    DESCRIPTION = ("Reports whether the Insight executor wrap is installed, plus "
                   "the current torch/cuda memory snapshot.")
    CATEGORY = "MaskEditControl/Diagnostic"
    FUNCTION = "report"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def report(self):
        installed = install()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            mb = lambda x: f"{x / (1 << 20):.0f}MB"
            mem = (f"cuda free={mb(free)}/{mb(total)} "
                   f"alloc={mb(torch.cuda.memory_allocated())} "
                   f"peak={mb(torch.cuda.max_memory_allocated())}")
        else:
            mem = "cpu"
        return (f"insight_installed={installed} | {mem}",)


NODE_CLASS_MAPPINGS = {"InsightStatusMEC": InsightStatusMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"InsightStatusMEC": "Insight Status (MEC)"}
