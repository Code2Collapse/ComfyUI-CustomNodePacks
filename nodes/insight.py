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

import asyncio
import functools
import logging
import os
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import torch

log = logging.getLogger("MEC.insight")

try:
    import psutil  # type: ignore
    _PROC = psutil.Process(os.getpid())
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore
    _PROC = None


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

    # The actual per-node call goes through different functions depending on
    # ComfyUI version. We wrap whichever exists, in priority order:
    #   - current (2024+): top-level `execute(server, dynprompt, caches,
    #     current_item, extra_data, executed, prompt_id, ...)`
    #   - older          : `recursive_execute(...)`
    #   - ancient        : `_map_node_over_list(...)`
    candidates = [
        ("execute",             getattr(_exec, "execute", None)),
        ("recursive_execute",   getattr(_exec, "recursive_execute", None)),
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
        log.warning("[insight] no execute / recursive_execute / _map_node_over_list found")
        return False

    def _extract_node_id(args, kwargs):
        for key in ("unique_id", "node_id", "current_item"):
            if key in kwargs:
                return kwargs[key]
        if len(args) >= 4:
            # current `execute(server, dynprompt, caches, current_item, ...)` and
            # older `recursive_execute(server, prompt, outputs, current_item, ...)`
            return args[3]
        return None

    def _snapshot_mem():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return (torch.cuda.memory_allocated(),
                    torch.cuda.max_memory_allocated())
        return (0, 0)

    def _snapshot_cpu() -> Tuple[float, int]:
        """(process_cpu_seconds, rss_bytes). Falls back to time.process_time()
        + 0 when psutil isn't available."""
        try:
            if _PROC is not None:
                ts = _PROC.cpu_times()
                cpu = float(ts.user) + float(ts.system)
                rss = int(_PROC.memory_info().rss)
                return cpu, rss
        except Exception:
            pass
        return float(time.process_time()), 0

    def _parent_subgraph_id(node_id) -> Optional[str]:
        """Subgraph child node ids are encoded as "<parent>:<child>" (or
        deeper with multiple ':'). Return the top-level parent id when the
        node is inside a subgraph, else None."""
        if isinstance(node_id, str) and ":" in node_id:
            return node_id.split(":", 1)[0]
        return None

    def _emit_done(node_id, t0, cpu0, rss0, mem_before, peak_before, result):
        elapsed = (time.time() - t0) * 1000.0
        mem_after, peak_after = _snapshot_mem()
        cpu1, rss1 = _snapshot_cpu()
        # ComfyUI's `execute()` returns an ExecutionResult enum. On FAILURE
        # it doesn't raise — it returns the failure value. Detect that.
        is_failure = False
        try:
            ER = getattr(_exec, "ExecutionResult", None)
            if ER is not None and result is not None:
                fail_val = getattr(ER, "FAILURE", None)
                # `execute` returns a tuple (ExecutionResult, error, ex) on the
                # current API; older versions just return the enum.
                check = result[0] if isinstance(result, tuple) else result
                if fail_val is not None and check == fail_val:
                    is_failure = True
        except Exception:
            pass
        ev = {
            "type": "node_error" if is_failure else "node_done",
            "node_id": node_id,
            "parent_id": _parent_subgraph_id(node_id),
            "elapsed_ms": elapsed,
            "cpu_ms": max(0.0, (cpu1 - cpu0) * 1000.0),
            "vram_delta_mb": (mem_after - mem_before) / (1 << 20),
            "vram_peak_mb": (peak_after - peak_before) / (1 << 20),
            "ram_delta_mb": (rss1 - rss0) / (1 << 20) if rss0 else 0.0,
        }
        if is_failure and isinstance(result, tuple) and len(result) >= 3:
            err, ex = result[1], result[2]
            if isinstance(err, dict):
                ev["exc_type"] = err.get("exception_type") or (
                    type(ex).__name__ if ex else "Exception")
                ev["exc_msg"] = err.get("exception_message") or (
                    str(ex) if ex else "")
            else:
                ev["exc_type"] = type(ex).__name__ if ex else "Exception"
                ev["exc_msg"] = str(ex) if ex else ""
            ev["hint"] = explain_exception(ex) if ex else ""
        _emit(ev)

    def _emit_raise(node_id, t0, e):
        elapsed = (time.time() - t0) * 1000.0
        _emit({
            "type": "node_error",
            "node_id": node_id,
            "parent_id": _parent_subgraph_id(node_id),
            "elapsed_ms": elapsed,
            "exc_type": type(e).__name__,
            "exc_msg": str(e),
            "hint": explain_exception(e),
            "trace": traceback.format_exc(limit=12),
        })

    if asyncio.iscoroutinefunction(target_fn):
        @functools.wraps(target_fn)
        async def wrapped(*args, **kwargs):
            node_id = _extract_node_id(args, kwargs)
            mem_before, peak_before = _snapshot_mem()
            cpu0, rss0 = _snapshot_cpu()
            t0 = time.time()
            try:
                result = await target_fn(*args, **kwargs)
                _emit_done(node_id, t0, cpu0, rss0, mem_before, peak_before, result)
                return result
            except BaseException as e:
                _emit_raise(node_id, t0, e)
                raise
    else:
        @functools.wraps(target_fn)
        def wrapped(*args, **kwargs):
            node_id = _extract_node_id(args, kwargs)
            mem_before, peak_before = _snapshot_mem()
            cpu0, rss0 = _snapshot_cpu()
            t0 = time.time()
            try:
                result = target_fn(*args, **kwargs)
                _emit_done(node_id, t0, cpu0, rss0, mem_before, peak_before, result)
                return result
            except BaseException as e:
                _emit_raise(node_id, t0, e)
                raise

    setattr(_exec, target_fn_name, wrapped)
    log.info("[insight] wrapped execution.%s (async=%s)",
             target_fn_name, asyncio.iscoroutinefunction(target_fn))
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
