"""
tensor_inspector.py — Phase 4: Live Tensor Inspector.

Wraps `execution.get_output_data` so we record cheap statistical summaries
(shape, dtype, min/max/mean, has_nan) for every slot of every node, keyed by
`(prompt_id, node_id, slot_index)`. The frontend can then query
`/mec/tensor_snapshot?node_id=X&slot=Y` and render a panel with the stats.

Design rules
------------
- Wrapper is async-safe and re-entrant.
- On ANY internal error we still return the original outputs unchanged — we
  never break a running prompt for the sake of inspection.
- ZERO work per node at run time (L2.51): the wrapper records only metadata (shape, dtype, device) and a WEAK
  reference to each output tensor. The statistics are computed when the inspector panel asks for them, and only
  for that node. Reducing every output of every node cost 0.42 s per 97-frame 720p IMAGE on the CPU, and on the
  expanded view core's Mask to Image returns it walked hundreds of millions of single elements (> 60 s for 16
  frames; the owner: "convert mask to image why does it take time").
- Never keep a tensor alive: weak references only. An output ComfyUI has dropped reports "no longer in memory".
- Kill switch: environment C2C_TENSOR_INSPECTOR=0 installs nothing.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional

log = logging.getLogger("MEC.tensor_inspector")

_MAX_NODES = 256          # ring of (node_id) → snapshot
_MAX_SLOTS_PER_NODE = 16  # safety cap per node

# Snapshot store: key=str(node_id) → { prompt_id, ts, slots: [<slot_dict>] }
_STORE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_STORE_LOCK = threading.Lock()

_INSTALLED = False


# ---------------------------------------------------------------------
# Stat extraction (cheap, allocation-aware)
# ---------------------------------------------------------------------
_STAT_CHUNK = 1 << 22    # elements per reduction step (16 MB of fp32)


def _dedupe_broadcast(value):
    """Drop the repeats of a broadcast (stride-0) dimension: ``expand`` views (core's Mask to Image returns one)
    repeat the same elements, so their statistics come from one copy. Returns (tensor, repeat factor)."""
    index, factor = [], 1
    for size, stride in zip(value.shape, value.stride()):
        if stride == 0 and size > 1:
            index.append(slice(0, 1))
            factor *= size
        else:
            index.append(slice(None))
    return (value[tuple(index)] if factor > 1 else value), factor


def _flat_chunks(value):
    """The tensor's elements as 1-D pieces of at most _STAT_CHUNK. A contiguous tensor is not copied; a strided one
    is copied a bounded run of leading slices at a time (never element by element)."""
    if value.ndim == 0:
        yield value.reshape(1)
        return
    if value.is_contiguous():
        flat = value.view(-1)
        for i in range(0, flat.numel(), _STAT_CHUNK):
            yield flat[i:i + _STAT_CHUNK]
        return
    per_slice = max(1, value[0].numel())
    step = max(1, _STAT_CHUNK // per_slice)
    for i in range(0, value.shape[0], step):
        flat = value[i:i + step].contiguous().view(-1)
        for j in range(0, flat.numel(), _STAT_CHUNK):
            yield flat[j:j + _STAT_CHUNK]


def _tensor_stats(value) -> Dict[str, Any]:
    """Exact min / max / mean / std (unbiased) and NaN / Inf counts, reduced chunk by chunk in float64.

    The whole-tensor version (`value[torch.isfinite(value)]`) made boolean-mask indexing build an int64 index of
    [elements, ndim]: summarising one 2K x 24 IMAGE batch (0.59 GB) peaked at 6.1 GB and took 2.4 s - after EVERY
    node, every run (L7.59 / L2.42). Here the extra memory is a few chunks, whatever the size."""
    import math

    import torch

    is_float = value.is_floating_point()
    n = n_nan = n_inf = 0
    vmin, vmax, s, ss = math.inf, -math.inf, 0.0, 0.0
    value, repeat = _dedupe_broadcast(value.detach())
    for c in _flat_chunks(value):
        if is_float:
            nan, inf = torch.isnan(c), torch.isinf(c)
            k_nan, k_inf = int(nan.sum()), int(inf.sum())
            n_nan += k_nan
            n_inf += k_inf
            if k_nan or k_inf:
                c = c[~(nan | inf)]                       # a chunk-sized copy at most
        if c.numel() == 0:
            continue
        c64 = c.double()
        vmin = min(vmin, float(c64.min()))
        vmax = max(vmax, float(c64.max()))
        s += float(c64.sum())
        ss += float((c64 * c64).sum())
        n += c.numel()
    out: Dict[str, Any] = {}
    if is_float:
        out["n_nan"], out["n_inf"] = n_nan * repeat, n_inf * repeat   # counts are of the full (broadcast) tensor
    if n == 0:
        out["min"] = out["max"] = out["mean"] = None
        return out
    mean = s / n
    out["min"] = vmin if is_float else int(vmin)
    out["max"] = vmax if is_float else int(vmax)
    out["mean"] = mean
    if n > 1:
        out["std"] = math.sqrt(max(0.0, (ss - s * mean) / (n - 1)))
    return out


def _summarize_value(value: Any) -> Dict[str, Any]:
    """Return a JSON-serializable summary for any node output value.

    For torch tensors we compute reductions in-graph (no copy). For lists we
    summarize the first element. For dicts we list keys. For everything else
    we record only the repr-truncated type and length.
    """
    info: Dict[str, Any] = {}

    # --- torch tensor ----------------------------------------------------
    try:
        import torch  # type: ignore
        if isinstance(value, torch.Tensor):
            info["kind"] = "tensor"
            info["shape"] = list(value.shape)
            info["dtype"] = str(value.dtype).replace("torch.", "")
            info["device"] = str(value.device)
            info["numel"] = int(value.numel())
            info["requires_grad"] = bool(value.requires_grad)
            if value.numel() == 0:
                info["empty"] = True
                return info
            try:
                info.update(_tensor_stats(value))
            except Exception as e:
                info["stat_error"] = str(e)
            return info
    except Exception:
        # torch not importable — fall through
        pass

    # --- numpy ------------------------------------------------------------
    try:
        import numpy as _np  # type: ignore
        if isinstance(value, _np.ndarray):
            info["kind"] = "ndarray"
            info["shape"] = list(value.shape)
            info["dtype"] = str(value.dtype)
            if value.size > 0:
                try:
                    info["min"] = float(value.min())
                    info["max"] = float(value.max())
                    info["mean"] = float(value.mean())
                except Exception:
                    pass
            return info
    except Exception:
        pass

    # --- dict (LATENT/CONDITIONING/etc) ----------------------------------
    if isinstance(value, dict):
        info["kind"] = "dict"
        info["keys"] = list(value.keys())[:32]
        # Drill into a `samples` tensor (LATENT common case)
        if "samples" in value:
            info["samples"] = _summarize_value(value["samples"])
        return info

    # --- list/tuple ------------------------------------------------------
    if isinstance(value, (list, tuple)):
        info["kind"] = type(value).__name__
        info["len"] = len(value)
        if value:
            info["first"] = _summarize_value(value[0])
        return info

    # --- str / bytes -----------------------------------------------------
    if isinstance(value, (str, bytes)):
        info["kind"] = type(value).__name__
        info["len"] = len(value)
        if isinstance(value, str):
            info["preview"] = value[:80]
        return info

    # --- scalar ----------------------------------------------------------
    if isinstance(value, (int, float, bool)) or value is None:
        info["kind"] = type(value).__name__ if value is not None else "NoneType"
        info["value"] = value
        return info

    info["kind"] = type(value).__name__
    info["repr"] = repr(value)[:120]
    return info


def _describe(value: Any, refs: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    """Run-time description: metadata only, O(1). Tensors are held by WEAK reference in ``refs`` so their
    statistics can be computed later, on request (``_fill_stats``)."""
    try:
        import torch  # type: ignore
        if isinstance(value, torch.Tensor):
            info = {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", ""),
                    "device": str(value.device), "numel": int(value.numel()),
                    "requires_grad": bool(value.requires_grad)}
            if value.numel() == 0:
                info["empty"] = True
            else:
                try:
                    refs[path] = weakref.ref(value)
                    info["stats"] = "on request"
                except TypeError:
                    info["stats"] = "unavailable"
            return info
    except Exception:
        pass
    if isinstance(value, dict):
        info = {"kind": "dict", "keys": list(value.keys())[:32]}
        if "samples" in value:
            info["samples"] = _describe(value["samples"], refs, path + ".samples")
        return info
    if isinstance(value, (list, tuple)):
        info = {"kind": type(value).__name__, "len": len(value)}
        if value:
            info["first"] = _describe(value[0], refs, path + ".first")
        return info
    return _summarize_value(value)          # strings, numbers, numpy, other: already cheap


def _fill_stats(summary: Dict[str, Any], refs: Dict[str, Any], path: str = "") -> None:
    """Compute the deferred statistics in place (exact, chunked). Called by the route, never during a run."""
    if summary.get("kind") == "tensor" and summary.get("stats") == "on request":
        ref = refs.get(path)
        tensor = ref() if ref is not None else None
        if tensor is None:
            summary["stats"] = "no longer in memory"
            return
        try:
            summary.update(_tensor_stats(tensor))
            summary["stats"] = "exact"
        except Exception as e:
            summary["stat_error"] = str(e)
            summary["stats"] = "failed"
        return
    if isinstance(summary.get("samples"), dict):
        _fill_stats(summary["samples"], refs, path + ".samples")
    if isinstance(summary.get("first"), dict):
        _fill_stats(summary["first"], refs, path + ".first")


def _record(node_id: Any, prompt_id: Any, output: Any) -> None:
    """Record per-slot descriptions for a single executed node. O(1) per slot: no reductions here."""
    if not isinstance(output, (list, tuple)):
        return
    slots: List[Dict[str, Any]] = []
    refs: Dict[str, Any] = {}
    for slot_idx, slot_val in enumerate(output[:_MAX_SLOTS_PER_NODE]):
        # ComfyUI's get_output_data returns [slot][batch] — describe the first batch item.
        try:
            if isinstance(slot_val, (list, tuple)) and slot_val:
                sample = slot_val[0]
                batch_len = len(slot_val)
            else:
                sample = slot_val
                batch_len = 1
            summary = _describe(sample, refs, str(slot_idx))
            summary["slot"] = slot_idx
            summary["batch_len"] = batch_len
            slots.append(summary)
        except Exception as e:
            slots.append({"slot": slot_idx, "error": str(e)})

    key = str(node_id)
    snapshot = {
        "prompt_id": str(prompt_id) if prompt_id is not None else None,
        "ts": time.time(),
        "slots": slots,
        "_refs": refs,
    }
    with _STORE_LOCK:
        if key in _STORE:
            _STORE.move_to_end(key)
        _STORE[key] = snapshot
        while len(_STORE) > _MAX_NODES:
            _STORE.popitem(last=False)


def snapshot_with_stats(node_id: Any, slot: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """The stored snapshot of one node with its statistics computed now (JSON-ready), or None."""
    import copy

    with _STORE_LOCK:
        snap = _STORE.get(str(node_id))
        if snap is None:
            return None
        refs = snap.get("_refs") or {}
        slots = copy.deepcopy(snap["slots"])
        data = {"prompt_id": snap.get("prompt_id"), "ts": snap.get("ts")}
    if slot is not None:
        slots = [x for x in slots if x.get("slot") == slot]
    for summary in slots:
        _fill_stats(summary, refs, str(summary.get("slot")))
    data["slots"] = slots
    return data


# ---------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------
def install() -> bool:
    """Idempotent install. Returns True on success."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get("C2C_TENSOR_INSPECTOR", "").strip() == "0":
        log.info("[tensor_inspector] disabled by C2C_TENSOR_INSPECTOR=0")
        return False

    try:
        import execution as _exec  # type: ignore
    except Exception as e:
        log.warning("[tensor_inspector] cannot import execution: %s", e)
        return False

    target = getattr(_exec, "get_output_data", None)
    if target is None or not asyncio.iscoroutinefunction(target):
        log.warning("[tensor_inspector] get_output_data not async-coroutine — abort")
        return False

    if getattr(target, "_mec_inspector_wrapped", False):
        _INSTALLED = True
        return True

    async def wrapped(prompt_id, unique_id, obj, input_data_all,
                      execution_block_cb=None, pre_execute_cb=None, v3_data=None):
        result = await target(
            prompt_id, unique_id, obj, input_data_all,
            execution_block_cb=execution_block_cb,
            pre_execute_cb=pre_execute_cb,
            v3_data=v3_data,
        )
        # Result shape: (output, ui, has_subgraph, has_pending_task)
        try:
            if isinstance(result, tuple) and len(result) >= 1:
                output = result[0]
                has_pending = result[3] if len(result) >= 4 else False
                if not has_pending:
                    _record(unique_id, prompt_id, output)
        except Exception as e:
            log.debug("[tensor_inspector] record failed: %s", e)
        return result

    wrapped._mec_inspector_wrapped = True  # type: ignore[attr-defined]
    setattr(_exec, "get_output_data", wrapped)
    _INSTALLED = True
    log.info("[tensor_inspector] wrapped execution.get_output_data")
    return True


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
def register_routes(server: Any) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[tensor_inspector] aiohttp unavailable: %s", e)
        return

    install()
    routes = server.routes

    @routes.get("/mec/tensor_snapshot")
    async def _snap(req: web.Request) -> web.Response:
        node_id = req.query.get("node_id")
        slot = req.query.get("slot")
        if not node_id:
            return web.json_response(
                {"success": False, "error": "missing_node_id"}, status=400)
        slot_idx = None
        if slot is not None:
            try:
                slot_idx = int(slot)
            except ValueError:
                slot_idx = -1
        # exact statistics of this one node, computed now and off the event loop (L2.51: never during a run)
        data = await asyncio.get_running_loop().run_in_executor(None, snapshot_with_stats, node_id, slot_idx)
        if data is None:
            return web.json_response({
                "success": True,
                "data": {"available": False, "node_id": node_id},
            })
        data["available"] = True
        data["node_id"] = node_id
        return web.json_response({"success": True, "data": data})

    @routes.get("/mec/tensor_snapshot/index")
    async def _index(req: web.Request) -> web.Response:
        with _STORE_LOCK:
            keys = list(_STORE.keys())
        return web.json_response({"success": True, "data": {"node_ids": keys}})

    log.info("[tensor_inspector] Routes registered: /mec/tensor_snapshot[/index]")
