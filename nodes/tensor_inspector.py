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
- Cap memory: keep only the last `_MAX_NODES` snapshots and never store raw
  tensors. Only metadata + scalar reductions.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
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
                if value.is_floating_point():
                    finite = torch.isfinite(value)
                    n_finite = int(finite.sum().item())
                    info["n_nan"] = int(torch.isnan(value).sum().item())
                    info["n_inf"] = int(torch.isinf(value).sum().item())
                    if n_finite > 0:
                        clean = value[finite]
                        info["min"] = float(clean.min().item())
                        info["max"] = float(clean.max().item())
                        info["mean"] = float(clean.mean().item())
                        try:
                            info["std"] = float(clean.float().std().item())
                        except Exception:
                            pass
                    else:
                        info["min"] = info["max"] = info["mean"] = None
                else:
                    info["min"] = int(value.min().item())
                    info["max"] = int(value.max().item())
                    info["mean"] = float(value.float().mean().item())
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


def _record(node_id: Any, prompt_id: Any, output: Any) -> None:
    """Record per-slot summaries for a single executed node."""
    if not isinstance(output, (list, tuple)):
        return
    slots: List[Dict[str, Any]] = []
    for slot_idx, slot_val in enumerate(output[:_MAX_SLOTS_PER_NODE]):
        # ComfyUI's get_output_data returns [slot][batch] — sample the first
        # batch item to keep cost bounded.
        try:
            if isinstance(slot_val, (list, tuple)) and slot_val:
                sample = slot_val[0]
                batch_len = len(slot_val)
            else:
                sample = slot_val
                batch_len = 1
            summary = _summarize_value(sample)
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
    }
    with _STORE_LOCK:
        if key in _STORE:
            _STORE.move_to_end(key)
        _STORE[key] = snapshot
        while len(_STORE) > _MAX_NODES:
            _STORE.popitem(last=False)


# ---------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------
def install() -> bool:
    """Idempotent install. Returns True on success."""
    global _INSTALLED
    if _INSTALLED:
        return True

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
        with _STORE_LOCK:
            snap = _STORE.get(str(node_id))
            data = dict(snap) if snap else None
        if data is None:
            return web.json_response({
                "success": True,
                "data": {"available": False, "node_id": node_id},
            })
        if slot is not None:
            try:
                slot_idx = int(slot)
            except ValueError:
                slot_idx = -1
            data["slots"] = [s for s in data["slots"]
                             if s.get("slot") == slot_idx]
        data["available"] = True
        data["node_id"] = node_id
        return web.json_response({"success": True, "data": data})

    @routes.get("/mec/tensor_snapshot/index")
    async def _index(req: web.Request) -> web.Response:
        with _STORE_LOCK:
            keys = list(_STORE.keys())
        return web.json_response({"success": True, "data": {"node_ids": keys}})

    log.info("[tensor_inspector] Routes registered: /mec/tensor_snapshot[/index]")
