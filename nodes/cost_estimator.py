"""
cost_estimator.py — Phase 13: Render Cost Estimator.

Predicts wall-clock time, peak VRAM, and approximate output size for a given
graph BEFORE the user hits Queue. Uses the rolling history from
`mec_diagnostics_api._BUFFER` to compute per-node-class averages.

Strategy
--------
For each (node_class) we maintain a rolling mean of:
    elapsed_ms, vram_peak_mb
from past successful "node_done" events. Unknown classes fall back to a
default of 200 ms / 0 MB so the prediction is never zero.

Route: POST /mec/cost_estimate
Body:  {"workflow": { "<node_id>": {"class_type": "...", "inputs": {...}} } }
Reply:
{
    "success": true,
    "data": {
        "total_ms": float,
        "total_seconds": float,
        "peak_vram_mb": float,
        "per_node": [
            {"node_id": "5", "class_type": "KSampler", "ms": 1234.5,
             "vram_mb": 1024.0, "samples": 17, "confidence": 0.93},
            ...
        ],
        "samples_in_history": int,
        "warnings": [str, ...]
    }
}
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("MEC.cost_estimator")

# Sensible last-resort defaults (ms / MB) per node-class prefix.
_FALLBACK_PROFILE: Dict[str, Tuple[float, float]] = {
    "KSampler":       (4500.0, 1800.0),
    "SamplerCustom":  (4500.0, 1800.0),
    "VAEDecode":      ( 800.0,  600.0),
    "VAEEncode":      ( 400.0,  500.0),
    "CLIPTextEncode": ( 200.0,  100.0),
    "CheckpointLoader": (1200.0, 2500.0),
    "ControlNet":     (1500.0, 800.0),
    "LoraLoader":     ( 300.0, 200.0),
}
_DEFAULT_PROFILE = (200.0, 0.0)


def _collect_history() -> Dict[str, List[Dict[str, Any]]]:
    """Snapshot the diagnostics ring buffer and group by node_class."""
    try:
        from . import mec_diagnostics_api as diag
    except Exception:
        return {}

    buf = getattr(diag, "_BUFFER", None)
    lock = getattr(diag, "_BUFFER_LOCK", None)
    if buf is None:
        return {}

    snap: List[Dict[str, Any]]
    if lock is not None:
        with lock:
            snap = list(buf)
    else:
        snap = list(buf)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for ev in snap:
        if ev.get("type") != "node_done":
            continue
        cls = ev.get("node_class")
        if not cls:
            continue
        grouped.setdefault(cls, []).append(ev)
    return grouped


def _profile_for(cls: str, history: Dict[str, List[Dict[str, Any]]]
                 ) -> Tuple[float, float, int, float]:
    """Return (elapsed_ms_mean, vram_peak_mean, n_samples, confidence)."""
    events = history.get(cls, [])
    n = len(events)
    if n == 0:
        # Try the fallback profile by exact match, then by prefix.
        if cls in _FALLBACK_PROFILE:
            ms, vm = _FALLBACK_PROFILE[cls]
            return ms, vm, 0, 0.1
        for key, val in _FALLBACK_PROFILE.items():
            if cls.startswith(key):
                return val[0], val[1], 0, 0.1
        return _DEFAULT_PROFILE[0], _DEFAULT_PROFILE[1], 0, 0.05

    elapsed = [float(e.get("elapsed_ms") or 0.0) for e in events]
    vram    = [float(e.get("vram_peak_mb") or 0.0) for e in events]
    ms_mean = sum(elapsed) / n
    vm_mean = sum(vram)   / n
    # Confidence: rises with sample count, capped at 0.95.
    confidence = min(0.95, 0.4 + 0.1 * math.log2(n + 1))
    return ms_mean, vm_mean, n, confidence


def estimate(workflow: Dict[str, Any]) -> Dict[str, Any]:
    history = _collect_history()
    per_node: List[Dict[str, Any]] = []
    total_ms = 0.0
    peak_vram = 0.0
    warnings: List[str] = []
    cold_nodes = 0

    if not isinstance(workflow, dict):
        return {
            "total_ms": 0.0, "total_seconds": 0.0, "peak_vram_mb": 0.0,
            "per_node": [], "samples_in_history": 0,
            "warnings": ["workflow_not_dict"],
        }

    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type") or node.get("type") or "Unknown"
        ms, vm, n, conf = _profile_for(cls, history)
        if n == 0:
            cold_nodes += 1
        total_ms += ms
        peak_vram = max(peak_vram, vm)  # Comfy frees between nodes; treat as max.
        per_node.append({
            "node_id":    str(nid),
            "class_type": cls,
            "ms":         round(ms, 1),
            "vram_mb":    round(vm, 1),
            "samples":    n,
            "confidence": round(conf, 2),
        })

    if cold_nodes > 0:
        warnings.append(f"{cold_nodes} node class(es) have no history — "
                        f"estimate uses fallback defaults.")

    return {
        "total_ms":      round(total_ms, 1),
        "total_seconds": round(total_ms / 1000.0, 2),
        "peak_vram_mb":  round(peak_vram, 1),
        "per_node":      per_node,
        "samples_in_history": sum(len(v) for v in history.values()),
        "warnings":      warnings,
    }


def register_routes(server: Any) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[cost_estimator] aiohttp unavailable: %s", e)
        return

    routes = server.routes

    @routes.post("/mec/cost_estimate")
    async def _estimate(req: web.Request) -> web.Response:
        try:
            body = await req.json()
        except Exception:
            body = {}
        wf = body.get("workflow") if isinstance(body, dict) else None
        if not isinstance(wf, dict):
            return web.json_response(
                {"success": False, "error": "missing_workflow"}, status=400)
        try:
            data = estimate(wf)
        except Exception as e:
            log.exception("[cost_estimator] estimate failed")
            return web.json_response(
                {"success": False, "error": "estimate_failed", "message": str(e)},
                status=500)
        return web.json_response({"success": True, "data": data})

    log.info("[cost_estimator] Route registered: POST /mec/cost_estimate")
