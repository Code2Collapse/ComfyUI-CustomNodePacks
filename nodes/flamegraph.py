"""
flamegraph.py — Phase 3: Execution Flame Graph.

Reads the ring buffer maintained by `mec_diagnostics_api._BUFFER` (populated by
`insight.py`'s execute() hook) and exposes a route that returns a sorted list
of per-node timing rows ready to render as a horizontal bar chart.

Route
-----
GET /mec/diagnostics/flamegraph?limit=50&prompt_id=...
    Returns:
        {
            "success": true,
            "data": {
                "prompt_id":   "...",
                "total_ms":    <sum>,
                "node_count":  N,
                "rows": [
                    {"node_id":"5", "node_class":"KSampler", "elapsed_ms":1234.5,
                     "cpu_ms":..., "vram_delta_mb":..., "error":false},
                    ...
                ]
            }
        }

The frontend (`js/mec_diagnostics_sidebar.js`) renders the "Flame" tab from
this payload.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("MEC.flamegraph")


def _events_snapshot() -> List[Dict[str, Any]]:
    """Take a thread-safe snapshot of the diagnostics ring buffer."""
    try:
        from . import mec_diagnostics_api as diag  # type: ignore
    except Exception as e:
        log.warning("[flamegraph] mec_diagnostics_api unavailable: %s", e)
        return []
    with diag._BUFFER_LOCK:
        return list(diag._BUFFER)


def _resolve_node_class(node_id: Any) -> str:
    """Look up the node's class name from ComfyUI's live PromptServer state.

    Insight events only carry node_id; ComfyUI keeps the live prompt graph
    on PromptServer.instance.prompt_queue. We do a best-effort lookup.
    """
    if node_id is None:
        return "unknown"
    try:
        import server as _comfy_server  # type: ignore
        ps = _comfy_server.PromptServer.instance
        if ps is None:
            return str(node_id)
        # Live executing prompt: PromptServer keeps the last_prompt_id and prompt dict
        last_prompt = getattr(ps, "last_prompt", None) or getattr(ps, "prompt", None)
        if isinstance(last_prompt, dict):
            node = last_prompt.get(str(node_id))
            if isinstance(node, dict):
                ct = node.get("class_type")
                if ct:
                    return str(ct)
        # Fallback: search queue items
        queue = getattr(ps, "prompt_queue", None)
        if queue is not None:
            try:
                hist = queue.get_history()
                for _pid, entry in (hist or {}).items():
                    prompt = entry.get("prompt") if isinstance(entry, dict) else None
                    if isinstance(prompt, list) and len(prompt) >= 3:
                        graph = prompt[2]
                        if isinstance(graph, dict):
                            node = graph.get(str(node_id))
                            if isinstance(node, dict):
                                ct = node.get("class_type")
                                if ct:
                                    return str(ct)
            except Exception:
                pass
    except Exception:
        pass
    return str(node_id)


def build_flamegraph(prompt_id: Optional[str] = None,
                     limit: int = 50) -> Dict[str, Any]:
    """Aggregate the ring buffer into a sorted timing report."""
    events = _events_snapshot()

    # Filter to node_done / node_error events with a non-zero elapsed_ms
    rows: List[Dict[str, Any]] = []
    last_prompt_id: Optional[str] = None
    for ev in events:
        etype = ev.get("type")
        if etype not in ("node_done", "node_error"):
            continue
        pid = ev.get("prompt_id")
        if pid:
            last_prompt_id = pid
        if prompt_id and pid and pid != prompt_id:
            continue
        node_id = ev.get("node_id")
        rows.append({
            "node_id":       node_id,
            "node_class":    _resolve_node_class(node_id),
            "elapsed_ms":    float(ev.get("elapsed_ms") or 0.0),
            "cpu_ms":        float(ev.get("cpu_ms") or 0.0),
            "vram_delta_mb": float(ev.get("vram_delta_mb") or 0.0),
            "vram_peak_mb":  float(ev.get("vram_peak_mb") or 0.0),
            "ram_delta_mb":  float(ev.get("ram_delta_mb") or 0.0),
            "error":         etype == "node_error",
            "exc_type":      ev.get("exc_type"),
            "exc_msg":       ev.get("exc_msg"),
            "ts":            ev.get("ts"),
            "prompt_id":     pid,
        })

    # If no prompt_id was given, restrict to the most recent prompt
    if not prompt_id and last_prompt_id is not None:
        rows = [r for r in rows if r.get("prompt_id") == last_prompt_id]

    rows.sort(key=lambda r: r["elapsed_ms"], reverse=True)
    rows = rows[:max(1, int(limit))]

    total_ms = sum(r["elapsed_ms"] for r in rows)

    return {
        "prompt_id":  prompt_id or last_prompt_id,
        "total_ms":   round(total_ms, 2),
        "node_count": len(rows),
        "rows":       rows,
    }


def register_routes(server: Any) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[flamegraph] aiohttp unavailable: %s", e)
        return

    routes = server.routes

    @routes.get("/mec/diagnostics/flamegraph")
    async def _flame(req: web.Request) -> web.Response:
        try:
            limit = int(req.query.get("limit", "50"))
        except ValueError:
            limit = 50
        prompt_id = req.query.get("prompt_id") or None

        try:
            data = build_flamegraph(prompt_id=prompt_id, limit=limit)
        except Exception as e:
            log.exception("[flamegraph] build failed")
            return web.json_response(
                {"success": False, "error": "build_failed", "message": str(e)},
                status=500,
            )
        return web.json_response({"success": True, "data": data})

    log.info("[flamegraph] Route registered: GET /mec/diagnostics/flamegraph")
