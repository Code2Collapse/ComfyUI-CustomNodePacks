"""integrity_bridge — server-side ring buffer for mask integrity reports.

Each time MaskTemporalMEC / MaskRefineMEC finishes, it calls :func:`publish`
with the integrity dict produced by ``compute_integrity``; GET
/c2c/mask_integrity/recent returns the latest reports (area / centroid drift /
IoU per frame plus the flagged frames) and POST /c2c/mask_integrity/clear
empties the buffer. The same report is also each node's JSON output.

L9.302: the Diagnostics sidebar's Masks tab (js/c2c_diagnostics_sidebar.js) shows
these reports. It reads /recent when opened and updates on the websocket event
EVENT, sent once per publish - no polling (the old 1.5 s HUD poller, removed in
2a1575b, stays removed: RAM budget).

The bridge is intentionally tiny and never raises: if aiohttp is missing or
publish is called before any consumer is listening, nothing breaks.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

log = logging.getLogger("MEC.integrity_bridge")

# Ring of the last N integrity reports.
_MAX_REPORTS = 64
_BUFFER: Deque[Dict[str, Any]] = deque(maxlen=_MAX_REPORTS)
EVENT = "c2c.mask_integrity"


def _running_node() -> str:
    """The display id of the node being executed (core sets it right before running a node), or ""."""
    try:
        from server import PromptServer
        nid = getattr(PromptServer.instance, "last_node_id", None)
        return "" if nid in (None, "") else str(nid)
    except Exception:  # noqa: BLE001 - unit tests / no server
        return ""


def _notify(entry: Dict[str, Any]) -> None:
    """One small websocket event per report (no per-frame arrays): the sidebar re-reads /recent if it is open."""
    try:
        from server import PromptServer
        inst = PromptServer.instance
    except Exception:  # noqa: BLE001
        return
    summary = {k: entry[k] for k in ("ts", "source", "node", "B", "flagged_count") if k in entry}
    summary["flagged_frames"] = list(entry.get("flagged_frames", []))[:64]
    try:
        inst.send_sync(EVENT, summary, getattr(inst, "client_id", None))
    except Exception:  # noqa: BLE001 - a report must never fail the node
        log.debug("[integrity_bridge] notify failed", exc_info=True)


def publish(source: str, report: Dict[str, Any],
            extra: Optional[Dict[str, Any]] = None) -> None:
    """Push one integrity report into the ring buffer.

    Parameters
    ----------
    source : str
        Node class name that produced the report (e.g. "MaskTemporalMEC").
    report : dict
        The dict returned by ``compute_integrity``. Must at minimum contain
        ``flagged_frames``, ``flagged_count``, ``B``.
    extra : dict, optional
        Per-frame metric arrays (``area``, ``centroid_dx``, ``centroid_dy``,
        ``iou_prev``) when the producer can supply them.
    """
    try:
        entry = {
            "ts": time.time(),
            "source": source,
            "node": _running_node(),
            "B": int(report.get("B", 0)),
            "flagged_count": int(report.get("flagged_count", 0)),
            "flagged_frames": list(report.get("flagged_frames", []))[:512],
            "reasons": dict(report.get("reasons", {})) if isinstance(report.get("reasons"), dict) else {},
        }
        if not entry["reasons"]:              # compute_integrity keeps the why per frame (frames[i]["warn"])
            for i, rec in enumerate(report.get("frames") or []):
                if isinstance(rec, dict) and rec.get("warn") and len(entry["reasons"]) < 64:
                    entry["reasons"][str(rec.get("i", i))] = str(rec["warn"])
        if extra and isinstance(extra, dict):
            for k in ("area", "centroid_dx", "centroid_dy", "iou_prev"):
                v = extra.get(k)
                if isinstance(v, list):
                    entry[k] = [float(x) for x in v[:1024]]
        _BUFFER.append(entry)
    except Exception:
        log.exception("[integrity_bridge] publish failed")
        return
    _notify(entry)


def get_recent(limit: int = 16) -> List[Dict[str, Any]]:
    if limit <= 0 or limit > _MAX_REPORTS:
        limit = _MAX_REPORTS
    out = list(_BUFFER)[-limit:]
    return out


def clear() -> None:
    _BUFFER.clear()


# ------------------------------------------------------------ aiohttp routes
def register_routes(server: Any) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[integrity_bridge] aiohttp unavailable: %s", e)
        return

    routes = server.routes

    @routes.get("/c2c/mask_integrity/recent")
    async def _recent(req: "web.Request") -> "web.Response":
        try:
            limit = int(req.query.get("limit", "16"))
        except Exception:
            limit = 16
        return web.json_response({"success": True, "data": get_recent(limit)})

    @routes.post("/c2c/mask_integrity/clear")
    async def _clear(_req: "web.Request") -> "web.Response":
        clear()
        return web.json_response({"success": True})

    log.info("[integrity_bridge] Routes registered: GET /c2c/mask_integrity/recent, "
             "POST /c2c/mask_integrity/clear")
