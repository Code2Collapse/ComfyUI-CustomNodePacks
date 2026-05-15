"""integrity_bridge — server-side ring buffer for mask integrity reports.

Sits between MaskTemporalMEC / MaskRefineMEC and the front-end Mask Integrity
HUD sidebar (`js/c2c_mask_integrity_hud.js`). Each time a temporal/refine
node finishes, it calls :func:`publish` with the integrity dict produced by
``compute_integrity``. The HUD polls GET /c2c/mask_integrity/recent every
~1.5 s and renders sparklines of area / centroid drift / IoU plus the list
of flagged frames.

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
            "B": int(report.get("B", 0)),
            "flagged_count": int(report.get("flagged_count", 0)),
            "flagged_frames": list(report.get("flagged_frames", []))[:512],
            "reasons": dict(report.get("reasons", {})) if isinstance(report.get("reasons"), dict) else {},
        }
        if extra and isinstance(extra, dict):
            for k in ("area", "centroid_dx", "centroid_dy", "iou_prev"):
                v = extra.get(k)
                if isinstance(v, list):
                    entry[k] = [float(x) for x in v[:1024]]
        _BUFFER.append(entry)
    except Exception:
        log.exception("[integrity_bridge] publish failed")


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
