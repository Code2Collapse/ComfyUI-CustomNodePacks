"""
mec_diagnostics_api.py — Backend for the MEC Diagnostics sidebar.

Doctor-style endpoints exposed under `/mec/diagnostics/*`:

    GET  /mec/diagnostics/recent           Recent execution events (ring buffer).
    GET  /mec/diagnostics/statistics       Aggregated counters per category / pattern.
    GET  /mec/diagnostics/patterns         Loaded pattern packs + counts.
    POST /mec/diagnostics/reload_patterns  Force pattern hot-reload.
    GET  /mec/diagnostics/settings         Current error_assistant settings.
    POST /mec/diagnostics/settings         Update settings (passthrough to error_assistant.save_settings).
    POST /mec/diagnostics/clear            Clear the in-memory ring buffer.
    GET  /mec/diagnostics/health           Liveness + counters (for sidebar status pill).

Coexists with ComfyUI-Doctor (`/doctor/*`) — separate namespace + IDs.

All responses use the Doctor-style envelope:
    { "success": True, "data": ... }                      OR
    { "success": False, "error": "<KEY>", "message": "<text>" }
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from typing import Any, Deque, Dict

log = logging.getLogger("MEC.diagnostics_api")

# ---------------------------------------------------------------------
# Ring buffer: most-recent-N execution events
# ---------------------------------------------------------------------
_BUFFER_LIMIT = 500
_BUFFER_LOCK = threading.Lock()
_BUFFER: Deque[Dict[str, Any]] = deque(maxlen=_BUFFER_LIMIT)
_PATTERN_HITS: Counter[str] = Counter()
_CATEGORY_HITS: Counter[str] = Counter()
_FIRST_SEEN: Dict[str, float] = {}
_LAST_SEEN: Dict[str, float] = {}


def record_event(event: Dict[str, Any]) -> None:
    """Insight executor wrapper calls this on every node_done / node_error.

    Also called by error_assistant.explain() so we can aggregate Tier-1 hits
    that happen in non-execution code paths (e.g. node validate).
    """
    ev = dict(event or {})
    ev.setdefault("ts", time.time())
    pid = ev.get("pattern_id")
    cat = ev.get("category")
    with _BUFFER_LOCK:
        _BUFFER.append(ev)
        if pid:
            _PATTERN_HITS[pid] += 1
            _LAST_SEEN[pid] = ev["ts"]
            _FIRST_SEEN.setdefault(pid, ev["ts"])
        if cat:
            _CATEGORY_HITS[cat] += 1


def _envelope_ok(data: Any) -> Dict[str, Any]:
    return {"success": True, "data": data}


def _envelope_err(error_key: str, message: str) -> Dict[str, Any]:
    return {"success": False, "error": error_key, "message": message}


# ---------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------
def register_routes(server) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[mec_diagnostics] aiohttp unavailable: %s", e)
        return
    routes = server.routes

    @routes.get("/mec/diagnostics/health")
    async def _health(_req):  # noqa: ARG001
        with _BUFFER_LOCK:
            buf_len = len(_BUFFER)
            errors = sum(1 for e in _BUFFER if e.get("type") == "node_error"
                         or e.get("severity") in ("error", "warn"))
        return web.json_response(_envelope_ok({
            "buffer_len": buf_len,
            "buffer_limit": _BUFFER_LIMIT,
            "error_count": errors,
            "patterns_distinct": len(_PATTERN_HITS),
        }))

    @routes.get("/mec/diagnostics/recent")
    async def _recent(req):
        try:
            limit = max(1, min(_BUFFER_LIMIT, int(req.query.get("limit", "100"))))
        except Exception:
            limit = 100
        kind = req.query.get("kind")  # "error" | "all"
        with _BUFFER_LOCK:
            items = list(_BUFFER)
        if kind == "error":
            items = [e for e in items if e.get("type") == "node_error"
                     or e.get("severity") in ("error", "warn")]
        return web.json_response(_envelope_ok(items[-limit:][::-1]))

    @routes.get("/mec/diagnostics/statistics")
    async def _statistics(_req):  # noqa: ARG001
        with _BUFFER_LOCK:
            hits = _PATTERN_HITS.most_common(50)
            categories = _CATEGORY_HITS.most_common()
            patterns = [
                {
                    "pattern_id": pid,
                    "count": cnt,
                    "first_seen": _FIRST_SEEN.get(pid),
                    "last_seen": _LAST_SEEN.get(pid),
                }
                for pid, cnt in hits
            ]
        return web.json_response(_envelope_ok({
            "patterns": patterns,
            "categories": [{"category": c, "count": n} for c, n in categories],
            "total_events": sum(_PATTERN_HITS.values()),
        }))

    @routes.get("/mec/diagnostics/patterns")
    async def _patterns(_req):  # noqa: ARG001
        try:
            from . import error_assistant as _ea  # type: ignore
        except Exception:
            try:
                from nodes import error_assistant as _ea  # type: ignore
            except Exception as e:
                return web.json_response(_envelope_err(
                    "error_assistant_unavailable", str(e)), status=500)
        try:
            pats = _ea._get_patterns()
            data = [
                {
                    "id": p.name,
                    "category": p.category,
                    "priority": p.priority,
                    "confidence": p.confidence,
                    "source": p.source,
                    "exc_types": list(p.exc_types),
                }
                for p in pats
            ]
            return web.json_response(_envelope_ok({
                "count": len(data),
                "patterns": data,
            }))
        except Exception as e:
            log.exception("[mec_diagnostics] /patterns failed")
            return web.json_response(_envelope_err(
                "patterns_load_failed", f"{type(e).__name__}: {e}"), status=500)

    @routes.post("/mec/diagnostics/reload_patterns")
    async def _reload(_req):  # noqa: ARG001
        try:
            from . import error_assistant as _ea  # type: ignore
        except Exception:
            try:
                from nodes import error_assistant as _ea  # type: ignore
            except Exception as e:
                return web.json_response(_envelope_err(
                    "error_assistant_unavailable", str(e)), status=500)
        try:
            n = _ea.reload_patterns()
            return web.json_response(_envelope_ok({"count": n}))
        except Exception as e:
            log.exception("[mec_diagnostics] /reload_patterns failed")
            return web.json_response(_envelope_err(
                "reload_failed", f"{type(e).__name__}: {e}"), status=500)

    @routes.get("/mec/diagnostics/settings")
    async def _get_settings(_req):  # noqa: ARG001
        try:
            from . import error_assistant as _ea  # type: ignore
        except Exception:
            try:
                from nodes import error_assistant as _ea  # type: ignore
            except Exception as e:
                return web.json_response(_envelope_err(
                    "error_assistant_unavailable", str(e)), status=500)
        try:
            return web.json_response(_envelope_ok(_ea.load_settings()))
        except Exception as e:
            return web.json_response(_envelope_err(
                "settings_read_failed", str(e)), status=500)

    @routes.post("/mec/diagnostics/settings")
    async def _set_settings(req):
        try:
            payload = await req.json()
        except Exception as e:
            return web.json_response(_envelope_err(
                "bad_json", str(e)), status=400)
        try:
            from . import error_assistant as _ea  # type: ignore
        except Exception:
            try:
                from nodes import error_assistant as _ea  # type: ignore
            except Exception as e:
                return web.json_response(_envelope_err(
                    "error_assistant_unavailable", str(e)), status=500)
        try:
            _ea.save_settings(payload or {})
            return web.json_response(_envelope_ok(_ea.load_settings()))
        except Exception as e:
            return web.json_response(_envelope_err(
                "settings_write_failed", str(e)), status=500)

    @routes.post("/mec/diagnostics/clear")
    async def _clear(_req):  # noqa: ARG001
        with _BUFFER_LOCK:
            _BUFFER.clear()
            _PATTERN_HITS.clear()
            _CATEGORY_HITS.clear()
            _FIRST_SEEN.clear()
            _LAST_SEEN.clear()
        return web.json_response(_envelope_ok({"cleared": True}))

    log.info("[mec_diagnostics] /mec/diagnostics/* routes registered")


# ---------------------------------------------------------------------
# Insight bridge: forward node_done / node_error events into our ring.
# ---------------------------------------------------------------------
def install_insight_bridge() -> bool:
    """Wraps `insight._emit` so every event is ALSO recorded for the sidebar.
    Idempotent."""
    try:
        from . import insight as _ins  # type: ignore
    except Exception:
        try:
            from nodes import insight as _ins  # type: ignore
        except Exception as e:
            log.debug("[mec_diagnostics] insight unavailable: %s", e)
            return False
    if getattr(_ins, "_MEC_DIAG_BRIDGED", False):
        return True
    orig_emit = _ins._emit

    def bridged_emit(event):
        try:
            record_event(event)
        except Exception:  # never break telemetry on observer failure
            pass
        try:
            return orig_emit(event)
        except Exception:
            pass

    _ins._emit = bridged_emit
    _ins._MEC_DIAG_BRIDGED = True
    log.info("[mec_diagnostics] insight bridge installed")
    return True
