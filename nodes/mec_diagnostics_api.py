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
import os
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

    # -----------------------------------------------------------------
    # Error Assistant — per-tier readiness + smoke test + secrets I/O
    # -----------------------------------------------------------------
    def _import_ea():
        try:
            from . import error_assistant as _ea  # type: ignore
            return _ea
        except Exception:
            try:
                from nodes import error_assistant as _ea  # type: ignore
                return _ea
            except Exception:
                return None

    def _import_secrets():
        try:
            from . import secrets_store as _ss  # type: ignore
            return _ss
        except Exception:
            try:
                from nodes import secrets_store as _ss  # type: ignore
                return _ss
            except Exception:
                return None

    def _import_local_llm():
        try:
            from . import local_llm as _ll  # type: ignore
            return _ll
        except Exception:
            try:
                from nodes import local_llm as _ll  # type: ignore
                return _ll
            except Exception:
                return None

    @routes.get("/mec/diagnostics/error_assistant/status")
    async def _ea_status(_req):  # noqa: ARG001
        ea = _import_ea()
        ss = _import_secrets()
        ll = _import_local_llm()
        s = ea.load_settings() if ea else {}
        # Tier 1
        try:
            t1_count = len(ea._get_patterns()) if ea else 0
            t1 = {"ready": t1_count > 0, "detail": f"{t1_count} pattern(s) loaded"}
        except Exception as e:
            t1 = {"ready": False, "detail": f"pattern load failed: {e}"}
        # Tier 2: model file resolvable + llama-cpp-python importable.
        t2_detail = []
        t2_ready = True
        if ll is not None:
            mid = s.get("local_model", "")
            try:
                path = ll._resolve_model_path(mid) if hasattr(ll, "_resolve_model_path") else None
            except Exception:
                path = None
            if path:
                t2_detail.append(f"model: {os.path.basename(path)}")
            else:
                t2_ready = False
                t2_detail.append(f"model not found ({mid or 'unset'})")
            try:
                import llama_cpp  # noqa: F401
                t2_detail.append("llama-cpp-python ok")
            except Exception:
                t2_ready = False
                t2_detail.append("llama-cpp-python missing")
        else:
            t2_ready = False
            t2_detail.append("local_llm module unavailable")
        t2 = {"ready": t2_ready, "detail": "; ".join(t2_detail)}
        # Tier 3: API key for selected provider.
        prov = s.get("cloud_provider", "openai")
        if ss is not None:
            has_key = bool(ss.has_key_for(prov))
        else:
            has_key = False
        t3 = {
            "ready": has_key,
            "detail": (f"{prov}: API key set" if has_key
                       else f"{prov}: API key MISSING — set it in Tier 3 below"),
        }
        return web.json_response(_envelope_ok({
            "tier1": t1,
            "tier2": t2,
            "tier3": t3,
            "settings": s,
        }))

    @routes.post("/mec/diagnostics/error_assistant/test_tier")
    async def _ea_test_tier(req):
        try:
            payload = await req.json()
        except Exception as e:
            return web.json_response(_envelope_err("bad_json", str(e)), status=400)
        tier = int(payload.get("tier", 0))
        if tier not in (1, 2, 3):
            return web.json_response(_envelope_err(
                "bad_tier", "tier must be 1, 2, or 3"), status=400)
        ea = _import_ea()
        if ea is None:
            return web.json_response(_envelope_err(
                "error_assistant_unavailable", "module not importable"), status=500)
        # Construct a representative test exception. The pattern matcher will
        # match on the message; LLMs get the prompt regardless.
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        except RuntimeError as test_exc:
            t0 = time.time()
            try:
                if tier == 1:
                    res = ea.explain(test_exc, mode="deterministic_only")
                elif tier == 2:
                    res = ea.explain(test_exc, mode="local_only")
                else:
                    res = ea.explain(test_exc, mode="cloud_only")
                elapsed_ms = int((time.time() - t0) * 1000)
                ok = res.get("tier") == tier
                return web.json_response(_envelope_ok({
                    "tier_requested": tier,
                    "tier_returned": res.get("tier"),
                    "ok": ok,
                    "elapsed_ms": elapsed_ms,
                    "headline": res.get("headline", "")[:200],
                    "preview": (res.get("cause") or "")[:240],
                }))
            except Exception as e:
                return web.json_response(_envelope_err(
                    "tier_test_failed", f"{type(e).__name__}: {e}"), status=500)

    @routes.get("/mec/diagnostics/error_assistant/secrets")
    async def _ea_get_secret(req):
        ss = _import_secrets()
        if ss is None:
            return web.json_response(_envelope_err(
                "secrets_unavailable", "secrets_store not importable"), status=500)
        prov = req.query.get("provider", "openai")
        try:
            key = ss.get_key(prov)
        except Exception:
            key = None
        if not key:
            return web.json_response(_envelope_ok({
                "provider": prov, "set": False, "preview": ""}))
        # Mask: show first 3 + last 4 chars only.
        if len(key) <= 8:
            preview = "*" * len(key)
        else:
            preview = key[:3] + "*" * max(4, len(key) - 7) + key[-4:]
        return web.json_response(_envelope_ok({
            "provider": prov, "set": True, "preview": preview,
        }))

    @routes.post("/mec/diagnostics/error_assistant/secrets")
    async def _ea_set_secret(req):
        try:
            payload = await req.json()
        except Exception as e:
            return web.json_response(_envelope_err("bad_json", str(e)), status=400)
        prov = (payload.get("provider") or "").strip()
        key = payload.get("api_key")
        if not prov:
            return web.json_response(_envelope_err(
                "bad_provider", "provider is required"), status=400)
        ss = _import_secrets()
        if ss is None:
            return web.json_response(_envelope_err(
                "secrets_unavailable", "secrets_store not importable"), status=500)
        try:
            if key is None or key == "":
                ss.delete_key(prov)
                return web.json_response(_envelope_ok({"provider": prov, "set": False}))
            ss.set_key(prov, str(key))
            return web.json_response(_envelope_ok({"provider": prov, "set": True}))
        except Exception as e:
            return web.json_response(_envelope_err(
                "secrets_write_failed", f"{type(e).__name__}: {e}"), status=500)

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
