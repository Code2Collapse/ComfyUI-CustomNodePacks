"""
error_translator.py — REST route for the JS toast translator (Phase 2).

The frontend `js/mec_error_toast.js` intercepts red error toasts and POSTs the
raw text here.  We reuse `error_assistant.explain()` which already implements
the 3-tier routing (pattern → local LLM → cloud LLM), but adapt it to operate
on a plain text message instead of a live exception object.

Route
-----
POST /mec/translate_error
    Body: {"message": "...", "node_class": "...", "traceback_tail": "..." (all optional)}
    Returns the same envelope shape as the existing error_assistant explain():
      {success: true, data: {tier, headline, cause, fixes, ...}}
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

log = logging.getLogger("MEC.error_translator")

# Try to pull a "TypeName: rest of message" out of the toast text
_EXC_HEAD_RE = re.compile(r"^([A-Z][A-Za-z0-9_.]*Error|[A-Z][A-Za-z0-9_.]*Exception)\s*:\s*(.*)$")


class _SyntheticExc(Exception):
    """A real Exception subclass so error_assistant.explain() works on text."""
    pass


def _make_synthetic_exc(message: str) -> BaseException:
    """Build a real Exception whose type name + message match the toast text."""
    text = (message or "").strip()
    # Try first line first — that's almost always the type and message
    first_line = text.splitlines()[0] if text else ""
    m = _EXC_HEAD_RE.match(first_line)
    if m:
        type_name, body = m.group(1), m.group(2).strip()
        # Dynamically create a class with the original name so error_assistant
        # pattern matching sees the right `type(exc).__name__`.
        cls = type(type_name, (Exception,), {})
        return cls(body or text)
    return _SyntheticExc(text or "Unknown error")


def translate_message(message: str,
                      node_class: Optional[str] = None,
                      traceback_tail: Optional[str] = None,
                      mode: Optional[str] = None) -> Dict[str, Any]:
    """Convert plain toast text → structured explanation via error_assistant."""
    try:
        from . import error_assistant  # type: ignore
    except Exception as e:
        log.warning("[error_translator] error_assistant unavailable: %s", e)
        return {
            "tier": 0,
            "headline": (message or "Unknown error").strip().splitlines()[0][:160],
            "cause": "Error explainer is unavailable.",
            "fixes": ["Check ComfyUI console for the full traceback."],
            "pattern_id": "no_explainer",
            "category": "uncategorized",
            "confidence": 0.0,
        }

    exc = _make_synthetic_exc(message or "")
    return error_assistant.explain(
        exc,
        node_class=node_class,
        traceback_tail=traceback_tail,
        mode=mode,
    )


# ─────────────────────────────────────────────────────────────────────────
# Route registration
# ─────────────────────────────────────────────────────────────────────────
def register_routes(server: Any) -> None:
    try:
        from aiohttp import web
    except Exception as e:
        log.warning("[error_translator] aiohttp unavailable: %s", e)
        return

    routes = server.routes

    @routes.post("/mec/translate_error")
    async def _translate(req: web.Request) -> web.Response:
        try:
            body = await req.json()
        except Exception:
            return web.json_response(
                {"success": False, "error": "bad_request", "message": "Body must be JSON."},
                status=400,
            )

        message = str(body.get("message") or "").strip()
        if not message:
            return web.json_response(
                {"success": False, "error": "empty_message",
                 "message": "Field 'message' is required."},
                status=400,
            )

        node_class     = body.get("node_class")
        traceback_tail = body.get("traceback_tail")
        mode           = body.get("mode")

        try:
            result = translate_message(message, node_class=node_class,
                                       traceback_tail=traceback_tail, mode=mode)
        except Exception as e:
            log.exception("[error_translator] translate failed: %s", e)
            return web.json_response(
                {"success": False, "error": "translate_failed", "message": str(e)},
                status=500,
            )

        return web.json_response({"success": True, "data": result})

    log.info("[error_translator] Route registered: POST /mec/translate_error")
