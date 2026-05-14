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

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("MEC.error_translator")

# ─────────────────────────────────────────────────────────────────────────
# Learning store — appends user/auto-learned patterns to
# patterns/user/learned.json so error_assistant's hot-reload picks them up.
# ─────────────────────────────────────────────────────────────────────────
_LEARN_LOCK = threading.Lock()


def _user_patterns_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    pack_root = os.path.dirname(here)
    user_dir = os.path.join(pack_root, "patterns", "user")
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, "learned.json")


def _slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", (text or "").strip().lower())
    s = s.strip("_")
    return s[:48] or "learned"


def _escape_regex_text(text: str) -> str:
    # Build a forgiving regex from raw error text: escape regex metachars,
    # collapse runs of whitespace and digits/hex/paths so the pattern still
    # matches near-duplicates.
    if not text:
        return r".+"
    first_line = text.strip().splitlines()[0][:240]
    esc = re.escape(first_line)
    esc = re.sub(r"(\\\s)+", r"\\s+", esc)
    # Collapse escaped digit runs → \d+
    esc = re.sub(r"(?:\\d|\d){2,}", r"\\d+", esc)
    # Collapse escaped path-like tokens → \S+
    esc = re.sub(r"(?:\\[/\\\\])\S{2,}", r"\\S+", esc)
    return esc


def _validate_regex(pattern: str) -> Optional[str]:
    try:
        re.compile(pattern, re.IGNORECASE | re.DOTALL)
        return None
    except re.error as e:
        return f"invalid regex: {e}"


def _load_learned() -> Dict[str, Any]:
    path = _user_patterns_path()
    if not os.path.isfile(path):
        return {
            "_schema": "MEC.error_patterns/1.0",
            "_doc": "User-taught and auto-learned patterns. Written by /mec/teach_error.",
            "patterns": [],
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "patterns" not in data:
            raise ValueError("malformed learned.json: missing 'patterns'")
        if not isinstance(data["patterns"], list):
            raise ValueError("malformed learned.json: 'patterns' is not a list")
        return data
    except Exception as e:
        log.warning("[error_translator] could not load %s (%s); starting fresh", path, e)
        return {
            "_schema": "MEC.error_patterns/1.0",
            "_doc": "User-taught and auto-learned patterns. Written by /mec/teach_error.",
            "patterns": [],
        }


def _save_learned(data: Dict[str, Any]) -> None:
    path = _user_patterns_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def teach_pattern(
    *,
    message: str,
    cause: str,
    fixes: Optional[List[str]] = None,
    regex: Optional[str] = None,
    pattern_id: Optional[str] = None,
    category: str = "user",
    exc_types: Optional[List[str]] = None,
    priority: int = 200,
    confidence: float = 0.7,
    source_tier: Optional[int] = None,
) -> Dict[str, Any]:
    """Persist a new pattern to patterns/user/learned.json.

    If regex is omitted, derive a forgiving one from `message`. Returns the
    final pattern dict that was written.
    """
    message = (message or "").strip()
    cause = (cause or "").strip()
    if not message and not regex:
        raise ValueError("either 'message' or 'regex' is required")
    if not cause:
        raise ValueError("'cause' is required")

    final_regex = regex.strip() if regex else _escape_regex_text(message)
    err = _validate_regex(final_regex)
    if err:
        raise ValueError(err)

    fixes_list = [str(x).strip() for x in (fixes or []) if str(x).strip()]
    pid = (pattern_id and _slugify(pattern_id)) or _slugify(message or cause)
    if source_tier:
        pid = f"{pid}_t{int(source_tier)}"

    entry = {
        "id": pid,
        "category": category or "user",
        "priority": int(priority),
        "confidence": float(confidence),
        "exc_types": list(exc_types or []),
        "regex": final_regex,
        "cause": cause,
        "fixes": fixes_list,
        "_meta": {
            "learned_at": int(time.time()),
            "source_tier": source_tier,
            "raw_message": message[:600],
        },
    }

    with _LEARN_LOCK:
        data = _load_learned()
        # If an entry with the same id exists, overwrite (latest teaching wins).
        existing = data["patterns"]
        idx = next((i for i, p in enumerate(existing) if p.get("id") == pid), -1)
        if idx >= 0:
            existing[idx] = entry
        else:
            existing.append(entry)
        _save_learned(data)

    # Trigger hot-reload in error_assistant so the new pattern matches
    # immediately on the next error.
    try:
        from . import error_assistant  # type: ignore
        error_assistant.reload_patterns()
    except Exception as e:
        log.warning("[error_translator] reload_patterns failed: %s", e)

    return entry

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

    @routes.post("/mec/teach_error")
    async def _teach(req: web.Request) -> web.Response:
        try:
            body = await req.json()
        except Exception:
            return web.json_response(
                {"success": False, "error": "bad_request", "message": "Body must be JSON."},
                status=400,
            )

        message = str(body.get("message") or "")
        cause   = str(body.get("cause") or "")
        if not cause:
            return web.json_response(
                {"success": False, "error": "missing_cause",
                 "message": "Field 'cause' is required."},
                status=400,
            )

        try:
            entry = teach_pattern(
                message    = message,
                cause      = cause,
                fixes      = body.get("fixes") or [],
                regex      = body.get("regex"),
                pattern_id = body.get("id"),
                category   = str(body.get("category") or "user"),
                exc_types  = body.get("exc_types") or [],
                priority   = int(body.get("priority", 200)),
                confidence = float(body.get("confidence", 0.7)),
                source_tier= body.get("source_tier"),
            )
        except ValueError as e:
            return web.json_response(
                {"success": False, "error": "invalid", "message": str(e)},
                status=400,
            )
        except Exception as e:
            log.exception("[error_translator] teach failed: %s", e)
            return web.json_response(
                {"success": False, "error": "teach_failed", "message": str(e)},
                status=500,
            )

        return web.json_response({
            "success": True,
            "data": {
                "id": entry["id"],
                "regex": entry["regex"],
                "category": entry["category"],
                "path": _user_patterns_path(),
            },
        })

    @routes.get("/mec/learned_patterns")
    async def _list_learned(req: web.Request) -> web.Response:
        try:
            data = _load_learned()
        except Exception as e:
            return web.json_response(
                {"success": False, "error": "load_failed", "message": str(e)},
                status=500,
            )
        # Strip raw_message bodies to keep payload light.
        items = []
        for p in data.get("patterns", []):
            meta = dict(p.get("_meta") or {})
            meta.pop("raw_message", None)
            items.append({
                "id": p.get("id"),
                "category": p.get("category"),
                "regex": p.get("regex"),
                "cause": p.get("cause"),
                "fixes": p.get("fixes") or [],
                "priority": p.get("priority"),
                "confidence": p.get("confidence"),
                "_meta": meta,
            })
        return web.json_response({"success": True, "data": {"count": len(items), "patterns": items}})

    log.info("[error_translator] Routes registered: "
             "POST /mec/translate_error, POST /mec/teach_error, GET /mec/learned_patterns")
