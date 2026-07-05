"""
error_translator.py — REST route for the JS toast translator (Phase 2).

The frontend `js/c2c_error_toast.js` intercepts red error toasts and POSTs the
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
                      mode: Optional[str] = None,
                      locale: Optional[str] = None,
                      workflow: Optional[Dict[str, Any]] = None,
                      node_id: Optional[Any] = None) -> Dict[str, Any]:
    """Convert plain toast text → structured explanation via error_assistant.

    If ``locale`` is provided and a matching overlay exists at
    ``patterns/i18n/<locale>.json``, the ``headline`` / ``cause`` / ``fixes``
    fields of the returned dict are replaced with their localized variants
    (keyed by ``i18n_key`` or ``pattern_id``). Unknown patterns pass through
    unchanged so the user still gets the English (or original) text.

    When ``node_class`` is supplied and the matched pattern is a None-input
    family (bbox / mask / latent / conditioning), the ``cause`` is rewritten
    to name the specific unconnected slot on that node. If ``workflow`` and
    ``node_id`` are also supplied we narrow to the slot that is actually
    unconnected in the live graph; otherwise we list all candidate slots of
    the expected type.
    """
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
    result = error_assistant.explain(
        exc,
        node_class=node_class,
        traceback_tail=traceback_tail,
        mode=mode,
    )
    # Node-specific phrasing: refine the cause/headline for Tier-1 matches
    # that we can localise to a specific unconnected input.
    if result.get("tier") == 1 and node_class:
        try:
            _apply_node_specific_phrasing(
                result,
                node_class=str(node_class),
                workflow=workflow,
                node_id=node_id,
            )
        except Exception as e:
            log.debug("[error_translator] node-specific phrasing failed: %s", e)
    if locale:
        _apply_locale_overlay(result, locale)
    return result


# ─────────────────────────────────────────────────────────────────────────
# Node-specific phrasing
# ─────────────────────────────────────────────────────────────────────────
# Map a pattern_id of the "None on attribute X" family → the ComfyUI
# slot type(s) we should look for in the failing node's INPUT_TYPES.
_NONE_PATTERN_TO_SLOT_TYPES: Dict[str, tuple] = {
    "none_bbox_attr":         ("BBOX",),
    "none_mask_attr":         ("MASK",),
    "none_latent_samples":    ("LATENT",),
    "none_conditioning_attr": ("CONDITIONING",),
}

# Friendly upstream-source phrasing per slot type. Listed in priority order.
_SLOT_TYPE_HINTS: Dict[str, str] = {
    "BBOX":         "a bounding-box detector (YOLO, FaceDetect, SAM, or any node with a BBOX output)",
    "MASK":         "a mask source (SAM, Solid Mask, Load Image (mask), or a segmenter)",
    "LATENT":       "Empty Latent Image (for txt2img) or VAEEncode (for img2img)",
    "CONDITIONING": "CLIPTextEncode (a positive and a negative prompt)",
}


def _lookup_node_class(node_class: str):
    """Return the ComfyUI node class object for `node_class`, or None.

    Tries ComfyUI's global registry first (`nodes.NODE_CLASS_MAPPINGS`),
    then falls back to our own pack registration. Robust to ComfyUI not
    being importable in offline unit tests.
    """
    if not node_class:
        return None
    # Comfy's top-level `nodes` module exposes NODE_CLASS_MAPPINGS once the
    # server has booted. Import is deferred so this module stays unit-testable.
    try:
        import nodes as _comfy_nodes  # type: ignore
        mapping = getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", None)
        if isinstance(mapping, dict) and node_class in mapping:
            return mapping[node_class]
    except Exception:
        pass
    return None


def _required_inputs_of_type(cls: Any, types_wanted: tuple) -> List[str]:
    """List required input-slot names on `cls` whose declared type is in
    `types_wanted`. Returns [] on any error (defensive: custom nodes have
    historically returned malformed INPUT_TYPES dicts).
    """
    try:
        get_inputs = getattr(cls, "INPUT_TYPES", None)
        if not callable(get_inputs):
            return []
        spec = get_inputs() or {}
        required = spec.get("required") or {}
        if not isinstance(required, dict):
            return []
        out: List[str] = []
        for slot_name, slot_spec in required.items():
            if not isinstance(slot_spec, (list, tuple)) or not slot_spec:
                continue
            t = slot_spec[0]
            if isinstance(t, str) and t in types_wanted:
                out.append(str(slot_name))
        return out
    except Exception:
        return []


def _connected_input_slots(workflow: Dict[str, Any], node_id: Any) -> List[str]:
    """Return the names of input slots on `node_id` that have an incoming
    link in `workflow`. Accepts both the "litegraph" workflow shape
    (top-level `nodes` + `links`) and the "API/prompt" shape
    (top-level dict {node_id: {inputs: {slot: [src_id, src_slot]}}}).
    Returns [] on any error.
    """
    if not isinstance(workflow, dict) or node_id is None:
        return []
    try:
        nid_str = str(node_id)
        # API/prompt shape: {"3": {"inputs": {"bbox": ["5", 0]}, "class_type": "..."}}
        node_api = workflow.get(nid_str)
        if isinstance(node_api, dict) and "inputs" in node_api:
            inputs = node_api.get("inputs") or {}
            return [
                str(k) for k, v in inputs.items()
                # In the API shape, a connected input is a 2-element list/tuple
                # [src_id, src_slot_index]; literal widget values are scalars
                # or dicts and should be ignored here.
                if isinstance(v, (list, tuple)) and len(v) == 2
            ]
        # Litegraph shape: workflow["nodes"] is a list of node dicts each with
        # "id", "inputs" (list of {name, link}). Links are resolved via the
        # top-level "links" array, but for our purpose just checking `link`
        # is non-null on the input slot is enough.
        nodes_list = workflow.get("nodes")
        if isinstance(nodes_list, list):
            for n in nodes_list:
                if not isinstance(n, dict):
                    continue
                if str(n.get("id")) != nid_str:
                    continue
                inputs = n.get("inputs") or []
                if not isinstance(inputs, list):
                    return []
                return [
                    str(i.get("name")) for i in inputs
                    if isinstance(i, dict) and i.get("link") is not None
                ]
    except Exception:
        return []
    return []


def _format_slot_list(names: List[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return f"**{names[0]}**"
    if len(names) == 2:
        return f"**{names[0]}** and **{names[1]}**"
    return ", ".join(f"**{n}**" for n in names[:-1]) + f", and **{names[-1]}**"


def _apply_node_specific_phrasing(
    result: Dict[str, Any],
    *,
    node_class: str,
    workflow: Optional[Dict[str, Any]] = None,
    node_id: Optional[Any] = None,
) -> None:
    """Mutate `result` in place: rewrite headline/cause to name the specific
    unconnected input on `node_class`. No-op if we can't be more specific.

    Records the original cause as `cause_original` and sets
    `node_specific: true` so JS/diagnostic UIs can show a badge.
    """
    pattern_id = result.get("pattern_id")
    slot_types = _NONE_PATTERN_TO_SLOT_TYPES.get(pattern_id or "")
    if not slot_types:
        return

    cls = _lookup_node_class(node_class)
    if cls is None:
        return
    candidates = _required_inputs_of_type(cls, slot_types)
    if not candidates:
        return

    # If we have a live graph, narrow to the slots that are NOT connected.
    unconnected = list(candidates)
    if workflow and node_id is not None:
        connected = set(_connected_input_slots(workflow, node_id))
        if connected:
            narrowed = [s for s in candidates if s not in connected]
            # Only narrow if we still have at least one candidate.
            if narrowed:
                unconnected = narrowed

    slot_label = _format_slot_list(unconnected)
    slot_type  = slot_types[0]
    hint       = _SLOT_TYPE_HINTS.get(slot_type, f"an upstream {slot_type} node")
    is_plural  = len(unconnected) > 1

    new_cause = (
        f"The {slot_label} input{'s' if is_plural else ''} on **{node_class}** "
        f"{'are' if is_plural else 'is'} not connected. "
        f"{'They' if is_plural else 'It'} {'expect' if is_plural else 'expects'} "
        f"a {slot_type} cable from {hint}, "
        f"but currently {'have' if is_plural else 'has'} no link."
    )
    new_headline = (
        f"{node_class}: the {slot_label} input "
        f"{'are' if is_plural else 'is'} not connected."
    )

    result["cause_original"]    = result.get("cause")
    result["headline_original"] = result.get("headline")
    result["cause"]             = new_cause
    result["headline"]          = new_headline
    result["node_class"]        = node_class
    result["node_specific"]     = True
    if node_id is not None:
        result["node_id"] = str(node_id)
    result["unconnected_slots"] = unconnected


# ─────────────────────────────────────────────────────────────────────────
# i18n overlays — patterns/i18n/<locale>.json
# Schema:
#   {"_locale": "hi",
#    "patterns": {"<i18n_key or pattern_id>":
#       {"headline": "...", "cause": "...", "fixes": ["...", "..."]}}}
# Only the keys present in the overlay are replaced; the rest of the
# explanation envelope (tier, pattern_id, confidence, etc.) is preserved.
# ─────────────────────────────────────────────────────────────────────────
_I18N_CACHE: Dict[str, Dict[str, Any]] = {}


def _i18n_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    pack_root = os.path.dirname(here)
    return os.path.join(pack_root, "patterns", "i18n")


def _load_locale(locale: str) -> Dict[str, Any]:
    locale = (locale or "").strip().lower()
    if not locale or locale == "en":
        return {}
    if locale in _I18N_CACHE:
        return _I18N_CACHE[locale]
    path = os.path.join(_i18n_dir(), f"{locale}.json")
    if not os.path.isfile(path):
        # Try the base language (e.g. "pt-br" → "pt")
        base = locale.split("-")[0]
        if base != locale:
            return _load_locale(base)
        _I18N_CACHE[locale] = {}
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("overlay must be a JSON object")
        patterns = data.get("patterns") or {}
        if not isinstance(patterns, dict):
            raise ValueError("'patterns' must be an object")
        _I18N_CACHE[locale] = patterns
        return patterns
    except Exception as e:
        log.warning("[error_translator] failed to load i18n %s: %s", path, e)
        _I18N_CACHE[locale] = {}
        return {}


def _apply_locale_overlay(result: Dict[str, Any], locale: str) -> None:
    overlay_all = _load_locale(locale)
    if not overlay_all:
        return
    # Prefer i18n_key (explicit), fall back to pattern_id
    key = result.get("i18n_key") or result.get("pattern_id")
    if not key:
        return
    entry = overlay_all.get(key)
    if not isinstance(entry, dict):
        return
    for field in ("headline", "cause"):
        v = entry.get(field)
        if isinstance(v, str) and v.strip():
            result[field] = v
    fixes = entry.get("fixes")
    if isinstance(fixes, list) and fixes:
        result["fixes"] = [str(f) for f in fixes if str(f).strip()]
    result["locale"] = locale


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

    # ── W5a: local AI model management (Tier-2 Qwen3.5-4B) ──────────────
    @routes.get("/c2c/ai/local_model/status")
    async def _local_model_status(_req: web.Request) -> web.Response:
        try:
            from .local_llm import get_status
            return web.json_response(get_status())
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @routes.post("/c2c/ai/local_model/download")
    async def _local_model_download(_req: web.Request) -> web.Response:
        try:
            from .local_llm import start_download
            return web.json_response(start_download())
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

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
        locale         = body.get("locale")
        workflow       = body.get("workflow")  # optional live graph for slot-narrowing
        node_id        = body.get("node_id")   # optional id of the failing node
        if not locale:
            # Fall back to browser's Accept-Language: take primary language tag
            al = req.headers.get("Accept-Language", "")
            if al:
                locale = al.split(",")[0].strip().split(";")[0].strip()

        try:
            result = translate_message(message, node_class=node_class,
                                       traceback_tail=traceback_tail, mode=mode,
                                       locale=locale, workflow=workflow,
                                       node_id=node_id)
        except Exception as e:
            log.exception("[error_translator] translate failed: %s", e)
            return web.json_response(
                {"success": False, "error": "translate_failed", "message": str(e)},
                status=500,
            )

        # Mirror every translated error into the diagnostics ring buffer so
        # the Insight / Integrity sidebar populates even when the insight
        # executor hook fails to install on a given ComfyUI build. Without
        # this the sidebar reports "no events" while the toast shows them.
        try:
            from . import mec_diagnostics_api as _diag
            sev_map = {"hint": "info", "warn": "warning",
                       "error": "error", "critical": "error"}
            _diag.record_event({
                "type": "node_error",
                "severity": sev_map.get(str(result.get("severity") or "error"),
                                        "error"),
                "node_type": node_class,
                "node_id": node_id,
                "exc_type": (result.get("category") or "").upper() or None,
                "exc_msg": (message or "")[:500],
                "pattern_id": result.get("pattern_id"),
                "category": result.get("category"),
                "summary": result.get("summary"),
            })
        except Exception:
            # Never let diagnostics break the translate response.
            pass

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
