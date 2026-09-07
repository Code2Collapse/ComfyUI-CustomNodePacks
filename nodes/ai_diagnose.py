"""ai_diagnose.py — graph-aware error diagnosis with an applicable fix.

Section-7(3): when a run fails, don't re-paste the traceback — read the
failing node's actual configuration, cross-reference the error, name the
parameter at fault in plain English, and (when confident) return a concrete
widget change the UI can apply after one confirmation.

POST /c2c/ai/diagnose
{
  "exc_type": "RuntimeError", "message": "...", "traceback": "...",
  "node_id": "12", "node_type": "KSampler",
  "widgets": {"steps": 150, "cfg": 30, ...},          # live values from the graph
  "upstream": [{"id": "3", "type": "CheckpointLoaderSimple"}, ...]
}
→ { ok, cause, fix, apply: {widget, value}|null, provider }

Chain: Tier-1 deterministic pattern match (instant, no LLM) → Tier-2/3 LLM
with the node's compact schema + live widget values, asked for STRICT JSON →
mechanical validation of any suggested apply (the widget must exist on the
node type and combo values must be legal) → humanise-rules fallback so the
route never returns nothing.

Author: Code2Collapse. Apache-2.0.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

log = logging.getLogger("MEC.ai_diagnose")

_DIAG_SYSTEM = """You debug ComfyUI workflows. Given a node's configuration and the
error it raised, reply with ONE JSON object only:
{"cause":"one plain-English sentence a non-engineer understands",
 "fix":"one sentence telling them what to change",
 "apply":{"widget":"widget_name","value":<new value>} or null}
Only set "apply" when a SINGLE widget change on THIS node is clearly the fix;
otherwise null. No prose outside the JSON, no markdown fences."""

# Deterministic fallbacks (mirrors the frontend humanise rules).
_RULES = [
    (re.compile(r"CUDA out of memory|OutOfMemoryError", re.I),
     "The GPU ran out of memory for this step.",
     "Lower the resolution, batch size, or steps — or enable an offload/lowvram option."),
    (re.compile(r"NoneType.*attribute", re.I),
     "A required input of this node is not connected.",
     "Connect the missing input (check the node's left-side sockets)."),
    (re.compile(r"FileNotFoundError|No such file", re.I),
     "A file or model this node points at does not exist.",
     "Re-pick the file/model in the node's dropdown."),
    (re.compile(r"size|shape.*match|dimension", re.I),
     "Two connected images/latents have different sizes.",
     "Resize/reformat one branch so both sides match."),
    (re.compile(r"ModuleNotFoundError|No module named", re.I),
     "A Python package this node needs is not installed.",
     "Install the package named in the error into the ComfyUI environment."),
]


def _fallback(exc_type: str, message: str) -> Dict[str, Any]:
    for rx, cause, fix in _RULES:
        if rx.search(message or "") or rx.search(exc_type or ""):
            return {"cause": cause, "fix": fix, "apply": None}
    first = next((l for l in (message or "").splitlines() if l.strip()), "")[:160]
    return {"cause": f"The node failed with: {first or exc_type}.",
            "fix": "Check the node's inputs and settings; the console has the full detail.",
            "apply": None}


def _validate_apply(node_type: str, apply: Any) -> Optional[Dict[str, Any]]:
    """Only pass through a suggestion that is mechanically possible."""
    if not isinstance(apply, dict):
        return None
    widget = apply.get("widget")
    value = apply.get("value")
    try:
        from .ai_workflow_builder import _widget_entries
    except Exception:
        import importlib
        _widget_entries = importlib.import_module(
            "nodes.ai_workflow_builder", package=__package__)._widget_entries
    for name, wtype, _dflt in _widget_entries(node_type):
        if name != widget:
            continue
        if isinstance(wtype, list):                    # combo: value must be legal
            return {"widget": widget, "value": value} if value in wtype else None
        if wtype == "INT":
            try:
                return {"widget": widget, "value": int(value)}
            except Exception:
                return None
        if wtype == "FLOAT":
            try:
                return {"widget": widget, "value": float(value)}
            except Exception:
                return None
        if wtype == "BOOLEAN":
            return {"widget": widget, "value": bool(value)}
        return {"widget": widget, "value": str(value)}
    return None


def diagnose(payload: Dict[str, Any]) -> Dict[str, Any]:
    exc_type = str(payload.get("exc_type") or "")
    message = str(payload.get("message") or "")
    node_type = str(payload.get("node_type") or "")
    widgets = payload.get("widgets") or {}
    upstream = payload.get("upstream") or []
    tb_tail = "\n".join(str(payload.get("traceback") or "").splitlines()[-8:])

    # Tier 1: the existing curated pattern library (instant).
    try:
        from . import error_assistant as ea
        pat = ea.match_pattern(exc_type, message)
        if pat is not None:
            return {"ok": True, "provider": "pattern",
                    "cause": getattr(pat, "cause", None) or getattr(pat, "explanation", str(pat))[:200],
                    "fix": getattr(pat, "fix", None) or getattr(pat, "suggestion", "") or
                           "See the suggested action in the error panel.",
                    "apply": None}
    except Exception:
        pass

    # Tier 2/3: LLM with the node's real schema + live values.
    schema = ""
    try:
        from .ai_workflow_builder import compact_schema, _complete, _extract_json, _registry
        if node_type and node_type in _registry():
            schema = compact_schema(node_type)
    except Exception as exc:
        log.info("[ai_diagnose] builder plumbing unavailable: %s", exc)
        return {"ok": True, "provider": "rules", **_fallback(exc_type, message)}

    prompt = (
        f"Failing node: {node_type} (id {payload.get('node_id')})\n"
        f"Node schema:\n{schema or '(unknown type)'}\n"
        f"Current widget values: {json.dumps(widgets)[:800]}\n"
        f"Upstream nodes: {json.dumps([u.get('type') for u in upstream])[:300]}\n"
        f"Error: {exc_type}: {message[:500]}\n"
        f"Traceback tail:\n{tb_tail[:600]}\n\nJSON:"
    )
    text, provider = _complete(prompt, system=_DIAG_SYSTEM)
    if text:
        # accept the last JSON object that has a "cause" key
        obj = None
        for m in re.finditer(r"\{", text):
            try:
                cand, _ = json.JSONDecoder().raw_decode(text[m.start():])
                if isinstance(cand, dict) and "cause" in cand:
                    obj = cand
            except Exception:
                continue
        if obj:
            return {"ok": True, "provider": provider,
                    "cause": str(obj.get("cause") or "")[:300],
                    "fix": str(obj.get("fix") or "")[:300],
                    "apply": _validate_apply(node_type, obj.get("apply"))}
    return {"ok": True, "provider": "rules", **_fallback(exc_type, message)}


def register_routes(server: Any) -> None:
    from aiohttp import web

    @server.routes.post("/c2c/ai/diagnose")
    async def _diag(request):  # noqa: ANN001
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, diagnose, body)
            return web.json_response(result)
        except Exception as exc:  # noqa: BLE001
            log.warning("[ai_diagnose] failed: %s", exc)
            return web.json_response(
                {"ok": True, "provider": "rules",
                 **_fallback(str(body.get("exc_type")), str(body.get("message")))})

    log.info("[MEC] ai_diagnose route registered: POST /c2c/ai/diagnose")
