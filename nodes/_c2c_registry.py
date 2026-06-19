"""
_c2c_registry.py — central record of every optional component the pack
tried to load, why it failed, and how a user could fix it.

This exists because of `ideas_summary.md` §2.1: the biggest reason the
pack *feels* like stubs is that `except Exception: pass` in `__init__.py`
files silently drops nodes/sub-backends — from the canvas it looks
identical to "the node was never written."

The fix is NOT to remove the try/except (one missing optional dep would
still take down the entire pack), but to record every failure with full
detail and surface it to the user.

Public API
----------
    try_import("MaskTemporalMEC",
               lambda: __import__("...temporal_node", ...),
               hint="Install opencv-python to enable mask temporal smoothing")
            → returns the imported module, or None on failure (after logging).

    record_failure(key, exc, hint=None, group="root")
            → manually note a failure (used inside hand-written try blocks).

    record_status(key, status, detail=None, group="backend")
            → note a non-error degraded state (e.g. backend has no weights).
            status ∈ {"ready", "experimental", "missing-deps", "missing-weights"}

    summary() → dict suitable for JSON serialisation, used by the
                /c2c/registry/status HTTP route and the boot toast.

    register_routes(server) → wires the HTTP route. Idempotent.

Output channels (every failure goes to ALL of them so users can't miss):
    1. Python `logging` at WARNING (filterable).
    2. `print()` to ComfyUI's console (the place users actually look).
    3. The in-memory `_FAILURES` list (queryable from the UI).
    4. `c2c:registry-update` PromptServer event (for live UI).

Apache-2.0
"""
from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

_log = logging.getLogger("C2C.Registry")
_lock = threading.Lock()


@dataclass
class FailureRecord:
    key: str                              # display name of the component
    group: str = "root"                   # logical bucket (root / mask_matting / segmenters / matters / ai)
    exception_type: str = ""
    message: str = ""
    hint: Optional[str] = None            # human-readable next step
    traceback: str = ""
    severity: str = "warning"             # "warning" | "error" | "info"


@dataclass
class StatusRecord:
    key: str
    group: str = "backend"
    status: str = "ready"                 # ready | experimental | missing-deps | missing-weights | disabled
    detail: Optional[str] = None
    hint: Optional[str] = None


@dataclass
class _State:
    failures: List[FailureRecord] = field(default_factory=list)
    statuses: Dict[str, StatusRecord] = field(default_factory=dict)


_STATE = _State()


# ── Helpers ────────────────────────────────────────────────────────────────
def _emit_event(payload: Dict[str, Any]) -> None:
    """Push a registry update over the PromptServer socket so the UI can
    refresh without polling. Soft-fails when the server isn't ready yet."""
    try:
        import server as _comfy_server  # type: ignore
        ps = _comfy_server.PromptServer.instance
        ps.send_sync("c2c.registry", payload)
    except Exception:
        pass


def _log_and_print(level: str, msg: str) -> None:
    """Belt-and-braces: log at the right level AND print to stdout where
    Comfy users actually look."""
    if level == "error":
        _log.error(msg)
    elif level == "info":
        _log.info(msg)
    else:
        _log.warning(msg)
    print(f"[C2C registry] {msg}", flush=True)


# ── Public API ─────────────────────────────────────────────────────────────
def record_failure(
    key: str,
    exc: BaseException,
    *,
    hint: Optional[str] = None,
    group: str = "root",
    severity: str = "warning",
) -> FailureRecord:
    """Record an exception from an optional import block.

    The exception object is inspected only — we do NOT re-raise. Returns
    the FailureRecord so callers can include it in their own logging if
    they wish.
    """
    rec = FailureRecord(
        key=str(key),
        group=str(group),
        exception_type=type(exc).__name__,
        message=str(exc),
        hint=hint,
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
        severity=severity,
    )
    with _lock:
        _STATE.failures.append(rec)
    line = f"{rec.group}/{rec.key} unavailable: {rec.exception_type}: {rec.message}"
    if hint:
        line += f"  →  {hint}"
    _log_and_print(severity if severity in ("warning", "error", "info") else "warning", line)
    _emit_event({"kind": "failure", "record": asdict(rec)})
    return rec


def record_status(
    key: str,
    status: str,
    *,
    detail: Optional[str] = None,
    hint: Optional[str] = None,
    group: str = "backend",
) -> StatusRecord:
    """Record a non-error degraded state (backend present but weights missing,
    experimental, disabled, etc.)."""
    rec = StatusRecord(key=str(key), group=str(group), status=str(status),
                       detail=detail, hint=hint)
    with _lock:
        _STATE.statuses[f"{group}/{key}"] = rec
    if status not in ("ready",):
        line = f"{group}/{key} status={status}"
        if detail:
            line += f" ({detail})"
        if hint:
            line += f"  →  {hint}"
        _log_and_print("info", line)
    _emit_event({"kind": "status", "record": asdict(rec)})
    return rec


def try_import(
    key: str,
    loader: Callable[[], Any],
    *,
    hint: Optional[str] = None,
    group: str = "root",
) -> Any:
    """Run ``loader`` (a zero-arg callable that performs the import) and
    return its result, or None on failure. The failure is recorded with
    full traceback + hint.

    Typical use::

        mod = try_import(
            "WanDirector",
            lambda: __import__("nodes.wan_director", fromlist=["NODE_CLASS_MAPPINGS"]),
            hint="Install `av` and `kornia` for full WanDirector functionality.",
            group="root",
        )
    """
    try:
        return loader()
    except BaseException as exc:  # noqa: BLE001 — yes, even SystemExit gets recorded
        record_failure(key, exc, hint=hint, group=group)
        return None


def summary() -> Dict[str, Any]:
    """Snapshot suitable for JSON serialisation."""
    with _lock:
        return {
            "failures": [asdict(r) for r in _STATE.failures],
            "statuses": [asdict(r) for r in _STATE.statuses.values()],
            "counts": {
                "failures": len(_STATE.failures),
                "ready":         sum(1 for r in _STATE.statuses.values() if r.status == "ready"),
                "experimental":  sum(1 for r in _STATE.statuses.values() if r.status == "experimental"),
                "missing_deps":  sum(1 for r in _STATE.statuses.values() if r.status == "missing-deps"),
                "missing_weights": sum(1 for r in _STATE.statuses.values() if r.status == "missing-weights"),
            },
        }


def clear() -> None:
    """Wipe state. Used by unit tests only."""
    with _lock:
        _STATE.failures.clear()
        _STATE.statuses.clear()


def record_client_failure(payload: Dict[str, Any]) -> FailureRecord:
    """Ingest a failure payload from the JS side (``_c2c_undo.js`` and
    other browser-side error surfaces post here). The payload shape is the
    same one ``_surfaceFailure`` builds:

        { scope, where, message, stack, context, ts }

    We translate it into a :class:`FailureRecord` (group=``"client.js"``)
    so it appears in the same ``/c2c/registry/status`` summary the Doctor
    panel and INT badge consume. Mojibake-safe: values are coerced to str
    and truncated.
    """
    def _s(v: Any, cap: int) -> str:
        if v is None:
            return ""
        try:
            t = str(v)
        except Exception:
            t = repr(v)
        return t[:cap]

    scope = _s(payload.get("scope"), 80) or "c2c.client"
    where = _s(payload.get("where"), 200) or "unknown"
    message = _s(payload.get("message"), 1000) or "(no message)"
    stack = _s(payload.get("stack"), 4000)
    ctx = payload.get("context")
    try:
        import json as _json
        ctx_str = _json.dumps(ctx, default=str)[:2000] if ctx is not None else ""
    except Exception:
        ctx_str = _s(ctx, 2000)

    rec = FailureRecord(
        key=f"{scope}:{where}",
        group="client.js",
        exception_type=scope,
        message=message,
        hint=ctx_str or None,
        traceback=stack,
        severity="warning",
    )
    with _lock:
        _STATE.failures.append(rec)
        # bound the ring buffer so a flaky frontend can't OOM us
        if len(_STATE.failures) > 500:
            del _STATE.failures[: len(_STATE.failures) - 500]
    line = f"client.js/{scope}:{where} {message}"
    _log_and_print("warning", line)
    _emit_event({"kind": "failure", "record": asdict(rec)})
    return rec


# ── HTTP route ─────────────────────────────────────────────────────────────
_ROUTES_REGISTERED = False


def register_routes(server) -> None:
    """Idempotently register ``GET /c2c/registry/status`` on the given
    PromptServer instance."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED or server is None:
        return
    try:
        from aiohttp import web
    except Exception as exc:  # pragma: no cover
        _log.warning("registry route disabled (aiohttp missing): %s", exc)
        return

    routes = server.routes if hasattr(server, "routes") else server.app.router

    @routes.get("/c2c/registry/status")
    async def _status(_request):  # noqa: ANN001
        return web.json_response(summary())

    @routes.post("/c2c/registry/failure")
    async def _failure(request):  # noqa: ANN001
        """Ingest a JS-side failure (e.g. from ``_c2c_undo.js``). Accepts
        an arbitrary JSON object; tolerant of malformed payloads (returns
        200 with ``ok=false`` rather than 4xx so the fire-and-forget JS
        path never spams console noise)."""
        try:
            payload = await request.json()
        except Exception as exc:
            return web.json_response(
                {"ok": False, "reason": "invalid_json", "detail": str(exc)[:200]},
                status=200,
            )
        if not isinstance(payload, dict):
            return web.json_response(
                {"ok": False, "reason": "payload_must_be_object"},
                status=200,
            )
        rec = record_client_failure(payload)
        return web.json_response({"ok": True, "key": rec.key, "group": rec.group}, status=200)

    # ALSO mirror under /mec/ for back-compat with v1 toasts and panels.
    @routes.get("/mec/registry/status")
    async def _status_mec(request):  # noqa: ANN001
        return await _status(request)

    @routes.post("/mec/registry/failure")
    async def _failure_mec(request):  # noqa: ANN001
        return await _failure(request)

    _ROUTES_REGISTERED = True
    _log.info(
        "[C2C registry] routes registered: "
        "GET /c2c/registry/status, POST /c2c/registry/failure (+/mec/ mirrors)"
    )
