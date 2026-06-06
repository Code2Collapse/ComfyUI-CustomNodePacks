"""
c2c_int_aggregator.py — P0.6 INT (Integrity) status aggregator.

Merges four signal sources into a single health status with a 4-color level:

  green   "ok"    — no warnings or errors anywhere
  yellow  "warn"  — at least one warning (doctor warnings, integrity events,
                    high VRAM headroom, recent OOM-like hints)
  red     "err"   — at least one error (doctor errors, runtime node_error
                    events in the recent window, damaged-package signals)
  purple  "crit"  — pip dependency check failed OR multiple recent OOMs
                    OR the last prompt definitively failed.

Signal sources (all best-effort — every read is wrapped so a missing module
never breaks the aggregator):
  1. Static workflow lint           — `workflow_doctor.analyze(workflow)`
  2. Runtime telemetry ring         — `mec_diagnostics_api._BUFFER`
                                      (populated via insight bridge)
  3. Package / checksum integrity   — `integrity_guard._LAST_REPORT`
  4. Component registry failures    — `_c2c_registry.summary()`

Public HTTP routes:
  GET  /c2c/int/health              — aggregate WITHOUT workflow lint
  POST /c2c/int/health              — body {"workflow": {...}, "window_s": int}
                                      → aggregate WITH workflow lint
  GET  /c2c/int/runs?n=50           — recent runtime events (filtered)

Response envelope:
  {
    "ok": true,
    "level": "ok" | "warn" | "err" | "crit",
    "label": "Healthy" | "Degraded" | "Errors" | "Critical",
    "counts": {
        "doctor_errors": int, "doctor_warnings": int, "doctor_infos": int,
        "runtime_errors": int, "runtime_total": int,
        "integrity_events": int, "checksum_drift": int,
        "registry_failures": int,
        "ooms_recent": int
    },
    "sections": {
        "doctor": {...},          # short summary if lint ran
        "runtime": {...},
        "integrity": {...},
        "registry": {...}
    },
    "vram": {"peak_mb_recent": float, "delta_mb_recent": float},
    "last_event_ts": float | None,
    "window_s": int,
    "ts": float
  }
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("C2C.IntAggregator")

# Default look-back window for runtime events (seconds).
_DEFAULT_WINDOW_S = 300

# Phrases that we treat as OOM evidence inside an exception text or hint.
_OOM_KEYWORDS = (
    "out of memory",
    "cuda out of memory",
    "outofmemoryerror",
    "cuda ran out of vram",
    "ran out of vram",
)


def _safe(getter, default=None):
    try:
        return getter()
    except Exception as exc:  # pragma: no cover (defensive)
        log.debug("[int] source unavailable: %s", exc)
        return default


# ─────────────────────────────────────────────────────────────────────────
# Source readers
# ─────────────────────────────────────────────────────────────────────────
def _read_runtime_buffer(window_s: int) -> Dict[str, Any]:
    """Read recent events from `mec_diagnostics_api._BUFFER`."""
    try:
        from . import mec_diagnostics_api as _mda
    except Exception:
        try:
            from nodes import mec_diagnostics_api as _mda  # type: ignore
        except Exception:
            return {"available": False}

    cutoff = time.time() - max(1, int(window_s))
    with _mda._BUFFER_LOCK:
        snapshot = list(_mda._BUFFER)

    recent = [e for e in snapshot if float(e.get("ts", 0) or 0) >= cutoff]
    errors = [e for e in recent
              if e.get("type") == "node_error" or e.get("severity") in ("error",)]
    warns = [e for e in recent if e.get("severity") in ("warn", "warning")]

    # Peak/delta VRAM in recent window
    peak_mb = 0.0
    delta_mb = 0.0
    for e in recent:
        v = e.get("vram_peak_mb")
        if isinstance(v, (int, float)) and v > peak_mb:
            peak_mb = float(v)
        d = e.get("vram_delta_mb")
        if isinstance(d, (int, float)) and d > delta_mb:
            delta_mb = float(d)

    # OOM signal: count node_error events whose exc_type or hint matches OOM.
    oom_recent = 0
    last_error: Optional[Dict[str, Any]] = None
    for e in errors:
        msg = " ".join(str(e.get(k, "") or "") for k in ("exc_type", "exc_msg", "hint")).lower()
        if any(kw in msg for kw in _OOM_KEYWORDS):
            oom_recent += 1
        last_error = e

    last_ts = max((float(e.get("ts", 0) or 0) for e in snapshot), default=None)

    return {
        "available": True,
        "buffer_len": len(snapshot),
        "recent_total": len(recent),
        "recent_errors": len(errors),
        "recent_warnings": len(warns),
        "ooms_recent": oom_recent,
        "vram_peak_mb_recent": round(peak_mb, 2),
        "vram_delta_mb_recent": round(delta_mb, 2),
        "last_error": last_error,
        "last_event_ts": last_ts,
    }


def _read_integrity_report() -> Dict[str, Any]:
    """Read `integrity_guard._LAST_REPORT`."""
    try:
        from . import integrity_guard as _ig
    except Exception:
        try:
            from nodes import integrity_guard as _ig  # type: ignore
        except Exception:
            return {"available": False}
    try:
        with _ig._LOCK:
            r = dict(_ig._LAST_REPORT)
    except Exception:
        return {"available": False}

    events = r.get("events") or []
    pip = r.get("pip_check") or {}
    drift = r.get("checksum_drift") or []
    severities = [str(e.get("severity", "info")).lower() for e in events
                  if isinstance(e, dict)]
    n_err = sum(1 for s in severities if s in ("error", "critical"))
    n_warn = sum(1 for s in severities if s in ("warn", "warning"))
    return {
        "available": True,
        "ready": bool(r.get("ready")),
        "events_total": len(events),
        "events_error": n_err,
        "events_warn": n_warn,
        "pip_check_ok": bool(pip.get("ok", True)),
        "pip_check_detail": pip.get("detail") or pip.get("output") or pip.get("stdout") or "",
        "checksum_drift": len(drift),
        "suspicious_files": int(r.get("suspicious_files") or 0),
        "ts": r.get("ts"),
    }


def _read_environment(disk_refresh: bool = False) -> Dict[str, Any]:
    """Read environment diagnostics from c2c_doctor (pyenv + disk)."""
    try:
        from . import c2c_doctor as _cd
    except Exception:
        try:
            from nodes import c2c_doctor as _cd  # type: ignore
        except Exception:
            return {"available": False}
    pyenv = _safe(_cd.collect_pyenv, {}) or {}
    disk = _safe(lambda: _cd.collect_disk(refresh=disk_refresh), {}) or {}
    # Flatten a few headline counters for the badge / popover header.
    py_warnings = 0
    py_errors = 0
    try:
        for pkg in (pyenv.get("packages") or []):
            st = (pkg.get("status") or "").lower()
            if st in ("missing", "error"):
                py_errors += 1
            elif st in ("outdated", "warn", "warning"):
                py_warnings += 1
    except Exception:
        pass
    return {
        "available": True,
        "pyenv": pyenv,
        "disk": disk,
        "py_warnings": py_warnings,
        "py_errors": py_errors,
    }


def _read_registry_summary() -> Dict[str, Any]:
    try:
        from . import _c2c_registry as _reg
    except Exception:
        try:
            from nodes import _c2c_registry as _reg  # type: ignore
        except Exception:
            return {"available": False}
    try:
        s = _reg.summary()
    except Exception:
        return {"available": False}
    return {
        "available": True,
        "failures": int(s.get("counts", {}).get("failures", 0) or 0),
        "missing_deps": int(s.get("counts", {}).get("missing_deps", 0) or 0),
        "missing_weights": int(s.get("counts", {}).get("missing_weights", 0) or 0),
        "ready": int(s.get("counts", {}).get("ready", 0) or 0),
    }


def _run_doctor(workflow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not workflow:
        return {"available": False, "ran": False}
    try:
        from . import workflow_doctor as _wd
    except Exception:
        try:
            from nodes import workflow_doctor as _wd  # type: ignore
        except Exception:
            return {"available": False, "ran": False}
    try:
        res = _wd.analyze(workflow)
    except Exception as exc:
        log.warning("[int] doctor analyze failed: %s", exc)
        return {"available": True, "ran": False, "error": str(exc)}
    findings = res.get("findings") or []
    by_sev: Dict[str, int] = {"error": 0, "warning": 0, "info": 0}
    top: List[Dict[str, Any]] = []
    for f in findings:
        sev = str(f.get("severity", "info")).lower()
        by_sev[sev] = by_sev.get(sev, 0) + 1
        if sev in ("error", "warning") and len(top) < 5:
            top.append({
                "rule": f.get("id") or f.get("rule"),
                "severity": sev,
                "detail": f.get("detail", "")[:240],
                "node_id": f.get("node_id"),
                "node_type": f.get("node_type"),
                "has_fix": bool(f.get("fix")),
            })
    return {
        "available": True,
        "ran": True,
        "errors": by_sev.get("error", 0),
        "warnings": by_sev.get("warning", 0),
        "infos": by_sev.get("info", 0),
        "total": len(findings),
        "top": top,
        "stats": res.get("stats") or {},
    }


# ─────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────
_LEVEL_RANK = {"ok": 0, "warn": 1, "err": 2, "crit": 3}
_LEVEL_LABEL = {"ok": "Healthy", "warn": "Degraded", "err": "Errors", "crit": "Critical"}


def _bump(current: str, candidate: str) -> str:
    if _LEVEL_RANK.get(candidate, 0) > _LEVEL_RANK.get(current, 0):
        return candidate
    return current


def aggregate(workflow: Optional[Dict[str, Any]] = None,
              window_s: int = _DEFAULT_WINDOW_S) -> Dict[str, Any]:
    rt = _safe(lambda: _read_runtime_buffer(window_s), {"available": False}) or {"available": False}
    ig = _safe(_read_integrity_report, {"available": False}) or {"available": False}
    rg = _safe(_read_registry_summary, {"available": False}) or {"available": False}
    en = _safe(_read_environment, {"available": False}) or {"available": False}
    dr = _safe(lambda: _run_doctor(workflow), {"available": False, "ran": False}) \
        or {"available": False, "ran": False}

    level = "ok"

    # ── crit ──
    # Pip check failed → environment is broken: critical.
    if ig.get("available") and ig.get("pip_check_ok") is False:
        level = _bump(level, "crit")
    # >=2 OOMs in the window → out of memory situation
    if int(rt.get("ooms_recent", 0) or 0) >= 2:
        level = _bump(level, "crit")
    # Integrity events flagged 'critical' (subset of events_error already)
    # plus checksum drift on a deployed package = critical.
    if int(ig.get("checksum_drift", 0) or 0) > 0 and int(ig.get("events_error", 0) or 0) > 0:
        level = _bump(level, "crit")

    # ── err ──
    if int(dr.get("errors", 0) or 0) > 0:
        level = _bump(level, "err")
    if int(rt.get("recent_errors", 0) or 0) > 0:
        level = _bump(level, "err")
    if int(ig.get("events_error", 0) or 0) > 0:
        level = _bump(level, "err")
    if int(rg.get("failures", 0) or 0) > 0:
        level = _bump(level, "err")

    # ── warn ──
    if int(dr.get("warnings", 0) or 0) > 0:
        level = _bump(level, "warn")
    if int(rt.get("recent_warnings", 0) or 0) > 0:
        level = _bump(level, "warn")
    if int(ig.get("events_warn", 0) or 0) > 0:
        level = _bump(level, "warn")
    if int(rt.get("ooms_recent", 0) or 0) == 1:
        level = _bump(level, "warn")
    if int(ig.get("checksum_drift", 0) or 0) > 0:
        level = _bump(level, "warn")
    if int(rg.get("missing_deps", 0) or 0) > 0 or int(rg.get("missing_weights", 0) or 0) > 0:
        level = _bump(level, "warn")
    if int(en.get("py_errors", 0) or 0) > 0:
        level = _bump(level, "err")
    if int(en.get("py_warnings", 0) or 0) > 0:
        level = _bump(level, "warn")

    counts = {
        "doctor_errors": int(dr.get("errors", 0) or 0),
        "doctor_warnings": int(dr.get("warnings", 0) or 0),
        "doctor_infos": int(dr.get("infos", 0) or 0),
        "runtime_errors": int(rt.get("recent_errors", 0) or 0),
        "runtime_warnings": int(rt.get("recent_warnings", 0) or 0),
        "runtime_total": int(rt.get("recent_total", 0) or 0),
        "integrity_events": int(ig.get("events_total", 0) or 0),
        "integrity_errors": int(ig.get("events_error", 0) or 0),
        "checksum_drift": int(ig.get("checksum_drift", 0) or 0),
        "registry_failures": int(rg.get("failures", 0) or 0),
        "ooms_recent": int(rt.get("ooms_recent", 0) or 0),
        "env_errors": int(en.get("py_errors", 0) or 0),
        "env_warnings": int(en.get("py_warnings", 0) or 0),
    }

    return {
        "ok": True,
        "level": level,
        "label": _LEVEL_LABEL[level],
        "counts": counts,
        "sections": {
            "doctor": dr,
            "runtime": rt,
            "integrity": ig,
            "registry": rg,
            "environment": en,
        },
        "vram": {
            "peak_mb_recent": float(rt.get("vram_peak_mb_recent", 0.0) or 0.0),
            "delta_mb_recent": float(rt.get("vram_delta_mb_recent", 0.0) or 0.0),
        },
        "last_event_ts": rt.get("last_event_ts"),
        "window_s": int(window_s),
        "ts": time.time(),
    }


# ─────────────────────────────────────────────────────────────────────────
# HTTP routes
# ─────────────────────────────────────────────────────────────────────────
_ROUTES_REGISTERED = False


def register_routes(server) -> None:
    """Idempotently register the /c2c/int/* routes."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED or server is None:
        return
    try:
        from aiohttp import web
    except Exception as exc:  # pragma: no cover
        log.warning("[int] aiohttp unavailable: %s", exc)
        return

    routes = server.routes if hasattr(server, "routes") else server.app.router

    @routes.get("/c2c/int/health")
    async def _get_health(req):  # noqa: ANN001
        try:
            window_s = int(req.query.get("window_s") or _DEFAULT_WINDOW_S)
        except Exception:
            window_s = _DEFAULT_WINDOW_S
        return web.json_response(aggregate(workflow=None, window_s=window_s))

    @routes.post("/c2c/int/health")
    async def _post_health(req):  # noqa: ANN001
        body: Dict[str, Any] = {}
        try:
            body = await req.json()
        except Exception:
            body = {}
        wf = body.get("workflow")
        try:
            window_s = int(body.get("window_s") or req.query.get("window_s") or _DEFAULT_WINDOW_S)
        except Exception:
            window_s = _DEFAULT_WINDOW_S
        return web.json_response(aggregate(workflow=wf, window_s=window_s))

    @routes.get("/c2c/int/runs")
    async def _get_runs(req):  # noqa: ANN001
        try:
            n = max(1, min(500, int(req.query.get("n") or 50)))
        except Exception:
            n = 50
        try:
            from . import mec_diagnostics_api as _mda
        except Exception:
            try:
                from nodes import mec_diagnostics_api as _mda  # type: ignore
            except Exception:
                return web.json_response({"ok": False, "error": "mec_diagnostics_api unavailable",
                                          "items": []}, status=503)
        with _mda._BUFFER_LOCK:
            items = list(_mda._BUFFER)
        # newest first, project to a compact shape
        out: List[Dict[str, Any]] = []
        for e in items[-n:][::-1]:
            out.append({
                "ts": e.get("ts"),
                "type": e.get("type"),
                "node_id": e.get("node_id"),
                "elapsed_ms": e.get("elapsed_ms"),
                "vram_peak_mb": e.get("vram_peak_mb"),
                "vram_delta_mb": e.get("vram_delta_mb"),
                "exc_type": e.get("exc_type"),
                "exc_msg": e.get("exc_msg"),
                "hint": e.get("hint"),
                "severity": e.get("severity"),
            })
        return web.json_response({"ok": True, "items": out, "count": len(out)})

    _ROUTES_REGISTERED = True
    log.info("[C2C int] routes registered: GET/POST /c2c/int/health, GET /c2c/int/runs")
