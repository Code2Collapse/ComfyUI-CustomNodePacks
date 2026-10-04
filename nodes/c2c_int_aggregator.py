"""
c2c_int_aggregator.py — P0.6 INT (Integrity) status aggregator.

Merges four signal sources into a single health status with a 4-color level:

  green   "ok"    — no warnings or errors anywhere
  yellow  "warn"  — at least one warning (doctor warnings, integrity events,
                    high VRAM headroom, recent OOM-like hints)
  red     "err"   — at least one error (doctor errors, runtime node_error
                    events in the recent window, damaged-package signals)
  purple  "crit"  — a ComfyUI-core package fails pip check OR multiple recent OOMs
                  (other pip conflicts are "warn": most real envs have some)
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

import asyncio
import functools
import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

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
    # Never walk folders here: this runs for a status badge every few seconds.
    # The last snapshot comes back at once; a stale one refreshes in the background.
    disk = _safe(lambda: (_cd.collect_disk(refresh=True) if disk_refresh
                          else _cd.collect_disk_nowait()), {}) or {}
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


# Same bytes, same verdict - but the lint also looks at files on disk (missing
# models), so a verdict is reused for one minute at most.
_DOCTOR_CACHE: Dict[str, Any] = {"key": None, "data": None, "ts": 0.0}
_DOCTOR_TTL_S = 60.0


def _run_doctor(workflow: Optional[Dict[str, Any]], key: Optional[str] = None) -> Dict[str, Any]:
    """Lint the workflow. `key` identifies its exact content (a hash of the
    request body): the badge re-sends an unchanged workflow every few seconds,
    and the verdict on the same bytes is the same."""
    if not workflow:
        return {"available": False, "ran": False}
    if (key is not None and _DOCTOR_CACHE["key"] == key
            and time.time() - _DOCTOR_CACHE["ts"] < _DOCTOR_TTL_S):
        return _DOCTOR_CACHE["data"]
    res = _run_doctor_uncached(workflow)
    if key is not None:
        _DOCTOR_CACHE["key"] = key
        _DOCTOR_CACHE["data"] = res
        _DOCTOR_CACHE["ts"] = time.time()
    return res


def _run_doctor_uncached(workflow: Dict[str, Any]) -> Dict[str, Any]:
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
# Packages ComfyUI itself imports on start or on every run. A pip conflict in
# one of THESE (e.g. "torchvision X has requirement torch==Y") can break
# ComfyUI; a conflict elsewhere breaks at most the node pack that uses it.
_CORE_PKGS = frozenset({
    "torch", "torchvision", "torchaudio", "numpy", "safetensors", "aiohttp",
    "comfyui-frontend-package", "comfyui-workflow-templates", "comfyui-embedded-docs",
    "transformers", "tokenizers", "sentencepiece", "pillow", "pyyaml", "scipy",
    "einops", "kornia", "spandrel", "av", "torchsde", "psutil", "alembic", "sqlalchemy",
})


def _norm_pkg(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(".", "-")


def _pip_conflicts(ig: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Split `pip check` output into (core, other) lines. The package that
    is broken is the one named first: "<pkg> <ver> requires ..." /
    "<pkg> <ver> has requirement ...". Unparseable failure output counts as
    `other` - it is worth a look, not an alarm."""
    if not ig.get("available") or ig.get("pip_check_ok") is not False:
        return [], []
    lines = [ln.strip() for ln in str(ig.get("pip_check_detail") or "").splitlines() if ln.strip()]
    # uv says "Found 6 incompatibilities" first, then
    # "The package `mediapipe` requires `numpy<2`, but `2.4.6` is installed".
    lines = [ln for ln in lines if not re.match(r"(Found|Checked|Resolved|Using|warning:)\b", ln)]
    core: List[str] = []
    other: List[str] = []
    for ln in lines:
        m = re.match(r"The package `([^`]+)`", ln)
        head = m.group(1) if m else ln.split(" ", 1)[0]
        (core if _norm_pkg(head) in _CORE_PKGS else other).append(ln)
    if not lines:
        other.append("pip check failed without output")
    return core, other


_LEVEL_RANK = {"ok": 0, "warn": 1, "err": 2, "crit": 3}
_LEVEL_LABEL = {"ok": "Healthy", "warn": "Degraded", "err": "Errors", "crit": "Critical"}


def _bump(current: str, candidate: str) -> str:
    if _LEVEL_RANK.get(candidate, 0) > _LEVEL_RANK.get(current, 0):
        return candidate
    return current


def aggregate(workflow: Optional[Dict[str, Any]] = None,
              window_s: int = _DEFAULT_WINDOW_S,
              workflow_key: Optional[str] = None) -> Dict[str, Any]:
    rt = _safe(lambda: _read_runtime_buffer(window_s), {"available": False}) or {"available": False}
    ig = _safe(_read_integrity_report, {"available": False}) or {"available": False}
    rg = _safe(_read_registry_summary, {"available": False}) or {"available": False}
    en = _safe(_read_environment, {"available": False}) or {"available": False}
    dr = _safe(lambda: _run_doctor(workflow, workflow_key), {"available": False, "ran": False}) \
        or {"available": False, "ran": False}

    level = "ok"
    pip_core, pip_other = _pip_conflicts(ig)

    # ── crit ──
    # A package ComfyUI itself runs on is broken → critical. Any other pip
    # conflict is only Degraded: nearly every real ComfyUI env fails
    # `pip check` somewhere (an unused extra, a stale pin like mediapipe's
    # numpy<2) and still runs - treating those as Critical kept the badge
    # purple forever, so nobody looked at it.
    if pip_core:
        level = _bump(level, "crit")
    elif pip_other:
        level = _bump(level, "warn")
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
        "pip_core_conflicts": len(pip_core),
        "pip_conflicts": len(pip_core) + len(pip_other),
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

    # aggregate() does real work (package metadata, workflow lint, file
    # reads). On the event loop it froze the WHOLE server - queueing, the
    # progress websocket, every other route - for as long as it ran, and the
    # badge asks every few seconds. It runs on a worker thread instead.
    async def _aggregate_off_loop(**kw):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(aggregate, **kw))

    @routes.get("/c2c/int/health")
    async def _get_health(req):  # noqa: ANN001
        try:
            window_s = int(req.query.get("window_s") or _DEFAULT_WINDOW_S)
        except Exception:
            window_s = _DEFAULT_WINDOW_S
        return web.json_response(await _aggregate_off_loop(workflow=None, window_s=window_s))

    @routes.post("/c2c/int/health")
    async def _post_health(req):  # noqa: ANN001
        body: Dict[str, Any] = {}
        raw = b""
        try:
            raw = await req.read()
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}
        wf = body.get("workflow") if isinstance(body, dict) else None
        try:
            window_s = int(body.get("window_s") or req.query.get("window_s") or _DEFAULT_WINDOW_S)
        except Exception:
            window_s = _DEFAULT_WINDOW_S
        key = hashlib.sha1(raw).hexdigest() if raw else None
        return web.json_response(await _aggregate_off_loop(workflow=wf, window_s=window_s,
                                                           workflow_key=key))

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
