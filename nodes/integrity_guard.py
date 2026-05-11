# FILE: nodes/integrity_guard.py
# FEATURE: F7 — Conflict & Integrity Guard
# INTEGRATES WITH: web/extensions/nukenodemax/integrity_badges.js
"""
Background checks that fire on import (deferred 20s after startup so they
don't compete with ComfyUI-Manager's own update check):

    1. `pip check` — detects unmet / conflicting dependency versions.
       Uses `uv` if available (5–20x faster, no Python interpreter spin-up);
       falls back to `python -m pip check` automatically when uv is absent.
    2. Package list size — uses `uv pip list --format json` when uv is
       present; skipped entirely otherwise (it was a cosmetic stat that
       previously required `pipdeptree` and was the main startup hog).
    3. SHA-256 checksum of every .py under this pack vs a baseline file
       (third_party/checksums.json). Drift -> warning event.

Results are cached to ``.integrity_cache.json`` for 24h, keyed by a
fingerprint of the Python interpreter + this pack's .py file mtimes. On
restart we load the cache instantly instead of re-running subprocesses.

Findings are pushed to the JS frontend via PromptServer socket
"nukenodemax.integrity". The JS overlay draws warning badges on affected
nodes (or globally if pip).

Exposes a node `IntegrityStatusMEC` so users can request a status string
mid-graph (with optional force-rescan) and an HTTP endpoint
`/nukenodemax/reinstall?package=NAME` that runs `uv pip install --reinstall`
(or `pip install --force-reinstall` as fallback), guarded behind a confirm
flag.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("MEC.integrity")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK_ROOT = os.path.dirname(_HERE)
_CHECKSUM_PATH = os.path.join(_PACK_ROOT, "third_party", "checksums.json")
_CACHE_PATH = os.path.join(_PACK_ROOT, ".integrity_cache.json")

# Cache TTL: how long a successful scan stays valid before re-running.
_CACHE_TTL_SECONDS = 24 * 3600  # 24h

# Defer the initial background scan so ComfyUI + Manager finish their own
# startup work first. This stops Manager's update-check from being kicked
# off because our subprocess churn happens to land in its startup window.
_STARTUP_DELAY_SECONDS = 20

# Locate `uv` once at import time. uv is ~5–20x faster than `python -m pip`
# at metadata-only operations (pip check, pip tree). Falls back to pip if
# uv isn't installed.
_UV_BIN: Optional[str] = shutil.which("uv")

_LOCK = threading.Lock()
_LAST_REPORT: Dict[str, Any] = {"ready": False}


# =====================================================================
# Helpers
# =====================================================================
def _emit(event: Dict[str, Any]) -> None:
    try:
        import server as _comfy_server  # type: ignore
        ps = _comfy_server.PromptServer.instance
        ps.send_sync("nukenodemax.integrity", event)
    except Exception as e:
        log.debug("[integrity] socket emit failed: %s", e)


def _run(cmd: List[str], timeout: int = 60,
         stdout_limit: int = 8000, stderr_limit: int = 4000) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "stdout": proc.stdout if stdout_limit is None
                      else proc.stdout.strip()[:stdout_limit],
            "stderr": proc.stderr if stderr_limit is None
                      else proc.stderr.strip()[:stderr_limit],
        }
    except FileNotFoundError as e:
        return {"ok": False, "rc": -1, "stdout": "", "stderr": str(e)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "rc": -2, "stdout": "", "stderr": f"timeout after {timeout}s"}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_py_files() -> List[str]:
    out = []
    for dirpath, _, files in os.walk(_PACK_ROOT):
        if any(seg in dirpath for seg in (".git", "__pycache__", ".pytest_cache",
                                          ".ruff_cache")):
            continue
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(dirpath, f))
    return out


# =====================================================================
# Cache helpers
# =====================================================================
def _env_fingerprint() -> str:
    """Hash that changes when the Python env or this pack's .py files change.

    Captures:
      - sys.executable (which interpreter)
      - sys.version    (Python version)
      - mtimes of every .py in the pack (catches edits & git pulls)

    Does NOT capture installed-package versions — `pip check` itself is
    cheap enough to re-run on cache miss; capturing the full env would
    require listing all distributions which is the very cost we're trying
    to avoid.
    """
    h = hashlib.sha256()
    h.update(sys.executable.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(sys.version.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    files = sorted(_walk_py_files())
    for p in files:
        try:
            st = os.stat(p)
            h.update(os.path.relpath(p, _PACK_ROOT).replace("\\", "/").encode("utf-8"))
            h.update(f":{int(st.st_mtime)}:{st.st_size}\n".encode("ascii"))
        except OSError:
            pass
    return h.hexdigest()


def _load_cache() -> Optional[Dict[str, Any]]:
    """Return cached report if fresh AND fingerprint matches, else None."""
    try:
        if not os.path.isfile(_CACHE_PATH):
            return None
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = float(data.get("_cached_at", 0))
        if time.time() - ts > _CACHE_TTL_SECONDS:
            return None
        if data.get("_fingerprint") != _env_fingerprint():
            return None
        # Strip cache-only fields before returning.
        report = {k: v for k, v in data.items()
                  if k not in ("_cached_at", "_fingerprint")}
        return report
    except (OSError, json.JSONDecodeError, ValueError) as e:
        log.debug("[integrity] cache load failed: %s", e)
        return None


def _save_cache(report: Dict[str, Any]) -> None:
    try:
        payload = dict(report)
        payload["_cached_at"] = time.time()
        payload["_fingerprint"] = _env_fingerprint()
        tmp = _CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, _CACHE_PATH)
    except OSError as e:
        log.debug("[integrity] cache save failed: %s", e)


def _run_pip_check() -> Dict[str, Any]:
    """Run `pip check` via uv if available, else fall back to pip.

    NOTE: ``uv pip check`` writes its findings to stderr, while ``pip check``
    writes them to stdout. We merge both into ``stdout`` here so the caller
    can parse a single field regardless of backend.
    """
    if _UV_BIN:
        out = _run(
            [_UV_BIN, "pip", "check", "--python", sys.executable],
            timeout=45,
            stdout_limit=16000, stderr_limit=16000,
        )
        # uv emits the conflict list to stderr; promote it to stdout so the
        # downstream line-parser sees it.
        merged = "\n".join(s for s in (out.get("stdout", ""), out.get("stderr", "")) if s).strip()
        # Strip the "Using Python ... environment at: ..." preamble lines.
        cleaned = "\n".join(
            ln for ln in merged.splitlines()
            if ln.strip() and not ln.startswith("Using Python")
               and not ln.startswith("Checked ")
        )
        out["stdout"] = cleaned[:16000]
        return out
    return _run([sys.executable, "-m", "pip", "check"], timeout=45,
                stdout_limit=16000)


def _run_dep_tree_size() -> int:
    """Return installed-package count, or -1 if unavailable.

    Uses `uv pip list --format json` when uv is installed (fast, native).
    If uv is not installed we skip this step entirely instead of falling
    back to pipdeptree — pipdeptree is slow, requires a separate install,
    and the count is purely a cosmetic stat.
    """
    if not _UV_BIN:
        return -1
    out = _run(
        [_UV_BIN, "pip", "list", "--python", sys.executable, "--format", "json"],
        timeout=30,
        stdout_limit=None,  # JSON may be much larger than 8 KB
    )
    if not out["ok"]:
        return -1
    try:
        return len(json.loads(out["stdout"]))
    except (json.JSONDecodeError, ValueError):
        return -1


# =====================================================================
# Worker
# =====================================================================
def _worker(force: bool = False, delay: float = 0.0):
    if delay > 0:
        time.sleep(delay)

    # ---- cache fast-path ----
    if not force:
        cached = _load_cache()
        if cached is not None:
            cached["from_cache"] = True
            with _LOCK:
                _LAST_REPORT.clear()
                _LAST_REPORT.update(cached)
            _emit({"type": "report", **cached})
            log.info("[integrity] using cached report (%d events)",
                     len(cached.get("events", [])))
            return

    report: Dict[str, Any] = {"ready": True, "events": [], "from_cache": False}

    # ---- pip check (uv if available, else pip) ----
    pip_check = _run_pip_check()
    report["pip_check"] = pip_check
    report["used_uv"] = bool(_UV_BIN)
    if not pip_check["ok"]:
        for line in pip_check["stdout"].splitlines():
            line = line.strip()
            if not line:
                continue
            report["events"].append({
                "kind": "dependency_conflict",
                "severity": "warn",
                "message": line,
            })

    # ---- dep tree size (uv only; skipped otherwise) ----
    report["dep_tree_size"] = _run_dep_tree_size()

    # ---- checksum verification ----
    expected = {}
    if os.path.isfile(_CHECKSUM_PATH):
        try:
            expected = json.load(open(_CHECKSUM_PATH, "r", encoding="utf-8"))
        except Exception as e:
            log.warning("[integrity] cannot read %s: %s", _CHECKSUM_PATH, e)
    drift = []
    for path in _walk_py_files():
        rel = os.path.relpath(path, _PACK_ROOT).replace("\\", "/")
        if rel in expected:
            actual = _sha256(path)
            if actual != expected[rel]:
                drift.append({"file": rel, "expected": expected[rel], "actual": actual})
    report["checksum_drift"] = drift
    for d in drift:
        report["events"].append({
            "kind": "checksum_drift",
            "severity": "warn",
            "message": f"{d['file']} hash differs from baseline",
            "file": d["file"],
        })

    with _LOCK:
        _LAST_REPORT.clear()
        _LAST_REPORT.update(report)
    _save_cache(report)
    _emit({"type": "report", **report})
    log.info("[integrity] scan complete (%d events, uv=%s)",
             len(report["events"]), bool(_UV_BIN))


def start_background_scan(force: bool = False, delay: Optional[float] = None) -> None:
    """Spawn the background scan.

    On normal import-time autostart we delay by ``_STARTUP_DELAY_SECONDS``
    so ComfyUI and ComfyUI-Manager finish their own startup work before
    we touch pip metadata (this is what was causing Manager's update
    checker to be triggered). Manual triggers (rescan widget, reinstall
    route) run immediately.
    """
    if delay is None:
        delay = _STARTUP_DELAY_SECONDS if not force else 0.0
    t = threading.Thread(
        target=_worker,
        kwargs={"force": force, "delay": delay},
        name="MEC-integrity",
        daemon=True,
    )
    t.start()


# =====================================================================
# Server endpoint /nukenodemax/reinstall
# =====================================================================
def register_routes(server) -> None:
    try:
        from aiohttp import web
    except Exception:
        log.warning("[integrity] aiohttp unavailable")
        return
    routes = server.routes

    @routes.get("/nukenodemax/integrity_report")
    async def _report(req):  # noqa: ARG001
        with _LOCK:
            return web.json_response(dict(_LAST_REPORT))

    @routes.post("/nukenodemax/reinstall")
    async def _reinstall(req):
        body = {}
        try:
            body = await req.json()
        except Exception:
            pass
        package = (body or {}).get("package") or req.query.get("package")
        confirm = (body or {}).get("confirm") or req.query.get("confirm")
        if not package:
            return web.json_response({"ok": False, "error": "missing 'package'"}, status=400)
        if confirm != "yes":
            return web.json_response({
                "ok": False,
                "error": "destructive action: pass confirm=yes to proceed",
            }, status=400)
        # Restrict package name to a safe charset to defend against shell-injection.
        import re
        if not re.match(r"^[A-Za-z0-9._\-\[\]=<>!~+]+$", package):
            return web.json_response({"ok": False, "error": "invalid package name"},
                                      status=400)
        if _UV_BIN:
            install_cmd = [_UV_BIN, "pip", "install",
                           "--python", sys.executable,
                           "--reinstall", package]
        else:
            install_cmd = [sys.executable, "-m", "pip", "install",
                           "--force-reinstall", package]
        result = _run(install_cmd, timeout=600)
        # Rescan after the change — force, bypass cache, run immediately.
        start_background_scan(force=True, delay=0.0)
        return web.json_response({"ok": result["ok"], **result})

    log.info("[integrity] routes registered")


# =====================================================================
# Node
# =====================================================================
class IntegrityStatusMEC:
    DESCRIPTION = "Returns the latest integrity scan as a string."
    CATEGORY = "MaskEditControl/Diagnostic"
    FUNCTION = "status"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"trigger_rescan": ("BOOLEAN", {"default": False})}}

    def status(self, trigger_rescan: bool):
        if trigger_rescan:
            # User-initiated: bypass cache and run immediately.
            start_background_scan(force=True, delay=0.0)
        elif not _LAST_REPORT.get("ready"):
            # Auto-load from cache if available, else spawn a fresh scan
            # with no delay (the user is actively asking for a result).
            start_background_scan(force=False, delay=0.0)
        with _LOCK:
            r = dict(_LAST_REPORT)
        if not r.get("ready"):
            return ("integrity scan running…",)
        events = r.get("events", [])
        head = (f"events={len(events)} pip_check_ok={r['pip_check']['ok']} "
                f"checksum_drift={len(r.get('checksum_drift', []))}")
        body = "\n".join(f"  - {e['kind']}: {e['message']}" for e in events[:20])
        return (head + ("\n" + body if body else ""),)


NODE_CLASS_MAPPINGS = {"IntegrityStatusMEC": IntegrityStatusMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"IntegrityStatusMEC": "Integrity Status (MEC)"}
