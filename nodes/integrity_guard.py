# FILE: nodes/integrity_guard.py
# FEATURE: F7 — Conflict & Integrity Guard
# INTEGRATES WITH: web/extensions/nukenodemax/integrity_badges.js
"""
Background checks that fire on import:

    1. `pip check` — detects unmet / conflicting dependency versions.
    2. `pipdeptree --json` — builds a dep graph (best effort).
    3. SHA-256 checksum of every .py under this pack vs a baseline file
       (third_party/checksums.json). Drift -> warning event.

Findings are pushed to the JS frontend via PromptServer socket
"nukenodemax.integrity". The JS overlay draws warning badges on affected
nodes (or globally if pip).

Exposes a node `IntegrityStatusMEC` so users can request a status string
mid-graph and an HTTP endpoint `/nukenodemax/reinstall?package=NAME` that
runs `pip install --force-reinstall NAME` (guarded behind a confirm flag).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

log = logging.getLogger("MEC.integrity")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK_ROOT = os.path.dirname(_HERE)
_CHECKSUM_PATH = os.path.join(_PACK_ROOT, "third_party", "checksums.json")

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


def _run(cmd: List[str], timeout: int = 60) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "stdout": proc.stdout.strip()[:8000],
            "stderr": proc.stderr.strip()[:4000],
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
# Worker
# =====================================================================
def _worker():
    report: Dict[str, Any] = {"ready": True, "events": []}

    # ---- pip check ----
    pip_check = _run([sys.executable, "-m", "pip", "check"], timeout=45)
    report["pip_check"] = pip_check
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

    # ---- pipdeptree (best effort) ----
    pdt = _run([sys.executable, "-m", "pipdeptree", "--json-tree"], timeout=60)
    if pdt["ok"]:
        try:
            report["dep_tree_size"] = len(json.loads(pdt["stdout"]))
        except Exception:
            report["dep_tree_size"] = -1
    else:
        report["dep_tree_size"] = -1

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
    _emit({"type": "report", **report})
    log.info("[integrity] scan complete (%d events)", len(report["events"]))


def start_background_scan() -> None:
    t = threading.Thread(target=_worker, name="MEC-integrity", daemon=True)
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
        result = _run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", package],
            timeout=600,
        )
        # Rescan after the change.
        start_background_scan()
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
        if trigger_rescan or not _LAST_REPORT.get("ready"):
            start_background_scan()
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
