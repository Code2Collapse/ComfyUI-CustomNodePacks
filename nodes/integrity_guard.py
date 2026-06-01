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
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("MEC.integrity")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK_ROOT = os.path.dirname(_HERE)
_CHECKSUM_PATH = os.path.join(_PACK_ROOT, "third_party", "checksums.json")


def _pick_cache_path() -> str:
    """Choose a writable location for the integrity cache.

    Preference order:
      1. <pack_root>/.cache/integrity.json   (next to the code, normal case)
      2. <user_data_dir>/MEC/integrity_cache.json  (Comfy user dir)
      3. <tempdir>/MEC_integrity_cache.json  (last resort)

    Some installs (Manager-managed, read-only mounts, system-wide installs)
    cannot write inside the pack directory. Falling back keeps integrity
    checks working on those systems instead of repeatedly re-running the
    scan because the cache write silently failed.
    """
    candidates = [os.path.join(_PACK_ROOT, ".cache", "integrity.json")]
    try:
        import folder_paths  # type: ignore
        user_dir = getattr(folder_paths, "get_user_directory", lambda: None)()
        if user_dir:
            candidates.append(os.path.join(user_dir, "MEC", "integrity_cache.json"))
    except Exception:
        pass
    import tempfile
    candidates.append(os.path.join(tempfile.gettempdir(), "MEC_integrity_cache.json"))
    # Legacy location — read-only fallback so existing caches still load.
    legacy = os.path.join(_PACK_ROOT, ".integrity_cache.json")
    for path in candidates:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Probe writability without clobbering existing data.
            probe = path + ".probe"
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            return path
        except OSError:
            continue
    # Everything failed — return the first candidate, _save_cache will log it.
    return legacy


_CACHE_PATH = _pick_cache_path()
_LEGACY_CACHE_PATH = os.path.join(_PACK_ROOT, ".integrity_cache.json")

# Cache TTL: how long a successful scan stays valid before re-running.
_CACHE_TTL_SECONDS = 24 * 3600  # 24h

# Defer the initial background scan so ComfyUI + Manager finish their own
# startup work first. Tuned down from 20s — users open the diagnostics
# panel quickly and a stale "scan running…" frustrates them more than the
# tiny chance of competing with Manager's startup check.
_STARTUP_DELAY_SECONDS = 8

# Locate `uv` once at import time. uv is ~5–20x faster than `python -m pip`
# at metadata-only operations (pip check, pip tree). Falls back to pip if
# uv isn't installed.
_UV_BIN: Optional[str] = shutil.which("uv")

_LOCK = threading.Lock()
_LAST_REPORT: Dict[str, Any] = {"ready": False, "status": "pending"}


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
    # On Windows, prevent a console window from flashing when the parent
    # ComfyUI process is a GUI launcher (pythonw.exe, embedded Python
    # spawned by Comfy-Desktop, etc.). On other OSes this flag is 0.
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout, check=False,
            # Force UTF-8 with replace so non-ASCII package names / paths
            # don't crash decoding on Windows (default cp1252) or on
            # systems with a broken locale (containers, minimal images).
            text=True, encoding="utf-8", errors="replace",
            creationflags=creationflags,
        )
        return {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "stdout": proc.stdout if stdout_limit is None
                      else (proc.stdout or "").strip()[:stdout_limit],
            "stderr": proc.stderr if stderr_limit is None
                      else (proc.stderr or "").strip()[:stderr_limit],
        }
    except FileNotFoundError as e:
        return {"ok": False, "rc": -1, "stdout": "", "stderr": str(e)}
    except PermissionError as e:
        # Sandboxed installs (some Docker / Snap layouts) reject exec.
        return {"ok": False, "rc": -4, "stdout": "", "stderr": f"permission denied: {e}"}
    except OSError as e:
        return {"ok": False, "rc": -5, "stdout": "", "stderr": f"os error: {e}"}
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


# Suspicious-pattern signatures for the lightweight custom_nodes virus
# heuristic. Kept in sync with c2c_doctor._SUSPICIOUS_PY (single source
# of truth is c2c_doctor when it's importable; otherwise this fallback
# list is used).
_SUSP_PATTERNS_FALLBACK = [
    (re.compile(r"\beval\s*\(", re.IGNORECASE),                "eval()"),
    (re.compile(r"\bexec\s*\(", re.IGNORECASE),                "exec()"),
    (re.compile(r"base64\.b64decode\s*\(", re.IGNORECASE),     "base64.b64decode()"),
    (re.compile(r"urllib\.request\.urlopen\(\s*['\"]http",     re.IGNORECASE), "remote URL fetch"),
    (re.compile(r"subprocess\.[A-Za-z_]+\([^)]*shell\s*=\s*True", re.IGNORECASE), "shell=True"),
    (re.compile(r"os\.system\s*\(", re.IGNORECASE),            "os.system()"),
    (re.compile(r"compile\s*\(\s*['\"][^'\"]+['\"]\s*,",       re.IGNORECASE), "dynamic compile()"),
    (re.compile(r"marshal\.loads\s*\(", re.IGNORECASE),        "marshal.loads()"),
]

def _susp_patterns():
    try:
        from . import c2c_doctor as _cd
        if getattr(_cd, "_SUSPICIOUS_PY", None):
            return _cd._SUSPICIOUS_PY
    except Exception:
        pass
    return _SUSP_PATTERNS_FALLBACK

def _custom_nodes_root() -> Optional[str]:
    """Locate the ComfyUI/custom_nodes/ folder by walking up from this file."""
    p = os.path.dirname(_PACK_ROOT)
    # _PACK_ROOT is .../ComfyUI/custom_nodes/<this-pack>/nodes/integrity_guard.py
    # so dirname(_PACK_ROOT) is .../ComfyUI/custom_nodes
    if os.path.basename(p).lower() == "custom_nodes" and os.path.isdir(p):
        return p
    return None

def _scan_suspicious_custom_nodes(max_files: int = 4000,
                                  max_bytes_per_file: int = 256 * 1024) -> List[Dict[str, Any]]:
    """Scan every .py under ComfyUI/custom_nodes/ (except OUR pack) for known
    high-risk patterns. Returns one event per offending file (capped). Each
    event has severity=warn so it surfaces under Doctor → Package integrity.
    """
    findings: List[Dict[str, Any]] = []
    root = _custom_nodes_root()
    if not root:
        return findings
    patterns = _susp_patterns()
    seen = 0
    own_root = os.path.abspath(_PACK_ROOT)
    for dirpath, _dirs, files in os.walk(root):
        if any(seg in dirpath for seg in (".git", "__pycache__", "node_modules",
                                          ".pytest_cache", ".ruff_cache")):
            continue
        # Skip our own pack (we trust ourselves; avoid noise).
        try:
            if os.path.commonpath([os.path.abspath(dirpath), own_root]) == own_root:
                continue
        except Exception:
            pass
        for fn in files:
            if not fn.endswith(".py"):
                continue
            seen += 1
            if seen > max_files:
                return findings
            full = os.path.join(dirpath, fn)
            try:
                with open(full, "rb") as fh:
                    data = fh.read(max_bytes_per_file)
            except Exception:
                continue
            hits: List[str] = []
            for rx, name in patterns:
                try:
                    if rx.search(data.decode("utf-8", errors="replace")):
                        hits.append(name)
                except Exception:
                    continue
            if hits:
                rel = os.path.relpath(full, root).replace("\\", "/")
                findings.append({
                    "kind": "suspicious_pattern",
                    "severity": "warn",
                    "message": f"{rel}: {', '.join(hits[:6])}",
                    "file": rel,
                    "patterns": hits[:6],
                })
    return findings



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
    paths = [_CACHE_PATH]
    if _LEGACY_CACHE_PATH != _CACHE_PATH:
        paths.append(_LEGACY_CACHE_PATH)
    for path in paths:
        try:
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts = float(data.get("_cached_at", 0))
            if time.time() - ts > _CACHE_TTL_SECONDS:
                continue
            if data.get("_fingerprint") != _env_fingerprint():
                continue
            return {k: v for k, v in data.items()
                    if k not in ("_cached_at", "_fingerprint")}
        except (OSError, json.JSONDecodeError, ValueError) as e:
            log.debug("[integrity] cache load failed (%s): %s", path, e)
    return None


def _save_cache(report: Dict[str, Any]) -> None:
    try:
        payload = dict(report)
        payload["_cached_at"] = time.time()
        payload["_fingerprint"] = _env_fingerprint()
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        tmp = _CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, _CACHE_PATH)
    except OSError as e:
        log.warning("[integrity] cache save failed (%s): %s", _CACHE_PATH, e)


def _pip_module_available() -> bool:
    """Cheap probe: does `python -m pip --version` succeed?

    Embedded Python builds (portable ComfyUI on some systems) ship without
    pip. We need to know that BEFORE running `pip check`, otherwise the
    "no module named pip" stderr gets mis-categorised as a dependency
    conflict and the user sees no useful events.

    Cached for the lifetime of the process so we don't pay the ~50–200 ms
    interpreter spin-up on every rescan.
    """
    global _PIP_AVAILABLE_CACHE
    if _PIP_AVAILABLE_CACHE is not None:
        return _PIP_AVAILABLE_CACHE
    out = _run([sys.executable, "-m", "pip", "--version"], timeout=15)
    _PIP_AVAILABLE_CACHE = bool(out["ok"])
    return _PIP_AVAILABLE_CACHE


_PIP_AVAILABLE_CACHE: Optional[bool] = None


def _detect_env_kind() -> str:
    """Best-effort classification of the current Python environment.

    Returned strings are stable and used by the UI to tailor install hints
    (e.g. "conda env: install uv via `conda install -c conda-forge uv`").
    """
    exe = (sys.executable or "").replace("\\", "/").lower()
    # Conda: presence of CONDA_PREFIX env var OR "conda-meta" beside the exe.
    if os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV"):
        return "conda"
    conda_meta = os.path.join(os.path.dirname(sys.executable), "..", "conda-meta")
    if os.path.isdir(conda_meta):
        return "conda"
    # ComfyUI Windows portable bundles ship as python_embeded / python_embedded.
    if "python_embeded" in exe or "python_embedded" in exe:
        return "portable-embed"
    # venv / virtualenv: sys.prefix differs from sys.base_prefix.
    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        return "venv"
    if hasattr(sys, "real_prefix"):  # legacy virtualenv
        return "venv"
    # uv-managed projects expose UV_PROJECT_ENVIRONMENT pointing at .venv.
    if os.environ.get("UV_PROJECT_ENVIRONMENT") or os.environ.get("VIRTUAL_ENV"):
        return "venv"
    return "system"


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
        out["backend"] = "uv"
        return out
    if not _pip_module_available():
        return {
            "ok": False, "rc": -3, "stdout": "", "backend": "none",
            "stderr": (
                "pip is not available in this Python interpreter "
                f"({sys.executable}). Install pip (`python -m ensurepip`) "
                "or install `uv` (https://docs.astral.sh/uv/) so MEC can "
                "verify dependency consistency."
            ),
            "unavailable": True,
        }
    out = _run([sys.executable, "-m", "pip", "check"], timeout=120,
               stdout_limit=16000)
    out["backend"] = "pip"
    return out


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
    try:
        _worker_impl(force=force, delay=delay)
    except Exception as e:  # noqa: BLE001
        # The scan thread must never die silently — the UI would stay
        # stuck on "scan running…" forever. Surface the failure as an
        # event and a non-fatal report so the user can see what broke.
        log.exception("[integrity] worker crashed")
        err = {
            "ready": True, "status": "error", "from_cache": False,
            "events": [{"kind": "scan_error", "severity": "warn",
                         "message": f"{type(e).__name__}: {e}"}],
            "scanned_at": time.time(),
            "python": sys.executable,
            "pack_root": _PACK_ROOT,
        }
        with _LOCK:
            _LAST_REPORT.clear()
            _LAST_REPORT.update(err)
        _emit({"type": "report", **err})


def _worker_impl(force: bool = False, delay: float = 0.0):
    if delay > 0:
        time.sleep(delay)

    # Re-detect uv on force-rescan so users who just installed it get the
    # benefit without restarting ComfyUI.
    global _UV_BIN
    if force:
        _UV_BIN = shutil.which("uv")

    # ---- cache fast-path ----
    if not force:
        cached = _load_cache()
        if cached is not None:
            cached["from_cache"] = True
            cached.setdefault("ready", True)
            cached.setdefault("status", "ok")
            with _LOCK:
                _LAST_REPORT.clear()
                _LAST_REPORT.update(cached)
            _emit({"type": "report", **cached})
            log.info("[integrity] using cached report (%d events)",
                     len(cached.get("events", [])))
            return

    # Announce scan-in-progress so the UI can stop showing "looks clean"
    # before the first scan has actually run.
    scanning = {"ready": False, "status": "scanning", "events": [],
                "from_cache": False, "used_uv": bool(_UV_BIN)}
    with _LOCK:
        _LAST_REPORT.clear()
        _LAST_REPORT.update(scanning)
    _emit({"type": "scanning", **scanning})

    report: Dict[str, Any] = {
        "ready": True, "status": "ok", "events": [],
        "from_cache": False, "scanned_at": time.time(),
        "python": sys.executable, "python_version": sys.version.split()[0],
        "pack_root": _PACK_ROOT, "cache_path": _CACHE_PATH,
        "env_kind": _detect_env_kind(),
        "platform": sys.platform,
    }

    # ---- pip check (uv if available, else pip; or report unavailable) ----
    pip_check = _run_pip_check()
    report["pip_check"] = pip_check
    report["used_uv"] = bool(_UV_BIN)
    report["backend"] = pip_check.get("backend", "unknown")
    if pip_check.get("unavailable"):
        # Neither uv nor pip available — flag clearly with an env-specific
        # remediation hint so the user knows exactly what to run.
        kind = report.get("env_kind", "system")
        hint = {
            "portable-embed": (
                "ComfyUI portable embed lacks pip. Either drop a "
                "`get-pip.py` next to python.exe and run "
                "`python_embeded\\python.exe get-pip.py`, or install uv: "
                "`powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"`."
            ),
            "conda": (
                "Conda env without pip. Run `conda install pip` inside this "
                "env, or `conda install -c conda-forge uv`."
            ),
            "venv": (
                "venv missing pip. Recreate with `python -m venv --upgrade-deps "
                f"{sys.prefix}` or install uv globally."
            ),
            "system": (
                "Install pip via `python -m ensurepip --upgrade`, or "
                "install uv: https://docs.astral.sh/uv/getting-started/installation/"
            ),
        }.get(kind, "Install pip or uv so MEC can verify dependency consistency.")
        report["events"].append({
            "kind": "tooling_unavailable",
            "severity": "info",
            "message": (pip_check.get("stderr") or "pip/uv unavailable")
                       + f"\nHint: {hint}",
        })
    elif not pip_check["ok"]:
        lines = [ln.strip() for ln in (pip_check.get("stdout") or "").splitlines() if ln.strip()]
        if lines:
            for line in lines:
                report["events"].append({
                    "kind": "dependency_conflict",
                    "severity": "warn",
                    "message": line,
                })
        else:
            # pip check failed but produced no parseable output — surface stderr
            # so the user can see WHY (otherwise the UI shows "events=0" forever).
            stderr = (pip_check.get("stderr") or "").strip()
            if stderr:
                report["events"].append({
                    "kind": "pip_check_error",
                    "severity": "warn",
                    "message": stderr.splitlines()[0][:400],
                })

    # ---- dep tree size (uv only; skipped otherwise) ----
    report["dep_tree_size"] = _run_dep_tree_size()

    # ---- checksum verification ----
    expected = {}
    if os.path.isfile(_CHECKSUM_PATH):
        try:
            with open(_CHECKSUM_PATH, "r", encoding="utf-8") as f:
                expected = json.load(f)
        except Exception as e:
            log.warning("[integrity] cannot read %s: %s", _CHECKSUM_PATH, e)
            report["events"].append({
                "kind": "checksum_baseline_unreadable",
                "severity": "info",
                "message": f"checksums.json failed to parse: {e}",
            })
    else:
        # No baseline shipped — drift detection is dormant. Tell the user
        # rather than silently producing zero drift events.
        report["checksum_baseline"] = "missing"
        report["events"].append({
            "kind": "checksum_baseline_missing",
            "severity": "info",
            "message": "no third_party/checksums.json baseline present — file drift checks skipped",
        })
    drift = []
    if expected:
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

    # ---- suspicious-pattern scan across ALL custom_nodes (lightweight virus heuristic) ----
    try:
        susp = _scan_suspicious_custom_nodes()
        report["suspicious_files"] = len(susp)
        # Only push the first 50 into events to keep the report bounded.
        for ev in susp[:50]:
            report["events"].append(ev)
    except Exception as e:
        log.warning("[integrity] suspicious scan failed: %s", e)
        report["suspicious_files"] = 0

    with _LOCK:
        _LAST_REPORT.clear()
        _LAST_REPORT.update(report)
    _save_cache(report)
    _emit({"type": "report", **report})
    log.info("[integrity] scan complete (%d events, backend=%s)",
             len(report["events"]), report["backend"])


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
# Phase E.2 — Safety-belt uv installer for failed-import auto-heal
# =====================================================================
# Distribution-name regex: matches PEP 503 normalisation plus version
# specifiers we accept on input (no shell metacharacters).
_SAFE_DIST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")

# Critical packages we refuse to upgrade/downgrade under any heal path,
# even with force=True. Mirrors dependency_checker.CRITICAL_PACKAGES;
# duplicated here so this module stays import-safe even if
# dependency_checker is unavailable.
_HEAL_CRITICAL = frozenset({
    "torch", "torchvision", "torchaudio", "numpy", "transformers",
    "diffusers", "xformers", "accelerate", "timm", "safetensors",
})

# Hard cap on heal subprocess time. Anything longer almost certainly
# means uv is pulling a torch wheel — abort + roll back per plan.md E.4.
_HEAL_INSTALL_TIMEOUT = 120
_HEAL_CHECK_TIMEOUT = 30


def _heal_validate_packages(packages: List[str]) -> Tuple[List[str], Optional[str]]:
    """Reject anything that's a CRITICAL package or has unsafe characters.

    Returns ``(accepted, refusal_reason_or_None)``. If any package is
    rejected the entire batch is refused — partial install is never safe.
    """
    accepted: List[str] = []
    for pkg in packages:
        if not isinstance(pkg, str):
            return ([], f"non-string package entry: {pkg!r}")
        # Strip version specifier for the critical-package check.
        bare = re.split(r"[\[<>=!~]", pkg, maxsplit=1)[0].strip().lower()
        if not bare:
            return ([], f"empty package name in {pkg!r}")
        if not _SAFE_DIST_RE.match(pkg):
            return ([], f"unsafe characters in package name: {pkg!r}")
        if bare in _HEAL_CRITICAL:
            return ([], f"refusing to touch CRITICAL package: {bare}")
        accepted.append(pkg)
    return (accepted, None)


def install_missing(
    packages: List[str],
    *,
    force: bool = False,
    timeout: int = _HEAL_INSTALL_TIMEOUT,
) -> Dict[str, Any]:
    """Install ``packages`` using uv with the auto-heal safety belt.

    Refuses (`ok=False`) when:
      * ``_UV_BIN`` is None (uv not installed),
      * any package is in ``_HEAL_CRITICAL`` (even with ``force=True``),
      * any package name contains unsafe characters,
      * post-install ``uv pip check`` reports NEW conflicts (auto-rolls back).

    Returns a dict with keys:
      ``ok`` (bool), ``installed`` (list[str]), ``rolled_back`` (bool),
      ``stage`` (str: "validate" | "install" | "post_check" | "done"),
      ``stdout``, ``stderr``, ``rc``, ``duration_s``, ``error`` (optional).

    The ``force`` flag currently affects only the user-facing "risky"
    bucket discrimination in the caller (E.3); CRITICAL packages remain
    refused unconditionally per the G1 guardrail.
    """
    t0 = time.time()
    if not packages:
        return {"ok": False, "stage": "validate", "error": "no packages requested",
                "installed": [], "rolled_back": False, "rc": -1,
                "stdout": "", "stderr": "", "duration_s": 0.0}

    if _UV_BIN is None:
        return {"ok": False, "stage": "validate",
                "error": "uv is not installed — auto-heal requires uv. "
                         "Install via Doctor → 'Install uv' or "
                         "https://docs.astral.sh/uv/",
                "installed": [], "rolled_back": False, "rc": -1,
                "stdout": "", "stderr": "", "duration_s": 0.0}

    accepted, refusal = _heal_validate_packages(packages)
    if refusal is not None or not accepted:
        return {"ok": False, "stage": "validate",
                "error": refusal or "no installable packages after validation",
                "installed": [], "rolled_back": False, "rc": -1,
                "stdout": "", "stderr": "", "duration_s": round(time.time() - t0, 2)}

    # Snapshot pre-install conflicts so we know what's "new" later.
    # We compare the set of "requires X" lines — the trailing
    # "Checked N packages in Xms" line will differ after install
    # (153 -> 154 etc.), so a raw string compare is too strict.
    def _conflict_lines(text: str) -> frozenset:
        lines = set()
        for ln in (text or "").splitlines():
            ln = ln.strip()
            if ln.startswith("The package ") and " requires " in ln:
                lines.add(ln)
        return frozenset(lines)

    pre_check = _run([_UV_BIN, "pip", "check", "--python", sys.executable],
                     timeout=_HEAL_CHECK_TIMEOUT)
    pre_text = (pre_check.get("stderr") or "") + (pre_check.get("stdout") or "")
    pre_conflicts = _conflict_lines(pre_text)

    # Run a single uv invocation — one resolver run, one cache scan.
    install_cmd = [
        _UV_BIN, "pip", "install",
        "--python", sys.executable,
        "--link-mode", "copy",   # safer across drives than hardlink
        # NOTE: no --upgrade, no --reinstall, no --no-cache-dir.
        # uv's strict resolver is enabled by default; --strict alters
        # script-vs-module handling on some uv versions, so we omit it
        # and rely on the post-check + rollback to detect conflicts.
        *accepted,
    ]
    install = _run(install_cmd, timeout=timeout)
    if not install["ok"]:
        return {"ok": False, "stage": "install",
                "error": f"uv pip install failed (rc={install['rc']})",
                "installed": [], "rolled_back": False,
                "rc": install["rc"],
                "stdout": install["stdout"], "stderr": install["stderr"],
                "duration_s": round(time.time() - t0, 2)}

    # Post-install conflict check. If uv reports NEW problems, roll back.
    post_check = _run([_UV_BIN, "pip", "check", "--python", sys.executable],
                      timeout=_HEAL_CHECK_TIMEOUT)
    post_text = (post_check.get("stderr") or "") + (post_check.get("stdout") or "")
    post_conflicts = _conflict_lines(post_text)
    new_conflicts = post_conflicts - pre_conflicts
    if new_conflicts:
        # Auto-rollback: uninstall the packages we just added.
        uninstall = _run(
            [_UV_BIN, "pip", "uninstall", "--python", sys.executable, *accepted],
            timeout=timeout,
        )
        return {"ok": False, "stage": "post_check",
                "error": "post-install uv pip check reported new conflicts; "
                         "rolled back the newly installed packages",
                "installed": [], "rolled_back": uninstall["ok"],
                "rc": post_check["rc"],
                "stdout": post_check["stdout"], "stderr": post_check["stderr"],
                "new_conflicts": sorted(new_conflicts),
                "duration_s": round(time.time() - t0, 2)}

    # Success. Re-emit an integrity event so the UI refreshes its INT pill.
    _emit({"type": "heal_complete", "packages": accepted})

    return {"ok": True, "stage": "done", "installed": list(accepted),
            "rolled_back": False, "rc": install["rc"],
            "stdout": install["stdout"], "stderr": install["stderr"],
            "duration_s": round(time.time() - t0, 2)}


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

    # -------------------------------------------------------------- E.3
    # Phase E.3 — Failed-import discovery + heal routes
    @routes.get("/c2c/integrity/failed_imports")
    async def _failed_imports(req):  # noqa: ARG001
        try:
            from ..c2c_ai.import_heal import collect_failed_imports
        except Exception as exc:
            return web.json_response(
                {"success": False, "error": "import_heal_unavailable",
                 "message": str(exc)}, status=500)
        rescan = (req.query.get("rescan", "") or "").lower() in ("1", "true", "yes")
        try:
            packs = collect_failed_imports(rescan=rescan)
        except Exception as exc:
            log.exception("[integrity] failed_imports harvester crashed")
            return web.json_response(
                {"success": False, "error": "harvester_failed",
                 "message": str(exc)}, status=500)
        data = {
            "packs": [p.to_dict() for p in packs],
            "counts": {
                "total": len(packs),
                "auto_safe": sum(1 for p in packs
                                 if p.recommended_action == "auto_safe"),
                "needs_review": sum(1 for p in packs
                                    if p.recommended_action == "needs_review"),
                "blocked": sum(1 for p in packs
                               if p.recommended_action == "blocked"),
                "unknown": sum(1 for p in packs
                               if p.recommended_action == "unknown"),
            },
            "uv_available": bool(_UV_BIN),
        }
        return web.json_response({"success": True, "data": data})

    @routes.post("/c2c/integrity/heal")
    async def _heal(req):
        try:
            body = await req.json()
        except Exception:
            body = {}
        pack_name = (body or {}).get("pack")
        force = bool((body or {}).get("force", False))
        if not pack_name or not isinstance(pack_name, str):
            return web.json_response(
                {"success": False, "error": "missing_pack",
                 "message": "body must include 'pack': <pack name>"},
                status=400)
        try:
            from ..c2c_ai.import_heal import (
                collect_failed_imports,
                ACTION_AUTO_SAFE, ACTION_NEEDS_REVIEW, ACTION_BLOCKED,
            )
        except Exception as exc:
            return web.json_response(
                {"success": False, "error": "import_heal_unavailable",
                 "message": str(exc)}, status=500)
        packs = collect_failed_imports(rescan=False)
        pack = next((p for p in packs if p.name == pack_name), None)
        if pack is None:
            return web.json_response(
                {"success": False, "error": "pack_not_failed",
                 "message": f"no failed-import pack named {pack_name!r}"},
                status=404)
        if pack.recommended_action == ACTION_BLOCKED:
            return web.json_response(
                {"success": False, "error": "blocked",
                 "message": "this pack requires manual review — "
                            "a CRITICAL package would be modified",
                 "report": pack.report}, status=409)
        if pack.recommended_action == ACTION_NEEDS_REVIEW and not force:
            return web.json_response(
                {"success": False, "error": "needs_review",
                 "message": "this pack has risky entries — "
                            "POST with 'force': true to install anyway",
                 "report": pack.report}, status=409)
        result = install_missing(pack.safe_to_install, force=force)
        # On success, rescan integrity so the INT pill clears the warning.
        if result.get("ok"):
            start_background_scan(force=True, delay=0.0)
        return web.json_response({
            "success": bool(result.get("ok")),
            "data": {"pack": pack_name, "result": result},
        })

    log.info("[integrity] routes registered")


# =====================================================================
# Node
# =====================================================================
class IntegrityStatusMEC:
    DESCRIPTION = "Returns the latest integrity scan as a string."
    CATEGORY = "C2C/Diagnostics"
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
NODE_DISPLAY_NAME_MAPPINGS = {"IntegrityStatusMEC": "Integrity Status"}
