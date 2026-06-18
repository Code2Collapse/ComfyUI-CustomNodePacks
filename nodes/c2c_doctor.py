"""c2c_doctor.py - backend for the 13-tab Doctor mega-panel (P0.5).

Provides three live HTTP endpoints used by js/c2c_doctor.js:

  * GET  /c2c/doctor/pyenv      -> python+comfy+pkg versions for the env
  * GET  /c2c/doctor/disk?refresh=1 -> on-disk sizes for ComfyUI/models,
                                       ComfyUI/user, ComfyUI/custom_nodes
                                       (cached 60s; pass refresh=1 to bust)
  * POST /c2c/doctor/scan_file  -> multipart upload, returns integrity
                                   report for a single file (.json / .png /
                                   .safetensors / .ckpt / .py / .zip).

No stubs. Every value is computed from the actually-running environment
or from disk. Errors are returned as HTTP 200 with success=false so the JS
can render them rather than choke on fetch failures.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as _md
import io
import json
import os
import platform
import re
import shutil
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Disk cache - walking ComfyUI/models can take many seconds; cache for 60s.
# ---------------------------------------------------------------------------
_DISK_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_DISK_TTL_S = 60.0


def _comfy_root() -> Optional[Path]:
    """Return the ComfyUI root directory by walking up from folder_paths."""
    try:
        import folder_paths  # type: ignore
        # base_path is set by ComfyUI to the dir containing main.py
        bp = getattr(folder_paths, "base_path", None)
        if bp:
            return Path(bp)
    except Exception:
        pass
    # fallback: walk up looking for main.py
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "main.py").is_file() and (parent / "comfy").is_dir():
            return parent
    return None


def _dir_size_fast(p: Path) -> Tuple[int, int]:
    """Return (total_bytes, file_count). Skips symlinks (we have junctions)."""
    total = 0
    count = 0
    try:
        for root, dirs, files in os.walk(p, followlinks=False):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    st = os.stat(fp, follow_symlinks=False)
                    total += st.st_size
                    count += 1
                except OSError:
                    continue
    except Exception:
        pass
    return total, count


def _model_subdir_breakdown(models_root: Path) -> List[Dict[str, Any]]:
    """Return per-subdir (checkpoints/loras/vae/...) size+count."""
    out: List[Dict[str, Any]] = []
    if not models_root.is_dir():
        return out
    try:
        children = sorted([p for p in models_root.iterdir() if p.is_dir()],
                          key=lambda p: p.name.lower())
    except Exception:
        return out
    for sub in children:
        size, files = _dir_size_fast(sub)
        out.append({"name": sub.name, "path": str(sub), "bytes": size, "files": files})
    return out


def collect_disk(refresh: bool = False) -> Dict[str, Any]:
    """Return disk usage snapshot. Cached for _DISK_TTL_S seconds."""
    now = time.time()
    if (not refresh) and _DISK_CACHE["data"] is not None and (now - _DISK_CACHE["ts"]) < _DISK_TTL_S:
        d = dict(_DISK_CACHE["data"])
        d["cached"] = True
        d["age_s"] = round(now - _DISK_CACHE["ts"], 1)
        return d

    root = _comfy_root()
    out: Dict[str, Any] = {"success": True, "cached": False, "ts": now}
    if root is None:
        out["success"] = False
        out["error"] = "comfy_root_not_found"
        return out

    out["root"] = str(root)
    # Free / total on the drive that hosts ComfyUI root
    try:
        du = shutil.disk_usage(root)
        out["drive"] = {"total": du.total, "used": du.used, "free": du.free}
    except Exception as exc:
        out["drive"] = {"error": str(exc)}

    sections = {}
    for key, sub in (("models", "models"),
                     ("user", "user"),
                     ("custom_nodes", "custom_nodes"),
                     ("temp", "temp"),
                     ("output", "output"),
                     ("input", "input")):
        p = root / sub
        if p.is_dir():
            size, count = _dir_size_fast(p)
            sections[key] = {"path": str(p), "bytes": size, "files": count}
        else:
            sections[key] = {"path": str(p), "missing": True}
    out["sections"] = sections

    # Per-subdir breakdown for models (checkpoints, loras, vae, ...)
    out["models_breakdown"] = _model_subdir_breakdown(root / "models")

    _DISK_CACHE["data"] = out
    _DISK_CACHE["ts"] = now
    return out


# ---------------------------------------------------------------------------
# Python env snapshot
# ---------------------------------------------------------------------------
# Packages we always try to surface (if installed). Order = display order.
_RELEVANT_PKGS = [
    "torch", "torchvision", "torchaudio", "xformers",
    "numpy", "scipy", "pillow", "opencv-python",
    "diffusers", "transformers", "tokenizers", "accelerate",
    "safetensors", "sentencepiece", "huggingface-hub",
    "aiohttp", "yarl", "uvicorn", "fastapi",
    "onnxruntime", "onnxruntime-gpu", "onnx",
    "scikit-image", "scikit-learn",
    "gitpython", "psutil", "pyyaml", "tqdm",
    "ftfy", "cryptography", "keyring",
    "comfyui-frontend-package", "comfyui-workflow-templates",
    "comfyui-embedded-docs",
]


def _pkg_version(name: str) -> Optional[str]:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None
    except Exception:
        return None


def collect_pyenv() -> Dict[str, Any]:
    out: Dict[str, Any] = {"success": True, "ts": time.time()}
    out["python"] = {
        "version": sys.version.split()[0],
        "full": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    # ComfyUI version (best-effort)
    cv: Dict[str, Any] = {}
    try:
        import comfy  # type: ignore
        cv["module_path"] = getattr(comfy, "__file__", None)
    except Exception:
        pass
    try:
        import comfyui_version  # type: ignore
        cv["comfyui_version"] = getattr(comfyui_version, "__version__", None)
    except Exception:
        # try the pyproject embedded version
        root = _comfy_root()
        if root:
            pj = root / "pyproject.toml"
            if pj.is_file():
                try:
                    txt = pj.read_text(encoding="utf-8", errors="ignore")
                    m = re.search(r'^version\s*=\s*"([^"]+)"', txt, re.MULTILINE)
                    if m:
                        cv["comfyui_version"] = m.group(1)
                except Exception:
                    pass
    out["comfy"] = cv

    # CUDA / device snapshot
    dev: Dict[str, Any] = {}
    try:
        import torch  # type: ignore
        dev["torch"] = torch.__version__
        dev["cuda_available"] = bool(torch.cuda.is_available())
        if dev["cuda_available"]:
            dev["cuda"] = torch.version.cuda
            dev["cudnn"] = torch.backends.cudnn.version()
            dev["device_count"] = torch.cuda.device_count()
            dev["devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                dev["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_mem": int(props.total_memory),
                    "capability": f"{props.major}.{props.minor}",
                })
    except Exception as exc:
        dev["error"] = str(exc)
    out["device"] = dev

    # Package versions (only ones actually installed; report missing too)
    pkgs: List[Dict[str, Any]] = []
    for name in _RELEVANT_PKGS:
        v = _pkg_version(name)
        pkgs.append({"name": name, "version": v, "installed": v is not None})
    out["packages"] = pkgs

    return out


# ---------------------------------------------------------------------------
# File integrity scan
# ---------------------------------------------------------------------------
# Suspicious patterns in .py / __init__.py within a zip - keep tight to
# avoid false positives. These are *informational*, not verdicts.
_SUSPICIOUS_PY = [
    (re.compile(r"\beval\s*\(", re.IGNORECASE), "eval()"),
    (re.compile(r"\bexec\s*\(", re.IGNORECASE), "exec()"),
    (re.compile(r"subprocess\.(call|Popen|run)", re.IGNORECASE), "subprocess spawn"),
    (re.compile(r"\bos\.system\s*\(", re.IGNORECASE), "os.system()"),
    (re.compile(r"urllib\.request\.urlopen|requests\.(get|post)", re.IGNORECASE), "outbound HTTP"),
    (re.compile(r"socket\.socket\s*\(", re.IGNORECASE), "raw socket"),
    (re.compile(r"base64\.b64decode\s*\(", re.IGNORECASE), "base64 decode"),
    (re.compile(r"compile\s*\(\s*['\"]", re.IGNORECASE), "compile() of literal"),
    (re.compile(r"shutil\.rmtree\s*\(\s*['\"]/", re.IGNORECASE), "rmtree of root path"),
    (re.compile(r"pickle\.loads?\s*\(", re.IGNORECASE), "pickle.load (RCE risk)"),
]


def _scan_python_bytes(data: bytes, label: str) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return findings
    for rx, name in _SUSPICIOUS_PY:
        if rx.search(text):
            findings.append({"kind": name, "where": label})
    return findings


def _safetensors_header_ok(data: bytes) -> Tuple[bool, Dict[str, Any]]:
    """Validate the 8-byte length + JSON header of a safetensors file."""
    info: Dict[str, Any] = {}
    if len(data) < 8:
        return False, {"reason": "too_short"}
    n = struct.unpack("<Q", data[:8])[0]
    if n <= 0 or n > 100 * 1024 * 1024:
        return False, {"reason": "bad_header_len", "len": int(n)}
    if 8 + n > len(data):
        return False, {"reason": "header_overruns_file",
                       "header_len": int(n), "file_len": len(data)}
    try:
        hdr = json.loads(data[8:8 + n].decode("utf-8"))
    except Exception as exc:
        return False, {"reason": "header_not_json", "detail": str(exc)[:200]}
    info["tensor_count"] = sum(1 for k in hdr.keys() if k != "__metadata__")
    if "__metadata__" in hdr and isinstance(hdr["__metadata__"], dict):
        info["metadata_keys"] = list(hdr["__metadata__"].keys())[:32]
    return True, info


def _scan_zip_or_pack(data: bytes, label: str) -> Dict[str, Any]:
    """Inspect a zip or .ckpt (pickle tar) file. Returns a report dict."""
    out: Dict[str, Any] = {"format": "zip", "ok": True, "issues": []}
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        out["ok"] = False
        out["issues"].append({"severity": "error",
                              "kind": "not_a_zip",
                              "where": label})
        return out
    py_findings: List[Dict[str, str]] = []
    suspicious_names: List[str] = []
    for zi in zf.infolist():
        nm = zi.filename
        if nm.endswith((".pyc", ".pyo")):
            suspicious_names.append(nm)
        if nm.endswith((".py", ".pyi")) and zi.file_size < 2_000_000:
            try:
                with zf.open(zi) as fh:
                    py_findings.extend(_scan_python_bytes(fh.read(), nm))
            except Exception:
                continue
    out["entries"] = len(zf.infolist())
    if suspicious_names:
        out["issues"].append({"severity": "info",
                              "kind": "precompiled_bytecode",
                              "examples": suspicious_names[:5]})
    for f in py_findings[:50]:
        out["issues"].append({"severity": "info", **f})
    return out


def scan_file(filename: str, data: bytes) -> Dict[str, Any]:
    """Return an integrity/safety report for a single uploaded file."""
    out: Dict[str, Any] = {
        "success": True,
        "filename": filename,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "issues": [],
    }
    name_lc = filename.lower()
    ext = Path(filename).suffix.lower()

    # JSON
    if ext in (".json", ".geninfo") or (data[:1] in (b"{", b"[")):
        out["format"] = "json"
        try:
            j = json.loads(data.decode("utf-8", errors="strict"))
            out["json_root_type"] = type(j).__name__
            if isinstance(j, dict):
                out["json_keys"] = list(j.keys())[:32]
                # ComfyUI workflow detection
                if "nodes" in j and isinstance(j["nodes"], list):
                    out["workflow_node_count"] = len(j["nodes"])
                if "last_node_id" in j and "last_link_id" in j:
                    out["workflow_format"] = "litegraph"
        except UnicodeDecodeError as exc:
            out["issues"].append({"severity": "error",
                                  "kind": "not_utf8",
                                  "detail": str(exc)[:200]})
            out["success"] = False
        except json.JSONDecodeError as exc:
            out["issues"].append({"severity": "error",
                                  "kind": "json_parse_error",
                                  "detail": str(exc)[:200]})
            out["success"] = False
        return out

    # PNG - check signature + iTXt/tEXt for workflow metadata
    if ext == ".png" or data[:8] == b"\x89PNG\r\n\x1a\n":
        out["format"] = "png"
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            out["issues"].append({"severity": "error",
                                  "kind": "bad_png_signature"})
            out["success"] = False
            return out
        # walk chunks until IEND
        i = 8
        chunks: List[str] = []
        text_chunks: Dict[str, str] = {}
        while i + 12 <= len(data):
            ln = struct.unpack(">I", data[i:i + 4])[0]
            ctype = data[i + 4:i + 8].decode("ascii", errors="replace")
            if ln > len(data):
                out["issues"].append({"severity": "error",
                                      "kind": "png_chunk_length_overflow",
                                      "chunk": ctype, "len": int(ln)})
                out["success"] = False
                break
            chunks.append(ctype)
            payload = data[i + 8:i + 8 + ln]
            if ctype in ("tEXt", "iTXt") and len(payload) < 200_000:
                try:
                    # tEXt = key\0value (latin-1); iTXt = key\0\x00\x00lang\0tlate\0value (utf-8)
                    if ctype == "tEXt":
                        k, _, v = payload.partition(b"\x00")
                        text_chunks[k.decode("latin-1", errors="replace")] = \
                            v.decode("latin-1", errors="replace")
                    else:  # iTXt
                        parts = payload.split(b"\x00", 4)
                        if len(parts) >= 5:
                            k = parts[0].decode("utf-8", errors="replace")
                            v = parts[4].decode("utf-8", errors="replace")
                            text_chunks[k] = v
                except Exception:
                    pass
            i += 8 + ln + 4  # length + type + payload + crc
            if ctype == "IEND":
                break
        out["png_chunks"] = chunks
        # Detect embedded ComfyUI workflow
        for key in ("workflow", "prompt", "parameters"):
            v = text_chunks.get(key)
            if not v:
                continue
            preview = v[:200]
            out.setdefault("embedded_metadata", {})[key] = {
                "size": len(v), "preview": preview,
            }
            if key in ("workflow", "prompt"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, dict):
                        out["embedded_metadata"][key]["parsed_keys"] = \
                            list(parsed.keys())[:16]
                except Exception:
                    out["issues"].append({"severity": "warning",
                                          "kind": "embedded_json_invalid",
                                          "chunk": key})
        return out

    # safetensors
    if ext == ".safetensors":
        out["format"] = "safetensors"
        ok, info = _safetensors_header_ok(data)
        if not ok:
            out["issues"].append({"severity": "error",
                                  "kind": "safetensors_header_invalid",
                                  **info})
            out["success"] = False
        else:
            out.update(info)
            out["issues"].append({"severity": "info",
                                  "kind": "header_valid"})
        return out

    # legacy .ckpt / .pt / .bin - these are pickles. We DO NOT unpickle.
    if ext in (".ckpt", ".pt", ".bin", ".pth"):
        out["format"] = "pickle"
        out["issues"].append({"severity": "warning",
                              "kind": "pickle_format",
                              "detail": "Loading executes arbitrary code. "
                                        "Prefer .safetensors. Hash recorded "
                                        "above for offline verification."})
        # Best-effort: zip-wrapped pickles (PyTorch >=1.6) start with PK.
        if data[:2] == b"PK":
            zr = _scan_zip_or_pack(data, filename)
            out["zip_inspection"] = zr
        return out

    # zip / wheels / model packs
    if ext in (".zip", ".whl") or data[:2] == b"PK":
        out["format"] = "zip"
        out.update(_scan_zip_or_pack(data, filename))
        return out

    # python source
    if ext == ".py":
        out["format"] = "python"
        for f in _scan_python_bytes(data, filename):
            out["issues"].append({"severity": "info", **f})
        return out

    # txt / md / unknown
    out["format"] = ext.lstrip(".") or "unknown"
    out["issues"].append({"severity": "info",
                          "kind": "unrecognized_format",
                          "detail": "Recorded SHA-256 only."})
    return out


_WORKFLOW_PNG_KEYS = frozenset({"workflow", "Workflow", "prompt", "Prompt", "parameters"})


def clean_png_metadata(data: bytes, *, workflow_keys_only: bool = True) -> Dict[str, Any]:
    """Strip workflow-related tEXt/iTXt/zTXt chunks; preserve image data (IDAT)."""
    import struct

    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return {"success": False, "error": "not_png"}
    out = bytearray(data[:8])
    removed: List[str] = []
    i = 8
    while i + 12 <= len(data):
        ln = struct.unpack(">I", data[i:i + 4])[0]
        if ln > len(data) - i - 12:
            return {"success": False, "error": "png_chunk_length_overflow"}
        ctype = data[i + 4:i + 8]
        chunk = data[i:i + 12 + ln]
        drop = False
        if ctype in (b"tEXt", b"iTXt", b"zTXt"):
            if not workflow_keys_only:
                drop = True
            else:
                payload = data[i + 8:i + 8 + ln]
                nul = payload.find(b"\x00")
                if nul > 0:
                    key = payload[:nul].decode("latin-1", errors="replace")
                    if key in _WORKFLOW_PNG_KEYS:
                        drop = True
                elif ctype != b"tEXt":
                    drop = False
            if drop:
                removed.append(ctype.decode("ascii", errors="replace"))
        if not drop:
            out.extend(chunk)
        i += 12 + ln
        if ctype == b"IEND":
            break
    cleaned = bytes(out)
    return {
        "success": True,
        "removed_chunks": removed,
        "removed_count": len(removed),
        "bytes_before": len(data),
        "bytes_after": len(cleaned),
        "data_b64": __import__("base64").b64encode(cleaned).decode("ascii"),
    }


# ---------------------------------------------------------------------------
# Aiohttp route registration
# ---------------------------------------------------------------------------
_ROUTES_REGISTERED = False


# --- D.6 newbie-mode plain-English explainer ---------------------------------
# Translates the raw doctor signals (pyenv + queued import failures +
# missing API keys) into 1-3 sentence bullets that a fresh ComfyUI user
# can actually act on. Never depends on a network LLM — uses the
# RulePackBackend (tier-1) when present, otherwise a built-in lookup
# table.
def _explain_signals(*, newbie: bool = True) -> Dict[str, Any]:
    """Compile a plain-English health report for the Doctor panel.

    Returns ``{ success, generated_ts, newbie, items: [...], counts: {...} }``
    where each ``item`` is::

        { severity: 'ok'|'info'|'warning'|'error',
          title:    str,
          detail:   str,         # one paragraph, newbie-friendly
          fixes:    [str, ...],  # 0-3 concrete next steps
          source:   str }        # 'pyenv' / 'registry' / 'ai' / 'gguf' }
    """
    items: List[Dict[str, Any]] = []

    # 1) Python / CUDA sanity --------------------------------------------------
    try:
        pyenv = collect_pyenv()
        dev = pyenv.get("device") or {}
        if not dev.get("cuda_available"):
            items.append({
                "severity": "warning",
                "title": "No GPU detected",
                "detail": (
                    "Your Python install doesn't have a working CUDA build of "
                    "PyTorch. ComfyUI will still run on CPU but image and "
                    "video generation will be very slow."
                    if newbie else
                    "torch.cuda.is_available() == False"
                ),
                "fixes": [
                    "Re-install PyTorch matching your NVIDIA driver: "
                    "https://pytorch.org/get-started/locally/",
                    "If you have an AMD or Intel GPU, install the matching "
                    "ROCm / IPEX wheel instead.",
                ],
                "source": "pyenv",
            })
        # Flag missing relevant packages the user is likely to want.
        missing = [p["name"] for p in pyenv.get("packages", [])
                   if not p.get("installed")]
        if missing:
            items.append({
                "severity": "info",
                "title": f"{len(missing)} optional package(s) not installed",
                "detail": (
                    "Some packs need extra Python libraries to unlock all "
                    "their nodes. The Doctor → Heal tab can install them "
                    "in one click."
                    if newbie else
                    "Missing: " + ", ".join(missing[:10])
                ),
                "fixes": [
                    "Open Doctor → Heal and click 'Install missing'.",
                    "Or run pip install " + " ".join(missing[:6]) +
                    (" ..." if len(missing) > 6 else ""),
                ],
                "source": "pyenv",
            })
    except Exception as exc:
        items.append({
            "severity": "error",
            "title": "Could not read Python environment",
            "detail": str(exc)[:280],
            "fixes": ["Check the ComfyUI console for a full traceback."],
            "source": "pyenv",
        })

    # 2) Queued import / runtime failures -------------------------------------
    try:
        from . import _c2c_registry  # type: ignore
        snap = _c2c_registry.summary()
        failures = snap.get("failures") or []
        # Collapse duplicates by (group, key) keeping the latest message.
        seen: Dict[str, Dict[str, Any]] = {}
        for f in failures:
            k = f"{f.get('group')}/{f.get('key')}"
            seen[k] = f
        for k, f in seen.items():
            sev = f.get("severity") or "warning"
            if sev not in ("info", "warning", "error"):
                sev = "warning"
            fixes: List[str] = []
            if f.get("hint"):
                fixes.append(str(f["hint"]))
            items.append({
                "severity": sev,
                "title": (
                    f"{f.get('key')} is unavailable" if newbie else
                    f"{k}: {f.get('exception_type')}"
                ),
                "detail": (
                    "A pack tried to load this feature and the import "
                    f"failed. Reason: {f.get('exception_type')}: "
                    f"{(f.get('message') or '')[:200]}"
                    if newbie else
                    (f.get("message") or "")[:400]
                ),
                "fixes": fixes,
                "source": "registry",
            })
    except Exception:
        pass  # registry missing -> nothing to add

    # 3) AI explainer readiness ------------------------------------------------
    try:
        from .. import c2c_ai as _cai  # type: ignore
        router = _cai.get_router() if hasattr(_cai, "get_router") else None
        backends = router.all_backends() if router else []
        cloud = [b for b in backends if b.info.tier.value == "cloud"]
        local = [b for b in backends if b.info.tier.value == "local"]
        if not cloud and not local:
            items.append({
                "severity": "info",
                "title": "AI error explanations not configured",
                "detail": (
                    "ComfyUI errors will still be matched against the "
                    "built-in rule pack (82 patterns), but you'll get richer "
                    "plain-English explanations if you connect an AI backend."
                    if newbie else
                    "No cloud or local backends registered in the router."
                ),
                "fixes": [
                    "Settings → C2C AI → enter an OpenAI / Anthropic / "
                    "OpenRouter key (free OpenRouter tier works).",
                    "Or install a local GGUF: Settings → C2C AI → Download "
                    "→ Qwen3-4B-Instruct (~2.5 GB).",
                ],
                "source": "ai",
            })
        else:
            items.append({
                "severity": "ok",
                "title": f"{len(backends)} AI backend(s) ready",
                "detail": ", ".join(b.info.display_name for b in backends[:5]),
                "fixes": [],
                "source": "ai",
            })
    except Exception as exc:
        items.append({
            "severity": "info",
            "title": "AI spine not loaded",
            "detail": str(exc)[:200],
            "fixes": [],
            "source": "ai",
        })

    # 4) Counts summary --------------------------------------------------------
    counts = {"ok": 0, "info": 0, "warning": 0, "error": 0}
    for it in items:
        counts[it.get("severity", "info")] = counts.get(it.get("severity", "info"), 0) + 1

    return {
        "success": True,
        "generated_ts": time.time(),
        "newbie": bool(newbie),
        "items": items,
        "counts": counts,
    }


def register_routes(server) -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED or server is None:
        return
    try:
        from aiohttp import web
    except Exception:
        return
    import asyncio
    import functools

    routes = server.routes if hasattr(server, "routes") else server.app.router

    async def _run_blocking(fn, *a, **kw):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *a, **kw))

    @routes.get("/c2c/doctor/pyenv")
    async def _pyenv(_req):
        data = await _run_blocking(collect_pyenv)
        return web.json_response(data)

    @routes.get("/c2c/doctor/disk")
    async def _disk(req):
        refresh = req.query.get("refresh") in ("1", "true", "yes")
        data = await _run_blocking(collect_disk, refresh=refresh)
        return web.json_response(data)

    @routes.get("/c2c/doctor/explain")
    async def _explain(req):
        # ?newbie=0 disables friendly rewording (devs / power users).
        nb = req.query.get("newbie", "1") not in ("0", "false", "no")
        data = await _run_blocking(_explain_signals, newbie=nb)
        return web.json_response(data)

    # Upload limit: 64 MiB. Anything bigger is rejected (don't OOM the server).
    _MAX_UPLOAD = 64 * 1024 * 1024

    @routes.post("/c2c/doctor/scan_file")
    async def _scan_file(req):
        ctype = (req.headers.get("Content-Type") or "").lower()
        try:
            if "multipart/form-data" in ctype:
                reader = await req.multipart()
                filename = "unnamed"
                buf = bytearray()
                while True:
                    part = await reader.next()
                    if part is None:
                        break
                    if part.name == "file":
                        filename = part.filename or "unnamed"
                        while True:
                            chunk = await part.read_chunk(size=1 << 16)
                            if not chunk:
                                break
                            buf.extend(chunk)
                            if len(buf) > _MAX_UPLOAD:
                                return web.json_response(
                                    {"success": False,
                                     "error": "file_too_large",
                                     "limit": _MAX_UPLOAD,
                                     "size": len(buf)},
                                    status=200,
                                )
                if not buf:
                    return web.json_response(
                        {"success": False, "error": "no_file_part"},
                        status=200,
                    )
                result = await _run_blocking(scan_file, filename, bytes(buf))
                return web.json_response(result)
            else:
                data = await req.read()
                if len(data) > _MAX_UPLOAD:
                    return web.json_response(
                        {"success": False, "error": "file_too_large",
                         "limit": _MAX_UPLOAD, "size": len(data)},
                        status=200,
                    )
                filename = req.query.get("name") or "unnamed.bin"
                result = await _run_blocking(scan_file, filename, data)
                return web.json_response(result)
        except Exception as exc:
            return web.json_response(
                {"success": False, "error": "exception",
                 "detail": str(exc)[:500]},
                status=200,
            )

    @routes.post("/c2c/doctor/clean_png")
    async def _clean_png(req):
        """Remove damaged or unwanted workflow metadata chunks from a PNG."""
        try:
            reader = await req.multipart()
            filename = "image.png"
            buf = bytearray()
            workflow_only = req.query.get("workflow_only", "1") not in ("0", "false", "no")
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == "file":
                    filename = part.filename or filename
                    while True:
                        chunk = await part.read_chunk(size=1 << 16)
                        if not chunk:
                            break
                        buf.extend(chunk)
                        if len(buf) > _MAX_UPLOAD:
                            return web.json_response(
                                {"success": False, "error": "file_too_large"},
                                status=200,
                            )
            if not buf:
                return web.json_response({"success": False, "error": "no_file_part"}, status=200)
            result = await _run_blocking(
                clean_png_metadata, bytes(buf), workflow_keys_only=workflow_only,
            )
            result["filename"] = filename
            return web.json_response(result)
        except Exception as exc:
            return web.json_response(
                {"success": False, "error": "exception", "detail": str(exc)[:500]},
                status=200,
            )

    _ROUTES_REGISTERED = True
