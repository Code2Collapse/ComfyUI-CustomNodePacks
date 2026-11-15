"""Curated GGUF catalog for the Diagnostics Tier 2 (llama.cpp) assistant (L6.05).

The catalog is data (``data/llm_catalog.json``): every repo / file / size / SHA-256 was read from the Hugging Face API
by ``tools/c2c_llm_catalog_verify.py``. This module loads it, maps a catalog id to its real file name (catalog ids
such as ``qwen3-0.6b-instruct-q8_0`` are not substrings of the file ``Qwen3-0.6B-Q8_0.gguf``, so a name search alone
never found them), and recommends the largest general-purpose model that fits the memory free right now.

L6.09: ``hf_folders`` - Hugging Face Transformers model folders (format ``hf_folder``) for the AI spine's hf_local
backend, pinned to a commit with every file's size and hash. ``download_hf_folder`` fetches one into
``models/LLM/<name>`` on the shared download manager (resumable ``.part`` files, SHA-256 for LFS files, git blob SHA-1
for the others, size for all). It is started only by an explicit user action (the ``/c2c/ai/hf_catalog/download``
route); nothing here downloads at import, while listing, or in a probe.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import urllib.parse
from pathlib import Path
from typing import Optional

log = logging.getLogger("C2C.LLMCatalog")
CATALOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "llm_catalog.json")
GIB = 1 << 30
# llama.cpp needs the weights plus the KV cache and scratch buffers: ~20 % on top of the file + 0.5 GiB at the
# assistant's context size. Only a fraction of free RAM is offered: ComfyUI and its models live in the same RAM.
OVERHEAD_FRACTION, OVERHEAD_FIXED, RAM_SHARE = 0.20, GIB // 2, 0.6
_PREFERRED_FAMILIES = ("Qwen 3", "Qwen 2.5 / Coder", "Google Gemma 3", "Meta Llama 3.2")
_LOCK = threading.Lock()
_CACHE: dict = {"mtime": None, "data": None}


def load(path: Optional[str] = None) -> dict:
    """The catalog dict ({schema, default, models: [...]}); re-read when the file changes. Never raises: a missing
    or broken file gives an empty catalog and a log line."""
    p = path or CATALOG_PATH
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        log.warning("[llm_catalog] %s is missing; the model list is empty", p)
        return {"schema": 1, "default": "", "models": []}
    with _LOCK:
        if path is None and _CACHE["mtime"] == mtime and _CACHE["data"] is not None:
            return _CACHE["data"]
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data.get("models"), list):
                raise ValueError("no 'models' list")
        except (OSError, ValueError) as exc:
            log.warning("[llm_catalog] %s is unreadable (%s); the model list is empty", p, exc)
            return {"schema": 1, "default": "", "models": []}
        if path is None:
            _CACHE.update(mtime=mtime, data=data)
        return data


def by_id(model_id: str, catalog: Optional[dict] = None) -> Optional[dict]:
    mid = (model_id or "").strip().lower()
    return next((m for m in (catalog or load())["models"] if m.get("id", "").lower() == mid), None)


def by_repo_file(repo: str, fname: str, catalog: Optional[dict] = None) -> Optional[dict]:
    return next((m for m in (catalog or load())["models"] if m.get("repo") == repo and m.get("file") == fname), None)


def need_bytes(entry: dict) -> int:
    return int(int(entry.get("size") or 0) * (1 + OVERHEAD_FRACTION)) + OVERHEAD_FIXED


_GPU_OFFLOAD: dict = {}


def llama_gpu_offload() -> bool:
    """True only when the installed llama-cpp-python can put layers on the GPU (the CPU wheel cannot, whatever torch
    sees). Checked once per process; False when llama-cpp-python is not installed."""
    if "ok" not in _GPU_OFFLOAD:
        ok = False
        try:
            import importlib.util
            if importlib.util.find_spec("llama_cpp") is not None:
                import llama_cpp  # type: ignore
                fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
                ok = bool(fn()) if callable(fn) else False
        except Exception:  # noqa: BLE001 - a broken install counts as "no GPU offload"
            ok = False
        _GPU_OFFLOAD["ok"] = ok
    return _GPU_OFFLOAD["ok"]


def memory_free() -> dict:
    """{ram_available, vram_free} in bytes (None when unknown or unusable). Cheap: no model is touched. Free VRAM is
    only reported when llama.cpp can offload to the GPU."""
    out = {"ram_available": None, "vram_free": None, "gpu_offload": llama_gpu_offload()}
    try:
        import psutil  # type: ignore
        out["ram_available"] = int(psutil.virtual_memory().available)
    except Exception:
        pass
    if out["gpu_offload"]:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                out["vram_free"] = int(torch.cuda.mem_get_info()[0])
        except Exception:
            pass
    return out


def recommend(catalog: Optional[dict] = None, mem: Optional[dict] = None) -> dict:
    """{id, fits, budget, reason}: the largest 'chat' model (preferred families first) whose estimated need fits the
    budget = RAM_SHARE x available RAM (or free VRAM when that is larger: llama.cpp can offload all layers)."""
    cat = catalog or load()
    mem = mem if mem is not None else memory_free()
    ram = mem.get("ram_available") or 0
    vram = mem.get("vram_free") or 0
    budget = max(int(ram * RAM_SHARE), int(vram * 0.9))
    chat = [m for m in cat["models"] if m.get("use") == "chat" and m.get("size")]
    if not chat:
        return {"id": cat.get("default") or "", "fits": False, "budget": budget, "reason": "the catalog is empty"}
    if not budget:
        return {"id": cat.get("default") or chat[0]["id"], "fits": False, "budget": 0,
                "reason": "free memory is unknown; showing the default"}
    rank = {f: i for i, f in enumerate(_PREFERRED_FAMILIES)}
    fitting = [m for m in chat if need_bytes(m) <= budget]
    if not fitting:
        small = min(chat, key=lambda m: m["size"])
        return {"id": small["id"], "fits": False, "budget": budget,
                "reason": f"even the smallest model needs ~{need_bytes(small) / GIB:.1f} GiB; "
                          f"{budget / GIB:.1f} GiB is free for it"}
    best = max(fitting, key=lambda m: (m["size"] // (GIB // 2), -rank.get(m.get("group"), 99), m["size"]))
    where = "GPU" if vram * 0.9 >= ram * RAM_SHARE and vram else "RAM"
    return {"id": best["id"], "fits": True, "budget": budget,
            "reason": f"largest general model that fits {budget / GIB:.1f} GiB of free {where} "
                      f"(needs ~{need_bytes(best) / GIB:.1f} GiB)"}


def _label(n: int) -> str:
    return f"~{n / GIB:.1f} GB" if n >= GIB else f"~{n / (1 << 20):.0f} MB"


def public(catalog: Optional[dict] = None, mem: Optional[dict] = None) -> dict:
    """What the sidebar needs: the models with a size label and estimated memory, plus the recommendation."""
    cat = catalog or load()
    models = []
    for m in cat["models"]:
        need = need_bytes(m)
        size = int(m.get("size") or 0)
        models.append({**m, "ram": _label(need), "need_bytes": need, "size_label": _label(size)})
    return {"schema": cat.get("schema", 1), "verified": cat.get("verified", ""), "default": cat.get("default", ""),
            "models": models, "recommended": recommend(cat, mem), "hf_folders": hf_folders(cat)}


# ── L6.09: Hugging Face Transformers folders ─────────────────────────────────────────────────────────────────────
_NAME_RE = re.compile(r"^[A-Za-z0-9][\w.-]*$")
_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
_REV_RE = re.compile(r"^[0-9a-f]{40}$")
PARTIAL_SUFFIX = ".c2c-partial"
_HF_JOBS: dict = {}
_HF_TASKS: dict = {}


def hf_folders(catalog: Optional[dict] = None) -> list:
    """The catalog's Transformers folder entries (format hf_folder); [] when there are none."""
    cat = catalog or load()
    out = cat.get("hf_folders") or []
    return [e for e in out if isinstance(e, dict) and e.get("format") == "hf_folder"]


def hf_folder_by_id(model_id: str, catalog: Optional[dict] = None) -> Optional[dict]:
    mid = (model_id or "").strip().lower()
    return next((e for e in hf_folders(catalog) if e.get("id", "").lower() == mid), None)


def git_blob_sha1(path: str) -> str:
    """The git blob id of a file (what the Hugging Face tree API lists as `oid` for a non-LFS file)."""
    h = hashlib.sha1()
    h.update(b"blob %d\x00" % os.path.getsize(path))
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_llm_root() -> Path:
    """models/LLM - the folder the AI spine's hf_local backend scans (c2c_ai/hf_scan.py)."""
    try:
        import folder_paths  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"ComfyUI's folder_paths is not available, so models/LLM is unknown ({exc})") from exc
    return Path(folder_paths.models_dir) / "LLM"


def _browser_routes():
    try:
        from . import model_browser_routes as mb  # type: ignore
    except ImportError:
        from nodes import model_browser_routes as mb  # type: ignore
    return mb


def _manager():
    try:
        from .model_download_manager import get_manager  # type: ignore
    except ImportError:
        from nodes.model_download_manager import get_manager  # type: ignore
    return get_manager()


def _check_entry(entry: dict) -> str:
    """'' when the entry is safe to download, else the reason in words."""
    name = str(entry.get("name") or "")
    if not _NAME_RE.fullmatch(name) or ".." in name:
        return f"folder name {name!r} is not a plain folder name"
    if not _REPO_RE.fullmatch(str(entry.get("repo") or "")):
        return f"repo {entry.get('repo')!r} is not 'owner/name'"
    if not _REV_RE.fullmatch(str(entry.get("revision") or "")):
        return "the entry is not pinned to a commit (revision must be 40 hex characters)"
    files = entry.get("files") or []
    if not files:
        return "the entry lists no files"
    for f in files:
        p = str(f.get("path") or "")
        if p in ("", ".", "..") or "/" in p or "\\" in p or ".." in p:
            return f"file name {p!r} is not a plain file name"
        if not isinstance(f.get("size"), int) or f["size"] <= 0:
            return f"{p}: no size"
        if not (f.get("sha256") or f.get("git_oid")):
            return f"{p}: no hash"
    return ""


def _file_ok(path: Path, f: dict) -> bool:
    try:
        if path.stat().st_size != int(f["size"]):
            return False
        if f.get("sha256"):
            return _sha256(str(path)) == f["sha256"].lower()
        return git_blob_sha1(str(path)) == f["git_oid"].lower()
    except OSError:
        return False


def _state(model_id: str, **kw) -> dict:
    st = _HF_JOBS.setdefault(model_id, {"id": model_id, "status": "idle", "file": "", "files_done": 0,
                                        "files_total": 0, "bytes_done": 0, "bytes_total": 0, "error": "",
                                        "dest": ""})
    st.update(kw)
    return st


async def download_hf_folder(entry: dict, *, root=None, manager=None) -> dict:
    """Fetch one hf_folder entry into <root>/<name> (root defaults to models/LLM). Files go to <name>.c2c-partial,
    config.json last, and the folder is renamed into place only when every file is verified, so the hf_local scan
    never sees half a model. Re-running resumes: verified files are kept and '.part' files continue with a Range
    request. Returns {status: done|exists|error|paused|cancelled, dest, error, files_done, files_total}; never raises.
    """
    mid = str(entry.get("id") or entry.get("name") or "")
    files = list(entry.get("files") or [])
    out = {"status": "error", "dest": "", "error": "", "files_done": 0, "files_total": len(files)}

    def finish(**kw):
        out.update(kw)
        _state(mid, status=out["status"], error=out["error"], files_done=out["files_done"], dest=out["dest"])
        return out

    why = _check_entry(entry)
    if why:
        return finish(error=f"refused: {why}")
    try:
        base = Path(root) if root is not None else hf_llm_root()
    except Exception as exc:  # noqa: BLE001
        return finish(error=str(exc))
    final = base / entry["name"]
    out["dest"] = str(final)
    if (final / "config.json").exists():
        return finish(status="exists", files_done=len(files))
    partial = base / (entry["name"] + PARTIAL_SUFFIX)
    try:
        partial.mkdir(parents=True, exist_ok=True)
        mb = _browser_routes()
        manager = manager if manager is not None else _manager()
    except Exception as exc:  # noqa: BLE001
        return finish(error=f"could not prepare the download: {exc}")
    ordered = [f for f in files if f["path"] != "config.json"] + [f for f in files if f["path"] == "config.json"]
    total = sum(int(f["size"]) for f in files)
    done_bytes = 0
    _state(mid, status="running", files_total=len(files), bytes_total=total, dest=str(final), error="")
    headers = {"User-Agent": mb._BROWSER_UA}
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"              # a header only, never the URL
    for i, f in enumerate(ordered):
        dest = partial / f["path"]
        _state(mid, file=f["path"], files_done=i, bytes_done=done_bytes, job_id="")
        if dest.exists():
            if await asyncio.to_thread(_file_ok, dest, f):
                done_bytes += int(f["size"])
                continue
            dest.unlink(missing_ok=True)
        url = f"{mb._HF_BASE}/{entry['repo']}/resolve/{entry['revision']}/{urllib.parse.quote(f['path'])}"
        try:
            job = manager.add(url, dest, headers, sha256=f.get("sha256"), label=f"{entry['repo']} / {f['path']}",
                              meta=None)
            _state(mid, job_id=job.id)
            await manager.wait(job.id)
        except Exception as exc:  # noqa: BLE001
            return finish(error=f"{f['path']}: {exc}", files_done=i)
        if job.status != "done":
            status = job.status if job.status in ("paused", "cancelled") else "error"
            return finish(status=status, error=job.error or f"{f['path']}: download {job.status}", files_done=i)
        try:
            got = dest.stat().st_size
        except OSError as exc:
            return finish(error=f"{f['path']}: missing after download ({exc})", files_done=i)
        if got != int(f["size"]):
            dest.unlink(missing_ok=True)
            return finish(error=f"{f['path']}: size {got} != {f['size']} bytes listed by Hugging Face; removed. "
                                "Try again.", files_done=i)
        if f.get("git_oid"):
            oid = await asyncio.to_thread(git_blob_sha1, str(dest))
            if oid != f["git_oid"].lower():
                dest.unlink(missing_ok=True)
                return finish(error=f"{f['path']}: git hash mismatch (expected {f['git_oid']}, got {oid}); "
                                    "removed. Try again.", files_done=i)
        done_bytes += int(f["size"])
    try:
        os.replace(partial, final)
    except OSError as exc:
        return finish(error=f"every file is verified in {partial}, but it could not be renamed to {final}: {exc}",
                      files_done=len(files))
    log.info("[llm_catalog] %s downloaded and verified: %s", entry.get("repo"), final)
    _state(mid, bytes_done=total, file="", job_id="")
    return finish(status="done", files_done=len(files))


def hf_folder_progress(model_id: str) -> Optional[dict]:
    entry = hf_folder_by_id(model_id)
    st = _HF_JOBS.get(entry["id"] if entry else model_id)
    if st is None:
        return None
    out = dict(st)
    if out.get("status") == "running" and out.get("job_id"):
        try:
            job = _manager().jobs.get(out["job_id"])
            if job is not None:
                out["bytes_done"] = int(out.get("bytes_done") or 0) + int(job.done or 0)
        except Exception:  # noqa: BLE001 - progress is a hint
            pass
    return out


def start_hf_folder_download(model_id: str, *, root=None, manager=None) -> dict:
    """The explicit user action (the download route): schedule download_hf_folder on the running event loop and
    return its state. Raises KeyError for an id that is not in the catalog. A second click while it runs returns
    the running job."""
    entry = hf_folder_by_id(model_id)
    if entry is None:
        raise KeyError(model_id)
    mid = entry["id"]
    task = _HF_TASKS.get(mid)
    if task is not None and not task.done():
        return dict(_HF_JOBS.get(mid) or _state(mid, status="running"))
    _state(mid, status="running", error="", files_done=0, files_total=len(entry.get("files") or []))
    _HF_TASKS[mid] = asyncio.get_running_loop().create_task(download_hf_folder(entry, root=root, manager=manager))
    return dict(_HF_JOBS[mid])


def stop_hf_folder_download(model_id: str) -> bool:
    """Pause the file being fetched (its '.part' is kept, so the next download resumes)."""
    entry = hf_folder_by_id(model_id)
    st = _HF_JOBS.get(entry["id"] if entry else model_id) or {}
    if not st.get("job_id") or st.get("status") != "running":
        return False
    return bool(_manager().pause(st["job_id"]))
