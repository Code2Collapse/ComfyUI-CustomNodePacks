"""Installed-model index + sidecar metadata (L5.06).

- Every model the Model Browser finishes downloading is recorded by SHA-256 in `<user dir>/c2c/model_index.json`
  and gets a sidecar `<file>.civitai.json` (model / version ids, name, base model, trigger words, source).
- "Installed?" questions are answered from that index, never by walking or hashing model folders in a request
  (owner rule: no disk scans in request handlers). A file that is gone or changed size is not reported installed.
- "Identify" hashes ONE local file on demand (cached by size + mtime), asks Civitai GET /model-versions/by-hash/{sha}
  (developer.civitai.com reference: SHA256 accepted, 404 when unknown) and writes the sidecar.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("C2C.ModelIndex")
_LOCK = threading.Lock()
SIDECAR_SUFFIX = ".civitai.json"
NOT_MODEL_KEYS = {"custom_nodes"}


def _index_path() -> Path:
    try:
        import folder_paths
        base = Path(folder_paths.get_user_directory())
    except Exception:
        base = Path(__file__).resolve().parents[1] / "user"
    return base / "c2c" / "model_index.json"


def _load(path: Optional[Path] = None) -> dict:
    p = path or _index_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("by_sha", {})
            data.setdefault("by_path", {})
            return data
    except (OSError, ValueError):
        pass
    return {"by_sha": {}, "by_path": {}}


def _save(data: dict, path: Optional[Path] = None) -> None:
    p = path or _index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def cached_sha256(path: Path, index_path: Optional[Path] = None) -> str:
    """SHA-256 of one file, reused while its size and mtime are unchanged (hashing a 6 GB checkpoint takes a while)."""
    path = Path(path).resolve()
    st = path.stat()
    key = str(path)
    with _LOCK:
        data = _load(index_path)
        hit = data["by_path"].get(key)
    if hit and hit.get("size") == st.st_size and hit.get("mtime_ns") == st.st_mtime_ns:
        return hit["sha256"]
    sha = sha256_file(path)
    with _LOCK:
        data = _load(index_path)
        data["by_path"][key] = {"sha256": sha, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        _save(data, index_path)
    return sha


def write_sidecar(model_path: Path, meta: dict) -> Path:
    side = Path(str(model_path) + SIDECAR_SUFFIX)
    payload = {"source": "C2C Model Browser", "written": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta}
    side.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return side


def record(model_path: Path, sha256: str, meta: Optional[dict] = None, index_path: Optional[Path] = None) -> None:
    """Register a finished download (and write its sidecar when there is metadata)."""
    model_path = Path(model_path).resolve()
    st = model_path.stat()
    sha = sha256.lower()
    with _LOCK:
        data = _load(index_path)
        data["by_sha"][sha] = {"path": str(model_path), "size": st.st_size}
        data["by_path"][str(model_path)] = {"sha256": sha, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        _save(data, index_path)
    if meta:
        write_sidecar(model_path, {"sha256": sha, **meta})


def locate(path: Path) -> Optional[dict]:
    """{folder, name} as ComfyUI lists the file (folder key + name relative to that folder), or None."""
    try:
        import folder_paths
    except Exception:
        return None
    path = Path(path).resolve()
    best = None
    for key, paths in (getattr(folder_paths, "folder_names_and_paths", None) or {}).items():
        if key in NOT_MODEL_KEYS:
            continue
        for root in paths[0]:
            r = Path(root).resolve()
            if r in path.parents and (best is None or len(str(r)) > len(str(best[1]))):
                best = (key, r)
    if not best:
        return None
    return {"folder": best[0], "name": os.path.relpath(path, best[1])}


def installed(shas: list, index_path: Optional[Path] = None) -> dict:
    """{sha: {path, folder, name}} for the hashes that are on disk now (size unchanged)."""
    with _LOCK:
        data = _load(index_path)
    out = {}
    for s in shas[:500]:
        hit = data["by_sha"].get(str(s).lower())
        if not hit:
            continue
        p = Path(hit["path"])
        try:
            if not p.is_file() or p.stat().st_size != hit.get("size"):
                continue
        except OSError:
            continue
        out[str(s)] = {"path": str(p), **(locate(p) or {})}
    return out


def civitai_meta(version: dict) -> dict:
    """The sidecar fields from a Civitai model-version object (by-hash or /model-versions/{id})."""
    model = version.get("model") or {}
    return {"civitai_model_id": version.get("modelId"), "civitai_version_id": version.get("id"),
            "model_name": model.get("name") or "", "model_type": model.get("type") or "",
            "version_name": version.get("name") or "", "base_model": version.get("baseModel") or "",
            "trained_words": [str(w) for w in (version.get("trainedWords") or []) if str(w).strip()][:50],
            "page": f"https://civitai.com/models/{version.get('modelId')}?modelVersionId={version.get('id')}"
            if version.get("modelId") else ""}
