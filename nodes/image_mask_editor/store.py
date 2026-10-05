"""Persistent mask store for ImageMaskEditorC2C (stdlib + numpy + PIL only)."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "ImageMaskEditorC2C requires Pillow. Install with: pip install pillow"
    ) from e

_EDITOR_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")
_MAX_FRAME = 100_000
_MAX_PNG_BYTES = 64 * 1024 * 1024
_MAX_DIM = 16_384

_LOCKS_GUARD = threading.Lock()
_LOCKS: Dict[str, threading.Lock] = {}


def _root() -> Path:
    import folder_paths

    return Path(folder_paths.get_input_directory()) / "c2c_masks"


def _lock_for(editor_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lk = _LOCKS.get(editor_id)
        if lk is None:
            lk = threading.Lock()
            _LOCKS[editor_id] = lk
        return lk


def validate_editor_id(editor_id: str) -> str:
    eid = (editor_id or "").strip()
    if not eid or not _EDITOR_ID_RE.fullmatch(eid):
        raise ValueError(f"invalid editor_id: {editor_id!r}")
    if ".." in eid or "/" in eid or "\\" in eid:
        raise ValueError(f"invalid editor_id: {editor_id!r}")
    return eid


def validate_frame(frame: int) -> int:
    f = int(frame)
    if f < 0 or f > _MAX_FRAME:
        raise ValueError(f"frame out of range: {frame}")
    return f


def _editor_dir(editor_id: str) -> Path:
    eid = validate_editor_id(editor_id)
    return _root() / eid


def _meta_path(editor_id: str) -> Path:
    return _editor_dir(editor_id) / "meta.json"


def _frame_path(editor_id: str, frame: int) -> Path:
    return _editor_dir(editor_id) / f"{validate_frame(frame):05d}.png"


def _png_to_mask(png_bytes: bytes) -> np.ndarray:
    if not png_bytes or len(png_bytes) > _MAX_PNG_BYTES:
        raise ValueError("empty or too-large PNG payload")
    im = Image.open(io.BytesIO(png_bytes))
    if im.mode != "L":
        im = im.convert("L")
    arr = np.array(im, dtype=np.uint8)
    h, w = arr.shape[:2]
    if h > _MAX_DIM or w > _MAX_DIM:
        raise ValueError(f"mask dimensions exceed {_MAX_DIM}")
    return arr


def _mask_to_png(arr: np.ndarray) -> bytes:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False, compress_level=3)
    return buf.getvalue()


def frame_sha1(shape_hw: tuple[int, int], arr: np.ndarray) -> str:
    h, w = shape_hw
    payload = f"{h},{w}|".encode("ascii") + np.ascontiguousarray(arr).tobytes()
    return hashlib.sha1(payload).hexdigest()


def _load_meta_unlocked(ed_dir: Path) -> dict:
    mp = ed_dir / "meta.json"
    if not mp.is_file():
        return {"shape": None, "frames": {}, "updated": None}
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"shape": None, "frames": {}, "updated": None}
    frames = data.get("frames") or {}
    if not isinstance(frames, dict):
        frames = {}
    shape = data.get("shape")
    return {
        "shape": list(shape) if shape else None,
        "frames": {str(k): str(v) for k, v in frames.items()},
        "updated": data.get("updated"),
    }


def _write_meta_atomic(ed_dir: Path, meta: dict) -> None:
    ed_dir.mkdir(parents=True, exist_ok=True)
    tmp = ed_dir / "meta.json.tmp"
    body = json.dumps(meta, separators=(",", ":"), ensure_ascii=False)
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, ed_dir / "meta.json")


def put_frame(editor_id: str, frame: int, png_bytes: bytes) -> str:
    """Store one frame PNG. Returns sha1 hex of (shape + raw bytes)."""
    eid = validate_editor_id(editor_id)
    fnum = validate_frame(frame)
    arr = _png_to_mask(png_bytes)
    h, w = arr.shape[:2]
    sha = frame_sha1((h, w), arr)
    canonical = _mask_to_png(arr)   # one format on disk (8-bit L), whatever the client sent

    ed_dir = _editor_dir(eid)
    frame_file = f"{fnum:05d}.png"
    tmp_frame = ed_dir / f"{frame_file}.tmp"

    with _lock_for(eid):
        ed_dir.mkdir(parents=True, exist_ok=True)
        tmp_frame.write_bytes(canonical)
        os.replace(tmp_frame, ed_dir / frame_file)
        meta = _load_meta_unlocked(ed_dir)
        meta["shape"] = [h, w]
        meta["frames"][str(fnum)] = sha
        meta["updated"] = datetime.now(timezone.utc).isoformat()
        _write_meta_atomic(ed_dir, meta)
    return sha


def get_frame(editor_id: str, frame: int) -> Optional[np.ndarray]:
    """Return uint8 HxW mask or None if missing. No lock during decode."""
    try:
        eid = validate_editor_id(editor_id)
        fnum = validate_frame(frame)
    except ValueError:
        return None
    path = _editor_dir(eid) / f"{fnum:05d}.png"
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    try:
        return _png_to_mask(data)
    except ValueError:
        return None


def get_frame_png(editor_id: str, frame: int) -> Optional[bytes]:
    """The stored PNG bytes (canonical 8-bit L since put_frame writes it), or None."""
    try:
        path = _editor_dir(validate_editor_id(editor_id)) / f"{validate_frame(frame):05d}.png"
    except ValueError:
        return None
    try:
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


def list_frames(editor_id: str) -> List[int]:
    try:
        eid = validate_editor_id(editor_id)
    except ValueError:
        return []
    with _lock_for(eid):
        meta = _load_meta_unlocked(_editor_dir(eid))
    return sorted(int(k) for k in meta.get("frames", {}).keys() if str(k).isdigit())


def delete_frame(editor_id: str, frame: int) -> None:
    eid = validate_editor_id(editor_id)
    fnum = validate_frame(frame)
    path = _editor_dir(eid) / f"{fnum:05d}.png"
    with _lock_for(eid):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        ed_dir = _editor_dir(eid)
        meta = _load_meta_unlocked(ed_dir)
        meta["frames"].pop(str(fnum), None)
        if not meta["frames"]:
            meta["shape"] = None
        meta["updated"] = datetime.now(timezone.utc).isoformat()
        _write_meta_atomic(ed_dir, meta)


def clear(editor_id: str) -> None:
    eid = validate_editor_id(editor_id)
    ed_dir = _editor_dir(eid)
    with _lock_for(eid):
        if ed_dir.is_dir():
            for child in ed_dir.iterdir():
                try:
                    if child.is_file():
                        child.unlink()
                except OSError:
                    pass
        _write_meta_atomic(ed_dir, {"shape": None, "frames": {}, "updated": None})


def copy(src_id: str, dst_id: str) -> None:
    src = validate_editor_id(src_id)
    dst = validate_editor_id(dst_id)
    if src == dst:
        return
    src_dir = _editor_dir(src)
    dst_dir = _editor_dir(dst)
    frame_files: list[Path] = []
    with _lock_for(src):
        if not src_dir.is_dir():
            meta_src = {"shape": None, "frames": {}, "updated": None}
        else:
            meta_src = _load_meta_unlocked(src_dir)
            frame_files = [p for p in src_dir.glob("*.png") if not p.name.endswith(".tmp")]
    with _lock_for(dst):
        if dst_dir.is_dir():
            shutil.rmtree(dst_dir, ignore_errors=True)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fp in frame_files:
            if fp.name.endswith(".tmp"):
                continue
            shutil.copy2(fp, dst_dir / fp.name)
        _write_meta_atomic(dst_dir, dict(meta_src))


def digest(editor_id: str) -> str:
    eid = (editor_id or "").strip()
    if not eid:
        return ""
    try:
        validate_editor_id(eid)
    except ValueError:
        return ""
    with _lock_for(eid):
        meta = _load_meta_unlocked(_editor_dir(eid))
    frames = meta.get("frames") or {}
    if not frames:
        return ""
    parts = [f"{k}:{frames[k]}" for k in sorted(frames.keys(), key=lambda x: int(x))]
    return hashlib.sha1("|".join(parts).encode("ascii")).hexdigest()


def get_shape(editor_id: str) -> Optional[tuple[int, int]]:
    eid = (editor_id or "").strip()
    if not eid:
        return None
    try:
        validate_editor_id(eid)
    except ValueError:
        return None
    with _lock_for(eid):
        meta = _load_meta_unlocked(_editor_dir(eid))
    shape = meta.get("shape")
    if not shape or len(shape) != 2:
        return None
    return int(shape[0]), int(shape[1])
