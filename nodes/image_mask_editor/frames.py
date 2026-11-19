"""Record input IMAGE batches to temp storage for the mask editor UI."""
from __future__ import annotations

import hashlib
import os
import re
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from .store import validate_editor_id

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

RECORD_BUDGET_PIXELS = 268_435_456
_THUMB_MAX = 384
_THUMB_QUALITY = 80
_CACHE_MAX = 64

_FRAME_PNG = re.compile(r"^(\d{5})\.png$")
_FRAME_THUMB = re.compile(r"^t_(\d{5})\.jpg$")

# editor_id -> (fingerprint_hex, B, W, H, quick_key)
_FINGERPRINT_CACHE: OrderedDict[str, tuple] = OrderedDict()
# L9.277: parallel PNG/JPEG encoding (Pillow releases the GIL while encoding); each worker holds one frame
_WRITE_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))


def _temp_editor_dir(eid: str) -> Path:
    import folder_paths

    return Path(folder_paths.get_temp_directory()) / "c2c_ime" / eid


def _frame_rgb_u8(image: torch.Tensor, i: int) -> np.ndarray:
    """One frame as contiguous uint8 RGB; peak memory is one frame."""
    frame = image[i].detach().to("cpu", torch.float32).numpy()   # no copy for CPU float32; numpy has no bf16
    rgb = np.clip(frame[:, :, :3] * 255.0 + 0.5, 0, 255).astype(np.uint8)    # round, not truncate (L9.282: -0.5 LSB)
    return np.ascontiguousarray(rgb)


def _fingerprint_stream(image: torch.Tensor, b: int, h: int, w: int) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"{b},{h},{w},".encode("ascii"))
    for i in range(b):
        digest.update(_frame_rgb_u8(image, i).tobytes())
    return digest.hexdigest()


def _quick_key(image: torch.Tensor) -> tuple:
    """L9.277: the identity of the batch, exact and cheap: a weak reference to the very same tensor object plus torch's
    in-place version counter and the shape. While that object is alive no other tensor can be it, so an unchanged
    re-run (ComfyUI hands the cached upstream output again) skips hashing every pixel (measured 2.0 s for 120 frames
    of 1344x768); a new tensor or an in-place edit falls back to the full hash."""
    try:
        ref = weakref.ref(image)
    except TypeError:                                         # not weak-referenceable: always hash
        return ()
    return (ref, int(getattr(image, "_version", 0)), tuple(image.shape))


def _same_batch(qk: tuple, image: torch.Tensor) -> bool:
    return (len(qk) == 3 and qk[0]() is image and qk[1] == int(getattr(image, "_version", 0))
            and qk[2] == tuple(image.shape))


def _cache_touch(eid: str, fp: str, b: int, h: int, w: int, qk: tuple = ()) -> None:
    _FINGERPRINT_CACHE[eid] = (fp, b, h, w, qk)
    _FINGERPRINT_CACHE.move_to_end(eid)
    while len(_FINGERPRINT_CACHE) > _CACHE_MAX:
        _FINGERPRINT_CACHE.popitem(last=False)


def _files_complete(ed_dir: Path, count: int) -> bool:
    if count <= 0 or not ed_dir.is_dir():
        return False
    for i in range(count):
        if not (ed_dir / f"{i:05d}.png").is_file():
            return False
        if not (ed_dir / f"t_{i:05d}.jpg").is_file():
            return False
    return True


def _payload(eid: str, fp: str, b: int, w: int, h: int) -> dict:
    return {
        "subfolder": f"c2c_ime/{eid}",
        "type": "temp",
        "count": b,
        "width": w,
        "height": h,
        "version": fp[:16],
    }


def _delete_from_index(ed_dir: Path, start: int) -> None:
    if not ed_dir.is_dir():
        return
    for child in ed_dir.iterdir():
        if not child.is_file():
            continue
        name = child.name
        m = _FRAME_PNG.match(name)
        if m and int(m.group(1)) >= start:
            try:
                child.unlink()
            except OSError:
                pass
            continue
        m = _FRAME_THUMB.match(name)
        if m and int(m.group(1)) >= start:
            try:
                child.unlink()
            except OSError:
                pass


def _atomic_replace(tmp: Path, final: Path) -> None:
    os.replace(tmp, final)


def _write_png_atomic(rgb: np.ndarray, path: Path) -> None:
    tmp = path.parent / f".{path.name}.tmp"
    Image.fromarray(rgb, mode="RGB").save(
        tmp, format="PNG", compress_level=1, optimize=False,
    )
    _atomic_replace(tmp, path)


def _thumb_u8(image: torch.Tensor, i: int) -> Optional[np.ndarray]:
    """L9.169: the thumbnail through the shared preview encoder (area downscale of the float frame on its own device,
    rounded uint8); None -> the caller falls back to resizing the full-size uint8 frame."""
    try:
        from .._c2c_preview_encode import preview_frames
    except ImportError:  # pragma: no cover - loaded outside the package
        return None
    frames, _why = preview_frames(image[..., :3], indices=[i], long_edge=_THUMB_MAX)
    return frames[0] if frames else None


def _write_thumb_atomic(rgb: np.ndarray, path: Path, thumb: Optional[np.ndarray] = None) -> None:
    if thumb is not None:
        im = Image.fromarray(thumb, mode="RGB")
    else:
        im = Image.fromarray(rgb, mode="RGB")
        w, h = im.size
        scale = _THUMB_MAX / max(w, h)
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    tmp = path.parent / f".{path.name}.tmp"
    im.save(tmp, format="JPEG", quality=_THUMB_QUALITY, optimize=False)
    _atomic_replace(tmp, path)


def record_input_frames(
    image: torch.Tensor,
    editor_id: str,
) -> tuple[Optional[dict], list[str]]:
    """Write input frames to temp storage; return ComfyUI view payload or None."""
    notes: list[str] = []
    try:
        eid = validate_editor_id(editor_id)
    except ValueError:
        return None, notes

    if Image is None:
        return None, ["frame recording unavailable (Pillow missing)"]

    try:
        B, H, W, _C = image.shape
        if B <= 0 or H <= 0 or W <= 0:
            return None, notes

        if B * H * W > RECORD_BUDGET_PIXELS:
            notes.append(
                f"input too large to record for the editor ({B} frames of {W}x{H}); "
                "wire a C2C/VHS loader directly to edit exact frames"
            )
            return None, notes

        ed_dir = _temp_editor_dir(eid)
        qk = _quick_key(image)
        cached = _FINGERPRINT_CACHE.get(eid)
        if cached and len(cached) > 4 and _same_batch(cached[4], image) and _files_complete(ed_dir, B):
            fp = cached[0]                                    # same tensor, untouched: no full hash needed
            _cache_touch(eid, fp, B, H, W, qk)
            _delete_from_index(ed_dir, B)
            return _payload(eid, fp, B, W, H), notes

        fp = _fingerprint_stream(image, B, H, W)
        if cached and cached[0] == fp and _files_complete(ed_dir, B):
            _cache_touch(eid, fp, B, H, W, qk)
            _delete_from_index(ed_dir, B)
            return _payload(eid, fp, B, W, H), notes

        ed_dir.mkdir(parents=True, exist_ok=True)

        def _write(i: int) -> None:
            rgb = _frame_rgb_u8(image, i)
            _write_png_atomic(rgb, ed_dir / f"{i:05d}.png")
            _write_thumb_atomic(rgb, ed_dir / f"t_{i:05d}.jpg", _thumb_u8(image, i))

        # the full-size PNGs stay lossless (the editor paints on them); each worker holds one frame at a time
        if _WRITE_WORKERS > 1 and B > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=_WRITE_WORKERS) as pool:
                for _ in pool.map(_write, range(B)):         # re-raises the first write error
                    pass
        else:
            for i in range(B):
                _write(i)

        _delete_from_index(ed_dir, B)
        _cache_touch(eid, fp, B, H, W, qk)
        return _payload(eid, fp, B, W, H), notes

    except Exception as exc:  # noqa: BLE001 — must never break mask output
        notes.append(f"frame recording failed: {exc}")
        return None, notes
