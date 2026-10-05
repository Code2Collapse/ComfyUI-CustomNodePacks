"""Frame planning + on-demand thumbnails for VideoMaskEditor (L2.03)."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import OrderedDict
from typing import Any

import numpy as np

from .handle import C2CVideo
from .nodes_load import (
    build_images_handle,
    build_loaded_handle,
    is_changed_sequence,
    resolve_probe,
    source_times_for_handle,
    strip_path,
)
from .probe import probe_file
from .reader import read

_VIDEO_UPLOAD_TYPES = frozenset({"LoadVideoC2C", "VHS_LoadVideo"})
_VIDEO_PATH_TYPES = frozenset({"LoadVideoPathC2C", "VHS_LoadVideoPath"})
_IMAGES_TYPES = frozenset({"LoadImagesPathC2C", "VHS_LoadImagesPath"})
_LOAD_IMAGE_TYPES = frozenset({"LoadImage"})
_SUPPORTED_TYPES = _VIDEO_UPLOAD_TYPES | _VIDEO_PATH_TYPES | _IMAGES_TYPES | _LOAD_IMAGE_TYPES

_HANDLE_CACHE: OrderedDict[str, tuple[C2CVideo, C2CVideo]] = OrderedDict()
_HANDLE_CACHE_MAX = 32

_THUMB_LRU: OrderedDict[str, bytes] = OrderedDict()
_THUMB_BYTES = 0
_THUMB_BUDGET = 64 * 1024 * 1024


class PlanError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = int(status)
        self.message = str(message)
        super().__init__(message)


def _int_widget(widgets: dict[str, Any], key: str, default: int = 0) -> int:
    val = widgets.get(key, default)
    try:
        return int(val)
    except (TypeError, ValueError):
        raise PlanError(400, f"Invalid value for {key}.")


def _float_widget(widgets: dict[str, Any], key: str, default: float = 0.0) -> float:
    val = widgets.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        raise PlanError(400, f"Invalid value for {key}.")


def _canonical_widgets(widgets: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(widgets):
        val = widgets[key]
        if isinstance(val, str):
            out[key] = strip_path(val)
        elif isinstance(val, (int, float, bool)) or val is None:
            out[key] = val
        else:
            out[key] = str(val)
    return out


def _media_stat_key(path: str) -> tuple[int, int]:
    abspath = os.path.abspath(strip_path(path))
    if os.path.isdir(abspath):
        digest = is_changed_sequence(abspath)
        return (0, int(hashlib.sha1(digest.encode("utf-8")).hexdigest()[:16], 16))
    if os.path.isfile(abspath):
        st = os.stat(abspath)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        return (int(st.st_size), mtime_ns)
    raise PlanError(404, f"Nothing found at {abspath}.")


def _resolve_path_for_type(node_type: str, widgets: dict[str, Any]) -> str:
    """Same path rules as the probe route (media only, upload names confined to the input
    folder); the shared resolver's RouteError becomes this module's PlanError."""
    from .routes import RouteError

    try:
        return _resolve_path_unchecked(node_type, widgets)
    except RouteError as exc:
        raise PlanError(exc.status, exc.message) from exc


def _resolve_path_unchecked(node_type: str, widgets: dict[str, Any]) -> str:
    from .routes import resolve_media_path

    if node_type in _VIDEO_UPLOAD_TYPES:
        video = strip_path(str(widgets.get("video") or ""))
        if not video:
            raise PlanError(400, "Choose a video from the input folder.")
        return resolve_media_path({"filename": video, "type": "input"})
    if node_type in _VIDEO_PATH_TYPES:
        video = strip_path(str(widgets.get("video") or ""))
        if not video:
            raise PlanError(400, "Enter a video file, a folder of images, or a sequence pattern.")
        return resolve_media_path({"path": video})
    if node_type in _IMAGES_TYPES:
        directory = strip_path(str(widgets.get("directory") or ""))
        if not directory:
            raise PlanError(400, "Enter a directory of sequential images.")
        return resolve_media_path({"mode": "images", "directory": directory})
    if node_type in _LOAD_IMAGE_TYPES:
        image = strip_path(str(widgets.get("image") or ""))
        if not image:
            raise PlanError(400, "Choose an image from the input folder.")
        return resolve_media_path({"filename": image, "type": "input"})
    raise PlanError(400, f"Unsupported node type: {node_type}.")


def _build_handles(node_type: str, widgets: dict[str, Any]) -> tuple[C2CVideo, C2CVideo]:
    path = _resolve_path_for_type(node_type, widgets)

    if node_type in _LOAD_IMAGE_TYPES:
        raw = probe_file(path)
        return raw, raw

    if node_type in _IMAGES_TYPES:
        raw_dir = path
        from .probe import probe_sequence

        raw = probe_sequence(raw_dir)
        n_before = len(raw)
        loaded = build_images_handle(
            raw_dir,
            skip_first_images=_int_widget(widgets, "skip_first_images", 0),
            select_every_nth=max(1, _int_widget(widgets, "select_every_nth", 1)),
            image_load_cap=_int_widget(widgets, "image_load_cap", 0),
        )
        if len(loaded) == 0:
            raise PlanError(
                400,
                f"Load Images Path: no frames selected - the clip has {n_before} frames "
                f"and skip_first_images={_int_widget(widgets, 'skip_first_images', 0)} leaves none.",
            )
        return raw, loaded

    sequence_fps = _float_widget(widgets, "sequence_fps", 24.0)
    raw = resolve_probe(path, sequence_fps=sequence_fps)
    force_rate = _float_widget(widgets, "force_rate", 0.0)
    skip = _int_widget(widgets, "skip_first_frames", 0)
    h = raw
    if force_rate > 0:
        h = h.retimed(force_rate, source_times_for_handle(raw))
    n_before = len(h)
    _, loaded = build_loaded_handle(
        raw,
        force_rate=force_rate,
        skip_first_frames=skip,
        select_every_nth=max(1, _int_widget(widgets, "select_every_nth", 1)),
        frame_load_cap=_int_widget(widgets, "frame_load_cap", 0),
        format_name=str(widgets.get("format") or "None"),
        custom_width=_int_widget(widgets, "custom_width", 0),
        custom_height=_int_widget(widgets, "custom_height", 0),
        vae=None,
    )
    if len(loaded) == 0:
        retime_note = " after retiming" if force_rate > 0 else ""
        raise PlanError(
            400,
            f"Load Video: no frames selected - the clip has {n_before} frames"
            f"{retime_note} and skip_first_frames={skip} leaves none.",
        )
    return raw, loaded


def _make_token(node_type: str, widgets: dict[str, Any], path: str) -> str:
    canon = _canonical_widgets(widgets)
    size, mtime = _media_stat_key(path)
    blob = json.dumps(
        {"node_type": node_type, "widgets": canon, "size": size, "mtime": mtime},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _cache_handle(token: str, raw: C2CVideo, loaded: C2CVideo) -> None:
    if token in _HANDLE_CACHE:
        _HANDLE_CACHE.move_to_end(token)
    else:
        _HANDLE_CACHE[token] = (raw, loaded)
        while len(_HANDLE_CACHE) > _HANDLE_CACHE_MAX:
            _HANDLE_CACHE.popitem(last=False)


def get_cached_handle(token: str) -> tuple[C2CVideo, C2CVideo]:
    try:
        entry = _HANDLE_CACHE[token]
    except KeyError:
        raise PlanError(400, "Frame plan expired or unknown. Re-open the editor or reconnect the loader.")
    _HANDLE_CACHE.move_to_end(token)
    return entry


def plan_frames(node_type: str, widgets: dict[str, Any]) -> dict[str, Any]:
    node_type = str(node_type or "").strip()
    if node_type not in _SUPPORTED_TYPES:
        raise PlanError(400, f"Unsupported node type: {node_type or '(empty)'}.")

    if not isinstance(widgets, dict):
        raise PlanError(400, "Widgets must be a JSON object.")

    path = _resolve_path_for_type(node_type, widgets)
    raw, loaded = _build_handles(node_type, widgets)
    token = _make_token(node_type, widgets, path)
    _cache_handle(token, raw, loaded)

    if node_type in _LOAD_IMAGE_TYPES:
        count = 1
        fps = float(raw.fps) if float(raw.fps) > 0 else 0.0
        width, height = int(raw.width), int(raw.height)
    else:
        count = len(loaded)
        fps = float(loaded.fps)
        width, height = int(loaded.width), int(loaded.height)

    return {
        "count": count,
        "width": width,
        "height": height,
        "fps": fps,
        "token": token,
        "node_type": node_type,
    }


def _scale_rgb(rgb: np.ndarray, max_px: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max_px <= 0 or max(h, w) <= max_px:
        return rgb
    if w >= h:
        nw = max(2, int(max_px))
        nh = max(2, int(round(h * max_px / w)))
    else:
        nh = max(2, int(max_px))
        nw = max(2, int(round(w * max_px / h)))
    nw = max(2, nw // 2 * 2)
    nh = max(2, nh // 2 * 2)
    try:
        from PIL import Image

        img = Image.fromarray(rgb, mode="RGB")
        resample = getattr(Image, "Resampling", Image).BILINEAR
        img = img.resize((nw, nh), resample)
        return np.asarray(img, dtype=np.uint8)
    except ImportError:
        import cv2

        return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def _encode_rgb(rgb: np.ndarray, fmt: str) -> bytes:
    fmt = (fmt or "jpeg").lower()
    if fmt not in ("jpeg", "jpg", "png"):
        raise PlanError(400, "fmt must be jpeg or png.")
    try:
        from PIL import Image

        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        if fmt == "png":
            img.save(buf, format="PNG")
        else:
            img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except ImportError:
        import cv2

        ext = ".png" if fmt == "png" else ".jpg"
        ok, enc = cv2.imencode(ext, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise PlanError(500, "Could not encode thumbnail.")
        return enc.tobytes()


def _thumb_cache_get(key: str) -> bytes | None:
    data = _THUMB_LRU.get(key)
    if data is not None:
        _THUMB_LRU.move_to_end(key)
    return data


def _thumb_cache_put(key: str, data: bytes) -> None:
    global _THUMB_BYTES
    old = _THUMB_LRU.pop(key, None)
    if old is not None:
        _THUMB_BYTES -= len(old)
    _THUMB_LRU[key] = data
    _THUMB_BYTES += len(data)
    while _THUMB_BYTES > _THUMB_BUDGET and _THUMB_LRU:
        _, evicted = _THUMB_LRU.popitem(last=False)
        _THUMB_BYTES -= len(evicted)


def thumb_cache_stats() -> tuple[int, int]:
    """Return (entry_count, total_bytes) for tests."""
    return len(_THUMB_LRU), _THUMB_BYTES


def reset_thumb_cache() -> None:
    global _THUMB_BYTES
    _THUMB_LRU.clear()
    _THUMB_BYTES = 0


def encode_thumb(token: str, index: int, max_px: int = 384, fmt: str = "jpeg") -> bytes:
    _, loaded = get_cached_handle(token)
    i = int(index)
    n = len(loaded)
    if i < 0 or i >= n:
        raise PlanError(400, f"Frame index {i} is out of range (0..{n - 1}).")

    max_px = int(max_px)
    cache_key = f"{token}|{i}|{max_px}|{(fmt or 'jpeg').lower()}"
    cached = _thumb_cache_get(cache_key)
    if cached is not None:
        return cached

    one = loaded.trimmed(i, i + 1)
    batch = read(one, [0], fmt="uint8")
    rgb = np.ascontiguousarray(batch[0, ..., :3])
    rgb = _scale_rgb(rgb, max_px)
    data = _encode_rgb(rgb, fmt)
    _thumb_cache_put(cache_key, data)
    return data
