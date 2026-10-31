"""HTTP routes for C2C video loader probe + H.264 preview proxies."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from fractions import Fraction
from typing import Any, Mapping

import numpy as np

from . import media_roots as ROOTS
from .handle import C2CVideo
from .nodes_load import (
    SEQUENCE_IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    audio_output_mode,
    build_images_handle,
    build_loaded_handle,
    is_sequence_path,
    load_plan,
    load_summary,
    media_summary,
    resolve_probe,
    source_times_for_handle,
    strip_path,
    validate_images_directory,
    validate_video_path,
)
from .probe import resolve_sequence
from .reader import iter_chunks, read

log = logging.getLogger("c2c_video.routes")

_ROUTES_REGISTERED = False
# Our own bounded pools, not the event loop's shared default executor: at
# boot 100+ packs park blocking work there, and a probe queued behind them
# never answered (measured). Probes are quick and must stay responsive;
# one preview encode at a time keeps a low-end machine usable.
_PROBE_POOL = None
_PREVIEW_POOL = None
_FRAMES_POOL = None


def _pools():
    global _PROBE_POOL, _PREVIEW_POOL, _FRAMES_POOL
    if _PROBE_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _PROBE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="c2c-video-probe")
        _PREVIEW_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="c2c-video-preview")
        _FRAMES_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="c2c-frames")
    return _PROBE_POOL, _PREVIEW_POOL, _FRAMES_POOL
_PREVIEW_LOCKS: dict[str, threading.Lock] = {}
_PREVIEW_LOCK_GUARD = threading.Lock()
_encode_calls = 0

_PREVIEW_CHUNK = 8
_DEFAULT_MAX_FRAMES = 240
_PREVIEW_MAX_LONG = 640
_MAX_FRAMES_CAP = 2000  # a preview, not an export: bound what one request can decode


class RouteError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = int(status)
        self.message = str(message)
        super().__init__(message)


def _import_folder_paths():
    import folder_paths  # noqa: WPS433

    return folder_paths


def _int_param(raw: Mapping[str, str], key: str, default: int = 0) -> int:
    val = raw.get(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        raise RouteError(400, f"Invalid value for {key}.")


def _float_param(raw: Mapping[str, str], key: str, default: float = 0.0) -> float:
    val = raw.get(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        raise RouteError(400, f"Invalid value for {key}.")


def parse_request_params(raw: Mapping[str, str]) -> dict[str, Any]:
    params: dict[str, Any] = dict(raw)
    if params.get("directory"):
        params["mode"] = "images"
        params["skip_first_images"] = _int_param(raw, "skip_first_images", 0)
        params["image_load_cap"] = _int_param(raw, "image_load_cap", 0)
        params["select_every_nth"] = _int_param(raw, "select_every_nth", 1)
    else:
        params["mode"] = "video"
        params["force_rate"] = _float_param(raw, "force_rate", 0.0)
        params["skip_first_frames"] = _int_param(raw, "skip_first_frames", 0)
        params["select_every_nth"] = _int_param(raw, "select_every_nth", 1)
        params["frame_load_cap"] = _int_param(raw, "frame_load_cap", 0)
        params["format"] = raw.get("format") or "None"
        params["sequence_fps"] = _float_param(raw, "sequence_fps", 24.0)
        # L2.53: the rest of the node's settings, so the probe reports exactly what a run loads (all optional; the
        # defaults are the node's, so an older front end gets the old answer)
        params["seconds_cap"] = _float_param(raw, "seconds_cap", 0.0)
        params["custom_width"] = _int_param(raw, "custom_width", 0)
        params["custom_height"] = _int_param(raw, "custom_height", 0)
        params["resolution"] = raw.get("resolution") or "custom"
        params["divisible_by"] = max(1, _int_param(raw, "divisible_by", 1))
        params["decoder"] = raw.get("decoder") or "auto"
        params["audio_when_missing"] = raw.get("audio_when_missing") or "silence"
        params["crop"] = tuple(_float_param(raw, k, d) for k, d in
                               (("crop_x", 0.0), ("crop_y", 0.0), ("crop_w", 1.0), ("crop_h", 1.0)))
        # L9.52: the handle settings, so the node shows what a run will add BEFORE it runs
        params["handles"] = raw.get("handles") or "off"
        params["handle_frames"] = max(0, _int_param(raw, "handle_frames", 8))
        params["handle_target"] = max(0, _int_param(raw, "handle_target", 0))
        params["handle_side"] = raw.get("handle_side") or "head"
        params["handle_grid"] = raw.get("handle_grid") or "from format"
        params["handle_custom_step"] = max(1, _int_param(raw, "handle_custom_step", 4))
        params["handle_custom_plus"] = max(0, _int_param(raw, "handle_custom_plus", 1))
        params["extra_handles"] = max(0, _int_param(raw, "extra_handles", 0))          # L9.56
        # L9.89: the unified loader's inputs (absent = the old answer: auto, everything, skip as the Path node did)
        params["location"] = raw.get("location") or "input folder"
        params["source_mode"] = raw.get("source_mode") or "auto"
        params["frame_range"] = raw.get("frame_range") or ""
        params["reverse"] = str(raw.get("reverse") or "").strip().lower() in ("1", "true", "yes")
        params["frame_offset"] = _int_param(raw, "frame_offset", 0)
        params["missing_frames"] = raw.get("missing_frames") or "skip (as before)"
        params["edge_policy"] = raw.get("edge_policy") or "hold"
        params["exr_layer"] = raw.get("exr_layer") or "auto"
    params["max_frames"] = min(_MAX_FRAMES_CAP, max(1, _int_param(raw, "max_frames", _DEFAULT_MAX_FRAMES)))
    return params


def _is_media_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext.lstrip(".") in VIDEO_EXTENSIONS or ext in SEQUENCE_IMAGE_EXTENSIONS


def _roots_or_403(path: str) -> None:
    """D0.35 (owner 2026-10-31): a browser-facing route reads only inside input / output / temp + the media roots."""
    blocked = ROOTS.check_route_path(path)
    if blocked:
        raise RouteError(403, blocked)


def resolve_media_path(params: dict[str, Any]) -> str:
    if params.get("mode") == "images":
        directory = strip_path(str(params.get("directory") or ""))
        valid = validate_images_directory(directory)
        if valid is not True:
            raise RouteError(404, valid if isinstance(valid, str) else f"Nothing found at {directory}.")
        _roots_or_403(os.path.abspath(directory))
        return os.path.abspath(directory)

    if params.get("path"):
        path = strip_path(str(params["path"]))
        valid = validate_video_path(path)
        if valid is not True:
            raise RouteError(404, valid if isinstance(valid, str) else f"Nothing found at {path}.")
        path = os.path.abspath(path)
        _roots_or_403(path)
        # This is an HTTP route: it reads nothing but media. Folders and
        # patterns only ever resolve to image-sequence files.
        if os.path.isfile(path) and not _is_media_file(path):
            raise RouteError(400, f"{os.path.basename(path)} is not a video or image file.")
        return path

    filename = params.get("filename")
    if not filename:
        raise RouteError(400, "Provide path, directory, or filename.")
    fp = _import_folder_paths()
    sub = (params.get("subfolder") or "").strip()
    name = strip_path(str(filename))
    annotated = os.path.join(sub, name) if sub else name
    try:
        path = fp.get_annotated_filepath(annotated)
    except Exception:
        path = fp.get_annotated_filepath(name)
    if not path or not os.path.exists(path):
        # L9.89: a listed sequence pattern (plates/shot.####.exr) is not a file, but resolves inside input/
        found = False
        if path and is_sequence_path(path):
            try:
                found = bool(resolve_sequence(path))
            except (OSError, FileNotFoundError, ValueError):
                found = False
        if not found:
            raise RouteError(404, f"Nothing found at {name}.")
    path = os.path.abspath(path)
    # an upload name must stay inside ComfyUI's input folder ("../" escapes it)
    base = os.path.abspath(fp.get_input_directory())
    try:
        inside = os.path.commonpath([base, path]) == base
    except ValueError:  # different drives on Windows
        inside = False
    if not inside or not _is_media_file(path):
        raise RouteError(400, f"{name} is not a video in the input folder.")
    return path


def plan_from_params(params: dict[str, Any], *, full: bool = False):
    """The run's own load_plan for a probe / preview request. ``full``: with the node's size and crop (what a run
    loads); without, the whole frame (what the preview shows). Engine refusals come back as 400s in words."""
    path = resolve_media_path(params)
    try:
        return load_plan(
            path,
            source_mode=str(params.get("source_mode") or "auto"),
            exr_layer=str(params.get("exr_layer") or "auto"),
            sequence_fps=float(params.get("sequence_fps", 24)),
            frame_range=str(params.get("frame_range") or ""),
            missing_frames=str(params.get("missing_frames") or "skip (as before)"),
            edge_policy=str(params.get("edge_policy") or "hold"),
            reverse=bool(params.get("reverse")),
            force_rate=float(params.get("force_rate", 0)),
            skip_first_frames=int(params.get("skip_first_frames", 0)),
            select_every_nth=int(params.get("select_every_nth", 1)),
            frame_load_cap=int(params.get("frame_load_cap", 0)),
            format_name=str(params.get("format") or "None"),
            custom_width=int(params.get("custom_width", 0)) if full else 0,
            custom_height=int(params.get("custom_height", 0)) if full else 0,
            vae=None,
            crop=params.get("crop") if full else None,
            seconds_cap=float(params.get("seconds_cap", 0.0) or 0.0),
            resolution=str(params.get("resolution") or "custom") if full else "custom",
            divisible_by=int(params.get("divisible_by", 1) or 1) if full else 1,
            snap_frames=str(params.get("handles") or "off") != "fit" if full else True,
        )
    except PermissionError as exc:
        raise RouteError(403, str(exc)) from exc
    except FileNotFoundError as exc:
        raise RouteError(404, str(exc)) from exc
    except ValueError as exc:
        raise RouteError(400, str(exc)) from exc


def build_handle_from_params(params: dict[str, Any]) -> tuple[C2CVideo, C2CVideo]:
    if params.get("mode") == "images":
        directory = resolve_media_path(params)
        from .probe import probe_sequence

        raw = probe_sequence(directory)
        n_before = len(raw)
        loaded = build_images_handle(
            directory,
            skip_first_images=int(params["skip_first_images"]),
            select_every_nth=int(params["select_every_nth"]),
            image_load_cap=int(params["image_load_cap"]),
        )
        if len(loaded) == 0:
            raise RouteError(
                400,
                f"Load Images Path (C2C): no frames selected - the clip has {n_before} frames "
                f"and skip_first_images={int(params['skip_first_images'])} leaves none.",
            )
        return raw, loaded

    plan = plan_from_params(params)
    if len(plan.loaded) == 0:
        retime_note = " after retiming" if float(params.get("force_rate", 0)) > 0 else ""
        raise RouteError(
            400,
            f"Load Video (C2C): no frames selected - the clip has {plan.n_before_skip} frames"
            f"{retime_note} and skip_first_frames={int(params.get('skip_first_frames', 0))} leaves none.",
        )
    return plan.raw, plan.loaded


def _codec_name(raw: C2CVideo) -> str:
    if raw.source.kind == "file":                          # by kind, not class (vendored copies)
        import av

        try:
            with av.open(raw.source.path) as container:
                vs = container.streams.video[raw.stream_index]
                name = getattr(vs.codec_context.codec, "name", None) or vs.codec_context.name
                return str(name or "unknown")
        except Exception as exc:
            raise RouteError(500, f"Could not read codec information: {exc}") from exc
    if raw.source.kind == "sequence" and raw.source.files:
        ext = os.path.splitext(raw.source.files[0])[1].lower().lstrip(".")
        return ext or "sequence"
    return "unknown"


def _colour_dict(raw: C2CVideo) -> dict[str, str | None]:
    c = raw.colour
    return {
        "matrix": c.matrix,
        "range": c.range,
        "primaries": c.primaries,
        "transfer": c.transfer,
    }


def _load_report(raw: C2CVideo, params: dict[str, Any]) -> dict[str, Any] | None:
    """What a run with these settings loads - size after crop / resolution / divisible_by, the part of the clip in
    seconds, the reader auto would use, the audio - for the node's grey hints and details line (L2.53)."""
    if params.get("mode") == "images":
        return None
    from .fast_decode import device_is_cuda, plan_decoder

    load = plan_from_params(params, full=True).loaded          # as a run: size, crop, handles' grid rule (L9.89)
    if not len(load):
        return None
    plan = plan_decoder(load, str(params.get("decoder") or "auto"), cuda=device_is_cuda())
    summary = load_summary(raw, load, format_name=str(params.get("format") or "None"), plan=plan,
                           audio_mode=audio_output_mode(load, str(params.get("audio_when_missing") or "silence")))
    return {**summary["loaded"], "decoder": summary.get("decoder"), "audio": summary["audio"],
            "handles": _handles_plan(len(load), params)}


def _handles_plan(n: int, params: dict[str, Any]) -> dict[str, Any] | None:
    """What the run's handles will add to these n frames (None when handles are off) - the same planner the run uses."""
    from . import handles_io as HIO
    from .nodes_load import FORMAT_PRESETS

    mode = str(params.get("handles") or "off")
    if mode not in ("fit", "add") or HIO.disabled():
        return None
    fmt = str(params.get("format") or "None")
    try:
        model, step, plus, label = HIO.grid_for(str(params.get("handle_grid") or HIO.FROM_FORMAT),
                                                FORMAT_PRESETS.get(fmt, {}), fmt,
                                                int(params.get("handle_custom_step") or 4),
                                                int(params.get("handle_custom_plus") or 0))
        out = HIO.plan_handles(n, mode=mode, frames=int(params.get("handle_frames") or 0),
                               target=int(params.get("handle_target") or 0),
                               side=str(params.get("handle_side") or "head"), model=model, step=step, plus=plus,
                               extra_steps=int(params.get("extra_handles") or 0))
    except Exception as exc:  # noqa: BLE001 - shown in the node, the run reports the same error
        return {"error": True, "note": str(exc)}
    return dict(out, grid=label, frames_in=int(n))


def _colour_plan(raw: C2CVideo, params: dict[str, Any]) -> dict[str, Any] | None:
    """L9.62: what Load Video's colour management will do with these settings - before a run (no decode: the pixel
    check runs with the frames)."""
    from . import colour_cm as CM
    c = CM.plan(raw, str(params.get("colorspace_in") or CM.CS_OFF), str(params.get("working_space") or CM.WORK_DEFAULT))
    return None if c is None else c.record


def probe_payload(params: dict[str, Any]) -> dict[str, Any]:
    raw, loaded = build_handle_from_params(params)
    media = None
    if params.get("mode") != "images":
        try:
            plan = plan_from_params(params)
            media = media_summary(plan.raw, plan.loaded, plan.facts, plan.timeline,
                                  data_layer=plan.facts.layer_kind not in ("", "colour", "alpha"))
        except RouteError:
            raise
        except Exception as exc:  # noqa: BLE001 - the probe still answers; the media chips stay empty
            log.debug("media summary failed: %s", exc)
    sel = list(loaded.selected_indices())
    src_fps = float(raw.fps)
    try:
        load = _load_report(raw, params)
    except Exception as exc:  # noqa: BLE001 - the probe still answers; the hints just stay empty
        log.debug("load report failed: %s", exc)
        load = None
    try:
        cm = _colour_plan(raw, params)
    except Exception as exc:  # noqa: BLE001 - shown in the node; the run reports the same
        cm = {"error": str(exc)}
    return {
        "file": os.path.basename(raw.source.path) if raw.source.kind == "file" else raw.source.pattern,
        "media": media,                                   # L9.89: kind, pattern, numbers, missing, layers, warnings
        "load": load,
        "cm": cm,
        "kind": raw.source.kind,
        "width": int(raw.width),
        "height": int(raw.height),
        "fps": src_fps,
        "frame_count": int(raw.frame_count),
        "duration": int(raw.frame_count) / src_fps if src_fps > 0 else 0.0,
        "pix_fmt": raw.pix_fmt,
        "bit_depth": int(raw.bit_depth),
        "has_alpha": bool(raw.has_alpha),
        "colour": _colour_dict(raw),
        "rotation": int(raw.rotation),
        "audio": (
            {"sample_rate": int(raw.audio.sample_rate), "channels": int(raw.audio.channels)}
            if raw.audio is not None
            else None
        ),
        "codec": _codec_name(raw),
        "index_mode": raw.index_mode,
        "selection": {
            "count": len(loaded),
            "fps": float(loaded.fps),
            "first_source": int(sel[0]) if sel else 0,
            "last_source": int(sel[-1]) if sel else 0,
            "source_frame_count": int(raw.frame_count),
        },
    }


def _preview_scale(w: int, h: int, max_long: int = _PREVIEW_MAX_LONG) -> tuple[int, int]:
    w, h = max(1, int(w)), max(1, int(h))
    if w >= h:
        nw = max_long
        nh = max(2, int(round(h * max_long / w)))
    else:
        nh = max_long
        nw = max(2, int(round(w * max_long / h)))
    nw = max(2, nw // 2 * 2)
    nh = max(2, nh // 2 * 2)
    return nw, nh


def _cache_dir() -> str:
    fp = _import_folder_paths()
    path = os.path.join(fp.get_temp_directory(), "c2c_video_preview")
    os.makedirs(path, exist_ok=True)
    return path


# Bumped when the preview encoding changes, so older cached previews are not served (2: short GOP, L2.53).
_PREVIEW_VERSION = 2
# A keyframe every 12 frames: the player seeks, scrubs, shows hover thumbnails and plays backwards by seeking, and
# each seek decodes from the previous keyframe (x264's default of 250 made a seek decode up to 250 frames).
_PREVIEW_GOP = 12


def _cache_key(loaded: C2CVideo, max_frames: int, out_w: int, out_h: int, ext: str) -> str:
    # transfer: the same frames shown as data (normalised) or as a picture are two previews
    blob = f"v{_PREVIEW_VERSION}|{loaded.fingerprint()}|{max_frames}|{out_w}x{out_h}|{ext}|{loaded.colour.transfer}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _lock_for(key: str) -> threading.Lock:
    with _PREVIEW_LOCK_GUARD:
        if key not in _PREVIEW_LOCKS:
            _PREVIEW_LOCKS[key] = threading.Lock()
        return _PREVIEW_LOCKS[key]


def _is_linear(raw: C2CVideo) -> bool:
    return (raw.colour.transfer or "").lower() == "linear"


def _is_data(raw: C2CVideo) -> bool:
    return (raw.colour.transfer or "").lower() == "data"


def _data_to_rgb_uint8(chunk: np.ndarray) -> np.ndarray:
    """L9.89: a data layer (depth, normals, position, motion) for the EYE only: each frame stretched between its 1st
    and 99th percentile. The loaded values are never touched (this is the preview's copy)."""
    out = np.empty(chunk.shape[:3] + (3,), dtype=np.uint8)
    for i, f in enumerate(chunk[..., :3].astype(np.float32)):
        finite = f[np.isfinite(f)]
        lo, hi = (np.percentile(finite, 1), np.percentile(finite, 99)) if finite.size else (0.0, 1.0)
        span = hi - lo if hi > lo else 1.0
        out[i] = (np.clip((np.nan_to_num(f) - lo) / span, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return out


def _chunk_to_rgb_uint8(chunk: np.ndarray, linear: bool, data: bool = False) -> np.ndarray:
    if data:
        return _data_to_rgb_uint8(chunk)
    if linear:
        # simple sRGB view — proper OCIO comes later
        rgb = np.clip(chunk[..., :3].astype(np.float32), 0.0, 1.0)
        rgb = np.power(rgb, 1.0 / 2.2)
        return (np.clip(rgb * 255.0, 0, 255)).astype(np.uint8)
    if chunk.dtype == np.uint8:
        # RGBA sources (ProRes 4444) give a strided view; the encoder wants rows
        return np.ascontiguousarray(chunk[..., :3])
    return (np.clip(chunk[..., :3], 0, 255)).astype(np.uint8)


def _pick_encoder() -> tuple[str, str]:
    import av

    for codec, ext in (("libx264", "mp4"), ("libopenh264", "mp4"), ("libvpx-vp9", "webm")):
        try:
            av.Codec(codec, "w")
            return codec, ext
        except Exception:
            continue
    raise RouteError(500, "No H.264 or WebM encoder is available to build a preview.")


def _encode_preview(loaded: C2CVideo, raw: C2CVideo, out_path: str, max_frames: int) -> None:
    global _encode_calls
    import av

    pw, ph = _preview_scale(loaded.width, loaded.height)
    scaled = loaded.scaled(pw, ph)
    n = min(len(scaled), max(1, int(max_frames)))
    capped = scaled.capped(n) if n < len(scaled) else scaled
    linear = _is_linear(raw)
    data = _is_data(raw)
    fmt = "float32" if (linear or data) else "uint8"

    codec_name, _ext = _pick_encoder()
    fps = float(capped.fps) or 24.0
    # faststart is a MUXER option: the index goes first so the browser can
    # start playing before the whole file has arrived
    container = av.open(out_path, mode="w", format=_ext,
                        options={"movflags": "+faststart"} if _ext == "mp4" else {})
    try:
        stream = container.add_stream(codec_name, rate=Fraction(fps).limit_denominator(1_000_000))
        stream.width = pw
        stream.height = ph
        stream.pix_fmt = "yuv420p"
        from .testmedia import _rgb_to_yuv, _tag_bt709
        _tag_bt709(stream)                              # L9.67: BT.709 limited, said in the file
        stream.codec_context.gop_size = _PREVIEW_GOP
        if codec_name == "libx264":
            stream.options = {"crf": "23", "preset": "veryfast", "g": str(_PREVIEW_GOP)}
        written = 0
        for chunk in iter_chunks(capped, _PREVIEW_CHUNK, fmt=fmt):
            rgb_batch = _chunk_to_rgb_uint8(chunk, linear, data)
            for row in rgb_batch:
                if written >= n:
                    break
                frame = _rgb_to_yuv(row, "yuv420p")
                frame.pts = written
                for packet in stream.encode(frame):
                    container.mux(packet)
                written += 1
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    _encode_calls += 1


def ensure_preview(params: dict[str, Any]) -> str:
    raw, loaded = build_handle_from_params(params)
    max_frames = int(params.get("max_frames", _DEFAULT_MAX_FRAMES))
    pw, ph = _preview_scale(loaded.width, loaded.height)
    codec_name, ext = _pick_encoder()
    key = _cache_key(loaded, max_frames, pw, ph, ext)
    out_path = os.path.join(_cache_dir(), f"{key}.{ext}")
    if os.path.isfile(out_path):
        return out_path

    lock = _lock_for(key)
    with lock:
        if os.path.isfile(out_path):
            return out_path
        tmp = f"{out_path}.{os.getpid()}.tmp.{ext}"
        try:
            _encode_preview(loaded, raw, tmp, max_frames)
            os.replace(tmp, out_path)
        except Exception as exc:
            try:
                os.remove(tmp)
            except OSError:
                pass
            if isinstance(exc, RouteError):
                raise
            raise RouteError(500, f"Could not build video preview: {exc}") from exc
    return out_path


def still_png(params: dict[str, Any]) -> bytes:
    """PNG of one loaded position (``i``, default 0) at most ``max`` px on its long side, display-converted for the
    browser only: sRGB-ish view of scene-linear light, data layers normalised (Rev 3 3.7 / 3.12)."""
    import io

    from PIL import Image

    plan = plan_from_params(params)
    loaded = plan.loaded
    if not len(loaded):
        raise RouteError(400, "nothing selected to show")
    i = min(max(0, int(params.get("i", 0) or 0)), len(loaded) - 1)
    long_side = max(16, min(4096, int(params.get("max", 1024) or 1024)))
    pw, ph = _preview_scale(loaded.width, loaded.height, long_side)
    small = loaded.scaled(pw, ph) if max(loaded.width, loaded.height) > long_side else loaded
    raw = plan.raw
    linear, data = _is_linear(raw), _is_data(raw)
    arr = read(small, [i], fmt="float32" if (linear or data) else "uint8")
    rgb = _chunk_to_rgb_uint8(arr, linear, data)[0]
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG")
    return buf.getvalue()


def register_routes() -> bool:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception as exc:
        log.debug("c2c_video routes skipped: %s", exc)
        return False

    inst = getattr(PromptServer, "instance", None)
    routes = getattr(inst, "routes", None) if inst else None
    if routes is None or not hasattr(routes, "get"):
        return False

    async def _handle_probe(request):
        import asyncio

        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(
                _pools()[0], probe_payload, parse_request_params(dict(request.query)),
            )
            return web.json_response(data)
        except RouteError as exc:
            return web.json_response({"error": exc.message}, status=exc.status)
        except Exception as exc:
            log.exception("probe failed")
            return web.json_response(
                {"error": f"Could not probe video: {exc}"},
                status=500,
            )

    async def _handle_preview(request):
        import asyncio

        loop = asyncio.get_running_loop()
        try:
            path = await loop.run_in_executor(
                _pools()[1], ensure_preview, parse_request_params(dict(request.query)),
            )
            return web.FileResponse(path)
        except RouteError as exc:
            return web.json_response({"error": exc.message}, status=exc.status)
        except Exception as exc:
            log.exception("preview failed")
            return web.json_response(
                {"error": f"Could not build video preview: {exc}"},
                status=500,
            )

    async def _handle_frames_plan(request):
        import asyncio

        from .frames import PlanError, plan_frames

        loop = asyncio.get_running_loop()
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Request body must be JSON."}, status=400)
        try:
            data = await loop.run_in_executor(
                _pools()[2],
                plan_frames,
                str(body.get("node_type") or ""),
                body.get("widgets") if isinstance(body.get("widgets"), dict) else {},
            )
            return web.json_response(data)
        except PlanError as exc:
            return web.json_response({"error": exc.message}, status=exc.status)
        except Exception as exc:
            log.exception("frames plan failed")
            return web.json_response(
                {"error": f"Could not plan frames: {exc}"},
                status=500,
            )

    async def _handle_frames_thumb(request):
        import asyncio

        from .frames import PlanError, encode_thumb

        loop = asyncio.get_running_loop()
        q = dict(request.query)
        try:
            token = str(q.get("token") or "").strip()
            if not token:
                raise PlanError(400, "token is required.")
            index = _int_param(q, "i", -1)
            max_px = _int_param(q, "max", 384)
            fmt = str(q.get("fmt") or "jpeg").lower()
            data = await loop.run_in_executor(
                _pools()[2],
                encode_thumb,
                token,
                index,
                max_px,
                fmt,
            )
            ctype = "image/png" if fmt == "png" else "image/jpeg"
            return web.Response(body=data, content_type=ctype)
        except PlanError as exc:
            return web.json_response({"error": exc.message}, status=exc.status)
        except Exception as exc:
            log.exception("frames thumb failed")
            return web.json_response(
                {"error": f"Could not build thumbnail: {exc}"},
                status=500,
            )

    async def _handle_save_matrix(request):
        """L9.40: what Save Video (C2C) can write here - container -> codec -> profile -> bit depth / chroma, with the
        reason for every choice this machine cannot write. Probed once (ffmpeg CLI + PyAV), then cached."""
        import asyncio

        from .save_matrix import cached_matrix

        try:
            data = await asyncio.get_running_loop().run_in_executor(_pools()[2], cached_matrix)
            return web.json_response(data)
        except Exception as exc:
            log.exception("save matrix failed")
            return web.json_response({"error": f"Could not list the formats: {exc}"}, status=500)

    async def _handle_save_next_path(request):
        """L9.40: the path Save Video (C2C) will write next (dry run). Body: {prompt, unique_id, naming, format,
        container, codec, profile, bit_depth, chroma, filename_prefix, subfolder, save_output}."""
        import asyncio

        from .save_matrix import Choice, DEFAULT_CHOICE, resolve
        from .save_naming import predict_path

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Request body must be JSON."}, status=400)

        def work():
            d = DEFAULT_CHOICE
            ch = Choice(body.get("container") or d.container, body.get("codec") or d.codec,
                        body.get("profile") or d.profile, str(body.get("bit_depth") or d.bit_depth),
                        body.get("chroma") or d.chroma)
            try:
                spec, _how = resolve(str(body.get("format") or ""), ch)
                ext, seq = spec.ext, spec.sequence
            except Exception:  # noqa: BLE001 - an unfinished choice still gets a path; the encode reports it
                ext, seq = "mp4", False
            return predict_path(prompt=body.get("prompt") or {}, unique_id=body.get("unique_id"),
                                naming=str(body.get("naming") or "default"), ext=ext, sequence=seq,
                                filename_prefix=str(body.get("filename_prefix") or ""),
                                subfolder=str(body.get("subfolder") or ""),
                                save_output=bool(body.get("save_output", True)),
                                follow=str(body.get("follow_load") or "auto"))

        try:
            return web.json_response(await asyncio.get_running_loop().run_in_executor(_pools()[2], work))
        except Exception as exc:
            log.exception("save next path failed")
            return web.json_response({"error": f"Could not work out the next path: {exc}"}, status=500)

    async def _handle_cm_reload(request):
        """L9.62: a colour-management setting changed - forget the loaded OCIO config. Body {id, value}: the new
        value, used at once (the front end persists it itself, possibly after this request)."""
        from . import save_ocio as SO
        from . import colour_cm as CM
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - an empty body just reloads
            body = {}
        live = {str(body["id"]): body.get("value")} if isinstance(body, dict) and body.get("id") else None
        SO.reload_config(live)
        CM._PROCS.clear()
        from . import colour_gpu as G
        G.reset()
        cfg = CM.config()
        return web.json_response({"config": cfg.getName() if cfg is not None else None,
                                  "source": CM.config_source(), "warning": SO._CFG_WARN})

    async def _handle_still(request):
        import asyncio

        q = dict(request.query)
        try:
            params = parse_request_params(q)
            params["i"] = _int_param(q, "i", 0)
            params["max"] = _int_param(q, "max", 1024)
            data = await asyncio.get_running_loop().run_in_executor(_pools()[0], still_png, params)
            return web.Response(body=data, content_type="image/png")
        except RouteError as exc:
            return web.json_response({"error": exc.message}, status=exc.status)
        except Exception as exc:
            log.exception("still failed")
            return web.json_response({"error": f"Could not show the frame: {exc}"}, status=500)

    async def _handle_scan(request):
        """L9.89: the candidate sequences of a picked frame or a folder (inside the media roots)."""
        import asyncio

        from .media_upload import scan_candidates

        q = dict(request.query)
        try:
            if q.get("path"):
                path = os.path.abspath(strip_path(q["path"]))
                _roots_or_403(path)
            else:
                fp = _import_folder_paths()
                base = os.path.abspath(fp.get_input_directory())
                sub_ = (q.get("subfolder") or "").strip()
                name = strip_path(str(q.get("filename") or ""))
                path = os.path.abspath(os.path.join(base, sub_, name))
                try:
                    inside = os.path.commonpath([base, path]) == base
                except ValueError:
                    inside = False
                if not inside:
                    raise RouteError(400, f"{name} is not in the input folder.")
            if not os.path.exists(path):
                raise RouteError(404, f"Nothing found at {path}.")
            data = await asyncio.get_running_loop().run_in_executor(_pools()[0], scan_candidates, path)
            return web.json_response(data)
        except RouteError as exc:
            return web.json_response({"error": exc.message}, status=exc.status)
        except Exception as exc:
            log.exception("scan failed")
            return web.json_response({"error": f"Could not scan: {exc}"}, status=500)

    async def _handle_upload(request):
        """L9.89: upload stills, EXRs, sequences and movies into the input folder (media_upload.py has the rules)."""
        from .media_upload import UploadError, handle_upload

        try:
            data = await handle_upload(request, _import_folder_paths().get_input_directory())
            return web.json_response(data)
        except UploadError as exc:
            return web.json_response({"error": exc.message}, status=exc.status)
        except Exception as exc:
            log.exception("upload failed")
            return web.json_response({"error": f"Upload failed: {exc}"}, status=500)

    async def _handle_formats(request):
        """L9.89: what this ComfyUI can actually read (the upload dialog's accept list; Rev 3 3.3)."""
        from .media_upload import allowed_extensions

        return web.json_response({"extensions": sorted(allowed_extensions()),
                                  "roots": ROOTS.media_roots(), "local": ROOTS.server_is_local()})

    routes.get("/c2c/media/still")(_handle_still)
    routes.get("/c2c/media/scan")(_handle_scan)
    routes.post("/c2c/media/upload")(_handle_upload)
    routes.get("/c2c/media/formats")(_handle_formats)
    routes.post("/c2c/video/cm/reload")(_handle_cm_reload)
    routes.get("/c2c/save_video/matrix")(_handle_save_matrix)
    routes.post("/c2c/save_video/next_path")(_handle_save_next_path)
    routes.get("/c2c/video/probe")(_handle_probe)
    routes.get("/c2c/video/preview")(_handle_preview)
    routes.post("/c2c/frames/plan")(_handle_frames_plan)
    routes.get("/c2c/frames/thumb")(_handle_frames_thumb)
    _ROUTES_REGISTERED = True
    log.info(
        "[c2c_video] registered GET /c2c/video/probe, /c2c/video/preview, "
        "POST /c2c/frames/plan, GET /c2c/frames/thumb, /c2c/media/still|scan|formats, POST /c2c/media/upload",
    )
    return True
