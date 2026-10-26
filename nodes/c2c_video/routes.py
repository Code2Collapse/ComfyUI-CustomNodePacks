"""HTTP routes for C2C video loader probe + H.264 preview proxies."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from fractions import Fraction
from typing import Any, Mapping

import numpy as np

from .handle import C2CVideo, FileSource, SequenceSource
from .nodes_load import (
    SEQUENCE_IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    audio_output_mode,
    build_images_handle,
    build_loaded_handle,
    load_summary,
    resolve_probe,
    source_times_for_handle,
    strip_path,
    validate_images_directory,
    validate_video_path,
)
from .reader import iter_chunks

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
    params["max_frames"] = min(_MAX_FRAMES_CAP, max(1, _int_param(raw, "max_frames", _DEFAULT_MAX_FRAMES)))
    return params


def _is_media_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext.lstrip(".") in VIDEO_EXTENSIONS or ext in SEQUENCE_IMAGE_EXTENSIONS


def resolve_media_path(params: dict[str, Any]) -> str:
    if params.get("mode") == "images":
        directory = strip_path(str(params.get("directory") or ""))
        valid = validate_images_directory(directory)
        if valid is not True:
            raise RouteError(404, valid if isinstance(valid, str) else f"Nothing found at {directory}.")
        return os.path.abspath(directory)

    if params.get("path"):
        path = strip_path(str(params["path"]))
        valid = validate_video_path(path)
        if valid is not True:
            raise RouteError(404, valid if isinstance(valid, str) else f"Nothing found at {path}.")
        path = os.path.abspath(path)
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

    path = resolve_media_path(params)
    raw = resolve_probe(path, sequence_fps=float(params.get("sequence_fps", 24)))
    force_rate = float(params.get("force_rate", 0))
    skip = int(params.get("skip_first_frames", 0))
    h = raw
    if force_rate > 0:
        h = h.retimed(force_rate, source_times_for_handle(raw))
    n_before = len(h)
    _, loaded = build_loaded_handle(
        raw,
        force_rate=force_rate,
        skip_first_frames=skip,
        select_every_nth=int(params.get("select_every_nth", 1)),
        frame_load_cap=int(params.get("frame_load_cap", 0)),
        format_name=str(params.get("format") or "None"),
        custom_width=0,
        custom_height=0,
        vae=None,
        seconds_cap=float(params.get("seconds_cap", 0.0) or 0.0),
    )
    if len(loaded) == 0:
        retime_note = " after retiming" if force_rate > 0 else ""
        raise RouteError(
            400,
            f"Load Video Path (C2C): no frames selected - the clip has {n_before} frames"
            f"{retime_note} and skip_first_frames={skip} leaves none.",
        )
    return raw, loaded


def _codec_name(raw: C2CVideo) -> str:
    if isinstance(raw.source, FileSource):
        import av

        try:
            with av.open(raw.source.path) as container:
                vs = container.streams.video[raw.stream_index]
                name = getattr(vs.codec_context.codec, "name", None) or vs.codec_context.name
                return str(name or "unknown")
        except Exception as exc:
            raise RouteError(500, f"Could not read codec information: {exc}") from exc
    if isinstance(raw.source, SequenceSource) and raw.source.files:
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

    _, load = build_loaded_handle(
        raw,
        force_rate=float(params.get("force_rate", 0)),
        skip_first_frames=int(params.get("skip_first_frames", 0)),
        select_every_nth=int(params.get("select_every_nth", 1)),
        frame_load_cap=int(params.get("frame_load_cap", 0)),
        format_name=str(params.get("format") or "None"),
        custom_width=int(params.get("custom_width", 0)),
        custom_height=int(params.get("custom_height", 0)),
        vae=None,
        crop=params.get("crop"),
        seconds_cap=float(params.get("seconds_cap", 0.0) or 0.0),
        resolution=str(params.get("resolution") or "custom"),
        divisible_by=int(params.get("divisible_by", 1) or 1),
        snap_frames=str(params.get("handles") or "off") != "fit",   # as a run: fit pads up, the preset does not cut
    )
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


def probe_payload(params: dict[str, Any]) -> dict[str, Any]:
    raw, loaded = build_handle_from_params(params)
    sel = list(loaded.selected_indices())
    src_fps = float(raw.fps)
    try:
        load = _load_report(raw, params)
    except Exception as exc:  # noqa: BLE001 - the probe still answers; the hints just stay empty
        log.debug("load report failed: %s", exc)
        load = None
    return {
        "file": os.path.basename(raw.source.path) if raw.source.kind == "file" else raw.source.pattern,
        "load": load,
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
    blob = f"v{_PREVIEW_VERSION}|{loaded.fingerprint()}|{max_frames}|{out_w}x{out_h}|{ext}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _lock_for(key: str) -> threading.Lock:
    with _PREVIEW_LOCK_GUARD:
        if key not in _PREVIEW_LOCKS:
            _PREVIEW_LOCKS[key] = threading.Lock()
        return _PREVIEW_LOCKS[key]


def _is_linear(raw: C2CVideo) -> bool:
    return (raw.colour.transfer or "").lower() == "linear"


def _chunk_to_rgb_uint8(chunk: np.ndarray, linear: bool) -> np.ndarray:
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
    fmt = "float32" if linear else "uint8"

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
        stream.codec_context.gop_size = _PREVIEW_GOP
        if codec_name == "libx264":
            stream.options = {"crf": "23", "preset": "veryfast", "g": str(_PREVIEW_GOP)}
        written = 0
        for chunk in iter_chunks(capped, _PREVIEW_CHUNK, fmt=fmt):
            rgb_batch = _chunk_to_rgb_uint8(chunk, linear)
            for row in rgb_batch:
                if written >= n:
                    break
                frame = av.VideoFrame.from_ndarray(row, format="rgb24")
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

    routes.get("/c2c/save_video/matrix")(_handle_save_matrix)
    routes.post("/c2c/save_video/next_path")(_handle_save_next_path)
    routes.get("/c2c/video/probe")(_handle_probe)
    routes.get("/c2c/video/preview")(_handle_preview)
    routes.post("/c2c/frames/plan")(_handle_frames_plan)
    routes.get("/c2c/frames/thumb")(_handle_frames_thumb)
    _ROUTES_REGISTERED = True
    log.info(
        "[c2c_video] registered GET /c2c/video/probe, /c2c/video/preview, "
        "POST /c2c/frames/plan, GET /c2c/frames/thumb",
    )
    return True
