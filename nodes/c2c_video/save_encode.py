"""Chunked encode pipeline for Save Video (C2C)."""

from __future__ import annotations

import json
from dataclasses import dataclass
import os
import wave
from fractions import Fraction
from typing import Any

import numpy as np
import torch

from .budget import bytes_per_frame, frames_that_fit
from .save_formats import FormatSpec, SaveVideoError
from .save_ocio import transform_chunk

_BT709 = 1
_RANGE_TV = 1
_SWS_FLAGS = 0x4 | 0x40000 | 0x2000 | 0x4000
_PYAV_MATRICES = frozenset({1, 4, 5, 7})      # swscale matrices PyAV's reformat applies (L9.40, measured)


def _progress_and_interrupt():
    try:
        import comfy.model_management as mm
        from comfy.utils import ProgressBar
        return ProgressBar, mm.throw_exception_if_processing_interrupted
    except Exception:
        return None, lambda: None


def _import_folder_paths():
    import folder_paths  # noqa: WPS433
    return folder_paths


def read_comfy_setting(key: str, default: str) -> str:
    try:
        fp = _import_folder_paths()
        base = fp.get_user_directory()
        if not base:
            return default
        path = os.path.join(base, "default", "comfy.settings.json")
        if not os.path.isfile(path):
            return default
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        val = data.get(key, default)
        return str(val) if val is not None else default
    except Exception:
        return default


def effective_naming(naming: str, naming_resolved: str) -> str:
    """default = the C2C setting c2c.saveVideo.naming; its default is the Folder Version Incrementer (L9.40, owner
    2026-10-23: "when we select folder-version-incrementer or default it automatically fills the information")."""
    from .save_naming import DEFAULT_SETTING, canonical
    if naming and naming != "default":
        return canonical(naming)
    if naming_resolved and naming_resolved.strip():
        return canonical(naming_resolved.strip())
    return canonical(read_comfy_setting("c2c.saveVideo.naming", DEFAULT_SETTING))


def pad_edge_replicate(images: torch.Tensor, width: bool = True, height: bool = True,
                       ) -> tuple[torch.Tensor, str | None]:
    """Pad odd W/H by one pixel (edge replicate): both for 4:2:0 codecs, the width for 4:2:2 (L2.53)."""
    _, h, w, _ = images.shape
    pad_w = w % 2 if width else 0
    pad_h = h % 2 if height else 0
    if not pad_w and not pad_h:
        return images, None
    nw, nh = w + pad_w, h + pad_h
    out = torch.empty((images.shape[0], nh, nw, images.shape[3]), dtype=images.dtype)
    out[..., :h, :w, :] = images
    if pad_w:
        out[..., :h, w:nw, :] = images[..., :h, w - 1:w, :]
    if pad_h:
        out[..., h:nh, :, :] = out[..., h - 1:h, :, :]
    return out, f"{w}×{h} → {nw}×{nh}"


def quantize_rgb(rgb: np.ndarray, bit_depth: int, clamp_hdr: bool = True) -> np.ndarray:
    x = rgb.astype(np.float32)
    if clamp_hdr:
        x = np.clip(x, 0.0, 1.0)
    if bit_depth <= 8:
        scale = 255.0
        return np.rint(x * scale).astype(np.uint8)
    scale = float((1 << bit_depth) - 1)
    return np.rint(x * scale).astype(np.uint16 if bit_depth <= 16 else np.uint32)


_CHUNK_BYTES = 64 * 1024 * 1024


def _rgb_to_av_frame(rgb: np.ndarray, pix_fmt: str, bit_depth: int, colorspace: int = _BT709,
                     color_range: int = _RANGE_TV):
    """float32 RGB(A) 0-1 -> an av.VideoFrame in `pix_fmt`.

    8-bit targets start from rgb24 / rgba. Deeper targets start from 16-bit FULL-scale rgb48le / rgba64le and
    swscale reduces to 10/12-bit (accurate rounding flags), so nothing passes through 8 bits on the way (A8).
    Alpha is kept only when the target pix_fmt has it."""
    import av
    rgb = np.asarray(rgb, dtype=np.float32)
    want_alpha = rgb.shape[-1] >= 4 and ("a" in pix_fmt.replace("gray", "") and ("yuva" in pix_fmt or "gbra" in pix_fmt
                                                                              or pix_fmt.startswith("rgba")
                                                                              or pix_fmt.startswith("bgra")))
    x = rgb[..., :4] if want_alpha else rgb[..., :3]
    if bit_depth <= 8:
        q = quantize_rgb(x, 8)
        src_fmt = "rgba" if want_alpha else "rgb24"
    else:
        q = quantize_rgb(x, 16)
        src_fmt = "rgba64le" if want_alpha else "rgb48le"
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(q), format=src_fmt)
    if pix_fmt == src_fmt:
        return frame
    if pix_fmt.startswith(("yuv", "yuva")):
        return frame.reformat(format=pix_fmt, dst_colorspace=colorspace, dst_color_range=color_range,
                              interpolation=_SWS_FLAGS)
    return frame.reformat(format=pix_fmt, interpolation=_SWS_FLAGS)


def quantize_fast(rgb: np.ndarray, bits: int) -> np.ndarray:
    """quantize_rgb(rgb, 8 or 16) with torch's threads: the same float32 clip, scale and round-half-to-even, so the
    same integers (tests/test_save_video_4k.py checks it)."""
    t = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32))
    q = t.clamp(0.0, 1.0).mul_(255.0 if bits <= 8 else 65535.0).round_()
    if bits <= 8:
        return q.to(torch.uint8).numpy()
    return q.to(torch.int32).numpy().astype(np.uint16)


@dataclass(frozen=True)
class Extras:
    """L9.54 subformat options (owner 2026-10-25). Every default = the encoder's behaviour before them."""
    encoder_preset: str = "default"      # x264 / x265 speed names; NVENC and SVT-AV1 get the matching step
    gop: int = 0                         # keyframe interval in frames (0 = the encoder's own)
    bitrate_kbps: int = 0                # > 0: average bitrate instead of constant quality
    channels: str = "auto"               # auto = alpha when the frames have it; RGB drops it; RGBA adds opaque
    png_compression: int = -1            # PNG / PNG-in-MOV zlib level 0-9 (-1 = the writer's default)


X264_PRESETS = ("default", "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower",
                "veryslow")
_NVENC_STEP = {"ultrafast": "p1", "superfast": "p1", "veryfast": "p2", "faster": "p3", "fast": "p4", "medium": "p5",
               "slow": "p6", "slower": "p7", "veryslow": "p7"}
_SVT_STEP = {"ultrafast": "12", "superfast": "11", "veryfast": "10", "faster": "9", "fast": "8", "medium": "6",
             "slow": "4", "slower": "3", "veryslow": "2"}


def encoder_options(spec: FormatSpec, quality: int, extras: "Extras | None" = None) -> dict[str, str]:
    """Codec options for ``spec`` at ``quality`` (0-100): the same for the PyAV and the ffmpeg path."""
    opts = _quality_options(spec, quality)
    x = extras or Extras()
    preset = str(x.encoder_preset or "default")
    if preset != "default":
        if spec.codec in ("libx264", "libx265"):
            opts["preset"] = preset
        elif spec.quality_param == "cq":
            opts["preset"] = _NVENC_STEP.get(preset, opts.get("preset", "p5"))
        elif spec.quality_param == "svt":
            opts["preset"] = _SVT_STEP.get(preset, opts.get("preset", "8"))
    if int(x.gop or 0) > 0 and spec.codec in ("libx264", "libx265", "libvpx-vp9", "libsvtav1", "h264_nvenc",
                                               "hevc_nvenc", "av1_nvenc"):
        opts["g"] = str(int(x.gop))
    if int(x.bitrate_kbps or 0) > 0 and spec.codec in ("libx264", "libx265", "libvpx-vp9", "h264_nvenc",
                                                        "hevc_nvenc", "av1_nvenc"):
        opts.pop("crf", None)
        opts.pop("cq", None)
        opts["b"] = f"{int(x.bitrate_kbps)}k"
    if int(x.png_compression) >= 0 and spec.codec == "png":
        opts["compression_level"] = str(max(0, min(9, int(x.png_compression))))
    return opts


def _quality_options(spec: FormatSpec, quality: int) -> dict[str, str]:
    opts: dict[str, str] = {}
    if spec.profile:
        opts["profile"] = spec.profile
    if spec.quality_param == "crf" and spec.codec in ("libx264", "libx265"):
        opts["crf"] = quality_to_crf(quality)
        if spec.codec == "libx264":
            opts["preset"] = "medium"
    elif spec.quality_param == "crf" and spec.codec == "libvpx-vp9":
        opts["crf"] = quality_to_vp9_crf(quality)
    elif spec.quality_param == "qscale" and spec.codec == "libwebp":
        opts["qscale"] = str(max(0, min(100, 100 - quality)))
    elif spec.quality_param == "cq":
        # NVENC constant quality, as the 4K pack sets it (cq on the crf scale; p5 = its slower, better preset)
        opts.update({"rc": "vbr", "cq": quality_to_crf(quality), "b": "0", "preset": "p5"})
    elif spec.quality_param == "vp9":                        # L9.40 matrix: VP9 by name
        opts["crf"] = quality_to_vp9_crf(quality)
        opts["b"] = "0"                                      # constant quality: crf alone is capped by a bitrate
    elif spec.quality_param == "svt":                        # SVT-AV1: crf 0-63, preset 8 (speed / quality middle)
        opts.update({"crf": quality_to_vp9_crf(quality), "preset": "8"})
    elif spec.quality_param == "cfhd":                       # CineForm: 12 = low ... 0 = film3+
        opts["quality"] = str(max(0, min(12, int(round((100 - max(0, min(100, int(quality)))) * 12 / 100)))))
    return opts


def _convert(arr: np.ndarray, colorspace_in: str, colorspace_out: str, cm=None) -> np.ndarray:
    """L9.53: with colour on 'auto (from Load)' the plan's transform (cm = (Transform, inverse)) - exactly the inverse
    of what Load Video did; otherwise the conversion the two widgets name, as before."""
    if cm is not None:
        from .colour_cm import apply
        return apply(np.ascontiguousarray(arr, dtype=np.float32), cm[0], inverse=cm[1])
    if colorspace_out and colorspace_out != "same as input":
        return transform_chunk(arr, colorspace_in, colorspace_out)
    return arr


def colour_tags(spec: FormatSpec, has_alpha: bool, colorspace_out: str) -> dict[str, str] | None:
    """What a YUV file says about its colour (L2.53). The frames are always converted with the BT.709 matrix to
    limited range, so those two are always stated. Primaries and transfer only when the output space is one we can
    name: sRGB (as ComfyUI's own Save Video tags it) or a Rec.709 / Rec.1886 display."""
    pix = spec.pix_fmt_for(has_alpha)
    if not pix.startswith(("yuv", "nv12", "p010")):
        return None
    tags = {"colorspace": "bt709", "color_range": "tv"}
    name = (colorspace_out or "").lower()
    if any(k in name for k in ("linear", "acescg", "aces2065", "log", "pq", "hlg", "2020", "p3")):
        return tags
    if "srgb" in name:
        tags.update(color_primaries="bt709", color_trc="iec61966-2-1")
    elif "1886" in name or "rec.709" in name or "bt.709" in name:
        tags.update(color_primaries="bt709", color_trc="bt709")
    return tags


# PyAV takes the enum numbers (libavutil/pixfmt.h)
_TAG_NUM = {"bt709": 1, "iec61966-2-1": 13, "tv": 1}


def output_metadata(prompt: Any = None, extra_pnginfo: Any = None) -> dict[str, Any] | None:
    """The workflow and prompt as ComfyUI's own Save Video stores them (keys "workflow", "prompt", JSON values), so a
    saved file dragged into ComfyUI rebuilds its graph. None when there is nothing to store or ComfyUI was started
    with --disable-metadata."""
    try:
        from comfy.cli_args import args as _args
        if getattr(_args, "disable_metadata", False):
            return None
    except Exception:  # noqa: BLE001 - not inside ComfyUI
        pass
    meta: dict[str, Any] = {}
    if isinstance(extra_pnginfo, dict):
        meta.update(extra_pnginfo)
    if prompt is not None:
        meta["prompt"] = prompt
    return meta or None


def workflow_png_path(output_path: str) -> str:
    """'<name>.png' next to a written file, or next to a sequence's folder ('<folder>.png' - never inside it, so it
    cannot collide with a frame)."""
    p = os.path.normpath(output_path)
    if os.path.isdir(p):
        return p + ".png"
    return os.path.splitext(p)[0] + ".png"


def write_workflow_png(frame: Any, output_path: str, prompt: Any = None, extra_pnginfo: Any = None) -> str | None:
    """L9.52 (owner 2026-10-25): the workflow and prompt go into a PNG next to the saved file - its first written frame,
    8-bit RGB, text chunks exactly as ComfyUI's SaveImage writes them - never into the video. None (nothing written)
    with --disable-metadata, nothing to store, or no frame. -> the PNG's file name."""
    meta = output_metadata(prompt, extra_pnginfo)
    if not meta or frame is None or not output_path:
        return None
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    t = frame[0] if getattr(frame, "ndim", 0) == 4 else frame
    arr = np.clip(t[..., :3].detach().float().cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
    info = PngInfo()
    for key, value in meta.items():
        info.add_text(str(key), json.dumps(value))
    path = workflow_png_path(output_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(arr).save(path, pnginfo=info, compress_level=4)
    return os.path.basename(path)


def quality_to_crf(quality: int) -> str:
    q = max(0, min(100, int(quality)))
    return str(int(round(51 - q * 0.45)))


def quality_to_vp9_crf(quality: int) -> str:
    q = max(0, min(100, int(quality)))
    return str(int(round(63 - q * 0.5)))


def prepare_audio(
    audio: dict | None,
    n_frames: int,
    fps: float,
) -> tuple[np.ndarray, int] | None:
    if audio is None:
        return None
    w = audio["waveform"]
    if torch.is_tensor(w):
        w = w.detach().cpu().float()
    else:
        w = torch.as_tensor(w, dtype=torch.float32)
    if w.ndim == 3:
        w = w[0]
    if w.ndim == 1:
        w = w.unsqueeze(0)
    sr = int(audio["sample_rate"])
    want = max(0, int(round(n_frames / float(fps) * sr)))
    if want <= 0:
        return np.zeros((w.shape[0], 0), dtype=np.float32), sr
    cur = int(w.shape[-1])
    if cur > want:
        w = w[..., :want]
    elif cur < want:
        pad = torch.zeros(w.shape[0], want - cur, dtype=torch.float32)
        w = torch.cat([w, pad], dim=-1)
    return w.numpy().astype(np.float32), sr


def write_wav_sidecar(path: str, pcm: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    ch = int(pcm.shape[0])
    data = np.clip(pcm, -1.0, 1.0)
    i16 = (np.rint(data * 32767.0)).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.T.reshape(-1).tobytes())


def resolve_output_path(
    *,
    spec: FormatSpec,
    naming: str,
    filename_prefix: str,
    subfolder: str,
    save_output: bool,
    width: int,
    height: int,
    base_dir: str | None = None,
) -> tuple[str, str, str, str]:
    """Return (full_path, filename, subfolder_rel, folder_type). base_dir: an absolute folder outside ComfyUI's output
    (a Folder Version Incrementer base_path) - the file goes there and folder_type is "external"."""
    fp = _import_folder_paths()
    if save_output and base_dir:
        out_dir = base_dir
        folder_type = "external"
    elif save_output:
        out_dir = fp.get_output_directory()
        folder_type = "output"
    else:
        out_dir = os.path.join(fp.get_temp_directory(), "c2c_save_video")
        folder_type = "temp"
    os.makedirs(out_dir, exist_ok=True)

    prefix = (filename_prefix or "C2C/video").strip().replace("\\", "/")
    sub = (subfolder or "").strip().replace("\\", "/")

    if naming in ("Folder Version", "Folder Version Incrementer"):
        if not sub:
            raise SaveVideoError(
                "Folder Version naming needs the subfolder input wired from "
                "the Folder Version Incrementer."
            )
        stem = os.path.basename(prefix) or "video"
        rel_sub = sub
        if spec.sequence:
            rel_dir = os.path.join(rel_sub, stem).replace("\\", "/")
            full_dir = os.path.join(out_dir, rel_dir)
            if os.path.exists(full_dir):
                raise SaveVideoError(
                    f"Folder Version: {rel_dir} already exists — will not overwrite."
                )
            os.makedirs(full_dir, exist_ok=False)
            return full_dir, stem, rel_dir, folder_type
        fname = f"{stem}.{spec.ext}"
        rel_path = os.path.join(rel_sub, fname).replace("\\", "/")
        full = os.path.join(out_dir, rel_path)
        if os.path.exists(full):
            raise SaveVideoError(f"Folder Version: {rel_path} already exists — will not overwrite.")
        os.makedirs(os.path.join(out_dir, rel_sub), exist_ok=True)
        return full, fname, rel_path, folder_type

    full_output_folder, filename, counter, counter_subfolder, _pfx = fp.get_save_image_path(
        prefix, out_dir, width, height,
    )
    os.makedirs(full_output_folder, exist_ok=True)
    rel_sub = counter_subfolder.replace("\\", "/") if counter_subfolder else ""
    # prefix_00001 like core SaveImage and VHS: get_save_image_path returns the next free counter for the prefix
    # (it scans names of that shape), so a second run never overwrites the first.
    stem = f"{filename}_{int(counter):05}"
    if spec.sequence:
        seq_dir = os.path.join(full_output_folder, stem)
        os.makedirs(seq_dir, exist_ok=True)
        rel = os.path.join(rel_sub, stem).replace("\\", "/") if rel_sub else stem
        return seq_dir, stem, rel, folder_type
    fname = f"{stem}.{spec.ext}"
    full = os.path.join(full_output_folder, fname)
    rel = os.path.join(rel_sub, fname).replace("\\", "/") if rel_sub else fname
    return full, fname, rel, folder_type


PROXY_MAX_FRAMES = 20000
PROXY_MAX_LONG = 1280


def proxy_disabled() -> bool:
    return os.environ.get("C2C_SAVE_SOURCE_PROXY", "").strip().lower() in ("0", "off", "false", "no")


def write_proxy(images, fps: float, tag: str) -> dict | None:
    """A browser-playable H.264 proxy (long edge <= 1280 px, every frame, in chunks) of the frames as they arrive -
    the preview run and the A/B source of Save Video (L9.40). Reads a FrameStream front to back (it consumes it).
    None when there is nothing to show or the clip is longer than PROXY_MAX_FRAMES."""
    import uuid
    from fractions import Fraction

    import av

    n = int(images.shape[0])
    if n == 0 or n > PROXY_MAX_FRAMES:
        return None
    fp = _import_folder_paths()
    sub = "c2c_save_video"
    out_dir = os.path.join(fp.get_temp_directory(), sub)
    os.makedirs(out_dir, exist_ok=True)
    name = f"{tag}_{uuid.uuid4().hex[:12]}.mp4"
    _, h, w, _ = images.shape
    scale = min(1.0, PROXY_MAX_LONG / float(max(h, w)))
    nw, nh = max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2)
    container = av.open(os.path.join(out_dir, name), mode="w", format="mp4", options={"movflags": "+faststart"})
    try:
        stream = container.add_stream("libx264", rate=Fraction(float(fps or 24.0)).limit_denominator(1_000_000))
        stream.width, stream.height, stream.pix_fmt = nw, nh, "yuv420p"
        stream.options = {"crf": "20", "preset": "veryfast"}
        k = 0
        for a in range(0, n, 32):
            x = images[a:min(n, a + 32)][..., :3].detach().float().cpu().clamp(0, 1)
            if (nh, nw) != tuple(x.shape[1:3]):
                x = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(nh, nw), mode="area"
                                                    ).permute(0, 2, 3, 1)
            for row in (x * 255.0 + 0.5).to(torch.uint8).numpy():
                vf = av.VideoFrame.from_ndarray(row, format="rgb24")
                vf.pts = k
                k += 1
                for packet in stream.encode(vf):
                    container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    return {"filename": name, "subfolder": sub, "type": "temp", "frames": n, "width": nw, "height": nh}


def build_preview_from_saved(paths: list[str], spec: FormatSpec) -> dict:
    """An H.264 preview read back from the file just written, in chunks - for a streamed save, whose frames are gone
    by the time the file is closed."""
    import uuid

    from .probe import probe_file, probe_sequence
    from .routes import _DEFAULT_MAX_FRAMES, _encode_preview, _pick_encoder

    fp = _import_folder_paths()
    raw = probe_sequence(os.path.dirname(paths[0])) if spec.sequence else probe_file(paths[0])
    _codec, ext = _pick_encoder()
    name = f"c2c_save_preview_{uuid.uuid4().hex[:12]}.{ext}"
    temp = fp.get_temp_directory()
    os.makedirs(temp, exist_ok=True)
    _encode_preview(raw, raw, os.path.join(temp, name), _DEFAULT_MAX_FRAMES)
    return {"filename": name, "subfolder": "", "type": "temp"}


def build_preview(images: torch.Tensor, fps: float, spec: FormatSpec, saved: dict) -> dict:
    if spec.browser_playable and saved.get("type") != "external":
        return {
            "filename": saved["filename"],
            "subfolder": saved["subfolder"],
            "type": saved["type"],
        }
    from .._video_comparer_io import save_preview_media
    prev = save_preview_media(images, "save_preview", fps)
    if not prev:
        raise SaveVideoError("Could not build H.264 preview for this format.")
    return {
        "filename": prev["filename"],
        "subfolder": prev["subfolder"],
        "type": prev.get("type", "temp"),
    }


def _encode_av_stream(
    images: torch.Tensor,
    spec: FormatSpec,
    path: str,
    fps: float,
    quality: int,
    colorspace_in: str,
    colorspace_out: str,
    audio: dict | None,
    has_alpha: bool,
    metadata: dict[str, Any] | None = None,
    colour=None,
    extras: "Extras | None" = None,
    cm=None,
) -> list[str]:
    import av

    from .save_ffmpeg import encode_with_ffmpeg, exact_rate, ffmpeg_for

    ProgressBar, interrupt = _progress_and_interrupt()
    n, h, w, c = images.shape
    channels = 4 if has_alpha and c >= 4 else 3
    bpf = bytes_per_frame(w, h, channels, np.float32)
    # By bytes too: each chunk is copied twice on its way to the encoder, so a whole-batch chunk cost 2x the batch
    # (L7.59: Save alone peaked at 2.5x). The encoder sees the same frames in the same order.
    chunk = max(1, min(64, frames_that_fit(w, h, channels, np.float32) or 1, _CHUNK_BYTES // max(1, bpf)))
    from .save_matrix import apply_colour, conversion
    pix_fmt = spec.pix_fmt_for(has_alpha)
    yuv = pix_fmt.startswith(("yuv", "nv12", "p010", "p016"))
    tags = apply_colour(colour_tags(spec, has_alpha, colorspace_out), colour, yuv)
    conv = conversion(colour)

    def chunks():
        for start in range(0, n, chunk):
            arr = images[start:start + chunk].cpu().numpy().astype(np.float32)
            arr = _convert(arr, colorspace_in, colorspace_out, cm)
            yield arr if has_alpha else arr[..., :3]

    exe = ffmpeg_for(spec, pix=pix_fmt)
    if exe is not None:                     # L2.53: ffmpeg converts and encodes in its own process (3-4x faster)
        pbar = ProgressBar(n) if ProgressBar else None
        pcm = prepare_audio(audio, n, fps) if spec.audio_codec and spec.audio_codec != "wav_sidecar" else None
        return encode_with_ffmpeg(
            exe, images, spec, path, fps=fps, has_alpha=has_alpha, encoder_opts=encoder_options(spec, quality, extras),
            tags=tags, metadata=metadata, pcm=pcm, chunks=chunks(), quantize=quantize_fast, interrupt=interrupt,
            progress=(lambda done, total: pbar.update_absolute(done, total)) if pbar else (lambda d, t: None),
            matrix=conv[0], out_range=conv[2])

    if yuv and conv[1] not in _PYAV_MATRICES:
        # measured (PyAV 17): reformat knows BT.709 / FCC / BT.601 / SMPTE-240M only and converts anything else
        # with BT.601 while the file would say BT.2020 - refuse instead of writing wrong colours
        raise SaveVideoError(
            f"{spec.codec} {pix_fmt} is written through PyAV here, and PyAV's converter has no BT.2020 matrix. "
            "Install ffmpeg 5.1 or newer (on PATH) for BT.2020, or choose BT.709 / BT.601 primaries.")

    # PyAV's muxer names differ from the file extension (mkv -> matroska), and muxer options must be given at open
    # time: Container.options is read-only in this PyAV build.
    av_format = {"mkv": "matroska"}.get(spec.container, spec.container)
    mux_opts = {}
    if spec.container in ("mp4", "mov"):
        # use_metadata_tags: the workflow / prompt keys go where ComfyUI's front end reads them (as core does)
        mux_opts = {"movflags": "+faststart" + ("+use_metadata_tags" if metadata else "")}
    container = av.open(path, mode="w", format=av_format if spec.container != "dir" else None, options=mux_opts)
    created = True
    try:
        for key, value in (metadata or {}).items():
            container.metadata[key] = value if isinstance(value, str) else json.dumps(value)
        stream = container.add_stream(spec.codec, rate=exact_rate(fps))
        stream.width = w
        stream.height = h
        stream.pix_fmt = pix_fmt
        if spec.codec_tag:
            stream.codec_tag = spec.codec_tag
        if tags:
            from .save_matrix import TAG_NUM
            cc = stream.codec_context
            if tags.get("colorspace"):
                cc.colorspace = TAG_NUM[tags["colorspace"]]
            if tags.get("color_range"):
                cc.color_range = TAG_NUM[tags["color_range"]]
            if tags.get("color_primaries"):
                cc.color_primaries = TAG_NUM[tags["color_primaries"]]
            if tags.get("color_trc"):
                cc.color_trc = TAG_NUM[tags["color_trc"]]
        opts = encoder_options(spec, quality, extras)
        if opts:
            stream.options = opts

        audio_stream = None
        pcm_info = prepare_audio(audio, n, fps) if spec.audio_codec and spec.audio_codec != "wav_sidecar" else None
        if pcm_info and spec.audio_codec:
            pcm, sr = pcm_info
            audio_stream = container.add_stream(spec.audio_codec, rate=sr)
            if spec.audio_codec == "aac":
                audio_stream.bit_rate = 192_000

        pbar = ProgressBar(n) if ProgressBar else None
        written = 0
        for arr in chunks():
            interrupt()
            for i in range(arr.shape[0]):
                vf = _rgb_to_av_frame(arr[i], pix_fmt, spec.bit_depth, conv[1], conv[3])
                vf.pts = written
                for packet in stream.encode(vf):
                    container.mux(packet)
                written += 1
            if pbar:
                pbar.update_absolute(min(written, n), n)

        for packet in stream.encode(None):
            container.mux(packet)

        if audio_stream and pcm_info:
            pcm, sr = pcm_info
            import av
            layout = "stereo" if pcm.shape[0] >= 2 else "mono"
            for si in range(0, pcm.shape[1], sr // 10 or 4800):
                chunk_pcm = pcm[:, si:si + sr // 10 or 4800]
                if chunk_pcm.shape[1] == 0:
                    continue
                frame = av.AudioFrame.from_ndarray(chunk_pcm, format="fltp", layout=layout)
                frame.sample_rate = sr
                frame.pts = si
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
            for packet in audio_stream.encode(None):
                container.mux(packet)
    except Exception:
        container.close()
        if created and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise
    else:
        container.close()
    return [path.replace("\\", "/")]


def _encode_gif(images: torch.Tensor, path: str, fps: float, quality: int,
                colorspace_in: str, colorspace_out: str, cm=None) -> list[str]:
    from PIL import Image

    ProgressBar, interrupt = _progress_and_interrupt()
    n = int(images.shape[0])
    frames: list[Image.Image] = []
    pbar = ProgressBar(n) if ProgressBar else None
    for i in range(n):
        interrupt()
        row = images[i].cpu().numpy().astype(np.float32)
        row = _convert(row[None, ...], colorspace_in, colorspace_out, cm)[0]
        u8 = quantize_rgb(row[..., :3], 8)
        frames.append(Image.fromarray(u8, mode="RGB"))
        if pbar:
            pbar.update_absolute(i + 1, n)
    if not frames:
        raise SaveVideoError("No frames to write.")
    dur = max(1, int(round(1000.0 / max(fps, 0.001))))
    q = max(1, min(100, int(quality)))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=dur,
        loop=0,
        optimize=False,
    )
    return [path.replace("\\", "/")]


def _encode_webp(images: torch.Tensor, path: str, fps: float, quality: int,
                 colorspace_in: str, colorspace_out: str, has_alpha: bool, cm=None) -> list[str]:
    from PIL import Image

    ProgressBar, interrupt = _progress_and_interrupt()
    n = int(images.shape[0])
    frames: list[Image.Image] = []
    pbar = ProgressBar(n) if ProgressBar else None
    for i in range(n):
        interrupt()
        row = images[i].cpu().numpy().astype(np.float32)
        row = _convert(row[None, ...], colorspace_in, colorspace_out, cm)[0]
        if has_alpha and row.shape[-1] >= 4:
            frames.append(Image.fromarray(quantize_rgb(row[..., :4], 8), mode="RGBA"))
        else:
            frames.append(Image.fromarray(quantize_rgb(row[..., :3], 8), mode="RGB"))
        if pbar:
            pbar.update_absolute(i + 1, n)
    if not frames:
        raise SaveVideoError("No frames to write.")
    dur = max(1, int(round(1000.0 / max(fps, 0.001))))
    q = max(0, min(100, int(quality)))
    frames[0].save(path, format="WEBP", save_all=True, append_images=frames[1:], duration=dur, loop=0,
                   quality=q, lossless=q >= 100, method=4)
    return [path.replace("\\", "/")]


_DNXHR_MIN = (256, 120)


_MATRIX_SEQ = frozenset({"oiio_png", "oiio_tiff", "oiio_dpx", "oiio_jpeg", "oiio_exr"})


def _write_matrix_frame(oiio, spec: FormatSpec, fpath: str, row: np.ndarray, has_alpha: bool, quality: int,
                        extras: "Extras | None" = None, interop_id: str | None = None) -> None:
    """One frame of a matrix image sequence (L9.40). Integer formats are clipped to 0-1 and rounded like the PNG16
    path; DPX stores 10 / 12 bits in 16-bit words (OIIO's BitsPerSample); EXR stores premultiplied colour (Nuke's
    Read assumes it) and TIFF / PNG straight alpha, as each format defines."""
    h, w = row.shape[:2]
    alpha = has_alpha and spec.supports_alpha and row.shape[-1] >= 4
    names = ["R", "G", "B"] + (["A"] if alpha else [])
    data = row[..., :len(names)].astype(np.float32)
    bits = spec.bit_depth
    if spec.codec == "oiio_exr":
        if alpha:
            data = np.concatenate([data[..., :3] * data[..., 3:4], data[..., 3:4]], axis=-1)
        fmt, arr = (oiio.HALF if bits == 16 else oiio.FLOAT), data
    elif spec.codec == "oiio_tiff" and bits == 32:
        fmt, arr = oiio.FLOAT, data
    elif bits <= 8:
        fmt, arr = oiio.UINT8, np.rint(np.clip(data, 0, 1) * 255.0).astype(np.uint8)
    else:
        top = float((1 << bits) - 1) if spec.codec == "oiio_dpx" else 65535.0
        fmt, arr = oiio.UINT16, np.rint(np.clip(data, 0, 1) * top).astype(np.uint16)
    ospec = oiio.ImageSpec(w, h, len(names), fmt)
    ospec.channelnames = tuple(names)
    if alpha:
        ospec.alpha_channel = len(names) - 1
        if spec.codec in ("oiio_png", "oiio_tiff"):
            ospec.attribute("oiio:UnassociatedAlpha", 1)       # straight alpha, as ComfyUI's IMAGE + mask is
    if spec.codec == "oiio_dpx":
        ospec.attribute("oiio:BitsPerSample", int(bits))
    elif spec.codec in ("oiio_tiff", "oiio_exr") and spec.profile:
        ospec.attribute("compression", str(spec.profile))
    if spec.codec == "oiio_exr" and interop_id:
        ospec.attribute("colorInteropID", str(interop_id))   # L9.53: what Load Video (and Nuke 16+) read back
    elif spec.codec == "oiio_jpeg":
        ospec.attribute("Compression", f"jpeg:{max(1, min(100, int(quality)))}")
        ospec.attribute("jpeg:subsampling", str(spec.profile or "4:2:0"))
    if spec.codec == "oiio_png" and extras is not None and int(extras.png_compression) >= 0:
        ospec.attribute("png:compressionLevel", max(0, min(9, int(extras.png_compression))))
    out = oiio.ImageOutput.create(fpath)
    if out is None:
        raise SaveVideoError(f"OpenImageIO cannot write {fpath}: {oiio.geterror()}")
    try:
        if not out.open(fpath, ospec):
            raise SaveVideoError(out.geterror())
        if not out.write_image(np.ascontiguousarray(arr)):
            raise SaveVideoError(out.geterror())
    finally:
        out.close()


def _encode_sequence_oiio(
    images: torch.Tensor,
    spec: FormatSpec,
    out_dir: str,
    stem: str,
    fps: float,
    colorspace_in: str,
    colorspace_out: str,
    audio: dict | None,
    has_alpha: bool,
    clamp_hdr: bool,
    quality: int = 90,
    extras: "Extras | None" = None,
    cm=None,
    interop_id: str | None = None,
) -> list[str]:
    import OpenImageIO as oiio  # type: ignore[import-not-found]

    ProgressBar, interrupt = _progress_and_interrupt()
    n = int(images.shape[0])
    ext = spec.file_ext or "exr"
    paths: list[str] = []
    pbar = ProgressBar(n) if ProgressBar else None
    for i in range(n):
        interrupt()
        row = images[i].cpu().numpy().astype(np.float32)
        row = _convert(row[None, ...], colorspace_in, colorspace_out, cm)[0]
        fname = f"{stem}.{i + 1:04d}.{ext}"
        fpath = os.path.join(out_dir, fname)
        h, w = row.shape[:2]
        if spec.codec in _MATRIX_SEQ:
            _write_matrix_frame(oiio, spec, fpath, row, has_alpha, quality, extras, interop_id)
        elif spec.codec == "oiio_png16":
            names = ["R", "G", "B"]
            planes = [np.clip(row[..., k], 0, 1) for k in range(3)]
            if has_alpha and row.shape[-1] >= 4:
                names.append("A")
                planes.append(np.clip(row[..., 3], 0, 1))
            data = np.stack(planes, axis=-1)
            u16 = np.rint(data * 65535.0).astype(np.uint16)
            ospec = oiio.ImageSpec(w, h, len(names), oiio.UINT16)
            ospec.channelnames = tuple(names)
            if len(names) == 4:
                ospec.alpha_channel = 3
                # PNG alpha is STRAIGHT by spec and so is ComfyUI's: tell OIIO, or it un-premultiplies on write
                # (measured: max error 1.0 on a gradient alpha).
                ospec.attribute("oiio:UnassociatedAlpha", 1)
            out = oiio.ImageOutput.create(fpath)
            if out is None:
                raise SaveVideoError(f"OpenImageIO cannot write {fpath}: {oiio.geterror()}")
            try:
                if not out.open(fpath, ospec):
                    raise SaveVideoError(out.geterror())
                if not out.write_image(u16):
                    raise SaveVideoError(out.geterror())
            finally:
                out.close()
        else:
            half = spec.codec == "oiio_exr_half"
            fmt = oiio.HALF if half else oiio.FLOAT
            names = ["R", "G", "B"]
            planes = [row[..., k].astype(np.float32) for k in range(3)]
            if has_alpha and row.shape[-1] >= 4:
                # EXR stores PREMULTIPLIED colour (Nuke's Read assumes it); ComfyUI's IMAGE + alpha is straight.
                a_ = row[..., 3].astype(np.float32)
                planes = [pl * a_ for pl in planes]
                names.append("A")
                planes.append(a_)
            stacked = np.stack(planes, axis=-1)
            ospec = oiio.ImageSpec(w, h, len(names), fmt)
            ospec.channelnames = tuple(names)
            if len(names) == 4:
                ospec.alpha_channel = 3
            ospec.attribute("compression", "zip")
            if interop_id:
                ospec.attribute("colorInteropID", str(interop_id))
            out = oiio.ImageOutput.create(fpath)
            if out is None:
                raise SaveVideoError(f"OpenImageIO cannot write {fpath}: {oiio.geterror()}")
            try:
                if not out.open(fpath, ospec):
                    raise SaveVideoError(out.geterror())
                if not out.write_image(stacked):
                    raise SaveVideoError(out.geterror())
            finally:
                out.close()
        paths.append(fpath.replace("\\", "/"))
        if pbar:
            pbar.update_absolute(i + 1, n)

    if spec.audio_codec == "wav_sidecar" and audio is not None:
        pcm_info = prepare_audio(audio, n, fps)
        if pcm_info:
            wav_path = os.path.join(out_dir, f"{stem}.wav")
            write_wav_sidecar(wav_path, pcm_info[0], pcm_info[1])
            paths.append(wav_path.replace("\\", "/"))
    return paths


def encode_video(
    images: torch.Tensor,
    spec: FormatSpec,
    *,
    path_info: tuple[str, str, str, str],
    fps: float,
    quality: int,
    colorspace_in: str,
    colorspace_out: str,
    audio: dict | None,
    pad_note: str | None,
    metadata: dict[str, Any] | None = None,
    colour=None,
    extras: "Extras | None" = None,
    cm=None,
    interop_id: str | None = None,
) -> dict[str, Any]:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    x = extras or Extras()
    warnings: list[str] = []
    n, h, w, c = images.shape
    if x.channels == "RGBA" and c < 4:
        if not spec.supports_alpha:
            warnings.append("channels RGBA: this format has no alpha channel - written as RGB.")
        elif torch.is_tensor(images):                        # an opaque alpha the file can carry
            images = torch.cat([images[..., :3], torch.ones_like(images[..., :1])], dim=-1)
            c = 4
    has_alpha = c >= 4 and (x.channels == "RGBA" or images[..., 3].max() > 0.001)
    if x.channels == "RGB":
        has_alpha = False                                    # the alpha the frames have is not written
    alpha_out = has_alpha and spec.supports_alpha
    if has_alpha and not spec.supports_alpha:
        warnings.append("Alpha channel dropped — this format does not support transparency.")
        has_alpha = False

    full_path, filename, rel_sub, folder_type = path_info
    # L9.52 (owner 2026-10-25: "make sure that .mov has no metadata in it"): the workflow / prompt never reach a video
    # container, whoever calls this - they go into '<name>.png' (write_workflow_png)
    metadata = {k: v for k, v in (metadata or {}).items()
                if str(k).lower() not in ("workflow", "prompt")} or None
    out_cs = colorspace_out
    if out_cs in ("same as input", "same", ""):
        out_cs = colorspace_in

    if spec.codec == "dnxhd" and (w < _DNXHR_MIN[0] or h < _DNXHR_MIN[1]):
        # libavcodec refuses with a bare EINVAL below this size (measured: 128x96 fails, 256x144 works)
        raise SaveVideoError(f"DNxHR needs frames of at least {_DNXHR_MIN[0]}x{_DNXHR_MIN[1]}; these are {w}x{h}. "
                             f"Pick ProRes or FFV1 for smaller frames.")

    if spec.sequence:
        paths = _encode_sequence_oiio(
            images, spec, full_path, filename, fps,
            colorspace_in, out_cs, audio, alpha_out,
            clamp_hdr=spec.codec in ("oiio_png16", "oiio_png", "oiio_dpx", "oiio_jpeg")
            or (spec.codec == "oiio_tiff" and spec.bit_depth < 32),
            quality=quality,
            extras=x,
            cm=cm,
            interop_id=interop_id,
        )
    elif spec.codec == "pil_gif":
        paths = _encode_gif(images, full_path, fps, quality, colorspace_in, out_cs, cm)
    elif spec.codec == "pil_webp":
        paths = _encode_webp(images, full_path, fps, quality, colorspace_in, out_cs, alpha_out, cm)
    else:
        paths = _encode_av_stream(
            images, spec, full_path, fps, quality,
            colorspace_in, out_cs, audio, alpha_out, metadata, colour, extras=x, cm=cm,
        )

    if pad_note:
        warnings.append(f"Padded to even size for 4:2:0 encoding: {pad_note}.")

    # ComfyUI's /view takes the file NAME and the folder it sits in, relative to output/temp. resolve_output_path
    # returns the path of the file (or of a sequence's folder) relative to that root, so split it here: passing the
    # whole relative path as "subfolder" made every preview URL point at <file>/<file> (no supported source).
    rel = (rel_sub or "").replace("\\", "/")
    if spec.sequence:
        ui_filename = os.path.basename(paths[0]) if paths else filename
        ui_subfolder = rel                       # the sequence's own folder holds the frames
    else:
        ui_filename = os.path.basename(full_path)
        ui_subfolder = rel.rsplit("/", 1)[0] if "/" in rel else ""

    saved_meta = {
        "filename": ui_filename,
        "subfolder": ui_subfolder,
        "type": folder_type,
    }
    external = folder_type == "external"          # outside output/: /view cannot serve it, preview from temp
    if getattr(images, "delivered", None) is not None and (not spec.browser_playable or external):
        preview = build_preview_from_saved(paths, spec)          # streamed: the frames are already gone
    else:
        preview = build_preview(images, fps, spec, saved_meta)

    return {
        "paths": paths,
        "filename": saved_meta["filename"],
        "subfolder": ui_subfolder,
        "type": folder_type,
        "format": spec.label,
        "frames": n,
        "fps": fps,
        "width": w,
        "height": h,
        "bit_depth": spec.bit_depth,
        "colorspace": out_cs,
        "alpha": alpha_out,
        "warnings": warnings,
        "preview": preview,
    }
