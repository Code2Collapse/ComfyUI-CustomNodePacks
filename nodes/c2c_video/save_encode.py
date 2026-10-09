"""Chunked encode pipeline for Save Video (C2C)."""

from __future__ import annotations

import json
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
    if naming and naming != "default":
        return naming
    if naming_resolved and naming_resolved.strip():
        return naming_resolved.strip()
    return read_comfy_setting("c2c.saveVideo.naming", "ComfyUI counter")


def pad_edge_replicate(images: torch.Tensor) -> tuple[torch.Tensor, str | None]:
    """Pad odd W/H by one pixel (edge replicate) for 4:2:0 codecs."""
    _, h, w, _ = images.shape
    pad_w = w % 2
    pad_h = h % 2
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


def _rgb_to_av_frame(rgb: np.ndarray, pix_fmt: str, bit_depth: int):
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
        return frame.reformat(format=pix_fmt, dst_colorspace=_BT709, dst_color_range=_RANGE_TV,
                              interpolation=_SWS_FLAGS)
    return frame.reformat(format=pix_fmt, interpolation=_SWS_FLAGS)


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
) -> tuple[str, str, str, str]:
    """Return (full_path, filename, subfolder_rel, folder_type)."""
    fp = _import_folder_paths()
    if save_output:
        out_dir = fp.get_output_directory()
        folder_type = "output"
    else:
        out_dir = os.path.join(fp.get_temp_directory(), "c2c_save_video")
        folder_type = "temp"
    os.makedirs(out_dir, exist_ok=True)

    prefix = (filename_prefix or "C2C/video").strip().replace("\\", "/")
    sub = (subfolder or "").strip().replace("\\", "/")

    if naming == "Folder Version":
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


def build_preview(images: torch.Tensor, fps: float, spec: FormatSpec, saved: dict) -> dict:
    if spec.browser_playable:
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
) -> list[str]:
    import av

    ProgressBar, interrupt = _progress_and_interrupt()
    n, h, w, c = images.shape
    channels = 4 if has_alpha and c >= 4 else 3
    bpf = bytes_per_frame(w, h, channels, np.float32)
    # By bytes too: each chunk is copied twice on its way to the encoder, so a whole-batch chunk cost 2x the batch
    # (L7.59: Save alone peaked at 2.5x). The encoder sees the same frames in the same order.
    chunk = max(1, min(64, frames_that_fit(w, h, channels, np.float32) or 1, _CHUNK_BYTES // max(1, bpf)))

    pix_fmt = spec.pix_fmt_for(has_alpha)
    # PyAV's muxer names differ from the file extension (mkv -> matroska), and muxer options must be given at open
    # time: Container.options is read-only in this PyAV build.
    av_format = {"mkv": "matroska"}.get(spec.container, spec.container)
    mux_opts = {"movflags": "+faststart"} if spec.container in ("mp4", "mov") else {}
    container = av.open(path, mode="w", format=av_format if spec.container != "dir" else None, options=mux_opts)
    created = True
    try:
        stream = container.add_stream(spec.codec, rate=Fraction(fps).limit_denominator(1_000_000))
        stream.width = w
        stream.height = h
        stream.pix_fmt = pix_fmt
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
        for start in range(0, n, chunk):
            interrupt()
            batch = images[start:start + chunk]
            arr = batch.cpu().numpy().astype(np.float32)
            if colorspace_out and colorspace_out != "same as input":
                arr = transform_chunk(arr, colorspace_in, colorspace_out)
            elif colorspace_in:
                arr = transform_chunk(arr, colorspace_in, colorspace_in)
            for i in range(arr.shape[0]):
                row = arr[i] if has_alpha else arr[i][..., :3]
                vf = _rgb_to_av_frame(row, pix_fmt, spec.bit_depth)
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
                colorspace_in: str, colorspace_out: str) -> list[str]:
    from PIL import Image

    ProgressBar, interrupt = _progress_and_interrupt()
    n = int(images.shape[0])
    frames: list[Image.Image] = []
    pbar = ProgressBar(n) if ProgressBar else None
    for i in range(n):
        interrupt()
        row = images[i].cpu().numpy().astype(np.float32)
        if colorspace_out and colorspace_out != "same as input":
            row = transform_chunk(row[None, ...], colorspace_in, colorspace_out)[0]
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
                 colorspace_in: str, colorspace_out: str, has_alpha: bool) -> list[str]:
    from PIL import Image

    ProgressBar, interrupt = _progress_and_interrupt()
    n = int(images.shape[0])
    frames: list[Image.Image] = []
    pbar = ProgressBar(n) if ProgressBar else None
    for i in range(n):
        interrupt()
        row = images[i].cpu().numpy().astype(np.float32)
        if colorspace_out and colorspace_out != "same as input":
            row = transform_chunk(row[None, ...], colorspace_in, colorspace_out)[0]
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
        if colorspace_out and colorspace_out != "same as input":
            row = transform_chunk(row[None, ...], colorspace_in, colorspace_out)[0]
        fname = f"{stem}.{i + 1:04d}.{ext}"
        fpath = os.path.join(out_dir, fname)
        h, w = row.shape[:2]
        if spec.codec == "oiio_png16":
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
) -> dict[str, Any]:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    n, h, w, c = images.shape
    has_alpha = c >= 4 and images[..., 3].max() > 0.001
    alpha_out = has_alpha and spec.supports_alpha
    warnings: list[str] = []
    if has_alpha and not spec.supports_alpha:
        warnings.append("Alpha channel dropped — this format does not support transparency.")
        has_alpha = False

    full_path, filename, rel_sub, folder_type = path_info
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
            clamp_hdr=spec.codec == "oiio_png16",
        )
    elif spec.codec == "pil_gif":
        paths = _encode_gif(images, full_path, fps, quality, colorspace_in, out_cs)
    elif spec.codec == "pil_webp":
        paths = _encode_webp(images, full_path, fps, quality, colorspace_in, out_cs, alpha_out)
    else:
        paths = _encode_av_stream(
            images, spec, full_path, fps, quality,
            colorspace_in, out_cs, audio, alpha_out,
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
