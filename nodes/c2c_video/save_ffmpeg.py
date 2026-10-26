"""Fast encode path for Save Video (C2C): frames piped to an ffmpeg subprocess (L2.53, owner A14).

Why: the PyAV path converts RGB -> YUV in Python's thread one frame at a time, then encodes. Piping the quantized RGB
to ffmpeg lets ffmpeg convert (slice-threaded) and encode in its own process while Python prepares the next frames.
Measured on this box (CPU encoders, 24 frames): 1080p H.264 66.9 -> 15.6 ms/frame, ProRes 422 HQ 89.9 -> 20.6,
H.265 10-bit 146.7 -> 47.6; 4K H.264 232.2 -> 53.2 (docs/evidence/L2.53).

Same pixels: ffmpeg's scale filter is given the PyAV path's own swscale parameters (BT.709, limited range, bicubic +
accurate rounding + full chroma), and the RGB is quantized the same way (clip, scale, round half to even). Measured:
the YUV planes are byte-identical to the PyAV path's for yuv420p, yuv422p, yuv422p10le, yuv420p10le and yuv444p10le.
The encoders are the same libraries (x264 / x265 / prores_ks / ...), possibly a different build.

Off switch: C2C_SAVE_FFMPEG=0 (the PyAV path is always the fallback: no ffmpeg, an old one, or an encoder it lacks).

PORTED FROM: HallettVisual/ComfyUI-4K-Video-Save-Load@7ae9ab9 video4k/media.py (MIT, Copyright (c) 2026 Matthew
Hallett): the FFMETADATA escaping (ffmetadata_text) and the exact NTSC rate rule (exact_rate). Marked below.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from fractions import Fraction
from typing import Any

import numpy as np

from .fast_decode import _NO_WINDOW, _big_pipe, ffmpeg_version, find_ffmpeg

_CLI_CODECS = frozenset({"libx264", "libx265", "prores_ks", "dnxhd", "ffv1", "libvpx-vp9",
                         "h264_nvenc", "hevc_nvenc", "av1_nvenc",
                         "libsvtav1", "cfhd", "utvideo", "png", "qtrle", "v210"})   # L9.40 matrix
_FLAGS = "bicubic+accurate_rnd+full_chroma_int+full_chroma_inp"
_ENCODERS: dict[str, frozenset[str]] = {}


def exact_rate(fps: float) -> Fraction:
    """fps as the container should get it, NTSC rates exactly: 23.976 / 23.98 -> 24000/1001, 29.97 -> 30000/1001,
    59.94 -> 60000/1001. Real rates like 24 sit 0.1% from their NTSC neighbour, far outside the tolerance."""
    fps = float(fps)
    # PORTED FROM: HallettVisual/ComfyUI-4K-Video-Save-Load@7ae9ab9 video4k/media.py exact_rate (MIT)
    base = round(fps * 1.001)
    if base and abs(fps * 1.001 / base - 1) < 5e-4:
        return Fraction(base * 1000, 1001)
    # END PORTED
    return Fraction(fps).limit_denominator(1_000_000)


def ffmetadata_text(metadata: dict[str, Any]) -> str:
    """An FFMETADATA file body (keys "workflow", "prompt", ... -> JSON), so a long workflow never meets the command
    line limit."""
    lines = [";FFMETADATA1"]
    for key, value in metadata.items():
        text = value if isinstance(value, str) else json.dumps(value)
        # PORTED FROM: HallettVisual/ComfyUI-4K-Video-Save-Load@7ae9ab9 video4k/media.py write_metadata_file (MIT)
        for char in ("\\", "=", ";", "#", "\n"):
            text = text.replace(char, "\\" + char)
        # END PORTED
        lines.append(f"{key}={text}")
    return "\n".join(lines) + "\n"


def _encoders(exe: str) -> frozenset[str]:
    if exe not in _ENCODERS:
        names: set[str] = set()
        try:
            out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, timeout=30,
                                 creationflags=_NO_WINDOW).stdout.decode("utf-8", "replace")
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
                    names.add(parts[1])
        except Exception:  # noqa: BLE001
            pass
        _ENCODERS[exe] = frozenset(names)
    return _ENCODERS[exe]


def ffmpeg_for(spec, pix: str | None = None) -> str | None:
    """The ffmpeg to encode ``spec`` with, or None for the PyAV path. A matrix spec (L9.40) also needs this ffmpeg's
    encoder to take the pixel format ``pix``; otherwise PyAV, which may (save_matrix.backend_for)."""
    if os.environ.get("C2C_SAVE_FFMPEG", "").strip() == "0":
        return None
    if spec.sequence or spec.codec not in _CLI_CODECS:
        return None
    exe = find_ffmpeg()
    if not exe:
        return None
    ver = ffmpeg_version(exe)
    if ver is None or ver < (5, 1):
        return None
    enc = _encoders(exe)
    if spec.codec not in enc or (spec.audio_codec and spec.audio_codec not in enc):
        return None
    if getattr(spec, "matrix", False):
        from .save_matrix import backend_for
        if backend_for(spec.codec, pix or spec.pix_fmt) != "ffmpeg":
            return None
    return exe


def _cli_options(opts: dict[str, str]) -> list[str]:
    out: list[str] = []
    for k, v in opts.items():
        out += [{"profile": "-profile:v", "b": "-b:v"}.get(k, "-" + k), str(v)]
    return out


class _Drain:
    def __init__(self, stream):
        self.tail: list[str] = []
        self._t = threading.Thread(target=self._run, args=(stream,), name="c2c-save-ffmpeg-log", daemon=True)
        self._t.start()

    def _run(self, stream) -> None:
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    self.tail = (self.tail + [line])[-20:]
        except (ValueError, OSError):
            pass

    def text(self) -> str:
        self._t.join(timeout=5)
        return " | ".join(self.tail[-6:]) or "no message"


def encode_with_ffmpeg(
    exe: str,
    images,
    spec,
    path: str,
    *,
    fps: float,
    has_alpha: bool,
    encoder_opts: dict[str, str],
    tags: dict[str, str] | None,
    metadata: dict[str, Any] | None,
    pcm: tuple[np.ndarray, int] | None,
    chunks,
    quantize,
    interrupt,
    progress,
    matrix: str = "bt709",
    out_range: str = "tv",
    carry: dict | None = None,
) -> list[str]:
    """Write ``images`` to ``path``. ``chunks`` yields float32 [k, H, W, C] numpy chunks (OCIO applied);
    ``quantize(chunk, bits)`` makes them integers exactly as the PyAV path does."""
    from .save_formats import SaveVideoError

    n, h, w, _c = images.shape
    deep = spec.bit_depth > 8
    src_pix = ("rgba64le" if has_alpha else "rgb48le") if deep else ("rgba" if has_alpha else "rgb24")
    pix = spec.pix_fmt_for(has_alpha)
    yuv = pix.startswith(("yuv", "nv12", "p010", "p016"))
    vf = (f"scale=out_color_matrix={matrix}:out_range={out_range}:flags={_FLAGS},format={pix}" if yuv
          else f"scale=flags={_FLAGS},format={pix}")
    if tags:
        # the frames leaving the filter carry the tags (the -color_* options alone were overridden by the filter
        # output's unspecified primaries / transfer - measured); setparams only labels, it changes no pixel
        params = [f"{k}={tags[k]}" for k in ("color_primaries", "color_trc", "colorspace") if tags.get(k)]
        if yuv:
            params.append(f"range={out_range}")
        if params:
            vf += ",setparams=" + ":".join(params)
    par = (carry or {}).get("pixel_aspect")
    if par:                                             # L9.55: the source's pixel aspect
        vf += f",setsar={int(par[0])}/{int(par[1])}"
    rate = exact_rate(fps)
    tmp = tempfile.mkdtemp(prefix="c2c_save_")
    args = [exe, "-hide_banner", "-nostdin", "-v", "error", "-y",
            "-filter_threads", str(max(1, min(16, os.cpu_count() or 4))),
            "-f", "rawvideo", "-pix_fmt", src_pix, "-s", f"{w}x{h}", "-framerate", f"{rate.numerator}/{rate.denominator}",
            "-i", "pipe:0"]
    inputs = 1
    maps = ["-map", "0:v"]
    if pcm is not None and pcm[0].shape[-1] > 0:
        audio_path = os.path.join(tmp, "audio.f32")
        np.ascontiguousarray(pcm[0].T.astype(np.float32)).tofile(audio_path)
        args += ["-f", "f32le", "-ar", str(int(pcm[1])), "-ac", str(int(pcm[0].shape[0])), "-i", audio_path]
        maps += ["-map", f"{inputs}:a"]
        inputs += 1
    meta_idx = None
    if metadata:
        meta_path = os.path.join(tmp, "metadata.txt")
        with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(ffmetadata_text(metadata))
        args += ["-f", "ffmetadata", "-i", meta_path]
        meta_idx = inputs
        inputs += 1
    args += maps
    if meta_idx is not None:
        args += ["-map_metadata", str(meta_idx)]
    args += ["-vf", vf, "-c:v", spec.codec, "-pix_fmt", pix] + _cli_options(encoder_opts)
    if spec.codec == "libx265":
        args += ["-x265-params", "log-level=error"]
    if spec.codec_tag:
        args += ["-tag:v", spec.codec_tag]
    if tags:
        for k in ("color_primaries", "color_trc", "colorspace", "color_range"):
            if tags.get(k):
                args += ["-" + k, tags[k]]
    if len(maps) > 2:
        args += ["-c:a", spec.audio_codec]
        if spec.audio_codec == "aac":
            args += ["-b:a", "192k"]
    if spec.container in ("mp4", "mov"):
        args += ["-movflags", "+faststart" + ("+use_metadata_tags" if metadata else "")]
    if (carry or {}).get("timecode") and spec.container in ("mov", "mp4", "mkv"):
        args += ["-timecode", str(carry["timecode"])]              # L9.55: the first frame's source timecode
    if (carry or {}).get("reel"):
        args += ["-metadata:s:v:0", f"reel_name={carry['reel']}"]
    args += [path]

    rfd, wfd = _big_pipe()
    try:
        proc = subprocess.Popen(args, stdin=rfd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                creationflags=_NO_WINDOW)
    except Exception as exc:
        os.close(rfd)
        os.close(wfd)
        _rmtree(tmp)
        raise SaveVideoError(f"ffmpeg did not start: {exc}") from exc
    os.close(rfd)
    drain = _Drain(proc.stderr)
    sink = os.fdopen(wfd, "wb")
    ok = False
    try:
        written = 0
        for arr in chunks:
            interrupt()
            q = quantize(arr, 16 if deep else 8)
            try:
                sink.write(memoryview(np.ascontiguousarray(q)).cast("B"))
            except (BrokenPipeError, OSError):
                raise SaveVideoError(f"ffmpeg stopped while encoding: {drain.text()}") from None
            written += arr.shape[0]
            progress(min(written, n), n)
        try:
            sink.close()
        except (BrokenPipeError, OSError):
            pass
        code = proc.wait()
        if code != 0:
            hint = ""
            if spec.hw == "nvenc":
                hint = (" NVENC could not start: an NVIDIA GPU with a current driver is needed (or the GPU is busy"
                        " with too many encode sessions). Pick MP4 H.264 or MP4 H.265 to encode on the CPU.")
            raise SaveVideoError(f"ffmpeg could not write {os.path.basename(path)}: {drain.text()}.{hint}")
        ok = True
    finally:
        if not ok:
            try:
                sink.close()
            except Exception:  # noqa: BLE001
                pass
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        try:
            proc.stderr.close()
        except Exception:  # noqa: BLE001
            pass
        _rmtree(tmp)
    return [path.replace("\\", "/")]


def _rmtree(path: str) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
