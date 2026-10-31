"""Fast decode path for Load Video (C2C) / Load Video Path (C2C): an ffmpeg subprocess (L2.53, owner A14).

Owner, 2026-10-15: take the 4K pack's fast path, with "Auto, with a choice": a ``decoder`` input
  * ``exact`` - today's reader (reader.py): PyAV, frame identity from the index table, everything it supports.
  * ``auto``  - the fast path only where it is FASTER and gives the SAME colour, else the exact reader.
  * ``fast``  - the fast path wherever it can run (8-bit 4:2:0 files), even where its colour is not identical.

What was measured (docs/evidence/L2.53/video4k_bench.json, CPU, 32 threads):
  * ffmpeg decodes, selects, scales and converts; Python only normalises. With ffmpeg converting YUV -> 16-bit RGB
    with the SAME swscale call the exact reader makes (matrix, range, flags, chroma siting pinned) the frames are
    bit-identical to the exact reader (max difference 0, 1080p and 4K, scaled, odd sizes, every selection kind).
    But on this CPU it is NOT faster: 4K 39.9-41.1 vs 32.3-36.0 ms/frame, 1080p 9.5-9.7 vs 9.5-9.6 (interleaved
    runs) - PyAV's reader already threads its decode and conversion.
  * On a CUDA device ``fast`` ships nv12 (a quarter of the bytes) and converts on the GPU, as the 4K pack does, with a
    bicubic, correctly sited chroma upsample instead of its nearest-neighbour one. Against the exact reader (same
    arithmetic, run on the CPU): mean 0.47/255, max 25.7/255 at saturated colour edges (nearest-neighbour: max
    255/255). Speed on a GPU: not measured here (the owner's numbers).
  * So ``auto`` is the exact reader today: neither fast variant met both conditions on the measured box.
    ``C2C_VIDEO_AUTO=fast`` lets auto take the identical ffmpeg path (after its self-check), ``C2C_VIDEO_AUTO=gpu``
    also the GPU path on CUDA - for a machine where those measure faster. ``C2C_VIDEO_FAST=rgb48`` makes ``fast`` use
    the identical conversion on CUDA too; ``C2C_VIDEO_FAST=0`` turns the fast path off.
  * Frame identity: every frame ffmpeg delivers carries its presentation timestamp (``-copyts`` + ``showinfo``), and
    is looked up in the same index table the exact reader uses. A frame the selection did not expect, a missing frame,
    a size change or an ffmpeg error stops the fast path; the caller then reads with the exact reader instead, so the
    result is always the exact reader's frames.
  * Windows pipes default to a 4 KB buffer: 4K frames crawled at 112 ms/frame through it, 28.5 ms with 1 MB.

PORTED FROM: HallettVisual/ComfyUI-4K-Video-Save-Load@7ae9ab9 video4k/media.py and video4k/accel.py (MIT,
Copyright (c) 2026 Matthew Hallett): ffmpeg discovery (find_ffmpeg), the fixed-size frame reader with recycled buffers
(_read_frames) and the nv12 -> RGB matrix arithmetic (_nv12_to_rgb). Each ported block is marked below.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .handle import C2CVideo

log = logging.getLogger("c2c_video.fast_decode")

DECODER_CHOICES = ["auto", "fast", "exact"]
DECODER_TIP = ("auto: the fast path only where it is faster with the same colour - measured, that is the exact reader "
               "on the tested machine. fast: ffmpeg decodes 8-bit 4:2:0 video (the same frames; on an NVIDIA GPU the "
               "colour conversion runs there, mean 0.5/255 from the exact reader). exact: the frame-accurate PyAV "
               "reader (10-bit, HDR, alpha, ProRes and image sequences always use it).")

# 8-bit 4:2:0 sources (the owner's scope for the fast path). nv12 is a chroma relayout of these, not a resample.
FAST_PIX_FMTS = frozenset({"yuv420p", "yuvj420p", "nv12", "nv21"})
_NV12_OK = frozenset({"yuv420p", "nv12"})          # GPU path: limited range only (yuvj would be range-converted)

# swscale names for the matrices probe.py decides on
_MATRIX_ARG = {"bt709": "bt709", "bt601": "bt601", "bt2020": "bt2020", "smpte240m": "smpte240m", "fcc": "fcc"}
# (Kr, Kb) per matrix, for the GPU conversion
_KRKB = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114), "bt2020": (0.2627, 0.0593),
         "smpte240m": (0.212, 0.087), "fcc": (0.30, 0.11)}

# The exact reader's swscale flags, spelled for ffmpeg's scale filter (reader._SWS_FLAGS / _SWS_SCALE_FLAGS).
_FLAGS_SAME_SIZE = "bicubic+accurate_rnd+full_chroma_int+full_chroma_inp"
_FLAGS_RESIZE = "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp"
# PyAV hands swscale no chroma location, so swscale's default applies: MPEG-2 "left" (h 0, v 128). The ffmpeg CLI
# takes the stream's own, which differed (measured: max 150/255 at colour edges until this was pinned).
_CHROMA_POS = "in_h_chr_pos=0:in_v_chr_pos=128"

# auto (when allowed by C2C_VIDEO_AUTO): below this many output pixels (frames x width x height) ffmpeg's start-up
# (~0.05-0.1 s measured) outweighs any gain.
AUTO_MIN_PIXELS = int(os.environ.get("C2C_VIDEO_FAST_MIN_PIXELS", str(1280 * 720 * 12)))
# auto: the self-check must agree to within this (float, per channel). Measured: 0.
SELFTEST_MAX_DIFF = 2.0 / 65535.0
_PIPE_BYTES = 1 << 20
_SEEK_MIN_SECONDS = 2.0


class FastDecodeError(RuntimeError):
    """The fast path could not deliver exactly the selected frames; read with the exact reader instead."""


class _Overshoot(Exception):
    """A seek landed after the first wanted frame: decode again from the start of the file."""


# ── ffmpeg discovery ─────────────────────────────────────────────────────────────────────────────────────────────
_FFMPEG: dict[str, object] = {}


def find_ffmpeg() -> str | None:
    """C2C_FFMPEG, then VHS_FORCE_FFMPEG_PATH, then ffmpeg on PATH, then imageio-ffmpeg's binary."""
    if "path" in _FFMPEG:
        return _FFMPEG["path"]  # type: ignore[return-value]
    # PORTED FROM: HallettVisual/ComfyUI-4K-Video-Save-Load@7ae9ab9 video4k/media.py _find_ffmpeg (MIT)
    found = None
    for env in ("C2C_FFMPEG", "VHS_FORCE_FFMPEG_PATH"):
        if os.environ.get(env):
            found = os.environ[env]
            break
    if found is None:
        found = shutil.which("ffmpeg")
    if found is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            found = get_ffmpeg_exe()
        except Exception:  # noqa: BLE001 - not installed, or its binary missing
            found = None
    # END PORTED
    _FFMPEG["path"] = found
    return found


def ffmpeg_version(exe: str | None = None) -> tuple[int, int] | None:
    exe = exe or find_ffmpeg()
    if not exe:
        return None
    key = f"version:{exe}"
    if key not in _FFMPEG:
        ver = None
        try:
            out = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, timeout=20,
                                 creationflags=_NO_WINDOW).stdout.decode("utf-8", "replace")
            m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", out)
            ver = (int(m.group(1)), int(m.group(2))) if m else (99, 0)   # a git build prints a hash: assume recent
        except Exception:  # noqa: BLE001
            ver = None
        _FFMPEG[key] = ver
    return _FFMPEG[key]  # type: ignore[return-value]


_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0      # CREATE_NO_WINDOW: no console flash inside ComfyUI


# ── eligibility and the decision ─────────────────────────────────────────────────────────────────────────────────
def ineligible_reason(h: C2CVideo) -> str | None:
    """Why the fast path cannot read ``h`` (None = it can)."""
    if getattr(h.source, "kind", "") != "file":
        return "image sequences use the exact reader"
    if h.has_alpha:
        return "sources with alpha use the exact reader"
    if int(h.bit_depth) > 8:
        return f"{h.bit_depth}-bit sources use the exact reader"
    if (h.pix_fmt or "") not in FAST_PIX_FMTS:
        return f"{h.pix_fmt} sources use the exact reader (the fast path takes 8-bit 4:2:0)"
    if (h.colour.matrix or "") not in _MATRIX_ARG:
        return f"colour matrix {h.colour.matrix} uses the exact reader"
    if not len(h):
        return "nothing selected"
    sel = h.selected_indices()
    if any(i < 0 for i in sel):
        return "black frames (a missing-frame or edge policy) use the exact reader"
    if any(b < a for a, b in zip(sel, sel[1:])):
        return "the selection is not in file order"
    exe = find_ffmpeg()
    if not exe:
        return "ffmpeg was not found (install it, put it on PATH, or pip install imageio-ffmpeg)"
    ver = ffmpeg_version(exe)
    if ver is None:
        return "ffmpeg did not run"
    if ver < (5, 1):
        return f"ffmpeg {ver[0]}.{ver[1]} is too old for the fast path (5.1 or newer)"
    return None


@dataclass(frozen=True)
class DecoderPlan:
    asked: str
    mode: str          # "exact" | "rgb48" | "nv12"
    note: str = ""

    @property
    def label(self) -> str:
        return {"exact": "exact (PyAV reader)", "rgb48": "fast (ffmpeg, identical conversion)",
                "nv12": "fast (ffmpeg nv12, GPU conversion)"}[self.mode]


def plan_decoder(h: C2CVideo, decoder: str, *, cuda: bool = False) -> DecoderPlan:
    """Which reader a load uses. ``h`` is the LOADED handle (selection, crop and size applied)."""
    asked = decoder if decoder in DECODER_CHOICES else "auto"
    if asked == "exact" or os.environ.get("C2C_VIDEO_FAST", "").strip() == "0":
        return DecoderPlan(asked, "exact", "" if asked == "exact" else "C2C_VIDEO_FAST=0")
    why = ineligible_reason(h)
    if why:
        return DecoderPlan(asked, "exact", why)
    if asked == "fast":
        want_nv12 = cuda and os.environ.get("C2C_VIDEO_FAST", "").strip().lower() != "rgb48" and _nv12_ok(h)
        return DecoderPlan(asked, "nv12" if want_nv12 else "rgb48")
    # auto (see the module notes: measured, the exact reader is as fast on the CPU; the GPU path is not identical)
    allow = os.environ.get("C2C_VIDEO_AUTO", "").strip().lower()
    if allow not in ("fast", "gpu"):
        return DecoderPlan(asked, "exact", "auto: the exact reader measured as fast as ffmpeg here")
    if len(h) * int(h.width) * int(h.height) < AUTO_MIN_PIXELS:
        return DecoderPlan(asked, "exact", "small load: ffmpeg's start-up outweighs any gain")
    if allow == "gpu" and cuda and _nv12_ok(h):
        return DecoderPlan(asked, "nv12", "C2C_VIDEO_AUTO=gpu")
    ok, diff = selftest()
    if not ok:
        return DecoderPlan(asked, "exact", f"this ffmpeg's conversion differs from the exact reader ({diff})")
    return DecoderPlan(asked, "rgb48", "C2C_VIDEO_AUTO=fast")


def _nv12_ok(h: C2CVideo) -> bool:
    return (h.crop is None and h.pix_fmt in _NV12_OK and (h.colour.range or "tv") == "tv"
            and h.width % 2 == 0 and h.height % 2 == 0)


# ── the self-check behind auto (C2C_VIDEO_AUTO=fast) ───────────────────────────────────────────────────────────────────────────────────
def selftest() -> tuple[bool, str]:
    """Decode a small colourful clip with both readers (unscaled and scaled to an odd size) and compare. Cached per
    ffmpeg binary for the life of the process."""
    exe = find_ffmpeg()
    key = f"selftest:{exe}"
    if key in _FFMPEG:
        return _FFMPEG[key]  # type: ignore[return-value]
    result: tuple[bool, str]
    folder = tempfile.mkdtemp(prefix="c2c_fast_selftest_")
    try:
        from .probe import probe_file
        from .reader import read

        path = os.path.join(folder, "selftest.mp4")
        _write_selftest_clip(path)
        h = probe_file(path)
        worst = 0.0
        for hh in (h, h.scaled(61, 37)):
            want = read(hh, range(len(hh)), fmt="float32")
            got = np.concatenate(list(iter_chunks_fast(hh, 8, "rgb48")))
            if got.shape != want.shape:
                raise FastDecodeError(f"shape {got.shape} != {want.shape}")
            worst = max(worst, float(np.abs(got - want).max()))
        ok = worst <= SELFTEST_MAX_DIFF
        result = (ok, f"max difference {worst * 255:.4f}/255")
        log.info("[c2c_video] fast decode self-check with %s: %s (%s)", exe, "identical" if ok else "DIFFERS",
                 result[1])
    except Exception as exc:  # noqa: BLE001 - any failure means: do not use the fast path automatically
        result = (False, f"self-check failed: {exc}")
        log.warning("[c2c_video] fast decode self-check failed: %s", exc)
    finally:
        try:
            from .reader import drain_container_pool
            drain_container_pool()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(folder, ignore_errors=True)
    _FFMPEG[key] = result
    return result


def _write_selftest_clip(path: str) -> None:
    """Three 96x64 frames of saturated colour blocks with hard edges (where chroma handling shows), tagged bt709 tv."""
    import av

    from .testmedia import _rgb_to_yuv, _tag_bt709

    rng = np.random.default_rng(7)
    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.options = {"qscale": "2"}
        _tag_bt709(stream)
        for i in range(3):
            blocks = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
            rgb = np.ascontiguousarray(np.kron(blocks, np.ones((8, 8, 1), dtype=np.uint8)))
            frame = _rgb_to_yuv(rgb, "yuv420p")
            frame.pts = i
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
    finally:
        container.close()


# ── pipes ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _big_pipe() -> tuple[int, int]:
    """(read_fd, write_fd) with a 1 MB buffer where the OS allows it."""
    if sys.platform == "win32":
        import _winapi
        import msvcrt

        rh, wh = _winapi.CreatePipe(None, _PIPE_BYTES)
        return msvcrt.open_osfhandle(rh, os.O_RDONLY), msvcrt.open_osfhandle(wh, 0)
    r, w = os.pipe()
    try:
        import fcntl

        fcntl.fcntl(w, 1031, _PIPE_BYTES)          # F_SETPIPE_SZ (Linux); capped by /proc/sys/fs/pipe-max-size
    except Exception:  # noqa: BLE001 - macOS / no permission: the default buffer only costs speed
        pass
    return r, w


def _read_frames(stream, frame_bytes: int, depth: int = 3) -> Iterator[bytearray]:
    """Fixed-size frames from a pipe, read ahead on a worker thread into recycled buffers; the buffer being yielded is
    reused only once the consumer asks for the next frame."""
    # PORTED FROM: HallettVisual/ComfyUI-4K-Video-Save-Load@7ae9ab9 video4k/accel.py read_frames (MIT)
    filled: queue.Queue = queue.Queue()
    free: queue.Queue = queue.Queue()
    for _ in range(depth):
        free.put(bytearray(frame_bytes))

    def read() -> None:
        try:
            while True:
                frame = free.get()
                if frame is None:
                    return
                view = memoryview(frame)
                offset = 0
                while offset < frame_bytes:
                    got = stream.readinto(view[offset:])
                    if not got:
                        break
                    offset += got
                if offset < frame_bytes:
                    break
                filled.put(frame)
        except (ValueError, OSError):
            pass                                    # the consumer closed the pipe to stop us
        filled.put(None)

    worker = threading.Thread(target=read, name="c2c-fast-decode-read", daemon=True)
    worker.start()
    try:
        while True:
            frame = filled.get()
            if frame is None:
                return
            yield frame
            free.put(frame)
    finally:
        free.put(None)
    # END PORTED


_SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\b.*?\bs:(\d+)x(\d+)")


class _Stderr:
    """Drains ffmpeg's stderr: showinfo's (pts, w, h) per frame into a queue, the rest kept for error messages."""

    def __init__(self, stream):
        self.frames: queue.Queue = queue.Queue()
        self.tail: list[str] = []
        self._t = threading.Thread(target=self._run, args=(stream,), name="c2c-fast-decode-log", daemon=True)
        self._t.start()

    def _run(self, stream) -> None:
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                m = _SHOWINFO.search(line)
                if m:
                    self.frames.put((int(m.group(2)), int(m.group(3)), int(m.group(4))))
                elif line:
                    self.tail = (self.tail + [line])[-30:]
        except (ValueError, OSError):
            pass
        self.frames.put(None)

    def message(self) -> str:
        useful = [t for t in self.tail if not t.lstrip().startswith(("Stream", "Input", "Output", "Duration",
                                                                     "Metadata", "Press", "  "))]
        return " | ".join(useful[-4:]) or "no message"


# ── the reader ───────────────────────────────────────────────────────────────────────────────────────────────────
def iter_chunks_fast(h: C2CVideo, chunk_frames: int, mode: str = "rgb48", *, device=None) -> Iterator[np.ndarray]:
    """The selection of ``h`` as float32 [<=chunk_frames, H, W, 3] chunks, like reader.iter_chunks(fmt="float32").
    Raises FastDecodeError when it cannot deliver exactly those frames (the caller re-reads with the exact reader)."""
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    why = ineligible_reason(h)
    if why:
        raise FastDecodeError(why)
    if mode == "nv12" and (h.crop is not None or h.width % 2 or h.height % 2):
        mode = "rgb48"
    try:
        yield from _run(h, chunk_frames, mode, device, seek=True)
    except _Overshoot:
        yield from _run(h, chunk_frames, mode, device, seek=False)


def _filters(h: C2CVideo, mode: str, p0: int, p1: int, step: int | None, span: int) -> str:
    parts = [f"select='gte(pts\\,{p0})*lte(pts\\,{p1})'"]
    if step and step > 1:
        parts.append(f"select='not(mod(n\\,{step}))*lte(n\\,{span})'")
    parts.append("showinfo=checksum=0")
    rot = h.rotation in (90, 270)
    if mode == "nv12":
        if h.is_scaled:
            w0, h0 = (h.height, h.width) if rot else (h.width, h.height)
            parts.append(f"scale={w0}:{h0}:flags=lanczos")
        parts.append("format=nv12")
        return ",".join(parts)
    m = _MATRIX_ARG[h.colour.matrix or "bt709"]
    rng = "pc" if (h.colour.range or "tv") == "pc" else "tv"
    common = f"in_color_matrix={m}:in_range={rng}:out_range=pc:{_CHROMA_POS}"
    if h.crop is None and h.is_scaled:
        w0, h0 = (h.height, h.width) if rot else (h.width, h.height)
        parts.append(f"scale={w0}:{h0}:{common}:flags={_FLAGS_RESIZE}")
    else:
        parts.append(f"scale={common}:flags={_FLAGS_SAME_SIZE}")
    parts.append("format=rgb48le")
    return ",".join(parts)


def _coded_size(h: C2CVideo) -> tuple[int, int] | None:
    """The size ffmpeg delivers (before our rotation), when known in advance."""
    if h.crop is not None:
        sw, sh = h.crop_source
    else:
        sw, sh = h.width, h.height
    return (sh, sw) if h.rotation in (90, 270) else (sw, sh)


def _run(h: C2CVideo, chunk_frames: int, mode: str, device, *, seek: bool) -> Iterator[np.ndarray]:
    from .probe import get_index_table
    from .reader import _ROT_K, _ChunkSink, _crop_and_fit, _out_shape, _white_level, _write

    exe = find_ffmpeg()
    sel = list(h.selected_indices())
    n = len(sel)
    table = get_index_table(h.source.stat_key, stream_index=h.stream_index)
    first, last = sel[0], sel[-1]
    p0, p1 = table.pts_of(first), table.pts_of(last)
    distinct = sorted(set(sel))
    rel = [i - first for i in distinct]
    diffs = {b - a for a, b in zip(rel, rel[1:])}
    step = diffs.pop() if len(diffs) == 1 else None
    w_out, h_out = _coded_size(h)
    k_rot = _ROT_K.get(h.rotation, 0)

    args = [exe, "-hide_banner", "-nostdin", "-nostats", "-v", "info",
            "-filter_threads", str(max(1, min(16, os.cpu_count() or 4))), "-copyts", "-noautorotate"]
    start_s = float(p0 * table.time_base)
    if seek and start_s > _SEEK_MIN_SECONDS:
        args += ["-seek_timestamp", "1", "-noaccurate_seek", "-ss", f"{start_s - 0.5:.6f}"]
    args += ["-i", h.source.path, "-map", f"0:v:{h.stream_index}", "-an", "-sn", "-dn",
             "-vf", _filters(h, mode, p0, p1, step, rel[-1]),
             "-fps_mode", "passthrough"]
    expected = len(distinct) if step else (last - first + 1)
    args += ["-frames:v", str(expected), "-f", "rawvideo", "-"]

    if mode == "nv12":
        frame_bytes = w_out * h_out * 3 // 2
        import torch

        dev = device if device is not None else _compute_device()
        kr, kb = _KRKB[h.colour.matrix or "bt709"]
    else:
        frame_bytes = w_out * h_out * 6
        level = _white_level(h.pix_fmt, h.colour.matrix, h.colour.range, "rgb48le")

    rfd, wfd = _big_pipe()
    try:
        proc = subprocess.Popen(args, stdout=wfd, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                creationflags=_NO_WINDOW)
    except Exception as exc:
        os.close(rfd)
        os.close(wfd)
        raise FastDecodeError(f"ffmpeg did not start: {exc}") from exc
    os.close(wfd)
    out_pipe = os.fdopen(rfd, "rb", buffering=0)
    err = _Stderr(proc.stderr)
    sink = _ChunkSink(n, chunk_frames, _out_shape(h), np.float32)
    pos = 0
    size_ref: tuple[int, int] | None = None
    delivered = 0
    try:
        for buf in _read_frames(out_pipe, frame_bytes):
            try:
                info = err.frames.get(timeout=60)
            except queue.Empty:
                info = None
            if info is None:
                raise FastDecodeError(f"ffmpeg gave a frame without its timestamp ({err.message()})")
            pts, fw, fh = info
            if size_ref is None:
                size_ref = (fw, fh)
                if h.crop is not None or not h.is_scaled:
                    if (fw, fh) != (w_out, h_out):
                        raise FastDecodeError(f"ffmpeg decodes {fw}x{fh}, the probe said {w_out}x{h_out}")
            elif (fw, fh) != size_ref:
                raise FastDecodeError("the frame size changes inside the file")
            idx = table.index_of_pts(pts)
            if idx is None:
                raise FastDecodeError(f"a frame at timestamp {pts} is not in the index")
            if delivered == 0 and idx > first:
                if seek:
                    raise _Overshoot()
                raise FastDecodeError(f"frame {first} was not decoded (the first was {idx})")
            delivered += 1
            if idx < sel[pos]:
                continue                                       # between wanted frames (range select)
            if idx > sel[pos]:
                raise FastDecodeError(f"frame {sel[pos]} was not decoded")
            slot = sink.slot()
            if mode == "nv12":
                rgb = _nv12_to_rgb(buf, w_out, h_out, kr, kb, dev)
                if k_rot:
                    rgb = torch.rot90(rgb, k_rot, (0, 1))
                torch.from_numpy(slot).copy_(rgb)
            else:
                raw = np.frombuffer(buf, dtype=np.uint16).reshape(h_out, w_out, 3)
                if h.crop is not None:
                    sw, sh = h.crop_source
                    full = np.empty((sh, sw, 3), dtype=np.float32)
                    _write(raw, level, "float32", full, k_rot)
                    slot[...] = _crop_and_fit(full, h)
                else:
                    _write(raw, level, "float32", slot, k_rot)
            src = slot
            first_row = True
            while pos < n and sel[pos] == idx:
                if not first_row:
                    sink.slot()[...] = src
                first_row = False
                pos += 1
                full_chunk = sink.commit()
                if full_chunk is not None:
                    yield full_chunk
            if pos >= n:
                break
        if pos < n:
            raise FastDecodeError(f"ffmpeg stopped before frame {sel[pos]} ({err.message()})")
        rest = sink.rest()
        if rest is not None:
            yield rest
    finally:
        try:
            out_pipe.close()
        except Exception:  # noqa: BLE001
            pass
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.stderr.close()
        except Exception:  # noqa: BLE001
            pass


def _compute_device():
    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:  # noqa: BLE001
        import torch

        return torch.device("cpu")


def device_is_cuda() -> bool:
    try:
        return getattr(_compute_device(), "type", "cpu") == "cuda"
    except Exception:  # noqa: BLE001
        return False


# ── nv12 -> RGB on the compute device ────────────────────────────────────────────────────────────────────────────
def _bicubic(x: float, b: float = 0.0, c: float = 0.6) -> float:
    """swscale's bicubic kernel (B = 0, C = 0.6)."""
    x = abs(x)
    if x < 1:
        return ((12 - 9 * b - 6 * c) * x ** 3 + (-18 + 12 * b + 6 * c) * x ** 2 + (6 - 2 * b)) / 6
    if x < 2:
        return ((-b - 6 * c) * x ** 3 + (6 * b + 30 * c) * x ** 2 + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) / 6
    return 0.0


# 2x chroma upsampling taps (offset -> weight). Horizontal: chroma co-sited with even luma columns ("left");
# vertical: chroma between two luma rows (centre). Each output phase sums to 1.
_TAPS_H = ({0: 1.0}, {-1: _bicubic(1.5), 0: _bicubic(0.5), 1: _bicubic(0.5), 2: _bicubic(1.5)})
_TAPS_V = ({-2: _bicubic(1.75), -1: _bicubic(0.75), 0: _bicubic(0.25), 1: _bicubic(1.25)},
           {-1: _bicubic(1.25), 0: _bicubic(0.25), 1: _bicubic(0.75), 2: _bicubic(1.75)})


def _up2(c, dim: int, taps):
    import torch
    import torch.nn.functional as F

    n = c.shape[dim]
    pad = (2, 2, 0, 0) if dim == 3 else (0, 0, 2, 2)
    cp = F.pad(c, pad, mode="replicate")
    phases = []
    for tp in taps:
        acc = None
        for off, wt in tp.items():
            part = cp.narrow(dim, 2 + off, n) * wt
            acc = part if acc is None else acc + part
        phases.append(acc)
    out = torch.stack(phases, dim + 1)
    shape = list(c.shape)
    shape[dim] *= 2
    return out.reshape(shape)


def _nv12_to_rgb(frame: bytearray, width: int, height: int, kr: float, kb: float, device):
    """One limited-range nv12 frame -> [H, W, 3] float32 RGB on ``device``."""
    import torch

    # PORTED FROM: HallettVisual/ComfyUI-4K-Video-Save-Load@7ae9ab9 video4k/accel.py nv12_to_rgb (MIT) - the plane
    # split and the matrix; the chroma upsample is ours (bicubic, sited like swscale) instead of repeat_interleave.
    plane = torch.frombuffer(frame, dtype=torch.uint8).to(device, non_blocking=True)
    y = plane[:width * height].view(1, 1, height, width).float()
    uv = plane[width * height:].view(1, height // 2, width // 2, 2).permute(0, 3, 1, 2).float()
    uv = _up2(_up2(uv, 2, _TAPS_V), 3, _TAPS_H)
    y = (y - 16) / 219
    u = (uv[:, 0:1] - 128) / 224
    v = (uv[:, 1:2] - 128) / 224
    kg = 1 - kr - kb
    r = y + (2 - 2 * kr) * v
    b = y + (2 - 2 * kb) * u
    g = y - (2 * kr * (1 - kr) / kg) * v - (2 * kb * (1 - kb) / kg) * u
    rgb = torch.cat((r, g, b), 1).clamp_(0, 1).squeeze(0).permute(1, 2, 0)
    # END PORTED
    return rgb.to("cpu")


def iter_chunks_for(h: C2CVideo, chunk_frames: int, plan: DecoderPlan) -> Iterator[np.ndarray]:
    """The chunks a plan reads (float32). The exact reader for mode "exact"."""
    if plan.mode == "exact":
        from .reader import iter_chunks

        return iter_chunks(h, chunk_frames, fmt="float32")
    return iter_chunks_fast(h, chunk_frames, plan.mode)
