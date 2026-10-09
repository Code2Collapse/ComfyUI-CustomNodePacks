"""Frame-accurate decode-on-demand video reader.

Nothing here holds a clip in memory: ``read`` decodes exactly the frames
asked for (one seek per run of nearby frames), ``iter_chunks`` streams the
selection in order through a private container. Frame identity comes from
the file's index table (probe.py) - presentation timestamps, never "the
n-th frame the decoder happened to emit"."""

from __future__ import annotations

import atexit
import bisect
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Literal

import numpy as np

from .handle import C2CVideo, FileSource, SequenceSource
from .probe import IndexTable, get_index_table, rebuild_exact_index

try:  # always present inside ComfyUI; the numpy path covers tools without it
    import torch as _torch
except ImportError:  # pragma: no cover
    _torch = None

logger = logging.getLogger("c2c_video.reader")

OutputFmt = Literal["uint8", "uint16", "float16", "float32"]


# ── container pool ───────────────────────────────────────────────────────
# Opening a container costs a header parse (and, for MP4, the whole sample
# table), so random reads reuse a few open ones. Each entry has its own lock:
# one decoder is never driven from two threads, and an evicted container is
# closed only under ITS lock - never out from under a decode in progress.

_POOL: OrderedDict[str, "_ContainerEntry"] = OrderedDict()
_MAX_POOL = 4
_POOL_GUARD = threading.Lock()
_REBUILD_LOGGED: set[tuple[str, int, int]] = set()


def _open_container(path: str):
    import av

    try:
        container = av.open(path, metadata_errors="ignore")
    except TypeError:
        container = av.open(path)
    for st in container.streams.video:
        try:
            st.thread_type = "AUTO"  # frame + slice threads inside the decoder
        except Exception:
            pass
    return container


def _quiesce(container) -> None:
    """Drain every opened video decoder before the container is released (L2.36).

    A selection that stops mid-file (a cap, every-nth, a random read) leaves the frame-threaded decoder with
    frames in flight in its worker threads. PyAV frees the decoder when the Stream object is deallocated - after
    close(), on a refcount drop or in a cyclic-GC pass at any later moment - and avcodec_free_context then waits
    for those workers (the suite hang: SleepConditionVariableSRW inside avcodec_free_context, called from
    av/stream.pyd). Sending EOF here finishes the in-flight frames, so every worker is idle when the free comes."""
    try:
        streams = list(container.streams.video)
    except Exception:
        return
    for st in streams:
        try:
            cc = st.codec_context
            if cc.is_open:
                cc.decode(None)
        except Exception:
            pass


def _release(container) -> None:
    _quiesce(container)
    try:
        container.close()
    except Exception:
        pass


class _ContainerEntry:
    def __init__(self, path: str):
        self.path = path
        self.container = _open_container(path)
        self.lock = threading.Lock()
        self.closed = False

    def close(self) -> None:
        with self.lock:
            if not self.closed:
                self.closed = True
                _release(self.container)


def _pool_get(path: str) -> _ContainerEntry:
    key = os.path.normpath(os.path.abspath(path))
    victims: list[_ContainerEntry] = []
    with _POOL_GUARD:
        ent = _POOL.pop(key, None)
        if ent is None or ent.closed:
            while len(_POOL) >= _MAX_POOL:
                victims.append(_POOL.popitem(last=False)[1])
            ent = _ContainerEntry(key)
        _POOL[key] = ent
    for v in victims:  # outside the guard: may wait for a decode to finish
        v.close()
    return ent


@contextmanager
def _pooled(path: str):
    """A locked, open pooled container for ``path``."""
    while True:
        ent = _pool_get(path)
        ent.lock.acquire()
        if not ent.closed:
            break
        ent.lock.release()  # evicted between lookup and lock: fetch again
    try:
        yield ent.container
    finally:
        ent.lock.release()


def drain_container_pool() -> None:
    with _POOL_GUARD:
        victims = list(_POOL.values())
        _POOL.clear()
    for v in victims:
        v.close()


atexit.register(drain_container_pool)


# swscale flags (libswscale/swscale.h). PyAV 17 hands `interpolation` to the
# SwsContext as its raw flags, so an int reaches swscale unchanged.
# BICUBIC | ACCURATE_RND | FULL_CHR_H_INT | FULL_CHR_H_INP. PyAV's default
# (BILINEAR alone) was MEASURED broken here: 10-bit 4:2:0 -> rgb48 came out
# with a mean error of 55 levels (max 255) and 10-bit -> rgb24 truncated
# (bias -1.9 levels). With these flags: 0.04 levels, ~10 ms per 4K frame.
_SWS_FLAGS = 0x4 | 0x40000 | 0x2000 | 0x4000
# When the frame is also RESIZED (custom_width/height): Lanczos instead of
# bicubic, the sharper filter for downscales (VHS resizes with lanczos too).
_SWS_SCALE_FLAGS = 0x200 | 0x40000 | 0x2000 | 0x4000

# AVColorSpace / AVColorRange (libavutil/pixfmt.h). PyAV 17 converts through
# sws_scale_frame, which takes the matrix from the FRAME's own metadata; its
# src_colorspace argument only maps four values and silently ignores BT.2020.
# So the decision made at probe time is stamped onto the frame instead.
# swscale has no BT.2020 constant-luminance path; NCL is the nearest.
_AVCOL_SPC = {"bt709": 1, "fcc": 4, "bt601": 6, "smpte240m": 7, "bt2020": 9}
_AVCOL_RANGE = {"tv": 1, "pc": 2}

_TLS = threading.local()
_WHITE: dict[tuple, np.ndarray] = {}


def _reformatter():
    """One VideoReformatter per thread: it keeps its SwsContext between calls
    with the same parameters instead of rebuilding it for every frame."""
    r = getattr(_TLS, "reformatter", None)
    if r is None:
        import av.video.reformatter as vr

        r = _TLS.reformatter = vr.VideoReformatter()
    return r


def _stamp(frame, matrix: str | None, in_range: str | None) -> None:
    cs = _AVCOL_SPC.get((matrix or "").lower())
    if cs is not None:
        frame.colorspace = cs
    rng = _AVCOL_RANGE.get((in_range or "").lower())
    if rng is not None:
        frame.color_range = rng


def _to_rgb(frame, out_pix: str, width: int | None = None, height: int | None = None):
    flags = _SWS_FLAGS if width is None and height is None else _SWS_SCALE_FLAGS
    return _reformatter().reformat(frame, width=width, height=height, format=out_pix,
                                   dst_color_range=2, interpolation=flags)


def _white_level(src_pix: str, matrix: str | None, in_range: str | None, out_pix: str) -> np.ndarray:
    """What full white of ``src_pix`` becomes in ``out_pix``, per channel.

    swscale's 16-bit RGB output from YUV puts white at 65280, not 65535
    (float reads would top out at 0.996), and an RGB or alpha source may
    differ again. Rather than hard-code one FFmpeg build's constant, push a
    white frame through the very same conversion once and measure it."""
    key = (src_pix, matrix, in_range, out_pix)
    hit = _WHITE.get(key)
    if hit is not None:
        return hit
    import av

    full = 255.0 if out_pix in ("rgb24", "rgba") else 65535.0
    n = 4 if out_pix.startswith("rgba") else 3
    level = np.full(n, full, dtype=np.float32)
    try:
        # The white is made at 16 bits so it lands on the SOURCE format's own
        # nominal white at any depth. Made from 8-bit 255 it arrives as 65280
        # and every 10/12/16-bit source was over-scaled by 0.4% (measured on
        # ProRes 4444 XQ, CineForm 12-bit, 16-bit PNG-in-MOV, FFV1 16-bit).
        white = av.VideoFrame.from_ndarray(np.full((16, 64, n), 65535, np.uint16),
                                           format="rgba64le" if n == 4 else "rgb48le")
        rng = _AVCOL_RANGE.get((in_range or "").lower(), 2)
        enc = white.reformat(format=src_pix, dst_colorspace=1, dst_color_range=rng,
                             interpolation=_SWS_FLAGS)
        _stamp(enc, matrix, in_range)
        got = _to_rgb(enc, out_pix).to_ndarray().reshape(-1, n)
        measured = np.median(got.astype(np.float32), axis=0)
        # a sane white is near full scale; anything else means the probe
        # itself failed and full scale is the safer assumption
        if np.all(measured > full * 0.9):
            level = measured
    except Exception as exc:  # exotic/hardware formats swscale cannot write
        logger.debug("white-level probe failed for %s -> %s: %s", src_pix, out_pix, exc)
    _WHITE[key] = level
    return level


# np.rot90's k for each clockwise display rotation
_ROT_K = {90: 3, 180: 2, 270: 1}


def _write_numpy(raw: np.ndarray, level: np.ndarray, fmt: OutputFmt, out: np.ndarray, k: int) -> None:
    """Fallback when torch is missing (tests, tools). Same numbers, one core."""
    if k:
        raw = np.rot90(raw, k)
    src_full = 255.0 if raw.dtype == np.uint8 else 65535.0
    if fmt in ("float32", "float16"):
        out[...] = raw.astype(np.float32) * (1.0 / level).astype(np.float32)
        return
    if np.all(level == src_full) and raw.dtype == out.dtype:
        out[...] = raw
        return
    u16 = np.clip(np.rint(raw.astype(np.float32) * (65535.0 / level)), 0, 65535)
    out[...] = u16 if fmt == "uint16" else np.rint(u16 / 257.0)


def _write(raw: np.ndarray, level: np.ndarray, fmt: OutputFmt, out: np.ndarray, k: int) -> None:
    """swscale's RGB(A) ``raw`` (uint8 or uint16) -> ``out`` in ``fmt``,
    normalised so the measured white is full scale, turned upright by k
    quarter-turns. torch's elementwise kernels use every core: measured on a
    4K frame, float 80 -> 14 ms and 10-bit -> uint8 260 -> 27 ms against
    single-threaded numpy, with identical results. uint8 is the uint16 image
    rounded (rint(u16 / 257)), so reads at different depths agree."""
    if _torch is None:
        _write_numpy(raw, level, fmt, out, k)
        return
    t = _torch
    src_full = 255.0 if raw.dtype == np.uint8 else 65535.0
    exact = bool(np.all(level == src_full))
    src = t.from_numpy(raw)
    dst = t.from_numpy(out)
    if exact and raw.dtype == out.dtype:  # 8-bit -> uint8, or 16-bit -> uint16 as-is
        dst.copy_(t.rot90(src, k, (0, 1)) if k else src)
        return
    if fmt == "float32" and not k:
        dst.copy_(src)  # the cast happens in the copy: no temporary
        dst.mul_(t.from_numpy((1.0 / level).astype(np.float32)))
        return
    work = src.to(t.float32)
    if fmt in ("float32", "float16"):
        work.mul_(t.from_numpy((1.0 / level).astype(np.float32)))
    else:
        work.mul_(t.from_numpy((65535.0 / level).astype(np.float32))).round_().clamp_(0, 65535)
        if fmt == "uint8":
            work.div_(257.0).round_()
    dst.copy_(t.rot90(work, k, (0, 1)) if k else work)


def _convert_into(frame, handle: C2CVideo, fmt: OutputFmt, out: np.ndarray) -> None:
    """Decoded frame -> ``out`` ([H,W,C] slot of the result, already the
    upright size). 8-bit sources read as uint8 go straight through swscale;
    everything else goes through 16-bit and is normalised to the measured
    white (see _white_level), so all four output formats agree."""
    matrix, in_range = handle.colour.matrix, handle.colour.range
    _stamp(frame, matrix, in_range)
    alpha = handle.has_alpha
    # a stream that changes size mid-file is scaled back to the probed size
    w0, h0 = (handle.height, handle.width) if handle.rotation in (90, 270) else (handle.width, handle.height)
    size = (None, None) if (frame.width, frame.height) == (w0, h0) else (w0, h0)
    if fmt == "uint8" and handle.bit_depth <= 8:
        pix = "rgba" if alpha else "rgb24"
    else:
        pix = "rgba64le" if alpha else "rgb48le"
    raw = _to_rgb(frame, pix, *size).to_ndarray()
    level = _white_level(frame.format.name, matrix, in_range, pix)
    _write(raw, level, fmt, out, _ROT_K.get(handle.rotation, 0))


def _frame_to_ndarray(frame, handle: C2CVideo, fmt: OutputFmt) -> np.ndarray:
    """One decoded frame as a fresh array (tools and tests; readers write in place)."""
    out = np.empty(_out_shape(handle), dtype=_NP_OUT[fmt])
    _convert_into(frame, handle, fmt, out)
    return out


_OIIO_READ = {"uint8": "uint8", "uint16": "uint16", "float16": "half", "float32": "float"}
_NP_OUT = {"uint8": np.uint8, "uint16": np.uint16, "float16": np.float16, "float32": np.float32}


def _to_rgb_channels(arr: np.ndarray, alpha_channel: int, want_alpha: bool) -> np.ndarray:
    """Any channel layout -> RGB or RGBA: grey is replicated, grey+alpha
    keeps its alpha, extra AOV channels beyond RGBA are dropped."""
    n = arr.shape[2]
    if n == 1:
        rgb = np.repeat(arr, 3, axis=2)
        a = None
    elif n == 2:
        rgb = np.repeat(arr[..., :1], 3, axis=2)
        a = arr[..., 1:2]
    else:
        rgb = arr[..., :3]
        a = arr[..., alpha_channel:alpha_channel + 1] if 0 <= alpha_channel < n else None
    if not want_alpha:
        return np.ascontiguousarray(rgb)
    if a is None:
        one = {np.uint8: 255, np.uint16: 65535}.get(arr.dtype.type, 1.0)
        a = np.full(rgb.shape[:2] + (1,), one, dtype=arr.dtype)
    return np.concatenate([rgb, a], axis=2)


def _read_image_oiio(path: str, fmt: OutputFmt, want_alpha: bool) -> np.ndarray:
    import OpenImageIO as oiio  # type: ignore[import-not-found]

    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise RuntimeError(f"cannot open image {path!r}: {oiio.geterror()}")
    try:
        spec = inp.spec()
        # OIIO converts float -> integer with clamping and rounding; HDR
        # values above 1.0 survive only in the float formats, as they should
        arr = inp.read_image(0, 0, 0, spec.nchannels, _OIIO_READ[fmt])
        if arr is None:
            raise RuntimeError(f"cannot read image {path!r}: {inp.geterror()}")
        arr = np.asarray(arr).reshape(spec.height, spec.width, spec.nchannels)
        return _to_rgb_channels(arr, spec.alpha_channel, want_alpha).astype(_NP_OUT[fmt], copy=False)
    finally:
        inp.close()


def _read_image_cv2(path: str, fmt: OutputFmt, want_alpha: bool) -> np.ndarray:
    """Fallback when OpenImageIO is not installed (PNG/JPEG/TIFF; not EXR)."""
    import cv2

    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"cannot read image {path!r} (install OpenImageIO for EXR/DPX)")
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[2] >= 3:  # BGR(A) -> RGB(A)
        arr = np.concatenate([arr[..., 2::-1], arr[..., 3:]], axis=2)
    arr = _to_rgb_channels(arr, 3 if arr.shape[2] >= 4 else -1, want_alpha)
    if arr.dtype == _NP_OUT[fmt]:
        return arr
    src_max = {np.uint8: 255.0, np.uint16: 65535.0}.get(arr.dtype.type, 1.0)
    f = arr.astype(np.float32) / src_max
    if fmt in ("float32", "float16"):
        return f.astype(_NP_OUT[fmt])
    top = 255.0 if fmt == "uint8" else 65535.0
    return np.clip(np.rint(f * top), 0, top).astype(_NP_OUT[fmt])


def _resize_image(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-average when shrinking, Lanczos when enlarging; keeps dtype and
    HDR values (float EXR stays float, above 1.0 included)."""
    h0, w0 = arr.shape[:2]
    if (w0, h0) == (width, height):
        return arr
    try:
        import cv2

        src = arr.astype(np.float32) if arr.dtype == np.float16 else arr
        interp = cv2.INTER_AREA if width * height < w0 * h0 else cv2.INTER_LANCZOS4
        out = cv2.resize(src, (width, height), interpolation=interp)
        if out.ndim == 2:
            out = out[..., None]
        return out.astype(arr.dtype, copy=False)
    except ImportError:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)[None]
        t = F.interpolate(t, size=(height, width), mode="area" if width < w0 else "bicubic",
                          **({} if width < w0 else {"align_corners": False}))
        out = t[0].permute(1, 2, 0).numpy()
        if arr.dtype in (np.uint8, np.uint16):
            top = 255 if arr.dtype == np.uint8 else 65535
            out = np.clip(np.rint(out), 0, top)
        return out.astype(arr.dtype)


def _image_reader():
    try:
        import OpenImageIO  # noqa: F401  # type: ignore[import-not-found]
        return _read_image_oiio
    except ImportError:
        return _read_image_cv2


# ── index-driven decode ──────────────────────────────────────────────────

class _TableWrong(Exception):
    """An arithmetic (constant-rate) index met a frame it cannot place."""


# Wanted frames this close together are decoded through in one pass: one
# decode is cheaper than a seek plus re-decoding up to a whole GOP.
_RUN_GAP = 48


def _frame_ts(frame) -> int | None:
    ts = frame.pts if frame.pts is not None else getattr(frame, "dts", None)
    return None if ts is None else int(ts)


def _seek_plan(vs, table: IndexTable, first: int) -> list[dict]:
    """Seeks to try, nearest first. A timestamp seek can land PAST its
    target: MP4 seeks on decode time, and MPEG-TS was measured landing on
    the NEXT keyframe every time, even for the first frame of the file. So
    each fallback starts further back - by whole keyframes when the table
    knows them - and the last two rewind to timestamp 0 and to byte 0."""
    plan: list[dict] = []
    offsets: set[int] = set()

    def add(offset: int) -> None:
        if offset not in offsets:
            offsets.add(offset)
            plan.append(dict(offset=offset, stream=vs, backward=True, any_frame=False))

    if table.keyframe_pts:
        k = bisect.bisect_right(table.keyframe_pts, table.pts_of(first)) - 1
        for back in (0, 1, 2, 4, 8):
            add(table.keyframe_pts[max(0, k - back)])
    else:
        for back in (0, 8, 32, 128, 512):
            add(table.pts_of(max(0, first - back)))
    add(0)
    plan.append(dict(offset=0, unsupported_byte_offset=True))
    return plan


def _decode_from(container, vs, table: IndexTable, first: int, last: int, final: bool):
    """Decode after a seek, yielding (index, frame) for first..last. Returns
    True when the seek landed past ``first`` (and nothing was yielded)."""
    seen = False
    prev = -1
    for frame in container.decode(vs):
        ts = _frame_ts(frame)
        if ts is None:
            continue
        idx = table.index_of_pts(ts)
        if idx is None:
            if ts < table.start_pts:
                continue  # pre-roll before the first shown frame
            if not table.is_exact:
                raise _TableWrong(ts)
            continue
        if not table.is_exact and seen and idx != prev + 1:
            # every timestamp on the grid but a slot left empty: a dropped
            # frame, so "index = time / step" no longer counts frames
            raise _TableWrong(ts)
        prev = idx
        if not seen:
            seen = True
            if idx > first:
                if not final:
                    return True
                if not table.is_exact:
                    raise _TableWrong(ts)  # even the file start is not where the table says
        if idx > last:
            return False
        if idx >= first:
            yield idx, frame
        if idx == last:
            return False
    if not seen and not final:
        return True  # landed at end of file: past every frame (TS, seeking the last GOP)
    if not table.is_exact:
        raise _TableWrong(None)  # end of file before ``last``
    return False


def _decode_span(container, vs, table: IndexTable, first: int, last: int) -> Iterator[tuple[int, object]]:
    """Yield (index, frame) for the frames first..last in presentation order.
    An arithmetic table that meets a timestamp off its grid, or runs out of
    file, raises _TableWrong; an exact table's undecodable frames simply do
    not appear (the callers hold the nearest good frame)."""
    plan = _seek_plan(vs, table, first)
    worked: list[dict] = []
    for i, seek in enumerate(plan):
        try:
            container.seek(**seek)
        except Exception:
            continue  # e.g. byte seeks: MP4 does not support them
        worked.append(seek)
        overshot = yield from _decode_from(container, vs, table, first, last, final=i == len(plan) - 1)
        if not overshot:
            return
    if not worked:
        raise RuntimeError("the file cannot be seeked")
    # the last resort could not seek at all: settle for the best seek that did
    container.seek(**worked[-1])
    yield from _decode_from(container, vs, table, first, last, final=True)


def _runs(indices: list[int], gap: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for i in indices:
        if runs and i - runs[-1][1] <= gap:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return runs


def _rebuilt(handle: C2CVideo, why) -> IndexTable:
    key = handle.source.stat_key
    if key not in _REBUILD_LOGGED:
        _REBUILD_LOGGED.add(key)
        logger.warning("%s: timestamps are not constant-rate (%s); indexing every frame exactly",
                       os.path.basename(handle.source.path), why)
    return rebuild_exact_index(handle.source.path, key, stream_index=handle.stream_index)


def _fill_missing(out: np.ndarray, rows_by_src: dict[int, list[int]], done: set[int], path: str) -> None:
    missing = sorted(set(rows_by_src) - done)
    if not missing:
        return
    if not done:
        raise RuntimeError(f"no requested frame of {os.path.basename(path)} could be decoded")
    have = sorted(done)
    logger.warning("%s: %d frame(s) could not be decoded; holding the nearest good frame",
                   os.path.basename(path), len(missing))
    for idx in missing:
        near = min(have, key=lambda d: (abs(d - idx), d))
        for r in rows_by_src[idx]:
            out[r] = out[rows_by_src[near][0]]


def _read_file_into(handle: C2CVideo, rows_by_src: dict[int, list[int]], out: np.ndarray, fmt: OutputFmt) -> None:
    assert isinstance(handle.source, FileSource)
    wanted = sorted(rows_by_src)
    table = get_index_table(handle.source.stat_key, stream_index=handle.stream_index)
    while True:
        done: set[int] = set()
        try:
            with _pooled(handle.source.path) as container:
                vs = container.streams.video[handle.stream_index]
                for first, last in _runs(wanted, _RUN_GAP):
                    for idx, frame in _decode_span(container, vs, table, first, last):
                        rows = rows_by_src.get(idx)
                        if rows and idx not in done:
                            _convert_into(frame, handle, fmt, out[rows[0]])
                            for r in rows[1:]:
                                out[r] = out[rows[0]]
                            done.add(idx)
            break
        except _TableWrong as exc:
            if table.is_exact:
                raise
            table = _rebuilt(handle, exc)  # and read again with the exact table
    _fill_missing(out, rows_by_src, done, handle.source.path)


def _out_shape(handle: C2CVideo) -> tuple[int, int, int]:
    return handle.height, handle.width, 4 if handle.has_alpha else 3


def _place(out: np.ndarray, rows: list[int], arr: np.ndarray, path: str) -> None:
    if arr.shape != out.shape[1:]:
        raise ValueError(f"{os.path.basename(path)} is {arr.shape[1]}x{arr.shape[0]}; "
                         f"the sequence is {out.shape[2]}x{out.shape[1]}")
    for r in rows:
        out[r] = arr


def read(handle: C2CVideo, indices: Sequence[int], *, fmt: OutputFmt = "float32") -> np.ndarray:
    """[N,H,W,C] for positions within the handle's selection (0-based). Any
    order, repeats allowed; every distinct frame is decoded once."""
    sel = handle.selected_indices()
    rows_by_src: dict[int, list[int]] = {}
    positions = list(indices)
    for row, pos in enumerate(positions):
        p = int(pos)
        if p < 0 or p >= len(sel):
            raise IndexError(f"selection index {p} out of range 0..{len(sel) - 1}")
        rows_by_src.setdefault(sel[p], []).append(row)
    out = np.empty((len(positions),) + _out_shape(handle), dtype=_NP_OUT[fmt])
    if not rows_by_src:
        return out
    if handle.source.kind == "sequence":
        assert isinstance(handle.source, SequenceSource)
        read_image = _image_reader()
        for src, rows in rows_by_src.items():
            path = handle.source.files[src]
            arr = read_image(path, fmt, handle.has_alpha)
            if arr.shape[:2] != (handle.height, handle.width):
                # a scaled handle, or a folder of mixed sizes (VHS fits them to the first)
                arr = _resize_image(arr, handle.width, handle.height)
            _place(out, rows, arr, path)
    else:
        _read_file_into(handle, rows_by_src, out, fmt)
    return out


def iter_chunks(handle: C2CVideo, chunk_frames: int, *, fmt: OutputFmt = "float32") -> Iterator[np.ndarray]:
    """Stream the selection in order as [<=chunk_frames,H,W,C] arrays. Peak
    memory is one chunk plus one frame, whatever the clip length."""
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    sel = handle.selected_indices()
    n = len(sel)
    if not n:
        return
    if handle.source.kind == "sequence":
        for s in range(0, n, chunk_frames):
            yield read(handle, range(s, min(n, s + chunk_frames)), fmt=fmt)
        return
    yield from _iter_file_chunks(handle, sel, chunk_frames, fmt)


class _ChunkSink:
    """Hands out the next [H,W,C] slot of the chunk being filled and returns
    the chunk once full. A chunk, once yielded, belongs to the consumer."""

    def __init__(self, total: int, size: int, shape: tuple[int, int, int], dtype):
        self.left, self.size, self.shape, self.dtype = total, size, shape, dtype
        self.buf: np.ndarray | None = None
        self.k = 0

    def slot(self) -> np.ndarray:
        if self.buf is None:
            self.buf = np.empty((min(self.size, self.left),) + self.shape, dtype=self.dtype)
        return self.buf[self.k]

    def commit(self) -> np.ndarray | None:
        self.k += 1
        self.left -= 1
        if self.k == len(self.buf):
            full, self.buf, self.k = self.buf, None, 0
            return full
        return None

    def rest(self) -> np.ndarray | None:
        return self.buf[: self.k] if self.buf is not None and self.k else None


def _iter_file_chunks(handle: C2CVideo, sel, chunk_frames: int, fmt: OutputFmt) -> Iterator[np.ndarray]:
    assert isinstance(handle.source, FileSource)
    n = len(sel)
    sink = _ChunkSink(n, chunk_frames, _out_shape(handle), _NP_OUT[fmt])
    table = get_index_table(handle.source.stat_key, stream_index=handle.stream_index)
    # a private container: the consumer may do anything between chunks,
    # including random reads of this same file through the pool
    container = _open_container(handle.source.path)
    try:
        vs = container.streams.video[handle.stream_index]
        pos = 0                                # next selection position to deliver
        last: tuple[int, int] | None = None    # (index, pts) of the last frame delivered
        prev: np.ndarray | None = None         # the last delivered picture, for holds
        held = 0
        while pos < n:
            try:
                for idx, frame in _decode_span(container, vs, table, sel[pos], sel[-1]):
                    if idx < sel[pos]:
                        continue  # between wanted frames (every-Nth selections)
                    while sel[pos] < idx and prev is not None:
                        # a wanted frame the decoder never produced: hold the previous one
                        sink.slot()[...] = prev
                        held, pos = held + 1, pos + 1
                        full = sink.commit()
                        if full is not None:
                            yield full
                    slot = sink.slot()
                    _convert_into(frame, handle, fmt, slot)
                    prev, first = slot, True
                    while pos < n and sel[pos] <= idx:  # this frame (and, at the very
                        if not first:                    # start, frames never decoded)
                            sink.slot()[...] = prev
                        held += sel[pos] < idx
                        first, pos = False, pos + 1
                        full = sink.commit()
                        if full is not None:
                            yield full
                    last = (idx, _frame_ts(frame))
                    if pos >= n:
                        break
                break
            except _TableWrong as exc:
                if table.is_exact:
                    raise
                exact = _rebuilt(handle, exc)
                if last is not None and exact.index_of_pts(last[1]) != last[0]:
                    raise RuntimeError(
                        f"{os.path.basename(handle.source.path)}: its timestamps are not what its "
                        f"frame rate declares, and frames already delivered were numbered by the "
                        f"declared rate. The exact index is now built: run again.") from exc
                table = exact
        while pos < n:  # the file ended early (exact table, undecodable tail)
            if prev is None:
                raise RuntimeError(f"no frame of {os.path.basename(handle.source.path)} could be decoded")
            sink.slot()[...] = prev
            held, pos = held + 1, pos + 1
            full = sink.commit()
            if full is not None:
                yield full
        if held:
            logger.warning("%s: %d frame(s) could not be decoded; held the previous good frame",
                           os.path.basename(handle.source.path), held)
        rest = sink.rest()
        if rest is not None:
            yield rest
    finally:
        _release(container)
