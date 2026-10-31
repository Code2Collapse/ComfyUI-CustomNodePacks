"""Probe video files and image sequences into lazy C2CVideo handles."""

from __future__ import annotations

import bisect
import dataclasses
import logging
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

from .handle import (
    AudioDescriptor,
    ColourTags,
    C2CVideo,
    FileSource,
    SequenceSource,
    stat_key_for,
)

logger = logging.getLogger("c2c_video.probe")

_INDEX_CACHE: OrderedDict[tuple[str, int, int], IndexTable] = OrderedDict()
_MAX_INDEX_CACHE = 32

# Test hook: incremented on each full demux build.
_DEMUX_CALLS = 0

_SEQ_EXT = {".exr", ".hdr", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpx"}


def demux_call_count() -> int:
    return _DEMUX_CALLS


def reset_demux_call_count() -> None:
    global _DEMUX_CALLS
    _DEMUX_CALLS = 0


@dataclass(frozen=True)
class IndexTable:
    """Per-file frame index: exact (every packet timestamp, demuxed) or
    arithmetic (constant frame rate, confirmed by sampling - see
    _cfr_confirmed). Index i is the i-th frame in PRESENTATION order."""

    time_base: Fraction
    frame_count: int
    is_exact: bool
    start_pts: int
    pts_step: Fraction | None  # arithmetic: exact ticks per frame (29.97 in 1/1000 = 1001/30)
    _pts: np.ndarray | None  # exact: sorted presentation timestamps
    keyframe_pts: tuple[int, ...]
    stat_key: tuple[str, int, int]

    def pts_of(self, index: int) -> int:
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        if self._pts is not None:
            return int(self._pts[index])
        assert self.pts_step is not None
        # what av_rescale_q does when a muxer stamps frame i: round half up
        return self.start_pts + math.floor(index * self.pts_step + Fraction(1, 2))

    def index_of_pts(self, pts: int) -> int | None:
        """The frame index shown at ``pts``, or None when no frame is. The
        arithmetic grid tolerates one tick of rounding (muxers disagree on
        round-vs-truncate) whenever a frame spans 3 ticks or more."""
        if self._pts is not None:
            i = int(np.searchsorted(self._pts, pts))
            return i if i < self.frame_count and int(self._pts[i]) == pts else None
        if not self.pts_step:
            return None
        i = math.floor((pts - self.start_pts) / self.pts_step + Fraction(1, 2))
        if not 0 <= i < self.frame_count:
            return None
        tol = 1 if self.pts_step >= 3 else 0
        return i if abs(self.pts_of(i) - pts) <= tol else None

    def keyframe_at_or_before(self, pts: int) -> int | None:
        k = bisect.bisect_right(self.keyframe_pts, pts)
        return self.keyframe_pts[k - 1] if k else None

    def expected_pts_for_source_index(self, src_index: int) -> int:
        return self.pts_of(src_index)


def get_index_table(stat_key: tuple[str, int, int], *, stream_index: int = 0) -> IndexTable:
    """The file's index table. An evicted table is rebuilt the way probe_file
    built it - the arithmetic fast path when the file is CFR - never with a
    full demux by default (that reads every byte of a possibly 30 GB file)."""
    if stat_key in _INDEX_CACHE:
        _INDEX_CACHE.move_to_end(stat_key)
        return _INDEX_CACHE[stat_key]
    probe_file(stat_key[0], stream_index=stream_index, index="auto")
    return _INDEX_CACHE[stat_key]


def invalidate_index_table(stat_key: tuple[str, int, int]) -> None:
    _INDEX_CACHE.pop(stat_key, None)


def _cache_put(table: IndexTable) -> IndexTable:
    _INDEX_CACHE[table.stat_key] = table
    _INDEX_CACHE.move_to_end(table.stat_key)
    while len(_INDEX_CACHE) > _MAX_INDEX_CACHE:
        _INDEX_CACHE.popitem(last=False)
    return table


def _fraction_near(a: Fraction, b: Fraction, tol: float = 0.02) -> bool:
    if b == 0:
        return False
    return abs(float(a - b)) <= tol * abs(float(b))


def _stream_fps(vs) -> Fraction | None:
    rate = getattr(vs, "average_rate", None) or getattr(vs, "avg_frame_rate", None)
    if rate is None:
        return None
    try:
        if rate.denominator == 0:
            return None
        return Fraction(rate.numerator, rate.denominator)
    except Exception:
        return None


def _guess_cfr(vs, frame_count_hint: int, duration: float | None) -> tuple[bool, Fraction | None]:
    fps = _stream_fps(vs)
    r_rate = getattr(vs, "rate", None) or getattr(vs, "r_frame_rate", None)
    r_fps = None
    if r_rate is not None and getattr(r_rate, "denominator", 0):
        r_fps = Fraction(r_rate.numerator, r_rate.denominator)
    if fps is None:
        return False, None
    if r_fps is not None and not _fraction_near(fps, r_fps, tol=0.001):
        return False, fps
    frames_meta = int(getattr(vs, "frames", 0) or 0)
    if frames_meta <= 0 and (duration is None or duration <= 0):
        return False, fps
    expected = frame_count_hint
    if duration and fps:
        approx = round(float(duration * fps))
        if frames_meta > 0 and abs(frames_meta - approx) > 1 and abs(frames_meta - expected) > 1:
            return False, fps
        if abs(approx - expected) > 1 and frames_meta <= 0:
            return False, fps
    return True, fps


def _build_arithmetic_index_table(
    path: str,
    stat_key: tuple[str, int, int],
    *,
    time_base: Fraction,
    frame_count: int,
    start_pts: int,
    fps: Fraction,
) -> IndexTable:
    # ticks per frame, kept EXACT: rounding it (1000/29.97 -> 33) drifts a
    # whole frame every ~3 s of a millisecond-timebase clip
    step = Fraction(time_base.denominator, time_base.numerator) / fps
    return IndexTable(
        time_base=time_base,
        frame_count=frame_count,
        is_exact=False,
        start_pts=int(start_pts),
        pts_step=step,
        _pts=None,
        keyframe_pts=(),
        stat_key=stat_key,
    )


# Packets sampled at each end of the file to confirm constant frame rate.
# Small on purpose: a 4K ProRes packet is 10-20 MB.
_SAMPLE_PACKETS = 24
# A tail read that runs this long means the seek fell back to the start of
# the file; a full demux is then the honest (and equally expensive) answer.
_TAIL_PACKET_CAP = 4000


def _packet_pts(container, vs, limit: int) -> list[int] | None:
    out: list[int] = []
    for pkt in container.demux(vs):
        ts = pkt.pts if pkt.pts is not None else pkt.dts
        if ts is None:
            if pkt.size:
                return None  # a real packet with no timestamp: cannot confirm
            continue  # the empty flush packet at EOF
        out.append(int(ts))
        if len(out) >= limit:
            break
    return out


# Packets arrive in DECODE order: with B-frames the newest few presentation
# slots of a sample may still be missing (their frames come later), and a
# tail sample may start with open-GOP leading frames. Contiguity is judged
# away from those edges.
_REORDER_MARGIN = 8


def _all_on_grid(table: IndexTable, pts: list[int], *, contiguous: str) -> list[int] | None:
    """Indices of ``pts`` when every one lands on the grid, once each, with
    no empty slot (a dropped frame keeps timestamps on the grid but breaks
    "index = time / step"). ``contiguous`` is "head" or "tail": which end of
    the sorted sample must be gap-free."""
    idx = [table.index_of_pts(t) for t in pts]
    if not idx or any(i is None for i in idx) or len(set(idx)) != len(idx):
        return None
    s = sorted(idx)  # type: ignore[type-var]
    span = s[:max(1, len(s) - _REORDER_MARGIN)] if contiguous == "head" else s[min(len(s) - 1, _REORDER_MARGIN):]
    if span[-1] - span[0] != len(span) - 1 or (contiguous == "head" and s[0] != 0):
        return None
    return idx  # type: ignore[return-value]


def _cfr_confirmed(path: str, stream_index: int, table: IndexTable, claimed: int) -> int | None:
    """Is the file really constant-rate? Metadata alone cannot say: a phone
    or screen recording can declare 30 fps and stamp frames irregularly.
    Every packet timestamp in a sample from the head AND the tail must land
    on the arithmetic grid. Returns the true frame count (the tail's last
    index + 1, which also corrects an estimated or stale nb_frames), or None
    to demand an exact index. Irregularity between the samples is caught at
    read time: a timestamp the grid cannot place rebuilds the exact index."""
    import av

    try:
        container = av.open(path)
    except Exception:
        return None
    try:
        vs = container.streams.video[stream_index]
        head = _packet_pts(container, vs, _SAMPLE_PACKETS)
        if not head or _all_on_grid(table, head, contiguous="head") is None:
            return None
        back = max(0, min(claimed, table.frame_count) - _SAMPLE_PACKETS)
        container.seek(table.pts_of(back), stream=vs, backward=True, any_frame=False)
        tail = _packet_pts(container, vs, _TAIL_PACKET_CAP)
        if not tail or len(tail) >= _TAIL_PACKET_CAP:
            return None
        tail_idx = _all_on_grid(table, tail, contiguous="tail")
        if tail_idx is None:
            return None
        count = max(tail_idx) + 1
        # a container that COUNTS its samples (MP4/MOV stts) must agree; a
        # dropped frame between the samples shows up as a count mismatch
        declared = int(getattr(vs, "frames", 0) or 0)
        if declared and declared != count:
            return None
        return count
    except Exception as exc:
        logger.debug("CFR sample failed for %s: %s", path, exc)
        return None
    finally:
        container.close()


def _fps_candidates(vs, fps: Fraction) -> list[Fraction]:
    """Nominal rate first (r_frame_rate is the exact 30000/1001 where the
    average can be a rounded 2997/100), then the average."""
    out: list[Fraction] = []
    for rate in (getattr(vs, "base_rate", None), getattr(vs, "guessed_rate", None), fps):
        try:
            f = Fraction(rate.numerator, rate.denominator) if rate is not None else None
        except (AttributeError, ZeroDivisionError, TypeError):
            f = None
        if f and f > 0 and f not in out:
            out.append(f)
    return out


# ── exact index, persisted ───────────────────────────────────────────────
# A full demux reads every byte of the file once (I/O bound: ~1-2 s per GB
# from NVMe). The result - 8 bytes per frame - is kept on disk keyed by path,
# size and mtime, so a file is indexed once, not once per session.

_DISK_INDEX_VERSION = 1
_DISK_INDEX_MAX_FILES = 4000


def _index_cache_dir() -> str:
    env = os.environ.get("C2C_VIDEO_INDEX_DIR")
    if env:
        return env
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") \
        or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "c2c_video", "index")


def _disk_index_path(stat_key: tuple[str, int, int], stream_index: int) -> str:
    import hashlib

    raw = f"{stat_key[0]}|{stat_key[1]}|{stat_key[2]}|{stream_index}|v{_DISK_INDEX_VERSION}"
    return os.path.join(_index_cache_dir(), hashlib.sha1(raw.encode("utf-8")).hexdigest() + ".npz")


def _disk_index_load(stat_key: tuple[str, int, int], stream_index: int) -> IndexTable | None:
    path = _disk_index_path(stat_key, stream_index)
    try:
        with np.load(path, allow_pickle=False) as z:
            pts = z["pts"].astype(np.int64)
            keys = tuple(int(k) for k in z["keys"])
            tb = Fraction(int(z["tb"][0]), int(z["tb"][1]))
    except (OSError, KeyError, ValueError):
        return None
    return IndexTable(time_base=tb, frame_count=len(pts), is_exact=True,
                      start_pts=int(pts[0]) if len(pts) else 0, pts_step=None,
                      _pts=pts, keyframe_pts=keys, stat_key=stat_key)


def _disk_index_save(table: IndexTable, stream_index: int) -> None:
    path = _disk_index_path(table.stat_key, stream_index)
    try:
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        np.savez(tmp, pts=table._pts, keys=np.array(table.keyframe_pts, dtype=np.int64),
                 tb=np.array([table.time_base.numerator, table.time_base.denominator], dtype=np.int64))
        os.replace(tmp, path)  # atomic: a reader never sees half a file
        names = os.listdir(folder)
        if len(names) > _DISK_INDEX_MAX_FILES:  # keep the most recently written
            full = sorted((os.path.join(folder, n) for n in names), key=os.path.getmtime)
            for old in full[: len(full) - _DISK_INDEX_MAX_FILES]:
                try:
                    os.remove(old)
                except OSError:
                    pass
    except OSError as exc:  # a read-only or full disk only costs speed
        logger.debug("could not persist index for %s: %s", table.stat_key[0], exc)


def _build_exact_index_table(path: str, stat_key: tuple[str, int, int], *, stream_index: int = 0) -> IndexTable:
    cached = _disk_index_load(stat_key, stream_index)
    if cached is not None:
        return cached
    table = _demux_exact_index_table(path, stat_key, stream_index=stream_index)
    if table.frame_count:
        _disk_index_save(table, stream_index)
    return table


def _demux_exact_index_table(path: str, stat_key: tuple[str, int, int], *, stream_index: int = 0) -> IndexTable:
    global _DEMUX_CALLS
    _DEMUX_CALLS += 1
    import av

    container = av.open(path)
    try:
        vs = container.streams.video[stream_index]
        tb = Fraction(vs.time_base.numerator, vs.time_base.denominator)
        records: list[tuple[int, bool, int]] = []
        ordinal = 0
        for packet in container.demux(vs):
            if not packet.size:
                continue  # the empty flush packet at EOF
            if getattr(packet, "is_discard", False):
                continue  # edit-list pre-roll: decoded, never shown
            ts = packet.pts if packet.pts is not None else packet.dts
            if ts is None:
                continue  # no timestamp: the reader could never identify it either
            if ts < 0 and packet.pts is not None:
                continue  # before presentation start (negative-pts pre-roll)
            records.append((int(ts), bool(packet.is_keyframe), ordinal))
            ordinal += 1
        records.sort(key=lambda r: (r[0], r[2]))
        # one frame per timestamp: a duplicated pts cannot be told apart on decode
        pts_arr = np.unique(np.array([r[0] for r in records], dtype=np.int64))
        keyframes = tuple(sorted({r[0] for r in records if r[1]}))
        start_pts = int(pts_arr[0]) if len(pts_arr) else 0
        return IndexTable(
            time_base=tb,
            frame_count=len(pts_arr),
            is_exact=True,
            start_pts=start_pts,
            pts_step=None,
            _pts=pts_arr,
            keyframe_pts=keyframes,
            stat_key=stat_key,
        )
    finally:
        container.close()


def _probe_index_table(
    path: str,
    stat_key: tuple[str, int, int],
    *,
    stream_index: int,
    index: Literal["auto", "exact"],
    meta_frame_count: int,
    time_base: Fraction,
    start_pts: int,
    vs,
    duration: float,
) -> IndexTable:
    if index == "exact":
        return _build_exact_index_table(path, stat_key, stream_index=stream_index)
    # The arithmetic fast path needs a container that COUNTS its frames
    # (MP4/MOV stts, AVI header): that count is the check that catches a
    # dropped frame between the sampled ends. MKV/WebM/TS only estimate from
    # duration, so they are indexed exactly - once, then from the disk cache.
    if int(getattr(vs, "frames", 0) or 0) <= 0:
        return _build_exact_index_table(path, stat_key, stream_index=stream_index)
    is_cfr, fps = _guess_cfr(vs, meta_frame_count, duration if duration > 0 else None)
    if is_cfr and fps is not None:
        # map with headroom: an estimated count can be a few frames short
        slack = max(16, meta_frame_count // 50)
        for cand in _fps_candidates(vs, fps):
            table = _build_arithmetic_index_table(
                path,
                stat_key,
                time_base=time_base,
                frame_count=meta_frame_count + slack,
                start_pts=start_pts,
                fps=cand,
            )
            n = _cfr_confirmed(path, stream_index, table, meta_frame_count)
            if n:
                return dataclasses.replace(table, frame_count=n)
    return _build_exact_index_table(path, stat_key, stream_index=stream_index)


# FFmpeg colour enums (libavutil/pixfmt.h). PyAV 17 hands these back as ints;
# older PyAV returned names - both are accepted.
_MATRIX = {0: "rgb", 1: "bt709", 4: "fcc", 5: "bt601", 6: "bt601", 7: "smpte240m",
           9: "bt2020", 10: "bt2020"}
_PRIMARIES = {1: "bt709", 4: "bt470m", 5: "bt470bg", 6: "smpte170m", 7: "smpte240m",
              9: "bt2020", 11: "smpte431", 12: "smpte432"}
_TRANSFER = {1: "bt709", 4: "gamma22", 5: "gamma28", 6: "smpte170m", 7: "smpte240m",
             8: "linear", 13: "srgb", 14: "bt2020-10", 15: "bt2020-12", 16: "pq", 18: "hlg"}
_RANGE = {1: "tv", 2: "pc"}


def _tag(raw, table: dict) -> str | None:
    """Name of an FFmpeg colour tag, or None when unspecified/unknown."""
    if raw is None:
        return None
    try:
        return table.get(int(raw))
    except (TypeError, ValueError):
        low = str(raw).strip().lower()
        if not low or low in ("unknown", "unspecified", "reserved"):
            return None
        for name in table.values():
            if name in low or low in name:
                return name
        return None


def _normalize_matrix(raw, height: int, is_rgb: bool) -> str:
    """The YUV->RGB matrix to decode with. Untagged follows FFmpeg/players:
    BT.601 for SD, BT.709 for HD AND UHD - untagged 4K is almost always SDR
    BT.709; BT.2020 only when the file says so."""
    if is_rgb:
        return "rgb"
    tagged = _tag(raw, _MATRIX)
    if tagged and tagged != "rgb":
        return tagged
    return "bt709" if height >= 720 else "bt601"


def _normalize_range(raw, pix_fmt: str, is_rgb: bool) -> str:
    """tv (limited) or pc (full). Untagged: yuvj* and RGB are full, YUV is tv."""
    tagged = _tag(raw, _RANGE)
    if tagged:
        return tagged
    if is_rgb or (pix_fmt or "").lower().startswith("yuvj"):
        return "pc"
    return "tv"


def _pix_info(pix_fmt: str) -> tuple[int, bool, bool]:
    """(bit depth, has alpha, is rgb) from FFmpeg's own pixel-format
    descriptor - never from the name (yuv422p is 8-bit; gray has no alpha)."""
    try:
        from av.video.format import VideoFormat
        f = VideoFormat(pix_fmt)
        bits = max((c.bits for c in f.components), default=8)
        alpha = any(getattr(c, "is_alpha", False) for c in f.components)
        return int(bits), bool(alpha), bool(getattr(f, "is_rgb", False))
    except Exception:
        return 8, False, False


def _first_frame(container, vs):
    try:
        return next(container.decode(vs), None)
    except Exception:
        return None


def _colour_tags_raw(vs, frame) -> dict[str, object]:
    """Colour tags of the stream, completed from the first decoded frame.
    Some codecs carry them only in the bitstream (ProRes: the frame header),
    so the stream-level fields read "unspecified" until a frame is decoded -
    ffprobe hides this by decoding during probing. One frame costs ~ms."""
    cc = vs.codec_context
    raw: dict[str, object] = {k: getattr(cc, k, None) for k in
                              ("colorspace", "color_primaries", "color_trc", "color_range")}
    unspecified = {"colorspace": 2, "color_primaries": 2, "color_trc": 2, "color_range": 0}

    def missing(k: str) -> bool:
        v = raw[k]
        try:
            return v is None or int(v) == unspecified[k]
        except (TypeError, ValueError):
            return str(v).lower() in ("", "unknown", "unspecified")

    if frame is not None:
        for k in raw:
            if missing(k):
                raw[k] = getattr(frame, k, None)
    return raw


def _rotation_cw(frame) -> int:
    """Clockwise degrees (0/90/180/270) to turn decoded pixels upright.
    PyAV 17 exposes the display matrix only per frame: ``frame.rotation`` is
    its angle COUNTER-clockwise in [-180, 180] (av_display_rotation_get), and
    ffmpeg's own autorotate turns by the negation. Verified against ffmpeg
    autorotate for -display_rotation 90, -90 and 180."""
    try:
        ccw = int(round(float(getattr(frame, "rotation", 0) or 0)))
    except (TypeError, ValueError):
        return 0
    cw = (-ccw) % 360
    return cw if cw in (90, 180, 270) else 0


def probe_file(
    path: str,
    *,
    stream_index: int = 0,
    index: Literal["auto", "exact"] = "auto",
) -> C2CVideo:
    import av

    path = os.path.abspath(path)
    sk = stat_key_for(path)
    container = av.open(path)
    try:
        vs = container.streams.video[stream_index]
        tb = Fraction(vs.time_base.numerator, vs.time_base.denominator)
        duration = float(vs.duration * vs.time_base) if vs.duration else (
            float(container.duration / av.time_base) if container.duration else 0.0
        )
        meta_frames = int(getattr(vs, "frames", 0) or 0)
        if meta_frames <= 0 and duration > 0:
            fps_guess = _stream_fps(vs)
            if fps_guess:
                meta_frames = max(1, round(float(duration * fps_guess)))
        if meta_frames <= 0:
            # unknown count — must demux
            index = "exact"
            table = _build_exact_index_table(path, sk, stream_index=stream_index)
            meta_frames = table.frame_count
        else:
            start_pts = int(getattr(vs, "start_time", 0) or 0)
            table = _probe_index_table(
                path,
                sk,
                stream_index=stream_index,
                index=index,
                meta_frame_count=meta_frames,
                time_base=tb,
                start_pts=start_pts,
                vs=vs,
                duration=duration,
            )
        _cache_put(table)

        fps = _stream_fps(vs) or Fraction(24, 1)
        pix_fmt = str(vs.codec_context.pix_fmt or "unknown")
        bit_depth, has_alpha, is_rgb = _pix_info(pix_fmt)
        first = _first_frame(container, vs)
        rot = _rotation_cw(first) if first is not None else 0
        w, h = int(vs.codec_context.width), int(vs.codec_context.height)
        if rot in (90, 270):
            w, h = h, w
        tags = _colour_tags_raw(vs, first)
        matrix = _normalize_matrix(tags["colorspace"], h, is_rgb)
        colour = ColourTags(
            primaries=_tag(tags["color_primaries"], _PRIMARIES),
            transfer=_tag(tags["color_trc"], _TRANSFER),
            matrix=matrix,
            range=_normalize_range(tags["color_range"], pix_fmt, is_rgb),
        )
        audio_desc = None
        audios = [s for s in container.streams if s.type == "audio"]
        if audios:
            a = audios[0]
            adur = float(a.duration * a.time_base) if a.duration else duration
            audio_desc = AudioDescriptor(
                stream_index=a.index,
                sample_rate=int(a.rate or 48000),
                channels=int(a.channels or 2),
                duration=adur,
            )
        return C2CVideo(
            source=FileSource(path=path, stat_key=sk),
            stream_index=stream_index,
            width=w,
            height=h,
            fps=fps,
            frame_count=table.frame_count,
            time_base=tb,
            pix_fmt=pix_fmt,
            bit_depth=bit_depth,
            has_alpha=has_alpha,
            colour=colour,
            input_colorspace=None,
            audio=audio_desc,
            rotation=rot,
            selection=(0, table.frame_count, 1),
            transforms=(),
            index_mode="exact" if table.is_exact else "auto",
        )
    finally:
        container.close()


def _natural_key(name: str) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


# Frame tokens, every host's spelling: #### (Nuke), @@@@ (Shake), %04d (printf),
# $F4 (Houdini). The number is the padding; 0 means unpadded.
_TOKENS = (
    (re.compile(r"#+"), lambda m: len(m.group(0))),
    (re.compile(r"@+"), lambda m: len(m.group(0))),
    (re.compile(r"%0?(\d*)d"), lambda m: int(m.group(1) or 0)),
    (re.compile(r"\$F(\d*)"), lambda m: int(m.group(1) or 0)),
)
_FRAME_FILE = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)(?P<suffix>\.[A-Za-z0-9]+)$")


def _scan_frames(directory: str, prefix: str, suffix: str, pad: int,
                 lo: int | None = None, hi: int | None = None) -> tuple[str, ...]:
    """Every file in ``directory`` named prefix + frame number + suffix.

    ``pad`` > 0 means zero-padded to that width, which a larger number may
    exceed (frame 10000 in a #### sequence); 0 means any width. Frames are
    returned in NUMERIC order whatever the first frame is (1001, 86400...)."""
    try:
        names = os.listdir(directory or ".")
    except OSError:
        return ()
    rx = re.compile("^" + re.escape(prefix) + r"(\d+)" + re.escape(suffix) + "$")
    found: list[tuple[int, str]] = []
    for name in names:
        m = rx.match(name)
        if not m:
            continue
        digits = m.group(1)
        if pad and not (len(digits) == pad or (len(digits) > pad and not digits.startswith("0"))):
            continue
        n = int(digits)
        if (lo is not None and n < lo) or (hi is not None and n > hi):
            continue
        found.append((n, os.path.join(directory, name)))
    found.sort()
    return tuple(f for _, f in found)


def resolve_sequence(pattern_or_dir: str) -> tuple[str, ...]:
    """Files of an image sequence, in frame order. Accepts a directory, a
    pattern (####, @@@@, %04d, $F4, optionally with a [1001-1100] range), or
    ONE frame of a sequence - which resolves to the whole sequence, the way
    a compositor's Read does when you drop a single frame on it."""
    path = os.path.abspath(pattern_or_dir)
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if os.path.splitext(f)[1].lower() in _SEQ_EXT]
        files.sort(key=lambda p: _natural_key(os.path.basename(p)))
        return tuple(files)

    directory, name = os.path.split(path)
    lo = hi = None
    rng = re.search(r"\[(\d+)-(\d+)\]", name)
    if rng:
        lo, hi = int(rng.group(1)), int(rng.group(2))
        name = name[:rng.start()] + name[rng.end():]
    for rx, pad_of in _TOKENS:
        m = rx.search(name)
        if m:
            files = _scan_frames(directory, name[:m.start()], name[m.end():], pad_of(m), lo, hi)
            if files:
                return files
            raise FileNotFoundError(f"no frames match {pattern_or_dir!r}")
    if rng:   # "shot.[1001-1100].exr": the range stands where the number goes
        stem, ext = os.path.splitext(name)
        files = _scan_frames(directory, stem, ext, 0, lo, hi)
        if files:
            return files

    if os.path.isfile(path):
        m = _FRAME_FILE.match(name)
        if m:
            files = _scan_frames(directory, m.group("prefix"), m.group("suffix"),
                                 len(m.group("num")))
            if len(files) > 1:
                return files
        return (path,)
    raise FileNotFoundError(f"no sequence files resolved from {pattern_or_dir!r}")


def _detect_gaps(files: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    nums: list[int] = []
    for f in files:
        m = re.search(r"(\d+)(?=\.[^.]+$)", os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    gaps: list[tuple[int, int]] = []
    for a, b in zip(nums, nums[1:]):
        if b != a + 1:
            gaps.append((a + 1, b))
    return tuple(gaps)


def probe_sequence(pattern_or_dir: str) -> C2CVideo:
    import OpenImageIO as oiio  # type: ignore[import-not-found]

    files = resolve_sequence(pattern_or_dir)
    if not files:
        raise FileNotFoundError(f"empty sequence: {pattern_or_dir!r}")
    gaps = _detect_gaps(files)
    inp = oiio.ImageInput.open(files[0])
    if inp is None:
        raise RuntimeError(f"OpenImageIO could not open {files[0]!r}: {oiio.geterror()}")
    try:
        spec = inp.spec()
        w, h = int(spec.width), int(spec.height)
        fmt = str(spec.format).upper()
        nch = int(spec.nchannels)
        has_alpha = nch >= 4 or "A" in spec.channelnames
        if fmt in ("HALF", "FLOAT"):
            bit_depth = 16 if fmt == "HALF" else 32
            pix_fmt = "exr_half" if fmt == "HALF" else "exr_float"
        elif fmt == "UINT16":
            bit_depth, pix_fmt = 16, "sequence16"
        else:
            bit_depth, pix_fmt = 8, "sequence8"
    finally:
        inp.close()

    return C2CVideo(
        source=SequenceSource(pattern=pattern_or_dir, files=files, gaps=gaps),
        stream_index=0,
        width=w,
        height=h,
        fps=Fraction(24, 1),
        frame_count=len(files),
        time_base=Fraction(1, 24),
        pix_fmt=pix_fmt,
        bit_depth=bit_depth,
        has_alpha=has_alpha,
        # Float images (EXR, HDR) hold scene-linear light; integer ones (PNG,
        # JPEG, TIFF, DPX) are display-encoded, sRGB unless told otherwise.
        # Either way the pixels are RGB: there is no YUV matrix.
        colour=ColourTags(None, "linear" if pix_fmt.startswith("exr") or fmt in ("HALF", "FLOAT") else "srgb",
                          "rgb", "pc"),
        input_colorspace=None,
        audio=None,
        rotation=0,
        selection=(0, len(files), 1),
        transforms=(),
        index_mode="exact",
    )


def rebuild_exact_index(path: str, stat_key: tuple[str, int, int], *, stream_index: int = 0) -> IndexTable:
    """Force exact demux index rebuild (called when auto table fails verification)."""
    invalidate_index_table(stat_key)
    table = _build_exact_index_table(path, stat_key, stream_index=stream_index)
    _cache_put(table)
    logger.warning("rebuilt exact index table for %s after PTS mismatch", path)
    return table
