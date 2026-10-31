"""One entry point for every source the C2C loader takes: a movie, an image or EXR sequence, one still or one EXR.

Owner A21, Revision 3 Part 3.2 / 3.5 / 3.6 / 3.9 (ledger L9.89). ``probe_media`` decides what a path is (or obeys the
source mode), probes it lazily - one directory listing and one header, no pixels - and returns the handle with its
rate set and the facts the loader reports. ``apply_timeline`` turns a frame range and the missing-frame / edge
policies into the handle's selection (frame numbers in, source indices out; -1 = black).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import lru_cache

from .handle import C2CVideo
from .sequence import FrameRange, SequenceInfo, Timeline, build_timeline, detect_from_file, grid_step, \
    info_from_files, parse_frame_range

SOURCE_MODES = ("auto", "video file", "image sequence", "EXR sequence", "single image", "single EXR")
VIDEO_EXTENSIONS: tuple[str, ...] = (
    "mp4", "mov", "mkv", "webm", "avi", "m4v", "mxf", "gif", "ts", "mts",
    "m2ts", "wmv", "flv", "mpg", "mpeg", "3gp", "ogv", "y4m",
)
# Rev 3 Part 3.3: the still / sequence formats, offered only where a backend can read them (image_extensions)
STILL_EXTENSIONS_WANTED = (".exr", ".dpx", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".hdr", ".webp")
_CV2_READS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
_PATTERN = re.compile(r"(#+|@+|%0?\d*d|\$F\d*|\[\d+-\d+\])")
DEFAULT_FPS = Fraction(24, 1)


@lru_cache(maxsize=1)
def image_extensions() -> frozenset[str]:
    """Still / sequence extensions the installed backends actually read (never one that would fail on load)."""
    try:
        import OpenImageIO as oiio  # type: ignore[import-not-found]

        listed: set[str] = set()
        for entry in str(oiio.get_string_attribute("extension_list") or "").split(";"):
            _fmt, _, exts = entry.partition(":")
            listed.update("." + e.strip().lower() for e in exts.split(",") if e.strip())
        return frozenset(e for e in STILL_EXTENSIONS_WANTED if e in listed)
    except ImportError:
        return frozenset(e for e in STILL_EXTENSIONS_WANTED if e in _CV2_READS)


def is_pattern(path: str) -> bool:
    return os.path.isdir(path) or bool(_PATTERN.search(os.path.basename(path)))


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


@dataclass(frozen=True)
class MediaFacts:
    kind: str                                   # video | image_sequence | exr_sequence | still | exr_still
    mode: str
    path: str
    pattern: str = ""
    sequence: SequenceInfo | None = None
    related: tuple[SequenceInfo, ...] = ()      # other candidates in the folder sharing the prefix
    image: object | None = None                 # exr.ImageInfo of the first frame (stills / sequences)
    layer: str = ""
    layer_kind: str = ""
    fps: Fraction = DEFAULT_FPS
    fps_from: str = "file"                      # file | user | embedded | default
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_exr(self) -> bool:
        return self.kind in ("exr_sequence", "exr_still")

    def to_dict(self) -> dict:
        seq = self.sequence.to_dict() if self.sequence is not None else None
        return {
            "kind": self.kind, "mode": self.mode, "path": self.path, "pattern": self.pattern or None,
            "sequence": seq, "candidates": [c.to_dict() for c in self.related] if len(self.related) > 1 else [],
            "layer": self.layer or None, "layer_kind": self.layer_kind or None,
            "layers": [lay.to_dict() for lay in getattr(self.image, "layers", ())],
            "fps": f"{self.fps.numerator}/{self.fps.denominator}", "fps_from": self.fps_from,
            "warnings": list(self.warnings),
        }


def _seq_rate(h: C2CVideo, sequence_fps: float, embedded: Fraction | None) -> tuple[Fraction, str]:
    if float(sequence_fps or 0) > 0:
        return Fraction(sequence_fps).limit_denominator(1_000_000), "user"
    if embedded:
        return embedded, "embedded"
    return DEFAULT_FPS, "default"


def _with_rate(h: C2CVideo, fps: Fraction) -> C2CVideo:
    return replace(h, fps=fps, time_base=Fraction(fps.denominator, fps.numerator))


def probe_media(path: str, *, mode: str = "auto", layer: str = "auto", sequence_fps: float = 0.0,
                frame_hint: int | None = None) -> tuple[C2CVideo, MediaFacts]:
    """(lazy handle with its rate, facts). ``mode`` is one of SOURCE_MODES; ``layer`` an EXR layer key or "auto";
    ``sequence_fps`` 0 = the file's own rate (EXR FramesPerSecond / DPX frame rate) else 24; ``frame_hint`` picks the
    frame of a sequence a single-frame mode loads (default: its first)."""
    from .exr import inspect_image, pick_layer
    from .probe import probe_file, probe_sequence, resolve_sequence

    mode = mode if mode in SOURCE_MODES else "auto"
    path = os.path.abspath(str(path or "").strip().strip('"'))
    ext = _ext(path)
    pattern = is_pattern(path)
    images = image_extensions()
    warnings: list[str] = []

    if mode == "video file" or (mode == "auto" and not pattern and ext.lstrip(".") in VIDEO_EXTENSIONS):
        if pattern or ext in images:
            raise ValueError(f"source_mode 'video file': {os.path.basename(path)} is an image or a sequence - "
                             f"choose auto, image sequence or single image")
        h = probe_file(path)
        return h, MediaFacts(kind="video", mode=mode, path=path, fps=Fraction(h.fps), fps_from="file")
    if mode == "auto" and not pattern and ext not in images:
        if os.path.isfile(path):                  # an unknown extension: let the movie demuxer have a go
            h = probe_file(path)
            return h, MediaFacts(kind="video", mode=mode, path=path, fps=Fraction(h.fps), fps_from="file")
        raise FileNotFoundError(f"nothing found at {path}")

    want_exr = mode in ("EXR sequence", "single EXR")
    single = mode in ("single image", "single EXR")
    seq: SequenceInfo | None = None
    related: tuple[SequenceInfo, ...] = ()
    if pattern:
        files = resolve_sequence(path)
        seq = info_from_files(files)
        if seq is not None:
            related = (seq,)
    else:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"nothing found at {path}")
        seq, rel = detect_from_file(path, exts=images | {ext})
        related = tuple(rel)
        if single or seq is None or seq.count <= 1:
            files = (path,)
        else:
            files = seq.files
        if len([c for c in related if c.count > 1]) > 1:
            others = [c for c in related if c is not seq and c.count > 1]
            warnings.append("other sequences share this name: " + "; ".join(
                f"{c.pattern} ({c.first}-{c.last}, {c.count} frames)" for c in others[:4])
                + " - not merged; pick one to load it")
    if not files:
        raise FileNotFoundError(f"no frames found at {path}")
    exts = {_ext(f) for f in files}
    if len(exts) > 1:
        raise ValueError(f"{os.path.basename(path)}: the files mix {', '.join(sorted(exts))} - a sequence has one "
                         f"format; give a pattern with the extension (shot.####.exr)")
    if want_exr and exts != {".exr"}:
        raise ValueError(f"source_mode '{mode}': {os.path.basename(files[0])} is not an EXR")
    if mode in ("image sequence", "EXR sequence") and len(files) == 1 and not pattern:
        warnings.append(f"no other frames of {os.path.basename(files[0])} were found next to it: a one-frame "
                        f"sequence")
    if single and len(files) > 1:                 # a pattern / folder in a single-frame mode: one of its frames
        pick = files[0]
        if frame_hint is not None and seq is not None and int(frame_hint) in seq.numbers:
            pick = seq.files[seq.numbers.index(int(frame_hint))]
        files = (pick,)
    if not os.path.isfile(files[0]):
        raise FileNotFoundError(f"nothing found at {files[0]}")

    is_exr = exts == {".exr"}
    info = inspect_image(files[0])
    if info.deep:
        raise ValueError(f"{os.path.basename(files[0])} is a deep EXR - deep data is not loaded (flatten it first)")
    lay_arg = (layer or "auto") if is_exr else (None if str(layer or "auto").lower() in ("", "auto") else layer)
    lay = pick_layer(info, lay_arg) if lay_arg is not None else None
    still = single or (len(files) == 1 and not pattern and mode == "auto")
    h = probe_sequence(path if pattern else files[0], files=tuple(files), still=still, layer=lay_arg)
    fps, fps_from = _seq_rate(h, sequence_fps, info.fps)
    step = grid_step(source_numbers(h))
    if step > 1:
        warnings.append(f"rendered every {step} frames: plays at {fps / step} fps of a {fps} fps timebase "
                        f"(frame_range ...x1 holds the in-between frames at {fps} fps)")
    h = _with_rate(h, fps / step)
    kind = ("exr_still" if is_exr else "still") if still else ("exr_sequence" if is_exr else "image_sequence")
    pat = seq.pattern_path if (seq is not None and not still) else (path if pattern else "")
    return h, MediaFacts(kind=kind, mode=mode, path=path, pattern=pat, sequence=None if still else seq,
                         related=related, image=info, layer=lay.key if lay else "", layer_kind=lay.kind if lay else "",
                         fps=fps, fps_from=fps_from, warnings=tuple(warnings))


def source_numbers(h: C2CVideo) -> tuple[int, ...]:
    """The frame number of each source frame: a sequence's own numbers, else 0..n-1."""
    src = h.source
    nums = tuple(getattr(src, "numbers", ()) or ())
    if getattr(src, "kind", "") == "sequence" and nums and len(nums) == int(h.frame_count):   # by kind, not class
        return nums
    return tuple(range(int(h.frame_count)))


def source_times(h: C2CVideo) -> list[float] | None:
    """Seconds of each source frame for a sequence, from its frame NUMBERS (frame 1051 after a gap plays at 1051,
    not one frame after 1049); None for a movie (its timestamps live in the index table)."""
    if h.source.kind != "sequence":
        return None
    nums = source_numbers(h)
    rate = float(h.fps) * grid_step(nums) or 24.0          # frame numbers per second (the handle's fps is per frame)
    return [(n - nums[0]) / rate for n in nums]


def apply_timeline(h: C2CVideo, *, frame_range: str | FrameRange = "", missing: str = "error", edge: str = "hold",
                   name: str | None = None) -> tuple[C2CVideo, Timeline]:
    """The handle restricted / filled per the range and policies. When the result is every frame in order (no range,
    nothing missing) the handle comes back UNCHANGED - the old load, frame for frame.

    Rate: the handle's fps is per source frame. A sequence rendered on twos has fps = timebase / 2; a range that steps
    by 1 over it (``1001-1100x1``) holds or blacks the in-between numbers and plays at the full timebase."""
    nums = source_numbers(h)
    label = name or os.path.basename(str(getattr(h.source, "path", "") or getattr(h.source, "pattern", "") or "source"))
    fr = parse_frame_range(frame_range) if isinstance(frame_range, str) else frame_range
    tl = build_timeline(nums, frame_range=fr, missing=missing, edge=edge, name=label)
    if tl.indices == tuple(range(len(nums))):
        return h, tl
    src_step = grid_step(nums)
    fps = Fraction(h.fps) * src_step / tl.step if tl.step != src_step else None
    return h.with_timeline(tl.indices, fps=fps), tl
