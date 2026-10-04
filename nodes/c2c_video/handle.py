"""C2C video lazy handles — no pixels, immutable value objects."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Literal


@dataclass(frozen=True)
class ColourTags:
    primaries: str | None
    transfer: str | None
    matrix: str | None
    range: str | None  # "tv" | "pc" | None


@dataclass(frozen=True)
class AudioDescriptor:
    stream_index: int
    sample_rate: int
    channels: int
    duration: float


@dataclass(frozen=True)
class TransformOp:
    op: Literal["trim", "step", "cap", "retime", "crop", "scale"]
    args: tuple[object, ...]


@dataclass(frozen=True)
class FileSource:
    kind: Literal["file"] = "file"
    path: str = ""
    stat_key: tuple[str, int, int] = field(default_factory=tuple)


@dataclass(frozen=True)
class SequenceSource:
    kind: Literal["sequence"] = "sequence"
    pattern: str = ""
    files: tuple[str, ...] = field(default_factory=tuple)
    gaps: tuple[tuple[int, int], ...] = field(default_factory=tuple)


def _normpath(path: str) -> str:
    import os

    return os.path.normpath(os.path.abspath(path))


def stat_key_for(path: str) -> tuple[str, int, int]:
    import os

    st = os.stat(path)
    return (_normpath(path), int(st.st_size), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))))


@dataclass(frozen=True)
class C2CVideo:
    source: FileSource | SequenceSource
    stream_index: int
    width: int
    height: int
    fps: Fraction
    frame_count: int
    time_base: Fraction
    pix_fmt: str
    bit_depth: int
    has_alpha: bool
    colour: ColourTags
    input_colorspace: str | None
    audio: AudioDescriptor | None
    rotation: int
    selection: tuple[int, int, int]
    transforms: tuple[TransformOp, ...]
    index_mode: Literal["auto", "exact"] = "auto"
    # Explicit source indices, when the selection is not a plain range (a
    # retime to another frame rate repeats or skips frames). Overrides
    # ``selection``.
    index_map: tuple[int, ...] | None = None

    # Every selection operation SLICES the current selection, so they compose
    # in any order: every(2) then trimmed(0, 10) is ten frames, not five.

    def selected_indices(self) -> Sequence[int]:
        if self.index_map is not None:
            return self.index_map
        return range(*self.selection)

    def _selected(self, idx: Sequence[int], op: TransformOp) -> C2CVideo:
        if isinstance(idx, range):
            return replace(self, selection=(idx.start, idx.stop, idx.step), index_map=None,
                           transforms=self.transforms + (op,))
        return replace(self, index_map=tuple(int(i) for i in idx), transforms=self.transforms + (op,))

    def trimmed(self, start: int, stop: int | None = None) -> C2CVideo:
        """Positions start..stop (exclusive) of the current selection."""
        start = max(0, int(start))
        stop = None if stop is None else max(start, int(stop))
        return self._selected(self.selected_indices()[start:stop], TransformOp("trim", (start, stop)))

    def every(self, step: int) -> C2CVideo:
        step = max(1, int(step))
        return self._selected(self.selected_indices()[::step], TransformOp("step", (step,)))

    def capped(self, count: int) -> C2CVideo:
        """At most ``count`` frames; 0 means no cap (VHS's frame_load_cap)."""
        count = int(count)
        if count <= 0:
            return self
        return self._selected(self.selected_indices()[:count], TransformOp("cap", (count,)))

    def retimed(self, fps: Fraction | float, source_times: Sequence[float]) -> C2CVideo:
        """Resample to ``fps``. Output frame j shows the first selected source
        frame whose time is at or after j / fps - VHS's force_rate rule - but
        judged on the file's REAL timestamps (``source_times``, seconds, one
        per source frame), so variable-frame-rate clips retime correctly.
        Slower than the source repeats frames; faster skips them."""
        import bisect

        fps = Fraction(fps).limit_denominator(1_000_000)
        if fps <= 0:
            return self
        sel = self.selected_indices()
        if not len(sel):
            return replace(self, fps=fps)
        times = [source_times[i] for i in sel]
        t0, t_end = times[0], times[-1]
        eps = 1e-6
        out: list[int] = []
        j = 0
        while True:
            t = t0 + float(j / fps)
            if t > t_end + eps:
                break
            k = bisect.bisect_left(times, t - eps)
            out.append(sel[min(k, len(sel) - 1)])
            j += 1
        h = self._selected(out, TransformOp("retime", (str(fps),)))
        return replace(h, fps=fps)

    def scaled(self, width: int, height: int) -> C2CVideo:
        """Frames arrive at width x height (upright), resampled while they are
        converted - never a full-size float frame first."""
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) == (self.width, self.height):
            return self
        return replace(self, width=width, height=height,
                       transforms=self.transforms + (TransformOp("scale", (width, height)),))

    @property
    def is_scaled(self) -> bool:
        return any(t.op == "scale" for t in self.transforms)

    def with_colorspace(self, cs: str | None) -> C2CVideo:
        return replace(self, input_colorspace=cs)

    def __len__(self) -> int:
        return len(self.selected_indices())

    def fingerprint(self) -> str:
        if self.source.kind == "file":
            src_key: object = self.source.stat_key
        else:
            src_key = self.source.files
        payload = {
            "source": src_key,
            "stream_index": self.stream_index,
            "selection": self.selection,
            "index_map": hashlib.sha256(repr(self.index_map).encode()).hexdigest() if self.index_map else None,
            "size": (self.width, self.height),
            "transforms": [(t.op, t.args) for t in self.transforms],
            "input_colorspace": self.input_colorspace,
            "index_mode": self.index_mode,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
