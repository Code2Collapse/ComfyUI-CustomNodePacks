"""Image-sequence grammar and the source timeline: frame numbers, padding, steps, missing frames, frame ranges.

Owner A21, Revision 3 Part 3.5 / 3.6 / 3.8 (ledger L9.89). Design: docs/evidence/A21/DESIGN_P0_unified_loader.md.

Two rules the old resolver broke (measured 2026-10-31):

* A frame number is a property of the FILE, never of its position. A sequence 1001-1100 with 1050 missing has 100
  frame slots; frame 1051 is not "the 50th file". Missing slots follow an explicit policy (``error`` by default).
* The padding class comes from all the siblings, not from the frame you happened to pick: in an unpadded sequence
  1..100, picking ``shot.10.exr`` must not drop frames 1-9.

Everything here is pure Python (``os`` + ``re``): no decoder is opened. The timeline is expressed as an index map over
the source's real frames, ``-1`` meaning a black frame, so movies and sequences share one mechanism.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from math import gcd

# A frame number is a WHOLE run of digits right before the extension, not directly after "v" / "V": a version tag
# (shot_v002.exr next to shot_v001.exr) is two versions, not a two-frame sequence. frame0001.png, 0001.png,
# name.0001.exr and render_v2_0001.exr stay sequences.
FRAME_FILE = re.compile(r"^(?P<prefix>.*?)(?<![vV0-9])(?P<num>\d+)(?P<suffix>\.[A-Za-z0-9]+)$")

MISSING_POLICIES = ("error", "black", "hold", "nearest", "skip")
EDGE_POLICIES = ("hold", "black", "loop", "bounce", "error")
# a timeline this sparse is not one sequence (and would allocate millions of slots)
MAX_MISSING = 100_000
DEFAULT_SCAN_LIMIT = 50_000


def parse_frame_name(name: str) -> tuple[str, str, str] | None:
    """(prefix, digits, suffix) of a numbered file name, or None."""
    m = FRAME_FILE.match(name)
    if not m:
        return None
    return m.group("prefix"), m.group("num"), m.group("suffix")


def _leading_zero(digits: str) -> bool:
    return len(digits) > 1 and digits[0] == "0"


def grid_step(numbers) -> int:
    """The greatest common divisor of the gaps between frame numbers (1 for fewer than two)."""
    step = 0
    for a, b in zip(numbers, numbers[1:]):
        step = gcd(step, int(b) - int(a))
    return max(1, step)


@dataclass(frozen=True)
class SequenceInfo:
    """One candidate sequence in a folder: frames of one prefix, extension and padding class."""

    directory: str
    prefix: str
    suffix: str                         # the extension with its dot, as spelled on disk (".exr")
    padding: int                        # digits of a zero-padded sequence; 0 = unpadded (natural numbers)
    numbers: tuple[int, ...]            # numeric order
    files: tuple[str, ...]              # same order as numbers
    step: int = 1
    missing: tuple[int, ...] = field(default_factory=tuple)
    sparse: bool = False                # more than MAX_MISSING gaps: not one sequence

    @property
    def ext(self) -> str:
        return self.suffix.lower()

    @property
    def first(self) -> int:
        return self.numbers[0]

    @property
    def last(self) -> int:
        return self.numbers[-1]

    @property
    def count(self) -> int:
        return len(self.numbers)

    @property
    def pattern(self) -> str:
        """Nuke-style: shot.####.exr (padded) or shot.#.exr (unpadded: any width, no leading zeros)."""
        return self.prefix + "#" * max(1, self.padding) + self.suffix

    @property
    def pattern_path(self) -> str:
        return os.path.join(self.directory, self.pattern)

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern, "directory": self.directory, "ext": self.ext, "padding": self.padding,
            "first": self.first, "last": self.last, "count": self.count, "step": self.step,
            "missing": list(self.missing[:200]), "missing_count": len(self.missing), "sparse": self.sparse,
            "first_file": os.path.basename(self.files[0]),
        }


def _make_info(directory: str, prefix: str, suffix: str, padding: int, entries: list[tuple[int, str]]) -> SequenceInfo:
    entries = sorted(entries)
    numbers = tuple(n for n, _ in entries)
    step = grid_step(numbers)
    have = set(numbers)
    span = (numbers[-1] - numbers[0]) // step + 1 if numbers else 0
    missing: tuple[int, ...] = ()
    sparse = span - len(numbers) > MAX_MISSING          # not listed one by one; build_timeline refuses it in words
    if 0 < span - len(numbers) <= MAX_MISSING:
        missing = tuple(n for n in range(numbers[0], numbers[-1] + 1, step) if n not in have)
    return SequenceInfo(directory=directory, prefix=prefix, suffix=suffix, padding=padding, numbers=numbers,
                        files=tuple(os.path.join(directory, f) for _, f in entries), step=step, missing=missing,
                        sparse=sparse)


def _classes(directory: str, prefix: str, suffix: str, items: list[tuple[str, str]]) -> list[SequenceInfo]:
    """Split one prefix + suffix group into padding classes (see the module notes)."""
    padded: dict[int, list[tuple[int, str]]] = {}
    natural: list[tuple[str, str]] = []
    for digits, name in items:
        if _leading_zero(digits):
            padded.setdefault(len(digits), []).append((int(digits), name))
        else:
            natural.append((digits, name))
    out: list[SequenceInfo] = []
    widths = sorted(padded)
    for w in widths:
        # frame 10000 of a #### sequence has no leading zero but belongs to it
        members = padded[w] + [(int(d), n) for d, n in natural if len(d) >= w]
        out.append(_make_info(directory, prefix, suffix, w, members))
    if natural and (not widths or any(len(d) < widths[0] for d, _ in natural)):
        # 1001..1100 has no leading zero to prove its padding; one shared width reads as that padding (shot.####.exr,
        # what a compositor shows), mixed widths as unpadded (u.#.png)
        lens = {len(d) for d, _ in natural}
        pad = lens.pop() if len(lens) == 1 and not widths else 0
        out.append(_make_info(directory, prefix, suffix, pad if pad > 1 else 0, [(int(d), n) for d, n in natural]))
    return out


def scan_directory(directory: str, *, exts=None, limit: int = DEFAULT_SCAN_LIMIT,
                   names: list[str] | None = None) -> tuple[list[SequenceInfo], list[str]]:
    """(candidate sequences, other files) of one folder. ``exts``: lower-case extensions with the dot to consider
    (None = any). Single numbered files are candidates too (count 1); callers decide what a one-frame "sequence" is.
    ``limit`` stops a pathological folder (the scan never walks into subfolders); ``names``: a listing the caller
    already has (the folder is then not listed again)."""
    if names is None:
        try:
            names = os.listdir(directory or ".")
        except OSError:
            return [], []
    if len(names) > limit:
        names = sorted(names)[:limit]
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    others: list[str] = []
    for name in names:
        ext = os.path.splitext(name)[1].lower()
        if exts is not None and ext not in exts:
            continue
        parsed = parse_frame_name(name)
        if parsed is None:
            others.append(name)
            continue
        prefix, digits, suffix = parsed
        groups.setdefault((prefix, suffix), []).append((digits, name))
    seqs: list[SequenceInfo] = []
    for (prefix, suffix), items in sorted(groups.items()):
        seqs.extend(_classes(directory, prefix, suffix, items))
    return seqs, sorted(others)


def detect_from_file(path: str, *, exts=None) -> tuple[SequenceInfo | None, list[SequenceInfo]]:
    """The sequence a picked frame belongs to, and every candidate in its folder that shares its prefix (other
    padding, other extension) - shown to the user, never merged. Only the file's own folder is read."""
    path = os.path.abspath(path)
    directory, name = os.path.split(path)
    parsed = parse_frame_name(name)
    if parsed is None:
        return None, []
    prefix, digits, suffix = parsed
    seqs, _ = scan_directory(directory, exts=exts)
    related = [s for s in seqs if s.prefix.lower() == prefix.lower()]
    here = [s for s in related if s.suffix == suffix and path in {os.path.abspath(f) for f in s.files}]
    if not here:
        return None, related
    # a natural number can sit in a padded class and the unpadded one: the class with a leading-zero member that
    # matches this file's width first, then the larger class
    want = len(digits) if _leading_zero(digits) else 0
    here.sort(key=lambda s: (s.padding != want, -s.count))
    return here[0], related


def numbers_of(files) -> tuple[int, ...]:
    """Frame numbers of a resolved file list when every file is prefix + number + extension of one prefix
    (else (): a folder of unrelated images is numbered by position)."""
    out: list[int] = []
    key = None
    for f in files:
        parsed = parse_frame_name(os.path.basename(f))
        if parsed is None:
            return ()
        k = (parsed[0], parsed[2].lower())
        if key is None:
            key = k
        elif k != key:
            return ()
        out.append(int(parsed[1]))
    if len(set(out)) != len(out):
        return ()
    return tuple(out)


# ── frame ranges ────────────────────────────────────────────────────────────

_RANGE = re.compile(r"^\s*(\d+)\s*(?:[-:]\s*(\d+)\s*)?(?:[xX]\s*(\d+)\s*)?$")


@dataclass(frozen=True)
class FrameRange:
    first: int | None = None
    last: int | None = None
    step: int | None = None             # None = the source's own grid step

    @property
    def is_all(self) -> bool:
        return self.first is None and self.last is None and self.step is None


def parse_frame_range(text) -> FrameRange:
    """Nuke's syntax: "" = all, "1050" = one frame, "1001-1100", "1001-1100x2" ("1001:1100" works too)."""
    s = str(text or "").strip()
    if not s:
        return FrameRange()
    if s.startswith("-"):
        raise ValueError(f"frame_range {s!r}: negative frame numbers are not supported")
    if re.fullmatch(r"\s*[xX]\s*\d+\s*", s):
        return FrameRange(step=max(1, int(re.sub(r"\D", "", s))))
    m = _RANGE.match(s)
    if not m:
        raise ValueError(f"frame_range {s!r}: write it as 1001-1100, 1001-1100x2 or one frame like 1050")
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) is not None else a
    step = int(m.group(3)) if m.group(3) is not None else None
    if b < a:
        raise ValueError(f"frame_range {s!r}: the first frame is after the last - use reverse to play backwards")
    if step is not None and step < 1:
        raise ValueError(f"frame_range {s!r}: the step must be 1 or more")
    return FrameRange(a, b, step)


# ── the timeline: frame numbers -> source indices (-1 = black) ──────────────

@dataclass(frozen=True)
class Timeline:
    indices: tuple[int, ...]            # source index shown at each output position; -1 = a black frame
    numbers: tuple[int, ...]            # the frame number each output position stands for
    step: int                           # the frame-number step between output positions
    missing: tuple[int, ...]            # numbers inside the source with no file, that the range asked for
    outside: tuple[int, ...]            # numbers the range asked for beyond the source's first..last
    notes: tuple[str, ...] = ()


def _fmt_numbers(nums, limit: int = 20) -> str:
    shown = ", ".join(str(n) for n in nums[:limit])
    return shown + (f" ... ({len(nums)} in all)" if len(nums) > limit else "")


def build_timeline(numbers, *, frame_range: FrameRange | str = "", missing: str = "error", edge: str = "hold",
                   name: str = "the source") -> Timeline:
    """Output positions for a source whose frames carry ``numbers`` (sorted ascending, unique).

    The range (default: first..last on the source's own grid step) asks for frame numbers. A number inside
    first..last with no frame follows ``missing``; one outside follows ``edge``. ``skip`` drops missing slots - the
    old behaviour, which shifts every later frame in time; only migrated Load Video Path nodes use it."""
    if missing not in MISSING_POLICIES:
        raise ValueError(f"missing_frames {missing!r}: one of {', '.join(MISSING_POLICIES)}")
    if edge not in EDGE_POLICIES:
        raise ValueError(f"edge_policy {edge!r}: one of {', '.join(EDGE_POLICIES)}")
    nums = [int(n) for n in numbers]
    if not nums:
        raise ValueError(f"{name} has no frames")
    fr = parse_frame_range(frame_range) if isinstance(frame_range, str) else frame_range
    src_step = grid_step(nums)
    step = fr.step or src_step
    first, last = nums[0], nums[-1]
    a = fr.first if fr.first is not None else first
    b = fr.last if fr.last is not None else last
    if (b - a) // step + 1 > MAX_MISSING + len(nums):
        raise ValueError(f"{name}: the range {a}-{b} asks for {(b - a) // step + 1} frames, far more than the "
                         f"{len(nums)} it has - check frame_range")
    pos_of = {n: i for i, n in enumerate(nums)}
    want = list(range(a, b + 1, step))
    span = (last - first) // src_step + 1          # grid slots of the source itself, for loop / bounce

    def grid_number(k: int) -> int:
        return first + k * src_step

    def edge_target(n: int) -> int:
        if edge in ("hold", "black", "error"):
            return n
        k = (n - first) // src_step if n >= first else -((first - n + src_step - 1) // src_step)
        if edge == "loop":
            return grid_number(k % span)
        period = max(1, 2 * span - 2)              # bounce: 0 1 2 1 0 1 2 ...
        r = k % period if span > 1 else 0
        return grid_number(r if r < span else period - r)

    indices: list[int] = []
    out_numbers: list[int] = []
    miss: list[int] = []
    outside: list[int] = []
    for n in want:
        target = n
        if n < first or n > last:
            outside.append(n)
            if edge == "error":
                continue
            if edge == "black":
                indices.append(-1)
                out_numbers.append(n)
                continue
            if edge == "hold":
                indices.append(0 if n < first else len(nums) - 1)
                out_numbers.append(n)
                continue
            target = edge_target(n)
        i = pos_of.get(target)
        if i is not None:
            indices.append(i)
            out_numbers.append(n)
            continue
        miss.append(target)
        if missing in ("error", "skip"):
            continue
        if missing == "black":
            indices.append(-1)
        elif missing == "hold":
            j = max(0, _bisect_left(nums, target) - 1)      # the previous frame; at the very start, the first one
            indices.append(j)
        else:                                               # nearest; a tie goes to the earlier frame
            j = _bisect_left(nums, target)
            if j >= len(nums):
                j = len(nums) - 1
            elif j > 0 and (target - nums[j - 1]) <= (nums[j] - target):
                j -= 1
            indices.append(j)
        out_numbers.append(n)
    if outside and edge == "error":
        raise ValueError(f"{name}: frame_range {a}-{b} reaches outside its frames {first}-{last} "
                         f"({_fmt_numbers(outside)}) - choose edge_policy hold / black / loop / bounce, or narrow "
                         f"the range")
    if miss and missing == "error":
        raise ValueError(f"{name} is missing {len(miss)} frame(s): {_fmt_numbers(sorted(set(miss)))} - choose "
                         f"missing_frames black / hold / nearest to load it anyway")
    if not indices:
        raise ValueError(f"{name}: frame_range {a}-{b} selects no frames (its frames are {first}-{last})")
    notes = []
    if miss:
        notes.append(f"{len(miss)} missing frame(s) {missing}: {_fmt_numbers(sorted(set(miss)))}")
    if outside:
        notes.append(f"{len(outside)} frame(s) outside {first}-{last}: {edge}")
    return Timeline(tuple(indices), tuple(out_numbers), step, tuple(sorted(set(miss))), tuple(outside), tuple(notes))


def _bisect_left(a: list[int], x: int) -> int:
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def info_from_files(files) -> SequenceInfo | None:
    """A SequenceInfo for an already-resolved file list (a pattern or a folder) when the files are one numbered
    sequence; None for a folder of unrelated names."""
    files = tuple(files)
    nums = numbers_of(files)
    if not files or not nums:
        return None
    parsed = parse_frame_name(os.path.basename(files[0]))
    assert parsed is not None
    digits = [parse_frame_name(os.path.basename(f))[1] for f in files]       # type: ignore[index]
    lens = {len(d) for d in digits}
    if any(_leading_zero(d) for d in digits):
        padding = min(len(d) for d in digits if _leading_zero(d))
    else:                                       # the folder scan's rule: one shared width reads as that padding
        padding = lens.pop() if len(lens) == 1 and len(digits[0]) > 1 else 0
    pairs = sorted(zip(nums, files))
    return _make_info(os.path.dirname(files[0]), parsed[0], parsed[2], padding,
                      [(n, os.path.basename(f)) for n, f in pairs])
