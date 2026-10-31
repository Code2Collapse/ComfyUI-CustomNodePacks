"""Frame ledger - the one authoritative map between a source clip's frames and an output timeline.

Owner A20, Revision 2 Part 6.1 / 6.3 / 8.4 (ledger L9.74). Pure Python (fractions only): every mapping is exact. The
ledger is a plain JSON-safe dict, contract ``h3.frame_ledger`` v1 (the contract name comes from the first consumer,
the MiniMax H3 Suite; nothing here knows any model's frame grid - a model-specific planner hands in finished
segments).

Shape::

    {"c2c_contract": "h3.frame_ledger", "c2c_contract_version": "1.0",
     "source": {"path", "fps": "num/den", "time_base": "num/den" | None, "frames": N, "pts": [...] | None},
     "output": {"fps": "num/den", "frames": M, "offset": first global frame number},
     "map":    [[position_start, source_start, count, step], ...],     # step -1 = reversed run
     "segments": [{"id", "global": [a, b], "pad_head", "pad_tail", "overlap_head", "overlap_tail",
                   "context_frames", "accepted": [a, b] | None}, ...],
     "audio":  None | {"sample_rate", "offset_samples", "reverse"},
     "ops":    ["trim 10..50", "reverse", "retime 30000/1001 -> 25", ...]}

Positions are 0-based output indices; the global frame number is ``position + output.offset``. Padding, overlap,
context and the accepted range are separate segment fields (the six handle meanings of Part 6.3 never mix).
"""
from __future__ import annotations

import copy
from bisect import bisect_right
from fractions import Fraction

CONTRACT = "h3.frame_ledger"
VERSION = "1.0"


def _frac(x) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, str) and "/" in x:
        n, d = x.split("/", 1)
        return Fraction(int(n), int(d))
    return Fraction(x).limit_denominator(1_000_000)


def _fstr(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def compress(sources: list[int]) -> list[list[int]]:
    """Runs [position_start, source_start, count, step] covering positions 0..len-1, each with a constant step."""
    runs: list[list[int]] = []
    i, n = 0, len(sources)
    while i < n:
        if i + 1 == n:
            runs.append([i, sources[i], 1, 1])
            break
        step = sources[i + 1] - sources[i]               # 0 = a held frame, -1 = reversed, 2 = every other frame
        j = i + 1
        while j + 1 < n and sources[j + 1] - sources[j] == step:
            j += 1
        runs.append([i, sources[i], j - i + 1, step])
        i = j + 1
    return runs


def expand(ledger: dict) -> list[int]:
    """The source frame of every output position, in order."""
    out: list[int] = []
    for p0, s0, count, step in ledger["map"]:
        if p0 != len(out):
            raise ValueError(f"frame ledger: map runs are not contiguous at position {len(out)} (run starts {p0})")
        out.extend(s0 + k * step for k in range(count))
    return out


def new(source_frames: list[int] | range, *, fps, path: str | None = None, time_base=None, total_source: int | None = None,
        pts: list[int] | None = None, offset: int = 0, audio: dict | None = None) -> dict:
    """A ledger whose output position k shows source frame ``source_frames[k]``."""
    src = [int(s) for s in source_frames]
    return {
        "c2c_contract": CONTRACT, "c2c_contract_version": VERSION,
        "source": {"path": path, "fps": _fstr(_frac(fps)), "time_base": _fstr(_frac(time_base)) if time_base else None,
                   "frames": int(total_source if total_source is not None else (max(src) + 1 if src else 0)),
                   "pts": list(pts) if pts is not None else None},
        "output": {"fps": _fstr(_frac(fps)), "frames": len(src), "offset": int(offset)},
        "map": compress(src),
        "segments": [],
        "audio": dict(audio) if audio else None,
        "ops": [],
    }


def from_window(window: dict, *, total_source: int | None = None, audio: dict | None = None) -> dict:
    """A ledger for the rows of an ``h3.frame_window`` (H3 Frame Window Reader's window_json), handles included."""
    if window.get("c2c_contract") != "h3.frame_window":
        raise ValueError("frame ledger: window_json is not an h3.frame_window contract")
    return new(window["source_frames"], fps=window.get("fps") or 1, path=window.get("source"),
               time_base=window.get("time_base"), total_source=total_source, pts=None, audio=audio)


def _with(ledger: dict, sources: list[int], op: str, **output) -> dict:
    out = copy.deepcopy(ledger)
    out["map"] = compress(sources)
    out["output"]["frames"] = len(sources)
    out["output"].update(output)
    out["ops"].append(op)
    out["segments"] = []                                # segments describe a timeline; a new timeline drops them
    return out


def trim(ledger: dict, start: int, stop: int | None = None) -> dict:
    """Keep output positions start..stop (exclusive), clamped to the timeline."""
    src = expand(ledger)
    a = max(0, min(int(start), len(src)))
    b = len(src) if stop is None else max(a, min(int(stop), len(src)))
    return _with(ledger, src[a:b], f"trim {a}..{b}")


def offset(ledger: dict, first_global: int) -> dict:
    out = copy.deepcopy(ledger)
    out["output"]["offset"] = int(first_global)
    out["ops"].append(f"offset {int(first_global)}")
    return out


def reverse(ledger: dict) -> dict:
    """The output plays backwards. The audio flag flips so the sound follows the pictures."""
    out = _with(ledger, expand(ledger)[::-1], "reverse")
    if out.get("audio"):
        out["audio"]["reverse"] = not bool(out["audio"].get("reverse"))
    return out


def retime(ledger: dict, fps) -> dict:
    """Constant-rate conversion to ``fps`` by nearest frame (drops or repeats source frames); the duration is kept."""
    src = expand(ledger)
    old, new_fps = _frac(ledger["output"]["fps"]), _frac(fps)
    if new_fps == old or not src:
        return copy.deepcopy(ledger)
    count = max(1, int(round(len(src) * new_fps / old)))
    picked = [src[min(len(src) - 1, int(Fraction(j) * old / new_fps + Fraction(1, 2)))] for j in range(count)]
    return _with(ledger, picked, f"retime {_fstr(old)} -> {_fstr(new_fps)}", fps=_fstr(new_fps))


def source_of(ledger: dict, global_frame: int) -> int:
    """The source frame shown at a global frame number."""
    p = int(global_frame) - int(ledger["output"]["offset"])
    if not 0 <= p < int(ledger["output"]["frames"]):
        raise IndexError(f"global frame {global_frame} is outside the timeline "
                         f"{ledger['output']['offset']}..{ledger['output']['offset'] + ledger['output']['frames'] - 1}")
    runs = ledger["map"]
    i = bisect_right([r[0] for r in runs], p) - 1
    p0, s0, _count, step = runs[i]
    return s0 + (p - p0) * step


def global_of(ledger: dict, source_frame: int) -> list[int]:
    """Every global frame showing a source frame (several after a retime that repeats frames; none if dropped)."""
    s = int(source_frame)
    off = int(ledger["output"]["offset"])
    hits = []
    for p0, s0, count, step in ledger["map"]:
        if step == 0:
            if s == s0:
                hits.extend(off + p0 + k for k in range(count))
            continue
        k, rem = divmod(s - s0, step)
        if rem == 0 and 0 <= k < count:
            hits.append(off + p0 + k)
    return sorted(hits)


def add_segments(ledger: dict, segments: list[dict]) -> dict:
    """Attach a planner's segments (global frame numbers). Each is checked to lie inside the timeline."""
    out = copy.deepcopy(ledger)
    off, m = int(out["output"]["offset"]), int(out["output"]["frames"])
    clean = []
    for i, seg in enumerate(segments):
        a, b = (int(x) for x in seg["global"])
        if not (off <= a < b <= off + m):
            raise ValueError(f"frame ledger: segment {seg.get('id', i)} {a}..{b} lies outside the timeline {off}..{off + m}")
        acc = seg.get("accepted")
        if acc is not None and not (a <= int(acc[0]) <= int(acc[1]) <= b):
            raise ValueError(f"frame ledger: segment {seg.get('id', i)} accepted {acc} lies outside its range {a}..{b}")
        clean.append({"id": seg.get("id", i), "global": [a, b],
                      **{k: int(seg.get(k, 0)) for k in ("pad_head", "pad_tail", "overlap_head", "overlap_tail",
                                                          "context_frames")},
                      "accepted": [int(acc[0]), int(acc[1])] if acc is not None else None})
    out["segments"] = clean
    return out


def validate(ledger: dict) -> list[str]:
    """Problems as plain sentences (empty = consistent). Planners mid-run may legitimately have gaps: callers decide."""
    problems: list[str] = []
    if ledger.get("c2c_contract") != CONTRACT:
        return ["not an h3.frame_ledger contract"]
    try:
        src = expand(ledger)
    except ValueError as exc:
        return [str(exc)]
    if len(src) != int(ledger["output"]["frames"]):
        problems.append(f"the map covers {len(src)} frames but output.frames says {ledger['output']['frames']}")
    n_src = int(ledger["source"].get("frames") or 0)
    bad = [s for s in src if s < 0 or (n_src and s >= n_src)]
    if bad:
        problems.append(f"{len(bad)} output frames point outside the source (0..{n_src - 1}), e.g. {bad[:3]}")
    accepted = sorted(tuple(s["accepted"]) for s in ledger.get("segments") or [] if s.get("accepted"))
    for (a0, a1), (b0, b1) in zip(accepted, accepted[1:]):
        if b0 < a1:
            problems.append(f"accepted ranges overlap: {a0}..{a1} and {b0}..{b1}")
        elif b0 > a1:
            problems.append(f"gap between accepted ranges: {a1}..{b0} is accepted by no segment")
    return problems


def audio_span(ledger: dict, first_global: int, last_global: int) -> dict:
    """The source sample range that plays under global frames first..last (inclusive), from the SOURCE timeline."""
    aud = ledger.get("audio")
    if not aud:
        raise ValueError("frame ledger: no audio described")
    frames = [source_of(ledger, g) for g in range(int(first_global), int(last_global) + 1)]
    fps = _frac(ledger["source"]["fps"])
    sr = int(aud["sample_rate"])
    lo, hi = min(frames), max(frames) + 1
    start = int(aud.get("offset_samples", 0)) + int(Fraction(lo) / fps * sr)
    end = int(aud.get("offset_samples", 0)) + int(Fraction(hi) / fps * sr)
    return {"start": start, "end": end, "reverse": bool(aud.get("reverse"))}


def table(ledger: dict, head: int = 5, tail: int = 3) -> str:
    """A short mapping table: the first and last rows, for a node preview."""
    src = expand(ledger)
    off = int(ledger["output"]["offset"])
    rows = list(range(min(head, len(src)))) + [p for p in range(max(head, len(src) - tail), len(src))]
    lines = ["global  source"]
    last = -1
    for p in rows:
        if last >= 0 and p != last + 1:
            lines.append("  ...")
        lines.append(f"{off + p:6d}  {src[p]:6d}")
        last = p
    return "\n".join(lines)


__all__ = ["CONTRACT", "VERSION", "add_segments", "audio_span", "compress", "expand", "from_window", "global_of",
           "new", "offset", "retime", "reverse", "source_of", "table", "trim", "validate"]
