"""What Load Video carries to Save Video about the SOURCE (L9.55, owner 2026-10-25): its frame numbers (a sequence keeps
them: "handles trimmed, frames written as 1001-1100 again; In / Out keeps its offset"), its timecode, pixel aspect and
reel name ("Timecode (Recommended), ... Pixel aspect / reel name").

Load records them for the first LOADED source frame in video_info["c2c_source"]; Save maps each delivered frame back to
its source frame (after the handles Load added are trimmed, delivered frame k is selection index In + k) and writes the
numbers, the timecode of its first frame, the pixel aspect and the reel into the file. Where that mapping is not exact
- a retimed load, gaps in a sequence, more frames than the source selection - Save numbers from 1 as before and says so.
Camera vendor metadata is not carried yet (L9.64: needs real camera files).
"""

from __future__ import annotations

import os
import re
from fractions import Fraction
from typing import Any

INFO_KEY = "c2c_source"
_TC = re.compile(r"^(\d{1,2})[:;.](\d{2})[:;.](\d{2})([:;.])(\d{2})$")
_NUM = re.compile(r"(\d+)(?=\D*$)")


# ---- timecode -------------------------------------------------------------------------------------------------------

def tc_base(fps: float) -> int:
    """The timecode frame base: 24 for 23.976, 30 for 29.97, ..."""
    return max(1, int(round(float(fps))))


def parse_tc(tc: str | None) -> tuple[int, int, int, int, bool] | None:
    m = _TC.match(str(tc or "").strip())
    if not m:
        return None
    return int(m[1]), int(m[2]), int(m[3]), int(m[5]), m[4] == ";"


def tc_to_frames(tc: str, fps: float) -> int | None:
    p = parse_tc(tc)
    if p is None:
        return None
    hh, mm, ss, ff, drop = p
    base = tc_base(fps)
    n = ((hh * 60 + mm) * 60 + ss) * base + ff
    if drop and base in (30, 60):                       # SMPTE drop frame: 2 (or 4) numbers skipped each minute
        d = 2 if base == 30 else 4                      # except every tenth
        total_min = hh * 60 + mm
        n -= d * (total_min - total_min // 10)
    return n


def frames_to_tc(n: int, fps: float, drop: bool = False) -> str:
    base = tc_base(fps)
    n = max(0, int(n))
    if drop and base in (30, 60):
        d = 2 if base == 30 else 4
        per10 = base * 600 - d * 9                     # frames in ten minutes
        per1 = base * 60 - d                           # frames in a dropped minute
        tens, rem = divmod(n, per10)
        n += d * 9 * tens + (d * ((rem - d) // per1) if rem > d else 0)
    ff = n % base
    ss = (n // base) % 60
    mm = (n // (base * 60)) % 60
    hh = (n // (base * 3600)) % 24
    return f"{hh:02d}:{mm:02d}:{ss:02d}{';' if drop else ':'}{ff:02d}"


def tc_add(tc: str, frames: int, fps: float) -> str | None:
    start = tc_to_frames(tc, fps)
    p = parse_tc(tc)
    return None if start is None or p is None else frames_to_tc(start + int(frames), fps, p[4])


def pack_tc(tc: str) -> int | None:
    """SMPTE 12M / OpenEXR TimeCode packing (BCD), for OIIO's smpte:TimeCode (EXR, DPX)."""
    p = parse_tc(tc)
    if p is None:
        return None
    hh, mm, ss, ff, drop = p
    bcd = lambda v: ((v // 10) << 4) | (v % 10)                     # noqa: E731
    return (bcd(ff) & 0x3F) | (0x40 if drop else 0) | (bcd(ss) & 0x7F) << 8 | (bcd(mm) & 0x7F) << 16 | \
        (bcd(hh) & 0x3F) << 24


# ---- reading the source ---------------------------------------------------------------------------------------------

_CACHE: dict = {}


def _file_meta(path: str, stream_index: int) -> dict[str, Any]:
    import av
    try:
        st = os.stat(path)
        key = (path, st.st_size, st.st_mtime_ns, stream_index)
    except OSError:
        key = None
    if key in _CACHE:
        return dict(_CACHE[key])
    out: dict[str, Any] = {"timecode": None, "pixel_aspect": None, "reel": None}
    with av.open(path) as c:
        vs = c.streams.video[stream_index]
        data = [s for s in c.streams if s.type == "data"]
        metas = [dict(vs.metadata or {})] + [dict(s.metadata or {}) for s in data] + [dict(c.metadata or {})]
        for m in metas:
            if out["timecode"] is None and parse_tc(m.get("timecode")):
                out["timecode"] = str(m["timecode"])
            if out["reel"] is None and str(m.get("reel_name") or "").strip():
                out["reel"] = str(m["reel_name"]).strip()
        sar = getattr(vs, "sample_aspect_ratio", None) or getattr(vs.codec_context, "sample_aspect_ratio", None)
        if sar and Fraction(sar) not in (0, 1):
            out["pixel_aspect"] = [Fraction(sar).numerator, Fraction(sar).denominator]
    if key is not None:
        _CACHE[key] = dict(out)
        while len(_CACHE) > 128:
            _CACHE.pop(next(iter(_CACHE)))
    return out


def _sequence_meta(first_file: str) -> dict[str, Any]:
    out: dict[str, Any] = {"timecode": None, "pixel_aspect": None, "reel": None}
    try:
        import OpenImageIO as oiio  # type: ignore[import-not-found]
    except ImportError:
        return out
    inp = oiio.ImageInput.open(first_file)
    if inp is None:
        return out
    try:
        spec = inp.spec()
        tc = spec.get_string_attribute("smpte:TimeCode")
        if parse_tc(tc):
            out["timecode"] = tc
        par = float(spec.get_float_attribute("PixelAspectRatio", 1.0) or 1.0)
        if abs(par - 1.0) > 1e-6:
            f = Fraction(par).limit_denominator(1000)
            out["pixel_aspect"] = [f.numerator, f.denominator]
        reel = spec.get_string_attribute("reelName") or spec.get_string_attribute("dpx:InputDevice")
        if reel.strip():
            out["reel"] = reel.strip()
    finally:
        inp.close()
    return out


def frame_numbers(files: Any) -> list[int] | None:
    """The frame number in each sequence file name (the last group of digits), or None when one has none."""
    nums = []
    for f in files or ():
        m = _NUM.search(os.path.splitext(os.path.basename(str(f)))[0])
        if not m:
            return None
        nums.append(int(m[1]))
    return nums


def record(raw: Any, loaded: Any) -> dict[str, Any]:
    """video_info["c2c_source"] for the first LOADED source frame (before any handle Load adds)."""
    sel = list(loaded.selected_indices())
    retimed = any(getattr(t, "op", "") == "retime" for t in getattr(loaded, "transforms", ())) or \
        (getattr(loaded, "index_map", None) is not None and float(loaded.fps) != float(raw.fps))
    steps = {b - a for a, b in zip(sel, sel[1:])}
    step = steps.pop() if len(steps) == 1 else (1 if len(sel) < 2 else None)
    try:
        meta = (_file_meta(raw.source.path, int(raw.stream_index)) if raw.source.kind == "file"
                else _sequence_meta(raw.source.files[0]))
    except Exception:  # noqa: BLE001 - metadata is a bonus; a file PyAV cannot reopen just carries none
        meta = {"timecode": None, "pixel_aspect": None, "reel": None}
    first_index = sel[0] if sel else 0
    # L9.89: a black frame (-1, a missing-frame / edge policy) has no source position, so no exact mapping either
    rec: dict[str, Any] = {"first_index": int(first_index), "step": step, "count": len(sel),
                           "fps": float(raw.fps),
                           "exact": bool(sel) and step is not None and not retimed and min(sel) >= 0,
                           "first_number": None, "timecode": None, **{k: meta[k] for k in ("pixel_aspect", "reel")}}
    if raw.source.kind != "file" and rec["exact"]:
        own = tuple(getattr(raw.source, "numbers", ()) or ())
        if own and len(own) == len(raw.source.files):
            rec["first_number"] = int(own[int(first_index)])      # the file's own number, gaps or not (L9.89)
        else:
            nums = frame_numbers(raw.source.files)
            gaps = bool(getattr(raw.source, "gaps", ()))
            if nums is not None and not gaps and nums == list(range(nums[0], nums[0] + len(nums))):
                rec["first_number"] = nums[0] + int(first_index)
    if meta["timecode"] and rec["exact"]:
        rec["timecode"] = tc_add(meta["timecode"], int(first_index), float(raw.fps))
        rec["source_timecode"] = meta["timecode"]
    return rec


# ---- Save: the delivered frames back to the source ------------------------------------------------------------------

def delivered(rec: Any, offset: int, count: int, trim_exact: bool) -> dict[str, Any]:
    """What a Save writes for `count` frames starting at selection index `offset` (its In point, after the handles):
    {numbers: [..] | None, timecode, pixel_aspect, reel, note}."""
    out: dict[str, Any] = {"numbers": None, "timecode": None, "pixel_aspect": None, "reel": None, "note": None,
                           "step": 1, "fps": float((rec or {}).get("fps") or 24) if isinstance(rec, dict) else 24.0}
    if not isinstance(rec, dict):
        return out
    out["pixel_aspect"], out["reel"] = rec.get("pixel_aspect"), rec.get("reel")
    if not rec.get("exact"):
        out["note"] = "the Load was retimed or skipped unevenly: frames numbered from 1, no timecode"
        return out
    if not trim_exact:
        out["note"] = "the handles were not trimmed as Load added them: frames numbered from 1, no timecode"
        return out
    if offset + count > int(rec.get("count") or 0):
        out["note"] = (f"{count} frames from In {offset} but the Load gave {rec.get('count')}: frames numbered from 1, "
                       "no timecode")
        return out
    step = int(rec.get("step") or 1)
    out["step"] = step
    if rec.get("first_number") is not None:
        n0 = int(rec["first_number"]) + offset * step
        out["numbers"] = [n0 + k * step for k in range(count)]
    if rec.get("timecode"):
        out["timecode"] = tc_add(rec["timecode"], offset * step, float(rec.get("fps") or 24))
    return out
