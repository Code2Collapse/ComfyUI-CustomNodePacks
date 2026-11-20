"""L9.171 (owner A25, spec Part 5 items 71 / 72 / 74): what a loaded source IS beyond its pixels - every stream of a
movie (audio tracks, timecode track, colour tags), the sequences a folder holds before anything loads, the source's
timeline (time base, timecode, variable frame rate, audio details) and a frame integrity report (missing, duplicated,
irregular or unreadable frames).

Everything here reads headers and packet timestamps only - never a pixel - and never raises into a load: a report that
cannot be made says why in its own "error" field. No directory walk: one listing of one folder at most.
"""

from __future__ import annotations

import os
import statistics
from collections import Counter
from fractions import Fraction
from typing import Any

from . import source_meta as SMETA

try:                                   # the vendored shared core, as nodes_load imports it
    from ...c2c_core.video.sequence import grid_step, scan_directory
except (ImportError, ValueError):      # tests: the pack root is on sys.path
    from c2c_core.video.sequence import grid_step, scan_directory

_HEADER_CACHE: dict[tuple, dict] = {}
_PACKET_CACHE: dict[tuple, dict] = {}
_CACHE_MAX = 64
SEQUENCE_HEADER_LIMIT = 10_000          # frames whose header the integrity report opens (beyond: said in the report)


def _cache_put(cache: dict, key, value) -> None:
    cache[key] = value
    while len(cache) > _CACHE_MAX:
        cache.pop(next(iter(cache)))


def _stat_key(path: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.normcase(os.path.abspath(path)), int(st.st_size), int(st.st_mtime_ns))


def _secs(value, time_base) -> float | None:
    if value is None or time_base is None:
        return None
    return round(float(value * Fraction(time_base)), 6)


def _frac(f) -> str | None:
    if not f:
        return None
    f = Fraction(f)
    return f"{f.numerator}/{f.denominator}"


def _tc_info(tc: str | None, where: str | None) -> dict[str, Any] | None:
    p = SMETA.parse_tc(tc)
    if p is None:
        return None
    return {"start": str(tc), "from": where, "drop_frame": bool(p[4])}


# ---- 72.1 / 74.1: the streams of a movie ------------------------------------------------------------------------------

def movie_streams(path: str) -> dict[str, Any]:
    """Every stream of a movie from its headers: video, audio (codec, rate, channels, layout, language, title,
    default flag, start, duration), subtitle and data (a QuickTime timecode track) streams, the container, and the
    source timecode with where it was found (video stream tag, timecode track or container). Cached per file state."""
    key = _stat_key(path)
    if key is not None and key in _HEADER_CACHE:
        return _HEADER_CACHE[key]
    try:
        import av

        with av.open(path, metadata_errors="ignore") as c:
            out = _streams_of(c)
    except Exception as exc:  # noqa: BLE001 - a report, never a reason not to load
        return {"error": f"{type(exc).__name__}: {exc}"}
    if key is not None:
        _cache_put(_HEADER_CACHE, key, out)
    return out


def _disposition(st) -> dict[str, bool]:
    d = getattr(st, "disposition", None)
    try:
        val = int(d) if d is not None else 0
    except (TypeError, ValueError):
        val = 0
    return {"default": bool(val & 0x1), "forced": bool(val & 0x40)}     # AV_DISPOSITION_DEFAULT / _FORCED


def _streams_of(c) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"video": [], "audio": [], "subtitle": [], "data": [], "other": []}
    timecode, tc_from = None, None
    v_n = a_n = 0
    for st in c.streams:
        cc = getattr(st, "codec_context", None)
        tags = {str(k): str(v) for k, v in dict(st.metadata or {}).items()}
        tb = st.time_base
        row: dict[str, Any] = {
            "index": int(st.index), "type": st.type, "codec": getattr(cc, "name", None) if cc is not None else None,
            "profile": getattr(cc, "profile", None) if cc is not None else None,
            "language": tags.get("language"), "title": tags.get("title") or tags.get("handler_name"),
            "time_base": _frac(tb), "start_seconds": _secs(st.start_time, tb), "duration_seconds": _secs(st.duration, tb),
            "frames": int(st.frames or 0), **_disposition(st), "tags": tags,
        }
        if st.type == "video":
            row.update({"number": v_n, "width": int(getattr(cc, "width", 0) or 0),
                        "height": int(getattr(cc, "height", 0) or 0),
                        "pix_fmt": str(getattr(cc, "pix_fmt", "") or "") or None,
                        "average_fps": _frac(st.average_rate), "base_fps": _frac(getattr(st, "base_rate", None)),
                        "sample_aspect": _frac(getattr(st, "sample_aspect_ratio", None))})
            v_n += 1
            if timecode is None and SMETA.parse_tc(tags.get("timecode")):
                timecode, tc_from = tags["timecode"], f"video stream {st.index}"
        elif st.type == "audio":
            layout = getattr(st, "layout", None)
            fmt = getattr(cc, "format", None) if cc is not None else None
            row.update({"track": a_n + 1, "sample_rate": int(getattr(st, "rate", 0) or 0),
                        "channels": int(getattr(st, "channels", 0) or 0),
                        "layout": getattr(layout, "name", None), "sample_format": getattr(fmt, "name", None),
                        "bit_rate": int(getattr(st, "bit_rate", 0) or 0) or None})
            a_n += 1
        group = st.type if st.type in groups else "other"
        groups[group].append(row)
    if timecode is None:
        for row in groups["data"]:
            if SMETA.parse_tc(row["tags"].get("timecode")):
                timecode, tc_from = row["tags"]["timecode"], f"timecode track (stream {row['index']})"
                break
    ctags = {str(k): str(v) for k, v in dict(c.metadata or {}).items()}
    if timecode is None and SMETA.parse_tc(ctags.get("timecode")):
        timecode, tc_from = ctags["timecode"], "container"
    return {
        "container": {"format": c.format.name, "duration_seconds": round(c.duration / 1_000_000, 6) if c.duration
                      else None, "bit_rate": int(c.bit_rate or 0) or None, "tags": ctags},
        **groups,
        "counts": {k: len(v) for k, v in groups.items()},
        "timecode": _tc_info(timecode, tc_from),
    }


def colour_tags(h) -> dict[str, Any]:
    """The colour metadata the probe read (bitstream first, then container), as the loader uses it."""
    c = h.colour
    return {"primaries": c.primaries, "transfer": c.transfer, "matrix": c.matrix, "range": c.range,
            "pix_fmt": h.pix_fmt, "bit_depth": int(h.bit_depth), "has_alpha": bool(h.has_alpha),
            "input_colorspace": h.input_colorspace}


def _loaded_tc(rec: dict | None, count: int) -> dict[str, Any] | None:
    """Timecode of the first / last loaded source frame from video_info's c2c_source record (exact loads only)."""
    if not isinstance(rec, dict) or not rec.get("timecode"):
        return None
    step = int(rec.get("step") or 1)
    fps = float(rec.get("fps") or 24)
    last = SMETA.tc_add(rec["timecode"], max(0, int(count) - 1) * step, fps)
    return {"first": rec["timecode"], "last": last}


def loaded_timecode(rec: dict | None) -> dict[str, Any] | None:
    """First / last loaded frame's timecode from a c2c_source record, over the frames it was made for."""
    return _loaded_tc(rec, int(rec.get("count") or 0)) if isinstance(rec, dict) else None


def load_streams(raw, loaded, video_info: dict) -> dict[str, Any]:
    """The Load Video info's ``streams`` / ``timecode`` / ``colour_tags`` (spec 72.1): every stream of the source,
    which audio track feeds the AUDIO output, the source timecode and the timecode of the first / last loaded frame."""
    rec = video_info.get(SMETA.INFO_KEY) if isinstance(video_info, dict) else None
    if raw.source.kind == "file":
        streams = movie_streams(raw.source.path)
        tc = dict(streams.get("timecode") or {}) or None
    else:
        files = tuple(raw.source.files)
        streams = {"image_sequence": {"files": len(files), "first_file": os.path.basename(files[0]) if files else None,
                                      "ext": os.path.splitext(files[0])[1].lower() if files else None},
                   "audio": [], "counts": {"video": 0, "audio": 0, "subtitle": 0, "data": 0, "other": 0}}
        src_tc = rec.get("source_timecode") if isinstance(rec, dict) else None
        tc = _tc_info(src_tc, "first file header (smpte:TimeCode)")
    if tc is not None:
        lt = _loaded_tc(rec, int(rec.get("count") or len(loaded)) if isinstance(rec, dict) else 0)
        tc["loaded"] = lt if lt is not None else None
        if lt is None:
            tc["note"] = "the load is not an exact run of source frames (retimed / uneven): no loaded timecode"
    audio_used = None
    if getattr(raw, "audio", None) is not None and raw.source.kind == "file":
        audio_used = int(raw.audio.stream_index)
    out = {"streams": streams, "timecode": tc, "colour_tags": colour_tags(raw), "audio_stream_used": audio_used}
    if streams.get("audio") and len(streams["audio"]) > 1:
        out["audio_note"] = (f"{len(streams['audio'])} audio tracks: the audio output carries stream {audio_used} (the "
                       "first track); the others are listed here")
    return out


# ---- 71.1: sequence discovery ----------------------------------------------------------------------------------------

def _numbering(info) -> str:
    if info.count <= 1:
        return "single numbered file"
    if info.padding > 1:
        return f"zero-padded to {info.padding} digits ({info.prefix}{'#' * info.padding}{info.suffix})"
    return f"unpadded natural numbers ({info.prefix}#{info.suffix})"




def sequence_discovery(path: str, files_loaded=(), *, exts=None) -> dict[str, Any]:
    """What a folder (or the folder of a pattern / one frame) holds, from ONE listing and no pixel read (spec 71.1):
    every candidate sequence with its pattern, frame range, step, numbering convention (zero padding) and missing
    frames; unnumbered images; how many other files were ignored; and which of them the loader reads
    (``files_loaded``)."""
    path = os.path.abspath(str(path or ""))
    # a folder, or (no extension) a folder that is not there; else the folder of a pattern / one frame
    folder = path if os.path.isdir(path) or not os.path.splitext(path)[1] else os.path.dirname(path)
    try:
        names = os.listdir(folder)
    except OSError as exc:
        return {"directory": folder, "error": f"{type(exc).__name__}: {exc}"}
    wanted = {e.lower() for e in exts} if exts is not None else None
    files = [n for n in names if os.path.isfile(os.path.join(folder, n))]
    images = [n for n in files if wanted is None or os.path.splitext(n)[1].lower() in wanted]
    seqs, unnumbered = scan_directory(folder, exts=wanted, names=images)
    loaded = {os.path.normcase(os.path.abspath(f)) for f in files_loaded}
    rows: list[dict[str, Any]] = []
    for sq in seqs:
        member = {os.path.normcase(os.path.abspath(f)) for f in sq.files}
        hit = len(member & loaded)
        d = sq.to_dict()
        d.update({"range": f"{sq.first}-{sq.last}" + (f"x{sq.step}" if sq.step > 1 else ""),
                  "numbering": _numbering(sq),
                  "loads": "all" if loaded and hit == len(member) else ("part" if hit else "no"),
                  "loaded_frames": hit})
        rows.append(d)
    loose = [n for n in unnumbered if os.path.normcase(os.path.abspath(os.path.join(folder, n))) in loaded]
    feeding = [r for r in rows if r["loads"] != "no"]
    notes: list[str] = []
    if len(feeding) + len(loose) > 1:
        example = feeding[0]["pattern"] if feeding else "shot.####.exr"
        notes.append(f"{len(feeding)} sequence(s) and {len(loose)} unnumbered image(s) load one after another as ONE "
                     f"clip in name order (what this loader always did); type one pattern, e.g. {example}, in "
                     "directory to load just that one")
    for r in feeding:
        if r["missing_count"]:
            notes.append(f"{r['pattern']}: {r['missing_count']} frame number(s) missing inside {r['range']} - the "
                         "loaded frames close up over the gap")
    return {"directory": folder, "sequences": rows, "unnumbered_images": sorted(unnumbered)[:200],
            "unnumbered_count": len(unnumbered), "other_files": len(files) - len(images),
            "loaded_files": len(loaded), "mixed": len(feeding) + len(loose) > 1, "notes": notes}


# ---- 74.1 / 74.2: timeline and frame integrity ----------------------------------------------------------------------

def packet_scan(path: str, stream_index: int = 0) -> dict[str, Any]:
    """Every presentation timestamp of a movie's video stream, from its packets (demux only, nothing decoded), the
    way the loader's exact index reads them: discarded pre-roll, negative pre-roll and packets without a timestamp
    are counted, not used. Cached per file state."""
    sk = _stat_key(path)
    key = (sk, int(stream_index))
    if sk is not None and key in _PACKET_CACHE:
        return _PACKET_CACHE[key]
    import av

    pts: list[int] = []
    keyframes = 0
    no_ts = discard = preroll = dts_back = 0
    last_dts = None
    with av.open(path, metadata_errors="ignore") as c:
        vs = c.streams.video[int(stream_index)]
        tb = Fraction(vs.time_base.numerator, vs.time_base.denominator)
        fps = vs.average_rate or getattr(vs, "base_rate", None) or getattr(vs, "guessed_rate", None)
        declared = int(vs.frames or 0)
        for pkt in c.demux(vs):
            if not pkt.size:
                continue
            if getattr(pkt, "is_discard", False):
                discard += 1
                continue
            ts = pkt.pts if pkt.pts is not None else pkt.dts
            if ts is None:
                no_ts += 1
                continue
            if ts < 0 and pkt.pts is not None:
                preroll += 1
                continue
            if pkt.dts is not None:
                if last_dts is not None and pkt.dts < last_dts:
                    dts_back += 1
                last_dts = pkt.dts
            pts.append(int(ts))
            keyframes += bool(pkt.is_keyframe)
    out = {"time_base": tb, "fps": Fraction(fps) if fps else None, "declared_frames": declared, "pts": pts,
           "keyframes": keyframes, "no_timestamp": no_ts, "discarded": discard, "preroll": preroll,
           "dts_backwards": dts_back}
    if sk is not None:
        _cache_put(_PACKET_CACHE, key, out)
    return out


def analyse_pts(pts, time_base, fps=None, *, declared_frames: int = 0, examples: int = 50) -> dict[str, Any]:
    """Frame spacing of a list of presentation timestamps (ticks of ``time_base``). One frame = the most common step.
    Duplicated timestamps; gaps (a whole number of frames, 2 or more, within the tolerance: dropped frames on the
    grid); irregular steps (anything else more than 1 tick and 2 % off one frame); variable frame rate (3 or more
    irregular steps and at least 5 % of all). Frame indices are positions in sorted, de-duplicated order - the
    loader's index."""
    tb = Fraction(time_base)
    srt = sorted(int(p) for p in pts)
    uniq: list[int] = []
    dups: list[dict[str, Any]] = []
    for p in srt:
        if uniq and uniq[-1] == p:
            if dups and dups[-1]["pts"] == p:
                dups[-1]["copies"] += 1
            else:
                dups.append({"frame": len(uniq) - 1, "pts": p, "seconds": _secs(p - srt[0], tb), "copies": 2})
            continue
        uniq.append(p)
    deltas = [b - a for a, b in zip(uniq, uniq[1:])]
    # one frame = the most common step (a header's average rate is already lowered by dropped frames); the header
    # rate only when there is no step to count
    if deltas:
        step = float(Counter(deltas).most_common(1)[0][0])
    elif fps:
        step = float(1 / (Fraction(fps) * tb))
    else:
        step = 0.0
    tol = max(1.0, 0.02 * step)
    gaps: list[dict[str, Any]] = []
    irregular: list[dict[str, Any]] = []
    for i, d in enumerate(deltas):
        k = int(round(d / step)) if step > 0 else 0
        # a gap is a WHOLE number of frames missing (a dropped frame keeps the grid); any other step is irregular
        if step > 0 and k >= 2 and abs(d - k * step) <= tol:
            gaps.append({"after_frame": i, "seconds": _secs(uniq[i] - uniq[0], tb),
                         "missing_frames": k - 1, "gap_seconds": _secs(d, tb)})
        elif step > 0 and abs(d - step) > tol:
            irregular.append({"after_frame": i, "seconds": _secs(uniq[i] - uniq[0], tb),
                              "frame_seconds": _secs(d, tb)})
    vfr = len(irregular) >= max(3, int(0.05 * max(1, len(deltas))))     # one shifted frame (2 odd steps) is not VFR
    span = (uniq[-1] - uniq[0]) * tb if len(uniq) > 1 else Fraction(0)
    mismatch = bool(declared_frames) and int(declared_frames) != len(uniq)
    return {
        "frames": len(uniq), "timestamps": len(srt), "time_base": _frac(tb),
        "nominal_fps": round(float(fps), 6) if fps else None,
        "frame_step_fps": round(1 / (step * float(tb)), 6) if step > 0 else None,
        "average_fps": round((len(uniq) - 1) / float(span), 6) if span > 0 else None,
        "frame_seconds": ({"min": _secs(min(deltas), tb), "median": _secs(statistics.median(deltas), tb),
                           "max": _secs(max(deltas), tb)} if deltas else None),
        "variable_frame_rate": vfr,
        "duplicated": {"count": sum(d["copies"] - 1 for d in dups), "examples": dups[:examples]},
        "gaps": {"count": len(gaps), "missing_frames": sum(g["missing_frames"] for g in gaps),
                 "examples": gaps[:examples]},
        "irregular": {"count": len(irregular), "examples": irregular[:examples]},
        "declared_frames": int(declared_frames) or None, "count_mismatch": mismatch,
    }


def _selection_span(h) -> tuple[int, int] | None:
    sel = [int(i) for i in h.selected_indices() if int(i) >= 0]
    return (min(sel), max(sel)) if sel else None


def _issue_lists(rep: dict[str, Any]) -> list[str]:
    out = []
    if rep["duplicated"]["count"]:
        out.append(f"{rep['duplicated']['count']} duplicated timestamp(s) (a duplicate cannot be told apart on "
                   "decode: the loader keeps one)")
    if rep["gaps"]["count"]:
        out.append(f"{rep['gaps']['count']} gap(s), about {rep['gaps']['missing_frames']} missing frame(s)")
    if rep["irregular"]["count"]:
        out.append(f"{rep['irregular']['count']} irregular frame step(s)"
                   + (" - variable frame rate" if rep["variable_frame_rate"] else ""))
    if rep["count_mismatch"]:
        out.append(f"the header declares {rep['declared_frames']} frames, {rep['frames']} have timestamps")
    return out


def movie_integrity(h) -> dict[str, Any]:
    scan = packet_scan(h.source.path, int(h.stream_index))
    rep = analyse_pts(scan["pts"], scan["time_base"], scan["fps"], declared_frames=scan["declared_frames"],
                      examples=100_000)
    issues = _issue_lists(rep)
    for k, words in (("no_timestamp", "packet(s) without a timestamp (skipped by the loader)"),
                     ("dts_backwards", "packet(s) whose decode time runs backwards")):
        if scan[k]:
            issues.append(f"{scan[k]} {words}")
    span = _selection_span(h)
    if span is not None:
        lo, hi = span
        rep["in_loaded"] = {
            "first_frame": lo, "last_frame": hi,
            "gaps": sum(1 for g in rep["gaps"]["examples"] if lo <= g["after_frame"] < hi),
            "irregular": sum(1 for g in rep["irregular"]["examples"] if lo <= g["after_frame"] < hi),
            "duplicated": sum(1 for d in rep["duplicated"]["examples"] if lo <= d["frame"] <= hi),
        }
    for k in ("duplicated", "gaps", "irregular"):              # the counts above saw every case; keep 50 examples
        rep[k]["examples"] = rep[k]["examples"][:50]
    rep.update({"kind": "movie", "keyframes": scan["keyframes"], "preroll_skipped": scan["preroll"] + scan["discarded"],
                "no_timestamp": scan["no_timestamp"], "dts_backwards": scan["dts_backwards"],
                "issues": issues, "ok": not issues, "checked": True})
    return rep


def _header(path: str):
    """(width, height, channels) of an image file's header, a reason string when it cannot be read, or None when no
    header reader is installed."""
    try:
        import OpenImageIO as oiio  # type: ignore[import-not-found]
    except ImportError:
        return None
    inp = oiio.ImageInput.open(path)
    if inp is None:
        return f"unreadable: {oiio.geterror() or 'no reader'}"
    try:
        spec = inp.spec()
        return int(spec.width), int(spec.height), int(spec.nchannels)
    finally:
        inp.close()


def _size(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def sequence_integrity(files, numbers=(), *, header_limit: int = SEQUENCE_HEADER_LIMIT,
                       examples: int = 200) -> dict[str, Any]:
    """Missing and duplicated frame numbers, empty files, unreadable headers and frames whose size / channel count
    differ from the first (headers only; at most ``header_limit`` are opened)."""
    files = [str(f) for f in files]
    nums = [int(n) for n in numbers] if numbers and len(numbers) == len(files) else []
    issues: list[str] = []
    rep: dict[str, Any] = {"kind": "image sequence", "frames": len(files), "checked": True}
    if nums:
        seen: dict[int, int] = {}
        for n in nums:
            seen[n] = seen.get(n, 0) + 1
        dup = sorted(n for n, k in seen.items() if k > 1)
        uniq = sorted(seen)
        step = grid_step(uniq)
        span = (uniq[-1] - uniq[0]) // step + 1
        n_missing = span - len(uniq)
        missing = [n for n in range(uniq[0], uniq[-1] + 1, step) if n not in seen] if 0 < n_missing <= 100_000 else []
        rep.update({"first_number": uniq[0], "last_number": uniq[-1], "step": step,
                    "missing": {"count": n_missing, "numbers": missing[:examples]},
                    "duplicated_numbers": dup[:examples]})
        if n_missing:
            issues.append(f"{n_missing} missing frame number(s) inside {uniq[0]}-{uniq[-1]}")
        if dup:
            issues.append(f"{len(dup)} frame number(s) used by more than one file")
    else:
        rep["missing"] = None
        rep["note"] = "the files carry no common frame numbering: numbered by position, gaps cannot be seen"
    empty = [os.path.basename(f) for f in files if _size(f) == 0]
    if empty:
        issues.append(f"{len(empty)} empty file(s)")
    rep["empty_files"] = empty[:examples]
    bad: list[dict[str, Any]] = []
    first = None
    opened = 0
    have_reader = True
    for i, f in enumerate(files[:header_limit]):
        hd = _header(f)
        if hd is None:
            have_reader = False
            break
        opened += 1
        if isinstance(hd, str):
            bad.append({"frame": i, "file": os.path.basename(f), "problem": hd})
            continue
        if first is None:
            first = hd
        elif hd != first:
            bad.append({"frame": i, "file": os.path.basename(f),
                        "problem": f"{hd[0]}x{hd[1]} {hd[2]} channels, the first frame is {first[0]}x{first[1]} "
                                   f"{first[2]} channels"})
    if bad:
        issues.append(f"{len(bad)} frame(s) unreadable or of a different size / channel count")
    rep["unexpected_frames"] = bad[:examples]
    rep["headers_checked"] = opened
    if not have_reader:
        rep["note_headers"] = "OpenImageIO is not installed: headers not checked"
    elif len(files) > header_limit:
        rep["note_headers"] = f"headers of the first {header_limit} of {len(files)} frames checked"
    rep.update({"issues": issues, "ok": not issues})
    return rep


def integrity_report(h) -> dict[str, Any]:
    """Spec 74.2 for any C2C_VIDEO handle: a movie's packet timestamps, or a sequence's numbers and headers."""
    try:
        if h.source.kind == "file":
            return movie_integrity(h)
        return sequence_integrity(h.source.files, getattr(h.source, "numbers", ()) or ())
    except Exception as exc:  # noqa: BLE001 - a report, never a crash
        return {"checked": False, "ok": False, "issues": [f"could not check: {type(exc).__name__}: {exc}"]}


def _sequence_tc(first_file: str) -> str | None:
    try:
        import OpenImageIO as oiio  # type: ignore[import-not-found]
    except ImportError:
        return None
    inp = oiio.ImageInput.open(first_file)
    if inp is None:
        return None
    try:
        tc = inp.spec().get_string_attribute("smpte:TimeCode")
        return tc if SMETA.parse_tc(tc) else None
    finally:
        inp.close()


def _loaded_tc_from_selection(h, start_tc: str, fps: float) -> dict[str, Any] | None:
    """First / last loaded frame's timecode from the selection itself (no c2c_source record): an even, forward or
    backward run of real source frames that was not retimed."""
    sel = [int(i) for i in h.selected_indices()]
    if not sel or min(sel) < 0 or any(t.op == "retime" for t in h.transforms):
        return None
    steps = {b - a for a, b in zip(sel, sel[1:])}
    if len(steps) > 1:
        return None
    return {"first": SMETA.tc_add(start_tc, sel[0], fps), "last": SMETA.tc_add(start_tc, sel[-1], fps)}


def timeline_report(h, video_info: dict | None = None) -> dict[str, Any]:
    """Spec 74.1: the source's time base, start, nominal and measured rate, variable frame rate, timecode (start, end,
    drop frame, first / last loaded frame) and its audio streams; plus where the loaded frames sit in it."""
    vi = video_info if isinstance(video_info, dict) else {}
    rec = vi.get(SMETA.INFO_KEY)
    span = _selection_span(h)
    loaded_n = len([i for i in h.selected_indices() if int(i) >= 0])
    try:
        if h.source.kind == "file":
            streams = movie_streams(h.source.path)
            scan = packet_scan(h.source.path, int(h.stream_index))
            sp = analyse_pts(scan["pts"], scan["time_base"], scan["fps"], declared_frames=scan["declared_frames"])
            uniq = sorted(set(scan["pts"]))
            tb = scan["time_base"]
            fps = float(scan["fps"]) if scan["fps"] else float(h.fps)
            out: dict[str, Any] = {
                "kind": "movie", "container": (streams.get("container") or {}).get("format"),
                "video_stream": int(h.stream_index), "time_base": _frac(tb),
                "start_pts": uniq[0] if uniq else None, "start_seconds": _secs(uniq[0], tb) if uniq else None,
                "nominal_fps": sp["nominal_fps"], "average_fps": sp["average_fps"], "frames": sp["frames"],
                "duration_seconds": (round(float((uniq[-1] - uniq[0]) * tb) + (1 / fps if fps else 0), 6)
                                     if uniq else 0.0),
                "variable_frame_rate": sp["variable_frame_rate"], "frame_seconds": sp["frame_seconds"],
                "audio": streams.get("audio") or [],
                "audio_stream_used": int(h.audio.stream_index) if h.audio is not None else None,
            }
            tc = dict(streams.get("timecode") or {}) or None
            if span is not None and uniq and span[1] < len(uniq):
                out["loaded"] = {"first_frame": span[0], "last_frame": span[1],
                                 "start_seconds": _secs(uniq[span[0]] - uniq[0], tb),
                                 "end_seconds": _secs(uniq[span[1]] - uniq[0], tb), "frames": loaded_n,
                                 "fps": round(float(h.fps), 6)}
            n_frames = sp["frames"]
        else:
            files = tuple(h.source.files)
            nums = [int(n) for n in (getattr(h.source, "numbers", ()) or ())]
            fps = float(vi.get("source_fps") or h.fps)
            out = {"kind": "image sequence",
                   "time_base": _frac(1 / Fraction(fps).limit_denominator(1_000_000)) if fps else None,
                   "nominal_fps": round(fps, 6), "average_fps": round(fps, 6), "frames": len(files),
                   "duration_seconds": round(len(files) / fps, 6) if fps else None, "variable_frame_rate": False,
                   "first_number": nums[0] if nums else None, "last_number": nums[-1] if nums else None,
                   "audio": [], "audio_stream_used": None}
            src_tc = rec.get("source_timecode") if isinstance(rec, dict) else None
            src_tc = src_tc or (_sequence_tc(files[0]) if files else None)
            tc = _tc_info(src_tc, "first file header (smpte:TimeCode)")
            if span is not None:
                out["loaded"] = {"first_frame": span[0], "last_frame": span[1], "frames": loaded_n,
                                 "first_number": nums[span[0]] if len(nums) > span[1] else None,
                                 "last_number": nums[span[1]] if len(nums) > span[1] else None,
                                 "fps": round(float(h.fps), 6)}
            n_frames = len(files)
        if tc is not None:
            tc["end"] = SMETA.tc_add(tc["start"], max(0, n_frames - 1), fps)
            if isinstance(rec, dict) and rec.get("timecode"):
                tc["loaded"] = _loaded_tc(rec, int(rec.get("count") or loaded_n))
            else:
                tc["loaded"] = _loaded_tc_from_selection(h, tc["start"], fps)
        out["timecode"] = tc
        return out
    except Exception as exc:  # noqa: BLE001 - a report, never a crash
        return {"error": f"{type(exc).__name__}: {exc}"}
