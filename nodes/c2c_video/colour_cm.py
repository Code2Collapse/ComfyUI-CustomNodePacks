"""Colour management for Load / Save Video (C2C) - L9.53 (owner 2026-10-25 / 26).

What colour space a file is in (Nuke / OCIO methods, the most reliable source first) and the transform Load applies
while decoding, recorded in ``video_info`` so Save can reverse exactly that. The owner's rules:

- the working default is sRGB display; a source that is already Rec.709 / sRGB stays as it is;
- scene-referred sources (log, linear, wide gamut) go through the config's tone-mapped VIEW (ACES 2.0 SDR in the
  built-in config - what Nuke's viewer shows), so the frames stay in 0-1 for the models;
- an EXR with no colour metadata is the config's scene_linear role (Nuke's working space and default Write);
- a movie / integer image with no colour tags is display (no conversion, and the record says so).

Detection order: sidecar > EXR ``acesImageContainer`` / ``colorInteropID`` (ASWF Color Interop Forum) > EXR
chromaticities > DPX transfer / image header > movie colour tags > the type default above. OCIO file rules are only a
suggestion. Camera vendor keys are not read yet (UNVERIFIED which keys PyAV exposes). Design:
docs/LOAD_SAVE_VIDEO_CM_DESIGN.md. Kill switch: C2C_VIDEO_CM=off.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import numpy as np

from . import save_ocio as SO

KILL_SWITCH = "C2C_VIDEO_CM"
CS_OFF = "off (as before)"          # the Python default: saved workflows and API prompts without the input keep it
CS_AUTO = "auto (detect)"
WORK_DEFAULT = "sRGB display (keep Rec.709 / sRGB)"
WORK_PRESERVE = "preserve source"
INFO_KEY = "c2c_colour"
SRGB_DISPLAY = "sRGB - Display"

# Rec.709 / sRGB encodings the owner keeps as they are ("if its already rec709 or srgb, leave it be")
_KEEP_IDS = frozenset({"srgb_rec709_display", "g24_rec709_display", "g22_rec709_display", "srgb_rec709_scene",
                       "g22_rec709_scene", "ocio:g24_rec709_scene", "ocio:itu709_rec709_scene"})

# movie colour tags (probe names) -> ASWF interop ID; partial / unspecified tags are handled in _from_tags
_TAG_IDS = {
    ("bt709", "bt709"): "g24_rec709_display", ("bt709", "smpte170m"): "g24_rec709_display",
    ("bt709", "srgb"): "srgb_rec709_display", ("bt709", "gamma22"): "g22_rec709_display",
    ("bt709", "linear"): "lin_rec709_scene",
    ("bt2020", "pq"): "pq_rec2020_display", ("bt2020", "hlg"): "hlg_rec2020_display",
    ("bt2020", "linear"): "lin_rec2020_scene",
    ("smpte432", "srgb"): "srgb_p3d65_display", ("smpte432", "pq"): "pq_p3d65_display",
    ("smpte432", "linear"): "lin_p3d65_scene",
}
_SD_PRIMARIES = ("smpte170m", "bt470bg", "bt470m")

# EXR chromaticities (Rx Ry Gx Gy Bx By Wx Wy) of linear spaces -> interop ID (OpenEXR: linear by definition)
_CHROMA_IDS = (
    ((0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767), "lin_ap0_scene"),
    ((0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767), "lin_ap1_scene"),
    ((0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290), "lin_rec709_scene"),
    ((0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290), "lin_p3d65_scene"),
    ((0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290), "lin_rec2020_scene"),
)

_FLOAT_EXT = (".exr", ".hdr")


def disabled() -> bool:
    return os.environ.get(KILL_SWITCH, "").strip().lower() in ("0", "off", "false", "no")


def config():
    """The active OCIO config (``$OCIO``, else the built-in studio config), or None without PyOpenColorIO."""
    return SO._CFG if SO._load_ocio() else None


def config_source() -> str:
    SO._load_ocio()
    return SO._CFG_SOURCE or "built-in"


def display_view(cfg, display: str) -> tuple[str, str | None]:
    """(view, note) for a display: the C2C setting's view when this config has it there, else the config's default."""
    want = SO._setting(SO.SETTING_VIEW, "")
    try:
        views = list(cfg.getViews(display))
    except Exception:  # noqa: BLE001
        views = []
    if want and want in views:
        return want, None
    default = cfg.getDefaultView(display)
    return default, (f"the view {want!r} (C2C setting) is not in this config for {display}: using {default!r}"
                     if want else None)


TARGET_OCIO = (2, 6)


def ocio_version_note() -> str | None:
    """L9.58: a one-line note on an OpenColorIO older than the version C2C targets (2.6) - it keeps working."""
    try:
        import PyOpenColorIO as ocio  # type: ignore[import-not-found]
        v = tuple(int(x) for x in str(ocio.__version__).split(".")[:2])
    except Exception:  # noqa: BLE001
        return None
    if v >= TARGET_OCIO:
        return None
    return (f"OpenColorIO {ocio.__version__} works; C2C targets 2.6 (built-in ACES 2.1 config) - "
            "pip install -U opencolorio")


def colorspace_in_choices() -> list[str]:
    return [CS_OFF, CS_AUTO] + SO.colorspace_choices()


def working_choices() -> list[str]:
    return [WORK_DEFAULT, WORK_PRESERVE] + SO.colorspace_choices()


def resolve(cfg, name_or_id: str | None) -> str | None:
    """A colour-space name, alias or interop ID -> the config's name. ASWF rule: the full ID, then without its
    leftmost namespace; the built-in config namespaces its own IDs ("ocio:arrilogc3_awg3_scene")."""
    if cfg is None or not name_or_id:
        return None
    s = str(name_or_id).strip()
    tries = [s]
    if ":" in s:
        tries.append(s.split(":", 1)[1])
    else:
        tries.append("ocio:" + s)
    for t in tries:
        cs = cfg.getColorSpace(t)
        if cs is not None:
            return cs.getName()
    return None


def interop_id(cfg, name: str | None) -> str | None:
    """The space's ASWF Color Interop ID (OCIO 2.5+), or None - never a guessed one."""
    if cfg is None or not name:
        return None
    cs = cfg.getColorSpace(name)
    try:
        return (cs.getInteropID() or None) if cs is not None else None
    except AttributeError:                       # OCIO < 2.5
        return None


def is_display_referred(cfg, name: str) -> bool:
    import PyOpenColorIO as ocio  # type: ignore[import-not-found]
    cs = cfg.getColorSpace(name)
    return cs is not None and cs.getReferenceSpaceType() == ocio.REFERENCE_SPACE_DISPLAY


def is_data(cfg, name: str | None) -> bool:
    cs = cfg.getColorSpace(name) if (cfg is not None and name) else None
    return bool(cs is not None and cs.isData())


def keeps(cfg, name: str | None) -> bool:
    """Already Rec.709 / sRGB: the owner's working default leaves it as it is."""
    return (interop_id(cfg, name) or "") in _KEEP_IDS


# ---- detection ---------------------------------------------------------------------------------------------------

@dataclass
class Detection:
    space: str | None
    source: str                      # what said so, in words
    confidence: str | None           # "high" | "medium" | "assumed" | "manual" | None (unknown)
    notes: list[str] = field(default_factory=list)
    suggestion: str | None = None    # OCIO file rule: shown, never applied


def _first_path(raw) -> str:
    return raw.source.path if raw.source.kind == "file" else raw.source.files[0]


def sidecar_paths(first: str, is_file: bool) -> list[str]:
    """`<file>.colorspace` (its first line is the space) or `<file>.c2c.json` ({"colorspace": ...}); a sequence also
    `<stem>.colorspace` / `<stem>.c2c.json` next to its frames (shot.1001.exr -> shot.colorspace)."""
    out = [first + ".colorspace", first + ".c2c.json"]
    if not is_file:
        d, name = os.path.split(first)
        stem = re.sub(r"[._]?\d+$", "", os.path.splitext(name)[0])
        if stem:
            out += [os.path.join(d, stem + ".colorspace"), os.path.join(d, stem + ".c2c.json")]
    return out


def _sidecar_paths(raw) -> list[str]:
    return sidecar_paths(_first_path(raw), raw.source.kind == "file")


def sidecar_token(first: str, is_file: bool) -> str:
    """Part of Load's IS_CHANGED: a sidecar added / edited next to the file re-runs the node."""
    paths = sidecar_paths(first, is_file)
    parts = []
    for p in paths:
        try:
            st = os.stat(p)
            parts.append(f"{os.path.basename(p)}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            continue
    return "cm:" + ",".join(parts)


def _from_sidecar(raw, cfg) -> Detection | None:
    for p in _sidecar_paths(raw):
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
            name = json.loads(text).get("colorspace") if p.endswith(".json") else text.strip().splitlines()[0]
        except (OSError, ValueError, AttributeError, IndexError):
            return Detection(None, f"sidecar {os.path.basename(p)}", None, [f"{os.path.basename(p)} is unreadable"])
        found = resolve(cfg, name)
        if found:
            return Detection(found, f"sidecar {os.path.basename(p)}", "high")
        return Detection(None, f"sidecar {os.path.basename(p)}", None,
                         [f"{os.path.basename(p)} names {name!r}, which this OCIO config does not have"])
    return None


def _header_attrs(path: str) -> dict[str, Any]:
    try:
        import OpenImageIO as oiio  # type: ignore[import-not-found]
    except ImportError:
        return {}
    inp = oiio.ImageInput.open(path)
    if inp is None:
        return {}
    try:
        spec = inp.spec()
        return {a.name: a.value for a in spec.extra_attribs}
    finally:
        inp.close()


def _match_chromaticities(vals) -> str | None:
    try:
        v = [float(x) for x in vals]
    except (TypeError, ValueError):
        return None
    if len(v) != 8:
        return None
    for ref, iid in _CHROMA_IDS:
        if max(abs(a - b) for a, b in zip(v, ref)) <= 0.002:
            return iid
    return None


def _from_exr(path: str, cfg) -> Detection:
    attrs = _header_attrs(path)
    if attrs.get("acesImageContainer") is not None:
        return Detection(resolve(cfg, "lin_ap0_scene"), "EXR acesImageContainer (ACES2065-1)", "high")
    iid = attrs.get("colorInteropID")
    if iid is not None and str(iid).strip().lower() not in ("", "unknown"):
        if str(iid).strip().lower() == "data":
            return Detection(resolve(cfg, "data") or "Raw", "EXR colorInteropID 'data'", "high")
        found = resolve(cfg, iid)
        if found:
            return Detection(found, f"EXR colorInteropID {iid!r}", "high")
        notes = [f"the EXR says colorInteropID {iid!r}, which this OCIO config does not have"]
    else:
        notes = []
    chroma = attrs.get("chromaticities")
    if chroma is not None:
        cid = _match_chromaticities(chroma)
        if cid and resolve(cfg, cid):
            return Detection(resolve(cfg, cid), f"EXR chromaticities ({cid})", "medium", notes)
        notes.append("the EXR chromaticities match no linear space this config knows")
    role = cfg.getRoleColorSpace("scene_linear") if cfg is not None else None
    if role:
        return Detection(role, "no colour metadata: the config's scene_linear role (Nuke's working space)",
                         "assumed", notes)
    return Detection(None, "no colour metadata, and the config has no scene_linear role", None, notes)


_DPX_DISPLAY = ("itu-r 709-4", "smpte 274m", "unspecified video")


def _from_image_header(path: str, ext: str, cfg) -> Detection:
    attrs = _header_attrs(path)
    if ext == ".dpx":
        tr = str(attrs.get("dpx:Transfer") or "").strip()
        low = tr.lower()
        if low in ("logarithmic", "printing density"):
            cineon = [n for n in SO.colorspace_choices() if "cineon" in n.lower()]
            if cineon:
                return Detection(cineon[0], f"DPX transfer {tr!r}", "medium")
            return Detection(None, f"DPX transfer {tr!r}", None,
                             [f"the DPX says {tr!r} (Cineon log) and this OCIO config has no Cineon space"])
        if low == "linear":
            return Detection(resolve(cfg, "lin_rec709_scene"), "DPX transfer 'Linear'", "medium")
        if low in _DPX_DISPLAY:
            return Detection(resolve(cfg, "g24_rec709_display"), f"DPX transfer {tr!r}", "medium")
    named = attrs.get("oiio:ColorSpace")
    if named:
        found = resolve(cfg, named)
        if found:
            return Detection(found, f"image header ({named})", "medium")
    return Detection(resolve(cfg, "srgb_rec709_display") or SRGB_DISPLAY,
                     "no colour tags: read as sRGB display (not converted)", "assumed")


def _from_tags(raw, cfg) -> Detection:
    t = raw.colour
    prim, trc = (t.primaries if t else None), (t.transfer if t else None)
    if prim is None and trc is None:
        return Detection(resolve(cfg, "g24_rec709_display"), "no colour tags: read as Rec.709 display (not converted)",
                         "assumed")
    p = "bt709" if prim in _SD_PRIMARIES else prim
    tr = "bt709" if trc in ("smpte170m", "bt470bg") else trc
    iid = _TAG_IDS.get((p or "bt709", tr or "bt709"))
    label = f"colour tags {prim or 'unspecified'} / {trc or 'unspecified'}"
    notes = ["SD primaries read as Rec.709"] if prim in _SD_PRIMARIES else []
    if prim is None or trc is None:
        notes.append("the colour tags are incomplete: the missing one read as BT.709")
    found = resolve(cfg, iid) if iid else None
    if found:
        return Detection(found, label, "medium", notes)
    return Detection(None, label, None, notes + [f"no colour space in this OCIO config matches {label}"])


def detect(raw, cfg) -> Detection:
    """What the file is, with what said so and how sure that is."""
    side = _from_sidecar(raw, cfg)
    if side is not None and side.space:
        return side
    path = _first_path(raw)
    ext = os.path.splitext(path)[1].lower()
    if raw.source.kind != "file":
        det = _from_exr(path, cfg) if ext == ".exr" else (
            _from_exr(path, cfg) if ext in _FLOAT_EXT else _from_image_header(path, ext, cfg))
    else:
        det = _from_tags(raw, cfg)
    if side is not None:
        det.notes = side.notes + det.notes
    if det.space is None:
        try:
            rule = cfg.getColorSpaceFromFilepath(path)
            if rule and not cfg.filepathOnlyMatchesDefaultRule(path):
                det.suggestion = rule
        except Exception:  # noqa: BLE001 - a config without file rules
            pass
    return det


# ---- the transform ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Transform:
    kind: str                    # "view" (tone-mapped display / view) | "colorspace" (plain conversion)
    src: str
    dst: str                     # the display for "view", the colour space for "colorspace"
    view: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "src": self.src, "dst": self.dst, "view": self.view}

    @staticmethod
    def from_dict(d: dict | None) -> "Transform | None":
        if not isinstance(d, dict) or not d.get("src") or not d.get("dst"):
            return None
        return Transform(str(d.get("kind") or "colorspace"), str(d["src"]), str(d["dst"]), d.get("view"))

    def describe(self) -> str:
        if self.kind == "view":
            return f"{self.src} -> {self.dst} through the view '{self.view}'"
        return f"{self.src} -> {self.dst}"


# ---- processors ----------------------------------------------------------------------------------------------------
# Always OCIO's exact CPU processor, on threads (applyRGB releases the GIL), in bands of 64k pixels. The tone-mapped
# ACES 2.0 view costs ~620 ms per megapixel on one thread and ~54 ms on 16 (OCIO 2.5.2 and 2.6.0 alike; measured
# 2026-10-26, docs/evidence/L9.53). A baked 3D LUT was 3-5x faster but missed the exact result by up to 18 % on
# saturated colours (ACES 2.0's gamut compressor), so it is not used. Rec.709 / sRGB sources cost nothing.
_PROCS: dict = {}
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max(1, min(16, os.cpu_count() or 4)), thread_name_prefix="c2c_cm")
    return _POOL


def _run(proc, rgb: np.ndarray) -> None:
    """applyRGB on a C-contiguous float32 [..., 3] array, in place, in parallel bands."""
    rows = rgb.reshape(-1, 3)
    step = 1 << 16
    if rows.shape[0] <= step:
        proc.applyRGB(rows)
        return
    list(_pool().map(lambda i: proc.applyRGB(rows[i:i + step]), range(0, rows.shape[0], step)))


def processor(cfg, t: Transform, inverse: bool = False):
    """OCIO's exact CPU processor for the transform (cached)."""
    key = (id(cfg), t, inverse)
    proc = _PROCS.get(key)
    if proc is None:
        import PyOpenColorIO as ocio  # type: ignore[import-not-found]
        if t.kind == "view":
            direction = ocio.TRANSFORM_DIR_INVERSE if inverse else ocio.TRANSFORM_DIR_FORWARD
            p = cfg.getProcessor(ocio.DisplayViewTransform(src=t.src, display=t.dst, view=t.view or "",
                                                           direction=direction))
        else:
            p = cfg.getProcessor(t.dst, t.src) if inverse else cfg.getProcessor(t.src, t.dst)
        proc = _PROCS[key] = p.getDefaultCPUProcessor()
    return proc


def apply(chunk: np.ndarray, t: Transform, inverse: bool = False, cfg=None) -> np.ndarray:
    """Transform the RGB of a float32 [N,H,W,C] chunk in place (alpha untouched); returns the chunk."""
    cfg = cfg if cfg is not None else config()
    proc = processor(cfg, t, inverse)
    out = chunk if (chunk.dtype == np.float32 and chunk.flags.writeable) else chunk.astype(np.float32)
    if out.shape[-1] == 3 and out.flags.c_contiguous:
        _run(proc, out)
        return out
    rgb = np.ascontiguousarray(out[..., :3])
    _run(proc, rgb)
    out[..., :3] = rgb
    return out


def pixel_class(chunk: np.ndarray) -> str:
    """Warn-only sanity check of the first frames: 'scene-linear' (values above 1 / below 0), 'log-like' (lifted
    blacks and nothing near white) or 'display-like'."""
    x = np.asarray(chunk[..., :3][:, ::4, ::4], dtype=np.float32)
    if x.size == 0:
        return "unknown"
    if float((x > 1.002).mean()) > 5e-4 or float((x < -0.002).mean()) > 5e-4:
        return "scene-linear"
    lo, hi = np.percentile(x.mean(axis=-1), [0.5, 99.5])
    if lo >= 0.06 and hi <= 0.93:
        return "log-like"
    return "display-like"


@dataclass
class Colour:
    """Load's colour plan: the record that goes into video_info, and the transform applied on decode."""
    record: dict[str, Any]
    transform: Transform | None
    cfg: Any = None

    def chunks(self, chunks: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        first = True
        for chunk in chunks:
            if first:
                first = False
                self._check_pixels(chunk)
                if self.transform is not None:
                    self.record["method"] = f"exact OCIO processor ({_pool()._max_workers} threads)"
            yield apply(chunk, self.transform, cfg=self.cfg) if self.transform is not None else chunk

    def _check_pixels(self, chunk: np.ndarray) -> None:
        if self.record.get("pixel") is not None:         # the exact-reader fallback decodes again
            return
        cls = pixel_class(chunk)
        self.record["pixel"] = cls
        src = self.record.get("source")
        if not src or self.cfg is None:
            return
        display = is_display_referred(self.cfg, src)
        if cls == "log-like" and display:
            self.record["notes"].append(
                f"the picture looks like camera log (lifted blacks, nothing near white) but reads as {src}: if it is "
                "log, set colorspace_in to the camera's space (e.g. ARRI LogC3 (EI800))")
        elif cls == "scene-linear" and display:
            self.record["notes"].append(f"the pixels go above 1 / below 0, which {src} (a display space) cannot hold")


def _working(cfg, src: str, working_space: str) -> tuple[str, Transform | None, str | None]:
    """(working space, transform or None, note) for a known source."""
    if is_data(cfg, src):
        return src, None, "data (no colour): not converted"
    if working_space == WORK_PRESERVE:
        return src, None, None
    if working_space in (WORK_DEFAULT, "", None):
        if keeps(cfg, src):
            return src, None, None                                 # owner: Rec.709 / sRGB stay as they are
        target = SRGB_DISPLAY if SRGB_DISPLAY in list(cfg.getDisplays()) else cfg.getDefaultDisplay()
    else:
        target = resolve(cfg, working_space)
        if target is None:
            return src, None, f"the working space {working_space!r} is not in this OCIO config: not converted"
    if target == src:
        return src, None, None
    if not is_display_referred(cfg, src) and target in list(cfg.getDisplays()):
        view, vnote = display_view(cfg, target)
        return target, Transform("view", src, target, view), vnote          # owner: the tone-mapped view
    return target, Transform("colorspace", src, target), None


def plan(raw, colorspace_in: str | None, working_space: str | None) -> Colour | None:
    """None = colour management off (saved workflows, API prompts without the input, the kill switch)."""
    cs = str(colorspace_in or "").strip()
    if disabled() or not cs or cs == CS_OFF:
        return None
    cfg = config()
    record: dict[str, Any] = {"mode": "auto" if cs == CS_AUTO else "manual", "source": None, "detected_from": None,
                              "confidence": None, "working": None, "transform": None, "notes": [],
                              "suggestion": None, "pixel": None, "config": None, "config_source": None}
    if cfg is None:
        record["notes"].append("PyOpenColorIO is not available: no colour management (pip install opencolorio)")
        return Colour(record, None)
    record["config"], record["config_source"] = cfg.getName(), config_source()
    if SO._CFG_WARN:
        record["notes"].append(SO._CFG_WARN)
    note = ocio_version_note()
    if note:
        record["notes"].append(note)
    if cs == CS_AUTO:
        det = detect(raw, cfg)
    else:
        name = resolve(cfg, cs)
        det = Detection(name, "colorspace_in", "manual" if name else None,
                        [] if name else [f"{cs!r} is not in this OCIO config: not converted"])
    record.update(source=det.space, detected_from=det.source, confidence=det.confidence, suggestion=det.suggestion)
    record["notes"].extend(det.notes)
    if det.space is None:
        record["notes"].append("Pick the colour space this file was rendered in (colorspace_in): not converted"
                               + (f" - the config's file rules suggest {det.suggestion!r}" if det.suggestion else ""))
        return Colour(record, None, cfg)
    working, t, note = _working(cfg, det.space, str(working_space or WORK_DEFAULT))
    if note:
        record["notes"].append(note)
    record["working"] = working
    record["source_interop_id"] = interop_id(cfg, det.space)
    record["working_interop_id"] = interop_id(cfg, working)
    if t is not None:
        record["transform"] = t.to_dict()
        record["transform_text"] = t.describe()
    return Colour(record, t, cfg)


# ---- Save Video: back to the file's space ---------------------------------------------------------------------------
FROM_LOAD = "auto (from Load)"


def between(cfg, a: str, b: str) -> tuple[Transform, bool]:
    """(transform, inverse) from space a to space b: scene -> display through the display's view, display -> scene
    through its inverse (the same rule Load uses, so the two always meet), anything else a plain conversion."""
    displays = list(cfg.getDisplays())
    a_disp, b_disp = is_display_referred(cfg, a), is_display_referred(cfg, b)
    if a_disp and not b_disp and a in displays:
        return Transform("view", b, a, display_view(cfg, a)[0]), True
    if b_disp and not a_disp and b in displays:
        return Transform("view", a, b, display_view(cfg, b)[0]), False
    return Transform("colorspace", a, b), False


@dataclass
class SavePlan:
    cs_in: str
    cs_out: str
    cm: tuple[Transform, bool] | None                # None: no conversion (or the widgets' own, see save_video)
    notes: list[str]
    reverses_load: bool = False

    def report(self) -> dict[str, Any]:
        t = self.cm[0] if self.cm else None
        return {"in": self.cs_in, "out": self.cs_out, "reverses_load": self.reverses_load,
                "transform": None if t is None else dict(t.to_dict(), inverse=self.cm[1]),
                "transform_text": None if t is None else ("inverse of " if self.cm[1] else "") + t.describe(),
                "notes": list(self.notes)}


def save_plan(record: dict | None, colorspace_in: str | None, colorspace_out: str | None,
              default_in: str) -> SavePlan:
    """Save Video with colorspace_in / colorspace_out on 'auto (from Load)': in = the space Load converted the frames
    to, out = the file's own space (owner: Save reverses Load); exactly Load's transform, inverted, when both are auto."""
    cfg = config()
    auto_in, auto_out = colorspace_in == FROM_LOAD, colorspace_out == FROM_LOAD
    rec = record if isinstance(record, dict) and record.get("source") else None
    notes: list[str] = []
    if rec is None:
        notes.append("no colour record from a Load Video (C2C) upstream" + (
            ": the frames are taken as " + default_in if auto_in else ""))
    if cfg is None:
        cs_in = default_in if auto_in else str(colorspace_in or default_in)
        notes.append("PyOpenColorIO is not available: written without a colour conversion")
        return SavePlan(cs_in, cs_in, None, notes)
    src, work = (rec.get("source"), rec.get("working") or rec.get("source")) if rec else (None, None)
    cs_in = (work or default_in) if auto_in else (resolve(cfg, colorspace_in) or str(colorspace_in or default_in))
    if auto_out:
        cs_out = src or cs_in
    elif str(colorspace_out or "") in ("same as input", "same", ""):
        cs_out = cs_in
    else:
        cs_out = resolve(cfg, colorspace_out) or str(colorspace_out)
    for name in (cs_in, cs_out):
        if cfg.getColorSpace(name) is None:
            notes.append(f"{name!r} is not in this OCIO config: written without a colour conversion")
            return SavePlan(cs_in, cs_in, None, notes)
    if cs_in == cs_out:
        return SavePlan(cs_in, cs_out, None, notes)
    t = Transform.from_dict(rec.get("transform")) if rec else None
    if t is not None and t.dst == cs_in and t.src == cs_out:
        notes.append("reverses Load Video: " + t.describe())
        return SavePlan(cs_in, cs_out, (t, True), notes, reverses_load=True)
    return SavePlan(cs_in, cs_out, between(cfg, cs_in, cs_out), notes)
