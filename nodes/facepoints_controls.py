"""User controls for Face Landmarks (FacepointsLandmarksC2C), A27 FACE-001.

Pure functions, no model: colours, alpha drawing, processing / output resolution maps, face-mask operations,
temporal landmark stabilisation and stable face identity. The node (nodes/facepoints_landmarks.py) calls them; every
default reproduces the node's earlier output exactly.

Coordinates: points are (x, y) in pixels of the image they belong to. A Resample maps source pixels to another size:
x' = x * sx, y' = y * sy (pixel corners at 0, so a full-size box maps exactly).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── colours ────────────────────────────────────────────────────────────────────────────────────────────────────────

_HEX = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def parse_hex(value: str, what: str) -> Tuple[int, int, int]:
    """'#RRGGBB' / 'RRGGBB' / '#RGB' -> (r, g, b) 0..255. Anything else is an error that names the control."""
    m = _HEX.match(str(value or "").strip())
    if not m:
        raise ValueError(f"{what}: '{value}' is not a colour - write it as #RRGGBB (e.g. #FF8C00)")
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """The node draws with OpenCV on BGR canvases; the controls are RGB (what a colour picker shows)."""
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def bgr_to_hex(bgr: Sequence[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))


def scale_colour(bgr: Sequence[int], k: float) -> Tuple[int, int, int]:
    return tuple(int(v * k) for v in bgr)


# ── drawing with real alpha ────────────────────────────────────────────────────────────────────────────────────────

def _circles(target, pts, colours, radius) -> None:
    import cv2
    for (x, y), col in zip(pts, colours):
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(target, (int(round(x)), int(round(y))), int(radius), col, -1, lineType=cv2.LINE_AA)


def draw_points(canvas: np.ndarray, pts: np.ndarray, colour, radius: int, alpha=None, factor=None) -> None:
    """Filled anti-aliased dots, in point order.

    factor (per point, 0..1) scales a point's colour (the old 'faint' look). alpha (scalar or per point, 0..1):
    fully opaque points are drawn straight onto the canvas in point order - exactly the old drawing; points below 1
    are drawn on a copy and alpha-blended over the canvas, one blend per alpha level (real transparency, not
    darkening). Alpha 0 draws nothing."""
    import cv2
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    n = len(pts)
    a = np.ones(n) if alpha is None else np.broadcast_to(np.asarray(alpha, np.float64), (n,)).copy()
    f = np.ones(n) if factor is None else np.broadcast_to(np.asarray(factor, np.float64), (n,)).copy()
    cols = [colour if f[i] >= 1 else scale_colour(colour, f[i]) for i in range(n)]
    a = np.round(a, 4)
    for level in sorted(set(a.tolist()), reverse=True):
        sel = np.nonzero(a == level)[0]
        if level <= 0 or len(sel) == 0:
            continue
        if level >= 1:
            _circles(canvas, pts[sel], [cols[i] for i in sel], radius)
            continue
        layer = canvas.copy()
        _circles(layer, pts[sel], [cols[i] for i in sel], radius)
        cv2.addWeighted(layer, float(level), canvas, 1.0 - float(level), 0.0, dst=canvas)


def point_alpha(conf: Optional[np.ndarray], n: int, threshold: float, mode: str, low_opacity: float,
                opacity: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """(alpha per point, colour factor per point) for low-confidence handling.

    mode 'dim' = the old look: an unsure point keeps its opacity but its colour is scaled by low_opacity;
    'fade' = real transparency at low_opacity; 'hide' = not drawn; 'keep' = drawn like a sure point.
    opacity multiplies every point's alpha (overlay blend strength)."""
    alpha = np.full(n, float(opacity))
    factor = np.ones(n)
    if conf is None or mode == "keep":
        return alpha, factor
    unsure = np.asarray(conf, np.float64) < float(threshold)
    if mode == "dim":
        factor[unsure] = float(low_opacity)
    elif mode == "fade":
        alpha[unsure] *= float(low_opacity)
    elif mode == "hide":
        alpha[unsure] = 0.0
    else:
        raise ValueError(f"unknown low-confidence mode {mode!r}")
    return alpha, factor


# ── resolution ─────────────────────────────────────────────────────────────────────────────────────────────────────

PROCESSING_SIZES = ("source", "512", "768", "1024", "1536", "2048", "custom")
OUTPUT_SIZES = ("original", "512 x 512", "768 x 768", "1024 x 1024", "1280 x 720", "720 x 1280", "1920 x 1080",
                "1080 x 1920", "custom")


@dataclass(frozen=True)
class Resample:
    """Source pixels -> target pixels: x' = x * sx, y' = y * sy."""
    sx: float = 1.0
    sy: float = 1.0
    width: int = 0
    height: int = 0

    @property
    def identity(self) -> bool:
        return self.sx == 1.0 and self.sy == 1.0

    def points(self, p: np.ndarray) -> np.ndarray:
        if self.identity:
            return p
        return np.asarray(p, np.float64) * np.array([self.sx, self.sy])

    def box(self, b: Sequence[float]) -> Tuple[float, ...]:
        """(x0, y0, side) square crop box: the side scales with the mean factor (it stays a square)."""
        if self.identity:
            return tuple(b)
        x0, y0, side = b
        return (x0 * self.sx, y0 * self.sy, side * (self.sx + self.sy) / 2.0)

    def length(self, v):
        """A length in px (std, face size): the mean factor (exact when the aspect is kept)."""
        return v if self.identity else np.asarray(v, np.float64) * (self.sx + self.sy) / 2.0

    def inverse(self, src_w: int, src_h: int) -> "Resample":
        return Resample(1.0 / self.sx, 1.0 / self.sy, src_w, src_h)


def processing_resample(w: int, h: int, preset: str, custom_max: int) -> Resample:
    """Downscale only: the longer side to at most the chosen maximum, aspect kept. 'source' = no change."""
    if preset not in PROCESSING_SIZES:
        raise ValueError(f"resolution_processing: unknown preset {preset!r} (one of {', '.join(PROCESSING_SIZES)})")
    if preset == "source":
        return Resample(1.0, 1.0, w, h)
    limit = int(custom_max) if preset == "custom" else int(preset)
    if limit < 64:
        raise ValueError("resolution_processing_max: at least 64 px")
    if max(w, h) <= limit:
        return Resample(1.0, 1.0, w, h)
    s = limit / float(max(w, h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return Resample(nw / w, nh / h, nw, nh)


def output_resample(w: int, h: int, preset: str, cw: int, ch: int, keep_aspect: bool) -> Resample:
    """'original' = the input size. A target W x H: with keep_aspect the image is scaled to fit inside it (the
    output is that fitted size, no padding); without, it is stretched to exactly W x H."""
    if preset not in OUTPUT_SIZES:
        raise ValueError(f"resolution_output: unknown preset {preset!r} (one of {', '.join(OUTPUT_SIZES)})")
    if preset == "original":
        return Resample(1.0, 1.0, w, h)
    if preset == "custom":
        tw, th = int(cw), int(ch)
    else:
        tw, th = (int(v) for v in preset.split(" x "))
    if tw < 16 or th < 16:
        raise ValueError("resolution_output_width / _height: at least 16 px")
    if keep_aspect:
        s = min(tw / w, th / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    else:
        nw, nh = tw, th
    if (nw, nh) == (w, h):
        return Resample(1.0, 1.0, w, h)
    return Resample(nw / w, nh / h, nw, nh)


def resize_image(img: np.ndarray, r: Resample) -> np.ndarray:
    import cv2
    if r.identity:
        return img
    shrink = r.width * r.height < img.shape[0] * img.shape[1]
    return cv2.resize(img, (r.width, r.height), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


# ── face mask ──────────────────────────────────────────────────────────────────────────────────────────────────────

MASK_PREVIEWS = ("off", "tint overlay", "outline on overlay")


def chaikin(poly: np.ndarray, iterations: int) -> np.ndarray:
    """Corner cutting (Chaikin): each pass replaces every edge by its 1/4 and 3/4 points - a rounder closed hull."""
    p = np.asarray(poly, np.float64).reshape(-1, 2)
    for _ in range(int(iterations)):
        q = np.roll(p, -1, axis=0)
        p = np.stack([0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q], 1).reshape(-1, 2)
    return p


def hull_polygon(points: np.ndarray, smoothing: int) -> np.ndarray:
    import cv2
    hull = cv2.convexHull(np.round(points).astype(np.int32)).reshape(-1, 2)
    if smoothing <= 0:
        return hull
    return chaikin(hull, smoothing)


def fill_hull(mask_u8: np.ndarray, points: np.ndarray, smoothing: int) -> None:
    """The old mask (smoothing 0: integer convex hull, fillConvexPoly) or a Chaikin-smoothed hull (sub-pixel)."""
    import cv2
    poly = hull_polygon(points, smoothing)
    if smoothing <= 0:
        cv2.fillConvexPoly(mask_u8, poly, 255)
    else:
        cv2.fillPoly(mask_u8, [np.round(poly * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)


def shape_mask(mask_u8: np.ndarray, grow: int, shrink: int, feather: float, softness: float,
               threshold: float) -> np.ndarray:
    """uint8 hull mask -> float32 0..1. Order: grow (dilate) -> shrink (erode) -> softness (inward ramp over N px,
    the outer edge stays put) -> feather (Gaussian blur, sigma px) -> threshold (> 0: hard mask at that level)."""
    import cv2
    m = mask_u8
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(grow) + 1, 2 * int(grow) + 1))
        m = cv2.dilate(m, k)
    if shrink > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(shrink) + 1, 2 * int(shrink) + 1))
        m = cv2.erode(m, k)
    out = m.astype(np.float32) / 255.0
    if softness > 0:
        inside = (m > 127).astype(np.uint8)
        dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
        out = np.minimum(out, np.clip(dist / float(softness), 0.0, 1.0)).astype(np.float32)
    if feather > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=float(feather), sigmaY=float(feather))
    if threshold > 0:
        out = (out >= float(threshold)).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def preview_mask(overlay_bgr: np.ndarray, mask: np.ndarray, mode: str, colour_bgr=(0, 200, 80),
                 opacity: float = 0.35) -> None:
    import cv2
    if mode == "off":
        return
    if mode == "tint overlay":
        a = (mask * float(opacity))[..., None]
        tint = np.empty_like(overlay_bgr, dtype=np.float32)
        tint[:] = colour_bgr
        overlay_bgr[:] = np.clip(overlay_bgr.astype(np.float32) * (1 - a) + tint * a + 0.5, 0, 255).astype(np.uint8)
    elif mode == "outline on overlay":
        contours, _ = cv2.findContours((mask >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay_bgr, contours, -1, colour_bgr, 1, lineType=cv2.LINE_AA)
    else:
        raise ValueError(f"mask_preview: unknown mode {mode!r} (one of {', '.join(MASK_PREVIEWS)})")


# ── temporal stabilisation ─────────────────────────────────────────────────────────────────────────────────────────

def frame_change(prev_bgr: Optional[np.ndarray], bgr: np.ndarray) -> float:
    """Mean absolute difference of two frames on a 64 px thumbnail, 0..1 (a scene cut is large)."""
    import cv2
    if prev_bgr is None or prev_bgr.shape != bgr.shape:
        return 1.0
    a = cv2.resize(prev_bgr, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.abs(a - b).mean() / 255.0)


@dataclass
class PointFilter:
    """An adaptive (One-Euro style) filter over a set of points: still points are smoothed, moving points follow.

    For each point: speed = |raw - previous output| in units of the face size. The blend toward the new raw point is
    a = base + (1 - base) * min(1, speed / fast), with base = 1 - strength and fast = 1 % + 5 % x strength of the
    face per frame (unless given). Strength 0 = no smoothing (a = 1); near 1 = heavy smoothing of still points. A
    fast move (a turn, a jump, a quick expression) drives a toward 1, so genuine motion is not held back. confidence-aware: a point the model is unsure of moves less toward its new position
    (a scaled by its confidence relative to the previous one). max_offset > 0: the output never lags the raw point
    by more than this many px."""
    strength: float = 0.5
    confidence_aware: bool = True
    max_offset: float = 0.0
    fast: Optional[float] = None           # a move of this fraction of the face per frame counts as fast
    state: Optional[np.ndarray] = None
    conf: Optional[np.ndarray] = None

    def reset(self):
        self.state, self.conf = None, None

    def __call__(self, raw: np.ndarray, conf: Optional[np.ndarray], face_px: float) -> np.ndarray:
        raw = np.asarray(raw, np.float64)
        if self.state is None or self.state.shape != raw.shape or self.strength <= 0:
            self.state, self.conf = raw.copy(), None if conf is None else np.asarray(conf, np.float64).copy()
            return raw.copy()
        speed = np.linalg.norm(raw - self.state, axis=1) / max(1e-6, float(face_px))
        base = 1.0 - min(0.999, float(self.strength))
        fast = self.fast if self.fast else 0.01 + 0.05 * float(self.strength)
        a = base + (1.0 - base) * np.minimum(1.0, speed / fast)
        if self.confidence_aware and conf is not None:
            c = np.asarray(conf, np.float64)
            prev = self.conf if self.conf is not None else c
            rel = c / np.maximum(1e-6, c + prev)              # 0.5 when as sure as before, < 0.5 when less sure
            a = np.clip(a * 2.0 * rel, base * 0.25, 1.0)
        out = self.state + a[:, None] * (raw - self.state)
        if self.max_offset > 0:
            d = out - raw
            n = np.linalg.norm(d, axis=1, keepdims=True)
            over = n[:, 0] > self.max_offset
            out[over] = raw[over] + d[over] / n[over] * self.max_offset
        self.state = out.copy()
        if conf is not None:
            self.conf = np.asarray(conf, np.float64).copy()
        return out


@dataclass
class FaceTrack:
    """One face identity across frames with a filter per point set."""
    ident: int
    centre: np.ndarray
    size: float
    filters: Dict[str, PointFilter] = field(default_factory=dict)
    missed: int = 0


def match_faces(tracks: List[FaceTrack], centres: List[np.ndarray], sizes: List[float]) -> List[Optional[int]]:
    """Greedy nearest match of this frame's faces to existing tracks (centre distance < 0.6 x face size).
    Returns, per face, the index of its track or None (a new face)."""
    pairs = []
    for fi, (c, s) in enumerate(zip(centres, sizes)):
        for ti, t in enumerate(tracks):
            d = float(np.linalg.norm(np.asarray(c) - t.centre))
            if d < 0.6 * max(s, t.size):
                pairs.append((d, fi, ti))
    pairs.sort()
    out: List[Optional[int]] = [None] * len(centres)
    used = set()
    for d, fi, ti in pairs:
        if out[fi] is None and ti not in used:
            out[fi] = ti
            used.add(ti)
    return out


# ── connection lines (pose / contour strength, work order 2026-11-14 Face Landmarks) ──────────────────────────────
# The published point orders of the two standard sets (iBUG 300-W 68 and WFLW 98): (first, last, closed) runs.
LINES_68 = ((0, 16, False), (17, 21, False), (22, 26, False), (27, 30, False), (31, 35, False), (36, 41, True),
            (42, 47, True), (48, 59, True), (60, 67, True))
LINES_98 = ((0, 32, False), (33, 41, True), (42, 50, True), (51, 54, False), (55, 59, False), (60, 67, True),
            (68, 75, True), (76, 87, True), (88, 95, True))
MASK_FACES = ("all faces", "largest face", "face index")


def draw_lines(canvas: np.ndarray, pts: np.ndarray, runs, colour, width: int, alpha: float) -> None:
    """Anti-aliased polylines along the set's contours; alpha < 1 is blended over the canvas (real transparency)."""
    import cv2
    if alpha <= 0:
        return
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    target = canvas if alpha >= 1 else canvas.copy()
    for a, b, closed in runs:
        if b >= len(pts):
            continue
        seg = pts[a:b + 1]
        if not np.isfinite(seg).all():
            continue
        cv2.polylines(target, [np.round(seg * 16).astype(np.int32)], bool(closed), colour, int(width),
                      lineType=cv2.LINE_AA, shift=4)
    if alpha < 1:
        cv2.addWeighted(target, float(alpha), canvas, 1.0 - float(alpha), 0.0, dst=canvas)


# L9.252 F4: a track not seen for this many frames is dropped (5 s at 24 fps - far beyond any normal occlusion), so a
# long crowded clip keeps a bounded track list and a new face entering an old face's spot gets a new identity.
MAX_MISSED_FRAMES = 120


def prune_tracks(tracks: List["FaceTrack"], max_missed: int = MAX_MISSED_FRAMES) -> List["FaceTrack"]:
    return [t for t in tracks if t.missed <= max_missed]


def sticky_largest(sizes: List[float], identities: List[Optional[int]], locked: Optional[int]
                   ) -> Tuple[List[int], Optional[int]]:
    """L9.252 F2: "largest face" on video with identity tracking. The face that is largest when the choice is made is
    kept while it stays in view (it does not jump to another person who is momentarily bigger); a new choice is made
    when it is gone (or after a scene cut, where the caller passes locked=None). Returns (picked, locked identity)."""
    if not sizes:
        return [], locked
    if locked is not None and locked in identities:
        return [identities.index(locked)], locked
    k = int(np.argmax(sizes))
    return [k], identities[k]


def mask_face_selection(sizes: List[float], mode: str, index: int) -> List[int]:
    """Which faces (by position in the output order) fill the face mask."""
    if mode not in MASK_FACES:
        raise ValueError(f"mask_faces must be one of {', '.join(MASK_FACES)}; got {mode!r}")
    n = len(sizes)
    if n == 0 or mode == MASK_FACES[0]:
        return list(range(n))
    if mode == MASK_FACES[1]:
        return [int(np.argmax(sizes))]
    return [int(index)] if 0 <= int(index) < n else []


# ── head parts for face_mask (L9.279, owner 2026-11-17; design docs/research/FACE_PARTS_ROBUST_design.md) ────────────
# 68 set (iBUG 300-W): jaw 0-16, brows 17-21 / 22-26, nose 27-35, eyes 36-41 / 42-47, outer lips 48-59, inner 60-67.

PART_DEFAULT = "whole face (default)"
PART_CUSTOM = "custom list"
PARTS_FEATURES = ("eyes", "eye (image left)", "eye (image right)", "eyelids", "brows", "nose", "lips",
                  "inner mouth / teeth")
PARTS_AREAS = ("around the mouth", "cheeks", "chin + jaw")
PARTS_ESTIMATED = ("forehead (estimated)", "ears (estimated)", "hair (estimated)", "head + hair (estimated)")
MASK_PARTS = (PART_DEFAULT,) + PARTS_FEATURES + PARTS_AREAS + PARTS_ESTIMATED + (PART_CUSTOM,)
HEAD_REFINE = ("image edges (GrabCut)", "off")


@dataclass
class FaceFrame:
    """A face's own axes (image px): up = chin -> brow middle, right = perpendicular; H = that distance, W = jaw width.
    angle_deg = counter-clockwise tilt of up from the image's up (the _mask_directional convention)."""
    centre: np.ndarray
    up: np.ndarray
    right: np.ndarray
    H: float
    W: float
    angle_deg: float


def face_frame(p68: np.ndarray) -> FaceFrame:
    p = np.asarray(p68, np.float64)
    brow = p[17:27].mean(0)
    chin = p[8]
    v = brow - chin
    H = float(np.hypot(*v)) or 1.0
    up = v / H
    right = np.array([-up[1], up[0]])
    if np.dot(right, p[16] - p[0]) < 0:
        right = -right
    W = float(np.hypot(*(p[16] - p[0]))) or H
    angle = float(np.degrees(np.arctan2(-up[0], -up[1])))
    return FaceFrame((brow + chin) / 2.0, up, right, H, W, angle)


def _poly(mask: np.ndarray, pts, smoothing: int = 0) -> None:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) < 3 or not np.isfinite(pts).all():
        return
    fill_hull(mask, pts, smoothing)


def _grown(pts, k: float, about=None) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    c = pts.mean(0) if about is None else np.asarray(about, np.float64)
    return c + (pts - c) * float(k)


def _ellipse(fr: FaceFrame, centre, ax_right: float, ax_up: float, n: int = 48) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return (np.asarray(centre)[None] + np.cos(t)[:, None] * ax_right * fr.right[None]
            + np.sin(t)[:, None] * ax_up * fr.up[None])


def part_mask(part: str, shape, p68: np.ndarray, pts620: np.ndarray, classes: np.ndarray, mesh: np.ndarray,
              smoothing: int = 0) -> np.ndarray:
    """uint8 mask (0 / 255) of one named part for one face, image px. 'whole face' is the mesh hull."""
    m = np.zeros(shape, np.uint8)
    p = np.asarray(p68, np.float64)
    fr = face_frame(p)
    cls = np.asarray(classes)
    pts = np.asarray(pts620, np.float64)
    if part in ("whole face", PART_DEFAULT):
        _poly(m, pts[mesh], smoothing)
    elif part in ("eyes", "eye (image left)", "eye (image right)"):
        eyes = sorted([p[36:42], p[42:48]], key=lambda e: float(e[:, 0].mean()))      # image-left first
        pick = eyes if part == "eyes" else [eyes[0] if "left" in part else eyes[1]]
        for e in pick:
            _poly(m, _grown(e, 1.6), smoothing)
    elif part == "eyelids":
        lid = pts[cls == "lid"]
        for e in (p[36:42], p[42:48]):
            near = lid[np.linalg.norm(lid - e.mean(0), axis=1) < 0.25 * fr.W] if len(lid) else lid
            _poly(m, np.concatenate([_grown(e, 1.35), near]) if len(near) else _grown(e, 1.5), smoothing)
    elif part == "brows":
        for b in (p[17:22], p[22:27]):
            _poly(m, np.concatenate([b + fr.up * 0.06 * fr.H, b - fr.up * 0.06 * fr.H]), smoothing)
    elif part == "nose":
        _poly(m, _grown(p[27:36], 1.15), smoothing)
    elif part == "lips":
        _poly(m, p[48:60], smoothing)
    elif part == "inner mouth / teeth":
        teeth = pts[np.isin(cls, ("teeth_u", "teeth_l"))]
        _poly(m, np.concatenate([p[60:68], teeth]) if len(teeth) else p[60:68], smoothing)
    elif part == "around the mouth":
        lips = p[48:60]
        _poly(m, np.concatenate([lips, lips + fr.up * 0.12 * fr.H, lips - fr.up * 0.12 * fr.H,
                                 _grown(lips, 1.25)]), smoothing)
    elif part == "cheeks":
        _poly(m, np.stack([p[1], p[2], p[3], p[4], p[5], p[48], p[31], p[40], p[41], p[36]]), smoothing)
        _poly(m, np.stack([p[15], p[14], p[13], p[12], p[11], p[54], p[35], p[47], p[46], p[45]]), smoothing)
    elif part == "chin + jaw":
        _poly(m, np.concatenate([p[4:13], p[55:60]]), smoothing)
    elif part == "forehead (estimated)":
        b = p[17:27]
        _poly(m, np.concatenate([b, b + fr.up * 0.55 * fr.H]), smoothing)
    elif part == "ears (estimated)":
        for side, sgn in ((p[0:4], -1.0), (p[13:17], 1.0)):
            _poly(m, np.concatenate([side, side + fr.right * sgn * 0.18 * fr.W]), smoothing)
    elif part in ("head + hair (estimated)", "hair (estimated)"):
        brow = p[17:27].mean(0)
        top = brow + fr.up * 0.95 * fr.H
        centre = (top + brow) / 2.0
        head = np.zeros(shape, np.uint8)
        _poly(head, np.concatenate([_ellipse(fr, centre, 0.62 * fr.W * 1.15, 0.475 * fr.H + 0.2 * fr.H),
                                    pts[mesh], p[0:17]]), smoothing)
        if part == "head + hair (estimated)":
            m = head
        else:
            face = np.zeros(shape, np.uint8)
            _poly(face, pts[mesh], smoothing)
            _poly(face, np.concatenate([p[17:27], p[17:27] + fr.up * 0.12 * fr.H]), smoothing)  # brows stay face
            m = np.where(face > 0, 0, head).astype(np.uint8) & above_eyes(shape, p)
    else:
        raise ValueError(f"mask_part: unknown part {part!r}; one of {', '.join(MASK_PARTS)}")
    return m


def above_eyes(shape, p68: np.ndarray, below: float = 0.15) -> np.ndarray:
    """uint8 0 / 255: the side of the face's eye line where the hair grows (along the head's own up), reaching
    `below` x face height under the eye line so sideburns are kept. Hair never comes from the jaw or the neck."""
    p = np.asarray(p68, np.float64)
    fr = face_frame(p)
    eye = p[36:48].mean(0)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    d = (xx - eye[0]) * fr.up[0] + (yy - eye[1]) * fr.up[1]
    return (d > -below * fr.H).astype(np.uint8) * 255


def parse_parts(part: str, custom: str) -> List[str]:
    """The selected parts; custom list = comma separated names (case and spaces ignored), each one checked."""
    if part != PART_CUSTOM:
        return [part]
    names = {n.lower(): n for n in MASK_PARTS if n not in (PART_DEFAULT, PART_CUSTOM)}
    names["whole face"] = "whole face"
    out = []
    for raw in str(custom or "").split(","):
        k = raw.strip().lower()
        if not k:
            continue
        if k not in names:
            raise ValueError(f"mask_parts_custom: '{raw.strip()}' is not a part; use names from mask_part, e.g. "
                             "'eyes, lips, hair (estimated)'")
        out.append(names[k])
    if not out:
        raise ValueError("mask_parts_custom is empty: list the parts, e.g. 'eyes, lips'")
    return out


def refine_head(image_bgr: np.ndarray, estimate_u8: np.ndarray, sure_u8: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Fit an estimated head / hair region to the picture with GrabCut: sure foreground = the face, probable
    foreground = the estimate, probable background = a band around it, background beyond. Runs on a crop scaled to
    <= 256 px, so it is cheap. Returns uint8 0 / 255 (the estimate itself when GrabCut cannot run)."""
    import cv2
    ys, xs = np.nonzero(estimate_u8)
    if len(xs) < 16:
        return estimate_u8
    h, w = estimate_u8.shape
    cx0, cx1, cy0, cy1 = xs.min(), xs.max(), ys.min(), ys.max()
    pw, ph = int(0.4 * (cx1 - cx0 + 1)), int(0.4 * (cy1 - cy0 + 1))
    x0, x1, y0, y1 = max(0, cx0 - pw), min(w, cx1 + pw + 1), max(0, cy0 - ph), min(h, cy1 + ph + 1)
    crop = image_bgr[y0:y1, x0:x1]
    s = min(1.0, 256.0 / max(crop.shape[:2]))
    size = (max(8, int(crop.shape[1] * s)), max(8, int(crop.shape[0] * s)))
    small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    est = cv2.resize(estimate_u8[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST)
    sure = cv2.resize(sure_u8[y0:y1, x0:x1], size, interpolation=cv2.INTER_NEAREST)
    gc = np.full(est.shape, cv2.GC_BGD, np.uint8)
    k = max(3, int(0.08 * max(size)) | 1)
    band = cv2.dilate(est, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    gc[band > 0] = cv2.GC_PR_BGD
    gc[est > 0] = cv2.GC_PR_FGD
    gc[sure > 0] = cv2.GC_FGD
    if not (gc == cv2.GC_BGD).any() or not (gc == cv2.GC_FGD).any():
        return estimate_u8
    try:
        cv2.grabCut(np.ascontiguousarray(small), gc, None, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), int(iterations), cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return estimate_u8
    fg = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    out = np.zeros_like(estimate_u8)
    out[y0:y1, x0:x1] = cv2.resize(fg, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return out


# ── robust detection (L9.280, owner 2026-11-17): analysis + a DETECTION-ONLY enhancement ladder ──────────────────────
# The enhanced copies are only shown to the face finder and the landmark model; every output keeps the original frame.

ROBUST_MODES = ("auto (when a frame finds no face)", "off", "always")


def analyse_frame(bgr: np.ndarray) -> Dict[str, float]:
    """Luma median / 5th / 95th percentile (0..1), colour cast (largest channel-mean ratio - 1) and sharpness
    (variance of the Laplacian of the luma at <= 512 px)."""
    import cv2
    small = bgr if max(bgr.shape[:2]) <= 512 else cv2.resize(
        bgr, (int(bgr.shape[1] * 512 / max(bgr.shape[:2])), int(bgr.shape[0] * 512 / max(bgr.shape[:2]))),
        interpolation=cv2.INTER_AREA)
    luma = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    means = small.reshape(-1, 3).mean(0) + 1e-3
    return {"luma_median": round(float(np.median(luma)), 4), "luma_p05": round(float(np.percentile(luma, 5)), 4),
            "luma_p95": round(float(np.percentile(luma, 95)), 4),
            "colour_cast": round(float(means.max() / means.min() - 1.0), 4),
            "sharpness": round(float(cv2.Laplacian((luma * 255).astype(np.uint8), cv2.CV_32F).var()), 2)}


def _lift_light(bgr: np.ndarray, a: Dict[str, float]) -> np.ndarray:
    """Exposure + gamma toward a median luma of ~0.42, then CLAHE on L (Lab): faces in the dark get contrast back."""
    import cv2
    x = bgr.astype(np.float32) / 255.0
    lo, hi = a["luma_p05"], max(a["luma_p95"], a["luma_p05"] + 1e-3)
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)                                    # stretch the used range
    med = float(np.clip((a["luma_median"] - lo) / (hi - lo), 1e-3, 0.999))
    g = float(np.clip(np.log(0.42) / np.log(med), 0.25, 1.6))
    x = np.power(x, g)
    lab = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _neutral_colour(bgr: np.ndarray) -> np.ndarray:
    """Grey-world white balance (each channel scaled to the common mean)."""
    x = bgr.astype(np.float32)
    m = x.reshape(-1, 3).mean(0) + 1e-3
    return np.clip(x * (m.mean() / m), 0, 255).astype(np.uint8)


def _luma_only(bgr: np.ndarray) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.merge([g, g, g])


def _sharpen(bgr: np.ndarray, a: Dict[str, float]) -> np.ndarray:
    """Unsharp mask, stronger the blurrier the frame (sharpness = Laplacian variance; ~100+ is a sharp frame)."""
    import cv2
    amount = float(np.clip(1.8 - a["sharpness"] / 80.0, 0.5, 1.8))
    sigma = float(np.clip(3.0 - a["sharpness"] / 60.0, 1.0, 3.0))
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def enhancement_ladder(bgr: np.ndarray, a: Optional[Dict[str, float]] = None) -> List[Tuple[str, np.ndarray]]:
    """[(name, enhanced BGR copy)] in the order to try, chosen from the analysis: low light, colour cast, blur, and
    every problem at once. Cheap (OpenCV, one frame)."""
    a = a or analyse_frame(bgr)
    dark = a["luma_median"] < 0.3 or a["luma_p95"] < 0.55
    cast = a["colour_cast"] > 0.25
    blurry = a["sharpness"] < 60.0
    import cv2
    rungs: List[Tuple[str, np.ndarray]] = []
    lit = _lift_light(bgr, a)
    rungs.append(("light", lit))
    if dark:                                 # lifting dark footage lifts its sensor noise: smooth first, then lift
        quiet = cv2.bilateralFilter(cv2.GaussianBlur(bgr, (0, 0), 1.2), 7, 40, 7)
        rungs.append(("light + denoise", _lift_light(quiet, analyse_frame(quiet))))
    if cast:
        rungs.append(("colour", _neutral_colour(lit)))
    rungs.append(("luma only", _luma_only(lit)))
    if blurry:
        rungs.append(("deblur", _sharpen(lit, a)))
    both = _sharpen(_neutral_colour(lit), a) if (cast or blurry) else lit
    if cast and blurry:
        rungs.append(("light + colour + deblur", both))
    if not dark and not cast and not blurry:
        rungs.append(("contrast", _sharpen(lit, {**a, "sharpness": 60.0})))
    return rungs
