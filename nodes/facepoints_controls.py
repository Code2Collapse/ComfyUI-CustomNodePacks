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
