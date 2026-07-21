"""
_spline_curves.py — accurate curve families + parametric primitives for the
spline mask editor. Single source of truth for the RASTER side; the JS editor
mirrors this math so the on-canvas preview matches the generated mask exactly.

Curve families (interpolate/approximate a list of control points):
  * b_spline      — uniform cubic B-spline (C2, approximating; smooth roto)
  * nurbs         — rational uniform cubic B-spline w/ per-point weights
                    (equal weights == b_spline; higher weight pulls the curve
                    toward that point). Exact conics come from the primitives.
  * natural_cubic — interpolating natural cubic spline (C2, passes THROUGH every
                    control point; zero end-curvature / periodic when closed)
  * cardinal      — cardinal (Hermite) spline with a tension knob
                    (tension 0 == uniform Catmull-Rom)
catmull_rom / bezier / polyline stay in spline_mask_editor.py (unchanged).

Parametric primitives (exact by construction, from a bounding box + params):
  circle · ellipse · rectangle · rounded_rect · polygon(n) · star(n) · arc

Per-point cusp/smooth: a point flagged ``corner`` breaks the smooth curve into
a sharp vertex — the curve is split at corners and each run sampled as open,
then rejoined (freehand + corner mix, Nuke-style).

Everything returns a flat list of (x, y) float tuples ready for polygon fill.
numpy only; no torch, no cv2 (the caller rasterizes).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

# Curve families implemented HERE (dispatched in _rasterize_splines).
CURVE_TYPES = ("b_spline", "nurbs", "natural_cubic", "cardinal")
PRIMITIVE_TYPES = ("circle", "ellipse", "rectangle", "rounded_rect",
                   "polygon", "star", "arc")


# ── small helpers ────────────────────────────────────────────────────
def _as_xy(points: Sequence) -> List[Point]:
    return [(float(p[0]), float(p[1])) for p in points]


def _linear(points: List[Point], spb: int) -> List[Point]:
    """Fallback: straight polyline sampling (used when <3 points)."""
    if len(points) < 2:
        return list(points)
    out: List[Point] = []
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        for s in range(spb):
            t = s / spb
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    out.append(points[-1])
    return out


# ── uniform cubic B-spline (and rational NURBS via weights) ──────────
_BSPLINE_M = np.array([[-1, 3, -3, 1],
                       [3, -6, 3, 0],
                       [-3, 0, 3, 0],
                       [1, 4, 1, 0]], dtype=np.float64) / 6.0


def _bspline_like(points: List[Point], spb: int, closed: bool,
                  weights: Sequence[float] | None = None) -> List[Point]:
    """Uniform cubic B-spline; rational (NURBS) when weights are given.

    Closed => periodic wrap. Open => endpoints clamped (triple end control
    points) so the curve starts/ends exactly on the first/last point.
    """
    n = len(points)
    if n < 3:
        return _linear(points, spb)
    P = np.asarray(points, dtype=np.float64)
    if weights is None:
        W = np.ones(n)
    else:
        W = np.asarray([float(w) for w in weights], dtype=np.float64)
        if W.shape[0] != n:
            W = np.ones(n)
    W = np.clip(W, 1e-6, None)

    if closed:
        idx = [(-1) % n] + list(range(n)) + [0, 1]
        segs = n
    else:
        idx = [0, 0] + list(range(n)) + [n - 1, n - 1]
        segs = len(idx) - 3

    out: List[Point] = []
    last_seg = segs - 1
    for s in range(segs):
        c = idx[s:s + 4]
        Pc = P[c]                      # (4,2)
        Wc = W[c].reshape(4, 1)        # (4,1)
        steps = spb + (1 if (not closed and s == last_seg) else 0)
        for k in range(steps):
            t = k / spb
            tvec = np.array([t ** 3, t ** 2, t, 1.0])
            basis = tvec @ _BSPLINE_M            # (4,)
            b = basis.reshape(4, 1)
            num = (b * Wc * Pc).sum(axis=0)      # rational numerator
            den = float((b * Wc).sum())
            out.append((float(num[0] / den), float(num[1] / den)))
    return out


# ── natural cubic spline (interpolating, C2) ─────────────────────────
def _natural_cubic_1d(y: np.ndarray, closed: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Return (a=y, second-derivatives k) for a 1-D natural/periodic cubic."""
    n = len(y)
    if closed:
        # periodic cubic spline: solve cyclic tridiagonal for second derivs
        A = np.zeros((n, n))
        rhs = np.zeros(n)
        for i in range(n):
            A[i, (i - 1) % n] = 1.0
            A[i, i] = 4.0
            A[i, (i + 1) % n] = 1.0
            rhs[i] = 6.0 * (y[(i + 1) % n] - 2 * y[i] + y[(i - 1) % n])
        k = np.linalg.solve(A, rhs)
        return y, k
    # natural: k0 = k_{n-1} = 0, solve interior tridiagonal
    k = np.zeros(n)
    if n >= 3:
        m = n - 2
        A = np.zeros((m, m))
        rhs = np.zeros(m)
        for i in range(m):
            A[i, i] = 4.0
            if i > 0:
                A[i, i - 1] = 1.0
            if i < m - 1:
                A[i, i + 1] = 1.0
            rhs[i] = 6.0 * (y[i + 2] - 2 * y[i + 1] + y[i])
        k[1:-1] = np.linalg.solve(A, rhs)
    return y, k


def _natural_cubic(points: List[Point], spb: int, closed: bool) -> List[Point]:
    n = len(points)
    if n < 3:
        return _linear(points, spb)
    P = np.asarray(points, dtype=np.float64)
    _, kx = _natural_cubic_1d(P[:, 0], closed)
    _, ky = _natural_cubic_1d(P[:, 1], closed)
    segs = n if closed else n - 1
    out: List[Point] = []
    last_seg = segs - 1
    for i in range(segs):
        j = (i + 1) % n
        steps = spb + (1 if (not closed and i == last_seg) else 0)
        for s in range(steps):
            t = s / spb
            u = 1.0 - t
            # Hermite form of the cubic segment from second derivatives.
            def _c(p, pj, ki, kj):
                return (u * p + t * pj
                        + ((u ** 3 - u) * ki + (t ** 3 - t) * kj) / 6.0)
            out.append((_c(P[i, 0], P[j, 0], kx[i], kx[j]),
                        _c(P[i, 1], P[j, 1], ky[i], ky[j])))
    return out


# ── cardinal / Hermite spline (tension) ──────────────────────────────
def _cardinal(points: List[Point], spb: int, closed: bool,
              tension: float = 0.0) -> List[Point]:
    """Cardinal spline; tension in [0,1] (0 = Catmull-Rom, 1 = straight)."""
    n = len(points)
    if n < 3:
        return _linear(points, spb)
    c = 1.0 - max(0.0, min(1.0, float(tension)))
    P = points
    if closed:
        ext = [P[-1]] + list(P) + [P[0], P[1]]
        segs = n
    else:
        p_start = (2 * P[0][0] - P[1][0], 2 * P[0][1] - P[1][1])
        p_end = (2 * P[-1][0] - P[-2][0], 2 * P[-1][1] - P[-2][1])
        ext = [p_start] + list(P) + [p_end]
        segs = n - 1
    out: List[Point] = []
    last_seg = segs - 1
    for i in range(segs):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        m1 = (c * (p2[0] - p0[0]) * 0.5, c * (p2[1] - p0[1]) * 0.5)
        m2 = (c * (p3[0] - p1[0]) * 0.5, c * (p3[1] - p1[1]) * 0.5)
        steps = spb + (1 if (not closed and i == last_seg) else 0)
        for s in range(steps):
            t = s / spb
            t2, t3 = t * t, t * t * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            out.append((h00 * p1[0] + h10 * m1[0] + h01 * p2[0] + h11 * m2[0],
                        h00 * p1[1] + h10 * m1[1] + h01 * p2[1] + h11 * m2[1]))
    return out


# ── per-point cusp handling ──────────────────────────────────────────
def _sample_family(points: List[Point], curve_type: str, spb: int,
                   closed: bool, *, weights=None, tension=0.0) -> List[Point]:
    if curve_type == "b_spline":
        return _bspline_like(points, spb, closed)
    if curve_type == "nurbs":
        return _bspline_like(points, spb, closed, weights=weights)
    if curve_type == "natural_cubic":
        return _natural_cubic(points, spb, closed)
    if curve_type == "cardinal":
        return _cardinal(points, spb, closed, tension=tension)
    return _linear(points, spb)


def sample_with_cusps(points: Sequence, curve_type: str, spb: int, closed: bool,
                      *, cusps: Sequence[bool] | None = None,
                      weights: Sequence[float] | None = None,
                      tension: float = 0.0) -> List[Point]:
    """Sample one of the new curve families, honoring per-point corner flags.

    A run of non-corner points is a smooth sub-curve; corner points join runs
    with a sharp vertex. With no corners this is the plain smooth curve.
    """
    pts = _as_xy(points)
    n = len(pts)
    if n < 3:
        return _linear(pts, spb)
    if not cusps or not any(bool(c) for c in cusps):
        return _sample_family(pts, curve_type, spb, closed,
                              weights=weights, tension=tension)

    corner = [bool(cusps[i]) if i < len(cusps) else False for i in range(n)]
    # Build open runs between consecutive corners (wrapping when closed).
    order = list(range(n)) + ([0] if closed else [])
    runs: List[List[int]] = []
    cur: List[int] = []
    for i in order:
        cur.append(i)
        if corner[i] and len(cur) > 1:
            runs.append(cur)
            cur = [i]
    if len(cur) > 1:
        runs.append(cur)

    out: List[Point] = []
    sub_w = None
    for run in runs:
        run_pts = [pts[i] for i in run]
        if weights is not None:
            sub_w = [float(weights[i]) if i < len(weights) else 1.0 for i in run]
        seg = _sample_family(run_pts, curve_type, spb, closed=False,
                             weights=sub_w, tension=tension)
        if out and seg and math.isclose(out[-1][0], seg[0][0], abs_tol=1e-6) \
                and math.isclose(out[-1][1], seg[0][1], abs_tol=1e-6):
            seg = seg[1:]
        out.extend(seg)
    return out


# ── parametric primitives (exact) ────────────────────────────────────
def _bbox(points: Sequence) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def make_primitive(shape_type: str, points: Sequence, samples: int = 64,
                   params: dict | None = None) -> List[Point]:
    """Exact outline for a primitive, from a 2-point bounding box + params.

    params keys: sides (polygon/star), inner_ratio (star), rotation (radians),
    start_angle/end_angle (arc, radians), corner_radius (rounded_rect, px).
    """
    params = params or {}
    if len(points) < 2:
        return _as_xy(points)
    x0, y0, x1, y1 = _bbox(points)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = max(1e-6, (x1 - x0) / 2.0), max(1e-6, (y1 - y0) / 2.0)
    rot = float(params.get("rotation", 0.0))
    cr, sr = math.cos(rot), math.sin(rot)

    def place(dx: float, dy: float) -> Point:
        return (cx + dx * cr - dy * sr, cy + dx * sr + dy * cr)

    n = max(3, int(samples))
    if shape_type in ("circle", "ellipse"):
        r = min(rx, ry) if shape_type == "circle" else None
        return [place((r or rx) * math.cos(2 * math.pi * i / n),
                      (r or ry) * math.sin(2 * math.pi * i / n)) for i in range(n)]
    if shape_type == "rectangle":
        return [place(-rx, -ry), place(rx, -ry), place(rx, ry), place(-rx, ry)]
    if shape_type == "rounded_rect":
        rad = float(params.get("corner_radius", min(rx, ry) * 0.25))
        rad = max(0.0, min(rad, min(rx, ry)))
        per = max(2, n // 4)
        out: List[Point] = []
        corners = [(rx - rad, ry - rad, 0.0), (-rx + rad, ry - rad, math.pi / 2),
                   (-rx + rad, -ry + rad, math.pi), (rx - rad, -ry + rad, 1.5 * math.pi)]
        for ccx, ccy, a0 in corners:
            for k in range(per + 1):
                a = a0 + (math.pi / 2) * (k / per)
                out.append(place(ccx + rad * math.cos(a), ccy + rad * math.sin(a)))
        return out
    if shape_type == "polygon":
        sides = max(3, int(params.get("sides", 6)))
        a0 = float(params.get("start_angle", -math.pi / 2))
        return [place(rx * math.cos(a0 + 2 * math.pi * i / sides),
                      ry * math.sin(a0 + 2 * math.pi * i / sides)) for i in range(sides)]
    if shape_type == "star":
        sides = max(2, int(params.get("sides", 5)))
        inner = max(0.05, min(1.0, float(params.get("inner_ratio", 0.5))))
        a0 = float(params.get("start_angle", -math.pi / 2))
        out = []
        for i in range(sides * 2):
            rr = 1.0 if i % 2 == 0 else inner
            a = a0 + math.pi * i / sides
            out.append(place(rx * rr * math.cos(a), ry * rr * math.sin(a)))
        return out
    if shape_type == "arc":
        a0 = float(params.get("start_angle", 0.0))
        a1 = float(params.get("end_angle", math.pi))
        r = min(rx, ry)
        return [place(r * math.cos(a0 + (a1 - a0) * i / (n - 1)),
                      r * math.sin(a0 + (a1 - a0) * i / (n - 1))) for i in range(n)]
    return _as_xy(points)
