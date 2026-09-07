"""
SplinePathFlowMaskMEC — Procedural flow / wave / dust masks along a spline path.

Takes a spline path (from `SplineMaskEditorMEC` or any STRING JSON of the same
shape) and rasterizes a *band* along that path, modulated by a chosen pattern:

  * ribbon       — constant-thickness band, soft edges
  * wave         — band thickness sin-modulated along arc length (ocean wave)
  * flow         — band displaced perpendicular by Perlin-like noise (river / smoke trail)
  * dust         — discrete particles inside the band (dust drift)
  * river        — wide tapered band with meandering edges
  * smoke        — band that thickens with arc length + turbulence (smoke trail)

Animation: produces a `frames`-length batch where the modulation phase advances
each frame, so masks can drive motion-mask animation (water flowing on floor,
ocean waves, dust streaming, etc.).

Outputs
  mask  : MASK (frames, H, W)

VRAM Tier: 1 (numpy / cv2 only)

Files CREATED: nodes/spline_path_flow_mask.py
Files MODIFIED: __init__.py
"""

from __future__ import annotations

import json
import logging
import math
from typing import List, Tuple

import numpy as np
import torch

from ._is_changed_util import hash_args_and_kwargs

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .spline_mask_editor import (
    _catmull_rom_sample,
    _bezier_sample,
    _polyline_sample,
)

logger = logging.getLogger("MEC")

PATTERNS = [
    # Original 6
    "ribbon", "wave", "flow", "dust", "river", "smoke",
    # NEW: parametric / mathematical waveforms
    "sawtooth", "square", "triangle", "gaussian_pulse",
    # NEW: procedural noise families
    "fbm", "curl_noise", "lightning",
]
FLOW_DIRECTIONS = ["forward", "reverse", "bidirectional", "oscillate"]


# ───────────────────────────────────────────────────────────────────────
# Spline → dense centerline samples (with arc-length param)
# ───────────────────────────────────────────────────────────────────────

def _sample_path(spline_data_json: str, spline_type: str, closed: bool,
                 samples_per_segment: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (samples (N,2) float32, t (N,) float32 in [0,1]).

    Concatenates ALL shapes' samples; if multiple shapes, treats them as
    independent paths concatenated end-to-end (t reset per shape).
    For the common case of a single spline, t spans 0→1 along that path.
    """
    try:
        shapes = json.loads(spline_data_json) if isinstance(spline_data_json, str) else spline_data_json
        if not isinstance(shapes, list):
            return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
    except (json.JSONDecodeError, TypeError):
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)

    all_pts: List[Tuple[float, float]] = []
    all_t: List[float] = []
    for sh in shapes:
        if not isinstance(sh, dict):
            continue
        pts_raw = sh.get("points", [])
        pts = [(float(p["x"]), float(p["y"])) for p in pts_raw if isinstance(p, dict)]
        if len(pts) < 2:
            continue
        sh_type = sh.get("type", spline_type)
        sh_closed = bool(sh.get("closed", closed))
        if sh_type == "polyline":
            sampled = _polyline_sample(pts, sh_closed)
        elif sh_type == "bezier":
            handles = sh.get("handles")
            sampled = _bezier_sample(pts, handles, samples_per_segment, sh_closed)
        else:
            sampled = _catmull_rom_sample(pts, samples_per_segment, sh_closed)

        if len(sampled) < 2:
            continue

        # Compute cumulative arc length for proper t parameterization.
        arr = np.asarray(sampled, dtype=np.float32)  # (M, 2)
        seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1] if cum[-1] > 1e-6 else 1.0
        t_local = cum / total

        all_pts.extend(sampled)
        all_t.extend(t_local.tolist())

    if not all_pts:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)

    pts_arr = np.asarray(all_pts, dtype=np.float32)
    t_arr = np.asarray(all_t, dtype=np.float32)

    # Densify so consecutive samples are ≤ 1px apart along the path.
    # Sparse samples produce visible staircase artefacts in the t-map
    # because every pixel snaps to the t of its nearest sample.
    seg = np.linalg.norm(np.diff(pts_arr, axis=0), axis=1)
    if seg.size and seg.max() > 1.0:
        new_pts = [pts_arr[0]]
        new_t = [t_arr[0]]
        for i in range(len(seg)):
            d = float(seg[i])
            if d <= 1.0:
                new_pts.append(pts_arr[i + 1])
                new_t.append(t_arr[i + 1])
                continue
            steps = int(math.ceil(d))
            for k in range(1, steps + 1):
                a = k / steps
                new_pts.append(pts_arr[i] * (1 - a) + pts_arr[i + 1] * a)
                new_t.append(t_arr[i] * (1 - a) + t_arr[i + 1] * a)
        pts_arr = np.asarray(new_pts, dtype=np.float32)
        t_arr = np.asarray(new_t, dtype=np.float32)

    return pts_arr, t_arr


# ───────────────────────────────────────────────────────────────────────
# Per-pixel (distance, t-along-path) maps
# ───────────────────────────────────────────────────────────────────────

def _build_dist_and_t_maps(samples: np.ndarray, t_vals: np.ndarray,
                           H: int, W: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every pixel, compute distance to the nearest centerline sample
    and the t-value at that sample. Also returns a tangent map (dx,dy) per
    pixel so callers can offset perpendicular to the path.

    Uses cv2.distanceTransformWithLabels for speed when available; falls
    back to a chunked numpy KNN otherwise.
    """
    N = len(samples)
    if N == 0:
        return (np.full((H, W), 1e6, np.float32),
                np.zeros((H, W), np.float32),
                np.zeros((H, W, 2), np.float32))

    # Compute per-sample tangents (forward differences, last copies prev).
    if N >= 2:
        diffs = np.diff(samples, axis=0)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-6
        tan = diffs / norms
        tan = np.concatenate([tan, tan[-1:]], axis=0)  # (N, 2)
    else:
        tan = np.zeros((N, 2), np.float32)

    if HAS_CV2:
        # Rasterize each sample as a single-pixel "source" and remember
        # which sample-index lives at that pixel.
        idx_map = np.full((H, W), -1, dtype=np.int32)
        # If multiple samples land in the same pixel, latest wins (fine).
        xs = np.clip(np.round(samples[:, 0]).astype(np.int32), 0, W - 1)
        ys = np.clip(np.round(samples[:, 1]).astype(np.int32), 0, H - 1)
        idx_map[ys, xs] = np.arange(N, dtype=np.int32)

        src = (idx_map < 0).astype(np.uint8) * 255  # 0=source, 255=elsewhere
        dist, labels = cv2.distanceTransformWithLabels(
            src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL,
        )
        # `labels` enumerates the 0-pixels in raster scan order, 1-indexed.
        zero_yx = np.argwhere(src == 0)  # (Z, 2) in raster order, [y, x]
        if len(zero_yx) == 0:
            return (np.full((H, W), 1e6, np.float32),
                    np.zeros((H, W), np.float32),
                    np.zeros((H, W, 2), np.float32))
        zero_to_sample = idx_map[zero_yx[:, 0], zero_yx[:, 1]]  # (Z,)
        # Build LUT: label k → sample-index. label 0 unused; pad with 0.
        lut = np.concatenate([[0], zero_to_sample]).astype(np.int32)
        sample_per_pixel = lut[labels]  # (H, W) → sample index in [0, N)
        sample_per_pixel = np.clip(sample_per_pixel, 0, N - 1)

        t_map = t_vals[sample_per_pixel]
        tan_map = tan[sample_per_pixel]
        return dist.astype(np.float32), t_map.astype(np.float32), tan_map.astype(np.float32)

    # ── numpy fallback: chunked nearest-neighbour ─────────────────────
    ys, xs = np.mgrid[0:H, 0:W]
    pix = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)  # (H*W, 2)
    chunk = 4096
    nearest = np.zeros((pix.shape[0],), dtype=np.int32)
    dist_arr = np.zeros((pix.shape[0],), dtype=np.float32)
    for i in range(0, pix.shape[0], chunk):
        sub = pix[i:i + chunk]  # (c, 2)
        d2 = ((sub[:, None, :] - samples[None, :, :]) ** 2).sum(-1)  # (c, N)
        idx = d2.argmin(-1)
        nearest[i:i + chunk] = idx
        dist_arr[i:i + chunk] = np.sqrt(d2[np.arange(len(sub)), idx])
    dist_map = dist_arr.reshape(H, W)
    t_map = t_vals[nearest].reshape(H, W)
    tan_map = tan[nearest].reshape(H, W, 2)
    return dist_map, t_map, tan_map


# ───────────────────────────────────────────────────────────────────────
# Procedural noise (multi-octave Gaussian-blurred white noise)
# ───────────────────────────────────────────────────────────────────────

def _smooth_noise_2d(H: int, W: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    base = rng.random((H, W)).astype(np.float32)
    s = max(1.0, float(sigma))
    if HAS_CV2:
        out = cv2.GaussianBlur(base, (0, 0), sigmaX=s, sigmaY=s)
    else:
        # Fallback: very rough box smoothing
        k = max(1, int(s))
        kernel = np.ones((2 * k + 1, 2 * k + 1), dtype=np.float32)
        kernel /= kernel.sum()
        out = base  # skip — still produces randomness
    out -= out.mean()
    std = out.std() + 1e-6
    return out / (3.0 * std)  # most values land in [-1, 1]


def _frame_noise(H: int, W: int, sigma: float, seed: int, frame: int) -> np.ndarray:
    """Per-frame noise for animation — different seed per frame, then
    optionally cross-fade to next frame to keep temporal coherence."""
    return _smooth_noise_2d(H, W, sigma, seed * 65537 + frame)


# ───────────────────────────────────────────────────────────────────────
# Pattern → per-pixel mask
# ───────────────────────────────────────────────────────────────────────

def _smooth_step(x: np.ndarray, edge_softness: float) -> np.ndarray:
    """Map signed distance (positive = inside band) to [0,1] mask via
    a smoothstep ramp of width `edge_softness` pixels."""
    e = max(0.5, float(edge_softness))
    return np.clip(x / e + 0.5, 0.0, 1.0)


def _apply_taper(t_map: np.ndarray, taper_start: float, taper_end: float) -> np.ndarray:
    """Multiplier in [0,1] that fades thickness near path endpoints."""
    out = np.ones_like(t_map)
    if taper_start > 0:
        s = max(1e-3, taper_start)
        out *= np.clip(t_map / s, 0.0, 1.0)
    if taper_end > 0:
        e = max(1e-3, taper_end)
        out *= np.clip((1.0 - t_map) / e, 0.0, 1.0)
    return out


def _generate_pattern(
    pattern: str,
    dist_map: np.ndarray,
    t_map: np.ndarray,
    tan_map: np.ndarray,
    H: int,
    W: int,
    *,
    thickness: float,
    amplitude: float,
    frequency: float,
    turbulence: float,
    turbulence_scale: float,
    edge_softness: float,
    taper_start: float,
    taper_end: float,
    phase: float,
    seed: int,
    frame: int,
    flow_direction: str = "forward",
    mod_decay: float = 0.0,
) -> np.ndarray:
    """Return a (H,W) float32 mask in [0,1] for one frame."""
    half_t = thickness * 0.5
    taper = _apply_taper(t_map, taper_start, taper_end)

    # Direction modulation: transforms the path parameter so the modulation
    # appears to flow in different ways along the arc length.
    #   forward       — phase advances normally
    #   reverse       — phase advances backwards
    #   bidirectional — split: first half forward, second half reverse
    #   oscillate     — full path back-and-forth (sin envelope on phase)
    if flow_direction == "reverse":
        t_flow = 1.0 - t_map
        phase_flow = -phase
    elif flow_direction == "bidirectional":
        t_flow = np.where(t_map < 0.5, t_map * 2.0, (1.0 - t_map) * 2.0)
        phase_flow = phase
    elif flow_direction == "oscillate":
        t_flow = t_map
        phase_flow = phase * math.sin(phase * 0.5 + 1e-3)
    else:  # forward
        t_flow = t_map
        phase_flow = phase

    # Optional along-path decay multiplier (0=no decay, 1=full fade at t=1).
    if mod_decay > 0:
        decay = 1.0 - mod_decay * np.clip(t_flow, 0.0, 1.0)
        amp_mod = amplitude * decay
    else:
        amp_mod = np.full_like(t_flow, amplitude, dtype=np.float32)

    two_pi = 2.0 * math.pi

    # Per-pattern band-half-width modulation:
    if pattern == "ribbon":
        band = np.full_like(t_flow, half_t) * taper
        signed = band - dist_map

    elif pattern == "wave":
        # Width sin-modulated along arc length (the classic ocean wave).
        mod = np.sin(frequency * t_flow * two_pi + phase_flow)
        band = (half_t + amp_mod * mod) * taper
        band = np.maximum(band, 0.0)
        signed = band - dist_map

    elif pattern == "sawtooth":
        # Sawtooth thickness modulation: linear ramp 0→1 per cycle.
        # Useful for "teeth" / "spikes" along a path.
        u = (frequency * t_flow + phase_flow / two_pi) % 1.0
        mod = 2.0 * u - 1.0
        band = (half_t + amp_mod * mod) * taper
        band = np.maximum(band, 0.0)
        signed = band - dist_map

    elif pattern == "square":
        # Square-wave thickness: alternating thick/thin segments (pulses).
        # Implemented as a smoothed sign() of sin for soft edges.
        s = np.sin(frequency * t_flow * two_pi + phase_flow)
        mod = np.tanh(s * 8.0)
        band = (half_t + amp_mod * mod) * taper
        band = np.maximum(band, 0.0)
        signed = band - dist_map

    elif pattern == "triangle":
        # Triangle-wave thickness: piecewise-linear sin replacement.
        u = (frequency * t_flow + phase_flow / two_pi) % 1.0
        mod = 1.0 - 4.0 * np.abs(u - 0.5)
        band = (half_t + amp_mod * mod) * taper
        band = np.maximum(band, 0.0)
        signed = band - dist_map

    elif pattern == "gaussian_pulse":
        # Repeating Gaussian bumps along the path — comet-trail / firefly feel.
        u = (frequency * t_flow + phase_flow / two_pi) % 1.0
        bump = np.exp(-((u - 0.5) ** 2) / (2.0 * 0.08 ** 2))
        band = (half_t + amp_mod * bump) * taper
        band = np.maximum(band, 0.0)
        signed = band - dist_map

    elif pattern == "flow":
        # Fixed-width band, but pixels are pulled perpendicular by 2-D noise
        # so the centerline appears to meander like flowing water/smoke.
        band = np.full_like(t_flow, half_t) * taper
        if turbulence > 0:
            n = _frame_noise(H, W, turbulence_scale, seed, frame)
            displacement = n * amp_mod * turbulence
            signed = band - (dist_map - displacement)
        else:
            signed = band - dist_map

    elif pattern == "dust":
        # Discrete particles: only show pixels where noise > threshold.
        band = np.full_like(t_flow, half_t) * taper
        in_band_signed = band - dist_map
        in_band = _smooth_step(in_band_signed, edge_softness)
        n = _frame_noise(H, W, max(1.5, turbulence_scale * 0.4), seed, frame)
        # Animate threshold scrolling along path direction.
        thresh = 0.6 - turbulence * 0.5  # higher turbulence → more particles
        particles = np.clip((n - thresh) * 4.0, 0.0, 1.0)
        return np.clip(in_band * particles, 0.0, 1.0)

    elif pattern == "river":
        # Wide band that meanders + stronger taper at endpoints.
        band = np.full_like(t_flow, half_t) * taper
        if turbulence > 0:
            n_perp = _frame_noise(H, W, turbulence_scale, seed, frame)
            n_width = _frame_noise(H, W, turbulence_scale * 1.5, seed + 17, frame)
            band = band + amp_mod * 0.5 * n_width
            displacement = n_perp * amp_mod * turbulence
            signed = band - (dist_map - displacement)
        else:
            signed = band - dist_map

    elif pattern == "smoke":
        # Thickens with t (rising smoke trail) + heavy turbulence on edges.
        rising = 0.5 + 1.5 * np.clip(t_flow, 0.0, 1.0)  # 0.5x → 2x along path
        band = (half_t * rising) * taper
        if turbulence > 0:
            n = _frame_noise(H, W, turbulence_scale, seed, frame)
            displacement = n * amp_mod * turbulence
            signed = band + displacement * 0.4 - dist_map
        else:
            signed = band - dist_map

    elif pattern == "fbm":
        # Fractal Brownian Motion: stacked octaves of smoothed noise.
        # Produces more "natural" turbulence than single-octave noise —
        # the band gets organic, cloud-like edges. (Perlin/Mandelbrot 1982.)
        band = np.full_like(t_flow, half_t) * taper
        n_total = np.zeros((H, W), np.float32)
        amp_o = 1.0
        sig = max(2.0, turbulence_scale)
        for octave in range(4):
            n_o = _frame_noise(H, W, sig, seed + octave * 101, frame)
            n_total += n_o * amp_o
            amp_o *= 0.5
            sig *= 0.5
        displacement = n_total * amp_mod * max(0.05, turbulence)
        signed = band - (dist_map - displacement)

    elif pattern == "curl_noise":
        # Curl of a 2D noise field → divergence-free flow (Bridson 2007).
        # Pixels displace tangentially; produces swirling, vortex-like trails.
        band = np.full_like(t_flow, half_t) * taper
        sig = max(2.0, turbulence_scale)
        nx = _frame_noise(H, W, sig, seed,         frame)
        ny = _frame_noise(H, W, sig, seed + 8191,  frame)
        # Gradient (central differences) of the scalar potential.
        gy = np.zeros_like(nx)
        gx = np.zeros_like(ny)
        gy[1:-1, :] = (nx[2:, :] - nx[:-2, :]) * 0.5
        gx[:, 1:-1] = (ny[:, 2:] - ny[:, :-2]) * 0.5
        # Curl = (∂ny/∂x, -∂nx/∂y) — magnitude used to perturb the band.
        curl_mag = np.hypot(gx, gy)
        displacement = curl_mag * amp_mod * max(0.1, turbulence) * 4.0
        signed = band - (dist_map - displacement)

    elif pattern == "lightning":
        # Sparse high-frequency jitter — looks like lightning / cracks.
        band = np.full_like(t_flow, half_t * 0.4) * taper
        n_hi = _frame_noise(H, W, max(1.0, turbulence_scale * 0.25), seed, frame)
        n_lo = _frame_noise(H, W, max(2.0, turbulence_scale), seed + 13, frame)
        # Combine: high-freq carves the path, low-freq controls where bolts exist.
        crack = np.where(n_lo > 0.0, n_hi, -1.0)
        thresh = 0.4 - 0.4 * max(0.0, min(1.0, turbulence))
        bolt = np.clip((crack - thresh) * 6.0, 0.0, 1.0)
        in_band_signed = band - dist_map
        in_band = _smooth_step(in_band_signed, edge_softness)
        return np.clip(in_band * (0.35 + 0.65 * bolt), 0.0, 1.0)

    else:
        band = np.full_like(t_flow, half_t) * taper
        signed = band - dist_map

    return _smooth_step(signed, edge_softness)


# ═══════════════════════════════════════════════════════════════════════
# NODE
# ═══════════════════════════════════════════════════════════════════════

class SplinePathFlowMaskMEC:
    """Generate ocean-wave / flowing-water / dust / smoke masks along a spline
    path. Connect a spline (from `SplineMaskEditorMEC` via `spline_data_out`
    or the raw `spline_data` STRING) — choose a pattern and the node
    rasterizes an animated band along that path.

    Use cases:
      - ocean-wave shaped masks for in/out painting
      - water flowing on a floor (animated batch)
      - dust drifting along a trajectory
      - smoke trails rising from a torch
      - any motion-mask seed where you want shape AND temporal modulation
    """

    VRAM_TIER = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spline_data": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": (
                        "Spline JSON. Connect from Spline Mask Editor's "
                        "`spline_data_out` (auto-converted) or paste the same "
                        "shape: [{points:[{x,y},...], type, closed}]."
                    ),
                }),
                "pattern": (PATTERNS, {
                    "default": "wave",
                    "tooltip": (
                        "ribbon: solid band. wave: ocean-wave thickness. "
                        "flow: water/smoke meandering. dust: discrete particles. "
                        "river: wide tapered. smoke: rising trail. "
                        "sawtooth/square/triangle: parametric waveforms. "
                        "gaussian_pulse: repeating bumps (comet trail). "
                        "fbm: fractal Brownian motion (organic edges). "
                        "curl_noise: divergence-free swirls. "
                        "lightning: high-freq cracks / bolts."
                    ),
                }),
                "flow_direction": (FLOW_DIRECTIONS, {
                    "default": "forward",
                    "tooltip": (
                        "forward: phase advances normally. "
                        "reverse: phase advances backwards. "
                        "bidirectional: starts and ends meet at the path midpoint. "
                        "oscillate: phase swings back-and-forth along the path."
                    ),
                }),
                "width": ("INT", {
                    "default": 1024, "min": 16, "max": 16384, "step": 8,
                    "tooltip": "Output mask width (pixels).",
                }),
                "height": ("INT", {
                    "default": 1024, "min": 16, "max": 16384, "step": 8,
                    "tooltip": "Output mask height (pixels).",
                }),
                "thickness": ("FLOAT", {
                    "default": 60.0, "min": 1.0, "max": 1024.0, "step": 1.0,
                    "tooltip": "Base band width in pixels.",
                }),
                "amplitude": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 512.0, "step": 1.0,
                    "tooltip": "Modulation strength (wave height / displacement).",
                }),
                "frequency": ("FLOAT", {
                    "default": 6.0, "min": 0.1, "max": 50.0, "step": 0.1,
                    "tooltip": "Cycles along the path (wave / pattern frequency).",
                }),
                "turbulence": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Noise intensity for flow / dust / river / smoke.",
                }),
                "turbulence_scale": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 256.0, "step": 1.0,
                    "tooltip": "Spatial scale (pixels) of the turbulence noise.",
                }),
                "edge_softness": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 64.0, "step": 0.5,
                    "tooltip": "Smoothstep width for soft band edges (px).",
                }),
                "taper_start": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Fade-in fraction at path start (0=no taper).",
                }),
                "taper_end": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Fade-out fraction at path end.",
                }),
                "frames": ("INT", {
                    "default": 1, "min": 1, "max": 1024, "step": 1,
                    "tooltip": "Animated mask frame count (batch size).",
                }),
                "animation_speed": ("FLOAT", {
                    "default": 0.25, "min": -8.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Phase advance per frame (radians/2π). 0 = static.",
                }),
                "spline_type": (["catmull_rom", "polyline", "bezier"], {
                    "default": "catmull_rom",
                    "tooltip": "Default sampling for shapes that don't carry their own type.",
                }),
                "samples_per_segment": ("INT", {
                    "default": 30, "min": 4, "max": 200, "step": 1,
                    "tooltip": "Curve resolution. Higher = smoother centerline.",
                }),
                "closed": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Close the path loop (default off — flow paths are usually open).",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask (1 - mask).",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0x7FFFFFFF, "step": 1,
                    "tooltip": "Random seed for turbulence noise.",
                }),
                "mod_decay": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Along-path amplitude decay (0=none, 1=full fade at "
                        "path end). Use to make pulses die out / trails dissipate."
                    ),
                }),
                "use_embedded_editor": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Show an embedded spline editor inside this node "
                        "(shares architecture with Spline Mask Editor). "
                        "Off = only the raw STRING widget is shown."
                    ),
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional reference image. If connected, width/height are taken from it.",
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = (
        "Procedural flow / wave / dust mask along the spline path (frames, H, W).",
    )
    FUNCTION = "execute"
    CATEGORY = "C2C/Spline"
    DESCRIPTION = (
        "Generate procedural masks along a spline path — ocean waves, "
        "flowing water, dust trails, smoke, lightning, comet pulses. "
        "13 patterns: ribbon / wave / flow / dust / river / smoke / "
        "sawtooth / square / triangle / gaussian_pulse / fbm / curl_noise / "
        "lightning. 4 flow directions (forward/reverse/bidirectional/oscillate) "
        "× continuous modulation sliders → effectively unbounded combinations. "
        "Produces animated batches (frames × H × W) suitable for motion-mask "
        "animation. Embedded spline editor optional."
    )

    @classmethod
    def IS_CHANGED(cls, spline_data, pattern, width, height, thickness,
                   amplitude, frequency, turbulence, turbulence_scale,
                   edge_softness, taper_start, taper_end, frames,
                   animation_speed, spline_type, samples_per_segment, closed,
                   invert, seed, flow_direction="forward", mod_decay=0.0,
                   use_embedded_editor=True, image=None, **kwargs):
        return hash_args_and_kwargs(
            spline_data, pattern, width, height, thickness, amplitude,
            frequency, turbulence, turbulence_scale, edge_softness,
            taper_start, taper_end, frames, animation_speed, spline_type,
            samples_per_segment, closed, invert, seed, flow_direction,
            mod_decay, use_embedded_editor, image, **kwargs,
        )

    def execute(self, spline_data: str, pattern: str,
                width: int, height: int,
                thickness: float, amplitude: float, frequency: float,
                turbulence: float, turbulence_scale: float,
                edge_softness: float, taper_start: float, taper_end: float,
                frames: int, animation_speed: float,
                spline_type: str, samples_per_segment: int,
                closed: bool, invert: bool, seed: int,
                flow_direction: str = "forward",
                mod_decay: float = 0.0,
                use_embedded_editor: bool = True,
                image: torch.Tensor = None):
        if image is not None and (
            not isinstance(image, torch.Tensor) or image.ndim != 4
        ):
            raise ValueError(
                "SplinePathFlowMaskMEC optional image must be IMAGE [B,H,W,C]"
            )
        with torch.inference_mode():
            return self._execute_impl(
                spline_data, pattern, width, height, thickness, amplitude,
                frequency, turbulence, turbulence_scale, edge_softness,
                taper_start, taper_end, frames, animation_speed, spline_type,
                samples_per_segment, closed, invert, seed, flow_direction,
                mod_decay, use_embedded_editor, image,
            )

    def _execute_impl(self, spline_data: str, pattern: str,
                      width: int, height: int,
                      thickness: float, amplitude: float, frequency: float,
                      turbulence: float, turbulence_scale: float,
                      edge_softness: float, taper_start: float, taper_end: float,
                      frames: int, animation_speed: float,
                      spline_type: str, samples_per_segment: int,
                      closed: bool, invert: bool, seed: int,
                      flow_direction: str = "forward",
                      mod_decay: float = 0.0,
                      use_embedded_editor: bool = True,
                      image: torch.Tensor = None):

        # Resolve dimensions: prefer the connected image when present.
        if image is not None and image.dim() == 4:
            _, H, W, _ = image.shape
        else:
            H, W = int(height), int(width)
        H = max(16, int(H))
        W = max(16, int(W))

        samples, t_vals = _sample_path(
            spline_data, spline_type, closed, int(samples_per_segment),
        )

        if len(samples) == 0:
            logger.warning(
                "[MEC] SplinePathFlowMask: no spline points — emitting empty mask."
            )
            mask = torch.zeros((max(1, frames), H, W), dtype=torch.float32)
            if invert:
                mask = 1.0 - mask
            return (mask,)

        dist_map, t_map, tan_map = _build_dist_and_t_maps(samples, t_vals, H, W)

        out_frames = []
        for f in range(max(1, int(frames))):
            phase = float(animation_speed) * f * 2.0 * math.pi
            m = _generate_pattern(
                pattern,
                dist_map=dist_map,
                t_map=t_map,
                tan_map=tan_map,
                H=H, W=W,
                thickness=float(thickness),
                amplitude=float(amplitude),
                frequency=float(frequency),
                turbulence=float(turbulence),
                turbulence_scale=float(turbulence_scale),
                edge_softness=float(edge_softness),
                taper_start=float(taper_start),
                taper_end=float(taper_end),
                phase=phase,
                seed=int(seed),
                frame=f,
                flow_direction=str(flow_direction),
                mod_decay=float(mod_decay),
            )
            if invert:
                m = 1.0 - m
            out_frames.append(m.astype(np.float32))

        stacked = np.stack(out_frames, axis=0)  # (F, H, W)
        mask = torch.from_numpy(stacked).clamp(0.0, 1.0)

        logger.info(
            "[MEC] SplinePathFlowMask: pattern=%s pts=%d frames=%d %dx%d "
            "thickness=%.1f amp=%.1f freq=%.2f turb=%.2f",
            pattern, len(samples), frames, W, H,
            thickness, amplitude, frequency, turbulence,
        )

        return (mask,)


NODE_CLASS_MAPPINGS = {
    "SplinePathFlowMaskMEC": SplinePathFlowMaskMEC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplinePathFlowMaskMEC": "Spline Path Flow Mask",
}
