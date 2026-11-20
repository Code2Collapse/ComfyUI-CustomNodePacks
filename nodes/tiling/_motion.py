"""Small measuring helpers shared by the tiling nodes (L9.172, A25): luma, detail energy, FFT phase correlation.

Written fresh for this Apache-2.0 pack (textbook phase correlation: the normalised cross-power spectrum of two frames
has a single peak at their relative translation). torch only; no model, no download.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def luma(x: torch.Tensor) -> torch.Tensor:
    """[..., H, W, C] (ComfyUI IMAGE layout) -> [..., H, W] float32 luma (Rec.601 weights; mean for 1-2 channels)."""
    t = x.float()
    if t.shape[-1] >= 3:
        return 0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2]
    return t.mean(dim=-1)


def gradient_magnitude(l: torch.Tensor) -> torch.Tensor:
    """|grad| of a [B, H, W] luma with forward differences (replicated at the far edge)."""
    gx = torch.zeros_like(l)
    gy = torch.zeros_like(l)
    if l.shape[-1] > 1:
        gx[..., :, :-1] = l[..., :, 1:] - l[..., :, :-1]
        gx[..., :, -1] = gx[..., :, -2]
    if l.shape[-2] > 1:
        gy[..., :-1, :] = l[..., 1:, :] - l[..., :-1, :]
        gy[..., -1, :] = gy[..., -2, :]
    return torch.sqrt(gx * gx + gy * gy)


def box_mean(l: torch.Tensor, size: int = 7) -> torch.Tensor:
    """Local mean of a [B, H, W] map over a size x size box (edges average only the pixels that exist)."""
    pad = size // 2
    return F.avg_pool2d(l.unsqueeze(1), size, stride=1, padding=pad, count_include_pad=False).squeeze(1)


def _hann2d(h: int, w: int, device) -> torch.Tensor:
    wy = torch.hann_window(h, periodic=False, device=device) if h > 1 else torch.ones(1, device=device)
    wx = torch.hann_window(w, periodic=False, device=device) if w > 1 else torch.ones(1, device=device)
    return wy[:, None] * wx[None, :]


def _subpixel(c_m: float, c_0: float, c_p: float) -> float:
    """Parabolic refinement of a peak at 0 from its two neighbours (-0.5..0.5)."""
    den = c_m - 2.0 * c_0 + c_p
    if abs(den) < 1e-12:
        return 0.0
    return max(-0.5, min(0.5, 0.5 * (c_m - c_p) / den))


def phase_correlate(ref: torch.Tensor, mov: torch.Tensor, *, window: bool = True) -> tuple[float, float]:
    """Translation (dy, dx) in pixels such that `mov` ~= `ref` shifted by (dy, dx) (torch.roll convention).

    ref / mov: [H, W] float. A Hann window keeps the frame border from dominating the spectrum.
    """
    a = ref.float()
    b = mov.float()
    h, w = int(a.shape[-2]), int(a.shape[-1])
    if h < 2 or w < 2:
        return 0.0, 0.0
    a = a - a.mean()
    b = b - b.mean()
    if window:
        win = _hann2d(h, w, a.device)
        a = a * win
        b = b * win
    fa = torch.fft.fft2(a)
    fb = torch.fft.fft2(b)
    cross = torch.conj(fa) * fb
    cross = cross / cross.abs().clamp(min=1e-12)
    r = torch.fft.ifft2(cross).real
    idx = int(torch.argmax(r))
    py, px = divmod(idx, w)

    def at(y, x):
        return float(r[y % h, x % w])

    dy = py + _subpixel(at(py - 1, px), at(py, px), at(py + 1, px))
    dx = px + _subpixel(at(py, px - 1), at(py, px), at(py, px + 1))
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dy), float(dx)


def downscale_factor(h: int, w: int, max_side: int = 512) -> int:
    return max(1, int(math.ceil(max(h, w) / float(max_side))))


def max_global_shift(frames_bhw: torch.Tensor, *, max_side: int = 512) -> float:
    """The largest per-axis translation between consecutive frames of a [B, H, W] luma clip, in full-size pixels."""
    if frames_bhw.shape[0] < 2:
        return 0.0
    f = downscale_factor(int(frames_bhw.shape[-2]), int(frames_bhw.shape[-1]), max_side)
    small = frames_bhw.float()
    if f > 1:
        small = F.avg_pool2d(small.unsqueeze(1), f, stride=f).squeeze(1)
    worst = 0.0
    for t in range(1, int(small.shape[0])):
        dy, dx = phase_correlate(small[t - 1], small[t])
        worst = max(worst, abs(dy) * f, abs(dx) * f)
    return worst


def shift_mask(m: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """[H, W] mask moved by integer (dy, dx); what leaves the frame is dropped, what enters is 0 (no wrap)."""
    out = torch.zeros_like(m)
    h, w = int(m.shape[-2]), int(m.shape[-1])
    dy, dx = int(dy), int(dx)
    if abs(dy) >= h or abs(dx) >= w:
        return out
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[..., yd, xd] = m[..., ys, xs]
    return out


def bbox(m: torch.Tensor, thr: float = 0.5):
    """(y0, y1, x0, x1) of m > thr, end-exclusive; None when empty."""
    on = m > thr
    if not bool(on.any()):
        return None
    rows = torch.nonzero(on.any(dim=-1)).flatten()
    cols = torch.nonzero(on.any(dim=-2)).flatten()
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def pad_box(box, h: int, w: int, frac: float = 0.25):
    y0, y1, x0, x1 = box
    py = int(math.ceil((y1 - y0) * frac))
    px = int(math.ceil((x1 - x0) * frac))
    return max(0, y0 - py), min(h, y1 + py), max(0, x0 - px), min(w, x1 + px)
