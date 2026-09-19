# Torch mask ops, ported from ComfyUI_LayerStyle by chflame163 (MIT,
# https://github.com/chflame163/ComfyUI_LayerStyle), py/imagefunc.py.
# No PIL in the hot path; batch-correct [B,H,W] / [B,H,W,C].
# WHY no_grad AND NOT inference_mode: inference_mode marks the tensors it
# produces as "inference tensors", and an inference tensor cannot be mutated
# in place outside inference mode. ComfyUI hands a node's output straight to
# the next node, and plenty of nodes - core ones included - do in-place work on
# an IMAGE or MASK they were given. Those blow up with
# "Inplace update to inference tensor outside InferenceMode is not allowed",
# which names nothing the user can act on and points at the wrong node.
# no_grad skips the autograd graph just the same, without the version-counter
# restriction. Pinned by test_no_node_returns_an_inference_tensor.

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from ..layer_effects._ops import (
    align_batch,
    clamp_with_report,
    clone_output,
    ensure_bhw4_image,
    ensure_bhw_mask,
    frame_limit,
    morph_grow,
    parse_hex_color,
    shift_mask,
)

BLEND_EPS: float = 1e-7


def build_report(node_name: str, batch: int, notes: list[str], coverage: float | None = None) -> str:
    parts = [f"{node_name} processed {batch} frame(s)."]
    if coverage is not None:
        parts.append(f"Coverage {coverage:.1f}%.")
    if notes:
        parts.append(" ".join(notes))
    return " ".join(parts)


def _cross_dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(mask.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)


def _cross_erode(mask: torch.Tensor) -> torch.Tensor:
    return 1.0 - _cross_dilate(1.0 - mask)


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """rgb [...,3] in 0-1 -> hsv same shape, h in 0-1."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = torch.max(rgb, dim=-1).values
    minc = torch.min(rgb, dim=-1).values
    v = maxc
    delt = maxc - minc
    s = torch.where(maxc > BLEND_EPS, delt / maxc.clamp(min=BLEND_EPS), torch.zeros_like(maxc))
    rc = (maxc - r) / delt.clamp(min=BLEND_EPS)
    gc = (maxc - g) / delt.clamp(min=BLEND_EPS)
    bc = (maxc - b) / delt.clamp(min=BLEND_EPS)
    h = torch.zeros_like(maxc)
    h = torch.where((maxc == r) & (delt > BLEND_EPS), (bc - gc) % 6.0, h)
    h = torch.where((maxc == g) & (delt > BLEND_EPS), 2.0 + rc - bc, h)
    h = torch.where((maxc == b) & (delt > BLEND_EPS), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return torch.stack([h, s, v], dim=-1)


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """rgb [...,3] in 0-1 -> Lab with L in 0-1, a/b roughly -1..1."""
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    def _f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > 0.008856, t.pow(1.0 / 3.0), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = _f(x / 0.95047), _f(y), _f(z / 1.08883)
    l = (116.0 * fy - 16.0) / 100.0
    a = (500.0 * (fx - fy)) / 128.0
    bch = (200.0 * (fy - fz)) / 128.0
    return torch.stack([l.clamp(0.0, 1.0), a, bch], dim=-1)


def extract_channel(image: torch.Tensor, channel: str) -> torch.Tensor:
    """Return scalar field [B,H,W] in roughly 0-1."""
    ch = channel.lower()
    if ch == "luma":
        return (
            0.2126 * image[..., 0]
            + 0.7152 * image[..., 1]
            + 0.0722 * image[..., 2]
        ).clamp(0.0, 1.0)
    if ch in ("red", "r"):
        return image[..., 0].clamp(0.0, 1.0)
    if ch in ("green", "g"):
        return image[..., 1].clamp(0.0, 1.0)
    if ch in ("blue", "b"):
        return image[..., 2].clamp(0.0, 1.0)
    hsv = _rgb_to_hsv(image.clamp(0.0, 1.0))
    if ch in ("hue", "h"):
        return hsv[..., 0]
    if ch in ("saturation", "s", "sat"):
        return hsv[..., 1]
    if ch in ("value", "v", "val"):
        return hsv[..., 2]
    lab = _rgb_to_lab(image.clamp(0.0, 1.0))
    if ch in ("l",):
        return lab[..., 0]
    if ch in ("a",):
        return ((lab[..., 1] + 1.0) * 0.5).clamp(0.0, 1.0)
    if ch in ("b", "lab_b"):
        return ((lab[..., 2] + 1.0) * 0.5).clamp(0.0, 1.0)
    raise ValueError(f"Unknown channel {channel!r}.")


def mask_from_color(
    image: torch.Tensor,
    color_hex: str,
    colorspace: str,
    tolerance: float,
    soft_falloff: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-pixel match mask [B,H,W] with soft band outside tolerance."""
    tr, tg, tb = parse_hex_color(color_hex)
    target = torch.tensor([tr, tg, tb], device=device, dtype=dtype)
    tol = max(0.0, float(tolerance)) / 100.0
    soft = max(0.0, float(soft_falloff)) / 100.0
    cs = colorspace.lower()
    if cs == "rgb":
        diff = (image[..., :3] - target.view(1, 1, 1, 3)).abs().max(dim=-1).values
        scale = math.sqrt(3.0)
    elif cs == "hsv":
        targ_hsv = _rgb_to_hsv(target.view(1, 1, 1, 3)).squeeze(0).squeeze(0)
        src_hsv = _rgb_to_hsv(image[..., :3])
        dh = (src_hsv[..., 0] - targ_hsv[0]).abs()
        dh = torch.min(dh, 1.0 - dh)
        ds = (src_hsv[..., 1] - targ_hsv[1]).abs()
        dv = (src_hsv[..., 2] - targ_hsv[2]).abs()
        diff = torch.max(torch.max(dh * 0.5, ds), dv)
        scale = 1.0
    elif cs == "lab":
        targ_lab = _rgb_to_lab(target.view(1, 1, 1, 3)).squeeze(0).squeeze(0)
        src_lab = _rgb_to_lab(image[..., :3])
        diff = (src_lab - targ_lab.view(1, 1, 3)).pow(2).sum(dim=-1).sqrt() / math.sqrt(3.0)
        scale = 1.0
    else:
        raise ValueError(f"Unknown colorspace {colorspace!r}; use rgb, hsv, or lab.")
    dist = diff / max(scale * tol, BLEND_EPS) if tol > BLEND_EPS else diff
    if soft <= BLEND_EPS:
        return (dist <= 1.0).float()
    inner = dist <= 1.0
    outer = dist <= 1.0 + soft
    ramp = (1.0 + soft - dist) / soft
    return torch.where(inner, torch.ones_like(dist), torch.where(outer, ramp.clamp(0.0, 1.0), torch.zeros_like(dist)))


def _ramp_1d(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(0.0, 1.0)


def gradient_mask_bhw(
    batch: int,
    height: int,
    width: int,
    gradient_type: str,
    angle_deg: float,
    center_x: float,
    center_y: float,
    start: float,
    end: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gtype = gradient_type.lower()
    if gtype == "linear":
        rad = math.radians(float(angle_deg))
        cx, cy = math.cos(rad), math.sin(rad)
        t = (xx - 0.5) * cx + (yy - 0.5) * cy
        t = (t - t.min()) / (t.max() - t.min() + BLEND_EPS)
    elif gtype == "radial":
        dx = xx - float(center_x)
        dy = yy - float(center_y)
        t = torch.sqrt(dx * dx + dy * dy)
        t = t / (t.max() + BLEND_EPS)
    elif gtype == "angular":
        dx = xx - float(center_x)
        dy = yy - float(center_y)
        ang = torch.atan2(dy, dx) / (2.0 * math.pi) + 0.5
        offset = float(angle_deg) / 360.0
        t = (ang + offset) % 1.0
    else:
        raise ValueError(f"Unknown gradient_type {gradient_type!r}.")
    ramp = _ramp_1d(t)
    lo, hi = float(start), float(end)
    if lo > hi:
        lo, hi = hi, lo
    out = lo + ramp * (hi - lo)
    return out.unsqueeze(0).expand(batch, height, width).clone()


def mask_grain_bhw(
    mask: torch.Tensor,
    amount: int,
    seed: int,
    grain_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if amount <= 0:
        return mask.clone()
    b, h, w = mask.shape
    gsize = max(1, int(grain_size))
    gh = max(1, h // gsize)
    gw = max(1, w // gsize)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) & 0x7FFFFFFF)
    noise_lo = torch.rand(b, gh, gw, generator=gen, device="cpu", dtype=torch.float32)
    noise_hi = torch.rand(b, gh, gw, generator=gen, device="cpu", dtype=torch.float32)
    noise = (noise_lo + noise_hi) * 0.5
    noise = noise.to(device=device, dtype=dtype)
    noise = F.interpolate(noise.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
    amp = float(amount) / 127.0
    out = (mask + (noise - 0.5) * amp).clamp(0.0, 1.0)
    return out.clone()


def _line_blur_single(mask_hw: torch.Tensor, angle_deg: float, dist: int) -> torch.Tensor:
    """O(dist) line blur via shifted copies — one tap per pixel along the line."""
    ang = math.radians(float(angle_deg))
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)
    n_taps = 2 * dist + 1
    accum = torch.zeros_like(mask_hw)
    for i in range(-dist, dist + 1):
        dx = int(round(i * cos_a))
        dy = int(round(i * sin_a))
        accum = accum + shift_mask(mask_hw, dx, dy)
    return (accum / float(n_taps)).clamp(0.0, 1.0)


def motion_blur_mask(
    mask: torch.Tensor,
    angle_deg: float,
    distance: int,
    notes: list[str],
) -> torch.Tensor:
    h, w = mask.shape[-2], mask.shape[-1]
    limit = frame_limit(h, w)
    dist = clamp_with_report(int(distance), limit, "distance", notes)
    if dist <= 0:
        return mask.clone()
    if mask.ndim == 2:
        return _line_blur_single(mask, angle_deg, dist).clone()
    outs = [_line_blur_single(mask[i], angle_deg, dist) for i in range(mask.shape[0])]
    return torch.stack(outs, dim=0).clone()


def _dilate_colors(
    color: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """color BCHW, weight B1HW — spread known colours to neighbours."""
    c_nb = F.max_pool2d(color, kernel_size=3, stride=1, padding=1)
    w_nb = F.max_pool2d(weight, kernel_size=3, stride=1, padding=1)
    new_w = torch.clamp(weight + (w_nb - weight).clamp(min=0), 0.0, 1.0)
    fill = w_nb > weight
    new_c = torch.where(fill, c_nb, color)
    return new_c, new_w


def edge_spread_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    spread: int,
    notes: list[str],
) -> torch.Tensor:
    """Push interior plate colour outward under the matte edge."""
    b, h, w, c = image.shape
    limit = frame_limit(h, w)
    steps = clamp_with_report(int(spread), limit, "spread", notes)
    m = mask.clamp(0.0, 1.0)
    interior = _cross_erode(m)
    if steps > 0:
        zone = morph_grow(interior, steps, notes, "spread")
    else:
        zone = interior.clone()
    img = image.permute(0, 3, 1, 2)
    color = img * interior.unsqueeze(1)
    weight = interior.unsqueeze(1)
    for _ in range(steps):
        color, weight = _dilate_colors(color, weight)
    spread_rgb = color.permute(0, 2, 3, 1)
    use = (zone.unsqueeze(-1) > 0.01) & (weight.permute(0, 2, 3, 1) > BLEND_EPS)
    out = torch.where(use, spread_rgb, image)
    return out.clone()


def resolve_size(
    width: int,
    height: int,
    size_as: torch.Tensor | None,
    notes: list[str],
) -> tuple[int, int]:
    if size_as is not None:
        img = ensure_bhw4_image(size_as, "MaskGradientMEC")
        h, w = int(img.shape[1]), int(img.shape[2])
        notes.append(
            f"size taken from the connected image ({w}x{h}), width/height ignored"
        )
        return w, h
    return int(width), int(height)


def align_mask_batch(
    tensors: Sequence[torch.Tensor | None],
) -> tuple[list[torch.Tensor | None], int]:
    return align_batch(tensors)
