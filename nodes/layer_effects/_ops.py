# Image ops ported from ComfyUI_LayerStyle by chflame163 (MIT,
# https://github.com/chflame163/ComfyUI_LayerStyle), py/imagefunc.py:
# shift_image, expand_mask, gradient and the composite path.
#
# Torch, not PIL: upstream converts tensor -> PIL -> numpy -> PIL ->
# tensor for EVERY frame, so a 100-frame comp pays that round trip a
# hundred times and never touches the GPU the tensor is already on.
"""Torch image ops for layer effects — no PIL in hot path."""
from __future__ import annotations

import math
import re
from typing import Sequence

import torch
import torch.nn.functional as F

from ._blend import blend_rgba

BLEND_EPS: float = 1e-7

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def parse_hex_color(value: str) -> tuple[float, float, float]:
    s = (value or "").strip()
    if not s.startswith("#"):
        s = "#" + s
    if len(s) == 4:
        s = "#" + "".join(ch * 2 for ch in s[1:])
    if not _HEX_RE.match(s):
        raise ValueError(f"Invalid hex colour {value!r}; use #RRGGBB or #RGB.")
    r = int(s[1:3], 16) / 255.0
    g = int(s[3:5], 16) / 255.0
    b = int(s[5:7], 16) / 255.0
    return (r, g, b)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def step_value(start: float, end: float, total_step: int, step: int) -> float:
    if total_step <= 0:
        return end
    factor = step / total_step
    return (end - start) * factor + start


def step_color_hex(start_hex: str, end_hex: str, total_step: int, step: int) -> str:
    sr, sg, sb = parse_hex_color(start_hex)
    er, eg, eb = parse_hex_color(end_hex)
    return rgb_to_hex((
        int(step_value(sr * 255, er * 255, total_step, step)),
        int(step_value(sg * 255, eg * 255, total_step, step)),
        int(step_value(sb * 255, eb * 255, total_step, step)),
    ))


def frame_limit(height: int, width: int) -> int:
    return max(height, width)


def clamp_with_report(
    requested: int,
    limit: int,
    label: str,
    notes: list[str],
) -> int:
    req = int(requested)
    if req < 0:
        req = 0
    if req > limit:
        notes.append(
            f"{label} {requested} exceeded the frame; clamped to {limit} "
            f"(no visible difference beyond the frame size)"
        )
        return limit
    return req


def align_batch(
    tensors: Sequence[torch.Tensor | None],
) -> tuple[list[torch.Tensor | None], int]:
    present = [t for t in tensors if t is not None]
    if not present:
        raise ValueError("No tensors supplied to align_batch.")
    b = max(t.shape[0] for t in present)
    out: list[torch.Tensor | None] = []
    for t in tensors:
        if t is None:
            out.append(None)
            continue
        if t.shape[0] == b:
            out.append(t.clone())
        elif t.shape[0] == 1:
            out.append(t.expand(b, *t.shape[1:]).clone())
        elif t.shape[0] < b:
            pad = t[-1:].expand(b - t.shape[0], *t.shape[1:])
            out.append(torch.cat([t, pad], dim=0).clone())
        else:
            out.append(t[:b].clone())
    return out, b


def ensure_bhw4_image(image: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(f"{label} expects IMAGE tensor [B,H,W,C].")
    return image


def ensure_bhw_mask(mask: torch.Tensor | None, label: str) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"{label} expects MASK tensor [H,W] or [B,H,W].")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"{label} expects MASK tensor [H,W] or [B,H,W].")
    return mask


def image_to_rgba(image: torch.Tensor) -> torch.Tensor:
    if image.shape[-1] == 4:
        return image.clone()
    alpha = torch.ones(
        (*image.shape[:-1], 1),
        device=image.device,
        dtype=image.dtype,
    )
    return torch.cat([image, alpha], dim=-1).clone()


def resolve_layer_mask(
    layer: torch.Tensor,
    layer_mask: torch.Tensor | None,
    invert_mask: bool,
    *,
    node_name: str,
    notes: list[str],
) -> torch.Tensor:
    b, h, w, _ = layer.shape
    mask: torch.Tensor | None = None
    if layer_mask is not None:
        (layer_mask_aligned,), _ = align_batch([layer_mask])
        layer_mask = layer_mask_aligned
        mask = layer_mask.clone()
    elif layer.shape[-1] >= 4:
        mask = layer[..., 3].clone()
    if mask is None:
        raise ValueError(
            f"{node_name} needs a layer mask: connect layer_mask or use an RGBA layer_image "
            f"with an alpha channel."
        )
    if mask.shape[-2] != h or mask.shape[-1] != w:
        old_h, old_w = int(mask.shape[-2]), int(mask.shape[-1])
        mask = F.interpolate(
            mask.unsqueeze(1),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        notes.append(f"mask resized {old_w}x{old_h} -> {w}x{h}")
    if invert_mask:
        mask = 1.0 - mask
    return mask.clamp(0.0, 1.0).clone()


def shift_mask(mask: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    out = torch.zeros_like(mask)
    h, w = mask.shape[-2], mask.shape[-1]
    sx0 = max(0, -dx)
    sx1 = min(w, w - dx) if dx >= 0 else w
    sy0 = max(0, -dy)
    sy1 = min(h, h - dy) if dy >= 0 else h
    dx0 = max(0, dx)
    dy0 = max(0, dy)
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)
    if sx1 > sx0 and sy1 > sy0:
        out[..., dy0:dy1, dx0:dx1] = mask[..., sy0:sy1, sx0:sx1]
    return out.clone()


def _cross_dilate(mask: torch.Tensor) -> torch.Tensor:
    x = mask.unsqueeze(1)
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1).squeeze(1)


def _cross_erode(mask: torch.Tensor) -> torch.Tensor:
    return 1.0 - _cross_dilate(1.0 - mask)


def morph_grow(mask: torch.Tensor, grow: int, notes: list[str], label: str) -> torch.Tensor:
    h, w = mask.shape[-2], mask.shape[-1]
    limit = frame_limit(h, w)
    steps = clamp_with_report(abs(int(grow)), limit, label, notes)
    out = mask.clone()
    if grow == 0 or steps == 0:
        return out
    for _ in range(steps):
        out = _cross_dilate(out) if grow > 0 else _cross_erode(out)
    return out.clamp(0.0, 1.0).clone()


def _pad_for_conv(
    x: torch.Tensor,
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
) -> torch.Tensor:
    """Pad BCHW tensor; clamp pad and fall back to replicate when reflect is invalid."""
    _, _, h, w = x.shape
    pl = max(0, int(pad_left))
    pr = max(0, int(pad_right))
    pt = max(0, int(pad_top))
    pb = max(0, int(pad_bottom))
    mode = "reflect"
    if w <= 1 or pl >= w or pr >= w or (pl + pr) >= w:
        pl = min(pl, max(w - 1, 0))
        pr = min(pr, max(w - 1, 0))
        mode = "replicate"
    if h <= 1 or pt >= h or pb >= h or (pt + pb) >= h:
        pt = min(pt, max(h - 1, 0))
        pb = min(pb, max(h - 1, 0))
        mode = "replicate"
    return F.pad(x, (pl, pr, pt, pb), mode=mode)


def _box_blur_1d(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = 2 * radius + 1
    weight_h = torch.full((1, 1, 1, k), 1.0 / k, device=x.device, dtype=x.dtype)
    weight_v = torch.full((1, 1, k, 1), 1.0 / k, device=x.device, dtype=x.dtype)
    x = x.unsqueeze(1)
    x = _pad_for_conv(x, radius, radius, 0, 0)
    x = F.conv2d(x, weight_h)
    x = _pad_for_conv(x, 0, 0, radius, radius)
    x = F.conv2d(x, weight_v)
    return x.squeeze(1)


def _separable_gaussian_blur(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.clone()
    k = 2 * radius + 1
    sigma = max(radius / 2.0, BLEND_EPS)
    coords = torch.arange(k, device=mask.device, dtype=mask.dtype) - radius
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = (g / g.sum())
    weight_h = g.view(1, 1, 1, k)
    weight_v = g.view(1, 1, k, 1)
    x = mask.unsqueeze(1)
    x = _pad_for_conv(x, radius, radius, 0, 0)
    x = F.conv2d(x, weight_h)
    x = _pad_for_conv(x, 0, 0, radius, radius)
    x = F.conv2d(x, weight_v)
    return x.squeeze(1).clone()


def _box_blur_approx(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.clone()
    sigma = radius / 2.0
    w_ideal = math.sqrt((12 * sigma * sigma / 3) + 1)
    wl = int(round(w_ideal))
    if wl % 2 == 0:
        wl += 1
    r = (wl - 1) // 2
    out = mask.clone()
    for _ in range(3):
        out = _box_blur_1d(out, r)
    return out


def blur_mask(mask: torch.Tensor, blur: int, notes: list[str]) -> torch.Tensor:
    h, w = mask.shape[-2], mask.shape[-1]
    limit = frame_limit(h, w)
    eff = clamp_with_report(int(blur), limit, "blur", notes)
    if eff <= 0:
        return mask.clone()
    if eff <= 8:
        return _separable_gaussian_blur(mask, eff)
    return _box_blur_approx(mask, eff)


def expand_mask(
    mask: torch.Tensor,
    grow: int,
    blur: int,
    notes: list[str],
    *,
    grow_label: str = "grow",
) -> torch.Tensor:
    grown = morph_grow(mask, grow, notes, grow_label)
    return blur_mask(grown, blur, notes)


def subtract_masks(outer: torch.Tensor, inner: torch.Tensor) -> torch.Tensor:
    return (outer - inner).clamp(0.0, 1.0).clone()


def outer_band_footprint(
    mask: torch.Tensor,
    grow: int,
    blur: int,
    notes: list[str],
    *,
    grow_label: str = "grow",
) -> torch.Tensor:
    """dilate(mask, grow) minus mask — the band outside the subject, blurred."""
    expanded = expand_mask(mask, grow, blur, notes, grow_label=grow_label)
    return subtract_masks(expanded, mask)


def inner_band_footprint(
    mask: torch.Tensor,
    grow: int,
    blur: int,
    notes: list[str],
    *,
    grow_label: str = "grow",
) -> torch.Tensor:
    """mask AND NOT erode(mask, grow) — the band inside the subject, blurred."""
    eroded = expand_mask(mask, -abs(int(grow)), blur, notes, grow_label=grow_label)
    return (mask * (1.0 - eroded)).clamp(0.0, 1.0).clone()


def drop_shadow_footprint(
    mask: torch.Tensor,
    distance_x: int,
    distance_y: int,
    grow: int,
    blur: int,
    notes: list[str],
) -> torch.Tensor:
    """Shifted/grown shadow minus the layer — visible shadow not hidden under the subject."""
    shifted = shift_mask(mask, int(distance_x), int(distance_y))
    shadow = expand_mask(shifted, grow, blur, notes)
    return subtract_masks(shadow, mask)


def inner_shadow_footprint(
    mask: torch.Tensor,
    distance_x: int,
    distance_y: int,
    grow: int,
    blur: int,
    notes: list[str],
) -> torch.Tensor:
    """Layer mask intersected with the shifted/grown shadow band."""
    shifted = shift_mask(mask, int(distance_x), int(distance_y))
    shadow = expand_mask(shifted, grow, blur, notes)
    return (mask * shadow).clamp(0.0, 1.0).clone()


def linear_gradient_bhwc(
    batch: int,
    height: int,
    width: int,
    start_rgb: tuple[float, float, float],
    end_rgb: tuple[float, float, float],
    start_alpha: float,
    end_alpha: float,
    angle_deg: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    rad = math.radians(float(angle_deg))
    cx, cy = math.cos(rad), math.sin(rad)
    t = ((xx * cx + yy * cy) + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    t = t.unsqueeze(0).unsqueeze(-1).expand(batch, height, width, 1)
    sr, sg, sb = start_rgb
    er, eg, eb = end_rgb
    start = torch.tensor([sr, sg, sb, start_alpha], device=device, dtype=dtype)
    end = torch.tensor([er, eg, eb, end_alpha], device=device, dtype=dtype)
    grad = start.view(1, 1, 1, 4) * (1.0 - t) + end.view(1, 1, 1, 4) * t
    return grad.expand(batch, height, width, 4).clone()


def gradient_ramp_report(
    start_color: str,
    end_color: str,
    *,
    start_alpha: int | None = None,
    end_alpha: int | None = None,
    mid_color: str | None = None,
    mid_point: float | None = None,
    angle: int | None = None,
) -> str:
    if mid_color is not None and mid_point is not None:
        ramp = f"ramp {start_color} -> {mid_color} @ {mid_point:.2f} -> {end_color}"
    else:
        ramp = f"ramp {start_color} -> {end_color}"
    parts = [ramp]
    if start_alpha is not None and end_alpha is not None:
        parts.append(f"alpha {start_alpha}..{end_alpha}")
    if angle is not None:
        parts.append(f"angle {angle}°")
    return ", ".join(parts)


def make_transparent_background(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.zeros((batch, height, width, 4), device=device, dtype=dtype).clone()


def rgba_to_rgb(rgba: torch.Tensor) -> torch.Tensor:
    return rgba[..., :3].clone()


def alpha_composite(
    canvas: torch.Tensor,
    layer_rgba: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    src_a = layer_rgba[..., 3:4] * m
    out = canvas.clone()
    rgb = layer_rgba[..., :3]
    out[..., :3] = canvas[..., :3] * (1.0 - src_a) + rgb * src_a
    out[..., 3] = torch.maximum(canvas[..., 3], src_a[..., 0])
    return out


def paste_rgb_layer(
    canvas_rgb: torch.Tensor,
    layer_rgb: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    return (canvas_rgb * (1.0 - m) + layer_rgb * m).clone()


def chop_layer(
    backdrop_rgba: torch.Tensor,
    source_rgba: torch.Tensor,
    blend_mode: str,
    opacity_pct: int,
    *,
    dissolve_seed: int = 0,
    batch_index: int = 0,
) -> torch.Tensor:
    op = float(opacity_pct) / 100.0
    return blend_rgba(
        backdrop_rgba,
        source_rgba,
        blend_mode,
        op,
        dissolve_seed=dissolve_seed,
        batch_index=batch_index,
    )


def build_report(base: str, batch: int, notes: list[str]) -> str:
    extra = " ".join(notes)
    if extra:
        return f"{base} Processed {batch} frame(s). {extra}"
    return f"{base} Processed {batch} frame(s)."


def clone_output(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone()
