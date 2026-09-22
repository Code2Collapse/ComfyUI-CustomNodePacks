# PORTED FROM: ComfyUI-NKD-Basic-Tools by Nekodificador (MIT)
# https://github.com/Nekodificador/ComfyUI-NKD-Basic-Tools
#
# Sources: nkd_frequency.py, nkd_film_grain.py, helpers.py (_luminance,
# _srgb_to_linear, _linear_to_srgb, _resize_mask, _film_grain and _GRAIN_*).
# This is a PORT, not a clean-room reimplementation.
#
# Colour-space contract (read this before wiring Separate -> Combine):
#   * Separate runs the split in WORKING space (linear sRGB when linear=True).
#   * high_frequency leaves Separate in WORKING space as a raw ratio (Divide)
#     or difference (Subtract). It may exceed [0, 1] — it is an intermediate,
#     not a display picture.
#   * low_frequency leaves Separate in DISPLAY space (gamma/sRGB when linear=True)
#     so it previews correctly and matches a typical relit plate on the Combine
#     low_frequency input. Combine re-linearises that input before recombining.
#   * mode and linear must match between Separate and Combine.
# WHY no_grad AND NOT inference_mode: see nodes/layer_effects/_ops.py.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..layer_effects._ops import expand_mask

_EPS = 1e-6
_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
_MEDIAN_RADIUS_CAP = 7

# Calibration knobs, carried over from upstream verbatim (their comment: a
# monitor and a taste need tuning a formula can't see). Change these only
# against a reference render - they are what makes amount=100 read right.
_GRAIN_MAX_AMOUNT = 0.22    # grain std at amount=100 (unit-normalised field)
_GRAIN_SIZE_SCALE_MAX = 7.0  # size=100 downsamples the noise field by ~1+this
_GRAIN_COARSE_MULT = 2.6    # the coarse roughness field is this much lower-frequency
_GRAIN_ROUGH_MAX = 0.6      # roughness=100 blends in this much of the coarse field
_GRAIN_CHANNEL_W = (2.0, 1.0, 3.0)  # per-channel emphasis for color grain (R, G, B)
_GRAIN_FRAME_CHUNK = 8      # frames upscaled/composited at once (VRAM bound)


def luminance_rgb(x: torch.Tensor) -> torch.Tensor:
    """Rec.709 luminance. x is [..., 3] in 0..1."""
    w = torch.tensor(_LUMA_WEIGHTS, device=x.device, dtype=x.dtype)
    return (x[..., :3] * w).sum(-1)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def image_to_nchw(image: torch.Tensor) -> torch.Tensor:
    return image[..., :3].permute(0, 3, 1, 2).contiguous().clone()


def nchw_to_image(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1).contiguous().clone()


def resize_mask(mask: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[1] == height and mask.shape[2] == width:
        return mask.clone()
    x = mask.unsqueeze(1).float()
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.squeeze(1).clone()


def work_device(x: torch.Tensor) -> torch.Tensor:
    """Hop CPU tensors to CUDA for heavy filters; fall back on failure."""
    if x.device.type != "cpu":
        return x
    if not torch.cuda.is_available():
        return x
    try:
        return x.to("cuda", non_blocking=True)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return x


def box_blur_nchw(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 1:
        return x.clone()
    r = int(radius)
    k = 2 * r + 1
    c = x.shape[1]
    kh = torch.ones(c, 1, 1, k, device=x.device, dtype=x.dtype) / k
    kv = kh.transpose(2, 3)
    out = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kh, groups=c)
    out = F.conv2d(F.pad(out, (0, 0, r, r), mode="replicate"), kv, groups=c)
    return out.clone()


def gaussian_blur_nchw(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius < 1:
        return x.clone()
    r = int(radius)
    sigma = max(r / 2.0, 0.5)
    k = 2 * r + 1
    t = torch.arange(k, device=x.device, dtype=x.dtype) - r
    g = torch.exp(-(t * t) / (2 * sigma * sigma))
    g = g / g.sum()
    c = x.shape[1]
    kh = g.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    kv = kh.transpose(2, 3)
    out = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kh, groups=c)
    out = F.conv2d(F.pad(out, (0, 0, r, r), mode="replicate"), kv, groups=c)
    return out.clone()


def median_blur_nchw(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Windowed median, radius r. O((2r+1)^2) per pixel — capped and row-chunked."""
    if radius < 1:
        return x.clone()
    r = min(int(radius), _MEDIAN_RADIUS_CAP)
    k = 2 * r + 1
    b, c, h, w = x.shape
    xp = F.pad(x, (r, r, r, r), mode="replicate")
    out = torch.empty_like(x)
    rows = max(1, 64 // (k * k) + 1)
    for y0 in range(0, h, rows):
        y1 = min(y0 + rows, h)
        patch = xp[:, :, y0:y1 + 2 * r, :].unfold(2, k, 1).unfold(3, k, 1)
        patch = patch.contiguous().view(b, c, y1 - y0, w, k * k)
        out[:, :, y0:y1, :] = patch.median(dim=-1).values
    return out.clone()


def guided_filter_nchw(
    x: torch.Tensor,
    guide: torch.Tensor,
    radius: int,
    eps: float,
) -> torch.Tensor:
    r = max(int(radius), 1)
    mean_g = box_blur_nchw(guide, r)
    mean_x = box_blur_nchw(x, r)
    corr_gg = box_blur_nchw(guide * guide, r)
    corr_gx = box_blur_nchw(guide * x, r)
    var_g = corr_gg - mean_g * mean_g
    cov_gx = corr_gx - mean_g * mean_x
    a = cov_gx / (var_g + eps)
    b_coef = mean_x - a * mean_g
    return (box_blur_nchw(a, r) * guide + box_blur_nchw(b_coef, r)).clone()


def rolling_guidance_nchw(
    x: torch.Tensor,
    radius: int,
    eps: float,
    iters: int = 4,
) -> torch.Tensor:
    g = gaussian_blur_nchw(x, radius)
    for _ in range(iters):
        g = guided_filter_nchw(x, g, radius, eps)
    return g.clone()


def low_frequency_nchw(
    img: torch.Tensor,
    method: str,
    radius: int,
    edge_threshold: float,
) -> tuple[torch.Tensor, dict]:
    """img NCHW -> low-frequency NCHW. Returns (lf, stats dict)."""
    requested = int(radius)
    effective = requested
    if method == "Median":
        effective = min(requested, _MEDIAN_RADIUS_CAP)
    eps = max(float(edge_threshold), 1e-3) ** 2
    if method == "Gaussian":
        lf = gaussian_blur_nchw(img, requested)
    elif method == "Median":
        lf = median_blur_nchw(img, requested)
    elif method == "Rolling Guidance":
        lf = rolling_guidance_nchw(img, requested, eps)
    else:  # Guided (default)
        lf = guided_filter_nchw(img, img, requested, eps)
    stats = {
        "method": method,
        "radius_requested": requested,
        "radius_effective": effective,
    }
    return lf, stats


def _mask_coverage(mask: torch.Tensor) -> float:
    return float(mask.clamp(0.0, 1.0).mean().item()) * 100.0


def _batch_index_mask(mask: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    fidx = torch.arange(batch, device=device).clamp(max=mask.shape[0] - 1)
    return mask[fidx]


def frequency_separate(
    image: torch.Tensor,
    method: str,
    radius: int,
    edge_threshold: float,
    mode: str,
    detail: str,
    linear: bool,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Split image into HF (WORK) and LF (DISPLAY when linear). Never mutates image."""
    notes: list[str] = []
    src = image_to_nchw(image)
    work = work_device(src)
    if linear:
        work = srgb_to_linear(work)

    lf_work, stats = low_frequency_nchw(work, method, radius, edge_threshold)
    notes.append(
        f"filter {stats['method']} radius {stats['radius_requested']}"
        + (
            f" (effective {stats['radius_effective']})"
            if stats["radius_effective"] != stats["radius_requested"]
            else ""
        )
    )
    notes.append(f"linear light {'on' if linear else 'off'}")
    notes.append(f"mode {mode} detail {detail}")

    lw = torch.tensor(_LUMA_WEIGHTS, device=work.device, dtype=work.dtype).view(1, 3, 1, 1)
    if detail == "Luminance":
        luma_img = (work * lw).sum(1, keepdim=True)
        luma_lf = (lf_work * lw).sum(1, keepdim=True)
        if mode == "Divide":
            hf = (luma_img / (luma_lf + _EPS)).repeat(1, 3, 1, 1)
        else:
            hf = (luma_img - luma_lf).repeat(1, 3, 1, 1)
    else:
        hf = work / (lf_work + _EPS) if mode == "Divide" else work - lf_work

    hf_masked = hf.clone()
    if mask is not None:
        m = resize_mask(mask.to(work.device), work.shape[3], work.shape[2])
        m = m.clamp(0.0, 1.0)
        notes.append(f"mask applied coverage {_mask_coverage(m):.1f}%")
        if m.shape[0] < image.shape[0]:
            notes.append(
                f"mask batch {m.shape[0]} < image batch {image.shape[0]}; "
                "extra frames reuse the last mask"
            )
        m = m.unsqueeze(1)
        m = _batch_index_mask(m, work.shape[0], work.device)
        neutral = 1.0 if mode == "Divide" else 0.0
        hf_masked = hf * m + neutral * (1.0 - m)
    else:
        notes.append("no mask")

    lf_disp = linear_to_srgb(lf_work) if linear else lf_work
    hf_out = nchw_to_image(hf).to(image.device)
    lf_out = nchw_to_image(lf_disp).to(image.device)
    hf_masked_out = nchw_to_image(hf_masked).to(image.device)
    # nchw_to_image already returns a fresh tensor; cloning again would
    # double-copy three full-resolution batches (~490MB each on a
    # 124-frame 768x432 clip).
    return hf_out, lf_out, hf_masked_out, notes


def frequency_combine(
    high_frequency: torch.Tensor,
    low_frequency: torch.Tensor,
    mode: str,
    linear: bool,
    detail_strength: float,
    mask: Optional[torch.Tensor] = None,
    mask_feather: float = 0.0,
) -> tuple[torch.Tensor, list[str]]:
    """Recombine HF onto LF. Preserves RGBA alpha from low_frequency."""
    notes: list[str] = []
    notes.append(f"mode {mode} linear light {'on' if linear else 'off'}")
    notes.append(f"detail_strength {float(detail_strength):.2f}")

    hf = work_device(image_to_nchw(high_frequency))
    base = image_to_nchw(low_frequency).to(hf.device)
    if base.shape[2:] != hf.shape[2:]:
        base = F.interpolate(
            base, size=hf.shape[2:], mode="bilinear", align_corners=False,
        )
    lf = srgb_to_linear(base) if linear else base

    s = float(detail_strength)
    if mode == "Divide":
        # Negative HF ratios are discarded before the power — upstream behaviour.
        recombined = lf * (hf.clamp(min=0.0) ** s)
    else:
        recombined = lf + hf * s

    if mask is not None:
        m = resize_mask(mask.to(hf.device), hf.shape[3], hf.shape[2]).clamp(0.0, 1.0)
        feather_notes: list[str] = []
        if mask_feather >= 1.0:
            # expand_mask expects [B,H,W]; grow=0, blur=feather matches upstream _mask_grow(0, blur).
            m = expand_mask(m, 0, int(mask_feather), feather_notes)
            notes.extend(feather_notes)
            notes.append(f"mask_feather {int(mask_feather)}px")
        m = m.unsqueeze(1)
        if m.shape[0] < hf.shape[0]:
            notes.append(
                f"mask batch {m.shape[0]} < image batch {hf.shape[0]}; "
                "extra frames reuse the last mask"
            )
        m = _batch_index_mask(m, hf.shape[0], hf.device)
        notes.append(f"mask applied coverage {_mask_coverage(m.squeeze(1)):.1f}%")
        recombined = lf * (1.0 - m) + recombined * m
    else:
        notes.append("no mask")

    out = linear_to_srgb(recombined) if linear else recombined
    # IMAGE socket is 0..1 display range — scene-linear values above 1.0 are clipped here.
    out = out.clamp(0.0, 1.0)
    notes.append(
        "output clamped to 0..1 — values above 1.0 are clipped; "
        "this socket is an 8-bit-range IMAGE, not a scene-linear plate"
    )

    out_hwc = nchw_to_image(out).to(low_frequency.device)
    if low_frequency.shape[-1] > 3:
        # torch.cat allocates its own output, so the alpha slice is read, never
        # aliased, and the result is already a tensor we own.
        out_hwc = torch.cat([out_hwc, low_frequency[..., 3:]], dim=-1)
    return out_hwc, notes


def film_grain(
    image: torch.Tensor,
    amount: float,
    size: float,
    roughness: float,
    color: float,
    seed: int,
    animate: bool = True,
    mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, list[str]]:
    """Lightroom-style film grain. Returns (image clone, notes)."""
    notes: list[str] = []
    b, h, w, c = image.shape
    src_device = image.device
    dev = device or src_device

    strength = (max(0.0, amount) / 100.0) * _GRAIN_MAX_AMOUNT
    if strength <= 0.0:
        notes.append("amount 0 — input unchanged")
        return image.clone(), notes

    color01 = min(max(color, 0.0), 100.0) / 100.0
    rough01 = min(max(roughness, 0.0), 100.0) / 100.0
    notes.append(f"amount {amount:.1f} size {size:.1f} roughness {roughness:.1f} color {color:.1f}")
    notes.append(f"animate {'on' if animate else 'off'} seed {int(seed)}")

    mask_full = None
    if mask is not None:
        mm = mask if mask.dim() == 3 else mask.unsqueeze(0)
        mask_full = resize_mask(mm.to(dev), w, h).clamp(0.0, 1.0)
        notes.append(f"mask applied coverage {_mask_coverage(mask_full):.1f}%")
        if mask_full.shape[0] < b:
            notes.append(
                f"mask batch {mask_full.shape[0]} < image batch {b}; "
                "extra frames reuse the last mask"
            )
    else:
        notes.append("no mask")

    scale = 1.0 + (min(max(size, 0.0), 100.0) / 100.0) * _GRAIN_SIZE_SCALE_MAX
    fine_h, fine_w = max(1, round(h / scale)), max(1, round(w / scale))
    coarse_h = max(1, round(h / (scale * _GRAIN_COARSE_MULT)))
    coarse_w = max(1, round(w / (scale * _GRAIN_COARSE_MULT)))
    notes.append(f"grain field {fine_w}x{fine_h} coarse {coarse_w}x{coarse_h}")

    frames = b if animate else 1
    gen = torch.Generator(device=dev).manual_seed(int(seed) & 0xffffffffffffffff)
    fine = torch.randn(frames, 3, fine_h, fine_w, generator=gen, device=dev)
    coarse = torch.randn(frames, 3, coarse_h, coarse_w, generator=gen, device=dev)

    weights = torch.tensor(_GRAIN_CHANNEL_W, device=dev).view(1, 1, 1, 3)
    out = image.to(dev).clone()

    for s in range(0, b, _GRAIN_FRAME_CHUNK):
        e = min(b, s + _GRAIN_FRAME_CHUNK)
        idx = slice(s, e) if animate else slice(0, 1)
        n = e - s
        ff = F.interpolate(fine[idx], size=(h, w), mode="bilinear", align_corners=False)
        fc = F.interpolate(coarse[idx], size=(h, w), mode="bilinear", align_corners=False)
        field = ff * (1.0 - rough01 * _GRAIN_ROUGH_MAX) + fc * (rough01 * _GRAIN_ROUGH_MAX)
        field = field.permute(0, 2, 3, 1)
        if not animate and n > 1:
            field = field.expand(n, -1, -1, -1)
        std = field.reshape(field.shape[0], -1).std(dim=1).clamp_min(1e-5).view(-1, 1, 1, 1)
        field = field / std
        mono = field[..., 1:2].expand(-1, -1, -1, 3)
        grain = mono * (1.0 - color01) + field * weights * color01
        grain = grain * strength
        if mask_full is not None:
            fidx = torch.arange(s, e, device=dev).clamp(max=mask_full.shape[0] - 1)
            grain = grain * mask_full[fidx].unsqueeze(-1)
        out[s:e, :, :, :3] = (out[s:e, :, :, :3] + grain).clamp(0.0, 1.0)

    # `out` was built by image.to(dev).clone() above, so it is already ours;
    # .to() copies again when the device differs and is a no-op when it
    # does not. Either way a further clone is a wasted full-batch copy.
    return out.to(src_device), notes


def build_report(node_name: str, batch: int, notes: list[str]) -> str:
    extra = " ".join(notes)
    if extra:
        return f"{node_name} processed {batch} frame(s). {extra}"
    return f"{node_name} processed {batch} frame(s)."
