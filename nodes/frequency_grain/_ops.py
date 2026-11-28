# PORTED FROM: ComfyUI-NKD-Basic-Tools by Nekodificador (MIT)
# https://github.com/Nekodificador/ComfyUI-NKD-Basic-Tools @ 613fd3c (2026-09-20, the latest upstream commit
# when this was ported, MIT). Upstream moved to GPL-3.0 on 2026-10-29: never re-sync from a later revision
# into this Apache-2.0 pack (A24 licence audit R8).
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
_SEPARATE_FRAME_CHUNK = 8   # frames filtered at once by Frequency Separate (audit 2026-11-27: was the whole clip)
_GRAIN_ROLLOFF = 0.1        # grain fades to 0 over the last 10 % toward pure black (and pure white on SDR)


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
    """Hop CPU tensors to ComfyUI's compute device for heavy filters; fall back on failure.

    L9.297: this used to test torch.cuda.is_available() only, so the filters ran on the GPU also under ComfyUI
    --cpu. The device now comes from comfy.model_management (which honours --cpu / --directml / --gpu-only);
    without ComfyUI the tensor stays where it is."""
    if x.device.type != "cpu":
        return x
    try:
        import comfy.model_management as mm
        target = mm.get_torch_device()
    except Exception:  # noqa: BLE001 - no ComfyUI (scripts, tests): keep the input device
        return x
    if getattr(target, "type", "cpu") == "cpu":
        return x
    try:
        return x.to(target, non_blocking=True)
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
    return out


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
    return out


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
    return out


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
    return box_blur_nchw(a, r) * guide + box_blur_nchw(b_coef, r)


def rolling_guidance_nchw(
    x: torch.Tensor,
    radius: int,
    eps: float,
    iters: int = 4,
) -> torch.Tensor:
    g = gaussian_blur_nchw(x, radius)
    for _ in range(iters):
        g = guided_filter_nchw(x, g, radius, eps)
    return g


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
    """Split image into HF (WORK) and LF (DISPLAY when linear). Never mutates image.

    Filtered _SEPARATE_FRAME_CHUNK frames at a time (every filter is per frame, so the result is identical): the
    whole clip at once took 20+ GB on the GPU for a 124-frame 2K Guided split (audit 2026-11-27)."""
    notes: list[str] = []
    stats = None
    m_full = None
    if mask is not None:
        m_full = resize_mask(mask.to(image.device), image.shape[2], image.shape[1]).clamp(0.0, 1.0)
    hf_parts, lf_parts, hfm_parts = [], [], []
    for s0 in range(0, image.shape[0], _SEPARATE_FRAME_CHUNK):
        s1 = min(image.shape[0], s0 + _SEPARATE_FRAME_CHUNK)
        work = work_device(image_to_nchw(image[s0:s1]))
        if linear:
            work = srgb_to_linear(work)
        lf_work, stats = low_frequency_nchw(work, method, radius, edge_threshold)

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

        if m_full is not None:
            fidx = torch.arange(s0, s1).clamp(max=m_full.shape[0] - 1)
            m = m_full[fidx].to(work.device).unsqueeze(1)
            neutral = 1.0 if mode == "Divide" else 0.0
            hf_masked = hf * m + neutral * (1.0 - m)
        else:
            hf_masked = hf

        lf_disp = linear_to_srgb(lf_work) if linear else lf_work
        hf_parts.append(nchw_to_image(hf).to(image.device))
        lf_parts.append(nchw_to_image(lf_disp).to(image.device))
        hfm_parts.append(nchw_to_image(hf_masked).to(image.device))
        del work, lf_work, hf, hf_masked, lf_disp

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
    if m_full is not None:
        notes.append(f"mask applied coverage {_mask_coverage(m_full):.1f}%")
        if m_full.shape[0] < image.shape[0]:
            notes.append(
                f"mask batch {m_full.shape[0]} < image batch {image.shape[0]}; "
                "extra frames reuse the last mask"
            )
    else:
        notes.append("no mask")
    # nchw_to_image already returns a fresh tensor; cloning again would
    # double-copy three full-resolution batches (~490MB each on a
    # 124-frame 768x432 clip).
    return torch.cat(hf_parts, 0), torch.cat(lf_parts, 0), torch.cat(hfm_parts, 0), notes


def frequency_combine(
    high_frequency: torch.Tensor,
    low_frequency: torch.Tensor,
    mode: str,
    linear: bool,
    detail_strength: float,
    mask: Optional[torch.Tensor] = None,
    mask_feather: float = 0.0,
) -> tuple[torch.Tensor, list[str]]:
    """Recombine HF onto LF, _SEPARATE_FRAME_CHUNK frames at a time (agy review 2026-11-28: the whole clip went to
    the GPU at once - 124 frames of 2K did not fit). Every step is per frame, so the result is identical."""
    B = int(high_frequency.shape[0])
    if B <= _SEPARATE_FRAME_CHUNK:
        return _frequency_combine_frames(high_frequency, low_frequency, mode, linear, detail_strength, mask,
                                         mask_feather)
    outs, notes = [], None
    for s0 in range(0, B, _SEPARATE_FRAME_CHUNK):
        s1 = min(B, s0 + _SEPARATE_FRAME_CHUNK)
        lf_c = low_frequency[s0:s1] if low_frequency.shape[0] == B else low_frequency
        m_c = None
        if mask is not None:
            mb = mask if mask.dim() == 3 else mask.reshape(-1, *mask.shape[-2:])
            if mb.shape[0] == 1:
                m_c = mb
            else:                                           # extra frames reuse the last mask, as before
                idx = torch.arange(s0, s1).clamp(max=mb.shape[0] - 1)
                m_c = mb[idx]
        o, n = _frequency_combine_frames(high_frequency[s0:s1], lf_c, mode, linear, detail_strength, m_c,
                                         mask_feather)
        outs.append(o.cpu() if o.device != high_frequency.device else o)
        if notes is None:
            notes = n
    notes.append(f"processed {B} frames, {_SEPARATE_FRAME_CHUNK} at a time")
    return torch.cat(outs, 0).to(low_frequency.device), notes


def _frequency_combine_frames(
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
    lf_rgb = low_frequency[..., :3]
    if bool(lf_rgb.min() < 0.0) or bool(lf_rgb.max() > 1.0):
        # audit 2026-11-27: an HDR / scene-linear plate (values outside 0..1 on low_frequency) is not clipped.
        notes.append("HDR input (values outside 0..1): output not clamped")
    else:
        # IMAGE socket is 0..1 display range for an SDR plate - overshoot from the detail band is clipped here.
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

    # audit 2026-11-27: (2, 1, 3) unnormalised made colour grain up to 1.8x stronger than mono grain at the same
    # amount; scaled to unit RMS it keeps the per-channel emphasis at the same overall strength.
    cw = torch.tensor(_GRAIN_CHANNEL_W, device=dev, dtype=torch.float32)
    weights = (cw / cw.pow(2).mean().sqrt()).view(1, 1, 1, 3)
    hdr = bool(image[..., :3].min() < 0.0) or bool(image[..., :3].max() > 1.0)
    notes.append("luminance rolloff: no grain at pure black" + ("" if hdr else " and pure white"))
    if hdr:
        notes.append("HDR input: not clamped")
    # The batch stays where it is; only one chunk at a time goes to `dev` (it was all copied there first).
    out = image.clone()

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
        chunk = out[s:e, :, :, :3].to(dev)
        # Luminance rolloff (audit 2026-11-27): uniform additive grain on pure black is half clipped away, so black
        # was lifted (mean 0 -> 0.044 at amount 50) and white pulled down. The grain fades out over the last 10 %.
        y = luminance_rgb(chunk.clamp(0.0, 1.0)).unsqueeze(-1)
        resp = _smoothstep(0.0, _GRAIN_ROLLOFF, y)
        if not hdr:
            resp = resp * _smoothstep(0.0, _GRAIN_ROLLOFF, 1.0 - y)
        res = chunk + grain.to(chunk.dtype) * resp
        if not hdr:
            res = res.clamp(0.0, 1.0)
        out[s:e, :, :, :3] = res.to(out.device)

    # `out` is image.clone() on the input's device: already ours, no further copy.
    return out.to(src_device), notes


def _smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_report(node_name: str, batch: int, notes: list[str]) -> str:
    extra = " ".join(notes)
    if extra:
        return f"{node_name} processed {batch} frame(s). {extra}"
    return f"{node_name} processed {batch} frame(s)."
