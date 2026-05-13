"""VFX-grade post-processing helpers for ``MaskMattingMEC``.

Implements Hollywood-pipeline operations that turn a coarse alpha into a
production-ready key:

  * ``despill``           — colour decontamination on a configurable backing
  * ``lightwrap``         — light spill from the (new) background onto the
                             subject edge for natural compositing
  * ``edge_inside_outside`` — separate the alpha into three masks suitable
                             for layered colour-grading
  * ``tta_flip_fuse``     — horizontal-flip test-time augmentation ensemble
  * ``multiscale_fuse``   — geometric multi-resolution ensembling
  * ``crf_refine``        — DenseCRF wrapper (falls through gracefully)
  * ``guided_refine``     — guided-filter edge snap (torch-only)
  * ``score_quality``     — boundary-alignment / coherence / size / smoothness

All ops accept ``(B,H,W,3) image`` + ``(B,H,W) mask`` tensors in [0,1] and
return tensors in [0,1] (or floats / dicts for scoring).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


logger = logging.getLogger("MEC.MaskMatting.VFX")


# ──────────────────────────────────────────────────────────────────────
# Backing colours (broadcast standards)
# ──────────────────────────────────────────────────────────────────────
_BACKING_COLOR: Dict[str, Tuple[float, float, float]] = {
    "green":  (0.10, 0.85, 0.10),
    "blue":   (0.10, 0.20, 0.85),
    "red":    (0.85, 0.10, 0.10),
    "magenta": (0.85, 0.10, 0.85),
    "cyan":    (0.10, 0.85, 0.85),
    "yellow":  (0.85, 0.85, 0.10),
    "white":   (0.95, 0.95, 0.95),
    "black":   (0.05, 0.05, 0.05),
    "auto":    (0.0, 0.0, 0.0),  # estimated from corners at runtime
}


def _auto_backing(img_bhwc: torch.Tensor) -> torch.Tensor:
    """Estimate the dominant backing colour from the four image corners.

    Returns a (B,3) tensor on the input device.
    """
    B, H, W, _ = img_bhwc.shape
    k = max(8, min(H, W) // 32)
    corners = torch.stack([
        img_bhwc[:, :k, :k].mean(dim=(1, 2)),
        img_bhwc[:, :k, -k:].mean(dim=(1, 2)),
        img_bhwc[:, -k:, :k].mean(dim=(1, 2)),
        img_bhwc[:, -k:, -k:].mean(dim=(1, 2)),
    ], dim=0)
    return corners.median(dim=0).values


# ──────────────────────────────────────────────────────────────────────
# Despill (colour decontamination)
# ──────────────────────────────────────────────────────────────────────

def despill(image_bhwc: torch.Tensor, alpha_bhw: torch.Tensor, *,
            backing: str = "green", strength: float = 1.0,
            preserve_skin: bool = True) -> torch.Tensor:
    """Suppress backing-colour bleed on the subject.

    Strategy (per channel):
        spill = max(0, channel - max(other_channels))
        out   = channel - strength * spill * alpha
    For "auto", subtract the projection of pixel colour onto the
    estimated backing direction.
    Optional ``preserve_skin`` keeps warm pixels (R > G > B) untouched.
    """
    if strength <= 0:
        return image_bhwc.clone()
    img = image_bhwc.clone()
    a = alpha_bhw.unsqueeze(-1).clamp(0, 1)
    if backing == "auto":
        bc = _auto_backing(image_bhwc).to(image_bhwc.device).to(image_bhwc.dtype)
        # Project each pixel onto bc direction and subtract weighted by alpha.
        norm = (bc ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        for b in range(img.shape[0]):
            proj = (img[b] * bc[b]).sum(dim=-1, keepdim=True) / norm[b]
            sub = (proj * bc[b]) * (strength * a[b])
            img[b] = (img[b] - sub).clamp(0, 1)
        return img
    if backing == "green":
        spill = (img[..., 1:2] - torch.max(img[..., 0:1], img[..., 2:3])).clamp_min(0)
        img[..., 1:2] = img[..., 1:2] - strength * spill * a
    elif backing == "blue":
        spill = (img[..., 2:3] - torch.max(img[..., 0:1], img[..., 1:2])).clamp_min(0)
        img[..., 2:3] = img[..., 2:3] - strength * spill * a
    elif backing == "red":
        spill = (img[..., 0:1] - torch.max(img[..., 1:2], img[..., 2:3])).clamp_min(0)
        img[..., 0:1] = img[..., 0:1] - strength * spill * a
    else:
        # Use specified backing colour as direction.
        bc = torch.tensor(_BACKING_COLOR.get(backing, (0, 1, 0)),
                          device=img.device, dtype=img.dtype)
        norm = (bc ** 2).sum().clamp_min(1e-6)
        proj = (img * bc).sum(dim=-1, keepdim=True) / norm
        sub = (proj * bc) * (strength * a)
        img = (img - sub).clamp(0, 1)
    if preserve_skin:
        # Warm pixels with R > G+0.1 and R > B+0.05 are skin/wood — restore them.
        skin = ((image_bhwc[..., 0] > image_bhwc[..., 1] + 0.10) &
                (image_bhwc[..., 0] > image_bhwc[..., 2] + 0.05)).unsqueeze(-1).float()
        img = img * (1.0 - skin) + image_bhwc * skin
    return img.clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────
# Light wrap
# ──────────────────────────────────────────────────────────────────────

def lightwrap_layer(image_bhwc: torch.Tensor, alpha_bhw: torch.Tensor, *,
                    bg_color: Optional[Tuple[float, float, float]] = None,
                    radius: int = 8, strength: float = 0.4) -> torch.Tensor:
    """Produce a "light wrap" RGBA layer to comp on top of the new BG.

    Wrap = bg_color * (alpha_blurred - alpha) * strength
    Returned tensor is (B,H,W,4) with the wrap RGB and the wrap alpha so
    the compositor can ADD it after putting the subject over the new BG.
    """
    if strength <= 0:
        z = torch.zeros((*image_bhwc.shape[:3], 4), device=image_bhwc.device,
                        dtype=image_bhwc.dtype)
        return z
    if bg_color is None:
        bg = _auto_backing(image_bhwc).to(image_bhwc.device)  # (B,3)
    else:
        bg = torch.tensor(bg_color, device=image_bhwc.device,
                          dtype=image_bhwc.dtype).expand(image_bhwc.shape[0], 3)
    a = alpha_bhw.unsqueeze(1)
    k = 2 * radius + 1
    box = torch.ones((1, 1, k, k), device=a.device, dtype=a.dtype) / float(k * k)
    blurred = F.conv2d(F.pad(a, (radius,) * 4, mode="reflect"), box)
    halo = (blurred - a).clamp_min(0).squeeze(1) * strength
    halo_rgb = halo.unsqueeze(-1) * bg.view(-1, 1, 1, 3)
    return torch.cat([halo_rgb, halo.unsqueeze(-1)], dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Edge / Inside / Outside masks
# ──────────────────────────────────────────────────────────────────────

def edge_inside_outside(alpha_bhw: torch.Tensor, edge_radius: int = 4
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the alpha into three production masks.

    inside  = eroded(alpha > 0.99, r)   — safe to colour-grade
    outside = eroded(alpha < 0.01, r)   — safe to defocus
    edge    = everything else (soft band where matting matters)
    """
    a = alpha_bhw.unsqueeze(1)
    r = max(1, int(edge_radius))
    k = 2 * r + 1
    kernel = torch.ones((1, 1, k, k), device=a.device, dtype=a.dtype)
    # Max-pool acts as binary dilation; -max(-x) acts as erosion.
    fg = (a > 0.99).float()
    bg = (a < 0.01).float()
    fg_erode = -F.max_pool2d(-fg, kernel_size=k, stride=1, padding=r)
    bg_erode = -F.max_pool2d(-bg, kernel_size=k, stride=1, padding=r)
    inside = fg_erode.squeeze(1).clamp(0, 1)
    outside = bg_erode.squeeze(1).clamp(0, 1)
    edge = (1.0 - inside - outside).clamp(0, 1)
    return edge, inside, outside


# ──────────────────────────────────────────────────────────────────────
# TTA (horizontal-flip ensemble)
# ──────────────────────────────────────────────────────────────────────

def tta_flip_fuse(image_bhwc: torch.Tensor,
                  segment_fn: Callable[[torch.Tensor], torch.Tensor],
                  reduce: str = "mean") -> torch.Tensor:
    """Run ``segment_fn`` on the image AND its horizontal flip; fuse.

    ``segment_fn`` is a closure that maps (B,H,W,3) → (B,H,W) mask.
    """
    m1 = segment_fn(image_bhwc).float().clamp(0, 1)
    flipped = torch.flip(image_bhwc, dims=[2])
    m2 = segment_fn(flipped).float().clamp(0, 1)
    m2 = torch.flip(m2, dims=[2])
    if reduce == "min":
        return torch.minimum(m1, m2)
    if reduce == "max":
        return torch.maximum(m1, m2)
    return 0.5 * (m1 + m2)


# ──────────────────────────────────────────────────────────────────────
# Multi-scale fusion
# ──────────────────────────────────────────────────────────────────────

def multiscale_fuse(image_bhwc: torch.Tensor,
                    segment_fn: Callable[[torch.Tensor], torch.Tensor],
                    scales: Tuple[float, ...] = (0.75, 1.0, 1.25),
                    reduce: str = "mean") -> torch.Tensor:
    """Run the segmenter at multiple scales and fuse back at native res.

    Geometric averaging tends to remove scale-specific failure modes.
    """
    B, H, W, _ = image_bhwc.shape
    masks: List[torch.Tensor] = []
    for s in scales:
        if abs(s - 1.0) < 1e-3:
            mi = segment_fn(image_bhwc).float().clamp(0, 1)
        else:
            sh, sw = max(32, int(H * s)), max(32, int(W * s))
            chw = image_bhwc.permute(0, 3, 1, 2)
            chw_s = F.interpolate(chw, size=(sh, sw), mode="bilinear",
                                   align_corners=False)
            img_s = chw_s.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
            mi = segment_fn(img_s).float().clamp(0, 1)
            mi = F.interpolate(mi.unsqueeze(1), size=(H, W), mode="bilinear",
                                align_corners=False).squeeze(1)
        masks.append(mi)
    stack = torch.stack(masks, dim=0)
    if reduce == "max":
        return stack.max(dim=0).values
    if reduce == "min":
        return stack.min(dim=0).values
    return stack.mean(dim=0)


# ──────────────────────────────────────────────────────────────────────
# Guided filter & CRF wrappers (delegate to mask_refine when available)
# ──────────────────────────────────────────────────────────────────────

def guided_refine(image_bhwc: torch.Tensor, mask_bhw: torch.Tensor, *,
                  radius: int = 8, epsilon: float = 1e-4) -> torch.Tensor:
    """Torch-only guided filter edge-snap (He et al)."""
    I = image_bhwc.permute(0, 3, 1, 2).contiguous()
    if I.shape[1] == 3:
        I = 0.2126 * I[:, 0:1] + 0.7152 * I[:, 1:2] + 0.0722 * I[:, 2:3]
    p = mask_bhw.unsqueeze(1)
    k = 2 * int(radius) + 1
    box = torch.ones((1, 1, k, k), device=I.device, dtype=I.dtype) / float(k * k)
    pad = (int(radius),) * 4

    def _box(x):
        return F.conv2d(F.pad(x, pad, mode="reflect"), box)

    mean_I = _box(I)
    mean_p = _box(p)
    corr_I = _box(I * I)
    corr_Ip = _box(I * p)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + float(epsilon))
    b = mean_p - a * mean_I
    out = _box(a) * I + _box(b)
    return out.squeeze(1).clamp(0, 1)


def crf_refine(image_bhwc: torch.Tensor, mask_bhw: torch.Tensor, *,
               iterations: int = 5, gauss_sxy: float = 3.0,
               bilateral_sxy: float = 50.0, bilateral_srgb: float = 13.0
               ) -> torch.Tensor:
    """DenseCRF wrapper; falls back to guided filter if pydensecrf is missing."""
    try:
        import pydensecrf.densecrf as dcrf  # type: ignore[import-not-found]
        from pydensecrf.utils import unary_from_softmax  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("[VFX] pydensecrf missing — falling back to guided filter for CRF refine.")
        return guided_refine(image_bhwc, mask_bhw, radius=8, epsilon=1e-4)

    out = mask_bhw.clone()
    B = mask_bhw.shape[0]
    for b in range(B):
        rgb = (image_bhwc[b].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        sm = mask_bhw[b].detach().cpu().numpy().clip(1e-6, 1 - 1e-6).astype(np.float32)
        H, W = sm.shape
        probs = np.stack([1.0 - sm, sm], axis=0).reshape(2, -1)
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(unary_from_softmax(probs))
        d.addPairwiseGaussian(sxy=(gauss_sxy, gauss_sxy), compat=3)
        d.addPairwiseBilateral(
            sxy=(bilateral_sxy, bilateral_sxy),
            srgb=(bilateral_srgb, bilateral_srgb, bilateral_srgb),
            rgbim=np.ascontiguousarray(rgb), compat=10,
        )
        Q = d.inference(int(iterations))
        out[b] = torch.from_numpy(
            np.array(Q).reshape(2, H, W)[1].astype(np.float32)
        ).to(out.device)
    return out.clamp(0, 1)


# ──────────────────────────────────────────────────────────────────────
# No-GT quality scoring
# ──────────────────────────────────────────────────────────────────────

def score_quality(image_bhwc: torch.Tensor, alpha_bhw: torch.Tensor, *,
                  binarize_at: float = 0.5) -> Dict[str, Any]:
    """Return a dict with per-frame scores + an overall production score.

    Heuristics (each in [0,1]):
      * boundary-gradient alignment
      * FG/BG colour-distance coherence
      * size-sanity (penalises full-black or full-white masks)
      * boundary compactness
    Final = weighted geometric mean.
    """
    img = image_bhwc.detach().cpu().numpy()
    msk = alpha_bhw.detach().cpu().numpy()
    frames = []
    overalls: List[float] = []
    for b in range(img.shape[0]):
        rgb = img[b]
        m = msk[b]
        H, W = m.shape
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        grad = np.sqrt(gx * gx + gy * gy)
        grad = grad / max(grad.max(), 1e-6)
        mgx = np.gradient(m, axis=1)
        mgy = np.gradient(m, axis=0)
        bnd = (np.sqrt(mgx * mgx + mgy * mgy) > 0.05)
        if bnd.sum() > 5:
            edge_align = float((grad * bnd).sum() / (bnd.sum() + 1e-6))
            edge_align = float(np.clip(edge_align * 4.0, 0.0, 1.0))
        else:
            edge_align = 0.0
        fg = m > binarize_at
        bgm = ~fg
        if fg.sum() > 50 and bgm.sum() > 50:
            fm, bm = rgb[fg].mean(axis=0), rgb[bgm].mean(axis=0)
            fs = rgb[fg].std(axis=0) + 1e-6
            d = float(np.linalg.norm((fm - bm) / fs))
            coherence = float(np.clip(d / 3.0, 0.0, 1.0))
        else:
            coherence = 0.0
        area = float(fg.sum()) / float(H * W)
        if area < 0.001:
            size_score = float(area / 0.001)
        elif area > 0.95:
            size_score = float((1.0 - area) / 0.05)
        else:
            size_score = 1.0
        size_score = float(np.clip(size_score, 0.0, 1.0))
        if fg.sum() > 50 and bnd.sum() > 0:
            compactness = (4.0 * np.pi * fg.sum()) / max(int(bnd.sum()) ** 2, 1)
            smoothness = float(np.clip(compactness, 0.0, 1.0))
        else:
            smoothness = 0.0
        geo = (max(edge_align, 1e-3) ** 0.4 *
               max(coherence, 1e-3) ** 0.3 *
               max(size_score, 1e-3) ** 0.2 *
               max(smoothness, 1e-3) ** 0.1)
        frames.append({
            "edge_align": edge_align, "coherence": coherence,
            "size_sanity": size_score, "smoothness": smoothness,
            "area_frac": area, "overall": float(geo),
        })
        overalls.append(float(geo))
    return {"frames": frames,
            "overall": float(np.mean(overalls)) if overalls else 0.0}
