"""
InpaintSuiteMEC — Inpaint Crop Pro + Inpaint Stitch Pro.

Two nodes for crop -> inpaint -> put back (L7.65 P08: seven nodes became these two):
  1. InpaintCropProMEC    — crop_mode "pro" (crop around the mask, video aware), "smart aspect" (the former Smart
                            Image Crop) or "mask prepare" (the former Inpaint Mask Prepare: no crop, clean masks)
  2. InpaintStitchProMEC  — stitch_mode "blend" (blend modes + colour match), "paste back" (the former Inpaint Paste
                            Back), "smart" (the former Smart Image Stitch) or "context blend" (the former Context
                            Inpainter: colour correction and lightness rescue, no stitcher)
The former nodes' classes stay below / in smart_crop.py / mec_paint_suite.py as the engines of those modes and are
not registered. Design: docs/design/inpaint_crop_stitch_unified.md.

Key innovations over InpaintCropAndStitch:
  - Separated inpaint_mask_mode (what model sees) from stitch_blend_mode (how stitch composites)
  - Edge-aware blend: Sobel-guided boundary snapping
  - Laplacian pyramid blend: real multi-level frequency decomposition
  - Frequency blend: FFT-domain blending
  - video_stable_crop: lock bbox across all frames
  - Temporal blend mask stabilization

VRAM Tier: 1 (pure tensor ops, no models)
"""

from __future__ import annotations

import gc
import math
import logging
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn.functional as F

from ._is_changed_util import hash_args_and_kwargs, hash_kwargs
from . import smart_crop as _sc
from .smart_crop import SmartImageCropMEC, SmartImageStitcherMEC, plan_crop

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger("MEC")


# ══════════════════════════════════════════════════════════════════════
#  Device helper
# ══════════════════════════════════════════════════════════════════════

def _get_device(tensor: torch.Tensor) -> torch.device:
    """Return the device of a tensor — never hardcode 'cuda'."""
    return tensor.device


# ══════════════════════════════════════════════════════════════════════
#  Gaussian kernel helpers
# ══════════════════════════════════════════════════════════════════════

def _gauss_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a normalized 1D Gaussian kernel."""
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    x = torch.arange(size, device=device, dtype=dtype) - radius
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur_2d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 2D Gaussian blur. tensor: (B, C, H, W) or (B, 1, H, W).

    Pure torch — no cv2 dependency.
    """
    if sigma <= 0:
        return tensor
    device = _get_device(tensor)
    k1d = _gauss_kernel_1d(sigma, device, tensor.dtype)
    pad = len(k1d) // 2
    C = tensor.shape[1]
    # Horizontal pass
    kh = k1d.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    out = F.conv2d(F.pad(tensor, (pad, pad, 0, 0), mode="replicate"), kh, groups=C)
    # Vertical pass
    kv = k1d.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="replicate"), kv, groups=C)
    return out


def _gaussian_blur_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a (B, H, W) mask with 2D Gaussian. Returns (B, H, W)."""
    if sigma <= 0:
        return mask
    m4 = mask.unsqueeze(1)  # (B, 1, H, W)
    blurred = _gaussian_blur_2d(m4, sigma)
    return blurred.squeeze(1)


# ══════════════════════════════════════════════════════════════════════
#  Sobel edge detection — pure torch
# ══════════════════════════════════════════════════════════════════════

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)

_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0],
     [ 0.0,  0.0,  0.0],
     [ 1.0,  2.0,  1.0]], dtype=torch.float32
).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)


def _sobel_edges(gray: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude from (B, H, W) grayscale. Returns (B, H, W)."""
    device = _get_device(gray)
    sx = _SOBEL_X.to(device=device, dtype=gray.dtype)
    sy = _SOBEL_Y.to(device=device, dtype=gray.dtype)
    inp = gray.unsqueeze(1)  # (B, 1, H, W)
    gx = F.conv2d(F.pad(inp, (1, 1, 1, 1), mode="replicate"), sx)
    gy = F.conv2d(F.pad(inp, (1, 1, 1, 1), mode="replicate"), sy)
    magnitude = (gx.pow(2) + gy.pow(2)).sqrt().squeeze(1)  # (B, H, W)
    return magnitude


# ══════════════════════════════════════════════════════════════════════
#  Edge-aware blend mask — Sobel-guided boundary snapping
# ══════════════════════════════════════════════════════════════════════

def _edge_aware_blend_mask(image: torch.Tensor, mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Create an edge-aware blend mask that snaps blend boundaries to image edges.

    image: (B, H, W, C) float32
    mask:  (B, H, W) float32 binary/soft
    radius: blend feather radius
    Returns: (B, H, W) soft blend mask
    """
    B, H, W, C = image.shape

    # Step 1: Compute image luminance
    luma = 0.2126 * image[:, :, :, 0] + 0.7152 * image[:, :, :, 1] + 0.0722 * image[:, :, :, 2]

    # Step 2: Compute Sobel edge magnitude
    edge_mag = _sobel_edges(luma)  # (B, H, W)
    # Normalize edge magnitude to [0, 1] per frame
    for b in range(B):
        emax = edge_mag[b].max()
        if emax > 0:
            edge_mag[b] = edge_mag[b] / emax

    # Step 3: Create base Gaussian blend mask (dilated + blurred)
    binary = (mask > 0.5).float()

    # Step 4: Compute edge-snapped mask
    # Where image edges are strong, the blend boundary should be sharper (snap to edges)
    # Where image edges are weak, use the smooth Gaussian boundary
    sigma_base = radius * 0.4
    sigma_min = max(0.5, sigma_base * 0.15)

    # Create a spatially-varying sharpness via blending sharp + smooth versions
    sharp_blend = _gaussian_blur_mask(binary, sigma_min)   # (B, H, W)
    smooth_blend = _gaussian_blur_mask(binary, sigma_base)  # (B, H, W)

    # Blend: at strong edges use sharp, at weak edges use smooth
    edge_weight = edge_mag.clamp(0.0, 1.0)
    result = edge_weight * sharp_blend + (1.0 - edge_weight) * smooth_blend

    # Ensure the mask interior is 1.0 and exterior is 0.0
    result = torch.where(binary > 0.5, torch.max(result, binary), result)

    return result.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Laplacian pyramid blend — real multi-level decomposition (pure torch)
# ══════════════════════════════════════════════════════════════════════

def _build_laplacian_pyramid_torch(img: torch.Tensor, levels: int) -> List[torch.Tensor]:
    """Build a Laplacian pyramid from (B, C, H, W) image.

    Returns list of tensors: levels Laplacian layers + 1 residual (coarsest).
    Each Laplacian layer = current - upsample(downsample(current)).
    """
    pyramid: List[torch.Tensor] = []
    current = img
    for i in range(levels):
        h, w = current.shape[2], current.shape[3]
        # Downsample by 2x with Gaussian pre-filter
        down = _gaussian_blur_2d(current, sigma=1.0)
        down = F.interpolate(down, size=(max(1, h // 2), max(1, w // 2)), mode="bilinear", align_corners=False)
        # Upsample back
        up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
        # Laplacian = difference between current and upsampled-downsampled
        laplacian = current - up
        pyramid.append(laplacian)
        current = down
    # Residual (coarsest level)
    pyramid.append(current)
    return pyramid


def _reconstruct_from_pyramid(pyramid: List[torch.Tensor]) -> torch.Tensor:
    """Reconstruct image from Laplacian pyramid. Returns (B, C, H, W)."""
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        h, w = pyramid[i].shape[2], pyramid[i].shape[3]
        up = F.interpolate(current, size=(h, w), mode="bilinear", align_corners=False)
        current = up + pyramid[i]
    return current


def _laplacian_pyramid_blend(img_a: torch.Tensor, img_b: torch.Tensor,
                              blend_mask: torch.Tensor, levels: int = 5) -> torch.Tensor:
    """Blend two (B, C, H, W) images using Laplacian pyramid blending.

    blend_mask: (B, 1, H, W) — 0.0 = use img_a, 1.0 = use img_b
    """
    # Clamp levels based on image size
    min_dim = min(img_a.shape[2], img_a.shape[3])
    max_levels = max(1, int(math.log2(max(min_dim, 1))))
    levels = min(levels, max_levels)

    pyr_a = _build_laplacian_pyramid_torch(img_a, levels)
    pyr_b = _build_laplacian_pyramid_torch(img_b, levels)

    # Build Gaussian pyramid for the mask
    mask_pyr: List[torch.Tensor] = []
    current_mask = blend_mask
    for i in range(levels):
        mask_pyr.append(current_mask)
        h, w = current_mask.shape[2], current_mask.shape[3]
        current_mask = F.interpolate(current_mask, size=(max(1, h // 2), max(1, w // 2)),
                                     mode="bilinear", align_corners=False)
    mask_pyr.append(current_mask)

    # Blend each level
    blended_pyr: List[torch.Tensor] = []
    for i in range(len(pyr_a)):
        m = mask_pyr[i]
        blended = (1.0 - m) * pyr_a[i] + m * pyr_b[i]
        blended_pyr.append(blended)

    return _reconstruct_from_pyramid(blended_pyr)


# ══════════════════════════════════════════════════════════════════════
#  Frequency blend — FFT-domain blending
# ══════════════════════════════════════════════════════════════════════

def _frequency_blend(img_a: torch.Tensor, img_b: torch.Tensor,
                     blend_mask: torch.Tensor) -> torch.Tensor:
    """FFT-based frequency domain blending of two (B, C, H, W) images.

    blend_mask: (B, 1, H, W) — 0.0 = use img_a, 1.0 = use img_b
    """
    # Compute FFT of both images
    fft_a = torch.fft.rfft2(img_a)
    fft_b = torch.fft.rfft2(img_b)

    B, C, H, W = img_a.shape
    freq_h, freq_w = fft_a.shape[2], fft_a.shape[3]

    # Build low-pass and high-pass versions of the blend mask in spatial domain
    low_mask = _gaussian_blur_2d(blend_mask, sigma=max(H, W) * 0.1)
    high_mask = blend_mask

    # Map spatial masks to frequency domain shape
    low_mask_freq = F.interpolate(low_mask, size=(freq_h, freq_w), mode="bilinear", align_corners=False)
    high_mask_freq = F.interpolate(high_mask, size=(freq_h, freq_w), mode="bilinear", align_corners=False)

    # Create radial frequency coordinate
    fy = torch.arange(freq_h, device=img_a.device, dtype=torch.float32) / max(freq_h, 1)
    fx = torch.arange(freq_w, device=img_a.device, dtype=torch.float32) / max(freq_w, 1)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
    freq_radius = (fy_grid.pow(2) + fx_grid.pow(2)).sqrt()
    max_r = freq_radius.max()
    if max_r > 0:
        freq_radius = freq_radius / max_r
    freq_radius = freq_radius.unsqueeze(0).unsqueeze(0)  # (1,1,fH,fW)

    # Interpolate: at low freq use low_mask_freq, at high freq use high_mask_freq
    freq_weight = freq_radius.clamp(0.0, 1.0)
    final_freq_mask = (1.0 - freq_weight) * low_mask_freq + freq_weight * high_mask_freq

    # Blend in frequency domain
    fft_blended = (1.0 - final_freq_mask) * fft_a + final_freq_mask * fft_b

    # Inverse FFT
    result = torch.fft.irfft2(fft_blended, s=(H, W))
    return result.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Video stable bbox — union of all frame masks
# ══════════════════════════════════════════════════════════════════════

def _compute_bbox_single(mask_2d: torch.Tensor) -> Tuple[int, int, int, int]:
    """Compute (x, y, w, h) bounding box of nonzero region in a 2D mask.

    Returns (-1, -1, -1, -1) if mask is empty.
    """
    nonzero = torch.nonzero(mask_2d > 0.5, as_tuple=False)
    if nonzero.numel() == 0:
        return (-1, -1, -1, -1)
    y_min = nonzero[:, 0].min().item()
    y_max = nonzero[:, 0].max().item()
    x_min = nonzero[:, 1].min().item()
    x_max = nonzero[:, 1].max().item()
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def _compute_stable_bbox(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """Compute the union bounding box across all frames.

    mask: (B, H, W) float32
    Returns: (x, y, w, h) — single bbox covering all frames' mask regions.
    """
    B, H, W = mask.shape
    union_x_min, union_y_min = W, H
    union_x_max, union_y_max = 0, 0
    any_valid = False

    for b in range(B):
        x, y, w, h = _compute_bbox_single(mask[b])
        if x < 0:
            continue
        any_valid = True
        union_x_min = min(union_x_min, x)
        union_y_min = min(union_y_min, y)
        union_x_max = max(union_x_max, x + w)
        union_y_max = max(union_y_max, y + h)

    if not any_valid:
        return (-1, -1, -1, -1)

    return (union_x_min, union_y_min,
            union_x_max - union_x_min, union_y_max - union_y_min)


# ══════════════════════════════════════════════════════════════════════
#  Temporal Gaussian smoothing along batch dimension
# ══════════════════════════════════════════════════════════════════════

def _temporal_gaussian_smooth(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooth a (B, H, W) mask along the batch (temporal) dimension.

    Applies 1D Gaussian convolution along dim=0 independently per spatial pixel.
    Returns (B, H, W).
    """
    B, H, W = mask.shape
    if B <= 1 or sigma <= 0:
        return mask
    device = _get_device(mask)
    k1d = _gauss_kernel_1d(sigma, device, mask.dtype)  # (K,)
    K = len(k1d)
    pad = K // 2

    # Reshape: (B, H*W) → (H*W, 1, B) for conv1d
    flat = mask.reshape(B, H * W).permute(1, 0)  # (H*W, B)
    flat = flat.unsqueeze(1)  # (H*W, 1, B)

    kernel = k1d.view(1, 1, -1)  # (1, 1, K)
    padded = F.pad(flat, (pad, pad), mode="replicate")
    smoothed = F.conv1d(padded, kernel)  # (H*W, 1, B)

    # Reshape back: (H*W, 1, B) → (B, H, W)
    smoothed = smoothed.squeeze(1).permute(1, 0).reshape(B, H, W)
    return smoothed.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Morphological helpers — fill holes and remove small regions (torch)
# ══════════════════════════════════════════════════════════════════════

def _fill_holes_torch(mask: torch.Tensor) -> torch.Tensor:
    """Fill interior holes in a (B, H, W) binary mask.

    Uses cv2 contour hierarchy if available, else morphological closing as fallback.
    """
    B, H, W = mask.shape
    device = _get_device(mask)
    binary = (mask > 0.5).float()

    if HAS_CV2:
        results = []
        for b in range(B):
            mask_np = binary[b].cpu().numpy().astype(np.uint8) * 255
            contours, hierarchy = cv2.findContours(
                mask_np, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            if hierarchy is not None:
                for i, h in enumerate(hierarchy[0]):
                    if h[3] >= 0:
                        cv2.drawContours(mask_np, contours, i, 255, -1)
            results.append(torch.from_numpy((mask_np > 127).astype(np.float32)))
        return torch.stack(results, dim=0).to(device)
    else:
        # Torch fallback: morphological closing with increasing kernel sizes
        result = binary.unsqueeze(1)  # (B, 1, H, W)
        for k_size in [3, 5, 7, 11]:
            pad = k_size // 2
            # Dilation (max_pool)
            dilated = F.max_pool2d(result, kernel_size=k_size, stride=1, padding=pad)
            # Erosion (-max_pool(-x))
            eroded = -F.max_pool2d(-dilated, kernel_size=k_size, stride=1, padding=pad)
            result = eroded
        return result.squeeze(1).clamp(0.0, 1.0)


def _remove_small_regions_torch(mask: torch.Tensor, min_area: int) -> torch.Tensor:
    """Remove connected components smaller than min_area in (B, H, W) mask.

    Uses cv2 connectedComponents if available, else erosion/dilation approximation.
    """
    if min_area <= 0:
        return mask
    B, H, W = mask.shape
    device = _get_device(mask)

    if HAS_CV2:
        results = []
        for b in range(B):
            binary = (mask[b].cpu().numpy() > 0.5).astype(np.uint8)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            filtered = np.zeros_like(binary)
            for i in range(1, n_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    filtered[labels == i] = 1
            results.append(torch.from_numpy(filtered.astype(np.float32)))
        return torch.stack(results, dim=0).to(device)
    else:
        # Torch fallback: erode then dilate to remove small blobs
        k_size = max(3, int(math.sqrt(min_area)))
        if k_size % 2 == 0:
            k_size += 1
        pad = k_size // 2
        m4 = mask.unsqueeze(1)
        eroded = -_max_pool_square(-m4, k_size, pad)
        dilated = _max_pool_square(eroded, k_size, pad)
        return dilated.squeeze(1).clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Color matching — mean+std transfer
# ══════════════════════════════════════════════════════════════════════

def _color_match_mean_std(source: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    """Match source colors to target using mean+std transfer within masked region.

    source: (B, H, W, C) — the inpainted result to adjust
    target: (B, H, W, C) — the original image reference
    mask: (B, H, W) — region where color matching applies
    Returns adjusted source: (B, H, W, C)
    """
    result = source.clone()
    B = source.shape[0]
    binary = (mask > 0.5).float()

    for b in range(B):
        region_mask = binary[b]  # (H, W)
        if region_mask.sum() < 10:
            continue
        m3 = region_mask.unsqueeze(-1)  # (H, W, 1)

        src_pixels = source[b] * m3
        tgt_pixels = target[b] * m3
        n_pixels = region_mask.sum().clamp(min=1)

        src_mean = src_pixels.sum(dim=(0, 1)) / n_pixels
        tgt_mean = tgt_pixels.sum(dim=(0, 1)) / n_pixels

        src_var = ((source[b] - src_mean) * m3).pow(2).sum(dim=(0, 1)) / n_pixels
        tgt_var = ((target[b] - tgt_mean) * m3).pow(2).sum(dim=(0, 1)) / n_pixels

        src_std = src_var.sqrt().clamp(min=1e-6)
        tgt_std = tgt_var.sqrt().clamp(min=1e-6)

        adjusted = (source[b] - src_mean) * (tgt_std / src_std) + tgt_mean
        result[b] = source[b] * (1.0 - m3) + adjusted * m3

    return result.clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Inpaint mask mode application
# ══════════════════════════════════════════════════════════════════════

def _apply_inpaint_mask_mode(mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert mask to the format expected by the inpaint model.

    mask: (B, H, W) float32
    mode: 'hard_binary' | 'slight_feather' | 'soft_blend'
    Returns: (B, H, W)
    """
    if mode == "hard_binary":
        return (mask > 0.5).float()
    elif mode == "slight_feather":
        binary = (mask > 0.5).float()
        feathered = _gaussian_blur_mask(binary, sigma=1.5)
        return feathered.clamp(0.0, 1.0)
    elif mode == "soft_blend":
        return _gaussian_blur_mask(mask, sigma=3.0).clamp(0.0, 1.0)
    else:
        return (mask > 0.5).float()


# ══════════════════════════════════════════════════════════════════════
#  Blend mask generation dispatcher
# ══════════════════════════════════════════════════════════════════════

def _generate_stitch_blend_mask_video_stable(
    mask: torch.Tensor,
    spatial_dilate_px: int = 24,
    spatial_blur_sigma: float = 18.0,
    temporal_sigma: float = 3.0,
) -> torch.Tensor:
    """Stitch blend mask designed for video with jittery segmentation.

    Pipeline (composition order matters):
      1. Binarize to get a clean starting boundary.
      2. Morphological dilation — push the blend zone outward into the
         visually flat background, away from the noisy segmentation edge.
      3. Temporal Gaussian smooth (along batch dim) — kills frame-to-frame
         jitter in boundary position. Re-binarize so the spatial blur in
         step 4 produces a clean gradient instead of a triple-feathered one.
      4. Wide spatial Gaussian feather — soft seam invisible against the
         background that the dilation pushed the blend zone into.

    The model-facing inpaint_mask is NEVER touched by this — it stays sharp.

    mask: (B, H, W)
    Returns: (B, H, W) soft blend mask, range [0, 1].
    """
    if mask.dim() != 3:
        raise ValueError(f"video_stable expects (B,H,W) mask, got {mask.shape}")
    B = mask.shape[0]
    binary = (mask > 0.5).float()

    # Step 1: dilate (push blend zone into background)
    if spatial_dilate_px > 0:
        k = int(spatial_dilate_px) * 2 + 1
        pad = int(spatial_dilate_px)
        m4 = binary.unsqueeze(1)  # (B, 1, H, W)
        dilated = _max_pool_square(m4, k, pad)
        binary = dilated.squeeze(1).clamp(0.0, 1.0)

    # Step 2: temporal smooth → re-binarize (kills jitter at the boundary)
    if temporal_sigma > 0 and B > 1:
        smoothed = _temporal_gaussian_smooth(binary, sigma=float(temporal_sigma))
        # Re-binarize at 0.3 (lower than 0.5) so the blend zone stays slightly
        # generous — better to err toward generated content covering the seam
        # than the original frame leaking through.
        binary = (smoothed > 0.3).float()

    # Step 3: wide spatial Gaussian feather
    blend = _gaussian_blur_mask(binary, sigma=float(spatial_blur_sigma))
    return blend.clamp(0.0, 1.0)


def _generate_stitch_blend_mask(image: torch.Tensor, mask: torch.Tensor,
                                 mode: str, radius: int,
                                 video_stable_temporal_sigma: float = 3.0,
                                 video_stable_dilate_px: int = -1,
                                 video_stable_blur_sigma: float = -1.0) -> torch.Tensor:
    """Generate the stitch blend mask based on the selected mode.

    image: (B, H, W, C)
    mask: (B, H, W)
    mode: 'edge_aware' | 'gaussian' | 'laplacian_pyramid' | 'frequency_blend' | 'video_stable' | 'exact'
    radius: feather radius
    video_stable_*: only used when mode == 'video_stable'. dilate/blur defaults
    of -1 mean "derive from radius" (dilate=radius, blur=radius*0.75).
    Returns: (B, H, W) soft blend mask
    """
    if mode == "edge_aware":
        return _edge_aware_blend_mask(image, mask, radius)
    elif mode == "gaussian":
        binary = (mask > 0.5).float()
        return _gaussian_blur_mask(binary, sigma=radius * 0.4)
    elif mode == "laplacian_pyramid":
        binary = (mask > 0.5).float()
        return _gaussian_blur_mask(binary, sigma=radius * 0.5)
    elif mode == "frequency_blend":
        binary = (mask > 0.5).float()
        return _gaussian_blur_mask(binary, sigma=radius * 0.6)
    elif mode == "video_stable":
        # Jitter-tolerant compositing for video: dilate → temporal smooth →
        # wide feather. Keeps the model-facing inpaint mask untouched.
        dilate = int(video_stable_dilate_px) if video_stable_dilate_px >= 0 else int(radius)
        blur = float(video_stable_blur_sigma) if video_stable_blur_sigma >= 0 else float(radius) * 0.75
        return _generate_stitch_blend_mask_video_stable(
            mask,
            spatial_dilate_px=dilate,
            spatial_blur_sigma=blur,
            temporal_sigma=float(video_stable_temporal_sigma),
        )
    elif mode == "exact":
        # Hard-binary alpha — no feather, no leakage outside the mask.
        return (mask > 0.5).float()
    else:
        binary = (mask > 0.5).float()
        return _gaussian_blur_mask(binary, sigma=radius * 0.4)


# ══════════════════════════════════════════════════════════════════════
#  Size mode helpers
# ══════════════════════════════════════════════════════════════════════

def _apply_size_mode(crop_w: int, crop_h: int, size_mode: str,
                     forced_width: int, forced_height: int,
                     min_size: int, max_size: int,
                     padding_multiple: int) -> Tuple[int, int]:
    """Compute the final output dimensions based on size_mode."""
    if size_mode == "forced_size":
        tw, th = forced_width, forced_height
    elif size_mode == "ranged_size":
        aspect = crop_w / max(crop_h, 1)
        tw, th = crop_w, crop_h
        if tw < min_size or th < min_size:
            if tw < th:
                tw = min_size
                th = max(min_size, int(tw / aspect))
            else:
                th = min_size
                tw = max(min_size, int(th * aspect))
        if tw > max_size or th > max_size:
            if tw > th:
                tw = max_size
                th = max(1, int(tw / aspect))
            else:
                th = max_size
                tw = max(1, int(th * aspect))
        tw = max(tw, min_size)
        th = max(th, min_size)
    else:
        tw, th = crop_w, crop_h

    if padding_multiple > 1:
        tw = int(math.ceil(tw / padding_multiple) * padding_multiple)
        th = int(math.ceil(th / padding_multiple) * padding_multiple)

    tw = max(tw, padding_multiple)
    th = max(th, padding_multiple)
    return (tw, th)


# ══════════════════════════════════════════════════════════════════════
#  Resize helpers (IMAGE and MASK)
# ══════════════════════════════════════════════════════════════════════

def _resize_image(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize (B, H, W, C) image to (B, target_h, target_w, C) using bilinear."""
    if image.shape[1] == target_h and image.shape[2] == target_w:
        return image
    img_bchw = image.permute(0, 3, 1, 2)
    resized = F.interpolate(img_bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def _resize_mask(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize (B, H, W) mask to (B, target_h, target_w) using bilinear."""
    if mask.shape[1] == target_h and mask.shape[2] == target_w:
        return mask
    m4 = mask.unsqueeze(1)
    resized = F.interpolate(m4, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized.squeeze(1)


# ══════════════════════════════════════════════════════════════════════
#  Multi-algorithm resize (lquesada-compatible: every option distinct)
# ══════════════════════════════════════════════════════════════════════

_TORCH_INTERP = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "area": "area",
}
_CV2_INTERP = {
    "nearest": 0,    # cv2.INTER_NEAREST
    "bilinear": 1,   # cv2.INTER_LINEAR
    "bicubic": 2,    # cv2.INTER_CUBIC
    "lanczos": 4,    # cv2.INTER_LANCZOS4
    "box":     3,    # cv2.INTER_AREA
    "hamming": 3,    # cv2.INTER_AREA (closest)
    "area":    3,    # cv2.INTER_AREA
}


def _resize_image_alg(image: torch.Tensor, target_h: int, target_w: int, alg: str) -> torch.Tensor:
    """Resize (B,H,W,C) image with the requested algorithm. Distinct paths per name."""
    if image.shape[1] == target_h and image.shape[2] == target_w:
        return image
    if alg in _TORCH_INTERP:
        mode = _TORCH_INTERP[alg]
        img_bchw = image.permute(0, 3, 1, 2).contiguous()
        if mode in ("nearest", "area"):
            out = F.interpolate(img_bchw, size=(target_h, target_w), mode=mode)
        else:
            out = F.interpolate(img_bchw, size=(target_h, target_w), mode=mode, align_corners=False)
        return out.permute(0, 2, 3, 1).contiguous()
    # lanczos / box / hamming -> cv2 if available, else bicubic torch fallback
    if HAS_CV2 and alg in _CV2_INTERP:
        out = []
        for b in range(image.shape[0]):
            arr = (image[b].clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
            res = cv2.resize(arr, (target_w, target_h), interpolation=_CV2_INTERP[alg])
            out.append(torch.from_numpy(res).to(image.device, image.dtype) / 255.0)
        return torch.stack(out, dim=0)
    img_bchw = image.permute(0, 3, 1, 2).contiguous()
    out = F.interpolate(img_bchw, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return out.permute(0, 2, 3, 1).contiguous()


def _resize_mask_alg(mask: torch.Tensor, target_h: int, target_w: int, alg: str) -> torch.Tensor:
    """Resize (B,H,W) mask with the requested algorithm."""
    if mask.shape[1] == target_h and mask.shape[2] == target_w:
        return mask
    if alg in _TORCH_INTERP:
        mode = _TORCH_INTERP[alg]
        m4 = mask.unsqueeze(1)
        if mode in ("nearest", "area"):
            out = F.interpolate(m4, size=(target_h, target_w), mode=mode)
        else:
            out = F.interpolate(m4, size=(target_h, target_w), mode=mode, align_corners=False)
        return out.squeeze(1).clamp(0.0, 1.0)
    if HAS_CV2 and alg in _CV2_INTERP:
        out = []
        for b in range(mask.shape[0]):
            arr = (mask[b].clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
            res = cv2.resize(arr, (target_w, target_h), interpolation=_CV2_INTERP[alg])
            out.append(torch.from_numpy(res).to(mask.device, mask.dtype) / 255.0)
        return torch.stack(out, dim=0).clamp(0.0, 1.0)
    m4 = mask.unsqueeze(1)
    out = F.interpolate(m4, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return out.squeeze(1).clamp(0.0, 1.0)


def _pad_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return x
    return int(math.ceil(x / m) * m)


def _hipass_filter(mask: torch.Tensor, threshold: float) -> torch.Tensor:
    """Zero out values below threshold; keep above unchanged. Real distinct from binary."""
    if threshold <= 0.0:
        return mask
    keep = (mask >= threshold).float()
    return (mask * keep).clamp(0.0, 1.0)


def _max_pool_square(m4: torch.Tensor, k: int, pad: int) -> torch.Tensor:
    """F.max_pool2d(m4, k, stride=1, padding=pad) with a square kernel, as two 1-D passes (L7.99). A square
    window's max is the max of its rows' maxima, so the result is bit-identical; the work is 2k instead of k*k per
    pixel (mask_blend_pixels=32 on a 1080p frame: 57 s -> 2 s on CPU, measured)."""
    out = F.max_pool2d(m4, kernel_size=(k, 1), stride=1, padding=(pad, 0))
    return F.max_pool2d(out, kernel_size=(1, k), stride=1, padding=(0, pad))


def _expand_mask_pixels(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """Max-pool dilation by `pixels` (lquesada-compatible)."""
    if pixels <= 0:
        return mask
    k = 2 * pixels + 1
    m4 = mask.unsqueeze(1)
    out = _max_pool_square(m4, k, pixels)
    return out.squeeze(1).clamp(0.0, 1.0)


def _edge_replicate_pad(image: torch.Tensor, top: int, bottom: int, left: int, right: int) -> torch.Tensor:
    """Pad (B,H,W,C) image with edge replication (BCHW replicate)."""
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    img_bchw = image.permute(0, 3, 1, 2).contiguous()
    out = F.pad(img_bchw, (left, right, top, bottom), mode="replicate")
    return out.permute(0, 2, 3, 1).contiguous()


# ══════════════════════════════════════════════════════════════════════
#  Legacy blend-mode routing — every name produces a genuinely different,
#  working blend. No aliases collapse to the same engine.
#
#    poisson       → edge_aware        (Sobel-guided seamless boundary)
#    poisson_mixed → laplacian_pyramid  (multi-level frequency mixing)
#    exact         → exact              (hard-binary alpha, no leakage)
#    linear_exact  → exact              (same hard-binary cut)
#    classic       → gaussian           (lquesada-style soft feather)
# ══════════════════════════════════════════════════════════════════════

_LEGACY_BLEND_ALIASES = {
    "poisson":       "edge_aware",
    "poisson_mixed": "laplacian_pyramid",
    "exact":         "exact",
    "linear_exact":  "exact",
    "classic":       "gaussian",
}

def _normalize_blend_mode(mode: str) -> str:
    return _LEGACY_BLEND_ALIASES.get(mode, mode)


# ══════════════════════════════════════════════════════════════════════
#  L7.99: the crop preview is drawn from the EXECUTED window (owner A11)
# ══════════════════════════════════════════════════════════════════════

_PREVIEW_LONG_SIDE = 768


def _fit_window_axis(pos: int, size: int, limit: int) -> int:
    """Place a crop window on one axis. A window that fits stays inside the image (moved the least distance, so it
    shows real pixels instead of smeared edge padding); a larger one covers the whole axis and is padded."""
    lo, hi = min(0, limit - size), max(0, limit - size)
    return int(min(max(pos, lo), hi))


def _crop_plan_record(stitcher: dict, image_hw, out_hw, stable: bool, frames: int) -> dict:
    """What ran, per frame, in processed-image coordinates (after pre-resize / outpaint extension)."""
    recs = []
    for j in range(len(stitcher["cropped_to_canvas_x"])):
        x = int(stitcher["cropped_to_canvas_x"][j] - stitcher["canvas_to_orig_x"][j])
        y = int(stitcher["cropped_to_canvas_y"][j] - stitcher["canvas_to_orig_y"][j])
        w = int(stitcher["cropped_to_canvas_w"][j])
        h = int(stitcher["cropped_to_canvas_h"][j])
        recs.append({"window": [x, y, w, h],
                     "padding": {"left": max(0, -x), "top": max(0, -y),
                                 "right": max(0, x + w - int(image_hw[1])), "bottom": max(0, y + h - int(image_hw[0]))}})
    return {"image_size": [int(image_hw[1]), int(image_hw[0])], "output_size": [int(out_hw[1]), int(out_hw[0])],
            "frames_total": int(frames), "stable": bool(stable), "frames": recs}


def _draw_crop_preview(frame_hwc: torch.Tensor, regen_mask_hw: torch.Tensor, window, cropped_hwc: torch.Tensor,
                       lines) -> "np.ndarray":
    """ONE picture: the original frame (discarded area dimmed, regenerated area tinted, the window outlined - orange
    on a side where it runs past the image and is padded) next to the crop the sampler gets, with a label under both.
    One image, not two: a node preview holding two images reads as a 2-frame clip (the frame scrubber picked it up).
    Returns RGB float [0, 1]."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    def clean(t):
        return np.nan_to_num(t[..., :3].detach().float().cpu().numpy(), nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)

    img = clean(frame_hwc)
    m = (regen_mask_hw.detach().float().cpu().numpy() > 0.5)
    Hh, Ww = img.shape[:2]
    x, y, w, h = (int(v) for v in window)
    keep = np.zeros((Hh, Ww), bool)
    keep[max(0, y):min(Hh, y + h), max(0, x):min(Ww, x + w)] = True
    out = img.copy()
    out[~keep] *= 0.30                                                   # discarded surroundings
    tint = np.array([1.0, 0.25, 0.35], np.float32)
    out[m & keep] = out[m & keep] * 0.55 + tint * 0.45                   # regenerated (masked) area
    s = min(1.0, _PREVIEW_LONG_SIDE / float(max(Hh, Ww)))
    pw, ph = max(1, int(round(Ww * s))), max(1, int(round(Hh * s)))
    ov = Image.fromarray((out * 255 + 0.5).astype("uint8")).resize((pw, ph), Image.BILINEAR)
    d = ImageDraw.Draw(ov)
    x0, y0 = max(0, x) * s, max(0, y) * s
    x1, y1 = min(Ww, x + w) * s - 1, min(Hh, y + h) * s - 1
    ok, pad = (90, 220, 120), (255, 160, 40)
    lw = max(2, pw // 256)
    d.line([(x0, y0), (x1, y0)], fill=pad if y < 0 else ok, width=lw)
    d.line([(x0, y1), (x1, y1)], fill=pad if y + h > Hh else ok, width=lw)
    d.line([(x0, y0), (x0, y1)], fill=pad if x < 0 else ok, width=lw)
    d.line([(x1, y0), (x1, y1)], fill=pad if x + w > Ww else ok, width=lw)
    crop = clean(cropped_hwc)
    ch, cw = crop.shape[:2]
    rw = max(1, int(round(cw * ph / float(max(1, ch)))))
    res = Image.fromarray((crop * 255 + 0.5).astype("uint8")).resize((rw, ph), Image.BILINEAR)
    gap = max(6, pw // 64)
    total_w = pw + gap + rw
    fsize = max(14, total_w // 34)                                       # stays readable when the node shrinks it
    try:
        font = ImageFont.load_default(size=fsize)
    except TypeError:                                                    # Pillow < 10.1
        font = ImageFont.load_default()
    bar = int(fsize * 1.35) * len(lines) + fsize // 2
    canvas = Image.new("RGB", (total_w, ph + bar), (18, 20, 26))
    canvas.paste(ov, (0, 0))
    canvas.paste(res, (pw + gap, 0))
    dc = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        dc.text((fsize // 2, ph + fsize // 4 + i * int(fsize * 1.35)), line, fill=(225, 228, 235), font=font)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _crop_preview_lines(plan: dict, fr: dict, k: int, B: int) -> list:
    """The label under the preview (shared by the run preview and the live one)."""
    x, y, w, h = fr["window"]
    ow, oh = plan["output_size"]
    scale = ow / float(max(1, w))
    where = (f"one window for all {B} frames (their union), frame {k} shown" if plan["stable"] and B > 1
             else (f"own window per frame, frame {k} of {B} shown" if B > 1 else "single image"))
    pads = ", ".join(f"{side} {v}" for side, v in fr["padding"].items() if v)
    return [f"crop {w}x{h} at ({x}, {y}) of {plan['image_size'][0]}x{plan['image_size'][1]} -> {ow}x{oh} "
            f"(x{scale:.2f})",
            where + (f"; padded {pads}" if pads else "")]


def _save_preview(frame_hwc, regen_mask_hw, window, cropped_hwc, lines, plan: dict):
    """Draw the preview picture and save it for core's node preview. Returns (ui, picture); a picture that could not
    be drawn gives ({}, None) - a preview must never fail the crop."""
    try:
        pic = _draw_crop_preview(frame_hwc, regen_mask_hw, window, cropped_hwc, lines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[InpaintCropPro] preview not drawn: %s", exc)
        return {}, None
    try:
        from ._video_comparer_io import save_preview_png
        saved = save_preview_png(pic, "inpaint_crop_preview")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[InpaintCropPro] preview not saved: %s", exc)
        saved = None
    return ({"images": [saved], "c2c_crop_plan": [plan]} if saved else {}), pic


def _crop_preview_ui(image, regen_mask, cropped, plan: dict, preview_frame: int):
    """Core's node preview (ui.images, both renderers): one picture of one frame - the window on the original next
    to the crop. Display only; it never changes what is cropped. Returns (ui, picture) - the picture is also the
    node's `preview` output (L7.65 P08)."""
    B = int(image.shape[0])
    k = max(0, min(int(preview_frame), B - 1))
    fr = plan["frames"][k if len(plan["frames"]) > 1 else 0]
    lines = _crop_preview_lines(plan, fr, k, B)
    return _save_preview(image[k], regen_mask[k], fr["window"], cropped[min(k, int(cropped.shape[0]) - 1)], lines,
                         plan)


# ── L7.65 P08: the crop modes "smart aspect" and "mask prepare" ────────────────────────────────────────────────────

CROP_MODES = ["pro", "smart aspect", "mask prepare"]
STITCH_MODES = ["blend", "paste back", "smart", "context blend"]
# The former nodes' widgets, in their order, under a prefix (INPUT_TYPES builds them from the engines' own specs;
# tests check the two agree)
_SMART_CROP_KEYS = ("smart_resolution_mode", "smart_max_resolution", "smart_min_resolution", "smart_manual_width",
                    "smart_manual_height", "smart_mask_grow_pixels", "smart_patch_mask_holes", "smart_no_mask_mode",
                    "smart_force_divisibility")
_PREPARE_KEYS = ("prepare_fill_holes", "prepare_remove_small_regions", "prepare_min_region_area",
                 "prepare_grow_pixels", "prepare_inpaint_edge_mode", "prepare_stitch_edge_mode",
                 "prepare_stitch_feather_radius", "prepare_temporal_smooth", "prepare_temporal_sigma")
_PASTE_KEYS = ("paste_upscale_method", "paste_feather_edges", "paste_feather_radius")
_SMART_STITCH_KEYS = ("smart_feather_pixels", "smart_blend_mode", "smart_resize_full_image_output",
                      "smart_enable_color_match", "smart_color_match_amount")
_CONTEXT_KEYS = ("context_crop_padding", "context_blend_softness", "context_mask_expansion_blend",
                 "context_enable_color_correction", "context_enable_lightness_rescue",
                 "context_enable_differential_diffusion", "context_sampling_mask_blur_size",
                 "context_sampling_mask_blur_strength", "context_color_reference")


def _mode_inputs(spec: dict, prefix: str, tag: str, skip=()) -> dict:
    """A former node's widgets as one mode's inputs: same type, options and default, the name prefixed, the tooltip
    tagged with the mode. No c2c_section / advanced flag: js/c2c_mode_widgets.js shows them with their mode."""
    out = {}
    for name, s in spec.items():
        if name in skip:
            continue
        opts = dict(s[1]) if len(s) > 1 and isinstance(s[1], dict) else {}
        tip = opts.get("tooltip")
        opts["tooltip"] = f"[{tag}] " + (tip if tip else name.replace("_", " "))
        out[prefix + name] = (list(s[0]) if isinstance(s[0], (list, tuple)) else s[0], opts)
    return out


def _unprefix(values: dict, keys, prefix: str) -> dict:
    """The engine's keyword arguments from one mode's prefixed inputs. A key a save does not carry (an input added
    later, e.g. context_color_reference) is left out, so the engine's own default applies."""
    return {k[len(prefix):]: values[k] for k in keys if k in values}


_MODE_DEFAULTS: dict = {}


def _mode_defaults(node_cls) -> dict:
    """Defaults of the mode inputs (from INPUT_TYPES, so a call that omits them gets what the UI would send)."""
    if node_cls.__name__ not in _MODE_DEFAULTS:
        spec = node_cls.INPUT_TYPES()
        vals = {}
        for group in ("required", "optional"):
            for name, s in (spec.get(group) or {}).items():
                opts = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
                if "default" in opts:
                    vals[name] = opts["default"]
                elif isinstance(s[0], (list, tuple)) and s[0]:
                    vals[name] = s[0][0]
        _MODE_DEFAULTS[node_cls.__name__] = vals
    return _MODE_DEFAULTS[node_cls.__name__]


def _roto_erode(mask: torch.Tensor) -> torch.Tensor:
    """Roto-Sync's 1 px pre-erosion of the inpaint mask (shared by the crop and the live preview). Padding with 1
    treats outside as foreground, so a subject touching the frame edge is not eaten; min-pool = -max_pool(-x)."""
    _m = mask
    if _m.dim() == 2:
        _m = _m.unsqueeze(0)
    _m4 = _m.unsqueeze(1)
    _m4_pad = F.pad(_m4, (1, 1, 1, 1), mode="constant", value=1.0)
    _neg = -_m4_pad
    _pooled = F.max_pool2d(_neg, kernel_size=3, stride=1, padding=0)
    return (-_pooled).squeeze(1).clamp(0.0, 1.0)


def _window_record(window, W: int, H: int) -> dict:
    x, y, w, h = (int(v) for v in window)
    return {"window": [x, y, w, h],
            "padding": {"left": max(0, -x), "top": max(0, -y), "right": max(0, x + w - W), "bottom": max(0, y + h - H)}}


def _smart_plan_record(st: dict, B: int, H: int, W: int) -> dict:
    """The plan of a crop in "smart aspect" mode, from its stitcher (what ran) or from plan_crop (the same dict)."""
    if st.get("bypass") or not st.get("x"):
        frames = [_window_record((0, 0, W, H), W, H) for _ in range(int(B))]
        out = [int(W), int(H)]
    else:
        frames = [_window_record((st["x"][j], st["y"][j], st["w"][j], st["h"][j]), W, H) for j in range(len(st["x"]))]
        out = [int(st["target_w"][0]), int(st["target_h"][0])]
    return {"mode": "smart aspect", "image_size": [int(W), int(H)], "output_size": out, "frames_total": int(B),
            "stable": False, "frames": frames}


def _smart_preview_line(st: dict) -> str:
    if st.get("bypass"):
        return "smart aspect: mask empty, image passed through (Bypass)"
    if st.get("full_image_resize"):
        return "smart aspect: mask empty, whole frame resized"
    return "smart aspect: one output size for every frame"


def _prepare_plan_record(B: int, H: int, W: int) -> dict:
    """"mask prepare" does not crop: the window is the whole frame."""
    return {"mode": "mask prepare", "image_size": [int(W), int(H)], "output_size": [int(W), int(H)],
            "frames_total": int(B), "stable": True, "frames": [_window_record((0, 0, W, H), W, H) for _ in range(int(B))]}


def _prepare_preview_lines(plan: dict, k: int, B: int) -> list:
    W, H = plan["image_size"]
    return [f"no crop: masks prepared for the whole {W}x{H} frame (right: R inpaint mask, G stitch blend mask)",
            f"frame {k} of {B} shown" if B > 1 else "single image"]


def _fit_mask_hw(m: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """A (h, w) mask at (H, W), nearest (for the preview tint only)."""
    if tuple(m.shape[-2:]) == (H, W):
        return m
    return F.interpolate(m.float()[None, None], size=(H, W), mode="nearest")[0, 0]


# L7.99 slice 3 (ORDERS 11.4: at most 10 visible controls): the rarely used inputs are flagged core's "advanced" (Nodes
# 2.0 folds them behind "Show advanced inputs") and named a "c2c_section" for the classic canvas's "More" bar
# (js/_c2c_sections.js). Inputs a switch reveals (pre-resize sizes, outpaint factors, target size, video_stable_*) are
# not here: js/c2c_mode_widgets.js shows them when their switch is on. Order and defaults are unchanged.
_CROP_SECTIONS = {
    **dict.fromkeys(("mask_fill_holes", "mask_invert", "mask_hipass_filter", "wan_mask_polarity"), "Mask"),
    **dict.fromkeys(("output_resize_to_target_size", "output_padding", "wan_align_multiple", "auto_context_factor",
                     "downscale_algorithm", "upscale_algorithm"), "Size"),
    **dict.fromkeys(("preresize", "extend_for_outpainting"), "Resize / outpaint"),
    **dict.fromkeys(("wan_temporal_smooth_frames",), "Video"),
    **dict.fromkeys(("inpaint_mask_mode", "blend_radius", "fill_masked_area", "roto_quality"), "Stitch"),
    **dict.fromkeys(("device_mode",), "Device"),
}


# The settings _geometry() reads; _LIVE_KEYS = every setting the live preview may change
_GEOMETRY_KEYS = (
    "downscale_algorithm", "upscale_algorithm", "preresize", "preresize_mode", "preresize_min_width",
    "preresize_min_height", "preresize_max_width", "preresize_max_height", "mask_fill_holes", "mask_expand_pixels",
    "mask_invert", "mask_blend_pixels", "mask_hipass_filter", "extend_for_outpainting", "extend_up_factor",
    "extend_down_factor", "extend_left_factor", "extend_right_factor", "context_from_mask_extend_factor",
    "auto_context_factor", "aspect_preset", "output_target_width", "output_target_height",
    "wan_temporal_smooth_frames", "wan_stable_crop", "wan_mask_polarity",
)
_LIVE_KEYS = _GEOMETRY_KEYS + ("output_resize_to_target_size", "output_padding", "wan_align_multiple", "preview_frame",
                                # L7.65 P08: Roto-Sync's erosion (the cache keeps the mask as it arrived), the mode and
                                # the two new modes' settings
                                "roto_quality", "crop_mode") + _SMART_CROP_KEYS + _PREPARE_KEYS


def _with_crop_sections(spec):
    for group in ("required", "optional"):
        for name, s in (spec.get(group) or {}).items():
            section = _CROP_SECTIONS.get(name)
            if section and isinstance(s, tuple) and len(s) > 1 and isinstance(s[1], dict):
                s[1]["c2c_section"] = section
                s[1]["advanced"] = True
    return spec


# ══════════════════════════════════════════════════════════════════════
#  NODE 1: InpaintCropProMEC — lquesada-compatible API + Wan 2.2 Animate aware
#
#  Mirrors lquesada/ComfyUI-InpaintCropAndStitch::InpaintCropImproved
#  (preresize + mask preprocessing pipeline + aspect-ratio-aware crop_magic
#  + edge-replicate canvas extension + simple mask*inp+(1-mask)*canvas blend).
#
#  Wan 2.2 Animate (arXiv:2509.14055) extras:
#    * wan_align_multiple        - VAE patchify alignment (default 16)
#    * wan_temporal_smooth_frames - per-frame mask coherence along time axis
#    * wan_stable_crop           - single union bbox across all frames
#                                  (replacement-mode requires consistent crop)
#    * wan_mask_polarity         - regenerate_subject (lquesada default,
#                                  mask=1 -> inpaint) vs preserve_subject
#                                  (Wan2.2 replacement-mode polarity,
#                                  mask=0 -> regenerate)
# ══════════════════════════════════════════════════════════════════════

class InpaintCropProMEC:
    """lquesada-style crop + Wan 2.2 Animate-aware extras."""

    VRAM_TIER = 1
    RESIZE_ALGS = ["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming", "area"]
    PRERESIZE_MODES = [
        "ensure minimum resolution",
        "ensure maximum resolution",
        "ensure minimum and maximum resolution",
    ]
    OUTPUT_PADDING_CHOICES = ["0", "8", "16", "32", "64", "128", "256", "512"]
    # P0.4: aspect presets. "Custom" preserves the legacy
    # output_target_width/_height widgets exactly (back-compat default).
    # "Auto" picks the closest of {Square, 16:9, 9:16, 4:3, 3:4} from the
    # mask bbox shape. Named presets keep the SHORTER of (target_w,target_h)
    # and recompute the longer side from the AR.
    ASPECT_PRESETS = ["Custom", "Auto", "Square", "16:9", "9:16", "4:3", "3:4"]
    ASPECT_RATIOS = {
        "Square": 1.0,
        "16:9": 16.0 / 9.0,
        "9:16": 9.0 / 16.0,
        "4:3":  4.0 / 3.0,
        "3:4":  3.0 / 4.0,
    }
    DEVICE_MODES = ["cpu (compatible)", "gpu (much faster)"]
    MASK_POLARITY = ["regenerate_subject", "preserve_subject"]
    INPAINT_MASK_MODES = ["hard_binary", "slight_feather", "soft_blend"]
    STITCH_BLEND_MODES = ["gaussian", "edge_aware", "laplacian_pyramid", "frequency_blend", "video_stable"]
    FILL_MODES = ["none", "edge_pad", "neutral_gray", "original"]

    @classmethod
    def INPUT_TYPES(cls):
        return _with_crop_sections({
            "required": {
                # L7.65 P08: "image" is the first OPTIONAL input now (a migrated Mask Prepare has none); the sockets
                # keep their slots because the front end adds socket inputs first, in definition order
                "downscale_algorithm": (cls.RESIZE_ALGS, {"default": "bilinear"}),
                "upscale_algorithm":   (cls.RESIZE_ALGS, {"default": "bicubic"}),

                "preresize": ("BOOLEAN", {"default": False,
                    "tooltip": "Resize input image before processing (lquesada-style)."}),
                "preresize_mode": (cls.PRERESIZE_MODES, {"default": "ensure minimum resolution"}),
                "preresize_min_width":  ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 1}),
                "preresize_min_height": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 1}),
                "preresize_max_width":  ("INT", {"default": 16384, "min": 0, "max": 16384, "step": 1}),
                "preresize_max_height": ("INT", {"default": 16384, "min": 0, "max": 16384, "step": 1}),

                "mask_fill_holes":   ("BOOLEAN", {"default": True,
                    "tooltip": "Mark fully-enclosed regions as masked."}),
                "mask_expand_pixels":("INT", {"default": 0, "min": 0, "max": 16384, "step": 1,
                    "tooltip": "Dilate mask by this many pixels."}),
                "mask_invert":       ("BOOLEAN", {"default": False,
                    "tooltip": "Invert mask (anything masked is kept)."}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Pixels of feather for stitch blending (lquesada default 32)."}),
                "mask_hipass_filter":("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Zero out mask values below this threshold."}),

                "extend_for_outpainting": ("BOOLEAN", {"default": False,
                    "tooltip": "Extend image with edge-replicated padding for outpainting."}),
                "extend_up_factor":    ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_down_factor":  ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_left_factor":  ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_right_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),

                "context_from_mask_extend_factor": ("FLOAT", {
                    "default": 1.2, "min": 0.1, "max": 100.0, "step": 0.01,
                    "tooltip": ("Resize context bbox around the mask. 1.0 = exact mask bbox. "
                                ">1 grows outward (1.5 = +50% on every side). "
                                "<1 shrinks INWARD so the crop is tighter than the mask bbox "
                                "(0.7 = inset 15% on every side). Useful for 16:9 / portrait "
                                "single-still crops where the mask bbox is larger than the "
                                "subject you actually want sampled.")}),
                "auto_context_factor": ("BOOLEAN", {
                    "default": False,
                    "tooltip": ("Override context_from_mask_extend_factor automatically so the "
                                "mask fills ~70% of the resulting crop area. Uses actual mask "
                                "AREA (not just bbox) to handle sparse / thin / non-convex "
                                "masks. Result clamped to [0.30, 10.00]. OFF = use the manual "
                                "slider above. Recommended ON for batch / unattended runs.")}),
                "aspect_preset": (cls.ASPECT_PRESETS, {
                    "default": "Custom",
                    "tooltip": ("Force the output crop aspect ratio.\n"
                                "  Custom : use output_target_width / _height as-is "
                                "(legacy lquesada behaviour - DEFAULT).\n"
                                "  Auto   : pick best of {Square, 16:9, 9:16, 4:3, 3:4} from "
                                "the mask bbox aspect.\n"
                                "  Square / 16:9 / 9:16 / 4:3 / 3:4 : keep the SHORTER of "
                                "(output_target_width, output_target_height) and recompute the "
                                "longer side from this AR. Multiple-of-padding rounding still "
                                "applies downstream.")}),

                "output_resize_to_target_size": ("BOOLEAN", {"default": True,
                    "tooltip": "Force output to a specific resolution for sampling."}),
                "output_target_width":  ("INT", {"default": 512, "min": 64, "max": 16384, "step": 1}),
                "output_target_height": ("INT", {"default": 512, "min": 64, "max": 16384, "step": 1}),
                "output_padding": (cls.OUTPUT_PADDING_CHOICES, {"default": "32"}),

                "device_mode": (cls.DEVICE_MODES, {"default": "gpu (much faster)"}),

                "wan_align_multiple": ("INT", {
                    "default": 16, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Force final crop W/H to multiples of this (Wan VAE patchify; 16 recommended)."}),
                "wan_temporal_smooth_frames": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1,
                    "tooltip": "Gaussian smoothing of mask along time axis (frames). 0 disables."}),
                "wan_stable_crop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use a single union bbox across all frames (Wan replacement-mode)."}),
                "wan_mask_polarity": (cls.MASK_POLARITY, {
                    "default": "regenerate_subject",
                    "tooltip": "regenerate_subject: mask=1 -> regenerate (lquesada).  preserve_subject: mask=0 -> regenerate (Wan2.2 replacement: mask=1 keeps environment)."}),

                "inpaint_mask_mode": (cls.INPAINT_MASK_MODES, {
                    "default": "hard_binary",
                    "tooltip": "What the inpaint sampler sees: hard_binary (crisp), slight_feather (gentle), soft_blend (very soft)."}),
                "stitch_blend_mode": (cls.STITCH_BLEND_MODES, {
                    "default": "gaussian",
                    "tooltip": "How the result is composited back: gaussian, edge_aware (Sobel), laplacian_pyramid, frequency_blend, video_stable."}),
                "blend_radius": ("INT", {
                    "default": 32, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Feather radius for the stitch blend mask (independent of mask_blend_pixels)."}),
                "video_stable_temporal_sigma": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": ("[video_stable only] Temporal Gaussian sigma in frames. "
                                "3.0 ≈ 9-frame window. Higher = smoother but laggier on fast motion. 0 = off.")}),
                "video_stable_dilate_px": ("INT", {
                    "default": -1, "min": -1, "max": 128, "step": 1,
                    "tooltip": ("[video_stable only] Pixels to push the blend zone into background "
                                "BEFORE feathering. -1 = derive from blend_radius. 16-32 typical.")}),
                "video_stable_blur_sigma": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 128.0, "step": 0.5,
                    "tooltip": ("[video_stable only] Spatial Gaussian sigma for the wide feather. "
                                "-1 = derive from blend_radius (×0.75). Match to dilate value.")}),
                "fill_masked_area": (cls.FILL_MODES, {
                    "default": "none",
                    "tooltip": "Fill masked region in the cropped image: none, edge_pad (Gaussian smear), neutral_gray, original."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "optional_context_mask": ("MASK",),
                "roto_quality": ("BOOLEAN", {
                    "default": False,
                    "tooltip": ("Roto-Sync mode: tightens the inpaint seam for clean alpha edges. "
                                "Forces laplacian_pyramid blend, halves blend_radius (min 4), "
                                "and pre-erodes the inpaint mask by 1 px so the stitch falls just "
                                "inside the subject. Safe to leave OFF for general inpaint; turn ON "
                                "for compositing / roto / VFX work where boundaries must not bleed.")}),
                # L7.99 (owner A11): appended last so saved workflows keep their values
                "preview_frame": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": ("Which frame the preview under the node shows (0 = the first). Display only: "
                                "every frame is still cropped exactly as before.")}),
                # L7.65 P08: the former Smart Image Crop and Inpaint Mask Prepare, as modes (appended last)
                "crop_mode": (CROP_MODES, {
                    "default": "pro",
                    "tooltip": ("pro: the crop above (lquesada-style, video aware). smart aspect: the stills crop that "
                                "sizes the output between a minimum and a maximum side (the former Smart Image Crop). "
                                "mask prepare: no crop - clean the mask and make the inpaint and stitch masks for the "
                                "whole frame (the former Inpaint Mask Prepare).")}),
                **_mode_inputs(SmartImageCropMEC.INPUT_TYPES()["required"], "smart_", "smart aspect",
                               ("image", "mask")),
                **_mode_inputs(InpaintMaskPrepareMEC.INPUT_TYPES()["required"], "prepare_", "mask prepare",
                               ("mask",)),
            },
            # L7.99 slice 2: which node ran, so the live preview finds this run's masks (no widget)
            "hidden": {"unique_id": "UNIQUE_ID"},
        })

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("stitcher", "cropped_image", "inpaint_mask", "stitch_blend_mask", "info", "preview")
    FUNCTION = "inpaint_crop"
    CATEGORY = "C2C/Inpaint"
    DESCRIPTION = ("Crop around mask for inpainting (lquesada API + Wan 2.2 Animate aware). crop_mode 'smart aspect' "
                   "is the stills crop sized between a min and a max side; 'mask prepare' cleans the mask without "
                   "cropping. Pair with Inpaint Stitch Pro (C2C).")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_kwargs(**kwargs)

    def inpaint_crop(self, image=None, downscale_algorithm="bilinear", upscale_algorithm="bicubic",
                     preresize=False, preresize_mode="ensure minimum resolution",
                     preresize_min_width=1024, preresize_min_height=1024,
                     preresize_max_width=16384, preresize_max_height=16384,
                     mask_fill_holes=True, mask_expand_pixels=0, mask_invert=False,
                     mask_blend_pixels=32, mask_hipass_filter=0.1,
                     extend_for_outpainting=False,
                     extend_up_factor=1.0, extend_down_factor=1.0,
                     extend_left_factor=1.0, extend_right_factor=1.0,
                     context_from_mask_extend_factor=1.2,
                     auto_context_factor=False,
                     aspect_preset="Custom",
                     output_resize_to_target_size=True,
                     output_target_width=512, output_target_height=512, output_padding="32",
                     device_mode="gpu (much faster)",
                     wan_align_multiple=16, wan_temporal_smooth_frames=0.0,
                     wan_stable_crop=True, wan_mask_polarity="regenerate_subject",
                     inpaint_mask_mode="hard_binary", stitch_blend_mode="gaussian", blend_radius=32,
                     video_stable_temporal_sigma=3.0, video_stable_dilate_px=-1, video_stable_blur_sigma=-1.0,
                     fill_masked_area="none",
                     mask=None, optional_context_mask=None,
                     roto_quality: bool = False, preview_frame: int = 0, unique_id=None,
                     crop_mode: str = "pro", **mode_values):
        args = {k: v for k, v in locals().items() if k not in ("self", "mode_values")}
        unknown = set(mode_values) - set(_SMART_CROP_KEYS) - set(_PREPARE_KEYS)
        if unknown:
            raise TypeError(f"inpaint_crop() got unexpected keyword arguments: {sorted(unknown)}")
        defaults = _mode_defaults(type(self))
        mv = {k: mode_values.get(k, defaults[k]) for k in _SMART_CROP_KEYS + _PREPARE_KEYS}
        live_settings = {k: v for k, v in {**args, **mv}.items() if k in _LIVE_KEYS}

        if crop_mode == "smart aspect":
            with torch.no_grad():
                return self._smart_crop(image, mask, mv, preview_frame, unique_id, live_settings)
        if crop_mode == "mask prepare":
            with torch.no_grad():
                return self._prepare_crop(image, mask, mv, downscale_algorithm, upscale_algorithm, preview_frame,
                                          unique_id, live_settings)
        if crop_mode != "pro":
            raise ValueError(f"Inpaint Crop Pro: unknown crop_mode {crop_mode!r} (use one of {CROP_MODES}).")
        if image is None:
            raise ValueError("Inpaint Crop Pro: connect an image (crop_mode 'pro').")

        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("InpaintCropProMEC expects IMAGE tensor [B,H,W,C]")
        for _label, _m in (("mask", mask), ("optional_context_mask", optional_context_mask)):
            if _m is not None and (
                not isinstance(_m, torch.Tensor) or _m.ndim not in (2, 3)
            ):
                raise ValueError(
                    f"InpaintCropProMEC {_label} expects MASK tensor [H,W] or [B,H,W]"
                )

        with torch.no_grad():
            return self._inpaint_crop_core(
                image, downscale_algorithm, upscale_algorithm,
                preresize, preresize_mode,
                preresize_min_width, preresize_min_height,
                preresize_max_width, preresize_max_height,
                mask_fill_holes, mask_expand_pixels, mask_invert,
                mask_blend_pixels, mask_hipass_filter,
                extend_for_outpainting,
                extend_up_factor, extend_down_factor,
                extend_left_factor, extend_right_factor,
                context_from_mask_extend_factor,
                auto_context_factor,
                aspect_preset,
                output_resize_to_target_size,
                output_target_width, output_target_height, output_padding,
                device_mode,
                wan_align_multiple, wan_temporal_smooth_frames,
                wan_stable_crop, wan_mask_polarity,
                inpaint_mask_mode, stitch_blend_mode, blend_radius,
                video_stable_temporal_sigma, video_stable_dilate_px, video_stable_blur_sigma,
                fill_masked_area,
                mask, optional_context_mask,
                roto_quality, preview_frame, unique_id, live_settings,
            )

    # ── L7.65 P08: crop_mode "smart aspect" (the former Smart Image Crop) ─────────────────────────────────────────
    def _smart_crop(self, image, mask, mv, preview_frame, unique_id, live_settings):
        if image is None:
            raise ValueError("Inpaint Crop Pro: connect an image (crop_mode 'smart aspect').")
        if not isinstance(image, torch.Tensor):
            raise ValueError("InpaintCropProMEC expects IMAGE tensor [B,H,W,C]")
        img = _sc.to_bhwc(image)
        B, H, W = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
        if mask is None:                                     # the old node required one; empty = no_mask_mode
            mask = torch.zeros((B, H, W), dtype=torch.float32, device=img.device)
        k = max(0, min(int(preview_frame), B - 1))
        if unique_id is not None:
            try:
                from . import _inpaint_preview_live as _live
                _live.store(str(unique_id), shape_bhw=(B, H, W), mask=mask, opt_mask=None, frame=img[k],
                            frame_index=k, settings=live_settings)
            except Exception as _exc:  # noqa: BLE001 - the live preview never fails a run
                logger.debug("[InpaintCropPro] live preview cache skipped: %s", _exc)
        sv = _unprefix(mv, _SMART_CROP_KEYS, "smart_")
        # the engine: the former node's own code, so every output is what Smart Image Crop gave
        crop_image, crop_mask, st, overlay, info = SmartImageCropMEC().crop(image, mask, **sv)
        # the preview is drawn from what ran (the engine's stitcher); plan_crop only supplies the measured mask tint
        _geometry, msk = plan_crop(mask, (B, H, W), **sv)
        plan = _smart_plan_record(st, B, H, W)
        fr = plan["frames"][k if len(plan["frames"]) > 1 else 0]
        lines = _crop_preview_lines(plan, fr, k, B) + [_smart_preview_line(st)]
        ui, _pic = _save_preview(img[k], _fit_mask_hw(msk[k % int(msk.shape[0])], H, W), fr["window"],
                                 crop_image[min(k, int(crop_image.shape[0]) - 1)], lines, plan)
        # stitch_blend_mask = crop_mask: the mask Smart's "Mask Feather" stitch blends along
        return {"ui": ui, "result": (st, crop_image, crop_mask, crop_mask, info, overlay)}

    # ── L7.65 P08: crop_mode "mask prepare" (the former Inpaint Mask Prepare) ─────────────────────────────────────
    def _prepare_crop(self, image, mask, mv, downscale_algorithm, upscale_algorithm, preview_frame, unique_id,
                      live_settings):
        if mask is None:
            raise ValueError("Inpaint Crop Pro: connect a mask (crop_mode 'mask prepare').")
        pv = _unprefix(mv, _PREPARE_KEYS, "prepare_")
        # the engine: the former node's own code (image = its reference_image, used by stitch_edge_mode edge_aware)
        inpaint_mask, stitch_blend_mask, debug_preview, info = InpaintMaskPrepareMEC().prepare_mask(
            mask, reference_image=image, **pv)
        if image is None:                                    # masks only: nothing to crop, nothing to stitch
            return {"ui": {}, "result": (None, None, inpaint_mask, stitch_blend_mask, info, debug_preview)}
        B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        Bm = int(stitch_blend_mask.shape[0])
        k = max(0, min(int(preview_frame), B - 1))
        if unique_id is not None:
            try:
                from . import _inpaint_preview_live as _live
                _live.store(str(unique_id), shape_bhw=(B, H, W), mask=mask, opt_mask=None, frame=image[k],
                            frame_index=k, settings=live_settings)
            except Exception as _exc:  # noqa: BLE001
                logger.debug("[InpaintCropPro] live preview cache skipped: %s", _exc)
        # a whole-frame Pro stitcher, so Inpaint Stitch Pro (blend) puts a full-frame inpaint back with these masks
        stitcher = {
            "downscale_algorithm": downscale_algorithm, "upscale_algorithm": upscale_algorithm, "blend_pixels": 0,
            "blend_radius": int(pv["stitch_feather_radius"]), "stitch_blend_mode": "gaussian",
            "wan_mask_polarity": "regenerate_subject",
            "canvas_to_orig_x": [0] * B, "canvas_to_orig_y": [0] * B,
            "canvas_to_orig_w": [W] * B, "canvas_to_orig_h": [H] * B,
            "canvas_image": [image[i].cpu() for i in range(B)],
            "cropped_to_canvas_x": [0] * B, "cropped_to_canvas_y": [0] * B,
            "cropped_to_canvas_w": [W] * B, "cropped_to_canvas_h": [H] * B,
            "cropped_mask_for_blend": [stitch_blend_mask[min(i, Bm - 1)].cpu() for i in range(B)],
            "device_mode": "cpu (compatible)", "roto_quality": False,
        }
        plan = _prepare_plan_record(B, H, W)
        km = min(k, Bm - 1)
        tint = _fit_mask_hw(inpaint_mask[min(k, int(inpaint_mask.shape[0]) - 1)], H, W)
        ui, _pic = _save_preview(image[k], tint, [0, 0, W, H], debug_preview[km], _prepare_preview_lines(plan, k, B),
                                 plan)
        return {"ui": ui, "result": (stitcher, image, inpaint_mask, stitch_blend_mask, info, debug_preview)}

    def _inpaint_crop_core(self, image, downscale_algorithm, upscale_algorithm,
                     preresize, preresize_mode,
                     preresize_min_width, preresize_min_height,
                     preresize_max_width, preresize_max_height,
                     mask_fill_holes, mask_expand_pixels, mask_invert,
                     mask_blend_pixels, mask_hipass_filter,
                     extend_for_outpainting,
                     extend_up_factor, extend_down_factor,
                     extend_left_factor, extend_right_factor,
                     context_from_mask_extend_factor,
                     auto_context_factor,
                     aspect_preset,
                     output_resize_to_target_size,
                     output_target_width, output_target_height, output_padding,
                     device_mode,
                     wan_align_multiple, wan_temporal_smooth_frames,
                     wan_stable_crop, wan_mask_polarity,
                     inpaint_mask_mode, stitch_blend_mode, blend_radius,
                     video_stable_temporal_sigma, video_stable_dilate_px, video_stable_blur_sigma,
                     fill_masked_area,
                     mask=None, optional_context_mask=None,
                     roto_quality: bool = False, preview_frame: int = 0, unique_id=None, live_settings=None):
        image = image.clone()
        raw_mask = mask                         # L7.65 P08: the live cache keeps the mask as it arrived
        # ── Roto-Sync: tighten seam for clean alpha boundaries ───────────
        # Applied here (single point) so every downstream branch — single
        # crop, video crop, stitch metadata — inherits the change.
        if roto_quality:
            stitch_blend_mode = "laplacian_pyramid"
            blend_radius = max(4, int(blend_radius) // 2)
            if mask is not None:
                # Pre-erode by 1 px so the inpaint sampler regenerates a
                # hair INSIDE the subject, leaving a clean fringe-free
                # stitch (_roto_erode; the live preview applies the same).
                mask = _roto_erode(mask)
        if mask is not None:
            mask = mask.clone()
        if optional_context_mask is not None:
            optional_context_mask = optional_context_mask.clone()

        # Device migration
        if device_mode == "gpu (much faster)":
            try:
                import comfy.model_management as mm
                target_device = mm.get_torch_device()
            except Exception:
                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image = image.to(target_device)
            if mask is not None: mask = mask.to(target_device)
            if optional_context_mask is not None:
                optional_context_mask = optional_context_mask.to(target_device)
        else:
            image = image.cpu()
            if mask is not None: mask = mask.cpu()
            if optional_context_mask is not None:
                optional_context_mask = optional_context_mask.cpu()

        output_padding_int = int(output_padding)

        B, H, W, _ = image.shape
        if unique_id is not None:
            try:
                from . import _inpaint_preview_live as _live
                _live_settings = (live_settings if live_settings is not None
                                  else {k: v for k, v in locals().items() if k in _LIVE_KEYS})
                _k = max(0, min(int(preview_frame), B - 1))
                _live.store(str(unique_id), shape_bhw=(B, H, W), mask=raw_mask, opt_mask=optional_context_mask,
                            frame=image[_k], frame_index=_k, settings=_live_settings)
            except Exception as _exc:  # noqa: BLE001 - the live preview never fails a run
                logger.debug("[InpaintCropPro] live preview cache skipped: %s", _exc)
        # L7.99: masks and every window number come from _geometry(), which the live preview route calls too
        geo = self._geometry(
            (B, H, W), mask, optional_context_mask, device=image.device, dtype=image.dtype,
            downscale_algorithm=downscale_algorithm, upscale_algorithm=upscale_algorithm,
            preresize=preresize, preresize_mode=preresize_mode,
            preresize_min_width=preresize_min_width, preresize_min_height=preresize_min_height,
            preresize_max_width=preresize_max_width, preresize_max_height=preresize_max_height,
            mask_fill_holes=mask_fill_holes, mask_expand_pixels=mask_expand_pixels, mask_invert=mask_invert,
            mask_blend_pixels=mask_blend_pixels, mask_hipass_filter=mask_hipass_filter,
            extend_for_outpainting=extend_for_outpainting, extend_up_factor=extend_up_factor,
            extend_down_factor=extend_down_factor, extend_left_factor=extend_left_factor,
            extend_right_factor=extend_right_factor,
            context_from_mask_extend_factor=context_from_mask_extend_factor, auto_context_factor=auto_context_factor,
            aspect_preset=aspect_preset, output_target_width=output_target_width,
            output_target_height=output_target_height,
            wan_temporal_smooth_frames=wan_temporal_smooth_frames, wan_stable_crop=wan_stable_crop,
            wan_mask_polarity=wan_mask_polarity,
        )
        if geo["preresize"] is not None:                                # Step 1 for the image
            new_w, new_h, alg = geo["preresize"]
            image = _resize_image_alg(image, new_h, new_w, alg)
        if geo["extend_pads"] is not None:                              # Step 3 for the image
            image = _edge_replicate_pad(image, *geo["extend_pads"])
        B, H, W = geo["B"], geo["H"], geo["W"]
        mask, bbox_mask, optional_context_mask = geo["mask"], geo["bbox_mask"], geo["optional_context_mask"]
        bx_list, by_list, bw_list, bh_list = geo["bx"], geo["by"], geo["bw"], geo["bh"]
        resolved_tgt_w, resolved_tgt_h = geo["resolved_tgt"]
        resolved_factor = geo["resolved_factor"]

        # ── Step 5: per-frame crop_magic ───────────────────────────────────
        if output_resize_to_target_size:
            tgt_w, tgt_h = resolved_tgt_w, resolved_tgt_h
        else:
            tgt_w, tgt_h = -1, -1

        result_image_list = []
        result_mask_list = []
        result_blend_src_list = []     # L7.99: the mask padded by edge replication (stitch blend source)
        result_stitcher = {
            "downscale_algorithm": downscale_algorithm,
            "upscale_algorithm": upscale_algorithm,
            "blend_pixels": mask_blend_pixels,
            "blend_radius": blend_radius,
            "stitch_blend_mode": stitch_blend_mode,
            "wan_mask_polarity": wan_mask_polarity,
            "canvas_to_orig_x": [],
            "canvas_to_orig_y": [],
            "canvas_to_orig_w": [],
            "canvas_to_orig_h": [],
            "canvas_image": [],
            "cropped_to_canvas_x": [],
            "cropped_to_canvas_y": [],
            "cropped_to_canvas_w": [],
            "cropped_to_canvas_h": [],
            "cropped_mask_for_blend": [],
            "device_mode": device_mode,
            "roto_quality": bool(roto_quality),
        }

        for i in range(B):
            sub_img = image[i:i+1]
            sub_msk = mask[i:i+1]
            bx, by, bw, bh = bx_list[i], by_list[i], bw_list[i], bh_list[i]

            (canvas_image, cto_x, cto_y, cto_w, cto_h,
             cropped_image, cropped_mask,
             ctc_x, ctc_y, ctc_w, ctc_h, cropped_blend_src) = self._crop_magic(
                sub_img, sub_msk, bx, by, bw, bh,
                tgt_w if tgt_w > 0 else bw,
                tgt_h if tgt_h > 0 else bh,
                output_padding_int, downscale_algorithm, upscale_algorithm,
                output_resize_to_target_size, wan_align_multiple,
                context_factor=resolved_factor,
            )

            result_image_list.append(cropped_image.squeeze(0))
            result_mask_list.append(cropped_mask.squeeze(0))
            result_blend_src_list.append(cropped_blend_src.squeeze(0))
            result_stitcher["canvas_to_orig_x"].append(cto_x)
            result_stitcher["canvas_to_orig_y"].append(cto_y)
            result_stitcher["canvas_to_orig_w"].append(cto_w)
            result_stitcher["canvas_to_orig_h"].append(cto_h)
            result_stitcher["canvas_image"].append(canvas_image.squeeze(0).cpu())
            result_stitcher["cropped_to_canvas_x"].append(ctc_x)
            result_stitcher["cropped_to_canvas_y"].append(ctc_y)
            result_stitcher["cropped_to_canvas_w"].append(ctc_w)
            result_stitcher["cropped_to_canvas_h"].append(ctc_h)
            result_stitcher["cropped_mask_for_blend"].append(cropped_mask.squeeze(0).cpu())

        # Stack outputs
        try:
            out_image = torch.stack(result_image_list, dim=0)
            out_mask  = torch.stack(result_mask_list,  dim=0)
            blend_src = torch.stack(result_blend_src_list, dim=0)
        except RuntimeError:
            ref_h, ref_w = result_image_list[0].shape[0], result_image_list[0].shape[1]
            tmp_img, tmp_msk, tmp_bs = [], [], []
            for im, mk, bs in zip(result_image_list, result_mask_list, result_blend_src_list):
                if im.shape[0] != ref_h or im.shape[1] != ref_w:
                    im = _resize_image_alg(im.unsqueeze(0), ref_h, ref_w, upscale_algorithm).squeeze(0)
                    mk = _resize_mask_alg(mk.unsqueeze(0), ref_h, ref_w, upscale_algorithm).squeeze(0)
                    bs = _resize_mask_alg(bs.unsqueeze(0), ref_h, ref_w, upscale_algorithm).squeeze(0)
                tmp_img.append(im)
                tmp_msk.append(mk)
                tmp_bs.append(bs)
            out_image = torch.stack(tmp_img, dim=0)
            out_mask  = torch.stack(tmp_msk, dim=0)
            blend_src = torch.stack(tmp_bs, dim=0)

        # ── Step 6: post-process cropped image (fill_masked_area) ─────────
        if fill_masked_area != "none" and fill_masked_area != "original":
            binary_3d = (out_mask > 0.5).float().unsqueeze(-1)
            if fill_masked_area == "neutral_gray":
                out_image = out_image * (1.0 - binary_3d) + 0.5 * binary_3d
            elif fill_masked_area == "edge_pad":
                img_bchw = out_image.permute(0, 3, 1, 2)
                blurred = _gaussian_blur_2d(img_bchw, sigma=max(out_image.shape[1], out_image.shape[2]) * 0.15)
                blurred = blurred.permute(0, 2, 3, 1)
                out_image = out_image * (1.0 - binary_3d) + blurred * binary_3d

        # ── Step 7: derive inpaint_mask (model-facing) and stitch_blend_mask ─
        inpaint_mask = _apply_inpaint_mask_mode(out_mask, inpaint_mask_mode)
        # L7.99 D3: the blend comes from the mask padded by edge replication, not the model-facing mask (padded with
        # 1.0 past the border): a mask that never touches the border cannot feather into the plate there, so an empty
        # mask stitches back as an exact identity.
        stitch_blend_mask = _generate_stitch_blend_mask(
            out_image, blend_src, stitch_blend_mode, blend_radius,
            video_stable_temporal_sigma=float(video_stable_temporal_sigma),
            video_stable_dilate_px=int(video_stable_dilate_px),
            video_stable_blur_sigma=float(video_stable_blur_sigma),
        )

        # Replace stitcher's per-frame blend mask with the proper feathered one
        result_stitcher["cropped_mask_for_blend"] = [stitch_blend_mask[i].cpu() for i in range(stitch_blend_mask.shape[0])]

        # Wan polarity: flip user-visible masks back
        if wan_mask_polarity == "preserve_subject":
            inpaint_mask = (1.0 - inpaint_mask).clamp(0.0, 1.0)
            stitch_blend_mask = (1.0 - stitch_blend_mask).clamp(0.0, 1.0)

        info = (
            f"InpaintCropProMEC (lquesada+Wan2.2):\n"
            f"  in: {B}x{H}x{W}\n"
            f"  preresize={preresize} mode={preresize_mode}\n"
            f"  mask: fill_holes={mask_fill_holes} expand={mask_expand_pixels}px "
            f"invert={mask_invert} blend={mask_blend_pixels}px hipass={mask_hipass_filter:.2f}\n"
            f"  bbox[0]: x={bx_list[0]} y={by_list[0]} w={bw_list[0]} h={bh_list[0]}  "
            f"(stable={wan_stable_crop})\n"
            f"  context_factor={context_from_mask_extend_factor:.2f}"
            f"{'  (AUTO → ' + format(resolved_factor, '.2f') + ')' if auto_context_factor else ''}\n"
            f"  aspect_preset={aspect_preset} → target {resolved_tgt_w}x{resolved_tgt_h}\n"
            f"  out: target={'on' if output_resize_to_target_size else 'off'} "
            f"{output_target_width}x{output_target_height} pad={output_padding} "
            f"align={wan_align_multiple}\n"
            f"  wan: stable_crop={wan_stable_crop} temporal={wan_temporal_smooth_frames} "
            f"polarity={wan_mask_polarity}\n"
            f"  inpaint_mask_mode={inpaint_mask_mode} stitch_blend_mode={stitch_blend_mode} "
            f"blend_radius={blend_radius} fill={fill_masked_area}\n"
            f"  device={device_mode}"
        )
        # L7.99 D4: the preview is drawn from what just ran (the stitcher's windows), never estimated
        plan = _crop_plan_record(result_stitcher, (H, W), (int(out_image.shape[1]), int(out_image.shape[2])),
                                 bool(wan_stable_crop), B)
        fr0 = plan["frames"][0]
        info += (f"\n  window[0] (processed image {W}x{H}): x={fr0['window'][0]} y={fr0['window'][1]} "
                 f"w={fr0['window'][2]} h={fr0['window'][3]}"
                 + "".join(f" pad_{k}={v}" for k, v in fr0["padding"].items() if v))
        ui, pic = _crop_preview_ui(image, bbox_mask, out_image, plan, preview_frame)
        if pic is not None:                     # L7.65 P08: the `preview` output is the picture the node shows
            preview = torch.from_numpy(pic).unsqueeze(0)
        else:
            preview = out_image
            info += "\n  preview output: the cropped image (the preview picture was not drawn)"
        result = (result_stitcher, out_image, inpaint_mask, stitch_blend_mask, info, preview)
        # Always the {"ui", "result"} form ComfyUI accepts, so a caller sees one shape whether or not a preview was drawn
        return {"ui": ui, "result": result}

    # ── L7.99: geometry shared by the node and the live preview route ──────────────────────────────────────
    @classmethod
    def _geometry(cls, shape_bhw, mask, optional_context_mask, *, device, dtype, geometry_only=False, **s) -> dict:
        """Masks and context boxes exactly as the crop computes them, without touching the image: returns the
        processed masks, the per-frame context boxes, the resolved target / factor, and what the image needs (the
        pre-resize size + filter, the outpaint padding). The order of operations is the node's (lquesada order)."""
        B, H, W = (int(v) for v in shape_bhw)
        if mask is None:
            mask = torch.zeros((B, H, W), device=device, dtype=dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1).clone()
        if mask.shape[0] != B:
            if mask.shape[0] < B:
                mask = mask[:1].expand(B, -1, -1).clone()
            else:
                mask = mask[:B]
        if mask.shape[1] != H or mask.shape[2] != W:
            mask = _resize_mask_alg(mask, H, W, "bilinear")

        if optional_context_mask is None:
            optional_context_mask = torch.zeros_like(mask)
        else:
            if optional_context_mask.dim() == 2:
                optional_context_mask = optional_context_mask.unsqueeze(0)
            if optional_context_mask.shape[0] == 1 and B > 1:
                optional_context_mask = optional_context_mask.expand(B, -1, -1).clone()
            if optional_context_mask.shape[0] != B:
                if optional_context_mask.shape[0] < B:
                    optional_context_mask = optional_context_mask[:1].expand(B, -1, -1).clone()
                else:
                    optional_context_mask = optional_context_mask[:B]
            if optional_context_mask.shape[1] != H or optional_context_mask.shape[2] != W:
                optional_context_mask = _resize_mask_alg(optional_context_mask, H, W, "bilinear")

        # Wan polarity: invert mask up-front for "preserve_subject" so all
        # downstream logic treats mask=1 as "regenerate" (lquesada-canonical).
        if s["wan_mask_polarity"] == "preserve_subject":
            mask = (1.0 - mask).clamp(0.0, 1.0)

        # ── Step 1: pre-resize ─────────────────────────────────────────────
        preresize_plan = None
        if s["preresize"]:
            new_w, new_h = cls._preresize_dims(W, H, s["preresize_mode"], s["preresize_min_width"],
                                               s["preresize_min_height"], s["preresize_max_width"],
                                               s["preresize_max_height"])
            if (new_w, new_h) != (W, H):
                alg = s["upscale_algorithm"] if (new_w * new_h > W * H) else s["downscale_algorithm"]
                mask = _resize_mask_alg(mask, new_h, new_w, alg)
                optional_context_mask = _resize_mask_alg(optional_context_mask, new_h, new_w, alg)
                preresize_plan = (new_w, new_h, alg)
                H, W = new_h, new_w

        # ── Step 2: mask preprocessing (lquesada order) ────────────────────
        # CRITICAL: bbox is computed from the *bbox_mask* (no blend feathering),
        # while the *mask* used for output/stitch carries the soft feather.
        if s["mask_fill_holes"]:
            mask = _fill_holes_torch(mask)
        if s["mask_expand_pixels"] > 0:
            mask = _expand_mask_pixels(mask, s["mask_expand_pixels"])
        if s["mask_invert"]:
            mask = (1.0 - mask).clamp(0.0, 1.0)

        # Snapshot the *binary-ish* mask used for bbox computation BEFORE blend.
        bbox_mask = mask.clone()

        # geometry_only (the live preview): the feathered soft mask never reaches a window number - the boxes
        # come from bbox_mask and the context mask - so its blend / hi-pass / smoothing is skipped (81 x 1080p: the
        # bulk of the time). The returned "mask" is then the unfeathered one.
        if s["mask_blend_pixels"] > 0 and not geometry_only:
            mask = _expand_mask_pixels(mask, s["mask_blend_pixels"])
            mask = _gaussian_blur_mask(mask, sigma=s["mask_blend_pixels"] * 0.5)
        if s["mask_hipass_filter"] >= 0.01:
            if not geometry_only:
                mask = _hipass_filter(mask, s["mask_hipass_filter"])
            optional_context_mask = _hipass_filter(optional_context_mask, s["mask_hipass_filter"])

        # Wan: temporal coherence (apply to BOTH so they stay aligned)
        if s["wan_temporal_smooth_frames"] > 0.0 and B > 1:
            if not geometry_only:
                mask = _temporal_gaussian_smooth(mask, s["wan_temporal_smooth_frames"])
            bbox_mask = _temporal_gaussian_smooth(bbox_mask, s["wan_temporal_smooth_frames"])

        # ── Step 3: extend for outpainting ─────────────────────────────────
        extend_pads = None
        if s["extend_for_outpainting"]:
            pads = cls._extend_pads(H, W, s["extend_up_factor"], s["extend_down_factor"],
                                    s["extend_left_factor"], s["extend_right_factor"])
            if any(pads):
                up_pad, down_pad, left_pad, right_pad = pads
                fp = (left_pad, right_pad, up_pad, down_pad)
                mask = F.pad(mask.unsqueeze(1), fp, mode="constant", value=1.0).squeeze(1).clamp(0.0, 1.0)
                optional_context_mask = F.pad(optional_context_mask.unsqueeze(1), fp, mode="constant",
                                              value=0.0).squeeze(1).clamp(0.0, 1.0)
                # Apply same padding to bbox_mask so its coords match
                bbox_mask = F.pad(bbox_mask.unsqueeze(1), fp, mode="constant", value=1.0).squeeze(1).clamp(0.0, 1.0)
                extend_pads = pads
                H, W = H + up_pad + down_pad, W + left_pad + right_pad

        # ── Step 4: per-frame (or stable) context bbox ─────────────────────
        # Always use *bbox_mask* (no blend feathering) so bbox is tight.
        bx_list, by_list, bw_list, bh_list = [], [], [], []
        # P0.4 resolve aspect_preset → (tgt_w, tgt_h) used both for AUTO
        # factor heuristic and for the actual crop step downstream. Falls
        # back to the manual widget values when preset == "Custom".
        resolved_tgt_w, resolved_tgt_h = cls._resolve_target_aspect(
            s["aspect_preset"], int(s["output_target_width"]), int(s["output_target_height"]),
            bbox_mask,
        )
        # P0.4 AUTO context factor — overrides manual slider when ON.
        resolved_factor = float(s["context_from_mask_extend_factor"])
        if bool(s["auto_context_factor"]):
            resolved_factor = cls._compute_auto_factor(
                bbox_mask, resolved_tgt_w, resolved_tgt_h, target_fill=0.70,
            )
        if s["wan_stable_crop"]:
            sx, sy, sw, sh = _compute_stable_bbox(bbox_mask)
            if sw <= 0 or sh <= 0:
                sx, sy, sw, sh = 0, 0, W, H
            sx, sy, sw, sh = cls._grow_context(sx, sy, sw, sh,
                                               resolved_factor, W, H)
            ox, oy, ow, oh = _compute_stable_bbox(optional_context_mask)
            if ow > 0 and oh > 0:
                nx = min(sx, ox)
                ny = min(sy, oy)
                nx2 = max(sx + sw, ox + ow)
                ny2 = max(sy + sh, oy + oh)
                sx, sy, sw, sh = nx, ny, nx2 - nx, ny2 - ny
            for _ in range(B):
                bx_list.append(sx)
                by_list.append(sy)
                bw_list.append(sw)
                bh_list.append(sh)
        else:
            for i in range(B):
                fx, fy, fw, fh = _compute_bbox_single(bbox_mask[i])
                if fw <= 0 or fh <= 0:
                    fx, fy, fw, fh = 0, 0, W, H
                fx, fy, fw, fh = cls._grow_context(fx, fy, fw, fh,
                                                  resolved_factor, W, H)
                ox, oy, ow, oh = _compute_bbox_single(optional_context_mask[i])
                if ow > 0 and oh > 0:
                    nx = min(fx, ox)
                    ny = min(fy, oy)
                    nx2 = max(fx + fw, ox + ow)
                    ny2 = max(fy + fh, oy + oh)
                    fx, fy, fw, fh = nx, ny, nx2 - nx, ny2 - ny
                bx_list.append(fx)
                by_list.append(fy)
                bw_list.append(fw)
                bh_list.append(fh)
        return {"B": B, "H": H, "W": W, "mask": mask, "bbox_mask": bbox_mask,
                "optional_context_mask": optional_context_mask, "bx": bx_list, "by": by_list, "bw": bw_list,
                "bh": bh_list, "resolved_tgt": (resolved_tgt_w, resolved_tgt_h), "resolved_factor": resolved_factor,
                "preresize": preresize_plan, "extend_pads": extend_pads}

    @classmethod
    def plan_windows(cls, shape_bhw, mask, optional_context_mask, **s) -> dict:
        """The crop windows the node would execute for these inputs and settings (the live preview's numbers), plus
        the processed masks. Same functions the node runs; no image work. ``mask`` is the mask as it arrived (before
        Roto-Sync's erosion, which is applied here as the node applies it)."""
        mode = s.get("crop_mode", "pro")
        if mode == "smart aspect":                           # L7.65 P08
            B, H, W = (int(v) for v in shape_bhw)
            geometry, msk = plan_crop(mask, (B, H, W), **_unprefix(s, _SMART_CROP_KEYS, "smart_"))
            return {"plan": _smart_plan_record(geometry, B, H, W), "geometry": geometry, "msk": msk}
        if mode == "mask prepare":
            B, H, W = (int(v) for v in shape_bhw)
            return {"plan": _prepare_plan_record(B, H, W)}
        if s.get("roto_quality") and mask is not None:
            mask = _roto_erode(mask)
        geo = cls._geometry(shape_bhw, mask, optional_context_mask, device=torch.device("cpu"),
                            dtype=torch.float32, geometry_only=True, **{k: s[k] for k in _GEOMETRY_KEYS})
        if s["output_resize_to_target_size"]:
            tgt_w, tgt_h = geo["resolved_tgt"]
        else:
            tgt_w, tgt_h = -1, -1
        frames = []
        out_size = None
        for i in range(geo["B"]):
            bx, by, bw, bh = geo["bx"][i], geo["by"][i], geo["bw"][i], geo["bh"][i]
            win = cls._crop_window(bx, by, bw, bh, tgt_w if tgt_w > 0 else bw, tgt_h if tgt_h > 0 else bh,
                                   int(s["output_padding"]), bool(s["output_resize_to_target_size"]),
                                   int(s["wan_align_multiple"]), geo["W"], geo["H"],
                                   context_factor=geo["resolved_factor"])
            x, y, w, h = win["window"]
            frames.append({"window": [x, y, w, h],
                           "padding": {"left": max(0, -x), "top": max(0, -y), "right": max(0, x + w - geo["W"]),
                                       "bottom": max(0, y + h - geo["H"])}})
            if out_size is None:
                out_size = list(win["output_size"])
        return {"plan": {"image_size": [geo["W"], geo["H"]], "output_size": out_size, "frames_total": geo["B"],
                         "stable": bool(s["wan_stable_crop"]), "frames": frames}, "geo": geo}

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _preresize_dims(W, H, mode, min_w, min_h, max_w, max_h):
        """The pre-resize target size (pure; shared with the live preview)."""
        new_w, new_h = W, H

        def fit_min(w, h, mw, mh):
            if w >= mw and h >= mh:
                return w, h
            sw = mw / max(w, 1)
            sh = mh / max(h, 1)
            s = max(sw, sh)
            return max(int(round(w * s)), mw), max(int(round(h * s)), mh)

        def fit_max(w, h, mw, mh):
            if w <= mw and h <= mh:
                return w, h
            sw = mw / max(w, 1)
            sh = mh / max(h, 1)
            s = min(sw, sh)
            return min(int(round(w * s)), mw), min(int(round(h * s)), mh)

        if mode == "ensure minimum resolution":
            new_w, new_h = fit_min(W, H, min_w, min_h)
        elif mode == "ensure maximum resolution":
            new_w, new_h = fit_max(W, H, max_w, max_h)
        elif mode == "ensure minimum and maximum resolution":
            new_w, new_h = fit_min(W, H, min_w, min_h)
            new_w, new_h = fit_max(new_w, new_h, max_w, max_h)
        return new_w, new_h

    @staticmethod
    def _extend_pads(H, W, up_f, down_f, left_f, right_f):
        """Outpaint padding (up, down, left, right) in pixels (pure; shared with the live preview)."""
        return (max(0, int(H * (up_f - 1.0))), max(0, int(H * (down_f - 1.0))),
                max(0, int(W * (left_f - 1.0))), max(0, int(W * (right_f - 1.0))))

    @staticmethod
    def _preresize(image, mask, opt_mask, mode, min_w, min_h, max_w, max_h, down_alg, up_alg):
        H, W = image.shape[1], image.shape[2]
        new_w, new_h = W, H

        def fit_min(w, h, mw, mh):
            if w >= mw and h >= mh:
                return w, h
            sw = mw / max(w, 1)
            sh = mh / max(h, 1)
            s = max(sw, sh)
            return max(int(round(w * s)), mw), max(int(round(h * s)), mh)

        def fit_max(w, h, mw, mh):
            if w <= mw and h <= mh:
                return w, h
            sw = mw / max(w, 1)
            sh = mh / max(h, 1)
            s = min(sw, sh)
            return min(int(round(w * s)), mw), min(int(round(h * s)), mh)

        if mode == "ensure minimum resolution":
            new_w, new_h = fit_min(W, H, min_w, min_h)
        elif mode == "ensure maximum resolution":
            new_w, new_h = fit_max(W, H, max_w, max_h)
        elif mode == "ensure minimum and maximum resolution":
            new_w, new_h = fit_min(W, H, min_w, min_h)
            new_w, new_h = fit_max(new_w, new_h, max_w, max_h)

        if new_w == W and new_h == H:
            return image, mask, opt_mask

        alg = up_alg if (new_w * new_h > W * H) else down_alg
        image = _resize_image_alg(image, new_h, new_w, alg)
        mask = _resize_mask_alg(mask, new_h, new_w, alg)
        opt_mask = _resize_mask_alg(opt_mask, new_h, new_w, alg)
        return image, mask, opt_mask

    @staticmethod
    def _extend_for_outpainting(image, mask, opt_mask, up_f, down_f, left_f, right_f):
        H, W = image.shape[1], image.shape[2]
        up_pad = max(0, int(H * (up_f - 1.0)))
        down_pad = max(0, int(H * (down_f - 1.0)))
        left_pad = max(0, int(W * (left_f - 1.0)))
        right_pad = max(0, int(W * (right_f - 1.0)))
        if up_pad == 0 and down_pad == 0 and left_pad == 0 and right_pad == 0:
            return image, mask, opt_mask
        image = _edge_replicate_pad(image, up_pad, down_pad, left_pad, right_pad)
        mask = F.pad(mask.unsqueeze(1), (left_pad, right_pad, up_pad, down_pad),
                     mode="constant", value=1.0).squeeze(1)
        opt_mask = F.pad(opt_mask.unsqueeze(1), (left_pad, right_pad, up_pad, down_pad),
                         mode="constant", value=0.0).squeeze(1)
        return image, mask.clamp(0.0, 1.0), opt_mask.clamp(0.0, 1.0)

    @staticmethod
    @staticmethod
    def _grow_context(x, y, w, h, factor, img_w, img_h):
        # factor == 1.0  -> tight mask bbox (no change).
        # factor >  1.0  -> grow outward by (factor-1)/2 on each side.
        # factor <  1.0  -> shrink INWARD by (1-factor)/2 on each side
        #                   (centre preserved). Used when the user wants a
        #                   tighter crop than the raw mask bbox (single-still
        #                   16:9 / portrait subjects, etc.).
        if factor == 1.0 or w <= 0 or h <= 0:
            return x, y, w, h
        if factor > 1.0:
            gx = int(round(w * (factor - 1.0) / 2.0))
            gy = int(round(h * (factor - 1.0) / 2.0))
            nx = max(0, x - gx)
            ny = max(0, y - gy)
            nx2 = min(img_w, x + w + gx)
            ny2 = min(img_h, y + h + gy)
            return nx, ny, nx2 - nx, ny2 - ny
        # factor < 1.0  (shrink)
        sx = int(round(w * (1.0 - factor) / 2.0))
        sy = int(round(h * (1.0 - factor) / 2.0))
        nx = x + sx
        ny = y + sy
        nw = max(1, w - 2 * sx)
        nh = max(1, h - 2 * sy)
        # Clamp to image and guarantee >=1 px on each axis
        nx = max(0, min(nx, img_w - 1))
        ny = max(0, min(ny, img_h - 1))
        nw = max(1, min(nw, img_w - nx))
        nh = max(1, min(nh, img_h - ny))
        return nx, ny, nw, nh

    @classmethod
    def _resolve_target_aspect(cls, preset, manual_w, manual_h, bbox_mask):
        """P0.4 — resolve aspect_preset → (target_w, target_h).

        - "Custom"   : returns manual values unchanged (legacy behaviour).
        - "Auto"     : picks the closest of the named ratios to the union
                       mask bbox aspect ratio. Falls back to "Custom" if
                       the mask is empty.
        - named ARs  : keeps the SHORTER of (manual_w, manual_h) and
                       recomputes the longer side from the AR. Floors at
                       64 px on the short side to stay sampler-friendly.
        """
        if preset == "Custom" or preset not in (cls.ASPECT_PRESETS):
            return int(manual_w), int(manual_h)
        if preset == "Auto":
            try:
                sx, sy, sw, sh = _compute_stable_bbox(bbox_mask)
            except Exception:
                sw, sh = 0, 0
            if sw <= 0 or sh <= 0:
                return int(manual_w), int(manual_h)
            ar = float(sw) / float(max(sh, 1))
            # nearest neighbour in log-space (perceptual aspect distance)
            import math
            best = min(cls.ASPECT_RATIOS.items(),
                       key=lambda kv: abs(math.log(kv[1]) - math.log(max(ar, 1e-6))))
            target_ar = best[1]
        else:
            target_ar = cls.ASPECT_RATIOS[preset]
        short = max(64, min(int(manual_w), int(manual_h)))
        if target_ar >= 1.0:
            return int(round(short * target_ar)), short
        return short, int(round(short / target_ar))

    @staticmethod
    def _compute_auto_factor(bbox_mask, target_w, target_h, target_fill=0.70):
        """P0.4 — compute context_factor so the mask fills ~target_fill of
        the resulting crop. Uses actual mask AREA divided by mask-bbox area
        as a sparsity estimate. Result clamped to the widget range
        [0.30, 10.00]."""
        import math
        try:
            sx, sy, sw, sh = _compute_stable_bbox(bbox_mask)
        except Exception:
            return 1.0
        if sw <= 0 or sh <= 0:
            return 1.0
        # Mask area inside its bbox (use >0.5 binarisation for stability).
        m = bbox_mask
        if m.dim() == 4:
            m = m.squeeze(1)
        # union across the batch so AUTO matches the stable_crop union bbox.
        try:
            union = (m > 0.5).any(dim=0)
            sub = union[sy:sy+sh, sx:sx+sw]
            mask_area = float(sub.sum().item())
        except Exception:
            return 1.0
        bbox_area = float(sw * sh)
        if mask_area <= 0 or bbox_area <= 0:
            return 1.0
        fill_ratio = mask_area / bbox_area
        # crop_area_after_factor ≈ bbox_area * factor² (square approximation).
        # fill_in_crop = mask_area / crop_area = fill_ratio / factor².
        # solve fill_in_crop = target_fill → factor = sqrt(fill_ratio / target_fill).
        factor = math.sqrt(fill_ratio / max(target_fill, 1e-3))
        return max(0.30, min(10.0, float(factor)))

    @staticmethod
    def _crop_window(x, y, w, h, target_w, target_h, padding, resize_output, align_multiple, image_w, image_h,
                     context_factor=1.0) -> dict:
        """The crop window for one frame (pure integers; shared by the node and the live preview).

        When ``context_factor < 1.0`` the caller has deliberately shrunk the
        bbox to request a TIGHTER crop. In that case the aspect-ratio step
        must NOT grow the bbox back out (which would defeat the shrink and
        leave the user wondering why values below 1.0 do nothing on a
        non-square target aspect). Instead we SHRINK the longer axis to
        match the target aspect, keeping the centre fixed.
        """
        # 1. Pad target dims to multiple of `padding` AND `align_multiple`
        m = max(padding, align_multiple, 1)
        target_w = _pad_to_multiple(target_w, m)
        target_h = _pad_to_multiple(target_h, m)

        # 2. Fit bbox to target aspect ratio.
        #    factor >= 1.0  → GROW shorter axis (lquesada-original behaviour;
        #                    preserves all of the user's mask area).
        #    factor <  1.0  → SHRINK longer axis (user asked for tighter crop;
        #                    we honour that even at the cost of cropping a
        #                    sliver of the original bbox).
        target_ar = target_w / max(target_h, 1)
        ctx_ar = w / max(h, 1)
        shrink_mode = (context_factor < 1.0)
        if shrink_mode:
            if ctx_ar > target_ar:
                # Too wide → shrink width to match target aspect, keep h.
                new_w = max(1, int(round(h * target_ar)))
                new_h = h
                new_x = x + (w - new_w) // 2
                new_y = y
            else:
                # Too tall → shrink height to match target aspect, keep w.
                new_w = w
                new_h = max(1, int(round(w / target_ar)))
                new_x = x
                new_y = y + (h - new_h) // 2
        else:
            if ctx_ar < target_ar:
                new_w = int(h * target_ar)
                new_h = h
                new_x = x - (new_w - w) // 2
                new_y = y
            else:
                new_w = w
                new_h = int(w / target_ar)
                new_x = x
                new_y = y - (new_h - h) // 2

        # 3. If not resizing output, ensure new dims >= target dims
        if not resize_output:
            if new_w < target_w:
                new_x -= (target_w - new_w) // 2
                new_w = target_w
            if new_h < target_h:
                new_y -= (target_h - new_h) // 2
                new_h = target_h

        # 3b. L7.99 D2: keep the window on real pixels. One that fits inside the image is moved the least distance
        #     to lie inside it (it used to be padded with smeared edge pixels: 41 % of a 16:9 window at the left
        #     border); a larger one covers the whole axis and only the excess is padded.
        new_x = _fit_window_axis(new_x, new_w, image_w)
        new_y = _fit_window_axis(new_y, new_h, image_h)

        return {"window": (new_x, new_y, new_w, new_h), "target": (target_w, target_h),
                "output_size": ((target_w, target_h) if resize_output else (new_w, new_h))}

    @staticmethod
    def _crop_magic(image, mask, x, y, w, h, target_w, target_h, padding,
                    down_alg, up_alg, resize_output, align_multiple,
                    context_factor=1.0):
        """Aspect-ratio fit + edge-replicate canvas + crop + (optional) resize.

        When ``context_factor < 1.0`` the caller has deliberately shrunk the
        bbox to request a TIGHTER crop. In that case the aspect-ratio step
        must NOT grow the bbox back out (which would defeat the shrink and
        leave the user wondering why values below 1.0 do nothing on a
        non-square target aspect). Instead we SHRINK the longer axis to
        match the target aspect, keeping the centre fixed.
        """
        B, image_h, image_w, C = image.shape

        if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
            return (image, 0, 0, image_w, image_h, image, mask, 0, 0, image_w, image_h, mask)

        win = InpaintCropProMEC._crop_window(x, y, w, h, target_w, target_h, padding, resize_output, align_multiple,
                                             image_w, image_h, context_factor=context_factor)
        new_x, new_y, new_w, new_h = win["window"]
        target_w, target_h = win["target"]

        # 4. Compute canvas padding
        up_padding = max(0, -new_y)
        down_padding = max(0, (new_y + new_h) - image_h)
        left_padding = max(0, -new_x)
        right_padding = max(0, (new_x + new_w) - image_w)

        canvas_image = _edge_replicate_pad(image, up_padding, down_padding, left_padding, right_padding)
        canvas_mask = F.pad(mask.unsqueeze(1),
                            (left_padding, right_padding, up_padding, down_padding),
                            mode="constant", value=1.0).squeeze(1)
        # The stitch blend's source: the same mask continued past the border by edge replication (L7.99 D3)
        if up_padding or down_padding or left_padding or right_padding:
            canvas_blend_src = F.pad(mask.unsqueeze(1),
                                     (left_padding, right_padding, up_padding, down_padding),
                                     mode="replicate").squeeze(1)
        else:
            canvas_blend_src = mask

        cto_x, cto_y = left_padding, up_padding
        cto_w, cto_h = image_w, image_h

        ctc_x = new_x + left_padding
        ctc_y = new_y + up_padding
        ctc_w = new_w
        ctc_h = new_h

        cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :]
        cropped_mask  = canvas_mask[:,  ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
        cropped_blend_src = canvas_blend_src[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

        # 5. Resize to target
        if resize_output:
            alg = up_alg if (target_w > ctc_w or target_h > ctc_h) else down_alg
            cropped_image = _resize_image_alg(cropped_image, target_h, target_w, alg)
            cropped_mask  = _resize_mask_alg(cropped_mask,   target_h, target_w, alg)
            cropped_blend_src = _resize_mask_alg(cropped_blend_src, target_h, target_w, alg)

        return (canvas_image, cto_x, cto_y, cto_w, cto_h,
                cropped_image, cropped_mask,
                ctc_x, ctc_y, ctc_w, ctc_h, cropped_blend_src)


# ══════════════════════════════════════════════════════════════════════
#  L7.65 P08: stitch helpers for the modes "paste back" and "smart"
# ══════════════════════════════════════════════════════════════════════

def _stitcher_layout(st) -> str:
    """Which crop made this stitcher: "smart" (Inpaint Crop Pro in smart aspect mode / the former Smart Image Crop:
    bypass + x, y, w, h, no pixels), "pro" (the lquesada-style layout with the canvas), or "other"."""
    if not isinstance(st, dict):
        return "other"
    if "bypass" in st and "x" in st:
        return "smart"
    if "cropped_to_canvas_x" in st:
        return "pro"
    return "other"


def _paste_back_alpha(stitcher: dict, B: int, feather_edges, feather_radius) -> torch.Tensor:
    """The alpha Inpaint Paste Back's engine applied, per output frame, in original-image pixels (the same window,
    the same feather; blend_mask_used of stitch_mode "paste back")."""
    n = len(stitcher["cropped_to_canvas_x"])
    override = (n == 1 and B != 1)
    feather_active = bool(feather_edges) and int(feather_radius) > 0
    outs = []
    for i in range(B):
        j = 0 if override else i
        cto_x, cto_y = stitcher["canvas_to_orig_x"][j], stitcher["canvas_to_orig_y"][j]
        cto_w, cto_h = stitcher["canvas_to_orig_w"][j], stitcher["canvas_to_orig_h"][j]
        ctc_x, ctc_y = stitcher["cropped_to_canvas_x"][j], stitcher["cropped_to_canvas_y"][j]
        ctc_w, ctc_h = stitcher["cropped_to_canvas_w"][j], stitcher["cropped_to_canvas_h"][j]
        dtype = stitcher["canvas_image"][j].dtype
        if not feather_active:
            alpha = torch.ones((ctc_h, ctc_w), dtype=dtype)
        else:
            r = max(1, int(feather_radius))
            inner_h = max(1, ctc_h - 2 * r)
            inner_w = max(1, ctc_w - 2 * r)
            inner = torch.zeros(1, 1, ctc_h, ctc_w, dtype=dtype)
            sy, sx = (ctc_h - inner_h) // 2, (ctc_w - inner_w) // 2
            inner[:, :, sy:sy + inner_h, sx:sx + inner_w] = 1.0
            alpha = _gaussian_blur_2d(inner, sigma=r * 0.5).clamp(0.0, 1.0)[0, 0]
        full = torch.zeros((cto_h, cto_w), dtype=torch.float32)
        paste_x, paste_y = ctc_x - cto_x, ctc_y - cto_y
        x0, y0 = max(0, paste_x), max(0, paste_y)
        x1, y1 = min(cto_w, paste_x + ctc_w), min(cto_h, paste_y + ctc_h)
        if x1 > x0 and y1 > y0:
            sx0, sy0 = x0 - paste_x, y0 - paste_y
            full[y0:y1, x0:x1] = alpha[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)].float()
        outs.append(full)
    return torch.stack(outs, dim=0)


def _smart_stitch_alpha(original_image, processed_image, stitcher: dict, feather_pixels, blend_mode,
                        resize_full_image_output, mask) -> torch.Tensor:
    """The weight Smart Image Stitch's engine applied, in original-image pixels, following its branches (bypass,
    keep-resized, Mask Feather / Hard Paste / Box Feather, frames it pasted); blend_mask_used of stitch_mode "smart"."""
    base = _sc.to_bhwc(original_image)
    processed = _sc.to_bhwc(processed_image)
    B_base, H, W = int(base.shape[0]), int(base.shape[1]), int(base.shape[2])
    out = torch.zeros((B_base, H, W), dtype=torch.float32)
    if stitcher.get("bypass", False):
        return out
    if stitcher.get("full_image_resize", False) and resize_full_image_output == "Keep Resized Image":
        return torch.ones(tuple(processed.shape[:3]), dtype=torch.float32)
    mask_input = None
    if blend_mode == "Mask Feather":
        if mask is None:
            blend_mode = "Box Feather"
        else:
            mask_input = _sc.mask_to_bhw(mask).contiguous()
    sx, sy, sw, sh = stitcher["x"], stitcher["y"], stitcher["w"], stitcher["h"]
    if not sx or processed.shape[0] == 0:
        return out
    for b in range(min(B_base, len(sx))):
        x, y, w, h = sx[b], sy[b], sw[b], sh[b]
        if w <= 0 or h <= 0:
            continue
        if blend_mode == "Mask Feather" and mask_input is not None:
            mi = b % mask_input.shape[0]
            m = _sc.resize_mask(mask_input[mi:mi + 1], w, h)
            if feather_pixels > 0:
                m = _sc.gaussian_blur(m, blur_px=feather_pixels)
        elif blend_mode == "Hard Paste":
            m = torch.ones((1, h, w), dtype=torch.float32, device=base.device)
        else:
            m = _sc.box_feather_mask(h, w, feather_pixels, base.device, torch.float32)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        px0, py0 = max(0, -x), max(0, -y)
        vw, vh = x1 - x0, y1 - y0
        if vw <= 0 or vh <= 0:
            continue
        out[b, y0:y1, x0:x1] = m[0, py0:py0 + vh, px0:px0 + vw].to(processed.dtype).float().cpu()
    return out


# ══════════════════════════════════════════════════════════════════════
#  NODE 2: InpaintStitchProMEC — lquesada-compatible 2-input stitcher
# ══════════════════════════════════════════════════════════════════════

class InpaintStitchProMEC:
    """lquesada-style stitch + override blend modes (gaussian / edge_aware /
    laplacian_pyramid / frequency_blend) + optional color match.
    """

    VRAM_TIER = 1
    BLEND_OVERRIDES = ["from_crop", "gaussian", "edge_aware", "laplacian_pyramid", "frequency_blend", "video_stable"]

    @classmethod
    def INPUT_TYPES(cls):
        from .mec_paint_suite import MECContextInpainter
        return {
            "required": {
                # L7.65 P08: "stitcher" and "inpainted_image" are the first two OPTIONAL inputs now (context blend
                # has no stitcher); same order, so the sockets keep slots 0 and 1
                "blend_mode_override": (cls.BLEND_OVERRIDES, {
                    "default": "from_crop",
                    "tooltip": "Override the blend mode chosen at crop time, or use 'from_crop'."}),
                "color_match": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply mean+std color transfer before stitching to reduce color shift."}),
                "stitch_temporal_sigma": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": ("Post-hoc temporal Gaussian smoothing applied to the per-frame "
                                "blend mask before compositing. 0 = off. "
                                "2-4 = good for jittery segmentation video (≈3 means a 9-frame window). "
                                "Works on top of any blend mode (gaussian / edge_aware / video_stable / etc.).")}),
                "stitch_dilate_px": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": ("Optional dilation (in pixels) applied to the binary core of the blend "
                                "mask before temporal smoothing — pushes the seam into flat background. "
                                "Use with stitch_temporal_sigma > 0 for jittery video. 0 = off.")}),
            },
            "optional": {
                "stitcher": ("STITCHER",),
                "inpainted_image": ("IMAGE",),
                "roto_quality_override": (["from_crop", "force_on", "force_off"], {
                    "default": "from_crop",
                    "tooltip": ("Roto-Sync at stitch time. 'from_crop' honors the InpaintCropProMEC "
                                "flag (recommended). 'force_on' applies tight-seam stitching even if "
                                "crop didn't set it; 'force_off' disables it.")}),
                # L7.65 P08: the former Inpaint Paste Back, Smart Image Stitch and Context Inpainter, as modes
                "stitch_mode": (STITCH_MODES, {
                    "default": "blend",
                    "tooltip": ("blend: the crop's blend mask and blend modes (as before). paste back: resize and "
                                "paste the crop, optional feathered edge (the former Inpaint Paste Back). smart: the "
                                "stitch for a crop made in 'smart aspect' mode; needs original_image (the former "
                                "Smart Image Stitch). context blend: no stitcher - blend a full-frame inpainted image "
                                "over original_image under mask, with colour correction and lightness rescue (the "
                                "former Context Inpainter).")}),
                "original_image": ("IMAGE", {"tooltip": "[smart / context blend] The original image."}),
                "mask": ("MASK", {"tooltip": "[smart: Mask Feather / context blend] The mask to blend along."}),
                **_mode_inputs(InpaintPasteBackMEC.INPUT_TYPES()["required"], "paste_", "paste back",
                               ("stitcher", "inpainted_image")),
                **_mode_inputs(SmartImageStitcherMEC.INPUT_TYPES()["required"], "smart_", "smart",
                               ("original_image", "processed_image", "stitcher")),
                **_mode_inputs(MECContextInpainter.INPUT_TYPES()["required"], "context_", "context blend",
                               ("original_image", "mask", "inpainted_image")),
                # appended last (every earlier slot stays): the Context Inpainter's optional colour reference
                # (L9.39, added on main after P08 branched - around mask = object removal)
                **_mode_inputs({"color_reference": MECContextInpainter.INPUT_TYPES()["optional"]["color_reference"]},
                               "context_", "context blend"),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "blend_mask_used", "info")
    FUNCTION = "inpaint_stitch"
    CATEGORY = "C2C/Inpaint"
    DESCRIPTION = ("Stitch the inpainted crop back into the original (lquesada-compatible) with blend overrides + "
                   "color match; stitch_mode 'paste back', 'smart' (for a smart aspect crop) or 'context blend' "
                   "(colour correction and lightness rescue over the original, no stitcher).")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_kwargs(**kwargs)

    def inpaint_stitch(self, stitcher=None, inpainted_image=None,
                       blend_mode_override="from_crop", color_match=False,
                       stitch_temporal_sigma: float = 0.0,
                       stitch_dilate_px: int = 0,
                       roto_quality_override: str = "from_crop",
                       stitch_mode: str = "blend", original_image=None, mask=None, **mode_values):
        unknown = set(mode_values) - set(_PASTE_KEYS) - set(_SMART_STITCH_KEYS) - set(_CONTEXT_KEYS)
        if unknown:
            raise TypeError(f"inpaint_stitch() got unexpected keyword arguments: {sorted(unknown)}")
        defaults = _mode_defaults(type(self))
        mv = {k: mode_values.get(k, defaults[k]) for k in _PASTE_KEYS + _SMART_STITCH_KEYS + _CONTEXT_KEYS}
        if stitch_mode not in STITCH_MODES:
            raise ValueError(f"Inpaint Stitch Pro: unknown stitch_mode {stitch_mode!r} (use one of {STITCH_MODES}).")
        if stitch_mode == "context blend":
            with torch.no_grad():
                return self._context_blend(stitcher, inpainted_image, original_image, mask, mv)

        layout = _stitcher_layout(stitcher)
        if stitch_mode == "smart":
            if stitcher is None:
                raise ValueError("Inpaint Stitch Pro: connect the stitcher output of Inpaint Crop Pro "
                                 "(crop_mode 'smart aspect').")
            if layout == "pro":
                raise ValueError("Inpaint Stitch Pro: this stitcher comes from Inpaint Crop Pro in 'pro' or "
                                 "'mask prepare' mode - set stitch_mode to 'blend' or 'paste back'.")
            if original_image is None:
                raise ValueError("Inpaint Stitch Pro: connect original_image (stitch_mode 'smart').")
            if inpainted_image is None:
                raise ValueError("Inpaint Stitch Pro: connect inpainted_image.")
            sv = _unprefix(mv, _SMART_STITCH_KEYS, "smart_")
            with torch.no_grad():
                # the engine: the former Smart Image Stitch's own code
                image, info = SmartImageStitcherMEC().stitch(original_image, inpainted_image, stitcher, mask=mask,
                                                             **sv)
                used = _smart_stitch_alpha(original_image, inpainted_image, stitcher, sv["feather_pixels"],
                                           sv["blend_mode"], sv["resize_full_image_output"], mask)
            return (image, used, info)

        # blend / paste back read the Pro layout
        if stitcher is None:
            raise ValueError("Inpaint Stitch Pro: connect the stitcher output of Inpaint Crop Pro.")
        if layout == "smart":
            raise ValueError("Inpaint Stitch Pro: this stitcher comes from Inpaint Crop Pro in 'smart aspect' mode - "
                             "set stitch_mode to 'smart' and connect original_image.")
        if inpainted_image is None:
            raise ValueError("Inpaint Stitch Pro: connect inpainted_image.")
        if not isinstance(inpainted_image, torch.Tensor) or inpainted_image.ndim != 4:
            raise ValueError("InpaintStitchProMEC expects IMAGE tensor [B,H,W,C]")
        if stitch_mode == "paste back":
            pv = _unprefix(mv, _PASTE_KEYS, "paste_")
            with torch.no_grad():
                # the engine: the former Inpaint Paste Back's own code
                out, info = InpaintPasteBackMEC().paste_back(stitcher, inpainted_image, pv["upscale_method"],
                                                             pv["feather_edges"], pv["feather_radius"])
                used = _paste_back_alpha(stitcher, int(inpainted_image.shape[0]), pv["feather_edges"],
                                         pv["feather_radius"])
            return (out, used, info)

        with torch.no_grad():
            return self._inpaint_stitch_core(
                stitcher, inpainted_image,
                blend_mode_override, color_match,
                stitch_temporal_sigma, stitch_dilate_px,
                roto_quality_override,
            )

    def _context_blend(self, stitcher, inpainted_image, original_image, mask, mv):
        """stitch_mode "context blend": the former Context Inpainter (full frame, no stitcher)."""
        from .mec_paint_suite import MECContextInpainter
        for name, val in (("original_image", original_image), ("mask", mask), ("inpainted_image", inpainted_image)):
            if val is None:
                raise ValueError(f"Inpaint Stitch Pro: connect {name} (stitch_mode 'context blend').")
        cv = _unprefix(mv, _CONTEXT_KEYS, "context_")
        # the engine: the former node's own code (every frame of the batch since L9.39)
        img, dbg = MECContextInpainter().execute(original_image, mask, inpainted_image, **cv)
        n = int(original_image.shape[0]) if original_image.ndim == 4 else 1
        done = int(img.shape[0]) if getattr(img, "ndim", 0) == 4 else 1
        info = (f"Inpaint Stitch Pro (context blend):\n"
                f"  frames: {n} given, {done} blended; colour reference: "
                f"{cv.get('color_reference', 'inside mask (same subject)')}\n"
                f"  colour correction={bool(cv['enable_color_correction'])} "
                f"lightness rescue={bool(cv['enable_lightness_rescue'])} "
                f"differential diffusion={bool(cv['enable_differential_diffusion'])}\n"
                f"  blend softness={float(cv['blend_softness']):g}px expansion={int(cv['mask_expansion_blend'])}px")
        if stitcher is not None:
            info += "\n  the stitcher input is not used in this mode"
        return (img, dbg, info)

    def _inpaint_stitch_core(self, stitcher, inpainted_image,
                       blend_mode_override="from_crop", color_match=False,
                       stitch_temporal_sigma: float = 0.0,
                       stitch_dilate_px: int = 0,
                       roto_quality_override: str = "from_crop"):
        if not isinstance(stitcher, dict):
            raise ValueError(
                "InpaintStitchProMEC: 'stitcher' input is missing or invalid "
                "(got "
                f"{type(stitcher).__name__}). Connect the 'stitcher' output "
                "of InpaintCropProMEC to this node."
            )
        # Resolve roto-quality flag (override > crop-time setting).
        _crop_roto = bool(stitcher.get("roto_quality", False))
        if roto_quality_override == "force_on":
            roto_quality = True
        elif roto_quality_override == "force_off":
            roto_quality = False
        else:
            roto_quality = _crop_roto
        inpainted_image = inpainted_image.clone()
        results = []
        blend_masks_out = []

        device_mode = stitcher.get("device_mode", "cpu (compatible)")
        if device_mode == "gpu (much faster)":
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inpainted_image = inpainted_image.to(device)
        else:
            device = torch.device("cpu")
            inpainted_image = inpainted_image.cpu()

        downscale_algorithm = stitcher["downscale_algorithm"]
        upscale_algorithm   = stitcher["upscale_algorithm"]
        wan_polarity        = stitcher.get("wan_mask_polarity", "regenerate_subject")
        stored_mode         = stitcher.get("stitch_blend_mode", "gaussian")
        blend_radius        = int(stitcher.get("blend_radius", 32))

        # Resolve effective blend mode
        if blend_mode_override == "from_crop":
            effective_mode = stored_mode
        else:
            effective_mode = blend_mode_override

        # Roto-Sync at stitch time: prefer laplacian_pyramid (sharper),
        # halve the blend radius (min 4) for a tighter seam.
        if roto_quality:
            if effective_mode in ("gaussian", "edge_aware"):
                effective_mode = "laplacian_pyramid"
            blend_radius = max(4, int(blend_radius) // 2)

        B = inpainted_image.shape[0]
        n = len(stitcher["cropped_to_canvas_x"])
        override_idx = (n == 1 and B != 1)

        # ── Pre-loop temporal stabilization of per-frame blend masks ───────
        # Stack all stored blend masks into a (N,H,W) tensor (only when sizes
        # match — they will when wan_stable_crop is on, which is the typical
        # video case). Apply optional dilation + temporal Gaussian smoothing
        # in one shot, then write back as the new source list. This kills
        # frame-to-frame jitter regardless of which blend mode generated the
        # mask originally.
        stabilized_masks: Optional[List[torch.Tensor]] = None
        temporal_applied = False
        if (stitch_temporal_sigma > 0.0 or stitch_dilate_px > 0) and n > 1:
            stored_masks = stitcher["cropped_mask_for_blend"]
            shapes = {tuple(m.shape) for m in stored_masks}
            if len(shapes) == 1:
                batch = torch.stack(
                    [m.to(device) for m in stored_masks], dim=0
                ).clamp(0.0, 1.0)
                if stitch_dilate_px > 0:
                    binary = (batch > 0.5).float()
                    k = int(stitch_dilate_px) * 2 + 1
                    pad = int(stitch_dilate_px)
                    m4 = binary.unsqueeze(1)
                    dilated = _max_pool_square(m4, k, pad)
                    batch = dilated.squeeze(1).clamp(0.0, 1.0)
                if stitch_temporal_sigma > 0.0:
                    batch = _temporal_gaussian_smooth(
                        batch, sigma=float(stitch_temporal_sigma)
                    )
                    temporal_applied = True
                stabilized_masks = [batch[k] for k in range(batch.shape[0])]
            else:
                # Variable per-frame mask sizes (rare) — skip stabilization.
                stabilized_masks = None

        for i in range(B):
            j = 0 if override_idx else i
            cto_x = stitcher["canvas_to_orig_x"][j]
            cto_y = stitcher["canvas_to_orig_y"][j]
            cto_w = stitcher["canvas_to_orig_w"][j]
            cto_h = stitcher["canvas_to_orig_h"][j]
            ctc_x = stitcher["cropped_to_canvas_x"][j]
            ctc_y = stitcher["cropped_to_canvas_y"][j]
            ctc_w = stitcher["cropped_to_canvas_w"][j]
            ctc_h = stitcher["cropped_to_canvas_h"][j]
            canvas = stitcher["canvas_image"][j].to(device).unsqueeze(0).clone()
            if stabilized_masks is not None:
                blend_mask = stabilized_masks[j].to(device).unsqueeze(0).clone()
            else:
                blend_mask = stitcher["cropped_mask_for_blend"][j].to(device).unsqueeze(0).clone()

            # Wan replacement-mode polarity: stored mask is "regenerate region",
            # i.e. lquesada-canonical. No per-frame inversion needed at stitch time
            # because crop already inverted polarity up-front.

            sub_inp = inpainted_image[i:i+1]

            # Resize inpainted result + blend mask to canvas crop dims
            inp_h, inp_w = sub_inp.shape[1], sub_inp.shape[2]
            if ctc_w > inp_w or ctc_h > inp_h:
                resized_inp = _resize_image_alg(sub_inp, ctc_h, ctc_w, upscale_algorithm)
                resized_msk = _resize_mask_alg(blend_mask, ctc_h, ctc_w, upscale_algorithm)
            else:
                resized_inp = _resize_image_alg(sub_inp, ctc_h, ctc_w, downscale_algorithm)
                resized_msk = _resize_mask_alg(blend_mask, ctc_h, ctc_w, downscale_algorithm)
            resized_msk = resized_msk.clamp(0.0, 1.0)

            canvas_crop = canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :]
            # The original plate may be RGBA while the sampler returns RGB (owner's Linux run 2026-10-12: "size of
            # tensor a (4) must match ... b (3) at dimension 3"). Inpaint the colour, keep the plate's own alpha.
            resized_inp = resized_inp.to(device=canvas_crop.device, dtype=canvas_crop.dtype)
            resized_msk = resized_msk.to(device=canvas_crop.device)
            c_can, c_inp = canvas_crop.shape[-1], resized_inp.shape[-1]
            if c_inp != c_can:
                if c_inp == 3 and c_can == 4:
                    resized_inp = torch.cat([resized_inp, canvas_crop[..., 3:4]], dim=-1)
                elif c_inp == 4 and c_can == 3:
                    resized_inp = resized_inp[..., :3]
                else:
                    raise ValueError(f"Inpaint stitch: the inpainted image has {c_inp} channels, the original "
                                     f"{c_can} - give both as RGB (3) or RGBA (4).")

            # Optional color match (mean+std on masked region)
            if color_match:
                resized_inp = _color_match_mean_std(resized_inp, canvas_crop, resized_msk)

            # Regenerate the blend mask if the override differs from stored
            if blend_mode_override != "from_crop" and blend_mode_override != stored_mode:
                # base binary from the stored feathered mask
                base_binary = (resized_msk > 0.5).float()
                resized_msk = _generate_stitch_blend_mask(
                    canvas_crop, base_binary, effective_mode, blend_radius
                ).clamp(0.0, 1.0)

            # Composite per blend mode
            if effective_mode == "laplacian_pyramid":
                a = canvas_crop.permute(0, 3, 1, 2)
                b = resized_inp.permute(0, 3, 1, 2)
                m_b1 = resized_msk.unsqueeze(1)
                blended = _laplacian_pyramid_blend(a, b, m_b1, levels=5).permute(0, 2, 3, 1).clamp(0.0, 1.0)
            elif effective_mode == "frequency_blend":
                a = canvas_crop.permute(0, 3, 1, 2)
                b = resized_inp.permute(0, 3, 1, 2)
                m_b1 = resized_msk.unsqueeze(1)
                blended = _frequency_blend(a, b, m_b1).permute(0, 2, 3, 1).clamp(0.0, 1.0)
            else:
                # gaussian / edge_aware / unknown -> simple alpha (mask already encodes mode)
                m3 = resized_msk.unsqueeze(-1)
                blended = (m3 * resized_inp + (1.0 - m3) * canvas_crop).clamp(0.0, 1.0)

            # L7.99: where the blend weight is 0 the plate stays exactly as it was. The pyramid / FFT blends
            # reconstruct the whole crop and leaked there (frequency_blend up to 0.0021 on ~2200 px, measured), so
            # blend_mask_used is now the true footprint of the stitch.
            blended = torch.where((resized_msk > 0).unsqueeze(-1), blended, canvas_crop)

            canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :] = blended

            output = canvas[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w, :]
            results.append(output.squeeze(0))

            # Build a full-res blend mask in original-image coords for output
            full_mask = torch.zeros((1, cto_h, cto_w), device=device, dtype=resized_msk.dtype)
            # mask is in canvas coords; project back into original via the
            # (cto_x, cto_y) offset and (ctc_x, ctc_y) canvas-crop offset:
            paste_x = ctc_x - cto_x
            paste_y = ctc_y - cto_y
            x0 = max(0, paste_x)
            y0 = max(0, paste_y)
            x1 = min(cto_w, paste_x + ctc_w)
            y1 = min(cto_h, paste_y + ctc_h)
            if x1 > x0 and y1 > y0:
                src_x0 = x0 - paste_x
                src_y0 = y0 - paste_y
                src_x1 = src_x0 + (x1 - x0)
                src_y1 = src_y0 + (y1 - y0)
                full_mask[:, y0:y1, x0:x1] = resized_msk[:, src_y0:src_y1, src_x0:src_x1]
            blend_masks_out.append(full_mask.squeeze(0))

        out = torch.stack(results, dim=0).cpu()
        out_blend = torch.stack(blend_masks_out, dim=0).cpu()

        # Wan polarity: present user-visible blend mask in their convention
        if wan_polarity == "preserve_subject":
            out_blend = (1.0 - out_blend).clamp(0.0, 1.0)

        info = (
            f"InpaintStitchProMEC:\n"
            f"  frames: {B}  device={device_mode}\n"
            f"  blend_mode: stored={stored_mode} override={blend_mode_override} -> effective={effective_mode}\n"
            f"  blend_radius={blend_radius}  color_match={color_match}  wan_polarity={wan_polarity}\n"
            f"  temporal: sigma={stitch_temporal_sigma:.2f} dilate={stitch_dilate_px}px applied={temporal_applied}\n"
            f"  out: {out.shape[0]}x{out.shape[1]}x{out.shape[2]}x{out.shape[3]}"
        )
        return (out, out_blend, info)



# ══════════════════════════════════════════════════════════════════════
#  ENGINE: InpaintMaskPrepareMEC - since L7.65 P08 Inpaint Crop Pro in crop_mode "mask prepare" runs this
#  class; it is not registered as a node (saved workflows migrate). Its body is kept as shipped (d97cc13).
# ══════════════════════════════════════════════════════════════════════

class InpaintMaskPrepareMEC:
    """Standalone mask preparation: clean up, grow, produce dual inpaint + stitch masks.

    Separates inpaint_mask (what model sees) from stitch_blend_mask (what composite uses).
    Optional temporal smoothing for video batch consistency.
    """

    VRAM_TIER = 1
    INPAINT_EDGE_MODES = ["hard_binary", "slight_feather"]
    STITCH_EDGE_MODES = ["gaussian", "edge_aware"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Raw input mask (B,H,W)"}),
                "fill_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill interior holes in the mask"}),
                "remove_small_regions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove small disconnected blobs"}),
                "min_region_area": ("INT", {
                    "default": 100, "min": 0, "max": 100000, "step": 10,
                    "tooltip": "Minimum region area in pixels to keep"}),
                "grow_pixels": ("INT", {
                    "default": 4, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Dilate mask by this many pixels"}),
                "inpaint_edge_mode": (cls.INPAINT_EDGE_MODES, {
                    "default": "hard_binary",
                    "tooltip": "Edge style for inpaint mask: hard_binary or slight_feather"}),
                "stitch_edge_mode": (cls.STITCH_EDGE_MODES, {
                    "default": "gaussian",
                    "tooltip": "Edge style for stitch blend mask: gaussian or edge_aware"}),
                "stitch_feather_radius": ("INT", {
                    "default": 16, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Feather radius for the stitch blend mask"}),
                "temporal_smooth": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply Gaussian temporal smoothing along batch dimension (for video)"}),
                "temporal_sigma": ("FLOAT", {
                    "default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Temporal Gaussian sigma in frames"}),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Reference image for edge-aware stitch blend mask (required for edge_aware mode)"}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("inpaint_mask", "stitch_blend_mask", "debug_preview", "info")
    FUNCTION = "prepare_mask"
    CATEGORY = "C2C/Inpaint"
    DESCRIPTION = "Clean, grow, and prepare dual masks: inpaint_mask for model + stitch_blend_mask for composite."

    @classmethod
    def IS_CHANGED(cls, mask, fill_holes, remove_small_regions, min_region_area,
                   grow_pixels, inpaint_edge_mode, stitch_edge_mode,
                   stitch_feather_radius, temporal_smooth, temporal_sigma,
                   reference_image=None, **kwargs):
        return hash_args_and_kwargs(
            mask, fill_holes, remove_small_regions, min_region_area,
            grow_pixels, inpaint_edge_mode, stitch_edge_mode,
            stitch_feather_radius, temporal_smooth, temporal_sigma,
            reference_image, **kwargs,
        )

    def prepare_mask(self, mask: torch.Tensor, fill_holes: bool,
                     remove_small_regions: bool, min_region_area: int,
                     grow_pixels: int, inpaint_edge_mode: str,
                     stitch_edge_mode: str, stitch_feather_radius: int,
                     temporal_smooth: bool, temporal_sigma: float,
                     reference_image: Optional[torch.Tensor] = None):
        if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3):
            raise ValueError("InpaintMaskPrepareMEC expects MASK tensor [H,W] or [B,H,W]")
        if reference_image is not None and (
            not isinstance(reference_image, torch.Tensor) or reference_image.ndim != 4
        ):
            raise ValueError("InpaintMaskPrepareMEC reference_image expects IMAGE [B,H,W,C]")

        with torch.no_grad():
            return self._prepare_mask_core(
                mask, fill_holes, remove_small_regions, min_region_area,
                grow_pixels, inpaint_edge_mode, stitch_edge_mode,
                stitch_feather_radius, temporal_smooth, temporal_sigma,
                reference_image,
            )

    def _prepare_mask_core(self, mask: torch.Tensor, fill_holes: bool,
                     remove_small_regions: bool, min_region_area: int,
                     grow_pixels: int, inpaint_edge_mode: str,
                     stitch_edge_mode: str, stitch_feather_radius: int,
                     temporal_smooth: bool, temporal_sigma: float,
                     reference_image: Optional[torch.Tensor] = None):
        device = _get_device(mask)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        B, H, W = mask.shape
        working = mask.clone()

        holes_filled = 0
        regions_removed = 0

        # Step 1: Fill holes
        if fill_holes:
            before_sum = (working > 0.5).float().sum().item()
            working = _fill_holes_torch(working)
            after_sum = (working > 0.5).float().sum().item()
            holes_filled = int(after_sum - before_sum)

        # Step 2: Remove small regions
        if remove_small_regions and min_region_area > 0:
            before_sum = (working > 0.5).float().sum().item()
            working = _remove_small_regions_torch(working, min_region_area)
            after_sum = (working > 0.5).float().sum().item()
            regions_removed = int(before_sum - after_sum)

        # Step 3: Grow mask
        if grow_pixels > 0:
            k = 2 * grow_pixels + 1
            m4 = working.unsqueeze(1)
            working = _max_pool_square(m4, k, grow_pixels).squeeze(1)
            working = working.clamp(0.0, 1.0)

        # Step 4: Generate inpaint mask
        inpaint_mask = _apply_inpaint_mask_mode(working, inpaint_edge_mode)

        # Step 5: Generate stitch blend mask
        if stitch_edge_mode == "edge_aware" and reference_image is not None:
            ref = reference_image
            if ref.shape[0] == 1 and B > 1:
                ref = ref.expand(B, -1, -1, -1)
            if ref.shape[1] != H or ref.shape[2] != W:
                ref = _resize_image(ref, H, W)
            stitch_blend_mask = _edge_aware_blend_mask(ref, working, stitch_feather_radius)
        else:
            binary = (working > 0.5).float()
            stitch_blend_mask = _gaussian_blur_mask(binary, sigma=stitch_feather_radius * 0.4)

        # Step 6: Temporal smoothing
        temporal_variance_before = 0.0
        temporal_variance_after = 0.0
        if temporal_smooth and B > 1:
            temporal_variance_before = stitch_blend_mask.var(dim=0).mean().item()
            stitch_blend_mask = _temporal_gaussian_smooth(stitch_blend_mask, temporal_sigma)
            temporal_variance_after = stitch_blend_mask.var(dim=0).mean().item()

        # Step 7: Build debug preview
        debug_r = inpaint_mask.unsqueeze(-1)
        debug_g = stitch_blend_mask.unsqueeze(-1)
        debug_b = torch.zeros(B, H, W, 1, device=device, dtype=mask.dtype)
        debug_preview = torch.cat([debug_r, debug_g, debug_b], dim=-1)

        # Step 8: Build info string
        info_lines = [
            f"InpaintMaskPrepareMEC:",
            f"  input: {B}x{H}x{W}",
            f"  fill_holes: {fill_holes} (pixels filled: {holes_filled})",
            f"  remove_small_regions: {remove_small_regions} (pixels removed: {regions_removed})",
            f"  grow_pixels: {grow_pixels}",
            f"  inpaint_edge_mode: {inpaint_edge_mode}",
            f"  stitch_edge_mode: {stitch_edge_mode}, radius={stitch_feather_radius}",
            f"  inpaint_mask range: [{inpaint_mask.min().item():.4f}, {inpaint_mask.max().item():.4f}]",
            f"  stitch_blend_mask range: [{stitch_blend_mask.min().item():.4f}, {stitch_blend_mask.max().item():.4f}]",
        ]
        if temporal_smooth and B > 1:
            info_lines.append(
                f"  temporal_smooth: sigma={temporal_sigma:.1f}, "
                f"variance {temporal_variance_before:.6f} → {temporal_variance_after:.6f}"
            )
        info = "\n".join(info_lines)

        return (inpaint_mask, stitch_blend_mask, debug_preview, info)


# ══════════════════════════════════════════════════════════════════════
#  ENGINE: InpaintPasteBackMEC  (clean resize + paste, no blend pipeline) - since L7.65 P08 Inpaint Stitch Pro in
#  stitch_mode "paste back" runs this class; it is not registered as a node. Its body is kept as shipped (d97cc13).
# ══════════════════════════════════════════════════════════════════════

class InpaintPasteBackMEC:
    """Paste inpainted crop back onto original using a STITCHER dict.

    Simple resize + alpha-paste with optional Gaussian-feathered edges.
    Reads the new lquesada-compatible STITCHER schema produced by
    InpaintCropProMEC.
    """

    VRAM_TIER = 1
    INTERP_METHODS = ["lanczos", "bicubic", "bilinear", "nearest", "area"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER", {"tooltip": "Stitcher dict from InpaintCropProMEC"}),
                "inpainted_image": ("IMAGE", {"tooltip": "Inpainted crop result (B,H,W,C)"}),
                "upscale_method": (cls.INTERP_METHODS, {
                    "default": "bicubic",
                    "tooltip": "Interpolation method for resizing crop back"}),
                "feather_edges": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply Gaussian feather at crop boundary"}),
                "feather_radius": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Feather radius in pixels (only used if feather_edges)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "paste_back"
    CATEGORY = "C2C/Inpaint"
    DESCRIPTION = "Paste inpainted crop back using STITCHER, with optional feathered rectangle edges."

    @classmethod
    def IS_CHANGED(cls, stitcher, inpainted_image, upscale_method,
                   feather_edges, feather_radius, **kwargs):
        return hash_args_and_kwargs(
            stitcher, inpainted_image, upscale_method,
            feather_edges, feather_radius, **kwargs,
        )

    def paste_back(self, stitcher, inpainted_image, upscale_method, feather_edges, feather_radius):
        if not isinstance(inpainted_image, torch.Tensor) or inpainted_image.ndim != 4:
            raise ValueError("InpaintPasteBackMEC expects IMAGE tensor [B,H,W,C]")

        with torch.no_grad():
            return self._paste_back_core(
                stitcher, inpainted_image, upscale_method, feather_edges, feather_radius,
            )

    def _paste_back_core(self, stitcher, inpainted_image, upscale_method, feather_edges, feather_radius):
        if not isinstance(stitcher, dict):
            raise ValueError("InpaintPasteBackMEC: stitcher must be a dict from InpaintCropProMEC")

        device_mode = stitcher.get("device_mode", "cpu (compatible)")
        if device_mode == "gpu (much faster)":
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inpainted_image = inpainted_image.to(device)
        else:
            device = torch.device("cpu")
            inpainted_image = inpainted_image.cpu()

        B = inpainted_image.shape[0]
        n = len(stitcher["cropped_to_canvas_x"])
        override = (n == 1 and B != 1)
        results = []
        feather_active = bool(feather_edges) and int(feather_radius) > 0

        for i in range(B):
            j = 0 if override else i
            cto_x = stitcher["canvas_to_orig_x"][j]
            cto_y = stitcher["canvas_to_orig_y"][j]
            cto_w = stitcher["canvas_to_orig_w"][j]
            cto_h = stitcher["canvas_to_orig_h"][j]
            ctc_x = stitcher["cropped_to_canvas_x"][j]
            ctc_y = stitcher["cropped_to_canvas_y"][j]
            ctc_w = stitcher["cropped_to_canvas_w"][j]
            ctc_h = stitcher["cropped_to_canvas_h"][j]
            canvas = stitcher["canvas_image"][j].to(device).unsqueeze(0).clone()

            sub_inp = inpainted_image[i:i+1]
            inp_resized = _resize_image_alg(sub_inp, ctc_h, ctc_w, upscale_method)

            if not feather_active:
                canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :] = inp_resized.clamp(0.0, 1.0)
            else:
                r = max(1, int(feather_radius))
                inner_h = max(1, ctc_h - 2 * r)
                inner_w = max(1, ctc_w - 2 * r)
                inner = torch.zeros(1, 1, ctc_h, ctc_w, device=device, dtype=canvas.dtype)
                sy, sx = (ctc_h - inner_h) // 2, (ctc_w - inner_w) // 2
                inner[:, :, sy:sy + inner_h, sx:sx + inner_w] = 1.0
                alpha = _gaussian_blur_2d(inner, sigma=r * 0.5).clamp(0.0, 1.0)
                alpha = alpha.squeeze(0).squeeze(0).unsqueeze(0).unsqueeze(-1)
                base = canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :]
                blended = base * (1.0 - alpha) + inp_resized.clamp(0.0, 1.0) * alpha
                canvas[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :] = blended

            output = canvas[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w, :]
            results.append(output.squeeze(0))

        out = torch.stack(results, dim=0).cpu()
        info = (
            f"InpaintPasteBackMEC:\n"
            f"  frames: {B}  upscale_method={upscale_method}\n"
            f"  feather: {'on r=' + str(feather_radius) if feather_active else 'off'}\n"
            f"  output: {out.shape[0]}x{out.shape[1]}x{out.shape[2]}x{out.shape[3]}"
        )
        return (out, info)


# InpaintCompositeMEC removed 2026-05-18 — use InpaintStitchProMEC (stitch_mode
# "blend" or "paste back", L7.65 P08).
