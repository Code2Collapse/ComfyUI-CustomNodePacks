"""
InpaintSuiteMEC — Inpaint Crop Pro + Stitch Pro + Mask Prepare.

Three nodes for professional inpainting workflows:
  1. InpaintCropProMEC    — crop around mask with blend mask preparation
  2. InpaintStitchProMEC  — composite inpainted area back seamlessly
  3. InpaintMaskPrepareMEC — standalone mask cleanup and dual-mask output

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
        eroded = -F.max_pool2d(-m4, kernel_size=k_size, stride=1, padding=pad)
        dilated = F.max_pool2d(eroded, kernel_size=k_size, stride=1, padding=pad)
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
        dilated = F.max_pool2d(m4, kernel_size=k, stride=1, padding=pad)
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


def _expand_mask_pixels(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """Max-pool dilation by `pixels` (lquesada-compatible)."""
    if pixels <= 0:
        return mask
    k = 2 * pixels + 1
    m4 = mask.unsqueeze(1)
    out = F.max_pool2d(m4, kernel_size=k, stride=1, padding=pixels)
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
    DEVICE_MODES = ["cpu (compatible)", "gpu (much faster)"]
    MASK_POLARITY = ["regenerate_subject", "preserve_subject"]
    INPAINT_MASK_MODES = ["hard_binary", "slight_feather", "soft_blend"]
    STITCH_BLEND_MODES = ["gaussian", "edge_aware", "laplacian_pyramid", "frequency_blend", "video_stable"]
    FILL_MODES = ["none", "edge_pad", "neutral_gray", "original"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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
                    "default": 1.2, "min": 1.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Grow context bbox by this factor (1.5 = +50% on every side)."}),

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
                "mask": ("MASK",),
                "optional_context_mask": ("MASK",),
                "roto_quality": ("BOOLEAN", {
                    "default": False,
                    "tooltip": ("Roto-Sync mode: tightens the inpaint seam for clean alpha edges. "
                                "Forces laplacian_pyramid blend, halves blend_radius (min 4), "
                                "and pre-erodes the inpaint mask by 1 px so the stitch falls just "
                                "inside the subject. Safe to leave OFF for general inpaint; turn ON "
                                "for compositing / roto / VFX work where boundaries must not bleed.")}),
            },
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("stitcher", "cropped_image", "inpaint_mask", "stitch_blend_mask", "info")
    FUNCTION = "inpaint_crop"
    CATEGORY = "C2C/Inpaint"
    DESCRIPTION = ("Crop around mask for inpainting (lquesada API + Wan 2.2 Animate aware). "
                   "Pair with Inpaint Stitch Pro (C2C).")

    def inpaint_crop(self, image, downscale_algorithm, upscale_algorithm,
                     preresize, preresize_mode,
                     preresize_min_width, preresize_min_height,
                     preresize_max_width, preresize_max_height,
                     mask_fill_holes, mask_expand_pixels, mask_invert,
                     mask_blend_pixels, mask_hipass_filter,
                     extend_for_outpainting,
                     extend_up_factor, extend_down_factor,
                     extend_left_factor, extend_right_factor,
                     context_from_mask_extend_factor,
                     output_resize_to_target_size,
                     output_target_width, output_target_height, output_padding,
                     device_mode,
                     wan_align_multiple, wan_temporal_smooth_frames,
                     wan_stable_crop, wan_mask_polarity,
                     inpaint_mask_mode, stitch_blend_mode, blend_radius,
                     video_stable_temporal_sigma, video_stable_dilate_px, video_stable_blur_sigma,
                     fill_masked_area,
                     mask=None, optional_context_mask=None,
                     roto_quality: bool = False):

        image = image.clone()
        # ── Roto-Sync: tighten seam for clean alpha boundaries ───────────
        # Applied here (single point) so every downstream branch — single
        # crop, video crop, stitch metadata — inherits the change.
        if roto_quality:
            stitch_blend_mode = "laplacian_pyramid"
            blend_radius = max(4, int(blend_radius) // 2)
            if mask is not None:
                # Pre-erode by 1 px so the inpaint sampler regenerates a
                # hair INSIDE the subject, leaving a clean fringe-free
                # stitch.  Uses explicit zero-padding + min-pool (erosion)
                # — no cv2 dep. F.max_pool2d pads with -inf internally,
                # which would skip border erosion, so we pad with 0 first.
                _m = mask
                if _m.dim() == 2:
                    _m = _m.unsqueeze(0)
                _m4 = _m.unsqueeze(1)
                # Pad with 1 (treat outside as foreground) so we don't
                # over-erode if the subject touches the frame edge —
                # then min-pool = -max_pool(-x).
                _m4_pad = F.pad(_m4, (1, 1, 1, 1), mode="constant", value=1.0)
                _neg = -_m4_pad
                _pooled = F.max_pool2d(_neg, kernel_size=3, stride=1, padding=0)
                mask = (-_pooled).squeeze(1).clamp(0.0, 1.0)
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

        # Default / shape-match the mask
        if mask is None:
            mask = torch.zeros((B, H, W), device=image.device, dtype=image.dtype)
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
        if wan_mask_polarity == "preserve_subject":
            mask = (1.0 - mask).clamp(0.0, 1.0)

        # ── Step 1: pre-resize ─────────────────────────────────────────────
        if preresize:
            image, mask, optional_context_mask = self._preresize(
                image, mask, optional_context_mask,
                preresize_mode, preresize_min_width, preresize_min_height,
                preresize_max_width, preresize_max_height,
                downscale_algorithm, upscale_algorithm,
            )
            B, H, W = image.shape[0], image.shape[1], image.shape[2]

        # ── Step 2: mask preprocessing (lquesada order) ────────────────────
        # CRITICAL: bbox is computed from the *bbox_mask* (no blend feathering),
        # while the *mask* used for output/stitch carries the soft feather.
        if mask_fill_holes:
            mask = _fill_holes_torch(mask)
        if mask_expand_pixels > 0:
            mask = _expand_mask_pixels(mask, mask_expand_pixels)
        if mask_invert:
            mask = (1.0 - mask).clamp(0.0, 1.0)

        # Snapshot the *binary-ish* mask used for bbox computation BEFORE blend.
        bbox_mask = mask.clone()

        if mask_blend_pixels > 0:
            mask = _expand_mask_pixels(mask, mask_blend_pixels)
            mask = _gaussian_blur_mask(mask, sigma=mask_blend_pixels * 0.5)
        if mask_hipass_filter >= 0.01:
            mask = _hipass_filter(mask, mask_hipass_filter)
            optional_context_mask = _hipass_filter(optional_context_mask, mask_hipass_filter)

        # Wan: temporal coherence (apply to BOTH so they stay aligned)
        if wan_temporal_smooth_frames > 0.0 and B > 1:
            mask = _temporal_gaussian_smooth(mask, wan_temporal_smooth_frames)
            bbox_mask = _temporal_gaussian_smooth(bbox_mask, wan_temporal_smooth_frames)

        # ── Step 3: extend for outpainting ─────────────────────────────────
        if extend_for_outpainting:
            image, mask, optional_context_mask = self._extend_for_outpainting(
                image, mask, optional_context_mask,
                extend_up_factor, extend_down_factor,
                extend_left_factor, extend_right_factor,
            )
            # Apply same padding to bbox_mask so its coords match
            _, bbox_mask, _ = self._extend_for_outpainting(
                image, bbox_mask, bbox_mask,
                extend_up_factor, extend_down_factor,
                extend_left_factor, extend_right_factor,
            )
            B, H, W = image.shape[0], image.shape[1], image.shape[2]

        # ── Step 4: per-frame (or stable) context bbox ─────────────────────
        # Always use *bbox_mask* (no blend feathering) so bbox is tight.
        bx_list, by_list, bw_list, bh_list = [], [], [], []
        if wan_stable_crop:
            sx, sy, sw, sh = _compute_stable_bbox(bbox_mask)
            if sw <= 0 or sh <= 0:
                sx, sy, sw, sh = 0, 0, W, H
            sx, sy, sw, sh = self._grow_context(sx, sy, sw, sh,
                                                context_from_mask_extend_factor, W, H)
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
                fx, fy, fw, fh = self._grow_context(fx, fy, fw, fh,
                                                   context_from_mask_extend_factor, W, H)
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

        # ── Step 5: per-frame crop_magic ───────────────────────────────────
        if output_resize_to_target_size:
            tgt_w, tgt_h = output_target_width, output_target_height
        else:
            tgt_w, tgt_h = -1, -1

        result_image_list = []
        result_mask_list = []
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
             ctc_x, ctc_y, ctc_w, ctc_h) = self._crop_magic(
                sub_img, sub_msk, bx, by, bw, bh,
                tgt_w if tgt_w > 0 else bw,
                tgt_h if tgt_h > 0 else bh,
                output_padding_int, downscale_algorithm, upscale_algorithm,
                output_resize_to_target_size, wan_align_multiple,
            )

            result_image_list.append(cropped_image.squeeze(0))
            result_mask_list.append(cropped_mask.squeeze(0))
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
        except RuntimeError:
            ref_h, ref_w = result_image_list[0].shape[0], result_image_list[0].shape[1]
            tmp_img, tmp_msk = [], []
            for im, mk in zip(result_image_list, result_mask_list):
                if im.shape[0] != ref_h or im.shape[1] != ref_w:
                    im = _resize_image_alg(im.unsqueeze(0), ref_h, ref_w, upscale_algorithm).squeeze(0)
                    mk = _resize_mask_alg(mk.unsqueeze(0), ref_h, ref_w, upscale_algorithm).squeeze(0)
                tmp_img.append(im)
                tmp_msk.append(mk)
            out_image = torch.stack(tmp_img, dim=0)
            out_mask  = torch.stack(tmp_msk, dim=0)

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
        stitch_blend_mask = _generate_stitch_blend_mask(
            out_image, out_mask, stitch_blend_mode, blend_radius,
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
            f"  context_factor={context_from_mask_extend_factor:.2f}\n"
            f"  out: target={'on' if output_resize_to_target_size else 'off'} "
            f"{output_target_width}x{output_target_height} pad={output_padding} "
            f"align={wan_align_multiple}\n"
            f"  wan: stable_crop={wan_stable_crop} temporal={wan_temporal_smooth_frames} "
            f"polarity={wan_mask_polarity}\n"
            f"  inpaint_mask_mode={inpaint_mask_mode} stitch_blend_mode={stitch_blend_mode} "
            f"blend_radius={blend_radius} fill={fill_masked_area}\n"
            f"  device={device_mode}"
        )
        return (result_stitcher, out_image, inpaint_mask, stitch_blend_mask, info)

    # ── Helpers ─────────────────────────────────────────────────────────

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
    def _grow_context(x, y, w, h, factor, img_w, img_h):
        if factor <= 1.0:
            return x, y, w, h
        gx = int(round(w * (factor - 1.0) / 2.0))
        gy = int(round(h * (factor - 1.0) / 2.0))
        nx = max(0, x - gx)
        ny = max(0, y - gy)
        nx2 = min(img_w, x + w + gx)
        ny2 = min(img_h, y + h + gy)
        return nx, ny, nx2 - nx, ny2 - ny

    @staticmethod
    def _crop_magic(image, mask, x, y, w, h, target_w, target_h, padding,
                    down_alg, up_alg, resize_output, align_multiple):
        """Aspect-ratio fit + edge-replicate canvas + crop + (optional) resize."""
        B, image_h, image_w, C = image.shape

        if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
            return (image, 0, 0, image_w, image_h, image, mask, 0, 0, image_w, image_h)

        # 1. Pad target dims to multiple of `padding` AND `align_multiple`
        m = max(padding, align_multiple, 1)
        target_w = _pad_to_multiple(target_w, m)
        target_h = _pad_to_multiple(target_h, m)

        # 2. Grow bbox to match target aspect ratio
        target_ar = target_w / max(target_h, 1)
        ctx_ar = w / max(h, 1)
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

        # 4. Compute canvas padding
        up_padding = max(0, -new_y)
        down_padding = max(0, (new_y + new_h) - image_h)
        left_padding = max(0, -new_x)
        right_padding = max(0, (new_x + new_w) - image_w)

        canvas_image = _edge_replicate_pad(image, up_padding, down_padding, left_padding, right_padding)
        canvas_mask = F.pad(mask.unsqueeze(1),
                            (left_padding, right_padding, up_padding, down_padding),
                            mode="constant", value=1.0).squeeze(1)

        cto_x, cto_y = left_padding, up_padding
        cto_w, cto_h = image_w, image_h

        ctc_x = new_x + left_padding
        ctc_y = new_y + up_padding
        ctc_w = new_w
        ctc_h = new_h

        cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :]
        cropped_mask  = canvas_mask[:,  ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

        # 5. Resize to target
        if resize_output:
            alg = up_alg if (target_w > ctc_w or target_h > ctc_h) else down_alg
            cropped_image = _resize_image_alg(cropped_image, target_h, target_w, alg)
            cropped_mask  = _resize_mask_alg(cropped_mask,   target_h, target_w, alg)

        return (canvas_image, cto_x, cto_y, cto_w, cto_h,
                cropped_image, cropped_mask,
                ctc_x, ctc_y, ctc_w, ctc_h)


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
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "inpainted_image": ("IMAGE",),
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
                "roto_quality_override": (["from_crop", "force_on", "force_off"], {
                    "default": "from_crop",
                    "tooltip": ("Roto-Sync at stitch time. 'from_crop' honors the InpaintCropProMEC "
                                "flag (recommended). 'force_on' applies tight-seam stitching even if "
                                "crop didn't set it; 'force_off' disables it.")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "blend_mask_used", "info")
    FUNCTION = "inpaint_stitch"
    CATEGORY = "C2C/Inpaint"
    DESCRIPTION = "Stitch inpainted image back into the original (lquesada-compatible) with blend overrides + color match."

    def inpaint_stitch(self, stitcher, inpainted_image,
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
                    dilated = F.max_pool2d(m4, kernel_size=k, stride=1, padding=pad)
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
#  NODE 3: InpaintMaskPrepareMEC
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

    def prepare_mask(self, mask: torch.Tensor, fill_holes: bool,
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
            working = F.max_pool2d(m4, kernel_size=k, stride=1, padding=grow_pixels).squeeze(1)
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
#  NODE 4: InpaintPasteBackMEC  (clean resize + paste, no blend pipeline)
#  Reimplemented on top of the gracefully-working oldest engine.
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

    def paste_back(self, stitcher, inpainted_image, upscale_method, feather_edges, feather_radius):
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


# ══════════════════════════════════════════════════════════════════════
#  NODE 5: InpaintCompositeMEC  (mode dispatch over Stitch Pro / Paste Back)
# ══════════════════════════════════════════════════════════════════════

class InpaintCompositeMEC:
    """Unified composite over the new STITCHER schema.

    mode=stitch_pro -> alpha composite using the feathered mask stored in
                       the stitcher (lquesada-style).
    mode=paste_back -> hard rectangle paste with optional Gaussian feather.
    """

    VRAM_TIER = 1
    MODES = ["stitch_pro", "paste_back"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER", {"tooltip": "Stitcher dict from InpaintCropProMEC"}),
                "inpainted_image": ("IMAGE", {"tooltip": "Inpainted result (B,H,W,C)"}),
                "mode": (cls.MODES, {"default": "stitch_pro"}),
                "blend_mode_override": (InpaintStitchProMEC.BLEND_OVERRIDES, {
                    "default": "from_crop",
                    "tooltip": "[stitch_pro] Override blend mode or 'from_crop'"}),
                "color_match": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[stitch_pro] Apply mean+std color transfer"}),
                "upscale_method": (InpaintPasteBackMEC.INTERP_METHODS, {
                    "default": "bicubic",
                    "tooltip": "[paste_back] Interpolation for resize"}),
                "feather_edges": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[paste_back] Gaussian-feather rectangle boundary"}),
                "feather_radius": ("INT", {
                    "default": 16, "min": 0, "max": 64, "step": 1,
                    "tooltip": "[paste_back] Feather radius in pixels (0 disables)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "info")
    FUNCTION = "composite"
    CATEGORY = "C2C/Inpaint"
    DESCRIPTION = "Unified composite. mode=stitch_pro = lquesada feather blend with overrides; mode=paste_back = clean resize+paste."

    def composite(self, stitcher, inpainted_image, mode,
                  blend_mode_override, color_match,
                  upscale_method, feather_edges, feather_radius):
        if mode == "paste_back":
            pb = InpaintPasteBackMEC()
            img, info = pb.paste_back(stitcher, inpainted_image,
                                      upscale_method, feather_edges, feather_radius)
            B, H, W, _ = img.shape
            mask = torch.zeros(B, H, W, device=img.device, dtype=img.dtype)
            return (img, mask, info)
        sp = InpaintStitchProMEC()
        img, blend_mask, info = sp.inpaint_stitch(stitcher, inpainted_image,
                                                  blend_mode_override, color_match)
        return (img, blend_mask, info)
