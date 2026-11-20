# PORTED FROM: https://github.com/ausboss/ComfyUI-AusBoss nodes/_mask_helpers.py grow_shrink_mask and the trimap rule
# of matting_refine (MIT, Copyright (c) 2026 ausboss). Adapted: the trimap's "unknown" band is returned as the region
# where Mask Refiner may change the matte (L9.172 / A25 spec item 46.1, uncertainty-driven refinement).
"""Where a matte is uncertain: the band between definite subject and definite background.

AusBoss builds a matting trimap from a mask this way: the binarized mask eroded by the edge radius AND opaque in the
given mask is definite foreground; anything inside the dilated mask OR carrying any alpha at all is possible
foreground; the unknown band is possible minus definite. On a binary mask that is exactly the erode / dilate ring; on
a feathered mask the soft ramp widens it. Torch only, no optional dependency.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# below this an alpha reads as empty, above 1 - this as fully opaque (AusBoss _SOFT_EPSILON)
SOFT_EPSILON = 1e-3
# long batches go a few frames at a time (same result to the bit, less peak memory)
CHUNK = 8


def grow_shrink_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """Dilate (positive) or erode (negative) a [B,H,W] mask by whole pixels with one separable square max filter
    (n 3x3 dilations == one (2n+1) square dilation). Soft values survive."""
    steps = abs(int(pixels))
    if steps == 0:
        return mask
    if mask.shape[0] > CHUNK:
        return torch.cat([grow_shrink_mask(part, pixels) for part in mask.split(CHUNK)])
    size = 2 * steps + 1
    x = mask.unsqueeze(1)
    if pixels > 0:
        x = F.max_pool2d(x, kernel_size=(1, size), stride=1, padding=(0, steps))
        x = F.max_pool2d(x, kernel_size=(size, 1), stride=1, padding=(steps, 0))
    else:
        x = -F.max_pool2d(-x, kernel_size=(1, size), stride=1, padding=(0, steps))
        x = -F.max_pool2d(-x, kernel_size=(size, 1), stride=1, padding=(steps, 0))
    return x.squeeze(1)


def uncertain_band(mask: torch.Tensor, band_px: int) -> torch.Tensor:
    """bool [B,H,W]: True where the matte is neither definitely subject nor definitely background (AusBoss trimap
    'unknown' with the edge radius = band_px). An empty or full frame has no uncertain pixel."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"uncertain_band expects a [B,H,W] mask, got shape {tuple(mask.shape)}")
    band = max(1, int(band_px))
    alpha = mask.detach().float().clamp(0.0, 1.0)
    solid = (alpha >= 0.5).float()
    sure_fg = (grow_shrink_mask(solid, -band) >= 0.5) & (alpha >= 1.0 - SOFT_EPSILON)
    possible = (grow_shrink_mask(solid, band) >= 0.5) | (alpha > SOFT_EPSILON)
    return possible & ~sure_fg
