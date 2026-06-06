# SPDX-License-Identifier: Apache-2.0
"""Confidence-aware re-anchoring helpers for RobustMaskPipelineMEC.

Pure-logic module (no ComfyUI node registration). Three re-anchor strategies:

  * ``compute_confidence``  — IoU × size-ratio between consecutive masks.
  * ``DINORelocator``       — DINOv2 patch-feature cosine-similarity search.
  * ``compute_farneback_flow`` + ``flow_warp_reanchor`` — optical-flow warp
    of the last good mask into the current frame's coordinate system.
  * ``blend_masks``         — convex blend between SAM2 mask and warped mask.

All functions take torch tensors and live entirely on the chosen device. No
training, no fine-tuning — DINOv2 weights are fetched lazily from HF Hub
the first time the relocator is asked for a region.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("MEC.MaskMatting.Reanchor")

# ─────────────────────────────────────────────────────────────────────
# Confidence metric
# ─────────────────────────────────────────────────────────────────────
def compute_confidence(mask_t: torch.Tensor,
                       mask_prev: torch.Tensor,
                       thresh: float = 0.5) -> float:
    """IoU × size-ratio between two (H,W) masks. Range [0, 1].

    confidence = consistency * area_ratio
        consistency = |M_t ∩ M_prev| / |M_t ∪ M_prev|
        area_ratio  = min(|M_t|, |M_prev|) / max(|M_t|, |M_prev|)

    If both masks are empty → 1.0 (consistently empty).
    If exactly one is empty → 0.0 (the tracker just lost or gained the
    object).
    """
    if mask_t is None or mask_prev is None:
        return 0.0
    a = (mask_t > thresh).float()
    b = (mask_prev > thresh).float()
    area_a = float(a.sum().item())
    area_b = float(b.sum().item())
    if area_a < 1e-6 and area_b < 1e-6:
        return 1.0
    if area_a < 1e-6 or area_b < 1e-6:
        return 0.0
    inter = float((a * b).sum().item())
    union = area_a + area_b - inter
    iou = inter / max(union, 1e-6)
    size_ratio = min(area_a, area_b) / max(area_a, area_b)
    return float(iou * size_ratio)


# ─────────────────────────────────────────────────────────────────────
# DINOv2 patch-feature re-localization
# ─────────────────────────────────────────────────────────────────────
_DINO_MODEL = None     # tuple (model, processor, device_str)


def _load_dinov2(device: str, repo_id: str = "facebook/dinov2-small"):
    """Lazy-load a frozen DINOv2 backbone. Returns (model, processor).

    Uses HuggingFace ``transformers`` (already a dep via ViTMatteMatter).
    Cached at module level so repeated calls are free.
    """
    global _DINO_MODEL
    if _DINO_MODEL is not None and _DINO_MODEL[2] == device:
        return _DINO_MODEL[0], _DINO_MODEL[1]
    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        raise RuntimeError(
            f"transformers is required for DINOv2 re-anchoring: {exc}"
        )
    processor = AutoImageProcessor.from_pretrained(repo_id)
    model = AutoModel.from_pretrained(repo_id).to(device).eval()
    _DINO_MODEL = (model, processor, device)
    return model, processor


def release_dinov2() -> None:
    """Free the cached DINOv2 model."""
    global _DINO_MODEL
    _DINO_MODEL = None


class DINORelocator:
    """Find the best bounding box for an object in a query image.

    Workflow:
      1. encode_reference(ref_hwc, ref_mask) → stores a mean-pooled
         descriptor of patches that overlap the mask.
      2. find_best_region(query_hwc) → returns (x0, y0, x1, y1) in the
         query image's pixel coordinates.

    DINOv2-small uses 14×14 patches at a 518² input. We resize to that,
    extract patch tokens, compute mean reference descriptor over masked
    patches, then a sliding cosine-similarity window picks the highest
    average region.
    """

    PATCH = 14
    INPUT_SIZE = 518     # default for DINOv2-small (37×37 patches)

    def __init__(self, device: str = "cuda", repo_id: str = "facebook/dinov2-small"):
        self.device = device
        self.repo_id = repo_id
        self._ref_desc: Optional[torch.Tensor] = None  # (D,)
        self._ref_shape: Optional[Tuple[int, int]] = None

    # --- internal -----------------------------------------------------
    def _features(self, image_hwc: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Run DINOv2 → (P, D) patch features + (n_h, n_w) grid size."""
        model, processor = _load_dinov2(self.device, self.repo_id)
        # processor wants HWC uint8 numpy
        img_np = (image_hwc.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        inputs = processor(images=img_np, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.inference_mode():
            out = model(pixel_values)
        # last_hidden_state: (1, 1+P, D), token 0 = CLS
        hidden = out.last_hidden_state[0]
        patches = hidden[1:]  # (P, D)
        # Normalize
        patches = F.normalize(patches, dim=-1)
        H_in = pixel_values.shape[-2]
        W_in = pixel_values.shape[-1]
        n_h = H_in // self.PATCH
        n_w = W_in // self.PATCH
        return patches, n_h, n_w

    # --- public api ---------------------------------------------------
    def encode_reference(self,
                         ref_hwc: torch.Tensor,
                         ref_mask: torch.Tensor) -> None:
        """Cache a descriptor vector for the reference object."""
        patches, n_h, n_w = self._features(ref_hwc)
        H, W = ref_hwc.shape[:2]
        # Downsample mask to (n_h, n_w) so each entry tags one patch.
        m = ref_mask.to(self.device).float()
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(0)
        small = F.adaptive_max_pool2d(m, (n_h, n_w))[0, 0]
        flat = small.reshape(-1) > 0.5
        if flat.sum().item() < 1:
            # Mask is empty; fall back to whole image
            self._ref_desc = patches.mean(dim=0)
        else:
            self._ref_desc = patches[flat].mean(dim=0)
        self._ref_desc = F.normalize(self._ref_desc, dim=-1)
        self._ref_shape = (H, W)

    def find_best_region(self,
                         query_hwc: torch.Tensor,
                         window: int = 5) -> Tuple[int, int, int, int]:
        """Return (x0,y0,x1,y1) in the query image's original pixels.

        Slides a (window × window) box over the patch-similarity heatmap
        and returns the argmax window expanded to pixel coords.
        """
        if self._ref_desc is None:
            raise RuntimeError("DINORelocator.encode_reference() not called.")
        H, W = query_hwc.shape[:2]
        patches, n_h, n_w = self._features(query_hwc)
        # Per-patch cosine similarity vs reference descriptor.
        sim = (patches @ self._ref_desc).reshape(n_h, n_w)  # (n_h, n_w)
        # Smooth with a window-sized average pool to favour coherent blobs.
        sim_4d = sim.unsqueeze(0).unsqueeze(0)
        pad = window // 2
        sim_pooled = F.avg_pool2d(sim_4d, kernel_size=window, stride=1,
                                  padding=pad)[0, 0]
        flat_idx = int(torch.argmax(sim_pooled).item())
        cy_p = flat_idx // n_w
        cx_p = flat_idx % n_w
        # Convert patch grid coords back to original pixel coords.
        px_per_patch_x = W / n_w
        px_per_patch_y = H / n_h
        half_x = (window / 2.0) * px_per_patch_x
        half_y = (window / 2.0) * px_per_patch_y
        cx = (cx_p + 0.5) * px_per_patch_x
        cy = (cy_p + 0.5) * px_per_patch_y
        x0 = int(max(0, math.floor(cx - half_x)))
        y0 = int(max(0, math.floor(cy - half_y)))
        x1 = int(min(W, math.ceil(cx + half_x)))
        y1 = int(min(H, math.ceil(cy + half_y)))
        if x1 <= x0:
            x1 = min(W, x0 + 1)
        if y1 <= y0:
            y1 = min(H, y0 + 1)
        return x0, y0, x1, y1


# ─────────────────────────────────────────────────────────────────────
# Optical flow warp re-anchor (Farneback — no GPU required)
# ─────────────────────────────────────────────────────────────────────
def compute_farneback_flow(img_a_hwc: torch.Tensor,
                           img_b_hwc: torch.Tensor) -> torch.Tensor:
    """Dense optical flow from a→b. Returns (2,H,W) tensor (dx, dy).

    Uses cv2.calcOpticalFlowFarneback. Returns zeros if cv2 unavailable.
    Frames are CPU-numpy converted internally; output tensor is placed on
    the same device as ``img_a_hwc``.
    """
    H, W = img_a_hwc.shape[:2]
    device = img_a_hwc.device
    try:
        import cv2  # type: ignore
    except Exception:
        log.warning("[reanchor] cv2 unavailable — flow warp disabled.")
        return torch.zeros(2, H, W, device=device, dtype=torch.float32)
    a = (img_a_hwc.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    b = (img_b_hwc.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    a_g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_g = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    flow_np = cv2.calcOpticalFlowFarneback(
        a_g, b_g, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    flow = torch.from_numpy(flow_np.astype(np.float32)).to(device)
    # (H,W,2) → (2,H,W)
    return flow.permute(2, 0, 1).contiguous()


def flow_warp_reanchor(mask_hw: torch.Tensor,
                       flow_2hw: torch.Tensor) -> torch.Tensor:
    """Warp a (H,W) mask by a (2,H,W) flow field via grid_sample.

    The output at pixel (x,y) is the input mask sampled at (x-dx, y-dy)
    — i.e. forward warping by the flow.
    """
    H, W = mask_hw.shape[-2:]
    device = mask_hw.device
    dtype = mask_hw.dtype if mask_hw.dtype.is_floating_point else torch.float32
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    flow_2hw = flow_2hw.to(device=device, dtype=dtype)
    dx = flow_2hw[0]
    dy = flow_2hw[1]
    src_x = xx - dx
    src_y = yy - dy
    nx = (src_x / max(W - 1, 1)) * 2.0 - 1.0
    ny = (src_y / max(H - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([nx, ny], dim=-1).unsqueeze(0)
    inp = mask_hw.to(dtype=dtype).view(1, 1, H, W)
    out = F.grid_sample(inp, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out[0, 0].clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────
# Convex blend
# ─────────────────────────────────────────────────────────────────────
def blend_masks(m_sam: torch.Tensor,
                m_warped: torch.Tensor,
                alpha: float) -> torch.Tensor:
    """Return α · m_sam + (1-α) · m_warped, clamped to [0,1]."""
    a = float(max(0.0, min(1.0, alpha)))
    out = a * m_sam.float() + (1.0 - a) * m_warped.float()
    return out.clamp(0, 1)


__all__ = [
    "compute_confidence",
    "DINORelocator",
    "release_dinov2",
    "compute_farneback_flow",
    "flow_warp_reanchor",
    "blend_masks",
]
