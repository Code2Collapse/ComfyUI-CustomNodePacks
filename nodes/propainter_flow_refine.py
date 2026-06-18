# FILE: nodes/propainter_flow_refine.py
# FEATURE: P2 — FlowRefineMEC (RAFT bidirectional flow + grid_sample warp + consistency mask)
# INTEGRATES WITH: nodes/propainter_bridge.py (RAFT loader)
"""
Standalone RAFT flow node. Takes two IMAGE batches, returns:
    - flow_field   (B, H, W, 2)  U,V in pixel units
    - warped       (B, H, W, C)  frame_a warped toward frame_b
    - consistency  (B, H, W)     1.0 where bidirectional flow agrees
"""
from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

from ._is_changed_util import hash_args_and_kwargs

from .propainter_bridge import (
    HAS_PROPAINTER,
    get_device,
    load_models,
    require_propainter,
)

log = logging.getLogger("MEC.flow_refine")


def _to_raft(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Comfy IMAGE (B,H,W,C) [0,1] -> RAFT input (B,C,H,W) [0,255]."""
    x = images.to(device).permute(0, 3, 1, 2).contiguous() * 255.0
    return x.clamp(0.0, 255.0)


def _grid_from_flow(flow_bchw: torch.Tensor) -> torch.Tensor:
    """flow (B, 2, H, W) -> sampling grid (B, H, W, 2) for grid_sample."""
    B, _, H, W = flow_bchw.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=flow_bchw.device, dtype=flow_bchw.dtype),
        torch.arange(W, device=flow_bchw.device, dtype=flow_bchw.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    sample = grid + flow_bchw  # absolute pixel coords
    nx = 2.0 * sample[:, 0] / max(W - 1, 1) - 1.0
    ny = 2.0 * sample[:, 1] / max(H - 1, 1) - 1.0
    return torch.stack((nx, ny), dim=-1)


@torch.no_grad()
def _raft_pair(raft, a_b3hw: torch.Tensor, b_b3hw: torch.Tensor,
               iters: int) -> torch.Tensor:
    """Returns flow (B, 2, H, W) without padding artefacts."""
    from .propainter_bridge import InputPadder  # vendored
    if InputPadder is None:
        raise RuntimeError("InputPadder unavailable — vendored ProPainter missing at third_party/ProPainter/")
    padder = InputPadder(a_b3hw.shape)
    ap, bp = padder.pad(a_b3hw, b_b3hw)
    _, flow = raft(ap, bp, iters=iters, test_mode=True)
    return padder.unpad(flow)


# =====================================================================
# Lucas-Kanade pyramid fallback (cv2-free, pure torch) — used when
# ProPainter / RAFT unavailable so the node still produces flow.
# =====================================================================
@torch.no_grad()
def _lk_pyramid_flow(a: torch.Tensor, b: torch.Tensor, levels: int = 3,
                     win: int = 9, iters: int = 5) -> torch.Tensor:
    """Coarse Lucas-Kanade dense flow. a,b: (B,1,H,W) gray; returns (B,2,H,W)."""
    B, _, H, W = a.shape
    flow = torch.zeros(B, 2, H, W, device=a.device, dtype=a.dtype)
    pyr_a = [a]
    pyr_b = [b]
    for _ in range(levels - 1):
        pyr_a.append(F.avg_pool2d(pyr_a[-1], 2))
        pyr_b.append(F.avg_pool2d(pyr_b[-1], 2))
    flow_lvl = torch.zeros(B, 2, pyr_a[-1].shape[-2], pyr_a[-1].shape[-1],
                           device=a.device, dtype=a.dtype)
    for lvl in reversed(range(levels)):
        if lvl != levels - 1:
            flow_lvl = F.interpolate(flow_lvl, scale_factor=2,
                                     mode="bilinear", align_corners=False) * 2.0
        ai, bi = pyr_a[lvl], pyr_b[lvl]
        # Spatial gradients (Sobel)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          device=a.device, dtype=a.dtype).view(1, 1, 3, 3)
        sy = sx.transpose(2, 3)
        Ix = F.conv2d(ai, sx, padding=1)
        Iy = F.conv2d(ai, sy, padding=1)
        for _ in range(iters):
            warped = F.grid_sample(bi, _grid_from_flow(flow_lvl),
                                   mode="bilinear", padding_mode="border",
                                   align_corners=True)
            It = warped - ai
            kw = torch.ones(1, 1, win, win, device=a.device, dtype=a.dtype)
            Ixx = F.conv2d(Ix * Ix, kw, padding=win // 2)
            Iyy = F.conv2d(Iy * Iy, kw, padding=win // 2)
            Ixy = F.conv2d(Ix * Iy, kw, padding=win // 2)
            Ixt = F.conv2d(Ix * It, kw, padding=win // 2)
            Iyt = F.conv2d(Iy * It, kw, padding=win // 2)
            det = (Ixx * Iyy - Ixy * Ixy).clamp_min(1e-4)
            du = (-Iyy * Ixt + Ixy * Iyt) / det
            dv = (Ixy * Ixt - Ixx * Iyt) / det
            flow_lvl = flow_lvl + torch.cat((du, dv), dim=1)
        if lvl == 0:
            flow = flow_lvl
    return flow


def _rgb_to_gray(x_b3hw: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.299, 0.587, 0.114], device=x_b3hw.device,
                     dtype=x_b3hw.dtype).view(1, 3, 1, 1)
    return (x_b3hw * w).sum(dim=1, keepdim=True)


# =====================================================================
# Public functional API (importable by the node and by tests)
# =====================================================================
def compute_flow(frame_a: torch.Tensor, frame_b: torch.Tensor,
                 iters: int = 20, consistency_thr: float = 1.5
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    frame_a / frame_b: (B, H, W, C) Comfy IMAGE.
    Returns:
        flow_bhw2     (B, H, W, 2)  pixel-unit U/V
        warped_bhwc   (B, H, W, C)  a warped toward b
        consistency   (B, H, W)     1.0 where fwd/bwd agree
    """
    dev = get_device()
    a = frame_a.to(dev)
    b = frame_b.to(dev)
    if a.shape != b.shape:
        raise ValueError(f"frame_a/b shape mismatch: {a.shape} vs {b.shape}")

    if HAS_PROPAINTER:
        models = load_models(half=False)
        a_pp = _to_raft(a, models.device)
        b_pp = _to_raft(b, models.device)
        fwd = _raft_pair(models.raft, a_pp, b_pp, iters=iters)
        bwd = _raft_pair(models.raft, b_pp, a_pp, iters=iters)
    else:
        log.warning("[FlowRefineMEC] ProPainter unavailable — using LK fallback")
        a_g = _rgb_to_gray(a.permute(0, 3, 1, 2).contiguous())
        b_g = _rgb_to_gray(b.permute(0, 3, 1, 2).contiguous())
        fwd = _lk_pyramid_flow(a_g, b_g, levels=3, win=9, iters=5)
        bwd = _lk_pyramid_flow(b_g, a_g, levels=3, win=9, iters=5)

    grid_fwd = _grid_from_flow(fwd)
    warped = F.grid_sample(b.permute(0, 3, 1, 2), grid_fwd,
                           mode="bilinear", padding_mode="border",
                           align_corners=True)
    warped_bhwc = warped.permute(0, 2, 3, 1).contiguous()

    # Bidirectional consistency: bwd warped by fwd should cancel fwd.
    bwd_warped = F.grid_sample(bwd, grid_fwd, mode="bilinear",
                               padding_mode="border", align_corners=True)
    diff = (fwd + bwd_warped).pow(2).sum(dim=1).sqrt()  # (B, H, W)
    consistency = (diff < consistency_thr).to(fwd.dtype)

    flow_bhw2 = fwd.permute(0, 2, 3, 1).contiguous()
    return flow_bhw2.cpu(), warped_bhwc.cpu(), consistency.cpu()


# =====================================================================
# Node
# =====================================================================
class FlowRefineMEC:
    DESCRIPTION = ("RAFT-based dense optical flow with bidirectional consistency "
                   "and grid_sample warping. Falls back to multi-scale LK when "
                   "ProPainter / RAFT isn't installed.")
    CATEGORY = "C2C/VFX"
    FUNCTION = "refine_flow"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("flow_field_rgb", "warped", "consistency")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_a":          ("IMAGE",),
                "frame_b":          ("IMAGE",),
                "iters":            ("INT", {"default": 20, "min": 1, "max": 100}),
                "consistency_thr":  ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0,
                                               "step": 0.05}),
            },
            "optional": {
                "mask": ("MASK",),  # restricts visualization but flow is dense
            },
        }

    @classmethod
    def IS_CHANGED(cls, frame_a, frame_b, iters, consistency_thr, mask=None, **kwargs):
        return hash_args_and_kwargs(frame_a, frame_b, iters, consistency_thr, mask, **kwargs)

    def refine_flow(self, frame_a, frame_b, iters, consistency_thr, mask=None):
        if not isinstance(frame_a, torch.Tensor) or frame_a.ndim != 4 or frame_a.shape[-1] != 3:
            raise ValueError(
                f"FlowRefineMEC: frame_a must be IMAGE [B,H,W,3], got {tuple(getattr(frame_a, 'shape', ()))}"
            )
        if not isinstance(frame_b, torch.Tensor) or frame_b.ndim != 4 or frame_b.shape[-1] != 3:
            raise ValueError(
                f"FlowRefineMEC: frame_b must be IMAGE [B,H,W,3], got {tuple(getattr(frame_b, 'shape', ()))}"
            )
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.ndim not in (3, 4)):
            raise ValueError(
                f"FlowRefineMEC: mask must be MASK [B,H,W], got {tuple(getattr(mask, 'shape', ()))}"
            )
        with torch.inference_mode():
            return self._refine_flow_impl(frame_a, frame_b, iters, consistency_thr, mask)

    def _refine_flow_impl(self, frame_a, frame_b, iters, consistency_thr, mask=None):
        flow, warped, consistency = compute_flow(
            frame_a, frame_b, iters=iters, consistency_thr=consistency_thr,
        )
        # Pack flow into an IMAGE (B,H,W,3): R=normalised U, G=normalised V, B=magnitude.
        u = flow[..., 0]
        v = flow[..., 1]
        mag = (u * u + v * v).sqrt()
        norm = mag.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        flow_rgb = torch.stack(
            (
                (u / norm).clamp(-1.0, 1.0) * 0.5 + 0.5,
                (v / norm).clamp(-1.0, 1.0) * 0.5 + 0.5,
                (mag / norm).clamp(0.0, 1.0),
            ),
            dim=-1,
        ).contiguous()
        if mask is not None:
            m = mask.to(consistency.device)
            if m.dim() == 4:
                m = m.squeeze(-1)
            consistency = consistency * m.cpu()
        return (flow_rgb, warped, consistency)


NODE_CLASS_MAPPINGS = {"FlowRefineMEC": FlowRefineMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"FlowRefineMEC": "Flow Refine — RAFT"}
