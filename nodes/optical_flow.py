# FILE: nodes/optical_flow.py
# FEATURE: F3 — OpticalFlowMEC (RAFT primary, LK pyramid fallback, masked re-vector)
# INTEGRATES WITH: nodes/propainter_flow_refine.py (shared RAFT + LK kernels)
"""
Optical-flow re-vector node.

Computes dense forward flow between frame_a and frame_b. Uses RAFT (via
propainter_bridge) when available; otherwise pure-torch Lucas-Kanade pyramid.

Inside the supplied `mask`, the destination region of frame_a is re-vectored
toward frame_b using grid_sample. Outside the mask the original frame_a is
preserved untouched.

Hard rule:
    - NEVER blur to fake motion. Bad results -> assign DUMB_BLUR diag and
      rewrite. We never call torch.nn.functional.gaussian_blur or similar to
      hide flow holes.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .propainter_flow_refine import compute_flow

log = logging.getLogger("MEC.optical_flow")


def _grid_from_flow(flow_bchw: torch.Tensor) -> torch.Tensor:
    B, _, H, W = flow_bchw.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=flow_bchw.device, dtype=flow_bchw.dtype),
        torch.arange(W, device=flow_bchw.device, dtype=flow_bchw.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    s = grid + flow_bchw
    nx = 2.0 * s[:, 0] / max(W - 1, 1) - 1.0
    ny = 2.0 * s[:, 1] / max(H - 1, 1) - 1.0
    return torch.stack((nx, ny), dim=-1)


class OpticalFlowMEC:
    DESCRIPTION = ("Dense optical flow re-vector. RAFT primary, LK pyramid fallback. "
                   "Re-vectoring is restricted to the mask region; rest is untouched.")
    CATEGORY = "MaskEditControl/VFX"
    FUNCTION = "revector"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("re_vectored", "flow_rgb", "consistency")

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
                "mask":   ("MASK",),
                "scale":  ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0,
                                     "step": 0.05}),
            },
        }

    def revector(self, frame_a, frame_b, iters, consistency_thr,
                 mask=None, scale=1.0):
        flow_bhw2, warped, consistency = compute_flow(
            frame_a, frame_b, iters=iters, consistency_thr=consistency_thr,
        )
        flow_bchw = flow_bhw2.permute(0, 3, 1, 2).contiguous() * float(scale)

        a = frame_a.cpu()
        a_chw = a.permute(0, 3, 1, 2).contiguous()
        warped_chw = F.grid_sample(a_chw, _grid_from_flow(flow_bchw),
                                   mode="bilinear", padding_mode="border",
                                   align_corners=True)
        warped_bhwc = warped_chw.permute(0, 2, 3, 1).contiguous()

        if mask is not None:
            m = mask.cpu()
            if m.dim() == 4:
                m = m.squeeze(-1)
            m3 = m.unsqueeze(-1)
            re_vectored = a * (1.0 - m3) + warped_bhwc * m3
        else:
            re_vectored = warped_bhwc

        # Pack flow into IMAGE for downstream visualization (R=u, G=v, B=mag).
        u = flow_bhw2[..., 0]
        v = flow_bhw2[..., 1]
        mag = (u * u + v * v).sqrt()
        norm = mag.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        flow_rgb = torch.stack(
            (
                (u / norm * 0.5 + 0.5).clamp(0.0, 1.0),
                (v / norm * 0.5 + 0.5).clamp(0.0, 1.0),
                (mag / norm).clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        return (re_vectored, flow_rgb, consistency)


NODE_CLASS_MAPPINGS = {"OpticalFlowMEC": OpticalFlowMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"OpticalFlowMEC": "Optical Flow Re-Vector (MEC)"}
