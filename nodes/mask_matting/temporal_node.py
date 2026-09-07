"""MaskTemporalMEC — temporal stabilization + integrity check for mask batches.

This is the third pillar of the mask+matting stack on `main`, sitting between
``MaskOpsMEC`` (which produces per-frame masks) and ``MaskRefineMEC`` (which
cleans them spatially). It addresses two longstanding issues that show up on
video / animation batches:

1.  **Mask drag** — the existing ``_temporal_gaussian_smooth`` Gaussian along
    the batch dimension blurs the mask in time. On fast-moving subjects this
    leaves a smeared trail on every limb edge. We add an optical-flow warp
    mode (RAFT) so each frame's mask is *warped* into the next frame's
    coordinate system before blending — preserves crisp edges, removes
    flicker.

2.  **Silent failures** — when SAM 3 / SAM 3.1 drops out for a frame (text
    prompt mismatch, occlusion, low confidence) the produced mask is just
    blank or wildly different from its neighbours. We compute per-frame
    area / centroid / IoU drift versus the previous frame and emit a JSON
    report + a human-readable warning string. The report is also pushed to
    ``window.__C2C_MASK_INTEGRITY__`` (when ``workflow_doctor`` is loaded)
    so the diagnostics sidebar can pick it up.

Modes
-----
* ``none``        — passthrough.
* ``gaussian``    — legacy Gaussian along batch (kept for back-compat).
* ``raft_flow``   — torchvision RAFT-small optical flow, warp prev mask
                    forward, blend with current.

Resource budget
---------------
* RAFT weights (~50 MB) are lazy-loaded on first use and released by
  ``free_vram()`` on errors.
* The model runs in inference_mode + fp16 (when on CUDA) at the input
  resolution.
* Single batch pass: O(B) forward calls, no growth in held state.

License: Apache-2.0 (this file). RAFT weights are torchvision's standard
``Raft_Small_Weights.DEFAULT`` redistributable under torchvision's BSD-3.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .._is_changed_util import hash_args_and_kwargs
from .utils import free_vram

log = logging.getLogger("MEC.MaskTemporal")

# ── lazy RAFT loader (torchvision) ──────────────────────────────────────
_RAFT_MODEL = None
_RAFT_DEV = None


def _load_raft(device: str):
    """Lazy-load RAFT-small. Returns (model, preprocess) or raises."""
    global _RAFT_MODEL, _RAFT_DEV
    if _RAFT_MODEL is not None and _RAFT_DEV == device:
        return _RAFT_MODEL
    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "torchvision optical_flow.raft_small unavailable — "
            f"upgrade torchvision (>=0.13) to use raft_flow mode: {exc}"
        )
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=False).to(device).eval()
    _RAFT_MODEL = (model, weights.transforms())
    _RAFT_DEV = device
    return _RAFT_MODEL


def _release_raft():
    global _RAFT_MODEL, _RAFT_DEV
    _RAFT_MODEL = None
    _RAFT_DEV = None
    free_vram()


# ── flow helpers ────────────────────────────────────────────────────────
def _warp_mask_by_flow(mask_hw: torch.Tensor, flow_2hw: torch.Tensor) -> torch.Tensor:
    """Warp a (H,W) mask by a (2,H,W) flow field via grid_sample.

    Flow is in pixel units, channel-0 = dx, channel-1 = dy. Output is the
    mask warped *forward* by the flow (i.e. the value at output pixel
    (x,y) is the input mask at (x-dx, y-dy)).
    """
    H, W = mask_hw.shape[-2:]
    device = mask_hw.device
    dtype = mask_hw.dtype
    # Build identity grid in normalized [-1,1] coords.
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = flow_2hw[0].to(device=device, dtype=dtype)
    dy = flow_2hw[1].to(device=device, dtype=dtype)
    src_x = xx - dx
    src_y = yy - dy
    nx = (src_x / max(W - 1, 1)) * 2.0 - 1.0
    ny = (src_y / max(H - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([nx, ny], dim=-1).unsqueeze(0)  # (1,H,W,2)
    inp = mask_hw.view(1, 1, H, W)
    out = F.grid_sample(inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[0, 0].clamp(0, 1)


def _raft_flow_pair(img_a_chw: torch.Tensor, img_b_chw: torch.Tensor, device: str) -> torch.Tensor:
    """Compute optical flow from img_a -> img_b. Returns (2,H,W) on `device`."""
    model, preprocess = _load_raft(device)
    a = img_a_chw.unsqueeze(0).to(device)
    b = img_b_chw.unsqueeze(0).to(device)
    a, b = preprocess(a, b)
    with torch.inference_mode():
        flows = model(a, b)
    return flows[-1][0]  # final iteration, drop batch


def _gaussian_temporal(mask_bhw: torch.Tensor, sigma: float) -> torch.Tensor:
    """1-D Gaussian along the batch (time) axis. Same as legacy
    ``_temporal_gaussian_smooth`` in inpaint_suite, reproduced here so
    this node is self-contained.
    """
    if sigma <= 0 or mask_bhw.shape[0] <= 1:
        return mask_bhw
    k = max(3, int(round(sigma * 4)) | 1)  # odd
    half = k // 2
    coords = torch.arange(-half, half + 1, dtype=mask_bhw.dtype, device=mask_bhw.device)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    pad = mask_bhw[:1].expand(half, *mask_bhw.shape[1:])
    pad2 = mask_bhw[-1:].expand(half, *mask_bhw.shape[1:])
    padded = torch.cat([pad, mask_bhw, pad2], dim=0)
    out = torch.zeros_like(mask_bhw)
    for i in range(k):
        out = out + padded[i : i + mask_bhw.shape[0]] * g[i]
    return out.clamp(0, 1)


# ── integrity check ─────────────────────────────────────────────────────
def _mask_stats(m_hw: torch.Tensor) -> Tuple[float, Tuple[float, float]]:
    """Return (area_fraction, (cx_norm, cy_norm))."""
    H, W = m_hw.shape[-2:]
    bin_m = (m_hw > 0.5).float()
    area = float(bin_m.sum().item())
    total = float(H * W)
    if area < 1e-6:
        return 0.0, (0.5, 0.5)
    ys, xs = torch.nonzero(bin_m, as_tuple=True)
    cx = float(xs.float().mean().item()) / max(W - 1, 1)
    cy = float(ys.float().mean().item()) / max(H - 1, 1)
    return area / total, (cx, cy)


def _iou(a_hw: torch.Tensor, b_hw: torch.Tensor) -> float:
    a = (a_hw > 0.5).float()
    b = (b_hw > 0.5).float()
    inter = float((a * b).sum().item())
    union = float(((a + b) > 0).float().sum().item())
    return inter / max(union, 1e-6)


def compute_integrity(mask_bhw: torch.Tensor,
                       drop_threshold: float = 0.4,
                       jump_threshold: float = 0.15) -> Dict[str, Any]:
    """Per-frame drift detection.

    * Total mask dropout (area falls to 0) and re-appearance are always
      flagged.
    * area_ratio: frame i's area / frame i-1's area; when the relative
      drop meets or exceeds ``drop_threshold`` the frame is flagged
      (e.g. ``drop_threshold=0.4`` flags any frame whose mask shrank to
      60% or less of the previous frame).
    * centroid_delta: L2 distance in normalised coords; values greater
      than ``jump_threshold`` flag the frame.
    * iou_prev: IoU with the previous binary mask; when both frames
      are non-empty but IoU is below ``1 - drop_threshold`` the
      subject has likely jumped or been re-targeted.

    Returns dict suitable for JSON serialisation + a list of flagged
    frame indices.
    """
    B = mask_bhw.shape[0]
    frames: List[Dict[str, Any]] = []
    flagged: List[int] = []
    prev_area: Optional[float] = None
    prev_cx: Optional[float] = None
    prev_cy: Optional[float] = None
    prev_mask: Optional[torch.Tensor] = None
    for i in range(B):
        area, (cx, cy) = _mask_stats(mask_bhw[i])
        rec: Dict[str, Any] = {"i": i, "area": round(area, 5),
                                "cx": round(cx, 4), "cy": round(cy, 4)}
        if prev_area is not None:
            ratio = area / max(prev_area, 1e-6)
            cdist = float(((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5)
            iou = _iou(mask_bhw[i], prev_mask) if prev_mask is not None else 1.0
            rec.update({
                "area_ratio": round(ratio, 4),
                "centroid_delta": round(cdist, 4),
                "iou_prev": round(iou, 4),
            })
            why: List[str] = []
            # Total mask dropout (mask had area, now zero) — always
            # flagged regardless of drop_threshold; this is the single
            # most important integrity event.
            if prev_area > 0 and area <= 0:
                why.append("mask dropout (area=0)")
            # Total mask reappearance after dropout.
            elif prev_area <= 0 and area > 0:
                why.append("mask reappeared after dropout")
            # Partial area drop (relative).  ratio = area / prev_area;
            # a drop of >= drop_threshold means new area is <= (1 - dt) of
            # previous — e.g. dt=0.4 flags any frame where mask shrank to
            # 60% or less of its previous size.
            elif area > 0 and prev_area > 0 and ratio <= (1.0 - drop_threshold):
                why.append(f"area drop to {ratio:.2f}x")
            if cdist > jump_threshold:
                why.append(f"centroid jump {cdist:.2f}")
            # IoU disconnect: when both masks are non-empty but their
            # overlap is below (1 - drop_threshold), the subject has
            # likely jumped or been re-targeted.
            if iou < (1.0 - drop_threshold) and prev_area > 0 and area > 0:
                why.append(f"low IoU {iou:.2f}")
            if why:
                rec["warn"] = "; ".join(why)
                flagged.append(i)
        frames.append(rec)
        prev_area, prev_cx, prev_cy = area, cx, cy
        prev_mask = mask_bhw[i]
    return {
        "B": B,
        "flagged_count": len(flagged),
        "flagged_frames": flagged,
        "frames": frames,
        "drop_threshold": drop_threshold,
        "jump_threshold": jump_threshold,
    }


# ── ComfyUI node ────────────────────────────────────────────────────────
class MaskTemporalMEC:
    CATEGORY = "MaskEditControl/MaskMatting"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING", "STRING")
    RETURN_NAMES = ("mask", "integrity_json", "warning")
    DESCRIPTION = (
        "Temporal stabilization + integrity check for video mask batches. "
        "Modes: none/gaussian/raft_flow. Emits a per-frame integrity report "
        "(area / centroid / IoU drift) and a human-readable warning string "
        "listing flagged frame indices."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":  ("IMAGE",),
                "mask":   ("MASK",),
                "temporal_mode": (["none", "gaussian", "raft_flow"], {"default": "none"}),
                "blend":  ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Mix factor: 1.0 = pure warped-prev, 0.0 = current only."}),
                "sigma":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.1,
                                      "tooltip": "Gaussian sigma (gaussian mode only)."}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "drop_threshold":  ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "Area-ratio / IoU below this flags the frame."}),
                "jump_threshold":  ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "Centroid jump in normalized coords above this flags the frame."}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, image, mask, temporal_mode, blend, sigma, device,
                   drop_threshold, jump_threshold, **kwargs):
        return hash_args_and_kwargs(
            image, mask, temporal_mode, blend, sigma, device,
            drop_threshold, jump_threshold, **kwargs,
        )

    def run(self, image, mask, temporal_mode, blend, sigma, device,
             drop_threshold, jump_threshold):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("MaskTemporalMEC expects IMAGE tensor [B,H,W,C]")
        if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3, 4):
            raise ValueError("MaskTemporalMEC expects MASK tensor [H,W] or [B,H,W]")
        with torch.inference_mode():
            return self._run_impl(
                image, mask, temporal_mode, blend, sigma, device,
                drop_threshold, jump_threshold,
            )

    def _run_impl(self, image, mask, temporal_mode, blend, sigma, device,
                  drop_threshold, jump_threshold):
        # Normalize tensor shapes.
        m = mask
        if m.dim() == 4:  # (B,H,W,1)
            m = m.squeeze(-1)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        m = m.float().clamp(0, 1)
        B, H, W = m.shape

        dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

        if temporal_mode == "none" or B <= 1:
            out = m
        elif temporal_mode == "gaussian":
            out = _gaussian_temporal(m.to(dev), float(sigma)).to(m.device)
        elif temporal_mode == "raft_flow":
            try:
                img = image
                if img.dim() == 4 and img.shape[-1] in (1, 3):
                    img_bchw = img.permute(0, 3, 1, 2).contiguous()
                else:
                    img_bchw = img
                if img_bchw.shape[1] == 1:
                    img_bchw = img_bchw.repeat(1, 3, 1, 1)
                img_bchw = img_bchw.float().to(dev)
                # Ensure spatial alignment.
                if img_bchw.shape[-2:] != (H, W):
                    img_bchw = F.interpolate(img_bchw, size=(H, W), mode="bilinear", align_corners=False)
                out_frames = [m[0].to(dev)]
                for i in range(1, B):
                    flow = _raft_flow_pair(img_bchw[i - 1], img_bchw[i], dev)
                    warped = _warp_mask_by_flow(out_frames[-1], flow)
                    fused = float(blend) * warped + (1.0 - float(blend)) * m[i].to(dev)
                    out_frames.append(fused.clamp(0, 1))
                out = torch.stack(out_frames, dim=0).to(m.device)
            except Exception as exc:
                log.warning("[MaskTemporal] raft_flow failed (%s) — passthrough.", exc)
                _release_raft()
                out = m
        else:
            out = m

        report = compute_integrity(out, float(drop_threshold), float(jump_threshold))
        warning = ""
        if report["flagged_count"] > 0:
            flagged_str = ", ".join(str(i) for i in report["flagged_frames"][:32])
            extra = "" if report["flagged_count"] <= 32 else f" (+{report['flagged_count']-32} more)"
            warning = (
                f"[MaskIntegrity] {report['flagged_count']}/{report['B']} "
                f"frames flagged: {flagged_str}{extra}"
            )
            log.warning(warning)

        # Release RAFT after the call if we loaded it (resource discipline).
        if temporal_mode == "raft_flow":
            _release_raft()

        # Publish to the HUD bridge so the Mask Integrity sidebar can show it.
        try:
            from .integrity_bridge import publish as _integrity_publish
            extra: Dict[str, Any] = {}
            frames_arr = report.get("frames", [])
            if isinstance(frames_arr, list) and frames_arr:
                extra["area"] = [float(f.get("area", 0.0)) for f in frames_arr]
                extra["centroid_dx"] = [float(f.get("centroid_delta", 0.0)) for f in frames_arr]
                extra["iou_prev"] = [float(f.get("iou_prev", 1.0)) for f in frames_arr]
            _integrity_publish("MaskTemporalMEC", report, extra)
        except Exception:
            log.exception("[MaskTemporal] integrity publish failed")

        return (out, json.dumps(report), warning)


NODE_CLASS_MAPPINGS = {"MaskTemporalMEC": MaskTemporalMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskTemporalMEC": "Mask Temporal Stabilizer + Integrity"}
