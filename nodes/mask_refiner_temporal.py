"""Mask Refiner — Temporal (MEC).

Without any extra training, the masking quality of SAM / SeC / "any-segmentation"
models can be substantially improved on video sequences by exploiting two
sources of information they don't see in single-frame mode:

  1. **Temporal coherence** — neighbouring frames have nearly-identical
     foregrounds.  An object that flickers in/out of a mask between frame
     t-1 and t+1 is almost always a segmentation false-negative on frame t.
  2. **Image-guided edge structure** — the original RGB has crisp object
     edges that the upstream segmenter often blurs or under-cuts.

This node combines both signals via three CPU-only passes (no torch, no
extra weights, sub-second per frame on 720p):

  Pass A — Optical-flow warp & fuse:
    Compute Farneback dense flow between every consecutive pair, warp the
    previous mask forward into the current frame, and take a weighted
    union with the upstream mask.  Repeats backward to catch single-frame
    drop-outs.  A Kalman-style centroid-drift check prevents fusing
    masks that have already diverged (occlusion / cut).

  Pass B — Temporal cosine smoothing:
    Average each frame's mask with its neighbours using a cosine window
    so isolated spikes are damped without softening genuine transitions.

  Pass C — Image-guided edge snap:
    Apply OpenCV's guided filter (or a fast bilateral fallback) using the
    RGB frame as the guide so the mask edge re-locks onto the real
    object outline.

Inputs:  IMAGE (B,H,W,C), MASK (B,H,W) — both float32, [0,1].
Output:  MASK (B,H,W) refined.

Dependencies:
  cv2 (opencv-python).  ximgproc is auto-detected — if the user has
  opencv-contrib-python the guided filter is used; otherwise a bilateral
  filter on the mask provides a still-useful fallback.  No new packages
  required if the user already has opencv-python (every ComfyUI install
  effectively does).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch

log = logging.getLogger("MEC.mask_refiner_temporal")

# Optional cv2 — degrade with a clear error if absent.
try:
    import cv2  # type: ignore
    _CV2 = True
except Exception as _e:  # noqa: BLE001
    cv2 = None  # type: ignore
    _CV2 = False
    log.warning("[MEC] mask_refiner_temporal: cv2 not importable (%s); "
                "install opencv-python to enable the node.", _e)

# Optional ximgproc (only in opencv-contrib-python).
_HAS_XIMG = False
if _CV2:
    try:
        _ = cv2.ximgproc.guidedFilter  # type: ignore[attr-defined]
        _HAS_XIMG = True
    except Exception:
        _HAS_XIMG = False


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────
def _to_uint8_gray(frame_rgb_f32: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor((frame_rgb_f32 * 255.0).clip(0, 255).astype(np.uint8),
                     cv2.COLOR_RGB2GRAY)
    return g


def _warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp a single-channel float mask forward along a dense flow field.

    flow[y,x] = (dx, dy) describes how the pixel at (x,y) in the SOURCE
    moves into the NEXT frame.  To produce the next-frame mask we sample
    the source at (x - dx, y - dy)  --  i.e. invert the displacement.
    """
    H, W = mask.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (xx - flow[..., 0]).astype(np.float32)
    map_y = (yy - flow[..., 1]).astype(np.float32)
    return cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _centroid(m: np.ndarray) -> Tuple[float, float, float]:
    """Return (cx, cy, mass).  Mass is sum of mask values."""
    mass = float(m.sum())
    if mass < 1.0:
        return (0.0, 0.0, 0.0)
    H, W = m.shape
    ys, xs = np.mgrid[0:H, 0:W]
    cx = float((xs * m).sum() / mass)
    cy = float((ys * m).sum() / mass)
    return (cx, cy, mass)


def _guided_edge_snap(rgb_f32: np.ndarray, mask_f32: np.ndarray,
                      radius: int, eps: float) -> np.ndarray:
    """Snap mask edges to the image edges. RGB is the guide."""
    if _HAS_XIMG:
        guide = (rgb_f32 * 255.0).clip(0, 255).astype(np.uint8)
        m8 = (mask_f32 * 255.0).clip(0, 255).astype(np.uint8)
        out = cv2.ximgproc.guidedFilter(  # type: ignore[attr-defined]
            guide=guide, src=m8, radius=int(radius), eps=float(eps * 255.0 * 255.0)
        )
        return out.astype(np.float32) / 255.0
    # Fallback: joint bilateral via cv2.bilateralFilter on the mask. Not
    # truly guided but still produces a noticeably crisper edge for typical
    # subjects. The radius/eps are reinterpreted into bilateral params.
    m8 = (mask_f32 * 255.0).clip(0, 255).astype(np.uint8)
    sigma_color = max(8.0, 60.0 * (eps ** 0.5))
    sigma_space = max(2.0, float(radius))
    out = cv2.bilateralFilter(m8, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return out.astype(np.float32) / 255.0


def _cosine_window(half_width: int) -> np.ndarray:
    """Symmetric cosine window of length 2*half_width+1, normalized to sum=1."""
    n = 2 * half_width + 1
    if n == 1:
        return np.array([1.0], dtype=np.float32)
    x = np.linspace(-np.pi / 2.0, np.pi / 2.0, n, dtype=np.float32)
    w = np.cos(x) ** 2  # raised-cosine; peaks at centre
    return w / w.sum()


# ─────────────────────────────────────────────────────────────────────
#  Node
# ─────────────────────────────────────────────────────────────────────
class MaskRefinerTemporalMEC:
    """Improve segmentation-model masks on video using optical flow, temporal
    smoothing, and image-guided edge snapping.  CPU-only, no extra weights,
    works with any upstream mask (SAM / SeC / Matte-Anything / hand-painted)."""

    VRAM_TIER = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Source video frames (B,H,W,C), float32 in [0,1].",
                }),
                "mask": ("MASK", {
                    "tooltip": "Per-frame mask from upstream segmenter (B,H,W).",
                }),
                "enable_flow_fuse": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Pass A: warp previous-frame mask via optical flow "
                               "and fuse with current frame to fill drop-outs.",
                }),
                "fuse_weight": ("FLOAT", {
                    "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much of the warped neighbour mask to keep. "
                               "0=ignore, 1=trust neighbour as much as current.",
                }),
                "drift_threshold": ("FLOAT", {
                    "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "If centroid moves more than this fraction of the "
                               "image diagonal between frames, suppress fusion "
                               "(prevents bleed across cuts / occlusions).",
                }),
                "temporal_window": ("INT", {
                    "default": 2, "min": 0, "max": 6, "step": 1,
                    "tooltip": "Pass B: half-width of the cosine smoothing "
                               "window. 0 disables temporal smoothing.",
                }),
                "enable_edge_snap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Pass C: image-guided filter that locks the "
                               "mask edge onto the RGB edge of the object.",
                }),
                "edge_radius": ("INT", {
                    "default": 8, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Guided/bilateral filter radius (pixels).",
                }),
                "edge_eps": ("FLOAT", {
                    "default": 0.001, "min": 1e-5, "max": 0.1, "step": 1e-4,
                    "tooltip": "Guided filter regularization. Smaller = sharper edges.",
                }),
                "binarize_threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "If > 0, hard-threshold the final mask at this value. "
                               "0 keeps soft mask (recommended for matting downstream).",
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = (
        "Temporally and edge-refined mask, same shape (B,H,W) as input.",
    )
    FUNCTION = "execute"
    CATEGORY = "MaskEditControl/Refine"
    DESCRIPTION = (
        "Improve any per-frame mask (SAM / SeC / hand-painted) using "
        "optical-flow temporal fusion, cosine-window smoothing, and "
        "RGB-guided edge snapping. CPU-only, no extra model weights."
    )

    # -----------------------------------------------------------------
    def execute(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        enable_flow_fuse: bool,
        fuse_weight: float,
        drift_threshold: float,
        temporal_window: int,
        enable_edge_snap: bool,
        edge_radius: int,
        edge_eps: float,
        binarize_threshold: float,
    ) -> Tuple[torch.Tensor]:

        if not _CV2:
            raise RuntimeError(
                "MaskRefinerTemporalMEC requires opencv-python. "
                "Install with: pip install opencv-python  (or "
                "opencv-contrib-python for the higher-quality guided filter)."
            )

        # Tensor → numpy
        img_np = image.detach().cpu().float().numpy()      # (B,H,W,C) in [0,1]
        msk_np = mask.detach().cpu().float().numpy()       # (B,H,W)   in [0,1]
        if img_np.ndim != 4 or msk_np.ndim != 3:
            raise ValueError(
                f"Expected image (B,H,W,C) and mask (B,H,W), got "
                f"image={img_np.shape}, mask={msk_np.shape}"
            )

        B, H, W, C = img_np.shape
        if msk_np.shape[0] != B:
            # Broadcast a single mask across all frames.
            if msk_np.shape[0] == 1:
                msk_np = np.broadcast_to(msk_np, (B, H, W)).copy()
            else:
                raise ValueError(
                    f"Mask batch ({msk_np.shape[0]}) must match image batch ({B}) or be 1."
                )

        # Clamp.
        img_np = img_np.clip(0.0, 1.0)
        msk_np = msk_np.clip(0.0, 1.0)
        out = msk_np.copy()
        diag = float(np.hypot(H, W))

        # ---------------- Pass A: optical-flow fuse ------------------
        if enable_flow_fuse and B >= 2:
            grays = [_to_uint8_gray(img_np[i]) for i in range(B)]

            # Forward pass: warp prev → current
            for i in range(1, B):
                flow = cv2.calcOpticalFlowFarneback(
                    grays[i - 1], grays[i], None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                warped = _warp_mask(out[i - 1], flow)
                # Centroid drift gate (prevents fusion across cuts).
                cx_w, cy_w, m_w = _centroid(warped)
                cx_c, cy_c, m_c = _centroid(out[i])
                if m_w > 1.0 and m_c > 1.0:
                    drift = np.hypot(cx_w - cx_c, cy_w - cy_c) / max(diag, 1.0)
                    if drift > drift_threshold:
                        continue  # don't fuse — divergent
                # Weighted union: take element-wise max, biased by fuse_weight
                out[i] = np.maximum(out[i], warped * fuse_weight)
                np.clip(out[i], 0.0, 1.0, out=out[i])

            # Backward pass: warp next → current to catch leading drop-outs
            for i in range(B - 2, -1, -1):
                flow = cv2.calcOpticalFlowFarneback(
                    grays[i + 1], grays[i], None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                warped = _warp_mask(out[i + 1], flow)
                cx_w, cy_w, m_w = _centroid(warped)
                cx_c, cy_c, m_c = _centroid(out[i])
                if m_w > 1.0 and m_c > 1.0:
                    drift = np.hypot(cx_w - cx_c, cy_w - cy_c) / max(diag, 1.0)
                    if drift > drift_threshold:
                        continue
                out[i] = np.maximum(out[i], warped * (fuse_weight * 0.8))
                np.clip(out[i], 0.0, 1.0, out=out[i])

        # ---------------- Pass B: temporal smoothing -----------------
        if temporal_window > 0 and B > 1:
            half = int(temporal_window)
            window = _cosine_window(half)  # length 2*half+1
            padded = np.concatenate([
                np.repeat(out[:1], half, axis=0),
                out,
                np.repeat(out[-1:], half, axis=0),
            ], axis=0)  # (B + 2*half, H, W)
            smoothed = np.zeros_like(out)
            for k, w in enumerate(window):
                smoothed += w * padded[k:k + B]
            out = smoothed.clip(0.0, 1.0)

        # ---------------- Pass C: image-guided edge snap -------------
        if enable_edge_snap:
            for i in range(B):
                out[i] = _guided_edge_snap(img_np[i], out[i],
                                           radius=edge_radius, eps=edge_eps)
            np.clip(out, 0.0, 1.0, out=out)

        # ---------------- Optional binarize --------------------------
        if binarize_threshold > 0.0:
            out = (out >= float(binarize_threshold)).astype(np.float32)

        return (torch.from_numpy(out).to(mask.device, dtype=torch.float32),)


# ─────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {"MaskRefinerTemporalMEC": MaskRefinerTemporalMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskRefinerTemporalMEC": "Mask Refiner — Temporal (MEC)",
}
