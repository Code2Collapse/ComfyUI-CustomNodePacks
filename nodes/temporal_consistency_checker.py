"""
TemporalConsistencyCheckerMEC – Score frame-to-frame stability of a mask
or image sequence.

Metrics:
  - ``mask_iou``: Per-pair IoU of binary masks. Detects flicker /
    label switches.
  - ``pixel_diff``: L1 mean diff in [0, 1] across consecutive frames.
    No external deps. Pure-torch fallback.
  - ``flow_warp``: Compute Farneback optical flow (cv2) between
    frames i and i+1, warp frame i forward, then measure L1 diff
    against frame i+1. Requires OpenCV; otherwise transparently
    degrades to ``pixel_diff``.

Outputs a JSON of per-pair scores plus aggregate min/max/mean and a
``flicker_score`` (1 - mean of normalized stability).

This is read-only / diagnostic — input is passed through unchanged.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import torch

from ._is_changed_util import hash_args_and_kwargs

logger = logging.getLogger("MEC.TemporalConsistencyChecker")

try:
    import cv2  # type: ignore[import-not-found]
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _to_uint8_gray(arr_hw3: np.ndarray) -> np.ndarray:
    rgb = (np.clip(arr_hw3, 0.0, 1.0) * 255.0).astype(np.uint8)
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if _HAS_CV2 else rgb.mean(axis=-1).astype(np.uint8)
    return rgb


def _flow_warp_diff(prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> float:
    """Forward-warp prev → curr via Farneback flow, return mean L1 diff."""
    prev_g = _to_uint8_gray(prev_rgb)
    curr_g = _to_uint8_gray(curr_rgb)
    flow = cv2.calcOpticalFlowFarneback(
        prev_g, curr_g, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    h, w = prev_g.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(prev_rgb.astype(np.float32), map_x, map_y,
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return float(np.mean(np.abs(warped - curr_rgb.astype(np.float32))))


def _edge_band(a: np.ndarray, b: np.ndarray, thr: float, radius: int) -> np.ndarray:
    """Pixels near either matte's edge: soft alpha in either, or within `radius` of either binarised boundary."""
    soft = ((a > 0.02) & (a < 0.98)) | ((b > 0.02) & (b < 0.98))
    if not _HAS_CV2:
        return soft
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    band = soft.copy()
    for m in (a, b):
        bin_u8 = (m > thr).astype(np.uint8)
        edge = cv2.dilate(bin_u8, k) != cv2.erode(bin_u8, k)
        band |= edge
    return band


def _edge_stability(prev_m: np.ndarray, curr_m: np.ndarray, prev_img, curr_img, thr: float) -> tuple[float, bool]:
    """L7.127: mean alpha change in a band around the matte edge between two frames (0 = the edge holds, 1 = it
    jumps fully). With frames and OpenCV the previous matte is first moved along the picture's optical flow
    (Farneback, sampled current -> previous), so an edge that follows the subject does not count as flicker."""
    h, w = curr_m.shape
    moved = prev_m
    compensated = False
    if _HAS_CV2 and prev_img is not None and curr_img is not None:
        flow = cv2.calcOpticalFlowFarneback(_to_uint8_gray(curr_img), _to_uint8_gray(prev_img), None,
                                            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
                                            poly_sigma=1.2, flags=0)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        moved = cv2.remap(prev_m.astype(np.float32), gx + flow[..., 0], gy + flow[..., 1],
                          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        compensated = True
    radius = max(2, int(round(min(h, w) * 0.01)))
    band = _edge_band(moved, curr_m, thr, radius)
    if not band.any():
        return 0.0, compensated
    return float(np.abs(moved[band] - curr_m[band]).mean()), compensated


def _mask_iou(prev: torch.Tensor, curr: torch.Tensor, thr: float = 0.5) -> float:
    a = (prev > thr)
    b = (curr > thr)
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    if union == 0:
        return 1.0  # both empty → consider them perfectly consistent
    return float(inter) / float(union)


class TemporalConsistencyCheckerMEC:
    """Diagnose flicker / drift in mask or image sequences."""

    # L7.127: edge_stability appended last, so saved values keep their meaning (Mask Track reuses this list)
    METRICS = ["mask_iou", "pixel_diff", "flow_warp", "edge_stability"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metric": (cls.METRICS, {
                    "default": "pixel_diff",
                    "tooltip": (
                        "mask_iou: requires MASK input; per-frame IoU.\n"
                        "pixel_diff: L1 diff between consecutive frames (image or mask).\n"
                        "flow_warp: Farneback flow + warp + L1 diff (requires OpenCV).\n"
                        "edge_stability: needs MASK; how much the matte EDGE jumps between frames (alpha change "
                        "in a band around the edge). With the image wired it follows the picture's motion first, "
                        "so only flicker counts - a steady edge can be grown or shrunk later, a flickering one "
                        "cannot."
                    ),
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Image batch (B,H,W,3) for pixel_diff / flow_warp.",
                }),
                "mask": ("MASK", {
                    "tooltip": "Mask batch (B,H,W) for mask_iou or pixel_diff.",
                }),
                "binarize_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "Threshold to binarize mask values when computing mask_iou",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT", "STRING")
    RETURN_NAMES = ("image_passthrough", "mask_passthrough", "flicker_score", "report_json")
    OUTPUT_TOOLTIPS = (
        "Input image batch passed through unchanged.",
        "Input mask batch passed through unchanged.",
        "Aggregate flicker score in [0, 1] (lower is more stable).",
        "JSON report with per-pair scores, min/max/mean, and metadata.",
    )
    FUNCTION = "check"
    CATEGORY = "C2C/Diagnostics"
    DESCRIPTION = (
        "Score frame-to-frame stability of an IMAGE or MASK batch. "
        "Returns flicker_score in [0, 1] (lower = more stable) and a per-pair JSON report."
    )

    @classmethod
    def IS_CHANGED(cls, metric, image=None, mask=None, binarize_threshold=0.5,
                   **kwargs):
        return hash_args_and_kwargs(
            metric, image, mask, binarize_threshold, **kwargs,
        )

    def check(
        self,
        metric: str,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        binarize_threshold: float = 0.5,
    ):
        if image is not None and (
            not isinstance(image, torch.Tensor) or image.ndim != 4
        ):
            raise ValueError(
                "TemporalConsistencyCheckerMEC expects IMAGE tensor [B,H,W,C]"
            )
        if mask is not None and (
            not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3)
        ):
            raise ValueError(
                "TemporalConsistencyCheckerMEC expects MASK tensor [H,W] or [B,H,W]"
            )
        with torch.no_grad():
            return self._check_impl(metric, image, mask, binarize_threshold)

    def _check_impl(
        self,
        metric: str,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        binarize_threshold: float = 0.5,
    ):
        if metric in ("mask_iou", "edge_stability") and mask is None:
            raise ValueError(f"{metric} requires a MASK input.")
        if metric in ("pixel_diff", "flow_warp") and image is None and mask is None:
            raise ValueError(f"{metric} requires either an IMAGE or a MASK input.")

        # Decide which tensor we score over
        actual_metric = metric
        if metric == "flow_warp" and not _HAS_CV2:
            logger.warning("[MEC] flow_warp unavailable (no OpenCV); using pixel_diff.")
            actual_metric = "pixel_diff"

        if actual_metric in ("mask_iou", "edge_stability"):
            seq = mask
        elif image is not None:
            seq = image
        else:
            seq = mask

        if seq is None or seq.shape[0] < 2:
            report = {
                "metric": actual_metric,
                "n_frames": 0 if seq is None else int(seq.shape[0]),
                "pairs": [],
                "flicker_score": 0.0,
                "note": "Sequence too short to score; need >=2 frames.",
            }
            return (image, mask, 0.0, json.dumps(report, indent=2))

        B = int(seq.shape[0])
        scores: list[float] = []
        seq_cpu = seq.detach().cpu()
        img_cpu = image.detach().cpu() if (image is not None and actual_metric == "edge_stability") else None
        if img_cpu is not None and (img_cpu.shape[0] != B or tuple(img_cpu.shape[1:3]) != tuple(seq_cpu.shape[-2:])):
            img_cpu = None                                # frames that do not match the matte: score uncompensated
        compensated_any = False
        for i in range(B - 1):
            a = seq_cpu[i]
            b = seq_cpu[i + 1]
            if actual_metric == "edge_stability":
                s, comp = _edge_stability(a.float().numpy(), b.float().numpy(),
                                          None if img_cpu is None else img_cpu[i, ..., :3].float().numpy(),
                                          None if img_cpu is None else img_cpu[i + 1, ..., :3].float().numpy(),
                                          binarize_threshold)
                compensated_any |= comp
                scores.append(s)
            elif actual_metric == "mask_iou":
                s = _mask_iou(a, b, binarize_threshold)
                # Convert IoU (1 = stable) to instability in [0, 1]
                scores.append(1.0 - s)
            elif actual_metric == "pixel_diff":
                scores.append(float((a.float() - b.float()).abs().mean().item()))
            elif actual_metric == "flow_warp":
                a_np = a.numpy()
                b_np = b.numpy()
                if a_np.ndim == 2:
                    a_np = np.stack([a_np] * 3, axis=-1)
                    b_np = np.stack([b_np] * 3, axis=-1)
                # Already H,W,3 for IMAGE input
                scores.append(_flow_warp_diff(a_np, b_np) / 255.0)

        flicker = float(np.mean(scores)) if scores else 0.0
        report = {
            "metric": actual_metric,
            "requested_metric": metric,
            "n_frames": B,
            "n_pairs": len(scores),
            "min": float(np.min(scores)) if scores else 0.0,
            "max": float(np.max(scores)) if scores else 0.0,
            "mean": flicker,
            "std": float(np.std(scores)) if scores else 0.0,
            "pairs": [{"i": i, "score": float(s)} for i, s in enumerate(scores)],
            "flicker_score": flicker,
        }
        if actual_metric == "edge_stability":
            report["motion_compensated"] = bool(compensated_any)
            report["worst_pairs"] = [int(i) for i in np.argsort(scores)[::-1][:3]] if scores else []
            report["note"] = ("Alpha change around the matte edge, 0 = the edge holds, 1 = it jumps fully. "
                              + ("Motion was followed with optical flow." if compensated_any else
                                 "No matching frames wired (or no OpenCV): moving edges count as change too."))
        return (image, mask, flicker, json.dumps(report, indent=2))


NODE_CLASS_MAPPINGS = {"TemporalConsistencyCheckerMEC": TemporalConsistencyCheckerMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TemporalConsistencyCheckerMEC": "Temporal Consistency Checker"
}
