"""
MaskTrackerMEC – Unified video mask tracking / propagation / consistency.

Combines (via composition) the previously-separate video-tracking nodes:
    motion             → MotionMaskTrackerMEC      (per-frame motion detection)
    propagate          → MaskPropagateVideo        (one-frame seed → all frames)
    anchor             → TemporalAnchorMEC         (SDF-based mask interpolation)
    consistency_check  → TemporalConsistencyCheckerMEC (flicker scoring)

Single ``mode`` widget selects which engine runs. All modes share the
``mask`` + ``video`` (IMAGE) input pair.

Unified RETURN_TYPES (5 ports):
    0 MASK    masks            (per-frame mask batch)
    1 IMAGE   preview          (preview overlay or passthrough)
    2 FLOAT   score            (motion intensity / confidence / flicker score)
    3 STRING  info_json        (mode-specific diagnostic JSON)
    4 STRING  metric           (which mode/metric produced the result)
"""

from __future__ import annotations

import json

import torch

from .motion_mask_tracker import MotionMaskTrackerMEC
from .mask_propagate_video import MaskPropagateVideo
from .temporal_anchor import TemporalAnchorMEC
from .temporal_consistency_checker import TemporalConsistencyCheckerMEC


def _empty_mask_like(video: torch.Tensor | None,
                     mask: torch.Tensor | None) -> torch.Tensor:
    if video is not None and isinstance(video, torch.Tensor) and video.dim() == 4:
        return torch.zeros(video.shape[0], video.shape[1], video.shape[2],
                           dtype=torch.float32)
    if mask is not None and isinstance(mask, torch.Tensor):
        if mask.dim() == 3:
            return torch.zeros_like(mask, dtype=torch.float32)
        if mask.dim() == 2:
            return torch.zeros(1, mask.shape[0], mask.shape[1],
                               dtype=torch.float32)
    return torch.zeros(1, 64, 64, dtype=torch.float32)


def _empty_image_like(video: torch.Tensor | None) -> torch.Tensor:
    if video is not None and isinstance(video, torch.Tensor) and video.dim() == 4:
        return torch.zeros_like(video, dtype=torch.float32)
    return torch.zeros(1, 64, 64, 3, dtype=torch.float32)


class MaskTrackerMEC:
    """Unified video-mask tracker — motion / propagate / anchor / consistency.

    All modes accept the same (mask, video) pair plus mode-specific
    widgets. Heavy work runs on CPU/torch; optional cv2 paths kick in
    automatically when available.
    """

    MODES = ["motion", "propagate", "anchor", "consistency_check"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (cls.MODES, {
                    "default": "motion",
                    "tooltip": (
                        "motion: per-frame motion mask (pixel/flow/bg/hist).\n"
                        "propagate: seed mask on one frame, push to all frames.\n"
                        "anchor: SDF interpolation between anchor masks.\n"
                        "consistency_check: score flicker between consecutive frames."
                    ),
                }),

                # ── motion-mode params ──
                "camera_compensation": ("BOOLEAN", {"default": True,
                    "tooltip": "[motion] subtract global camera motion"}),
                "stabilization_method": (["homography", "affine", "translation"], {
                    "default": "homography",
                    "tooltip": "[motion] camera-motion model"}),
                "detection_mode": (["combined", "pixel_diff", "optical_flow",
                                     "background_sub", "histogram_diff"], {
                    "default": "combined",
                    "tooltip": "[motion] active method(s)"}),
                "pixel_diff_enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "[motion] enable pixel-diff method"}),
                "pixel_diff_threshold": ("FLOAT", {
                    "default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "[motion] pixel-diff threshold"}),
                "flow_enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "[motion] enable optical flow"}),
                "flow_threshold": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 50.0, "step": 0.1,
                    "tooltip": "[motion] flow magnitude threshold"}),
                "flow_algorithm": (["farneback", "phase_correlation"], {
                    "default": "farneback",
                    "tooltip": "[motion] flow algorithm"}),
                "bg_sub_enabled": ("BOOLEAN", {"default": False,
                    "tooltip": "[motion] background subtraction"}),
                "bg_model_frames": ("INT", {
                    "default": 5, "min": 1, "max": 30, "step": 1,
                    "tooltip": "[motion] frames for bg model"}),
                "bg_sub_threshold": ("FLOAT", {
                    "default": 0.1, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "[motion] bg-diff threshold"}),
                "hist_enabled": ("BOOLEAN", {"default": False,
                    "tooltip": "[motion] histogram diff"}),
                "hist_grid_size": ("INT", {
                    "default": 16, "min": 4, "max": 64, "step": 4,
                    "tooltip": "[motion] histogram grid NxN"}),
                "hist_threshold": ("FLOAT", {
                    "default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "[motion] histogram L2 threshold"}),
                "combine_method": (["union", "intersection"], {
                    "default": "union",
                    "tooltip": "[motion] method combination"}),
                "grow_pixels": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 64.0, "step": 1.0,
                    "tooltip": "[motion] dilate result"}),
                "min_region_size": ("INT", {
                    "default": 100, "min": 0, "max": 10000, "step": 10,
                    "tooltip": "[motion] noise filter"}),
                "temporal_smooth": ("BOOLEAN", {"default": True,
                    "tooltip": "[motion] gaussian time smoothing"}),

                # ── propagate-mode params ──
                "source_frame": ("INT", {
                    "default": 0, "min": 0, "max": 99999,
                    "tooltip": "[propagate] frame where mask is drawn"}),
                "propagate_mode": (MaskPropagateVideo.PROPAGATION_MODES, {
                    "default": "static",
                    "tooltip": "[propagate] propagation method"}),
                "prop_flow_threshold": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 50.0, "step": 0.5,
                    "tooltip": "[propagate] optical-flow threshold"}),
                "fade_start": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "[propagate] opacity at source frame"}),
                "fade_end": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "[propagate] opacity at last frame"}),
                "bidirectional": ("BOOLEAN", {"default": True,
                    "tooltip": "[propagate] forward+backward from source"}),

                # ── anchor-mode params ──
                "anchor_frames": ("STRING", {
                    "default": "0",
                    "tooltip": "[anchor] CSV frame indices for each anchor mask"}),
                "total_frames": ("INT", {
                    "default": 30, "min": 1, "max": 99999,
                    "tooltip": "[anchor] total output frames"}),
                "easing": (["linear", "ease_in", "ease_out", "smooth_step"], {
                    "default": "smooth_step",
                    "tooltip": "[anchor] easing curve"}),
                "sdf_iterations": ("INT", {
                    "default": 64, "min": 4, "max": 512, "step": 4,
                    "tooltip": "[anchor] SDF diffusion iterations"}),
                "flow_refinement": ("BOOLEAN", {"default": False,
                    "tooltip": "[anchor] optical-flow refine (needs video)"}),

                # ── consistency-check params ──
                "metric": (TemporalConsistencyCheckerMEC.METRICS, {
                    "default": "pixel_diff",
                    "tooltip": "[consistency_check] metric"}),
                "binarize_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "[consistency_check] mask binarize threshold"}),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Required by propagate (seed), anchor (anchor stack), "
                               "consistency_check (mask_iou). Optional for motion.",
                }),
                "video": ("IMAGE", {
                    "tooltip": "Video frame batch (B,H,W,C). Required by motion, "
                               "propagate, anchor flow_refinement, and pixel/flow consistency.",
                }),
                "sam_model": ("SAM_MODEL", {
                    "tooltip": "[propagate sam2_video mode] SAM2 model",
                }),
                "points_json": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "[propagate sam2_video mode] point prompts",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("masks", "preview", "score", "info_json", "metric")
    OUTPUT_TOOLTIPS = (
        "Per-frame mask batch (B,H,W).",
        "Preview overlay (propagate) or video passthrough.",
        "Mode-specific scalar: motion intensity / mean confidence / flicker score.",
        "Mode-specific JSON diagnostic payload.",
        "Mode/metric label string.",
    )
    FUNCTION = "execute"
    CATEGORY = "MaskEditControl/Video"
    DESCRIPTION = (
        "Unified video-mask tracker. Pick a mode and the corresponding "
        "engine runs. All modes share the (mask, video) input pair. "
        "Heavy work is chunked/vectorized; CPU fallback always available."
    )

    # ──────────────────────────────────────────────────────────────────
    def execute(self, mode,
                camera_compensation, stabilization_method, detection_mode,
                pixel_diff_enabled, pixel_diff_threshold,
                flow_enabled, flow_threshold, flow_algorithm,
                bg_sub_enabled, bg_model_frames, bg_sub_threshold,
                hist_enabled, hist_grid_size, hist_threshold,
                combine_method, grow_pixels, min_region_size,
                temporal_smooth,
                source_frame, propagate_mode, prop_flow_threshold,
                fade_start, fade_end, bidirectional,
                anchor_frames, total_frames, easing, sdf_iterations,
                flow_refinement,
                metric, binarize_threshold,
                mask=None, video=None, sam_model=None, points_json=""):

        if mode == "motion":
            return self._mode_motion(
                video, camera_compensation, stabilization_method,
                detection_mode, pixel_diff_enabled, pixel_diff_threshold,
                flow_enabled, flow_threshold, flow_algorithm,
                bg_sub_enabled, bg_model_frames, bg_sub_threshold,
                hist_enabled, hist_grid_size, hist_threshold,
                combine_method, grow_pixels, min_region_size,
                temporal_smooth,
            )
        if mode == "propagate":
            return self._mode_propagate(
                video, mask, source_frame, propagate_mode,
                prop_flow_threshold, fade_start, fade_end,
                bidirectional, sam_model, points_json,
            )
        if mode == "anchor":
            return self._mode_anchor(
                mask, anchor_frames, total_frames, easing,
                sdf_iterations, flow_refinement, video,
            )
        if mode == "consistency_check":
            return self._mode_consistency(
                metric, video, mask, binarize_threshold,
            )
        # Fallback
        return (_empty_mask_like(video, mask), _empty_image_like(video),
                0.0, "{}", mode)

    # ── motion ───────────────────────────────────────────────────────
    def _mode_motion(self, video, *args):
        if video is None:
            return (_empty_mask_like(None, None),
                    _empty_image_like(None), 0.0,
                    json.dumps({"error": "motion mode requires video"}),
                    "motion")
        impl = MotionMaskTrackerMEC()
        motion_mask, intensity, info = impl.execute(video, *args)
        return (motion_mask, video, float(intensity),
                info if isinstance(info, str) else json.dumps(info),
                "motion")

    # ── propagate ────────────────────────────────────────────────────
    def _mode_propagate(self, video, mask, source_frame, mode,
                        flow_threshold, fade_start, fade_end,
                        bidirectional, sam_model, points_json):
        if video is None or mask is None:
            return (_empty_mask_like(video, mask),
                    _empty_image_like(video), 0.0,
                    json.dumps({"error": "propagate requires video + mask"}),
                    "propagate")
        impl = MaskPropagateVideo()
        masks, preview = impl.propagate(
            images=video, mask=mask, source_frame=int(source_frame),
            mode=str(mode), flow_threshold=float(flow_threshold),
            fade_start=float(fade_start), fade_end=float(fade_end),
            bidirectional=bool(bidirectional),
            sam_model=sam_model, points_json=str(points_json),
        )
        info = json.dumps({"mode": mode, "source_frame": int(source_frame)})
        return (masks, preview, 1.0, info, f"propagate:{mode}")

    # ── anchor ───────────────────────────────────────────────────────
    def _mode_anchor(self, mask, anchor_frames, total_frames, easing,
                     sdf_iterations, flow_refinement, video):
        if mask is None:
            return (_empty_mask_like(video, None),
                    _empty_image_like(video), 0.0,
                    json.dumps({"error": "anchor mode requires mask stack"}),
                    "anchor")
        impl = TemporalAnchorMEC()
        full_masks, confidence_list, info = impl.execute(
            anchor_masks=mask, anchor_frames=str(anchor_frames),
            total_frames=int(total_frames), easing=str(easing),
            sdf_iterations=int(sdf_iterations),
            flow_refinement=bool(flow_refinement), images=video,
        )
        try:
            conf = (sum(float(c) for c in confidence_list)
                    / max(1, len(confidence_list)))
        except (TypeError, ValueError):
            conf = 0.0
        preview = video if video is not None else _empty_image_like(None)
        return (full_masks, preview, float(conf),
                info if isinstance(info, str) else json.dumps(info),
                "anchor")

    # ── consistency_check ───────────────────────────────────────────
    def _mode_consistency(self, metric, video, mask, binarize_threshold):
        impl = TemporalConsistencyCheckerMEC()
        try:
            img_out, mask_out, flicker, report = impl.check(
                metric=str(metric), image=video, mask=mask,
                binarize_threshold=float(binarize_threshold),
            )
        except ValueError as exc:
            return (_empty_mask_like(video, mask),
                    _empty_image_like(video), 0.0,
                    json.dumps({"error": str(exc)}),
                    f"consistency_check:{metric}")
        return (mask_out if mask_out is not None
                else _empty_mask_like(video, mask),
                img_out if img_out is not None
                else _empty_image_like(video),
                float(flicker),
                report if isinstance(report, str) else json.dumps(report),
                f"consistency_check:{metric}")


NODE_CLASS_MAPPINGS = {"MaskTrackerMEC": MaskTrackerMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskTrackerMEC": "Mask Tracker — Motion/Propagate/Anchor/Consistency (MEC)",
}
