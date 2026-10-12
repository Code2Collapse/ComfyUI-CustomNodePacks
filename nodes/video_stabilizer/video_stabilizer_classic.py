# PORTED FROM: ComfyUI-Video-Stabilizer by nomadoor and contributors (MIT) - https://github.com/nomadoor/ComfyUI-Video-Stabilizer
# nodes/video_stabilizer_classic.py at d516bf0 (2026-07-04), code unchanged. Licence: ./LICENSE. Loaded by nodes/video_stabilizer_mec.py (L7.90:
# it used to load from the gitignored third_party/ folder, so the four Video Stabilizer nodes failed on every fresh install).

"""
Feature-tracking video stabilizer node (Classic backend) for ComfyUI V3.

The implementation follows the requirements in docs/requirements/001-video-stabilizer.md:
 - CPU-only workflow using sparse feature tracking (GFTT + LK optical flow)
 - Deterministic transforms (translation/similarity/perspective) with automatic fallback
 - Framing controls (crop, crop+pad, expand) and mandatory padding mask emission
 - Metadata emission covering confidence, fallback events, and applied transforms
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    import comfy.model_management as model_management
except ImportError:  # pragma: no cover - fallback when comfy isn't available
    model_management = None
from comfy_api.latest import ComfyExtension, io
from comfy.utils import ProgressBar

from .motion_meta import applied_motion_meta_from_stabilization_warp
from .stabilizer_utils import (
    _build_stabilization_warp_meta,
    _compute_bounding_boxes,
    _compute_crop_with_keep_fov_parametric,
    _convert_masks_for_output,
    _ensure_rgb,
    _make_gray_for_estimation,
    _matrix_to_params,
    _rescale_transform_to_full,
    _working_estimation_size,
    _min_content_ratio,
    _normalize_video_input,
    _params_to_matrix,
    _parse_padding_color,
    _prepare_expand_transform,
    _reconstruct_video,
    _refine_no_padding_crop,
    _smooth_path,
    FramingMode,
    StabilizationResult,
    TransformMode,
    VideoContext,
)

JSONType = io.Custom("JSON")
logger = logging.getLogger(__name__)


def _attach_motion_meta(meta: Dict[str, Any], fps: float) -> Dict[str, Any]:
    try:
        motion_meta = applied_motion_meta_from_stabilization_warp(
            meta["stabilization_warp"],
            fps=fps,
            source="estimated_classic",
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        logger.debug("Failed to derive motion_meta from stabilization_warp.", exc_info=True)
        return meta
    meta["motion_meta"] = motion_meta
    return meta


def _estimate_motion_pair(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    requested_mode: TransformMode,
) -> Tuple[np.ndarray, TransformMode, float]:
    """Estimate geometric transform between consecutive frames with graceful fallback."""
    feature_mask = None
    features = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=21,
        mask=feature_mask,
    )
    if features is None or len(features) < 12:
        # Lack of tracks - return identity transform and let smoothing handle it.
        return np.eye(3, dtype=np.float32), "translation", 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        features,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01),
    )

    status = status.reshape(-1)
    good_prev = features[status == 1]
    good_curr = next_pts[status == 1]
    if len(good_prev) < 8:
        return np.eye(3, dtype=np.float32), "translation", 0.0

    # The order defines fallback precedence.
    mode_priority: Dict[TransformMode, List[TransformMode]] = {
        "perspective": ["perspective", "similarity", "translation"],
        "similarity": ["similarity", "translation"],
        "translation": ["translation"],
    }
    tried_modes = mode_priority[requested_mode]

    for mode in tried_modes:
        if mode == "perspective" and len(good_prev) >= 4:
            matrix, inliers = cv2.findHomography(
                good_prev,
                good_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,
                maxIters=2000,
                confidence=0.992,
            )
            if matrix is not None:
                confidence = float(inliers.sum()) / float(len(good_prev))
                if confidence < 0.15:
                    continue
                return matrix.astype(np.float32), "perspective", confidence
        elif mode == "similarity" and len(good_prev) >= 3:
            matrix, inliers = cv2.estimateAffinePartial2D(
                good_prev,
                good_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=2000,
                confidence=0.992,
            )
            if matrix is not None:
                confidence = float(inliers.sum()) / float(len(good_prev)) if inliers is not None else 0.0
                if confidence < 0.1:
                    continue
                affine3x3 = np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
                return affine3x3.astype(np.float32), "similarity", confidence
        elif mode == "translation":
            shifts = good_curr - good_prev
            if shifts.ndim > 2:
                shifts = shifts.reshape(-1, shifts.shape[-1])
            delta = np.median(shifts, axis=0).reshape(-1).astype(np.float32)
            tx = float(delta[0]) if delta.size > 0 else 0.0
            ty = float(delta[1]) if delta.size > 1 else 0.0
            matrix = np.array(
                [
                    [1.0, 0.0, tx],
                    [0.0, 1.0, ty],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            confidence = float(len(good_prev)) / float(len(features))
            return matrix, "translation", confidence

    return np.eye(3, dtype=np.float32), "translation", 0.0


def _stabilize_frames(
    context: VideoContext,
    framing_mode: FramingMode,
    transform_mode: TransformMode,
    camera_lock: bool,
    strength: float,
    smooth: float,
    keep_fov: float,
    padding_rgb: Tuple[int, int, int],
    frame_rate: float,
) -> StabilizationResult:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the video stabilizer node.")

    frames = context.frames
    total_frames = len(frames)

    fps_candidate = frame_rate
    if not isinstance(fps_candidate, (int, float)) or not np.isfinite(fps_candidate) or fps_candidate <= 0.0:
        fps_candidate = (
            context.fps
            if isinstance(context.fps, (int, float)) and np.isfinite(context.fps) and context.fps > 0.0
            else 16.0
        )
    fps_effective = float(max(1.0, fps_candidate))
    fps_requested = float(frame_rate) if isinstance(frame_rate, (int, float)) and frame_rate > 0.0 else None

    if total_frames == 0:
        meta = {
            "frames": 0,
            "note": "Empty frame sequence; nothing to stabilise.",
            "transform_mode_requested": transform_mode,
            "transform_mode_applied": "identity",
            "camera_lock": camera_lock,
            "strength": strength,
            "strength_effective": 0.0,
            "smooth": smooth,
            "fps_requested": fps_requested,
            "fps_effective": fps_effective,
            "framing": {
                "mode": framing_mode,
                "input_size": [context.width, context.height],
                "padding_color_rgb": [int(c) for c in padding_rgb],
            },
            "keep_fov_applied": False,
            "padding_color_rgb": [int(c) for c in padding_rgb],
            "stabilization_warp": _build_stabilization_warp_meta(
                source_size=(context.width, context.height),
                output_size=(context.width, context.height),
                framing_mode=framing_mode,
                applied_matrices=[],
            ),
            "estimated_motion": {"per_transition": [], "path": [], "target_path": [], "target_path_effective": []},
            "padding_fraction_mean": 0.0,
            "padding_fraction_max": 0.0,
        }
        return StabilizationResult([], [], _attach_motion_meta(meta, fps_effective))

    def _check_interrupt() -> None:
        if model_management is not None:
            model_management.throw_exception_if_processing_interrupted()

    # Progress spans the motion-estimation pass (n-1 transitions) and the final
    # warp pass (n frames) so the bar advances during the heavy pre-processing
    # instead of sitting idle until warping begins.
    estimation_steps = max(0, total_frames - 1)
    progress_total = estimation_steps + total_frames
    pbar = ProgressBar(progress_total)
    progress_done = 0
    progress_stride = 10
    progress_pending = 0

    if len(frames) == 1:
        zero_mask = np.zeros((context.height, context.width, 1), dtype=np.float32)
        frame_rgb = _ensure_rgb(frames[0])
        meta = {
            "frames": 1,
            "note": "Single-frame input; bypassed stabilization.",
            "transform_mode": transform_mode,
            "framing_mode": framing_mode,
            "stabilization_warp": _build_stabilization_warp_meta(
                source_size=(context.width, context.height),
                output_size=(context.width, context.height),
                framing_mode=framing_mode,
                applied_matrices=[np.eye(3, dtype=np.float32)],
            ),
            "fps_requested": fps_requested,
            "fps_effective": fps_effective,
        }
        pbar.update_absolute(progress_total, progress_total)
        return StabilizationResult([frame_rgb], [zero_mask], _attach_motion_meta(meta, fps_effective))

    working_size = _working_estimation_size(context.width, context.height)
    gray_frames = [_make_gray_for_estimation(frame, working_size) for frame in frames]
    base_mode = transform_mode

    delta_params: List[np.ndarray] = []
    matrices: List[np.ndarray] = []
    confidences: List[float] = []
    modes_used: List[TransformMode] = []

    active_mode = transform_mode
    for idx in range(1, len(frames)):
        matrix, used_mode, confidence = _estimate_motion_pair(
            gray_frames[idx - 1],
            gray_frames[idx],
            active_mode,
        )
        if used_mode != active_mode:
            active_mode = used_mode
        if working_size is not None:
            matrix = _rescale_transform_to_full(matrix, (context.width, context.height), working_size)
        matrices.append(matrix)
        confidences.append(confidence)
        modes_used.append(used_mode)
        delta_params.append(_matrix_to_params(matrix, base_mode))
        progress_pending += 1
        if progress_pending >= progress_stride or idx == len(frames) - 1:
            progress_done += progress_pending
            pbar.update_absolute(progress_done, progress_total)
            progress_pending = 0
        _check_interrupt()

    del gray_frames  # grayscale copies are only needed during estimation

    path = np.zeros((len(frames), delta_params[0].shape[0]), dtype=np.float64)
    for idx, delta in enumerate(delta_params, start=1):
        path[idx] = path[idx - 1] + delta

    strength = float(np.clip(strength, 0.0, 1.0))
    smooth = float(np.clip(smooth, 0.0, 1.0))

    if camera_lock:
        smooth = max(smooth, 0.85)
        target_path = np.zeros_like(path)
    else:
        smooth_path = _smooth_path(path, smooth, fps_effective)
        target_path = path + strength * (smooth_path - path)

    diffs = target_path - path
    delta_params_full: List[np.ndarray] = [diff.copy() for diff in diffs]

    keep_fov_clamped = float(np.clip(keep_fov, 0.0, 1.0))
    keep_fov_applied = framing_mode == "crop" and keep_fov_clamped > 1e-6
    stabilization_scale = 1.0
    keep_fov_status = "disabled"
    keep_fov_note: str | None = None
    keep_fov_effective_value = 1.0
    crop_origin: List[float] = [0.0, 0.0]
    crop_size: List[float] = [float(context.width), float(context.height)]

    final_matrices: List[np.ndarray]
    apply_matrices: List[np.ndarray]
    final_content_masks: List[np.ndarray] | None = None

    if framing_mode == "crop":
        if keep_fov_clamped >= 0.9999:
            zero_mask = np.zeros((context.height, context.width, 1), dtype=np.float32)
            meta = {
                "frames": len(frames),
                "note": "keep_fov~=1.0 in crop mode; returning original frames.",
                "transform_mode_requested": transform_mode,
                "transform_mode_applied": "identity",
                "camera_lock": camera_lock,
                "strength": strength,
                "strength_effective": 0.0,
                "smooth": smooth,
                "fps_requested": fps_requested,
                "fps_effective": fps_effective,
                "framing": {
                    "mode": framing_mode,
                    "input_size": [context.width, context.height],
                    "keep_fov_requested": keep_fov_clamped,
                    "keep_fov_effective": 1.0,
                    "min_content_ratio": 1.0,
                    "padding_color_rgb": [int(c) for c in padding_rgb],
                    "stabilization_scale": 0.0,
                },
                "keep_fov_applied": False,
                "stabilization_warp": _build_stabilization_warp_meta(
                    source_size=(context.width, context.height),
                    output_size=(context.width, context.height),
                    framing_mode=framing_mode,
                    applied_matrices=[np.eye(3, dtype=np.float32) for _ in frames],
                ),
                "estimated_motion": {
                    "per_transition": [],
                    "path": path.tolist(),
                    "target_path": target_path.tolist(),
                    "target_path_effective": path.tolist(),
                },
                "padding_fraction_mean": 0.0,
                "padding_fraction_max": 0.0,
            }
            pbar.update_absolute(progress_total, progress_total)
            frames_rgb = [_ensure_rgb(frame) for frame in frames]
            return StabilizationResult(frames_rgb, [zero_mask] * len(frames_rgb), _attach_motion_meta(meta, fps_effective))

        safety_margin_px = max(0.5, 0.02 * max(context.width, context.height))
        (
            final_matrices,
            apply_matrices,
            final_content_masks,
            keep_fov_effective_value,
            keep_fov_status,
            keep_fov_note,
            stabilization_scale,
            crop_origin,
            crop_size,
        ) = _compute_crop_with_keep_fov_parametric(
            _params_to_matrix,
            base_mode,
            delta_params_full,
            context.width,
            context.height,
            keep_fov_clamped,
            safety_margin_px,
            interrupt_check=_check_interrupt,
            return_masks=False,
        )
        (
            final_matrices,
            final_content_masks,
            crop_origin,
            crop_size,
            keep_fov_effective_value,
        ) = _refine_no_padding_crop(
            final_matrices,
            context.width,
            context.height,
            safety_shrink_px=1,
            interrupt_check=_check_interrupt,
        )
        # Release the full-resolution refine masks before the warp pass; the loop
        # recomputes per-frame content (bit-identical) so this list does not stay
        # resident alongside the output buffers, which is where peak memory sits.
        final_content_masks = None
        output_size = (context.width, context.height)
    else:
        apply_matrices = [_params_to_matrix(diff, base_mode) for diff in delta_params_full]
        final_matrices = apply_matrices
        output_size = (context.width, context.height)

    mins, maxs = _compute_bounding_boxes(apply_matrices, context.width, context.height)

    framing_meta: Dict[str, Any] = {
        "mode": framing_mode,
        "input_size": [context.width, context.height],
        "padding_color_rgb": [int(c) for c in padding_rgb],
        "min_content_ratio": _min_content_ratio(mins, maxs, context.width, context.height),
    }

    if framing_mode == "crop":
        framing_meta["min_content_ratio"] = _min_content_ratio(mins, maxs, context.width, context.height)
        framing_meta.update(
            {
                "keep_fov_status": keep_fov_status,
                "keep_fov_effective": keep_fov_effective_value,
                "crop_origin": crop_origin,
                "crop_size": crop_size,
                "actual_content_ratio": keep_fov_effective_value,
                "stabilization_scale": float(stabilization_scale),
            }
        )
        if keep_fov_applied:
            framing_meta["keep_fov_requested"] = keep_fov_clamped
        if keep_fov_note:
            framing_meta["keep_fov_note"] = keep_fov_note
    elif framing_mode == "crop_and_pad":
        x0 = float(np.max(mins[:, 0]))
        y0 = float(np.max(mins[:, 1]))
        x1 = float(np.min(maxs[:, 0]))
        y1 = float(np.min(maxs[:, 1]))
        intersection_w = max(1.0, x1 - x0)
        intersection_h = max(1.0, y1 - y0)
        center_x = (x0 + x1) * 0.5
        center_y = (y0 + y1) * 0.5
        frame_center_x = context.width * 0.5
        frame_center_y = context.height * 0.5
        offset_x = frame_center_x - center_x
        offset_y = frame_center_y - center_y
        translate_matrix = np.array(
            [
                [1.0, 0.0, offset_x],
                [0.0, 1.0, offset_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        final_matrices = [translate_matrix @ mat for mat in apply_matrices]
        framing_meta.update(
            {
                "safe_region_origin": [x0, y0],
                "safe_region_size": [intersection_w, intersection_h],
                "actual_content_ratio": min(intersection_w / context.width, intersection_h / context.height),
                "center_offset": [offset_x, offset_y],
            }
        )
    else:
        translate_matrix, output_size = _prepare_expand_transform(mins, maxs)
        final_matrices = [translate_matrix @ mat for mat in apply_matrices]
        framing_meta["expanded_size"] = list(output_size)

    effective_diffs = (
        np.array([_matrix_to_params(mat, base_mode) for mat in apply_matrices])
        if framing_mode == "crop"
        else np.array(delta_params_full)
    )
    stabilization_scale = float(np.clip(stabilization_scale, 0.0, 1.0))
    strength_effective = strength * stabilization_scale
    effective_target_path = path + effective_diffs

    padded_ratios: List[float] = []
    padding_detected = False

    padding_array = np.array(padding_rgb, dtype=np.float32) / 255.0
    frame_border_value: Any = float(np.mean(padding_array)) if context.channels == 1 else padding_array.tolist()

    frame_count = len(final_matrices)

    # Write directly into preallocated arrays instead of building Python lists and
    # stacking them later; on large sequences the intermediate list plus the stack
    # copy roughly doubled peak memory. Input frames are released as they are
    # consumed so resident memory does not hold the whole source sequence at once.
    out_w, out_h = output_size
    stabilized_frames = np.empty((frame_count, out_h, out_w, 3), dtype=np.float32)
    padding_masks = np.empty((frame_count, out_h, out_w, 1), dtype=np.float32)

    for idx, matrix in enumerate(final_matrices):
        warped = cv2.warpPerspective(
            frames[idx],
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=frame_border_value,
        )
        stabilized_frames[idx] = _ensure_rgb(warped.astype(np.float32))
        frames[idx] = None  # release the source frame; it is no longer needed

        if framing_mode == "crop" and final_content_masks is not None and idx < len(final_content_masks):
            content = final_content_masks[idx][..., 0]
        else:
            content = cv2.warpPerspective(
                np.ones((context.height, context.width), dtype=np.float32),
                matrix,
                output_size,
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        mask = 1.0 - (content > 0.5).astype(np.float32)
        mask[mask < 1e-3] = 0.0
        if not padding_detected and float(np.max(mask)) > 1e-3:
            padding_detected = True
        padded_ratios.append(float(mask.mean()))
        padding_masks[idx, ..., 0] = mask
        progress_pending += 1
        if progress_pending >= progress_stride or idx == frame_count - 1:
            progress_done += progress_pending
            pbar.update_absolute(progress_done, progress_total)
            progress_pending = 0
        _check_interrupt()

    framing_meta["padding_detected"] = padding_detected

    meta = {
        "frames": len(frames),
        "transform_mode_requested": transform_mode,
        "transform_mode_applied": active_mode,
        "camera_lock": camera_lock,
        "strength": strength,
        "strength_effective": strength_effective,
        "smooth": smooth,
        "fps_requested": fps_requested,
        "fps_effective": fps_effective,
        "framing": framing_meta,
        "keep_fov_applied": keep_fov_applied,
        "padding_color_rgb": [int(c) for c in padding_rgb],
        "stabilization_warp": _build_stabilization_warp_meta(
            source_size=(context.width, context.height),
            output_size=output_size,
            framing_mode=framing_mode,
            applied_matrices=final_matrices,
        ),
        "estimated_motion": {
            "per_transition": [
                {
                    "index": idx,
                    "mode": mode,
                    "confidence": confidence,
                    "matrix": matrices[idx].astype(np.float32).tolist()
                    if idx < len(matrices)
                    else np.eye(3, dtype=np.float32).tolist(),
                }
                for idx, (mode, confidence) in enumerate(zip(modes_used, confidences))
            ],
            "path": path.tolist(),
            "target_path": target_path.tolist(),
            "target_path_effective": effective_target_path.tolist(),
        },
        "padding_fraction_mean": float(np.mean(padded_ratios)),
        "padding_fraction_max": float(np.max(padded_ratios)),
    }
    return StabilizationResult(stabilized_frames, padding_masks, _attach_motion_meta(meta, fps_effective))


class VideoStabilizerClassic(io.ComfyNode):
    """Classic feature-tracking video stabilizer aligned with the V3 schema."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = io.Schema(
            node_id="video_stabilizer_classic",
            display_name="Video Stabilizer Classic",
            category="Video/Stabilization",
            description=(
                "CPU-friendly video stabilization using sparse feature tracking with configurable transforms "
                "and framing, emitting both stabilized frames and a padding mask."
            ),
        )
        schema.inputs = [
            io.Image.Input("frames", display_name="Frames"),
            io.Float.Input(
                "frame_rate",
                default=16.0,
                min=1.0,
                step=0.1,
                display_name="Input FPS",
                tooltip="Frame rate in frames per second used to scale smoothing window.",
            ),
            io.Combo.Input(
                "framing_mode",
                options=["crop", "crop_and_pad", "expand"],
                default="crop_and_pad",
                display_name="Framing Mode",
                tooltip="Choose how to handle borders produced by stabilization.",
            ),
            io.Combo.Input(
                "transform_mode",
                options=["translation", "similarity", "perspective"],
                default="similarity",
                display_name="Transform Mode",
                tooltip="Select the geometric model used to estimate camera motion.",
            ),
            io.Boolean.Input(
                "camera_lock",
                default=False,
                display_name="Camera Lock",
                tooltip="Treat the shot as tripod-like by aggressively damping motion.",
            ),
            io.Float.Input(
                "strength",
                default=0.7,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Strength",
                tooltip="Removal gain (0 keeps original motion, 1 removes it based on smoothing).",
                display_mode=io.NumberDisplay.slider,
            ),
            io.Float.Input(
                "smooth",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Smooth",
                tooltip="Temporal smoothing amount applied to the estimated motion path.",
                display_mode=io.NumberDisplay.slider,
            ),
            io.Float.Input(
                "keep_fov",
                default=0.6,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Keep FOV",
                tooltip=(
                    "[Crop only] How much of the original FOV to preserve (1.0 = no zoom, 0.0 = maximum zoom). "
                    "Ignored when framing_mode is crop_and_pad or expand."
                ),
                display_mode=io.NumberDisplay.slider,
            ),
            io.Color.Input(
                "padding_color",
                default="#7F7F7F",
                display_name="Padding Color",
                tooltip="HEX padding color applied in crop_and_pad / expand (e.g. #404040).",
            ),
        ]
        schema.outputs = [
            io.Image.Output("frames_stabilized", display_name="Stabilized Frames"),
            io.Mask.Output("padding_mask", display_name="Padding Mask"),
            JSONType.Output("meta", display_name="Motion Meta"),
        ]
        return schema

    @classmethod
    def execute(
        cls,
        frames: Any,
        frame_rate: float,
        framing_mode: FramingMode,
        transform_mode: TransformMode,
        camera_lock: bool,
        strength: float,
        smooth: float,
        keep_fov: float,
        padding_color: str,
    ) -> io.NodeOutput:
        context = _normalize_video_input(frames)
        padding_rgb = _parse_padding_color(padding_color)
        result = _stabilize_frames(
            context=context,
            framing_mode=framing_mode,
            transform_mode=transform_mode,
            camera_lock=camera_lock,
            strength=strength,
            smooth=smooth,
            keep_fov=keep_fov,
            padding_rgb=padding_rgb,
            frame_rate=frame_rate,
        )

        video_payload = _reconstruct_video(result.frames, context)
        mask_payload = _convert_masks_for_output(result.masks)

        return io.NodeOutput(video_payload, mask_payload, result.meta)


class VideoStabilizerClassicExtension(ComfyExtension):
    """Extension entrypoint used by ComfyUI to discover the node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VideoStabilizerClassic]
