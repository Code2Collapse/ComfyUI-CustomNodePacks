# PORTED FROM: ComfyUI-Video-Stabilizer by nomadoor and contributors (MIT) - https://github.com/nomadoor/ComfyUI-Video-Stabilizer
# nodes/video_stabilizer_flow.py at d516bf0 (2026-07-04), code unchanged. Licence: ./LICENSE. Loaded by nodes/video_stabilizer_mec.py (L7.90:
# it used to load from the gitignored third_party/ folder, so the four Video Stabilizer nodes failed on every fresh install).

"""
Dense optical flow-based video stabilizer node for ComfyUI V3.

Implements the requirements in docs/requirements/002-video-stabilizer-flow.md:
 - Reuses the same public schema as the classic tracker node
 - Estimates motion using dense optical flow (DIS by default, TV-L1 fallback)
 - Shares framing/metadata semantics with the classic stabilizer
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Tuple

import cv2
import numpy as np

try:
    import cv2.optflow  # type: ignore[attr-defined]
    _HAS_OPTFLOW = True
except Exception:  # pragma: no cover - optional contrib module
    _HAS_OPTFLOW = False

try:
    import comfy.model_management as model_management
except ImportError:  # pragma: no cover - fallback for static analysis
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
    _min_content_ratio,
    _rescale_transform_to_full,
    _working_estimation_size,
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

FlowBackend = Literal["DIS", "TVL1", "phase_correlate"]

JSONType = io.Custom("JSON")
logger = logging.getLogger(__name__)


def _attach_motion_meta(meta: Dict[str, Any], fps: float) -> Dict[str, Any]:
    try:
        motion_meta = applied_motion_meta_from_stabilization_warp(
            meta["stabilization_warp"],
            fps=fps,
            source="estimated_flow",
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        logger.debug("Failed to derive motion_meta from stabilization_warp.", exc_info=True)
        return meta
    meta["motion_meta"] = motion_meta
    return meta


def _create_flow_backend(backend: Literal["DIS", "TVL1"]) -> Any:
    if backend == "TVL1":
        if not _HAS_OPTFLOW:
            raise RuntimeError("OpenCV TV-L1 optical flow is unavailable (cv2.optflow missing).")
        return cv2.optflow.DualTVL1OpticalFlow_create()

    dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(2)
    dis.setPatchSize(8)
    dis.setPatchStride(4)
    dis.setUseSpatialPropagation(True)
    return dis


def _select_flow_backend() -> Tuple[FlowBackend, Any | None, str | None]:
    try:
        return "DIS", _create_flow_backend("DIS"), None
    except Exception as exc:
        dis_error = str(exc) or exc.__class__.__name__

    if _HAS_OPTFLOW:
        try:
            return "TVL1", _create_flow_backend("TVL1"), f"DIS unavailable ({dis_error}); using TV-L1."
        except Exception as exc:
            tvl1_error = str(exc) or exc.__class__.__name__
            return (
                "phase_correlate",
                None,
                f"DIS unavailable ({dis_error}); TV-L1 unavailable ({tvl1_error}); using phase correlation.",
            )

    return "phase_correlate", None, f"DIS unavailable ({dis_error}); cv2.optflow missing; using phase correlation."


def _estimate_motion_phase_correlate(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[np.ndarray, TransformMode, float, float]:
    try:
        shift, response = cv2.phaseCorrelate(prev_gray.astype(np.float32), curr_gray.astype(np.float32))
        tx, ty = float(shift[0]), float(shift[1])
        confidence = float(response)
        if not (np.isfinite(tx) and np.isfinite(ty) and np.isfinite(confidence)):
            raise ValueError("phase correlation returned non-finite values")
    except Exception:
        tx = 0.0
        ty = 0.0
        confidence = 0.0

    matrix = np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return matrix, "translation", confidence, 0.0


def _estimate_motion_flow(
    backend_obj: Any,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    requested_mode: TransformMode,
    sample_step: int = 8,
) -> Tuple[np.ndarray, TransformMode, float, float]:
    flow = backend_obj.calc(prev_gray, curr_gray, None)
    h, w = prev_gray.shape
    ys = np.arange(0, h, sample_step, dtype=np.int32)
    xs = np.arange(0, w, sample_step, dtype=np.int32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    prev_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    flow_samples = flow[grid_y, grid_x].reshape(-1, 2)
    curr_points = prev_points + flow_samples

    # Remove invalid points (NaN / inf)
    valid = np.isfinite(curr_points).all(axis=1)
    prev_valid = prev_points[valid]
    curr_valid = curr_points[valid]
    if len(prev_valid) < 12:
        return np.eye(3, dtype=np.float32), "translation", 0.0, 0.0

    mode_priority: Dict[TransformMode, List[TransformMode]] = {
        "perspective": ["perspective", "similarity", "translation"],
        "similarity": ["similarity", "translation"],
        "translation": ["translation"],
    }
    for mode in mode_priority[requested_mode]:
        if mode == "perspective" and len(prev_valid) >= 4:
            matrix, inliers = cv2.findHomography(
                prev_valid,
                curr_valid,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,
                maxIters=2000,
                confidence=0.992,
            )
            if matrix is not None and inliers is not None:
                confidence = float(inliers.sum()) / float(len(prev_valid))
                if confidence >= 0.15:
                    residuals = np.abs((prev_valid @ matrix[:2, :2].T + matrix[:2, 2]) - curr_valid)
                    return matrix.astype(np.float32), "perspective", confidence, float(residuals.mean())
        elif mode == "similarity" and len(prev_valid) >= 3:
            matrix, inliers = cv2.estimateAffinePartial2D(
                prev_valid,
                curr_valid,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=2000,
                confidence=0.992,
            )
            if matrix is not None:
                confidence = float(inliers.sum()) / float(len(prev_valid)) if inliers is not None else 0.0
                if confidence >= 0.1:
                    affine3x3 = np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
                    residuals = np.abs((prev_valid @ matrix[:, :2].T + matrix[:, 2]) - curr_valid)
                    return affine3x3.astype(np.float32), "similarity", confidence, float(residuals.mean())
        elif mode == "translation":
            shifts = curr_valid - prev_valid
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
            confidence = float(len(prev_valid)) / float(len(prev_points))
            residuals = np.abs((prev_valid + np.array([tx, ty], dtype=np.float32)) - curr_valid)
            return matrix, "translation", confidence, float(residuals.mean())

    return np.eye(3, dtype=np.float32), "translation", 0.0, 0.0


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
    flow_backend: FlowBackend = "DIS"
    flow_fallback_reason: str | None = None

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
            "flow_backend": flow_backend,
            "flow_fallback_reason": flow_fallback_reason,
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
            "keep_fov_applied": False,
            "flow_backend": flow_backend,
            "flow_fallback_reason": flow_fallback_reason,
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

    flow_backend, backend_obj, flow_fallback_reason = _select_flow_backend()

    delta_params: List[np.ndarray] = []
    matrices: List[np.ndarray] = []
    confidences: List[float] = []
    residuals: List[float] = []
    modes_used: List[TransformMode] = []

    active_mode = transform_mode
    for idx in range(1, len(frames)):
        if backend_obj is None:
            matrix, used_mode, confidence, residual = _estimate_motion_phase_correlate(
                gray_frames[idx - 1],
                gray_frames[idx],
            )
        else:
            matrix, used_mode, confidence, residual = _estimate_motion_flow(
                backend_obj,
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
        residuals.append(residual)
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
                    "padding_color_rgb": list(int(c) for c in padding_rgb),
                    "stabilization_scale": 0.0,
                },
                "keep_fov_applied": False,
                "flow_backend": flow_backend,
                "flow_fallback_reason": flow_fallback_reason,
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
        "padding_color_rgb": list(int(c) for c in padding_rgb),
        "min_content_ratio": _min_content_ratio(mins, maxs, context.width, context.height),
    }

    if framing_mode == "crop":
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
        "padding_color_rgb": list(int(c) for c in padding_rgb),
        "flow_backend": flow_backend,
        "flow_fallback_reason": flow_fallback_reason,
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
                    "residual": residual,
                    "matrix": matrices[idx].astype(np.float32).tolist()
                    if idx < len(matrices)
                    else np.eye(3, dtype=np.float32).tolist(),
                }
                for idx, (mode, confidence, residual) in enumerate(zip(modes_used, confidences, residuals))
            ],
            "path": path.tolist(),
            "target_path": target_path.tolist(),
            "target_path_effective": effective_target_path.tolist(),
        },
        "padding_fraction_mean": float(np.mean(padded_ratios)),
        "padding_fraction_max": float(np.max(padded_ratios)),
    }

    return StabilizationResult(stabilized_frames, padding_masks, _attach_motion_meta(meta, fps_effective))


class VideoStabilizerFlow(io.ComfyNode):
    """Dense optical flow-based stabilizer aligned with the V3 schema."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = io.Schema(
            node_id="video_stabilizer_flow",
            display_name="Video Stabilizer Flow",
            category="Video/Stabilization",
            description=(
                "CPU-friendly video stabilization using dense optical flow with configurable transforms "
                "and framing, emitting stabilized frames, a padding mask, and motion diagnostics."
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
                tooltip="Choose how borders produced by stabilization are handled.",
            ),
            io.Combo.Input(
                "transform_mode",
                options=["translation", "similarity", "perspective"],
                default="similarity",
                display_name="Transform Mode",
                tooltip="Select the geometric model fitted to the optical flow.",
            ),
            io.Boolean.Input(
                "camera_lock",
                default=False,
                display_name="Camera Lock",
                tooltip="Aggressively pull the motion curve toward a locked tripod-like solution.",
            ),
            io.Float.Input(
                "strength",
                default=0.7,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Strength",
                tooltip="Removal gain (0 keeps original motion, 1 removes it using the smoothed motion curve).",
                display_mode=io.NumberDisplay.slider,
            ),
            io.Float.Input(
                "smooth",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                display_name="Smooth",
                tooltip="Temporal smoothing amount applied to the motion curve before removal.",
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
