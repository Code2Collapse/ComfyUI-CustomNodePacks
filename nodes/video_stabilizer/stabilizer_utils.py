# PORTED FROM: ComfyUI-Video-Stabilizer by nomadoor and contributors (MIT) - https://github.com/nomadoor/ComfyUI-Video-Stabilizer
# nodes/stabilizer_utils.py at d516bf0 (2026-07-04), code unchanged. Licence: ./LICENSE. Loaded by nodes/video_stabilizer_mec.py (L7.90:
# it used to load from the gitignored third_party/ folder, so the four Video Stabilizer nodes failed on every fresh install).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch optional for some utilities
    torch = None

FramingMode = Literal["crop", "crop_and_pad", "expand"]
TransformMode = Literal["translation", "similarity", "perspective"]

ParamsToMatrix = Callable[[np.ndarray, TransformMode], np.ndarray]

__all__ = [
    "FrameAdapter",
    "FramingMode",
    "InverseStabilizationResult",
    "StabilizationResult",
    "TransformMode",
    "VideoContext",
    "_apply_inverse_stabilization",
    "_build_stabilization_warp_meta",
    "_ensure_rgb",
    "_make_gray",
    "_make_gray_for_estimation",
    "_working_estimation_size",
    "_rescale_transform_to_full",
    "_matrix_to_params",
    "_normalize_video_input",
    "_params_to_matrix",
    "_prepare_expand_transform",
    "_reconstruct_video",
    "_smooth_path",
    "_largest_axis_aligned_rectangle",
    "_largest_aspect_ratio_rectangle",
    "_compute_crop_with_keep_fov_parametric",
    "_refine_no_padding_crop",
    "_parse_padding_color",
    "_compute_bounding_boxes",
    "_min_content_ratio",
    "_convert_masks_for_output",
    "_resolve_fps",
]


@dataclass
class FrameAdapter:
    """Keeps enough context to project numpy frames back to the original container type."""

    dtype: np.dtype
    channel_first: bool
    value_range: Literal["0_1", "0_255"]
    origin: Literal["numpy", "torch"]
    squeeze_last_dim: bool


@dataclass
class VideoContext:
    frames: List[np.ndarray]
    adapter: FrameAdapter
    width: int
    height: int
    channels: int
    fps: float | None
    template_kind: Literal["dict", "sequence"]
    template_meta: Dict[str, Any]


def _resolve_fps(context: VideoContext, frame_rate: float, default: float = 16.0) -> float:
    for candidate in (context.fps, frame_rate, default):
        if isinstance(candidate, (int, float)) and np.isfinite(candidate) and candidate > 0.0:
            return float(candidate)
    return float(default)


@dataclass
class StabilizationResult:
    frames: List[np.ndarray]
    masks: List[np.ndarray]
    meta: Dict[str, Any]


@dataclass
class InverseStabilizationResult:
    frames: List[np.ndarray]
    masks: List[np.ndarray]
    meta: Dict[str, Any]


def _to_numpy_frame(frame: Any) -> Tuple[np.ndarray, FrameAdapter]:
    """Convert an incoming frame to float32 numpy in RGB order with range 0..1."""
    origin: Literal["numpy", "torch"] = "numpy"
    if torch is not None and isinstance(frame, torch.Tensor):
        origin = "torch"
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)

    channel_first = False
    squeeze_last_dim = False
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
        channel_first = True
        arr = np.moveaxis(arr, 0, -1)
    elif arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
        squeeze_last_dim = True
    elif arr.ndim == 3 and arr.shape[2] == 1:
        squeeze_last_dim = True

    dtype = arr.dtype
    value_range: Literal["0_1", "0_255"]

    if dtype == np.uint8:
        # uint8 always needs a fresh float32 buffer; scaling it in place is safe.
        arr = arr.astype(np.float32)
        arr /= 255.0
        value_range = "0_255"
    elif bool(arr.size) and float(arr.max()) > 1.5:
        # Float data in 0..255 range: own the buffer before scaling in place.
        arr = arr.astype(np.float32)
        arr /= 255.0
        value_range = "0_255"
    else:
        value_range = "0_1"
        if dtype != np.float32 or not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        # Otherwise keep the already-float32, contiguous, 0..1 array as a view to
        # avoid duplicating the whole sequence; it is only read downstream
        # (warp/grayscale), never mutated in place.

    adapter = FrameAdapter(
        dtype=dtype,
        channel_first=channel_first,
        value_range=value_range,
        origin=origin,
        squeeze_last_dim=squeeze_last_dim,
    )
    return arr, adapter


def _normalize_video_input(value: Any) -> VideoContext:
    """Extract list of numpy frames plus metadata required for reconstruction."""
    if isinstance(value, dict):
        candidates = ("frames", "images", "video")
        frames_seq = None
        for key in candidates:
            if key in value:
                frames_seq = value[key]
                break
        if frames_seq is None:
            raise ValueError("Video input dictionary must contain 'frames'.")
        template_kind: Literal["dict", "sequence"] = "dict"
        template_meta = {k: v for k, v in value.items() if k not in candidates}
        fps = template_meta.get("fps")
    else:
        frames_seq = value
        template_kind = "sequence"
        template_meta = {}
        fps = None

    frames_np: List[np.ndarray] = []
    adapter_ref: FrameAdapter | None = None
    width = height = channels = 0

    for frame in frames_seq:
        arr, adapter = _to_numpy_frame(frame)
        if adapter_ref is None:
            adapter_ref = adapter
        else:
            if adapter.channel_first != adapter_ref.channel_first or adapter.origin != adapter_ref.origin:
                raise ValueError("Mixed tensor layouts within the same video sequence are not supported.")
        arr = _ensure_rgb(arr)
        frames_np.append(arr)

    if not frames_np:
        raise ValueError("The input video sequence is empty.")

    height, width, channels = frames_np[0].shape
    return VideoContext(
        frames=frames_np,
        adapter=adapter_ref,  # type: ignore[arg-type]
        width=width,
        height=height,
        channels=channels,
        fps=fps,
        template_kind=template_kind,
        template_meta=template_meta,
    )


def _reconstruct_video(frames: Iterable[np.ndarray], context: VideoContext) -> Any:
    """Pack frames into a BHWC torch tensor (or numpy) following Comfy conventions."""
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        # Already a preallocated BHWC buffer; avoid an extra full-sequence copy.
        stacked = frames if frames.shape[0] else np.zeros((1, context.height, context.width, 3), dtype=np.float32)
    else:
        frame_list = list(frames)
        if not frame_list:
            stacked = np.zeros((1, context.height, context.width, 3), dtype=np.float32)
        else:
            stacked = np.stack(frame_list, axis=0)
    stacked = np.ascontiguousarray(stacked, dtype=np.float32)
    if torch is not None:
        tensor = torch.from_numpy(stacked)
    else:
        tensor = stacked

    if context.template_kind == "dict":
        payload = dict(context.template_meta)
        payload["frames"] = tensor
        return payload
    return tensor


def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
    """Ensure output frame has three channels in RGB order."""
    if frame.ndim == 2:
        frame = frame[..., np.newaxis]
    channels = frame.shape[2]
    if channels == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif channels > 3:
        frame = frame[..., :3]
    return frame


def _make_gray(frame: np.ndarray) -> np.ndarray:
    """Convert frame to uint8 grayscale for tracking."""
    if frame.shape[2] == 1:
        gray = frame[..., 0]
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.clip(gray * 255.0, 0, 255).astype(np.uint8)


DEFAULT_ESTIMATION_MAX_SIDE = 960


def _working_estimation_size(
    width: int,
    height: int,
    max_side: int = DEFAULT_ESTIMATION_MAX_SIDE,
) -> Tuple[int, int] | None:
    """Pick a reduced size for motion estimation, or None to use full resolution.

    Camera-motion estimation does not need full detail; capping the longest side
    keeps the tracking pass fast and light on large footage while leaving the
    full-resolution warp untouched. Returns None when the frame is already small
    enough so smaller inputs are processed unchanged.
    """
    longest = max(int(width), int(height))
    if longest <= max_side:
        return None
    scale = max_side / float(longest)
    small_w = max(1, int(round(width * scale)))
    small_h = max(1, int(round(height * scale)))
    if small_w >= width or small_h >= height:
        return None
    return small_w, small_h


def _make_gray_for_estimation(frame: np.ndarray, working_size: Tuple[int, int] | None) -> np.ndarray:
    """Grayscale a frame for tracking, optionally downscaled to ``working_size``."""
    gray = _make_gray(frame)
    if working_size is None:
        return gray
    return cv2.resize(gray, working_size, interpolation=cv2.INTER_AREA)


def _rescale_transform_to_full(
    matrix: np.ndarray,
    source_size: Tuple[int, int],
    working_size: Tuple[int, int],
) -> np.ndarray:
    """Map a transform estimated at ``working_size`` back to full-resolution coords.

    A coordinate in the full frame maps to the working frame via S = diag(sx, sy);
    the full-resolution transform is therefore S^-1 @ M_working @ S, which scales
    translation while leaving rotation/scale invariant.
    """
    src_w, src_h = source_size
    small_w, small_h = working_size
    sx = small_w / float(src_w)
    sy = small_h / float(src_h)
    scale = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    inv_scale = np.array([[1.0 / sx, 0.0, 0.0], [0.0, 1.0 / sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    full = inv_scale @ matrix.astype(np.float64) @ scale
    return full.astype(np.float32)


def _matrix_to_params(matrix: np.ndarray, base_mode: TransformMode) -> np.ndarray:
    """Project a transform matrix into the parameter space used for smoothing."""
    if base_mode == "translation":
        return np.array([matrix[0, 2], matrix[1, 2]], dtype=np.float64)
    if base_mode == "similarity":
        tx, ty = matrix[0, 2], matrix[1, 2]
        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]
        scale = math.sqrt(max(a * a + c * c, 1e-10))
        theta = math.atan2(c, a)
        log_scale = math.log(scale)
        return np.array([tx, ty, theta, log_scale], dtype=np.float64)
    return np.array(
        [
            matrix[0, 0] - 1.0,
            matrix[0, 1],
            matrix[0, 2],
            matrix[1, 0],
            matrix[1, 1] - 1.0,
            matrix[1, 2],
            matrix[2, 0],
            matrix[2, 1],
        ],
        dtype=np.float64,
    )


def _params_to_matrix(params: np.ndarray, base_mode: TransformMode) -> np.ndarray:
    """Convert parameter offset back to a homogeneous transform matrix."""
    if base_mode == "translation":
        return np.array(
            [
                [1.0, 0.0, params[0]],
                [0.0, 1.0, params[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    if base_mode == "similarity":
        tx, ty, theta, log_scale = params
        scale = math.exp(log_scale)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return np.array(
            [
                [scale * cos_t, -scale * sin_t, tx],
                [scale * sin_t, scale * cos_t, ty],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return np.array(
        [
            [params[0] + 1.0, params[1], params[2]],
            [params[3], params[4] + 1.0, params[5]],
            [params[6], params[7], 1.0],
        ],
        dtype=np.float32,
    )


def _smooth_path(path: np.ndarray, smooth: float, fps: float) -> np.ndarray:
    """Apply symmetric moving-average smoothing to the path."""
    smooth = float(np.clip(smooth, 0.0, 1.0))
    if smooth <= 0.0 or len(path) <= 2:
        return path.copy()

    fps = float(max(1.0, fps))
    min_seconds = 3.0 / 16.0
    max_seconds = 13.0 / 16.0
    window_seconds = min_seconds + smooth * (max_seconds - min_seconds)
    window = int(round(window_seconds * fps))
    window = max(3, window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)

    smoothed = np.zeros_like(path)
    for dim in range(path.shape[1]):
        series = path[:, dim]
        padded = np.pad(series, (pad, pad), mode="edge")
        smoothed[:, dim] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _prepare_expand_transform(
    mins: np.ndarray,
    maxs: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Compute translation to keep all frames within the expanded canvas."""
    x_min = float(np.min(mins[:, 0]))
    y_min = float(np.min(mins[:, 1]))
    x_max = float(np.max(maxs[:, 0]))
    y_max = float(np.max(maxs[:, 1]))

    out_w = int(math.ceil(x_max - x_min))
    out_h = int(math.ceil(y_max - y_min))
    translate = np.array(
        [
            [1.0, 0.0, -x_min],
            [0.0, 1.0, -y_min],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return translate, (max(out_w, 1), max(out_h, 1))


def _largest_axis_aligned_rectangle(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute the largest axis-aligned rectangle fully contained in ``binary_mask``.

    Args:
        binary_mask: 2D uint8/float mask with values in {0, 1}.

    Returns:
        (x0, y0, w, h) describing the rectangle of ones with the maximum area.
    """
    height, width = binary_mask.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_rect = (0, 0, width, height)

    for y in range(height):
        row = binary_mask[y]
        heights = (heights + 1) * (row > 0)
        stack: List[int] = []
        x = 0
        while x <= width:
            curr_h = heights[x] if x < width else 0
            if not stack or curr_h >= heights[stack[-1]]:
                stack.append(x)
                x += 1
            else:
                top = stack.pop()
                h = heights[top]
                left = stack[-1] + 1 if stack else 0
                w = x - left
                area = h * w
                if area > best_area:
                    best_area = area
                    x0 = left
                    y0 = y - h + 1
                    best_rect = (x0, y0, w, h)
    return best_rect


def _largest_aspect_ratio_rectangle(
    binary_mask: np.ndarray,
    target_width: int,
    target_height: int,
) -> Tuple[float, float, float, float] | None:
    """Find the largest all-valid crop that keeps the target aspect ratio.

    The crop dimensions remain floats so the caller can derive a single scale
    factor exactly. Validity is checked conservatively against the enclosing
    integer pixel rectangle.
    """
    if target_width <= 0 or target_height <= 0:
        return None

    height, width = binary_mask.shape
    aspect_ratio = float(target_width) / float(target_height)
    mask = (binary_mask > 0).astype(np.uint8)
    integral = cv2.integral(mask, sdepth=cv2.CV_64F)

    def find_fit(crop_height: int) -> Tuple[int, int] | None:
        crop_width = int(math.ceil(aspect_ratio * crop_height))
        if crop_height <= 0 or crop_height > height or crop_width > width:
            return None
        sums = (
            integral[crop_height:, crop_width:]
            - integral[:-crop_height, crop_width:]
            - integral[crop_height:, :-crop_width]
            + integral[:-crop_height, :-crop_width]
        )
        matches = sums == crop_width * crop_height
        if not np.any(matches):
            return None

        # Prefer the centered crop when it is valid so refinement does not
        # introduce a needless framing shift.
        y0 = int(np.clip(round((height - crop_height) * 0.5), 0, matches.shape[0] - 1))
        x0 = int(np.clip(round((width - crop_width) * 0.5), 0, matches.shape[1] - 1))
        if not matches[y0, x0]:
            y0, x0 = np.unravel_index(int(np.argmax(matches)), matches.shape)
        return int(x0), int(y0)

    low, high = 1, min(height, int(math.floor(width / aspect_ratio)))
    best: Tuple[int, int, int] | None = None
    while low <= high:
        crop_height = (low + high) // 2
        location = find_fit(crop_height)
        if location is None:
            high = crop_height - 1
        else:
            best = (location[0], location[1], crop_height)
            low = crop_height + 1

    if best is None:
        return None

    x0, y0, crop_height = best
    return float(x0), float(y0), aspect_ratio * crop_height, float(crop_height)


def _scale_deltas_parametric(
    params_to_matrix: ParamsToMatrix,
    base_mode: TransformMode,
    deltas: Sequence[np.ndarray],
    scale: float,
) -> List[np.ndarray]:
    """Convert per-frame parameter deltas into matrices after scaling by ``scale``."""
    scale = float(np.clip(scale, 0.0, 1.0))
    return [params_to_matrix(delta * scale, base_mode) for delta in deltas]


def _compute_crop_with_keep_fov_parametric(
    params_to_matrix: ParamsToMatrix,
    base_mode: TransformMode,
    delta_params: Sequence[np.ndarray],
    width: int,
    height: int,
    keep_fov_target: float,
    safety_margin_px: float,
    max_iterations: int = 18,
    interrupt_check: Callable[[], None] | None = None,
    return_masks: bool = True,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    float,
    str,
    str | None,
    float,
    List[float],
    List[float],
]:
    """
    Solve for a stabilisation scale that satisfies the requested ``keep_fov``.

    This performs a binary search over a scalar ``s`` applied to the motion removal
    strength. For each candidate the intersection of warped bounds is measured, a
    safety margin is applied, and the resulting crop is evaluated using binary masks.
    """
    keep_fov_clamped = float(np.clip(keep_fov_target, 0.0, 1.0))
    target_ratio = keep_fov_clamped
    eps = 1e-4

    def evaluate_bbox_only(scale: float) -> Tuple[float, Dict[str, object]]:
        mats = _scale_deltas_parametric(params_to_matrix, base_mode, delta_params, scale)
        mins, maxs = _compute_bounding_boxes(mats, width, height)
        x0 = float(np.max(mins[:, 0]))
        y0 = float(np.max(mins[:, 1]))
        x1 = float(np.min(maxs[:, 0]))
        y1 = float(np.min(maxs[:, 1]))

        safe_w = max(0.0, x1 - x0)
        safe_h = max(0.0, y1 - y0)
        margin = min(safety_margin_px, safe_w * 0.25, safe_h * 0.25)
        safe_x0 = x0 + margin
        safe_y0 = y0 + margin
        safe_w = max(0.0, safe_w - 2.0 * margin)
        safe_h = max(0.0, safe_h - 2.0 * margin)

        if safe_w <= 0.0 or safe_h <= 0.0:
            return 0.0, {
                "scale": scale,
                "pre_crop": mats,
                "final": mats,
                "crop_origin": [0.0, 0.0],
                "crop_size": [float(width), float(height)],
                "has_overlap": False,
            }

        crop_ratio = min(1.0, safe_w / width, safe_h / height)
        crop_w = width * crop_ratio
        crop_h = height * crop_ratio
        crop_x0 = safe_x0 + (safe_w - crop_w) * 0.5
        crop_y0 = safe_y0 + (safe_h - crop_h) * 0.5

        # A single scale preserves the source aspect ratio while still filling
        # the fixed-size crop output without padding.
        crop_scale = width / crop_w
        crop_matrix = np.array(
            [
                [crop_scale, 0.0, -crop_scale * crop_x0],
                [0.0, crop_scale, -crop_scale * crop_y0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        final_mats = [crop_matrix @ mat for mat in mats]
        return crop_ratio, {
            "scale": scale,
            "pre_crop": mats,
            "final": final_mats,
            "crop_origin": [crop_x0, crop_y0],
            "crop_size": [crop_w, crop_h],
            "has_overlap": True,
        }

    def finalize_with_masks(candidate: Dict[str, object]) -> Dict[str, object]:
        ones = np.ones((height, width), dtype=np.float32)
        kernel = np.ones((3, 3), np.uint8)
        masks: List[np.ndarray] = []
        min_ratio = 1.0
        best_origin: List[float] = list(candidate.get("crop_origin", [0.0, 0.0]))  # type: ignore[index]
        best_size: List[float] = list(candidate.get("crop_size", [float(width), float(height)]))  # type: ignore[index]

        for matrix in candidate["final"]:
            if interrupt_check is not None:
                interrupt_check()
            content = cv2.warpPerspective(
                ones,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            content = (content > 0.5).astype(np.float32)
            content = cv2.dilate(content, kernel, iterations=1)
            content = cv2.erode(content, kernel, iterations=1)
            # The caller in the node path discards these masks (the refine pass
            # rebuilds them), so skip retaining the full-resolution list there and
            # only keep the min content ratio that drives keep_fov status.
            if return_masks:
                masks.append(content[..., None])

            coords = np.argwhere(content > 0.5)
            if coords.size == 0:
                ratio = 0.0
                origin = [0.0, 0.0]
                size = [0.0, 0.0]
            else:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                origin = [float(x_min), float(y_min)]
                size = [float(max(1, x_max - x_min + 1)), float(max(1, y_max - y_min + 1))]
                ratio = min(size[0] / width, size[1] / height)

            if ratio < min_ratio:
                min_ratio = ratio

        candidate = dict(candidate)
        candidate.update(
            {
                "content_masks": masks,
                "ratio_final": float(min_ratio),
                "crop_origin": best_origin,
                "crop_size": best_size,
            }
        )
        return candidate

    ratio_full, raw_full = evaluate_bbox_only(1.0)
    if keep_fov_clamped <= eps:
        if bool(raw_full["has_overlap"]):
            raw = raw_full
            stabilization_scale = 1.0
            note = None
        else:
            _, raw = evaluate_bbox_only(0.0)
            stabilization_scale = 0.0
            note = "No common crop region at full stabilization; stabilization was disabled."
        candidate = finalize_with_masks(raw)
        return (
            candidate["final"],
            raw["pre_crop"],
            candidate["content_masks"],
            candidate["ratio_final"],
            "disabled",
            note,
            stabilization_scale,
            candidate["crop_origin"],
            candidate["crop_size"],
        )

    if ratio_full >= target_ratio - eps:
        candidate = finalize_with_masks(raw_full)
        return (
            candidate["final"],
            raw_full["pre_crop"],
            candidate["content_masks"],
            candidate["ratio_final"],
            "met",
            None,
            1.0,
            candidate["crop_origin"],
            candidate["crop_size"],
        )

    low, high = 0.0, 1.0
    best_candidate: Dict[str, object] | None = None
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        ratio_mid, raw_mid = evaluate_bbox_only(mid)
        if ratio_mid >= target_ratio - eps:
            best_candidate = raw_mid
            low = mid
        else:
            high = mid

    if best_candidate is None:
        _, raw_zero = evaluate_bbox_only(0.0)
        candidate_zero = finalize_with_masks(raw_zero)
        note = (
            None
            if keep_fov_clamped <= eps
            else f"keep_fov target {keep_fov_clamped:.3f} could not be satisfied even with zero stabilisation."
        )
        return (
            candidate_zero["final"],
            raw_zero["pre_crop"],
            candidate_zero["content_masks"],
            candidate_zero["ratio_final"],
            "failed" if keep_fov_clamped > eps else "disabled",
            note,
            0.0,
            candidate_zero["crop_origin"],
            candidate_zero["crop_size"],
        )

    candidate_final = finalize_with_masks(best_candidate)
    status = "met" if candidate_final["ratio_final"] >= target_ratio - eps else "clamped"
    note = None
    scale_best = float(best_candidate["scale"])
    if status == "clamped" and keep_fov_clamped > eps:
        note = (
            f"keep_fov target {keep_fov_clamped:.3f} reduced to {candidate_final['ratio_final']:.3f} "
            f"at stabilisation scale {scale_best:.3f}."
        )

    return (
        candidate_final["final"],
        best_candidate["pre_crop"],
        candidate_final["content_masks"],
        candidate_final["ratio_final"],
        status,
        note,
        scale_best,
        candidate_final["crop_origin"],
        candidate_final["crop_size"],
    )


def _refine_no_padding_crop(
    final_matrices: Sequence[np.ndarray],
    width: int,
    height: int,
    safety_shrink_px: int = 1,
    interrupt_check: Callable[[], None] | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], float]:
    """
    Post-process crop matrices to guarantee padding-free output in crop mode.
    """
    ones = np.ones((height, width), dtype=np.float32)

    def _warp_binary(matrix: np.ndarray) -> np.ndarray:
        content = cv2.warpPerspective(
            ones,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return (content > 0.5).astype(np.uint8)

    # AND the per-frame content masks into ``common`` incrementally instead of
    # retaining the whole full-resolution mask list; the list is only needed in
    # the rare degenerate fallbacks below, where it is cheap to recompute.
    common: np.ndarray | None = None
    for matrix in final_matrices:
        if interrupt_check is not None:
            interrupt_check()
        mask = _warp_binary(matrix)
        common = mask.copy() if common is None else (common & mask)

    if safety_shrink_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + 2 * safety_shrink_px, 1 + 2 * safety_shrink_px))
        common = cv2.erode(common, kernel, iterations=1)

    if common.max() == 0:
        return (
            list(final_matrices),
            [_warp_binary(matrix)[..., None].astype(np.float32) for matrix in final_matrices],
            [0.0, 0.0],
            [float(width), float(height)],
            0.0,
        )

    aspect_crop = _largest_aspect_ratio_rectangle(common, width, height)
    if aspect_crop is None:
        return (
            list(final_matrices),
            [_warp_binary(matrix)[..., None].astype(np.float32) for matrix in final_matrices],
            [0.0, 0.0],
            [float(width), float(height)],
            0.0,
        )

    x0, y0, crop_w, crop_h = aspect_crop
    crop_scale = width / crop_w
    crop_matrix = np.array(
        [
            [crop_scale, 0.0, -crop_scale * x0],
            [0.0, crop_scale, -crop_scale * y0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    refined_mats = [crop_matrix @ matrix for matrix in final_matrices]
    refined_masks: List[np.ndarray] = []
    origin_best = [x0, y0]
    size_best = [crop_w, crop_h]

    for matrix in refined_mats:
        if interrupt_check is not None:
            interrupt_check()
        content = cv2.warpPerspective(
            ones,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        mask = (content > 0.5).astype(np.float32)
        refined_masks.append(mask[..., None])
        # Keeping origin_best/size_best from pre-scale solve; refined masks
        # are only used for padding detection downstream.

    return refined_mats, refined_masks, origin_best, size_best, 1.0


DEFAULT_PADDING_RGB = (127, 127, 127)


def _parse_padding_color(value: str | int) -> Tuple[int, int, int]:
    """Parse a #RRGGBB string or 0xRRGGBB integer into an RGB tuple."""
    if isinstance(value, str):
        stripped = value.strip()
        if "," in stripped or "/" in stripped:
            try:
                parts = stripped.replace("/", ",").replace(" ", ",").split(",")
                ints = [int(part) for part in parts if part != ""]
                if len(ints) == 1:
                    ints = [ints[0]] * 3
                if len(ints) != 3:
                    return DEFAULT_PADDING_RGB
                return tuple(int(np.clip(channel, 0, 255)) for channel in ints)
            except (TypeError, ValueError):
                return DEFAULT_PADDING_RGB
        hex_value = stripped.removeprefix("#")
        if len(hex_value) == 3:
            hex_value = "".join(channel * 2 for channel in hex_value)
        if len(hex_value) != 6:
            return DEFAULT_PADDING_RGB
        try:
            rgb_int = int(hex_value, 16)
        except (TypeError, ValueError):
            return DEFAULT_PADDING_RGB
    else:
        try:
            rgb_int = int(value)
        except (TypeError, ValueError):
            return DEFAULT_PADDING_RGB
    rgb_int = int(np.clip(rgb_int, 0, 0xFFFFFF))
    return (rgb_int >> 16) & 0xFF, (rgb_int >> 8) & 0xFF, rgb_int & 0xFF


def _build_stabilization_warp_meta(
    *,
    source_size: Tuple[int, int],
    output_size: Tuple[int, int],
    framing_mode: FramingMode,
    applied_matrices: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """Describe the exact per-frame matrices applied during stabilization."""
    return {
        "source_size": [int(source_size[0]), int(source_size[1])],
        "output_size": [int(output_size[0]), int(output_size[1])],
        "framing_mode": framing_mode,
        "matrix_convention": "source_to_stabilized",
        "per_frame": [
            {
                "index": int(idx),
                "applied_matrix": np.asarray(matrix, dtype=np.float32).tolist(),
            }
            for idx, matrix in enumerate(applied_matrices)
        ],
    }


def _read_size_pair(meta: Dict[str, Any], key: str) -> Tuple[int, int]:
    value = meta.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"stabilization_warp.{key} must be [width, height].")
    try:
        width = int(value[0])
        height = int(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stabilization_warp.{key} must contain integer width/height.") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"stabilization_warp.{key} must contain positive width/height.")
    return width, height


def _read_applied_matrix(entry: Any, expected_index: int) -> np.ndarray:
    if not isinstance(entry, dict):
        raise ValueError(f"stabilization_warp.per_frame[{expected_index}] must be an object.")
    if entry.get("index") != expected_index:
        raise ValueError(
            f"stabilization_warp.per_frame[{expected_index}].index must be {expected_index}, "
            f"got {entry.get('index')!r}."
        )
    if "applied_matrix" not in entry:
        raise ValueError(f"stabilization_warp.per_frame[{expected_index}].applied_matrix is missing.")
    matrix = np.asarray(entry["applied_matrix"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"stabilization_warp.per_frame[{expected_index}].applied_matrix must be 3x3.")
    return matrix


def _apply_inverse_stabilization(
    context: VideoContext,
    meta: Dict[str, Any],
    padding_rgb: Tuple[int, int, int],
) -> InverseStabilizationResult:
    """Apply inverse stabilization matrices and restore frames to the original canvas size."""
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dictionary containing stabilization_warp.")
    warp_meta = meta.get("stabilization_warp")
    if not isinstance(warp_meta, dict):
        raise ValueError("meta.stabilization_warp is required for inverse stabilization.")
    if warp_meta.get("matrix_convention") != "source_to_stabilized":
        raise ValueError(
            "stabilization_warp.matrix_convention must be 'source_to_stabilized' "
            f"for inverse stabilization, got {warp_meta.get('matrix_convention')!r}."
        )

    source_size = _read_size_pair(warp_meta, "source_size")
    output_size = _read_size_pair(warp_meta, "output_size")
    if (context.width, context.height) != output_size:
        raise ValueError(
            "Input frames must match stabilization_warp.output_size "
            f"{output_size}, got {(context.width, context.height)}."
        )

    per_frame = warp_meta.get("per_frame")
    if not isinstance(per_frame, list):
        raise ValueError("stabilization_warp.per_frame must be a list.")
    if len(per_frame) != len(context.frames):
        raise ValueError(
            "Frame count mismatch: "
            f"got {len(context.frames)} frame(s), metadata has {len(per_frame)} matrix entry/entries."
        )

    restored_frames: List[np.ndarray] = []
    padding_masks: List[np.ndarray] = []
    padding_array = np.array(padding_rgb, dtype=np.float32) / 255.0
    frame_border_value: Any = float(np.mean(padding_array)) if context.channels == 1 else padding_array.tolist()

    for idx, (frame, entry) in enumerate(zip(context.frames, per_frame, strict=True)):
        matrix = _read_applied_matrix(entry, idx)
        try:
            inverse_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"stabilization_warp.per_frame[{idx}].applied_matrix is not invertible.") from exc

        restored = cv2.warpPerspective(
            frame,
            inverse_matrix.astype(np.float32),
            source_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=frame_border_value,
        )
        restored_frames.append(_ensure_rgb(restored.astype(np.float32)))

        content = cv2.warpPerspective(
            np.ones((context.height, context.width), dtype=np.float32),
            inverse_matrix.astype(np.float32),
            source_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        mask = (1.0 - (content > 0.5).astype(np.float32))[..., np.newaxis]
        mask[mask < 1e-3] = 0.0
        padding_masks.append(mask)

    result_meta = dict(meta)
    result_meta["inverse_stabilization"] = {
        "source_size": [int(source_size[0]), int(source_size[1])],
        "input_size": [int(output_size[0]), int(output_size[1])],
        "output_size": [int(source_size[0]), int(source_size[1])],
        "matrix_convention": "stabilized_to_source",
        "source_matrix_convention": warp_meta.get("matrix_convention"),
        "framing_mode": warp_meta.get("framing_mode"),
        "note": "Restores original motion/canvas; pixels discarded by crop framing cannot be recovered.",
    }
    return InverseStabilizationResult(restored_frames, padding_masks, result_meta)


def _compute_bounding_boxes(
    matrices: Sequence[np.ndarray],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [width, 0.0, 1.0],
            [0.0, height, 1.0],
            [width, height, 1.0],
        ],
        dtype=np.float64,
    ).T

    mins = []
    maxs = []
    for matrix in matrices:
        warped = matrix @ corners
        warped /= warped[2, :]
        xs = warped[0, :]
        ys = warped[1, :]
        mins.append([xs.min(), ys.min()])
        maxs.append([xs.max(), ys.max()])
    return np.array(mins), np.array(maxs)


def _min_content_ratio(
    mins: np.ndarray,
    maxs: np.ndarray,
    width: int,
    height: int,
) -> float:
    x0 = np.max(mins[:, 0])
    y0 = np.max(mins[:, 1])
    x1 = np.min(maxs[:, 0])
    y1 = np.min(maxs[:, 1])

    intersection_w = max(0.0, x1 - x0)
    intersection_h = max(0.0, y1 - y0)
    if intersection_w <= 0.0 or intersection_h <= 0.0:
        return 1e-6
    return max(1e-6, min(intersection_w / width, intersection_h / height))


def _convert_masks_for_output(masks: Iterable[np.ndarray]) -> Any:
    """Convert internal mask list into tensor/array payload for Comfy outputs."""
    if isinstance(masks, np.ndarray) and masks.ndim in (3, 4):
        # Already a preallocated (N, H, W) or (N, H, W, 1) buffer.
        if not masks.shape[0]:
            stacked = np.zeros((1, 1, 1), dtype=np.float32)
        else:
            stacked = masks[..., 0] if masks.ndim == 4 else masks
    else:
        masks_2d: List[np.ndarray] = []
        for mask in masks:
            mask_2d = mask[..., 0] if mask.ndim == 3 else mask
            masks_2d.append(mask_2d.astype(np.float32))

        if not masks_2d:
            stacked = np.zeros((1, 1, 1), dtype=np.float32)
        else:
            stacked = np.stack(masks_2d, axis=0)

    stacked = np.ascontiguousarray(stacked, dtype=np.float32)
    if torch is not None:
        return torch.from_numpy(stacked)
    return stacked
