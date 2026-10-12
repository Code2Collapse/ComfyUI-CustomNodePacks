# PORTED FROM: ComfyUI-Video-Stabilizer by nomadoor and contributors (MIT) - https://github.com/nomadoor/ComfyUI-Video-Stabilizer
# nodes/motion_meta.py at d516bf0 (2026-07-04), code unchanged. Licence: ./LICENSE. Loaded by nodes/video_stabilizer_mec.py (L7.90:
# it used to load from the gitignored third_party/ folder, so the four Video Stabilizer nodes failed on every fresh install).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FrameTransform:
    index: int
    matrix: np.ndarray


@dataclass(frozen=True)
class MotionMeta:
    source: str
    frame_count: int
    fps: float
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    per_frame: list[FrameTransform]
    generator: Dict[str, Any] | None = None


def _read_positive_size(block_name: str, block: Dict[str, Any], key: str) -> Tuple[int, int]:
    value = block.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{block_name}.{key} must be [width, height].")
    try:
        width = int(value[0])
        height = int(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{block_name}.{key} must contain integer width/height.") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{block_name}.{key} must contain positive width/height.")
    return width, height


def _read_finite_matrix(block_name: str, entry: Any, expected_index: int, key: str) -> np.ndarray:
    if not isinstance(entry, dict):
        raise ValueError(f"{block_name}.per_frame[{expected_index}] must be an object.")
    if entry.get("index") != expected_index:
        raise ValueError(
            f"{block_name}.per_frame[{expected_index}].index must be {expected_index}, "
            f"got {entry.get('index')!r}."
        )
    if key not in entry:
        raise ValueError(f"{block_name}.per_frame[{expected_index}].{key} is missing.")
    matrix = np.asarray(entry[key], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{block_name}.per_frame[{expected_index}].{key} must be 3x3.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{block_name}.per_frame[{expected_index}].{key} must contain finite numbers.")
    try:
        np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{block_name}.per_frame[{expected_index}].{key} is not invertible.") from exc
    return matrix


def validate_motion_meta(block: Dict[str, Any]) -> None:
    if not isinstance(block, dict):
        raise ValueError("motion_meta must be an object.")
    if block.get("version") != 2:
        raise ValueError(f"motion_meta.version must be 2, got {block.get('version')!r}.")
    if block.get("matrix_convention") != "input_to_output":
        raise ValueError(
            "motion_meta.matrix_convention must be 'input_to_output', "
            f"got {block.get('matrix_convention')!r}."
        )
    source = block.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("motion_meta.source must be a non-empty string.")
    try:
        frame_count = int(block.get("frame_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("motion_meta.frame_count must be an integer.") from exc
    if frame_count < 0:
        raise ValueError("motion_meta.frame_count must be non-negative.")
    try:
        fps = float(block.get("fps"))
    except (TypeError, ValueError) as exc:
        raise ValueError("motion_meta.fps must be a positive number.") from exc
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("motion_meta.fps must be a positive number.")
    _read_positive_size("motion_meta", block, "input_size")
    _read_positive_size("motion_meta", block, "output_size")
    per_frame = block.get("per_frame")
    if not isinstance(per_frame, list):
        raise ValueError("motion_meta.per_frame must be a list.")
    if len(per_frame) != frame_count:
        raise ValueError(
            "motion_meta.frame_count mismatch: "
            f"frame_count is {frame_count}, per_frame has {len(per_frame)} entry/entries."
        )
    for idx, entry in enumerate(per_frame):
        _read_finite_matrix("motion_meta", entry, idx, "matrix")
    if source == "generated_shake" and not isinstance(block.get("generator"), dict):
        raise ValueError("motion_meta.generator is required when source is 'generated_shake'.")


def _motion_meta_from_block(block: Dict[str, Any]) -> MotionMeta:
    validate_motion_meta(block)
    per_frame = [
        FrameTransform(
            index=idx,
            matrix=np.asarray(entry["matrix"], dtype=np.float64),
        )
        for idx, entry in enumerate(block["per_frame"])
    ]
    return MotionMeta(
        source=str(block["source"]),
        frame_count=int(block["frame_count"]),
        fps=float(block["fps"]),
        input_size=_read_positive_size("motion_meta", block, "input_size"),
        output_size=_read_positive_size("motion_meta", block, "output_size"),
        per_frame=per_frame,
        generator=dict(block["generator"]) if isinstance(block.get("generator"), dict) else None,
    )


def build_motion_meta_v2(
    *,
    source: str,
    frame_count: int,
    fps: float,
    input_size: Tuple[int, int],
    output_size: Tuple[int, int],
    matrices: Sequence[np.ndarray],
    generator: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "version": 2,
        "source": source,
        "frame_count": int(frame_count),
        "fps": float(fps),
        "input_size": [int(input_size[0]), int(input_size[1])],
        "output_size": [int(output_size[0]), int(output_size[1])],
        "matrix_convention": "input_to_output",
        "per_frame": [
            {
                "index": int(idx),
                "matrix": np.asarray(matrix, dtype=np.float64).tolist(),
            }
            for idx, matrix in enumerate(matrices)
        ],
    }
    if generator is not None:
        block["generator"] = dict(generator)
    validate_motion_meta(block)
    return block


def motion_meta_from_stabilization_warp(
    warp_meta: Dict[str, Any],
    fps: float,
    source: str,
) -> Dict[str, Any] | None:
    if not isinstance(warp_meta, dict):
        raise ValueError("stabilization_warp must be an object.")
    if warp_meta.get("matrix_convention") != "source_to_stabilized":
        raise ValueError(
            "stabilization_warp.matrix_convention must be 'source_to_stabilized', "
            f"got {warp_meta.get('matrix_convention')!r}."
        )
    source_size = _read_positive_size("stabilization_warp", warp_meta, "source_size")
    output_size = _read_positive_size("stabilization_warp", warp_meta, "output_size")
    per_frame = warp_meta.get("per_frame")
    if not isinstance(per_frame, list):
        raise ValueError("stabilization_warp.per_frame must be a list.")

    matrices: list[np.ndarray] = []
    for idx, entry in enumerate(per_frame):
        matrix = _read_finite_matrix("stabilization_warp", entry, idx, "applied_matrix")
        try:
            matrices.append(np.linalg.inv(matrix))
        except np.linalg.LinAlgError:
            return None

    return build_motion_meta_v2(
        source=source,
        frame_count=len(matrices),
        fps=fps,
        input_size=output_size,
        output_size=source_size,
        matrices=matrices,
    )


def applied_motion_meta_from_stabilization_warp(
    warp_meta: Dict[str, Any],
    fps: float,
    source: str,
) -> Dict[str, Any]:
    if not isinstance(warp_meta, dict):
        raise ValueError("stabilization_warp must be an object.")
    if warp_meta.get("matrix_convention") != "source_to_stabilized":
        raise ValueError(
            "stabilization_warp.matrix_convention must be 'source_to_stabilized', "
            f"got {warp_meta.get('matrix_convention')!r}."
        )
    source_size = _read_positive_size("stabilization_warp", warp_meta, "source_size")
    output_size = _read_positive_size("stabilization_warp", warp_meta, "output_size")
    per_frame = warp_meta.get("per_frame")
    if not isinstance(per_frame, list):
        raise ValueError("stabilization_warp.per_frame must be a list.")

    matrices = [
        _read_finite_matrix("stabilization_warp", entry, idx, "applied_matrix")
        for idx, entry in enumerate(per_frame)
    ]
    return build_motion_meta_v2(
        source=source,
        frame_count=len(matrices),
        fps=fps,
        input_size=source_size,
        output_size=output_size,
        matrices=matrices,
    )


def resolve_motion_meta(meta: Dict[str, Any]) -> MotionMeta:
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dictionary containing motion_meta or stabilization_warp.")
    motion_block = meta.get("motion_meta")
    if isinstance(motion_block, dict):
        return _motion_meta_from_block(motion_block)
    warp_meta = meta.get("stabilization_warp")
    if isinstance(warp_meta, dict):
        block = motion_meta_from_stabilization_warp(warp_meta, fps=16.0, source="legacy_stabilization")
        if block is None:
            raise ValueError("stabilization_warp contains a non-invertible applied_matrix.")
        return _motion_meta_from_block(block)
    raise ValueError("meta must contain motion_meta or stabilization_warp.")
