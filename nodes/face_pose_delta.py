"""face_pose_delta.py — Flagship #2: Face / Pose Delta Editor backend.

The user's vision (see ``third_party/ideas_summary.md §5``):

    "I edited the brows on one frame to make him angry, that change
    should follow his face through the whole clip — even when he
    turns his head."

A naive constant pixel offset drifts as the head moves. The correct
model is an **anchor-relative delta**:

  1. Compute a per-frame face anchor (centre + scale, optionally rotation)
     from a stable landmark subset (typically the inter-ocular base).
  2. Convert all landmarks into anchor-relative space: ``rel = (px - c) / s``.
  3. The user's edit on a keyframe is a delta in this normalised space:
     ``delta = rel_new - rel_old``.
  4. Propagate the delta to every frame in **anchor-relative** space, then
     convert back to per-frame pixel coordinates. The expression now
     sticks to the face as it moves.

Multi-keyframe edits blend with eased weights — reused from
``temporal_anchor.py``'s easing primitives — so the expression evolves
across the shot (e.g. frame 0 "slightly angry" → frame 30 "very angry").

This module ships:
  * Pure-math core (``compute_anchors``, ``to_relative``, ``from_relative``,
    ``eased_blend_weights``, ``propagate_keyframe_deltas``). All NumPy,
    deterministic, unit-testable without ComfyUI.
  * One ComfyUI node ``FacePoseDeltaCoreMEC`` that wraps the math behind a
    clean LANDMARKS-in / LANDMARKS-out signature plus a diagnostic JSON
    output. The UI editor that *produces* the keyframe-edit JSON is a
    separate JS extension (out of scope for this file — see
    ``js/c2c_face_pose_delta.js``).

License: Apache-2.0
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Easing functions are deliberately reused from temporal_anchor so the
# delta-editor blend looks identical to the SDF anchor blend the rest of
# the pack already uses.
from .temporal_anchor import _EASING_MAP  # noqa: WPS437 (intentional re-export)

log = logging.getLogger("MEC.FacePoseDelta")

# ───────────────────────────────────────────────────────────────────────
# Anchor extraction
# ───────────────────────────────────────────────────────────────────────

# MediaPipe FaceMesh indices used as a fallback inter-ocular base. The
# WanAnimatePreprocessV2 pose detector emits 478 landmarks per face in
# this layout (refine_landmarks=True). Indices 33 and 263 are the lateral
# canthi (outer corners of the eyes) — stable across blinks and yaw.
MP_LEFT_EYE_OUTER  = 33
MP_RIGHT_EYE_OUTER = 263

# A small numerical floor for scale so we never divide by zero on a tiny
# / undetected face. 1 pixel is the legitimate minimum.
_MIN_SCALE = 1.0


def _np2d(landmarks: Any) -> np.ndarray:
    """Coerce input to a contiguous ``(T, N, 2)`` float32 array.

    Accepts:
      * np.ndarray of shape ``(T, N, 2)`` or ``(T, N, 3)`` (z is dropped).
      * Python list-of-list-of-(x,y) or nested.
      * torch.Tensor (detached, moved to CPU, then numpy).
    Raises ValueError if the shape can't be reasonably interpreted.
    """
    # torch path — kept lazy so this module imports without torch installed
    if hasattr(landmarks, "detach") and hasattr(landmarks, "cpu"):
        landmarks = landmarks.detach().cpu().numpy()
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] in (2, 3):
        # Single frame (N, 2) → (1, N, 2)
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[2] not in (2, 3):
        raise ValueError(
            f"landmarks must be (T,N,2) or (T,N,3); got shape={arr.shape}"
        )
    if arr.shape[2] == 3:
        arr = arr[..., :2]
    return np.ascontiguousarray(arr, dtype=np.float32)


def compute_anchors(
    landmarks: np.ndarray,
    *,
    mode: str = "intereye",
    left_eye_idx: int = MP_LEFT_EYE_OUTER,
    right_eye_idx: int = MP_RIGHT_EYE_OUTER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-frame ``(centers, scales)`` for ``landmarks (T,N,2)``.

    ``mode``:
        * ``"intereye"`` — centre = midpoint of two given eye indices,
          scale = euclidean distance between them. Robust under yaw/pitch
          and blinks. Default — matches the user-confirmed plan in §5.4.
        * ``"centroid"`` — centre = arithmetic mean of all landmarks,
          scale = mean distance from centre. No index assumptions, but
          drifts more when the head moves and lips/eyebrows are weighted
          equally.

    Returns:
        centers : ``(T, 2)`` float32
        scales  : ``(T,)``   float32, clamped to ``>= _MIN_SCALE``
    """
    if landmarks.ndim != 3 or landmarks.shape[-1] != 2:
        raise ValueError(f"expected (T,N,2); got {landmarks.shape}")
    T, N, _ = landmarks.shape

    if mode == "intereye":
        if N <= max(left_eye_idx, right_eye_idx):
            log.warning(
                "compute_anchors: not enough landmarks (N=%d) for intereye "
                "indices (%d, %d); falling back to centroid",
                N, left_eye_idx, right_eye_idx,
            )
            mode = "centroid"
        else:
            le = landmarks[:, left_eye_idx, :]   # (T, 2)
            re = landmarks[:, right_eye_idx, :]  # (T, 2)
            centers = (le + re) * 0.5
            scales  = np.linalg.norm(re - le, axis=1)  # (T,)
            scales  = np.maximum(scales, _MIN_SCALE).astype(np.float32)
            return centers.astype(np.float32), scales

    # centroid fallback
    centers = landmarks.mean(axis=1)  # (T, 2)
    diffs   = landmarks - centers[:, None, :]
    scales  = np.linalg.norm(diffs, axis=2).mean(axis=1)  # (T,)
    scales  = np.maximum(scales, _MIN_SCALE).astype(np.float32)
    return centers.astype(np.float32), scales


def to_relative(
    landmarks: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Pixel → anchor-relative.  ``rel = (px - c) / s``."""
    return ((landmarks - centers[:, None, :]) / scales[:, None, None]).astype(np.float32)


def from_relative(
    rel: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Anchor-relative → pixel.  ``px = rel * s + c``."""
    return (rel * scales[:, None, None] + centers[:, None, :]).astype(np.float32)


# ───────────────────────────────────────────────────────────────────────
# Multi-keyframe eased blend
# ───────────────────────────────────────────────────────────────────────

def eased_blend_weights(
    frame: int,
    keyframes: Sequence[int],
    *,
    ease: str = "smooth_step",
    extrapolate: str = "hold",
) -> List[float]:
    """Return one weight per keyframe for ``frame``.

    The contract per ideas_summary §5.3:
        landmark[f] = base_rel[f] + Σ w_k(f) · delta_k
    where ``w_k`` is 1.0 at its own keyframe, smoothly blends to neighbours,
    and the sum across keyframes is 1.0 when ``frame`` lies inside the
    keyframe span. Outside the span behaviour is governed by ``extrapolate``:

        * ``"hold"``   — pin to nearest keyframe (deltas keep applying).
        * ``"zero"``   — return all-zero weights (deltas vanish).
        * ``"loop"``   — wrap modulo the keyframe span.
    """
    if not keyframes:
        return []
    # Sort and remember original positions so the caller can map weights
    # back to its keyframe list unchanged.
    order  = sorted(range(len(keyframes)), key=lambda i: keyframes[i])
    sorted_kfs = [keyframes[i] for i in order]
    K = len(sorted_kfs)
    weights_sorted = [0.0] * K
    ease_fn = _EASING_MAP.get(ease, _EASING_MAP["smooth_step"])

    f = int(frame)
    lo, hi = sorted_kfs[0], sorted_kfs[-1]

    if K == 1:
        # One keyframe: hold its delta across the entire timeline (subject
        # to extrapolate=zero overriding outside ±0 span).
        if extrapolate == "zero" and f != sorted_kfs[0]:
            weights_sorted[0] = 0.0
        else:
            weights_sorted[0] = 1.0
    elif f <= lo:
        if extrapolate == "hold":
            weights_sorted[0] = 1.0
        elif extrapolate == "loop":
            span = hi - lo
            f = lo + ((f - lo) % span) if span > 0 else lo
            # Fall through to the in-span branch on next iteration:
            return eased_blend_weights(f, keyframes, ease=ease, extrapolate="hold")
        # zero → leave all weights at 0
    elif f >= hi:
        if extrapolate == "hold":
            weights_sorted[-1] = 1.0
        elif extrapolate == "loop":
            span = hi - lo
            f = lo + ((f - lo) % span) if span > 0 else lo
            return eased_blend_weights(f, keyframes, ease=ease, extrapolate="hold")
        # zero → leave all at 0
    else:
        # Inside span — find the bracketing keyframes and ease between them.
        for k in range(K - 1):
            k0, k1 = sorted_kfs[k], sorted_kfs[k + 1]
            if k0 <= f <= k1:
                if k0 == k1:
                    weights_sorted[k] = 1.0
                    break
                t = (f - k0) / (k1 - k0)
                w1 = ease_fn(float(t))
                w0 = 1.0 - w1
                weights_sorted[k]     += w0
                weights_sorted[k + 1] += w1
                break

    # Re-permute back to caller order.
    out = [0.0] * K
    for src, dst in enumerate(order):
        out[dst] = weights_sorted[src]
    return out


def propagate_keyframe_deltas(
    base_rel: np.ndarray,
    keyframes: Sequence[Dict[str, Any]],
    *,
    ease: str = "smooth_step",
    extrapolate: str = "hold",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply user edits to ``base_rel (T,N,2)`` in anchor-relative space.

    Each entry in ``keyframes`` is::

        {
          "frame": int,                              # 0..T-1
          "deltas": {"<landmark_index>": [dx, dy]},  # in anchor-relative units
        }

    Returns
    -------
    rel_out : ``(T, N, 2)`` float32 — modified anchor-relative landmarks.
    info    : dict with diagnostic counters (per-keyframe usage etc.).
    """
    if base_rel.ndim != 3 or base_rel.shape[-1] != 2:
        raise ValueError(f"expected (T,N,2); got {base_rel.shape}")
    if not keyframes:
        return base_rel.copy(), {"keyframes_used": 0, "deltas_total": 0}

    T, N, _ = base_rel.shape
    # Build a dense delta tensor per keyframe: (K, N, 2)
    K = len(keyframes)
    kf_frames = [int(k.get("frame", 0)) for k in keyframes]
    kf_deltas = np.zeros((K, N, 2), dtype=np.float32)
    deltas_total = 0
    skipped = 0
    for ki, k in enumerate(keyframes):
        for idx_str, dxy in (k.get("deltas") or {}).items():
            try:
                idx = int(idx_str)
                if 0 <= idx < N and isinstance(dxy, (list, tuple)) and len(dxy) >= 2:
                    kf_deltas[ki, idx, 0] = float(dxy[0])
                    kf_deltas[ki, idx, 1] = float(dxy[1])
                    deltas_total += 1
                else:
                    skipped += 1
            except (ValueError, TypeError):
                skipped += 1

    rel_out = base_rel.copy()
    weight_log: List[List[float]] = []
    for f in range(T):
        w = eased_blend_weights(f, kf_frames, ease=ease, extrapolate=extrapolate)
        if any(abs(x) > 1e-9 for x in w):
            # rel_out[f] += Σ_k w_k · kf_deltas[k]
            stack = np.tensordot(np.asarray(w, dtype=np.float32), kf_deltas, axes=([0], [0]))
            rel_out[f] += stack
        if f < 5 or f >= T - 5:  # log only endpoints to keep info compact
            weight_log.append([round(x, 4) for x in w])

    info = {
        "keyframes_used":     K,
        "keyframe_frames":    kf_frames,
        "deltas_total":       deltas_total,
        "deltas_skipped":     skipped,
        "ease":               ease,
        "extrapolate":        extrapolate,
        "weights_head_tail":  weight_log,
    }
    return rel_out, info


def edit_landmarks(
    landmarks_in: Any,
    keyframes: Sequence[Dict[str, Any]],
    *,
    anchor_mode: str = "intereye",
    left_eye_idx: int = MP_LEFT_EYE_OUTER,
    right_eye_idx: int = MP_RIGHT_EYE_OUTER,
    ease: str = "smooth_step",
    extrapolate: str = "hold",
    external_centers: Optional[np.ndarray] = None,
    external_scales: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """End-to-end one-shot helper.

    Convert landmarks → anchor-relative → apply deltas → back to pixels.
    Returns the modified ``(T, N, 2)`` landmarks plus a diagnostic dict.

    If ``external_centers`` / ``external_scales`` are supplied (e.g. from
    a more accurate face tracker upstream) they bypass ``compute_anchors``.
    """
    lm = _np2d(landmarks_in)
    if external_centers is not None and external_scales is not None:
        centers = np.asarray(external_centers, dtype=np.float32).reshape(-1, 2)
        scales  = np.maximum(
            np.asarray(external_scales, dtype=np.float32).reshape(-1), _MIN_SCALE,
        )
        if centers.shape[0] != lm.shape[0] or scales.shape[0] != lm.shape[0]:
            raise ValueError(
                f"external anchors length ({centers.shape[0]}/{scales.shape[0]}) "
                f"must match T={lm.shape[0]}"
            )
    else:
        centers, scales = compute_anchors(
            lm, mode=anchor_mode,
            left_eye_idx=left_eye_idx, right_eye_idx=right_eye_idx,
        )

    rel = to_relative(lm, centers, scales)
    rel_out, info = propagate_keyframe_deltas(
        rel, keyframes, ease=ease, extrapolate=extrapolate,
    )
    out = from_relative(rel_out, centers, scales)
    info["anchor_mode"]   = anchor_mode
    info["frames"]        = int(lm.shape[0])
    info["landmark_count"] = int(lm.shape[1])
    info["scale_mean"]    = float(scales.mean())
    info["scale_min"]     = float(scales.min())
    info["scale_max"]     = float(scales.max())
    return out, info


# ───────────────────────────────────────────────────────────────────────
# ComfyUI node wrapper
# ───────────────────────────────────────────────────────────────────────

_EXAMPLE_KEYFRAMES_JSON = (
    '{\n'
    '  "keyframes": [\n'
    '    {"frame": 0,  "deltas": {"61": [-0.02,  0.01]}},\n'
    '    {"frame": 30, "deltas": {"61": [-0.08,  0.04]}}\n'
    '  ],\n'
    '  "ease": "smooth_step",\n'
    '  "extrapolate": "hold"\n'
    '}'
)


def _parse_keyframes(text: str) -> Tuple[List[Dict[str, Any]], str, str]:
    """Parse the user's keyframe JSON. Returns (keyframes, ease, extrapolate)
    and is defensive — empty / malformed input gives a no-op pass-through.
    """
    if not text or not text.strip():
        return [], "smooth_step", "hold"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("[FacePoseDelta] keyframes JSON invalid: %s", e)
        return [], "smooth_step", "hold"
    if not isinstance(data, dict):
        return [], "smooth_step", "hold"
    kfs = data.get("keyframes") or []
    if not isinstance(kfs, list):
        kfs = []
    ease = str(data.get("ease") or "smooth_step")
    extrap = str(data.get("extrapolate") or "hold")
    if ease not in _EASING_MAP:
        ease = "smooth_step"
    if extrap not in ("hold", "zero", "loop"):
        extrap = "hold"
    return kfs, ease, extrap


class FacePoseDeltaCoreMEC:
    """Anchor-relative landmark editor.

    Takes a per-frame landmark stream (T, N, 2) — typically MediaPipe
    FaceMesh's 478 points from WanAnimatePreprocessV2 — and applies
    user-specified deltas from one or more keyframes in **anchor-relative
    space**, so the edit follows the face as the head moves.

    The keyframe deltas are expressed as a JSON blob:

        {
          "keyframes": [
            {"frame": 0,  "deltas": {"61": [-0.02, 0.01]}},   ← left mouth corner pulled left
            {"frame": 30, "deltas": {"61": [-0.08, 0.04]}}    ← more pulled by frame 30
          ],
          "ease": "smooth_step",
          "extrapolate": "hold"
        }

    Units are **anchor-relative**: 1.0 = inter-ocular distance. So a delta
    of [0.1, 0] moves a landmark sideways by 10% of the eye-to-eye span,
    independent of how big the face is in any given frame.

    Pair this with the upcoming ``c2c_face_pose_delta.js`` editor (drag
    landmarks on a keyframe, save → JSON) for a fully no-code workflow.
    """

    CATEGORY = "MaskEditControl/Pose"
    FUNCTION = "execute"
    DESCRIPTION = (
        "Apply user-edited landmark deltas to a per-frame face/pose "
        "landmark stream in anchor-relative space, so the edit follows "
        "the face as the head moves. Multi-keyframe with eased blend."
    )
    RETURN_TYPES = ("LANDMARKS", "STRING")
    RETURN_NAMES = ("landmarks_modified", "info_json")
    OUTPUT_TOOLTIPS = (
        "Modified landmarks (T, N, 2) — same shape and ordering as input.",
        "JSON: per-frame anchor stats, keyframes used, deltas applied, "
        "weights at head/tail frames. Useful for debugging the blend.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "landmarks": ("LANDMARKS", {
                    "tooltip": "Per-frame landmarks (T,N,2) — typically MediaPipe FaceMesh "
                               "from WanAnimatePreprocessV2's gaze/pose detector. Single "
                               "frame (N,2) is auto-promoted to T=1.",
                }),
                "keyframe_edits_json": ("STRING", {
                    "multiline": True,
                    "default": _EXAMPLE_KEYFRAMES_JSON,
                    "tooltip": "JSON describing one or more keyframe edits in "
                               "anchor-relative space (1.0 unit = inter-ocular distance).",
                }),
                "anchor_mode": (["intereye", "centroid"], {
                    "default": "intereye",
                    "tooltip": "How to compute the per-frame face anchor. 'intereye' "
                               "is robust under head turns; 'centroid' is a fallback "
                               "for non-face landmark sets.",
                }),
                "left_eye_idx": ("INT", {
                    "default": MP_LEFT_EYE_OUTER, "min": 0, "max": 100000,
                    "tooltip": "Landmark index of the outer-left eye corner. "
                               "MediaPipe FaceMesh = 33.",
                }),
                "right_eye_idx": ("INT", {
                    "default": MP_RIGHT_EYE_OUTER, "min": 0, "max": 100000,
                    "tooltip": "Landmark index of the outer-right eye corner. "
                               "MediaPipe FaceMesh = 263.",
                }),
            },
            "optional": {
                "external_anchors_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional override for per-frame anchors. JSON: "
                               "{\"centers\": [[cx,cy], ...], \"scales\": [s, ...]}. "
                               "If present, replaces the computed intereye/centroid anchors.",
                }),
            },
        }

    def execute(
        self,
        landmarks,
        keyframe_edits_json: str,
        anchor_mode: str,
        left_eye_idx: int,
        right_eye_idx: int,
        external_anchors_json: str = "",
    ):
        kfs, ease, extrap = _parse_keyframes(keyframe_edits_json)

        ext_centers = ext_scales = None
        if external_anchors_json and external_anchors_json.strip():
            try:
                ed = json.loads(external_anchors_json)
                if isinstance(ed, dict):
                    if "centers" in ed:
                        ext_centers = np.asarray(ed["centers"], dtype=np.float32)
                    if "scales" in ed:
                        ext_scales = np.asarray(ed["scales"], dtype=np.float32)
            except json.JSONDecodeError as e:
                log.warning("[FacePoseDelta] external_anchors_json invalid: %s", e)

        out, info = edit_landmarks(
            landmarks, kfs,
            anchor_mode=anchor_mode,
            left_eye_idx=int(left_eye_idx),
            right_eye_idx=int(right_eye_idx),
            ease=ease, extrapolate=extrap,
            external_centers=ext_centers,
            external_scales=ext_scales,
        )
        return (out, json.dumps(info, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {"FacePoseDeltaCoreMEC": FacePoseDeltaCoreMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"FacePoseDeltaCoreMEC": "Face/Pose Delta Editor (C2C)"}
