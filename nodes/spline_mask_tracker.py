"""Spline Mask Tracker (C2C) — Item #4.

User goal: "Draw a spline on the first frame, some middle frames and the
final frame; the node tracks the pixel shifts and draws a continuous
mask across every frame in between."

Approach (CPU-only, no extra weights, real working algorithm):

  1. Parse a per-keyframe control-point JSON
       [{"frame": 0,  "points": [[x,y], ...]},
        {"frame": 8,  "points": [[x,y], ...]},
        {"frame": 15, "points": [[x,y], ...]}]
     All keyframes must have the SAME number of points (the user's
     "shape" — we morph control-point-by-control-point).

  2. Between every pair of adjacent keyframes, run sparse Lucas-Kanade
     pyramidal optical flow on the control-point set to learn the true
     per-frame pixel motion (not just linear interpolation):
        cv2.calcOpticalFlowPyrLK with sub-pixel cornerSubPix refinement.

  3. Blend the tracker's predicted position with linear interpolation
     between keyframes.  This gives a result that:
        * passes EXACTLY through every user-keyframed control point
        * follows real image motion in between (so a tracked feature
          sliding across the frame is followed, not lerped through air).

  4. For each frame, expand the control points into a smooth Catmull-Rom
     curve and rasterize with cv2.fillPoly (closed) / cv2.polylines (open).

Inputs:
  image           IMAGE  (B,H,W,C) float32 [0,1]
  keyframes_json  STRING  JSON list described above
  closed          BOOL    fillPoly vs polylines
  samples_per_segment INT (default 24) — curve resolution
  feather_radius  FLOAT   Gaussian blur applied to final mask
  tracking_weight FLOAT   0=pure lerp, 1=pure tracker. Default 0.7.
  klt_window      INT     LK search window. Default 21.

Outputs:
  mask  MASK (B,H,W) float32 [0,1]
"""

from __future__ import annotations

import json
import logging
from typing import List, Tuple

import numpy as np
import torch

log = logging.getLogger("MEC.spline_mask_tracker")

try:
    import cv2  # type: ignore
    _CV2 = True
except Exception as _e:  # noqa: BLE001
    cv2 = None  # type: ignore
    _CV2 = False
    log.warning("[MEC] spline_mask_tracker: cv2 not importable (%s)", _e)


# ─────────────────────────────────────────────────────────────────────
#  Catmull-Rom spline (centripetal, alpha=0.5)
# ─────────────────────────────────────────────────────────────────────
def _catmull_rom_segment(p0, p1, p2, p3, n: int, alpha: float = 0.5) -> np.ndarray:
    """Return n+1 points on the centripetal Catmull-Rom segment p1->p2."""
    def tj(ti, pi, pj):
        d = np.hypot(pj[0] - pi[0], pj[1] - pi[1])
        return ti + (d ** alpha if d > 1e-6 else 1e-6)

    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    ts = np.linspace(t1, t2, n + 1)

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    pts = np.zeros((len(ts), 2), dtype=np.float64)
    for i, t in enumerate(ts):
        a1 = (t1 - t) / max(t1 - t0, 1e-9) * p0 + (t - t0) / max(t1 - t0, 1e-9) * p1
        a2 = (t2 - t) / max(t2 - t1, 1e-9) * p1 + (t - t1) / max(t2 - t1, 1e-9) * p2
        a3 = (t3 - t) / max(t3 - t2, 1e-9) * p2 + (t - t2) / max(t3 - t2, 1e-9) * p3
        b1 = (t2 - t) / max(t2 - t0, 1e-9) * a1 + (t - t0) / max(t2 - t0, 1e-9) * a2
        b2 = (t3 - t) / max(t3 - t1, 1e-9) * a2 + (t - t1) / max(t3 - t1, 1e-9) * a3
        c  = (t2 - t) / max(t2 - t1, 1e-9) * b1 + (t - t1) / max(t2 - t1, 1e-9) * b2
        pts[i] = c
    return pts


def _catmull_rom_curve(control_pts: np.ndarray, samples_per_segment: int,
                       closed: bool) -> np.ndarray:
    """Build a smooth curve through `control_pts` (Nx2)."""
    n = len(control_pts)
    if n < 2:
        return control_pts.copy()
    if n == 2:
        # Straight line.
        ts = np.linspace(0, 1, samples_per_segment + 1)[:, None]
        return control_pts[0] + ts * (control_pts[1] - control_pts[0])

    out = []
    if closed:
        for i in range(n):
            p0 = control_pts[(i - 1) % n]
            p1 = control_pts[i % n]
            p2 = control_pts[(i + 1) % n]
            p3 = control_pts[(i + 2) % n]
            seg = _catmull_rom_segment(p0, p1, p2, p3, samples_per_segment)
            out.append(seg[:-1])  # drop last to avoid dup
        return np.concatenate(out, axis=0)
    # Open: clamp the endpoints by reflecting.
    pad_start = 2 * control_pts[0] - control_pts[1]
    pad_end = 2 * control_pts[-1] - control_pts[-2]
    pts = np.vstack([pad_start, control_pts, pad_end])
    for i in range(1, len(pts) - 2):
        seg = _catmull_rom_segment(pts[i - 1], pts[i], pts[i + 1], pts[i + 2],
                                   samples_per_segment)
        out.append(seg[:-1])
    out.append(pts[-2:-1])  # final endpoint
    return np.concatenate(out, axis=0)


# ─────────────────────────────────────────────────────────────────────
#  Tracking
# ─────────────────────────────────────────────────────────────────────
def _track_points_lk(prev_gray: np.ndarray, curr_gray: np.ndarray,
                     pts: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pyramidal LK + sub-pixel refinement.  Returns (new_pts, status)."""
    pts32 = np.ascontiguousarray(pts.reshape(-1, 1, 2).astype(np.float32))
    lk_params = dict(
        winSize=(int(win), int(win)),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    new_pts, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts32, None,
                                                  **lk_params)
    if new_pts is None:
        return pts.copy(), np.zeros(len(pts), dtype=np.uint8)
    new_pts = new_pts.reshape(-1, 2)
    status = (st.reshape(-1) if st is not None else np.zeros(len(pts), dtype=np.uint8))
    # Sub-pixel refinement on the tracked locations (helps when control
    # points sit on corners).
    valid = status.astype(bool) & np.all(np.isfinite(new_pts), axis=1)
    if valid.any():
        refine = np.ascontiguousarray(new_pts[valid].reshape(-1, 1, 2).astype(np.float32))
        try:
            cv2.cornerSubPix(
                curr_gray, refine,
                winSize=(5, 5), zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            new_pts[valid] = refine.reshape(-1, 2)
        except cv2.error:
            pass
    return new_pts, status


def _to_uint8_gray(frame_rgb_f32: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((frame_rgb_f32 * 255.0).clip(0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY)


# ─────────────────────────────────────────────────────────────────────
#  Keyframe parsing
# ─────────────────────────────────────────────────────────────────────
def _parse_keyframes(js: str, B: int) -> List[Tuple[int, np.ndarray]]:
    """Return sorted list of (frame_index, Nx2 control-point array)."""
    if not js or not js.strip():
        return []
    try:
        data = json.loads(js)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"keyframes_json is not valid JSON: {e}") from e
    if not isinstance(data, list):
        raise ValueError("keyframes_json must be a JSON list")
    parsed: List[Tuple[int, np.ndarray]] = []
    n_first = None
    for kf in data:
        if not isinstance(kf, dict):
            raise ValueError("Each keyframe must be an object")
        f = int(kf.get("frame", 0))
        f = max(0, min(B - 1, f))
        raw_pts = kf.get("points", [])
        if not raw_pts:
            continue
        pts = np.asarray(raw_pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                f"Keyframe at frame {f}: points must be a list of [x,y] pairs"
            )
        if n_first is None:
            n_first = len(pts)
        elif len(pts) != n_first:
            raise ValueError(
                f"Keyframe at frame {f} has {len(pts)} points; expected "
                f"{n_first} to match the first keyframe (the shape topology "
                f"must be consistent across keyframes)."
            )
        parsed.append((f, pts))
    parsed.sort(key=lambda x: x[0])
    return parsed


# ─────────────────────────────────────────────────────────────────────
#  Node
# ─────────────────────────────────────────────────────────────────────
class SplineMaskTrackerMEC:
    """Multi-keyframe spline tracker. Draw a closed/open spline on a few
    keyframes (first, middle, last) and the node uses Lucas-Kanade
    optical flow + Catmull-Rom interpolation to draw a continuous mask
    on every frame in between, following actual pixel motion."""

    VRAM_TIER = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Video frames (B,H,W,C). Tracker runs on these.",
                }),
                "keyframes_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": (
                        "List of keyframes — JSON.\n"
                        "Each keyframe: {\"frame\": <int>, \"points\": [[x,y],...]}\n"
                        "All keyframes must have the SAME number of points (the\n"
                        "shape morphs control-point-by-control-point)."
                    ),
                }),
                "closed": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Closed spline (fill interior) vs. open spline (stroke).",
                }),
                "samples_per_segment": ("INT", {
                    "default": 24, "min": 4, "max": 128, "step": 1,
                    "tooltip": "Catmull-Rom resolution per segment.",
                }),
                "tracking_weight": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Blend between linear interpolation and LK tracking.\n"
                        "0.0 = pure lerp (ignores image content)\n"
                        "1.0 = pure tracker (follows pixels exactly)\n"
                        "0.7 = recommended — tracker-led, lerp-anchored."
                    ),
                }),
                "klt_window": ("INT", {
                    "default": 21, "min": 5, "max": 51, "step": 2,
                    "tooltip": "Lucas-Kanade search window (px). Larger = "
                               "handles bigger motion but less precise.",
                }),
                "feather_radius": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 32.0, "step": 0.5,
                    "tooltip": "Gaussian blur on the final mask edge.",
                }),
                "stroke_width": ("INT", {
                    "default": 3, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Stroke width for OPEN splines (ignored if closed=True).",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    OUTPUT_TOOLTIPS = (
        "Per-frame rasterized mask (B,H,W) float32 [0,1].",
        "Diagnostic JSON: keyframes used, tracking failure count, etc.",
    )
    FUNCTION = "execute"
    CATEGORY = "C2C/Spline"
    DESCRIPTION = (
        "Draw a spline on a few keyframes (first, middle, last) and the "
        "node uses Lucas-Kanade tracking + Catmull-Rom to draw a "
        "continuous mask on every frame in between, following real "
        "pixel motion. No model weights, CPU-only."
    )

    # -----------------------------------------------------------------
    def execute(
        self,
        image: torch.Tensor,
        keyframes_json: str,
        closed: bool,
        samples_per_segment: int,
        tracking_weight: float,
        klt_window: int,
        feather_radius: float,
        stroke_width: int,
    ) -> Tuple[torch.Tensor, str]:

        if not _CV2:
            raise RuntimeError(
                "SplineMaskTrackerMEC requires opencv-python. "
                "Install with: pip install opencv-python"
            )

        img_np = image.detach().cpu().float().numpy().clip(0.0, 1.0)
        if img_np.ndim != 4:
            raise ValueError(f"Expected image (B,H,W,C), got {img_np.shape}")
        B, H, W, C = img_np.shape

        kfs = _parse_keyframes(keyframes_json, B)
        if not kfs:
            log.warning("[MEC.spline_tracker] no keyframes; returning zero mask")
            empty = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)
            info = json.dumps({"status": "empty", "keyframes": 0})
            return (empty, info)

        # Auto-insert sentinel keyframes if user didn't pin start/end —
        # we just duplicate the nearest keyframe so the tracker has work
        # to do on those frames. (Common case: user gives kf at 0 and 15
        # of an 8-frame batch.)
        if kfs[0][0] > 0:
            kfs.insert(0, (0, kfs[0][1].copy()))
        if kfs[-1][0] < B - 1:
            kfs.append((B - 1, kfs[-1][1].copy()))

        n_pts = len(kfs[0][1])

        # ----- Step A: track control points frame-by-frame -----------
        # We track FORWARD from each keyframe to the next one, then BLEND
        # with linear interpolation. This way the user-pinned keyframes
        # always match exactly.
        grays = [_to_uint8_gray(img_np[i]) for i in range(B)]
        tracked_per_frame = np.zeros((B, n_pts, 2), dtype=np.float64)
        # Initialize all frames to the first keyframe (used only as a fallback).
        for f in range(B):
            tracked_per_frame[f] = kfs[0][1]

        fail_count = 0
        for seg_idx in range(len(kfs) - 1):
            f_start, pts_start = kfs[seg_idx]
            f_end, pts_end = kfs[seg_idx + 1]
            if f_end <= f_start:
                continue

            # FORWARD track from f_start using pts_start.
            fwd = np.zeros((f_end - f_start + 1, n_pts, 2), dtype=np.float64)
            fwd[0] = pts_start
            for k in range(1, f_end - f_start + 1):
                new_pts, status = _track_points_lk(
                    grays[f_start + k - 1], grays[f_start + k],
                    fwd[k - 1], klt_window,
                )
                # Where LK failed (status=0), fall back to previous position.
                bad = status == 0
                fail_count += int(bad.sum())
                new_pts[bad] = fwd[k - 1][bad]
                fwd[k] = new_pts

            # BACKWARD track from f_end using pts_end.
            bwd = np.zeros((f_end - f_start + 1, n_pts, 2), dtype=np.float64)
            bwd[-1] = pts_end
            for k in range(f_end - f_start - 1, -1, -1):
                new_pts, status = _track_points_lk(
                    grays[f_start + k + 1], grays[f_start + k],
                    bwd[k + 1], klt_window,
                )
                bad = status == 0
                fail_count += int(bad.sum())
                new_pts[bad] = bwd[k + 1][bad]
                bwd[k] = new_pts

            # BLEND forward + backward by linear weight (alpha=0 at f_start,
            # alpha=1 at f_end), then BLEND that with pure lerp by
            # tracking_weight.
            n_seg = f_end - f_start
            for k in range(n_seg + 1):
                alpha = k / max(n_seg, 1)
                tracked_combo = (1 - alpha) * fwd[k] + alpha * bwd[k]
                lerp = (1 - alpha) * pts_start + alpha * pts_end
                tw = float(tracking_weight)
                tracked_per_frame[f_start + k] = tw * tracked_combo + (1 - tw) * lerp

        # Force exact keyframe match (the blend pinned the endpoints, but
        # rounding may have drifted a hair — set them dead-on).
        for f, pts in kfs:
            tracked_per_frame[f] = pts

        # ----- Step B: rasterize per-frame mask ----------------------
        out = np.zeros((B, H, W), dtype=np.float32)
        for f in range(B):
            curve = _catmull_rom_curve(tracked_per_frame[f],
                                       samples_per_segment, closed)
            # Clip and convert to int32 for fillPoly.
            curve_int = np.round(curve).astype(np.int32)
            curve_int[:, 0] = curve_int[:, 0].clip(0, W - 1)
            curve_int[:, 1] = curve_int[:, 1].clip(0, H - 1)
            frame_mask = np.zeros((H, W), dtype=np.uint8)
            if closed:
                cv2.fillPoly(frame_mask, [curve_int], 255)
            else:
                cv2.polylines(frame_mask, [curve_int], False, 255,
                              thickness=int(stroke_width), lineType=cv2.LINE_AA)
            if feather_radius > 0.0:
                k = max(1, int(round(feather_radius)) * 2 + 1)
                frame_mask = cv2.GaussianBlur(frame_mask, (k, k), 0)
            out[f] = frame_mask.astype(np.float32) / 255.0

        info = json.dumps({
            "status": "ok",
            "keyframes": [int(f) for f, _ in kfs],
            "n_control_points": int(n_pts),
            "lk_failures": int(fail_count),
            "frames": int(B),
        })
        return (torch.from_numpy(out).to(image.device, dtype=torch.float32), info)


# ─────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {"SplineMaskTrackerMEC": SplineMaskTrackerMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplineMaskTrackerMEC": "Spline Mask Tracker",
}
