"""
BBoxSmooth – temporal smoothing for video-tracked bounding boxes.

Apr 2026 cleanup: BBoxCreate / BBoxFromMask / BBoxToMask / BBoxPad / BBoxCrop
were removed from this pack — they duplicated Impact Pack and core ComfyUI
nodes. Original source preserved at ``_deprecated/bbox_nodes.snapshot.py``.
"""

import json
import numpy as np

from ._is_changed_util import hash_args_and_kwargs


class BBoxSmooth:
    """Smooth a sequence of bounding boxes across video frames to reduce jitter.

    Supports moving average, exponential, and median-based outlier rejection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes_json": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "JSON array of [x, y, w, h] bboxes, one per frame. e.g. [[10,20,100,100],[12,19,102,101],...]"}),
                "smoothing_radius": ("INT", {
                    "default": 3, "min": 1, "max": 30, "step": 1,
                    "tooltip": "Temporal window radius for smoothing (higher = smoother but more lag)"}),
                "method": (["median_then_exponential", "moving_average", "exponential", "median"], {
                    "default": "median_then_exponential",
                    "tooltip": (
                        "median_then_exponential: median filter for outlier rejection, then exponential smooth (recommended). "
                        "moving_average: uniform window average. "
                        "exponential: recent frames weighted more. "
                        "median: pure median filter (removes jumps)."
                    )}),
                "alpha": ("FLOAT", {
                    "default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Exponential smoothing factor (lower = smoother). Only used by exponential and median_then_exponential methods."}),
            },
        }

    RETURN_TYPES = ("STRING", "BBOX",)
    RETURN_NAMES = ("smoothed_bboxes_json", "first_bbox",)
    OUTPUT_TOOLTIPS = (
        "JSON array of smoothed [x, y, w, h] bboxes, one per input frame.",
        "First bbox of the smoothed sequence as a BBOX tuple.",
    )
    FUNCTION = "smooth"
    CATEGORY = "C2C/BBox"
    DESCRIPTION = "Smooth bounding boxes across video frames to eliminate jitter. Median-based outlier rejection + exponential smoothing for best results."

    @classmethod
    def IS_CHANGED(cls, bboxes_json, smoothing_radius, method, alpha, **kwargs):
        return hash_args_and_kwargs(bboxes_json, smoothing_radius, method, alpha, **kwargs)

    def smooth(self, bboxes_json, smoothing_radius, method, alpha):
        try:
            bboxes = json.loads(bboxes_json)
        except (json.JSONDecodeError, TypeError):
            bbox = [0, 0, 128, 128]
            return (json.dumps([bbox]), bbox)

        if not bboxes or not isinstance(bboxes, list):
            bbox = [0, 0, 128, 128]
            return (json.dumps([bbox]), bbox)

        n = len(bboxes)
        if n <= 1:
            return (json.dumps(bboxes), bboxes[0] if bboxes else [0, 0, 128, 128])

        arr = np.array(bboxes, dtype=np.float64)  # (N, 4)

        # Step 1: Median filter for outlier rejection
        if method in ("median", "median_then_exponential"):
            filtered = np.copy(arr)
            for i in range(n):
                start = max(0, i - smoothing_radius)
                end = min(n, i + smoothing_radius + 1)
                window = arr[start:end]
                filtered[i] = np.median(window, axis=0)
            arr = filtered

        # Step 2: Smoothing
        if method == "moving_average":
            smoothed = np.copy(arr)
            for i in range(n):
                start = max(0, i - smoothing_radius)
                end = min(n, i + smoothing_radius + 1)
                window = arr[start:end]
                smoothed[i] = np.mean(window, axis=0)
            arr = smoothed
        elif method in ("exponential", "median_then_exponential"):
            smoothed = np.copy(arr)
            for i in range(1, n):
                smoothed[i] = alpha * arr[i] + (1.0 - alpha) * smoothed[i - 1]
            arr = smoothed

        result = [[int(round(row[0])), int(round(row[1])),
                   int(round(row[2])), int(round(row[3]))]
                  for row in arr]

        return (json.dumps(result), result[0])
