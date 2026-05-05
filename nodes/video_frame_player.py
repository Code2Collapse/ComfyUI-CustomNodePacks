"""VideoFramePlayerMEC â€“ lightweight in-graph video scrubber.

Shows an IMAGE batch as a scrubable timeline with a frame slider (OpenRV
style). Outputs the currently-selected frame as a single IMAGE so it can be
piped into other nodes.

Lightweight design:
  - Sends each frame as a small JPEG to the browser (configurable preview
    width); no client-side video decoding needed.
  - JS widget renders one frame at a time on a single canvas + timeline bar.
  - Server returns the *full-resolution* frame at the current `frame_index`
    widget value as the IMAGE output (so downstream nodes get full quality).
"""

from __future__ import annotations

from . import _interrupt_check as _IC

import os
import io
import hashlib

import numpy as np
import torch
from PIL import Image

import folder_paths


from . import _progress as _PB
_PREVIEW_QUALITY = 80


class VideoFramePlayerMEC:
    """Lightweight video frame player with scrubber slider.

    Inputs an IMAGE batch (B,H,W,C). The widget shows a timeline; the
    `frame_index` widget controls which frame is emitted as the IMAGE output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Frame batch (B,H,W,C). B=number of frames."}),
                "frame_index": ("INT", {
                    "default": 0, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Current frame to emit on the IMAGE output. Drag the slider in the widget to scrub."}),
                "preview_width": ("INT", {
                    "default": 480, "min": 96, "max": 1920, "step": 16,
                    "tooltip": "Width (px) of preview JPEGs sent to the browser. Lower = lighter."}),
                "preview_quality": ("INT", {
                    "default": _PREVIEW_QUALITY, "min": 30, "max": 95, "step": 5,
                    "tooltip": "JPEG quality for previews."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("frame", "frame_index", "frame_count")
    OUTPUT_TOOLTIPS = (
        "Single full-resolution frame at the slider position (1,H,W,C).",
        "Echo of the selected frame_index (clamped to range).",
        "Total number of frames in the input batch.",
    )
    FUNCTION = "play"
    CATEGORY = "MaskEditControl/Preview"
    OUTPUT_NODE = True
    DESCRIPTION = "Lightweight in-graph video scrubber. Drag the timeline to scrub frames; selected frame is emitted as IMAGE."

    def play(
        self,
        frames: torch.Tensor,
        frame_index: int = 0,
        preview_width: int = 480,
        preview_quality: int = _PREVIEW_QUALITY,
    ):
        if frames is None or not hasattr(frames, "shape"):
            raise ValueError("VideoFramePlayerMEC: 'frames' input is missing or invalid (expected IMAGE B,H,W,C).")
        if frames.dim() != 4:
            raise ValueError(f"VideoFramePlayerMEC: expected 4D IMAGE (B,H,W,C), got shape {tuple(frames.shape)}.")

        B, H, W, C = frames.shape
        idx = max(0, min(int(frame_index), B - 1))

        # Compute preview thumbnails (small JPEGs, single hash key per batch)
        # Hash on shape + first/last/mid frames so identical batches reuse cache.
        digest_src = (
            f"{B}x{H}x{W}x{C}_{preview_width}_{preview_quality}_"
        ).encode()
        for sample_idx in {0, B // 2, B - 1}:
            try:
                sample = (frames[sample_idx, :8, :8, :].detach().cpu().numpy() * 255).astype(np.uint8)
                digest_src += sample.tobytes()
            except Exception:
                pass
        batch_id = hashlib.sha256(digest_src).hexdigest()[:12]

        temp_dir = folder_paths.get_temp_directory()
        previews = []
        thumb_h = max(1, int(round(H * preview_width / max(W, 1))))
        for i in _PB.track(range(B), B, "FramePlayer"):
            _IC.check()
            name = f"vfp_{batch_id}_{i:05d}.jpg"
            path = os.path.join(temp_dir, name)
            if not os.path.exists(path):
                arr = (frames[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                if pil.size[0] != preview_width:
                    pil = pil.resize((preview_width, thumb_h), Image.BILINEAR)
                pil.save(path, "JPEG", quality=int(preview_quality), optimize=False)
            previews.append({"filename": name, "subfolder": "", "type": "temp"})

        out_frame = frames[idx:idx + 1].clone()

        ui_payload = {
            "frames": previews,
            "frame_count": [B],
            "current_index": [idx],
            "width": [W],
            "height": [H],
        }

        return {
            "ui": ui_payload,
            "result": (out_frame, idx, B),
        }
