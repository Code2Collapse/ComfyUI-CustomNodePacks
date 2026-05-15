"""WanFrameBridgeMEC — last-frame extractor for inter-shot continuity."""
from __future__ import annotations

import torch

from ._common import log


class WanFrameBridgeMEC:
    """
    Extract a single bridge frame from a generated clip, intended to be fed
    back as the start image of the next shot to preserve visual continuity.

    By default returns the LAST frame. ``offset`` lets you pick a frame
    N positions before the end (clamped to range). When ``mode`` is "average",
    a temporal mean of the last ``avg_frames`` is returned to soften motion
    that would otherwise create a hard cut.
    """
    CATEGORY = "C2C/wan_director"
    FUNCTION = "extract"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("bridge_image",)
    DESCRIPTION = (
        "Pick a frame from the end of a clip to use as the start-image of "
        "the next shot, for visual continuity. Modes: last (default), "
        "offset (N before end), average (mean of last N)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":     ("IMAGE",),
                "mode":       (["last", "offset", "average"], {"default": "last"}),
                "offset":     ("INT",   {"default": 0,  "min": 0,  "max": 256, "step": 1}),
                "avg_frames": ("INT",   {"default": 3,  "min": 1,  "max": 32,  "step": 1}),
            },
        }

    def extract(self, images: torch.Tensor, mode: str, offset: int, avg_frames: int):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(
                f"WanFrameBridge expects IMAGE [B,H,W,C]; got {type(images).__name__}"
            )
        b = images.shape[0]
        if b == 0:
            raise ValueError("WanFrameBridge: empty image batch")
        if mode == "average":
            n = max(1, min(int(avg_frames), b))
            frame = images[b - n: b].mean(dim=0, keepdim=True)
        else:
            idx = b - 1 - max(0, int(offset)) if mode == "offset" else b - 1
            idx = max(0, min(b - 1, idx))
            frame = images[idx: idx + 1].clone()
        frame = frame.clamp(0.0, 1.0).contiguous()
        log.info("[WanFrameBridge] mode=%s -> shape=%s", mode, tuple(frame.shape))
        return (frame,)
