"""WanShotConcatMEC — concatenate up to 6 IMAGE clips into one timeline.

Supports an optional linear crossfade transition that overlaps the tail
of clip N with the head of clip N+1 over ``crossfade_frames`` frames.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ._common import log


def _match_hw(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize a [B,H,W,C] tensor to (h, w) via bilinear if needed."""
    if t.shape[1] == h and t.shape[2] == w:
        return t
    # [B,H,W,C] -> [B,C,H,W] for interpolate
    x = t.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


def _crossfade(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """Linear alpha-blend the last N frames of `a` with the first N of `b`.

    Returns a [N,H,W,C] tensor; alpha runs 0->1 across the window.
    Caller is responsible for slicing the non-overlapping prefix of `a`
    and suffix of `b` themselves.
    """
    alpha = torch.linspace(0.0, 1.0, steps=n, device=a.device, dtype=a.dtype)
    alpha = alpha.view(n, 1, 1, 1)
    return (a * (1.0 - alpha) + b * alpha).clamp(0.0, 1.0)


class WanShotConcatMEC:
    """
    Concatenate up to 6 IMAGE batches (clips) into a single timeline.

    All clips are bilinearly resized to the spatial size of the FIRST
    non-empty clip. ``crossfade_frames`` (0..64) controls a linear
    transition between adjacent clips. With 0, the result is a plain
    temporal concat; with N>0 each junction overlaps N frames so the
    final length is ``sum(lengths) - (k-1)*N`` for k clips.
    """
    CATEGORY = "C2C/wan_director"
    FUNCTION = "concat"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "num_frames")
    DESCRIPTION = (
        "Stitch up to 6 IMAGE clips into one timeline. Optional linear "
        "crossfade overlaps adjacent clips by N frames. All clips are "
        "auto-resized to the first clip's dimensions."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crossfade_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 64, "step": 1,
                     "tooltip": "Frames to crossfade at every junction."},
                ),
            },
            "optional": {
                "clip_1": ("IMAGE",),
                "clip_2": ("IMAGE",),
                "clip_3": ("IMAGE",),
                "clip_4": ("IMAGE",),
                "clip_5": ("IMAGE",),
                "clip_6": ("IMAGE",),
            },
        }

    def concat(self, crossfade_frames: int, **kwargs) -> tuple:
        clips: list[torch.Tensor] = []
        for i in range(1, 7):
            c = kwargs.get(f"clip_{i}")
            if isinstance(c, torch.Tensor) and c.ndim == 4 and c.shape[0] > 0:
                clips.append(c)
        if not clips:
            raise ValueError("WanShotConcat: no input clips provided")
        # Use first clip's spatial size as the target.
        _, h, w, _ = clips[0].shape
        clips = [_match_hw(c.to(clips[0].dtype), h, w) for c in clips]

        n = max(0, int(crossfade_frames))
        if n == 0 or len(clips) == 1:
            out = torch.cat(clips, dim=0)
            log.info("[WanShotConcat] no-xfade: %d clips -> %d frames",
                     len(clips), out.shape[0])
            return (out.contiguous(), int(out.shape[0]))

        # Crossfade path.
        pieces: list[torch.Tensor] = []
        for i, clip in enumerate(clips):
            cf = min(n, clip.shape[0] - 1) if 0 < i < len(clips) else n
            cf = max(0, min(cf, clip.shape[0]))
            if i == 0:
                # head + (tail that will be blended)
                cut = max(0, clip.shape[0] - cf) if cf > 0 and len(clips) > 1 else clip.shape[0]
                pieces.append(clip[:cut])
                if cf > 0 and len(clips) > 1:
                    prev_tail = clip[cut:]
                else:
                    prev_tail = None
            else:
                cf_eff = min(n, clip.shape[0], (prev_tail.shape[0] if prev_tail is not None else 0))
                if cf_eff > 0 and prev_tail is not None:
                    blended = _crossfade(prev_tail[:cf_eff], clip[:cf_eff], cf_eff)
                    pieces.append(blended)
                    middle_start = cf_eff
                else:
                    middle_start = 0
                if i == len(clips) - 1:
                    pieces.append(clip[middle_start:])
                    prev_tail = None
                else:
                    cut = max(middle_start, clip.shape[0] - n)
                    pieces.append(clip[middle_start:cut])
                    prev_tail = clip[cut:]
        out = torch.cat([p for p in pieces if p.numel() > 0], dim=0).contiguous()
        log.info("[WanShotConcat] xfade=%d: %d clips -> %d frames",
                 n, len(clips), out.shape[0])
        return (out, int(out.shape[0]))
