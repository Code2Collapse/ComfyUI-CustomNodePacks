"""RVM (Robust Video Matting) matter — fast realtime alpha for video.

Uses the user-installed ``rvm`` checkpoint (.pth). Reference repo:
https://github.com/PeterL1n/RobustVideoMatting

Recurrent state is preserved across frames for temporal stability.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import (
    backend_first_root,
    free_vram,
    interruptible_range,
    resolve_backend_weight,
    to_bhwc,
    to_mask,
)
from . import BaseMatter, register

logger = logging.getLogger("MEC.MaskMatting.RVM")


def _have_rvm_repo() -> bool:
    """RVM is installed via Torch Hub or a vendored ``model`` module."""
    try:
        import importlib
        importlib.util.find_spec("model")
        return True
    except Exception:
        # Torch hub fallback always available
        return True


@register
class RVMMatter(BaseMatter):
    KEY = "rvm"
    DISPLAY = "RVM (Robust Video Matting)"
    MODELS_KEY = "rvm"
    NEEDS_TRIMAP = False
    STATUS = "ready"

    def load(self) -> None:
        if self._model is not None:
            return
        ckpt = resolve_backend_weight(self.MODELS_KEY, self.model_name)
        # Choose architecture from filename: rvm_mobilenetv3 / rvm_resnet50.
        stem = os.path.splitext(os.path.basename(self.model_name or ""))[0].lower()
        variant = "mobilenetv3" if "mobilenet" in stem else "resnet50"
        try:
            model = torch.hub.load("PeterL1n/RobustVideoMatting", variant, pretrained=(ckpt is None))
            if ckpt is not None and os.path.isfile(ckpt):
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state, strict=False)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load RVM (variant={variant}). Place a checkpoint in "
                f"{backend_first_root(self.MODELS_KEY)} or ensure internet access. "
                f"Underlying error: {exc}"
            )
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(self.precision, torch.float16)
        self._model = model.to(self.device, dtype=dtype).eval()
        self._dtype = dtype

    def matte(self, image_bhwc, coarse_mask, *, trimap=None, edge_radius=4, memory_size=8):
        try:
            self.load()
            img = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape
            # RVM expects (B,3,H,W). It does NOT consume coarse_mask but we
            # gate the output by it so the user's prompt actually constrains
            # the result — typical use is "RVM after a points-driven SAM".
            x = img.permute(0, 3, 1, 2).to(self.device, dtype=self._dtype)
            rec = [None] * 4   # recurrent state
            outs = []
            with torch.inference_mode():
                for i in interruptible_range(B, label="rvm"):
                    frame = x[i:i + 1]
                    fgr, pha, *rec = self._model(frame, *rec, downsample_ratio=0.4)
                    outs.append(pha[0, 0].float().cpu())
            alpha = torch.stack(outs, 0).clamp(0, 1)
            # Optional gating by coarse_mask so user prompts still bound it.
            if coarse_mask is not None:
                m = to_mask(coarse_mask)
                if m.shape == alpha.shape:
                    # take min so RVM never exceeds the segmentation envelope
                    # (dilated slightly to keep hair detail)
                    grow = F.max_pool2d(m.unsqueeze(1), 9, 1, 4).squeeze(1)
                    alpha = torch.minimum(alpha, grow)
            return {"alpha": alpha, "info": {"backend": self.KEY, "variant": stem or "auto"}}
        except Exception:
            free_vram()
            raise
