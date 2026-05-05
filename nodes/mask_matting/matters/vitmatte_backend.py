"""ViTMatte matter backend — uses HuggingFace ``transformers``.

Loads ViTMatte weights from ``ComfyUI/models/vitmatte/``. If the user only
has a HF model id (e.g. ``hustvl/vitmatte-small-composition-1k``), this
backend will pull from the HF cache transparently.
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
    list_backend_files,
    mask_to_trimap,
    resolve_backend_weight,
    to_bhwc,
    to_mask,
)
from . import BaseMatter, register

logger = logging.getLogger("MEC.MaskMatting.ViTMatte")


def _have_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


@register
class ViTMatteMatter(BaseMatter):
    KEY = "vitmatte"
    DISPLAY = "ViTMatte"
    MODELS_KEY = "vitmatte"
    NEEDS_TRIMAP = True
    STATUS = "ready" if _have_transformers() else "missing-deps"

    def load(self) -> None:
        if self._model is not None:
            return
        if not _have_transformers():
            raise RuntimeError("transformers not installed. `pip install transformers`.")
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        path = resolve_backend_weight(self.MODELS_KEY, self.model_name)
        if path and os.path.isdir(path):
            src = path
        elif path and os.path.isfile(path):
            # User supplied a single .safetensors — assume sibling config.json
            src = os.path.dirname(path)
        else:
            # Fallback: treat model_name as HF repo id.
            src = self.model_name or "hustvl/vitmatte-small-composition-1k"
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(self.precision, torch.float16)
        self._processor = VitMatteImageProcessor.from_pretrained(src)
        self._model = VitMatteForImageMatting.from_pretrained(src, torch_dtype=dtype).to(self.device).eval()
        self._dtype = dtype

    def matte(self, image_bhwc, coarse_mask, *, trimap=None, edge_radius=4, memory_size=8):
        try:
            self.load()
            img = to_bhwc(image_bhwc)
            B, H, W, _ = img.shape
            if trimap is None:
                trimap = mask_to_trimap(coarse_mask, dilate=int(edge_radius) * 2, erode=int(edge_radius))
            trimap = to_mask(trimap)

            alphas = []
            for i in interruptible_range(B, label="vitmatte"):
                frame = (img[i].cpu().numpy() * 255).astype(np.uint8)
                tri_np = (trimap[i].cpu().numpy() * 255).astype(np.uint8)
                inputs = self._processor(images=frame, trimaps=tri_np, return_tensors="pt")
                inputs = {k: v.to(self.device, dtype=self._dtype if v.dtype.is_floating_point else v.dtype) for k, v in inputs.items()}
                with torch.inference_mode(), torch.autocast(self.device, dtype=self._dtype, enabled=(self.device == "cuda")):
                    out = self._model(**inputs)
                alpha = out.alphas
                # Resize to original
                alpha = F.interpolate(alpha, size=(H, W), mode="bilinear", align_corners=False)
                alphas.append(alpha[0, 0].float().cpu())
            alpha_t = torch.stack(alphas, 0).clamp(0, 1)
            return {"alpha": alpha_t, "info": {"backend": self.KEY}}
        except Exception:
            free_vram()
            raise
