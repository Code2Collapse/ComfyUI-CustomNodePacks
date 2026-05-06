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

        # ── Resolve a usable HF model directory ──────────────────────────
        # Acceptable layouts under ComfyUI/models/vitmatte/:
        #   * <root>/<repo-name>/{preprocessor_config.json, config.json, *.safetensors}
        #   * <root>/{preprocessor_config.json, config.json, *.safetensors}    (flat)
        #   * a single .safetensors file with a sibling preprocessor_config.json
        # Anything else → fall back to the HF Hub repo id so the user gets a
        # real model instead of OSError("Can't load image processor for ...").
        def _is_hf_model_dir(p: str) -> bool:
            return (
                isinstance(p, str)
                and os.path.isdir(p)
                and os.path.isfile(os.path.join(p, "preprocessor_config.json"))
                and os.path.isfile(os.path.join(p, "config.json"))
            )

        candidates: list[str] = []
        path = resolve_backend_weight(self.MODELS_KEY, self.model_name) if self.model_name else None
        if path and os.path.isdir(path):
            candidates.append(path)
        elif path and os.path.isfile(path):
            candidates.append(os.path.dirname(path))

        # Walk the backend root to find any HF-style sub-folder the user
        # might have downloaded (e.g. via huggingface-cli or git-lfs).
        try:
            root = backend_first_root(self.MODELS_KEY)
            if root and os.path.isdir(root):
                if _is_hf_model_dir(root):
                    candidates.append(root)
                for entry in sorted(os.listdir(root)):
                    sub = os.path.join(root, entry)
                    if _is_hf_model_dir(sub):
                        candidates.append(sub)
        except Exception:
            pass

        src: Optional[str] = None
        for c in candidates:
            if _is_hf_model_dir(c):
                src = c
                break

        if src is None:
            # No usable local layout → use HF id (or a sensible default).
            src = self.model_name or "hustvl/vitmatte-small-composition-1k"
            # If the resolved name still looks like a bare filename (no '/'),
            # prepend the canonical HF org so transformers can fetch it.
            if "/" not in src and not os.path.isabs(src):
                src = "hustvl/vitmatte-small-composition-1k"
            logger.info(
                "[ViTMatte] no local HF model dir found under "
                "ComfyUI/models/vitmatte/ \u2014 falling back to HF Hub: %s", src,
            )

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
