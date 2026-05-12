"""Salient / background-removal segmenter backends via HF transformers.

Wraps BiRefNet, RMBG-2.0 and InSPyReNet under a common interface so they
become first-class options in the unified ``MaskMattingMEC`` node.

All three follow the same recipe:
  * lazy import of ``transformers`` + ``Pillow`` so a missing dep doesn't
    break the pack;
  * ``AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)``
    with weights resolved through ``folder_paths`` first
    (``models/<KEY>/``), HF Hub second when ``auto_download`` is on;
  * single-image inference at the model's native input size, with
    interruptible iteration across the batch.

Hollywood VFX note: these are SALIENT-object models (no prompts). They
shine for hero-object cleanup, fast plate prep, and as a second opinion
for trimap-from-uncertainty fusion in the production pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import (
    backend_first_root,
    free_vram,
    interruptible_range,
    list_backend_files,
)
from . import BaseSegmenter, register

logger = logging.getLogger("MEC.MaskMatting.Salient")


def _have_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


def _model_path_or_repo(models_key: str, model_name: str, default_repo: str) -> str:
    """Prefer a folder under ``models/<key>/<model_name>`` if present, else HF repo."""
    if model_name:
        root = backend_first_root(models_key)
        if root:
            cand = os.path.join(root, model_name)
            if os.path.isdir(cand):
                return cand
            cand2 = os.path.join(root, os.path.splitext(model_name)[0])
            if os.path.isdir(cand2):
                return cand2
    return default_repo


def _to_pil(img_hwc: np.ndarray):
    from PIL import Image
    arr = (img_hwc * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _normalize_imagenet(pil_img, size: int):
    """Resize → normalize for ImageNet-style salient models."""
    from torchvision import transforms  # imported lazily, ComfyUI ships it
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(pil_img)


class _SalientBase(BaseSegmenter):
    """Shared inference plumbing for the three HF-style salient models."""

    SUPPORTS_MODES = {"auto"}
    STATUS = "ready" if _have_transformers() else "missing-deps"
    _DEFAULT_REPO: str = ""
    _INPUT_SIZE: int = 1024
    _TRUST_REMOTE_CODE: bool = True

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForImageSegmentation
        path = _model_path_or_repo(self.MODELS_KEY, self.model_name, self._DEFAULT_REPO)
        logger.warning("[%s] loading %s", self.KEY, path)
        dtype = torch.float16 if self.precision == "fp16" else (
            torch.bfloat16 if self.precision == "bf16" else torch.float32)
        m = AutoModelForImageSegmentation.from_pretrained(
            path, trust_remote_code=self._TRUST_REMOTE_CODE
        )
        try:
            m = m.to(self.device).to(dtype)
        except Exception:
            m = m.to(self.device)
        m.eval()
        self._model = m
        self._dtype = dtype

    @torch.no_grad()
    def _infer_one(self, img_hwc: np.ndarray) -> np.ndarray:
        from PIL import Image
        pil = _to_pil(img_hwc)
        H0, W0 = img_hwc.shape[:2]
        x = _normalize_imagenet(pil, self._INPUT_SIZE).unsqueeze(0).to(self.device)
        try:
            x = x.to(self._dtype)
        except Exception:
            pass
        out = self._model(x)
        # Different HF wrappers return different shapes — normalize them.
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if hasattr(out, "logits"):
            out = out.logits
        if isinstance(out, list):
            out = out[-1]
        if out.ndim == 4 and out.shape[1] > 1:
            out = out[:, :1]
        prob = torch.sigmoid(out.float())
        prob = F.interpolate(prob, size=(H0, W0), mode="bilinear", align_corners=False)
        return prob.squeeze().cpu().numpy().astype(np.float32)

    def segment(self, image_bhwc: torch.Tensor, **kwargs) -> Dict[str, Any]:
        try:
            self.load()
            arr = image_bhwc.detach().cpu().numpy()
            B = arr.shape[0]
            masks = np.empty((B, arr.shape[1], arr.shape[2]), dtype=np.float32)
            for i in interruptible_range(B, label=f"{self.KEY} frame"):
                masks[i] = self._infer_one(arr[i])
            mask_t = torch.from_numpy(masks).clamp(0.0, 1.0)
            score = float(mask_t.mean())  # proxy
            return {
                "mask": mask_t,
                "score": score,
                "info": {"backend": self.KEY, "frames": int(B)},
            }
        finally:
            free_vram(unload_models=False)


# ──────────────────────────────────────────────────────────────────────
# BiRefNet — Bilateral Reference for high-res dichotomous segmentation.
# ──────────────────────────────────────────────────────────────────────
@register
class BiRefNetSegmenter(_SalientBase):
    KEY = "birefnet"
    DISPLAY = "BiRefNet (high-res salient)"
    MODELS_KEY = "birefnet"
    _DEFAULT_REPO = "ZhengPeng7/BiRefNet"
    _INPUT_SIZE = 1024


# ──────────────────────────────────────────────────────────────────────
# RMBG-2.0 — Bria's production background-removal model.
# ──────────────────────────────────────────────────────────────────────
@register
class RMBG2Segmenter(_SalientBase):
    KEY = "rmbg"
    DISPLAY = "RMBG-2.0 (BG removal)"
    MODELS_KEY = "rmbg"
    _DEFAULT_REPO = "briaai/RMBG-2.0"
    _INPUT_SIZE = 1024


# ──────────────────────────────────────────────────────────────────────
# InSPyReNet — Inverse-Saliency Pyramid Reconstruction.
# ──────────────────────────────────────────────────────────────────────
@register
class InSPyReNetSegmenter(_SalientBase):
    KEY = "inspyrenet"
    DISPLAY = "InSPyReNet (high-res salient)"
    MODELS_KEY = "inspyrenet"
    _DEFAULT_REPO = "plemeri/InSPyReNet_SwinB"
    _INPUT_SIZE = 1024
