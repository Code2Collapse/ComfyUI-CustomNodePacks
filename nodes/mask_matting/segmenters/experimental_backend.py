"""Newly-wired segmenter backends.

These were previously listed in ``stubs.py`` as ``NotImplementedError``
placeholders. Promoted 2026-05-12:

  * ``dis``           – DIS / IS-Net high-res salient via ``briaai/RMBG-1.4``
                         (Bria's open IS-Net weights, ApacheL2 ✓).
  * ``person-mask``   – Ultralytics YOLOv8-seg person-only mask.
  * ``grounding-dino``– IDEA-Research/grounding-dino-tiny text→bbox; the
                         bbox is rasterised into a binary MASK so downstream
                         nodes can consume a normal MASK output (or feed it
                         into SAM2/SAM3 as a bbox prompt).

All three follow the existing ``BaseSegmenter`` contract: ``load()``,
``segment(image_bhwc, ...) -> {mask, score, info}``. Optional deps are
imported lazily and ``STATUS`` reflects whether the import succeeds.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import backend_first_root, free_vram, interruptible_range
from . import BaseSegmenter, register

logger = logging.getLogger("MEC.MaskMatting.Experimental")


# ──────────────────────────────────────────────────────────────────────
# Optional-dep probes
# ──────────────────────────────────────────────────────────────────────
def _have_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except Exception:
        return False


def _have_ultralytics() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def _to_pil(img_hwc: np.ndarray):
    from PIL import Image
    return Image.fromarray((img_hwc * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")


def _model_path_or_repo(models_key: str, model_name: str, default_repo: str) -> str:
    if model_name:
        root = backend_first_root(models_key)
        if root:
            for cand in (os.path.join(root, model_name),
                         os.path.join(root, os.path.splitext(model_name)[0])):
                if os.path.isdir(cand) or os.path.isfile(cand):
                    return cand
    return default_repo


# ══════════════════════════════════════════════════════════════════════
# DIS / IS-Net (high-res salient)
# ══════════════════════════════════════════════════════════════════════
@register
class DISSegmenter(BaseSegmenter):
    """High-res salient object segmentation via Bria's IS-Net weights.

    IS-Net (Qin et al. 2022) is the architecture behind the DIS5K
    benchmark; ``briaai/RMBG-1.4`` ships exactly those weights with a
    permissive licence and HF ``AutoModelForImageSegmentation`` wrapper.
    """
    KEY = "dis"
    DISPLAY = "DIS / IS-Net (high-res salient)"
    MODELS_KEY = "dis"
    SUPPORTS_MODES = {"auto"}
    STATUS = "ready" if _have_transformers() else "missing-deps"
    _DEFAULT_REPO = "briaai/RMBG-1.4"
    _INPUT_SIZE = 1024

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForImageSegmentation
        path = _model_path_or_repo(self.MODELS_KEY, self.model_name, self._DEFAULT_REPO)
        logger.warning("[%s] loading %s", self.KEY, path)
        dtype = (torch.float16 if self.precision == "fp16" else
                 torch.bfloat16 if self.precision == "bf16" else torch.float32)
        m = AutoModelForImageSegmentation.from_pretrained(path, trust_remote_code=True)
        try:
            m = m.to(self.device).to(dtype)
        except Exception:
            m = m.to(self.device)
        m.eval()
        self._model = m
        self._dtype = dtype

    @torch.no_grad()
    def _infer_one(self, img_hwc: np.ndarray) -> np.ndarray:
        from torchvision import transforms
        H0, W0 = img_hwc.shape[:2]
        pil = _to_pil(img_hwc)
        tfm = transforms.Compose([
            transforms.Resize((self._INPUT_SIZE, self._INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),  # IS-Net normalisation
        ])
        x = tfm(pil).unsqueeze(0).to(self.device)
        try:
            x = x.to(self._dtype)
        except Exception:
            pass
        out = self._model(x)
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

    def segment(self, image_bhwc: torch.Tensor, **kw) -> Dict[str, Any]:
        try:
            self.load()
            arr = image_bhwc.detach().cpu().numpy()
            B = arr.shape[0]
            masks = np.empty((B, arr.shape[1], arr.shape[2]), dtype=np.float32)
            for i in interruptible_range(B, label="dis frame"):
                masks[i] = self._infer_one(arr[i])
            mask_t = torch.from_numpy(masks).clamp(0.0, 1.0)
            return {"mask": mask_t,
                    "score": float(mask_t.mean()),
                    "info": {"backend": "dis", "frames": int(B),
                             "input_size": self._INPUT_SIZE}}
        finally:
            free_vram(unload_models=False)


# ══════════════════════════════════════════════════════════════════════
# PersonMask via Ultralytics YOLOv8-seg (class 0 = person)
# ══════════════════════════════════════════════════════════════════════
@register
class PersonMaskSegmenter(BaseSegmenter):
    """Quick person-only segmentation via Ultralytics YOLOv8-seg.

    Uses the COCO class 0 (``person``). For multi-person frames every
    detected person mask is OR-fused into the single output mask. Default
    weights ``yolov8n-seg.pt`` (~7 MB) auto-download on first use; users
    can pass ``yolov8m-seg.pt`` / ``yolov8x-seg.pt`` via ``model_name``
    for higher accuracy.
    """
    KEY = "person-mask"
    DISPLAY = "PersonMask (YOLOv8-seg, person)"
    MODELS_KEY = "ultralytics_segm"
    SUPPORTS_MODES = {"auto"}
    STATUS = "ready" if _have_ultralytics() else "missing-deps"
    _DEFAULT_WEIGHTS = "yolov8n-seg.pt"
    _PERSON_CLASS = 0  # COCO

    def load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO
        weights = self.model_name or self._DEFAULT_WEIGHTS
        path = _model_path_or_repo(self.MODELS_KEY, weights, weights)
        logger.warning("[%s] loading %s", self.KEY, path)
        self._model = YOLO(path)

    @torch.no_grad()
    def _infer_one(self, img_hwc: np.ndarray) -> np.ndarray:
        H, W = img_hwc.shape[:2]
        bgr = (img_hwc[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        result = self._model.predict(bgr, classes=[self._PERSON_CLASS],
                                     verbose=False, device=self.device)[0]
        if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
            return np.zeros((H, W), dtype=np.float32)
        masks = result.masks.data.detach().cpu().numpy().astype(np.float32)
        # Ultralytics returns masks at model resolution; reshape to source.
        if masks.shape[1] != H or masks.shape[2] != W:
            mt = torch.from_numpy(masks).unsqueeze(1)
            mt = F.interpolate(mt, size=(H, W), mode="bilinear", align_corners=False)
            masks = mt.squeeze(1).numpy()
        # OR-fuse all detected persons.
        fused = np.clip(masks.max(axis=0), 0.0, 1.0)
        return fused.astype(np.float32)

    def segment(self, image_bhwc: torch.Tensor, **kw) -> Dict[str, Any]:
        try:
            self.load()
            arr = image_bhwc.detach().cpu().numpy()
            B, H, W = arr.shape[:3]
            masks = np.empty((B, H, W), dtype=np.float32)
            for i in interruptible_range(B, label="person-mask frame"):
                masks[i] = self._infer_one(arr[i])
            mask_t = torch.from_numpy(masks).clamp(0.0, 1.0)
            return {"mask": mask_t,
                    "score": float(mask_t.mean()),
                    "info": {"backend": "person-mask",
                             "weights": self.model_name or self._DEFAULT_WEIGHTS,
                             "frames": int(B)}}
        finally:
            free_vram(unload_models=False)


# ══════════════════════════════════════════════════════════════════════
# Grounding-DINO (text→bbox→binary mask)
# ══════════════════════════════════════════════════════════════════════
@register
class GroundingDinoSegmenter(BaseSegmenter):
    """Open-vocabulary detector. ``text_prompt`` selects the target class.

    Returns a binary MASK rasterised from the best-scoring bbox (per
    frame). For pixel-accurate masks, feed the bbox into a SAM backend
    via the unified MaskMatting node's "bbox → SAM" routing — but this
    backend alone is already enough for crop/cleanup workflows.
    """
    KEY = "grounding-dino"
    DISPLAY = "Grounding-DINO (text → bbox mask)"
    MODELS_KEY = "grounding-dino"
    SUPPORTS_MODES = {"text"}
    STATUS = "ready" if _have_transformers() else "missing-deps"
    _DEFAULT_REPO = "IDEA-Research/grounding-dino-tiny"

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
        )
        path = _model_path_or_repo(self.MODELS_KEY, self.model_name, self._DEFAULT_REPO)
        logger.warning("[%s] loading %s", self.KEY, path)
        self._processor = AutoProcessor.from_pretrained(path)
        m = AutoModelForZeroShotObjectDetection.from_pretrained(path)
        m = m.to(self.device).eval()
        self._model = m

    @staticmethod
    def _normalise_prompt(text: str) -> str:
        """G-DINO expects a lowercase period-terminated string per class."""
        t = (text or "").strip().lower()
        if not t:
            return ""
        # Multiple classes separated by '.' or ','.
        parts = [p.strip() for p in t.replace(",", ".").split(".") if p.strip()]
        return ". ".join(parts) + "."

    @torch.no_grad()
    def _infer_one(self, img_hwc: np.ndarray, prompt: str,
                   box_thresh: float, text_thresh: float
                   ) -> Tuple[np.ndarray, float, Optional[Tuple[int, int, int, int]]]:
        H, W = img_hwc.shape[:2]
        pil = _to_pil(img_hwc)
        inputs = self._processor(images=pil, text=prompt,
                                 return_tensors="pt").to(self.device)
        out = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            out, inputs.input_ids,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=[(H, W)],
        )
        r = results[0] if results else None
        mask = np.zeros((H, W), dtype=np.float32)
        if not r or len(r.get("boxes", [])) == 0:
            return mask, 0.0, None
        boxes = r["boxes"].detach().cpu().numpy()
        scores = r["scores"].detach().cpu().numpy()
        best = int(np.argmax(scores))
        x1, y1, x2, y2 = boxes[best].astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
        return mask, float(scores[best]), (int(x1), int(y1), int(x2), int(y2))

    def segment(self, image_bhwc: torch.Tensor, *,
                text_prompt: str = "", **kw) -> Dict[str, Any]:
        try:
            self.load()
            prompt = self._normalise_prompt(text_prompt)
            if not prompt:
                raise ValueError("Grounding-DINO requires a non-empty text_prompt "
                                 "(e.g. 'person.' or 'red car. dog.').")
            arr = image_bhwc.detach().cpu().numpy()
            B, H, W = arr.shape[:3]
            masks = np.empty((B, H, W), dtype=np.float32)
            scores = []
            bboxes = []
            box_t = float(kw.get("box_threshold", 0.30))
            text_t = float(kw.get("text_threshold", 0.25))
            for i in interruptible_range(B, label="grounding-dino frame"):
                m, s, bb = self._infer_one(arr[i], prompt, box_t, text_t)
                masks[i] = m
                scores.append(s)
                bboxes.append(bb)
            mask_t = torch.from_numpy(masks).clamp(0.0, 1.0)
            mean_score = float(np.mean(scores)) if scores else 0.0
            return {"mask": mask_t,
                    "score": mean_score,
                    "info": {"backend": "grounding-dino",
                             "prompt": prompt,
                             "frames": int(B),
                             "bboxes": bboxes,
                             "box_threshold": box_t,
                             "text_threshold": text_t}}
        finally:
            free_vram(unload_models=False)
