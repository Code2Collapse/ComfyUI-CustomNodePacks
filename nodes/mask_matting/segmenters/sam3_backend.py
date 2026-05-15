"""SAM 3 segmenter backend (text + points + bbox).

SAM 3 (Meta, 2026) extends SAM 2 with native text prompting. We import the
public ``sam3`` package lazily; when absent the backend reports
``missing-deps`` and the combo entry shows a [missing-deps] badge.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..utils import (
    backend_first_root,
    free_vram,
    interruptible_range,
    resolve_backend_weight,
)
from . import BaseSegmenter, register

logger = logging.getLogger("MEC.MaskMatting.SAM3")


def _have_sam3() -> bool:
    try:
        import sam3  # noqa: F401
        return True
    except ImportError:
        return False


@register
class SAM3Segmenter(BaseSegmenter):
    KEY = "sam3"
    DISPLAY = "SAM 3 (text + points + bbox)"
    MODELS_KEY = "sam3"
    SUPPORTS_MODES = {"points", "bbox", "text", "auto"}
    STATUS = "ready" if _have_sam3() else "missing-deps"

    def load(self) -> None:
        if self._model is not None:
            return
        if not _have_sam3():
            raise RuntimeError("sam3 package not installed.")
        from sam3.build_sam3 import build_sam3
        from sam3.sam3_image_predictor import SAM3ImagePredictor
        ckpt = resolve_backend_weight(self.MODELS_KEY, self.model_name)
        if ckpt is None or not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"SAM3 checkpoint '{self.model_name}' not under {backend_first_root(self.MODELS_KEY)}."
            )
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(self.precision, torch.float16)
        self._dtype = dtype
        with torch.inference_mode():
            self._image_model = build_sam3(ckpt, device=self.device)
            self._predictor = SAM3ImagePredictor(self._image_model)
        self._model = self._predictor

    def _seg_one(self, frame_hwc: np.ndarray, pos, neg, bbox, neg_bbox, text):
        img_u8 = (frame_hwc * 255).clip(0, 255).astype(np.uint8)
        with torch.inference_mode(), torch.autocast(self.device, dtype=self._dtype, enabled=(self.device == "cuda")):
            self._predictor.set_image(img_u8)

            def _predict(_pos, _neg, _bbox, _text):
                kw = {"multimask_output": True}
                if _pos or _neg:
                    kw["point_coords"] = np.array(_pos + _neg, dtype=np.float32)
                    kw["point_labels"] = np.array([1] * len(_pos) + [0] * len(_neg), dtype=np.int64)
                if _bbox is not None:
                    kw["box"] = np.array(_bbox, dtype=np.float32)
                if _text:
                    kw["text"] = _text
                m, sc, _ = self._predictor.predict(**kw)
                best = int(np.argmax(sc))
                return m[best].astype(np.float32), float(sc[best])

            # Positive pass.
            mask_pos, score = _predict(pos, neg, bbox, text)

            # Negative bbox: SAM 3's reference image predictor does not expose
            # labeled boxes — subtract a separate "what's inside neg_bbox"
            # mask. Cheap (already on GPU, same set_image) and keeps the
            # public API symmetric with SAM 3.1.
            if neg_bbox is not None:
                try:
                    mask_neg, _ = _predict([], [], neg_bbox, "")
                    mask_pos = np.clip(mask_pos - mask_neg, 0.0, 1.0)
                except Exception as exc:
                    logger.warning("[SAM3] neg_bbox subtraction skipped: %s", exc)

        return mask_pos, score

    def segment(self, image_bhwc, *, mode="auto",
                positive_points=None, negative_points=None,
                bbox=None, neg_bbox=None, text_prompt="", frame_annotation=0,
                object_id=0, max_frames=0, memory_size=8,
                start_frame=0, end_frame=-1, individual_objects=False,
                tracking_direction="forward", seed=0):
        try:
            self.load()
            B = image_bhwc.shape[0]
            pos = positive_points or []
            neg = negative_points or []
            outs = []
            score = 0.0
            for i in interruptible_range(B, label="sam3"):
                m, s = self._seg_one(image_bhwc[i].cpu().numpy(), pos, neg, bbox, neg_bbox, text_prompt)
                outs.append(m)
                score = s
            mask_t = torch.from_numpy(np.stack(outs, 0))
            return {"mask": mask_t.float(), "score": float(score), "info": {"backend": self.KEY}}
        except Exception:
            free_vram()
            raise
