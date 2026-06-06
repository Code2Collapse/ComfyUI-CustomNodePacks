"""LocateAnything-3B grounding segmenter backend.

Wraps NVIDIA's LocateAnything-3B as a text-grounding segmenter that produces
bounding-box masks from natural-language prompts. This backend excels at
open-vocabulary object localization and serves as a grounding identifier
for downstream SAM-based refinement in the auto_best cascade.

LocateAnything-3B uses a vision-language model to detect objects described
in text and returns precise bounding boxes — no training required.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..utils import free_vram, interruptible_range
from . import BaseSegmenter, register

logger = logging.getLogger("MEC.MaskMatting.LocateAnything")

_MODEL_CACHE: Dict[str, Tuple[Any, Any, Any]] = {}


def _have_deps() -> bool:
    try:
        import transformers  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


@register
class LocateAnythingSegmenter(BaseSegmenter):
    """Open-vocabulary grounding via NVIDIA LocateAnything-3B.

    Supports ``text`` mode: given a text prompt, locates all matching objects
    and produces filled bounding-box masks. When used in the auto_best
    cascade, the bbox output feeds directly into SAM for pixel-precise
    refinement — giving training-free, near-100% accuracy on described objects.
    """

    KEY = "locate_anything"
    DISPLAY = "LocateAnything-3B (NVIDIA)"
    STATUS = "ready" if _have_deps() else "missing-deps"
    MODELS_KEY = "locate_anything"
    SUPPORTS_MODES = {"text", "auto"}

    DEFAULT_REPO = "nvidia/LocateAnything-3B"

    def __init__(self, model_name: str = "", device: str = "cuda",
                 precision: str = "fp16", attention: str = "auto",
                 offload: str = "none"):
        super().__init__(model_name, device, precision, attention, offload)
        self._repo = self.DEFAULT_REPO

    def load(self) -> None:
        if self._model is not None:
            return
        key = self._repo
        if key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[key]
            return
        try:
            from transformers import AutoModel, AutoTokenizer, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "LocateAnything-3B requires `transformers`. "
                "Install with: pip install transformers"
            ) from e

        dtype = torch.bfloat16 if self.precision != "fp32" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            self._repo, trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            self._repo, trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            self._repo, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.device).eval()

        bundle = (model, tokenizer, processor)
        _MODEL_CACHE[key] = bundle
        self._model = bundle
        logger.info("LocateAnything-3B loaded on %s (%s)", self.device, self.precision)

    def unload(self) -> None:
        self._model = None
        key = self._repo
        if key in _MODEL_CACHE:
            del _MODEL_CACHE[key]
        free_vram()

    def segment(
        self,
        image_bhwc: torch.Tensor,
        *,
        mode: str = "auto",
        positive_points: Optional[List[Tuple[float, float]]] = None,
        negative_points: Optional[List[Tuple[float, float]]] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        neg_bbox: Optional[Tuple[int, int, int, int]] = None,
        text_prompt: str = "",
        frame_annotation: int = 0,
        object_id: int = 0,
        max_frames: int = 0,
        memory_size: int = 8,
        start_frame: int = 0,
        end_frame: int = -1,
        individual_objects: bool = False,
        tracking_direction: str = "forward",
        seed: int = 0,
    ) -> Dict[str, Any]:
        import re
        from PIL import Image

        if not text_prompt or not text_prompt.strip():
            return {
                "mask": torch.zeros(image_bhwc.shape[:3], dtype=torch.float32),
                "score": 0.0,
                "info": {"error": "LocateAnything requires a text prompt"},
                "bboxes": [],
            }

        self.load()
        model, tokenizer, processor = self._model
        B, H, W = image_bhwc.shape[0], image_bhwc.shape[1], image_bhwc.shape[2]
        mask_out = torch.zeros((B, H, W), dtype=torch.float32)
        all_bboxes: List[List[Dict[str, Any]]] = []
        total_score = 0.0

        for b in interruptible_range(B):
            img_np = (image_bhwc[b].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, "RGB")

            query = (
                f"Locate all the instances that match the following "
                f"description: {text_prompt}."
            )
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": query},
            ]}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            images, videos = processor.process_vision_info(messages)
            inputs = processor(
                text=[text], images=images, videos=videos,
                return_tensors="pt",
            ).to(self.device)

            pixel_values = inputs["pixel_values"].to(
                torch.bfloat16 if self.precision != "fp32" else torch.float32
            )
            gen_kwargs: Dict[str, Any] = {
                "pixel_values": pixel_values,
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "tokenizer": tokenizer,
                "max_new_tokens": 2048,
                "use_cache": True,
                "generation_mode": "hybrid",
                "temperature": 0.1,
                "do_sample": False,
            }
            hws = inputs.get("image_grid_hws", None)
            if hws is not None:
                gen_kwargs["image_grid_hws"] = hws

            with torch.no_grad():
                response = model.generate(**gen_kwargs)

            answer = response[0] if isinstance(response, tuple) else response
            answer_str = str(answer)

            bboxes: List[Dict[str, Any]] = []
            for m in re.finditer(
                r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer_str,
            ):
                x1n, y1n, x2n, y2n = (int(g) for g in m.groups())
                bboxes.append({
                    "x1": x1n / 1000 * W,
                    "y1": y1n / 1000 * H,
                    "x2": x2n / 1000 * W,
                    "y2": y2n / 1000 * H,
                    "label": text_prompt,
                    "score": 1.0,
                })

            for box in bboxes:
                expand = 0.02
                bw, bh = box["x2"] - box["x1"], box["y2"] - box["y1"]
                ix1 = max(0, int(round(box["x1"] - bw * expand)))
                iy1 = max(0, int(round(box["y1"] - bh * expand)))
                ix2 = min(W, int(round(box["x2"] + bw * expand)))
                iy2 = min(H, int(round(box["y2"] + bh * expand)))
                if ix2 > ix1 and iy2 > iy1:
                    mask_out[b, iy1:iy2, ix1:ix2] = 1.0

            all_bboxes.append(bboxes)
            total_score += (1.0 if bboxes else 0.0)

        avg_score = total_score / max(B, 1)
        flat = [box for frame in all_bboxes for box in frame]

        return {
            "mask": mask_out,
            "score": avg_score,
            "info": {
                "segmenter": self.KEY,
                "num_detections": len(flat),
                "prompt": text_prompt,
            },
            "bboxes": flat,
        }
