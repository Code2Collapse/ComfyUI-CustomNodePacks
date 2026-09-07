"""LocateAnything-3B grounding + SAM prompt converter nodes.

``LocateAnythingGroundingMEC`` wraps NVIDIA's LocateAnything-3B model for
open-vocabulary object grounding.  It returns bounding boxes, an annotated
preview image, and the raw model response.

``LocateAnythingToSAMMEC`` converts the BBOX_LIST output into filled
rectangle masks that downstream SAM nodes can consume as box prompts.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ._is_changed_util import hash_args_and_kwargs

logger = logging.getLogger("MEC.LocateAnything")


class LocateAnythingGroundingMEC:
    """Open-vocabulary object grounding via NVIDIA LocateAnything-3B."""

    _model_cache: Dict[str, Tuple[Any, Any, Any]] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "person", "multiline": True}),
            },
            "optional": {
                "model_path": ("STRING", {"default": "nvidia/LocateAnything-3B"}),
                "generation_mode": (["hybrid", "fast", "slow"], {"default": "hybrid"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
            },
        }

    RETURN_TYPES = ("BBOX_LIST", "IMAGE", "STRING")
    RETURN_NAMES = ("bboxes", "annotated_image", "raw_response")
    FUNCTION = "run"
    CATEGORY = "MaskEnhancedControl/Grounding"

    @classmethod
    def IS_CHANGED(cls, image, prompt, model_path="nvidia/LocateAnything-3B",
                   generation_mode="hybrid", max_new_tokens=2048, device="cuda",
                   confidence_threshold=0.0, **kwargs):
        return hash_args_and_kwargs(
            image, prompt, model_path, generation_mode, max_new_tokens,
            device, confidence_threshold, **kwargs,
        )

    @classmethod
    def _load_model(cls, model_path: str, device: str):
        if model_path in cls._model_cache:
            return cls._model_cache[model_path]
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device).eval()
        cls._model_cache[model_path] = (model, tokenizer, processor)
        return model, tokenizer, processor

    def run(
        self,
        image: torch.Tensor,
        prompt: str,
        model_path: str = "nvidia/LocateAnything-3B",
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        device: str = "cuda",
        confidence_threshold: float = 0.0,
    ):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError(
                "LocateAnythingGroundingMEC expects IMAGE tensor [B,H,W,C]"
            )
        with torch.inference_mode():
            return self._run_impl(
                image, prompt, model_path, generation_mode, max_new_tokens,
                device, confidence_threshold,
            )

    def _run_impl(
        self,
        image: torch.Tensor,
        prompt: str,
        model_path: str = "nvidia/LocateAnything-3B",
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        device: str = "cuda",
        confidence_threshold: float = 0.0,
    ):
        from PIL import Image

        # Cross-platform safety: the device widget defaults to "cuda", but on a
        # CPU-only box (no NVIDIA driver / hardware accel off) `.to("cuda")` raises.
        # Resolve to CPU when CUDA is unavailable so the node degrades gracefully
        # instead of crashing — and both the model and the input tensors use the
        # same resolved device. Verified statically; no GPU needed to confirm.
        # Source: https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        B = image.shape[0]
        all_bboxes: List[List[Dict[str, Any]]] = []
        all_annotated: List[np.ndarray] = []
        all_raw: List[str] = []

        model, tokenizer, processor = self._load_model(model_path, device)

        for b in range(B):
            img_np = (image[b].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np, "RGB")
            W, H = pil_img.size

            query = (
                f"Locate all the instances that match the following "
                f"description: {prompt}."
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
            ).to(device)

            pixel_values = inputs["pixel_values"].to(torch.bfloat16)
            input_ids = inputs["input_ids"]
            image_grid_hws = inputs.get("image_grid_hws", None)

            gen_kwargs: Dict[str, Any] = {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": inputs["attention_mask"],
                "tokenizer": tokenizer,
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "generation_mode": generation_mode,
                "temperature": 0.1,
                "do_sample": False,
            }
            if image_grid_hws is not None:
                gen_kwargs["image_grid_hws"] = image_grid_hws

            with torch.no_grad():
                response = model.generate(**gen_kwargs)

            answer = response[0] if isinstance(response, tuple) else response
            answer_str = str(answer)

            bboxes: List[Dict[str, Any]] = []
            for m in re.finditer(
                r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer_str,
            ):
                x1, y1, x2, y2 = (int(g) for g in m.groups())
                bboxes.append({
                    "x1": x1 / 1000 * W,
                    "y1": y1 / 1000 * H,
                    "x2": x2 / 1000 * W,
                    "y2": y2 / 1000 * H,
                    "label": prompt,
                })

            annotated = img_np.copy()
            try:
                import cv2
                for box in bboxes:
                    cv2.rectangle(
                        annotated,
                        (int(box["x1"]), int(box["y1"])),
                        (int(box["x2"]), int(box["y2"])),
                        (0, 255, 0), 2,
                    )
                    cv2.putText(
                        annotated, box.get("label", ""),
                        (int(box["x1"]), max(int(box["y1"]) - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )
            except ImportError:
                pass

            all_bboxes.append(bboxes)
            all_annotated.append(annotated)
            all_raw.append(answer_str)

        flat_bboxes = [box for frame_boxes in all_bboxes for box in frame_boxes]

        annotated_stack = np.stack(all_annotated, axis=0)
        annotated_t = torch.from_numpy(
            annotated_stack.astype(np.float32) / 255.0
        )

        raw_combined = "\n---\n".join(all_raw) if len(all_raw) > 1 else (
            all_raw[0] if all_raw else ""
        )

        return (flat_bboxes, annotated_t, raw_combined)


class LocateAnythingToSAMMEC:
    """Convert LocateAnything bounding boxes into filled-rectangle masks
    that downstream SAM nodes can use as box prompts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX_LIST",),
                "image": ("IMAGE",),
            },
            "optional": {
                "expand_ratio": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = "MaskEnhancedControl/Grounding"

    @classmethod
    def IS_CHANGED(cls, bboxes, image, expand_ratio=0.05, **kwargs):
        return hash_args_and_kwargs(bboxes, image, expand_ratio, **kwargs)

    def run(
        self,
        bboxes: List[Dict[str, Any]],
        image: torch.Tensor,
        expand_ratio: float = 0.05,
    ):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError(
                "LocateAnythingToSAMMEC expects IMAGE tensor [B,H,W,C]"
            )
        with torch.inference_mode():
            B, H, W = image.shape[0], image.shape[1], image.shape[2]
            mask = torch.zeros((B, H, W), dtype=torch.float32)

            for box in bboxes:
                x1, y1 = float(box["x1"]), float(box["y1"])
                x2, y2 = float(box["x2"]), float(box["y2"])

                if expand_ratio > 0:
                    bw = x2 - x1
                    bh = y2 - y1
                    x1 = x1 - bw * expand_ratio
                    y1 = y1 - bh * expand_ratio
                    x2 = x2 + bw * expand_ratio
                    y2 = y2 + bh * expand_ratio

                ix1 = max(0, int(round(x1)))
                iy1 = max(0, int(round(y1)))
                ix2 = min(W, int(round(x2)))
                iy2 = min(H, int(round(y2)))

                if ix2 > ix1 and iy2 > iy1:
                    mask[:, iy1:iy2, ix1:ix2] = 1.0

            return (mask,)


NODE_CLASS_MAPPINGS = {
    "LocateAnythingGroundingMEC": LocateAnythingGroundingMEC,
    "LocateAnythingToSAMMEC": LocateAnythingToSAMMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LocateAnythingGroundingMEC": "LocateAnything Grounding (MEC)",
    "LocateAnythingToSAMMEC": "LocateAnything \u2192 SAM Prompt (MEC)",
}
