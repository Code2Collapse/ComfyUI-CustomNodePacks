"""
SemanticSegmentMEC – Face / Body / Clothes semantic parsing using SegFormer.

Models supported:
  - **segformer_face** (jonathandinu/face-parsing) – 19-class facial parts
    (skin, nose, eyes, eyebrows, ears, mouth, lips, hair, hat, glasses, …)
  - **segformer_clothes** (mattmdjaga/segformer_b2_clothes) – 18-class apparel
    (hat, hair, sunglasses, upper-clothes, skirt, pants, dress, belt, shoe, bag,
    scarf, face, left/right arm/leg, …)

Output: One combined MASK for all selected classes.
Each run processes the full batch, giving per-frame masks for video workflows.
"""


from __future__ import annotations

from . import _interrupt_check as _IC
from ._is_changed_util import hash_args_and_kwargs

import gc
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from .model_manager import (
    MODEL_REGISTRY,
    get_or_load_model,
    clear_cache,
)

from . import _progress as _PB

logger = logging.getLogger("MEC")

# ── Class labels for each model ───────────────────────────────────────

FACE_CLASSES = [
    "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye",
    "eye_g", "l_ear", "r_ear", "ear_r", "nose", "mouth", "u_lip",
    "l_lip", "neck", "necklace", "cloth", "hair", "hat",
]

CLOTHES_CLASSES = [
    "background", "hat", "hair", "sunglasses", "upper_clothes", "skirt",
    "pants", "dress", "belt", "left_shoe", "right_shoe", "face",
    "left_leg", "right_leg", "left_arm", "right_arm", "bag", "scarf",
]


class SemanticSegmentMEC:
    """Parse face or clothing regions from images using SegFormer.

    Select which semantic classes to include in the output mask.
    Multiple classes are merged into a single binary mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = []
        for name, reg in sorted(MODEL_REGISTRY.items()):
            if reg.get("family") in ("segformer_face", "segformer_clothes"):
                models.append(name)
        if not models:
            models = ["segformer_face", "segformer_clothes"]

        face_opts = [c for c in FACE_CLASSES if c != "background"]
        clothes_opts = [c for c in CLOTHES_CLASSES if c != "background"]

        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image(s) to parse.",
                }),
                "model_name": (models, {
                    "default": models[0],
                    "tooltip": (
                        "segformer_face: 19-class facial parts.\n"
                        "segformer_clothes: 18-class apparel."
                    ),
                }),
                "classes_csv": ("STRING", {
                    "default": "skin,hair",
                    "multiline": False,
                    "tooltip": (
                        "Comma-separated class names to include in mask.\n"
                        "Face: skin, l_brow, r_brow, l_eye, r_eye, eye_g, "
                        "l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip, "
                        "neck, necklace, cloth, hair, hat\n"
                        "Clothes: hat, hair, sunglasses, upper_clothes, skirt, "
                        "pants, dress, belt, left_shoe, right_shoe, face, "
                        "left_leg, right_leg, left_arm, right_arm, bag, scarf"
                    ),
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Confidence threshold for class assignment.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the output mask.",
                }),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM between runs.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    OUTPUT_TOOLTIPS = (
        "Combined binary mask covering all selected semantic classes.",
        "JSON summary of model, classes used, and per-class pixel counts.",
    )
    FUNCTION = "parse"
    CATEGORY = "C2C/Segmentation"
    DESCRIPTION = (
        "Semantic face / clothes parsing using SegFormer.\n"
        "Select classes by name (comma-separated) to build a combined mask.\n"
        "Face model: skin, eyes, nose, mouth, hair, hat, glasses, ears.\n"
        "Clothes model: upper_clothes, pants, dress, shoes, bag, scarf, etc."
    )

    @classmethod
    def IS_CHANGED(cls, image, model_name, classes_csv, threshold, invert,
                   keep_model_loaded=True, **kwargs):
        return hash_args_and_kwargs(
            image, model_name, classes_csv, threshold, invert,
            keep_model_loaded, **kwargs,
        )

    def parse(
        self,
        image: torch.Tensor,
        model_name: str,
        classes_csv: str,
        threshold: float,
        invert: bool,
        keep_model_loaded: bool = True,
    ):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("SemanticSegmentMEC expects IMAGE tensor [B,H,W,C]")
        with torch.inference_mode():
            return self._parse_impl(
                image, model_name, classes_csv, threshold, invert,
                keep_model_loaded,
            )

    def _parse_impl(
        self,
        image: torch.Tensor,
        model_name: str,
        classes_csv: str,
        threshold: float,
        invert: bool,
        keep_model_loaded: bool = True,
    ):
        B, H, W, C = image.shape
        # MANUAL bug-fix (Apr 2026): full device autodetect (cuda > mps > cpu).
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        loaded = get_or_load_model(model_name, precision="fp32", device=device)
        model = loaded["model"]
        processor = loaded["processor"]
        dev = next(model.parameters()).device

        # Determine class list for this model
        reg = MODEL_REGISTRY.get(model_name, {})
        family = reg.get("family", "")
        if "face" in family:
            all_classes = FACE_CLASSES
        else:
            all_classes = CLOTHES_CLASSES

        # Parse target classes
        target = {c.strip().lower() for c in classes_csv.split(",") if c.strip()}
        target_indices = set()
        for idx, label in enumerate(all_classes):
            if label.lower() in target:
                target_indices.add(idx)

        if not target_indices:
            logger.warning(
                "[MEC] SemanticSegment: no matching classes for '%s'. "
                "Available: %s",
                classes_csv,
                ", ".join(all_classes[1:]),
            )
            return (torch.zeros(B, H, W), json.dumps({"error": "no matching classes"}))

        masks = []

        # Downsample very large images for inference to keep runtime sane.
        # SegFormer is shift-equivariant; logits will be upsampled back to (H, W) below.
        SEG_MAX_EDGE = 2048
        long_edge = max(H, W)
        if long_edge > SEG_MAX_EDGE:
            scale = SEG_MAX_EDGE / float(long_edge)
            inf_h = max(1, int(round(H * scale)))
            inf_w = max(1, int(round(W * scale)))
        else:
            inf_h, inf_w = H, W

        for i in _PB.track(range(B), B, "SemanticSeg"):
            _IC.check()
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = PILImage.fromarray(img_np[:, :, :3])
            if (inf_h, inf_w) != (H, W):
                pil_img = pil_img.resize((inf_w, inf_h), PILImage.BILINEAR)

            inputs = processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)

            logits = outputs.logits  # (1, num_classes, h, w)

            # Upsample to original resolution
            upsampled = F.interpolate(
                logits, (H, W), mode="bilinear", align_corners=False,
            )

            probs = torch.softmax(upsampled[0], dim=0)  # (num_classes, H, W)

            # Combine selected class probabilities
            combined = torch.zeros(H, W, device=dev)
            for ci in target_indices:
                if ci < probs.shape[0]:
                    combined = torch.maximum(combined, probs[ci])

            mask = (combined > threshold).float().cpu()
            # MANUAL bug-fix (Apr 2026): edge-refine after upsample. Apply a
            # cheap guided-filter-style refinement using the source image
            # luminance: align the mask boundary with strong image gradients
            # so the bilinear-upsampled segmentation snaps back to true edges.
            try:
                src_lum = image[i, ..., :3].mean(dim=-1).cpu().numpy().astype(np.float32)
                m_np = mask.numpy().astype(np.float32)
                # Joint bilateral via cv2 if available; falls back silently.
                import cv2 as _cv2  # noqa: F401
                refined = _cv2.ximgproc.guidedFilter(
                    guide=src_lum, src=m_np, radius=4, eps=1e-3
                ) if hasattr(_cv2, "ximgproc") else _cv2.bilateralFilter(
                    m_np, d=5, sigmaColor=0.1, sigmaSpace=4
                )
                mask = torch.from_numpy(np.clip(refined, 0.0, 1.0))
            except Exception:
                pass
            masks.append(mask)

        result = torch.stack(masks).clamp(0.0, 1.0)

        if invert:
            result = 1.0 - result

        if not keep_model_loaded:
            clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        matched = [all_classes[i] for i in sorted(target_indices) if i < len(all_classes)]
        info = json.dumps({
            "model": model_name,
            "family": family,
            "matched_classes": matched,
            "frames": B,
            "threshold": threshold,
            "invert": invert,
        }, indent=2)

        return (result, info)
