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
                # L7.97 (owner 2026-10-12): appended last; off = never download
                "allow_download": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Download the SegFormer model if it is not installed (into models/segformer/). Off: an "
                               "installed copy is used wherever it is; without one only 'skin' can be made "
                               "model-free (by colour, approximate) and the node names the model to install.",
                }),
                # L9.172 (A25 spec item 52): appended after A28; the defaults run exactly the old code
                "groups_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Your own class groups, used as '@name' in classes_csv: {\"my_face\": [\"@eyes\", "
                               "\"nose\", \"@lips\"]}. Built-in groups - face: @eyes @brows @lips @mouth_area @ears "
                               "@face_features @head @accessories; clothes: @shoes @legs @arms @body_skin @garments "
                               "@accessories. A leading '-' in classes_csv removes a class or group ('@head, -hair'). "
                               "Your groups win over built-ins of the same name. Empty = built-ins only.",
                }),
                "temporal_smooth": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "Video: smooth each frame's class probability over time before thresholding - a "
                               "forward and a backward running average across frames (weight of the previous frame "
                               "= this value), averaged. Fills one-frame dropouts and calms flicker; higher = "
                               "smoother but slower to follow real changes. 0 = off (each frame on its own).",
                }),
                "class_report": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add a per-frame report to info: the share of the picture each selected class covers "
                               "(pixels where it is the most likely class) and the frames where a class appears, "
                               "disappears or changes size by more than 1.5x.",
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
        allow_download: bool = False,
        groups_json: str = "",
        temporal_smooth: float = 0.0,
        class_report: bool = False,
    ):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("SemanticSegmentMEC expects IMAGE tensor [B,H,W,C]")
        try:
            temporal_smooth = float(temporal_smooth or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Semantic Segment (MEC): temporal_smooth must be a number from 0 to 0.95, got "
                             f"{temporal_smooth!r}.") from exc
        if not 0.0 <= temporal_smooth <= 0.95:
            raise ValueError(f"Semantic Segment (MEC): temporal_smooth must be from 0 to 0.95 (0 = off), got "
                             f"{temporal_smooth:g}.")
        from ._download_policy import allow_downloads
        with torch.no_grad(), allow_downloads(allow_download):
            return self._parse_impl(
                image, model_name, classes_csv, threshold, invert,
                keep_model_loaded, groups_json, temporal_smooth, bool(class_report),
            )

    def _parse_impl(
        self,
        image: torch.Tensor,
        model_name: str,
        classes_csv: str,
        threshold: float,
        invert: bool,
        keep_model_loaded: bool = True,
        groups_json: str = "",
        temporal_smooth: float = 0.0,
        class_report: bool = False,
    ):
        B, H, W, C = image.shape
        # L9.172: groups / exclusions are expanded (and checked) before the model loads, so a typo costs nothing
        from ._semantic_groups import expand_classes, uses_groups
        _family0 = MODEL_REGISTRY.get(model_name, {}).get("family", "")
        _classes0 = FACE_CLASSES if "face" in _family0 else CLOTHES_CLASSES
        grouped = uses_groups(classes_csv)
        included, excluded, used_groups = expand_classes(classes_csv, _classes0, "face" if "face" in _family0
                                                         else "clothes", groups_json)
        groups_info = {"used": used_groups, "excluded": excluded} if grouped else None
        # MANUAL bug-fix (Apr 2026): full device autodetect (cuda > mps > cpu).
        try:                                       # L9.262 (CNP-NODES-1 audit): ComfyUI's device first (--cpu, the chosen GPU); torch's own guess only outside ComfyUI
            import comfy.model_management as _mm
            device = str(_mm.get_torch_device())
        except Exception:  # noqa: BLE001
            device = None
        if device is not None:
            pass
        elif torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        try:
            loaded = get_or_load_model(model_name, precision="fp32", device=device)
        except Exception as exc:
            from .mask_matting import _classic
            if not (_classic.fallback_allowed() and _classic.is_missing_model_error(exc)):
                raise
            return self._model_free(image, model_name, classes_csv, invert, exc,
                                    included if grouped else None, groups_info, temporal_smooth, class_report)
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

        # Parse target classes (L9.172: "@group" / "-x" expanded above; a plain list gives the same set as before)
        target = set(included)
        target_indices = set()
        for idx, label in enumerate(all_classes):
            if label.lower() in target:
                target_indices.add(idx)

        if not target_indices:
            # A22 L9.136: a typo returned an all-zero mask (and ignored invert); say which classes exist instead
            raise ValueError(f"Semantic Segment (MEC): no class matches {classes_csv!r}"
                             + (f" (after groups: {included})" if grouped else "") + ". Available: "
                             f"{', '.join(all_classes[1:])}.")

        masks = []
        # L9.172: temporal_smooth keeps each frame's combined probability and thresholds after smoothing over time;
        # 0 (or a still) runs the per-frame threshold exactly as before. class_report reads the argmax coverage.
        smooth = temporal_smooth > 0 and B > 1
        kept_probs = []
        coverage = []
        report_indices = sorted(i for i in target_indices if i < len(all_classes))

        def _edge_refine(i, mask):
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
            return mask

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
            img_np = (np.clip(image[i].cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)   # A22 L9.136: no HDR wrap
            pil_img = PILImage.fromarray(img_np[:, :, :3])
            if (inf_h, inf_w) != (H, W):
                pil_img = pil_img.resize((inf_w, inf_h), PILImage.BILINEAR)

            inputs = processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            with torch.no_grad():
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

            if class_report:
                arg = probs.argmax(dim=0)
                coverage.append({all_classes[ci]: float((arg == ci).float().mean()) for ci in report_indices})
            if smooth:
                kept_probs.append(combined.float().cpu())
                continue
            mask = (combined > threshold).float().cpu()
            masks.append(_edge_refine(i, mask))

        if smooth:
            a = float(temporal_smooth)
            p = torch.stack(kept_probs)                       # (B, H, W) on the CPU
            fwd = p.clone()
            bwd = p.clone()
            for t in range(1, B):
                fwd[t] = a * fwd[t - 1] + (1.0 - a) * p[t]
            for t in range(B - 2, -1, -1):
                bwd[t] = a * bwd[t + 1] + (1.0 - a) * p[t]
            p = (fwd + bwd) / 2.0
            for i in range(B):
                masks.append(_edge_refine(i, (p[i] > threshold).float()))

        result = torch.stack(masks).clamp(0.0, 1.0)

        if invert:
            result = 1.0 - result

        if not keep_model_loaded:
            clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        matched = [all_classes[i] for i in sorted(target_indices) if i < len(all_classes)]
        from ._download_policy import model_status  # owner D7.91: found / loaded / missing / needs permission
        info_d = {
            "model": model_name,
            "model_status": model_status(),
            "family": family,
            "matched_classes": matched,
            "frames": B,
            "threshold": threshold,
            "invert": invert,
        }
        if groups_info is not None:                                   # L9.172: only when a group / "-x" was used
            info_d["groups"] = groups_info
        if temporal_smooth > 0 or class_report:
            info_d["temporal"] = _temporal_info(temporal_smooth, coverage if class_report else None)
        info = json.dumps(info_d, indent=2)

        return (result, info)

    @staticmethod
    def _model_free(image, model_name, classes_csv, invert, exc, included=None, groups_info=None,
                    temporal_smooth=0.0, class_report=False):
        """L7.97: SegFormer is not installed and may not be downloaded. Only skin can be found without it (by
        colour); every other class needs the model - said plainly, never faked. L9.172: `included` = the class list
        after group expansion (None = a plain list, read exactly as before)."""
        from .mask_matting import _classic
        if included is None:
            wanted = [c.strip().lower() for c in classes_csv.split(",") if c.strip()]
        else:
            wanted = list(included)
        possible = [c for c in wanted if c == "skin"]
        if not possible:
            raise ValueError(f"{exc} Without the model only 'skin' can be found (by colour); "
                             f"'{classes_csv}' needs the model.")
        B, H, W, _C = image.shape
        masks = [torch.from_numpy(_classic.skin_mask((np.clip(image[i, ..., :3].cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)))
                 for i in range(B)]
        result = torch.stack(masks).clamp(0.0, 1.0)
        coverage = [{"skin": float((result[i] > 0.5).float().mean())} for i in range(B)] if class_report else None
        if invert:
            result = 1.0 - result
        logger.warning("[MEC] SemanticSegment: %s - skin by colour only (model-free).", str(exc)[:200])
        from ._download_policy import model_status
        info_d = {
            "model": model_name, "model_status": model_status(fallback_why=str(exc)), "approximate": True,
            "model_free": {**_classic.describe_fallback(model_name, exc), "used": _classic.LABEL_SKIN},
            "matched_classes": possible, "needs_the_model": [c for c in wanted if c != "skin"], "frames": B,
            "invert": invert,
        }
        if groups_info is not None:
            info_d["groups"] = groups_info
        if temporal_smooth > 0 or class_report:
            info_d["temporal"] = _temporal_info(temporal_smooth, coverage)
            if temporal_smooth > 0:
                info_d["temporal"]["note"] = ("temporal_smooth was not applied: the model-free skin stand-in has no "
                                              "class probabilities to smooth.")
        info = json.dumps(info_d, indent=2)
        return (result, info)


COVERAGE_MIN = 0.001      # a class "is there" from 0.1 % of the picture
JUMP_RATIO = 1.5          # a size change by more than this factor (either way) is reported as a jump


def coverage_changes(per_frame):
    """L9.172 class_report: [{class: coverage}] per frame -> [{"frame", "class", "event", "from", "to"}], where event
    is "appeared" / "disappeared" (coverage crosses COVERAGE_MIN) or "jump" (both >= COVERAGE_MIN and the ratio
    new / old > JUMP_RATIO or < 1 / JUMP_RATIO)."""
    out = []
    for t in range(1, len(per_frame)):
        prev, cur = per_frame[t - 1], per_frame[t]
        for c in cur:
            a, b = float(prev.get(c, 0.0)), float(cur[c])
            ev = None
            if a < COVERAGE_MIN <= b:
                ev = "appeared"
            elif b < COVERAGE_MIN <= a:
                ev = "disappeared"
            elif a >= COVERAGE_MIN and b >= COVERAGE_MIN and (b / a > JUMP_RATIO or b / a < 1.0 / JUMP_RATIO):
                ev = "jump"
            if ev:
                out.append({"frame": t, "class": c, "event": ev, "from": round(a, 6), "to": round(b, 6)})
    return out


def _temporal_info(temporal_smooth, coverage):
    """info["temporal"]: {"smooth": a} and, with class_report, the per-frame coverage and the change events."""
    d = {"smooth": float(temporal_smooth)}
    if coverage is not None:
        d["per_frame"] = [{c: round(v, 6) for c, v in row.items()} for row in coverage]
        d["changes"] = coverage_changes(coverage)
    return d


# A28 (ORDERS 29, docs/audit/A28/PARAM_CNP_A.md "SemanticSegmentMEC"): which controls show when, read by js/_c2c_node_layout.js.
# Display only: every input keeps its name, order, type and default, is saved and reaches the node's function as before.
try:
    from ._layout_decl import declare_layout, merge, sections
except ImportError:  # pragma: no cover - loaded outside the pack
    declare_layout = None
if declare_layout is not None:
    declare_layout(SemanticSegmentMEC, sections(Performance=["keep_model_loaded"], Model_setup=["allow_download"],
                                                Groups=["groups_json"], Video=["temporal_smooth", "class_report"]))

# L9.172 (A25 spec item 52): inputs appended after the A28 audit, in order (read by tests/test_a28_cnp_a_layout.py)
APPENDED_AFTER_A28 = ("groups_json", "temporal_smooth", "class_report")
