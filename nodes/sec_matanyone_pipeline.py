"""
SeCMatAnyonePipelineMEC – SeC segmentation + MatAnyone2 alpha matting pipeline.

Pipeline stages:
  1. **SeC coarse segmentation** – MLLM-based object tracking (video-aware)
  2. **MatAnyone2 alpha matting** – Temporal-consistent alpha with warmup protocol
  3. **Edge refinement** – Optional ViTMatte / guided-filter edge cleanup
  4. **Post-processing** – Hole filling, small region removal

Best for: Video object segmentation with compositing-grade alpha edges,
especially scenes with occlusions, re-appearances, and complex motion.

Key differences from SAM + ViTMatte Pipeline:
  - SeC uses a Large Vision-Language Model for semantic understanding (text prompts)
  - MatAnyone2 provides temporal consistency across video frames
  - Better for long video sequences with appearance changes
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

from .model_manager import (
    MODEL_REGISTRY,
    get_or_load_model,
    clear_cache,
    scan_model_dir,
)

from . import _progress as _PB
from .utils import (
    HAS_CV2,
    compute_edge_band_np,
    fill_holes,
    remove_small_regions,
    make_split_preview,
)

logger = logging.getLogger("MEC")


class SeCMatAnyonePipelineMEC:
    """End-to-end SeC → MatAnyone2 pipeline.

    Combines SeC MLLM segmentation with MatAnyone2 temporal alpha matting
    for production-grade video masking with compositing-quality edges.
    """

    REFINE_METHODS = ["none", "vitmatte", "guided_filter", "multi_scale_guided"]

    @classmethod
    def INPUT_TYPES(cls):
        seg_models = []
        for name, reg in sorted(MODEL_REGISTRY.items()):
            if reg.get("family") in ("sec", "sam2", "sam3"):
                seg_models.append(name)
        if not seg_models:
            seg_models = ["sec_4b"]

        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Single image or video frames (B>1 for video).",
                }),
                "segmentation_model": (seg_models, {
                    "default": seg_models[0],
                    "tooltip": (
                        "Segmentation model for coarse masks.\n"
                        "SeC: best for video with text prompts.\n"
                        "SAM2/3: best for point/bbox prompts."
                    ),
                }),
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Text description of target object (e.g. 'cat', 'person in red').\n"
                        "Used by SeC for semantic tracking. Leave empty for point/bbox prompts."
                    ),
                }),
                "points_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": 'Point prompts: [{"x":100,"y":200,"label":1}, ...]',
                }),
                "bbox_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Bounding box: [x1,y1,x2,y2]",
                }),
                "matting_backend": (["matanyone2", "vitmatte_small", "vitmatte_base", "auto"], {
                    "default": "auto",
                    "tooltip": (
                        "Alpha matting backend.\n"
                        "auto: MatAnyone2 for video (B>1), ViTMatte for single images.\n"
                        "matanyone2: Video matting with temporal consistency.\n"
                        "vitmatte_small/base: Neural matting (best edge quality per frame)."
                    ),
                }),
                "edge_radius": ("INT", {
                    "default": 15, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Edge refinement radius in pixels.",
                }),
                "n_warmup": ("INT", {
                    "default": 5, "min": 1, "max": 30, "step": 1,
                    "tooltip": "MatAnyone2 warmup frames (more = better temporal init).",
                }),
                "precision": (["fp16", "bf16", "fp32"], {
                    "default": "fp16",
                    "tooltip": "Segmentation model precision.",
                }),
                "fill_holes_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill interior holes in the final alpha.",
                }),
                "min_region_size": ("INT", {
                    "default": 64, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Remove isolated regions smaller than N pixels.",
                }),
            },
            "optional": {
                "positive_coords": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Positive points from Points Mask Editor.",
                }),
                "negative_coords": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Negative points from Points Mask Editor.",
                }),
                "bbox": ("BBOX", {
                    "tooltip": "Positive bbox from upstream node.",
                }),
                "edge_refine_method": (cls.REFINE_METHODS, {
                    "default": "none",
                    "tooltip": (
                        "Optional post-matting edge refinement.\n"
                        "none: use raw MatAnyone2 output.\n"
                        "vitmatte/guided_filter: refine edges after matting."
                    ),
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep models in VRAM between runs.",
                }),
                # L7.97 (owner 2026-10-12): appended last; off = never download
                "allow_download": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Download the segmentation / matting models if they are not installed (into their "
                               "models/ subfolders). Off: installed copies are used wherever they are; without them "
                               "the node segments and mattes model-free (GrabCut + optical flow + guided filter, "
                               "approximate) and names the models to install.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("rgb", "alpha_mask", "coarse_mask", "preview", "info")
    OUTPUT_TOOLTIPS = (
        "Original RGB frames passed through for downstream nodes.",
        "Compositing-grade alpha mask after MatAnyone2 + optional refinement.",
        "Coarse segmentation mask before alpha matting.",
        "Side-by-side preview of input frames and final alpha overlay.",
        "JSON summary of stages, models, and per-stage timings.",
    )
    FUNCTION = "process"
    CATEGORY = "C2C/Pipeline"
    DESCRIPTION = (
        "SeC + MatAnyone2 end-to-end pipeline:\n"
        "1. SeC/SAM segmentation → coarse masks\n"
        "2. MatAnyone2 temporal alpha matting → compositing-grade alpha\n"
        "3. Optional edge refinement\n"
        "4. Post-processing (hole fill, region cleanup)\n\n"
        "Best for video scenes with occlusions, re-appearances, and complex motion."
    )

    @classmethod
    def IS_CHANGED(cls, image, segmentation_model, text_prompt, points_json, bbox_json,
                   matting_backend, edge_radius, n_warmup, precision,
                   fill_holes_enabled, min_region_size,
                   positive_coords=None, negative_coords=None, bbox=None,
                   edge_refine_method="none", keep_model_loaded=True, **kwargs):
        return hash_args_and_kwargs(
            image, segmentation_model, text_prompt, points_json, bbox_json,
            matting_backend, edge_radius, n_warmup, precision,
            fill_holes_enabled, min_region_size,
            positive_coords, negative_coords, bbox,
            edge_refine_method, keep_model_loaded, **kwargs,
        )

    def process(
        self,
        image: torch.Tensor,
        segmentation_model: str,
        text_prompt: str,
        points_json: str,
        bbox_json: str,
        matting_backend: str,
        edge_radius: int,
        n_warmup: int,
        precision: str,
        fill_holes_enabled: bool,
        min_region_size: int,
        positive_coords: str | None = None,
        negative_coords: str | None = None,
        bbox=None,
        edge_refine_method: str = "none",
        keep_model_loaded: bool = True,
        allow_download: bool = False,
    ):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("SeCMatAnyonePipelineMEC expects IMAGE tensor [B,H,W,C]")

        from ._download_policy import allow_downloads
        with torch.no_grad(), allow_downloads(allow_download):
            return self._process_impl(
                image, segmentation_model, text_prompt, points_json, bbox_json,
                matting_backend, edge_radius, n_warmup, precision,
                fill_holes_enabled, min_region_size,
                positive_coords, negative_coords, bbox,
                edge_refine_method, keep_model_loaded,
            )

    def _process_impl(
        self,
        image: torch.Tensor,
        segmentation_model: str,
        text_prompt: str,
        points_json: str,
        bbox_json: str,
        matting_backend: str,
        edge_radius: int,
        n_warmup: int,
        precision: str,
        fill_holes_enabled: bool,
        min_region_size: int,
        positive_coords: str | None = None,
        negative_coords: str | None = None,
        bbox=None,
        edge_refine_method: str = "none",
        keep_model_loaded: bool = True,
    ):
        B, H, W, C = image.shape
        is_video = B > 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Stage 1: Coarse Segmentation ──────────────────────────────
        # MANUAL bug-fix (Apr 2026): the legacy import targeted
        # `unified_segmentation_node.UnifiedSegmentationNode` which lives in
        # `_deprecated/`.  The active replacement is
        # `unified_segmentation.UnifiedSegmentation` whose `segment()` exposes
        # a smaller kwarg surface — call it with only the supported keys.
        from .mask_matting import _classic
        model_free: dict = {}
        from .unified_segmentation import UnifiedSegmentation
        seg_node = UnifiedSegmentation()
        try:
            coarse_masks, score, seg_info = seg_node.segment(
                image=image,
                model_name=segmentation_model,
                points_json=points_json,
                bbox_json=bbox_json,
                multimask=True,
                mask_index=0,
                precision=precision,
                bbox=bbox,
                text_prompt=text_prompt,
                existing_mask=None,
            )
        except Exception as exc:
            # L7.97: the segmentation model is not installed (and may not be downloaded) -> model-free, labelled
            if not (_classic.fallback_allowed() and _classic.is_missing_model_error(exc)):
                raise
            coarse_masks, score = self._model_free_segment(image, points_json, positive_coords, negative_coords,
                                                           bbox_json, bbox, text_prompt)
            model_free["segmentation"] = {**_classic.describe_fallback(segmentation_model, exc),
                                          "used": _classic.LABEL_SEGMENT}
            logger.warning("[MEC] SeC pipeline: %s - model-free segmentation.", str(exc)[:200])

        # ── Stage 2: Alpha Matting ────────────────────────────────────
        actual_backend = matting_backend
        if actual_backend == "auto":
            actual_backend = "matanyone2" if is_video else "vitmatte_small"

        if actual_backend == "matanyone2":
            try:
                alpha_mask = self._run_matanyone2(
                    image, coarse_masks, n_warmup, B, H, W, device,
                )
            except Exception as e:
                # MatAnyone2 weights are not yet on HuggingFace (the upstream
                # `pq-yang/MatAnyone2` repo doesn't exist as of May 2026).  We
                # fall back to ViTMatte so the pipeline still produces a
                # usable alpha — the user just loses temporal consistency.
                logger.warning(
                    "[MEC] MatAnyone2 unavailable (%s: %s) → falling back to "
                    "vitmatte_small. Install MatAnyone2 weights manually to "
                    "models/matanyone2/ to enable temporal matting.",
                    type(e).__name__, str(e)[:120],
                )
                actual_backend = "vitmatte_small"
                try:
                    alpha_mask = self._run_vitmatte(
                        image, coarse_masks, actual_backend, edge_radius, B, H, W, device,
                    )
                except Exception as exc:
                    if not (_classic.fallback_allowed() and _classic.is_missing_model_error(exc)):
                        raise
                    model_free["matting"] = {**_classic.describe_fallback("matanyone2 / vitmatte_small", exc),
                                             "used": _classic.LABEL_MATTE}
                    logger.warning("[MEC] SeC pipeline: %s - model-free guided-filter matte.", str(exc)[:200])
                    actual_backend = "guided_filter (model-free)"
                    alpha_mask = self._model_free_matte(image, coarse_masks, edge_radius)
        else:
            alpha_mask = None
            try:
                alpha_mask = self._run_vitmatte(
                    image, coarse_masks, actual_backend, edge_radius, B, H, W, device,
                )
            except Exception as exc:
                if not (_classic.fallback_allowed() and _classic.is_missing_model_error(exc)):
                    raise
                model_free["matting"] = {**_classic.describe_fallback(actual_backend, exc),
                                         "used": _classic.LABEL_MATTE}
                logger.warning("[MEC] SeC pipeline: %s - model-free guided-filter matte.", str(exc)[:200])
                actual_backend = "guided_filter (model-free)"
                alpha_mask = self._model_free_matte(image, coarse_masks, edge_radius)

        # ── Stage 3: Optional Edge Refinement ─────────────────────────
        if edge_refine_method != "none":
            alpha_mask = self._edge_refine(
                image, alpha_mask, edge_refine_method, edge_radius, B, H, W,
            )

        # ── Stage 4: Post-processing ─────────────────────────────────
        if fill_holes_enabled or min_region_size > 0:
            alpha_mask = self._post_process(
                alpha_mask, fill_holes_enabled, min_region_size, B, H, W,
            )

        alpha_mask = alpha_mask.float().clamp(0.0, 1.0)

        # Build outputs
        rgb = image[:B] * alpha_mask.unsqueeze(-1)

        # Preview: side-by-side original vs masked
        try:
            preview = make_split_preview(
                image[0], coarse_masks[0], alpha_mask[0],
            )
            preview = preview.unsqueeze(0)
        except Exception:
            preview = rgb[:1]

        # Free VRAM
        if not keep_model_loaded:
            clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from ._download_policy import model_status  # owner D7.91 status line
        info = json.dumps({
            "segmentation_model": segmentation_model,
            "matting_backend": actual_backend,
            "is_video": is_video,
            "frames": B,
            "seg_score": round(score, 4),
            "edge_refine": edge_refine_method,
            "n_warmup": n_warmup,
            # owner D7.91: found / loaded / missing / needs permission, per model
            "model_status": "; ".join(
                f"{part}: " + (model_status(fallback_why=model_free[part].get("why", "")) if part in model_free
                                else "loaded") for part in ("segmentation", "matting")),
            **({"approximate": True, "model_free": model_free} if model_free else {}),
        }, indent=2)

        return (rgb, alpha_mask, coarse_masks, preview, info)

    # ── model-free stand-ins (L7.97) ──────────────────────────────────

    @staticmethod
    def _model_free_segment(image, points_json, positive_coords, negative_coords, bbox_json, bbox, text_prompt):
        """The prompts this node takes, through the model-free segmenter (GrabCut, tracked by optical flow)."""
        from .mask_matting.segmenters.classic_backend import ClassicSegmenter
        from .mask_matting.utils import parse_points
        from .utils import parse_bbox_input
        pos, neg = parse_points(points_json or "")
        for raw, is_pos in ((positive_coords, True), (negative_coords, False)):
            a, b = parse_points(raw or "")
            (pos if is_pos else neg).extend(a + b)
        box = parse_bbox_input(bbox_json, bbox)
        box = tuple(float(v) for v in np.asarray(box).reshape(-1)[:4]) if box is not None else None
        out = ClassicSegmenter().segment(image, positive_points=pos, negative_points=neg, bbox=box,
                                         text_prompt=text_prompt or "", tracking_direction="bidirectional")
        return out["mask"].float().clamp(0, 1), float(out["score"])

    def _model_free_matte(self, image, coarse_masks, edge_radius):
        from .mask_matting import _classic
        alphas = []
        for i in range(coarse_masks.shape[0]):
            tri = torch.from_numpy(self._generate_trimap(coarse_masks[i].cpu().numpy().astype(np.float32),
                                                         int(edge_radius)).astype(np.float32) / 255.0)
            img = image[min(i, image.shape[0] - 1):min(i, image.shape[0] - 1) + 1, ..., :3].float().cpu()
            alphas.append(_classic.guided_matte(img, tri.unsqueeze(0), radius=max(2, int(edge_radius)))[0])
        return torch.stack(alphas)

    # ── MatAnyone2 matting ────────────────────────────────────────────

    def _run_matanyone2(self, image, mask, n_warmup, B, H, W, device):
        """MatAnyone2 temporal alpha matting with warmup protocol."""
        loaded = get_or_load_model("matanyone2", precision="fp32", device=device)
        core = loaded["core"] if isinstance(loaded, dict) else loaded

        alphas = []

        first_img = image[0].permute(2, 0, 1).to(device)
        first_mask = mask[0].unsqueeze(0).to(device)
        first_mask_bin = (first_mask > 0.5).float()

        try:
            core.step(first_img, first_mask_bin)
        except Exception as exc:
            logger.debug("[MEC] MatAnyone2 initial step: %s", exc)

        for _ in _PB.track(range(n_warmup), n_warmup, "MatAnyone"):
            try:
                core.step(first_img, first_frame_pred=True)
            except TypeError:
                try:
                    core.step(first_img, first_mask_bin)
                except Exception:
                    break
            except Exception:
                break

        for i in _PB.track(range(B), B, "MatAnyone"):
            _IC.check()
            img_t = image[min(i, image.shape[0] - 1)].permute(2, 0, 1).to(device)
            m_t = mask[i].unsqueeze(0).to(device)
            m_bin = (m_t > 0.5).float()

            try:
                alpha = core.step(img_t, m_bin) if i == 0 else core.step(img_t)
                if isinstance(alpha, torch.Tensor):
                    alphas.append(alpha.cpu().squeeze())
                else:
                    alphas.append(torch.from_numpy(np.array(alpha, dtype=np.float32)))
            except Exception as exc:
                logger.warning("[MEC] MatAnyone2 frame %d: %s", i, exc)
                alphas.append(mask[i].cpu())

        return torch.stack(alphas)

    # ── ViTMatte matting ──────────────────────────────────────────────

    def _run_vitmatte(self, image, mask, variant, edge_radius, B, H, W, device):
        """ViTMatte per-frame alpha matting."""
        from PIL import Image as PILImage

        loaded = get_or_load_model(variant, precision="fp32", device=device)
        if isinstance(loaded, dict):
            model, processor = loaded["model"], loaded["processor"]
        else:
            model, processor = loaded, None

        if processor is None:
            from transformers import VitMatteImageProcessor
            processor = VitMatteImageProcessor()

        dev = next(model.parameters()).device
        alphas = []

        for i in _PB.track(range(B), B, "MatAnyone"):
            _IC.check()
            img_np = (image[min(i, image.shape[0] - 1)].cpu().numpy() * 255).astype(np.uint8)
            mask_np = mask[i].cpu().numpy().astype(np.float32)

            trimap_u8 = self._generate_trimap(mask_np, edge_radius)
            pil_img = PILImage.fromarray(img_np[:, :, :3])
            pil_tri = PILImage.fromarray(trimap_u8, mode="L")

            inputs = processor(images=pil_img, trimaps=pil_tri, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs)

            a = out.alphas[0, 0].cpu()
            if a.shape[0] != H or a.shape[1] != W:
                a = F.interpolate(
                    a.unsqueeze(0).unsqueeze(0), (H, W),
                    mode="bilinear", align_corners=False,
                )[0, 0]

            edge_band = compute_edge_band_np(mask_np, edge_radius)
            eb_t = torch.from_numpy(edge_band)
            blended = mask[i].cpu() * (1 - eb_t) + a * eb_t
            alphas.append(blended)

        return torch.stack(alphas)

    # ── Edge refinement ───────────────────────────────────────────────

    def _edge_refine(self, image, alpha, method, edge_radius, B, H, W):
        """Post-matting edge refinement."""
        if method == "vitmatte":
            return self._run_vitmatte(
                image, alpha, "vitmatte_small", edge_radius, B, H, W,
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        elif method == "guided_filter" and HAS_CV2:
            import cv2
            refined = []
            for i in _PB.track(range(B), B, "MatAnyone"):
                _IC.check()
                img_np = (image[min(i, image.shape[0] - 1)].cpu().numpy() * 255).astype(np.uint8)
                a_np = alpha[i].cpu().numpy()
                guide = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                r = cv2.ximgproc.guidedFilter(
                    guide, a_np.astype(np.float32), edge_radius, 1e-4,
                ) if hasattr(cv2, 'ximgproc') else a_np
                refined.append(torch.from_numpy(r))
            return torch.stack(refined)
        elif method == "multi_scale_guided":
            from .utils import multi_scale_guided_refine
            refined = []
            for i in _PB.track(range(B), B, "MatAnyone"):
                _IC.check()
                img_np = (image[min(i, image.shape[0] - 1)].cpu().numpy() * 255).astype(np.uint8)
                a_np = alpha[i].cpu().numpy()
                r = multi_scale_guided_refine(img_np, a_np, edge_radius, 0.5)
                if r is not None:
                    refined.append(torch.from_numpy(r))
                else:
                    refined.append(alpha[i])
            return torch.stack(refined)

        return alpha

    # ── Post-processing ───────────────────────────────────────────────

    def _post_process(self, alpha, do_fill, min_size, B, H, W):
        """Fill holes and remove small regions."""
        processed = []
        for i in _PB.track(range(B), B, "MatAnyone"):
            _IC.check()
            a = alpha[i].cpu().numpy()
            if do_fill:
                a = fill_holes(a)
            if min_size > 0:
                a = remove_small_regions(a, min_size)
            processed.append(torch.from_numpy(a))
        return torch.stack(processed)

    # ── Trimap generation ─────────────────────────────────────────────

    @staticmethod
    def _generate_trimap(mask_np: np.ndarray, edge_radius: int) -> np.ndarray:
        """Generate 3-region trimap: 255=fg, 128=unknown, 0=bg."""
        if not HAS_CV2:
            out = np.zeros(mask_np.shape, dtype=np.uint8)
            out[mask_np > 0.5] = 255
            return out

        import cv2
        binary = (mask_np > 0.5).astype(np.uint8) * 255
        ksize = edge_radius * 2 + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        eroded = cv2.erode(binary, kern, iterations=1)
        dilated = cv2.dilate(binary, kern, iterations=1)
        unknown = (dilated > 127) & (eroded < 128)

        trimap = np.zeros(mask_np.shape, dtype=np.uint8)
        trimap[eroded > 127] = 255
        trimap[unknown] = 128
        return trimap
