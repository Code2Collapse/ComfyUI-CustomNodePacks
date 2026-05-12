"""MaskMattingMEC — single multi-backend segmentation + matting node."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .matters import all_matters, get_matter_cls
from .matters import list_keys as list_matter_keys
from .segmenters import all_segmenters, get_segmenter_cls
from .segmenters import list_keys as list_segmenter_keys
from . import _vfx
from .utils import (
    apply_subject_preset,
    bbox_from_mask,
    bbox_to_json,
    download_preset,
    free_vram,
    list_backend_files,
    list_backend_presets,
    mask_to_trimap,
    parse_bbox,
    parse_points,
    to_bhwc,
    to_mask,
)

logger = logging.getLogger("MEC.MaskMatting")


def _segmenter_choices() -> List[str]:
    out: List[str] = []
    for k, cls in all_segmenters().items():
        badge = "" if cls.STATUS == "ready" else f"  [{cls.STATUS}]"
        out.append(f"{k}{badge}")
    return out or ["sam2.1"]


def _matter_choices() -> List[str]:
    out: List[str] = ["none"]
    for k, cls in all_matters().items():
        badge = "" if cls.STATUS == "ready" else f"  [{cls.STATUS}]"
        out.append(f"{k}{badge}")
    return out


def _strip_badge(s: str) -> str:
    return s.split("  [")[0].strip()


def _all_weight_files() -> List[str]:
    """Aggregate weights + presets across every backend.

    Entries are tagged so the JS layer can filter by selected backend:
      * ``(auto)``                              — let the backend pick
      * ``<key>/<filename>``                    — installed local weight
      * ``[preset:<key>] <filename>``           — downloadable preset
    """
    items: List[str] = ["(auto)"]
    seen = {"(auto)"}
    for cls in list(all_segmenters().values()) + list(all_matters().values()):
        key = cls.MODELS_KEY
        if not key:
            continue
        for f in list_backend_files(key):
            tag = f"{key}/{f}"
            if tag not in seen:
                items.append(tag); seen.add(tag)
        for p in list_backend_presets(key):
            tag = f"[preset:{key}] {p['name']}"
            if tag not in seen:
                items.append(tag); seen.add(tag)
    return items


_PRESET_PREFIX = "[preset:"


def _resolve_model_choice(choice: str, expected_key: str,
                          *, auto_download: bool = False) -> str:
    """Resolve a dropdown entry to a concrete weight filename in ``expected_key``.

    Handles three cases:
      * ``(auto)`` / empty → first installed weight for ``expected_key``,
        else first preset (auto-download if allowed).
      * ``<key>/<file>`` → strip prefix; if ``key`` mismatches the expected
        backend, fall back to any installed weight in the expected folder.
      * ``[preset:<key>] <file>`` → download into the backend folder when
        missing and ``auto_download`` is True; raise otherwise.
    """
    if not expected_key:
        return ""

    # Preset entry — bring it down if needed.
    if choice and choice.startswith(_PRESET_PREFIX):
        try:
            tag, name = choice.split("] ", 1)
            preset_key = tag[len(_PRESET_PREFIX):]
        except ValueError:
            preset_key, name = expected_key, ""
        if preset_key != expected_key:
            avail = list_backend_files(expected_key)
            if avail:
                return avail[0]
            # else fall through to download under expected_key (best-effort)
            preset_key = expected_key
        # already on disk?
        avail = list_backend_files(expected_key)
        if name in avail:
            return name
        if not auto_download:
            raise FileNotFoundError(
                f"Preset '{name}' is not installed for '{expected_key}'. "
                f"Tick `auto_download` to fetch it, or place the file in "
                f"models/{expected_key}/ manually."
            )
        download_preset(expected_key, name)
        return name

    if choice == "(auto)" or not choice:
        avail = list_backend_files(expected_key)
        if avail:
            return avail[0]
        # No local weight — auto-download the first preset if allowed.
        presets = list_backend_presets(expected_key)
        if presets and auto_download:
            download_preset(expected_key, presets[0]["name"])
            return presets[0]["name"]
        return ""

    if "/" in choice:
        prefix, _, tail = choice.partition("/")
        if prefix == expected_key:
            return tail
        avail = list_backend_files(expected_key)
        return avail[0] if avail else tail
    return choice


# ══════════════════════════════════════════════════════════════════════
class MaskMattingMEC:
    """Unified segmenter + matter node.

    Pick a ``segmenter`` (e.g. SAM 2.1) and an optional ``matter`` (e.g.
    ViTMatte / RVM). The node auto-detects the prompt mode from the
    inputs you actually wire — points, bbox, text, or video — and routes
    them through the chosen backend.
    """

    CATEGORY = "MaskEditControl/Pipeline"
    DESCRIPTION = (
        "Production-grade segmentation + matting in a single node. "
        "Multi-backend (SAM 2.1 / SAM 3 / SAM 3.1 / BiRefNet / RMBG-2.0 / "
        "InSPyReNet + ViTMatte / RVM / MatAnyone) with VFX post-processing: "
        "TTA flip-fuse, multi-scale ensembling, despill, light wrap, "
        "edge/inside/outside separation, CRF / guided-filter refinement, "
        "no-GT quality scoring, holdout/garbage matte support, and "
        "premultiplied output for compositors."
    )
    FUNCTION = "execute"
    RETURN_TYPES = (
        "MASK", "MASK", "IMAGE", "MASK", "BBOX", "STRING", "FLOAT", "STRING",
        "IMAGE", "IMAGE", "MASK", "MASK", "MASK",
    )
    RETURN_NAMES = (
        "mask", "alpha", "preview", "trimap", "bbox", "bbox_json", "score", "info",
        "despilled", "lightwrap_rgba", "edge_mask", "inside_mask", "outside_mask",
    )
    OUTPUT_TOOLTIPS = (
        "Coarse mask from the segmenter (B,H,W).",
        "Refined alpha after matter + optional CRF / guided refine (production output).",
        "image * alpha premultiplied preview.",
        "Trimap (0/0.5/1) used by the matter.",
        "Tight bbox around the alpha as [x0,y0,x1,y1].",
        "Same bbox as JSON {'x','y','w','h'}.",
        "Overall production-quality score in [0,1] (boundary + coherence + size + smoothness).",
        "JSON: backends, modes, per-frame quality breakdown, settings used.",
        "Image with backing-colour spill suppressed (when `despill_strength`>0).",
        "RGBA light-wrap layer to ADD over the new background.",
        "Soft edge band where matting actually matters.",
        "Solid-fg interior mask (safe to colour-grade).",
        "Solid-bg exterior mask (safe to defocus / replace).",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image or video frames (B,H,W,C)."}),
                "segmenter": (_segmenter_choices(), {
                    "default": _segmenter_choices()[0],
                    "tooltip": "Coarse-mask backend. Entries tagged [experimental] / [missing-deps] won't run yet.",
                }),
                "matter": (_matter_choices(), {
                    "default": "vitmatte" if "vitmatte" in [k for k in list_matter_keys()] else "none",
                    "tooltip": "Optional alpha refinement. 'none' returns the segmenter mask as alpha.",
                }),
                "model": (_all_weight_files(), {
                    "default": "(auto)",
                    "tooltip": "Specific weight file to use. Tag prefix selects the backend folder; '(auto)' lets each backend pick.",
                }),
                "matter_model": (_all_weight_files(), {
                    "default": "(auto)",
                    "tooltip": "Weight file for the matter backend.",
                }),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "attention": (["auto", "sdpa", "flash", "sage", "xformers", "eager"], {"default": "auto"}),
                "offload":   (["none", "cpu", "sequential"], {"default": "none"}),
                "subject_preset": (
                    list(["custom", "hair", "fur", "cloth", "skin_face", "hard_edge", "soft_glow"]),
                    {"default": "custom", "tooltip": "Override trimap_dilate/erode/edge with subject-tuned values."},
                ),
                "trimap_dilate": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "trimap_erode":  ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "edge_radius":   ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "individual_objects": ("BOOLEAN", {"default": False, "tooltip": "If supported by the backend, return one mask per detected object."}),
                "tracking_direction": (["forward", "backward", "bidirectional"], {"default": "forward"}),
                "frame_annotation": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Frame index (in clip) where prompts are anchored."}),
                "object_id":   ("INT", {"default": 0, "min": 0, "max": 1024}),
                "max_frames_to_track": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 = no cap."}),
                "memory_size": ("INT", {"default": 8, "min": 1, "max": 256}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "end_frame":   ("INT", {"default": -1, "min": -1, "max": 100000, "tooltip": "-1 = last frame."}),
                "auto_download": ("BOOLEAN", {"default": False, "tooltip": "Allow lazy auto-download from HF/torch.hub when a weight is missing."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                # ── VFX / production post-processing ────────────────────
                "tta_flip": ("BOOLEAN", {"default": False, "tooltip": "Test-time augmentation: run segmenter on the H-flipped image and average. Slower but cleaner."}),
                "multiscale": ("BOOLEAN", {"default": False, "tooltip": "Run the segmenter at 0.75x / 1.0x / 1.25x and fuse. Helps small / thin subjects."}),
                "post_refine": (["none", "guided", "crf", "crf+guided"], {"default": "none", "tooltip": "Final alpha refinement. 'guided' = guided filter (fast, torch-only). 'crf' = DenseCRF (requires pydensecrf, sharpest edges)."}),
                "refine_radius": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "Spatial radius for guided / CRF refinement."}),
                "refine_iterations": ("INT", {"default": 5, "min": 1, "max": 30, "tooltip": "CRF inference iterations."}),
                "despill": (["off", "green", "blue", "red", "magenta", "cyan", "yellow", "white", "black", "auto"], {"default": "off", "tooltip": "Colour decontamination on the named backing. 'auto' estimates the colour from image corners."}),
                "despill_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "How aggressively to subtract the spill (0 = off)."}),
                "preserve_skin": ("BOOLEAN", {"default": True, "tooltip": "Keep warm pixels (R>G>B) untouched during despill."}),
                "lightwrap_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Light-wrap intensity. 0 = off; ~0.3-0.6 = natural blend over the new BG."}),
                "lightwrap_radius": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "Light-wrap halo radius in pixels."}),
                "edge_band_radius": ("INT", {"default": 4, "min": 1, "max": 64, "tooltip": "Width of the soft edge band when splitting edge/inside/outside masks."}),
                "premultiply": ("BOOLEAN", {"default": True, "tooltip": "Premultiply preview by alpha. Disable for straight-alpha outputs."}),
            },
            "optional": {
                # Slot order shown on the node's left edge: these two come
                # immediately after `image` and BEFORE the bbox slots so the
                # standard wiring from PointsBBoxMaskEditor lines up cleanly.
                # forceInput=True makes them appear as connector slots, not
                # as multiline text widgets at the bottom of the node.
                "positive_coords": ("STRING", {"forceInput": True, "default": "",
                    "tooltip": "JSON list of positive points [[x,y],...] from PointsBBoxMaskEditor (positive_coords output)."}),
                "negative_coords": ("STRING", {"forceInput": True, "default": "",
                    "tooltip": "JSON list of negative points [[x,y],...] from PointsBBoxMaskEditor (negative_coords output)."}),
                "pos_bbox":   ("BBOX", {"tooltip": "Single positive bbox [x0,y0,x1,y1]."}),
                "neg_bbox":   ("BBOX", {"tooltip": "Optional negative bbox (excluded region)."}),
                "normal_bbox": ("BBOX", {"tooltip": "Generic bbox if you don't care about polarity."}),
                "text_prompt": ("STRING", {"forceInput": True, "default": "",
                    "tooltip": "Open-vocabulary text prompt (SAM3 / GroundingDINO / VideoMaMa). Wire from any STRING source."}),
                "external_mask": ("MASK", {"tooltip": "Optional mask used as a hint or overridden when input_mode='auto' falls through."}),
                "external_trimap": ("MASK", {"tooltip": "Optional pre-computed trimap that bypasses internal trimap generation."}),
                "holdout_mask": ("MASK", {"tooltip": "Garbage / holdout matte. Pixels where this is >0 are FORCED to alpha=0 (used to chop out boom mics, rigs, etc)."}),
                "core_mask": ("MASK", {"tooltip": "Core / inside matte. Pixels where this is >0 are FORCED to alpha=1 (used to lock down opaque interiors)."}),
            },
        }

    # ------------------------------------------------------------------
    def _resolve_mode(self, mode: str, B: int, has_pts: bool, has_bbox: bool,
                      has_text: bool, supports: set) -> str:
        if mode != "auto":
            return mode
        # video first if multi-frame and supported
        if B > 1 and "video" in supports:
            return "video"
        if has_text and "text" in supports:
            return "text"
        if has_bbox and "bbox" in supports:
            return "bbox"
        if has_pts and "points" in supports:
            return "points"
        # final fallback
        if "auto" in supports:
            return "auto"
        for fb in ("points", "bbox", "text", "video"):
            if fb in supports:
                return fb
        return "auto"

    # ------------------------------------------------------------------
    def execute(self, image, segmenter, matter, model, matter_model,
                precision, attention, offload, subject_preset,
                trimap_dilate, trimap_erode, edge_radius,
                individual_objects, tracking_direction, frame_annotation,
                object_id, max_frames_to_track, memory_size, start_frame, end_frame,
                auto_download, seed,
                tta_flip=False, multiscale=False, post_refine="none",
                refine_radius=8, refine_iterations=5,
                despill="off", despill_strength=1.0, preserve_skin=True,
                lightwrap_strength=0.0, lightwrap_radius=8,
                edge_band_radius=4, premultiply=True,
                positive_coords="", negative_coords="",
                pos_points="", neg_points="", pos_bbox=None, neg_bbox=None,
                normal_bbox=None, text_prompt="", external_mask=None,
                external_trimap=None, holdout_mask=None, core_mask=None):
        # Merge slot inputs (positive_coords/negative_coords) with the legacy
        # widget inputs (pos_points/neg_points). Slot wins if both supplied.
        pos_points = positive_coords or pos_points or ""
        neg_points = negative_coords or neg_points or ""
        # Always auto: node infers mode (points/bbox/text/video) from wired inputs and B>1.
        input_mode = "auto"
        seg_key = _strip_badge(segmenter)
        mat_key = _strip_badge(matter)
        seg_cls = get_segmenter_cls(seg_key)
        if seg_cls is None:
            raise ValueError(f"Unknown segmenter '{seg_key}'. Choices: {list(all_segmenters())}")
        if seg_cls.STATUS != "ready":
            logger.warning("[MaskMatting] segmenter '%s' is %s — attempting anyway.", seg_key, seg_cls.STATUS)
        mat_cls = None
        if mat_key not in ("none", ""):
            mat_cls = get_matter_cls(mat_key)
            if mat_cls is None:
                raise ValueError(f"Unknown matter '{mat_key}'.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            img_bhwc = to_bhwc(image)
            B, H, W, _ = img_bhwc.shape

            pos_pts, _ = parse_points(pos_points)
            neg_a, neg_b = parse_points(neg_points)
            neg_pts = neg_a + neg_b
            bbox_used = (
                parse_bbox(pos_bbox)
                or parse_bbox(normal_bbox)
            )

            # If a bbox slot is wired but the upstream produced nothing usable,
            # bail loudly. Silent fallthrough was masking misconfigured editors
            # (e.g. a PointsMaskEditor with no box drawn whose 'primary_bbox'
            # output is still connected here).
            bbox_wired = (pos_bbox is not None) or (normal_bbox is not None)
            if bbox_wired and bbox_used is None:
                raise ValueError(
                    "[MaskMatting] A bbox input is connected but no valid box was "
                    "received. Draw a bounding box in the upstream editor "
                    "(or disconnect the bbox link if you only want point/text prompting)."
                )
            mode = self._resolve_mode(
                input_mode, B,
                has_pts=bool(pos_pts or neg_pts),
                has_bbox=bbox_used is not None,
                has_text=bool(text_prompt.strip()),
                supports=seg_cls.SUPPORTS_MODES,
            )

            logger.warning(
                "[MaskMatting] seg=%s matter=%s mode=%s B=%d  "
                "pts(+%d/-%d) bbox=%s text=%r model=%r matter_model=%r ext_mask=%s ext_trimap=%s",
                seg_key, mat_key, mode, B,
                len(pos_pts), len(neg_pts),
                bbox_used, (text_prompt or "")[:32],
                model, matter_model,
                None if external_mask is None else tuple(external_mask.shape),
                None if external_trimap is None else tuple(external_trimap.shape),
            )

            seg_inst = seg_cls(
                model_name=_resolve_model_choice(model, seg_cls.MODELS_KEY,
                                                 auto_download=bool(auto_download)),
                device=device, precision=precision,
                attention=attention, offload=offload,
            )

            def _segment_once(img_in: torch.Tensor) -> torch.Tensor:
                out = seg_inst.segment(
                    img_in, mode=mode,
                    positive_points=pos_pts, negative_points=neg_pts,
                    bbox=bbox_used, text_prompt=text_prompt,
                    frame_annotation=int(frame_annotation), object_id=int(object_id),
                    max_frames=int(max_frames_to_track), memory_size=int(memory_size),
                    start_frame=int(start_frame), end_frame=int(end_frame),
                    individual_objects=bool(individual_objects),
                    tracking_direction=tracking_direction, seed=int(seed),
                )
                return out["mask"].float().clamp(0, 1)

            # First pass — also captures score metadata.
            seg_out = seg_inst.segment(
                img_bhwc, mode=mode,
                positive_points=pos_pts, negative_points=neg_pts,
                bbox=bbox_used, text_prompt=text_prompt,
                frame_annotation=int(frame_annotation), object_id=int(object_id),
                max_frames=int(max_frames_to_track), memory_size=int(memory_size),
                start_frame=int(start_frame), end_frame=int(end_frame),
                individual_objects=bool(individual_objects),
                tracking_direction=tracking_direction, seed=int(seed),
            )
            mask_t: torch.Tensor = seg_out["mask"].float().clamp(0, 1)
            score = float(seg_out.get("score", 1.0))

            # Optional ensembling — fuse first pass with augmented passes.
            if bool(multiscale) and bool(tta_flip):
                fused = _vfx.multiscale_fuse(
                    img_bhwc, lambda x: _vfx.tta_flip_fuse(x, _segment_once))
                mask_t = 0.5 * (mask_t + fused.to(mask_t.device))
            elif bool(tta_flip):
                fused = _vfx.tta_flip_fuse(img_bhwc, _segment_once)
                mask_t = 0.5 * (mask_t + fused.to(mask_t.device))
            elif bool(multiscale):
                fused = _vfx.multiscale_fuse(img_bhwc, _segment_once)
                mask_t = 0.5 * (mask_t + fused.to(mask_t.device))
            logger.warning("[MaskMatting] seg done \u2014 mask sum=%.1f score=%.3f shape=%s",
                           float(mask_t.sum()), score, tuple(mask_t.shape))

            if external_mask is not None:
                em = to_mask(external_mask)
                if em.shape == mask_t.shape:
                    # Skip AND-merge if the external mask is effectively empty
                    # (e.g. an unconfigured SplineMaskEditor with 0 shapes).
                    em_sum = float(em.sum())
                    if em_sum < 1.0:
                        logger.warning(
                            "[MaskMatting] external_mask is empty (sum=%.1f) \u2014 skipping AND-merge",
                            em_sum,
                        )
                    else:
                        # logical AND: refine the user's hint
                        mask_t = torch.minimum(mask_t, em)

            # Trimap
            d, e, edge = apply_subject_preset(subject_preset, int(trimap_dilate), int(trimap_erode), int(edge_radius))
            if external_trimap is not None:
                trimap_t = to_mask(external_trimap)
            else:
                trimap_t = mask_to_trimap(mask_t, dilate=d, erode=e)

            # Matte
            if mat_cls is not None:
                mat_inst = mat_cls(
                    model_name=_resolve_model_choice(matter_model, mat_cls.MODELS_KEY,
                                                     auto_download=bool(auto_download)),
                    device=device, precision=precision,
                    attention=attention, offload=offload,
                )
                mat_out = mat_inst.matte(
                    img_bhwc, mask_t, trimap=trimap_t,
                    edge_radius=edge, memory_size=int(memory_size),
                )
                alpha_t = mat_out["alpha"].float().clamp(0, 1)
            else:
                alpha_t = mask_t

            # ── VFX post-processing pipeline ─────────────────────────
            # 1. Post refinement (CRF / guided filter).
            if post_refine in ("guided", "crf+guided"):
                alpha_t = _vfx.guided_refine(
                    img_bhwc.to(alpha_t.device), alpha_t,
                    radius=int(refine_radius), epsilon=1e-4)
            if post_refine in ("crf", "crf+guided"):
                alpha_t = _vfx.crf_refine(
                    img_bhwc.to(alpha_t.device), alpha_t,
                    iterations=int(refine_iterations))
            # 2. Holdout / core overrides (garbage matte semantics).
            if holdout_mask is not None:
                hm = to_mask(holdout_mask)
                if hm.shape[-2:] == alpha_t.shape[-2:]:
                    if hm.shape[0] != alpha_t.shape[0] and hm.shape[0] == 1:
                        hm = hm.expand_as(alpha_t)
                    alpha_t = (alpha_t * (1.0 - hm.to(alpha_t.device))).clamp(0, 1)
            if core_mask is not None:
                cm = to_mask(core_mask)
                if cm.shape[-2:] == alpha_t.shape[-2:]:
                    if cm.shape[0] != alpha_t.shape[0] and cm.shape[0] == 1:
                        cm = cm.expand_as(alpha_t)
                    alpha_t = torch.maximum(alpha_t, cm.to(alpha_t.device)).clamp(0, 1)
            # 3. Despill on the source image (using the FINAL alpha as mask).
            if despill != "off" and float(despill_strength) > 0:
                despilled = _vfx.despill(
                    img_bhwc.to(alpha_t.device), alpha_t,
                    backing=despill, strength=float(despill_strength),
                    preserve_skin=bool(preserve_skin)).cpu()
            else:
                despilled = img_bhwc.cpu().clone()
            # 4. Light wrap layer.
            if float(lightwrap_strength) > 0:
                lightwrap = _vfx.lightwrap_layer(
                    img_bhwc.to(alpha_t.device), alpha_t,
                    bg_color=None, radius=int(lightwrap_radius),
                    strength=float(lightwrap_strength)).cpu()
            else:
                lightwrap = torch.zeros((*alpha_t.shape, 4),
                                         dtype=img_bhwc.dtype)
            # 5. Edge / inside / outside masks.
            edge_m, inside_m, outside_m = _vfx.edge_inside_outside(
                alpha_t, edge_radius=int(edge_band_radius))
            edge_m, inside_m, outside_m = edge_m.cpu(), inside_m.cpu(), outside_m.cpu()

            # Preview = image * alpha (premultiplied) or straight alpha.
            alpha_cpu = alpha_t.cpu()
            if bool(premultiply):
                preview = (despilled * alpha_cpu.unsqueeze(-1)).clamp(0, 1)
            else:
                preview = despilled.clamp(0, 1)

            # Bbox from alpha (first frame)
            x0, y0, x1, y1 = bbox_from_mask(alpha_cpu[0].numpy())
            bbox_list = [int(x0), int(y0), int(x1), int(y1)]
            bjson = bbox_to_json((x0, y0, x1, y1))

            # 6. Production quality scoring (no GT).
            quality = _vfx.score_quality(img_bhwc, alpha_t.cpu())
            final_score = float(quality["overall"])

            info_obj = {
                "segmenter": seg_key,
                "matter": mat_key,
                "mode": mode,
                "frames": int(B),
                "segmenter_score": score,
                "production_score": final_score,
                "subject_preset": subject_preset,
                "trimap": {"dilate": d, "erode": e, "edge": edge},
                "vfx": {
                    "tta_flip": bool(tta_flip),
                    "multiscale": bool(multiscale),
                    "post_refine": post_refine,
                    "despill": despill,
                    "despill_strength": float(despill_strength),
                    "lightwrap_strength": float(lightwrap_strength),
                    "premultiply": bool(premultiply),
                },
                "quality": quality,
            }
            return (
                mask_t.cpu(),
                alpha_cpu,
                preview,
                trimap_t.cpu(),
                bbox_list,
                bjson,
                final_score,
                json.dumps(info_obj),
                despilled,
                lightwrap,
                edge_m,
                inside_m,
                outside_m,
            )
        finally:
            free_vram()


NODE_CLASS_MAPPINGS = {"MaskMattingMEC": MaskMattingMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskMattingMEC": "Mask + Matting (MEC)"}
