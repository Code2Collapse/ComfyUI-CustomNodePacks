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
        "Multi-backend segmentation + matting in a single node. Supports "
        "SAM 2.1 / SAM 3 / SeC + ViTMatte / RVM / MatAnyone with auto-mode "
        "selection (points / bbox / text / video)."
    )
    FUNCTION = "execute"
    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "MASK", "BBOX", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("mask", "alpha", "preview", "trimap", "bbox", "bbox_json", "score", "info")
    OUTPUT_TOOLTIPS = (
        "Coarse mask straight from the segmenter (B,H,W).",
        "Refined alpha from the matter (= mask if matter='none').",
        "RGB preview = image * alpha (handy debug output).",
        "Trimap fed to the matter (0=bg, 0.5=unknown, 1=fg).",
        "Tight bbox around the alpha as [x0,y0,x1,y1].",
        "Same bbox as JSON {'x','y','w','h'}.",
        "Segmenter confidence score in [0,1].",
        "JSON dict with backend ids, modes used, and per-frame counts.",
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
                positive_coords="", negative_coords="",
                pos_points="", neg_points="", pos_bbox=None, neg_bbox=None,
                normal_bbox=None, text_prompt="", external_mask=None,
                external_trimap=None):
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

            # Preview = image * alpha
            preview = (img_bhwc.cpu() * alpha_t.unsqueeze(-1)).clamp(0, 1)

            # Bbox from alpha (first frame)
            x0, y0, x1, y1 = bbox_from_mask(alpha_t[0].cpu().numpy())
            bbox_list = [int(x0), int(y0), int(x1), int(y1)]
            bjson = bbox_to_json((x0, y0, x1, y1))

            info_obj = {
                "segmenter": seg_key,
                "matter": mat_key,
                "mode": mode,
                "frames": int(B),
                "score": score,
                "subject_preset": subject_preset,
                "trimap": {"dilate": d, "erode": e, "edge": edge},
            }
            return (
                mask_t,
                alpha_t,
                preview,
                trimap_t,
                bbox_list,
                bjson,
                score,
                json.dumps(info_obj),
            )
        finally:
            free_vram()


NODE_CLASS_MAPPINGS = {"MaskMattingMEC": MaskMattingMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskMattingMEC": "Mask + Matting (MEC)"}
