"""MaskOpsMEC — single multi-backend segmentation + matting + refine + diagnostics node.

Absorbs:
    * MaskMattingMEC (segmenter + matter + VFX) — base node
    * MaskRefineMEC (11-stage training-free mask refinement)
    * TrimapGeneratorMEC (advanced edge-aware trimap)
    * LuminanceKeyerMEC (Nuke-style luma key pre-stage)
    * MaskFailureExplainerMEC (post-mortem severity / suggestion diagnostics)

All previous standalone nodes are hard-removed from the ComfyUI registry; their
Python classes remain importable for use as internal helpers.
"""
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
from ._reanchor import (
    DINORelocator,
    blend_masks,
    compute_confidence,
    compute_farneback_flow,
    flow_warp_reanchor,
    release_dinov2,
)
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

logger = logging.getLogger("MEC.MaskOps")


# ──────────────────────────────────────────────────────────────────────
# Lazy helper imports — internal classes that used to be standalone nodes.
# Imported on first use to avoid heavy module-load cost.
# ──────────────────────────────────────────────────────────────────────
def _get_luma_keyer():
    from ..luminance_keyer import LuminanceKeyerMEC
    return LuminanceKeyerMEC()


def _get_trimap_advanced():
    from ..trimap_generator import TrimapGeneratorMEC
    return TrimapGeneratorMEC()


def _get_explainer():
    from ..mask_failure_explainer import MaskFailureExplainerMEC
    return MaskFailureExplainerMEC()


def _segmenter_choices() -> List[str]:
    out: List[str] = ["auto_best", "auto"]  # auto_best = full cascade; auto = single-pick heuristic
    for k, cls in all_segmenters().items():
        badge = "" if cls.STATUS == "ready" else f"  [{cls.STATUS}]"
        out.append(f"{k}{badge}")
    return out or ["sam2.1"]


def _matter_choices() -> List[str]:
    out: List[str] = ["none", "auto"]
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
                items.append(tag)
                seen.add(tag)
        for p in list_backend_presets(key):
            tag = f"[preset:{key}] {p['name']}"
            if tag not in seen:
                items.append(tag)
                seen.add(tag)
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


def _la_sam_text_grounding(
    seg_image: torch.Tensor,
    text_prompt: str,
    avail_c: set,
    *,
    device: str,
    precision: str,
    attention: str,
    offload: str,
    auto_download: bool,
    seed: int,
) -> tuple:
    """LocateAnything-3B → SAM bbox refine (training-free, text-only).

    Tries every detection box and keeps the highest-scoring SAM mask.
    Returns ``(mask | None, score, trail_dict)``.
    """
    prompt = (text_prompt or "").strip()
    if not prompt or "locate_anything" not in avail_c:
        return None, 0.0, {}
    sam_key = None
    for k in ("sam2.1", "sam3", "sam2"):
        if k in avail_c:
            sam_key = k
            break
    if sam_key is None:
        return None, 0.0, {}

    la_cls = get_segmenter_cls("locate_anything")
    sam_cls = get_segmenter_cls(sam_key)
    if not la_cls or la_cls.STATUS != "ready" or not sam_cls or sam_cls.STATUS != "ready":
        return None, 0.0, {}

    try:
        la_inst = la_cls(device=device, precision=precision)
        la_out = la_inst.segment(seg_image, mode="text", text_prompt=prompt, seed=int(seed))
        bboxes = la_out.get("bboxes") or []
        if not bboxes:
            la_mask = la_out.get("mask")
            la_score = float(la_out.get("score", 0.0))
            if la_mask is not None and la_score > 0.0:
                return la_mask.float().clamp(0, 1), la_score, {
                    "backend": "locate_anything",
                    "score": la_score,
                    "bboxes_found": 0,
                }
            return None, 0.0, {}

        sam_inst = sam_cls(
            model_name=_resolve_model_choice("(auto)", sam_cls.MODELS_KEY,
                                             auto_download=bool(auto_download)),
            device=device, precision=precision,
            attention=attention, offload=offload,
        )
        best_mask = None
        best_score = 0.0
        H, W = int(seg_image.shape[1]), int(seg_image.shape[2])
        for box in bboxes:
            x1 = max(0, min(W - 1, int(round(box["x1"]))))
            y1 = max(0, min(H - 1, int(round(box["y1"]))))
            x2 = max(0, min(W, int(round(box["x2"]))))
            y2 = max(0, min(H, int(round(box["y2"]))))
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue
            r_out = sam_inst.segment(
                seg_image, mode="bbox",
                bbox=(x1, y1, x2, y2), seed=int(seed),
            )
            r_mask = r_out["mask"].float().clamp(0, 1)
            r_score = float(r_out.get("score", 0.0))
            if r_score > best_score:
                best_mask, best_score = r_mask, r_score

        if best_mask is not None:
            return best_mask, best_score, {
                "backend": f"locate_anything→{sam_key}",
                "score": best_score,
                "bboxes_found": len(bboxes),
            }
        la_mask = la_out.get("mask")
        if la_mask is not None:
            return la_mask.float().clamp(0, 1), float(la_out.get("score", 0.0)), {
                "backend": "locate_anything",
                "score": float(la_out.get("score", 0.0)),
                "bboxes_found": len(bboxes),
            }
    except Exception as exc:
        logger.warning("[MaskMatting] LA→SAM text grounding failed: %s", exc)
    return None, 0.0, {}


# ══════════════════════════════════════════════════════════════════════
class MaskOpsMEC:
    """Unified segmenter + matter + refine + diagnose node.

    Pick a ``segmenter`` (e.g. SAM 2.1) and an optional ``matter`` (e.g.
    ViTMatte / RVM). The node auto-detects the prompt mode from the
    inputs you actually wire — points, bbox, text, or video — and routes
    them through the chosen backend. Optional pre-stage luminance key,
    advanced edge-aware trimap, automatic quality-handling (CLAHE /
    unsharp / denoise / chroma stretch for hard images + guided-filter
    alpha polish), and automatic failure diagnostics are bolted into
    the same node, so the entire mask-creation pipeline lives behind
    one socket. For manual 11-stage refinement, chain MaskRefineMEC
    downstream.
    """

    CATEGORY = "MaskEditControl/Pipeline"
    DESCRIPTION = (
        "Production-grade segmentation + matting + AUTO-QUALITY + "
        "diagnostics in one node. Multi-backend (SAM 2.1 / SAM 3 / "
        "SAM 3.1 / BiRefNet / RMBG-2.0 / InSPyReNet + ViTMatte / RVM "
        "/ MatAnyone). Optional Nuke-style luma-key pre-stage, edge-"
        "aware trimap, and AUTO-QUALITY pipeline that detects motion "
        "blur, low light, low contrast, speckle noise and similar bg/"
        "fg, then applies just-enough pre-processing before the "
        "segmenter and a light guided-filter polish on the alpha — no "
        "knobs to tune. When you supply pos+neg points (e.g. pos on "
        "face, neg on neck) it picks the SAM candidate that excludes "
        "the neg points, so you get the face only, not the whole "
        "person. For manual 11-stage refinement (hole-fill, joint "
        "bilateral, DenseCRF, etc.) chain MaskRefineMEC downstream."
    )
    FUNCTION = "execute"
    RETURN_TYPES = (
        "MASK", "MASK", "IMAGE", "MASK", "BBOX", "STRING", "FLOAT", "STRING",
        "IMAGE", "IMAGE", "MASK", "MASK", "MASK",
        # NEW outputs (refine + diagnose + luma-key debug)
        "MASK", "MASK", "FLOAT", "STRING",
    )
    RETURN_NAMES = (
        "mask", "alpha", "preview", "trimap", "bbox", "bbox_json", "score", "info",
        "despilled", "lightwrap_rgba", "edge_mask", "inside_mask", "outside_mask",
        "luma_key_mask", "problem_regions", "severity", "suggested_method",
    )
    OUTPUT_TOOLTIPS = (
        "Coarse mask from the segmenter (B,H,W).",
        "Refined alpha after matter + refinement (production output).",
        "image * alpha premultiplied preview.",
        "Trimap (0/0.5/1) used by the matter.",
        "Tight bbox around the alpha as [x0,y0,x1,y1].",
        "Same bbox as JSON {'x','y','w','h'}.",
        "Overall production-quality score in [0,1] (boundary + coherence + size + smoothness).",
        "JSON: backends, modes, per-frame quality breakdown, refine stages run, settings used.",
        "Image with backing-colour spill suppressed (when `despill_strength`>0).",
        "RGBA light-wrap layer to ADD over the new background.",
        "Soft edge band where matting actually matters.",
        "Solid-fg interior mask (safe to colour-grade).",
        "Solid-bg exterior mask (safe to defocus / replace).",
        "Luminance keyer output (empty if `enable_luma_key` is off).",
        "Diagnostic problem-region heatmap from the failure explainer.",
        "Severity score [0,1] from the failure explainer.",
        "Suggested next masking method (string) from the failure explainer.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image or video frames (B,H,W,C)."}),
                "segmenter": (_segmenter_choices(), {
                    "default": _segmenter_choices()[0],
                    "tooltip": "Coarse-mask backend. Entries tagged [missing-deps] need an optional pip install to activate.",
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

                # ── Luma-key pre-stage (Nuke-style LumaKeyer) ───────────
                "enable_luma_key": ("BOOLEAN", {"default": False, "tooltip": "Run a luminance keyer on the source image BEFORE segmentation and use it as a hint / external_mask."}),
                "luma_mode": (["auto", "highlights", "midtones", "shadows", "custom"], {"default": "auto"}),
                "luma_low":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "luma_high":  ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "luma_gamma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "luma_falloff": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "luma_invert": ("BOOLEAN", {"default": False}),
                "luma_mix": (["intersect", "union", "replace", "hint_only"], {"default": "hint_only",
                    "tooltip": "How to combine the luma-key mask with the segmenter result. 'hint_only' = use as external_mask hint; 'intersect/union/replace' = combine with the final alpha."}),

                # ── Advanced edge-aware trimap (Trimap Generator) ───────
                "enable_advanced_trimap": ("BOOLEAN", {"default": False, "tooltip": "Use the edge-aware trimap generator (asymmetric inner/outer scaling, image-edge snapping, smoothing) instead of the simple dilate/erode trimap."}),
                "trimap_inner_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "trimap_outer_scale": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1}),
                "trimap_smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "trimap_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),

                # ── Auto-Quality (zero-knob hard-case handling) ─────────
                # Replaces the old 25-widget refine cluster.  Power users
                # who want manual refinement should chain MaskRefineMEC.
                "auto_quality": ("BOOLEAN", {"default": True, "tooltip":
                    "AUTOMATIC robustness for hard images. Detects motion "
                    "blur, low light, low contrast, speckle noise and "
                    "low bg/fg colour separation, then applies just-enough "
                    "pre-processing (CLAHE, unsharp, NL-means, chroma "
                    "stretch) BEFORE the segmenter, and a light guided-"
                    "filter+edge-snap polish on the alpha. No knobs."}),
                "auto_disambiguate": ("BOOLEAN", {"default": True, "tooltip":
                    "When BOTH positive and negative points are supplied, "
                    "score SAM's 3 candidate masks by (pos-coverage − "
                    "neg-coverage − size-penalty) instead of raw score. "
                    "This is what makes `pos=face, neg=neck` return just "
                    "the face, not the whole person."}),
                "quality_mode": (["fast", "balanced", "max_fidelity"],
                    {"default": "balanced", "tooltip":
                        "Strength of auto_quality pre/post processing. "
                        "fast = mild, balanced = default, max_fidelity = "
                        "NL-means denoise + larger guided filter."}),

                # ── Failure diagnostics ─────────────────────────────────
                "enable_diagnose": ("BOOLEAN", {"default": True, "tooltip": "Run automatic mask-failure diagnostics (severity score + suggested method)."}),
                "diag_ring_width": ("INT", {"default": 5, "min": 1, "max": 50}),
                "diag_blur_threshold": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 1000.0, "step": 1.0}),
                "diag_brightness_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),

                # ── Robust propagation (video mode) ─────────────────────
                # Confidence-aware re-anchor loop. Survives motion blur,
                # lighting drift, gamma/gain mismatch, occlusion. Only
                # active when B>1 AND segmenter ran in video mode.
                "robust_propagation": ("BOOLEAN", {"default": False, "tooltip":
                    "Confidence-aware re-anchor loop for video. After SAM2 "
                    "propagation, every frame's mask is scored vs the last "
                    "good mask (IoU \u00d7 size-ratio). When confidence "
                    "drops below the threshold, the chosen re-anchor "
                    "strategy fires (flow warp / DINOv2 region search / "
                    "convex blend). Robust against motion blur + lighting "
                    "drift. Single-image inputs skip this stage."}),
                "robust_confidence_threshold": ("FLOAT", {
                    "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Below this confidence the re-anchor fires."}),
                "robust_reanchor_method": (["blend", "flow", "dino", "none"],
                    {"default": "blend", "tooltip":
                        "flow=Farneback optical-flow warp of last good mask. "
                        "dino=DINOv2 patch-feature region search + SAM2 "
                        "re-prompt. blend=convex mix of current+warped. "
                        "none=accept drifted output (debug)."}),
                "robust_blend_alpha": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend weight for current SAM2 mask when method=blend."}),
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
                # NEW: luma key
                enable_luma_key=False, luma_mode="auto",
                luma_low=0.0, luma_high=1.0, luma_gamma=1.0,
                luma_falloff=1.0, luma_invert=False, luma_mix="hint_only",
                # NEW: advanced trimap
                enable_advanced_trimap=False,
                trimap_inner_scale=1.0, trimap_outer_scale=1.5,
                trimap_smooth=0.0, trimap_threshold=0.5,
                # NEW: auto-quality (replaces the old refine_* cluster)
                auto_quality=True, auto_disambiguate=True,
                quality_mode="balanced",
                # NEW: diagnose
                enable_diagnose=True, diag_ring_width=5,
                diag_blur_threshold=50.0, diag_brightness_threshold=0.15,
                # NEW: robust propagation (folded-in pipeline)
                robust_propagation=False,
                robust_confidence_threshold=0.65,
                robust_reanchor_method="blend",
                robust_blend_alpha=0.7,
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

        # ── auto_route: pick best segmenter/matter from image stats ────
        # Triggered when user picks "auto" (added to the choices list)
        # OR auto_quality=True AND segmenter="sam2.1" (default) AND no
        # explicit user choice override.  Heuristic:
        #   text prompt present                  → sam3 (open-vocab)
        #   B>1 (video)                          → sam2.1 (video propagation)
        #   boundary_ambig>0.6 or hair preset    → vitmatte + birefnet
        #   speckle_score>0.5 + B==1             → sam3 (more robust)
        #   else                                 → sam2.1 (general best)
        # Auto-route metadata gets baked into the `info` JSON.
        _auto_routed = False
        _auto_route_reasons: list[str] = []
        _cascade_trail: list = []
        # ── auto_best: full cascade (SAM2.1 + SAM3 ensemble → SeC video
        # fallback → BiRefNet salient fallback → trimap → ViTMatte +
        # guided polish + CLAHE pre-stage + motion-blur temporal median).
        _cascade_mode = False
        if seg_key == "auto_best":
            _cascade_mode = True
            _auto_routed = True
            _avail0 = set(all_segmenters().keys())
            # primary = SAM2.1 (best general / only sam with video memory).
            if "sam2.1" in _avail0:
                seg_key = "sam2.1"
            elif "sam3" in _avail0:
                seg_key = "sam3"
            elif "birefnet" in _avail0:
                seg_key = "birefnet"
            else:
                seg_key = next(iter(_avail0), "sam2.1")
            _auto_route_reasons.append(f"auto_best_primary→{seg_key}")
            # auto_best implies the production-quality bundle.
            auto_quality = True
            enable_advanced_trimap = True
            if mat_key in ("none", ""):
                mat_key = "auto"
            # post_refine guided polish unless user explicitly chose crf.
            if post_refine == "none":
                post_refine = "guided"

        if seg_key == "auto":
            _auto_routed = True
            try:
                from ._auto_quality import analyze_image as _ar_analyze
                _stats = _ar_analyze(to_bhwc(image))
            except Exception:
                _stats = {"boundary_ambig": 0.0, "speckle_score": 0.0,
                          "blur_score": 0.0, "lowlight_score": 0.0}
            _avail = set(all_segmenters().keys())
            _has_text = bool(text_prompt and text_prompt.strip())
            _B = int(image.shape[0]) if hasattr(image, "shape") else 1
            if _has_text and "sam3" in _avail:
                seg_key = "sam3"
                _auto_route_reasons.append("text_prompt→sam3")
            elif _B > 1 and "sam2.1" in _avail:
                seg_key = "sam2.1"
                _auto_route_reasons.append(f"video(B={_B})→sam2.1")
            elif _stats.get("boundary_ambig", 0) > 0.6 and "birefnet" in _avail:
                seg_key = "birefnet"
                _auto_route_reasons.append(
                    f"boundary_ambig={_stats['boundary_ambig']:.2f}→birefnet"
                )
            elif _stats.get("speckle_score", 0) > 0.5 and "sam3" in _avail:
                seg_key = "sam3"
                _auto_route_reasons.append(
                    f"speckle={_stats['speckle_score']:.2f}→sam3"
                )
            else:
                # General best: prefer sam2.1, then sam3, then first ready.
                for _pref in ("sam2.1", "sam3", "rmbg2", "birefnet"):
                    if _pref in _avail:
                        seg_key = _pref
                        _auto_route_reasons.append(f"default→{_pref}")
                        break
                else:
                    seg_key = next(iter(_avail), "sam2.1")
                    _auto_route_reasons.append(f"fallback→{seg_key}")
            logger.info(
                "[MaskMatting] auto_route picked segmenter=%s (reasons: %s)",
                seg_key, "; ".join(_auto_route_reasons),
            )

        if mat_key == "auto":
            _auto_routed = True
            _avail_m = set(all_matters().keys())
            # If we used a salient segmenter (birefnet/rmbg2) the alpha
            # is already a soft matte → "none". Otherwise ViTMatte wins
            # for hair/fur/cloth; matanyone for video; bgmattingv2 last.
            if seg_key in ("birefnet", "rmbg2"):
                mat_key = "none"
                _auto_route_reasons.append("salient_seg→matter=none")
            else:
                try:
                    _B2 = int(image.shape[0]) if hasattr(image, "shape") else 1
                except Exception:
                    _B2 = 1
                if _B2 > 1 and "matanyone" in _avail_m:
                    mat_key = "matanyone"
                    _auto_route_reasons.append(f"video(B={_B2})→matanyone")
                elif "vitmatte" in _avail_m:
                    mat_key = "vitmatte"
                    _auto_route_reasons.append("default→vitmatte")
                else:
                    mat_key = "none"
                    _auto_route_reasons.append("no_matter_available→none")
            logger.info(
                "[MaskMatting] auto_route picked matter=%s (reasons: %s)",
                mat_key, "; ".join(_auto_route_reasons),
            )

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

            # ── Auto-Quality: detect hard-case issues + preprocess ─────
            # Original `img_bhwc` is preserved for matting + preview;
            # only the segmenter sees the cleaned copy.
            auto_q_info: Dict[str, Any] = {"enabled": bool(auto_quality)}
            seg_image = img_bhwc
            if bool(auto_quality):
                try:
                    from ._auto_quality import (
                        analyze_image as _analyze,
                        preprocess_for_segmentation as _preproc,
                    )
                    issues = _analyze(img_bhwc)
                    seg_image, pre_steps = _preproc(
                        img_bhwc, issues, quality_mode=quality_mode,
                    )
                    auto_q_info.update({
                        "issues": issues,
                        "pre_steps": pre_steps,
                        "mode": quality_mode,
                    })
                except Exception as _e:
                    logger.warning("[MaskOps] auto preprocess failed: %s", _e)
                    seg_image = img_bhwc

            # ── Luma-key pre-stage ────────────────────────────────────
            luma_mask_t: Optional[torch.Tensor] = None
            if bool(enable_luma_key):
                try:
                    keyer = _get_luma_keyer()
                    luma_mask_t, _luma_info = keyer._key_luminance_impl(
                        img_bhwc, luma_mode,
                        float(luma_low), float(luma_high),
                        float(luma_gamma), float(luma_falloff),
                        bool(luma_invert),
                    )
                    # Use as hint when nothing else is wired.
                    if external_mask is None and luma_mix == "hint_only":
                        external_mask = luma_mask_t
                except Exception as _e:
                    logger.warning("[MaskOps] luma keyer failed: %s", _e)
                    luma_mask_t = None

            pos_pts, _ = parse_points(pos_points)
            neg_a, neg_b = parse_points(neg_points)
            neg_pts = neg_a + neg_b
            bbox_used = (
                parse_bbox(pos_bbox)
                or parse_bbox(normal_bbox)
            )
            neg_bbox_used = parse_bbox(neg_bbox)

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
                "pts(+%d/-%d) bbox=%s neg_bbox=%s text=%r model=%r matter_model=%r ext_mask=%s ext_trimap=%s",
                seg_key, mat_key, mode, B,
                len(pos_pts), len(neg_pts),
                bbox_used, neg_bbox_used, (text_prompt or "")[:32],
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
            # Tell the backend whether to use smart pos/neg disambiguation
            # when SAM returns multiple candidate masks.
            try:
                setattr(seg_inst, "auto_disambiguate", bool(auto_disambiguate))
            except Exception:
                pass

            def _segment_once(img_in: torch.Tensor) -> torch.Tensor:
                out = seg_inst.segment(
                    img_in, mode=mode,
                    positive_points=pos_pts, negative_points=neg_pts,
                    bbox=bbox_used, neg_bbox=neg_bbox_used,
                    text_prompt=text_prompt,
                    frame_annotation=int(frame_annotation), object_id=int(object_id),
                    max_frames=int(max_frames_to_track), memory_size=int(memory_size),
                    start_frame=int(start_frame), end_frame=int(end_frame),
                    individual_objects=bool(individual_objects),
                    tracking_direction=tracking_direction, seed=int(seed),
                )
                return out["mask"].float().clamp(0, 1)

            # Text-only / text-primary: LocateAnything → SAM bbox refine
            # runs first (no training). Beats raw SAM text mode on open-vocab.
            _la_done = False
            if (text_prompt and text_prompt.strip()
                    and B == 1 and not pos_pts and not neg_pts
                    and bbox_used is None):
                _avail_la = set(all_segmenters().keys())
                m_la, s_la, tr_la = _la_sam_text_grounding(
                    seg_image, text_prompt, _avail_la,
                    device=device, precision=precision,
                    attention=attention, offload=offload,
                    auto_download=bool(auto_download), seed=int(seed),
                )
                if m_la is not None and s_la >= 0.20:
                    mask_t = m_la
                    score = s_la
                    _la_done = True
                    if tr_la:
                        _cascade_trail.append(tr_la)
                        _auto_route_reasons.append(
                            f"text_grounding:{tr_la.get('backend', 'la')}({s_la:.2f})")

            # First pass — also captures score metadata.
            if not _la_done:
                seg_out = seg_inst.segment(
                seg_image, mode=mode,
                positive_points=pos_pts, negative_points=neg_pts,
                bbox=bbox_used, neg_bbox=neg_bbox_used,
                text_prompt=text_prompt,
                frame_annotation=int(frame_annotation), object_id=int(object_id),
                max_frames=int(max_frames_to_track), memory_size=int(memory_size),
                start_frame=int(start_frame), end_frame=int(end_frame),
                individual_objects=bool(individual_objects),
                tracking_direction=tracking_direction, seed=int(seed),
                )
                mask_t = seg_out["mask"].float().clamp(0, 1)
                score = float(seg_out.get("score", 1.0))

            # ── auto_best cascade: ensemble + fallback ────────────────
            # Primary already ran above (typically sam2.1). For B==1 we
            # also run sam3 (if available) and pick the higher-scoring
            # mask; if BOTH score below 0.30 we drop to BiRefNet salient.
            # For B>1 we keep sam2.1's video-memory mask; if its score
            # is below 0.40 we fall back to SeC (memory-based propagation).
            if _cascade_mode:
                _avail_c = set(all_segmenters().keys())
                def _run_alt(alt_key: str) -> tuple:
                    alt_cls = get_segmenter_cls(alt_key)
                    if alt_cls is None or alt_cls.STATUS != "ready":
                        return None, 0.0
                    alt_mode = self._resolve_mode(
                        "auto", B,
                        has_pts=bool(pos_pts or neg_pts),
                        has_bbox=bbox_used is not None,
                        has_text=bool(text_prompt.strip()),
                        supports=alt_cls.SUPPORTS_MODES,
                    )
                    try:
                        alt_inst = alt_cls(
                            model_name=_resolve_model_choice(
                                "(auto)", alt_cls.MODELS_KEY,
                                auto_download=bool(auto_download)),
                            device=device, precision=precision,
                            attention=attention, offload=offload,
                        )
                        try:
                            setattr(alt_inst, "auto_disambiguate",
                                    bool(auto_disambiguate))
                        except Exception:
                            pass
                        alt_out = alt_inst.segment(
                            seg_image, mode=alt_mode,
                            positive_points=pos_pts, negative_points=neg_pts,
                            bbox=bbox_used, neg_bbox=neg_bbox_used,
                            text_prompt=text_prompt,
                            frame_annotation=int(frame_annotation),
                            object_id=int(object_id),
                            max_frames=int(max_frames_to_track),
                            memory_size=int(memory_size),
                            start_frame=int(start_frame),
                            end_frame=int(end_frame),
                            individual_objects=bool(individual_objects),
                            tracking_direction=tracking_direction,
                            seed=int(seed),
                        )
                        return (alt_out["mask"].float().clamp(0, 1),
                                float(alt_out.get("score", 0.0)))
                    except Exception as _exc:
                        logger.warning("[MaskMatting] auto_best alt '%s' failed: %s",
                                       alt_key, _exc)
                        return None, 0.0

                # LocateAnything → SAM: upgrade mask when text prompt + cascade
                # (skipped if text_grounding already ran as primary).
                if (text_prompt and text_prompt.strip()
                        and not _la_done and B == 1):
                    m_la, s_la, tr_la = _la_sam_text_grounding(
                        seg_image, text_prompt, _avail_c,
                        device=device, precision=precision,
                        attention=attention, offload=offload,
                        auto_download=bool(auto_download), seed=int(seed),
                    )
                    if m_la is not None and s_la > score:
                        mask_t, score = m_la, s_la
                        if tr_la:
                            _cascade_trail.append(tr_la)
                            _auto_route_reasons.append(
                                f"cascade:{tr_la.get('backend', 'la')}({s_la:.2f})")

                if B == 1 and "sam3" in _avail_c and seg_key != "sam3":
                    m2, s2 = _run_alt("sam3")
                    _cascade_trail.append({"backend": "sam3", "score": s2})
                    if m2 is not None and m2.shape == mask_t.shape:
                        if s2 > score and s2 >= 0.30:
                            prev_score = score
                            mask_t, score = m2, s2
                            _auto_route_reasons.append(
                                f"cascade:sam3({s2:.2f})>primary({prev_score:.2f})")
                        elif score >= 0.50 and s2 >= 0.50:
                            # both confident → union (max) for hair / thin edges.
                            mask_t = torch.maximum(mask_t, m2.to(mask_t.device))
                            _auto_route_reasons.append(
                                f"cascade:union(sam2.1,sam3)")

                if score < 0.30:
                    if B > 1 and "sec" in _avail_c:
                        m3, s3 = _run_alt("sec")
                        _cascade_trail.append({"backend": "sec", "score": s3})
                        if m3 is not None and m3.shape == mask_t.shape and s3 > score:
                            mask_t, score = m3, s3
                            seg_key = "sec"
                            _auto_route_reasons.append(
                                f"cascade:fallback→sec({s3:.2f})")
                    elif B == 1 and "birefnet" in _avail_c:
                        m3, s3 = _run_alt("birefnet")
                        _cascade_trail.append({"backend": "birefnet", "score": s3})
                        if m3 is not None and m3.shape == mask_t.shape and s3 > score:
                            mask_t, score = m3, s3
                            seg_key = "birefnet"
                            _auto_route_reasons.append(
                                f"cascade:fallback→birefnet({s3:.2f})")
                elif B > 1 and score < 0.40 and "sec" in _avail_c:
                    m3, s3 = _run_alt("sec")
                    _cascade_trail.append({"backend": "sec", "score": s3})
                    if m3 is not None and m3.shape == mask_t.shape and s3 > score:
                        mask_t, score = m3, s3
                        seg_key = "sec"
                        _auto_route_reasons.append(
                            f"cascade:video_fallback→sec({s3:.2f})")

            # Optional ensembling — fuse first pass with augmented passes.
            if bool(multiscale) and bool(tta_flip):
                fused = _vfx.multiscale_fuse(
                    seg_image, lambda x: _vfx.tta_flip_fuse(x, _segment_once))
                mask_t = 0.5 * (mask_t + fused.to(mask_t.device))
            elif bool(tta_flip):
                fused = _vfx.tta_flip_fuse(seg_image, _segment_once)
                mask_t = 0.5 * (mask_t + fused.to(mask_t.device))
            elif bool(multiscale):
                fused = _vfx.multiscale_fuse(seg_image, _segment_once)
                mask_t = 0.5 * (mask_t + fused.to(mask_t.device))
            logger.warning("[MaskMatting] seg done \u2014 mask sum=%.1f score=%.3f shape=%s",
                           float(mask_t.sum()), score, tuple(mask_t.shape))

            # ── Robust propagation: confidence-aware re-anchor ────────
            # Only meaningful for video (B>1) with a single seed frame.
            robust_info: Dict[str, Any] = {"enabled": bool(robust_propagation)}
            if bool(robust_propagation) and mask_t.ndim == 3 and mask_t.shape[0] > 1:
                try:
                    method = str(robust_reanchor_method).lower()
                    conf_thr = float(robust_confidence_threshold)
                    blend_w = float(robust_blend_alpha)
                    src_idx = max(0, min(int(frame_annotation), mask_t.shape[0] - 1))
                    confidences: List[float] = [1.0] * mask_t.shape[0]
                    events: List[Dict[str, Any]] = []
                    last_good = mask_t[src_idx].clone()
                    last_good_idx = src_idx
                    dino = None
                    if method == "dino":
                        try:
                            dino = DINORelocator(device=device)
                            dino.encode_reference(seg_image[src_idx], last_good)
                        except Exception as _e:
                            logger.warning("[MaskOps] dino init failed: %s — falling back to flow", _e)
                            method = "flow"
                    for t in range(mask_t.shape[0]):
                        if t == src_idx:
                            continue
                        conf = compute_confidence(mask_t[t], last_good)
                        confidences[t] = conf
                        if conf >= conf_thr:
                            last_good = mask_t[t].clone()
                            last_good_idx = t
                            continue
                        # Re-anchor
                        if method == "none":
                            events.append({"frame": t, "conf": conf, "action": "skip"})
                            continue
                        if method == "flow":
                            try:
                                flow = compute_farneback_flow(
                                    seg_image[last_good_idx], seg_image[t])
                                warped = flow_warp_reanchor(last_good, flow)
                                mask_t[t] = warped.to(mask_t.device).clamp(0, 1)
                                events.append({"frame": t, "conf": conf, "action": "flow"})
                            except Exception as _e:
                                events.append({"frame": t, "conf": conf, "action": f"flow_err:{_e}"})
                        elif method == "blend":
                            try:
                                flow = compute_farneback_flow(
                                    seg_image[last_good_idx], seg_image[t])
                                warped = flow_warp_reanchor(last_good, flow).to(mask_t.device)
                                mask_t[t] = blend_masks(mask_t[t], warped, alpha=blend_w).clamp(0, 1)
                                events.append({"frame": t, "conf": conf, "action": "blend"})
                            except Exception as _e:
                                events.append({"frame": t, "conf": conf, "action": f"blend_err:{_e}"})
                        elif method == "dino" and dino is not None:
                            try:
                                x0, y0, x1, y1 = dino.find_best_region(seg_image[t])
                                # Re-prompt SAM2 single-frame on this region
                                alt_out = seg_inst.segment(
                                    seg_image[t:t+1], mode="bbox",
                                    positive_points=[], negative_points=[],
                                    bbox=[x0, y0, x1, y1], neg_bbox=None,
                                    text_prompt="",
                                    frame_annotation=0, object_id=int(object_id),
                                    max_frames=0, memory_size=int(memory_size),
                                    start_frame=0, end_frame=0,
                                    individual_objects=False,
                                    tracking_direction="forward", seed=int(seed),
                                )
                                alt_m = alt_out["mask"].float().clamp(0, 1).to(mask_t.device)
                                if alt_m.ndim == 3 and alt_m.shape[0] >= 1:
                                    mask_t[t] = alt_m[0]
                                events.append({"frame": t, "conf": conf, "action": "dino",
                                               "bbox": [x0, y0, x1, y1]})
                            except Exception as _e:
                                events.append({"frame": t, "conf": conf, "action": f"dino_err:{_e}"})
                        # Update last_good if re-anchored mask is now consistent.
                        new_conf = compute_confidence(mask_t[t], last_good)
                        if new_conf >= conf_thr:
                            last_good = mask_t[t].clone()
                            last_good_idx = t
                    mean_conf = float(sum(confidences) / max(len(confidences), 1))
                    robust_info.update({
                        "method": method,
                        "threshold": conf_thr,
                        "blend_alpha": blend_w,
                        "mean_confidence": mean_conf,
                        "events": events[:64],   # cap for JSON size
                        "n_events": len(events),
                    })
                    if method == "dino":
                        try:
                            release_dinov2()
                        except Exception:
                            pass
                except Exception as _e:
                    logger.warning("[MaskOps] robust_propagation failed: %s", _e)
                    robust_info["error"] = str(_e)

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
            elif bool(enable_advanced_trimap):
                # Edge-aware trimap via the absorbed TrimapGeneratorMEC.
                try:
                    tg = _get_trimap_advanced()
                    trimap_t, _fg_m, _unk_m = tg.generate(
                        mask_t,
                        edge_radius=int(edge if edge > 0 else 15),
                        inner_erosion=float(trimap_inner_scale),
                        outer_dilation=float(trimap_outer_scale),
                        smooth=float(trimap_smooth),
                        threshold=float(trimap_threshold),
                        image=img_bhwc,
                    )
                except Exception as _e:
                    logger.warning("[MaskOps] advanced trimap failed (%s) — falling back to simple.", _e)
                    trimap_t = mask_to_trimap(mask_t, dilate=d, erode=e)
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
            # 1.5. AUTO-QUALITY post-matting polish (replaces the
            # old 25-widget refine cluster).  Power users who want a
            # full 11-stage refinement chain should connect a
            # MaskRefineMEC node downstream.
            auto_q_steps_post: list = []
            if bool(auto_quality):
                try:
                    from ._auto_quality import polish_alpha as _polish
                    alpha_t = _polish(img_bhwc, alpha_t,
                                       quality_mode=quality_mode)
                    auto_q_steps_post.append(f"polish({quality_mode})")
                except Exception as _e:
                    logger.warning("[MaskOps] auto polish failed: %s", _e)
            # 1.55. auto_best motion-blur temporal median (B>=3 only).
            if _cascade_mode and alpha_t.ndim == 3 and alpha_t.shape[0] >= 3:
                try:
                    from ._auto_quality import temporal_alpha_median as _tmed
                    alpha_t = _tmed(alpha_t)
                    auto_q_steps_post.append("temporal_median(3-tap)")
                except Exception as _e:
                    logger.warning("[MaskOps] temporal median failed: %s", _e)
            # 1.6. Luma-key combine (intersect/union/replace post-segmentation).
            if luma_mask_t is not None and luma_mix in ("intersect", "union", "replace"):
                lm = luma_mask_t.to(alpha_t.device).float().clamp(0, 1)
                if lm.shape[0] != alpha_t.shape[0] and lm.shape[0] == 1:
                    lm = lm.expand_as(alpha_t)
                if lm.shape[-2:] == alpha_t.shape[-2:]:
                    if luma_mix == "intersect":
                        alpha_t = torch.minimum(alpha_t, lm)
                    elif luma_mix == "union":
                        alpha_t = torch.maximum(alpha_t, lm)
                    elif luma_mix == "replace":
                        alpha_t = lm
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

            # 7. NEW: failure diagnostics.
            explanation_str = ""
            problem_heatmap = torch.zeros_like(alpha_cpu)
            severity_val = 0.0
            suggested = ""
            if bool(enable_diagnose):
                try:
                    explainer = _get_explainer()
                    explanation_str, problem_heatmap, severity_val, suggested = \
                        explainer._analyze_impl(
                            img_bhwc, alpha_t,
                            int(diag_ring_width),
                            float(diag_blur_threshold),
                            float(diag_brightness_threshold),
                        )
                    problem_heatmap = problem_heatmap.cpu().float().clamp(0, 1)
                    severity_val = float(severity_val)
                except Exception as _e:
                    logger.warning("[MaskOps] diagnose failed: %s", _e)
                    explanation_str = f"diagnose_error: {_e}"

            info_obj = {
                "segmenter": seg_key,
                "matter": mat_key,
                "auto_route": {
                    "applied": bool(_auto_routed),
                    "reasons": _auto_route_reasons,
                    "cascade_mode": bool(_cascade_mode),
                    "cascade_trail": _cascade_trail if _cascade_mode else [],
                },
                "mode": mode,
                "frames": int(B),
                "segmenter_score": score,
                "production_score": final_score,
                "subject_preset": subject_preset,
                "trimap": {"dilate": d, "erode": e, "edge": edge,
                           "advanced": bool(enable_advanced_trimap)},
                "vfx": {
                    "tta_flip": bool(tta_flip),
                    "multiscale": bool(multiscale),
                    "post_refine": post_refine,
                    "despill": despill,
                    "despill_strength": float(despill_strength),
                    "lightwrap_strength": float(lightwrap_strength),
                    "premultiply": bool(premultiply),
                },
                "luma_key": {
                    "enabled": bool(enable_luma_key),
                    "mode": luma_mode,
                    "mix": luma_mix,
                },
                "refine": {
                    "enabled": False,
                    "note": "11-stage refinement now lives in MaskRefineMEC; chain it downstream for power-user tweaking.",
                },
                "auto_quality": {
                    **auto_q_info,
                    "post_steps": auto_q_steps_post,
                    "disambiguate": bool(auto_disambiguate),
                },
                "diagnose": {
                    "enabled": bool(enable_diagnose),
                    "severity": float(severity_val),
                    "suggested_method": suggested,
                    "explanation": explanation_str,
                },
                "robust_propagation": robust_info,
                "quality": quality,
            }
            # Output luma-key mask (zeros if not run).
            if luma_mask_t is None:
                luma_out = torch.zeros_like(alpha_cpu)
            else:
                luma_out = luma_mask_t.cpu().float().clamp(0, 1)
                if luma_out.shape != alpha_cpu.shape:
                    luma_out = torch.zeros_like(alpha_cpu)
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
                luma_out,
                problem_heatmap,
                float(severity_val),
                suggested,
            )
        finally:
            free_vram()


NODE_CLASS_MAPPINGS = {"MaskOpsMEC": MaskOpsMEC}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskOpsMEC": "Mask + Matting"}
