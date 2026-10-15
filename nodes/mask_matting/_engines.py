"""The other matte pipelines, as backends of MaskOps (L7.65 P05, owner card 2026-10-09).

SAM + ViTMatte, SeC + MatAnyone2 and Background Remover used to be three separate nodes. They are now values of
MaskOps' `backend` input, and their workflows migrate into MaskOps on load (nodes/_legacy_replacements.json). Their
code is not rewritten: `run_engine` calls the old classes' own entry points with the old arguments, so a migrated node
gives the old results (tests/test_consolidation_p05_matte.py compares every mapped output). `finish` then adds what
every MaskOps backend shares - holdout / core masks, the edge mode, despill, light wrap, the edge / inside / outside
split, diagnostics and the foreground - without copying the image when a feature is off.

Design: docs/design/matte_unified.md. The old modules are imported lazily, inside the engine functions, so a broken
optional dependency of one of them can never stop MaskOps from loading.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("MEC.MaskOps.Engines")

NATIVE = "segment + matte"
SAM_VITMATTE = "SAM + ViTMatte"
SEC_MATANYONE = "SeC + MatAnyone2"
REMOVE_BG = "remove background"
BACKENDS = (NATIVE, SAM_VITMATTE, SEC_MATANYONE, REMOVE_BG)
ENGINES = (SAM_VITMATTE, SEC_MATANYONE, REMOVE_BG)


def is_engine(backend: Any) -> bool:
    return str(backend) in ENGINES


# ── the registry combos, exactly as the old nodes built them ─────────────────────────────────────────────────────

def _registry_names(families: Tuple[str, ...], fallback: str) -> List[str]:
    try:
        from ..model_manager import MODEL_REGISTRY
        names = [name for name, reg in sorted(MODEL_REGISTRY.items()) if reg.get("family") in families]
    except Exception as exc:  # noqa: BLE001 - MaskOps must load even when the model registry does not
        logger.warning("[MaskOps] model registry unavailable (%s) - %s list reduced to %s", exc, families, fallback)
        names = []
    return names or [fallback]


def sec_model_choices() -> List[str]:
    """SeC + MatAnyone2's segmentation models (the old node's list: families sec / sam2 / sam3)."""
    return _registry_names(("sec", "sam2", "sam3"), "sec_4b")


def bg_model_choices() -> List[str]:
    """Background removal models (the old node's list: families rmbg / birefnet)."""
    return _registry_names(("rmbg", "birefnet"), "rmbg_2.0")


def appended_inputs() -> Dict[str, tuple]:
    """MaskOps' inputs added for the three backends, in order. Types, defaults, ranges and option lists are the old
    nodes' own, so every saved value still validates. A fresh dict on every call: INPUT_TYPES adds section keys."""
    sec_models = sec_model_choices()
    bg_models = bg_model_choices()
    sam = "SAM + ViTMatte: "
    sec = "SeC + MatAnyone2: "
    bg = "remove background: "
    return {
        "backend": (list(BACKENDS), {
            "default": NATIVE,
            "tooltip": "Which pipeline makes the matte. 'segment + matte' = this node's segmenter and matter "
                       "(pipeline, segmenter, matter). 'SAM + ViTMatte' = a SAM model from a SAM Model Loader, then "
                       "ViTMatte or a guided filter on the edges. 'SeC + MatAnyone2' = SeC / SAM segmentation, then "
                       "MatAnyone2 (video) or ViTMatte. 'remove background' = RMBG-2.0 or BiRefNet, no prompt. "
                       "Each control's tooltip names the backend that reads it."}),
        "sam_model": ("SAM_MODEL", {"tooltip": sam + "the model from a SAM Model Loader."}),
        "points_json": ("STRING", {
            "default": "[]", "multiline": True,
            "tooltip": sam.rstrip(": ") + " and " + sec + 'point prompts: [{"x":100,"y":200,"label":1}, ...] '
                       "(label 1 = keep, 0 = remove). The click editor writes this."}),
        "bbox_json": ("STRING", {
            "default": "", "multiline": False,
            "tooltip": sam.rstrip(": ") + " and " + sec + "box prompt [x1,y1,x2,y2]. Empty = points only. "
                       "A wired pos_bbox wins."}),
        "sam_iterations": ("INT", {
            "default": 2, "min": 1, "max": 5, "step": 1,
            "tooltip": sam + "SAM passes; each pass uses the previous mask to place better prompts. 2-3 is ideal."}),
        "sam_refine_method": (["auto", "vitmatte", "guided_filter", "multi_scale_guided", "color_aware",
                               "laplacian_blend"], {
            "default": "auto",
            "tooltip": sam + "edge refinement. auto = ViTMatte, else multi-scale guided filter, else guided "
                             "filter. vitmatte = neural matting. guided_filter = fast. multi_scale_guided = three "
                             "scales (best without a model). color_aware = LAB colour edges. laplacian_blend = "
                             "frequency blend."}),
        "sam_edge_radius": ("INT", {
            "default": 12, "min": 1, "max": 200, "step": 1,
            "tooltip": sam + "pixels around the edge that are refined (larger = softer transitions)."}),
        "sam_detail_preservation": ("FLOAT", {
            "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": sam + "fine detail (hair, fur, lace) to keep. 0 = smooth, 1 = all."}),
        "sam_edge_contrast": ("FLOAT", {
            "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
            "tooltip": sam + "edge contrast for hard lighting. Above 1 sharpens the boundary."}),
        "sam_multimask_output": ("BOOLEAN", {
            "default": True, "tooltip": sam + "SAM returns three candidate masks (off = its single best)."}),
        "sam_mask_index": ("INT", {
            "default": 0, "min": 0, "max": 2,
            "tooltip": sam + "which of SAM's three candidates to keep (with multimask output on)."}),
        "sam_score_threshold": ("FLOAT", {
            "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": sam + "skip SAM candidates below this confidence (0 = use mask_index)."}),
        "sam_trimap_dilate": ("INT", {
            "default": 0, "min": 0, "max": 200, "step": 1,
            "tooltip": sam + "outer trimap radius (0 = edge radius x 1.5)."}),
        "sam_trimap_erode": ("INT", {
            "default": 0, "min": 0, "max": 200, "step": 1,
            "tooltip": sam + "inner trimap radius (0 = edge radius x 1.0)."}),
        "sam_batch_mode": ("BOOLEAN", {
            "default": False,
            "tooltip": sam + "process every frame of the batch. Off = the first frame only (every output then has "
                             "one frame)."}),
        "sec_model": (sec_models, {
            "default": sec_models[0],
            "tooltip": sec + "segmentation model. SeC = video with text prompts; SAM 2 / 3 = point or box prompts."}),
        "sec_text_prompt": ("STRING", {
            "default": "", "multiline": False,
            "tooltip": sec + "what to track, e.g. 'cat' or 'person in red'. A wired text_prompt wins. Empty = "
                             "point / box prompts."}),
        "sec_matting_backend": (["matanyone2", "vitmatte_small", "vitmatte_base", "auto"], {
            "default": "auto",
            "tooltip": sec + "alpha matting. auto = MatAnyone2 for video (more than one frame), ViTMatte small for "
                             "an image. MatAnyone2 keeps the matte steady over time; ViTMatte has the best edge per "
                             "frame."}),
        "sec_edge_radius": ("INT", {
            "default": 15, "min": 1, "max": 200, "step": 1,
            "tooltip": sec + "edge refinement radius in pixels."}),
        "sec_n_warmup": ("INT", {
            "default": 5, "min": 1, "max": 30, "step": 1,
            "tooltip": sec + "MatAnyone2 warm-up passes on the first frame (more = steadier start)."}),
        "sec_edge_refine_method": (["none", "vitmatte", "guided_filter", "multi_scale_guided"], {
            "default": "none",
            "tooltip": sec + "optional edge refinement after matting. none = the matting output as it is."}),
        "bg_model": (bg_models, {
            "default": bg_models[0],
            "tooltip": bg + "rmbg_2.0 = general purpose, fast. birefnet_general = detailed edges. "
                            "birefnet_portrait = people."}),
        "bg_threshold": ("FLOAT", {
            "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
            "tooltip": bg + "above 0 the mask is cut hard at this value (0 = keep the soft mask)."}),
        "bg_invert": ("BOOLEAN", {
            "default": False, "tooltip": bg + "invert the mask (keep the background instead)."}),
        "bg_mask_blur": ("INT", {
            "default": 0, "min": 0, "max": 50, "step": 1,
            "tooltip": bg + "Gaussian blur on the mask edge, in pixels (0 = none)."}),
        "fill_holes_enabled": ("BOOLEAN", {
            "default": True,
            "tooltip": "SAM + ViTMatte and SeC + MatAnyone2: fill holes inside the mask."}),
        "min_region_size": ("INT", {
            "default": 64, "min": 0, "max": 10000, "step": 1,
            "tooltip": "SAM + ViTMatte and SeC + MatAnyone2: remove separate regions smaller than this many "
                       "pixels (0 = keep all)."}),
        "keep_model_loaded": ("BOOLEAN", {
            "default": True,
            "tooltip": "SeC + MatAnyone2 and remove background: keep the models in memory between runs (off = "
                       "free them after each run)."}),
        "split_masks": ("BOOLEAN", {
            "default": True,
            "tooltip": "Make the edge / inside / outside masks (and, for the three other backends, a trimap from "
                       "the final alpha). Off leaves those outputs empty, which saves three to four full-size "
                       "masks per frame on long clips."}),
    }


# ── running an engine ────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class EngineResult:
    mask: torch.Tensor
    alpha: torch.Tensor
    preview: Optional[torch.Tensor]
    score: Optional[float]
    bbox: Optional[list]
    edge_mask: Optional[torch.Tensor]
    foreground: Optional[torch.Tensor]
    info: Dict[str, Any]
    bbox_xywh: bool = False


def _import_engine(label: str, module: str, name: str):
    try:
        from importlib import import_module
        return getattr(import_module(f"..{module}", package=__package__), name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"The {label} backend is unavailable in this install: {exc}") from exc


def _prompt_bbox(kw: Dict[str, Any]):
    return kw.get("pos_bbox") if kw.get("pos_bbox") is not None else kw.get("normal_bbox")


def _run_sam_vitmatte(image: torch.Tensor, kw: Dict[str, Any]) -> EngineResult:
    cls = _import_engine(SAM_VITMATTE, "sam_vitmatte_pipeline", "SAMViTMattePipelineMEC")
    if kw.get("sam_model") is None:
        raise ValueError("SAM + ViTMatte needs a SAM model: connect a SAM Model Loader to sam_model.")
    refined, coarse, edge, preview, det_bbox, score, info = cls().execute(
        kw["sam_model"], image, kw["subject_preset"], kw["points_json"], kw["bbox_json"],
        kw["sam_iterations"], kw["sam_refine_method"], kw["sam_edge_radius"],
        kw["sam_detail_preservation"], kw["sam_edge_contrast"], kw["fill_holes_enabled"],
        kw["min_region_size"], kw["sam_multimask_output"], kw["sam_mask_index"],
        kw["sam_score_threshold"], bbox=_prompt_bbox(kw), existing_mask=kw.get("external_mask"),
        trimap=kw.get("external_trimap"), trimap_dilate=kw["sam_trimap_dilate"],
        trimap_erode=kw["sam_trimap_erode"], batch_mode=kw["sam_batch_mode"])
    return EngineResult(mask=coarse, alpha=refined, preview=preview, score=float(score), bbox=det_bbox,
                        edge_mask=edge, foreground=None, info=json.loads(info), bbox_xywh=True)


def _run_sec_matanyone(image: torch.Tensor, kw: Dict[str, Any]) -> EngineResult:
    cls = _import_engine(SEC_MATANYONE, "sec_matanyone_pipeline", "SeCMatAnyonePipelineMEC")
    text = str(kw.get("text_prompt") or "")
    text = text if text.strip() else kw["sec_text_prompt"]      # a wired text_prompt wins over the typed one
    rgb, alpha, coarse, preview, info = cls().process(
        image, kw["sec_model"], text, kw["points_json"], kw["bbox_json"], kw["sec_matting_backend"],
        kw["sec_edge_radius"], kw["sec_n_warmup"], kw["precision"], kw["fill_holes_enabled"],
        kw["min_region_size"], kw.get("positive_coords"), kw.get("negative_coords"), _prompt_bbox(kw),
        kw["sec_edge_refine_method"], kw["keep_model_loaded"], kw["auto_download"])
    info = json.loads(info)
    return EngineResult(mask=coarse, alpha=alpha, preview=preview, score=float(info.get("seg_score", 0.0)),
                        bbox=None, edge_mask=None, foreground=rgb, info=info)


def _run_remove_background(image: torch.Tensor, kw: Dict[str, Any]) -> EngineResult:
    cls = _import_engine(REMOVE_BG, "background_remover", "BackgroundRemoverMEC")
    fg, mask, info = cls().remove_bg(image, kw["bg_model"], kw["bg_threshold"], kw["bg_invert"],
                                     kw["bg_mask_blur"], kw["keep_model_loaded"], kw["auto_download"])
    return EngineResult(mask=mask, alpha=mask, preview=None, score=None, bbox=None, edge_mask=None,
                        foreground=fg, info=json.loads(info))


_RUNNERS = {
    SAM_VITMATTE: _run_sam_vitmatte,
    SEC_MATANYONE: _run_sec_matanyone,
    REMOVE_BG: _run_remove_background,
}


def run_engine(backend: str, image: torch.Tensor, kw: Dict[str, Any]) -> EngineResult:
    runner = _RUNNERS.get(str(backend))
    if runner is None:
        raise ValueError(f"Unknown backend '{backend}'. Pick one of: {', '.join(BACKENDS)}")
    return runner(image, kw)


# ── what every backend shares ────────────────────────────────────────────────────────────────────────────────────

def _empty(shape) -> torch.Tensor:
    """All zeros in the given shape without the memory: a zero-stride view of one value (ComfyUI core's own Mask to
    Image returns such a view). For outputs a backend does not produce or a feature that is off."""
    shape = tuple(int(s) for s in shape)
    return torch.zeros((1,) * len(shape), dtype=torch.float32).expand(*shape)


def finish(res: EngineResult, image: torch.Tensor, backend: str, kw: Dict[str, Any]) -> tuple:
    """The engine's matte through MaskOps' shared finishing stage -> MaskOps' 19 outputs.

    At the defaults (no holdout / core mask, edge mode soft, despill off) the alpha, mask, preview and foreground
    are the engine's own tensors, unchanged and uncopied."""
    from . import _vfx
    from .utils import bbox_from_mask, bbox_to_json, to_bhwc, to_mask

    alpha0 = res.alpha
    B, H, W = (int(s) for s in alpha0.shape)
    frames = image[:B]
    alpha = alpha0

    # holdout / core overrides - the same rules as segment + matte (only a mask of the same size counts; one frame
    # covers the batch)
    holdout_mask, core_mask = kw.get("holdout_mask"), kw.get("core_mask")
    if holdout_mask is not None:
        hm = to_mask(holdout_mask)
        if hm.shape[-2:] == alpha.shape[-2:]:
            if hm.shape[0] != alpha.shape[0] and hm.shape[0] == 1:
                hm = hm.expand_as(alpha)
            alpha = (alpha * (1.0 - hm.to(alpha.device))).clamp(0, 1)
    if core_mask is not None:
        cm = to_mask(core_mask)
        if cm.shape[-2:] == alpha.shape[-2:]:
            if cm.shape[0] != alpha.shape[0] and cm.shape[0] == 1:
                cm = cm.expand_as(alpha)
            alpha = torch.maximum(alpha, cm.to(alpha.device)).clamp(0, 1)
    edge_mode = str(kw.get("edge_mode", "soft"))
    if edge_mode not in ("soft", ""):
        alpha = _vfx.edge_treatment(alpha, edge_mode, threshold=float(kw.get("edge_threshold", 0.5)),
                                    feather=float(kw.get("edge_feather", 0.0)))
    changed = alpha is not alpha0

    despill = str(kw.get("despill", "off"))
    despill_on = despill != "off" and float(kw.get("despill_strength", 1.0)) > 0
    if despill_on:
        despilled = _vfx.despill(to_bhwc(frames), alpha, backing=despill,
                                 strength=float(kw.get("despill_strength", 1.0)),
                                 preserve_skin=bool(kw.get("preserve_skin", True))).cpu()
    else:
        despilled = frames                                   # the input itself: no copy
    lw_strength = float(kw.get("lightwrap_strength", 0.0))
    if lw_strength > 0:
        lightwrap = _vfx.lightwrap_layer(to_bhwc(frames), alpha, bg_color=None,
                                         radius=int(kw.get("lightwrap_radius", 8)), strength=lw_strength).cpu()
    else:
        lightwrap = _empty((B, H, W, 4))

    split = bool(kw.get("split_masks", True))
    if split:
        band, inside, outside = _vfx.edge_inside_outside(alpha, edge_radius=int(kw.get("edge_band_radius", 4)))
        band, inside, outside = band.cpu(), inside.cpu(), outside.cpu()
    else:
        band = inside = outside = _empty((B, H, W))
    edge_mask = res.edge_mask if res.edge_mask is not None else band
    external_trimap = kw.get("external_trimap")
    if external_trimap is not None:
        trimap, trimap_src = to_mask(external_trimap), "external"
    elif split:
        trimap, trimap_src = (inside + 0.5 * band).clamp(0, 1), "derived from the final alpha"
    else:
        trimap, trimap_src = _empty((B, H, W)), "off"

    if res.foreground is not None and not changed and despilled is frames:
        foreground = res.foreground                          # the engine's own image x alpha
    else:
        foreground = (despilled * alpha.unsqueeze(-1)).clamp(0, 1)
    if res.preview is not None:
        preview = res.preview
    else:
        preview = foreground if bool(kw.get("premultiply", True)) else despilled

    if res.score is not None:
        score, score_src = float(res.score), "the engine's own"
    else:
        score, score_src = float(_vfx.score_quality(frames.float(), alpha)["overall"]), "production-quality score"

    if res.bbox is not None:
        bbox = res.bbox
    else:
        bbox = [int(v) for v in bbox_from_mask(alpha[0].numpy())]
    if res.bbox_xywh:
        x, y, w, h = (int(v) for v in list(bbox)[:4])
        bbox_json = json.dumps({"x": x, "y": y, "w": w, "h": h})
    else:
        bbox_json = bbox_to_json(tuple(int(v) for v in bbox))

    explanation, severity, suggested = "", 0.0, ""
    heat = _empty((B, H, W))
    if bool(kw.get("enable_diagnose", True)):
        try:
            from ..mask_failure_explainer import MaskFailureExplainerMEC
            explanation, heat, severity, suggested = MaskFailureExplainerMEC()._analyze_impl(
                to_bhwc(frames), alpha, int(kw.get("diag_ring_width", 5)),
                float(kw.get("diag_blur_threshold", 50.0)), float(kw.get("diag_brightness_threshold", 0.15)))
            heat = heat.cpu().float().clamp(0, 1)
            severity = float(severity)
        except Exception as exc:  # noqa: BLE001 - diagnostics never fail the matte (as in segment + matte)
            logger.warning("[MaskOps] diagnose failed: %s", exc)
            explanation, heat, severity, suggested = f"diagnose_error: {exc}", _empty((B, H, W)), 0.0, ""

    info = dict(res.info)
    info["backend"] = backend
    info["maskops"] = {
        "finishing": {"edge_mode": edge_mode, "despill": despill if despill_on else "off",
                      "lightwrap_strength": lw_strength, "holdout": holdout_mask is not None,
                      "core": core_mask is not None},
        "trimap": trimap_src,
        "bbox_format": "x,y,w,h" if res.bbox_xywh else "x0,y0,x1,y1",
        "split_masks": split,
        "score": score_src,
        "diagnose": {"enabled": bool(kw.get("enable_diagnose", True)), "severity": float(severity),
                     "suggested_method": suggested, "explanation": explanation},
    }
    return (res.mask, alpha, preview, trimap, bbox, bbox_json, score, json.dumps(info, indent=2),
            despilled, lightwrap, edge_mask, inside, outside, _empty((B, H, W)), heat, float(severity), suggested,
            alpha, foreground)


# ── kill switch ──────────────────────────────────────────────────────────────────────────────────────────────────

_LEGACY = (
    ("SAMViTMattePipelineMEC", "sam_vitmatte_pipeline", "SAM + ViTMatte Pipeline (legacy)"),
    ("SeCMatAnyonePipelineMEC", "sec_matanyone_pipeline", "SeC + MatAnyone2 Pipeline (legacy)"),
    ("BackgroundRemoverMEC", "background_remover", "Background Remover (legacy)"),
)


def legacy_matte_mappings() -> Tuple[Dict[str, type], Dict[str, str]]:
    """C2C_LEGACY_MATTE_NODES=1 registers the three former nodes again, from their unchanged code. The workflow
    migration only rewrites node types that are not registered, so old workflows then load the original nodes."""
    if os.environ.get("C2C_LEGACY_MATTE_NODES") != "1":
        return {}, {}
    from importlib import import_module
    mappings: Dict[str, type] = {}
    display: Dict[str, str] = {}
    for node_id, module, label in _LEGACY:
        try:
            mappings[node_id] = getattr(import_module(f"..{module}", package=__package__), node_id)
            display[node_id] = label
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MaskOps] C2C_LEGACY_MATTE_NODES: %s not registered (%s)", node_id, exc)
    return mappings, display
