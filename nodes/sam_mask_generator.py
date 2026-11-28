"""
SAMMaskGeneratorMEC – Generate masks using a loaded SAM model with
point prompts, bounding-box prompts, text prompts (via GroundingDINO), or any
combination.  Supports VRAM offload, multi-mask output, score thresholding,
and iterative refinement.
"""

import torch
import numpy as np
import json
import gc
import logging

from ._is_changed_util import hash_args_and_kwargs
from ._mask_edge_refine import refine_edges, parse_bboxes, parse_spline, EDGE_MODES
from .utils import (
    HAS_CV2,
    get_sam_predictor,
    sam_predict,
    augment_prompts_from_mask,
    mask_to_sam_logits,
    parse_points_json,
    parse_bbox_input,
)
from .model_manager import get_or_load_model, MODEL_REGISTRY, scan_model_dir


# ── L9.172 (A25 spec item 49): candidate refinement + several objects ──────────────────────────────
CANDIDATE_MODES = ["mask_index (as before)", "best score", "best quality"]
_OBJECTS_FORMAT = ('objects_json must be a JSON list like [{"name": "cup", "frame": 0, "points": [{"x": 120, "y": 80, '
                   '"label": 1}], "bbox": [x1, y1, x2, y2]}, ...]: "name" is required, "frame" defaults to 0, and each '
                   'entry needs "points" or "bbox" (or both).')
_NO_PROPAGATION = ("frames without a prompt are empty; use Segment (C2C) / Mask Track to propagate")


def _erode3(fg: np.ndarray) -> np.ndarray:
    """3x3 binary erosion (outside the frame counts as background), numpy only."""
    p = np.pad(fg, 1, mode="constant", constant_values=False)
    H, W = fg.shape
    out = np.ones_like(fg, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= p[dy:dy + H, dx:dx + W]
    return out


def _sobel_mag(grey: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a float grey image (edge-replicated border), numpy only."""
    p = np.pad(grey, 1, mode="edge")
    H, W = grey.shape

    def s(dy, dx):
        return p[dy:dy + H, dx:dx + W]
    gx = (s(0, 2) + 2 * s(1, 2) + s(2, 2)) - (s(0, 0) + 2 * s(1, 0) + s(2, 0))
    gy = (s(2, 0) + 2 * s(2, 1) + s(2, 2)) - (s(0, 0) + 2 * s(0, 1) + s(0, 2))
    return np.hypot(gx, gy)


def _components(fg: np.ndarray):
    """(count, largest size) of 8-connected foreground components; (None, None) without OpenCV and SciPy."""
    if not fg.any():
        return 0, 0
    if HAS_CV2:
        import cv2
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        if n <= 1:
            return 0, 0
        return int(n - 1), int(stats[1:, cv2.CC_STAT_AREA].max())
    try:
        from scipy import ndimage
    except ImportError:
        return None, None
    lab, n = ndimage.label(fg, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return 0, 0
    return int(n), int(np.bincount(lab.ravel())[1:].max())


def candidate_quality(masks_np, scores, img_np) -> list:
    """L9.172: measure every SAM candidate mask. Returns one dict per candidate:

    index, score; area_fraction (foreground share of the frame); components (8-connected pieces of mask > 0.5, None
    when neither OpenCV nor SciPy is installed); largest_component_share (largest piece / all foreground, 0 when
    empty; counted as 1 when the pieces cannot be counted); compactness (4*pi*area / perimeter^2, perimeter = boundary
    pixels = foreground XOR its 3x3 erosion, clamped 0..1, 0 when empty); edge_alignment (mean normalised Sobel
    gradient of the grey image on the boundary pixels / its mean over the whole image; the gradient is divided by its
    99th percentile and clipped to 1; 0 with no boundary or a flat image) and

        quality = 0.5 * score + 0.25 * largest_component_share + 0.25 * min(1, edge_alignment / 2)

    a documented heuristic (a confident, one-piece mask whose outline sits on image edges scores high), not a learned
    measure. Numbers are rounded to 4 decimals.
    """
    masks = np.asarray(masks_np)
    if masks.ndim == 2:
        masks = masks[None]
    sc = scores.tolist() if hasattr(scores, "tolist") else list(scores)
    img = np.asarray(img_np, dtype=np.float32)
    grey = img[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32) if img.ndim == 3 else img
    mag = _sobel_mag(grey)
    p99 = float(np.percentile(mag, 99)) if mag.size else 0.0
    norm = np.clip(mag / p99, 0.0, 1.0) if p99 > 0 else np.zeros_like(mag)
    img_mean = float(norm.mean()) if norm.size else 0.0
    out = []
    for i in range(masks.shape[0]):
        fg = masks[i].astype(np.float32) > 0.5
        area = int(fg.sum())
        n_comp, largest = _components(fg)
        if area == 0:
            lcs = 0.0
        elif largest is None:
            lcs = 1.0
        else:
            lcs = largest / area
        boundary = fg & ~_erode3(fg)
        perim = int(boundary.sum())
        compact = min(1.0, max(0.0, 4.0 * np.pi * area / (perim * perim))) if (area and perim) else 0.0
        if perim and img_mean > 0:
            edge = float(norm[boundary].mean()) / img_mean
        else:
            edge = 0.0
        score = float(sc[i]) if i < len(sc) else 0.0
        q = 0.5 * score + 0.25 * lcs + 0.25 * min(1.0, edge / 2.0)
        out.append({"index": i, "score": round(score, 4), "area_fraction": round(area / float(fg.size or 1), 4),
                    "components": n_comp, "largest_component_share": round(float(lcs), 4),
                    "compactness": round(float(compact), 4), "edge_alignment": round(float(edge), 4),
                    "quality": round(float(q), 4)})
    return out


def _select_candidate(mode, scores_list, mask_index, score_threshold, quality=None, final=False):
    """One pick rule for the refinement passes and the final pick (L9.172). Returns the index, or None on the final
    pick when no candidate reaches score_threshold.

    "mask_index (as before)": the old code exactly - with score_threshold > 0 the first candidate at or above it
    (a pass with none falls back to 0), else mask_index. "best score" / "best quality": among the candidates with
    score >= score_threshold (all when it is 0) the highest score / quality (a pass with none valid uses all)."""
    if mode == CANDIDATE_MODES[0]:
        if score_threshold > 0:
            valid = [i for i, s in enumerate(scores_list) if s >= score_threshold]
            if not valid:
                return None if final else 0
            return valid[0]
        return min(mask_index, len(scores_list) - 1)
    valid = ([i for i, s in enumerate(scores_list) if s >= score_threshold] if score_threshold > 0
             else list(range(len(scores_list))))
    if not valid:
        if final:
            return None
        valid = list(range(len(scores_list)))
    if mode == "best quality":
        key = {q["index"]: q["quality"] for q in (quality or [])}
        return max(valid, key=lambda i: (key.get(i, 0.0), -i))
    return max(valid, key=lambda i: (float(scores_list[i]), -i))


def parse_objects_json(raw, batch: int) -> list:
    """objects_json -> [{"name", "frame", "points" (list or None), "bbox" (list or None)}] in entry order; "" or [] =
    off (an empty list). Malformed input -> ValueError with the expected format."""
    if raw is None or not str(raw).strip():
        return []
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SAM Mask Generator: objects_json is not valid JSON ({exc}). {_OBJECTS_FORMAT}") from exc
    if not isinstance(data, list):
        raise ValueError(f"SAM Mask Generator: objects_json is a {type(data).__name__}, not a list. {_OBJECTS_FORMAT}")
    seen = set()
    out = []
    for n, e in enumerate(data):
        where = f"objects_json entry {n}"
        if not isinstance(e, dict):
            raise ValueError(f"SAM Mask Generator: {where} is not an object. {_OBJECTS_FORMAT}")
        name = e.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"SAM Mask Generator: {where} needs a non-empty \"name\". {_OBJECTS_FORMAT}")
        name = name.strip()
        frame = e.get("frame", 0)
        if isinstance(frame, bool) or not isinstance(frame, (int, float)) or int(frame) != frame:
            raise ValueError(f"SAM Mask Generator: {where} ({name!r}) has frame {frame!r}; it must be a whole "
                             f"number. {_OBJECTS_FORMAT}")
        frame = int(frame)
        if not 0 <= frame < batch:
            raise ValueError(f"SAM Mask Generator: {where} ({name!r}) is on frame {frame}, but the image has {batch} "
                             f"frame(s) (0..{batch - 1}).")
        points = e.get("points")
        if points is not None:
            if not isinstance(points, list):
                raise ValueError(f"SAM Mask Generator: {where} ({name!r}) \"points\" must be a list. {_OBJECTS_FORMAT}")
            clean = []
            for p in points:
                if not isinstance(p, dict) or not all(isinstance(p.get(k), (int, float)) and not isinstance(
                        p.get(k), bool) for k in ("x", "y")):
                    raise ValueError(f"SAM Mask Generator: {where} ({name!r}) has a point {p!r}; each point is "
                                     f"{{\"x\": .., \"y\": .., \"label\": 1 or 0}}. {_OBJECTS_FORMAT}")
                lab = p.get("label", 1)
                if lab not in (0, 1):
                    raise ValueError(f"SAM Mask Generator: {where} ({name!r}) point label {lab!r} must be 1 "
                                     "(foreground) or 0 (background).")
                clean.append({"x": float(p["x"]), "y": float(p["y"]), "label": int(lab)})
            points = clean or None
        bbox = e.get("bbox")
        if bbox is not None:
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox)):
                raise ValueError(f"SAM Mask Generator: {where} ({name!r}) \"bbox\" must be [x1, y1, x2, y2]. "
                                 f"{_OBJECTS_FORMAT}")
            if not (bbox[2] > bbox[0] and bbox[3] > bbox[1]):
                raise ValueError(f"SAM Mask Generator: {where} ({name!r}) bbox {bbox} needs x2 > x1 and y2 > y1.")
            bbox = [float(v) for v in bbox]
        if points is None and bbox is None:
            raise ValueError(f"SAM Mask Generator: {where} ({name!r}) has no \"points\" and no \"bbox\" - give at "
                             f"least one. {_OBJECTS_FORMAT}")
        if (name, frame) in seen:
            raise ValueError(f"SAM Mask Generator: objects_json has {name!r} twice on frame {frame}; give each object "
                             "one entry per frame (merge the points).")
        seen.add((name, frame))
        out.append({"name": name, "frame": frame, "points": points, "bbox": bbox})
    return out


class SAMMaskGeneratorMEC:
    """Run SAM inference with point prompts + bounding box prompts.
    Handles automatic GPU offload if the model was loaded with offload mode.
    Supports iterative refinement for maximum accuracy."""

    @classmethod
    def INPUT_TYPES(cls):
        # Scan for GroundingDINO models
        gdino_models = ["none"]
        for name, reg in MODEL_REGISTRY.items():
            if reg.get("family") == "groundingdino":
                gdino_models.append(name)
        return {
            "required": {
                "sam_model": ("SAM_MODEL", {"tooltip": "Loaded SAM model from SAM Model Loader"}),
                "image": ("IMAGE", {"tooltip": "Input image to segment (first frame is used)"}),
                "points_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": (
                        'JSON array: [{"x":100,"y":200,"label":1}, ...]. '
                        'label=1=foreground, label=0=background.'
                    ),
                }),
                "bbox_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        'Bounding box as JSON: [x1, y1, x2, y2] or {"x":..,"y":..,"w":..,"h":..}. '
                        'Leave empty to use only point prompts.'
                    ),
                }),
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Text description of target object (e.g. 'person', 'dog', 'car').\n"
                        "Requires a GroundingDINO model. Converts text to bounding box, "
                        "then feeds to SAM for precise mask generation."
                    ),
                }),
                "negative_text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Text description of objects to EXCLUDE (e.g. 'background', 'wall').\n"
                        "Uses GroundingDINO to detect these regions, then generates negative "
                        "points from them to suppress unwanted areas in the mask."
                    ),
                }),
                "grounding_model": (gdino_models, {
                    "default": "none",
                    "tooltip": (
                        "GroundingDINO model for text-to-bbox grounding.\n"
                        "Set to 'none' to disable text prompting."
                    ),
                }),
                "text_threshold": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "GroundingDINO text threshold: how strongly a word of the prompt must match a found box "
                               "for that word to label it.",
                }),
                "text_box_threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "GroundingDINO box threshold: the minimum confidence for a found box to be kept.",
                }),
                "multimask_output": ("BOOLEAN", {"default": True,
                                                   "tooltip": "Return 3 candidate masks (SAM default) vs 1"}),
                "mask_index": ("INT", {"default": 0, "min": 0, "max": 2,
                                        "tooltip": "Which of the 3 candidates to return when multimask=True, in the model's own "
                                                   "order (0 = first, NOT necessarily the best score: see the score "
                                                   "output; score_threshold keeps the first candidate above it)"}),
                "score_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                               "tooltip": "Discard masks below this confidence score"}),
                "apply_bbox_crop": ("BOOLEAN", {"default": False,
                                                  "tooltip": "Crop output to bbox region"}),
                "refine_iterations": ("INT", {
                    "default": 1, "min": 1, "max": 5, "step": 1,
                    "tooltip": (
                        "Iterative refinement passes.  Each pass feeds the previous mask "
                        "back into SAM with augmented prompts.  2-3 significantly improves accuracy."
                    ),
                }),
                "auto_negative_points": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Automatically sample negative points just outside the mask boundary.  "
                        "Helps in cluttered scenes and similar-color backgrounds."
                    ),
                }),
                "edge_refine": (EDGE_MODES, {
                    "default": "guided",
                    "tooltip": (
                        "Edge-perfect refinement of the SAM mask. SAM gives a blocky/binary mask; "
                        "these snap it to the true object boundary → a soft, roto-grade alpha, robust "
                        "to motion blur and dull/over-bright colours.\n"
                        "  none          — raw binary SAM mask\n"
                        "  guided        — fast edge-aware guided filter (recommended, no model)\n"
                        "  guided_strong — wider/softer for hair + fine detail (no model)\n"
                        "  matte         — AI alpha matting (ViTMatte/BiRefNet) for the hardest hair + "
                        "motion-blur edges; auto-falls back to 'guided' if no matte model installed."
                    ),
                }),
                "edge_radius": ("INT", {
                    "default": 8, "min": 1, "max": 64,
                    "tooltip": "Guided-filter radius (px). Larger = smoother/softer; smaller hugs "
                               "fine detail. Ignored when edge_refine = none.",
                }),
            },
            "optional": {
                "bbox": ("BBOX", {"tooltip": "Bounding box from BBox node (overrides bbox_json)"}),
                "bboxes_json": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "MULTIPLE boxes to select several regions at once: "
                               "[[x1,y1,x2,y2], ...]. Each is segmented and unioned into one mask.",
                }),
                "spline_json": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Outline the target with a SPLINE and connect/paste its data here "
                               "(the Image Mask Editor's spline_data_out, spline mode). Its polygon "
                               "becomes positive points + a "
                               "bounding box so SAM snaps to what you drew.",
                }),
                "existing_mask": ("MASK", {
                    "tooltip": "Use this mask as the starting point instead of running SAM from scratch",
                }),
                # L7.97 (owner 2026-10-12): appended last; off = never download
                "allow_download": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Download the grounding model for text prompts if it is not installed (into its models/ subfolder). Off: an installed copy is used wherever it is; without one a text prompt needs points or a box as well.",
                }),
                # L9.172 (A25 spec item 49): appended after A28; the defaults run exactly the old code
                "candidate_select": (CANDIDATE_MODES, {
                    "default": CANDIDATE_MODES[0],
                    "tooltip": "Which SAM candidate to keep, in every refinement pass and at the end. "
                               "mask_index (as before): mask_index, or with score_threshold the first candidate "
                               "above it. best score: the candidate with the highest SAM score. best quality: the "
                               "highest of 0.5 x score + 0.25 x share of its largest piece + 0.25 x how well its "
                               "outline sits on image edges (a measured heuristic; see the quality output). Both "
                               "'best' modes only consider candidates at or above score_threshold.",
                }),
                "objects_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Several objects, each with its own prompts, optionally on different frames: "
                               "[{\"name\": \"cup\", \"frame\": 0, \"points\": [{\"x\": 120, \"y\": 80, \"label\": 1}], "
                               "\"bbox\": [x1, y1, x2, y2]}, ...]. Each entry needs points or a bbox; the same name on "
                               "other frames gives that object frame-specific prompts. When set, the other point / "
                               "box / spline / text prompts are not used. Each object's frames go to object_masks; "
                               "frames without a prompt stay empty (no tracking). Empty = off.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "BBOX", "FLOAT", "STRING", "MASK", "STRING",)
    RETURN_NAMES = ("mask", "all_masks", "detected_bbox", "score", "info", "object_masks", "quality",)
    OUTPUT_TOOLTIPS = (
        "Selected mask for the chosen mask_index.",
        "All candidate masks returned by SAM (when multimask_output is True).",
        "Bounding box derived from the selected mask.",
        "SAM confidence score of the selected mask.",
        "JSON summary of prompts, scores, and refinement steps.",
        "objects_json set: one mask per object per frame, object by object (K objects x B frames, object k's frames "
        "at k*B..k*B+B-1, in order of first appearance); frames without a prompt for that object are empty. "
        "objects_json empty: a copy of mask.",
        "JSON: the candidate selection mode, the selected candidate, every candidate's measurements (score, area, "
        "pieces, largest piece share, compactness, edge alignment, quality) and the pick of each refinement pass; "
        "with objects_json, one such report per entry.",
    )
    FUNCTION = "generate"
    CATEGORY = "C2C/SAM"
    DESCRIPTION = (
        "Generate masks using SAM with point + bounding box prompts, "
        "iterative refinement, and VRAM offload support. "
        "candidate_select can keep the best-scoring or best-measured candidate (quality output); objects_json "
        "segments several named objects with their own prompts on chosen frames (object_masks output)."
    )

    @classmethod
    def IS_CHANGED(cls, sam_model, image, points_json, bbox_json,
                   text_prompt, negative_text_prompt, grounding_model,
                   text_threshold, text_box_threshold,
                   multimask_output, mask_index, score_threshold,
                   apply_bbox_crop, refine_iterations=1,
                   auto_negative_points=False, bbox=None, existing_mask=None, **kwargs):
        return hash_args_and_kwargs(
            sam_model, image, points_json, bbox_json,
            text_prompt, negative_text_prompt, grounding_model,
            text_threshold, text_box_threshold,
            multimask_output, mask_index, score_threshold,
            apply_bbox_crop, refine_iterations,
            auto_negative_points, bbox, existing_mask, **kwargs,
        )

    def generate(self, sam_model, image, points_json, bbox_json,
                 text_prompt, negative_text_prompt, grounding_model,
                 text_threshold, text_box_threshold,
                 multimask_output, mask_index, score_threshold,
                 apply_bbox_crop, refine_iterations=1,
                 auto_negative_points=False, edge_refine="guided", edge_radius=8,
                 bbox=None, bboxes_json="", spline_json="", existing_mask=None, allow_download=False,
                 candidate_select=CANDIDATE_MODES[0], objects_json=""):
        from ._download_policy import allow_downloads
        with allow_downloads(allow_download):
            return self._generate_entry(sam_model, image, points_json, bbox_json, text_prompt,
                                        negative_text_prompt, grounding_model, text_threshold, text_box_threshold,
                                        multimask_output, mask_index, score_threshold, apply_bbox_crop,
                                        refine_iterations, auto_negative_points, edge_refine, edge_radius, bbox,
                                        bboxes_json, spline_json, existing_mask, candidate_select, objects_json)

    def _generate_entry(self, sam_model, image, points_json, bbox_json,
                        text_prompt, negative_text_prompt, grounding_model,
                        text_threshold, text_box_threshold,
                        multimask_output, mask_index, score_threshold,
                        apply_bbox_crop, refine_iterations=1,
                        auto_negative_points=False, edge_refine="guided", edge_radius=8,
                        bbox=None, bboxes_json="", spline_json="", existing_mask=None,
                        candidate_select=CANDIDATE_MODES[0], objects_json=""):

        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("SAMMaskGeneratorMEC expects IMAGE tensor [B,H,W,C]")
        if existing_mask is not None and (
            not isinstance(existing_mask, torch.Tensor) or existing_mask.ndim not in (2, 3)
        ):
            raise ValueError("SAMMaskGeneratorMEC existing_mask expects MASK [H,W] or [B,H,W]")
        if candidate_select not in CANDIDATE_MODES:
            raise ValueError(f"SAM Mask Generator: candidate_select {candidate_select!r} is not one of "
                             f"{CANDIDATE_MODES}.")
        objects = parse_objects_json(objects_json, int(image.shape[0]))   # L9.172: checked before any model work

        with torch.no_grad():
            r = self._generate_impl(
                sam_model, image, points_json, bbox_json,
                text_prompt, negative_text_prompt, grounding_model,
                text_threshold, text_box_threshold,
                multimask_output, mask_index, score_threshold,
                apply_bbox_crop, refine_iterations,
                auto_negative_points, bbox, existing_mask,
                edge_refine, edge_radius, bboxes_json, spline_json,
                candidate_select, objects,
            )
        if objects:                        # objects_json path: already (..., object_masks, quality)
            return r
        info_s = r[4]
        B = int(image.shape[0])
        if B > 1:
            # Audit 2026-11-27 (masking.md M-13): only frame 0 is segmented; the outputs stay one frame, but say so.
            note = (f"Only frame 0 of the {B} input frames was segmented (this node works on one still). For a clip, "
                    "track the mask with Mask + Matting (MaskOpsMEC) or Segment (C2C).")
            logging.getLogger("MEC").warning("SAM Mask Generator: %s", note)
            try:
                info_d = json.loads(info_s) if info_s else {}
                if isinstance(info_d, dict):
                    info_d["frames_used"] = f"0 of {B}"
                    info_d["frames_note"] = note
                    info_s = json.dumps(info_d, indent=2)
            except (TypeError, ValueError):
                pass
        # objects_json empty: object_masks = a copy of mask, behaviour unchanged
        return (r[0], r[1], r[2], r[3], info_s, r[0].clone(), r[5])

    def _generate_impl(self, sam_model, image, points_json, bbox_json,
                       text_prompt, negative_text_prompt, grounding_model,
                       text_threshold, text_box_threshold,
                       multimask_output, mask_index, score_threshold,
                       apply_bbox_crop, refine_iterations=1,
                       auto_negative_points=False, bbox=None, existing_mask=None,
                       edge_refine="guided", edge_radius=8,
                       bboxes_json="", spline_json="",
                       candidate_select=CANDIDATE_MODES[0], objects=None):

        model_info = sam_model
        model = model_info["model"]
        model_type = model_info["model_type"]
        target_device = model_info["device"]
        offload = model_info["offload_to_cpu"]
        model_dtype = model_info["dtype"]

        if objects:                        # L9.172: several objects / frame-specific prompts
            ignored = [n for n, v in (("points_json", points_json and parse_points_json(points_json)),
                                      ("bbox_json", bbox_json and str(bbox_json).strip()),
                                      ("bboxes_json", bboxes_json and str(bboxes_json).strip()),
                                      ("spline_json", spline_json and str(spline_json).strip()),
                                      ("bbox", bbox is not None),
                                      ("text_prompt", text_prompt and str(text_prompt).strip()),
                                      ("negative_text_prompt", negative_text_prompt and
                                       str(negative_text_prompt).strip()),
                                      ("apply_bbox_crop", bool(apply_bbox_crop)),
                                      ("existing_mask", existing_mask is not None)) if v]
            if offload and hasattr(model, "to"):
                model.to(target_device)
            try:
                return self._run_objects(model, model_type, model_info, image, objects, multimask_output,
                                         mask_index, score_threshold, target_device, model_dtype, refine_iterations,
                                         auto_negative_points, edge_refine, edge_radius, candidate_select, ignored)
            finally:
                if offload and hasattr(model, "to"):
                    model.to("cpu")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # ── Text prompt → bbox via GroundingDINO ───────────────────────
        # A22 L9.136: a GroundingDINO failure stops the run with the reason (the text prompt used to be dropped
        # silently). The model-free stand-in cannot read text anyway (L7.98), so there it is still only logged.
        strict_text = not (isinstance(model_info, dict) and model_info.get("model_free"))
        text_bbox = None
        if text_prompt and text_prompt.strip() and grounding_model != "none":
            text_bbox = self._text_to_bbox(
                image, text_prompt.strip(), grounding_model,
                text_threshold, text_box_threshold, target_device, strict=strict_text,
            )
            if text_bbox is not None:
                # Text-derived bbox supplements existing bbox_json
                if not bbox_json or not bbox_json.strip():
                    bbox_json = json.dumps(text_bbox.tolist())

        # ── Negative text prompt → negative points via GroundingDINO ───
        # Audit 2026-11-27 (masking.md M-03): this used to put 5 NEGATIVE points on a lattice inside the detected box
        # - points that land on the subject (a necklace box lies on the chest), telling SAM the skin is background.
        # Now the detected object is segmented in its own SAM pass and subtracted from the result (below).
        negative_points_from_text = []
        neg_bbox = None
        if negative_text_prompt and negative_text_prompt.strip() and grounding_model != "none":
            neg_bbox = self._text_to_bbox(
                image, negative_text_prompt.strip(), grounding_model,
                text_threshold, text_box_threshold, target_device, strict=strict_text,
            )

        # ── SPLINE prompt → positive points + bbox ─────────────────────
        # Outline the target with a spline; its polygon vertices become positive
        # point prompts and its bounding box seeds SAM, merged with points_json.
        spline = parse_spline(spline_json)
        if spline is not None:
            s_coords, _s_labels, s_bbox = spline
            try:
                existing = parse_points_json(points_json) or []
            except Exception:
                existing = []
            merged = list(existing) + [{"x": float(x), "y": float(y), "label": 1} for x, y in s_coords]
            points_json = json.dumps(merged)
            if (not bbox_json or not bbox_json.strip()) and bbox is None:
                bbox_json = json.dumps([float(v) for v in s_bbox])

        # ── MULTI-BBOX prompt → union of per-box SAM masks ─────────────
        # Select several disjoint elements at once: run SAM once per box and OR
        # the results into a single mask (a lone box just seeds bbox_json).
        boxes = parse_bboxes(bboxes_json)
        if len(boxes) == 1 and (not bbox_json or not bbox_json.strip()) and bbox is None:
            bbox_json = json.dumps(boxes[0])
            boxes = []

        # L7.98: the model-free stand-in cannot read text; with no point, box or mask left there is nothing to
        # segment - say why instead of returning an empty mask.
        has_prompt = bool(boxes or (points_json and parse_points_json(points_json))
                          or (bbox_json and bbox_json.strip()) or bbox is not None or existing_mask is not None)
        if isinstance(model_info, dict) and model_info.get("model_free") and not has_prompt:
            from .mask_matting._classic import TEXT_NEEDS_MODEL
            raise ValueError(TEXT_NEEDS_MODEL if (text_prompt and text_prompt.strip()) else
                             "No SAM model is installed, and the model-free stand-in needs a point, a box or a mask.")
        # L9.262 (CNP-NODES-1 audit): a text prompt that found nothing used to continue silently with no prompt
        if (strict_text and not has_prompt and text_bbox is None and text_prompt and text_prompt.strip()
                and grounding_model != "none"):
            raise ValueError(
                f"SAM Mask Generator: GroundingDINO found nothing for '{text_prompt.strip()}' (box threshold "
                f"{float(text_box_threshold):g}, text threshold {float(text_threshold):g}). Lower text_box_threshold, "
                "change the words, or add a point or box.")

        # ── Move model to GPU if offloaded ─────────────────────────────
        if offload and hasattr(model, "to"):
            model.to(target_device)

        try:
            if len(boxes) >= 2:
                # L9.262 (CNP-NODES-1 audit): each box runs on its own, so these prompts are not used - say so
                ignored = []
                if points_json and parse_points_json(points_json):
                    ignored.append("points_json")
                if bbox_json and bbox_json.strip():
                    ignored.append("bbox_json")
                if bbox is not None:
                    ignored.append("bbox")
                if apply_bbox_crop:
                    ignored.append("apply_bbox_crop")
                if ignored:
                    logging.getLogger("MEC").warning("SAM Mask Generator: %d boxes in bboxes_json - %s not used (each box is "
                                   "segmented on its own and the masks are joined).", len(boxes), ", ".join(ignored))
                union_mask, best_all, det, best_score, first_info = None, None, None, 0.0, None
                first_quality = None
                for bx in boxes:
                    r = self._run_inference(
                        model, model_type, model_info, image, "",
                        json.dumps(bx), None,
                        multimask_output, mask_index, score_threshold,
                        False, target_device, model_dtype,
                        refine_iterations, auto_negative_points, existing_mask,
                        negative_points_from_text, edge_refine, edge_radius, candidate_select,
                    )
                    union_mask = r[0] if union_mask is None else torch.maximum(union_mask, r[0])
                    if best_all is None:
                        best_all, first_info, first_quality = r[1], r[4], r[5]
                    if float(r[3]) > best_score:
                        best_score, det = float(r[3]), r[2]
                # Recompute bbox over the union
                if union_mask is not None:
                    nz = torch.nonzero(union_mask[0] > 0.5, as_tuple=False)
                    if nz.shape[0] > 0:
                        y0, y1 = int(nz[:, 0].min()), int(nz[:, 0].max())
                        x0, x1 = int(nz[:, 1].min()), int(nz[:, 1].max())
                        det = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
                try:
                    info_d = json.loads(first_info) if first_info else {}
                except (TypeError, ValueError):
                    info_d = {}
                info_d.update({"union_of_boxes": len(boxes), "best_score": best_score,
                               "detected_bbox": det or [0, 0, int(image.shape[2]), int(image.shape[1])]})
                if ignored:
                    info_d["ignored_inputs"] = ignored
                    info_d["note"] = ("bboxes_json has several boxes: each is segmented on its own and the masks are "
                                      "joined, so " + ", ".join(ignored) + " were not used.")
                if candidate_select != CANDIDATE_MODES[0]:      # L9.172: key only when the feature is on
                    info_d["candidate_select"] = candidate_select
                try:                                             # L9.172 quality: the first box run + a box count
                    q_d = json.loads(first_quality) if first_quality else {}
                except (TypeError, ValueError):
                    q_d = {}
                q_d["boxes"] = len(boxes)
                result = (union_mask, best_all if best_all is not None else union_mask,
                          det or [0, 0, int(image.shape[2]), int(image.shape[1])],
                          best_score, json.dumps(info_d, indent=2), json.dumps(q_d, indent=2))
            else:
                result = self._run_inference(
                    model, model_type, model_info, image, points_json, bbox_json, bbox,
                    multimask_output, mask_index, score_threshold,
                    apply_bbox_crop, target_device, model_dtype,
                    refine_iterations, auto_negative_points, existing_mask,
                    negative_points_from_text, edge_refine, edge_radius, candidate_select,
                )
            if neg_bbox is not None:
                result = self._subtract_negative(result, model, model_type, model_info, image, neg_bbox,
                                                 target_device, model_dtype, edge_refine, edge_radius)
        finally:
            # ── Offload back to CPU ────────────────────────────────────
            if offload and hasattr(model, "to"):
                model.to("cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return result

    def _run_inference(self, model, model_type, model_info, image, points_json, bbox_json,
                       bbox_input, multimask_output, mask_index, score_threshold,
                       apply_bbox_crop, device, dtype,
                       refine_iterations=1, auto_negative_points=False,
                       existing_mask=None, negative_points_from_text=None,
                       edge_refine="guided", edge_radius=8, candidate_select=CANDIDATE_MODES[0]):
        """One SAM run on image[0]. Returns (mask, all_masks, detected_bbox, score, info, quality_json); the last
        element (L9.172) is the candidate report."""
        mode = candidate_select if candidate_select in CANDIDATE_MODES else CANDIDATE_MODES[0]
        passes = []

        # Convert image: (B, H, W, C) float [0,1] → numpy uint8
        img_tensor = image[0]  # first image in batch
        img_np = (np.clip(img_tensor.cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)   # A22 L9.136: no HDR wrap
        H, W = img_np.shape[:2]

        # Parse points (shared utility)
        points_list = parse_points_json(points_json)
        point_coords, point_labels = None, None
        if points_list:
            coords = [[float(p["x"]), float(p["y"])] for p in points_list]
            labels = [int(p.get("label", 1)) for p in points_list]
            point_coords = np.array(coords, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)

        # ── Merge negative points from text prompt ─────────────────────
        if negative_points_from_text:
            neg_coords = np.array([[p[0], p[1]] for p in negative_points_from_text],
                                  dtype=np.float32)
            neg_labels = np.zeros(len(negative_points_from_text), dtype=np.int32)
            if point_coords is not None:
                point_coords = np.concatenate([point_coords, neg_coords], axis=0)
                point_labels = np.concatenate([point_labels, neg_labels], axis=0)
            else:
                point_coords = neg_coords
                point_labels = neg_labels

        # Parse bbox (shared utility)
        box_np = parse_bbox_input(bbox_json, bbox_input)

        # ── Build predictor (shared utility) ───────────────────────────
        predictor = get_sam_predictor(model, model_type, img_np)

        if predictor is None:
            # A22 L9.136: an empty mask here flowed into inpainting as if SAM had found nothing
            raise RuntimeError(f"SAM Mask Generator (MEC): no predictor fits this model ({model_type!r}); load it "
                               "with SAM Model Loader (MEC) (SAM 2.1) or use ComfyUI core's SAM 3 nodes.")

        # ── Use existing mask as starting logits if provided ───────────
        current_mask = None
        if existing_mask is not None:
            m = existing_mask[0] if existing_mask.dim() == 3 else existing_mask
            if m.shape[0] != H or m.shape[1] != W:
                import torch.nn.functional as F
                m = F.interpolate(m.unsqueeze(0).unsqueeze(0), (H, W),
                                  mode="bilinear", align_corners=False)[0, 0]
            current_mask = (m.cpu().numpy() > 0.5).astype(np.float32)

        # ── Iterative SAM refinement ───────────────────────────────────
        masks_np = None
        scores = None
        best_score = 0.0

        for iteration in range(refine_iterations):
            iter_coords = point_coords
            iter_labels = point_labels
            iter_box = box_np

            # Augment prompts from the previous pass - or, on the first pass, from existing_mask (A22 L9.136: it
            # was only read from pass 2, so at the default refine_iterations=1 it did nothing)
            if current_mask is not None:
                iter_coords, iter_labels, iter_box = augment_prompts_from_mask(
                    current_mask, point_coords, point_labels, box_np,
                    H, W, auto_negative=auto_negative_points,
                )

            # Prepare mask input (logits from the previous pass / existing_mask)
            mask_input = None
            if current_mask is not None:
                mask_input = mask_to_sam_logits(current_mask)

            # Run SAM
            try:
                masks_np, scores, _ = sam_predict(
                    predictor, model_info,
                    point_coords=iter_coords,
                    point_labels=iter_labels,
                    box=iter_box,
                    mask_input=mask_input,
                    multimask_output=multimask_output,
                )
            except TypeError:
                # Predictor doesn't support mask_input
                masks_np, scores, _ = sam_predict(
                    predictor, model_info,
                    point_coords=iter_coords,
                    point_labels=iter_labels,
                    box=iter_box,
                    multimask_output=multimask_output,
                )
            except Exception as exc:
                # A22 L9.136: the first pass failing used to return an all-zero mask (_fallback_predict). Later
                # refinement passes keep the mask of the pass before.
                if masks_np is None:
                    raise RuntimeError(f"SAM Mask Generator (MEC): SAM prediction failed: {type(exc).__name__}: "
                                       f"{exc}") from exc
                break

            if masks_np is None:
                break

            # Select best mask (L9.172: one rule for the passes and the final pick; the default is the old code)
            scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            pass_q = candidate_quality(masks_np, scores_list, img_np) if mode == "best quality" else None
            idx = _select_candidate(mode, scores_list, mask_index, score_threshold, pass_q, final=False)

            current_mask = masks_np[idx].astype(np.float32)
            best_score = float(scores_list[idx])
            passes.append({"pass": iteration, "selected": int(idx), "score": round(best_score, 4)})

            if best_score > 0.98:
                break

        # Fallback if nothing produced
        if masks_np is None:
            empty = torch.zeros(1, H, W, dtype=torch.float32)
            return (empty, empty, [0, 0, W, H], 0.0, "No masks generated",
                    json.dumps({"select": mode, "selected_index": None, "candidates": [], "passes": passes,
                                "reason": "No masks generated"}, indent=2))

        # Convert to tensors
        all_masks_t = torch.from_numpy(masks_np.astype(np.float32))
        if all_masks_t.dim() == 2:
            all_masks_t = all_masks_t.unsqueeze(0)

        scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)

        # Pick best mask by score or by index (L9.172: candidate_select; the default is the old code)
        final_q = candidate_quality(masks_np, scores_list, img_np)
        idx = _select_candidate(mode, scores_list, mask_index, score_threshold, final_q, final=True)
        if idx is None:
            # A22 L9.136: "discard masks below this score" - the below-threshold mask 0 came back anyway
            empty = torch.zeros(1, H, W, dtype=torch.float32)
            best = max(scores_list) if scores_list else 0.0
            why = (f"No mask reached score_threshold {score_threshold:g} (best score {best:.3f}); "
                   "lower score_threshold or add prompts.")
            return (empty, all_masks_t, [0, 0, W, H], float(best), why,
                    json.dumps({"select": mode, "selected_index": None, "candidates": [], "passes": passes,
                                "reason": why}, indent=2))

        selected_mask = all_masks_t[idx:idx+1]
        selected_score = float(scores_list[idx])

        # ── Edge-perfect refinement ───────────────────────────────────
        # SAM emits a blocky/binary mask; snap it to the true image edges → a
        # soft, roto-grade alpha (robust to motion blur + dull/bright colours).
        if edge_refine and edge_refine != "none":
            selected_mask = refine_edges(selected_mask, image[:1], edge_refine, int(edge_radius))

        # Detect bbox from selected mask
        coords = torch.nonzero(selected_mask[0] > 0.5, as_tuple=False)
        if coords.shape[0] > 0:
            y_min = int(coords[:, 0].min().item())
            y_max = int(coords[:, 0].max().item())
            x_min = int(coords[:, 1].min().item())
            x_max = int(coords[:, 1].max().item())
            det_bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
        else:
            det_bbox = [0, 0, W, H]

        # Crop to bbox if requested
        if apply_bbox_crop and box_np is not None:
            bx1, by1, bx2, by2 = [int(v) for v in box_np]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(W, bx2), min(H, by2)
            selected_mask = selected_mask[:, by1:by2, bx1:bx2]

        from ._download_policy import sam_model_status  # owner D7.91 status line
        info = json.dumps({
            "model_type": model_type,
            "model_status": sam_model_status(model_info),
            "num_masks": len(scores_list),
            "scores": scores_list,
            "selected_index": idx,
            "detected_bbox": det_bbox,
            "refine_iterations": refine_iterations,
            # L7.98: a model-free stand-in from the SAM Model Loader made this mask - say so
            **({"approximate": True, "model_free": model_info.get("fallback_from")}
               if isinstance(model_info, dict) and model_info.get("model_free") else {}),
            # L9.172: the key appears only when the feature is on, so a default run's info is unchanged
            **({"candidate_select": mode} if mode != CANDIDATE_MODES[0] else {}),
        }, indent=2)
        quality = json.dumps({"select": mode, "selected_index": int(idx), "candidates": final_q, "passes": passes},
                             indent=2)

        return (selected_mask, all_masks_t, det_bbox, selected_score, info, quality)

    def _run_objects(self, model, model_type, model_info, image, objects, multimask_output, mask_index,
                     score_threshold, device, dtype, refine_iterations, auto_negative_points, edge_refine,
                     edge_radius, candidate_select, ignored):
        """L9.172 objects_json: one single-run inference per entry on its own frame, gathered per object.
        Returns all seven outputs."""
        B, H, W = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        names = []
        for e in objects:
            if e["name"] not in names:
                names.append(e["name"])
        K = len(names)
        obj_masks = torch.zeros(K * B, H, W, dtype=torch.float32)
        per_entry, entry_q, scores = [], [], []
        rows = {n: {"name": n, "frames": [], "scores": []} for n in names}
        for e in objects:
            f = e["frame"]
            r = self._run_inference(
                model, model_type, model_info, image[f:f + 1],
                json.dumps(e["points"]) if e["points"] else "[]",
                json.dumps(e["bbox"]) if e["bbox"] else "", None,
                multimask_output, mask_index, score_threshold,
                False, device, dtype, refine_iterations, auto_negative_points, None, None,
                edge_refine, edge_radius, candidate_select,
            )
            m = r[0][0].float()
            if m.shape != (H, W):            # never cropped here (apply_bbox_crop is not used), but stay safe
                raise RuntimeError(f"SAM Mask Generator: object {e['name']!r} came back {tuple(m.shape)}, "
                                   f"expected {(H, W)}.")
            k = names.index(e["name"])
            obj_masks[k * B + f] = m
            per_entry.append(m)
            scores.append(float(r[3]))
            rows[e["name"]]["frames"].append(f)
            rows[e["name"]]["scores"].append(round(float(r[3]), 4))
            try:
                q = json.loads(r[5])
            except (TypeError, ValueError):
                q = {"raw": str(r[5])[:300]}
            entry_q.append({"name": e["name"], "frame": f, "quality": q})

        union = obj_masks.view(K, B, H, W).amax(dim=0)
        nz = torch.nonzero(union.amax(dim=0) > 0.5, as_tuple=False)
        if nz.shape[0] > 0:
            y0, y1 = int(nz[:, 0].min()), int(nz[:, 0].max())
            x0, x1 = int(nz[:, 1].min()), int(nz[:, 1].max())
            det = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
        else:
            det = [0, 0, W, H]
        notes = []
        if ignored:
            notes.append("objects_json is set, so " + ", ".join(ignored) + " were not used (each object runs on "
                         "its own prompts; apply_bbox_crop and existing_mask do not apply).")
        notes.append(_NO_PROPAGATION)
        from ._download_policy import sam_model_status
        info = json.dumps({"objects": [rows[n] for n in names], "notes": notes,
                           "model_status": sam_model_status(model_info)}, indent=2)
        quality = json.dumps({"objects": entry_q, **({"ignored_inputs": ignored} if ignored else {})}, indent=2)
        score = float(sum(scores) / len(scores)) if scores else 0.0
        return (union, torch.stack(per_entry), det, score, info, obj_masks, quality)

    @staticmethod
    def _fallback_predict(model, img_np, point_coords, point_labels, box_np, multimask):
        """Generic fallback using model's forward pass directly."""
        H, W = img_np.shape[:2]
        # Return an empty mask as fallback
        n = 3 if multimask else 1
        return np.zeros((n, H, W), dtype=np.float32), np.zeros(n, dtype=np.float32)

    @staticmethod
    def _text_to_bbox(image, text_prompt, grounding_model, text_threshold,
                      box_threshold, device, strict=False):
        """Run GroundingDINO to convert text prompt to bounding box.

        Returns np.ndarray [x1, y1, x2, y2] or None.
        """
        try:
            gdino_loaded = get_or_load_model(grounding_model, precision="fp32", device=device)
        except Exception as e:
            if strict:
                raise RuntimeError(f"SAM Mask Generator (MEC): the text prompt needs GroundingDINO "
                                   f"({grounding_model}), which failed to load: {e}") from e
            import logging
            logging.getLogger("MEC").warning("[MEC] GroundingDINO load failed: %s", e)
            return None

        gdino_model = gdino_loaded["model"]
        img_np = (np.clip(image[0].cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)   # A22 L9.136: no HDR wrap
        H, W = img_np.shape[:2]

        try:
            from groundingdino.util.inference import predict as gdino_predict
            from PIL import Image as PILImage
            import torchvision.transforms.functional as TF

            pil_img = PILImage.fromarray(img_np)
            # GroundingDINO expects a normalized tensor
            img_tensor = TF.to_tensor(pil_img)
            img_tensor = TF.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            boxes, logits, phrases = gdino_predict(
                gdino_model,
                img_tensor,
                text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            if len(boxes) == 0:
                return None

            # boxes are in cx, cy, w, h format normalized to [0,1]
            # Convert to x1, y1, x2, y2 absolute
            best_idx = logits.argmax().item()
            cx, cy, bw, bh = boxes[best_idx].tolist()
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            return np.array([x1, y1, x2, y2], dtype=np.float32)

        except Exception as e:
            if strict:
                raise RuntimeError(f"SAM Mask Generator (MEC): GroundingDINO failed on the text prompt "
                                   f"{text_prompt!r}: {type(e).__name__}: {e}") from e
            import logging
            logging.getLogger("MEC").warning("[MEC] GroundingDINO predict error: %s", e)
            return None

    def _subtract_negative(self, result, model, model_type, model_info, image, neg_bbox, device, dtype,
                           edge_refine, edge_radius):
        """negative_text_prompt: segment the detected object in its own SAM pass (its box as the prompt) and remove
        it from the mask and every candidate: out = mask * (1 - N)."""
        nr = self._run_inference(
            model, model_type, model_info, image, "", json.dumps([float(v) for v in neg_bbox]), None,
            False, 0, 0.0, False, device, dtype, 1, False, None, None, edge_refine, edge_radius,
            CANDIDATE_MODES[0],
        )
        neg = nr[0].to(result[0].device, result[0].dtype).clamp(0.0, 1.0)
        mask = (result[0] * (1.0 - neg)).clamp(0.0, 1.0)
        cands = result[1]
        if isinstance(cands, torch.Tensor) and cands.shape[-2:] == neg.shape[-2:]:
            cands = (cands * (1.0 - neg[:1])).clamp(0.0, 1.0)
        info_s = result[4]
        try:
            info_d = json.loads(info_s) if info_s else {}
            if isinstance(info_d, dict):
                info_d["negative_text"] = {"box_xyxy": [round(float(v), 1) for v in neg_bbox],
                                           "removed_px": int(((result[0] > 0.5) & (neg > 0.5)).sum()),
                                           "method": "own SAM pass, subtracted"}
                info_s = json.dumps(info_d, indent=2)
        except (TypeError, ValueError):
            pass
        return (mask, cands, result[2], result[3], info_s) + tuple(result[5:])

    @staticmethod
    def _sample_points_in_bbox(bbox_np, num_points: int = 5) -> list:
        """Sample evenly-distributed points within a bounding box.

        Args:
            bbox_np: np.ndarray [x1, y1, x2, y2]
            num_points: Number of points to sample

        Returns:
            List of [x, y] pairs inside the bbox
        """
        x1, y1, x2, y2 = bbox_np
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1

        if num_points == 1:
            return [[cx, cy]]

        points = [[cx, cy]]  # Always include center
        # Add points at 25% and 75% positions
        offsets = [0.25, 0.75]
        for ox in offsets:
            for oy in offsets:
                if len(points) >= num_points:
                    break
                points.append([x1 + w * ox, y1 + h * oy])
        return points[:num_points]


# A28 (ORDERS 29, docs/audit/A28/PARAM_CNP_A.md "SAMMaskGeneratorMEC"): which controls show when, read by js/_c2c_node_layout.js.
# Display only: every input keeps its name, order, type and default, is saved and reaches the node's function as before.
# The text prompt and its tuning do nothing with grounding_model 'none' (:255, :267); the points / box editor
# writes the four JSON prompts (never a text row).
try:
    from ._layout_decl import declare_layout, merge, sections
except ImportError:  # pragma: no cover - loaded outside the pack
    declare_layout = None
if declare_layout is not None:
    _SAMG_TEXT = {"grounding_model": {"not": ["none"]}}
    declare_layout(SAMMaskGeneratorMEC, {
        "points_json": {"c2c_hidden": True},
        "bbox_json": {"c2c_hidden": True},
        "bboxes_json": {"c2c_hidden": True},
        "spline_json": {"c2c_hidden": True},
        "text_prompt": {"c2c_show_if": _SAMG_TEXT},
        "negative_text_prompt": {"c2c_section": "Text", "c2c_show_if": _SAMG_TEXT},
        "text_threshold": {"c2c_section": "Text", "c2c_show_if": _SAMG_TEXT},
        "text_box_threshold": {"c2c_section": "Text", "c2c_show_if": _SAMG_TEXT},
        "multimask_output": {"c2c_section": "Candidates"},
        "mask_index": {"c2c_section": "Candidates", "c2c_show_if": {"multimask_output": [True]}},
        "score_threshold": {"c2c_section": "Candidates", "c2c_show_if": {"multimask_output": [True]}},
        "apply_bbox_crop": {"c2c_section": "Refine"},
        "refine_iterations": {"c2c_section": "Refine"},
        "auto_negative_points": {"c2c_section": "Refine"},
        "edge_radius": {"c2c_show_if": {"edge_refine": {"not": ["none"]}}},
        "allow_download": {"c2c_section": "Model setup", "c2c_show_if": _SAMG_TEXT},
        # L9.172: typed by the user, so shown (never c2c_hidden)
        "candidate_select": {"c2c_section": "Candidates"},
        "objects_json": {"c2c_section": "Objects"},
    })

# L9.172 (A25 spec item 49): inputs appended after the A28 audit, in order (read by tests/test_a28_cnp_a_layout.py)
APPENDED_AFTER_A28 = ("candidate_select", "objects_json")
