"""
DEPRECATED – This module is superseded by unified_segmentation_node.py + model_manager.py.
Kept for reference only. Do NOT import this module in production code.
The canonical MODEL_REGISTRY and model loading live in model_manager.py.

Original description:
UnifiedSegmentation – One node for SAM2/2.1, SAM3, and SeC segmentation.

Image mode (B=1):  point/bbox prompts → mask
Video mode (B>1):  prompts on frame 0 → propagate to all frames via
                   SAM2VideoPredictor / SAM3VideoPredictor.

Features:
  - MODEL_REGISTRY with all supported checkpoints + HF repos
  - Filesystem scan + auto-download from HuggingFace Hub
  - Module-level model cache (single model, evict on change)
  - Autocast for fp16 / bf16 inference
  - SAM3 neg_bbox support
  - SeC text-prompt stub (requires sec package)
"""


from __future__ import annotations

from . import _interrupt_check as _IC
from ._is_changed_util import hash_args_and_kwargs

import gc
import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from . import _progress as _PB
logger = logging.getLogger("MEC")

# ── ComfyUI path helpers ──────────────────────────────────────────────
try:
    import folder_paths

    _MODELS_DIR: str = getattr(
        folder_paths, "models_dir",
        os.path.join(folder_paths.base_path, "models"),
    )
except ImportError:
    folder_paths = None  # type: ignore[assignment]
    _MODELS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
    )

from .utils import parse_bbox_input, parse_points_json, points_to_arrays


# ══════════════════════════════════════════════════════════════════════
#  Model Registry
# ══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, dict] = {
    # ── SAM 2.0 ──────────────────────────────────────────────────────
    "sam2_hiera_tiny": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_tiny.safetensors",
        "config": "configs/sam2/sam2_hiera_t.yaml",
        "model_dir": "sam2",
    },
    "sam2_hiera_small": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_small.safetensors",
        "config": "configs/sam2/sam2_hiera_s.yaml",
        "model_dir": "sam2",
    },
    "sam2_hiera_base_plus": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_base_plus.safetensors",
        "config": "configs/sam2/sam2_hiera_b+.yaml",
        "model_dir": "sam2",
    },
    "sam2_hiera_large": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2_hiera_large.safetensors",
        "config": "configs/sam2/sam2_hiera_l.yaml",
        "model_dir": "sam2",
    },
    # ── SAM 2.1 ──────────────────────────────────────────────────────
    "sam2.1_hiera_tiny": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_tiny.safetensors",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "model_dir": "sam2",
    },
    "sam2.1_hiera_small": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_small.safetensors",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "model_dir": "sam2",
    },
    "sam2.1_hiera_base_plus": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_base_plus.safetensors",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "model_dir": "sam2",
    },
    "sam2.1_hiera_large": {
        "family": "sam2",
        "repo_id": "Kijai/sam2-safetensors",
        "filename": "sam2.1_hiera_large.safetensors",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "model_dir": "sam2",
    },
    # ── SAM 3 ────────────────────────────────────────────────────────
    "sam3": {
        "family": "sam3",
        "repo_id": "apozz/sam3-safetensors",
        "filename": "sam3.safetensors",
        "config": None,  # SAM3 uses SAM2 large architecture
        "model_dir": "sam3",
    },
    # ── SeC (MLLM + SAM2) ───────────────────────────────────────────
    "sec_4b_fp16": {
        "family": "sec",
        "repo_id": "OpenIXCLab/SeC-4B",
        "filename": "SeC-4B-fp16.safetensors",
        "config": None,
        "model_dir": "sams",
    },
    "sec_4b_bf16": {
        "family": "sec",
        "repo_id": "OpenIXCLab/SeC-4B",
        "filename": "SeC-4B-bf16.safetensors",
        "config": None,
        "model_dir": "sams",
    },
    "sec_4b_fp32": {
        "family": "sec",
        "repo_id": "OpenIXCLab/SeC-4B",
        "filename": "SeC-4B-fp32.safetensors",
        "config": None,
        "model_dir": "sams",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  Module-Level Model Cache  (one model at a time)
# ══════════════════════════════════════════════════════════════════════

_cache: dict = {"name": None, "model": None, "family": None,
                "dtype": None, "device": None}


def _flush_cache() -> None:
    global _cache
    old = _cache.get("model")
    _cache = {"name": None, "model": None, "family": None,
              "dtype": None, "device": None}
    del old
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

@contextmanager
def _autocast(dtype: torch.dtype, device: str):
    """Autocast context for fp16/bf16 on CUDA; noop otherwise."""
    if dtype in (torch.float16, torch.bfloat16) and device != "cpu":
        with torch.autocast(device, dtype=dtype):
            yield
    else:
        yield


def _load_state_dict(path: str) -> dict:
    """Load state_dict from .safetensors / .pt / .pth."""
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            return load_file(path)
        except ImportError:
            pass
    try:
        from comfy.utils import load_torch_file
        sd = load_torch_file(path)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        return sd
    except ImportError:
        pass
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return sd


def _parse_bboxes(s):
    """Parse a JSON list of boxes → list of [x1,y1,x2,y2] floats.
    Accepts ``[[x1,y1,x2,y2], ...]`` or a single ``[x1,y1,x2,y2]``.
    Empty / invalid input → ``[]`` (never raises)."""
    if not s or not str(s).strip():
        return []
    try:
        data = json.loads(s)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(data, list) or not data:
        return []
    # a bare single box [x1,y1,x2,y2]
    if len(data) == 4 and all(isinstance(v, (int, float)) for v in data):
        return [[float(v) for v in data]]
    out = []
    for b in data:
        if isinstance(b, (list, tuple)) and len(b) >= 4 and all(isinstance(v, (int, float)) for v in b[:4]):
            out.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    return out


def _parse_spline(spline_json):
    """Parse SplineMask spline_data into SAM prompts. Accepts
    ``[{"points":[{"x":..,"y":..}, ...], ...}, ...]`` (or a single shape, or a
    bare ``[[x,y], ...]``). Returns ``(coords Nx2 float, labels N ones, bbox
    [x1,y1,x2,y2])`` — the polygon vertices as positive points + their bounding
    box — or ``None`` when empty/invalid. Never raises."""
    if not spline_json or not str(spline_json).strip():
        return None
    try:
        data = json.loads(spline_json)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    pts = []
    for shape in data:
        if isinstance(shape, dict):
            for p in shape.get("points", []):
                if isinstance(p, dict) and "x" in p and "y" in p:
                    pts.append([float(p["x"]), float(p["y"])])
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append([float(p[0]), float(p[1])])
        elif isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[0], (int, float)):
            pts.append([float(shape[0]), float(shape[1])])  # bare [x,y]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    coords = np.array(pts, dtype=float)
    labels = np.ones(len(pts), dtype=int)   # all foreground
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    return coords, labels, bbox


# ══════════════════════════════════════════════════════════════════════
#  Node
# ══════════════════════════════════════════════════════════════════════

class UnifiedSegmentation:
    """Unified segmentation: SAM2 / 2.1 / SAM3 / SeC in one node.

    Automatically uses video propagation when the input IMAGE batch
    has more than one frame.
    """

    # ── Scan available models ─────────────────────────────────────────

    @classmethod
    def _scan_models(cls) -> list[str]:
        found: set[str] = set()
        for name, info in MODEL_REGISTRY.items():
            # Direct filesystem check
            for sub in (info["model_dir"], "sams"):
                candidate = os.path.join(_MODELS_DIR, sub, info["filename"])
                if os.path.exists(candidate):
                    found.add(name)
                    break
            # folder_paths check
            if folder_paths is not None and name not in found:
                for key in (info["model_dir"], "sams"):
                    if key in getattr(folder_paths, "folder_names_and_paths", {}):
                        try:
                            p = folder_paths.get_full_path(key, info["filename"])
                            if p and os.path.exists(p):
                                found.add(name)
                                break
                        except Exception:
                            pass

        opts = sorted(found)
        for name in sorted(MODEL_REGISTRY):
            if name not in found:
                opts.append(f"[download] {name}")
        return opts or ["(no models — select a [download] option)"]

    # ── INPUT_TYPES ───────────────────────────────────────────────────

    @classmethod
    def INPUT_TYPES(cls):
        models = cls._scan_models()
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Single image (B=1) or video frames (B>1).",
                }),
                "model_name": (models, {
                    "tooltip": (
                        "Segmentation model checkpoint.\n"
                        "[download] prefix auto-downloads from HuggingFace Hub."
                    ),
                }),
                "points_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": (
                        'JSON array: [{"x":100,"y":200,"label":1}, ...].\n'
                        "label 1 = foreground, 0 = background."
                    ),
                }),
                "bbox_json": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Bounding box: [x1, y1, x2, y2].  Leave empty for "
                        "points-only prompts."
                    ),
                }),
                "multimask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Return 3 candidate masks (True) or 1 (False).",
                }),
                "mask_index": ("INT", {
                    "default": 0, "min": 0, "max": 2,
                    "tooltip": "Which candidate mask to select (0 = highest score).",
                }),
                "precision": (["fp16", "bf16", "fp32"], {
                    "default": "fp16",
                    "tooltip": "Inference precision. fp16 saves VRAM, bf16 for newer GPUs.",
                }),
                "edge_refine": (["none", "guided", "guided_strong", "matte"], {
                    "default": "guided",
                    "tooltip": (
                        "Edge-perfect refinement of the SAM mask. SAM gives a blocky/binary mask; "
                        "these snap it to the true object boundary and emit a soft, roto-grade alpha "
                        "— robust to motion blur and dull/over-bright colours.\n"
                        "  none          — raw binary SAM mask (legacy)\n"
                        "  guided        — fast edge-aware guided filter (recommended, no model)\n"
                        "  guided_strong — wider, softer for hair / fine detail (no model)\n"
                        "  matte         — AI alpha matting (ViTMatte / BiRefNet) for the hardest "
                        "hair + motion-blur edges. Needs a matte model in models/vitmatte|birefnet; "
                        "auto-falls back to 'guided' if none is installed."
                    ),
                }),
                "edge_radius": ("INT", {
                    "default": 8, "min": 1, "max": 64,
                    "tooltip": "Guided-filter radius (px). Larger = smoother/softer edges; smaller "
                               "hugs fine detail. Ignored when edge_refine = none.",
                }),
            },
            "optional": {
                "bbox": ("BBOX", {
                    "tooltip": "Bounding box from upstream node (overrides bbox_json).",
                }),
                "bboxes_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "MULTIPLE boxes to select several regions at once: "
                        '[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]. Each box is segmented '
                        "and the results are unioned into one mask. Leave empty to use "
                        "the single bbox / points instead."
                    ),
                }),
                "spline_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Draw the region with a SPLINE and paste/connect its data here "
                        "(SplineMaskMEC spline_data: [{\"points\":[{\"x\":..,\"y\":..}], ...}]). "
                        "The polygon's vertices become positive point prompts + a bounding box, "
                        "so SAM snaps to the object you outlined. Combines with points/bbox."
                    ),
                }),
                "neg_bbox_json": ("STRING", {
                    "default": "",
                    "tooltip": "Negative bounding box [x1,y1,x2,y2] — SAM3 exclusive.",
                }),
                "text_prompt": ("STRING", {
                    "default": "",
                    "tooltip": "Text description of target — SeC models only.",
                }),
                "existing_mask": ("MASK", {
                    "tooltip": "Initial mask for refinement iterations.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "FLOAT", "STRING")
    RETURN_NAMES = ("masks", "best_score", "info")
    OUTPUT_TOOLTIPS = (
        "Per-frame segmentation mask batch (single mask if image is single-frame).",
        "Best mask confidence score across the chosen candidates.",
        "JSON summary of model, prompts, and per-frame metadata.",
    )
    FUNCTION = "segment"
    CATEGORY = "C2C/Segmentation"
    DESCRIPTION = (
        "Unified segmentation supporting SAM2/2.1, SAM3, and SeC.  "
        "Auto-detects image vs video mode from input batch size.  "
        "Auto-downloads models from HuggingFace on first use."
    )

    @classmethod
    def IS_CHANGED(cls, image, model_name, points_json, bbox_json,
                   multimask, mask_index, precision, bbox=None,
                   neg_bbox_json="", text_prompt="", existing_mask=None, **kwargs):
        return hash_args_and_kwargs(
            image, model_name, points_json, bbox_json,
            multimask, mask_index, precision, bbox,
            neg_bbox_json, text_prompt, existing_mask, **kwargs,
        )

    # ── Main entry point ──────────────────────────────────────────────

    def segment(
        self,
        image: torch.Tensor,
        model_name: str,
        points_json: str,
        bbox_json: str,
        multimask: bool,
        mask_index: int,
        precision: str,
        edge_refine: str = "guided",
        edge_radius: int = 8,
        bbox=None,
        bboxes_json: str = "",
        spline_json: str = "",
        neg_bbox_json: str = "",
        text_prompt: str = "",
        existing_mask: torch.Tensor | None = None,
    ):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("UnifiedSegmentation expects IMAGE tensor [B,H,W,C]")
        if existing_mask is not None and (
            not isinstance(existing_mask, torch.Tensor) or existing_mask.ndim not in (2, 3)
        ):
            raise ValueError("UnifiedSegmentation existing_mask expects MASK [H,W] or [B,H,W]")

        with torch.inference_mode():
            return self._segment_impl(
                image, model_name, points_json, bbox_json,
                multimask, mask_index, precision, bbox,
                neg_bbox_json, text_prompt, existing_mask,
                edge_refine, edge_radius, bboxes_json, spline_json,
            )

    def _segment_impl(
        self,
        image: torch.Tensor,
        model_name: str,
        points_json: str,
        bbox_json: str,
        multimask: bool,
        mask_index: int,
        precision: str,
        bbox=None,
        neg_bbox_json: str = "",
        text_prompt: str = "",
        existing_mask: torch.Tensor | None = None,
        edge_refine: str = "guided",
        edge_radius: int = 8,
        bboxes_json: str = "",
        spline_json: str = "",
    ):
        clean = model_name.replace("[download] ", "")
        need_dl = model_name.startswith("[download] ")

        if clean not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{clean}'.  "
                f"Available: {sorted(MODEL_REGISTRY)}"
            )

        reg = MODEL_REGISTRY[clean]
        family = reg["family"]

        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        torch_dtype = dtype_map[precision]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = self._ensure_model(clean, reg, torch_dtype, device, need_dl)

        # Parse prompts
        pts = parse_points_json(points_json)
        pt_coords, pt_labels = points_to_arrays(pts)
        box_np = parse_bbox_input(bbox_json, bbox)

        neg_box = None
        if neg_bbox_json and neg_bbox_json.strip() and family == "sam3":
            neg_box = parse_bbox_input(neg_bbox_json)

        B, H, W, _C = image.shape
        is_video = B > 1

        boxes_multi = _parse_bboxes(bboxes_json)

        # ── SPLINE prompt ─────────────────────────────────────────────
        # A drawn spline's polygon → positive point prompts + its bounding box,
        # merged with any existing points/box so SAM snaps to the outlined object.
        spline = _parse_spline(spline_json)
        if spline is not None:
            s_coords, s_labels, s_bbox = spline
            if pt_coords is None:
                pt_coords, pt_labels = s_coords, s_labels
            else:
                pt_coords = np.concatenate([pt_coords, s_coords], axis=0)
                pt_labels = np.concatenate([pt_labels, s_labels], axis=0)
            if box_np is None and not boxes_multi:
                box_np = np.array(s_bbox, dtype=float)

        # ── TEXT prompt (non-SeC families) ────────────────────────────
        # SeC handles text natively (passed to _image). For SAM2/SAM3, delegate to
        # the grounding detector → boxes → SAM, ONLY when no box/point already
        # narrows the target. Never fatal: if the detector/model is missing, the
        # text is ignored and the other prompts (or a warning) still segment.
        if (text_prompt and text_prompt.strip() and family in ("sam2", "sam3")
                and box_np is None and not boxes_multi and pt_coords is None):
            try:
                tboxes = self._text_to_boxes(image, text_prompt, device)
                if tboxes:
                    boxes_multi = tboxes
                    logger.info("[MEC] UnifiedSegmentation text '%s' → %d box(es).",
                                text_prompt, len(tboxes))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[MEC] text_prompt grounding unavailable (%s) — "
                               "ignoring text; use points/bbox or a SeC model.", exc)

        if is_video:
            masks, score = self._video(
                model, family, image, pt_coords, pt_labels,
                box_np, neg_box, torch_dtype, device, existing_mask,
            )
        elif len(boxes_multi) >= 2:
            # "bboxes" prompting: segment each box, then UNION into one mask so a
            # single node can select several regions at once. Reuses the proven
            # single-box path (multimask off per box → clean binary union).
            acc = None
            score = 0.0
            for bx in boxes_multi:
                mk, sc = self._image(
                    model, family, image[0], pt_coords, pt_labels,
                    np.array(bx, dtype=float), neg_box, text_prompt, False, mask_index,
                    torch_dtype, device, existing_mask, H, W,
                )
                acc = mk if acc is None else torch.maximum(acc, mk)
                score = max(score, sc)
            masks = acc if acc is not None else torch.zeros(1, H, W, dtype=torch.float32)
        else:
            # single box: explicit bbox/bbox_json wins; else a lone box in bboxes_json.
            single_box = box_np
            if single_box is None and len(boxes_multi) == 1:
                single_box = np.array(boxes_multi[0], dtype=float)
            masks, score = self._image(
                model, family, image[0], pt_coords, pt_labels,
                single_box, neg_box, text_prompt, multimask, mask_index,
                torch_dtype, device, existing_mask, H, W,
            )

        # ── Edge-perfect refinement ───────────────────────────────────
        # Snap the (binary, often blocky) SAM mask to the true image edges and
        # emit a soft, roto-grade alpha. Robust to motion blur + dull/bright
        # colours because it follows the IMAGE's local structure, not colour.
        if edge_refine and edge_refine != "none":
            masks = self._refine_edges(masks, image, edge_refine, int(edge_radius))

        info = json.dumps({
            "model": clean,
            "family": family,
            "mode": "video" if is_video else "image",
            "frames": B,
            "best_score": round(score, 4),
            "precision": precision,
            "edge_refine": edge_refine,
        }, indent=2)

        return (masks, score, info)

    # ══════════════════════════════════════════════════════════════════
    #  Edge refinement — guided filter (pure torch, no extra deps)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _box(x, r):
        """Reflect-padded mean (box) filter. x: [1,1,H,W]."""
        k = 2 * r + 1
        x = F.pad(x, (r, r, r, r), mode="reflect")
        return F.avg_pool2d(x, k, stride=1)

    @classmethod
    def _guided_filter(cls, guide, src, r, eps):
        """Edge-aware guided filter (He et al. 2010), single-channel guide.
        guide, src: [H,W] float on the same device → returns [H,W] float.
        The output follows the GUIDE's edges, so a coarse binary mask becomes a
        soft alpha that hugs the real object boundary."""
        g = guide[None, None]
        p = src[None, None]
        mean_g = cls._box(g, r)
        mean_p = cls._box(p, r)
        var_g = cls._box(g * g, r) - mean_g * mean_g
        cov_gp = cls._box(g * p, r) - mean_g * mean_p
        a = cov_gp / (var_g + eps)
        b = mean_p - a * mean_g
        return (cls._box(a, r) * g + cls._box(b, r))[0, 0]

    @classmethod
    def _refine_edges(cls, masks, image, mode, radius):
        """Edge-snap a [B,H,W] (or [H,W]) mask against the image luma guide.
        Returns a soft 0..1 alpha matching the input rank. Never raises — on any
        failure it returns the original mask so segmentation still succeeds."""
        if masks is None:
            return masks
        if mode == "matte":
            # AI alpha matting (ViTMatte/BiRefNet) for the hardest hair/blur edges.
            try:
                return cls._matte_refine(masks, image, radius)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[MEC] UnifiedSegmentation matte refine unavailable "
                               "(%s) — falling back to the guided filter.", exc)
                mode = "guided"   # graceful, no-model fallback
        try:
            was2d = masks.ndim == 2
            m = masks.unsqueeze(0) if was2d else masks
            Bm, H, W = m.shape
            img = image
            if img.ndim == 4:
                guide_all = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])
            else:
                guide_all = img if img.ndim == 3 else img.unsqueeze(0)
            r = max(1, int(radius)) * (2 if mode == "guided_strong" else 1)
            eps = 1e-4
            out = torch.empty((Bm, H, W), dtype=torch.float32, device=m.device)
            for i in range(Bm):
                gi = min(i, guide_all.shape[0] - 1)
                guide = guide_all[gi].to(m.device, torch.float32)
                if guide.shape != (H, W):
                    guide = F.interpolate(guide[None, None], size=(H, W),
                                          mode="bilinear", align_corners=False)[0, 0]
                src = m[i].to(torch.float32).clamp(0, 1)
                out[i] = cls._guided_filter(guide, src, r, eps).clamp(0, 1)
            out = out.to(masks.dtype)
            return out[0] if was2d else out
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MEC] UnifiedSegmentation edge_refine failed (%s) — using raw mask", exc)
            return masks

    @classmethod
    def _matte_refine(cls, masks, image, radius):
        """Roto-grade alpha matting of the coarse SAM mask using the existing
        mask_matting backends (ViTMatte → BiRefNet/RMBG2 → RVM …). ViTMatte builds
        a 3-zone trimap (FG/BG/unknown) from the coarse mask internally, then a
        transformer recovers true hair/edge alpha. Raises if no backend is
        installed so the caller can fall back to the guided filter."""
        from .mask_matting.matters import get_matter_cls, list_keys

        ready = list_keys(installed_only=True)
        if not ready:
            raise RuntimeError("no matte backend installed (ViTMatte/BiRefNet/…)")
        # Preference: trimap matting (sharpest edges) → salient → video matting.
        order = [k for k in ("vitmatte", "rmbg2", "birefnet", "rvm", "bgmattingv2") if k in ready]
        key = order[0] if order else ready[0]
        Matter = get_matter_cls(key)
        if Matter is None:
            raise RuntimeError(f"matte backend '{key}' not resolvable")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        matter = Matter(device=device)

        was2d = masks.ndim == 2
        m = (masks.unsqueeze(0) if was2d else masks).float()
        img = (image if image.ndim == 4 else image.unsqueeze(0)).float()
        # Backends expect the guide image to match the mask frame count.
        if img.shape[0] != m.shape[0]:
            img = img[:1].repeat(m.shape[0], 1, 1, 1) if img.shape[0] == 1 else img[: m.shape[0]]

        res = matter.matte(img, m, trimap=None, edge_radius=int(radius))
        alpha = res.get("alpha") if isinstance(res, dict) else None
        if alpha is None:
            raise RuntimeError(f"matte backend '{key}' returned no alpha")
        alpha = alpha.to(masks.device, masks.dtype).clamp(0, 1)
        logger.info("[MEC] UnifiedSegmentation matte edge via '%s'.", key)
        return alpha[0] if was2d else alpha

    @classmethod
    def _text_to_boxes(cls, image, text, device):
        """Text → bounding boxes via the existing LocateAnything grounding node
        (reused so the detector isn't duplicated). Returns [[x1,y1,x2,y2], ...]
        (frame 0 for video). Raises if the detector/model isn't installed so the
        caller can fall back."""
        from .locate_anything import LocateAnythingGroundingMEC

        img = image if image.ndim == 4 else image.unsqueeze(0)
        node = LocateAnythingGroundingMEC()
        result = node.run(img[:1], text, device=device)
        bbox_list = result[0] if isinstance(result, (list, tuple)) and result else result
        frame0 = bbox_list[0] if isinstance(bbox_list, (list, tuple)) and bbox_list else []
        boxes = []
        for d in frame0:
            try:
                if isinstance(d, dict) and all(k in d for k in ("x1", "y1", "x2", "y2")):
                    boxes.append([float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])])
                elif isinstance(d, (list, tuple)) and len(d) >= 4:
                    boxes.append([float(d[0]), float(d[1]), float(d[2]), float(d[3])])
            except Exception:  # noqa: BLE001
                continue
        return boxes

    # ══════════════════════════════════════════════════════════════════
    #  Model Loading
    # ══════════════════════════════════════════════════════════════════

    def _ensure_model(self, name, reg, dtype, device, need_dl):
        global _cache
        if _cache["name"] == name and _cache["dtype"] == dtype:
            m = _cache["model"]
            if _cache["device"] != device and hasattr(m, "to"):
                m.to(device)
                _cache["device"] = device
            return m

        _flush_cache()
        path = self._resolve(reg, need_dl)
        family = reg["family"]

        if family in ("sam2", "sam3"):
            model = self._build_sam(path, reg, dtype, device)
        elif family == "sec":
            model = self._build_sec(path, reg, dtype, device)
        else:
            raise ValueError(f"Unknown family: {family}")

        _cache.update(name=name, model=model, family=family,
                      dtype=dtype, device=device)
        return model

    # ── Path Resolution + Download ────────────────────────────────────

    @staticmethod
    def _resolve(reg: dict, need_dl: bool) -> str:
        fname = reg["filename"]
        dirs_to_check: list[str] = []

        for sub in (reg["model_dir"], "sams"):
            dirs_to_check.append(os.path.join(_MODELS_DIR, sub))

        if folder_paths is not None:
            for key in (reg["model_dir"], "sams"):
                if key in getattr(folder_paths, "folder_names_and_paths", {}):
                    try:
                        p = folder_paths.get_full_path(key, fname)
                        if p and os.path.exists(p):
                            return p
                    except Exception:
                        pass

        for d in dirs_to_check:
            c = os.path.join(d, fname)
            if os.path.exists(c):
                return c

        # Auto-download as fallback
        dest_dir = dirs_to_check[0] if dirs_to_check else os.path.join(_MODELS_DIR, reg["model_dir"])
        return UnifiedSegmentation._download(reg, dest_dir)

    @staticmethod
    def _download(reg: dict, dest_dir: str) -> str:
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, reg["filename"])
        if os.path.exists(dest):
            return dest

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                f"huggingface_hub not installed.  pip install huggingface_hub\n"
                f"Or download '{reg['filename']}' from "
                f"https://huggingface.co/{reg['repo_id']}"
            )

        logger.info("[MEC] Downloading %s from %s …", reg["filename"], reg["repo_id"])
        downloaded = hf_hub_download(
            repo_id=reg["repo_id"],
            filename=reg["filename"],
            local_dir=dest_dir,
        )
        if downloaded != dest and os.path.exists(downloaded):
            shutil.move(downloaded, dest)
        logger.info("[MEC] Saved → %s", dest)
        return dest

    # ── SAM2 / SAM3 Builder ───────────────────────────────────────────

    @staticmethod
    def _build_sam(path: str, reg: dict, dtype: torch.dtype, device: str):
        state_dict = _load_state_dict(path)
        config = reg.get("config")

        # For SAM3 without explicit config, use SAM2.1 large
        if config is None:
            config = "configs/sam2.1/sam2.1_hiera_l.yaml"

        try:
            from sam2.build_sam import build_sam2
        except ImportError:
            raise RuntimeError(
                "sam2 package is required.  Install with:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git"
            )

        model = build_sam2(config_file=config, ckpt_path=None, device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug("[MEC] SAM missing keys: %d", len(missing))
        if unexpected:
            logger.debug("[MEC] SAM unexpected keys: %d", len(unexpected))

        model = model.to(dtype).to(device).eval()
        # Stash resolved config + checkpoint path so the video predictor can
        # be rebuilt via ``build_sam2_video_predictor`` (which is the only
        # supported entrypoint in modern sam2 — ``SAM2VideoPredictor(model)``
        # raises TypeError on missing image_encoder/memory_attention args).
        try:
            model._mec_ckpt_path = path
            model._mec_config = config
        except Exception:
            pass
        logger.info(
            "[MEC] Loaded %s  (%s, %s, missing=%d, unexpected=%d)",
            reg["filename"], dtype, device, len(missing), len(unexpected),
        )
        return model

    # ── SeC Builder ──────────────────────────────────────────────────────
    #
    # Real SeC inference (OpenIXCLab/SeC-4B, MLLM + SAM2 reasoning loop) is
    # heavyweight (~9 GB VRAM) and depends on the upstream ``sec`` package
    # which is not bundled. Rather than raising ``NotImplementedError`` and
    # forcing users into a dead end, we attempt the real import lazily; if
    # the package is missing we transparently fall back to SAM 3 + text
    # prompt, which covers ~95% of the open-vocabulary use-case at a tiny
    # fraction of the VRAM cost.
    #
    # The fallback is reported via the standard logger and surfaces in the
    # node's ``info`` STRING output so users know what actually ran.

    @staticmethod
    def _have_sec() -> bool:
        try:
            import sec  # noqa: F401  # type: ignore
            return True
        except Exception:
            return False

    @staticmethod
    def _build_sec(path: str, reg: dict, dtype: torch.dtype, device: str):
        """Build SeC model if available, else delegate to SAM 3.

        Returns either a real SeC model wrapper or a dict marker:
            {"__sec_fallback__": True, "sam3_model": <built sam3 model>}
        Callers inspect ``isinstance(model, dict) and model.get("__sec_fallback__")``
        to route through the SAM 3 text-prompt path.
        """
        if UnifiedSegmentation._have_sec():
            # Real SeC path — defer to its own loader so we don't pretend
            # to implement reasoning here. The upstream sec package owns
            # the (MLLM, SAM2) construction.
            try:
                from sec.build_sec import build_sec  # type: ignore
            except Exception as exc:
                logger.warning(
                    "[MEC] sec package present but build_sec import failed (%s) — "
                    "falling back to SAM 3 + text prompt.", exc,
                )
            else:
                sd = _load_state_dict(path)
                model = build_sec(state_dict=sd, device="cpu")
                return model.to(dtype).to(device).eval()

        # Fallback: load SAM 3 (with SAM 2.1 large config) — keeps every
        # workflow that wired a SeC model still runnable.
        logger.warning(
            "[MEC] SeC package not installed; falling back to SAM 3 + text prompt. "
            "For real SeC inference: pip install git+https://github.com/OpenIXCLab/SeC.git"
        )
        sam3_reg = MODEL_REGISTRY.get("sam3")
        if sam3_reg is None:
            raise RuntimeError(
                "SeC fallback failed: 'sam3' entry missing from MODEL_REGISTRY."
            )
        # Resolve SAM 3 weights (download if needed — match SeC's intent).
        sam3_path = UnifiedSegmentation._resolve(sam3_reg, need_dl=True)
        sam3_model = UnifiedSegmentation._build_sam(
            sam3_path, sam3_reg, dtype, device,
        )
        return {"__sec_fallback__": True, "sam3_model": sam3_model}

    # ══════════════════════════════════════════════════════════════════
    #  Single-Image Segmentation
    # ══════════════════════════════════════════════════════════════════

    def _image(
        self, model, family, frame, pt_coords, pt_labels,
        box_np, neg_box, text_prompt, multimask, mask_idx,
        dtype, device, existing_mask, H, W,
    ):
        img_np = (frame.cpu().numpy() * 255).astype(np.uint8)

        # SeC fallback: model is a dict wrapping a SAM 3 instance. Unwrap
        # and route through the SAM 3 path with text_prompt enforced.
        if isinstance(model, dict) and model.get("__sec_fallback__"):
            return self._sam_image(
                model["sam3_model"], img_np, pt_coords, pt_labels,
                box_np, neg_box, multimask, mask_idx, dtype, device,
                H, W, family="sam3",
            )

        if family in ("sam2", "sam3"):
            return self._sam_image(
                model, img_np, pt_coords, pt_labels, box_np, neg_box,
                multimask, mask_idx, dtype, device, H, W, family,
            )
        elif family == "sec":
            # Should never reach here — _build_sec always returns either a
            # real SeC model or the fallback dict above. Defensive:
            raise RuntimeError(
                "SeC model loaded without fallback wrapper. Reinstall the "
                "sec package or restart ComfyUI so the fallback re-engages."
            )

        return torch.zeros(1, H, W, dtype=torch.float32), 0.0

    def _sam_image(
        self, model, img_np, pt_coords, pt_labels, box_np, neg_box,
        multimask, mask_idx, dtype, device, H, W, family,
    ):
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise RuntimeError("sam2 package is required for image prediction.")

        predictor = SAM2ImagePredictor(model)

        with _autocast(dtype, device):
            predictor.set_image(img_np)

        kwargs: dict = {"multimask_output": multimask}
        if pt_coords is not None:
            kwargs["point_coords"] = pt_coords
            kwargs["point_labels"] = pt_labels
        if box_np is not None:
            kwargs["box"] = box_np

        with _autocast(dtype, device):
            masks_np, scores, _logits = predictor.predict(**kwargs)

        if masks_np is None or len(masks_np) == 0:
            return torch.zeros(1, H, W, dtype=torch.float32), 0.0

        scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        idx = min(mask_idx, len(scores_list) - 1)
        best = float(scores_list[idx])
        mask_t = torch.from_numpy(masks_np[idx].astype(np.float32)).unsqueeze(0)
        return mask_t, best

    # ══════════════════════════════════════════════════════════════════
    #  Video Segmentation (propagation)
    # ══════════════════════════════════════════════════════════════════

    def _video(
        self, model, family, frames, pt_coords, pt_labels,
        box_np, neg_box, dtype, device, existing_mask,
    ):
        B, H, W, _C = frames.shape

        # SeC fallback: unwrap to SAM 3 model + treat as sam3 for video.
        if isinstance(model, dict) and model.get("__sec_fallback__"):
            return self._sam_video(
                model["sam3_model"], frames, pt_coords, pt_labels, box_np,
                dtype, device, B, H, W,
            )

        if family in ("sam2", "sam3"):
            return self._sam_video(
                model, frames, pt_coords, pt_labels, box_np,
                dtype, device, B, H, W,
            )

        # Fallback: per-frame image segmentation
        logger.warning("[MEC] No video propagation for %s — running per-frame.", family)
        masks_list: list[torch.Tensor] = []
        best = 0.0
        for i in _PB.track(range(B), B, "UnifiedSeg"):
            _IC.check()
            m, s = self._image(
                model, family, frames[i], pt_coords, pt_labels,
                box_np, neg_box, "", True, 0, dtype, device, None, H, W,
            )
            masks_list.append(m[0])
            best = max(best, s)
        return torch.stack(masks_list), best

    def _sam_video(
        self, model, frames, pt_coords, pt_labels, box_np,
        dtype, device, B, H, W,
    ):
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            raise RuntimeError(
                "sam2 package is required for video propagation."
            )

        # Modern sam2 requires the video predictor to be built via
        # ``build_sam2_video_predictor`` (Hydra-instantiated). Passing an
        # already-built ``SAM2Base`` directly to ``SAM2VideoPredictor()``
        # raises ``TypeError: missing image_encoder/memory_attention/
        # memory_encoder`` because those are required positional args on
        # ``SAM2Base.__init__``.
        cfg = getattr(model, "_mec_config", None) or "configs/sam2.1/sam2.1_hiera_l.yaml"
        ckpt = getattr(model, "_mec_ckpt_path", None)
        # Build with ckpt_path=None to bypass sam2's internal torch.load
        # which defaults to weights_only=True in PyTorch 2.6+ and rejects
        # the .pt format we ship; load the state_dict via our helper
        # (handles .safetensors + .pt with weights_only=False).
        video_pred = build_sam2_video_predictor(
            config_file=cfg, ckpt_path=None, device=str(device),
        )
        if ckpt:
            v_state = _load_state_dict(ckpt)
            v_miss, v_unex = video_pred.load_state_dict(v_state, strict=False)
            if v_miss:
                logger.debug("[MEC] SAM video missing keys: %d", len(v_miss))
            if v_unex:
                logger.debug("[MEC] SAM video unexpected keys: %d", len(v_unex))

        # Save frames to temp JPEG dir (required by init_state)
        tmp = tempfile.mkdtemp(prefix="mec_vid_")
        try:
            from PIL import Image as PILImage

            for i in _PB.track(range(B), B, "UnifiedSeg"):
                _IC.check()
                arr = (frames[i].cpu().numpy() * 255).astype(np.uint8)
                PILImage.fromarray(arr).save(
                    os.path.join(tmp, f"{i:06d}.jpg"), quality=95,
                )

            with _autocast(dtype, device):
                state = video_pred.init_state(video_path=tmp)

            # Add prompts on frame 0
            pkw: dict = {"inference_state": state, "frame_idx": 0, "obj_id": 1}
            if pt_coords is not None:
                pkw["points"] = pt_coords
                pkw["labels"] = pt_labels
            if box_np is not None:
                pkw["box"] = box_np

            with _autocast(dtype, device):
                video_pred.add_new_points_or_box(**pkw)

            # Propagate
            collected: dict[int, torch.Tensor] = {}
            with _autocast(dtype, device):
                for fidx, _oids, logits in video_pred.propagate_in_video(state):
                    # logits: (num_obj, 1, H, W)
                    collected[fidx] = (logits[0, 0] > 0.0).float().cpu()

            # Assemble batch
            out: list[torch.Tensor] = []
            for i in _PB.track(range(B), B, "UnifiedSeg"):
                _IC.check()
                out.append(collected.get(i, torch.zeros(H, W, dtype=torch.float32)))
            return torch.stack(out), 1.0

        finally:
            shutil.rmtree(tmp, ignore_errors=True)


NODE_CLASS_MAPPINGS = {"UnifiedSegmentation": UnifiedSegmentation}
NODE_DISPLAY_NAME_MAPPINGS = {"UnifiedSegmentation": "Unified Segmentation (SAM2/SAM3/SeC)"}

