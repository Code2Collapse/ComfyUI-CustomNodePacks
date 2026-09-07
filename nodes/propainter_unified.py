"""
ProPainterMEC — unified ProPainter node.
========================================

Absorbs the five legacy ProPainter nodes into one mode-switched node:

    mode = "temporal"      -> ProPainterTemporalMEC      (video inpaint inside InpaintCrop/Stitch flow)
    mode = "remove"        -> ProPainterRemoveMEC        (model-free object/wire/plate removal)
    mode = "stitch"        -> ProPainterStitchMEC        (flow-aware composite of inpainted crop)
    mode = "stitch_refine" -> ProPainterStitchRefineMEC  (seam ring refinement of already-stitched output)
    mode = "flow"          -> FlowRefineMEC              (RAFT optical flow viz / warp / consistency)

Helper classes are kept on disk as importable Python classes; only this
single node is registered with ComfyUI. Lazy imports keep load-time cost
near zero.

Outputs are a fixed superset so downstream wires never break across modes:

    (IMAGE image_out, MASK mask_out, IMAGE aux_image, MASK aux_mask, STRING info)

For each mode the slots carry:

    temporal      : (inpainted, zeros,         zeros,                   zeros,        info)
    remove        : (filled,    fill_mask,     zeros,                   zeros,        info)
    stitch        : (stitched,  repaint_mask,  zeros,                   zeros,        info)
    stitch_refine : (refined,   ring_mask,     zeros,                   zeros,        info)
    flow          : (warped,    consistency,   flow_field_rgb,          zeros,        info)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from ._is_changed_util import hash_args_and_kwargs

log = logging.getLogger("MEC.ProPainter")


# ────────────────────────────────────────────────────────────────────────
# Lazy helper getters (no import cost unless the mode is actually used).
# ────────────────────────────────────────────────────────────────────────
def _get_temporal_cls():
    from .propainter_temporal_inpaint import ProPainterTemporalMEC
    return ProPainterTemporalMEC


def _get_remove_cls():
    from .propainter_stitch_suite import ProPainterRemoveMEC
    return ProPainterRemoveMEC


def _get_stitch_cls():
    from .propainter_stitch_suite import ProPainterStitchMEC
    return ProPainterStitchMEC


def _get_stitch_refine_cls():
    from .propainter_stitch_suite import ProPainterStitchRefineMEC
    return ProPainterStitchRefineMEC


def _get_flow_cls():
    from .propainter_flow_refine import FlowRefineMEC
    return FlowRefineMEC


_MODES = ["temporal", "remove", "stitch", "stitch_refine", "flow"]


class ProPainterMEC:
    """Unified ProPainter dispatcher — one node, five modes."""

    CATEGORY = "C2C/Inpaint"
    FUNCTION = "execute"
    DESCRIPTION = (
        "Unified ProPainter node. Absorbs ProPainterTemporal / Remove / "
        "Stitch / StitchRefine / FlowRefine. Pick a mode and only the "
        "relevant widgets are read; others are ignored."
    )

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image_out", "mask_out", "aux_image", "aux_mask", "info")
    OUTPUT_TOOLTIPS = (
        "Primary image output (inpainted / filled / stitched / warped per mode).",
        "Primary mask output (fill mask / repaint mask / consistency per mode).",
        "Auxiliary IMAGE (flow_field_rgb in 'flow' mode, zeros otherwise).",
        "Auxiliary MASK (reserved, zeros for now).",
        "Info string with timings, coverage, mode-specific stats.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (_MODES, {"default": "remove",
                    "tooltip": "Pick the ProPainter operation. Each mode reads a different subset of the optional inputs/widgets below."}),

                # Common knobs (used by all flow-based modes).
                "use_half":          ("BOOLEAN", {"default": True}),
                "color_match_mode":  (["off", "reinhard", "lab", "lab_transfer", "none"],
                                      {"default": "reinhard",
                                       "tooltip": "Per-frame masked colour match between fill and surroundings (ignored in 'flow' mode)."}),

                # Shared ProPainter knobs.
                "raft_iter":         ("INT", {"default": 12, "min": 1, "max": 100}),
                "neighbor_stride":   ("INT", {"default": 5,  "min": 1, "max": 32}),
                "ref_stride":        ("INT", {"default": 10, "min": 1, "max": 64}),
                "subvideo_length":   ("INT", {"default": 8,  "min": 2, "max": 300,
                    "tooltip": "Frames per InpaintGenerator window. 8GB cards: 8-30. 12GB: 40-60. 24GB: 80+."}),
                "raft_chunk":        ("INT", {"default": 16, "min": 1, "max": 64,
                    "tooltip": "Frame-pairs per RAFT forward pass."}),

                # ── 'temporal' specific ──────────────────────────────
                "blend_boundary":    ("BOOLEAN", {"default": True,
                    "tooltip": "[temporal] Blend the inpaint with original at the crop boundary using stitch_data."}),

                # ── 'remove' specific ────────────────────────────────
                "remove_quality":    (["fast", "balanced", "quality"],
                                     {"default": "balanced",
                                      "tooltip": "[remove] Preset that overrides raft_iter/neighbor_stride/ref_stride/subvideo_length."}),
                "remove_dilate_pixels": ("INT", {"default": 3, "min": 0, "max": 32,
                    "tooltip": "[remove] Dilate the mask by N px before filling to cover anti-aliasing."}),

                # ── 'stitch' specific ────────────────────────────────
                "boundary_band_pixels": ("INT", {"default": 12, "min": 0, "max": 96,
                    "tooltip": "[stitch] Width of the boundary band re-painted (0 = no boundary repaint)."}),
                "preserve_inpaint_center": ("BOOLEAN", {"default": True,
                    "tooltip": "[stitch] Keep generative inpaint untouched at the centre, only repaint the seam."}),
                "upscale_method": (["lanczos", "bicubic", "bilinear", "nearest"],
                                   {"default": "lanczos",
                                    "tooltip": "[stitch] How to scale the inpainted crop to the canvas region."}),

                # ── 'stitch_refine' specific ─────────────────────────
                "ring_pixels": ("INT", {"default": 8, "min": 1, "max": 64,
                    "tooltip": "[stitch_refine] Half-width of the seam ring in pixels."}),

                # ── 'flow' specific ──────────────────────────────────
                "flow_consistency_thr": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.05,
                    "tooltip": "[flow] Forward/backward consistency threshold (pixels)."}),
            },
            "optional": {
                # Image / mask / data inputs — wired only as the mode needs.
                "images":           ("IMAGE",
                    {"tooltip": "[temporal/remove] Source frames."}),
                "masks":            ("MASK",
                    {"tooltip": "[temporal/remove] Region to inpaint."}),
                "stitch_data":      ("STITCH_DATA",
                    {"tooltip": "[temporal/stitch/stitch_refine] STITCH_DATA from InpaintCropProMEC."}),
                "inpainted_image":  ("IMAGE",
                    {"tooltip": "[stitch] Crop-sized generative inpaint output."}),
                "stitched_image":   ("IMAGE",
                    {"tooltip": "[stitch_refine] Already-stitched canvas image (output of stitch)."}),
                "mask_override":    ("MASK",
                    {"tooltip": "[stitch_refine] Optional explicit canvas-space mask (overrides stitch_data)."}),
                "frame_a":          ("IMAGE",
                    {"tooltip": "[flow] First frame for optical-flow computation."}),
                "frame_b":          ("IMAGE",
                    {"tooltip": "[flow] Second frame for optical-flow computation."}),
                "flow_mask":        ("MASK",
                    {"tooltip": "[flow] Optional mask restricting the consistency visualisation."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, mode, use_half, color_match_mode, raft_iter, neighbor_stride,
                   ref_stride, subvideo_length, raft_chunk, blend_boundary,
                   remove_quality, remove_dilate_pixels, boundary_band_pixels,
                   preserve_inpaint_center, upscale_method, ring_pixels,
                   flow_consistency_thr,
                   images=None, masks=None, stitch_data=None, inpainted_image=None,
                   stitched_image=None, mask_override=None, frame_a=None, frame_b=None,
                   flow_mask=None, **kwargs):
        return hash_args_and_kwargs(
            mode, use_half, color_match_mode, raft_iter, neighbor_stride,
            ref_stride, subvideo_length, raft_chunk, blend_boundary,
            remove_quality, remove_dilate_pixels, boundary_band_pixels,
            preserve_inpaint_center, upscale_method, ring_pixels,
            flow_consistency_thr,
            images, masks, stitch_data, inpainted_image,
            stitched_image, mask_override, frame_a, frame_b, flow_mask, **kwargs,
        )

    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _normalise_cm(cm: str) -> str:
        """Different sub-nodes accept slightly different colour-match labels."""
        if cm in ("off", "none"):
            return "off"
        if cm == "lab_transfer":
            return "lab"
        return cm

    @staticmethod
    def _zero_image_like(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4:
            return torch.zeros_like(t)
        if t.dim() == 3:  # MASK
            return torch.zeros(t.shape[0], t.shape[1], t.shape[2], 3, dtype=t.dtype)
        return torch.zeros(1, 8, 8, 3, dtype=torch.float32)

    @staticmethod
    def _zero_mask_like(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4:
            return torch.zeros(t.shape[0], t.shape[1], t.shape[2], dtype=t.dtype)
        if t.dim() == 3:
            return torch.zeros_like(t)
        return torch.zeros(1, 8, 8, dtype=torch.float32)

    # ────────────────────────────────────────────────────────────────
    def execute(self, mode: str,
                use_half: bool, color_match_mode: str,
                raft_iter: int, neighbor_stride: int, ref_stride: int,
                subvideo_length: int, raft_chunk: int,
                blend_boundary: bool,
                remove_quality: str, remove_dilate_pixels: int,
                boundary_band_pixels: int, preserve_inpaint_center: bool,
                upscale_method: str,
                ring_pixels: int,
                flow_consistency_thr: float,
                # optional tensors
                images: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None,
                stitch_data: Optional[Dict[str, Any]] = None,
                inpainted_image: Optional[torch.Tensor] = None,
                stitched_image: Optional[torch.Tensor] = None,
                mask_override: Optional[torch.Tensor] = None,
                frame_a: Optional[torch.Tensor] = None,
                frame_b: Optional[torch.Tensor] = None,
                flow_mask: Optional[torch.Tensor] = None):

        cm = self._normalise_cm(color_match_mode)

        if mode == "temporal":
            if images is None or masks is None:
                raise ValueError("[ProPainterMEC] mode='temporal' requires 'images' and 'masks'.")
            node = _get_temporal_cls()()
            cm_t = "reinhard" if cm == "off" else (
                "lab_transfer" if cm == "lab" else cm)
            if cm_t not in ("none", "reinhard", "lab_transfer"):
                cm_t = "reinhard"
            out_img, info = node.inpaint_temporal(
                images=images, masks=masks,
                stitch_data=(stitch_data or {}),
                neighbor_stride=int(neighbor_stride),
                ref_stride=int(ref_stride),
                raft_iter=int(raft_iter),
                subvideo_length=int(subvideo_length),
                use_half=bool(use_half),
                blend_boundary=bool(blend_boundary),
                color_match_mode=cm_t,
                raft_chunk=int(raft_chunk),
            )
            return (out_img, self._zero_mask_like(out_img),
                    self._zero_image_like(out_img), self._zero_mask_like(out_img),
                    f"mode=temporal | {info}")

        if mode == "remove":
            if images is None or masks is None:
                raise ValueError("[ProPainterMEC] mode='remove' requires 'images' and 'masks'.")
            node = _get_remove_cls()()
            out_img, used_mask, info = node.remove(
                images=images, masks=masks,
                quality=remove_quality,
                use_half=bool(use_half),
                dilate_mask_pixels=int(remove_dilate_pixels),
                color_match_mode=cm,
            )
            return (out_img, used_mask,
                    self._zero_image_like(out_img), self._zero_mask_like(out_img),
                    f"mode=remove | {info}")

        if mode == "stitch":
            if stitch_data is None or inpainted_image is None:
                raise ValueError("[ProPainterMEC] mode='stitch' requires 'stitch_data' and 'inpainted_image'.")
            node = _get_stitch_cls()()
            out_img, used_mask, info = node.stitch(
                stitch_data=stitch_data,
                inpainted_image=inpainted_image,
                boundary_band_pixels=int(boundary_band_pixels),
                preserve_inpaint_center=bool(preserve_inpaint_center),
                raft_iter=int(raft_iter),
                neighbor_stride=int(neighbor_stride),
                ref_stride=int(ref_stride),
                subvideo_length=int(subvideo_length),
                use_half=bool(use_half),
                color_match_mode=cm,
                upscale_method=upscale_method,
            )
            return (out_img, used_mask,
                    self._zero_image_like(out_img), self._zero_mask_like(out_img),
                    f"mode=stitch | {info}")

        if mode == "stitch_refine":
            if stitched_image is None:
                raise ValueError("[ProPainterMEC] mode='stitch_refine' requires 'stitched_image'.")
            node = _get_stitch_refine_cls()()
            out_img, used_mask, info = node.refine(
                stitched_image=stitched_image,
                stitch_data=(stitch_data or {}),
                ring_pixels=int(ring_pixels),
                raft_iter=int(raft_iter),
                neighbor_stride=int(neighbor_stride),
                ref_stride=int(ref_stride),
                subvideo_length=int(subvideo_length),
                use_half=bool(use_half),
                color_match_mode=cm,
                mask_override=mask_override,
            )
            return (out_img, used_mask,
                    self._zero_image_like(out_img), self._zero_mask_like(out_img),
                    f"mode=stitch_refine | {info}")

        if mode == "flow":
            if frame_a is None or frame_b is None:
                raise ValueError("[ProPainterMEC] mode='flow' requires 'frame_a' and 'frame_b'.")
            node = _get_flow_cls()()
            flow_rgb, warped, consistency = node.refine_flow(
                frame_a=frame_a, frame_b=frame_b,
                iters=int(raft_iter),
                consistency_thr=float(flow_consistency_thr),
                mask=flow_mask,
            )
            return (warped, consistency,
                    flow_rgb, self._zero_mask_like(consistency),
                    f"mode=flow | iters={int(raft_iter)} consistency_thr={float(flow_consistency_thr):.3f}")

        raise ValueError(f"[ProPainterMEC] unknown mode {mode!r}; expected one of {_MODES}.")


NODE_CLASS_MAPPINGS = {"ProPainterMEC": ProPainterMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProPainterMEC": "ProPainter — Temporal / Remove / Stitch / Refine / Flow",
}
