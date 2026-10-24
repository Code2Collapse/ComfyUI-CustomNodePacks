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

    temporal      : (inpainted, inpaint_mask,  difference x4,           changed,      info)
    remove        : (filled,    fill_mask,     difference x4,           changed,      info)
    stitch        : (stitched,  repaint_mask,  difference x4,           changed,      info)
    stitch_refine : (refined,   ring_mask,     difference x4,           changed,      info)
    flow          : (warped,    consistency,   flow_field_rgb,          unreliable,   info)

L9.39: aux_image / aux_mask were zeros (aux_mask "reserved") and temporal's mask_out was zeros. Now they are QC:
"difference x4" = |output - input| amplified 4x, "changed" = pixels the node altered (more than 1.5/255 in any
channel), measured against the mode's input (images / the stitch canvas / stitched_image); "unreliable" = 1 -
consistency. The info line says how much changed and how much of that lies OUTSIDE mask_out (spill / seams). When
the output and its input differ in size, both stay zeros and info says why.
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
        "QC image: |output - input| x4 (flow mode: the flow field as colour).",
        "QC mask: pixels this node changed (flow mode: where the flow is unreliable, 1 - consistency).",
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

    @staticmethod
    def _qc(out: torch.Tensor, ref: Optional[torch.Tensor], region: torch.Tensor):
        """(difference x4 image, changed mask, info text) of `out` against `ref`; zeros when sizes differ."""
        if not (isinstance(ref, torch.Tensor) and ref.dim() == 4 and out.dim() == 4
                and tuple(ref.shape[1:3]) == tuple(out.shape[1:3])):
            why = "no input to compare" if ref is None else \
                f"input {tuple(ref.shape[1:3])} vs output {tuple(out.shape[1:3])}"
            return ProPainterMEC._zero_image_like(out), ProPainterMEC._zero_mask_like(out), f"qc: off ({why})"
        r = ref[..., :3].to(out.device, out.dtype)
        if r.shape[0] != out.shape[0]:
            r = r[torch.arange(out.shape[0]) % r.shape[0]]
        diff = (out[..., :3] - r).abs()
        changed = (diff.amax(-1) > 1.5 / 255.0).to(out.dtype)
        txt = f"qc: changed {100.0 * float(changed.mean()):.2f}% of pixels"
        if isinstance(region, torch.Tensor) and region.dim() == 3 and tuple(region.shape) == tuple(changed.shape):
            outside = changed * (region.to(changed.device, changed.dtype) <= 0.5)
            txt += f", {100.0 * float(outside.mean()):.2f}% outside mask_out"
        return (diff * 4.0).clamp(0, 1).contiguous(), changed.contiguous(), txt

    @staticmethod
    def _mask_like(masks: Optional[torch.Tensor], out: torch.Tensor) -> torch.Tensor:
        """The input mask at the output's size and frame count (zeros when there is none)."""
        if not isinstance(masks, torch.Tensor):
            return ProPainterMEC._zero_mask_like(out)
        m = masks if masks.dim() == 3 else masks.reshape(-1, *masks.shape[-2:])
        m = m.to(out.device, torch.float32)
        if tuple(m.shape[1:]) != tuple(out.shape[1:3]):
            m = torch.nn.functional.interpolate(m.unsqueeze(1), size=tuple(out.shape[1:3]), mode="bilinear",
                                                align_corners=False).squeeze(1)
        if m.shape[0] != out.shape[0]:
            m = m[torch.arange(out.shape[0]) % m.shape[0]]
        return m.clamp(0, 1).contiguous()

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
            region = self._mask_like(masks, out_img)
            diff, changed, qc = self._qc(out_img, images, region)
            return (out_img, region, diff, changed, f"mode=temporal | {info} | {qc}")

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
            diff, changed, qc = self._qc(out_img, images, used_mask)
            return (out_img, used_mask, diff, changed, f"mode=remove | {info} | {qc}")

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
            canvas = stitch_data.get("canvas_image") if isinstance(stitch_data, dict) else None
            diff, changed, qc = self._qc(out_img, canvas, used_mask)
            return (out_img, used_mask, diff, changed, f"mode=stitch | {info} | {qc}")

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
            diff, changed, qc = self._qc(out_img, stitched_image, used_mask)
            return (out_img, used_mask, diff, changed, f"mode=stitch_refine | {info} | {qc}")

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
            unreliable = (1.0 - consistency.float()).clamp(0, 1) if isinstance(consistency, torch.Tensor) \
                else self._zero_mask_like(warped)
            return (warped, consistency,
                    flow_rgb, unreliable,
                    f"mode=flow | iters={int(raft_iter)} consistency_thr={float(flow_consistency_thr):.3f}")

        raise ValueError(f"[ProPainterMEC] unknown mode {mode!r}; expected one of {_MODES}.")


NODE_CLASS_MAPPINGS = {"ProPainterMEC": ProPainterMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProPainterMEC": "ProPainter — Temporal / Remove / Stitch / Refine / Flow",
}
