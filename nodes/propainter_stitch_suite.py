# FILE: nodes/propainter_stitch_suite.py
# FEATURE: ProPainter-powered alternatives to the InpaintCrop / Stitch / Inpaint pipeline.
# INTEGRATES WITH:
#   - nodes/inpaint_suite.py  (consumes/produces STITCH_DATA dicts)
#   - nodes/propainter_temporal_inpaint.py  (re-uses RAFT + RFC + InpaintGenerator pipeline)
"""
Three ProPainter integration nodes:

  A. ProPainterStitchRefineMEC
     Run ProPainter on a thin RING along the seam of an already-stitched
     image. Cheapest fix for boundary flicker / colour break when the
     inpaint content itself is fine but the merge with the surroundings
     looks bad. Works on the OUTPUT of InpaintStitchProMEC.

  B. ProPainterStitchMEC
     Drop-in replacement for InpaintStitchProMEC. Takes the same
     STITCH_DATA contract but uses ProPainter's flow-aware composite over
     the entire crop region instead of Laplacian / frequency / edge-aware
     pyramids. Handles both v2 (single-position) and v3 (per-frame
     offsets) stitch data.

  C. ProPainterRemoveMEC
     Inpaint-free object removal. Takes plain (IMAGE, MASK) and runs the
     full ProPainter pipeline. No SD / Flux model required. Use when the
     un-masked context is enough to fill the hole (object removal, plate
     cleanup, wire removal). Wraps ProPainterTemporalMEC with friendlier
     defaults and a quality preset.

All three honour ComfyUI's interrupt and free VRAM in `finally`.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .propainter_bridge import (
    HAS_PROPAINTER,
    ProPainterMissingError,
    free_models,
    from_propainter_video,
    load_models,
    require_propainter,
    to_propainter_mask,
    to_propainter_video,
)
from .propainter_temporal_inpaint import (
    ProPainterTemporalMEC,
    _color_match,
    _complete_flow,
    _compute_bidirectional_flow,
    _inpaint_window,
)

log = logging.getLogger("MEC.propainter_stitch")


# =====================================================================
# helpers
# =====================================================================
def _check_interrupt() -> None:
    try:
        import comfy.model_management as mm  # type: ignore
        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _release_vram() -> None:
    try:
        import comfy.model_management as mm  # type: ignore
        mm.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_bhw_mask(m: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
    """Coerce any (B,H,W) / (1,H,W) / (B,H,W,1) / (H,W) mask to (B,H,W) float [0,1]."""
    x = m.float()
    if x.dim() == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    # Resize spatially if needed.
    if x.shape[-2:] != (H, W):
        x = F.interpolate(x.unsqueeze(1), size=(H, W),
                          mode="bilinear", align_corners=False).squeeze(1)
    if x.shape[0] == 1 and B > 1:
        x = x.expand(B, -1, -1).contiguous()
    elif x.shape[0] != B:
        # Best-effort broadcast.
        x = x[:1].expand(B, -1, -1).contiguous()
    return x.clamp(0.0, 1.0)


def _binary_dilate(mask_bhw: torch.Tensor, radius: int) -> torch.Tensor:
    """Square-kernel binary dilation via max-pool."""
    if radius <= 0:
        return mask_bhw
    k = radius * 2 + 1
    x = (mask_bhw > 0.5).float().unsqueeze(1)
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    return x.squeeze(1)


def _binary_erode(mask_bhw: torch.Tensor, radius: int) -> torch.Tensor:
    """Erosion via inverted dilation."""
    if radius <= 0:
        return mask_bhw
    inv = 1.0 - (mask_bhw > 0.5).float()
    inv_d = _binary_dilate(inv, radius)
    return 1.0 - inv_d


def _ring_mask(base_mask_bhw: torch.Tensor, ring_pixels: int) -> torch.Tensor:
    """Build a ring-shaped boundary mask: dilate(R) AND NOT erode(R)."""
    r = max(1, int(ring_pixels))
    dil = _binary_dilate(base_mask_bhw, r)
    ero = _binary_erode(base_mask_bhw, r)
    ring = (dil > 0.5).float() - (ero > 0.5).float()
    return ring.clamp(0.0, 1.0)


def _stitch_data_get_canvas_mask(stitch_data: Dict[str, Any],
                                 H_canvas: int, W_canvas: int,
                                 B: int) -> Optional[torch.Tensor]:
    """Recover the inpaint mask in canvas/full-image coordinates.

    Returns a (B,H,W) float mask in [0,1] or None if not recoverable.
    Handles v1 (original_mask in original-image space), v2 (single offset),
    and v3 (per-frame offsets) STITCH_DATA contracts.
    """
    version = int(stitch_data.get("version", 1))
    om = stitch_data.get("original_mask")
    if om is None:
        return None
    om = om.float()
    if om.dim() == 4 and om.shape[-1] == 1:
        om = om.squeeze(-1)
    if om.dim() == 2:
        om = om.unsqueeze(0)

    if version >= 2:
        ctc_x = int(stitch_data["ctc_x"])
        ctc_y = int(stitch_data["ctc_y"])
        ctc_w = int(stitch_data["ctc_w"])
        ctc_h = int(stitch_data["ctc_h"])
        cto_x = int(stitch_data["cto_x"])
        cto_y = int(stitch_data["cto_y"])
        cto_w = int(stitch_data["cto_w"])
        cto_h = int(stitch_data["cto_h"])
        # original_mask is in original-image (pre-canvas) space sized cto_w x cto_h.
        # First place it into the canvas at (cto_x, cto_y).
        if om.shape[-2:] != (cto_h, cto_w):
            om = F.interpolate(om.unsqueeze(1), size=(cto_h, cto_w),
                               mode="bilinear", align_corners=False).squeeze(1)
        canvas_mask = torch.zeros(om.shape[0], H_canvas, W_canvas,
                                  dtype=om.dtype, device=om.device)
        y2 = min(cto_y + cto_h, H_canvas)
        x2 = min(cto_x + cto_w, W_canvas)
        canvas_mask[:, cto_y:y2, cto_x:x2] = om[:, :y2 - cto_y, :x2 - cto_x]
    else:
        # v1: stitch_blend_mask_crop sits at (crop_x, crop_y) of original_image.
        crop_x = int(stitch_data.get("crop_x", 0))
        crop_y = int(stitch_data.get("crop_y", 0))
        crop_w = int(stitch_data.get("crop_w", om.shape[-1]))
        crop_h = int(stitch_data.get("crop_h", om.shape[-2]))
        if om.shape[-2:] != (crop_h, crop_w):
            om = F.interpolate(om.unsqueeze(1), size=(crop_h, crop_w),
                               mode="bilinear", align_corners=False).squeeze(1)
        canvas_mask = torch.zeros(om.shape[0], H_canvas, W_canvas,
                                  dtype=om.dtype, device=om.device)
        y2 = min(crop_y + crop_h, H_canvas)
        x2 = min(crop_x + crop_w, W_canvas)
        canvas_mask[:, crop_y:y2, crop_x:x2] = om[:, :y2 - crop_y, :x2 - crop_x]

    # Broadcast to B if the mask was a single frame.
    if canvas_mask.shape[0] == 1 and B > 1:
        canvas_mask = canvas_mask.expand(B, -1, -1).contiguous()
    return canvas_mask.clamp(0.0, 1.0)


# =====================================================================
# A. ProPainterStitchRefineMEC — boundary ring refine
# =====================================================================
class ProPainterStitchRefineMEC:
    """Refine the seam of an already-stitched inpaint with ProPainter.

    Builds a thin ring along the boundary of the original inpaint mask and
    runs ProPainter on that ring only. Inpaint content (centre) and the
    untouched surroundings are preserved bit-for-bit. Cheapest of the
    three ProPainter-stitch nodes — only the seam pixels change.

    Pipeline:
      1. Recover original mask in full-canvas / output coordinates from
         STITCH_DATA. (Falls back to a user-supplied MASK if missing.)
      2. ring = dilate(mask, R) - erode(mask, R)
      3. Run ProPainterTemporalMEC.inpaint_temporal on (stitched_image, ring)
      4. Composite: keep stitched_image outside ring, ProPainter fill inside.
    """

    VRAM_TIER = 3
    COLOR = "#3a6fb5"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitched_image": ("IMAGE",
                    {"tooltip": "Output of InpaintStitchProMEC (or any stitched inpaint result)."}),
                "stitch_data":    ("STITCH_DATA",
                    {"tooltip": "STITCH_DATA from InpaintCropProMEC — used to recover the mask in output coords."}),
                "ring_pixels":    ("INT",
                    {"default": 8, "min": 1, "max": 64, "step": 1,
                     "tooltip": "Half-width of the seam ring in pixels. Larger = blends a wider band."}),
                "raft_iter":      ("INT",
                    {"default": 12, "min": 4, "max": 32}),
                "neighbor_stride":("INT",
                    {"default": 5, "min": 1, "max": 32}),
                "ref_stride":     ("INT",
                    {"default": 10, "min": 2, "max": 64}),
                "subvideo_length":("INT",
                    {"default": 8, "min": 2, "max": 80}),
                "use_half":       ("BOOLEAN", {"default": True}),
                "color_match_mode":(["off", "reinhard", "lab"],
                    {"default": "off",
                     "tooltip": "Per-frame masked colour match between ring fill and surrounding pixels."}),
            },
            "optional": {
                "mask_override": ("MASK",
                    {"tooltip": "Optional explicit canvas-space mask. If provided, used instead of recovering from stitch_data."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("refined_image", "ring_mask_used", "info")
    OUTPUT_TOOLTIPS = (
        "Stitched image with the boundary ring refined by ProPainter.",
        "The actual ring mask that was inpainted (debugging / re-use).",
        "Info string: ring stats, ProPainter timings, frame count.",
    )
    FUNCTION = "refine"
    CATEGORY = "MaskEditControl/Inpaint"
    DESCRIPTION = "Run ProPainter on the seam ring of an already-stitched image to fix boundary flicker / hard edges."

    def refine(self, stitched_image: torch.Tensor, stitch_data: Dict[str, Any],
               ring_pixels: int, raft_iter: int, neighbor_stride: int,
               ref_stride: int, subvideo_length: int, use_half: bool,
               color_match_mode: str, mask_override: Optional[torch.Tensor] = None):
        if not HAS_PROPAINTER:
            require_propainter()
        t0 = time.time()
        info: List[str] = []

        if stitched_image.dim() != 4 or stitched_image.shape[-1] != 3:
            raise ValueError(
                f"ProPainterStitchRefineMEC: expected IMAGE shape (B,H,W,3) got {tuple(stitched_image.shape)}")
        B, H, W, _ = stitched_image.shape

        # ----- recover canvas-space mask -----
        if mask_override is not None:
            base = _ensure_bhw_mask(mask_override, B, H, W)
            info.append("mask_source=override")
        elif isinstance(stitch_data, dict) and stitch_data:
            recovered = _stitch_data_get_canvas_mask(stitch_data, H, W, B)
            if recovered is None:
                raise ValueError(
                    "ProPainterStitchRefineMEC: stitch_data has no 'original_mask'. "
                    "Either reconnect a STITCH_DATA from InpaintCropProMEC, or wire a MASK into mask_override.")
            base = recovered
            info.append(f"mask_source=stitch_data v{stitch_data.get('version', 1)}")
        else:
            raise ValueError(
                "ProPainterStitchRefineMEC: need either valid stitch_data or mask_override.")

        # ----- build ring -----
        ring = _ring_mask(base, ring_pixels)
        coverage = float(ring.mean())
        info.append(f"ring_pixels={ring_pixels} ring_coverage={coverage:.4f}")
        if coverage < 1e-5:
            log.warning("[propainter_stitch_refine] ring is empty — returning input unchanged.")
            return (stitched_image, ring, " | ".join(info + ["NOOP_empty_ring"]))

        # ----- delegate to the verified ProPainter temporal node -----
        try:
            base_node = ProPainterTemporalMEC()
            out_img, sub_info = base_node.inpaint_temporal(
                images=stitched_image,
                masks=ring,
                stitch_data={},                           # boundary blend off
                neighbor_stride=neighbor_stride,
                ref_stride=ref_stride,
                raft_iter=raft_iter,
                subvideo_length=subvideo_length,
                use_half=use_half,
                blend_boundary=False,
                color_match_mode=color_match_mode if color_match_mode != "off" else "reinhard",
            )
            # When color_match is off, snap unmasked region back exactly.
            if color_match_mode == "off":
                m4 = ring.unsqueeze(-1)
                out_img = stitched_image * (1.0 - m4) + out_img * m4
            info.append(sub_info.replace("\n", " | "))
        finally:
            # Don't free here — the bundle is reusable across nodes within the same Queue.
            _check_interrupt()

        info.append(f"total={time.time() - t0:.2f}s")
        return (out_img, ring, " | ".join(info))


# =====================================================================
# B. ProPainterStitchMEC — drop-in stitch using ProPainter composite
# =====================================================================
class ProPainterStitchMEC:
    """Drop-in replacement for InpaintStitchProMEC using ProPainter.

    Honours the same STITCH_DATA contract (v2 / v3) but instead of
    Laplacian-pyramid / frequency-domain blending, it:

      1. Pastes the inpainted crop into the canvas at the recorded offset.
      2. Builds a feathered band around the crop boundary plus the actual
         original mask.
      3. Runs ProPainter on that combined region so the inpaint content
         AND its boundary are flow-consistent across frames.
      4. Optionally re-snaps the inpaint center to exactly the model output
         (preserving generative content) — only the seam is re-painted.

    Outputs match InpaintStitchProMEC: (IMAGE, MASK, STRING).
    """

    VRAM_TIER = 4
    COLOR = "#2a5d9f"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitch_data":    ("STITCH_DATA",),
                "inpainted_image":("IMAGE",
                    {"tooltip": "Generative inpaint output — sized to the crop, not the full canvas."}),
                "boundary_band_pixels":("INT",
                    {"default": 12, "min": 0, "max": 96, "step": 1,
                     "tooltip": "Width of the boundary band repainted by ProPainter (0 = no boundary repaint)."}),
                "preserve_inpaint_center":("BOOLEAN",
                    {"default": True,
                     "tooltip": "Keep the inpaint generative content untouched at the centre and only repaint the seam."}),
                "raft_iter":      ("INT", {"default": 12, "min": 4, "max": 32}),
                "neighbor_stride":("INT", {"default": 5, "min": 1, "max": 32}),
                "ref_stride":     ("INT", {"default": 10, "min": 2, "max": 64}),
                "subvideo_length":("INT", {"default": 8, "min": 2, "max": 80}),
                "use_half":       ("BOOLEAN", {"default": True}),
                "color_match_mode":(["off", "reinhard", "lab"], {"default": "reinhard"}),
                "upscale_method": (["lanczos", "bicubic", "bilinear", "nearest"],
                    {"default": "lanczos"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "blend_mask_used", "info")
    OUTPUT_TOOLTIPS = (
        "Final canvas-resolution image with ProPainter-blended inpaint.",
        "Combined boundary + original mask actually repainted.",
        "Info string with timings, coverage, mode.",
    )
    FUNCTION = "stitch"
    CATEGORY = "MaskEditControl/Inpaint"
    DESCRIPTION = "Stitch inpainted crop back using ProPainter for flow-consistent seams. Drop-in replacement for InpaintStitchProMEC."

    def _resize_to(self, t: torch.Tensor, h: int, w: int, method: str) -> torch.Tensor:
        # IMAGE (B,H,W,C) -> resized.
        x = t.permute(0, 3, 1, 2)  # (B,C,H,W)
        mode_map = {
            "nearest": ("nearest", None),
            "bilinear": ("bilinear", False),
            "bicubic": ("bicubic", False),
            "lanczos": ("bicubic", False),  # F.interpolate has no lanczos; bicubic is close.
        }
        mode, ac = mode_map.get(method, ("bicubic", False))
        kwargs: Dict[str, Any] = {"size": (h, w), "mode": mode}
        if ac is not None:
            kwargs["align_corners"] = ac
        x = F.interpolate(x, **kwargs)
        return x.permute(0, 2, 3, 1).contiguous()

    def stitch(self, stitch_data: Dict[str, Any], inpainted_image: torch.Tensor,
               boundary_band_pixels: int, preserve_inpaint_center: bool,
               raft_iter: int, neighbor_stride: int, ref_stride: int,
               subvideo_length: int, use_half: bool, color_match_mode: str,
               upscale_method: str):
        if not HAS_PROPAINTER:
            require_propainter()
        t0 = time.time()
        info: List[str] = []

        if not isinstance(stitch_data, dict) or not stitch_data:
            raise ValueError("ProPainterStitchMEC: stitch_data missing or empty.")
        version = int(stitch_data.get("version", 1))
        if version < 2:
            raise ValueError(
                "ProPainterStitchMEC: only v2 / v3 STITCH_DATA is supported. "
                "Use the latest InpaintCropProMEC.")

        canvas = stitch_data["canvas_image"].clone()
        if canvas.dim() != 4 or canvas.shape[-1] != 3:
            raise ValueError(f"canvas_image bad shape {tuple(canvas.shape)}")
        B, H_canvas, W_canvas, _ = canvas.shape
        ctc_x = int(stitch_data["ctc_x"])
        ctc_y = int(stitch_data["ctc_y"])
        ctc_w = int(stitch_data["ctc_w"])
        ctc_h = int(stitch_data["ctc_h"])
        frame_offsets = stitch_data.get("frame_offsets")  # v3 list[(x,y)] or None
        if frame_offsets is not None and len(frame_offsets) != B:
            raise ValueError(
                f"frame_offsets length ({len(frame_offsets)}) != batch ({B}).")

        # ----- 1. Resize inpainted to crop region; paste into canvas -----
        if inpainted_image.shape[0] != B:
            if inpainted_image.shape[0] == 1:
                inpainted_image = inpainted_image.expand(B, -1, -1, -1).contiguous()
            else:
                raise ValueError(
                    f"inpainted_image batch ({inpainted_image.shape[0]}) "
                    f"does not match canvas batch ({B}).")
        inp_resized = self._resize_to(inpainted_image, ctc_h, ctc_w, upscale_method)

        offsets = ([(int(fx), int(fy)) for (fx, fy) in frame_offsets]
                   if frame_offsets is not None
                   else [(ctc_x, ctc_y)] * B)

        pasted = canvas.clone()
        for b, (fx, fy) in enumerate(offsets):
            y2 = min(fy + ctc_h, H_canvas)
            x2 = min(fx + ctc_w, W_canvas)
            pasted[b, fy:y2, fx:x2, :] = inp_resized[b, :y2 - fy, :x2 - fx, :]

        # ----- 2. Build mask = (boundary band) ∪ (original mask) -----
        repaint_mask = torch.zeros(B, H_canvas, W_canvas, dtype=torch.float32)
        # Boundary band: 1 inside a band of width `boundary_band_pixels` along
        # the crop rectangle perimeter.
        if boundary_band_pixels > 0:
            for b, (fx, fy) in enumerate(offsets):
                y0 = max(0, fy)
                y1 = min(H_canvas, fy + ctc_h)
                x0 = max(0, fx)
                x1 = min(W_canvas, fx + ctc_w)
                # Outer band (outside crop).
                oy0 = max(0, y0 - boundary_band_pixels)
                oy1 = min(H_canvas, y1 + boundary_band_pixels)
                ox0 = max(0, x0 - boundary_band_pixels)
                ox1 = min(W_canvas, x1 + boundary_band_pixels)
                repaint_mask[b, oy0:oy1, ox0:ox1] = 1.0
                # Inner band (inside crop, near edge).
                iy0 = min(y1, y0 + boundary_band_pixels)
                iy1 = max(y0, y1 - boundary_band_pixels)
                ix0 = min(x1, x0 + boundary_band_pixels)
                ix1 = max(x0, x1 - boundary_band_pixels)
                # Carve out the centre keep-region.
                if iy0 < iy1 and ix0 < ix1:
                    repaint_mask[b, iy0:iy1, ix0:ix1] = 0.0

        # OR with the original generative mask if the user wants the inpaint
        # boundary blended ALSO inside the mask itself.
        canvas_orig_mask = _stitch_data_get_canvas_mask(
            stitch_data, H_canvas, W_canvas, B,
        )
        if canvas_orig_mask is not None and not preserve_inpaint_center:
            repaint_mask = (repaint_mask + canvas_orig_mask).clamp(0, 1)
        elif canvas_orig_mask is not None and preserve_inpaint_center:
            # Only blend a thin ring around the original mask, NOT its centre.
            ring = _ring_mask(canvas_orig_mask, max(1, boundary_band_pixels // 2))
            repaint_mask = (repaint_mask + ring).clamp(0, 1)

        coverage = float(repaint_mask.mean())
        info.append(f"version={version} coverage={coverage:.4f} band={boundary_band_pixels} preserve={preserve_inpaint_center}")
        if coverage < 1e-5:
            info.append("NOOP_empty_repaint_mask")
            return (pasted, repaint_mask, " | ".join(info))

        # ----- 3. ProPainter on the repaint mask -----
        try:
            base_node = ProPainterTemporalMEC()
            out_img, sub_info = base_node.inpaint_temporal(
                images=pasted,
                masks=repaint_mask,
                stitch_data={},
                neighbor_stride=neighbor_stride,
                ref_stride=ref_stride,
                raft_iter=raft_iter,
                subvideo_length=subvideo_length,
                use_half=use_half,
                blend_boundary=False,
                color_match_mode=color_match_mode if color_match_mode != "off" else "reinhard",
            )
            if color_match_mode == "off":
                m4 = repaint_mask.unsqueeze(-1)
                out_img = pasted * (1.0 - m4) + out_img * m4
            info.append(sub_info.replace("\n", " | "))
        finally:
            _check_interrupt()

        info.append(f"total={time.time() - t0:.2f}s")
        return (out_img, repaint_mask, " | ".join(info))


# =====================================================================
# C. ProPainterRemoveMEC — model-free object / wire / plate removal
# =====================================================================
class ProPainterRemoveMEC:
    """Inpaint-free removal: just (IMAGE, MASK) -> filled IMAGE.

    No SD / Flux model required. Works best when the OFF-mask context
    contains the truth — i.e. the masked object moves and the background
    is visible at some other frame. Use cases:

      - Object / character removal from a static plate
      - Wire / rig removal in VFX
      - Watermark / logo cleanup across a video
      - Cleaning up tracker patches before stabilising

    For generative re-paint (creating content that never existed), use
    InpaintCropProMEC + a diffusion model + InpaintStitchProMEC instead.
    """

    VRAM_TIER = 3
    COLOR = "#1f7a44"

    QUALITY_PRESETS = {
        "fast":     dict(raft_iter=8,  neighbor_stride=10, ref_stride=20, subvideo_length=8),
        "balanced": dict(raft_iter=12, neighbor_stride=5,  ref_stride=10, subvideo_length=8),
        "quality":  dict(raft_iter=20, neighbor_stride=3,  ref_stride=6,  subvideo_length=12),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":  ("IMAGE",),
                "masks":   ("MASK",
                    {"tooltip": "Region to fill. 1 = fill, 0 = keep. Will be auto-broadcast across frames."}),
                "quality": (list(cls.QUALITY_PRESETS.keys()), {"default": "balanced"}),
                "use_half":("BOOLEAN", {"default": True}),
                "dilate_mask_pixels":("INT",
                    {"default": 3, "min": 0, "max": 32, "step": 1,
                     "tooltip": "Dilate mask by N px before fill — helps cover anti-aliasing artefacts at the edge."}),
                "color_match_mode":(["off", "reinhard", "lab"], {"default": "reinhard"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("filled_image", "fill_mask_used", "info")
    OUTPUT_TOOLTIPS = (
        "Image with the masked region filled by ProPainter.",
        "Final fill mask (after dilation).",
        "Quality preset used + ProPainter timings.",
    )
    FUNCTION = "remove"
    CATEGORY = "MaskEditControl/Inpaint"
    DESCRIPTION = "ProPainter-only object / wire / plate removal. No SD model required."

    def remove(self, images: torch.Tensor, masks: torch.Tensor,
               quality: str, use_half: bool, dilate_mask_pixels: int,
               color_match_mode: str):
        if not HAS_PROPAINTER:
            require_propainter()
        t0 = time.time()
        info: List[str] = [f"quality={quality}"]

        if images.dim() != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"ProPainterRemoveMEC: IMAGE shape (B,H,W,3) expected, got {tuple(images.shape)}")
        B, H, W, _ = images.shape
        m = _ensure_bhw_mask(masks, B, H, W)
        if dilate_mask_pixels > 0:
            m = _binary_dilate(m, dilate_mask_pixels)
            info.append(f"dilate={dilate_mask_pixels}")
        coverage = float(m.mean())
        info.append(f"coverage={coverage:.4f}")
        if coverage < 1e-5:
            info.append("NOOP_empty_mask")
            return (images, m, " | ".join(info))

        preset = self.QUALITY_PRESETS[quality]
        try:
            base_node = ProPainterTemporalMEC()
            out_img, sub_info = base_node.inpaint_temporal(
                images=images,
                masks=m,
                stitch_data={},
                neighbor_stride=preset["neighbor_stride"],
                ref_stride=preset["ref_stride"],
                raft_iter=preset["raft_iter"],
                subvideo_length=preset["subvideo_length"],
                use_half=use_half,
                blend_boundary=False,
                color_match_mode=color_match_mode if color_match_mode != "off" else "reinhard",
            )
            if color_match_mode == "off":
                m4 = m.unsqueeze(-1)
                out_img = images * (1.0 - m4) + out_img * m4
            info.append(sub_info.replace("\n", " | "))
        finally:
            _check_interrupt()

        info.append(f"total={time.time() - t0:.2f}s")
        return (out_img, m, " | ".join(info))


# =====================================================================
NODE_CLASS_MAPPINGS = {
    "ProPainterStitchRefineMEC": ProPainterStitchRefineMEC,
    "ProPainterStitchMEC":       ProPainterStitchMEC,
    "ProPainterRemoveMEC":       ProPainterRemoveMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProPainterStitchRefineMEC": "ProPainter Stitch Refine — seam ring (MEC)",
    "ProPainterStitchMEC":       "ProPainter Stitch — flow-aware composite (MEC)",
    "ProPainterRemoveMEC":       "ProPainter Remove — object / wire / plate (MEC)",
}
