"""
MaskEditMEC – Unified mask transform / fixup / draw / edit dispatcher.

Combines (via composition) the previously-separate nodes:
    transform     → MaskTransformXY
    draw_shape    → DrawShapeMEC
    draw_advanced → MaskDrawFrame (raw shape_params_json power-mode)
    points_bbox   → PointsMaskEditor (interactive canvas)
    bbox_smooth   → BBoxSmooth

Single ``mode`` widget selects which engine runs; unused outputs are
filled with safe defaults so the unified RETURN_TYPES is consistent
across every mode. Pure-CPU, no models, no GPU dependency.

Output schema (8 ports) — same across all modes:
    0 MASK      mask
    1 STRING    positive_coords  (points-mode SAM coords)
    2 STRING    negative_coords  (points-mode SAM neg coords)
    3 BBOX      bboxes           (positive bboxes list)
    4 BBOX      neg_bboxes       (negative bboxes list)
    5 STRING    points_json      (full points JSON)
    6 STRING    bbox_json        (full bbox JSON / bbox_smooth result)
    7 BBOX      primary_bbox     ([x,y,w,h])
"""

from __future__ import annotations

import json

import torch

from .mask_transform_xy import MaskTransformXY
from .mask_draw_frame import MaskDrawFrame, DrawShapeMEC
from .points_mask_editor import PointsMaskEditor
from .bbox_nodes import BBoxSmooth


def _empty_mask(ref: torch.Tensor | None = None) -> torch.Tensor:
    if ref is not None and isinstance(ref, torch.Tensor):
        if ref.dim() == 4:  # IMAGE B,H,W,C
            return torch.zeros(ref.shape[0], ref.shape[1], ref.shape[2],
                               dtype=torch.float32)
        if ref.dim() == 3:  # MASK B,H,W
            return torch.zeros_like(ref, dtype=torch.float32)
        if ref.dim() == 2:
            return torch.zeros(1, ref.shape[0], ref.shape[1], dtype=torch.float32)
    return torch.zeros(1, 64, 64, dtype=torch.float32)


_EMPTY_BBOX = [0, 0, 0, 0]


class MaskEditMEC:
    """Unified mask edit / transform / draw / points-bbox / bbox-smooth node.

    Pick a ``mode`` and the relevant widgets drive that engine. Unused
    parameters are simply ignored. All processing is CPU-only.
    """

    MODES = ["transform", "draw_shape", "draw_advanced",
             "points_bbox", "bbox_smooth"]

    SHAPES = DrawShapeMEC.SHAPES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (cls.MODES, {
                    "default": "transform",
                    "tooltip": (
                        "transform: morph / blur / offset / feather / threshold (needs mask).\n"
                        "draw_shape: pick a shape from a dropdown, set its params (12 shapes).\n"
                        "draw_advanced: power-mode shape_params_json (raw JSON).\n"
                        "points_bbox: interactive points + bbox canvas (SAM/SeC coords).\n"
                        "bbox_smooth: temporally smooth a sequence of [x,y,w,h] boxes."
                    ),
                }),
                # ── transform ──
                "expand_x": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1,
                                      "tooltip": "[transform] dilate/erode along X"}),
                "expand_y": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1,
                                      "tooltip": "[transform] dilate/erode along Y"}),
                "blur_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                                      "tooltip": "[transform] Gaussian sigma X"}),
                "blur_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                                      "tooltip": "[transform] Gaussian sigma Y"}),
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                                      "tooltip": "[transform] pixel shift X"}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                                      "tooltip": "[transform] pixel shift Y"}),
                "feather": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5,
                                       "tooltip": "[transform/draw] feather radius"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "[transform] binarize threshold"}),
                "invert": ("BOOLEAN", {"default": False,
                                        "tooltip": "[transform] invert output"}),

                # ── canvas / shape common ──
                "width": ("INT", {"default": 512, "min": 1, "max": 16384,
                                   "tooltip": "[draw_*/points_bbox] canvas width"}),
                "height": ("INT", {"default": 512, "min": 1, "max": 16384,
                                    "tooltip": "[draw_*/points_bbox] canvas height"}),

                # ── draw_shape ──
                "shape": (cls.SHAPES, {"default": "circle",
                                        "tooltip": "[draw_shape] geometry"}),
                "cx": ("FLOAT", {"default": 256.0, "min": -16384.0, "max": 16384.0, "step": 0.5}),
                "cy": ("FLOAT", {"default": 256.0, "min": -16384.0, "max": 16384.0, "step": 0.5}),
                "radius": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 8192.0, "step": 0.5}),
                "size_w": ("FLOAT", {"default": 200.0, "min": 0.0, "max": 16384.0, "step": 0.5}),
                "size_h": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 16384.0, "step": 0.5}),
                "rx": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 8192.0, "step": 0.5}),
                "ry": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 8192.0, "step": 0.5}),
                "top_left_x": ("FLOAT", {"default": 100.0, "min": -16384.0, "max": 16384.0, "step": 0.5}),
                "top_left_y": ("FLOAT", {"default": 100.0, "min": -16384.0, "max": 16384.0, "step": 0.5}),
                "x2": ("FLOAT", {"default": 400.0, "min": -16384.0, "max": 16384.0, "step": 0.5}),
                "y2": ("FLOAT", {"default": 400.0, "min": -16384.0, "max": 16384.0, "step": 0.5}),
                "thickness": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 500.0, "step": 0.5}),
                "outer_r": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 8192.0, "step": 0.5}),
                "inner_r": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 8192.0, "step": 0.5}),
                "num_points": ("INT", {"default": 5, "min": 3, "max": 50}),
                "corner_radius": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 4096.0, "step": 0.5}),
                "cross_size": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 8192.0, "step": 0.5}),
                "arrow_length": ("FLOAT", {"default": 200.0, "min": 0.0, "max": 16384.0, "step": 0.5}),
                "head_length": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 8192.0, "step": 0.5}),
                "head_width": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 8192.0, "step": 0.5}),
                "points_json_shape": ("STRING", {
                    "default": "[[100,100],[400,100],[400,400],[100,400]]",
                    "multiline": True,
                    "tooltip": "[draw_shape] polygon vertices when shape=polygon",
                }),
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "[draw_*] fill intensity"}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.5,
                                        "tooltip": "[draw_*] rotation deg"}),
                "operation": (["set", "add", "subtract", "max", "min"], {
                    "default": "set",
                    "tooltip": "[draw_*] blend op",
                }),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 256,
                                        "tooltip": "[draw_shape] number of frames"}),

                # ── draw_advanced ──
                "shape_params_json": ("STRING", {
                    "default": '{"cx": 256, "cy": 256, "radius": 50}',
                    "multiline": True,
                    "tooltip": "[draw_advanced] raw shape_params JSON (see MaskDrawFrame)",
                }),

                # ── points_bbox ──
                "editor_data": ("STRING", {
                    "default": '{"points":[],"bboxes":[]}',
                    "multiline": True,
                    "tooltip": "[points_bbox] JSON from the interactive canvas",
                }),
                "default_radius": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 256.0, "step": 0.5,
                                              "tooltip": "[points_bbox] default brush radius"}),
                "softness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                                        "tooltip": "[points_bbox] gaussian sigma multiplier"}),
                "normalize": ("BOOLEAN", {"default": True,
                                           "tooltip": "[points_bbox] clamp output to [0,1]"}),

                # ── bbox_smooth ──
                "bboxes_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "[bbox_smooth] JSON array of [x,y,w,h] per frame",
                }),
                "smoothing_radius": ("INT", {"default": 3, "min": 1, "max": 30, "step": 1,
                                              "tooltip": "[bbox_smooth] window radius"}),
                "smoothing_method": (["median_then_exponential", "moving_average",
                                       "exponential", "median"], {
                    "default": "median_then_exponential",
                    "tooltip": "[bbox_smooth] smoothing strategy",
                }),
                "alpha": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05,
                                     "tooltip": "[bbox_smooth] exponential factor"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Source mask (transform requires this)"}),
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional reference; canvas size matches it when supplied",
                }),
                "existing_mask": ("MASK", {
                    "tooltip": "Existing mask to blend onto in draw modes",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "STRING", "BBOX", "BBOX",
                    "STRING", "STRING", "BBOX")
    RETURN_NAMES = ("mask", "positive_coords", "negative_coords",
                    "bboxes", "neg_bboxes",
                    "points_json", "bbox_json", "primary_bbox")
    OUTPUT_TOOLTIPS = (
        "Rendered mask for the active mode (zero-mask in bbox_smooth mode).",
        "Positive points JSON (points_bbox mode); '[]' otherwise.",
        "Negative points JSON (points_bbox mode); '[]' otherwise.",
        "Positive bbox list (points_bbox); single-element list in bbox_smooth.",
        "Negative bbox list (points_bbox mode).",
        "All points JSON (points_bbox); '[]' otherwise.",
        "All bbox JSON (points_bbox) or smoothed bboxes JSON (bbox_smooth).",
        "Primary [x,y,w,h]: first positive bbox or smoothed first bbox.",
    )
    FUNCTION = "execute"
    CATEGORY = "MaskEditControl/Edit"
    DESCRIPTION = (
        "Unified mask edit dispatcher. Pure-CPU. Modes: transform, "
        "draw_shape, draw_advanced, points_bbox, bbox_smooth. "
        "Pick a mode and the corresponding widgets drive that engine. "
        "All outputs are normalized to the same 8-port schema."
    )

    # ──────────────────────────────────────────────────────────────────
    def execute(self, mode, expand_x, expand_y, blur_x, blur_y, offset_x,
                offset_y, feather, threshold, invert,
                width, height,
                shape, cx, cy, radius, size_w, size_h, rx, ry,
                top_left_x, top_left_y, x2, y2, thickness,
                outer_r, inner_r, num_points, corner_radius, cross_size,
                arrow_length, head_length, head_width, points_json_shape,
                value, rotation, operation, batch_size,
                shape_params_json,
                editor_data, default_radius, softness, normalize,
                bboxes_json, smoothing_radius, smoothing_method, alpha,
                mask=None, reference_image=None, existing_mask=None):

        if mode == "transform":
            return self._mode_transform(
                mask, expand_x, expand_y, blur_x, blur_y,
                offset_x, offset_y, feather, threshold, invert,
            )
        if mode == "draw_shape":
            return self._mode_draw_shape(
                width, height, shape, cx, cy, radius, size_w, size_h,
                rx, ry, top_left_x, top_left_y, x2, y2, thickness,
                outer_r, inner_r, num_points, corner_radius, cross_size,
                arrow_length, head_length, head_width, points_json_shape,
                value, feather, rotation, operation, batch_size,
                existing_mask, reference_image,
            )
        if mode == "draw_advanced":
            return self._mode_draw_advanced(
                width, height, shape, shape_params_json,
                value, feather, rotation, operation,
                existing_mask, reference_image,
            )
        if mode == "points_bbox":
            return self._mode_points_bbox(
                width, height, editor_data, default_radius, softness,
                normalize, reference_image, existing_mask,
            )
        if mode == "bbox_smooth":
            return self._mode_bbox_smooth(
                bboxes_json, smoothing_radius, smoothing_method, alpha,
            )
        # Fallback: zero mask
        return (_empty_mask(mask if mask is not None else reference_image),
                "[]", "[]", [], [], "[]", "[]", _EMPTY_BBOX)

    # ── transform ────────────────────────────────────────────────────
    def _mode_transform(self, mask, expand_x, expand_y, blur_x, blur_y,
                        offset_x, offset_y, feather, threshold, invert):
        if mask is None:
            return (_empty_mask(), "[]", "[]", [], [], "[]", "[]", _EMPTY_BBOX)
        impl = MaskTransformXY()
        (out_mask,) = impl.transform(
            mask=mask, expand_x=int(expand_x), expand_y=int(expand_y),
            blur_x=float(blur_x), blur_y=float(blur_y),
            offset_x=int(offset_x), offset_y=int(offset_y),
            feather=float(feather), threshold=float(threshold),
            invert=bool(invert),
        )
        return (out_mask, "[]", "[]", [], [], "[]", "[]", _EMPTY_BBOX)

    # ── draw_shape ───────────────────────────────────────────────────
    def _mode_draw_shape(self, width, height, shape, cx, cy, radius,
                         size_w, size_h, rx, ry, top_left_x, top_left_y,
                         x2, y2, thickness, outer_r, inner_r, num_points,
                         corner_radius, cross_size, arrow_length,
                         head_length, head_width, points_json_shape,
                         value, feather, rotation, operation, batch_size,
                         existing_mask, reference_image):
        impl = DrawShapeMEC()
        result = impl.draw(
            width=int(width), height=int(height), shape=str(shape),
            cx=float(cx), cy=float(cy), radius=float(radius),
            size_w=float(size_w), size_h=float(size_h),
            rx=float(rx), ry=float(ry),
            top_left_x=float(top_left_x), top_left_y=float(top_left_y),
            x2=float(x2), y2=float(y2), thickness=float(thickness),
            outer_r=float(outer_r), inner_r=float(inner_r),
            num_points=int(num_points),
            corner_radius=float(corner_radius),
            cross_size=float(cross_size),
            arrow_length=float(arrow_length),
            head_length=float(head_length), head_width=float(head_width),
            points_json=str(points_json_shape),
            value=float(value), feather=float(feather),
            rotation=float(rotation), operation=str(operation),
            batch_size=int(batch_size),
            coords_json="",
            existing_mask=existing_mask,
            reference_image=reference_image,
        )
        out_mask = result[0] if isinstance(result, tuple) else result
        return (out_mask, "[]", "[]", [], [], "[]", "[]", _EMPTY_BBOX)

    # ── draw_advanced ────────────────────────────────────────────────
    def _mode_draw_advanced(self, width, height, shape, shape_params_json,
                            value, feather, rotation, operation,
                            existing_mask, reference_image):
        impl = MaskDrawFrame()
        result = impl.draw(
            width=int(width), height=int(height), shape=str(shape),
            shape_params_json=str(shape_params_json),
            value=float(value), feather=float(feather),
            rotation=float(rotation), operation=str(operation),
            existing_mask=existing_mask,
            reference_image=reference_image,
        )
        out_mask = result[0] if isinstance(result, tuple) else result
        return (out_mask, "[]", "[]", [], [], "[]", "[]", _EMPTY_BBOX)

    # ── points_bbox ──────────────────────────────────────────────────
    def _mode_points_bbox(self, width, height, editor_data, default_radius,
                          softness, normalize, reference_image,
                          existing_mask):
        impl = PointsMaskEditor()
        result = impl.generate(
            width=int(width), height=int(height),
            editor_data=str(editor_data),
            default_radius=float(default_radius),
            softness=float(softness),
            normalize=bool(normalize),
            reference_image=reference_image,
            existing_mask=existing_mask,
        )
        # PointsMaskEditor returns ComfyUI {"ui": ..., "result": (8-tuple)}
        # or, in some paths, a raw 8-tuple. Handle both.
        if isinstance(result, dict) and "result" in result:
            payload = result["result"]
        else:
            payload = result
        if isinstance(payload, tuple) and len(payload) == 8:
            # Normalize any None / unexpected BBOX-port values so downstream
            # nodes never receive None on a declared BBOX port.
            mask, pos_c, neg_c, bboxes, neg_bboxes, pts_json, bbox_json, prim_bbox = payload
            if bboxes is None:
                bboxes = []
            if neg_bboxes is None:
                neg_bboxes = []
            if prim_bbox is None or not isinstance(prim_bbox, (list, tuple)) \
                    or len(prim_bbox) < 4:
                prim_bbox = list(_EMPTY_BBOX)
            return (mask, pos_c, neg_c, bboxes, neg_bboxes,
                    pts_json, bbox_json, prim_bbox)
        # Defensive fallback
        return (_empty_mask(reference_image), "[]", "[]", [], [],
                "[]", "[]", list(_EMPTY_BBOX))

    # ── bbox_smooth ──────────────────────────────────────────────────
    def _mode_bbox_smooth(self, bboxes_json, smoothing_radius, method, alpha):
        impl = BBoxSmooth()
        smoothed_json, first_bbox = impl.smooth(
            bboxes_json=str(bboxes_json),
            smoothing_radius=int(smoothing_radius),
            method=str(method),
            alpha=float(alpha),
        )
        # Build a bboxes list from the smoothed JSON (defensive).
        try:
            bboxes_list = json.loads(smoothed_json)
            if not isinstance(bboxes_list, list):
                bboxes_list = []
        except (ValueError, TypeError):
            bboxes_list = []

        return (_empty_mask(), "[]", "[]", bboxes_list, [],
                "[]", smoothed_json, first_bbox)


NODE_CLASS_MAPPINGS = {"MaskEditMEC": MaskEditMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskEditMEC": "Mask Edit — Transform/Draw/Points/BBox",
}
