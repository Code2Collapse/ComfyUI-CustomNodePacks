"""Run one Image Mask Editor mode by the former node's own code (L7.65 P01). Returns
(mask [B,H,W] float, uses_input_mask, outputs dict, ui dict, info str); the node applies the shared result stage."""
from __future__ import annotations

from typing import Any, Optional

import torch

EMPTY_BBOX = [0, 0, 0, 0]


def _outputs(**kw) -> dict:
    out = {"positive_coords": "[]", "negative_coords": "[]", "bboxes": [], "neg_bboxes": [], "points_json": "[]",
           "bbox_json": "[]", "primary_bbox": list(EMPTY_BBOX), "spline_data_out": "[]", "painted_image": None}
    out.update(kw)
    return out


def _as_bhw(mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    return m


def run_points(a: dict, image: Optional[torch.Tensor], input_mask: Optional[torch.Tensor]):
    from ..mask_edit_mec import MaskEditMEC
    t = MaskEditMEC()._mode_points_bbox(
        a["points_width"], a["points_height"], a["points_editor_data"], a["points_default_radius"],
        a["points_softness"], a["points_normalize"], image, input_mask)
    mask, pos, neg, bboxes, neg_bboxes, pts, bbox_json, prim = t
    return _as_bhw(mask), True, _outputs(positive_coords=pos, negative_coords=neg, bboxes=bboxes,
                                         neg_bboxes=neg_bboxes, points_json=pts, bbox_json=bbox_json,
                                         primary_bbox=prim), {}, "points + box"


def run_shape(a: dict, image: Optional[torch.Tensor], input_mask: Optional[torch.Tensor]):
    from ..mask_edit_mec import MaskEditMEC
    me = MaskEditMEC()
    if a["shape_input"] == "JSON":
        t = me._mode_draw_advanced(a["shape_width"], a["shape_height"], a["shape_kind"], a["shape_json"],
                                   a["shape_value"], a["shape_feather"], a["shape_rotation"], a["shape_operation"],
                                   input_mask, image)
    else:
        t = me._mode_draw_shape(
            a["shape_width"], a["shape_height"], a["shape_kind"], a["shape_cx"], a["shape_cy"], a["shape_radius"],
            a["shape_size_w"], a["shape_size_h"], a["shape_rx"], a["shape_ry"], a["shape_top_left_x"],
            a["shape_top_left_y"], a["shape_x2"], a["shape_y2"], a["shape_thickness"], a["shape_outer_r"],
            a["shape_inner_r"], a["shape_num_points"], a["shape_corner_radius"], a["shape_cross_size"],
            a["shape_arrow_length"], a["shape_head_length"], a["shape_head_width"], a["shape_points_json"],
            a["shape_value"], a["shape_feather"], a["shape_rotation"], a["shape_operation"], a["shape_batch_size"],
            input_mask, image)
    return _as_bhw(t[0]), True, _outputs(), {}, f"shape ({a['shape_kind']}, {a['shape_input']})"


def run_spline(a: dict, image: Optional[torch.Tensor], node_id: Any):
    from ..spline_mask_mec import SplineMaskMEC
    sm = SplineMaskMEC()
    if a["spline_fill"] == "flow pattern":
        mask, _c, data, info, _b = sm._mode_flow_path(
            a["spline_data"], a["spline_flow_pattern"], a["spline_width"], a["spline_height"],
            a["spline_flow_thickness"], a["spline_flow_amplitude"], a["spline_flow_frequency"],
            a["spline_flow_turbulence"], a["spline_flow_turbulence_scale"], a["spline_flow_edge_softness"],
            a["spline_flow_taper_start"], a["spline_flow_taper_end"], a["spline_flow_frames"],
            a["spline_flow_animation_speed"], a["spline_type"], a["spline_samples_per_segment"], a["spline_closed"],
            a["spline_invert"], a["spline_flow_seed"], a["spline_flow_flow_direction"], a["spline_flow_mod_decay"],
            a["spline_flow_use_embedded_editor"], image)
        return _as_bhw(mask), False, _outputs(spline_data_out=data), {}, info
    r = sm._mode_edit(image, a["spline_data"], a["spline_type"], a["spline_closed"], a["spline_smoothing"],
                      a["spline_samples_per_segment"], a["spline_feather_radius"], a["spline_invert"],
                      a["spline_centripetal_alpha"], a["spline_width"], a["spline_height"], a["spline_mask_color"],
                      a["spline_mask_opacity"], node_id)
    mask, coords, data, bbox_json, bbox = r["result"]
    return _as_bhw(mask), False, _outputs(positive_coords=coords, spline_data_out=data, bbox_json=bbox_json,
                                          primary_bbox=bbox), r.get("ui") or {}, "spline"


def run_roto(a: dict):
    from ..roto import VectorRotoMEC
    mask, info = VectorRotoMEC().rasterize(a["roto_json"], a["roto_frame_count"], a["roto_width"], a["roto_height"],
                                           a["roto_samples_per_seg"], a["roto_feather_px"], a["roto_frame"])
    return _as_bhw(mask), False, _outputs(), {}, info


def run_video(a: dict, image: Optional[torch.Tensor], input_mask: Optional[torch.Tensor]):
    from ..video_mask_editor import VideoMaskEditorMEC
    if image is None:
        raise ValueError("Image Mask Editor (video): connect the video frames to image.")
    mask, info = VideoMaskEditorMEC().execute(image, a["video_session_id"], a["video_tween_mode"],
                                              a["video_feather"], a["video_threshold"], input_mask,
                                              a["video_outside_keys"])
    return _as_bhw(mask), True, _outputs(), {}, info


def run_colour_layer(a: dict, image: Optional[torch.Tensor]):
    """The Paint Canvas: (painted_image, its processed mask)."""
    from ..mec_paint_suite import MECAdvancedPaintCanvas
    painted, mask = MECAdvancedPaintCanvas().execute(
        a["canvas_width"], a["canvas_height"], a["canvas_brush_type"], a["canvas_brush_color"],
        a["canvas_brush_opacity"], a["canvas_brush_hardness"], a["canvas_brush_size"], a["canvas_mask_hardness"],
        a["canvas_mask_expansion"], a["canvas_mask_blur_radius"], a["canvas_mask_blur_strength"],
        a["canvas_data"], reference_image=image)
    return painted, _as_bhw(mask)
