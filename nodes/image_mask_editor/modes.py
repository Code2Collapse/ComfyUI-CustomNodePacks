"""Image Mask Editor modes (L7.65 P01 = L7.54): every kind of mask drawing in one node.

The node keeps its own inputs first (image, editor_id, frame_mode, grow, feather, threshold, invert, input_mask,
combine) and APPENDS `mode` and each mode's controls, prefixed with the mode, in the order of the node they come from
(design: docs/design/image_mask_editor_modes.md; owner D7.65c). Saved widget values are restored by position, so
nothing before them may move.

Each mode runs the former node's own code, so a migrated node gives the old node's result:
    paint         this node's mask layer (the modal editor + mask store) and, from the Paint Canvas, a colour layer
    points + box  Mask Edit "points_bbox"       (PointsMaskEditor)
    shape         Mask Edit "draw_shape" / "draw_advanced" (DrawShapeMEC / MaskDrawFrame)
    spline        Spline Mask "edit" / "flow_path" (SplineMaskEditorMEC / SplinePathFlowMaskMEC)
    roto          Vector Roto                   (VectorRotoMEC.rasterize)
    video         Video Mask Editor             (VideoMaskEditorMEC)
"""
from __future__ import annotations

import copy
from typing import Any

MODES = ["paint", "points + box", "shape", "spline", "roto", "video"]
SHAPE_INPUTS = ["parameters", "JSON"]
SPLINE_FILLS = ["mask", "flow pattern"]
CANVAS_LAYERS = ["mask", "colour"]

# (prefix, the old node's widget, its name here) - the old name when it already carries the prefix
_POINTS = ("width", "height", "editor_data", "default_radius", "softness", "normalize")
_SHAPE = ("width", "height", "shape", "cx", "cy", "radius", "size_w", "size_h", "rx", "ry", "top_left_x",
          "top_left_y", "x2", "y2", "thickness", "outer_r", "inner_r", "num_points", "corner_radius", "cross_size",
          "arrow_length", "head_length", "head_width", "points_json_shape", "value", "rotation", "operation",
          "batch_size", "shape_params_json", "feather")
_SHAPE_RENAME = {"shape": "shape_kind", "points_json_shape": "shape_points_json", "shape_params_json": "shape_json"}
_SPLINE_EDIT = ("spline_data", "spline_type", "closed", "samples_per_segment", "feather_radius", "invert", "smoothing",
                "centripetal_alpha", "width", "height", "mask_color", "mask_opacity")
_SPLINE_FLOW = ("pattern", "thickness", "amplitude", "frequency", "turbulence", "turbulence_scale", "edge_softness",
                "taper_start", "taper_end", "frames", "animation_speed", "flow_direction", "mod_decay", "seed",
                "use_embedded_editor")
_ROTO = ("roto_json", "frame_count", "width", "height", "samples_per_seg", "feather_px", "roto_frame")
_VIDEO = ("session_id", "tween_mode", "feather", "threshold", "outside_keys")
_CANVAS = ("canvas_width", "canvas_height", "brush_type", "brush_color", "brush_opacity", "brush_hardness",
           "brush_size", "mask_hardness", "mask_expansion", "mask_blur_radius", "mask_blur_strength", "canvas_data")


def _name(prefix: str, old: str, rename: dict | None = None) -> str:
    if rename and old in rename:
        return rename[old]
    return old if old.startswith(prefix + "_") else f"{prefix}_{old}"


def _copy(spec: Any, label: str) -> tuple:
    s = copy.deepcopy(spec)
    opts = dict(s[1]) if len(s) > 1 and isinstance(s[1], dict) else {}
    tip = str(opts.get("tooltip", "")).strip()
    for tag in ("[edit/flow_path] ", "[edit/track] ", "[all] ", "[edit] ", "[track] ", "[flow_path] ",
                "[points_bbox] ", "[draw_shape] ", "[draw_advanced] ", "[draw_*] "):
        if tip.startswith(tag):
            tip = tip[len(tag):]
    opts["tooltip"] = f"[{label}] {tip}" if tip else f"[{label}]"
    return (s[0], opts)


def mode_inputs() -> dict:
    """The appended optional inputs, in order: mode, then every mode's controls."""
    from ..mask_edit_mec import MaskEditMEC
    from ..mec_paint_suite import MECAdvancedPaintCanvas
    from ..roto import VectorRotoMEC
    from ..spline_mask_mec import SplineMaskMEC
    from ..video_mask_editor import VideoMaskEditorMEC

    def flat(cls):
        spec = cls.INPUT_TYPES()
        return {**(spec.get("required") or {}), **(spec.get("optional") or {})}

    me, sm, ro, vm, pc = flat(MaskEditMEC), flat(SplineMaskMEC), flat(VectorRotoMEC), flat(VideoMaskEditorMEC), \
        flat(MECAdvancedPaintCanvas)
    out: dict = {"mode": (MODES, {
        "default": "paint",
        "tooltip": "How the mask is drawn. paint: brush, shapes, lasso, fill and smart select in the editor (and a "
                   "colour layer); points + box: clicks and boxes for SAM / SeC; shape: a parametric shape; spline: "
                   "a spline shape or a pattern along it; roto: Bezier shapes keyed over frames; video: painted "
                   "keyframes, tweened between. Each mode shows only its own controls."})}
    out["canvas_layer"] = (CANVAS_LAYERS, {"default": "mask", "tooltip": "[paint] mask: paint the mask in the "
                                           "editor; colour: paint colour strokes on the canvas (painted_image), "
                                           "their alpha also becomes mask."})
    for n in _CANVAS:
        out[_name("canvas", n)] = _copy(pc[n], "paint: colour")
    for n in _POINTS:
        out[_name("points", n)] = _copy(me[n], "points + box")
    out["shape_input"] = (SHAPE_INPUTS, {"default": "parameters", "tooltip": "[shape] parameters: the shape from "
                                         "the controls below; JSON: the shapes from shape_json (power mode)."})
    for n in _SHAPE:
        out[_name("shape", n, _SHAPE_RENAME)] = _copy(me[n], "shape")
    for n in _SPLINE_EDIT:
        out[_name("spline", n)] = _copy(sm[n], "spline")
    out["spline_fill"] = (SPLINE_FILLS, {"default": "mask", "tooltip": "[spline] mask: fill (closed) or stroke "
                                         "(open) the spline; flow pattern: a procedural pattern along it (ribbon, "
                                         "wave, dust, smoke...), animated over frames."})
    for n in _SPLINE_FLOW:
        out[f"spline_flow_{n}"] = _copy(sm[n], "spline: flow pattern")
    # the front end gives an INT named "seed" a control_after_generate on its own; under its new name the seed needs it
    # asked for, or a migrated Spline Mask would lose its fixed / randomize setting (frontend 1.53.10, F2.32)
    out["spline_flow_seed"][1]["control_after_generate"] = True
    for n in _ROTO:
        out[_name("roto", n)] = _copy(ro[n], "roto")
    for n in _VIDEO:
        out[_name("video", n)] = _copy(vm[n], "video")
    return out


def defaults(spec: dict) -> dict:
    out = {}
    for k, v in spec.items():
        opts = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
        if "default" in opts:
            out[k] = opts["default"]
        elif isinstance(v[0], (list, tuple)) and v[0]:
            out[k] = v[0][0]
    return out
