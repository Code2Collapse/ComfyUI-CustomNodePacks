"""
SplineMaskMEC – Unified spline-based mask editor / tracker / flow-path.

Combines (via composition) the previously-separate spline nodes:
    edit       → SplineMaskEditorMEC  (interactive canvas, single-frame seed)
    track      → SplineMaskTrackerMEC (multi-keyframe LK optical-flow tracker)
    flow_path  → SplinePathFlowMaskMEC (procedural ribbon/wave/dust/lightning…)

Single ``mode`` widget selects which engine runs; the rich JS spline
canvas widget binds to this unified class so the same interactive
control-point editor drives every mode.

Unified RETURN_TYPES (5 ports):
    0 MASK         mask          (always)
    1 STRING       coords_json   (edit mode SAM coords; "[]" otherwise)
    2 SPLINE_DATA  spline_data   (edit mode structured payload; passthrough)
    3 STRING       info_json     (track diagnostics, flow params, or bbox JSON)
    4 BBOX         bbox          ([x,y,w,h] of control points; [0,0,0,0] if N/A)
"""

from __future__ import annotations

import hashlib
import json

import torch

from .spline_mask_editor import SplineMaskEditorMEC
from .spline_mask_tracker import SplineMaskTrackerMEC
from .spline_path_flow_mask import SplinePathFlowMaskMEC


_EMPTY_BBOX = [0, 0, 0, 0]


def _empty_mask_from_image(image: torch.Tensor | None,
                            H_default: int = 512, W_default: int = 512) -> torch.Tensor:
    if image is not None and isinstance(image, torch.Tensor) and image.dim() == 4:
        return torch.zeros(image.shape[0], image.shape[1], image.shape[2],
                           dtype=torch.float32)
    return torch.zeros(1, H_default, W_default, dtype=torch.float32)


class SplineMaskMEC:
    """Unified spline mask node — edit / track / flow_path.

    Always uses the same canvas-based spline-data JSON as input; the
    interpretation depends on the chosen mode.
    """

    MODES = ["edit", "track", "flow_path"]

    PATTERNS = [
        "ribbon", "wave", "flow", "dust", "river", "smoke",
        "sawtooth", "square", "triangle", "gaussian_pulse",
        "fbm", "curl_noise", "lightning",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (cls.MODES, {
                    "default": "edit",
                    "tooltip": (
                        "edit: rasterize a single spline (closed/open) to a mask.\n"
                        "track: Lucas-Kanade multi-keyframe tracker across a video.\n"
                        "flow_path: procedural pattern along the spline (waves/dust/lightning…)."
                    ),
                }),

                # ── shared spline payload ──
                "spline_data": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "tooltip": (
                        "Spline payload from the JS canvas.\n"
                        "edit/flow_path: single shape list.\n"
                        "track: keyframes list [{frame:int, points:[[x,y],…]}, …]."
                    ),
                }),
                "spline_type": (["catmull_rom", "bezier", "polyline"], {
                    "default": "catmull_rom",
                    "tooltip": "[edit/flow_path] interpolation method",
                }),
                "closed": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "[all] closed loop vs open path",
                }),
                "samples_per_segment": ("INT", {
                    "default": 20, "min": 2, "max": 128, "step": 1,
                    "tooltip": "[all] curve resolution per segment",
                }),
                "feather_radius": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.5,
                    "tooltip": "[edit/track] gaussian edge feather",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[edit/flow_path] invert mask",
                }),

                # ── edit-mode params ──
                "smoothing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "[edit] enable spline smoothing",
                }),
                "centripetal_alpha": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "[edit] Catmull-Rom alpha (0.5 = centripetal)",
                }),
                "width": ("INT", {
                    "default": 0, "min": 0, "max": 16384, "step": 1,
                    "tooltip": "[edit/flow_path] output width (0 = inherit image)",
                }),
                "height": ("INT", {
                    "default": 0, "min": 0, "max": 16384, "step": 1,
                    "tooltip": "[edit/flow_path] output height (0 = inherit image)",
                }),
                "mask_color": ("STRING", {
                    "default": "#ff00ff",
                    "tooltip": "[edit] preview overlay color (hex)",
                }),
                "mask_opacity": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "[edit] preview overlay opacity",
                }),

                # ── track-mode params ──
                "tracking_weight": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "[track] lerp/tracker blend (1=pure LK, 0=pure lerp)",
                }),
                "klt_window": ("INT", {
                    "default": 21, "min": 5, "max": 51, "step": 2,
                    "tooltip": "[track] Lucas-Kanade window size",
                }),
                "stroke_width": ("INT", {
                    "default": 3, "min": 1, "max": 64, "step": 1,
                    "tooltip": "[track] stroke width for open splines",
                }),

                # ── flow_path-mode params ──
                "pattern": (cls.PATTERNS, {
                    "default": "ribbon",
                    "tooltip": "[flow_path] procedural pattern",
                }),
                "thickness": ("FLOAT", {
                    "default": 12.0, "min": 0.0, "max": 1024.0, "step": 0.5,
                    "tooltip": "[flow_path] base stroke thickness",
                }),
                "amplitude": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 1024.0, "step": 0.5,
                    "tooltip": "[flow_path] modulation amplitude",
                }),
                "frequency": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 64.0, "step": 0.1,
                    "tooltip": "[flow_path] modulation frequency",
                }),
                "turbulence": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "[flow_path] noise turbulence strength",
                }),
                "turbulence_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 32.0, "step": 0.05,
                    "tooltip": "[flow_path] noise spatial scale",
                }),
                "edge_softness": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 32.0, "step": 0.1,
                    "tooltip": "[flow_path] edge softness in pixels",
                }),
                "taper_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "[flow_path] start-end taper amount",
                }),
                "taper_end": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "[flow_path] tail-end taper amount",
                }),
                "frames": ("INT", {
                    "default": 1, "min": 1, "max": 4096, "step": 1,
                    "tooltip": "[flow_path] number of animation frames",
                }),
                "animation_speed": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 4.0, "step": 0.005,
                    "tooltip": "[flow_path] phase advance per frame",
                }),
                "flow_direction": (["forward", "reverse", "bidirectional",
                                     "oscillate"], {
                    "default": "forward",
                    "tooltip": "[flow_path] flow direction",
                }),
                "mod_decay": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 4.0, "step": 0.01,
                    "tooltip": "[flow_path] modulation falloff over time",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "tooltip": "[flow_path] noise seed",
                }),
                "use_embedded_editor": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "[flow_path] show embedded spline preview",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "[edit/track/flow_path] reference / source video frames",
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "SPLINE_DATA", "STRING", "BBOX")
    RETURN_NAMES = ("mask", "coords_json", "spline_data_out",
                     "info_json", "bbox")
    OUTPUT_TOOLTIPS = (
        "Rasterized mask (B,H,W).",
        "SAM-compatible point coords (edit mode); '[]' otherwise.",
        "Structured spline data (edit mode); pass-through input otherwise.",
        "Mode diagnostics: edit→bbox_json, track→info, flow_path→params.",
        "AABB of control points as [x,y,w,h]; [0,0,0,0] when N/A.",
    )
    FUNCTION = "execute"
    CATEGORY = "C2C/Spline"
    DESCRIPTION = (
        "Unified spline mask node. Same control-point canvas drives "
        "three modes: edit (single-frame rasterize), track (LK optical "
        "flow across video), flow_path (procedural ribbon/wave/dust). "
        "CPU + small VRAM. No models required."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, mode, image=None, **kwargs):
        if mode == "track" and image is None:
            return "[C2C] Connect a video/image to the Image input for track mode."
        return True

    @classmethod
    def IS_CHANGED(cls, mode, spline_data, **kwargs):
        h = hashlib.md5()
        h.update(str(mode).encode())
        h.update(str(spline_data).encode())
        for k in sorted(kwargs):
            v = kwargs[k]
            if isinstance(v, torch.Tensor):
                h.update(v.cpu().numpy().tobytes())
            elif v is not None:
                h.update(str(v).encode())
        return h.hexdigest()

    # ──────────────────────────────────────────────────────────────────
    def execute(self, mode, spline_data, spline_type, closed,
                samples_per_segment, feather_radius, invert,
                smoothing, centripetal_alpha, width, height,
                mask_color, mask_opacity,
                tracking_weight, klt_window, stroke_width,
                pattern, thickness, amplitude, frequency,
                turbulence, turbulence_scale, edge_softness,
                taper_start, taper_end, frames, animation_speed,
                flow_direction, mod_decay, seed, use_embedded_editor,
                image=None, node_id=None):

        if mode == "edit":
            return self._mode_edit(
                image, spline_data, spline_type, closed, smoothing,
                samples_per_segment, feather_radius, invert,
                centripetal_alpha, width, height, mask_color,
                mask_opacity, node_id,
            )
        if mode == "track":
            return self._mode_track(
                image, spline_data, closed, samples_per_segment,
                tracking_weight, klt_window, feather_radius,
                stroke_width,
            )
        if mode == "flow_path":
            return self._mode_flow_path(
                spline_data, pattern, width, height, thickness,
                amplitude, frequency, turbulence, turbulence_scale,
                edge_softness, taper_start, taper_end, frames,
                animation_speed, spline_type, samples_per_segment,
                closed, invert, seed, flow_direction, mod_decay,
                use_embedded_editor, image,
            )
        # Fallback
        return (_empty_mask_from_image(image), "[]", spline_data,
                "{}", _EMPTY_BBOX)

    # ── edit ─────────────────────────────────────────────────────────
    def _mode_edit(self, image, spline_data, spline_type, closed,
                   smoothing, samples_per_segment, feather_radius,
                   invert, centripetal_alpha, width, height,
                   mask_color, mask_opacity, node_id):
        if image is None:
            # Synthetic 512x512 transparent image so the editor still works.
            image = torch.zeros(1, 512, 512, 3, dtype=torch.float32)
        impl = SplineMaskEditorMEC()
        result = impl.execute(
            image=image, spline_data=str(spline_data),
            spline_type=str(spline_type), closed=bool(closed),
            smoothing=bool(smoothing),
            samples_per_segment=int(samples_per_segment),
            feather_radius=float(feather_radius), invert=bool(invert),
            centripetal_alpha=float(centripetal_alpha),
            width=int(width), height=int(height),
            mask_color=str(mask_color), mask_opacity=float(mask_opacity),
            node_id=node_id,
        )
        # SplineMaskEditorMEC returns dict {"ui": ..., "result": (...)}
        if isinstance(result, dict):
            ui = result.get("ui", {})
            payload = result["result"]
        else:
            ui = {}
            payload = result
        mask, coords_json, spline_data_out, bbox_json, bbox = payload
        return {
            "ui": ui,
            "result": (mask, coords_json, spline_data_out, bbox_json, bbox),
        }

    # ── track ────────────────────────────────────────────────────────
    def _mode_track(self, image, keyframes_json, closed,
                    samples_per_segment, tracking_weight, klt_window,
                    feather_radius, stroke_width):
        if image is None:
            return (_empty_mask_from_image(None), "[]", keyframes_json,
                    json.dumps({"error": "track mode requires image input"}),
                    _EMPTY_BBOX)
        impl = SplineMaskTrackerMEC()
        mask, info = impl.execute(
            image=image, keyframes_json=str(keyframes_json),
            closed=bool(closed),
            samples_per_segment=int(samples_per_segment),
            tracking_weight=float(tracking_weight),
            klt_window=int(klt_window),
            feather_radius=float(feather_radius),
            stroke_width=int(stroke_width),
        )
        return (mask, "[]", keyframes_json, info, _EMPTY_BBOX)

    # ── flow_path ────────────────────────────────────────────────────
    def _mode_flow_path(self, spline_data, pattern, width, height,
                        thickness, amplitude, frequency, turbulence,
                        turbulence_scale, edge_softness, taper_start,
                        taper_end, frames, animation_speed, spline_type,
                        samples_per_segment, closed, invert, seed,
                        flow_direction, mod_decay, use_embedded_editor,
                        image):
        impl = SplinePathFlowMaskMEC()
        result = impl.execute(
            spline_data=str(spline_data), pattern=str(pattern),
            width=int(width), height=int(height),
            thickness=float(thickness), amplitude=float(amplitude),
            frequency=float(frequency), turbulence=float(turbulence),
            turbulence_scale=float(turbulence_scale),
            edge_softness=float(edge_softness),
            taper_start=float(taper_start), taper_end=float(taper_end),
            frames=int(frames), animation_speed=float(animation_speed),
            spline_type=str(spline_type),
            samples_per_segment=int(samples_per_segment),
            closed=bool(closed), invert=bool(invert), seed=int(seed),
            flow_direction=str(flow_direction),
            mod_decay=float(mod_decay),
            use_embedded_editor=bool(use_embedded_editor),
            image=image,
        )
        out_mask = result[0] if isinstance(result, tuple) else result
        info = json.dumps({
            "pattern": pattern, "frames": int(frames),
            "thickness": float(thickness),
            "flow_direction": flow_direction,
            "seed": int(seed),
        })
        return (out_mask, "[]", spline_data, info, _EMPTY_BBOX)


NODE_CLASS_MAPPINGS = {"SplineMaskMEC": SplineMaskMEC}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplineMaskMEC": "Spline Mask — Edit/Track/Flow-Path",
}
