# Ported from ComfyUI_LayerStyle by chflame163 (MIT,
# https://github.com/chflame163/ComfyUI_LayerStyle), the LayerMask family:
# mask_by_color, create_gradient_mask, mask_gradient, mask_grain,
# mask_motion_blur and pixel_spread.
#
# Only the five capabilities this pack did NOT already have were taken. The
# rest of LayerMask is covered by MaskRefineMEC, LayerEffectStrokeMEC and
# nodes/bbox_nodes.py; see NOTICE for the full list of what was skipped and
# why. Reimplemented in torch, not copied: upstream round-trips every frame
# through PIL.
"""MEC Mask toolkit nodes — torch-native, batch-correct."""
from __future__ import annotations

import numpy as np
import torch

from .._is_changed_util import hash_args_and_kwargs
from ._ops import (
    align_mask_batch,
    build_report,
    clone_output,
    edge_spread_image,
    ensure_bhw4_image,
    ensure_bhw_mask,
    extract_channel,
    gradient_mask_bhw,
    mask_from_color,
    mask_grain_bhw,
    motion_blur_mask,
    resolve_size,
)

_CATEGORY = "MEC/Mask"

_COLOR_DESC = (
    "Pick a colour and tolerance to build a matte. Gap filling and edge cleanup are "
    "deliberately not here — chain Mask Refine (MEC), which does hole fill, morph, "
    "guided filter and CRF properly."
)


class MaskFromColorMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Plate to sample against the picked colour."}),
                "color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Target colour as #RRGGBB — the hue you want to keep or remove.",
                }),
                "colorspace": (["rgb", "hsv", "lab"], {
                    "default": "rgb",
                    "tooltip": (
                        "Distance metric. LAB separates dark navy from dark brown; "
                        "RGB treats them as similar luminance."
                    ),
                }),
                "tolerance": ("FLOAT", {
                    "default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "How far a pixel's colour can drift from the pick and still count as a match.",
                }),
                "soft_falloff": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": (
                        "Graded band outside tolerance — stops a colour key from cutting "
                        "like a pair of scissors."
                    ),
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Swap matched and unmatched regions.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = _COLOR_DESC

    @classmethod
    def IS_CHANGED(cls, image, color, colorspace, tolerance, soft_falloff, invert, **kwargs):
        return hash_args_and_kwargs(image, color, colorspace, tolerance, soft_falloff, invert, **kwargs)

    def execute(
        self,
        image: torch.Tensor,
        color: str,
        colorspace: str,
        tolerance: float,
        soft_falloff: float,
        invert: bool,
    ) -> tuple[torch.Tensor, str]:
        image = ensure_bhw4_image(image, "MaskFromColorMEC")
        b, h, w, _ = image.shape
        notes: list[str] = []
        mask = mask_from_color(
            image, color, colorspace, tolerance, soft_falloff,
            device=image.device, dtype=image.dtype,
        )
        if invert:
            mask = 1.0 - mask
        mask = clone_output(mask)
        cov = mask.mean().item() * 100.0
        rep = build_report("MaskFromColorMEC", b, notes, coverage=cov)
        return mask, rep


class MaskGradientMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512, "min": 4, "max": 16384, "step": 1,
                    "tooltip": "Output width when size_as is not connected.",
                }),
                "height": ("INT", {
                    "default": 512, "min": 4, "max": 16384, "step": 1,
                    "tooltip": "Output height when size_as is not connected.",
                }),
                "gradient_type": (["linear", "radial", "angular"], {
                    "default": "linear",
                    "tooltip": "Ramp shape — linear edge fade, radial vignette, or angular wipe.",
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1,
                    "tooltip": "Direction for linear, or rotation offset for angular ramps.",
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Radial/angular origin, normalised 0–1 across the frame.",
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Radial/angular origin, normalised 0–1 down the frame.",
                }),
                "start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mask value at the ramp start (black end).",
                }),
                "end": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mask value at the ramp end (white end).",
                }),
            },
            "optional": {
                "size_as": ("IMAGE", {
                    "tooltip": (
                        "When connected, output size is taken from this image and "
                        "width/height sliders are ignored."
                    ),
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional matte to multiply into — fade a garbage matte off at the frame edge.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Procedural linear, radial, or angular ramp matte with optional multiply."

    @classmethod
    def IS_CHANGED(cls, width, height, gradient_type, angle, center_x, center_y, start, end, **kwargs):
        return hash_args_and_kwargs(
            width, height, gradient_type, angle, center_x, center_y, start, end, **kwargs,
        )

    def execute(
        self,
        width: int,
        height: int,
        gradient_type: str,
        angle: float,
        center_x: float,
        center_y: float,
        start: float,
        end: float,
        size_as: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str]:
        notes: list[str] = []
        w, h = resolve_size(width, height, size_as, notes)
        batch = 1
        device = torch.device("cpu")
        dtype = torch.float32
        if mask is not None:
            mask = ensure_bhw_mask(mask, "MaskGradientMEC")
            batch = max(batch, mask.shape[0])
            device, dtype = mask.device, mask.dtype
        if size_as is not None:
            size_img = ensure_bhw4_image(size_as, "MaskGradientMEC")
            batch = max(batch, size_img.shape[0])
            device, dtype = size_img.device, size_img.dtype
        ramp = gradient_mask_bhw(
            batch, h, w, gradient_type, angle, center_x, center_y, start, end,
            device=device, dtype=dtype,
        )
        if mask is not None:
            (mask_aligned,), b = align_mask_batch([mask])
            assert mask_aligned is not None
            if mask_aligned.shape[-2:] != (h, w):
                mask_aligned = torch.nn.functional.interpolate(
                    mask_aligned.unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            ramp = ramp[:b] * mask_aligned[:b]
            batch = b
        ramp = clone_output(ramp)
        cov = ramp.mean().item() * 100.0
        rep = build_report("MaskGradientMEC", batch, notes, coverage=cov)
        return ramp, rep


class MaskGrainMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Matte to dither — breaks 8-bit banding when stretched."}),
                "amount": ("INT", {
                    "default": 6, "min": 0, "max": 127, "step": 1,
                    "tooltip": "Noise strength — enough to hide banding, not enough to read as texture.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0x7FFFFFFF, "step": 1,
                    "tooltip": (
                        "Fixed seed holds the grain still through a sequence; "
                        "upstream reseeded every frame and the matte crawled."
                    ),
                }),
                "grain_size": ("INT", {
                    "default": 4, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Noise frequency — larger values give coarser, less speckly grain.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flip mask polarity before adding grain.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Film grain on a matte to break quantisation banding."

    @classmethod
    def IS_CHANGED(cls, mask, amount, seed, grain_size, invert, **kwargs):
        return hash_args_and_kwargs(mask, amount, seed, grain_size, invert, **kwargs)

    def execute(
        self,
        mask: torch.Tensor,
        amount: int,
        seed: int,
        grain_size: int,
        invert: bool,
    ) -> tuple[torch.Tensor, str]:
        mask = ensure_bhw_mask(mask, "MaskGrainMEC")
        assert mask is not None
        notes: list[str] = []
        if invert:
            mask = 1.0 - mask
        out = mask_grain_bhw(
            mask, amount, seed, grain_size,
            device=mask.device, dtype=mask.dtype,
        )
        out = clone_output(out)
        cov = out.mean().item() * 100.0
        rep = build_report("MaskGrainMEC", mask.shape[0], notes, coverage=cov)
        return out, rep


class MaskMotionBlurMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": "Matte to smear — match motion blur already on the plate.",
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1,
                    "tooltip": "Blur direction in degrees, 0 = horizontal right.",
                }),
                "distance": ("INT", {
                    "default": 20, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "Blur length in pixels along the angle.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flip mask polarity before blurring.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Directional blur on a matte to match a moving plate."

    @classmethod
    def IS_CHANGED(cls, mask, angle, distance, invert, **kwargs):
        return hash_args_and_kwargs(mask, angle, distance, invert, **kwargs)

    def execute(
        self,
        mask: torch.Tensor,
        angle: float,
        distance: int,
        invert: bool,
    ) -> tuple[torch.Tensor, str]:
        mask = ensure_bhw_mask(mask, "MaskMotionBlurMEC")
        assert mask is not None
        notes: list[str] = []
        if invert:
            mask = 1.0 - mask
        out = clone_output(motion_blur_mask(mask, angle, distance, notes))
        cov = out.mean().item() * 100.0
        rep = build_report("MaskMotionBlurMEC", mask.shape[0], notes, coverage=cov)
        return out, rep


class EdgeSpreadMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Unpremultiplied plate — interior colour is pushed under the edge.",
                }),
                "spread": ("INT", {
                    "default": 4, "min": 0, "max": 9999, "step": 1,
                    "tooltip": (
                        "How many pixels to push inward colour outward — kills the dark "
                        "fringe on a lighter background."
                    ),
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flip mask polarity when your matte is white-on-black.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Matte defining the subject edge; alpha used if omitted."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Push interior plate colour outward under the matte edge. "
        "Output is an image, not a mask — fixes unpremult dark fringes."
    )

    @classmethod
    def IS_CHANGED(cls, image, spread, invert_mask, **kwargs):
        return hash_args_and_kwargs(image, spread, invert_mask, **kwargs)

    def execute(
        self,
        image: torch.Tensor,
        spread: int,
        invert_mask: bool,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str]:
        image = ensure_bhw4_image(image, "EdgeSpreadMEC")
        notes: list[str] = []
        if mask is None:
            if image.shape[-1] >= 4:
                matte = image[..., 3].clone()
            else:
                matte = torch.ones(image.shape[:-1], device=image.device, dtype=image.dtype)
        else:
            matte = ensure_bhw_mask(mask, "EdgeSpreadMEC")
            assert matte is not None
        (image, matte), b = align_mask_batch([image, matte])
        assert image is not None and matte is not None
        if invert_mask:
            matte = 1.0 - matte
        rgb = image[..., :3]
        outs = []
        for i in range(b):
            outs.append(edge_spread_image(rgb[i:i + 1], matte[i:i + 1], spread, notes))
        out = clone_output(torch.cat(outs, dim=0))
        rep = build_report("EdgeSpreadMEC", b, notes)
        return out, rep


# ── Mask Tools: keyers and matte operations in one node (L7.65 P03, owner ORDERS 11.4) ──────────────────────────
#
# One mode at a time. Eight former nodes are its engines and migrate here with their mode set (Mask From Color,
# Luminance Keyer, Mask Gradient, Mask Grain, Mask Motion Blur, Edge Spread, Mask Batch Combine, Shuffle); three modes
# are new: difference key, grade, and directional grow / shrink. Every control is prefixed with its mode and shown
# only in that mode (js/c2c_mode_widgets.js).

def _engine(name):
    if name == "luma_key":
        from ..luminance_keyer import LuminanceKeyerMEC
        return LuminanceKeyerMEC
    if name == "shuffle":
        from ..shuffle import ShuffleMEC
        return ShuffleMEC
    if name == "combine":
        from ..helpers.helpers import MaskBatchCombineMEC
        return MaskBatchCombineMEC
    if name == "transform":
        from ..mask_transform_xy import MaskTransformXY
        return MaskTransformXY
    return {"colour_key": MaskFromColorMEC, "gradient": MaskGradientMEC, "grain": MaskGrainMEC,
            "motion_blur": MaskMotionBlurMEC, "edge_spread": EdgeSpreadMEC}[name]


# mode label, key, the engine's parameters (None = a new mode, parameters declared below)
MASK_TOOL_MODES = (
    ("key: colour", "colour_key", ("color", "colorspace", "tolerance", "soft_falloff", "invert")),
    ("key: luma", "luma_key", ("mode", "low", "high", "gamma", "falloff", "invert", "channel", "low_soft", "high_soft",
                               "invert_key")),
    ("key: difference", "difference_key", None),
    ("grade", "grade", None),
    ("grow / shrink", "grow", None),
    ("combine", "combine", ("op",)),
    ("gradient", "gradient", ("width", "height", "gradient_type", "angle", "center_x", "center_y", "start", "end")),
    ("grain", "grain", ("amount", "seed", "grain_size", "invert")),
    ("motion blur", "motion_blur", ("angle", "distance", "invert")),
    ("edge spread", "edge_spread", ("spread", "invert_mask")),
    ("shuffle", "shuffle", ("out_R", "out_G", "out_B", "out_A", "premultiply_output")),
    # L7.65 P01 (owner 2026-10-27, D7.65c): the former Mask Edit "transform" mode (Mask Transform XY's code)
    ("transform", "transform", ("expand_x", "expand_y", "blur_x", "blur_y", "offset_x", "offset_y", "feather",
                                "threshold", "invert")),
    # L9.172 (A25 spec item 82): chroma key + despill, temporal matte cleanup - appended LAST (old saves keep values)
    ("key: chroma", "chroma_key", None),
    ("temporal cleanup", "temporal_cleanup", None),
)
# modes added after the node shipped: their controls are OPTIONAL inputs after the sockets, so a saved node keeps
# every widget value in place (values are restored by position) and an older API prompt without them stays valid
_APPENDED_MODES = {"transform"}
# L9.172 (A25): their controls are optional inputs placed after the L7.48 grade extras (the very end), so every widget
# that existed before keeps its position
_A25_MODES = {"chroma_key", "temporal_cleanup"}
_NEW_MODE_PARAMS = {
    "difference_key": {
        "tolerance": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005,
                                "tooltip": "How far a pixel may differ from the clean plate and still count as "
                                           "background (0..1 of the RGB distance)."}),
        "softness": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.005,
                               "tooltip": "Width of the soft edge above the tolerance."}),
        "invert": ("BOOLEAN", {"default": False, "tooltip": "Key the unchanged part instead of what changed."}),
    },
    "grade": {
        "gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.01,
                           "tooltip": "Multiply the matte (above 1 pushes greys toward white)."}),
        "gamma": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.01,
                            "tooltip": "Bend the greys: above 1 lifts them, below 1 sinks them; black and white "
                                       "stay put."}),
        "clamp": ("BOOLEAN", {"default": True, "tooltip": "Keep the result inside 0..1."}),
        "invert": ("BOOLEAN", {"default": False, "tooltip": "Flip the matte at the end."}),
    },
    "grow": {
        "left": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                         "tooltip": "Pixels to grow the matte toward the left (negative shrinks its left edge)."}),
        "right": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                          "tooltip": "Pixels to grow toward the right (negative shrinks its right edge)."}),
        "up": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                       "tooltip": "Pixels to grow upward (negative shrinks its top edge)."}),
        "down": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                         "tooltip": "Pixels to grow downward (negative shrinks its bottom edge)."}),
        "feather": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                            "tooltip": "Soften the result by this blur radius in pixels."}),
    },
    # L9.172 (A25 spec 82.1): colour-difference keyer with despill - _a25_ops.chroma_key
    "chroma_key": {
        "screen": (["green", "blue", "custom"], {
            "default": "green",
            "tooltip": "The backdrop: green (#00B140) or blue (#0047BB) screen, or custom (the colour below; its "
                       "strongest channel is the screen channel)."}),
        "color": ("STRING", {"default": "#00B140",
                             "tooltip": "custom screen: the backdrop colour as #RRGGBB (pick it from the plate)."}),
        "balance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                              "tooltip": "How the screen channel is compared with the other two: 1 = against their "
                                         "max (removes less backdrop), 0 = against their min (removes more), 0.5 "
                                         "between."}),
        "gain": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01,
                           "tooltip": "Screen strength: above 1 removes the backdrop harder (cleaner background, "
                                      "thinner edges), below 1 softer."}),
        "clip_black": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                 "tooltip": "Matte values at or below this become 0 (crush backdrop noise)."}),
        "clip_white": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                 "tooltip": "Matte values at or above this become 1 (make a see-through core solid). "
                                            "Must be above clip_black."}),
        "despill": (["average", "max (gentle)", "min (strong)", "off"], {
            "default": "average",
            "tooltip": "Spill suppression on the image output: the screen channel is limited to the average, the "
                       "max (removes least) or the min (removes most) of the other two channels; off = image "
                       "unchanged."}),
        "despill_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "How much of the spill above the limit is removed (1 = all)."}),
        "invert": ("BOOLEAN", {"default": False,
                               "tooltip": "Output the backdrop's matte instead of the subject's."}),
    },
    # L9.172 (A25 spec 82.2): stabilise a matte over time - _a25_ops.temporal_cleanup
    "temporal_cleanup": {
        "window": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                           "tooltip": "Frames on each side that vote (fewer at the clip's ends)."}),
        "method": (["median", "mean"], {
            "default": "median",
            "tooltip": "median: removes one-frame drop-outs and pops and keeps edges hard (an even count takes the "
                       "lower middle value); mean: softer, averages flicker away but blurs fast motion."}),
        "hysteresis": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.45, "step": 0.01,
                                 "tooltip": "Above 0: a pixel switches on only above 0.5 + this and off only below "
                                            "0.5 - this, so boundary chatter stops; the output becomes binary. "
                                            "0 = off."}),
        "protect_motion": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "With `image` wired (same frames and size): where the picture changes "
                                                "by more than this between frames, the original matte is kept so "
                                                "fast motion does not lag. 0 = off."}),
    },
}
_MODE_KEY = {label: key for label, key, _p in MASK_TOOL_MODES}
GROW_UNITS = ("px", "% of the matte size")


def _grade_extra_inputs() -> dict:
    """L7.48 (owner 2026-10-08, Nuke-style matte grade): Nuke Grade's black point, white point, lift and offset for the
    grade mode, appended LAST (after the L9.281 grow extras) so saved widget positions never shift. Defaults = the old
    grade, bit for bit. Nuke Grade: A = (gain - lift) / (whitepoint - blackpoint), B = offset + lift - A * blackpoint,
    out = (A * in + B) ** (1 / gamma)."""
    return {
        "grade_blackpoint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001,
                                       "tooltip": "grade: this matte value becomes black (0) - raise it to crush "
                                                  "faint spill."}),
        "grade_whitepoint": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.001,
                                       "tooltip": "grade: this matte value becomes white (1) - lower it to make a "
                                                  "weak core solid."}),
        "grade_lift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001,
                                 "tooltip": "grade: raise (or lower) the blacks while white stays put."}),
        "grade_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001,
                                   "tooltip": "grade: add this to every value (before gamma)."}),
    }


def _grow_extra_inputs() -> dict:
    """L9.281 (owner 2026-11-17): grow / shrink at any angle, in px or % of the matte, with a soft fall-off, and
    optionally along a face's tilt (POSE_KEYPOINT from Face Landmarks). Defaults = the old grow, bit for bit."""
    return {
        "grow_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                                 "tooltip": "grow / shrink: turn the up / down / left / right axes by this many "
                                            "degrees (counter-clockwise); 0 = the picture's axes."}),
        "grow_units": (list(GROW_UNITS), {"default": GROW_UNITS[0],
                                          "tooltip": "grow / shrink: left / right / up / down in pixels, or in % of "
                                                     "the matte's own width (left / right) and height (up / down)."}),
        "grow_falloff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4096.0, "step": 1.0,
                                   "tooltip": "grow / shrink: soft fall-off of the grown band, same units (0 = hard "
                                              "edge)."}),
        "grow_follow_face": ("BOOLEAN", {"default": True,
                                         "tooltip": "grow / shrink: with face_pose connected, turn the axes with each "
                                                    "frame's head tilt (added to grow_angle)."}),
        "face_pose": ("POSE_KEYPOINT", {"tooltip": "grow / shrink: Face Landmarks' pose_keypoint - the axes follow "
                                                   "the head's tilt per frame (first face)."}),
    }


def _pose_angle(pose, i: int):
    """Head tilt (degrees, counter-clockwise) of the first face in frame i of an OpenPose face list, or None."""
    try:
        frames = pose if isinstance(pose, list) else [pose]
        fr = frames[min(i, len(frames) - 1)]
        k = np.asarray(fr["people"][0]["face_keypoints_2d"], np.float64).reshape(-1, 3)[:68, :2]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not np.isfinite(k).all() or (k == 0).all():
        return None
    from ..facepoints_controls import face_frame
    return face_frame(k).angle_deg


def _directional_ext(m, a: dict, extra: dict):
    """The extended grow: per frame, rotated axes, units, fall-off (nodes/_mask_directional.py)."""
    import torch
    from .._mask_directional import extend
    out, notes = [], []
    pose = extra.get("face_pose")
    for i in range(m.shape[0]):
        f = m[i].detach().float().cpu().numpy()
        sx = sy = 1.0
        if extra["grow_units"] == GROW_UNITS[1]:
            ys, xs = np.nonzero(f > 0.5)
            if len(xs):
                sx, sy = (xs.max() - xs.min() + 1) / 100.0, (ys.max() - ys.min() + 1) / 100.0
            else:
                sx = sy = 0.0
        ang = float(extra["grow_angle"])
        if pose is not None and extra["grow_follow_face"]:
            pa = _pose_angle(pose, i)
            if pa is None:
                notes.append(f"frame {i}: no face in face_pose - picture axes used")
            else:
                ang += pa
        g = extend(f, up=a["up"] * sy, down=a["down"] * sy, left=a["left"] * sx, right=a["right"] * sx,
                   angle_deg=ang, falloff_px=float(extra["grow_falloff"]) * (sy if sx == sy else (sx + sy) / 2))
        out.append(torch.from_numpy(g))
    return torch.stack(out).to(m.device), notes[:8]
# controls most keys never touch: in a section, so a mode shows at most 10 controls (ORDERS 11.4)
_ADVANCED = {("luma_key", "channel"), ("luma_key", "low_soft"), ("luma_key", "high_soft"), ("luma_key", "invert_key")}


def mask_tool_widget(key: str, param: str) -> str:
    """colour_key + color -> colour_key_color, grain + grain_size -> grain_size, edge_spread + spread -> edge_spread."""
    last = key.rsplit("_", 1)[-1]
    if param == last:
        return key
    if param.startswith(last + "_"):
        return key + param[len(last):]
    return f"{key}_{param}"


def _directional(mask: torch.Tensor, left: int, right: int, up: int, down: int) -> torch.Tensor:
    """Grow (positive) or shrink (negative) a matte toward each side, grey-scale. Growing pads with nothing; shrinking
    repeats the frame edge, so a matte that runs off the frame is not eaten from outside it."""
    import torch.nn.functional as F

    x = mask.unsqueeze(1)                                         # [B,1,H,W]
    for n, axis, toward_low in ((left, 3, True), (right, 3, False), (up, 2, True), (down, 2, False)):
        if not n:
            continue
        k = abs(int(n))
        grow = n > 0
        # growing toward the low side looks at the k pixels on the high side, and the other way round; shrinking a
        # low edge looks at the k pixels on the low side
        pad_high = (grow and toward_low) or (not grow and not toward_low)
        if axis == 3:
            pad = (0, k, 0, 0) if pad_high else (k, 0, 0, 0)
            kernel = (1, k + 1)
        else:
            pad = (0, 0, 0, k) if pad_high else (0, 0, k, 0)
            kernel = (k + 1, 1)
        if grow:
            x = F.max_pool2d(F.pad(x, pad, mode="constant", value=0.0), kernel, stride=1)
        else:
            x = -F.max_pool2d(F.pad(-x, pad, mode="replicate"), kernel, stride=1)
    return x.squeeze(1)


class MaskToolsMEC:
    """Mask Tools: colour / luma / difference keys, matte grade, directional grow and shrink, combine, gradient, grain,
    motion blur, edge spread and channel shuffle - one mode at a time."""

    @classmethod
    def INPUT_TYPES(cls):
        import copy

        required = {"mode": ([label for label, _k, _p in MASK_TOOL_MODES], {
            "default": "key: colour",
            "tooltip": "What the node does. Each mode shows only its own controls."})}
        appended = {}
        a25 = {}
        for label, key, params in MASK_TOOL_MODES:
            target = appended if key in _APPENDED_MODES else (a25 if key in _A25_MODES else required)
            if params is None:
                spec = _NEW_MODE_PARAMS[key]
                names = list(spec)
            else:
                eng = _engine(key).INPUT_TYPES()
                spec = {**(eng.get("required") or {}), **(eng.get("optional") or {})}
                names = list(params)
            for p in names:
                s = copy.deepcopy(spec[p])
                opts = dict(s[1]) if len(s) > 1 else {}
                if key == "grain" and p == "seed":
                    opts["control_after_generate"] = True
                if (key, p) in _ADVANCED:                  # L7.81 / A28: the "More" bar, both renderers
                    opts["c2c_section"] = "Advanced"
                tip = opts.get("tooltip", "")
                opts["tooltip"] = f"{label}: {tip}" if tip else label
                target[mask_tool_widget(key, p)] = (s[0], opts)
        return {
            "required": required,
            "optional": {
                "image": ("IMAGE", {"tooltip": "keys, edge spread and shuffle: the picture to work on (chroma key: "
                                               "the green / blue screen plate). Temporal cleanup: the clip, for "
                                               "protect_motion."}),
                "mask": ("MASK", {"tooltip": "grade, grow / shrink, combine (first matte), grain, motion blur, "
                                             "transform, temporal cleanup; edge spread's matte; gradient's size when "
                                             "nothing else gives one."}),
                "mask_b": ("MASK", {"tooltip": "combine: the second matte."}),
                "plate": ("IMAGE", {"tooltip": "key: difference - the clean plate (same shot, without the subject)."}),
                "size_as": ("IMAGE", {"tooltip": "gradient: take the size from this image."}),
                **appended,
                **_grow_extra_inputs(),          # L9.281: appended LAST so saved widget positions never shift
                **_grade_extra_inputs(),         # L7.48: appended after them, same reason
                **a25,                           # L9.172 (A25): chroma key + temporal cleanup, appended LAST
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "image", "report")
    OUTPUT_TOOLTIPS = (
        "The matte (shuffle: the new alpha; edge spread: the matte it used).",
        "edge spread / shuffle: the processed picture. Other modes: the matte as a grey image, to look at.",
        "What the node did, and anything worth knowing (clamped values, resized inputs).",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Keyers and matte operations in one node: colour, luma and difference keys; grade (gain, gamma, clamp, "
        "invert, black / white point, lift, offset - as Nuke Grade); grow or shrink each side (left / right / up / "
        "down) with a feather; combine two mattes; gradient; "
        "grain; motion blur; edge spread; channel shuffle; transform (grow, blur and offset per axis, feather, "
        "threshold, invert); chroma key with despill (green / blue / custom screen); temporal cleanup of a matte "
        "over a clip (median / mean window, hysteresis, motion protection)."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    @staticmethod
    def _need(value, what, mode):
        if value is None:
            raise ValueError(f"Mask Tools ({mode}): connect {what}.")
        return value

    def execute(self, mode="key: colour", image=None, mask=None, mask_b=None, plate=None, size_as=None, **kw):
        key = _MODE_KEY.get(mode)
        if key is None:
            raise ValueError(f"Mask Tools: unknown mode {mode!r}.")
        params = dict((k, p) for _l, k, p in MASK_TOOL_MODES)[key]
        names = list(params) if params is not None else list(_NEW_MODE_PARAMS[key])
        args = {p: kw[mask_tool_widget(key, p)] for p in names if mask_tool_widget(key, p) in kw}
        if key in (_APPENDED_MODES | _A25_MODES) and len(args) < len(names):   # an optional control left out
            optional = self.INPUT_TYPES()["optional"]
            for p in names:
                w = mask_tool_widget(key, p)
                if p not in args and w in optional:
                    args[p] = optional[w][1].get("default")
        # L9.281: the grow extras (appended inputs) reach _run through the instance, its signature stays
        self._grow_extra = {k: kw.get(k, v[1].get("default")) for k, v in _grow_extra_inputs().items()}
        self._grade_extra = {k: kw.get(k, v[1].get("default")) for k, v in _grade_extra_inputs().items()}
        with torch.no_grad():
            out_mask, out_image, report = self._run(key, mode, args, image, mask, mask_b, plate, size_as)
        if out_image is None and out_mask is not None:
            m = out_mask if out_mask.dim() == 3 else out_mask.unsqueeze(0)
            out_image = clone_output(m.unsqueeze(-1).expand(*m.shape, 3))
        return out_mask, out_image, report

    def _run(self, key, mode, a, image, mask, mask_b, plate, size_as):
        if key == "colour_key":
            m, rep = _engine(key)().execute(self._need(image, "an image", mode), **a)
            return m, None, rep
        if key == "luma_key":
            m, rep = _engine(key)().key_luminance(self._need(image, "an image", mode), **a)
            return m, None, rep
        if key == "gradient":
            m, rep = _engine(key)().execute(size_as=size_as, mask=mask, **a)
            return m, None, rep
        if key in ("grain", "motion_blur"):
            m, rep = _engine(key)().execute(self._need(mask, "a mask", mode), **a)
            return m, None, rep
        if key == "edge_spread":
            img, rep = _engine(key)().execute(self._need(image, "an image", mode), mask=mask, **a)
            if mask is None:            # A22 L9.136: None went out as the MASK and crashed what was wired to it
                mask = torch.zeros(img.shape[:3], dtype=torch.float32, device=img.device)
            return mask, img, rep
        if key == "combine":
            (m,) = _engine(key)().combine(self._need(mask, "a mask", mode), self._need(mask_b, "mask_b", mode), **a)
            return m, None, f"Mask Tools: combine ({a.get('op')})"
        if key == "shuffle":
            img, alpha = _engine(key)().shuffle(self._need(image, "an image", mode), **a)
            return alpha, img, "Mask Tools: shuffle"
        if key == "difference_key":
            img = ensure_bhw4_image(self._need(image, "an image", mode), "Mask Tools")
            ref = ensure_bhw4_image(self._need(plate, "the clean plate", mode), "Mask Tools")
            notes: list[str] = []
            if ref.shape[1:3] != img.shape[1:3]:
                import torch.nn.functional as F
                ref = F.interpolate(ref.permute(0, 3, 1, 2), size=img.shape[1:3], mode="bilinear",
                                    align_corners=False).permute(0, 2, 3, 1)
                notes.append(f"plate resized to {img.shape[2]}x{img.shape[1]}")
            if ref.shape[0] != img.shape[0]:
                if ref.shape[0] < img.shape[0]:
                    # L9.262 (CNP-NODES-1 audit): a plate shorter than the clip (but not 1 frame) crashed with a raw
                    # shape error; its last frame now holds for the rest, and the report says so
                    idx = torch.clamp(torch.arange(img.shape[0]), max=ref.shape[0] - 1)
                    if ref.shape[0] > 1:
                        notes.append(f"the plate has {ref.shape[0]} frames for {img.shape[0]}: its last frame was "
                                     "held for the rest")
                    ref = ref[idx]
                else:
                    ref = ref[: img.shape[0]]
            dist = ((img[..., :3] - ref[..., :3]) ** 2).sum(-1).sqrt() / (3 ** 0.5)
            tol, soft = float(a["tolerance"]), max(float(a["softness"]), 1e-6)
            m = ((dist - tol) / soft).clamp(0.0, 1.0)
            m = m * m * (3.0 - 2.0 * m)                                   # smoothstep edge
            if a["invert"]:
                m = 1.0 - m
            return clone_output(m), None, build_report("Mask Tools: difference key", img.shape[0], notes,
                                                     coverage=float(m.mean()) * 100.0)
        if key == "transform":
            (m,) = _engine(key)().transform(
                mask=self._need(mask, "a mask", mode), expand_x=int(a["expand_x"]), expand_y=int(a["expand_y"]),
                blur_x=float(a["blur_x"]), blur_y=float(a["blur_y"]), offset_x=int(a["offset_x"]),
                offset_y=int(a["offset_y"]), feather=float(a["feather"]), threshold=float(a["threshold"]),
                invert=bool(a["invert"]))
            return m, None, build_report("Mask Tools: transform", m.shape[0] if m.dim() == 3 else 1, [],
                                         coverage=float(m.float().mean()) * 100.0)
        if key == "chroma_key":                          # L9.172 (A25 spec 82.1)
            from ._a25_ops import chroma_key
            img = ensure_bhw4_image(self._need(image, "an image", mode), "Mask Tools")
            m, rgb, notes = chroma_key(img, str(a["screen"]), str(a["color"]), float(a["balance"]), float(a["gain"]),
                                       float(a["clip_black"]), float(a["clip_white"]), str(a["despill"]),
                                       float(a["despill_strength"]), bool(a["invert"]))
            return (clone_output(m), clone_output(rgb),
                    build_report("Mask Tools: chroma key", img.shape[0], notes, coverage=float(m.mean()) * 100.0))
        if key == "temporal_cleanup":                    # L9.172 (A25 spec 82.2)
            from ._a25_ops import temporal_cleanup
            src = ensure_bhw_mask(self._need(mask, "a mask", mode), "Mask Tools")
            g, notes = temporal_cleanup(src, int(a["window"]), str(a["method"]), float(a["hysteresis"]),
                                        float(a["protect_motion"]), image)
            return clone_output(g), None, build_report("Mask Tools: temporal cleanup", src.shape[0], notes,
                                                       coverage=float(g.mean()) * 100.0)
        m = ensure_bhw_mask(self._need(mask, "a mask", mode), "Mask Tools")
        if key == "grade":
            ge = getattr(self, "_grade_extra", None) or {k: v[1].get("default") for k, v in _grade_extra_inputs().items()}
            bp, wp = float(ge["grade_blackpoint"]), float(ge["grade_whitepoint"])
            lift, off = float(ge["grade_lift"]), float(ge["grade_offset"])
            if (bp, wp, lift, off) == (0.0, 1.0, 0.0, 0.0):
                g = m * float(a["gain"])                    # the old grade, bit for bit
            else:
                if abs(wp - bp) < 1e-6:
                    raise ValueError("Mask Tools grade: white point and black point are the same value - move one "
                                     "of them apart (white point is usually above black point).")
                A = (float(a["gain"]) - lift) / (wp - bp)
                g = m * A + (off + lift - A * bp)
            g = g.clamp(min=0.0) ** (1.0 / float(a["gamma"]))
            if a["clamp"]:
                g = g.clamp(0.0, 1.0)
            if a["invert"]:
                g = 1.0 - g
            return clone_output(g), None, build_report("Mask Tools: grade", m.shape[0], [],
                                                       coverage=float(g.mean()) * 100.0)
        if key == "grow":
            notes = []
            extra = getattr(self, "_grow_extra", None) or {k: v[1].get("default")
                                                           for k, v in _grow_extra_inputs().items()}
            if extra["grow_units"] not in GROW_UNITS:
                raise ValueError(f"Mask Tools: grow_units must be one of {', '.join(GROW_UNITS)}")
            if (float(extra["grow_angle"]) % 360.0 == 0.0 and extra["grow_units"] == GROW_UNITS[0]
                    and float(extra["grow_falloff"]) == 0.0
                    and (extra["face_pose"] is None or not extra["grow_follow_face"])):
                g = _directional(m.float(), a["left"], a["right"], a["up"], a["down"])    # the old grow, exactly
            else:
                g, n2 = _directional_ext(m.float(), a, extra)
                notes += n2
            if int(a["feather"]) > 0:
                from ..layer_effects._ops import blur_mask
                g = blur_mask(g, int(a["feather"]), notes)
            return clone_output(g.clamp(0.0, 1.0)), None, build_report("Mask Tools: grow / shrink", m.shape[0], notes,
                                                                        coverage=float(g.mean()) * 100.0)
        raise ValueError(f"Mask Tools: mode {mode!r} is not implemented.")


NODE_CLASS_MAPPINGS = {
    "MaskToolsMEC": MaskToolsMEC,
}

# L9.172 (A25): the inputs appended after the A28 pin (read by tests/test_a28_cnp_a_layout.py) - the chroma key and
# temporal cleanup controls, in INPUT_TYPES order
APPENDED_AFTER_A28 = tuple(mask_tool_widget(k, p) for _l, k, _p in MASK_TOOL_MODES if k in _A25_MODES
                           for p in _NEW_MODE_PARAMS[k])

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToolsMEC": "Mask Tools (C2C)",
}


# A28 (ORDERS 29, docs/audit/A28/PARAM_CNP_A.md "MaskToolsMEC"): which controls show when, read by js/_c2c_node_layout.js.
# Display only: every input keeps its name, order, type and default, is saved and reaches the node's function as before.
# A mode's controls carry its key as a prefix (MASK_TOOL_MODES); the luma presets set low / high themselves
# (custom only); a gradient takes its size from size_as when that is wired (_ops.py:323-329).
try:
    from .._layout_decl import declare_layout, merge, sections
except ImportError:  # pragma: no cover - loaded outside the pack
    declare_layout = None
if declare_layout is not None:
    def _mask_tools_layout(spec):
        keys = [(label, key) for label, key, _p in MASK_TOOL_MODES]
        out = {}
        for group in ("required", "optional"):
            for name in (spec.get(group) or {}):
                hit = next((lab for lab, k in keys if name == k or name.startswith(k + "_")), None)
                if hit is None:
                    continue
                out[name] = {"c2c_show_if": {"mode": [hit]}}
                s = spec[group][name]
                if len(s) > 1 and isinstance(s[1], dict) and s[1].get("c2c_section"):
                    out[name]["c2c_section"] = s[1]["c2c_section"]
        for n in ("luma_key_low", "luma_key_high"):
            out[n] = {"c2c_show_if": {"mode": ["key: luma"], "luma_key_mode": ["custom"]}}
        if "chroma_key_color" in out:            # L9.172: the colour is read only for a custom screen
            out["chroma_key_color"] = {"c2c_show_if": {"mode": ["key: chroma"], "chroma_key_screen": ["custom"]}}
        for n in ("gradient_width", "gradient_height"):
            out.setdefault(n, {})["c2c_hide_if_linked"] = ["size_as"]
        # L9.281 directional grow (grow_angle / grow_units / grow_falloff / grow_follow_face get the grow / shrink
        # rule from their prefix above, no section, like the other grow controls): following the face's tilt only
        # means something with face_pose wired
        if "grow_follow_face" in out:
            out["grow_follow_face"]["c2c_show_if_linked"] = ["face_pose"]
        return out
    declare_layout(MaskToolsMEC, _mask_tools_layout)
