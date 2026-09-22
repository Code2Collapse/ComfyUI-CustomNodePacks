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


NODE_CLASS_MAPPINGS = {
    "MaskFromColorMEC": MaskFromColorMEC,
    "MaskGradientMEC": MaskGradientMEC,
    "MaskGrainMEC": MaskGrainMEC,
    "MaskMotionBlurMEC": MaskMotionBlurMEC,
    "EdgeSpreadMEC": EdgeSpreadMEC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromColorMEC": "Mask From Color",
    "MaskGradientMEC": "Mask Gradient",
    "MaskGrainMEC": "Mask Grain",
    "MaskMotionBlurMEC": "Mask Motion Blur",
    "EdgeSpreadMEC": "Edge Spread",
}
