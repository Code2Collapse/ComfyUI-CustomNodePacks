# PORTED FROM: ComfyUI-NKD-Basic-Tools by Nekodificador (MIT)
# https://github.com/Nekodificador/ComfyUI-NKD-Basic-Tools
"""MEC Frequency separation and film grain nodes — torch-native, batch-correct."""
from __future__ import annotations

import torch

from .._is_changed_util import hash_args_and_kwargs
from ._ops import build_report, film_grain, frequency_combine, frequency_separate

_CATEGORY = "MEC/Frequency"

_CLAMP_TOOLTIP = (
    "Output is clamped to 0..1 on the IMAGE socket — values above 1.0 are clipped; "
    "this is an 8-bit-range IMAGE, not a scene-linear plate."
)

_METHOD_TOOLTIP = (
    "How the base is smoothed. Guided keeps edges clean without the halo Gaussian "
    "leaves; Rolling Guidance erases texture by size while keeping shapes; Median "
    "is for spot blemishes (radius capped internally)."
)

_RADIUS_TOOLTIP = (
    "Detail scale. Larger = coarser base, more goes into the detail layer. "
    "Median mode caps the effective radius at 7 — see the report for what ran."
)

_EDGE_TOOLTIP = (
    "How strongly edges are protected (Guided / Rolling Guidance). "
    "Lower = sharper edges kept."
)

_MODE_TOOLTIP = (
    "How detail is encoded. Divide (ratio) is lighting-invariant — best for "
    "transferring detail between differently-lit images. Negative ratios are "
    "discarded on Combine before recombine."
)

_DETAIL_TOOLTIP = (
    "Luminance keeps detail achromatic so a Divide recombine never touches color. "
    "RGB carries chromatic detail too."
)

_LINEAR_TOOLTIP = (
    "Process in linear light (correct). Turn off to work in gamma like classic "
    "Photoshop. Must match between Separate and Combine."
)

_MASK_SEP_TOOLTIP = (
    "Optional — attenuate the detail layer outside the mask on high_frequency_masked."
)

_LF_TOOLTIP = (
    "Base image the detail lands on (e.g. the relit result). Leaves Separate in "
    "display space when linear=True; Combine re-linearises it."
)

_MASK_COMB_TOOLTIP = (
    "Optional — apply the detail only inside the mask (feathered by its values)."
)

_FEATHER_TOOLTIP = "Soften the mask edge before applying detail."

_STRENGTH_TOOLTIP = (
    "Attenuate (<1) or boost (>1) the transferred detail. 0.0 restores the base only."
)


class FrequencySeparateMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Split into a soft base (low frequency) and a fine detail layer "
                        "(high frequency). Pair with Frequency Combine to transfer texture."
                    ),
                }),
                "method": (["Gaussian", "Guided", "Rolling Guidance", "Median"], {
                    "default": "Guided",
                    "tooltip": _METHOD_TOOLTIP,
                }),
                "radius": ("INT", {
                    "default": 8, "min": 1, "max": 128, "step": 1,
                    "tooltip": _RADIUS_TOOLTIP,
                }),
                "edge_threshold": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": _EDGE_TOOLTIP,
                }),
                "mode": (["Divide", "Subtract"], {
                    "default": "Divide",
                    "tooltip": _MODE_TOOLTIP,
                }),
                "detail": (["Luminance", "RGB"], {
                    "default": "Luminance",
                    "tooltip": _DETAIL_TOOLTIP,
                }),
                "linear": ("BOOLEAN", {
                    "default": True,
                    "tooltip": _LINEAR_TOOLTIP,
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": _MASK_SEP_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("high_frequency", "low_frequency", "high_frequency_masked", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Split an image into low-frequency base and high-frequency detail. "
        "HF leaves in working space; LF leaves in display space when linear=True."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(
        self,
        image,
        method,
        radius,
        edge_threshold,
        mode,
        detail,
        linear,
        mask=None,
    ):
        with torch.no_grad():
            hf, lf, hf_masked, notes = frequency_separate(
                image, method, int(radius), float(edge_threshold),
                mode, detail, bool(linear), mask=mask,
            )
            rep = build_report("Frequency Separate (MEC)", image.shape[0], notes)
            return hf.clone(), lf.clone(), hf_masked.clone(), rep


class FrequencyCombineMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_frequency": ("IMAGE", {
                    "tooltip": "Detail layer from Frequency Separate (mode/linear must match).",
                }),
                "low_frequency": ("IMAGE", {
                    "tooltip": _LF_TOOLTIP,
                }),
                "mode": (["Divide", "Subtract"], {
                    "default": "Divide",
                    "tooltip": "Must match the Separate that made the HF.",
                }),
                "linear": ("BOOLEAN", {
                    "default": True,
                    "tooltip": _LINEAR_TOOLTIP,
                }),
                "detail_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": _STRENGTH_TOOLTIP,
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": _MASK_COMB_TOOLTIP}),
                "mask_feather": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 1.0,
                    "tooltip": _FEATHER_TOOLTIP,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Merge a high-frequency detail layer onto a low-frequency base. "
        f"{_CLAMP_TOOLTIP}"
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(
        self,
        high_frequency,
        low_frequency,
        mode,
        linear,
        detail_strength,
        mask=None,
        mask_feather=0.0,
    ):
        with torch.no_grad():
            out, notes = frequency_combine(
                high_frequency, low_frequency, mode, bool(linear),
                float(detail_strength), mask=mask, mask_feather=float(mask_feather),
            )
            rep = build_report("Frequency Combine (MEC)", low_frequency.shape[0], notes)
            return out.clone(), rep


class FilmGrainMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Lightroom / Camera Raw-style film grain: Amount, Size, Roughness. "
                        "Monochrome by default; raise Color for dye-cloud colour grain."
                    ),
                }),
                "amount": ("FLOAT", {
                    "default": 25.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "How much grain shows through (Lightroom Amount).",
                }),
                "size": ("FLOAT", {
                    "default": 25.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Grain size — bigger = coarser; noise is generated at reduced resolution then upscaled.",
                }),
                "roughness": ("FLOAT", {
                    "default": 50.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": (
                        "Irregularity of the grain: blends a fine and a coarse field, like Lightroom."
                    ),
                }),
                "color": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "0 = clean monochrome grain. Raise for coloured dye-cloud grain.",
                }),
                "animate": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Fresh grain on every frame of the batch — real film shimmer. "
                        "Off = the same static grain on all frames."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "Same seed, same grain.",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": (
                        "Optional — confine the grain to the mask, feathered by its values "
                        "(soft edges blend in)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Lightroom-style film grain with per-frame animation for video batches."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(
        self,
        image,
        amount,
        size,
        roughness,
        color,
        animate,
        seed,
        mask=None,
    ):
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                out, notes = film_grain(
                    image, float(amount), float(size), float(roughness), float(color),
                    int(seed), bool(animate), mask=mask, device=device,
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                out, notes = film_grain(
                    image, float(amount), float(size), float(roughness), float(color),
                    int(seed), bool(animate), mask=mask, device=torch.device("cpu"),
                )
            rep = build_report("Film Grain (MEC)", image.shape[0], notes)
            return out.clone(), rep


NODE_CLASS_MAPPINGS = {
    "FrequencySeparateMEC": FrequencySeparateMEC,
    "FrequencyCombineMEC": FrequencyCombineMEC,
    "FilmGrainMEC": FilmGrainMEC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrequencySeparateMEC": "Frequency Separate (MEC)",
    "FrequencyCombineMEC": "Frequency Combine (MEC)",
    "FilmGrainMEC": "Film Grain (MEC)",
}
