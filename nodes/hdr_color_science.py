"""HDR Color Science nodes for professional video output.

Provides ACES filmic tone mapping, linear/sRGB conversion, and
HDR-aware processing for Wan video outputs. These nodes sit
between VAE decode and final output to improve color quality.

Inspired by fxtdstudios/radiance's 32-bit color science pipeline.
Clean-room implementation of standard color science operations.
"""
from __future__ import annotations

import logging
import math
from typing import Tuple

import torch

from ._is_changed_util import hash_args_and_kwargs

log = logging.getLogger("MEC.HDRColor")


class C2CACESTonemap:
    """Apply ACES filmic tone mapping for HDR-to-SDR conversion.

    Uses the Stephen Hill ACES approximation (from Unity/Unreal standard
    libraries). Converts linear-light RGB to display-referred sRGB with
    film-like highlight rolloff and shadow lift.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_space": (["sRGB", "Linear", "Log C3"], {
                    "default": "sRGB",
                    "tooltip": "Input color space. The image is linearised from this space before ACES processing.",
                }),
                "exposure": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 10.0, "step": 0.05,
                    "tooltip": "Exposure multiplier applied before tone mapping.",
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Contrast adjustment (applied in log space).",
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Color saturation. <1 desaturates, >1 boosts.",
                }),
                "output_colorspace": (["sRGB (gamma)", "Linear", "ACES AP1"], {
                    "default": "sRGB (gamma)",
                    "tooltip": "Output color space. sRGB for display, Linear for compositing.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_tonemap"
    CATEGORY = "MEC/Color Science"
    DESCRIPTION = (
        "ACES filmic tone mapping with exposure, contrast, and saturation "
        "controls. Converts linear-light or overbright pixels to "
        "display-ready output with film-like highlight rolloff."
    )

    @classmethod
    def IS_CHANGED(cls, image, source_space, exposure, contrast, saturation,
                   output_colorspace, **kwargs):
        return hash_args_and_kwargs(
            image, source_space, exposure, contrast, saturation, output_colorspace, **kwargs,
        )

    def apply_tonemap(self, image, source_space, exposure, contrast, saturation,
                      output_colorspace):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("C2CACESTonemap expects IMAGE tensor [B,H,W,C]")
        with torch.inference_mode():
            x = image.clone().float()

            # Linearise from source color space
            if source_space == "sRGB":
                x = _srgb_to_linear(x)
            elif source_space == "Log C3":
                x = _logc3_to_linear(x)
            # "Linear" → already linear, no conversion needed

            # Exposure
            x = x * exposure

            # Contrast in log space (around mid-gray 0.18)
            if abs(contrast - 1.0) > 0.01:
                x = x.clamp(min=1e-6)
                log_x = torch.log2(x / 0.18)
                log_x = log_x * contrast
                x = 0.18 * (2.0 ** log_x)

            # Saturation
            if abs(saturation - 1.0) > 0.01:
                luma = 0.2126 * x[..., 0:1] + 0.7152 * x[..., 1:2] + 0.0722 * x[..., 2:3]
                x = luma + saturation * (x - luma)
                x = x.clamp(min=0.0)

            # ACES filmic curve
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            result = (x * (a * x + b)) / (x * (c * x + d) + e)
            result = result.clamp(0.0, 1.0)

            # Output colorspace
            if output_colorspace == "sRGB (gamma)":
                result = _linear_to_srgb(result)
            elif output_colorspace == "Linear":
                pass  # already linear after ACES
            # ACES AP1 stays as-is (ACES output)

            return (result,)


class C2CVAEQualityDecode:
    """High-quality VAE decode with fp32 precision and spatial-only tiling.

    Wan VAE produces significantly better results when decoded in fp32.
    This node wraps the standard VAE decode with quality improvements:
    - Forces fp32 computation during decode
    - Uses spatial-only tiling (temporal coherence preserved)
    - Optional ACES tone mapping post-decode
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "force_fp32": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force fp32 during VAE decode for maximum quality.",
                }),
                "tile_size": ("INT", {
                    "default": 0, "min": 0, "max": 1024, "step": 64,
                    "tooltip": "Spatial tile size (0=auto/no tiling). Set 256+ for 1080p.",
                }),
                "apply_aces": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply ACES filmic tone mapping after decode.",
                }),
                "exposure": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 10.0, "step": 0.05,
                    "tooltip": "Exposure for ACES (only used if apply_aces=True).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "MEC/Color Science"
    DESCRIPTION = (
        "High-fidelity VAE decode for Wan video. Forces fp32 precision, "
        "uses spatial-only tiling to prevent frame flickering, and "
        "optionally applies ACES tone mapping for HDR-quality output."
    )

    @classmethod
    def IS_CHANGED(cls, samples, vae, force_fp32, tile_size, apply_aces, exposure, **kwargs):
        return hash_args_and_kwargs(
            samples, vae, force_fp32, tile_size, apply_aces, exposure, **kwargs,
        )

    def decode(self, samples, vae, force_fp32, tile_size, apply_aces, exposure):
        with torch.inference_mode():
            dtype = torch.float32 if force_fp32 else torch.float16

            latent = samples["samples"]

            if latent.ndim == 5 and tile_size > 0:
                try:
                    from .wan_director.features._local_vae_hdr import decode_wan_spatial_tiled
                    result = decode_wan_spatial_tiled(vae, latent, tile_size=tile_size, dtype=dtype)
                except Exception as exc:
                    log.warning("Spatial-tiled decode failed (%s); falling back to standard.", exc)
                    result = vae.decode(latent.to(dtype=dtype))
            else:
                original_dtype = None
                if force_fp32 and hasattr(vae, "first_stage_model"):
                    try:
                        original_dtype = next(vae.first_stage_model.parameters()).dtype
                        if original_dtype != torch.float32:
                            vae.first_stage_model.to(dtype=torch.float32)
                    except (StopIteration, AttributeError):
                        pass

                result = vae.decode(latent.to(dtype=dtype))

                if original_dtype is not None and original_dtype != torch.float32:
                    try:
                        vae.first_stage_model.to(dtype=original_dtype)
                    except (AttributeError, RuntimeError):
                        pass

            if isinstance(result, torch.Tensor):
                result = result.float().clamp(0.0, 1.0)

            if apply_aces:
                a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
                x = result * exposure
                result = (x * (a * x + b)) / (x * (c * x + d) + e)
                result = result.clamp(0.0, 1.0)
                result = _linear_to_srgb(result)

            return (result,)


class C2CColorSpaceConvert:
    """Convert between color spaces (sRGB, Linear, Log).

    Professional workflows need to move between color spaces:
    - sRGB → Linear for compositing / blending
    - Linear → sRGB for display
    - Linear → Log C3 for color grading (DaVinci Resolve)
    - Log C3 → Linear for returning to pipeline
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_space": (["sRGB", "Linear", "Log C3"], {
                    "default": "sRGB",
                }),
                "target_space": (["sRGB", "Linear", "Log C3"], {
                    "default": "Linear",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "MEC/Color Science"

    @classmethod
    def IS_CHANGED(cls, image, source_space, target_space, **kwargs):
        return hash_args_and_kwargs(image, source_space, target_space, **kwargs)

    def convert(self, image, source_space, target_space):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("C2CColorSpaceConvert expects IMAGE tensor [B,H,W,C]")
        if source_space == target_space:
            return (image,)

        with torch.inference_mode():
            x = image.clone().float()

            # To linear first
            if source_space == "sRGB":
                x = _srgb_to_linear(x)
            elif source_space == "Log C3":
                x = _logc3_to_linear(x)

            # From linear to target
            if target_space == "sRGB":
                x = _linear_to_srgb(x)
            elif target_space == "Log C3":
                x = _linear_to_logc3(x)

            return (x.clamp(0.0, 1.0),)


# ── Color space conversion helpers ────────────────────────────────────

def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    low = x * 12.92
    high = 1.055 * x.clamp(min=1e-6).pow(1.0 / 2.4) - 0.055
    return torch.where(x <= 0.0031308, low, high)


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    low = x / 12.92
    high = ((x + 0.055) / 1.055).clamp(min=1e-6).pow(2.4)
    return torch.where(x <= 0.04045, low, high)


def _linear_to_logc3(x: torch.Tensor) -> torch.Tensor:
    """ARRI LogC3 (EI 800) encode."""
    cut = 0.010591
    a, b, c, d, e, f = 5.555556, 0.052272, 0.247190, 0.385537, -0.052272, 5.367655
    low = e * x + f
    high = c * torch.log10(a * x.clamp(min=1e-10) + b) + d
    return torch.where(x < cut, low, high)


def _logc3_to_linear(x: torch.Tensor) -> torch.Tensor:
    """ARRI LogC3 (EI 800) decode."""
    cut_log = 0.010591
    a, b, c, d, e, f = 5.555556, 0.052272, 0.247190, 0.385537, -0.052272, 5.367655
    cut_logc = c * math.log10(a * cut_log + b) + d
    low = (x - f) / e
    high = (10.0 ** ((x - d) / c) - b) / a
    return torch.where(x < cut_logc, low, high)


NODE_CLASS_MAPPINGS = {
    "C2CACESTonemap":        C2CACESTonemap,
    "C2CVAEQualityDecode":   C2CVAEQualityDecode,
    "C2CColorSpaceConvert":  C2CColorSpaceConvert,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "C2CACESTonemap":        "C2C ACES Tonemap",
    "C2CVAEQualityDecode":   "C2C VAE Quality Decode (HDR)",
    "C2CColorSpaceConvert":  "C2C Color Space Convert",
}
