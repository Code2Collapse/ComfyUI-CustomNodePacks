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
        with torch.no_grad():
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
            result = _aces_fit(x)

            # Output colorspace
            if output_colorspace == "sRGB (gamma)":
                result = _linear_to_srgb(result)
            elif output_colorspace == "Linear":
                pass  # already linear after ACES
            # ACES AP1 stays as-is (ACES output)

            return (result,)


class C2CVAEQualityDecode:
    """VAE decode with optional fp32, spatial-only tiling, and unclamped output.

    Measured (L7.41 M1): fp32 adds at most 0.17 dB to round-trip on Flux/Wan 2.1
    at roughly 2x VAE memory. Spatial tiling preserves temporal coherence.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "force_fp32": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Decode in fp32. Measured gain: at most 0.17 dB on "
                               "Flux/Wan 2.1 round-trip (~2x VAE memory).",
                }),
                "tile_size": ("INT", {
                    "default": 0, "min": 0, "max": 1024, "step": 64,
                    "tooltip": "Spatial tile size in PIXELS, used when tile_mode is manual (0 = no tiling). "
                               "Converted with the VAE spatial factor (usually 8). 512 measured best for Wan on 8 GB.",
                }),
                "apply_aces": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply ACES filmic tone mapping after decode (the decode is treated as sRGB: "
                               "linearised, tone-mapped, encoded once - same as C2C ACES Tonemap from sRGB).",
                }),
                "exposure": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 10.0, "step": 0.05,
                    "tooltip": "Exposure for ACES (only used if apply_aces=True).",
                }),
            },
            "optional": {
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Core clamps decoder output to 0..1. Off keeps "
                               "out-of-range values (4.6x / 5.3x more accurate "
                               "there on Flux/Wan, measured M2). For HDR/EXR chains.",
                }),
                "tile_mode": (["auto", "off", "manual"], {
                    "default": "auto",
                    "tooltip": "auto: tile only when the untiled decode would not fit the GPU (measured on an 8 GB "
                               "card: a 2K Wan decode took 502 s untiled, spilling into system RAM, and 37 s tiled, "
                               "with the same picture). off: never tile. manual: tile at tile_size. Video is tiled "
                               "in space only - splitting a video VAE in time costs ~21 dB.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "MEC/Color Science"
    DESCRIPTION = (
        "VAE decode with optional fp32 (<=0.17 dB measured gain), spatial-only "
        "tiling for Wan video, optional unclamped output for HDR chains, and "
        "optional ACES tone mapping."
    )

    @classmethod
    def IS_CHANGED(cls, samples, vae, force_fp32, tile_size, apply_aces, exposure, **kwargs):
        return hash_args_and_kwargs(
            samples, vae, force_fp32, tile_size, apply_aces, exposure, **kwargs,
        )

    @staticmethod
    def _is_dtype_mismatch_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "should be the same" in msg
            or "expected scalar type" in msg
            or "dtype" in msg
        )

    @staticmethod
    def _is_vae_dynamic(vae) -> bool:
        if hasattr(vae, "is_dynamic") and callable(vae.is_dynamic):
            return bool(vae.is_dynamic())
        patcher = getattr(vae, "patcher", None)
        if patcher is not None and hasattr(patcher, "is_dynamic") and callable(patcher.is_dynamic):
            return bool(patcher.is_dynamic())
        return False

    def _run_decode(self, vae, latent, tile_size):
        from ._vae_tiled import decode_wan_spatial_tiled

        if latent.ndim == 5 and tile_size > 0:
            factor = (
                vae.spacial_compression_decode()
                if hasattr(vae, "spacial_compression_decode") else 8
            )
            tile_latent = max(1, tile_size // factor)
            # 25% overlap. Measured on Wan 2.1 (16-latent tiles vs a full decode): overlap 2 -> 42.9 dB,
            # 4 -> 48.4 dB (core decode_tiled at 4: 47.9), 7 -> 51.1 dB at ~3x the tiles
            # (docs/evidence/L7.41/tile_overlap_sweep.json). Larger tiles cut seams more cheaply.
            overlap_latent = max(1, tile_latent // 4)
            try:
                return decode_wan_spatial_tiled(vae, latent, tile_latent, overlap_latent)
            except Exception as exc:
                raise RuntimeError(
                    f"Spatial-tiled VAE decode failed ({exc}). "
                    "Try a larger tile_size, tile_size=0 for no tiling, "
                    "or check that the VAE supports 5D video latents."
                ) from exc
        if latent.ndim == 4 and tile_size > 0 and hasattr(vae, "decode_tiled"):
            factor = vae.spacial_compression_decode() if hasattr(vae, "spacial_compression_decode") else 8
            t = max(1, tile_size // factor)
            return vae.decode_tiled(latent, tile_x=t, tile_y=t, overlap=max(1, t // 4))
        return vae.decode(latent)

    @staticmethod
    def _auto_tile_px(vae, latent, use_fp32: bool) -> int:
        """0 when the untiled decode fits the GPU, else the largest tile (px) that does, by core's own estimate.

        On a CPU nothing spills, and the untiled decode is the reference: 0."""
        try:
            import comfy.model_management as mm

            dev = getattr(vae, "device", None)
            if dev is None or getattr(dev, "type", "cpu") == "cpu" or not hasattr(vae, "memory_used_decode"):
                return 0
            dtype = torch.float32 if use_fp32 else getattr(vae, "vae_dtype", torch.float16)
            budget = mm.get_free_memory(dev) * 0.8
            if vae.memory_used_decode(tuple(latent.shape), dtype) <= budget:
                return 0
            factor = vae.spacial_compression_decode() if hasattr(vae, "spacial_compression_decode") else 8
            for px in (1024, 768, 512, 384, 256):
                lat = max(1, px // factor)
                shape = list(latent.shape)
                shape[-1], shape[-2] = min(shape[-1], lat), min(shape[-2], lat)
                if vae.memory_used_decode(tuple(shape), dtype) <= budget:
                    return px
            return 256
        except Exception as exc:  # noqa: BLE001 - an estimate failing must not stop the decode
            log.debug("auto tile estimate failed: %s", exc)
            return 0

    def decode(self, samples, vae, force_fp32, tile_size, apply_aces, exposure,
               clamp_output=True, tile_mode="auto"):
        from contextlib import nullcontext

        from ._vae_tiled import fp32_vae_copy, unclamped_output, vae_compute_dtype

        with torch.no_grad():
            latent = samples["samples"]
            if tile_mode == "off":
                tile_size = 0
            elif tile_mode != "manual":                       # "auto", and anything unknown
                tile_size = self._auto_tile_px(vae, latent, bool(force_fp32))
            self._last_tile_px = tile_size

            def _decode_with_contexts(use_fp32: bool):
                target, dtype_ctx = vae, nullcontext()
                if use_fp32:
                    # a core VAE gets its own fp32 copy (in-place casting broke core's pinned weights on a GPU);
                    # anything else (dynamic VAEs, stand-ins) keeps the old cast-for-one-decode
                    copy = None if self._is_vae_dynamic(vae) else fp32_vae_copy(vae)
                    if copy is not None:
                        target = copy
                    else:
                        dtype_ctx = vae_compute_dtype(vae, torch.float32)
                unclamp_ctx = unclamped_output(target) if not clamp_output else nullcontext()
                with dtype_ctx, unclamp_ctx:
                    return self._run_decode(target, latent, tile_size)

            try:
                result = _decode_with_contexts(force_fp32)
            except RuntimeError as exc:
                if (
                    force_fp32
                    and self._is_vae_dynamic(vae)
                    and self._is_dtype_mismatch_error(exc)
                ):
                    log.warning(
                        "fp32 decode failed on a dynamic VAE (%s); "
                        "retrying at the VAE's native dtype.",
                        exc,
                    )
                    result = _decode_with_contexts(False)
                else:
                    raise

            if isinstance(result, dict):
                result = result.get(
                    "samples", result.get("sample", next(iter(result.values())))
                )

            if isinstance(result, torch.Tensor) and result.ndim == 5:
                result = result.reshape(-1, *result.shape[-3:])

            src_frames = samples.get("c2c_source_frames")
            if (
                isinstance(src_frames, int)
                and latent.shape[0] == 1
                and isinstance(result, torch.Tensor)
                and result.shape[0] > src_frames
            ):
                result = result[:src_frames]

            result = result.float()
            if clamp_output:
                result = result.clamp(0.0, 1.0)

            if apply_aces:
                # The decode is sRGB-ENCODED (display-referred), so linearise before the curve and encode once after
                # it. Tone-mapping the encoded values as if linear, then encoding again, applied the gamma twice: a
                # lifted, washed-out image (L7.42). Same result as C2C ACES Tonemap with source_space = sRGB.
                result = _linear_to_srgb(_aces_fit(_srgb_to_linear(result) * exposure))

            return (result,)


class C2CColorSpaceConvert:
    """Convert between color spaces (sRGB, Linear, Log).

    Professional workflows need to move between color spaces:
    - sRGB → Linear for compositing / blending
    - Linear → sRGB for display
    - Linear → Log C3 for color grading (DaVinci Resolve)
    - Log C3 → Linear for returning to pipeline
    """

    DESCRIPTION = (
        "Move an image between sRGB, scene-linear and ARRI Log C3. Compositing "
        "and blending are only correct in LINEAR - adding two sRGB images "
        "together adds their display curves as well as their light, which is "
        "why a screen blend in sRGB looks wrong. Convert in, work, convert "
        "back out. Log C3 is for handing off to a grade."
    )

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

        with torch.no_grad():
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

def _aces_fit(x: torch.Tensor) -> torch.Tensor:
    """Narkowicz's fit of the ACES filmic curve, scene-linear in, display-linear 0..1 out."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)


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
