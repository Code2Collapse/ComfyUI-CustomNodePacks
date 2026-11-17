"""HDR-aware VAE decode for Wan video outputs.

Registered node: C2CVAEQualityDecode (fp32, spatial tiling, optional ACES on decode).

Tone map and colour-space convert nodes (C2CACESTonemap, C2CColorSpaceConvert) were
removed in L7.65; use ComfyUI-NukeMaxNodes HDR Tone Map and Color Space Convert instead.
"""
from __future__ import annotations

import logging

import torch

from ._is_changed_util import hash_args_and_kwargs
try:                                                       # A28 layout declarations (display only)
    from ._c2c_layout import with_layout
except ImportError:  # loaded as a plain file (tests): the same module by path, never an absolute import
    try:
        import importlib.util as _c2c_ilu
        import os as _c2c_os
        _c2c_spec = _c2c_ilu.spec_from_file_location("c2c_layout_by_path", _c2c_os.path.join(
            _c2c_os.path.dirname(_c2c_os.path.abspath(__file__)), '_c2c_layout.py'))
        _c2c_mod = _c2c_ilu.module_from_spec(_c2c_spec)
        _c2c_spec.loader.exec_module(_c2c_mod)
        with_layout = _c2c_mod.with_layout
    except Exception:  # pragma: no cover - layout keys are display only: never take the node down
        def with_layout(spec, layout):
            return spec

log = logging.getLogger("MEC.HDRColor")



# A28 (docs/audit/A28/PARAM_CNP_B.md, C2CVAEQualityDecode): which controls show when. Display only. Replaces the
# c2c_mode_widgets.js rule (one visibility owner per node). With `image` wired the decode is skipped, so the decode
# controls hide then. tile_size also shows in "auto" while it is above 0: _decode uses a positive size as given there
# (L9.263, saves from before tile_mode) - the old rule hid it although it was read.
_NOT_IMAGE = {"c2c_hide_if_linked": ["image"]}
_CLEAN_ON = {"clean": [True]}
_VQD_LAYOUT = {
    "force_fp32": {"c2c_section": "Decode", **_NOT_IMAGE},
    "tile_size": {"c2c_section": "Decode", **_NOT_IMAGE, "c2c_show_if_any": [
        {"tile_mode": ["manual"]}, {"tile_mode": ["auto"], "tile_size": {"gt": 0}}]},
    "exposure": {"c2c_show_if": {"apply_aces": [True]}},
    "clamp_output": {"c2c_section": "Decode", **_NOT_IMAGE},
    "tile_mode": {"c2c_section": "Decode", **_NOT_IMAGE},
    "balance_mode": {"c2c_section": "Clean", "c2c_show_if": _CLEAN_ON},
    "balance_strength": {"c2c_section": "Clean", "c2c_show_if": {**_CLEAN_ON, "balance_mode": {"not": ["off"]}}},
    "saturation": {"c2c_section": "Clean", "c2c_show_if": _CLEAN_ON},
    "contrast_restore": {"c2c_section": "Clean", "c2c_show_if": _CLEAN_ON},
    "chroma_cleanup": {"c2c_section": "Clean", "c2c_show_if": _CLEAN_ON},
    "restore_unchanged": {"c2c_section": "Clean", "c2c_show_if": _CLEAN_ON},
    "restore_radius": {"c2c_section": "Clean", "c2c_show_if": {**_CLEAN_ON, "restore_unchanged": {"gt": 0}}},
}


class C2CVAEQualityDecode:
    """VAE Decode (C2C): decode with optional fp32, spatial-only tiling and unclamped output, then optionally clean
    the colour (cast, saturation, crushed shadows, chroma speckle - the former VAE Clean node, merged in L7.65 P15;
    its saved workflows migrate here with an image wired instead of a latent).

    Measured (L7.41 M1): fp32 adds at most 0.17 dB to round-trip on Flux/Wan 2.1
    at roughly 2x VAE memory. Spatial tiling preserves temporal coherence.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .vae_clean import VAECleanMEC
        clean = VAECleanMEC.INPUT_TYPES()
        clean_req, clean_opt = clean["required"], clean["optional"]

        def _clean_tip(name, spec):
            t, o = spec[0], dict(spec[1]) if len(spec) > 1 else {}
            o["tooltip"] = "Clean (when clean is on): " + o.get("tooltip", "")
            return (t, o)

        return with_layout({
            "required": {
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
                               "linearised, tone-mapped with the ACES fit, encoded once).",
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
                               "with the same picture); a tile_size above 0 is used as given. off: never tile. manual: tile at "
                               "tile_size. Video is tiled "
                               "in space only - splitting a video VAE in time costs ~21 dB.",
                }),
                # sockets were required before L7.65; optional now so an already-decoded image can be cleaned
                "samples": ("LATENT", {"tooltip": "The latent to decode (with vae). Leave empty to clean 'image'."}),
                "vae": ("VAE",),
                "image": ("IMAGE", {"tooltip": "An already-decoded picture to clean instead of decoding a latent."}),
                # appended after tile_mode so saved VAE Quality Decode values keep their positions
                "clean": ("BOOLEAN", {"default": False,
                          "tooltip": "Measure the decode and correct what is wrong with it: colour cast, "
                                     "oversaturation, crushed shadows, chroma speckle. The report says what it "
                                     "found, including clipping it cannot undo."}),
                **{k: _clean_tip(k, v) for k, v in clean_req.items() if k != "image"},
                **{k: _clean_tip(k, v) for k, v in clean_opt.items()},
            },
        }, _VQD_LAYOUT)

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("IMAGE", "report", "cast_strength", "saturation")
    OUTPUT_TOOLTIPS = (
        "The decoded (and, with clean on, corrected) picture.",
        "Clean: what was measured and corrected (empty when clean is off).",
        "Clean: how strong the colour cast was.",
        "Clean: measured saturation.",
    )
    FUNCTION = "decode"
    CATEGORY = "MEC/Color Science"
    DESCRIPTION = (
        "VAE decode with optional fp32 (<=0.17 dB measured gain), spatial-only "
        "tiling for Wan video, optional unclamped output for HDR chains, "
        "optional ACES tone mapping, and an optional clean step that measures "
        "and corrects colour cast, oversaturation and decode speckle (also for "
        "an already-decoded image)."
    )

    @classmethod
    def IS_CHANGED(cls, samples=None, vae=None, force_fp32=True, tile_size=0, apply_aces=False, exposure=1.0,
                   **kwargs):
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

    def decode(self, samples=None, vae=None, force_fp32=True, tile_size=0, apply_aces=False, exposure=1.0,
               clamp_output=True, tile_mode="auto", image=None, clean=False, reference=None, **clean_args):
        if image is not None:
            if samples is not None:
                raise ValueError("VAE Decode (C2C): wire a latent (with its VAE) OR an image to clean - not both.")
            result = image.float()
            if apply_aces:
                result = _linear_to_srgb(_aces_fit(_srgb_to_linear(result) * exposure))
            if not clean:
                return (result, "", 0.0, 0.0)
            return self._clean(result, reference, clean_args)
        if samples is None or vae is None:
            raise ValueError("VAE Decode (C2C): connect a latent and its VAE, or an image to clean.")
        if not clean:
            return (*self._decode(samples, vae, force_fp32, tile_size, apply_aces, exposure, clamp_output,
                                  tile_mode), "", 0.0, 0.0)
        (decoded,) = self._decode(samples, vae, force_fp32, tile_size, False, exposure, clamp_output, tile_mode)
        image_out, report, cast, sat = self._clean(decoded, reference, clean_args)
        if apply_aces:                                   # clean measures the decode itself, then the curve
            image_out = _linear_to_srgb(_aces_fit(_srgb_to_linear(image_out) * exposure))
        return (image_out, report, cast, sat)

    @staticmethod
    def _clean(image, reference, clean_args):
        from .vae_clean import VAECleanMEC
        spec = VAECleanMEC.INPUT_TYPES()
        args = {k: v[1].get("default") for k, v in {**spec["required"], **spec["optional"]}.items()
                if k not in ("image", "reference") and len(v) > 1}
        args.update({k: v for k, v in clean_args.items() if k in args})
        return VAECleanMEC().clean(image=image, reference=reference, **args)

    def _decode(self, samples, vae, force_fp32, tile_size, apply_aces, exposure, clamp_output=True,
                tile_mode="auto"):
        from contextlib import nullcontext

        from ._vae_tiled import fp32_vae_copy, unclamped_output, vae_compute_dtype

        with torch.no_grad():
            latent = samples["samples"]
            if tile_mode == "off":
                tile_size = 0
            elif tile_mode != "manual" and int(tile_size) <= 0:   # "auto" (and anything unknown) with no size set
                # L9.263: a tile_size above 0 is the user's (saves from before tile_mode existed set it and now get
                # "auto" for the appended widget) - it is used as given instead of being replaced
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
                # lifted, washed-out image (L7.42). (Same result the removed C2C ACES Tonemap gave from sRGB.)
                result = _linear_to_srgb(_aces_fit(_srgb_to_linear(result) * exposure))

            return (result,)


# ── Color space helpers (VAE decode ACES path) ─────────────────────────

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


NODE_CLASS_MAPPINGS = {
    "C2CVAEQualityDecode":   C2CVAEQualityDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "C2CVAEQualityDecode":   "VAE Decode (C2C)",
}
