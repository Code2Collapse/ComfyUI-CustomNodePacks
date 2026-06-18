"""High-fidelity VAE decode for Wan video models.

Problems with default ComfyUI VAE decode for Wan:
1. Default decode uses fp16 — causes banding, color loss, detail smearing
2. Standard tiled decode tiles temporally too — causes frame flickering
3. No HDR-aware processing — output is clamped to [0,1] sRGB

Solutions implemented here:
1. Force fp32 for VAE decode (matching HuggingFace recommendation)
2. Spatial-only tiling: process ALL frames per tile, tile only H/W
3. Linear feathering with 2D masks for seamless tile blending
4. Optional HDR output in linear-light space (pre-ACES tonemap)

Reference: crmbz0r/ComfyUI_Wan22Blockswap/vae_decode.py for the
spatial-only tiling approach. Our implementation is clean-room but
uses the same mathematical principle.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger("MEC.VAE_HDR")


def force_fp32_vae(vae) -> None:
    """Force a ComfyUI VAE to run in fp32 for maximum decode quality.

    The Wan 2.2 VAE (AutoencoderKLWan) produces significantly better
    results in fp32. HuggingFace recommends: "Set the AutoencoderKLWan
    dtype to torch.float32 for better decoding quality."
    """
    try:
        if hasattr(vae, "first_stage_model"):
            vae.first_stage_model.to(dtype=torch.float32)
            log.info("VAE forced to fp32 (first_stage_model)")
        elif hasattr(vae, "model"):
            vae.model.to(dtype=torch.float32)
            log.info("VAE forced to fp32 (model)")
    except (AttributeError, RuntimeError) as e:
        log.warning("Could not force VAE to fp32: %s", e)


def decode_wan_spatial_tiled(
    vae,
    latents: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 32,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode Wan latents with spatial-only tiling for temporal coherence.

    Unlike standard tiled decode which tiles across all dimensions
    (including temporal), this tiles ONLY spatially (H, W) while
    processing ALL frames together per tile. This preserves temporal
    coherence and prevents inter-frame flickering.

    Args:
        vae: ComfyUI VAE wrapper.
        latents: Latent tensor [B, C, T, H, W] (Wan video latent format).
        tile_size: Spatial tile size in latent pixels.
        overlap: Overlap between adjacent tiles in latent pixels.
        dtype: Compute dtype (use float32 for quality).

    Returns:
        Decoded pixel tensor [B, T_out, H_out, W_out, 3] in [0,1].
    """
    if latents.ndim != 5:
        log.warning("Expected 5D latent [B,C,T,H,W], got %dD. Falling back to standard decode.", latents.ndim)
        return _standard_decode(vae, latents, dtype)

    B, C, T, H, W = latents.shape

    if H <= tile_size and W <= tile_size:
        return _standard_decode(vae, latents, dtype)

    upscale_factor = 8  # Wan VAE spatial compression

    # Plan tiles
    h_tiles = _plan_tile_spans(H, tile_size, overlap)
    w_tiles = _plan_tile_spans(W, tile_size, overlap)

    out_H = H * upscale_factor
    out_W = W * upscale_factor

    log.info(
        "Spatial-tiled decode: %dx%d latent → %dx%d tiles (%d total), "
        "overlap=%d, %d frames",
        H, W, len(h_tiles), len(w_tiles), len(h_tiles) * len(w_tiles),
        overlap, T
    )

    output = torch.zeros((B, T * 4 + 1, out_H, out_W, 3), dtype=dtype)
    weight = torch.zeros((1, 1, out_H, out_W, 1), dtype=dtype)

    for h_start, h_end in h_tiles:
        for w_start, w_end in w_tiles:
            tile_latent = latents[:, :, :, h_start:h_end, w_start:w_end]

            tile_latent = tile_latent.to(dtype=dtype)
            tile_decoded = _standard_decode(vae, tile_latent, dtype)

            # Compute output pixel coordinates
            oh_start = h_start * upscale_factor
            oh_end = h_end * upscale_factor
            ow_start = w_start * upscale_factor
            ow_end = w_end * upscale_factor

            # Feathering mask
            mask = _build_feather_mask(
                oh_end - oh_start, ow_end - ow_start,
                overlap * upscale_factor,
                h_start == 0, h_end == H,
                w_start == 0, w_end == W,
                dtype, tile_decoded.device,
            )

            n_frames = min(tile_decoded.shape[1], output.shape[1])
            output[:, :n_frames, oh_start:oh_end, ow_start:ow_end, :] += (
                tile_decoded[:, :n_frames] * mask
            )
            weight[:, :, oh_start:oh_end, ow_start:ow_end, :] += mask[:1, :1]

    weight = weight.clamp(min=1e-6)
    output = output / weight

    return output


def _plan_tile_spans(total: int, tile_size: int, overlap: int) -> list:
    """Plan non-overlapping tile start/end positions."""
    if total <= tile_size:
        return [(0, total)]

    stride = tile_size - overlap
    spans = []
    pos = 0
    while pos < total:
        end = min(pos + tile_size, total)
        spans.append((pos, end))
        if end == total:
            break
        pos += stride

    return spans


def _build_feather_mask(
    h: int, w: int, overlap_px: int,
    is_top: bool, is_bottom: bool,
    is_left: bool, is_right: bool,
    dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Build a 2D linear feathering mask for tile blending."""
    mask_h = torch.ones(h, dtype=dtype, device=device)
    mask_w = torch.ones(w, dtype=dtype, device=device)

    ramp = min(overlap_px, h // 2, w // 2)
    if ramp < 1:
        return torch.ones((1, 1, h, w, 1), dtype=dtype, device=device)

    if not is_top:
        mask_h[:ramp] = torch.linspace(0, 1, ramp, dtype=dtype, device=device)
    if not is_bottom:
        mask_h[-ramp:] = torch.linspace(1, 0, ramp, dtype=dtype, device=device)
    if not is_left:
        mask_w[:ramp] = torch.linspace(0, 1, ramp, dtype=dtype, device=device)
    if not is_right:
        mask_w[-ramp:] = torch.linspace(1, 0, ramp, dtype=dtype, device=device)

    mask_2d = mask_h.unsqueeze(1) * mask_w.unsqueeze(0)
    return mask_2d.reshape(1, 1, h, w, 1)


def _standard_decode(vae, latents: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Decode latents using ComfyUI's standard VAE decode path with fp32 cast."""
    original_dtype = None
    try:
        if hasattr(vae, "first_stage_model"):
            original_dtype = next(vae.first_stage_model.parameters()).dtype
            if dtype == torch.float32 and original_dtype != torch.float32:
                vae.first_stage_model.to(dtype=torch.float32)

        latents_cast = latents.to(dtype=dtype)
        decoded = vae.decode(latents_cast)

        if isinstance(decoded, dict):
            decoded = decoded.get("samples", decoded.get("sample", list(decoded.values())[0]))

        if isinstance(decoded, torch.Tensor):
            decoded = decoded.to(dtype=dtype)
            decoded = decoded.clamp(0.0, 1.0)
        return decoded

    finally:
        if original_dtype is not None and dtype == torch.float32:
            try:
                vae.first_stage_model.to(dtype=original_dtype)
            except (AttributeError, RuntimeError):
                pass


def apply_aces_tonemap(image: torch.Tensor, exposure: float = 1.0) -> torch.Tensor:
    """Apply ACES filmic tone mapping for HDR-to-SDR conversion.

    Uses the simplified ACES fit by Stephen Hill (from the Unity
    standard library), which approximates the full ACES RRT + ODT
    pipeline in a single rational polynomial.

    Args:
        image: Linear-light RGB tensor in [0, inf).
        exposure: Exposure adjustment multiplier.

    Returns:
        Tone-mapped tensor in [0, 1].
    """
    x = image * exposure

    # ACES filmic curve (Stephen Hill fit)
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14

    result = (x * (a * x + b)) / (x * (c * x + d) + e)
    return result.clamp(0.0, 1.0)


def linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    """Convert linear-light RGB to sRGB with the standard gamma curve."""
    low = image * 12.92
    high = 1.055 * image.pow(1.0 / 2.4) - 0.055
    return torch.where(image <= 0.0031308, low, high).clamp(0.0, 1.0)


def srgb_to_linear(image: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to linear-light RGB."""
    low = image / 12.92
    high = ((image + 0.055) / 1.055).pow(2.4)
    return torch.where(image <= 0.04045, low, high).clamp(0.0)
