# PORTED FROM: ComfyUI-WanNodeExperiments/nodes/wan_director/features/_local_vae_hdr.py
#   (Apache-2.0, same author - Code2Collapse; WNE @ 88ca5f3). Ported for L7.39: CNP's
#   C2CVAEQualityDecode imported it from .wan_director, which left CNP when WanDirector moved to WNE (D1.03),
#   so its spatial-tiled decode always failed and silently fell back to a plain decode.
"""Spatial-only VAE decode for Wan video latents.

Measured (L7.41 M1): forcing fp32 adds at most 0.17 dB to VAE round-trip on Flux
and Wan 2.1; half-precision on its own is ~0.15 of an 8-bit level on average.
This module tiles only H/W (all frames per tile) to avoid temporal flicker.
Clamping is core's process_output; callers choose clamp_output on the node.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F

log = logging.getLogger("MEC.VAE_HDR")


def _is_vae_dynamic(vae) -> bool:
    if hasattr(vae, "is_dynamic") and callable(vae.is_dynamic):
        return bool(vae.is_dynamic())
    patcher = getattr(vae, "patcher", None)
    if patcher is not None and hasattr(patcher, "is_dynamic") and callable(patcher.is_dynamic):
        return bool(patcher.is_dynamic())
    return False


@contextmanager
def vae_compute_dtype(vae, dtype: torch.dtype) -> Iterator[None]:
    """Set vae.vae_dtype for one decode; cast legacy (non-dynamic) weights."""
    if getattr(vae, "vae_dtype", None) == dtype:
        yield
        return

    prev_vae_dtype = getattr(vae, "vae_dtype", None)
    prev_weight_dtype = None
    cast_weights = (
        not _is_vae_dynamic(vae)
        and hasattr(vae, "first_stage_model")
    )
    try:
        vae.vae_dtype = dtype
        if cast_weights:
            prev_weight_dtype = next(vae.first_stage_model.parameters()).dtype
            if prev_weight_dtype != dtype:
                vae.first_stage_model.to(dtype=dtype)
        yield
    finally:
        if prev_vae_dtype is not None or hasattr(vae, "vae_dtype"):
            vae.vae_dtype = prev_vae_dtype
        if prev_weight_dtype is not None and hasattr(vae, "first_stage_model"):
            try:
                vae.first_stage_model.to(dtype=prev_weight_dtype)
            except (AttributeError, RuntimeError):
                pass


def standard_output_mapping(vae) -> bool:
    """True when process_output is core's (x+1)/2 then clamp mapping."""
    try:
        po = vae.process_output
        if po(torch.tensor([-3.0])) != 0.0:
            return False
        if po(torch.tensor([0.0])) != 0.5:
            return False
        if po(torch.tensor([3.0])) != 1.0:
            return False
        return True
    except Exception:
        return False


@contextmanager
def unclamped_output(vae) -> Iterator[None]:
    """Replace process_output with unclamped (x+1)/2 when the probe passes."""
    if not standard_output_mapping(vae):
        log.info(
            "VAE process_output is not core's standard mapping; "
            "clamp_output=False has no effect on this VAE."
        )
        yield
        return

    orig = vae.process_output
    try:
        vae.process_output = lambda t: t.add_(1.0).div_(2.0)
        yield
    finally:
        vae.process_output = orig


def _normalize_decoded(decoded: torch.Tensor) -> torch.Tensor:
    """Return video decode as [B, T, H, W, C]. Image decode [B,H,W,C] -> T=1."""
    if decoded.ndim == 4:
        return decoded.unsqueeze(1)
    if decoded.ndim == 5:
        return decoded
    raise ValueError(f"VAE decode returned unexpected shape {tuple(decoded.shape)}")


def _standard_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents; no dtype juggling or output clamp (caller owns both)."""
    decoded = vae.decode(latents)
    if isinstance(decoded, dict):
        decoded = decoded.get("samples", decoded.get("sample", next(iter(decoded.values()))))
    if not isinstance(decoded, torch.Tensor):
        raise TypeError(f"VAE decode returned {type(decoded).__name__}, expected a tensor.")
    return _normalize_decoded(decoded.float())


def _plan_tile_spans(total: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    """Plan tile [start, end) spans along one spatial axis; always terminates."""
    tile_size = max(1, tile_size)
    overlap = min(max(0, overlap), (tile_size - 1) // 2)
    if total <= tile_size:
        return [(0, total)]

    stride = tile_size - overlap
    if stride < 1:
        raise ValueError(
            f"tile overlap {overlap} is too large for tile size {tile_size} "
            f"(stride would be {stride})"
        )

    spans: list[tuple[int, int]] = []
    pos = 0
    while pos < total:
        end = min(pos + tile_size, total)
        spans.append((pos, end))
        if end == total:
            break
        pos += stride
    return spans


def decode_wan_spatial_tiled(
    vae,
    latents: torch.Tensor,
    tile_latent: int,
    overlap_latent: int,
) -> torch.Tensor:
    """Decode Wan latents [B,C,T,H,W] with spatial-only tiling.

    Returns decoded pixels [B, T_out, H_out, W_out, 3]. Frame count comes from
    the first decoded tile (Wan: 1+4*(T_latent-1), not T*4+1).
    """
    if latents.ndim != 5:
        log.warning(
            "Expected 5D latent [B,C,T,H,W], got %dD; using standard decode.",
            latents.ndim,
        )
        return _standard_decode(vae, latents)

    B, _C, T, H, W = latents.shape
    factor = vae.spacial_compression_decode() if hasattr(vae, "spacial_compression_decode") else 8
    tile_latent = max(1, tile_latent)
    eff = min(max(0, overlap_latent), (tile_latent - 1) // 2)

    if H <= tile_latent and W <= tile_latent:
        return _standard_decode(vae, latents)

    h_tiles = _plan_tile_spans(H, tile_latent, eff)
    w_tiles = _plan_tile_spans(W, tile_latent, eff)
    out_H = H * factor
    out_W = W * factor
    overlap_px = eff * factor

    log.info(
        "Spatial-tiled decode: %dx%d latent -> %dx%d tiles (%d total), "
        "overlap=%d latent, %d latent frames",
        H, W, len(h_tiles), len(w_tiles), len(h_tiles) * len(w_tiles),
        eff, T,
    )

    h0, h1 = h_tiles[0]
    w0, w1 = w_tiles[0]
    first = _standard_decode(vae, latents[:, :, :, h0:h1, w0:w1])
    T_out = first.shape[1]
    dtype = first.dtype
    device = first.device

    output = torch.zeros((B, T_out, out_H, out_W, 3), dtype=dtype, device=device)
    weight = torch.zeros((1, 1, out_H, out_W, 1), dtype=dtype, device=device)

    def _accumulate(tile_decoded, h_start, h_end, w_start, w_end):
        oh_start = h_start * factor
        oh_end = h_end * factor
        ow_start = w_start * factor
        ow_end = w_end * factor
        mask = _build_feather_mask(
            oh_end - oh_start, ow_end - ow_start, overlap_px,
            h_start == 0, h_end == H, w_start == 0, w_end == W,
            dtype, device,
        )
        n_frames = min(tile_decoded.shape[1], T_out)
        output[:, :n_frames, oh_start:oh_end, ow_start:ow_end, :] += (
            tile_decoded[:, :n_frames] * mask
        )
        weight[:, :, oh_start:oh_end, ow_start:ow_end, :] += mask[:1, :1]

    _accumulate(first, h0, h1, w0, w1)

    for h_start, h_end in h_tiles:
        for w_start, w_end in w_tiles:
            if h_start == h0 and w_start == w0:
                continue
            tile_lat = latents[:, :, :, h_start:h_end, w_start:w_end]
            tile_decoded = _standard_decode(vae, tile_lat)
            _accumulate(tile_decoded, h_start, h_end, w_start, w_end)

    weight = weight.clamp(min=1e-6)
    return output / weight


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
