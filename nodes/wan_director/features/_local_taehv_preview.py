"""Local TAEHV-preview fallback (latent → RGB preview).

TAEHV (Tiny AutoEncoder for HunYuan Video) is a small distilled VAE
that produces fast RGB previews from video latents. The real TAEHV
needs a small weight file (~5 MB) that we may not have locally; this
module provides a **deterministic, weight-free RGB preview** as a
fallback so the Director's live-preview can still render *something*
useful for the user.

The fallback is a fixed linear map from N channels to 3 (RGB):
    rgb[..., c] = sum_i  weights[c, i] * latent[..., i]
where ``weights`` is a deterministic per-channel projection derived
from a fixed seed. Then we tonemap to [0, 1] using a robust
percentile normalisation so each preview is roughly aligned in
brightness regardless of latent stats.

Pure-tensor, no weights, no IO.
"""
from __future__ import annotations

import torch


def _make_projection(n_in: int, *, seed: int = 1729,
                     device, dtype) -> torch.Tensor:
    """Build a deterministic (3, n_in) projection matrix."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    W = torch.randn(3, n_in, generator=g, dtype=torch.float32)
    # Centre + normalise rows so each output channel has unit gain.
    W = W - W.mean(dim=1, keepdim=True)
    W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return W.to(device=device, dtype=dtype)


def latent_to_rgb_preview(
    latent: torch.Tensor,
    *,
    seed:       int   = 1729,
    percentile: float = 0.02,
) -> torch.Tensor:
    """Convert a latent tensor to an RGB preview tensor.

    Args:
        latent: shape ``(B, C, T, H, W)`` (Wan-style) or
                ``(B, C, H, W)`` (per-frame). Channel dim = 1.
        seed:   PRNG seed for the deterministic projection matrix.
        percentile: tonemap clip (e.g., 0.02 → 2% / 98% percentiles).

    Returns:
        Tensor with the channel dim mapped to 3 (RGB) and values in
        ``[0, 1]``. Shape: same as input but with C=3.
    """
    if latent.dim() not in (4, 5) or latent.shape[1] < 1:
        raise ValueError(
            f"latent_to_rgb_preview: expected (B,C,...) tensor with C>=1; "
            f"got shape {tuple(latent.shape)}"
        )
    n_in = latent.shape[1]
    W = _make_projection(n_in, seed=seed,
                         device=latent.device, dtype=latent.dtype)
    # einsum over the channel dim.
    if latent.dim() == 5:
        rgb = torch.einsum("ci,bithw->bcthw", W, latent)
    else:  # dim() == 4
        rgb = torch.einsum("ci,bihw->bchw", W, latent)
    # Robust per-batch tonemap to [0, 1].
    flat = rgb.reshape(rgb.shape[0], -1)
    lo = torch.quantile(flat, percentile,        dim=1, keepdim=True)
    hi = torch.quantile(flat, 1.0 - percentile,  dim=1, keepdim=True)
    rng = (hi - lo).clamp_min(1e-6)
    flat = (flat - lo) / rng
    rgb = flat.reshape(rgb.shape).clamp_(0.0, 1.0)
    return rgb
