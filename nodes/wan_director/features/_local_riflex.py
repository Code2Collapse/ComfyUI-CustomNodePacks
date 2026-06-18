"""Local re-implementation of RIFLEx (RoPE Interpolation Frequency Extension).

Paper: arXiv:2502.15894 (He et al., 2025) "RIFLEx: A Free Lunch for
Length Extrapolation in Video Diffusion Transformers".

Idea: a video DiT trained on N frames uses RoPE (rotary position
embeddings) whose frequencies are computed for that exact length. To
extend to N' > N frames at inference WITHOUT retraining, RIFLEx
rescales the lowest-frequency band of the temporal RoPE so the model
treats N' frames as if they spanned the same phase range as N frames.

Pure-tensor primitive: given a vector of RoPE frequencies (one per
dim), produce a rescaled vector for the target length. Optionally
rescale only the **lowest-k** frequencies (the "long-period" ones)
which carry the global temporal structure.

API:
    rescale_rope_freqs(freqs, *, source_len, target_len, k=2) → Tensor
"""
from __future__ import annotations

import torch


def rescale_rope_freqs(
    freqs: torch.Tensor,
    *,
    source_len: int,
    target_len: int,
    k: int = 2,
) -> torch.Tensor:
    """Rescale the lowest-``k`` RoPE frequencies for length extrapolation.

    Args:
        freqs:      1-D tensor of RoPE frequencies (smaller=lower freq).
        source_len: frame count the model was trained on.
        target_len: frame count we want to run at inference.
        k:          how many of the lowest-frequency dims to rescale.

    Returns:
        New 1-D tensor of the same shape with the lowest ``k`` entries
        multiplied by ``source_len / target_len``.
    """
    if freqs.dim() != 1:
        raise ValueError("rescale_rope_freqs: expected 1-D tensor")
    if source_len <= 0 or target_len <= 0:
        raise ValueError("rescale_rope_freqs: lengths must be positive")
    if k < 0:
        raise ValueError("rescale_rope_freqs: k must be >= 0")
    if target_len == source_len or k == 0:
        return freqs.clone()
    out = freqs.clone()
    ratio = source_len / target_len
    k_eff = min(k, out.numel())
    # RoPE convention: index 0 = lowest frequency (longest period).
    out[:k_eff] = out[:k_eff] * ratio
    return out


def riflex_extrapolation_active(target_len: int, source_len: int) -> bool:
    """True iff RIFLEx will modify anything (target is longer than source)."""
    return target_len > source_len > 0
