"""Local re-implementation of FETA (Frequency-Enhanced Temporal Attention).

Paper / refs: arXiv:2507.04984 and the FETA implementation in kijai's
ComfyUI-WanVideoWrapper. FETA injects a learned **temporal-frequency
bias** into the attention logits along the time axis so that the model
gives more weight to frequencies associated with realistic motion.

Pure-tensor primitive: given attention logits of shape
``(B, H, T, T)`` (B batches, H heads, T tokens along time), compute a
boost mask in the frequency domain and add it to the logits.

The boost is a cosine ramp centred at ``freq_center`` with bandwidth
``freq_bandwidth`` (both in normalised frequency [0, 0.5]). Strength
controlled by ``feta_scale``. When scale=0 this is a no-op identity.
"""
from __future__ import annotations

import torch


def _temporal_freq_mask(T: int, *, center: float, bandwidth: float,
                       device, dtype) -> torch.Tensor:
    """Build a (T,) frequency-domain boost mask, fft-shifted layout."""
    f = torch.fft.fftshift(torch.fft.fftfreq(T, device=device, dtype=dtype))
    # Cosine ramp: 1 at f=±center, falls to 0 at |f-center| >= bandwidth.
    dist  = (f.abs() - center).abs().clamp_max(bandwidth)
    ramp  = 0.5 * (1.0 + torch.cos(torch.pi * dist / max(bandwidth, 1e-6)))
    return ramp


def feta_attention_bias(
    attn_logits: torch.Tensor,
    *,
    feta_scale:     float = 0.5,
    freq_center:    float = 0.20,
    freq_bandwidth: float = 0.15,
) -> torch.Tensor:
    """Apply FETA frequency-domain bias to attention logits.

    Args:
        attn_logits: shape ``(..., T, T)`` — temporal attention logits.
        feta_scale:  bias strength; 0 disables FETA (returns input).
        freq_center: normalised frequency of the boost peak in [0, 0.5].
        freq_bandwidth: half-width of the boost ramp.

    Returns:
        Boosted logits with the same shape/dtype as the input.
    """
    if feta_scale == 0.0:
        return attn_logits
    if attn_logits.dim() < 2:
        raise ValueError("feta_attention_bias: need (..., T, T)")
    T = attn_logits.shape[-1]
    if attn_logits.shape[-2] != T:
        raise ValueError(
            f"feta_attention_bias: last 2 dims must be (T, T) — got "
            f"{tuple(attn_logits.shape[-2:])}"
        )
    mask = _temporal_freq_mask(
        T, center=freq_center, bandwidth=freq_bandwidth,
        device=attn_logits.device, dtype=attn_logits.dtype,
    )
    # FFT along last dim → bias high freqs around `center` → IFFT.
    spec = torch.fft.fftshift(torch.fft.fft(attn_logits, dim=-1), dim=-1)
    spec = spec * (1.0 + feta_scale * mask)
    boosted = torch.fft.ifft(torch.fft.ifftshift(spec, dim=-1), dim=-1).real
    return boosted.to(dtype=attn_logits.dtype)
