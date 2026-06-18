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


def apply_feta(model, feta_scale: float = 0.5,
               freq_center: float = 0.20,
               freq_bandwidth: float = 0.15,
               start_percent: float = 0.0,
               end_percent: float = 1.0):
    """Patch a ComfyUI model to apply FETA attention rescaling.

    Hooks into the self-attention output to boost frequency components
    associated with realistic temporal motion.

    Args:
        model:          ComfyUI model wrapper.
        feta_scale:     Boost strength (0 = disabled).
        freq_center:    Normalised frequency of boost peak in [0, 0.5].
        freq_bandwidth: Half-width of the cosine ramp.
        start_percent:  Denoising fraction where FETA activates.
        end_percent:    Denoising fraction where FETA deactivates.

    Returns:
        Cloned + patched model.
    """
    if feta_scale == 0.0:
        return model

    m = model.clone()

    def feta_attn1_output_patch(out, extra_options):
        sigma = extra_options.get("sigmas", None)
        if sigma is not None:
            model_sampling = model.get_model_object("model_sampling")
            if model_sampling is not None:
                pct = float(model_sampling.percent_through_sigma(sigma[0]))
            else:
                pct = 0.5
            if not (start_percent <= pct < end_percent):
                return out

        if out.dim() < 2:
            return out

        T = out.shape[-2]
        if T < 2:
            return out

        try:
            boosted = feta_attention_bias(
                out, feta_scale=feta_scale,
                freq_center=freq_center,
                freq_bandwidth=freq_bandwidth,
            )
            return boosted
        except Exception:
            return out

    m.set_model_attn1_output_patch(feta_attn1_output_patch)
    return m
