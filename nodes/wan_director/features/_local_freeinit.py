"""Local re-implementation of FreeInit (paper arXiv:2312.07537).

FreeInit improves temporal consistency in video diffusion by, at the
start of every refinement iteration, replacing the **low-frequency
component** of the current sampling noise with the low-frequency
component of an early-step latent. The high-frequency component (the
"fine details") is fresh random noise. This keeps the overall motion
trajectory while letting the model re-roll noise-dependent details.

Pipeline (per refinement iteration ``i`` of N):
    1. Run a forward sampler pass from t=T down to a checkpoint step
       (or until completion).
    2. Take the latent at the checkpoint step → ``z_low_source``.
    3. Sample fresh gaussian noise ``z_new``.
    4. Compute ``z_mixed = freq_mix_3d(z_low_source, z_new, filter)`` —
       low-freq from source, high-freq from new.
    5. Use ``z_mixed`` as the starting noise for iteration ``i+1``.

This module provides the **frequency-domain mixing** primitives.
Sampler integration is done by the Director when it wraps the chosen
sampler in ``_apply_quality_stack``.

Filters supported (kijai-compatible names):
    "gaussian"      isotropic gaussian low-pass
    "butterworth"   Butterworth low-pass of order n
    "box"           ideal box low-pass (sharp cutoff)
    "ideal"         alias for "box"

The mixing operates in the **spatio-temporal frequency domain** via
torch.fft.fftn / ifftn on dims (-3, -2, -1) → (T, H, W). Channel
axis is untouched (each channel filtered independently).
"""
from __future__ import annotations

import torch


# ── Filter constructors ───────────────────────────────────────────────


def _meshgrid_freq(shape: tuple[int, int, int], device, dtype) -> torch.Tensor:
    """Build a 3D normalised-frequency magnitude tensor of shape ``shape``.

    Each coordinate u in dim k spans [-0.5, 0.5) (fft-shifted form), so
    the returned magnitude is ``sqrt(u_t^2 + u_h^2 + u_w^2)``.
    """
    T, H, W = shape
    ft = torch.fft.fftshift(torch.fft.fftfreq(T, device=device, dtype=dtype))
    fh = torch.fft.fftshift(torch.fft.fftfreq(H, device=device, dtype=dtype))
    fw = torch.fft.fftshift(torch.fft.fftfreq(W, device=device, dtype=dtype))
    grid_t, grid_h, grid_w = torch.meshgrid(ft, fh, fw, indexing="ij")
    return torch.sqrt(grid_t * grid_t + grid_h * grid_h + grid_w * grid_w)


def get_freq_filter(
    shape: tuple[int, int, int],
    *,
    filter_type: str = "butterworth",
    n: int = 4,
    d_s: float = 1.0,
    d_t: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a 3-D low-pass filter of shape ``(T, H, W)``.

    Args:
        shape: target (T, H, W) of the latent's spatial-temporal volume.
        filter_type: "gaussian", "butterworth", "box", or "ideal".
        n: Butterworth order (≥1).
        d_s: spatial cutoff scale (1.0 = nyquist/2).
        d_t: temporal cutoff scale (1.0 = nyquist/2).
        device / dtype: where to put the filter.

    Returns:
        A real-valued filter in **fft-shifted** layout matching the
        output of ``torch.fft.fftshift(torch.fft.fftn(x))``.
    """
    if d_s <= 0 or d_t <= 0:
        raise ValueError("FreeInit: d_s and d_t must be positive")
    T, H, W = shape
    ft = torch.fft.fftshift(torch.fft.fftfreq(T, device=device, dtype=dtype))
    fh = torch.fft.fftshift(torch.fft.fftfreq(H, device=device, dtype=dtype))
    fw = torch.fft.fftshift(torch.fft.fftfreq(W, device=device, dtype=dtype))
    # Anisotropic frequency magnitude (separate spatial/temporal scales).
    gt, gh, gw = torch.meshgrid(ft / d_t, fh / d_s, fw / d_s, indexing="ij")
    r = torch.sqrt(gt * gt + gh * gh + gw * gw)
    cutoff = 0.25  # half nyquist; d_s/d_t already rescale around this
    ftype = filter_type.lower()
    if ftype == "gaussian":
        sigma = cutoff / 2.355  # FWHM → sigma
        return torch.exp(-(r * r) / (2 * sigma * sigma))
    if ftype == "butterworth":
        n = max(1, int(n))
        return 1.0 / (1.0 + (r / cutoff).pow(2 * n))
    if ftype in ("box", "ideal"):
        return (r <= cutoff).to(dtype)
    raise ValueError(f"FreeInit: unknown filter_type {filter_type!r}")


# ── Frequency-domain mix ──────────────────────────────────────────────


def freq_mix_3d(
    x_low_source: torch.Tensor,
    x_new: torch.Tensor,
    filter_lp: torch.Tensor,
) -> torch.Tensor:
    """Replace low-freq of ``x_new`` with low-freq of ``x_low_source``.

    Inputs are latent tensors with at least 3 trailing dims (T, H, W).
    Common shape: ``(B, C, T, H, W)``. The FFT is taken over the last
    3 dims; batch and channel dims are independent.

    Args:
        x_low_source: tensor providing the low-frequency content.
        x_new: tensor providing the high-frequency content.
        filter_lp: low-pass filter, shape (T, H, W), in fft-shifted
            layout (see ``get_freq_filter``).

    Returns:
        Tensor with the same shape and dtype as ``x_new``.
    """
    if x_low_source.shape != x_new.shape:
        raise ValueError(
            f"freq_mix_3d: shape mismatch "
            f"{tuple(x_low_source.shape)} vs {tuple(x_new.shape)}"
        )
    if x_new.shape[-3:] != filter_lp.shape:
        raise ValueError(
            f"freq_mix_3d: filter shape {tuple(filter_lp.shape)} "
            f"does not match x.shape[-3:] {tuple(x_new.shape[-3:])}"
        )
    # FFT both inputs over the last 3 dims.
    X_src = torch.fft.fftshift(torch.fft.fftn(x_low_source, dim=(-3, -2, -1)),
                               dim=(-3, -2, -1))
    X_new = torch.fft.fftshift(torch.fft.fftn(x_new,        dim=(-3, -2, -1)),
                               dim=(-3, -2, -1))
    # Filter broadcasts over leading dims.
    f = filter_lp.to(device=X_src.device, dtype=X_src.real.dtype)
    # Linear mix in freq domain: low-freq from source, high-freq from new.
    X_out = X_src * f + X_new * (1.0 - f)
    X_out = torch.fft.ifftshift(X_out, dim=(-3, -2, -1))
    x_out = torch.fft.ifftn(X_out, dim=(-3, -2, -1)).real
    return x_out.to(dtype=x_new.dtype)
