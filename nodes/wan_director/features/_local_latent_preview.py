"""Latent → RGB preview using ComfyUI's published Wan21/Wan22 factors.

These are the same per-channel coefficient matrices that ComfyUI core
uses for its built-in fast preview when decoding through the full VAE
would be too slow. The values are taken directly from
``comfy/latent_formats.py`` (``Wan21.latent_rgb_factors`` for 16-channel
Wan 2.1 latents, ``Wan22.latent_rgb_factors`` for 48-channel Wan 2.2
latents) so the previews match what the user already sees in the
"Latent Preview" sidebar.

The function ``latent_to_rgb_preview`` auto-routes by channel count:
    C == 16 → Wan 2.1 factors
    C == 48 → Wan 2.2 factors
Other channel counts raise ``ValueError``; we do NOT silently substitute
a random projection.

Pure-tensor, no model weights, deterministic.
"""
from __future__ import annotations

import torch


# Taken verbatim from ComfyUI ``comfy.latent_formats.Wan21`` so previews
# render identically to ComfyUI's built-in fast preview.
_WAN21_RGB_FACTORS: list[list[float]] = [
    [-0.1299, -0.1692,  0.2932],
    [ 0.0671,  0.0406,  0.0442],
    [ 0.3568,  0.2548,  0.1747],
    [ 0.0372,  0.2344,  0.1420],
    [ 0.0313,  0.0189, -0.0328],
    [ 0.0296, -0.0956, -0.0665],
    [-0.3477, -0.4059, -0.2925],
    [ 0.0166,  0.1902,  0.1975],
    [-0.0412,  0.0267, -0.1364],
    [-0.1293,  0.0740,  0.1636],
    [ 0.0680,  0.3019,  0.1128],
    [ 0.0032,  0.0581,  0.0639],
    [-0.1251,  0.0927,  0.1699],
    [ 0.0060, -0.0633,  0.0005],
    [ 0.3477,  0.2275,  0.2950],
    [ 0.1984,  0.0913,  0.1861],
]
_WAN21_RGB_BIAS: list[float] = [-0.1835, -0.0868, -0.3360]

# Taken verbatim from ComfyUI ``comfy.latent_formats.Wan22``.
_WAN22_RGB_FACTORS: list[list[float]] = [
    [ 0.0119,  0.0103,  0.0046], [-0.1062, -0.0504,  0.0165],
    [ 0.0140,  0.0409,  0.0491], [-0.0813, -0.0677,  0.0607],
    [ 0.0656,  0.0851,  0.0808], [ 0.0264,  0.0463,  0.0912],
    [ 0.0295,  0.0326,  0.0590], [-0.0244, -0.0270,  0.0025],
    [ 0.0443, -0.0102,  0.0288], [-0.0465, -0.0090, -0.0205],
    [ 0.0359,  0.0236,  0.0082], [-0.0776,  0.0854,  0.1048],
    [ 0.0564,  0.0264,  0.0561], [ 0.0006,  0.0594,  0.0418],
    [-0.0319, -0.0542, -0.0637], [-0.0268,  0.0024,  0.0260],
    [ 0.0539,  0.0265,  0.0358], [-0.0359, -0.0312, -0.0287],
    [-0.0285, -0.1032, -0.1237], [ 0.1041,  0.0537,  0.0622],
    [-0.0086, -0.0374, -0.0051], [ 0.0390,  0.0670,  0.2863],
    [ 0.0069,  0.0144,  0.0082], [ 0.0006, -0.0167,  0.0079],
    [ 0.0313, -0.0574, -0.0232], [-0.1454, -0.0902, -0.0481],
    [ 0.0714,  0.0827,  0.0447], [-0.0304, -0.0574, -0.0196],
    [ 0.0401,  0.0384,  0.0204], [-0.0758, -0.0297, -0.0014],
    [ 0.0568,  0.1307,  0.1372], [-0.0055, -0.0310, -0.0380],
    [ 0.0239, -0.0305,  0.0325], [-0.0663, -0.0673, -0.0140],
    [-0.0416, -0.0047, -0.0023], [ 0.0166,  0.0112, -0.0093],
    [-0.0211,  0.0011,  0.0331], [ 0.1833,  0.1466,  0.2250],
    [-0.0368,  0.0370,  0.0295], [-0.3441, -0.3543, -0.2008],
    [-0.0479, -0.0489, -0.0420], [-0.0660, -0.0153,  0.0800],
    [-0.0101,  0.0068,  0.0156], [-0.0690, -0.0452, -0.0927],
    [-0.0145,  0.0041,  0.0015], [ 0.0421,  0.0451,  0.0373],
    [ 0.0504, -0.0483, -0.0356], [-0.0837,  0.0168,  0.0055],
]
_WAN22_RGB_BIAS: list[float] = [0.0317, -0.0878, -0.1388]


_FACTORS_BY_CHANNELS: dict[int, tuple[list[list[float]], list[float], str]] = {
    16: (_WAN21_RGB_FACTORS, _WAN21_RGB_BIAS, "Wan21"),
    48: (_WAN22_RGB_FACTORS, _WAN22_RGB_BIAS, "Wan22"),
}


def supported_channel_counts() -> tuple[int, ...]:
    """Channel counts (C dim) for which a Wan latent-RGB matrix exists."""
    return tuple(sorted(_FACTORS_BY_CHANNELS))


def latent_to_rgb_preview(latent: torch.Tensor) -> torch.Tensor:
    """Convert a Wan video latent tensor to an RGB preview tensor.

    Args:
        latent: shape ``(B, C, T, H, W)`` (Wan video) or
                ``(B, C, H, W)`` (per-frame). Channel dim must be 16
                (Wan 2.1) or 48 (Wan 2.2).

    Returns:
        Tensor with the channel dim mapped to 3 (RGB) and values
        clamped to ``[0, 1]`` after the per-model bias is applied
        plus the standard ``+ 0.5`` recentre that ComfyUI uses for
        latent previews.
    """
    if latent.dim() not in (4, 5):
        raise ValueError(
            f"latent_to_rgb_preview: expected 4D or 5D tensor, got "
            f"shape {tuple(latent.shape)}"
        )
    C = latent.shape[1]
    if C not in _FACTORS_BY_CHANNELS:
        raise ValueError(
            f"latent_to_rgb_preview: no Wan latent_rgb_factors for "
            f"C={C}; supported = {supported_channel_counts()}"
        )
    factors_l, bias_l, _model = _FACTORS_BY_CHANNELS[C]
    factors = torch.tensor(factors_l, device=latent.device,
                           dtype=latent.dtype)              # (C, 3)
    bias    = torch.tensor(bias_l,    device=latent.device,
                           dtype=latent.dtype)              # (3,)
    if latent.dim() == 5:
        # einsum: (B, C, T, H, W) × (C, 3) → (B, 3, T, H, W)
        rgb = torch.einsum("bcthw,cr->brthw", latent, factors)
        rgb = rgb + bias.view(1, 3, 1, 1, 1)
    else:
        # (B, C, H, W) × (C, 3) → (B, 3, H, W)
        rgb = torch.einsum("bchw,cr->brhw", latent, factors)
        rgb = rgb + bias.view(1, 3, 1, 1)
    # ComfyUI's preview convention: shift to [0, 1] via +0.5, then clamp.
    return (rgb + 0.5).clamp_(0.0, 1.0)


def latent_model_for_channels(C: int) -> str:
    """Return the Wan model name whose factors are used for ``C`` channels.

    Raises ``ValueError`` for unsupported counts. Useful for the
    Director's info-output reporting.
    """
    if C not in _FACTORS_BY_CHANNELS:
        raise ValueError(
            f"latent_model_for_channels: C={C} not supported "
            f"(have {supported_channel_counts()})"
        )
    return _FACTORS_BY_CHANNELS[C][2]
