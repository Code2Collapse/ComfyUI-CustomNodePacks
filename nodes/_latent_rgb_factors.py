"""Channel-count RGB factor tables for raw-latent preview decoding.

Used by ``_c2c_preview_guard._decode_raw_latent_preview`` as the safety-net
decoder for samplers (e.g. res4lyf / Radiance) that pass the RAW latent tensor
as the preview payload. Standard ComfyUI cannot render a raw latent, so this
maps it to RGB by channel count:

    C == 16 -> Wan 2.1 factors (comfy.latent_formats.Wan21)
    C == 48 -> Wan 2.2 factors (comfy.latent_formats.Wan22)
    other  -> None (caller falls back to a mean-projection grayscale)

The factor values are taken verbatim from ``comfy/latent_formats.py`` so the
safety-net preview matches ComfyUI's own built-in fast preview for Wan.
Pure-tensor, no model weights, deterministic.
"""
from __future__ import annotations

import torch

# comfy.latent_formats.Wan21.latent_rgb_factors (16 channels)
_WAN21_RGB_FACTORS = [
    [-0.1299, -0.1692, 0.2932], [0.0671, 0.0406, 0.0442],
    [0.3568, 0.2548, 0.1747], [0.0372, 0.2344, 0.1420],
    [0.0313, 0.0189, -0.0328], [0.0296, -0.0956, -0.0665],
    [-0.3477, -0.4059, -0.2925], [0.0166, 0.1902, 0.1975],
    [-0.0412, 0.0267, -0.1364], [-0.1293, 0.0740, 0.1636],
    [0.0680, 0.3019, 0.1128], [0.0032, 0.0581, 0.0639],
    [-0.1251, 0.0927, 0.1699], [0.0060, -0.0633, 0.0005],
    [0.3477, 0.2275, 0.2950], [0.1984, 0.0913, 0.1861],
]
_WAN21_RGB_BIAS = [-0.1835, -0.0868, -0.3360]

# comfy.latent_formats.Wan22.latent_rgb_factors (48 channels)
_WAN22_RGB_FACTORS = [
    [0.0119, 0.0103, 0.0046], [-0.1062, -0.0504, 0.0165],
    [0.0140, 0.0409, 0.0491], [-0.0813, -0.0677, 0.0607],
    [0.0656, 0.0851, 0.0808], [0.0264, 0.0463, 0.0912],
    [0.0295, 0.0326, 0.0590], [-0.0244, -0.0270, 0.0025],
    [0.0443, -0.0102, 0.0288], [-0.0465, -0.0090, -0.0205],
    [0.0359, 0.0236, 0.0082], [-0.0776, 0.0854, 0.1048],
    [0.0564, 0.0264, 0.0561], [0.0006, 0.0594, 0.0418],
    [-0.0319, -0.0542, -0.0637], [-0.0268, 0.0024, 0.0260],
    [0.0539, 0.0265, 0.0358], [-0.0359, -0.0312, -0.0287],
    [-0.0285, -0.1032, -0.1237], [0.1041, 0.0537, 0.0622],
    [-0.0086, -0.0374, -0.0051], [0.0390, 0.0670, 0.2863],
    [0.0069, 0.0144, 0.0082], [0.0006, -0.0167, 0.0079],
    [0.0313, -0.0574, -0.0232], [-0.1454, -0.0902, -0.0481],
    [0.0714, 0.0827, 0.0447], [-0.0304, -0.0574, -0.0196],
    [0.0401, 0.0384, 0.0204], [-0.0758, -0.0297, -0.0014],
    [0.0568, 0.1307, 0.1372], [-0.0055, -0.0310, -0.0380],
    [0.0239, -0.0305, 0.0325], [-0.0663, -0.0673, -0.0140],
    [-0.0416, -0.0047, -0.0023], [0.0166, 0.0112, -0.0093],
    [-0.0211, 0.0011, 0.0331], [0.1833, 0.1466, 0.2250],
    [-0.0368, 0.0370, 0.0295], [-0.3441, -0.3543, -0.2008],
    [-0.0479, -0.0489, -0.0420], [-0.0660, -0.0153, 0.0800],
    [-0.0101, 0.0068, 0.0156], [-0.0690, -0.0452, -0.0927],
    [-0.0145, 0.0041, 0.0015], [0.0421, 0.0451, 0.0373],
    [0.0504, -0.0483, -0.0356], [-0.0837, 0.0168, 0.0055],
]
_WAN22_RGB_BIAS = [0.0317, -0.0878, -0.1388]

_FACTORS = {16: (_WAN21_RGB_FACTORS, _WAN21_RGB_BIAS), 48: (_WAN22_RGB_FACTORS, _WAN22_RGB_BIAS)}


def decode_channels_to_rgb(x: torch.Tensor):
    """Map a (C,H,W) latent slice to a (3,H,W) RGB tensor in [0,1].

    Returns None for unknown channel counts (caller falls back to grayscale).
    Applies the model bias + the +0.5 recentre ComfyUI uses for previews.
    """
    c = x.shape[0]
    if c not in _FACTORS:
        return None
    factors_l, bias_l = _FACTORS[c]
    factors = torch.tensor(factors_l, device=x.device, dtype=x.dtype)   # (C,3)
    bias = torch.tensor(bias_l, device=x.device, dtype=x.dtype)         # (3,)
    # (C,H,W) x (C,3) -> (3,H,W)
    rgb = torch.einsum("chw,cr->rhw", x, factors) + bias.view(3, 1, 1)
    return (rgb + 0.5).clamp_(0.0, 1.0)
