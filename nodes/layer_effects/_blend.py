# Blend-mode table ported from ComfyUI_LayerStyle by chflame163 (MIT,
# https://github.com/chflame163/ComfyUI_LayerStyle), py/blendmodes.py.
# LayerStyle in turn took that table from the Virtuoso Pack by Chris
# Freilich (https://github.com/chrisfreilich/virtuoso-nodes) - its own
# imagefunc.py:2506 says so - so both are credited here.
#
# Reimplemented in torch rather than copied, for one concrete reason:
# upstream's blendmodes.py imports the `blend_modes` pip package, which
# is NOT installed in this ComfyUI's python. A straight copy would have
# failed at import on the machine it was written for. The torch form
# also runs the whole batch in one pass instead of a PIL round-trip per
# frame. tests/test_layer_effects.py checks all 30 modes against a
# numpy transcription of the upstream formulas.
"""Torch-native RGBA blend modes on [B,H,W,4] float tensors in 0..1."""
from __future__ import annotations

import torch

BLEND_EPS: float = 1e-7

BLEND_MODE_NAMES: tuple[str, ...] = (
    "normal",
    "dissolve",
    "darken",
    "multiply",
    "color burn",
    "linear burn",
    "darker color",
    "lighten",
    "screen",
    "color dodge",
    "linear dodge(add)",
    "lighter color",
    "dodge",
    "overlay",
    "soft light",
    "hard light",
    "vivid light",
    "linear light",
    "pin light",
    "hard mix",
    "difference",
    "exclusion",
    "subtract",
    "divide",
    "hue",
    "saturation",
    "color",
    "luminosity",
    "grain extract",
    "grain merge",
)

GLOW_BLEND_MODE_NAMES: tuple[str, ...] = (
    "screen",
    "linear dodge(add)",
    "color dodge",
    "lighten",
    "dodge",
    "hard light",
    "linear light",
    *tuple(m for m in BLEND_MODE_NAMES if m not in {
        "screen", "linear dodge(add)", "color dodge", "lighten", "dodge",
        "hard light", "linear light",
    }),
)

# simple_mode / reference parity: do not clamp blend before opacity composite
_BLEND_NO_PRECLAMP: frozenset[str] = frozenset({
    "subtract",
    "linear burn",
    "exclusion",
    "linear light",
    "pin light",
    "grain extract",
    "grain merge",
})


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / torch.clamp(den, min=BLEND_EPS)


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """rgb [...,3] in 0..1 -> hsv same shape, h in 0..1."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc, _ = torch.max(rgb, dim=-1)
    minc, _ = torch.min(rgb, dim=-1)
    delta = maxc - minc
    h = torch.zeros_like(maxc)
    s = torch.zeros_like(maxc)
    v = maxc
    mask = delta > BLEND_EPS
    rc = ((g - b) / torch.clamp(delta, min=BLEND_EPS)) % 6.0
    gc = (b - r) / torch.clamp(delta, min=BLEND_EPS) + 2.0
    bc = (r - g) / torch.clamp(delta, min=BLEND_EPS) + 4.0
    h = torch.where((r == maxc) & mask, rc, h)
    h = torch.where((g == maxc) & mask, gc, h)
    h = torch.where((b == maxc) & mask, bc, h)
    h = (h / 6.0) % 1.0
    s = torch.where(maxc > BLEND_EPS, delta / maxc, s)
    return torch.stack((h, s, v), dim=-1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).floor().long() % 6
    f = h * 6.0 - (h * 6.0).floor()
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    return _clamp01(torch.stack((r, g, b), dim=-1))


def _blend_rgb(base: torch.Tensor, blend: torch.Tensor, mode: str) -> torch.Tensor:
    a, b = base, blend
    if mode == "normal":
        out = b
    elif mode == "multiply":
        out = a * b
    elif mode == "screen":
        out = 1.0 - (1.0 - a) * (1.0 - b)
    elif mode == "overlay":
        out = torch.where(a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    elif mode == "hard light":
        out = torch.where(b < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    elif mode == "soft light":
        d = torch.where(
            b <= 0.5,
            a - (1.0 - 2.0 * b) * a * (1.0 - a),
            a + (2.0 * b - 1.0) * (torch.sqrt(torch.clamp(a, min=0.0)) - a),
        )
        out = d
    elif mode == "difference":
        out = torch.abs(a - b)
    elif mode == "darken":
        out = torch.minimum(a, b)
    elif mode == "lighten":
        out = torch.maximum(a, b)
    elif mode == "linear dodge(add)":
        out = a + b
    elif mode == "dodge":
        out = _safe_div(a, 1.0 - b)
    elif mode == "color dodge":
        out = _safe_div(a, 1.0 - b)
    elif mode == "color burn":
        out = 1.0 - _safe_div(1.0 - a, b)
    elif mode == "linear burn":
        out = a + b - 1.0
    elif mode == "linear light":
        out = a + (2.0 * b) - 1.0
    elif mode == "exclusion":
        out = a + b - (2.0 * a * b)
    elif mode == "subtract":
        out = a - b
    elif mode == "divide":
        out = _safe_div(a, b)
    elif mode == "vivid light":
        out = torch.where(
            b <= 0.5,
            _safe_div(a, 1.0 - 2.0 * b),
            1.0 - _safe_div(1.0 - a, 2.0 * b - 0.5),
        )
    elif mode == "pin light":
        out = torch.where(
            b <= 0.5,
            torch.minimum(a, 2.0 * b),
            torch.maximum(a, 2.0 * (b - 0.5)),
        )
    elif mode == "grain extract":
        out = a - b + 0.5
    elif mode == "grain merge":
        out = a + b - 0.5
    elif mode == "hard mix":
        ll = _clamp01(a + (2.0 * b) - 1.0)
        return torch.round(ll)
    else:
        raise ValueError(f"Unknown rgb blend mode: {mode}")
    if mode in _BLEND_NO_PRECLAMP:
        return out
    return _clamp01(out)


def _hsv_channel_blend(
    backdrop: torch.Tensor,
    source: torch.Tensor,
    opacity: float,
    channel: str,
) -> torch.Tensor:
    bg_rgb = backdrop[..., :3]
    src_rgb = source[..., :3]
    src_a = source[..., 3:4]
    w = float(opacity) * src_a
    bg_h = _rgb_to_hsv(bg_rgb)
    src_h = _rgb_to_hsv(src_rgb)
    out_h = bg_h.clone()
    if channel == "hue":
        out_h[..., 0] = (1.0 - w[..., 0]) * bg_h[..., 0] + w[..., 0] * src_h[..., 0]
    elif channel == "saturation":
        out_h[..., 1] = (1.0 - w[..., 0]) * bg_h[..., 1] + w[..., 0] * src_h[..., 1]
    elif channel == "luminance":
        out_h[..., 2] = (1.0 - w[..., 0]) * bg_h[..., 2] + w[..., 0] * src_h[..., 2]
    elif channel == "color":
        out_h[..., :2] = (1.0 - w) * bg_h[..., :2] + w * src_h[..., :2]
    new_rgb = _hsv_to_rgb(out_h)
    return (1.0 - w) * bg_rgb + w * new_rgb


def _darker_lighter_color(
    backdrop: torch.Tensor,
    source: torch.Tensor,
    opacity: float,
    pick: str,
) -> torch.Tensor:
    bg = backdrop[..., :3]
    src = source[..., :3]
    src_a = source[..., 3:4]
    bg_v = _rgb_to_hsv(bg)[..., 2:3]
    src_v = _rgb_to_hsv(src)[..., 2:3]
    if pick == "dark":
        use_src = src_v < bg_v
    else:
        use_src = src_v > bg_v
    blend = torch.where(use_src, src, bg)
    w = src_a * opacity
    new_rgb = (1.0 - w) * bg + w * blend
    new_a = torch.maximum(backdrop[..., 3:4], source[..., 3:4])
    return torch.cat((_clamp01(new_rgb), new_a), dim=-1)


def _dissolve(
    backdrop: torch.Tensor,
    source: torch.Tensor,
    opacity: float,
    seed: int,
    batch_index: int,
) -> torch.Tensor:
    bg = backdrop[..., :3]
    src = source[..., :3]
    src_a = source[..., 3:4]
    trans = opacity * src_a
    gen = torch.Generator(device=backdrop.device)
    gen.manual_seed(int(seed) + int(batch_index) * 10007)
    rnd = torch.rand(
        backdrop.shape[:-1],
        generator=gen,
        device=backdrop.device,
        dtype=backdrop.dtype,
    )
    pick = rnd.unsqueeze(-1) < trans
    blended = torch.where(pick, src, bg)
    new_rgb = (1.0 - src_a) * bg + src_a * blended
    new_a = torch.maximum(backdrop[..., 3:4], source[..., 3:4])
    return torch.cat((_clamp01(new_rgb), new_a), dim=-1)


def _apply_blend_rgb(
    backdrop: torch.Tensor,
    source: torch.Tensor,
    opacity: float,
    mode: str,
) -> torch.Tensor:
    bg = backdrop[..., :3]
    src = source[..., :3]
    src_a = source[..., 3:4]
    op = opacity
    if mode == "normal":
        blend = src
    elif mode in ("hue", "saturation", "color", "luminosity"):
        ch_map = {
            "hue": "hue",
            "saturation": "saturation",
            "color": "color",
            "luminosity": "luminance",
        }
        new_rgb = _hsv_channel_blend(backdrop, source, op, ch_map[mode])
        new_a = torch.maximum(backdrop[..., 3:4], source[..., 3:4])
        return torch.cat((_clamp01(new_rgb), new_a), dim=-1)
    elif mode == "dissolve":
        raise RuntimeError("dissolve must use _dissolve()")
    elif mode in ("darker color", "lighter color"):
        return _darker_lighter_color(
            backdrop, source, op, "dark" if mode == "darker color" else "light"
        )
    else:
        blend = _blend_rgb(bg, src, mode)
    w = src_a * op
    new_rgb = (1.0 - w) * bg + w * blend
    new_a = torch.maximum(backdrop[..., 3:4], source[..., 3:4])
    return torch.cat((_clamp01(new_rgb), new_a), dim=-1)


def blend_rgba(
    backdrop: torch.Tensor,
    source: torch.Tensor,
    mode: str,
    opacity: float,
    *,
    dissolve_seed: int = 0,
    batch_index: int = 0,
) -> torch.Tensor:
    """Blend source over backdrop. Both [B,H,W,4] float 0..1. Returns new tensor."""
    if mode not in BLEND_MODE_NAMES:
        raise ValueError(f"Unsupported blend mode {mode!r}.")
    op = float(opacity)
    if mode == "dissolve":
        return _dissolve(backdrop, source, op, dissolve_seed, batch_index)
    return _apply_blend_rgb(backdrop, source, op, mode)


def solid_rgba(
    batch: int,
    height: int,
    width: int,
    rgb: tuple[float, float, float],
    alpha: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    t = torch.zeros((batch, height, width, 4), device=device, dtype=dtype)
    t[..., 0] = rgb[0]
    t[..., 1] = rgb[1]
    t[..., 2] = rgb[2]
    t[..., 3] = alpha
    return t.clone()
