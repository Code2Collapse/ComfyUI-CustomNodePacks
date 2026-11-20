"""Mask Tools modes added for L9.172 / A25 (spec item 82): chroma key with despill, and temporal matte cleanup.

Torch only, batch-correct, device-preserving. Apache-2.0, C2C.
"""
from __future__ import annotations

import torch

# screen colours (#RRGGBB) used as the reference of the green / blue presets
SCREEN_PRESETS = {"green": "#00B140", "blue": "#0047BB"}
CHROMA_SCREENS = ("green", "blue", "custom")
DESPILL_MODES = ("average", "max (gentle)", "min (strong)", "off")
CLEANUP_METHODS = ("median", "mean")
# channel order used when the custom colour has a tie: G, then B, then R
_TIE_ORDER = (1, 2, 0)


def parse_hex(text: str) -> tuple[float, float, float]:
    s = str(text or "").strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Mask Tools chroma key: colour {text!r} is not #RRGGBB (e.g. #00B140).")
    try:
        return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"Mask Tools chroma key: colour {text!r} is not #RRGGBB (e.g. #00B140).") from exc


def _screen_channels(screen: str, color: str) -> tuple[int, int, int, tuple[float, float, float]]:
    """(S, C1, C2, key colour) for a screen choice."""
    if screen not in CHROMA_SCREENS:
        raise ValueError(f"Mask Tools chroma key: screen must be one of {', '.join(CHROMA_SCREENS)}, not {screen!r}.")
    key = parse_hex(SCREEN_PRESETS.get(screen, color))
    if screen == "green":
        s = 1
    elif screen == "blue":
        s = 2
    else:
        top = max(key)
        s = next(c for c in _TIE_ORDER if key[c] == top)
    c1, c2 = (c for c in (0, 1, 2) if c != s)
    return s, c1, c2, key


def _difference(S, C1, C2, balance: float):
    hi, lo = torch.maximum(C1, C2), torch.minimum(C1, C2)
    return S - (balance * hi + (1.0 - balance) * lo)


def chroma_key(rgb: torch.Tensor, screen: str, color: str, balance: float, gain: float, clip_black: float,
               clip_white: float, despill: str, despill_strength: float, invert: bool):
    """rgb [B,H,W,3] 0..1 -> (mask [B,H,W], despilled rgb [B,H,W,3], notes). mask = foreground alpha (1 = subject)
    unless invert; colour-difference keyer: d = S - (balance*max(C1,C2) + (1-balance)*min(C1,C2)), normalised by d
    of the key colour."""
    if float(clip_white) <= float(clip_black):
        raise ValueError("Mask Tools chroma key: clip_white must be above clip_black.")
    if despill not in DESPILL_MODES:
        raise ValueError(f"Mask Tools chroma key: despill must be one of {', '.join(DESPILL_MODES)}, not {despill!r}.")
    s, c1, c2, key = _screen_channels(screen, color)
    names = "RGB"
    b = float(balance)
    d_ref = key[s] - (b * max(key[c1], key[c2]) + (1.0 - b) * min(key[c1], key[c2]))
    if d_ref <= 1e-4:
        raise ValueError(f"Mask Tools chroma key: this colour is not a screen colour - its {names[s]} channel does "
                         "not stand out from the other two. Pick the backdrop's colour.")
    rgb = rgb[..., :3].float()
    S, C1, C2 = rgb[..., s], rgb[..., c1], rgb[..., c2]
    d = _difference(S, C1, C2, b)
    screen_alpha = (d / d_ref * float(gain)).clamp(0.0, 1.0)
    fg = 1.0 - screen_alpha
    fg = ((fg - float(clip_black)) / (float(clip_white) - float(clip_black))).clamp(0.0, 1.0)
    mask = 1.0 - fg if invert else fg
    out = rgb.clone()
    notes = []
    if despill != "off":
        if despill == "average":
            limit = (C1 + C2) * 0.5
        elif despill == "max (gentle)":
            limit = torch.maximum(C1, C2)
        else:
            limit = torch.minimum(C1, C2)
        spill = (S - limit).clamp(min=0.0)
        out[..., s] = S - float(despill_strength) * spill
        touched = float(((spill * float(despill_strength)) > 1e-6).float().mean()) * 100.0
        notes.append(f"spill removed from {touched:.1f} % of pixels ({despill})")
    return mask, out, notes


def flicker(m: torch.Tensor) -> float:
    """Mean over t >= 1 of mean |m_t - m_{t-1}| (0 for a single frame)."""
    if m.shape[0] < 2:
        return 0.0
    return float((m[1:] - m[:-1]).abs().mean())


def temporal_cleanup(m: torch.Tensor, window: int, method: str, hysteresis: float, protect_motion: float,
                     image: torch.Tensor | None = None):
    """m [B,H,W] -> (cleaned [B,H,W], notes). Median / mean over a window of frames each side (shorter at the clip
    ends), optional motion protection (keep the original where the picture moved more than protect_motion), optional
    hysteresis (binary output that only switches on above 0.5 + h and off below 0.5 - h)."""
    if method not in CLEANUP_METHODS:
        raise ValueError(f"Mask Tools temporal cleanup: method must be one of {', '.join(CLEANUP_METHODS)}, "
                         f"not {method!r}.")
    m = m.float()
    B = int(m.shape[0])
    notes: list[str] = []
    if B < 2:
        notes.append("a single frame: nothing to clean over time")
        return m.clone(), notes
    w = max(1, int(window))
    out = torch.empty_like(m)
    for t in range(B):
        win = m[max(0, t - w): min(B - 1, t + w) + 1]
        out[t] = torch.median(win, dim=0).values if method == "median" else win.mean(dim=0)
    if float(protect_motion) > 0.0:
        if image is None or image.shape[0] != B:
            notes.append("protect_motion skipped: wire `image` with the same frame count as the mask")
        else:
            img = image[..., :3].float().to(m.device)
            if img.shape[1:3] != m.shape[1:]:
                notes.append("protect_motion skipped: `image` and the mask differ in size")
            else:
                step = (img[1:] - img[:-1]).abs().amax(dim=-1)            # [B-1,H,W]
                motion = torch.zeros_like(m)
                motion[1:] = step
                motion[:-1] = torch.maximum(motion[:-1], step)
                keep = motion > float(protect_motion)
                out = torch.where(keep, m, out)
                notes.append(f"motion kept the original matte on {float(keep.float().mean()) * 100.0:.1f} % of pixels")
    h = float(hysteresis)
    if h > 0.0:
        state = out[0] >= 0.5
        res = torch.empty_like(out)
        res[0] = state.float()
        for t in range(1, B):
            v = out[t]
            state = torch.where(v > 0.5 + h, torch.ones_like(state),
                                torch.where(v < 0.5 - h, torch.zeros_like(state), state))
            res[t] = state.float()
        out = res
        notes.append("hysteresis output is binary")
    notes.append(f"flicker before {flicker(m):.4g} / after {flicker(out):.4g}")
    return out, notes
