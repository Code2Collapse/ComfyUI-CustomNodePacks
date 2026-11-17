"""One preview encoder for every C2C node (L9.282 P1.1 / L9.169; research docs/research/L9282_CNP_vs_AusBoss_previews.md).

preview_frames(t, ...) turns an IMAGE [B,H,W,C] (C = 1, 3 or 4) or MASK [B,H,W] into small uint8 frames for the eye:
- works in chunks of <= CHUNK frames and area-downscales ON THE TENSOR'S DEVICE first, so only the small result is
  copied to the CPU (the comparer's whole-batch path cost +4.1 GB at 120 x 1080p; chunked ~+0.3 GB, measured);
- rounds to the nearest code value (+0.5): the old truncation made previews half a code value too dark;
- transfer: "srgb" (display values, clamped), "linear" (the sRGB OETF - not gamma 2.2 - with an optional soft roll-off
  above 1.0), "data" (each frame stretched between its 1st and 99th percentile: depth, normals, motion);
- alpha: a 4th channel is kept as RGBA (encode with PNG / WebP); "alpha": True in the result;
- NEVER raises: returns (None, reason) and the node's real outputs are untouched.
encode_still(rgb_or_rgba_u8, fmt) -> bytes picks a format that can hold the frame (JPEG cannot hold alpha -> PNG).
"""
from __future__ import annotations

import io
from typing import List, Optional, Sequence, Tuple

import numpy as np

CHUNK = 4
TRANSFERS = ("srgb", "linear", "data")


def _srgb_oetf(x):
    import torch
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-12).pow(1.0 / 2.4) - 0.055)


def _rolloff(x, knee: float = 0.8):
    """Above the knee, highlights ease toward 1 instead of clipping flat (display only)."""
    import torch
    span = 1.0 - knee
    return torch.where(x > knee, knee + span * (1.0 - torch.exp(-(x - knee) / span)), x)


def target_size(w: int, h: int, long_edge: Optional[int]) -> Tuple[int, int]:
    if not long_edge or max(w, h) <= int(long_edge):
        return int(w), int(h)
    s = float(long_edge) / float(max(w, h))
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def preview_frames(t, indices: Optional[Sequence[int]] = None, long_edge: Optional[int] = 960,
                   transfer: str = "srgb", rolloff: bool = False,
                   size: Optional[Tuple[int, int]] = None) -> Tuple[Optional[List[np.ndarray]], str]:
    """-> ([uint8 HxWx3 or HxWx4 per frame], "") or (None, reason). size = exact (w, h), overriding long_edge (an
    encoder that needs even sides)."""
    try:
        import torch
        import torch.nn.functional as F
        if t is None:
            return None, "nothing to preview"
        if transfer not in TRANSFERS:
            return None, f"unknown preview transfer {transfer!r}"
        x = t if t.ndim == 4 else t.unsqueeze(-1)                 # MASK [B,H,W] -> [B,H,W,1]
        if x.ndim != 4 or x.shape[0] == 0:
            return None, f"cannot preview a tensor of shape {tuple(t.shape)}"
        b, h, w, c = x.shape
        idx = list(range(b)) if indices is None else [int(i) for i in indices if 0 <= int(i) < b]
        tw, th = (int(size[0]), int(size[1])) if size else target_size(w, h, long_edge)
        out: List[np.ndarray] = []
        for s in range(0, len(idx), CHUNK):
            sel = idx[s:s + CHUNK]
            with torch.no_grad():
                # consecutive frames: a VIEW of the batch (no copy); otherwise a small gather of <= CHUNK frames
                src = x[sel[0]:sel[-1] + 1] if sel == list(range(sel[0], sel[-1] + 1)) else x[sel]
                part = src.permute(0, 3, 1, 2)
                if part.dtype != torch.float32:
                    part = part.float()
                if (tw, th) != (w, h):
                    part = F.interpolate(part, size=(th, tw), mode="area")      # the copy is already small
                else:
                    part = part.clone()
                part = torch.nan_to_num_(part, nan=0.0, posinf=1.0, neginf=0.0)
                colour = part[:, :3] if c >= 3 else part[:, :1].expand(-1, 3, -1, -1)
                if transfer == "linear":
                    colour = _srgb_oetf(_rolloff(colour) if rolloff else colour)
                elif transfer == "data":
                    flat = colour.reshape(colour.shape[0], -1)
                    lo = torch.quantile(flat[:, ::max(1, flat.shape[1] // 65536)], 0.01, dim=1)[:, None, None, None]
                    hi = torch.quantile(flat[:, ::max(1, flat.shape[1] // 65536)], 0.99, dim=1)[:, None, None, None]
                    colour = (colour - lo) / (hi - lo).clamp_min(1e-6)
                chans = [colour.clamp(0.0, 1.0)]
                if c == 4:
                    chans.append(part[:, 3:4].clamp(0.0, 1.0))
                u8 = (torch.cat(chans, 1) * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)
                u8 = u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()     # only the small frames cross to the CPU
            out.extend(list(u8))
        return out, ""
    except Exception as exc:  # noqa: BLE001 - a preview never fails the node; the reason goes to the UI
        return None, f"preview unavailable: {type(exc).__name__}: {exc}"[:240]


def encode_still(frame_u8: np.ndarray, fmt: str = "jpeg", quality: int = 85) -> Tuple[bytes, str]:
    """-> (bytes, actual format). A frame with alpha is never written as JPEG (it raised before): PNG instead."""
    from PIL import Image
    alpha = frame_u8.ndim == 3 and frame_u8.shape[2] == 4
    fmt = (fmt or "jpeg").lower()
    if alpha and fmt in ("jpeg", "jpg"):
        fmt = "png"
    im = Image.fromarray(frame_u8, "RGBA" if alpha else "RGB")
    buf = io.BytesIO()
    if fmt in ("jpeg", "jpg"):
        im.save(buf, "JPEG", quality=int(quality))
        return buf.getvalue(), "jpeg"
    if fmt == "webp":
        im.save(buf, "WEBP", quality=int(quality))
        return buf.getvalue(), "webp"
    im.save(buf, "PNG", compress_level=1)
    return buf.getvalue(), "png"
