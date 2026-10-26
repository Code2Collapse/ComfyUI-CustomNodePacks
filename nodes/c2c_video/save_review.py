"""Save Video review copy + burn-in (L9.57, owner 2026-10-25: "review copy is only for preview right? if yes, then 1st
option" = '<master>_review.mp4', H.264, OCIO display / view, half resolution, off by default; burn-in "Name + version,
Frame + timecode, Colour spaces, Custom template").

The review is written from the same frames as the master, chunk by chunk through Save's own encoder (no second decode,
no full-size copy in memory): each chunk is converted to the config's display (sRGB - Display) when the frames are in
another space, halved, and burnt in. The master never gets a burn-in. A streamed save (latent + vae) reads its frames
once, so it gets no review copy (said in the result).

Template: four corners "top-left | top-right | bottom-left | bottom-right" with {shot} {version} {frame} {tc} {date}
{artist} {colour}.
"""

from __future__ import annotations

import datetime
import os
from typing import Any

import numpy as np
import torch

REVIEW_SUFFIX = "_review"
BURN_MODES = ("standard", "custom", "off")
STANDARD_TEMPLATE = "{shot} {version}|{frame}  {tc}|{colour}|{date}"
REVIEW_DISPLAY = "sRGB - Display"


def review_path(master_path: str) -> str:
    """'<name>_review.mp4' next to the file - next to a sequence's folder, never inside it."""
    p = os.path.normpath(master_path)
    if os.path.isdir(p):
        return p + REVIEW_SUFFIX + ".mp4"
    return os.path.splitext(p)[0] + REVIEW_SUFFIX + ".mp4"


def corners(template: str) -> list[str]:
    parts = [s.strip() for s in str(template or "").split("|")]
    return (parts + ["", "", "", ""])[:4]


class _Safe(dict):
    def __missing__(self, key):                       # an unknown {token} stays as it is
        return "{" + key + "}"


def corner_texts(template: str, ctx: dict[str, Any]) -> list[str]:
    return [c.format_map(_Safe(ctx)) if c else "" for c in corners(template)]


def _font(px: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=max(8, int(px)))      # Pillow >= 10.1: scalable built-in font
    except TypeError:
        return ImageFont.load_default()


def burn(frame_u8: np.ndarray, texts: list[str]) -> np.ndarray:
    """Draw the four corner texts on an RGB uint8 frame (white on a dark box, readable on any picture)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame_u8)
    d = ImageDraw.Draw(im)
    h, w = frame_u8.shape[:2]
    size = max(9, h // 28)
    font, pad = _font(size), max(2, size // 4)
    for k, text in enumerate(texts):
        if not text:
            continue
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
        tw, th = x1 - x0, y1 - y0
        x = pad if k in (0, 2) else w - tw - 2 * pad
        y = pad if k in (0, 1) else h - th - 3 * pad
        d.rectangle([x - pad, y - pad, x + tw + pad, y + th + 2 * pad], fill=(0, 0, 0))
        d.text((x - x0, y - y0 + pad // 2), text, font=font, fill=(235, 235, 235))
    return np.asarray(im)


class ReviewFrames:
    """The review frames of `images`, made on read: display transform, half size (even), burn-in. Sliced by Save's
    encoders exactly like an IMAGE batch."""

    def __init__(self, images, *, transform=None, cfg=None, burn_mode: str = "standard", template: str = "",
                 ctx: dict[str, Any] | None = None, numbers: list[int] | None = None, timecode: str | None = None,
                 tc_step: int = 1, fps: float = 24.0):
        n, h, w, _c = images.shape
        self._src = images
        self._t = transform
        self._cfg = cfg
        self._burn = burn_mode if burn_mode in BURN_MODES else "standard"
        self._template = STANDARD_TEMPLATE if self._burn == "standard" else str(template or "")
        self._ctx = dict(ctx or {})
        self._numbers = numbers
        self._tc, self._tc_step, self._fps = timecode, int(tc_step or 1), float(fps or 24)
        h2, w2 = max(2, (h // 2) // 2 * 2), max(2, (w // 2) // 2 * 2)        # half size, even for 4:2:0
        self.shape = (int(n), int(h2), int(w2), 3)
        self.ndim = 4
        self.dtype = torch.float32

    def _frame_ctx(self, i: int) -> dict[str, Any]:
        from .source_meta import frames_to_tc, tc_add
        num = self._numbers[i] if self._numbers and i < len(self._numbers) else i + 1
        tc = tc_add(self._tc, i * self._tc_step, self._fps) if self._tc else frames_to_tc(i, self._fps)
        return dict(self._ctx, frame=num, tc=tc)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key + 1)
            one = True
        else:
            one = False
        a = 0 if key.start is None else int(key.start)
        b = self.shape[0] if key.stop is None else min(int(key.stop), self.shape[0])
        x = self._src[a:b]
        x = (x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x)))[..., :3].float()
        if self._t is not None:
            from .colour_cm import apply
            x = torch.from_numpy(apply(x.cpu().numpy().copy(), self._t[0], inverse=self._t[1], cfg=self._cfg))
        x = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=self.shape[1:3], mode="area")
        x = x.permute(0, 2, 3, 1).clamp(0, 1)
        if self._burn != "off" and self._template:
            u8 = (x.numpy() * 255.0 + 0.5).astype(np.uint8)
            out = np.stack([burn(u8[k], corner_texts(self._template, self._frame_ctx(a + k))) for k in range(len(u8))])
            x = torch.from_numpy(out.astype(np.float32) / 255.0)
        return x[0] if one else x

    def cpu(self):
        return self


def write_review(images, *, master_path: str, out_root: str, folder_type: str, fps: float, audio=None,
                 cs_in: str | None, burn_mode: str, template: str, ctx: dict[str, Any], carry: dict | None,
                 quality: int = 70) -> dict[str, Any]:
    """Encode '<master>_review.mp4' -> {filename, subfolder, type, path, width, height, frames}."""
    from . import colour_cm as CM
    from . import save_matrix as SM
    from .save_encode import encode_video
    cfg = CM.config()
    transform = None
    if cfg is not None and cs_in and cfg.getColorSpace(cs_in) is not None and cs_in != REVIEW_DISPLAY \
            and REVIEW_DISPLAY in list(cfg.getDisplays()):
        transform = CM.between(cfg, cs_in, REVIEW_DISPLAY)
    frames = ReviewFrames(images, transform=transform, cfg=cfg, burn_mode=burn_mode, template=template,
                          ctx=dict(ctx, colour=(f"{cs_in} -> {REVIEW_DISPLAY}" if transform else (cs_in or ""))),
                          numbers=(carry or {}).get("numbers"), timecode=(carry or {}).get("timecode"),
                          tc_step=int((carry or {}).get("step") or 1), fps=fps)
    spec = SM.build_spec(SM.Choice("MP4", "H.264", "High", "8", "4:2:0"))
    path = review_path(master_path)
    rel = os.path.relpath(path, out_root).replace("\\", "/") if out_root else os.path.basename(path)
    res = encode_video(frames, spec, path_info=(path, os.path.splitext(os.path.basename(path))[0], rel, folder_type),
                       fps=fps, quality=int(quality), colorspace_in=REVIEW_DISPLAY, colorspace_out=REVIEW_DISPLAY,
                       audio=audio, pad_note=None)
    return {"filename": res["filename"], "subfolder": res["subfolder"], "type": res["type"], "path": path,
            "width": frames.shape[2], "height": frames.shape[1], "frames": frames.shape[0],
            "burn_in": burn_mode, "display": REVIEW_DISPLAY, "converted_from": cs_in if transform else None}


def standard_ctx(*, shot: str, version: str | None, artist: str = "") -> dict[str, Any]:
    return {"shot": shot, "version": version or "", "artist": artist,
            "date": datetime.date.today().isoformat()}
