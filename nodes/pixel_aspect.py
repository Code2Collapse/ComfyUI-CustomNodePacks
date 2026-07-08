"""pixel_aspect.py — anamorphic / non-square Pixel Aspect Ratio round-trip.

AI models (and ComfyUI itself) assume SQUARE pixels. Anamorphic plates —
e.g. a 4448x3840 scan with PAR 1.7266 that displays as 2:1 in Nuke — read as
a squarish 1.158:1 image here, so generations come back distorted once the
lens squeeze is reapplied. Industry pipeline (Nuke Reformat semantics):

  PARDesqueezeMEC   plate → square pixels (stretch width by PAR, or squash
                    height) BEFORE any AI step, emitting a par_info JSON
                    that records exactly how to undo it.
  PARResqueezeMEC   AI output + par_info → back to the original pixel
                    dimensions for delivery back into Nuke, PAR untouched.

4448x3840 @ PAR 1.7266 → desqueeze(stretch_width) → 7680x3840 (true 2:1
square-pixel frame) → AI → resqueeze → 4448x3840 again. Lossless contract on
geometry: resqueeze restores the exact original W×H.

Author: Code2Collapse. Apache-2.0.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ._is_changed_util import hash_args_and_kwargs

# Common industry PARs. "custom" uses the pixel_aspect float widget.
PAR_PRESETS: Dict[str, float] = {
    "custom": 0.0,
    "square 1.0": 1.0,
    "anamorphic 2x (2.0)": 2.0,
    "anamorphic 1.8x (1.8)": 1.8,
    "ARRI 4448x3840→2:1 (1.7266)": 1.7266,
    "anamorphic 1.5x (1.5)": 1.5,
    "anamorphic 1.33x (1.33)": 1.33,
    "NTSC DV (0.9091)": 0.9091,
    "PAL DV (1.0940)": 1.0940,
}


def _resize(img: torch.Tensor, w: int, h: int, filt: str) -> torch.Tensor:
    x = img.permute(0, 3, 1, 2)
    kw = {} if filt in ("nearest", "area") else {"align_corners": False}
    x = F.interpolate(x, size=(h, w), mode=filt, **kw)
    return x.permute(0, 2, 3, 1).clamp(0, 1)


def _resolve_par(preset: str, pixel_aspect: float) -> float:
    v = PAR_PRESETS.get(preset, 0.0)
    par = v if v > 0 else float(pixel_aspect)
    if par <= 0:
        raise ValueError("pixel_aspect must be > 0 (use the preset or the custom float).")
    return par


class PARDesqueezeMEC:
    DESCRIPTION = ("Convert an anamorphic plate to SQUARE pixels before AI processing "
                   "(Nuke: 4448x3840 @ PAR 1.7266 → 7680x3840 at 2:1). Wire par_info "
                   "into PARResqueezeMEC to restore the original geometry losslessly.")
    CATEGORY = "MEC/Plate"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "par_info")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {}),
            "par_preset": (tuple(PAR_PRESETS.keys()),
                           {"default": "ARRI 4448x3840→2:1 (1.7266)"}),
            "pixel_aspect": ("FLOAT", {"default": 1.7266, "min": 0.1, "max": 4.0,
                             "step": 0.0001,
                             "tooltip": "Used when par_preset = custom. PAR>1 = pixels wider than tall."}),
            "method": (("stretch_width", "squash_height"), {"default": "stretch_width",
                       "tooltip": "stretch_width keeps every scanline (recommended); "
                                  "squash_height keeps the pixel count low."}),
            "filter": (("bicubic", "bilinear", "nearest", "area"), {"default": "bicubic"}),
        }}

    def execute(self, image: torch.Tensor, par_preset: str, pixel_aspect: float,
                method: str, filter: str) -> Tuple[torch.Tensor, str]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        b, h0, w0, c = image.shape
        par = _resolve_par(par_preset, pixel_aspect)
        if abs(par - 1.0) < 1e-6:
            out, nw, nh = image, w0, h0
        elif method == "squash_height":
            nw, nh = w0, max(1, round(h0 / par))
            out = _resize(image, nw, nh, filter)
        else:
            nw, nh = max(1, round(w0 * par)), h0
            out = _resize(image, nw, nh, filter)
        info = json.dumps({
            "orig_width": w0, "orig_height": h0, "pixel_aspect": par,
            "method": method, "filter": filter,
            "square_width": nw, "square_height": nh,
            "display_aspect": round((w0 * par) / h0, 4),
        })
        return (out, info)


class PARResqueezeMEC:
    DESCRIPTION = ("Return an AI-processed square-pixel frame to the plate's original "
                   "pixel dimensions (PAR metadata is reapplied in Nuke). Wire "
                   "par_info from PARDesqueezeMEC — the round trip restores the "
                   "exact original W×H even if the AI changed resolution.")
    CATEGORY = "MEC/Plate"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {}),
            "par_info": ("STRING", {"default": "", "forceInput": True}),
            "filter": (("bicubic", "bilinear", "nearest", "area"), {"default": "bicubic"}),
        }}

    def execute(self, image: torch.Tensor, par_info: str, filter: str) -> Tuple[torch.Tensor, str]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        try:
            meta: Dict[str, Any] = json.loads(par_info or "{}")
            w = int(meta["orig_width"])
            h = int(meta["orig_height"])
        except Exception as exc:
            raise ValueError(
                "par_info is not valid PARDesqueezeMEC output — wire the desqueeze "
                f"node's par_info here ({exc})."
            ) from exc
        out = image if (image.shape[2] == w and image.shape[1] == h) \
            else _resize(image, w, h, filter)
        return (out, json.dumps({"restored_width": w, "restored_height": h,
                                 "pixel_aspect": meta.get("pixel_aspect", 1.0)}))


NODE_CLASS_MAPPINGS = {
    "PARDesqueezeMEC": PARDesqueezeMEC,
    "PARResqueezeMEC": PARResqueezeMEC,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PARDesqueezeMEC": "PAR Desqueeze (anamorphic → square px)",
    "PARResqueezeMEC": "PAR Resqueeze (back to plate)",
}
