"""
_vendored_resize.py — real resize/scale/aspect code vendored from ComfyUI_LayerStyle.

This is NOT a reimplementation. The functions below are copied verbatim (or as a
thin tensor wrapper around the verbatim functions) from:

    chflame163 / ComfyUI_LayerStyle  —  py/imagefunc.py + py/image_scale_by_aspect_ratio_v2.py
    License: MIT.  https://github.com/chflame163/ComfyUI_LayerStyle

We vendor it (rather than import the installed pack) so our nodes carry the same
proven resize behaviour even on a box where LayerStyle isn't installed. The MIT
licence is preserved at third_party/ComfyUI_LayerStyle/LICENSE.

`scale_tensor_by_aspect_ratio` reproduces LayerStyle's `ImageScaleByAspectRatioV2`
target-size math + per-frame `fit_resize_image`, operating on a ComfyUI IMAGE
batch tensor [B,H,W,C] (0..1 float) instead of a single PIL image.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

# ── verbatim MIT helpers from LayerStyle/py/imagefunc.py ─────────────────────

def tensor2pil(t_image: torch.Tensor) -> Image.Image:
    if t_image.dtype != torch.float32:
        t_image = t_image.float()
    return Image.fromarray(
        np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def image2mask(image: Image.Image) -> torch.Tensor:
    if image.mode == "L":
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    image = image.convert("RGB").split()[0]
    return torch.tensor([pil2tensor(image)[0, :, :].tolist()])


def num_round_up_to_multiple(number: int, multiple: int) -> int:
    remainder = number % multiple
    if remainder == 0:
        return number
    factor = (number + multiple - 1) // multiple
    return factor * multiple


def fit_resize_image(image: Image.Image, target_width: int, target_height: int,
                     fit: str, resize_sampler, background_color: str = "#000000") -> Image.Image:
    image = image.convert("RGB")
    orig_width, orig_height = image.size
    if image is not None:
        if fit == "letterbox":
            if orig_width / orig_height > target_width / target_height:
                fit_width = target_width
                fit_height = int(target_width / orig_width * orig_height)
            else:
                fit_height = target_height
                fit_width = int(target_height / orig_height * orig_width)
            fit_image = image.resize((fit_width, fit_height), resize_sampler)
            ret_image = Image.new("RGB", size=(target_width, target_height), color=background_color)
            ret_image.paste(fit_image, box=((target_width - fit_width) // 2, (target_height - fit_height) // 2))
        elif fit == "crop":
            if orig_width / orig_height > target_width / target_height:
                fit_width = int(orig_height * target_width / target_height)
                fit_image = image.crop(
                    ((orig_width - fit_width) // 2, 0, (orig_width - fit_width) // 2 + fit_width, orig_height))
            else:
                fit_height = int(orig_width * target_height / target_width)
                fit_image = image.crop(
                    (0, (orig_height - fit_height) // 2, orig_width, (orig_height - fit_height) // 2 + fit_height))
            ret_image = fit_image.resize((target_width, target_height), resize_sampler)
        else:  # fill
            ret_image = image.resize((target_width, target_height), resize_sampler)
    return ret_image


# user-facing option lists (mirror LayerStyle's node so a UI can offer the same)
ASPECT_RATIOS = ["original", "custom", "1:1", "3:2", "4:3", "16:9", "2:3", "3:4", "9:16"]
FIT_MODES = ["letterbox", "crop", "fill"]
RESAMPLE_METHODS = ["lanczos", "bicubic", "hamming", "bilinear", "box", "nearest"]
SCALE_TO_SIDE = ["None", "longest", "shortest", "width", "height", "total_pixel(kilo pixel)"]

_SAMPLERS = {
    "lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC, "hamming": Image.HAMMING,
    "bilinear": Image.BILINEAR, "box": Image.BOX, "nearest": Image.NEAREST,
}


def compute_target_size(orig_width, orig_height, aspect_ratio, proportional_width,
                        proportional_height, scale_to_side, scale_to_length, round_to_multiple):
    """LayerStyle ImageScaleByAspectRatioV2 target-size math (verbatim logic)."""
    if aspect_ratio == "original":
        ratio = orig_width / orig_height
    elif aspect_ratio == "custom":
        ratio = proportional_width / proportional_height
    else:
        s = aspect_ratio.split(":")
        ratio = int(s[0]) / int(s[1])

    if ratio > 1:
        if scale_to_side == "longest":
            target_width = scale_to_length; target_height = int(target_width / ratio)
        elif scale_to_side == "shortest":
            target_height = scale_to_length; target_width = int(target_height * ratio)
        elif scale_to_side == "width":
            target_width = scale_to_length; target_height = int(target_width / ratio)
        elif scale_to_side == "height":
            target_height = scale_to_length; target_width = int(target_height * ratio)
        elif scale_to_side == "total_pixel(kilo pixel)":
            target_width = int(math.sqrt(ratio * scale_to_length * 1000)); target_height = int(target_width / ratio)
        else:
            target_width = orig_width; target_height = int(target_width / ratio)
    else:
        if scale_to_side == "longest":
            target_height = scale_to_length; target_width = int(target_height * ratio)
        elif scale_to_side == "shortest":
            target_width = scale_to_length; target_height = int(target_width / ratio)
        elif scale_to_side == "width":
            target_width = scale_to_length; target_height = int(target_width / ratio)
        elif scale_to_side == "height":
            target_height = scale_to_length; target_width = int(target_height * ratio)
        elif scale_to_side == "total_pixel(kilo pixel)":
            target_width = int(math.sqrt(ratio * scale_to_length * 1000)); target_height = int(target_width / ratio)
        else:
            target_height = orig_height; target_width = int(target_height * ratio)

    if round_to_multiple != "None":
        multiple = int(round_to_multiple)
        target_width = num_round_up_to_multiple(target_width, multiple)
        target_height = num_round_up_to_multiple(target_height, multiple)
    return max(1, int(target_width)), max(1, int(target_height))


# user-facing fit/sizing modes for the in-node ControlAOV resize
SIZING_MODES = ["off", "width/height", "scale"]
FIT_WH = ["stretch", "pad", "crop"]          # -> LayerStyle fill / letterbox / crop
_FIT_MAP = {"stretch": "fill", "pad": "letterbox", "crop": "crop"}


def resize_wh_or_scale(images, mode="off", width=1024, height=1024, scale=1.0,
                       divisible_by=16, fit="crop", method="lanczos", pad_color="#000000"):
    """Resize an IMAGE [B,H,W,C] / MASK [B,H,W] by explicit width/height OR a scale
    factor, snapped to divisible_by, using the real LayerStyle fit_resize_image.

    mode: 'off' (passthrough) | 'width/height' (use width,height; 0 = derive from
    aspect) | 'scale' (multiply source size by `scale`).
    fit: stretch / pad(letterbox) / crop. Returns (out, target_w, target_h).
    """
    if images is None or images.shape[0] == 0 or mode == "off":
        return images, 0, 0
    is_mask = (images.ndim == 3)
    ow, oh = tensor2pil(images[0].unsqueeze(0)).size

    if mode == "scale":
        s = max(1e-4, float(scale))
        tw, th = max(1, round(ow * s)), max(1, round(oh * s))
    else:  # width/height
        tw, th = int(width), int(height)
        if tw == 0 and th == 0:
            tw, th = ow, oh
        elif tw == 0:
            tw = max(1, round(ow * th / oh))
        elif th == 0:
            th = max(1, round(oh * tw / ow))

    d = int(divisible_by)
    if d > 1:
        tw = max(d, int(round(tw / d)) * d)
        th = max(d, int(round(th / d)) * d)
    tw, th = max(1, tw), max(1, th)

    sampler = _SAMPLERS.get(method, Image.LANCZOS)
    ls_fit = _FIT_MAP.get(fit, "crop")
    out = []
    for i in range(images.shape[0]):
        frame = images[i].unsqueeze(0)
        if is_mask:
            pil = tensor2pil(frame).convert("L").convert("RGB")
            r = fit_resize_image(pil, tw, th, ls_fit, sampler, pad_color).convert("L")
            out.append(image2mask(r))
        else:
            pil = tensor2pil(frame).convert("RGB")
            r = fit_resize_image(pil, tw, th, ls_fit, sampler, pad_color)
            out.append(pil2tensor(r))
    return torch.cat(out, dim=0), tw, th


def scale_tensor_by_aspect_ratio(images, aspect_ratio="original", proportional_width=1,
                                 proportional_height=1, fit="letterbox", method="lanczos",
                                 round_to_multiple="None", scale_to_side="longest",
                                 scale_to_length=1024, background_color="#000000"):
    """Run LayerStyle's ImageScaleByAspectRatioV2 over an IMAGE batch [B,H,W,C].

    Returns (resized_batch, target_width, target_height). Mask [B,H,W] also handled.
    """
    if images is None or images.shape[0] == 0:
        return images, 0, 0
    is_mask = (images.ndim == 3)
    sampler = _SAMPLERS.get(method, Image.LANCZOS)
    first = images[0].unsqueeze(0)
    ow, oh = tensor2pil(first if not is_mask else first).size
    tw, th = compute_target_size(ow, oh, aspect_ratio, proportional_width, proportional_height,
                                 scale_to_side, scale_to_length, round_to_multiple)
    out = []
    for i in range(images.shape[0]):
        frame = images[i].unsqueeze(0)
        if is_mask:
            pil = tensor2pil(frame).convert("L").convert("RGB")
            r = fit_resize_image(pil, tw, th, fit, sampler, background_color).convert("L")
            out.append(image2mask(r))
        else:
            pil = tensor2pil(frame).convert("RGB")
            r = fit_resize_image(pil, tw, th, fit, sampler, background_color)
            out.append(pil2tensor(r))
    return torch.cat(out, dim=0), tw, th
