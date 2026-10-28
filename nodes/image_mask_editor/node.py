"""ImageMaskEditorC2C — native-resolution mask editor node."""
from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np
import torch

from .._is_changed_util import hash_args_and_kwargs
from ..smart_crop import gaussian_blur, grow_or_shrink
from . import dispatch as D
from . import frames, store
from .modes import MODES

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False

# Preview overlay: fixed colour + opacity (scene-linear friendly constants).
_PREVIEW_RGB = (0.65, 0.89, 0.63)
_PREVIEW_ALPHA = 0.45

_FEATHER_TOOLTIP = (
    "Soften the mask edge, in whole pixels (0 = off). Gaussian with sigma = 0.33 x this value, "
    "the same feather the Smart Crop/Stitch nodes use."
)


def _resize_mask_u8(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if mask.shape == (th, tw):
        return mask
    if _HAS_CV2:
        return cv2.resize(mask, (tw, th), interpolation=cv2.INTER_LINEAR)
    if Image is None:
        raise RuntimeError("Pillow required to resize masks without OpenCV")
    im = Image.fromarray(mask, mode="L").resize((tw, th), Image.BILINEAR)
    return np.array(im, dtype=np.uint8)


def _load_painted_batch(
    editor_id: str,
    batch: int,
    target_hw: Tuple[int, int],
    frame_mode: str,
) -> tuple[np.ndarray, list[str]]:
    """Return uint8 [B,H,W] painted masks and info notes."""
    H, W = target_hw
    notes: list[str] = []
    eid = (editor_id or "").strip()
    out = np.zeros((batch, H, W), dtype=np.uint8)
    if not eid:
        notes.append("no mask painted yet - open the editor")
        return out, notes

    shared = frame_mode == "shared"
    for i in range(batch):
        src_frame = 0 if shared else i
        arr = store.get_frame(eid, src_frame)
        if arr is None:
            continue
        if arr.shape != (H, W):
            old = list(arr.shape)
            arr = _resize_mask_u8(arr, (H, W))
            notes.append(f"resized stored frame {src_frame} from {old} to [{H},{W}]")
        out[i] = arr
    return out, notes


def _combine_masks(
    painted: torch.Tensor,
    input_mask: Optional[torch.Tensor],
    combine: str,
    batch: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """painted, result: float [B,H,W] in 0..1."""
    base = painted.float() / 255.0
    if input_mask is None or input_mask.numel() == 0:
        return base
    im = input_mask
    if im.ndim == 2:
        im = im.unsqueeze(0)
    if im.shape[0] == 1 and batch > 1:
        im = im.expand(batch, -1, -1)
    im = im.detach().to(base.device).float().clamp(0.0, 1.0)   # masks may arrive on the GPU
    if im.shape[-2:] != (h, w):
        im = torch.nn.functional.interpolate(
            im.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False,
        ).squeeze(1)
    mode = combine or "replace"
    if mode == "replace":
        return base
    if mode == "add":
        return torch.clamp(base + im, 0.0, 1.0)
    if mode == "subtract":
        return torch.clamp(im - base, 0.0, 1.0)
    if mode == "intersect":
        return base * im
    return base


_EXTRA_OUTPUTS = ("positive_coords", "negative_coords", "bboxes", "neg_bboxes", "points_json", "bbox_json",
                  "primary_bbox", "spline_data_out", "painted_image")
_DEFAULTS_CACHE: dict = {}


def _MODE_DEFAULTS() -> dict:
    """Every mode control at its default: an API prompt (or a node saved before a mode existed) may leave them out."""
    if not _DEFAULTS_CACHE:
        from .modes import defaults, mode_inputs
        _DEFAULTS_CACHE.update(defaults(mode_inputs()))
    return _DEFAULTS_CACHE


def _fit(mask: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
    """A mask [B,H,W] (or [H,W]) brought to b frames of h x w; unchanged when it already fits."""
    m = mask.float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.shape[0] == 1 and b > 1:
        m = m.expand(b, -1, -1)
    if tuple(m.shape[-2:]) != (h, w):
        m = torch.nn.functional.interpolate(m.unsqueeze(1), size=(h, w), mode="bilinear",
                                            align_corners=False).squeeze(1)
    return m


def _combine_float(base: torch.Tensor, input_mask: Optional[torch.Tensor], combine: str, b: int, h: int,
                   w: int) -> torch.Tensor:
    """_combine_masks for a float mask (the modes other than paint)."""
    if input_mask is None or input_mask.numel() == 0 or (combine or "replace") == "replace":
        return base
    im = _fit(input_mask.detach().to(base.device), b, h, w).clamp(0.0, 1.0)
    if combine == "add":
        return torch.clamp(base + im, 0.0, 1.0)
    if combine == "subtract":
        return torch.clamp(im - base, 0.0, 1.0)
    if combine == "intersect":
        return base * im
    return base


def _preview_overlay(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """IMAGE [B,H,W,3] with fixed-colour mask overlay."""
    m = mask.to(image.device, image.dtype).unsqueeze(-1)
    r, g, b = _PREVIEW_RGB
    overlay = torch.tensor([r, g, b], device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
    a = _PREVIEW_ALPHA
    return torch.clamp(image * (1.0 - m * a) + overlay * (m * a), 0.0, 1.0)


class ImageMaskEditorC2C:
    """Edit masks over the upstream IMAGE at native resolution."""

    VRAM_TIER = 0
    CATEGORY = "C2C/Masking"
    DESCRIPTION = (
        "Paint a mask over the image that actually reaches this node. "
        "Open the editor to draw; grow, feather, threshold and invert are "
        "non-destructive node parameters. mode picks how the mask is drawn: paint (with a colour layer), "
        "points + box, shape, spline, roto or video; each mode shows only its own controls. "
        "Docs: docs/image-mask-editor.md in this pack."
    )
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        from .modes import mode_inputs

        return {
            "required": {
                "editor_id": ("STRING", {
                    "default": "",
                    "tooltip": "Auto-set by the editor UI. Don't edit by hand.",
                }),
                "frame_mode": (
                    ["per_frame", "shared"],
                    {"default": "per_frame",
                     "tooltip": "per_frame: each batch index uses stored frame i; "
                                "shared: stored frame 0 applies to all."},
                ),
                "grow": ("INT", {
                    "default": 0, "min": -256, "max": 256, "step": 1,
                    "tooltip": "Grow (positive) or shrink (negative) the mask by this many pixels. Any "
                               "non-zero value makes the edge hard first (every partly covered pixel "
                               "counts); feather softens it again afterwards.",
                }),
                "feather": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 256.0, "step": 1.0,
                    "tooltip": _FEATHER_TOOLTIP,
                }),
                "threshold": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Binarize after feather. 0 = keep soft mask.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert mask after threshold.",
                }),
            },
            "optional": {
                # image moved here from required (L7.65 P01): roto, spline, shape and the colour layer draw without
                # one. It is still the only socket before input_mask, so saved links keep their slots.
                "image": ("IMAGE", {
                    "tooltip": "The picture to draw on (B,H,W,3); drives B/H/W. paint and video need it; points, "
                               "shape and spline take their size from it when connected.",
                }),
                "input_mask": ("MASK", {
                    "tooltip": "Optional mask to combine with the painted mask. points + box and shape draw onto "
                               "it; video uses it on the frames your keyframes do not fill.",
                }),
                "combine": (
                    ["replace", "add", "subtract", "intersect"],
                    {"default": "replace",
                     "tooltip": "How painted mask combines with input_mask. "
                                "Without input_mask only replace applies."},
                ),
                **mode_inputs(),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "STRING", "STRING", "BBOX", "BBOX", "STRING", "STRING", "BBOX",
                    "SPLINE_DATA", "IMAGE")
    RETURN_NAMES = ("mask", "preview", "info", "positive_coords", "negative_coords", "bboxes", "neg_bboxes",
                    "points_json", "bbox_json", "primary_bbox", "spline_data_out", "painted_image")
    OUTPUT_TOOLTIPS = (
        "The final mask [B,H,W] in 0-1: the active mode's mask, combined with input_mask, then grow, feather, "
        "threshold and invert.",
        "The input image with the mask tinted over it (the mask as a grey picture without an image).",
        "Frame count, stored masks, mode and notes (resized frames, frame recording skipped).",
        "points + box: positive points for SAM / SeC; spline: points along the spline. '[]' otherwise.",
        "points + box: negative points. '[]' otherwise.",
        "points + box: the positive boxes. Empty otherwise.",
        "points + box: the negative boxes. Empty otherwise.",
        "points + box: every point as JSON. '[]' otherwise.",
        "points + box: the first positive box [x1,y1,x2,y2]; spline: the spline's box. '[]' otherwise.",
        "points + box: the first positive box [x,y,w,h]; spline: the spline's box. [0,0,0,0] otherwise.",
        "spline: the spline data, for a tracker or another spline node.",
        "paint: the colour layer composited over the image (the image itself without colour strokes).",
    )
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(
        cls,
        image=None,
        editor_id="",
        frame_mode="per_frame",
        grow=0,
        feather=0.0,
        threshold=0.0,
        invert=False,
        input_mask=None,
        combine="replace",
        mode="paint",
        **kwargs,
    ):
        extra = ()
        if mode == "video":                      # keyframe edits live in the session store, not in the prompt
            from ..video_mask_editor import session_digest
            extra = (session_digest(str(kwargs.get("video_session_id") or "").strip()),)
        return hash_args_and_kwargs(
            image, editor_id, frame_mode, grow, feather, threshold, invert,
            input_mask, combine, mode,
            store.digest((editor_id or "").strip()),
            *extra,
            **kwargs,
        )

    def execute(
        self,
        image: Optional[torch.Tensor] = None,
        editor_id: str = "",
        frame_mode: str = "per_frame",
        grow: int = 0,
        feather: float = 0.0,
        threshold: float = 0.0,
        invert: bool = False,
        input_mask: Optional[torch.Tensor] = None,
        combine: str = "replace",
        mode: str = "paint",
        node_id=None,
        **mode_kw,
    ):
        if image is not None and (not isinstance(image, torch.Tensor) or image.ndim != 4):
            raise ValueError("ImageMaskEditorC2C: image must be IMAGE tensor [B,H,W,C]")
        mode = mode or "paint"                 # a node saved before the modes existed (or a prompt without it)
        if mode not in MODES:
            raise ValueError(f"Image Mask Editor: unknown mode {mode!r}; one of {', '.join(MODES)}.")
        a = {**_MODE_DEFAULTS(), **{k: v for k, v in mode_kw.items() if v is not None}}
        with torch.no_grad():
            if mode == "paint":
                return self._paint(image, editor_id, frame_mode, grow, feather, threshold, invert, input_mask,
                                   combine, a)
            return self._other_mode(mode, image, grow, feather, threshold, invert, input_mask, combine, a,
                                    node_id)

    @staticmethod
    def _finish(mask, grow, feather, threshold, invert):
        """The shared result stage: grow / shrink, feather, threshold, invert (identity at the defaults)."""
        out = grow_or_shrink(mask, int(grow))
        out = gaussian_blur(out, int(round(float(feather))))
        if float(threshold) > 0:
            out = (out >= float(threshold)).float()
        if invert:
            out = 1.0 - out
        return out

    def _paint(self, image, editor_id, frame_mode, grow, feather, threshold, invert, input_mask, combine, a):
        """paint: the mask layer (the modal editor and its store) and, when it has strokes, the colour layer (the
        former Paint Canvas). A node with no colour strokes runs exactly as before."""
        colour = str(a.get("canvas_data") or "").strip()
        if image is None and not colour:
            raise ValueError("Image Mask Editor (paint): connect an image to paint on "
                             "(or paint colour strokes with canvas_layer = colour).")
        painted = None
        if colour:
            painted, colour_mask = D.run_colour_layer(a, image)
        if image is None:                                    # a colour-only canvas: its picture and its mask
            mask = self._finish(colour_mask, grow, feather, threshold, invert)
            info = f"paint: colour layer {painted.shape[2]}x{painted.shape[1]}"
            return {"ui": {"c2c_ime_frames": []},
                    "result": (mask, _preview_overlay(painted[..., :3], mask), info, *self._empty_outputs(painted))}
        r = self._execute_impl(image, editor_id, frame_mode, grow, feather, threshold, invert, input_mask, combine,
                               extra_mask=colour_mask if colour else None)
        mask, preview, info = r["result"]
        if colour:
            info = f"{info}; colour layer"
        r["result"] = (mask, preview, info, *self._empty_outputs(painted if painted is not None else image))
        return r

    @staticmethod
    def _empty_outputs(painted_image):
        o = D._outputs(painted_image=painted_image)
        return tuple(o[k] for k in _EXTRA_OUTPUTS)

    def _other_mode(self, mode, image, grow, feather, threshold, invert, input_mask, combine, a, node_id):
        if mode == "points + box":
            mask, used, outs, ui, info = D.run_points(a, image, input_mask)
        elif mode == "shape":
            mask, used, outs, ui, info = D.run_shape(a, image, input_mask)
        elif mode == "spline":
            mask, used, outs, ui, info = D.run_spline(a, image, node_id)
        elif mode == "roto":
            mask, used, outs, ui, info = D.run_roto(a)
        else:
            mask, used, outs, ui, info = D.run_video(a, image, input_mask)
        if not used:                                         # the engine did not take input_mask: combine it here
            b, h, w = mask.shape
            mask = _combine_float(mask, input_mask, combine, b, h, w)
        mask = self._finish(mask, grow, feather, threshold, invert)
        if image is not None and image.shape[0] == mask.shape[0] and tuple(image.shape[1:3]) == tuple(mask.shape[1:]):
            preview = _preview_overlay(image, mask)
        else:
            preview = mask.unsqueeze(-1).expand(*mask.shape, 3).clone()
        if outs.get("painted_image") is None:
            outs["painted_image"] = image if image is not None else preview
        # info: the mode's own engine text, unchanged - a migrated node's info output is exactly the old node's
        result = (mask, preview, info, *(outs[k] for k in _EXTRA_OUTPUTS))
        return {"ui": {"c2c_ime_frames": [], **(ui or {})}, "result": result}

    def _execute_impl(
        self,
        image: torch.Tensor,
        editor_id: str,
        frame_mode: str,
        grow: int,
        feather: float,
        threshold: float,
        invert: bool,
        input_mask: Optional[torch.Tensor],
        combine: str,
        extra_mask: Optional[torch.Tensor] = None,
    ):
        B, H, W, _C = image.shape
        frame_payload, frame_notes = frames.record_input_frames(image, editor_id)
        painted_u8, notes = _load_painted_batch(
            editor_id, B, (H, W), frame_mode,
        )
        if frame_notes:
            notes = list(notes) + list(frame_notes)
        painted_t = torch.from_numpy(painted_u8)
        combined = _combine_masks(painted_t, input_mask, combine, B, H, W)
        if extra_mask is not None:                           # the colour layer's mask joins the mask layer
            combined = torch.maximum(combined, _fit(extra_mask, B, H, W))
        blurred = self._finish(combined, grow, feather, threshold, invert)
        preview = _preview_overlay(image, blurred)
        info_parts = [
            f"frames={B}",
            f"stored={len(store.list_frames((editor_id or '').strip())) if (editor_id or '').strip() else 0}",
            f"mode={frame_mode}",
        ]
        if notes:
            info_parts.extend(notes)
        info = "; ".join(dict.fromkeys(info_parts))
        return {
            "ui": {"c2c_ime_frames": [frame_payload] if frame_payload else []},
            "result": (blurred, preview, info),
        }


NODE_CLASS_MAPPINGS = {"ImageMaskEditorC2C": ImageMaskEditorC2C}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageMaskEditorC2C": "Image Mask Editor (C2C)"}
