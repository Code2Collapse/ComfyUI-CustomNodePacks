"""ImageMaskEditorC2C — native-resolution mask editor node."""
from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np
import torch

from .._is_changed_util import hash_args_and_kwargs
from ..smart_crop import gaussian_blur, grow_or_shrink
from . import frames, store

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
        "non-destructive node parameters. Docs: docs/image-mask-editor.md in this pack."
    )
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input batch (B,H,W,3). Drives B/H/W.",
                }),
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
                "input_mask": ("MASK", {
                    "tooltip": "Optional mask to combine with the painted mask.",
                }),
                "combine": (
                    ["replace", "add", "subtract", "intersect"],
                    {"default": "replace",
                     "tooltip": "How painted mask combines with input_mask. "
                                "Without input_mask only replace applies."},
                ),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "preview", "info")
    OUTPUT_TOOLTIPS = (
        "The final mask [B,H,W] in 0-1: painted frames, combined with input_mask, then grow, feather, "
        "threshold and invert.",
        "The input image with the mask tinted over it.",
        "Frame count, stored masks, mode and notes (resized frames, frame recording skipped).",
    )
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(
        cls,
        image,
        editor_id,
        frame_mode,
        grow,
        feather,
        threshold,
        invert,
        input_mask=None,
        combine="replace",
        **kwargs,
    ):
        return hash_args_and_kwargs(
            image, editor_id, frame_mode, grow, feather, threshold, invert,
            input_mask, combine,
            store.digest((editor_id or "").strip()),
            **kwargs,
        )

    def execute(
        self,
        image: torch.Tensor,
        editor_id: str,
        frame_mode: str,
        grow: int,
        feather: float,
        threshold: float,
        invert: bool,
        input_mask: Optional[torch.Tensor] = None,
        combine: str = "replace",
    ):
        if image is None or not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("ImageMaskEditorC2C: image must be IMAGE tensor [B,H,W,C]")
        with torch.no_grad():
            return self._execute_impl(
                image, editor_id, frame_mode, grow, feather, threshold, invert,
                input_mask, combine,
            )

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
        grown = grow_or_shrink(combined, int(grow))
        blurred = gaussian_blur(grown, int(round(float(feather))))
        if float(threshold) > 0:
            blurred = (blurred >= float(threshold)).float()
        if invert:
            blurred = 1.0 - blurred
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
