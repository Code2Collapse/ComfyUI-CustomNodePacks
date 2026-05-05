"""VideoFramePlayerMEC - lightweight in-graph video scrubber + drag-crop + resize.

Originally a frame scrubber. Extended with:
  - Aspect ratio presets (free / 1:1 / 4:3 / 3:4 / 16:9 / 9:16 / 2:1 / 21:9 / custom / original)
  - Drag-crop overlay on the player canvas (8 handles + interior drag,
    aspect-locked snap, dim-overlay, rule-of-thirds guides)
  - Integrated resize with lanczos / bicubic / bilinear / area / nearest-exact
  - Optional upscale factor on top of the resize
  - Output modes: 'current_frame' (single frame) or 'all_frames' (whole batch
    processed with the same crop + resize so it can chain into samplers).

All operations are pure tensor ops. Crop is normalized in [0,1] so values
survive resolution swaps. Aspect-lock is enforced both client-side (drag)
and server-side (snap final crop to preset).

UX inspiration (no code copied — clean-room implementation, see NOTICE.md):
  - Olm DragCrop (Olli Sorjonen, https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop)
    popularised the in-node drag-rectangle crop UX. Its licence is
    source-available / not OSS, so its code cannot be vendored; this
    implementation was written independently using standard HTML5 canvas
    drag-handle patterns.
  - WhatDreamsCost-ComfyUI 'Load Video UI' (Jonathan Watkins,
    https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI) is GPL-3.0;
    its widget layout inspired this node's widget set. No GPL source was
    copied — both the widget definitions and the canvas overlay are original.
"""

from __future__ import annotations

from . import _interrupt_check as _IC

import os
import hashlib
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths


from . import _progress as _PB
_PREVIEW_QUALITY = 80


# ----------------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------------

ASPECT_PRESETS = {
    "free":        None,           # user-drag any rectangle
    "original":    None,           # locked to source W/H (computed at runtime)
    "1:1":         1.0,
    "4:3":         4.0 / 3.0,
    "3:4":         3.0 / 4.0,
    "16:9":        16.0 / 9.0,
    "9:16":        9.0 / 16.0,
    "2:1":         2.0,
    "21:9":        21.0 / 9.0,
    "custom":      None,           # uses custom_aspect_w / custom_aspect_h
}

RESIZE_METHODS = ["none", "lanczos", "bicubic", "bilinear", "area", "nearest-exact"]

OUTPUT_MODES = ["current_frame", "all_frames"]


# ----------------------------------------------------------------------
#  Resize helper
# ----------------------------------------------------------------------

def _resize_lanczos_bhwc(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Lanczos resize for (B,H,W,C). GPU antialiased bicubic for B>4
    (much faster than CPU PIL Lanczos, visually identical at moderate
    scale factors); single frames go through PIL Lanczos for max fidelity."""
    B, H, W, C = image.shape
    if H == target_h and W == target_w:
        return image
    if B > 4:
        img_bchw = image.permute(0, 3, 1, 2)
        try:
            resized = F.interpolate(
                img_bchw, size=(target_h, target_w),
                mode="bicubic", align_corners=False, antialias=True,
            )
        except TypeError:
            resized = F.interpolate(
                img_bchw, size=(target_h, target_w),
                mode="bicubic", align_corners=False,
            )
        return resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)
    out = []
    for i in range(B):
        _IC.check()
        arr = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
        if C == 1:
            arr = arr[:, :, 0]
        pil = Image.fromarray(arr).resize((target_w, target_h), Image.LANCZOS)
        t = torch.from_numpy(np.array(pil).astype("float32") / 255.0)
        if C == 1:
            t = t.unsqueeze(-1)
        out.append(t)
    return torch.stack(out).to(image.device)


_TORCH_INTERP = {
    "bicubic": "bicubic",
    "bilinear": "bilinear",
    "area": "area",
    "nearest-exact": "nearest-exact",
}


def _resize_bhwc(image: torch.Tensor, target_h: int, target_w: int, method: str) -> torch.Tensor:
    if method == "none":
        return image
    if image.shape[1] == target_h and image.shape[2] == target_w:
        return image
    if method == "lanczos":
        return _resize_lanczos_bhwc(image, target_h, target_w)
    mode = _TORCH_INTERP.get(method, "bilinear")
    img_bchw = image.permute(0, 3, 1, 2)
    kwargs = {"size": (target_h, target_w), "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    return F.interpolate(img_bchw, **kwargs).permute(0, 2, 3, 1).clamp(0.0, 1.0)


def _aspect_for(preset: str, src_w: int, src_h: int,
                custom_w: float, custom_h: float):
    if preset == "original":
        return (src_w / src_h) if src_h > 0 else None
    if preset == "custom":
        if custom_h <= 0 or custom_w <= 0:
            return None
        return float(custom_w) / float(custom_h)
    return ASPECT_PRESETS.get(preset)


def _snap_crop_to_aspect(cx: float, cy: float, cw: float, ch: float,
                         aspect) -> Tuple[float, float, float, float]:
    """Adjust normalized crop (cx,cy,cw,ch) to the target aspect, keeping
    the centre fixed and shrinking the longer dimension. All values in [0,1]."""
    if aspect is None or aspect <= 0:
        return (cx, cy, cw, ch)
    cur_ar = cw / max(ch, 1e-6)
    centre_x = cx + cw / 2.0
    centre_y = cy + ch / 2.0
    if cur_ar > aspect:
        cw = ch * aspect
    else:
        ch = cw / aspect
    cx = max(0.0, min(1.0 - cw, centre_x - cw / 2.0))
    cy = max(0.0, min(1.0 - ch, centre_y - ch / 2.0))
    return (cx, cy, cw, ch)


def _apply_crop_bhwc(image: torch.Tensor, cx: float, cy: float, cw: float, ch: float) -> torch.Tensor:
    """Crop (B,H,W,C) by normalized rect. Clamps to image bounds."""
    B, H, W, C = image.shape
    x0 = max(0, min(W - 1, int(round(cx * W))))
    y0 = max(0, min(H - 1, int(round(cy * H))))
    x1 = max(x0 + 1, min(W, int(round((cx + cw) * W))))
    y1 = max(y0 + 1, min(H, int(round((cy + ch) * H))))
    return image[:, y0:y1, x0:x1, :].contiguous()


# ----------------------------------------------------------------------
#  Node
# ----------------------------------------------------------------------

class VideoFramePlayerMEC:
    """Video frame player + drag-crop + integrated resize.

    Pipeline:
        frames -> pick (current_frame | all_frames)
              -> optional crop (drag rect, aspect-locked)
              -> optional resize (lanczos / bicubic / bilinear / area / nearest)
              -> optional upscale (multiplier)
              -> output

    Existing widgets keep their v1 behaviour. New widgets default to no-op
    so existing graphs produce identical output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Frame batch (B,H,W,C). B=number of frames."}),
                "frame_index": ("INT", {
                    "default": 0, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Current frame to emit on the IMAGE output. Drag the timeline to scrub."}),
                "output_mode": (OUTPUT_MODES, {
                    "default": "current_frame",
                    "tooltip": "current_frame: emit only the selected frame. all_frames: apply crop+resize to every frame."}),
                # crop
                "crop_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable the drag-crop rectangle on the preview."}),
                "aspect_ratio": (list(ASPECT_PRESETS.keys()), {
                    "default": "free",
                    "tooltip": "Aspect lock. 'original' = source W:H. 'custom' = custom_aspect_w:custom_aspect_h."}),
                "custom_aspect_w": ("FLOAT", {
                    "default": 16.0, "min": 0.0, "max": 999.0, "step": 0.1,
                    "tooltip": "Custom aspect width (used when aspect_ratio = custom)."}),
                "custom_aspect_h": ("FLOAT", {
                    "default": 9.0, "min": 0.0, "max": 999.0, "step": 0.1,
                    "tooltip": "Custom aspect height (used when aspect_ratio = custom)."}),
                "crop_x": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Crop left edge as fraction of source width [0..1]. Set by dragging the rectangle on the preview."}),
                "crop_y": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Crop top edge as fraction of source height [0..1]."}),
                "crop_w": ("FLOAT", {
                    "default": 1.0, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "Crop width as fraction of source width (0..1]."}),
                "crop_h": ("FLOAT", {
                    "default": 1.0, "min": 0.001, "max": 1.0, "step": 0.001,
                    "tooltip": "Crop height as fraction of source height (0..1]."}),
                # resize
                "resize_method": (RESIZE_METHODS, {
                    "default": "none",
                    "tooltip": "Post-crop resize. 'lanczos' = high-quality."}),
                "target_width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Target width after crop (0 = keep crop width)."}),
                "target_height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Target height after crop (0 = keep crop height)."}),
                "upscale_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.05,
                    "tooltip": "Multiplier applied AFTER target_width/target_height (1.0 = no upscale)."}),
                # preview
                "preview_width": ("INT", {
                    "default": 480, "min": 96, "max": 1920, "step": 16,
                    "tooltip": "Width (px) of preview JPEGs sent to the browser."}),
                "preview_quality": ("INT", {
                    "default": _PREVIEW_QUALITY, "min": 30, "max": 95, "step": 5,
                    "tooltip": "JPEG quality for previews."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "IMAGE", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "frame", "frame_index", "frame_count",
        "processed", "out_width", "out_height",
        "crop_x_px", "crop_y_px", "crop_w_px", "crop_h_px",
    )
    OUTPUT_TOOLTIPS = (
        "Single full-resolution frame at the slider position (1,H,W,C) - pre-crop, pre-resize.",
        "Echo of the selected frame_index (clamped to range).",
        "Total number of frames in the input batch.",
        "Processed output: the selected frame OR the whole batch (per output_mode), with crop + resize + upscale applied.",
        "Width (px) of the processed output.",
        "Height (px) of the processed output.",
        "Crop left edge in source pixels.",
        "Crop top edge in source pixels.",
        "Crop width in source pixels.",
        "Crop height in source pixels.",
    )
    FUNCTION = "play"
    CATEGORY = "MaskEditControl/Preview"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Video scrubber + drag-crop + integrated resize. Drag the timeline to scrub frames. "
        "Toggle crop_enabled and drag the rectangle on the preview to crop (aspect-locked when a preset is set). "
        "resize_method + target_width/target_height + upscale_factor produce the final output. "
        "Set output_mode = all_frames to process the whole batch."
    )

    @staticmethod
    def _compute_target_size(crop_w_px: int, crop_h_px: int,
                             target_w: int, target_h: int,
                             upscale: float) -> Tuple[int, int]:
        if target_w <= 0 and target_h <= 0:
            tw, th = crop_w_px, crop_h_px
        elif target_w > 0 and target_h <= 0:
            tw = target_w
            th = max(1, int(round(target_w * crop_h_px / max(crop_w_px, 1))))
        elif target_h > 0 and target_w <= 0:
            th = target_h
            tw = max(1, int(round(target_h * crop_w_px / max(crop_h_px, 1))))
        else:
            tw, th = target_w, target_h
        if upscale != 1.0:
            tw = max(1, int(round(tw * upscale)))
            th = max(1, int(round(th * upscale)))
        return (tw, th)

    def play(
        self,
        frames: torch.Tensor,
        frame_index: int = 0,
        output_mode: str = "current_frame",
        crop_enabled: bool = False,
        aspect_ratio: str = "free",
        custom_aspect_w: float = 16.0,
        custom_aspect_h: float = 9.0,
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        crop_w: float = 1.0,
        crop_h: float = 1.0,
        resize_method: str = "none",
        target_width: int = 0,
        target_height: int = 0,
        upscale_factor: float = 1.0,
        preview_width: int = 480,
        preview_quality: int = _PREVIEW_QUALITY,
    ):
        if frames is None or not hasattr(frames, "shape"):
            raise ValueError("VideoFramePlayerMEC: 'frames' input is missing or invalid (expected IMAGE B,H,W,C).")
        if frames.dim() != 4:
            raise ValueError(f"VideoFramePlayerMEC: expected 4D IMAGE (B,H,W,C), got shape {tuple(frames.shape)}.")

        B, H, W, C = frames.shape
        idx = max(0, min(int(frame_index), B - 1))

        # preview thumbnails (cached on disk by content digest)
        digest_src = (
            f"{B}x{H}x{W}x{C}_{preview_width}_{preview_quality}_"
        ).encode()
        for sample_idx in {0, B // 2, B - 1}:
            try:
                sample = (frames[sample_idx, :8, :8, :].detach().cpu().numpy() * 255).astype(np.uint8)
                digest_src += sample.tobytes()
            except Exception:
                pass
        batch_id = hashlib.sha256(digest_src).hexdigest()[:12]

        temp_dir = folder_paths.get_temp_directory()
        previews = []
        thumb_h = max(1, int(round(H * preview_width / max(W, 1))))
        for i in _PB.track(range(B), B, "FramePlayer: previews"):
            _IC.check()
            name = f"vfp_{batch_id}_{i:05d}.jpg"
            path = os.path.join(temp_dir, name)
            if not os.path.exists(path):
                arr = (frames[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                if pil.size[0] != preview_width:
                    pil = pil.resize((preview_width, thumb_h), Image.BILINEAR)
                pil.save(path, "JPEG", quality=int(preview_quality), optimize=False)
            previews.append({"filename": name, "subfolder": "", "type": "temp"})

        # backward-compat output: untouched current frame
        out_frame = frames[idx:idx + 1].clone()

        # crop
        cx, cy, cw, ch = float(crop_x), float(crop_y), float(crop_w), float(crop_h)
        if not crop_enabled:
            cx, cy, cw, ch = 0.0, 0.0, 1.0, 1.0
        else:
            aspect = _aspect_for(aspect_ratio, W, H, custom_aspect_w, custom_aspect_h)
            cx, cy, cw, ch = _snap_crop_to_aspect(cx, cy, cw, ch, aspect)
            # clamp
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            cw = max(0.001, min(1.0 - cx, cw))
            ch = max(0.001, min(1.0 - cy, ch))

        src = frames if output_mode == "all_frames" else out_frame

        if crop_enabled and (cx > 0.0 or cy > 0.0 or cw < 1.0 or ch < 1.0):
            processed = _apply_crop_bhwc(src, cx, cy, cw, ch)
        else:
            processed = src.clone() if src is out_frame else src

        # crop pixel coords (post-crop dims)
        crop_x_px = max(0, min(W - 1, int(round(cx * W))))
        crop_y_px = max(0, min(H - 1, int(round(cy * H))))
        crop_w_px = int(processed.shape[2])
        crop_h_px = int(processed.shape[1])

        # resize
        cur_h = processed.shape[1]
        cur_w = processed.shape[2]
        tw, th = self._compute_target_size(cur_w, cur_h, int(target_width), int(target_height), float(upscale_factor))
        if (tw, th) != (cur_w, cur_h):
            method = resize_method if resize_method != "none" else "bilinear"
            processed = _resize_bhwc(processed, th, tw, method)

        out_h, out_w = int(processed.shape[1]), int(processed.shape[2])

        ui_payload = {
            "frames": previews,
            "frame_count": [B],
            "current_index": [idx],
            "width": [W],
            "height": [H],
            "crop_enabled": [bool(crop_enabled)],
            "aspect_ratio": [aspect_ratio],
            "custom_aspect_w": [float(custom_aspect_w)],
            "custom_aspect_h": [float(custom_aspect_h)],
            "crop_x": [cx],
            "crop_y": [cy],
            "crop_w": [cw],
            "crop_h": [ch],
            "out_w": [out_w],
            "out_h": [out_h],
            "crop_x_px": [int(crop_x_px)],
            "crop_y_px": [int(crop_y_px)],
            "crop_w_px": [int(crop_w_px)],
            "crop_h_px": [int(crop_h_px)],
        }

        return {
            "ui": ui_payload,
            "result": (
                out_frame, idx, B,
                processed, out_w, out_h,
                int(crop_x_px), int(crop_y_px),
                int(crop_w_px), int(crop_h_px),
            ),
        }
