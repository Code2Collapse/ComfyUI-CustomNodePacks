"""ImageMaskEditorC2C — native-resolution mask editor node."""
from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np
import torch

from .._is_changed_util import hash_args_and_kwargs
from ..smart_crop import gaussian_blur, grow_or_shrink
from . import dispatch as D
from . import frames, store, temporal
from .modes import MODES

# L9.172 (A25 spec 5.8 item 43): appended after every mode input (saved widget values are restored by position)
APPENDED_AFTER_A28 = ("layer_output", "spline_keyframes", "spline_keyframe_easing")

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


def _load_layer_batch(editor_id: str, batch: int, target_hw: Tuple[int, int], frame_mode: str,
                      layer_output: str) -> tuple[np.ndarray, list[str]]:
    """uint8 [B,H,W] from the editor's layer stack: "visible" = the visible layers re-composited (store.
    composite_layers, the editor's own merge), else the one layer whose id or name is ``layer_output``. Plus one info
    note listing the stack."""
    H, W = target_hw
    eid = (editor_id or "").strip()
    manifest = store.get_layers_manifest(eid) if eid else None
    if not manifest:
        raise ValueError("Image Mask Editor: layer_output is set but this editor has no layer stack (it has one plain "
                         "layer) - add layers in the editor, or clear layer_output.")
    layers = manifest["layers"]
    if layer_output.lower() == "visible":
        pick = None
    else:
        hits = [l for l in layers if l["id"] == layer_output] or [l for l in layers if l["name"] == layer_output]
        if not hits:
            names = ", ".join(f"{l['name']!r} ({l['id']})" for l in layers)
            raise ValueError(f"Image Mask Editor: no layer named {layer_output!r} - the layers are {names}, or "
                             "'visible' for the visible ones together.")
        pick = hits[0]
    out = np.zeros((batch, H, W), dtype=np.uint8)
    cover = {l["id"]: 0.0 for l in layers}
    for i in range(batch):
        f = 0 if frame_mode == "shared" else i
        bufs = {}
        for l in layers:
            arr = store.get_layer_frame(eid, l["id"], f)
            if arr is None:
                continue
            if arr.shape != (H, W):
                arr = _resize_mask_u8(arr, (H, W))
            bufs[l["id"]] = arr
            cover[l["id"]] += float((arr > 127).mean()) / batch
        if pick is None:
            out[i] = store.composite_layers(layers, bufs, (H, W))
        elif pick["id"] in bufs:
            out[i] = bufs[pick["id"]]
    desc = "; ".join(
        f"{l['name'] or l['id']} [{l['id']}] {l['mode']}{'' if l['visible'] else ' hidden'}"
        f"{' locked' if l['locked'] else ''}{(' meta=' + json.dumps(l['meta'])) if l.get('meta') else ''}"
        f" cover={cover[l['id']] * 100:.1f}%" for l in layers)
    what = "visible layers re-composited" if pick is None else f"layer {pick['name'] or pick['id']!r} only"
    return out, [f"layer_output: {what}; stack (bottom to top): {desc}"]


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
                "layer_output": ("STRING", {
                    "default": "",
                    "tooltip": ("[paint] Which mask the layer stack gives. Empty: the flattened mask the editor "
                                "saved (as before). visible: the visible layers re-composited here with their blend "
                                "modes (add / subtract / intersect), exactly as the editor merges them. A layer name "
                                "or id: that layer's own pixels only. info lists every layer (mode, visible, locked, "
                                "metadata, coverage).")}),
                "spline_keyframes": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": ("[spline] Animate the drawn spline over frames. JSON list of keys "
                                "[{\"frame\": 0, \"x\": 0, \"y\": 0, \"scale\": 1, \"rotation\": 0}, ...] (x / y "
                                "in pixels, rotation in degrees) or {\"keys\": [...], \"frames\": N, \"pivot\": "
                                "[x, y]}. Interpolated between keys, held before the first and after the last; the "
                                "frame count is the image batch, else frames, else the last key + 1. Fill 'mask' "
                                "only. Empty = off (as before).")}),
                "spline_keyframe_easing": (temporal.EASINGS, {
                    "default": "linear",
                    "tooltip": "[spline] spline_keyframes: linear, or smooth (ease in and out between keys)."}),
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
        if str(kwargs.get("layer_output") or "").strip():     # L9.172: the layer files are read, so watch them
            extra = (store.layers_digest((editor_id or "").strip()),)
        if mode == "video":                      # keyframe edits live in the session store, not in the prompt
            from ..video_mask_editor import session_digest
            extra = extra + (session_digest(str(kwargs.get("video_session_id") or "").strip()),)
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
        layer_output: str = "",
        spline_keyframes: str = "",
        spline_keyframe_easing: str = "linear",
        **mode_kw,
    ):
        if image is not None and (not isinstance(image, torch.Tensor) or image.ndim != 4):
            raise ValueError("ImageMaskEditorC2C: image must be IMAGE tensor [B,H,W,C]")
        mode = mode or "paint"                 # a node saved before the modes existed (or a prompt without it)
        if mode not in MODES:
            raise ValueError(f"Image Mask Editor: unknown mode {mode!r}; one of {', '.join(MODES)}.")
        a = {**_MODE_DEFAULTS(), **{k: v for k, v in mode_kw.items() if v is not None}}
        layer_output = str(layer_output or "").strip()
        keys = str(spline_keyframes or "").strip()
        if layer_output and mode != "paint":
            raise ValueError("Image Mask Editor: layer_output reads the paint layer stack - set mode to paint, or "
                             "clear layer_output.")
        a["_keyframes"] = (keys, spline_keyframe_easing) if keys else None
        with torch.no_grad():
            if mode == "paint":
                return self._paint(image, editor_id, frame_mode, grow, feather, threshold, invert, input_mask,
                                   combine, a, layer_output=layer_output)
            return self._other_mode(mode, image, grow, feather, threshold, invert, input_mask, combine, a,
                                    node_id)

    @staticmethod
    def _finish(mask, grow, feather, threshold, invert):
        """The shared result stage: grow / shrink, feather, threshold, invert (identity at the defaults)."""
        out = grow_or_shrink(mask, int(grow), binary=False)      # soft masks stay soft (audit M-11)
        out = gaussian_blur(out, int(round(float(feather))))
        if float(threshold) > 0:
            out = (out >= float(threshold)).float()
        if invert:
            out = 1.0 - out
        return out

    def _paint(self, image, editor_id, frame_mode, grow, feather, threshold, invert, input_mask, combine, a,
               layer_output=""):
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
                               extra_mask=colour_mask if colour else None, layer_output=layer_output)
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
            if a.get("_keyframes"):                             # L9.172: keyed transforms over frames
                if a["spline_fill"] != "mask":
                    raise ValueError("Image Mask Editor: spline_keyframes animates spline_fill 'mask' (the flow "
                                     "pattern animates itself) - clear spline_keyframes or set spline_fill to mask.")
                text, easing = a["_keyframes"]
                mask, note = temporal.animate(mask, text, easing, int(image.shape[0]) if image is not None else None)
                info = f"{info}; {note}"
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
        layer_output: str = "",
    ):
        B, H, W, _C = image.shape
        frame_payload, frame_notes = frames.record_input_frames(image, editor_id)
        if layer_output:                                      # L9.172: the layer stack, read here
            painted_u8, notes = _load_layer_batch(editor_id, B, (H, W), frame_mode, layer_output)
        else:
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
        ui = {"c2c_ime_frames": [frame_payload] if frame_payload else []}
        if not frame_payload and frame_notes:
            # audit 2026-11-27 (c2c_backend_audit frames.py:22): over the recording budget (e.g. 124 frames of 2048x1080
            # = 274 M px > 268 M) the editor fell back to an upstream preview that said "run once" - after a run. The
            # reason now reaches the editor's banner too, not only the info output.
            ui["c2c_ime_frames_note"] = ["; ".join(frame_notes)]
        return {
            "ui": ui,
            "result": (blurred, preview, info),
        }


NODE_CLASS_MAPPINGS = {"ImageMaskEditorC2C": ImageMaskEditorC2C}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageMaskEditorC2C": "Image Mask Editor (C2C)"}
