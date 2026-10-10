# Ported from ComfyUI_LayerStyle by chflame163 (MIT,
# https://github.com/chflame163/ComfyUI_LayerStyle), the LayerMask family:
# mask_by_color, create_gradient_mask, mask_gradient, mask_grain,
# mask_motion_blur and pixel_spread.
#
# Only the five capabilities this pack did NOT already have were taken. The
# rest of LayerMask is covered by MaskRefineMEC, LayerEffectStrokeMEC and
# nodes/bbox_nodes.py; see NOTICE for the full list of what was skipped and
# why. Reimplemented in torch, not copied: upstream round-trips every frame
# through PIL.
"""MEC Mask toolkit nodes — torch-native, batch-correct."""
from __future__ import annotations

import torch

from .._is_changed_util import hash_args_and_kwargs
from ._ops import (
    align_mask_batch,
    build_report,
    clone_output,
    edge_spread_image,
    ensure_bhw4_image,
    ensure_bhw_mask,
    extract_channel,
    gradient_mask_bhw,
    mask_from_color,
    mask_grain_bhw,
    motion_blur_mask,
    resolve_size,
)

_CATEGORY = "MEC/Mask"

_COLOR_DESC = (
    "Pick a colour and tolerance to build a matte. Gap filling and edge cleanup are "
    "deliberately not here — chain Mask Refine (MEC), which does hole fill, morph, "
    "guided filter and CRF properly."
)


class MaskFromColorMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Plate to sample against the picked colour."}),
                "color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Target colour as #RRGGBB — the hue you want to keep or remove.",
                }),
                "colorspace": (["rgb", "hsv", "lab"], {
                    "default": "rgb",
                    "tooltip": (
                        "Distance metric. LAB separates dark navy from dark brown; "
                        "RGB treats them as similar luminance."
                    ),
                }),
                "tolerance": ("FLOAT", {
                    "default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "How far a pixel's colour can drift from the pick and still count as a match.",
                }),
                "soft_falloff": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": (
                        "Graded band outside tolerance — stops a colour key from cutting "
                        "like a pair of scissors."
                    ),
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Swap matched and unmatched regions.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = _COLOR_DESC

    @classmethod
    def IS_CHANGED(cls, image, color, colorspace, tolerance, soft_falloff, invert, **kwargs):
        return hash_args_and_kwargs(image, color, colorspace, tolerance, soft_falloff, invert, **kwargs)

    def execute(
        self,
        image: torch.Tensor,
        color: str,
        colorspace: str,
        tolerance: float,
        soft_falloff: float,
        invert: bool,
    ) -> tuple[torch.Tensor, str]:
        image = ensure_bhw4_image(image, "MaskFromColorMEC")
        b, h, w, _ = image.shape
        notes: list[str] = []
        mask = mask_from_color(
            image, color, colorspace, tolerance, soft_falloff,
            device=image.device, dtype=image.dtype,
        )
        if invert:
            mask = 1.0 - mask
        mask = clone_output(mask)
        cov = mask.mean().item() * 100.0
        rep = build_report("MaskFromColorMEC", b, notes, coverage=cov)
        return mask, rep


class MaskGradientMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512, "min": 4, "max": 16384, "step": 1,
                    "tooltip": "Output width when size_as is not connected.",
                }),
                "height": ("INT", {
                    "default": 512, "min": 4, "max": 16384, "step": 1,
                    "tooltip": "Output height when size_as is not connected.",
                }),
                "gradient_type": (["linear", "radial", "angular"], {
                    "default": "linear",
                    "tooltip": "Ramp shape — linear edge fade, radial vignette, or angular wipe.",
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1,
                    "tooltip": "Direction for linear, or rotation offset for angular ramps.",
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Radial/angular origin, normalised 0–1 across the frame.",
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Radial/angular origin, normalised 0–1 down the frame.",
                }),
                "start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mask value at the ramp start (black end).",
                }),
                "end": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Mask value at the ramp end (white end).",
                }),
            },
            "optional": {
                "size_as": ("IMAGE", {
                    "tooltip": (
                        "When connected, output size is taken from this image and "
                        "width/height sliders are ignored."
                    ),
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional matte to multiply into — fade a garbage matte off at the frame edge.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Procedural linear, radial, or angular ramp matte with optional multiply."

    @classmethod
    def IS_CHANGED(cls, width, height, gradient_type, angle, center_x, center_y, start, end, **kwargs):
        return hash_args_and_kwargs(
            width, height, gradient_type, angle, center_x, center_y, start, end, **kwargs,
        )

    def execute(
        self,
        width: int,
        height: int,
        gradient_type: str,
        angle: float,
        center_x: float,
        center_y: float,
        start: float,
        end: float,
        size_as: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str]:
        notes: list[str] = []
        w, h = resolve_size(width, height, size_as, notes)
        batch = 1
        device = torch.device("cpu")
        dtype = torch.float32
        if mask is not None:
            mask = ensure_bhw_mask(mask, "MaskGradientMEC")
            batch = max(batch, mask.shape[0])
            device, dtype = mask.device, mask.dtype
        if size_as is not None:
            size_img = ensure_bhw4_image(size_as, "MaskGradientMEC")
            batch = max(batch, size_img.shape[0])
            device, dtype = size_img.device, size_img.dtype
        ramp = gradient_mask_bhw(
            batch, h, w, gradient_type, angle, center_x, center_y, start, end,
            device=device, dtype=dtype,
        )
        if mask is not None:
            (mask_aligned,), b = align_mask_batch([mask])
            assert mask_aligned is not None
            if mask_aligned.shape[-2:] != (h, w):
                mask_aligned = torch.nn.functional.interpolate(
                    mask_aligned.unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            ramp = ramp[:b] * mask_aligned[:b]
            batch = b
        ramp = clone_output(ramp)
        cov = ramp.mean().item() * 100.0
        rep = build_report("MaskGradientMEC", batch, notes, coverage=cov)
        return ramp, rep


class MaskGrainMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Matte to dither — breaks 8-bit banding when stretched."}),
                "amount": ("INT", {
                    "default": 6, "min": 0, "max": 127, "step": 1,
                    "tooltip": "Noise strength — enough to hide banding, not enough to read as texture.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0x7FFFFFFF, "step": 1,
                    "tooltip": (
                        "Fixed seed holds the grain still through a sequence; "
                        "upstream reseeded every frame and the matte crawled."
                    ),
                }),
                "grain_size": ("INT", {
                    "default": 4, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Noise frequency — larger values give coarser, less speckly grain.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flip mask polarity before adding grain.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Film grain on a matte to break quantisation banding."

    @classmethod
    def IS_CHANGED(cls, mask, amount, seed, grain_size, invert, **kwargs):
        return hash_args_and_kwargs(mask, amount, seed, grain_size, invert, **kwargs)

    def execute(
        self,
        mask: torch.Tensor,
        amount: int,
        seed: int,
        grain_size: int,
        invert: bool,
    ) -> tuple[torch.Tensor, str]:
        mask = ensure_bhw_mask(mask, "MaskGrainMEC")
        assert mask is not None
        notes: list[str] = []
        if invert:
            mask = 1.0 - mask
        out = mask_grain_bhw(
            mask, amount, seed, grain_size,
            device=mask.device, dtype=mask.dtype,
        )
        out = clone_output(out)
        cov = out.mean().item() * 100.0
        rep = build_report("MaskGrainMEC", mask.shape[0], notes, coverage=cov)
        return out, rep


class MaskMotionBlurMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": "Matte to smear — match motion blur already on the plate.",
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1,
                    "tooltip": "Blur direction in degrees, 0 = horizontal right.",
                }),
                "distance": ("INT", {
                    "default": 20, "min": 0, "max": 9999, "step": 1,
                    "tooltip": "Blur length in pixels along the angle.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flip mask polarity before blurring.",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Directional blur on a matte to match a moving plate."

    @classmethod
    def IS_CHANGED(cls, mask, angle, distance, invert, **kwargs):
        return hash_args_and_kwargs(mask, angle, distance, invert, **kwargs)

    def execute(
        self,
        mask: torch.Tensor,
        angle: float,
        distance: int,
        invert: bool,
    ) -> tuple[torch.Tensor, str]:
        mask = ensure_bhw_mask(mask, "MaskMotionBlurMEC")
        assert mask is not None
        notes: list[str] = []
        if invert:
            mask = 1.0 - mask
        out = clone_output(motion_blur_mask(mask, angle, distance, notes))
        cov = out.mean().item() * 100.0
        rep = build_report("MaskMotionBlurMEC", mask.shape[0], notes, coverage=cov)
        return out, rep


class EdgeSpreadMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Unpremultiplied plate — interior colour is pushed under the edge.",
                }),
                "spread": ("INT", {
                    "default": 4, "min": 0, "max": 9999, "step": 1,
                    "tooltip": (
                        "How many pixels to push inward colour outward — kills the dark "
                        "fringe on a lighter background."
                    ),
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flip mask polarity when your matte is white-on-black.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Matte defining the subject edge; alpha used if omitted."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Push interior plate colour outward under the matte edge. "
        "Output is an image, not a mask — fixes unpremult dark fringes."
    )

    @classmethod
    def IS_CHANGED(cls, image, spread, invert_mask, **kwargs):
        return hash_args_and_kwargs(image, spread, invert_mask, **kwargs)

    def execute(
        self,
        image: torch.Tensor,
        spread: int,
        invert_mask: bool,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str]:
        image = ensure_bhw4_image(image, "EdgeSpreadMEC")
        notes: list[str] = []
        if mask is None:
            if image.shape[-1] >= 4:
                matte = image[..., 3].clone()
            else:
                matte = torch.ones(image.shape[:-1], device=image.device, dtype=image.dtype)
        else:
            matte = ensure_bhw_mask(mask, "EdgeSpreadMEC")
            assert matte is not None
        (image, matte), b = align_mask_batch([image, matte])
        assert image is not None and matte is not None
        if invert_mask:
            matte = 1.0 - matte
        rgb = image[..., :3]
        outs = []
        for i in range(b):
            outs.append(edge_spread_image(rgb[i:i + 1], matte[i:i + 1], spread, notes))
        out = clone_output(torch.cat(outs, dim=0))
        rep = build_report("EdgeSpreadMEC", b, notes)
        return out, rep


# ── Mask Tools: keyers and matte operations in one node (L7.65 P03, owner ORDERS 11.4) ──────────────────────────
#
# One mode at a time. Eight former nodes are its engines and migrate here with their mode set (Mask From Color,
# Luminance Keyer, Mask Gradient, Mask Grain, Mask Motion Blur, Edge Spread, Mask Batch Combine, Shuffle); three modes
# are new: difference key, grade, and directional grow / shrink. Every control is prefixed with its mode and shown
# only in that mode (js/c2c_mode_widgets.js).

def _engine(name):
    if name == "luma_key":
        from ..luminance_keyer import LuminanceKeyerMEC
        return LuminanceKeyerMEC
    if name == "shuffle":
        from ..shuffle import ShuffleMEC
        return ShuffleMEC
    if name == "combine":
        from ..helpers.helpers import MaskBatchCombineMEC
        return MaskBatchCombineMEC
    return {"colour_key": MaskFromColorMEC, "gradient": MaskGradientMEC, "grain": MaskGrainMEC,
            "motion_blur": MaskMotionBlurMEC, "edge_spread": EdgeSpreadMEC}[name]


# mode label, key, the engine's parameters (None = a new mode, parameters declared below)
MASK_TOOL_MODES = (
    ("key: colour", "colour_key", ("color", "colorspace", "tolerance", "soft_falloff", "invert")),
    ("key: luma", "luma_key", ("mode", "low", "high", "gamma", "falloff", "invert", "channel", "low_soft", "high_soft",
                               "invert_key")),
    ("key: difference", "difference_key", None),
    ("grade", "grade", None),
    ("grow / shrink", "grow", None),
    ("combine", "combine", ("op",)),
    ("gradient", "gradient", ("width", "height", "gradient_type", "angle", "center_x", "center_y", "start", "end")),
    ("grain", "grain", ("amount", "seed", "grain_size", "invert")),
    ("motion blur", "motion_blur", ("angle", "distance", "invert")),
    ("edge spread", "edge_spread", ("spread", "invert_mask")),
    ("shuffle", "shuffle", ("out_R", "out_G", "out_B", "out_A", "premultiply_output")),
)
_NEW_MODE_PARAMS = {
    "difference_key": {
        "tolerance": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005,
                                "tooltip": "How far a pixel may differ from the clean plate and still count as "
                                           "background (0..1 of the RGB distance)."}),
        "softness": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.005,
                               "tooltip": "Width of the soft edge above the tolerance."}),
        "invert": ("BOOLEAN", {"default": False, "tooltip": "Key the unchanged part instead of what changed."}),
    },
    "grade": {
        "gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.01,
                           "tooltip": "Multiply the matte (above 1 pushes greys toward white)."}),
        "gamma": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.01,
                            "tooltip": "Bend the greys: above 1 lifts them, below 1 sinks them; black and white "
                                       "stay put."}),
        "clamp": ("BOOLEAN", {"default": True, "tooltip": "Keep the result inside 0..1."}),
        "invert": ("BOOLEAN", {"default": False, "tooltip": "Flip the matte at the end."}),
    },
    "grow": {
        "left": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                         "tooltip": "Pixels to grow the matte toward the left (negative shrinks its left edge)."}),
        "right": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                          "tooltip": "Pixels to grow toward the right (negative shrinks its right edge)."}),
        "up": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                       "tooltip": "Pixels to grow upward (negative shrinks its top edge)."}),
        "down": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1,
                         "tooltip": "Pixels to grow downward (negative shrinks its bottom edge)."}),
        "feather": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                            "tooltip": "Soften the result by this blur radius in pixels."}),
    },
}
_MODE_KEY = {label: key for label, key, _p in MASK_TOOL_MODES}


def mask_tool_widget(key: str, param: str) -> str:
    """colour_key + color -> colour_key_color, grain + grain_size -> grain_size, edge_spread + spread -> edge_spread."""
    last = key.rsplit("_", 1)[-1]
    if param == last:
        return key
    if param.startswith(last + "_"):
        return key + param[len(last):]
    return f"{key}_{param}"


def _directional(mask: torch.Tensor, left: int, right: int, up: int, down: int) -> torch.Tensor:
    """Grow (positive) or shrink (negative) a matte toward each side, grey-scale. Growing pads with nothing; shrinking
    repeats the frame edge, so a matte that runs off the frame is not eaten from outside it."""
    import torch.nn.functional as F

    x = mask.unsqueeze(1)                                         # [B,1,H,W]
    for n, axis, toward_low in ((left, 3, True), (right, 3, False), (up, 2, True), (down, 2, False)):
        if not n:
            continue
        k = abs(int(n))
        grow = n > 0
        # growing toward the low side looks at the k pixels on the high side, and the other way round; shrinking a
        # low edge looks at the k pixels on the low side
        pad_high = (grow and toward_low) or (not grow and not toward_low)
        if axis == 3:
            pad = (0, k, 0, 0) if pad_high else (k, 0, 0, 0)
            kernel = (1, k + 1)
        else:
            pad = (0, 0, 0, k) if pad_high else (0, 0, k, 0)
            kernel = (k + 1, 1)
        if grow:
            x = F.max_pool2d(F.pad(x, pad, mode="constant", value=0.0), kernel, stride=1)
        else:
            x = -F.max_pool2d(F.pad(-x, pad, mode="replicate"), kernel, stride=1)
    return x.squeeze(1)


class MaskToolsMEC:
    """Mask Tools: colour / luma / difference keys, matte grade, directional grow and shrink, combine, gradient, grain,
    motion blur, edge spread and channel shuffle - one mode at a time."""

    @classmethod
    def INPUT_TYPES(cls):
        import copy

        required = {"mode": ([label for label, _k, _p in MASK_TOOL_MODES], {
            "default": "key: colour",
            "tooltip": "What the node does. Each mode shows only its own controls."})}
        for label, key, params in MASK_TOOL_MODES:
            if params is None:
                spec = _NEW_MODE_PARAMS[key]
                names = list(spec)
            else:
                eng = _engine(key).INPUT_TYPES()
                spec = {**(eng.get("required") or {}), **(eng.get("optional") or {})}
                names = list(params)
            for p in names:
                s = copy.deepcopy(spec[p])
                opts = dict(s[1]) if len(s) > 1 else {}
                if key == "grain" and p == "seed":
                    opts["control_after_generate"] = True
                tip = opts.get("tooltip", "")
                opts["tooltip"] = f"{label}: {tip}" if tip else label
                required[mask_tool_widget(key, p)] = (s[0], opts)
        return {
            "required": required,
            "optional": {
                "image": ("IMAGE", {"tooltip": "keys, edge spread and shuffle: the picture to work on."}),
                "mask": ("MASK", {"tooltip": "grade, grow / shrink, combine (first matte), grain, motion blur; edge "
                                             "spread's matte; gradient's size when nothing else gives one."}),
                "mask_b": ("MASK", {"tooltip": "combine: the second matte."}),
                "plate": ("IMAGE", {"tooltip": "key: difference - the clean plate (same shot, without the subject)."}),
                "size_as": ("IMAGE", {"tooltip": "gradient: take the size from this image."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "image", "report")
    OUTPUT_TOOLTIPS = (
        "The matte (shuffle: the new alpha; edge spread: the matte it used).",
        "edge spread / shuffle: the processed picture. Other modes: the matte as a grey image, to look at.",
        "What the node did, and anything worth knowing (clamped values, resized inputs).",
    )
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Keyers and matte operations in one node: colour, luma and difference keys; grade (gain, gamma, clamp, "
        "invert); grow or shrink each side (left / right / up / down) with a feather; combine two mattes; gradient; "
        "grain; motion blur; edge spread; channel shuffle."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    @staticmethod
    def _need(value, what, mode):
        if value is None:
            raise ValueError(f"Mask Tools ({mode}): connect {what}.")
        return value

    def execute(self, mode="key: colour", image=None, mask=None, mask_b=None, plate=None, size_as=None, **kw):
        key = _MODE_KEY.get(mode)
        if key is None:
            raise ValueError(f"Mask Tools: unknown mode {mode!r}.")
        params = dict((k, p) for _l, k, p in MASK_TOOL_MODES)[key]
        names = list(params) if params is not None else list(_NEW_MODE_PARAMS[key])
        args = {p: kw[mask_tool_widget(key, p)] for p in names if mask_tool_widget(key, p) in kw}
        with torch.no_grad():
            out_mask, out_image, report = self._run(key, mode, args, image, mask, mask_b, plate, size_as)
        if out_image is None and out_mask is not None:
            m = out_mask if out_mask.dim() == 3 else out_mask.unsqueeze(0)
            out_image = clone_output(m.unsqueeze(-1).expand(*m.shape, 3))
        return out_mask, out_image, report

    def _run(self, key, mode, a, image, mask, mask_b, plate, size_as):
        if key == "colour_key":
            m, rep = _engine(key)().execute(self._need(image, "an image", mode), **a)
            return m, None, rep
        if key == "luma_key":
            m, rep = _engine(key)().key_luminance(self._need(image, "an image", mode), **a)
            return m, None, rep
        if key == "gradient":
            m, rep = _engine(key)().execute(size_as=size_as, mask=mask, **a)
            return m, None, rep
        if key in ("grain", "motion_blur"):
            m, rep = _engine(key)().execute(self._need(mask, "a mask", mode), **a)
            return m, None, rep
        if key == "edge_spread":
            img, rep = _engine(key)().execute(self._need(image, "an image", mode), mask=mask, **a)
            return mask, img, rep
        if key == "combine":
            (m,) = _engine(key)().combine(self._need(mask, "a mask", mode), self._need(mask_b, "mask_b", mode), **a)
            return m, None, f"Mask Tools: combine ({a.get('op')})"
        if key == "shuffle":
            img, alpha = _engine(key)().shuffle(self._need(image, "an image", mode), **a)
            return alpha, img, "Mask Tools: shuffle"
        if key == "difference_key":
            img = ensure_bhw4_image(self._need(image, "an image", mode), "Mask Tools")
            ref = ensure_bhw4_image(self._need(plate, "the clean plate", mode), "Mask Tools")
            notes: list[str] = []
            if ref.shape[1:3] != img.shape[1:3]:
                import torch.nn.functional as F
                ref = F.interpolate(ref.permute(0, 3, 1, 2), size=img.shape[1:3], mode="bilinear",
                                    align_corners=False).permute(0, 2, 3, 1)
                notes.append(f"plate resized to {img.shape[2]}x{img.shape[1]}")
            if ref.shape[0] != img.shape[0]:
                ref = ref[:1].expand(img.shape[0], *ref.shape[1:]) if ref.shape[0] == 1 else ref[: img.shape[0]]
            dist = ((img[..., :3] - ref[..., :3]) ** 2).sum(-1).sqrt() / (3 ** 0.5)
            tol, soft = float(a["tolerance"]), max(float(a["softness"]), 1e-6)
            m = ((dist - tol) / soft).clamp(0.0, 1.0)
            m = m * m * (3.0 - 2.0 * m)                                   # smoothstep edge
            if a["invert"]:
                m = 1.0 - m
            return clone_output(m), None, build_report("Mask Tools: difference key", img.shape[0], notes,
                                                     coverage=float(m.mean()))
        m = ensure_bhw_mask(self._need(mask, "a mask", mode), "Mask Tools")
        if key == "grade":
            g = m * float(a["gain"])
            g = g.clamp(min=0.0) ** (1.0 / float(a["gamma"]))
            if a["clamp"]:
                g = g.clamp(0.0, 1.0)
            if a["invert"]:
                g = 1.0 - g
            return clone_output(g), None, build_report("Mask Tools: grade", m.shape[0], [], coverage=float(g.mean()))
        if key == "grow":
            notes = []
            g = _directional(m.float(), a["left"], a["right"], a["up"], a["down"])
            if int(a["feather"]) > 0:
                from ..layer_effects._ops import blur_mask
                g = blur_mask(g, int(a["feather"]), notes)
            return clone_output(g.clamp(0.0, 1.0)), None, build_report("Mask Tools: grow / shrink", m.shape[0], notes,
                                                                        coverage=float(g.mean()))
        raise ValueError(f"Mask Tools: mode {mode!r} is not implemented.")


NODE_CLASS_MAPPINGS = {
    "MaskToolsMEC": MaskToolsMEC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToolsMEC": "Mask Tools (C2C)",
}
