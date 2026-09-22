# The eight layer effects, ported from ComfyUI_LayerStyle by chflame163
# (MIT, https://github.com/chflame163/ComfyUI_LayerStyle): drop_shadow*,
# inner_shadow*, outer_glow*, inner_glow*, stroke*, color_overlay*,
# gradient_overlay* and gradient_map.py.
#
# One node per effect, carrying the UNION of upstream's V1/V2/V3
# variants, rather than three nodes that differ by one widget. Four
# defects in the source are fixed here and each has a test:
#   * distance_x == distance_y == 0 raised UnboundLocalError
#   * a mask of a different size was replaced with solid white, turning
#     the shadow into a full-frame rectangle
#   * a missing mask returned a python list through an IMAGE socket
#   * RGBA was returned on an IMAGE socket, which is [B,H,W,3]
# The sign of distance_x/distance_y is also deliberately NOT upstream's:
# there, positive moved the shadow left. See the tooltips.
"""MEC Layer Effect nodes — torch-native, batch-correct."""
from __future__ import annotations

import torch

from .._is_changed_util import hash_args_and_kwargs
from ._blend import BLEND_MODE_NAMES, GLOW_BLEND_MODE_NAMES, solid_rgba
from ._ops import (
    align_batch,
    alpha_composite,
    build_report,
    chop_layer,
    clone_output,
    ensure_bhw4_image,
    ensure_bhw_mask,
    expand_mask,
    gradient_ramp_report,
    image_to_rgba,
    inner_band_footprint,
    linear_gradient_bhwc,
    make_transparent_background,
    parse_hex_color,
    paste_rgb_layer,
    resolve_layer_mask,
    rgba_to_rgb,
    shift_mask,
    step_color_hex,
    step_value,
    subtract_masks,
)

_CATEGORY = "MEC/LayerEffects"

_DISSOLVE_SEED_TOOLTIP = (
    "Only used by the 'dissolve' blend mode. Upstream reseeds every frame, "
    "so a dissolve flickers through a sequence; a fixed seed here holds still."
)

_DISTANCE_X_TOOLTIP = (
    "Horizontal offset in pixels. Positive moves the shadow right — the opposite of "
    "the upstream LayerStyle node, which negated it."
)

_DISTANCE_Y_TOOLTIP = (
    "Vertical offset in pixels. Positive moves the shadow down — the opposite of "
    "the upstream LayerStyle node, which negated it."
)

_GROW_TOOLTIP = (
    "Expand the shadow before blurring. Positive spreads it past the subject edge; "
    "negative pulls it inside, which keeps a tight contact shadow from haloing."
)


def _dissolve_seed_widget() -> tuple:
    return ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF, "step": 1,
                     "tooltip": _DISSOLVE_SEED_TOOLTIP})


def _compositing_sockets(*, bg_optional: bool = True) -> dict:
    req = {
        "layer_image": ("IMAGE", {"tooltip": "Foreground layer; alpha or layer_mask defines the effect region."}),
        "invert_mask": ("BOOLEAN", {"default": True,
                                    "tooltip": "Flip mask polarity when your matte is white-on-black."}),
        "blend_mode": (BLEND_MODE_NAMES, {"tooltip": "How the effect colour blends with the plate."}),
        "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1,
                            "tooltip": "Effect strength after the blend mode is applied."}),
        "dissolve_seed": _dissolve_seed_widget(),
    }
    opt = {"layer_mask": ("MASK", {"tooltip": "Optional matte; overrides RGBA alpha when connected."})}
    if bg_optional:
        opt["background_image"] = ("IMAGE", {"tooltip": "Plate to composite onto; empty = transparent canvas."})
    else:
        req = {"background_image": ("IMAGE", {"tooltip": "Plate to composite onto."}), **req}
    return {"required": req, "optional": opt}


def _prepare_compositing(
    node_name: str,
    layer_image: torch.Tensor,
    background_image: torch.Tensor | None,
    layer_mask: torch.Tensor | None,
    invert_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], int]:
    layer_image = ensure_bhw4_image(layer_image, node_name)
    layer_mask = ensure_bhw_mask(layer_mask, node_name)
    notes: list[str] = []
    if background_image is None:
        b, h, w, _ = layer_image.shape
        background_image = make_transparent_background(
            b, h, w, device=layer_image.device, dtype=layer_image.dtype,
        )
    else:
        background_image = ensure_bhw4_image(background_image, node_name)
    (background_image, layer_image, layer_mask), b = align_batch(
        [background_image, layer_image, layer_mask],
    )
    assert background_image is not None and layer_image is not None
    masks = resolve_layer_mask(layer_image, layer_mask, invert_mask, node_name=node_name, notes=notes)
    bg_rgba = image_to_rgba(background_image)
    layer_rgba = image_to_rgba(layer_image)
    return bg_rgba, layer_rgba, masks, notes, b


def _finish(
    rgba: torch.Tensor,
    effect_mask: torch.Tensor,
    node_name: str,
    notes: list[str],
    batch: int,
    extra: str = "",
) -> tuple[torch.Tensor, torch.Tensor, str]:
    rgb = clone_output(rgba_to_rgb(rgba))
    em = clone_output(effect_mask)
    rep = build_report(node_name + (f" {extra}" if extra else ""), batch, notes)
    return rgb, em, rep


class LayerEffectDropShadowMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_image": ("IMAGE", {"tooltip": "Foreground layer whose alpha defines the cast shadow."}),
                "invert_mask": ("BOOLEAN", {"default": True,
                                            "tooltip": "Flip mask polarity when your matte is white-on-black."}),
                "blend_mode": (BLEND_MODE_NAMES,),
                "opacity": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "distance_x": ("INT", {"default": 25, "min": -9999, "max": 9999, "step": 1,
                                       "tooltip": _DISTANCE_X_TOOLTIP}),
                "distance_y": ("INT", {"default": 25, "min": -9999, "max": 9999, "step": 1,
                                       "tooltip": _DISTANCE_Y_TOOLTIP}),
                "grow": ("INT", {"default": 6, "min": -9999, "max": 9999, "step": 1, "tooltip": _GROW_TOOLTIP}),
                "blur": ("INT", {"default": 18, "min": 0, "max": 1000, "step": 1,
                                 "tooltip": "Gaussian blur radius on the shadow matte."}),
                "shadow_color": ("STRING", {"default": "#000000"}),
                "dissolve_seed": _dissolve_seed_widget(),
            },
            "optional": {
                "background_image": ("IMAGE", {"tooltip": "Plate; empty = transparent canvas (V3 union)."}),
                "layer_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "effect_mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Cast an outer drop shadow; effect_mask is the shadow footprint."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(
        self,
        layer_image,
        invert_mask,
        blend_mode,
        opacity,
        distance_x,
        distance_y,
        grow,
        blur,
        shadow_color,
        dissolve_seed=0,
        background_image=None,
        layer_mask=None,
    ):
        with torch.no_grad():
            return self._run(
                layer_image, invert_mask, blend_mode, opacity,
                distance_x, distance_y, grow, blur, shadow_color,
                dissolve_seed, background_image, layer_mask,
            )

    def _run(self, layer_image, invert_mask, blend_mode, opacity,
             distance_x, distance_y, grow, blur, shadow_color,
             dissolve_seed, background_image, layer_mask):
        name = "Layer Effect: Drop Shadow (MEC)"
        bg, layer, masks, notes, b = _prepare_compositing(
            name, layer_image, background_image, layer_mask, invert_mask,
        )
        sr, sg, sb = parse_hex_color(shadow_color)
        out_frames = []
        effect_frames = []
        for i in range(b):
            n: list[str] = []
            m = masks[i : i + 1]
            shifted = shift_mask(m, int(distance_x), int(distance_y))
            shadow_m = expand_mask(shifted, grow, blur, n)
            notes.extend(n)
            h, w = m.shape[-2], m.shape[-1]
            shadow_rgba = solid_rgba(
                1, h, w, (sr, sg, sb), 1.0,
                device=bg.device, dtype=bg.dtype,
            )
            canvas = bg[i : i + 1].clone()
            blended = chop_layer(
                canvas, shadow_rgba, blend_mode, opacity,
                dissolve_seed=dissolve_seed, batch_index=i,
            )
            canvas = alpha_composite(canvas, blended, shadow_m)
            canvas = alpha_composite(canvas, layer[i : i + 1], m)
            out_frames.append(canvas)
            effect_frames.append(subtract_masks(shadow_m, m))
        rgba = torch.cat(out_frames, dim=0)
        effect = torch.cat(effect_frames, dim=0).squeeze(1) if effect_frames[0].dim() == 4 else torch.cat(effect_frames, dim=0)
        if effect.dim() == 4:
            effect = effect.squeeze(1)
        return _finish(rgba, effect, name, notes, b)


class LayerEffectInnerShadowMEC(LayerEffectDropShadowMEC):
    DESCRIPTION = "Inner shadow along the layer edge; effect_mask is the shadow footprint."

    @classmethod
    def INPUT_TYPES(cls):
        import copy
        t = copy.deepcopy(LayerEffectDropShadowMEC.INPUT_TYPES())
        t["required"]["distance_x"][1]["default"] = 5
        t["required"]["distance_y"][1]["default"] = 5
        t["required"]["grow"][1]["default"] = 2
        t["required"]["blur"][1]["default"] = 15
        t["required"]["blur"][1]["max"] = 1000
        return t

    def _run(self, layer_image, invert_mask, blend_mode, opacity,
             distance_x, distance_y, grow, blur, shadow_color,
             dissolve_seed, background_image, layer_mask):
        name = "Layer Effect: Inner Shadow (MEC)"
        bg, layer, masks, notes, b = _prepare_compositing(
            name, layer_image, background_image, layer_mask, invert_mask,
        )
        sr, sg, sb = parse_hex_color(shadow_color)
        out_frames = []
        effect_frames = []
        for i in range(b):
            n: list[str] = []
            m = masks[i : i + 1]
            shifted = shift_mask(m, int(distance_x), int(distance_y))
            shadow_m = expand_mask(shifted, grow, blur, n)
            inner_m = (m * shadow_m).clamp(0, 1)
            notes.extend(n)
            h, w = m.shape[-2], m.shape[-1]
            shadow_rgba = solid_rgba(1, h, w, (sr, sg, sb), 1.0, device=bg.device, dtype=bg.dtype)
            layer_slice = layer[i : i + 1].clone()
            blended = chop_layer(
                layer_slice, shadow_rgba, blend_mode, opacity,
                dissolve_seed=dissolve_seed, batch_index=i,
            )
            comp = alpha_composite(layer_slice, blended, inner_m)
            bg_slice = rgba_to_rgb(bg[i : i + 1])
            layer_rgb = rgba_to_rgb(comp)
            final_rgb = paste_rgb_layer(bg_slice, layer_rgb, m.squeeze(0))
            alpha = m.unsqueeze(-1)
            final = torch.cat([final_rgb, alpha], dim=-1)
            out_frames.append(final)
            effect_frames.append((m * shadow_m).clamp(0.0, 1.0).clone())
        rgba = torch.cat(out_frames, dim=0)
        effect = torch.cat(effect_frames, dim=0).squeeze(1)
        return _finish(rgba, effect, name, notes, b)


class LayerEffectOuterGlowMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "blend_mode": (GLOW_BLEND_MODE_NAMES,),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "brightness": ("INT", {"default": 5, "min": 2, "max": 20, "step": 1,
                                       "tooltip": "Number of glow passes from core to edge."}),
                "glow_range": ("INT", {"default": 48, "min": -9999, "max": 9999, "step": 1,
                                       "tooltip": "Initial spread for the outer glow; clamped to frame size."}),
                "blur": ("INT", {"default": 25, "min": 0, "max": 9999, "step": 1}),
                "light_color": ("STRING", {"default": "#FFBF30"}),
                "glow_color": ("STRING", {"default": "#FE0000"}),
                "dissolve_seed": _dissolve_seed_widget(),
            },
            "optional": {
                "background_image": ("IMAGE",),
                "layer_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "effect_mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Outer glow halo; effect_mask is the glow footprint."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(self, layer_image, invert_mask, blend_mode, opacity, brightness,
                glow_range, blur, light_color, glow_color, dissolve_seed=0,
                background_image=None, layer_mask=None):
        with torch.no_grad():
            return self._glow(
                layer_image, invert_mask, blend_mode, opacity, brightness,
                glow_range, blur, light_color, glow_color, dissolve_seed,
                background_image, layer_mask, inner=False,
            )

    def _glow(self, layer_image, invert_mask, blend_mode, opacity, brightness,
              glow_range, blur, light_color, glow_color, dissolve_seed,
              background_image, layer_mask, *, inner: bool):
        name = "Layer Effect: Inner Glow (MEC)" if inner else "Layer Effect: Outer Glow (MEC)"
        bg, layer, masks, notes, b = _prepare_compositing(
            name, layer_image, background_image, layer_mask, invert_mask,
        )
        blur_factor = float(blur) / 20.0
        out_frames = []
        effect_frames = []
        for i in range(b):
            n: list[str] = []
            m = masks[i : i + 1]
            h, w = m.shape[-2], m.shape[-1]
            grow = int(glow_range)
            canvas = rgba_to_rgb(bg[i : i + 1]).clone()
            layer_work = rgba_to_rgb(layer[i : i + 1]).clone()
            painted = torch.zeros_like(m)
            for step in range(int(brightness)):
                step_blur = int(grow * blur_factor)
                color_hex = step_color_hex(glow_color, light_color, int(brightness), step + 1)
                cr, cg, cb = parse_hex_color(color_hex)
                step_notes: list[str] = []
                if inner:
                    eroded = expand_mask(
                        m, -grow, step_blur, step_notes, grow_label="glow_range",
                    )
                    gmask = (1.0 - eroded).clamp(0, 1)
                    step_footprint = inner_band_footprint(
                        m, grow, step_blur, step_notes, grow_label="glow_range",
                    )
                else:
                    expanded = expand_mask(
                        m, grow, step_blur, step_notes, grow_label="glow_range",
                    )
                    gmask = expanded
                    step_footprint = subtract_masks(expanded, m)
                painted = torch.maximum(painted, step_footprint)
                notes.extend(step_notes)
                color_rgba = solid_rgba(1, h, w, (cr, cg, cb), 1.0, device=bg.device, dtype=bg.dtype)
                op = int(step_value(1, opacity, int(brightness), step + 1))
                target = canvas if not inner else layer_work
                target_rgba = torch.cat([target, torch.ones((*target.shape[:-1], 1), device=target.device, dtype=target.dtype)], dim=-1)
                blended = chop_layer(
                    target_rgba, color_rgba, blend_mode, op,
                    dissolve_seed=dissolve_seed, batch_index=i * 100 + step,
                )
                if inner:
                    layer_work = paste_rgb_layer(layer_work, rgba_to_rgb(blended), gmask.squeeze(0))
                else:
                    canvas = paste_rgb_layer(canvas, rgba_to_rgb(blended), gmask.squeeze(0))
                grow = grow - int(glow_range / int(brightness))
            if inner:
                canvas = paste_rgb_layer(layer_work, rgba_to_rgb(bg[i : i + 1]), (1.0 - m).squeeze(0))
                final = torch.cat([canvas, m.unsqueeze(-1)], dim=-1)
            else:
                canvas = paste_rgb_layer(canvas, rgba_to_rgb(layer[i : i + 1]), m.squeeze(0))
                final = torch.cat([canvas, torch.ones((*canvas.shape[:-1], 1), device=canvas.device, dtype=canvas.dtype)], dim=-1)
            out_frames.append(final)
            effect_frames.append(painted)
        rgba = torch.cat(out_frames, dim=0)
        effect = torch.cat(effect_frames, dim=0).squeeze(1)
        return _finish(rgba, effect, name, notes, b)


class LayerEffectInnerGlowMEC(LayerEffectOuterGlowMEC):
    DESCRIPTION = "Inner glow inside the layer edge; effect_mask is the glow footprint."

    def execute(self, layer_image, invert_mask, blend_mode, opacity, brightness,
                glow_range, blur, light_color, glow_color, dissolve_seed=0,
                background_image=None, layer_mask=None):
        with torch.no_grad():
            return self._glow(
                layer_image, invert_mask, blend_mode, opacity, brightness,
                glow_range, blur, light_color, glow_color, dissolve_seed,
                background_image, layer_mask, inner=True,
            )


class LayerEffectStrokeMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "blend_mode": (BLEND_MODE_NAMES,),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "stroke_grow": ("INT", {"default": 0, "min": -999, "max": 999, "step": 1}),
                "stroke_width": ("INT", {"default": 8, "min": 0, "max": 999, "step": 1}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "stroke_color": ("STRING", {"default": "#FF0000"}),
                "dissolve_seed": _dissolve_seed_widget(),
            },
            "optional": {"background_image": ("IMAGE",), "layer_mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "effect_mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = "Outer stroke ring; effect_mask is the stroke band."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(self, layer_image, invert_mask, blend_mode, opacity, stroke_grow,
                stroke_width, blur, stroke_color, dissolve_seed=0,
                background_image=None, layer_mask=None):
        with torch.no_grad():
            name = "Layer Effect: Stroke (MEC)"
            bg, layer, masks, notes, b = _prepare_compositing(
                name, layer_image, background_image, layer_mask, invert_mask,
            )
            sr, sg, sb = parse_hex_color(stroke_color)
            grow_offset = int(stroke_width / 2)
            inner_stroke = int(stroke_grow) - grow_offset
            outer_stroke = inner_stroke + int(stroke_width)
            out_frames = []
            effect_frames = []
            for i in range(b):
                n: list[str] = []
                m = masks[i : i + 1]
                inner_m = expand_mask(m, inner_stroke, blur, n, grow_label="stroke_grow")
                outer_m = expand_mask(m, outer_stroke, blur, n, grow_label="stroke_grow")
                ring = subtract_masks(outer_m, inner_m)
                notes.extend(n)
                h, w = m.shape[-2], m.shape[-1]
                stroke_rgba = solid_rgba(1, h, w, (sr, sg, sb), 1.0, device=bg.device, dtype=bg.dtype)
                layer_slice = torch.cat([
                    rgba_to_rgb(layer[i : i + 1]),
                    torch.ones((1, h, w, 1), device=bg.device, dtype=bg.dtype),
                ], dim=-1)
                blended = chop_layer(
                    layer_slice, stroke_rgba, blend_mode, opacity,
                    dissolve_seed=dissolve_seed, batch_index=i,
                )
                canvas = rgba_to_rgb(bg[i : i + 1]).clone()
                canvas = paste_rgb_layer(canvas, rgba_to_rgb(layer[i : i + 1]), m.squeeze(0))
                canvas = paste_rgb_layer(canvas, rgba_to_rgb(blended), ring.squeeze(0))
                final = torch.cat([canvas, torch.ones((1, h, w, 1), device=bg.device, dtype=bg.dtype)], dim=-1)
                out_frames.append(final)
                effect_frames.append(ring)
            rgba = torch.cat(out_frames, dim=0)
            effect = torch.cat(effect_frames, dim=0).squeeze(1)
            return _finish(rgba, effect, name, notes, b)


class LayerEffectColorOverlayMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "blend_mode": (BLEND_MODE_NAMES,),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "color": ("STRING", {"default": "#FFBF30"}),
                "dissolve_seed": _dissolve_seed_widget(),
            },
            "optional": {"background_image": ("IMAGE",), "layer_mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "effect_mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Solid colour over the layer; effect_mask is the resolved layer mask "
        "(after invert and resize), not a separate halo."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(self, layer_image, invert_mask, blend_mode, opacity, color,
                dissolve_seed=0, background_image=None, layer_mask=None):
        with torch.no_grad():
            name = "Layer Effect: Color Overlay (MEC)"
            bg, layer, masks, notes, b = _prepare_compositing(
                name, layer_image, background_image, layer_mask, invert_mask,
            )
            cr, cg, cb = parse_hex_color(color)
            out_frames = []
            for i in range(b):
                m = masks[i : i + 1]
                h, w = m.shape[-2], m.shape[-1]
                overlay = solid_rgba(1, h, w, (cr, cg, cb), 1.0, device=bg.device, dtype=bg.dtype)
                layer_slice = torch.cat([
                    rgba_to_rgb(layer[i : i + 1]),
                    torch.ones((1, h, w, 1), device=bg.device, dtype=bg.dtype),
                ], dim=-1)
                comp = chop_layer(
                    layer_slice, overlay, blend_mode, opacity,
                    dissolve_seed=dissolve_seed, batch_index=i,
                )
                canvas = rgba_to_rgb(bg[i : i + 1]).clone()
                canvas = paste_rgb_layer(canvas, rgba_to_rgb(comp), m.squeeze(0))
                final = torch.cat([canvas, torch.ones((1, h, w, 1), device=bg.device, dtype=bg.dtype)], dim=-1)
                out_frames.append(final)
            rgba = torch.cat(out_frames, dim=0)
            return _finish(rgba, masks, name, notes, b)


class LayerEffectGradientOverlayMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": True}),
                "blend_mode": (BLEND_MODE_NAMES,),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "start_color": ("STRING", {"default": "#FFBF30"}),
                "start_alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "end_color": ("STRING", {"default": "#FE0000"}),
                "end_alpha": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "angle": ("INT", {"default": 0, "min": -180, "max": 180, "step": 1}),
                "dissolve_seed": _dissolve_seed_widget(),
            },
            "optional": {"background_image": ("IMAGE",), "layer_mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "effect_mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Linear gradient over the layer; effect_mask is the resolved layer mask. "
        "Ramp details appear in report for front-end display."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(
        self, layer_image, invert_mask, blend_mode, opacity,
        start_color, start_alpha, end_color, end_alpha, angle,
        dissolve_seed=0, background_image=None, layer_mask=None,
    ):
        with torch.no_grad():
            name = "Layer Effect: Gradient Overlay (MEC)"
            bg, layer, masks, notes, b = _prepare_compositing(
                name, layer_image, background_image, layer_mask, invert_mask,
            )
            sr, sg, sb = parse_hex_color(start_color)
            er, eg, eb = parse_hex_color(end_color)
            sa = float(start_alpha) / 255.0
            ea = float(end_alpha) / 255.0
            h, w = layer.shape[1], layer.shape[2]
            ramp_desc = gradient_ramp_report(
                start_color, end_color, start_alpha=start_alpha,
                end_alpha=end_alpha, angle=angle,
            )
            notes.append(ramp_desc)
            grad = linear_gradient_bhwc(
                1, h, w, (sr, sg, sb), (er, eg, eb), sa, ea, float(angle),
                device=layer.device, dtype=layer.dtype,
            )
            alpha_grad = linear_gradient_bhwc(
                1, h, w, (sa, sa, sa), (ea, ea, ea), 1.0, 1.0, float(angle),
                device=layer.device, dtype=layer.dtype,
            )
            comp_alpha = (1.0 - alpha_grad[..., 3:4]).squeeze(-1)
            out_frames = []
            for i in range(b):
                m = masks[i : i + 1]
                layer_slice = torch.cat([
                    rgba_to_rgb(layer[i : i + 1]),
                    torch.ones((1, h, w, 1), device=bg.device, dtype=bg.dtype),
                ], dim=-1)
                comp = chop_layer(
                    layer_slice, grad, blend_mode, opacity,
                    dissolve_seed=dissolve_seed, batch_index=i,
                )
                if start_alpha < 255 or end_alpha < 255:
                    comp_rgb = rgba_to_rgb(comp)
                    layer_rgb = rgba_to_rgb(layer[i : i + 1])
                    comp_rgb = comp_rgb * (1.0 - comp_alpha.unsqueeze(-1)) + layer_rgb * comp_alpha.unsqueeze(-1)
                    comp = torch.cat([comp_rgb, torch.ones((1, h, w, 1), device=bg.device, dtype=bg.dtype)], dim=-1)
                canvas = rgba_to_rgb(bg[i : i + 1]).clone()
                canvas = paste_rgb_layer(canvas, rgba_to_rgb(comp), m.squeeze(0))
                final = torch.cat([canvas, torch.ones((1, h, w, 1), device=bg.device, dtype=bg.dtype)], dim=-1)
                out_frames.append(final)
            rgba = torch.cat(out_frames, dim=0)
            return _finish(rgba, masks, name, notes, b)


class LayerEffectGradientMapMEC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image to remap by luminance."}),
                "start_color": ("STRING", {"default": "#015A52"}),
                "mid_color": ("STRING", {"default": "#02AF9F"}),
                "end_color": ("STRING", {"default": "#7FFFEC"}),
                "mid_point": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {"layer_mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "effect_mask", "report")
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Remap luminance through a three-stop gradient; effect_mask is the resolved "
        "layer mask when connected, otherwise full frame. Ramp in report for front-end."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(self, image, start_color, mid_color, end_color, mid_point, opacity, layer_mask=None):
        with torch.no_grad():
            name = "Layer Effect: Gradient Map (MEC)"
            image = ensure_bhw4_image(image, name)
            layer_mask = ensure_bhw_mask(layer_mask, name)
            notes: list[str] = []
            ramp_desc = gradient_ramp_report(
                start_color, end_color, mid_color=mid_color, mid_point=mid_point,
            )
            notes.append(ramp_desc)
            b, h, w, _ = image.shape
            if layer_mask is not None:
                (layer_mask,), _ = align_batch([layer_mask])
                fake_layer = image
                mask = resolve_layer_mask(
                    fake_layer, layer_mask, invert_mask=False,
                    node_name=name, notes=notes,
                )
            else:
                mask = torch.ones((b, h, w), device=image.device, dtype=image.dtype)
            sr, sg, sb = parse_hex_color(start_color)
            mr, mg, mb = parse_hex_color(mid_color)
            er, eg, eb = parse_hex_color(end_color)
            mid_idx = int(255 * float(mid_point))
            lut_len = 256
            t1 = torch.linspace(0, 1, max(mid_idx + 1, 1), device=image.device, dtype=image.dtype)
            t2 = torch.linspace(0, 1, max(lut_len - mid_idx, 1), device=image.device, dtype=image.dtype)
            g1 = torch.stack([
                sr + (mr - sr) * t1, sg + (mg - sg) * t1, sb + (mb - sb) * t1,
            ], dim=-1)
            g2 = torch.stack([
                mr + (er - mr) * t2, mg + (eg - mg) * t2, mb + (eb - mb) * t2,
            ], dim=-1)
            if g1.shape[0] > 0 and g2.shape[0] > 0:
                lut = torch.cat([g1[:-1] if g1.shape[0] > 1 else g1, g2], dim=0)
            else:
                lut = g2
            if lut.shape[0] < lut_len:
                pad = lut[-1:].expand(lut_len - lut.shape[0], -1)
                lut = torch.cat([lut, pad], dim=0)
            lut = lut[:lut_len]
            luma = (
                image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
            ).clamp(0, 1)
            idx = (luma * 255).long().clamp(0, lut_len - 1)
            mapped = lut[idx]
            orig_luma = luma.unsqueeze(-1)
            mixed = mapped * orig_luma + image[..., :3] * (1.0 - orig_luma)
            op = float(opacity) / 100.0
            out_rgb = image[..., :3] * (1.0 - op) + mixed * op
            out_rgb = out_rgb * mask.unsqueeze(-1) + image[..., :3] * (1.0 - mask.unsqueeze(-1))
            final = torch.cat([
                out_rgb,
                torch.ones((b, h, w, 1), device=image.device, dtype=image.dtype),
            ], dim=-1)
            return _finish(final, mask, name, notes, b)


NODE_CLASS_MAPPINGS = {
    "LayerEffectDropShadowMEC": LayerEffectDropShadowMEC,
    "LayerEffectInnerShadowMEC": LayerEffectInnerShadowMEC,
    "LayerEffectOuterGlowMEC": LayerEffectOuterGlowMEC,
    "LayerEffectInnerGlowMEC": LayerEffectInnerGlowMEC,
    "LayerEffectStrokeMEC": LayerEffectStrokeMEC,
    "LayerEffectColorOverlayMEC": LayerEffectColorOverlayMEC,
    "LayerEffectGradientOverlayMEC": LayerEffectGradientOverlayMEC,
    "LayerEffectGradientMapMEC": LayerEffectGradientMapMEC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerEffectDropShadowMEC": "Layer Effect: Drop Shadow (MEC)",
    "LayerEffectInnerShadowMEC": "Layer Effect: Inner Shadow (MEC)",
    "LayerEffectOuterGlowMEC": "Layer Effect: Outer Glow (MEC)",
    "LayerEffectInnerGlowMEC": "Layer Effect: Inner Glow (MEC)",
    "LayerEffectStrokeMEC": "Layer Effect: Stroke (MEC)",
    "LayerEffectColorOverlayMEC": "Layer Effect: Color Overlay (MEC)",
    "LayerEffectGradientOverlayMEC": "Layer Effect: Gradient Overlay (MEC)",
    "LayerEffectGradientMapMEC": "Layer Effect: Gradient Map (MEC)",
}
