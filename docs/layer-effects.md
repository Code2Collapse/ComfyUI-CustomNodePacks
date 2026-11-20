# Layer Effects (C2C) (`LayerEffectsMEC`)

**Menu:** 🐺 C2C/🧰 Core/Layer Effects · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#layereffectsmec)

No model, no extra package. The effects are ported from ComfyUI_LayerStyle by chflame163 (MIT) and run in torch on
the whole batch (91cd7e6).

## What it is for

Photoshop-style layer effects on a cut-out layer, in one node: drop shadow, outer glow, gradient map, gradient
overlay, colour overlay, inner glow, inner shadow and stroke. Switch on the effects you want; each is applied to the
result of the one before, in the order you set. The layer's shape comes from its alpha or from `layer_mask`.

## It replaced eight nodes

Since L7.65 P13 (8b9314d) the eight single-effect nodes are one node. Saved workflows with them load as Layer Effects
(C2C) with only that effect switched on and its values moved to the prefixed controls
([MIGRATION.md](MIGRATION.md)). The commit reports a bit-for-bit comparison of every old node against the new one.

| Former node | Effect switch in Layer Effects (C2C) |
|---|---|
| Layer Effect: Drop Shadow (MEC) (`LayerEffectDropShadowMEC`) | `drop_shadow` |
| Layer Effect: Outer Glow (MEC) (`LayerEffectOuterGlowMEC`) | `outer_glow` |
| Layer Effect: Gradient Map (MEC) (`LayerEffectGradientMapMEC`) | `gradient_map` |
| Layer Effect: Gradient Overlay (MEC) (`LayerEffectGradientOverlayMEC`) | `gradient_overlay` |
| Layer Effect: Color Overlay (MEC) (`LayerEffectColorOverlayMEC`) | `color_overlay` |
| Layer Effect: Inner Glow (MEC) (`LayerEffectInnerGlowMEC`) | `inner_glow` |
| Layer Effect: Inner Shadow (MEC) (`LayerEffectInnerShadowMEC`) | `inner_shadow` |
| Layer Effect: Stroke (MEC) (`LayerEffectStrokeMEC`) | `stroke` |

## Minimal wiring

```
cut-out subject (IMAGE) ──► layer_image ┐
its matte (MASK) ─────────► layer_mask  ├─► Layer Effects (C2C) ─ image ──────► Save Image
plate (IMAGE, optional) ──► background_image ┘                    ─ effect_mask ► (where the effects landed)
                                                                  ─ report
```

## Key controls

- One on / off switch per effect: `drop_shadow` (on by default), `outer_glow`, `gradient_map`, `gradient_overlay`,
  `color_overlay`, `inner_glow`, `inner_shadow`, `stroke`. An effect's controls are shown only while it is on. They
  carry its name as a prefix, for example `drop_shadow_distance_x`, `outer_glow_range`, `color_overlay_color`,
  `stroke_width`.
- `order` (in the **Order** section of the More bar): the switched-on effects, comma separated, applied in that order.
  The default is `drop_shadow, outer_glow, gradient_map, gradient_overlay, color_overlay, inner_glow, inner_shadow,
  stroke`. An effect that is on but not listed runs after the listed ones.
- `invert_mask` (on by default) flips the matte. A MASK from ComfyUI's Load Image is the inverted alpha, so on gives
  the subject back. Turn it off for a matte where white is the subject (most segmentation masks), and for an RGBA
  `layer_image` without `layer_mask`, whose alpha is flipped too.
- Every effect except Gradient map has `_blend_mode` (30 modes; the glows list the light modes first) and `_opacity`
  (0-100). `dissolve_seed` (Order section) is shown only while an effect that is on uses `dissolve`; a fixed seed
  keeps a dissolve from flickering through a sequence.
- Shadows: positive `_distance_x` moves the shadow right and positive `_distance_y` moves it down (the opposite of
  upstream LayerStyle). `_grow` spreads the shadow before the blur; negative pulls it inside.
- Colours are hex strings, `#RRGGBB` or `#RGB`.

## Reading the outputs

- `image`: the result, RGB. Without `background_image` the plate is a transparent canvas, and the RGB output shows it
  as black: the alpha is not returned.
- `effect_mask`: where the effects landed - the union of every switched-on effect's footprint (shadow, glow, stroke
  band; for the overlays and the gradient map, the matte itself).
- `report`: one line per effect applied, with notes such as "mask resized WxH -> WxH" or "... exceeded the frame;
  clamped to N (no visible difference beyond the frame size)". With no effect switched on it says "Layer Effects: no
  effect switched on - the layer over the background."

## Common mistakes

- No matte: "... needs a layer mask: connect layer_mask or use an RGBA layer_image with an alpha channel." An image
  from Load Image is RGB, so wire its MASK to `layer_mask`.
- The effect lands on the background instead of the subject: `invert_mask` is the wrong way round for your matte.
- A name in `order` that is not an effect: "Layer Effects: unknown effect(s) in order: ... Use names from: ...".
- A colour that is not hex: "Invalid hex colour '...'; use #RRGGBB or #RGB."
- Gradient map works differently from the others: it recolours the image it receives by brightness, inside the matte.
  When it runs first it reads `layer_mask` as it is (not inverted) and the whole frame without one, and it does not
  use `background_image`. With only Gradient map on, `background_image` is ignored.
