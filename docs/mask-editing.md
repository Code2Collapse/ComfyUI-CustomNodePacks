# Mask Tools (C2C)

One node for keying and matte operations: colour, luma and difference keys, a Nuke-style grade, grow or shrink per
side, combining two mattes, gradients, grain, motion blur, edge spread, channel shuffle and a per-axis transform. You
pick one `mode`; the node shows only that mode's controls. It loads no model.

For drawing a mask by hand (brush, shapes, splines, roto, points and boxes) use the
[Image Mask Editor](image-mask-editor.md). For mattes from a model use [Mask + Matting](mask-matting-pipeline.md) or
the [segmentation nodes](sam-segmentation.md).

## Mask Tools (`MaskToolsMEC`)
**Menu:** 🐺 C2C/🧰 Core/Mask · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#masktoolsmec)

### What it is for

Small matte jobs that would otherwise need a chain of nodes: pull a key from a plate, crush or lift a matte, push one
edge out, join two mattes, make a ramp. Every control carries its mode's name as a prefix (`grade_gain`,
`grow_left`, `colour_key_tolerance` ...), and only the controls of the chosen `mode` are shown. Hidden controls keep
their values and are saved with the workflow.

### Minimal wiring

```
                      ┌─ image  (keys, edge spread, shuffle)
Load Image ───────────┤
                      └─ plate  (key: difference - the same shot without the subject)
Any mask ──────────────► mask   (grade, grow / shrink, combine, grain, motion blur, transform)
Second mask ───────────► mask_b (combine)
                              Mask Tools ──► mask ─► (your inpaint / composite node)
                                         ──► image ─► Preview Image
                                         ──► report
```

A mode that needs a socket you did not connect stops with `Mask Tools (<mode>): connect an image.` (or `a mask`,
`mask_b`, `the clean plate`).

### The modes

| `mode` | What it does |
|---|---|
| `key: colour` | A matte from one colour (`colour_key_color`, `#RRGGBB`). `colour_key_tolerance` is how far a pixel may drift from that colour; `colour_key_soft_falloff` adds a graded band outside it. `colour_key_colorspace` measures the distance in `rgb`, `hsv` or `lab`. |
| `key: luma` | A matte from brightness. `luma_key_mode` = `auto` picks a range from the image, `highlights` / `midtones` / `shadows` use 0.7–1.0 / 0.3–0.7 / 0.0–0.3, `custom` uses `luma_key_low` / `luma_key_high` (these two show only in `custom`). `luma_key_falloff` softens the transition, `luma_key_gamma` bends the result. The Advanced section holds `luma_key_channel` (key on red, green, blue, hue, saturation, value or LAB `L` / `a` / `b` instead of luma), `luma_key_low_soft` / `luma_key_high_soft` and `luma_key_invert_key`. |
| `key: difference` | Compares `image` with a clean `plate` of the same shot. Pixels that differ by more than `difference_key_tolerance` (0..1 of the RGB distance) become white, with a soft edge `difference_key_softness` wide. A plate of another size is resized to the image; a plate with fewer frames than the clip holds its last frame for the rest. |
| `grade` | Grades a matte like Nuke's Grade node: `grade_blackpoint` becomes 0, `grade_whitepoint` becomes 1, `grade_lift` raises the blacks while white stays, `grade_gain` multiplies, `grade_offset` adds to every value, then `grade_gamma` bends the greys. `grade_clamp` keeps the result in 0..1, `grade_invert` flips it at the end. Black / white point, lift and offset were added on 2026-11-19 (commit ed8c5a3); at their defaults the grade gives the same result as before. |
| `grow / shrink` | Grows (positive) or shrinks (negative) the matte toward each side on its own: `grow_left`, `grow_right`, `grow_up`, `grow_down`, in pixels, then `grow_feather` blurs the result. Shrinking repeats the frame edge, so a matte that runs off the frame is not eaten from outside. Extra controls: `grow_angle` turns the four axes, `grow_units` switches to `% of the matte size`, `grow_falloff` gives the grown band a soft fall-off, and with Face Landmarks' `pose_keypoint` wired into `face_pose` the axes follow each frame's head tilt (`grow_follow_face`). |
| `combine` | Joins `mask` and `mask_b` with `combine_op`: `union` (the larger value), `intersect` (the smaller value), `diff` and `subtract` (both give A minus B, clamped), `xor` (the difference between A and B), `add` (A plus B, clamped). |
| `gradient` | A linear, radial or angular ramp (`gradient_type`, `gradient_angle`, `gradient_center_x` / `_y`, `gradient_start` / `_end`). The size comes from `size_as` when it is connected, otherwise from `gradient_width` / `gradient_height`. A connected `mask` is multiplied into the ramp, so you can fade a matte toward the frame edge. |
| `grain` | Adds noise to a matte to break up banding. `grain_amount` sets the strength, `grain_size` the coarseness. `grain_seed` has ComfyUI's control-after-generate menu: set it to `fixed` to keep the same grain on every run (each frame of a clip gets its own noise, drawn from that one seed). |
| `motion blur` | Blurs the matte in one direction: `motion_blur_angle` (0 = toward the right) and `motion_blur_distance` in pixels. |
| `edge spread` | Works on the picture, not the matte: pushes colour from inside the matte outward under its edge by `edge_spread` pixels, to remove a dark fringe. The matte is `mask`, or the image's alpha when no mask is connected. |
| `shuffle` | Rebuilds the channels: each of `shuffle_out_R` / `_G` / `_B` / `_A` takes `R`, `G`, `B`, `A`, `Lum`, an inverted channel (`InvR` ...), `zero` or `one`. The new alpha leaves on `mask`; `shuffle_premultiply_output` multiplies the colour by it. An image without alpha counts as alpha 1. |
| `transform` | The former Mask Transform XY: grow / erode per axis (`transform_expand_x` / `_y`), blur per axis (`transform_blur_x` / `_y`), shift (`transform_offset_x` / `_y`), then `transform_feather`, `transform_threshold` and `transform_invert`, in that order. |

### Reading the outputs

- `mask`: the result matte. In `shuffle` it is the new alpha. In `edge spread` it is the mask you connected, or an
  empty mask when none was connected.
- `image`: in `edge spread` and `shuffle`, the processed picture (RGB). In every other mode, the matte as a grey
  image, so you can look at it with Preview Image.
- `report`: one line, for example `Mask Tools: grade processed 24 frame(s). Coverage 37.5%.` (coverage = the mean
  matte value in percent), plus notes such as a resized plate, a held plate frame, or `frame 3: no face in face_pose -
  picture axes used`. `key: luma` writes several lines instead (mode used, thresholds, mean luminance, coverage per
  frame). `combine` and `shuffle` only name the mode.

### Common mistakes

- **The transform result is hard-edged.** `transform_threshold` defaults to 0.5 and binarises the matte. Set it to
  0 to keep soft values.
- **`combine` stops with `mask shapes differ`.** Both mattes must have the same frame count and size.
- **`grade` stops with "white point and black point are the same value".** Move one of them apart.
- **`key: luma` ignores `luma_key_low` / `luma_key_high`.** They are used only when `luma_key_mode` is `custom`;
  the presets set their own range.
- **The gradient ignores `gradient_width` / `gradient_height`.** With `size_as` connected, the size comes from that
  image (the report says so).
- **`grow_follow_face` does nothing.** It needs `face_pose` connected. A frame where no face is found uses the
  picture's axes, and the report lists it.

## Where did my old node go

These ids were merged. Saved workflows load them as the node and mode below, with their values and links.

| Old node | Now |
|---|---|
| Mask Transform XY (`MaskTransformXY`) | Mask Tools, `mode` = `transform` |
| Mask From Color (`MaskFromColorMEC`) | Mask Tools, `mode` = `key: colour` |
| Luminance Keyer (`LuminanceKeyerMEC`) | Mask Tools, `mode` = `key: luma` |
| Mask Gradient (`MaskGradientMEC`) | Mask Tools, `mode` = `gradient` |
| Mask Grain (`MaskGrainMEC`) | Mask Tools, `mode` = `grain` |
| Mask Motion Blur (`MaskMotionBlurMEC`) | Mask Tools, `mode` = `motion blur` |
| Edge Spread (`EdgeSpreadMEC`) | Mask Tools, `mode` = `edge spread` |
| Mask Batch Combine (`MaskBatchCombineMEC`) | Mask Tools, `mode` = `combine` |
| Shuffle (`ShuffleMEC`) | Mask Tools, `mode` = `shuffle` |
| Mask Edit (`MaskEditMEC`), mode `transform` | Mask Tools, `mode` = `transform` |
| Mask Draw Frame (`MaskDrawFrame`) | [Image Mask Editor](image-mask-editor.md#modes), `mode` = `shape`, `shape_input` = `JSON` |
| Draw Shape (`DrawShapeMEC`) | [Image Mask Editor](image-mask-editor.md#modes), `mode` = `shape`, `shape_input` = `parameters` |
| Spline Mask Editor (`SplineMaskEditorMEC`) | [Image Mask Editor](image-mask-editor.md#modes), `mode` = `spline`, `spline_fill` = `mask` |

The full list, including the input renames, is in [MIGRATION.md](MIGRATION.md).

Three older nodes are still registered so old workflows load, but are hidden from search and get no guide here:
Mask Composite Advanced ([`MaskCompositeAdvanced`](migration/MaskCompositeAdvanced.md)), Mask Math
([`MaskMath`](migration/MaskMath.md)) and Mask Preview Overlay ([`MaskPreviewOverlay`](migration/MaskPreviewOverlay.md)).

Mask Batch Manager (`MaskBatchManager`) is no longer part of the pack (removed in commit 71b56e8) and has no
automatic replacement.
