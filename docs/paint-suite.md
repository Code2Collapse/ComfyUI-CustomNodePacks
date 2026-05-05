# Paint Suite (MEC)

> **Interactive paint canvas with Nuke-style procedural mask math + smart
> inpaint composite.** Two nodes that work together to draw a region and
> blend an inpaint back cleanly.

| Class | Display name | Purpose |
|---|---|---|
| `MECAdvancedPaintCanvas` | MEC Advanced Paint Canvas | Interactive brush widget that produces an image + a multi-stage processed mask |
| `MECContextInpainter` | MEC Context Inpainter | Smart-blend an inpainted region back over the original with color match / lightness rescue / differential diffusion |

Both live under `MEC/Paint`.

---

## MEC Advanced Paint Canvas

The JS widget posts a base64 PNG (RGBA) into the hidden `canvas_data`
string on every serialise. Python decodes it, optionally composites over
`reference_image`, and derives `processed_mask` from the alpha channel
through **four ordered stages**:

1. **Raw mask** — alpha of painted pixels normalised to `[0, 1]`.
2. **`mask_hardness`** — pixels brighter than `(1 − hardness)` clamp to
   `1.0` → produces a solid inner core whose width is the user-selected
   hardness fraction of the brush profile.
3. **`mask_expansion`** — morphological dilate (`>0`) / erode (`<0`) in
   pixels (cv2 ellipse kernel when available).
4. **`mask_blur`** — Gaussian blur of `mask_blur_radius` px, blended
   against the hard mask by `mask_blur_strength` (0 = hard, 1 = full
   blur).

### Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `canvas_width` / `canvas_height` | `512` | 64–4096 (step 8) | Canvas resolution |
| `brush_type` | `paint` | `paint` / `mask_only` | `paint` = composite color strokes; `mask_only` = build mask without coloring |
| `brush_color` | `#000000` | hex | Stroke color (ignored in `mask_only`) |
| `brush_opacity` | `1.0` | 0–1 | Stroke opacity |
| `brush_hardness` | `0.8` | 0–1 | Brush profile hardness (0 = soft, 1 = hard edge) |
| `brush_size` | `20` | 1–500 px | Brush diameter |
| `mask_hardness` | `0.5` | 0–1 | Threshold for solid inner core |
| `mask_expansion` | `0` | -100…100 px | Dilate (+) / erode (-) in pixels |
| `mask_blur_radius` | `0.0` | 0–100 px | Gaussian blur on edge |
| `mask_blur_strength` | `1.0` | 0–1 | Blend factor between hard mask (0) and fully blurred mask (1) |
| `canvas_data` | (hidden) | string | Internal base64 PNG payload — do not edit |

### Optional input

| Slot | Description |
|---|---|
| `reference_image` | Optional background image; painted strokes are composited over it when supplied |

### Outputs

| Output | Type | Description |
|---|---|---|
| `painted_image` | IMAGE | Painted RGB image (composited over `reference_image` when supplied) |
| `processed_mask` | MASK | Mask after hardness → expansion → blur stages |

### JS canvas controls

- **Left drag** — paint strokes
- **Right drag** — erase strokes
- **Scroll wheel** — adjust brush size
- **Shift + Scroll** — adjust brush hardness
- **Z** — undo · **Y** — redo
- **C** — clear all
- **R** — reset view

---

## MEC Context Inpainter

Smart-blend an inpainted region back over the original. The math runs in
this order:

1. **`crop_padding`** — extend the masked bbox by a multiplier so the
   inpaint sees context.
2. **`mask_expansion_blend`** — dilate / erode of the **blend** mask
   (not the inpaint mask).
3. **`blend_softness`** — Gaussian feather on the expanded mask.
4. **`enable_color_correction`** — Reinhard mean+std color match
   (per-channel, masked region).
5. **`enable_lightness_rescue`** — CIE LAB L-channel comparison; if the
   inpaint is more than ~5 % darker, lerp L upward by the deficit.
6. **`enable_differential_diffusion`** — `abs(orig − inpaint)` per
   pixel used as a soft preservation weight, so unchanged pixels stay
   sharp.
7. **`sampling_mask_blur_*`** — extra blur applied to the output
   `debug_mask` (used when feeding back into another sampler).

### Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `original_image` | — | IMAGE | Base for the blend |
| `mask` | — | MASK | Inpainted region |
| `inpainted_image` | — | IMAGE | Inpaint output |
| `crop_padding` | `1.2` | 1.0–2.0 | Multiplier extending masked bbox |
| `blend_softness` | `8.0` | 0–200 px | Gaussian feather on blend mask |
| `mask_expansion_blend` | `0` | -100–100 px | Dilate/erode the blend mask |
| `enable_color_correction` | `true` | bool | Reinhard color match |
| `enable_lightness_rescue` | `true` | bool | Lift CIE LAB L when inpaint is darker |
| `enable_differential_diffusion` | `false` | bool | Soft preservation weight |
| `sampling_mask_blur_size` | `21` | 0–201 (odd) | Kernel size for output `debug_mask` blur |
| `sampling_mask_blur_strength` | `1.0` | 0–1 | Blend factor for that blur |

### Optional (informational)

| Slot | Description |
|---|---|
| `face_positive_prompt` | Region-attached positive prompt; parsed for `{a\|b\|c}` wildcards per region and logged. **Does not sample here** — pair with a downstream sampler |
| `face_negative_prompt` | Same for negatives |

### Outputs

| Output | Type | Description |
|---|---|---|
| `blended_image` | IMAGE | Final composited image |
| `debug_mask` | MASK | Mask used for the blend (after expansion + blur) |

---

## Use cases

### Quick rotoscope mask

```
MEC Advanced Paint Canvas (mask_only) ──▶ ViTMatte Refiner ──▶ MASK out
```

Paint a rough region, refine to compositing-grade alpha. No segmenter
needed.

### Image generation — paint where to inpaint

```
Load Image ─▶ MEC Advanced Paint Canvas (paint mode, reference_image=Load Image)
              ├─ painted_image ──▶ Save (debug)
              └─ processed_mask ──▶ Inpaint Crop Pro
```

### Smart compositing of any inpaint result

```
Inpaint pipeline ─▶ inpainted_image ─┐
Original image ──────────────────────┼─▶ MEC Context Inpainter ─▶ blended_image
Mask ────────────────────────────────┘
```

Replaces ad-hoc lerp with feather + color match + LAB lightness rescue.
Use when an inpaint comes back slightly off in color or brightness.

---

## Recipes

| Goal | Settings |
|---|---|
| Soft mask for photoreal blend | `mask_hardness=0.3`, `mask_blur_radius=8`, `mask_blur_strength=1.0` |
| Hard graphic mask | `mask_hardness=1.0`, `mask_blur_radius=0` |
| Grow then feather | `mask_expansion=8`, `mask_blur_radius=4` |
| Blend with strong color shift correction | `enable_color_correction=true`, `enable_lightness_rescue=true`, `blend_softness=12` |
| Preserve unchanged pixels (small edits) | `enable_differential_diffusion=true`, `blend_softness=4` |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Empty `processed_mask` | Brush hardness too high + small strokes | Lower `mask_hardness`, raise `brush_size` |
| Visible halo around inpaint | Color shift not corrected | Enable `enable_color_correction` and `enable_lightness_rescue` |
| Inpaint looks flat | `enable_differential_diffusion` too aggressive | Disable it for global edits |
| Canvas resets between runs | `canvas_data` widget cleared | Don't refresh the page mid-edit; use Save Workflow |
