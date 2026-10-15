# Inpaint Suite

> **Category:** `C2C/Inpaint`
> **VRAM Tier:** 1 (pure tensor math — no models loaded)
> **File:** [`nodes/inpaint_suite.py`](../nodes/inpaint_suite.py) (engines of two modes in
> [`nodes/smart_crop.py`](../nodes/smart_crop.py) and [`nodes/mec_paint_suite.py`](../nodes/mec_paint_suite.py))

Two nodes: **Inpaint Crop Pro** crops around the mask so the inpaint model works on the region at its own
resolution; **Inpaint Stitch Pro** puts the result back. The full parameter tables are in the main
[README](../README.md#inpainting) and in [NODE_REFERENCE.md](../NODE_REFERENCE.md#inpaintcroppromec).

Since L7.65 P08 (2026-10-15) the two nodes also do what five other nodes did. Saved workflows with those nodes load
migrated: the node becomes Inpaint Crop Pro / Inpaint Stitch Pro with the mode set, links and values kept
([MIGRATION.md](MIGRATION.md)). Design: [design/inpaint_crop_stitch_unified.md](design/inpaint_crop_stitch_unified.md).

| Former node | Now |
|---|---|
| Smart Image Crop (`SmartImageCropMEC`) | Inpaint Crop Pro, `crop_mode` = `smart aspect` |
| Inpaint Mask Prepare (`InpaintMaskPrepareMEC`) | Inpaint Crop Pro, `crop_mode` = `mask prepare` |
| Smart Image Stitch (`SmartImageStitcherMEC`) | Inpaint Stitch Pro, `stitch_mode` = `smart` |
| Inpaint Paste Back (`InpaintPasteBackMEC`) | Inpaint Stitch Pro, `stitch_mode` = `paste back` |
| Context Inpainter / Fixer (`MECContextInpainter`) | Inpaint Stitch Pro, `stitch_mode` = `context blend` |

Each mode runs the former node's own code, so the results are the same bit for bit
(`tests/test_consolidation_inpaint.py`, against the code as shipped at d97cc13).

---

## Pipeline

```
[Image] + [Mask] ─▶ Inpaint Crop Pro ── cropped_image, inpaint_mask ─▶ any inpaint model
                         │                                                   │
                         └── stitcher ──────────────────────────┐            │ inpainted
                                                                ▼            ▼
                                                          Inpaint Stitch Pro ─▶ result
```

Which modes go together:

| Inpaint Crop Pro `crop_mode` | Inpaint Stitch Pro `stitch_mode` |
|---|---|
| `pro` (default) | `blend` (default) or `paste back` |
| `smart aspect` | `smart` (connect `original_image` too) |
| `mask prepare` (with an image) | `blend` - the stitcher covers the whole frame |
| — (no crop at all) | `context blend` (connect `original_image`, `mask`, `inpainted_image`) |

A stitcher from the other kind of crop is refused with a message that names the stitch mode to pick.

---

## Inpaint Crop Pro

### crop_mode `pro`

The lquesada-style crop, video aware: optional pre-resize, mask clean-up (fill holes, expand, invert, feather,
hi-pass), optional outpaint extension, the context box (mask box grown by `context_from_mask_extend_factor`, plus
`optional_context_mask`), fit to the target aspect, rounded to `output_padding` / `wan_align_multiple`, resized to the
target size. `wan_stable_crop` (default on) uses one window for every frame of a batch. The window stays on real
pixels whenever it fits inside the image.

### crop_mode `smart aspect` (the former Smart Image Crop)

For stills. A mask value as low as 1/255 still counts; holes are closed and filled (`smart_patch_mask_holes`); the
mask is grown or shrunk (`smart_mask_grow_pixels`); the window is widened to the output's aspect (never shrunk); the
output size is chosen between `smart_min_resolution` (shortest side) and `smart_max_resolution` (longest side) in
`Automatic`, or is `smart_manual_width` x `smart_manual_height` in `Manual`, snapped to `smart_force_divisibility`. A
batch gets one output size, from its largest mask; each frame keeps its own window. `smart_no_mask_mode` decides what
an empty mask does: `Bypass` (the image passes through and the stitch does nothing), `Resize Full Image`, or
`Crop Full Image`.

### crop_mode `mask prepare` (the former Inpaint Mask Prepare)

No crop. Fills holes, removes regions smaller than `prepare_min_region_area`, grows by `prepare_grow_pixels`, then
makes the inpaint mask (`hard_binary` / `slight_feather`) and the stitch blend mask (`gaussian` / `edge_aware`, radius
`prepare_stitch_feather_radius`), optionally smoothed over time. `image` is optional here: it is the reference for
`edge_aware`, it is passed through as `cropped_image`, and with it `stitcher` is a whole-frame stitcher that Inpaint
Stitch Pro (`blend`) can use with the prepared blend mask. Without an image those two outputs are empty.

### Outputs

| # | Output | `pro` | `smart aspect` | `mask prepare` |
|---|---|---|---|---|
| 0 | `stitcher` | for `blend` / `paste back` | for `smart` | whole frame, for `blend` (empty without an image) |
| 1 | `cropped_image` | the crop | the crop | the image, uncropped |
| 2 | `inpaint_mask` | the mask the model sees | the crop mask | the inpaint mask |
| 3 | `stitch_blend_mask` | the stitch's blend mask | the crop mask | the stitch blend mask |
| 4 | `info` | settings and frame 0's window | Smart Image Crop's report | Mask Prepare's report |
| 5 | `preview` | the preview picture | the image with the mask tinted and the window drawn | R = inpaint mask, G = stitch blend mask |

### Preview

After a run the node shows one picture of one frame (`preview_frame`): the original with the discarded area dimmed,
the regenerated area tinted and the window outlined (orange where it runs past the image), next to what the model
gets, with the numbers written under it. It is drawn from what ran, in every mode. Changing a crop setting - including
`crop_mode` and the smart / prepare controls - redraws it without a run, from the last run's masks, with the node's
own geometry (marked "live"). `mask prepare` without an image has no picture.

---

## Inpaint Stitch Pro

### stitch_mode `blend`

The crop's blend mask, its blend mode (or `blend_mode_override`), optional mean + std colour match, optional temporal
smoothing of the blend mask for jittery video masks. Nothing outside `blend_mask_used` changes, bit for bit.

### stitch_mode `paste back` (the former Inpaint Paste Back)

Resize the inpainted crop back to its window (`paste_upscale_method`) and paste it; `paste_feather_edges` softens the
rectangle's edge by `paste_feather_radius`. `blend_mask_used` is the alpha that was applied.

### stitch_mode `smart` (the former Smart Image Stitch)

For a crop made in `smart aspect`. Needs `original_image`. `smart_blend_mode`: `Box Feather` (soft rectangle),
`Mask Feather` (along `mask`; without a mask it falls back to Box Feather and says so) or `Hard Paste`.
`smart_enable_color_match` pulls the crop toward the plate by `smart_color_match_amount`. After a `Resize Full Image`
crop, `smart_resize_full_image_output` restores the original size or keeps the resized image.

### stitch_mode `context blend` (the former Context Inpainter)

No stitcher. Blends a full-frame `inpainted_image` over `original_image` under `mask`: grow / shrink the blend mask
(`context_mask_expansion_blend`), feather it (`context_blend_softness`), Reinhard mean / std colour correction, CIE
LAB lightness rescue when the inpaint is more than about 5 % darker, optional differential diffusion (keep pixels
that did not change). `blend_mask_used` is the old node's debug mask (`context_sampling_mask_blur_*`). It blends frame
0 of each input only and returns one frame, as the old node did; the info output says how many frames were given.
The old node's `face_positive_prompt` / `face_negative_prompt` only wrote to the log and were dropped.

---

## Interpolation filters (`pro` / `blend` / `paste back`)

| Name | Engine |
|---|---|
| `nearest`, `bilinear`, `bicubic`, `area` | torch `F.interpolate` |
| `lanczos`, `box`, `hamming` | OpenCV (8-bit) when installed, else bicubic |

`smart aspect` / `smart` resize like the former Smart nodes: bicubic up, area down, nearest for masks.
