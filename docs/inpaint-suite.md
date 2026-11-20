# Inpaint Suite

Two nodes that work as a pair. **Inpaint Crop Pro** cuts the masked area out of a still or a clip, at the size your
inpaint model wants. Your inpaint workflow regenerates that crop. **Inpaint Stitch Pro** puts the result back into
the original frame. Neither node loads a model: they only crop, resize and blend.

| Node | Use it to |
|---|---|
| [Inpaint Crop Pro](#inpaint-crop-pro-inpaintcroppromec) | crop around a mask for inpainting (`pro`, `smart aspect`), or only clean a mask (`mask prepare`) |
| [Inpaint Stitch Pro](#inpaint-stitch-pro-inpaintstitchpromec) | put the inpainted crop back (`blend`, `paste back`, `smart`), or blend a full-frame inpaint over the original (`context blend`) |

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

Each mode runs the former node's own code. At the merge (7f7d161) every mode was compared bit for bit against the
former nodes' code. Since then `context blend` blends every frame of a batch and has a colour-reference choice
(b066d9c, see below).

## Inpaint Crop/Stitch Pro vs Video Inpaint (ProPainter)

- **Inpaint Crop Pro + Inpaint Stitch Pro**: no model of their own. Your inpaint model (any image or video model)
  fills the crop; a clip is cropped frame by frame (with one shared window when `wan_stable_crop` is on).
- **ProPainter — Temporal / Remove / Stitch / Refine / Flow (`ProPainterMEC`, [guide](propainter.md))**: fills a
  hole in a video from the clip's other frames. Its `temporal` / `remove` modes need the ProPainter code and weights
  (S-Lab License 1.0, non-commercial; not shipped), or its `backend` set to a model-free OpenCV fill (labelled
  APPROXIMATE; tested on a synthetic clip only, b9ec253). Its stitch modes accept Inpaint Crop Pro's `stitcher`, but
  only when the crop has the same size on every frame: keep `wan_stable_crop` on.

---

## Inpaint Crop Pro (`InpaintCropProMEC`)

**Menu:** 🐺 C2C/🧰 Core/Inpaint · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#inpaintcroppromec)

### What it is for

It finds the box around the mask, adds context around it, fits it to the shape and size you ask for, and hands the
crop to your inpaint model at that size. Small areas get the model's full resolution instead of being regenerated
at the size they have in the frame. The `stitcher` output remembers where every crop came from.

### When to pick it

- `crop_mode` = `pro` (default): stills and video. lquesada-style crop with Wan 2.2 Animate extras.
- `crop_mode` = `smart aspect`: stills. The output size lies between a minimum and a maximum side.
- `crop_mode` = `mask prepare`: no crop. It only cleans the mask and makes the two masks for a full-frame inpaint.
- For video hole filling from other frames, see ProPainter above.

### Minimal wiring

```
[Image] + [Mask] ─▶ Inpaint Crop Pro ── cropped_image, inpaint_mask ─▶ your inpaint workflow
                         │                                                   │
                         └── stitcher ──────────────────────────┐            │ inpainted image
                                                                ▼            ▼
                                     Inpaint Stitch Pro (stitcher, inpainted_image) ─▶ image
```

Which modes go together:

| Inpaint Crop Pro `crop_mode` | Inpaint Stitch Pro `stitch_mode` |
|---|---|
| `pro` (default) | `blend` (default) or `paste back` |
| `smart aspect` | `smart` (connect `original_image` too) |
| `mask prepare` (with an image) | `blend` - the stitcher covers the whole frame |
| — (no crop at all) | `context blend` (connect `original_image`, `mask`, `inpainted_image`) |

### Key controls

The node shows only the controls of the chosen `crop_mode`. Controls a switch enables (pre-resize sizes, outpaint
factors, the target size, the `video_stable_*` controls) appear when the switch is on. Rarely used controls sit in
the sections of the **More** bar at the bottom of the node: Mask, Size, Resize / outpaint, Video, Stitch, Device.

`pro`:
- `context_from_mask_extend_factor` grows the mask's box (1.2 = default); `auto_context_factor` picks it so the mask
  fills about 70 % of the crop. `optional_context_mask` adds an area the crop must include without regenerating it.
- `output_target_width` / `output_target_height` with `aspect_preset` set the crop size; the size is rounded to
  `output_padding` and `wan_align_multiple` (16 by default).
- `wan_stable_crop` (on by default) uses one window for every frame of a batch.
- `wan_mask_polarity`: `regenerate_subject` (white = regenerate) or `preserve_subject` (black = regenerate, Wan 2.2
  replacement mode).
- `inpaint_mask_mode` (what the model sees) and `stitch_blend_mode` / `blend_radius` (how the stitch blends:
  `gaussian`, `edge_aware`, `laplacian_pyramid`, `frequency_blend`, `video_stable`).
- `roto_quality`: tighter seam - the stitch uses `laplacian_pyramid` instead of `gaussian` / `edge_aware`, halves
  `blend_radius` (at least 4) and erodes the inpaint mask by 1 px.

`smart aspect` (the former Smart Image Crop): a mask value as low as 1/255 counts. Holes are closed and filled
(`smart_patch_mask_holes`); the mask is grown or shrunk (`smart_mask_grow_pixels`). The window is widened to the
output's aspect, never shrunk. In `Automatic` the output size lies between `smart_min_resolution` (shortest side)
and `smart_max_resolution` (longest side); in `Manual` it is `smart_manual_width` x `smart_manual_height`. Sizes snap
to `smart_force_divisibility`. A batch gets one output size, from its largest mask; each frame keeps its own window.
`smart_no_mask_mode` decides what an empty mask does: `Bypass` (the image passes through and the stitch does
nothing), `Resize Full Image`, or `Crop Full Image`.

`mask prepare` (the former Inpaint Mask Prepare): fills holes, removes regions smaller than
`prepare_min_region_area`, grows by `prepare_grow_pixels`, then makes the inpaint mask (`hard_binary` /
`slight_feather`) and the stitch blend mask (`gaussian` / `edge_aware`, radius `prepare_stitch_feather_radius`),
optionally smoothed over time. `image` is optional here. With an image, it is the reference for `edge_aware`, it is
passed through as `cropped_image`, and `stitcher` covers the whole frame for Inpaint Stitch Pro (`blend`).

### Reading the outputs

| # | Output | `pro` | `smart aspect` | `mask prepare` |
|---|---|---|---|---|
| 0 | `stitcher` | for `blend` / `paste back` | for `smart` | whole frame, for `blend` (none without an image) |
| 1 | `cropped_image` | the crop | the crop | the image, uncropped (none without an image) |
| 2 | `inpaint_mask` | the mask the model sees | the crop mask | the inpaint mask |
| 3 | `stitch_blend_mask` | the stitch's blend mask | the crop mask | the stitch blend mask |
| 4 | `info` | settings and frame 0's window | Smart Image Crop's report | Mask Prepare's report |
| 5 | `preview` | the preview picture | the image with the mask tinted and the window drawn | R = inpaint mask, G = stitch blend mask |

`info` in `pro` lists the input size, the pre-resize and mask settings, frame 0's mask box, the context factor (with
`AUTO →` the value it chose), the aspect preset and resulting target, the output size, padding and alignment, the Wan
settings, the mask and blend modes, the device, and `window[0]` with any `pad_*` where the window runs past the
image. In `smart aspect` the report says which hole filling ran, the output size and scale, and notes such as
"The masks in this batch are different sizes ..." or "N crop(s) were widened to match the output aspect".

**Preview.** After a run the node shows one picture of one frame (`preview_frame`): the original with the discarded
area dimmed, the regenerated area tinted and the window outlined (orange where it runs past the image), next to what
the model gets, with the numbers written under it. It is drawn from what ran, in every mode. Changing a crop setting
- including `crop_mode` and the smart / prepare controls - redraws it without a run, from the last run's masks
(marked "live - follows the settings; run to update the crop"). `mask prepare` without an image has no picture.

### Common mistakes

- No image in `pro` or `smart aspect`: "Inpaint Crop Pro: connect an image (crop_mode 'pro')." (or `'smart aspect'`).
- No mask in `mask prepare`: "Inpaint Crop Pro: connect a mask (crop_mode 'mask prepare')."
- `mask prepare` without an image but `stitcher` or `cropped_image` wired: the run stops with "... needs the image
  input. Connect the image, or use only the mask outputs."
- `smart aspect` with `smart_min_resolution` above `smart_max_resolution`: "Nothing can satisfy both."
- `smart aspect` without OpenCV: holes are filled with a torch flood fill that gives up on holes wider than about
  128 px; the report says so. Install `opencv-python` if a large hole survives.
- Using `preserve_subject` by accident: with it, black in the mask is what gets regenerated.

---

## Inpaint Stitch Pro (`InpaintStitchProMEC`)

**Menu:** 🐺 C2C/🧰 Core/Inpaint · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#inpaintstitchpromec)

### What it is for

It puts the inpainted crops back into the full frames, using the `stitcher` from Inpaint Crop Pro, and returns the
mask the blend used. In `context blend` it needs no stitcher: it blends a full-frame inpaint over the original.

### When to pick it

- `stitch_mode` = `blend` (default): after a `pro` or `mask prepare` crop. The crop's blend mask and blend mode (or
  `blend_mode_override`), optional mean + std `color_match`, optional temporal smoothing of the blend mask
  (`stitch_temporal_sigma`, `stitch_dilate_px`) for jittery video masks. Where the blend weight is 0 the original is
  kept unchanged, so nothing outside `blend_mask_used` changes.
- `stitch_mode` = `paste back` (the former Inpaint Paste Back): resizes the crop back to its window
  (`paste_upscale_method`) and pastes it; `paste_feather_edges` softens the rectangle's edge by
  `paste_feather_radius`.
- `stitch_mode` = `smart` (the former Smart Image Stitch): after a `smart aspect` crop. Needs `original_image`.
  `smart_blend_mode`: `Box Feather` (soft rectangle), `Mask Feather` (along `mask`) or `Hard Paste`.
  `smart_enable_color_match` pulls the crop toward the plate by `smart_color_match_amount`. After a
  `Resize Full Image` crop, `smart_resize_full_image_output` restores the original size or keeps the resized image.
- `stitch_mode` = `context blend` (the former Context Inpainter): no stitcher. Blends `inpainted_image` over
  `original_image` under `mask`: grow / shrink the blend mask (`context_mask_expansion_blend`), feather it
  (`context_blend_softness`), Reinhard mean / std colour correction, CIE LAB lightness rescue when the inpaint is
  more than about 5 % darker, optional differential diffusion (keep pixels that did not change).
  `context_color_reference` picks where the target colours come from: `inside mask (same subject)` or
  `around mask (object removal)` (the ring between the mask and its box grown by `context_crop_padding`). It blends
  every frame of a batch; a single image or mask is used for every frame. The old node's `face_positive_prompt` /
  `face_negative_prompt` only wrote to the log and were dropped.

### Minimal wiring

```
Inpaint Crop Pro ─ stitcher ──────────────► Inpaint Stitch Pro ─ image ─► Save Image
your inpaint workflow ─ decoded image ────► inpainted_image   ─ blend_mask_used
                                                              ─ info
```

`smart` also needs `original_image`; `context blend` needs `original_image`, `mask` and `inpainted_image`.

### Key controls

Only the chosen `stitch_mode`'s controls are shown. In `blend`, `stitch_temporal_sigma`, `stitch_dilate_px` and
`roto_quality_override` sit in the **Blend** section of the More bar. `roto_quality_override`: `from_crop` (the crop's
`roto_quality`), `force_on` or `force_off`. A `blend_mode_override` other than the crop's mode rebuilds the blend mask
from the stored mask's core (values above 0.5).

### Reading the outputs

- `image`: the full frames with the crops blended back.
- `blend_mask_used`: the mask the blend used, at full-frame size (in `paste back` and `smart`, the alpha that was
  applied; in `context blend`, the old node's debug mask, blurred by `context_sampling_mask_blur_*`).
- `info` (`blend`): frame count and device; `blend_mode: stored=... override=... -> effective=...`; `blend_radius`,
  `color_match`, `wan_polarity`; `temporal: sigma=... dilate=... applied=...`; the output size. `applied=False` with a
  sigma above 0 means the smoothing was skipped: it needs a clip whose crops all have one size (keep
  `wan_stable_crop` on).
- `info` (`context blend`): frames given and blended, the colour reference, the three switches, softness and
  expansion, and "the stitcher input is not used in this mode" when one is wired.

### Common mistakes

- A stitcher from the other kind of crop is refused with a message that names the stitch mode to pick:
  "this stitcher comes from Inpaint Crop Pro in 'smart aspect' mode - set stitch_mode to 'smart' and connect
  original_image", or "... in 'pro' or 'mask prepare' mode - set stitch_mode to 'blend' or 'paste back'".
- Missing inputs: "connect the stitcher output of Inpaint Crop Pro", "connect inpainted_image",
  "connect original_image (stitch_mode 'smart')", "connect mask (stitch_mode 'context blend')".
- `smart` with `Mask Feather` and no `mask`: Box Feather is used instead, and the info says so.
- An RGB inpaint over an RGBA original keeps the original's alpha. Other channel counts stop with "give both as RGB
  (3) or RGBA (4)".

---

## Interpolation filters (`pro` / `blend` / `paste back`)

| Name | Engine |
|---|---|
| `nearest`, `bilinear`, `bicubic`, `area` | torch `F.interpolate` |
| `lanczos`, `box`, `hamming` | OpenCV (8-bit) when installed (`box` and `hamming` use OpenCV's area filter), else bicubic |

`smart aspect` / `smart` resize like the former Smart nodes: bicubic up, area down, nearest for masks.
