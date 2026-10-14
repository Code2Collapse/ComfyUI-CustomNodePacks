# One crop node, one stitch node (L7.65 P08)

Owner card P08 (2026-10-09): "Inpaint Crop (takes Smart aspect and mask prepare as options) and Inpaint Stitch (takes
paste-back, colour match and lightness rescue). Five nodes migrate into the two."

Survivors keep their ids: `InpaintCropProMEC` (Inpaint Crop Pro) and `InpaintStitchProMEC` (Inpaint Stitch Pro).
Removed from the mappings, migrated on load: `SmartImageCropMEC`, `InpaintMaskPrepareMEC` (into the crop),
`SmartImageStitcherMEC`, `InpaintPasteBackMEC`, `MECContextInpainter` (into the stitch).

## 1. Inventory (as shipped at d97cc13)

| Node | Inputs (sockets; widgets) | Outputs |
|---|---|---|
| Inpaint Crop Pro | image (req), mask, optional_context_mask; 37 required widgets + roto_quality, preview_frame | stitcher, cropped_image, inpaint_mask, stitch_blend_mask, info (+ ui preview) |
| Smart Image Crop | image, mask (both req); resolution_mode, max_resolution, min_resolution, manual_width, manual_height, mask_grow_pixels, patch_mask_holes, no_mask_mode, force_divisibility | crop_image, crop_mask, stitcher, preview_overlay, info |
| Inpaint Mask Prepare | mask (req), reference_image; fill_holes, remove_small_regions, min_region_area, grow_pixels, inpaint_edge_mode, stitch_edge_mode, stitch_feather_radius, temporal_smooth, temporal_sigma | inpaint_mask, stitch_blend_mask, debug_preview, info |
| Inpaint Stitch Pro | stitcher, inpainted_image (both req); blend_mode_override, color_match, stitch_temporal_sigma, stitch_dilate_px, roto_quality_override | image, blend_mask_used, info |
| Smart Image Stitch | original_image, processed_image, stitcher (req), mask; feather_pixels, blend_mode, resize_full_image_output, enable_color_match, color_match_amount | image, info |
| Inpaint Paste Back | stitcher, inpainted_image (req); upscale_method, feather_edges, feather_radius | image, info |
| Context Inpainter / Fixer | original_image, mask, inpainted_image (req); crop_padding, blend_softness, mask_expansion_blend, enable_color_correction, enable_lightness_rescue, enable_differential_diffusion, sampling_mask_blur_size, sampling_mask_blur_strength, face_positive_prompt, face_negative_prompt | blended_image, debug_mask |

### The two STITCHER layouts (measured from the code)

* **Pro layout** (Inpaint Crop Pro; lquesada-compatible): `downscale_algorithm`, `upscale_algorithm`, `blend_pixels`,
  `blend_radius`, `stitch_blend_mode`, `wan_mask_polarity`, `device_mode`, `roto_quality`, and per-frame lists
  `canvas_to_orig_x/y/w/h`, `canvas_image` (the edge-padded original, CPU), `cropped_to_canvas_x/y/w/h`,
  `cropped_mask_for_blend`. It carries the original, so the stitch needs no original image.
  Consumers: Inpaint Stitch Pro, Inpaint Paste Back.
* **Smart layout** (Smart Image Crop): `bypass`, `original_size` (W, H), `full_image_resize`, and per-frame lists
  `x, y, w, h` (window in original pixels, may run past the border), `padL, padT`, `target_w, target_h`. It carries no
  pixels: the stitch needs `original_image`. Consumer: Smart Image Stitch.
* Context Inpainter takes no stitcher: it blends a full-frame inpainted image over the original under a mask, and only
  frame 0 of each input (it returns one frame).

No old graph mixed the layouts: a Smart stitcher into Stitch Pro / Paste Back raised `KeyError`, a Pro stitcher into
Smart Image Stitch raised "missing x, y, w, h". So a migrated graph always delivers the layout its new mode reads.

## 2. New options on the survivors (appended; defaults = the old behaviour)

### Inpaint Crop Pro

* `crop_mode`: `pro` (default, unchanged) / `smart aspect` / `mask prepare`.
* `smart_*` (9): the Smart Image Crop widgets, same specs and defaults, used only in `smart aspect`.
* `prepare_*` (9): the Mask Prepare widgets, same specs and defaults, used only in `mask prepare`.
* Output appended: `preview` (IMAGE, index 5). `smart aspect`: Smart's preview_overlay. `mask prepare`: Mask Prepare's
  debug_preview (R = inpaint mask, G = stitch blend mask). `pro`: the picture the node already draws for its preview
  (one frame: the window on the original next to the crop) - no extra work per run.
* `image` moves from required to the first optional input: a migrated Mask Prepare has no image. Frontend 1.53.10
  (read from its bundle, `addInputs`) adds socket inputs first, in definition order, so the sockets stay at slots
  image 0, mask 1, optional_context_mask 2; the widget order (positional `widgets_values`) does not change. `pro` and
  `smart aspect` raise a plain error without an image.

Mode outputs:

| Output | pro | smart aspect | mask prepare |
|---|---|---|---|
| 0 stitcher | Pro layout | Smart layout | Pro layout, whole frame (window = the image), blend = the prepared stitch mask; None without an image |
| 1 cropped_image | crop | crop_image | the image, uncropped; None without an image |
| 2 inpaint_mask | as before | crop_mask | inpaint_mask |
| 3 stitch_blend_mask | as before | crop_mask (the mask Smart's "Mask Feather" blends along) | stitch_blend_mask |
| 4 info | as before | Smart's info | Mask Prepare's info |
| 5 preview | the preview picture | preview_overlay | debug_preview |

### Inpaint Stitch Pro

* `stitch_mode`: `blend` (default, unchanged) / `paste back` / `smart` / `context blend`.
* `stitcher` and `inpainted_image` move from required to the first two optional inputs, in the same order (sockets
  stay at slots 0 and 1): a migrated Context Inpainter has no stitcher. A mode that needs one raises a plain error.
* Appended sockets `original_image`, `mask` (used by `smart` and `context blend`).
* `paste_*` (3), `smart_*` (5), `context_*` (8): the old nodes' widgets, same specs and defaults.
* No new outputs: image -> 0, the old info -> 2, Context Inpainter's debug_mask -> 1 (blend_mask_used).
  `blend_mask_used` in `paste back` / `smart` is the alpha that was applied, in original-image pixels (new there).

| Mode | Stitcher | Needs |
|---|---|---|
| blend | Pro layout | inpainted_image |
| paste back | Pro layout | inpainted_image |
| smart | Smart layout | inpainted_image, original_image (mask for "Mask Feather") |
| context blend | none (ignored if wired, and said so) | original_image, mask, inpainted_image |

A stitcher of the other layout raises an error that names the mode to pick (no silent fallback).

## 3. Migration rows (nodes/_legacy_replacements.json)

| Old | crop_mode / stitch_mode | Inputs | Outputs |
|---|---|---|---|
| SmartImageCropMEC | `smart aspect` | image, mask; each widget -> `smart_<name>` | 0->1, 1->2, 2->0, 3->5, 4->4 |
| InpaintMaskPrepareMEC | `mask prepare` | mask, reference_image -> image; each widget -> `prepare_<name>` | 0->2, 1->3, 2->5, 3->4 |
| SmartImageStitcherMEC | `smart` | original_image, processed_image -> inpainted_image, stitcher, mask; widgets -> `smart_<name>` | 0->0, 1->2 |
| InpaintPasteBackMEC | `paste back` | stitcher, inpainted_image; widgets -> `paste_<name>` | 0->0, 1->2 |
| MECContextInpainter | `context blend` | original_image, mask, inpainted_image; widgets -> `context_<name>` | 0->0, 1->1 |

Dropped: Context Inpainter's `face_positive_prompt` / `face_negative_prompt` (they were only written to the log; no
output depended on them).

## 4. What stays bit-equal, and how it is checked

The five old classes stay in the code, unregistered, as the engines of their modes (the P03 / P13 pattern; Face Fixer
already imports Context Inpainter's helpers). Their bodies are not edited. The survivor calls them with the migrated
values, so every mapped output is the old node's own result.

`tests/test_consolidation_inpaint.py` feeds each old node's inputs through its row (`migrate_kwargs`, `check` from
`test_consolidation_wave2.py`) into the survivor and compares every mapped output bit for bit with the old node as
shipped at d97cc13 (read with `git show` into a temp package; where the history is missing, as in a depth-1 CI clone,
the in-tree engine is the reference and a second test pins each engine's source to its d97cc13 hash). The JS
migration core is run under Node on a saved node of each old type to check that links and values land on the right
slots.

New code, checked against what executes:

* the crop preview (run and live) in `smart aspect` and `mask prepare` draws the windows the stitcher records;
  `plan_windows` (the live route) returns the same plan as a run for each new setting (tests next to the L7.99 ones);
* `blend_mask_used` in `paste back` / `smart` reproduces the engine's result (`out = a * inpainted + (1 - a) * plate`).

## 5. Live preview (L7.99 must not regress)

The cache keeps the masks as given (before Roto-Sync's 1 px erosion, which the live geometry now applies itself, so
toggling roto_quality, or switching mode, previews exactly). `_LIVE_KEYS` gains `crop_mode`, `roto_quality`,
`smart_*`, `prepare_*`. `plan_windows` branches on `crop_mode`: Smart's geometry comes from one pure function in
`nodes/smart_crop.py` (`plan_crop`), tested window-for-window against the engine's stitcher; `mask prepare` plans the
whole frame. Without an image (`mask prepare`) there is no picture, so nothing is cached and the route answers stale.

## 6. Front end

`js/c2c_mode_widgets.js` owns both nodes' visibility (one owner each). Crop: `crop_mode` shows its group; in `pro`
the L7.99 rules apply unchanged; in the other modes every Pro control (and its section bar) hides. Smart: manual size
only in Manual, min only in Automatic, max in Automatic or with "Resize Full Image". Prepare: min area only with
remove_small_regions, sigma only with temporal_smooth. Stitch: `stitch_mode` shows its group (feather radius only with
feather_edges, colour amount only with colour match). The new widgets carry no `c2c_section` / `advanced` flag: they
show with their mode, not behind "advanced". Works the same in both renderers (`setWidgetVisible` +
`vueSyncNodeWidgets` + `vueFitNodeHeight`) and inside subgraphs (the rule hooks every node instance by type).
`js/c2c_inpaint_crop_live.js` already sends every widget value; the server decides which ones it knows.

## 7. Failure modes

| Case | Behaviour |
|---|---|
| pro / smart aspect without an image | plain error: "connect an image" |
| mask prepare without a mask | plain error: "connect a mask" |
| smart aspect without a mask | treated as an empty mask, so `smart_no_mask_mode` decides (the old node required one) |
| stitcher of the other layout | plain error naming the stitch_mode to use |
| blend / paste back / smart without a stitcher | plain error: connect the crop's stitcher |
| smart without original_image, context blend without original_image or mask | plain error naming the input |
| context blend with a batch | only frame 0 is blended, as the old node did; info says how many frames were given |
| an old Inpaint Crop Pro / Stitch Pro save | unchanged: defaults pro / blend, widget order kept, sockets on the same slots |
| frontends that list inputs differently | sockets are re-matched by name on load (1.53.10 `configure`), so a slot shift would not lose a link there; older frontends unverified |

Budgets: no work is added to a `pro` / `blend` run (the preview picture is the one already drawn; the mode check is a
string compare). `mask prepare` with an image keeps references to the frames (no copy).
Kill switch: none needed for the backend (the default modes are the old code paths); the live preview keeps its
setting (C2C > Canvas > Inpaint crop > Live crop preview).
