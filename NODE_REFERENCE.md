
# ComfyUI-CustomNodePacks — Node Reference

This pack registers **33 nodes**.  Each section lists the node's category, return types and every input widget exposed in ComfyUI.


## `BatchVersionManagerMEC` — Batch Version Manager (MEC)

> Compute (and optionally atomically reserve) the next v### directory under <root>/<show>/<shot>/<task>/. Forward-slash output paths.

- **Category:** `MaskEditControl/IO`
- **Returns:** `STRING, INT, STRING, STRING` → `version_path, version_int, version_label, info_json`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `root` | STRING | `""`  | Absolute output root (e.g. D:/projects/renders). |
| `show` | STRING | `"show"`  | Show / project name (top-level folder under root) |
| `shot` | STRING | `"sh010"`  | Shot identifier (folder under show) |
| `task` | STRING | `"comp"`  | Task name (folder under shot, e.g. comp, matte, render) |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `reserve` | BOOLEAN | `False`  | Atomically reserve the version with a .lock file. When False, only computes the path — no disk writes. |
| `padding` | INT | `3` (min 1, max 6) | Zero-pad width for v### (3 → v001, 4 → v0001). |
| `max_retries` | INT | `5` (min 1, max 50) | On lock-race contention, advance version and retry this many times. |
| `min_version` | INT | `1` (min 1, max 999999) | Floor for the first version when no v### exists yet. |
| `forward_slash` | BOOLEAN | `True`  | When True (default), output paths use forward slashes for cross-platform compatibility. Set False to keep native (Windows backslash) separators. |
| `write_manifest` | BOOLEAN | `True`  | When `reserve=True`, also write `version_manifest.json` alongside the .lock containing workflow_hash + user + host + timestamp + show/shot/task triple. Provides full audit trail. Ignored when reserve=False. |


## `FolderIncrementer` — Folder Version Incrementer

> Auto-incrementing per-label / per-date version counter. Scans the output directory for existing `vNNN` subfolders and emits the next one, plus folder name, subfolder path, filename prefix, and full output filename.

- **Category:** `utils`
- **Returns:** `STRING, INT, STRING, STRING, STRING, STRING` → `version_string, version_number, folder_name, subfolder_path, filename_prefix, output_filename`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `prefix` | STRING | `"v"`  | Prefix before the version number (e.g. 'v' → v001) |
| `padding` | INT | `3` (min 1, max 10) | Zero-pad width (3 → 001) |
| `label` | STRING | `"default"`  | Fallback folder name (used only when no source file is connected) |
| `date_format` | combo[MM-DD-YYYY, DD-MM-YYYY, YYYY-MM-DD] | `"MM-DD-YYYY"`  | Date format for the date subfolder (e.g. 02-22-2026 or 2026-02-22) |
| `path_style` | combo[auto, windows, linux, macos] | `"auto"`  | Path separator style for output strings. auto=detect from current OS, windows=backslash, linux/macos=forward slash. Use 'auto' unless you design workflows on one OS and run on another. |
| `source_choice` | combo[auto, image, video, custom] | `"auto"`  | Where the source name comes from. 'image' → trigger_image, 'video' → trigger_video, 'auto' → prefer video if connected, else image, else legacy `trigger`. 'custom' → use the ``custom_name`` widget verbatim and ignore all triggers. |
| `name_format` | combo[basename, strip_tags, first_segment] | `"basename"`  | How to format the detected filename for folder + prefix:   basename      — strip extension only (e.g. clip_2160_25fps)   strip_tags    — also strip trailing res/fps tags (clip)   first_segment — keep only the first chunk before . or _ (clip) The original file extension is preserved on output_filename. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `trigger` | * | `""`  | Legacy generic trigger – connect any output here. |
| `trigger_image` | IMAGE | `""`  | Connect a LoadImage / image source here. Used when source_choice = 'image' or 'auto'. |
| `trigger_video` | * | `""`  | Connect a LoadVideo / VHS_LoadVideo / video source here. Used when source_choice = 'video' or 'auto' (preferred). |
| `source_filename` | STRING | `""`  | Auto-filled by JS from the connected node. Drives folder name + output filename. |
| `custom_name` | STRING | `""`  | Manual source name. Only used when source_choice='custom'. May include an extension (e.g. 'my_shot.mp4'); if no extension is given, output_filename will have none either. Sanitized for cross-platform safety. |
| `base_path` | STRING | `""`  | Override base output directory.  Leave empty → ComfyUI output dir. |
| `folder_name_override` | STRING | `""`  | Force a specific folder name instead of deriving from the input filename. Sanitized for cross-platform safety. |
| `reserve_version` | BOOLEAN | `False`  | If True, create the version directory and write a `.reserved` marker file to claim the version number atomically. Prevents collisions in batch/render-farm workflows. Leave False for normal use (the directory will be created by ComfyUI's Save node when output is actually written). |


## `FolderIncrementerReset` — Folder Version Check

> Report the current version state for a label/date folder. Scans <base>/<label>/<MM-DD-YYYY>/ and returns how many vNNN folders already exist plus the highest version number. To truly 'reset' a label, delete its date subfolder from disk.

- **Category:** `utils`
- **Returns:** `STRING, INT` → `status, current_version`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `label` | STRING | `"default"`  | Folder name to inspect |
| `date_format` | combo[MM-DD-YYYY, DD-MM-YYYY, YYYY-MM-DD] | `"MM-DD-YYYY"`  | Date format (must match what FolderIncrementer uses) |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `trigger` | * | `""`  | Optional any-type trigger input. Connect any upstream output here to force this node to re-run after that node finishes (e.g. wire it to a SaveImage filename to recheck the version state after a save). |
| `base_path` | STRING | `""`  | Override base directory.  Leave empty → ComfyUI output dir. |


## `FolderIncrementerSet` — Folder Version Set

> Reserve version slots by creating empty placeholder directories under <base>/<label>/<MM-DD-YYYY>/. Creates v001 ... v{value} so the next FolderIncrementer run will produce v{value+1}. Useful for skipping ahead or reserving a known version range.

- **Category:** `utils`
- **Returns:** `STRING, INT` → `status, next_version`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `label` | STRING | `"default"`  | Folder name under the output directory |
| `value` | INT | `1` (min 1, max 999999) | Create placeholder dirs up to this version number |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `trigger` | * | `""`  | Optional any-type trigger input. Wire any upstream output here to control when this node runs in the graph. |
| `prefix` | STRING | `"v"`  | Version-folder prefix. Default 'v' produces v001, v002, ... Must match what FolderIncrementer is using. |
| `padding` | INT | `3` (min 1, max 10) | Zero-pad width for the version number (3 → v001, 4 → v0001). Must match what FolderIncrementer is using. |
| `base_path` | STRING | `""`  | Override base directory. Leave empty → ComfyUI output dir. |
| `date_format` | combo[MM-DD-YYYY, DD-MM-YYYY, YYYY-MM-DD] | `"MM-DD-YYYY"`  | Date format (must match what FolderIncrementer uses) |


## `InpaintCompositeMEC` — Inpaint Composite (MEC)

> Unified composite. mode=stitch_pro = lquesada feather blend with overrides; mode=paste_back = clean resize+paste.

- **Category:** `MaskEditControl/Inpaint`
- **Returns:** `IMAGE, MASK, STRING` → `image, mask, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `stitcher` | STITCHER | `""`  | Stitcher dict from InpaintCropProMEC |
| `inpainted_image` | IMAGE | `""`  | Inpainted result (B,H,W,C) |
| `mode` | combo[stitch_pro, paste_back] | `"stitch_pro"`  |  |
| `blend_mode_override` | combo[from_crop, gaussian, edge_aware, laplacian_pyramid, frequency_blend, video_stable] | `"from_crop"`  | [stitch_pro] Override blend mode or 'from_crop' |
| `color_match` | BOOLEAN | `False`  | [stitch_pro] Apply mean+std color transfer |
| `upscale_method` | combo[lanczos, bicubic, bilinear, nearest, area] | `"bicubic"`  | [paste_back] Interpolation for resize |
| `feather_edges` | BOOLEAN | `False`  | [paste_back] Gaussian-feather rectangle boundary |
| `feather_radius` | INT | `16` (min 0, max 64, step 1) | [paste_back] Feather radius in pixels (0 disables) |


## `InpaintCropProMEC` — Inpaint Crop Pro (MEC)

> Crop around mask for inpainting (lquesada API + Wan 2.2 Animate aware). Pair with Inpaint Stitch Pro (MEC).

- **Category:** `MaskEditControl/Inpaint`
- **Returns:** `STITCHER, IMAGE, MASK, MASK, STRING` → `stitcher, cropped_image, inpaint_mask, stitch_blend_mask, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `image` | IMAGE | `""`  |  |
| `downscale_algorithm` | combo[nearest, bilinear, bicubic, lanczos, box, hamming, area] | `"bilinear"`  |  |
| `upscale_algorithm` | combo[nearest, bilinear, bicubic, lanczos, box, hamming, area] | `"bicubic"`  |  |
| `preresize` | BOOLEAN | `False`  | Resize input image before processing (lquesada-style). |
| `preresize_mode` | combo[ensure minimum resolution, ensure maximum resolution, ensure minimum and maximum resolution] | `"ensure minimum resolution"`  |  |
| `preresize_min_width` | INT | `1024` (min 0, max 16384, step 1) |  |
| `preresize_min_height` | INT | `1024` (min 0, max 16384, step 1) |  |
| `preresize_max_width` | INT | `16384` (min 0, max 16384, step 1) |  |
| `preresize_max_height` | INT | `16384` (min 0, max 16384, step 1) |  |
| `mask_fill_holes` | BOOLEAN | `True`  | Mark fully-enclosed regions as masked. |
| `mask_expand_pixels` | INT | `0` (min 0, max 16384, step 1) | Dilate mask by this many pixels. |
| `mask_invert` | BOOLEAN | `False`  | Invert mask (anything masked is kept). |
| `mask_blend_pixels` | INT | `32` (min 0, max 64, step 1) | Pixels of feather for stitch blending (lquesada default 32). |
| `mask_hipass_filter` | FLOAT | `0.1` (min 0.0, max 1.0, step 0.01) | Zero out mask values below this threshold. |
| `extend_for_outpainting` | BOOLEAN | `False`  | Extend image with edge-replicated padding for outpainting. |
| `extend_up_factor` | FLOAT | `1.0` (min 0.01, max 100.0, step 0.01) |  |
| `extend_down_factor` | FLOAT | `1.0` (min 0.01, max 100.0, step 0.01) |  |
| `extend_left_factor` | FLOAT | `1.0` (min 0.01, max 100.0, step 0.01) |  |
| `extend_right_factor` | FLOAT | `1.0` (min 0.01, max 100.0, step 0.01) |  |
| `context_from_mask_extend_factor` | FLOAT | `1.2` (min 1.0, max 100.0, step 0.01) | Grow context bbox by this factor (1.5 = +50% on every side). |
| `output_resize_to_target_size` | BOOLEAN | `True`  | Force output to a specific resolution for sampling. |
| `output_target_width` | INT | `512` (min 64, max 16384, step 1) |  |
| `output_target_height` | INT | `512` (min 64, max 16384, step 1) |  |
| `output_padding` | combo[0, 8, 16, 32, 64, 128, 256, 512] | `"32"`  |  |
| `device_mode` | combo[cpu (compatible), gpu (much faster)] | `"gpu (much faster)"`  |  |
| `wan_align_multiple` | INT | `16` (min 1, max 256, step 1) | Force final crop W/H to multiples of this (Wan VAE patchify; 16 recommended). |
| `wan_temporal_smooth_frames` | FLOAT | `0.0` (min 0.0, max 64.0, step 0.1) | Gaussian smoothing of mask along time axis (frames). 0 disables. |
| `wan_stable_crop` | BOOLEAN | `True`  | Use a single union bbox across all frames (Wan replacement-mode). |
| `wan_mask_polarity` | combo[regenerate_subject, preserve_subject] | `"regenerate_subject"`  | regenerate_subject: mask=1 -> regenerate (lquesada).  preserve_subject: mask=0 -> regenerate (Wan2.2 replacement: mask=1 keeps environment). |
| `inpaint_mask_mode` | combo[hard_binary, slight_feather, soft_blend] | `"hard_binary"`  | What the inpaint sampler sees: hard_binary (crisp), slight_feather (gentle), soft_blend (very soft). |
| `stitch_blend_mode` | combo[gaussian, edge_aware, laplacian_pyramid, frequency_blend, video_stable] | `"gaussian"`  | How the result is composited back: gaussian, edge_aware (Sobel), laplacian_pyramid, frequency_blend, video_stable. |
| `blend_radius` | INT | `32` (min 1, max 256, step 1) | Feather radius for the stitch blend mask (independent of mask_blend_pixels). |
| `video_stable_temporal_sigma` | FLOAT | `3.0` (min 0.0, max 10.0, step 0.5) | [video_stable only] Temporal Gaussian sigma in frames. 3.0 ≈ 9-frame window. Higher = smoother but laggier on fast motion. 0 = off. |
| `video_stable_dilate_px` | INT | `-1` (min -1, max 128, step 1) | [video_stable only] Pixels to push the blend zone into background BEFORE feathering. -1 = derive from blend_radius. 16-32 typical. |
| `video_stable_blur_sigma` | FLOAT | `-1.0` (min -1.0, max 128.0, step 0.5) | [video_stable only] Spatial Gaussian sigma for the wide feather. -1 = derive from blend_radius (×0.75). Match to dilate value. |
| `fill_masked_area` | combo[none, edge_pad, neutral_gray, original] | `"none"`  | Fill masked region in the cropped image: none, edge_pad (Gaussian smear), neutral_gray, original. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `mask` | MASK | `""`  |  |
| `optional_context_mask` | MASK | `""`  |  |


## `InpaintPasteBackMEC` — Inpaint Paste Back — legacy (MEC)

> Paste inpainted crop back using STITCHER, with optional feathered rectangle edges.

- **Category:** `MaskEditControl/Inpaint`
- **Returns:** `IMAGE, STRING` → `image, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `stitcher` | STITCHER | `""`  | Stitcher dict from InpaintCropProMEC |
| `inpainted_image` | IMAGE | `""`  | Inpainted crop result (B,H,W,C) |
| `upscale_method` | combo[lanczos, bicubic, bilinear, nearest, area] | `"bicubic"`  | Interpolation method for resizing crop back |
| `feather_edges` | BOOLEAN | `False`  | Apply Gaussian feather at crop boundary |
| `feather_radius` | INT | `16` (min 0, max 64, step 1) | Feather radius in pixels (only used if feather_edges) |


## `InpaintStitchProMEC` — Inpaint Stitch Pro — legacy (MEC)

> Stitch inpainted image back into the original (lquesada-compatible) with blend overrides + color match.

- **Category:** `MaskEditControl/Inpaint`
- **Returns:** `IMAGE, MASK, STRING` → `image, blend_mask_used, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `stitcher` | STITCHER | `""`  |  |
| `inpainted_image` | IMAGE | `""`  |  |
| `blend_mode_override` | combo[from_crop, gaussian, edge_aware, laplacian_pyramid, frequency_blend, video_stable] | `"from_crop"`  | Override the blend mode chosen at crop time, or use 'from_crop'. |
| `color_match` | BOOLEAN | `False`  | Apply mean+std color transfer before stitching to reduce color shift. |
| `stitch_temporal_sigma` | FLOAT | `0.0` (min 0.0, max 10.0, step 0.5) | Post-hoc temporal Gaussian smoothing applied to the per-frame blend mask before compositing. 0 = off. 2-4 = good for jittery segmentation video (≈3 means a 9-frame window). Works on top of any blend mode (gaussian / edge_aware / video_stable / etc.). |
| `stitch_dilate_px` | INT | `0` (min 0, max 128, step 1) | Optional dilation (in pixels) applied to the binary core of the blend mask before temporal smoothing — pushes the seam into flat background. Use with stitch_temporal_sigma > 0 for jittery video. 0 = off. |


## `InsightStatusMEC` — Insight Status (MEC)

> Reports whether the Insight executor wrap is installed, plus the current torch/cuda memory snapshot.

- **Category:** `MaskEditControl/Diagnostic`
- **Returns:** `STRING` → `status`


## `IntegrityStatusMEC` — Integrity Status (MEC)

> Returns the latest integrity scan as a string.

- **Category:** `MaskEditControl/Diagnostic`
- **Returns:** `STRING` → `report`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `trigger_rescan` | BOOLEAN | `False`  |  |


## `MECAdvancedPaintCanvas` — Advanced Paint Canvas (MEC)

> Interactive paint canvas with procedural mask math: hardness, expansion, and blur stages are applied in order to the alpha channel of painted strokes.

- **Category:** `MEC/Paint`
- **Returns:** `IMAGE, MASK` → `painted_image, processed_mask`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `canvas_width` | INT | `512` (min 64, max 4096, step 8) | Canvas width in pixels. |
| `canvas_height` | INT | `512` (min 64, max 4096, step 8) | Canvas height in pixels. |
| `brush_type` | combo[paint, mask_only] | `"paint"`  | paint: composite color strokes onto the image. mask_only: build the mask without coloring. |
| `brush_color` | STRING | `"#000000"`  | Hex color used by the paint brush (ignored in mask_only mode). |
| `brush_opacity` | FLOAT | `1.0` (min 0.0, max 1.0, step 0.01) | Brush stroke opacity (0 = invisible, 1 = fully opaque). |
| `brush_hardness` | FLOAT | `0.8` (min 0.0, max 1.0, step 0.01) | Brush profile hardness (0 = soft, 1 = hard edge). |
| `brush_size` | INT | `20` (min 1, max 500, step 1) | Brush diameter in pixels. |
| `mask_hardness` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) | Threshold for the solid inner core: pixels brighter than (1 - hardness) clamp to 1.0. |
| `mask_expansion` | INT | `0` (min -100, max 100, step 1) | Morphological dilate (positive) / erode (negative) in pixels. |
| `mask_blur_radius` | FLOAT | `0.0` (min 0.0, max 100.0, step 0.1) | Gaussian blur radius applied to the mask edge in pixels. |
| `mask_blur_strength` | FLOAT | `1.0` (min 0.0, max 1.0, step 0.01) | Blend factor between the hard mask (0) and the fully-blurred mask (1). |
| `canvas_data` | STRING | `""`  | Internal base64 PNG payload from the JS canvas widget. Do not edit manually. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `reference_image` | IMAGE | `""`  | Optional background image; painted strokes are composited over it when supplied. |


## `MECBuilderSampler` — Builder Sampler (MEC)

> KSampler with adaptive CFG curves (Constant, Linear, Ease Down) plus an optional self-correction polish pass and resolution presets.

- **Category:** `MEC/Paint`
- **Returns:** `LATENT, IMAGE` → `latent, preview_image`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `model` | MODEL | `""`  | Diffusion model to sample with. |
| `positive` | CONDITIONING | `""`  | Positive conditioning. |
| `negative` | CONDITIONING | `""`  | Negative conditioning. |
| `steps` | INT | `20` (min 1, max 200) | Number of sampling steps. |
| `cfg` | FLOAT | `8.0` (min 0.0, max 30.0, step 0.1) | Starting CFG scale (also constant CFG when cfg_mode is Constant). |
| `sampler_name` | combo[euler, euler_cfg_pp, euler_ancestral, euler_ancestral_cfg_pp, heun, heunpp2, exp_heun_2_x0, exp_heun_2_x0_sde…] | `""`  | Sampler algorithm. |
| `scheduler` | combo[simple, sgm_uniform, karras, exponential, ddim_uniform, beta, normal, linear_quadratic…] | `""`  | Sigma schedule. |
| `denoise` | FLOAT | `1.0` (min 0.0, max 1.0, step 0.01) | Denoise strength (1.0 = full sampling, lower = partial img2img). |
| `cfg_mode` | combo[Constant, Linear, Ease Down] | `"Constant"`  | Adaptive CFG curve shape across steps. |
| `cfg_finish` | FLOAT | `4.0` (min 0.0, max 30.0, step 0.1) | Final CFG value at the end of the schedule (used by Linear / Ease Down). |
| `cfg_pivot` | FLOAT | `5.0` (min 0.0, max 30.0, step 0.1) | Pivot CFG value used by Ease Down to control the curve knee. |
| `self_correction` | BOOLEAN | `False`  | Run a 2-step polish pass after the main sampling. |
| `resolution_preset` | combo[SDXL (1024x1024), SD1.5 (512x512), Custom] | `"SD1.5 (512x512)"`  | Preset resolution; choose Custom to use custom_width/custom_height. |
| `custom_width` | INT | `512` (min 64, max 4096, step 8) | Custom output width in pixels (used when preset is Custom). |
| `custom_height` | INT | `512` (min 64, max 4096, step 8) | Custom output height in pixels (used when preset is Custom). |
| `seed` | INT | `0` (min 0, max 18446744073709551615) | Random seed. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `vae` | VAE | `""`  | Optional VAE; when provided, decodes the latent into preview_image. |
| `latent_image` | LATENT | `""`  | Optional input latent (img2img). When omitted, an empty latent is created. |


## `MECContextInpainter` — Context Inpainter / Fixer (MEC)

> Smart-blend an inpainted region back over the original image with crop padding, feathered blend mask, optional color correction, lightness rescue, and differential diffusion.

- **Category:** `MEC/Paint`
- **Returns:** `IMAGE, MASK` → `blended_image, debug_mask`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `original_image` | IMAGE | `""`  | Original (un-inpainted) image used as the blend base. |
| `mask` | MASK | `""`  | Mask defining the inpainted region. |
| `inpainted_image` | IMAGE | `""`  | Inpainted image to blend back over the original. |
| `crop_padding` | FLOAT | `1.2` (min 1.0, max 2.0, step 0.01) | Multiplier extending the masked bbox so the inpaint sees more context. |
| `blend_softness` | FLOAT | `8.0` (min 0.0, max 200.0, step 0.5) | Gaussian feather radius applied to the blend mask in pixels. |
| `mask_expansion_blend` | INT | `0` (min -100, max 100, step 1) | Per-blend dilate (positive) / erode (negative) of the blend mask in pixels. |
| `enable_color_correction` | BOOLEAN | `True`  | Reinhard mean/std color match between original and inpainted regions. |
| `enable_lightness_rescue` | BOOLEAN | `True`  | Lift CIE LAB L-channel of the inpaint when it is more than ~5% darker. |
| `enable_differential_diffusion` | BOOLEAN | `False`  | Use \|orig - inpaint\| as a soft preservation weight to keep unchanged pixels. |
| `sampling_mask_blur_size` | INT | `21` (min 0, max 201, step 1) | Kernel size (odd) for the additional blur on the output debug mask. |
| `sampling_mask_blur_strength` | FLOAT | `1.0` (min 0.0, max 1.0, step 0.01) | Blend factor for the sampling mask blur applied to debug_mask. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `face_positive_prompt` | STRING | `""`  | Optional region-attached positive prompt; parsed for `{a\|b\|c}` wildcards per detected mask region and logged. Does NOT sample here -- pair with a FaceInpaint/KSampler downstream. |
| `face_negative_prompt` | STRING | `""`  | Same as face_positive_prompt but for negatives. |


## `MECFaceFixer` — Face Fixer (MEC)

> Auto face detection (YOLO11) + per-face crop + AI pre-upscale + context-aware sampling + smart blend with wildcard per-face prompts. Behavioural clone of Forbidden Vision's Fixer with Impact-Pack wildcard syntax.

- **Category:** `MEC/Paint`
- **Returns:** `IMAGE, MASK, STRING` → `image, face_mask, info_json`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `image` | IMAGE | `""`  | Source image (single frame or batch; processed independently per frame). |
| `model` | MODEL | `""`  | Diffusion model to sample with. |
| `positive` | CONDITIONING | `""`  | Base positive conditioning. Wildcards in face_positive_prompt override per face. |
| `negative` | CONDITIONING | `""`  | Base negative conditioning. Wildcards in face_negative_prompt override per face. |
| `vae` | VAE | `""`  | VAE used to encode/decode the per-face crops. |
| `face_model` | combo[none] | `"none"`  | YOLO11 face-detection .pt/.onnx in ComfyUI/models/ultralytics/bbox/. Choose 'none' to use the optional mask input instead. |
| `confidence` | FLOAT | `0.5` (min 0.05, max 0.95, step 0.01) | Minimum detection confidence. |
| `max_faces` | INT | `8` (min 0, max 32, step 1) | Maximum number of faces to process per frame (0 = all). |
| `crop_padding` | FLOAT | `1.4` (min 1.0, max 3.0, step 0.05) | Bbox padding multiplier so the sampler sees context around each face. |
| `crop_resolution` | INT | `768` (min 256, max 2048, step 64) | Resize each face crop to this longer-side resolution before sampling. |
| `denoise` | FLOAT | `0.4` (min 0.0, max 1.0, step 0.01) | Per-face denoise strength (0.3 = subtle, 0.7 = aggressive reshape). |
| `steps` | INT | `20` (min 1, max 100) | Sampling steps per face. |
| `cfg` | FLOAT | `6.0` (min 0.0, max 30.0, step 0.1) | CFG scale for face sampling. |
| `sampler_name` | combo[euler, euler_cfg_pp, euler_ancestral, euler_ancestral_cfg_pp, heun, heunpp2, exp_heun_2_x0, exp_heun_2_x0_sde…] | `"euler"`  | Sampler algorithm. |
| `scheduler` | combo[simple, sgm_uniform, karras, exponential, ddim_uniform, beta, normal, linear_quadratic…] | `"normal"`  | Sigma schedule. |
| `seed` | INT | `0` (min 0, max 18446744073709551615) | Base seed; each face gets seed+i. |
| `blend_softness` | FLOAT | `6.0` (min 0.0, max 64.0, step 0.5) | Feather radius (px) on the per-face blend mask. |
| `mask_dilate` | INT | `4` (min -32, max 32, step 1) | Dilate (>0) / erode (<0) of the per-face blend mask. |
| `color_match` | BOOLEAN | `True`  | Reinhard mean/std colour match per face. |
| `lightness_rescue` | BOOLEAN | `True`  | Lift the per-face L channel if the sample comes back darker than the original. |
| `differential_diffusion` | BOOLEAN | `True`  | Weight the blend by \|orig - sampled\| so unchanged pixels stay sharp. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `mask` | MASK | `""`  | Optional manual face mask. Used directly when face_model='none' or detection finds nothing. |
| `upscale_model` | UPSCALE_MODEL | `""`  | Optional UPSCALE_MODEL applied to faces below crop_resolution before sampling. |
| `face_positive_prompt` | STRING | `""`  | Per-face positive prompt with wildcards: [SEP] separates faces; [ASC]/[DSC]/[ASC-SIZE]/[DSC-SIZE] order; [SKIP] leaves a face untouched. |
| `face_negative_prompt` | STRING | `""`  | Same syntax as face_positive_prompt for negatives. |


## `MECToneRefiner` — Tone Refiner (MEC)

> Auto-correct tone (black/white-point + gray-world), optionally upscale, and apply a fake center-focus depth-of-field blur.

- **Category:** `MEC/Paint`
- **Returns:** `IMAGE, LATENT` → `refined_image, refined_latent`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `image` | IMAGE | `""`  | Image to refine. |
| `neural_corrector` | BOOLEAN | `True`  | Enable deterministic tone + gray-world correction (not a learned model). |
| `corrector_tone` | FLOAT | `0.6` (min 0.0, max 1.0, step 0.01) | Blend amount toward the tone-corrected image (0 = original, 1 = full correction). |
| `corrector_color` | FLOAT | `0.4` (min 0.0, max 1.0, step 0.01) | Blend amount toward the gray-world color-corrected image. |
| `highlight_protection` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) | Roll-off applied above 95th-percentile to prevent highlight clipping (0 = none, 1 = strong). |
| `shadow_lift` | FLOAT | `0.0` (min 0.0, max 1.0, step 0.01) | Lift shadows below the 5th-percentile (0 = none, 1 = strong; mirrors highlight_protection on the dark side). |
| `enable_upscale` | BOOLEAN | `False`  | Upscale by upscale_factor; uses upscale_model if provided, bicubic otherwise. |
| `upscale_factor` | FLOAT | `1.5` (min 1.0, max 4.0, step 0.05) | Upscale multiplier applied when enable_upscale is True. |
| `ai_enable_dof` | BOOLEAN | `False`  | Apply depth-based DOF (uses depth_map if connected, else fake center-focus). |
| `ai_dof_strength` | FLOAT | `1.0` (min 0.0, max 4.0, step 0.05) | Strength of the DOF blur (also scales the maximum blur radius). |
| `ai_dof_focus_depth` | FLOAT | `0.7` (min 0.0, max 1.0, step 0.01) | Focus plane: when depth_map is connected this is the in-focus depth value (0=near,1=far); without depth_map it controls center-focus tightness. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `latent` | LATENT | `""`  | Optional pre-existing latent; passed through when supplied (skips VAE encode). |
| `vae` | VAE | `""`  | VAE used to encode the refined image into refined_latent (when auto_upscale is True). |
| `upscale_model` | UPSCALE_MODEL | `""`  | Optional UPSCALE_MODEL (RealESRGAN / 4x-NMKD / etc.). When connected and enable_upscale is True, used instead of bicubic and resized to upscale_factor. |
| `depth_map` | MASK | `""`  | Optional depth map (0=near,1=far). Drives DOF when connected; replaces the fake center-focus radial gradient. |
| `auto_upscale` | BOOLEAN | `True`  | When True (default, back-compat), encode the refined image through the supplied VAE to produce `refined_latent`. Set False to skip the VAE-encode and return a zero placeholder. |


## `MaskEditMEC` — Mask Edit — Transform/Draw/Points/BBox (MEC)

> Unified mask edit dispatcher. Pure-CPU. Modes: transform, draw_shape, draw_advanced, points_bbox, bbox_smooth. Pick a mode and the corresponding widgets drive that engine. All outputs are normalized to the same 8-port schema.

- **Category:** `MaskEditControl/Edit`
- **Returns:** `MASK, STRING, STRING, BBOX, BBOX, STRING, STRING, BBOX` → `mask, positive_coords, negative_coords, bboxes, neg_bboxes, points_json, bbox_json, primary_bbox`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `mode` | combo[transform, draw_shape, draw_advanced, points_bbox, bbox_smooth] | `"transform"`  | transform: morph / blur / offset / feather / threshold (needs mask). draw_shape: pick a shape from a dropdown, set its params (12 shapes). draw_advanced: power-mode shape_params_json (raw JSON). points_bbox: interactive points + bbox canvas (SAM/SeC coords). bbox_smooth: temporally smooth a sequence of [x,y,w,h] boxes. |
| `expand_x` | INT | `0` (min -512, max 512, step 1) | [transform] dilate/erode along X |
| `expand_y` | INT | `0` (min -512, max 512, step 1) | [transform] dilate/erode along Y |
| `blur_x` | FLOAT | `0.0` (min 0.0, max 128.0, step 0.5) | [transform] Gaussian sigma X |
| `blur_y` | FLOAT | `0.0` (min 0.0, max 128.0, step 0.5) | [transform] Gaussian sigma Y |
| `offset_x` | INT | `0` (min -4096, max 4096, step 1) | [transform] pixel shift X |
| `offset_y` | INT | `0` (min -4096, max 4096, step 1) | [transform] pixel shift Y |
| `feather` | FLOAT | `0.0` (min 0.0, max 128.0, step 0.5) | [transform/draw] feather radius |
| `threshold` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) | [transform] binarize threshold |
| `invert` | BOOLEAN | `False`  | [transform] invert output |
| `width` | INT | `512` (min 1, max 16384) | [draw_*/points_bbox] canvas width |
| `height` | INT | `512` (min 1, max 16384) | [draw_*/points_bbox] canvas height |
| `shape` | combo[circle, rectangle, ellipse, polygon, line, triangle, star, diamond…] | `"circle"`  | [draw_shape] geometry |
| `cx` | FLOAT | `256.0` (min -16384.0, max 16384.0, step 0.5) |  |
| `cy` | FLOAT | `256.0` (min -16384.0, max 16384.0, step 0.5) |  |
| `radius` | FLOAT | `50.0` (min 0.0, max 8192.0, step 0.5) |  |
| `size_w` | FLOAT | `200.0` (min 0.0, max 16384.0, step 0.5) |  |
| `size_h` | FLOAT | `100.0` (min 0.0, max 16384.0, step 0.5) |  |
| `rx` | FLOAT | `100.0` (min 0.0, max 8192.0, step 0.5) |  |
| `ry` | FLOAT | `50.0` (min 0.0, max 8192.0, step 0.5) |  |
| `top_left_x` | FLOAT | `100.0` (min -16384.0, max 16384.0, step 0.5) |  |
| `top_left_y` | FLOAT | `100.0` (min -16384.0, max 16384.0, step 0.5) |  |
| `x2` | FLOAT | `400.0` (min -16384.0, max 16384.0, step 0.5) |  |
| `y2` | FLOAT | `400.0` (min -16384.0, max 16384.0, step 0.5) |  |
| `thickness` | FLOAT | `5.0` (min 0.0, max 500.0, step 0.5) |  |
| `outer_r` | FLOAT | `100.0` (min 0.0, max 8192.0, step 0.5) |  |
| `inner_r` | FLOAT | `40.0` (min 0.0, max 8192.0, step 0.5) |  |
| `num_points` | INT | `5` (min 3, max 50) |  |
| `corner_radius` | FLOAT | `20.0` (min 0.0, max 4096.0, step 0.5) |  |
| `cross_size` | FLOAT | `100.0` (min 0.0, max 8192.0, step 0.5) |  |
| `arrow_length` | FLOAT | `200.0` (min 0.0, max 16384.0, step 0.5) |  |
| `head_length` | FLOAT | `60.0` (min 0.0, max 8192.0, step 0.5) |  |
| `head_width` | FLOAT | `80.0` (min 0.0, max 8192.0, step 0.5) |  |
| `points_json_shape` | STRING | `"[[100,100],[400,100],[400,400],[100,400]]"`  | [draw_shape] polygon vertices when shape=polygon |
| `value` | FLOAT | `1.0` (min 0.0, max 1.0, step 0.01) | [draw_*] fill intensity |
| `rotation` | FLOAT | `0.0` (min -360.0, max 360.0, step 0.5) | [draw_*] rotation deg |
| `operation` | combo[set, add, subtract, max, min] | `"set"`  | [draw_*] blend op |
| `batch_size` | INT | `1` (min 1, max 256) | [draw_shape] number of frames |
| `shape_params_json` | STRING | `"{"cx": 256, "cy": 256, "radius": 50}"`  | [draw_advanced] raw shape_params JSON (see MaskDrawFrame) |
| `editor_data` | STRING | `"{"points":[],"bboxes":[]}"`  | [points_bbox] JSON from the interactive canvas |
| `default_radius` | FLOAT | `3.0` (min 0.5, max 256.0, step 0.5) | [points_bbox] default brush radius |
| `softness` | FLOAT | `1.0` (min 0.0, max 10.0, step 0.1) | [points_bbox] gaussian sigma multiplier |
| `normalize` | BOOLEAN | `True`  | [points_bbox] clamp output to [0,1] |
| `bboxes_json` | STRING | `"[]"`  | [bbox_smooth] JSON array of [x,y,w,h] per frame |
| `smoothing_radius` | INT | `3` (min 1, max 30, step 1) | [bbox_smooth] window radius |
| `smoothing_method` | combo[median_then_exponential, moving_average, exponential, median] | `"median_then_exponential"`  | [bbox_smooth] smoothing strategy |
| `alpha` | FLOAT | `0.3` (min 0.05, max 1.0, step 0.05) | [bbox_smooth] exponential factor |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `mask` | MASK | `""`  | Source mask (transform requires this) |
| `reference_image` | IMAGE | `""`  | Optional reference; canvas size matches it when supplied |
| `existing_mask` | MASK | `""`  | Existing mask to blend onto in draw modes |


## `MaskOpsMEC` — Mask Ops — Seg + Matte + Refine + Diagnose (MEC)

> Production-grade segmentation + matting + refine + diagnostics in one node. Multi-backend (SAM 2.1 / SAM 3 / SAM 3.1 / BiRefNet / RMBG-2.0 / InSPyReNet + ViTMatte / RVM / MatAnyone). Optional Nuke-style luma-key pre-stage, edge-aware trimap, 11-stage training-free mask refinement (hole-fill → morph → thin-recover → joint-bilateral → guided → DenseCRF → edge-snap → cascade → feather → gamma → threshold), and automatic mask-failure diagnostics with severity score + suggested method. Replaces the standalone MaskMattingMEC, MaskRefineMEC, TrimapGeneratorMEC, LuminanceKeyerMEC, and MaskFailureExplainerMEC nodes.

- **Category:** `MaskEditControl/Pipeline`
- **Returns:** `MASK, MASK, IMAGE, MASK, BBOX, STRING, FLOAT, STRING, IMAGE, IMAGE, MASK, MASK, MASK, MASK, MASK, FLOAT, STRING` → `mask, alpha, preview, trimap, bbox, bbox_json, score, info, despilled, lightwrap_rgba, edge_mask, inside_mask, outside_mask, luma_key_mask, problem_regions, severity, suggested_method`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `image` | IMAGE | `""`  | Source image or video frames (B,H,W,C). |
| `segmenter` | combo[sam2.1, sam3  [missing-deps], sam3.1, birefnet, rmbg, inspyrenet, dis, person-mask  [missing-deps]…] | `"sam2.1"`  | Coarse-mask backend. Entries tagged [missing-deps] need an optional pip install to activate. |
| `matter` | combo[none, vitmatte, rvm, bgmattingv2, matanyone  [missing-deps], birefnet, rmbg] | `"vitmatte"`  | Optional alpha refinement. 'none' returns the segmenter mask as alpha. |
| `model` | combo[(auto), sam2/sam2.1_hiera_base_plus-fp16.safetensors, sam2/sam2.1_hiera_base_plus.safetensors, sam2/sam2.1_hiera_large.safetensors, sam2/sam2.1_hiera_small.safetensors, sam2/sam2.1_hiera_tiny-fp16.safetensors, sam2/sam2.1_hiera_tiny.safetensors, sam2/sam2_hiera_base_plus.safetensors…] | `"(auto)"`  | Specific weight file to use. Tag prefix selects the backend folder; '(auto)' lets each backend pick. |
| `matter_model` | combo[(auto), sam2/sam2.1_hiera_base_plus-fp16.safetensors, sam2/sam2.1_hiera_base_plus.safetensors, sam2/sam2.1_hiera_large.safetensors, sam2/sam2.1_hiera_small.safetensors, sam2/sam2.1_hiera_tiny-fp16.safetensors, sam2/sam2.1_hiera_tiny.safetensors, sam2/sam2_hiera_base_plus.safetensors…] | `"(auto)"`  | Weight file for the matter backend. |
| `precision` | combo[fp16, bf16, fp32] | `"fp16"`  |  |
| `attention` | combo[auto, sdpa, flash, sage, xformers, eager] | `"auto"`  |  |
| `offload` | combo[none, cpu, sequential] | `"none"`  |  |
| `subject_preset` | combo[custom, hair, fur, cloth, skin_face, hard_edge, soft_glow] | `"custom"`  | Override trimap_dilate/erode/edge with subject-tuned values. |
| `trimap_dilate` | INT | `8` (min 0, max 128, step 1) |  |
| `trimap_erode` | INT | `8` (min 0, max 128, step 1) |  |
| `edge_radius` | INT | `4` (min 0, max 64, step 1) |  |
| `individual_objects` | BOOLEAN | `False`  | If supported by the backend, return one mask per detected object. |
| `tracking_direction` | combo[forward, backward, bidirectional] | `"forward"`  |  |
| `frame_annotation` | INT | `0` (min 0, max 100000) | Frame index (in clip) where prompts are anchored. |
| `object_id` | INT | `0` (min 0, max 1024) |  |
| `max_frames_to_track` | INT | `0` (min 0, max 100000) | 0 = no cap. |
| `memory_size` | INT | `8` (min 1, max 256) |  |
| `start_frame` | INT | `0` (min 0, max 100000) |  |
| `end_frame` | INT | `-1` (min -1, max 100000) | -1 = last frame. |
| `auto_download` | BOOLEAN | `False`  | Allow lazy auto-download from HF/torch.hub when a weight is missing. |
| `seed` | INT | `0` (min 0, max 18446744073709551615) |  |
| `tta_flip` | BOOLEAN | `False`  | Test-time augmentation: run segmenter on the H-flipped image and average. Slower but cleaner. |
| `multiscale` | BOOLEAN | `False`  | Run the segmenter at 0.75x / 1.0x / 1.25x and fuse. Helps small / thin subjects. |
| `post_refine` | combo[none, guided, crf, crf+guided] | `"none"`  | Final alpha refinement. 'guided' = guided filter (fast, torch-only). 'crf' = DenseCRF (requires pydensecrf, sharpest edges). |
| `refine_radius` | INT | `8` (min 1, max 64) | Spatial radius for guided / CRF refinement. |
| `refine_iterations` | INT | `5` (min 1, max 30) | CRF inference iterations. |
| `despill` | combo[off, green, blue, red, magenta, cyan, yellow, white…] | `"off"`  | Colour decontamination on the named backing. 'auto' estimates the colour from image corners. |
| `despill_strength` | FLOAT | `1.0` (min 0.0, max 2.0, step 0.05) | How aggressively to subtract the spill (0 = off). |
| `preserve_skin` | BOOLEAN | `True`  | Keep warm pixels (R>G>B) untouched during despill. |
| `lightwrap_strength` | FLOAT | `0.0` (min 0.0, max 2.0, step 0.05) | Light-wrap intensity. 0 = off; ~0.3-0.6 = natural blend over the new BG. |
| `lightwrap_radius` | INT | `8` (min 1, max 64) | Light-wrap halo radius in pixels. |
| `edge_band_radius` | INT | `4` (min 1, max 64) | Width of the soft edge band when splitting edge/inside/outside masks. |
| `premultiply` | BOOLEAN | `True`  | Premultiply preview by alpha. Disable for straight-alpha outputs. |
| `enable_luma_key` | BOOLEAN | `False`  | Run a luminance keyer on the source image BEFORE segmentation and use it as a hint / external_mask. |
| `luma_mode` | combo[auto, highlights, midtones, shadows, custom] | `"auto"`  |  |
| `luma_low` | FLOAT | `0.0` (min 0.0, max 1.0, step 0.01) |  |
| `luma_high` | FLOAT | `1.0` (min 0.0, max 1.0, step 0.01) |  |
| `luma_gamma` | FLOAT | `1.0` (min 0.01, max 10.0, step 0.01) |  |
| `luma_falloff` | FLOAT | `1.0` (min 0.0, max 10.0, step 0.1) |  |
| `luma_invert` | BOOLEAN | `False`  |  |
| `luma_mix` | combo[intersect, union, replace, hint_only] | `"hint_only"`  | How to combine the luma-key mask with the segmenter result. 'hint_only' = use as external_mask hint; 'intersect/union/replace' = combine with the final alpha. |
| `enable_advanced_trimap` | BOOLEAN | `False`  | Use the edge-aware trimap generator (asymmetric inner/outer scaling, image-edge snapping, smoothing) instead of the simple dilate/erode trimap. |
| `trimap_inner_scale` | FLOAT | `1.0` (min 0.1, max 3.0, step 0.1) |  |
| `trimap_outer_scale` | FLOAT | `1.5` (min 0.5, max 5.0, step 0.1) |  |
| `trimap_smooth` | FLOAT | `0.0` (min 0.0, max 20.0, step 0.5) |  |
| `trimap_threshold` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) |  |
| `enable_refine` | BOOLEAN | `False`  | Run the unified 11-stage refinement pipeline on the alpha after matting. |
| `refine_hole_fill` | BOOLEAN | `False`  | Stage 1: scipy.binary_fill_holes on the binarised alpha. |
| `refine_hole_fill_thresh` | FLOAT | `0.5` (min 0.05, max 0.95, step 0.05) |  |
| `refine_morph_op` | combo[none, close, open, dilate, erode] | `"none"`  | Stage 2: morphology with a circular SE. |
| `refine_morph_radius` | INT | `3` (min 1, max 64) |  |
| `refine_thin_recover` | BOOLEAN | `False`  | Stage 3: skeletonize+keep-long-branches+dilate to re-inject hair/wire. |
| `refine_thin_threshold` | FLOAT | `0.5` (min 0.1, max 0.95, step 0.05) |  |
| `refine_thin_min_branch_len` | INT | `8` (min 1, max 256) |  |
| `refine_thin_branch_dilate` | INT | `2` (min 1, max 16) |  |
| `refine_joint_bilateral` | BOOLEAN | `False`  | Stage 4: cv2.ximgproc joint bilateral (RGB guide). |
| `refine_jb_diameter` | INT | `9` (min 3, max 31) |  |
| `refine_jb_sigma_color` | FLOAT | `25.0` (min 1.0, max 200.0, step 1.0) |  |
| `refine_jb_sigma_space` | FLOAT | `7.0` (min 1.0, max 200.0, step 1.0) |  |
| `refine_guided_filter` | BOOLEAN | `True`  | Stage 5: He et al. guided filter (torch, always available). |
| `refine_gf_radius` | INT | `8` (min 1, max 64) |  |
| `refine_gf_epsilon` | FLOAT | `0.0001` (min 1e-06, max 0.1, step 1e-05) |  |
| `refine_dense_crf` | BOOLEAN | `False`  | Stage 6: DenseCRF (requires pydensecrf). |
| `refine_crf_iterations` | INT | `5` (min 1, max 30) |  |
| `refine_crf_gauss_sxy` | FLOAT | `3.0` (min 0.1, max 50.0, step 0.1) |  |
| `refine_crf_bilateral_sxy` | FLOAT | `50.0` (min 1.0, max 200.0, step 1.0) |  |
| `refine_crf_bilateral_srgb` | FLOAT | `13.0` (min 1.0, max 100.0, step 0.5) |  |
| `refine_edge_snap` | BOOLEAN | `False`  | Stage 7: modulate mask boundary by RGB gradient magnitude. |
| `refine_edge_snap_strength` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.05) |  |
| `refine_edge_snap_band` | INT | `6` (min 1, max 32) |  |
| `refine_cascade_passes` | INT | `0` (min 0, max 5) | Stage 8: CascadePSP-style repeats with shrinking radii. |
| `refine_feather_sigma` | FLOAT | `0.0` (min 0.0, max 20.0, step 0.1) | Stage 9: Gaussian blur on the soft alpha. |
| `refine_gamma` | FLOAT | `1.0` (min 0.1, max 5.0, step 0.05) | Stage 10: pow curve on the soft alpha. |
| `refine_threshold` | FLOAT | `0.0` (min 0.0, max 1.0, step 0.01) | Stage 11: hard binarise at this value (0 = keep soft). |
| `enable_diagnose` | BOOLEAN | `True`  | Run automatic mask-failure diagnostics (severity score + suggested method). |
| `diag_ring_width` | INT | `5` (min 1, max 50) |  |
| `diag_blur_threshold` | FLOAT | `50.0` (min 0.0, max 1000.0, step 1.0) |  |
| `diag_brightness_threshold` | FLOAT | `0.15` (min 0.0, max 1.0, step 0.01) |  |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `positive_coords` | STRING | `""`  | JSON list of positive points [[x,y],...] from PointsBBoxMaskEditor (positive_coords output). |
| `negative_coords` | STRING | `""`  | JSON list of negative points [[x,y],...] from PointsBBoxMaskEditor (negative_coords output). |
| `pos_bbox` | BBOX | `""`  | Single positive bbox [x0,y0,x1,y1]. |
| `neg_bbox` | BBOX | `""`  | Optional negative bbox (excluded region). |
| `normal_bbox` | BBOX | `""`  | Generic bbox if you don't care about polarity. |
| `text_prompt` | STRING | `""`  | Open-vocabulary text prompt (SAM3 / GroundingDINO / VideoMaMa). Wire from any STRING source. |
| `external_mask` | MASK | `""`  | Optional mask used as a hint or overridden when input_mode='auto' falls through. |
| `external_trimap` | MASK | `""`  | Optional pre-computed trimap that bypasses internal trimap generation. |
| `holdout_mask` | MASK | `""`  | Garbage / holdout matte. Pixels where this is >0 are FORCED to alpha=0 (used to chop out boom mics, rigs, etc). |
| `core_mask` | MASK | `""`  | Core / inside matte. Pixels where this is >0 are FORCED to alpha=1 (used to lock down opaque interiors). |


## `MaskTrackerMEC` — Mask Tracker — Motion/Propagate/Anchor/Consistency (MEC)

> Unified video-mask tracker. Pick a mode and the corresponding engine runs. All modes share the (mask, video) input pair. Heavy work is chunked/vectorized; CPU fallback always available.

- **Category:** `MaskEditControl/Video`
- **Returns:** `MASK, IMAGE, FLOAT, STRING, STRING` → `masks, preview, score, info_json, metric`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `mode` | combo[motion, propagate, anchor, consistency_check] | `"motion"`  | motion: per-frame motion mask (pixel/flow/bg/hist). propagate: seed mask on one frame, push to all frames. anchor: SDF interpolation between anchor masks. consistency_check: score flicker between consecutive frames. |
| `camera_compensation` | BOOLEAN | `True`  | [motion] subtract global camera motion |
| `stabilization_method` | combo[homography, affine, translation] | `"homography"`  | [motion] camera-motion model |
| `detection_mode` | combo[combined, pixel_diff, optical_flow, background_sub, histogram_diff] | `"combined"`  | [motion] active method(s) |
| `pixel_diff_enabled` | BOOLEAN | `True`  | [motion] enable pixel-diff method |
| `pixel_diff_threshold` | FLOAT | `0.05` (min 0.001, max 1.0, step 0.001) | [motion] pixel-diff threshold |
| `flow_enabled` | BOOLEAN | `True`  | [motion] enable optical flow |
| `flow_threshold` | FLOAT | `1.0` (min 0.1, max 50.0, step 0.1) | [motion] flow magnitude threshold |
| `flow_algorithm` | combo[farneback, phase_correlation] | `"farneback"`  | [motion] flow algorithm |
| `bg_sub_enabled` | BOOLEAN | `False`  | [motion] background subtraction |
| `bg_model_frames` | INT | `5` (min 1, max 30, step 1) | [motion] frames for bg model |
| `bg_sub_threshold` | FLOAT | `0.1` (min 0.001, max 1.0, step 0.001) | [motion] bg-diff threshold |
| `hist_enabled` | BOOLEAN | `False`  | [motion] histogram diff |
| `hist_grid_size` | INT | `16` (min 4, max 64, step 4) | [motion] histogram grid NxN |
| `hist_threshold` | FLOAT | `0.15` (min 0.01, max 1.0, step 0.01) | [motion] histogram L2 threshold |
| `combine_method` | combo[union, intersection] | `"union"`  | [motion] method combination |
| `grow_pixels` | FLOAT | `4.0` (min 0.0, max 64.0, step 1.0) | [motion] dilate result |
| `min_region_size` | INT | `100` (min 0, max 10000, step 10) | [motion] noise filter |
| `temporal_smooth` | BOOLEAN | `True`  | [motion] gaussian time smoothing |
| `source_frame` | INT | `0` (min 0, max 99999) | [propagate] frame where mask is drawn |
| `propagate_mode` | combo[static, optical_flow, sam2_video, fade, scale_linear] | `"static"`  | [propagate] propagation method |
| `prop_flow_threshold` | FLOAT | `2.0` (min 0.0, max 50.0, step 0.5) | [propagate] optical-flow threshold |
| `fade_start` | FLOAT | `1.0` (min 0.0, max 1.0, step 0.01) | [propagate] opacity at source frame |
| `fade_end` | FLOAT | `0.0` (min 0.0, max 1.0, step 0.01) | [propagate] opacity at last frame |
| `bidirectional` | BOOLEAN | `True`  | [propagate] forward+backward from source |
| `anchor_frames` | STRING | `"0"`  | [anchor] CSV frame indices for each anchor mask |
| `total_frames` | INT | `30` (min 1, max 99999) | [anchor] total output frames |
| `easing` | combo[linear, ease_in, ease_out, smooth_step] | `"smooth_step"`  | [anchor] easing curve |
| `sdf_iterations` | INT | `64` (min 4, max 512, step 4) | [anchor] SDF diffusion iterations |
| `flow_refinement` | BOOLEAN | `False`  | [anchor] optical-flow refine (needs video) |
| `metric` | combo[mask_iou, pixel_diff, flow_warp] | `"pixel_diff"`  | [consistency_check] metric |
| `binarize_threshold` | FLOAT | `0.5` (min 0.01, max 0.99, step 0.01) | [consistency_check] mask binarize threshold |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `mask` | MASK | `""`  | Required by propagate (seed), anchor (anchor stack), consistency_check (mask_iou). Optional for motion. |
| `video` | IMAGE | `""`  | Video frame batch (B,H,W,C). Required by motion, propagate, anchor flow_refinement, and pixel/flow consistency. |
| `sam_model` | SAM_MODEL | `""`  | [propagate sam2_video mode] SAM2 model |
| `points_json` | STRING | `""`  | [propagate sam2_video mode] point prompts |


## `ModelMetadataExtractorMEC` — Model Metadata Extractor (MEC)

> Inspect model file metadata WITHOUT unpickling or loading weights. Safe to run on untrusted .ckpt files. Reports tensor count, params, training metadata, and a quick fingerprint.

- **Category:** `MaskEditControl/Diagnostics`
- **Returns:** `STRING, STRING, INT, STRING, STRING` → `metadata_json, model_kind, total_params, fingerprint, lineage_json`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `file_path` | STRING | `""`  | Absolute path to a model file (.safetensors / .pt / .pth / .ckpt). |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `compute_fingerprint` | BOOLEAN | `True`  | Compute SHA256 over (size, first 1 MB, last 1 MB). Suitable cache key; far faster than full-file hashing. |


## `ParameterHistoryMEC` — Parameter History (MEC)

> Query the parameter history database. Shows what parameters were changed, when, and what the previous values were. Supports run-to-run diffs.

- **Category:** `MaskEditControl/Utils`
- **Returns:** `STRING` → `history_report`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `mode` | combo[all_history, last_run_diff, node_class_filter] | `"all_history"`  | all_history: show recent parameter changes across all nodes last_run_diff: show what changed between the last two runs node_class_filter: filter history to a specific node class |
| `last_n_runs` | INT | `5` (min 1, max 100, step 1) | How many recent runs to include |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `node_class_filter` | STRING | `""`  | Node class name to filter by (e.g. 'KSampler') |
| `run_a` | INT | `0` (min 0) | First run number for diff mode (0 = auto-detect last two) |
| `run_b` | INT | `0` (min 0) | Second run number for diff mode (0 = auto-detect last two) |


## `ProPainterMEC` — ProPainter — Temporal / Remove / Stitch / Refine / Flow (MEC)

> Unified ProPainter node. Absorbs ProPainterTemporal / Remove / Stitch / StitchRefine / FlowRefine. Pick a mode and only the relevant widgets are read; others are ignored.

- **Category:** `MaskEditControl/Inpaint`
- **Returns:** `IMAGE, MASK, IMAGE, MASK, STRING` → `image_out, mask_out, aux_image, aux_mask, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `mode` | combo[temporal, remove, stitch, stitch_refine, flow] | `"remove"`  | Pick the ProPainter operation. Each mode reads a different subset of the optional inputs/widgets below. |
| `use_half` | BOOLEAN | `True`  |  |
| `color_match_mode` | combo[off, reinhard, lab, lab_transfer, none] | `"reinhard"`  | Per-frame masked colour match between fill and surroundings (ignored in 'flow' mode). |
| `raft_iter` | INT | `12` (min 1, max 100) |  |
| `neighbor_stride` | INT | `5` (min 1, max 32) |  |
| `ref_stride` | INT | `10` (min 1, max 64) |  |
| `subvideo_length` | INT | `8` (min 2, max 300) | Frames per InpaintGenerator window. 8GB cards: 8-30. 12GB: 40-60. 24GB: 80+. |
| `raft_chunk` | INT | `16` (min 1, max 64) | Frame-pairs per RAFT forward pass. |
| `blend_boundary` | BOOLEAN | `True`  | [temporal] Blend the inpaint with original at the crop boundary using stitch_data. |
| `remove_quality` | combo[fast, balanced, quality] | `"balanced"`  | [remove] Preset that overrides raft_iter/neighbor_stride/ref_stride/subvideo_length. |
| `remove_dilate_pixels` | INT | `3` (min 0, max 32) | [remove] Dilate the mask by N px before filling to cover anti-aliasing. |
| `boundary_band_pixels` | INT | `12` (min 0, max 96) | [stitch] Width of the boundary band re-painted (0 = no boundary repaint). |
| `preserve_inpaint_center` | BOOLEAN | `True`  | [stitch] Keep generative inpaint untouched at the centre, only repaint the seam. |
| `upscale_method` | combo[lanczos, bicubic, bilinear, nearest] | `"lanczos"`  | [stitch] How to scale the inpainted crop to the canvas region. |
| `ring_pixels` | INT | `8` (min 1, max 64) | [stitch_refine] Half-width of the seam ring in pixels. |
| `flow_consistency_thr` | FLOAT | `1.5` (min 0.0, max 20.0, step 0.05) | [flow] Forward/backward consistency threshold (pixels). |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `images` | IMAGE | `""`  | [temporal/remove] Source frames. |
| `masks` | MASK | `""`  | [temporal/remove] Region to inpaint. |
| `stitch_data` | STITCH_DATA | `""`  | [temporal/stitch/stitch_refine] STITCH_DATA from InpaintCropProMEC. |
| `inpainted_image` | IMAGE | `""`  | [stitch] Crop-sized generative inpaint output. |
| `stitched_image` | IMAGE | `""`  | [stitch_refine] Already-stitched canvas image (output of stitch). |
| `mask_override` | MASK | `""`  | [stitch_refine] Optional explicit canvas-space mask (overrides stitch_data). |
| `frame_a` | IMAGE | `""`  | [flow] First frame for optical-flow computation. |
| `frame_b` | IMAGE | `""`  | [flow] Second frame for optical-flow computation. |
| `flow_mask` | MASK | `""`  | [flow] Optional mask restricting the consistency visualisation. |


## `SeCMatAnyonePipelineMEC` — SeC + MatAnyone2 Pipeline (MEC)

> SeC + MatAnyone2 end-to-end pipeline:

- **Category:** `MaskEditControl/Pipeline`
- **Returns:** `IMAGE, MASK, MASK, IMAGE, STRING` → `rgb, alpha_mask, coarse_mask, preview, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `image` | IMAGE | `""`  | Single image or video frames (B>1 for video). |
| `segmentation_model` | combo[sam2.1_hiera_base_plus, sam2.1_hiera_large, sam2.1_hiera_small, sam2.1_hiera_tiny, sam2_hiera_base_plus, sam2_hiera_large, sam2_hiera_small, sam2_hiera_tiny…] | `"sam2.1_hiera_base_plus"`  | Segmentation model for coarse masks. SeC: best for video with text prompts. SAM2/3: best for point/bbox prompts. |
| `text_prompt` | STRING | `""`  | Text description of target object (e.g. 'cat', 'person in red'). Used by SeC for semantic tracking. Leave empty for point/bbox prompts. |
| `points_json` | STRING | `"[]"`  | Point prompts: [{"x":100,"y":200,"label":1}, ...] |
| `bbox_json` | STRING | `""`  | Bounding box: [x1,y1,x2,y2] |
| `matting_backend` | combo[matanyone2, vitmatte_small, vitmatte_base, auto] | `"auto"`  | Alpha matting backend. auto: MatAnyone2 for video (B>1), ViTMatte for single images. matanyone2: Video matting with temporal consistency. vitmatte_small/base: Neural matting (best edge quality per frame). |
| `edge_radius` | INT | `15` (min 1, max 200, step 1) | Edge refinement radius in pixels. |
| `n_warmup` | INT | `5` (min 1, max 30, step 1) | MatAnyone2 warmup frames (more = better temporal init). |
| `precision` | combo[fp16, bf16, fp32] | `"fp16"`  | Segmentation model precision. |
| `fill_holes_enabled` | BOOLEAN | `True`  | Fill interior holes in the final alpha. |
| `min_region_size` | INT | `64` (min 0, max 10000, step 1) | Remove isolated regions smaller than N pixels. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `positive_coords` | STRING | `""`  | Positive points from Points Mask Editor. |
| `negative_coords` | STRING | `""`  | Negative points from Points Mask Editor. |
| `bbox` | BBOX | `""`  | Positive bbox from upstream node. |
| `edge_refine_method` | combo[none, vitmatte, guided_filter, multi_scale_guided] | `"none"`  | Optional post-matting edge refinement. none: use raw MatAnyone2 output. vitmatte/guided_filter: refine edges after matting. |
| `keep_model_loaded` | BOOLEAN | `True`  | Keep models in VRAM between runs. |


## `SplineMaskMEC` — Spline Mask — Edit/Track/Flow-Path (MEC)

> Unified spline mask node. Same control-point canvas drives three modes: edit (single-frame rasterize), track (LK optical flow across video), flow_path (procedural ribbon/wave/dust). CPU + small VRAM. No models required.

- **Category:** `MaskEditControl/Spline`
- **Returns:** `MASK, STRING, SPLINE_DATA, STRING, BBOX` → `mask, coords_json, spline_data_out, info_json, bbox`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `mode` | combo[edit, track, flow_path] | `"edit"`  | edit: rasterize a single spline (closed/open) to a mask. track: Lucas-Kanade multi-keyframe tracker across a video. flow_path: procedural pattern along the spline (waves/dust/lightning…). |
| `spline_data` | STRING | `"[]"`  | Spline payload from the JS canvas. edit/flow_path: single shape list. track: keyframes list [{frame:int, points:[[x,y],…]}, …]. |
| `spline_type` | combo[catmull_rom, bezier, polyline] | `"catmull_rom"`  | [edit/flow_path] interpolation method |
| `closed` | BOOLEAN | `True`  | [all] closed loop vs open path |
| `samples_per_segment` | INT | `20` (min 2, max 128, step 1) | [all] curve resolution per segment |
| `feather_radius` | FLOAT | `0.0` (min 0.0, max 64.0, step 0.5) | [edit/track] gaussian edge feather |
| `invert` | BOOLEAN | `False`  | [edit/flow_path] invert mask |
| `smoothing` | BOOLEAN | `True`  | [edit] enable spline smoothing |
| `centripetal_alpha` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.05) | [edit] Catmull-Rom alpha (0.5 = centripetal) |
| `width` | INT | `0` (min 0, max 16384, step 1) | [edit/flow_path] output width (0 = inherit image) |
| `height` | INT | `0` (min 0, max 16384, step 1) | [edit/flow_path] output height (0 = inherit image) |
| `mask_color` | STRING | `"#ff00ff"`  | [edit] preview overlay color (hex) |
| `mask_opacity` | FLOAT | `0.4` (min 0.0, max 1.0, step 0.05) | [edit] preview overlay opacity |
| `tracking_weight` | FLOAT | `0.7` (min 0.0, max 1.0, step 0.05) | [track] lerp/tracker blend (1=pure LK, 0=pure lerp) |
| `klt_window` | INT | `21` (min 5, max 51, step 2) | [track] Lucas-Kanade window size |
| `stroke_width` | INT | `3` (min 1, max 64, step 1) | [track] stroke width for open splines |
| `pattern` | combo[ribbon, wave, flow, dust, river, smoke, sawtooth, square…] | `"ribbon"`  | [flow_path] procedural pattern |
| `thickness` | FLOAT | `12.0` (min 0.0, max 1024.0, step 0.5) | [flow_path] base stroke thickness |
| `amplitude` | FLOAT | `8.0` (min 0.0, max 1024.0, step 0.5) | [flow_path] modulation amplitude |
| `frequency` | FLOAT | `2.0` (min 0.0, max 64.0, step 0.1) | [flow_path] modulation frequency |
| `turbulence` | FLOAT | `0.0` (min 0.0, max 4.0, step 0.05) | [flow_path] noise turbulence strength |
| `turbulence_scale` | FLOAT | `1.0` (min 0.01, max 32.0, step 0.05) | [flow_path] noise spatial scale |
| `edge_softness` | FLOAT | `1.0` (min 0.0, max 32.0, step 0.1) | [flow_path] edge softness in pixels |
| `taper_start` | FLOAT | `0.0` (min 0.0, max 1.0, step 0.01) | [flow_path] start-end taper amount |
| `taper_end` | FLOAT | `0.0` (min 0.0, max 1.0, step 0.01) | [flow_path] tail-end taper amount |
| `frames` | INT | `1` (min 1, max 4096, step 1) | [flow_path] number of animation frames |
| `animation_speed` | FLOAT | `0.05` (min 0.0, max 4.0, step 0.005) | [flow_path] phase advance per frame |
| `flow_direction` | combo[forward, reverse, bidirectional, oscillate] | `"forward"`  | [flow_path] flow direction |
| `mod_decay` | FLOAT | `0.0` (min 0.0, max 4.0, step 0.01) | [flow_path] modulation falloff over time |
| `seed` | INT | `0` (min 0, max 4294967295, step 1) | [flow_path] noise seed |
| `use_embedded_editor` | BOOLEAN | `True`  | [flow_path] show embedded spline preview |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `image` | IMAGE | `""`  | [edit/track/flow_path] reference / source video frames |


## `VAEBlockInspectorMEC` — VAE Block Inspector (MEC)

> Per-block weight stats for a VAE (mean/std/abs_mean/count).

- **Category:** `MaskEditControl/ModelAnalysis`
- **Returns:** `STRING, STRING, FLOAT` → `report_json, outlier_tensor_names, anomaly_score`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `vae` | VAE | `""`  | VAE whose per-block weight statistics will be inspected. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `anomaly_threshold` | FLOAT | `5.0` (min 1.5, max 50.0, step 0.5) | Tensors whose abs_mean exceeds this multiple of the cohort median are flagged as magnitude outliers. Lower => more sensitive (more flags). |


## `VAELatentInspectorMEC` — VAE Latent Inspector (MEC)

> Inspect a LATENT tensor: per-channel min/max/mean/std, NaN & Inf counts, and a one-word verdict (healthy/low_contrast/saturated/corrupt). Latent is passed through unchanged.

- **Category:** `MaskEditControl/Diagnostics`
- **Returns:** `LATENT, STRING, STRING, INT, INT` → `latent_passthrough, info_json, verdict, nan_count, inf_count`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `latent` | LATENT | `""`  | ComfyUI LATENT dict (must contain 'samples'). |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `fail_on_corrupt` | BOOLEAN | `False`  | If True, raise ValueError when NaN/Inf detected. |


## `VAEMergeMEC` — VAE Merge (MEC)

> Merge 2 or 3 VAEs with 13 strategies (weighted_sum, sigmoid, geometric, slerp, dare_ties, …). Optional per-block weights, auto-alpha from block cosine similarity, latent-space probe to report reconstruction MSE/PSNR, dry-run, and recipe export.

- **Category:** `MaskEditControl/VAE`
- **Returns:** `VAE, STRING, STRING, STRING` → `vae, info, recipe_json, probe_report`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `vae_a` | VAE | `""`  | Primary VAE (acts as base; deep-copied clone is returned). |
| `vae_b` | VAE | `""`  | Secondary VAE blended into vae_a. |
| `merge_mode` | combo[weighted_sum, add_difference, tensor_sum, triple_sum, slerp, sigmoid, geometric, max_abs…] | `"weighted_sum"`  | Blend strategy. add_difference / triple_sum / smooth_add_diff need vae_c. sigmoid + geometric mimic TechnoByte/meh behaviour. distribution_xover keeps A unless B has higher detail energy. dare_ties = sparse delta + sign election. |
| `alpha` | FLOAT | `0.3` (min 0.0, max 1.0, step 0.01) | Primary blend weight. weighted_sum: 0=A, 1=B. |
| `beta` | FLOAT | `0.7` (min 0.0, max 1.0, step 0.01) | Secondary blend weight (used by add_difference / 3-VAE modes). |
| `brightness` | FLOAT | `0.0` (min -1.0, max 1.0, step 0.01) | Post-merge brightness shift on decoder.conv_out. |
| `contrast` | FLOAT | `0.0` (min -1.0, max 1.0, step 0.01) | Post-merge contrast gain on decoder.conv_out. |
| `use_blocks` | BOOLEAN | `False`  | Enable per-block sliders. When False all keys use 'alpha'. |
| `auto_alpha` | BOOLEAN | `False`  | Data-driven block weights. When True, computes per-block cosine similarity between A and B; dissimilar blocks bias toward A. Overrides the manual block sliders. |
| `block_conv_in` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) | Weight for encoder/decoder conv_in. |
| `block_conv_out` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) | Weight for encoder/decoder conv_out. |
| `block_norm_out` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) | Weight for encoder/decoder norm_out. |
| `block_0` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) | First down/up block pair. |
| `block_1` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) | Second down/up block pair. |
| `block_2` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) | Third down/up block pair. |
| `block_3` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) | Fourth down/up block pair (SDXL). |
| `block_mid` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) | Mid block. |
| `device` | combo[cpu, cuda, auto] | `"cpu"`  | Compute device. CPU is safe (default); CUDA is faster but uses VRAM. |
| `dry_run` | BOOLEAN | `False`  | If True, skip the merge and return only the recipe + similarity report. Use this for fast block-similarity inspection without waiting for the merge. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `vae_c` | VAE | `""`  | Optional third VAE for add_difference / triple_sum / smooth_add_diff. |
| `reference_image` | IMAGE | `""`  | Optional reference image. If connected, encodes/decodes it through A, B, and the merged VAE; reports MSE/PSNR per VAE in probe_report. |
| `recipe_in` | STRING | `""`  | Optional recipe JSON from a previous run. When provided, overrides every widget value above so the exact merge is reproduced. |


## `VAESimilarityAnalyserMEC` — VAE Similarity Analyser (MEC)

> Cosine similarity between two VAEs (per tensor + per block).

- **Category:** `MaskEditControl/ModelAnalysis`
- **Returns:** `STRING, FLOAT, STRING` → `report_json, global_cosine, most_divergent_blocks`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `vae_a` | VAE | `""`  | First VAE to compare. |
| `vae_b` | VAE | `""`  | Second VAE to compare. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `include_per_tensor` | BOOLEAN | `False`  | Include per-tensor cosine entries in the JSON report (verbose). |


## `VideoComparerMEC` — Video Comparer — Wipe/Diff/Scopes/Audio (MEC)

> Nuke-grade A/B comparer for image / video / EXR / audio with wipe, onion, diff, scopes, bit-depth crush, and audio analysis.

- **Category:** `MaskEditControl/Preview`
- **Returns:** `IMAGE, MASK, IMAGE, STRING` → `preview, diff_mask, scope, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `mode` | combo[wipe, onion, diff, side_by_side, per_channel, false_color, waveform_scope, parade_scope…] | `"wipe"`  |  |
| `bit_depth` | combo[8, 10, 12, 16, 32] | `"32"`  | Quantization for bit_depth_crush mode. |
| `wipe_position` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) |  |
| `onion_alpha` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.01) |  |
| `diff_gain` | FLOAT | `16.0` (min 1.0, max 1024.0, step 1.0) |  |
| `diff_gamma` | FLOAT | `1.0` (min 0.1, max 4.0, step 0.05) |  |
| `diff_threshold` | FLOAT | `0.0` (min 0.0, max 0.5, step 0.001) |  |
| `diff_mode` | combo[absolute, signed, luminance] | `"absolute"`  |  |
| `false_color_lut` | combo[viridis, plasma, inferno, magma, turbo, hot, coolwarm] | `"turbo"`  |  |
| `scope_intensity` | FLOAT | `0.35` (min 0.05, max 1.0, step 0.05) |  |
| `frame_index` | INT | `0` (min 0, max 100000, step 1) |  |
| `label_a` | STRING | `"A"`  |  |
| `label_b` | STRING | `"B"`  |  |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `image_a` | IMAGE | `""`  |  |
| `image_b` | IMAGE | `""`  |  |
| `audio_a` | AUDIO | `""`  |  |
| `audio_b` | AUDIO | `""`  |  |
| `file_a` | combo[, 4098429-uhd_4096_2160_25fps.mp4, C1799.MP4 Comp 1.mp4, Jibaro  Love, Death and Robots - Gamerstrix (1080p, h264).mp4, Kapaa.exr, Munyu-Njoya.jpg, TVR7E3Kuzg2iRhKkjZPeWk-1920-80.jpg, _test_binary_square.png…] | `""`  |  |
| `file_b` | combo[, 4098429-uhd_4096_2160_25fps.mp4, C1799.MP4 Comp 1.mp4, Jibaro  Love, Death and Robots - Gamerstrix (1080p, h264).mp4, Kapaa.exr, Munyu-Njoya.jpg, TVR7E3Kuzg2iRhKkjZPeWk-1920-80.jpg, _test_binary_square.png…] | `""`  |  |


## `VideoFramePlayerMEC` — Video Frame Player (MEC)

> Video scrubber + drag-crop + integrated resize. Drag the timeline to scrub frames. Toggle crop_enabled and drag the rectangle on the preview to crop (aspect-locked when a preset is set). resize_method + target_width/target_height + upscale_factor produce the final output. Set output_mode = all_frames to process the whole batch.

- **Category:** `MaskEditControl/Preview`
- **Returns:** `IMAGE, INT, INT, IMAGE, INT, INT, INT, INT, INT, INT, INT, FLOAT, IMAGE, INT, INT, INT, INT, FLOAT, STRING` → `frame, frame_index, frame_count, processed, out_width, out_height, crop_x_px, crop_y_px, crop_w_px, crop_h_px, trimmed_count, playback_fps, frames_trimmed, trim_start_idx, trim_end_idx, width, height, duration, video_info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `frames` | IMAGE | `""`  | Frame batch (B,H,W,C). B=number of frames. |
| `frame_index` | INT | `0` (min 0, max 99999, step 1) | Current frame to emit on the IMAGE output. Drag the timeline to scrub. |
| `output_mode` | combo[current_frame, all_frames] | `"current_frame"`  | current_frame: emit only the selected frame. all_frames: apply trim+stride+crop+resize to every frame. |
| `frame_start` | INT | `0` (min 0, max 99999, step 1) | First frame of the trim range. Drag the green marker on the timeline. |
| `frame_end` | INT | `-1` (min -1, max 99999, step 1) | Last frame of the trim range (inclusive). -1 = last frame. Drag the red marker on the timeline. |
| `frame_stride` | INT | `1` (min 1, max 64, step 1) | In 'all_frames' mode, output every Nth frame within the trim range. |
| `playback_fps` | FLOAT | `24.0` (min 0.1, max 240.0, step 0.5) | Preview playback speed (frames per second). |
| `loop_mode` | combo[once, loop, ping-pong] | `"loop"`  | Preview playback at end of trim range: once / loop / ping-pong. |
| `crop_enabled` | BOOLEAN | `False`  | Enable the drag-crop rectangle on the preview. |
| `crop_locked` | BOOLEAN | `False`  | Lock the crop rect to prevent accidental drags. Press R on the canvas to reset. |
| `aspect_ratio` | combo[free, original, 1:1, 4:3, 3:4, 16:9, 9:16, 2:1…] | `"free"`  | Aspect lock. 'original' = source W:H. 'custom' = custom_aspect_w:custom_aspect_h. |
| `custom_aspect_w` | FLOAT | `16.0` (min 0.0, max 999.0, step 0.1) | Custom aspect width (used when aspect_ratio = custom). |
| `custom_aspect_h` | FLOAT | `9.0` (min 0.0, max 999.0, step 0.1) | Custom aspect height (used when aspect_ratio = custom). |
| `crop_x` | FLOAT | `0.0` (min 0.0, max 1.0, step 0.001) | Crop left edge as fraction of source width [0..1]. Set by dragging the rectangle on the preview. |
| `crop_y` | FLOAT | `0.0` (min 0.0, max 1.0, step 0.001) | Crop top edge as fraction of source height [0..1]. |
| `crop_w` | FLOAT | `1.0` (min 0.001, max 1.0, step 0.001) | Crop width as fraction of source width (0..1]. |
| `crop_h` | FLOAT | `1.0` (min 0.001, max 1.0, step 0.001) | Crop height as fraction of source height (0..1]. |
| `resize_method` | combo[none, lanczos, bicubic, bilinear, area, nearest-exact] | `"none"`  | Post-crop resize. 'lanczos' = high-quality. |
| `target_width` | INT | `0` (min 0, max 8192, step 8) | Target width after crop (0 = keep crop width). |
| `target_height` | INT | `0` (min 0, max 8192, step 8) | Target height after crop (0 = keep crop height). |
| `upscale_factor` | FLOAT | `1.0` (min 0.1, max 8.0, step 0.05) | Multiplier applied AFTER target_width/target_height (1.0 = no upscale). |
| `preview_width` | INT | `480` (min 96, max 1920, step 16) | Width (px) of preview frames sent to the browser. Lower = lighter UI; full-resolution IMAGE outputs are unaffected. |
| `preview_format` | combo[png, jpeg] | `"png"`  | png = lossless, exact colors (recommended). jpeg = smaller payload, slight chroma loss. The IMAGE outputs are always the full-precision source tensor regardless of this setting. |
| `preview_quality` | INT | `95` (min 30, max 100, step 5) | JPEG quality (ignored when preview_format=png). |


## `VideoMaskEditorMEC` — Video Mask Editor (MEC)

> Open the in-browser video mask editor (brush / erase / fill / lasso / onion-skin) to pin per-frame keyframes, then tweens non-keyframed frames using distance-transform interpolation.

- **Category:** `MaskEditControl/VideoMask`
- **Returns:** `MASK, STRING` → `mask, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `image` | IMAGE | `""`  | Input video batch (B,H,W,3). Drives B/H/W. |
| `session_id` | STRING | `""`  | Auto-set by the editor UI. Don't edit by hand. |
| `tween_mode` | combo[distance_transform, linear, hold] | `"distance_transform"`  | How to interpolate non-keyframed frames. |
| `feather` | FLOAT | `0.0` (min 0.0, max 32.0, step 0.1) | Gaussian feather radius (px). 0 = none. |
| `threshold` | FLOAT | `0.0` (min 0.0, max 1.0, step 0.01) | Binarize after tween. 0 = keep soft mask. |

**Optional inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `input_mask` | MASK | `""`  | Fallback mask used if no keyframes are set. |


## `VideoStabilizerAutoMEC` — Video Stabilizer — Auto (MEC)

> Auto stabilizer that picks classic vs. flow backend based on clip length and VRAM.

- **Category:** `MaskEditControl/Stabilization`
- **Returns:** `IMAGE, MASK, STRING` → `stabilized_frames, padding_mask, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `frames` | IMAGE | `""`  |  |
| `frame_rate` | FLOAT | `16.0` (min 1.0, step 0.1) |  |
| `force_backend` | combo[auto, classic, flow] | `"auto"`  |  |
| `preset` | combo[handheld_light, handheld_heavy, vehicle, tripod_lock] | `"handheld_light"`  |  |
| `padding_color` | STRING | `"127, 127, 127"`  |  |


## `VideoStabilizerClassicMEC` — Video Stabilizer — Classic (MEC)

> Feature-tracking video stabilizer (vendored MIT ComfyUI-Video-Stabilizer).

- **Category:** `MaskEditControl/Stabilization`
- **Returns:** `IMAGE, MASK, STRING` → `stabilized_frames, padding_mask, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `frames` | IMAGE | `""`  |  |
| `frame_rate` | FLOAT | `16.0` (min 1.0, step 0.1) |  |
| `framing_mode` | combo[crop, crop_and_pad, expand] | `"crop_and_pad"`  |  |
| `transform_mode` | combo[translation, similarity, perspective] | `"similarity"`  |  |
| `camera_lock` | BOOLEAN | `False`  |  |
| `strength` | FLOAT | `0.7` (min 0.0, max 1.0, step 0.05) |  |
| `smooth` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.05) |  |
| `keep_fov` | FLOAT | `0.6` (min 0.0, max 1.0, step 0.05) |  |
| `padding_color` | STRING | `"127, 127, 127"`  |  |


## `VideoStabilizerFlowMEC` — Video Stabilizer — RAFT Flow (MEC)

> RAFT dense-flow video stabilizer (vendored MIT ComfyUI-Video-Stabilizer).

- **Category:** `MaskEditControl/Stabilization`
- **Returns:** `IMAGE, MASK, STRING` → `stabilized_frames, padding_mask, info`

**Required inputs**

| Name | Type | Default (range) | Description |
|------|------|-----------------|-------------|
| `frames` | IMAGE | `""`  |  |
| `frame_rate` | FLOAT | `16.0` (min 1.0, step 0.1) |  |
| `framing_mode` | combo[crop, crop_and_pad, expand] | `"crop_and_pad"`  |  |
| `transform_mode` | combo[translation, similarity, perspective] | `"similarity"`  |  |
| `camera_lock` | BOOLEAN | `False`  |  |
| `strength` | FLOAT | `0.7` (min 0.0, max 1.0, step 0.05) |  |
| `smooth` | FLOAT | `0.5` (min 0.0, max 1.0, step 0.05) |  |
| `keep_fov` | FLOAT | `0.6` (min 0.0, max 1.0, step 0.05) |  |
| `padding_color` | STRING | `"127, 127, 127"`  |  |
| `raft_iters` | INT | `12` (min 4, max 32) |  |
| `use_half` | BOOLEAN | `True`  |  |
