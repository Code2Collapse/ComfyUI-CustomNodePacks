# C2C / MEC Custom Node Packs — Node Reference

> **REBASELINE 2026-09-06.** This pack registers **89** nodes, not 104. The
> cross-pack dedup moved six VFX modules to ComfyUI-NukeMaxNodes and
> WanDirectorC2C to ComfyUI-WanNodeExperiments (one owner per id); two vault
> nodes were then added. Measured this session with the production loader.
>
> This file is marked auto-generated, but **no generator exists in the repo**,
> so the per-node bodies below cannot be regenerated and may drift from the
> tooltips they claim to mirror. Treat the live schema as authoritative.

*Auto-generated from the live `NODE_CLASS_MAPPINGS` on 2026-07-29 — 104 nodes. Every parameter description below is the node's own tooltip, so this file cannot drift from the code.*

Regenerate after changing any node's `INPUT_TYPES`.


## Contents

- **C2C/AI Image** (1)
  - [Nano Banana · Gemini Image (C2C)](#nanobananac2c)
- **C2C/Control** (1)
  - [Control AOV — Multi-Control Fusion (C2C)](#controlaovc2c)
- **C2C/Diagnostics** (5)
  - [Insight Status](#insightstatusmec)
  - [Integrity Status](#integritystatusmec)
  - [Mask Failure Explainer — Diagnostics](#maskfailureexplainermec)
  - [Model Metadata Extractor](#modelmetadataextractormec)
  - [VAE Latent Inspector](#vaelatentinspectormec)
- **C2C/Helpers** (12)
  - [Aspect Ratio Preset (C2C)](#aspectpresetmec)
  - [Conditional Switch](#conditionalswitchmec)
  - [Dimensions Snap (C2C)](#dimensionssnapmec)
  - [Execution Timer (C2C)](#executiontimermec)
  - [Image Batch Slice (C2C)](#imagebatchslicemec)
  - [Image Batch Split (C2C)](#imagebatchsplitmec)
  - [Image Stats Probe (C2C)](#imagestatsprobemec)
  - [Mask Area Probe (C2C)](#maskareaprobemec)
  - [Mask Batch Combine (C2C)](#maskbatchcombinemec)
  - [Number Lerp (C2C)](#numberlerpmec)
  - [Seed List Generator (C2C)](#seedlistmec)
  - [Text Template (C2C)](#texttemplatemec)
- **C2C/IO** (1)
  - [Batch Version Manager](#batchversionmanagermec)
- **C2C/Inpaint** (5)
  - [Inpaint Crop Pro](#inpaintcroppromec)
  - [Inpaint Mask Prepare](#inpaintmaskpreparemec)
  - [Inpaint Paste Back](#inpaintpastebackmec)
  - [Inpaint Stitch Pro](#inpaintstitchpromec)
  - [ProPainter — Temporal / Remove / Stitch / Refine / Flow](#propaintermec)
- **C2C/Keying** (1)
  - [Luminance Keyer — Highlights / Shadows / Custom](#luminancekeyermec)
- **C2C/Matting** (1)
  - [Background Remover — RMBG / BiRefNet](#backgroundremovermec)
- **C2C/ModelAnalysis** (2)
  - [VAE Block Inspector](#vaeblockinspectormec)
  - [VAE Similarity Analyser](#vaesimilarityanalysermec)
- **C2C/Paint** (5)
  - [Advanced Paint Canvas](#mecadvancedpaintcanvas)
  - [Builder Sampler](#mecbuildersampler)
  - [Context Inpainter / Fixer](#meccontextinpainter)
  - [Face Fixer](#mecfacefixer)
  - [Tone Refiner](#mectonerefiner)
- **C2C/Pipeline** (3)
  - [Mask Refiner](#maskrefinemec)
  - [SAM + ViTMatte Pipeline — Full Quality](#samvitmattepipelinemec)
  - [SeC + MatAnyone2 Pipeline](#secmatanyonepipelinemec)
- **C2C/Preview** (2)
  - [Video Comparer — Player + Wipe/Diff/Scopes](#videocomparerc2c)
  - [Video Frame Player](#videoframeplayermec)
- **C2C/SAM** (3)
  - [SAM Mask Generator — Points + BBox + Text](#sammaskgeneratormec)
  - [SAM Model Loader — SAM2.1 / SAM3](#sammodelloadermec)
  - [SAM Multi-Mask Picker — 3 candidates + scores](#sammultimaskpickermec)
- **C2C/Segmentation** (1)
  - [Semantic Segment — Face / Clothes Parsing](#semanticsegmentmec)
- **C2C/Spline** (1)
  - [Spline Mask — Edit/Track/Flow-Path](#splinemaskmec)
- **C2C/Stabilization** (4)
  - [Video Stabilizer — Auto (deprecated)](#videostabilizerautomec)
  - [Video Stabilizer — Classic (deprecated)](#videostabilizerclassicmec)
  - [Video Stabilizer — RAFT Flow (deprecated)](#videostabilizerflowmec)
  - [Video Stabilizer](#videostabilizermec)
- **C2C/Utils** (1)
  - [Parameter History](#parameterhistorymec)
- **C2C/VAE** (1)
  - [VAE Merge](#vaemergemec)
- **C2C/Video** (1)
  - [Mask Tracker — Motion/Propagate/Anchor/Consistency](#masktrackermec)
- **C2C/VideoMask** (1)
  - [Video Mask Editor](#videomaskeditormec)
- **C2C/Wan_Director** (1)
  - [Wan Director](#wandirectorc2c)
- **Code2Collapse/Sampling** (1)
  - [AsymFlow Sampler Patch (Lakonik signal-shift)](#asymflowsamplerpatch)
- **ComfyUI-CustomNodePacks/PromptRelay** (5)
  - [Prompt Relay Advanced Options](#promptrelayadvancedoptionsc2c)
  - [Prompt Relay Encode](#promptrelayencodec2c)
  - [Prompt Relay Encode (Kijai) — deprecated](#promptrelayencodekijaic2c)
  - [Prompt Relay Encode (Smart) — deprecated](#promptrelayencodesmartc2c)
  - [Prompt Relay Restore (Kijai)](#promptrelayrestorekijaic2c)
- **MEC/Audio** (1)
  - [Audio Reverser](#audioreversermec)
- **MEC/Color Science** (3)
  - [C2C ACES Tonemap](#c2cacestonemap)
  - [C2C Color Space Convert](#c2ccolorspaceconvert)
  - [C2C VAE Quality Decode (HDR)](#c2cvaequalitydecode)
- **MEC/Masking** (1)
  - [Mask Placement — Prompt/Ref → Place → Track](#maskplacementmec)
- **MEC/Plate** (2)
  - [PAR Desqueeze (anamorphic → square px)](#pardesqueezemec)
  - [PAR Resqueeze (back to plate)](#parresqueezemec)
- **MEC/RenderFarm** (3)
  - [C2C Farm Cluster Status](#c2c-clusterstatus)
  - [C2C Farm Job History (Audit Log)](#c2c-jobhistory)
  - [C2C Farm Submit — Remote Render](#c2c-submit)
- **MEC/Temporal** (2)
  - [Fluid Shot Decoder (Restore Timing)](#fluidshotdecodermec)
  - [Fluid Shot Encoder (Temporal Normalizer)](#fluidshotencodermec)
- **MaskEditControl/Channels** (1)
  - [Shuffle — Channels (MEC)](#shufflemec)
- **MaskEditControl/Clipboard** (2)
  - [TCL Parse (MEC)](#tclparsemec)
  - [TCL Serialize (MEC)](#tclserializemec)
- **MaskEditControl/Color** (3)
  - [Color Space Convert (MEC)](#colorspaceconvertmec)
  - [Exposure Grade (MEC)](#exposuregrademec)
  - [LUT Apply (.cube) (MEC)](#lutapplymec)
- **MaskEditControl/Edit** (1)
  - [Mask Edit — Transform/Draw/Points/BBox](#maskeditmec)
- **MaskEditControl/Geometry** (3)
  - [Depth Warp (MEC)](#depthwarpmec)
  - [Normal → Curvature (MEC)](#normaltocurvaturemec)
  - [Position Pass Splitter (MEC)](#positionpasssplittermec)
- **MaskEditControl/IO** (3)
  - [EXR Metadata Reader (MEC)](#exrmetadatareadermec)
  - [Load EXR (MEC)](#loadexrmec)
  - [Save EXR (MEC)](#saveexrmec)
- **MaskEditControl/MaskMatting** (1)
  - [Mask Temporal Stabilizer + Integrity](#masktemporalmec)
- **MaskEditControl/Metadata** (3)
  - [Frame Range Router (MEC)](#framerangeroutermec)
  - [Metadata Writer (MEC)](#metadatawritermec)
  - [Shot Metadata Reader (MEC)](#shotmetadatanodemec)
- **MaskEditControl/Pipeline** (1)
  - [Mask + Matting](#maskopsmec)
- **MaskEditControl/PlateTools** (4)
  - [Clean Plate Extractor (MEC)](#cleanplateextractormec)
  - [Difference Matte (MEC)](#differencemattemec)
  - [Grain Match (MEC)](#grainmatchmec)
  - [Plate Stabilizer (MEC)](#platestabilizermec)
- **MaskEditControl/Pose** (1)
  - [Face/Pose Delta Editor (C2C)](#faceposedeltacoremec)
- **MaskEditControl/Render** (2)
  - [Depth-of-Field Mask (MEC)](#depthoffieldmaskmec)
  - [Merge Render Passes (MEC)](#mergerenderpassesmec)
- **MaskEditControl/Roto** (1)
  - [Vector Roto — Bezier (MEC)](#vectorrotomec)
- **MaskEditControl/VFX** (1)
  - [Optical Flow Re-Vector (MEC)](#opticalflowmec)
- **MaskEditControl/Video** (1)
  - [Video Frame Extractor (MEC)](#videoframeextractormec)
- **MaskEnhancedControl/Grounding** (2)
  - [LocateAnything Grounding (MEC)](#locateanythinggroundingmec)
  - [LocateAnything → SAM Prompt (MEC)](#locateanythingtosammec)
- **utils** (3)
  - [Folder Version Incrementer](#folderincrementer)
  - [Folder Version Check](#folderincrementerreset)
  - [Folder Version Set](#folderincrementerset)


---

## C2C/AI Image


### NanoBananaC2C

**Shown in the menu as:** Nano Banana · Gemini Image (C2C)

Generate or edit images with Google's Nano Banana family (gemini-3-pro-image / gemini-2.5-flash-image). Modes: text-to-image, edit, multi-image compose, style transfer. Five prompting styles (raw / cinematic / structured JSON / character-consistency / product shot). Wire up to 4 IMAGE inputs as references (batches flatten, max 14). Needs a Google AI Studio key in `api_key` or GEMINI_API_KEY.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `prompt` | `STRING` | default `""`, multiline | What to generate / how to edit. With prompt_style=structured_json you may paste a full JSON brief. |
| `model` | choice: `gemini-3-pro-image-preview`, `gemini-2.5-flash-image`, `gemini-2.5-flash-image-preview`, `gemini-2.0-flash-preview-image-generation` | default `"gemini-3-pro-image-preview"` | gemini-3-pro-image-preview = Nano Banana Pro (1K/2K/4K, best text, 14 refs). gemini-2.5-flash-image = fast GA model. |
| `mode` | choice: `text_to_image`, `edit_image`, `compose_images`, `style_transfer` | default `"text_to_image"` | edit_image needs ≥1 reference image; style_transfer needs ≥2 (first = content, last = style). |
| `prompt_style` | choice: `raw`, `cinematic`, `structured_json`, `character_consistency`, `product_shot` | default `"raw"` | Wraps your prompt in a proven template: cinematic photo, structured JSON brief, character-consistency lock, or commercial product shot. |
| `aspect_ratio` | choice: `auto`, `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `21:9`, `3:2`, … (+3) | default `"auto"` | — |
| `resolution` | choice: `auto`, `1K`, `2K`, `4K` | default `"auto"` | 1K/2K/4K are Nano Banana **Pro** only; flash models ignore this. |
| `enhance_prompt` | `BOOLEAN` | default `False` | Higgsfield-style enhancer: a fast Gemini text pass expands your prompt into a rich, detailed image brief before generation. |
| `response_modalities` | choice: `IMAGE+TEXT`, `IMAGE` | default `"IMAGE+TEXT"` | IMAGE = picture only; IMAGE+TEXT also returns the model's commentary on the text_response output. |
| `temperature` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.05 | — |
| `batch_count` | `INT` | default `1`, range 1…4 | Sequential API calls; results are stacked into one batch. |
| `seed` | `INT` | default `0`, range 0…2147483647 | Gemini has no true seed — this controls ComfyUI caching: same seed reuses the cached result, new seed regenerates. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `api_key` | `STRING` | default `""` | Google AI Studio key. Empty = use GEMINI_API_KEY / GOOGLE_API_KEY environment variable. |
| `system_instruction` | `STRING` | default `""`, multiline | Optional system-level art direction applied to every generation (brand style, banned elements, palette rules). |
| `image_1` | `IMAGE` |  | Reference image(s). Batches flatten in order. |
| `image_2` | `IMAGE` |  | — |
| `image_3` | `IMAGE` |  | — |
| `image_4` | `IMAGE` |  | For style_transfer the LAST image is the style. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | — |
| 1 | `text_response` | `STRING` | — |
| 2 | `info` | `STRING` | — |


---

## C2C/Control


### ControlAOVC2C

**Shown in the menu as:** Control AOV — Multi-Control Fusion (C2C)

VFX-AOV control fusion: emit depth/canny/pose/normal/motion/ID as separate passes, a channel-packed image (depth=R/canny=G/pose=B), and a convenience blend. Runs Canny + optical-flow motion internally; accepts depth/pose/normal/ID maps as inputs. Feeds any ControlNet / union / control-video. Stack the separate passes for maximum spatial lock.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `blend_mode` | choice: `screen`, `lighten_max`, `linear_dodge`, `multiply`, `average`, `weighted_avg`, `overlay` | default `"screen"` | How the 'blended' overlay combines passes. screen = least-destructive; linear_dodge clips; multiply darkens. |
| `preview_layout` | choice: `horizontal_3`, `vertical_3`, `grid_2x2` | default `"horizontal_3"` | Layout for the 'combined' output: horizontal_3 = depth \| pose \| canny; grid_2x2 = depth\|pose // canny\|original. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Source frames — preprocessors below run on this. |
| `depth_model` | choice: `off`, `da_v2`, `da_v1`, `midas`, `da3`, `depth_pro`, `depthcrafter`, `dvd`, … (+4) | default `"off"` | Depth backend. VENDORED (run inside this pack): da_v2/da_v1/midas (transformers), da3 (Depth-Anything-3), depth_pro, depthcrafter (video), dvd (DVD — deterministic Wan-2.1 VIDEO depth, EnVision-Research; needs the DVD ckpt in models/DVD/, CC BY-NC). The depth_anything_* options delegate to comfyui_controlnet_aux. 'off' = wire an external depth map. |
| `normal_model` | choice: `off`, `sobel_from_depth`, `normalcrafter` | default `"off"` | Normal backend (vendored): sobel_from_depth (no model) or normalcrafter (video). 'off' = wire an external normal map. |
| `depth_size` | choice: `small`, `medium`, `large`, `giant` | default `"small"` | DepthAnything backbone: small=ViT-S (fastest, test default) → giant=ViT-G (best). v1 has no giant (falls back to large). Ignored by metric/zoe. |
| `depth_custom_ckpt` | choice: `(auto)`, `depth_anything_v2_vits.pth`, `depth_anything_v2_vitb.pth`, `depth_anything_v2_vitl.pth`, `depth_anything_v2_vitg.pth`, `depth_anything_vits14.pth`, `depth_anything_vitb14.pth`, `depth_anything_vitl14.pth` | default `"(auto)"` | Pick your own depth model, or '(auto)' = use the backend/size default. |
| `pose_model` | choice: `off`, `vitpose`, `dwpose`, `openpose`, `animal_pose`, `densepose` | default `"off"` | Pose backend. vitpose = WanV2 ViTPose chain (uses models/detection/*.onnx). dwpose/openpose/... delegate to comfyui_controlnet_aux. |
| `pose_ckpt` | `STRING` | default `""` | Custom ViTPose .onnx pose model filename (none auto-detected; type one or leave blank for default). |
| `id_matte_model` | choice: `off`, `sam_auto` | default `"off"` | ID/segmentation matte. sam_auto = automatic SAM segmentation (controlnet_aux SAMPreprocessor). 'off' = wire an external matte. |
| `edge_model` | choice: `internal_canny`, `off`, `canny`, `lineart`, `anyline`, `hed`, `pidinet`, `teed` | default `"internal_canny"` | internal_canny = OpenCV (no model). Others delegate to controlnet_aux. |
| `run_canny` | `BOOLEAN` | default `True` | Used only when edge_model = internal_canny. |
| `canny_low` | `INT` | default `100`, range 0…255 | — |
| `canny_high` | `INT` | default `200`, range 0…255 | — |
| `canny_aperture` | choice: `3`, `5`, `7` | default `3` | Sobel aperture for internal Canny (odd 3/5/7). |
| `depth_invert` | `BOOLEAN` | default `False` | Invert depth (1 - depth). Use when source is 'far=bright' but the ControlNet expects 'near=bright'. |
| `preproc_resolution` | `INT` | default `512`, range 64…4096, step 8 | Resolution passed to delegated preprocessors. |
| `run_motion` | `BOOLEAN` | default `False` | Optical-flow motion-vector pass (needs an image batch ≥ 2 frames). |
| `depth_weight` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.05 | — |
| `canny_weight` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.05 | — |
| `pose_weight` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.05 | — |
| `normal_weight` | `FLOAT` | default `0.0`, range 0.0…2.0, step 0.05 | — |
| `match_to` | choice: `depth`, `canny`, `pose`, `image`, `largest` | default `"image"` | Internal: align every AOV pass to this source's size before resize/output. |
| `resize` | choice: `off`, `width/height`, `scale` | default `"off"` | Resize ALL outputs in-node. off = keep source size; width/height = resize to an exact width x height; scale = multiply size by 'scale'. |
| `width` | `INT` | default `1024`, range 0…16384, step 8 | Target width (resize=width/height). 0 = derive from height + aspect. |
| `height` | `INT` | default `1024`, range 0…16384, step 8 | Target height (resize=width/height). 0 = derive from width + aspect. |
| `scale` | `FLOAT` | default `1.0`, range 0.05…8.0, step 0.05 | Scale factor applied to source size (resize=scale). |
| `divisible_by` | `INT` | default `16`, range 1…512, step 1 | Snap output W/H to a multiple (VAE-safe: 16=Wan, 64=SD). 1 = no snapping. |
| `fit` | choice: `stretch`, `pad`, `crop` | default `"crop"` | Aspect handling: stretch = distort to fit; pad = letterbox with pad_color; crop = centre-crop to fill. |
| `resize_filter` | choice: `lanczos`, `bicubic`, `hamming`, `bilinear`, `box`, `nearest` | default `"lanczos"` | Resample filter. lanczos = sharpest; nearest = hard pixels. |
| `pad_color` | `STRING` | default `"#000000"` | Pad colour (hex), used only when fit=pad. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `blended` | `IMAGE` | Single image, all passes OVERLAID via the chosen blend mode. |
| 1 | `combined` | `IMAGE` | Single image, depth \| pose \| canny shown SIDE-BY-SIDE (or grid) — the clearest 'all 3 at once' view. |
| 2 | `channel_packed` | `IMAGE` | depth=R, canny=G, pose=B — lossless packing for union ControlNets. |
| 3 | `depth` | `IMAGE` | Depth AOV (passthrough). |
| 4 | `canny` | `IMAGE` | Canny AOV (run internally if 'image' wired). |
| 5 | `pose` | `IMAGE` | Pose AOV (passthrough). |
| 6 | `normal` | `IMAGE` | Normal AOV (passthrough). |
| 7 | `motion` | `IMAGE` | Motion-vector AOV (optical flow, run internally on a frame batch). |
| 8 | `id_matte` | `IMAGE` | ID / segmentation matte AOV (passthrough). |
| 9 | `info` | `STRING` | Per-pass STATUS + recommended max-control wiring. |


---

## C2C/Diagnostics


### InsightStatusMEC

**Shown in the menu as:** Insight Status

Reports whether the Insight executor wrap is installed, plus the current torch/cuda memory snapshot.


**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `status` | `STRING` | — |


### IntegrityStatusMEC

**Shown in the menu as:** Integrity Status

Returns the latest integrity scan as a string.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `trigger_rescan` | `BOOLEAN` | default `False` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `report` | `STRING` | — |


### MaskFailureExplainerMEC

**Shown in the menu as:** Mask Failure Explainer — Diagnostics

Diagnose why a mask might be failing. Analyzes brightness, blur, boundary contrast, color confusion, and background complexity. Outputs a detailed explanation, problem heatmap, severity score, and suggested masking method.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Input image(s) — (B,H,W,C) float32 [0,1]. |
| `mask` | `MASK` |  | Mask to diagnose — (B,H,W) float32 [0,1]. Can be from any segmentation method. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `ring_width` | `INT` | default `5`, range 1…50, step 1 | Width in pixels of the boundary ring used for contrast/color analysis. |
| `blur_threshold` | `FLOAT` | default `50.0`, range 0.0…1000.0, step 1.0 | Laplacian variance threshold below which the image is considered blurry. |
| `brightness_threshold` | `FLOAT` | default `0.15`, range 0.0…1.0, step 0.01 | Mean brightness threshold below which the scene is considered dark. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `explanation` | `STRING` | Human-readable diagnosis explaining likely failure causes. |
| 1 | `problem_regions_mask` | `MASK` | Heatmap mask highlighting regions most likely to be problematic. |
| 2 | `severity_score` | `FLOAT` | Overall severity score in [0, 100] (higher means more issues). |
| 3 | `suggested_method` | `STRING` | Suggested masking method or refinement to try next. |


### ModelMetadataExtractorMEC

**Shown in the menu as:** Model Metadata Extractor

Inspect model file metadata WITHOUT unpickling or loading weights. Safe to run on untrusted .ckpt files. Reports tensor count, params, training metadata, and a quick fingerprint.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `file_path` | `STRING` | default `""` | Absolute path to a model file (.safetensors / .pt / .pth / .ckpt). |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `compute_fingerprint` | `BOOLEAN` | default `True` | Compute SHA256 over (size, first 1 MB, last 1 MB). Suitable cache key; far faster than full-file hashing. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `metadata_json` | `STRING` | Full metadata as JSON (header fields, training tags, format). |
| 1 | `model_kind` | `STRING` | Detected model kind (e.g. checkpoint, lora, vae, controlnet, unknown). |
| 2 | `total_params` | `INT` | Total parameter count summed across all tensors. |
| 3 | `fingerprint` | `STRING` | SHA256 fingerprint over (size, first 1 MB, last 1 MB) when enabled. |
| 4 | `lineage_json` | `STRING` | License + lineage information as JSON when present in metadata. |


### VAELatentInspectorMEC

**Shown in the menu as:** VAE Latent Inspector

Inspect a LATENT tensor: per-channel min/max/mean/std, NaN & Inf counts, and a one-word verdict (healthy/low_contrast/saturated/corrupt). Latent is passed through unchanged.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `latent` | `LATENT` |  | ComfyUI LATENT dict (must contain 'samples'). |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `fail_on_corrupt` | `BOOLEAN` | default `False` | If True, raise ValueError when NaN/Inf detected. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `latent_passthrough` | `LATENT` | Pass-through of the original LATENT input (unchanged). |
| 1 | `info_json` | `STRING` | JSON with shape, dtype, device, per-channel stats, range, and verdict. |
| 2 | `verdict` | `STRING` | One-word verdict: healthy / low_contrast / saturated / corrupt. |
| 3 | `nan_count` | `INT` | Total NaN element count in latent['samples']. |
| 4 | `inf_count` | `INT` | Total Inf element count in latent['samples']. |


---

## C2C/Helpers


### AspectPresetMEC

**Shown in the menu as:** Aspect Ratio Preset (C2C)

Common aspect ratios scaled to a base resolution and snapped to a multiple. Wan 480p/720p presets emit native Wan target sizes directly.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `preset` | choice: `1:1 square`, `16:9 landscape`, `9:16 portrait`, `4:3 landscape`, `3:4 portrait`, `21:9 ultrawide`, `2.39:1 cinema`, `Wan 480p land`, … (+3) | default `"16:9 landscape"` | — |
| `base` | `INT` | default `1024`, range 64…8192 | Long edge target (ignored for Wan presets). |
| `multiple` | `INT` | default `64`, range 1…256 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `width` | `INT` | — |
| 1 | `height` | `INT` | — |


### ConditionalSwitchMEC

**Shown in the menu as:** Conditional Switch

Return value_true when condition is True, else value_false. Both inputs are wildcard (*) so it works for any type.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `condition` | `BOOLEAN` | default `True` | — |
| `value_true` | `*` |  | — |
| `value_false` | `*` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `out` | `*` | — |


### DimensionsSnapMEC

**Shown in the menu as:** Dimensions Snap (C2C)

Round (w,h) to the nearest multiple of N (default 64). Wan/Flux/SDXL all require dimensions divisible by 8/16/64 — this prevents shape-mismatch crashes at sample time.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `width` | `INT` | default `1024`, range 8…16384 | — |
| `height` | `INT` | default `1024`, range 8…16384 | — |
| `multiple` | `INT` | default `64`, range 1…256 | — |
| `direction` | choice: `nearest`, `down`, `up` | default `"down"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `width` | `INT` | — |
| 1 | `height` | `INT` | — |


### ExecutionTimerMEC

**Shown in the menu as:** Execution Timer (C2C)

Stopwatch: returns seconds since the *previous* execution of the same label. First tick returns 0. Passes input through unchanged.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `passthrough` | `*` |  | — |
| `label` | `STRING` | default `"stage_a"` | — |
| `reset` | `BOOLEAN` | default `False` | If true, this tick is treated as the first. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `passthrough` | `*` | — |
| 1 | `report` | `STRING` | — |
| 2 | `elapsed_seconds` | `FLOAT` | — |


### ImageBatchSliceMEC

**Shown in the menu as:** Image Batch Slice (C2C)

Extract a [start:end:step] range from an IMAGE batch. Negative end values count back from the end. step=2 keeps every second frame, etc.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |
| `start` | `INT` | default `0`, range -100000…100000 | — |
| `end` | `INT` | default `-1`, range -100000…100000 | Exclusive. -1 = end of batch. |
| `step` | `INT` | default `1`, range 1…1024 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | — |
| 1 | `frame_count` | `INT` | — |


### ImageBatchSplitMEC

**Shown in the menu as:** Image Batch Split (C2C)

Split an IMAGE batch into two pieces at a frame index OR by a fractional ratio (0.0–1.0).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |
| `mode` | choice: `index`, `ratio` | default `"index"` | — |
| `index` | `INT` | default `1`, range 0…100000 | — |
| `ratio` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `first_part` | `IMAGE` | — |
| 1 | `remainder` | `IMAGE` | — |
| 2 | `first_count` | `INT` | — |
| 3 | `remainder_count` | `INT` | — |


### ImageStatsProbeMEC

**Shown in the menu as:** Image Stats Probe (C2C)

Passthrough: returns the input image unchanged plus a stats report. Useful for debugging black-frame / over-bright generations.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | — |
| 1 | `report` | `STRING` | — |
| 2 | `mean` | `FLOAT` | — |
| 3 | `std` | `FLOAT` | — |
| 4 | `bright_pct` | `FLOAT` | — |


### MaskAreaProbeMEC

**Shown in the menu as:** Mask Area Probe (C2C)

Passthrough: returns the mask unchanged plus a coverage report. Per-frame coverage = (mask > threshold).mean().


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | — |
| `threshold` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | — |
| 1 | `report` | `STRING` | — |
| 2 | `coverage_mean_pct` | `FLOAT` | — |
| 3 | `coverage_min_pct` | `FLOAT` | — |
| 4 | `coverage_max_pct` | `FLOAT` | — |


### MaskBatchCombineMEC

**Shown in the menu as:** Mask Batch Combine (C2C)

Combine two MASK batches with one of: union (max), intersect (min), diff (A - B), xor, add (clamp), subtract (clamp). Sizes must match.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask_a` | `MASK` |  | — |
| `mask_b` | `MASK` |  | — |
| `op` | choice: `union`, `intersect`, `diff`, `xor`, `add`, `subtract` | default `"union"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | — |


### NumberLerpMEC

**Shown in the menu as:** Number Lerp (C2C)

Linear / smoothstep / cosine interpolation between two values. t is clamped to [0,1].


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `a` | `FLOAT` | default `0.0`, range -1000000000.0…1000000000.0, step 0.0001 | — |
| `b` | `FLOAT` | default `1.0`, range -1000000000.0…1000000000.0, step 0.0001 | — |
| `t` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.001 | — |
| `curve` | choice: `linear`, `smoothstep`, `cosine` | default `"linear"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `float` | `FLOAT` | — |
| 1 | `int` | `INT` | — |


### SeedListMEC

**Shown in the menu as:** Seed List Generator (C2C)

Generate N deterministic seeds from a base seed. Modes: increment (base, base+1, …), hash (sha256 of base+index, useful for de-correlated samples), random (mt19937 stream).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `base_seed` | `INT` | default `0`, range 0…4294967295 | — |
| `count` | `INT` | default `4`, range 1…1024 | — |
| `mode` | choice: `increment`, `hash`, `random` | default `"increment"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `first_seed` | `INT` | — |
| 1 | `csv_all_seeds` | `STRING` | — |


### TextTemplateMEC

**Shown in the menu as:** Text Template (C2C)

Substitute {a}{b}{c}{d} placeholders in a template string. Useful for building prompts from per-shot variables.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `template` | `STRING` | default `"a {a} of {b}, {c}, {d}"`, multiline | — |
| `a` | `STRING` | default `""` | — |
| `b` | `STRING` | default `""` | — |
| `c` | `STRING` | default `""` | — |
| `d` | `STRING` | default `""` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `text` | `STRING` | — |


---

## C2C/IO


### BatchVersionManagerMEC

**Shown in the menu as:** Batch Version Manager

Compute (and optionally atomically reserve) the next v### directory under <root>/<show>/<shot>/<task>/. Forward-slash output paths.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `root` | `STRING` | default `""` | Absolute output root (e.g. D:/projects/renders). |
| `show` | `STRING` | default `"show"` | Show / project name (top-level folder under root) |
| `shot` | `STRING` | default `"sh010"` | Shot identifier (folder under show) |
| `task` | `STRING` | default `"comp"` | Task name (folder under shot, e.g. comp, matte, render) |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `reserve` | `BOOLEAN` | default `False` | Atomically reserve the version with a .lock file. When False, only computes the path — no disk writes. |
| `padding` | `INT` | default `3`, range 1…6 | Zero-pad width for v### (3 → v001, 4 → v0001). |
| `max_retries` | `INT` | default `5`, range 1…50 | On lock-race contention, advance version and retry this many times. |
| `min_version` | `INT` | default `1`, range 1…999999 | Floor for the first version when no v### exists yet. |
| `forward_slash` | `BOOLEAN` | default `True` | When True (default), output paths use forward slashes for cross-platform compatibility. Set False to keep native (Windows backslash) separators. |
| `write_manifest` | `BOOLEAN` | default `True` | When `reserve=True`, also write `version_manifest.json` alongside the .lock containing workflow_hash + user + host + timestamp + show/shot/task triple. Provides full audit trail. Ignored when reserve=False. |

**Hidden inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `prompt` | `PROMPT` |  | — |
| `extra_pnginfo` | `EXTRA_PNGINFO` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `version_path` | `STRING` | Full path to the next-version directory (forward-slash by default). |
| 1 | `version_int` | `INT` | Integer version number that was allocated. |
| 2 | `version_label` | `STRING` | Padded version label such as v001. |
| 3 | `info_json` | `STRING` | JSON metadata: show, shot, task, user, host, timestamp, reservation status. |


---

## C2C/Inpaint


### InpaintCropProMEC

**Shown in the menu as:** Inpaint Crop Pro

Crop around mask for inpainting (lquesada API + Wan 2.2 Animate aware). Pair with Inpaint Stitch Pro (C2C).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `downscale_algorithm` | choice: `nearest`, `bilinear`, `bicubic`, `lanczos`, `box`, `hamming`, `area` | default `"bilinear"` | — |
| `upscale_algorithm` | choice: `nearest`, `bilinear`, `bicubic`, `lanczos`, `box`, `hamming`, `area` | default `"bicubic"` | — |
| `preresize` | `BOOLEAN` | default `False` | Resize input image before processing (lquesada-style). |
| `preresize_mode` | choice: `ensure minimum resolution`, `ensure maximum resolution`, `ensure minimum and maximum resolution` | default `"ensure minimum resolution"` | — |
| `preresize_min_width` | `INT` | default `1024`, range 0…16384, step 1 | — |
| `preresize_min_height` | `INT` | default `1024`, range 0…16384, step 1 | — |
| `preresize_max_width` | `INT` | default `16384`, range 0…16384, step 1 | — |
| `preresize_max_height` | `INT` | default `16384`, range 0…16384, step 1 | — |
| `mask_fill_holes` | `BOOLEAN` | default `True` | Mark fully-enclosed regions as masked. |
| `mask_expand_pixels` | `INT` | default `0`, range 0…16384, step 1 | Dilate mask by this many pixels. |
| `mask_invert` | `BOOLEAN` | default `False` | Invert mask (anything masked is kept). |
| `mask_blend_pixels` | `INT` | default `32`, range 0…64, step 1 | Pixels of feather for stitch blending (lquesada default 32). |
| `mask_hipass_filter` | `FLOAT` | default `0.1`, range 0.0…1.0, step 0.01 | Zero out mask values below this threshold. |
| `extend_for_outpainting` | `BOOLEAN` | default `False` | Extend image with edge-replicated padding for outpainting. |
| `extend_up_factor` | `FLOAT` | default `1.0`, range 0.01…100.0, step 0.01 | — |
| `extend_down_factor` | `FLOAT` | default `1.0`, range 0.01…100.0, step 0.01 | — |
| `extend_left_factor` | `FLOAT` | default `1.0`, range 0.01…100.0, step 0.01 | — |
| `extend_right_factor` | `FLOAT` | default `1.0`, range 0.01…100.0, step 0.01 | — |
| `context_from_mask_extend_factor` | `FLOAT` | default `1.2`, range 0.1…100.0, step 0.01 | Resize context bbox around the mask. 1.0 = exact mask bbox. >1 grows outward (1.5 = +50% on every side). <1 shrinks INWARD so the crop is tighter than the mask bbox (0.7 = inset 15% on every side). Useful for 16:9 / portrait single-still crops where the mask bbox is larger than the subject you actually want sampled. |
| `auto_context_factor` | `BOOLEAN` | default `False` | Override context_from_mask_extend_factor automatically so the mask fills ~70% of the resulting crop area. Uses actual mask AREA (not just bbox) to handle sparse / thin / non-convex masks. Result clamped to [0.30, 10.00]. OFF = use the manual slider above. Recommended ON for batch / unattended runs. |
| `aspect_preset` | choice: `Custom`, `Auto`, `Square`, `16:9`, `9:16`, `4:3`, `3:4` | default `"Custom"` | Force the output crop aspect ratio.   Custom : use output_target_width / _height as-is (legacy lquesada behaviour - DEFAULT).   Auto   : pick best of {Square, 16:9, 9:16, 4:3, 3:4} from the mask bbox aspect.   Square / 16:9 / 9:16 / 4:3 / 3:4 : keep the SHORTER of (output_target_width, output_target_height) and recompute the longer side from this AR. Multiple-of-padding rounding still applies downstream. |
| `output_resize_to_target_size` | `BOOLEAN` | default `True` | Force output to a specific resolution for sampling. |
| `output_target_width` | `INT` | default `512`, range 64…16384, step 1 | — |
| `output_target_height` | `INT` | default `512`, range 64…16384, step 1 | — |
| `output_padding` | choice: `0`, `8`, `16`, `32`, `64`, `128`, `256`, `512` | default `"32"` | — |
| `device_mode` | choice: `cpu (compatible)`, `gpu (much faster)` | default `"gpu (much faster)"` | — |
| `wan_align_multiple` | `INT` | default `16`, range 1…256, step 1 | Force final crop W/H to multiples of this (Wan VAE patchify; 16 recommended). |
| `wan_temporal_smooth_frames` | `FLOAT` | default `0.0`, range 0.0…64.0, step 0.1 | Gaussian smoothing of mask along time axis (frames). 0 disables. |
| `wan_stable_crop` | `BOOLEAN` | default `True` | Use a single union bbox across all frames (Wan replacement-mode). |
| `wan_mask_polarity` | choice: `regenerate_subject`, `preserve_subject` | default `"regenerate_subject"` | regenerate_subject: mask=1 -> regenerate (lquesada).  preserve_subject: mask=0 -> regenerate (Wan2.2 replacement: mask=1 keeps environment). |
| `inpaint_mask_mode` | choice: `hard_binary`, `slight_feather`, `soft_blend` | default `"hard_binary"` | What the inpaint sampler sees: hard_binary (crisp), slight_feather (gentle), soft_blend (very soft). |
| `stitch_blend_mode` | choice: `gaussian`, `edge_aware`, `laplacian_pyramid`, `frequency_blend`, `video_stable` | default `"gaussian"` | How the result is composited back: gaussian, edge_aware (Sobel), laplacian_pyramid, frequency_blend, video_stable. |
| `blend_radius` | `INT` | default `32`, range 1…256, step 1 | Feather radius for the stitch blend mask (independent of mask_blend_pixels). |
| `video_stable_temporal_sigma` | `FLOAT` | default `3.0`, range 0.0…10.0, step 0.5 | [video_stable only] Temporal Gaussian sigma in frames. 3.0 ≈ 9-frame window. Higher = smoother but laggier on fast motion. 0 = off. |
| `video_stable_dilate_px` | `INT` | default `-1`, range -1…128, step 1 | [video_stable only] Pixels to push the blend zone into background BEFORE feathering. -1 = derive from blend_radius. 16-32 typical. |
| `video_stable_blur_sigma` | `FLOAT` | default `-1.0`, range -1.0…128.0, step 0.5 | [video_stable only] Spatial Gaussian sigma for the wide feather. -1 = derive from blend_radius (×0.75). Match to dilate value. |
| `fill_masked_area` | choice: `none`, `edge_pad`, `neutral_gray`, `original` | default `"none"` | Fill masked region in the cropped image: none, edge_pad (Gaussian smear), neutral_gray, original. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | — |
| `optional_context_mask` | `MASK` |  | — |
| `roto_quality` | `BOOLEAN` | default `False` | Roto-Sync mode: tightens the inpaint seam for clean alpha edges. Forces laplacian_pyramid blend, halves blend_radius (min 4), and pre-erodes the inpaint mask by 1 px so the stitch falls just inside the subject. Safe to leave OFF for general inpaint; turn ON for compositing / roto / VFX work where boundaries must not bleed. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `stitcher` | `STITCHER` | — |
| 1 | `cropped_image` | `IMAGE` | — |
| 2 | `inpaint_mask` | `MASK` | — |
| 3 | `stitch_blend_mask` | `MASK` | — |
| 4 | `info` | `STRING` | — |


### InpaintMaskPrepareMEC

**Shown in the menu as:** Inpaint Mask Prepare

Clean, grow, and prepare dual masks: inpaint_mask for model + stitch_blend_mask for composite.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | Raw input mask (B,H,W) |
| `fill_holes` | `BOOLEAN` | default `True` | Fill interior holes in the mask |
| `remove_small_regions` | `BOOLEAN` | default `True` | Remove small disconnected blobs |
| `min_region_area` | `INT` | default `100`, range 0…100000, step 10 | Minimum region area in pixels to keep |
| `grow_pixels` | `INT` | default `4`, range 0…256, step 1 | Dilate mask by this many pixels |
| `inpaint_edge_mode` | choice: `hard_binary`, `slight_feather` | default `"hard_binary"` | Edge style for inpaint mask: hard_binary or slight_feather |
| `stitch_edge_mode` | choice: `gaussian`, `edge_aware` | default `"gaussian"` | Edge style for stitch blend mask: gaussian or edge_aware |
| `stitch_feather_radius` | `INT` | default `16`, range 1…128, step 1 | Feather radius for the stitch blend mask |
| `temporal_smooth` | `BOOLEAN` | default `False` | Apply Gaussian temporal smoothing along batch dimension (for video) |
| `temporal_sigma` | `FLOAT` | default `1.5`, range 0.1…10.0, step 0.1 | Temporal Gaussian sigma in frames |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `reference_image` | `IMAGE` |  | Reference image for edge-aware stitch blend mask (required for edge_aware mode) |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `inpaint_mask` | `MASK` | — |
| 1 | `stitch_blend_mask` | `MASK` | — |
| 2 | `debug_preview` | `IMAGE` | — |
| 3 | `info` | `STRING` | — |


### InpaintPasteBackMEC

**Shown in the menu as:** Inpaint Paste Back

Paste inpainted crop back using STITCHER, with optional feathered rectangle edges.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `stitcher` | `STITCHER` |  | Stitcher dict from InpaintCropProMEC |
| `inpainted_image` | `IMAGE` |  | Inpainted crop result (B,H,W,C) |
| `upscale_method` | choice: `lanczos`, `bicubic`, `bilinear`, `nearest`, `area` | default `"bicubic"` | Interpolation method for resizing crop back |
| `feather_edges` | `BOOLEAN` | default `False` | Apply Gaussian feather at crop boundary |
| `feather_radius` | `INT` | default `16`, range 0…64, step 1 | Feather radius in pixels (only used if feather_edges) |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `info` | `STRING` | — |


### InpaintStitchProMEC

**Shown in the menu as:** Inpaint Stitch Pro

Stitch inpainted image back into the original (lquesada-compatible) with blend overrides + color match.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `stitcher` | `STITCHER` |  | — |
| `inpainted_image` | `IMAGE` |  | — |
| `blend_mode_override` | choice: `from_crop`, `gaussian`, `edge_aware`, `laplacian_pyramid`, `frequency_blend`, `video_stable` | default `"from_crop"` | Override the blend mode chosen at crop time, or use 'from_crop'. |
| `color_match` | `BOOLEAN` | default `False` | Apply mean+std color transfer before stitching to reduce color shift. |
| `stitch_temporal_sigma` | `FLOAT` | default `0.0`, range 0.0…10.0, step 0.5 | Post-hoc temporal Gaussian smoothing applied to the per-frame blend mask before compositing. 0 = off. 2-4 = good for jittery segmentation video (≈3 means a 9-frame window). Works on top of any blend mode (gaussian / edge_aware / video_stable / etc.). |
| `stitch_dilate_px` | `INT` | default `0`, range 0…128, step 1 | Optional dilation (in pixels) applied to the binary core of the blend mask before temporal smoothing — pushes the seam into flat background. Use with stitch_temporal_sigma > 0 for jittery video. 0 = off. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `roto_quality_override` | choice: `from_crop`, `force_on`, `force_off` | default `"from_crop"` | Roto-Sync at stitch time. 'from_crop' honors the InpaintCropProMEC flag (recommended). 'force_on' applies tight-seam stitching even if crop didn't set it; 'force_off' disables it. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `blend_mask_used` | `MASK` | — |
| 2 | `info` | `STRING` | — |


### ProPainterMEC

**Shown in the menu as:** ProPainter — Temporal / Remove / Stitch / Refine / Flow

Unified ProPainter node. Absorbs ProPainterTemporal / Remove / Stitch / StitchRefine / FlowRefine. Pick a mode and only the relevant widgets are read; others are ignored.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mode` | choice: `temporal`, `remove`, `stitch`, `stitch_refine`, `flow` | default `"remove"` | Pick the ProPainter operation. Each mode reads a different subset of the optional inputs/widgets below. |
| `use_half` | `BOOLEAN` | default `True` | — |
| `color_match_mode` | choice: `off`, `reinhard`, `lab`, `lab_transfer`, `none` | default `"reinhard"` | Per-frame masked colour match between fill and surroundings (ignored in 'flow' mode). |
| `raft_iter` | `INT` | default `12`, range 1…100 | — |
| `neighbor_stride` | `INT` | default `5`, range 1…32 | — |
| `ref_stride` | `INT` | default `10`, range 1…64 | — |
| `subvideo_length` | `INT` | default `8`, range 2…300 | Frames per InpaintGenerator window. 8GB cards: 8-30. 12GB: 40-60. 24GB: 80+. |
| `raft_chunk` | `INT` | default `16`, range 1…64 | Frame-pairs per RAFT forward pass. |
| `blend_boundary` | `BOOLEAN` | default `True` | [temporal] Blend the inpaint with original at the crop boundary using stitch_data. |
| `remove_quality` | choice: `fast`, `balanced`, `quality` | default `"balanced"` | [remove] Preset that overrides raft_iter/neighbor_stride/ref_stride/subvideo_length. |
| `remove_dilate_pixels` | `INT` | default `3`, range 0…32 | [remove] Dilate the mask by N px before filling to cover anti-aliasing. |
| `boundary_band_pixels` | `INT` | default `12`, range 0…96 | [stitch] Width of the boundary band re-painted (0 = no boundary repaint). |
| `preserve_inpaint_center` | `BOOLEAN` | default `True` | [stitch] Keep generative inpaint untouched at the centre, only repaint the seam. |
| `upscale_method` | choice: `lanczos`, `bicubic`, `bilinear`, `nearest` | default `"lanczos"` | [stitch] How to scale the inpainted crop to the canvas region. |
| `ring_pixels` | `INT` | default `8`, range 1…64 | [stitch_refine] Half-width of the seam ring in pixels. |
| `flow_consistency_thr` | `FLOAT` | default `1.5`, range 0.0…20.0, step 0.05 | [flow] Forward/backward consistency threshold (pixels). |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | [temporal/remove] Source frames. |
| `masks` | `MASK` |  | [temporal/remove] Region to inpaint. |
| `stitch_data` | `STITCH_DATA` |  | [temporal/stitch/stitch_refine] STITCH_DATA from InpaintCropProMEC. |
| `inpainted_image` | `IMAGE` |  | [stitch] Crop-sized generative inpaint output. |
| `stitched_image` | `IMAGE` |  | [stitch_refine] Already-stitched canvas image (output of stitch). |
| `mask_override` | `MASK` |  | [stitch_refine] Optional explicit canvas-space mask (overrides stitch_data). |
| `frame_a` | `IMAGE` |  | [flow] First frame for optical-flow computation. |
| `frame_b` | `IMAGE` |  | [flow] Second frame for optical-flow computation. |
| `flow_mask` | `MASK` |  | [flow] Optional mask restricting the consistency visualisation. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image_out` | `IMAGE` | Primary image output (inpainted / filled / stitched / warped per mode). |
| 1 | `mask_out` | `MASK` | Primary mask output (fill mask / repaint mask / consistency per mode). |
| 2 | `aux_image` | `IMAGE` | Auxiliary IMAGE (flow_field_rgb in 'flow' mode, zeros otherwise). |
| 3 | `aux_mask` | `MASK` | Auxiliary MASK (reserved, zeros for now). |
| 4 | `info` | `STRING` | Info string with timings, coverage, mode-specific stats. |


---

## C2C/Keying


### LuminanceKeyerMEC

**Shown in the menu as:** Luminance Keyer — Highlights / Shadows / Custom

Professional luminance keyer inspired by Nuke's LumaKeyer.
Computes ITU-R BT.709 luminance from an image and extracts a matte
using adjustable thresholds with smooth S-curve falloff and gamma.
Modes: auto, highlights, midtones, shadows, custom.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Input image(s) to extract luminance key from. |
| `mode` | choice: `auto`, `highlights`, `midtones`, `shadows`, `custom` | default `"auto"` | Preset luminance range or custom thresholds. auto: Analyzes image brightness to pick best range. highlights: Keys bright regions (0.7–1.0). midtones: Keys mid-range luminance (0.3–0.7). shadows: Keys dark regions (0.0–0.3). custom: Uses the low/high sliders directly. |
| `low` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | Low threshold – pixels with luminance below this become 0 in the mask. Only used directly in custom mode; presets override this. |
| `high` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | High threshold – pixels with luminance above this become 1 in the mask. Only used directly in custom mode; presets override this. |
| `gamma` | `FLOAT` | default `1.0`, range 0.01…10.0, step 0.01 | Gamma correction applied after keying. >1 compresses mask toward black (reduces coverage). <1 expands mask toward white (increases coverage). |
| `falloff` | `FLOAT` | default `1.0`, range 0.0…10.0, step 0.1 | Smoothness of the transition between low and high thresholds. 0 = hard binary edge. 1 = standard smooth. >1 = very gradual transition. |
| `invert` | `BOOLEAN` | default `False` | Invert the output mask (swap keyed and unkeyed regions). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | Luminance-based matte for the chosen brightness range. |
| 1 | `info` | `STRING` | JSON summary of effective mode and thresholds used. |


---

## C2C/Matting


### BackgroundRemoverMEC

**Shown in the menu as:** Background Remover — RMBG / BiRefNet

One-click background removal using RMBG-2.0 or BiRefNet.
Outputs foreground (premultiplied RGB) and high-quality alpha mask.
Ideal for portraits, product photos, and compositing workflows.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Input image(s) to remove background from. |
| `model_name` | choice: `birefnet_general`, `birefnet_portrait`, `rmbg_2.0` | default `"birefnet_general"` | Background removal model: rmbg_2.0: General-purpose, fast. birefnet_general: High-detail edges. birefnet_portrait: Optimized for human portraits. |
| `threshold` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Alpha threshold for hard mask (0=soft, 0.5=balanced, 1=hard). |
| `invert` | `BOOLEAN` | default `False` | Invert the alpha mask (keep background instead). |
| `mask_blur` | `INT` | default `0`, range 0…50, step 1 | Gaussian blur applied to final mask edges. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `keep_model_loaded` | `BOOLEAN` | default `True` | Keep model in VRAM between runs. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `foreground` | `IMAGE` | Foreground RGB image with background removed (premultiplied by alpha). |
| 1 | `mask` | `MASK` | High-quality alpha mask for the foreground subject. |
| 2 | `info` | `STRING` | JSON summary of the parameters used for this run. |


---

## C2C/ModelAnalysis


### VAEBlockInspectorMEC

**Shown in the menu as:** VAE Block Inspector

Per-block weight stats for a VAE (mean/std/abs_mean/count).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vae` | `VAE` |  | VAE whose per-block weight statistics will be inspected. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `anomaly_threshold` | `FLOAT` | default `5.0`, range 1.5…50.0, step 0.5 | Tensors whose abs_mean exceeds this multiple of the cohort median are flagged as magnitude outliers. Lower => more sensitive (more flags). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `report_json` | `STRING` | Per-block weight statistics (mean/std/abs_mean/count) plus outlier details as JSON. |
| 1 | `outlier_tensor_names` | `STRING` | Newline-separated list of tensor names flagged as outliers. |
| 2 | `anomaly_score` | `FLOAT` | Aggregate anomaly score in [0, 1] (higher means more outliers detected). |


### VAESimilarityAnalyserMEC

**Shown in the menu as:** VAE Similarity Analyser

Cosine similarity between two VAEs (per tensor + per block).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vae_a` | `VAE` |  | First VAE to compare. |
| `vae_b` | `VAE` |  | Second VAE to compare. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `include_per_tensor` | `BOOLEAN` | default `False` | Include per-tensor cosine entries in the JSON report (verbose). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `report_json` | `STRING` | Full similarity report as JSON (per-block cosine, missing keys, optional per-tensor). |
| 1 | `global_cosine` | `FLOAT` | Global cosine similarity across all common tensors. |
| 2 | `most_divergent_blocks` | `STRING` | JSON list of the 10 most divergent blocks (lowest cosine first). |


---

## C2C/Paint


### MECAdvancedPaintCanvas

**Shown in the menu as:** Advanced Paint Canvas

Interactive paint canvas with procedural mask math: hardness, expansion, and blur stages are applied in order to the alpha channel of painted strokes.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `canvas_width` | `INT` | default `512`, range 64…4096, step 8 | Canvas width in pixels. |
| `canvas_height` | `INT` | default `512`, range 64…4096, step 8 | Canvas height in pixels. |
| `brush_type` | choice: `paint`, `mask_only` | default `"paint"` | paint: composite color strokes onto the image. mask_only: build the mask without coloring. |
| `brush_color` | `STRING` | default `"#000000"` | Hex color used by the paint brush (ignored in mask_only mode). |
| `brush_opacity` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | Brush stroke opacity (0 = invisible, 1 = fully opaque). |
| `brush_hardness` | `FLOAT` | default `0.8`, range 0.0…1.0, step 0.01 | Brush profile hardness (0 = soft, 1 = hard edge). |
| `brush_size` | `INT` | default `20`, range 1…500, step 1 | Brush diameter in pixels. |
| `mask_hardness` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Threshold for the solid inner core: pixels brighter than (1 - hardness) clamp to 1.0. |
| `mask_expansion` | `INT` | default `0`, range -100…100, step 1 | Morphological dilate (positive) / erode (negative) in pixels. |
| `mask_blur_radius` | `FLOAT` | default `0.0`, range 0.0…100.0, step 0.1 | Gaussian blur radius applied to the mask edge in pixels. |
| `mask_blur_strength` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | Blend factor between the hard mask (0) and the fully-blurred mask (1). |
| `canvas_data` | `STRING` | default `""` | Internal base64 PNG payload from the JS canvas widget. Do not edit manually. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `reference_image` | `IMAGE` |  | Optional background image; painted strokes are composited over it when supplied. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `painted_image` | `IMAGE` | Painted RGB image (composited over reference_image when supplied). |
| 1 | `processed_mask` | `MASK` | Processed mask after hardness, expansion, and blur stages. |


### MECBuilderSampler

**Shown in the menu as:** Builder Sampler

KSampler with adaptive CFG curves (Constant, Linear, Ease Down) plus an optional self-correction polish pass and resolution presets.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | Diffusion model to sample with. |
| `positive` | `CONDITIONING` |  | Positive conditioning. |
| `negative` | `CONDITIONING` |  | Negative conditioning. |
| `steps` | `INT` | default `20`, range 1…200 | Number of sampling steps. |
| `cfg` | `FLOAT` | default `8.0`, range 0.0…30.0, step 0.1 | Starting CFG scale (also constant CFG when cfg_mode is Constant). |
| `sampler_name` | choice: `euler`, `dpmpp_2m` |  | Sampler algorithm. |
| `scheduler` | choice: `normal`, `karras` |  | Sigma schedule. |
| `denoise` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | Denoise strength (1.0 = full sampling, lower = partial img2img). |
| `cfg_mode` | choice: `Constant`, `Linear`, `Ease Down` | default `"Constant"` | Adaptive CFG curve shape across steps. |
| `cfg_finish` | `FLOAT` | default `4.0`, range 0.0…30.0, step 0.1 | Final CFG value at the end of the schedule (used by Linear / Ease Down). |
| `cfg_pivot` | `FLOAT` | default `5.0`, range 0.0…30.0, step 0.1 | Pivot CFG value used by Ease Down to control the curve knee. |
| `self_correction` | `BOOLEAN` | default `False` | Run a 2-step polish pass after the main sampling. |
| `resolution_preset` | choice: `SDXL (1024x1024)`, `SD1.5 (512x512)`, `Custom` | default `"SD1.5 (512x512)"` | Preset resolution; choose Custom to use custom_width/custom_height. |
| `custom_width` | `INT` | default `512`, range 64…4096, step 8 | Custom output width in pixels (used when preset is Custom). |
| `custom_height` | `INT` | default `512`, range 64…4096, step 8 | Custom output height in pixels (used when preset is Custom). |
| `seed` | `INT` | default `0`, range 0…18446744073709551615 | Random seed. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vae` | `VAE` |  | Optional VAE; when provided, decodes the latent into preview_image. |
| `latent_image` | `LATENT` |  | Optional input latent (img2img). When omitted, an empty latent is created. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `latent` | `LATENT` | Sampled latent (with optional polish pass applied). |
| 1 | `preview_image` | `IMAGE` | VAE-decoded preview image when a VAE is provided (zero image otherwise). |


### MECContextInpainter

**Shown in the menu as:** Context Inpainter / Fixer

Smart-blend an inpainted region back over the original image with crop padding, feathered blend mask, optional color correction, lightness rescue, and differential diffusion.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `original_image` | `IMAGE` |  | Original (un-inpainted) image used as the blend base. |
| `mask` | `MASK` |  | Mask defining the inpainted region. |
| `inpainted_image` | `IMAGE` |  | Inpainted image to blend back over the original. |
| `crop_padding` | `FLOAT` | default `1.2`, range 1.0…2.0, step 0.01 | Multiplier extending the masked bbox so the inpaint sees more context. |
| `blend_softness` | `FLOAT` | default `8.0`, range 0.0…200.0, step 0.5 | Gaussian feather radius applied to the blend mask in pixels. |
| `mask_expansion_blend` | `INT` | default `0`, range -100…100, step 1 | Per-blend dilate (positive) / erode (negative) of the blend mask in pixels. |
| `enable_color_correction` | `BOOLEAN` | default `True` | Reinhard mean/std color match between original and inpainted regions. |
| `enable_lightness_rescue` | `BOOLEAN` | default `True` | Lift CIE LAB L-channel of the inpaint when it is more than ~5% darker. |
| `enable_differential_diffusion` | `BOOLEAN` | default `False` | Use \|orig - inpaint\| as a soft preservation weight to keep unchanged pixels. |
| `sampling_mask_blur_size` | `INT` | default `21`, range 0…201, step 1 | Kernel size (odd) for the additional blur on the output debug mask. |
| `sampling_mask_blur_strength` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | Blend factor for the sampling mask blur applied to debug_mask. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `face_positive_prompt` | `STRING` | default `""`, multiline | Optional region-attached positive prompt; parsed for `{a\|b\|c}` wildcards per detected mask region and logged. Does NOT sample here -- pair with a FaceInpaint/KSampler downstream. |
| `face_negative_prompt` | `STRING` | default `""`, multiline | Same as face_positive_prompt but for negatives. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `blended_image` | `IMAGE` | Final composited image with the inpainted region blended back into the original. |
| 1 | `debug_mask` | `MASK` | Debug mask used for the blend (after expansion and blur). |


### MECFaceFixer

**Shown in the menu as:** Face Fixer

Auto face detection (YOLO11) + per-face crop + AI pre-upscale + context-aware sampling + smart blend with wildcard per-face prompts. Behavioural clone of Forbidden Vision's Fixer with Impact-Pack wildcard syntax.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Source image (single frame or batch; processed independently per frame). |
| `model` | `MODEL` |  | Diffusion model to sample with. |
| `positive` | `CONDITIONING` |  | Base positive conditioning. Wildcards in face_positive_prompt override per face. |
| `negative` | `CONDITIONING` |  | Base negative conditioning. Wildcards in face_negative_prompt override per face. |
| `vae` | `VAE` |  | VAE used to encode/decode the per-face crops. |
| `face_model` | choice: `none` | default `"none"` | YOLO11 face-detection .pt/.onnx in ComfyUI/models/ultralytics/bbox/. Choose 'none' to use the optional mask input instead. |
| `confidence` | `FLOAT` | default `0.5`, range 0.05…0.95, step 0.01 | Minimum detection confidence. |
| `max_faces` | `INT` | default `8`, range 0…32, step 1 | Maximum number of faces to process per frame (0 = all). |
| `crop_padding` | `FLOAT` | default `1.4`, range 1.0…3.0, step 0.05 | Bbox padding multiplier so the sampler sees context around each face. |
| `crop_resolution` | `INT` | default `768`, range 256…2048, step 64 | Resize each face crop to this longer-side resolution before sampling. |
| `denoise` | `FLOAT` | default `0.4`, range 0.0…1.0, step 0.01 | Per-face denoise strength (0.3 = subtle, 0.7 = aggressive reshape). |
| `steps` | `INT` | default `20`, range 1…100 | Sampling steps per face. |
| `cfg` | `FLOAT` | default `6.0`, range 0.0…30.0, step 0.1 | CFG scale for face sampling. |
| `sampler_name` | choice: `euler`, `dpmpp_2m` | default `"euler"` | Sampler algorithm. |
| `scheduler` | choice: `normal`, `karras` | default `"normal"` | Sigma schedule. |
| `seed` | `INT` | default `0`, range 0…18446744073709551615 | Base seed; each face gets seed+i. |
| `blend_softness` | `FLOAT` | default `6.0`, range 0.0…64.0, step 0.5 | Feather radius (px) on the per-face blend mask. |
| `mask_dilate` | `INT` | default `4`, range -32…32, step 1 | Dilate (>0) / erode (<0) of the per-face blend mask. |
| `color_match` | `BOOLEAN` | default `True` | Reinhard mean/std colour match per face. |
| `lightness_rescue` | `BOOLEAN` | default `True` | Lift the per-face L channel if the sample comes back darker than the original. |
| `differential_diffusion` | `BOOLEAN` | default `True` | Weight the blend by \|orig - sampled\| so unchanged pixels stay sharp. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | Optional manual face mask. Used directly when face_model='none' or detection finds nothing. |
| `upscale_model` | `UPSCALE_MODEL` |  | Optional UPSCALE_MODEL applied to faces below crop_resolution before sampling. |
| `face_positive_prompt` | `STRING` | default `""`, multiline | Per-face positive prompt with wildcards: [SEP] separates faces; [ASC]/[DSC]/[ASC-SIZE]/[DSC-SIZE] order; [SKIP] leaves a face untouched. |
| `face_negative_prompt` | `STRING` | default `""`, multiline | Same syntax as face_positive_prompt for negatives. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | Image with detected faces detailed and blended back over the original. |
| 1 | `face_mask` | `MASK` | Combined face-detection mask covering all processed faces. |
| 2 | `info_json` | `STRING` | JSON metadata: per-face bbox, score, prompt, denoise. |


### MECToneRefiner

**Shown in the menu as:** Tone Refiner

Auto-correct tone (black/white-point + gray-world), optionally upscale, and apply a fake center-focus depth-of-field blur.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Image to refine. |
| `neural_corrector` | `BOOLEAN` | default `True` | Enable deterministic tone + gray-world correction (not a learned model). |
| `corrector_tone` | `FLOAT` | default `0.6`, range 0.0…1.0, step 0.01 | Blend amount toward the tone-corrected image (0 = original, 1 = full correction). |
| `corrector_color` | `FLOAT` | default `0.4`, range 0.0…1.0, step 0.01 | Blend amount toward the gray-world color-corrected image. |
| `highlight_protection` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Roll-off applied above 95th-percentile to prevent highlight clipping (0 = none, 1 = strong). |
| `shadow_lift` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | Lift shadows below the 5th-percentile (0 = none, 1 = strong; mirrors highlight_protection on the dark side). |
| `enable_upscale` | `BOOLEAN` | default `False` | Upscale by upscale_factor; uses upscale_model if provided, bicubic otherwise. |
| `upscale_factor` | `FLOAT` | default `1.5`, range 1.0…4.0, step 0.05 | Upscale multiplier applied when enable_upscale is True. |
| `ai_enable_dof` | `BOOLEAN` | default `False` | Apply depth-based DOF (uses depth_map if connected, else fake center-focus). |
| `ai_dof_strength` | `FLOAT` | default `1.0`, range 0.0…4.0, step 0.05 | Strength of the DOF blur (also scales the maximum blur radius). |
| `ai_dof_focus_depth` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.01 | Focus plane: when depth_map is connected this is the in-focus depth value (0=near,1=far); without depth_map it controls center-focus tightness. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `latent` | `LATENT` |  | Optional pre-existing latent; passed through when supplied (skips VAE encode). |
| `vae` | `VAE` |  | VAE used to encode the refined image into refined_latent (when auto_upscale is True). |
| `upscale_model` | `UPSCALE_MODEL` |  | Optional UPSCALE_MODEL (RealESRGAN / 4x-NMKD / etc.). When connected and enable_upscale is True, used instead of bicubic and resized to upscale_factor. |
| `depth_map` | `MASK` |  | Optional depth map (0=near,1=far). Drives DOF when connected; replaces the fake center-focus radial gradient. |
| `auto_upscale` | `BOOLEAN` | default `True` | When True (default, back-compat), encode the refined image through the supplied VAE to produce `refined_latent`. Set False to skip the VAE-encode and return a zero placeholder. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `refined_image` | `IMAGE` | Tone- and color-corrected image (optionally upscaled and DOF-blurred). |
| 1 | `refined_latent` | `LATENT` | VAE-encoded latent of the refined image (zero placeholder when auto_upscale is False). |


---

## C2C/Pipeline


### MaskRefineMEC

**Shown in the menu as:** Mask Refiner

Unified, training-free mask refinement. Toggle each stage on or off and chain them in fixed order: hole-fill → morphology → thin-structure recover → joint bilateral → guided filter → DenseCRF → edge-snap → optional CascadePSP-style multi-pass → feather → gamma → threshold. No model weights. Optional deps: opencv-contrib-python (ximgproc joint bilateral), scikit-image (thin recovery), scipy (hole fill / morphology helpers), pydensecrf (DenseCRF). Each missing dep silently disables only its stage; the rest still run.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | RGB guide. Required for all edge-aware stages. |
| `mask` | `MASK` |  | Mask to refine (soft or hard). |
| `preset` | choice: `balanced`, `fast`, `hair`, `aggressive`, `crf_heavy` | default `"balanced"` | Picks sensible numeric defaults for every enabled stage. Override with `advanced_overrides_json` if needed. |
| `auto_edge_lock` | `BOOLEAN` | default `True` | Pin the mask to real image edges to stop bleeding into the background. Force-enables guided_filter + edge_snap with parameters tuned by `subject_class`. Recommended for any face / garment / product / hair workflow. |
| `subject_class` | choice: `general`, `face`, `hair`, `garment`, `object`, `hard_surface` | default `"general"` | Tunes auto_edge_lock for the dominant subject:   general     — balanced edge protection   face        — tight 2-3 px band, high snap, no morph dilate   hair        — thin-recover ON, soft band, low snap   garment     — medium band, medium snap, close holes   object      — medium band, high snap, fill holes   hard_surface— thin band, max snap, threshold > 0.5 |
| `enable_hole_fill` | `BOOLEAN` | default `False` | — |
| `morph_op` | choice: `none`, `close`, `open`, `dilate`, `erode` | default `"none"` | — |
| `enable_thin_recover` | `BOOLEAN` | default `False` | — |
| `enable_joint_bilateral` | `BOOLEAN` | default `False` | — |
| `enable_guided_filter` | `BOOLEAN` | default `True` | — |
| `enable_dense_crf` | `BOOLEAN` | default `False` | — |
| `enable_edge_snap` | `BOOLEAN` | default `False` | — |
| `cascade_passes` | `INT` | default `0`, range 0…5 | — |
| `feather_sigma` | `FLOAT` | default `0.0`, range 0.0…20.0, step 0.1 | — |
| `gamma` | `FLOAT` | default `1.0`, range 0.1…5.0, step 0.05 | — |
| `threshold` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | — |
| `enable_domain_transform` | `BOOLEAN` | default `False` | Gastal & Oliveira (2011) Domain Transform RGB-edge filter. Often sharper than guided filter on hair / fine detail; much faster than DenseCRF. Requires opencv-contrib-python (ximgproc). |
| `enable_color_decontaminate` | `BOOLEAN` | default `False` | Push alpha in the boundary band toward 0/1 using local LAB-distance to fg/bg means. Fixes 'halo' alpha bleed when bg has similar luminance. |
| `enable_unsharp_alpha` | `BOOLEAN` | default `False` | Sharpen the soft alpha (α + amount·(α − gauss(α))). |
| `enable_anti_alias` | `BOOLEAN` | default `False` | Sub-pixel boundary smoothing (bilinear up 2× → soft contrast → down). |
| `enable_chroma_lock` | `BOOLEAN` | default `False` | When fg/bg luminance is similar but chroma differs, weight the boundary by LAB chroma gradient instead of luma. Helps red-on-red, green-on-green, etc. |
| `enable_speck_removal` | `BOOLEAN` | default `False` | Drop foreground components below `speck_min_area` and fill background holes inside the subject. |
| `enable_temporal_smooth` | `BOOLEAN` | default `False` | Bidirectional alpha EMA across the batch dim (for VIDEO masks only). Removes flicker without lag. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `advanced_overrides_json` | `STRING` | default `""`, multiline | Optional JSON overriding any preset numeric. Example: {"gf_radius":12, "jb_sigma_color":40, "crf_iterations":8}. Recognised keys: hole_fill_threshold, morph_radius, thin_threshold, thin_min_branch_len, thin_branch_dilate, jb_diameter, jb_sigma_color, jb_sigma_space, gf_radius, gf_epsilon, crf_iterations, crf_gauss_sxy, crf_bilateral_sxy, crf_bilateral_srgb, edge_snap_strength, edge_snap_band, dt_sigma_s, dt_sigma_r, decontam_band, decontam_strength, unsharp_sigma, unsharp_amount, anti_alias_strength, chroma_lock_strength, chroma_lock_band, speck_min_area, speck_fill_holes_below, temporal_alpha, temporal_bidi. |
| `enable_integrity_check` | `BOOLEAN` | default `False` | Compute per-frame integrity stats on the refined mask (coverage, abrupt drops, frame-to-frame jumps) and append them to the `info` JSON. Adds negligible cost for single frames; cheap for short clips. |
| `integrity_drop_threshold` | `FLOAT` | default `0.4`, range 0.0…1.0, step 0.01 | Relative coverage drop that flags a frame. |
| `integrity_jump_threshold` | `FLOAT` | default `0.15`, range 0.0…1.0, step 0.01 | Relative frame-to-frame coverage jump that flags a frame. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | Refined mask, same (B,H,W) as input. |
| 1 | `alpha` | `MASK` | Soft alpha (same as mask before threshold). |
| 2 | `preview` | `IMAGE` | RGB×alpha preview for quick visual diff. |
| 3 | `info` | `STRING` | JSON describing which stages ran / were skipped. |


### SAMViTMattePipelineMEC

**Shown in the menu as:** SAM + ViTMatte Pipeline — Full Quality

SAM + ViTMatte combined pipeline for compositing-grade alpha mattes. Iterative SAM refinement → edge-aware matting → multi-scale fusion → cleanup.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `sam_model` | `SAM_MODEL` |  | Loaded SAM model from SAM Model Loader |
| `image` | `IMAGE` |  | Input image / batch to segment + matte |
| `subject_type` | choice: `custom`, `hair`, `fur`, `cloth`, `skin_face`, `hard_edge`, `soft_glow` | default `"custom"` | Auto-tune trimap & matting params based on subject boundary character.   custom    : honor manual widgets verbatim (default).   hair      : SAM → ViTMatte, wide trimap, high detail (portraits).   fur       : SAM → ViTMatte, very wide trimap (animals).   cloth     : tighter trimap, structural edges preserved.   skin_face : multi-scale guided, soft skin boundary.   hard_edge : minimal trimap, binary feel (vehicles, props).   soft_glow : laplacian blend, very wide soft band. |
| `points_json` | `STRING` | default `"[]"`, multiline | JSON array: [{"x":100,"y":200,"label":1}, ...] |
| `bbox_json` | `STRING` | default `""` | Bounding box: [x1,y1,x2,y2]. Leave empty for points-only. |
| `sam_iterations` | `INT` | default `2`, range 1…5, step 1 | Number of SAM refinement iterations.  Each pass uses the previous mask to generate better prompts.  2-3 is ideal. |
| `refine_method` | choice: `auto`, `vitmatte`, `guided_filter`, `multi_scale_guided`, `color_aware`, `laplacian_blend` | default `"auto"` | Edge refinement backend. auto: best available (vitmatte → multi_scale_guided → guided_filter) vitmatte: HuggingFace ViTMatte neural matting guided_filter: fast image-guided alpha multi_scale_guided: guided filter at 3 scales (best non-neural) color_aware: LAB-space color-sensitive edge refinement laplacian_blend: Laplacian pyramid frequency blending |
| `edge_radius` | `INT` | default `12`, range 1…200, step 1 | Pixels around edges to refine (larger = softer transitions) |
| `detail_preservation` | `FLOAT` | default `0.85`, range 0.0…1.0, step 0.05 | How much fine detail (hair, fur, lace) to preserve. 0=smooth, 1=maximum detail. |
| `edge_contrast` | `FLOAT` | default `1.0`, range 0.0…3.0, step 0.1 | Boost edge contrast for challenging lighting. >1 sharpens boundaries. |
| `fill_holes_enabled` | `BOOLEAN` | default `True` | Fill interior holes in the mask |
| `min_region_size` | `INT` | default `64`, range 0…10000, step 1 | Remove isolated mask regions smaller than N pixels (0=disabled) |
| `multimask_output` | `BOOLEAN` | default `True` | Return 3 candidate masks from SAM (vs single best) |
| `mask_index` | `INT` | default `0`, range 0…2 | Which SAM candidate mask to keep when multimask_output is True |
| `score_threshold` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | Discard SAM masks below this confidence score |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `bbox` | `BBOX` |  | Bounding box from BBox node (overrides bbox_json) |
| `existing_mask` | `MASK` |  | Use as initial mask instead of SAM first pass |
| `trimap` | `MASK` |  | Custom trimap for ViTMatte (overrides auto-generated) |
| `trimap_dilate` | `INT` | default `0`, range 0…200, step 1 | Outer trimap radius (0 = use edge_radius * 1.5). |
| `trimap_erode` | `INT` | default `0`, range 0…200, step 1 | Inner trimap erosion radius (0 = use edge_radius * 1.0). |
| `batch_mode` | `BOOLEAN` | default `False` | Process every frame in the input batch (off = first frame only). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `refined_mask` | `MASK` | Final compositing-grade alpha matte after SAM + matting refinement. |
| 1 | `coarse_mask` | `MASK` | SAM coarse mask before edge refinement and cleanup. |
| 2 | `edge_mask` | `MASK` | Edge-band mask highlighting where matting changed the boundary. |
| 3 | `preview` | `IMAGE` | Side-by-side preview of input image and refined mask overlay. |
| 4 | `detected_bbox` | `BBOX` | Bounding box derived from the refined mask. |
| 5 | `score` | `FLOAT` | Best SAM confidence score from iterative refinement. |
| 6 | `info` | `STRING` | JSON summary of stages, parameters, and timings. |


### SeCMatAnyonePipelineMEC

**Shown in the menu as:** SeC + MatAnyone2 Pipeline

SeC + MatAnyone2 end-to-end pipeline:
1. SeC/SAM segmentation → coarse masks
2. MatAnyone2 temporal alpha matting → compositing-grade alpha
3. Optional edge refinement
4. Post-processing (hole fill, region cleanup)

Best for video scenes with occlusions, re-appearances, and complex motion.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Single image or video frames (B>1 for video). |
| `segmentation_model` | choice: `sam2.1_hiera_base_plus`, `sam2.1_hiera_large`, `sam2.1_hiera_small`, `sam2.1_hiera_tiny`, `sam2_hiera_base_plus`, `sam2_hiera_large`, `sam2_hiera_small`, `sam2_hiera_tiny`, … (+2) | default `"sam2.1_hiera_base_plus"` | Segmentation model for coarse masks. SeC: best for video with text prompts. SAM2/3: best for point/bbox prompts. |
| `text_prompt` | `STRING` | default `""` | Text description of target object (e.g. 'cat', 'person in red'). Used by SeC for semantic tracking. Leave empty for point/bbox prompts. |
| `points_json` | `STRING` | default `"[]"`, multiline | Point prompts: [{"x":100,"y":200,"label":1}, ...] |
| `bbox_json` | `STRING` | default `""` | Bounding box: [x1,y1,x2,y2] |
| `matting_backend` | choice: `matanyone2`, `vitmatte_small`, `vitmatte_base`, `auto` | default `"auto"` | Alpha matting backend. auto: MatAnyone2 for video (B>1), ViTMatte for single images. matanyone2: Video matting with temporal consistency. vitmatte_small/base: Neural matting (best edge quality per frame). |
| `edge_radius` | `INT` | default `15`, range 1…200, step 1 | Edge refinement radius in pixels. |
| `n_warmup` | `INT` | default `5`, range 1…30, step 1 | MatAnyone2 warmup frames (more = better temporal init). |
| `precision` | choice: `fp16`, `bf16`, `fp32` | default `"fp16"` | Segmentation model precision. |
| `fill_holes_enabled` | `BOOLEAN` | default `True` | Fill interior holes in the final alpha. |
| `min_region_size` | `INT` | default `64`, range 0…10000, step 1 | Remove isolated regions smaller than N pixels. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `positive_coords` | `STRING` | **connection-only** | Positive points from Points Mask Editor. |
| `negative_coords` | `STRING` | **connection-only** | Negative points from Points Mask Editor. |
| `bbox` | `BBOX` |  | Positive bbox from upstream node. |
| `edge_refine_method` | choice: `none`, `vitmatte`, `guided_filter`, `multi_scale_guided` | default `"none"` | Optional post-matting edge refinement. none: use raw MatAnyone2 output. vitmatte/guided_filter: refine edges after matting. |
| `keep_model_loaded` | `BOOLEAN` | default `True` | Keep models in VRAM between runs. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `rgb` | `IMAGE` | Original RGB frames passed through for downstream nodes. |
| 1 | `alpha_mask` | `MASK` | Compositing-grade alpha mask after MatAnyone2 + optional refinement. |
| 2 | `coarse_mask` | `MASK` | Coarse segmentation mask before alpha matting. |
| 3 | `preview` | `IMAGE` | Side-by-side preview of input frames and final alpha overlay. |
| 4 | `info` | `STRING` | JSON summary of stages, models, and per-stage timings. |


---

## C2C/Preview


### VideoComparerC2C

**Shown in the menu as:** Video Comparer — Player + Wipe/Diff/Scopes

Nuke-grade A/B comparer for image / video / EXR / audio.
• synced_player (default): dual <video> elements share one transport — play/pause, seek, frame-step ◀ 1f / 1f ▶, in/out points, loop, playback rate, live FPS. Frame-perfect via requestVideoFrameCallback.
• Overlay modes: wipe, onion, diff, side_by_side, per_channel, false_color, bit_depth_crush — all render LIVE in the browser, no Queue needed (drag canvas to wipe, ←/→ to scrub).
• Queue-time modes: scopes (waveform/parade/vector/histogram), audio waveform/spectrogram/loudness.
All comparer presets in one node. Switch between them via the mode combo at any time.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mode` | choice: `synced_player`, `wipe`, `onion`, `diff`, `side_by_side`, `per_channel`, `false_color`, `waveform_scope`, … (+7) | default `"synced_player"` | synced_player: dual <video> player with shared transport — play, pause, seek, frame-step (◀ 1f / 1f ▶), in/out points, loop, playback rate. Use this for real-time A/B comparison of two videos. Switch to wipe / onion / diff / side_by_side / false_color / bit_depth_crush for overlay analysis. |
| `bit_depth` | choice: `8`, `10`, `12`, `16`, `32` | default `"32"` | Quantization for bit_depth_crush mode. |
| `wipe_position` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | — |
| `onion_alpha` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | — |
| `diff_gain` | `FLOAT` | default `16.0`, range 1.0…1024.0, step 1.0 | — |
| `diff_gamma` | `FLOAT` | default `1.0`, range 0.1…4.0, step 0.05 | — |
| `diff_threshold` | `FLOAT` | default `0.0`, range 0.0…0.5, step 0.001 | — |
| `diff_mode` | choice: `absolute`, `signed`, `luminance` | default `"absolute"` | — |
| `false_color_lut` | choice: `viridis`, `plasma`, `inferno`, `magma`, `turbo`, `hot`, `coolwarm` | default `"turbo"` | — |
| `scope_intensity` | `FLOAT` | default `0.35`, range 0.05…1.0, step 0.05 | — |
| `frame_index` | `INT` | default `0`, range 0…100000, step 1 | — |
| `label_a` | `STRING` | default `"A"` | — |
| `label_b` | `STRING` | default `"B"` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image_a` | `IMAGE` |  | — |
| `image_b` | `IMAGE` |  | — |
| `audio_a` | `AUDIO` |  | — |
| `audio_b` | `AUDIO` |  | — |
| `file_a` | choice: `` | default `""` | — |
| `file_b` | choice: `` | default `""` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `preview` | `IMAGE` | — |
| 1 | `diff_mask` | `MASK` | — |
| 2 | `scope` | `IMAGE` | — |
| 3 | `info` | `STRING` | — |


### VideoFramePlayerMEC

**Shown in the menu as:** Video Frame Player

Video scrubber + drag-crop + integrated resize. Drag the timeline to scrub frames. Toggle crop_enabled and drag the rectangle on the preview to crop (aspect-locked when a preset is set). resize_method + target_width/target_height + upscale_factor produce the final output. Set output_mode = all_frames to process the whole batch.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `frames` | `IMAGE` |  | Frame batch (B,H,W,C). B=number of frames. |
| `frame_index` | `INT` | default `0`, range 0…99999, step 1 | Current frame to emit on the IMAGE output. Drag the timeline to scrub. |
| `output_mode` | choice: `current_frame`, `all_frames` | default `"current_frame"` | current_frame: emit only the selected frame. all_frames: apply trim+stride+crop+resize to every frame. |
| `frame_start` | `INT` | default `0`, range 0…99999, step 1 | First frame of the trim range. Drag the green marker on the timeline. |
| `frame_end` | `INT` | default `-1`, range -1…99999, step 1 | Last frame of the trim range (inclusive). -1 = last frame. Drag the red marker on the timeline. |
| `frame_stride` | `INT` | default `1`, range 1…64, step 1 | In 'all_frames' mode, output every Nth frame within the trim range. |
| `playback_fps` | `FLOAT` | default `24.0`, range 0.1…240.0, step 0.5 | Preview playback speed (frames per second). |
| `loop_mode` | choice: `once`, `loop`, `ping-pong` | default `"loop"` | Preview playback at end of trim range: once / loop / ping-pong. |
| `crop_enabled` | `BOOLEAN` | default `False` | Enable the drag-crop rectangle on the preview. |
| `crop_locked` | `BOOLEAN` | default `False` | Lock the crop rect to prevent accidental drags. Press R on the canvas to reset. |
| `aspect_ratio` | choice: `free`, `original`, `1:1`, `4:3`, `3:4`, `16:9`, `9:16`, `2:1`, … (+2) | default `"free"` | Aspect lock. 'original' = source W:H. 'custom' = custom_aspect_w:custom_aspect_h. |
| `custom_aspect_w` | `FLOAT` | default `16.0`, range 0.0…999.0, step 0.1 | Custom aspect width (used when aspect_ratio = custom). |
| `custom_aspect_h` | `FLOAT` | default `9.0`, range 0.0…999.0, step 0.1 | Custom aspect height (used when aspect_ratio = custom). |
| `crop_x` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.001 | Crop left edge as fraction of source width [0..1]. Set by dragging the rectangle on the preview. |
| `crop_y` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.001 | Crop top edge as fraction of source height [0..1]. |
| `crop_w` | `FLOAT` | default `1.0`, range 0.001…1.0, step 0.001 | Crop width as fraction of source width (0..1]. |
| `crop_h` | `FLOAT` | default `1.0`, range 0.001…1.0, step 0.001 | Crop height as fraction of source height (0..1]. |
| `resize_method` | choice: `none`, `lanczos`, `bicubic`, `bilinear`, `area`, `nearest-exact` | default `"none"` | Post-crop resize. 'lanczos' = high-quality. |
| `target_width` | `INT` | default `0`, range 0…8192, step 8 | Target width after crop (0 = keep crop width). |
| `target_height` | `INT` | default `0`, range 0…8192, step 8 | Target height after crop (0 = keep crop height). |
| `upscale_factor` | `FLOAT` | default `1.0`, range 0.1…8.0, step 0.05 | Multiplier applied AFTER target_width/target_height (1.0 = no upscale). |
| `preview_width` | `INT` | default `480`, range 96…1920, step 16 | Width (px) of preview frames sent to the browser. Lower = lighter UI; full-resolution IMAGE outputs are unaffected. |
| `preview_format` | choice: `png`, `jpeg` | default `"png"` | png = lossless, exact colors (recommended). jpeg = smaller payload, slight chroma loss. The IMAGE outputs are always the full-precision source tensor regardless of this setting. |
| `preview_quality` | `INT` | default `95`, range 30…100, step 5 | JPEG quality (ignored when preview_format=png). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `frame` | `IMAGE` | Single full-resolution frame at the slider position (1,H,W,C) - pre-crop, pre-resize. Full source precision (no JPEG re-encode). |
| 1 | `frame_index` | `INT` | Echo of the selected frame_index (clamped to trim range). |
| 2 | `frame_count` | `INT` | Total number of frames in the input batch. |
| 3 | `processed` | `IMAGE` | Processed output: the selected frame OR the trimmed/strided batch (per output_mode), with crop + resize + upscale applied. Full source precision. |
| 4 | `out_width` | `INT` | Width (px) of the processed output. |
| 5 | `out_height` | `INT` | Height (px) of the processed output. |
| 6 | `crop_x_px` | `INT` | Crop left edge in source pixels. |
| 7 | `crop_y_px` | `INT` | Crop top edge in source pixels. |
| 8 | `crop_w_px` | `INT` | Crop width in source pixels. |
| 9 | `crop_h_px` | `INT` | Crop height in source pixels. |
| 10 | `trimmed_count` | `INT` | Number of frames emitted on 'processed' after trim+stride (1 in current_frame mode). |
| 11 | `playback_fps` | `FLOAT` | Echo of playback_fps (handy as input to video savers). |
| 12 | `frames_trimmed` | `IMAGE` | Trimmed source batch (frame_start..frame_end inclusive, with stride). NO crop/resize applied. Use this to feed downstream video savers without losing source resolution. |
| 13 | `trim_start_idx` | `INT` | Resolved trim start frame index. |
| 14 | `trim_end_idx` | `INT` | Resolved trim end frame index (inclusive). |
| 15 | `width` | `INT` | Source width in pixels. |
| 16 | `height` | `INT` | Source height in pixels. |
| 17 | `duration` | `FLOAT` | Total duration of the trimmed range in seconds (trimmed_count / playback_fps). |
| 18 | `video_info` | `STRING` | JSON string with all video metadata: width, height, fps, duration, frame_count, trim_*, crop_*, etc. Wire to a Show Text node or any downstream automation. |


---

## C2C/SAM


### SAMMaskGeneratorMEC

**Shown in the menu as:** SAM Mask Generator — Points + BBox + Text

Generate masks using SAM with point + bounding box prompts, iterative refinement, and VRAM offload support.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `sam_model` | `SAM_MODEL` |  | Loaded SAM model from SAM Model Loader |
| `image` | `IMAGE` |  | Input image to segment (first frame is used) |
| `points_json` | `STRING` | default `"[]"`, multiline | JSON array: [{"x":100,"y":200,"label":1}, ...]. label=1=foreground, label=0=background. |
| `bbox_json` | `STRING` | default `""` | Bounding box as JSON: [x1, y1, x2, y2] or {"x":..,"y":..,"w":..,"h":..}. Leave empty to use only point prompts. |
| `text_prompt` | `STRING` | default `""` | Text description of target object (e.g. 'person', 'dog', 'car'). Requires a GroundingDINO model. Converts text to bounding box, then feeds to SAM for precise mask generation. |
| `negative_text_prompt` | `STRING` | default `""` | Text description of objects to EXCLUDE (e.g. 'background', 'wall'). Uses GroundingDINO to detect these regions, then generates negative points from them to suppress unwanted areas in the mask. |
| `grounding_model` | choice: `none`, `groundingdino_swint_ogc`, `groundingdino_swinb_cogcoor` | default `"none"` | GroundingDINO model for text-to-bbox grounding. Set to 'none' to disable text prompting. |
| `text_threshold` | `FLOAT` | default `0.25`, range 0.0…1.0, step 0.01 | GroundingDINO box confidence threshold. |
| `text_box_threshold` | `FLOAT` | default `0.3`, range 0.0…1.0, step 0.01 | GroundingDINO text-box association threshold. |
| `multimask_output` | `BOOLEAN` | default `True` | Return 3 candidate masks (SAM default) vs 1 |
| `mask_index` | `INT` | default `0`, range 0…2 | Which mask to return when multimask=True (0=best score) |
| `score_threshold` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | Discard masks below this confidence score |
| `apply_bbox_crop` | `BOOLEAN` | default `False` | Crop output to bbox region |
| `refine_iterations` | `INT` | default `1`, range 1…5, step 1 | Iterative refinement passes.  Each pass feeds the previous mask back into SAM with augmented prompts.  2-3 significantly improves accuracy. |
| `auto_negative_points` | `BOOLEAN` | default `False` | Automatically sample negative points just outside the mask boundary.  Helps in cluttered scenes and similar-color backgrounds. |
| `edge_refine` | choice: `none`, `guided`, `guided_strong`, `matte` | default `"guided"` | Edge-perfect refinement of the SAM mask. SAM gives a blocky/binary mask; these snap it to the true object boundary → a soft, roto-grade alpha, robust to motion blur and dull/over-bright colours.   none          — raw binary SAM mask   guided        — fast edge-aware guided filter (recommended, no model)   guided_strong — wider/softer for hair + fine detail (no model)   matte         — AI alpha matting (ViTMatte/BiRefNet) for the hardest hair + motion-blur edges; auto-falls back to 'guided' if no matte model installed. |
| `edge_radius` | `INT` | default `8`, range 1…64 | Guided-filter radius (px). Larger = smoother/softer; smaller hugs fine detail. Ignored when edge_refine = none. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `bbox` | `BBOX` |  | Bounding box from BBox node (overrides bbox_json) |
| `bboxes_json` | `STRING` | default `""`, multiline | MULTIPLE boxes to select several regions at once: [[x1,y1,x2,y2], ...]. Each is segmented and unioned into one mask. |
| `spline_json` | `STRING` | default `""`, multiline | Outline the target with a SPLINE and connect/paste its data here (SplineMaskMEC spline_data). Its polygon becomes positive points + a bounding box so SAM snaps to what you drew. |
| `existing_mask` | `MASK` |  | Use this mask as the starting point instead of running SAM from scratch |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | Selected mask for the chosen mask_index. |
| 1 | `all_masks` | `MASK` | All candidate masks returned by SAM (when multimask_output is True). |
| 2 | `detected_bbox` | `BBOX` | Bounding box derived from the selected mask. |
| 3 | `score` | `FLOAT` | SAM confidence score of the selected mask. |
| 4 | `info` | `STRING` | JSON summary of prompts, scores, and refinement steps. |


### SAMModelLoaderMEC

**Shown in the menu as:** SAM Model Loader — SAM2.1 / SAM3

Load SAM 2.1 or SAM 3 model. Auto-detects architecture from filename. Supports VRAM offload.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model_name` | choice: `(place model in models/sams/ or models/sam2/)`, `[download] sam2.1_hiera_base_plus.pt`, `[download] sam2.1_hiera_large.pt`, `[download] sam2.1_hiera_small.pt`, `[download] sam2.1_hiera_tiny.pt` |  | SAM checkpoint (.pth/.pt/.safetensors). Models prefixed with [download] will be auto-downloaded from HuggingFace Hub on first use. |
| `model_type` | choice: `auto`, `sam2.1`, `sam3` | default `"auto"` | Model architecture. 'auto' detects from filename. sam2.1: Segment Anything 2.1 (requires sam2 package) sam3:   SAM3 (uses SAM2 infrastructure) |
| `device` | choice: `auto`, `cuda`, `cpu` | default `"auto"` | Device to load the model on. 'auto' picks CUDA when available. |
| `offload_to_cpu` | `BOOLEAN` | default `False` | Keep model on CPU between inferences. Saves ~2-4 GB VRAM at cost of slower inference. |
| `dtype` | choice: `float16`, `bfloat16`, `float32` | default `"float16"` | Model precision. float16 saves VRAM, bfloat16 for newer GPUs. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `sam_model` | `SAM_MODEL` | Loaded SAM model wrapper for downstream SAM nodes. |


### SamMultiMaskPickerMEC

**Shown in the menu as:** SAM Multi-Mask Picker — 3 candidates + scores

Run SAM inference to get 3 candidate masks with confidence scores. Pick interactively via the JS widget (click thumbnail or press 1/2/3). Press R to re-run. Works with SAM1, SAM2, and HQ-SAM models.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Input image (B, H, W, C) float32 [0, 1]. First frame used for inference. |
| `model_name` | choice: `sam2.1_hiera_base_plus`, `sam2.1_hiera_large`, `sam2.1_hiera_small`, `sam2.1_hiera_tiny`, `sam2_hiera_base_plus`, `sam2_hiera_large`, `sam2_hiera_small`, `sam2_hiera_tiny`, … (+7) | default `"sam2.1_hiera_base_plus"` | SAM model variant to use for inference. Larger models = better quality, more VRAM. |
| `points_json` | `STRING` | default `"[{"x": 256, "y": 256, "label": 1}]"`, multiline | JSON array of point prompts: [{"x":100,"y":200,"label":1}, ...]. label=1=foreground, label=0=background. |
| `bbox_json` | `STRING` | default `""` | Optional bounding box as JSON: [x1, y1, x2, y2]. Leave empty to use only point prompts. |
| `precision` | choice: `fp32`, `fp16`, `bf16` | default `"fp32"` | Model precision. fp16/bf16 use less VRAM but may reduce quality on some GPUs. |
| `selected_index` | `INT` | default `0`, range 0…2, step 1 | Which of the 3 candidate masks to output (0-2). Updated by JS widget on click/keyboard. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `sam_model` | `SAM_MODEL` |  | Pre-loaded SAM model from SAM Model Loader node. Overrides model_name if connected. |
| `bbox` | `BBOX` |  | Bounding box from BBox node (overrides bbox_json). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `selected_mask` | `MASK` | Mask chosen by the user (or selected_index input). |
| 1 | `all_masks` | `MASK` | Stack of all 3 SAM candidate masks. |
| 2 | `selected_index` | `INT` | Index of the currently picked mask. |
| 3 | `scores` | `STRING` | Comma-separated SAM confidence scores for each candidate. |
| 4 | `info` | `STRING` | JSON summary of prompts, scores, and selection. |


---

## C2C/Segmentation


### SemanticSegmentMEC

**Shown in the menu as:** Semantic Segment — Face / Clothes Parsing

Semantic face / clothes parsing using SegFormer.
Select classes by name (comma-separated) to build a combined mask.
Face model: skin, eyes, nose, mouth, hair, hat, glasses, ears.
Clothes model: upper_clothes, pants, dress, shoes, bag, scarf, etc.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Input image(s) to parse. |
| `model_name` | choice: `segformer_clothes`, `segformer_face` | default `"segformer_clothes"` | segformer_face: 19-class facial parts. segformer_clothes: 18-class apparel. |
| `classes_csv` | `STRING` | default `"skin,hair"` | Comma-separated class names to include in mask. Face: skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip, neck, necklace, cloth, hair, hat Clothes: hat, hair, sunglasses, upper_clothes, skirt, pants, dress, belt, left_shoe, right_shoe, face, left_leg, right_leg, left_arm, right_arm, bag, scarf |
| `threshold` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Confidence threshold for class assignment. |
| `invert` | `BOOLEAN` | default `False` | Invert the output mask. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `keep_model_loaded` | `BOOLEAN` | default `True` | Keep model in VRAM between runs. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | Combined binary mask covering all selected semantic classes. |
| 1 | `info` | `STRING` | JSON summary of model, classes used, and per-class pixel counts. |


---

## C2C/Spline


### SplineMaskMEC

**Shown in the menu as:** Spline Mask — Edit/Track/Flow-Path

Unified spline mask node. Same control-point canvas drives three modes: edit (single-frame rasterize), track (LK optical flow across video), flow_path (procedural ribbon/wave/dust). CPU + small VRAM. No models required.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mode` | choice: `edit`, `track`, `flow_path` | default `"edit"` | edit: rasterize a single spline (closed/open) to a mask. track: Lucas-Kanade multi-keyframe tracker across a video. flow_path: procedural pattern along the spline (waves/dust/lightning…). |
| `spline_data` | `STRING` | default `"[]"` | Spline payload from the JS canvas. edit/flow_path: single shape list. track: keyframes list [{frame:int, points:[[x,y],…]}, …]. |
| `spline_type` | choice: `catmull_rom`, `bezier`, `polyline`, `b_spline`, `nurbs`, `natural_cubic`, `cardinal` | default `"catmull_rom"` | [edit/flow_path] interpolation method. catmull_rom/natural_cubic/cardinal pass through the points; b_spline/nurbs are smooth approximating curves; bezier uses tangent handles; polyline = straight. |
| `closed` | `BOOLEAN` | default `True` | [all] closed loop vs open path |
| `samples_per_segment` | `INT` | default `20`, range 2…128, step 1 | [all] curve resolution per segment |
| `feather_radius` | `FLOAT` | default `0.0`, range 0.0…64.0, step 0.5 | [edit/track] gaussian edge feather |
| `invert` | `BOOLEAN` | default `False` | [edit/flow_path] invert mask |
| `smoothing` | `BOOLEAN` | default `True` | [edit] enable spline smoothing |
| `centripetal_alpha` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.05 | [edit] Catmull-Rom alpha (0.5 = centripetal) |
| `width` | `INT` | default `0`, range 0…16384, step 1 | [edit/flow_path] output width (0 = inherit image) |
| `height` | `INT` | default `0`, range 0…16384, step 1 | [edit/flow_path] output height (0 = inherit image) |
| `mask_color` | `STRING` | default `"#ff00ff"` | [edit] preview overlay color (hex) |
| `mask_opacity` | `FLOAT` | default `0.4`, range 0.0…1.0, step 0.05 | [edit] preview overlay opacity |
| `tracking_weight` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.05 | [track] lerp/tracker blend (1=pure LK, 0=pure lerp) |
| `klt_window` | `INT` | default `21`, range 5…51, step 2 | [track] Lucas-Kanade window size |
| `stroke_width` | `INT` | default `3`, range 1…64, step 1 | [track] stroke width for open splines |
| `pattern` | choice: `ribbon`, `wave`, `flow`, `dust`, `river`, `smoke`, `sawtooth`, `square`, … (+5) | default `"ribbon"` | [flow_path] procedural pattern |
| `thickness` | `FLOAT` | default `12.0`, range 0.0…1024.0, step 0.5 | [flow_path] base stroke thickness |
| `amplitude` | `FLOAT` | default `8.0`, range 0.0…1024.0, step 0.5 | [flow_path] modulation amplitude |
| `frequency` | `FLOAT` | default `2.0`, range 0.0…64.0, step 0.1 | [flow_path] modulation frequency |
| `turbulence` | `FLOAT` | default `0.0`, range 0.0…4.0, step 0.05 | [flow_path] noise turbulence strength |
| `turbulence_scale` | `FLOAT` | default `1.0`, range 0.01…32.0, step 0.05 | [flow_path] noise spatial scale |
| `edge_softness` | `FLOAT` | default `1.0`, range 0.0…32.0, step 0.1 | [flow_path] edge softness in pixels |
| `taper_start` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | [flow_path] start-end taper amount |
| `taper_end` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | [flow_path] tail-end taper amount |
| `frames` | `INT` | default `1`, range 1…4096, step 1 | [flow_path] number of animation frames |
| `animation_speed` | `FLOAT` | default `0.05`, range 0.0…4.0, step 0.005 | [flow_path] phase advance per frame |
| `flow_direction` | choice: `forward`, `reverse`, `bidirectional`, `oscillate` | default `"forward"` | [flow_path] flow direction |
| `mod_decay` | `FLOAT` | default `0.0`, range 0.0…4.0, step 0.01 | [flow_path] modulation falloff over time |
| `seed` | `INT` | default `0`, range 0…4294967295, step 1 | [flow_path] noise seed |
| `use_embedded_editor` | `BOOLEAN` | default `True` | [flow_path] show embedded spline preview |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | [edit/track/flow_path] reference / source video frames |

**Hidden inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `node_id` | `UNIQUE_ID` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | Rasterized mask (B,H,W). |
| 1 | `coords_json` | `STRING` | SAM-compatible point coords (edit mode); '[]' otherwise. |
| 2 | `spline_data_out` | `SPLINE_DATA` | Structured spline data (edit mode); pass-through input otherwise. |
| 3 | `info_json` | `STRING` | Mode diagnostics: edit→bbox_json, track→info, flow_path→params. |
| 4 | `bbox` | `BBOX` | AABB of control points as [x,y,w,h]; [0,0,0,0] when N/A. |


---

## C2C/Stabilization


### VideoStabilizerAutoMEC

**Shown in the menu as:** Video Stabilizer — Auto (deprecated)

DEPRECATED — use VideoStabilizerMEC (method=auto) instead.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `frames` | `IMAGE` |  | — |
| `frame_rate` | `FLOAT` | default `16.0`, range 1.0…–, step 0.1 | — |
| `force_backend` | choice: `auto`, `classic`, `flow` | default `"auto"` | — |
| `preset` | choice: `handheld_light`, `handheld_heavy`, `vehicle`, `tripod_lock` | default `"handheld_light"` | — |
| `padding_color` | `STRING` | default `"127, 127, 127"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `stabilized_frames` | `IMAGE` | — |
| 1 | `padding_mask` | `MASK` | — |
| 2 | `info` | `STRING` | — |


### VideoStabilizerClassicMEC

**Shown in the menu as:** Video Stabilizer — Classic (deprecated)

DEPRECATED — use VideoStabilizerMEC (method=classic) instead.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `frames` | `IMAGE` |  | — |
| `frame_rate` | `FLOAT` | default `16.0`, range 1.0…–, step 0.1 | — |
| `framing_mode` | choice: `crop`, `crop_and_pad`, `expand` | default `"crop_and_pad"` | — |
| `transform_mode` | choice: `translation`, `similarity`, `perspective` | default `"similarity"` | — |
| `camera_lock` | `BOOLEAN` | default `False` | — |
| `strength` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.05 | — |
| `smooth` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.05 | — |
| `keep_fov` | `FLOAT` | default `0.6`, range 0.0…1.0, step 0.05 | — |
| `padding_color` | `STRING` | default `"127, 127, 127"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `stabilized_frames` | `IMAGE` | — |
| 1 | `padding_mask` | `MASK` | — |
| 2 | `info` | `STRING` | — |


### VideoStabilizerFlowMEC

**Shown in the menu as:** Video Stabilizer — RAFT Flow (deprecated)

DEPRECATED — use VideoStabilizerMEC (method=raft_flow) instead.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `frames` | `IMAGE` |  | — |
| `frame_rate` | `FLOAT` | default `16.0`, range 1.0…–, step 0.1 | — |
| `framing_mode` | choice: `crop`, `crop_and_pad`, `expand` | default `"crop_and_pad"` | — |
| `transform_mode` | choice: `translation`, `similarity`, `perspective` | default `"similarity"` | — |
| `camera_lock` | `BOOLEAN` | default `False` | — |
| `strength` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.05 | — |
| `smooth` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.05 | — |
| `keep_fov` | `FLOAT` | default `0.6`, range 0.0…1.0, step 0.05 | — |
| `padding_color` | `STRING` | default `"127, 127, 127"` | — |
| `raft_iters` | `INT` | default `12`, range 4…32 | — |
| `use_half` | `BOOLEAN` | default `True` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `stabilized_frames` | `IMAGE` | — |
| 1 | `padding_mask` | `MASK` | — |
| 2 | `info` | `STRING` | — |


### VideoStabilizerMEC

**Shown in the menu as:** Video Stabilizer

Unified video stabilizer (auto / classic / raft_flow). Wraps the vendored MIT ComfyUI-Video-Stabilizer. Outputs frames + padding mask so it can chain straight into InpaintCropProMEC to fill borders.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `frames` | `IMAGE` |  | — |
| `method` | choice: `auto`, `classic`, `raft_flow` | default `"auto"` | — |
| `preset` | choice: `handheld_light`, `handheld_heavy`, `vehicle`, `tripod_lock`, `manual` | default `"handheld_light"` | Preset overrides the manual widgets below unless set to 'manual'. |
| `frame_rate` | `FLOAT` | default `16.0`, range 1.0…–, step 0.1 | — |
| `padding_color` | `STRING` | default `"127, 127, 127"` | — |
| `framing_mode` | choice: `crop`, `crop_and_pad`, `expand` | default `"crop_and_pad"` | — |
| `transform_mode` | choice: `translation`, `similarity`, `perspective` | default `"similarity"` | — |
| `camera_lock` | `BOOLEAN` | default `False` | — |
| `strength` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.05 | — |
| `smooth` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.05 | — |
| `keep_fov` | `FLOAT` | default `0.6`, range 0.0…1.0, step 0.05 | — |
| `raft_iters` | `INT` | default `12`, range 4…32 | RAFT iterations. Used only by method=raft_flow (and auto when it picks flow). |
| `use_half` | `BOOLEAN` | default `True` | fp16 for RAFT. Used only by method=raft_flow. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `stabilized_frames` | `IMAGE` | — |
| 1 | `padding_mask` | `MASK` | — |
| 2 | `info` | `STRING` | — |


---

## C2C/Utils


### ParameterHistoryMEC

**Shown in the menu as:** Parameter History

Query the parameter history database. Shows what parameters were changed, when, and what the previous values were. Supports run-to-run diffs.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mode` | choice: `all_history`, `last_run_diff`, `node_class_filter` | default `"all_history"` | all_history: show recent parameter changes across all nodes last_run_diff: show what changed between the last two runs node_class_filter: filter history to a specific node class |
| `last_n_runs` | `INT` | default `5`, range 1…100, step 1 | How many recent runs to include |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `node_class_filter` | `STRING` | default `""` | Node class name to filter by (e.g. 'KSampler') |
| `run_a` | `INT` | default `0`, range 0…– | First run number for diff mode (0 = auto-detect last two) |
| `run_b` | `INT` | default `0`, range 0…– | Second run number for diff mode (0 = auto-detect last two) |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `history_report` | `STRING` | Human-readable parameter history report for the chosen mode. |


---

## C2C/VAE


### VAEMergeMEC

**Shown in the menu as:** VAE Merge

Merge 2 or 3 VAEs with 13 strategies (weighted_sum, sigmoid, geometric, slerp, dare_ties, …). Optional per-block weights, auto-alpha from block cosine similarity, latent-space probe to report reconstruction MSE/PSNR, dry-run, and recipe export.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vae_a` | `VAE` |  | Primary VAE (acts as base; deep-copied clone is returned). |
| `vae_b` | `VAE` |  | Secondary VAE blended into vae_a. |
| `merge_mode` | choice: `weighted_sum`, `add_difference`, `tensor_sum`, `triple_sum`, `slerp`, `sigmoid`, `geometric`, `max_abs`, … (+5) | default `"weighted_sum"` | Blend strategy. add_difference / triple_sum / smooth_add_diff need vae_c. sigmoid + geometric mimic TechnoByte/meh behaviour. distribution_xover keeps A unless B has higher detail energy. dare_ties = sparse delta + sign election. |
| `alpha` | `FLOAT` | default `0.3`, range 0.0…1.0, step 0.01 | Primary blend weight. weighted_sum: 0=A, 1=B. |
| `beta` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.01 | Secondary blend weight (used by add_difference / 3-VAE modes). |
| `brightness` | `FLOAT` | default `0.0`, range -1.0…1.0, step 0.01 | Post-merge brightness shift on decoder.conv_out. |
| `contrast` | `FLOAT` | default `0.0`, range -1.0…1.0, step 0.01 | Post-merge contrast gain on decoder.conv_out. |
| `use_blocks` | `BOOLEAN` | default `False` | Enable per-block sliders. When False all keys use 'alpha'. |
| `auto_alpha` | `BOOLEAN` | default `False` | Data-driven block weights. When True, computes per-block cosine similarity between A and B; dissimilar blocks bias toward A. Overrides the manual block sliders. |
| `block_conv_in` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Weight for encoder/decoder conv_in. |
| `block_conv_out` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Weight for encoder/decoder conv_out. |
| `block_norm_out` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Weight for encoder/decoder norm_out. |
| `block_0` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | First down/up block pair. |
| `block_1` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Second down/up block pair. |
| `block_2` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Third down/up block pair. |
| `block_3` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Fourth down/up block pair (SDXL). |
| `block_mid` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Mid block. |
| `device` | choice: `cpu`, `cuda`, `auto` | default `"cpu"` | Compute device. CPU is safe (default); CUDA is faster but uses VRAM. |
| `dry_run` | `BOOLEAN` | default `False` | If True, skip the merge and return only the recipe + similarity report. Use this for fast block-similarity inspection without waiting for the merge. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vae_c` | `VAE` |  | Optional third VAE for add_difference / triple_sum / smooth_add_diff. |
| `reference_image` | `IMAGE` |  | Optional reference image. If connected, encodes/decodes it through A, B, and the merged VAE; reports MSE/PSNR per VAE in probe_report. |
| `recipe_in` | `STRING` | default `""`, multiline | Optional recipe JSON from a previous run. When provided, overrides every widget value above so the exact merge is reproduced. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `vae` | `VAE` | Merged VAE (or vae_a clone when dry_run=True). |
| 1 | `info` | `STRING` | Human-readable info string summarising mode, alpha, blocks, and timing. |
| 2 | `recipe_json` | `STRING` | Reproducible recipe JSON; feed back into recipe_in to repeat the exact merge. |
| 3 | `probe_report` | `STRING` | Probe report with MSE/PSNR per VAE when reference_image is connected. |


---

## C2C/Video


### MaskTrackerMEC

**Shown in the menu as:** Mask Tracker — Motion/Propagate/Anchor/Consistency

Unified video-mask tracker. Pick a mode and the corresponding engine runs. All modes share the (mask, video) input pair. Heavy work is chunked/vectorized; CPU fallback always available.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mode` | choice: `motion`, `propagate`, `anchor`, `consistency_check` | default `"motion"` | motion: per-frame motion mask (pixel/flow/bg/hist). propagate: seed mask on one frame, push to all frames. anchor: SDF interpolation between anchor masks. consistency_check: score flicker between consecutive frames. |
| `camera_compensation` | `BOOLEAN` | default `True` | [motion] subtract global camera motion |
| `stabilization_method` | choice: `homography`, `affine`, `translation` | default `"homography"` | [motion] camera-motion model |
| `detection_mode` | choice: `combined`, `pixel_diff`, `optical_flow`, `background_sub`, `histogram_diff` | default `"combined"` | [motion] active method(s) |
| `pixel_diff_enabled` | `BOOLEAN` | default `True` | [motion] enable pixel-diff method |
| `pixel_diff_threshold` | `FLOAT` | default `0.05`, range 0.001…1.0, step 0.001 | [motion] pixel-diff threshold |
| `flow_enabled` | `BOOLEAN` | default `True` | [motion] enable optical flow |
| `flow_threshold` | `FLOAT` | default `1.0`, range 0.1…50.0, step 0.1 | [motion] flow magnitude threshold |
| `flow_algorithm` | choice: `farneback`, `phase_correlation` | default `"farneback"` | [motion] flow algorithm |
| `bg_sub_enabled` | `BOOLEAN` | default `False` | [motion] background subtraction |
| `bg_model_frames` | `INT` | default `5`, range 1…30, step 1 | [motion] frames for bg model |
| `bg_sub_threshold` | `FLOAT` | default `0.1`, range 0.001…1.0, step 0.001 | [motion] bg-diff threshold |
| `hist_enabled` | `BOOLEAN` | default `False` | [motion] histogram diff |
| `hist_grid_size` | `INT` | default `16`, range 4…64, step 4 | [motion] histogram grid NxN |
| `hist_threshold` | `FLOAT` | default `0.15`, range 0.01…1.0, step 0.01 | [motion] histogram L2 threshold |
| `combine_method` | choice: `union`, `intersection` | default `"union"` | [motion] method combination |
| `grow_pixels` | `FLOAT` | default `4.0`, range 0.0…64.0, step 1.0 | [motion] dilate result |
| `min_region_size` | `INT` | default `100`, range 0…10000, step 10 | [motion] noise filter |
| `temporal_smooth` | `BOOLEAN` | default `True` | [motion] gaussian time smoothing |
| `source_frame` | `INT` | default `0`, range 0…99999 | [propagate] frame where mask is drawn |
| `propagate_mode` | choice: `static`, `optical_flow`, `sam2_video`, `fade`, `scale_linear` | default `"static"` | [propagate] propagation method |
| `prop_flow_threshold` | `FLOAT` | default `2.0`, range 0.0…50.0, step 0.5 | [propagate] optical-flow threshold |
| `fade_start` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | [propagate] opacity at source frame |
| `fade_end` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | [propagate] opacity at last frame |
| `bidirectional` | `BOOLEAN` | default `True` | [propagate] forward+backward from source |
| `anchor_frames` | `STRING` | default `"0"` | [anchor] CSV frame indices for each anchor mask |
| `total_frames` | `INT` | default `30`, range 1…99999 | [anchor] total output frames |
| `easing` | choice: `linear`, `ease_in`, `ease_out`, `smooth_step` | default `"smooth_step"` | [anchor] easing curve |
| `sdf_iterations` | `INT` | default `64`, range 4…512, step 4 | [anchor] SDF diffusion iterations |
| `flow_refinement` | `BOOLEAN` | default `False` | [anchor] optical-flow refine (needs video) |
| `metric` | choice: `mask_iou`, `pixel_diff`, `flow_warp` | default `"pixel_diff"` | [consistency_check] metric |
| `binarize_threshold` | `FLOAT` | default `0.5`, range 0.01…0.99, step 0.01 | [consistency_check] mask binarize threshold |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | Required by propagate (seed), anchor (anchor stack), consistency_check (mask_iou). Optional for motion. |
| `video` | `IMAGE` |  | Video frame batch (B,H,W,C). Required by motion, propagate, anchor flow_refinement, and pixel/flow consistency. |
| `sam_model` | `SAM_MODEL` |  | [propagate sam2_video mode] SAM2 model |
| `points_json` | `STRING` | default `""`, multiline | [propagate sam2_video mode] point prompts |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `masks` | `MASK` | Per-frame mask batch (B,H,W). |
| 1 | `preview` | `IMAGE` | Preview overlay (propagate) or video passthrough. |
| 2 | `score` | `FLOAT` | Mode-specific scalar: motion intensity / mean confidence / flicker score. |
| 3 | `info_json` | `STRING` | Mode-specific JSON diagnostic payload. |
| 4 | `metric` | `STRING` | Mode/metric label string. |


---

## C2C/VideoMask


### VideoMaskEditorMEC

**Shown in the menu as:** Video Mask Editor

Open the in-browser video mask editor (brush / erase / fill / lasso / onion-skin) to pin per-frame keyframes, then tweens non-keyframed frames using distance-transform interpolation.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Input video batch (B,H,W,3). Drives B/H/W. |
| `session_id` | `STRING` | default `""` | Auto-set by the editor UI. Don't edit by hand. |
| `tween_mode` | choice: `distance_transform`, `linear`, `hold` | default `"distance_transform"` | How to interpolate non-keyframed frames. |
| `feather` | `FLOAT` | default `0.0`, range 0.0…32.0, step 0.1 | Gaussian feather radius (px). 0 = none. |
| `threshold` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | Binarize after tween. 0 = keep soft mask. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `input_mask` | `MASK` |  | Fallback mask used if no keyframes are set. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | — |
| 1 | `info` | `STRING` | — |


---

## C2C/Wan_Director


### WanDirectorC2C

**Shown in the menu as:** Wan Director

Visual timeline director for Wan 2.1 / 2.2 / Fun / Animate. Drag image, text and audio clips onto the timeline, choose a model_variant, and the node emits the matching CONDITIONING, LATENT, FPS and AUDIO bundle. Inspired by WhatDreamsCost / LTX Director (MIT), redesigned for the Wan VAE shape (16ch, /8 spatial, /4 temporal) and Wan-specific options (dual-CFG for 2.2, reference image for Animate, control track for Fun).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `backend` | choice: `native`, `kijai` | default `"native"` | Which video-model stack to drive.    native — ComfyUI's built-in Wan implementation. Connect `model` + `clip`.   kijai  — Kijai's ComfyUI-WanVideoWrapper. Connect `wan_model` + `wan_t5` (optional sockets).  PromptRelay (if enabled) is applied to whichever backbone is active and falls back to the generic-introspection patcher for any third-party model. |
| `model_variant` | choice: `wan2.1_t2v`, `wan2.1_i2v`, `wan2.2_t2v`, `wan2.2_i2v`, `wan_fun_inp`, `wan_fun_control`, `wan_animate`, `wan2.2_animate_everanimate` | default `"wan2.1_i2v"` | Which Wan family / mode this timeline targets. Changes which optional sliders are visible and how the latent + conditioning are assembled. |
| `duration_frames` | `INT` | default `81`, range 1…10000, step 1 | Total timeline length in pixel-space frames. Wan 2.x defaults to 81 frames (≈ 5 s @ 16 fps). |
| `duration_seconds` | `FLOAT` | default `5.0`, range 0.1…1000.0, step 0.01 | Total timeline duration in seconds (synced from frames by the UI). |
| `frame_rate` | `FLOAT` | default `16.0`, range 1.0…240.0, step 1.0 | FPS. Wan 2.x is trained at 16 fps; raise for slow-motion-like output. |
| `global_prompt` | `STRING` | default `""`, multiline | Persistent context prepended to every per-clip prompt (characters, lighting, style anchors). |
| `timeline_data` | `STRING` | default `""`, multiline | — |
| `local_prompts` | `STRING` | default `""`, multiline | — |
| `negative_prompts` | `STRING` | default `""`, multiline | — |
| `segment_lengths` | `STRING` | default `""` | — |
| `guide_strength` | `STRING` | default `""` | — |
| `display_mode` | choice: `seconds`, `frames` | default `"seconds"` | — |
| `custom_width` | `INT` | default `832`, range 0…8192, step 8 | Target width. 0 = inherit from first image clip. |
| `custom_height` | `INT` | default `480`, range 0…8192, step 8 | Target height. 0 = inherit from first image clip. |
| `resize_method` | choice: `maintain aspect ratio`, `stretch to fit`, `pad`, `crop` | default `"maintain aspect ratio"` | — |
| `cfg_high_noise` | `FLOAT` | default `3.5`, range 0.0…20.0, step 0.1 | Wan 2.2 high-noise expert CFG. Ignored for non-2.2 variants. |
| `cfg_low_noise` | `FLOAT` | default `3.5`, range 0.0…20.0, step 0.1 | Wan 2.2 low-noise expert CFG. Ignored for non-2.2 variants. |
| `ref_strength` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.05 | Wan Animate reference-image influence. Ignored for other variants. |
| `everanimate_stage` | choice: `stage1_480p`, `stage2_480p`, `stage3_720p_beta` | default `"stage2_480p"` | Which EverAnimate LoRA checkpoint to apply on top of Wan2.2-Animate-14B:   stage1_480p     — base motion fidelity (480p training).   stage2_480p     — Restorative Flow Matching, sharper temporal coherence (recommended).   stage3_720p_beta — 720p beta with higher detail; needs more VRAM. Ignored for non-EverAnimate variants. |
| `everanimate_num_chunks` | `INT` | default `1`, range 1…50, step 1 | Long-horizon chunk count. 1 = single ~5 s clip (standard Wan2.2-Animate). ≥2 enables EverAnimate's Persistent Latent Propagation across anchor frames for minute-scale animation. Ignored for non-EverAnimate variants. |
| `everanimate_overlap_frames` | `INT` | default `4`, range 0…16, step 1 | Frames of latent overlap between consecutive chunks (anchor padding). Higher = smoother seams but slower. Ignored if num_chunks=1 or non-EverAnimate variant. |
| `everanimate_lora_strength` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.05 | EverAnimate rank-32 LoRA strength. 1.0 = paper default. Ignored for non-EverAnimate variants. |
| `everanimate_anchor_strategy` | choice: `auto`, `first_only`, `first_plus_random_3` | default `"auto"` | Anchor-frame selection for chunks 2+:   auto                 — first chunk uses first frame only, later chunks use first + 3 random.   first_only           — always 1 anchor (faster, slight quality loss).   first_plus_random_3  — always 4 anchors (paper-default; best quality). Ignored for non-EverAnimate variants. |
| `audio_target` | choice: `music_44k_stereo`, `speech_16k_mono` | default `"music_44k_stereo"` | Output AUDIO format. Use `speech_16k_mono` if you intend to feed Wan-S2V or any speech-driven pipeline downstream. |
| `enable_prompt_relay` | `BOOLEAN` | default `True` | Internal PromptRelay: bias each backbone cross-attention block so the timeline's per-clip prompts only steer their own frame span. On by default — the whole point of the timeline. Works on native ComfyUI MODEL, Kijai WANVIDEOMODEL, and arbitrary video-diffusion models (auto-falls back to generic introspection). With 2+ text/image clips the per-clip prompts become the local prompts and `global_prompt` is the anchor; with 0 or 1 clip it is a no-op. Turn off to encode one flat prompt for the whole clip. |
| `prompt_relay_epsilon` | `FLOAT` | default `0.001`, range 1e-06…0.99, step 0.0001 | PromptRelay penalty decay. <0.1 = sharp boundaries; ≥0.5 softer. |
| `enable_dynamic_cfg` | `BOOLEAN` | default `False` | Cosine-ramped dynamic CFG across denoising steps. Early steps get 1.2× CFG (stronger structure), late steps get 0.7× (softer detail). Prevents oversaturation and improves quality. |
| `guidance_rescale_phi` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.05 | Guidance rescale (phi). Rescales guided output to match conditional std-deviation, preventing color oversaturation at high CFG. 0=off, 0.7=recommended for Wan 2.2. Requires enable_dynamic_cfg=True. |
| `pag_scale` | `FLOAT` | default `0.0`, range 0.0…5.0, step 0.1 | Perturbed Attention Guidance scale. Improves prompt adherence by guiding away from identity-attention outputs. 0=off, 1.0–3.0 typical. |
| `enable_phase_shift` | `BOOLEAN` | default `False` | Phase-shift sampling: Euler for early steps (structure), DPM++ 2M for late steps (detail). Uses smooth sigma crossfade. |
| `phase_shift_pct` | `FLOAT` | default `0.7`, range 0.3…0.95, step 0.05 | Step fraction where phase-shift transitions from Euler to DPM++. |
| `vae_fp32_decode` | `BOOLEAN` | default `True` | Force VAE decode in fp32 for maximum quality. Wan VAE produces significantly better results in fp32 (recommended by HuggingFace). Uses more VRAM during decode only. |
| `enable_multi_clip` | `BOOLEAN` | default `False` | Multi-slot CLIP conditioning. Split prompts into structure (early) and detail (late) phases for finer control over generation. |
| `structure_prompt` | `STRING` | default `""`, multiline | Structure prompt (active during early denoising, 0–35%). Focus on composition, layout, camera angles, scene description. Only used when enable_multi_clip=True. |
| `detail_prompt` | `STRING` | default `""`, multiline | Detail prompt (active during late denoising, 55–100%). Focus on textures, materials, lighting, color grading. Only used when enable_multi_clip=True. |
| `enable_nag` | `BOOLEAN` | default `False` | Normalized Attention Guidance: boosts prompt adherence via attention-space CFG. |
| `nag_scale` | `FLOAT` | default `11.0`, range 0.0…30.0, step 0.5 | NAG guidance scale. Higher = stronger guidance. |
| `enable_asymflow` | `BOOLEAN` | default `False` | AsymFlow time-shift for improved temporal consistency. |
| `asymflow_shift` | `FLOAT` | default `3.0`, range 0.1…20.0, step 0.1 | AsymFlow shift parameter. |
| `cache_type` | choice: `none`, `teacache`, `magcache`, `easycache` | default `"none"` | Inference caching strategy. Speeds up generation by skipping redundant transformer passes. |
| `cache_threshold` | `FLOAT` | default `0.1`, range 0.0…1.0, step 0.01 | Cache skip threshold. Lower = more aggressive caching (faster but less accurate). |
| `enable_slg` | `BOOLEAN` | default `False` | Skip-Layer Guidance: run a second pass with layers removed for quality boost. |
| `slg_layers` | `STRING` | default `""` | Comma-separated layer indices to skip (e.g. '7,8,9'). Empty = auto-select. |
| `slg_scale` | `FLOAT` | default `0.7`, range 0.0…2.0, step 0.05 | SLG guidance scale. |
| `enable_feta` | `BOOLEAN` | default `False` | Frequency-Enhanced Temporal Attention for better frame coherence. |
| `feta_scale` | `FLOAT` | default `0.5`, range 0.0…2.0, step 0.05 | FETA scale. 0 = off. |
| `enable_riflex` | `BOOLEAN` | default `False` | RIFLEx RoPE rescaling for length extrapolation beyond training length. |
| `riflex_k` | `INT` | default `2`, range 1…8, step 1 | Number of lowest RoPE frequencies to rescale. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | Native Wan MODEL (2.1 / 2.2 / Fun / Animate). Required when backend='native'. |
| `clip` | `CLIP` |  | Text encoder paired with the Wan model (UMT5 for 2.x). Required when backend='native'. |
| `vae` | `VAE` |  | Wan VAE. If connected, the Director encodes reference images for i2v and forces fp32 when vae_fp32_decode=True. |
| `clip_vision` | `CLIP_VISION` |  | CLIP Vision model for image embeddings (Kijai i2v/Animate). If not connected, image_embeds output is None. |
| `optional_latent` | `LATENT` |  | Override the auto-built empty latent. |
| `control_video` | `IMAGE` |  | For Wan Fun / Animate: control sequence (depth/pose/canny). Passed through to control_video output. |
| `control_mask` | `MASK` |  | For Wan Fun Inpaint: per-frame mask track. Passed through to control_mask output. |
| `wan_model` | `WANVIDEOMODEL` |  | Kijai WanVideoWrapper model patcher (required when backend='kijai'). |
| `wan_t5` | `WANTEXTENCODER` |  | Kijai T5 text encoder (required when backend='kijai'). |
| `t5gemma` | `T5GEMMA_ENCODER` |  | R&D: connect a 'T5Gemma Encoder Loader (WNE)' to encode the Director's composed prompt through T5Gemma instead of CLIP (backend='native'). NOTE: T5Gemma hidden states are a different space/width from Wan's UMT5-XXL — this only works on a Wan model finetuned/adapted for T5Gemma; a stock Wan checkpoint will error or produce garbage. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | Native MODEL (patched with PromptRelay + NAG + PAG + Dynamic CFG + AsymFlow as enabled). When backend='kijai', passthrough of input `model` if connected. |
| 1 | `positive` | `CONDITIONING` | Positive CONDITIONING (native branch) with guide_strength embedded. Empty list when backend='kijai'. |
| 2 | `negative` | `CONDITIONING` | Negative CONDITIONING (native branch). Empty list when backend='kijai'. |
| 3 | `video_latent` | `LATENT` | Wan latent: VAE-encoded reference for i2v (if VAE connected) or empty latent. Channels=16, /8 spatial, /4 temporal. |
| 4 | `frame_rate` | `FLOAT` | Frame rate echoed for downstream sampler/saver nodes. |
| 5 | `combined_audio` | `AUDIO` | Audio waveform mixed from the timeline's audio segments. |
| 6 | `reference_image` | `IMAGE` | Reference image (first image clip) — used by Wan I2V/Animate as the start/reference frame. Black image if none. |
| 7 | `info` | `STRING` | JSON: resolved backend, variant, latent shape, segment count, audio sample rate, prompt-relay status, quality stack status, warnings. |
| 8 | `wan_model` | `WANVIDEOMODEL` | Kijai WANVIDEOMODEL (only populated when backend='kijai'; PromptRelay-patched in place if enabled). |
| 9 | `wan_text_embeds` | `WANVIDEOTEXTEMBEDS` | Kijai WANVIDEOTEXTEMBEDS dict (only populated when backend='kijai'). Feed directly into WanVideoSampler. |
| 10 | `tracks_program` | `STRING` | JSON: timeline schema_version + normalised lora/camera/seed/pose tracks for downstream applier nodes. |
| 11 | `control_video` | `IMAGE` | Control video passthrough (IMAGE) for Wan Fun/Animate. None if not connected. |
| 12 | `control_mask` | `MASK` | Control mask passthrough (MASK) for Wan Fun Inpaint. None if not connected. |
| 13 | `guide_data` | `STRING` | JSON: per-segment guide_strength values for downstream guide applier nodes. |
| 14 | `quality_recipe` | `STRING` | JSON: quality recipe config (SLG, FETA, RIFLEx, cache, FreeInit, phase-shift) for downstream sampler. |


---

## Code2Collapse/Sampling


### AsymFlowSamplerPatch

**Shown in the menu as:** AsymFlow Sampler Patch (Lakonik signal-shift)

Replace the model's flow-matching schedule with AsymFlow's shifted-signal-ratio mapping (sigma = r/(shift+r), r=sqrt(t/(1-t))). Inference-only; no retraining required. Best on flow models (SD3, Flux, AsymFlow-trained checkpoints).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `shift` | `FLOAT` | default `3.0`, range 0.05…20.0, step 0.05 | AsymFlow signal-shift. shift=1 -> linear flow. >1 spends more steps at high noise (recommended for high-resolution / Flux-like models). |
| `multiplier` | `INT` | default `1000`, range 1…10000, step 1 | Discretization multiplier (timesteps scale). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |


---

## ComfyUI-CustomNodePacks/PromptRelay


### PromptRelayAdvancedOptionsC2C

**Shown in the menu as:** Prompt Relay Advanced Options

Optional per-stream tuning for the Prompt Relay encoders.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `video_strength` | `FLOAT` | default `1.0`, range 0.0…10.0, step 0.05 | Multiplier on the temporal penalty. 0 disables segmentation. Most useful in 0–1 to soften boundaries; >1 saturates quickly at the default epsilon — raise epsilon to ~0.1 to make >1 meaningful. |
| `video_window_scale` | `FLOAT` | default `1.0`, range 0.0…4.0, step 0.05 | Scales the flat anchor zone (default L/2 - 2 frames). <1 starts falloff sooner; >1 widens the rigid zone. |
| `audio_epsilon` | `FLOAT` | default `0.0`, range 0.0…0.99, step 0.0001 | LTX audio stream epsilon. 0 = inherit from the encoder. |
| `audio_strength` | `FLOAT` | default `1.0`, range 0.0…10.0, step 0.05 | Multiplier on the temporal penalty. 0 disables segmentation. Most useful in 0–1 to soften boundaries; >1 saturates quickly at the default epsilon — raise epsilon to ~0.1 to make >1 meaningful. |
| `audio_window_scale` | `FLOAT` | default `1.0`, range 0.0…4.0, step 0.05 | Scales the flat anchor zone (default L/2 - 2 frames). <1 starts falloff sooner; >1 widens the rigid zone. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `relay_options` | `RELAY_OPTIONS` | — |


### PromptRelayEncodeC2C

**Shown in the menu as:** Prompt Relay Encode

Unified Prompt Relay encoder (native / smart / kijai). Dynamic sockets via the JS extension.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `backend` | choice: `native`, `smart`, `kijai` | default `"native"` | native = MODEL+CLIP; smart = MODEL+CLIP with auto-segmented prompt; kijai = WanVideoWrapper. |
| `global_prompt` | `STRING` | default `""`, multiline | Persistent prompt across the whole video. |
| `epsilon` | `FLOAT` | default `0.001`, range 1e-06…0.99, step 0.0001 | Temporal penalty decay (sharpness of segment boundaries). |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | native/smart only. |
| `clip` | `CLIP` |  | native/smart only. |
| `latent` | `LATENT` |  | native/smart only — frame count read from shape. |
| `wan_model` | `WANVIDEOMODEL` |  | kijai only — from WanVideoModelLoader. |
| `wan_t5` | `WANTEXTENCODER` |  | kijai only — from LoadWanVideoT5TextEncoder. |
| `local_prompts` | `STRING` | default `""`, multiline | Per-segment prompts separated by '\|' (native/kijai). |
| `segment_lengths` | `STRING` | default `""` | Comma-separated pixel-space frame counts. Empty = equal (native/kijai). |
| `smart_prompt` | `STRING` | default `""`, multiline | smart only — auto-parsed (`\|` or `Scene N:`). |
| `normalize_by_tokens` | `BOOLEAN` | default `False` | smart only — scale segment weight by token count. |
| `latent_frames` | `INT` | default `81`, range 1…10000, step 1 | kijai only — (pixel_frames-1)//4 + 1. |
| `negative_prompt` | `STRING` | default `""`, multiline | kijai only — encoded once for negative_prompt_embeds. |
| `encode_device` | choice: `gpu`, `cpu` | default `"gpu"` | kijai only — device for T5 encode. |
| `relay_options` | `RELAY_OPTIONS` |  | Optional Prompt Relay Advanced Options bundle. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |
| 1 | `positive` | `CONDITIONING` | — |
| 2 | `wan_model` | `WANVIDEOMODEL` | — |
| 3 | `wan_text_embeds` | `WANVIDEOTEXTEMBEDS` | — |


### PromptRelayEncodeKijaiC2C

**Shown in the menu as:** Prompt Relay Encode (Kijai) — deprecated

DEPRECATED — use PromptRelayEncodeC2C with backend='kijai'.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `WANVIDEOMODEL` |  | — |
| `t5` | `WANTEXTENCODER` |  | — |
| `latent_frames` | `INT` | default `81`, range 1…10000, step 1 | — |
| `global_prompt` | `STRING` | default `""`, multiline | — |
| `local_prompts` | `STRING` | default `""`, multiline | — |
| `segment_lengths` | `STRING` | default `""` | — |
| `negative_prompt` | `STRING` | default `""`, multiline | — |
| `epsilon` | `FLOAT` | default `0.001`, range 1e-06…0.99, step 0.0001 | — |
| `encode_device` | choice: `gpu`, `cpu` | default `"gpu"` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `relay_options` | `RELAY_OPTIONS` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `WANVIDEOMODEL` | — |
| 1 | `text_embeds` | `WANVIDEOTEXTEMBEDS` | — |


### PromptRelayEncodeSmartC2C

**Shown in the menu as:** Prompt Relay Encode (Smart) — deprecated

DEPRECATED — use PromptRelayEncodeC2C with backend='smart'.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `clip` | `CLIP` |  | — |
| `latent` | `LATENT` |  | — |
| `global_prompt` | `STRING` | default `""`, multiline | — |
| `smart_prompt` | `STRING` | default `""`, multiline | — |
| `normalize_by_tokens` | `BOOLEAN` | default `False` | — |
| `epsilon` | `FLOAT` | default `0.001`, range 1e-06…0.99, step 0.0001 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `relay_options` | `RELAY_OPTIONS` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |
| 1 | `positive` | `CONDITIONING` | — |


### PromptRelayRestoreKijaiC2C

**Shown in the menu as:** Prompt Relay Restore (Kijai)

Undo a Prompt Relay (Kijai) patch — restores the original cross_attn.forward.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `WANVIDEOMODEL` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `WANVIDEOMODEL` | — |


---

## MEC/Audio


### AudioReverserMEC

**Shown in the menu as:** Audio Reverser

Reverses the audio waveform (torch.flip on the sample dim), preserving sample rate and channels. TIMEMAP maps original to reversed timestamps (t_new = duration - t_orig).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `audio` | `AUDIO` |  | Standard ComfyUI audio ({'waveform': [B,C,S], 'sample_rate': int}). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `audio` | `AUDIO` | — |
| 1 | `time_map` | `TIMEMAP` | — |


---

## MEC/Color Science


### C2CACESTonemap

**Shown in the menu as:** C2C ACES Tonemap

ACES filmic tone mapping with exposure, contrast, and saturation controls. Converts linear-light or overbright pixels to display-ready output with film-like highlight rolloff.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `source_space` | choice: `sRGB`, `Linear`, `Log C3` | default `"sRGB"` | Input color space. The image is linearised from this space before ACES processing. |
| `exposure` | `FLOAT` | default `1.0`, range 0.01…10.0, step 0.05 | Exposure multiplier applied before tone mapping. |
| `contrast` | `FLOAT` | default `1.0`, range 0.5…2.0, step 0.05 | Contrast adjustment (applied in log space). |
| `saturation` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.05 | Color saturation. <1 desaturates, >1 boosts. |
| `output_colorspace` | choice: `sRGB (gamma)`, `Linear`, `ACES AP1` | default `"sRGB (gamma)"` | Output color space. sRGB for display, Linear for compositing. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `out0` | `IMAGE` | — |


### C2CColorSpaceConvert

**Shown in the menu as:** C2C Color Space Convert


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `source_space` | choice: `sRGB`, `Linear`, `Log C3` | default `"sRGB"` | — |
| `target_space` | choice: `sRGB`, `Linear`, `Log C3` | default `"Linear"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `out0` | `IMAGE` | — |


### C2CVAEQualityDecode

**Shown in the menu as:** C2C VAE Quality Decode (HDR)

High-fidelity VAE decode for Wan video. Forces fp32 precision, uses spatial-only tiling to prevent frame flickering, and optionally applies ACES tone mapping for HDR-quality output.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `samples` | `LATENT` |  | — |
| `vae` | `VAE` |  | — |
| `force_fp32` | `BOOLEAN` | default `True` | Force fp32 during VAE decode for maximum quality. |
| `tile_size` | `INT` | default `0`, range 0…1024, step 64 | Spatial tile size (0=auto/no tiling). Set 256+ for 1080p. |
| `apply_aces` | `BOOLEAN` | default `False` | Apply ACES filmic tone mapping after decode. |
| `exposure` | `FLOAT` | default `1.0`, range 0.01…10.0, step 0.05 | Exposure for ACES (only used if apply_aces=True). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `out0` | `IMAGE` | — |


---

## MEC/Masking


### MaskPlacementMEC

**Shown in the menu as:** Mask Placement — Prompt/Ref → Place → Track

Universal mask placement: source an alpha (wire a mask, or give a text prompt + SAM model and it segments the object for you), place it with a draggable 4-corner perspective quad, and propagate across the video. Slice 1 ships static propagation; object tracking (Cutie) and landmark-locked follow are staged next.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | Video frames or a single still (B,H,W,3). |
| `prompt` | `STRING` | default `""` | What to cut out, e.g. 'dog', 'car', 'mouth'. Needs sam_model (+ a GroundingDINO model) wired. Ignored when source_mask is wired. |
| `grounding_model` | choice: `none`, `groundingdino_swint_ogc`, `groundingdino_swinb_cogcoor` | default `"groundingdino_swinb_cogcoor"` | GroundingDINO model for text->box grounding (same list as SAM Mask Generator). |
| `track_mode` | choice: `static`, `object_track`, `landmark_lock`, `auto` | default `"static"` | How the placement follows the video. Slice 1: 'static' is live; 'object_track' (Cutie) and 'landmark_lock' fall back to static for now and say so in `info`. |
| `anchor_frame` | `INT` | default `0`, range 0…99999 | The frame the quad was placed on (and segmented from, when using prompt without ref_image). |
| `placement_json` | `STRING` | default `""`, multiline | Editor-owned: {"corners":[[x,y]x4 TL,TR,BR,BL], "feather":px} in anchor-frame pixel coords. Empty = centered 60%-size quad. |
| `feather_px` | `INT` | default `6`, range 0…128 | Edge feather (Gaussian, px) applied to the placed alpha. placement_json's feather wins when present. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `source_mask` | `MASK` |  | BYO alpha (e.g. SAM Mask Generator / matting output). Skips prompt segmentation entirely. |
| `source_image` | `IMAGE` |  | RGB that source_mask cuts out of (for the placed_rgb output). When absent, the mask is placed without pixels. |
| `ref_image` | `IMAGE` |  | Segment the prompt from THIS image instead of the video's anchor frame (e.g. a photo of the dog you want). |
| `sam_model` | `SAM_MODEL` |  | From SAM Model Loader — needed only for prompt-based segmentation. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `masks` | `MASK` | Per-frame placed alpha (B,H,W float 0..1). Feed to inpaint/composite. |
| 1 | `placed_rgb` | `IMAGE` | Per-frame RGB of the placed source pixels over black (use with `masks` to composite). |
| 2 | `overlay_preview` | `IMAGE` | Frames with the placement tinted + quad drawn — visual verification. |
| 3 | `info` | `STRING` | JSON: source mode, quad, per-frame status, and which tracking actually ran. |


---

## MEC/Plate


### PARDesqueezeMEC

**Shown in the menu as:** PAR Desqueeze (anamorphic → square px)

Convert an anamorphic plate to SQUARE pixels before AI processing (Nuke: 4448x3840 @ PAR 1.7266 → 7680x3840 at 2:1). Wire par_info into PARResqueezeMEC to restore the original geometry losslessly.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `par_preset` | choice: `custom`, `square 1.0`, `anamorphic 2x (2.0)`, `anamorphic 1.8x (1.8)`, `ARRI 4448x3840→2:1 (1.7266)`, `anamorphic 1.5x (1.5)`, `anamorphic 1.33x (1.33)`, `NTSC DV (0.9091)`, … (+1) | default `"ARRI 4448x3840→2:1 (1.7266)"` | — |
| `pixel_aspect` | `FLOAT` | default `1.7266`, range 0.1…4.0, step 0.0001 | Used when par_preset = custom. PAR>1 = pixels wider than tall. |
| `method` | choice: `stretch_width`, `squash_height` | default `"stretch_width"` | stretch_width keeps every scanline (recommended); squash_height keeps the pixel count low. |
| `filter` | choice: `bicubic`, `bilinear`, `nearest`, `area` | default `"bicubic"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `par_info` | `STRING` | — |


### PARResqueezeMEC

**Shown in the menu as:** PAR Resqueeze (back to plate)

Return an AI-processed square-pixel frame to the plate's original pixel dimensions (PAR metadata is reapplied in Nuke). Wire par_info from PARDesqueezeMEC — the round trip restores the exact original W×H even if the AI changed resolution.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `par_info` | `STRING` | default `""`, **connection-only** | — |
| `filter` | choice: `bicubic`, `bilinear`, `nearest`, `area` | default `"bicubic"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `info` | `STRING` | — |


---

## MEC/RenderFarm


### C2C_ClusterStatus

**Shown in the menu as:** C2C Farm Cluster Status

Live render-farm capacity (queue depth, VRAM, reachability) as JSON.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `backend` | choice: `all`, `local-selftest`, `lan-workstation`, `aks-h100-pool`, `runpod-pod` |  | One backend, or 'all' for the whole cluster. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `cluster_json` | `STRING` | — |


### C2C_JobHistory

**Shown in the menu as:** C2C Farm Job History (Audit Log)

Render-farm audit log: per-user/status stats + recent jobs table.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `limit` | `INT` | default `50`, range 1…1000 | — |
| `user_filter` | `STRING` | default `""` | Empty = all users. |
| `project_filter` | `STRING` | default `""` | Empty = all projects. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `stats_json` | `STRING` | — |
| 1 | `jobs_table` | `STRING` | — |


### C2C_Submit

**Shown in the menu as:** C2C Farm Submit — Remote Render

Dispatch a full workflow JSON to a remote ComfyUI render-farm backend.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `prompt_json` | `STRING` | default `""`, multiline | The workflow to run remotely, in ComfyUI API format (Save (API Format) / the {'node_id': {class_type, inputs}} map). Accepts a bare prompt map or {'prompt': {...}}. |
| `backend` | choice: `local-selftest`, `lan-workstation`, `aks-h100-pool`, `runpod-pod` |  | Registered compute backend (renderfarm/config/backends.json). |
| `compute_profile` | choice: `heavy_94gb`, `light_30gb` |  | Gateway routing profile: heavy_94gb = 1 replica per 94GB GPU, light_30gb = 3 replicas per GPU. |
| `priority` | `INT` | default `5`, range 1…10 | Spool priority — higher dispatches first. |
| `project_name` | `STRING` | default `""` | Audit-log project tag. |
| `wait_for_completion` | `BOOLEAN` | default `True` | On: block until the remote render finishes and pull outputs back into input/c2c_farm_results/<job>/. Off: return the job id immediately and track it in the C2C Farm dashboard. |
| `timeout_minutes` | `INT` | default `120`, range 1…1440 | — |

**Hidden inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `unique_id` | `UNIQUE_ID` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `job_id` | `STRING` | — |
| 1 | `result_paths` | `STRING` | — |
| 2 | `info` | `STRING` | — |


---

## MEC/Temporal


### FluidShotDecoderMEC

**Shown in the menu as:** Fluid Shot Decoder (Restore Timing)

Selects exactly the frames that correspond to the ORIGINAL input frames using the encoder's TIMEMAP — restores the original frame count and slow-to-fast pacing.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | AI-processed frames (the expanded, normalized batch). |
| `time_map` | `TIMEMAP` |  | From Fluid Shot Encoder. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | — |


### FluidShotEncoderMEC

**Shown in the menu as:** Fluid Shot Encoder (Temporal Normalizer)

Analyzes optical-flow speed and inserts in-between frames on fast sections so motion per frame is constant — safe for Wan-style AI video. Pair with Fluid Shot Decoder to restore the original timing exactly.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | Video frames [B,H,W,C] 0-1. Speed-ramped source. |
| `max_px_per_frame` | `FLOAT` | default `5.0`, range 0.5…100.0, step 0.5 | Max allowed pixel displacement between consecutive output frames (95th-percentile optical-flow magnitude). Fast sections get flow-warped in-betweens until each step is under this. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `normalized_images` | `IMAGE` | — |
| 1 | `time_map` | `TIMEMAP` | — |


---

## MaskEditControl/Channels


### ShuffleMEC

**Shown in the menu as:** Shuffle — Channels (MEC)

Nuke-style 4-channel shuffle in a single node.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `out_R` | choice: `R`, `G`, `B`, `A`, `Lum`, `InvR`, `InvG`, `InvB`, … (+3) | default `"R"` | — |
| `out_G` | choice: `R`, `G`, `B`, `A`, `Lum`, `InvR`, `InvG`, `InvB`, … (+3) | default `"G"` | — |
| `out_B` | choice: `R`, `G`, `B`, `A`, `Lum`, `InvR`, `InvG`, `InvB`, … (+3) | default `"B"` | — |
| `out_A` | choice: `R`, `G`, `B`, `A`, `Lum`, `InvR`, `InvG`, `InvB`, … (+3) | default `"A"` | — |
| `premultiply_output` | `BOOLEAN` | default `False` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `alpha` | `MASK` | — |


---

## MaskEditControl/Clipboard


### TclParseMEC

**Shown in the menu as:** TCL Parse (MEC)

Parse Nuke-style TCL into a subgraph JSON description.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `tcl` | `STRING` | default `""`, multiline | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `subgraph_json` | `STRING` | — |
| 1 | `node_count` | `INT` | — |


### TclSerializeMEC

**Shown in the menu as:** TCL Serialize (MEC)

Serialize a subgraph JSON description into Nuke-style TCL.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `subgraph_json` | `STRING` | default `"{"nodes":[],"links":[]}"`, multiline | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `tcl` | `STRING` | — |


---

## MaskEditControl/Color


### ColorSpaceConvertMEC

**Shown in the menu as:** Color Space Convert (MEC)

Convert IMAGE between sRGB, linear, Rec.709, and ACEScg.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `src_space` | choice: `srgb`, `linear`, `rec709`, `acescg` | default `"srgb"` | — |
| `dst_space` | choice: `srgb`, `linear`, `rec709`, `acescg` | default `"linear"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |


### ExposureGradeMEC

**Shown in the menu as:** Exposure Grade (MEC)

Exposure (stops), WB (temp/tint), and contrast around a pivot.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `exposure_stops` | `FLOAT` | default `0.0`, range -10.0…10.0, step 0.05 | Exposure adjustment in stops (linear multiply by 2**stops). |
| `temperature` | `FLOAT` | default `0.0`, range -100.0…100.0, step 1.0 | — |
| `tint` | `FLOAT` | default `0.0`, range -100.0…100.0, step 1.0 | — |
| `contrast` | `FLOAT` | default `1.0`, range 0.0…4.0, step 0.05 | Contrast multiplier around the mid-grey pivot. |
| `pivot` | `FLOAT` | default `0.18`, range 0.001…0.999, step 0.001 | Mid-grey pivot for contrast (0.18 = scene-linear grey). |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `operate_in_linear` | `BOOLEAN` | default `True` | If True (recommended), input is treated as sRGB-encoded, linearized for the math, then re-encoded. If False, the math is done directly on the encoded values (legacy / display-referred). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |


### LUTApplyMEC

**Shown in the menu as:** LUT Apply (.cube) (MEC)

Apply a .cube LUT (Adobe format, 1D or 3D) with optional strength blend.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `lut_path` | `STRING` | default `""` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `strength` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `info_json` | `STRING` | — |


---

## MaskEditControl/Edit


### MaskEditMEC

**Shown in the menu as:** Mask Edit — Transform/Draw/Points/BBox

Unified mask edit dispatcher. Pure-CPU. Modes: transform, draw_shape, draw_advanced, points_bbox, bbox_smooth. Pick a mode and the corresponding widgets drive that engine. All outputs are normalized to the same 8-port schema.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mode` | choice: `transform`, `draw_shape`, `draw_advanced`, `points_bbox`, `bbox_smooth` | default `"transform"` | transform: morph / blur / offset / feather / threshold (needs mask). draw_shape: pick a shape from a dropdown, set its params (12 shapes). draw_advanced: power-mode shape_params_json (raw JSON). points_bbox: interactive points + bbox canvas (SAM/SeC coords). bbox_smooth: temporally smooth a sequence of [x,y,w,h] boxes. |
| `expand_x` | `INT` | default `0`, range -512…512, step 1 | [transform] dilate/erode along X |
| `expand_y` | `INT` | default `0`, range -512…512, step 1 | [transform] dilate/erode along Y |
| `blur_x` | `FLOAT` | default `0.0`, range 0.0…128.0, step 0.5 | [transform] Gaussian sigma X |
| `blur_y` | `FLOAT` | default `0.0`, range 0.0…128.0, step 0.5 | [transform] Gaussian sigma Y |
| `offset_x` | `INT` | default `0`, range -4096…4096, step 1 | [transform] pixel shift X |
| `offset_y` | `INT` | default `0`, range -4096…4096, step 1 | [transform] pixel shift Y |
| `feather` | `FLOAT` | default `0.0`, range 0.0…128.0, step 0.5 | [transform/draw] feather radius |
| `threshold` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | [transform] binarize threshold |
| `invert` | `BOOLEAN` | default `False` | [transform] invert output |
| `width` | `INT` | default `512`, range 1…16384 | [draw_*/points_bbox] canvas width |
| `height` | `INT` | default `512`, range 1…16384 | [draw_*/points_bbox] canvas height |
| `shape` | choice: `circle`, `rectangle`, `ellipse`, `polygon`, `line`, `triangle`, `star`, `diamond`, … (+4) | default `"circle"` | [draw_shape] geometry |
| `cx` | `FLOAT` | default `256.0`, range -16384.0…16384.0, step 0.5 | — |
| `cy` | `FLOAT` | default `256.0`, range -16384.0…16384.0, step 0.5 | — |
| `radius` | `FLOAT` | default `50.0`, range 0.0…8192.0, step 0.5 | — |
| `size_w` | `FLOAT` | default `200.0`, range 0.0…16384.0, step 0.5 | — |
| `size_h` | `FLOAT` | default `100.0`, range 0.0…16384.0, step 0.5 | — |
| `rx` | `FLOAT` | default `100.0`, range 0.0…8192.0, step 0.5 | — |
| `ry` | `FLOAT` | default `50.0`, range 0.0…8192.0, step 0.5 | — |
| `top_left_x` | `FLOAT` | default `100.0`, range -16384.0…16384.0, step 0.5 | — |
| `top_left_y` | `FLOAT` | default `100.0`, range -16384.0…16384.0, step 0.5 | — |
| `x2` | `FLOAT` | default `400.0`, range -16384.0…16384.0, step 0.5 | — |
| `y2` | `FLOAT` | default `400.0`, range -16384.0…16384.0, step 0.5 | — |
| `thickness` | `FLOAT` | default `5.0`, range 0.0…500.0, step 0.5 | — |
| `outer_r` | `FLOAT` | default `100.0`, range 0.0…8192.0, step 0.5 | — |
| `inner_r` | `FLOAT` | default `40.0`, range 0.0…8192.0, step 0.5 | — |
| `num_points` | `INT` | default `5`, range 3…50 | — |
| `corner_radius` | `FLOAT` | default `20.0`, range 0.0…4096.0, step 0.5 | — |
| `cross_size` | `FLOAT` | default `100.0`, range 0.0…8192.0, step 0.5 | — |
| `arrow_length` | `FLOAT` | default `200.0`, range 0.0…16384.0, step 0.5 | — |
| `head_length` | `FLOAT` | default `60.0`, range 0.0…8192.0, step 0.5 | — |
| `head_width` | `FLOAT` | default `80.0`, range 0.0…8192.0, step 0.5 | — |
| `points_json_shape` | `STRING` | default `"[[100,100],[400,100],[400,400],[100,400]]"`, multiline | [draw_shape] polygon vertices when shape=polygon |
| `value` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | [draw_*] fill intensity |
| `rotation` | `FLOAT` | default `0.0`, range -360.0…360.0, step 0.5 | [draw_*] rotation deg |
| `operation` | choice: `set`, `add`, `subtract`, `max`, `min` | default `"set"` | [draw_*] blend op |
| `batch_size` | `INT` | default `1`, range 1…256 | [draw_shape] number of frames |
| `shape_params_json` | `STRING` | default `"{"cx": 256, "cy": 256, "radius": 50}"`, multiline | [draw_advanced] raw shape_params JSON (see MaskDrawFrame) |
| `editor_data` | `STRING` | default `"{"points":[],"bboxes":[]}"`, multiline | [points_bbox] JSON from the interactive canvas |
| `default_radius` | `FLOAT` | default `3.0`, range 0.5…256.0, step 0.5 | [points_bbox] default brush radius |
| `softness` | `FLOAT` | default `1.0`, range 0.0…10.0, step 0.1 | [points_bbox] gaussian sigma multiplier |
| `normalize` | `BOOLEAN` | default `True` | [points_bbox] clamp output to [0,1] |
| `bboxes_json` | `STRING` | default `"[]"`, multiline | [bbox_smooth] JSON array of [x,y,w,h] per frame |
| `smoothing_radius` | `INT` | default `3`, range 1…30, step 1 | [bbox_smooth] window radius |
| `smoothing_method` | choice: `median_then_exponential`, `moving_average`, `exponential`, `median` | default `"median_then_exponential"` | [bbox_smooth] smoothing strategy |
| `alpha` | `FLOAT` | default `0.3`, range 0.05…1.0, step 0.05 | [bbox_smooth] exponential factor |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | Source mask (transform requires this) |
| `reference_image` | `IMAGE` |  | Optional reference; canvas size matches it when supplied |
| `existing_mask` | `MASK` |  | Existing mask to blend onto in draw modes |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | Rendered mask for the active mode (zero-mask in bbox_smooth mode). |
| 1 | `positive_coords` | `STRING` | Positive points JSON (points_bbox mode); '[]' otherwise. |
| 2 | `negative_coords` | `STRING` | Negative points JSON (points_bbox mode); '[]' otherwise. |
| 3 | `bboxes` | `BBOX` | Positive bbox list (points_bbox); single-element list in bbox_smooth. |
| 4 | `neg_bboxes` | `BBOX` | Negative bbox list (points_bbox mode). |
| 5 | `points_json` | `STRING` | All points JSON (points_bbox); '[]' otherwise. |
| 6 | `bbox_json` | `STRING` | All bbox JSON (points_bbox) or smoothed bboxes JSON (bbox_smooth). |
| 7 | `primary_bbox` | `BBOX` | Primary [x,y,w,h]: first positive bbox or smoothed first bbox. |


---

## MaskEditControl/Geometry


### DepthWarpMEC

**Shown in the menu as:** Depth Warp (MEC)

Horizontal parallax warp driven by a depth pass.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `depth` | `IMAGE` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `max_shift_pixels` | `FLOAT` | default `16.0`, range -512.0…512.0, step 0.5 | — |
| `pivot` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |


### NormalToCurvatureMEC

**Shown in the menu as:** Normal → Curvature (MEC)

Compute curvature mask from a tangent-space normal pass.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `normal` | `IMAGE` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `scale` | `FLOAT` | default `1.0`, range 0.1…32.0, step 0.1 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `curvature` | `MASK` | — |


### PositionPassSplitterMEC

**Shown in the menu as:** Position Pass Splitter (MEC)

Split position pass into X/Y/Z masks (auto- or manually-ranged).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `position` | `IMAGE` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `auto_normalize` | `BOOLEAN` | default `True` | — |
| `x_min` | `FLOAT` | default `0.0`, range -1000000.0…1000000.0 | — |
| `x_max` | `FLOAT` | default `1.0`, range -1000000.0…1000000.0 | — |
| `y_min` | `FLOAT` | default `0.0`, range -1000000.0…1000000.0 | — |
| `y_max` | `FLOAT` | default `1.0`, range -1000000.0…1000000.0 | — |
| `z_min` | `FLOAT` | default `0.0`, range -1000000.0…1000000.0 | — |
| `z_max` | `FLOAT` | default `1.0`, range -1000000.0…1000000.0 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `x_mask` | `MASK` | — |
| 1 | `y_mask` | `MASK` | — |
| 2 | `z_mask` | `MASK` | — |


---

## MaskEditControl/IO


### EXRMetadataReaderMEC

**Shown in the menu as:** EXR Metadata Reader (MEC)

Read OpenEXR header (compression, channels, custom attributes) without decoding pixels. Uses OpenEXR if installed, otherwise pure-Python parser.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `file_path` | `STRING` | default `""` | Absolute path to a .exr file. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `force_pure_python` | `BOOLEAN` | default `False` | Skip OpenEXR even if installed; useful for benchmarking. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `metadata_json` | `STRING` | — |
| 1 | `width` | `INT` | — |
| 2 | `height` | `INT` | — |
| 3 | `channels_csv` | `STRING` | — |


### LoadEXRMEC

**Shown in the menu as:** Load EXR (MEC)

Load EXR as scene-linear IMAGE. Tries OpenEXR → imageio.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `file_path` | `STRING` | default `""` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `info_json` | `STRING` | — |


### SaveEXRMEC

**Shown in the menu as:** Save EXR (MEC)

Save IMAGE batch as EXR(s).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `file_path` | `STRING` | default `""` | Absolute output path. Batches add _0001, _0002 suffixes. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `half_float` | `BOOLEAN` | default `True` | 16-bit half (smaller, recommended). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `info_json` | `STRING` | — |


---

## MaskEditControl/MaskMatting


### MaskTemporalMEC

**Shown in the menu as:** Mask Temporal Stabilizer + Integrity

Temporal stabilization + integrity check for video mask batches. Modes: none/gaussian/raft_flow. Emits a per-frame integrity report (area / centroid / IoU drift) and a human-readable warning string listing flagged frame indices.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `mask` | `MASK` |  | — |
| `temporal_mode` | choice: `none`, `gaussian`, `raft_flow` | default `"none"` | — |
| `blend` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.05 | Mix factor: 1.0 = pure warped-prev, 0.0 = current only. |
| `sigma` | `FLOAT` | default `1.0`, range 0.0…8.0, step 0.1 | Gaussian sigma (gaussian mode only). |
| `device` | choice: `cuda`, `cpu` | default `"cuda"` | — |
| `drop_threshold` | `FLOAT` | default `0.4`, range 0.0…1.0, step 0.05 | Area-ratio / IoU below this flags the frame. |
| `jump_threshold` | `FLOAT` | default `0.15`, range 0.0…1.0, step 0.01 | Centroid jump in normalized coords above this flags the frame. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | — |
| 1 | `integrity_json` | `STRING` | — |
| 2 | `warning` | `STRING` | — |


---

## MaskEditControl/Metadata


### FrameRangeRouterMEC

**Shown in the menu as:** Frame Range Router (MEC)

Slice a video batch by [start:end:step].


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |
| `start` | `INT` | default `0`, range -100000…100000 | — |
| `end` | `INT` | default `-1`, range -100000…100000 | — |
| `step` | `INT` | default `1`, range 1…1000 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | — |
| 1 | `mask` | `MASK` | — |
| 2 | `frame_count` | `INT` | — |


### MetadataWriterMEC

**Shown in the menu as:** Metadata Writer (MEC)

Write a JSON sidecar; pass the image through.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `sidecar_path` | `STRING` | default `""` | — |
| `metadata_json` | `STRING` | default `"{}"`, multiline | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `merge_existing` | `BOOLEAN` | default `False` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `written_path` | `STRING` | — |


### ShotMetadataNodeMEC

**Shown in the menu as:** Shot Metadata Reader (MEC)

Read a shot.json descriptor; missing fields return empty defaults.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `shot_json_path` | `STRING` | default `""` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `show` | `STRING` | — |
| 1 | `shot` | `STRING` | — |
| 2 | `task` | `STRING` | — |
| 3 | `frame_in` | `INT` | — |
| 4 | `frame_out` | `INT` | — |
| 5 | `fps` | `FLOAT` | — |
| 6 | `raw_json` | `STRING` | — |


---

## MaskEditControl/Pipeline


### MaskOpsMEC

**Shown in the menu as:** Mask + Matting

Production-grade segmentation + matting + AUTO-QUALITY + diagnostics in one node. Multi-backend (SAM 2.1 / SAM 3 / SAM 3.1 / BiRefNet / RMBG-2.0 / InSPyReNet + ViTMatte / RVM / MatAnyone). Optional Nuke-style luma-key pre-stage, edge-aware trimap, and AUTO-QUALITY pipeline that detects motion blur, low light, low contrast, speckle noise and similar bg/fg, then applies just-enough pre-processing before the segmenter and a light guided-filter polish on the alpha — no knobs to tune. When you supply pos+neg points (e.g. pos on face, neg on neck) it picks the SAM candidate that excludes the neg points, so you get the face only, not the whole person. For manual 11-stage refinement (hole-fill, joint bilateral, DenseCRF, etc.) chain MaskRefineMEC downstream.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Source image or video frames (B,H,W,C). |
| `segmenter` | choice: `auto_best`, `auto`, `sam2.1  [missing-deps]`, `sam3  [missing-deps]`, `sam3.1`, `birefnet`, `rmbg`, `inspyrenet`, … (+8) | default `"auto_best"` | Coarse-mask backend. Entries tagged [missing-deps] need an optional pip install to activate. |
| `matter` | choice: `none`, `auto`, `vitmatte`, `rvm`, `bgmattingv2`, `matanyone  [missing-deps]`, `birefnet`, `rmbg` | default `"vitmatte"` | Optional alpha refinement. 'none' returns the segmenter mask as alpha. |
| `model` | choice: `(auto)`, `[preset:sam2] sam2.1_hiera_tiny.safetensors`, `[preset:sam2] sam2.1_hiera_small.safetensors`, `[preset:sam2] sam2.1_hiera_base_plus.safetensors`, `[preset:sam2] sam2.1_hiera_large.safetensors`, `[preset:sam3] sam3.safetensors`, `[preset:sam3.1] sam3.1_multiplex_fp16.safetensors`, `[preset:sam3.1] sam3.1_multiplex_fp32.safetensors`, … (+12) | default `"(auto)"` | Specific weight file to use. Tag prefix selects the backend folder; '(auto)' lets each backend pick. |
| `matter_model` | choice: `(auto)`, `[preset:sam2] sam2.1_hiera_tiny.safetensors`, `[preset:sam2] sam2.1_hiera_small.safetensors`, `[preset:sam2] sam2.1_hiera_base_plus.safetensors`, `[preset:sam2] sam2.1_hiera_large.safetensors`, `[preset:sam3] sam3.safetensors`, `[preset:sam3.1] sam3.1_multiplex_fp16.safetensors`, `[preset:sam3.1] sam3.1_multiplex_fp32.safetensors`, … (+12) | default `"(auto)"` | Weight file for the matter backend. |
| `precision` | choice: `fp16`, `bf16`, `fp32` | default `"fp16"` | — |
| `attention` | choice: `auto`, `sdpa`, `flash`, `sage`, `xformers`, `eager` | default `"auto"` | — |
| `offload` | choice: `none`, `cpu`, `sequential` | default `"none"` | — |
| `subject_preset` | choice: `custom`, `hair`, `fur`, `cloth`, `skin_face`, `hard_edge`, `soft_glow` | default `"custom"` | Override trimap_dilate/erode/edge with subject-tuned values. |
| `trimap_dilate` | `INT` | default `8`, range 0…128, step 1 | — |
| `trimap_erode` | `INT` | default `8`, range 0…128, step 1 | — |
| `edge_radius` | `INT` | default `4`, range 0…64, step 1 | — |
| `individual_objects` | `BOOLEAN` | default `False` | If supported by the backend, return one mask per detected object. |
| `tracking_direction` | choice: `forward`, `backward`, `bidirectional` | default `"forward"` | — |
| `frame_annotation` | `INT` | default `0`, range 0…100000 | Frame index (in clip) where prompts are anchored. |
| `object_id` | `INT` | default `0`, range 0…1024 | — |
| `max_frames_to_track` | `INT` | default `0`, range 0…100000 | 0 = no cap. |
| `memory_size` | `INT` | default `8`, range 1…256 | — |
| `start_frame` | `INT` | default `0`, range 0…100000 | — |
| `end_frame` | `INT` | default `-1`, range -1…100000 | -1 = last frame. |
| `auto_download` | `BOOLEAN` | default `False` | Allow lazy auto-download from HF/torch.hub when a weight is missing. |
| `seed` | `INT` | default `0`, range 0…18446744073709551615 | — |
| `tta_flip` | `BOOLEAN` | default `False` | Test-time augmentation: run segmenter on the H-flipped image and average. Slower but cleaner. |
| `multiscale` | `BOOLEAN` | default `False` | Run the segmenter at 0.75x / 1.0x / 1.25x and fuse. Helps small / thin subjects. |
| `post_refine` | choice: `none`, `guided`, `crf`, `crf+guided` | default `"none"` | Final alpha refinement. 'guided' = guided filter (fast, torch-only). 'crf' = DenseCRF (requires pydensecrf, sharpest edges). |
| `refine_radius` | `INT` | default `8`, range 1…64 | Spatial radius for guided / CRF refinement. |
| `refine_iterations` | `INT` | default `5`, range 1…30 | CRF inference iterations. |
| `despill` | choice: `off`, `green`, `blue`, `red`, `magenta`, `cyan`, `yellow`, `white`, … (+2) | default `"off"` | Colour decontamination on the named backing. 'auto' estimates the colour from image corners. |
| `despill_strength` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.05 | How aggressively to subtract the spill (0 = off). |
| `preserve_skin` | `BOOLEAN` | default `True` | Keep warm pixels (R>G>B) untouched during despill. |
| `lightwrap_strength` | `FLOAT` | default `0.0`, range 0.0…2.0, step 0.05 | Light-wrap intensity. 0 = off; ~0.3-0.6 = natural blend over the new BG. |
| `lightwrap_radius` | `INT` | default `8`, range 1…64 | Light-wrap halo radius in pixels. |
| `edge_band_radius` | `INT` | default `4`, range 1…64 | Width of the soft edge band when splitting edge/inside/outside masks. |
| `premultiply` | `BOOLEAN` | default `True` | Premultiply preview by alpha. Disable for straight-alpha outputs. |
| `enable_luma_key` | `BOOLEAN` | default `False` | Run a luminance keyer on the source image BEFORE segmentation and use it as a hint / external_mask. |
| `luma_mode` | choice: `auto`, `highlights`, `midtones`, `shadows`, `custom` | default `"auto"` | — |
| `luma_low` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | — |
| `luma_high` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | — |
| `luma_gamma` | `FLOAT` | default `1.0`, range 0.01…10.0, step 0.01 | — |
| `luma_falloff` | `FLOAT` | default `1.0`, range 0.0…10.0, step 0.1 | — |
| `luma_invert` | `BOOLEAN` | default `False` | — |
| `luma_mix` | choice: `intersect`, `union`, `replace`, `hint_only` | default `"hint_only"` | How to combine the luma-key mask with the segmenter result. 'hint_only' = use as external_mask hint; 'intersect/union/replace' = combine with the final alpha. |
| `enable_advanced_trimap` | `BOOLEAN` | default `False` | Use the edge-aware trimap generator (asymmetric inner/outer scaling, image-edge snapping, smoothing) instead of the simple dilate/erode trimap. |
| `trimap_inner_scale` | `FLOAT` | default `1.0`, range 0.1…3.0, step 0.1 | — |
| `trimap_outer_scale` | `FLOAT` | default `1.5`, range 0.5…5.0, step 0.1 | — |
| `trimap_smooth` | `FLOAT` | default `0.0`, range 0.0…20.0, step 0.5 | — |
| `trimap_threshold` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | — |
| `auto_quality` | `BOOLEAN` | default `True` | AUTOMATIC robustness for hard images. Detects motion blur, low light, low contrast, speckle noise and low bg/fg colour separation, then applies just-enough pre-processing (CLAHE, unsharp, NL-means, chroma stretch) BEFORE the segmenter, and a light guided-filter+edge-snap polish on the alpha. No knobs. |
| `auto_disambiguate` | `BOOLEAN` | default `True` | When BOTH positive and negative points are supplied, score SAM's 3 candidate masks by (pos-coverage − neg-coverage − size-penalty) instead of raw score. This is what makes `pos=face, neg=neck` return just the face, not the whole person. |
| `quality_mode` | choice: `fast`, `balanced`, `max_fidelity` | default `"balanced"` | Strength of auto_quality pre/post processing. fast = mild, balanced = default, max_fidelity = NL-means denoise + larger guided filter. |
| `enable_diagnose` | `BOOLEAN` | default `True` | Run automatic mask-failure diagnostics (severity score + suggested method). |
| `diag_ring_width` | `INT` | default `5`, range 1…50 | — |
| `diag_blur_threshold` | `FLOAT` | default `50.0`, range 0.0…1000.0, step 1.0 | — |
| `diag_brightness_threshold` | `FLOAT` | default `0.15`, range 0.0…1.0, step 0.01 | — |
| `robust_propagation` | `BOOLEAN` | default `False` | Confidence-aware re-anchor loop for video. After SAM2 propagation, every frame's mask is scored vs the last good mask (IoU × size-ratio). When confidence drops below the threshold, the chosen re-anchor strategy fires (flow warp / DINOv2 region search / convex blend). Robust against motion blur + lighting drift. Single-image inputs skip this stage. |
| `robust_confidence_threshold` | `FLOAT` | default `0.65`, range 0.0…1.0, step 0.05 | Below this confidence the re-anchor fires. |
| `robust_reanchor_method` | choice: `blend`, `flow`, `dino`, `none` | default `"blend"` | flow=Farneback optical-flow warp of last good mask. dino=DINOv2 patch-feature region search + SAM2 re-prompt. blend=convex mix of current+warped. none=accept drifted output (debug). |
| `robust_blend_alpha` | `FLOAT` | default `0.7`, range 0.0…1.0, step 0.05 | Blend weight for current SAM2 mask when method=blend. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `positive_coords` | `STRING` | default `""`, **connection-only** | JSON list of positive points [[x,y],...] from PointsBBoxMaskEditor (positive_coords output). |
| `negative_coords` | `STRING` | default `""`, **connection-only** | JSON list of negative points [[x,y],...] from PointsBBoxMaskEditor (negative_coords output). |
| `pos_bbox` | `BBOX` |  | Single positive bbox [x0,y0,x1,y1]. |
| `neg_bbox` | `BBOX` |  | Optional negative bbox (excluded region). |
| `normal_bbox` | `BBOX` |  | Generic bbox if you don't care about polarity. |
| `text_prompt` | `STRING` | default `""`, **connection-only** | Open-vocabulary text prompt (SAM3 / GroundingDINO / VideoMaMa). Wire from any STRING source. |
| `external_mask` | `MASK` |  | Optional mask used as a hint or overridden when input_mode='auto' falls through. |
| `external_trimap` | `MASK` |  | Optional pre-computed trimap that bypasses internal trimap generation. |
| `holdout_mask` | `MASK` |  | Garbage / holdout matte. Pixels where this is >0 are FORCED to alpha=0 (used to chop out boom mics, rigs, etc). |
| `core_mask` | `MASK` |  | Core / inside matte. Pixels where this is >0 are FORCED to alpha=1 (used to lock down opaque interiors). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | Coarse mask from the segmenter (B,H,W). |
| 1 | `alpha` | `MASK` | Refined alpha after matter + refinement (production output). |
| 2 | `preview` | `IMAGE` | image * alpha premultiplied preview. |
| 3 | `trimap` | `MASK` | Trimap (0/0.5/1) used by the matter. |
| 4 | `bbox` | `BBOX` | Tight bbox around the alpha as [x0,y0,x1,y1]. |
| 5 | `bbox_json` | `STRING` | Same bbox as JSON {'x','y','w','h'}. |
| 6 | `score` | `FLOAT` | Overall production-quality score in [0,1] (boundary + coherence + size + smoothness). |
| 7 | `info` | `STRING` | JSON: backends, modes, per-frame quality breakdown, refine stages run, settings used. |
| 8 | `despilled` | `IMAGE` | Image with backing-colour spill suppressed (when `despill_strength`>0). |
| 9 | `lightwrap_rgba` | `IMAGE` | RGBA light-wrap layer to ADD over the new background. |
| 10 | `edge_mask` | `MASK` | Soft edge band where matting actually matters. |
| 11 | `inside_mask` | `MASK` | Solid-fg interior mask (safe to colour-grade). |
| 12 | `outside_mask` | `MASK` | Solid-bg exterior mask (safe to defocus / replace). |
| 13 | `luma_key_mask` | `MASK` | Luminance keyer output (empty if `enable_luma_key` is off). |
| 14 | `problem_regions` | `MASK` | Diagnostic problem-region heatmap from the failure explainer. |
| 15 | `severity` | `FLOAT` | Severity score [0,1] from the failure explainer. |
| 16 | `suggested_method` | `STRING` | Suggested next masking method (string) from the failure explainer. |


---

## MaskEditControl/PlateTools


### CleanPlateExtractorMEC

**Shown in the menu as:** Clean Plate Extractor (MEC)

Median across a batch (with optional mask exclusion) → clean plate.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `exclude_mask` | `MASK` |  | Pixels where mask>=0.5 are excluded from the median. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `clean_plate` | `IMAGE` | — |


### DifferenceMatteMEC

**Shown in the menu as:** Difference Matte (MEC)

Difference matte: |a-b| → MASK with threshold + softness.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image_a` | `IMAGE` |  | — |
| `image_b` | `IMAGE` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `metric` | choice: `l2`, `l1` | default `"l2"` | — |
| `threshold` | `FLOAT` | default `0.05`, range 0.0…1.0, step 0.001 | — |
| `softness` | `FLOAT` | default `0.05`, range 0.0…1.0, step 0.001 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | — |


### GrainMatchMEC

**Shown in the menu as:** Grain Match (MEC)

Extract grain from a reference plate and re-apply it to target.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `reference` | `IMAGE` |  | — |
| `target` | `IMAGE` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `intensity` | `FLOAT` | default `1.0`, range 0.0…4.0, step 0.05 | — |
| `denoise_kernel` | `INT` | default `5`, range 3…15, step 2 | — |
| `seed` | `INT` | default `0`, range 0…2147483647 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `info_json` | `STRING` | — |


### PlateStabilizerMEC

**Shown in the menu as:** Plate Stabilizer (MEC)

Stabilize a video batch to frame 0 via ORB+affine (cv2) or FFT translation.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `max_features` | `INT` | default `500`, range 50…5000, step 50 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | — |
| 1 | `info_json` | `STRING` | — |


---

## MaskEditControl/Pose


### FacePoseDeltaCoreMEC

**Shown in the menu as:** Face/Pose Delta Editor (C2C)

Apply user-edited landmark deltas to a per-frame face/pose landmark stream in anchor-relative space, so the edit follows the face as the head moves. Multi-keyframe with eased blend.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `landmarks` | `LANDMARKS` |  | Per-frame landmarks (T,N,2) — typically MediaPipe FaceMesh from WanAnimatePreprocessV2's gaze/pose detector. Single frame (N,2) is auto-promoted to T=1. |
| `keyframe_edits_json` | `STRING` | default `"{   "keyframes": [     {"frame": 0,  "deltas": {"61": [-0.02,  0.01]}},     {"frame": 30, "deltas": {"61": [-0.08,  0.04]}}   ],   "ease": "smooth_step",   "extrapolate": "hold" }"`, multiline | JSON describing one or more keyframe edits in anchor-relative space (1.0 unit = inter-ocular distance). |
| `anchor_mode` | choice: `intereye`, `centroid` | default `"intereye"` | How to compute the per-frame face anchor. 'intereye' is robust under head turns; 'centroid' is a fallback for non-face landmark sets. |
| `left_eye_idx` | `INT` | default `33`, range 0…100000 | Landmark index of the outer-left eye corner. MediaPipe FaceMesh = 33. |
| `right_eye_idx` | `INT` | default `263`, range 0…100000 | Landmark index of the outer-right eye corner. MediaPipe FaceMesh = 263. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `external_anchors_json` | `STRING` | default `""`, multiline | Optional override for per-frame anchors. JSON: {"centers": [[cx,cy], ...], "scales": [s, ...]}. If present, replaces the computed intereye/centroid anchors. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `landmarks_modified` | `LANDMARKS` | Modified landmarks (T, N, 2) — same shape and ordering as input. |
| 1 | `info_json` | `STRING` | JSON: per-frame anchor stats, keyframes used, deltas applied, weights at head/tail frames. Useful for debugging the blend. |


---

## MaskEditControl/Render


### DepthOfFieldMaskMEC

**Shown in the menu as:** Depth-of-Field Mask (MEC)

Convert depth pass → CoC mask (defocus alpha) and in-focus mask.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `depth` | `IMAGE` |  | — |
| `focus_distance` | `FLOAT` | default `0.5`, range 0.0…100.0, step 0.001 | — |
| `aperture` | `FLOAT` | default `0.1`, range 0.001…100.0, step 0.001 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `invert` | `BOOLEAN` | default `False` | Invert the focus mask (1 = in focus, 0 = defocus). |
| `depth_channel` | choice: `R`, `G`, `B`, `luma` | default `"R"` | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `coc_mask` | `MASK` | — |
| 1 | `in_focus_mask` | `MASK` | — |


### MergeRenderPassesMEC

**Shown in the menu as:** Merge Render Passes (MEC)

Composite beauty + auxiliary render passes.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `beauty` | `IMAGE` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `diffuse` | `IMAGE` |  | — |
| `specular` | `IMAGE` |  | — |
| `emission` | `IMAGE` |  | — |
| `ao` | `IMAGE` |  | — |
| `diffuse_gain` | `FLOAT` | default `0.0`, range -2.0…4.0, step 0.05 | — |
| `specular_gain` | `FLOAT` | default `0.0`, range -2.0…4.0, step 0.05 | — |
| `emission_gain` | `FLOAT` | default `1.0`, range 0.0…4.0, step 0.05 | — |
| `ao_strength` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |


---

## MaskEditControl/Roto


### VectorRotoMEC

**Shown in the menu as:** Vector Roto — Bezier (MEC)

Cubic-Bezier vector roto with Catmull-Rom keyframe tweening. Hands evaluated polylines off to a scanline rasteriser.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `roto_json` | `STRING` | default `"{"canvas": {"w": 1024, "h": 1024}, "frames": [{"frame": 0, "splines": []}]}"`, multiline | — |
| `frame_count` | `INT` | default `1`, range 1…4096 | — |
| `width` | `INT` | default `1024`, range 16…8192 | — |
| `height` | `INT` | default `1024`, range 16…8192 | — |
| `samples_per_seg` | `INT` | default `24`, range 2…256 | — |
| `feather_px` | `FLOAT` | default `0.0`, range 0.0…64.0, step 0.5 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | — |
| 1 | `info` | `STRING` | — |


---

## MaskEditControl/VFX


### OpticalFlowMEC

**Shown in the menu as:** Optical Flow Re-Vector (MEC)

Dense optical flow re-vector. RAFT primary, LK pyramid fallback. Re-vectoring is restricted to the mask region; rest is untouched.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `frame_a` | `IMAGE` |  | — |
| `frame_b` | `IMAGE` |  | — |
| `iters` | `INT` | default `20`, range 1…100 | — |
| `consistency_thr` | `FLOAT` | default `1.5`, range 0.0…20.0, step 0.05 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | — |
| `scale` | `FLOAT` | default `1.0`, range -4.0…4.0, step 0.05 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `re_vectored` | `IMAGE` | — |
| 1 | `flow_rgb` | `IMAGE` | — |
| 2 | `consistency` | `MASK` | — |


---

## MaskEditControl/Video


### VideoFrameExtractorMEC

**Shown in the menu as:** Video Frame Extractor (MEC)

Extract a single frame from a video batch. Single images pass through unchanged. Reports total frame count and whether input is a video batch.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | Image batch (B,H,W,C). Single images pass through; video batches select one frame. |
| `frame_index` | `INT` | default `0`, range 0…999999, step 1 | Which frame to extract (0-based). Clamped to batch length. |
| `mode` | choice: `specific_frame`, `first`, `last`, `middle` | default `"first"` | Frame selection mode: first: always frame 0 last: final frame middle: middle frame (B//2) specific_frame: use frame_index value |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `frame` | `IMAGE` | — |
| 1 | `total_frames` | `INT` | — |
| 2 | `is_video` | `BOOLEAN` | — |


---

## MaskEnhancedControl/Grounding


### LocateAnythingGroundingMEC

**Shown in the menu as:** LocateAnything Grounding (MEC)


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `prompt` | `STRING` | default `"person"`, multiline | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model_path` | `STRING` | default `"nvidia/LocateAnything-3B"` | — |
| `generation_mode` | choice: `hybrid`, `fast`, `slow` | default `"hybrid"` | — |
| `max_new_tokens` | `INT` | default `2048`, range 256…8192 | — |
| `device` | choice: `cuda`, `cpu` | default `"cuda"` | — |
| `confidence_threshold` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.05 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `bboxes` | `BBOX_LIST` | — |
| 1 | `annotated_image` | `IMAGE` | — |
| 2 | `raw_response` | `STRING` | — |


### LocateAnythingToSAMMEC

**Shown in the menu as:** LocateAnything → SAM Prompt (MEC)


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `bboxes` | `BBOX_LIST` |  | — |
| `image` | `IMAGE` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `expand_ratio` | `FLOAT` | default `0.05`, range 0.0…0.5, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | — |


---

## utils


### FolderIncrementer

**Shown in the menu as:** Folder Version Incrementer

Auto-incrementing per-label / per-date version counter. Scans the output directory for existing `vNNN` subfolders and emits the next one, plus folder name, subfolder path, filename prefix, and full output filename.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `prefix` | `STRING` | default `"v"` | Prefix before the version number (e.g. 'v' → v001) |
| `padding` | `INT` | default `3`, range 1…10 | Zero-pad width (3 → 001) |
| `suffix` | `STRING` | default `""` | Optional tag like '_mask' or '_wan'. WHERE it lands is controlled by suffix_mode:   filename  → '.../v001/clip_mask.mov'   subfolder → '.../v001/mask/clip.mov'  (separate folder per run)   folder    → 'ATG_..._mask/<date>/v001/clip.mov'  (separate top folder) Leave empty to disable. Sanitized for cross-platform safety. |
| `suffix_mode` | choice: `filename`, `subfolder`, `folder` | default `"filename"` | Where the `suffix` is applied:   filename  — append to the output basename (legacy behaviour)   subfolder — nest a folder AFTER the version, e.g. v001/mask/ — keeps mask vs wan outputs of the SAME run side by side, same version   folder    — append to the TOP folder name, e.g. ATG_..._mask/ — fully separate folder tree with its own versioning. For subfolder/folder a leading '_' or '-' is stripped from the folder name (so '_mask' → 'mask') and the basename is left clean. |
| `label` | `STRING` | default `"default"` | Fallback folder name (used only when no source file is connected) |
| `date_format` | choice: `MM-DD-YYYY`, `DD-MM-YYYY`, `YYYY-MM-DD` | default `"MM-DD-YYYY"` | Date format for the date subfolder (e.g. 02-22-2026 or 2026-02-22) |
| `path_style` | choice: `auto`, `windows`, `linux`, `macos` | default `"auto"` | Path separator style for output strings. auto=detect from current OS, windows=backslash, linux/macos=forward slash. Use 'auto' unless you design workflows on one OS and run on another. |
| `source_choice` | choice: `auto`, `image`, `video`, `custom` | default `"auto"` | Where the source name comes from. 'image' → trigger_image, 'video' → trigger_video, 'auto' → prefer video if connected, else image, else legacy `trigger`. 'custom' → use the ``custom_name`` widget verbatim and ignore all triggers. |
| `name_format` | choice: `basename`, `strip_tags`, `first_segment` | default `"basename"` | How to format the detected filename for folder + prefix:   basename      — strip extension only (e.g. clip_2160_25fps)   strip_tags    — also strip trailing res/fps tags (clip)   first_segment — keep only the first chunk before . or _ (clip) The original file extension is preserved on output_filename. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `trigger` | `*` |  | Legacy generic trigger – connect any output here. |
| `trigger_image` | `IMAGE` |  | Connect a LoadImage / image source here. Used when source_choice = 'image' or 'auto'. |
| `trigger_video` | `*` |  | Connect a LoadVideo / VHS_LoadVideo / video source here. Used when source_choice = 'video' or 'auto' (preferred). |
| `source_filename` | `STRING` | default `""` | Auto-filled from the connected loader (basename only, no extension). Drives folder name + output basename. Extension is stored separately in source_extension for output_filename. |
| `source_extension` | `STRING` | default `""` | Auto-filled file extension from the connected loader (e.g. .mp4, .mov). Used only for output_filename — not part of the folder name. |
| `custom_name` | `STRING` | default `""` | Manual source name. Only used when source_choice='custom'. May include an extension (e.g. 'my_shot.mp4'); if no extension is given, output_filename will have none either. Sanitized for cross-platform safety. |
| `base_path` | `STRING` | default `""` | Override base output directory.  Leave empty → ComfyUI output dir. |
| `folder_name_override` | `STRING` | default `""` | Force a specific folder name instead of deriving from the input filename. Sanitized for cross-platform safety. |
| `reserve_version` | `BOOLEAN` | default `False` | If True, create the version directory and write a `.reserved` marker file to claim the version number atomically. Prevents collisions in batch/render-farm workflows. Leave False for normal use (the directory will be created by ComfyUI's Save node when output is actually written). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `version_string` | `STRING` | Zero-padded version string, e.g. `v001`. |
| 1 | `version_number` | `INT` | Raw integer version number. |
| 2 | `folder_name` | `STRING` | Sanitized folder name derived from the input source (or label). |
| 3 | `subfolder_path` | `STRING` | Full subfolder path: `<base>/<folder>/<date>/<version>`. |
| 4 | `filename_prefix` | `STRING` | Filename prefix combining folder + version (for SaveImage). |
| 5 | `output_filename` | `STRING` | Final output filename including extension. |


### FolderIncrementerReset

**Shown in the menu as:** Folder Version Check

Report the current version state for a label/date folder. Scans <base>/<label>/<MM-DD-YYYY>/ and returns how many vNNN folders already exist plus the highest version number. To truly 'reset' a label, delete its date subfolder from disk.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `label` | `STRING` | default `"default"` | Folder name to inspect |
| `date_format` | choice: `MM-DD-YYYY`, `DD-MM-YYYY`, `YYYY-MM-DD` | default `"MM-DD-YYYY"` | Date format (must match what FolderIncrementer uses) |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `trigger` | `*` |  | Optional any-type trigger input. Connect any upstream output here to force this node to re-run after that node finishes (e.g. wire it to a SaveImage filename to recheck the version state after a save). |
| `base_path` | `STRING` | default `""` | Override base directory.  Leave empty → ComfyUI output dir. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `status` | `STRING` | Human-readable status describing how many versions exist for this label/date. |
| 1 | `current_version` | `INT` | Highest existing version number (0 if none yet). |


### FolderIncrementerSet

**Shown in the menu as:** Folder Version Set

Reserve version slots by creating empty placeholder directories under <base>/<label>/<MM-DD-YYYY>/. Creates v001 ... v{value} so the next FolderIncrementer run will produce v{value+1}. Useful for skipping ahead or reserving a known version range.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `label` | `STRING` | default `"default"` | Folder name under the output directory |
| `value` | `INT` | default `1`, range 1…999999 | Create placeholder dirs up to this version number |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `trigger` | `*` |  | Optional any-type trigger input. Wire any upstream output here to control when this node runs in the graph. |
| `prefix` | `STRING` | default `"v"` | Version-folder prefix. Default 'v' produces v001, v002, ... Must match what FolderIncrementer is using. |
| `padding` | `INT` | default `3`, range 1…10 | Zero-pad width for the version number (3 → v001, 4 → v0001). Must match what FolderIncrementer is using. |
| `base_path` | `STRING` | default `""` | Override base directory. Leave empty → ComfyUI output dir. |
| `date_format` | choice: `MM-DD-YYYY`, `DD-MM-YYYY`, `YYYY-MM-DD` | default `"MM-DD-YYYY"` | Date format (must match what FolderIncrementer uses) |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `status` | `STRING` | Status message confirming how many placeholder version dirs were created. |
| 1 | `next_version` | `INT` | The version number the *next* FolderIncrementer run will produce (value+1). |
