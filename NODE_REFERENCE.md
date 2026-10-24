# C2C / MEC Custom Node Packs — Node Reference

<!-- C2C:NODE-COUNTS:BEGIN -->
**ComfyUI-CustomNodePacks** registers **113** nodes across these families:

| Family | Nodes |
|--------|------:|
| Vault | 4 |
| Folder Incrementer | 5 |
| Render farm | 3 |
| C2C Distributed | 2 |
| Fluid Shots/Audio | 3 |
| C2C video | 4 |
| MaskEditControl | 14 |
| Image Mask Editor | 1 |
| Model analysis | 1 |
| MEC Paint Suite | 2 |
| Face Fixer | 1 |
| Face/Pose Delta | 1 |
| Mask + Matting | 2 |
| Layer Effects | 1 |
| Mask toolkit | 1 |
| Frequency / Grain | 3 |
| Smart Crop/Stitch | 2 |
| Tiled refinement | 4 |
| Magnific | 15 |
| AV Handles | 1 |
| Save Video | 1 |
| SAM Multi-Mask Picker | 1 |
| SAM Loader/Generator | 2 |
| Mask Placement | 1 |
| Semantic Segment | 1 |
| NukeNodeMax | 1 |
| Video Stabilizer | 4 |
| C2C helpers | 8 |
| Prompt Relay | 3 |
| AsymFlow sampler | 1 |
| HDR Color Science | 1 |
| LocateAnything | 2 |
| Nano Banana | 1 |
| Control AOV | 1 |
| Clipboard TCL | 2 |
| Restored VFX | 6 |
| Legacy (deprecated) | 7 |

Startup log line to verify the pack loaded:

`[C2C] CustomNodePacks: 113 nodes loaded (Vault 4, Folder Incrementer 5, Render farm 3, C2C Distributed 2, Fluid Shots/Audio 3, C2C video 4, MaskEditControl 14, Image Mask Editor 1, Model analysis 1, MEC Paint Suite 2, Face Fixer 1, Face/Pose Delta 1, Mask + Matting 2, Layer Effects 1, Mask toolkit 1, Frequency / Grain 3, Smart Crop/Stitch 2, Tiled refinement 4, Magnific 15, AV Handles 1, Save Video 1, SAM Multi-Mask Picker 1, SAM Loader/Generator 2, Mask Placement 1, Semantic Segment 1, NukeNodeMax 1, Video Stabilizer 4, C2C helpers 8, Prompt Relay 3, AsymFlow sampler 1, HDR Color Science 1, LocateAnything 2, Nano Banana 1, Control AOV 1, Clipboard TCL 2, Restored VFX 6, Legacy (deprecated) 7) - 0 failed`
<!-- C2C:NODE-COUNTS:END -->


The counts above are generated from live registration (`python tests/_load_summary.py --write-docs`,
checked by `tests/test_node_counts_match_docs.py`). The per-node sections below were generated from the
code on 2026-07-29 and have **no generator in this repo yet**, so they can drift from the tooltips they
mirror; treat the live node schema as authoritative until they are regenerated.


<!-- C2C:MOVED-NODES:BEGIN -->

**Documented in other packs** (these ids used to live here and moved; saved workflows load them from the pack that registers them now):

- ComfyUI-NukeMaxNodes (NODE_REFERENCE.md of ComfyUI-NukeMaxNodes): `CleanPlateExtractorMEC`, `ColorSpaceConvertMEC`, `DepthOfFieldMaskMEC`, `DepthWarpMEC`, `DifferenceMatteMEC`, `EXRMetadataReaderMEC`, `ExposureGradeMEC`, `FrameRangeRouterMEC`, `GrainMatchMEC`, `LUTApplyMEC`, `MergeRenderPassesMEC`, `MetadataWriterMEC`, `NormalToCurvatureMEC`, `PlateStabilizerMEC`, `PositionPassSplitterMEC`, `ShotMetadataNodeMEC`
- ComfyUI-WanNodeExperiments (NODE_REFERENCE.md of ComfyUI-WanNodeExperiments): `WanDirectorC2C`

<!-- C2C:MOVED-NODES:END -->

## Contents

- **C2C/AI Image** (1)
  - [Nano Banana · Gemini Image (C2C)](#nanobananac2c)
- **C2C/Control** (1)
  - [Control AOV — Multi-Control Fusion (C2C)](#controlaovc2c)
- **C2C/Diagnostics** (2)
  - [Mask Failure Explainer — Diagnostics](#maskfailureexplainermec)
  - [Model Metadata Extractor](#modelmetadataextractormec)
- **C2C/Distributed** (2)
  - [C2C GPU Inventory](#c2cgpuinventory)
  - [C2C Multi-GPU Split](#c2csequenceparallel)
- **C2C/Frequency** (3)
  - [Film Grain (MEC)](#filmgrainmec)
  - [Frequency Combine (MEC)](#frequencycombinemec)
  - [Frequency Separate (MEC)](#frequencyseparatemec)
- **C2C/Helpers** (8)
  - [Size (C2C)](#aspectpresetmec)
  - [Conditional Switch](#conditionalswitchmec)
  - [Execution Timer (C2C)](#executiontimermec)
  - [Batch Range (C2C)](#imagebatchslicemec)
  - [Probe (C2C)](#imagestatsprobemec)
  - [Number Lerp (C2C)](#numberlerpmec)
  - [Seed List Generator (C2C)](#seedlistmec)
  - [Text Template (C2C)](#texttemplatemec)
- **C2C/Inpaint** (7)
  - [Inpaint Crop Pro](#inpaintcroppromec)
  - [Inpaint Mask Prepare](#inpaintmaskpreparemec)
  - [Inpaint Paste Back](#inpaintpastebackmec)
  - [Inpaint Stitch Pro](#inpaintstitchpromec)
  - [ProPainter — Temporal / Remove / Stitch / Refine / Flow](#propaintermec)
  - [Smart Image Crop](#smartimagecropmec)
  - [Smart Image Stitch](#smartimagestitchermec)
- **C2C/Legacy** (7)
  - [BBox Crop (MEC) (legacy)](#bboxcrop)
  - [BBox From Mask (MEC) (legacy)](#bboxfrommask)
  - [BBox Pad (MEC) (legacy)](#bboxpad)
  - [BBox To Mask (MEC) (legacy)](#bboxtomask)
  - [Mask Composite Advanced (MEC) (legacy)](#maskcompositeadvanced)
  - [Mask Math (MEC) (legacy)](#maskmath)
  - [Mask Preview Overlay (MEC) (legacy)](#maskpreviewoverlay)
- **C2C/Magnific** (15)
  - [OmniScale Creation Picker](#magnificcreationpicker)
  - [OmniScale Generate Image](#magnificgenerateimage)
  - [OmniScale Generate Music](#magnificgeneratemusic)
  - [OmniScale Generate Video](#magnificgeneratevideo)
  - [OmniScale Library Reference](#magnificlibraryreference)
  - [OmniScale Metadata (unpack)](#magnificmetadata)
  - [OmniScale Remove Background](#magnificremovebackground)
  - [OmniScale Retouch (inpaint)](#magnificretouch)
  - [OmniScale Save To (project/folder)](#magnificsaveto)
  - [OmniScale Skin Enhancer](#magnificskinenhancer)
  - [OmniScale Stock Picker](#magnificstockpicker)
  - [OmniScale Stock Search](#magnificstocksearch)
  - [OmniScale Upscale Image](#magnificupscaleimage)
  - [OmniScale Upscale Video](#magnificupscalevideo)
  - [OmniScale Voiceover (TTS)](#magnificvoiceover)
- **C2C/Mask** (1)
  - [Image Mask Editor (C2C)](#imagemaskeditorc2c)
- **C2C/ModelAnalysis** (1)
  - [VAE Inspect](#vaeblockinspectormec)
- **C2C/Paint** (3)
  - [Advanced Paint Canvas](#mecadvancedpaintcanvas)
  - [Context Inpainter / Fixer](#meccontextinpainter)
  - [Face Fixer](#mecfacefixer)
- **C2C/Pipeline** (1)
  - [Mask Refiner](#maskrefinemec)
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
- **C2C/Tiling** (4)
  - [Identity Lock — refine without moving](#identitylockmec)
  - [Tile Merge](#tilemergemec)
  - [Tile Plan](#tileplanmec)
  - [Tile Split](#tilesplitmec)
- **C2C/Utils** (3)
  - [Parameter History](#parameterhistorymec)
  - [Folder Version Control (check / set)](#folderversioncontrol)
  - [Folder Version Variant (per output)](#folderversionvariant)
- **C2C/VAE** (1)
  - [VAE Merge](#vaemergemec)
- **C2C/Vault** (4)
  - [Vault input (edit view)](#c2c_vaulteditinput)
  - [Vault output (edit view)](#c2c_vaulteditoutput)
  - [C2C Vault — Locked (password to run)](#c2c_vaultlocked)
  - [C2C Vault — Sealed (runs without password)](#c2c_vaultsealed)
- **C2C/Video** (7)
  - [Mask Track — Motion/Propagate/Anchor/Consistency/Stabilize](#masktrackermec)
  - [AV Handles — Add / Trim](#avhandlesmec)
  - [Load Images Path (C2C)](#loadimagespathc2c)
  - [Load Video (C2C)](#loadvideoc2c)
  - [Load Video Path (C2C)](#loadvideopathc2c)
  - [Save Video (C2C)](#savevideoc2c)
  - [Video Info (C2C)](#videoinfoc2c)
- **C2C/VideoMask** (1)
  - [Video Mask Editor](#videomaskeditormec)
- **Code2Collapse/Sampling** (1)
  - [AsymFlow Sampler Patch (Lakonik signal-shift)](#asymflowsamplerpatch)
- **ComfyUI-CustomNodePacks/PromptRelay** (3)
  - [Prompt Relay Advanced Options](#promptrelayadvancedoptionsc2c)
  - [Prompt Relay Encode](#promptrelayencodec2c)
  - [Prompt Relay Restore (Kijai)](#promptrelayrestorekijaic2c)
- **MEC/Audio** (1)
  - [Audio Reverser](#audioreversermec)
- **MEC/Color Science** (1)
  - [VAE Decode (C2C)](#c2cvaequalitydecode)
- **MEC/LayerEffects** (1)
  - [Layer Effects (C2C)](#layereffectsmec)
- **MEC/Mask** (1)
  - [Mask Tools (C2C)](#masktoolsmec)
- **MEC/Masking** (1)
  - [Mask Placement — Prompt/Ref → Place → Track](#maskplacementmec)
- **MEC/Plate** (2)
  - [PAR Desqueeze (anamorphic → square px)](#pardesqueezemec)
  - [PAR Resqueeze (back to plate)](#parresqueezemec)
- **MEC/RenderFarm** (3)
  - [C2C Farm Cluster Status](#c2c_clusterstatus)
  - [C2C Farm Job History (Audit Log)](#c2c_jobhistory)
  - [C2C Farm Submit — Remote Render](#c2c_submit)
- **MEC/Temporal** (2)
  - [Fluid Shot Decoder (Restore Timing)](#fluidshotdecodermec)
  - [Fluid Shot Encoder (Temporal Normalizer)](#fluidshotencodermec)
- **MaskEditControl/Clipboard** (2)
  - [TCL Parse (MEC)](#tclparsemec)
  - [TCL Serialize (MEC)](#tclserializemec)
- **MaskEditControl/Edit** (1)
  - [Mask Edit — Transform/Draw/Points/BBox](#maskeditmec)
- **MaskEditControl/IO** (2)
  - [Load EXR (MEC)](#loadexrmec)
  - [Save EXR (MEC)](#saveexrmec)
- **MaskEditControl/Pipeline** (1)
  - [Mask + Matting](#maskopsmec)
- **MaskEditControl/Pose** (1)
  - [Face/Pose Delta Editor (C2C)](#faceposedeltacoremec)
- **MaskEditControl/Roto** (1)
  - [Vector Roto — Bezier (MEC)](#vectorrotomec)
- **MaskEditControl/VFX** (1)
  - [Optical Flow Re-Vector (MEC)](#opticalflowmec)
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


---

## C2C/Distributed


### C2CGPUInventory

**Shown in the menu as:** C2C GPU Inventory

Lists the GPUs (architecture Turing to Blackwell, memory, BF16 / FP8 / FP4 support, whether this torch build has kernels for them), how they are linked (P2P, NVLink), Grace / aarch64 hosts, whether NCCL / gloo really work here, and which multi-GPU modes this box can run.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `probe_backends` | `BOOLEAN` | default `True` | Really create a NCCL / gloo process group in a subprocess to see whether it works (a few seconds, cached until ComfyUI restarts). Off: report only what the hardware says. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `summary` | `STRING` | — |
| 1 | `report_json` | `STRING` | — |


### C2CSequenceParallel

**Shown in the menu as:** C2C Multi-GPU Split

Splits every model call over the GPUs of this machine. Any model: the call's batch items (cond and uncond, an image batch, Tile Split's tiles) go to different GPUs, with no exchange at all. Wan 2.1 / 2.2 and MiniMax H3 also split the tokens of one item (Ulysses all-to-all / Ring, exact). One model copy per GPU, made by ComfyUI's own multi-GPU clones. Works on Windows and Linux; with one GPU the model passes through unchanged. Put it last in the model chain (after LoRA / attention nodes). Replaces core's 'MultiGPU CFG Split' for this model.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `model` | `MODEL` |  | — |
| `gpus` | `INT` | default `0`, range 0…64 | How many GPUs to use; 0 = all that ComfyUI sees. Each GPU holds a full copy of the model. |
| `strategy` | choice: `auto`, `ulysses`, `ring` | default `"auto"` | auto: Ulysses as far as the head count and fast links allow, Ring for the rest (USP). ulysses: prefer head-parallel all-to-all (best on NVLink). ring: K/V blocks passed around (best across slow links). |
| `min_tokens_per_gpu` | `INT` | default `1024`, range 1…1048576 | Calls with fewer tokens per GPU run on one GPU: below this the exchange costs more than it saves. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `fast_link_gpus` | `INT` | default `0`, range 0…64 | How many GPUs share fast links (NVLink domain). 0 = all of them. Ulysses stays inside this many GPUs. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `MODEL` | — |
| 1 | `plan` | `STRING` | — |


---

## C2C/Frequency


### FilmGrainMEC

**Shown in the menu as:** Film Grain (MEC)

Lightroom-style film grain with per-frame animation for video batches.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Lightroom / Camera Raw-style film grain: Amount, Size, Roughness. Monochrome by default; raise Color for dye-cloud colour grain. |
| `amount` | `FLOAT` | default `25.0`, range 0.0…100.0, step 0.5 | How much grain shows through (Lightroom Amount). |
| `size` | `FLOAT` | default `25.0`, range 0.0…100.0, step 0.5 | Grain size — bigger = coarser; noise is generated at reduced resolution then upscaled. |
| `roughness` | `FLOAT` | default `50.0`, range 0.0…100.0, step 0.5 | Irregularity of the grain: blends a fine and a coarse field, like Lightroom. |
| `color` | `FLOAT` | default `0.0`, range 0.0…100.0, step 1.0 | 0 = clean monochrome grain. Raise for coloured dye-cloud grain. |
| `animate` | `BOOLEAN` | default `True` | Fresh grain on every frame of the batch — real film shimmer. Off = the same static grain on all frames. |
| `seed` | `INT` | default `0`, range 0…18446744073709551615 | Same seed, same grain. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | Optional — confine the grain to the mask, feathered by its values (soft edges blend in). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `report` | `STRING` | — |


### FrequencyCombineMEC

**Shown in the menu as:** Frequency Combine (MEC)

Merge a high-frequency detail layer onto a low-frequency base. Output is clamped to 0..1 on the IMAGE socket — values above 1.0 are clipped; this is an 8-bit-range IMAGE, not a scene-linear plate.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `high_frequency` | `IMAGE` |  | Detail layer from Frequency Separate (mode/linear must match). |
| `low_frequency` | `IMAGE` |  | Base image the detail lands on (e.g. the relit result). Leaves Separate in display space when linear=True; Combine re-linearises it. |
| `mode` | choice: `Divide`, `Subtract` | default `"Divide"` | Must match the Separate that made the HF. |
| `linear` | `BOOLEAN` | default `True` | Process in linear light (correct). Turn off to work in gamma like classic Photoshop. Must match between Separate and Combine. |
| `detail_strength` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.01 | Attenuate (<1) or boost (>1) the transferred detail. 0.0 restores the base only. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | Optional — apply the detail only inside the mask (feathered by its values). |
| `mask_feather` | `FLOAT` | default `0.0`, range 0.0…64.0, step 1.0 | Soften the mask edge before applying detail. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `report` | `STRING` | — |


### FrequencySeparateMEC

**Shown in the menu as:** Frequency Separate (MEC)

Split an image into low-frequency base and high-frequency detail. HF leaves in working space; LF leaves in display space when linear=True.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Split into a soft base (low frequency) and a fine detail layer (high frequency). Pair with Frequency Combine to transfer texture. |
| `method` | choice: `Gaussian`, `Guided`, `Rolling Guidance`, `Median` | default `"Guided"` | How the base is smoothed. Guided keeps edges clean without the halo Gaussian leaves; Rolling Guidance erases texture by size while keeping shapes; Median is for spot blemishes (radius capped internally). |
| `radius` | `INT` | default `8`, range 1…128, step 1 | Detail scale. Larger = coarser base, more goes into the detail layer. Median mode caps the effective radius at 7 — see the report for what ran. |
| `edge_threshold` | `FLOAT` | default `0.1`, range 0.0…1.0, step 0.01 | How strongly edges are protected (Guided / Rolling Guidance). Lower = sharper edges kept. |
| `mode` | choice: `Divide`, `Subtract` | default `"Divide"` | How detail is encoded. Divide (ratio) is lighting-invariant — best for transferring detail between differently-lit images. Negative ratios are discarded on Combine before recombine. |
| `detail` | choice: `Luminance`, `RGB` | default `"Luminance"` | Luminance keeps detail achromatic so a Divide recombine never touches color. RGB carries chromatic detail too. |
| `linear` | `BOOLEAN` | default `True` | Process in linear light (correct). Turn off to work in gamma like classic Photoshop. Must match between Separate and Combine. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | Optional — attenuate the detail layer outside the mask on high_frequency_masked. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `high_frequency` | `IMAGE` | — |
| 1 | `low_frequency` | `IMAGE` | — |
| 2 | `high_frequency_masked` | `IMAGE` | — |
| 3 | `report` | `STRING` | — |


---

## C2C/Helpers


### AspectPresetMEC

**Shown in the menu as:** Size (C2C)

Width and height for a render: a common aspect ratio scaled to a long edge (Wan 480p / 720p presets give Wan's native sizes), or your own width x height — snapped to a multiple of N, because Wan, Flux and SDXL need sizes divisible by 8 / 16 / 64.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `preset` | choice: `1:1 square`, `16:9 landscape`, `9:16 portrait`, `4:3 landscape`, `3:4 portrait`, `21:9 ultrawide`, `2.39:1 cinema`, `Wan 480p land`, `Wan 480p port`, `Wan 720p land`, `Wan 720p port`, `Custom (width x height)` | default `"16:9 landscape"` | — |
| `base` | `INT` | default `1024`, range 64…8192 | Long edge target (ignored for Wan presets and Custom). |
| `multiple` | `INT` | default `64`, range 1…256 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `width` | `INT` | default `1024`, range 8…16384 | Custom: width before snapping. |
| `height` | `INT` | default `1024`, range 8…16384 | Custom: height before snapping. |
| `direction` | choice: `down`, `nearest`, `up` | default `"down"` | Snap to the multiple below, the nearest one, or the one above. |

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

**Shown in the menu as:** Batch Range (C2C)

Pick frames from an IMAGE batch: a start:end:step range (negative values count from the end, step=2 keeps every second frame), a split at a frame index or a fraction, or a single frame (first, middle, last or by index).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |
| `start` | `INT` | default `0`, range -100000…100000 | Range mode: first frame. Negative counts from the end. |
| `end` | `INT` | default `-1`, range -100000…100000 | Range mode: exclusive end. -1 = end of batch. |
| `step` | `INT` | default `1`, range 1…1024 | Range mode: keep every Nth frame. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mode` | choice: `range`, `split at index`, `split at ratio`, `first frame`, `middle frame`, `last frame`, `frame at index` | default `"range"` | What to pick: a start:end:step range, a split (at a frame index or a fraction), or one frame (first, middle, last or by index). |
| `split_index` | `INT` | default `1`, range 0…100000 | `split at index`: frames before this index go to `images`. |
| `split_ratio` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | `split at ratio`: this fraction of the batch goes to `images`. |
| `frame_index` | `INT` | default `0`, range 0…999999 | `frame at index`: 0-based, clamped to the last frame. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | The selected frames (the first part when splitting). |
| 1 | `frame_count` | `INT` | How many frames `images` holds. |
| 2 | `remainder` | `IMAGE` | Split modes: the frames after the split point. Empty in the other modes. |
| 3 | `remainder_count` | `INT` | How many frames `remainder` holds. |
| 4 | `total_frames` | `INT` | How many frames came in. |
| 5 | `is_video` | `BOOLEAN` | True when more than one frame came in. |


### ImageStatsProbeMEC

**Shown in the menu as:** Probe (C2C)

Pass-through probe for debugging: wire an image, a mask and/or a latent and read their statistics — black or blown-out frames, how much of each frame a mask covers, NaN / Inf or saturated latents. Connect at least one of `images`, `mask`, or `latent`.


**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |
| `mask` | `MASK` |  | — |
| `latent` | `LATENT` |  | ComfyUI LATENT dict (must contain `samples`). |
| `threshold` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Mask: a pixel counts as covered above this value. |
| `fail_on_corrupt` | `BOOLEAN` | default `False` | Latent: stop the run when it holds NaN or Inf. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | The image, unchanged. |
| 1 | `report` | `STRING` | One line per wired input (image stats, mask coverage, latent verdict). |
| 2 | `mean` | `FLOAT` | Image mean (0 when no image is wired). |
| 3 | `std` | `FLOAT` | Image standard deviation. |
| 4 | `bright_pct` | `FLOAT` | Percent of image pixels brighter than 0.95. |
| 5 | `mask` | `MASK` | The mask, unchanged. |
| 6 | `coverage_mean_pct` | `FLOAT` | Mean per-frame mask coverage in percent (pixels above threshold). |
| 7 | `coverage_min_pct` | `FLOAT` | Lowest per-frame coverage in percent. |
| 8 | `coverage_max_pct` | `FLOAT` | Highest per-frame coverage in percent. |
| 9 | `latent` | `LATENT` | The latent, unchanged. |
| 10 | `info_json` | `STRING` | Latent shape, dtype, per-channel stats, NaN / Inf counts and verdict as JSON. |
| 11 | `verdict` | `STRING` | Latent verdict: healthy / low_contrast / saturated / corrupt. |
| 12 | `nan_count` | `INT` | NaN elements in the latent. |
| 13 | `inf_count` | `INT` | Inf elements in the latent. |


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


### SmartImageCropMEC

**Shown in the menu as:** Smart Image Crop

Crop what the mask points at, at the aspect it will be generated in, scaled to a size the VAE accepts.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `mask` | `MASK` |  | — |
| `resolution_mode` | choice: `Automatic`, `Manual` | default `"Automatic"` | — |
| `max_resolution` | `INT` | default `2048`, range 256…16384, step 64 | Upper limit on the longest side. A crop larger than this is scaled DOWN to fit. |
| `min_resolution` | `INT` | default `768`, range 64…2048, step 64 | Lower limit on the shortest side. A crop smaller than this is scaled UP, which is the whole point of cropping - a 200px face gets the model's full attention. |
| `manual_width` | `INT` | default `1024`, range 64…8192, step 8 | — |
| `manual_height` | `INT` | default `1024`, range 64…8192, step 8 | — |
| `mask_grow_pixels` | `INT` | default `32`, range -1024…1024, step 8 | Grow (or shrink) the mask before measuring the crop, so the model sees context around the region rather than a tight cut-out. |
| `patch_mask_holes` | `BOOLEAN` | default `True` | Close gaps and fill enclosed holes. A subject mask with holes regenerates those holes as background - the hole-in-the-face result. |
| `no_mask_mode` | choice: `Bypass`, `Resize Full Image`, `Crop Full Image` | default `"Bypass"` | What to do when the mask is empty. Bypass passes the image through untouched, which is the safe default for a batch where only some frames have a mask. |
| `force_divisibility` | choice: `8`, `16`, `32`, `64`, `112`, `128`, `256` | default `128` | Snap output dimensions to this multiple. A VAE silently pads anything else, which shifts the crop by a few pixels and breaks the stitch. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `crop_image` | `IMAGE` | — |
| 1 | `crop_mask` | `MASK` | — |
| 2 | `stitcher` | `STITCHER` | — |
| 3 | `preview_overlay` | `IMAGE` | — |
| 4 | `info` | `STRING` | — |


### SmartImageStitcherMEC

**Shown in the menu as:** Smart Image Stitch

Blend the processed crop back into the original frame.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `original_image` | `IMAGE` |  | — |
| `processed_image` | `IMAGE` |  | — |
| `stitcher` | `STITCHER` |  | — |
| `feather_pixels` | `INT` | default `32`, range 0…256, step 1 | — |
| `blend_mode` | choice: `Box Feather`, `Mask Feather`, `Hard Paste` | default `"Box Feather"` | Box Feather softens the crop's rectangular edge. Mask Feather blends along the mask itself, which keeps untouched detail outside the subject. Hard Paste shows the seam and is for checking alignment. |
| `resize_full_image_output` | choice: `Restore Original Size`, `Keep Resized Image` | default `"Restore Original Size"` | Only applies when the crop node ran in 'Resize Full Image' mode. |
| `enable_color_match` | `BOOLEAN` | default `False` | Pull the crop's exposure and cast toward the plate before blending. A feathered seam turns a small mismatch into a visible halo instead of hiding it. |
| `color_match_amount` | `FLOAT` | default `0.35`, range 0.0…1.0, step 0.01 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `info` | `STRING` | — |


---

## C2C/Legacy


### BBoxCrop

**Shown in the menu as:** BBox Crop (MEC) (legacy)

Deprecated — kept so old workflows load. Closest current node: SmartImageCropMEC.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `bbox` | `BBOX` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `cropped_image` | `IMAGE` | — |
| 1 | `cropped_mask` | `MASK` | — |
| 2 | `bbox` | `BBOX` | — |


### BBoxFromMask

**Shown in the menu as:** BBox From Mask (MEC) (legacy)

Deprecated — kept so old workflows load. No unified successor.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | — |
| `padding` | `INT` | default `0`, range 0…512, step 1 | Uniform padding around the detected bbox |
| `padding_x` | `INT` | default `0`, range 0…512, step 1 | Extra horizontal padding (added to both sides) |
| `padding_y` | `INT` | default `0`, range 0…512, step 1 | Extra vertical padding (added to both sides) |
| `threshold` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `bbox` | `BBOX` | — |
| 1 | `x` | `INT` | — |
| 2 | `y` | `INT` | — |
| 3 | `width` | `INT` | — |
| 4 | `height` | `INT` | — |
| 5 | `bbox_str` | `STRING` | — |


### BBoxPad

**Shown in the menu as:** BBox Pad (MEC) (legacy)

Deprecated — kept so old workflows load. No unified successor.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `bbox` | `BBOX` |  | — |
| `pad_left` | `INT` | default `0`, range -4096…4096 | — |
| `pad_right` | `INT` | default `0`, range -4096…4096 | — |
| `pad_top` | `INT` | default `0`, range -4096…4096 | — |
| `pad_bottom` | `INT` | default `0`, range -4096…4096 | — |
| `image_width` | `INT` | default `512`, range 1…16384 | — |
| `image_height` | `INT` | default `512`, range 1…16384 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `bbox` | `BBOX` | — |
| 1 | `bbox_str` | `STRING` | — |


### BBoxToMask

**Shown in the menu as:** BBox To Mask (MEC) (legacy)

Deprecated — kept so old workflows load. No unified successor.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `bbox` | `BBOX` |  | — |
| `image_width` | `INT` | default `512`, range 1…16384 | — |
| `image_height` | `INT` | default `512`, range 1…16384 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `reference_image` | `IMAGE` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `MASK` | `MASK` | — |


### MaskCompositeAdvanced

**Shown in the menu as:** Mask Composite Advanced (MEC) (legacy)

Deprecated — kept so old workflows load. Closest current node: core MaskComposite (fewer operations).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask_a` | `MASK` |  | — |
| `mask_b` | `MASK` |  | — |
| `operation` | choice: `union`, `intersect`, `subtract`, `xor`, `blend`, `min`, `max`, `difference` | default `"union"` | — |
| `blend_factor` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Blend ratio (only for 'blend' mode). 0=all A, 1=all B |
| `invert_a` | `BOOLEAN` | default `False` | — |
| `invert_b` | `BOOLEAN` | default `False` | — |
| `threshold` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | Binarize output (0 = keep soft) |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `MASK` | `MASK` | — |


### MaskMath

**Shown in the menu as:** Mask Math (MEC) (legacy)

Deprecated — kept so old workflows load. No unified successor.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mask` | `MASK` |  | — |
| `operation` | choice: `add_scalar`, `multiply_scalar`, `power`, `invert`, `clamp`, `remap_range`, `quantize`, `threshold_hysteresis`, `gamma`, `contrast`, `abs_diff_from_value` | default `"invert"` | — |
| `value_a` | `FLOAT` | default `0.0`, range -100.0…100.0, step 0.01 | Primary parameter (meaning depends on operation) |
| `value_b` | `FLOAT` | default `1.0`, range -100.0…100.0, step 0.01 | Secondary parameter (meaning depends on operation) |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `MASK` | `MASK` | — |


### MaskPreviewOverlay

**Shown in the menu as:** Mask Preview Overlay (MEC) (legacy)

Deprecated — kept so old workflows load. Closest current node: MaskOpsMEC preview output.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `mask` | `MASK` |  | — |
| `display_mode` | choice: `overlay`, `mask_only`, `side_by_side`, `checkerboard`, `edge_highlight` | default `"overlay"` | — |
| `overlay_color_r` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | — |
| `overlay_color_g` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | — |
| `overlay_color_b` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | — |
| `opacity` | `FLOAT` | default `0.4`, range 0.0…1.0, step 0.01 | — |
| `edge_width` | `INT` | default `2`, range 0…20, step 1 | Edge contour width in pixels |
| `show_bbox` | `BOOLEAN` | default `False` | Draw bounding box of mask region |
| `bbox_color_r` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | — |
| `bbox_color_g` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | — |
| `bbox_color_b` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `bbox` | `BBOX` |  | External bbox to draw (overrides auto-detect) |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `preview` | `IMAGE` | — |


---

## C2C/Magnific


### MagnificCreationPicker

**Shown in the menu as:** OmniScale Creation Picker




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `project` | choice: `— sign in (Magnific menu) and press R to refresh —` |  | — |
| `folder` | choice: `(project root)` | default `"(project root)"` | Folder inside the selected project; the list narrows to that project. |
| `content_type` | choice: `image`, `video`, `audio` | default `"image"` | Media type to browse; only the matching output is populated. |
| `selected` | `STRING` | default `""` | Set by clicking a thumbnail in the grid. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `video` | `VIDEO` | — |
| 2 | `audio` | `AUDIO` | — |
| 3 | `creation_identifier` | `STRING` | — |
| 4 | `metadata` | `STRING` | — |


### MagnificGenerateImage

**Shown in the menu as:** OmniScale Generate Image




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `prompt` | `STRING` | default `""` | — |
| `model` | choice: `auto` | default `"auto"` | — |
| `aspect_ratio` | choice: `auto`, `1:1`, `16:9`, `9:16`, `2:3`, `3:4`, `4:5`, `5:4`, `3:2`, `4:3`, `21:9`, `2:1`, … | default `"auto"` | — |
| `count` | `INT` | default `1`, range 1…8 | — |
| `seed` | `INT` | default `0`, range 0…4294967295 | Sent to Magnific. Same seed and settings reproduce the result on models that honor it; a batch uses seed, seed+1, … per image. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `folder` | `MAGNIFIC_FOLDER` |  | Optional — from a Magnific Save To node. Not connected → your Personal project. |
| `reference_image` | `IMAGE` |  | Optional reference picture. A batch counts one reference per image. Use the other reference_image sockets for pictures of different sizes; reference_type applies to all of them. |
| `reference_image_2` | `IMAGE` |  | Another reference picture (or batch), same reference_type as the first. |
| `reference_image_3` | `IMAGE` |  | Another reference picture (or batch), same reference_type as the first. |
| `reference_image_4` | `IMAGE` |  | Another reference picture (or batch), same reference_type as the first. |
| `reference_type` | choice: `image`, `style` | default `"image"` | — |
| `resolution` | choice: `auto` | default `"auto"` | auto = model default; narrows to the selected model. |
| `library_references` | `MAGNIFIC_LIBRARY_REFS` |  | Optional — from a Magnific Library Reference node. Attaches Library characters, styles, elements or locations; their @name tokens are added to the prompt when missing. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | — |
| 1 | `creation_identifiers` | `STRING` | — |
| 2 | `metadata` | `STRING` | — |


### MagnificGenerateMusic

**Shown in the menu as:** OmniScale Generate Music




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `prompt` | `STRING` | default `""` | — |
| `model` | choice: `Google Lyria`, `Lyria 3 Short`, `Lyria 3 Long`, `ElevenLabs Music`, `ElevenLabs Music v2`, `google-lyria`, `google-lyria-3`, `google-lyria-3-pro`, `elevenlabs-music-generation`, `elevenlabs-music-generation-v2` | default `"Google Lyria"` | — |
| `duration_seconds` | `INT` | default `30`, range 10…300 | Only Lyria 3 Long and ElevenLabs honor it; the rest are fixed at 30s. |
| `instrumental` | `BOOLEAN` | default `False` | — |
| `seed` | `INT` | default `0`, range 0…4294967295 | Music generation takes no seed: changing it only forces a new run. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `folder` | `MAGNIFIC_FOLDER` |  | Optional — from a Magnific Save To node. Not connected → your Personal project. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `audio` | `AUDIO` | — |
| 1 | `creation_identifier` | `STRING` | — |
| 2 | `metadata` | `STRING` | — |


### MagnificGenerateVideo

**Shown in the menu as:** OmniScale Generate Video




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `prompt` | `STRING` | default `""` | — |
| `model` | choice: `auto` | default `"auto"` | — |
| `duration` | choice: `5`, `10` | default `"5"` | Seconds; narrows to the selected model. |
| `resolution` | choice: `auto` | default `"auto"` | auto = model default; narrows to the selected model. |
| `aspect_ratio` | choice: `auto`, `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9` | default `"auto"` | — |
| `sound_effects` | `BOOLEAN` | default `False` | Only models with native audio honor it. |
| `seed` | `INT` | default `0`, range 0…4294967295 | Sent to Magnific. Same seed and settings reproduce the result on models that honor it (e.g. Seedance). |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `folder` | `MAGNIFIC_FOLDER` |  | Optional — from a Magnific Save To node. Not connected → your Personal project. |
| `start_frame` | `IMAGE` |  | Image-to-video start keyframe. |
| `end_frame` | `IMAGE` |  | End keyframe (models that support it). |
| `reference_image` | `IMAGE` |  | — |
| `reference_video` | `VIDEO` |  | Video reference (models that support it, e.g. Seedance 2.0). Not combinable with a start/end frame. |
| `library_references` | `MAGNIFIC_LIBRARY_REFS` |  | Optional — from a Magnific Library Reference node. Attaches Library characters, styles, elements or locations; their @name tokens are added to the prompt when missing. Video takes characters, elements and styles (no locations). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `video` | `VIDEO` | — |
| 1 | `creation_identifier` | `STRING` | — |
| 2 | `metadata` | `STRING` | — |


### MagnificLibraryReference

**Shown in the menu as:** OmniScale Library Reference




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `catalog` | choice: `My Library`, `Magnific` | default `"My Library"` | My Library: your own and shared assets. Magnific: the public catalog. |
| `type` | choice: `All`, `Characters`, `Styles`, `Elements`, `Locations` | default `"All"` | Narrows the asset list to one kind. |
| `asset` | choice: `— sign in (Magnific menu) and press R to refresh —` |  | @name (Type · Origin), grouped by type. Only ready assets are listed; press Refresh library or R. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `chain` | `MAGNIFIC_LIBRARY_REFS` |  | Another Library Reference node, to attach several assets to one generation. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `library_references` | `MAGNIFIC_LIBRARY_REFS` | — |


### MagnificMetadata

**Shown in the menu as:** OmniScale Metadata (unpack)




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `metadata` | `STRING` |  | The metadata output of a Magnific generation or edit node. |
| `index` | `INT` | default `0`, range 0…63 | Which result of a batch to read; clamped to the last one. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `model` | `STRING` | — |
| 1 | `prompt` | `STRING` | — |
| 2 | `seed` | `INT` | — |
| 3 | `creation_identifier` | `STRING` | — |
| 4 | `tool` | `STRING` | — |
| 5 | `url` | `STRING` | — |
| 6 | `json` | `STRING` | — |


### MagnificRemoveBackground

**Shown in the menu as:** OmniScale Remove Background




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `folder` | `MAGNIFIC_FOLDER` |  | Optional — from a Magnific Save To node. Not connected → your Personal project. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `mask` | `MASK` | — |
| 2 | `creation_identifier` | `STRING` | — |
| 3 | `metadata` | `STRING` | — |


### MagnificRetouch

**Shown in the menu as:** OmniScale Retouch (inpaint)




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `mask` | `MASK` |  | 1.0/white = area to change. Must be the same pixel size as the image. |
| `mode` | choice: `replace`, `erase` | default `"replace"` | — |
| `prompt` | `STRING` | default `""` | Replace mode only: what the masked area should become. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `folder` | `MAGNIFIC_FOLDER` |  | Optional — from a Magnific Save To node. Not connected → your Personal project. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `creation_identifier` | `STRING` | — |
| 2 | `metadata` | `STRING` | — |


### MagnificSaveTo

**Shown in the menu as:** OmniScale Save To (project/folder)




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `project` | choice: `— sign in (Magnific menu) and press R to refresh —` |  | — |
| `folder` | choice: `(project root)` | default `"(project root)"` | Folder inside the selected project; the list narrows to that project. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `folder` | `MAGNIFIC_FOLDER` | — |


### MagnificSkinEnhancer

**Shown in the menu as:** OmniScale Skin Enhancer




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `version` | choice: `faithful`, `creative`, `flexible` | default `"faithful"` | — |
| `sharpen` | `INT` | default `0`, range 0…100 | — |
| `smart_grain` | `INT` | default `0`, range 0…100 | — |
| `skin_detail` | `INT` | default `0`, range 0…100 | Faithful version only. |
| `optimized_for` | choice: `default`, `enhance_skin`, `enhance_everything`, `improve_lighting`, `transform_to_real`, `no_make_up` | default `"default"` | Flexible version only. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `folder` | `MAGNIFIC_FOLDER` |  | Optional — from a Magnific Save To node. Not connected → your Personal project. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `creation_identifier` | `STRING` | — |
| 2 | `metadata` | `STRING` | — |


### MagnificStockPicker

**Shown in the menu as:** OmniScale Stock Picker




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `query` | `STRING` | default `""` | — |
| `content_type` | choice: `any`, `photo`, `vector`, `illustration`, `psd`, `design` | default `"photo"` | — |
| `license` | choice: `any`, `free`, `premium` | default `"any"` | — |
| `orientation` | choice: `any`, `portrait`, `landscape`, `square`, `panoramic` | default `"any"` | — |
| `ai_generated` | choice: `any`, `only`, `excluded` | default `"any"` | — |
| `selected` | `STRING` | default `""` | Set by clicking a thumbnail in the results grid. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `folder` | `MAGNIFIC_FOLDER` |  | Optional: also save the stock item to this Magnific project/folder. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `info` | `STRING` | — |


### MagnificStockSearch

**Shown in the menu as:** OmniScale Stock Search




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `query` | `STRING` | default `""` | — |
| `content_type` | choice: `any`, `photo`, `vector`, `illustration`, `psd`, `design` | default `"photo"` | — |
| `license` | choice: `any`, `free`, `premium` | default `"any"` | — |
| `orientation` | choice: `any`, `portrait`, `landscape`, `square`, `panoramic` | default `"any"` | — |
| `ai_generated` | choice: `any`, `only`, `excluded` | default `"any"` | — |
| `selection` | choice: `index`, `random` | default `"index"` | index = take result_index; random = pick between range_from and range_to. |
| `result_index` | `INT` | default `0`, range 0…999 | Used when selection = index. |
| `range_from` | `INT` | default `0`, range 0…999 | Random range start (0 = first result). |
| `range_to` | `INT` | default `9`, range 0…999 | Random range end, inclusive — first 10 results by default. |
| `seed` | `INT` | default `0`, range 0…4294967295 | Drives the random pick: randomize for a different item per run, fix it to reproduce one. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `folder` | `MAGNIFIC_FOLDER` |  | Optional: also save the stock item to this Magnific project/folder. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `info` | `STRING` | — |


### MagnificUpscaleImage

**Shown in the menu as:** OmniScale Upscale Image




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `scale` | choice: `2x`, `4x`, `8x`, `16x` | default `"2x"` | — |
| `precision` | choice: `creative`, `precision` | default `"creative"` | — |
| `presets` | choice: `subtle`, `vivid`, `wild` | default `"subtle"` | — |
| `engine` | choice: `Automatic`, `Illusio`, `Sharpy`, `Sparkle`, `automatic`, `magnific_illusio`, `magnific_sharpy`, `magnific_sparkle` | default `"Automatic"` | — |
| `mode` | choice: `Automatic`, `Magnific v1 (high HDR)`, `Magnific v2 (sublime)`, `Magnific v2 (photo)`, `Magnific v2 (photo denoiser)`, `default`, `ultra`, `ultra-sublime`, `ultra-photo`, `ultra-denoiser` | default `"Automatic"` | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `folder` | `MAGNIFIC_FOLDER` |  | Optional — from a Magnific Save To node. Not connected → your Personal project. |
| `prompt` | `STRING` | default `""` | — |
| `creativity` | `INT` | default `0`, range -10…10 | — |
| `hdr` | `INT` | default `0`, range -10…10 | — |
| `resemblance` | `INT` | default `0`, range -10…10 | — |
| `fractality` | `INT` | default `0`, range -10…10 | — |
| `sharpness` | `INT` | default `0`, range 0…100 | — |
| `grain` | `INT` | default `0`, range 0…100 | — |
| `ultra_detail` | `INT` | default `0`, range 0…100 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `creation_identifier` | `STRING` | — |
| 2 | `metadata` | `STRING` | — |


### MagnificUpscaleVideo

**Shown in the menu as:** OmniScale Upscale Video




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `video` | `VIDEO` |  | — |
| `mode` | choice: `magnific`, `magnific_precision`, `topaz` | default `"magnific"` | — |
| `target_resolution` | choice: `1280`, `1920`, `2560`, `3840` | default `"1920"` | Output width (Magnific/Precision modes). |
| `upscale_factor` | choice: `2`, `4` | default `"2"` | Scale multiplier (Topaz mode). |
| `preset` | choice: `realistic`, `animationAnd3D`, `artistic`, `custom` | default `"realistic"` | Creative bundle for the magnific mode; custom uses the sliders below. |
| `fps_boost` | `BOOLEAN` | default `False` | — |
| `preview` | `BOOLEAN` | default `False` | 12-frame preview run instead of the full clip. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `folder` | `MAGNIFIC_FOLDER` |  | Optional — from a Magnific Save To node. Not connected → your Personal project. |
| `creativity` | `INT` | default `0`, range 0…100 | — |
| `sharpen` | `INT` | default `0`, range 0…100 | — |
| `smart_grain` | `INT` | default `0`, range 0…100 | — |
| `flavor` | choice: `natural`, `vivid` | default `"natural"` | — |
| `premium_quality` | `BOOLEAN` | default `True` | — |
| `turbo` | `BOOLEAN` | default `False` | — |
| `strength` | `INT` | default `50`, range 0…100 | Precision mode only. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `video` | `VIDEO` | — |
| 1 | `creation_identifier` | `STRING` | — |
| 2 | `metadata` | `STRING` | — |


### MagnificVoiceover

**Shown in the menu as:** OmniScale Voiceover (TTS)




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `text` | `STRING` | default `""` | — |
| `voice` | choice: `— sign in (Magnific menu) and press R to refresh —` |  | — |
| `speed` | `FLOAT` | default `1.0`, range 0.7…1.2, step 0.05 | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `folder` | `MAGNIFIC_FOLDER` |  | Optional — from a Magnific Save To node. Not connected → your Personal project. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `audio` | `AUDIO` | — |
| 1 | `creation_identifier` | `STRING` | — |
| 2 | `metadata` | `STRING` | — |


---

## C2C/Mask


### ImageMaskEditorC2C

**Shown in the menu as:** Image Mask Editor (C2C)

Paint a mask over the image that actually reaches this node. Open the editor to draw; grow, feather, threshold and invert are non-destructive node parameters. Docs: docs/image-mask-editor.md in this pack.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Input batch (B,H,W,3). Drives B/H/W. |
| `editor_id` | `STRING` | default `""` | Auto-set by the editor UI. Don't edit by hand. |
| `frame_mode` | choice: `per_frame`, `shared` | default `"per_frame"` | per_frame: each batch index uses stored frame i; shared: stored frame 0 applies to all. |
| `grow` | `INT` | default `0`, range -256…256, step 1 | Grow (positive) or shrink (negative) the mask by this many pixels. Any non-zero value makes the edge hard first (every partly covered pixel counts); feather softens it again afterwards. |
| `feather` | `FLOAT` | default `0.0`, range 0.0…256.0, step 1.0 | Soften the mask edge, in whole pixels (0 = off). Gaussian with sigma = 0.33 x this value, the same feather the Smart Crop/Stitch nodes use. |
| `threshold` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | Binarize after feather. 0 = keep soft mask. |
| `invert` | `BOOLEAN` | default `False` | Invert mask after threshold. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `input_mask` | `MASK` |  | Optional mask to combine with the painted mask. |
| `combine` | choice: `replace`, `add`, `subtract`, `intersect` | default `"replace"` | How painted mask combines with input_mask. Without input_mask only replace applies. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | The final mask [B,H,W] in 0-1: painted frames, combined with input_mask, then grow, feather, threshold and invert. |
| 1 | `preview` | `IMAGE` | The input image with the mask tinted over it. |
| 2 | `info` | `STRING` | Frame count, stored masks, mode and notes (resized frames, frame recording skipped). |


---

## C2C/ModelAnalysis


### VAEBlockInspectorMEC

**Shown in the menu as:** VAE Inspect

Inspect a VAE's per-block weight statistics and outlier tensors, or compare two VAEs block by block (cosine similarity) — for checking merges and broken checkpoints.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vae` | `VAE` |  | VAE to inspect (compare mode: the first VAE). |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `anomaly_threshold` | `FLOAT` | default `5.0`, range 1.5…50.0, step 0.5 | Inspect: tensors whose abs_mean exceeds this multiple of the cohort median are flagged as magnitude outliers. Lower => more sensitive (more flags). |
| `mode` | choice: `inspect`, `compare` | default `"inspect"` | inspect = weight statistics of one VAE; compare = cosine similarity to vae_b. |
| `vae_b` | `VAE` |  | Compare: the second VAE. |
| `include_per_tensor` | `BOOLEAN` | default `False` | Compare: include per-tensor cosine entries in the JSON report (verbose). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `report_json` | `STRING` | Inspect: per-block weight statistics plus outlier details. Compare: per-block cosine, missing keys and optional per-tensor entries. JSON. |
| 1 | `outlier_tensor_names` | `STRING` | Inspect: JSON list of tensor names flagged as outliers. |
| 2 | `anomaly_score` | `FLOAT` | Inspect: aggregate anomaly score in [0, 1] (higher means more outliers detected). |
| 3 | `global_cosine` | `FLOAT` | Compare: global cosine similarity across all common tensors. |
| 4 | `most_divergent_blocks` | `STRING` | Compare: JSON list of the 10 most divergent blocks (lowest cosine first). |


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

## C2C/Tiling


### IdentityLockMEC

**Shown in the menu as:** Identity Lock — refine without moving

Refine every pixel without letting anything move. Keeps the plate's low frequencies - where composition and face geometry live - and takes only the model's high frequencies. The face cannot move, because the only thing taken from the model is the band that does not encode where anything is.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `original` | `IMAGE` |  | The plate, at the SAME size as the refined image. If the refinement upscaled, upscale this to match first. |
| `refined` | `IMAGE` |  | What came back from the model. |
| `radius` | `FLOAT` | default `8.0`, range 0.0…128.0, step 0.5 | The split frequency. SMALLER locks harder and lets less refinement through; LARGER passes more detail and more movement with it. If a face is still moving, go SMALLER. |
| `strength` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | 0 = the raw refinement, 1 = fully locked. Set it to 0 WITH a protect mask to lock only the face and leave the background completely free. |
| `corrections` | `INT` | default `2`, range 0…8 | Extra passes that push the result's structure back onto the plate's. A Gaussian is not a brick-wall filter, so one pass alone holds only about 70% of a face shift; two hold about 94%, at a 2% cost in detail. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `protect` | `MASK` |  | Where the lock is at FULL strength regardless of the strength above - a face. Everywhere else gets whatever strength says. |
| `want_drift_map` | `BOOLEAN` | default `False` | Also output a map of where structure moved. Bright means it moved. Use it to find what the drift number is telling you about. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `drift_map` | `MASK` | — |
| 2 | `drift` | `FLOAT` | — |
| 3 | `info` | `STRING` | — |


### TileMergeMEC

**Shown in the menu as:** Tile Merge

Blend the tiles back into one clip, in float32, with no seam.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `tiles` | `IMAGE` |  | — |
| `tile_plan` | `TILE_PLAN` |  | — |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `scale` | `FLOAT` | default `0.0`, range 0.0…8.0, step 0.05 | How much the refiner enlarged each tile. 0 means work it out from the tiles themselves, which is right when every tile scaled by the same clean factor. Set it explicitly if the ratio is not a round number. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | — |
| 1 | `info` | `STRING` | — |


### TilePlanMEC

**Shown in the menu as:** Tile Plan

Decide the tiles once for a whole clip. Reusing one plan for every frame is what stops seams crawling.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Any frame at the size you will be tiling. Only its dimensions are read. |
| `mode` | choice: `tile size`, `tile count` | default `"tile size"` | — |
| `tile_size` | `INT` | default `1024`, range 128…8192, step 64 | A MAXIMUM, not a target. Tiles are equal-sized and must cover the frame exactly, so the size chosen is the largest at or below this that does. The report says what it picked. |
| `rows` | `INT` | default `2`, range 1…16 | — |
| `cols` | `INT` | default `2`, range 1…16 | — |
| `overlap` | `INT` | default `96`, range 0…512, step 8 | How far neighbouring tiles share pixels. The blend has to hide any difference between two tiles inside this band, so too narrow shows a seam. 64-128px is usual. |
| `blend` | choice: `cosine`, `linear`, `none` | default `"cosine"` | cosine is the right answer: the fade-in and fade-out sum to exactly 1 across the overlap, so there is no seam and no brightness dip. 'none' is for checking where the tiles are. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `tile_plan` | `TILE_PLAN` | — |
| 1 | `info` | `STRING` | — |


### TileSplitMEC

**Shown in the menu as:** Tile Split

Split a clip into overlapping tiles, emitted as a batch so any refiner can process them. Wire the result through whatever you like, then Tile Merge.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | — |
| `tile_plan` | `TILE_PLAN` |  | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `tiles` | `IMAGE` | — |
| 1 | `tile_count` | `INT` | — |
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


### FolderVersionControl

**Shown in the menu as:** Folder Version Control (check / set)

Check or set the version counter for a label/date folder. 'check' reports how many vNNN folders exist and what the next one will be; 'set' creates placeholders up to a number so the next run continues after it. To truly reset a label, delete its date folder from disk.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `action` | choice: `check`, `set` | default `"check"` | check -> report how many versions exist and what comes next. Reads only; touches nothing.  set   -> create empty placeholder directories up to `value`, so the next run produces value+1. Use it to skip ahead or to reserve a range. |
| `label` | `STRING` | default `"default"` | Folder name to inspect, under the output directory. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `value` | `INT` | default `1`, range 1…999999 | 'set' only: create placeholders up to this version number. Ignored when checking. |
| `trigger` | `*` |  | Optional any-type trigger. Wire any upstream output here to control when this runs. |
| `prefix` | `STRING` | default `"v"` | Version prefix. Must match the incrementer's. |
| `padding` | `INT` | default `3`, range 1…10 | Zero-pad width. Must match the incrementer's. |
| `base_path` | `STRING` | default `""` | Override base directory. Empty -> ComfyUI output dir. |
| `date_format` | choice: `MM-DD-YYYY`, `DD-MM-YYYY`, `YYYY-MM-DD` | default `"MM-DD-YYYY"` | Must match the incrementer's. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `status` | `STRING` | Human-readable description of what was found or done. |
| 1 | `current_version` | `INT` | Highest version that exists (0 when there are none yet). |
| 2 | `next_version` | `INT` | What the next incrementer run will produce. |


### FolderVersionVariant

**Shown in the menu as:** Folder Version Variant (per output)

Give one output its own prefix or suffix while keeping the version the Folder Version Incrementer decided. Put one of these between the incrementer and each save node when a single render writes several outputs - a mov, an exr sequence, a png sequence, a masked mov - so they all file under one version with different names.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `filename_prefix` | `STRING` | default `""` | Connect the Folder Version Incrementer's `filename_prefix` output. Every variant fed from the same incrementer lands under the same version. |
| `suffix` | `STRING` | default `""` | Tag for THIS output, e.g. '_masked', '_exr', '_png'. Leave empty for the one output that needs no tag. |
| `mode` | choice: `filename`, `subfolder`, `folder` | default `"subfolder"` | Where the tag goes.  filename  -> shot_a/v004/shot_a_masked.mov              everything in one folder, told apart by name  subfolder -> shot_a/v004/masked/shot_a.mov              one folder per output INSIDE the version. Use this for exr sequences - thousands of files should not share a directory with anything else.  folder    -> shot_a_masked/v004/shot_a.mov              a separate top-level tree, still on the same version. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `prefix` | `STRING` | default `""` | Tag placed BEFORE the name instead of after. Offered because a prefix faked with a suffix sorts in the wrong order, which is the whole reason to use one. |
| `version_string` | `STRING` | default `""` | Optional. Connect the incrementer's `version_string` so a subfolder tag is placed immediately after the version directory rather than at the end of the path. |
| `extension` | `STRING` | default `""` | Optional extension for the `output_filename` output, e.g. '.mov'. Most save nodes take `filename_prefix` and add their own. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `filename_prefix` | `STRING` | Re-tagged prefix. Wire to the save node's filename_prefix. |
| 1 | `subfolder_path` | `STRING` | Directory part only. |
| 2 | `output_filename` | `STRING` | Full name including the extension, when one was given. |
| 3 | `report` | `STRING` | What was tagged and where, and what stayed shared. |


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

## C2C/Vault


### C2C_VaultEditInput

**Shown in the menu as:** Vault input (edit view)

A vault input in the vault's edit view. Created by the vault editor; it never runs.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `index` | `INT` | default `0`, range 0…9 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `value` | `*` | — |


### C2C_VaultEditOutput

**Shown in the menu as:** Vault output (edit view)

A vault output in the vault's edit view. Created by the vault editor; it never runs.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `value` | `*` |  | — |
| `index` | `INT` | default `0`, range 0…7 | — |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|


### C2C_VaultLocked

**Shown in the menu as:** C2C Vault — Locked (password to run)

Run a password-locked subgraph. The wiring inside travels with the workflow as AES-GCM ciphertext, so a recipient sees an opaque blob instead of your node graph. Lock a canvas selection via right-click → C2C Vault. To run it, type the password in the field on this node and press Enter; the unlock lasts for the ComfyUI session and the password never enters the workflow JSON. SCOPE, honestly: this stops casual inspection and copying. It cannot stop someone who can run Python in this process - the subgraph must be decrypted to execute, so the plaintext exists in memory while it runs. A lock on a door, not a safe.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vault_id` | `STRING` | default `""` | Public vault handle — authenticated in the ciphertext header, never secret. Lets the UI match an unlock session to the right blob without revealing what is inside. |
| `vault_payload` | `STRING` | default `""` | AES-GCM ciphertext of the internal graph. This is what travels in the .json — opaque to anyone without the password. Do not hand-edit. |
| `vault_interface` | `STRING` | default `"{}"` | Clear-text manifest: socket names, types, node count, and any promoted parameters. This is the public API of the vault and it travels in the .json, so it carries a promoted parameter's LABEL and value only - which internal widget it drives is inside the ciphertext, because naming it here would describe the contents. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `input_0` | `*` |  | Positional boundary input 0 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[0] inside the ciphertext. |
| `input_1` | `*` |  | Positional boundary input 1 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[1] inside the ciphertext. |
| `input_2` | `*` |  | Positional boundary input 2 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[2] inside the ciphertext. |
| `input_3` | `*` |  | Positional boundary input 3 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[3] inside the ciphertext. |
| `input_4` | `*` |  | Positional boundary input 4 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[4] inside the ciphertext. |
| `input_5` | `*` |  | Positional boundary input 5 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[5] inside the ciphertext. |
| `input_6` | `*` |  | Positional boundary input 6 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[6] inside the ciphertext. |
| `input_7` | `*` |  | Positional boundary input 7 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[7] inside the ciphertext. |
| `input_8` | `*` |  | Positional boundary input 8 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[8] inside the ciphertext. |
| `input_9` | `*` |  | Positional boundary input 9 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[9] inside the ciphertext. |
| `param_0` | `*` |  | Promoted parameter 0 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_1` | `*` |  | Promoted parameter 1 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_2` | `*` |  | Promoted parameter 2 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_3` | `*` |  | Promoted parameter 3 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_4` | `*` |  | Promoted parameter 4 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_5` | `*` |  | Promoted parameter 5 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_6` | `*` |  | Promoted parameter 6 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_7` | `*` |  | Promoted parameter 7 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `output_0` | `*` | — |
| 1 | `output_1` | `*` | — |
| 2 | `output_2` | `*` | — |
| 3 | `output_3` | `*` | — |
| 4 | `output_4` | `*` | — |
| 5 | `output_5` | `*` | — |
| 6 | `output_6` | `*` | — |
| 7 | `output_7` | `*` | — |


### C2C_VaultSealed

**Shown in the menu as:** C2C Vault — Sealed (runs without password)

Sealed subgraph - RUNS WITHOUT A PASSWORD, so you can hand a workflow to someone and they just use it. The password is only needed to OPEN the vault for editing. Sealed for installs (the default): it runs without a password only on this install and on the installs whose public keys you pasted when you sealed it; each install keeps its own private key in the OS credential store. Portable seal: runs on any install with this pack, so it is obfuscation only. A legacy (v1) seal still runs and is marked; open it and lock it again to upgrade. HONEST SCOPE: a recipient install can always decrypt what it runs, so sealing stops reading and copying your graph out of the .json, not a determined extractor on a recipient machine. Use C2C_VaultLocked when the recipient should need your password to run it at all.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vault_id` | `STRING` | default `""` | Public vault handle — authenticated in the ciphertext header, never secret. Lets the UI match an unlock session to the right blob without revealing what is inside. |
| `vault_payload` | `STRING` | default `""` | AES-GCM ciphertext of the internal graph. This is what travels in the .json — opaque to anyone without the password. Do not hand-edit. |
| `vault_interface` | `STRING` | default `"{}"` | Clear-text manifest: socket names, types, node count, and any promoted parameters. This is the public API of the vault and it travels in the .json, so it carries a promoted parameter's LABEL and value only - which internal widget it drives is inside the ciphertext, because naming it here would describe the contents. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `input_0` | `*` |  | Positional boundary input 0 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[0] inside the ciphertext. |
| `input_1` | `*` |  | Positional boundary input 1 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[1] inside the ciphertext. |
| `input_2` | `*` |  | Positional boundary input 2 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[2] inside the ciphertext. |
| `input_3` | `*` |  | Positional boundary input 3 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[3] inside the ciphertext. |
| `input_4` | `*` |  | Positional boundary input 4 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[4] inside the ciphertext. |
| `input_5` | `*` |  | Positional boundary input 5 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[5] inside the ciphertext. |
| `input_6` | `*` |  | Positional boundary input 6 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[6] inside the ciphertext. |
| `input_7` | `*` |  | Positional boundary input 7 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[7] inside the ciphertext. |
| `input_8` | `*` |  | Positional boundary input 8 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[8] inside the ciphertext. |
| `input_9` | `*` |  | Positional boundary input 9 — the label you see is applied by the vault interface manifest, not by this socket name. Index must match boundary_in[9] inside the ciphertext. |
| `param_0` | `*` |  | Promoted parameter 0 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_1` | `*` |  | Promoted parameter 1 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_2` | `*` |  | Promoted parameter 2 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_3` | `*` |  | Promoted parameter 3 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_4` | `*` |  | Promoted parameter 4 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_5` | `*` |  | Promoted parameter 5 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_6` | `*` |  | Promoted parameter 6 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |
| `param_7` | `*` |  | Promoted parameter 7 — connect this only when a promoted parameter has been converted to an input. The vault decides which internal widget it drives; that mapping lives inside the ciphertext and is never in the workflow JSON. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `output_0` | `*` | — |
| 1 | `output_1` | `*` | — |
| 2 | `output_2` | `*` | — |
| 3 | `output_3` | `*` | — |
| 4 | `output_4` | `*` | — |
| 5 | `output_5` | `*` | — |
| 6 | `output_6` | `*` | — |
| 7 | `output_7` | `*` | — |


---

## C2C/Video


### MaskTrackerMEC

**Shown in the menu as:** Mask Track — Motion/Propagate/Anchor/Consistency/Stabilize

Unified video-mask tracker. Pick a mode and the corresponding engine runs. All modes share the (mask, video) input pair. Heavy work is chunked/vectorized; CPU fallback always available.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mode` | choice: `motion`, `propagate`, `anchor`, `consistency_check`, `stabilize` | default `"motion"` | motion: per-frame motion mask (pixel/flow/bg/hist). propagate: seed mask on one frame, push to all frames. anchor: SDF interpolation between anchor masks. consistency_check: score flicker between consecutive frames. |
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
| `points_json` | `STRING` | default `""` | [propagate sam2_video mode] point prompts |
| `stabilize_temporal_mode` | choice: `none`, `gaussian`, `raft_flow` | default `"none"` | [stabilize] none = integrity report only; gaussian = smooth across frames; raft_flow = blend each frame with the flow-warped previous one. |
| `stabilize_blend` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.05 | [stabilize] mix factor: 1.0 = pure warped-previous, 0.0 = current only. |
| `stabilize_sigma` | `FLOAT` | default `1.0`, range 0.0…8.0, step 0.1 | [stabilize] gaussian sigma (gaussian only). |
| `stabilize_device` | choice: `cuda`, `cpu` | default `"cuda"` | [stabilize] where to run. |
| `stabilize_drop_threshold` | `FLOAT` | default `0.4`, range 0.0…1.0, step 0.05 | [stabilize] area ratio / IoU below this flags the frame. |
| `stabilize_jump_threshold` | `FLOAT` | default `0.15`, range 0.0…1.0, step 0.01 | [stabilize] centroid jump (normalized) above this flags the frame. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `masks` | `MASK` | Per-frame mask batch (B,H,W). |
| 1 | `preview` | `IMAGE` | Preview overlay (propagate) or video passthrough. |
| 2 | `score` | `FLOAT` | Mode-specific scalar: motion intensity / mean confidence / flicker score. |
| 3 | `info_json` | `STRING` | Mode-specific JSON diagnostic payload. |
| 4 | `metric` | `STRING` | Mode/metric label string. |
| 5 | `warning` | `STRING` | [stabilize] frames whose mask dropped or jumped, in words; empty when none (and in the other modes). |


### AVHandlesMEC

**Shown in the menu as:** AV Handles — Add / Trim

Add repeated frames at the head (or tail) to stabilise a video model, then trim exactly those back off afterwards — without retyping the number. In trim mode it uses the count the add recorded, so the two ends cannot drift apart. Audio is moved by the matching duration.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mode` | choice: `add`, `trim`, `fit` | default `"fit"` | fit  — say how many frames you want OUT. The node works out what the model will accept, pads up to it, and remembers the target so the trim lands exactly there. This is the one to use.  add  — put a specific number of frames on, and remember how many.  trim — take them back off, using the remembered target or count. |
| `handle_frames` | `INT` | default `8`, range 0…400, step 1 | ADD: how many frames to put on. With a grid selected, 0 means 'just enough to land on the grid'.  TRIM: ignored unless trim_amount is 'manual'. |
| `trim_amount` | choice: `auto`, `manual` | default `"auto"` | TRIM mode only.  auto   — use the count the add side recorded (or the one wired into `handles`). This is the point of the node: the two ends cannot drift apart.  manual — use handle_frames instead. Only for undoing handles this node did not add.  It is a separate control because 'left at the default' and 'deliberately typed 8' cannot be told apart in one number, and guessing wrong silently trims the wrong amount. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | — |
| `audio` | `AUDIO` |  | — |
| `target_frames` | `INT` | default `0`, range 0…100000, step 1 | FIT mode: how many frames you want OUT. The model's own grid is an implementation detail — ask for 64 and you get 64, whatever the model needs to render to make them.  0 in fit mode means 'use the clip's current length as the target', which pads it to something the model accepts and trims straight back. |
| `model` | choice: `none (any frame count)`, `MiniMax H3 (17n+5)`, `Wan 2.x (4n+1)`, `LTX / LTX-2 (8n+1)`, `Hunyuan Video (4n+1)`, `CogVideoX (8n+1)`, `Mochi (6n+1)`, `custom (step / plus)` | default `"MiniMax H3 (17n+5)"` | FIT, and ADD with padding_mode 'model preset': which model's frame grid to land on. Every video model wants step*n + plus frames and they disagree only on the numbers: H3 is 17n+5, Wan is 4n+1, LTX is 8n+1.  Hand a model a count it does not accept and it does not complain — it reinterprets the clip, and the result comes out at the wrong speed. |
| `custom_step` | `INT` | default `4`, range 1…1000 | Only used when model is 'custom'. |
| `custom_plus` | `INT` | default `1`, range 0…1000 | Only used when model is 'custom'. |
| `side` | choice: `head`, `tail`, `both` | default `"head"` | Which end. Handles go on the head because that is the run a video model uses to settle.  both: handle_frames on the head AND the tail (fit splits its padding across the two ends).  TRIM takes a wired or remembered count off exactly the ends it went on; this setting only matters for a manual count. |
| `padding_mode` | choice: `disabled`, `model preset`, `WAN (4n+1)`, `LTX2 (8n+1)`, `H3 (17n+5)` | default `"model preset"` | ADD: round the total onto a model's frame grid. A model that wants 4n+1 and is handed 4n does not complain — it reinterprets the clip, and the result comes out at the wrong speed.  model preset: the grid of the `model` dropdown (or the custom step / plus). disabled: exactly handle_frames, no rounding. The fixed WAN / LTX2 / H3 entries are kept for workflows saved with them. |
| `manual_fps` | `FLOAT` | default `0.0`, range 0.0…240.0, step 0.001 | 0 measures it from the clip's own audio. Set it when there is no audio to measure against — a guessed 30 on 24fps material puts the silence 25% out. |
| `handles` | `HANDLES` |  | Wire the add node's `handles` output here for an exact, explicit count. Takes priority over the remembered one, and works across separate workflows where the memory cannot. |
| `memo_key` | `STRING` | default `""` | Only needed when two independent clips are being handled in one workflow — give each its own name so their counts do not overwrite each other. Empty is right for one clip. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `images` | `IMAGE` | Frames, with handles added or removed. |
| 1 | `audio` | `AUDIO` | Audio moved by the same duration, so it stays in step. |
| 2 | `total_frames` | `INT` | How many frames came out. |
| 3 | `handles` | `HANDLES` | The handle count, to wire into the trim node. Exact, and it does not depend on the memory — use it when the add and the trim are in different workflows. |
| 4 | `info` | `STRING` | What happened, including WHERE the trim count came from. |


### LoadImagesPathC2C

**Shown in the menu as:** Load Images Path (C2C)




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `directory` | `STRING` |  | Directory of sequential images. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image_load_cap` | `INT` | default `0`, range 0…1000000 | Max images; 0 = no cap. |
| `skip_first_images` | `INT` | default `0`, range 0…1000000 | Skip this many images first. |
| `select_every_nth` | `INT` | default `1`, range 1…100000 | Keep every Nth image. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `IMAGE` | `IMAGE` | — |
| 1 | `MASK` | `MASK` | — |
| 2 | `frame_count` | `INT` | — |
| 3 | `video` | `C2C_VIDEO` | — |


### LoadVideoC2C

**Shown in the menu as:** Load Video (C2C)




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `video` | choice: `4098429-uhd_4096_2160_25fps.mp4`, `C1799.MP4 Comp 1.mp4`, `Jibaro  Love, Death and Robots - Gamerstrix (1080p, h264).mp4`, `Wanimate_00002_p81-audio_hfclk_1783619197.mp4`, `bedroom.mp4`, `probe.mp4`, `wd_test_prores.mov` |  | Video file from the ComfyUI input folder. |
| `force_rate` | `FLOAT` | default `0`, range 0…240, step 1 | Output FPS; 0 keeps source. |
| `custom_width` | `INT` | default `0`, range 0…16384 | Output width; 0 keeps source or derives from height. |
| `custom_height` | `INT` | default `0`, range 0…16384 | Output height; 0 keeps source or derives from width. |
| `frame_load_cap` | `INT` | default `0`, range 0…1000000 | Max frames; 0 = no cap. |
| `skip_first_frames` | `INT` | default `0`, range 0…1000000 | Skip this many frames first. |
| `select_every_nth` | `INT` | default `1`, range 1…100000 | Keep every Nth frame. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vae` | `VAE` |  | Encode to LATENT instead of IMAGE. |
| `format` | choice: `None`, `AnimateDiff`, `Mochi`, `LTXV`, `Hunyuan`, `Cosmos`, `Wan`, `MiniMax H3` | default `"None"` | Model preset (frame grid / dim multiple). |
| `crop_x` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.001 | Crop, as a fraction of the frame (x, y, width, height; 0, 0, 1, 1 = no crop). Set it with the Crop tool in the player; custom_width / custom_height then resize the cropped frame. |
| `crop_y` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.001 | Crop, as a fraction of the frame (x, y, width, height; 0, 0, 1, 1 = no crop). Set it with the Crop tool in the player; custom_width / custom_height then resize the cropped frame. |
| `crop_w` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.001 | Crop, as a fraction of the frame (x, y, width, height; 0, 0, 1, 1 = no crop). Set it with the Crop tool in the player; custom_width / custom_height then resize the cropped frame. |
| `crop_h` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.001 | Crop, as a fraction of the frame (x, y, width, height; 0, 0, 1, 1 = no crop). Set it with the Crop tool in the player; custom_width / custom_height then resize the cropped frame. |
| `decoder` | choice: `auto`, `fast`, `exact` | default `"auto"` | auto: the fast path only where it is faster with the same colour - measured, that is the exact reader on the tested machine. fast: ffmpeg decodes 8-bit 4:2:0 video (the same frames; on an NVIDIA GPU the colour conversion runs there, mean 0.5/255 from the exact reader). exact: the frame-accurate PyAV reader (10-bit, HDR, alpha, ProRes and image sequences always use it). |
| `seconds_cap` | `FLOAT` | default `0.0`, range 0.0…86400.0, step 0.1 | Load at most this many seconds after skip_first_frames (0 = no limit), at the force_rate rate when set. Works with frame_load_cap: the smaller one wins, and a format preset then snaps the count. |
| `resolution` | choice: `source`, `2160p (4K)`, `1440p`, `1080p`, `720p`, `480p`, `2048 wide`, `1920 wide`, `1024 wide`, `custom` | default `"custom"` | Output size by its LONGEST side, keeping the aspect: 2160p is 3840, 1080p is 1920. custom uses custom_width / custom_height (0 = keep the source size or follow the aspect). source ignores them. |
| `divisible_by` | choice: `1`, `2`, `8`, `16`, `32`, `64` | default `1` | Round both output dimensions to a multiple of this. A format preset or a VAE may raise it. |
| `audio_when_missing` | choice: `silence`, `none` | default `"silence"` | When the file has no audio: silence outputs a silent track as long as the loaded frames (what the loader always did, so downstream nodes keep working); none outputs nothing. |
| `handles` | choice: `off`, `fit`, `add` | default `"off"` | Handles = extra frames for a video model to settle on, added here and trimmed off again by Save Video (C2C) (it reads them from video_info). fit: pad the clip UP to the model's frame grid (handle_target = frames you want out, 0 = this clip's length). add: put handle_frames on, rounded onto the grid. Repeated edge frames, never black; the audio moves with them. |
| `handle_frames` | `INT` | default `8`, range 0…400 | add: frames to put on (side both: on EACH end). With a grid, 0 = just enough to land on it. |
| `handle_target` | `INT` | default `0`, range 0…1000000 | fit: how many frames you want out after the trim. 0 = this clip's own length. |
| `handle_side` | choice: `head`, `tail`, `both` | default `"head"` | Which end gets the handles. head is where a video model settles; both splits them. |
| `handle_grid` | choice: `from format`, `none (any frame count)`, `MiniMax H3 (17n+5)`, `Wan 2.x (4n+1)`, `LTX / LTX-2 (8n+1)`, `Hunyuan Video (4n+1)`, `CogVideoX (8n+1)`, `Mochi (6n+1)`, `custom (step / plus)` | default `"from format"` | The model's frame grid: 'from format' uses the format preset above (MiniMax H3 17n+5, Wan 4n+1, LTXV 8n+1 ...), so there is one grid, not two; or pick one here. |
| `handle_custom_step` | `INT` | default `4`, range 1…1000 | custom grid: step (frames = step * n + plus). |
| `handle_custom_plus` | `INT` | default `1`, range 0…1000 | custom grid: plus. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `IMAGE` | `IMAGE,LATENT` | — |
| 1 | `frame_count` | `INT` | — |
| 2 | `audio` | `AUDIO` | — |
| 3 | `video_info` | `VHS_VIDEOINFO` | — |
| 4 | `video` | `C2C_VIDEO` | — |
| 5 | `mask` | `MASK` | — |
| 6 | `info` | `STRING` | — |


### LoadVideoPathC2C

**Shown in the menu as:** Load Video Path (C2C)




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `video` | `STRING` |  | Path to a video file or image sequence. |
| `force_rate` | `FLOAT` | default `0`, range 0…240, step 1 | Output FPS; 0 keeps source. |
| `custom_width` | `INT` | default `0`, range 0…16384 | Output width; 0 keeps source or derives from height. |
| `custom_height` | `INT` | default `0`, range 0…16384 | Output height; 0 keeps source or derives from width. |
| `frame_load_cap` | `INT` | default `0`, range 0…1000000 | Max frames; 0 = no cap. |
| `skip_first_frames` | `INT` | default `0`, range 0…1000000 | Skip this many frames first. |
| `select_every_nth` | `INT` | default `1`, range 1…100000 | Keep every Nth frame. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `vae` | `VAE` |  | Encode to LATENT instead of IMAGE. |
| `format` | choice: `None`, `AnimateDiff`, `Mochi`, `LTXV`, `Hunyuan`, `Cosmos`, `Wan`, `MiniMax H3` | default `"None"` | Model preset (frame grid / dim multiple). |
| `sequence_fps` | `FLOAT` | default `24`, range 1…240 | FPS for image sequences only. |
| `crop_x` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.001 | Crop, as a fraction of the frame (x, y, width, height; 0, 0, 1, 1 = no crop). Set it with the Crop tool in the player; custom_width / custom_height then resize the cropped frame. |
| `crop_y` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.001 | Crop, as a fraction of the frame (x, y, width, height; 0, 0, 1, 1 = no crop). Set it with the Crop tool in the player; custom_width / custom_height then resize the cropped frame. |
| `crop_w` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.001 | Crop, as a fraction of the frame (x, y, width, height; 0, 0, 1, 1 = no crop). Set it with the Crop tool in the player; custom_width / custom_height then resize the cropped frame. |
| `crop_h` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.001 | Crop, as a fraction of the frame (x, y, width, height; 0, 0, 1, 1 = no crop). Set it with the Crop tool in the player; custom_width / custom_height then resize the cropped frame. |
| `decoder` | choice: `auto`, `fast`, `exact` | default `"auto"` | auto: the fast path only where it is faster with the same colour - measured, that is the exact reader on the tested machine. fast: ffmpeg decodes 8-bit 4:2:0 video (the same frames; on an NVIDIA GPU the colour conversion runs there, mean 0.5/255 from the exact reader). exact: the frame-accurate PyAV reader (10-bit, HDR, alpha, ProRes and image sequences always use it). |
| `seconds_cap` | `FLOAT` | default `0.0`, range 0.0…86400.0, step 0.1 | Load at most this many seconds after skip_first_frames (0 = no limit), at the force_rate rate when set. Works with frame_load_cap: the smaller one wins, and a format preset then snaps the count. |
| `resolution` | choice: `source`, `2160p (4K)`, `1440p`, `1080p`, `720p`, `480p`, `2048 wide`, `1920 wide`, `1024 wide`, `custom` | default `"custom"` | Output size by its LONGEST side, keeping the aspect: 2160p is 3840, 1080p is 1920. custom uses custom_width / custom_height (0 = keep the source size or follow the aspect). source ignores them. |
| `divisible_by` | choice: `1`, `2`, `8`, `16`, `32`, `64` | default `1` | Round both output dimensions to a multiple of this. A format preset or a VAE may raise it. |
| `audio_when_missing` | choice: `silence`, `none` | default `"silence"` | When the file has no audio: silence outputs a silent track as long as the loaded frames (what the loader always did, so downstream nodes keep working); none outputs nothing. |
| `handles` | choice: `off`, `fit`, `add` | default `"off"` | Handles = extra frames for a video model to settle on, added here and trimmed off again by Save Video (C2C) (it reads them from video_info). fit: pad the clip UP to the model's frame grid (handle_target = frames you want out, 0 = this clip's length). add: put handle_frames on, rounded onto the grid. Repeated edge frames, never black; the audio moves with them. |
| `handle_frames` | `INT` | default `8`, range 0…400 | add: frames to put on (side both: on EACH end). With a grid, 0 = just enough to land on it. |
| `handle_target` | `INT` | default `0`, range 0…1000000 | fit: how many frames you want out after the trim. 0 = this clip's own length. |
| `handle_side` | choice: `head`, `tail`, `both` | default `"head"` | Which end gets the handles. head is where a video model settles; both splits them. |
| `handle_grid` | choice: `from format`, `none (any frame count)`, `MiniMax H3 (17n+5)`, `Wan 2.x (4n+1)`, `LTX / LTX-2 (8n+1)`, `Hunyuan Video (4n+1)`, `CogVideoX (8n+1)`, `Mochi (6n+1)`, `custom (step / plus)` | default `"from format"` | The model's frame grid: 'from format' uses the format preset above (MiniMax H3 17n+5, Wan 4n+1, LTXV 8n+1 ...), so there is one grid, not two; or pick one here. |
| `handle_custom_step` | `INT` | default `4`, range 1…1000 | custom grid: step (frames = step * n + plus). |
| `handle_custom_plus` | `INT` | default `1`, range 0…1000 | custom grid: plus. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `IMAGE` | `IMAGE,LATENT` | — |
| 1 | `frame_count` | `INT` | — |
| 2 | `audio` | `AUDIO` | — |
| 3 | `video_info` | `VHS_VIDEOINFO` | — |
| 4 | `video` | `C2C_VIDEO` | — |
| 5 | `mask` | `MASK` | — |
| 6 | `info` | `STRING` | — |


### SaveVideoC2C

**Shown in the menu as:** Save Video (C2C)

Save an IMAGE batch as H.264/H.265/ProRes/DNxHR/FFV1/WebM/GIF/WebP or 16-bit PNG / EXR sequences. Streams chunk-by-chunk within the RAM budget, muxes audio, applies OCIO colour transforms, and shows a browser preview with timeline scrub after each run.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `format` | choice: `custom (container / codec)`, `MP4 H.264`, `MP4 H.265 10-bit`, `MOV ProRes 422`, `MOV ProRes 422 HQ`, `MOV ProRes 4444`, `MOV ProRes 4444 XQ`, `MOV DNxHR HQ`, `MOV DNxHR HQX`, `MOV DNxHR 444`, `MKV FFV1 (lossless)`, `WebM VP9`, … | default `"custom (container / codec)"` | custom: choose container, codec, profile, bit depth and chroma below. The other entries are the old single list, kept so saved workflows encode as before. |
| `fps` | `FLOAT` | default `24.0`, range 0.001…240.0, step 0.001 | Frame rate for the written file. When video_info is wired and fps is left at 24, loaded_fps from the loader is used. |
| `filename_prefix` | `STRING` | default `"C2C/video"` | — |
| `quality` | `INT` | default `80`, range 0…100, step 1 | Encoder quality where the format supports it (CRF / qscale / ProRes profile). Hidden for lossless formats. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `images` | `IMAGE` |  | The frames to save. Or wire latent + vae instead to decode and save as one stream. |
| `audio` | `AUDIO` |  | — |
| `video_info` | `VHS_VIDEOINFO` |  | — |
| `colorspace_in` | choice: `ACES2065-1`, `ACEScc`, `ACEScct`, `ACEScg`, `ADX10`, `ADX16`, `ARRI LogC3 (EI800)`, `ARRI LogC4`, `Apple Log`, `BMDFilm WideGamut Gen5`, `Camera Rec.709`, `CanonLog2 CinemaGamut D55`, … | default `"sRGB - Display"` | — |
| `colorspace_out` | choice: `same as input`, `ACES2065-1`, `ACEScc`, `ACEScct`, `ACEScg`, `ADX10`, `ADX16`, `ARRI LogC3 (EI800)`, `ARRI LogC4`, `Apple Log`, `BMDFilm WideGamut Gen5`, `Camera Rec.709`, … | default `"same as input"` | — |
| `naming` | choice: `default`, `ComfyUI counter`, `Folder Version Incrementer`, `Folder Version` | default `"default"` | Folder Version Incrementer: the version folder and name come from the Folder Version Incrementer node in this graph (no wire needed), or from the incrementer's defaults when there is none; a wired subfolder still wins. ComfyUI counter: prefix_00001 like Save Image. default: the C2C setting (Folder Version Incrementer unless you changed it). |
| `naming_resolved` | `STRING` | default `""` | Set by the UI from c2c.saveVideo.naming. |
| `subfolder` | `STRING` | default `""` | — |
| `save_output` | `BOOLEAN` | default `True` | — |
| `latent` | `LATENT` |  | Stream save: a video latent decoded straight into the file, in order, keeping the VAE's causal state - the full frame batch is never held in RAM. Wire this and vae instead of images. |
| `vae` | `VAE` |  | The VAE for latent (stream save). |
| `tile_mode` | choice: `auto`, `off`, `manual` | default `"auto"` | Stream save, Wan VAEs: auto tiles only when the frame would not fit the GPU (2K on 8 GB -> 512 px). Each tile is decoded over the whole clip in one causal pass and blended on disk, so the clip never sits in RAM. off: no tiles. manual: tile_size. |
| `tile_size` | `INT` | default `512`, range 64…2048, step 64 | Tile size in pixels when tile_mode is manual. |
| `save_metadata` | `BOOLEAN` | default `True` | Store the workflow and prompt in the video file (MP4 / MOV / MKV / WebM), as ComfyUI's own Save Video does, so the file can be dragged back into ComfyUI to rebuild the graph. |
| `container` | choice: `MP4`, `MOV`, `MKV`, `WebM`, `GIF`, `WebP`, `Image sequence` | default `"MP4"` | The file type. Used when format is custom. |
| `codec` | choice: `H.264`, `H.265`, `AV1 (SVT)`, `H.264 (NVENC)`, `H.265 (NVENC)`, `AV1 (NVENC)`, `ProRes`, `DNxHR`, `CineForm`, `FFV1 (lossless)`, `UT Video (lossless)`, `VP9`, … | default `"H.264"` | The video codec; the list follows the container. |
| `profile` | choice: `High`, `Main`, `Baseline`, `High 10`, `High 4:2:2`, `High 4:4:4`, `Main 10`, `Main 12`, `Main 4:2:2 10`, `Main 4:2:2 12`, `Main 4:4:4`, `Main 4:4:4 10`, … | default `"High"` | The codec's profile (ProRes 422 HQ, DNxHR HQX, H.265 Main 10, EXR ZIP ...). |
| `bit_depth` | choice: `8`, `10`, `12`, `16`, `32 (float)` | default `"8"` | Bits per channel the profile allows. EXR: 16 = half float, 32 = float. |
| `chroma` | choice: `4:2:0`, `4:2:2`, `4:4:4`, `RGB` | default `"4:2:0"` | Chroma subsampling: 4:2:0 (delivery), 4:2:2 (edit), 4:4:4 / RGB (full). |
| `color_primaries` | choice: `auto`, `BT.709`, `BT.2020`, `Display P3`, `BT.601` | default `"auto"` | Colour primaries written into the file. auto: from colorspace_out (sRGB / Rec.709 named, others untagged). For YUV codecs the conversion matrix follows (BT.2020 -> BT.2020 NCL). |
| `color_transfer` | choice: `auto`, `BT.709`, `sRGB`, `linear`, `PQ (ST 2084)`, `HLG`, `gamma 2.2` | default `"auto"` | Transfer function written into the file (it labels; the pixels come from colorspace_out). |
| `color_range` | choice: `auto`, `limited`, `full` | default `"auto"` | YUV codecs: limited (16-235, broadcast, the default) or full (0-255). Changes the conversion and the tag together. RGB codecs are always full range. |
| `handles_trim` | choice: `auto`, `manual`, `off` | default `"auto"` | auto: trim exactly the handles Load Video (C2C) added (read from video_info; fit's target length wins). manual: trim_frames on trim_side. off: write every frame that arrives. |
| `trim_frames` | `INT` | default `8`, range 0…100000 | manual trim: frames to cut (side both: on each end). |
| `trim_side` | choice: `head`, `tail`, `both` | default `"head"` | manual trim: which end. |
| `range_in` | `INT` | default `0`, range 0…10000000 | Export range, first frame (of the trimmed clip). Set it with the player's In handle. 0 / 0 = the whole clip. |
| `range_out` | `INT` | default `0`, range 0…10000000 | Export range, last frame (inclusive). 0 = to the end. |
| `write_mode` | choice: `write`, `preview first` | default `"write"` | preview first: a run shows a quick preview of exactly what would be written (handles trimmed, range applied) and writes nothing; the player's Write file button then writes it - only the encode runs again, the graph above is cached. |
| `write_token` | `STRING` | default `""` | Set by the player's Write file button. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `filenames` | `STRING` | — |
| 1 | `preview` | `IMAGE` | — |


### VideoInfoC2C

**Shown in the menu as:** Video Info (C2C)




**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `video_info` | `VHS_VIDEOINFO` |  | Metadata from a C2C or VHS loader. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `source_fps` | `FLOAT` | — |
| 1 | `source_frame_count` | `INT` | — |
| 2 | `source_duration` | `FLOAT` | — |
| 3 | `source_width` | `INT` | — |
| 4 | `source_height` | `INT` | — |
| 5 | `loaded_fps` | `FLOAT` | — |
| 6 | `loaded_frame_count` | `INT` | — |
| 7 | `loaded_duration` | `FLOAT` | — |
| 8 | `loaded_width` | `INT` | — |
| 9 | `loaded_height` | `INT` | — |


---

### LoadVideoC2C / LoadVideoPathC2C: inputs added in L2.53

**Shown in the menu as:** Load Video (C2C) / Load Video Path (C2C). This reference predates these nodes; the table
lists only what L2.53 appended. The full guide is in README.md, "Load Video (C2C)", and every widget has a tooltip.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `format` | choice | `None` | Adds the **MiniMax H3** preset (24 fps, 17n+5 frames, size multiple 32) to the earlier presets. |
| `decoder` | choice: `auto`, `fast`, `exact` | `auto` | `exact` = PyAV reader. `fast` = ffmpeg pipe (bit-identical rgb48, or nv12 on CUDA). `auto` = exact unless `C2C_VIDEO_AUTO` opts in. |
| `seconds_cap` | `FLOAT` | `0.0` | Cap the load by time (0 = no cap). Clears `frame_load_cap`, and the other way round. |
| `resolution` | choice | `custom` | Size by the longest side, keeping the aspect. `custom` = `custom_width` / `custom_height` as before. |
| `divisible_by` | choice: `1`, `2`, `8`, `16`, `32`, `64` | `1` | Round the output size to this multiple. |
| `audio_when_missing` | choice: `silence`, `none` | `silence` | When the clip has no audio: a silent track as long as the frames, or nothing. |
| **output** `info` | `STRING` | | JSON: source and loaded frames, seconds, size, fps, and the decoder used. |

### SaveVideoC2C: inputs added in L2.53

**Shown in the menu as:** Save Video (C2C). New formats: H.265 8-bit (CPU), and H.264 / H.265 / H.265 10-bit / AV1
on NVENC. NTSC rates are stored exactly.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `save_metadata` | `BOOLEAN` | `True` | Store the workflow and prompt in MP4 / MOV / MKV / WebM so the file rebuilds the graph when dropped onto ComfyUI. Nothing is stored under `--disable-metadata`. |

### SaveVideoC2C: inputs added in L9.40 (2026-10-23)

`format` gains *custom (container / codec)* (the default for new nodes); the old labels still encode as before.

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `container` | choice | `MP4` | MP4, MOV, MKV, WebM, GIF, WebP, Image sequence. Used when format is custom. |
| `codec` | choice | `H.264` | The codecs the container takes (the list follows the container). |
| `profile` | choice | `High` | The codec's profile: ProRes 422 Proxy ... 4444 XQ, DNxHR LB ... 444, H.265 Main ... Main 4:4:4 12, VP9 0-3, EXR / TIFF compression ... |
| `bit_depth` | choice | `8` | What the profile allows; EXR 16 = half, 32 = float. |
| `chroma` | choice | `4:2:0` | 4:2:0, 4:2:2, 4:4:4 or RGB, as the profile allows. |
| `color_primaries` / `color_transfer` / `color_range` | choice | `auto` | Tags written into the file; for YUV codecs the matrix and range of the conversion follow them. |
| `handles_trim` | choice | `auto` | auto: trim the handles Load Video (C2C) added (from video_info); manual: `trim_frames` on `trim_side`; off. |
| `trim_frames` / `trim_side` | INT / choice | `8` / `head` | Manual trim. |
| `range_in` / `range_out` | INT | `0` / `0` | Export range (inclusive, of the trimmed clip), set with the player's In / Out. 0 / 0 = all. |
| `write_mode` | choice | `write` | preview first: a run previews and writes nothing; the player's Write file writes it. |
| `write_token` | STRING | `""` | Set by the Write file button (hidden). |

`naming` gains *Folder Version Incrementer* (the old *Folder Version* is the same); it fills the version path from
the incrementer node in the graph, or from its defaults.

### LoadVideoC2C / LoadVideoPathC2C: inputs added in L9.40 (2026-10-23)

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `handles` | choice | `off` | fit: pad up to the grid; add: put `handle_frames` on. The count goes into video_info for Save Video (C2C). |
| `handle_frames` | INT | `8` | add: frames to put on (side both: each end). |
| `handle_target` | INT | `0` | fit: frames wanted out; 0 = the clip's length. |
| `handle_side` | choice | `head` | head, tail or both. |
| `handle_grid` | choice | `from format` | The frame grid: from the format preset, or one of AV Handles' grids. |
| `handle_custom_step` / `handle_custom_plus` | INT | `4` / `1` | The custom grid. |

### VideoComparerC2C: inputs added in L9.40 (2026-10-23)

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `image_c` / `image_d`, `file_c` / `file_d`, `label_c` / `label_d` | IMAGE / file / STRING | | The 2 x 2 grid's C and D. |
| `b_offset` | INT | `0` | B starts this many frames after A; the player and the metrics use it. |
| `metrics` | BOOLEAN | `True` | Per-frame PSNR / SSIM / CIEDE2000 of A against B, drawn under the scrubber. |

## C2C/VideoMask


### VideoMaskEditorMEC

**Shown in the menu as:** Video Mask Editor

Paint masks on the frames of a video in the editor (Open Video Mask Editor on the node). A painted frame is a keyframe
and its mask is used on that frame; tween_mode and outside_keys decide the unpainted frames, and input_mask fills the
frames they leave empty. The editor shows the output mask of every frame. Keyframes are saved to
input/c2c_video_masks/<session>/ as you paint.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Input video batch (B,H,W,3). Drives B/H/W. |
| `session_id` | `STRING` | default `""` | Set by the editor; hidden on the node. |
| `tween_mode` | choice: `distance_transform`, `linear`, `hold`, `none` | default `"distance_transform"` | Frames BETWEEN two painted frames: distance_transform morphs the shape, linear blends, hold keeps the previous painted mask, none leaves them unpainted (frame by frame). |
| `feather` | `FLOAT` | default `0.0`, range 0.0…32.0, step 0.1 | Gaussian feather radius (px). 0 = none. |
| `threshold` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | Binarize after tween. 0 = keep soft mask. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `input_mask` | `MASK` |  | Per-frame base mask (e.g. a tracked mask): every frame your keyframes do not fill keeps its own frame of this mask. |
| `outside_keys` | choice: `empty`, `hold` | default `"empty"` | Frames before the first / after the last painted frame: empty (the mask stays on the frames you painted) or hold the nearest painted mask. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | One mask per frame. |
| 1 | `info` | `STRING` | JSON: painted frames, modes, and how many frames were painted / interpolated / held / input mask / empty. |


---|---|---|---|
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


### C2CVAEQualityDecode

**Shown in the menu as:** VAE Decode (C2C)

VAE decode with optional fp32 (<=0.17 dB measured gain), spatial-only tiling for Wan video, optional unclamped output for HDR chains, optional ACES tone mapping, and an optional clean step that measures and corrects colour cast, oversaturation and decode speckle (also for an already-decoded image).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `force_fp32` | `BOOLEAN` | default `True` | Decode in fp32. Measured gain: at most 0.17 dB on Flux/Wan 2.1 round-trip (~2x VAE memory). |
| `tile_size` | `INT` | default `0`, range 0…1024, step 64 | Spatial tile size in PIXELS, used when tile_mode is manual (0 = no tiling). Converted with the VAE spatial factor (usually 8). 512 measured best for Wan on 8 GB. |
| `apply_aces` | `BOOLEAN` | default `False` | Apply ACES filmic tone mapping after decode (the decode is treated as sRGB: linearised, tone-mapped with the ACES fit, encoded once). |
| `exposure` | `FLOAT` | default `1.0`, range 0.01…10.0, step 0.05 | Exposure for ACES (only used if apply_aces=True). |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `clamp_output` | `BOOLEAN` | default `True` | Core clamps decoder output to 0..1. Off keeps out-of-range values (4.6x / 5.3x more accurate there on Flux/Wan, measured M2). For HDR/EXR chains. |
| `tile_mode` | choice: `auto`, `off`, `manual` | default `"auto"` | auto: tile only when the untiled decode would not fit the GPU (measured on an 8 GB card: a 2K Wan decode took 502 s untiled, spilling into system RAM, and 37 s tiled, with the same picture). off: never tile. manual: tile at tile_size. Video is tiled in space only — splitting a video VAE in time costs ~21 dB. |
| `samples` | `LATENT` |  | The latent to decode (with vae). Leave empty to clean `image`. |
| `vae` | `VAE` |  | — |
| `image` | `IMAGE` |  | An already-decoded picture to clean instead of decoding a latent. |
| `clean` | `BOOLEAN` | default `False` | Measure the decode and correct what is wrong with it: colour cast, oversaturation, crushed shadows, chroma speckle. The report says what it found, including clipping it cannot undo. |
| `balance_mode` | choice: `off`, `grey world`, `match reference`, `white point` | default `"off"` | Clean (when clean is on): off = measure only; grey world / white point / match reference — see node tooltip for when each is safe. |
| `balance_strength` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.05 | Clean (when clean is on): how far toward neutral (1.0 = full correction). |
| `saturation` | `FLOAT` | default `1.0`, range 0.0…2.0, step 0.01 | Clean (when clean is on): chroma scale about luma (1.0 leaves it alone). |
| `contrast_restore` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.05 | Clean (when clean is on): lift crushed shadows / reduce excess contrast. |
| `chroma_cleanup` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.05 | Clean (when clean is on): smooth colour noise (VAE decode speckle) without blurring luma detail. |
| `reference` | `IMAGE` |  | Clean (when clean is on): required by `match reference`; plate for round-trip meter and `restore_unchanged`. |
| `restore_unchanged` | `FLOAT` | default `0.0`, range 0.0…0.5, step 0.005 | Clean (when clean is on): experimental write-back of unchanged pixels from `reference` (0 = off). |
| `restore_radius` | `INT` | default `8`, range 1…64 | Clean (when clean is on): window in pixels for write-back (experimental). |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `IMAGE` | `IMAGE` | The decoded (and, with clean on, corrected) picture. |
| 1 | `report` | `STRING` | Clean: what was measured and corrected (empty when clean is off). |
| 2 | `cast_strength` | `FLOAT` | Clean: how strong the colour cast was. |
| 3 | `saturation` | `FLOAT` | Clean: measured saturation. |


---

## MEC/LayerEffects


### LayerEffectsMEC

**Shown in the menu as:** Layer Effects (C2C)

Photoshop-style layer effects in one node: drop shadow, outer glow, gradient map, gradient overlay, colour overlay, inner glow, inner shadow and stroke. Switch each on and set its controls; they are applied in the order you list, each on the result of the one before.

Saved workflows with the eight former single-effect nodes (Drop Shadow, Inner Shadow, Outer Glow, Inner Glow, Stroke, Color Overlay, Gradient Overlay, Gradient Map) load here with only that effect switched on and give the same image. Each effect's controls show while it is on.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `layer_image` | `IMAGE` |  | Foreground layer; its alpha or layer_mask defines where the effects go. |
| `invert_mask` | `BOOLEAN` | default `True` | Flip mask polarity when your matte is white-on-black. |
| `order` | `STRING` | default `'drop_shadow, outer_glow, gradient_map, gradient_overlay, color_overlay, inner_glow, inner_shadow, stroke'` | The order the switched-on effects are applied in, comma separated; each is applied to the result of the one before. An effect that is on but not listed runs after the listed ones, in the default order. |
| `dissolve_seed` | `INT` | default `0`, range 0…2147483647, step 1 | Only used by the 'dissolve' blend mode. Upstream reseeds every frame, so a dissolve flickers through a sequence; a fixed seed here holds still. |
| `drop_shadow` | `BOOLEAN` | default `True` | Drop shadow: switch it on to show and use its controls. |
| `drop_shadow_blend_mode` | choice: `normal`, `dissolve`, `darken`, `multiply`, `color burn`, `linear burn`, `darker color`, `lighten`, `screen`, `color dodge`, `linear dodge(add)`, `lighter color`, … |  | Drop shadow |
| `drop_shadow_opacity` | `INT` | default `50`, range 0…100, step 1 | Drop shadow |
| `drop_shadow_distance_x` | `INT` | default `25`, range -9999…9999, step 1 | Drop shadow: Horizontal offset in pixels. Positive moves the shadow right — the opposite of the upstream LayerStyle node, which negated it. |
| `drop_shadow_distance_y` | `INT` | default `25`, range -9999…9999, step 1 | Drop shadow: Vertical offset in pixels. Positive moves the shadow down — the opposite of the upstream LayerStyle node, which negated it. |
| `drop_shadow_grow` | `INT` | default `6`, range -9999…9999, step 1 | Drop shadow: Expand the shadow before blurring. Positive spreads it past the subject edge; negative pulls it inside, which keeps a tight contact shadow from haloing. |
| `drop_shadow_blur` | `INT` | default `18`, range 0…1000, step 1 | Drop shadow: Gaussian blur radius on the shadow matte. |
| `drop_shadow_color` | `STRING` | default `'#000000'` | Drop shadow |
| `outer_glow` | `BOOLEAN` | default `False` | Outer glow: switch it on to show and use its controls. |
| `outer_glow_blend_mode` | choice: `screen`, `linear dodge(add)`, `color dodge`, `lighten`, `dodge`, `hard light`, `linear light`, `normal`, `dissolve`, `darken`, `multiply`, `color burn`, … |  | Outer glow |
| `outer_glow_opacity` | `INT` | default `100`, range 0…100, step 1 | Outer glow |
| `outer_glow_brightness` | `INT` | default `5`, range 2…20, step 1 | Outer glow: Number of glow passes from core to edge. |
| `outer_glow_range` | `INT` | default `48`, range -9999…9999, step 1 | Outer glow: Initial spread for the outer glow; clamped to frame size. |
| `outer_glow_blur` | `INT` | default `25`, range 0…9999, step 1 | Outer glow |
| `outer_glow_light_color` | `STRING` | default `'#FFBF30'` | Outer glow |
| `outer_glow_color` | `STRING` | default `'#FE0000'` | Outer glow |
| `gradient_map` | `BOOLEAN` | default `False` | Gradient map: switch it on to show and use its controls. |
| `gradient_map_start_color` | `STRING` | default `'#015A52'` | Gradient map |
| `gradient_map_mid_color` | `STRING` | default `'#02AF9F'` | Gradient map |
| `gradient_map_end_color` | `STRING` | default `'#7FFFEC'` | Gradient map |
| `gradient_map_mid_point` | `FLOAT` | default `0.6`, range 0.0…1.0, step 0.01 | Gradient map |
| `gradient_map_opacity` | `INT` | default `100`, range 0…100, step 1 | Gradient map |
| `gradient_overlay` | `BOOLEAN` | default `False` | Gradient overlay: switch it on to show and use its controls. |
| `gradient_overlay_blend_mode` | choice: `normal`, `dissolve`, `darken`, `multiply`, `color burn`, `linear burn`, `darker color`, `lighten`, `screen`, `color dodge`, `linear dodge(add)`, `lighter color`, … |  | Gradient overlay |
| `gradient_overlay_opacity` | `INT` | default `100`, range 0…100, step 1 | Gradient overlay |
| `gradient_overlay_start_color` | `STRING` | default `'#FFBF30'` | Gradient overlay |
| `gradient_overlay_start_alpha` | `INT` | default `255`, range 0…255, step 1 | Gradient overlay |
| `gradient_overlay_end_color` | `STRING` | default `'#FE0000'` | Gradient overlay |
| `gradient_overlay_end_alpha` | `INT` | default `255`, range 0…255, step 1 | Gradient overlay |
| `gradient_overlay_angle` | `INT` | default `0`, range -180…180, step 1 | Gradient overlay |
| `color_overlay` | `BOOLEAN` | default `False` | Colour overlay: switch it on to show and use its controls. |
| `color_overlay_blend_mode` | choice: `normal`, `dissolve`, `darken`, `multiply`, `color burn`, `linear burn`, `darker color`, `lighten`, `screen`, `color dodge`, `linear dodge(add)`, `lighter color`, … |  | Colour overlay |
| `color_overlay_opacity` | `INT` | default `100`, range 0…100, step 1 | Colour overlay |
| `color_overlay_color` | `STRING` | default `'#FFBF30'` | Colour overlay |
| `inner_glow` | `BOOLEAN` | default `False` | Inner glow: switch it on to show and use its controls. |
| `inner_glow_blend_mode` | choice: `screen`, `linear dodge(add)`, `color dodge`, `lighten`, `dodge`, `hard light`, `linear light`, `normal`, `dissolve`, `darken`, `multiply`, `color burn`, … |  | Inner glow |
| `inner_glow_opacity` | `INT` | default `100`, range 0…100, step 1 | Inner glow |
| `inner_glow_brightness` | `INT` | default `5`, range 2…20, step 1 | Inner glow: Number of glow passes from core to edge. |
| `inner_glow_range` | `INT` | default `48`, range -9999…9999, step 1 | Inner glow: Initial spread for the outer glow; clamped to frame size. |
| `inner_glow_blur` | `INT` | default `25`, range 0…9999, step 1 | Inner glow |
| `inner_glow_light_color` | `STRING` | default `'#FFBF30'` | Inner glow |
| `inner_glow_color` | `STRING` | default `'#FE0000'` | Inner glow |
| `inner_shadow` | `BOOLEAN` | default `False` | Inner shadow: switch it on to show and use its controls. |
| `inner_shadow_blend_mode` | choice: `normal`, `dissolve`, `darken`, `multiply`, `color burn`, `linear burn`, `darker color`, `lighten`, `screen`, `color dodge`, `linear dodge(add)`, `lighter color`, … |  | Inner shadow |
| `inner_shadow_opacity` | `INT` | default `50`, range 0…100, step 1 | Inner shadow |
| `inner_shadow_distance_x` | `INT` | default `5`, range -9999…9999, step 1 | Inner shadow: Horizontal offset in pixels. Positive moves the shadow right — the opposite of the upstream LayerStyle node, which negated it. |
| `inner_shadow_distance_y` | `INT` | default `5`, range -9999…9999, step 1 | Inner shadow: Vertical offset in pixels. Positive moves the shadow down — the opposite of the upstream LayerStyle node, which negated it. |
| `inner_shadow_grow` | `INT` | default `2`, range -9999…9999, step 1 | Inner shadow: Expand the shadow before blurring. Positive spreads it past the subject edge; negative pulls it inside, which keeps a tight contact shadow from haloing. |
| `inner_shadow_blur` | `INT` | default `15`, range 0…1000, step 1 | Inner shadow: Gaussian blur radius on the shadow matte. |
| `inner_shadow_color` | `STRING` | default `'#000000'` | Inner shadow |
| `stroke` | `BOOLEAN` | default `False` | Stroke: switch it on to show and use its controls. |
| `stroke_blend_mode` | choice: `normal`, `dissolve`, `darken`, `multiply`, `color burn`, `linear burn`, `darker color`, `lighten`, `screen`, `color dodge`, `linear dodge(add)`, `lighter color`, … |  | Stroke |
| `stroke_opacity` | `INT` | default `100`, range 0…100, step 1 | Stroke |
| `stroke_grow` | `INT` | default `0`, range -999…999, step 1 | Stroke |
| `stroke_width` | `INT` | default `8`, range 0…999, step 1 | Stroke |
| `stroke_blur` | `INT` | default `0`, range 0…100, step 1 | Stroke |
| `stroke_color` | `STRING` | default `'#FF0000'` | Stroke |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `background_image` | `IMAGE` |  | Plate to composite onto; empty = transparent canvas. |
| `layer_mask` | `MASK` |  | Optional matte; overrides the layer's alpha when connected. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `image` | `IMAGE` | The layer with its effects, over the background. |
| 1 | `effect_mask` | `MASK` | Where the effects landed: the union of every switched-on effect's footprint. |
| 2 | `report` | `STRING` | One line per effect applied, with anything worth knowing (clamped sizes, resized masks). |


---

## MEC/Mask


### MaskToolsMEC

**Shown in the menu as:** Mask Tools (C2C)

Keyers and matte operations in one node: colour, luma and difference keys; grade (gain, gamma, clamp, invert); grow or shrink each side (left / right / up / down) with a feather; combine two mattes; gradient; grain; motion blur; edge spread; channel shuffle.

Saved workflows with the eight former nodes (Mask From Color, Luminance Keyer, Mask Gradient, Mask Grain, Mask Motion Blur, Edge Spread, Mask Batch Combine, Shuffle) load here in their mode and give the same result. Each mode shows only its own controls, and its preview (colour tolerance, luma window, gradient shape, grain tile, blur dial).


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `mode` | choice: `key: colour`, `key: luma`, `key: difference`, `grade`, `grow / shrink`, `combine`, `gradient`, `grain`, `motion blur`, `edge spread`, `shuffle` | default `"key: colour"` | What the node does. Each mode shows only its own controls. |
| `colour_key_color` | `STRING` | default `"#FFFFFF"` | key: colour: Target colour as #RRGGBB — the hue you want to keep or remove. |
| `colour_key_colorspace` | choice: `rgb`, `hsv`, `lab` | default `"rgb"` | key: colour: Distance metric. LAB separates dark navy from dark brown; RGB treats them as similar luminance. |
| `colour_key_tolerance` | `FLOAT` | default `50.0`, range 0.0…100.0, step 1.0 | key: colour: How far a pixel's colour can drift from the pick and still count as a match. |
| `colour_key_soft_falloff` | `FLOAT` | default `10.0`, range 0.0…100.0, step 1.0 | key: colour: Graded band outside tolerance — stops a colour key from cutting like a pair of scissors. |
| `colour_key_invert` | `BOOLEAN` | default `False` | key: colour: Swap matched and unmatched regions. |
| `luma_key_mode` | choice: `auto`, `highlights`, `midtones`, `shadows`, `custom` | default `"auto"` | key: luma: Preset luminance range or custom thresholds. auto: Analyzes image brightness to pick best range. highlights: Keys bright regions (0.7–1.0). midtones: Keys mid-range luminance (0.3–0.7). shadows: Keys dark regions (0.0–0.3). custom: Uses the low/high sliders directly. |
| `luma_key_low` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | key: luma: Low threshold – pixels with luminance below this become 0 in the mask. Only used directly in custom mode; presets override this. |
| `luma_key_high` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | key: luma: High threshold – pixels with luminance above this become 1 in the mask. Only used directly in custom mode; presets override this. |
| `luma_key_gamma` | `FLOAT` | default `1.0`, range 0.01…10.0, step 0.01 | key: luma: Gamma correction applied after keying. >1 compresses mask toward black (reduces coverage). <1 expands mask toward white (increases coverage). |
| `luma_key_falloff` | `FLOAT` | default `1.0`, range 0.0…10.0, step 0.1 | key: luma: Smoothness of the transition between low and high thresholds. 0 = hard binary edge. 1 = standard smooth. >1 = very gradual transition. |
| `luma_key_invert` | `BOOLEAN` | default `False` | key: luma: Invert the output mask (swap keyed and unkeyed regions). |
| `luma_key_channel` | choice: `luma`, `red`, `green`, `blue`, `hue`, `saturation`, `value`, `L`, `a`, `b` | default `"luma"` | key: luma: Channel to key on. Luma is BT.709; hue/sat/value and LAB a/b isolate coloured spill without three separate nodes. |
| `luma_key_low_soft` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | key: luma: Blend-If split below low — ramp into the key instead of a hard cut that looks like a sticker. |
| `luma_key_high_soft` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | key: luma: Blend-If split above high — feather the highlight side of the range. |
| `luma_key_invert_key` | `BOOLEAN` | default `False` | key: luma: Invert the sampled channel before thresholding (ImageToMask behaviour). Distinct from invert, which flips the finished matte. |
| `difference_key_tolerance` | `FLOAT` | default `0.08`, range 0.0…1.0, step 0.005 | key: difference: How far a pixel may differ from the clean plate and still count as background (0..1 of the RGB distance). |
| `difference_key_softness` | `FLOAT` | default `0.05`, range 0.0…1.0, step 0.005 | key: difference: Width of the soft edge above the tolerance. |
| `difference_key_invert` | `BOOLEAN` | default `False` | key: difference: Key the unchanged part instead of what changed. |
| `grade_gain` | `FLOAT` | default `1.0`, range 0.0…16.0, step 0.01 | grade: Multiply the matte (above 1 pushes greys toward white). |
| `grade_gamma` | `FLOAT` | default `1.0`, range 0.05…8.0, step 0.01 | grade: Bend the greys: above 1 lifts them, below 1 sinks them; black and white stay put. |
| `grade_clamp` | `BOOLEAN` | default `True` | grade: Keep the result inside 0..1. |
| `grade_invert` | `BOOLEAN` | default `False` | grade: Flip the matte at the end. |
| `grow_left` | `INT` | default `0`, range -4096…4096, step 1 | grow / shrink: Pixels to grow the matte toward the left (negative shrinks its left edge). |
| `grow_right` | `INT` | default `0`, range -4096…4096, step 1 | grow / shrink: Pixels to grow toward the right (negative shrinks its right edge). |
| `grow_up` | `INT` | default `0`, range -4096…4096, step 1 | grow / shrink: Pixels to grow upward (negative shrinks its top edge). |
| `grow_down` | `INT` | default `0`, range -4096…4096, step 1 | grow / shrink: Pixels to grow downward (negative shrinks its bottom edge). |
| `grow_feather` | `INT` | default `0`, range 0…1000, step 1 | grow / shrink: Soften the result by this blur radius in pixels. |
| `combine_op` | choice: `union`, `intersect`, `diff`, `xor`, `add`, `subtract` | default `"union"` | combine |
| `gradient_width` | `INT` | default `512`, range 4…16384, step 1 | gradient: Output width when size_as is not connected. |
| `gradient_height` | `INT` | default `512`, range 4…16384, step 1 | gradient: Output height when size_as is not connected. |
| `gradient_type` | choice: `linear`, `radial`, `angular` | default `"linear"` | gradient: Ramp shape — linear edge fade, radial vignette, or angular wipe. |
| `gradient_angle` | `FLOAT` | default `0.0`, range -360.0…360.0, step 0.1 | gradient: Direction for linear, or rotation offset for angular ramps. |
| `gradient_center_x` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | gradient: Radial/angular origin, normalised 0–1 across the frame. |
| `gradient_center_y` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | gradient: Radial/angular origin, normalised 0–1 down the frame. |
| `gradient_start` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | gradient: Mask value at the ramp start (black end). |
| `gradient_end` | `FLOAT` | default `1.0`, range 0.0…1.0, step 0.01 | gradient: Mask value at the ramp end (white end). |
| `grain_amount` | `INT` | default `6`, range 0…127, step 1 | grain: Noise strength — enough to hide banding, not enough to read as texture. |
| `grain_seed` | `INT` | default `0`, range 0…2147483647, step 1 | grain: Fixed seed holds the grain still through a sequence; upstream reseeded every frame and the matte crawled. |
| `grain_size` | `INT` | default `4`, range 1…64, step 1 | grain: Noise frequency — larger values give coarser, less speckly grain. |
| `grain_invert` | `BOOLEAN` | default `False` | grain: Flip mask polarity before adding grain. |
| `motion_blur_angle` | `FLOAT` | default `0.0`, range -360.0…360.0, step 0.1 | motion blur: Blur direction in degrees, 0 = horizontal right. |
| `motion_blur_distance` | `INT` | default `20`, range 0…9999, step 1 | motion blur: Blur length in pixels along the angle. |
| `motion_blur_invert` | `BOOLEAN` | default `False` | motion blur: Flip mask polarity before blurring. |
| `edge_spread` | `INT` | default `4`, range 0…9999, step 1 | edge spread: How many pixels to push inward colour outward — kills the dark fringe on a lighter background. |
| `edge_spread_invert_mask` | `BOOLEAN` | default `False` | edge spread: Flip mask polarity when your matte is white-on-black. |
| `shuffle_out_R` | choice: `R`, `G`, `B`, `A`, `Lum`, `InvR`, `InvG`, `InvB`, `InvA`, `zero`, `one` | default `"R"` | shuffle |
| `shuffle_out_G` | choice: `R`, `G`, `B`, `A`, `Lum`, `InvR`, `InvG`, `InvB`, `InvA`, `zero`, `one` | default `"G"` | shuffle |
| `shuffle_out_B` | choice: `R`, `G`, `B`, `A`, `Lum`, `InvR`, `InvG`, `InvB`, `InvA`, `zero`, `one` | default `"B"` | shuffle |
| `shuffle_out_A` | choice: `R`, `G`, `B`, `A`, `Lum`, `InvR`, `InvG`, `InvB`, `InvA`, `zero`, `one` | default `"A"` | shuffle |
| `shuffle_premultiply_output` | `BOOLEAN` | default `False` | shuffle |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | keys, edge spread and shuffle: the picture to work on. |
| `mask` | `MASK` |  | grade, grow / shrink, combine (first matte), grain, motion blur; edge spread's matte; gradient's size when nothing else gives one. |
| `mask_b` | `MASK` |  | combine: the second matte. |
| `plate` | `IMAGE` |  | key: difference - the clean plate (same shot, without the subject). |
| `size_as` | `IMAGE` |  | gradient: take the size from this image. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `mask` | `MASK` | The matte (shuffle: the new alpha; edge spread: the matte it used). |
| 1 | `image` | `IMAGE` | edge spread / shuffle: the processed picture. Other modes: the matte as a grey image, to look at. |
| 2 | `report` | `STRING` | What the node did, and anything worth knowing (clamped values, resized inputs). |


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

## MaskEditControl/IO


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

## MaskEditControl/Pipeline


### MaskOpsMEC

**Shown in the menu as:** Mask + Matting

Production-grade segmentation + matting + AUTO-QUALITY + diagnostics in one node. Multi-backend (SAM 2.1 / SAM 3 / SAM 3.1 / BiRefNet / RMBG-2.0 / InSPyReNet + ViTMatte / RVM / MatAnyone). Optional Nuke-style luma-key pre-stage, edge-aware trimap, and AUTO-QUALITY pipeline that detects motion blur, low light, low contrast, speckle noise and similar bg/fg, then applies just-enough pre-processing before the segmenter and a light guided-filter polish on the alpha — no knobs to tune. When you supply pos+neg points (e.g. pos on face, neg on neck) it picks the SAM candidate that excludes the neg points, so you get the face only, not the whole person. For manual 11-stage refinement (hole-fill, joint bilateral, DenseCRF, etc.) chain MaskRefineMEC downstream. The `backend` input also runs SAM + ViTMatte, SeC + MatAnyone2 or background removal (RMBG-2.0 / BiRefNet) - the former separate nodes - through the same outputs.

**Backends (L7.65 P05).** `backend` = `segment + matte` (default: this node's pipeline / segmenter / matter, unchanged), `SAM + ViTMatte` (needs `sam_model` from a SAM Model Loader; reads `points_json`, `bbox_json`, `subject_preset`, the `sam_*` controls, `fill_holes_enabled`, `min_region_size`, `pos_bbox`, `external_mask`, `external_trimap`), `SeC + MatAnyone2` (reads `sec_*`, `points_json`, `bbox_json`, `precision`, `fill_holes_enabled`, `min_region_size`, `keep_model_loaded`, `pos_bbox`; a wired `text_prompt` wins over `sec_text_prompt`) and `remove background` (reads `bg_*`, `keep_model_loaded`). `auto_download` is the Allow Download of every backend. In the three added backends `preview`, `score`, `edge_mask` and (SAM + ViTMatte) `bbox` keep the former node's definition so migrated workflows give the same results; `split_masks` off leaves the edge / inside / outside masks empty. Design: `docs/design/matte_unified.md`.


**Required inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `image` | `IMAGE` |  | Source image or video frames (B,H,W,C). |
| `segmenter` | choice: `auto_best`, `auto`, `sam3  [missing-deps]`, `sam3.1  [missing-deps]`, `birefnet`, `rmbg`, `inspyrenet`, `dis`, `person-mask`, `grounding-dino`, `cutie  [missing-deps]`, `xmem  [missing-deps]`, … | default `"auto_best"` | Coarse-mask backend. Entries tagged [missing-deps] need an optional pip install to activate. |
| `matter` | choice: `none`, `auto`, `vitmatte`, `rvm`, `bgmattingv2`, `matanyone  [missing-deps]`, `birefnet`, `rmbg`, `guided_filter` | default `"vitmatte"` | Optional alpha refinement. 'none' returns the segmenter mask as alpha. |
| `model` | choice: `(auto)`, `[preset:sam3] sam3.safetensors`, `[preset:sam3.1] sam3.1_multiplex_fp16.safetensors`, `[preset:sam3.1] sam3.1_multiplex_fp32.safetensors`, `[preset:sam3.1] sam3.pt`, `[preset:birefnet] BiRefNet-general-epoch_244.pth`, `[preset:rmbg] RMBG-2.0.safetensors`, `[preset:inspyrenet] InSPyReNet_SwinB.pth`, `[preset:grounding-dino] groundingdino_swint_ogc.pth`, `[preset:sec] SeC-4B.safetensors`, `[preset:vitmatte] vitmatte-small-composition-1k.safetensors`, `[preset:vitmatte] vitmatte-base-composition-1k.safetensors`, … | default `"(auto)"` | Specific weight file to use. Tag prefix selects the backend folder; '(auto)' lets each backend pick. |
| `matter_model` | choice: `(auto)`, `[preset:sam3] sam3.safetensors`, `[preset:sam3.1] sam3.1_multiplex_fp16.safetensors`, `[preset:sam3.1] sam3.1_multiplex_fp32.safetensors`, `[preset:sam3.1] sam3.pt`, `[preset:birefnet] BiRefNet-general-epoch_244.pth`, `[preset:rmbg] RMBG-2.0.safetensors`, `[preset:inspyrenet] InSPyReNet_SwinB.pth`, `[preset:grounding-dino] groundingdino_swint_ogc.pth`, `[preset:sec] SeC-4B.safetensors`, `[preset:vitmatte] vitmatte-small-composition-1k.safetensors`, `[preset:vitmatte] vitmatte-base-composition-1k.safetensors`, … | default `"(auto)"` | Weight file for the matter backend. |
| `precision` | choice: `fp16`, `bf16`, `fp32` | default `"fp16"` | — |
| `attention` | choice: `auto`, `sdpa`, `flash`, `sage`, `xformers`, `eager` | default `"auto"` | — |
| `offload` | choice: `none`, `cpu`, `sequential` | default `"none"` | — |
| `subject_preset` | choice: `custom`, `hair`, `fur`, `cloth`, `skin_face`, `hard_edge`, `soft_glow` | default `"custom"` | Override trimap_dilate/erode/edge with subject-tuned values. |
| `trimap_dilate` | `INT` | default `8`, range 0…128, step 1 | — |
| `trimap_erode` | `INT` | default `8`, range 0…128, step 1 | — |
| `edge_radius` | `INT` | default `4`, range 0…64, step 1 | — |
| `edge_mode` | choice: `soft`, `feather`, `hard`, `hard+feather` | default `"soft"` | Final alpha edge. 'soft' = as the matter produced it. 'feather' = blur the existing edge. 'hard' = binary cut at edge_threshold, no partial alpha (what a holdout or garbage matte wants). 'hard+feather' = cut hard then feather — the edge width becomes yours rather than the model's, which is how you match edges across shots keyed by different backends. |
| `edge_threshold` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | Cut point for the hard modes. Lower keeps more of the soft fringe as solid; higher eats into it. |
| `edge_feather` | `FLOAT` | default `0.0`, range 0.0…64.0, step 0.5 | Feather radius in pixels, read as ~2 sigma of a true Gaussian. 0 with a feather mode selected does nothing. |
| `individual_objects` | `BOOLEAN` | default `False` | If supported by the backend, return one mask per detected object. |
| `tracking_direction` | choice: `forward`, `backward`, `bidirectional` | default `"forward"` | — |
| `frame_annotation` | `INT` | default `0`, range 0…100000 | Frame index (in clip) where prompts are anchored. |
| `object_id` | `INT` | default `0`, range 0…1024 | — |
| `max_frames_to_track` | `INT` | default `0`, range 0…100000 | 0 = no cap. |
| `memory_size` | `INT` | default `8`, range 1…256 | — |
| `start_frame` | `INT` | default `0`, range 0…100000 | — |
| `end_frame` | `INT` | default `-1`, range -1…100000 | -1 = last frame. |
| `auto_download` | `BOOLEAN` | default `False` | Allow Download: fetch a missing weight from Hugging Face / torch.hub into its models/ subfolder. Off: installed copies are used wherever they are (this pack, other packs, the HF cache); without one the model-free method runs and the info output says so. |
| `seed` | `INT` | default `0`, range 0…18446744073709551615 | — |
| `tta_flip` | `BOOLEAN` | default `False` | Test-time augmentation: run segmenter on the H-flipped image and average. Slower but cleaner. |
| `multiscale` | `BOOLEAN` | default `False` | Run the segmenter at 0.75x / 1.0x / 1.25x and fuse. Helps small / thin subjects. |
| `post_refine` | choice: `none`, `guided`, `crf`, `crf+guided` | default `"none"` | Final alpha refinement. 'guided' = guided filter (fast, torch-only). 'crf' = DenseCRF (requires pydensecrf, sharpest edges). |
| `refine_radius` | `INT` | default `8`, range 1…64 | Spatial radius for guided / CRF refinement. |
| `refine_iterations` | `INT` | default `5`, range 1…30 | CRF inference iterations. |
| `despill` | choice: `off`, `green`, `blue`, `red`, `magenta`, `cyan`, `yellow`, `white`, `black`, `auto` | default `"off"` | Colour decontamination on the named backing. 'auto' estimates the colour from image corners. |
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
| `pipeline` | choice: `onyx`, `cascade (legacy)` | default `"onyx"` | onyx = SAM3.1 video session + tiled ViTMatte. cascade (legacy) = previous multi-backend auto_best path. |
| `scene_prompts` | `STRING` | default `""`, multiline | Optional ONYX scene JSON (version 1). When empty, legacy prompt sockets map to object 1. |
| `band_scale` | `FLOAT` | default `1.0`, range 0.1…5.0, step 0.05 | Resolution-aware trimap band scale (ONYX path). |
| `matte_tile` | `INT` | default `1024`, range 256…4096, step 64 | ViTMatte tile size in pixels for both the ONYX and cascade pipelines. |
| `temporal_stabilise` | `BOOLEAN` | default `True` | 3-tap temporal median on alpha inside the unknown trimap band (ONYX path). |
| `matte_overlap` | `INT` | default `64`, range 16…256, step 1 | Overlap between ViTMatte tiles in pixels. Larger overlaps reduce seams but run more tiles. |
| `matte_tile_batch` | `INT` | default `4`, range 1…16, step 1 | How many tiles to run per GPU forward pass. Lower this if you hit out-of-memory errors. On the CPU tiles always run one at a time (faster there, and a third of the memory). |
| `backend` | choice: `segment + matte`, `SAM + ViTMatte`, `SeC + MatAnyone2`, `remove background` | default `"segment + matte"` | Which pipeline makes the matte. 'segment + matte' = this node's segmenter and matter (pipeline, segmenter, matter). 'SAM + ViTMatte' = a SAM model from a SAM Model Loader, then ViTMatte or a guided filter on the edges. 'SeC + MatAnyone2' = SeC / SAM segmentation, then MatAnyone2 (video) or ViTMatte. 'remove background' = RMBG-2.0 or BiRefNet, no prompt. Each control's tooltip names the backend that reads it. |
| `sam_model` | `SAM_MODEL` |  | SAM + ViTMatte: the model from a SAM Model Loader. |
| `points_json` | `STRING` | default `"[]"`, multiline | SAM + ViTMatte and SeC + MatAnyone2: point prompts: [{"x":100,"y":200,"label":1}, ...] (label 1 = keep, 0 = remove). The click editor writes this. |
| `bbox_json` | `STRING` | default `""` | SAM + ViTMatte and SeC + MatAnyone2: box prompt [x1,y1,x2,y2]. Empty = points only. A wired pos_bbox wins. |
| `sam_iterations` | `INT` | default `2`, range 1…5, step 1 | SAM + ViTMatte: SAM passes; each pass uses the previous mask to place better prompts. 2-3 is ideal. |
| `sam_refine_method` | choice: `auto`, `vitmatte`, `guided_filter`, `multi_scale_guided`, `color_aware`, `laplacian_blend` | default `"auto"` | SAM + ViTMatte: edge refinement. auto = ViTMatte, else multi-scale guided filter, else guided filter. vitmatte = neural matting. guided_filter = fast. multi_scale_guided = three scales (best without a model). color_aware = LAB colour edges. laplacian_blend = frequency blend. |
| `sam_edge_radius` | `INT` | default `12`, range 1…200, step 1 | SAM + ViTMatte: pixels around the edge that are refined (larger = softer transitions). |
| `sam_detail_preservation` | `FLOAT` | default `0.85`, range 0.0…1.0, step 0.05 | SAM + ViTMatte: fine detail (hair, fur, lace) to keep. 0 = smooth, 1 = all. |
| `sam_edge_contrast` | `FLOAT` | default `1.0`, range 0.0…3.0, step 0.1 | SAM + ViTMatte: edge contrast for hard lighting. Above 1 sharpens the boundary. |
| `sam_multimask_output` | `BOOLEAN` | default `True` | SAM + ViTMatte: SAM returns three candidate masks (off = its single best). |
| `sam_mask_index` | `INT` | default `0`, range 0…2 | SAM + ViTMatte: which of SAM's three candidates to keep (with multimask output on). |
| `sam_score_threshold` | `FLOAT` | default `0.0`, range 0.0…1.0, step 0.01 | SAM + ViTMatte: skip SAM candidates below this confidence (0 = use mask_index). |
| `sam_trimap_dilate` | `INT` | default `0`, range 0…200, step 1 | SAM + ViTMatte: outer trimap radius (0 = edge radius x 1.5). |
| `sam_trimap_erode` | `INT` | default `0`, range 0…200, step 1 | SAM + ViTMatte: inner trimap radius (0 = edge radius x 1.0). |
| `sam_batch_mode` | `BOOLEAN` | default `False` | SAM + ViTMatte: process every frame of the batch. Off = the first frame only (every output then has one frame). |
| `sec_model` | choice: `sam2.1_hiera_base_plus`, `sam2.1_hiera_large`, `sam2.1_hiera_small`, `sam2.1_hiera_tiny`, `sam2_hiera_base_plus`, `sam2_hiera_large`, `sam2_hiera_small`, `sam2_hiera_tiny`, `sam3`, `sec_4b` | default `"sam2.1_hiera_base_plus"` | SeC + MatAnyone2: segmentation model. SeC = video with text prompts; SAM 2 / 3 = point or box prompts. |
| `sec_text_prompt` | `STRING` | default `""` | SeC + MatAnyone2: what to track, e.g. 'cat' or 'person in red'. A wired text_prompt wins. Empty = point / box prompts. |
| `sec_matting_backend` | choice: `matanyone2`, `vitmatte_small`, `vitmatte_base`, `auto` | default `"auto"` | SeC + MatAnyone2: alpha matting. auto = MatAnyone2 for video (more than one frame), ViTMatte small for an image. MatAnyone2 keeps the matte steady over time; ViTMatte has the best edge per frame. |
| `sec_edge_radius` | `INT` | default `15`, range 1…200, step 1 | SeC + MatAnyone2: edge refinement radius in pixels. |
| `sec_n_warmup` | `INT` | default `5`, range 1…30, step 1 | SeC + MatAnyone2: MatAnyone2 warm-up passes on the first frame (more = steadier start). |
| `sec_edge_refine_method` | choice: `none`, `vitmatte`, `guided_filter`, `multi_scale_guided` | default `"none"` | SeC + MatAnyone2: optional edge refinement after matting. none = the matting output as it is. |
| `bg_model` | choice: `birefnet_general`, `birefnet_portrait`, `rmbg_2.0` | default `"birefnet_general"` | remove background: rmbg_2.0 = general purpose, fast. birefnet_general = detailed edges. birefnet_portrait = people. |
| `bg_threshold` | `FLOAT` | default `0.5`, range 0.0…1.0, step 0.01 | remove background: above 0 the mask is cut hard at this value (0 = keep the soft mask). |
| `bg_invert` | `BOOLEAN` | default `False` | remove background: invert the mask (keep the background instead). |
| `bg_mask_blur` | `INT` | default `0`, range 0…50, step 1 | remove background: Gaussian blur on the mask edge, in pixels (0 = none). |
| `fill_holes_enabled` | `BOOLEAN` | default `True` | SAM + ViTMatte and SeC + MatAnyone2: fill holes inside the mask. |
| `min_region_size` | `INT` | default `64`, range 0…10000, step 1 | SAM + ViTMatte and SeC + MatAnyone2: remove separate regions smaller than this many pixels (0 = keep all). |
| `keep_model_loaded` | `BOOLEAN` | default `True` | SeC + MatAnyone2 and remove background: keep the models in memory between runs (off = free them after each run). |
| `split_masks` | `BOOLEAN` | default `True` | Make the edge / inside / outside masks (and, for the three other backends, a trimap from the final alpha). Off leaves those outputs empty, which saves three to four full-size masks per frame on long clips. |

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
| 17 | `object_alphas` | `MASK` | Per-object alpha mattes stacked object-major as (O*B,H,W). |
| 18 | `foreground` | `IMAGE` | The image times the final alpha (premultiplied), despilled when despill is on. |


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
| `source_choice` | choice: `auto`, `image`, `video`, `exr`, `custom` | default `"auto"` | Where the source name comes from. 'image' → trigger_image, 'video' → trigger_video, 'auto' → prefer video if connected, else image, else legacy `trigger`. 'exr' → prefer an EXR/DPX/CIN plate (sequence or single); works WITHOUT a trigger — wire the plate path STRING into `source_path` and the stem is derived from it. 'custom' → use the ``custom_name`` widget verbatim and ignore all triggers. |
| `name_format` | choice: `basename`, `strip_tags`, `first_segment` | default `"basename"` | How to format the detected filename for folder + prefix:   basename      — strip extension only (e.g. clip_2160_25fps)   strip_tags    — also strip trailing res/fps tags (clip)   first_segment — keep only the first chunk before . or _ (clip) The original file extension is preserved on output_filename. |
| `numbered_still_mode` | choice: `auto`, `sequence`, `identity` | default `"auto"` | What a trailing number on a STILL means (png/exr/dpx/tif...):   auto     — >=4 digits is a frame index and is stripped              (plate.0001.png -> plate); <=3 digits is a shot              number and is KEPT (shot_010.png -> shot_010).   sequence — always strip a trailing 2-8 digit number. Use when              your frames are bare-numbered (plate_01.png ...).   identity — never strip bare digits; only #### and %04d go. #### and %04d are stripped in every mode — they are never part of a real name. auto errs toward KEEPING the number, because over-grouping silently mixes unrelated plates into one sequence, while over-splitting is merely inconvenient and visible. |

**Optional inputs**

| Parameter | Type | Constraints | What it does |
|---|---|---|---|
| `trigger` | `*` |  | Legacy generic trigger – connect any output here. |
| `trigger_image` | `IMAGE` |  | Connect a LoadImage / image source here. Used when source_choice = 'image' or 'auto'. |
| `trigger_video` | `*` |  | Connect a LoadVideo / VHS_LoadVideo / video source here. Used when source_choice = 'video' or 'auto' (preferred). |
| `source_filename` | `STRING` | default `""` | Auto-filled from the connected loader (basename only, no extension). Drives folder name + output basename. Extension is stored separately in source_extension for output_filename. May also be a FULL PATH — the basename is extracted automatically. |
| `source_extension` | `STRING` | default `""` | Auto-filled file extension from the connected loader (e.g. .mp4, .mov). Used only for output_filename — not part of the folder name. |
| `source_path` | `STRING` | default `""` | Full path to a source media file (image / video / EXR / DPX sequence). When provided, the node extracts the filename stem from this path and uses it as the source name — this takes PRECEDENCE over source_filename and custom_name. Connect a STRING output here (e.g. an OCIORead 'source' widget, or any loader that exposes its file path as STRING). The extracted stem is returned on the `source_stem` output, and a trailing version token (e.g. _v001) is incremented to _v002 on the `next_version_token` / `next_version_stem` outputs. |
| `custom_name` | `STRING` | default `""` | Manual source name. Only used when source_choice='custom'. May include an extension (e.g. 'my_shot.mp4'); if no extension is given, output_filename will have none either. Sanitized for cross-platform safety. |
| `base_path` | `STRING` | default `""` | Override base output directory.  Leave empty → ComfyUI output dir. |
| `folder_name_override` | `STRING` | default `""` | Force a specific folder name instead of deriving from the input filename. Sanitized for cross-platform safety. |
| `version_group` | `STRING` | default `""` | Which outputs share one version number.  LEAVE IT EMPTY (the normal case). Every incrementer in the same run whose folder is the same shot - including its _mask, _exr and _png variants - is handed the SAME version, so one render files as one version. An unrelated job in the same workflow keeps its own counter and is never dragged along.  Set a name here only when two outputs belong together but their folder names do not say so. Incrementers sharing a name share a version, whatever their folders. |
| `reserve_version` | `BOOLEAN` | default `False` | If True, create the version directory and write a `.reserved` marker file to claim the version number atomically. Prevents collisions in batch/render-farm workflows. Leave False for normal use (the directory will be created by ComfyUI's Save node when output is actually written). |
| `layout` | choice: `source / date / version`, `show / shot / task / version` | default `"source / date / version"` | Folder layout.   source / date / version  - <base>/<source name>/<date>/v### (the usual one)   show / shot / task / version - <base>/<show>/<shot>/<task>/v###, the pipeline layout: no date folder, versions always v###, and with reserve_version each version is claimed with an exclusive .lock (retrying when another job took it) plus an optional version_manifest.json. |
| `show` | `STRING` | default `"show"` | show / shot / task layout: show or project name (top folder under base). |
| `shot` | `STRING` | default `"sh010"` | show / shot / task layout: shot identifier (folder under show). |
| `task` | `STRING` | default `"comp"` | show / shot / task layout: task name (folder under shot, e.g. comp, matte). |
| `min_version` | `INT` | default `1`, range 0…999999 | show / shot / task layout: never hand out a version below this. |
| `max_retries` | `INT` | default `5`, range 1…100 | show / shot / task layout, reserve_version on: how many next numbers to try when another job claimed the version first. |
| `write_manifest` | `BOOLEAN` | default `True` | show / shot / task layout, reserve_version on: also write version_manifest.json (workflow hash, user, host, time) into the reserved folder. |
| `versioning` | `COMBO` | default `folder`; folder, filename | Where the version goes. folder: a version folder, shot_A/<date>/v001/shot_A.png (the usual one). filename: on the file, no version folder, shot_A/<date>/shot_A_v001.png - the next number no file (or version folder) of this name has yet. |
| `version_in_filename` | `BOOLEAN` | default `False` | folder versioning: also put the version on the file name - shot_A/<date>/v001/shot_A_v001.png. (Filename versioning always does.) |
| `output_name` | `STRING` | default `""` | Optional name for the output FILE only - the folder keeps the source-derived name (final_render -> shot_A/<date>/v001/final_render.png). May carry an extension; otherwise the source's extension is kept. |

**Outputs**

| # | Name | Type | What it is |
|---|---|---|---|
| 0 | `version_string` | `STRING` | Zero-padded version string, e.g. `v001`. |
| 1 | `version_number` | `INT` | Raw integer version number. |
| 2 | `folder_name` | `STRING` | Sanitized folder name derived from the input source (or label). |
| 3 | `subfolder_path` | `STRING` | Full subfolder path: `<base>/<folder>/<date>/<version>`. |
| 4 | `filename_prefix` | `STRING` | Filename prefix combining folder + version (for SaveImage). |
| 5 | `output_filename` | `STRING` | Final output filename including extension. |
| 6 | `source_stem` | `STRING` | Filename stem extracted from the connected source (extension and frame token stripped). e.g. `B_0151C002_260527_134258_a1IE7_v001`. |
| 7 | `current_version_token` | `STRING` | Version token found at the end of the source stem, e.g. `v001`. Empty when the source name has no trailing version token. |
| 8 | `next_version_token` | `STRING` | Source version token incremented by one, e.g. `v002`. Empty when the source name has no trailing version token. |
| 9 | `next_version_stem` | `STRING` | Source stem with its version token incremented, e.g. `B_0151C002_260527_134258_a1IE7_v002`. Empty when the source name has no trailing version token. |
| 10 | `version_path` | `STRING` | Absolute path of the version folder (show / shot / task layout: the folder it allocated, forward slashes unless path_style says otherwise). |
| 11 | `info_json` | `STRING` | JSON: what was allocated - folder, version, paths, whether it was reserved. |


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
