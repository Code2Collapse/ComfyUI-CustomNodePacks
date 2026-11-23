# ComfyUI-CustomNodePacks — documentation index

This folder holds the user guides for the nodes of this pack. The top-level [README](../README.md) is the landing page.

- **A guide** (the pages below) says what a node is for, when to pick it over a similar node, a minimal wiring, the
  key controls, how to read its info / report outputs, and common mistakes.
- **[NODE_REFERENCE.md](../NODE_REFERENCE.md)** lists every input and output with its default and tooltip. It is
  generated from the code, so the guides link to it instead of repeating parameter tables. Each "ref" link below
  goes to the node's entry there.

Every live node has exactly one guide section. Status words used in the guides: **EXPERIMENTAL** (off by default or
labelled so in the code), **NOT VERIFIED** (not run on a GPU, with real weights, or in the browser - the guide says
which). A guide never claims a quality gain without a measurement in the repository; where it cites a number, it
names the commit or evidence file.

---

## Which node do I want?

### Masks and mattes

| You want to | Use | Guide |
|---|---|---|
| Cut out a subject, image or clip, in one node: find it with a model, then pull a soft edge with a matting model (interactive viewer: find, click, preview, render, QC) | **Mask + Matting** (`MaskOpsMEC`) | [mask-matting-pipeline](mask-matting-pipeline.md) |
| Find objects by words (`person:3, dog`), clicks or boxes with ComfyUI's own SAM 3 / 3.1; one mask per object; follow them through a clip | **Segment (C2C)** (`SegmentC2C`) | [sam-segmentation](sam-segmentation.md) |
| One object in a still from points, a box, several boxes, a spline outline or a text prompt (SAM 2.1 / SAM 3) | **SAM Mask Generator** (+ SAM Model Loader) | [sam-segmentation](sam-segmentation.md) |
| See SAM's three guesses for a click and choose one by eye | **SAM Multi-Mask Picker** | [sam-segmentation](sam-segmentation.md) |
| Face parts or clothes by class name | **Semantic Segment**, or Segment (C2C) with `target` = `face parts` / `clothes` | [sam-segmentation](sam-segmentation.md) |
| Key, grade, grow / shrink one side, combine two mattes, make a gradient, add grain or motion blur - no model | **Mask Tools (C2C)** (`MaskToolsMEC`) | [mask-editing](mask-editing.md) |
| Clean a mask you already have (holes, specks, a bleeding edge) - no model weights | **Mask Refiner** (`MaskRefineMEC`) | [matting-refinement](matting-refinement.md) |
| Paint or draw the mask yourself (brush, shapes, polygon, spline, roto, points + box) | **Image Mask Editor (C2C)** | [image-mask-editor](image-mask-editor.md) |
| Spread a mask through a clip, morph between keyframe masks, score flicker, smooth a mask batch | **Mask Track** (`MaskTrackerMEC`) | [video-temporal-bbox](video-temporal-bbox.md) |
| Put a shape onto a shot with a 4-corner perspective quad and keep it there as the camera moves | **Mask Placement** | [mask-placement](mask-placement.md) |

Mask + Matting **makes** a matte; Mask Refiner and Mask Tools **change** one you already have. Chain them when you need
both (Mask + Matting `alpha` -> Mask Refiner `mask`).

### Loading video and frames

| You have | Use | Guide |
|---|---|---|
| A movie file, or a numbered sequence where the frame numbers matter (gaps, a range like `1001-1100`, an EXR layer, a colour space to convert), or frames you want to resize / retime / crop / put handles on / VAE-encode while loading | **Load Video (C2C)** | [video-io](video-io.md) |
| A folder to read as "every image, in name order" - no frame numbers, no audio, no resize | **Load Images Path (C2C)** | [video-io](video-io.md) |

### Inpainting

| You want to | Use | Guide |
|---|---|---|
| Inpaint a region of an image, or of each frame of a clip, with your own inpaint model, at the region's own resolution, then blend it back | **Inpaint Crop Pro** -> your sampler -> **Inpaint Stitch Pro** | [inpaint-suite](inpaint-suite.md) |
| Remove an object from a video by filling the hole from the clip's other frames (nothing generative), or blend an inpainted region back over time | **ProPainter** (`ProPainterMEC`) - its `temporal` / `remove` modes need the ProPainter code and weights (non-commercial licence, not shipped) or `backend` = the model-free OpenCV fill (APPROXIMATE) | [propainter](propainter.md) |

---

## Every node, by menu category

All nodes sit under **🐺 C2C/🧰 Core/** in the node menu. The categories below follow the README's node list.

### AI Image

| Node | id | Guide | Parameters |
|---|---|---|---|
| Nano Banana · Gemini Image (C2C) | `NanoBananaC2C` | [nano-banana](./nano-banana.md#nano-banana--gemini-image-c2c-nanobananac2c) | [ref](../NODE_REFERENCE.md#nanobananac2c) |

### Audio

| Node | id | Guide | Parameters |
|---|---|---|---|
| Audio Reverser | `AudioReverserMEC` | [fluid-shots-audio](./fluid-shots-audio.md#audio-reverser-audioreversermec) | [ref](../NODE_REFERENCE.md#audioreversermec) |

### Color Science

| Node | id | Guide | Parameters |
|---|---|---|---|
| VAE Decode (C2C) | `C2CVAEQualityDecode` | [vae-precision](./vae-precision.md#vae-decode-c2c-c2cvaequalitydecode) | [ref](../NODE_REFERENCE.md#c2cvaequalitydecode) |

### Control

| Node | id | Guide | Parameters |
|---|---|---|---|
| Control AOV — Multi-Control Fusion (C2C) | `ControlAOVC2C` | [control-aov](./control-aov.md#control-aov--multi-control-fusion-c2c-controlaovc2c) | [ref](../NODE_REFERENCE.md#controlaovc2c) |

### Diagnostics

| Node | id | Guide | Parameters |
|---|---|---|---|
| Mask Failure Explainer — Diagnostics | `MaskFailureExplainerMEC` | [utility-nodes](./utility-nodes.md#mask-failure-explainer--diagnostics-maskfailureexplainermec) | [ref](../NODE_REFERENCE.md#maskfailureexplainermec) |
| Model Metadata Extractor | `ModelMetadataExtractorMEC` | [utility-nodes](./utility-nodes.md#model-metadata-extractor-modelmetadataextractormec) | [ref](../NODE_REFERENCE.md#modelmetadataextractormec) |

### Distributed

| Node | id | Guide | Parameters |
|---|---|---|---|
| C2C GPU Inventory | `C2CGPUInventory` | [c2c-distributed](./c2c-distributed.md#c2c-gpu-inventory-c2cgpuinventory) | [ref](../NODE_REFERENCE.md#c2cgpuinventory) |
| C2C Multi-GPU Split | `C2CSequenceParallel` | [c2c-distributed](./c2c-distributed.md#c2c-multi-gpu-split-c2csequenceparallel) | [ref](../NODE_REFERENCE.md#c2csequenceparallel) |

### Face

| Node | id | Guide | Parameters |
|---|---|---|---|
| Face Landmarks (Facepoints, 620 + 68/98) | `FacepointsLandmarksC2C` | [face-landmarks](./face-landmarks.md#face-landmarks-facepoints-620--6898-facepointslandmarksc2c) | [ref](../NODE_REFERENCE.md#facepointslandmarksc2c) |

### Frequency

| Node | id | Guide | Parameters |
|---|---|---|---|
| Film Grain (MEC) | `FilmGrainMEC` | [frequency-grain](./frequency-grain.md#film-grain-mec-filmgrainmec) | [ref](../NODE_REFERENCE.md#filmgrainmec) |
| Frequency Combine (MEC) | `FrequencyCombineMEC` | [frequency-grain](./frequency-grain.md#frequency-combine-mec-frequencycombinemec) | [ref](../NODE_REFERENCE.md#frequencycombinemec) |
| Frequency Separate (MEC) | `FrequencySeparateMEC` | [frequency-grain](./frequency-grain.md#frequency-separate-mec-frequencyseparatemec) | [ref](../NODE_REFERENCE.md#frequencyseparatemec) |

### Helpers

| Node | id | Guide | Parameters |
|---|---|---|---|
| Batch Range (C2C) | `ImageBatchSliceMEC` | [utility-nodes](./utility-nodes.md#batch-range-c2c-imagebatchslicemec) | [ref](../NODE_REFERENCE.md#imagebatchslicemec) |
| Conditional Switch | `ConditionalSwitchMEC` | [utility-nodes](./utility-nodes.md#conditional-switch-conditionalswitchmec) | [ref](../NODE_REFERENCE.md#conditionalswitchmec) |
| Execution Timer (C2C) | `ExecutionTimerMEC` | [utility-nodes](./utility-nodes.md#execution-timer-c2c-executiontimermec) | [ref](../NODE_REFERENCE.md#executiontimermec) |
| Number Lerp (C2C) | `NumberLerpMEC` | [utility-nodes](./utility-nodes.md#number-lerp-c2c-numberlerpmec) | [ref](../NODE_REFERENCE.md#numberlerpmec) |
| Probe (C2C) | `ImageStatsProbeMEC` | [utility-nodes](./utility-nodes.md#probe-c2c-imagestatsprobemec) | [ref](../NODE_REFERENCE.md#imagestatsprobemec) |
| Seed List Generator (C2C) | `SeedListMEC` | [utility-nodes](./utility-nodes.md#seed-list-generator-c2c-seedlistmec) | [ref](../NODE_REFERENCE.md#seedlistmec) |
| Size (C2C) | `AspectPresetMEC` | [utility-nodes](./utility-nodes.md#size-c2c-aspectpresetmec) | [ref](../NODE_REFERENCE.md#aspectpresetmec) |
| Text Template (C2C) | `TextTemplateMEC` | [utility-nodes](./utility-nodes.md#text-template-c2c-texttemplatemec) | [ref](../NODE_REFERENCE.md#texttemplatemec) |

### Inpaint

| Node | id | Guide | Parameters |
|---|---|---|---|
| Inpaint Crop Pro | `InpaintCropProMEC` | [inpaint-suite](./inpaint-suite.md#inpaint-crop-pro-inpaintcroppromec) | [ref](../NODE_REFERENCE.md#inpaintcroppromec) |
| Inpaint Stitch Pro | `InpaintStitchProMEC` | [inpaint-suite](./inpaint-suite.md#inpaint-stitch-pro-inpaintstitchpromec) | [ref](../NODE_REFERENCE.md#inpaintstitchpromec) |
| ProPainter — Temporal / Remove / Stitch / Refine / Flow | `ProPainterMEC` | [propainter](./propainter.md#propainter-propaintermec) | [ref](../NODE_REFERENCE.md#propaintermec) |

### Layer Effects

| Node | id | Guide | Parameters |
|---|---|---|---|
| Layer Effects (C2C) | `LayerEffectsMEC` | [layer-effects](./layer-effects.md#layer-effects-c2c-layereffectsmec) | [ref](../NODE_REFERENCE.md#layereffectsmec) |

### Legacy

Deprecated, kept so saved workflows load. No guide: each has a migration page with its replacement and the measured parity.

| Node | id | Migration page | Parameters |
|---|---|---|---|
| BBox Crop (MEC) (legacy) | `BBoxCrop` | [migration/BBoxCrop.md](migration/BBoxCrop.md) | [ref](../NODE_REFERENCE.md#bboxcrop) |
| BBox From Mask (MEC) (legacy) | `BBoxFromMask` | [migration/BBoxFromMask.md](migration/BBoxFromMask.md) | [ref](../NODE_REFERENCE.md#bboxfrommask) |
| BBox Pad (MEC) (legacy) | `BBoxPad` | [migration/BBoxPad.md](migration/BBoxPad.md) | [ref](../NODE_REFERENCE.md#bboxpad) |
| BBox To Mask (MEC) (legacy) | `BBoxToMask` | [migration/BBoxToMask.md](migration/BBoxToMask.md) | [ref](../NODE_REFERENCE.md#bboxtomask) |
| Mask Composite Advanced (MEC) (legacy) | `MaskCompositeAdvanced` | [migration/MaskCompositeAdvanced.md](migration/MaskCompositeAdvanced.md) | [ref](../NODE_REFERENCE.md#maskcompositeadvanced) |
| Mask Math (MEC) (legacy) | `MaskMath` | [migration/MaskMath.md](migration/MaskMath.md) | [ref](../NODE_REFERENCE.md#maskmath) |
| Mask Preview Overlay (MEC) (legacy) | `MaskPreviewOverlay` | [migration/MaskPreviewOverlay.md](migration/MaskPreviewOverlay.md) | [ref](../NODE_REFERENCE.md#maskpreviewoverlay) |

### Magnific

| Node | id | Guide | Parameters |
|---|---|---|---|
| OmniScale Creation Picker | `MagnificCreationPicker` | [magnific-omniscale](./magnific-omniscale.md#omniscale-creation-picker-magnificcreationpicker) | [ref](../NODE_REFERENCE.md#magnificcreationpicker) |
| OmniScale Generate Image | `MagnificGenerateImage` | [magnific-omniscale](./magnific-omniscale.md#omniscale-generate-image-magnificgenerateimage) | [ref](../NODE_REFERENCE.md#magnificgenerateimage) |
| OmniScale Generate Music | `MagnificGenerateMusic` | [magnific-omniscale](./magnific-omniscale.md#omniscale-generate-music-magnificgeneratemusic) | [ref](../NODE_REFERENCE.md#magnificgeneratemusic) |
| OmniScale Generate Video | `MagnificGenerateVideo` | [magnific-omniscale](./magnific-omniscale.md#omniscale-generate-video-magnificgeneratevideo) | [ref](../NODE_REFERENCE.md#magnificgeneratevideo) |
| OmniScale Library Reference | `MagnificLibraryReference` | [magnific-omniscale](./magnific-omniscale.md#omniscale-library-reference-magnificlibraryreference) | [ref](../NODE_REFERENCE.md#magnificlibraryreference) |
| OmniScale Metadata (unpack) | `MagnificMetadata` | [magnific-omniscale](./magnific-omniscale.md#omniscale-metadata-unpack-magnificmetadata) | [ref](../NODE_REFERENCE.md#magnificmetadata) |
| OmniScale Remove Background | `MagnificRemoveBackground` | [magnific-omniscale](./magnific-omniscale.md#omniscale-remove-background-magnificremovebackground) | [ref](../NODE_REFERENCE.md#magnificremovebackground) |
| OmniScale Retouch (inpaint) | `MagnificRetouch` | [magnific-omniscale](./magnific-omniscale.md#omniscale-retouch-inpaint-magnificretouch) | [ref](../NODE_REFERENCE.md#magnificretouch) |
| OmniScale Save To (project/folder) | `MagnificSaveTo` | [magnific-omniscale](./magnific-omniscale.md#omniscale-save-to-projectfolder-magnificsaveto) | [ref](../NODE_REFERENCE.md#magnificsaveto) |
| OmniScale Skin Enhancer | `MagnificSkinEnhancer` | [magnific-omniscale](./magnific-omniscale.md#omniscale-skin-enhancer-magnificskinenhancer) | [ref](../NODE_REFERENCE.md#magnificskinenhancer) |
| OmniScale Stock Picker | `MagnificStockPicker` | [magnific-omniscale](./magnific-omniscale.md#omniscale-stock-picker-magnificstockpicker) | [ref](../NODE_REFERENCE.md#magnificstockpicker) |
| OmniScale Stock Search | `MagnificStockSearch` | [magnific-omniscale](./magnific-omniscale.md#omniscale-stock-search-magnificstocksearch) | [ref](../NODE_REFERENCE.md#magnificstocksearch) |
| OmniScale Upscale Image | `MagnificUpscaleImage` | [magnific-omniscale](./magnific-omniscale.md#omniscale-upscale-image-magnificupscaleimage) | [ref](../NODE_REFERENCE.md#magnificupscaleimage) |
| OmniScale Upscale Video | `MagnificUpscaleVideo` | [magnific-omniscale](./magnific-omniscale.md#omniscale-upscale-video-magnificupscalevideo) | [ref](../NODE_REFERENCE.md#magnificupscalevideo) |
| OmniScale Voiceover (TTS) | `MagnificVoiceover` | [magnific-omniscale](./magnific-omniscale.md#omniscale-voiceover-tts-magnificvoiceover) | [ref](../NODE_REFERENCE.md#magnificvoiceover) |

### Mask

| Node | id | Guide | Parameters |
|---|---|---|---|
| Image Mask Editor (C2C) | `ImageMaskEditorC2C` | [image-mask-editor](./image-mask-editor.md#image-mask-editor-imagemaskeditorc2c) | [ref](../NODE_REFERENCE.md#imagemaskeditorc2c) |
| Mask Placement — Prompt/Ref → Place → Track | `MaskPlacementMEC` | [mask-placement](./mask-placement.md#mask-placement--promptref--place--track-maskplacementmec) | [ref](../NODE_REFERENCE.md#maskplacementmec) |
| Mask Tools (C2C) | `MaskToolsMEC` | [mask-editing](./mask-editing.md#mask-tools-masktoolsmec) | [ref](../NODE_REFERENCE.md#masktoolsmec) |

### Model Analysis

| Node | id | Guide | Parameters |
|---|---|---|---|
| VAE Inspect | `VAEBlockInspectorMEC` | [vae-merge](./vae-merge.md#vae-inspect-vaeblockinspectormec) | [ref](../NODE_REFERENCE.md#vaeblockinspectormec) |

### Paint

| Node | id | Guide | Parameters |
|---|---|---|---|
| Face Fixer | `MECFaceFixer` | [face-fixer](./face-fixer.md#face-fixer-mecfacefixer) | [ref](../NODE_REFERENCE.md#mecfacefixer) |

### Pipeline

| Node | id | Guide | Parameters |
|---|---|---|---|
| Mask + Matting | `MaskOpsMEC` | [mask-matting-pipeline](./mask-matting-pipeline.md#mask--matting-maskopsmec) | [ref](../NODE_REFERENCE.md#maskopsmec) |
| Mask Refiner | `MaskRefineMEC` | [matting-refinement](./matting-refinement.md#mask-refiner-maskrefinemec) | [ref](../NODE_REFERENCE.md#maskrefinemec) |

### Plate

| Node | id | Guide | Parameters |
|---|---|---|---|
| PAR Desqueeze (C2C, anamorphic → square px) | `PARDesqueezeMEC` | [anamorphic-par](./anamorphic-par.md#par-desqueeze-pardesqueezemec) | [ref](../NODE_REFERENCE.md#pardesqueezemec) |
| PAR Resqueeze (C2C, back to plate) | `PARResqueezeMEC` | [anamorphic-par](./anamorphic-par.md#par-resqueeze-parresqueezemec) | [ref](../NODE_REFERENCE.md#parresqueezemec) |

### Preview

| Node | id | Guide | Parameters |
|---|---|---|---|
| Video Comparer — Player + Wipe/Diff/Scopes | `VideoComparerC2C` | [video-comparer](./video-comparer.md#video-comparer--player--wipediffscopes-videocomparerc2c) | [ref](../NODE_REFERENCE.md#videocomparerc2c) |
| Video Frame Player | `VideoFramePlayerMEC` | [video-frame-player](./video-frame-player.md#video-frame-player-videoframeplayermec) | [ref](../NODE_REFERENCE.md#videoframeplayermec) |

### Prompt Relay

| Node | id | Guide | Parameters |
|---|---|---|---|
| Prompt Relay Advanced Options | `PromptRelayAdvancedOptionsC2C` | [prompt-relay](./prompt-relay.md#prompt-relay-advanced-options-promptrelayadvancedoptionsc2c) | [ref](../NODE_REFERENCE.md#promptrelayadvancedoptionsc2c) |
| Prompt Relay Encode | `PromptRelayEncodeC2C` | [prompt-relay](./prompt-relay.md#prompt-relay-encode-promptrelayencodec2c) | [ref](../NODE_REFERENCE.md#promptrelayencodec2c) |
| Prompt Relay Restore (Kijai) | `PromptRelayRestoreKijaiC2C` | [prompt-relay](./prompt-relay.md#prompt-relay-restore-kijai-promptrelayrestorekijaic2c) | [ref](../NODE_REFERENCE.md#promptrelayrestorekijaic2c) |

### Render Farm

| Node | id | Guide | Parameters |
|---|---|---|---|
| C2C Farm Cluster Status | `C2C_ClusterStatus` | [render-farm](./render-farm.md#c2c-farm-cluster-status-c2c_clusterstatus) | [ref](../NODE_REFERENCE.md#c2c_clusterstatus) |
| C2C Farm Job History (Audit Log) | `C2C_JobHistory` | [render-farm](./render-farm.md#c2c-farm-job-history-audit-log-c2c_jobhistory) | [ref](../NODE_REFERENCE.md#c2c_jobhistory) |
| C2C Farm Submit — Remote Render | `C2C_Submit` | [render-farm](./render-farm.md#c2c-farm-submit--remote-render-c2c_submit) | [ref](../NODE_REFERENCE.md#c2c_submit) |

### SAM

| Node | id | Guide | Parameters |
|---|---|---|---|
| SAM Mask Generator — Points + BBox + Text | `SAMMaskGeneratorMEC` | [sam-segmentation](./sam-segmentation.md#sam-mask-generator--points--bbox--text-sammaskgeneratormec) | [ref](../NODE_REFERENCE.md#sammaskgeneratormec) |
| SAM Model Loader — SAM2.1 / SAM3 | `SAMModelLoaderMEC` | [sam-segmentation](./sam-segmentation.md#sam-model-loader--sam21--sam3-sammodelloadermec) | [ref](../NODE_REFERENCE.md#sammodelloadermec) |
| SAM Multi-Mask Picker — 3 candidates + scores | `SamMultiMaskPickerMEC` | [sam-segmentation](./sam-segmentation.md#sam-multi-mask-picker--3-candidates--scores-sammultimaskpickermec) | [ref](../NODE_REFERENCE.md#sammultimaskpickermec) |

### Sampling

| Node | id | Guide | Parameters |
|---|---|---|---|
| AsymFlow Sampler Patch (Lakonik signal-shift) | `AsymFlowSamplerPatch` | [asymflow-sampler](./asymflow-sampler.md#asymflow-sampler-patch-lakonik-signal-shift-asymflowsamplerpatch) | [ref](../NODE_REFERENCE.md#asymflowsamplerpatch) |

### Segmentation

| Node | id | Guide | Parameters |
|---|---|---|---|
| LocateAnything Grounding (MEC) | `LocateAnythingGroundingMEC` | [sam-segmentation](./sam-segmentation.md#locateanything-grounding-mec-locateanythinggroundingmec) | [ref](../NODE_REFERENCE.md#locateanythinggroundingmec) |
| LocateAnything → SAM Prompt (MEC) | `LocateAnythingToSAMMEC` | [sam-segmentation](./sam-segmentation.md#locateanything--sam-prompt-mec-locateanythingtosammec) | [ref](../NODE_REFERENCE.md#locateanythingtosammec) |
| Segment (C2C) | `SegmentC2C` | [sam-segmentation](./sam-segmentation.md#segment-c2c-segmentc2c) | [ref](../NODE_REFERENCE.md#segmentc2c) |
| Semantic Segment — Face / Clothes Parsing | `SemanticSegmentMEC` | [sam-segmentation](./sam-segmentation.md#semantic-segment--face--clothes-parsing-semanticsegmentmec) | [ref](../NODE_REFERENCE.md#semanticsegmentmec) |

### Stabilization

| Node | id | Guide | Parameters |
|---|---|---|---|
| Video Stabilizer | `VideoStabilizerMEC` | [video-stabilizer](./video-stabilizer.md#video-stabilizer-videostabilizermec) | [ref](../NODE_REFERENCE.md#videostabilizermec) |

### Temporal

| Node | id | Guide | Parameters |
|---|---|---|---|
| Fluid Shot Decoder (Restore Timing) | `FluidShotDecoderMEC` | [fluid-shots-audio](./fluid-shots-audio.md#fluid-shot-decoder-fluidshotdecodermec) | [ref](../NODE_REFERENCE.md#fluidshotdecodermec) |
| Fluid Shot Encoder (Temporal Normalizer) | `FluidShotEncoderMEC` | [fluid-shots-audio](./fluid-shots-audio.md#fluid-shot-encoder-fluidshotencodermec) | [ref](../NODE_REFERENCE.md#fluidshotencodermec) |

### Tiling

| Node | id | Guide | Parameters |
|---|---|---|---|
| Identity Lock — refine without moving | `IdentityLockMEC` | [tiled-refinement](./tiled-refinement.md#identity-lock--refine-without-moving-identitylockmec) | [ref](../NODE_REFERENCE.md#identitylockmec) |
| Tile Merge | `TileMergeMEC` | [tiled-refinement](./tiled-refinement.md#tile-merge-tilemergemec) | [ref](../NODE_REFERENCE.md#tilemergemec) |
| Tile Plan | `TilePlanMEC` | [tiled-refinement](./tiled-refinement.md#tile-plan-tileplanmec) | [ref](../NODE_REFERENCE.md#tileplanmec) |
| Tile Split | `TileSplitMEC` | [tiled-refinement](./tiled-refinement.md#tile-split-tilesplitmec) | [ref](../NODE_REFERENCE.md#tilesplitmec) |

### Utils

| Node | id | Guide | Parameters |
|---|---|---|---|
| Folder Version Check (deprecated) | `FolderIncrementerReset` | [utility-nodes](./utility-nodes.md#folder-version-check-deprecated-folderincrementerreset) | [ref](../NODE_REFERENCE.md#folderincrementerreset) |
| Folder Version Control (check / set) | `FolderVersionControl` | [utility-nodes](./utility-nodes.md#folder-version-control-check--set-folderversioncontrol) | [ref](../NODE_REFERENCE.md#folderversioncontrol) |
| Folder Version Incrementer | `FolderIncrementer` | [utility-nodes](./utility-nodes.md#folder-version-incrementer-folderincrementer) | [ref](../NODE_REFERENCE.md#folderincrementer) |
| Folder Version Set (deprecated) | `FolderIncrementerSet` | [utility-nodes](./utility-nodes.md#folder-version-set-deprecated-folderincrementerset) | [ref](../NODE_REFERENCE.md#folderincrementerset) |
| Folder Version Variant (per output) | `FolderVersionVariant` | [utility-nodes](./utility-nodes.md#folder-version-variant-per-output-folderversionvariant) | [ref](../NODE_REFERENCE.md#folderversionvariant) |
| Parameter History | `ParameterHistoryMEC` | [utility-nodes](./utility-nodes.md#parameter-history-parameterhistorymec) | [ref](../NODE_REFERENCE.md#parameterhistorymec) |

### VAE

| Node | id | Guide | Parameters |
|---|---|---|---|
| VAE Merge | `VAEMergeMEC` | [vae-merge](./vae-merge.md#vae-merge-vaemergemec) | [ref](../NODE_REFERENCE.md#vaemergemec) |

### VFX

| Node | id | Guide | Parameters |
|---|---|---|---|
| Optical Flow Re-Vector (MEC) | `OpticalFlowMEC` | [optical-flow](./optical-flow.md#optical-flow-re-vector-mec-opticalflowmec) | [ref](../NODE_REFERENCE.md#opticalflowmec) |

### Vault

| Node | id | Guide | Parameters |
|---|---|---|---|
| C2C Vault (protected subgraph) | `C2C_SecureSubgraph` | [vault_howto](./vault_howto.md#vault-nodes) | [ref](../NODE_REFERENCE.md#c2c_securesubgraph) |
| C2C Vault (protected subgraph, saves) | `C2C_SecureSubgraphOutput` | [vault_howto](./vault_howto.md#vault-nodes) | [ref](../NODE_REFERENCE.md#c2c_securesubgraphoutput) |
| C2C Vault — Locked (password to run) | `C2C_VaultLocked` | [vault_howto](./vault_howto.md#vault-nodes) | [ref](../NODE_REFERENCE.md#c2c_vaultlocked) |
| C2C Vault — Sealed (runs without password) | `C2C_VaultSealed` | [vault_howto](./vault_howto.md#vault-nodes) | [ref](../NODE_REFERENCE.md#c2c_vaultsealed) |
| Vault input (edit view) | `C2C_VaultEditInput` | [vault_howto](./vault_howto.md#vault-nodes) | [ref](../NODE_REFERENCE.md#c2c_vaulteditinput) |
| Vault output (edit view) | `C2C_VaultEditOutput` | [vault_howto](./vault_howto.md#vault-nodes) | [ref](../NODE_REFERENCE.md#c2c_vaulteditoutput) |

### Video

| Node | id | Guide | Parameters |
|---|---|---|---|
| AV Handles — Add / Trim | `AVHandlesMEC` | [video-io](./video-io.md#av-handles--add--trim-avhandlesmec) | [ref](../NODE_REFERENCE.md#avhandlesmec) |
| Load Images Path (C2C) | `LoadImagesPathC2C` | [video-io](./video-io.md#load-images-path-c2c-loadimagespathc2c) | [ref](../NODE_REFERENCE.md#loadimagespathc2c) |
| Load Video (C2C) | `LoadVideoC2C` | [video-io](./video-io.md#load-video-c2c-loadvideoc2c) | [ref](../NODE_REFERENCE.md#loadvideoc2c) |
| Mask Track — Motion/Propagate/Anchor/Consistency/Stabilize | `MaskTrackerMEC` | [video-temporal-bbox](./video-temporal-bbox.md#mask-track-masktrackermec) | [ref](../NODE_REFERENCE.md#masktrackermec) |
| Save Video (C2C) | `SaveVideoC2C` | [video-io](./video-io.md#save-video-c2c-savevideoc2c) | [ref](../NODE_REFERENCE.md#savevideoc2c) |
| Video Info (C2C) | `VideoInfoC2C` | [video-io](./video-io.md#video-info-c2c-videoinfoc2c) | [ref](../NODE_REFERENCE.md#videoinfoc2c) |

### Viewer helpers

| Node | id | Guide | Parameters |
|---|---|---|---|
| Viewer helper: Detect | `C2CViewerDetect` | [mask-matting-pipeline](./mask-matting-pipeline.md#viewer-helper-nodes-c2cviewerdetect-c2cviewerplateproxy-c2cviewerresult) | [ref](../NODE_REFERENCE.md#c2cviewerdetect) |
| Viewer helper: Plate Proxy | `C2CViewerPlateProxy` | [mask-matting-pipeline](./mask-matting-pipeline.md#viewer-helper-nodes-c2cviewerdetect-c2cviewerplateproxy-c2cviewerresult) | [ref](../NODE_REFERENCE.md#c2cviewerplateproxy) |
| Viewer helper: Result | `C2CViewerResult` | [mask-matting-pipeline](./mask-matting-pipeline.md#viewer-helper-nodes-c2cviewerdetect-c2cviewerplateproxy-c2cviewerresult) | [ref](../NODE_REFERENCE.md#c2cviewerresult) |
---

## Other pages

### Features that are not nodes

| Page | What it covers |
|---|---|
| [ai-spine](ai-spine.md) | Build a workflow from a plain-English description (Ctrl+Alt+B) and the error-diagnosis card; local or hosted AI backends |
| [live-preview](live-preview.md) | The **C2C ▸ Sampler latent preview** setting (Auto per model), preview decoder fetch, which video frame is previewed |
| [align-and-guides](align-and-guides.md) | Smart align (Alt+A), align / distribute shortcuts and drag guides for nodes and groups |
| [c2c-ui-framework](c2c-ui-framework.md) | The shared front-end kit the node panels are built with (for contributors) |
| [MIGRATION](MIGRATION.md) | Every merged node id, what replaced it, and how inputs and outputs map (generated) |
| [migration/](migration/README.md) | One page per deprecated or removed node, with measured parity |
| [APP_GATEWAY_DEPLOY](APP_GATEWAY_DEPLOY.md) | Deploying the C2C App Gateway (let others run a workflow without seeing it) |

### Keyboard shortcuts

Every shortcut this pack registers. All of them show in ComfyUI's **Settings ▸ Keybinding** panel, where you can
change or remove them.

| Keys | What it does |
|---|---|
| Alt+Shift+O | Show or hide the OmniBar |
| Ctrl+Alt+D | Open the Workflow Doctor panel |
| Ctrl+Alt+H | Open the auto-checkpoint history (earlier versions of this workflow) |
| Ctrl+Alt+B | Build a workflow from a plain-English description ([ai-spine](ai-spine.md)) |
| Ctrl+Alt+Shift+B / M / N | Bypass / mute / set back to normal every node of the selected node's type |
| Alt+Shift+L | Lock the selected nodes or group into a vault |
| Alt+Shift+T | Hide or show every node title ([node-identity](node-identity.md)) |
| Alt+A | Smart align the selected nodes ([align-and-guides](align-and-guides.md)) |
| Alt+Shift+← / → / ↑ / ↓ | Align left / right / top / bottom |
| Alt+Shift+X / Alt+Shift+Y | Centre on a vertical / horizontal line |
| Alt+Shift+H / Alt+Shift+V | Distribute horizontally / vertically |
| Alt+Shift+G | Turn the smart drag guides on or off |

### Design records (written before the code; kept as history)

[save-video-v2](save-video-v2.md), [LOAD_SAVE_VIDEO_CM_DESIGN](LOAD_SAVE_VIDEO_CM_DESIGN.md),
[VAULT_DESIGN_A17](VAULT_DESIGN_A17.md), [VAULT_DESIGN_A18](VAULT_DESIGN_A18.md), and the [design/](design/) folder.
The user guides above describe what is built.

### Nodes that moved to another pack (history pages)

| Page | Now in |
|---|---|
| [color-science](color-science.md) | ComfyUI-NukeMaxNodes (same node ids) |
| [render-pass](render-pass.md) | ComfyUI-NukeMaxNodes (same node ids) |
| [plate-tools](plate-tools.md) | ComfyUI-NukeMaxNodes (same node ids) |
| [exr-io](exr-io.md) | ComfyUI-NukeMaxNodes (`NukeMax_EXRSequenceLoad` / `NukeMax_EXRSequenceSave`) |
| [wan-director](wan-director.md) | ComfyUI-WanNodeExperiments |
| [paint-suite](paint-suite.md) | Merged here: Image Mask Editor and Inpaint Stitch Pro's `context blend` |

---

## Licence

The pack's code is Apache-2.0 ([LICENSE](../LICENSE), attribution in [NOTICE](../NOTICE)); credits and ported code
are listed in [CREDITS.md](../CREDITS.md), third-party licence texts in
[THIRD_PARTY_LICENSES/](../THIRD_PARTY_LICENSES/). Model weights keep their own licences; the guides name the
non-commercial ones where a node uses them.
