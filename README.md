<p align="center">
  <img src="https://i.pinimg.com/736x/5d/52/19/5d5219ef27c1530bea027ff95d352e05.jpg" alt="MEC Logo" width="120" />
</p>

> ### 📖 Full node & parameter reference
> **[NODE_REFERENCE.md](NODE_REFERENCE.md)** documents **every node in this pack**:
> each input's type, default, valid range and what it actually does, plus every
> output. It is generated directly from the live `NODE_CLASS_MAPPINGS`, so the
> descriptions there are the *same strings* you see as tooltips in ComfyUI and
> cannot drift out of date. Start there when you want to know what a knob does;
> read on here for how the pieces fit together.


<h1 align="center">ComfyUI-CustomNodePacks</h1>

<p align="center">
  <strong>MaskEditControl (MEC) + FolderIncrementer + VFX Suite</strong><br/>
  Production-grade mask editing, SAM1/2/3 segmentation, alpha matting, inpainting, diagnostics,<br/>
  temporal mask interpolation, luminance keying, color science, EXR I/O, render-pass<br/>
  compositing, plate tools, VAE merging &amp; analysis, and auto-versioned file output for ComfyUI.
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#node-reference">Node Reference</a> •
  <a href="#workflows">Workflows</a> •
  <a href="#troubleshooting">Troubleshooting</a> •
  <a href="#acknowledgements">Acknowledgements</a> •
  <a href="#license">License</a>
</p>

---

> ⚠️ **Non-Commercial Model Notice**: The **RMBG-2.0** model weights
> (`briaai/RMBG-2.0`) are subject to the **BRIA AI RMBG License**, which
> restricts commercial use. If you use Mask + Matting's `remove background` backend with
> `bg_model` = `rmbg_2.0` for commercial purposes you **must obtain a commercial
> licence from Bria AI** at <https://bria.ai>. All other models used in this
> pack (SAM, SAM2, BiRefNet, ViTMatte, SeC, etc.) are Apache-2.0 or MIT
> licensed and may be used commercially. See [NOTICE.md](NOTICE.md) for
> full attribution and license details.

---

## Overview

<!-- C2C:NODE-COUNTS:BEGIN -->
**ComfyUI-CustomNodePacks** registers **103** nodes across these families:

| Family | Nodes |
|--------|------:|
| Vault | 6 |
| Folder Incrementer | 5 |
| Render farm | 3 |
| C2C Distributed | 2 |
| Fluid Shots/Audio | 3 |
| C2C video | 3 |
| MaskEditControl | 9 |
| Image Mask Editor | 1 |
| Model analysis | 1 |
| Face Fixer | 1 |
| Mask + Matting | 5 |
| Layer Effects | 1 |
| Mask toolkit | 1 |
| Frequency / Grain | 3 |
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
| Facepoints | 1 |
| Nano Banana | 1 |
| Control AOV | 1 |
| Restored VFX | 3 |
| Legacy (deprecated) | 7 |

Startup log line to verify the pack loaded:

`[C2C] CustomNodePacks: 103 nodes loaded (Vault 6, Folder Incrementer 5, Render farm 3, C2C Distributed 2, Fluid Shots/Audio 3, C2C video 3, MaskEditControl 9, Image Mask Editor 1, Model analysis 1, Face Fixer 1, Mask + Matting 5, Layer Effects 1, Mask toolkit 1, Frequency / Grain 3, Tiled refinement 4, Magnific 15, AV Handles 1, Save Video 1, SAM Multi-Mask Picker 1, SAM Loader/Generator 2, Mask Placement 1, Semantic Segment 1, NukeNodeMax 1, Video Stabilizer 4, C2C helpers 8, Prompt Relay 3, AsymFlow sampler 1, HDR Color Science 1, LocateAnything 2, Facepoints 1, Nano Banana 1, Control AOV 1, Restored VFX 3, Legacy (deprecated) 7) - 0 failed`
<!-- C2C:NODE-COUNTS:END -->

With **ComfyUI-OmniScale** installed next to this pack, the 15 Magnific nodes are registered by OmniScale instead,
so this pack then loads 88.


### Who is this for?

- **VFX / compositing artists** who need Nuke-quality mask control inside ComfyUI
- **Video creators** who need temporally consistent masks across hundreds of frames
- **Inpainters** who want crop → inpaint → stitch pipelines with edge-aware blending
- **Anyone** tired of SAM giving 3 masks and not knowing which one to pick
- **Beginners** who want one-click background removal or a node that tells them *why* their mask failed

---

## Installation

### 1. Clone the repo

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Code2Collapse/ComfyUI-CustomNodePacks.git
```

Or install via **ComfyUI Manager** → search "CustomNodePacks".

### 2. Install dependencies

> **⚠️ IMPORTANT: Check your existing package versions before installing.**
>
> ComfyUI bundles its own `torch`, `torchvision`, `numpy`, and `Pillow`. Blindly running `pip install -r requirements.txt` can **overwrite** them with incompatible versions and break your ComfyUI installation.
>
> **Recommended approach:** Open `requirements.txt`, comment out anything already installed, and only install what you're missing:
>
> ```bash
> # Check what you already have:
> pip list | grep -i "opencv\|scipy\|safetensors\|transformers"
>
> # Then install only what's missing:
> pip install opencv-python>=4.7.0 scipy>=1.10.0 safetensors>=0.4.0
> ```

The `requirements.txt` lists:

| Package | Required? | Purpose |
|---------|-----------|---------|
| `opencv-python` | **Yes** | Edge detection, morphological ops, guided filter |
| `scipy` | **Yes** | Gaussian filters, signal processing |
| `safetensors` | **Yes** | Safe model weight loading |
| `transformers` | Optional | ViTMatte neural matting (best quality edges) |
| `pillow` | Optional | Image I/O for ViTMatte |

Core packages (`torch`, `torchvision`, `numpy`) are already provided by ComfyUI — do **not** reinstall them.

### 3. Place SAM model checkpoints

Download SAM 2.1 / SAM 3 weights and place them in:

```
ComfyUI/models/sams/        ← SAM 3
ComfyUI/models/sam2/        ← SAM 2.1
```

| Model | File | Source |
|-------|------|--------|
| SAM2.1 Large | `sam2.1_hiera_large.pt` | [Meta AI](https://github.com/facebookresearch/sam2) |
| SAM3 | `sam3_hiera_large.pt` | [Meta AI](https://github.com/facebookresearch/sam3) |

The **SAM Model Loader** node auto-detects model type from the filename.

> **Note:** Legacy SAM 1 (ViT-H/L/B) and the original SAM 2.0 line are no longer supported. Only SAM 2.1 and SAM 3 are exposed.

### 4. (Optional) ViTMatte matting

For the highest-quality edge refinement (hair, fur, glass, lace):

```bash
pip install transformers pillow
```

ViTMatte (~400 MB) is used from an installed copy: `ComfyUI/models/vitmatte/<name>/`, another pack's
`models/*/vitmatte-small-composition-1k`, or the Hugging Face cache. It is downloaded only when you tick MaskOps'
`auto_download`; without it MaskOps uses a guided filter and labels the result approximate. See
[Model weights and downloads](#model-weights-and-downloads).

### Model weights and downloads

No node downloads model weights unless you allow it (owner decision 2026-10-12). Before anything is called missing,
the node looks for an installed copy in its own folder, in every model folder ComfyUI knows (other packs' folders
too), in another pack's copy of a Transformers model (`models/<anything>/<repo name>`) and in the Hugging Face cache,
and reuses it. The pack creates no model folders at start-up; a folder is created only when a permitted download
saves into it.

| Node | Switch (default off) | Without the model |
|------|----------------------|-------------------|
| Mask + Matting, backend `remove background` (was Background Remover) | `auto_download` | model-free cut-out from the frame border, labelled approximate |
| Semantic Segment | `allow_download` | `skin` by colour, labelled approximate; other classes need the model |
| SAM Model Loader, SAM Mask Generator, SAM Multi-Mask Picker, Mask Placement | `allow_download` | GrabCut stand-in, labelled approximate |
| Mask + Matting, backend `SeC + MatAnyone2` (was SeC + MatAnyone) | `auto_download` | GrabCut + optical-flow tracking and a guided-filter matte, labelled approximate |
| LocateAnything | `allow_download` | none (reading a text prompt needs the model): the node names the folder |
| Face Landmarks (Facepoints) | `allow_download` | none (the landmarks need the model): the node names `models/facepoints`; weights CC BY-NC 4.0 (non-commercial) |
| MaskOps | `auto_download` | its labelled model-free segmenter / matter |

When a model is missing and downloading is off, the message names the model, the **absolute** folder and the file,
for example `Expected file 'sam2.1_hiera_base_plus.safetensors' in folder: D:\ComfyUI\models\sam2`. Put the file
there, or tick the switch to download it there. LocateAnything goes to `models/LLM/LocateAnything-3B`.

Each of these nodes reports a `model_status` line in its `info` output (SAM Model Loader: in its `SAM_MODEL`):
`loaded - found at <path>`, `needs permission - model-free result` (missing, downloading off),
`missing - model-free result` (missing for another reason) or, on Mask Placement, `not used (source_mask wired)`.

A saved SAM Model Loader choice `[download] ...` no longer downloads by itself: tick `allow_download`.

#### Model Browser downloads (Ctrl+Alt+M)

The Model Browser searches the models already on disk, Civitai and Hugging Face, and downloads into a ComfyUI model
folder you pick (only folders ComfyUI knows; anything else is refused). Downloads run as a queue with progress, speed
and time left, and can be paused, resumed (they continue where they stopped) or cancelled. A finished file is checked
against the SHA-256 the site publishes and removed if it does not match. When a version lists several files, the
`.safetensors` one is preferred. The GGUF models you download from the C2C Diagnostics sidebar (llama.cpp) use the same
queue, so they also resume, are hash-checked and appear in this list. That GGUF list is `data/llm_catalog.json`: 37
models whose repository, file, size and SHA-256 were read from the Hugging Face API
(`tools/c2c_llm_catalog_verify.py` in the workspace re-checks them). Each entry shows its download size and the memory
it needs, and ★ marks the largest general model that fits the memory free right now. Free GPU memory only counts
when your llama-cpp-python build can use the GPU.
The catalog also lists three small Hugging Face model folders (Qwen3 0.6B / 1.7B, Qwen2.5 1.5B Instruct) for the
local transformers backend. A folder is downloaded only when you ask (C2C AI Backends panel > Download a model
folder, from this machine only) into `models/LLM/<name>`, every file checked; it appears under that name only once all files pass.

- `CIVITAI_API_KEY`, `HF_TOKEN`: your tokens for restricted or gated files. They are sent as a header, never in a URL
  or a log.
- `C2C_CIVITAI_API_BASE`, `C2C_CIVITAI_DOWNLOAD_BASE`, `C2C_HF_API_BASE`, `C2C_HF_BASE`: a mirror or proxy instead
  of the public sites (defaults: civitai.com and huggingface.co).


### 5. Restart ComfyUI

Look for `[MEC] Loaded 44 MaskEditControl nodes.` in the console to confirm.

---

## Detailed Documentation

> **📚 Start here: [`docs/README.md`](docs/README.md)** — the central
> documentation index with quick-nav by task, pack-by-pack reference,
> and image-vs-video pipeline guidance.

Every node's parameters, modes, and outputs are documented in depth in the `docs/` folder:

| Guide | Nodes | What's inside |
|-------|------:|---------------|
| [**Documentation Index**](docs/README.md) | — | Central nav: quick lookup by task, pack-by-pack reference, image-vs-video helpers |
| [Inpaint Suite](docs/inpaint-suite.md) | 2 | Inpaint Crop Pro (modes pro / smart aspect / mask prepare) and Inpaint Stitch Pro (blend / paste back / smart / context blend) |
| [**ProPainter Suite**](docs/propainter.md) ★ new | 5 | Temporal video inpaint, stitch (drop-in for legacy), seam-only refine, object remove (needs ProPainter), RAFT flow refine |
| [**Face Fixer**](docs/face-fixer.md) | 1 | YOLO11 face detection → per-face KSampler → smart blend with `[SEP]/[ASC]/[DSC]/[SKIP]` wildcard prompts (connect `clip` for per-face prompts); seven controls shown, the rest in Detection / Blend / Face prompts sections; how-to in the link |
| [**Paint Suite**](docs/paint-suite.md) ★ new | 1 | Interactive paint canvas (hardness/expansion/blur stages); the context inpainter (color match, lightness rescue, differential diffusion) is Inpaint Stitch Pro's `context blend` mode |
| [**Mask + Matting Pipeline**](docs/mask-matting-pipeline.md) ★ new | 1 | Combined `MaskOpsMEC` (Mask + Matting) — pick segmenter (SAM2.1/SAM3/SeC/…) + matter (ViTMatte/RVM/MatAnyone2/…) + slot-based prompts; opens with a viewer: Find, click the subject, preview, render, QC |
| [**Video Stabilizer**](docs/video-stabilizer.md) | 1 | Feature tracking, OpenCV dense flow or RAFT optical flow (experimental), 4 presets, measured accuracy table; padding-mask output wires into Inpaint Crop Pro |
| [**Image Mask Editor**](docs/image-mask-editor.md) ★ new | 1 | Native-resolution mask editor over the image that reaches the node: brush/eraser/shapes/polygon/lasso/bucket, colour range (RGB/HSV/LAB), SAM clicks and boxes, ViTMatte edge-refine brush, batch frame strip, masks survive restarts, non-destructive grow/feather/threshold; modes paint (+ colour layer) / points + box / shape / spline / roto / video |
| [**VAE precision**](docs/vae-precision.md) ★ new | 2 | Where VAE round trips lose precision (measured) and the options that restrict it: fp32 and unclamped decode, tiled video decode, round-trip meter, original-pixel write-back |
| [Mask Editing](docs/mask-editing.md) | 8 | Transform XY, Draw Frame (12 shapes), Draw Shape (unified dropdown), Composite Advanced (8 ops), Math (11 ops), Batch Manager, Preview Overlay, Spline Mask Editor |
| [SAM & Segmentation](docs/sam-segmentation.md) | 8 | Model Loader, Mask Generator, Multi-Mask Picker, Unified Segmentation, Semantic Segment, and Mask + Matting's SAM + ViTMatte / SeC + MatAnyone2 / remove-background backends |
| [Matting & Refinement](docs/matting-refinement.md) | 4 | Matting Node (7 backends), ViTMatte Refiner (7 methods), Trimap Generator, Mask Tools (luma key) |
| [Video, Temporal & BBox](docs/video-temporal-bbox.md) | 10 | Batch Range (C2C) single-frame modes, Mask Propagate (5 modes), Temporal Anchor (SDF), Motion Mask Tracker (4 methods), 6 BBox nodes |
| [Utility & Interactive](docs/utility-nodes.md) | 8 | Points Mask Editor, Image Comparer, Mask Failure Explainer, Parameter History, Universal Reroute, 3 Folder Incrementer nodes |
| [VAE Tools](docs/vae-merge.md) | 4 | VAE Merge (8 algorithms), VAE Inspect (inspect / compare), Probe (C2C) latent |
| [Color Science](docs/color-science.md) | 3 | sRGB/linear/Rec.709/ACEScg convert, `.cube` LUT apply, exposure/WB/contrast grade |
| [EXR I/O](docs/exr-io.md) | 2 | Load and save OpenEXR with imageio + TIFF fallback chain |
| [Render Passes](docs/render-pass.md) | 2 | Merge beauty + AO/diffuse/spec/emission, depth→CoC mask |
| [Plate Tools](docs/plate-tools.md) | 4 | Grain match, plate stabilizer (ORB/FFT), clean-plate extractor, difference matte |
| [Video Frame Player](docs/video-frame-player.md) | 1 | In-graph scrubber + drag-crop (8 handles, aspect-lock) + trim range + frame-stride + lanczos resize/upscale, all in one node |

---

## Nodes

CustomNodePacks registers **103 nodes** (counts per family and the startup line: [NODE_REFERENCE.md](NODE_REFERENCE.md)). Every node's parameters, defaults and tooltips are in [NODE_REFERENCE.md](NODE_REFERENCE.md), generated from the code, so it cannot drift. This list is generated from the same registration (the Magnific sign-in nodes are skipped when ComfyUI-OmniScale is installed, which replaces them).

### 🐺 C2C/🧰 Core/AI Image

| Node | id | What it does |
|---|---|---|
| Nano Banana · Gemini Image (C2C) | [`NanoBananaC2C`](NODE_REFERENCE.md#nanobananac2c) | Generate or edit images with Google's Nano Banana family (gemini-3-pro-image / gemini-2.5-flash-image). |

### 🐺 C2C/🧰 Core/Audio

| Node | id | What it does |
|---|---|---|
| Audio Reverser | [`AudioReverserMEC`](NODE_REFERENCE.md#audioreversermec) | Reverses the audio waveform (torch.flip on the sample dim), preserving sample rate and channels. |

### 🐺 C2C/🧰 Core/Color Science

| Node | id | What it does |
|---|---|---|
| VAE Decode (C2C) | [`C2CVAEQualityDecode`](NODE_REFERENCE.md#c2cvaequalitydecode) | VAE decode with optional fp32 (<=0.17 dB measured gain), spatial-only tiling for Wan video, optional unclamped output for HDR chains, optional ACES tone mapping, and an optional clean step that measures and corrects colo... |

### 🐺 C2C/🧰 Core/Control

| Node | id | What it does |
|---|---|---|
| Control AOV — Multi-Control Fusion (C2C) | [`ControlAOVC2C`](NODE_REFERENCE.md#controlaovc2c) | VFX-AOV control fusion: emit depth/canny/pose/normal/motion/ID as separate passes, a channel-packed image (depth=R/canny=G/pose=B), and a convenience blend. |

### 🐺 C2C/🧰 Core/Diagnostics

| Node | id | What it does |
|---|---|---|
| Mask Failure Explainer — Diagnostics | [`MaskFailureExplainerMEC`](NODE_REFERENCE.md#maskfailureexplainermec) | Diagnose why a mask might be failing. |
| Model Metadata Extractor | [`ModelMetadataExtractorMEC`](NODE_REFERENCE.md#modelmetadataextractormec) | Inspect model file metadata WITHOUT unpickling or loading weights. |

### 🐺 C2C/🧰 Core/Distributed

| Node | id | What it does |
|---|---|---|
| C2C GPU Inventory | [`C2CGPUInventory`](NODE_REFERENCE.md#c2cgpuinventory) | Lists the GPUs (architecture Turing to Blackwell, memory, BF16 / FP8 / FP4 support, whether this torch build has kernels for them), how they are linked (P2P, NVLink), Grace / aarch64 hosts, whether NCCL / gloo really wor... |
| C2C Multi-GPU Split | [`C2CSequenceParallel`](NODE_REFERENCE.md#c2csequenceparallel) | Splits every model call over the GPUs of this machine. |

### 🐺 C2C/🧰 Core/Face

| Node | id | What it does |
|---|---|---|
| Face Landmarks (Facepoints, 620 + 68/98) | [`FacepointsLandmarksC2C`](NODE_REFERENCE.md#facepointslandmarksc2c) | Dense face landmarks for any model: 620 points on Google's GNM Head mesh, each with its own error estimate, plus the standard 68-point (iBUG) and 98-point (WFLW) sets and both pupils, from myned-ai/facepoints. |

### 🐺 C2C/🧰 Core/Frequency

| Node | id | What it does |
|---|---|---|
| Film Grain (MEC) | [`FilmGrainMEC`](NODE_REFERENCE.md#filmgrainmec) | Lightroom-style film grain with per-frame animation for video batches. |
| Frequency Combine (MEC) | [`FrequencyCombineMEC`](NODE_REFERENCE.md#frequencycombinemec) | Merge a high-frequency detail layer onto a low-frequency base. |
| Frequency Separate (MEC) | [`FrequencySeparateMEC`](NODE_REFERENCE.md#frequencyseparatemec) | Split an image into low-frequency base and high-frequency detail. |

### 🐺 C2C/🧰 Core/Helpers

| Node | id | What it does |
|---|---|---|
| Batch Range (C2C) | [`ImageBatchSliceMEC`](NODE_REFERENCE.md#imagebatchslicemec) | Pick frames from an IMAGE batch: a start:end:step range (negative values count from the end, step=2 keeps every second frame), a split at a frame index or a fraction, or a single frame (first, middle, last or by index). |
| Conditional Switch | [`ConditionalSwitchMEC`](NODE_REFERENCE.md#conditionalswitchmec) | Return value_true when condition is True, else value_false. |
| Execution Timer (C2C) | [`ExecutionTimerMEC`](NODE_REFERENCE.md#executiontimermec) | Stopwatch: returns seconds since the *previous* execution of the same label. |
| Number Lerp (C2C) | [`NumberLerpMEC`](NODE_REFERENCE.md#numberlerpmec) | Linear / smoothstep / cosine interpolation between two values. |
| Probe (C2C) | [`ImageStatsProbeMEC`](NODE_REFERENCE.md#imagestatsprobemec) | Pass-through probe for debugging: wire an image, a mask and/or a latent and read their statistics - black or blown-out frames, how much of each frame a mask covers, NaN / Inf or saturated latents. |
| Seed List Generator (C2C) | [`SeedListMEC`](NODE_REFERENCE.md#seedlistmec) | Generate N deterministic seeds from a base seed. |
| Size (C2C) | [`AspectPresetMEC`](NODE_REFERENCE.md#aspectpresetmec) | Width and height for a render: a common aspect ratio scaled to a long edge (Wan 480p / 720p presets give Wan's native sizes), or your own width x height - snapped to a multiple of N, because Wan, Flux and SDXL need sizes... |
| Text Template (C2C) | [`TextTemplateMEC`](NODE_REFERENCE.md#texttemplatemec) | Substitute {a}{b}{c}{d} placeholders in a template string. |

### 🐺 C2C/🧰 Core/Inpaint

| Node | id | What it does |
|---|---|---|
| Inpaint Crop Pro | [`InpaintCropProMEC`](NODE_REFERENCE.md#inpaintcroppromec) | Crop around mask for inpainting (lquesada API + Wan 2.2 Animate aware). |
| Inpaint Stitch Pro | [`InpaintStitchProMEC`](NODE_REFERENCE.md#inpaintstitchpromec) | Stitch the inpainted crop back into the original (lquesada-compatible) with blend overrides + color match; stitch_mode 'paste back', 'smart' (for a smart aspect crop) or 'context blend' (colour correction and lightness r... |
| ProPainter — Temporal / Remove / Stitch / Refine / Flow | [`ProPainterMEC`](NODE_REFERENCE.md#propaintermec) | Unified ProPainter node. |

### 🐺 C2C/🧰 Core/Layer Effects

| Node | id | What it does |
|---|---|---|
| Layer Effects (C2C) | [`LayerEffectsMEC`](NODE_REFERENCE.md#layereffectsmec) | Photoshop-style layer effects in one node: drop shadow, outer glow, gradient map, gradient overlay, colour overlay, inner glow, inner shadow and stroke. |

### 🐺 C2C/🧰 Core/Legacy

| Node | id | What it does |
|---|---|---|
| BBox Crop (MEC) (legacy) | [`BBoxCrop`](NODE_REFERENCE.md#bboxcrop) | Deprecated — kept so old workflows load. |
| BBox From Mask (MEC) (legacy) | [`BBoxFromMask`](NODE_REFERENCE.md#bboxfrommask) | Deprecated — kept so old workflows load. |
| BBox Pad (MEC) (legacy) | [`BBoxPad`](NODE_REFERENCE.md#bboxpad) | Deprecated — kept so old workflows load. |
| BBox To Mask (MEC) (legacy) | [`BBoxToMask`](NODE_REFERENCE.md#bboxtomask) | Deprecated — kept so old workflows load. |
| Mask Composite Advanced (MEC) (legacy) | [`MaskCompositeAdvanced`](NODE_REFERENCE.md#maskcompositeadvanced) | Deprecated — kept so old workflows load. |
| Mask Math (MEC) (legacy) | [`MaskMath`](NODE_REFERENCE.md#maskmath) | Deprecated — kept so old workflows load. |
| Mask Preview Overlay (MEC) (legacy) | [`MaskPreviewOverlay`](NODE_REFERENCE.md#maskpreviewoverlay) | Deprecated — kept so old workflows load. |

### 🐺 C2C/🧰 Core/Magnific

| Node | id | What it does |
|---|---|---|
| OmniScale Creation Picker | [`MagnificCreationPicker`](NODE_REFERENCE.md#magnificcreationpicker) |  |
| OmniScale Generate Image | [`MagnificGenerateImage`](NODE_REFERENCE.md#magnificgenerateimage) |  |
| OmniScale Generate Music | [`MagnificGenerateMusic`](NODE_REFERENCE.md#magnificgeneratemusic) |  |
| OmniScale Generate Video | [`MagnificGenerateVideo`](NODE_REFERENCE.md#magnificgeneratevideo) |  |
| OmniScale Library Reference | [`MagnificLibraryReference`](NODE_REFERENCE.md#magnificlibraryreference) |  |
| OmniScale Metadata (unpack) | [`MagnificMetadata`](NODE_REFERENCE.md#magnificmetadata) |  |
| OmniScale Remove Background | [`MagnificRemoveBackground`](NODE_REFERENCE.md#magnificremovebackground) |  |
| OmniScale Retouch (inpaint) | [`MagnificRetouch`](NODE_REFERENCE.md#magnificretouch) |  |
| OmniScale Save To (project/folder) | [`MagnificSaveTo`](NODE_REFERENCE.md#magnificsaveto) |  |
| OmniScale Skin Enhancer | [`MagnificSkinEnhancer`](NODE_REFERENCE.md#magnificskinenhancer) |  |
| OmniScale Stock Picker | [`MagnificStockPicker`](NODE_REFERENCE.md#magnificstockpicker) |  |
| OmniScale Stock Search | [`MagnificStockSearch`](NODE_REFERENCE.md#magnificstocksearch) |  |
| OmniScale Upscale Image | [`MagnificUpscaleImage`](NODE_REFERENCE.md#magnificupscaleimage) |  |
| OmniScale Upscale Video | [`MagnificUpscaleVideo`](NODE_REFERENCE.md#magnificupscalevideo) |  |
| OmniScale Voiceover (TTS) | [`MagnificVoiceover`](NODE_REFERENCE.md#magnificvoiceover) |  |

### 🐺 C2C/🧰 Core/Mask

| Node | id | What it does |
|---|---|---|
| Image Mask Editor (C2C) | [`ImageMaskEditorC2C`](NODE_REFERENCE.md#imagemaskeditorc2c) | Paint a mask over the image that actually reaches this node. |
| Mask Placement — Prompt/Ref → Place → Track | [`MaskPlacementMEC`](NODE_REFERENCE.md#maskplacementmec) | Universal mask placement: source an alpha (wire a mask, or give a text prompt + SAM model and it segments the object for you), place it with a draggable 4-corner perspective quad, and propagate across the video. |
| Mask Tools (C2C) | [`MaskToolsMEC`](NODE_REFERENCE.md#masktoolsmec) | Keyers and matte operations in one node: colour, luma and difference keys; grade (gain, gamma, clamp, invert); grow or shrink each side (left / right / up / down) with a feather; combine two mattes; gradient; grain; moti... |

### 🐺 C2C/🧰 Core/Model Analysis

| Node | id | What it does |
|---|---|---|
| VAE Inspect | [`VAEBlockInspectorMEC`](NODE_REFERENCE.md#vaeblockinspectormec) | Inspect a VAE's per-block weight statistics and outlier tensors, or compare two VAEs block by block (cosine similarity) - for checking merges and broken checkpoints. |

### 🐺 C2C/🧰 Core/Paint

| Node | id | What it does |
|---|---|---|
| Face Fixer | [`MECFaceFixer`](NODE_REFERENCE.md#mecfacefixer) | Auto face detection (YOLO11) + per-face crop + AI pre-upscale + context-aware sampling + smart blend with wildcard per-face prompts. |

### 🐺 C2C/🧰 Core/Pipeline

| Node | id | What it does |
|---|---|---|
| Mask + Matting | [`MaskOpsMEC`](NODE_REFERENCE.md#maskopsmec) | Production-grade segmentation + matting + AUTO-QUALITY + diagnostics in one node. |
| Mask Refiner | [`MaskRefineMEC`](NODE_REFERENCE.md#maskrefinemec) | Unified, training-free mask refinement. |

### 🐺 C2C/🧰 Core/Plate

| Node | id | What it does |
|---|---|---|
| PAR Desqueeze (anamorphic → square px) | [`PARDesqueezeMEC`](NODE_REFERENCE.md#pardesqueezemec) | Convert an anamorphic plate to SQUARE pixels before AI processing (Nuke: 4448x3840 @ PAR 1.7266 → 7680x3840 at 2:1). |
| PAR Resqueeze (back to plate) | [`PARResqueezeMEC`](NODE_REFERENCE.md#parresqueezemec) | Return an AI-processed square-pixel frame to the plate's original pixel dimensions (PAR metadata is reapplied in Nuke). |

### 🐺 C2C/🧰 Core/Preview

| Node | id | What it does |
|---|---|---|
| Video Comparer — Player + Wipe/Diff/Scopes | [`VideoComparerC2C`](NODE_REFERENCE.md#videocomparerc2c) | Nuke-grade A/B comparer for image / video / EXR / audio. |
| Video Frame Player | [`VideoFramePlayerMEC`](NODE_REFERENCE.md#videoframeplayermec) | Video scrubber + drag-crop + integrated resize. |

### 🐺 C2C/🧰 Core/Prompt Relay

| Node | id | What it does |
|---|---|---|
| Prompt Relay Advanced Options | [`PromptRelayAdvancedOptionsC2C`](NODE_REFERENCE.md#promptrelayadvancedoptionsc2c) | Optional per-stream tuning for the Prompt Relay encoders. |
| Prompt Relay Encode | [`PromptRelayEncodeC2C`](NODE_REFERENCE.md#promptrelayencodec2c) | Unified Prompt Relay encoder (native / smart / kijai). |
| Prompt Relay Restore (Kijai) | [`PromptRelayRestoreKijaiC2C`](NODE_REFERENCE.md#promptrelayrestorekijaic2c) | Undo a Prompt Relay (Kijai) patch — restores the original cross_attn.forward. |

### 🐺 C2C/🧰 Core/Render Farm

| Node | id | What it does |
|---|---|---|
| C2C Farm Cluster Status | [`C2C_ClusterStatus`](NODE_REFERENCE.md#c2c_clusterstatus) | Live render-farm capacity (queue depth, VRAM, reachability) as JSON. |
| C2C Farm Job History (Audit Log) | [`C2C_JobHistory`](NODE_REFERENCE.md#c2c_jobhistory) | Render-farm audit log: per-user/status stats + recent jobs table. |
| C2C Farm Submit — Remote Render | [`C2C_Submit`](NODE_REFERENCE.md#c2c_submit) | Dispatch a full workflow JSON to a remote ComfyUI render-farm backend. |

### 🐺 C2C/🧰 Core/SAM

| Node | id | What it does |
|---|---|---|
| SAM Mask Generator — Points + BBox + Text | [`SAMMaskGeneratorMEC`](NODE_REFERENCE.md#sammaskgeneratormec) | Generate masks using SAM with point + bounding box prompts, iterative refinement, and VRAM offload support. |
| SAM Model Loader — SAM2.1 / SAM3 | [`SAMModelLoaderMEC`](NODE_REFERENCE.md#sammodelloadermec) | Load SAM 2.1 or SAM 3 model. |
| SAM Multi-Mask Picker — 3 candidates + scores | [`SamMultiMaskPickerMEC`](NODE_REFERENCE.md#sammultimaskpickermec) | Run SAM inference to get 3 candidate masks with confidence scores. |

### 🐺 C2C/🧰 Core/Sampling

| Node | id | What it does |
|---|---|---|
| AsymFlow Sampler Patch (Lakonik signal-shift) | [`AsymFlowSamplerPatch`](NODE_REFERENCE.md#asymflowsamplerpatch) | Replace the model's flow-matching schedule with AsymFlow's shifted-signal-ratio mapping (sigma = r/(shift+r), r=sqrt(t/(1-t))). |

### 🐺 C2C/🧰 Core/Segmentation

| Node | id | What it does |
|---|---|---|
| LocateAnything Grounding (MEC) | [`LocateAnythingGroundingMEC`](NODE_REFERENCE.md#locateanythinggroundingmec) | Find things in an image by describing them in words - 'the person on the left', 'red car', 'coffee cup' - and get back bounding boxes. |
| LocateAnything → SAM Prompt (MEC) | [`LocateAnythingToSAMMEC`](NODE_REFERENCE.md#locateanythingtosammec) | Turn bounding boxes into filled rectangles a SAM node can take as box prompts. |
| Semantic Segment — Face / Clothes Parsing | [`SemanticSegmentMEC`](NODE_REFERENCE.md#semanticsegmentmec) | Semantic face / clothes parsing using SegFormer. |

### 🐺 C2C/🧰 Core/Stabilization

| Node | id | What it does |
|---|---|---|
| Video Stabilizer | [`VideoStabilizerMEC`](NODE_REFERENCE.md#videostabilizermec) | Video stabilizer: classic feature tracking, RAFT optical flow or OpenCV dense flow, with presets. |
| Video Stabilizer — Auto (deprecated) | [`VideoStabilizerAutoMEC`](NODE_REFERENCE.md#videostabilizerautomec) | DEPRECATED — use VideoStabilizerMEC (method=auto) instead. |
| Video Stabilizer — Classic (deprecated) | [`VideoStabilizerClassicMEC`](NODE_REFERENCE.md#videostabilizerclassicmec) | DEPRECATED — use VideoStabilizerMEC (method=classic) instead. |
| Video Stabilizer — Dense Flow (deprecated) | [`VideoStabilizerFlowMEC`](NODE_REFERENCE.md#videostabilizerflowmec) | DEPRECATED — use VideoStabilizerMEC. |

### 🐺 C2C/🧰 Core/Temporal

| Node | id | What it does |
|---|---|---|
| Fluid Shot Decoder (Restore Timing) | [`FluidShotDecoderMEC`](NODE_REFERENCE.md#fluidshotdecodermec) | Selects exactly the frames that correspond to the ORIGINAL input frames using the encoder's TIMEMAP — restores the original frame count and slow-to-fast pacing. |
| Fluid Shot Encoder (Temporal Normalizer) | [`FluidShotEncoderMEC`](NODE_REFERENCE.md#fluidshotencodermec) | Analyzes optical-flow speed and inserts in-between frames on fast sections so motion per frame is constant — safe for Wan-style AI video. |

### 🐺 C2C/🧰 Core/Tiling

| Node | id | What it does |
|---|---|---|
| Identity Lock — refine without moving | [`IdentityLockMEC`](NODE_REFERENCE.md#identitylockmec) | Refine every pixel without letting anything move. |
| Tile Merge | [`TileMergeMEC`](NODE_REFERENCE.md#tilemergemec) | Blend the tiles back into one clip, in float32, with no seam. |
| Tile Plan | [`TilePlanMEC`](NODE_REFERENCE.md#tileplanmec) | Decide the tiles once for a whole clip. |
| Tile Split | [`TileSplitMEC`](NODE_REFERENCE.md#tilesplitmec) | Split a clip into overlapping tiles, emitted as a batch so any refiner can process them. |

### 🐺 C2C/🧰 Core/Utils

| Node | id | What it does |
|---|---|---|
| Folder Version Check (deprecated) | [`FolderIncrementerReset`](NODE_REFERENCE.md#folderincrementerreset) | Report the current version state for a label/date folder. |
| Folder Version Control (check / set) | [`FolderVersionControl`](NODE_REFERENCE.md#folderversioncontrol) | Check or set the version counter for a label/date folder. |
| Folder Version Incrementer | [`FolderIncrementer`](NODE_REFERENCE.md#folderincrementer) | Auto-incrementing per-label / per-date version counter. |
| Folder Version Set (deprecated) | [`FolderIncrementerSet`](NODE_REFERENCE.md#folderincrementerset) | Reserve version slots by creating empty placeholder directories under <base>/<label>/<MM-DD-YYYY>/. |
| Folder Version Variant (per output) | [`FolderVersionVariant`](NODE_REFERENCE.md#folderversionvariant) | Give one output its own prefix or suffix while keeping the version the Folder Version Incrementer decided. |
| Parameter History | [`ParameterHistoryMEC`](NODE_REFERENCE.md#parameterhistorymec) | Query the parameter history database. |

### 🐺 C2C/🧰 Core/VAE

| Node | id | What it does |
|---|---|---|
| VAE Merge | [`VAEMergeMEC`](NODE_REFERENCE.md#vaemergemec) | Merge 2 or 3 VAEs with 13 strategies (weighted_sum, sigmoid, geometric, slerp, dare_ties, …). |

### 🐺 C2C/🧰 Core/VFX

| Node | id | What it does |
|---|---|---|
| Optical Flow Re-Vector (MEC) | [`OpticalFlowMEC`](NODE_REFERENCE.md#opticalflowmec) | Dense optical flow re-vector. |

### 🐺 C2C/🧰 Core/Vault

| Node | id | What it does |
|---|---|---|
| C2C Vault (protected subgraph) | [`C2C_SecureSubgraph`](NODE_REFERENCE.md#c2c_securesubgraph) | A C2C Vault: a subgraph whose inside is encrypted. |
| C2C Vault (protected subgraph, saves) | [`C2C_SecureSubgraphOutput`](NODE_REFERENCE.md#c2c_securesubgraphoutput) | A C2C Vault: a subgraph whose inside is encrypted. |
| C2C Vault — Locked (password to run) | [`C2C_VaultLocked`](NODE_REFERENCE.md#c2c_vaultlocked) | Run a password-locked subgraph. |
| C2C Vault — Sealed (runs without password) | [`C2C_VaultSealed`](NODE_REFERENCE.md#c2c_vaultsealed) | Sealed subgraph - RUNS WITHOUT A PASSWORD, so you can hand a workflow to someone and they just use it. |
| Vault input (edit view) | [`C2C_VaultEditInput`](NODE_REFERENCE.md#c2c_vaulteditinput) | A vault input in the vault's edit view. |
| Vault output (edit view) | [`C2C_VaultEditOutput`](NODE_REFERENCE.md#c2c_vaulteditoutput) | A vault output in the vault's edit view. |

### 🐺 C2C/🧰 Core/Video

| Node | id | What it does |
|---|---|---|
| AV Handles — Add / Trim | [`AVHandlesMEC`](NODE_REFERENCE.md#avhandlesmec) | Add repeated frames at the head (or tail) to stabilise a video model, then trim exactly those back off afterwards — without retyping the number. |
| Load Images Path (C2C) | [`LoadImagesPathC2C`](NODE_REFERENCE.md#loadimagespathc2c) |  |
| Load Video (C2C) | [`LoadVideoC2C`](NODE_REFERENCE.md#loadvideoc2c) |  |
| Mask Track — Motion/Propagate/Anchor/Consistency/Stabilize | [`MaskTrackerMEC`](NODE_REFERENCE.md#masktrackermec) | Unified video-mask tracker. |
| Save Video (C2C) | [`SaveVideoC2C`](NODE_REFERENCE.md#savevideoc2c) | Save an IMAGE batch as H.264/H.265/ProRes/DNxHR/FFV1/WebM/GIF/WebP or 16-bit PNG / EXR sequences. |
| Video Info (C2C) | [`VideoInfoC2C`](NODE_REFERENCE.md#videoinfoc2c) |  |

### 🐺 C2C/🧰 Core/Viewer helpers

| Node | id | What it does |
|---|---|---|
| Viewer helper: Detect | [`C2CViewerDetect`](NODE_REFERENCE.md#c2cviewerdetect) | Viewer helper (hidden): finds objects on one frame (text, points or boxes) for the Mask + Matting viewer and writes one mask per object. |
| Viewer helper: Plate Proxy | [`C2CViewerPlateProxy`](NODE_REFERENCE.md#c2cviewerplateproxy) | Viewer helper (hidden): writes the plate as proxy frames (at most 960 px, at most 240 frames) for the Mask + Matting viewer. |
| Viewer helper: Result | [`C2CViewerResult`](NODE_REFERENCE.md#c2cviewerresult) | Viewer helper (hidden): writes the rendered alpha and the plate as proxy frames and measures QC for the Mask + Matting viewer. |

## Merged and removed nodes

These ids were folded into the nodes above (or moved to another pack). A saved workflow that uses one loads with the replacement node, its links and values carried over (ComfyUI's node replacement; C2C also migrates API prompts).

| Old id | Now |
|---|---|
| `BackgroundRemoverMEC` | `MaskOpsMEC` |
| `BatchVersionManagerMEC` | `FolderIncrementer` |
| `BBoxSmooth` | `MaskTrackerMEC` |
| `C2CACESTonemap` | `NukeMax_HDRToneMap` |
| `C2CColorSpaceConvert` | `ColorSpaceConvertMEC` |
| `DimensionsSnapMEC` | `AspectPresetMEC` |
| `DrawShapeMEC` | `ImageMaskEditorC2C` |
| `EdgeSpreadMEC` | `MaskToolsMEC` |
| `ImageBatchSplitMEC` | `ImageBatchSliceMEC` |
| `InpaintMaskPrepareMEC` | `InpaintCropProMEC` |
| `InpaintPasteBackMEC` | `InpaintStitchProMEC` |
| `LayerEffectColorOverlayMEC` | `LayerEffectsMEC` |
| `LayerEffectDropShadowMEC` | `LayerEffectsMEC` |
| `LayerEffectGradientMapMEC` | `LayerEffectsMEC` |
| `LayerEffectGradientOverlayMEC` | `LayerEffectsMEC` |
| `LayerEffectInnerGlowMEC` | `LayerEffectsMEC` |
| `LayerEffectInnerShadowMEC` | `LayerEffectsMEC` |
| `LayerEffectOuterGlowMEC` | `LayerEffectsMEC` |
| `LayerEffectStrokeMEC` | `LayerEffectsMEC` |
| `LoadEXRMEC` | `NukeMax_EXRSequenceLoad` |
| `LoadVideoPathC2C` | `LoadVideoC2C` |
| `LuminanceKeyerMEC` | `MaskToolsMEC` |
| `MaskAreaProbeMEC` | `ImageStatsProbeMEC` |
| `MaskBatchCombineMEC` | `MaskToolsMEC` |
| `MaskDrawFrame` | `ImageMaskEditorC2C` |
| `MaskEditMEC` | `MaskToolsMEC` |
| `MaskFromColorMEC` | `MaskToolsMEC` |
| `MaskGradientMEC` | `MaskToolsMEC` |
| `MaskGrainMEC` | `MaskToolsMEC` |
| `MaskMattingMEC` | `MaskOpsMEC` |
| `MaskMotionBlurMEC` | `MaskToolsMEC` |
| `MaskPropagateVideo` | `MaskTrackerMEC` |
| `MaskTemporalMEC` | `MaskTrackerMEC` |
| `MaskTransformXY` | `MaskToolsMEC` |
| `MECAdvancedPaintCanvas` | `ImageMaskEditorC2C` |
| `MECBuilderSampler` | `KSampler` |
| `MECContextInpainter` | `InpaintStitchProMEC` |
| `MECToneRefiner` | `NukeMax_Grade` |
| `MotionMaskTrackerMEC` | `MaskTrackerMEC` |
| `PointsMaskEditor` | `ImageMaskEditorC2C` |
| `PromptRelayEncodeKijaiC2C` | `PromptRelayEncodeC2C` |
| `PromptRelayEncodeSmartC2C` | `PromptRelayEncodeC2C` |
| `ProPainterTemporalMEC` | `ProPainterMEC` |
| `SAMViTMattePipelineMEC` | `MaskOpsMEC` |
| `SaveEXRMEC` | `NukeMax_EXRSequenceSave` |
| `SeCMatAnyonePipelineMEC` | `MaskOpsMEC` |
| `ShuffleMEC` | `MaskToolsMEC` |
| `SmartImageCropMEC` | `InpaintCropProMEC` |
| `SmartImageStitcherMEC` | `InpaintStitchProMEC` |
| `SplineMaskEditorMEC` | `ImageMaskEditorC2C` |
| `SplineMaskMEC` | `ImageMaskEditorC2C` |
| `SplineMaskTrackerMEC` | `MaskTrackerMEC` |
| `SplinePathFlowMaskMEC` | `ImageMaskEditorC2C` |
| `TclParseMEC` | `NukeMax_NkScriptParse` |
| `TclSerializeMEC` | `NukeMax_NkScriptSerialize` |
| `TemporalAnchorMEC` | `MaskTrackerMEC` |
| `TemporalConsistencyCheckerMEC` | `MaskTrackerMEC` |
| `VAECleanMEC` | `C2CVAEQualityDecode` |
| `VAELatentInspectorMEC` | `ImageStatsProbeMEC` |
| `VAESimilarityAnalyserMEC` | `VAEBlockInspectorMEC` |
| `VectorRotoMEC` | `ImageMaskEditorC2C` |
| `VideoFrameExtractorMEC` | `ImageBatchSliceMEC` |
| `VideoMaskEditorMEC` | `ImageMaskEditorC2C` |
| `VideoStabilizerAutoMEC` | `VideoStabilizerMEC` |
| `VideoStabilizerClassicMEC` | `VideoStabilizerMEC` |
| `VideoStabilizerFlowMEC` | `VideoStabilizerMEC` |
| `InsightStatusMEC` | removed: per-node time and memory show as the Insight badges under each node. |
| `IntegrityStatusMEC` | removed: the integrity report is in the Diagnostics sidebar, Integrity tab. |

## Feature notes

In-depth notes on the larger nodes (hand-written; parameters are in NODE_REFERENCE.md).

#### SAM Model Loader (MEC)

Loads SAM 2.1 / SAM 3 checkpoints with optional VRAM offload.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | — | Checkpoint filename (auto-listed from `models/sams/` and `models/sam2/`) |
| `model_type` | `auto` | Force type: `sam2.1`, `sam3`, or `auto` |
| `device` | `cuda` | `cuda` / `cpu` |
| `offload_to_cpu` | `false` | Keep model on CPU, move to GPU only during inference (saves ~2–4 GB VRAM) |
| `dtype` | `float16` | `float16` / `bfloat16` / `float32` |
| `allow_download` | `false` | Download a missing `[download] ...` checkpoint into `models/sam2/`. Off: an installed copy anywhere ComfyUI knows is used; none gives the labelled model-free stand-in |

**Output:** `SAM_MODEL` — connect to SAM Mask Generator, SAM Multi-Mask Picker, or Mask + Matting (`sam_model`, backend SAM + ViTMatte). It
carries `model_status`.

---

#### SAM Mask Generator (MEC)

Runs SAM 2.1 / SAM 3 inference with point + bbox prompts, iterative refinement, and auto-negative point sampling.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | — | Input image |
| `sam_model` | — | From SAM Model Loader |
| `points_json` | `""` | `[{"x":int, "y":int, "label":1}, ...]` |
| `bbox_json` | `""` | `[x1, y1, x2, y2]` |
| `multimask` | `true` | Return 3 candidate masks |
| `mask_index` | `0` | Which candidate (0–2) to use |
| `refine_iterations` | `1` | Iterative SAM passes (each tightens boundaries) |
| `auto_negative_points` | `0` | Sample N negative points from outside the mask |
| `allow_download` | `false` | Allow its text-grounding model to download if missing |

**Outputs:** `masks`, `best_score`, `info`

---

#### SAM Multi-Mask Picker (MEC)

Run SAM inference and view **all 3 candidate masks** side-by-side with IoU scores. An interactive JS widget renders thumbnails — click a mask or press **1** / **2** / **3** to select it instantly.

**How it works:** SAM always outputs 3 masks ranked by confidence. Instead of guessing which `mask_index` is best, this node shows all three with quality scores and lets you pick visually. Works with SAM 2.1 and SAM 3 (or the labelled model-free stand-in when no SAM model is installed).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | — | Input image |
| `model_name` | — | SAM model (SAM 2.1 / SAM 3, auto-listed) |
| `points_json` | `[{"x":256,"y":256,"label":1}]` | Point prompts JSON |
| `bbox_json` | `""` | Optional bounding box `[x1,y1,x2,y2]` |
| `precision` | `fp32` | `fp32` / `fp16` / `bf16` |
| `selected_index` | `0` | Which candidate (0–2) — updated by widget click |
| `sam_model` | (optional) | Pre-loaded SAM model from SAM Model Loader |
| `bbox` | (optional) | BBox from BBox pipeline (overrides `bbox_json`) |
| `allow_download` | `false` | Download the SAM checkpoint if it is not installed |

**Outputs:** `selected_mask`, `all_masks` (3×H×W), `selected_index`, `scores` (JSON), `info` (with `model_status`)

> **Who is it for?** When SAM gives you 3 options and you don't know which is best — see them all at once, pick with one click, and pipe the winner downstream.

---

#### SAM + ViTMatte (a backend of Mask + Matting)

End-to-end pipeline: **SAM coarse segmentation → iterative refinement → neural alpha matting**. Since L7.65 P05 this is
Mask + Matting (`MaskOpsMEC`) with `backend` = `SAM + ViTMatte`; saved SAM + ViTMatte Pipeline nodes migrate into it
when a workflow loads, with the same results. Connect a SAM Model Loader to `sam_model`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sam_iterations` | `2` | Iterative SAM passes |
| `sam_refine_method` | `auto` | `auto` / `vitmatte` / `guided_filter` / `multi_scale_guided` / `color_aware` / `laplacian_blend` |
| `sam_edge_radius` | `12` | Pixels around edges to refine |
| `sam_detail_preservation` | `0.85` | Fine detail retention (hair, fur, lace) |
| `sam_edge_contrast` | `1.0` | Edge sharpness boost (>1 = sharper) |
| `fill_holes_enabled` | `true` | Fill holes inside mask |
| `min_region_size` | `64` | Remove noise < N pixels |

**Outputs (Mask + Matting):** `alpha` (was refined_mask), `mask` (coarse_mask), `edge_mask`, `preview`, `bbox`
(detected_bbox, still `[x, y, w, h]` in this backend), `score` (best SAM score), `info`, plus `foreground` and the other
Mask + Matting outputs.

**Pipeline stages:**
1. SAM coarse mask from point/bbox prompts
2. Iterative refinement (re-run SAM with mask-derived prompts)
3. Edge-aware matting (ViTMatte / guided filter / LAB color)
4. Edge contrast boost
5. Post-processing (hole fill, small region removal)

---

#### SeC + MatAnyone2 (a backend of Mask + Matting)

End-to-end pipeline: **segmentation → MatAnyone2 temporal alpha matting**. A SeC model choice currently segments with
SAM 3 + text prompt: the SeC engine ships in the pack (`nodes/sec_engine`, L7.90) but is not wired into this backend
yet (L9.152). Since L7.65 P05 this is Mask +
Matting (`MaskOpsMEC`) with `backend` = `SeC + MatAnyone2`; saved SeC + MatAnyone2 Pipeline nodes migrate into it
when a workflow loads, with the same results.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sec_model` | — | SeC / SAM2 / SAM3 model selector |
| `sec_text_prompt` | `""` | Text description of target object (e.g. "cat", "person in red"); a wired `text_prompt` wins |
| `sec_matting_backend` | `auto` | `auto` / `matanyone2` / `vitmatte_small` / `vitmatte_base` |
| `sec_edge_radius` | `15` | Edge refinement radius in pixels |
| `sec_n_warmup` | `5` | MatAnyone2 warmup frames (more = better temporal init) |
| `sec_edge_refine_method` | `none` | Optional post-matting refinement: `vitmatte` / `guided_filter` / `multi_scale_guided` |
| `fill_holes_enabled` | `true` | Fill interior holes in alpha |
| `min_region_size` | `64` | Remove isolated regions < N pixels |
| `auto_download` | `false` | Download the segmentation / matting models if they are not installed |

**Outputs (Mask + Matting):** `foreground` (was rgb, premultiplied), `alpha` (alpha_mask), `mask` (coarse_mask),
`preview`, `info` (`model_status` per model), plus the other Mask + Matting outputs.

**Pipeline stages:**
1. SeC/SAM coarse segmentation (text or point/bbox prompts)
2. MatAnyone2 temporal alpha matting with warmup protocol
3. Optional edge refinement (ViTMatte / guided filter)
4. Post-processing (hole fill, small region removal)

**Key advantages over SAM + ViTMatte:**
- SeC uses a Vision-Language Model for semantic understanding (text prompts)
- MatAnyone2 provides temporal consistency across video frames
- Better for long sequences with occlusions and re-appearances

---

#### Semantic Segment (MEC)

Face / body / clothes semantic parsing using SegFormer. Select classes by name to build a combined mask.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | — | `segformer_face` (19-class facial) or `segformer_clothes` (18-class apparel) |
| `classes_csv` | `"skin,hair"` | Comma-separated class names to include |
| `threshold` | `0.5` | Confidence threshold |
| `invert` | `false` | Invert output mask |
| `allow_download` | `false` | Download SegFormer if it is not installed; off without the model: `skin` by colour only |

**Face classes:** skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip, neck, necklace, cloth, hair, hat

**Clothes classes:** hat, hair, sunglasses, upper_clothes, skirt, pants, dress, belt, left_shoe, right_shoe, face, left_leg, right_leg, left_arm, right_arm, bag, scarf

**Output:** `mask`, `info`

> **Who is it for?** Portrait retouchers who need to isolate specific facial features or clothing items without manual masking.

---

#### Remove background (a backend of Mask + Matting)

One-click background removal using RMBG-2.0 or BiRefNet. Since L7.65 P05 this is Mask + Matting (`MaskOpsMEC`) with
`backend` = `remove background`; saved Background Remover nodes migrate into it when a workflow loads, with the same
results.

| Model | Quality | Best for |
|-------|---------|----------|
| `rmbg_2.0` | ★★★★☆ | General-purpose, fast |
| `birefnet_general` | ★★★★★ | High-detail edges |
| `birefnet_portrait` | ★★★★★ | Human portraits |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bg_model` | — | Background removal model selector |
| `bg_threshold` | `0.5` | Alpha threshold (0 = soft, 1 = hard) |
| `bg_invert` | `false` | Keep background instead |
| `bg_mask_blur` | `0` | Gaussian blur on mask edges |
| `auto_download` | `false` | Download the model if it is not installed; off without it: a labelled model-free cut-out |

**Outputs (Mask + Matting):** `foreground` (premultiplied), `alpha` (was mask), `info` (with `model_status`), plus the
other Mask + Matting outputs.

> **Who is it for?** E-commerce product photography, quick compositing, batch background removal.

---

#### Luma key — Mask Tools (C2C)

In **Mask Tools (C2C)** with `mode` = `key: luma` (this was the Luminance Keyer node; saved workflows migrate automatically). Its controls carry the prefix `luma_key_`, e.g. `luma_key_mode`, `luma_key_low`.

Professional luminance keyer inspired by Nuke's LumaKeyer. Extracts a matte based on image brightness using ITU-R BT.709 luminance with smooth S-curve falloff and gamma correction.

**How it works:** Converts the image to BT.709 luminance (`0.2126R + 0.7152G + 0.0722B`), then applies a threshold range with Hermite smoothstep falloff. Five modes target different brightness ranges — **auto** mode analyzes the image and picks the best range automatically.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | — | Input image(s) to key |
| `mode` | `auto` | `auto` / `highlights` (0.7–1.0) / `midtones` (0.3–0.7) / `shadows` (0.0–0.3) / `custom` |
| `low` | `0.0` | Low threshold (custom mode only) |
| `high` | `1.0` | High threshold (custom mode only) |
| `gamma` | `1.0` | Gamma correction — >1 shrinks mask, <1 expands |
| `falloff` | `1.0` | Edge smoothness — 0 = hard binary, 1 = smooth, >1 = very gradual |
| `invert` | `false` | Flip the mask |

**Outputs:** `mask`, `info` (mode, thresholds, luminance stats, per-frame coverage)

> **Who is it for?** VFX artists pulling luminance keys (sky replacement, highlight isolation, shadow grading), colorists building luminance-driven masks for selective color grading, anyone who needs brightness-based masking without a model.

---

#### Mask Composite Advanced (MEC)

Combine two masks with compositing operations.

| Operation | Description |
|-----------|-------------|
| `union` | Combine both (max) |
| `intersect` | Overlap only (min) |
| `subtract` | A minus B |
| `xor` | Exclusive — one or the other, not both |
| `blend` | Weighted average |
| `min` / `max` | Per-pixel min / max |
| `difference` | Absolute difference |

Automatically resizes masks to match if spatial dimensions differ.

---

#### Mask Math (MEC)

Mathematical operations on a single mask.

| Operation | Description |
|-----------|-------------|
| `add_scalar` | Add constant value |
| `multiply_scalar` | Multiply by value |
| `power` | Raise to power |
| `invert` | `1 − mask` |
| `clamp` | Clamp to `[value_a, value_b]` range |
| `remap_range` | Remap from `[value_a, value_b]` → `[0, 1]` |
| `quantize` | Snap to N discrete levels |
| `threshold_hysteresis` | Dual-threshold with connected regions |
| `gamma` | Gamma correction |
| `contrast` | Contrast adjustment |
| `abs_diff_from_value` | Absolute difference from value |

All outputs are clamped to `[0, 1]`.

---

### Inpainting

Two nodes: **Inpaint Crop Pro** crops, **Inpaint Stitch Pro** puts the result back. Since L7.65 P08 they also do
what five other nodes did (Smart Image Crop, Inpaint Mask Prepare, Smart Image Stitch, Inpaint Paste Back, Context
Inpainter); saved workflows with those nodes load migrated, links and values kept (docs/MIGRATION.md). API prompts
follow the same rule as the front end: only ids that are no longer registered are rewritten.

#### Inpaint Crop Pro

Crops the image around the mask so the inpaint model works on the region at its own resolution; **Inpaint Stitch
Pro** puts the result back. Same workflow shape as lquesada's Inpaint Crop and Stitch, with Wan 2.2 Animate extras.

**Modes** (`crop_mode`, the node shows only the chosen mode's controls):

| crop_mode | What it does | Use it with |
|-----------|--------------|-------------|
| `pro` (default) | The crop described below (video aware, fixed target size) | Stitch Pro `blend` or `paste back` |
| `smart aspect` | The former Smart Image Crop, for stills: a faint mask still counts, holes are closed, the window takes the output's aspect, and the output size is picked between `smart_min_resolution` and `smart_max_resolution` (or `smart_manual_width` x `smart_manual_height`), snapped to `smart_force_divisibility`. `smart_no_mask_mode` decides what an empty mask does (Bypass / Resize Full Image / Crop Full Image) | Stitch Pro `smart` (with `original_image`) |
| `mask prepare` | The former Inpaint Mask Prepare: no crop. Fills holes, removes small regions, grows, and makes the inpaint mask and a stitch blend mask for the whole frame (`prepare_*` controls; `image` is optional and is the reference for `edge_aware`) | Stitch Pro `blend` (a whole-frame stitcher when an image is wired) |

**What `pro` does, in order:** optional pre-resize of the whole image; mask clean-up (fill holes, expand, invert,
feather, hi-pass); optional outpaint extension; the context box = the mask's bounding box grown by
`context_from_mask_extend_factor` (plus `optional_context_mask`); fit to the target aspect, rounded to
`output_padding` and `wan_align_multiple`; resize to the target size.

**The crop window stays on real pixels.** A window that fits inside the image is moved inside it (the least
distance); only a window larger than the image is padded with edge pixels. (Before 2026-10-13 a mask near the border
got a window padded with smeared edges - 41 % of a 16:9 window - although one fitting the image existed.)

**Preview.** After a run the node shows one picture: the original with the discarded area dimmed, the regenerated
area tinted and the window outlined (an orange side = padded past the image), next to the crop the model gets, with
the window size, position, output size, scale and padding written under it. It is drawn from the window that ran, in
every mode (`smart aspect`: the Smart window per frame; `mask prepare`: the whole frame, next to the two masks).
`preview_frame` picks the frame shown (default 0, the first); it changes nothing else.
After that first run the picture follows the crop settings as you change them - including `crop_mode` and the
smart / prepare controls - without running the graph (the same geometry, from the last run's masks; marked "live";
Settings > C2C > Canvas > Inpaint crop > Live crop preview turns it off). The node's outputs change when it runs.
Inside a subgraph, or for another preview frame, the picture changes on the next run.

**Batches and video.** `wan_stable_crop` on (default): one window for every frame - the union of all frames'
masks - so the crop does not jump. Off: each frame gets its own window, and every frame is cropped and stitched with
its own placement. The preview label says which. (`smart aspect` always gives each frame its own window and one
output size for the batch, from the largest mask.)

**Controls.** Ten are shown in `pro`: `mask_expand_pixels`, `mask_blend_pixels`, `context_from_mask_extend_factor`,
`aspect_preset`, `output_target_width`, `output_target_height`, `wan_stable_crop`, `stitch_blend_mode`,
`preview_frame`, `crop_mode`. The rest sit in sections - Mask, Size, Resize / outpaint, Video, Stitch, Device - opened
from the node's **More** bar (classic canvas) or core's **Show advanced inputs** (Nodes 2.0). A switch shows its own
controls when it is on (pre-resize sizes, outpaint factors, the `video_stable` settings). In `smart aspect` and
`mask prepare` only `crop_mode`, that mode's controls and `preview_frame` are shown.

**Output size.** The stitched image has the size of the image the crop worked on: the input size, or the
pre-resized size with `preresize`, or the extended size with `extend_for_outpainting`.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `image`, `mask` | — | Image batch and the area to regenerate (white). `optional_context_mask` widens the context box. `image` is optional only for `mask prepare` |
| `context_from_mask_extend_factor` | `1.2` | Context box size relative to the mask box; below 1 crops tighter than the mask box |
| `auto_context_factor` | off | Picks the factor so the mask fills about 70 % of the crop |
| `aspect_preset` | `Custom` | `Custom` = the target width / height as set; `Auto` or a named ratio keeps the shorter side and derives the other |
| `output_resize_to_target_size`, `output_target_width`, `output_target_height` | on, 512, 512 | Size the crop is resized to |
| `output_padding`, `wan_align_multiple` | 32, 16 | The crop size is rounded up to a multiple of the larger |
| `mask_fill_holes`, `mask_expand_pixels`, `mask_invert`, `mask_blend_pixels`, `mask_hipass_filter` | on, 0, off, 32, 0.1 | Mask clean-up before the box is measured (blend and hi-pass shape the soft mask only) |
| `preresize`, `preresize_mode`, `preresize_min/max_width/height` | off | Resize the whole image first (minimum / maximum / both) |
| `extend_for_outpainting`, `extend_up/down/left/right_factor` | off, 1.0 | Grow the canvas for outpainting; the new area is masked |
| `wan_stable_crop` | on | One window for the whole batch (see above) |
| `wan_temporal_smooth_frames` | 0 | Smooth the mask over time (frames) |
| `wan_mask_polarity` | `regenerate_subject` | `preserve_subject` = Wan replacement masks (1 = keep) |
| `inpaint_mask_mode` | `hard_binary` | The mask the model sees: `hard_binary`, `slight_feather`, `soft_blend` |
| `stitch_blend_mode`, `blend_radius` | `gaussian`, 32 | How the result is blended back: `gaussian`, `edge_aware`, `laplacian_pyramid`, `frequency_blend`, `video_stable` |
| `video_stable_temporal_sigma`, `video_stable_dilate_px`, `video_stable_blur_sigma` | 3.0, -1, -1 | `video_stable` only |
| `fill_masked_area` | `none` | Fill the masked area of the crop: `edge_pad`, `neutral_gray`, `original` |
| `roto_quality` | off | Tighter seam for roto / compositing (Laplacian blend, half radius, 1 px erosion) |
| `device_mode`, `downscale_algorithm`, `upscale_algorithm` | gpu, bilinear, bicubic | Where it runs and the resampling filters |
| `preview_frame` | 0 | Frame shown in the preview (display only) |
| `crop_mode` | `pro` | See Modes |
| `smart_resolution_mode`, `smart_min_resolution`, `smart_max_resolution`, `smart_manual_width`, `smart_manual_height` | Automatic, 768, 2048, 1024, 1024 | `smart aspect`: the output size (Automatic = between min and max, aspect kept; Manual = the manual size) |
| `smart_mask_grow_pixels`, `smart_patch_mask_holes`, `smart_no_mask_mode`, `smart_force_divisibility` | 32, on, Bypass, 128 | `smart aspect`: grow (or shrink, negative) the mask, close holes, what an empty mask does, the size multiple |
| `prepare_fill_holes`, `prepare_remove_small_regions`, `prepare_min_region_area`, `prepare_grow_pixels` | on, on, 100, 4 | `mask prepare`: clean-up |
| `prepare_inpaint_edge_mode`, `prepare_stitch_edge_mode`, `prepare_stitch_feather_radius` | hard_binary, gaussian, 16 | `mask prepare`: the two masks' edges |
| `prepare_temporal_smooth`, `prepare_temporal_sigma` | off, 1.5 | `mask prepare`: smooth the blend mask over time |

**Outputs:** `stitcher` (to Stitch Pro), `cropped_image`, `inpaint_mask` (for the model), `stitch_blend_mask`,
`info` (settings and frame 0's window), `preview` (`pro`: the preview picture; `smart aspect`: the image with the mask
tinted and the window drawn; `mask prepare`: R = inpaint mask, G = stitch blend mask). In `smart aspect`
`stitch_blend_mask` is the crop mask (what the Smart stitch's Mask Feather blends along); in `mask prepare` without
an image `stitcher` and `cropped_image` are empty.

---

#### Inpaint Stitch Pro

Puts the inpainted crop back into the image Inpaint Crop Pro worked on, frame by frame, at the window each frame
was cropped from.

**Modes** (`stitch_mode`, the node shows only the chosen mode's controls):

| stitch_mode | What it does | Needs |
|-------------|--------------|-------|
| `blend` (default) | The crop's blend mask and blend modes, colour match, temporal smoothing (as before) | a stitcher from `pro` or `mask prepare` |
| `paste back` | The former Inpaint Paste Back: resize the crop and paste it, optionally with a feathered rectangle edge (`paste_upscale_method`, `paste_feather_edges`, `paste_feather_radius`) | a stitcher from `pro` or `mask prepare` |
| `smart` | The former Smart Image Stitch: Box Feather, Mask Feather (along `mask`) or Hard Paste, optional colour match (`smart_*` controls) | a stitcher from `smart aspect`, and `original_image` |
| `context blend` | The former Context Inpainter: no stitcher; blends a full-frame inpainted image over `original_image` under `mask`, with colour correction (Reinhard mean / std), lightness rescue (CIE LAB L when the inpaint is more than about 5 % darker) and optional differential diffusion (`context_*` controls). It blends every frame of a batch, and `context_color_reference` picks where colour match and lightness rescue look (inside the mask, or around it for object removal), as the node does since L9.39 | `original_image`, `mask`, `inpainted_image` |

A stitcher from the other kind of crop is refused with a message that names the mode to pick.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `stitcher` | — | From Inpaint Crop Pro (not used by `context blend`) |
| `inpainted_image` | — | The sampler's result (RGB; an RGBA plate keeps its own alpha) |
| `blend_mode_override` | `from_crop` | Use another blend mode than the crop chose (`blend`) |
| `color_match` | off | Mean + std colour transfer before blending (`blend`) |
| `stitch_temporal_sigma`, `stitch_dilate_px` | 0, 0 | Smooth / widen the blend mask over time for jittery video masks (`blend`) |
| `roto_quality_override` | `from_crop` | `force_on` / `force_off` the crop's roto setting (`blend`) |
| `stitch_mode` | `blend` | See Modes |
| `original_image`, `mask` | — | `smart` and `context blend` |
| `paste_upscale_method`, `paste_feather_edges`, `paste_feather_radius` | bicubic, off, 16 | `paste back` |
| `smart_feather_pixels`, `smart_blend_mode`, `smart_resize_full_image_output`, `smart_enable_color_match`, `smart_color_match_amount` | 32, Box Feather, Restore Original Size, off, 0.35 | `smart` |
| `context_crop_padding`, `context_blend_softness`, `context_mask_expansion_blend` | 1.2, 8, 0 | `context blend`: context box, feather radius, grow / shrink of the blend mask |
| `context_enable_color_correction`, `context_enable_lightness_rescue`, `context_enable_differential_diffusion` | on, on, off | `context blend` |
| `context_sampling_mask_blur_size`, `context_sampling_mask_blur_strength` | 21, 1.0 | `context blend`: blur of the `blend_mask_used` output |

**Outputs:** `image`, `blend_mask_used` (`blend`: exactly where the stitch changed pixels - everywhere it is 0 the
original is untouched, bit for bit; `paste back` / `smart`: the alpha that was applied; `context blend`: the
Context Inpainter's debug mask), `info`.

Checked by `tests/test_inpaint_crop_geometry.py`: an empty mask stitches back as an exact identity; nothing outside
the blend mask changes for every blend mode; without resizing the round trip is exact; the result lands where it
was cropped; RGBA, odd sizes, `preserve_subject`, per-frame and stable batches. The four modes that replaced other
nodes give those nodes' results bit for bit (`tests/test_consolidation_inpaint.py`, against the code as shipped at
d97cc13).

---

#### Load Video (C2C): images, EXRs and sequences (2026-10-31)

Load Video (C2C) loads every source a shot starts from, without converting it to a movie first:

- a movie: mp4, mov, mkv, webm, avi, mxf and the other containers PyAV reads;
- one still image: png, jpg, tif / tiff, bmp, tga, hdr, webp, dpx;
- one EXR (half or float, alpha, any layer);
- a numbered image or EXR sequence.

The image formats offered are the ones this ComfyUI can actually read: OpenImageIO for EXR / DPX, otherwise OpenCV.

- **Pick or upload.** The `video` list shows movies, stills and each sequence once, as its pattern
  (`plates/shot.####.exr`), from the input folder and one level of subfolders.
  - *Upload files…* takes several files at once; a set of frames lands in its own subfolder as one sequence.
  - *Upload folder…* keeps the folder's name. You can also drop files on the node.
  - A same-named file with the same bytes is reused. A different one gets a new name. A sequence is never renamed
    frame by frame: it goes to a new folder `shot (2)/`, unless *overwrite* is ticked.
  - Every upload is checked by its first bytes as well as its extension. An `.exr` that is really a PNG is refused.
- **Server path.** `location` = *server path* reads any folder: a clip, an image, a folder, or a pattern (`####`,
  `%04d`, `$F4`, `@@@@`, `[1001-1100]`). This is what **Load Video Path (C2C)** did. That node is merged into this one,
  and saved Load Video Path nodes and API prompts are migrated automatically (same path, same frames).
  - The browser-facing routes (probe, preview, scan) only read inside ComfyUI's input / output / temp folders and the
    folders you list in the `C2C_MEDIA_ROOTS` environment variable or in `c2c_media_roots.txt` in the ComfyUI folder.
  - A queued run reads any path, unless ComfyUI listens on a network address (`--listen`). Then runs keep to those
    folders too.
- **Source mode.** *auto*:
  - a movie is a movie;
  - one frame of a numbered set finds its whole sequence (as a compositor's Read does);
  - a lone image is one frame.

  *single image* / *single EXR* load exactly one file even when it has siblings (`frame_range` picks which frame).
  *image sequence* / *EXR sequence* force a sequence. A still is one frame at `sequence_fps` and is never encoded
  into a movie.
- **Frame numbers.**
  - A sequence keeps its own numbers. 1001-1100 loads as 100 frames and `frame_map` still says 1001..1100.
  - `frame_range` takes Nuke's syntax: `1001-1100`, `1001-1100x2`, or one frame like `1050`.
  - `reverse` plays the selection last to first, audio too.
  - `frame_offset` renumbers `frame_map`'s output frames.
  - A render on twos (1001, 1003, ...) is a step, not missing frames. `1001-1100x1` holds the in-between frames at
    the full rate.
- **Missing frames.** A hole inside a sequence stops the run by default and names the missing numbers. *black*,
  *hold previous* and *nearest* fill it (and say so). *skip (as before)* drops it, which is the old behaviour: every
  later frame then plays one frame early.
  - Frames asked for beyond the first / last follow `edge_policy`: *hold*, *black*, *loop*, *bounce* or *error*.
  - The padding comes from all the files, not from the frame you picked: an unpadded `u.1 … u.100` loads whole from
    `u.10`.
  - Files with other padding or another extension under the same name are listed as other candidates. They are
    never merged.
- **EXR layers.** `exr_layer` picks what IMAGE shows: *auto* (the beauty / rgba), or any layer the info lists, such as
  diffuse, depth, N, P or crypto00. The node shows the layers as chips; click one to load it.
  - Multipart EXRs list each part.
  - Data layers (depth, normals, position, motion, cryptomatte) load as raw values, are never colour-managed, and
    their preview is normalised for the eye only.
  - Floats are never clamped; negatives and alpha are kept.
  - `exr_passes` plugs into NukeMax's **Shuffle Pass**, which gives any layer for every loaded frame.
- **Preview.**
  - A still image or a single EXR shows as a picture, not as a one-frame clip.
  - A sequence has a *contact sheet* chip: 16 frames spread over the selection, each labelled with its frame number.
    Click the sheet to go back to the player.
- **New outputs, appended:**
  - `frame_map`: output frame to source frame and its frame number (contract `h3.frame_ledger` v1.1; -1 = a black
    frame);
  - `exr_passes`: every EXR layer, lazily, as `C2C_EXR_PASSES`;
  - `metadata`: the file's header or the movie's tags, as JSON.

  `info` now also says what the source is, its numbers, missing frames and layers, and how much memory the IMAGE
  takes. The `video` output stays lazy for every source: nothing is decoded until a node reads a window.
- **Saved workflows.** Every new input is appended and defaults to the old behaviour, so a saved movie workflow loads
  the same frames.
  - Kill switches: `C2C_MEDIA_LIST=videos` brings back the movie-only list; `C2C_FRAME_CACHE_MB=0` turns off the
    decoded-frame cache (512 MB by default) that overlapping window reads share.
  - `C2C_UPLOAD_MAX_MB` caps one upload (20000 by default).

#### Load Video (C2C): player, frame selection and crop

The node plays its video in the node body. Pro formats (ProRes, DNxHR, HEVC 10-bit, FFV1) are transcoded for the
preview on the server. The preview has a keyframe every 12 frames, so seeking and scrubbing stay quick.

- **Transport:** play / pause, play backwards, previous / next frame, first / last frame of the selection.
- **Speed:** 0.25x, 0.5x, 1x, 2x, 4x (click the speed button; Shift+click or right-click to slow down).
- **Loop mode:** loop the selection, loop the whole clip, play once, or ping-pong inside the selection.
- **Scrubber and ruler:** the bar shows the whole clip with a ruler in frames or in time (the *frames / time*
  button). Click or drag to seek. Hover to see that frame in a small thumbnail without moving the playhead.
- **Go to a frame:** click the frame number, type a frame or a timecode (`hh:mm:ss:ff`), press Enter.
- **Choosing frames:** drag the two handles on the scrubber, or press I / O at the playhead. They set
  `skip_first_frames` and `frame_load_cap`. The label under the bar says how many frames the run will load and how
  long they last (`select_every_nth` included). Typing in the widgets moves the handles. *all* selects the whole clip.
- **Crop:** click *crop*, then drag the box or its handles on the preview. Pick a shape (Free, Original, 1:1, 4:5,
  5:4, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 2:1, 1:2). Click *crop* again to finish; *no crop* removes it. The label shows
  the crop in pixels. The run decodes each frame, cuts the box out, then applies `custom_width` / `custom_height`.
  The preview keeps showing the whole frame with the outside shaded. The crop is stored in four hidden widgets
  (`crop_x`, `crop_y`, `crop_w`, `crop_h`, fractions of the frame), so it is saved with the workflow.
- **Full screen:** the ⛶ button or F.
- **Keys** (click the player first): Space play / pause, Left / Right step (Shift: 10 frames), J / K / L
  backwards / stop / forwards (press J or L again for 2x, 4x), Up / Down speed, Home / End, I / O set in / out,
  F full screen.
- The player works the same in server path mode (crop included) and in Load Images Path (C2C) (no crop), in both
  renderers, and inside subgraphs.
- When ComfyUI is reachable from other machines (`--listen`), Load Images Path (C2C) reads only folders inside the
  C2C media roots, the same rule as Load Video's server path mode (add more folders with `C2C_MEDIA_ROOTS`).

#### Load Video (C2C): outputs are decoded only when something reads them

Load Video / Load Video Path / Load Images Path (C2C) decode IMAGE and MASK only if a node reads them; otherwise
those outputs are empty and the run only probes the clip. Linking IMAGE later, without changing a widget, now runs the
loader again. Until 2026-10-23 ComfyUI reused the empty output it had cached, and the next node failed on `None`
(for example AV Handles Add switched to audio-only and the resize after it crashed).

#### Load Video (C2C): 4K path, presets and readable settings

- **Formats** (`format`) set the frame grid and the size multiple a model needs:
  - AnimateDiff, Mochi, LTXV, Hunyuan, Cosmos and Wan.
  - **MiniMax H3**: 24 fps, 17n+5 frames (5, 22, 39, 56 ...), sizes in multiples of 32.

  Under a grid, the `frame_load_cap` arrows step through the legal counts, and a typed count snaps to the nearest
  legal one. A preset never forces the frame rate. When the clip's rate differs, the node says which `force_rate` to
  set.
- **What loads:** `seconds_cap` caps by time; `frame_load_cap` and `seconds_cap` clear each other. `resolution` sizes
  by the longest side (or `custom` = `custom_width` / `custom_height` as before), rounded to `divisible_by`.
  `audio_when_missing` gives a silent track (or none) when the clip has no audio.
- **Reading the settings:**
  - Beside each widget, a grey value shows what will actually load, with a ↺ arrow that resets the widget. In Nodes
    2.0 these are chips in the panel.
  - *reset all to defaults* keeps the clip.
  - A details line reads, for example, "loads 0.0s to 0.9s (0.9s, 22 frames) at 2048x1088". It is the server's own
    plan, so it always agrees with the player's "N will load" and with the run.
  - Picking a new clip resets the settings; reopening a workflow keeps them.
- **`decoder`:** `exact` is the PyAV reader. `fast` is an ffmpeg pipe that is bit-identical in `rgb48`, or nv12 on
  CUDA (mean 0.47/255 from exact). `auto` uses `exact`: on the test CPU the ffmpeg path was not reliably faster.
  `C2C_VIDEO_AUTO=fast` or `C2C_VIDEO_AUTO=gpu` lets auto take the fast path.
- **`info`** output: a JSON summary of what loaded (frames, seconds, size, fps, decoder).
- **8K:** UHD 7680x4320 and DCI 8192x4320 are tested.
  - The fast path gives the exact reader's pixels.
  - A selection that does not fit in RAM is refused before decoding, with how many frames would fit.
  - Save Video writes 8K as H.265, ProRes 422 HQ, FFV1 and H.264.
  - The tests are opt-in: `C2C_TEST_8K=1 pytest tests/test_c2c_video_8k.py`.

#### Save Video (C2C): ffmpeg encode, NVENC, workflow PNG

- Encodes through an ffmpeg pipe. On CPU, H.264 / H.265 / ProRes / FFV1 are 2-4x faster than before; DNxHR and VP9
  are about 1.2x.
- Formats include H.264, H.265 8 / 10-bit, AV1 (NVENC on NVIDIA GPUs), ProRes 422 / HQ / 4444 / XQ, DNxHR, FFV1,
  VP9, GIF, WebP, and PNG / EXR sequences. A format your ffmpeg cannot write is marked `[unavailable]`.
- NTSC rates are exact (23.976 → 24000/1001).
- `save_metadata` (on by default) writes `<name>.png` next to the saved file: its first written frame, with the
  workflow and prompt inside, as Save Image stores them. Drop the PNG onto ComfyUI to rebuild the graph. For an
  image sequence, the PNG sits beside the frame folder. The video itself never carries the workflow (owner,
  2026-10-25). The encoder drops `workflow` / `prompt` whoever calls it; tests scan the raw MP4 / MOV / MKV bytes.
  Nothing is written when ComfyUI runs with `--disable-metadata`.

Ported in part from [ComfyUI-4K-Video-Save-Load](https://github.com/HallettVisual/ComfyUI-4K-Video-Save-Load) (MIT);
see CREDITS.md.

#### Save Video (C2C): format controls, naming, player, built-in handles (2026-10-23)

- **Format in five controls.** With `format` = *custom* (the default for new nodes) you choose **container ->
  codec -> profile -> bit depth -> chroma**, and each list shows only what the choices before it allow: MOV offers
  ProRes, DNxHR, CineForm, H.264 / H.265, PNG, QuickTime Animation and uncompressed 10-bit; ProRes 4444 offers 10-bit
  4:4:4. The lists come from what this machine can actually write (ffmpeg and PyAV are checked once); a combination
  it cannot write says why. New: AV1 (SVT), CineForm, UT Video, QuickTime Animation, PNG in MOV, uncompressed 10-bit,
  ProRes in MKV, H.264 / H.265 in MOV and MKV, H.264 4:2:2 / 4:4:4, H.265 Main 12 / 4:2:2 / 4:4:4, VP9 profiles 1-3,
  and TIFF / DPX / JPEG sequences next to PNG / EXR (EXR and TIFF with a choice of compression).
- **Colour tags:** `color_primaries` (BT.709, BT.2020, Display P3, BT.601), `color_transfer` (BT.709, sRGB, linear,
  PQ, HLG, gamma 2.2) and `color_range` (limited / full). For YUV codecs the conversion follows the choice (BT.2020
  primaries convert with the BT.2020 matrix, full range converts to full range), so the file says what was done to
  it. *auto* keeps the earlier behaviour.
- **Saved workflows** keep encoding exactly as before. A workflow saved with the old single list (for example
  "MOV ProRes 422 HQ") opens converted to the same five choices, and the node says so.
- **Named formats are presets.** Picking one in `format` (for example "EXR sequence (half)") fills the five controls
  and keeps them on screen. Until 2026-10-25 a named format hid every option. Colour tags show only where the codec
  takes them, so EXR / PNG / DPX sequences hide them.
- **Naming:** *Folder Version Incrementer* (and *default*, unless you change the C2C setting) takes the version
  folder and name from the Folder Version Incrementer node in the graph. No wire is needed, and its settings are
  used. Without one, the incrementer's defaults are used. A wired `subfolder` still wins. The node shows the path the
  next write will use, and a version is used only when a file is written.
- **Player:** Load Video's player on the whole clip (play, backwards, J / K / L, step, loop, timecode, full screen).
  Its In / Out handles set the **export range** (`range_in` / `range_out`): the next run writes only those frames.
  The transport is one row of buttons in every player; the node never gets narrower than that row needs. Before
  the first run the node shows one line instead of an empty player.
- **A/B:** compare the saved file with the frames that came in. Wipe (vertical, horizontal, diagonal), difference
  with gain, onion, flicker, zoom (mouse wheel), pan, 1:1, and a loupe showing pixel values.
- **Preview first:** with `write_mode` = *preview first*, a run shows exactly what would be written and writes
  nothing. *Write file* in the player then writes it. Only the encode runs again; the graph above is cached.
- **Handles:** `handles_trim` = *auto* trims exactly the handles Load Video (C2C) added, read from `video_info`. If
  Load used *fit*, it trims back to fit's target length. *manual* trims `trim_frames` on `trim_side`; *off* trims
  nothing. *auto* never uses a count remembered from another run.
- **No `video_info` wire needed** (2026-10-25). Save follows its `images` (or `latent`) input back through each
  node's first image input, subgraphs included, to the Load Video it came from. It then uses that Load's
  `video_info`: the handles to trim and its frame rate. It uses it only if that Load ran with its current settings.
  Otherwise the node says why nothing was trimmed. A wired `video_info` always wins. Measured: Load (+8 handles) ->
  Save with no wire, in a subgraph, wrote the 48 source frames 0-47 with the audio exactly in place.

#### Save Video (C2C): Format -> Subformat -> options (2026-10-26)

- **Format, then Subformat.** `format` is *Video*, *Image Sequence* or *Custom*. *Video* lists H.264 / MP4,
  H.265 / HEVC / MP4, the six ProRes profiles (422 Proxy, LT, 422, HQ, 4444, 4444 XQ), DNxHR (LB / SQ / HQ / HQX /
  444) and FFV1 / MKV. *Image Sequence* lists OpenEXR, DPX, PNG, TIFF and JPEG. These controls and the options for
  the chosen subformat are the first rows on the node. The less-used options sit under **Advanced**.
- **Only that subformat's options.**
  - OpenEXR: compression (ZIP, PIZ, ZIPS, DWAA, DWAB or none), precision (16 = half, 32 = float) and channels.
  - H.264 / H.265: quality (CRF), encoder preset, profile, bit depth and chroma. Under Advanced: keyframe interval
    (GOP), bitrate and the colour tags.
  - ProRes: the profile. Channels appear only for 4444 / 4444 XQ, the profiles that can carry alpha. Picking another
    profile moves the subformat with it, so "ProRes 422 HQ" never writes 4444.
  - PNG: bit depth, channels and compression level.
  - DPX: bit depth.
- **Channels:** *auto* keeps alpha when the images have it. *RGB* drops it. *RGBA* writes an opaque alpha. On a
  format with no alpha, the file is written as RGB and the node says so.
- **Remembers each subformat:** switch from OpenEXR / DWAB to PNG and back, and OpenEXR is on DWAB again.
- **Custom** is the full container -> codec -> profile -> bit depth -> chroma cascade from 2026-10-23.
- **Saved workflows:** a workflow with the old single list ("MOV ProRes 422 HQ") opens on its subformat and encodes
  the same file. Values are saved in input order and restored by name, so the new layout loads in every frontend.
  Checked live on frontend 1.53.10 in both renderers and inside a subgraph: an OpenEXR / DWAB / float run wrote a
  DWAB float EXR, and a save -> reload round trip kept every value.

#### Save Video (C2C): which Load Video it follows (2026-10-26)

- **Load ▾ picker.** Save Video takes handles, frame rate, colour and (next) the name from one Load Video. *auto*
  follows the Load reached through each node's first image input, never execution order. The node shows which one
  before a run ("Follows Load Video #12 - shot.mov"), live, through subgraphs in both directions.
- **More than one Load upstream** (for example a blend of two plates): the first-input Load is used, and a warning
  names the others until you pick one in the Load ▾ menu. A pick stays until the graph changes. If the picked Load is
  no longer upstream, Save goes back to auto and says so.
- The server applies the same rule at run time and reports it in the result, so API runs behave the same. A wired
  `video_info` still wins. The pick is stored in the new `follow_load` input (appended; saved workflows = auto).
- Checked live on frontend 1.53.10 in both renderers: a single chain, two Loads, the Loads inside a subgraph (ids
  `5:1` / `5:3` - the server followed exactly `5:3`), and Save inside a subgraph.

#### Save Video (C2C): review copy + burn-in (2026-10-26)

- `review_copy` (off by default) also writes `<name>_review.mp4` next to the file, or next to a sequence's folder.
  It is H.264 at half size, in the OCIO display (sRGB - Display; frames in another space go through the config's
  view), for review only.
- **Burn-in, on the review only** (the master stays clean). *standard*: name + version (top left), source frame +
  timecode (top right), colour space (bottom left), date (bottom right). *custom*: `burn_in_template`, four corners
  separated by `|`, with `{shot} {version} {frame} {tc} {date} {artist} {colour}`. *off*: none. The frame and
  timecode are the source's when Save carried them.
- It is written from the same frames chunk by chunk, so there is no second decode and no full-size copy. A streamed
  save (latent + vae) gets no review copy, and says so.
- Checked live on frontend 1.53.10 in both renderers: the burn-in settings show only when needed. A 960x540 plate
  gave a 480x270 review with a readable burn-in: "1001 01:00:00:00" from the source.

#### Save Video (C2C): source frame numbers, timecode, pixel aspect, reel (2026-10-26)

- **A sequence keeps its source frame numbers.** Load `shot.1001-1100.exr` with handles, and Save writes
  `.1001`-`.1100` again: the handles are trimmed and In / Out keeps its offset. With skip or every-Nth, the frames keep
  their own source numbers.
- **Timecode.** The first written frame's source timecode goes into the file: a timecode track in MOV / MP4 / MKV,
  `smpte:TimeCode` per frame in EXR / DPX. Drop-frame (29.97 / 59.94) is counted per SMPTE.
- **Pixel aspect** goes in as the video SAR and the EXR / DPX / TIFF pixel aspect. The **reel name** goes into MOV
  metadata and the EXR `reelName` (DPX has no field for it).
- **Where the mapping is not exact**, Save numbers from 1 with no timecode and says why: a retimed or unevenly skipped
  load, a sequence with gaps, handles not trimmed as Load added them, or more frames than the source selection.
  Camera vendor metadata is not carried yet: it needs real camera files.
- Checked by tests (EXR sequence with skip, handles and In; a ProRes MOV with timecode, 2:1 pixel aspect and reel,
  through both encode paths and MP4), and live on frontend 1.53.10 in both renderers: files `.1003`-`.1010`, timecode
  `01:00:00:02`.
- **Old workflows:** a Load Video saved before its name fields existed now opens with them empty. The generic
  migration had let the upload button's value land on a new text input.

#### Save Video (C2C): the file named after the Load Video (2026-10-26)

- New Save nodes start with `filename_prefix` = *auto (from Load)*. The file is named after the Load Video they
  follow: its `name_prefix` + the source's name + its `name_suffix`. Both fields are new on Load Video and empty by
  default, so nothing is added unless you add it.
  - The source's name loses its folders, extension and sequence frame token: `D:/plates/shot010.####.exr` ->
    `shot010`. A movie keeps its digits: `shot_010.mov` -> `shot_010`.
  - The source path is never written.
- **With the version folder** (the default), the file carries the folder's version. `shot010_comp_v003.mov`
  becomes `.../v004/shot010_comp_v004.mp4`, and a name without a version gets `_v004`. The next write takes the next
  version, in the folder and the name. With *ComfyUI counter* the file is `C2C/<name>_00001`.
- Typing a name on Save wins. A wired `video_info` names the file after the Load it comes from. Without a Load the
  file is `video`, and the node says why.
- The next-path line shows the name before a run and updates when the Load's name fields change. It is right
  inside subgraphs too.
- Saved workflows keep their names (their `filename_prefix` is not *auto*).
- Checked live on frontend 1.53.10 in both renderers: the preview showed `proj_plate_a_comp_v003.mp4`, the run
  wrote exactly that file in `v003`, and the preview then moved to `v004`.

#### Load / Save Video (C2C): colour management, Nuke / OCIO style (2026-10-26)

- **Load Video finds the file's colour space** (`colorspace_in` = *auto (detect)*), from the most reliable source first:
  1. a `<file>.colorspace` or `<file>.c2c.json` sidecar (a sequence also takes `<stem>.colorspace`);
  2. the EXR `acesImageContainer` or `colorInteropID` (the ASWF Color Interop ID);
  3. the EXR chromaticities;
  4. the DPX transfer or the image header;
  5. the movie's colour tags.

  An EXR with no colour metadata is read as the OCIO config's `scene_linear` role (Nuke's working space). An
  untagged movie or image is read as display and not converted. Pick a space to override.
  *Since 2026-11-10 (L9.177):* this works with **any** OCIO config. Before, the spaces were looked up only by their
  ASWF interop IDs, which only OCIO 2.5's v4 built-in configs carry. With the *studio / cg ACES 1.3* built-in choice,
  an older built-in, Nuke's `nuke-default` or an ACES 1.x `$OCIO`, a Rec.709 MOV from Nuke read "Colour: unknown".
  Now the space is also found by the names it has in those configs, or by OpenColorIO's own numeric match.
- **Working space** (`working_space`): *sRGB display* by default. A Rec.709 / sRGB source stays exactly as it is.
  A log, linear or wide-gamut source goes through the config's tone-mapped view (ACES 2.0 SDR in the built-in
  config, what Nuke's viewer shows), so the frames stay in 0-1 for the models. *preserve source* converts nothing.
- **A pixel check warns, never changes anything:** for example, a clip tagged BT.709 that looks like camera log
  (lifted blacks, nothing near white) gets a note naming the log space to pick.
- **Save Video reverses exactly what Load did.** With `colorspace_in` / `colorspace_out` on *auto (from Load)* (the
  default for new nodes), Save follows its images back to the Load Video (no wire needed). It applies the exact
  inverse of the recorded transform and writes the file in the source's own space; an EXR also gets its
  `colorInteropID`.
  - Measured live on frontend 1.53.10 (both renderers, also inside a subgraph): an ACEScg plate went Load -> Save
    and back to ACEScg with p95 error 0.00003.
  - Colours outside Rec.709 and highlights above SDR white do not come back exactly through a tone-mapped view.
- **Saved workflows keep their pixels.** A Load node saved before this opens with colour management *off (as
  before)*, and Save's explicit choices convert exactly as before.
  - The fix that makes this true also repairs the older "new inputs shifted" case on frontend 1.53.10. That
    frontend restores widget values before extensions can correct them, so the corrected values are now applied by
    name.
- **OpenColorIO 2.6** is now in `requirements.txt` (owner: "we need latest OCIO"). Its built-in ACES 1.3 / 2.x studio
  and cg configs mean nothing is downloaded. 2.5 keeps working, and the node says once that 2.6 is the target.
  Checked: every colour space, interop ID and call the nodes use is the same on 2.6.0 (ACES 2.1 studio config
  v5.0.0) as on 2.5.2, and the Load -> Save round trip matches exactly.
- **Settings** (C2C > Video > Colour Management):
  - an OCIO config file, used when `$OCIO` is not set;
  - the built-in config (studio or cg, latest or ACES 1.3);
  - the display view (empty = the config's default, e.g. ACES 2.0 SDR; or 'Un-tone-mapped').
  A change takes effect on the next load; the colour-space lists follow after a page reload.
- **Before a run** Load Video shows what will happen, for example "Colour: ACEScg (EXR colorInteropID, high) ->
  sRGB - Display (view ACES 2.0 - SDR 100 nits) · config built-in (studio-config-v4.0.0)". Save Video shows what
  it will reverse.
- **Config:** `$OCIO` (your own config), else the setting's file, else the built-in config. A broken `$OCIO` falls back to the
  built-in config and the nodes say so. `C2C_VIDEO_CM=off` switches colour management off.
- **Cost (measured):** Rec.709 / sRGB sources cost nothing. The ACES 2.0 view costs about 54-72 ms per megapixel on
  the CPU (OCIO 2.5.2 and 2.6.0 alike, 16 threads). That is several times the decode of a light H.264, so a 24-frame
  1080p LogC3 clip loads in 4.3 s instead of 0.7 s. Every transform is OCIO's exact processor; a faster baked LUT was
  tried and rejected (up to 18 % off on saturated colours).
- **GPU (2026-10-26):** the same OCIO transform can run on the GPU, using OCIO's own GPU shader and tables; nothing is
  approximated. Settings: C2C > Video > Colour Management > Processing.
  - *Auto* (default) uses the GPU for a transform only if all three hold on your machine:
    - it matched the exact CPU within a strict tolerance over about 1.9 million test colours;
    - it also matched on the clip's first frame;
    - it was measured faster on that frame.
  - *Exact CPU* never uses the GPU.
  - *OCIO GPU* skips the speed check.
  - Pixels outside the tested range are computed on the CPU: very bright, saturated colours, display peak white, NaN.
  - Anything that fails falls back to the CPU: no `moderngl`, no OpenGL 4.0, or an error mid-clip. Load Video's
    `info` colour record says which processor ran and why; Save Video's status line says *OCIO GPU* or *exact CPU*.
  - Measured on an RTX 4060 Laptop GPU, LogC3 to the ACES 2.0 sRGB view:
    - 1080p: 30 ms per frame vs 100 ms on 16 CPU threads (3.3x);
    - 4K: 109 vs 366 ms (3.35x);
    - largest difference from the CPU: 8e-6.
  - Results are close to the CPU, not bit-identical.
  - `C2C_VIDEO_CM_GPU=off` switches the GPU off.
- **Previews show true colours (2026-10-26):** Load Video's preview, Save Video's A/B source and the video comparer's
  preview are browser H.264 files.
  - Before, they used PyAV's default BT.601 conversion and were untagged. Players show that as BT.709, so pure green
    was 40/255 off, and the A/B difference view showed colour that was not in the saved file.
  - Now they convert with BT.709 in limited range and the files say so: within 2 steps (measured).
  - Evidence and design: `docs/LOAD_SAVE_VIDEO_CM_DESIGN.md` section 10.

#### Load Video (C2C): extra handles in model steps (2026-10-26)

- `extra handles (model steps)` adds k whole steps of the model's frame grid on top of fit / add: MiniMax H3 = 17,
  Wan = 4, LTX = 8, no grid = 1. One step keeps a clip that is on the grid on the grid, so the count never breaks the
  model. The widget's +/- moves one step.
- They go on `handle side`; *both* splits the steps, with the odd one at the head. The details line says what a run
  will add before it runs, for example "+ 2 x 17 = 34 extra frames -> 56". Save Video (C2C) trims them with the rest.
- It is shown only while handles are fit or add. Default 0, so saved workflows are unchanged.
- Checked live on frontend 1.53.10 in both renderers: 10 frames, fit on MiniMax H3 + 2 steps = 56 frames. Save
  trimmed back exactly to the 10 source frames, numbered 1001-1010.

#### Load Video (C2C): built-in handles (2026-10-23)

`handles` = *fit* or *add* puts handle frames on the clip as it loads (repeated edge frames, never black; the audio
moves with them). This works for IMAGE, mask and the VAE (LATENT) path alike, without decoding the clip first.
*fit* pads the clip up to the model's frame grid; `target frames` 0 means this clip's length. *add* adds
`handle frames`, rounded onto the grid. The details line says what a run will add **before** it runs, for example
"adds 9 head + 0 tail -> 57 frames (8 asked, +1 to land on Wan 2.x (4n+1))". It uses the same planner as the run. `grid` = *from format* uses the format preset above it (MiniMax H3 17n+5,
Wan 4n+1, LTXV 8n+1, ...), so there is one grid, not two. The count travels in `video_info`, and Save Video (C2C)
trims exactly that. The counts come from AV Handles' own planner, and AV Handles stays available as a separate
node.

#### Video Comparer (C2C): player, metrics, A/B/C/D (2026-10-23)

- In `synced_player` mode (the default), Load Video's transport drives A. B follows it frame by frame, shifted by
  `b_offset` frames.
- **Views:** A, B, wipe (vertical / horizontal / diagonal), difference with gain, onion, flicker, side by side, and
  a 2 x 2 grid of A / B / C / D (`image_c` / `image_d`, `file_c` / `file_d`, with labels). Zoom, pan, 1:1 and a
  loupe with pixel values.
- **Metrics:** after a run, per-frame PSNR, SSIM (on luma) and colour difference (CIEDE2000, checked against Sharma
  et al. 2005) appear as a strip under the scrubber. Hover to read a frame's values, click to go to it; the worst
  frame of each is marked. `metrics` turns it off.
- Previews of image batches keep every frame (the old preview stopped at 600).
- **Mistyped wires are repaired** (2026-10-25). A saved workflow can hold a wire whose type its socket cannot take.
  The owner's had Load Video's IMAGE on `bit_depth`, and the server refused the whole prompt with "Return type
  mismatch ... bit_depth". When a workflow loads, every such wire into a Code2Collapse node moves to the node's first
  free socket that takes it, or is removed when none does. A toast says what moved where. Other packs' nodes are not
  touched. Setting: *C2C › Graph › Repair mistyped wires* (on by default). Checked on frontends 1.53.10 and 1.57.0, in
  both renderers.

#### Batch Range (C2C) — single frame

Extract a single frame from a video batch with **Batch Range (C2C)** (`ImageBatchSliceMEC`, C2C/Helpers).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `first frame` | `first frame` / `last frame` / `middle frame` / `frame at index` |
| `frame_index` | `0` | Frame index for `frame at index` mode (clamped to batch size) |

**Outputs:** `images` (IMAGE), `total_frames` (INT), `is_video` (BOOLEAN)

---

### BBox Tools

Six nodes for bounding box manipulation:

| Node | Description |
|------|-------------|
| **BBox Create** | Manual `[x, y, width, height]` entry |
| **BBox From Mask** | Extract tight bbox from non-zero mask pixels with per-axis padding |
| **BBox To Mask** | Convert bbox to a rectangular mask |
| **BBox Pad** | Asymmetric padding (top/bottom/left/right) with canvas clamping |
| **BBox Crop** | Crop image + mask to bbox region |
| **BBox Smooth Temporal** | Smooth bbox sequences across video frames (moving-average / exponential) |

All BBox nodes clamp outputs to valid canvas bounds — no negative dimensions, no out-of-bounds errors.

---

#### Mask Failure Explainer (MEC)

Diagnose why a mask failed and get actionable fix suggestions. Pure tensor analysis — no models, no VRAM.

**How it works:** Runs a 5-metric analysis pipeline on your image + mask pair:

| Metric | What it measures | Failure threshold |
|--------|-----------------|-------------------|
| **Brightness** | BT.709 mean luminance | Dark scene < 0.15 |
| **Blur** | Laplacian variance × 1000 | Blurry image < 50 |
| **Boundary contrast** | Std deviation at mask edge ring | Low contrast < 0.05 |
| **Color confusion** | Mean color distance inside vs. outside mask | High confusion < 0.1 |
| **Background complexity** | Sobel edge density outside mask | Busy background > 0.3 |

Each metric contributes up to 20 points to a severity score (0–100). The node outputs a human-readable explanation, a problem-regions heatmap, the severity score, and a suggested method (e.g. "try ViTMatte for complex edges" or "use BiRefNet for busy backgrounds").

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | — | Input image(s) |
| `mask` | — | Mask to diagnose |
| `ring_width` | `5` | Boundary ring width for contrast analysis |
| `blur_threshold` | `50.0` | Laplacian variance threshold |
| `brightness_threshold` | `0.15` | Dark scene threshold |

**Outputs:** `explanation` (text), `problem_regions_mask` (heatmap), `severity_score` (0–100), `suggested_method` (string)

> **Who is it for?** Anyone whose mask looks wrong and doesn't know why. Plug in your image + bad mask, read the diagnosis, follow the suggestion. Especially useful for beginners learning which segmentation method works for which scenario.

---

#### Image Mask Editor (C2C)

Paints a mask over the image that actually reaches the node, at native resolution (`C2C/Masking`).
Click **✏️ Edit mask** to open the full-window editor.

**Modes** (L7.65 P01; each shows only its own controls and drawing UI, and runs the code of the node it took over):
`paint` (this editor, plus a colour layer: the former Paint Canvas, output `painted_image`), `points + box` (clicks
and boxes for SAM / SeC, the former Mask Edit points editor), `shape` (12 parametric shapes or JSON), `spline` (a
spline shape, or a procedural flow pattern along it), `roto` (Bezier shapes keyed over frames, the former Vector
Roto) and `video` (painted keyframes, tweened: the former Video Mask Editor). Details:
[docs/image-mask-editor.md](docs/image-mask-editor.md#modes). The extra outputs (points, boxes, spline, painted
image) come after `mask` / `preview` / `info`, so saved links keep their slots.

- **What you edit:**
  - a directly wired LoadImage or C2C/VHS loader shows its frames without a run;
  - any other source (a crop, an upscale, a generated image) shows the exact frames the node received in its last run;
  - before that first run you get the nearest upstream preview, with a banner saying so.
- **Tools:** brush and eraser (size, hardness, opacity), rectangle, ellipse, polygon, lasso and bucket fill (tolerance). Alt + stroke subtracts. Overlay / matte / image views.
- **Batches:** a frame strip, one mask per frame (or one shared mask with `frame_mode = shared`), and "Copy previous frame".
- **Non-destructive result stage:** `grow`, `feather`, `threshold`, `invert` and `combine` (with an optional `input_mask`) are node parameters, so changing a feather never means repainting.
- **Storage:** masks are PNGs in `input/c2c_masks/<editor_id>/` and survive a server restart. The workflow keeps only the editor id. The masks' content hash is part of the cache key, so an edit re-runs the graph and an unchanged graph stays cached.

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| **B / E** | Brush / eraser | **[ / ]** | Brush size |
| **R / O** | Rectangle / ellipse | **F / 1** | Fit / 100% |
| **P / L** | Polygon / lasso | **V** | Cycle view |
| **G** | Bucket fill | **, / .** | Previous / next frame |
| **Ctrl+Z / Ctrl+Y** | Undo / redo | **Enter / Esc** | Save / cancel |
| **Space or middle drag** | Pan | **?** | Shortcut list |

**Outputs:** `mask` (MASK), `preview` (IMAGE, mask tinted over the input) and `info` (STRING: frames, stored masks, notes).

#### Mask Track: spline keyframes and box smoothing

**Mask Track** has two more modes (L7.65 P01; the controls of each mode are prefixed and shown only in that mode):

| Mode | What it does | Controls | Outputs it fills |
|---|---|---|---|
| `spline_track` | Follows spline keyframes (`[{frame, points}]`) through the `video` with Lucas-Kanade, the same tracker as Spline Mask's `track` mode | `spline_data`, `spline_closed`, `spline_samples_per_segment`, `spline_tracking_weight`, `spline_klt_window`, `spline_feather_radius`, `spline_stroke_width` | `masks`, `info_json`, `spline_data_out` |
| `bbox_smooth` | Smooths a list of boxes over time (median, moving average, exponential or median then exponential), the same code as Mask Edit's `bbox_smooth` mode | `bbox_json`, `bbox_radius`, `bbox_method`, `bbox_alpha` | `bboxes`, `primary_bbox`, `info_json` (the smoothed boxes as JSON) |

The three new outputs (`bboxes`, `primary_bbox`, `spline_data_out`) come after the old ones, so saved links keep their
slots. Mask Tools gained `transform` the same way: grow, blur and offset per axis, feather, threshold and invert (Mask
Edit's `transform`, the same Mask Transform XY code).

#### Video Frame Player (MEC)

In-graph video scrubber with drag-crop overlay, trim range, frame stride,
lanczos resize, and upscale — all in one node so you don't have to chain
`Load → Trim → Crop → Resize → Preview`. Plays inside the node, supports
play/pause + ping-pong loop + I/O hotkeys (mark IN/OUT), and emits the
trimmed/cropped/resized batch ready to feed a sampler.

| Group | Highlights |
|-------|------------|
| **Source** | `frames`, `frame_index`, `output_mode` (current_frame / all_frames) |
| **Trim & playback** | `frame_start`/`frame_end` (drag green/red markers on the timeline), `frame_stride` (orange ticks), `playback_fps`, `loop_mode` (once/loop/ping-pong) |
| **Crop** | `crop_enabled`, `crop_locked`, `aspect_ratio` (free / 1:1 / 4:3 / 3:4 / 16:9 / 9:16 / 2:1 / 21:9 / original / custom), `crop_x/y/w/h` set by drag overlay (8 handles + interior move, dim outside, rule-of-thirds guides, live W×H readout) |
| **Resize** | `resize_method` (none / lanczos / bicubic / bilinear / area / nearest-exact), `target_width`, `target_height`, `upscale_factor` |
| **Outputs** | `frame`, `frame_index`, `frame_count`, `processed`, `out_width`, `out_height`, `crop_x_px`, `crop_y_px`, `crop_w_px`, `crop_h_px`, `trimmed_count`, `playback_fps` |

**Hotkeys** (canvas focused): `Space` play/pause, `←/→` step ±1 (Shift = ±10), `Home/End` jump to trim bounds, `I/O` mark IN/OUT at current frame, `R` reset crop to full frame.

Full parameter table, pipeline order, and recipes (TikTok 9:16, half-rate stride, 2× lanczos upscale, hero-frame img2img): see [docs/video-frame-player.md](docs/video-frame-player.md).

---

#### Mask Preview Overlay (MEC)

Visualize masks with 5 display modes:

| Mode | Description |
|------|-------------|
| `overlay` | Red-tinted mask overlaid on the image |
| `binary_mask` | Black & white mask only |
| `edge_only` | Show mask edges / contours |
| `side_by_side` | Original image next to masked version |
| `alpha_channel` | RGBA with alpha from mask |

Handles batch size mismatches automatically (expands or repeats to match).

---

#### Parameter History (MEC)

Tracks every parameter change across ComfyUI runs in a local SQLite database. Query with:

| Mode | Description |
|------|-------------|
| `all_history` | Full parameter history for last N runs |
| `last_run_diff` | What changed between the last two runs |
| `node_class_filter` | Filter history by node class name |

The in-editor side (right-click a node > Parameter History; Alt + hover a widget) records every change in both
renderers without taking over the widget's value, so a saved workflow shows its own values in Nodes 2.0.

Secrets are never recorded: widgets named like a key, password, passphrase, credential or token, and any value that
looks like a provider key (sk-, hf_, gh*_, xox*-, AIza...), are left out by the editor and again by the server. A
database written by an older version is cleaned once on the next start.

#### Nano Banana (C2C): where the Gemini key lives

The node reads `GEMINI_API_KEY` from the C2C key store (Settings > C2C AI, kept in the operating system's credential
store) or from the `GEMINI_API_KEY` / `GOOGLE_API_KEY` environment variables. Leave the `api_key` box empty. A key
pasted into it, or found in a workflow saved by an older version, is moved into the key store and the box is
cleared, so the key is not saved in the workflow or in the image metadata (save the workflow again to remove it from
an old file). If the key store is unavailable the key stays in the box and a warning says it will be saved.

Every run is billed by Google. When ComfyUI runs the node again with exactly the same request (same model, prompt,
settings, reference images, seed and batch count), the images from the earlier run this session are returned and
`info` says `reused` - nothing is paid twice. `C2C_NANO_BANANA_DEDUPE=0` turns this off (2026-11-16).

#### Fixed 2026-11-05

- **Nano Banana (C2C)**: when one call of a batch fails, the images already generated (and billed) are returned,
  and `info` says which call failed (`partial`). It used to return nothing. A failure on the first call still stops.
- **Folder Version Control / Folder Incrementer Set**: `set` with a value above 999 creates only that version's
  folder (a typo could create up to 999,999 folders). Smaller values create every folder, as before.
- **C2C Vault**: a damaged or truncated vault shows "the vault payload is damaged or incomplete" instead of a raw
  decoding error.
- **C2C Farm Submit**: the same workflow on the same backend, profile and project reuses the job that is already
  running or finished, instead of uploading and paying for it again (ComfyUI can drop the node's cache between
  runs). The new `force_resubmit` input (off by default) always sends a new job.
- **Prompt Relay (kijai backend)**: the masks are patched into the loaded Kijai model, because his sampler offers no
  per-run hook (checked against WanVideoWrapper nodes_sampler.py). Deleting the node therefore does not remove
  them; run **Prompt Relay Restore (Kijai)** on the model or reload it. The node now says so.
- **Folder Incrementer**: in an API run with no source widget filled, re-queuing now moves to the next version
  when the shot's folder (found from the workflow) got a new one; it could keep returning the old version.

#### AI keys can be managed only from this machine (2026-11-05)

Setting, deleting, importing or listing the C2C AI keys (`/c2c/ai/keys/...`) now answers only the ComfyUI page
opened on this machine (127.0.0.1 or localhost), the same rule as the Magnific account routes. Before, on a server
started with `--listen`, any machine on the network could overwrite or delete your keys, and the import route
read any file path. Through an SSH tunnel the page still counts as local.

#### Plain errors instead of silent fallbacks (2026-11-04)

- **VAE Merge (MEC)** stops with the reason when a merge cannot be built (unreadable input, weights that do not fit,
  not enough memory to copy the VAE). It used to return the unmerged first VAE as if it had worked, and in the
  out-of-memory case it changed that input VAE in place.
- **Control AOV (C2C)** stops when a pass you selected (depth, pose, normal, SAM matte) produces nothing, and names
  the backend's error. It used to hand ControlNet an all-black map. Passes left on `off` still output black.
- **Mask + Matting `auto` / `auto_best` pick only backends that can run here** (2026-11-07). On a clip they used to pick
  SeC, which is never installed, and then silently fell back to the approximate GrabCut path even with SAM 3 available.
- **Node explanations never call a paid service on hover** (2026-11-07). The explainer's `auto` setting uses a local
  GGUF model if one is installed, else the built-in text; your cloud key is used only when you pick `api`.
- **Image Mask Editor**: pressing Enter on a focused button (for example Cancel) no longer saves and overwrites the mask;
  Ctrl+Shift+Z redo and Caps Lock undo work in every C2C editor (2026-11-07).
- **Workflow Doctor (panel)** shows its findings again - it showed "All clear" for every graph - with working Apply-fix
  buttons that can be undone; the workflow wizard no longer keeps the canvas repainting after you close it (2026-11-07).
- **Control AOV (C2C)** has a new last input, `allow_download` (off by default; saved workflows keep their values).
  Without it no backend downloads anything: an installed copy (a models folder or the Hugging Face cache) is used, or
  the info output says what is missing. Before, Depth Anything / DepthCrafter / NormalCrafter / DVD fetched weights
  (several GB, DVD's are non-commercial) on first use without asking. Depth-Pro now runs through transformers' own
  model when Apple's package is not installed, and is returned as inverse depth (near = bright) like the other depth
  backends; Depth Anything 3 and DVD say how to install them instead of pointing at a `third_party` folder that a
  normal install never has.
- **Tile Plan / Tile Split / Tile Merge / Identity Lock** cache again: an unchanged re-queue no longer re-runs them
  (and the model between them) or logs an IS_CHANGED warning every time.
- **SAM Model Loader (MEC)** refuses a checkpoint that does not fit the SAM 2.1 architecture (a SAM 3 file used to
  load into it with random weights and give garbage masks), and says when the `sam2` package is missing. The
  model-free fallback (labelled approximate) still applies unless `C2C_MODEL_FALLBACK=off`.
- **SAM Mask Generator (MEC)** stops with the reason when SAM cannot run (no fitting predictor, a prediction error)
  or when GroundingDINO fails on a text prompt; it used to return an empty mask or drop the text silently. With
  the model-free stand-in the text is still only noted (that stand-in cannot read text).
- **Face Fixer (MEC)** with denoise 0 leaves the faces as they are (it used to regenerate them completely).
- **ProPainter (MEC)**: colour match "off" really is off in temporal, remove and stitch modes (they ran Reinhard
  on the fill; remove / stitch fixed 2026-11-16).
- **Mask Refiner (MEC)**: `preset` and `advanced_overrides_json` take effect with auto edge lock on (the subject
  profile used to overwrite them); `balanced` gives exactly the same result as before (2026-11-16).
- **SAM Mask Generator (MEC)**: the two GroundingDINO threshold tooltips were swapped; `mask_index` is the model's
  own candidate order, not "best score" (2026-11-16).
- **Mask + Matting (MEC)**: a `holdout_mask` / `core_mask` of another size is resized to the matte and named in
  `info` (it was ignored silently) (2026-11-16).
- **VAE Decode (C2C)**: a workflow saved before `tile_mode` existed keeps its `tile_size` (the new `auto` replaced
  it); under `auto` a `tile_size` above 0 is used as given (2026-11-16).
- **AsymFlow Sampler Patch**: refuses a model that is not a flow model (it gave wrong output silently on EPS /
  v-prediction models) (2026-11-16).
- **Fluid Shot Encoder**: stops with a message when OpenCV is missing (it returned the clip unchanged) and when a
  clip would grow past 4x its frames (+16) (2026-11-16).
- **Folder Version Control**: `set` reports the version the next run will really make when higher versions already
  exist (2026-11-16).
- **Face Fixer (MEC)**: when no face was fixed (no detector and no mask, or nothing found), `info` says why
  (2026-11-16).
- **VAE Merge (MEC)**: `brightness` is a real offset (added to the decoder output layer's bias; it was also added to
  its weights, which made it depend on the picture). Results with brightness 0 are unchanged (2026-11-16).
- **Control AOV (C2C)**: a requested edge or motion pass that produced nothing (e.g. OpenCV missing) stops the run
  with the reason, like the other passes (2026-11-16).
- **Mask Tools (C2C)** `key: difference`: a clean plate with fewer frames than the clip holds its last frame (it
  crashed); the report says so (2026-11-16).
- **SAM Mask Generator (MEC)**: a text prompt that GroundingDINO finds nothing for, with no point / box / mask, stops
  with the words and thresholds to change (it ran on with no prompt) (2026-11-16).
- **Video Comparer (C2C)**: the audio spectrogram / loudness modes name the missing package (librosa / pyloudnorm)
  instead of showing a black picture (2026-11-16).
- **LocateAnything Grounding**: can be stopped between frames and shows progress (2026-11-16).
- **Mask + Matting, SAM Model Loader (auto), SAM Multi-Mask Picker, Semantic Segment**: run on ComfyUI's device
  (`--cpu`, the chosen GPU) instead of any GPU torch can see (2026-11-16).
- **Mask Track (MEC)**: new `allow_download` switch (last input, off by default, shown only for stabilize +
  raft_flow). Without it, raft_flow no longer downloads torchvision's RAFT-small weights; when they are missing the
  mask passes through and the `warning` output says why (2026-11-17).
- **SAM Mask Generator (MEC)**: with several boxes in `bboxes_json`, `info` is JSON and names the points / box / crop
  settings that are not used (each box is segmented on its own) (2026-11-17).
- **Image Mask Editor (C2C)**: recording the input frames for the editor is faster: an unchanged re-run skips the
  full content hash (measured 2.0 s -> 0.007 s for 120 frames of 1344x768) and frames are encoded in parallel (13.8 s
  -> 5.3 s first time) (2026-11-17).
- **Face Landmarks (C2C)**: only the controls that matter are shown, in collapsible sections (Tracking, Confidence,
  Output, Model setup and the appearance groups), and the node resizes to fit in both renderers and inside subgraphs
  (default node 1434 px / 55 controls -> 272 px / 3). Saved workflows keep every value (2026-11-17).
- **PAR Desqueeze / Resqueeze** keep scene-linear EXR values above 1.0 and below 0 (they were clamped to 0-1).
- **Video Stabilizer (MEC)**: the padding mask has the size of the stabilised frames when the framing changes it.
- **LocateAnything to SAM (MEC)**: on a batch, each box marks only the frame it was found in (each frame used to get
  the union of every frame's boxes).
- **Save Video (C2C)** Folder Version naming: on a server reachable from other machines (`--listen`) the subfolder
  must stay inside the C2C media roots (a `../` or absolute subfolder used to write anywhere). Local servers are
  unchanged.
- **ProPainter (MEC)** stitch and stitch_refine now take the stitcher output of **Inpaint Crop Pro**. That input
  was typed `STITCH_DATA`, which no node produces, so these modes could never be wired. Per-frame crop positions
  are kept. A clip whose crop size changes from frame to frame is refused with a message.
- **Face Fixer (MEC)**: when face detection cannot run (ultralytics missing, model not found, a model error), it
  still falls back to the mask's box, and `info_json` now says why (`detection`). Each frame also records where its
  faces came from (`source`: detector / mask / none). Before, the image could come back untouched with no word.
- **Locate Anything Grounding (MEC)**: switching `device` loads the model on that device (the cached copy stayed
  on the old one), only one copy is kept, and ComfyUI's memory release frees it. `confidence_threshold` has no
  effect (the model returns no scores); its tooltip now says so.
- **Mask Refine (MEC)**: broken `advanced_overrides_json` stops with a message (it was ignored silently); unknown
  keys still run and are listed in `info` (`overrides_ignored`).
- **SAM Multi-Mask Picker (MEC)** stops with the reason when SAM cannot run (no predictor, a prediction error,
  out of memory on the CPU retry too); it returned an empty mask. Its mask thumbnails now refresh whenever the
  masks change.
- **SAM Mask Generator (MEC)**: `existing_mask` is used from the first pass (at the default one refinement pass it
  did nothing), and `score_threshold` really discards: when no mask reaches it the mask is empty and the info says
  the best score.
- SAM nodes and Semantic Segment no longer wrap HDR values above 1.0 when they convert the frame for the model.
- **Mask Placement (MEC)**: `feather_px` applies when the placement has no feather of its own (a fixed 6 px was
  used), and a missing opencv is reported in words.
- **Mask Tools (C2C)** edge_spread with no mask wired outputs an empty mask instead of nothing.
- **Mask Tools (C2C)** `grow / shrink` extends a matte in any direction: `grow_angle` turns the up / down / left /
  right axes, `grow_units` takes pixels or % of the matte's own size, `grow_falloff` gives the grown band a soft edge,
  and with Face Landmarks' `pose_keypoint` wired into `face_pose` the axes follow each frame's head tilt
  (`grow_follow_face`). At the defaults the result is exactly the old grow (2026-11-17).
- **Image Mask Editor (C2C)** previews round to the nearest code value instead of truncating (they were half a level
  too dark) (2026-11-17).
- **More switches in Settings > C2C:** Diagnostics > Per-node time and memory, Masks > Matte viewer, Models &
  Downloads > Model-free fallbacks, Video > Save > Encode with FFmpeg, Vault > Unlock lasts (minutes), Masks > SAM 3.1 refine pass, Video > Save > Write a
  source proxy and Previews > Do not force sampler previews on. Each was
  only an environment variable before; the variables still work and take priority (2026-11-18).
- **C2C Control Center:** open it from the C2C OmniPill (Tools) or the command palette ("C2C: Open Control
  Center"). It shows the ComfyUI, frontend, PyTorch and CUDA versions, free RAM and VRAM, every installed C2C pack
  with its version and node count (and any node ids another pack took over), problems found at start-up, and
  settings fixed by environment variables. "Copy diagnostics report" copies all of it for a bug report, with no
  keys or tokens in it. It answers only on the ComfyUI machine itself (2026-11-18).
- **New settings:** Settings > C2C > Models & Downloads > **Never download models** (no C2C node downloads weights,
  even with its own Allow Download ticked) and Video > Upload > **Largest upload (MB)**. The environment variables
  `C2C_NO_DOWNLOADS` / `C2C_UPLOAD_MAX_MB` still work and take priority (2026-11-18).
- **Folder Version Incrementer (old Set node):** asking it to reserve more than 999 versions makes only the highest
  folder (since L9.135); its status now says so instead of claiming the whole range (2026-11-18).
- **Sharper previews:** Video Comparer (wipe / onion / side by side / difference views) and Video Frame Player now
  draw at your screen's pixel density and at the canvas zoom, so they stay sharp on high-DPI screens and when you
  zoom in; nothing else about them changes (2026-11-18).
- **Security:** the model browser sent your Civitai / Hugging Face token to any address that merely *contained*
  "civitai" or "huggingface" (for example `civitai.example.net`); now only to civitai.com, huggingface.co and their
  own subdomains (or the base you configured). Keys are also removed from C2C log lines, error records and AI
  payloads, including the newer key shapes (Groq, xAI, Perplexity, Fireworks, Replicate, OpenAI project keys)
  (2026-11-18).
- **Saved workflows are leaner:** display-only panels (previews, status strips, editors) no longer write empty values
  into the workflow file. Old files still load with every setting intact (2026-11-18).
- **Face Landmarks:** with tracking on, "largest face" stays on the same person while they are in view instead of
  jumping to whoever is momentarily bigger; it chooses again after a scene cut (2026-11-18).
- **Fluid Shot Decoder:** a new `info` output says when the processed clip had a different frame count and the
  timing is only approximate; turn off `allow_approximate` to stop with an error instead (2026-11-18).
- **Magnific (sign-in nodes):** a slow or offline service no longer delays ComfyUI loading its node list: the last
  lists are shown while they refresh in the background (2026-11-18).
- **C2C Bypass Manager** (new tool, not a node): right-click a node -> "Every <type>" -> bypass / mute / set to normal
  every node of that type, including inside subgraphs; the canvas menu does it by name pattern and restores
  everything; Ctrl+Alt+Shift+B / M / N do it for the selected node's type; the "Bypass" chip in the C2C bar opens a
  list of every node type with a one-click normal / bypassed / muted button and named sets saved with the workflow.
  One undo step per action; pinned nodes are never changed. Setting: C2C > Canvas > Bypass Manager (2026-11-18).
- **Security:** the C2C AI configuration, policy, spend cap and backend test, the stored secrets (`/c2c/secrets/*`,
  incl. the Vault's keys) and the Diagnostics API-key routes answer only the ComfyUI page on this machine (an SSH
  tunnel counts as local). On a `--listen` server a network client could otherwise point an AI backend at its own
  host and receive your stored key (2026-11-18).
- **Previews come back after a workflow-tab switch** (Video Frame Player, Video Comparer, Save Video, Frequency,
  Image Mask Editor): they used to come back empty; now the last result shows again without re-running, and core's
  duplicate image under these nodes is hidden. Save Video's change check no longer copies the whole clip on every
  queue (measured 120 x 1080p: +2.8 GB / 4 s before, +63 MB / 0.4 s now) (2026-11-18).
- **All CNP nodes** show only the controls that matter for the current settings, in collapsible sections, and resize
  to fit in both renderers and inside subgraphs; saved workflows keep every value (2026-11-18).
- **Inpaint Crop Pro** `mask prepare` without an image stops with a message when `stitcher` or `cropped_image` is
  wired (the next node crashed); the mask-only use is unchanged.
- **Mask Failure Explainer (MEC)**: `ring_width`, `blur_threshold` and `brightness_threshold` now change the
  analysis (they were ignored).
- **Video Frame Player (C2C)** shows the right frames when two clips of the same size share black borders (it
  could show the previous clip).

---

### FolderIncrementer

Filesystem-aware auto-versioning for output files.

Fixed 2026-11-04: a shot whose name ends in a number lost that number when it came from an image sequence.
`SC_67_SH_2130 (2)/SC_67_SH_2130.####.png` was saved as `SC_67_SH/<date>/v003/SC_67_SH.mov`, because the frame token
was removed twice (`.####`, then `_2130` as if it were a frame number). Exactly one frame token is removed now:
`SC_67_SH_2130/<date>/v001/SC_67_SH_2130.mov`. A numbered frame file (`SC_67_SH_2130.1001.png`) still loses only its
frame number. This applies to Save Video (C2C)'s built-in incrementer, the Folder Version Incrementer and its
on-node preview.

| Node | Description |
|------|-------------|
| **Folder Version Incrementer** | Scans output directory for `v001`, `v002`, … and returns the next available version |
| **Folder Version Check** | Reports how many versions exist |
| **Folder Version Set** | Reserves version slots by creating placeholder directories |

**Key features:**
- **Filesystem-based** — no global counter file; scans the actual directory
- **Cancel-safe** — if you cancel mid-execution, no version is wasted
- **Extension-preserving** — `photo.png` → `photo/v001/photo.png`
- **Optional suffix** — set the new `suffix` widget (e.g. `_Inpaint`) and the
  basename of `filename_prefix` / `output_filename` becomes `photo_Inpaint`
  (folder + `version_string` are unchanged). No more downstream `Concatenate`
  nodes just to tag a pass.
- **Names from the nearest loader upstream** — core, VHS, Pixaroma, OCIO and the NukeMax readers (Read, EXR Pass
  Reader, EXR sequence loaders) all count, so the folder is named after the file that loader reads.

**Outputs:**

| Output | Example | Purpose |
|--------|---------|---------|
| `version_string` | `v001` | Version tag |
| `version_number` | `1` | Integer version |
| `folder_name` | `photo` | Derived from source filename |
| `subfolder_path` | `photo/v001` | For Save Image subfolder |
| `filename_prefix` | `photo/v001/photo` | Without extension |
| `output_filename` | `photo/v001/photo.png` | Full output path |

---

## Workflows

### Best Quality Masking (Single Image)

```
[SAM Model Loader]          [Load Image]
  offload_to_cpu: true         ↓
  dtype: float16          [Points Mask Editor]
        ↓                    ↓ points_json, bbox_json
[Mask + Matting] ←──────────┘  (sam_model ← SAM Model Loader)
  backend: SAM + ViTMatte
  sam_iterations: 2–3
  sam_refine_method: auto
  sam_detail_preservation: 0.85
  sam_edge_contrast: 1.2
        ↓
  alpha → compositing
```

### Fast Iteration

```
[SAM Mask Generator]  →  [ViTMatte Edge Refiner]
  refine_iterations: 2       method: multi_scale_guided
```

### Pick the Best SAM Mask

```
[Load Image]  →  [Points Mask Editor]  →  [SAM Multi-Mask Picker]
                                              ↓ click thumbnail
                                         selected_mask → downstream
```

### Inpaint Pipeline

```
[Load Image] + [Mask]  →  [Inpaint Crop Pro]     →  [Any Inpaint Model]
                              context_expand: 1.5        ↓
                              stitch_blend: edge_aware   [Inpaint Stitch Pro]
                                    ↓ stitch_data ──────→     ↓
                                                         seamless result
```

### Video Masking

```
[Load Video]  →  [Batch Range (C2C)]  →  [Points Mask Editor]
                   mode: first frame              ↓
                                            [SAM Mask Generator]
                                                    ↓
                                            [Mask Propagate Video]
                                              mode: sam2_video
                                                    ↓
                                              per-frame masks
```

### Temporal Mask Morphing

```
[Define masks on frames 0, 15, 30]  →  [Temporal Anchor System]
                                          anchor_frames: "0,15,30"
                                          total_frames: 60
                                          easing: smooth_step
                                                ↓
                                          60 interpolated masks
```

### Alpha Matting (From Any Coarse Mask)

```
[Any Segmentation Node]  →  [Matting Node]
                               backend: auto
                               edge_radius: 15
                                    ↓
                              alpha_mask → compositing
```

### Diagnose a Bad Mask

```
[Load Image] + [Bad Mask]  →  [Mask Failure Explainer]
                                    ↓
                              explanation: "Dark scene (mean=0.12), low boundary contrast"
                              suggested_method: "Try ViTMatte with edge_radius=20"
                              severity: 65/100
```

### Luminance Keying

```
[Load Image]  →  [Mask Tools: key: luma]
                    luma_key_mode: auto
                    luma_key_falloff: 1.0
                    luma_key_gamma: 1.0
                       ↓
                  luminance mask (highlights / shadows / midtones)
```

### SeC + MEC (Video Object Segmentation)

[SeC](https://github.com/9nate-drake/Comfyui-SecNodes) handles tracking, MEC handles edge quality:

```
[SeC Video Segmentation]  →  [ViTMatte Edge Refiner (MEC)]  →  refined masks
```

Install SeC: `cd ComfyUI/custom_nodes && git clone https://github.com/9nate-drake/Comfyui-SecNodes`

### Spline Mask Drawing

```
[Load Image]  →  [Image Mask Editor, mode: spline]
                    spline_type: catmull_rom
                    spline_closed: true
                    spline_feather_radius: 2.0
                       ↓
                  mask → any mask input
                  coords_json → SAM Mask Generator
                  spline_data → Motion Mask Tracker
```

### Motion Detection (Video)

```
[Load Video Frames]  →  [Motion Mask Tracker]
                           camera_compensation: true
                           detection_mode: combined
                           grow_pixels: 4
                                ↓
                          motion_mask → compositing / routing
                          motion_intensity → Switch node
```

### Draw Shape (Quick Masking)

```
[Draw Shape (MEC)]
   shape: star
   cx: 256, cy: 256
   outer_r: 100, inner_r: 40
   num_points: 5
   feather: 4.0
        ↓
   mask → downstream
```

---

## Face Landmarks: look, strength, size, screening, mask and smoothing

**Face Landmarks (Facepoints, 620 + 68/98)** has 46 optional controls, in collapsible sections on the node (open a
section with its chip; both renderers). With every control at its default, the images are exactly what they were
before (tested bit for bit). The name prefix gives the group:

- **appearance_**: landmark-image dot and background colours; overlay colours per 620-point kind (`lmk`, `skin`, `lid`,
  `teeth_u`, `teeth_l`, `iris`) and for the 68 and 98 sets; one colour for the whole overlay. Colours are `#RRGGBB`
  and take effect with `appearance_scheme = custom`. They never change the pose numbers.
- **strength_**: overlay opacity (real alpha), landmark-image dot intensity, how strongly unsure points are drawn,
  the confidence below which a point counts as unsure, and a multiplier for the confidence values in pose_keypoint.
  The coordinates are never touched. `strength_connection_lines` (0 = off) draws the face contours (jaw, brows,
  nose, eyes, lips) as lines between the points, on the landmark image and the overlay; `appearance_line_width`
  sets their width. Lines follow the published 68 and 98 point orders; the 620 dense points have none, so no lines
  are drawn for that set and the JSON says so.
- **resolution_**:
  - processing size: find and crop the face on a smaller frame; points are still reported in source pixels;
  - output size: presets or custom, aspect kept or stretched. Images, mask, pose, JSON and boxes all follow, so
    everything stays aligned.
- **screening_**: minimum face confidence (separate from the face finder's own `finder_confidence`), minimum point
  confidence (pose c = 0, x / y kept), how unsure points are drawn (dim, fade, hide or keep), and rejected faces in
  the JSON for diagnosis. The JSON always keeps every point.
- **mask_**: which faces fill the mask (`mask_faces`: all faces, the largest face, or one face by its position in
  pose_keypoint with `mask_face_index`), grow, shrink, feather, edge softness, threshold, hull smoothing, and a
  preview on the overlay. The MASK stays 0..1.
- **temporal_**: landmark smoothing for video. It adapts: still points stay steady, real motion is followed at once,
  and points the model is unsure of move less. It also offers a lag limit, a scene-cut restart, a restart after a
  lost face, and a stable face order by identity. Raw points stay in the JSON. This is separate from `track_video`.
  With smoothing or stable order on, a scene cut also drops the previous shot's tracking box, so the new shot's
  faces are searched for afresh. A `mask_face_index` past the number of faces leaves the mask empty and says so in
  the JSON.

Measured with the real model on CPU: mean landmark jitter 1.47 → 0.29 px on a shaky clip (strength 0.8); output
resize aligned within 0.07 px. Evidence: `docs/evidence/A27/face/` (workspace).

**620-point colours (GNM classes).** The model's 620 points come in six classes, read by name from the model's
`config.json`. Default overlay colours (visualisation only; editable with `appearance_scheme = custom`):

| class | colour | points |
|---|---|---|
| lmk | `#1E88E5` blue | 416 |
| skin | `#FF8C00` orange | 150 |
| lid | `#00C850` green | 36 |
| teeth_u | `#E63296` magenta | 7 |
| teeth_l | `#FF4DA6` pink | 9 |
| iris | `#FFD700` yellow | 2 |

Colours are written as RGB `#RRGGBB`; the node converts to OpenCV's BGR once for drawing and back to RGB once for
the IMAGE outputs. A class the node does not know is drawn grey and named in the JSON notes.

**When an output is black.** `landmark_image` is black wherever no face was accepted. `landmarks_json` now says why:
each frame has `diagnostics` (face boxes found, faces inferred / accepted / rejected with the reason, points drawn
per class), the summary has each output's size, value range and lit pixels, and a note names the frames without a
face ("the face finder found no face" - try `face_finder = whole image` for a face crop or a lower
`finder_confidence` - or "every face was rejected: ..."). On a large frame with a small face the landmark image is
mostly black even when correct; `lit_fraction` shows it. Evidence: `docs/evidence/FACE_GNM/` (workspace).

**Which part of the head the mask covers (`mask_part`, Mask section).** `whole face (default)` is the mask as
before. Features: eyes, one eye (image left / right), eyelids, brows, nose, lips, inner mouth / teeth. Areas: around
the mouth, cheeks, chin + jaw. Estimated (no landmark covers them; they follow the head's tilt and, with
`mask_head_refine = image edges`, are fitted to the picture): forehead, ears, hair, head + hair. `custom list` joins
several, e.g. `eyes, lips` in `mask_parts_custom`. `mask_invert` gives everything except the chosen parts.
`mask_extend_up / down / left / right` (% of the face height, negative shrinks) extend the mask along the head's own
axes (`mask_extend_follow_tilt`), with an optional soft fall-off. All defaults give the old mask exactly.

**Low light, colour casts, motion blur (`screening_robust_detection`).** `auto` (default): a frame where no face is
found is analysed (brightness, colour cast, sharpness) and retried on enhanced copies (light lift with CLAHE, light
lift after denoise, colour balance, luma only, deblur, a 1.5x look), then with the face box of a nearby frame. The
copies are only shown to the detector; every output keeps the original footage. Frames that already work are not
touched. Measured on the real model: a very dark frame (x0.06), a dark noisy frame, a heavily blurred frame and a
dark motion-blurred frame went from no face to a face; nothing that was found was lost. `always` detects every frame
on the enhanced copy; `off` is the old behaviour. Each frame's `diagnostics.robust` says what was tried and what
worked.

## C2C Vault: lock or seal part of a workflow

A vault is a **protected subgraph**. Outside it is one node with the subgraph's inputs, outputs and promoted
controls. Its inside travels in the workflow as AES-256-GCM ciphertext (format v2; the password is stretched with
Argon2id). ComfyUI's own scheduler runs the inside, exactly like an ordinary subgraph: lazy inputs (LazySwitchKJ),
hidden inputs (NKD's V3 nodes), lists, caching, interrupts and progress all work. The rest of the workflow stays
ordinary JSON.

**Make one:** select the nodes, right-click the canvas and choose **🔒 Seal as C2C Vault** or **🔒 Lock as C2C Vault**.
You can also right-click a subgraph node and choose **Protect this subgraph as C2C Vault**. The dialog shows what each
choice means (who can run it, who can open it, where it runs, where the key lives) and asks for the vault password.
This works on the root graph and inside subgraphs.
Protect is one step in the undo history (Ctrl+Z gives back the original nodes); if any stage fails, the graph and
the undo history are exactly as before. Link reroutes on the vault's wires are kept. A second Protect while one is
running is refused.
A Locked vault runs only with a key unlocked in the same browser tab (this applies to old vault nodes too); five
wrong passwords lock that vault for five minutes.

| Profile | Runs | Opens for editing |
|---|---|---|
| **Sealed, authorized installations** (default) | without a password, on your install and the installs whose codes you add | vault password + your authenticator code |
| **Locked** | with the vault password, asked once per browser tab | vault password + your authenticator code |
| **Legacy portable** | anywhere, without a password. **Obfuscation only**: a knowledgeable recipient can extract the graph | vault password + your authenticator code |

- **Open it:** double-click the vault (or right-click → **Open / edit vault…**). After the password and a code from
  your authenticator app, the real nodes open as a native subgraph, and you edit them like any subgraph. Subgraphs
  nested inside work too.
- **Leaving it locks it.** The moment you leave the subgraph, the inside is encrypted again and the vault node is
  back.
- **While it is open,** saving, autosave, exporting and queueing all write the vault locked. A run sends the server
  the vault, never its inner nodes, and copying the open vault is refused.
- **What others see:** the inputs, outputs and promoted controls, previews the inside makes, progress, and the node
  count. Messages, history and saved images name only the vault. An error inside says the inner node's type and the
  message, never its values or traceback.
- **Authenticator:** you set it up once per install (QR code or setup key, plus 8 recovery codes). A code works
  once, so the next sign-in needs the next code, and the dialog says so. A wrong device clock is diagnosed, never
  accepted. **Re-enrol** if a setup key was ever pasted anywhere outside your authenticator app.
- **Authorized installations:** a recipient sends you **My installation code** (right-click a vault), as text or as
  a key file (**Save key file**, a `.c2ckey` holding only the public key). Paste the code, or **Import key file…**,
  when sealing. Your own install is always included.
- **Manage installations & editors / Change password** (right-click a vault, 2026-10-26):
  - After the vault password and a code from your authenticator, add or remove the installations that may run it.
  - Add or remove the installations that may open it for editing. Each editor needs the password and its own
    authenticator; your install stays an editor.
  - Name the entries; the names are stored inside the encryption only.
  - Set a new password.
  - **Apply** re-encrypts the vault with fresh keys and swaps it in only when that worked.
  - Save the workflow afterwards.
  - Honest limit: removing an installation protects new copies only. A copy that installation already has still
    runs there, and a copy saved before a password change still opens with the old password.
- **Missing custom nodes** (2026-10-26):
  - A vault records, inside its encryption, which packs its nodes come from (git URL + exact commit).
  - On a machine without them, every run attempt stops and offers **Install missing custom nodes**. The offer says
    how many packs are needed and from where, never their names. It is also on the vault's right-click menu.
  - One click clones exactly those commits; restart ComfyUI, then run again.
  - Only https on github.com / gitlab.com / codeberg.org / bitbucket.org, and an existing folder is never touched.
    Installing runs the packs' code at the next start, so install only workflows you trust.
  - `C2C_VAULT_INSTALL=off` switches the install off.
- **Old vaults** (C2C Vault - Locked / Sealed nodes) upgrade automatically when a workflow opens. They keep the same
  payload and run on the native scheduler.
- **Where the install key and the authenticator live.** They go in the OS credential store (Windows Credential
  Manager, macOS Keychain, libsecret), or in an encrypted file in ComfyUI's `user/c2c/` when there is none. The OS
  store is shared by every ComfyUI install of the same OS user. Set `C2C_SECRETS_BACKEND=file` to give one install
  (a test copy, a second ComfyUI) its own key and its own authenticator. The C2C testbed sets it.
- Off switch: `C2C_VAULT_NATIVE=off` refuses native vault runs. Old vault nodes keep working.

Scope, honestly: a vault keeps your nodes out of the workflow file, browser storage, history, messages and saved
images. A vault has to be decrypted to run, so anyone who can run it can in principle extract it, and so can anyone
who runs Python inside that ComfyUI process. That means the holder of the password for a Locked vault, or the owner
of an authorized install for a Sealed one. The authenticator guards editing through the UI; it is not extra
encryption. Design: [docs/VAULT_DESIGN_A18.md](docs/VAULT_DESIGN_A18.md).

**Copy / paste with metadata** (Ctrl+C / Ctrl+V, or Ctrl+Alt+C / Ctrl+Alt+V) copies and pastes into the graph you
are looking at, inside subgraphs too. Wires between the copied nodes are kept.

The other C2C canvas tools work inside subgraphs too: sticky notes and wire labels belong to the graph they were made on, and align, find, Doctor, bookmarks, graph health and the undo history act on, or open, the graph that holds the node.

**Workflow issues.** While the graph has a model name that no longer exists or a required input that is not
connected (subgraphs included), a **⚠ N workflow issues** button shows. Click it for the list with one-click fixes.
An input set by a widget (a dropdown, a number, a toggle) never counts as unconnected, on old- and new-style nodes.
- Drag the button anywhere; Alt + arrow keys move it too.
- It never covers the console (Logs / Terminal), the canvas info or an open sidebar: it steps aside and comes back
  to your spot when they close.
- Its position, and the list's position and size, are kept in your ComfyUI settings, not in the workflow.
- Right-click: Reset position, Hide the button. Bring it back from Settings › C2C › Workflow issues or the command
  palette (**C2C: Show the Workflow Issues button**).

Design: [docs/design/workflow_issues_button.md](docs/design/workflow_issues_button.md).

## C2C App Gateway (preview): let others run a workflow without seeing it

A vault protects a workflow on your own machine. To let **other people** run a workflow while only you can open it,
run it on a server you control and put the **C2C App Gateway** in front of it. People sign in to the gateway, see a
simple app page (the inputs and outputs you chose), upload files, change the settings you allowed, follow the
progress, cancel, and download the results. They never reach ComfyUI itself: no editor, no workflow file, no node
graph, no queue or history.

- It is a separate service: `python -m c2c_app_gateway serve --config gateway.json`. ComfyUI listens only on a private
  address (for example `--listen 127.0.0.1`), and only the gateway is reachable from outside, behind your HTTPS proxy.
- Roles: **owner** (everything, including the editor at `/studio/`), **publisher** (publishes their own apps),
  **runner** (runs the apps shared with them). Add the first owner with
  `python -m c2c_app_gateway add-user --config gateway.json --name you --role owner` (it asks for the password).
- Publishing by non-owners is **off by default** (`publishers_enabled`). When you turn it on, every node type in a
  publisher's release must be on your `publisher_allowed_class_types` list; owner releases are not limited. Failed
  sign-ins are logged with a hash of the typed name, never the name itself.
- Each published version is kept unchanged, and you can roll back to an older one. The workflow is stored encrypted
  on the gateway; the app page is built from a separate list of public inputs and outputs.
- What it does not do: it does not hide the workflow from someone with administrator access to the server. Hiding
  ComfyUI's panels (App Mode) is not protection; the gateway does not rely on it.

**Publish an app:** open ComfyUI through the gateway at `/studio/` (signed in as owner or publisher), build the
workflow, then right-click the canvas and choose **📤 Publish as C2C App…** (also in the command palette). Tick the
widgets runners may set (with their limits) and the outputs they get, pick who can run it, and publish. Widgets inside
subgraphs can be published too. Ids and labels you give are public; the nodes behind them are not.

Status: preview. Tested against a real (CPU) ComfyUI with a browser: publishing from the Studio, running, progress,
cancelling and downloading, and the runner being refused everywhere else. Setting it up on a server (HTTPS proxy,
firewall, service, backups): [docs/APP_GATEWAY_DEPLOY.md](docs/APP_GATEWAY_DEPLOY.md). Finished runs and their
outputs are deleted after 72 hours by default. The Vault's `owner_hosted` option is not switched on yet.

## Automation: schedules, watch folders, webhooks

Two panels run workflows without you pressing Run. Open them from the OmniBar (**⏰ Schedule**, **👁 Watch**) or from
the command palette: **C2C: Scheduled Queue & Webhooks** and **C2C: Watch Folder**. The palette commands also work in
Lite mode, which has no OmniBar.

Every panel takes either **Current workflow** (the graph on screen, including its subgraphs) or an **API workflow
file** (Workflow › Export (API)). A normal saved workflow (nodes and links) cannot be queued without the browser, so
it is refused with that explanation. Runs are queued on this ComfyUI's own address and port, through its normal
`/prompt` check, so a broken workflow shows the same error you would get from Run. A prompt saved from ComfyUI's
History carries values ComfyUI stored during that run (`is_changed`). They are removed before queueing; otherwise
ComfyUI would serve the whole run from its cache and do nothing.

- **Schedules** take a 5-field cron (`minute hour day month weekday`, with lists, ranges, steps, `mon`/`jan` names
  and `@daily` / `@hourly`). The panel shows the next three runs as you type. Weekday 0 and 7 are Sunday. If both
  day fields are set, either one matches, as in standard cron. **New seed each run** is on by default: ComfyUI skips
  a prompt it has already run, so without new seeds every run after the first does nothing. A run missed because
  the server was busy is caught up for up to 10 minutes. The scheduler runs only while an enabled job exists.
- **Watch Folder** reacts to new image and video files. A file counts only once it has stopped growing (same size
  on two checks 2 s apart), so half-written renders are not picked up. Files already in the folder are not events.
  **Log it** records the file. **Copy to input + offer a Load node** copies it into `input/c2c_watch/` and shows a
  notice with **Add Load Image** (or **Add Load Video**), which adds the node to the graph you are looking at, inside
  a subgraph too. **Run a workflow on it** puts the file into the workflow's first Load Image (images) or Load Video
  (videos) and queues it. Scans run off the server thread and look at the top folder unless you tick **Include
  subfolders**. Watching ComfyUI's own output or temp folder with *Run a workflow* is refused, because every file
  the workflow writes would start it again.
- **Webhooks** (second tab of the Schedule panel). *Inbound*: register a workflow and get a URL. A `POST` to that
  URL queues the workflow. A JSON body `{"inputs": {"<node id>": {"<input>": value}}}` changes literal inputs for
  that run (wires are never changed). With a secret, sign the raw body: `X-Hub-Signature-256: sha256=<HMAC-SHA256>`.
  *Outbound*: a URL gets a `POST` with `{"event", "data", "ts", "history_url"}` when a run finishes, fails or is
  interrupted (other events are optional). With a secret it carries `X-C2C-Signature`. **Send test** checks the URL.

Schedules, watchers and webhooks are kept across restarts in `user/default/c2c/automation.json`. Secrets are never
written there: they go to the C2C secrets store, or stay in memory if no secrets backend is installed (the panel
says so).

## Multi-GPU: C2C Distributed

Two nodes under **🐺 C2C › 🧰 Core › Distributed** (design and measurements: `docs/c2c-distributed.md`):

- **C2C GPU Inventory** lists every GPU with its architecture (Turing, Ampere, Ada, Hopper, Blackwell incl. B200 /
  GB200 / B300 / RTX 50 / GB10), memory, BF16 / FP8 / FP4 support, and whether this torch build has kernels for it.
  It shows the P2P and NVLink links between the GPUs and spots Grace / aarch64 hosts. It also checks whether NCCL and
  gloo really work here: it creates a process group in a subprocess rather than trusting the OS name. Last, it lists
  which multi-GPU modes this machine can run.
  - **qualify** (off by default) runs the real test: one process per physical GPU, checked all-reduce /
    all-gather / reduce-scatter, latency and bandwidth, a hung GPU detected within the timeout, and the split's
    workers on every GPU.
  - It answers **MULTI_GPU_VERIFIED** only for two or more distinct GPUs that passed; otherwise SINGLE_GPU_ONLY,
    NO_GPU or FAILED.
  - On Windows, torch has no NCCL and its gloo cannot connect processes, so the multi-process part runs on Linux
    (or WSL).
- **C2C Multi-GPU Split** splits each model call over the GPUs of one machine, on Windows and Linux (no NCCL
  needed). Each GPU holds a full copy of the model, made by ComfyUI's own multi-GPU loader. Put it last in the model
  chain. With one GPU the model passes through unchanged.
  - **Any model:** the call's batch items go to different GPUs: cond and uncond with CFG, an image batch, the tiles
    from Tile Split. Measured with Flux: same result as one GPU (float32 rounding).
  - **Wan 2.1 / 2.2 and MiniMax H3 (text-to-video verified):** the tokens of one item are split too, and attention is
    exchanged between the GPUs (Ulysses all-to-all, Ring, or both). The result is the same as on one GPU:
    bit-identical for Ulysses at 2 and 4 GPUs, and within the precision's own rounding otherwise.
  - **Speed-up has not been measured yet** (that needs a machine with two or more GPUs).
  - **Limits:**
    - every GPU holds the whole model, so this does not pool memory: a model too big for one GPU does not fit here
      either (that is model sharding, a separate project);
    - ControlNet and self-attention masks are refused on the token-split path;
    - other models are batch-split only.

    The node's **plan** output lists the mode, devices, memory per GPU and how data moves between GPUs before
    sampling starts.

With Wan or MiniMax H3, the batch items go to GPU groups first (no exchange needed) and each group splits the
tokens: on 8 GPUs with CFG that is cond and uncond on 4 GPUs each. You do not need ComfyUI's own **MultiGPU CFG
Split** with this node: it splits the batch itself, over all GPUs instead of one GPU per cond.
`C2C_DISTRIBUTED=off` switches the nodes off. `C2C_DISTRIBUTED_SIMULATE=2` runs two ranks on the one GPU (or
CPU) you have, to check a setup before renting GPUs. It is slower, never faster.

**2026-11-15.** A GPU whose architecture this PyTorch build has no kernels for is refused before the run, with the
wheel to install (cu126 / cu128), instead of failing mid-run. The worker processes' model copies are released by
ComfyUI's **Unload models** (and its out-of-memory path); `C2C_DISTRIBUTED_IDLE_UNLOAD_S=<seconds>` also releases them
when idle. The inventory now records which attention backends really ran and, on cc 8.9+, an FP8 check. Multi-GPU
behaviour and speed are still not measured on real hardware.

**2026-11-13 repairs (A27).**
- The split now refuses a model that already carries block patches from other nodes, such as core's MiniMax H3 Fun
  ControlNet or a block cache. Before, it silently replaced them, so the ControlNet stopped working.
- A stuck GPU thread can no longer hang the sampler.
- With `C2C_DISTRIBUTED=off`, core's own CFG split stays intact.
- A call that falls back to one GPU is logged as a warning.
- **C2C GPU Inventory** additionally reports:
  - PTX, read from the real build flags;
  - build coverage for V100 / T4 / A100 / L40S / H100 / B200 / B300;
  - driver, CUDA, cuDNN and NCCL versions;
  - GPU UUIDs and optional extensions;
  - the four multi-GPU modes, listed separately (jobs, pieces of one job, sequence parallel, weight sharding: only
    the last one lowers memory per GPU, and no node does it yet).
- **qualify** also runs fp16 / bf16 kernels on every GPU.
- The multi-process (Linux / NCCL) rank runtime has a self-test: `python -m c2c_distributed.procrun --world 2`. It
  is verified on CPU ranks only; GPUs are not verified yet.
- **C2C Multi-GPU Split** has a new optional input, **rank_backend**, at the end. The default, `threads`, matches every
  saved workflow. `processes (Linux, NCCL)` runs one worker process per GPU, each loading its own copy of Wan or
  MiniMax H3 from the same files with the same LoRA patches. Results match one process (checked on CPU test ranks),
  but this is not yet verified on real GPUs.

## VRAM Management

All nodes are designed to work on **low-end GPUs** (4–6 GB VRAM):

- **VRAM monitoring** — model loading logs a warning when free VRAM drops below 2 GB
- **OOM recovery** — if GPU runs out of memory, nodes fall back to CPU instead of crashing
- **Offload to CPU** — SAM Model Loader has `offload_to_cpu` to keep models in system RAM
- **Auto memory cleanup** — all nodes run `gc.collect()` + `torch.cuda.empty_cache()` in `finally` blocks
- **dtype control** — use `float16` or `bfloat16` to halve VRAM usage
- **Torch-only fallbacks** — if `opencv-python` is not installed, all nodes fall back to pure PyTorch implementations (slightly slower but functional)
- **VRAM tiers** — each node declares its tier (1 = no model, 2 = one model, 3 = multiple models) so you can plan your workflow

---

## Running Tests

The test suite stays on the developer's machine and is not in this repository (owner rule 2026-10-23). CI boots
ComfyUI with this pack and fails on an import failure, no registered node, or a traceback from the pack
(`.github/c2c_boot_check.py`). Where the suite exists locally, it covers all pure-tensor nodes (no GPU models
required):

```bash
# Run all tests:
python -m pytest tests/ -v --tb=short

# Run a specific node's tests:
python -m pytest tests/test_luminance_keyer.py -v
python -m pytest tests/test_inpaint_suite.py -v
python -m pytest tests/test_mask_failure_explainer.py -v
python -m pytest tests/test_temporal_anchor.py -v
python -m pytest tests/test_sam_multi_mask_picker.py -v
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: cv2` | `pip install opencv-python>=4.7.0` — nodes will use torch fallbacks but cv2 gives better quality |
| `ModuleNotFoundError: transformers` | `pip install transformers pillow` — only needed for ViTMatte matting |
| Torch version mismatch after `pip install -r requirements.txt` | Reinstall your ComfyUI torch: check [PyTorch install page](https://pytorch.org/get-started/locally/) for your CUDA version |
| SAM model not found | Place `.pth` / `.pt` / `.safetensors` in `ComfyUI/models/sams/` or `ComfyUI/models/sam2/` |
| `Model '...' is not installed and downloading is off` | Put the named file in the folder the message gives, or tick the node's `allow_download` (MaskOps: `auto_download`) |
| Result says `approximate` / `model-free` | The model is missing; the node used its model-free method. Its `model_status` says why |
| CUDA out of memory | Enable `offload_to_cpu` in SAM Model Loader, use `float16` dtype, reduce image resolution |
| Nodes not showing in menu | Check console for `[MEC] Loaded 44` message. If missing, check for import errors in the console output |
| The browser hangs on the loading screen or the canvas stops responding | Start ComfyUI with `C2C_NO_FRONTEND=1`. This pack then serves no browser code, and its nodes still run. If the problem goes away, it comes from this pack's front end: please send the browser console (F12) output. |
| Runs feel slow on big image batches | Image batching (Settings > C2C > Image batching) only splits a batch when it would not fit in memory. The tensor inspector computes nothing during a run. `C2C_TENSOR_INSPECTOR=0` turns it off completely. The Insight node timer no longer waits for the GPU after each node; `C2C_INSIGHT=0` turns that hook off, `C2C_INSIGHT_SYNC=1` brings back exact per-node GPU times for profiling. |
| A core mask / image node is slow on long batches | Set Image batching to **Universal**. For operations C2C has a proven faster version of (today: core **Grow Mask**, **Feather Mask** and **Quantize Image**), Universal runs that version instead: it is checked against the original node on this machine before first use (results must be identical to the bit) and timed once per frame size, and the original runs wherever it is faster (e.g. tiny masks). Measured on the CPU testbed, on 97 frames of 720p, for the whole graph: Grow Mask 1.6-2.9x faster; Feather Mask 1.3x faster at a 64 px feather and 2.6x at 256 px (no measurable change at 16 px). Quantize Image runs core's own per-frame code on several frames at once (8 threads): 3.0-4.4x faster on 24 frames of 2048x1080. `C2C_AUTOBATCH_ADAPTERS=0` turns these replacements off; the timings are kept in `user/default/c2c/autobatch_profiles.json`. |
| AV Handles adds nothing with a model preset | In add mode set `padding_mode` to **model preset** (the default for new nodes): the total is then rounded onto the `model` grid or your custom step / plus. `disabled` adds exactly `handle_frames`. The report under the node says which grid was used and where the trim count came from. |
| Which nodes did Image batching change? | After a run, a node that used a faster version gets a green badge (e.g. `⚡ 1.7×`) and a node whose batch was split gets an amber one (e.g. `▦ 4×24`), in the classic canvas and in Nodes 2.0. The **Batching** tab of the C2C Diagnostics sidebar lists the same nodes for the last run; click a row to select the node. Badges clear when the next run starts. |
| ViTMatte download fails | Download manually from [HuggingFace](https://huggingface.co/hustvl/vitmatte-small-composition-1k) → place in `ComfyUI/models/vitmatte/vitmatte-small-composition-1k/` |
| My mask looks bad | Connect your image + mask to **Mask Failure Explainer** — it will diagnose the issue and suggest the right method |
| After **Convert to Subgraph** a node's inputs are swapped (for example a reference video arrives on an audio input) | A ComfyUI front-end bug with growing input lists (core MiniMax H3 Reference to Video and others). This pack puts every link back on the input it was on (Settings > C2C > Canvas > Subgraphs; on by default). A subgraph converted before this fix keeps its wrong links: undo the conversion and convert again, or re-wire the swapped inputs inside the subgraph. |

---

## Project Structure

```
ComfyUI-CustomNodePacks/
├── __init__.py              # registers every node family (guarded: one failing family never hides the rest)
├── _c2c_menu.py             # one Add Node menu root: 🐺 C2C / 🧰 Core / <family>
├── folder_incrementer.py    # Folder Version Incrementer / Control / Variant
├── nodes/                   # node implementations
│   ├── mask_matting/        # Mask + Matting (MaskOpsMEC): segmenters, matters, temporal
│   ├── c2c_video/           # Load / Save Video (C2C), Video Info, Load Images Path, handles
│   ├── magnific/            # OmniScale (Magnific API) nodes
│   ├── video_stabilizer/    # Video Stabilizer engine (classic, RAFT, dense flow)
│   ├── tiling/              # Tile Plan / Split / Merge, Identity Lock
│   ├── image_mask_editor/   # Image Mask Editor (C2C)
│   ├── prompt_relay/        # Prompt Relay
│   ├── frequency_grain/     # Frequency Separate / Combine, Film Grain
│   ├── layer_effects/, mask_toolkit/, helpers/, depthcrafter/, normalcrafter/
│   ├── legacy_compat.py, _legacy_replacements.json   # old saved workflows -> current nodes
│   └── *.py                 # the other nodes (inpaint suite, SAM, ProPainter, comparer, ...)
├── c2c_core/                # shared engine (video handle, probe, window reader...): no nodes, no pack imports;
│                            #   other packs ship a byte-identical copy (the MiniMax H3 Suite: mmx_vendor/c2c_core)
├── c2c_distributed/         # multi-GPU: GPU Inventory, Multi-GPU Split (docs/c2c-distributed.md)
├── c2c_ai/                  # C2C AI spine: model router, backends, keychain, prompt library
├── renderfarm/              # C2C Farm: spooler, backends, courier
├── js/                      # front-end extensions (classic canvas and Nodes 2.0)
├── docs/                    # per-feature documentation
├── example_workflows/       # example graphs (.json + preview)
├── NODE_REFERENCE.md        # every registered node: inputs, outputs, menu name
└── THIRD_PARTY_LICENSES/, NOTICE, CREDITS.md
```

Tests are kept out of the published repository; the maintainers run them locally.

---

## Acknowledgements

This project builds on or references the following open-source work:

| Project | Author | License | Description |
|---------|--------|---------|-------------|
| [Segment Anything](https://github.com/facebookresearch/segment-anything) | Meta AI | Apache-2.0 | SAM foundation model |
| [Segment Anything 2](https://github.com/facebookresearch/sam2) | Meta AI | Apache-2.0 | SAM2 / SAM2.1 video segmentation |
| [SAM-HQ](https://github.com/SysCV/sam-hq) | SysCV (ETH Zurich) | Apache-2.0 | High-quality SAM variant (NeurIPS 2023) |
| [ViTMatte](https://github.com/hustvl/ViTMatte) | HUST VL | MIT | Vision Transformer alpha matting |
| [Matte-Anything](https://github.com/hustvl/Matte-Anything) | HUST VL | MIT | Interactive SAM + matting pipeline |
| [MatAnyone2](https://github.com/pq-yang/MatAnyone2) | NTU S-Lab | S-Lab-1.0 | Video matting with learned quality evaluator |
| [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | PeterL1n / ByteDance | GPL-3.0 | Real-time human video matting |
| [Cutie](https://github.com/hkchengrex/Cutie) | hkchengrex | MIT | Video object segmentation (CVPR 2024) |
| [ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2) | kijai | Apache-2.0 | SAM2 ComfyUI nodes with points editor |
| [ComfyUI-SecNodes](https://github.com/9nate-drake/Comfyui-SecNodes) | 9nate-drake | Apache-2.0 | SeC 4B video segmentation nodes |
| [ComfyUI-MatAnyone](https://github.com/FuouM/ComfyUI-MatAnyone) | FuouM | MIT | MatAnyone ComfyUI wrapper |
| [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) | 1038lab | GPL-3.0 | RMBG-2.0 / BiRefNet background removal |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | kijai | GPL-3.0 | Utility node collection |
| [ComfyUI-Inpaint-CropAndStitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch) | lquesada | GPL-3.0 | Crop-and-stitch inpainting reference |
| [ComfyUI-VideoMaMa](https://github.com/okdalto/ComfyUI-VideoMaMa) | okdalto | — | VideoMaMa mask-guided video matting |
| [ComfyUI LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) | chflame163 | MIT | Layer compositing node suite |
| [ComfyUI LayerStyle Advance](https://github.com/chflame163/ComfyUI_LayerStyle_Advance) | chflame163 | MIT | Advanced nodes (BiRefNet, Florence2, etc.) |

---

## License

MIT

---

## Use in image/video generation pipelines (Flux / Qwen-Image / Wan / Z-Image / ERNIE-VL)

This umbrella pack aggregates nodes across four sub-namespaces — Folder Incrementer (`_FOLDER_MAPPINGS`), MEC tools (`_MEC_MAPPINGS` covering masking, SAM/ViT-Matte segmentation, inpaint crop/stitch, spline editors, motion mask trackers, draw shapes, VAE merge/inspect, batch version manager, temporal consistency, model metadata, image comparer, parameter history, semantic segment, luminance keyer, mask failure explainer, temporal anchor, SAM multi-mask picker), Paint nodes (`_PAINT_MAPPINGS`), and MA nodes (`_MA_MAPPINGS`). The nodes are model-agnostic — they speak ComfyUI's native `IMAGE`, `MASK`, `LATENT`, `VAE`, and `MODEL` types — so they slot into any sampler graph regardless of which diffusion family produces the pixels.

| Model family | Recommended bridges |
|---|---|
| **Flux** | Use `InpaintCropProMEC` + `InpaintStitchProMEC` (`stitch_mode` blend or paste back) to confine high-resolution Flux refinement to a tracked region. `MaskOpsMEC` (Mask + Matting, backend SAM + ViTMatte, with its own trimap settings) produces the high-quality alpha that Flux-driven matte refinement needs. `FolderIncrementer` keeps versioned outputs across sweeps. `VAEMergeMEC` and **Probe (C2C)** with the latent wired (or **VAE Inspect** in compare mode) are useful when comparing Flux VAE variants. |
| **Qwen-Image** | Identical inpaint and matte pattern as Flux. `MaskToolsMEC` in `transform` mode (shift / grow / feather) and `MaskTrackerMEC` (propagate one mask over a sequence) are useful when reusing a single Qwen-Image generated mask across a small sequence. `VideoComparerC2C` for A/B sweeps over guidance / sampler choices (images too). |
| **Wan 2.x (video)** | `MaskTrackerMEC` (motion tracking, propagation, key-frame anchors and a temporal-consistency check in one node) and `MaskOpsMEC` (backend SeC + MatAnyone2) are designed for the video domain — keep masks coherent across Wan-Animate frames, anchor a key-frame mask, and audit temporal jitter. `MaskFailureExplainerMEC` flags problem frames before re-rendering. Pair with `ComfyUI-WanAnimatePreprocessV2` (human) or `ComfyUI-WanAnimalPreprocess` (animal) for the pose / face conditioning. |
| **Z-Image** | Same inpaint and refine bridges as Flux/Qwen. The spline-mask, draw-shape, and points-mask editors are useful when authoring a custom region for Z-Image's refinement passes. |
| **GLM-Image** | The IMAGE outputs feed cleanly into the GLM-Image I2I path (via the sampler's optional `image` input + `denoise_strength`). `MaskOpsMEC` (backend remove background) + `SAMMaskGeneratorMEC` provide the alpha pre-stage. |
| **ERNIE-VL** | Not applicable for sampling (ERNIE-VL is a multimodal LLM). The `ModelMetadataExtractorMEC` and `ParameterHistoryMEC` nodes can still be useful for cataloguing assets that ERNIE-VL has captioned. |

Two general patterns recur across all sampler families:

1. **Inpaint pair** — `InpaintCropProMEC` → diffusion sampler (any family) → `InpaintStitchProMEC` (`stitch_mode` paste back for a plain paste, blend for soft seams). Allows full-resolution refinement at a fraction of the VRAM, regardless of which model is loaded.
2. **Versioned saves** — wire the Folder Version Incrementer's `version_string` (layout show / shot / task / version for a pipeline tree) into the sampler's `Save Image.filename_prefix` so every queued prompt produces a clean, ordered file series.

ComfyUI's own reroutes (and the SAM / ViT-Matte loaders) keep the graph readable as it scales from a single Flux T2I bench into a multi-model Wan + Qwen-refine + Flux-detail composite.

---

<p align="center">
  Made with ❤️ for the ComfyUI community
</p>


---

## Full Node & Parameter Reference

A complete, auto-generated reference covering every node, return type and widget exposed by this pack lives in [NODE_REFERENCE.md](./NODE_REFERENCE.md). It is regenerated whenever `INPUT_TYPES` change.

