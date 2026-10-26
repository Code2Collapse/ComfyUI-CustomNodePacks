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
**ComfyUI-CustomNodePacks** registers **115** nodes across these families:

| Family | Nodes |
|--------|------:|
| Vault | 6 |
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

`[C2C] CustomNodePacks: 115 nodes loaded (Vault 6, Folder Incrementer 5, Render farm 3, C2C Distributed 2, Fluid Shots/Audio 3, C2C video 4, MaskEditControl 14, Image Mask Editor 1, Model analysis 1, MEC Paint Suite 2, Face Fixer 1, Face/Pose Delta 1, Mask + Matting 2, Layer Effects 1, Mask toolkit 1, Frequency / Grain 3, Smart Crop/Stitch 2, Tiled refinement 4, Magnific 15, AV Handles 1, Save Video 1, SAM Multi-Mask Picker 1, SAM Loader/Generator 2, Mask Placement 1, Semantic Segment 1, NukeNodeMax 1, Video Stabilizer 4, C2C helpers 8, Prompt Relay 3, AsymFlow sampler 1, HDR Color Science 1, LocateAnything 2, Nano Banana 1, Control AOV 1, Clipboard TCL 2, Restored VFX 6, Legacy (deprecated) 7) - 0 failed`
<!-- C2C:NODE-COUNTS:END -->


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
| MaskOps | `auto_download` | its labelled model-free segmenter / matter |

When a model is missing and downloading is off, the message names the model, the **absolute** folder and the file,
for example `Expected file 'sam2.1_hiera_base_plus.safetensors' in folder: D:\ComfyUI\models\sam2`. Put the file
there, or tick the switch to download it there. LocateAnything goes to `models/LLM/LocateAnything-3B`.

Each of these nodes reports a `model_status` line in its `info` output (SAM Model Loader: in its `SAM_MODEL`):
`loaded - found at <path>`, `needs permission - model-free result` (missing, downloading off),
`missing - model-free result` (missing for another reason) or, on Mask Placement, `not used (source_mask wired)`.

A saved SAM Model Loader choice `[download] ...` no longer downloads by itself: tick `allow_download`.

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
| [Inpaint Suite](docs/inpaint-suite.md) | 4 | Crop Pro, Stitch Pro, Paste Back, Mask Prepare — full pipeline with fill modes, blend modes, interpolation methods |
| [**ProPainter Suite**](docs/propainter.md) ★ new | 5 | Temporal video inpaint, stitch (drop-in for legacy), seam-only refine, model-free remove, RAFT flow refine |
| [**Face Fixer**](docs/face-fixer.md) ★ new | 1 | YOLO11 face detection → per-face KSampler → smart blend with `[SEP]/[ASC]/[DSC]/[SKIP]` wildcard prompts |
| [**Paint Suite**](docs/paint-suite.md) ★ new | 2 | Interactive paint canvas (hardness/expansion/blur stages) + smart context inpainter (color match, lightness rescue, differential diffusion) |
| [**Mask + Matting Pipeline**](docs/mask-matting-pipeline.md) ★ new | 1 | Combined `MaskOpsMEC` (Mask + Matting) — pick segmenter (SAM2.1/SAM3/SeC/…) + matter (ViTMatte/RVM/MatAnyone2/…) + slot-based prompts |
| [**Video Stabilizer**](docs/video-stabilizer.md) | 1 | Feature tracking, OpenCV dense flow or RAFT optical flow (experimental), 4 presets, measured accuracy table; padding-mask output wires into Inpaint Crop Pro |
| [**Image Mask Editor**](docs/image-mask-editor.md) ★ new | 1 | Native-resolution mask editor over the image that reaches the node: brush/eraser/shapes/polygon/lasso/bucket, colour range (RGB/HSV/LAB), SAM clicks and boxes, ViTMatte edge-refine brush, batch frame strip, masks survive restarts, non-destructive grow/feather/threshold |
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

## Node Reference

### Legend

All nodes appear in the Add Node menu under **🐺 C2C / 🧰 Core**. That is the one menu root shared by every Code2Collapse pack: NukeMax, MiniMax H3, Wan Experiments, Wan Animate Preprocess, GLM Image, Wan Animal Preprocess and LTX Enhanced each have their own branch under it.

| Symbol | Meaning |
|--------|---------|
| ★★★★★ | Best quality (may need optional deps) |
| ★★★★☆ | Great quality |
| ★★★☆☆ | Good / fast fallback |

---

### Segmentation & SAM

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

#### Unified Segmentation (MEC)

One-node dispatcher for **SAM 2.1, SAM 3, SeC, VideoMaMa** with automatic image vs. video detection.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | — | Model selector |
| `points_json` / `bbox_json` | `""` | Point and box prompts |
| `precision` | `fp16` | `fp16` / `bf16` / `fp32` |
| `attention_mode` | `auto` | `sdpa` / `flash_attn` / `sage_attn` / `xformers` |
| `text_prompt` | — | (optional) Text prompt for grounding models |

**Outputs:** `masks`, `best_score`, `info`

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

End-to-end pipeline: **SeC MLLM segmentation → MatAnyone2 temporal alpha matting**. Since L7.65 P05 this is Mask +
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

### Alpha Matting

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

#### Matting Node (MEC)

Unified alpha matting with **7 backends**. Takes a coarse mask and returns compositing-grade alpha.

| Backend | Quality | Requires | Best for |
|---------|---------|----------|----------|
| `vitmatte_small` / `vitmatte_base` | ★★★★★ | `transformers` | Hair, fur, glass, transparency |
| `matanyone2` | ★★★★★ | Model download | Video matting, temporal consistency |
| `rvm_mobilenetv3` / `rvm_resnet50` | ★★★★☆ | Model download | Real-time video matting |
| `cutie` | ★★★★☆ | Model download | Video object cutout |
| `sam_hq` | ★★★☆☆ | Model download | Quick high-quality mask |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `auto` | Backend selection (auto picks best available) |
| `edge_radius` | `15` | Trimap unknown-band width |
| `erode_dilate` | `0` | Pre-process mask morphology (−50 to +50) |
| `n_warmup` | `5` | Warmup frames for video backends |

**Outputs:** `rgb` (premultiplied), `alpha_mask`

---

#### ViTMatte Edge Refiner (MEC)

Standalone edge refinement — feed any coarse mask and get clean edges.

| Method | Quality | Requires | Best for |
|--------|---------|----------|----------|
| `vitmatte` | ★★★★★ | `transformers` | Hair, fur, glass, complex edges |
| `multi_scale_guided` | ★★★★☆ | `opencv-python` | General high-quality |
| `color_aware` | ★★★★☆ | `opencv-python` | Challenging lighting |
| `guided_filter` | ★★★☆☆ | `opencv-python` | Fast good-quality |
| `laplacian_blend` | ★★★☆☆ | `opencv-python` | Smooth blending |
| `gaussian_blur` | ★★☆☆☆ | (none) | Simple fallback |

---

#### Trimap Generator (MEC)

Generates a 3-region trimap (white = foreground, black = background, gray = unknown) for ViTMatte input.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `edge_radius` | `15` | Width of the unknown boundary in pixels |
| `inner_erosion` | `1.0` | Foreground erosion scale (<1 tighter, >1 wider) |
| `outer_dilation` | `1.5` | Background dilation scale |
| `smooth` | `0.0` | Gaussian smoothing of boundaries |
| `threshold` | `0.5` | Binarization threshold |

**Outputs:** `trimap`, `foreground`, `unknown`

---

### Keying

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

### Mask Editing & Transform

#### Mask Transform XY (MEC)

Independent per-axis mask manipulation: erode/expand, directional blur, offset, feather, threshold, invert.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expand_x` / `expand_y` | `0` | Per-axis morphological expand (negative = erode) |
| `blur_x` / `blur_y` | `0.0` | Per-axis directional Gaussian blur |
| `offset_x` / `offset_y` | `0` | Translate mask in pixels |
| `feather` | `0.0` | Isotropic edge feathering |
| `threshold` | `0.0` | Binarization threshold (0 = disabled) |
| `invert` | `false` | Invert the mask |

---

#### Mask Draw Frame (MEC)

Low-level shape drawing. Accepts a `shape` + JSON `shape_params_json`. Supports all 12 shapes (circle, rectangle, ellipse, polygon, line, triangle, star, diamond, cross, rounded_rectangle, heart, arrow) with SDF rendering, feathering, rotation, and blend operations.

> **Prefer Draw Shape (MEC)** for new workflows — it exposes all parameters as named inputs instead of raw JSON.

---

#### Draw Shape (MEC) ★ New

Unified 12-shape drawing node with a single dropdown. All parameters are visible as named inputs with tooltips — irrelevant ones are ignored per shape. Replaces the 5 legacy per-shape wrapper nodes.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shape` | `circle` | Dropdown: circle, rectangle, ellipse, polygon, line, triangle, star, diamond, cross, rounded_rectangle, heart, arrow |
| `cx` / `cy` | `256` | Center position (center-based shapes) |
| `radius` | `50` | Circle radius, triangle/heart size |
| `size_w` / `size_h` | `200` / `100` | Rectangle, rounded rect, diamond, arrow width/height |
| `rx` / `ry` | `100` / `50` | Ellipse radii |
| `top_left_x` / `top_left_y` | `100` | Rectangle/line start position |
| `x2` / `y2` | `400` | Line end position |
| `thickness` | `5` | Line, cross thickness |
| `outer_r` / `inner_r` | `100` / `40` | Star radii |
| `num_points` | `5` | Star/polygon point count |
| `corner_radius` | `20` | Rounded rectangle corners |
| `arrow_length` / `head_length` / `head_width` | — | Arrow dimensions |
| `points_json` | — | Polygon vertices: `[[x1,y1], ...]` |
| `value` | `1.0` | Fill intensity |
| `feather` | `0.0` | Edge softness |
| `rotation` | `0.0` | Rotation in degrees |
| `operation` | `set` | `set` / `add` / `subtract` / `max` / `min` |
| `batch_size` | `1` | Number of mask frames |

**Optional:** `coords_json` (per-frame position override), `existing_mask`, `reference_image`

**Output:** `mask` (MASK)

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

#### Inpaint Crop Pro

Crops the image around the mask so the inpaint model works on the region at its own resolution; **Inpaint Stitch
Pro** puts the result back. Same workflow shape as lquesada's Inpaint Crop and Stitch, with Wan 2.2 Animate extras.

**What it does, in order:** optional pre-resize of the whole image; mask clean-up (fill holes, expand, invert,
feather, hi-pass); optional outpaint extension; the context box = the mask's bounding box grown by
`context_from_mask_extend_factor` (plus `optional_context_mask`); fit to the target aspect, rounded to
`output_padding` and `wan_align_multiple`; resize to the target size.

**The crop window stays on real pixels.** A window that fits inside the image is moved inside it (the least
distance); only a window larger than the image is padded with edge pixels. (Before 2026-10-13 a mask near the border
got a window padded with smeared edges - 41 % of a 16:9 window - although one fitting the image existed.)

**Preview.** After a run the node shows one picture: the original with the discarded area dimmed, the regenerated
area tinted and the window outlined (an orange side = padded past the image), next to the crop the model gets, with
the window size, position, output size, scale and padding written under it. It is drawn from the window that ran.
`preview_frame` picks the frame shown (default 0, the first); it changes nothing else.
After that first run the picture follows the crop settings as you change them, without running the graph (the
same geometry, from the last run's masks; marked "live"; Settings > C2C > Canvas > Inpaint crop > Live crop
preview turns it off). The node's outputs change when it runs. Inside a subgraph, or for another preview frame,
the picture changes on the next run.

**Batches and video.** `wan_stable_crop` on (default): one window for every frame - the union of all frames'
masks - so the crop does not jump. Off: each frame gets its own window, and every frame is cropped and stitched with
its own placement. The preview label says which.

**Controls.** Nine are shown: `mask_expand_pixels`, `mask_blend_pixels`, `context_from_mask_extend_factor`,
`aspect_preset`, `output_target_width`, `output_target_height`, `wan_stable_crop`, `stitch_blend_mode`,
`preview_frame`. The rest sit in sections - Mask, Size, Resize / outpaint, Video, Stitch, Device - opened from the
node's **More** bar (classic canvas) or core's **Show advanced inputs** (Nodes 2.0). A switch shows its own
controls when it is on (pre-resize sizes, outpaint factors, the `video_stable` settings).

**Output size.** The stitched image has the size of the image the crop worked on: the input size, or the
pre-resized size with `preresize`, or the extended size with `extend_for_outpainting`.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `image`, `mask` | — | Image batch and the area to regenerate (white). `optional_context_mask` widens the context box |
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

**Outputs:** `stitcher` (to Stitch Pro), `cropped_image`, `inpaint_mask` (for the model), `stitch_blend_mask`,
`info` (settings and frame 0's window).

---

#### Inpaint Stitch Pro

Puts the inpainted crop back into the image Inpaint Crop Pro worked on, frame by frame, at the window each frame
was cropped from.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `stitcher` | — | From Inpaint Crop Pro |
| `inpainted_image` | — | The sampler's result (RGB; an RGBA plate keeps its own alpha) |
| `blend_mode_override` | `from_crop` | Use another blend mode than the crop chose |
| `color_match` | off | Mean + std colour transfer before blending |
| `stitch_temporal_sigma`, `stitch_dilate_px` | 0, 0 | Smooth / widen the blend mask over time for jittery video masks |
| `roto_quality_override` | `from_crop` | `force_on` / `force_off` the crop's roto setting |

**Outputs:** `image`, `blend_mask_used` (exactly where the stitch changed pixels: everywhere it is 0 the original is
untouched, bit for bit), `info`.

Checked by `tests/test_inpaint_crop_geometry.py`: an empty mask stitches back as an exact identity; nothing outside
the blend mask changes for every blend mode; without resizing the round trip is exact; the result lands where it
was cropped; RGBA, odd sizes, `preserve_subject`, per-frame and stable batches.

---

#### Inpaint Mask Prepare (MEC)

Standalone mask cleanup and dual-mask preparation for inpainting workflows.

**How it works:** Fills holes, removes small disconnected blobs, dilates the mask, then produces two outputs — a clean inpaint mask (for the model) and a stitch blend mask (for compositing). Optional temporal smoothing for video batch consistency.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mask` | — | Raw input mask |
| `fill_holes` | `true` | Fill interior holes |
| `remove_small_regions` | `true` | Remove disconnected blobs |
| `min_region_area` | `100` | Minimum region area in pixels |
| `grow_pixels` | `4` | Dilate mask by N pixels |
| `inpaint_edge_mode` | `hard_binary` | `hard_binary` / `slight_feather` |
| `stitch_edge_mode` | `gaussian` | `gaussian` / `edge_aware` |
| `stitch_feather_radius` | `16` | Feather radius for stitch blend |
| `temporal_smooth` | `false` | Gaussian smoothing along batch dimension |
| `temporal_sigma` | `1.5` | Temporal smoothing sigma (frames) |
| `reference_image` | (optional) | For edge-aware stitch blend |

**Outputs:** `inpaint_mask`, `stitch_blend_mask`, `debug_preview`, `info`

---

#### Inpaint Paste Back (MEC)

Simple paste-back node: resize the inpainted crop and composite it onto the original with optional Gaussian-feathered alpha blending.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stitch_data` | — | From Inpaint Crop Pro |
| `inpainted_image` | — | Inpainted result from any model |
| `feather_edges` | `true` | Gaussian-feathered alpha blending at crop boundary |
| `feather_radius` | `8` | Blur radius for edge feathering (pixels) |

**Outputs:** `image`, `info`

> **Paste Back vs Stitch Pro:** Stitch Pro blends the result back with the crop's blend mask (and its blend modes); Paste Back is a plain resize + paste with an optional feathered edge.

---

#### Image Comparer (MEC)

Interactive before/after comparison widget with 3 modes: drag-slider split, overlay blend, and difference heatmap.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_a` | — | Left / "before" image |
| `image_b` | — | Right / "after" image |
| `label_a` | `"Before"` | (optional) Label for image A |
| `label_b` | `"After"` | (optional) Label for image B |

**Modes:** ◧ Compare (drag slider) · ⊕ Overlay (alpha blend) · ≠ Diff (heatmap)

> **Who is it for?** Quick visual comparison of before/after results — inpainting QA, model comparison, parameter tuning.

---

### Batch & Video

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
- The player works the same in Load Video Path (C2C) (crop included) and Load Images Path (C2C) (no crop), in both
  renderers, and inside subgraphs.

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

#### Video Mask Editor (MEC)

Paint masks on the frames of a video. Run the workflow once so the editor has frames, then click *Open Video Mask
Editor* on the node.

- **Frame by frame.** A frame you paint becomes a keyframe, and its mask is used on that frame only. Unpainted frames
  follow two choices, made in the editor (or on the node):
  - *Between two painted frames:* morph the shape, blend, hold the previous one, or none (frame by frame).
  - *Before the first / after the last:* empty (default) or hold the nearest painted frame.
  - With `input_mask` connected (for example a tracked mask), every frame these rules leave empty keeps its own frame
    of the input mask.
- **What you see is what the node outputs.** Every frame shows its real mask, with a label: painted, interpolated
  (frames 2-6), held, input mask, or no mask. Painting on an unpainted frame starts from the mask shown and turns that
  frame into a keyframe. The timeline marks keyframes and shades the interpolated and held spans.
- **Full resolution.** Frames are shown at the clip's own size (the *frame size* setting caps it, 4096 px by default),
  and the mask is painted at the clip's resolution.
- **Tools:** brush, eraser, fill, lasso, rectangle, ellipse, polygon, star, line, roto spline; add / subtract (X, or
  Alt while painting); undo / redo; copy the nearest keyframe; clear, fill, invert the frame.
- **Views:** overlay (5 colours, opacity), mask (white on black), matte (the frame cut out by the mask). Onion skin
  shows the neighbouring keyframes.
- **Saved as you go.** Keyframes are written to `input/c2c_video_masks/<session>/`, so they survive a restart of
  ComfyUI. *Revert* undoes every change made since the editor opened; *Done* closes it.
- **Keys:** B E G L R C P T N Y tools, [ ] size, Left / Right frame, Shift+Left / Right previous / next keyframe,
  Home / End, / play, K make keyframe, Delete clear, I invert, V view, O onion skin, Shift+F fit, Ctrl+Z /
  Ctrl+Shift+Z undo / redo, wheel zoom, Space or middle-drag pan.
- Works in both renderers and inside subgraphs.

Until 2026-10-23 the first / last painted frame was held over the whole clip, so one painted frame masked every
frame, and a connected `input_mask` was reduced to its first frame. Saved workflows now get *empty* outside the
painted frames; choose *hold* on `outside_keys` for the old result.

#### Mask Batch Manager (MEC)

Manipulate mask batches for video workflows.

| Operation | Description |
|-----------|-------------|
| `slice` | Extract frame range `[start:end]` |
| `pick_frames` | Select specific frames by index |
| `repeat` | Repeat mask N times |
| `reverse` | Reverse frame order |
| `concat` | Join two mask batches |
| `interleave` | Alternate frames from two batches |
| `insert` | Insert mask_b at position |
| `remove` | Remove frame at index |

---

#### Mask Propagate Video (MEC)

Draw mask on frame 1 → propagate across all frames.

| Mode | Description |
|------|-------------|
| `static` | Same mask on every frame |
| `fade` | Linear fade to zero |
| `scale_linear` | Linear scale over time |
| `optical_flow` | Track mask using optical flow |
| `sam2_video` | Use SAM2 video predictor for tracking |

---

#### Temporal Anchor System (MEC)

Mask interpolation over time using Signed Distance Fields (SDF). Define masks on a few key frames and let the node smoothly morph between them across the full video.

**How it works:** Computes an SDF for each anchor mask (distance from boundary, negative inside, positive outside), then interpolates the SDF fields between anchor frames using configurable easing. The zero-crossing of the blended SDF produces the interpolated mask boundary. Optional optical flow refinement (Farneback or FFT phase correlation) warps the SDF to follow motion.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `anchor_masks` | — | One mask per anchor frame (A, H, W) |
| `anchor_frames` | `"0"` | Comma-separated frame indices (e.g. `"0,10,30"`) |
| `total_frames` | `30` | Total output frames |
| `easing` | `smooth_step` | `linear` / `ease_in` / `ease_out` / `smooth_step` |
| `sdf_iterations` | `64` | SDF diffusion iterations (more = more accurate) |
| `flow_refinement` | `false` | Enable optical flow refinement |
| `images` | (optional) | Video frames for optical flow estimation |

**Outputs:** `full_masks` (total_frames × H × W), `confidence` (per-frame float list), `info`

> **Who is it for?** Video editors who need smooth mask morphing between keyframes — rotoscoping helpers, animated mask transitions, temporal mask interpolation where SAM tracking is overkill.

---

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

### Diagnostics

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

### Interactive Editor

#### Image Mask Editor (C2C)

Paints a mask over the image that actually reaches the node, at native resolution (`C2C/Masking`).
Click **✏️ Edit mask** to open the full-window editor.

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

#### Points Mask Editor (MEC)

Full-featured interactive canvas editor for placing points and bounding boxes directly on your image.

| Action | Effect |
|--------|--------|
| **Left click** | Add positive point (foreground) |
| **Right click** | Add negative point (background) |
| **CTRL + Left drag** | Draw positive bounding box (green) |
| **CTRL + Right drag** | Draw negative bounding box (red) |
| **Shift + Click** | Delete element under cursor |
| **Scroll wheel** | Adjust point radius |
| **CTRL + Scroll** | Zoom in/out |
| **Middle mouse drag** | Pan canvas |
| **Delete / Backspace** | Delete hovered element |
| **CTRL + Z / CTRL + Shift + Z** | Undo / Redo |
| **R** | Reset view |

**Toolbar:** Pill counters for +pts / −pts / bbox count / radius. Buttons for **✕ Pts**, **✕ BBox**, **✕ All**, **↶ Undo / Redo ↷**, **▣ Fit**.

**Outputs (8):**

| Output | Type | Description |
|--------|------|-------------|
| `mask` | MASK | Rendered points/bboxes mask |
| `positive_coords` | STRING | `[{"x":int,"y":int}, ...]` for SAM |
| `negative_coords` | STRING | `[{"x":int,"y":int}, ...]` for SAM |
| `bboxes` | BBOX | Positive bounding boxes |
| `neg_bboxes` | BBOX | Negative bounding boxes (SAM3) |
| `points_json` | STRING | Full point data for SAM Mask Generator |
| `bbox_json` | STRING | Primary bbox for SAM Mask Generator |
| `primary_bbox` | BBOX | `[x,y,w,h]` for BBox pipeline |

Connect a `reference_image` to see it as the editor background for precise placement.

---

#### Spline Mask Editor (MEC) ★ New

Interactive canvas for drawing spline-based masks directly on your image. Supports Catmull-Rom (smooth curves through points), Bezier (with tangent handles), and polyline (straight segments) modes.

| Action | Effect |
|--------|--------|
| **Left click** | Add control point (or close path by clicking first point) |
| **Shift + click** | Delete point under cursor |
| **Ctrl + click** | Insert point on nearest curve segment |
| **Right-click** | Context menu (Delete Point, Open/Close, Smooth/Sharp) |
| **S key** | Toggle smooth / sharp interpolation for current path |
| **Scroll wheel** | Zoom in/out |
| **Middle mouse drag** | Pan canvas |

**Key improvements:** Normalized [0,1] coordinates (resolution-independent), zoom-relative point sizes, segment insertion, close-by-clicking-first-point affordance (highlighted orange), property-based persistence.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spline_type` | `catmull_rom` | `catmull_rom` / `bezier` / `polyline` |
| `closed` | `true` | Close the spline loop for filled region |
| `smoothing` | `true` | Enable spline smoothing |
| `samples_per_segment` | `20` | Curve resolution (higher = smoother) |
| `feather_radius` | `0.0` | Gaussian edge softness |
| `invert` | `false` | Fill outside spline instead of inside |

**Outputs:** `mask` (MASK), `coords_json` (SAM-compatible points), `spline_data_out` (for Motion Mask Tracker)

---

#### Motion Mask Tracker (MEC) ★ New

Per-frame motion detection with 4 independent methods and camera stabilization. Feed a video batch — get a motion mask highlighting what moved between frames.

| Method | Description |
|--------|-------------|
| `pixel_diff` | Absolute per-pixel brightness change |
| `optical_flow` | Farneback dense flow or phase correlation |
| `background_sub` | Static background model from first N frames |
| `histogram_diff` | Per-region color histogram distance |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_compensation` | `true` | Subtract global camera motion (homography/affine/translation) |
| `detection_mode` | `combined` | `combined` / individual method |
| `combine_method` | `union` | `union` (any triggers) / `intersection` (all agree) |
| `grow_pixels` | `4.0` | Expand detected regions |
| `min_region_size` | `100` | Remove noise regions < N pixels |
| `temporal_smooth` | `true` | Gaussian smoothing across frames |

**Outputs:** `motion_mask` (MASK), `motion_intensity` (FLOAT), `info` (STRING)

> **Who is it for?** Video editors who need automatic motion masks for selective effects, stabilization-aware change detection, or motion-triggered workflows.

---

### Preview

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

### Utilities

#### Universal Reroute / Dot (MEC)

Nuke-style reroute dot that accepts **any** connection type (IMAGE, MASK, LATENT, STRING, INT, FLOAT, etc.). Use it to keep your workflow wires clean and organized.

#### Parameter History (MEC)

Tracks every parameter change across ComfyUI runs in a local SQLite database. Query with:

| Mode | Description |
|------|-------------|
| `all_history` | Full parameter history for last N runs |
| `last_run_diff` | What changed between the last two runs |
| `node_class_filter` | Filter history by node class name |

The in-editor side (right-click a node > Parameter History; Alt + hover a widget) records every change in both
renderers without taking over the widget's value, so a saved workflow shows its own values in Nodes 2.0.

---

### FolderIncrementer

Filesystem-aware auto-versioning for output files.

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

## Node Quick-Reference Table

Nodes at a glance (a selection - NODE_REFERENCE.md lists every node):

| # | Node | Category | VRAM Tier | What it does |
|---|------|----------|-----------|-------------|
| 1 | SAM Model Loader | SAM | 2 | Load SAM 2.1 / SAM 3 checkpoints |
| 2 | SAM Mask Generator | SAM | 2 | SAM inference with point + bbox prompts |
| 3 | SAM Multi-Mask Picker | SAM | 2 | View all 3 SAM candidates, pick interactively |
| 4 | Unified Segmentation | SAM | 2 | One-node dispatcher for all segmentation backends |
| 5 | Mask + Matting, backend SAM + ViTMatte | SAM / Matting | 2 | SAM → refinement → ViTMatte end-to-end |
| 6 | Mask + Matting, backend SeC + MatAnyone2 | SAM / Matting | 3 | SeC → MatAnyone2 temporal video pipeline |
| 7 | Semantic Segment | Segmentation | 2 | SegFormer face/body/clothes parsing |
| 8 | Mask + Matting, backend remove background | Matting | 2 | One-click RMBG / BiRefNet removal |
| 9 | Matting Node | Matting | 2 | Unified 7-backend alpha matting |
| 10 | ViTMatte Edge Refiner | Matting | 2 | Standalone edge refinement (7 methods) |
| 11 | Trimap Generator | Matting | 1 | Generate trimap for ViTMatte input |
| 12 | Mask Tools (C2C) | Keying / matte ops | 1 | colour, luma (BT.709, smoothstep) and difference keys; grade; directional grow / shrink; combine; gradient; grain; motion blur; edge spread; shuffle |
| 13 | Mask Transform XY | Editing | 1 | Per-axis erode/expand/blur/offset |
| 14 | Mask Draw Frame | Editing | 1 | Draw 12 shapes with rotation and SDF |
| 15 | Draw Shape | Editing | 1 | Unified 12-shape dropdown with all params — replaces per-shape wrappers |
| 16 | Mask Composite Advanced | Editing | 1 | Boolean/blend two masks (8 operations) |
| 17 | Mask Math | Editing | 1 | Mathematical mask operations (11 ops) |
| 18 | Spline Mask Editor | Interactive | 1 | Interactive Catmull-Rom / Bezier / polyline spline drawing on canvas |
| 19 | Inpaint Crop Pro | Inpaint | 1 | Crop around mask for inpainting |
| 20 | Inpaint Stitch Pro | Inpaint | 1 | Blend the inpainted crop back at the window it came from |
| 21 | Inpaint Paste Back | Inpaint | 1 | Simple paste-back with feathered alpha |
| 22 | Inpaint Mask Prepare | Inpaint | 1 | Clean + dual-mask preparation |
| 23 | Image Comparer | Preview | 1 | Interactive before/after comparison (3 modes) |
| 24 | Mask Batch Manager | Video | 1 | Slice/concat/interleave mask batches |
| 25 | Mask Propagate Video | Video | 1–2 | Propagate mask across video frames |
| 26 | Temporal Anchor System | Video | 2 | SDF-based mask interpolation between keyframes |
| 27 | Motion Mask Tracker | Video | 1 | Per-frame motion detection (4 methods, camera stabilization) |
| 28 | Batch Range (C2C) | Helpers | 1 | Extract single frame (`first frame` / `last` / `middle` / `frame at index`) |
| 29 | BBox Create | BBox | 1 | Manual bbox entry |
| 30 | BBox From Mask | BBox | 1 | Extract bbox from mask |
| 31 | BBox To Mask | BBox | 1 | Convert bbox to mask |
| 32 | BBox Pad | BBox | 1 | Asymmetric bbox padding |
| 33 | BBox Crop | BBox | 1 | Crop image + mask to bbox |
| 34 | BBox Smooth Temporal | BBox | 1 | Smooth bbox sequences across video frames |
| 35 | Mask Failure Explainer | Diagnostics | 1 | Diagnose bad masks, suggest fixes |
| 36 | Points Mask Editor | Interactive | 1 | Canvas editor for points/bboxes |
| 37 | Mask Preview Overlay | Preview | 1 | 5-mode mask visualization |
| 38 | Universal Reroute / Dot | Utility | 1 | Any-type wire reroute |
| 39 | Parameter History | Utility | 1 | Track parameter changes over runs |
| 40 | Folder Version Incrementer | Output | 1 | Auto-versioned file output |
| 41 | Folder Version Check | Output | 1 | Check existing versions |
| 42 | Folder Version Set | Output | 1 | Reserve version slots |
| 43–47 | Draw Circle / Rectangle / Ellipse / Polygon / Line | Editing | 1 | *(Deprecated)* Legacy per-shape wrappers — use Draw Shape instead |
| 48 | VAE Merge | VAE | 1 | Merge 2/3 VAEs (8 algorithms, per-block alpha) |
| 49 | Probe (C2C) | Helpers | 1 | Latent stats, NaN/Inf checks, health verdict (wire `latent`) |
| 50 | VAE Inspect | Model analysis | 1 | Compare two VAEs (`mode` compare, `vae` + `vae_b`) |
| 51 | VAE Inspect | Model analysis | 1 | Per-block weight stats for a single VAE (`mode` inspect) |
| 52 | Color Space Convert | Color | 1 | sRGB ↔ linear ↔ Rec.709 ↔ ACEScg |
| 53 | LUT Apply (.cube) | Color | 1 | Adobe `.cube` 1D/3D LUT with strength blend |
| 54 | Exposure Grade | Color | 1 | Stops + WB temp/tint + contrast pivot |
| 55 | Load EXR | I/O | 1 | OpenEXR → imageio → TIFF fallback |
| 56 | Save EXR | I/O | 1 | Half-float by default, TIFF fallback |
| 57 | EXR Metadata Reader | I/O | 1 | Pure-python header parser (no OpenEXR needed) |
| 58 | Merge Render Passes | Render | 1 | Beauty + diffuse/spec/emission/AO compositing |
| 59 | Depth-of-Field Mask | Render | 1 | Depth pass → CoC alpha mask |
| 60 | Depth Warp | Geometry | 1 | Horizontal parallax warp from depth |
| 61 | Normal → Curvature | Geometry | 1 | Curvature mask from a tangent-space normal pass |
| 62 | Position Pass Splitter | Geometry | 1 | Split position pass into X/Y/Z masks |
| 63 | Grain Match | Plate | 1 | Extract grain from reference plate, re-apply to target |
| 64 | Plate Stabilizer | Plate | 1 | ORB+RANSAC affine (cv2) → FFT translation fallback |
| 65 | Clean Plate Extractor | Plate | 1 | Median across batch with optional mask exclusion |
| 66 | Difference Matte | Plate | 1 | L1/L2 image diff → MASK with threshold + softness |
| 67 | Metadata Writer | Metadata | 1 | Write/merge JSON sidecars next to outputs |
| 68 | Frame Range Router | Metadata | 1 | Slice IMAGE/MASK batches `[start:end:step]` |
| 69 | Shot Metadata Reader | Metadata | 1 | Read shot.json (show / shot / task / frame in/out / fps) |
| 70 | Folder Version Incrementer (layout show / shot / task / version) | Output | 1 | `<root>/<show>/<shot>/<task>/v###/` with atomic reservation (was Batch Version Manager) |
| 71 | Temporal Consistency Checker | Diagnostics | 1 | Flicker score (mask_iou / pixel_diff / Farneback flow) |
| 72 | Model Metadata Extractor | Diagnostics | 1 | Inspect safetensors/checkpoints **without unpickling** |

**VRAM Tiers:** 1 = pure tensor math (CPU/GPU, no models), 2 = loads a model (~1–4 GB), 3 = loads multiple models

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
[Load Image]  →  [Spline Mask Editor]
                    spline_type: catmull_rom
                    closed: true
                    feather_radius: 2.0
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

## Automation: schedules, watch folders, webhooks

Two panels run workflows without you pressing Run. Open them from the OmniBar (**⏰ Schedule**, **👁 Watch**) or from
the command palette: **C2C: Scheduled Queue & Webhooks** and **C2C: Watch Folder**. The palette commands also work in
Lite mode, which has no OmniBar.

Every panel takes either **Current workflow** (the graph on screen, including its subgraphs) or an **API workflow
file** (Workflow › Export (API)). A normal saved workflow (nodes and links) cannot be queued without the browser, so
it is refused with that explanation. Runs are queued on this ComfyUI's own address and port, through its normal
`/prompt` check, so a broken workflow shows the same error you would get from Run.

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
- **C2C Multi-GPU Split** splits each model call over the GPUs of one machine, on Windows and Linux (no NCCL
  needed). Each GPU holds a full copy of the model, made by ComfyUI's own multi-GPU loader. Put it last in the model
  chain. With one GPU the model passes through unchanged.
  - **Any model:** the call's batch items go to different GPUs: cond and uncond with CFG, an image batch, the tiles
    from Tile Split. Measured with Flux: same result as one GPU (float32 rounding).
  - **Wan 2.1 / 2.2 and MiniMax H3 (text-to-video verified):** the tokens of one item are split too, and attention is
    exchanged between the GPUs (Ulysses all-to-all, Ring, or both). The result is the same as on one GPU:
    bit-identical for Ulysses at 2 and 4 GPUs, and within the precision's own rounding otherwise.
  - **Speed-up has not been measured yet** (that needs a machine with two or more GPUs).

With Wan or MiniMax H3, the batch items go to GPU groups first (no exchange needed) and each group splits the
tokens: on 8 GPUs with CFG that is cond and uncond on 4 GPUs each. You do not need ComfyUI's own **MultiGPU CFG
Split** with this node: it splits the batch itself, over all GPUs instead of one GPU per cond.
`C2C_DISTRIBUTED=off` switches the nodes off. `C2C_DISTRIBUTED_SIMULATE=2` runs two ranks on the one GPU (or
CPU) you have, to check a setup before renting GPUs. It is slower, never faster.

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

This umbrella pack aggregates nodes across four sub-namespaces — Folder Incrementer (`_FOLDER_MAPPINGS`), MEC tools (`_MEC_MAPPINGS` covering masking, SAM/ViT-Matte segmentation, inpaint crop/stitch/paste-back, spline editors, motion mask trackers, draw shapes, VAE merge/inspect, batch version manager, temporal consistency, model metadata, image comparer, parameter history, semantic segment, luminance keyer, mask failure explainer, temporal anchor, SAM multi-mask picker), Paint nodes (`_PAINT_MAPPINGS`), and MA nodes (`_MA_MAPPINGS`). The nodes are model-agnostic — they speak ComfyUI's native `IMAGE`, `MASK`, `LATENT`, `VAE`, and `MODEL` types — so they slot into any sampler graph regardless of which diffusion family produces the pixels.

| Model family | Recommended bridges |
|---|---|
| **Flux** | Use `InpaintCropProMEC` + `InpaintStitchProMEC` + `InpaintPasteBackMEC` to confine high-resolution Flux refinement to a tracked region. `MaskOpsMEC` (Mask + Matting, backend SAM + ViTMatte, with its own trimap settings) produces the high-quality alpha that Flux-driven matte refinement needs. `FolderIncrementer` keeps versioned outputs across sweeps. `VAEMergeMEC` and **Probe (C2C)** with the latent wired (or **VAE Inspect** in compare mode) are useful when comparing Flux VAE variants. |
| **Qwen-Image** | Identical inpaint and matte pattern as Flux. `MaskEditMEC` (shift / grow / feather) and `MaskTrackerMEC` (propagate one mask over a sequence) are useful when reusing a single Qwen-Image generated mask across a small sequence. `VideoComparerC2C` for A/B sweeps over guidance / sampler choices (images too). |
| **Wan 2.x (video)** | `MaskTrackerMEC` (motion tracking, propagation, key-frame anchors and a temporal-consistency check in one node) and `MaskOpsMEC` (backend SeC + MatAnyone2) are designed for the video domain — keep masks coherent across Wan-Animate frames, anchor a key-frame mask, and audit temporal jitter. `MaskFailureExplainerMEC` flags problem frames before re-rendering. Pair with `ComfyUI-WanAnimatePreprocessV2` (human) or `ComfyUI-WanAnimalPreprocess` (animal) for the pose / face conditioning. |
| **Z-Image** | Same inpaint and refine bridges as Flux/Qwen. The spline-mask, draw-shape, and points-mask editors are useful when authoring a custom region for Z-Image's refinement passes. |
| **GLM-Image** | The IMAGE outputs feed cleanly into the GLM-Image I2I path (via the sampler's optional `image` input + `denoise_strength`). `MaskOpsMEC` (backend remove background) + `SAMMaskGeneratorMEC` provide the alpha pre-stage. |
| **ERNIE-VL** | Not applicable for sampling (ERNIE-VL is a multimodal LLM). The `ModelMetadataExtractorMEC` and `ParameterHistoryMEC` nodes can still be useful for cataloguing assets that ERNIE-VL has captioned. |

Two general patterns recur across all sampler families:

1. **Inpaint trio** — `InpaintCropProMEC` → diffusion sampler (any family) → `InpaintPasteBackMEC` (or `InpaintStitchProMEC` for soft seams). Allows full-resolution refinement at a fraction of the VRAM, regardless of which model is loaded.
2. **Versioned saves** — wire the Folder Version Incrementer's `version_string` (layout show / shot / task / version for a pipeline tree) into the sampler's `Save Image.filename_prefix` so every queued prompt produces a clean, ordered file series.

ComfyUI's own reroutes (and the SAM / ViT-Matte loaders) keep the graph readable as it scales from a single Flux T2I bench into a multi-model Wan + Qwen-refine + Flux-detail composite.

---

<p align="center">
  Made with ❤️ for the ComfyUI community
</p>


---

## Full Node & Parameter Reference

A complete, auto-generated reference covering every node, return type and widget exposed by this pack lives in [NODE_REFERENCE.md](./NODE_REFERENCE.md). It is regenerated whenever `INPUT_TYPES` change.

