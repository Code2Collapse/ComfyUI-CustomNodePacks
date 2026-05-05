# ComfyUI-CustomNodePacks — Documentation Index

> **All node docs live here.** The top-level [`README.md`](../README.md)
> is the project landing page; this folder is the deep reference manual.
>
> Every page documents **every parameter, every mode, every output**, with
> use-cases for image generation, video generation, VFX, and inpainting.

---

## Quick navigation

| If you want to… | Start with… |
|---|---|
| Cut out a subject from a single image | [`matting-refinement.md`](matting-refinement.md), [`mask-matting-pipeline.md`](mask-matting-pipeline.md), [`sam-segmentation.md`](sam-segmentation.md) |
| Cut a subject across a video clip | [`mask-matting-pipeline.md`](mask-matting-pipeline.md), [`video-temporal-bbox.md`](video-temporal-bbox.md) |
| Inpaint a region of an image | [`inpaint-suite.md`](inpaint-suite.md), [`paint-suite.md`](paint-suite.md) |
| Inpaint or remove an object across a **video** | [`propainter.md`](propainter.md) |
| Auto-fix faces with per-face prompts | [`face-fixer.md`](face-fixer.md) |
| Stabilise a handheld clip before inpainting | [`video-stabilizer.md`](video-stabilizer.md) |
| Draw, transform, or composite masks | [`mask-editing.md`](mask-editing.md) |
| Place points / boxes interactively on the canvas | [`utility-nodes.md`](utility-nodes.md) (Points Mask Editor) |
| Inspect, merge, or compare VAEs | [`vae-merge.md`](vae-merge.md) |
| Read / write OpenEXR with metadata | [`exr-io.md`](exr-io.md) |
| Convert colorspaces / apply LUTs | [`color-science.md`](color-science.md) |
| Composite render passes | [`render-pass.md`](render-pass.md) |
| Match grain / build clean plates | [`plate-tools.md`](plate-tools.md) |
| Scrub / trim / crop a video in-graph | [`video-frame-player.md`](video-frame-player.md) |

---

## Pack-by-pack reference

### 🎯 Segmentation, Matting & Mask Pipelines
| Doc | Key nodes | What it covers |
|---|---|---|
| [`sam-segmentation.md`](sam-segmentation.md) | SAM Model Loader, SAM Mask Generator, **SAM Multi-Mask Picker**, Unified Segmentation, Semantic Segment, Background Remover, SAM+ViTMatte, SeC+MatAnyone | Loading SAM 2.1 / SAM 3, prompting (points / boxes / text), iterative refinement, multi-candidate visual picking, end-to-end pipelines |
| [`matting-refinement.md`](matting-refinement.md) | Matting Node, ViTMatte Refiner, Trimap Generator, Luminance Keyer | 7 matting backends, 7 refinement methods, trimap construction, BT.709 luminance keying |
| [**`mask-matting-pipeline.md`**](mask-matting-pipeline.md) ★ new | **MaskMattingMEC** (combined) | Single-node pipeline: pick segmenter + matter + weights, point/box/text/video auto-mode, slot-based positive/negative coords, subject presets |

### 🖌️ Mask Editing, Drawing & Painting
| Doc | Key nodes | What it covers |
|---|---|---|
| [`mask-editing.md`](mask-editing.md) | Mask Transform XY, Draw Frame, **Draw Shape**, Composite Advanced, Mask Math, Spline Mask Editor, Batch Manager, Preview Overlay | Per-axis erode/expand/blur, 12 SDF shapes, 8 composite ops, 11 math ops, Bezier spline editor |
| [**`paint-suite.md`**](paint-suite.md) ★ new | **MEC Advanced Paint Canvas**, **MEC Context Inpainter** | Interactive brush widget, hardness/expansion/blur stages, color match, lightness rescue, differential diffusion |

### 🎬 Inpainting (Image & Video)
| Doc | Key nodes | What it covers |
|---|---|---|
| [`inpaint-suite.md`](inpaint-suite.md) | Inpaint Crop Pro, Inpaint Composite, Stitch Pro (legacy), Paste Back (legacy), Mask Prepare | Crop+stitch pipeline, 4 blend modes (gaussian / edge_aware / Laplacian-pyramid / FFT frequency), color match, video-stable crop, all 11 fill modes |
| [**`propainter.md`**](propainter.md) ★ new | **ProPainter Temporal**, **Stitch**, **Stitch Refine**, **Remove**, **Flow Refine** | Temporal video inpaint with bidirectional RAFT flow, RFC flow completion, sliding-window InpaintGenerator, model-free object/wire/plate removal |
| [**`face-fixer.md`**](face-fixer.md) ★ new | **MEC Face Fixer** | YOLO11 face detection → per-face crop → optional upscale → KSampler → smart blend with `[SEP]/[ASC]/[DSC]/[SKIP]` wildcard prompts |

### 📹 Video, Temporal Stability & BBox
| Doc | Key nodes | What it covers |
|---|---|---|
| [`video-temporal-bbox.md`](video-temporal-bbox.md) | Mask Propagate, Temporal Anchor, Motion Mask Tracker, BBox Create/From Mask/To Mask/Pad/Crop/Smooth | SDF interpolation, optical-flow / SAM2-video tracking, 6 BBox utilities |
| [**`video-stabilizer.md`**](video-stabilizer.md) ★ new | **Stabilizer Classic / Flow / Auto** | Sparse GFTT+LK (CPU) vs RAFT dense flow (GPU), framing modes, padding mask output, 4 presets |
| [`video-frame-player.md`](video-frame-player.md) | Video Frame Player | In-graph scrubber + drag-crop (8 handles, aspect lock) + trim + stride + lanczos resize/upscale |

### 🛠️ Utilities, VFX & Diagnostics
| Doc | Key nodes | What it covers |
|---|---|---|
| [`utility-nodes.md`](utility-nodes.md) | Points Mask Editor, Image Comparer, Mask Failure Explainer, Parameter History, Universal Reroute, Folder Incrementer (×3) | Interactive canvas (points/boxes/zoom/pan/undo), drag-slider compare, mask-failure diagnostics, SQLite parameter history, auto-versioned output |
| [`color-science.md`](color-science.md) | sRGB↔Linear, Rec.709↔ACEScg, LUT, Grade | Colorspace transforms, `.cube` LUT loader, exposure/WB/contrast |
| [`exr-io.md`](exr-io.md) | EXR Load, EXR Save | Multi-layer OpenEXR with imageio + TIFF fallback, metadata pass-through |
| [`render-pass.md`](render-pass.md) | Merge Passes, Depth→CoC | Beauty + AO/diffuse/spec/emission compositing, depth-of-field mask synthesis |
| [`plate-tools.md`](plate-tools.md) | Grain Match, Plate Stabilizer (ORB/FFT), Clean Plate, Difference Matte | Grain transplant, sparse plate stabilisation, multi-frame median clean plate |
| [`vae-merge.md`](vae-merge.md) | VAE Merge, Latent Inspector, Similarity Analyser, Block Inspector | 8 merge algorithms, per-block alpha, latent statistics |

---

## How to use these docs

Each page follows the same template:

1. **Overview** — what the node does in one sentence + when to use it
2. **Parameters** — every input with default, type, range, and effect
3. **Outputs** — every output with shape and downstream wiring
4. **Use cases** — concrete scenarios (image gen, video gen, VFX, inpaint)
5. **Recipes** — copy-paste graph snippets / parameter combos
6. **Troubleshooting** — common failure modes and fixes

If a parameter is missing from a doc, please open an issue at
<https://github.com/Code2Collapse/ComfyUI-CustomNodePacks/issues>.

---

## Image vs. Video helpers

The whole pack is built so you can wire the **same nodes** for single-image
and video workflows — temporal-aware nodes detect a `B > 1` batch and switch
modes automatically:

| Stage | Image-gen node | Video-gen equivalent |
|---|---|---|
| Segment subject | SAM Mask Generator / SAM+ViTMatte | **MaskMattingMEC** with SAM2.1 video / SeC + MatAnyone2 |
| Refine alpha edges | ViTMatte Refiner | Same node — works per-frame |
| Inpaint a region | Inpaint Crop Pro → KSampler → Inpaint Composite | Inpaint Crop Pro → **ProPainter Temporal** → Inpaint Composite |
| Remove an object | Crop → SD inpaint → Composite | **ProPainter Remove** (no SD model needed) |
| Stabilise jitter | n/a | **Video Stabilizer Auto / Flow / Classic** |
| Smooth mask between keyframes | n/a | **Temporal Anchor System** (SDF interpolation) |
| Lock crop bbox over time | `video_stable_crop=true` on Inpaint Crop Pro | same |
| Per-face detail pass | **MEC Face Fixer** | same — runs frame-by-frame |

---

## License & attribution

All MEC code is **Apache-2.0** with the strong attribution NOTICE
([`../NOTICE.md`](../NOTICE.md)). Vendored or downloaded model weights
follow their own licenses — see [`../THIRD_PARTY_LICENSES/`](../THIRD_PARTY_LICENSES/).
