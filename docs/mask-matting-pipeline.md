# Mask + Matting

One node cuts a subject out of an image or a clip: it finds the subject (the segmenter), builds a soft alpha edge
(the matter) and finishes the matte (holdout / core masks, edge mode, despill, light wrap). A viewer on the node lets
you find and pick the subject with the mouse, preview a few frames and render the shot. Three hidden helper nodes run
the viewer's small jobs; you never add them yourself.

| Node | Use it to |
|---|---|
| [Mask + Matting](#mask--matting-maskopsmec) (`MaskOpsMEC`) | Make a mask and a soft alpha from points, a box, words, a mask or the viewer's pick |
| [Viewer helpers](#viewer-helper-nodes-c2cviewerdetect-c2cviewerplateproxy-c2cviewerresult) (`C2CViewerDetect`, `C2CViewerPlateProxy`, `C2CViewerResult`) | Nothing by hand: the viewer creates them |

## Mask + Matting vs Mask Refiner

- **Mask + Matting** makes the matte: it runs a model (SAM 3 / 3.1, BiRefNet, RMBG, SeC, ...) to find the subject,
  then a matter (ViTMatte, RVM, MatAnyone, ...) for the edge.
- **[Mask Refiner](matting-refinement.md)** (`MaskRefineMEC`) cleans a mask you already have: hole fill, morphology,
  guided filter, DenseCRF, edge snap, feather. No model weights.
- Chain them when you need both: Mask + Matting's `alpha` into Mask Refiner's `mask`. Mask + Matting's own
  `post_refine` (guided filter / DenseCRF) is a smaller version of the same idea and runs on the cascade pipeline only.

---

## Mask + Matting (`MaskOpsMEC`)

**Menu:** 🐺 C2C/🧰 Core/Pipeline · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#maskopsmec)

Needs model weights for its model backends (a SAM 3 / 3.1 checkpoint; ViTMatte needs the `transformers` package).
With a model missing, a labelled model-free method runs instead (GrabCut + optical flow, guided filter) and `info`
says `"approximate": true`; the C2C setting **Model-free fallbacks** (Settings → C2C → Models & Downloads, env
`C2C_MODEL_FALLBACK`) turns that off, and the node then stops with an error. Some weights are NON-COMMERCIAL: `rmbg` (Bria licences), `locate_anything` (NVIDIA
License, research only), `matanyone` (S-Lab License 1.0). SAM 3 is under Meta's SAM License; `vitmatte` is
Apache-2.0.

### What it is for

Cutting a person, an animal or an object out of a still or a video, with a soft edge for compositing. It replaces a
chain of loader, segmenter, trimap and matting nodes: the old `MaskMattingMEC`, `SAMViTMattePipelineMEC`,
`SeCMatAnyonePipelineMEC` and `BackgroundRemoverMEC` were merged into it, and saved workflows load as Mask + Matting
(see [MIGRATION.md](MIGRATION.md)).

### When to pick it

- You need a soft alpha (hair, fur, motion blur), not only a hard mask: pick this node.
- You already have a mask and only want cleaner edges: [Mask Refiner](matting-refinement.md).
- You want a quick SAM mask from clicks or words on a still, without matting: SAM Mask Generator
  ([sam-segmentation.md](sam-segmentation.md)).
- You want to grow, shrink, key, combine or grade a mask: Mask Tools ([mask-editing.md](mask-editing.md)).
- You want to move, smooth or check a mask over time: Mask Track ([video-temporal-bbox.md](video-temporal-bbox.md)).

### Minimal wiring

With the viewer (no prompt nodes needed):

```
Load Video / Load Image ──image──► Mask + Matting ──alpha──► (composite, inpaint, Save)
                                        │
                     viewer on the node: Load plate → Find / click → Preview → Render full shot
```

With wired prompts instead of the viewer:

```
Load Image ──► Image Mask Editor (C2C), mode points + box
                  positive_coords ──► Mask + Matting.positive_coords
                  negative_coords ──► Mask + Matting.negative_coords
Load Image ──────────────────────────► Mask + Matting.image ──alpha──► ...
```

A box (`pos_bbox`, `[x0,y0,x1,y1]`) or a `text_prompt` (for example `person`) works too. A clip (more than one frame in `image`) is tracked from the frame
in `frame_annotation`.

### The viewer

The node shows a viewer panel. It works in three steps shown at its top: **Plate → Subject → Render** (with
`backend` = `remove background` there is no Subject step: that backend takes no prompt).

1. **Load plate** runs the nodes that feed `image` and shows the frames as small proxies (at most 960 px, at most
   240 frames; a longer clip shows every k-th frame). Scrub with the slider, ◀ / ▶ or the arrow keys, or type a
   frame number.
2. **Find**: type words, comma-separated (`person, dog`), and press Find (or Enter). Every match on the current frame
   is tinted. Then click:
   - **click** = the subject (the other found objects are excluded);
   - **Shift-click** = add a part;
   - **Alt-click** = cut a part away;
   - **Ctrl-click** = remove the nearest marker.
   A click with nothing found asks the detector about that point. **Clear** forgets the subject and every marker.
   The drop-down at the top picks the SAM checkpoint for Find and clicks, strongest first (SAM 3.1 Multiplex, SAM
   3.1, SAM 3) from `models/sam3.1/`, `models/sam3/` and a SAM 3 file in `models/checkpoints/`. Find runs on
   ComfyUI's own SAM 3 (ComfyUI 0.39 or newer). Nothing is downloaded here.
3. **Preview N frames** renders a short chunk starting at the pick frame (the number box, 17 by default), then
   **Render full shot** renders everything.

The **Result** tab shows the render: **Comp** (the cut-out on the ground), **Alpha**, **Checker** and **Overlay**
(the matte tinted on the plate). Drag on the frame to wipe between the plate and the result. Three QC chips grade the
render green / amber / red:

| Chip | What it measures |
|---|---|
| Coverage | Share of the selection (the segmenter's mask) that the matte covers. Low = the matte lost part of the subject |
| Edge detail | Edge sharpness of the matte compared with the coarse mask. A smeared edge drops toward 0; hair strands raise it |
| Flicker | How much the matte edge changes from frame to frame after following the motion (optical flow on the plate) |

The thresholds were measured on synthetic clips with a known matte ([design/matte_viewer.md](design/matte_viewer.md));
the chips' tooltips say so. They are not a claim about your footage.

What gets saved: your selection is the `picks` input (JSON: frame, subject, excluded objects, added and cut parts).
When `picks` is set it replaces the prompt inputs of the chosen backend; `info.viewer.replaced_inputs` names the
wired inputs it ignored. A pick after frame 0 with `tracking_direction` = `forward` is run `bidirectional` so the
frames before the pick are tracked too (the info notes say so). Renders driven by the viewer keep the tracking and
edge stages in a cache, so changing only a finishing setting (despill, edge mode, light wrap ...) and rendering again
does not re-run SAM. The other controls sit behind **All settings ▸**.

Turn the viewer off with the C2C setting **Matte viewer** (Settings → C2C → Masks) or the environment variable
`C2C_VIEWER=0`. The node then shows its plain widgets (`picks` as a text box); a saved selection still applies.

### Live matte while running

While the node works through a clip, the viewer shows the matte of the newest finished frame and a count such as
"Segmentation 41 / 120" or "Matting ...". At most two updates a second, one small frame (at most 640 px). The normal
result replaces it at the end. It works in subgraphs too.

- RVM, MatAnyone and BGMattingV2 show the frame count only, without a picture.
- Nothing is shown for SAM 3 clip tracking on ComfyUI's own SAM 3 in the cascade pipeline, nor for the
  `SAM + ViTMatte`, `SeC + MatAnyone2` and `remove background` backends.
- Turn it off with the C2C setting **Live matte while running** (Settings → C2C → Masks) or
  `C2C_MASK_LIVE_PREVIEW=0`. It is also off when the viewer is off.

### Pipeline: `onyx` or `cascade (legacy)`

`pipeline` decides how `backend` = `segment + matte` works. A new node starts on `onyx`.

| | `onyx` | `cascade (legacy)` |
|---|---|---|
| Segmenter | SAM 3.1 video session (on ComfyUI's own SAM 3 when the SAM 3.1 library is not installed); `segmenter` is ignored and hidden | The `segmenter` you pick, or `auto` / `auto_best` routing |
| Matter | `matter` (`auto` = ViTMatte, else the guided filter), tiled (`matte_tile`) | `matter`, plus the trimap controls |
| Edge band | Scales with the resolution (`band_scale`) | `trimap_dilate` / `trimap_erode` / `edge_radius`, or the advanced trimap |
| Extras | `temporal_stabilise` (3-frame median inside the edge band); several objects through `scene_prompts` | `auto_quality`, luma pre-key, `tta_flip`, `multiscale`, `post_refine`, `robust_propagation` |

Measured on CPU, 20 frames, against an independent subject mask: `onyx` on ComfyUI's own SAM 3 reached 0.985 IoU on
the last frame; the model-free fallback 0.778; the cascade 0.983 (commit 8c2833e). One clip, not a general result.

On `onyx`, the prompt sockets (`positive_coords`, `negative_coords`, `pos_bbox` or `normal_bbox`, `neg_bbox`,
`text_prompt`, and `external_mask` as a mask prompt) become object 1 on the frame `frame_annotation`. For more objects
(up to 16) write `scene_prompts` JSON (version 1). Saved workflows from the old `MaskMattingMEC` load pinned to
`cascade (legacy)`, so they keep their old path.

### Backends (`backend`)

| `backend` | What runs | Needs |
|---|---|---|
| `segment + matte` | This node's `pipeline`, `segmenter` and `matter` | The models you pick |
| `SAM + ViTMatte` | The former SAM + ViTMatte node's code: SAM, then ViTMatte or a guided filter on the edges. Reads the points on every frame (it does not track) | `sam_model` from SAM Model Loader |
| `SeC + MatAnyone2` | The former SeC + MatAnyone2 node's code: SeC / SAM segmentation, then MatAnyone2 (video) or ViTMatte | SeC or SAM weights; MatAnyone (non-commercial) |
| `remove background` | RMBG-2.0 or BiRefNet, no prompt | Their weights (RMBG is non-commercial) |

Each control's tooltip names the backend that reads it; controls a backend does not read are hidden.

### Key controls

- `pipeline`, `backend`, `segmenter`, `matter`: see above. Entries tagged `[missing-deps]` need an extra Python
  package first.
- `auto_download` (Allow Download, off by default): fetch a missing weight from Hugging Face / torch.hub into its
  `models/` subfolder. Off: installed copies are used wherever they are.
- `edge_mode`: `soft` (as the matter made it), `feather`, `hard` (binary cut at `edge_threshold`) or `hard+feather`.
- `holdout_mask` forces alpha 0, `core_mask` forces alpha 1 (rigs, boom mics, opaque interiors).
- `despill` and `lightwrap_strength` for compositing over a new background.
- `split_masks`: off skips the edge / inside / outside masks (saves memory on long clips).
- `attention` and `offload` have no effect: no backend reads them, so they are folded away. Their saved values are
  kept.

The full list with tooltips is in [NODE_REFERENCE](../NODE_REFERENCE.md#maskopsmec).

### Reading the outputs

| Output | What it is |
|---|---|
| `mask` | The coarse mask from the segmenter |
| `alpha` | The final matte: use this one |
| `preview`, `foreground` | The image times the alpha (`foreground` is despilled when despill is on) |
| `trimap` | The trimap the matter used (0 / 0.5 / 1) |
| `bbox`, `bbox_json` | The box around the alpha |
| `score` | A quality score in 0..1 from boundary, coherence, size and smoothness |
| `despilled`, `lightwrap_rgba` | Compositing layers |
| `edge_mask`, `inside_mask`, `outside_mask` | The soft edge band, the solid inside, the solid outside |
| `luma_key_mask` | The luma pre-key (empty unless `enable_luma_key` is on) |
| `problem_regions`, `severity`, `suggested_method` | The failure check (`enable_diagnose`) |
| `object_alphas` | One alpha per object, stacked object by object |

`info` is JSON. Useful fields:
- `pipeline`, `backend`, `segmenter`, `matter`: what actually ran;
- `auto_route.reasons`: why `auto` / `auto_best` picked a backend;
- `approximate` and `model_free`: a model was missing and the model-free method ran (`wanted`, `why`);
- `text_grounding`: words were turned into a box by another model (for example LocateAnything);
- `diagnose.severity`, `diagnose.suggested_method`, `diagnose.explanation`;
- `mask_inputs`: a holdout / core mask of another size was resized to the matte;
- `background_plate`: how the BGMattingV2 clean plate was used;
- `onyx` and `objects`: the onyx pipeline's details per object;
- `viewer`: on viewer renders, the selection, the frames, `replaced_inputs`, notes and `empty_frames`.

### Common mistakes

| What you see | Why | What to do |
|---|---|---|
| "Model '...' is not installed and downloading is off" | The weight is not on disk and Model-free fallbacks is off | Put the file in the folder the message names, or tick `auto_download` |
| `info` says `"approximate": true` | A model was missing; the model-free method ran | Install the model (see `model_free.why`) |
| "SAM 3.1 is not installed ..." | No SAM 3.1 library, ComfyUI older than 0.39, and Model-free fallbacks is off | Update ComfyUI to 0.39+, or turn Model-free fallbacks on (points / a box / a mask then work, approximately) |
| "No prompts found — wire points, bbox, text, or external_mask ..." (`onyx`) | Nothing tells the node what to cut | Pick in the viewer, wire a prompt, or write `scene_prompts` |
| "A bbox input is connected but no valid box was received" | A box socket is wired but the editor drew no box | Draw a box upstream, or unlink the socket |
| "The plate changed shape since the selection was made" | The image size changed after picking | Pick again in the viewer, or Clear |
| Wired prompts are ignored | `picks` is set and replaces them (`info.viewer.replaced_inputs`) | Press Clear in the viewer |
| Frames before the pick are empty with `SeC + MatAnyone2` | SeC starts its clip on the pick frame | Pick on frame 0, or use `segment + matte` |
| `SAM + ViTMatte` returns one frame | `sam_batch_mode` is off (first frame only) | Turn `sam_batch_mode` on |
| BGMattingV2 matte is poor | No `background_plate` wired: a plate is synthesised from the coarse mask | Wire a clean plate of the same shot |
| `post_refine` changes nothing | It runs on `cascade (legacy)` only | Switch the pipeline, or chain [Mask Refiner](matting-refinement.md) |
| `crf` does nothing | DenseCRF needs the `pydensecrf` package | Install it, or use `guided` |
| "scene_prompts version must be 1" | Wrong JSON | Start the object list with `"version": 1` |

---

## Viewer helper nodes (`C2CViewerDetect`, `C2CViewerPlateProxy`, `C2CViewerResult`)

**Menu:** hidden (shown only with ComfyUI's Dev Mode, under 🐺 C2C/🧰 Core/Viewer helpers) · **Parameters:**
[Detect](../NODE_REFERENCE.md#c2cviewerdetect), [Plate Proxy](../NODE_REFERENCE.md#c2cviewerplateproxy),
[Result](../NODE_REFERENCE.md#c2cviewerresult)

The Mask + Matting viewer creates these nodes in small jobs it puts at the front of ComfyUI's queue. Do not add them
by hand.

| Node | Menu name | What the viewer uses it for |
|---|---|---|
| `C2CViewerPlateProxy` | Viewer helper: Plate Proxy | **Load plate**: the plate as proxy frames (at most 960 px, at most 240 frames) |
| `C2CViewerDetect` | Viewer helper: Detect | **Find** and clicks: one mask per object found on one frame (ComfyUI's own SAM 3 / 3.1). With no SAM checkpoint installed its list shows `MISSING: <file>` and the run says where to put which file |
| `C2CViewerResult` | Viewer helper: Result | **Preview** / **Render**: the rendered alpha and the plate as proxies, and the QC numbers |

They have no outputs to wire, and they are never added to your graph: the viewer sends them in prompts of its own.
With the viewer off (`C2C_VIEWER=0`) they are not used.
