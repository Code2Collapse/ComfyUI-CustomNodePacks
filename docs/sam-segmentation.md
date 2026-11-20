# SAM & Segmentation

Nodes that find an object, a face part or a piece of clothing in an image and give you its mask. Segment (C2C) runs on
ComfyUI's own SAM 3 / 3.1; the SAM Mask Generator and the Multi-Mask Picker run SAM through this pack's own model
loading; Semantic Segment parses faces and clothes; LocateAnything finds things from a written description. For soft,
matted edges (hair, motion blur) feed the result to [Mask + Matting](mask-matting-pipeline.md).

| Node | Use it to |
|---|---|
| [Segment (C2C)](#segment-c2c-segmentc2c) | Find objects by words, clicks or boxes on core SAM 3 / 3.1, one mask per object, and track them through a clip |
| [SAM Model Loader](#sam-model-loader--sam21--sam3-sammodelloadermec) | Load a SAM 2.1 checkpoint for the two SAM nodes below |
| [SAM Mask Generator](#sam-mask-generator--points--bbox--text-sammaskgeneratormec) | One mask from points, boxes, a spline or a GroundingDINO text prompt, with edge refinement |
| [SAM Multi-Mask Picker](#sam-multi-mask-picker--3-candidates--scores-sammultimaskpickermec) | See SAM's three candidate masks and click the one you want |
| [Semantic Segment](#semantic-segment--face--clothes-parsing-semanticsegmentmec) | Face parts (skin, eyes, lips ...) or clothes (upper_clothes, pants ...) by class name |
| [LocateAnything Grounding](#locateanything-grounding-mec-locateanythinggroundingmec) | Boxes from a description such as "the person on the left" (non-commercial model) |
| [LocateAnything → SAM Prompt](#locateanything--sam-prompt-mec-locateanythingtosammec) | Turn those boxes into a rectangle mask for a SAM node |

## Which one to pick

| You want | Pick | Why |
|---|---|---|
| Several objects by a word ("person:3, dog"), or a whole clip tracked | **Segment (C2C)** | Words need no extra model; one mask per object; tracks a clip with one tracking state |
| One object from clicks, a box, a spline or several boxes, with a soft edge | **SAM Mask Generator** | Takes every kind of prompt, has `edge_refine`; works on the first frame only |
| To see SAM's three guesses and choose by eye | **SAM Multi-Mask Picker** | Shows three thumbnails with scores; first frame only |
| A phrase that names one of several similar things ("the red car on the left") | **LocateAnything Grounding** | Reads a full description; needs a 3B model under a non-commercial licence |
| Face parts or clothes by class | **Semantic Segment**, or Segment (C2C) with `target` = `face parts` / `clothes` | Segment gives one mask per class; Semantic Segment gives one combined mask |

Removed nodes:
- **Unified Segmentation** (`UnifiedSegmentationNode`) was removed from the pack (commit 71b56e8) and has no
  automatic replacement. Use Segment (C2C) for SAM 3. [Mask + Matting](mask-matting-pipeline.md) also segments
  with SAM 2.1 / SAM 3 / SAM 3.1 (its `segmenter`) and with SeC (`backend` = `SeC + MatAnyone2`).
- **Background Remover** (`BackgroundRemoverMEC`), **SAM + ViTMatte Pipeline** (`SAMViTMattePipelineMEC`) and
  **SeC + MatAnyone Pipeline** (`SeCMatAnyonePipelineMEC`) were merged into Mask + Matting (`MaskOpsMEC`), `backend`
  = `remove background` / `SAM + ViTMatte` / `SeC + MatAnyone2`; saved workflows load as that node. See
  [mask-matting-pipeline.md](mask-matting-pipeline.md).

---

## Segment (C2C) (`SegmentC2C`)
**Menu:** 🐺 C2C/🧰 Core/Segmentation · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#segmentc2c)
Needs a SAM 3 / 3.1 checkpoint for ComfyUI's own SAM 3. Words, a box and clip tracking were run with the real
SAM 3.1 weights on CPU (commits c008aee, a48c567); clicks were not named in those checks. `refine` is EXPERIMENTAL and off by default. The `face parts` /
`clothes` targets are NOT VERIFIED with the real parsing model (commit 509fca6).

### What it is for

Find objects and get one mask per object, their union and their boxes. Name them in words (`person`,
`person:3, dog`), or click and draw boxes. On a clip it finds the objects on one frame and follows them through every
frame. With `target` = `face parts` or `clothes` it returns one mask per named class instead.

### Minimal wiring

```
Load Image / Load Video ──► image ──► Segment (C2C) ──► mask   (union of all objects, per frame)
                                                    ──► masks  (one per object, first frame)
                                                    ──► bboxes
                                                    ──► info
```

The SAM 3 model is picked in the node's own `model` list; no loader node is needed. It is the same file the
Mask + Matting viewer uses, and both share one copy in memory.

### Key controls

- `prompt` (objects): comma-separated words. A plain word returns the strongest match only; write `word:N` for up to
  N objects of that kind (`person:5`). When words are given, `picks` is ignored.
- `picks`: clicks and boxes in pixels, `{"positive": [[x, y]], "negative": [[x, y]], "boxes": [[x0, y0, x1, y1]]}`.
  Each box gives one object (the clicks go with every box); without boxes the clicks give one object.
- `max_objects`: keep at most this many objects per frame, strongest first.
- `video`: `track whole clip` finds the objects on `seed_frame` and follows them forward to the end and backward to
  frame 0, so they keep their order. `per frame` segments each frame on its own. Pick a `seed_frame` where the
  objects are clearly visible.
- `refine` (EXPERIMENTAL): re-decodes each object's mask with SAM from prompts taken from the mask itself (after the
  SAMRefiner paper). Measured on three synthetic shapes with exact truth (commit 2b023c8): boundary F 0.983 → 0.995,
  IoU 0.995 → 0.991, 1.7–2× the time. Mixed, so it ships off.
- `target`: `objects` (SAM 3), `face parts` or `clothes`. For the last two, `prompt` lists class names (`skin, hair,
  l_eye, mouth`; empty = `skin, hair` / `upper_clothes`) and `model` is not used. The classes are the ones listed
  under [Semantic Segment](#semantic-segment--face--clothes-parsing-semanticsegmentmec).
- `allow_download` (face parts / clothes): allows downloading the parsing model when it is missing.

### Reading the outputs

- `mask`: union of every object, one mask per frame.
- `masks` and `bboxes`: one mask and one `[x0, y0, x1, y1]` box per object **of frame 0**, strongest first.
- `info` (JSON): `model`, `target`, `prompt`, counts of `clicks` / `negative_clicks` / `boxes`, `frames`, `tracked`,
  `objects_first_frame`, `per_frame` (label, score, box and area of each object on each frame), `edge_stability`
  (mean / max of how much the union's edge moves between frames, for a clip), `refine` (what each refine pass
  changed), `seconds` and `notes`. Face parts / clothes report `parser`, `classes` and `parser_info` instead.

### Common mistakes

- **"say what to segment"**: `prompt` and `picks` are both empty.
- **"No SAM 3 / 3.1 checkpoint is installed"**: the `model` list shows `MISSING: sam3.1_multiplex_fp16.safetensors`.
  Put that file in the folder the message names (`models/sam3.1/`, `models/sam3/`, or `models/checkpoints/` with a
  name starting `sam3`), then press R in ComfyUI to refresh the list. The node does not download it.
- **Only one person found.** Write `person:N`.
- **Clicks seem ignored.** Words win; clear `prompt`. `info.notes` says "Words were given, so clicks and boxes were not
  used."
- **Objects swap order between frames.** `video` is `per frame`, or the model cannot track (the note says so).
- **`refine` did nothing.** It needs ComfyUI's own SAM 3 (core 0.39+); otherwise it is skipped and the notes say so.
- **"unknown class name(s)"** (face parts / clothes): the message lists the valid names.
- **Face parts / clothes stop with a "not installed" message.** The parser is missing and `allow_download` is off.
  Without the model only `skin` can be found (by colour, approximate); every other class needs the model.
- **Slow on CPU.** The commit's 3-frame CPU tracking test took about 145 s per run (a48c567).

---

## SAM Model Loader — SAM2.1 / SAM3 (`SAMModelLoaderMEC`)
**Menu:** 🐺 C2C/🧰 Core/SAM · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#sammodelloadermec)
Needs the `sam2` Python package (`pip install git+https://github.com/facebookresearch/sam2.git`) and a checkpoint.

### What it is for

Loads a SAM 2.1 (or SAM 3) checkpoint once for the [SAM Mask Generator](#sam-mask-generator--points--bbox--text-sammaskgeneratormec)
and the [Multi-Mask Picker](#sam-multi-mask-picker--3-candidates--scores-sammultimaskpickermec). Segment (C2C) does not
use it.

### Key controls

- `model_name`: files from `models/sams/`, `models/sam2/` and `models/sam3/`, plus `[download]` entries for the four
  SAM 2.1 Hiera sizes (tiny, small, base_plus, large) from `facebook/sam2.1-hiera-*` on Hugging Face.
- `allow_download`: a `[download]` entry is fetched into `models/sam2/` only when this is on.
- `model_type`: `auto` reads the family from the file name (`sam3` in the name → `sam3`, otherwise `sam2.1`).
- `device` (`auto` = ComfyUI's device), `dtype`, and `offload_to_cpu` (keeps the model on the CPU between runs:
  less VRAM, slower).

### Reading the outputs

`sam_model` carries the model, its type, device and a `model_status` that the SAM nodes copy into their `info`.

### Common mistakes

- **"Only 'sam2.1' and 'sam3' are supported"**: SAM 1 (vit_h / l / b) and SAM 2.0 checkpoints were removed from this
  loader.
- **No model installed.** With no usable checkpoint (none placed, or a `[download]` entry with `allow_download` off)
  the loader hands out a model-free stand-in: the SAM nodes then make an approximate GrabCut mask and their `info`
  says `"approximate": true`. To get an error instead, turn off C2C › Models & Downloads › **Model-free fallbacks**
  (or set the environment variable `C2C_MODEL_FALLBACK=off`).

---

## SAM Mask Generator — Points + BBox + Text (`SAMMaskGeneratorMEC`)
**Menu:** 🐺 C2C/🧰 Core/SAM · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#sammaskgeneratormec)
Needs a model from the SAM Model Loader. Text prompts need a GroundingDINO model as well.

### What it is for

One mask for one target in a still, from any mix of prompts: points (`points_json`), a box (`bbox_json` or the `bbox`
input), several boxes (`bboxes_json`, segmented one by one and joined), a spline outline (`spline_json`), a text
prompt through GroundingDINO, or a starting mask (`existing_mask`).

### Minimal wiring

```
SAM Model Loader ──► sam_model ─┐
Load Image ────────► image ─────┴─► SAM Mask Generator ──► mask ──► (Mask + Matting / inpaint)
                                                       ──► info
```

Only the first frame of `image` is used.

### Key controls

- `points_json`: `[{"x": 100, "y": 200, "label": 1}]`; label 1 = keep, 0 = exclude.
- `text_prompt` + `grounding_model` (`groundingdino_swint_ogc` or `groundingdino_swinb_cogcoor`): the text becomes a
  box. `negative_text_prompt` turns the boxes it finds into exclude points. `text_box_threshold` is the minimum box
  confidence, `text_threshold` how strongly a word must match. `allow_download` lets the grounding model download.
- `mask_index`: which of the three candidates to return, in the model's own order; 0 is not necessarily the best
  score (see the `score` output).
- `edge_refine` (default `guided`): `none` keeps SAM's hard mask; `guided` / `guided_strong` soften it along image
  edges with a guided filter (no model); `matte` uses ViTMatte or BiRefNet if one is installed and falls back to
  `guided` otherwise. `edge_radius` sets the filter radius.
- `refine_iterations`: feeds the mask back into SAM this many times. `auto_negative_points` samples exclude points
  just outside the mask.

### Reading the outputs

- `mask`: the selected mask (first frame). `all_masks`: every candidate.
- `detected_bbox`: the box of the selected mask, `[x, y, w, h]`. `score`: its SAM score.
- `info` (JSON): `model_type`, `model_status`, `num_masks`, `scores`, `selected_index`, `detected_bbox`,
  `refine_iterations`; `"approximate": true` when the loader's model-free stand-in made the mask. With several boxes:
  `union_of_boxes`, `best_score`, and `ignored_inputs` / `note` naming prompts that were not used.

### Common mistakes

- **"GroundingDINO found nothing for '...'"**: lower `text_box_threshold`, change the words, or add a point or a box.
- **The text prompt does nothing.** `grounding_model` is `none`.
- **Several boxes in `bboxes_json` ignore your points.** Each box is segmented on its own, so `points_json`,
  `bbox_json`, `bbox` and `apply_bbox_crop` are not used; `info` names them.
- **A video gives one mask.** This node segments the first frame only; use Segment (C2C) for a clip.
- **"No SAM model is installed, and the model-free stand-in needs a point, a box or a mask."** The stand-in cannot
  read text; install a SAM 2.1 model or give a point / box.

---

## SAM Multi-Mask Picker — 3 candidates + scores (`SamMultiMaskPickerMEC`)
**Menu:** 🐺 C2C/🧰 Core/SAM · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#sammultimaskpickermec)
Needs a SAM model: its own `model_name`, or one wired from the SAM Model Loader.

### What it is for

SAM returns three guesses for an ambiguous click (the shirt, the person, the person and the bag). This node shows the
three masks as thumbnails with their scores, and you pick one.

### Minimal wiring

```
Load Image ──► image ──► SAM Multi-Mask Picker ──► selected_mask ──► (next node)
(optional) SAM Model Loader ──► sam_model
```

Queue once to see the thumbnails. Click a thumbnail (or select the node and press 1 / 2 / 3); this sets
`selected_index`. Queue again to send the new choice downstream.

### Key controls

- `points_json` / `bbox_json` (or the `bbox` input): the prompts, as in the SAM Mask Generator. Only the first frame
  is used.
- `model_name`, `precision`: used when `sam_model` is not connected. `allow_download` lets `model_name` download into
  `models/sam2/`.

### Reading the outputs

- `selected_mask`, `all_masks` (the three candidates), `selected_index`.
- `scores`: a JSON list of the three scores in percent.
- `info` (JSON): `model_name`, `model_status`, `scores_pct`, `scores_raw`, `selected_index`, `image_size`,
  `num_points`, `has_bbox`, `precision`, `device`; `"approximate": true` when a model-free GrabCut mask was made
  because no model was installed.

---

## Semantic Segment — Face / Clothes Parsing (`SemanticSegmentMEC`)
**Menu:** 🐺 C2C/🧰 Core/Segmentation · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#semanticsegmentmec)
Needs a SegFormer parsing model (`jonathandinu/face-parsing` or `mattmdjaga/segformer_b2_clothes`, in
`models/segformer/`).

### What it is for

One combined mask for the classes you name, frame by frame, from a face or a clothes parsing model.

### Key controls

- `model_name`: `segformer_face` or `segformer_clothes`.
- `classes_csv`: comma-separated class names.
  - Face: `skin`, `l_brow`, `r_brow`, `l_eye`, `r_eye`, `eye_g`, `l_ear`, `r_ear`, `ear_r`, `nose`, `mouth`,
    `u_lip`, `l_lip`, `neck`, `necklace`, `cloth`, `hair`, `hat`.
  - Clothes: `hat`, `hair`, `sunglasses`, `upper_clothes`, `skirt`, `pants`, `dress`, `belt`, `left_shoe`,
    `right_shoe`, `face`, `left_leg`, `right_leg`, `left_arm`, `right_arm`, `bag`, `scarf`.
- `threshold`: how likely a pixel must be to belong to the classes. `invert` flips the mask.
- `allow_download`: download the model into `models/segformer/` if it is missing.

Frames with a long edge above 2048 px are parsed at 2048 px and the result is scaled back up.

### Reading the outputs

`info` (JSON): `model`, `model_status`, `family`, `matched_classes`, `frames`, `threshold`, `invert`. Without the model
it also says `"approximate": true`, `model_free` and `needs_the_model` (the classes that could not be made).

### Common mistakes

- **"no class matches ..."**: a typo or a class from the other model; the message lists the valid names.
- **"Without the model only 'skin' can be found (by colour)"**: the model is missing and `allow_download` is off,
  and no class you asked for can be made without it. When `skin` is among the classes, the node returns skin only
  (by colour, approximate) and `info.needs_the_model` lists the classes it left out.

---

## LocateAnything Grounding (MEC) (`LocateAnythingGroundingMEC`)
**Menu:** 🐺 C2C/🧰 Core/Segmentation · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#locateanythinggroundingmec)
Needs NVIDIA's LocateAnything-3B (about 7 GB) and the `transformers` package; a GPU in practice (on CPU it is very
slow). **Licence: NVIDIA License, non-commercial use only (academic / non-profit research).** No run with the real
weights is recorded in the commit history (NOT VERIFIED with real weights).

### What it is for

Finds things from a description (`the person on the left`, `red car`, `coffee cup`) and returns boxes. It is not
limited to a fixed list of classes.

### Minimal wiring

```
Load Image ──► image ──► LocateAnything Grounding ──► bboxes ──► LocateAnything → SAM Prompt ──► mask
               prompt: "the person on the left"   ──► annotated_image ──► Preview Image
```

### Key controls

- `prompt`: what to find, in words.
- `model_path`: the official repo `nvidia/LocateAnything-3B` or a folder inside a ComfyUI models folder. The model runs
  its own Python code when it loads, so any other repo or folder is refused unless you list it in the environment
  variable `C2C_TRUSTED_REMOTE_CODE`.
- `allow_download`: downloads the model into `models/LLM/LocateAnything-3B`. Off: an installed copy is used
  (`models/*/LocateAnything-3B` or the Hugging Face cache).
- `max_new_tokens`: raise it when a crowded image returns too few boxes.
- `confidence_threshold` has no effect (the model returns no scores); it stays so saved workflows load.

### Reading the outputs

- `bboxes`: every box in pixels, with its `label` and `frame`.
- `annotated_image`: the image with the boxes drawn on it.
- `raw_response`: the model's text answer (frames separated by `---`).

### Common mistakes

- **"model_path ... is not the official nvidia/LocateAnything-3B"** or **"... is a folder outside ComfyUI's models
  folders"**: see `model_path` above.
- **The model is not installed**: the node names the folder to put it in; there is no model-free way to read a text
  prompt.

## LocateAnything → SAM Prompt (MEC) (`LocateAnythingToSAMMEC`)
**Menu:** 🐺 C2C/🧰 Core/Segmentation · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#locateanythingtosammec)

Turns the `bboxes` from LocateAnything Grounding into filled rectangles, one mask per frame (a box marks only the
frame it was found in). `expand_ratio` grows each box by a fraction of its size on every side. The rectangle is a
prompt, not a final matte: wire `mask` into a node that takes a starting mask, such as the SAM Mask Generator's
`existing_mask`. Connect the same `image` the boxes were found in; it sets the mask size and frame count.
