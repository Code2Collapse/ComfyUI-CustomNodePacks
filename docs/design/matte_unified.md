# One Matte node - design (ledger L7.65, owner card P05)

Status: design, written before the code (owner order A7). Owner spec: `docs/evidence/L7.65/P05_owner_spec_2026-10-10.md`
in the workspace, section 1.

## 1. Goal and fixed decisions

One user-facing node for segmentation and matting. `MaskOpsMEC` ("Mask + Matting") keeps its id and is the survivor.
`SAMViTMattePipelineMEC`, `SeCMatAnyonePipelineMEC` and `BackgroundRemoverMEC` are removed from the node list and
migrate into it when a workflow loads (D0.14). Fixed by the owner and the consolidation rules:

- MaskOps keeps all 78 inputs (names, order, types, defaults) and all 18 outputs, in order. New inputs are appended
  (optional, with defaults that reproduce today's behaviour). One output is appended: `foreground`.
- A migrated node gives the same results as the old node: every mapped output is compared in a test.
- The code of the three old nodes is not rewritten. It stays in its modules (not registered) and MaskOps calls it.
- At most 10 controls on screen per backend at the defaults; the rest in sections. Hidden widgets keep their values
  and serialise in the same order.
- No new model backends in this task. Missing ones go to the backlog.

## 2. Inventory (measured from the code at d97cc13)

### 2.1 MaskOpsMEC (`nodes/mask_matting/node.py`)

- 78 inputs: `image` + 54 required widgets + 23 optional inputs (sockets `positive_coords`, `negative_coords`,
  `pos_bbox`, `neg_bbox`, `normal_bbox`, `text_prompt`, `external_mask`, `external_trimap`, `holdout_mask`,
  `core_mask`; widgets `pipeline`, `scene_prompts`, `band_scale`, `matte_tile`, `temporal_stabilise`,
  `matte_overlap`, `matte_tile_batch`).
- 18 outputs: `mask, alpha, preview, trimap, bbox, bbox_json, score, info, despilled, lightwrap_rgba, edge_mask,
  inside_mask, outside_mask, luma_key_mask, problem_regions, severity, suggested_method, object_alphas`.
- Two pipelines: `onyx` (SAM 3.1 video session + tiled ViTMatte, `_onyx_pipeline.py`) and `cascade (legacy)`
  (segmenter + matter registry, `auto_best` / `auto` routing).
- Segmenters registered (this machine): sam3, sam3.1, birefnet, rmbg, inspyrenet, dis, person-mask, grounding-dino,
  cutie, xmem, sec, videomama, locate_anything, classic (model-free GrabCut).
- Matters registered: vitmatte (tiled, HF `hustvl/vitmatte-*` or `models/vitmatte`), rvm (torch.hub
  `PeterL1n/RobustVideoMatting`, `models/rvm`), bgmattingv2, matanyone (MatAnyone v1, pip `matanyone`), birefnet,
  rmbg (both reuse the salient segmenters), guided_filter (model-free).
- Weights: `folder_paths` keys per backend; `auto_download` is the node's Allow Download (L7.97, default off).
- Caching: none across runs. Each run builds the backend and calls `free_vram()` at the end (SAM 3.1 reloads every run;
  memory note onyx-matte-pipeline).
- Front end: `js/mask_matting.js` (status strip with ONYX / Cascade buttons, filtered weight lists),
  `js/mask_visibility.js` (folding from the served spec `/c2c/mask/visibility`, `_visibility.py`),
  `js/c2c_maskops_health.js` (backend status pills), sections via `js/_c2c_sections.js`.
- On screen today at the defaults (rules + closed sections, widgets only, `tmp_p05/count_vis.py`): ONYX 12,
  cascade 20.

### 2.2 SAMViTMattePipelineMEC (`nodes/sam_vitmatte_pipeline.py`)

- Inputs: `sam_model` (SAM_MODEL socket, from SAM Model Loader), `image`; widgets `subject_type`
  (custom/hair/fur/cloth/skin_face/hard_edge/soft_glow), `points_json` ("[]", `[{x,y,label}]`), `bbox_json` ("",
  `[x1,y1,x2,y2]`), `sam_iterations` (2, 1..5), `refine_method` (auto/vitmatte/guided_filter/multi_scale_guided/
  color_aware/laplacian_blend), `edge_radius` (12, 1..200), `detail_preservation` (0.85), `edge_contrast` (1.0, 0..3),
  `fill_holes_enabled` (True), `min_region_size` (64, 0..10000), `multimask_output` (True), `mask_index` (0, 0..2),
  `score_threshold` (0.0); optional sockets `bbox` (BBOX), `existing_mask`, `trimap` (MASK); widgets
  `trimap_dilate` (0 = derive, 0..200), `trimap_erode` (0..200), `batch_mode` (False = first frame only).
- Outputs: `refined_mask, coarse_mask, edge_mask, preview (side by side), detected_bbox ([x, y, w, h]), score (best
  SAM score), info (JSON, has a wall-clock elapsed_sec)`.
- Models: the SAM model handed in (the loader owns it; a model-free stand-in is labelled, L7.98); ViTMatte through
  `nodes/utils.refine_with_vitmatte` (falls back to guided filters when it cannot load).
- Cache: the SAM model lives in the loader's output; per-frame `_vram_cleanup()`.

### 2.3 SeCMatAnyonePipelineMEC (`nodes/sec_matanyone_pipeline.py`)

- Inputs: `image`; widgets `segmentation_model` (MODEL_REGISTRY families sec/sam2/sam3: 10 entries, default
  sam2.1_hiera_base_plus), `text_prompt` (a WIDGET, ""), `points_json`, `bbox_json`, `matting_backend`
  (matanyone2/vitmatte_small/vitmatte_base/auto), `edge_radius` (15, 1..200), `n_warmup` (5, 1..30), `precision`
  (fp16/bf16/fp32), `fill_holes_enabled`, `min_region_size`; optional sockets `positive_coords`, `negative_coords`
  (read only by the model-free fallback), `bbox`; widgets `edge_refine_method` (none/vitmatte/guided_filter/
  multi_scale_guided), `keep_model_loaded` (True), `allow_download` (False).
- Outputs: `rgb (image x alpha), alpha_mask, coarse_mask, preview (one frame, side by side), info`.
- Models: `UnifiedSegmentation` (SeC-4B / SAM 2.x / SAM 3), MatAnyone2 and ViTMatte through `model_manager`
  (cached by (name, precision); `keep_model_loaded` off clears the cache). MatAnyone2 failing falls back to ViTMatte;
  missing models fall back to the labelled model-free path (GrabCut + optical flow, guided filter).

### 2.4 BackgroundRemoverMEC (`nodes/background_remover.py`)

- Inputs: `image`; widgets `model_name` (birefnet_general/birefnet_portrait/rmbg_2.0), `threshold` (0.5; > 0 makes
  the mask binary), `invert`, `mask_blur` (0..50); optional `keep_model_loaded` (True), `allow_download` (False).
- Outputs: `foreground (image x mask, keeps an RGBA input's 4 channels), mask, info`.
- Models: `model_manager` (BRIA RMBG-2.0, BiRefNet general / portrait), cached; model-free GrabCut when missing.

### 2.5 The card's backend list against the code

| Card | Where it exists today | Through the merged node |
|---|---|---|
| ViTMatte | MaskOps matter `vitmatte` (tiled); SAM + ViTMatte (`refine_with_vitmatte`); SeC (`vitmatte_small/base`) | all three |
| MatAnyone2 | SeC + MatAnyone2 only (`model_manager` "matanyone2"). MaskOps' own `matanyone` matter is MatAnyone v1 | backend "SeC + MatAnyone2" |
| RVM | MaskOps matter `rvm` | backend "segment + matte", matter rvm |
| BiRefNet | MaskOps segmenter + matter `birefnet`; Background Remover | "segment + matte" or "remove background" |
| RMBG-2.0 | MaskOps segmenter + matter `rmbg`; Background Remover | "segment + matte" or "remove background" |

Nothing on the card is missing. Duplicates exist (three ViTMatte paths, two BiRefNet / RMBG loaders, MatAnyone v1 and
v2 side by side); folding them into one implementation each is backlog, not this task (the old code must stay the
reference for the equivalence tests).

## 3. Target interface

### 3.1 The backend choice

New appended combo `backend`:

| Value | Runs | Notes |
|---|---|---|
| `segment + matte` (default) | MaskOps' own path, unchanged (`pipeline` onyx / cascade, segmenter, matter) | automatic choice stays inside it: segmenter `auto_best` / `auto`, matter `auto` |
| `SAM + ViTMatte` | `SAMViTMattePipelineMEC.execute` | needs `sam_model` wired (plain error otherwise) |
| `SeC + MatAnyone2` | `SeCMatAnyonePipelineMEC.process` | video aware; image works too |
| `remove background` | `BackgroundRemoverMEC.remove_bg` | no prompt; RMBG-2.0 / BiRefNet |

The default reproduces today's node, so every saved MaskOps workflow runs as before. There is no hidden automatic
switch between backends (a node that changes its own engine is surprising); the backend is the manual override and the
default backend picks its models itself. Whether an `auto` backend is wanted is an owner question (report).

### 3.2 Appended inputs (in this order)

`backend`, `sam_model` (socket), `points_json`, `bbox_json`, `sam_iterations`, `sam_refine_method`,
`sam_edge_radius`, `sam_detail_preservation`, `sam_edge_contrast`, `sam_multimask_output`, `sam_mask_index`,
`sam_score_threshold`, `sam_trimap_dilate`, `sam_trimap_erode`, `sam_batch_mode`, `sec_model`, `sec_text_prompt`,
`sec_matting_backend`, `sec_edge_radius`, `sec_n_warmup`, `sec_edge_refine_method`, `bg_model`, `bg_threshold`,
`bg_invert`, `bg_mask_blur`, `fill_holes_enabled`, `min_region_size`, `keep_model_loaded`, `split_masks`.

Each copies the old input's type, range, default and option list exactly, so every saved value validates. Names carry
the backend prefix where the meaning or the range differs from a MaskOps input (`sam_edge_radius` 1..200 default 12 is
not MaskOps' `edge_radius` 0..64 default 4). Shared, because the meaning is the same:

- MaskOps' own: `subject_preset` (same seven presets; each backend applies its own table), `precision`,
  `auto_download` (= Allow Download), `positive_coords` / `negative_coords`, `pos_bbox` (= old `bbox`),
  `external_mask` (= SAM + ViTMatte's `existing_mask`), `external_trimap` (= its `trimap`), `text_prompt` (socket:
  read by SeC when wired, before `sec_text_prompt`).
- New and shared by two backends: `points_json`, `bbox_json` (the click editor writes them), `fill_holes_enabled`,
  `min_region_size`, `keep_model_loaded`.
- `split_masks` (default on): make the edge / inside / outside masks. Off leaves them (and a derived trimap) empty,
  which saves three or four full-size masks per frame on long clips. All backends read it.

### 3.3 Outputs

The 18 outputs keep their names and order; `foreground` (IMAGE) is appended: the image (despilled when despill is
on) times the final alpha, premultiplied. On `segment + matte` with `premultiply` on it is the same tensor as
`preview` (no extra copy).

What each output carries per backend:

| Output | segment + matte | SAM + ViTMatte | SeC + MatAnyone2 | remove background |
|---|---|---|---|---|
| mask | coarse segmenter mask | coarse_mask | coarse_mask | the mask |
| alpha | refined alpha | refined_mask | alpha_mask | the mask |
| preview | image x alpha | its side-by-side preview | its one-frame side-by-side preview | = foreground |
| trimap | the matter's trimap | external, else derived from the final alpha (1 / 0.5 / 0) | same | same |
| bbox | [x0,y0,x1,y1] | detected_bbox, [x, y, w, h] (as before) | [x0,y0,x1,y1] | [x0,y0,x1,y1] |
| bbox_json | {x,y,w,h} | {x,y,w,h} | {x,y,w,h} | {x,y,w,h} |
| score | production-quality score | best SAM score (as before) | segmentation score | production-quality score |
| info | MaskOps JSON | the old JSON + `backend` + `maskops` keys | same | same |
| despilled | despilled image | despilled, or the image itself when despill is off (no copy) | same | same |
| lightwrap_rgba | light wrap | light wrap, or an empty layer (no memory) | same | same |
| edge_mask | edge band | its own edge mask (as before) | edge band | edge band |
| inside / outside | split | split (empty when split_masks is off) | same | same |
| luma_key_mask | luma key | empty (the luma key is part of segment + matte) | same | same |
| problem_regions, severity, suggested_method | diagnostics | diagnostics when enable_diagnose is on | same | same |
| object_alphas | per object | = alpha | = alpha | = alpha |
| foreground | image x alpha | image x alpha | its rgb | its foreground |

Three per-backend definitions are kept only so migrated graphs give identical results: SAM + ViTMatte's bbox format,
SAM + ViTMatte's and SeC's side-by-side preview, SAM + ViTMatte's edge mask. Unifying them is an owner question.

The finishing stage that every backend shares: holdout / core masks, `edge_mode` (hard / feather), despill, light
wrap, split masks, diagnostics, foreground. At the defaults none of them changes the alpha.

### 3.4 Controls on screen (target, defaults, widgets only)

| Backend | On screen | Sections ("More" bar on the classic canvas; core's advanced inputs in Nodes 2.0) |
|---|---|---|
| segment + matte, cascade | backend, pipeline, segmenter, matter, model, matter_model, subject_preset, edge_mode, despill (9) | Accuracy, Luma key, Trimap, Diagnostics, Video, Output, Performance |
| segment + matte, ONYX | backend, pipeline, matter, model, matter_model, scene_prompts, edge_mode, despill (8) | same |
| SAM + ViTMatte | backend, subject_preset, points_json, bbox_json, sam_refine_method, sam_edge_radius, sam_batch_mode, edge_mode, despill (9; the click editor replaces the two JSON boxes) | Accuracy, Trimap, Cleanup, Output, Diagnostics, Performance |
| SeC + MatAnyone2 | backend, sec_model, sec_text_prompt, points_json, bbox_json, sec_matting_backend, sec_edge_radius, edge_mode, despill (9) | Accuracy, Cleanup, Output, Diagnostics, Performance |
| remove background | backend, bg_model, bg_threshold, bg_invert, bg_mask_blur, edge_mode, despill (7) | Output, Diagnostics, Performance |

Sockets are never hidden. A switch and the controls it reveals sit in the same section (enable_luma_key with the
luma_* controls, enable_diagnose with diag_*, robust_propagation with robust_*), so a closed section never leaves a
revealed control without its switch. Section membership only adds the option keys `c2c_section` and `advanced`; it
changes no name, order or value.

The visibility rules stay server-driven (`_visibility.py` -> `/c2c/mask/visibility` -> `js/mask_visibility.js`):
the spec gains `backends` (what each engine backend reads) and `backend_only` (engine controls that `segment + matte`
never reads). The click editor (`js/points_bbox_editor.js`) binds to MaskOps and shows only while the backend reads
`points_json` (SAM + ViTMatte, SeC). The status strip gets the backend choice next to ONYX / Cascade.

## 4. Migration rows (nodes/_legacy_replacements.json)

`old_widget_ids` = the old node's widgets in order (sockets excluded), as the front end saves them.

### SAMViTMattePipelineMEC -> MaskOpsMEC

| Old input | New input |
|---|---|
| sam_model | sam_model |
| image | image |
| subject_type | subject_preset |
| points_json / bbox_json | points_json / bbox_json |
| sam_iterations, refine_method, edge_radius, detail_preservation, edge_contrast, multimask_output, mask_index, score_threshold, trimap_dilate, trimap_erode, batch_mode | sam_* of the same name (refine_method -> sam_refine_method, ...) |
| fill_holes_enabled, min_region_size | same names |
| bbox | pos_bbox |
| existing_mask | external_mask |
| trimap | external_trimap |
| set | backend = "SAM + ViTMatte", enable_diagnose = false, split_masks = false |

Outputs: refined_mask 0 -> alpha 1, coarse_mask 1 -> mask 0, edge_mask 2 -> edge_mask 10, preview 3 -> preview 2,
detected_bbox 4 -> bbox 4, score 5 -> score 6, info 6 -> info 7. Dropped: nothing.

### SeCMatAnyonePipelineMEC -> MaskOpsMEC

| Old input | New input |
|---|---|
| image | image |
| segmentation_model | sec_model |
| text_prompt (widget) | sec_text_prompt |
| points_json / bbox_json | same |
| matting_backend, edge_radius, n_warmup, edge_refine_method | sec_matting_backend, sec_edge_radius, sec_n_warmup, sec_edge_refine_method |
| precision, fill_holes_enabled, min_region_size, keep_model_loaded | same names |
| allow_download | auto_download |
| positive_coords / negative_coords | same names |
| bbox | pos_bbox |
| set | backend = "SeC + MatAnyone2", enable_diagnose = false, split_masks = false |

Outputs: rgb 0 -> foreground 18, alpha_mask 1 -> alpha 1, coarse_mask 2 -> mask 0, preview 3 -> preview 2, info 4 ->
info 7. Dropped: nothing.

### BackgroundRemoverMEC -> MaskOpsMEC

| Old input | New input |
|---|---|
| image | image |
| model_name, threshold, invert, mask_blur | bg_model, bg_threshold, bg_invert, bg_mask_blur |
| keep_model_loaded | keep_model_loaded |
| allow_download | auto_download |
| set | backend = "remove background", enable_diagnose = false, split_masks = false |

Outputs: foreground 0 -> foreground 18, mask 1 -> alpha 1, info 2 -> info 7. Dropped: nothing.

`enable_diagnose` and `split_masks` are set off on migrated nodes because the old nodes never computed them: a
migrated workflow then costs what it cost before (section 5). A fresh node has both on, like `segment + matte`.

## 5. Memory and time budget

P = one float mask plane = H x W x 4 bytes (8.3 MB at 1080p). Output memory per frame, measured as the sum of unique
tensor storages the node returns (test `test_p05_*_memory`):

| Backend | Old node | Merged node, migrated (split off, diagnostics off) | Merged node, fresh defaults |
|---|---|---|---|
| SAM + ViTMatte | refined + coarse + edge + preview 3P = 6P | 6P + foreground 3P = 9P | + inside/outside/trimap 3P = 12P |
| SeC + MatAnyone2 | rgb 3P + alpha + coarse = 5P (+ one preview frame) | 5P | + edge/inside/outside/trimap 4P + problem map P = 10P |
| remove background | foreground 3P + mask = 4P | 4P | + 4P + P = 9P |
| segment + matte | about 20P (unchanged) | - | unchanged |

Rules that keep it there: no copy of the input image (despilled is the image itself when despill is off); outputs that
a backend does not produce (luma key, an off light wrap, an off problem map) are zero-stride views, which take no
memory (ComfyUI core's own Mask to Image returns such a view); object_alphas and the remove-background preview are the
same tensor as alpha / foreground. No output is computed at a lower resolution or precision.

Time: the engines run their own code, so a migrated node takes the old time plus the finishing stage. Measured on this
CPU (1080p, one frame): `edge_inside_outside` 3.6 s (a 9 x 9 max-pool; a separable pool gives the same result in
0.22 s - backlog), `score_quality` 0.22 s. Migrated rows switch split masks and diagnostics off; the production score
runs for remove background (no score of its own).

Model reuse: SeC / MatAnyone2 / ViTMatte / RMBG / BiRefNet stay cached by `model_manager` while `keep_model_loaded`
is on, as before; the SAM model is the loader's. Caching the `segment + matte` backends across runs (SAM 3.1 reloads
every run today) is backlog: it needs a measured cache-eviction policy.

## 6. Failure modes

| Case | Behaviour |
|---|---|
| SAM + ViTMatte without `sam_model` | ValueError: "SAM + ViTMatte needs a SAM model: connect a SAM Model Loader to sam_model." |
| SAM model is the loader's model-free stand-in | the old labelled approximate path (info `approximate`, `model_free`) |
| SeC / MatAnyone2 / ViTMatte / RMBG / BiRefNet weights missing, Allow Download off | the old nodes' labelled model-free paths; info `model_status` names the model and folder |
| MatAnyone2 fails to load | falls back to ViTMatte small, as before (logged) |
| Out of memory at load | `model_manager`'s plain message, as before |
| `batch_mode` off with a clip | SAM + ViTMatte processes frame 0 only, as before; every output has one frame |
| holdout / core / external trimap of another size | ignored when H x W differs; a single-frame mask is spread over the batch (as `segment + matte` does) |
| an old engine module fails to import | MaskOps still loads (the import is lazy); choosing that backend raises a plain error naming the module |
| a saved value no longer in a list (registry changed) | the same validation the old node had (same option lists) |

## 7. Kill switch and fallback

- `C2C_LEGACY_MATTE_NODES=1` registers the three old nodes again, from their unchanged code. The workflow migration
  only rewrites node types that are not registered, so with the switch on old workflows load the original nodes.
- Settings > C2C > Workflows > "Migrate old C2C nodes on load" off: ComfyUI offers the replacement instead of the
  automatic rewrite (existing).
- `backend = segment + matte` is the default, so nothing changes for existing MaskOps users unless they choose another
  backend.

## 8. What stays exactly as it was

- MaskOps' id, its 78 inputs (name, order, type, default, range, options) and its first 18 outputs.
- The `segment + matte` path: no line of its computation changes; it only appends `foreground` and reads
  `split_masks` (default on = today's split).
- The three engine modules: byte for byte the d97cc13 files (a test pins their SHA-256), so the equivalence reference
  is the shipped code.
- The existing visibility rules for `segment + matte`; sections add `c2c_section` / `advanced` option keys only.

## 9. Tests

- `tests/test_consolidation_p05_matte.py`: for each row, the old node's inputs go through the row (the wave-2
  `migrate_kwargs`) into MaskOps, and every mapped output is compared with the old node's (bit-equal; info compared as
  JSON with the two added keys removed and SAM + ViTMatte's wall-clock `elapsed_sec` ignored). Inference functions are
  monkeypatched to deterministic fakes that record their arguments (both nodes must make the same calls); the
  model-free paths run for real.
- Spec parity: every appended input equals the old input's spec; the engine files' SHA-256; outputs 0-17 unchanged.
- Visibility: at most 10 widgets on screen per backend at the defaults; every engine control is reachable in its
  backend and hidden in the others; sockets never hidden.
- Memory: output storage per frame against section 5.
- Migration table tests (`tests/test_legacy_replacements.py`), fixtures, docs generator, example workflow.

## 10. Backlog found while designing (not in this task)

- One implementation each for ViTMatte (three today), BiRefNet / RMBG (two loaders), MatAnyone (v1 matter vs v2
  engine).
- Model caching for `segment + matte` backends across runs (SAM 3.1 reloads every run).
- `edge_inside_outside`: separable max-pool, same result, 3.6 s -> 0.22 s per 1080p frame on CPU (measured).
- `segment + matte` allocates zeros / a clone for features that are off (despill, light wrap, luma key, problem map:
  up to 9P per frame); zero-stride views would free it.
- A foreground colour estimate (e.g. multi-level foreground estimation) instead of image x alpha.
- Prompt sockets read by every backend (today SAM + ViTMatte reads only points_json / bbox_json).
