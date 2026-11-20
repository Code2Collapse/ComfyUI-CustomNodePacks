# Helpers, Diagnostics and Output Folders

Small nodes that sit around the real work: they pick sizes and frames, route values, time and measure a run,
explain a bad mask, read a model file without loading it, keep a parameter history, and name versioned output
folders. None of them loads a model or needs a GPU.

Every input and output has a tooltip in [NODE_REFERENCE.md](../NODE_REFERENCE.md). That file was generated before
the 2026-11-19 features (commits `4cf6765` and `4ebaa55`), so the inputs and outputs those commits added are
described on this page and are not in NODE_REFERENCE yet.

| Node | Use it to |
|---|---|
| [Size (C2C)](#size-c2c-aspectpresetmec) | Get a width and height from an aspect ratio, a model family's rule or a named size |
| [Conditional Switch](#conditional-switch-conditionalswitchmec) | Pass on one of two to five inputs, by a switch, a comparison or an index |
| [Execution Timer (C2C)](#execution-timer-c2c-executiontimermec) | Time a stage of a workflow and keep run statistics |
| [Batch Range (C2C)](#batch-range-c2c-imagebatchslicemec) | Pick frames from an IMAGE batch: a range, a split, one frame, a list of ranges or a new frame rate |
| [Probe (C2C)](#probe-c2c-imagestatsprobemec) | Read statistics of an image, a mask or a latent, and draw a scope |
| [Number Lerp (C2C)](#number-lerp-c2c-numberlerpmec) | Get a value between two numbers, or from keyframes |
| [Seed List Generator (C2C)](#seed-list-generator-c2c-seedlistmec) | Make repeatable seeds, per shot, with a manifest |
| [Text Template (C2C)](#text-template-c2c-texttemplatemec) | Build a prompt from a template and variables |
| [Mask Failure Explainer](#mask-failure-explainer--diagnostics-maskfailureexplainermec) | Find out why a mask is bad, frame by frame, and preview a fix |
| [Model Metadata Extractor](#model-metadata-extractor-modelmetadataextractormec) | Read a model file's metadata, compare two files, audit a pickle, without loading weights |
| [Parameter History](#parameter-history-parameterhistorymec) | See which settings changed between runs; save named snapshots |
| [Folder Version Incrementer](#folder-version-incrementer-folderincrementer) | Name a new version folder (or file) for every run |
| [Folder Version Variant (per output)](#folder-version-variant-per-output-folderversionvariant) | Give one output its own name under the shared version |
| [Folder Version Control (check / set)](#folder-version-control-check--set-folderversioncontrol) | See or move the version counter |

---

# Helpers

All eight are in **🐺 C2C/🧰 Core/Helpers**. They work on the CPU. New inputs from 2026-11-19 were appended with
defaults that give the old outputs, so saved workflows run as before (commit `4cf6765`).

## Size (C2C) (`AspectPresetMEC`)
**Menu:** 🐺 C2C/🧰 Core/Helpers · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#aspectpresetmec)

### What it is for
Gives a `width` and a `height` for an Empty Latent or a resize node. Pick an aspect ratio scaled to a long edge, a
Wan native size, or your own width x height; the result is snapped to a multiple. Dimensions Snap was merged into
this node (preset `Custom (width x height)`).

### Minimal wiring
```
Size (C2C) ──width──►  Empty Latent Image
           ──height─►
```

### Key controls
- `preset`: a ratio (`1:1 square`, `16:9 landscape`, `9:16 portrait`, `4:3 landscape`, `3:4 portrait`,
  `21:9 ultrawide`, `2.39:1 cinema`) scaled so the long edge is `base`; a Wan size (`Wan 480p land` 832x480,
  `Wan 480p port`, `Wan 720p land` 1280x720, `Wan 720p port`), returned exactly; or `Custom (width x height)`.
- `multiple` and `direction` (`down`, `nearest`, `up`): the snap. Wan presets ignore `multiple`.
- **Profiles** section (folded):
  - `model_profile` snaps to a model family's rule instead of `multiple`: `SD 1.5 / SDXL (x64)`,
    `SD3 / Flux / Qwen Image (x16)`, `Wan 2.1 / 2.2, HunyuanVideo (x16)`, `LTX-Video (x32)`.
  - `custom_profiles` holds your own named sizes, one per line: `Hero = 1920x1080` (an exact size, then snapped) or
    `Scope = 2.39:1` (a ratio scaled to `base`). `#` starts a comment. Copy the text to share it between workflows.
  - `profile` is the name of one of those lines; it is used instead of `preset`.
  - `overscan_pct` makes the size this many percent larger before snapping, as a margin to crop back later. It also
    applies to Wan presets; they are then snapped to x16 (or the model profile).

### Common mistakes
- A `profile` name that is not in `custom_profiles` stops the run and lists the names it found.
- A `custom_profiles` line that is not `Name = WxH` or `Name = A:B` stops the run with the line number.
- `width` / `height` are read only with `Custom (width x height)`; `base` is not read for Wan presets.

## Conditional Switch (`ConditionalSwitchMEC`)
**Menu:** 🐺 C2C/🧰 Core/Helpers · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#conditionalswitchmec)

### What it is for
Passes one of its inputs on to `out`. Inputs are wildcards, so it works for any type. The inputs are lazy: only the
branch that is picked is computed upstream.

### Minimal wiring
```
branch A ──► value_true  ┐
branch B ──► value_false ├─ Conditional Switch ──out──► next node
(optional) value_2..4, fallback, compare_a ┘
```

### Key controls
`source` decides what picks the branch:
- `condition` (the default): the `condition` switch. On passes `value_true`, off passes `value_false`.
- `compare`: tests `compare_a` `operator` `compare_b`. True passes `value_true`, false passes `value_false`.
  - The 12 operators: `==`, `!=`, `<`, `<=`, `>`, `>=`, `contains`, `starts with`, `ends with`, `matches regex`,
    `is empty`, `is not empty`.
  - `compare_as` (`auto`, `number`, `text`, `boolean`) sets how both sides are read. `auto` follows the type of
    `compare_a`. The four text operators always compare as text. `is empty` / `is not empty` ignore `compare_b`.
- `index`: 0 = `value_false`, 1 = `value_true`, 2-4 = `value_2`..`value_4`.

`fallback` is used when the picked branch is not wired or the index is out of range.

### Reading the outputs
With `source` `compare` or `index`, the console log says why a branch was picked, e.g.
`(number) 3.0 > 2.0 -> True -> value_true`.

### Common mistakes
- An index whose branch is not wired, with no `fallback` wired either, stops the run: "Wire it, or wire 'fallback'".
- A side that cannot be read as the chosen type stops the run, e.g. `compare_a 'abc' is not a number`.
- `<`, `<=`, `>`, `>=` on booleans stop the run.
- `compare` with any operator except `is empty` / `is not empty` needs `compare_a` wired.
- A regex longer than 256 characters, or an invalid one, stops the run.

## Execution Timer (C2C) (`ExecutionTimerMEC`)
**Menu:** 🐺 C2C/🧰 Core/Helpers · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#executiontimermec)

### What it is for
A stopwatch you put in a wire. It passes its input on unchanged. It reports the seconds since the last tick of the
same `label`, or since another label's tick.

### Minimal wiring
Time one stage: one timer before it, one after it.
```
VAE Decode ──► Timer (label "start") ──► Upscale ──► Timer (label "upscale", since_label "start") ──► Save
```

### Key controls
- `label`: timers with the same label measure from each other's ticks.
- `since_label`: measure from the last tick of that other label. The stage is then named `start->upscale`.
- `reset`: this tick counts as the first one.
- `history_size`: how many measurements per stage are kept for the statistics (default 50). They stay in memory
  until ComfyUI restarts.

### Reading the outputs
- `report`: `[stage] +1.2345s`.
- `elapsed_seconds`: the same number. It is 0 on the first tick.
- `stats_json` fields:
  - `stage`, `elapsed`, and `measured` (false for a first tick or a reset, which are not kept);
  - `stats`: count, last, median, p95, min, max and total of the kept measurements;
  - `memory`: `ram_rss_mb` (needs the `psutil` package, else null) and, on a CUDA machine, `vram_allocated_mb` and
    `vram_peak_mb`;
  - `tree`: every stage, grouped by `/` in the labels. For example `render/decode` adds to `render`.

### Common mistakes
- The node re-runs only when its inputs change. If nothing upstream changed, ComfyUI uses the cached result and the
  timer does not tick.

## Batch Range (C2C) (`ImageBatchSliceMEC`)
**Menu:** 🐺 C2C/🧰 Core/Helpers · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#imagebatchslicemec)

### What it is for
Picks frames from an IMAGE batch. Image Batch Slice, Image Batch Split and Video Frame Extractor are all modes of
this node now; their saved workflows load here. Image Batch Split loads as the `split at …` modes and Video Frame
Extractor as the single-frame modes.

### Minimal wiring
```
Load Video ──IMAGE──► Batch Range ──images────► (picked frames)
                                  ──remainder─► (the rest, split and ranges modes)
```

### Key controls
`mode` picks the job. Each mode shows only its own controls.
- `range`: `start` : `end` : `step`. A negative number counts from the end; `end` -1 = to the end; `end` is
  exclusive.
- `split at index` / `split at ratio`: the frames before the cut go to `images`, the rest to `remainder`.
- `first frame`, `middle frame`, `last frame`, `frame at index`: one frame.
- `ranges`: a comma-separated list in `ranges`. Here ends are included.
  - `0..9, 20..29/2, -5..` — `/N` keeps every Nth frame; an open end runs to the last frame.
  - `3-7` (plain frame numbers) also works.
  - With `fps` set, positions may be seconds (`1.5s..3s`) or timecodes (`00:00:02:00..00:00:03:12`, HH:MM:SS:FF).
  - `images` gets the frames in the order listed; `remainder` gets every frame not picked.
- `resample to fps`: re-times the batch from `fps` to `target_fps`. The clip keeps its length. Each new frame is the
  source frame nearest in time, so 30 -> 24 drops frames and 24 -> 48 repeats them. Frames are not blended.

### Reading the outputs
`frame_count` and `remainder_count` count the frames in `images` and `remainder`. `total_frames` is how many came in.
`is_video` is true when more than one came in.

### Common mistakes
- An empty batch stops the run in the single-frame, `ranges` and `resample to fps` modes.
- A range outside the batch, or with its start after its end, stops the run. It never returns an empty batch.
- Seconds or timecodes with `fps` 0 stop the run. `resample to fps` needs both `fps` and `target_fps` above 0.
- `ranges` includes its end frame; `range` mode does not.

## Probe (C2C) (`ImageStatsProbeMEC`)
**Menu:** 🐺 C2C/🧰 Core/Helpers · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#imagestatsprobemec)

### What it is for
Wire an image, a mask and/or a latent in the middle of a graph. Each passes through unchanged, and the probe reports
on it: black or blown-out frames, mask coverage, NaN / Inf or saturated latents. Mask Area Probe and VAE Latent
Inspector were merged in; all their outputs kept their places here.

### Minimal wiring
```
VAE Decode ──images──► Probe ──images──► Save Image
                             ──report──► Preview Text (or any text display)
                             ──scope───► Preview Image
```

### Key controls
- `threshold` (mask only): a pixel counts as covered above this value.
- `fail_on_corrupt` (latent only): stop the run when the latent holds NaN or Inf.
- **Analysis** section:
  - `scope`: `histogram` (R, G, B), `waveform` (luma per column) or `rgb parade` of the whole image batch. It is
    256 px high on a log scale. Long clips and large frames are sampled: at most 16 frames and 512 px per side.
  - `temporal_check`: measures every frame of the image and/or mask, then flags NaN / Inf frames and sudden jumps
    in exposure, colour, clipping or mask (alpha).
  - `anomaly_threshold` (default 0.05): how big a frame-to-frame change must be to be flagged.

### Reading the outputs
- `report`: one line per wired input, e.g. `shape=… dtype=… mean=… std=… min=… max=… bright>0.95=…%`, the mask coverage
  line, `latent verdict=… NaN=… Inf=…`, and `temporal: N flagged frame(s)`.
- `mean`, `std`, `bright_pct` describe the image.
- `coverage_mean_pct`, `coverage_min_pct`, `coverage_max_pct` give per-frame mask coverage.
- `info_json` holds the latent's shape, dtype, min / max / mean / std, per-channel stats and NaN / Inf counts.
  `verdict` is `healthy`, `low_contrast`, `saturated` or `corrupt`.
- `frame_stats_json` (only with `temporal_check` on): per-frame `luma`, `channel_mean`, `clipped_pct`, `nan`, `inf`,
  `alpha_mean`, and an `anomalies` list with the reason for each frame. `anomaly_count` counts them.
- `width`, `height`: from the image, else the mask. They are 0 when only a latent is wired.
- `frame_count`: from the image, else the mask, else the latent batch.
- `scope`: a black 64x64 picture when `scope` is `none`.

### Common mistakes
- Nothing wired stops the run: "connect an image, a mask or a latent".
- A scope with no image wired gives the black picture and a `scope …: no image wired` line in `report`.

## Number Lerp (C2C) (`NumberLerpMEC`)
**Menu:** 🐺 C2C/🧰 Core/Helpers · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#numberlerpmec)

Gives a value between `a` and `b` at `t` (0..1), as `float` and rounded to `int`.

- `curve`: `linear`, `smoothstep`, `cosine` (both ease in and out), or `hold` (keeps the earlier value until the
  next key).
- `keyframes`: `position:value` pairs, e.g. `0:0, 0.5:1, 1:0`. When set, `a` and `b` are ignored and `curve` shapes
  each segment. Before the first key and after the last, the value holds.
- `time_mode`: `normalised t` uses `t`, as before. `frame` or `seconds` uses `time` instead. Key positions are then
  frames or seconds; without keyframes, t = `time` / `duration`.
- `loop` (frame / seconds only): `clamp` holds the end value, `loop` restarts each `duration`, `ping-pong` runs
  forward then back. With keyframes, `duration` 0 means "up to the last key".

Mistakes that stop the run: `frame` / `seconds` with no keyframes and `duration` 0; a keyframe that is not two
numbers separated by `:`.

## Seed List Generator (C2C) (`SeedListMEC`)
**Menu:** 🐺 C2C/🧰 Core/Helpers · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#seedlistmec)

Makes `count` seeds from `base_seed`. The same inputs always give the same seeds.
- `mode`: `increment` (base, base+1, …), `hash` (from a SHA-256 of base and index, so the seeds are unrelated) or
  `random` (a seeded random stream).
- `shot_ids` (optional, e.g. `sh010, sh020`): each shot gets its own `count` seeds. A shot's seeds depend only on
  `base_seed` and its id, so reordering or adding shots never changes them.

Outputs:
- `first_seed` and `csv_all_seeds`.
- `seeds`: a list. A node wired to it runs once per seed.
- `seed_map_json`: `{shot id: [seeds]}`; `{}` without shot ids.
- `manifest_json`: the recipe (base seed, count, mode, the algorithm in words, shot ids), every seed, and run
  details: a SHA-256 of the prompt, the node id, the Python / torch / ComfyUI versions and the UTC time.

A repeated shot id stops the run.

## Text Template (C2C) (`TextTemplateMEC`)
**Menu:** 🐺 C2C/🧰 Core/Helpers · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#texttemplatemec)

Fills `{a}` `{b}` `{c}` `{d}` in `template` with the inputs of those names and returns `text`.

- `variables`: more names, one `name = value` per line (e.g. `style = anime`), used as `{style}`. They may override
  a-d. `#` lines are comments.
- `schema`: one rule per line, `name: text|int|float|bool|choice(x|y) [= default] [required]`, e.g.
  `steps: int = 20` or `style: choice(photo|anime) required`. Empty values take the default. A missing required
  value, or a value of the wrong type, stops the run with the list of problems.
- `engine`:
  - `format` (default): `{name}` substitution as before. Unknown names and broken braces are left as they are.
  - `extended`: also keeps or drops sections and applies filters:
    - `{?name}…{/name}` is kept when name is set and not false/0/no/off;
    - `{?name=value}…{/name}` is kept when name equals that value;
    - `{!name}…{/name}` is kept when name is empty;
    - filters: `{name|upper}`, `lower`, `title`, `capitalize`, `strip`, `default:x`, `prefix:x`, `suffix:x`,
      `truncate:N`;
    - `{{` and `}}` are literal braces.

  The extended engine only substitutes text; it does not evaluate expressions. An unknown name, an unknown filter,
  an unclosed section or a stray brace stops the run with a message.

`report` (JSON) lists the engine, every variable's final value after schema defaults and type checks, and, for
`extended`, which sections were kept or dropped.

---

# Diagnostics

These nodes were extended on 2026-11-19 (commit `4ebaa55`). The commit tested them on the CPU with stand-in data;
it states that GPU behaviour and the live UI are NOT VERIFIED.

## Mask Failure Explainer — Diagnostics (`MaskFailureExplainerMEC`)
**Menu:** 🐺 C2C/🧰 Core/Diagnostics · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#maskfailureexplainermec)
No model. Uses OpenCV when installed, else plain torch.

### What it is for
You have an image and a mask that looks wrong. This node measures the image around the mask and the mask itself.
It tells you, frame by frame, what is likely wrong and what to try next. It can also preview one fix on a copy of
the mask; the input mask is never changed.

### When to pick it
Use it to find the cause. To change the mask for real, use [Mask Refiner](../NODE_REFERENCE.md#maskrefinemec),
which its advice often names, or the mask tools in [mask-editing.md](mask-editing.md). For plain coverage numbers,
[Probe (C2C)](#probe-c2c-imagestatsprobemec) is lighter.

### Minimal wiring
```
Load Image ──IMAGE──► Mask Failure Explainer ──explanation──► (text display)
SAM / segmenter ──MASK──►                   ──problem_regions_mask──► Preview (as image)
                                            ──corrected_mask──► (try it downstream)
```
A single mask is used for every frame. A mask of another size is resized to the image.

### Key controls
- `ring_width` (5 px): the band around the mask edge used for contrast and colour.
- `blur_threshold` (50) and `brightness_threshold` (0.15): a frame below them is called blurry or dark.
- `area_jump_threshold` (0.25): a frame whose mask area changes by more than this fraction is flagged `area_jump`.
- `preview_fix` (default `none`): `fill_holes`, `remove_specks`, `close_gaps`, `smooth_edges`, `dilate`, `erode`,
  `invert`, or `auto`. `auto` picks from the findings: whole frame -> `invert`, holes -> `fill_holes`,
  fragments -> `remove_specks`, jagged edge -> `smooth_edges`.
- `fix_radius_px` (3): radius for dilate / erode / close_gaps / smooth_edges.

### Reading the outputs
- `explanation`: per frame, brightness, blur score, boundary contrast, colour confusion and background complexity,
  each with a warning or a tick. Then the mask-quality means over the batch: coverage, components, interior holes,
  boundary jaggedness (round = 1.00), soft-alpha pixels, edge-gradient IoU and frame-border share. Then
  recommendations.
- `severity_score`: 0 to 100; higher means more problems.
- `suggested_method`: up to four next steps joined by `→`, e.g. a ViTMatte, SAM2 or Mask Refiner suggestion. It is
  `auto (no significant issues …)` when nothing triggered.
- `problem_regions_mask`: a heatmap of where the problems are.
- `timeline_json`: one entry per frame with the five image metrics, coverage, `area_change`, `centroid_shift` and
  `flags`. Possible flags: `dark`, `blurry`, `low_boundary_contrast`, `color_confusion`, `busy_background`,
  `empty_mask`, `over_segmented`, `area_jump`, `centroid_jump` (the centroid moved more than 10 % of the image
  diagonal). `summary` lists the problem frames and how often each flag appeared.
- `corrected_mask`: the mask after `preview_fix`; the analysed mask unchanged when `preview_fix` is `none`.
- `fix_report`: the fix requested and applied, why, the findings, and the mask metrics `before` and `after` the
  fix, with `findings_after`.

### Common mistakes
- A mask batch whose length is neither 1 nor the image batch length stops the run.
- `remove_specks` needs OpenCV or SciPy; with neither installed it stops the run.
- Frames with a long edge above 2048 px are analysed at 2048 px; the heatmap is scaled back up.
- The tips name other tools (SAM2, ViTMatte, GroundingDINO, BiRefNet …). They are suggestions only; this node
  does not run them.

## Model Metadata Extractor (`ModelMetadataExtractorMEC`)
**Menu:** 🐺 C2C/🧰 Core/Diagnostics · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#modelmetadataextractormec)

### What it is for
Reads what a `.safetensors`, `.pt`, `.pth` or `.ckpt` file says about itself without loading its weights. It never
unpickles a file: safetensors headers, zip directories and pickle opcodes are read as bytes. It can also compare two
files and audit a file's integrity.

### Minimal wiring
```
Model Metadata Extractor (file_path = D:/models/loras/my_lora.safetensors) ──metadata_json──► (text display)
```
`file_path` is an absolute path typed or wired in; it is not a model picker.

### Key controls
- `compute_fingerprint` (on): a quick SHA-256 over the file size, the first 1 MB and the last 1 MB. It is meant as a
  cache key, not a download check.
- `compare_path`: a second file. The two are compared without loading weights.
- `integrity_report`: the audit described below.
- `full_hash`: also hashes the whole file with SHA-256, a hash you can match against the ones model sites publish.
  It reads every byte.

### Reading the outputs
- `metadata_json` covers the format, file size and the first 64 tensors (name, dtype, shape, parameters). It also
  has the `__metadata__` block of a safetensors file, or the zip member list and a disassembly excerpt of a torch
  file.
- `model_kind`: a guess from tensor names: `stable_diffusion_checkpoint`, `vae`, `lora`, `text_encoder`,
  `safetensors_other` or `unknown`.
- `total_params`: summed from the safetensors header. It is 0 for `.pt` / `.ckpt` files.
- `lineage_json`: licence, author, base model, architecture, prediction type, resolution, training steps / epochs,
  network module / dim / alpha, clip skip, and the raw `modelspec.*` / `ss_*` keys, when the file has them.
- `comparison_json` (with `compare_path`):
  - safetensors: tensor counts, parameter totals and `param_delta`, tensors only in A or only in B, shape and dtype
    mismatches, and `same_structure`;
  - torch zip files: archive members and their sizes only;
  - legacy pickles cannot be compared.
- `audit_json` (with `integrity_report` or `full_hash`):
  - `format_by_magic`, the real type read from the first bytes, with a warning when it does not match the
    extension;
  - `sha256_full`, when `full_hash` is on;
  - safetensors: tensor byte ranges that overlap, fall outside the file or do not match shape x dtype, plus gaps and
    trailing bytes;
  - `.pt` / `.ckpt`: `pickle_globals`, the callables the pickle would call on load, read from the opcodes of the
    first 1 MB, and `suspicious_globals`, those outside the torch / numpy / collections set;
  - `verdict`: `ok`, `warnings` or `suspicious`.

### Common mistakes
- An empty or wrong `file_path` (or `compare_path`) stops the run with "Model file not found".
- `suspicious` means a pickle names code outside the usual set. Treat that file with care. `ok` covers only what
  this audit checks; it does not prove a file is safe.
- `.gguf` files are recognised in the audit's format check only; metadata is not read from them.

## Parameter History (`ParameterHistoryMEC`)
**Menu:** 🐺 C2C/🧰 Core/Utils · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#parameterhistorymec)

### What it is for
The C2C front end records your settings each time you queue a workflow. For every node, it stores the values that
differ from the node's defaults in `param_history.db` in the pack folder. The last 30 runs are kept. Values that look
like API keys or passwords are not stored. This node reads that history, compares runs, and saves named snapshots.

The same front end also adds **Parameter History**, **Changes Since Last Run**, **Presets** and
**Reset to Defaults** to every node's right-click menu. Alt + hover over a widget shows its previous value.

### Minimal wiring
Place the node alone (it is an output node) and queue. Read `history_report` in a text display.

### Key controls
- `mode`:
  - `all_history`: recent changes on all nodes;
  - `last_run_diff`: what changed between two runs. `run_a` / `run_b` = 0 picks the last two;
  - `node_class_filter`: only one class, e.g. `KSampler`. Fill in `node_class_filter`.
- `last_n_runs`: how far back to look.
- `snapshot_action` (default `none`):
  - `save` stores run `run_b` (0 = the latest run) under `snapshot_name`. Snapshots are kept when old runs are
    pruned. Saving under an existing name replaces it.
  - `compare` lists the differences between `snapshot_name` and `compare_to` (empty = the latest run).
  - `restore_plan` lists only the values from `snapshot_name` that match `restore_filter`, e.g. `KSampler, cfg`. The
    filter matches node classes, titles, ids, parameter names or `Class.param`. If `snapshot_name` is empty, the
    latest run is used.

### Reading the outputs
- `history_report`: runs as `── Run #N (time) ──`, then each node with its changed values. Diff lines read
  `Title (Class) → param: old → new`.
- `snapshot_json`:
  - after `save`: the name, run, node and parameter counts, and whether it overwrote a snapshot;
  - after `compare`: the list of differences;
  - after `restore_plan`: a patch `{node_id: {params}}`.

  The restore plan is a list to apply by hand. The node does not change your workflow.

### Common mistakes
- No history yet: queue the workflow once first.
- `save` with no `snapshot_name`, a name that does not exist, or `restore_plan` with an empty `restore_filter` stops
  the run.
- Run numbers are counted by the browser page and start again at 1 after a reload. Runs from different sessions can
  share a number.
- `last_n_runs` limits the rows read (20 per run asked), so a workflow with many changed nodes can show fewer runs.

---

# Output folders and versions

These nodes live in `folder_incrementer.py` with `js/folder_incrementer.js`. Their menu entries are under
**🐺 C2C/🧰 Core/Utils**. They only make folder and file names; your save node writes the files.

## Folder Version Incrementer (`FolderIncrementer`)
**Menu:** 🐺 C2C/🧰 Core/Utils · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#folderincrementer)

### What it is for
Gives every run its own version folder, named after the source clip or image:
`<output>/<source name>/<date>/v001/<source name>.<ext>`. Each date starts again at v001. Batch Version Manager was
merged into it as the `show / shot / task / version` layout.

### When to pick it
Use it in front of any save node whose files should not overwrite the last run.
- [Folder Version Variant](#folder-version-variant-per-output-folderversionvariant) names several outputs of one run.
- [Folder Version Control](#folder-version-control-check--set-folderversioncontrol) reads or moves the counter.
- Save Video (C2C) can take the version path by itself; see [save-video-v2.md](save-video-v2.md).

### Minimal wiring
```
Load Video ──IMAGE──► trigger_video ─┐
                                     Folder Version Incrementer ──filename_prefix──► Save Image / Video Combine
```
The front end reads the connected loader's file name and fills `source_filename` and `source_extension`. If they
arrive empty (an API run, an unknown loader), the node reads the queued prompt for a loader. With no source at all,
`label` is the folder name.

### Key controls
- `source_choice`: `auto` (video, then image, then `trigger`), `image`, `video`, `exr` (wire a plate path into
  `source_path`, no trigger needed) or `custom` (use `custom_name`).
- `source_path`: a full media path; it wins over `source_filename` and `custom_name`.
- `name_format`: `basename`, `strip_tags` (also drops trailing resolution / fps tags) or `first_segment`.
- `numbered_still_mode`: what a trailing number on a still means. `auto` strips 4+ digits as a frame number and
  keeps 3 or fewer as a shot number. `sequence` always strips. `identity` never strips. `####` and `%04d` are
  always stripped.
- `prefix` / `padding` (v / 3 -> `v001`), `date_format`, `base_path` (empty = ComfyUI output folder),
  `folder_name_override`, `path_style` (`auto`, `windows`, `linux`, `macos` separators).
- `suffix` + `suffix_mode`: `filename` (`v001/clip_mask.mov`), `subfolder` (`v001/mask/clip.mov`) or `folder`
  (`clip_mask/<date>/v001/clip.mov`).
- `versioning`: `folder` (default) or `filename` (`clip/<date>/clip_v001.png`, no version folder).
  `version_in_filename` also puts the version on the file in folder mode. `output_name` renames only the file.
- `reserve_version`: creates the version folder (or a hidden marker file in filename mode) when the node runs. Off
  (the default), nothing is created until the save node writes.
- `version_group`: leave empty. Incrementers in one run whose folders are the same shot already share one version,
  including their `_mask` / `_exr` variants. Give the same name only to outputs that belong together but have
  different folder names.
- `layout` `show / shot / task / version`: `<base>/<show>/<shot>/<task>/v###`, no date folder.
  - Uses `show`, `shot`, `task` and `min_version`.
  - With `reserve_version`, each version is claimed with a lock (retried `max_retries` times), and
    `version_manifest.json` (workflow hash, user, host, time) is written when `write_manifest` is on.

### Reading the outputs
- `version_string` (`v001`), `version_number`, `folder_name`.
- `subfolder_path` (`<folder>/<date>/v001`), `filename_prefix` (wire it to the save node), `output_filename`.
- `version_path`: the absolute path of the version folder.
- `source_stem`: the source name without extension or frame number.
- `current_version_token` / `next_version_token` / `next_version_stem`: when the source name already ends in a
  version (e.g. `shot_v001`), its token and the next one (`v002`, `shot_v002`). They are empty otherwise.
- `info_json`: folder, date, version, paths, `reserved`, and `shared_with_run` (true when another incrementer in
  this run already chose this version).

### Common mistakes
- ComfyUI temp and upload names (`ComfyUI_temp_…`, `clipspace/…`, `00001_.png`, `input_…`) are not used as source
  names; `label` is used instead.
- Save nodes still add their own counter in filename mode (`shot_A_v001_00001_.png`). Save Video (C2C) with
  Folder Version naming does not.
- `prefix`, `padding`, `date_format` and `base_path` must match on Folder Version Control, or it looks in another
  folder.

## Folder Version Variant (per output) (`FolderVersionVariant`)
**Menu:** 🐺 C2C/🧰 Core/Utils · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#folderversionvariant)

One render often writes several outputs: a mov, an EXR sequence, a PNG sequence, a masked mov. Put one Variant
between the incrementer and each save node. They all keep the same version and get their own names. The node only
rewrites the path text; it never touches the disk.

```
Folder Version Incrementer ──filename_prefix──► Variant (suffix "_exr") ──filename_prefix──► EXR save
                           ──version_string───►
```
- `mode`: `filename` (`shot_a/v004/shot_a_masked.mov`), `subfolder` (default, `shot_a/v004/masked/shot_a.mov`;
  use it for EXR sequences) or `folder` (`shot_a_masked/v004/shot_a.mov`).
- `prefix`: a tag before the name instead of after it.
- Wire `version_string` too, so a subfolder tag lands right after the version folder.
- `extension` is added to `output_filename` only.
- `report` says what was tagged and what stayed shared.

## Folder Version Control (check / set) (`FolderVersionControl`)
**Menu:** 🐺 C2C/🧰 Core/Utils · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#folderversioncontrol)

Reads or moves the version counter of `<base>/<label>/<date>/`. Put the folder name the incrementer uses in
`label`.
- `action` `check` reads only. `status` says how many versions exist and what comes next; `current_version` is the
  highest one (0 if none) and `next_version` the next.
- `action` `set` creates empty folders up to `value`, so the next run makes `value`+1. Above 999 only the folder
  `value` itself is created. If higher versions already exist they are left alone, and `next_version` and `status`
  say what will really come next.
- To really reset a label, delete its date folder from disk. This node never deletes.

## Folder Version Check (deprecated) (`FolderIncrementerReset`)
Hidden from the menu; kept so old workflows load. Use
[Folder Version Control](#folder-version-control-check--set-folderversioncontrol) with `action` `check`. Note that it
always assumes `v` and 3 digits. [NODE_REFERENCE](../NODE_REFERENCE.md#folderincrementerreset)

## Folder Version Set (deprecated) (`FolderIncrementerSet`)
Hidden from the menu; kept so old workflows load. Use
[Folder Version Control](#folder-version-control-check--set-folderversioncontrol) with `action` `set`.
[NODE_REFERENCE](../NODE_REFERENCE.md#folderincrementerset)

---

## Removed nodes

These ids used to be documented here. Where a node was merged, saved workflows load as the node named.

- Points Mask Editor (`PointsMaskEditor`) was merged into `ImageMaskEditorC2C` (mode `points + box`); see
  [image-mask-editor.md](image-mask-editor.md).
- Image Comparer (`ImageComparerMEC`) is no longer registered and has no automatic migration. Replace it by hand
  with Video Comparer (`VideoComparerC2C`), which compares images, video, EXR and audio.
- Universal Reroute (`UniversalRerouteMEC`) moved to the ComfyUI-NukeMaxNodes pack as "Universal Reroute / Dot
  (NukeMax)". Install that pack to keep using it.
- Batch Version Manager (`BatchVersionManagerMEC`) was merged into `FolderIncrementer` (layout
  `show / shot / task / version`).
- Dimensions Snap (`DimensionsSnapMEC`) was merged into `AspectPresetMEC` (preset `Custom (width x height)`).
- Image Batch Split (`ImageBatchSplitMEC`) and Video Frame Extractor (`VideoFrameExtractorMEC`) were merged into
  `ImageBatchSliceMEC`.
- Mask Area Probe (`MaskAreaProbeMEC`) and VAE Latent Inspector (`VAELatentInspectorMEC`) were merged into
  `ImageStatsProbeMEC`.

The full list is in [MIGRATION.md](MIGRATION.md).
