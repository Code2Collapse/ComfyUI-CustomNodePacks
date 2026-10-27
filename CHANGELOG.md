# Changelog

All notable changes to ComfyUI-CustomNodePacks are documented here.

## Unreleased – 2026-10-27 (Image Mask Editor modes: one node for every kind of mask drawing, L7.65 P01)

- **Image Mask Editor** gains `mode`: `paint` (as before, the default; plus a colour layer, `canvas_layer` =
  `colour`: the Paint Canvas, output `painted_image`), `points + box` (Mask Edit's points editor), `shape` (Mask Edit's
  12 parametric shapes, or JSON), `spline` (Spline Mask's spline, or a flow pattern along it), `roto` (Vector Roto) and
  `video` (Video Mask Editor). Each mode runs the former node's code and shows only its own controls and drawing UI
  (the editors are installed the first time their mode is picked). `image` is now optional (roto, shape, spline and
  the colour layer draw without it); everything new is appended, so saved Image Mask Editor nodes, their values and
  their links load unchanged. The node is at least 420 px wide, so no label is cut.
- Checked: 29 cases give the former nodes' outputs bit for bit (`tests/test_image_mask_editor_modes.py`); live on the
  owner's stack every mode shows exactly its controls and its editor, switching through all modes twice leaves no stray
  element, and spline / roto / shape / points / paint run, in classic and Nodes 2.0, at the root and in a subgraph.
- Fixed on the way: in Nodes 2.0 an editor shown a second time (Mask Edit points, Spline Mask) stayed invisible;
  keyframes drawn in Spline Mask's track canvas were never saved (it wrote to a widget the node does not have); Mask
  Track's `spline_track` now has that tracker canvas; a stale editor thumbnail threw after a workflow reload.
- The former nodes stay registered for now; they are removed (with automatic migration) in the next step.

## Unreleased – 2026-10-27 (Mask Tools and Mask Track gain the non-drawing modes of Mask Edit and Spline Mask, L7.65 P01)

First step of owner card P01 (one Image Mask Editor for every kind of mask drawing; design
`docs/design/image_mask_editor_modes.md`). The owner chose on 2026-10-27 where the modes that are not drawing go;
they are added now, and Mask Edit / Spline Mask stay registered until the editor's modes ship.

- **Mask Tools** gains `transform`: grow, blur and offset per axis, feather, threshold, invert - Mask Edit's
  `transform`, the same Mask Transform XY code. Its controls (`transform_*`) are optional inputs after the existing
  ones, so saved Mask Tools nodes and older API prompts are unchanged.
- **Mask Track** gains `spline_track` (Spline Mask's `track`, the same Lucas-Kanade spline tracker; card P07) and
  `bbox_smooth` (Mask Edit's `bbox_smooth`, the same smoothing code), with three appended outputs: `bboxes`,
  `primary_bbox`, `spline_data_out`. Existing outputs keep their slots.
- Checked: the new modes give the old modes' results bit for bit for the same values
  (`tests/test_consolidation_p01.py`, 19 cases); live on the owner's stack (core 0.39, frontend 1.53.10), each mode
  shows only its own controls and runs, in classic and Nodes 2.0, at the root and inside a subgraph.
- Migration rows can now pick their successor by a saved value (`c2c_by_value`), in the front-end migrator and for
  API prompts - needed when Mask Edit and Spline Mask are removed (their modes go to three different nodes).

## Unreleased – 2026-10-27 (merged to main: one Inpaint Crop and one Inpaint Stitch node, L7.65 P08)

Owner card P08: seven crop / stitch nodes become two. Saved workflows load migrated (links and values kept); the
pack registers 110 nodes instead of 115. Design: `docs/design/inpaint_crop_stitch_unified.md`.

- **Inpaint Crop Pro** gains `crop_mode`: `pro` (the crop as before, the default), `smart aspect` (the former **Smart
  Image Crop**) and `mask prepare` (the former **Inpaint Mask Prepare**: no crop; with an image it also gives a
  whole-frame stitcher). The new controls are shown only in their mode. A 6th output, `preview`: the preview picture
  in `pro`, Smart's overlay in `smart aspect`, the R / G mask picture in `mask prepare`.
- **Inpaint Stitch Pro** gains `stitch_mode`: `blend` (as before, the default), `paste back` (the former **Inpaint
  Paste Back**), `smart` (the former **Smart Image Stitch**) and `context blend` (the former **Context Inpainter /
  Fixer**: colour correction and lightness rescue, no stitcher; every frame of a batch and the colour reference
  inside / around the mask, as the node does since L9.39). A stitcher from the other
  kind of crop is refused with a message that names the mode to pick.
- Each mode runs the former node's own code. Checked: 37 cases through the migration rows give the old nodes'
  outputs bit for bit, against the code as shipped at d97cc13 (`tests/test_consolidation_inpaint.py`).
- The crop preview is drawn from what ran in every mode, and the live preview follows `crop_mode` and the new
  controls (checked against real runs from each starting mode). It now also follows a `roto_quality` change; before,
  it kept the last run's 1 px erosion setting.
- `image` on Inpaint Crop Pro and `stitcher` / `inpainted_image` on Inpaint Stitch Pro are optional inputs now (a
  migrated Mask Prepare has no image, Context Inpainter no stitcher); a mode that needs the input says so. Older
  saves keep every link on its input: on Inpaint Crop Pro every input keeps its slot; on Inpaint Stitch Pro the new
  `original_image` / `mask` sockets move the widget slots by two, and a link into one of them is moved with its
  input when the workflow loads (frontend 1.53.10 does not move it by itself; checked on a save with a link on every
  input, also inside a subgraph).
- Dropped in migration: Context Inpainter's `face_positive_prompt` / `face_negative_prompt` (they were only written
  to the log).

## Unreleased – 2026-10-24 (C2C Multi-GPU Split: batch split for any model, L9.38)

- **C2C Sequence Parallel is now displayed as C2C Multi-GPU Split** (same node id; saved workflows unchanged).
- **Any model** (Flux, SDXL, Qwen, ... - anything without a token-split adapter) gets the batch split: cond / uncond,
  image batches and Tile Split's tiles go to different GPUs. Before, such models were refused. ControlNet residuals
  and pooled inputs are cut with the batch. Measured on a tiny Flux through core's KSampler: equal to one GPU
  (<= 2.8e-5 at latent magnitude 26). Core's own CFG split uses at most two GPUs (one per cond).

## Unreleased – 2026-10-24 (fix: Load Video (C2C) workflows saved before the handle inputs)

- **Owner log: `LoadVideoC2C ... handle_frames, , invalid literal for int()`, and every output after it ignored.**
  A save made before the built-in handle inputs were added (L9.40) ends with the values of two front-end widgets:
  the upload button (`"image"`) and the hidden preview state (`""`). On load those landed by position on `handles` and
  `handle_frames`. The pack's generic migration only padded missing values at the end, so it missed this.
  Reproduced in both renderers on the owner-stack testbed. Evidence (workspace):
  `docs/evidence/L9.38/loadvideo_handles_probe.before_fix.json` and `loadvideo_handles_probe.json`.
- The generic migration (`js/widget_migration.js`, all pack nodes) now inserts new inputs' defaults **before** the
  trailing front-end widgets when an old save ends with them.
- A workflow that was already re-saved with the shifted values is repaired on load. A non-number in an INT / FLOAT
  widget, a non-boolean in a BOOLEAN, and a combo holding a front-end widget's value (such as `"image"`) go back to
  the default. The console names each one; save the workflow once to keep the fix. Nothing else is touched.

## Unreleased – 2026-10-24 (C2C Distributed: GPU Inventory, Sequence Parallel, L9.38)

- **New sub-package `c2c_distributed/`** (design: `docs/c2c-distributed.md`). The kill switch is
  `C2C_DISTRIBUTED=off`.
- **C2C GPU Inventory**: GPUs, architecture (Turing to Blackwell, incl. B200 / GB200 / B300 / GB10), BF16 / FP8 /
  FP4, kernels in this torch build, P2P / NVLink, Grace hosts, and NCCL / gloo usability measured in a subprocess.
  It ends with which multi-GPU modes this machine can run.
- **C2C Sequence Parallel (multi-GPU)**: exact Ulysses / Ring / USP sequence parallelism for Wan 2.1 / 2.2 and MiniMax
  H3 on one machine, on any OS. It uses ComfyUI core's multi-GPU model copies and thread pool, and
  core's block and attention hooks; no model code is copied. Measured equal to the single-GPU result: CPU float64
  attention <= 2e-15; GPU bf16 / fp16 bit-identical for Ulysses 2 / 4; sampler end to end <= 1.05e-5 (Wan) and <= 3.8e-6 (H3 video + audio). Speed-up on
  two or more GPUs is not measured yet. ControlNet, other block patches and attention overrides set after the node
  are refused with the reason instead of producing a wrong picture.
- **Batch split first**: a call's batch items (cond / uncond, image batches) go to separate GPU groups with no
  exchange; sequence parallelism runs inside each group (e.g. 8 GPUs with CFG = 2 groups x Ulysses 4). Measured equal
  to the unsplit call (sampler end to end <= 1.12e-5; GPU bf16 error vs fp32 0.686 % vs 0.687 % unsplit).
- For other models ComfyUI core's **MultiGPU CFG Split** (0.39+) does the cond / uncond split; it is not rebuilt.
- `TorchComm` (the torch.distributed path for NCCL boxes) checked over real separate processes (gloo on Linux, 2 and
  4 ranks): all_to_all now uses `all_to_all_single` (gloo has no list form), and point-to-point calls and broadcast
  map group ranks to global ranks (sub-groups were broken). Attention over processes <= 1.3e-15 from one process.

## Unreleased – 2026-10-24 (Scheduled Queue, Watch Folder, Webhooks; Video Stabilizer RAFT; Control AOV inputs, L9.39)

The OmniBar's Schedule and Watch panels had shipped, but their server side was never loaded, so every action failed
(404). Evidence (workspace): `docs/evidence/L9.39/automation_live.json` - owner-stack testbed, both renderers, a
subgraph.

- **Registered and rebuilt.** Runs are queued on this server's own address and port. They used to go to a fixed
  `127.0.0.1:8188`, which is wrong on any other port.
- **Scheduler:** weekday was off by one day (cron's Sunday = 0 was read as Monday). The two day fields now follow
  standard cron. Runs are counted by minute, and minutes missed while the server was busy are caught up. A live
  preview shows the next runs. **New seed each run** (default on) makes repeated runs render instead of hitting
  ComfyUI's cache. The loop runs only while an enabled job exists.
- **Watch Folder:** scans no longer block the server: they run on a worker thread, top folder only unless asked,
  capped at 20 000 files. Half-written files wait until they stop growing. *Run a workflow* puts the file into the
  first Load Image / Load Video. Before, it built a prompt that could never pass the check. *Copy to input* offers
  **Add Load Image**, which works inside subgraphs. Watching ComfyUI's own output folder with *Run a workflow* is
  refused, because it would feed itself.
- **Webhooks:** they now have a UI (Schedule panel › Webhooks). The completion event listened to
  `execution_complete`, which ComfyUI never sends, so it never fired; it now uses `execution_success`. Execution
  events are hooked only once an outbound URL exists. Secrets go to the C2C secrets store, never to the saved
  settings. Inbound calls can change literal inputs per run.
- Everything survives a restart (`user/default/c2c/automation.json`). Both panels have command-palette entries, so
  they open in Lite mode too. They keep refreshing while a prompt runs.
- **Video Stabilizer: `raft_flow` is real RAFT now** (torchvision, BSD-3). It used to run OpenCV DIS, and its
  `raft_iters` / `use_half` did nothing. The old behaviour is `dense_flow`, and `auto` and the deprecated nodes
  render as before. RAFT is **experimental**: on measured synthetic shake, feature tracking (`classic`) was the most
  accurate, and RAFT did worst on near-featureless plates. Numbers in `docs/video-stabilizer.md`. RAFT weights are
  used when installed, and downloaded only with `allow_download`. The node shows the manual and RAFT settings only
  when they apply.
- **Video Stabilizer: put the camera motion back.** A new `motion_meta` output holds the exact per-frame warps.
  Wire it into a second Video Stabilizer's `motion_meta` input and the edited steady plate gets its original shake
  back (the engine's exact inverse; round trip under 0.01 mean error on a smooth plate).
- **MaskOps: BackgroundMattingV2 takes a real clean plate** (`background_plate`). It always synthesised one from a
  blurred, masked copy before. When several detected objects are matted as one, `info` now says why.
- **Face Fixer: per-face prompts work.** Connect `clip` and each face's prompt is encoded and sampled. Before, the
  prompts only replaced a metadata key that sampling never reads. Without `clip`, `info` says they were not applied.
- **Context Inpainter: every frame of a batch** (it returned frame 0 only). New `color_reference = around mask` for
  object removal: colour match uses the surroundings within `crop_padding`, which finally has a use, instead of the
  removed object.
- **Spline Mask: cut shapes.** A shape can subtract from the ones before it (toolbar ＋/－ or `X`; drawn dashed and
  red), like a Roto stack. Old data is a union as before.
- **ProPainter: QC outputs.** `aux_image` is |output − input| ×4 and `aux_mask` is the changed pixels (flow mode: the
  unreliable region), and `info` reports how much changed outside the mask. They were zeros. Temporal's `mask_out`
  is now the region it inpainted.
- **Frequency Separate: preview on the node again.** After a run it shows the input, base and detail (or both
  side by side) from the node's own result. Drag `radius` or switch `mode` / `detail` and it follows live with a
  Gaussian approximation, labelled as such, until the next run.
- **Render farm:** a URL-mapped loader keeps its other inputs (it was left with the URL only), and cancelling a
  running job interrupts only that prompt (it was a global interrupt).
- **Control AOV: wire your own depth / canny / pose / normal / ID maps again.** The sockets had been dropped while
  the node still used them. A wired pass wins over its model selector, and its weight / invert controls appear when
  it is wired.

## Unreleased – 2026-10-23 (Save Video v2, built-in handles, Video Comparer v2; tests stay local, L9.40 + L9.37)

Owner request 2026-10-23 (answers: workspace `docs/evidence/L9.40/owner_answers_2026-10-23.md`). Evidence
(workspace): `docs/evidence/L9.40/` - owner-stack testbed, both renderers, a subgraph, every format combination
encoded and probed. Design: `docs/save-video-v2.md`.

- **Save Video: format in five controls** (container, codec, profile, bit depth, chroma), each filtered by the ones
  before it, from what this machine can write. Many new codecs and profiles, TIFF / DPX / JPEG sequences, and colour
  tags (primaries, transfer, range) that also drive the conversion. Old workflows encode exactly as before and open
  converted.
- **Save Video: naming fills itself** from the Folder Version Incrementer in the graph, or its defaults. The next
  path is shown on the node, and a version is used only by a real write.
- **Save Video: Load Video's player**, an export range from its In / Out, A/B against the incoming frames (wipe,
  difference, onion, flicker, zoom, loupe), and *preview first, then Write file*.
- **Handles built into Load Video and Save Video:** Load adds them (fit / add, model grid, audio in step), and Save
  trims exactly what Load added.
- **Video Comparer: real player**, B offset, side by side and a 2 x 2 grid (C / D inputs), zoom / pan / loupe, and
  per-frame PSNR / SSIM / CIEDE2000 under the scrubber. Previews keep every frame.
- Measured: PyAV 17's converter has no BT.2020 matrix (it uses BT.601 silently). BT.2020 through PyAV is therefore
  refused with a message; the ffmpeg path converts it correctly.
- **Tests stay local** (owner rule 2026-10-23): `tests/` and `renderfarm/tests/` are no longer in the repository, and
  CI boots the pack instead (`.github/c2c_boot_check.py`).

## Unreleased – 2026-10-23 (Video Mask Editor rebuilt; Load Video cached-None fix, L9.34-L9.36 + L9.16/L9.17)

Owner report 2026-10-23: "video mask editor looks poor and mask is not working as expected; the mask is sticking
throughout rather than frame-by-frame; issue with load video c2c". Evidence (workspace): `docs/evidence/L9.34/`,
`docs/evidence/L9.36/` (owner-stack testbed, 4K clip, both renderers, subgraph, server restart).

- **Video Mask Editor: masks stay on the frames you paint.** The node used to hold the first / last painted frame over
  the whole clip, so one painted frame masked all 8 frames of the test clip; now only that frame is masked. The new
  `outside_keys` (empty / hold) and the existing `tween_mode` (+ `none`) decide unpainted frames; `input_mask` is a
  per-frame base (it was reduced to its first frame).
- **The editor shows the mask the node outputs on every frame** (painted, interpolated, held, input mask, none), and
  painting on an unpainted frame starts from it.
- **Rebuilt editor:** full-resolution frames (it drew 384 px JPEG thumbnails stretched over a 4K frame), a mask canvas
  updated only where a stroke touched (a 40-move stroke at 4K: 3.2 s before, 0.69 s now, software rendering), tool
  rail, timeline with keyframes and interpolated spans, overlay / mask / matte views, onion skin, playback.
- **Keyframes are saved as you go, to disk** (`input/c2c_video_masks/<session>/`): a server restart no longer loses
  them (L9.16). *Revert* undoes the session's changes. The session id widget is hidden (L9.17).
- **Load Video (C2C): linking IMAGE later no longer hands a cached `None` downstream.** The loaders decode only the
  outputs something reads; IS_CHANGED now includes which ones are linked (L9.34).

## Unreleased – 2026-10-21 (Image batching Universal: Quantize Image runs frames in parallel, L2.54 slice 3)

Evidence (workspace): `docs/evidence/L2.54/quantize_adapter_probe.json` (identical in every case, 2.5-7.6x for the
operation), `quantize_live.json` (owner-stack testbed, 24 real 2048x1080 frames, Off vs Universal interleaved).

- **Universal runs core Quantize Image on several frames at once.** It calls the node's own code, one frame per call,
  on up to 8 threads. PIL releases the GIL while it quantizes, the frames do not depend on each other, and the result
  is the same to the bit. No core code is copied.
  - Whole graph: 3.6x (no dither), 4.4x (Floyd-Steinberg), 3.0x (Bayer 4).
  - The first run per frame size also times the original once.
- Test oracles are written independently of core's source, because this pack is Apache-2.0 and core is GPL-3.0.

## Unreleased – 2026-10-21 (Image batching Universal: Feather Mask adapter, L2.54 slice 3)

Evidence (workspace): `docs/evidence/L2.54/feather_adapter_probe.json` (48 bit-identity cases, timings),
`feather_live.json` (owner-stack testbed, Off vs Universal interleaved).

- **Universal now also speeds up core Feather Mask.** Core multiplies one column or row at a time in a Python loop.
  The adapter does one multiply per border strip, in core's order and with the same per-pixel arithmetic, so the
  result is identical to the bit (float32 and float64; half-precision and GPU masks run core's own code).
  - Operation alone, CPU: 1.2-26x faster.
  - Whole graph, 97 x 720p: 1.29x at a 64 px feather, 2.64x at 256 px. At 16 px there is no measurable change.
  - Like Grow Mask, it is checked against the original before first use and timed once per frame size.
    `C2C_AUTOBATCH_ADAPTERS=0` turns it off.

## Unreleased – 2026-10-21 (4K video path in Load / Save Video (C2C), L2.53 + L2.56)

Ported in part from HallettVisual/ComfyUI-4K-Video-Save-Load (MIT, see CREDITS.md). Evidence is in
`docs/evidence/L2.53/` (workspace):
- `v4k_live.json`: owner-stack testbed, classic, Nodes 2.0 and a subgraph.
- `v4k_save_live.json`: the NTSC rate, the workflow stored in the file, and drag-back.
- `video4k_bench.json`: speed.
- `HANDOFF.md`: the measurements and the open owner questions.

- **Load Video (C2C) / Load Video Path (C2C), new inputs.** They are appended, so saved workflows load unchanged:
  - `decoder` (auto / exact / fast). `auto` uses the exact reader: on this CPU the ffmpeg path was not reliably faster.
  - `seconds_cap`.
  - `resolution` (by the longest side) and `divisible_by`.
  - `audio_when_missing`.
  - an `info` output (JSON).
  - the **MiniMax H3** format preset (24 fps, 17n+5 frames, sizes in multiples of 32).
- **Settings you can read.**
  - Grey real values beside each widget (in classic), or chips in the panel (in Nodes 2.0), with reset arrows and
    *reset all to defaults*.
  - A details line that agrees with the player's "N will load" and with the run (checked live).
  - A new clip resets the settings; reopening a workflow does not.
- **Fixed: a typed `frame_load_cap` under a frame grid.** It snapped to the wrong count: 60 became 73 under H3.
  The frontend rounds INT values to multiples of `step2` before the node sees them. The arrows now step through the
  legal counts (H3: 0 → 5 → 22 → 39), and a typed count snaps to the nearest legal one (60 → 56).
- **Fixed: a frame editor fed by Load Video (C2C)** now sees `seconds_cap` / `resolution` / `divisible_by`.
  Before, it planned the uncapped selection.
- **Save Video (C2C).**
  - An ffmpeg encode pipe. Measured on CPU against the PyAV path: H.264 / H.265 / ProRes / FFV1 are 2-4x faster
    (1080p H.264 66 → 26 ms/frame on a quiet machine); DNxHR and VP9 are about 1.2x.
  - New formats: H.265 8-bit (CPU), and H.264 / H.265 / H.265 10-bit / AV1 on NVENC (NVENC is not measured on a
    GPU yet).
  - Exact NTSC rates (23.976 is stored as 24000/1001).
  - `save_metadata`: the workflow and prompt are stored in MP4 / MOV / MKV / WebM the way core does, so dropping the
    file back into ComfyUI rebuilds the graph. Like core, nothing is stored when ComfyUI runs with
    `--disable-metadata`.

## Unreleased – 2026-10-16 (AV Handles: the model preset and custom formula now apply in add mode)

Evidence: `docs/evidence/L2.55/` (workspace): `handles_before.json` / `handles_after.json`, `handles_live_after.json`
(26 cases, classic + Nodes 2.0, owner-stack testbed), `handles_add_custom.*.png`.

- **Fixed: in add mode the `model` preset and the custom step / plus did nothing.** Add only ever rounded onto the
  fixed `padding_mode` table, whose default is `disabled`, so with any preset or custom formula the frame count came
  out unchanged (reported for every preset). `padding_mode` has a new first choice, **model preset**: the total is
  rounded up onto the grid of the `model` dropdown, or onto `custom_step * n + custom_plus`. It is the default for new
  nodes. Saved workflows keep the `padding_mode` they stored (`disabled`, `WAN`, `LTX2`, `H3`), so their frame
  counts do not change; the report says how to switch. The front end now shows `model` (and step / plus for custom)
  in add mode when `padding_mode` is model preset.
- The run report survives saving and reopening the workflow: ComfyUI does not re-send the report of a cached node.
- Tests: every preset x clip lengths x handle counts x head / tail / both, custom formulas, fit to a target, already
  compatible counts, and the old padding tables - each checks the real output frame count, that the original frames
  keep their order and content, and that the trim returns the original exactly.

## Unreleased – 2026-10-15 (Image batching Universal: proven faster operations)

Evidence: `docs/evidence/L2.54/` (workspace): design, `growmask_adapter_probe.json`, `growmask_live.json`.

- **Universal now replaces slow operations with proven faster versions, where it is measurably faster.** First one:
  core Grow Mask, which loops over every frame and every expand step through scipy. C2C's version does each step once
  over the whole batch with the same 3x3 maximum / minimum. Before first use it is checked against the original
  node on small masks covering the borders and both corner styles; the results must be identical to the bit, or it
  stays off (reason in `/c2c/autobatch/status`). The first call per frame size times both on at most four frames and
  keeps the faster; tiny masks keep the original. A changed core Grow Mask is never replaced until checked again.
  Measured, whole graph, 97 x 1280x720, CPU: expand 8 square 9.2 s -> 3.2 s, expand 8 tapered 5.1 s -> 2.6 s,
  expand -3 2.2 s -> 1.3 s. Off and Internal are unchanged. `C2C_AUTOBATCH_ADAPTERS=0` turns replacements off.
  Not measured on a GPU (the testbed is CPU-only).
- **See what Image batching did.** A node gets a small badge when Image batching changed how it ran (a faster
  version was used, or the batch was split into chunks): the classic title bar and the Nodes 2.0 header both show
  it, and the next run clears it. The C2C Diagnostics sidebar has a **Batching** tab listing those nodes for the
  last run (click a row to select the node). Nothing polls: the server sends one message per changed node.

## Unreleased – 2026-10-15 (one Matte node, L7.65 P05)

Design: `docs/design/matte_unified.md`. Tests: `tests/test_consolidation_p05_matte.py`.

### Changed: three matte nodes are now backends of Mask + Matting

- **Mask + Matting** (`MaskOpsMEC`) has a new `backend` input: `segment + matte` (the default - the node as it was),
  `SAM + ViTMatte`, `SeC + MatAnyone2` and `remove background`. The three added backends run the former nodes' own
  code with the same arguments; their files are unchanged (a test pins them byte for byte).
- **SAM + ViTMatte Pipeline, SeC + MatAnyone2 Pipeline and Background Remover** are no longer in the node list. Saved
  workflows migrate into Mask + Matting when they load: links and values move, nothing is dropped (three rows in
  `nodes/_legacy_replacements.json`, listed in `docs/MIGRATION.md`). Measured: 35 cases feed the old node's inputs
  through its row and compare every mapped output with the old node's own - all tensors and numbers bit-equal; `info`
  compared as JSON (two keys added, SAM + ViTMatte's `elapsed_sec` is wall-clock time). 23 cases use deterministic
  stand-ins for the models that record every call (the migrated node makes the same calls with the same arguments);
  12 run the model-free paths for real. Breaking any one mapped input in the table (41 tried) fails at least one case.
- **New output `foreground`** (the 19th; the first 18 keep their names and order): the image times the final alpha,
  premultiplied. With `premultiply` on it is the same tensor as `preview`, so it costs no memory there.
- **New input `split_masks`** (default on). Off leaves `edge_mask` / `inside_mask` / `outside_mask` empty without
  using memory. Measured on this CPU: that split takes 3.6 s per 1080p frame.
- **Migrated nodes cost what they cost before:** diagnostics and the split masks are off on them, because the old
  nodes never made them. Measured: a migrated SeC + MatAnyone2 or Background Remover returns no extra full-size
  tensor; a migrated SAM + ViTMatte adds only the new foreground. Outputs a backend does not make are empty views that
  take no memory, and the image is passed through, not copied, when despill is off.
- In the three added backends, `preview`, `score`, `edge_mask` and (SAM + ViTMatte) `bbox` keep the former node's
  definition, so migrated workflows give the same results; SAM + ViTMatte's `bbox` is still `[x, y, w, h]`.
- With one of the three added backends chosen, the node shows that backend's controls, the shared finishing controls
  and the sockets; on `segment + matte` the new controls stay folded. A new interface for this node is planned
  separately.
- Kill switch: `C2C_LEGACY_MATTE_NODES=1` registers the three former nodes again; old workflows then load them as
  they were.
- Example `sam_vitmatte_pipeline.json` is migrated with the same row (its thumbnail still shows the old node).

Not verified yet: runs with the real models (SAM, SeC, MatAnyone2, ViTMatte, RMBG-2.0, BiRefNet), and the live
browser check in both renderers and inside a subgraph.

## Unreleased – 2026-10-15 (Load Video player and crop; GPU sync per node removed; speed measured)

Evidence: `docs/evidence/L2.51/speed_summary.md`, `docs/evidence/L2.53/`, `docs/evidence/L2.52/` (workspace), on the
owner-equivalent testbed (core 0.39.0, frontend 1.53.10, the owner's third-party packs, CPU).

### Fixed: a GPU wait after every node

- **Insight waited for the GPU twice per node.** Its timer around every node called `torch.cuda.synchronize()` before
  and after the node whenever CUDA was available, so the next node could not start its CPU work while the GPU was
  still busy. It no longer syncs. Its VRAM numbers are unchanged (they never needed the sync); its node time is now
  the time the node took to run on the CPU side, GPU launches included. `C2C_INSIGHT_SYNC=1` brings back exact GPU
  times for profiling; `C2C_INSIGHT=0` turns the hook off. Not measured on a GPU yet (the testbed is CPU-only).

### Measured: image nodes are as fast with this pack as without it

- 12 common nodes (Mask to Image, Image to Mask, Invert, Upscale By, Blur, Sharpen, Grow Mask, Threshold Mask, core
  Canny, KJ Grow Mask With Blur, KJ Resize Image v2, controlnet_aux Canny), 97 frames of 1280 x 720, timed
  interleaved on two servers: this pack removed, and loaded with image batching Off, Internal and Universal. No
  difference rose above the machine's noise. Table: `docs/evidence/L2.51/speed_summary.md`.
- If a log still says `[C2C autobatch] install failed: unhashable type`, that ComfyUI runs code older than d97cc13.

### Added: Load Video (C2C) player

- Play backwards, speed 0.25x-4x, loop mode (selection, whole clip, once, ping-pong), J / K / L shuttle keys,
  Shift+Left / Right for 10 frames, Up / Down for speed.
- A ruler under the scrubber, in frames or in time. Hovering the scrubber shows that frame in a thumbnail.
- Click the frame number to type a frame or a timecode.
- The label under the bar also says how long the selected frames last. *all* selects the whole clip.
- Full screen (⛶ or F).
- The preview now has a keyframe every 12 frames (it had x264's default of 250), so seeking, hovering and playing
  backwards decode at most 12 frames. Older cached previews are rebuilt once.

### Added: crop in Load Video (C2C) and Load Video Path (C2C)

- A crop box over the preview with eight handles and shapes Free, Original, 1:1, 4:5, 5:4, 16:9, 9:16, 4:3, 3:4, 3:2,
  2:3, 2:1, 1:2 (behaviour modelled on WhatDreamsCost's Load Video UI; no code taken). The label shows the crop in
  pixels; *no crop* removes it.
- The run decodes each frame, cuts the box out, then applies `custom_width` / `custom_height`. The cropped frames are
  exactly a slice of the uncropped ones (tested on video and image sequences).
- Stored in four new inputs, `crop_x`, `crop_y`, `crop_w`, `crop_h` (fractions of the frame), appended after the old
  ones and hidden (the crop tool sets them). Old workflows load with no crop.

### Fixed: every C2C panel inside a new subgraph in Nodes 2.0

- Convert to Subgraph creates a new node with the same id; Nodes 2.0 kept showing the old node's panel, wired to a
  node that no longer runs. Measured over every node type of the C2C packs (660 types, 361 with panels): classic
  362 of 362 fine; Nodes 2.0 313 showed the dead copy before, 1 after (Folder Incrementer, whose status strip is added
  later; fixed by tagging a panel the moment a C2C node adds it, re-checked). The remaining "not attached" entries are
  core's multiline text boxes, which Nodes 2.0 draws itself. `js/_vue_dom_rebind.js` tags the panels of every C2C
  node and puts the live one back after each graph switch.

## Unreleased – 2026-10-14 (speed fixes; a real player in Load Video)

Evidence: `docs/evidence/L2.51/` and `docs/evidence/L2.50/` (workspace), measured on the owner-equivalent testbed
(core 0.39.0, frontend 1.53.10, the owner's third-party packs).

### Fixed: slow runs

- **Convert Mask to Image (and other nodes) were slow because of this pack.** The tensor inspector computed exact
  statistics over every output of every node on every run. On the expanded view that Mask to Image returns, it walked
  every element in Python. Measured:
  - one 16-frame 720p output: more than 60 s before, 0.0000 s now;
  - a normal 97-frame 720p image output: 0.42 s per node before, 0.0001 s now;
  - a real graph (97 × 720p: Empty Image → Image to Mask → Mask to Image → one frame → Preview): more than 14.5 min,
    unfinished, before; 0.42–1.13 s now.

  The inspector now records only shape, type and device while a workflow runs. It computes the statistics when you
  open its panel, for that node only. It never keeps an output in memory. `C2C_TENSOR_INSPECTOR=0` switches it off
  entirely.
- **Image batching stopped installing on many setups.** Packs such as Inspire, Impact and KJNodes use a wildcard
  socket type that cannot be hashed (`AlwaysEqualProxy`, `AnyType`). The first such node made the whole install fail
  ("install failed: unhashable type"), so no node was batched. The type test is now safe. A node that cannot be
  analysed is skipped on its own, without stopping the others. *Internal* mode no longer analyses other authors' nodes
  at all.
- **Canny on the CPU** is no longer split into 1-frame chunks. That rule protects GPU memory. On the CPU it only cost
  time: 2.64 s against 2.10 s for 8 frames. On a GPU the measured 1-frame chunks stay.

### Added: a real player in Load Video (C2C)

- Play / pause, previous / next frame, first / last frame, and a frame number with timecode.
- A scrubber over the whole clip. Drag to seek.
- In / out handles on the scrubber choose the frames to load. They set `skip_first_frames` and `frame_load_cap`; "61
  will load" is the exact number the run loads. *Loop the selection* plays only the chosen range.
- Keys while the player has focus: Space, Left / Right, Home / End, I / O.
- Also in Load Video Path (C2C) and Load Images Path (C2C).
- Checked live in the classic renderer and Nodes 2.0, after save and reload, and inside a subgraph. The frames each
  run loaded matched the count the player showed (61, and 97 inside the subgraph).

### Fixed: subgraphs in Nodes 2.0

- **A C2C node moved into a new subgraph showed a dead copy of its panel.** Convert to Subgraph creates a new node
  with the same id. Nodes 2.0 kept showing the old node's panel, wired to a node that no longer runs. C2C now puts the
  node's own panel back after every graph switch (`js/_vue_dom_rebind.js`). The Load Video loaders use it first.
- **Nodes 2.0 showed "Image failed to load" under Load Video.** Frontend 1.53 previews the source file in a panel that
  the old hide rule did not cover.

### Added: front-end off switch

- `C2C_NO_FRONTEND=1` makes ComfyUI serve none of this pack's browser code. The nodes keep working. One restart tells
  whether a browser problem comes from this pack.

## Unreleased – 2026-10-14 (every C2C canvas tool works inside subgraphs)

Evidence: `docs/evidence/L2.47/sites.md` (every site and its rule) and `subgraph_sweep_live.json` (workspace);
tests/test_subgraph_walk.py.

### Fixed

- **C2C tools assumed the root graph.** About 50 places in 29 files used `app.graph`, which is always the root. Inside
  a subgraph they acted on the wrong graph, found no node, or missed nodes. Now:
  - tools that act on what you are looking at use the graph on screen: align, command palette, AI-explainer insert,
    PNG load, image drop, Pack to Single Node, group presets, TCL copy / paste, wire-on-drop;
  - tools that follow a node the server names find it inside (nested) subgraphs, including composite ids such as
    `12:5`: AI diagnose, cost estimator, flame graph, Doctor fixes and focus, farm progress, workflow doctor,
    bookmarks, `#id` expressions;
  - whole-workflow tools see nodes inside subgraphs: Surprise Me, Find, Wizard, settings recommender, parameter
    memory, Magnific sync / migration, expression fields, linked values, complexity HUD;
  - focusing a node inside a subgraph opens that subgraph first.
- **Undo panel** takes a snapshot when you edit inside a subgraph. It hooked only the root graph's change events;
  measured A/B: 0 snapshots before, 1 after, for the same edit (L2.48).
- **Sticky notes and wire labels** are stored on the graph they were made on. A note made inside a subgraph used to
  show on every graph, and a label could land on an unrelated root link with the same id.
- **Frontend 1.52+ keeps `graph.links` in a Map**, and `Object.values()` of a Map is empty. Because of that:
  - wire labels could not draw at all;
  - graph health saw no wire;
  - group presets saved no links;
  - the complexity HUD counted no links.

  All four now read the Map.

## Unreleased – 2026-10-14 (C2C Vault and copy / paste work inside subgraphs; the vault can be filled from the node)

Evidence: `docs/evidence/L2.45/vault_subgraph_repro.before.json` / `.after.json` (workspace, live on the owner stack);
tests/test_vault_flatten.py, tests/test_c2c_runtime.py.

### Fixed

- **C2C Vault inside subgraphs.** A vault locked inside a subgraph lost the wire to the subgraph's output (its IO
  nodes are not found by `getNodeById`), so the run never started. Boundary wires to a subgraph's own inputs and
  outputs are now kept and reconnected. Measured: locked inside a subgraph, run → success.
- **C2C Vault with a subgraph node in the selection.** The payload named the subgraph's definition UUID as a class
  the server cannot run. Subgraph nodes, nested ones included, are now flattened into the vault with their wiring
  and the values typed on them. Reroute / Primitive helpers are resolved, not stored. Boundary names are unique (two
  `image` inputs used to overwrite each other).
- **Copy / paste with metadata inside subgraphs.** Explicit paste (Ctrl+Alt+V) landed on the root graph and copy read
  the root graph's wires. Both now use the graph you are looking at. Measured: pasted into the subgraph, wires kept.
- **C2C features switched off by core errors.** A core error while drawing a node that had just left its graph
  (subgraph conversion, vault lock) was counted against the C2C draw patches. After five, hints, colourspace badges,
  isolate and graph health were removed for the session, and core's method ran a second time. Core's own errors now
  pass through unchanged and are not counted.

### Added

- **Finding the vault.** "C2C Vault → Lock / Seal selected nodes" in the node right-click menu and as selection
  toolbox buttons (it was only in the empty-canvas menu). An empty vault node gets a **capture** input with
  **Lock what's connected** and **Lock selected nodes** buttons, so you can wire what to protect straight into it.
- README: "C2C Vault: lock or seal part of a workflow".

## Unreleased – 2026-10-13 (Inpaint Crop Pro shows the crop that runs; Stitch Pro round trip)

Evidence: `docs/evidence/L7.99/` (workspace); tests/test_inpaint_crop_geometry.py.

### Fixed

- **Inpaint Crop Pro: the crop window stays on real pixels.** Near a border the window was padded with smeared edge
  pixels although a window of the same size fits inside the image (41 % of a 16:9 window at the left border). A
  window that fits is now moved inside the image; only a window larger than the image is padded. Behaviour change:
  crops of masks near the border move inward.
- **An empty mask stitches back unchanged** (it changed the image by up to 0.033 through the stitch feather).
- **Inpaint Stitch Pro changes nothing outside `blend_mask_used`** (Laplacian and frequency blends changed pixels up
  to 0.0021 where the blend weight is 0).

### Changed

- **The crop preview is the crop that ran.** The old preview band drew a made-up centred mask. After a run the node
  now shows one picture: the original with the discarded area dimmed, the regenerated area tinted and the window
  outlined, next to the crop the model gets, with the window, output size, scale, padding and batch handling written
  under it. New optional `preview_frame` (default 0, added last; saved workflows unchanged) picks the frame shown -
  display only.
- The node returns ComfyUI's `{"ui", "result"}` form; `info` also gives frame 0's window.
- **The crop preview follows a changed setting without a run**: after one run, changing a crop setting redraws the
  picture from that run's masks with the node's own geometry (exact; checked against real runs for 17 settings).
  Setting C2C > Canvas > Inpaint crop > Live crop preview (on). 81 x 1080p clip: 3.4 s per refresh on CPU.
- Mask expand / blend / dilation in the inpaint suite is now separable: bit-identical, up to ~30x faster on CPU
  (`mask_blend_pixels` 32 on a 1080p frame: 57 s -> 2 s).
- Outpaint extension padded the box mask by an amount computed from the already-extended size, so the crop window
  was measured on a mask larger than the image; both now use the same padding.
- **Inpaint Crop Pro shows 9 controls instead of 28.** The others are in sections (classic: the More bar; Nodes
  2.0: Show advanced inputs); a switch shows the controls it enables. Order, names and defaults unchanged.
  `output_padding` is no longer hidden when resizing is off (it applies either way).

## Unreleased – 2026-10-13 (Nodes 2.0 shows a loaded workflow's own values)

Evidence: `docs/evidence/L2.43/` (workspace); tests/test_widget_value_accessor.py.

### Fixed

- **Nodes 2.0 showed default widget values after a saved workflow loaded** (the node held and ran the saved values,
  the inputs displayed the defaults). Parameter Memory replaced every widget's `value` accessor with its own copy,
  which cut it off from the value store Nodes 2.0 draws; it now wraps the accessor. Parameter Memory still records
  every change.

## Unreleased – 2026-10-13 (model weights are downloaded only with permission)

Evidence: `docs/evidence/L7.97/` (workspace); tests/test_download_policy.py.

### Changed

- **No node downloads model weights unless you allow it.** Background Remover, Semantic Segment, SAM Multi-Mask
  Picker, SeC + MatAnyone, SAM Mask Generator (its text grounding model), Mask Placement, LocateAnything and SAM
  Model Loader each get an `allow_download` switch, off by default and added after the existing inputs (saved
  workflows keep their values). MaskOps' existing `auto_download` now also covers its backends' own Hugging Face and
  torch.hub loads, which used to fetch weights regardless. Before anything is called missing, an installed copy is
  looked for in the node's own folder, every model folder ComfyUI knows (other packs' too), another pack's copy of a
  Transformers model (`models/<anything>/<repo name>`) and the Hugging Face cache - and reused. A permitted download
  goes into the model's own subfolder; the folder is created only then.
- When a model is missing and downloading is off, the node runs its labelled model-free method where one exists
  (Background Remover, Semantic Segment for `skin`, SAM nodes, SeC + MatAnyone) or says plainly that the model is
  needed. The message names the absolute folder and the expected file name.
- Every node with an `allow_download` switch reports a `model_status` line in its info output (SAM Model Loader in
  its SAM_MODEL): `loaded - found at <path>`, `needs permission - model-free result` (missing, downloading off),
  `missing - model-free result` (missing for another reason), or `not used (source_mask wired)` on Mask Placement.
  Background Remover's earlier `missing - model-free result` now says `needs permission …` when that is the reason.
- SAM Model Loader reuses a checkpoint another pack keeps (any folder ComfyUI knows) instead of downloading a
  second copy for a `[download] …` choice.
- **Behaviour change:** a saved SAM Model Loader choice `[download] …` no longer downloads by itself; tick
  `allow_download` (owner decision 2026-10-12).

### Fixed

- LocateAnything (node and MaskOps backend) answered a missing model with transformers' generic "couldn't connect"
  text. It now looks in `models/LLM/LocateAnything-3B`, any `models/*/LocateAnything-3B` and the Hugging Face cache
  (a cache holding only the tokenizer no longer counts as installed), names the absolute folder when the model is
  missing, and a permitted download goes into `models/LLM/LocateAnything-3B` instead of the Hugging Face cache.
- MaskOps' other Transformers models (ViTMatte, BiRefNet, RMBG, InSPyReNet, DIS, Grounding-DINO, DINOv2
  re-anchoring) did the same; they now share that look-up, message and download folder (`models/<backend
  folder>/<repo name>`).
- MaskOps SAM 3.1 messages named a relative `models/sam3.1/` (and the import hint the wrong folder, `models/sam3/`).
  Preset-not-installed, checkpoint-missing and library-missing messages now give the absolute folder and the file.
  `backend_first_root` no longer raises on a `folder_paths` without `models_dir`.
- SeC + MatAnyone downloaded SAM 2 without asking: its segmentation loader ignored its own `[download]` flag and
  fetched any missing checkpoint, creating the model folder first.
- `unified_segmentation.py` crashed at import wherever `folder_paths` had no `base_path` attribute.

## Unreleased – 2026-10-12 (the SAM nodes work without a SAM model, and say so)

Evidence: `docs/evidence/L7.91/inventory.fresh.after_l798.json` (workspace); tests/test_sam_model_free.py.

### Fixed

- **SAM Model Loader with no SAM model** (none placed, the `sam2` package missing, or no internet for a
  `[download]` entry) stopped every SAM node behind it. It now hands out a model-free stand-in, and the nodes that
  take it - SAM Mask Generator, SAM + ViTMatte, SAM Multi-Mask Picker, Mask Placement, the image mask editor's smart
  select, Mask Tracker's `sam2_video` propagation - work through GrabCut (Mask Tracker: optical-flow propagation).
  Their `info` output says `"model_type": "model-free"`, `"approximate": true` and why. A text prompt with no point
  or box is refused with a plain reason (the stand-in cannot read text). `C2C_MODEL_FALLBACK=off` keeps the error.
- `nodes/model_manager.py` crashed at import wherever `folder_paths` had no `base_path` attribute.

## Unreleased – 2026-10-12 (MaskOps works without models, and says so)

Evidence: `docs/evidence/L7.91/` (workspace); tests/test_mask_model_free.py. Design:
docs/research/l791_masking_without_weights.md (workspace).

### Fixed

- **MaskOps produced nothing on a fresh install.** With no SAM 3.1 library, no weights or no internet, every
  pipeline stopped with "SAM 3.1 library is missing ... Reinstall ComfyUI-CustomNodePacks" (reinstalling cannot
  help: that library is not part of this repo). It now falls back to a model-free method - GrabCut on the prompted
  frame, DIS optical flow plus GrabCut for the rest of a clip, a guided filter for the soft edge - and **labels the
  result**: the `info` output gets `"approximate": true` and `"model_free"` (what was asked for, why it could not
  run, what ran instead), the segmenter/matter fields name the model-free method, and the console says so. A
  model-free result is never reported as SAM. Text prompts still need a model; without one the node says so instead
  of guessing. To get the old error instead of an approximation, set the environment variable
  `C2C_MODEL_FALLBACK=off`.
- The missing-SAM-3.1 message now says what is missing, where it is looked for, where its weights go, its licence,
  and that points / a box / a mask work without it.
- `positive_coords` documents `[[x,y],...]`; that form was silently ignored (no points found). It is read now.
- No model folders are created in `models/` at start-up any more (16 empty folders appeared on every install); a
  folder is created only when weights are saved into it.

### Added

- `classic` segmenter and `guided_filter` matter in MaskOps: the model-free methods, also selectable directly. They
  are listed after the existing ones; no default changes.

## Unreleased – 2026-10-12 (ProPainter is never downloaded)

### Fixed

- **The pack no longer downloads ProPainter.** On every fresh install it fetched the ProPainter source from GitHub
  into `third_party/` the first time ComfyUI started, and the ProPainter nodes downloaded its weights on first use.
  ProPainter is non-commercial (S-Lab License 1.0), so whether to use it is your decision: the nodes now use it only
  when you already have it (environment variable `C2C_PROPAINTER_DIR`, or `third_party/ProPainter`) and the weights
  are in `models/propainter/`; otherwise they say what is missing and how to add it. Nothing is created in your
  models folder at start-up any more. See docs/propainter.md.

## Unreleased – 2026-10-12 (Folder Version Incrementer: filename versioning)

Evidence: `docs/evidence/L7.94/`; tests/test_folder_incrementer_filename_versioning.py.

### Added

- **Filename versioning** (`versioning` = filename): the version goes on the file instead of a folder -
  `shot_A/10-12-2026/shot_A_v001.png` instead of `shot_A/10-12-2026/v001/shot_A.png`. The number is the next one no
  file of that name has yet, whatever wrote it: a Save Video (C2C) "Folder Version" file or image-sequence folder,
  core Save Image's `shot_A_v002_00001_.png`, VHS's `shot_A_v003_00001.mp4`, any extension, and also any `v###`
  folder from folder versioning, so switching methods never reuses a number. A version already on the name is
  replaced, not doubled (`shot_A_v003` -> `shot_A_v001`).
- **Version in the file name** (`version_in_filename`) for folder versioning:
  `shot_A/10-12-2026/v001/shot_A_v001.png`.
- **Output name** (`output_name`): renames the file only; the folder keeps the source-derived name
  (`final_render` -> `shot_A/10-12-2026/v001/final_render_v001.png` with the option above). It may carry its own
  extension.
- The node's status line shows the file name a run will write when any of these is on.

Folder versioning stays the default. With the new options off, every output is the same as before: compared against
the previous version on 2016 input combinations, no difference. Saved workflows keep their values (the new inputs
are added after the existing ones).

## Unreleased – 2026-10-12 (Inpaint Stitch with an RGBA plate)

### Fixed

- **Inpaint Stitch Pro crashed when the original plate had an alpha channel** and the sampler returned RGB
  ("The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 3", from the owner's
  Linux run). The colour is now inpainted and the plate keeps its own alpha; an RGBA result on an RGB plate drops
  the alpha. Any other channel mismatch gets a plain message instead of a tensor error.

## Unreleased – 2026-10-12 (Video Stabilizer works on a fresh install)

Evidence: `docs/evidence/L7.90/exec_stabilizer.json`; tests/test_video_stabilizer_engine.py.

### Fixed

- **The four Video Stabilizer nodes failed on every fresh install** ("not vendored"): their engine was read from a
  folder that is not part of this repo. The engine (MIT, nomadoor/ComfyUI-Video-Stabilizer) now ships in
  `nodes/video_stabilizer/`. It needs no model weights: both methods are OpenCV (point tracking / DIS optical flow);
  on a synthetic shaky clip the frame-to-frame shake drops by more than half.
- The stabiliser's padding mask could never be returned (the engine hands back an array, the node expected a list).

## Unreleased – 2026-10-12 (image batching measured per preprocessor)

Evidence: `docs/evidence/L7.82/per_preprocessor_summary.json`, `canny_chunk_curve.json`, `canny_curated.json`.

### Fixed

- **Core Canny is now batched by default.** Run on 96 frames of 720p in one call it spilled past an 8 GB card and
  took 17 minutes (1042 s); the default Internal mode left it alone. It now runs one frame at a time in every
  batching mode: 10.4 s, identical edges. (Its edge tracing also gets slower per frame the more frames it gets at
  once - 0.107 s per frame alone vs 0.248 s in a 32-frame call - so single frames are fastest even when memory is
  plentiful.)
- Measured on the other preprocessors (same 96 frames): ControlNet Aux Canny and Depth Anything now run as one call
  (1.13 s and 8.30 s, previously split into 3 chunks, which built Depth Anything's model 3 times); DWPose is no
  longer touched by batching (5.03 s; previously a test run, an abort and a full run).

## Unreleased – 2026-10-12 (image batching hands on the complete batch, in order)

Evidence: `docs/evidence/L7.82` (scramble_scan.json; tests in tests/test_autobatch.py).

### Fixed

- **Batched nodes passed the wrong values downstream.** When a batch was split, the node's outputs were put back
  as "all images first, then everything else". A node returning image, width, height, mask (Kijai's Resize Image
  v2 shape) came back as image, mask, width, height, so the next node got a mask where it expected a number. 29 of
  the 469 nodes Universal could wrap had that output layout. Outputs now keep their own order.
- **Image batching leaves nodes alone that return anything besides images and masks** (counts, sizes, info text,
  pose keypoints, ...): only frame tensors can be stitched back together. 148 node types are no longer wrapped,
  among them Kijai's Resize Image v2 (which batches internally itself) and controlnet_aux DWPose.

### Changed

- **Double-click a slot = suggest and connect a node** (owner's choice): it connects a matching node nearby or
  inserts the predicted next node (Shift = a chain). Get/Set variable and Nothing stay in Settings › C2C ›
  Productivity › Slot double-click.

## Unreleased – 2026-10-12 (wire snapping matches ComfyUI again)

Evidence: `docs/evidence/L7.83` (snap_matrix.json: 24 core, third-party and C2C nodes, every socket input, five drop
points, C2C vs C2C disabled, with the owner's 17 third-party packs loaded).

### Fixed

- **Wires locked onto the wrong slot.** Holding a wire over a slot of a different type, ComfyUI shows no lock-in
  and connects nothing; C2C's "Wire drop snaps to the matching slot" locked onto the first matching slot instead,
  so a wire aimed at `negative` could connect `positive`. Over real slots C2C now does exactly what ComfyUI does
  (0 differences in 148 measured drags). The setting still helps over widget rows, where ComfyUI 1.52 gives up
  (three C2C nodes measured: Context Inpainter, OCIO Bridge, NukeMax Crop).

## Unreleased – 2026-10-12 (image batching no longer slows nodes down)

Evidence: `docs/evidence/L7.82` (before.json, after.json, exec_ImageFlip.json; design
`docs/research/autobatch_overhead.md`).

### Fixed

- **Image batching made nodes slower.** In the automatic modes (Internal / Universal) a batch was split into chunks
  even when it fitted in memory, and every extra call repeated the node's setup (nodes that build a model or
  detector each time they run). Measured on an 8 GB laptop GPU, 96 frames at 720p: a per-frame OpenCV node went
  from 0.82 s to 0.91 s, the same node with a 1 s setup from 1.82 s to 5.53 s. Now a batch that fits runs as one call
  (0.81 s and 2.19 s), and when a batch really does not fit, chunks are made as big as memory allows so the setup
  stays under 10 % of the run. Core Canny, whose full batch spills past the card (238 s), still gets chunked:
  3.1 - 3.8 s, identical output.
- Free VRAM was over-counted (PyTorch's cached memory twice, and memory that had spilled into system RAM).
- Your own limits (`budget_mb`, `max_frames`, per-node entries) work exactly as before. Off switch: Settings › C2C ›
  Image batching › Off, or `C2C_AUTOBATCH=0`.

## Unreleased – 2026-10-12 (control sections, second users)

Evidence: `docs/evidence/L7.81` (sections_two_nodes_live.json, vue_dead_space.json, sections2.*.png).

### Added

- **Control sections on Mask Tools and Mask Track.** Mask Tools' luma key in "custom" keeps its four fine controls
  (channel, low / high softness, invert) under "Advanced"; Mask Track's motion mode keeps its twelve detection
  controls under "Detection". Same behaviour as Mask + Matting: a "More" bar on the classic canvas, ComfyUI's
  "Show advanced inputs" in Nodes 2.0.

### Fixed

- **Mask Track in Nodes 2.0** showed every mode's controls at once; it now shows only the current mode's.
- **Empty space under Mask + Matting, Mask Tools and Mask Track in Nodes 2.0** (up to 333 px): their height counted
  the controls ComfyUI folds behind "Show advanced inputs". They now fit their content.

## Unreleased – 2026-10-10 (control sections)

Evidence: `docs/evidence/L7.81` (classic and Nodes 2.0, save + reload round trip).

### Added

- **Control sections on Mask + Matting.** Its advanced controls (Video, Accuracy, Output, Trimap, Performance) fold
  away. On the classic canvas a "More" bar under the node opens each section; your choice is saved with the
  workflow. In Nodes 2.0 the same controls sit behind ComfyUI's own "Show advanced inputs" button and appear under
  ADVANCED INPUTS in the properties panel. Setting: C2C › Canvas › Control sections; ComfyUI's "Always show advanced
  widgets" opens them all.

## Unreleased – 2026-10-10 (node consolidation, wave 3: Mask Track)

### Changed (saved workflows migrate automatically; results are identical)

- **Mask Track** (was Mask Tracker) gains the mode **stabilize** - the former Mask Temporal Stabilizer + Integrity
  (temporal smoothing and the per-frame integrity check), which it replaces. A `warning` output is added; every mode
  shows only its own controls.

## Unreleased – 2026-10-10 (node consolidation, wave 3: Mask Tools)

Evidence: `docs/evidence/L7.65/wave3_mt_live.json`, `wave3_mt_nodes.*.png`; `tests/test_consolidation_wave3.py`.

### Added

- **Mask Tools (C2C)**: keyers and matte operations in one node, one mode at a time - colour, luma and (new)
  difference keys against a clean plate; (new) grade - gain, gamma, clamp, invert; (new) grow or shrink each side
  separately - left / right / up / down - with a feather; combine; gradient; grain; motion blur; edge spread;
  channel shuffle. Each mode shows only its own controls and its preview.

### Changed (saved workflows migrate automatically; results are identical)

- Mask From Color, Luminance Keyer, Mask Gradient, Mask Grain, Mask Motion Blur, Edge Spread, Mask Batch Combine and
  Shuffle are modes of Mask Tools (C2C).

## Unreleased – 2026-10-10 (node consolidation, wave 3: versioning)

### Changed (saved workflows migrate automatically; results are identical)

- **Folder Version Incrementer** gains the layout "show / shot / task / version" -
  `<base>/<show>/<shot>/<task>/v###`, with exclusive version claiming, retries and a version manifest - and replaces
  **Batch Version Manager**. Two new outputs on every layout: `version_path` (the absolute version folder) and
  `info_json`. Each layout shows only its own controls.

## Unreleased – 2026-10-10 (node consolidation, wave 3: Layer Effects)

Evidence: `docs/evidence/L7.65/wave3_le_live.json`, `wave3_le_nodes.*.png`; `tests/test_consolidation_wave3.py`.

### Changed (saved workflows migrate automatically; results are identical)

- **Layer Effects (C2C)** replaces the eight Layer Effect nodes (Drop Shadow, Inner Shadow, Outer Glow, Inner Glow,
  Stroke, Color Overlay, Gradient Overlay, Gradient Map). Switch each effect on to show and use its controls - the
  shadow dial, gradient ramps and colour swatches included. Several effects stack in the order you list (Photoshop's
  order by default), each applied to the result of the one before - the same as chaining the old nodes.

## Unreleased – 2026-10-10 (node consolidation, wave 2)

Evidence: `docs/evidence/L7.65` (`wave2_live.json`, `wave2_nodes.*.png`; `tests/test_consolidation_wave2.py` checks every
merged node against its last shipped code, output by output).

### Changed (saved workflows migrate automatically; results are identical)

- **Size (C2C)** (was Aspect Ratio Preset) also takes a custom width x height and a snap direction - it replaces
  **Dimensions Snap**.
- **Batch Range (C2C)** (was Image Batch Slice) also splits a batch (at a frame or a fraction) and picks one frame
  (first, middle, last or by index) - it replaces **Image Batch Split** and **Video Frame Extractor**. New outputs:
  remainder, remainder_count, total_frames, is_video.
- **Probe (C2C)** (was Image Stats Probe) reports on whatever is wired in - image, mask and/or latent - and keeps every
  output of **Mask Area Probe** and **VAE Latent Inspector**, which it replaces.
- **VAE Inspect** (was VAE Block Inspector) also compares two VAEs (mode compare) - it replaces **VAE Similarity
  Analyser**.
- **VAE Decode (C2C)** (was C2C VAE Quality Decode) gains the clean step (colour cast, saturation, crushed shadows,
  chroma speckle) and accepts an already-decoded image - it replaces **VAE Clean**. Clean is off by default, so
  existing decodes are unchanged.
- These nodes show only the controls their current mode uses.

### Fixed

- **Seed List** in "hash" mode stopped the run with an error (a missing import); it works now.

## Unreleased – 2026-10-10 (node consolidation, wave 1)

Evidence: `docs/evidence/L7.65` in the work area (ComfyUI 0.36.0 / frontend 1.52.7 testbed, classic canvas and
Nodes 2.0). Every removal below was approved card by card by the owner.

### Added

- **Old workflows update themselves.** When a workflow uses a C2C node that was merged or removed, it is replaced
  with its successor as the workflow loads: links, values and seed settings move over, values are translated where
  the successor names them differently, and one notice lists what was updated, what behaves differently now and
  anything that could not be carried over. Nothing is dropped silently. Setting: Settings › C2C › Workflows ›
  Migrate old C2C nodes on load (off = ComfyUI offers the replacement instead). The full table is `docs/MIGRATION.md`.

### Removed (saved workflows migrate automatically)

- **Builder Sampler** → core **KSampler** (same seed, steps, CFG, sampler, scheduler, denoise). KSampler has no CFG
  curve, polish pass, size preset or preview image; connect an Empty Latent Image if nothing feeds `latent_image`.
- **Tone Refiner** → NukeMax **Grade**. Grade starts neutral, so the picture changes; set it by eye.
- **C2C Color Space Convert** → NukeMax **Color Space Convert** (which gained `logc3`). sRGB ↔ linear is
  bit-identical. The old Log C3 conversion turned near-black pixels white (wrong constants in the toe) and clipped
  linear values at 1.0; the new one follows ARRI's curve and keeps highlights.
- **C2C ACES Tonemap** → NukeMax **HDR Tone Map**. Same ACES curve; it expects linear input (add Color Space
  Convert in front for sRGB or Log C3 images), its sRGB output uses gamma 2.2, and contrast / saturation act after
  the curve - re-set them by eye if you had changed them.
- **Prompt Relay Encode (Smart) / (Kijai)** (deprecated) → **Prompt Relay Encode** with backend `smart` / `kijai`;
  identical results.
- **Insight Status** and **Integrity Status** (no successor): the Insight badges under each node and the Diagnostics
  sidebar's Integrity tab show the same information.
- **Mood Board** and **Style Presets** (front-end extras) and their settings. The Prompt Wizard still offers the
  built-in style presets.

## Unreleased – 2026-10-09 (owner list A9: settings, link snapping, auto-connect, Vault, image batching)

Evidence: `docs/evidence/L2.30`, `L2.38` and `L2.40` in the work area (ComfyUI 0.36.0 / frontend 1.52.7 testbed,
classic canvas and Nodes 2.0, with the owner's own C2C setting values).

### Fixed

- **VAE Quality Decode works on the GPU with its defaults.** With "force fp32" on (the default) it changed the shared VAE's
  weights in place. ComfyUI 0.36 pins those weights in memory, so the decode failed with "CUDA error: invalid argument".
  It now decodes with its own fp32 copy of the VAE and leaves yours untouched.
- **Big image batches no longer need 10x their size in RAM.** The C2C tensor inspector, which records min / max /
  mean for every node output, built a huge index while doing it. A 2K x 24 frame batch (0.6 GB) briefly took 6 GB more
  after every node, and runs on 16 GB machines could run out of memory. Measured on a 4K chain load → grade → blur →
  save: 9.0 GB → 3.4 GB peak and 15.7 s → 9.9 s. The statistics are the same.
- **Load Video (C2C) and Save Video (C2C) hold far less in RAM.** They staged 16 to 64 frames at a time, which is up to
  1.6 GB at 4K. They now stage about 64 MB at a time. Measured server chain load → save: 2.6x → 1.7x the batch, at 2K
  and at 4K. The frames are identical, and the speed is unchanged.
- **Clear workflow no longer leaks old widget values into new nodes** (a ComfyUI bug, fixed on our side). After
  "Clear workflow", a node added in the same spot of the id sequence took a deleted node's value for every widget with
  the same name, for example a new sampler node came up with the old sampler. Seen in ComfyUI 1.52.7 with every C2C
  front-end switched off, in both renderers. Setting: Settings › C2C › Canvas › Clear workflow (Off = ComfyUI's own
  behaviour).
- **No more "inputEl is deprecated" warnings** from the Folder Incrementer (it now uses the widget's element).
- **C2C settings no longer disappear.** Lite mode (it switches on by itself when the browser draws without the GPU)
  hid 23 settings, including every overlay switch, because they were registered when an extension started up and
  Lite skips that. Every C2C setting is now declared up front: 144 in Full and in Lite.
- **A wire released anywhere on a node connects to the matching slot again**, with the snap preview while you
  hover. ComfyUI 1.52 turned every widget row into an input socket: over a widget of another type the wire got no
  preview and the release did nothing. This was most of the body of our widget-heavy nodes. Measured on all 315 C2C
  node types with an IMAGE input: 8 failed on the classic canvas, now 0. Setting: Settings › C2C › Canvas › Link
  snapping. A drop on a slot that fits behaves exactly as before.
- **Nodes 2.0: a wire released over the C2C wolf mark** (the empty-state logo in charts and editors) connects. An SVG
  is not an HTML element, and ComfyUI's Nodes 2.0 drop ignored it.
- **Auto-connect wires a new node to the nearest node on its left**, by exact type: every free input whose type has
  exactly one matching output there. It used to wire the first input to whichever node was clicked last. Nothing
  is wired when two nodes are about equally near, when the node arrives with a paste, or when ComfyUI already wired
  it (a node created from a dragged wire). One Ctrl+Z removes the wires.
- **C2C Vault asks for its password on the node.** A Locked vault shows a password field, Unlock, and "Lock again";
  Enter unlocks. It shows the server's real session state, also after a reload. The password is never saved in
  the workflow or the queued prompt, and the field clears after unlocking. A Sealed vault shows the field for
  "Open for editing". The vault id is hidden, because it means nothing to a person. Double-click opens an
  unlocked vault for editing.
- **C2C Vault protects a whole workflow** (Settings › C2C › Vault › Scope: Selected nodes / Whole workflow).
  - Locked: "C2C Vault: Save workflow encrypted" writes the file as ciphertext. Opening it, by drag-drop or from
    the workflow list, asks for the password, and without it nothing loads. Saving again keeps it encrypted. While
    it is open, Export is refused, and the browser's tab-restore keeps only the ciphertext.
  - Sealed: "C2C Vault: Seal whole workflow" puts every node except the outputs into one sealed vault. The workflow
    runs without a password; the Preview / Save nodes stay outside so the results still show.
- **Load Video (C2C) and OmniScale Load Video preview ProRes, DNxHR, HEVC 10-bit and FFV1.** They played the original
  file in the browser, which cannot decode those formats. The preview is now transcoded on the server.
- **Load Video: "select every nth"** keeps the audio for the whole clip and reports the real frame rate (fps / n),
  as VHS does. The audio used to stop halfway.
- **Nodes 2.0: double-clicking a slot** (Get/Set, Suggest) works; it did nothing there.
- **C2C video reads that stop part-way through a file release their decoder cleanly.** A frame cap, every-nth or
  random read left about 17 frames in flight in the decoder's threads, and freeing it like that hung once in the
  test suite (inside FFmpeg). The decoder is now drained first; this costs nothing measurable on 1080p reads.
- **Mask Ops with ViTMatte on the CPU uses a third of the memory.** Tiles now run one at a time on the CPU. Measured
  with the real ViTMatte weights on a 1536×1152 hair plate at the node's defaults: peak RAM 7.7 GB → 2.5 GB,
  9.3 s → 5.8 s, the same matte. The GPU keeps the "tiles per forward pass" setting. The same measurement shows the
  tiled matte is as good as one full-frame pass (within 3 %), and downscaling the frame instead is far worse.
- **VAE Quality Decode: "apply ACES" no longer washes the picture out.** It tone-mapped the decoded (sRGB-encoded)
  image as if it were linear light and then encoded it again, so mid-grey came out at 0.79 instead of 0.55. It now
  matches C2C ACES Tonemap with source sRGB. Workflows that had it on will look darker in the mid-tones, as intended.
- **Image batching is findable.** The setting is called "Image batching" (Settings › C2C › Image batching, also
  found by searching the Settings dialog): Off / Internal (default) / Universal. Per node type: right-click a
  node › Image batching for this node type.

### Added

- **Save Video (C2C) stream save: wire a video latent and its VAE instead of images.** The clip is decoded in order,
  keeping the VAE's causal state, and each frame goes straight into the file, so the full frame batch is never held in
  RAM. The frames are identical to decoding first and then saving (checked on Wan 2.1, Wan 2.2, MiniMax H3 and LTX).
  Measured with Wan 2.1 at 960x544 x 81 frames: 0.48 GB peak RAM instead of 1.57 GB, and the peak stays flat as clips
  get longer. At 2K on an 8 GB card, tile mode Auto (default) / Off / Manual decodes 512 px tiles, each over the whole
  clip and blended on disk: 2048x1080 x 81 frames used 0.79 GB of RAM instead of 4.62 GB, with an identical file,
  in 250 s instead of 201 s. A VAE that cannot be streamed exactly is decoded whole, and the node says so.
- **VAE Quality Decode tiles by itself when the GPU is too small: tile mode Auto (default) / Off / Manual.** On an
  8 GB card a 2K Wan decode took 502 s untiled, because Windows moved the overflow into system RAM. Tiled it took
  37 s with the same picture. Video is tiled in space only: splitting a video VAE in time cost 21 dB. Saved workflows
  that set a tile size open as Manual.
- **A notice when a run spills GPU memory into system RAM** (Windows' NVIDIA driver does this instead of reporting
  out-of-memory). It tells you the driver setting that fixes it. Shown once per session; Settings › C2C ›
  Performance › GPU memory.
- **SeedVR2 preview.** The SeedVR2 Video Upscaler node (numz pack, unchanged) shows each decoded batch while it runs, with
  a before/after wipe. When it finishes, a frame slider steps through every frame, and Difference and Alpha views show
  what changed. Settings › C2C › Video › SeedVR2 preview.
- **Save Video (C2C)** (`🐺 C2C/…/Video`), the pro writer next to VHS Video Combine. It writes MP4 H.264 / H.265
  10-bit, MOV ProRes 422 / HQ / 4444 / 4444 XQ and DNxHR HQ / HQX / 444, lossless MKV FFV1, WebM VP9, GIF,
  animated WebP, and 16-bit PNG / half and float EXR sequences. Alpha is kept where the format has it, audio is
  muxed in (a WAV next to a sequence), and the frames stream to the encoder in chunks sized to free RAM.
  - Optional OCIO colour-space conversion on write (for example sRGB → ACEScg for an EXR plate), from `$OCIO` or the
    built-in ACES studio config.
  - The node plays what it saved, with a scrub timeline and format chips. Formats a browser cannot play (ProRes,
    DNxHR, FFV1, sequences) get a small H.264 preview.
  - Naming: ComfyUI counter (`name_00001`, never overwrites) or Folder Version (wire `subfolder` from the Folder
    Version Incrementer; an existing version is never overwritten), per node or from Settings › C2C › Video ›
    Save Video.
  - Measured round trip per format (PSNR, alpha, audio length): `docs/evidence/L7.56` in the work area.
    For example, ProRes 4444 gives 65.8 dB with alpha, and FFV1, 16-bit PNG and float EXR are exact.

## Unreleased – 2026-10-08 (owner list A9: graph-damaging bugs, settings, Vault)

Evidence for every item: `docs/evidence/L2.15`, `L2.23` and `L2.24` in the work area (Playwright runs on a
ComfyUI 0.36.0 / frontend 1.52.7 testbed, classic canvas and Nodes 2.0).

### Changed defaults (each one is still a setting)

- **Smart guides while dragging: off.** Settings › C2C › Canvas › Align turns them back on.
- **Floating ports: off.** With them on, a wire to a node above or below ends at a dot on that node's edge,
  which read as a stale line. Settings › C2C › Canvas › Floating ports turns them back on.
- **Ctrl+R reloads the page again.** Resetting nodes to their defaults with Ctrl+R is now opt-in
  (Settings › C2C › Reset to Defaults). Right-click a node › "Reset to ORIGINAL" still works.
- **Double-click a slot** is one setting: Nothing, Get/Set variable (default), or Suggest (experimental).

### Fixed

- **Ctrl+Z inside a C2C editor no longer undoes the whole workflow.** ComfyUI catches Ctrl+Z before any editor
  sees it, so one press undid the editor stroke and reloaded the entire graph. This affected the points/box
  editor, the spline editor and tracker, the video mask editor and Face Controller 3D. Ctrl+Z on the canvas
  still undoes the graph.
- **Tidy layout** runs only when you ask: on the selected nodes, or on the whole graph after a confirmation. It
  keeps the graph where it was, moves each group as one block, never moves pinned nodes, and is one undo step.
  It used to send every node to the origin and stack the graph in a single column.
- **Alt-drag duplicate** no longer drags the original along.
- **Dragging a group** no longer picks up nodes that merely overlap its new position.
- **Insert Reroute** (canvas menu on a link) works again.
- **Auto-connect** never wires nodes while a workflow loads, an undo replays, a paste runs or an Alt-drag copies.
- **The command palette** closes on Escape.
- **Get/Set by slot double-click** works at every zoom level and inside subgraphs. Its module had failed to load.
- **Reset to defaults:**
  - Settings › C2C › Reset to Defaults has a button that resets the C2C settings themselves. API keys are kept.
  - Resetting all nodes asks first and is one undo step.
- **C2C Vault works.**
  - Lock and Seal used to hang after the password.
  - A Locked vault could never run, and Unlock always failed.
  - Wires went into the wrong inputs.
  - Core V3 nodes returned the wrong value inside a vault.
  - The creator's session now opens on lock. "Lock session" takes effect on the next queue. The node shows only
    its real sockets.

## [1.14.1] – 2026-05-04

### Fixed

- **Hidden multiline-string widgets no longer leak as giant textareas on top
  of editor nodes.** On modern ComfyUI, multiline `STRING` widgets are
  backed by a real `<textarea>` DOM element parented to a
  `div.dom-widget` wrapper. Setting `widget.type = "hidden"` was no
  longer enough to hide them — the DOM textarea kept rendering on top of
  the canvas and made `PointsMaskEditor`, `SplineMaskEditorMEC`,
  `MECAdvancedPaintCanvas`, and `InpaintCompositeMEC` look like a wall of
  empty text fields. The hide path now also sets `element.style.display
  = "none"` and collapses the `dom-widget` wrapper so the visible UI
  matches what those widgets are supposed to be: invisible internal state.
  Affected widgets:
  - `PointsMaskEditor.editor_data`
  - `SplineMaskEditorMEC.spline_data` / `mask_color` / `mask_opacity`
  - `MECAdvancedPaintCanvas.canvas_data`
  - `InpaintCompositeMEC` mode-conditional widgets (`blend_mode_override`,
    `color_match`, `upscale_method`, `feather_edges`, `feather_radius`)
- **Verified end‐to‐end via Playwright**: created PointsMaskEditor and
  SplineMaskEditorMEC, captured pre/post screenshots, and exercised the
  interactive editors. Points editor accepts left/right click for
  positive/negative points and Ctrl-drag for bboxes — `editor_data`
  payload contains real coordinates with distinct `label:1`/`label:0`
  values (no hardcoded zeros). Spline editor accepts 4 control points
  and renders a smooth catmull-rom curve closed loop, not a polygon.

## [1.14.0] – 2026-05-04

### Fixed

- **SAM Multi-Mask Picker — render real per-mask thumbnails**: the JS
  widget previously called an empty `buildMaskThumbnails()` stub and then
  drew a flat red `fillRect` over all three thumbnails identically, so
  every candidate looked the same regardless of what SAM produced. The
  Python node now serializes the three candidate masks as grayscale PNGs
  to ComfyUI's temp directory (stable input-hash filenames) and ships
  them on the UI payload as `mask_thumbs`. The JS now loads each PNG via
  `/view?...&type=temp` into `this.maskImages[i]` and composites the
  real mask onto each thumbnail using an offscreen canvas with
  `globalCompositeOperation="source-in"` plus a per-thumb tint.
- **SAM Multi-Mask Picker — widget no longer renders ~600 px below the
  node body**: the `addCustomWidget` registration was forwarding
  `node.pos[0]` and `node.pos[1] + widgetY` into the inner draw/mouse
  handlers, but LiteGraph already translates the canvas context to the
  node origin before invoking widget callbacks — the result was a
  doubled offset that pushed the picker off the bottom of the node.
  Registration now passes widget-local coordinates (`0, widgetY` for
  draw; `pos[0], pos[1]` for mouse), so the picker lives inside the
  node body where it belongs.
- **SAM Multi-Mask Picker — stop force-rerunning SAM on every queue**:
  removed `IS_CHANGED → NaN`, which was unconditionally invalidating
  the cache and re-running SAM on identical inputs. Standard cache
  semantics now apply.

## [1.13.1] – 2026-05-04

### Fixed

- **Latest-ComfyUI compatibility**: removed deprecated `disable_noise=False`
  kwarg from two `comfy.sample.sample_custom(…)` call sites in
  `MECBuilderSampler` (mec_paint_suite.py). The argument was dropped from
  `comfy.sample.sample_custom`'s signature in recent ComfyUI builds and
  was raising `TypeError: unexpected keyword argument 'disable_noise'`
  at sample time.
- Audited all 41 nodes against the current ComfyUI runtime: full pack
  imports cleanly, no other deprecated APIs in use
  (`prepare_noise`, `sampler_object`, `calculate_sigmas`,
  `OUTPUT_NODE`, `IS_CHANGED` all match current signatures).

### Notes

- ComfyUI's V3 node API (`io.ComfyNode` / `define_schema` /
  `MatchType` / `Autogrow` / `DynamicCombo`) is **opt-in and additive**
  — the V1 API used throughout this pack is fully supported and not
  scheduled for removal. No mass migration required.

## [1.12.0] – 2026-05-04

### Added

#### Inpaint Composite (MEC) — unified Stitch + Paste Back

- New `InpaintCompositeMEC` node merges the previous two-node split
  (`InpaintStitchProMEC` + `InpaintPasteBackMEC`) behind a single `mode`
  dropdown:
  - `stitch_pro` — advanced blend pipeline (edge-aware, Laplacian
    pyramid, frequency, gaussian) with optional Reinhard color match.
  - `paste_back` — clean resize + paste with optional Gaussian-feathered
    edges. Faster, deterministic, no blend pipeline.
- Unified output signature `(IMAGE, MASK, STRING)`. In `paste_back` mode
  the MASK is the paste rectangle (feathered if enabled) so downstream
  nodes always receive a usable mask.
- New `js/inpaint_composite.js` extension hides parameters that don't
  apply to the selected mode (clean UI without losing widget state).
- Internally delegates to the existing Stitch and PasteBack classes — no
  duplicated logic, no behavioural drift.

### Changed

- Legacy `InpaintStitchProMEC` and `InpaintPasteBackMEC` are still
  registered for backward compatibility, but their display labels now
  read "— legacy (MEC)" to steer new workflows toward the unified node.
- README updated with a new "Inpaint Composite" section, mode-vs-mode
  guidance, and an updated reference table.

## [1.11.0] – 2026-05-08

### Added

#### Ronin-gimbal cinema stabilization (Inpaint Crop / Stitch / Paste-Back)

- New `video_stab_strength` (0–1) widget on `InpaintCropProMEC` activates a
  two-stage filter cascade on the per-frame mask centroid trajectory:
  1. **EMA low-pass** at `4 Hz × strength + 0.05` cutoff to kill HF jitter
     while preserving subject drift.
  2. **Critically-damped spring** (`k = 8·strength + 0.5`,
     `damping = 2·√k`) integrated with semi-implicit Euler for inertia
     and a tiny natural overshoot.
  - **Pre-warm pass**: the spring is run once on the first 12 frames, the
    final `(pos, vel)` state is captured, and the real pass is initialised
    from it — eliminating the snap-to-first-frame artefact that plagues
    naive spring filters.
- **99th-percentile center-displacement sizing** (not full envelope): crop
  width = P99(detected widths) + (P99(cx) − P1(cx)) + 2·padding. Matches
  cinema operator framing — tight on the subject, ignoring outliers.
- New `video_stab_fps` (FPS for filter time-base) and `video_stab_padding`
  widgets.
- `stitch_data` schema bumped to **v3** with new `frame_offsets` field
  (per-frame `(x, y)` paste positions in canvas-space). v2 remains fully
  supported by `InpaintStitchProMEC` and `InpaintPasteBackMEC`.
- `InpaintStitchProMEC._stitch_v2` and `InpaintPasteBackMEC.paste_back`
  rewritten with a per-frame paste loop that consumes `frame_offsets`
  when present (v3) and falls back to the original single-position paste
  for v2 round-trips.
- New helpers in `nodes/stabilization_utils.py`: `_median_filter_1d`,
  `_ema_lowpass`, `_spring_filter`, `ronin_filter_1d`,
  `compute_ronin_bbox_trajectory`.

### Changed

- `InpaintCropProMEC`: `crop_for_inpaint` signature gains
  `video_stab_strength`, `video_stab_fps`, `video_stab_padding` (all
  defaulted; backward compatible).
- `Step 8.6 crop_mask` now emits a per-frame rectangle stack when Ronin
  is active.
- `Step 10` info string reports the active stabilization mode.

### Validated

- Numerical: 11.13 px raw frame-to-frame jitter → 1.34 / 1.64 / 1.76 px at
  strength 0.3 / 0.7 / 1.0 (≥85% reduction); init-snap ≤ 1.2 px (warmup
  works).
- Round-trip: identity inpaint + paste-back is bit-perfect (mean abs
  diff = 0.0000) on 12-frame jittery video.
- Static-subject test: 0 px center spread on stationary subject with
  ±1 px detector noise.

## [1.9.0] – 2026-04-29

### Added

#### MEC Paint Suite (`MEC/Paint`)

- **MECAdvancedPaintCanvas** – interactive paint canvas (DOM widget) with
  Nuke-style procedural mask math: raw alpha → `mask_hardness` core threshold
  → `mask_expansion` morphological dilate/erode → Gaussian
  `mask_blur_radius` lerped against the hard mask by `mask_blur_strength`.
  Outputs the alpha-composited painted image plus the processed mask.
- **MECContextInpainter** – smart blend-back of an inpainted image. Per-channel
  Reinhard colour match, CIE LAB lightness rescue when the inpainted region
  is >5% darker, optional differential-diffusion preservation weight from
  `|orig − inpaint|`, plus `[SEP]` / `[SKIP]` / `[ASC]` / `[DSC]` wildcard
  parsing across connected mask regions.
- **MECToneRefiner** – percentile-based black/white-point tone curve
  (smoothstep), gray-world colour balance, optional bicubic upscale and
  centre-focus fake DOF driven by `ai_dof_focus_depth`.
- **MECBuilderSampler** – KSampler with adaptive CFG (`Constant` / `Linear` /
  `Ease Down` via `set_model_sampler_cfg_function` keyed on per-step sigma)
  and an optional 2-step self-correction polish pass at low denoise.
- **`js/mec_advanced_paint.js`** – RGBA canvas widget with L-paint / R-erase,
  linear-interpolated stamping (no dotted strokes on fast drags), brush
  cursor mirroring size + hardness, mouse-wheel brush sizing, base64-PNG
  serialisation into a hidden `canvas_data` widget so the Python node can
  decode the user's drawing.

## [1.8.0] – 2026-04-29

### Fixed

- **FolderIncrementer (JS)** – removed the `isInputLoader` block that was
  preventing filenames from `LoadImage` / `LoadVideo` / `VHS_LoadVideo`
  from being read. Loaders are now exactly the nodes whose filenames
  feed the versioning folder name.
- **FolderIncrementer (JS)** – added `source_choice` routing
  (`auto`/`image`/`video`) and `trigger_image` / `trigger_video` inputs.
  In `auto` mode video is preferred over image when both are connected.
  Re-syncs on disconnect and on `source_choice` change.
- **Points & BBox Mask Editor** – fixed canvas auto-zoom-in/out jitter
  on first image load and on graph execution. The double-pass `fitView`
  (rAF + 300 ms) was measuring a stale bounding rect from a pre-reflow
  layout, then snapping to the correct rect. Now a single deferred fit
  (two animation frames) runs after LiteGraph reflow has settled.
- **Points & BBox Mask Editor** – `_hasAutoFitted` is no longer reset on
  reference-image disconnect, so brief upstream disconnects keep the
  user's zoom/pan instead of crazy-refitting on reconnect.
- **Spline Mask Editor** – wheel-zoom is hard-clamped (80 px ↔ 8 000 px)
  to prevent the preview bounds from collapsing or exploding after a
  few scroll events. Auto-fit only re-runs when the underlying image
  dimensions change, not on every preview update.

### Added

#### VAE & latent diagnostics

- **VAEMergeMEC** – merge two (or three) VAEs with 8 algorithms
  (weighted_sum, add_difference, tensor_sum, triple_sum, slerp, dare_ties,
  block_swap, clamp_interp). Per-block alpha overrides via JSON or
  comma list, brightness/contrast post-tuning on `decoder.conv_out`,
  CPU-side merge in float32, architecture detection (sd1x / sdxl / flux /
  mochi). Returns the merged VAE plus a JSON info report.
- **VAELatentInspectorMEC** – per-channel min/max/mean/std/abs_mean
  statistics, NaN/Inf counts, and a `verdict` of corrupt / saturated /
  low_contrast / healthy. Optional `fail_on_corrupt` to raise hard.
- **VAESimilarityAnalyserMEC** – cosine similarity between two VAEs
  globally and per block, with optional per-tensor breakdown.
- **VAEBlockInspectorMEC** – per-block weight statistics for any VAE
  (mean / std / abs_mean / count).

#### Color science (`MaskEditControl/Color`)

- **ColorSpaceConvertMEC** – sRGB ↔ Linear ↔ Rec.709 ↔ ACEScg via
  built-in matrices and OETFs. No PyOpenColorIO required.
- **LUTApplyMEC** – Adobe `.cube` LUT loader (1D and 3D) with trilinear
  interpolation in pure torch and a strength blend.
- **ExposureGradeMEC** – exposure (stops), white-balance (temp / tint),
  and contrast around a configurable mid-grey pivot. Operates in linear
  by default with sRGB encode/decode round-trip.

#### EXR I/O & metadata (`MaskEditControl/IO` / `Metadata`)

- **LoadEXRMEC**, **SaveEXRMEC** – EXR read/write with OpenEXR-first,
  imageio-fallback, and final 16-bit TIFF fallback. Half-float by default.
- **EXRMetadataReaderMEC** – pure-python EXR header parser when OpenEXR
  isn't installed; surfaces width/height/channels/compression.
- **MetadataWriterMEC** – write/merge JSON sidecars next to image batches.
- **ShotMetadataNodeMEC** – read shot.json descriptor (show / shot / task /
  frame_in / frame_out / fps).
- **FrameRangeRouterMEC** – slice IMAGE/MASK batches by `[start:end:step]`
  with negative-index support.
- **BatchVersionManagerMEC** – `<root>/<show>/<shot>/<task>/v###/` layout
  with safe sanitization and atomic version reservation via lockfile.

#### Render passes & geometry (`MaskEditControl/Render` / `Geometry`)

- **MergeRenderPassesMEC** – composite beauty + diffuse / specular /
  emission / AO with independent gains.
- **DepthOfFieldMaskMEC** – depth → CoC mask with focus_distance and
  aperture controls; emits both CoC and in-focus masks.
- **DepthWarpMEC** – horizontal parallax warp driven by a depth pass
  (cheap stereo synthesis).
- **NormalToCurvatureMEC** – curvature mask from a tangent-space normal
  pass via central-difference divergence.
- **PositionPassSplitterMEC** – split a world-position pass into per-axis
  X / Y / Z masks (auto- or manually-ranged).

#### Plate tools (`MaskEditControl/PlateTools`)

- **GrainMatchMEC** – extract grain (reference − denoise(reference)) and
  re-apply it to a target plate, with seeded per-frame sampling.
- **PlateStabilizerMEC** – ORB+RANSAC affine stabilization (cv2) with
  FFT-translation fallback when cv2 isn't available.
- **CleanPlateExtractorMEC** – pixelwise median across a batch with
  optional mask exclusion of foreground samples.
- **DifferenceMatteMEC** – L1/L2 difference matte with threshold and
  softness.

#### Diagnostics

- **TemporalConsistencyCheckerMEC** – flicker / instability score across
  a video batch using mask_iou, pixel_diff, or Farneback flow_warp.
- **ModelMetadataExtractorMEC** – inspect safetensors / checkpoint files
  without ever unpickling. SHA256 fingerprint on size + first/last 1MB,
  bounded `pickletools.dis` for legacy pickles.

### Changed

- **FolderIncrementerMEC** – added `folder_name_override` STRING and
  `reserve_version` BOOLEAN. Names are sanitized against Windows reserved
  names and illegal characters. Outputs use POSIX-style paths.
- **SAMModelLoaderMEC** – added `original_device` tracking and module
  helpers (`move_to_inference_device`, `restore_device`) for safe
  CPU↔GPU shuffling on offload-enabled wrappers.
- **MattingNodeMEC** – `auto_download` switch + four-step fallback chain:
  vitmatte_base → vitmatte_small → cv2 guided filter → torch Gaussian.
- **UnifiedSegmentationNode** – added `force_mode` (auto / image / video).

## [1.7.0] – 2026-04-02

### Added

- **DrawShapeMEC** – unified 12-shape drawing node with a single dropdown.
  All parameters exposed as named inputs with descriptive tooltips — no
  more raw JSON editing. Replaces the 5 legacy per-shape wrapper nodes.
- **SplineMaskEditorMEC (JS rewrite)** – complete rewrite of the spline
  editor following Olm SplineMask patterns:
  - Normalized [0,1] coordinates (resolution-independent)
  - Segment insertion via Ctrl+click near curve
  - Close path by clicking first point (highlighted orange)
  - Right-click context menu (Delete, Open/Close, Smooth/Sharp)
  - Zoom-relative point sizes and fonts
  - Property-based persistence with backward-compatible deserialization
  - Status bar with keyboard hints

### Changed

- **DrawCircleMEC, DrawRectangleMEC, DrawEllipseMEC, DrawPolygonMEC,
  DrawLineMEC** – deprecated, now thin wrappers around DrawShapeMEC.
  Kept for backward compatibility with existing workflows.
- Node count updated: 38 → 47 (44 MEC + 3 FolderIncrementer).
- README, docs, and project structure updated for all new nodes.

## [1.6.0] – 2026-03-19

### Added

- **SplineMaskEditorMEC** – interactive spline mask drawing with
  Catmull-Rom, Bezier, and polyline modes. Outputs mask, SAM-compatible
  coords, and SPLINE_DATA for downstream chaining.
- **MotionMaskTrackerMEC** – per-frame motion detection with 4 methods
  (pixel diff, optical flow, background subtraction, histogram diff),
  camera stabilization (homography/affine/translation), and post-processing.
- **BBoxSmooth** – smooth bounding-box sequences across video frames
  using moving-average or exponential smoothing.
- **stabilization_utils.py** – shared camera stabilization helpers for
  motion tracker and inpaint suite.

### Changed

- **InpaintSuite** – uses stabilization_utils for camera-stable cropping.
- **BBoxNodes** – added BBoxSmooth to the 5 existing BBox nodes.

## [1.5.0] – 2025-07-17

### Added

- **InpaintCropProMEC – Canvas Expansion** – crops near image edges now
  extend beyond bounds via `F.pad(mode="replicate")`, producing perfectly
  centered crops everywhere. Ported from lquesada/ComfyUI-Inpaint-CropAndStitch.
- **InpaintCropProMEC – optional_context_mask** – new optional input: union
  of context mask bbox with crop bbox to include surrounding context.
- **InpaintCropProMEC – Iterative Hole-Filling** – `_fill_mask_holes` rewritten
  with 14-threshold iterative approach using scipy `binary_closing` +
  `binary_fill_holes` (with pure-torch fallback). Preserves gradient values
  in soft masks.
- **Stitch Data v2 Format** – new coordinate system: `ctc_x/y/w/h`
  (crop-to-canvas) + `cto_x/y/w/h` (canvas-to-original) for perfect
  reversal of canvas expansion in stitching.

### Changed

- **InpaintStitchProMEC** – v1/v2 dispatch: v2 uses canvas coordinates
  for seamless compositing of expanded crops; v1 fallback preserved for
  backward compatibility with existing workflows.
- **InpaintPasteBackMEC** – updated for v2 stitch format with canvas
  coordinate handling; v1 fallback preserved.
- **Points Editor (JS)** – `computeSize` override prevents LiteGraph
  relayout jitter; widget height computed from actual other-widget heights
  instead of magic number; locked dimensions after image load.
- **Image Comparer (JS)** – `computeSize` override prevents resize jitter
  after image comparison loads.

### Fixed

- **unified_segmentation.py** – marked as DEPRECATED (dead module superseded
  by unified_segmentation_node.py + model_manager.py). Prevents confusion
  from divergent MODEL_REGISTRY.

## [1.4.0] – 2025-07-16

### Added

- **InpaintCropProMEC** – 3 new mask preprocessing inputs from original
  InpaintCropAndStitch: `mask_invert` (flip inpaint/keep regions),
  `mask_fill_holes` (flood-fill enclosed gaps), `mask_hipass_filter`
  (threshold out near-transparent noise).
- **MaskBatchManager** – 2 new temporal operations for video workflows:
  `smooth_temporal` (Gaussian blur along time axis to reduce flicker) and
  `reduce_flicker` (median filter across frames to remove outliers).
- **BBoxSmooth (MEC)** – new node to smooth bounding-box sequences across
  video frames using moving-average or exponential smoothing, eliminating
  jitter in tracked crops.

### Changed

- **InpaintCropProMEC** – crop centering rewrite: bbox now expands
  symmetrically around the mask center, then clamps to image bounds. Fixes
  asymmetric off-center crops when the mask is near image edges.
- **Points Mask Editor (JS)** – complete toolbar UI overhaul: brighter
  button colors, 11 px font, 36 px toolbar height, `textBaseline="middle"`
  vertical centering, 120 ms active-state flash on click, wider separators,
  higher-opacity pills.
- **Points Mask Editor (Python)** – bbox rendering rewritten to use soft
  Gaussian-edged brushes: positive bboxes use `torch.max(mask, brush)`
  (additive, preserves soft point edges), negative bboxes use multiplicative
  erase. Replaces hard `mask = 1.0` overwrite that destroyed soft blending.

### Fixed

- **utils.py** – `multi_scale_guided_refine` and `color_aware_refine` now
  handle 2D grayscale input arrays (was IndexError when `img_np.ndim == 2`).
- **model_manager.py** – float8 dtype comparison guarded with
  `hasattr(torch, "float8_e4m3fn")` to prevent AttributeError on
  PyTorch < 2.1.
- **Points Editor (JS)** – fixed 4 memory leaks: `requestAnimationFrame` now
  cancelled in `onRemoved`, resize debounce timer cleared, keyboard event
  listener properly removed, complete cleanup chain on node deletion.
- **InpaintCropProMEC** – `_resize_lanczos` fix for single-channel masks:
  squeeze channel dim to 2D for PIL, restore after (was crashing with
  `Cannot handle this data type: (1, 1, 1), |u1`).

## [1.3.1] – 2025-07-15

### Added

- **InpaintPasteBackMEC** – new lightweight node to paste inpainted crop back
  onto the original image using stitch_data, with optional Gaussian-feathered
  alpha blending at the crop boundary.
- **InpaintCropProMEC** – `downscale_method` and `upscale_method` inputs
  with 5 interpolation modes: lanczos (PIL-based, highest quality), bicubic,
  bilinear, nearest-exact, area. Upscale method is stored in stitch_data and
  automatically used by InpaintStitchProMEC.

### Changed

- **InpaintCropProMEC** – `padding_multiple` now enforces step=2, min=2 to
  guarantee even-valued padding (required by many tiled/diffusion models).
- **Image Comparer (JS)** – mode labels now use Unicode icons
  (◧ Compare, ⊕ Overlay, ≠ Diff); labels fade during drag; divider grip
  uses a dot-grid pattern; overlay scrubber upgraded to rounded bar + circular
  handle.

### Fixed

- **InpaintStitchProMEC** – now reads `upscale_method` from stitch_data
  (defaulting to lanczos) instead of always using bilinear.

## [1.3.0] – 2025-07-14

### Added

- **ImageComparerMEC** – new interactive before/after comparison node with three
  view modes: drag-slider, adjustable-opacity overlay, and amplified difference
  heatmap. Rendered entirely in-node via a DOM canvas widget.
- **MaskDrawFrame** – 7 new shapes: triangle, star, diamond, cross,
  rounded_rectangle, heart, arrow (total: 12 shapes).
- **MaskDrawFrame** – `rotation` parameter (−360 ° to +360 °) applies to every
  shape including the original five.
- **SAMMaskGeneratorMEC** – `negative_text_prompt` input: describe what to
  exclude; GroundingDINO detects the region and injects negative points into SAM
  inference.
- **InpaintCropProMEC** – `downscale_factor` (0.25–1.0) to shrink the crop
  before sending to the inpainting model and automatically upscale on stitch.
- **InpaintCropProMEC** – `mask_blur` / `mask_grow` pre-processing: blur and
  morphologically dilate/erode the mask before computing the crop region.
- **InpaintCropProMEC** – `aspect_ratio` selector with 12 presets (1:1, 4:3,
  16:9, …) plus custom ratio; crop region is adjusted to match the chosen AR.
- **InpaintCropProMEC** – 6th output `crop_mask`: binary mask showing the
  cropped region in original image space.

### Fixed

- **Points/BBox Editor (JS)** – eliminated canvas jitter caused by
  `updateEditorSize()` re-entrantly calling `node.setSize()` in a
  ResizeObserver loop. Added 50 ms debounce, integer-snap for pixel positions,
  and a re-entrance guard.
- **Points/BBox Editor (JS)** – toolbar separators and pill buttons now render
  at integer pixel positions, removing sub-pixel aliasing artefacts.

### Changed

- **Model Manager** – `.safetensors` format is now preferred over `.pt`/`.pth`/
  `.bin` when a safetensors variant exists locally. SAM3 (`sam3.pt`) is
  explicitly excluded from this mapping.
- **InpaintCropProMEC** – crop dimensions are now snapped up to the nearest
  `padding_multiple` to avoid size mismatches with tiled models.

## [1.2.6] – 2025-07-13

### Fixed

- GitHub Actions publish workflow: added `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24`
  env var and fork guard.
- Added missing `MIT-License` file required by PyPI metadata.

### Changed

- README overview table expanded from 2 to 4 packs.
