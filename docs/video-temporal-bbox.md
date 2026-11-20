# Mask Track: masks over time

One node, Mask Track, does the frame-to-frame mask work for video: it finds motion, spreads a mask from one frame
to the whole clip, morphs between keyframe masks, scores flicker, smooths a mask batch, follows a spline and smooths
a list of boxes. Each job is a `mode`. Most of the separate nodes this page used to describe were merged into it;
the [last section](#old-nodes-that-were-on-this-page) says where each one went.

| Node | Use it to |
|---|---|
| [Mask Track](#mask-track-masktrackermec) (`MaskTrackerMEC`) | Make, propagate, check or smooth a mask across the frames of a clip |

## Mask Track (`MaskTrackerMEC`)

**Menu:** 🐺 C2C/🧰 Core/Video, shown as "Mask Track — Motion/Propagate/Anchor/Consistency/Stabilize" ·
**Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#masktrackermec)
Most modes run on the CPU with no model. Exceptions: `propagate` with `propagate_mode` = `sam2_video` needs a SAM 2.1
model; `stabilize` with `stabilize_temporal_mode` = `raft_flow` needs the torchvision RAFT-small weights.

### What it is for

You have a video (an IMAGE batch) and want a mask per frame, or you have a mask batch and want to check or steady
it. Pick a `mode`; the node shows only the controls that mode reads.

| `mode` | What it does | Needs |
|---|---|---|
| `motion` | Marks what moved between consecutive frames | `video` (2+ frames) |
| `propagate` | Takes a mask drawn on one frame and spreads it to every frame | `video` + `mask` |
| `anchor` | Morphs between keyframe masks (signed distance fields) to fill the frames in between | `mask` (the keyframe stack); `video` only for `flow_refinement` |
| `consistency_check` | Scores how much the mask (or picture) changes from frame to frame | `mask` and / or `video`, by `metric` |
| `stabilize` | Smooths a mask batch over time and flags frames where the mask drops or jumps | `video` + `mask` |
| `spline_track` | Follows spline keyframes through the clip (Lucas-Kanade) and draws them as a mask | `video` + keyframes from the spline canvas |
| `bbox_smooth` | Smooths a list of boxes, one `[x, y, w, h]` per frame, to remove jitter | `bbox_json` text |

### When to pick it

- For a mask on a still image, or to paint and edit one, use the Image Mask Editor
  ([image-mask-editor.md](image-mask-editor.md)).
- To cut out a subject with a segmentation model and track it through a clip, start with
  [mask-matting-pipeline.md](mask-matting-pipeline.md); use Mask Track afterwards to check or steady that matte.
- To remove camera shake from the picture itself, use the [Video Stabilizer](video-stabilizer.md). The `stabilize`
  mode here only smooths the mask.
- To pick frames or a frame range out of a batch, use Batch Range (C2C) (`ImageBatchSliceMEC`), documented in
  [utility-nodes.md](utility-nodes.md#batch-range-c2c-imagebatchslicemec).

### Minimal wiring

```
Load Video ─┬─────────────────────────────▶ video ┐
            └─▶ (mask on frame 0) ─▶ mask ────────┤  Mask Track (mode propagate)
                                                   ├─ masks ─────▶ your inpaint / composite
                                                   ├─ preview ───▶ Preview Image
                                                   └─ info_json ─▶ (show text)
```

For `bbox_smooth`, type or wire the box list into `bbox_json`; no video or mask is needed.

### Key controls

The full list, with ranges, is in [NODE_REFERENCE](../NODE_REFERENCE.md#masktrackermec).

**motion**
- `detection_mode`: `combined` runs every method whose switch is on (`pixel_diff_enabled`, `flow_enabled`,
  `bg_sub_enabled`, `hist_enabled`) and merges them by `combine_method` (`union` or `intersection`). Pick
  `pixel_diff`, `optical_flow`, `background_sub` or `histogram_diff` to run one method. The per-method thresholds sit
  in the **Detection** section.
- `camera_compensation` (on by default) removes the camera's own motion first, using `stabilization_method`
  (`homography`, `affine` or `translation`).
- `grow_pixels`, `min_region_size` and `temporal_smooth` clean the result.

**propagate**
- `source_frame`: the frame the mask was drawn on.
- `propagate_mode`: `static` (the same mask on every frame), `optical_flow` (the mask is warped along the picture's
  motion), `sam2_video` (SAM 2 video tracking), `fade` (the mask's opacity goes from `fade_start` to `fade_end`) or
  `scale_linear` (the mask is scaled about the centre from `fade_start` to `fade_end`).
- `bidirectional`: with `optical_flow` / `sam2_video`, propagate forwards and backwards from `source_frame`.
- `sam2_video` needs a SAM 2.1 model on `sam_model` (from the SAM Model Loader, `SAMModelLoaderMEC`); `points_json`
  adds point prompts. If the loader found no model and handed out its model-free stand-in, the node propagates by
  optical flow instead and says so (see [Reading the outputs](#reading-the-outputs)). Switching off model-free
  fallbacks (`C2C_MODEL_FALLBACK=off`, or the C2C setting) turns that case into an error.

**anchor**
- Wire the keyframe masks as one mask batch into `mask`, and write their frame numbers in `anchor_frames`
  (for example `0,10,30`). `total_frames` is the length of the output.
- If there are fewer frame numbers than masks, the rest are spread evenly over the clip; extra numbers are dropped.
- `easing`: `linear`, `ease_in`, `ease_out` or `smooth_step`. `sdf_iterations`: more is slower.
- `flow_refinement` uses the `video` to refine the in-between masks with optical flow.

**consistency_check**
- `metric`:
  - `mask_iou`: 1 - IoU of each pair of masks after `binarize_threshold`. Needs `mask`.
  - `pixel_diff`: mean absolute change between consecutive frames. It scores the `video` when one is wired, else the
    mask.
  - `flow_warp`: like `pixel_diff`, but frame i is first warped along the optical flow (Farneback). It needs
    OpenCV; without it the node uses `pixel_diff` and the report's `metric` says so.
  - `edge_stability` (added 2026-11-19, commit 03fbba0): how much the matte's **edge** changes from frame to frame.
    It compares alpha in a band around the edge (soft alpha, plus about 1 % of the frame's shorter side around the
    binarised boundary). With the `video` wired, the previous matte is first moved along the picture's optical
    flow, so an edge that follows the subject does not count as flicker. 0 = the edge holds, 1 = it jumps fully.
    Needs `mask`.
- After a `consistency_check` run, the node shows a line chart of the per-pair scores. The chart hides in the other
  modes and is not saved in the workflow.

**stabilize**
- `stabilize_temporal_mode`: `none` (only the integrity report), `gaussian` (smooth across frames with
  `stabilize_sigma`) or `raft_flow` (blend each frame with the flow-warped previous one by `stabilize_blend`).
- `stabilize_drop_threshold` and `stabilize_jump_threshold` decide which frames are flagged.
- `raft_flow` needs torchvision's RAFT-small weights (about 4 MB). Tick `allow_download` (in **Model setup**) to let
  the node fetch them; without them the mask passes through unchanged and `warning` says why.
- `stabilize_device` defaults to `cuda` and uses the CPU when no CUDA GPU is present.

**spline_track**
- The keyframes come from the spline canvas the node shows in this mode (`spline_data`, hidden).
  `spline_tracking_weight` blends the Lucas-Kanade tracker (1) with plain keyframe interpolation (0).
  `spline_closed` makes a filled shape; off gives a stroke of `spline_stroke_width`.

**bbox_smooth**
- `bbox_json`: a JSON array of `[x, y, w, h]`, one per frame. `bbox_radius` is the window radius in frames.
- `bbox_method`: `median_then_exponential` (the default: a median filter drops outliers, then exponential smoothing),
  `moving_average`, `exponential` or `median`. `bbox_alpha` is the exponential factor.

### QC viewer after a run

After a run with the `video` wired, the node shows the C2C viewer in QC mode (added 2026-11-19, commit 858484b): the
video with the output masks, Plate and Render steps, a scrub bar, Comp / Alpha / Checker / Overlay views and
Coverage / Edge detail / Flicker chips. It shows at most 72 proxy frames at 640 px and measures QC on at most 16
frame pairs. In `stabilize` the input mask is the QC reference. It is hidden in `bbox_smooth`, which has no matte.
The viewer only displays; it never changes the outputs and saves nothing in the workflow. Kill switch: environment
variable `C2C_VIEWER=0` (or the C2C setting "Matte viewer").

### Reading the outputs

| Output | What it holds |
|---|---|
| `masks` | One mask per frame. Empty in `bbox_smooth` |
| `preview` | `propagate`: the mask over the video. `consistency_check`: the input passed through. Other modes with a video: the video |
| `score` | `motion`: the mean share of the frame that moved (0-1). `anchor`: the mean confidence (1 at keyframes, lower in between). `consistency_check`: the flicker score (0-1, lower = steadier). `propagate`: 1.0. Others: 0.0 |
| `info_json` | The mode's report (below) |
| `metric` | Which mode ran: `motion`, `propagate:<propagate_mode>`, `anchor`, `consistency_check:<metric>`, `stabilize:<stabilize_temporal_mode>`, `spline_track` or `bbox_smooth` |
| `warning` | `stabilize` only: the flagged frames in words, for example `[MaskIntegrity] 3/48 frames flagged: 12, 13, 30`; empty otherwise |
| `bboxes`, `primary_bbox` | `bbox_smooth` only: the smoothed boxes, and the first one. Empty / `[0, 0, 0, 0]` in the other modes |
| `spline_data_out` | The spline keyframes, passed through |

What `info_json` holds per mode:
- `motion`: text with the frame count, the methods used, the combine / grow settings, the overall motion intensity
  and the per-frame coverage of the first 10 frames.
- `propagate`: `mode` and `source_frame`. When SAM 2 could not run and optical flow was used instead, it adds
  `"approximate": true` and a `model_free` entry (what was wanted, what was used, why), and `metric` ends in
  `(model-free optical flow)`.
- `anchor`: text with the anchor frames, easing, SDF and coverage statistics, the confidence range and mean, and
  whether flow refinement ran.
- `consistency_check`: `metric`, `requested_metric`, `n_frames`, `n_pairs`, `min`, `max`, `mean`, `std`, `pairs`
  (one score per frame pair) and `flicker_score`. `edge_stability` adds `motion_compensated` (was the video used to
  follow the motion), `worst_pairs` (the three worst pair indices) and a `note`.
- `stabilize`: `B`, `flagged_count`, `flagged_frames`, `drop_threshold`, `jump_threshold` and `frames`, one entry per
  frame with `area`, `cx`, `cy`, `area_ratio`, `centroid_delta`, `iou_prev` and a `warn` text on flagged frames.
- `spline_track`: `status`, `keyframes`, `n_control_points`, `lk_failures`, `frames`.
- `bbox_smooth`: the smoothed boxes as a JSON array.

### Common mistakes

| Message or symptom | Why | What to do |
|---|---|---|
| `motion mode needs the video input connected` | `motion` reads the video | Connect `video` |
| `propagate mode needs both the video and the mask inputs connected` | `propagate` reads both | Connect both |
| `anchor mode needs the mask input (the anchor mask stack)` | `anchor` reads the keyframe masks from `mask` | Connect the mask batch |
| `Mask Track (stabilize): connect the video (IMAGE) and the mask.` | `stabilize` reads both | Connect both |
| `Mask Track (spline_track): connect the video (IMAGE).` | The tracker needs the frames | Connect `video` |
| `mask_iou requires a MASK input.` (or `edge_stability ...`) | These metrics score the mask | Connect `mask` |
| `mode='sam2_video' needs a SAM2 model on the sam_model input` | Nothing on `sam_model` | Connect SAM Model Loader with a sam2.1 checkpoint, or pick another `propagate_mode` |
| `needs a sam2.1 model; got ...` | A SAM 1 checkpoint cannot track across frames | Load a sam2.1 checkpoint |
| `the seed mask on frame N is empty` | `source_frame` points at a frame with no mask | Set `source_frame` to the frame you drew on |
| `pixel_diff` score does not follow your mask | With `video` wired, `pixel_diff` scores the video | Disconnect `video`, or use `mask_iou` / `edge_stability` |
| `motion` returns an empty mask, info says `need >= 2 frames` | One frame has no motion | Feed 2 or more frames |
| `warning` starts with `[raft_flow] not applied` | RAFT-small weights missing | Tick `allow_download`, or use `gaussian` |
| No viewer after a run | No `video` wired, mode `bbox_smooth`, or `C2C_VIEWER=0` | Wire the video, or switch the viewer back on |

## Old nodes that were on this page

These were merged into Mask Track. A saved workflow that uses one loads as Mask Track in the mode shown, with its
links and values carried over.

| Old node | Now |
|---|---|
| Motion Mask Tracker (`MotionMaskTrackerMEC`) | Mask Track, mode `motion` |
| Mask Propagate Video (`MaskPropagateVideo`) | Mask Track, mode `propagate` (its `mode` became `propagate_mode`) |
| Temporal Anchor (`TemporalAnchorMEC`) | Mask Track, mode `anchor` |
| Temporal Consistency Checker (`TemporalConsistencyCheckerMEC`) | Mask Track, mode `consistency_check` |
| Mask Temporal Stabilizer + Integrity (`MaskTemporalMEC`) | Mask Track, mode `stabilize` |
| Spline Mask Tracker (`SplineMaskTrackerMEC`) | Mask Track, mode `spline_track` |
| BBox Smooth (`BBoxSmooth`) | Mask Track, mode `bbox_smooth` |

Other nodes this page used to describe:
- **BBox From Mask, BBox To Mask, BBox Pad, BBox Crop** (`BBoxFromMask`, `BBoxToMask`, `BBoxPad`, `BBoxCrop`) are
  legacy nodes, still registered under 🐺 C2C/🧰 Core/Legacy and marked deprecated. Their core replacements:
  [BBoxFromMask](migration/BBoxFromMask.md), [BBoxToMask](migration/BBoxToMask.md),
  [BBoxPad](migration/BBoxPad.md), [BBoxCrop](migration/BBoxCrop.md).
- **BBox Create** (`BBoxCreate`) was removed from this pack in April 2026 (it duplicated Impact Pack and core nodes;
  see the header of `nodes/bbox_nodes.py`).
- **Batch Range (C2C)** (`ImageBatchSliceMEC`) is documented in [utility-nodes.md](utility-nodes.md#batch-range-c2c-imagebatchslicemec).
