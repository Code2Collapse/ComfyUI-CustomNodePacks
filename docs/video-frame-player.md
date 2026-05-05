# Video Frame Player (MEC)

> **Node**: `VideoFramePlayerMEC` — *MaskEditControl/Preview*
> **File**: [`nodes/video_frame_player.py`](../nodes/video_frame_player.py) + [`js/video_frame_player.js`](../js/video_frame_player.js)
> **Category icon**: 🎞️

A single, in-graph video scrubber that combines four common video pre-processing
steps into one node so you don't have to chain `Load Video → Trim → Crop →
Resize → Preview` every time. Plays inside the node, lets you drag a crop
rectangle directly on the preview, and emits the trimmed/cropped/resized
batch ready to feed a sampler or video saver.

---

## What it's for

| You want to… | Use this node because… |
|---|---|
| Quickly inspect every frame of a video batch | Built-in scrubber with play/pause, FPS control, ping-pong loop |
| Cut a long batch down to the useful range before sampling | Trim handles on the timeline (`frame_start` / `frame_end`) |
| Run a long video at half rate without re-encoding | `frame_stride` emits every Nth frame |
| Crop a video to a specific aspect (e.g. 16:9 → 9:16 for TikTok) | Drag-crop overlay with aspect lock |
| Match a target resolution before sampling | `target_width` / `target_height` + lanczos resize |
| Upscale 2× with the best CPU/GPU resampler available | `upscale_factor` + `resize_method = lanczos` |
| Lock the crop while you fine-tune other parameters | `crop_locked = true` (handles disappear, border becomes dashed orange) |
| Pick a single hero frame for an img2img refine pass | `output_mode = current_frame` |

It is **not** a video loader (use ComfyUI's built-in or `VHS` for that) and
**not** a sampler. It sits between a loader and a sampler/saver.

---

## Inputs

There are 23 widgets, organized into five groups.

### 1. Source

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `frames` | IMAGE (B,H,W,C) | — | The frame batch from any video loader. B = number of frames. |
| `frame_index` | INT | `0` | Which frame to emit on the `frame` output. The timeline drags this. Clamped to the trim range. |
| `output_mode` | combo | `current_frame` | `current_frame` = emit the single selected frame on `processed`. `all_frames` = emit the whole trimmed/strided batch on `processed`. |

### 2. Trim / playback (Timeline group)

These only affect the `processed` output (and the live preview). The
`frame` output is always the single picked frame.

| Parameter | Type | Default | Behavior |
|---|---|---|---|
| `frame_start` | INT | `0` | First frame of the trim range. Drag the **green** marker on the timeline. |
| `frame_end` | INT | `-1` | Last frame (inclusive). `-1` = "last frame in batch" so it auto-tracks variable-length inputs. Drag the **red** marker. |
| `frame_stride` | INT | `1` | In `all_frames` mode, output every Nth frame. Useful to halve a 60 fps batch to 30 fps without re-encoding. Marked with orange ticks on the timeline. |
| `playback_fps` | FLOAT | `24.0` | Preview-only playback speed. Echoed on the `playback_fps` output for video savers. |
| `loop_mode` | combo | `loop` | Preview behavior at end of trim range: `once`, `loop`, or `ping-pong`. |

**Hotkeys (canvas focused):**

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `←` / `→` | Step ±1 frame (Shift = ±10) |
| `Home` / `End` | Jump to `frame_start` / `frame_end` |
| `I` | Mark IN at current frame (sets `frame_start`) |
| `O` | Mark OUT at current frame (sets `frame_end`) |
| `R` | Reset crop to full frame (no-op when `crop_locked`) |

### 3. Crop

The crop is stored as **normalized** [0..1] fractions, so it survives
resolution swaps. Server-side it's also clamped & aspect-snapped, so the
emitted frame can never escape the source rectangle.

| Parameter | Type | Default | Behavior |
|---|---|---|---|
| `crop_enabled` | BOOLEAN | `false` | When false the rectangle isn't drawn and crop is bypassed. |
| `crop_locked` | BOOLEAN | `false` | Disables drag/resize. Border switches to dashed orange. Use this once you've dialled in the crop. |
| `aspect_ratio` | combo | `free` | `free`, `original`, `1:1`, `4:3`, `3:4`, `16:9`, `9:16`, `2:1`, `21:9`, `custom`. When non-free the rectangle is aspect-locked during drag and snapped on the server. |
| `custom_aspect_w` | FLOAT | `16.0` | Used only when `aspect_ratio = custom`. |
| `custom_aspect_h` | FLOAT | `9.0` | Used only when `aspect_ratio = custom`. |
| `crop_x` / `crop_y` | FLOAT [0..1] | `0` / `0` | Top-left corner. Set automatically by the drag overlay. |
| `crop_w` / `crop_h` | FLOAT (0..1] | `1` / `1` | Width / height as a fraction of source. |

**Drag overlay UX:**

- 4 corner handles (resize from corner, opposite corner stays fixed)
- 4 edge handles (resize one side; aspect-snap re-anchors the opposite edge)
- Drag inside the rectangle to **move** without resizing
- Outside the crop is dimmed; rule-of-thirds guides drawn inside
- Live `W × H` (in source pixels) shown in the top-left of the rect
- Cropping NEVER scrolls outside the canvas border (clamped client + server)

### 4. Resize / upscale

Applied **after** crop, to whatever the cropped resolution is.

| Parameter | Type | Default | Behavior |
|---|---|---|---|
| `resize_method` | combo | `none` | `none`, `lanczos`, `bicubic`, `bilinear`, `area`, `nearest-exact`. Use `lanczos` for the best quality (PIL Lanczos for B≤4, GPU antialiased bicubic for larger batches — visually equivalent at moderate scale). |
| `target_width` | INT | `0` | Output width. `0` = keep crop width. If only one of W/H is set, the other is computed to preserve aspect. |
| `target_height` | INT | `0` | Output height. `0` = keep crop height. |
| `upscale_factor` | FLOAT | `1.0` | Multiplier applied **after** target W/H. So `target=512×512, upscale=2` → 1024×1024. |

### 5. Preview cache

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `preview_width` | INT | `480` | Width of the JPEG thumbnails sent to the browser. Lower = faster scrubbing. |
| `preview_quality` | INT | `80` | JPEG quality (30..95). Cached to disk by content digest, so the second execution is near-instant. |

---

## Outputs

| # | Name | Type | Meaning |
|---|---|---|---|
| 0 | `frame` | IMAGE (1,H,W,C) | The single full-resolution selected frame, **pre-crop, pre-resize**. Use this when you want a hero frame untouched. |
| 1 | `frame_index` | INT | Echo of the clamped `frame_index`. |
| 2 | `frame_count` | INT | Total frames in the input batch (B). |
| 3 | `processed` | IMAGE | The main output: 1 frame in `current_frame` mode, or N frames after trim+stride in `all_frames` mode, **with crop + resize + upscale applied**. |
| 4 | `out_width` | INT | Final width of `processed`. |
| 5 | `out_height` | INT | Final height of `processed`. |
| 6 | `crop_x_px` | INT | Crop left edge in source pixels. |
| 7 | `crop_y_px` | INT | Crop top edge in source pixels. |
| 8 | `crop_w_px` | INT | Crop width in source pixels (= width of `processed` before resize). |
| 9 | `crop_h_px` | INT | Crop height in source pixels. |
| 10 | `trimmed_count` | INT | How many frames `processed` actually contains (useful as the frame count to a video saver). |
| 11 | `playback_fps` | FLOAT | Echo of `playback_fps` — wire straight into a video saver. |

---

## Pipeline order (what happens when you press Queue)

```
frames (B,H,W,C)
  └─► pick mode ─────────────────────────────────┐
       current_frame: take frames[idx:idx+1]     │
       all_frames:    take frames[start:end+1:stride]
                                                 │
       └─► crop_enabled? ─────────────────────────┤
            yes: snap to aspect, clamp, slice     │
            no:  pass-through                     │
                                                 │
       └─► resize_method != none? ────────────────┤
            yes: target_w/h (auto-derive missing) │
                 lanczos / bicubic / bilinear / area / nearest
            no:  pass-through                     │
                                                 │
       └─► upscale_factor != 1.0? ────────────────┤
            yes: × factor with same resize_method │
            no:  pass-through                     │
                                                 ▼
                                            processed
```

The single-frame `frame` output never goes through this pipeline — it is the
raw source frame. Useful when you want the same frame for both a preview
chip and a refine pass.

---

## Recipes

### A. Trim a 240-frame load to frames 30–180, half rate, 1024-wide

| Widget | Value |
|---|---|
| `output_mode` | `all_frames` |
| `frame_start` | `30` |
| `frame_end` | `180` |
| `frame_stride` | `2` |
| `crop_enabled` | `false` |
| `resize_method` | `lanczos` |
| `target_width` | `1024` |
| `target_height` | `0`   *(auto-keep aspect)* |

`processed` will contain **76 frames** (`(180−30)/2 + 1`) at 1024×auto.

### B. Convert 16:9 plate → 9:16 vertical for TikTok

| Widget | Value |
|---|---|
| `output_mode` | `all_frames` |
| `crop_enabled` | `true` |
| `aspect_ratio` | `9:16` |
| `crop_w`/`crop_h` | drag in the canvas to position the vertical band |
| `resize_method` | `lanczos` |
| `target_width` | `1080` |
| `target_height` | `1920` |

### C. Pick one hero frame and refine via img2img

| Widget | Value |
|---|---|
| `frame_index` | drag timeline to the right shot |
| `output_mode` | `current_frame` |

Wire `frame` (output 0) → KSampler. The crop/resize section is ignored
unless you also need a cropped hero — in which case wire `processed` (3) and
set `output_mode = current_frame`.

### D. Lock the crop and tweak the resize independently

1. Drag the crop rect to the right framing.
2. Set `crop_locked = true` (border becomes dashed orange).
3. Now changing `target_width`, `upscale_factor`, etc. won't move the rect
   even if you click on the canvas.

### E. 2× lanczos upscale of an existing crop

| Widget | Value |
|---|---|
| `resize_method` | `lanczos` |
| `target_width` | `0` |
| `target_height` | `0` |
| `upscale_factor` | `2.0` |

When both targets are 0, the upscale multiplies the **post-crop** size, so
this is "make the crop 2× bigger with lanczos".

---

## Performance notes

- **Preview thumbnails are cached** under `ComfyUI/temp/` keyed by a
  content digest of the batch + `preview_width` + `preview_quality`. The
  second run with the same input is nearly instantaneous.
- **GPU lanczos for batches > 4** uses `F.interpolate(antialias=True,
  bicubic)` — visually equivalent to PIL Lanczos at moderate scale and
  100× faster. Single frames go through PIL Lanczos for max fidelity.
- **Crop is a tensor slice**, not a copy + paste, so it's effectively free.
- **Interrupt-aware**: hits `_interrupt_check.check()` inside the preview
  loop, so pressing the ComfyUI ⏹ button stops it cleanly.

---

## Wiring it into a video pipeline

```
VHS Load Video ──▶ Video Frame Player ──┬──▶ KSampler (img2img per frame)
                                        │
                                        ├── trimmed_count ──▶ (frame count to saver)
                                        └── playback_fps ───▶ (fps to saver)
```

A typical full pipeline:

```
Load Video → Video Frame Player → KSampler → VAE Decode → VHS Combine
                  │
                  └── set output_mode=all_frames, crop_enabled=true,
                       aspect_ratio=16:9, target_width=1280, lanczos
```

Then in `VHS Video Combine` wire `frame_rate ← playback_fps` from the
Player so the output FPS automatically follows what you see in the
preview.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Crop rectangle won't drag | Toggle `crop_enabled`. If still stuck, `crop_locked` is true (border is dashed orange) — turn it off. |
| Aspect ratio "snaps back" mid-drag | That's the aspect lock. Choose `free` if you want unrestricted drag. |
| `processed` has only 1 frame in `all_frames` mode | Check `frame_start` ≤ `frame_end`. The trim range may be 1 frame wide. |
| Preview is laggy on long clips | Lower `preview_width` (e.g. 320) or `preview_quality` (e.g. 60). The cached JPEGs get smaller. |
| Output is wrong size | If both `target_width` and `target_height` are 0, the post-crop size is kept. Set at least one. |
| Loop won't stop at the trim end | `loop_mode = once` plays the trim range once and pauses. `loop` and `ping-pong` are infinite. |
| `R` key doesn't reset crop | Click on the canvas first to give it focus, or `crop_locked` is on. |

---

## Acknowledgements

The drag-crop UX was inspired by **Olm DragCrop** by [Olli Sorjonen](https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop)
(source-available, not OSS) and the trim/timeline + resize widget layout
was inspired by **WhatDreamsCost-ComfyUI**'s Load Video UI by
[Jonathan Watkins](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI)
(GPL-3.0). **No source code was copied** from either project — both the
overlay and the timeline are clean-room implementations using standard
HTML5 canvas patterns. See [NOTICE.md](../NOTICE.md) for the full
attribution and license-compatibility statement.
