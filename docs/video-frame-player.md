# Video Frame Player

An in-graph scrubber for a frame batch. It plays inside the node, lets you set a trim range on its timeline and drag
a crop rectangle on the picture, and outputs the trimmed, cropped, rotated, padded and resized batch. It replaces a
chain of Trim -> Crop -> Resize -> Preview nodes.

## Video Frame Player (`VideoFramePlayerMEC`)

**Menu:** 🐺 C2C/🧰 Core/Preview · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#videoframeplayermec)
· **Files:** [`nodes/video_frame_player.py`](../nodes/video_frame_player.py),
[`js/video_frame_player.js`](../js/video_frame_player.js)

The dashed rotate / pad outline on the node's picture (2026-11-19, f6d8917) is NOT VERIFIED in the browser yet; the
rotate / pad outputs themselves are unit tested.

### What it is for

| You want to… | Use |
|---|---|
| Look through every frame of a batch | the built-in player: play / pause, `playback_fps`, `loop_mode` (`once`, `loop`, `ping-pong`) |
| Cut a batch down to the useful range | the green / red trim markers on the timeline (`frame_start` / `frame_end`) |
| Keep every Nth frame | `frame_stride` (with `output_mode` = `all_frames`) |
| Crop to an aspect (e.g. 16:9 -> 9:16) | `crop_enabled` + `aspect_ratio`, then drag the rectangle |
| Straighten a tilted shot, or add room for outpainting | `rotate_degrees`, `pad_left` / `pad_top` / `pad_right` / `pad_bottom`, and the `pad_mask` output |
| Reach a target size | `target_width` / `target_height`, `upscale_factor`, `resize_method` |
| Pick one frame for an image pass | `output_mode` = `current_frame` |

### When to pick it

It is not a loader. Put it after one, for example [Load Video (C2C)](video-io.md#load-video-c2c-loadvideoc2c), which
already has its own player, crop tool and in / out handles for choosing what to LOAD. Use the Frame Player when the
frames are already in the graph (a sampler's output, a mid-graph batch), or for rotate / pad and the `source_map`.
To compare two clips use the [Video Comparer](video-comparer.md).

### Minimal wiring

```
Load Video (C2C) ──IMAGE──► Video Frame Player ──processed──────► next node
                                               ──playback_fps───► Save Video (C2C).fps
                                               ──pad_mask───────► inpaint / outpaint mask   (when padding)
```

### Key controls

The full list with defaults is in [NODE_REFERENCE](../NODE_REFERENCE.md#videoframeplayermec). In the node, the
controls are folded into the sections Range, Player, Crop, Resize, Transform and Preview.

- `output_mode`: `current_frame` puts the selected frame on `processed`; `all_frames` puts the whole trimmed /
  strided batch there.
- `frame_index`: the frame shown and emitted on `frame`. The timeline drags it.
- `frame_start` / `frame_end` (inclusive; `-1` = the last frame) and `frame_stride`.
- Crop: `crop_enabled`, `aspect_ratio` (`free`, `original`, `1:1`, `4:3`, `3:4`, `16:9`, `9:16`, `2:1`, `21:9`,
  `custom` with `custom_aspect_w` / `custom_aspect_h`). The rectangle is stored as fractions of the frame
  (`crop_x` ... `crop_h`), so it survives a resolution change, and it is aspect-snapped and clamped on the server too.
- Transform (2026-11-19, f6d8917; mapped from AusBoss's Video Crop + Rotate + Pad, own implementation):
  `rotate_degrees` turns the picture inside the crop window (positive = clockwise) about its centre. Pixels come
  from the whole source frame, so the corners fill with real picture where there is any. `pad_*` adds pixels on
  each side. `pad_mode` fills the padding and any empty rotation corners: `color` (`pad_color`, `R, G, B` 0-255 or
  `#RRGGBB`), `edge` (nearest edge pixel) or `mirror`.
- Resize: `target_width` / `target_height` (0 = keep; one of them set = the other follows the aspect), then
  `upscale_factor` multiplies the result. `resize_method`: `lanczos` uses PIL Lanczos for up to 4 frames and
  antialiased bicubic for larger batches; `bicubic`, `bilinear`, `area`, `nearest-exact` as named.
- Preview: `preview_width`, `preview_format` (`png` default, lossless; `jpeg` smaller), `preview_quality` (JPEG
  only). These change only what the browser shows; the IMAGE outputs always use the full source frames.

Order of operations when you queue:

```
frames ─► current_frame: frames[frame_index]  |  all_frames: frames[frame_start : frame_end+1 : frame_stride]
       ─► crop (if crop_enabled)
       ─► rotate (if rotate_degrees ≠ 0) ─► pad (if any pad_* > 0)      pad_mask = 1 where filled
       ─► resize to target_width / target_height, × upscale_factor
       ─► processed
```

The `frame` output skips all of this: it is the untouched source frame at `frame_index`.

### Hotkeys and the crop overlay

Click the picture first so it has keyboard focus.

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `←` / `→` | Step 1 frame (Shift: 10) |
| `Home` / `End` | Jump to `frame_start` / `frame_end` |
| `I` / `O` | Set `frame_start` / `frame_end` to the current frame |
| `R` | Reset the crop to the full frame (not while `crop_locked`) |

The crop rectangle has 4 corner and 4 edge handles; drag inside it to move it. The outside is dimmed, rule-of-thirds
guides are drawn inside, and the crop size in source pixels is shown in its corner. With `crop_locked` on, the
handles disappear and the border turns dashed. With rotate or pad on, a dashed outline shows the source area the
output comes from, and a label lists the angle and the padding.

### Long clips: one proxy video (2026-11-19, c2ed103)

For clips longer than 120 frames the node writes ONE preview-size H.264 video instead of a picture per frame. Every
frame of it is a keyframe, so a seek in the browser lands on the exact frame. The file sits in ComfyUI's temp folder
and is reused when the same clip, size and fps come back. Shorter clips still get one PNG / JPEG per frame. Measured
in that commit, 300 frames at 320x180: 300 PNGs / 54.8 MB became one mp4 / 10.8 MB; writing took 2.84 s instead of
0.63 s.

- Change the threshold with the setting *C2C > Video > Preview > Frame Player: one proxy video above (frames)*, or
  the environment variable `C2C_FRAME_PLAYER_PROXY_ABOVE` (it wins). 0 = always one picture per frame.
- Without PyAV / libx264 the node shows at most that many evenly spaced stills, and says so; scrubbing then shows
  the nearest kept frame.

### Reading the outputs

- `frame` (the untouched frame at `frame_index`), `frame_index`, `frame_count` (all input frames).
- `processed`, `out_width`, `out_height`, `trimmed_count` (frames on `processed`).
- `crop_x_px`, `crop_y_px`, `crop_w_px`, `crop_h_px`: the crop in source pixels.
- `frames_trimmed`: the trimmed / strided source batch with NO crop or resize, for a saver that should keep the
  source size. `trim_start_idx` / `trim_end_idx`, `width` / `height` (source), `duration` (seconds, at
  `playback_fps`), `playback_fps`.
- `video_info` (JSON): width, height, frame_count, trim range and stride, trimmed_count, fps, duration_sec,
  loop_mode, crop (`crop_norm`, `crop_px`), output size, aspect_ratio, dtype, channels, preview_format.
- `pad_mask` (MASK, same size as `processed`): 1 where padding or an empty rotation corner was filled, 0 on real
  picture. All 0 without rotate / pad.
- `source_map` (JSON): for each frame on `processed` its `source_frame`, `seconds` and `timecode` (non-drop, at
  `playback_fps`, from 00:00:00:00), plus a `transform` block (crop in pixels, rotation, padding, sizes before and
  after the resize, scale) and `out_to_source_affine`: a 2x3 matrix that takes a `processed` pixel back to source
  pixels.

### Common mistakes

| What you see | Why / what to do |
|---|---|
| The crop rectangle will not move | `crop_enabled` is off, or `crop_locked` is on (dashed border). |
| The aspect "snaps back" while dragging | That is the aspect lock; pick `free`. |
| `processed` has 1 frame in `all_frames` | The trim range is one frame wide; check `frame_start` / `frame_end`. |
| The output did not change size | Both `target_width` and `target_height` are 0 and `upscale_factor` is 1. |
| The output was resized although `resize_method` is `none` | A target size or upscale is set; `none` then resizes with bilinear. |
| "mirror padding needs each pad smaller than the picture" | Use `edge` or `color` for wide pads. |
| "pad_color ... is not 'R, G, B' (0-255) or #RRGGBB" | Fix the text, e.g. `127, 127, 127`. |
| "proxy video failed to load - re-queue the node" (on the picture) | The temp proxy is gone; queue the node again. |
| Scrubbing a long clip jumps between frames | No PyAV / libx264: only evenly spaced stills were written (the node says so). |

### Acknowledgements

The drag-crop interaction was inspired by Olm DragCrop by Olli Sorjonen
(<https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop>), and the trim timeline and resize layout by WhatDreamsCost's
Load Video UI (<https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI>, GPL-3.0). Both are clean-room
implementations; no code was copied.
