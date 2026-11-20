# Video Stabilizer

One node removes camera shake from a frame batch. It outputs the steady frames and a padding mask that marks the
borders the stabilisation uncovered, so you can fill them with an inpaint step. The engine is the vendored
ComfyUI-Video-Stabilizer (MIT, nomadoor and contributors), in `nodes/video_stabilizer/`.

## Video Stabilizer (`VideoStabilizerMEC`)

**Menu:** 🐺 C2C/🧰 Core/Stabilization · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#videostabilizermec)
`raft_flow` is EXPERIMENTAL and needs the torchvision RAFT weights (see [RAFT weights](#raft-weights)). The other
methods need no model and run on the CPU.

### What it is for

Steadying a handheld, vehicle or drone clip before you paint, inpaint or composite on it. It can also put the
original camera motion back after you edited the steady plate (`motion_meta`), level the horizon, apply your own
keyframed path correction and undo rolling-shutter skew.

### When to pick it

- Use this node to remove the camera motion from the whole frame.
- To follow a mask or a moving subject instead, use Mask Track (`MaskTrackerMEC`, see
  [video-temporal-bbox.md](video-temporal-bbox.md)). Its `stabilize` mode smooths a mask over time; it does not move
  the picture.
- The old nodes Video Stabilizer Classic / Dense Flow / Auto were removed on 2026-11-18 (commit 1dca910). Saved
  workflows load as this node with the method each old node ran: Classic -> `classic`, Dense Flow -> `dense_flow`,
  Auto -> its `force_backend` (`flow` -> `dense_flow`). Details per old node:
  [VideoStabilizerClassicMEC](migration/VideoStabilizerClassicMEC.md),
  [VideoStabilizerFlowMEC](migration/VideoStabilizerFlowMEC.md),
  [VideoStabilizerAutoMEC](migration/VideoStabilizerAutoMEC.md). The environment variable
  `C2C_LEGACY_STABILIZERS=1` registers the old nodes again.

### Minimal wiring

```
Load Video ─▶ Video Stabilizer ─┬─ stabilized_frames ─▶ Inpaint Crop Pro (image)
                                └─ padding_mask      ─▶ Inpaint Crop Pro (mask) ─▶ … ─▶ Save
```

### Key controls

- `method`: how the shake is measured.

  | `method` | How it measures the shake | Needs |
  |---|---|---|
  | `classic` | Feature tracking (good-features + Lucas-Kanade) | nothing; CPU |
  | `dense_flow` | OpenCV DIS dense optical flow | nothing; CPU |
  | `raft_flow` | RAFT dense optical flow (torchvision), EXPERIMENTAL | the RAFT weights |
  | `auto` | `dense_flow` on a clip of more than 24 frames when a CUDA GPU has 4 GB free, else `classic` | none |

  Before 2026-10-24 the method named `raft_flow` actually ran OpenCV DIS, and `raft_iters` / `use_half` did nothing.
  That behaviour is now called `dense_flow`, and `raft_flow` runs RAFT.
- `preset`: sets the six motion settings for you. Pick `manual` to set `framing_mode`, `transform_mode`,
  `camera_lock`, `strength`, `smooth` and `keep_fov` yourself (they are hidden otherwise). The preset values:

  | Preset | transform | strength | smooth | camera lock | framing | keep FOV |
  |---|---|---:|---:|:-:|---|---:|
  | `handheld_light` | similarity | 0.7 | 0.5 | no | crop_and_pad | 0.6 |
  | `handheld_heavy` | perspective | 0.9 | 0.7 | no | crop_and_pad | 0.5 |
  | `vehicle` | similarity | 0.85 | 0.85 | no | expand | 0.5 |
  | `tripod_lock` | translation | 1.0 | 0.95 | yes | crop_and_pad | 0.8 |

- `frame_rate`: set it to the clip's real frame rate. The smoothing window is defined in seconds.
- `padding_color`: the `R, G, B` fill (0-255) for uncovered borders. `padding_mask` marks the same pixels.
- **Path corrections** section (added 2026-11-19, commit f6d8917). All four are applied inside the same per-frame
  warp as the stabilisation, so the frames are resampled once and `motion_meta` stays exact:
  - `horizon_lock`: removes the roll the smoothing left in, so the horizon stays as level as on the first frame.
    On a synthetic clip that rolls 1 degree per frame, the remaining roll over 10 frames went from 7.9 to
    0.5 degrees (commit f6d8917, local test `test_horizon_lock_levels_a_rolling_camera`).
  - `horizon_angle`: rotates every stabilised frame by this many degrees (positive = clockwise), for example to level
    a clip whose first frame was shot tilted.
  - `path_correction`: your own keyframed offset of the stabilised path, one line per keyframe:
    `frame: dx, dy` or `frame: dx, dy, degrees` (pixels; positive = right, down, clockwise). Example:
    `0: 0, 0` and `48: 12, -4, 0.5`. Values between keyframes are interpolated linearly and held before the first
    and after the last keyframe. A JSON object such as `{"48": [12, -4, 0.5]}` works too.
  - `rolling_shutter`: the sensor read-out time as a fraction of one frame (0 = off). It undoes the skew that fast
    pans cause, using the measured frame-to-frame motion. The tooltip notes that CMOS phones and DSLRs are often
    0.3-0.9. This was tested on a synthetic skewed pan only.

  Borders that these corrections uncover are padded and marked in `padding_mask`.

The full list with ranges is in [NODE_REFERENCE](../NODE_REFERENCE.md#videostabilizermec).

### Which method: what was measured

Synthetic hand-shake with known per-frame shifts, 24 frames at 480×270, three seeds. The error is read from the warp
the engine actually applied. Residual shake is in pixels per frame; lower is better. Source: workspace
`docs/evidence/L9.39/stabilizer_ab.json`.

| Plate | classic | dense_flow | raft_flow (RAFT-small) | raft_flow (RAFT-large) |
|---|---:|---:|---:|---:|
| textured | 0.001 | 0.18–0.22 | 0.15–0.20 | 0.03–0.05 |
| near-featureless (soft gradient + noise) | 0.21–0.23 | 0.69–0.86 | 2.6–4.1 | 3.6–4.5 |

On these clips feature tracking had the lowest error on both plates. RAFT-large beat DIS on the textured plate and
did worst on the near-featureless one, so `raft_flow` does not help on snow, sky or water shots. Real footage has
parallax and moving subjects, which this test does not have: check on your own shot.

### RAFT weights

RAFT-large is used on a CUDA GPU and RAFT-small on the CPU. An installed copy of the other size is used before
anything is downloaded. ComfyUI's own device decides, so `--cpu` means CPU. The files are torchvision's
(BSD-3-Clause) `raft_large_C_T_SKHT_V2-ff5fadd5.pth` (21 MB) and `raft_small_C_T_V2-01064c6d.pth` (4 MB), looked up
in `models/optical_flow/` and in the torch hub cache.

- `allow_download` off (default) and no copy installed: the node runs `dense_flow` instead, and `info` names the
  missing file and where it belongs.
- `allow_download` on: the file is fetched from download.pytorch.org into `models/optical_flow/`, and its hash is
  checked.

`raft_iters`, `use_half` and `allow_download` are shown only with `method` = `raft_flow`. `use_half` is fp16 on a
CUDA GPU and is ignored on the CPU.

### Put the camera motion back

Stabilise, edit the steady plate, then restore the original shake with a second Video Stabilizer:

```
Load Video ─▶ Video Stabilizer (framing expand) ─┬─ stabilized_frames ─▶ your edit ─▶ Video Stabilizer ─▶ Save
                                                 └─ motion_meta ─────────────────────────▲ (motion_meta)
```

With `motion_meta` connected, the second node applies the exact inverse of each frame's warp. It reads only
`padding_color` then (the other controls and the path corrections are hidden and ignored). The frames must have the
stabilised size and count. Pixels a crop framing threw away cannot come back, so use `framing_mode` = `expand` on the
first node (preset `vehicle`, or `manual`) if you need every pixel. Measured round trip on a smooth synthetic plate:
mean error under 0.01, which is the cost of resampling twice (local test
`test_motion_meta_round_trip_puts_the_camera_motion_back`).

### Reading the outputs

| Output | What it tells you |
|---|---|
| `stabilized_frames` | The steady clip. With `motion_meta` connected: the clip with its original camera motion back |
| `padding_mask` | 1 where the frame does not cover the canvas: the borders to inpaint |
| `info` | `method=` asked for, `picked=` the method that ran, `flow=` the flow engine (for example `RAFT-small (cpu, 12 iters)`), preset, frame count and size, transform, framing, strength, smooth, elapsed time. Fallback reasons follow on their own lines. With path corrections on, a `corrections:` line lists them |
| `motion_meta` | The exact per-frame warps as JSON. With path corrections on, a `c2c_corrections` block holds the settings and the trajectory: camera roll, output roll before and after, and the centre velocity per frame |

### Common mistakes

| What you see | Why | What to do |
|---|---|---|
| `info` says `RAFT not used (...) - dense_flow used` | RAFT weights missing and `allow_download` off | Put the named file in `models/optical_flow/`, or tick `allow_download` |
| `info` says `horizon_lock / rolling_shutter need the measured motion, which this run has none of - skipped` | The clip has a single frame, or no motion was measured | Feed a multi-frame clip |
| Error `path_correction line ...: write 'frame: dx, dy' or 'frame: dx, dy, degrees'` | A line without a colon, a negative frame, or not 2-3 numbers | Use `frame: dx, dy[, degrees]`, one keyframe per line |
| Error `rolling_shutter must be 0..1` | The read-out is a fraction of one frame | Use a value from 0 to 1 |
| Error `motion_meta is not the JSON a Video Stabilizer writes` | Something other than a Video Stabilizer's `motion_meta` is connected | Wire the `motion_meta` output of the node that made the frames |
| Edges flap or wobble | `strength` near 1 follows the smoothed path exactly | Preset `manual`, lower `strength` to 0.7-0.85 |
| Unwanted zoom-in | A crop framing | `framing_mode` = `crop_and_pad`, then inpaint the border with `padding_mask` |
| Error `The video stabilizer engine failed to load` | The vendored engine in `nodes/video_stabilizer/` is missing or broken | Reinstall the pack |

## Credits

Engine: **ComfyUI-Video-Stabilizer** by nomadoor and contributors (MIT), in `nodes/video_stabilizer/` with its
licence. RAFT: Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow" (ECCV 2020), using the
torchvision implementation and weights (BSD-3-Clause).
