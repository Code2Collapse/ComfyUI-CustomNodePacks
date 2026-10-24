# Video Stabilizer

One node, **Video Stabilizer** (`VideoStabilizerMEC`, menu `🐺 C2C/…/Stabilization`). It removes camera shake from a
frame batch and outputs the stabilised frames plus a **padding mask**: wire that mask into Inpaint Crop Pro to fill
the borders the stabilisation exposed. The engine is the vendored ComfyUI-Video-Stabilizer (MIT, nomadoor and
contributors). The three older nodes (Classic / Dense Flow / Auto, all marked deprecated) still load and render as
before.

## Methods

| `method` | How it measures the shake | Needs |
|---|---|---|
| `classic` | Feature tracking (good-features + Lucas-Kanade) | nothing; CPU, fast |
| `dense_flow` | OpenCV DIS dense optical flow | nothing; CPU |
| `raft_flow` | RAFT dense optical flow (torchvision), **experimental** | the RAFT weights (see below) |
| `auto` | `dense_flow` on a clip over 24 frames with 4 GB free on the GPU, else `classic` | — |

Before 2026-10-24 the method named `raft_flow` actually ran OpenCV DIS, and its `raft_iters` / `use_half` settings
did nothing. That behaviour is now called `dense_flow`, and `raft_flow` runs RAFT. `auto` and the deprecated nodes
still pick exactly what they picked before.

### Which one to use: what we measured

Synthetic hand-shake with known per-frame shifts, 24 frames at 480×270, three seeds. The error is read from the
warp the engine actually applied, so it does not depend on the picture. Residual shake is in pixels per frame;
lower is better. Source: workspace `docs/evidence/L9.39/stabilizer_ab.json`.

| Plate | classic | dense_flow | raft_flow (RAFT-small) | raft_flow (RAFT-large) |
|---|---:|---:|---:|---:|
| textured | 0.001 | 0.18–0.22 | 0.15–0.20 | 0.03–0.05 |
| near-featureless (soft gradient + noise) | 0.21–0.23 | 0.69–0.86 | 2.6–4.1 | 3.6–4.5 |

On these clips feature tracking was the most accurate on both plates. RAFT-large beat DIS on textured plates and
did worst on the near-featureless one, so `raft_flow` is not a fix for snow, sky or water shots. Real footage has
parallax and moving subjects, which this test does not have, so check on your own shot before you rely on a method.

## Settings

| Widget | Shown | Meaning |
|---|---|---|
| `method` | always | see above |
| `preset` | always | `handheld_light`, `handheld_heavy`, `vehicle`, `tripod_lock`, or `manual` |
| `frame_rate` | always | FPS hint for the smoothing window |
| `padding_color` | always | `R, G, B` fill for exposed borders |
| `framing_mode`, `transform_mode`, `camera_lock`, `strength`, `smooth`, `keep_fov` | `preset = manual` | the preset's values otherwise (table below) |
| `raft_iters` | `method = raft_flow` | RAFT refinement iterations (`num_flow_updates`) |
| `use_half` | `method = raft_flow` | fp16 on a CUDA GPU; ignored on CPU |
| `allow_download` | `method = raft_flow` | download missing RAFT weights into `models/optical_flow/` |

| Preset | transform | strength | smooth | camera lock | framing | keep FOV |
|---|---|---:|---:|:-:|---|---:|
| `handheld_light` | similarity | 0.7 | 0.5 | no | crop_and_pad | 0.6 |
| `handheld_heavy` | perspective | 0.9 | 0.7 | no | crop_and_pad | 0.5 |
| `vehicle` | similarity | 0.85 | 0.85 | no | expand | 0.5 |
| `tripod_lock` | translation | 1.0 | 0.95 | yes | crop_and_pad | 0.8 |

### RAFT weights

RAFT-large is used on a CUDA GPU and RAFT-small on CPU, and an installed copy of the other size is used before
anything is downloaded. ComfyUI's own device decides, so `--cpu` means CPU. The files are torchvision's
(BSD-3-Clause) `raft_large_C_T_SKHT_V2-ff5fadd5.pth` (21 MB) and `raft_small_C_T_V2-01064c6d.pth` (4 MB). They
are found in `models/optical_flow/` or the torch hub cache. With `allow_download` off and no copy installed, the
node runs `dense_flow` and its `info` output says which file is missing and where it belongs. With `allow_download`
on, the file is fetched from download.pytorch.org into `models/optical_flow/`, and its hash is checked.

## Outputs

| Output | Type | Description |
|---|---|---|
| `stabilized_frames` | IMAGE | the stabilised clip |
| `padding_mask` | MASK | 1 where the original frame does not cover the canvas: the borders to inpaint |
| `info` | STRING | method asked for and used, flow engine (e.g. `RAFT-small (cpu, 12 iters)`), settings, time, and why a fallback happened |
| `motion_meta` | STRING | the exact per-frame warps (JSON); wire it into a second Video Stabilizer's `motion_meta` input to put the camera motion back |

## Put the camera motion back

Stabilise, edit the steady plate (paint, inpaint, add an element), then restore the original shake. Use a second
**Video Stabilizer**: feed it the edited frames, connect the first one's `motion_meta` output to its `motion_meta`
input, and it applies the exact inverse of each frame's warp. The other settings are hidden then, except
`padding_color`. The frames must be the stabilised size and count. Use `framing_mode = expand` on the first node if
you need every pixel back, because pixels a crop framing threw away cannot be recovered. Measured round trip on a
smooth plate: mean error under 0.01, the cost of resampling twice (local test
`test_motion_meta_round_trip_puts_the_camera_motion_back`).

```
Load Video ─▶ Video Stabilizer (expand) ─┬─ stabilized_frames ─▶ your edit ─▶ Video Stabilizer ─▶ Save
                                         └─ motion_meta ─────────────────────────▲ (motion_meta)
```

## Recipes

| Goal | Settings |
|---|---|
| Quick handheld correction | `auto`, preset `handheld_light` |
| Strong shake (action cam) | `classic`, preset `handheld_heavy` |
| Vehicle / drone | `classic` or `dense_flow`, preset `vehicle` |
| Lock-off for a VFX plate | `classic`, preset `tripod_lock` |
| Keep the most field of view | preset `manual`, `framing_mode = expand`, `keep_fov = 1.0` |

Stabilise, then inpaint the borders:

```
Load Video ─▶ Video Stabilizer ─┬─ stabilized_frames ─▶ Inpaint Crop Pro
                                └─ padding_mask      ─▶ Inpaint Crop Pro (mask) ─▶ … ─▶ Save
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `info` says `RAFT not used (needs permission …)` | Put the named file in `models/optical_flow/`, or tick `allow_download` |
| Edges flap or wobble | Lower `strength` to 0.7–0.85 (preset `manual`) |
| Unwanted zoom-in | `framing_mode = crop_and_pad` and inpaint the border with `padding_mask` |
| Background pumping | `transform_mode = perspective` |

## Credits

Engine: **ComfyUI-Video-Stabilizer** by nomadoor and contributors (MIT), in `nodes/video_stabilizer/` with its
licence. RAFT: Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow" (ECCV 2020), using the
torchvision implementation and weights (BSD-3-Clause).
