# Video Stabilizer (MEC)

> **Vendored MIT-licensed ComfyUI-Video-Stabilizer with classic + RAFT
> backends and an auto-router.** Outputs both stabilized frames and a
> padding mask — feed the padding mask into Inpaint Crop Pro to fill the
> introduced borders.

All three nodes live under `MaskEditControl/Stabilization`.

| Class | Display name | Backend | Best for |
|---|---|---|---|
| `VideoStabilizerClassicMEC` | Video Stabilizer — Classic (MEC) | sparse GFTT + LK | Well-textured, short-medium clips, CPU-only |
| `VideoStabilizerFlowMEC` | Video Stabilizer — RAFT Flow (MEC) | dense RAFT (GPU) | Texture-poor footage where feature tracking fails |
| `VideoStabilizerAutoMEC` | Video Stabilizer — Auto (MEC) | auto | Picks classic vs flow based on clip length and free VRAM |

---

## Common parameters (Classic + Flow)

| Parameter | Default | Range / Choices | Description |
|---|---|---|---|
| `frames` | — | IMAGE (B,H,W,3) | Source clip |
| `frame_rate` | `16.0` | ≥1.0 | FPS hint for smoothing window |
| `framing_mode` | `crop_and_pad` | `crop` / `crop_and_pad` / `expand` | How to handle the un-stabilised borders |
| `transform_mode` | `similarity` | `translation` / `similarity` / `perspective` | Motion model to fit |
| `camera_lock` | `false` | bool | Lock to the first frame's pose (full lockoff) |
| `strength` | `0.7` | 0–1 | How much of the motion to remove |
| `smooth` | `0.5` | 0–1 | Temporal smoothing of the camera path |
| `keep_fov` | `0.6` | 0–1 | Trade-off between FOV and stability |
| `padding_color` | `"127, 127, 127"` | "R, G, B" 0–255 | Fill color when `framing_mode != crop` |

### Flow-only extras (`VideoStabilizerFlowMEC`)

| Parameter | Default | Description |
|---|---|---|
| `raft_iters` | `12` | RAFT iterations |
| `use_half` | `true` | fp16 inference |

---

## Outputs (all three nodes)

| Output | Type | Description |
|---|---|---|
| `stabilized_frames` | IMAGE | Stabilized clip |
| `padding_mask` | MASK | 1 where the original frame did **not** cover the canvas (introduced borders). Pass directly to InpaintCropProMEC to repaint |
| `info` | STRING | Backend, mode, timings |

---

## VideoStabilizerAutoMEC

Picks `flow` when CUDA is available with ≥4 GB free **and** B > 24
frames; else `classic`. Override with `force_backend`.

### Auto-only parameters

| Parameter | Default | Choices | Description |
|---|---|---|---|
| `force_backend` | `auto` | `auto` / `classic` / `flow` | Override |
| `preset` | `handheld_light` | `handheld_light` / `handheld_heavy` / `vehicle` / `tripod_lock` | Bundled tuning |

### Presets

| Preset | transform | strength | smooth | camera_lock | framing | keep_fov |
|---|---|---:|---:|:-:|---|---:|
| `handheld_light` | similarity | 0.7 | 0.5 | ❌ | crop_and_pad | 0.6 |
| `handheld_heavy` | perspective | 0.9 | 0.7 | ❌ | crop_and_pad | 0.5 |
| `vehicle` | similarity | 0.85 | 0.85 | ❌ | expand | 0.5 |
| `tripod_lock` | translation | 1.0 | 0.95 | ✅ | crop_and_pad | 0.8 |

---

## Image vs. video pipelines

| Workflow | Stabilizer placement |
|---|---|
| Inpaint a moving subject in a wobbly clip | Stabilize → Inpaint Crop Pro (`video_stable_crop=true`) → KSampler/ProPainter → Composite → optionally re-add motion |
| Wire / rig removal | Stabilize → ProPainter Remove → re-introduce motion if needed |
| Match-move plate prep | Stabilize (`tripod_lock` preset) → Save EXR sequence |
| LoRA training data | Stabilize → crop to face → batch save |

---

## Use cases

### 1. Stabilise then inpaint

```
Load Video ─▶ VideoStabilizerAuto (preset=handheld_light)
              ├─ stabilized_frames ─▶ Inpaint Crop Pro
              └─ padding_mask     ─▶ Inpaint Crop Pro (mask)
                                         │
                                         └▶ ProPainter Remove ─▶ Save
```

The padding_mask wired as the inpaint mask tells the inpainter
"fill these introduced borders" — you get a clean wider frame.

### 2. Texture-poor footage

```
Drone shot over snow ─▶ VideoStabilizerFlow (raft_iters=20, transform=perspective) ─▶ Save
```

Feature trackers fail on snow / sky / water / out-of-focus shots — RAFT
dense flow handles them.

### 3. Tripod lockoff for VFX plate

```
Load Video ─▶ VideoStabilizerAuto (force_backend=classic, preset=tripod_lock)
              └─ stabilized_frames ─▶ EXR Save (per-frame)
```

---

## Recipes

| Goal | Backend | Settings |
|---|---|---|
| Quick handheld correction | `auto` | preset=`handheld_light` |
| Aggressive shake removal (action cam) | `auto` | preset=`handheld_heavy`, force_backend=`flow` |
| Vehicle / drone | `auto` | preset=`vehicle` |
| Lockoff (tripod sim) | `classic` | preset=`tripod_lock`, transform=`translation` |
| Maximum FOV preservation | any | `framing=expand`, `keep_fov=1.0` |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Flow backend missing | `HAS_FLOW=false` (PyTorch RAFT weights) | Install via the upstream stabilizer's instructions; classic still works |
| Edges flap / wobble | `strength=1.0` over-stabilises | Drop to 0.7–0.85 |
| Unwanted zoom-in | `framing_mode=crop` | Switch to `crop_and_pad` and inpaint the border with the padding_mask |
| Background pumping | `transform_mode=similarity` insufficient | Switch to `perspective` |
| OOM on Flow backend | Too many frames or `use_half=false` | Drop B, enable `use_half`, or fall back to classic |

---

## Credits

The classic + flow algorithms are vendored from
**ComfyUI-Video-Stabilizer** (MIT) and exposed here with the MEC
output schema (image + padding mask + info string). See
[`../THIRD_PARTY_LICENSES/`](../THIRD_PARTY_LICENSES/) for full
attribution.
