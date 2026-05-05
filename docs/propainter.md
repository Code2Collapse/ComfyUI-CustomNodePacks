# ProPainter Suite (MEC)

> **Temporal video inpainting + model-free object/wire/plate removal.**
> Wraps the vendored ProPainter (RAFT + Recurrent Flow Completion +
> InpaintGenerator) with a ComfyUI-native API and chunked VRAM-safe
> execution.

All ProPainter nodes live under `MaskEditControl/Inpaint` (except
`Flow Refine — RAFT (MEC)` which lives under `MaskEditControl/VFX`).

| Class | Display name | Purpose |
|---|---|---|
| `ProPainterTemporalMEC` | ProPainter Temporal Inpaint (MEC) | Drop-in temporal inpainter between Crop Pro and Composite |
| `ProPainterStitchMEC` | ProPainter Stitch (MEC) | Drop-in for InpaintStitchProMEC; flow-consistent seam blend |
| `ProPainterStitchRefineMEC` | ProPainter Stitch Refine (MEC) | Refine only the seam ring — cheapest of the three stitches |
| `ProPainterRemoveMEC` | ProPainter Remove (MEC) | Object / wire / plate removal — no SD model needed |
| `FlowRefineMEC` | Flow Refine — RAFT (MEC) | Standalone RAFT bidirectional flow + warp + consistency |

---

## When to use which node

```
                         ┌─ ProPainterTemporalMEC      (full temporal repaint)
Generative video         │
inpaint pipeline ────────┼─ ProPainterStitchMEC        (replaces InpaintStitchProMEC)
                         │
                         └─ ProPainterStitchRefineMEC  (only seam ring repainted)

Removal pipeline ───────── ProPainterRemoveMEC          (no diffusion model required)

Flow analysis only ────── FlowRefineMEC                 (debug / control wiring)
```

---

## ProPainterTemporalMEC

**Pipeline**: bidirectional RAFT flow → consistency check → RFC flow
completion → sliding-window InpaintGenerator → per-frame Reinhard / LAB
color match → boundary blend with `stitch_blend_mask_crop` → SSIM check.

### Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `images` | — | IMAGE (B,H,W,3) | Cropped clip from `InpaintCropProMEC` (or any video) |
| `masks` | — | MASK (B,H,W) | Inpaint mask. Auto-broadcast across frames if B=1 |
| `stitch_data` | — | STITCH_DATA | From `InpaintCropProMEC`. Pass `{}` if not stitching |
| `neighbor_stride` | `5` | 1–20 | Frame stride for neighbor sampling inside the InpaintGenerator window |
| `ref_stride` | `10` | 1–50 | Frame stride for global reference sampling |
| `raft_iter` | `20` | 1–100 | RAFT iterations per pair — higher = more accurate flow, more VRAM |
| `subvideo_length` | `30` | 8–300 | Frames per InpaintGenerator window. **8 GB cards: 20–30, 12 GB: 40–60, 24 GB: 80+** |
| `raft_chunk` | `16` | 1–64 | Frame-pairs per RAFT forward pass. Lower if OOM at flow stage |
| `use_half` | `true` | bool | fp16 inference (recommended) |
| `blend_boundary` | `true` | bool | Apply `stitch_blend_mask_crop` boundary defense (BOUNDARY_SEAM) |
| `color_match_mode` | `reinhard` | `none` / `reinhard` / `lab_transfer` | Per-frame color matching |

### Outputs

| Output | Type | Description |
|---|---|---|
| `inpainted_image` | IMAGE | Filled clip at crop resolution |
| `info` | STRING | RAFT valid ratio, timings, mode flags |

### VRAM behaviour

The node calls `comfy.model_management.unload_all_models()` + cache
flush **before** loading ProPainter so your SD/Flux/Wan checkpoint is
unloaded automatically. After execution the bundle remains resident; if
you want to free it, queue any node that calls `unload_all_models` or
restart ComfyUI.

### Recipes

| Goal | Settings |
|---|---|
| 8 GB card, ≤ 720p, ≤ 60 frames | `subvideo_length=24`, `raft_chunk=8`, `raft_iter=12`, `use_half=true` |
| 12 GB card, 1080p, 60 frames | `subvideo_length=48`, `raft_chunk=16`, `raft_iter=20` |
| 24 GB card, 4K, 120+ frames | `subvideo_length=80`, `raft_chunk=32`, `raft_iter=20` |
| Long-shot single character removal | `color_match_mode=lab_transfer`, `blend_boundary=true` |
| Quick test | `raft_iter=8`, `subvideo_length=16` |

---

## ProPainterStitchMEC

Drop-in replacement for `InpaintStitchProMEC`. Honours the same
`STITCH_DATA` (v2/v3) but blends the seam with ProPainter instead of
Laplacian / FFT.

### Parameters (key)

| Parameter | Default | Description |
|---|---|---|
| `inpainted_image` | — | Generative inpaint sized to the crop |
| `boundary_band_pixels` | `12` | Width of the boundary band repainted by ProPainter (`0` disables boundary repaint) |
| `preserve_inpaint_center` | `true` | Keep the SD/Flux output untouched at the centre, only repaint the seam |
| `raft_iter` | `12` | RAFT iters |
| `neighbor_stride` | `5` | InpaintGenerator neighbor stride |
| `ref_stride` | `10` | Global reference stride |
| `subvideo_length` | `8` | Frames per window |
| `use_half` | `true` | fp16 |
| `color_match_mode` | `reinhard` | `off` / `reinhard` / `lab` |
| `upscale_method` | `lanczos` | `lanczos` / `bicubic` / `bilinear` / `nearest` |

### Outputs

`(IMAGE, MASK, STRING)` — final canvas, the combined boundary∪mask
actually repainted, and an info string.

### When to choose this over the legacy stitch

Use `ProPainterStitchMEC` when the seam between the SD/Flux inpaint and
the surrounding plate is **wobbling temporally** — Laplacian/FFT blends
are spatially clean but temporally independent. ProPainter sees all
frames and produces a flow-consistent seam.

---

## ProPainterStitchRefineMEC

Repaints **only a thin ring along the original mask boundary**.
Cheapest of the three. Inpaint content (centre) and the surroundings
are preserved bit-for-bit. Use when the SD/Flux inpaint is good and
you only need to hide the seam.

### Parameters (key)

| Parameter | Default | Description |
|---|---|---|
| `boundary_ring_pixels` | `12` | Half-width of the ring along the mask boundary that gets repainted |
| All other ProPainter params | same as above | |

### Output

Same shape as `ProPainterStitchMEC`. The returned mask shows the ring
that was actually touched.

---

## ProPainterRemoveMEC

**No diffusion model required.** Just `(IMAGE, MASK) → IMAGE`. Best when
the masked object moves and the background is visible at *some* other
frame.

### Use cases

- Remove a character from a static plate
- Wire / rig removal in VFX
- Watermark / logo cleanup across video
- Cleaning tracker patches before stabilisation

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `images` | — | Source clip |
| `masks` | — | Region to fill (`1`=fill, `0`=keep). Auto-broadcast if B=1 |
| `quality` | `balanced` | `fast` / `balanced` / `quality` — sets RAFT iters + strides + window |
| `use_half` | `true` | fp16 |
| `dilate_mask_pixels` | `3` | Pre-dilate the mask to cover anti-aliasing pixels |
| `color_match_mode` | `reinhard` | `off` / `reinhard` / `lab` |

#### Quality presets

| Preset | RAFT iters | neighbor_stride | ref_stride | subvideo_length |
|---|---:|---:|---:|---:|
| `fast` | 8 | 10 | 20 | 8 |
| `balanced` | 12 | 5 | 10 | 8 |
| `quality` | 20 | 3 | 6 | 12 |

### Outputs

`(filled_image, fill_mask_used, info)`.

---

## FlowRefineMEC

Standalone two-frame RAFT — useful as a debug / control node for
flow-driven workflows.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `frame_a` / `frame_b` | — | Two frames |
| `iters` | `20` | RAFT iterations |
| `consistency_thr` | `1.5` | Forward+backward agreement threshold |
| `mask` | (optional) | Restricts the visualisation, not the flow |

### Outputs

| Output | Format | Description |
|---|---|---|
| `flow_field_rgb` | IMAGE | R = normalised U, G = normalised V, B = magnitude |
| `warped` | IMAGE | `frame_a` warped to `frame_b` via `grid_sample` |
| `consistency` | MASK | 1 where forward+backward flow agrees |

Falls back to multi-scale Lucas-Kanade when ProPainter / RAFT isn't
installed.

---

## Image vs. video

| Stage | Pure image inpaint | Video inpaint |
|---|---|---|
| Mask preparation | InpaintCropProMEC | InpaintCropProMEC with `video_stable_crop=true` |
| Generative fill | KSampler / Flux / SDXL | (skip — use ProPainterRemove) **or** KSampler per frame |
| Stitch back | InpaintStitchProMEC / InpaintCompositeMEC | **ProPainterStitchMEC** or **ProPainterTemporalMEC** |
| Seam-only fix | n/a | **ProPainterStitchRefineMEC** |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ProPainterMissingError` | Vendored ProPainter not installed | `pip install -r requirements.txt`; ensure `third_party/ProPainter/` exists |
| OOM at RAFT stage | `raft_chunk` too high | Drop to 8 or 4 |
| OOM at InpaintGenerator | `subvideo_length` too high | Drop to 16–24 on 8 GB |
| Flicker on filled region | Color shift between windows | Set `color_match_mode=lab_transfer` |
| Visible seam ring | `blend_boundary=false` | Re-enable, or use `ProPainterStitchRefineMEC` |
| Filled region is gray | All frames have the masked region — no ground truth | Switch to a generative inpaint pipeline |
