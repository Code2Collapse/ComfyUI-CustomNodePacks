# MaskMatting Pipeline (MEC)

> **Single multi-backend segmentation + matting node.** Replaces the
> separate SAM-Loader / SAM-Generator / Trimap / ViTMatte chain with one
> node that auto-selects the right pipeline from the prompts you wire.

**Display name:** `Mask + Matting (MEC)`
**Class:** `MaskMattingMEC`
**Category:** `MaskEditControl/Pipeline`

---

## What it does

Pick a **segmenter** (SAM 2.1 / SAM 3 / SeC / VideoMaMa / GroundingDINO …)
and an optional **matter** (ViTMatte / RVM / MatAnyone2 / SAM-HQ …). Wire
in any combination of:

- **points** (positive_coords / negative_coords from PointsBBoxMaskEditor,
  or legacy multiline `pos_points` / `neg_points`)
- **bbox** (positive / negative / generic)
- **text prompt** (open-vocabulary)
- **video** (when image batch B > 1)

The node infers the prompt mode automatically (`auto`) and routes the
inputs through the chosen backend. The matter takes the coarse mask and
returns a compositing-grade alpha.

---

## Slot order (top to bottom)

```
image  ─┐
positive_coords  ─┐  STRING force-input slots
negative_coords  ─┘  (drive from PointsBBoxMaskEditor)
pos_bbox / neg_bbox / normal_bbox    BBOX slots
text_prompt                          STRING (multiline)
external_mask / external_trimap      MASK slots
```

The two `*_coords` slots accept the JSON output of
[`PointsMaskEditor`](utility-nodes.md#points-mask-editor-mec) directly —
no parsing required.

---

## Required parameters

| Parameter | Default | Choices / Range | Description |
|---|---|---|---|
| `image` | — | IMAGE (B,H,W,C) | Source image or video frames |
| `segmenter` | `sam2.1` | All registered backends; tags `[experimental]` / `[missing-deps]` indicate non-ready ones | Coarse-mask backend |
| `matter` | `vitmatte` if available else `none` | `none` / `vitmatte` / `vitmatte_base` / `rvm_*` / `matanyone2` / `sam_hq` … | Optional alpha refinement |
| `model` | `(auto)` | `(auto)`, `<key>/<file>` for installed weights, `[preset:<key>] <name>` for downloadable presets | Specific weight file for the segmenter |
| `matter_model` | `(auto)` | same syntax | Specific weight file for the matter |
| `precision` | `fp16` | `fp16` / `bf16` / `fp32` | Runtime dtype |
| `attention` | `auto` | `auto` / `sdpa` / `flash` / `sage` / `xformers` / `eager` | Attention backend hint |
| `offload` | `none` | `none` / `cpu` / `sequential` | VRAM offload strategy |
| `subject_preset` | `custom` | `custom` / `hair` / `fur` / `cloth` / `skin_face` / `hard_edge` / `soft_glow` | Override `trimap_dilate` / `trimap_erode` / `edge_radius` with subject-tuned values |
| `trimap_dilate` | `8` | 0–128 px | Foreground dilation when building trimap |
| `trimap_erode` | `8` | 0–128 px | Background erosion when building trimap |
| `edge_radius` | `4` | 0–64 px | Unknown-band width on the trimap |
| `individual_objects` | `false` | bool | If supported, return one mask per detected object |
| `tracking_direction` | `forward` | `forward` / `backward` / `bidirectional` | Direction for video propagation |
| `frame_annotation` | `0` | int | Frame index where the prompts are anchored |
| `object_id` | `0` | 0–1024 | Object slot id (for SAM-2 video memory) |
| `max_frames_to_track` | `0` | 0 = unlimited | Cap on video propagation length |
| `memory_size` | `8` | 1–256 | SAM-2 video memory bank size |
| `start_frame` / `end_frame` | `0` / `-1` | int | Slice the input clip (`-1` = last) |
| `auto_download` | `false` | bool | Allow lazy fetch from HF / torch.hub when a weight is missing |
| `seed` | `0` | int | Reproducibility seed |

---

## Optional input slots

| Slot | Type | Description |
|---|---|---|
| `positive_coords` | STRING (forceInput) | JSON `[[x,y],...]` of foreground points |
| `negative_coords` | STRING (forceInput) | JSON `[[x,y],...]` of background points |
| `pos_points` | STRING (multiline widget) | Legacy text fallback. Accepts KJ `{"positive":[[x,y]]}` or MEC `[{"x","y","label":1}]` |
| `neg_points` | STRING (multiline widget) | Legacy text fallback for negatives |
| `pos_bbox` | BBOX | Positive box `[x0,y0,x1,y1]` |
| `neg_bbox` | BBOX | Optional excluded region |
| `normal_bbox` | BBOX | Polarity-agnostic box |
| `text_prompt` | STRING | Open-vocabulary prompt (SAM3 / GroundingDINO / VideoMaMa) |
| `external_mask` | MASK | Used as a hint when prompts under-specify |
| `external_trimap` | MASK | Pre-computed trimap that bypasses internal generation |

> When both `positive_coords` (slot) and `pos_points` (widget) are filled,
> the slot wins.

---

## Outputs

| Output | Type | Shape / format | Description |
|---|---|---|---|
| `mask` | MASK | (B,H,W) | Coarse mask straight from the segmenter |
| `alpha` | MASK | (B,H,W) | Refined alpha from the matter (= `mask` if `matter='none'`) |
| `preview` | IMAGE | (B,H,W,3) | `image * alpha` — debug preview |
| `trimap` | MASK | (B,H,W) | Trimap fed to the matter (0=bg, 0.5=unknown, 1=fg) |
| `bbox` | BBOX | `[x0,y0,x1,y1]` | Tight box around the alpha |
| `bbox_json` | STRING | `{"x":…,"y":…,"w":…,"h":…}` | Same box as JSON |
| `score` | FLOAT | 0–1 | Segmenter confidence |
| `info` | STRING | JSON | Backends used, mode chosen, per-frame counts |

---

## Backend matrix

| Segmenter | Image | Video | Points | BBox | Text | Notes |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `sam2.1` | ✅ | ✅ (tracker) | ✅ | ✅ | ❌ | Default — robust, fast on small clips |
| `sam3` | ✅ | ✅ | ✅ | ✅ | ✅ | Open-vocabulary (Meta SAM-3) |
| `sec` | ✅ | ✅ | ✅ | ✅ | ✅ | Vision-Language MLLM segmenter |
| `videomama` | ❌ | ✅ | ❌ | ❌ | ✅ | Long-form video text segmenter |
| `groundingdino` | ✅ | ❌ | ❌ | ❌ | ✅ | Text-to-bbox, then SAM |

| Matter | Image | Video | Best for | Notes |
|---|:-:|:-:|---|---|
| `none` | ✅ | ✅ | quick masks | Returns segmenter mask as alpha |
| `vitmatte` / `vitmatte_base` | ✅ | ✅ (per-frame) | hair, fur, glass | Highest single-frame quality |
| `matanyone2` | ✅ | ✅ | long video, occlusions | Temporal warmup protocol |
| `rvm_mobilenetv3` / `rvm_resnet50` | ✅ | ✅ | real-time video | Lightweight |
| `sam_hq` | ✅ | ❌ | coarse refine | Uses HQ SAM head |

---

## Use cases

### Image generation — clean cutout for inpaint reference

```
Load Image ──▶ MaskMattingMEC (segmenter=sam2.1, matter=vitmatte)
                          ├▶ alpha ──▶ Save Image (PNG with alpha)
                          └▶ preview ──▶ Save Image (premultiplied)
```

Drives a cutout you can drop into a different scene, or feed as control
mask to ControlNet / IP-Adapter conditioning.

### Video generation — temporally consistent character mask

```
Load Video ─▶ MaskMattingMEC (segmenter=sam2.1, matter=matanyone2,
                              tracking_direction=bidirectional)
                          └▶ alpha (B,H,W) ──▶ ProPainterTemporalMEC OR
                                              Wan2.2 Animate replace
```

`B > 1` triggers the video pipeline automatically. MatAnyone2's warmup
keeps alpha stable across occlusions.

### Inpainting — drive InpaintCropProMEC with a clean mask

```
Load Image ─▶ PointsBBoxMaskEditor ─▶ MaskMattingMEC ─▶ InpaintCropProMEC
                  positive_coords ─┘     alpha ─────┘
                  negative_coords ─┘
```

Inpaint Crop Pro reads `alpha` as the inpaint mask — no manual painting
needed.

---

## Recipes

| Goal | Settings |
|---|---|
| Best-quality portrait cutout | `segmenter=sam2.1`, `matter=vitmatte`, `subject_preset=hair` |
| Fur / pets | `segmenter=sam2.1`, `matter=vitmatte`, `subject_preset=fur` |
| Real-time video matte | `segmenter=sam2.1`, `matter=rvm_mobilenetv3`, `precision=fp16` |
| Long video with occlusions | `segmenter=sec`, `matter=matanyone2`, `tracking_direction=bidirectional`, `memory_size=16` |
| Text-driven cutout | `segmenter=sam3`, `text_prompt="the red car"`, `matter=vitmatte` |
| Hard-edge product shot | `subject_preset=hard_edge`, `matter=vitmatte_base` |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[missing-deps]` badge on a backend | Optional package not installed | `pip install transformers` (for ViTMatte), or download weights for matanyone2 |
| `FileNotFoundError: Preset … not installed` | Preset entry chosen but `auto_download=false` | Tick `auto_download` or place the file under `models/<key>/` |
| Tiny holes in alpha | `subject_preset=hard_edge` over-eroded | Switch to `custom` and lower `trimap_erode` |
| Mask wobbles across frames | Per-frame matter without temporal model | Switch matter to `matanyone2` |
| Mask follows the wrong instance after first frame | Wrong `object_id` or first-frame prompts ambiguous | Increase `memory_size`, anchor prompts on a clearer frame via `frame_annotation` |
