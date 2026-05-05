# Face Fixer (MEC)

> **End-to-end face detail node.** YOLO11 face detection → per-face crop
> → optional AI upscale → per-face KSampler → smart blend back. Behaves
> like the legacy "Forbidden Vision Fixer" but with Impact-Pack wildcard
> syntax and ComfyUI-native sampling.

**Display name:** `Face Fixer (MEC)`
**Class:** `MECFaceFixer`
**Category:** `MEC/Paint`

---

## What it does

Iterates over every detected face in every frame and runs an isolated
KSampler pass at higher resolution, then blends the result back over the
original with feather + color match + lightness rescue. Per-face prompts
are supported via wildcard tokens.

Works on **single images and video batches** identically — each frame is
processed independently, but the seed seeds advance per face so the
output is reproducible.

---

## Required parameters

| Parameter | Default | Range / Choices | Description |
|---|---|---|---|
| `image` | — | IMAGE (B,H,W,C) | Source frame(s) |
| `model` | — | MODEL | Diffusion model used for the per-face KSampler |
| `positive` / `negative` | — | CONDITIONING | Base conditioning. Per-face wildcard prompts override |
| `vae` | — | VAE | VAE for encode/decode of crops |
| `face_model` | `none` | YOLO11 `.pt` / `.onnx` from `models/ultralytics/bbox/` | Face detector. Set `none` to use the optional `mask` input |
| `confidence` | `0.5` | 0.05–0.95 | Detection threshold |
| `max_faces` | `8` | 0–32 (`0` = all) | Cap per frame |
| `crop_padding` | `1.4` | 1.0–3.0 | Bbox padding multiplier so the sampler sees context |
| `crop_resolution` | `768` | 256–2048 (step 64) | Resize each crop's longer side before sampling |
| `denoise` | `0.4` | 0.0–1.0 | Per-face denoise strength (`0.3` subtle, `0.7` aggressive reshape) |
| `steps` | `20` | 1–100 | Sampling steps per face |
| `cfg` | `6.0` | 0–30 | CFG scale |
| `sampler_name` | `euler` | KSampler list | Sampler algorithm |
| `scheduler` | `normal` | KSampler list | Sigma schedule |
| `seed` | `0` | int | Base seed; each face gets `seed + face_index` |
| `blend_softness` | `6.0` | 0–64 px | Feather radius on per-face blend mask |
| `mask_dilate` | `4` | -32–32 | Dilate (>0) / erode (<0) of blend mask |
| `color_match` | `true` | bool | Reinhard mean+std colour match per face |
| `lightness_rescue` | `true` | bool | Lift CIE LAB L if the sample comes back darker than the original |
| `differential_diffusion` | `true` | bool | Weight the blend by abs(orig − sampled) so unchanged pixels stay sharp |

## Optional parameters

| Parameter | Description |
|---|---|
| `mask` | Manual face mask. Used when `face_model='none'` or detection is empty |
| `upscale_model` | UPSCALE_MODEL applied to faces below `crop_resolution` before sampling |
| `face_positive_prompt` | Per-face positive prompt (wildcard syntax below). Empty = use base `positive` |
| `face_negative_prompt` | Same syntax for negatives |

---

## Wildcard syntax (face_positive_prompt / face_negative_prompt)

| Token | Effect |
|---|---|
| `[SEP]` | Separates per-face prompts. Order = detection order |
| `[ASC]` | Order faces left-to-right |
| `[DSC]` | Order faces right-to-left |
| `[ASC-SIZE]` | Order faces small-to-large |
| `[DSC-SIZE]` | Order faces large-to-small |
| `[SKIP]` | Leave that face untouched (no sampling) |

### Examples

```text
red lipstick [SEP] blue eyes [SEP] [SKIP]
```

Three faces: face 1 gets "red lipstick", face 2 "blue eyes", face 3
skipped.

```text
[DSC-SIZE] hero glamour shot [SEP] background extra [SEP] background extra
```

Largest face = "hero glamour shot", smaller faces = "background extra".

---

## Outputs

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | Frame(s) with detailed faces blended back |
| `face_mask` | MASK | Combined mask of every processed face |
| `info_json` | STRING | Per-face metadata (bbox, score, prompt, denoise) |

---

## Use cases

### Image generation — auto face hi-fix

```
KSampler ──▶ VAE Decode ──▶ MECFaceFixer (denoise=0.35, crop_resolution=1024) ──▶ Save
```

Run a normal txt2img then auto-detail the face at higher resolution
without touching the rest of the image.

### Video generation — per-frame face polish

```
Wan2.2 Animate ──▶ MECFaceFixer (face_model=face_yolo11n, denoise=0.3) ──▶ VHS Combine
```

Each frame is independently detailed. Use a low denoise (`0.25–0.35`) to
preserve identity and reduce flicker.

### Multi-character scenes

```
… ──▶ MECFaceFixer
       face_positive_prompt = "[DSC-SIZE] cinematic glamour [SEP] sharp eyes [SEP] [SKIP]"
       max_faces = 3
```

Largest face gets glamour treatment, second face just eye sharpening,
third face skipped (e.g. background blur).

---

## Recipes

| Scenario | Settings |
|---|---|
| Subtle hi-fix | `denoise=0.25`, `steps=20`, `crop_resolution=768`, `differential_diffusion=true` |
| Aggressive reshape | `denoise=0.65`, `steps=30`, `cfg=7.5`, `crop_padding=1.6` |
| Tiny faces in wide shot | `crop_padding=1.8`, `upscale_model=4xUltraSharp`, `crop_resolution=1024` |
| Anti-flicker on video | `denoise≤0.35`, `seed=fixed`, `differential_diffusion=true`, `lightness_rescue=true` |
| Identity-preserving polish | `color_match=true`, `lightness_rescue=true`, `differential_diffusion=true` |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `face_model` dropdown empty | No YOLO11 weights installed | Place `.pt` in `models/ultralytics/bbox/` (e.g. `face_yolov8n.pt` works too) |
| No faces detected | Threshold too high / faces too small | Lower `confidence` to 0.3, raise `crop_padding` to 1.8 |
| Faces look "washed out" | Color match too aggressive on small faces | Disable `color_match`, keep `lightness_rescue` |
| Flicker on video | High `denoise` per frame | Drop to 0.25–0.35; enable `differential_diffusion` |
| Identity drift | Denoise too high or wildcard prompt too strong | Lower `denoise`, simplify wildcard |
