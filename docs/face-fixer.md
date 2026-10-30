# Face Fixer

Re-renders every face in an image or a video at a higher resolution with your own model, then blends it back. Only
the area around each face changes; the rest of the frame comes back untouched.

**Menu:** Face Fixer · **Node id:** `MECFaceFixer` · **Category:** `C2C/Paint`

## Where it goes

```
KSampler ──► VAE Decode ──► Face Fixer ──► Save Image
                              ▲  ▲  ▲  ▲
          model, positive, negative, vae: the same ones your KSampler uses
```

For each face the node:
1. finds the face (a YOLO face detector, or your mask);
2. cuts out the face with some context around it (`crop_padding`);
3. resizes the cut to `crop_resolution` on its longer side;
4. samples it image-to-image at `denoise`;
5. blends the result back through a soft oval.

## First run

1. Put Face Fixer after VAE Decode and connect `model`, `positive`, `negative` and `vae`.
2. Pick a face detector in `face_model`. The list shows the `.pt` / `.onnx` files in `models/ultralytics/bbox/` and
   needs the `ultralytics` Python package. No detector? Leave `none` and connect a `mask` that covers the face.
3. Start at `denoise` 0.3–0.4: higher values reshape the face more.
4. Run. To see where faces were found and changed, preview `face_mask`. `info_json` lists every face.

## The controls

Seven controls are always shown: `face_model`, `denoise`, `steps`, `cfg`, `sampler_name`, `scheduler` and `seed`.
The tuning sits in three sections:
- **Classic canvas:** open them from the **More:** bar at the bottom of the node.
- **Nodes 2.0:** open them with **Show advanced inputs**.

| Section | Controls | Defaults |
|---|---|---|
| Detection | `confidence`, `max_faces` (both hidden while `face_model` is `none`), `crop_padding`, `crop_resolution` | 0.5, 8 (0 = all), 1.4, 768 |
| Blend | `blend_softness` (feather, px), `mask_dilate` (px, negative shrinks), `color_match`, `lightness_rescue`, `differential_diffusion` | 6, 4, on, on, on |
| Face prompts | `face_positive_prompt`, `face_negative_prompt` | empty |

What each Blend switch does:
- `color_match`: matches the new face's colour mean and spread to the old one.
- `lightness_rescue`: brightens the new face if it came back darker.
- `differential_diffusion`: blends weaker where the new face barely differs, so those pixels stay as they were.

Saved workflows keep every value: the controls did not move, they are only folded.

## A different prompt per face

Connect `clip`, the same text encoder you used for `positive`, then write one prompt per face in
`face_positive_prompt` (and, if you like, `face_negative_prompt`):

```text
red lipstick [SEP] blue eyes [SEP] [SKIP]
```

This gives the first face "red lipstick" and the second "blue eyes", and leaves the third untouched.

| Token | Effect |
|---|---|
| `[SEP]` | Separates the faces' prompts |
| `[ASC]` / `[DSC]` | Faces in detection order (most confident first) / the reverse. `[ASC]` is the default |
| `[ASC-SIZE]` / `[DSC-SIZE]` | Faces from the smallest box to the largest / the largest to the smallest |
| `[SKIP]` | That face is not sampled and stays as it was |

- When there are more faces than prompts, the last prompt repeats.
- A face with an empty prompt uses `positive` / `negative`.
- Each distinct prompt is encoded once.
- Without `clip`, the face prompts do nothing and every face uses `positive` / `negative`. `info_json` says so in
  `face_prompts`.

`[SKIP]` works without `clip` too.

Example: the main character gets their own prompt, everyone else is left alone:

```text
[DSC-SIZE] sharp eyes, detailed skin [SEP] [SKIP]
```

## Video

Each frame is detected and sampled on its own. Face `i` of frame `f` uses the seed `seed + 1000·f + i`, so every
face has its own fixed seed. Neighbouring frames are not linked, so a face can come out a little different from frame to
frame. A lower `denoise` changes each face less.

## Using a mask instead of a detector

With `face_model` = `none`, or when the detector finds nothing, the node uses the `mask` input. The box around the
mask's white pixels (above 0.5) counts as one face. A mask batch gives one mask per frame; a single mask is used for
every frame. With neither a face nor a mask, the image comes back unchanged and `face_mask` is empty.

## Outputs

| Output | What it is |
|---|---|
| `image` | The frames with the faces re-rendered and blended back |
| `face_mask` | Where each face was blended in: a soft oval inside each face's padded box |
| `info_json` | One entry per frame with its faces |

Each face entry in `info_json` holds:
- `bbox`: the detected box, before padding;
- `score`;
- `pos_prompt` and `neg_prompt`;
- `prompt_used`: true when the face's own prompt was encoded;
- `denoise`.

A skipped face is listed as `{"bbox": [...], "skipped": true}`.

## Starting points

These are settings to try, not measured results:

| Goal | Try |
|---|---|
| Light clean-up | `denoise` 0.25–0.35 |
| Stronger reshape | `denoise` 0.55–0.7, `crop_padding` 1.6 |
| Small faces in a wide shot | `crop_padding` 1.8, `crop_resolution` 1024, an `upscale_model` |
| Visible seam around the face | raise `blend_softness`, or lower `mask_dilate` |

## Troubleshooting

| What you see | Why | What to do |
|---|---|---|
| `face_model` offers only `none` | No detector file found | Put a YOLO face model (`.pt` / `.onnx`) in `models/ultralytics/bbox/` and restart |
| Nothing changes | No face found and no mask; or `ultralytics` is not installed (the console says so) | Lower `confidence` in Detection, connect a mask, or install `ultralytics` |
| Face prompts have no effect | `clip` is not connected | Connect the same CLIP as your prompt |
| Faces look washed out | `color_match` on a small face | Turn `color_match` off in Blend |
| "required input(s) missing" | `model`, `vae`, `positive` or `negative` is not connected | Connect it |
