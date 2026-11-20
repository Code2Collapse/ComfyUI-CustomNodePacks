# Face Fixer (`MECFaceFixer`)

**Menu:** 🐺 C2C/🧰 Core/Paint · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#mecfacefixer)

Needs: your own diffusion `model`, `vae` and conditioning (it samples like a KSampler, on the same device). Face
detection needs a YOLO face model file in `models/ultralytics/bbox/` and the `ultralytics` Python package; without
them, connect a `mask` over the face.

Re-renders every face in an image or a video at `crop_resolution` with your own model, then blends it back. Only
the area around each face changes; the rest of the frame comes back untouched.

## When to pick it

- **Face Fixer**: you want faces re-sampled automatically, one by one, with the model you already use.
- **[Inpaint Crop Pro / Inpaint Stitch Pro](inpaint-suite.md)**: you want to crop one masked area yourself and
  inpaint it with any model or workflow, then put it back.
- **[Face Landmarks](face-landmarks.md)**: you want face points, a face mask or an OpenPose face for another node.
  It does not change the image.

## Where it goes

```
KSampler ──► VAE Decode ──► Face Fixer ──► Save Image
                              ▲  ▲  ▲  ▲
          model, positive, negative, vae: the same ones your KSampler uses
```

For each face the node:
1. finds the face (a YOLO face detector, or your mask);
2. cuts out the face with some context around it (`crop_padding`);
3. if the cut is smaller than `crop_resolution` and an `upscale_model` is connected, upscales it with that model;
4. resizes the cut to `crop_resolution` on its longer side (both sides rounded up to a multiple of 8);
5. samples it image-to-image at `denoise`;
6. blends the result back through a soft oval.

## First run

1. Put Face Fixer after VAE Decode and connect `model`, `positive`, `negative` and `vae`.
2. Pick a face detector in `face_model`. The list shows the `.pt` / `.onnx` / `.safetensors` files in
   `models/ultralytics/bbox/` (also `models/ultralytics/` and `models/yolo/`) and needs the `ultralytics` Python
   package. No detector? Leave `none` and connect a `mask` that covers the face.
3. Start at `denoise` 0.3–0.4 (the default is 0.4): higher values reshape the face more. `denoise` 0 skips sampling
   and keeps each face as it was.
4. Run. To see where faces were found and changed, preview `face_mask`. `info_json` lists every face.

## The controls

Seven controls are always shown: `face_model`, `denoise`, `steps`, `cfg`, `sampler_name`, `scheduler` and `seed`.
The tuning sits in three sections. Open them from the **More** bar at the bottom of the node (classic canvas and
Nodes 2.0 alike). Every section is open when the C2C setting **Canvas > Control sections** is off, or when ComfyUI's
"always show advanced widgets" setting is on.

| Section | Controls | Defaults |
|---|---|---|
| Detection | `confidence`, `max_faces` (both hidden while `face_model` is `none`), `crop_padding`, `crop_resolution` | 0.5, 8 (0 = all), 1.4, 768 |
| Blend | `blend_softness` (feather, px), `mask_dilate` (px, negative shrinks), `color_match`, `lightness_rescue`, `differential_diffusion` | 6, 4, on, on, on |
| Face prompts | `face_positive_prompt`, `face_negative_prompt` (shown only while `clip` is connected) | empty |

What each Blend switch does:
- `color_match`: matches the new face's colour mean and spread to the old one.
- `lightness_rescue`: brightens the new face if it came back darker.
- `differential_diffusion`: blends weaker where the new face barely differs, so those pixels stay as they were.

Saved workflows keep every value: the controls did not move, they are only folded. Every other input is described
in [NODE_REFERENCE](../NODE_REFERENCE.md#mecfacefixer).

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

With `face_model` = `none`, or when the detector finds nothing or cannot run, the node uses the `mask` input. The
box around the mask's white pixels (above 0.5) counts as one face. A mask batch gives one mask per frame; a single
mask is used for every frame. With neither a face nor a mask, the image comes back unchanged, `face_mask` is empty
and `info_json` carries a `warning`.

## Outputs

| Output | What it is |
|---|---|
| `image` | The frames with the faces re-rendered and blended back |
| `face_mask` | Where each face was blended in: a soft oval inside each face's padded box (weighted by how much the face changed when `differential_diffusion` is on) |
| `info_json` | One entry per frame with its faces |

`info_json` holds:
- `frames`: one entry per frame with `frame` (its index), `faces`, and `source` - where the faces came from
  (`detector`, `mask` or `none`);
- `detection`: why the detector did not run, when it did not (for example "ultralytics is not installed");
- `face_prompts`: whether the face prompts were encoded (present only when you wrote face prompts);
- `warning`: present when no face was fixed in any frame, with the reason.

Each face entry holds:
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
| Nothing changes; `info_json` has `warning: no face was fixed ...` | No face found and no mask | Lower `confidence` in Detection, pick a detector, or connect a face mask |
| `info_json` has `detection: ultralytics is not installed ...` | The detector package is missing | `pip install ultralytics`, or use a mask |
| `info_json` has `detection: ... was not found in models/ultralytics/bbox` or `could not load ...` | The model file is missing or not a YOLO model | Check the file in `models/ultralytics/bbox/` |
| Face prompts have no effect | `clip` is not connected | Connect the same CLIP as your prompt |
| "required input(s) missing" | `model`, `vae`, `positive` or `negative` is not connected | Connect it |
