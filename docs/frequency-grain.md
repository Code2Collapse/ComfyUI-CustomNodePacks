# Frequency separation and film grain

Three nodes for retouching and finishing. Frequency Separate splits an image into a soft base and a fine detail
layer; Frequency Combine puts a detail layer back onto a base (the same one, an edited one, or a different image);
Film Grain adds grain. They are ported from ComfyUI-NKD-Basic-Tools by Nekodificador (MIT, commit 613fd3c), work in
torch on whole batches, and need no model or extra package.

| Node | Use it to |
|---|---|
| [Frequency Separate (MEC)](#frequency-separate-mec-frequencyseparatemec) | split an image into a base and a detail layer |
| [Frequency Combine (MEC)](#frequency-combine-mec-frequencycombinemec) | merge a detail layer onto a base |
| [Film Grain (MEC)](#film-grain-mec-filmgrainmec) | add film grain, fresh on every frame of a clip |

Typical uses: keep the skin texture of a photo after relighting or retouching it (separate the original, combine its
detail onto the new base), or move fine texture from one image onto another.

```
image ─► Frequency Separate ─ low_frequency ─► (edit / relight the base) ─► low_frequency  ┐
                            └ high_frequency ────────────────────────────► high_frequency ─┴► Frequency Combine ─ image ─► Film Grain ─ image
```

---

## Frequency Separate (MEC) (`FrequencySeparateMEC`)

**Menu:** 🐺 C2C/🧰 Core/Frequency · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#frequencyseparatemec)

### What it is for

It smooths the image into a base (`low_frequency`) and stores what the smoothing removed as a detail layer
(`high_frequency`). The detail layer is not a picture: in `Divide` mode "no detail" is 1.0, in `Subtract` mode 0.0.
It is meant to go into Frequency Combine.

### Key controls

- `method`: `Guided` (default; edge-aware), `Rolling Guidance` (removes texture by size, keeps shapes), `Gaussian`,
  `Median` (spot blemishes; its radius is capped at 7). `edge_threshold` is shown only for Guided and Rolling Guidance.
- `radius` (8): the detail scale. Larger = a softer base and more in the detail layer.
- `mode`: `Divide` (default; a ratio, so detail survives a lighting change) or `Subtract`.
- `detail`: `Luminance` (default; detail without colour) or `RGB`.
- `linear` (on): work in linear light. Turn it off to work in gamma like classic Photoshop.
- `mask` (optional): only affects `high_frequency_masked` - detail is kept inside the mask, neutral outside.

### Preview

After a run the node shows the first frame's input, base and detail (buttons: Input, Base, Detail, Base | Detail),
at most 640 px on the longer side. Change `radius`, `mode` or `detail` and the panel recomputes a live
approximation in the browser - a Gaussian base with the same radius - labelled as such until the next run shows the
exact result of the chosen method. Only the first frame is sliced for the preview: on 60 frames of 720p the
preview's extra memory went from +647 MB to +29 MB (6178bc9).

### Reading the outputs

- `high_frequency`: the detail layer (working space: linear when `linear` is on).
- `low_frequency`: the base, back in display space.
- `high_frequency_masked`: the detail layer kept only inside the mask; the same as `high_frequency` without a mask.
- `report`: "Frequency Separate (MEC) processed N frame(s)." then the filter and radius (with the effective radius
  when Median capped it), linear light on / off, mode and detail, and the mask coverage in % (or "no mask").

### Common mistakes

- Saving or previewing `high_frequency` as a picture: its values are centred on 1.0 (`Divide`) or 0.0
  (`Subtract`, with negative values), so an image viewer or save node clips them. Wire it straight to Frequency
  Combine.
- A mask batch shorter than the image batch: the extra frames reuse the last mask (the report says so).

---

## Frequency Combine (MEC) (`FrequencyCombineMEC`)

**Menu:** 🐺 C2C/🧰 Core/Frequency · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#frequencycombinemec)

### What it is for

It puts a detail layer from Frequency Separate back onto a base: the base it came from (after you edited it), or a
different image.

### Key controls

- `mode` and `linear`: must be the same as in the Frequency Separate that made `high_frequency`.
- `detail_strength` (1.0): below 1 weakens the detail, above 1 boosts it, 0 returns the base only.
- `mask` (optional): apply the detail only inside the mask; `mask_feather` (shown only while a mask is connected)
  softens its edge.

### Reading the outputs

- `image`: the base with the detail merged on, clamped to 0..1 (values above 1.0 are clipped: it is a display-range
  IMAGE, not a scene-linear plate). An RGBA `low_frequency` keeps its alpha.
- `report`: mode and linear light, `detail_strength`, mask feather and coverage (or "no mask"), and the clamp note.

### Common mistakes

- `mode` or `linear` different from the Separate node: the detail is decoded the wrong way.
- In `Divide` mode, negative detail values are dropped before recombining.
- A `low_frequency` of another size than `high_frequency` is resized (bilinear) to the detail layer's size.

---

## Film Grain (MEC) (`FilmGrainMEC`)

**Menu:** 🐺 C2C/🧰 Core/Frequency · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#filmgrainmec)

### What it is for

Lightroom / Camera Raw-style grain with Amount, Size and Roughness. On a clip, `animate` (on by default) gives
every frame fresh grain; off, every frame gets the same grain.

### Key controls

- `amount` (25): how much grain shows; 0 returns the input unchanged.
- `size` (25): coarser grain - the noise is made at a lower resolution and scaled up.
- `roughness` (50): mixes a fine and a coarse noise field.
- `color` (0): 0 = monochrome grain; raise it for coloured grain.
- `seed`: the same seed gives the same grain. It has ComfyUI's "control after generate" setting: set it to `fixed`
  to get the same grain on every run.
- `mask` (optional): grain only inside the mask, soft where the mask is soft.

### Reading the outputs

- `image`: the image with grain added (RGB channels; values clamped to 0..1).
- `report`: the settings, `animate` and seed, mask coverage (or "no mask"), the grain field sizes, and "out of GPU
  memory on ...: computed on the CPU instead" when that happened.

### Common mistakes

- Expecting the same grain on a re-run while `seed` is set to change after each run.
- Grain runs on ComfyUI's own device (it follows `--cpu` and the chosen GPU); only an out-of-memory error falls back
  to the CPU, and the report says so.
