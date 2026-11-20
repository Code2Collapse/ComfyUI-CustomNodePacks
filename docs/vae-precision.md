# Keeping precision through the VAE

Every trip through a VAE loses a little of the picture. This page lists where the loss happens, how large each
loss is, and which C2C option restricts it. The node that carries those options is **VAE Decode (C2C)**.

| Node | Use it to |
|---|---|
| [VAE Decode (C2C)](#vae-decode-c2c-c2cvaequalitydecode) | Decode in fp32, unclamped or tiled; measure a round trip and write unchanged pixels back |

## Where the loss comes from

The numbers below were measured on 2026-10-06 on a CPU testbed (ComfyUI 0.36) with the Flux VAE and the Wan 2.1 VAE,
on a ramp plate and a real matte plate. They are PSNR against the original: higher is closer, and 99 dB means
identical. bf16 / fp16 were emulated on the CPU; the evidence file notes that a real GPU loses slightly more than
shown. Source: workspace `docs/evidence/L7.41/` (`vae_precision.json`, `vae_precision_part2.json`,
`ab_restrictions.json`, `tile_overlap_sweep.json`).

| Cause | Size | What restricts it |
|---|---|---|
| The VAE's own compression, one encode + decode | 37-47 dB (the floor; nothing removes it) | Encode and decode only when you must |
| Repeated encode/decode cycles | 3 round trips: about 8.5 dB worse than one, colour error about 3x | Keep the work in latent space; write back unchanged pixels (below) |
| Running the VAE in bf16/fp16 instead of fp32 | at most 0.17 dB; about 0.15 of an 8-bit level | `force_fp32` on **VAE Decode (C2C)** |
| ComfyUI clamps every decode to 0..1 | out-of-range pixels about 5x less accurate (4.6x Flux, 5.3x Wan) | `clamp_output` off on **VAE Decode (C2C)** |
| Wan VAE frame rule: a clip keeps 4n+1 frames | 10, 11 or 12 frames in -> 9 out, without a message | `frames: pad to 4n+1` on **Wan VAE Encode · tiled (WNE)** |
| Tiled decode | seams; on a Wan clip, worst pixel about 0.23 off a full decode (48.4 dB) | Tile only when memory requires it; larger tiles |
| Saving with 8-bit truncation | about 0.4 of an 8-bit level darker on average (ramp plate) | Save masters as EXR or 16-bit |

The fp32 row is the surprising one: fp32 is not where visible VAE loss comes from. Use it when you have the memory,
not as a fix for colour shifts. The clean step of **VAE Decode (C2C)** (`clean` on) diagnoses colour shifts.

## VAE Decode (C2C) (`C2CVAEQualityDecode`)

**Menu:** 🐺 C2C/🧰 Core/Color Science · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#c2cvaequalitydecode)

### What it is for

A VAE decode with control over precision, tiling and clamping, plus an optional clean step that measures the result
and corrects colour cast, oversaturation, crushed shadows and chroma speckle. It can also clean an image that was
already decoded. The former **VAE Clean** node was merged into it (`clean` on); saved workflows load as VAE Decode
(C2C).

### When to pick it

- Instead of ComfyUI's **VAE Decode** when you need fp32, unclamped values (HDR / EXR chains), automatic tiling for
  large video frames, or a round-trip measurement against the original plate.
- For Wan clips encoded with **Wan VAE Encode · tiled (WNE)** in `pad to 4n+1` mode: this node trims the padded frames
  back off; ComfyUI's VAE Decode does not.

### Minimal wiring

```
KSampler ──► samples ┐
Load VAE ──► vae ────┴─ VAE Decode (C2C) ──► IMAGE ──► Save
Original plate ──► reference (only with clean on)   └─► report / cast_strength / saturation
```

To clean a picture that is already decoded, wire it to `image` and leave `samples` empty.

### Key controls

- **`force_fp32`** (on by default). Decodes in fp32 even when ComfyUI loaded the VAE in bf16/fp16: a core VAE gets
  its own fp32 copy, so the loaded VAE is left as it was. It uses more VAE memory (the tooltip says about 2x).
- **`tile_mode`** (`auto` by default). `auto` tiles only when ComfyUI's own memory estimate says the untiled decode
  will not fit the GPU, and picks the largest tile that fits; on the CPU it never tiles. A `tile_size` above 0 is used
  as given. `off` never tiles; `manual` tiles at `tile_size`.
  - Measured on an RTX 4060 Laptop 8 GB, Wan 2.1 VAE, 2K × 17 frames: untiled 502 s (the driver spilled into system
    RAM), tiled 37 s, same picture (90.9 dB against the full decode) (commit 4ed8f33).
- **`tile_size`.** Pixels per tile. Tiles overlap by 25%. Video is tiled in space only: splitting a video VAE in time
  cost about 21 dB (commit 4ed8f33). On a Wan clip, 16-latent tiles with this overlap came to 48.4 dB against a full
  decode, ComfyUI's own tiled decode at the same overlap 47.9 dB (`tile_overlap_sweep.json`).
- **`clamp_output`** (on by default). Off keeps the values the decoder produces below 0 and above 1, for HDR or EXR
  chains. Other nodes may not expect values outside 0..1, so leave it on unless the chain is built for them.
- **`apply_aces` / `exposure`.** ACES filmic tone mapping after the decode: the decode is treated as sRGB, linearised,
  tone-mapped and encoded once.
- **Video latents.** The output is frames-as-batch, like ComfyUI's VAE Decode. A clip padded by the WNE encoder is
  trimmed back to its original length.

### Clean step (`clean` on): meter, correction and write-back

Connect the plate that went into the encode to **`reference`**.

- **The report** starts with a round-trip meter:
  - PSNR, worst error and mean colour difference (CIEDE2000) against the reference;
  - the share of values outside 0..1;
  - a note when the frame counts differ (for example a Wan clip that lost frames to the 4n+1 rule).
- **`balance_mode`**: `off` (measure only), `grey world`, `white point`, `match reference` (needs `reference`), with
  `balance_strength`. Also `saturation`, `contrast_restore` and `chroma_cleanup`. Their tooltips explain when each one
  fits; all corrections are done in linear light.
- **`restore_unchanged`** (EXPERIMENTAL, 0 = off) writes the reference's pixels back wherever the picture did not
  really change, and keeps your edit.
  - **Measured.** After three Flux round trips plus a painted edit, the area away from the edit went from 35.4 dB to
    the exact original. The edit stayed exactly as painted (`ab_restrictions.json`).
  - **The value** is how far the local average may move and still count as unchanged. One Flux round trip moves it
    by up to 0.037, three by up to 0.050, so start near **0.05-0.07**. Too low leaves VAE loss in place; too high
    also reverts subtle intended changes.
  - **`restore_radius`** is the window, in pixels (8 = one latent cell of an 8x VAE). An edit's influence reaches
    about three windows past its edge, so the boundary blends rather than cuts.

### Reading the outputs

- `IMAGE`: the decoded picture (corrected when `clean` is on).
- `report`: empty when `clean` is off. Otherwise the round-trip meter (with `reference`), what was measured (cast,
  saturation, black and white points, clipping, chroma noise) and what was corrected, including clipping it cannot
  undo.
- `cast_strength`: the largest channel cast measured before correction (0 when `clean` is off).
- `saturation`: the measured saturation before correction (0 when `clean` is off).

### Common mistakes

- **"wire a latent (with its VAE) OR an image to clean - not both"**: connect either `samples` + `vae`, or `image`.
- **"connect a latent and its VAE, or an image to clean"**: nothing to decode.
- **"restore_unchanged is set but no reference image is connected"**: wire the original plate, or set it to 0.
- **`match reference` without `reference`**: the clean step stops and asks for the reference.
- **"Spatial-tiled VAE decode failed"**: try a larger `tile_size`, `tile_size` 0, or check that the VAE takes 5-D
  video latents.
- **Extra frames at the end of a Wan clip**: the clip was padded by the WNE encoder and then decoded with ComfyUI's own
  VAE Decode, which does not know about the padding. Use VAE Decode (C2C) or Wan VAE Decode · tiled (WNE).

## Wan VAE Encode · tiled (WNE): frames

This node is in the ComfyUI-WanNodeExperiments pack.

- **as core** (default). Unchanged behaviour. The console names the frames the encoder will drop.
- **pad to 4n+1.** Repeats the last frame up to the next 4n+1 and marks the latent with the source length.
  - **VAE Decode (C2C)** and **Wan VAE Decode · tiled (WNE)** trim the result back to the original length.
  - ComfyUI's own VAE Decode does not know about the padding and keeps the extra frames.
  - Measured on a 10-frame Wan clip: 10 frames out, frames 1-9 at the same 39.4 dB as without padding
    (`ab_restrictions.json`).

## Habits that keep the most

- Inpaint with a latent noise mask rather than decode -> paint -> encode.
- Never decode and re-encode just to change format or resolution.
- When pixels must be edited between two VAE passes, run **VAE Decode (C2C)** with `clean` on and
  `restore_unchanged` afterwards.
