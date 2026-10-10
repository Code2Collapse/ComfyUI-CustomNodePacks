# Keeping precision through the VAE

Every trip through a VAE loses a little of the picture. This page lists where the loss happens, how large each
loss is, and which C2C option restricts it.

The numbers were measured on ComfyUI 0.36 with the Flux VAE and the Wan 2.1 VAE. They are PSNR against the
original: higher is closer, and 99 dB means identical.

## Where the loss comes from

| Cause | Size | What restricts it |
|---|---|---|
| The VAE's own compression, one encode + decode | 37-47 dB (the floor; nothing removes it) | Encode and decode only when you must |
| Repeated encode/decode cycles | 3 round trips: about 8.5 dB worse than one, colour error about 3x | Keep the work in latent space; write back unchanged pixels (below) |
| Running the VAE in bf16/fp16 instead of fp32 | at most 0.17 dB; about 0.15 of an 8-bit level | `force_fp32` on **VAE Decode (C2C)** (costs about 2x VAE memory) |
| ComfyUI clamps every decode to 0..1 | out-of-range pixels about 5x less accurate | `clamp_output` off on **VAE Decode (C2C)** |
| Wan VAE frame rule: a clip keeps 4n+1 frames | 10, 11 or 12 frames in -> 9 out, without a message | `frames: pad to 4n+1` on **Wan VAE Encode · tiled (WNE)** |
| Tiled decode | seams; worst pixel up to about 0.2 off | Tile only when memory requires it; larger tiles |
| Core **Save Image** truncates to 8 bits | half a level darker on average | Save masters as EXR or 16-bit; VHS Video Combine rounds already |

The fp32 row is the surprising one: fp32 is not where visible VAE loss comes from. Turn it on when you have the
memory, not as a fix for colour shifts. The clean step of **VAE Decode (C2C)** (`clean` = on) diagnoses colour shifts.

## VAE Decode (C2C)

- **force_fp32.** Decodes in fp32 even when ComfyUI loaded the VAE in bf16/fp16, and restores the VAE afterwards.
- **tile_size.** Pixels per tile, for video latents. 0 decodes the whole frame. Tiles overlap by 25%. Measured
  on a Wan clip, this comes closer to a full decode than ComfyUI's own tiled decode at the same overlap.
- **clamp_output** (optional, on by default). Off keeps the values the decoder produces below 0 and above 1, for
  HDR or EXR chains. ComfyUI's Preview and Save clamp on their own, but other nodes may not expect values
  outside 0..1, so leave it on unless the chain is built for them.
- **Video latents.** The output is frames-as-batch, like ComfyUI's VAE Decode. A clip padded by the WNE encoder
  is trimmed back to its original length.

## Clean step (VAE Decode (C2C), clean = on): meter and write-back

Connect the plate that went into the encode to **reference**.

- **The report** starts with a round-trip meter:
  - PSNR, worst error and mean colour difference (CIEDE2000) against the reference;
  - the share of values outside 0..1;
  - a note when the frame counts differ.
- **restore_unchanged** (experimental, 0 = off) writes the reference's pixels back wherever the picture did not
  really change, and keeps your edit.
  - **Measured.** After three Flux round trips plus a painted edit, the area away from the edit went from 35.4 dB
    to the exact original. The edit stayed exactly as painted.
  - **The value** is how far the local average may move and still count as unchanged. One Flux round trip moves
    it by up to 0.037, three by up to 0.050, so start near **0.05-0.07**. Too low leaves VAE loss in place; too
    high also reverts subtle intended changes.
  - **restore_radius** is the window, in pixels (8 = one latent cell). An edit's influence reaches about three
    windows past its edge, so the boundary blends rather than cuts.

## Wan VAE Encode · tiled (WNE): frames

- **as core** (default). Unchanged behaviour. The console now names the frames the encoder will drop.
- **pad to 4n+1.** Repeats the last frame up to the next 4n+1 and marks the latent with the source length.
  - **VAE Decode (C2C)** and **Wan VAE Decode · tiled (WNE)** trim the result back to the original length.
  - ComfyUI's own VAE Decode does not know about the padding and keeps the extra frames.

## Habits that keep the most

- Inpaint with a latent noise mask rather than decode -> paint -> encode.
- Never decode and re-encode just to change format or resolution.
- When pixels must be edited between two VAE passes, run **VAE Decode (C2C)** with `clean` on and `restore_unchanged` afterwards.
