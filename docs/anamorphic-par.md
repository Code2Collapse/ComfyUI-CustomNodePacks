# Anamorphic / pixel-aspect round trip

ComfyUI and the image models treat every pixel as square. An anamorphic plate, for example a `4448×3840` scan with a
pixel aspect ratio (PAR) of `1.7266` that Nuke displays as `2:1`, reads here as a squarish `1.158:1` picture. Anything
generated for that shape comes back distorted once the squeeze is reapplied. These two nodes convert the plate to
square pixels before the AI step and back to the plate's pixel size after it.

| Node | Use it to |
|---|---|
| [PAR Desqueeze](#par-desqueeze-pardesqueezemec) (`PARDesqueezeMEC`) | Turn non-square-pixel frames into square pixels, optionally with an overscan border |
| [PAR Resqueeze](#par-resqueeze-parresqueezemec) (`PARResqueezeMEC`) | Bring the processed frames back to the plate's original width × height, and check the round trip |

### Why a 4448×3840 plate looks wrong

With square pixels, `4448×3840` is a `1.158:1` frame. Nuke's format carries a PAR of `1.7266`, so it displays the
frame stretched to `2:1` (`4448 × 1.7266 ÷ 3840 ≈ 2.0`). ComfyUI has no pixel aspect, so it sees the unstretched
`1.158:1`. Bring that result back into Nuke, the `1.7266` squeeze is reapplied, and circles become ovals.

### The round trip

```
Plate 4448×3840 (PAR 1.7266) ─┬─▶ PAR Desqueeze (stretch_width)
                              │      ├─ image 7680×3840, square pixels ─▶ your AI graph ─▶ image ──┐
                              │      └─ par_info ──────────────────────────────────────▶ par_info ──┤ PAR Resqueeze
                              └─ (optional) ─────────────────────────────────────────▶ reference ─┘   └─ image 4448×3840 ─▶ Save / Nuke
```

Everything Resqueeze needs travels in the `par_info` string. Resqueeze always returns the plate's original
width × height, even if the AI graph changed the resolution. The geometry is exact; with a PAR other than 1.0 the
pixels are resampled on the way there and back, so they are not bit-identical. Wire `reference` to measure the difference (see PAR Resqueeze below).

## PAR Desqueeze (`PARDesqueezeMEC`)

**Menu:** 🐺 C2C/🧰 Core/Plate, shown as "PAR Desqueeze (anamorphic → square px)" ·
**Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#pardesqueezemec)

### What it is for

Resampling frames shot with non-square pixels (anamorphic lens, DV, HDV) to square pixels, and recording what it did
in `par_info` so PAR Resqueeze can undo it.

### When to pick it

Before any AI step on anamorphic or DV footage. For square-pixel footage you do not need it: with `par_preset` =
`square 1.0` the image passes through unchanged (an overscan border is still added if you ask for one), so it is safe
to leave in a template graph.

### Key controls

- `par_preset`: the camera or format. Presets: `square 1.0`, `anamorphic 2x (2.0)`, `anamorphic 1.8x (1.8)`,
  `ARRI 4448x3840→2:1 (1.7266)` (the default), `anamorphic 1.5x (1.5)`, `anamorphic 1.33x (1.33)`,
  `NTSC DV (0.9091)`, `PAL DV (1.0940)`, or `custom` to type the value in `pixel_aspect`.
- `method`: `stretch_width` keeps every scanline (`4448×3840` -> `7680×3840`); `squash_height` keeps the pixel count
  low (`4448×3840` -> `4448×2224`).
- `filter`: `bicubic`, `bilinear`, `nearest` (no new values; for masks and IDs) or `area`. `bicubic` is clamped to
  the input's own value range, so scene-linear highlights and negatives survive.
- `lens_metadata` (added 2026-11-19, commit f6d8917): camera and lens facts as `key: value` lines or a JSON object,
  for example `camera: ARRI Alexa 35`, `lens: Cooke 2x`, `squeeze: 2`. They are stored in `par_info` and given back
  by PAR Resqueeze. If a `pixel_aspect`, `par`, `squeeze`, `anamorphic_squeeze` or `lens_squeeze` value differs from
  the PAR used by more than 0.5 %, a warning is recorded.
- `overscan_percent` and `overscan_multiple` (added 2026-11-19, commit f6d8917): add an edge-extended border of this
  percent of each side, so the AI step has room at the edges, then round the padded size up to a multiple (for
  example 16 or 64 for a model's size rule). PAR Resqueeze crops the border off again. 0 / 1 = no border, as before.

### Reading the outputs

| Output | What it holds |
|---|---|
| `image` | The square-pixel frames, with the overscan border if you asked for one |
| `par_info` | JSON: `orig_width`, `orig_height`, `pixel_aspect`, `method`, `filter`, `square_width`, `square_height`, `display_aspect`; plus `lens` and `warnings` with lens metadata, and `overscan` (left / right / top / bottom, padded size, percent, multiple) with a border |
| `overscan_mask` | 1 on the added border, 0 on the real picture; all 0 without overscan. Use it to let the AI step repaint only the border |

### Common mistakes

| Message | What to do |
|---|---|
| `pixel_aspect must be > 0` | With `custom`, type a positive `pixel_aspect` |
| `lens_metadata starts with '{' but is not valid JSON` | Fix the JSON, or write `key: value` lines instead |
| `lens_metadata line '...' is not 'key: value' or 'key=value'` | One `key: value` per line (or separated by `;`) |

## PAR Resqueeze (`PARResqueezeMEC`)

**Menu:** 🐺 C2C/🧰 Core/Plate, shown as "PAR Resqueeze (back to plate)" ·
**Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#parresqueezemec)

### What it is for

Returning processed square-pixel frames to the plate's original pixel size, so Nuke can reapply the PAR. It leaves
the pixel aspect itself to Nuke.

### When to pick it

At the end of every graph that started with PAR Desqueeze, when the result goes back to the anamorphic plate.
For a square-pixel delivery, skip it: the desqueezed frame is already correct at PAR 1.0.

### Key controls

- `par_info`: wire it from PAR Desqueeze, unchanged.
- `filter`: the same choices as on PAR Desqueeze.
- `reference` (optional, added 2026-11-19, commit f6d8917): the original plate, as it went into PAR Desqueeze. The
  result is then compared with it.

If `par_info` holds an overscan border, the node first crops it off, scaled to the size of the incoming frames, then
resizes to the original size.

### Reading the outputs

| Output | What it holds |
|---|---|
| `image` | The frames at the plate's original width × height |
| `info` | JSON: `restored_width`, `restored_height`, `pixel_aspect`, and `overscan_removed` when a border was cropped |
| `pixel_aspect` | The plate's PAR from `par_info`, as a number to wire into a writer or format |
| `metadata` | JSON: size, `pixel_aspect`, `display_aspect`, `method`, a `nuke_format` line (`W H 0 0 W H PAR c2c_plate`), the lens metadata and warnings from PAR Desqueeze, the overscan removed, and with `reference` the `validation` block |
| `difference` | With `reference`: \|result − reference\| per pixel (mean of the channels, 0-1). All 0 without a reference |

The `validation` block: `result_size`, `reference_size`, `result_frames`, `reference_frames`, `geometry_ok`,
`frame_count_ok`, and when the sizes match `frames_compared`, `max_abs_diff`, `mean_abs_diff`, `psnr_db` and
`pixels_over_1_255_pct` (the percent of pixels that changed by more than 1/255). When the sizes differ there is no
pixel comparison and a `note` asks whether the `par_info` came from this plate.

### Common mistakes

| Message or symptom | What to do |
|---|---|
| `par_info is not valid PARDesqueezeMEC output` | `par_info` is empty or not from PAR Desqueeze: wire PAR Desqueeze's `par_info` output |
| `par_info has a broken overscan entry` | The `par_info` text was edited: wire it unchanged |
| `the overscan crop ... does not fit the ... input` | Wire the `par_info` of the PAR Desqueeze that made these frames, unchanged |
| The result still looks stretched in Nuke | The node restores pixel **dimensions**; check that your Nuke format's pixel aspect matches the one you desqueezed with |

## The NukeMax pair

ComfyUI-NukeMaxNodes has its own PAR Desqueeze / PAR Resqueeze (`NukeMax_PARDesqueeze`, `NukeMax_PARResqueeze`). They
use the same presets and the same core `par_info` keys (`orig_width`, `orig_height`, `pixel_aspect`, ...), but they
do not have the 2026-11-19 additions: no lens metadata, overscan, reference check or extra outputs. A `par_info` with
no overscan works in either pack's Resqueeze. A `par_info` made with overscan needs this pack's PAR Resqueeze: the
NukeMax one ignores the border and would squeeze it into the picture.

## License

Apache-2.0.
