# Anamorphic / Pixel-Aspect-Ratio round trip

> AI models — and ComfyUI itself — assume **square pixels**. An anamorphic
> plate (e.g. a `4448×3840` scan with a pixel aspect ratio of `1.7266` that
> displays as `2:1` in Nuke) reads here as a squarish `1.158:1` image. Run AI
> on it directly and everything comes back distorted the moment the lens
> squeeze is reapplied. These two nodes are the standard fix: **desqueeze to
> square pixels before AI, resqueeze back to the plate after.**

Nodes: **PARDesqueezeMEC** and **PARResqueezeMEC** (category **MEC/Plate**).
The identical pair also ships in NukeMax as **PAR Desqueeze / PAR Resqueeze**
(category NukeMax/Transform) — the `par_info` strings are interchangeable
between the two packs.

---

## Why your 4448×3840 plate looks wrong

If the pixels were square, `4448×3840` would be a `1.158:1` frame. But your
footage is anamorphic: Nuke's *format* carries a pixel aspect ratio of
`1.7266`, so it displays the frame stretched horizontally to `2:1`
(`4448 × 1.7266 ÷ 3840 ≈ 2.0`). ComfyUI has no concept of pixel aspect ratio —
it treats every pixel as square — so it sees the un-stretched `1.158:1` and any
generation is made for *that* shape. Bring the result back into Nuke, the
`1.7266` squeeze is reapplied, and circles become ovals, faces widen, etc.

---

## The pipeline

```
Nuke plate 4448×3840 (PAR 1.7266, displays 2:1)
        │
        ▼
  PARDesqueezeMEC  ── par_info ───────────────┐   (stretch_width)
        │                                      │
        ▼                                      │
  7680×3840  ← true 2:1, SQUARE pixels         │
        │                                      │
        ▼                                      │
  … your AI graph (SDXL / Wan / inpaint / …)   │
        │                                      │
        ▼                                      │
  PARResqueezeMEC ◀── par_info ────────────────┘
        │
        ▼
  4448×3840  ← exact original pixel dims, back into Nuke
```

Everything the resqueeze needs to undo the transform travels in the `par_info`
string, so the round trip restores the **exact** original `W×H` — even if your
AI graph changed the resolution (an upscale, a crop-and-stitch).

---

## PARDesqueezeMEC — parameters

| Parameter | Default | Effect |
|---|---|---|
| `image` | — | The anamorphic plate |
| `par_preset` | `ARRI 4448x3840→2:1 (1.7266)` | Named pixel aspect ratio. Presets: square `1.0`, anamorphic `2.0` / `1.8` / `1.5` / `1.33`, the `1.7266` 2:1 case, NTSC DV `0.9091`, PAL DV `1.0940`, or `custom` |
| `pixel_aspect` | `1.7266` | Used only when `par_preset = custom`. `>1` = pixels wider than tall |
| `method` | `stretch_width` | `stretch_width` keeps every scanline (recommended — `4448×3840 → 7680×3840`); `squash_height` keeps the pixel count low (`4448×3840 → 4448×2224`) |
| `filter` | `bicubic` | Resample filter (`bicubic` / `bilinear` / `nearest` / `area`) |

**Outputs:** `image` (square-pixel frame) and `par_info` (JSON — wire this into
the resqueeze node's `par_info` input).

## PARResqueezeMEC — parameters

| Parameter | Default | Effect |
|---|---|---|
| `image` | — | The AI-processed square-pixel frame |
| `par_info` | — | Connect from the desqueeze node's `par_info` output |
| `filter` | `bicubic` | Resample filter |

**Outputs:** `image` (restored to the plate's exact original pixel dimensions)
and `info` (JSON of what was restored).

---

## Recipes

**Standard desqueeze → AI → resqueeze:**

```
Load plate → PARDesqueezeMEC (stretch_width, 1.7266) → [square 7680×3840]
   → your AI nodes → PARResqueezeMEC (par_info wired from desqueeze)
   → Save / back to Nuke
```

**Custom camera PAR:** set `par_preset = custom` and type your exact pixel
aspect in `pixel_aspect` (e.g. `1.79` for a specific ARRI open-gate mode).

**Non-anamorphic (square) footage:** set `par_preset = square 1.0` — the node
is a no-op and passes the image through untouched, so you can leave it in a
template graph safely.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Resqueeze errors "par_info is not valid" | The `par_info` input isn't wired from a PARDesqueeze node (or is empty). Connect the desqueeze node's `par_info` output |
| Output still looks stretched in Nuke | Check your Nuke *format's* pixel aspect matches what you desqueezed with. The resqueeze restores pixel **dimensions**; Nuke reapplies the PAR |
| Want square-pixel delivery instead of a Nuke round trip | Skip the resqueeze — the desqueezed frame *is* correct at `1.0` PAR for any square-pixel target |

---

## License

Apache-2.0. Behaviour mirrors Nuke's Reformat pixel-aspect handling.
