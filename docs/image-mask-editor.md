# Image Mask Editor (C2C)

> **Category:** `C2C/Masking` · **VRAM tier:** 0 (no model; the result stage is tensor math on the CPU or GPU)
> **Files:** [`nodes/image_mask_editor/`](../nodes/image_mask_editor/) ·
> [`js/image_mask_editor.js`](../js/image_mask_editor.js) + [`js/image_mask_editor/`](../js/image_mask_editor/)

Paint a mask over the image that actually reaches the node, at native resolution, in a full-window
editor. The node keeps only an editor id in the workflow; the masks live on the server and survive
restarts.

## What the editor shows

| Source wired to `image` | Frames shown | Needs a run? |
|---|---|---|
| LoadImage | the selected file | no |
| C2C or VHS video/sequence loader | every frame of the plan, exact | no |
| anything else (crop, upscale, VAE decode, a batch...) | the exact frames this node received in its last run | once |
| anything else, before the first run | the nearest upstream preview, with a banner saying so | - |

After the first run the node writes its input frames to `temp/c2c_ime/<editor_id>/`:
- one PNG per frame plus a small JPEG for the strip;
- unchanged input is detected by a content fingerprint and not written again;
- inputs over 256 MP in total are not recorded, and `info` says so.

## Tools

| Key | Tool / action |
|---|---|
| **B** | Brush (size = diameter, hardness, opacity in the toolbar). An outline shows the brush under the pointer |
| **E** | Eraser |
| **R / O** | Rectangle / ellipse (drag; Shift = square / circle) |
| **P** | Polygon: click points, Enter or double-click closes, Backspace removes the last point |
| **L** | Lasso: freehand, closes on release |
| **G** | Bucket fill on the image colours (tolerance in the toolbar) |
| **C** | Colour range: click samples the plate colour; Shift+click adds a sample. Toolbar: space (RGB / HSV / LAB), tolerance (0–255), softness (0–128), contiguous. Live preview over the mask; Enter or Apply commits through the selection mode; Esc clears the preview |
| **Add / Subtract / Intersect** | Toolbar modes for rectangle, ellipse, polygon, lasso, bucket and colour range. Alt held during a stroke swaps Add and Subtract for that stroke |
| **Alt** + brush | Subtract instead of add (brush and eraser are unchanged by the mode buttons) |
| **Ctrl+Z / Ctrl+Y** | Undo / redo (per frame, capped at 128 MB) |
| **[ / ]** | Brush size |
| **V** | Overlay / matte / image view |
| **F / 1** | Fit / 100% |
| Wheel | Zoom at the pointer |
| **Space** + drag, or middle drag | Pan |
| **, / .** or the frame strip | Previous / next frame (batches) |
| **Enter / Esc** | Save and close / cancel (asks before discarding edits). Enter applies a colour preview or closes a polygon first; Esc clears an in-progress shape or colour preview before closing |
| **?** | Shortcut list |

The editor is a modal dialog: while it is open, ComfyUI's own shortcuts and graph undo do not react to
keys pressed in it.

## Parameters

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `image` | IMAGE | - | - | The image or batch to mask; it sets the mask size and frame count |
| `editor_id` | STRING | auto | - | Set by the editor; copies of the node get their own id and a copy of the masks |
| `frame_mode` | combo | `per_frame` | `per_frame` / `shared` | One mask per frame, or frame 0's mask for every frame |
| `grow` | INT | 0 | -256 - 256 | Grow (positive) or shrink (negative), in pixels |
| `feather` | FLOAT | 0 | 0 - 256 | Gaussian edge softening, sigma = 0.33 x value (the Smart Crop/Stitch feather) |
| `threshold` | FLOAT | 0 | 0 - 1 | Binarise after feather (0 keeps the soft mask) |
| `invert` | BOOLEAN | false | - | Invert after threshold |
| `input_mask` | MASK | - | optional | A mask to combine with the painted one |
| `combine` | combo | `replace` | `replace` / `add` / `subtract` / `intersect` | How the painted mask and `input_mask` combine |

All result-stage parameters are applied in Python every run and never baked into the stored pixels.

## Outputs

| Output | Type | Description |
|---|---|---|
| `mask` | MASK | The final mask, [B, H, W] in 0-1 |
| `preview` | IMAGE | The input with the mask tinted over it |
| `info` | STRING | Frame count, stored masks, mode and any notes (resized frames, recording skipped) |

## Storage and caching

- Masks are PNGs in `input/c2c_masks/<editor_id>/`, one file per frame, replaced atomically on save.
- The masks' content hash is part of the node's cache key: an edit re-runs the graph, and an unchanged graph
  stays cached.

## Limits

- Images over 16 MP are edited at reduced resolution (a banner says so) and the mask is upscaled on save.
- Undo history is per frame and resets when you switch frames.
- SAM clicks and the edge-refine brush arrive in a later slice (ledger L7.38).
