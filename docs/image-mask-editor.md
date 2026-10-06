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
| **B** | Brush (size = diameter, hardness, opacity in the toolbar). Pen pressure can scale size and/or opacity (toggles); mouse strokes ignore the toggles. An outline shows the brush under the pointer |
| **E** | Eraser |
| **R / O** | Rectangle / ellipse (drag; Shift = square / circle) |
| **P** | Polygon: click points, Enter or double-click closes, Backspace removes the last point |
| **L** | Lasso: freehand, closes on release |
| **G** | Bucket fill on the image colours (tolerance in the toolbar) |
| **C** | Colour range: click samples the plate colour; Shift+click adds a sample. Toolbar: space (RGB / HSV / LAB), tolerance (0–255), softness (0–128), contiguous. Live preview over the mask; Enter or Apply commits through the selection mode; Esc clears the preview |
| **S** | Smart select (SAM). Click adds a positive point, Alt+click a negative one; drag draws a box instead. The candidate shows over the image; Enter or Apply commits it through the selection mode; Esc clears it. The image is uploaded once per frame and the model's embedding is reused for every further click. Needs a SAM 2.1 model and the `sam2` Python package (see Requirements) |
| **M** | Edge refine. Paint a band over a hair or fur edge; on release ViTMatte (tiled) mattes that band only and writes soft alpha into it, leaving the rest of the mask untouched. Ctrl+Z restores the exact previous pixels. Needs ViTMatte weights (see Requirements) |
| **Add / Subtract / Intersect** | Toolbar modes for rectangle, ellipse, polygon, lasso, bucket and colour range. Alt held during a stroke swaps Add and Subtract for that stroke |
| **Alt** + brush | Subtract instead of add (brush and eraser are unchanged by the mode buttons) |
| **Ctrl+Z / Ctrl+Y** | Undo / redo on the current frame (one shared 128 MB budget across all frames) |
| **Ctrl+Shift+C / Ctrl+Shift+V** | Copy / paste the current frame's mask (paste uses the toolbar Add / Subtract / Intersect mode) |
| **[ / ]** | Brush size |
| **V** | Cycle view: Overlay → Matte → Rubylith → Outline → Image (remembered in the browser). Rubylith tints what is NOT selected red, the compositing convention; Outline draws the mask edge as a two-tone line that reads on any image. A pixel grid appears from 800% zoom |
| **Frame ▾** (toolbar menu) | Copy mask · Paste mask · Apply to all frames · Clear frame · Export PNG · Import PNG |
| **Layers ▾** (toolbar panel) | Up to 8 layers with add / subtract / intersect modes, visibility and lock. Add layer · reorder · rename (double-click) · delete. **Ctrl+Shift+N** adds a layer; **Alt+[** / **Alt+]** selects the layer below / above |
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

- Images up to 40 MP (8K UHD is 33 MP) are edited at their native resolution. The views are drawn from 512 px tiles kept under a 64 MB budget, so the browser holds the mask (1 byte per pixel), the image (4 bytes per pixel), at most 64 MB of tiles and at most 128 MB of undo. Larger images are edited at reduced resolution (a banner says so) and the mask is upscaled on save.
- For smart select on frames above 4 MP the editor sends SAM a 4 MP copy and scales the result back up; SAM works at about 1 MP internally either way.
- Undo is kept per frame under one shared 128 MB budget; switching frames does not discard another frame's stack.
- Export PNG saves the mask at the editing resolution (the image's own size, or the reduced size above 40 MP). Import PNG reads luminance, or the alpha channel when the PNG has transparency, resizes to the editing size, and commits through the toolbar selection mode as one undo step.
- Apply to all frames keeps every frame's active layer in the browser until you save, so it refuses above about 512 MB of masks and says so. For long clips set the node's `frame_mode` to `shared` instead: it uses frame 1's mask for every frame without copying it.
- **Layers.** Up to 8 layers share one stack for every frame; each frame holds its own pixels per layer. Tools edit the **active** layer; the viewer, export, save and edge refine read the **merged** stack (add / subtract / intersect, bottom to top). Copy, paste, clear and apply-to-all affect the active layer only. A locked active layer refuses edits. Painting is undoable per layer; adding, deleting, reordering, renaming and changing a layer's mode or visibility are not, so deleting a layer asks first. The 512 MB guard applies when **adding** a layer (not when switching frames). On disk, a simple single visible add layer keeps today's format (merged PNG only); a sidecar (`layers.json` plus `layers/<id>/` PNGs) is written only for multi-layer stacks or a non-default single layer.

## Requirements for smart select and edge refine

Both tools run on the server; the other tools need nothing extra.

- **Smart select (S).**
  - Needs the `sam2` Python package. If it is missing, the model list says so, the tooltip and the status line give
    the install command (`pip install git+https://github.com/facebookresearch/sam2.git`, then restart ComfyUI), and
    nothing is downloaded.
  - Models: any SAM 2.1 checkpoint in `ComfyUI/models/sam2/` (or `sams/`). Entries marked `[download]` are fetched
    from Hugging Face into `ComfyUI/models/sam2/` the first time you pick one; nothing is pre-selected, so a
    download only starts on your choice.
  - The model runs on ComfyUI's device (it follows `--cpu`) and is released when the editor closes.
- **Edge refine (M).** Needs a ViTMatte model folder (Hugging Face layout, `config.json` +
  `preprocessor_config.json`) under `ComfyUI/models/vitmatte/`. Without it the first stroke says so and the mask
  is not changed.
