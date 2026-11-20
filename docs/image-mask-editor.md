# Image Mask Editor (C2C)

One node for drawing masks by hand: paint over the image in a full-window editor, click points and boxes for SAM,
draw a parametric shape, a spline, a keyed roto shape or video keyframes. The masks are stored on the server, so they
survive a restart; the workflow keeps only an editor id.

## Image Mask Editor (`ImageMaskEditorC2C`)
**Menu:** 🐺 C2C/🧰 Core/Mask · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#imagemaskeditorc2c)
No model is needed, except for two optional editor tools: smart select (SAM 2.1) and edge refine (ViTMatte), see
[Requirements](#requirements-for-smart-select-and-edge-refine).

### What it is for

Paint a mask over the exact image that reaches this node, at its native resolution. `grow`, `feather`, `threshold`
and `invert` are node parameters applied every run, never baked into the stored mask, so you can change them without
repainting.

### When to pick it

- Draw or correct a mask by hand: this node.
- Key, grade, grow or combine an existing matte: [Mask Tools](mask-editing.md).
- Let a model find the object: [Segment (C2C) or the SAM nodes](sam-segmentation.md); for a matte with soft edges,
  [Mask + Matting](mask-matting-pipeline.md).

### Minimal wiring

```
Load Image ──► image ──► Image Mask Editor ──► mask ──► (inpaint / composite node)
                              ▲                 ──► preview ──► Preview Image
               optional: input_mask (combined with `combine`)
```

Click **✏️ Edit mask** on the node (`paint` mode), draw, press Enter (or **💾 Save**), then queue.

## Modes

`mode` picks how the mask is drawn. Each mode shows only its own controls (prefixed with the mode name) and its own
drawing UI, and runs the code of the node it replaced, so it gives that node's result.

| `mode` | Draw with | Controls | Replaces | Outputs it fills |
|---|---|---|---|---|
| `paint` (default), `canvas_layer` = `mask` | the full-window editor below (brush, shapes, lasso, fill, colour range, smart select, edge refine) | `frame_mode`, `combine` | - (this node) | `mask`, `preview` |
| `paint`, `canvas_layer` = `colour` | the paint canvas: colour strokes over the image; their alpha is also mask | `canvas_*` | Paint Canvas (`MECAdvancedPaintCanvas`) | `painted_image`, `mask` |
| `points + box` | clicks (positive / negative) and boxes on the image | `points_*` | Mask Edit `points_bbox` | `positive_coords`, `negative_coords`, `bboxes`, `neg_bboxes`, `points_json`, `bbox_json`, `primary_bbox` |
| `shape` | a parametric shape (12 kinds) or, with `shape_input` = `JSON`, shapes from `shape_json` | `shape_*` | Mask Edit `draw_shape` / `draw_advanced`, Draw Shape, Mask Draw Frame | `mask` |
| `spline` | the spline canvas; `spline_fill` = `mask` fills or strokes the spline, `flow pattern` draws a procedural pattern along it (ribbon, wave, dust, smoke...), animated over frames | `spline_*` | Spline Mask `edit` / `flow_path`, Spline Mask Editor | `mask`, `positive_coords`, `bbox_json`, `primary_bbox`, `spline_data_out` |
| `roto` | Bezier shapes keyed over frames, on the same spline canvas | `roto_*` | Vector Roto (`VectorRotoMEC`) | `mask` |
| `video` | painted keyframes, tweened between (Open Video Mask Editor) | `video_*` | Video Mask Editor (`VideoMaskEditorMEC`) | `mask` |

- `grow`, `feather`, `threshold` and `invert` apply to every mode, after it.
- `input_mask` is combined with `combine` in `paint`, `spline` and `roto`; `points + box` and `shape` draw onto it;
  `video` keeps it on the frames your keyframes do not fill.
- `image` is optional. `paint` and `video` need it. `points + box`, `shape` and `spline` take their size from it when
  it is connected, otherwise from their own width / height controls; `roto` and the colour layer use their own size.
- The non-drawing modes of the old Mask Edit and Spline Mask nodes moved elsewhere: Mask Edit `transform` is
  [Mask Tools](mask-editing.md) `transform`; Mask Edit `bbox_smooth` and Spline Mask `track` are Mask Track
  (`MaskTrackerMEC`) `bbox_smooth` / `spline_track`.

Saved workflows with any of the replaced nodes load as this node in the right mode, with their values and links
(full list: [MIGRATION.md](MIGRATION.md)).

## What the editor shows

| Source wired to `image` | Frames shown | Needs a run? |
|---|---|---|
| LoadImage | the selected file | no |
| C2C or VHS video/sequence loader | every frame of the plan, exact | no |
| anything else (crop, upscale, VAE decode, a batch...) | the exact frames this node received in its last run | once |
| anything else, before the first run | the nearest upstream preview, with a banner saying so | - |

After a run the node writes its input frames to `temp/c2c_ime/<editor_id>/`:
- one PNG per frame plus a small JPEG for the frame strip;
- unchanged input is recognised and not written again;
- inputs over 256 MP in total are not recorded, and `info` says so.

After you switch workflow tabs, the node shows its last result again without a new run (commit e98b458; that
commit's live check was made on Video Frame Player, which uses the same helper).

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
| **V** | Cycle view: Overlay → Matte → Rubylith → Outline → Image (remembered in the browser). Rubylith tints what is NOT selected red, the compositing convention; Outline draws the mask edge as a two-tone line. A pixel grid appears from 800% zoom |
| **Frame ▾** (toolbar menu) | Copy mask · Paste mask · Apply to all frames · Clear frame · Export PNG · Import PNG |
| **Layers ▾** (toolbar panel) | Up to 8 layers with add / subtract / intersect modes, visibility and lock. Add layer · reorder · rename (double-click) · delete. **Ctrl+Shift+N** adds a layer; **Alt+[** / **Alt+]** selects the layer below / above |
| **F / 1** | Fit / 100% |
| Wheel | Zoom at the pointer |
| **Space** + drag, or middle drag | Pan |
| **, / .** or the frame strip | Previous / next frame (batches) |
| **Enter / Esc** | Save and close / cancel (asks before discarding edits). Enter applies a colour preview or closes a polygon first; Esc clears an in-progress shape or colour preview before closing. With a toolbar button focused (Tab), Enter and Space press that button instead, so Tab to Cancel + Enter does not save |
| **?** | Shortcut list |

The editor is a modal dialog: while it is open, ComfyUI's own shortcuts and graph undo do not react to keys pressed in
it.

## Key controls

- `frame_mode` (`paint`): `per_frame` uses stored mask *i* for frame *i*; `shared` uses frame 0's mask for every
  frame.
- `grow`: grow (positive) or shrink (negative) in pixels. Any non-zero value first makes the edge hard (every partly
  covered pixel counts); `feather` softens it again.
- `feather`: Gaussian edge softening, sigma = 0.33 × the value (the same feather as the Inpaint Crop / Stitch nodes).
- `threshold`: binarise after feather (0 keeps the soft mask). `invert`: flip after threshold.
- `combine`: `replace`, `add`, `subtract` or `intersect` with `input_mask`. Without `input_mask` only `replace`
  applies.
- The mode controls (`canvas_*`, `points_*`, `shape_*`, `spline_*`, `roto_*`, `video_*`) are listed in
  [NODE_REFERENCE](../NODE_REFERENCE.md#imagemaskeditorc2c).

## Reading the outputs

| Output | What it is |
|---|---|
| `mask` | The final mask, [B, H, W] in 0–1: the mode's mask, combined with `input_mask`, then grow, feather, threshold, invert |
| `preview` | The input with the mask tinted over it; the mask as a grey picture when there is no image of the same size |
| `info` | `paint`: `frames=<B>; stored=<masks on disk>; mode=<frame_mode>` plus notes (resized frames, frame recording skipped, `colour layer`). Other modes: the text of the replaced node's own engine, so a migrated node's `info` reads as before |
| `positive_coords` ... `primary_bbox` | `points + box`: the points and boxes for SAM / SeC; `spline`: points along the spline and its box. Empty (`[]`, `[0,0,0,0]`) in other modes |
| `spline_data_out` | `spline`: the spline, for a tracker or another spline node |
| `painted_image` | `paint`: the colour layer over the image (the image itself when there are no colour strokes) |

The outputs after `info` were appended when the modes were added, so links saved before keep their slots.

## Storage and caching

- Masks are PNGs in `input/c2c_masks/<editor_id>/`, one file per frame, replaced atomically on save. Copies of the
  node get their own `editor_id` and a copy of the masks.
- The masks' content is part of the node's cache key: an edit re-runs the graph, an unchanged graph stays cached.

## Limits

- Images up to 40 MP (8K UHD is 33 MP) are edited at their native resolution. The views are drawn from 512 px tiles
  kept under a 64 MB budget, so the browser holds the mask (1 byte per pixel), the image (4 bytes per pixel), at most
  64 MB of tiles and at most 128 MB of undo. Larger images are edited at reduced resolution (a banner says so) and the
  mask is upscaled on save.
- For smart select on frames above 4 MP the editor sends SAM a 4 MP copy and scales the result back up.
- Undo is kept per frame under one shared 128 MB budget; switching frames does not discard another frame's stack.
- Export PNG saves the mask at the editing resolution (the image's own size, or the reduced size above 40 MP). Import
  PNG reads luminance, or the alpha channel when the PNG has transparency, resizes to the editing size, and commits
  through the toolbar selection mode as one undo step.
- Apply to all frames keeps every frame's active layer in the browser until you save, so it refuses above about
  512 MB of masks and says so. For long clips set `frame_mode` to `shared` instead: frame 0's mask is used for every
  frame without copying it.
- **Layers.** Up to 8 layers share one stack for every frame; each frame holds its own pixels per layer. Tools edit
  the **active** layer; the viewer, export, save and edge refine read the **merged** stack (add / subtract /
  intersect, bottom to top). Copy, paste, clear and apply-to-all affect the active layer only. A locked active layer
  refuses edits. Painting is undoable per layer; adding, deleting, reordering, renaming and changing a layer's mode or
  visibility are not, so deleting a layer asks first. The 512 MB guard applies when **adding** a layer. On disk, a
  single visible add layer keeps the plain format (merged PNG only); a sidecar (`layers.json` plus `layers/<id>/`
  PNGs) is written only for multi-layer stacks or a non-default single layer.

## Requirements for smart select and edge refine

Both tools run on the server; the other tools need nothing extra.

- **Smart select (S).**
  - Needs the `sam2` Python package. If it is missing, the model list says so, the tooltip and the status line give
    the install command (`pip install git+https://github.com/facebookresearch/sam2.git`, then restart ComfyUI), and
    nothing is downloaded.
  - Models: a SAM 2.1 checkpoint in `ComfyUI/models/sam2/`, `sams/` or `sam3/`. Entries marked `[download]` are
    fetched from Hugging Face into `ComfyUI/models/sam2/` the first time you pick one; nothing is pre-selected, so a
    download only starts on your choice.
  - The model runs on ComfyUI's device (it follows `--cpu`) and is released when the editor closes.
- **Edge refine (M).** Needs the `transformers` package and a ViTMatte model folder (Hugging Face layout,
  `config.json` + `preprocessor_config.json`) under `ComfyUI/models/vitmatte/`. Without it the first stroke says so
  and the mask is not changed.

## Common mistakes

- **"Image Mask Editor (paint): connect an image to paint on"** - `paint` needs `image` (or colour strokes with
  `canvas_layer` = `colour`). `video` likewise stops with "connect the video frames to image".
- **The editor shows an upstream preview with a banner, not the real frames.** The source is not a LoadImage or a
  video loader; queue once so the node records the frames it actually receives.
- **Edits do not reach the output.** Press Enter or **💾 Save** in the editor; Esc and Cancel discard.
- **Apply to all frames refuses on a long clip.** Use `frame_mode` = `shared`.
