# Paint Suite (moved)

The two Paint Suite nodes were merged into other nodes. Saved workflows load them as the new node, in the right mode,
with their values and links. There is nothing to install or rewire.

| Old node | Now | Mode | Guide |
|---|---|---|---|
| MEC Advanced Paint Canvas (`MECAdvancedPaintCanvas`) | Image Mask Editor (`ImageMaskEditorC2C`) | `mode` = `paint`, `canvas_layer` = `colour` | [image-mask-editor.md](image-mask-editor.md#modes) |
| MEC Context Inpainter (`MECContextInpainter`) | Inpaint Stitch Pro (`InpaintStitchProMEC`) | `stitch_mode` = `context blend` | [inpaint-suite.md](inpaint-suite.md) |

- **Paint Canvas.** Its controls are now the Image Mask Editor's `canvas_*` controls (for example `brush_size` is
  `canvas_brush_size`), shown when `mode` is `paint` and `canvas_layer` is `colour`; `reference_image` is now `image`.
  Its outputs `painted_image` and `processed_mask` are the editor's `painted_image` and `mask`.
- **Context Inpainter.** Its controls are now Inpaint Stitch Pro's `context_*` controls, shown when `stitch_mode` is
  `context blend`; its `debug_mask` output is now `blend_mask_used`. The face prompt inputs were dropped: they were
  only written to the log and never sampled.

The exact input renames are listed in [MIGRATION.md](MIGRATION.md).
