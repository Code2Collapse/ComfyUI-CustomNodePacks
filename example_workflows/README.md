# Example workflows

Drag any `.json` file from this folder onto the ComfyUI canvas, or open it from **Templates** in ComfyUI,
where this pack's examples appear with a thumbnail (`<name>.jpg`). Every example needs only this pack and
ComfyUI itself.

Before you run one, pick your own file in the image or video loader, and pick a model in the SAM loader:
either one you have installed, or one of its `[download]` entries.

| File | What it shows | Nodes |
|---|---|---|
| `basic_sam_segmentation.json` | Click points on an image, SAM turns them into a mask, preview it as an overlay | SAM Model Loader, Mask Edit (points_bbox), SAM Mask Generator, Mask Preview Overlay |
| `sam_vitmatte_pipeline.json` | Points and boxes into SAM + ViTMatte for a soft alpha on hair and fur | SAM Model Loader, Mask Edit (points_bbox), SAM + ViTMatte Pipeline |
| `mask_matting_sam3_vitmatte.json` | One click into Mask + Matting (cascade pipeline: SAM 3.1 then ViTMatte, hair preset), alpha preview | Mask Edit (points_bbox), Mask + Matting, Convert Mask to Image |
| `mask_editing_toolkit.json` | Draw a circle and a rectangle, expand and blur one, union them, adjust gamma, preview each stage | Mask Edit (draw_advanced, transform), Mask Composite Advanced, Mask Math, Mask Preview Overlay |
| `draw_shape_multi.json` | Two parametric shapes (rectangle, ellipse) combined into one mask | Mask Edit (draw_shape), Mask Composite Advanced, Mask Preview Overlay |
| `spline_mask_editor.json` | Draw a spline mask, invert it, and send its control points to SAM as prompts | Spline Mask (edit), Mask Math, SAM Model Loader, SAM Mask Generator, Mask Preview Overlay |
| `bbox_pipeline.json` | Bounding box from a mask, padded, cropped out, and drawn back as a mask | BBox From Mask, BBox Pad, BBox Crop, BBox To Mask, Mask Preview Overlay |
| `video_mask_propagation.json` | Mark the subject on the first frame and propagate the mask through the clip | Load Video (C2C), Batch Range (C2C), Mask Edit (points_bbox), Mask Tracker (propagate) |
| `video_motion_detection.json` | Motion masks from a clip, overlaid on the video, plus a bounding box of the motion | Load Video (C2C), Mask Tracker (motion), Mask Preview Overlay, BBox From Mask |
| `master_workflow.json` | Points into a SAM mask, ProPainter temporal inpaint, stabilise, save | SAM Model Loader, Mask Edit (points_bbox), SAM Mask Generator, ProPainter (temporal), Video Stabilizer (auto) |

Mask Preview Overlay, Mask Composite Advanced, Mask Math and the four BBox nodes show "(legacy)" in the node
menu: they were restored so old workflows keep loading and have no newer equivalent yet. They work as before.

## What changed on 2026-10-05

All ten examples were rebuilt with the current nodes and re-saved from a live ComfyUI (core 0.36.0,
frontend 1.52.7), then reloaded in both the classic and the Nodes 2.0 renderer with no missing node, no
invalid value and no browser error.

- Nodes that were merged into newer ones now use the newer node in the matching mode, for example Points
  Mask Editor became Mask Edit in `points_bbox` mode. The full old-to-new table is in
  [`docs/MIGRATION.md`](../docs/MIGRATION.md); old workflows of your own are converted by ComfyUI's
  **Replace Node** button.
- Several originals could not run: required inputs were left unconnected (the Mask Preview Overlay image in
  `bbox_pipeline` and `mask_editing_toolkit`, the SAM model and image in `spline_mask_editor`), one link
  pointed at a node that did not exist (`mask_matting`), and some links used outputs no node ever had. These
  are now wired the way each example describes.
- Video examples load with **Load Video (C2C)** instead of VideoHelperSuite's loader, so they no longer need
  that pack.
- `mask_matting_sam2_vitmatte.json` is now `mask_matting_sam3_vitmatte.json`: SAM 2 was removed from
  Mask + Matting, and SAM 3.1 replaces it.
