# The Matte viewer on Mask + Matting (L7.114 slice A, owner A14)

Approved design: `D:/PROJECT/Custom_Nodes/docs/evidence/A14/design_matte_segment_viewer.md` with the owner's answers
(`owner_answers_2026-10-15.md`, round 5): hidden helper nodes through ComfyUI's own queue; execution inside the node
with a stage cache; SAM checkpoints from `models/sam3.1/` and `models/sam3/`, strongest first. The reference (Comfy
Pro Alpha Generator) is PolyForm Noncommercial: only its behaviour notes were used, none of its code, strings or
layout values.

This file records what was built and what was measured. Slice B (the Segment (C2C) node) and slice C (Mask Refiner /
Mask Track in preview + QC mode) are not part of it.

## 1. Pieces

| Piece | Where | What it does |
|---|---|---|
| `picks` input | `nodes/mask_matting/node.py`, `_picks.py` | Appended optional STRING (last input). Empty: the node runs as before. Set: replaces the chosen backend's prompt inputs (section 2). |
| Stage cache | `nodes/mask_matting/_stage_cache.py` | Viewer-driven runs keep the TRACK and EDGE stages (section 3). |
| Plate Proxy | `nodes/c2c_viewer/plate_proxy.py` | Hidden helper: the plate as JPEG proxies (at most 960 px, at most 240 frames; every k-th frame of a longer clip, the last always). |
| Detect | `nodes/c2c_viewer/detect.py` | Hidden helper: one frame + words / clicks / boxes -> one mask (PNG, alpha channel) per object, box, label, score. Engine: ComfyUI's own SAM 3 / 3.1 (core 0.39+; owner answer 2026-10-28, Q4), called at run time - words through its detector head, clicks and boxes through its SAM decoder; CNP's own SAM backends only on an older core. |
| Viewer Result | `nodes/c2c_viewer/result.py`, `qc.py` | Hidden helper: the rendered alpha and plate as proxies + the QC numbers. |
| Viewer | `js/c2c_viewer/` (`core.js` pure logic, `transport.js` helper prompts, `viewer.js` the panel) | The face of the node. |

The three helpers are registered (so a helper prompt validates) with `DEV_ONLY = True`: frontend 1.53.10 hides
`dev_only` node definitions from the node library and the search box (node-def filter `core.dev_only`) and sets
`skip_list` so the Add Node menu leaves them out, unless the user turns on Comfy.DevMode. Read in the installed
frontend bundle (settingStore / GraphView assets, 2026-10-15); not yet verified in a running browser.

## 2. The selection (`picks`)

Schema version 1, coordinates 0..1 on the plate:

```json
{"v": 1, "frame": 12, "text": "person", "model": "sam3.1/sam3.1_multiplex_fp16.safetensors",
 "subject": {"x": 0.41, "y": 0.52, "box": [0.30, 0.20, 0.55, 0.90], "label": "person"},
 "excluded": [{"x": 0.72, "y": 0.50, "label": "person"}],
 "add": [{"x": 0.47, "y": 0.61}], "cut": [{"x": 0.40, "y": 0.30}],
 "plate": {"frames": 120, "w": 1920, "h": 1080}}
```

| Backend | What the selection becomes |
|---|---|
| segment + matte (cascade) | positive_coords = subject + added points, negative_coords = excluded + cut points, pos_bbox = subject box widened over the added parts (+4 %), frame_annotation = pick frame. The wired prompt inputs are replaced (the info output lists which). |
| segment + matte (ONYX) | The same, as ONYX's single-object legacy prompts. scene_prompts and the external_mask prompt are replaced. |
| SAM + ViTMatte | points_json (labels 1 / 0) and bbox_json. It reads the points on every frame (it does not track). |
| SeC + MatAnyone2 | points_json and bbox_json. SeC prompts the clip's first frame, so a pick after it starts the clip on the pick frame; the frames before it come back empty (alpha 0, the plate in despilled / un-premultiplied preview) and the info output lists them as `viewer.empty_frames` [first, pick) with a note (owner answer 2026-10-28, Q3). |
| remove background | No prompts; the selection only chooses the frames. |

A pick after frame 0 with `tracking_direction` = forward tracks both ways (otherwise every frame before the pick
would be empty); the info output says so. The text of Find is kept for the viewer and not sent to the segmenter:
text plus points would make SAM detect every match again. The viewer's own preview adds `"range": {"start", "count"}`;
the saved widget never has it. Refusals are sentences ("The selection was picked on frame 57, but the plate has 40
frames. Open the viewer ... and pick again").

## 3. Stage cache

Viewer-driven runs only (picks set). Keys: plate fingerprint (SHA-1 of every byte: 1.6 GB/s measured, 30 x 1080p
float32 frames, this machine) + selection-derived prompts + stage settings + which weight files are installed (so
placing a model invalidates the model-free result that ran without it). TRACK = segmenter / ONYX tracking; EDGE =
trimap + matter (+ ONYX stabilising); FINISH (holdout / core, edge mode, despill, light wrap, split masks,
diagnostics) always runs. The three engines (SAM + ViTMatte, SeC + MatAnyone2, remove background) are one stage each
(their code is the shipped file). Byte-bounded LRU, CPU only, `C2C_MATTE_CACHE_MB` (default 1024, 0 = off).
Tests: `tests/test_matte_stage_cache.py` counts the stages that ran.

## 4. QC thresholds (measured)

Computed at most 640 px on the long side (`qc.QC_SIDE`); edge band = coarse mask dilated minus eroded by 1 % of the
long side. Synthetic clip (`tests/test_viewer_qc.py::make_clip`): 10 frames, 320 x 180, a textured ellipse with 24
hair strands about 1.2 px wide moving 2 px per frame over a textured background, rendered at 4 x 4 supersampling
(true alpha = exact coverage). Coarse mask = the ellipse only, binary. Each case damages the true alpha one way:

| Case | Coverage | Edge detail | Flicker | Levels |
|---|---:|---:|---:|---|
| true alpha | 1.0000 | 0.7241 | 0.0145 | green, green, green |
| coarse mask as alpha | 1.0000 | 1.0000 | 0.0183 | green, green, green |
| Gaussian blur sigma 1.5 px | 1.0000 | 0.1349 | 0.0071 | green, amber, green |
| blur sigma 3 px | 0.9991 | 0.0596 | 0.0050 | green, red, green |
| blur sigma 6 px | 0.9938 | 0.0203 | 0.0030 | green, red, green |
| edge grown / shrunk 1 px at random per frame | 0.9774 | 0.5913 | 0.3028 | green, green, red |
| noise +-0.07 in the edge band per frame | 0.9999 | 0.7043 | 0.0303 | green, green, amber |
| noise +-0.15 in the edge band per frame | 0.9993 | 0.7114 | 0.0520 | green, green, red |
| right-most 5 % of the subject lost | 0.9449 | 0.5930 | 0.0119 | amber, green, green |
| right-most 15 % lost | 0.8496 | 0.5132 | 0.0101 | red, green, green |

Thresholds chosen between these: coverage green >= 0.97, amber >= 0.90; edge detail green >= 0.30, amber >= 0.10;
flicker green <= 0.025, amber <= 0.040 (alpha units). Measured with `scratchpad/tmp_viewer/measure_qc.py`,
2026-10-15, CPU, 16 ms per frame for all three numbers at 320 x 180. The test pins every case to its level. These are
numbers on synthetic clips, not a claim about real footage; the viewer's tooltip says so.

## 5. The panel (`js/c2c_viewer/viewer.js`, mounted by `js/mask_matting.js`)

- A non-serialised DOM widget at the top of the node; it replaces the status strip. Its height follows the node's
  width and the plate's aspect (frame view 140-640 px). `node.computeSize()` passes the node's minimum width to a
  widget, so the panel sizes from the node's actual width.
- Rows: steps (Plate / Subject / Render, each clickable) + model slot; frame view; scrubber + frame box; ruler (the
  rendered range, the empty part of a SeC render, the pick frame); Find box; the one prominent next action (+ a
  secondary one and the preview length); status; Pick / Result + Comp / Alpha / Checker / Overlay; QC badges; All
  settings.
- Canvas colours are literal values read from the `--cu` tokens of the panel's root (class `c2c-ui`).
- `picks` is written on every selection change. A reload restores the markers; the found objects' masks need a new Find.
- All settings closed folds every regular control. `mask_visibility.js` asks `node.__c2cViewerFold`; the widgets stay
  real and saved in the same order. `picks` is a row only with `C2C_VIEWER=0`.
- Subgraphs: the execution id is resolved through the wrapper path (the first wrapper found), and the panel names the
  wrapper. A removal is not a deletion: the panel is released only when the node is still in no graph 2 s later.
- Live check, both renderers, subgraph, save + reload, kill switch: `docs/evidence/L7.114/viewer_live.md` (workspace).
