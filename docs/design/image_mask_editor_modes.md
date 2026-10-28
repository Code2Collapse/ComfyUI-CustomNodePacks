# One Image Mask Editor for every kind of mask drawing (L7.65 P01 = L7.54)

Owner-approved card P01 (2026-10-09, text in `docs/evidence/L7.65/audit_page_source_2026-10-09.html` of the
workspace): *"Image Mask Editor gets modes: Paint, Points + Box, Spline, Roto, Video (keyframes + tween). The other
five are removed with automatic migration."* Card P07 (approved): Mask Track gets a **Spline track** mode. Owner
2026-10-15 (A14): *"The editors (P01) keep their drawing UIs."* D0.14: merged nodes are removed and saved workflows
migrate on load. This document is written before any code (A7).

## 1. What is merged

| Old node | Lines (py / js) | Becomes | Data it saves |
|---|---|---|---|
| `ImageMaskEditorC2C` (Image Mask Editor) | 275 / 4,700 | stays, mode **paint** (default) | `editor_id` -> server mask store |
| `MECAdvancedPaintCanvas` (Paint Canvas) | 753 / 1,008 | mode **paint** (colour layer: see Q4) | `canvas_data` = base64 RGBA PNG in the workflow |
| `MaskEditMEC` mode `points_bbox` | 457 / (points_bbox_editor.js) | mode **points + box** | `editor_data` JSON |
| `SplineMaskMEC` mode `edit` | 384 / 1,518 | mode **spline** | `spline_data` JSON |
| `SplineMaskMEC` mode `track` | (spline_mask_tracker.js 986) | **Mask Track**, mode `spline track` (card P07) | `spline_data` + tracker params |
| `VectorRotoMEC` (Vector Roto) | 284 / 193 + roto UI | mode **roto** | `roto_json` |
| `VideoMaskEditorMEC` (Video Mask Editor) | 751 / 1,617 | mode **video** | `session_id` -> server disk store |

Not covered by the card, decided by the owner (section 7): MaskEditMEC's `transform`, `draw_shape`,
`draw_advanced`, `bbox_smooth`; SplineMaskMEC's `flow_path`; the Paint Canvas colour strokes.

Eight existing replacement rows already point at the two multi-mode nodes (MaskTransformXY, MaskDrawFrame,
DrawShapeMEC, PointsMaskEditor, BBoxSmooth -> MaskEditMEC; SplineMaskEditorMEC, SplineMaskTrackerMEC,
SplinePathFlowMaskMEC -> SplineMaskMEC). The migrator runs **one pass** (`js/_c2c_migrate_core.js`: a row whose new
node is not registered is left untouched), and core's server-side NodeReplace does not chain either, so those rows
are **recomposed** to their final node (old -> final in one row). `test_every_new_node_importable` fails while any
row targets a removed id.

## 2. Constraints

1. Saved workflows of all seven node ids (and the eight older ids above) load as the new node with every value and
   link (D0.14), in classic and Nodes 2.0, at the root and inside subgraphs; API prompts with old ids still run
   (rows registered with core's NodeReplaceManager).
2. Existing Image Mask Editor saves are untouched: no input moves, new widgets and outputs are **appended**, the
   new `mode` defaults to `paint` (frontend 1.53 restores widget values by position - F2.x / memory
   frontend-153-widget-restore-order - so appending is the only safe change).
3. Same engines, same results: each mode calls the old node's own code. Migrated configurations are bit-exact
   with the old node (test per mode, P08 style); the shared result stage (grow, feather, threshold, invert,
   combine) is identity at its defaults, so it changes nothing for a migrated node.
4. Each mode keeps its drawing UI (A14). Only the active mode's UI is mounted: a mode switch disposes the old UI
   (DOM widgets, listeners, rAF loops, WebGL contexts) and mounts the new one.
5. Output types stay what downstream nodes expect: the points mode still gives SAM / SeC coordinate STRINGs and
   BBOXes, the spline mode still gives `SPLINE_DATA`, the paint canvas still gives its IMAGE.
6. RAM (owner): no polling of the graph, no editor mounted for an inactive mode, no second copy of a frame batch.

## 3. The node after the merge

`ImageMaskEditorC2C`, title **Image Mask Editor (C2C)**.

**Inputs** (existing first, unchanged order): `image` (moves from required to optional - it is the only socket
before `input_mask`, so the socket order and every saved link slot stay the same; roto, spline and paint canvas
saves have no image), `editor_id`, `frame_mode`, `grow`, `feather`, `threshold`, `invert`, `input_mask`,
`combine`; then appended: `mode` (`paint`, `points + box`, `spline`, `roto`, `video`, plus what Q1-Q3 add), then
the per-mode inputs with a prefix, in the old node's order (the P08 `_mode_inputs` pattern):
`points_*` (MaskEditMEC points_bbox: width, height, editor_data, default_radius, softness, normalize, reference
image, existing mask), `shape_*` (MaskEditMEC draw_shape / draw_advanced), `spline_*` (SplineMaskMEC edit and
flow_path, with `spline_fill`), `roto_*` (VectorRoto), `video_*` (VME session, tween, feather, threshold,
outside_keys), `canvas_*` (Paint Canvas, incl. the colour layer). The node shows only the active mode's controls
through `js/c2c_mode_widgets.js` (D7.65b: widgets only, never sockets or links).

**Outputs** (existing first): `mask`, `preview`, `info`; then appended, each documented with the mode that fills
it (other modes give an empty value of the right type): `positive_coords`, `negative_coords`, `bboxes`,
`neg_bboxes`, `points_json`, `bbox_json`, `primary_bbox` (points + box), `coords_json`, `spline_data`,
`bbox` (spline), `painted_image` (paint canvas, Q4). Owner rule from P24: outputs are kept, not folded.

**Paint mode, two layers** (Q4 answer, A14 "keep their drawing UIs"): `canvas_layer` = `mask` (default: the
existing modal mask editor, strokes in the mask store) or `colour` (the Paint Canvas UI: colour strokes over the
image, `canvas_data`). The mask output is the mask layer; with colour strokes present it is the union of both
layers' masks (only possible on a node that has both), and `painted_image` is the colour layer composited over
`image` (or a `canvas_width` x `canvas_height` canvas without one, as the Paint Canvas did). A migrated Paint
Canvas therefore gives exactly its old two outputs.

**Execution**: `execute` dispatches on `mode` to the old engine (imported, not copied), then applies the shared
result stage and the `input_mask` combine. `IS_CHANGED` hashes only the active mode's inputs plus the store
hash (editor_id / session_id), so editing an inactive mode's hidden value does not re-run the graph.

## 4. Front end

**As built (2026-10-27):** the pack already had a mode gate for multi-mode editors (`js/_mode_gate.js`: an editor is
installed the first time its mode is picked, then shown / hidden), proven on Mask Edit and Spline Mask - so it is
reused instead of a new mount / dispose layer (standing order 12: repo conventions). It gained a test function / list
for `activeWhen`, extra watched widgets, `onActivate`, widget lists and `keepSize`. The editors find their widgets
through `js/_ime_names.js` (the prefixed names by mode). Budgets as measured: switching through all modes twice
leaves the same number of DOM widgets (no orphan); an inactive editor is hidden, not re-created. The paint UI is
created with the node (not lazily): a saved node keeps its value after the Python widgets, which
`js/widget_migration.js` uses to repair saves made before the modes existed. The plan below was the alternative.

One controller for the node (`js/image_mask_editor.js` today) owns the mode. Each old editor is refactored from
"hook node type X in beforeRegisterNodeDef" into a module with `mount(node, api) -> { dispose }`:
paint = the existing modal editor (unchanged), points = `points_bbox_editor.js`, spline =
`spline_mask_editor.js`, roto = the roto UI on `_roto_format.js`, video = `video_mask_editor.js`, paint canvas =
`mec_advanced_paint.js`. The old files keep working for nothing else (no other node uses them - checked at build
time with a grep gate in the test suite).

Mode switch: `dispose()` the current module (removes its DOM widgets with the frontend's own `removeWidget`,
cancels rAF / timers, releases WebGL), then `mount()` the new one, then `setSize(computeSize())` once. The
switch must survive the known traps: Nodes 2.0 keeps a stale DOM element after Convert to Subgraph (rebind with
`_vue_dom_rebind.js`), 1.53 restores values by position before `onConfigure` (restore by name), a loaded workflow
mounts the saved mode once, after configure.

## 5. Migration

**As built (2026-10-28):** option (a). Five new rows (two with `c2c_by_value` cases: Mask Edit's base row is its
`transform` -> Mask Tools, Spline Mask's is `edit`); the eight older rows recomposed by composing old -> mid with the
mid node's case for the mode the old row set. Measured on the way: the frontend gives an INT named `seed` a
control_after_generate by itself, so the renamed `spline_flow_seed` asks for one (a migrated fixed / randomize choice
is carried); a saved node whose value is out of range is dropped by validation while the rest of the prompt still
reports success - the live check therefore counts the previews that produced an image, not only the status.

New rows (old -> `ImageMaskEditorC2C`): `MECAdvancedPaintCanvas`, `VectorRotoMEC`, `VideoMaskEditorMEC`,
`MaskEditMEC` (mode-dependent target, see below), `SplineMaskMEC` (edit -> editor, track -> Mask Track,
flow_path -> Q3). Recomposed rows: the eight older ids (section 1).

A row maps one old id to one new id, but MaskEditMEC and SplineMaskMEC have modes that go to different nodes.
Two options, chosen at build time by a test against core's NodeReplace format:
(a) the C2C migrator reads a `c2c_by_value` block (`{"widget": "mode", "cases": {"track": <row>, ...}}`) and picks
the row by the saved mode value - UI workflows; the server-side row registered with core stays the most common
case and API prompts of the other modes get a clear error naming the node to use; or (b) everything goes to the
editor and the editor runs the non-drawing modes too (Q1 option B). Option (a) is the default if Q1 = A.

Every row: `old_widget_ids` from the fixture of the old signature, `set_value` for `mode`, `output_mapping` onto
the appended outputs, `dropped` lists anything with no target, and the note says what happened. Values outside a
new combo are normalised in `VALIDATE_INPUTS` (as L2.14 did for MaskOps).

## 6. Failure modes, kill switch, budgets

| Failure | Handling |
|---|---|
| A mode's engine fails to import (missing optional dependency) | the mode stays in the list, running it raises a plain-English error naming the package; other modes work |
| Saved store id missing (mask store / VME session deleted) | empty mask + `info` says which id is missing; never a crash |
| Paint Canvas `canvas_data` is huge (base64 PNG in the workflow) | migrated as-is; first save moves it to the mask store (keeps the workflow small) |
| Mode switched with unsaved strokes | the editor asks before disposing (same dialog as closing the modal) |
| Old JS file still hooked by another pack's node | grep gate in the tests; the module also keeps its type hook behind the controller |
| Workflow with an old id on a machine without this pack version | unchanged: core shows the missing node |

Kill switch: `C2C_LEGACY_MASK_EDITORS=1` keeps the six old node ids registered (P05 precedent
`C2C_LEGACY_MATTE_NODES`), so a broken mode never blocks a user.

Budgets (measured, recorded in the evidence): a mode switch mounts in < 300 ms and leaves no orphaned DOM node,
listener or WebGL context (counted before / after 20 switches); a node in an inactive mode costs no rAF
callbacks; stroke latency of the paint mode unchanged (L7.45 numbers as the baseline).

## 7. Owner decisions (not on the card) - answered 2026-10-27: A on all four

Answers (question tool, 2026-10-27): Q1 A, Q2 A, Q3 A, Q4 A. So the editor's mode list is `paint`, `points + box`,
`shape`, `spline` (fill: mask / flow pattern), `roto`, `video`; Mask Tools gains `transform`; Mask Track gains
`spline track` and `bbox smooth`; the paint mode gains a colour layer and `painted_image`. Section 5 uses option
(a) (`c2c_by_value` rows for MaskEditMEC and SplineMaskMEC).

- **Q1** MaskEditMEC `transform` (expand / blur / offset per axis, feather, threshold, invert on a MASK input) and
  `bbox_smooth` (temporal smoothing of a bbox list) are operations, not drawing. A: `transform` -> Mask Tools mode
  `transform`, `bbox_smooth` -> Mask Track mode `bbox smooth` (recommended: they sit with the operations they
  belong to); B: both become Image Mask Editor modes.
- **Q2** MaskEditMEC `draw_shape` / `draw_advanced` (12 parametric shapes from numbers or JSON, batch-aware). A:
  Image Mask Editor mode `shape` (one mode, the JSON form as an advanced input) (recommended); B: Mask Tools mode.
- **Q3** SplineMaskMEC `flow_path` (procedural ribbon / wave / dust / smoke along a drawn spline, animated over
  frames). A: an option of the spline mode (`spline_fill`: mask / flow pattern), it needs the same spline canvas
  (recommended); B: removed (reported on load).
- **Q4** Paint Canvas colour strokes (`painted_image`, brush colour, composited over a reference image). A: the
  paint mode gets a colour layer and the `painted_image` output, so a saved painting still produces its picture
  (recommended); B: mask only, the colour strokes are dropped and reported.

## 8. Slices (each to main only through the A3 gate)

1. Python: mode dispatch, appended inputs/outputs, result stage identity, bit-exact tests per mode against the
   old nodes, recomposed rows, kill switch. Spline track into Mask Track (P07 remainder).
2. Front end: controller + `mount/dispose` per editor, mode switch budgets, both renderers, subgraph variant.
3. Migration live: a workflow built with all seven old ids (and the eight older ones) on the pre-merge main,
   loaded after the merge in classic + Nodes 2.0, root + subgraph, values / links / run checked (the
   `p08_merge_live.py` method), screenshots at 100 % zoom read.
4. Docs: README section, NODE_REFERENCE regenerated, MIGRATION.md regenerated, CHANGELOG, node count.
