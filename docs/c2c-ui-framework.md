# C2C UI framework

One front-end system for every Code2Collapse ComfyUI node: the in-node panels,
the full-screen editors, and their behaviour in both renderers (classic
LiteGraph canvas and Nodes 2.0). The quality bar is the user's reference set:
Pixaroma (Image Compare, Image Composer, Paint, 3D Builder), WhatDreamsCost
(LTX Director) and NKD Tools. The colours are ours: the night palette.

Nothing here is copied UI. Pixaroma's Nodes 2.0 *compatibility logic* (MIT,
`third_party/ComfyUI-Pixaroma/js/shared/nodes2.mjs`, `renderer_switch.mjs`,
`resize_floor.mjs`) is adapted with attribution; every visual component is our
own design.

## What the references do that we don't

| Reference | What it does | Ours today |
|---|---|---|
| Pixaroma Compare | One segmented pill bar (active pill filled with the accent), a secondary action row, a big stage with a quiet hint. | Mixed per widget; many rows of plain LiteGraph widgets. |
| Pixaroma Composer / Paint | Node = compact preview + one "Open …" button + a `480 × 270` readout. Editing happens in a full-screen editor: left tools, centre canvas with floating zoom bar, right layers/properties, Save/Close bottom-right, hint bar at the bottom. | Editors crammed into the node body (points, spline, paint, quad, FC3D). No full-screen mode. |
| Pixaroma empty state | Brand mark, title, accent subtitle, one-line instruction. | Checkerboard + hint on some editors; plain text elsewhere. |
| Pixaroma Nodes 2.0 | Live `canvasOnly` getter, renderer-flip listener, drag-time resize floor, canvases rendered at `dpr × zoom`. | One adapter (`_vue_canvas.js`), decided at creation, repaints every frame. 47 DOM-widget sites with no 2.0 handling. |
| NKD Tools | Real Vue components mounted inside nodes. | n/a |

## Distribution

- Canonical source: `ComfyUI-CustomNodePacks/js/c2c_ui/`. Byte-identical copies
  at each pack's web root: `NukeMax web/c2c_ui/`, `WanNodeExperiments web/c2c_ui/`,
  `MiniMaxSuite web/c2c_ui/`, `WanAnimatePreprocessV2 js/c2c_ui/`,
  `GLM_Image web/c2c_ui/`, `WanAnimalPreprocessor web/c2c_ui/`.
  `tests/test_c2c_ui.py` holds them in step (same rule as `_c2c_brand.js`).
- **No imports from ComfyUI** inside `c2c_ui/`: it reads
  `window.comfyAPI?.app?.app` and `window.LiteGraph` at call time. A wrong `../`
  depth 404s and takes a pack's whole front-end down; zero imports removes the risk.
- ComfyUI auto-loads every `.js` under a web root as an extension, so every
  module is side-effect free until called. The stylesheet is injected once per
  page under a versioned id (`c2c-ui-v1`), whichever pack gets there first.

## Tokens

Scoped to `.c2c-ui`. Each reads the live palette first and falls back to the
night literal, so NukeMax and MiniMax (which have no `_c2c_theme.js`) still get
the night look, and a palette switch still reaches every pack.

| Token | Source | Night |
|---|---|---|
| `--cu-ground` | `--c2c-bg3` | `#07081a` editor workspace |
| `--cu-sunken` | `--c2c-bg2` | `#0c0d23` stage, inputs |
| `--cu-panel` | `--c2c-panelBg` | `#1e1f47` sidebars, node panel |
| `--cu-raised` | `--c2c-surface0` | `#1d1e45` buttons |
| `--cu-edge` | `--c2c-surface1` | `#2a2a57` borders |
| `--cu-edge-strong` | `--c2c-surface2` | `#3b3a68` hover borders |
| `--cu-ink` | `--c2c-fg` | `#e8e6f7` |
| `--cu-ink-soft` | `--c2c-sub` | `#bab7db` labels |
| `--cu-ink-dim` | `--c2c-dim` | `#6f6d9b` hints |
| `--cu-accent` | `--c2c-mauve` | `#b494ff` the one saturated hue |
| `--cu-accent-hover` | `--c2c-accentLink` | `#c0aaff` |
| `--cu-on-accent` | `--c2c-bg3` | `#07081a` text on accent |
| `--cu-ok / --cu-warn / --cu-danger` | `--c2c-ok / yellow / red` | semantic only, never decoration |

Type: `'Segoe UI', system-ui, sans-serif` (ComfyUI's own), 12 px body in nodes,
13 px in editors; section headers 10.5 px uppercase, `letter-spacing: .08em`,
`--cu-ink-dim`; numeric readouts `ui-monospace` with `tabular-nums`.
Radius 6 px (controls), 8 px (panels). Borders 1 px `--cu-edge`. One accent:
active/primary only.

## Components (`c2c_ui/components.js`)

All return plain elements. Every control has a visible `:focus-visible` ring
(`--cu-accent`), a hover state, and a disabled state.

- `pillBar(items, {value, onChange, columns})` - segmented choice; active pill
  filled `--cu-accent` with `--cu-on-accent` text. Wraps into equal columns.
- `actionRow(buttons)` - equal-width secondary buttons (outline).
- `button(label, {primary, danger, icon, onClick})`.
- `section(title, ...children)` - uppercase header + body.
- `sliderRow(label, {min, max, step, value, onChange, format})` - label, track,
  numeric input kept in sync.
- `selectRow`, `colorRow`, `toggleRow`.
- `toolGrid(tools, {columns, value, onChange})` - icon + label tiles (Paint's tool grid).
- `zoomBar({onOut, onFit, onIn, getLabel})` - floating `− Fit 78% +`.
- `emptyState({title, hint, glyph})` - the wolf mark (inline SVG, ours), the
  node's name, `C2C` in accent, one instruction line. Replaces
  `_editor_empty_state.js` visuals over time.
- `stage({aspect, empty})` - preview area: `setImage(src, w, h)`,
  `setCanvas(canvas)`, `setEmpty()`, footer readout `480 × 270`.
- `statusLine()` - one-line state with semantic colour.

## Mounting a panel in a node (`c2c_ui/node_panel.js`)

`mountPanel(node, name, root, {minHeight, fill})` is the ONLY way a C2C node adds
a DOM widget. It encodes every rule we paid for:

1. **Margin math.** ComfyUI sizes a DOM element to `computedHeight − 2 × margin`
   (margin 10). The panel passes `margin` explicitly and reserves it in
   `getMinHeight`/`computeSize`, so nothing hangs past the node edge.
2. **Fit on create** (already in `_c2c_brand.js`): grow to computeSize, never shrink.
3. **Nodes 2.0:** `canvasOnly` is a live getter (false in 2.0, true in legacy);
   the drag-time resize floor is installed; renderer flips re-mount without F5.
4. **Reflow:** a `ResizeObserver` on the root drives repaint; canvases size
   their backing store at `dpr × zoom` and re-render on zoom change only
   (no per-frame repaint loops - see the FC3D 130/s freeze).
5. **Cleanup:** every observer, rAF and listener is released in `onRemoved`.

## Full-screen editor (`c2c_ui/editor.js`)

`openEditor({title, left, right, centre, onSave, onClose, help, hints})`:

- Header: wolf mark, `C2C` accent, tool name; Undo / Redo / Help at the right.
- Left sidebar 272 px (sections), right sidebar 300 px (layers/properties),
  centre workspace on `--cu-ground` with the floating zoom bar.
- Bottom-right: Save (primary) and Close. Bottom strip: keyboard hints.
- `Esc` closes (asks inside the page if there are unsaved changes), focus is
  trapped while open and restored on close, `role="dialog"`, `aria-modal`.
- Modal tier z-index (`--c2c-z-modal`), removed from the DOM on close.

Editor-class nodes keep a compact in-node view (preview + `Open editor` +
readout) and move the real editing into this shell.

## Rollout

Each stage is shot in BOTH renderers and passes the checklist before the next.

1. **Framework** - `c2c_ui/` + copies + `tests/test_c2c_ui.py` + one exemplar
   of each kind: Video Comparer (viewer), Advanced Paint Canvas (editor with
   full-screen mode), NukeMax status strip (kit).
2. **Viewers and scopes** - MiniMax w1-w10, NukeMax HDR scope / EXR preview /
   Viewer, Video Frame Player, Sigma/Drift plots.
3. **Editors** - points editor (SAM family), spline (Vector Roto, Roto Spline),
   Mask Placement, Light Rig, Face Controller 3D, then WanDirector's shell.
4. **Kits and strips** - Layer FX, Mask kit, MaskOps header, NukeMax/MiniMax strips.
5. **Everything else** - plain-widget nodes get the node chrome only (brand,
   fit, badges); no invented UI where a node needs none.

## Checklist (per node, both renderers)

Labels readable and unclipped; no dead space below the last element; custom
canvases render; content reflows on resize; overlays follow pan/zoom; errors in
plain English; no console errors; focus visible; nothing covers core UI.
