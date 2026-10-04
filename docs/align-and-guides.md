# C2C Align: smart guides and align shortcuts

User request (2026-09-28): while dragging, show proper alignment; select nodes,
press a shortcut, and they auto-align vertically or horizontally, groups
included.

## What exists already

| Source | Has | Missing |
|---|---|---|
| ComfyUI core (frontend 1.3x) | Right-click "Align Selected To" top/bottom/left/right, "Distribute Nodes", snap-to-grid (Shift while dragging, or always) | No keyboard commands, no centre alignment, **nodes only (groups ignored)**, no guides while dragging |
| [LVO-Node-Aligner](https://github.com/LVOcode/ComfyUI-LVO-Node-Aligner) | Toolbar align/row/column with exact gap, snap while moving/resizing (8 screen px, Shift bypasses) | "Group-frame snapping is not supported"; GPL-3 |
| [ComfyUI-NodeAlign](https://github.com/1038lab/ComfyUI-NodeAlign) | Toolbar, Shift+W/A/S/D align, distribute, equal size | No groups, no drag guides; GPL-3 |
| [ComfyUI-Align](https://github.com/Moooonet/ComfyUI-Align) | Panel on backtick, hold mode, distribute+align combos, stretch | All rights reserved: not usable |
| [ComfyUI-LJNodes](https://github.com/coolzilj/ComfyUI-LJNodes) | Alt+W/S/A/D align, Alt+H/V distribute | No groups, no guides |
| [comfyui-node-organizer](https://github.com/PBandDev/comfyui-node-organizer) | Whole-graph DAG auto layout, group-aware, Shift+O | Layout engine, not alignment; AGPL-3 |
| Blender Node Wrangler | **One shortcut (Shift+=) aligns the selection, picking the axis automatically, with even spacing** | - |
| Figma smart guides | While dragging: snap edges/centres to peers and to equal spacing; per axis the closest candidate wins | - |

CustomNodePacks is Apache-2.0, so none of the GPL/AGPL/proprietary code above is
copied. This is our own implementation of well-known behaviour.

## Design

**Units.** Every align/snap works on *movable units*: selected nodes and
selected groups. A selected node inside a selected group is not a unit (it
moves with its group). Bounds are `boundingRect` ([x, y, w, h], title bar
included; collapsed nodes use their collapsed width). Groups move with
`group.move(dx, dy)` so their children come along; nodes via `node.pos`, which
writes through to the layout store, so both renderers follow.

**Smart align (Alt+A)** - the "auto" one. Spread of unit centres decides the
axis: wider than tall = a row, else a column. Row: tops aligned to the first
(leftmost) unit, units kept in left-to-right order, equal gaps. Column: left
edges aligned to the topmost unit, top-to-bottom order, equal gaps. The gap is
the average of the current gaps, but never below the "minimum gap" setting (so
overlapping nodes separate). Running it twice changes nothing.

**Explicit commands.** Align left / right / top / bottom (to the selection's
outer edge), centre on a vertical line / on a horizontal line, distribute
horizontally / vertically (outermost units stay, equal gaps; if they do not
fit, stack with the minimum gap), equal width / equal height (largest wins;
groups resize too). All are ComfyUI commands, so every key is rebindable in
Settings > Keybinding. Also in the right-click menu and the selection toolbox.

| Command | Default key |
|---|---|
| Smart align (auto row / column) | Alt+A |
| Align left / right / top / bottom | Alt+Shift+Left / Right / Up / Down |
| Centre on vertical line / horizontal line | Alt+Shift+X / Alt+Shift+Y |
| Distribute horizontally / vertically | Alt+Shift+H / Alt+Shift+V |
| Toggle smart guides | Alt+Shift+G |
| Equal width / height | none (menu + keybinding panel) |

Checked against core defaults (Alt+C, Alt+M, Alt+=, Alt+-, Ctrl+B/M/G...),
Pixaroma (Alt+H, Alt+W) and ours (Alt+B, Ctrl+Shift+F/K/D, Ctrl+Alt+M).

**Smart guides while dragging.** Candidates: every node and group not being
dragged whose bounds are near the viewport. Per axis the closest of these wins,
within 8 screen px (setting): edge-to-edge and centre-to-centre alignment with
a peer, or equal spacing (the dragged unit sits so its gaps to its two
neighbours in the same row/column are equal, or match the gap between two
existing neighbours). Guide lines are drawn in the canvas foreground. Snapping
is a correction applied after LiteGraph moves the selection (`processMouseMove`
wrapper): the offset currently applied is remembered, so the raw pointer
position is always recovered and a snap releases smoothly. Shift stays core's
snap-to-grid (guides step aside). Guides are switched off/on with the setting or
Alt+Shift+G - not with Ctrl, because a Ctrl-drag does not move nodes in ComfyUI.

**Undo.** Each command is one undo step (`graph.beforeChange()` /
`afterChange()`); a drag stays one step, as today.

**Nodes 2.0.** Commands work in both renderers (they only set positions).
Drag guides use a window-level `pointermove` listener (captured pointer events
still bubble). Classic drags: `app.canvas.state.draggingItems`. Vue node drags:
`app.extensionManager.layoutStore.isDraggingVueNodes` (Vue ref; read `.value`),
with a `.lg-node` header + 3 px movement fallback.
