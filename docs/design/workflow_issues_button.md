# Workflow Issues button (L9.71, owner A20 Part 17.1 + 22.5)

## Problem

The validator in `js/c2c_workflow_wizard.js` showed its "⚠ N workflow issues" pill at a fixed spot (`left:74px;
bottom:14px`, z-index 9000). That spot is on top of LiteGraph's canvas info (T / I / N / V / FPS, drawn at the
canvas's bottom-left: x + 10, six lines of 13 px). The pill was also over the console (core's bottom panel,
`.bottom-panel`, Terminal / Logs) whenever that was open. It could not be moved, and it sat above dialogs.

## What the owner asked for (17.1, acceptance 22.5)

- A draggable floating button with a drag threshold, so a click still opens the list. Pointer capture, touch and
  keyboard. Issue count and a stable icon. Configurable size.
- Z-index above the canvas but below dialogs and menus.
- The position is the user's: saved in UI settings (not the workflow), restored on reload, clamped when the viewport
  changes, snapped to edges, with a reset.
- Never over the console. When the console opens, the button gets out of the way, but it never overrides where the
  user put it.
- Hide and restore.
- The list panel is draggable, resizable, keeps its position and size, is shared, is keyboard accessible and does
  not block anything when closed.

## Design

- `js/_c2c_float_place.js` is pure geometry with no DOM or app imports, tested with Node:
  - `clampRect`: keep the button inside the viewport, below the 60 px top toolbar gutter (`TOOLBAR_GUTTER`).
  - `avoidRects`: move the button out of keep-out rectangles. It tries above, then below, then to the right of the
    rectangle it overlaps, at most 4 passes.
  - `snapToEdges`: within 24 px of an edge, the button lands on the margin.
  - `toFraction` / `fromFraction`: the saved position is the button's centre as a fraction of the viewport.
- `js/c2c_workflow_issues.js` moves the validator out of the wizard file unchanged (the same scan, ids and fix
  actions) and adds the button and the panel:
  - **Keep-out rectangles**, measured on every layout pass:
    - the console (`.bottom-panel`, when shown);
    - the canvas info block (when `canvas.show_info`), at the bottom-left of the canvas element.
  - **Where the button goes:**
    - Shown position = `avoidRects(clampRect(saved or default))`.
    - Default = the canvas's bottom-left, just above the canvas info.
    - A displaced position is never saved. When the console closes, the button returns to the user's spot.
  - **Drag:**
    - Pointer capture. Below 5 px of movement, the press is a click.
    - On release, the same avoid and clamp rules apply, then snapping, then the position is saved once (not on every
      move).
    - `touch-action: none`, so touch drags work.
  - **Keyboard:** it is a `<button>`, so Enter / Space open the list. Alt + arrow keys move it 16 px (Shift: 64 px).
    Escape closes the list and returns focus to the button.
  - **Right-click menu:** Reset position, Hide the button.
  - **Commands** (command palette): show, hide, reset position, open the list.
  - **Settings, C2C › Workflow issues:**
    - Show the button (on). This is the kill switch: off = no button; the scan still runs, and "open the list"
      still works.
    - Button size (small / normal / large).
    - A hidden setting holds `{pos, panel}`.
  - **Panel:**
    - One shared element, `role="dialog"`, not modal.
    - A header bar drags it. CSS `resize: both` resizes it, and a ResizeObserver saves the size (debounced 400 ms).
    - The rows refresh while it is open.
    - It opens next to the button: above it when the button is in the lower half of the screen, below otherwise.
- **Z-index:** button 990, panel 991. That is above the graph canvas and Nodes 2.0's node layer, and below PrimeVue
  overlays (menus 1000, dialogs 1100), so dialogs and menus cover it.
- **Re-layout:** on window resize (which also catches browser zoom), when the console's size changes
  (ResizeObserver), and every 600 ms through the runtime governor while the button is visible. Each pass reads a few
  bounding rectangles and never walks the graph. No re-layout runs during a drag.
  - That timer is not "ambient": the governor pauses ambient timers in Lite mode and during runs, and the button must
    leave the console then too (measured: the testbed's software-GL browser runs in Lite).
- **Scanning:** the graph scan keeps its ambient 5 s safety poll and its `graphChanged` trigger. It also runs after
  every graph configure (`afterConfigureGraph`): loading fires no `graphChanged`, so in Lite mode a loaded broken
  workflow used to show nothing until the first edit.
- **Unchanged behaviour:** the button shows only while there are issues, as before; with none, it stays out of
  sight.

## Failure modes

| Failure | Behaviour |
|---|---|
| Settings API missing or the write fails | The position lives for the page only; reported once through `_c2c_report` |
| Saved position outside the viewport (another monitor, a smaller window) | Clamped into view; the saved value is replaced only when the user moves the button |
| Console covers the whole lower screen | The button goes above the console; if there is no room, it goes beside it |
| `.bottom-panel` renamed in a future frontend | The canvas info is still avoided; the console check finds nothing, so the button stays where the user put it (it is draggable) |

## Compatibility

- Element ids are unchanged (`c2c-wizard-validator-pill`, `c2c-wizard-validator-list`).
- Workflows are not touched: no graph or `extra` data is written.
