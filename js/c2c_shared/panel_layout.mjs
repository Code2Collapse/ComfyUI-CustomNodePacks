// PORTED FROM: ComfyUI-AusBoss js/shared/panel_layout.mjs (https://github.com/ausboss/ComfyUI-AusBoss), MIT License,
// Copyright (c) 2026 ausboss. Licence text: js/c2c_shared/LICENSE-AUSBOSS.txt. Code unchanged at port (2026-11-28,
// agy review Phase 3 item 12: one shared UI foundation for the C2C packs).
// How a DOM panel claims its share of a node's height. No DOM and no
// ComfyUI imports in here, so it stays testable under node:test.
//
// The frontend arranges a node's widgets in one pass
// (LGraphNode._arrangeWidgets):
//
//   if (w.computeSize)            -> fixed height, kept OUT of the split
//   else if (w.computeLayoutSize) -> joins distributeSpace(freeSpace, ...)
//   else                          -> one standard widget row
//
// It is an else-if, so a widget declaring BOTH is pinned by computeSize and
// its computeLayoutSize is never called. Every stage, player and filmstrip in
// this pack derived that fixed height from the node's WIDTH, which is why
// dragging a node taller only added dead space underneath: the panel had
// already been given a height and excluded from the leftover-space split.
//
// The frontend mounts a DOM widget's element inside a frame: DomWidgets.vue
// insets it by `options.margin` per side (default 10), so the element gets
// 20 fewer CSS pixels of height than the layout hands the widget. Any floor
// meant to guarantee room for fixed-height content must add this allowance,
// or the panel's bottom edge renders clipped flat - which is how the LoRA
// stack's rounded bottom border once went missing.
export const WIDGET_FRAME = 20;

// The frontend pads a DOM widget's minWidth before it becomes the node's
// floor: LGraphNode.computeSize adds the room a number widget's value box
// takes (BaseWidget.minValueWidth plus both arrows and margins - 104px on
// frontend 1.53), and a corner drag never goes below computeSize. So a
// panel declaring 320 cannot be dragged under 424, and a node that opened
// at 340 jumps to 444 the moment its corner is touched. This measures that
// padding from the frontend itself instead of copying its constants: a
// probe minWidth goes in, and whatever computeSize adds on top comes out.
// null when it cannot be measured right now (the widget is hidden, or the
// node cannot size itself), so the caller tries again later.
const PROBE_WIDTH = 100000;
let probing = false;

export function measureLayoutWidthPadding(node, widget) {
  if (probing || typeof node?.computeSize !== "function" || !widget) return null;
  const own = widget.computeLayoutSize;
  probing = true;
  widget.computeLayoutSize = () => ({ minWidth: PROBE_WIDTH, minHeight: 0 });
  let width = NaN;
  try {
    width = Number(node.computeSize()?.[0]);
  } catch {
    // A node that cannot size itself right now is measured next time.
  } finally {
    widget.computeLayoutSize = own;
    probing = false;
  }
  return Number.isFinite(width) && width >= PROBE_WIDTH ? width - PROBE_WIDTH : null;
}

// The padding is the frontend's, the same for every node: measured once.
let layoutWidthPadding = null;

const resolveWidth = (value) => {
  const resolved = Number(typeof value === "function" ? value() : value);
  return Number.isFinite(resolved) ? Math.max(0, resolved) : 0;
};

// A node can carry several panels (a card and a video viewer), each with its
// own minimum, and its floor is the widest of them. Every panel registers
// its number here, keyed by the widget, and the same tally feeds both
// renderers, so the two never fight over one node.
const FLOORS = Symbol("ausboss.widthFloors");
const WATCHED = Symbol("ausboss.floorWatched");

// A Nodes 2.0 node element only exists around a panel once the panel is
// laid out inside it, and the panel's first size arrives at that moment. So
// what a panel lends that node is written from one ResizeObserver per
// panel, each floor under its own name.
function whenLaidOut(panel, name, task) {
  if (!panel || typeof ResizeObserver !== "function") return;
  if (!panel[WATCHED]) {
    const tasks = new Map();
    panel[WATCHED] = tasks;
    new ResizeObserver(() => { for (const run of tasks.values()) run(); }).observe(panel);
  }
  panel[WATCHED].set(name, task);
}

function nodeWidthFloor(node) {
  let floor = 0;
  for (const value of node?.[FLOORS]?.values() ?? []) floor = Math.max(floor, resolveWidth(value));
  return floor;
}

// The classic renderer's corner drag never goes below node.computeSize(),
// and computeSize skips a widget's computeLayoutSize whenever the widget also
// pins its own computeSize (the fixed-height cards do): their declared
// minimum was never a floor at all, and such a node dragged down to the
// frontend's 210px default with the card's buttons cut off. Wrapping the
// node's own computeSize makes the tally the floor for every panel alike.
function holdClassicFloor(node) {
  const inherited = node.computeSize;
  if (typeof inherited !== "function") return;
  node.computeSize = function (out) {
    const size = inherited.call(this, out);
    const floor = nodeWidthFloor(this);
    if (size && size[0] < floor) size[0] = floor;
    return size;
  };
}

// Make this panel's minimum part of its node's floor, in both renderers.
// minWidth is a number or a function, for a floor that follows state. The
// node's width is then never below it: not by a corner drag, and, for a
// pinned card, not by the frontend's own default either. exactMinWidth on
// fillNodeHeight calls this too; a panel that pins its height (Seed, Save
// Image, the fixed widget cards) calls it directly.
export function holdNodeMinWidth(widget, minWidth) {
  const node = widget?.node;
  if (!node) return widget;
  if (!node[FLOORS]) {
    node[FLOORS] = new Map();
    holdClassicFloor(node);
  }
  node[FLOORS].set(widget, minWidth);
  const panel = widget.element;
  whenLaidOut(panel, "width", () => holdVueNodeMinWidth(panel, nodeWidthFloor(node)));
  return widget;
}

// A floor that follows state can rise while the node is narrower than it:
// the picture arrows of Image Compare A/B and of the preview panel show up
// only after a run brings several pictures. Hold the new floor in Nodes 2.0
// (a lower one too, once the arrows go) and widen a node that is now too
// narrow, in either renderer. A node already wide enough is left alone.
// A floor measured from the panel is only exact once the panel has the
// node's width, which Nodes 2.0 gives it a frame later, so a node that grew
// is measured once more then.
export function growToNodeFloor(widget, passes = 2) {
  const node = widget?.node;
  if (!node?.[FLOORS]) return false;
  const floor = nodeWidthFloor(node);
  holdVueNodeMinWidth(widget.element, floor);
  const width = Number(node.size?.[0]);
  if (!Number.isFinite(width) || width >= floor) return false;
  node.setSize?.([floor, node.size[1]]);
  node.setDirtyCanvas?.(true, true);
  if (passes > 1 && typeof requestAnimationFrame === "function") {
    requestAnimationFrame(() => growToNodeFloor(widget, passes - 1));
  }
  return true;
}

// distributeSpace reads a missing maxSize as Infinity, so declaring a floor
// with no ceiling means "take whatever is left" - which is exactly "fill the
// node". minWidth/minHeight accept a number or a function, for panels whose
// floor depends on state (the frame chooser is shorter until it has frames).
// exactMinWidth: true makes minWidth the node's real floor (see above), in
// Nodes 2.0 as well; without it the frontend's padding comes on top, as it
// always has. Every panel in the pack passes it (tests/panel_guards.test.mjs).
export function fillNodeHeight(widget, { minWidth = 0, minHeight = 0, minNodeSize, exactMinWidth = false } = {}) {
  if (!widget) return widget;
  const floor = resolveWidth;
  const padding = (node) => {
    if (!exactMinWidth) return 0;
    if (layoutWidthPadding === null) {
      layoutWidthPadding = measureLayoutWidthPadding(node ?? widget.node, widget);
    }
    return layoutWidthPadding ?? 0;
  };
  // Nodes 2.0 sizes a node by its content, not by this floor, so the
  // panel's element carries it there as CSS (holdVuePanelMinHeight below).
  // It is written again every time the floor is read, so a floor that
  // follows state is followed at once, and when the panel is first laid out
  // inside a node.
  const panel = widget.element;
  const holdHeight = (height) => holdVuePanelMinHeight(panel, height - WIDGET_FRAME);
  // Deleted, not overwritten: any own computeSize would win the else-if above.
  delete widget.computeSize;
  widget.computeLayoutSize = (node) => {
    const height = floor(minHeight);
    holdHeight(height);
    return {
      minWidth: Math.max(0, floor(minWidth) - padding(node)),
      minHeight: height,
    };
  };
  widget.options ??= {};
  if (minNodeSize) widget.options.minNodeSize = minNodeSize;
  if (exactMinWidth) holdNodeMinWidth(widget, minWidth);
  whenLaidOut(panel, "height", () => holdHeight(floor(minHeight)));
  return widget;
}

// The other way round: a panel that keeps its own height, such as a card
// whose height is the sum of its rows. Its own computeSize keeps it out of
// the classic renderer's split (the else-if at the top of this file), but
// Nodes 2.0 reads a different sign. Its widget grid gives a row a share of
// the node's spare height whenever the row's widget has a computeLayoutSize,
// and every DOM widget inherits one from the frontend. So a pinned card was
// stretched there all the same: above a picture, half the spare height went
// to the card as empty space under its last row and the picture was squeezed
// into the rest. Shadowing the inherited method makes the row fixed in Nodes
// 2.0. The classic renderer never calls it on a widget with a computeSize.
// The frontend reads the sign when the node joins a graph, so call this as
// the panel is built.
export function pinVuePanelHeight(widget) {
  if (widget) widget.computeLayoutSize = undefined;
  return widget;
}

// Nodes 2.0 (the Vue renderer) has no layout API for a node's minimum
// width: its corner drag stops at the node element's inline min-width, and
// at 225px when there is none. Lend it the panel's floor whenever the panel
// is laid out inside a Vue node. The classic renderer mounts panels outside
// any [data-node-id] element, so there this does nothing.
export function holdVueNodeMinWidth(panel, width) {
  const host = panel?.closest?.("[data-node-id]");
  const value = Number(width);
  if (!host?.style || !Number.isFinite(value) || value <= 0) return false;
  const css = `${Math.round(value)}px`;
  if (host.style.minWidth === css) return false;
  host.style.minWidth = css;
  return true;
}

// Nodes 2.0 has no layout API for a panel's minimum height either. It lays
// a node out by its content, and its corner drag stops where that content
// cannot get any shorter. A panel whose content has no height of its own (a
// picture kept out of flow, a stage that flexes) has nothing to stop at: a
// node could be dragged until the stage was a 2px line. So while a growing
// panel sits inside a Nodes 2.0 node, its element carries its floor as a CSS
// min-height: the declared floor less WIDGET_FRAME, which is the height the
// classic renderer promises the element. Outside a [data-node-id] element
// (the classic renderer, which keeps the floor in its own layout) the
// element carries none, so a panel moved between the two is always right.
export function holdVuePanelMinHeight(panel, height) {
  if (!panel?.style) return false;
  const value = Number(height);
  const inside = Boolean(panel.closest?.("[data-node-id]"));
  const css = inside && Number.isFinite(value) && value > 0 ? `${Math.round(value)}px` : "";
  if ((panel.style.minHeight || "") === css) return false;
  panel.style.minHeight = css;
  return true;
}

// Grow a node to the height its widgets ask for. A workflow saved before a
// panel or card existed carries the node's old size, and the frontend keeps
// that size on load - the new panel is then squeezed under its floor and
// clipped flat. Returns true when the node was resized.
export function ensureNodeMinHeight(node) {
  const min = Number(node?.computeSize?.()?.[1]);
  const current = Number(node?.size?.[1]);
  if (!Number.isFinite(min) || !Number.isFinite(current) || current >= min) return false;
  node.setSize?.([node.size[0], min]);
  return true;
}

// The height a node takes when its card's own height changes: a group such
// as More opens, or a row appears.
// - A new node, and a node that sat at its floor, hug the new floor.
// - A node the person made taller keeps that: when they changed the card by
//   hand it grows or shrinks by what the card did, so a picture under the
//   card keeps its size when More opens.
// - When the card changed by itself (a row that settles after a load, a
//   value written from outside) the node keeps the height it has, lifted to
//   the floor.
// - A workflow that is loading takes its saved height, lifted to the floor:
//   that height already holds the rows the saved values show. `saved` is
//   passed because the frontend has by then grown the node to fit the card
//   as it was before the saved values reached it.
export function nodeHeightAfterCardChange({ floor, current, saved, change = 0, first = false, restoring = false, byHand = false } = {}) {
  const min = Number(floor);
  const now = Number(current);
  if (!Number.isFinite(min)) return Number.isFinite(now) ? now : 0;
  if (first || !Number.isFinite(now)) return min;
  if (restoring) return Math.max(min, Number.isFinite(Number(saved ?? NaN)) ? Number(saved) : now);
  const delta = Number(change) || 0;
  // What the node had beyond its floor before the card changed.
  const spare = now - (min - delta);
  if (spare < 1) return min;
  return Math.max(min, byHand ? now + delta : now);
}

// Lift a node's computeSize() height to a floor the panel measures, such as
// the height a card's text needs. A corner drag stops at computeSize(), so
// the node cannot be dragged shorter than that. The floor is read on every
// call; anything but a finite number leaves the frontend's own height.
// This wraps the node's own method, not the prototype and not the widget,
// so the panel still takes all the height above its floor.
export function holdNodeMinHeight(node, floor) {
  const own = node?.computeSize;
  if (typeof own !== "function" || typeof floor !== "function") return node;
  node.computeSize = function (...args) {
    const size = own.apply(this, args);
    const min = Number(floor());
    if (size && Number.isFinite(min) && min > size[1]) size[1] = min;
    return size;
  };
  return node;
}
