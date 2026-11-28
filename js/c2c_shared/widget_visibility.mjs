// PORTED FROM: ComfyUI-AusBoss js/shared/widget_visibility.mjs (https://github.com/ausboss/ComfyUI-AusBoss), MIT License,
// Copyright (c) 2026 ausboss. Licence text: js/c2c_shared/LICENSE-AUSBOSS.txt. Code unchanged at port (2026-11-28,
// agy review Phase 3 item 12: one shared UI foundation for the C2C packs).
// Show/hide standard canvas widgets in place. Hiding keeps the widget (and
// its serialized value) in the widgets array — only its rendering collapses —
// so widgets_values order never changes and saved workflows stay compatible.
// Pure widget-object surgery, no DOM: testable under node:test.
//
// Two renderers, two switches. The classic canvas lays widgets out by
// computeSize / computeLayoutSize and paints them with draw, so those are
// zeroed. The Nodes 2.0 renderer ignores all of that and filters its widget
// list on `widget.options.hidden` instead, so that flag is set as well - a
// panel's hidden storage widget stays hidden whichever renderer is on.

const STASH = "__ausbossVisibilityStash";
const PAINT_KEYS = ["computeSize", "computeLayoutSize", "draw"];

// Collapse a widget to zero height. Returns true when this call changed
// visibility, false when it was already hidden (callers use the return to
// skip a needless node resize).
export function hideWidget(widget) {
  if (!widget) return false;
  if (widget[STASH]) {
    // Already hidden: re-assert the renderer flag, which the frontend
    // drops whenever it rebuilds the options object from the node spec.
    if (widget.options) widget.options.hidden = true;
    return false;
  }
  let options = widget.options ?? {};
  widget[STASH] = {
    computeSize: widget.computeSize,
    computeLayoutSize: widget.computeLayoutSize,
    draw: widget.draw,
    // Which of the three the widget carried itself. One it lacked is deleted
    // again on show: left behind as `undefined`, the frontend takes it for a
    // custom-drawn widget and App mode draws an empty row.
    own: Object.fromEntries(PAINT_KEYS.map((key) => [key, Object.prototype.hasOwnProperty.call(widget, key)])),
    hidden: widget.hidden,
    optionsHidden: options.hidden,
    optionsDescriptor: Object.getOwnPropertyDescriptor(widget, "options"),
  };
  widget.hidden = true;
  options.hidden = true;
  // The frontend swaps a widget's options object after creation (the
  // after-generate control does, and so does a workflow restore), which
  // would drop the flag. An accessor keeps re-applying it to whatever
  // object is assigned, so the widget stays hidden under Nodes 2.0.
  try {
    Object.defineProperty(widget, "options", {
      configurable: true,
      enumerable: true,
      get: () => options,
      set: (next) => {
        options = next ?? {};
        options.hidden = true;
      },
    });
  } catch {
    widget.options = options;
  }
  widget.computeSize = () => [0, -4];
  widget.computeLayoutSize = () => ({ minWidth: 0, minHeight: 0 });
  widget.draw = () => {};
  return true;
}

// Restore a widget hidden by hideWidget. Returns true when this call changed
// visibility, false when the widget was never hidden.
export function showWidget(widget) {
  const stash = widget?.[STASH];
  if (!stash) return false;
  for (const key of PAINT_KEYS) {
    if (stash.own && !stash.own[key]) delete widget[key];
    else widget[key] = stash[key];
  }
  widget.hidden = stash.hidden;
  const options = widget.options;
  try {
    delete widget.options; // drop the accessor, then put a plain property back
  } catch {
    // Not configurable: the accessor stays and keeps the flag; nothing to do.
  }
  if (stash.optionsDescriptor && "value" in stash.optionsDescriptor) {
    Object.defineProperty(widget, "options", { ...stash.optionsDescriptor, value: options });
  } else {
    widget.options = options;
  }
  if (widget.options) {
    if (stash.optionsHidden === undefined) delete widget.options.hidden;
    else widget.options.hidden = stash.optionsHidden;
  }
  widget[STASH] = undefined;
  return true;
}

// True while hideWidget holds the widget hidden.
export function isWidgetHidden(widget) {
  return Boolean(widget?.[STASH]);
}

// Drive a widget toward a target visibility; true when anything changed.
export function setWidgetVisible(widget, visible) {
  return visible ? showWidget(widget) : hideWidget(widget);
}

// Mark inputs hidden in the node DEFINITION, before any widget exists.
// The frontend rebuilds a widget's options from this spec at several
// points (workflow restore, the after-generate control, the Nodes 2.0
// mount), so a flag written on a live widget is not enough on its own -
// a storage widget a panel always drives is hidden at the source, and
// hideWidget on the live widget covers the classic canvas layout.
//
// Never throws: a renderer that hands out frozen definitions gets a copied
// entry where it allows one and is otherwise left alone - the live-widget
// path still hides the widget on the classic canvas.
export function hideInputsInDef(nodeData, names) {
  const groups = nodeData?.input;
  if (!groups || typeof groups !== "object") return 0;
  let marked = 0;
  for (const name of names) {
    for (const group of ["required", "optional"]) {
      const spec = groups[group]?.[name];
      if (!Array.isArray(spec)) continue;
      const options = spec[1] && typeof spec[1] === "object" ? spec[1] : {};
      if (options.hidden === true) {
        marked += 1;
        continue;
      }
      try {
        options.hidden = true;
        if (spec[1] !== options) spec[1] = options;
        if (spec[1]?.hidden === true) {
          marked += 1;
          continue;
        }
      } catch {
        // frozen options: fall through to replacing the entry
      }
      try {
        groups[group][name] = [spec[0], { ...options, hidden: true }, ...spec.slice(2)];
        if (groups[group][name]?.[1]?.hidden === true) marked += 1;
      } catch {
        // frozen group: nothing more to do here
      }
    }
  }
  return marked;
}
