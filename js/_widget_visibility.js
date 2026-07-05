// Shared widget show/hide helper for MEC nodes.
//
// Modern ComfyUI renders many widgets via DOM (number inputs, textareas,
// combo selects). Setting ``widget.type = "hidden"`` and collapsing
// ``computeSize`` only fixes the LiteGraph layout — the DOM element and
// its ``.dom-widget`` wrapper stay visible on top of the node, producing
// the "I toggled false then true and the field never came back / never
// went away" symptom. This helper toggles BOTH layers and remembers the
// original styles so re-showing restores them exactly.
//
// Usage:
//     import { setWidgetVisible } from "./_widget_visibility.js";
//     setWidgetVisible(w, true);   // show
//     setWidgetVisible(w, false);  // hide

export function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (visible) {
        if ("__mec_origType" in widget) {
            const t = widget.__mec_origType;
            if (t === undefined) delete widget.type; else widget.type = t;
            delete widget.__mec_origType;
        }
        if ("__mec_origComputeSize" in widget) {
            const cs = widget.__mec_origComputeSize;
            if (cs === undefined) delete widget.computeSize; else widget.computeSize = cs;
            delete widget.__mec_origComputeSize;
        }
        // No else: if no original was saved we never hid this widget, so its
        // computeSize is already the legitimate one. Deleting it here would
        // destroy DOM-widget sizing (multiline/preview widgets) on widgets that
        // were only ever shown. (Mirrors the `type` branch above — no else.)
        widget.hidden = false;
        const el = widget.element;
        if (el) {
            if ("__mec_origElDisplay" in widget) {
                const d = widget.__mec_origElDisplay;
                el.style.display = d ?? "";
                delete widget.__mec_origElDisplay;
            } else {
                el.style.display = "";
            }
            const wrap = el.parentElement;
            if (wrap && wrap.classList?.contains("dom-widget")) {
                if ("__mec_origWrapDisplay" in widget) {
                    const d = widget.__mec_origWrapDisplay;
                    wrap.style.display = d ?? "";
                    delete widget.__mec_origWrapDisplay;
                } else {
                    wrap.style.display = "";
                }
            }
        }
        delete widget.__mec_hidden;
        if (widget.options) widget.options.hidden = false;
    } else {
        if (!("__mec_origType" in widget)) widget.__mec_origType = widget.type;
        if (!("__mec_origComputeSize" in widget)) widget.__mec_origComputeSize = widget.computeSize;
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
        widget.hidden = true;
        widget.options = widget.options || {};
        widget.options.hidden = true;
        const el = widget.element;
        if (el) {
            if (!("__mec_origElDisplay" in widget)) widget.__mec_origElDisplay = el.style.display;
            el.style.display = "none";
            const wrap = el.parentElement;
            if (wrap && wrap.classList?.contains("dom-widget")) {
                if (!("__mec_origWrapDisplay" in widget)) widget.__mec_origWrapDisplay = wrap.style.display;
                wrap.style.display = "none";
            }
        }
        widget.__mec_hidden = true;
    }
}

export function setHidden(widget, hidden) {
    setWidgetVisible(widget, !hidden);
}

// ---------------------------------------------------------------------------
// Nodes 2.0 (Comfy.VueNodes.Enabled) compatibility.
//
// The Vue renderer builds each node's widget rows from a snapshot and decides
// visibility from `widget.options.hidden` — it ignores the legacy
// `type = "hidden"` / `computeSize = [0,-4]` collapse, rendering an empty row
// for every hidden widget (hundreds of px of dead space on mode-gated nodes).
// The snapshot only rebuilds when a widget is ADDED, so after a batch of
// hide/show changes we (1) mirror the legacy hidden state into
// `options.hidden` for every widget, then (2) nudge the rebuild by adding and
// immediately removing a throwaway widget (never rendered — the splice lands
// before the snapshot recomputes on the next frame).
// ---------------------------------------------------------------------------

function _vueNodesActive() {
    try {
        return window.app?.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled") === true;
    } catch (_) { return false; }
}

export function vueSyncNodeWidgets(node) {
    if (!node?.widgets || !_vueNodesActive()) return;
    for (const w of node.widgets) {
        if (!w) continue;
        w.options = w.options || {};
        w.options.hidden = (w.type === "hidden") || w.hidden === true;
    }
    // Only nudge once the Vue component for this node exists. Before the
    // initial mount the fresh snapshot reads options.hidden anyway, and a
    // rebuild racing the first mount detaches the node's DOM widgets
    // (timeline/player elements end up orphaned from their rows).
    if (!document.querySelector(`[data-node-id="${node.id}"]`)) return;
    if (node.__mecVueNudgePending) return;
    node.__mecVueNudgePending = true;
    requestAnimationFrame(() => requestAnimationFrame(() => {
        node.__mecVueNudgePending = false;
        try {
            const d = node.addWidget("number", "__mec_vue_sync__", 0, () => {}, { serialize: false });
            const i = node.widgets.indexOf(d);
            if (i >= 0) node.widgets.splice(i, 1);
        } catch (_) { /* node may be mid-removal */ }
    }));
}
