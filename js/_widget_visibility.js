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
    } else {
        if (!("__mec_origType" in widget)) widget.__mec_origType = widget.type;
        if (!("__mec_origComputeSize" in widget)) widget.__mec_origComputeSize = widget.computeSize;
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
        widget.hidden = true;
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
