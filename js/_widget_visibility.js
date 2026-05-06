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
        if (widget.__mec_origType !== undefined) {
            widget.type = widget.__mec_origType;
            delete widget.__mec_origType;
        }
        if (widget.__mec_origComputeSize !== undefined) {
            widget.computeSize = widget.__mec_origComputeSize;
            delete widget.__mec_origComputeSize;
        }
        widget.hidden = false;
        const el = widget.element;
        if (el) {
            el.style.display = widget.__mec_origElDisplay ?? "";
            delete widget.__mec_origElDisplay;
            const wrap = el.parentElement;
            if (wrap && wrap.classList?.contains("dom-widget")) {
                wrap.style.display = widget.__mec_origWrapDisplay ?? "";
                delete widget.__mec_origWrapDisplay;
            }
        }
        delete widget.__mec_hidden;
    } else {
        if (widget.__mec_origType === undefined) widget.__mec_origType = widget.type;
        if (widget.__mec_origComputeSize === undefined) widget.__mec_origComputeSize = widget.computeSize;
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
        widget.hidden = true;
        const el = widget.element;
        if (el) {
            if (widget.__mec_origElDisplay === undefined) {
                widget.__mec_origElDisplay = el.style.display || "";
            }
            el.style.display = "none";
            const wrap = el.parentElement;
            if (wrap && wrap.classList?.contains("dom-widget")) {
                if (widget.__mec_origWrapDisplay === undefined) {
                    widget.__mec_origWrapDisplay = wrap.style.display || "";
                }
                wrap.style.display = "none";
            }
        }
        widget.__mec_hidden = true;
    }
}

export function setHidden(widget, hidden) {
    setWidgetVisible(widget, !hidden);
}
