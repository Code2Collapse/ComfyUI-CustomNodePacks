import { app } from "../../scripts/app.js";

/**
 * InpaintCompositeMEC — conditional widget visibility.
 *
 * The unified composite node exposes both the Stitch Pro and Paste Back
 * parameter sets. We hide widgets that don't apply to the selected mode
 * so the node UI stays uncluttered. Hidden widgets keep their last value
 * and still serialize, so backend defaults are always honoured.
 */

const STITCH_WIDGETS = ["blend_mode_override", "color_match"];
const PASTE_WIDGETS = ["upscale_method", "feather_edges", "feather_radius"];

// Static recovery map for the legacy `type="hidden"` bug.
const ORIGINAL_TYPES = {
    mode: "combo",
    blend_mode_override: "combo",
    color_match: "toggle",
    upscale_method: "combo",
    feather_edges: "toggle",
    feather_radius: "number",
};

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    // Modern ComfyUI (Vue frontend) honours `widget.hidden` natively for
    // both layout and rendering. Older LiteGraph also responds to it.
    // We additionally mirror the flag onto `widget.options.hidden` because
    // some widget types (INT/sliders) check there.
    widget.hidden = !visible;
    if (widget.options) widget.options.hidden = !visible;
    // Collapse DOM-backed widgets too (textareas, sliders rendered via DOM).
    const el = widget.element;
    if (el) {
        if (!visible) {
            if (!Object.prototype.hasOwnProperty.call(widget, "origElDisplay")) {
                widget.origElDisplay = el.style.display || "";
            }
            el.style.display = "none";
            const wrapper = el.parentElement;
            if (wrapper && wrapper.classList?.contains("dom-widget")) {
                if (!Object.prototype.hasOwnProperty.call(widget, "origWrapperDisplay")) {
                    widget.origWrapperDisplay = wrapper.style.display || "";
                }
                wrapper.style.display = "none";
            }
        } else {
            if (Object.prototype.hasOwnProperty.call(widget, "origElDisplay")) {
                el.style.display = widget.origElDisplay ?? "";
                delete widget.origElDisplay;
            }
            const wrapper = el.parentElement;
            if (wrapper && wrapper.classList?.contains("dom-widget")) {
                if (Object.prototype.hasOwnProperty.call(widget, "origWrapperDisplay")) {
                    wrapper.style.display = widget.origWrapperDisplay ?? "";
                    delete widget.origWrapperDisplay;
                }
            }
        }
    }
    // Recover from the legacy type="hidden" + collapsed-computeSize bug.
    // When making a widget visible: if its type is still "hidden", restore
    // the canonical type from ORIGINAL_TYPES (or origType if present), and
    // strip any own `computeSize` override so LiteGraph's default sizer runs.
    if (visible) {
        if (widget.type === "hidden") {
            const orig = widget.origType ?? ORIGINAL_TYPES[widget.name];
            if (orig) widget.type = orig;
        }
        delete widget.origType;
        if (Object.prototype.hasOwnProperty.call(widget, "computeSize")) {
            delete widget.computeSize;
        }
        delete widget.origComputeSize;
    }
}

function applyMode(node, mode) {
    if (!node.widgets) return;
    const showStitch = mode === "stitch_pro";
    const showPaste = mode === "paste_back";
    for (const w of node.widgets) {
        if (STITCH_WIDGETS.includes(w.name)) setWidgetVisible(w, showStitch);
        else if (PASTE_WIDGETS.includes(w.name)) setWidgetVisible(w, showPaste);
    }
    // Force re-layout
    const sz = node.computeSize();
    node.size[1] = sz[1];
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "MEC.InpaintComposite",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "InpaintCompositeMEC") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);
            const modeWidget = this.widgets?.find((w) => w.name === "mode");
            if (modeWidget) {
                const origCallback = modeWidget.callback;
                modeWidget.callback = (value) => {
                    const r = origCallback?.call(modeWidget, value);
                    applyMode(this, value);
                    return r;
                };
                applyMode(this, modeWidget.value ?? "stitch_pro");
            }
            return ret;
        };

        // Re-apply after a workflow is loaded (widget values restored).
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const ret = onConfigure?.apply(this, arguments);
            const modeWidget = this.widgets?.find((w) => w.name === "mode");
            if (modeWidget) applyMode(this, modeWidget.value ?? "stitch_pro");
            return ret;
        };
    },
});
