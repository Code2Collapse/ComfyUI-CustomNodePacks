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

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (visible) {
        // Restore type if we previously hid it.
        if (widget.origType !== undefined) {
            widget.type = widget.origType;
            delete widget.origType;
        }
        if (widget.origComputeSize !== undefined) {
            widget.computeSize = widget.origComputeSize;
            delete widget.origComputeSize;
        }
    } else {
        if (widget.origType === undefined) widget.origType = widget.type;
        if (widget.origComputeSize === undefined) widget.origComputeSize = widget.computeSize;
        widget.type = "hidden";
        widget.computeSize = () => [0, -4]; // collapse layout slot
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
