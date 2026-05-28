// MotionMaskTrackerMEC / MaskTrackerMEC(mode=motion):
// per-method threshold/option widgets only visible when their method is
// active in detection_mode AND the parent mode is "motion" (unified node).
import { app } from "../../scripts/app.js";
import { setHidden } from "./_widget_visibility.js";

const TARGET_NODES = ["MaskTrackerMEC", "MotionMaskTrackerMEC"];

function _isActive(node) {
    if (node.comfyClass === "MotionMaskTrackerMEC") return true;
    if (node.comfyClass !== "MaskTrackerMEC") return false;
    const modeW = node.widgets?.find(w => w.name === "mode");
    return String(modeW?.value ?? "") === "motion";
}

const PIXEL_DIFF = ["pixel_diff_enabled", "pixel_diff_threshold"];
const FLOW       = ["flow_enabled", "flow_threshold", "flow_algorithm"];
const BG_SUB     = ["bg_sub_enabled", "bg_model_frames", "bg_sub_threshold"];
const HIST       = ["hist_enabled", "hist_grid_size", "hist_threshold"];
const COMBINE    = ["combine_method"];

function applyVisibility(node) {
    if (!_isActive(node)) return;
    const get = (n) => node.widgets?.find(w => w.name === n);
    const mode = String(get("detection_mode")?.value ?? "combined");
    const combined = mode === "combined";
    const showPixel = combined || mode === "pixel_diff";
    const showFlow  = combined || mode === "optical_flow";
    const showBg    = combined || mode === "background_sub";
    const showHist  = combined || mode === "histogram_diff";

    const setGroup = (names, show) => {
        for (const n of names) { const w = get(n); if (w) setHidden(w, !show); }
    };
    setGroup(PIXEL_DIFF, showPixel);
    setGroup(FLOW,       showFlow);
    setGroup(BG_SUB,     showBg);
    setGroup(HIST,       showHist);
    setGroup(COMBINE,    combined);

    const sz = node.computeSize();
    node.size[0] = Math.max(node.size[0], sz[0]);
    node.size[1] = sz[1];
    node.setDirtyCanvas(true, true);
}

function hookWidget(node, name) {
    const w = node.widgets?.find(x => x.name === name);
    if (!w) return;
    const orig = w.callback;
    w.callback = (v, ...rest) => {
        const r = orig?.call(w, v, ...rest);
        applyVisibility(node);
        return r;
    };
}

app.registerExtension({
    name: "MEC.MotionMaskTracker.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODES.includes(nodeData.name)) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            hookWidget(this, "detection_mode");
            hookWidget(this, "mode"); // unified-node mode flips
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
    },
});
