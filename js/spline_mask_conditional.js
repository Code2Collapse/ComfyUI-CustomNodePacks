// SplineMaskMEC: HIDE widgets that don't apply to the current `mode`
// (edit / track / flow_path). The node exposes 33 widgets — most belong to only
// one mode (the tooltips tag each as [edit] / [track] / [flow_path]). Showing all
// at once is the "wall of params" clutter; this collapses to just the relevant
// set per mode and resizes the node. Re-applied whenever `mode` changes.
import { app } from "../../scripts/app.js";
import { vueSyncNodeWidgets } from "./_widget_visibility.js";

// widget -> modes it applies to. Anything not listed is always shown
// (mode, spline_data, image, closed, samples_per_segment, the spline_editor canvas).
const MODE_OF = {
    spline_type:      ["edit", "flow_path"],
    invert:           ["edit", "flow_path"],
    width:            ["edit", "flow_path"],
    height:           ["edit", "flow_path"],
    feather_radius:   ["edit", "track"],
    smoothing:        ["edit"],
    centripetal_alpha:["edit"],
    mask_color:       ["edit"],
    mask_opacity:     ["edit"],
    tracking_weight:  ["track"],
    klt_window:       ["track"],
    stroke_width:     ["track"],
    pattern:          ["flow_path"],
    thickness:        ["flow_path"],
    amplitude:        ["flow_path"],
    frequency:        ["flow_path"],
    turbulence:       ["flow_path"],
    turbulence_scale: ["flow_path"],
    edge_softness:    ["flow_path"],
    taper_start:      ["flow_path"],
    taper_end:        ["flow_path"],
    frames:           ["flow_path"],
    animation_speed:  ["flow_path"],
    flow_direction:   ["flow_path"],
    mod_decay:        ["flow_path"],
    seed:             ["flow_path"],
    // seed's auto-generated companion widget — keep it with its seed so it
    // doesn't sit orphaned-visible when seed is hidden.
    control_after_generate: ["flow_path"],
    // editor toggle is used when drawing (edit or flow_path), not when tracking
    use_embedded_editor: ["edit", "flow_path"],
};

function setHidden(w, hidden) {
    if (!w) return;
    if (hidden) {
        if (!("__mec_origType" in w)) w.__mec_origType = w.type;
        if (!("__mec_origCS" in w)) w.__mec_origCS = w.computeSize;
        w.type = "hidden";
        w.computeSize = () => [0, -4];
        w.hidden = true;
        if (w.element) { if (!("__mec_origDisp" in w)) w.__mec_origDisp = w.element.style.display; w.element.style.display = "none"; }
    } else {
        if ("__mec_origType" in w) { w.type = w.__mec_origType; delete w.__mec_origType; }
        if ("__mec_origCS" in w) { const cs = w.__mec_origCS; if (cs === undefined) delete w.computeSize; else w.computeSize = cs; delete w.__mec_origCS; }
        w.hidden = false;
        if (w.element) { w.element.style.display = ("__mec_origDisp" in w) ? (w.__mec_origDisp ?? "") : ""; delete w.__mec_origDisp; }
    }
}

function applyVisibility(node) {
    const modeW = node.widgets?.find(w => w.name === "mode");
    const mode = String(modeW?.value ?? "edit");
    for (const w of node.widgets || []) {
        const modes = MODE_OF[w.name];
        // Only manage widgets we own. Everything else (the spline_editor canvas,
        // spline_data, etc.) is shown/hidden by the node's own native JS based on
        // use_embedded_editor — do NOT force it visible or we fight that logic.
        if (!modes) continue;
        setHidden(w, !modes.includes(mode));
    }
    vueSyncNodeWidgets(node);
    const sz = node.computeSize();
    node.size[0] = Math.max(node.size[0], sz[0]);
    node.size[1] = sz[1];
    node.setDirtyCanvas(true, true);
}

function hookWidget(node, name) {
    const w = node.widgets?.find(x => x.name === name);
    if (!w) return;
    const orig = w.callback;
    w.callback = (v, ...rest) => { const r = orig?.call(w, v, ...rest); applyVisibility(node); return r; };
}

app.registerExtension({
    name: "MEC.SplineMaskMEC.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SplineMaskMEC") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            hookWidget(this, "mode");
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
