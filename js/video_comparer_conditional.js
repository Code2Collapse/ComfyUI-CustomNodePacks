// VideoComparerC2C: hide the params that don't apply to the current `mode`.
// The node has 15 modes (wipe / onion / diff / side_by_side / per_channel /
// false_color / 4 scopes / bit_depth_crush / 3 audio / synced_player) but shows
// ALL ~15 param widgets at once — a wall of controls where most are inert for the
// selected mode. This collapses to just the relevant set per mode and resizes.
//
// The mode→widget map is derived from the backend dispatch in video_comparer.py
// (each _mode_* function's signature is the source of truth), so a hidden widget
// is genuinely unused by that mode — never a guess:
//   wipe            → wipe_position
//   onion           → onion_alpha
//   diff            → diff_gain, diff_gamma, diff_threshold, diff_mode
//   per_channel     → diff_gain
//   false_color     → diff_gain, false_color_lut
//   bit_depth_crush → bit_depth, diff_gain, false_color_lut
//   waveform/parade/vectorscope → scope_intensity   (histogram_scope uses none)
//   synced_player   → file_a, file_b, 📁 Upload A, 📁 Upload B
//   side_by_side / histogram_scope / audio_* → none (only the always-on set)
// Always shown: mode, frame_index, label_a, label_b, and the comparer_view canvas.
import { app } from "../../scripts/app.js";
import { vueSyncNodeWidgets } from "./_widget_visibility.js";

// widget -> modes it applies to. Anything not listed here is always shown and is
// never touched (notably the comparer_view COMPARER canvas, which the node's own
// JS manages — we must not fight it, per the spline-editor lesson).
const MODE_OF = {
    wipe_position:   ["wipe"],
    onion_alpha:     ["onion"],
    diff_gain:       ["diff", "per_channel", "false_color", "bit_depth_crush"],
    diff_gamma:      ["diff"],
    diff_threshold:  ["diff"],
    diff_mode:       ["diff"],
    false_color_lut: ["false_color", "bit_depth_crush"],
    bit_depth:       ["bit_depth_crush"],
    scope_intensity: ["waveform_scope", "parade_scope", "vectorscope"],
    file_a:          ["synced_player"],
    file_b:          ["synced_player"],
    "📁 Upload A":   ["synced_player"],
    "📁 Upload B":   ["synced_player"],
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
    const mode = String(modeW?.value ?? "synced_player");
    for (const w of node.widgets || []) {
        const modes = MODE_OF[w.name];
        // Only manage widgets we own; leave the comparer_view canvas and the
        // always-on widgets entirely alone.
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
    name: "MEC.VideoComparerC2C.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VideoComparerC2C" && nodeData.name !== "VideoComparerMEC") return;
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
