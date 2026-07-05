// MotionMaskTrackerMEC + the unified MaskTrackerMEC declutter.
//
// Two levels, one owner (so only one extension hooks `mode`/`detection_mode`
// and does the resize — no cross-extension stomping, per the spline-editor
// mode-gate lesson):
//
//   1. MODE level (unified MaskTrackerMEC only): the node has 4 modes
//      (motion / propagate / anchor / consistency_check) but 33 widgets. Each
//      non-motion mode uses a tiny param group; show only the active mode's
//      group and hide the rest. Groups are taken from the backend dispatch in
//      mask_tracker_mec.py `_execute_impl` (the source of truth) — a hidden
//      widget is genuinely unread by that mode, never a guess.
//
//   2. DETECTION level (motion mode + the standalone MotionMaskTrackerMEC):
//      the per-method threshold/option widgets only show when their method is
//      active in `detection_mode` (combined shows all).
import { app } from "../../scripts/app.js";
import { setHidden } from "./_widget_visibility.js";

const TARGET_NODES = ["MaskTrackerMEC", "MotionMaskTrackerMEC"];

// ── detection sub-groups (gated by detection_mode, motion only) ──────────────
const PIXEL_DIFF = ["pixel_diff_enabled", "pixel_diff_threshold"];
const FLOW       = ["flow_enabled", "flow_threshold", "flow_algorithm"];
const BG_SUB     = ["bg_sub_enabled", "bg_model_frames", "bg_sub_threshold"];
const HIST       = ["hist_enabled", "hist_grid_size", "hist_threshold"];
const COMBINE    = ["combine_method"];
const DETECTION  = [...PIXEL_DIFF, ...FLOW, ...BG_SUB, ...HIST, ...COMBINE];

// ── per-mode widget groups (from _execute_impl dispatch) ─────────────────────
const MOTION_ALWAYS = ["camera_compensation", "stabilization_method",
    "detection_mode", "grow_pixels", "min_region_size", "temporal_smooth"];
const PROPAGATE = ["source_frame", "propagate_mode", "prop_flow_threshold",
    "fade_start", "fade_end", "bidirectional", "points_json"];
const ANCHOR = ["anchor_frames", "total_frames", "easing", "sdf_iterations",
    "flow_refinement"];
const CONSISTENCY = ["metric", "binarize_threshold"];

function _setGroup(get, names, show) {
    for (const n of names) { const w = get(n); if (w) setHidden(w, !show); }
}

// Detection sub-grouping: show each method's widgets only when active.
function _applyDetection(get) {
    const dm = String(get("detection_mode")?.value ?? "combined");
    const combined = dm === "combined";
    _setGroup(get, PIXEL_DIFF, combined || dm === "pixel_diff");
    _setGroup(get, FLOW,       combined || dm === "optical_flow");
    _setGroup(get, BG_SUB,     combined || dm === "background_sub");
    _setGroup(get, HIST,       combined || dm === "histogram_diff");
    _setGroup(get, COMBINE,    combined);
}

function _resize(node) {
    const sz = node.computeSize();
    node.size[0] = Math.max(node.size[0], sz[0]);
    node.size[1] = sz[1];
    node.setDirtyCanvas(true, true);
}

function applyVisibility(node) {
    const get = (n) => node.widgets?.find(w => w.name === n);

    // Standalone motion tracker: detection sub-grouping only (no mode widget).
    if (node.comfyClass === "MotionMaskTrackerMEC") {
        _applyDetection(get);
        _resize(node);
        return;
    }
    if (node.comfyClass !== "MaskTrackerMEC") return;

    const mode = String(get("mode")?.value ?? "motion");
    if (mode === "motion") {
        _setGroup(get, MOTION_ALWAYS, true);
        _applyDetection(get);                 // refine detection widgets
        _setGroup(get, PROPAGATE, false);
        _setGroup(get, ANCHOR, false);
        _setGroup(get, CONSISTENCY, false);
    } else {
        // leaving motion: hide every motion + detection widget
        _setGroup(get, MOTION_ALWAYS, false);
        _setGroup(get, DETECTION, false);
        _setGroup(get, PROPAGATE,   mode === "propagate");
        _setGroup(get, ANCHOR,      mode === "anchor");
        _setGroup(get, CONSISTENCY, mode === "consistency_check");
    }
    _resize(node);
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
