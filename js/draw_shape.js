// MaskEditMEC / DrawShapeMEC: conditional widget visibility.
//
// MaskEditMEC is a unified dispatcher with ~40 widgets across 5 modes
// (transform / draw_shape / draw_advanced / points_bbox / bbox_smooth).
// Showing them all at once makes the node enormous. This extension is the
// single source of truth for MaskEditMEC widget visibility: it shows only
// the widgets that belong to the selected `mode`, and within draw_shape it
// further narrows the geometry widgets to the selected `shape`.
//
// (points_bbox_editor.js still owns the points_bbox *canvas* host via
// _mode_gate; this file only governs the scalar widgets, so the two compose
// without fighting.)
import { app } from "../../scripts/app.js";
import { setWidgetVisible } from "./_widget_visibility.js";

const TARGET_NODES = ["MaskEditMEC", "DrawShapeMEC"];

// Geometry widgets that draw_shape narrows down per shape.
const SHAPE_FIELDS = {
    circle:            ["cx", "cy", "radius"],
    rectangle:         ["top_left_x", "top_left_y", "size_w", "size_h"],
    ellipse:           ["cx", "cy", "rx", "ry"],
    polygon:           [],  // polygon uses the points_json[_shape] widget below
    line:              ["top_left_x", "top_left_y", "x2", "y2", "thickness"],
    triangle:          ["cx", "cy", "radius"],
    star:              ["cx", "cy", "outer_r", "inner_r", "num_points"],
    diamond:           ["cx", "cy", "size_w", "size_h"],
    cross:             ["cx", "cy", "cross_size", "thickness"],
    rounded_rectangle: ["top_left_x", "top_left_y", "size_w", "size_h", "corner_radius"],
    heart:             ["cx", "cy", "radius"],
    arrow:             ["cx", "cy", "arrow_length", "head_length", "head_width", "size_w", "rotation"],
};

// Non-geometry widgets that are always present in draw_shape mode.
const DRAW_SHAPE_COMMON = [
    "width", "height", "shape", "value", "feather", "rotation",
    "operation", "batch_size",
];

// Widgets visible for each mode OTHER than draw_shape (which is computed
// from SHAPE_FIELDS + DRAW_SHAPE_COMMON above).
const MODE_WIDGETS = {
    transform: [
        "expand_x", "expand_y", "blur_x", "blur_y", "offset_x", "offset_y",
        "feather", "threshold", "invert",
    ],
    draw_advanced: [
        "width", "height", "shape_params_json", "value", "rotation",
        "operation", "feather",
    ],
    points_bbox: [
        "width", "height", "editor_data", "default_radius", "softness",
        "normalize",
    ],
    bbox_smooth: [
        "bboxes_json", "smoothing_radius", "smoothing_method", "alpha",
    ],
};

// The polygon vertices widget is named differently on the two nodes.
const POLY_NAMES = new Set(["points_json", "points_json_shape"]);

function _mode(node) {
    if (node.comfyClass === "DrawShapeMEC") return "draw_shape";
    const w = node.widgets?.find((x) => x && x.name === "mode");
    return String(w?.value ?? "transform");
}

function _visibleSet(node) {
    const mode = _mode(node);
    if (mode === "draw_shape") {
        const shapeW = node.widgets?.find((w) => w && w.name === "shape");
        const shape = shapeW?.value ?? "circle";
        const set = new Set(DRAW_SHAPE_COMMON);
        for (const name of (SHAPE_FIELDS[shape] || [])) set.add(name);
        if (shape === "polygon") for (const n of POLY_NAMES) set.add(n);
        return set;
    }
    return new Set(MODE_WIDGETS[mode] || []);
}

function applyMaskEditVisibility(node) {
    if (!node || !Array.isArray(node.widgets)) return;
    if (!TARGET_NODES.includes(node.comfyClass)) return;
    const visible = _visibleSet(node);
    for (const w of node.widgets) {
        if (!w || !w.name) continue;
        if (w.name === "mode") continue;          // the selector is always shown
        setWidgetVisible(w, visible.has(w.name));
    }
    try {
        const sz = node.computeSize();
        node.size[0] = Math.max(node.size[0], sz[0]);
        node.size[1] = sz[1];
    } catch (_) { /* ignore */ }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "MEC.MaskEdit.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODES.includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const node = this;
            // Re-evaluate on both `mode` and `shape` changes.
            for (const wName of ["mode", "shape"]) {
                const w = node.widgets?.find((x) => x && x.name === wName);
                if (!w) continue;
                const orig = w.callback;
                w.callback = function (v, ...rest) {
                    const rr = orig?.call(this, v, ...rest);
                    applyMaskEditVisibility(node);
                    return rr;
                };
            }
            setTimeout(() => applyMaskEditVisibility(node), 0);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => applyMaskEditVisibility(this), 0);
            return r;
        };
    },
});
