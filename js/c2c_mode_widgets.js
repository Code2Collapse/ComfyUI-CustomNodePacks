/**
 * Show only the controls the current mode uses, on the nodes the L7.65 consolidation merged (Batch Range, Size,
 * VAE Inspect, VAE Decode, Mask Tools, Inpaint Crop Pro, Inpaint Stitch Pro, ...). A merged node carries every control of the nodes it replaced; a migrated "Extract First
 * Frame" should not show a range, a split point and a frame index it does not use.
 *
 * Widgets only - sockets and links are never touched. A hidden widget keeps its value and still saves and still goes
 * into the prompt (the shared _widget_visibility.js helper changes how it draws, not what it holds).
 */
import { app } from "../../scripts/app.js";
import { setWidgetVisible, vueFitNodeHeight, vueSyncNodeWidgets } from "./_widget_visibility.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { sectionOf, sectionOpen, updateSectionBar } from "./_c2c_sections.js";

// A28 (ORDERS 29): Image Batch Slice, Size, VAE Inspect, Layer Effects, Folder Version Incrementer, Face Fixer and
// VAE Decode (C2C) moved to the declarative layout engine (js/_c2c_node_layout.js; rules next to their inputs in
// Python, docs/audit/A28/PARAM_CNP_B.md) - one visibility owner per node, so they are no longer listed here.
// A28 CNP part A: Mask Tools, Inpaint Crop Pro, Inpaint Stitch Pro and Load Video (C2C) moved to the same engine
// (rules in Python, docs/audit/A28/PARAM_CNP_A.md). Image Mask Editor stays here: its shape fields follow tables
// this file shares with the editors.
// Only the node's own inputs are managed (read from its definition), never a widget another extension adds.
const defInputs = (type) => {
    const i = globalThis.LiteGraph?.registered_node_types?.[type]?.nodeData?.input ?? {};
    return new Set([...Object.keys(i.required ?? {}), ...Object.keys(i.optional ?? {})]);
};
// L7.65 P01: Image Mask Editor modes. A control belongs to the mode its prefix names (nodes/image_mask_editor/modes.py);
// the editors' own data (points / spline / roto / session / canvas) is written by their UIs and never shown; grow,
// feather, threshold and invert apply to every mode and are not managed here.
const IME_PREFIX = [["canvas_", "paint"], ["points_", "points + box"], ["shape_", "shape"], ["spline_", "spline"],
                    ["roto_", "roto"], ["video_", "video"]];
const IME_DATA = new Set(["editor_id", "points_editor_data", "spline_data", "roto_json", "video_session_id",
                          "canvas_data"]);
// the Paint Canvas edits these in its own panel (js/mec_advanced_paint.js hides them)
const IME_CANVAS_PANEL = new Set(["canvas_brush_color", "canvas_brush_opacity", "canvas_brush_hardness",
                                  "canvas_brush_size"]);
// Mask Edit's per-shape fields (js/draw_shape.js), here with the shape_ prefix
const IME_SHAPE_FIELDS = {
    circle: ["cx", "cy", "radius"], rectangle: ["top_left_x", "top_left_y", "size_w", "size_h"],
    ellipse: ["cx", "cy", "rx", "ry"], polygon: ["points_json"], line: ["top_left_x", "top_left_y", "x2", "y2", "thickness"],
    triangle: ["cx", "cy", "radius"], star: ["cx", "cy", "outer_r", "inner_r", "num_points"],
    diamond: ["cx", "cy", "size_w", "size_h"], cross: ["cx", "cy", "cross_size", "thickness"],
    rounded_rectangle: ["top_left_x", "top_left_y", "size_w", "size_h", "corner_radius"],
    heart: ["cx", "cy", "radius"], arrow: ["cx", "cy", "arrow_length", "head_length", "head_width", "size_w"],
};
const IME_SHAPE_COMMON = ["shape_input", "shape_width", "shape_height", "shape_kind", "shape_value", "shape_feather",
                          "shape_rotation", "shape_operation"];
const IME_SPLINE_EDIT_ONLY = ["spline_smoothing", "spline_centripetal_alpha", "spline_mask_color",
                              "spline_mask_opacity", "spline_feather_radius"];
const imeModeOf = (n) => IME_PREFIX.find(([p]) => n.startsWith(p))?.[1] ?? null;

function imeVisible(v, names) {
    const mode = v.mode || "paint";
    const keep = [];
    for (const n of names) {
        if (IME_DATA.has(n)) continue;                                         // never shown
        if (n === "frame_mode") { if (mode === "paint" && v.canvas_layer !== "colour") keep.push(n); continue; }
        // A28: combine mixes input_mask in, so it only does something while input_mask is wired (node.py:362, 395)
        if (n === "combine") {
            if (["paint", "spline", "roto"].includes(mode) && v["linked:input_mask"]) keep.push(n);
            continue;
        }
        if (n === "canvas_layer") { if (mode === "paint") keep.push(n); continue; }
        // L9.172: which layer of the paint editor's stack is output (alpha layers only - the colour canvas has none)
        if (n === "layer_output") { if (mode === "paint" && v.canvas_layer !== "colour") keep.push(n); continue; }
        if (imeModeOf(n) !== mode) continue;
        if (mode === "paint") {
            if (v.canvas_layer === "colour" && !IME_CANVAS_PANEL.has(n)) keep.push(n);
        } else if (mode === "shape") {
            const json = v.shape_input === "JSON";
            const fields = json ? ["shape_json"] : [...(IME_SHAPE_FIELDS[v.shape_kind] ?? []).map((f) => `shape_${f}`),
                                                   "shape_batch_size"];
            if (IME_SHAPE_COMMON.includes(n) || fields.includes(n)) keep.push(n);
        } else if (mode === "spline") {
            const flow = v.spline_fill === "flow pattern";
            if (n.startsWith("spline_flow_") ? flow : !(flow && IME_SPLINE_EDIT_ONLY.includes(n))) keep.push(n);
        } else {
            keep.push(n);
        }
    }
    return keep;
}

// node type -> { watch: widgets whose value decides, managed: widgets this file shows / hides (a list, or a test on the
// name), visible(values, names) -> the managed widgets to show }
const RULES = {
    ImageMaskEditorC2C: {
        watch: ["mode", "canvas_layer", "shape_input", "shape_kind", "spline_fill"],
        links: ["input_mask"],             // values["linked:input_mask"]; re-applied on connect / disconnect
        managed: (name) => name !== "mode" && !["grow", "feather", "threshold", "invert"].includes(name)
            && defInputs("ImageMaskEditorC2C").has(name),
        visible: imeVisible,
    },
};

function apply(node, rule) {
    const widgets = node.widgets || [];
    const values = Object.fromEntries(widgets.filter((w) => rule.watch.includes(w.name)).map((w) => [w.name, w.value]));
    for (const s of rule.links || []) values[`linked:${s}`] = node.inputs?.find((i) => i.name === s)?.link != null;
    const managed = typeof rule.managed === "function" ? rule.managed : (n) => rule.managed.includes(n);
    const show = new Set(rule.visible(values, widgets.map((w) => w.name).filter(managed)));
    let changed = false;
    let prevWant = null;                                 // a seed's control_after_generate follows its seed
    const shown = [];                                    // in a section and wanted by the rule (for the bar)
    for (const w of widgets) {
        const isControl = w.name === "control_after_generate";
        const inSection = !isControl && sectionOf(node, w.name) !== null;
        if (!managed(w.name) && !inSection && !(isControl && prevWant !== null)) { prevWant = null; continue; }
        // the rule decides what the mode uses; a closed section (L7.81, classic canvas) folds its members too
        const ruled = isControl ? prevWant : (managed(w.name) ? show.has(w.name) : true);
        if (ruled && inSection) shown.push(w.name);
        const want = isControl ? prevWant : (ruled && sectionOpen(node, w.name));
        prevWant = isControl ? null : want;
        if (want === (w.__mec_hidden === true)) {
            setWidgetVisible(w, want);
            changed = true;
        }
    }
    const hadBar = node.widgets?.some((x) => x.name === "_c2c_sections");
    updateSectionBar(node, shown, () => apply(node, rule));
    if (!changed && hadBar === node.widgets?.some((x) => x.name === "_c2c_sections")) return;
    vueSyncNodeWidgets(node);
    const sz = node.computeSize?.();
    if (sz) node.setSize?.([Math.max(node.size?.[0] ?? 0, sz[0]), sz[1]]);
    vueFitNodeHeight(node);                              // Nodes 2.0: content height, not the classic one
    node.setDirtyCanvas?.(true, true);
}

function hook(node, rule) {
    if (node.__c2cModeWidgets) return;
    node.__c2cModeWidgets = true;
    for (const w of node.widgets || []) {
        if (!rule.watch.includes(w.name)) continue;
        const cb = w.callback;
        w.callback = function (...a) {
            const r = cb?.apply(this, a);
            try { apply(node, rule); } catch (err) { __c2cReport("c2c_mode_widgets", err); }
            return r;
        };
    }
}

if (!globalThis.__c2cModeWidgetsExt) {
    globalThis.__c2cModeWidgetsExt = true;
    app.registerExtension({
        name: "C2C.ModeWidgets",
        beforeRegisterNodeDef(nodeType, nodeData) {
            const rule = RULES[nodeData?.name];
            if (!rule) return;
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...a) {
                const r = created?.apply(this, a);
                queueMicrotask(() => {
                    try { hook(this, rule); apply(this, rule); } catch (err) { __c2cReport("c2c_mode_widgets", err); }
                });
                return r;
            };
            // values arrive with configure (saved and migrated workflows); apply once the graph is in
            // A28: a rule that reads a socket (rule.links) runs again when a link is made or removed
            if (rule.links?.length) {
                const conn = nodeType.prototype.onConnectionsChange;
                nodeType.prototype.onConnectionsChange = function (...a) {
                    const r = conn?.apply(this, a);
                    if (this.__c2cModeWidgets) queueMicrotask(() => {
                        try { if (this.graph) apply(this, rule); } catch (err) { __c2cReport("c2c_mode_widgets", err); }
                    });
                    return r;
                };
            }
            const after = nodeType.prototype.onAfterGraphConfigured;
            nodeType.prototype.onAfterGraphConfigured = function (...a) {
                const r = after?.apply(this, a);
                try { hook(this, rule); apply(this, rule); } catch (err) { __c2cReport("c2c_mode_widgets", err); }
                return r;
            };
        },
    });
}
