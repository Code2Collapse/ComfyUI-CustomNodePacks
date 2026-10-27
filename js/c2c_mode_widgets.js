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

const CLEAN = ["balance_mode", "balance_strength", "saturation", "contrast_restore", "chroma_cleanup",
               "restore_unchanged", "restore_radius"];
// Layer Effects: each effect's controls carry its name as a prefix and show while the effect is switched on
const EFFECTS = ["drop_shadow", "outer_glow", "gradient_map", "gradient_overlay", "color_overlay", "inner_glow",
                 "inner_shadow", "stroke"];
const effectOf = (name) => EFFECTS.find((k) => name.startsWith(k + "_")) || null;
// Mask Tools: mode label -> the prefix its controls carry
const MT_MODES = {
    "key: colour": "colour_key", "key: luma": "luma_key", "key: difference": "difference_key", "grade": "grade",
    "grow / shrink": "grow", "combine": "combine", "gradient": "gradient", "grain": "grain",
    "motion blur": "motion_blur", "edge spread": "edge_spread", "shuffle": "shuffle", "transform": "transform",
};
const mtKeyOf = (name) => Object.values(MT_MODES).find((k) => name === k || name.startsWith(k + "_")) || null;
// Folder Version Incrementer: the pipeline layout's controls, and the source / date layout's
const FI_PIPELINE = ["show", "shot", "task", "min_version", "max_retries", "write_manifest"];
const FI_SOURCE = ["prefix", "suffix", "suffix_mode", "label", "date_format", "source_choice", "name_format",
                   "numbered_still_mode", "source_filename", "source_path", "custom_name", "folder_name_override",
                   "version_group", "versioning", "version_in_filename", "output_name"];

// Inpaint Crop Pro: what each switch reveals
const ICP_PRERESIZE = ["preresize_mode", "preresize_min_width", "preresize_min_height", "preresize_max_width",
                       "preresize_max_height"];
const ICP_EXTEND = ["extend_up_factor", "extend_down_factor", "extend_left_factor", "extend_right_factor"];
const ICP_VSTABLE = ["video_stable_temporal_sigma", "video_stable_dilate_px", "video_stable_blur_sigma"];
const ICP_GATED = [...ICP_PRERESIZE, ...ICP_EXTEND, "output_target_width", "output_target_height", ...ICP_VSTABLE];
// L7.65 P08: crop_mode / stitch_mode pick one group. A control belongs to the mode its prefix names; every other
// input of the node belongs to the default mode (pro / blend). Only the node's own inputs are managed (read from its
// definition), never a widget another extension adds; the mode switch itself (and preview_frame) always shows.
const defInputs = (type) => {
    const i = globalThis.LiteGraph?.registered_node_types?.[type]?.nodeData?.input ?? {};
    return new Set([...Object.keys(i.required ?? {}), ...Object.keys(i.optional ?? {})]);
};
const icpModeOf = (name) => (name.startsWith("smart_") ? "smart aspect"
    : name.startsWith("prepare_") ? "mask prepare" : "pro");
const ICP_ALWAYS = ["crop_mode", "preview_frame"];
const ispModeOf = (name) => (name.startsWith("paste_") ? "paste back" : name.startsWith("smart_") ? "smart"
    : name.startsWith("context_") ? "context blend" : "blend");

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
        if (n === "combine") { if (["paint", "spline", "roto"].includes(mode)) keep.push(n); continue; }
        if (n === "canvas_layer") { if (mode === "paint") keep.push(n); continue; }
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
    ImageBatchSliceMEC: {
        watch: ["mode"],
        managed: ["start", "end", "step", "split_index", "split_ratio", "frame_index"],
        visible: (v) => ({
            "range": ["start", "end", "step"],
            "split at index": ["split_index"],
            "split at ratio": ["split_ratio"],
            "frame at index": ["frame_index"],
        })[v.mode ?? "range"] ?? [],
    },
    AspectPresetMEC: {
        watch: ["preset"],
        managed: ["base", "width", "height"],
        visible: (v) => {
            const p = String(v.preset ?? "");
            if (p.startsWith("Custom")) return ["width", "height"];
            return p.startsWith("Wan") ? [] : ["base"];
        },
    },
    VAEBlockInspectorMEC: {
        watch: ["mode"],
        managed: ["anomaly_threshold", "include_per_tensor"],
        visible: (v) => (v.mode === "compare" ? ["include_per_tensor"] : ["anomaly_threshold"]),
    },
    LayerEffectsMEC: {
        watch: EFFECTS,
        managed: (name) => effectOf(name) !== null,
        visible: (v, names) => names.filter((n) => v[effectOf(n)]),
    },
    MaskToolsMEC: {
        // the luma key's presets set the window themselves: low / high only matter in its "custom" setting
        watch: ["mode", "luma_key_mode"],
        managed: (name) => mtKeyOf(name) !== null,
        visible: (v, names) => names.filter((n) => mtKeyOf(n) === MT_MODES[v.mode]
            && !((n === "luma_key_low" || n === "luma_key_high") && v.luma_key_mode !== "custom")),
    },
    FolderIncrementer: {
        // layout "show / shot / task / version" (the former Batch Version Manager) uses none of the source / date
        // controls; the default layout uses none of the pipeline ones. source_extension is the node's own (hidden).
        // version_in_filename only means something for folder versioning (filename versioning always does it).
        watch: ["layout", "versioning"],
        managed: [...FI_PIPELINE, ...FI_SOURCE],
        visible: (v) => (String(v.layout ?? "").startsWith("show") ? FI_PIPELINE
            : FI_SOURCE.filter((n) => !(n === "version_in_filename" && v.versioning === "filename"))),
    },
    InpaintCropProMEC: {
        // L7.99: a switch shows the controls it enables (this file is the node's one visibility owner since the
        // separate inpaint_crop_pro.js was retired); the rarely used switches themselves sit in sections.
        // L7.65 P08: crop_mode shows its own group; in smart aspect / mask prepare every pro control (and with them
        // the section bar) hides.
        watch: ["crop_mode", "preresize", "extend_for_outpainting", "output_resize_to_target_size", "stitch_blend_mode",
                "smart_resolution_mode", "smart_no_mask_mode", "prepare_remove_small_regions",
                "prepare_temporal_smooth"],
        managed: (name) => !ICP_ALWAYS.includes(name) && defInputs("InpaintCropProMEC").has(name),
        visible: (v, names) => {
            const mode = v.crop_mode ?? "pro";
            const auto = (v.smart_resolution_mode ?? "Automatic") !== "Manual";
            const hide = new Set(mode === "pro" ? ICP_GATED.filter((n) => !(
                (v.preresize && ICP_PRERESIZE.includes(n))
                || (v.extend_for_outpainting && ICP_EXTEND.includes(n))
                || (v.output_resize_to_target_size !== false && n.startsWith("output_target_"))
                || (v.stitch_blend_mode === "video_stable" && ICP_VSTABLE.includes(n))))
                : mode === "smart aspect" ? [
                    ...(auto ? ["smart_manual_width", "smart_manual_height"] : ["smart_min_resolution"]),
                    // the maximum also sizes "Resize Full Image" when the mask is empty
                    ...(!auto && v.smart_no_mask_mode !== "Resize Full Image" ? ["smart_max_resolution"] : []),
                ] : [
                    ...(v.prepare_remove_small_regions === false ? ["prepare_min_region_area"] : []),
                    ...(v.prepare_temporal_smooth ? [] : ["prepare_temporal_sigma"]),
                ]);
            return names.filter((n) => icpModeOf(n) === mode && !hide.has(n));
        },
    },
    InpaintStitchProMEC: {
        // L7.65 P08: stitch_mode shows its own group (blend = the controls the node always had)
        watch: ["stitch_mode", "paste_feather_edges", "smart_blend_mode", "smart_enable_color_match"],
        managed: (name) => name !== "stitch_mode" && defInputs("InpaintStitchProMEC").has(name),
        visible: (v, names) => names.filter((n) => ispModeOf(n) === (v.stitch_mode ?? "blend")
            && !(n === "paste_feather_radius" && !v.paste_feather_edges)
            && !(n === "smart_feather_pixels" && v.smart_blend_mode === "Hard Paste")
            && !(n === "smart_color_match_amount" && !v.smart_enable_color_match)),
    },
    ImageMaskEditorC2C: {
        watch: ["mode", "canvas_layer", "shape_input", "shape_kind", "spline_fill"],
        managed: (name) => name !== "mode" && !["grow", "feather", "threshold", "invert"].includes(name)
            && defInputs("ImageMaskEditorC2C").has(name),
        visible: imeVisible,
    },
    C2CVAEQualityDecode: {
        watch: ["tile_mode", "apply_aces", "clean"],
        managed: ["tile_size", "exposure", ...CLEAN],
        visible: (v) => [
            ...(v.tile_mode === "manual" ? ["tile_size"] : []),
            ...(v.apply_aces ? ["exposure"] : []),
            ...(v.clean ? CLEAN : []),
        ],
    },
};

function apply(node, rule) {
    const widgets = node.widgets || [];
    const values = Object.fromEntries(widgets.filter((w) => rule.watch.includes(w.name)).map((w) => [w.name, w.value]));
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
            const after = nodeType.prototype.onAfterGraphConfigured;
            nodeType.prototype.onAfterGraphConfigured = function (...a) {
                const r = after?.apply(this, a);
                try { hook(this, rule); apply(this, rule); } catch (err) { __c2cReport("c2c_mode_widgets", err); }
                return r;
            };
        },
    });
}
