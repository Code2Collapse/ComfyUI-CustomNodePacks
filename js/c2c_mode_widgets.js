/**
 * Show only the controls the current mode uses, on the nodes the L7.65 consolidation merged (Batch Range, Size,
 * VAE Inspect, VAE Decode). A merged node carries every control of the nodes it replaced; a migrated "Extract First
 * Frame" should not show a range, a split point and a frame index it does not use.
 *
 * Widgets only - sockets and links are never touched. A hidden widget keeps its value and still saves and still goes
 * into the prompt (the shared _widget_visibility.js helper changes how it draws, not what it holds).
 */
import { app } from "../../scripts/app.js";
import { setWidgetVisible, vueSyncNodeWidgets } from "./_widget_visibility.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const CLEAN = ["balance_mode", "balance_strength", "saturation", "contrast_restore", "chroma_cleanup",
               "restore_unchanged", "restore_radius"];
// Layer Effects: each effect's controls carry its name as a prefix and show while the effect is switched on
const EFFECTS = ["drop_shadow", "outer_glow", "gradient_map", "gradient_overlay", "color_overlay", "inner_glow",
                 "inner_shadow", "stroke"];
const effectOf = (name) => EFFECTS.find((k) => name.startsWith(k + "_")) || null;
// Folder Version Incrementer: the pipeline layout's controls, and the source / date layout's
const FI_PIPELINE = ["show", "shot", "task", "min_version", "max_retries", "write_manifest"];
const FI_SOURCE = ["prefix", "suffix", "suffix_mode", "label", "date_format", "source_choice", "name_format",
                   "numbered_still_mode", "source_filename", "source_path", "custom_name", "folder_name_override",
                   "version_group"];

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
    FolderIncrementer: {
        // layout "show / shot / task / version" (the former Batch Version Manager) uses none of the source / date
        // controls; the default layout uses none of the pipeline ones. source_extension is the node's own (hidden).
        watch: ["layout"],
        managed: [...FI_PIPELINE, ...FI_SOURCE],
        visible: (v) => (String(v.layout ?? "").startsWith("show") ? FI_PIPELINE : FI_SOURCE),
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
    for (const w of widgets) {
        if (!managed(w.name)) continue;
        const want = show.has(w.name);
        if (want === (w.__mec_hidden === true)) {
            setWidgetVisible(w, want);
            changed = true;
        }
    }
    if (!changed) return;
    vueSyncNodeWidgets(node);
    const sz = node.computeSize?.();
    if (sz) node.setSize?.([Math.max(node.size?.[0] ?? 0, sz[0]), sz[1]]);
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
