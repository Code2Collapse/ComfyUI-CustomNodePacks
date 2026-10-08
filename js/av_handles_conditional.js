// AVHandlesMEC: show only the widgets the chosen mode uses, repair workflows saved with the old widget layout,
// and show the run report on the node (A9: "not calculating frames, not adding or trimming at all, no control").
//
//   fit  — target_frames, model (+ custom_step / custom_plus for the custom grid), side, manual_fps, memo_key
//   add  — handle_frames, padding_mode, side, manual_fps, memo_key
//   trim — trim_amount (+ handle_frames when manual), side, manual_fps, memo_key
import { app } from "../../scripts/app.js";
import { setHidden, vueSyncNodeWidgets } from "./_widget_visibility.js";

const NODE = "AVHandlesMEC";
const CUSTOM = "custom (step / plus)";
const ALWAYS = new Set(["mode", "side", "manual_fps", "memo_key"]);

// Saved before 2026-09-22 (commit 4aad79d added target_frames, model, custom_step, custom_plus in the middle):
// ComfyUI restores widget values by POSITION, so those saves loaded the old side into target_frames, and so on.
const LEGACY_ORDER = ["mode", "handle_frames", "trim_amount", "side", "padding_mode", "manual_fps", "memo_key"];

function shown(node, name) {
    const get = (n) => node.widgets?.find((w) => w.name === n)?.value;
    const mode = get("mode");
    if (ALWAYS.has(name)) return true;
    if (mode === "fit") {
        if (name === "target_frames" || name === "model") return true;
        if (name === "custom_step" || name === "custom_plus") return get("model") === CUSTOM;
        return false;
    }
    if (mode === "add") return name === "handle_frames" || name === "padding_mode";
    if (mode === "trim") return name === "trim_amount" || (name === "handle_frames" && get("trim_amount") === "manual");
    return true;
}

function applyVisibility(node) {
    for (const w of node.widgets || []) {
        if (!w?.name || w.name === "avh_report") continue;
        if (w.type === "button" || w.options?.serialize === false) continue;
        setHidden(w, !shown(node, w.name));
    }
    vueSyncNodeWidgets(node);
    const sz = node.computeSize?.();
    if (sz) node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.setDirtyCanvas?.(true, true);
}

function hook(node, name) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w || w.__avhHooked) return;
    w.__avhHooked = true;
    const orig = w.callback;
    w.callback = function (v, ...rest) {
        const r = orig?.call(this, v, ...rest);
        applyVisibility(node);
        return r;
    };
}

function repairLegacyValues(node, info) {
    const v = info?.widgets_values;
    if (!Array.isArray(v) || v.length !== LEGACY_ORDER.length) return false;
    if (!["fit", "add", "trim"].includes(v[0]) || !["head", "tail", "both"].includes(v[3])) return false;
    const defaults = { target_frames: 0, custom_step: 4, custom_plus: 1 };
    for (const [name, value] of Object.entries(defaults)) {
        const w = node.widgets?.find((x) => x.name === name);
        if (w) w.value = value;
    }
    const model = node.widgets?.find((x) => x.name === "model");
    if (model?.options?.values?.length) model.value = model.options.values.includes(model.value) ? model.value : model.options.values[0];
    LEGACY_ORDER.forEach((name, i) => {
        const w = node.widgets?.find((x) => x.name === name);
        if (w) w.value = v[i];
    });
    console.info("[C2C AV Handles] repaired a workflow saved with the old widget layout (node %s)", node.id);
    return true;
}

function ensureReport(node) {
    if (node.__avhReport) return node.__avhReport;
    const el = document.createElement("div");
    el.style.cssText = "font:11px/1.35 ui-monospace,monospace;white-space:pre-wrap;overflow:auto;max-height:96px;"
        + "padding:4px 6px;border-radius:4px;background:var(--c2c-bg3, rgba(0,0,0,.25));color:var(--c2c-fg, #ccc);";
    el.textContent = "Run once to see what was added or trimmed.";
    node.addDOMWidget("avh_report", "report", el, { serialize: false, hideOnZoom: false, getHeight: () => 84 });
    node.__avhReport = el;
    return el;
}

app.registerExtension({
    name: "C2C.AVHandles.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            for (const name of ["mode", "model", "trim_amount"]) hook(this, name);
            ensureReport(this);
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            // copy BEFORE the chain: the generic MEC migration pads short saves to the full widget count in place
            const saved = Array.isArray(info?.widgets_values) ? info.widgets_values.slice() : null;
            const r = onConfigure?.apply(this, arguments);
            // the saved values are applied by position after this hook returns: repair on the next tick
            setTimeout(() => {
                repairLegacyValues(this, { widgets_values: saved });
                applyVisibility(this);
            }, 0);
            return r;
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            const text = message?.text;
            const el = ensureReport(this);
            if (el) {
                el.textContent = Array.isArray(text) ? text.join("\n") : String(text ?? "");
                const warn = /NOTHING WAS TRIMMED|is NOT on the/.test(el.textContent);
                el.style.color = warn ? "var(--c2c-warning, #f5c26b)" : "var(--c2c-fg, #ccc)";
            }
            return r;
        };
    },
});
