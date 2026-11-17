// AVHandlesMEC: repair workflows saved with the old widget layout, and show the run report on the node (A9: "not
// calculating frames, not adding or trimming at all, no control").
//
// WHICH widgets show (A28, ORDERS 29) is declared next to the inputs in nodes/av_handles.py and applied by
// js/_c2c_node_layout.js, the node's one visibility owner:
//   fit  — target_frames, model (+ custom_step / custom_plus for the custom grid), side, More: manual_fps, memo_key
//   add  — handle_frames, padding_mode (+ model, and custom_step / custom_plus for the custom grid, when
//          padding_mode is "model preset" - L2.55), side
//   trim — trim_amount (+ handle_frames when manual), side
import { app } from "../../scripts/app.js";
import { relayoutNode } from "./_c2c_node_layout.js";

const NODE = "AVHandlesMEC";

// Saved before 2026-09-22 (commit 4aad79d added target_frames, model, custom_step, custom_plus in the middle):
// ComfyUI restores widget values by POSITION, so those saves loaded the old side into target_frames, and so on.
const LEGACY_ORDER = ["mode", "handle_frames", "trim_amount", "side", "padding_mode", "manual_fps", "memo_key"];

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

function paintReport(el) {
    const warn = /NOTHING WAS TRIMMED|is NOT on the/.test(el.textContent);
    el.style.color = warn ? "var(--c2c-warning, #f5c26b)" : "var(--c2c-fg, #ccc)";
}

function ensureReport(node) {
    if (node.__avhReport) return node.__avhReport;
    const el = document.createElement("div");
    el.style.cssText = "font:11px/1.35 ui-monospace,monospace;white-space:pre-wrap;overflow:auto;max-height:96px;"
        + "padding:4px 6px;border-radius:4px;background:var(--c2c-bg3, rgba(0,0,0,.25));color:var(--c2c-fg, #ccc);";
    // the last report rides in node.properties: ComfyUI does not re-send a cached node's report (only output
    // nodes get emit_cached_output), so a reopened workflow would otherwise show nothing until the node re-runs
    el.textContent = node.properties?.avh_report || "Run once to see what was added or trimmed.";
    paintReport(el);
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
            ensureReport(this);
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
                if (this.__avhReport && this.properties?.avh_report) {
                    this.__avhReport.textContent = this.properties.avh_report;
                    paintReport(this.__avhReport);
                }
                relayoutNode(this);                         // the repair set values without callbacks
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
                paintReport(el);
                if (this.properties) this.properties.avh_report = el.textContent;
            }
            return r;
        };
    },
});
