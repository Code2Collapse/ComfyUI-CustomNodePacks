// C2C Distributed: show the GPU inventory / sequence-parallel plan on the node after a run (L9.38).
// Same report pattern as AV Handles: a DOM widget, the last report kept in node.properties so a reopened workflow
// (or a cached rerun, which sends no new report) still shows it. Works in the classic canvas and Nodes 2.0.
import { app } from "../../scripts/app.js";

const NODES = {
    C2CGPUInventory: { key: "c2c_dist_report", empty: "Run to list the GPUs and the multi-GPU modes this machine can run.", height: 240 },
    C2CSequenceParallel: { key: "c2c_dist_report", empty: "Run once to see the GPU plan (Ulysses x Ring).", height: 84 },
};

function paint(el) {
    const warn = /\[no \]|not usable|nothing to split|unsupported/.test(el.textContent);
    el.style.borderLeft = `3px solid ${warn ? "var(--c2c-warning, #f5c26b)" : "var(--c2c-accent, #5fb3a1)"}`;
}

function ensureReport(node, spec) {
    if (node.__c2cDistReport) return node.__c2cDistReport;
    const el = document.createElement("div");
    el.style.cssText = "font:11px/1.35 ui-monospace,monospace;white-space:pre-wrap;overflow:auto;"
        + `max-height:${spec.height}px;padding:4px 6px;border-radius:4px;`
        + "background:var(--c2c-bg3, rgba(0,0,0,.25));color:var(--c2c-fg, #ccc);";
    el.textContent = node.properties?.[spec.key] || spec.empty;
    paint(el);
    node.addDOMWidget(spec.key, "report", el, { serialize: false, hideOnZoom: false, getHeight: () => spec.height });
    node.__c2cDistReport = el;
    return el;
}

app.registerExtension({
    name: "C2C.Distributed.Report",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const spec = NODES[nodeData.name];
        if (!spec || nodeType.prototype.__c2cDistHooked) return;
        nodeType.prototype.__c2cDistHooked = true;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            ensureReport(this, spec);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            const el = ensureReport(this, spec);
            if (this.properties?.[spec.key]) {
                el.textContent = this.properties[spec.key];
                paint(el);
            }
            return r;
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            const text = message?.text;
            if (text == null) return r;
            const el = ensureReport(this, spec);
            el.textContent = Array.isArray(text) ? text.join("\n") : String(text);
            paint(el);
            this.properties = this.properties || {};
            this.properties[spec.key] = el.textContent;
            return r;
        };
    },
});
