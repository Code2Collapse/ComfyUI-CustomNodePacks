/**
 * The Matte viewer in QC mode on Mask Refiner (MaskRefineMEC) and Mask Track (MaskTrackerMEC) (L7.124, L7.114
 * slice C).
 *
 * The same panel as on Mask + Matting (js/c2c_viewer/viewer.js), created with mode "qc": no Subject step, no helper
 * prompts. Each run of the node sends its plate and its matte as proxy frames plus the QC numbers in its own ui output
 * (key "c2c_viewer", nodes/c2c_viewer/result.py node_payload); the panel shows Plate -> Render, scrubbable, with the
 * comp / alpha / checker / overlay views, the wipe and the QC chips.
 *
 * The panel is mounted on the first run that sends a payload, so with the kill switch (C2C_VIEWER=0 or the C2C
 * setting 'Matte viewer' off) the server sends none and the node stays exactly as it was. The panel is a
 * serialize:false DOM widget: saved widgets_values never change. Mask Track hides it in bbox_smooth (no matte).
 *
 * License: Apache-2.0
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { setWidgetVisible, vueFitNodeHeight, vueSyncNodeWidgets } from "./_widget_visibility.js";
import { afterRender, rebindNode, tagDomWidget } from "./_vue_dom_rebind.js";
import { createViewer } from "./c2c_viewer/viewer.js";

const NODES = {
    MaskRefineMEC: { empty: "Run the workflow: the plate and the refined alpha show here, with QC." },
    MaskTrackerMEC: { empty: "Run the workflow: the video and the tracked masks show here, with QC.",
        hiddenModes: new Set(["bbox_smooth"]) },
};
const KEY = "c2c_viewer";
const ST = "__c2cQcViewer";
const MIN_W = 420;

function fit(node) {
    try {
        vueSyncNodeWidgets(node);
        const sz = node.computeSize();
        node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
        vueFitNodeHeight(node);
        node.setDirtyCanvas?.(true, true);
    } catch (_e) { /* never break the graph */ }
}

function modeHidden(node) {
    const hidden = NODES[node.comfyClass]?.hiddenModes;
    if (!hidden) return false;
    return hidden.has(String((node.widgets || []).find((w) => w.name === "mode")?.value ?? ""));
}

function showForMode(node) {
    const st = node[ST];
    if (!st) return;
    const on = !modeHidden(node);
    if (st.visible === on) return;
    st.visible = on;
    setWidgetVisible(st.widget, on);
    fit(node);
}

function mount(node) {
    if (node[ST]) return node[ST];
    const v = createViewer(node, {
        app, api, mode: "qc",
        emptyText: NODES[node.comfyClass]?.empty,
        onLayout: () => requestAnimationFrame(() => fit(node)),
    });
    const nodeWidth = () => node.size?.[0] || MIN_W;
    const widget = node.addDOMWidget(KEY, "div", v.root, {
        serialize: false,
        margin: 4,
        getMinHeight: () => v.heightFor(nodeWidth()) + 8,
    });
    widget.serialize = false;                          // display only: never in widgets_values
    widget.computeSize = (w) => [w, v.heightFor(Math.max(w || 0, nodeWidth())) + 8];
    tagDomWidget(widget);                              // Nodes 2.0 inside a new subgraph: _vue_dom_rebind swaps ours in
    const st = { viewer: v, widget, visible: true };
    node[ST] = st;
    if ((node.size?.[0] || 0) < MIN_W) node.setSize([MIN_W, node.size[1]]);
    // "Convert to subgraph" removes the node and adds the SAME node inside: release only when it stays removed
    let teardown = null;
    const origRemoved = node.onRemoved;
    node.onRemoved = function (...a) {
        if (teardown) clearTimeout(teardown);
        teardown = setTimeout(() => { teardown = null; if (!node.graph) v.destroy(); }, 2000);
        return origRemoved?.apply(this, a);
    };
    const origAdded = node.onAdded;
    node.onAdded = function (...a) {
        const r = origAdded?.apply(this, a);
        if (teardown) { clearTimeout(teardown); teardown = null; }
        afterRender(() => rebindNode(node));
        return r;
    };
    st.visible = null;
    showForMode(node);
    afterRender(() => fit(node));
    return st;
}

app.registerExtension({
    name: "C2C.MaskQcViewer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!Object.prototype.hasOwnProperty.call(NODES, nodeData?.name)) return;
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const r = executed?.apply(this, arguments);
            try {
                const list = output?.[KEY];
                const payload = Array.isArray(list) ? list[list.length - 1] : null;
                if (payload?.kind === "result") mount(this).viewer.showResult(payload);
            } catch (e) { console.warn("[C2C] QC viewer:", e); }
            return r;
        };
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created?.apply(this, arguments);
            const node = this;
            const w = (node.widgets || []).find((x) => x.name === "mode");
            if (w && NODES[nodeData.name].hiddenModes) {
                const cb = w.callback;
                w.callback = function (...a) { const out = cb?.apply(this, a); showForMode(node); return out; };
            }
            return r;
        };
    },
});
