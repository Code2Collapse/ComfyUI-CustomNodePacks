/**
 * Inpaint Crop Pro: the preview follows a changed crop setting without running the graph (L7.99, owner A11).
 *
 * After a run the node shows the crop that ran. When a crop setting changes, this asks the server
 * (POST /c2c/inpaint_crop/plan) to redo the crop's geometry for the last run's masks with the new values - the
 * node's own functions, not a copy - and puts the returned picture in the node's preview through core's node-output
 * store (the path core takes when a node finishes). If the server cannot answer exactly (no run yet, another preview
 * frame, a clip too big to keep), the last run's picture stays. Root-graph nodes only.
 *
 * L7.65 P08: crop_mode and the smart_* / prepare_* controls are crop settings too - every widget value is sent and
 * the server keeps the ones it knows, so the preview follows a mode change with the mode's own geometry.
 *
 * Setting: C2C › Canvas › Inpaint crop › Live crop preview (off = the preview changes only when the node runs).
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const NODE = "InpaintCropProMEC";
const SETTING = "c2c.inpaintCrop.livePreview";
const DEBOUNCE_MS = 300;
let _enabled = true;

function nodeOutputStore() {
    const vueApp = app.vueApp ?? document.querySelector("#vue-app")?.__vue_app__;
    const store = vueApp?.config?.globalProperties?.$pinia?._s?.get?.("nodeOutput");
    return typeof store?.setNodeOutputsByExecutionId === "function" ? store : null;
}

function values(node) {
    const out = {};
    for (const w of node.widgets || []) {
        if (!w?.name || w.name.startsWith("_") || w.serialize === false || typeof w.value === "object") continue;
        out[w.name] = w.value;
    }
    return out;
}

async function refresh(node) {
    if (!_enabled || !node.graph || node.graph !== app.graph) return;      // subgraph nodes: run to refresh
    let body;
    try {
        const r = await api.fetchApi("/c2c/inpaint_crop/plan", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ node_id: String(node.id), values: values(node) }),
        });
        if (!r.ok) return;
        body = await r.json();
    } catch (_e) {
        return;                                                             // offline / old server: run to refresh
    }
    if (!body?.images?.length) return;                                      // stale: the last run's picture stays
    const store = nodeOutputStore();
    if (!store) return;
    store.setNodeOutputsByExecutionId(String(node.id), { images: body.images, c2c_crop_plan: [body.plan] });
    node.setDirtyCanvas?.(true, true);
}

function hook(node) {
    if (node.__c2cCropLive) return;
    node.__c2cCropLive = true;
    let timer = null;
    const schedule = () => {
        clearTimeout(timer);
        timer = setTimeout(() => refresh(node).catch((e) => __c2cReport("c2c_inpaint_crop_live", e)), DEBOUNCE_MS);
    };
    for (const w of node.widgets || []) {
        if (!w?.name || w.name.startsWith("_") || w.serialize === false) continue;
        const cb = w.callback;
        w.callback = function (...a) {
            const r = cb?.apply(this, a);
            schedule();
            return r;
        };
    }
    const removed = node.onRemoved;
    node.onRemoved = function (...a) {
        clearTimeout(timer);
        return removed?.apply(this, a);
    };
}

if (!globalThis.__c2cInpaintCropLiveExt) {
    globalThis.__c2cInpaintCropLiveExt = true;
    app.registerExtension({
        name: "C2C.InpaintCropLive",
        settings: [{
            id: SETTING,
            name: "Live crop preview",
            tooltip: "Inpaint Crop Pro's preview follows a changed crop setting right away (from the last run's "
                + "masks). Off = the preview changes only when the node runs.",
            type: "boolean",
            defaultValue: true,
            category: ["c2c", "Canvas", "Inpaint crop"],
            onChange: (v) => { _enabled = v !== false; },
        }],
        beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData?.name !== NODE) return;
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...a) {
                const r = created?.apply(this, a);
                queueMicrotask(() => { try { hook(this); } catch (e) { __c2cReport("c2c_inpaint_crop_live", e); } });
                return r;
            };
        },
    });
}
