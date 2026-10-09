/**
 * Clear workflow forgets old widget values (L7.36) - a core frontend data fix, behind a setting.
 *
 * Core keeps every widget's value in a store keyed "graph id : node id : widget name". LGraph.clear() empties that
 * store only when the graph id is NOT the zero UUID (frontend 1.52.7 .. 1.57.0), and clear() itself sets the id to
 * the zero UUID - so a fresh session, and any graph after its first clear, keeps the old values. A node added after
 * "Clear workflow" reuses id 1, 2, ... and silently takes the deleted node's value for any widget with the same name
 * (measured: a KSamplerSelect added after clearing a KSampler set to dpmpp_2m came up dpmpp_2m, core nodes only, all
 * C2C front-end blocked, both renderers - docs/evidence/L7.36).
 *
 * Fix: AFTER a root graph is cleared - clear() always leaves it on the zero UUID - call the store's own
 * clearGraph(zeroUuid), so the empty graph starts with no values. Before is not enough: the workflow store gives an
 * edited graph a real UUID, core then clears that id, and the values stored under the zero UUID before the edit
 * survive (measured: the before-clear version fixed 1 of 4 cases). Nothing else changes; if the store cannot be
 * found, core behaviour stands.
 */
import { app } from "../../scripts/app.js";
import { getRuntime } from "./_c2c_runtime.js";

const SETTING_ID = "c2c.coreFix.clearStaleWidgetValues";
const ZERO_UUID = "00000000-0000-0000-0000-000000000000";
let _enabled = true;
let _warned = false;

function widgetValueStore() {
    const vueApp = app.vueApp ?? document.querySelector("#vue-app")?.__vue_app__;
    const pinia = vueApp?.config?.globalProperties?.$pinia;
    const store = pinia?._s?.get?.("widgetValue");
    return typeof store?.clearGraph === "function" ? store : null;
}

function forgetZeroGraphValues(graph) {
    if (!_enabled || !graph || graph.id !== ZERO_UUID) return;
    if (graph.isRootGraph === false) return;
    const store = widgetValueStore();
    if (!store) {
        if (!_warned) {
            _warned = true;
            console.warn("[C2C] Clear-workflow fix: ComfyUI's widget value store was not found; core behaviour kept.");
        }
        return;
    }
    store.clearGraph(ZERO_UUID);
}

function install(proto) {
    if (!proto) return false;
    getRuntime().safePatch(proto, "clear", (orig) => function (...args) {
        const result = orig.apply(this, args);
        forgetZeroGraphValues(this);
        return result;
    }, { id: "clearStaleWidgets.clear" });
    return true;
}

// At module load, not in setup(): extension setups run late and one after another, and a clear in between kept the
// old values (measured: the setup()-time patch missed 1 run in 6 when the graph was cleared right after loading).
const _installedEarly = install((window.LGraph ?? window.LiteGraph?.LGraph)?.prototype);

app.registerExtension({
    name: "C2C.ClearStaleWidgetValues",
    settings: [{
        id: SETTING_ID,
        name: "Clear workflow forgets old widget values",
        tooltip: "ComfyUI keeps the widget values of cleared nodes, and a node added after 'Clear workflow' can take "
            + "an old node's value (for example its sampler). This clears them, as ComfyUI already does for saved "
            + "workflows. Off = ComfyUI's own behaviour.",
        type: "boolean",
        defaultValue: true,
        category: ["c2c", "Canvas", "Clear workflow"],
        onChange: (v) => { _enabled = v !== false; },
    }],
    async setup() {
        _enabled = app.ui.settings.getSettingValue(SETTING_ID) !== false;
        if (!_installedEarly) install(app.graph && Object.getPrototypeOf(app.graph));
    },
});
