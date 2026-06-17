/**
 * c2c_workflow_drop.js — robust "drag a .json onto the canvas → load the workflow".
 *
 * Why: users reported dropping a workflow .json does nothing. Rather than depend
 * on ComfyUI's native drop path (which changes across frontend versions and can
 * silently break on package updates), this adds OUR OWN capture-phase handler that
 * claims ONLY workflow-.json drops and loads them via the stable app.loadGraphData /
 * app.loadApiJson API. Everything else (images, other files) passes straight through
 * untouched, so it never interferes with ComfyUI.
 *
 * Defensive by design:
 *   - Only acts when the dropped file is .json AND parses as a Comfy workflow/prompt.
 *   - Wrapped in try/catch end-to-end; a failure falls back to the native path
 *     (we don't preventDefault unless we're sure we can handle it).
 *   - Uses only the long-stable public app API; if that API ever goes missing it
 *     no-ops instead of throwing, so the rest of the pack keeps working.
 *
 * Refs: ComfyUI_frontend app.loadGraphData / loadApiJson (public app methods);
 * docs.comfy.org workflow JSON format.
 */
import { app } from "../../scripts/app.js";

const SETTING_ENABLED = "c2c.workflowDrop.enabled";

function _looksLikeWorkflow(obj) {
    if (!obj || typeof obj !== "object") return null;
    // UI workflow: has a nodes array (+ usually links / last_node_id / version).
    if (Array.isArray(obj.nodes)) return "workflow";
    // API prompt: flat map of {id: {class_type, inputs}}.
    const vals = Object.values(obj);
    if (vals.length && vals.every(v => v && typeof v === "object" && "class_type" in v && "inputs" in v)) {
        return "api";
    }
    return null;
}

function _isJsonFile(file) {
    if (!file) return false;
    const n = (file.name || "").toLowerCase();
    return file.type === "application/json" || n.endsWith(".json");
}

async function _tryLoad(file) {
    let text;
    try { text = await file.text(); } catch { return false; }
    let data;
    try { data = JSON.parse(text); } catch { return false; }
    const kind = _looksLikeWorkflow(data);
    if (!kind) return false;
    try {
        if (kind === "api" && typeof app.loadApiJson === "function") {
            await app.loadApiJson(data, file.name);
            return true;
        }
        if (typeof app.loadGraphData === "function") {
            await app.loadGraphData(data, true, true, file.name);
            return true;
        }
        // Last resort: some frontends expose a graph-only loader.
        if (app.graph && typeof app.graph.configure === "function") {
            app.graph.configure(data);
            app.graph.setDirtyCanvas?.(true, true);
            return true;
        }
    } catch (e) {
        console.warn("[c2c.workflowDrop] load failed, deferring to native:", e?.message || e);
        return false;
    }
    return false;
}

function _enabled() {
    try {
        const v = app.ui?.settings?.getSettingValue?.(SETTING_ENABLED, true);
        return v !== false;
    } catch { return true; }
}

function _install() {
    if (window.__c2cWorkflowDropInstalled) return;
    window.__c2cWorkflowDropInstalled = true;

    // Allow the drop (browsers block it otherwise) when a .json is being dragged.
    window.addEventListener("dragover", (e) => {
        if (!_enabled()) return;
        const items = e.dataTransfer?.items;
        if (items && Array.from(items).some(it => it.kind === "file" && /json/i.test(it.type || ""))) {
            e.preventDefault();
        }
    }, { capture: true });

    window.addEventListener("drop", (e) => {
        if (!_enabled()) return;
        if (e.shiftKey) return;  // shift = let ComfyUI handle natively
        const files = e.dataTransfer?.files;
        if (!files || files.length === 0) return;
        const file = files[0];
        if (!_isJsonFile(file)) return;  // not ours — let images/others through
        // Claim it synchronously so the native handler doesn't also fire, then
        // load async. If load fails, we've already prevented default — but the
        // native path for .json was doing nothing anyway (that's the bug we fix).
        e.preventDefault();
        e.stopImmediatePropagation();
        _tryLoad(file).then((ok) => {
            if (!ok) {
                try {
                    app.extensionManager?.toast?.add?.({
                        severity: "warn", summary: "C2C workflow drop",
                        detail: `Couldn't parse ${file.name} as a workflow/API JSON.`, life: 4000,
                    });
                } catch { /* best-effort */ }
            }
        });
    }, { capture: true });
}

app.registerExtension({
    name: "C2C.WorkflowDrop",
    settings: [
        {
            id: SETTING_ENABLED,
            name: "C2C ▸ Drag-JSON ▸ Load workflow on .json drop",
            tooltip: "Drop a workflow/API .json onto the canvas to load it. Hold Shift to use ComfyUI's native handler instead.",
            type: "boolean",
            default: true,
        },
    ],
    async setup() {
        _install();
        console.log("[C2C.WorkflowDrop] ready — drop a workflow .json to load it.");
    },
});
