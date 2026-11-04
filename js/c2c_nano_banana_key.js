// C2C > Nano Banana: keep the Gemini API key out of workflows and image metadata (A22 L9.135).
//
// The node keeps its `api_key` input so old saves load unchanged (same name, same position). A key typed into it,
// or found in a loaded workflow, is moved into the C2C key store (POST /c2c/ai/keys/set -> OS keychain, name
// GEMINI_API_KEY) and the widget is cleared. The backend reads the key store when the widget is empty
// (nodes/nano_banana.py _resolve_key). If the store cannot take it, the key stays and the user is told plainly that
// it will be saved with the workflow.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "NanoBananaC2C";
const KEY_NAME = "GEMINI_API_KEY";

function toast(severity, detail) {
    try {
        app.extensionManager?.toast?.add?.({ severity, summary: "Nano Banana", detail, life: 7000 });
    } catch { /* best-effort */ }
}

async function moveKeyToStore(node, widget) {
    const value = String(widget?.value ?? "").trim();
    if (!value || widget._c2cStoring) return;
    widget._c2cStoring = true;
    try {
        const r = await api.fetchApi("/c2c/ai/keys/set", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: KEY_NAME, value }),
        });
        const j = await r.json().catch(() => null);
        if (!r.ok || !j?.success) throw new Error(j?.message || `HTTP ${r.status}`);
        if (String(widget.value ?? "").trim() === value) widget.value = "";
        node.setDirtyCanvas?.(true, true);
        toast("success", "Gemini key moved to the C2C key store (Settings > C2C AI). It is no longer saved in this "
            + "workflow; save the workflow again to remove it from the file.");
    } catch (e) {
        toast("warn", "Could not move the Gemini key to the C2C key store (" + (e?.message || e) + "). It stays in "
            + "the api_key widget and WILL be saved in the workflow and in image metadata. Set the GEMINI_API_KEY "
            + "environment variable instead to keep it out of files.");
    } finally {
        widget._c2cStoring = false;
    }
}

function keyWidget(node) {
    return node.widgets?.find((w) => w.name === "api_key") || null;
}

app.registerExtension({
    name: "C2C.NanoBananaKey",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const r = onCreated?.apply(this, args);
            const w = keyWidget(this);
            if (w && !w._c2cKeyHooked) {
                w._c2cKeyHooked = true;
                const cb = w.callback;
                const node = this;
                w.callback = function (...a) {
                    const out = cb?.apply(this, a);
                    void moveKeyToStore(node, w);
                    return out;
                };
            }
            return r;
        };

        // Values are restored before onConfigure (fe 1.53), so a key saved by an older version is visible here.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const r = onConfigure?.apply(this, args);
            const w = keyWidget(this);
            if (w && String(w.value ?? "").trim()) void moveKeyToStore(this, w);
            return r;
        };
    },
});
