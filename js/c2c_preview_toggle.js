/**
 * c2c_preview_toggle.js — enable/disable the NATIVE sampler latent preview.
 *
 * There is intentionally NO custom preview overlay (that overlapped core and
 * was removed). This only drives ComfyUI's OWN preview method via the backend
 * route POST /c2c/preview_method (registered by nodes/_c2c_preview_guard.py),
 * so the native in-node preview on any sampler turns on/off — no drawing, no
 * core damage. The choice is applied on load and persists across restarts.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SETTING_ID = "c2c.preview.method";

async function applyMethod(method) {
    try {
        await api.fetchApi("/c2c/preview_method", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ method }),
        });
    } catch (_) { /* server route missing / older ComfyUI — native default applies */ }
}

app.registerExtension({
    name: "C2C.PreviewToggle",
    settings: [
        {
            id: SETTING_ID,
            name: "C2C ▸ Sampler latent preview",
            tooltip:
                "Show the live denoising preview inside the sampler node (native " +
                "ComfyUI preview). Auto = Latent2RGB (fast, no model). TAESD = sharper " +
                "(needs a models/vae_approx decoder). Off = no preview.",
            type: "combo",
            options: [
                { text: "Auto (Latent2RGB — fast, no model)", value: "auto" },
                { text: "TAESD (sharper, needs vae_approx model)", value: "taesd" },
                { text: "Off", value: "off" },
            ],
            defaultValue: "auto",
            onChange: (v) => { applyMethod(v || "auto"); },
        },
    ],
    async setup() {
        // Push the saved choice to the server on load so it persists across restarts.
        const v = app.ui?.settings?.getSettingValue?.(SETTING_ID, "auto") || "auto";
        applyMethod(v);
    },
});
