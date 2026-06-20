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
                "Live denoising preview inside the sampler node — works for core " +
                "KSampler AND Kijai WanVideoSampler. 'Wan/video-aware' uses the " +
                "TAESD path, which routes Wan samplers to their own video previewer " +
                "(taehv if you drop taew2_1/taew2_2.safetensors in models/vae_approx, " +
                "otherwise a Wan-factor Latent2RGB fallback) — this is the one that " +
                "makes Wan/Kijai previews actually appear. 'Fast' is core Latent2RGB " +
                "(good for SD/Flux, blank for Wan). 'Off' = no preview.",
            type: "combo",
            options: [
                { text: "On — Wan/video-aware (recommended)", value: "taesd" },
                { text: "On — fast, SD/Flux only (Latent2RGB)", value: "latent2rgb" },
                { text: "Off", value: "off" },
            ],
            defaultValue: "taesd",
            onChange: (v) => { applyMethod(v || "taesd"); },
        },
    ],
    async setup() {
        // Push the saved choice to the server on load so it persists across restarts.
        // Default TAESD so Wan/Kijai samplers route to their own video previewer
        // (core Auto/Latent2RGB shows nothing for Wan latents).
        const v = app.ui?.settings?.getSettingValue?.(SETTING_ID, "taesd") || "taesd";
        applyMethod(v);
    },
});
