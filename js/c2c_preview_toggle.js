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
    // 1) Drive ComfyUI's NATIVE per-queue preview setting. This is the
    //    authoritative path on current ComfyUI (PR #11261): the value is sent as
    //    extra_data.preview_method on every prompt and overrides args.preview_method.
    //    "default" (the stock value) means "use the CLI flag" = --preview-method
    //    none = no preview, which is why nothing showed. Force a real method here.
    //    For "auto" we set core to auto too, but our backend wrapper is what
    //    makes it smart (video->TAESD, image->Auto per model — core's own Auto
    //    is blank for Wan).
    try {
        const native = (method === "off") ? "none" : method;  // auto | taesd | latent2rgb | none
        app.ui?.settings?.setSettingValue?.("Comfy.Execution.PreviewMethod", native);
    } catch (_) { /* older ComfyUI without the per-queue setting */ }
    // 2) Also hit our backend route (sets args + default_preview_method) — covers
    //    older ComfyUI without the per-queue override, and keeps both in sync.
    try {
        await api.fetchApi("/c2c/preview_method", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ method }),
        });
    } catch (_) { /* server route missing / older ComfyUI */ }
}

app.registerExtension({
    name: "C2C.PreviewToggle",
    settings: [
        {
            id: SETTING_ID,
            name: "C2C ▸ Sampler latent preview",
            tooltip:
                "Live denoising preview inside the sampler node — works for core " +
                "KSampler AND Kijai WanVideoSampler. 'Auto' picks the right previewer " +
                "per model on every run: video/Wan latents get the TAESD path (their " +
                "own video previewer — taehv if you drop taew2_1/taew2_2.safetensors " +
                "in models/vae_approx, otherwise a Wan-factor Latent2RGB fallback), " +
                "and image latents (SD/SDXL/Flux) get core Auto (TAESD when the " +
                "decoder is present, else fast Latent2RGB). Force a single method with " +
                "the other options. 'Off' = no preview.",
            type: "combo",
            options: [
                { text: "Auto — smart, per model (recommended)", value: "auto" },
                { text: "Force On — Wan/video-aware (TAESD)", value: "taesd" },
                { text: "Force On — fast, SD/Flux only (Latent2RGB)", value: "latent2rgb" },
                { text: "Off", value: "off" },
            ],
            defaultValue: "auto",
            onChange: (v) => { applyMethod(v || "auto"); },
        },
    ],
    async setup() {
        // Push the saved choice to the server on load so it persists across restarts.
        // Default Auto: the backend wrapper selects TAESD for video/Wan latents and
        // core Auto for images, per model, on every sampler callback.
        const v = app.ui?.settings?.getSettingValue?.(SETTING_ID, "auto") || "auto";
        applyMethod(v);
    },
});
