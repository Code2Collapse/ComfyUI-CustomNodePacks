/**
 * mec_whats_wired.js — Phase 8: "What's Wired" Mini Map Legend
 *
 * Floating chip showing the active high-level pipeline at a glance:
 *   Model · Sampler · Steps · CFG · Resolution · VAE · LoRAs
 *
 * Scans `app.graph._nodes` for known node classes and pulls widget values.
 *
 * Setting:
 *   mec.whats_wired.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

const HUD_ID   = "mec-whats-wired";
const STYLE_ID = "mec-whats-wired-style";

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${HUD_ID} {
    position: fixed;
    top: 40px;
    right: 16px;
    z-index: 99995;
    background: #181825;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 10px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 11px;
    color: #cdd6f4;
    max-width: 280px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    display: none;
}
#${HUD_ID}.visible { display: block; }
#${HUD_ID} .ww-row { display: flex; gap: 6px; align-items: baseline; }
#${HUD_ID} .ww-key { color: #6c7086; min-width: 60px; }
#${HUD_ID} .ww-val { color: #cdd6f4; font-family: monospace; }
#${HUD_ID} .ww-loras { color: #a6e3a1; }
#${HUD_ID} .ww-empty { color: #6c7086; font-style: italic; }
#${HUD_ID} .ww-title {
    font-weight: 700;
    color: #89b4fa;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 4px;
}
    `.trim();
    document.head.appendChild(style);
}

function _ensureHud() {
    let hud = document.getElementById(HUD_ID);
    if (!hud) {
        hud = document.createElement("div");
        hud.id = HUD_ID;
        document.body.appendChild(hud);
    }
    return hud;
}

function _findWidget(node, names) {
    for (const w of (node.widgets || [])) {
        if (!w || !w.name) continue;
        for (const n of names) {
            if (w.name.toLowerCase() === n.toLowerCase()) return w;
        }
    }
    return null;
}

function _scanGraph() {
    const out = {
        model:    null,
        sampler:  null,
        scheduler:null,
        steps:    null,
        cfg:      null,
        width:    null,
        height:   null,
        vae:      null,
        loras:    [],
    };
    const g = app.graph;
    if (!g || !g._nodes) return out;
    for (const node of g._nodes) {
        const t = (node.type || node.comfyClass || "").toLowerCase();

        // Checkpoint / UNet / Diffusion model loaders
        if (/checkpoint|unetloader|diffusionloader|fluxloader|sd3loader/.test(t)) {
            const w = _findWidget(node, ["ckpt_name", "model_name", "unet_name", "model"]);
            if (w && w.value) out.model = String(w.value);
        }

        // KSampler family
        if (/^k?sampler/.test(t) || t.includes("sampler")) {
            const sw = _findWidget(node, ["sampler_name", "sampler"]);
            if (sw && sw.value) out.sampler = String(sw.value);
            const sch = _findWidget(node, ["scheduler"]);
            if (sch && sch.value) out.scheduler = String(sch.value);
            const st = _findWidget(node, ["steps"]);
            if (st && typeof st.value === "number") out.steps = st.value;
            const cfg = _findWidget(node, ["cfg", "cfg_scale", "guidance"]);
            if (cfg && typeof cfg.value === "number") out.cfg = cfg.value;
        }

        // Empty latent / image size
        if (/emptylatentimage|emptyhunyuanlatentvideo|emptysd3latentimage|emptylatentaudio/i.test(t)) {
            const wW = _findWidget(node, ["width"]);
            const wH = _findWidget(node, ["height"]);
            if (wW && typeof wW.value === "number") out.width = wW.value;
            if (wH && typeof wH.value === "number") out.height = wH.value;
        }

        // VAE
        if (/vaeloader/.test(t)) {
            const w = _findWidget(node, ["vae_name", "vae"]);
            if (w && w.value) out.vae = String(w.value);
        }

        // LoRA loaders
        if (/loraloader|lora_loader|loralightningloader/.test(t)) {
            const w = _findWidget(node, ["lora_name"]);
            const sw = _findWidget(node, ["strength_model"]);
            if (w && w.value && w.value !== "None") {
                out.loras.push({
                    name: String(w.value),
                    strength: sw && typeof sw.value === "number" ? sw.value : 1.0,
                });
            }
        }
    }
    return out;
}

function _shortName(s) {
    if (!s) return "—";
    const parts = String(s).split(/[\\\/]/);
    return parts[parts.length - 1].replace(/\.(safetensors|ckpt|pt|bin|gguf)$/i, "");
}

function _render() {
    const enabled = (() => {
        try { return app.ui.settings.getSettingValue("mec.whats_wired.enabled", true); }
        catch { return true; }
    })();
    const hud = _ensureHud();
    if (!enabled) {
        hud.classList.remove("visible");
        return;
    }

    const info = _scanGraph();
    const anything = info.model || info.sampler || info.width || info.vae || info.loras.length;
    if (!anything) {
        hud.classList.remove("visible");
        return;
    }
    hud.classList.add("visible");

    const rows = [];
    const row = (k, v) => rows.push(`<div class="ww-row"><span class="ww-key">${k}</span><span class="ww-val">${v}</span></div>`);
    if (info.model)    row("Model",    _shortName(info.model));
    if (info.vae)      row("VAE",      _shortName(info.vae));
    if (info.sampler)  row("Sampler",  info.sampler + (info.scheduler ? " · " + info.scheduler : ""));
    if (info.steps !== null || info.cfg !== null) {
        const parts = [];
        if (info.steps !== null) parts.push(`${info.steps} steps`);
        if (info.cfg !== null) parts.push(`cfg ${info.cfg}`);
        row("Sampling", parts.join(" · "));
    }
    if (info.width && info.height) row("Size", `${info.width}×${info.height}`);
    if (info.loras.length) {
        const fmt = info.loras.slice(0, 4)
            .map(l => `${_shortName(l.name)}@${l.strength.toFixed(2)}`)
            .join(", ");
        const more = info.loras.length > 4 ? ` (+${info.loras.length - 4})` : "";
        row("LoRAs", `<span class="ww-loras">${fmt}${more}</span>`);
    }

    hud.innerHTML = `<div class="ww-title">What's Wired</div>${rows.join("")}`;
}

let _scheduled = false;
function _schedule() {
    if (_scheduled) return;
    _scheduled = true;
    requestAnimationFrame(() => { _scheduled = false; _render(); });
}

function _hook() {
    const g = app.graph;
    if (!g) return;
    const wrap = (obj, prop) => {
        if (typeof obj[prop] !== "function") return;
        const orig = obj[prop];
        if (orig._mecWWWrapped) return;
        obj[prop] = function (...args) {
            const r = orig.apply(this, args);
            try { _schedule(); } catch { /* ignore */ }
            return r;
        };
        obj[prop]._mecWWWrapped = true;
    };
    wrap(g, "add");
    wrap(g, "remove");
    wrap(g, "configure");
    wrap(g, "clear");
}

app.registerExtension({
    name: "MEC.WhatsWired",
    settings: [
        {
            id: "mec.whats_wired.enabled",
            name: "What's Wired: workflow legend",
            tooltip: "Floating chip listing the active model / sampler / size / LoRAs.",
            type: "boolean",
            defaultValue: true,
            onChange: _render,
        },
    ],
    async setup() {
        _injectStyle();
        _ensureHud();
        _hook();
        _render();
        // Re-scan every 3 s in case widget values change without a graph mutation
        setInterval(_render, 3000);
        console.log("[MEC.WhatsWired] Loaded.");
    },
});
