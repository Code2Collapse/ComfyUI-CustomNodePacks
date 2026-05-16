// c2c_model_swap.js — Quick-Swap Model Palette (C2C v2.0 §6.4)
// ---------------------------------------------------------------------
// What it does:
//   • Ctrl+M opens a centered palette listing every model-loader widget
//     in the current workflow (CheckpointLoader*, LoraLoader*, VAELoader,
//     ControlNetLoader, UNETLoader, CLIPLoader, IPAdapter*, etc.).
//   • Fuzzy-filter input; up/down to navigate; Enter to "swap" — opens a
//     submenu listing every available choice for that combo (queried
//     from the widget's own .options.values, no /object_info round-trip
//     required) with a fuzzy filter and instant preview of where in the
//     graph it would change.
//   • Enter commits the swap, pulses the node, closes the palette.
//   • Esc closes without changes.
//   • Settings: c2c.modelSwap.enabled, c2c.modelSwap.show_loras_only.
//
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";

const SETTING_ENABLED = "c2c.modelSwap.enabled";
const SETTING_LORAS_ONLY = "c2c.modelSwap.lorasOnly";

const LOADER_PATTERNS = [
    /Checkpoint/i, /Loader/i, /Lora/i, /VAE/i, /ControlNet/i, /CLIP/i,
    /UNET/i, /Diffuser/i, /IPAdapter/i, /InsightFace/i, /Upscale/i,
    /Style/i, /TextEncoder/i, /Hypernetwork/i,
];
const COMBO_WIDGET_TYPES = new Set(["combo", "COMBO"]);

let _enabled = true;
let _lorasOnly = false;

// Fuzzy: every chr in needle appears in order in haystack (case-insensitive).
function fuzzy(hay, needle) {
    if (!needle) return true;
    hay = String(hay).toLowerCase();
    needle = String(needle).toLowerCase();
    let j = 0;
    for (let i = 0; i < hay.length && j < needle.length; i++) {
        if (hay[i] === needle[j]) j++;
    }
    return j === needle.length;
}

function fuzzyScore(hay, needle) {
    if (!needle) return 1;
    hay = String(hay).toLowerCase(); needle = String(needle).toLowerCase();
    let j = 0, score = 0, last = -2;
    for (let i = 0; i < hay.length && j < needle.length; i++) {
        if (hay[i] === needle[j]) {
            score += (i - last === 1) ? 3 : 1;
            last = i; j++;
        }
    }
    if (j < needle.length) return 0;
    return score / hay.length;
}

function collectLoaderWidgets() {
    if (!app.graph) return [];
    const out = [];
    for (const n of app.graph._nodes || []) {
        const type = n.type || "";
        const isLoader = LOADER_PATTERNS.some(rx => rx.test(type));
        if (!isLoader) continue;
        if (_lorasOnly && !/lora/i.test(type)) continue;
        for (const w of (n.widgets || [])) {
            const isCombo = COMBO_WIDGET_TYPES.has(w.type) ||
                (w.options && Array.isArray(w.options.values));
            if (!isCombo) continue;
            if (!w.options || !Array.isArray(w.options.values)) continue;
            // Filter combos that aren't model files (skip mode toggles).
            if (w.options.values.length < 2) continue;
            const looksLikeFile = w.options.values.some(v =>
                typeof v === "string" && /\.(safetensors|ckpt|pt|pth|onnx|bin)$/i.test(v));
            if (!looksLikeFile && !/lora|model|name|checkpoint|vae|clip|unet|encoder/i.test(w.name)) continue;
            out.push({ node: n, widget: w });
        }
    }
    return out;
}

// ── Palette UI ─────────────────────────────────────────────────────────
let _root = null;

function close() {
    if (!_root) return;
    _root.remove();
    _root = null;
    document.removeEventListener("keydown", onGlobalKey, true);
}

function onGlobalKey(e) {
    if (e.key === "Escape") { e.preventDefault(); close(); }
}

function open() {
    if (!_enabled) return;
    if (_root) { close(); return; }
    const widgets = collectLoaderWidgets();
    _root = document.createElement("div");
    _root.style.cssText = `
        position: fixed; left: 50%; top: 18%; transform: translateX(-50%);
        z-index: 12000; width: 640px; max-height: 64vh; overflow: hidden;
        background: #161a22; color: #e5ecf5;
        border: 1px solid rgba(255,255,255,0.14); border-radius: 8px;
        font: 12px ui-sans-serif, system-ui, sans-serif;
        box-shadow: 0 18px 48px rgba(0,0,0,0.7);
        display:flex; flex-direction:column;
    `;
    const header = document.createElement("div");
    header.style.cssText = "padding:8px 12px; font-weight:600; border-bottom:1px solid rgba(255,255,255,0.08); color:#cfe0ff;";
    header.textContent = "Quick-Swap Model — pick a loader";
    _root.appendChild(header);
    const input = document.createElement("input");
    input.type = "text"; input.placeholder = "Filter loaders…";
    input.style.cssText = "padding:8px 12px; border:none; outline:none; background:rgba(255,255,255,0.04); color:#e5ecf5; font:12px ui-sans-serif;";
    _root.appendChild(input);
    const list = document.createElement("div");
    list.style.cssText = "overflow:auto; flex:1; max-height:50vh;";
    _root.appendChild(list);
    document.body.appendChild(_root);

    let active = 0;
    let filtered = widgets.slice();
    const render = () => {
        list.innerHTML = "";
        if (filtered.length === 0) {
            list.innerHTML = `<div style="padding:18px;color:#7a8492;text-align:center;">No loader widgets found in this workflow.</div>`;
            return;
        }
        filtered.forEach((it, i) => {
            const r = document.createElement("div");
            r.style.cssText = `padding:6px 12px; cursor:pointer; ${i === active ? "background:rgba(91,141,239,0.22);" : ""}`;
            r.innerHTML = `<span style="color:#cfe0ff;">#${it.node.id}</span> <span style="color:#8b96a5;">${escapeHtml(it.node.type)}</span> · <code>${escapeHtml(it.widget.name)}</code> = <span style="color:#ffd166;">${escapeHtml(String(it.widget.value))}</span>`;
            r.addEventListener("click", () => { active = i; openSwap(filtered[active]); });
            r.addEventListener("mouseenter", () => { active = i; render(); });
            list.appendChild(r);
        });
    };
    const refilter = () => {
        const q = input.value.trim();
        filtered = widgets
            .map(w => ({ w, s: fuzzyScore((w.node.type + " " + w.widget.name + " " + w.widget.value), q) }))
            .filter(x => x.s > 0 || !q)
            .sort((a, b) => b.s - a.s)
            .map(x => x.w);
        active = 0; render();
    };
    input.addEventListener("input", refilter);
    input.addEventListener("keydown", (e) => {
        if (e.key === "ArrowDown") { active = Math.min(filtered.length - 1, active + 1); render(); e.preventDefault(); }
        else if (e.key === "ArrowUp") { active = Math.max(0, active - 1); render(); e.preventDefault(); }
        else if (e.key === "Enter" && filtered[active]) { openSwap(filtered[active]); e.preventDefault(); }
    });
    setTimeout(() => input.focus(), 30);
    document.addEventListener("keydown", onGlobalKey, true);
    refilter();
}

function openSwap(item) {
    close();
    const values = (item.widget.options && item.widget.options.values) || [];
    _root = document.createElement("div");
    _root.style.cssText = `
        position: fixed; left: 50%; top: 18%; transform: translateX(-50%);
        z-index: 12000; width: 720px; max-height: 64vh; overflow: hidden;
        background: #161a22; color: #e5ecf5;
        border: 1px solid rgba(255,255,255,0.14); border-radius: 8px;
        font: 12px ui-sans-serif, system-ui, sans-serif;
        box-shadow: 0 18px 48px rgba(0,0,0,0.7);
        display:flex; flex-direction:column;
    `;
    const header = document.createElement("div");
    header.style.cssText = "padding:8px 12px; font-weight:600; border-bottom:1px solid rgba(255,255,255,0.08); color:#cfe0ff;";
    header.textContent = `Swap #${item.node.id} ${item.node.type} · ${item.widget.name}  (current: ${item.widget.value})`;
    _root.appendChild(header);
    const input = document.createElement("input");
    input.type = "text"; input.placeholder = `Filter ${values.length} choices…`;
    input.style.cssText = "padding:8px 12px; border:none; outline:none; background:rgba(255,255,255,0.04); color:#e5ecf5; font:12px ui-sans-serif;";
    _root.appendChild(input);
    const list = document.createElement("div");
    list.style.cssText = "overflow:auto; flex:1; max-height:50vh;";
    _root.appendChild(list);
    document.body.appendChild(_root);

    let active = 0;
    let filtered = values.slice();
    const commit = (v) => {
        try {
            item.widget.value = v;
            if (typeof item.widget.callback === "function") item.widget.callback(v);
            pulse(item.node);
        } catch (e) { console.warn("[c2c_model_swap] commit failed", e); }
        close();
    };
    const render = () => {
        list.innerHTML = "";
        filtered.slice(0, 400).forEach((v, i) => {
            const r = document.createElement("div");
            const isCur = String(v) === String(item.widget.value);
            r.style.cssText = `padding:6px 12px; cursor:pointer; ${i === active ? "background:rgba(91,141,239,0.22);" : ""} ${isCur ? "color:#ffd166;" : ""}`;
            r.innerHTML = `${isCur ? "★ " : "  "}<code>${escapeHtml(String(v))}</code>`;
            r.addEventListener("click", () => commit(v));
            r.addEventListener("mouseenter", () => { active = i; render(); });
            list.appendChild(r);
        });
    };
    const refilter = () => {
        const q = input.value.trim();
        filtered = values
            .map(v => ({ v, s: fuzzyScore(v, q) }))
            .filter(x => x.s > 0 || !q)
            .sort((a, b) => b.s - a.s)
            .map(x => x.v);
        active = 0; render();
    };
    input.addEventListener("input", refilter);
    input.addEventListener("keydown", (e) => {
        if (e.key === "ArrowDown") { active = Math.min(filtered.length - 1, active + 1); render(); e.preventDefault(); }
        else if (e.key === "ArrowUp") { active = Math.max(0, active - 1); render(); e.preventDefault(); }
        else if (e.key === "Enter" && filtered[active] !== undefined) { commit(filtered[active]); e.preventDefault(); }
    });
    setTimeout(() => input.focus(), 30);
    document.addEventListener("keydown", onGlobalKey, true);
    refilter();
}

function pulse(node) {
    if (!node) return;
    const orig = node.bgcolor;
    node.bgcolor = "#5b8def";
    app.canvas.centerOnNode(node);
    app.graph.setDirtyCanvas(true, true);
    setTimeout(() => { node.bgcolor = orig; app.graph.setDirtyCanvas(true, true); }, 700);
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]);
}

// ── Hotkey ────────────────────────────────────────────────────────────
function onKey(e) {
    if (!_enabled) return;
    // Ignore when typing in inputs (excluding our own palette input).
    const tag = (e.target && e.target.tagName) || "";
    const inOwnRoot = _root && _root.contains(e.target);
    if (!inOwnRoot && (tag === "INPUT" || tag === "TEXTAREA" || (e.target && e.target.isContentEditable))) return;
    if (e.key.toLowerCase() === "m" && (e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey) {
        e.preventDefault(); open();
    }
}

app.registerExtension({
    name: "C2C.ModelSwap",
    async setup() {
        app.ui.settings.addSetting({
            id: SETTING_ENABLED, name: "C2C ▸ Quick-Swap Models (Ctrl+M)",
            type: "boolean", defaultValue: true,
            onChange: v => { _enabled = !!v; },
        });
        app.ui.settings.addSetting({
            id: SETTING_LORAS_ONLY, name: "C2C ▸ Quick-Swap: LoRA loaders only",
            type: "boolean", defaultValue: false,
            onChange: v => { _lorasOnly = !!v; },
        });
        _enabled = app.ui.settings.getSettingValue(SETTING_ENABLED, true);
        _lorasOnly = app.ui.settings.getSettingValue(SETTING_LORAS_ONLY, false);
        window.addEventListener("keydown", onKey, true);
    },
});
