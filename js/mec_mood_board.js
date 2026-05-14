/**
 * mec_mood_board.js — Phase 17b: Workflow Mood Board
 *
 * Floating 🎨 button. Opens a side panel with a thumbnail grid of every
 * output from the current session (last 64). Click a thumbnail to view
 * full-size in a lightbox. Auto-captures from `execution_success`.
 *
 * Setting:
 *   mec.mood_board.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BTN_ID = "mec-mood-btn";
const PANEL_ID = "mec-mood-panel";
const LIGHT_ID = "mec-mood-lightbox";
const STYLE_ID = "mec-mood-style";
const MAX_THUMBS = 64;

const _items = []; // [{ ts, image }]

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${BTN_ID} {
    position: fixed;
    bottom: 340px;
    right: 16px;
    z-index: 99996;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #1e1e2e;
    border: 1px solid #45475a;
    color: #cba6f7;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
}
#${BTN_ID}:hover { border-color: #cba6f7; }
#${PANEL_ID} {
    position: fixed;
    bottom: 340px;
    right: 64px;
    z-index: 99996;
    width: 360px;
    max-height: 60vh;
    overflow-y: auto;
    background: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 10px;
    color: #cdd6f4;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.65);
    display: none;
}
#${PANEL_ID}.visible { display: block; }
#${PANEL_ID} h3 { margin: 0 0 6px 0; color: #cba6f7; font-size: 13px; display: flex; justify-content: space-between; }
#${PANEL_ID} .mb-close { background: none; border: none; color: #6c7086; cursor: pointer; font-size: 14px; }
#${PANEL_ID} .mb-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; }
#${PANEL_ID} .mb-grid img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 4px;
    cursor: pointer;
    transition: transform 0.1s;
    border: 1px solid transparent;
}
#${PANEL_ID} .mb-grid img:hover { border-color: #cba6f7; transform: scale(1.04); }
#${PANEL_ID} .mb-empty { color: #6c7086; text-align: center; padding: 16px 0; font-style: italic; }
#${LIGHT_ID} {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.9);
    z-index: 99998;
    display: none;
    align-items: center;
    justify-content: center;
    cursor: zoom-out;
}
#${LIGHT_ID}.visible { display: flex; }
#${LIGHT_ID} img { max-width: 92vw; max-height: 92vh; box-shadow: 0 0 40px rgba(0,0,0,0.8); }
    `.trim();
    document.head.appendChild(style);
}

function _urlFor(img) {
    if (!img?.filename) return null;
    const params = new URLSearchParams({
        filename: img.filename,
        subfolder: img.subfolder || "",
        type: img.type || "output",
    });
    return `${api.api_base || ""}/view?${params}`;
}

function _ensureUi() {
    if (!document.getElementById(BTN_ID)) {
        const b = document.createElement("button");
        b.id = BTN_ID;
        b.title = "Mood board";
        b.textContent = "🎨";
        b.addEventListener("click", _toggle);
        document.body.appendChild(b);
    }
    if (!document.getElementById(PANEL_ID)) {
        const p = document.createElement("div");
        p.id = PANEL_ID;
        document.body.appendChild(p);
    }
    if (!document.getElementById(LIGHT_ID)) {
        const l = document.createElement("div");
        l.id = LIGHT_ID;
        l.innerHTML = `<img>`;
        l.addEventListener("click", () => l.classList.remove("visible"));
        document.body.appendChild(l);
    }
}

let _open = false;
function _toggle() {
    _open = !_open;
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    if (_open) { p.classList.add("visible"); _render(); }
    else       { p.classList.remove("visible"); }
}

function _render() {
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    if (_items.length === 0) {
        p.innerHTML = `
            <h3>🎨 Mood Board <button class="mb-close">×</button></h3>
            <div class="mb-empty">No outputs in this session yet — queue something!</div>
        `;
        p.querySelector(".mb-close").addEventListener("click", _toggle);
        return;
    }
    const reversed = _items.slice().reverse();
    const thumbs = reversed.map((it, i) => {
        const url = _urlFor(it.image);
        return url ? `<img src="${url}" data-i="${i}" loading="lazy" title="${new Date(it.ts).toLocaleString()}">` : "";
    }).join("");
    p.innerHTML = `
        <h3>🎨 Mood Board <button class="mb-close">×</button></h3>
        <div class="mb-grid">${thumbs}</div>
        <div style="margin-top:8px;font-size:10px;color:#6c7086;text-align:center;">${_items.length} / ${MAX_THUMBS} images</div>
    `;
    p.querySelector(".mb-close").addEventListener("click", _toggle);
    p.querySelectorAll(".mb-grid img").forEach(img => {
        img.addEventListener("click", () => {
            const l = document.getElementById(LIGHT_ID);
            l.querySelector("img").src = img.src;
            l.classList.add("visible");
        });
    });
}

function _onSuccess(ev) {
    const prompt_id = ev?.detail?.prompt_id;
    if (!prompt_id) return;
    api.fetchApi(`/history/${prompt_id}`).then(async (resp) => {
        const data = await resp.json();
        const entry = data?.[prompt_id];
        if (!entry?.outputs) return;
        for (const nid of Object.keys(entry.outputs)) {
            const o = entry.outputs[nid];
            if (Array.isArray(o?.images)) {
                for (const img of o.images) _items.push({ ts: Date.now(), image: img });
            }
        }
        while (_items.length > MAX_THUMBS) _items.shift();
        if (_open) _render();
    }).catch(() => { /* ignore */ });
}

app.registerExtension({
    name: "MEC.MoodBoard",
    settings: [
        {
            id: "mec.mood_board.enabled",
            name: "Mood Board: session output gallery",
            type: "boolean",
            defaultValue: true,
            onChange: (v) => {
                const b = document.getElementById(BTN_ID);
                if (b) b.style.display = v ? "flex" : "none";
            },
        },
    ],
    async setup() {
        _injectStyle();
        _ensureUi();
        api.addEventListener("execution_success", _onSuccess);
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("mec.mood_board.enabled", true); }
            catch { return true; }
        })();
        document.getElementById(BTN_ID).style.display = enabled ? "flex" : "none";
        console.log("[MEC.MoodBoard] Loaded.");
    },
});
