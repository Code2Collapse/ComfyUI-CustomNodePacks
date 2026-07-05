/**
 * c2c_mood_board.js — Mood Board (god-level rebuild, 2026-05-27)
 *
 * Floating 🎨 button. Side panel with a session thumbnail grid (last 64),
 * lightbox viewer, AND a real k-means palette extractor that runs in-browser
 * over a thumbnail's pixels and surfaces dominant colors as copyable swatches.
 *
 * Features:
 *   1) Session thumbnail grid (auto-capture on execution_success)
 *   2) Lightbox viewer with palette panel
 *   3) Per-image k-means palette (k=2..8, Lab-space distance)
 *   4) Aggregated "session palette" across selected images
 *   5) Palette compare A vs B
 *   6) Copy palette as hex list / CSS vars / JSON
 *   7) Export palette PNG (swatch strip)
 *   8) AI mood description via streamAI (palette + image refs → vibe sentence)
 *   9) Body-only re-render preserving chrome
 *
 * The k-means implementation samples 4096 pixels per image (random offsets),
 * converts to Lab for perceptual clustering, runs k-means++ init + 12
 * iterations, then converts cluster centers back to sRGB hex.
 */

import { app } from "../../scripts/app.js";
import { LITE } from "./_c2c_lite.js";
import { api } from "../../scripts/api.js";
import { attachWindowChrome } from "./_c2c_window.js";
import { streamAI } from "./_c2c_ai_client.js";

const BTN_ID    = "mec-mood-btn";
const PANEL_ID  = "mec-mood-panel";
const LIGHT_ID  = "mec-mood-lightbox";
const STYLE_ID  = "mec-mood-style";
const MAX_THUMBS = 64;
const SAMPLE_PIXELS = 4096;
const KMEANS_ITERS = 12;

const _items = [];   // [{ id, ts, image, palette?, k:number }]
const _state = {
    open: false,
    view: "grid",          // "grid" | "image" | "compare"
    selected: new Set(),   // indices of _items (forward index)
    lightIdx: -1,
    k: 5,
    aiOutput: "",
    aiBusy: false,
};
const _listeners = [];
function _on(t, e, fn, opts) { t.addEventListener(e, fn, opts); _listeners.push(() => t.removeEventListener(e, fn, opts)); }
function _clear() { while (_listeners.length) { try { _listeners.pop()(); } catch {} } }

// ─────────────────────────────────────────────────────────────────────────────
// Style
// ─────────────────────────────────────────────────────────────────────────────

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
#${BTN_ID} {
    position: fixed; bottom: 340px; right: 16px;
    z-index: var(--c2c-z-hud, 1000);
    width: 38px; height: 38px; border-radius: 50%;
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface1); color: var(--c2c-mauve);
    font-size: 16px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex; align-items: center; justify-content: center;
}
#${BTN_ID}:hover { border-color: var(--c2c-mauve); }

#${PANEL_ID} {
    position: fixed; top: 80px; right: 80px;
    width: min(92vw, 560px); height: min(80vh, 720px);
    background: var(--c2c-bg2); border: 1px solid var(--c2c-surface1); border-radius: 8px;
    color: var(--c2c-fg); font-family: -apple-system, "Segoe UI", sans-serif; font-size: 12px;
    box-shadow: 0 6px 32px rgba(0,0,0,0.7);
    display: none; flex-direction: column; overflow: hidden;
}
#${PANEL_ID}.visible { display: flex; }
#${PANEL_ID} h3 {
    margin: 0; padding: 8px 12px; color: var(--c2c-mauve); font-size: 14px;
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(180deg, var(--c2c-bg) 0%, var(--c2c-bg2) 100%);
    border-bottom: 1px solid var(--c2c-surface0);
}
#${PANEL_ID} .mb-body { flex: 1 1 auto; min-height: 0; padding: 10px; overflow: auto; }
#${PANEL_ID} .mb-close { background: none; border: none; color: var(--c2c-overlay0); cursor: pointer; font-size: 16px; padding: 0 4px; }

#${PANEL_ID} .mb-toolbar {
    display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
    margin-bottom: 8px;
}
#${PANEL_ID} .mb-toolbar button, #${PANEL_ID} .mb-toolbar select {
    background: var(--c2c-surface0); border: 1px solid var(--c2c-surface1); color: var(--c2c-fg);
    border-radius: 4px; padding: 3px 8px; font-size: 11px; cursor: pointer;
}
#${PANEL_ID} .mb-toolbar button:hover, #${PANEL_ID} .mb-toolbar select:hover { border-color: var(--c2c-mauve); }
#${PANEL_ID} .mb-toolbar .mb-k-label { color: var(--c2c-overlay0); font-size: 10px; }
#${PANEL_ID} .mb-toolbar .mb-spacer { flex: 1; }

#${PANEL_ID} .mb-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; }
#${PANEL_ID} .mb-grid .mb-cell { position: relative; cursor: pointer; }
#${PANEL_ID} .mb-grid img {
    width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px;
    border: 1px solid transparent; transition: transform 0.1s;
    display: block;
}
#${PANEL_ID} .mb-grid .mb-cell:hover img { border-color: var(--c2c-mauve); transform: scale(1.04); }
#${PANEL_ID} .mb-grid .mb-cell.selected img { border-color: var(--c2c-yellow); box-shadow: 0 0 0 2px var(--c2c-yellow); }
#${PANEL_ID} .mb-grid .mb-cell .mb-sel-mark {
    position: absolute; top: 3px; right: 3px; background: var(--c2c-yellow); color: var(--c2c-bg3);
    border-radius: 50%; width: 16px; height: 16px; text-align: center;
    font-size: 11px; line-height: 16px; font-weight: 700;
}
#${PANEL_ID} .mb-empty { color: var(--c2c-overlay0); text-align: center; padding: 40px 0; font-style: italic; }

#${PANEL_ID} .mb-palette {
    margin-top: 10px; padding: 8px; background: var(--c2c-bg3); border: 1px solid var(--c2c-surface0);
    border-radius: 4px;
}
#${PANEL_ID} .mb-palette h4 { margin: 0 0 6px 0; color: var(--c2c-mauve); font-size: 11px; }
#${PANEL_ID} .mb-swatches { display: flex; gap: 4px; flex-wrap: wrap; }
#${PANEL_ID} .mb-swatch {
    width: 56px; height: 56px; border-radius: 4px; cursor: pointer; position: relative;
    border: 1px solid var(--c2c-surface0); color: var(--c2c-white); font-size: 9px; font-family: monospace;
    display: flex; align-items: flex-end; justify-content: center;
    padding: 2px; text-shadow: 0 0 3px rgba(0,0,0,0.9);
}
#${PANEL_ID} .mb-swatch:hover { border-color: var(--c2c-mauve); }
#${PANEL_ID} .mb-swatch .mb-pct {
    position: absolute; top: 2px; right: 4px; font-size: 9px;
    background: rgba(0,0,0,0.4); padding: 0 3px; border-radius: 2px;
}

#${PANEL_ID} .mb-copy-row { display: flex; gap: 4px; margin-top: 6px; font-size: 10px; }
#${PANEL_ID} .mb-copy-row button {
    background: var(--c2c-surface0); border: 1px solid var(--c2c-surface1); color: var(--c2c-fg);
    border-radius: 3px; padding: 2px 6px; cursor: pointer; font-size: 10px;
}

#${PANEL_ID} .mb-ai-box {
    margin-top: 8px; padding: 8px 10px; background: var(--c2c-bg);
    border: 1px solid var(--c2c-surface0); border-radius: 4px; font-size: 11px;
    line-height: 1.45; white-space: pre-wrap;
}
#${PANEL_ID} .mb-ai-box.empty { color: var(--c2c-overlay0); font-style: italic; }

#${PANEL_ID} .mb-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
#${PANEL_ID} .mb-compare > div { padding: 8px; background: var(--c2c-bg3); border: 1px solid var(--c2c-surface0); border-radius: 4px; }
#${PANEL_ID} .mb-compare img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 3px; }

#${LIGHT_ID} {
    position: fixed; inset: 0; background: rgba(0,0,0,0.92);
    z-index: var(--c2c-z-modal, 9999); display: none; align-items: center; justify-content: center; cursor: zoom-out;
}
#${LIGHT_ID}.visible { display: flex; }
#${LIGHT_ID} img { max-width: 90vw; max-height: 86vh; box-shadow: 0 0 40px rgba(0,0,0,0.8); }
#${LIGHT_ID} .mb-light-info {
    position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
    background: rgba(17,17,27,0.85); padding: 6px 12px; border-radius: 4px;
    color: var(--c2c-fg); font-size: 11px; font-family: monospace;
}
    `.trim();
    document.head.appendChild(s);
}

// ─────────────────────────────────────────────────────────────────────────────
// Color math (sRGB ↔ Lab) — perceptual k-means uses Lab
// ─────────────────────────────────────────────────────────────────────────────

function _srgb2lin(c) {
    c = c / 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
function _lin2srgb(c) {
    const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
    return Math.max(0, Math.min(255, Math.round(v * 255)));
}
function _rgb2lab(r, g, b) {
    const R = _srgb2lin(r), G = _srgb2lin(g), B = _srgb2lin(b);
    // D65 sRGB → XYZ
    const X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375;
    const Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750;
    const Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041;
    const Xn = 0.95047, Yn = 1.0, Zn = 1.08883;
    const f = (t) => t > 0.008856 ? Math.cbrt(t) : (7.787 * t + 16 / 116);
    const fx = f(X / Xn), fy = f(Y / Yn), fz = f(Z / Zn);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}
function _lab2rgb(L, a, b) {
    const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
    const ft = (t) => {
        const t3 = t * t * t;
        return t3 > 0.008856 ? t3 : (t - 16 / 116) / 7.787;
    };
    const Xn = 0.95047, Yn = 1.0, Zn = 1.08883;
    const X = ft(fx) * Xn, Y = ft(fy) * Yn, Z = ft(fz) * Zn;
    const R =  3.2404542 * X + -1.5371385 * Y + -0.4985314 * Z;
    const G = -0.9692660 * X +  1.8760108 * Y +  0.0415560 * Z;
    const B =  0.0556434 * X + -0.2040259 * Y +  1.0572252 * Z;
    return [_lin2srgb(R), _lin2srgb(G), _lin2srgb(B)];
}
function _hex(rgb) {
    const h = (n) => n.toString(16).padStart(2, "0");
    return `#${h(rgb[0])}${h(rgb[1])}${h(rgb[2])}`;
}
function _luma(rgb) {
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

// ─────────────────────────────────────────────────────────────────────────────
// k-means (k-means++ init, 12 iters, Lab-space)
// ─────────────────────────────────────────────────────────────────────────────

function _kmeansPalette(samples, k) {
    // samples: array of [L, a, b]
    const N = samples.length;
    if (!N) return [];
    k = Math.max(1, Math.min(k, N));
    // k-means++ init
    const centers = [samples[Math.floor(Math.random() * N)].slice()];
    while (centers.length < k) {
        const d2 = new Array(N);
        let total = 0;
        for (let i = 0; i < N; i++) {
            let best = Infinity;
            for (const c of centers) {
                const dl = samples[i][0] - c[0];
                const da = samples[i][1] - c[1];
                const db = samples[i][2] - c[2];
                const d = dl * dl + da * da + db * db;
                if (d < best) best = d;
            }
            d2[i] = best; total += best;
        }
        if (total <= 0) { centers.push(samples[Math.floor(Math.random() * N)].slice()); continue; }
        let r = Math.random() * total, idx = 0;
        for (; idx < N - 1; idx++) { r -= d2[idx]; if (r <= 0) break; }
        centers.push(samples[idx].slice());
    }
    const assign = new Int32Array(N);
    for (let iter = 0; iter < KMEANS_ITERS; iter++) {
        // assignment
        for (let i = 0; i < N; i++) {
            let best = 0, bestD = Infinity;
            for (let c = 0; c < k; c++) {
                const dl = samples[i][0] - centers[c][0];
                const da = samples[i][1] - centers[c][1];
                const db = samples[i][2] - centers[c][2];
                const d = dl * dl + da * da + db * db;
                if (d < bestD) { bestD = d; best = c; }
            }
            assign[i] = best;
        }
        // update
        const sums = Array.from({ length: k }, () => [0, 0, 0]);
        const counts = new Int32Array(k);
        for (let i = 0; i < N; i++) {
            const c = assign[i];
            sums[c][0] += samples[i][0]; sums[c][1] += samples[i][1]; sums[c][2] += samples[i][2];
            counts[c]++;
        }
        for (let c = 0; c < k; c++) {
            if (counts[c]) {
                centers[c][0] = sums[c][0] / counts[c];
                centers[c][1] = sums[c][1] / counts[c];
                centers[c][2] = sums[c][2] / counts[c];
            }
        }
    }
    // tally
    const counts = new Int32Array(k);
    for (let i = 0; i < N; i++) counts[assign[i]]++;
    const palette = centers.map((c, i) => ({
        lab: c,
        rgb: _lab2rgb(c[0], c[1], c[2]),
        hex: "",
        pct: counts[i] / N,
        count: counts[i],
    }));
    palette.forEach((p) => { p.hex = _hex(p.rgb); });
    // sort by frequency desc
    palette.sort((a, b) => b.pct - a.pct);
    return palette;
}

async function _samplePixels(imgUrl) {
    const im = await new Promise((res, rej) => {
        const i = new Image();
        i.crossOrigin = "anonymous";
        i.onload = () => res(i);
        i.onerror = rej;
        i.src = imgUrl;
    });
    const W = 256, H = Math.max(1, Math.round(256 * im.naturalHeight / Math.max(1, im.naturalWidth)));
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const ctx = c.getContext("2d");
    ctx.drawImage(im, 0, 0, W, H);
    const data = ctx.getImageData(0, 0, W, H).data;
    const total = W * H;
    const n = Math.min(SAMPLE_PIXELS, total);
    const samples = new Array(n);
    for (let i = 0; i < n; i++) {
        const idx = (Math.random() * total | 0) * 4;
        if (data[idx + 3] < 16) { i--; continue; } // skip transparent
        samples[i] = _rgb2lab(data[idx], data[idx + 1], data[idx + 2]);
    }
    return samples;
}

async function _ensurePalette(item, k) {
    if (item.palette && item.k === k) return item.palette;
    const url = _urlFor(item.image);
    if (!url) return [];
    try {
        const samples = await _samplePixels(url);
        item.palette = _kmeansPalette(samples, k);
        item.k = k;
        return item.palette;
    } catch (e) {
        console.warn("[C2C.MoodBoard] palette failed:", e);
        return [];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// URL + UI lifecycle
// ─────────────────────────────────────────────────────────────────────────────

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
        p.innerHTML = `
            <h3><span>🎨 Mood Board</span><button class="mb-close" title="Close">×</button></h3>
            <div class="mb-body" data-role="body"></div>
        `;
        document.body.appendChild(p);
        p.querySelector(".mb-close").addEventListener("click", _toggle);
        attachWindowChrome(p, { storageKey: "mood_board", headerSelector: "h3", titleSelector: "h3 > span", minW: 380, minH: 320 });
    }
    if (!document.getElementById(LIGHT_ID)) {
        const l = document.createElement("div");
        l.id = LIGHT_ID;
        l.innerHTML = `<img><div class="mb-light-info"></div>`;
        l.addEventListener("click", (e) => { if (e.target.tagName === "IMG" || e.target === l) l.classList.remove("visible"); });
        document.body.appendChild(l);
    }
}

function _toggle() {
    _state.open = !_state.open;
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    if (_state.open) { p.classList.add("visible"); _renderBody(); }
    else { p.classList.remove("visible"); _clear(); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Render
// ─────────────────────────────────────────────────────────────────────────────

function _renderBody() {
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    const body = p.querySelector('[data-role="body"]');
    if (!body) return;
    _clear();

    if (!_items.length) {
        body.innerHTML = `<div class="mb-empty">No outputs in this session yet — queue something!</div>`;
        return;
    }
    const selCount = _state.selected.size;
    body.innerHTML = `
        <div class="mb-toolbar">
            <span class="mb-k-label">k</span>
            <select data-role="k">${[2,3,4,5,6,7,8].map((n) => `<option value="${n}"${n === _state.k ? " selected" : ""}>${n}</option>`).join("")}</select>
            <button data-act="agg" ${selCount < 2 ? "disabled" : ""} title="Aggregate palette across selected">⌘ Aggregate (${selCount})</button>
            <button data-act="compare" ${selCount !== 2 ? "disabled" : ""} title="Compare exactly two selected">↔ Compare</button>
            <button data-act="clear-sel" ${!selCount ? "disabled" : ""}>Clear sel</button>
            <span class="mb-spacer"></span>
            <button data-act="ai" ${!selCount ? "disabled" : ""}>${_state.aiBusy ? "⌛ AI…" : "✨ Mood"}</button>
            <span style="color:var(--c2c-overlay0);font-size:10px;">${_items.length}/${MAX_THUMBS}</span>
        </div>
        <div class="mb-grid" data-role="grid"></div>
        <div data-role="extra"></div>
    `;
    _renderGrid(body);
    _on(body.querySelector('[data-role="k"]'), "change", (e) => {
        _state.k = parseInt(e.target.value, 10);
        // Invalidate cached palettes
        _items.forEach((it) => { it.palette = null; });
        _renderBody();
    });
    _on(body.querySelector('[data-act="clear-sel"]'), "click", () => { _state.selected.clear(); _renderBody(); });
    _on(body.querySelector('[data-act="agg"]'), "click", () => _renderAggregate(body));
    _on(body.querySelector('[data-act="compare"]'), "click", () => _renderCompare(body));
    _on(body.querySelector('[data-act="ai"]'), "click", () => _aiMood(body));
}

function _renderGrid(body) {
    const grid = body.querySelector('[data-role="grid"]');
    const reversed = _items.slice().reverse();
    grid.innerHTML = reversed.map((it, ri) => {
        const i = _items.length - 1 - ri;
        const url = _urlFor(it.image);
        if (!url) return "";
        const sel = _state.selected.has(i);
        const order = sel ? [..._state.selected].indexOf(i) + 1 : "";
        return `<div class="mb-cell ${sel ? "selected" : ""}" data-i="${i}">
            <img src="${url}" loading="lazy" title="${new Date(it.ts).toLocaleString()}">
            ${sel ? `<div class="mb-sel-mark">${order}</div>` : ""}
        </div>`;
    }).join("");
    for (const cell of grid.querySelectorAll(".mb-cell")) {
        const i = parseInt(cell.dataset.i, 10);
        _on(cell, "click", (e) => {
            if (e.shiftKey || e.ctrlKey || e.metaKey) {
                if (_state.selected.has(i)) _state.selected.delete(i);
                else _state.selected.add(i);
                _renderBody();
            } else {
                _state.lightIdx = i;
                _openLightbox(i);
            }
        });
    }
}

async function _openLightbox(i) {
    const it = _items[i];
    if (!it) return;
    const l = document.getElementById(LIGHT_ID);
    const img = l.querySelector("img");
    const info = l.querySelector(".mb-light-info");
    img.src = _urlFor(it.image);
    info.textContent = `Extracting palette (k=${_state.k})…`;
    l.classList.add("visible");
    const pal = await _ensurePalette(it, _state.k);
    if (!l.classList.contains("visible")) return;
    info.innerHTML = `
        <div style="display:flex;gap:4px;align-items:center;">
            ${pal.map((p) => `<div style="width:24px;height:24px;background:${p.hex};border-radius:3px;border:1px solid var(--c2c-surface0);" title="${p.hex} · ${(p.pct*100).toFixed(1)}%"></div>`).join("")}
            <span style="margin-left:8px;">${pal.map(p => p.hex).join(" · ")}</span>
        </div>
    `;
}

async function _renderPaletteBlock(extra, paletteTitle, palette) {
    extra.innerHTML = `
        <div class="mb-palette">
            <h4>${paletteTitle}</h4>
            <div class="mb-swatches"></div>
            <div class="mb-copy-row">
                <button data-act="copy-hex">Copy hex</button>
                <button data-act="copy-css">Copy CSS vars</button>
                <button data-act="copy-json">Copy JSON</button>
                <button data-act="export-png">Export PNG</button>
            </div>
        </div>
    `;
    const sw = extra.querySelector(".mb-swatches");
    sw.innerHTML = palette.map((p) => `
        <div class="mb-swatch" style="background:${p.hex};color:${_luma(p.rgb) > 128 ? "var(--c2c-black)" : "var(--c2c-white)"};" title="${p.hex} (${(p.pct*100).toFixed(1)}%)">
            <span class="mb-pct">${(p.pct*100).toFixed(0)}%</span>
            ${p.hex}
        </div>
    `).join("");
    const palObj = palette.map((p) => ({ hex: p.hex, pct: +(p.pct.toFixed(4)), rgb: p.rgb }));
    _on(extra.querySelector('[data-act="copy-hex"]'), "click", () => navigator.clipboard.writeText(palette.map((p) => p.hex).join("\n")));
    _on(extra.querySelector('[data-act="copy-css"]'), "click", () => {
        const css = palette.map((p, i) => `--c2c-mood-${i + 1}: ${p.hex};`).join("\n");
        navigator.clipboard.writeText(css);
    });
    _on(extra.querySelector('[data-act="copy-json"]'), "click", () => navigator.clipboard.writeText(JSON.stringify(palObj, null, 2)));
    _on(extra.querySelector('[data-act="export-png"]'), "click", () => _exportPalettePng(palette));
}

function _exportPalettePng(palette) {
    const W = 800, H = 200;
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const ctx = c.getContext("2d");
    let x = 0;
    for (const p of palette) {
        const w = Math.max(8, W * p.pct);
        ctx.fillStyle = p.hex;
        ctx.fillRect(x, 0, w, H);
        // Canvas 2D can't resolve CSS var(); black/white are theme-invariant.
        ctx.fillStyle = _luma(p.rgb) > 128 ? "#000000" : "#ffffff";
        ctx.font = "14px monospace";
        ctx.fillText(p.hex, x + 6, H - 12);
        x += w;
    }
    const a = document.createElement("a");
    a.href = c.toDataURL("image/png");
    a.download = `palette_${Date.now()}.png`;
    a.click();
}

async function _renderAggregate(body) {
    const extra = body.querySelector('[data-role="extra"]');
    extra.innerHTML = `<div style="padding:12px;color:var(--c2c-overlay0);">Aggregating across ${_state.selected.size} images…</div>`;
    const allSamples = [];
    for (const i of _state.selected) {
        const it = _items[i];
        if (!it) continue;
        try {
            const url = _urlFor(it.image);
            if (!url) continue;
            const s = await _samplePixels(url);
            for (const lab of s) allSamples.push(lab);
        } catch {}
    }
    if (!allSamples.length) {
        extra.innerHTML = `<div style="color:var(--c2c-red);">No samples — aggregation failed.</div>`;
        return;
    }
    const pal = _kmeansPalette(allSamples, _state.k);
    _renderPaletteBlock(extra, `Aggregated palette (${_state.selected.size} images, k=${_state.k})`, pal);
}

async function _renderCompare(body) {
    const extra = body.querySelector('[data-role="extra"]');
    const ids = [..._state.selected];
    if (ids.length !== 2) return;
    extra.innerHTML = `<div style="padding:12px;color:var(--c2c-overlay0);">Computing palettes…</div>`;
    const [iA, iB] = ids;
    const A = _items[iA], B = _items[iB];
    const [pA, pB] = await Promise.all([_ensurePalette(A, _state.k), _ensurePalette(B, _state.k)]);
    const uA = _urlFor(A.image), uB = _urlFor(B.image);
    extra.innerHTML = `
        <div class="mb-compare">
            <div>
                <img src="${uA}">
                <div style="margin-top:4px;font-size:10px;color:var(--c2c-okSoft);">A · k=${_state.k}</div>
                <div data-role="pA"></div>
            </div>
            <div>
                <img src="${uB}">
                <div style="margin-top:4px;font-size:10px;color:var(--c2c-yellow);">B · k=${_state.k}</div>
                <div data-role="pB"></div>
            </div>
        </div>
        <div style="margin-top:8px;color:var(--c2c-overlay0);font-size:11px;">
            ΔE (mean nearest-pair, Lab): <b style="color:var(--c2c-mauve);" data-role="dE"></b>
        </div>
    `;
    const pAEl = extra.querySelector('[data-role="pA"]');
    const pBEl = extra.querySelector('[data-role="pB"]');
    pAEl.innerHTML = `<div class="mb-swatches">${pA.map((p) => `<div class="mb-swatch" style="background:${p.hex};color:${_luma(p.rgb)>128?"var(--c2c-black)":"var(--c2c-white)"};font-size:8px;width:40px;height:40px;">${p.hex}</div>`).join("")}</div>`;
    pBEl.innerHTML = `<div class="mb-swatches">${pB.map((p) => `<div class="mb-swatch" style="background:${p.hex};color:${_luma(p.rgb)>128?"var(--c2c-black)":"var(--c2c-white)"};font-size:8px;width:40px;height:40px;">${p.hex}</div>`).join("")}</div>`;
    // Δ E
    let total = 0;
    for (const a of pA) {
        let best = Infinity;
        for (const b of pB) {
            const dl = a.lab[0] - b.lab[0];
            const da = a.lab[1] - b.lab[1];
            const db = a.lab[2] - b.lab[2];
            const d = Math.sqrt(dl * dl + da * da + db * db);
            if (d < best) best = d;
        }
        total += best * a.pct;
    }
    extra.querySelector('[data-role="dE"]').textContent = total.toFixed(2);
}

async function _aiMood(body) {
    if (_state.aiBusy) return;
    _state.aiBusy = true;
    const extra = body.querySelector('[data-role="extra"]');
    extra.innerHTML = `<div class="mb-ai-box" data-role="ai">Computing mood from selected palettes…</div>`;
    const sel = [..._state.selected].slice(0, 6);
    const palettes = [];
    for (const i of sel) {
        const it = _items[i];
        if (!it) continue;
        const p = await _ensurePalette(it, _state.k);
        palettes.push(p.map((sw) => `${sw.hex} (${(sw.pct*100|0)}%)`).join(" "));
    }
    const aiEl = extra.querySelector('[data-role="ai"]');
    _state.aiOutput = "";
    try {
        await streamAI({
            feature: "mood_describe",
            sensitivity: "normal",
            max_tokens: 200,
            temperature: 0.7,
            messages: [
                { role: "system", content: "You are an art-direction assistant. In ≤3 short sentences, describe the mood/vibe these palettes evoke and suggest 2 style tags. No bullet headers — flowing prose." },
                { role: "user", content: `Palettes:\n${palettes.join("\n")}` },
            ],
            onChunk: (c) => { _state.aiOutput += c; if (aiEl) aiEl.textContent = _state.aiOutput; },
            onError: (e) => { _state.aiOutput += `\n[error: ${e}]`; },
            onDone: () => {},
        });
    } catch (e) {
        _state.aiOutput += `\n[exception: ${e?.message || e}]`;
    } finally {
        _state.aiBusy = false;
        _renderBody();
        if (_state.aiOutput) {
            // re-attach output after re-render
            const newExtra = document.querySelector(`#${PANEL_ID} [data-role="extra"]`);
            if (newExtra) newExtra.innerHTML = `<div class="mb-ai-box">${_esc(_state.aiOutput)}</div>`;
        }
    }
}

function _esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));
}

// ─────────────────────────────────────────────────────────────────────────────
// Capture
// ─────────────────────────────────────────────────────────────────────────────

function _onExecuted(ev) {
    const d = ev?.detail;
    if (!d?.output?.images || !Array.isArray(d.output.images)) return;
    const nodeId = d.node;
    let added = false;
    for (const img of d.output.images) {
        const id = `${nodeId}:${img.filename}`;
        if (_items.some(it => it.id === id)) continue;
        _items.push({ id, ts: Date.now(), image: img, palette: null, k: 0 });
        added = true;
    }
    if (!added) return;
    while (_items.length > MAX_THUMBS) {
        _items.shift();
        const newSel = new Set();
        for (const idx of _state.selected) if (idx > 0) newSel.add(idx - 1);
        _state.selected = newSel;
    }
    if (_state.open) _renderBody();
    for (const img of d.output.images) {
        const item = _items.find(it => it.id === `${nodeId}:${img.filename}`);
        if (item && !item.palette) {
            _ensurePalette(item, _state.k).then(() => {
                if (_state.open) _renderBody();
            }).catch(() => {});
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extension entry
// ─────────────────────────────────────────────────────────────────────────────

if (!LITE) app.registerExtension({
    name: "C2C.MoodBoard",
    settings: [
        {
            id: "c2c.mood_board.enabled",
            name: "Mood Board: session output gallery",
            type: "boolean",
            default: true,
            onChange: (v) => {
                const b = document.getElementById(BTN_ID);
                if (b) b.style.display = v ? "flex" : "none";
            },
        },
    ],
    async setup() {
        _injectStyle();
        _ensureUi();
        api.addEventListener("executed", _onExecuted);
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("c2c.mood_board.enabled", true); }
            catch { return true; }
        })();
        const b = document.getElementById(BTN_ID);
        if (b) b.style.display = enabled ? "flex" : "none";
        console.log("[C2C.MoodBoard] godlevel-rebuild loaded.");
    },
});
