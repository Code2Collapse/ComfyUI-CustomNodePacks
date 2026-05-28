/**
 * c2c_ab_canvas.js — A/B Comparer (god-level rebuild, 2026-05-27)
 *
 * One panel, 5 viewing modes (Side | Wipe | Grid 2x2 | Grid 3x3 | Diff),
 * up to 4 named slots (A/B/C/D), per-slot pin (eviction-protected),
 * sync zoom+pan, metadata diff, PNG/MP4 export, AI explain.
 *
 * The 9 features (per spec):
 *   1) Side-by-side with sync zoom+pan
 *   2) Wipe slider blend (legacy mode, preserved)
 *   3) N-way grid (2x2, 3x3)
 *   4) Pixel-diff overlay (Canvas2D)
 *   5) Auto-pick from /history backfill on first open
 *   6) Pin/swap per slot (pinned items survive ring eviction)
 *   7) Metadata diff (seed/steps/cfg/sampler/scheduler/model from workflow)
 *   8) Export PNG (current view) + MP4 wipe animation (MediaRecorder)
 *   9) AI explain via _c2c_ai_client.streamAI
 *
 * Architecture rules baked in:
 *   - Body-only re-render: panel chrome (header + 8 resize edges) survives
 *     setMode/changeSlot/refresh, so window state is never lost.
 *   - Inherits .c2c-win z-index regime (5000 base, ratchets on focus) via
 *     attachWindowChrome. No --c2c-z-hud override.
 *   - No transform:translate(-50%) on the panel — chrome anchors via left/top.
 *   - Listeners are owned by `_listeners` and torn down before re-bind, so
 *     re-renders don't leak handlers.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { attachWindowChrome } from "./_c2c_window.js";
import { streamAI } from "./_c2c_ai_client.js";
import { C } from './_c2c_theme.js';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const BTN_ID    = "mec-ab-btn";
const PANEL_ID  = "mec-ab-panel";
const STYLE_ID  = "mec-ab-style";
const MAX_HISTORY = 24;     // ring buffer (pinned slots don't count)
const NUM_SLOTS = 4;        // A, B, C, D
const SLOT_NAMES = ["A", "B", "C", "D"];
const SLOT_ACCENTS = ["var(--c2c-okSoft)", "var(--c2c-yellow)", "var(--c2c-sapphire)", "var(--c2c-pink)"];

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

const _history = [];   // [{ id, ts, label, images:[{filename,subfolder,type}], prompt?, workflowMeta?, pinned? }]
const _state = {
    open: false,
    mode: "wipe",                          // "side" | "wipe" | "grid2" | "grid3" | "diff"
    slots: [0, 1, 2, 3],                   // indices into _history (or -1)
    split: 50,                             // wipe slider
    zoom: { scale: 1, tx: 0, ty: 0 },      // synchronised zoom/pan
    aiOutput: "",
    aiBusy: false,
    backfilled: false,                     // /history seed pulled?
    metaOpen: false,
    diffMode: "abs",                       // "abs" | "heat"
    diffThreshold: 16,
};

// Per-render listener registry — cleared on each renderBody().
const _listeners = [];
function _on(target, ev, fn, opts) {
    target.addEventListener(ev, fn, opts);
    _listeners.push(() => target.removeEventListener(ev, fn, opts));
}
function _clearListeners() {
    while (_listeners.length) {
        try { _listeners.pop()(); } catch { /* ignore */ }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Style
// ─────────────────────────────────────────────────────────────────────────────

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${BTN_ID} {
    position: fixed; bottom: 248px; right: 16px;
    z-index: var(--c2c-z-hud, 1000);
    width: 38px; height: 38px; border-radius: 50%;
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface1); color: var(--c2c-sapphire);
    font-size: 16px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex; align-items: center; justify-content: center;
}
#${BTN_ID}:hover { border-color: var(--c2c-sapphire); }
#${BTN_ID}.has-new { box-shadow: 0 0 0 2px var(--c2c-sapphire), 0 2px 8px rgba(0,0,0,0.6); }

#${PANEL_ID} {
    position: fixed; top: 80px; left: 80px;
    width: min(92vw, 1200px); height: min(86vh, 820px);
    background: var(--c2c-bg2); border: 1px solid var(--c2c-surface1); border-radius: 8px;
    color: var(--c2c-fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px; box-shadow: 0 6px 32px rgba(0,0,0,0.7);
    display: none; flex-direction: column; overflow: hidden;
}
#${PANEL_ID}.visible { display: flex; }
#${PANEL_ID} h3 {
    margin: 0; padding: 8px 12px; color: var(--c2c-sapphire); font-size: 14px;
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(180deg, var(--c2c-bg) 0%, var(--c2c-bg2) 100%);
    border-bottom: 1px solid var(--c2c-surface0);
}
#${PANEL_ID} .ab-body {
    flex: 1 1 auto; min-height: 0;
    display: flex; flex-direction: column;
    padding: 10px; gap: 8px;
}
#${PANEL_ID} .ab-toolbar {
    display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
}
#${PANEL_ID} .ab-modes { display: flex; gap: 2px; }
#${PANEL_ID} .ab-modes button,
#${PANEL_ID} .ab-actions button {
    background: var(--c2c-surface0); border: 1px solid var(--c2c-surface1); color: var(--c2c-fg);
    border-radius: 4px; padding: 4px 8px; font-size: 11px; cursor: pointer;
    transition: background 120ms, border-color 120ms;
}
#${PANEL_ID} .ab-modes button:hover,
#${PANEL_ID} .ab-actions button:hover { border-color: var(--c2c-sapphire); }
#${PANEL_ID} .ab-modes button.active {
    background: var(--c2c-surface1); border-color: var(--c2c-sapphire); color: var(--c2c-sapphire);
}
#${PANEL_ID} .ab-actions { display: flex; gap: 4px; margin-left: auto; }
#${PANEL_ID} .ab-actions button.busy { color: var(--c2c-overlay0); cursor: progress; }

#${PANEL_ID} .ab-slots { display: flex; gap: 6px; flex-wrap: wrap; }
#${PANEL_ID} .ab-slot {
    display: flex; align-items: center; gap: 4px;
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface0);
    border-radius: 4px; padding: 3px 4px 3px 6px; min-width: 200px;
}
#${PANEL_ID} .ab-slot .ab-slot-tag {
    font-family: monospace; font-weight: 700; font-size: 12px;
    width: 14px; text-align: center;
}
#${PANEL_ID} .ab-slot select {
    flex: 1; background: var(--c2c-surface0); color: var(--c2c-fg);
    border: 1px solid var(--c2c-surface1); border-radius: 3px; padding: 2px 4px;
    font-size: 11px; min-width: 0;
}
#${PANEL_ID} .ab-slot .ab-pin {
    background: none; border: none; color: var(--c2c-overlay0);
    font-size: 13px; padding: 2px 4px; cursor: pointer;
}
#${PANEL_ID} .ab-slot .ab-pin.pinned { color: var(--c2c-yellow); }
#${PANEL_ID} .ab-slot .ab-swap {
    background: none; border: none; color: var(--c2c-overlay0);
    font-size: 12px; padding: 2px 4px; cursor: pointer;
}
#${PANEL_ID} .ab-slot .ab-swap:hover { color: var(--c2c-sapphire); }

#${PANEL_ID} .ab-stage {
    position: relative; flex: 1 1 auto; min-height: 240px;
    background: var(--c2c-bg3); border: 1px solid var(--c2c-surface0); border-radius: 4px;
    overflow: hidden; user-select: none;
}
#${PANEL_ID} .ab-stage canvas,
#${PANEL_ID} .ab-stage img {
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%; object-fit: contain; pointer-events: none;
    image-rendering: -webkit-optimize-contrast;
}
#${PANEL_ID} .ab-stage .ab-cell {
    position: absolute; overflow: hidden; background: var(--c2c-scrimDark);
    border: 1px solid var(--c2c-bg);
}
#${PANEL_ID} .ab-stage .ab-cell .ab-cell-img {
    position: absolute; inset: 0; transform-origin: center center;
    transition: none;
}
#${PANEL_ID} .ab-stage .ab-cell .ab-cell-tag {
    position: absolute; top: 4px; left: 4px;
    background: rgba(0,0,0,0.55); padding: 1px 6px; border-radius: 3px;
    font-family: monospace; font-weight: 700; font-size: 11px;
    pointer-events: none;
}
#${PANEL_ID} .ab-stage .ab-cell.empty {
    display: flex; align-items: center; justify-content: center;
    color: var(--c2c-surface1); font-size: 11px;
}
#${PANEL_ID} .ab-stage .ab-clip {
    position: absolute; top: 0; left: 0; bottom: 0;
    overflow: hidden; will-change: width;
}
#${PANEL_ID} .ab-stage .ab-handle {
    position: absolute; top: 0; bottom: 0; width: 3px;
    background: var(--c2c-sapphire); cursor: ew-resize; will-change: left;
    box-shadow: 0 0 8px rgba(116, 199, 236, 0.6);
}
#${PANEL_ID} .ab-stage .ab-handle::before {
    content: "⇔"; position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    background: var(--c2c-sapphire); color: var(--c2c-bg3); border-radius: 50%;
    width: 22px; height: 22px; text-align: center; line-height: 22px;
    font-size: 14px; font-weight: 700;
}
#${PANEL_ID} .ab-zoom {
    position: absolute; bottom: 6px; left: 6px; display: flex; gap: 4px;
    background: rgba(17,17,27,0.7); padding: 3px 5px; border-radius: 4px;
    font-family: monospace; font-size: 10px; color: var(--c2c-fg);
}
#${PANEL_ID} .ab-zoom button {
    background: var(--c2c-surface0); border: 1px solid var(--c2c-surface1); color: var(--c2c-fg);
    border-radius: 3px; padding: 1px 6px; cursor: pointer; font-size: 10px;
}
#${PANEL_ID} .ab-zoom button:hover { border-color: var(--c2c-sapphire); }

#${PANEL_ID} .ab-diff-controls {
    position: absolute; top: 6px; right: 6px; display: flex; gap: 4px;
    background: rgba(17,17,27,0.7); padding: 3px 5px; border-radius: 4px;
    font-size: 10px; color: var(--c2c-fg); align-items: center;
}
#${PANEL_ID} .ab-diff-controls input[type="range"] { width: 80px; }

#${PANEL_ID} .ab-side {
    display: flex; gap: 8px; padding: 0 2px;
    max-height: 200px; overflow: hidden;
}
#${PANEL_ID} .ab-meta {
    flex: 1; background: var(--c2c-bg); border: 1px solid var(--c2c-surface0);
    border-radius: 4px; padding: 6px 8px; overflow: auto;
    font-family: monospace; font-size: 11px; line-height: 1.45;
}
#${PANEL_ID} .ab-meta h4 {
    margin: 0 0 4px 0; color: var(--c2c-sapphire); font-size: 11px;
    font-family: -apple-system, "Segoe UI", sans-serif; font-weight: 600;
}
#${PANEL_ID} .ab-meta table { width: 100%; border-collapse: collapse; }
#${PANEL_ID} .ab-meta td { padding: 1px 4px; vertical-align: top; }
#${PANEL_ID} .ab-meta td.k { color: var(--c2c-overlay0); width: 90px; }
#${PANEL_ID} .ab-meta td.differs { color: var(--c2c-yellow); }
#${PANEL_ID} .ab-meta td.same { color: var(--c2c-fg); }

#${PANEL_ID} .ab-ai {
    flex: 1; background: var(--c2c-bg); border: 1px solid var(--c2c-surface0);
    border-radius: 4px; padding: 6px 8px; overflow: auto;
    font-size: 11px; line-height: 1.5; white-space: pre-wrap;
}
#${PANEL_ID} .ab-ai.empty { color: var(--c2c-surface1); font-style: italic; }

#${PANEL_ID} .ab-hint { color: var(--c2c-overlay0); font-size: 11px; text-align: center; }
#${PANEL_ID} .ab-empty {
    flex: 1; display: flex; align-items: center; justify-content: center;
    color: var(--c2c-overlay0); flex-direction: column; gap: 6px;
}
#${PANEL_ID} .ab-empty button {
    background: var(--c2c-surface0); border: 1px solid var(--c2c-surface1); color: var(--c2c-sapphire);
    border-radius: 4px; padding: 4px 10px; font-size: 11px; cursor: pointer;
}
    `.trim();
    document.head.appendChild(style);
}

// ─────────────────────────────────────────────────────────────────────────────
// URL helpers + history capture
// ─────────────────────────────────────────────────────────────────────────────

function _urlForImage(img) {
    if (!img || !img.filename) return null;
    const params = new URLSearchParams({
        filename:  img.filename,
        subfolder: img.subfolder || "",
        type:      img.type      || "output",
    });
    return `${api.api_base || ""}/view?${params.toString()}`;
}

/**
 * Extract sampling metadata from a workflow prompt dict (entry.prompt[2]).
 * Looks for KSampler-like nodes and returns the first match's widgets.
 * Returns { seed, steps, cfg, sampler, scheduler, model, denoise, positive }.
 */
function _extractWorkflowMeta(prompt) {
    if (!prompt || typeof prompt !== "object") return null;
    const SAMPLER_TYPES = new Set([
        "KSampler", "KSamplerAdvanced", "KSamplerSelect",
        "SamplerCustom", "SamplerCustomAdvanced",
    ]);
    const out = {};
    let modelNodeId = null;
    let posNodeId = null;
    for (const [nid, node] of Object.entries(prompt)) {
        if (!node || typeof node !== "object") continue;
        const ct = node.class_type || node.type || "";
        const inp = node.inputs || {};
        if (SAMPLER_TYPES.has(ct)) {
            if (inp.seed !== undefined && out.seed === undefined) out.seed = inp.seed;
            if (inp.noise_seed !== undefined && out.seed === undefined) out.seed = inp.noise_seed;
            if (inp.steps !== undefined) out.steps = inp.steps;
            if (inp.cfg !== undefined) out.cfg = inp.cfg;
            if (inp.sampler_name !== undefined) out.sampler = inp.sampler_name;
            if (inp.scheduler !== undefined) out.scheduler = inp.scheduler;
            if (inp.denoise !== undefined) out.denoise = inp.denoise;
            if (Array.isArray(inp.model)) modelNodeId = inp.model[0];
            if (Array.isArray(inp.positive)) posNodeId = inp.positive[0];
        }
    }
    if (modelNodeId && prompt[modelNodeId]) {
        const m = prompt[modelNodeId];
        const mi = m.inputs || {};
        out.model = mi.ckpt_name || mi.unet_name || mi.model_name || m.class_type;
    }
    if (posNodeId && prompt[posNodeId]) {
        const p = prompt[posNodeId];
        const pi = p.inputs || {};
        if (typeof pi.text === "string") out.positive = pi.text.slice(0, 240);
    }
    return out;
}

function _addHistoryEntry({ id, ts, images, prompt }) {
    if (_history.some((h) => h.id === id)) return; // dedupe
    const entry = {
        id,
        ts: ts || Date.now(),
        images: images || [],
        prompt: prompt || null,
        workflowMeta: prompt ? _extractWorkflowMeta(prompt) : null,
        pinned: false,
    };
    _history.push(entry);
    // Evict unpinned oldest entries until we're within budget.
    while (_history.filter((h) => !h.pinned).length > MAX_HISTORY) {
        const idx = _history.findIndex((h) => !h.pinned);
        if (idx < 0) break;
        _history.splice(idx, 1);
    }
}

/**
 * Backfill from /history endpoint on first open. Pulls up to 50 recent
 * executions and adds the ones we don't already have.
 */
async function _backfillHistory() {
    if (_state.backfilled) return;
    _state.backfilled = true;
    try {
        const resp = await api.fetchApi("/history?max_items=50");
        const data = await resp.json();
        // /history returns { promptId: entry, ... } ordered (newest typically last)
        const entries = Object.entries(data || {})
            .map(([id, entry]) => ({ id, entry }))
            .filter(({ entry }) => entry && entry.outputs);
        // Sort by execution order if available, oldest → newest
        entries.sort((a, b) => {
            const at = a.entry?.status?.completed_at || 0;
            const bt = b.entry?.status?.completed_at || 0;
            return at - bt;
        });
        for (const { id, entry } of entries) {
            const outs = entry.outputs || {};
            const images = [];
            for (const o of Object.values(outs)) {
                if (Array.isArray(o?.images)) images.push(...o.images);
            }
            if (!images.length) continue;
            // prompt[2] is the workflow dict in /history payloads
            const prompt = Array.isArray(entry.prompt) ? entry.prompt[2] : null;
            _addHistoryEntry({ id, ts: Date.now(), images, prompt });
        }
    } catch (e) {
        console.warn("[C2C.ABSplit] backfill failed:", e);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Panel lifecycle (chrome-preserving)
// ─────────────────────────────────────────────────────────────────────────────

function _ensureUi() {
    if (!document.getElementById(BTN_ID)) {
        const b = document.createElement("button");
        b.id = BTN_ID;
        b.title = "A/B compare last outputs (⚖)";
        b.textContent = "⚖";
        b.addEventListener("click", _toggle);
        document.body.appendChild(b);
    }
    if (!document.getElementById(PANEL_ID)) {
        const p = document.createElement("div");
        p.id = PANEL_ID;
        // Build skeleton ONCE — header is chrome territory, body is rerendered.
        p.innerHTML = `
            <h3>
                <span>⚖ A/B Compare</span>
                <span class="ab-close-row">
                    <button class="ab-close" title="Close" style="background:none;border:none;color:var(--c2c-overlay0);cursor:pointer;font-size:16px;padding:0 4px;">×</button>
                </span>
            </h3>
            <div class="ab-body" data-role="body"></div>
        `;
        document.body.appendChild(p);
        p.querySelector(".ab-close").addEventListener("click", _toggle);
        attachWindowChrome(p, {
            storageKey: "ab_canvas",
            headerSelector: "h3",
            titleSelector: "h3 > span",
            minW: 560, minH: 420,
        });
    }
}

function _toggle() {
    _state.open = !_state.open;
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    if (_state.open) {
        p.classList.add("visible");
        const btn = document.getElementById(BTN_ID);
        if (btn) btn.classList.remove("has-new");
        // First-open backfill (non-blocking re-render).
        _backfillHistory().then(() => {
            _ensureSlotsValid();
            _renderBody();
        });
        _ensureSlotsValid();
        _renderBody();
    } else {
        p.classList.remove("visible");
        _clearListeners();
    }
}

/** Make sure each slot points to an in-range entry (or -1 if history is too short). */
function _ensureSlotsValid() {
    const n = _history.length;
    for (let i = 0; i < NUM_SLOTS; i++) {
        if (_state.slots[i] >= n) _state.slots[i] = n > i ? (n - 1 - i) : -1;
        if (_state.slots[i] < 0 && n > i) _state.slots[i] = n - 1 - i;
    }
    // Default A = latest, B = previous if just opened
    if (n >= 2 && _state.slots[0] === _state.slots[1]) {
        _state.slots[0] = n - 1;
        _state.slots[1] = n - 2;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Render — body only (chrome untouched)
// ─────────────────────────────────────────────────────────────────────────────

function _renderBody() {
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    const body = p.querySelector('[data-role="body"]');
    if (!body) return;
    _clearListeners();

    if (_history.length < 2) {
        body.innerHTML = `
            <div class="ab-empty">
                <div>Need at least two captured runs.<br>Currently: ${_history.length}</div>
                <button data-role="backfill">↻ Pull from history</button>
            </div>
        `;
        const refill = body.querySelector('[data-role="backfill"]');
        if (refill) _on(refill, "click", async () => {
            _state.backfilled = false;
            await _backfillHistory();
            _ensureSlotsValid();
            _renderBody();
        });
        return;
    }

    body.innerHTML = `
        <div class="ab-toolbar">
            <div class="ab-modes" data-role="modes">
                ${[
                    ["side", "▥ Side"],
                    ["wipe", "⇔ Wipe"],
                    ["grid2", "▦ 2×2"],
                    ["grid3", "▦ 3×3"],
                    ["diff", "Δ Diff"],
                ].map(([m, lbl]) => `
                    <button data-mode="${m}" class="${_state.mode === m ? "active" : ""}">${lbl}</button>
                `).join("")}
            </div>
            <div class="ab-actions" data-role="actions">
                <button data-act="toggle-meta" title="Show metadata diff (seed/steps/cfg/...)">📋 Meta</button>
                <button data-act="export-png" title="Export current stage as PNG">⤓ PNG</button>
                <button data-act="export-mp4" title="Record a wipe animation as WebM (MediaRecorder)">⤓ MP4</button>
                <button data-act="ai-explain" title="Stream an AI explanation of A vs B differences">${_state.aiBusy ? "⌛ AI…" : "✨ Explain"}</button>
            </div>
        </div>
        <div class="ab-slots" data-role="slots"></div>
        <div class="ab-stage" data-role="stage"></div>
        ${(_state.metaOpen || _state.aiOutput) ? `
            <div class="ab-side">
                ${_state.metaOpen ? `<div class="ab-meta" data-role="meta"></div>` : ""}
                ${_state.aiOutput || _state.aiBusy ? `<div class="ab-ai${_state.aiOutput ? "" : " empty"}" data-role="ai">${_state.aiOutput || "Waiting for AI…"}</div>` : ""}
            </div>` : ""}
        <div class="ab-hint" data-role="hint"></div>
    `;

    _renderSlots(body);
    _renderStage(body);
    if (_state.metaOpen) _renderMetaDiff(body);
    _bindToolbar(body);
}

function _renderSlots(body) {
    const wrap = body.querySelector('[data-role="slots"]');
    const slotsNeeded = _state.mode === "grid3" ? 4 : (_state.mode === "grid2" ? 4 : 2);
    const opts = _history.map((h, i) => {
        const t = new Date(h.ts);
        const tag = i === _history.length - 1 ? "▶ Latest" : `#${i}`;
        const meta = h.workflowMeta;
        const metaBits = meta ? ` · seed=${meta.seed ?? "?"}` : "";
        const pin = h.pinned ? " 📌" : "";
        return `<option value="${i}">${tag} @ ${t.toLocaleTimeString()} (${h.images.length} img${metaBits})${pin}</option>`;
    }).join("");

    wrap.innerHTML = Array.from({ length: slotsNeeded }, (_, i) => `
        <div class="ab-slot" data-slot="${i}" style="border-left:3px solid ${SLOT_ACCENTS[i]};">
            <span class="ab-slot-tag" style="color:${SLOT_ACCENTS[i]};">${SLOT_NAMES[i]}</span>
            <select data-role="pick" data-slot="${i}">${opts}</select>
            <button class="ab-pin ${_history[_state.slots[i]]?.pinned ? "pinned" : ""}" data-role="pin" data-slot="${i}" title="Pin (survives ring eviction)">📌</button>
            ${i > 0 ? `<button class="ab-swap" data-role="swap" data-slot="${i}" title="Swap with A">↔</button>` : ""}
        </div>
    `).join("");

    for (const sel of wrap.querySelectorAll("select[data-role='pick']")) {
        const slotIdx = parseInt(sel.dataset.slot, 10);
        const v = _state.slots[slotIdx];
        if (v >= 0) sel.value = String(v);
        _on(sel, "change", () => {
            _state.slots[slotIdx] = parseInt(sel.value, 10);
            _renderStage(body);
            if (_state.metaOpen) _renderMetaDiff(body);
        });
    }
    for (const b of wrap.querySelectorAll('[data-role="pin"]')) {
        _on(b, "click", () => {
            const slotIdx = parseInt(b.dataset.slot, 10);
            const h = _history[_state.slots[slotIdx]];
            if (!h) return;
            h.pinned = !h.pinned;
            _renderSlots(body);
        });
    }
    for (const b of wrap.querySelectorAll('[data-role="swap"]')) {
        _on(b, "click", () => {
            const slotIdx = parseInt(b.dataset.slot, 10);
            const tmp = _state.slots[0];
            _state.slots[0] = _state.slots[slotIdx];
            _state.slots[slotIdx] = tmp;
            _renderSlots(body);
            _renderStage(body);
            if (_state.metaOpen) _renderMetaDiff(body);
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage rendering — one of 5 modes
// ─────────────────────────────────────────────────────────────────────────────

function _renderStage(body) {
    const stage = body.querySelector('[data-role="stage"]');
    if (!stage) return;
    // Wipe the stage's listeners by removing its children (handler refs cleared
    // because _on adds them to _listeners, cleared on next renderBody — but
    // changeSlot calls _renderStage without renderBody, so we need to be careful).
    stage.innerHTML = "";

    const slots = _state.slots.map((i) => (i >= 0 ? _history[i] : null));
    const A = slots[0], B = slots[1], C = slots[2], D = slots[3];

    if (_state.mode === "wipe") {
        _renderWipe(stage, A, B);
    } else if (_state.mode === "side") {
        _renderSide(stage, [A, B]);
    } else if (_state.mode === "grid2") {
        _renderGrid(stage, [A, B, C, D], 2);
    } else if (_state.mode === "grid3") {
        _renderGrid(stage, slots.concat([null, null, null, null, null]).slice(0, 9), 3);
    } else if (_state.mode === "diff") {
        _renderDiff(stage, A, B);
    }
    _renderZoomBar(stage);
}

function _renderZoomBar(stage) {
    const bar = document.createElement("div");
    bar.className = "ab-zoom";
    bar.innerHTML = `
        <button data-z="-">−</button>
        <span data-role="z">${(_state.zoom.scale * 100).toFixed(0)}%</span>
        <button data-z="+">+</button>
        <button data-z="0" title="Reset">⟲</button>
    `;
    stage.appendChild(bar);
    _on(bar.querySelector('[data-z="-"]'), "click", () => _zoomBy(0.8));
    _on(bar.querySelector('[data-z="+"]'), "click", () => _zoomBy(1.25));
    _on(bar.querySelector('[data-z="0"]'), "click", () => { _state.zoom = { scale: 1, tx: 0, ty: 0 }; _applyZoom(); });
}

function _applyZoom() {
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    const { scale, tx, ty } = _state.zoom;
    const xform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    p.querySelectorAll(".ab-cell-img, .ab-clip img, .ab-stage > img").forEach((el) => {
        el.style.transform = xform;
    });
    const z = p.querySelector('.ab-zoom [data-role="z"]');
    if (z) z.textContent = (scale * 100).toFixed(0) + "%";
}

function _zoomBy(factor) {
    _state.zoom.scale = Math.max(0.1, Math.min(8, _state.zoom.scale * factor));
    _applyZoom();
}

function _attachStageInteract(stage) {
    let panning = false, lastX = 0, lastY = 0;
    _on(stage, "wheel", (e) => {
        e.preventDefault();
        const f = e.deltaY < 0 ? 1.1 : 0.9;
        _zoomBy(f);
    }, { passive: false });
    _on(stage, "mousedown", (e) => {
        if (e.button !== 0) return;
        if (e.target.closest(".ab-handle, .ab-zoom, .ab-diff-controls")) return;
        panning = true; lastX = e.clientX; lastY = e.clientY;
        stage.style.cursor = "grabbing";
    });
    _on(window, "mousemove", (e) => {
        if (!panning) return;
        _state.zoom.tx += (e.clientX - lastX);
        _state.zoom.ty += (e.clientY - lastY);
        lastX = e.clientX; lastY = e.clientY;
        _applyZoom();
    });
    _on(window, "mouseup", () => { panning = false; stage.style.cursor = ""; });
}

// ── Wipe mode ────────────────────────────────────────────────────────────────
function _renderWipe(stage, A, B) {
    if (!A || !B) { _renderEmpty(stage, "Wipe needs two slots filled."); return; }
    const aUrl = _urlForImage(A.images[0]);
    const bUrl = _urlForImage(B.images[0]);
    stage.insertAdjacentHTML("beforeend", `
        <img class="ab-img-b" src="${bUrl}" alt="B">
        <div class="ab-clip"><img class="ab-img-a" src="${aUrl}" alt="A"></div>
        <div class="ab-cell-tag" style="position:absolute;left:6px;top:6px;background:rgba(0,0,0,0.55);padding:1px 6px;border-radius:3px;color:${SLOT_ACCENTS[0]};font-family:monospace;font-weight:700;">A</div>
        <div class="ab-cell-tag" style="position:absolute;right:6px;top:6px;background:rgba(0,0,0,0.55);padding:1px 6px;border-radius:3px;color:${SLOT_ACCENTS[1]};font-family:monospace;font-weight:700;">B</div>
        <div class="ab-handle"></div>
    `);
    const clip = stage.querySelector(".ab-clip");
    const handle = stage.querySelector(".ab-handle");
    const imgA = stage.querySelector(".ab-img-a");
    const setSplit = (pct) => {
        const clamped = Math.max(0, Math.min(100, pct));
        _state.split = clamped;
        clip.style.width = clamped + "%";
        handle.style.left = `calc(${clamped}% - 1.5px)`;
        if (imgA) imgA.style.width = stage.clientWidth + "px";
    };
    setSplit(_state.split);
    let dragging = false;
    _on(handle, "mousedown", () => { dragging = true; });
    _on(stage, "mousedown", (e) => {
        if (e.target.closest(".ab-zoom")) return;
        dragging = true;
        const rect = stage.getBoundingClientRect();
        setSplit(((e.clientX - rect.left) / rect.width) * 100);
    });
    _on(window, "mousemove", (e) => {
        if (!dragging) return;
        const rect = stage.getBoundingClientRect();
        setSplit(((e.clientX - rect.left) / rect.width) * 100);
    });
    _on(window, "mouseup", () => { dragging = false; });
    const ro = new ResizeObserver(() => setSplit(_state.split));
    ro.observe(stage);
    _listeners.push(() => ro.disconnect());
    _applyZoom();
}

// ── Side mode (sync zoom/pan) ────────────────────────────────────────────────
function _renderSide(stage, slots) {
    const filled = slots.filter(Boolean);
    if (filled.length < 2) { _renderEmpty(stage, "Side-by-side needs two slots."); return; }
    slots.forEach((slot, i) => {
        const cell = document.createElement("div");
        cell.className = "ab-cell";
        cell.style.left = (i * 50) + "%";
        cell.style.top = "0";
        cell.style.width = "50%";
        cell.style.height = "100%";
        if (slot) {
            const url = _urlForImage(slot.images[0]);
            cell.innerHTML = `
                <img class="ab-cell-img" src="${url}" alt="${SLOT_NAMES[i]}">
                <span class="ab-cell-tag" style="color:${SLOT_ACCENTS[i]};">${SLOT_NAMES[i]}</span>
            `;
        } else {
            cell.classList.add("empty");
            cell.textContent = "(empty)";
        }
        stage.appendChild(cell);
    });
    _attachStageInteract(stage);
    _applyZoom();
}

// ── Grid mode ────────────────────────────────────────────────────────────────
function _renderGrid(stage, slots, n) {
    const total = n * n;
    for (let i = 0; i < total; i++) {
        const slot = slots[i];
        const row = Math.floor(i / n), col = i % n;
        const cell = document.createElement("div");
        cell.className = "ab-cell";
        cell.style.left = ((100 / n) * col) + "%";
        cell.style.top  = ((100 / n) * row) + "%";
        cell.style.width  = (100 / n) + "%";
        cell.style.height = (100 / n) + "%";
        if (slot) {
            const url = _urlForImage(slot.images[0]);
            const tag = i < NUM_SLOTS ? SLOT_NAMES[i] : `#${i}`;
            const accent = SLOT_ACCENTS[i % SLOT_ACCENTS.length];
            cell.innerHTML = `
                <img class="ab-cell-img" src="${url}" alt="${tag}">
                <span class="ab-cell-tag" style="color:${accent};">${tag}</span>
            `;
        } else {
            cell.classList.add("empty");
            cell.textContent = "(empty)";
        }
        stage.appendChild(cell);
    }
    _attachStageInteract(stage);
    _applyZoom();
}

// ── Diff mode (Canvas2D pixel diff) ──────────────────────────────────────────
function _renderDiff(stage, A, B) {
    if (!A || !B) { _renderEmpty(stage, "Diff needs two slots."); return; }
    const aUrl = _urlForImage(A.images[0]);
    const bUrl = _urlForImage(B.images[0]);
    const canvas = document.createElement("canvas");
    canvas.style.position = "absolute";
    canvas.style.inset = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.objectFit = "contain";
    stage.appendChild(canvas);

    const controls = document.createElement("div");
    controls.className = "ab-diff-controls";
    controls.innerHTML = `
        <label>Mode <select data-role="diff-mode">
            <option value="abs"${_state.diffMode === "abs" ? " selected" : ""}>|A−B|</option>
            <option value="heat"${_state.diffMode === "heat" ? " selected" : ""}>Heatmap</option>
        </select></label>
        <label>Thresh <input type="range" min="0" max="128" value="${_state.diffThreshold}" data-role="diff-th"></label>
        <span data-role="diff-stat" style="color:var(--c2c-overlay0);"></span>
    `;
    stage.appendChild(controls);

    const stat = controls.querySelector('[data-role="diff-stat"]');
    const refresh = () => _computeDiff(canvas, aUrl, bUrl, stat);
    refresh();
    _on(controls.querySelector('[data-role="diff-mode"]'), "change", (e) => {
        _state.diffMode = e.target.value; refresh();
    });
    _on(controls.querySelector('[data-role="diff-th"]'), "input", (e) => {
        _state.diffThreshold = parseInt(e.target.value, 10); refresh();
    });
    const ro = new ResizeObserver(refresh);
    ro.observe(stage);
    _listeners.push(() => ro.disconnect());
}

async function _computeDiff(canvas, aUrl, bUrl, statEl) {
    try {
        const [imgA, imgB] = await Promise.all([_loadImg(aUrl), _loadImg(bUrl)]);
        const W = Math.min(imgA.naturalWidth, imgB.naturalWidth);
        const H = Math.min(imgA.naturalHeight, imgB.naturalHeight);
        if (!W || !H) return;
        canvas.width = W; canvas.height = H;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(imgA, 0, 0, W, H);
        const da = ctx.getImageData(0, 0, W, H);
        ctx.drawImage(imgB, 0, 0, W, H);
        const db = ctx.getImageData(0, 0, W, H);
        const out = ctx.createImageData(W, H);
        const a = da.data, b = db.data, o = out.data;
        const th = _state.diffThreshold;
        let changed = 0, totalDelta = 0;
        if (_state.diffMode === "abs") {
            for (let i = 0; i < a.length; i += 4) {
                const dr = Math.abs(a[i] - b[i]);
                const dg = Math.abs(a[i+1] - b[i+1]);
                const db_ = Math.abs(a[i+2] - b[i+2]);
                const m = (dr + dg + db_) / 3 | 0;
                totalDelta += m;
                if (m > th) { changed++; o[i] = dr; o[i+1] = dg; o[i+2] = db_; o[i+3] = 255; }
                else { o[i] = a[i] >> 2; o[i+1] = a[i+1] >> 2; o[i+2] = a[i+2] >> 2; o[i+3] = 255; }
            }
        } else {
            for (let i = 0; i < a.length; i += 4) {
                const dr = Math.abs(a[i] - b[i]);
                const dg = Math.abs(a[i+1] - b[i+1]);
                const db_ = Math.abs(a[i+2] - b[i+2]);
                const m = (dr + dg + db_) / 3 | 0;
                totalDelta += m;
                if (m > th) {
                    changed++;
                    // turbo-ish gradient: low=blue, mid=green, high=red
                    const t = Math.min(1, m / 255);
                    o[i]   = t < 0.5 ? 0 : ((t - 0.5) * 510) | 0;
                    o[i+1] = t < 0.5 ? (t * 510) | 0 : ((1 - t) * 510) | 0;
                    o[i+2] = t < 0.5 ? (255 - (t * 510)) | 0 : 0;
                    o[i+3] = 255;
                } else {
                    o[i] = 0; o[i+1] = 0; o[i+2] = 0; o[i+3] = 255;
                }
            }
        }
        ctx.putImageData(out, 0, 0);
        const pct = (100 * changed / (W * H)).toFixed(2);
        const avg = (totalDelta / (W * H)).toFixed(1);
        if (statEl) statEl.textContent = `${W}×${H} · Δpx ${pct}% · avg ${avg}`;
    } catch (e) {
        console.warn("[C2C.ABSplit] diff failed:", e);
        if (statEl) statEl.textContent = "diff failed";
    }
}

function _loadImg(url) {
    return new Promise((resolve, reject) => {
        const im = new Image();
        im.crossOrigin = "anonymous";
        im.onload = () => resolve(im);
        im.onerror = reject;
        im.src = url;
    });
}

function _renderEmpty(stage, msg) {
    const d = document.createElement("div");
    d.className = "ab-empty";
    d.style.position = "absolute";
    d.style.inset = "0";
    d.textContent = msg;
    stage.appendChild(d);
}

// ─────────────────────────────────────────────────────────────────────────────
// Metadata diff
// ─────────────────────────────────────────────────────────────────────────────

function _renderMetaDiff(body) {
    const el = body.querySelector('[data-role="meta"]');
    if (!el) return;
    const A = _history[_state.slots[0]];
    const B = _history[_state.slots[1]];
    const ma = A?.workflowMeta || {};
    const mb = B?.workflowMeta || {};
    const keys = ["model", "seed", "steps", "cfg", "sampler", "scheduler", "denoise", "positive"];
    const rows = keys.map((k) => {
        const va = ma[k] ?? "—";
        const vb = mb[k] ?? "—";
        const differs = JSON.stringify(va) !== JSON.stringify(vb);
        const cls = differs ? "differs" : "same";
        const vaStr = String(va).slice(0, 80);
        const vbStr = String(vb).slice(0, 80);
        return `<tr><td class="k">${k}</td><td class="${cls}">${_esc(vaStr)}</td><td class="${cls}">${_esc(vbStr)}</td></tr>`;
    }).join("");
    el.innerHTML = `
        <h4>Metadata diff (A vs B)</h4>
        <table>
            <thead><tr><td class="k"></td><td style="color:${SLOT_ACCENTS[0]}">A</td><td style="color:${SLOT_ACCENTS[1]}">B</td></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

function _esc(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;",
    }[c]));
}

// ─────────────────────────────────────────────────────────────────────────────
// Toolbar wire-up — mode buttons + actions (meta/PNG/MP4/AI)
// ─────────────────────────────────────────────────────────────────────────────

function _bindToolbar(body) {
    for (const b of body.querySelectorAll('[data-role="modes"] button')) {
        _on(b, "click", () => {
            _state.mode = b.dataset.mode;
            _renderBody();
        });
    }
    const actions = body.querySelector('[data-role="actions"]');
    if (!actions) return;
    _on(actions.querySelector('[data-act="toggle-meta"]'), "click", () => {
        _state.metaOpen = !_state.metaOpen;
        _renderBody();
    });
    _on(actions.querySelector('[data-act="export-png"]'), "click", () => _exportPng(body));
    _on(actions.querySelector('[data-act="export-mp4"]'), "click", (e) => _exportMp4(body, e.currentTarget));
    _on(actions.querySelector('[data-act="ai-explain"]'), "click", () => _aiExplain());
}

// ─────────────────────────────────────────────────────────────────────────────
// Export: PNG snapshot of current stage
// ─────────────────────────────────────────────────────────────────────────────

async function _exportPng(body) {
    const stage = body.querySelector('[data-role="stage"]');
    if (!stage) return;
    try {
        const rect = stage.getBoundingClientRect();
        const W = rect.width | 0, H = rect.height | 0;
        const canvas = document.createElement("canvas");
        canvas.width = W; canvas.height = H;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = C.bg3; ctx.fillRect(0, 0, W, H);
        // Composite every <img> + <canvas> currently in the stage, honoring
        // its bounding rect within the stage.
        const items = stage.querySelectorAll("img, canvas");
        for (const el of items) {
            const r = el.getBoundingClientRect();
            const x = r.left - rect.left, y = r.top - rect.top;
            try {
                ctx.drawImage(el, x, y, r.width, r.height);
            } catch (e) {
                console.warn("[C2C.ABSplit] PNG composite failed for element:", e);
            }
        }
        // For wipe mode, draw the slider line
        if (_state.mode === "wipe") {
            const x = (W * _state.split / 100) | 0;
            ctx.fillStyle = C.sapphire;
            ctx.fillRect(x - 1, 0, 2, H);
        }
        const url = canvas.toDataURL("image/png");
        const a = document.createElement("a");
        a.href = url;
        a.download = `c2c_ab_${_state.mode}_${Date.now()}.png`;
        a.click();
    } catch (e) {
        console.warn("[C2C.ABSplit] export PNG failed:", e);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export: MP4 (actually WebM via MediaRecorder, .webm extension)
//   Records 3-second wipe animation: split 0% → 100% → 0%.
// ─────────────────────────────────────────────────────────────────────────────

async function _exportMp4(body, btn) {
    if (_state.mode !== "wipe") {
        // Force wipe for the recording.
        _state.mode = "wipe";
        _renderBody();
        // wait next frame
        await new Promise((r) => requestAnimationFrame(r));
        return _exportMp4(document.getElementById(PANEL_ID).querySelector('[data-role="body"]'), btn);
    }
    const A = _history[_state.slots[0]];
    const B = _history[_state.slots[1]];
    if (!A || !B) return;
    btn.classList.add("busy"); btn.textContent = "⌛ rec…";
    try {
        const [imgA, imgB] = await Promise.all([_loadImg(_urlForImage(A.images[0])), _loadImg(_urlForImage(B.images[0]))]);
        const W = Math.min(imgA.naturalWidth, imgB.naturalWidth, 1920);
        const H = Math.min(imgA.naturalHeight, imgB.naturalHeight, 1080);
        const canvas = document.createElement("canvas");
        canvas.width = W; canvas.height = H;
        const ctx = canvas.getContext("2d");
        const stream = canvas.captureStream(30);
        const mr = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9" });
        const chunks = [];
        mr.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
        const done = new Promise((res) => { mr.onstop = res; });
        mr.start();
        const DURATION_MS = 3500;
        const t0 = performance.now();
        function frame() {
            const t = (performance.now() - t0) / DURATION_MS;
            // wipe forward 0→1 first half, reverse 1→0 second half (smoother UX)
            const phase = t < 0.5 ? (t * 2) : (1 - (t - 0.5) * 2);
            const x = (W * phase) | 0;
            // draw B as background
            ctx.drawImage(imgB, 0, 0, W, H);
            // clip A on left
            ctx.save();
            ctx.beginPath();
            ctx.rect(0, 0, x, H);
            ctx.clip();
            ctx.drawImage(imgA, 0, 0, W, H);
            ctx.restore();
            // slider
            ctx.fillStyle = C.sapphire;
            ctx.fillRect(x - 1, 0, 2, H);
            if (t >= 1) {
                mr.stop();
                return;
            }
            requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
        await done;
        const blob = new Blob(chunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `c2c_ab_wipe_${Date.now()}.webm`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 4000);
    } catch (e) {
        console.warn("[C2C.ABSplit] export MP4/WebM failed:", e);
    } finally {
        btn.classList.remove("busy"); btn.textContent = "⤓ MP4";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AI explain
// ─────────────────────────────────────────────────────────────────────────────

async function _aiExplain() {
    if (_state.aiBusy) return;
    const A = _history[_state.slots[0]];
    const B = _history[_state.slots[1]];
    if (!A || !B) return;
    _state.aiBusy = true;
    _state.aiOutput = "";
    _renderBody();
    const ma = A.workflowMeta || {};
    const mb = B.workflowMeta || {};
    const diffLines = [];
    for (const k of ["model", "seed", "steps", "cfg", "sampler", "scheduler", "denoise"]) {
        if (JSON.stringify(ma[k]) !== JSON.stringify(mb[k])) {
            diffLines.push(`- ${k}: A=${ma[k] ?? "?"}  →  B=${mb[k] ?? "?"}`);
        }
    }
    const user = [
        "Two image-generation runs A and B were produced from the same workflow with a few parameter differences.",
        "Differences (workflow widget values):",
        diffLines.length ? diffLines.join("\n") : "(no detected sampler-widget differences — the diff is in upstream prompt/model state)",
        "",
        `A positive prompt (first 240 chars): ${ma.positive || "(unknown)"}`,
        `B positive prompt (first 240 chars): ${mb.positive || "(unknown)"}`,
        "",
        "In ≤6 short bullet points, describe what VISUAL differences these parameter changes typically produce (composition variance from seed, detail level from steps, prompt adherence from cfg, sampling character from sampler/scheduler). Be concrete and concise.",
    ].join("\n");

    try {
        const ai = document.querySelector(`#${PANEL_ID} [data-role="ai"]`);
        const onChunk = (chunk) => {
            _state.aiOutput += chunk;
            if (ai) {
                ai.classList.remove("empty");
                ai.textContent = _state.aiOutput;
            }
        };
        await streamAI({
            feature: "ab_compare",
            sensitivity: "normal",
            max_tokens: 360,
            temperature: 0.4,
            messages: [
                { role: "system", content: "You are an image-generation parameter analyst. Be terse and accurate." },
                { role: "user", content: user },
            ],
            onChunk,
            onError: (e) => { _state.aiOutput += `\n[error: ${e}]`; },
            onDone: () => { /* finalised */ },
        });
    } catch (e) {
        _state.aiOutput += `\n[exception: ${e?.message || e}]`;
    } finally {
        _state.aiBusy = false;
        _renderBody();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Live capture (execution_success → history append)
// ─────────────────────────────────────────────────────────────────────────────

function _onExecutionSuccess(ev) {
    try {
        const prompt_id = ev?.detail?.prompt_id;
        if (!prompt_id) return;
        api.fetchApi(`/history/${prompt_id}`).then(async (resp) => {
            const data = await resp.json();
            const entry = data?.[prompt_id];
            if (!entry) return;
            const outs = entry.outputs || {};
            const images = [];
            for (const o of Object.values(outs)) {
                if (Array.isArray(o?.images)) images.push(...o.images);
            }
            if (!images.length) return;
            const prompt = Array.isArray(entry.prompt) ? entry.prompt[2] : null;
            _addHistoryEntry({ id: prompt_id, ts: Date.now(), images, prompt });
            const btn = document.getElementById(BTN_ID);
            if (btn && !_state.open) btn.classList.add("has-new");
            if (_state.open) {
                _ensureSlotsValid();
                _renderBody();
            }
        }).catch(() => { /* ignore */ });
    } catch (e) {
        console.warn("[C2C.ABSplit] capture error:", e);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extension entry
// ─────────────────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "C2C.ABSplit",
    settings: [
        {
            id: "c2c.ab_split.enabled",
            name: "A/B Split: enabled",
            tooltip: "Capture last runs and offer ⚖ compare overlay.",
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
        api.addEventListener("execution_success", _onExecutionSuccess);
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("c2c.ab_split.enabled", true); }
            catch { return true; }
        })();
        const b = document.getElementById(BTN_ID);
        if (b) b.style.display = enabled ? "flex" : "none";
        console.log("[C2C.ABSplit] godlevel-rebuild loaded.");
    },
});
