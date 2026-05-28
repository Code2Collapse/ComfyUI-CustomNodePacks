/**
 * c2c_undo_panel.js — Visual Undo History (god-level rebuild, 2026-05-27)
 *
 * Captures a debounced snapshot of `graph.serialize()` plus a 96-px canvas
 * thumbnail on every change, then presents a side-scrollable visual stack
 * with diff stats, hover-preview, restore, branch-keeping, named bookmarks,
 * JSON copy/export, and AI explain-change.
 *
 * Features:
 *   1) Snapshot ring buffer (debounced) — settings-driven max
 *   2) Per-snapshot thumbnail of the canvas viewport
 *   3) Diff stats vs previous snapshot (Δnodes, Δlinks)
 *   4) Hover lightbox-style large preview
 *   5) One-click restore (suspends capture during restore)
 *   6) Bookmark / name a snapshot so it isn't evicted by ring rotation
 *   7) Copy snapshot JSON to clipboard / download
 *   8) AI "what changed?" between two snapshots via streamAI
 *   9) Body-only re-render preserves chrome; listener registry; chrome-safe z-index
 */

import { app } from "../../scripts/app.js";
import { attachWindowChrome } from "./_c2c_window.js";
import { streamAI } from "./_c2c_ai_client.js";

const BTN_ID   = "mec-undo-btn";
const PANEL_ID = "mec-undo-panel";
const STYLE_ID = "mec-undo-style";

const _state = {
    open: false,
    stack: [],            // [{ id, ts, snap, node_count, link_count, thumb, name? }]
    bookmarks: new Set(), // ids that should never be evicted
    selA: null,           // for AI compare
    selB: null,
    aiBusy: false,
    aiOutput: "",
};
let _debounce = null;
let _suspend = false;
let _seq = 0;

const _listeners = [];
function _on(t, e, fn, opts) { t.addEventListener(e, fn, opts); _listeners.push(() => t.removeEventListener(e, fn, opts)); }
function _clear() { while (_listeners.length) { try { _listeners.pop()(); } catch {} } }
function _esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c])); }

function _settingsEnabled() { try { return app.ui.settings.getSettingValue("c2c.undo_panel.enabled", true); } catch { return true; } }
function _maxStack()        { try { return Math.max(2, app.ui.settings.getSettingValue("c2c.undo_panel.max", 30)); }      catch { return 30; } }

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
#${BTN_ID} {
    position: fixed; bottom: 386px; right: 16px;
    z-index: var(--c2c-z-hud, 1000);
    width: 38px; height: 38px; border-radius: 50%;
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface1); color: var(--c2c-peach);
    font-size: 14px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex; align-items: center; justify-content: center;
}
#${BTN_ID}:hover { border-color: var(--c2c-peach); }

#${PANEL_ID} {
    position: fixed; top: 80px; right: 80px;
    width: min(94vw, 520px); height: min(82vh, 680px);
    background: var(--c2c-bg2); border: 1px solid var(--c2c-surface1); border-radius: 8px;
    color: var(--c2c-fg); font-family: -apple-system, "Segoe UI", sans-serif; font-size: 12px;
    box-shadow: 0 6px 32px rgba(0,0,0,0.7);
    display: none; flex-direction: column; overflow: hidden;
}
#${PANEL_ID}.visible { display: flex; }
#${PANEL_ID} .ud-header {
    margin: 0; padding: 8px 12px; color: var(--c2c-peach); font-size: 14px;
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(180deg,var(--c2c-bg),var(--c2c-bg2)); border-bottom: 1px solid var(--c2c-surface0);
}
#${PANEL_ID} .ud-close { background: none; border: none; color: var(--c2c-overlay0); cursor: pointer; font-size: 16px; padding: 0 4px; }
#${PANEL_ID} .ud-body { flex: 1 1 auto; min-height: 0; overflow: auto; padding: 10px; }
#${PANEL_ID} .ud-toolbar { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; align-items: center; }
#${PANEL_ID} .ud-toolbar button {
    background: var(--c2c-surface0); border: 1px solid var(--c2c-surface1); color: var(--c2c-fg); border-radius: 4px;
    padding: 3px 8px; cursor: pointer; font-size: 11px;
}
#${PANEL_ID} .ud-toolbar button:hover { border-color: var(--c2c-peach); }
#${PANEL_ID} .ud-toolbar button.active { background: var(--c2c-peach); color: var(--c2c-bg); border-color: var(--c2c-peach); }
#${PANEL_ID} .ud-toolbar .ud-spacer { flex: 1; }
#${PANEL_ID} .ud-stats { color: var(--c2c-overlay0); font-size: 10px; }

#${PANEL_ID} .ud-list { display: flex; flex-direction: column; gap: 6px; }
#${PANEL_ID} .ud-row {
    display: flex; gap: 8px; align-items: center; padding: 6px;
    border: 1px solid var(--c2c-surface0); border-radius: 5px; cursor: pointer;
}
#${PANEL_ID} .ud-row:hover { background: var(--c2c-surface0); border-color: var(--c2c-peach); }
#${PANEL_ID} .ud-row.sel-a { border-color: var(--c2c-blue); box-shadow: inset 0 0 0 1px var(--c2c-blue); }
#${PANEL_ID} .ud-row.sel-b { border-color: var(--c2c-okSoft); box-shadow: inset 0 0 0 1px var(--c2c-okSoft); }
#${PANEL_ID} .ud-thumb {
    width: 64px; height: 48px; flex-shrink: 0; background: var(--c2c-surface0);
    border-radius: 4px; object-fit: cover;
}
#${PANEL_ID} .ud-info { flex: 1; min-width: 0; }
#${PANEL_ID} .ud-name { font-weight: 600; color: var(--c2c-fg); font-size: 11px; }
#${PANEL_ID} .ud-name .ud-bookmark { color: var(--c2c-yellow); }
#${PANEL_ID} .ud-meta { color: var(--c2c-overlay0); font-size: 10px; }
#${PANEL_ID} .ud-delta { color: var(--c2c-okSoft); font-size: 10px; }
#${PANEL_ID} .ud-delta.neg { color: var(--c2c-red); }
#${PANEL_ID} .ud-actions { display: flex; gap: 4px; }
#${PANEL_ID} .ud-actions button {
    background: transparent; border: 1px solid var(--c2c-surface1); border-radius: 4px;
    padding: 2px 5px; cursor: pointer; font-size: 11px; color: var(--c2c-fg);
}
#${PANEL_ID} .ud-actions button:hover { border-color: var(--c2c-peach); }

#${PANEL_ID} .ud-empty { color: var(--c2c-overlay0); text-align: center; font-style: italic; padding: 30px 0; }

#${PANEL_ID} .ud-ai {
    background: var(--c2c-bg3); border: 1px solid var(--c2c-surface0); border-radius: 4px;
    padding: 8px; margin-bottom: 8px; max-height: 200px; overflow: auto;
    white-space: pre-wrap; color: var(--c2c-fg); font-size: 11px;
}

#mec-undo-preview {
    position: fixed; z-index: var(--c2c-z-toast, 99999); pointer-events: none;
    background: var(--c2c-bg); border: 1px solid var(--c2c-peach); border-radius: 6px;
    padding: 4px; box-shadow: 0 6px 24px rgba(0,0,0,0.7); display: none;
}
#mec-undo-preview img { display: block; max-width: 360px; max-height: 240px; }
    `.trim();
    document.head.appendChild(s);
}

function _ensureUi() {
    if (!document.getElementById(BTN_ID)) {
        const b = document.createElement("button");
        b.id = BTN_ID;
        b.title = "Undo history";
        b.textContent = "↶";
        b.addEventListener("click", _toggle);
        document.body.appendChild(b);
    }
    if (!document.getElementById(PANEL_ID)) {
        const p = document.createElement("div");
        p.id = PANEL_ID;
        p.innerHTML = `
            <div class="ud-header"><span>↶ Undo History</span><button class="ud-close">×</button></div>
            <div class="ud-body" data-role="body"></div>
        `;
        document.body.appendChild(p);
        p.querySelector(".ud-close").addEventListener("click", _toggle);
        attachWindowChrome(p, { storageKey: "undo_panel", headerSelector: ".ud-header", titleSelector: ".ud-header > span", minW: 360, minH: 320 });
    }
    if (!document.getElementById("mec-undo-preview")) {
        const pv = document.createElement("div");
        pv.id = "mec-undo-preview";
        pv.innerHTML = `<img>`;
        document.body.appendChild(pv);
    }
}

function _toggle() {
    _state.open = !_state.open;
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    if (_state.open) { p.classList.add("visible"); _render(); }
    else { p.classList.remove("visible"); _clear(); _hidePreview(); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Capture
// ─────────────────────────────────────────────────────────────────────────────

function _thumbnail() {
    try {
        const cv = app.canvas?.canvas;
        if (!cv) return null;
        const W = 128, H = 96;
        const c = document.createElement("canvas");
        c.width = W; c.height = H;
        const ctx = c.getContext("2d");
        ctx.drawImage(cv, 0, 0, cv.width, cv.height, 0, 0, W, H);
        const url = c.toDataURL("image/jpeg", 0.55);
        if (url.length > 90000) return null;
        return url;
    } catch { return null; }
}

function _capture() {
    if (_suspend) return;
    const g = app.graph;
    if (!g || typeof g.serialize !== "function") return;
    let snap;
    try { snap = g.serialize(); } catch { return; }
    const node_count = Array.isArray(snap?.nodes) ? snap.nodes.length : 0;
    const link_count = Array.isArray(snap?.links) ? snap.links.length : 0;
    const entry = { id: ++_seq, ts: Date.now(), snap, node_count, link_count, thumb: _thumbnail() };
    _state.stack.push(entry);
    // Evict oldest non-bookmarked beyond max
    const max = _maxStack();
    while (_state.stack.length > max) {
        const idx = _state.stack.findIndex((e) => !_state.bookmarks.has(e.id));
        if (idx === -1) break;
        _state.stack.splice(idx, 1);
    }
    if (_state.open) _render();
}

function _scheduleCapture() {
    if (!_settingsEnabled()) return;
    clearTimeout(_debounce);
    _debounce = setTimeout(_capture, 600);
}

function _restore(id) {
    const entry = _state.stack.find((e) => e.id === id);
    if (!entry?.snap) return;
    _suspend = true;
    try {
        app.loadGraphData(JSON.parse(JSON.stringify(entry.snap)));
        console.log("[C2C.UndoPanel] Restored snapshot", id);
    } catch (e) { console.warn("[C2C.UndoPanel] restore failed:", e); }
    finally { setTimeout(() => { _suspend = false; }, 1200); }
}

function _hookGraphChanges() {
    const g = app.graph;
    if (!g || g._mecUndoHooked) return;
    g._mecUndoHooked = true;
    const oC = g.onGraphChanged?.bind(g);
    g.onGraphChanged = function () { if (oC) oC.apply(g, arguments); _scheduleCapture(); };
    const oA = g.afterChange?.bind(g);
    g.afterChange = function () { if (oA) oA.apply(g, arguments); _scheduleCapture(); };
    _capture();
}

// ─────────────────────────────────────────────────────────────────────────────
// Render
// ─────────────────────────────────────────────────────────────────────────────

function _render() {
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    const body = p.querySelector('[data-role="body"]');
    if (!body) return;
    _clear();
    body.innerHTML = `
        <div class="ud-toolbar">
            <button data-act="snap">📸 Snap now</button>
            <button data-act="clear">🗑 Clear unmarked</button>
            <button data-act="compare" ${_state.selA && _state.selB ? "" : "disabled"}>🤖 AI compare A→B</button>
            <span class="ud-spacer"></span>
            <span class="ud-stats">${_state.stack.length} snapshot(s)</span>
        </div>
        <div data-role="ai"></div>
        <div class="ud-list" data-role="list"></div>
    `;
    _on(body.querySelector('[data-act="snap"]'), "click", () => { _capture(); });
    _on(body.querySelector('[data-act="clear"]'), "click", () => {
        _state.stack = _state.stack.filter((e) => _state.bookmarks.has(e.id));
        _state.selA = _state.selB = null;
        _render();
    });
    _on(body.querySelector('[data-act="compare"]'), "click", () => _aiCompare(body));
    _renderAi(body);
    _renderList(body);
}

function _renderAi(body) {
    const slot = body.querySelector('[data-role="ai"]');
    if (!slot) return;
    if (!_state.aiOutput && !_state.aiBusy) { slot.innerHTML = ""; return; }
    slot.innerHTML = `<div class="ud-ai">${_state.aiBusy ? "AI analysing diff…\n\n" : ""}${_esc(_state.aiOutput || "")}</div>`;
}

function _renderList(body) {
    const list = body.querySelector('[data-role="list"]');
    if (!list) return;
    if (!_state.stack.length) {
        list.innerHTML = `<div class="ud-empty">No snapshots yet. Make any change to the graph.</div>`;
        return;
    }
    const items = _state.stack.slice().reverse();
    list.innerHTML = items.map((e, i) => {
        const realIdx = _state.stack.length - 1 - i;
        const prev = _state.stack[realIdx - 1];
        const dn = prev ? (e.node_count - prev.node_count) : 0;
        const dl = prev ? (e.link_count - prev.link_count) : 0;
        const bookmarked = _state.bookmarks.has(e.id);
        const aMark = _state.selA === e.id ? "sel-a" : "";
        const bMark = _state.selB === e.id ? "sel-b" : "";
        return `
            <div class="ud-row ${aMark} ${bMark}" data-id="${e.id}">
                ${e.thumb ? `<img class="ud-thumb" data-thumb="${_esc(e.thumb)}" src="${_esc(e.thumb)}">` : `<div class="ud-thumb"></div>`}
                <div class="ud-info">
                    <div class="ud-name">${bookmarked ? `<span class="ud-bookmark">★</span> ` : ""}${_esc(e.name || `Snapshot #${e.id}`)}</div>
                    <div class="ud-meta">${new Date(e.ts).toLocaleTimeString()} · ${e.node_count} nodes · ${e.link_count} links</div>
                    ${prev ? `<div class="ud-delta ${dn<0||dl<0?"neg":""}">Δ ${dn>=0?"+":""}${dn} nodes, ${dl>=0?"+":""}${dl} links</div>` : ""}
                </div>
                <div class="ud-actions">
                    <button data-act="restore" data-id="${e.id}" title="Restore">↶</button>
                    <button data-act="selA" data-id="${e.id}" title="Mark as A (compare)">A</button>
                    <button data-act="selB" data-id="${e.id}" title="Mark as B (compare)">B</button>
                    <button data-act="bookmark" data-id="${e.id}" title="Bookmark">${bookmarked ? "★" : "☆"}</button>
                    <button data-act="rename" data-id="${e.id}" title="Rename">✎</button>
                    <button data-act="copy" data-id="${e.id}" title="Copy JSON">📋</button>
                    <button data-act="download" data-id="${e.id}" title="Download JSON">⇩</button>
                </div>
            </div>
        `;
    }).join("");
    list.querySelectorAll(".ud-row").forEach((el) => {
        _on(el, "click", (ev) => {
            const t = ev.target;
            const act = t.getAttribute?.("data-act");
            const id = parseInt(t.getAttribute?.("data-id") || el.getAttribute("data-id"), 10);
            if (act) ev.stopPropagation();
            switch (act) {
                case "restore": _restore(id); break;
                case "selA": _state.selA = id; _render(); break;
                case "selB": _state.selB = id; _render(); break;
                case "bookmark": _toggleBookmark(id); _render(); break;
                case "rename": _renameSnapshot(id); break;
                case "copy": _copyJson(id); break;
                case "download": _downloadJson(id); break;
                default: _restore(id);
            }
        });
        const img = el.querySelector("img.ud-thumb");
        if (img) {
            _on(img, "mousemove", (ev) => _showPreview(ev, img.getAttribute("data-thumb")));
            _on(img, "mouseleave", _hidePreview);
        }
    });
}

function _toggleBookmark(id) {
    if (_state.bookmarks.has(id)) _state.bookmarks.delete(id);
    else _state.bookmarks.add(id);
}

function _renameSnapshot(id) {
    const e = _state.stack.find((x) => x.id === id);
    if (!e) return;
    const n = prompt("Snapshot name:", e.name || `Snapshot #${id}`);
    if (n === null) return;
    e.name = n;
    _render();
}

function _copyJson(id) {
    const e = _state.stack.find((x) => x.id === id);
    if (!e) return;
    navigator.clipboard.writeText(JSON.stringify(e.snap, null, 2)).then(
        () => console.log("[C2C.UndoPanel] copied snapshot", id),
        () => alert("Clipboard write blocked.")
    );
}

function _downloadJson(id) {
    const e = _state.stack.find((x) => x.id === id);
    if (!e) return;
    const f = new Blob([JSON.stringify(e.snap, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(f);
    a.download = `snapshot_${id}_${Date.now()}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function _showPreview(ev, src) {
    if (!src) return;
    const pv = document.getElementById("mec-undo-preview");
    if (!pv) return;
    pv.querySelector("img").src = src;
    pv.style.display = "block";
    const x = Math.min(window.innerWidth - 380, ev.clientX + 16);
    const y = Math.min(window.innerHeight - 260, ev.clientY + 16);
    pv.style.left = x + "px";
    pv.style.top = y + "px";
}
function _hidePreview() {
    const pv = document.getElementById("mec-undo-preview");
    if (pv) pv.style.display = "none";
}

// ─────────────────────────────────────────────────────────────────────────────
// AI compare
// ─────────────────────────────────────────────────────────────────────────────

function _diffSummary(a, b) {
    const aTypes = new Map(); const bTypes = new Map();
    for (const n of (a.snap?.nodes || [])) aTypes.set(n.type, (aTypes.get(n.type) || 0) + 1);
    for (const n of (b.snap?.nodes || [])) bTypes.set(n.type, (bTypes.get(n.type) || 0) + 1);
    const added = [], removed = [];
    for (const [t, c] of bTypes) { const d = c - (aTypes.get(t) || 0); if (d > 0) added.push(`${t} x${d}`); }
    for (const [t, c] of aTypes) { const d = c - (bTypes.get(t) || 0); if (d > 0) removed.push(`${t} x${d}`); }
    return {
        a: { nodes: a.node_count, links: a.link_count, ts: a.ts },
        b: { nodes: b.node_count, links: b.link_count, ts: b.ts },
        added, removed,
    };
}

async function _aiCompare(body) {
    if (_state.aiBusy) return;
    const a = _state.stack.find((e) => e.id === _state.selA);
    const b = _state.stack.find((e) => e.id === _state.selB);
    if (!a || !b) { alert("Pick both A and B."); return; }
    const diff = _diffSummary(a, b);
    _state.aiBusy = true; _state.aiOutput = "";
    _renderAi(body);
    try {
        await streamAI({
            feature: "undo_diff",
            sensitivity: "normal",
            max_tokens: 500,
            temperature: 0.4,
            messages: [
                { role: "system", content: "You explain ComfyUI workflow diffs concisely. 4–8 sentences. Focus on intent (what user likely did) and risk (unintended side-effects)." },
                { role: "user", content: `Diff JSON:\n${JSON.stringify(diff, null, 2)}` },
            ],
            onChunk: (c) => { _state.aiOutput += c; _renderAi(body); },
            onError: (e) => { _state.aiOutput += `\n[error] ${e?.message || e}`; },
            onDone: () => {},
        });
    } catch (e) {
        _state.aiOutput += `\n[error] ${e?.message || e}`;
    } finally {
        _state.aiBusy = false;
        _renderAi(body);
    }
}

app.registerExtension({
    name: "C2C.UndoPanel",
    settings: [
        { id: "c2c.undo_panel.enabled", name: "Undo Panel: capture graph snapshots", type: "boolean", default: true },
        { id: "c2c.undo_panel.max",     name: "Undo Panel: max snapshots",            type: "number",  default: 30 },
    ],
    async setup() {
        _injectStyle();
        _ensureUi();
        const waitForGraph = setInterval(() => {
            if (app.graph) { _hookGraphChanges(); clearInterval(waitForGraph); }
        }, 200);
        const enabled = _settingsEnabled();
        const b = document.getElementById(BTN_ID);
        if (b) b.style.display = enabled ? "flex" : "none";
        console.log("[C2C.UndoPanel] godlevel-rebuild loaded.");
    },
});
