/**
 * mec_node_explain.js
 * "What does this node do?" — hover node title → floating explanation card.
 *
 * Behaviour:
 *   • Hover the title bar of any node for 800 ms → fetch /mec/node_explain/{class}
 *   • Floating popover appears above (or below) the node
 *   • Mouse can move onto the popover to scroll/read — it stays open
 *   • Both title-leave and popover-leave trigger a 150 ms hide timer
 *   • Cache: Map<className, data> — cleared on page reload
 *
 * Settings:
 *   mec.node_explain.backend   — auto | api | gguf | off  (default: auto)
 *   mec.node_explain.gguf_quant — Q4_K_M | Q5_K_M | Q8_0  (default: Q4_K_M)
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";  // kept for future use
import { capabilityFor, nodeColor, lighten } from "./c2c_node_taxonomy.js";

// ── constants ──────────────────────────────────────────────────────────────
const DEFAULT_DWELL_MS = 250;   // how long to hover before showing tooltip
                                // (was 800 — user reported too sluggish)
const HIDE_MS    = 120;   // delay before hiding after mouse leaves
const TITLE_H    = 30;    // LiteGraph default title bar height in graph-coords
const TRIGGER_W  = 22;    // graph-coord px — only the top-left corner of the
                          // header triggers the popover. Rest of the title bar
                          // stays draggable, so moving nodes is unobstructed.
const POPOVER_W  = 280;   // popover width px (Iter-4f: shrunk from 360 to free up canvas)
const POPOVER_ID = "mec-node-explain-popover";
const STYLE_ID   = "mec-node-explain-style";

// ── session cache ──────────────────────────────────────────────────────────
const _CACHE = new Map();   // Map<className, responseData>
const _PENDING = new Set(); // classes with a request in-flight

// ── state ──────────────────────────────────────────────────────────────────
let _dwellTimer   = null;
let _hideTimer    = null;
let _currentNode  = null;   // node whose title corner is under cursor
let _currentSlot  = null;   // {node, isInput, slotIndex} or null
let _currentWidgetKey = null;  // panel-mode: "<nodeId>|<widgetIdx>" or null
let _popoverNode  = null;   // node the open popover belongs to (header mode)
let _popoverSlot  = null;   // {node,isInput,slotIndex} when popover is in slot mode
let _nodeHovered  = false;
let _popoverHovered = false;
let _rafHandle    = null;   // P0.11 — re-anchor RAF while popover is visible
let _lastDs       = { sx: 0, ox: 0, oy: 0, nx: 0, ny: 0, nw: 0, gid: null, sk: "" };

const SLOT_RADIUS = 10;     // graph-coord radius around a slot dot considered
                            // a hit. LiteGraph draws slot circles ~4 px wide;
                            // we use a generous 10 to make hover forgiving.

// ── helpers ────────────────────────────────────────────────────────────────
function _getSetting(id, def) {
    try {
        return app.ui.settings.getSettingValue(id, def);
    } catch (_) {
        return def;
    }
}

/** Convert canvas event coords to graph-space coords.
 *  LiteGraph transform: screen = (graph + offset) * scale
 *  Inverse:             graph  =  screen/scale - offset
 *  (Previously this used `(screen - offset)/scale` which only matches at
 *  scale=1; at any other zoom the slot hit-test missed by ~slot-radius.) */
function _toGraph(e) {
    const canvas = app.canvas;
    const rect   = canvas.canvas.getBoundingClientRect();
    const ds     = canvas.ds;
    const gx = (e.clientX - rect.left) / ds.scale - ds.offset[0];
    const gy = (e.clientY - rect.top ) / ds.scale - ds.offset[1];
    return { gx, gy };
}

/** Find the node whose top-left header CORNER contains (gx, gy).
 *  Only the leftmost TRIGGER_W × TITLE_H pixels of the title bar are a hit
 *  zone — the rest of the bar stays free for dragging. */
function _nodeAtTitle(gx, gy) {
    // Subgraph-aware: when the user has drilled into a subgraph,
    // `app.canvas.graph` is that inner LGraph, not `app.graph` (which
    // remains the root). Hit-testing must use the actually-rendered
    // graph or the click would resolve to nodes the user can't even
    // see.
    const g = app.canvas?.graph || app.graph;
    const nodes = g?._nodes;
    if (!nodes) return null;
    // Iterate in reverse so topmost (highest index) node wins
    for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        if (!n || n.flags?.collapsed) continue;
        const nx = n.pos[0];
        const ny = n.pos[1];
        // Top-left corner of the title bar only.
        const titleTop    = ny - TITLE_H;
        const titleBottom = ny;
        const triggerRight = nx + Math.min(TRIGGER_W, n.size[0] || TRIGGER_W);
        if (gx >= nx && gx <= triggerRight && gy >= titleTop && gy <= titleBottom) {
            return n;
        }
    }
    return null;
}
/** Return the slot at (gx, gy) in graph space, or null.
 *  Uses LiteGraph's `node.getConnectionPos(is_input, slot_index)` for an
 *  accurate per-slot anchor (handles vertically-laid-out nodes, hidden
 *  slots, custom shapes, etc.). Falls back to a 20 px grid when that
 *  method isn't present. Works for any node — native or custom. */
function _slotHitTest(gx, gy) {
    const g = app.canvas?.graph || app.graph;
    const nodes = g?._nodes;
    if (!nodes) return null;
    for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        if (!n || n.flags?.collapsed) continue;
        // AABB reject around the node's full footprint plus a margin so the
        // slot-dot's left/right overhang is still counted.
        const nx = n.pos[0], ny = n.pos[1];
        const nw = n.size?.[0] || 0, nh = n.size?.[1] || 0;
        if (gx < nx - SLOT_RADIUS || gx > nx + nw + SLOT_RADIUS) continue;
        if (gy < ny - SLOT_RADIUS || gy > ny + nh + SLOT_RADIUS) continue;

        const r2 = SLOT_RADIUS * SLOT_RADIUS;
        const fallback = (isInput, idx) => [
            isInput ? nx : nx + nw,
            ny + 20 * (idx + 0.5),
        ];
        const ins  = n.inputs  || [];
        const outs = n.outputs || [];
        for (let j = 0; j < ins.length; j++) {
            const p = n.getConnectionPos ? n.getConnectionPos(true, j) : fallback(true, j);
            const dx = gx - p[0], dy = gy - p[1];
            if (dx*dx + dy*dy <= r2) return { node: n, isInput: true, slotIndex: j };
        }
        for (let j = 0; j < outs.length; j++) {
            const p = n.getConnectionPos ? n.getConnectionPos(false, j) : fallback(false, j);
            const dx = gx - p[0], dy = gy - p[1];
            if (dx*dx + dy*dy <= r2) return { node: n, isInput: false, slotIndex: j };
        }
    }
    return null;
}
// ── popover DOM ────────────────────────────────────────────────────────────
function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id    = STYLE_ID;
    style.textContent = `
#${POPOVER_ID} {
    position: absolute;
    z-index: var(--c2c-z-popover, 9000);
    width: ${POPOVER_W}px;
    max-height: 380px;
    overflow-y: auto;
    background: var(--c2c-bg);
    border: 1px solid var(--c2c-gray700);
    border-radius: 6px;
    padding: 8px 10px;
    box-shadow: 0 6px 24px color-mix(in srgb, var(--c2c-shadowBase) 65%, transparent);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px;
    line-height: 1.4;
    color: var(--c2c-fg);
    pointer-events: auto;
    display: none;
}
#${POPOVER_ID}.visible { display: block; }
#${POPOVER_ID} .mec-ne-headline {
    font-size: 13px;
    font-weight: 700;
    color: var(--c2c-blue);
    margin-bottom: 3px;
}
#${POPOVER_ID} .mec-ne-hint {
    margin: 4px 0 6px;
    padding: 5px 7px;
    background: var(--c2c-bg2);
    border-left: 2px solid var(--c2c-blue);
    border-radius: 3px;
    color: var(--c2c-subtext1);
    font-size: 11px;
}
#${POPOVER_ID} .mec-ne-suggest {
    margin-top: 8px;
}
#${POPOVER_ID} .mec-ne-suggest .mec-ne-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 0;
    border-bottom: 1px dotted var(--c2c-surface0);
}
#${POPOVER_ID} .mec-ne-suggest .mec-ne-row:last-child { border-bottom: none; }
#${POPOVER_ID} .mec-ne-suggest .mec-ne-rowtext {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
}
#${POPOVER_ID} .mec-ne-suggest .mec-ne-rowtext code { color:var(--c2c-yellow); font-size:10.5px; }
#${POPOVER_ID} .mec-ne-suggest .mec-ne-confbar {
    flex: 0 0 32px;
    height: 4px;
    background: var(--c2c-surface0);
    border-radius: 2px;
    overflow: hidden;
}
#${POPOVER_ID} .mec-ne-suggest .mec-ne-confbar > i {
    display:block; height:100%; background:var(--c2c-blue);
}
#${POPOVER_ID} .mec-ne-suggest button.mec-ne-insert {
    flex: 0 0 auto;
    font-size: 10px;
    padding: 2px 7px;
    background: var(--c2c-blueBg);
    color: var(--c2c-sky);
    border: 1px solid var(--c2c-surface1);
    border-radius: 3px;
    cursor: pointer;
}
#${POPOVER_ID} .mec-ne-suggest button.mec-ne-insert:hover { background:var(--c2c-surface1); }
#${POPOVER_ID} .mec-ne-tip {
    margin-top: 6px;
    font-size: 10px;
    color: var(--c2c-overlay0);
    font-style: italic;
}
#${POPOVER_ID} .mec-ne-kv {
    display:grid; grid-template-columns: 64px 1fr; gap:1px 8px; font-size:11px;
}
#${POPOVER_ID} .mec-ne-kv .k { color:var(--c2c-okSoft); }
#${POPOVER_ID} .mec-ne-kv .v { color:var(--c2c-fg); word-break:break-word; }
#${POPOVER_ID} .mec-ne-purpose {
    margin-bottom: 8px;
    color: var(--c2c-fg);
}
#${POPOVER_ID} .mec-ne-section-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--c2c-overlay0);
    margin: 8px 0 4px;
}
#${POPOVER_ID} table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}
#${POPOVER_ID} table td {
    padding: 2px 6px 2px 0;
    vertical-align: top;
}
#${POPOVER_ID} table td:first-child {
    font-weight: 600;
    color: var(--c2c-okSoft);
    white-space: nowrap;
    width: 30%;
}
#${POPOVER_ID} table.mec-ne-slots th {
    padding: 2px 6px 2px 0;
    font-size: 10px;
    text-transform: uppercase;
    color: var(--c2c-overlay0);
    text-align: left;
    border-bottom: 1px solid var(--c2c-surface0);
}
#${POPOVER_ID} table.mec-ne-slots td {
    color: var(--c2c-fg);
    width: auto;
    font-weight: 400;
}
#${POPOVER_ID} table.mec-ne-slots td:first-child {
    color: var(--c2c-teal);
    font-weight: 500;
    width: auto;
}
#${POPOVER_ID} table.mec-ne-slots code {
    color: var(--c2c-yellow);
    font-size: 11px;
}
#${POPOVER_ID} .mec-ne-muted {
    color: var(--c2c-overlay0);
}
#${POPOVER_ID} .mec-ne-footer {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--c2c-surface0);
    font-size: 11px;
    color: var(--c2c-overlay0);
}
#${POPOVER_ID} .mec-ne-badge {
    padding: 1px 7px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    background: var(--c2c-surface0);
    color: var(--c2c-blue);
    white-space: nowrap;
}
#${POPOVER_ID} .mec-ne-badge.tier-cloud  { background:var(--c2c-blueBg); color:var(--c2c-sky); }
#${POPOVER_ID} .mec-ne-badge.tier-gguf   { background:var(--c2c-okBgDark3); color:var(--c2c-okSoft); }
#${POPOVER_ID} .mec-ne-badge.tier-det    { background:var(--c2c-warnBg4); color:var(--c2c-peach); }
#${POPOVER_ID} .mec-ne-loading {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    color: var(--c2c-overlay0);
}
#${POPOVER_ID} .mec-ne-spinner {
    width: 18px;
    height: 18px;
    border: 2px solid var(--c2c-surface0);
    border-top-color: var(--c2c-blue);
    border-radius: 50%;
    animation: mec-spin 0.7s linear infinite;
    flex-shrink: 0;
}
@keyframes mec-spin {
    to { transform: rotate(360deg); }
}
#${POPOVER_ID} .mec-ne-error {
    color: var(--c2c-red);
    font-size: 12px;
}

/* ── Panel mode (fixed inspector dock) ──────────────────────────────── */
#${POPOVER_ID}.c2c-ne-panel-mode {
    position: fixed;
    display: flex !important;
    flex-direction: column;
    left: auto !important;
    top: auto !important;
    right: var(--c2c-insp-right, 16px);
    bottom: var(--c2c-insp-bottom, 96px);
    width: var(--c2c-insp-w, 300px);
    height: var(--c2c-insp-h, 380px);
    max-height: none;
    padding: 0;
    overflow: hidden;
    resize: none;
}
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-panel-head {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 7px;
    background: var(--c2c-bg2);
    border-bottom: 1px solid var(--c2c-surface0);
    cursor: move; user-select: none;
    flex: 0 0 auto;
}
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-panel-grip {
    color:var(--c2c-surface2); font-size:11px; letter-spacing:-1px; cursor:move;
}
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-panel-title {
    font-weight:700; color:var(--c2c-blue); font-size:11px;
    text-transform:uppercase; letter-spacing:0.06em; flex:0 0 auto;
}
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-panel-ctx {
    flex:1 1 auto; color:var(--c2c-sub); font-size:11px;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    text-align:right; font-style:italic;
}
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-panel-collapse {
    flex:0 0 auto; width:22px; height:18px; padding:0;
    background:var(--c2c-bg); color:var(--c2c-fg);
    border:1px solid var(--c2c-surface1); border-radius:3px; cursor:pointer;
    font-size:12px; line-height:14px; text-align:center;
}
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-panel-collapse:hover { background:var(--c2c-surface0); }
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-panel-body {
    flex: 1 1 auto;
    overflow-y: auto;
    padding: 8px 10px;
}
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-panel-body:empty::before {
    content: "Hover a slot, widget, or node to inspect it. Drag header to move \\2022 drag any edge or corner to resize.";
    color: var(--c2c-overlay0); font-size: 11px; font-style: italic;
    display: block; padding: 12px 4px;
}
/* 4-sided resize: 4 edges + 4 corners, slim strips on the panel border */
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-edge {
    position: absolute; background: transparent; z-index: 5;
}
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-edge.n  { top:0;    left:8px;   right:8px;  height:5px; cursor: ns-resize; }
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-edge.s  { bottom:0; left:8px;   right:8px;  height:5px; cursor: ns-resize; }
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-edge.e  { right:0;  top:8px;    bottom:8px; width:5px;  cursor: ew-resize; }
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-edge.w  { left:0;   top:8px;    bottom:8px; width:5px;  cursor: ew-resize; }
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-edge.nw { top:0;    left:0;     width:10px; height:10px; cursor: nwse-resize; }
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-edge.ne { top:0;    right:0;    width:10px; height:10px; cursor: nesw-resize; }
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-edge.sw { bottom:0; left:0;     width:10px; height:10px; cursor: nesw-resize; }
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-edge.se { bottom:0; right:0;    width:10px; height:10px; cursor: nwse-resize; }
#${POPOVER_ID}.c2c-ne-panel-mode .c2c-ne-edge:hover { background: rgba(137,180,250,0.18); }
#${POPOVER_ID}.c2c-ne-panel-mode[data-collapsed="1"] {
    height: auto !important;
}
#${POPOVER_ID}.c2c-ne-panel-mode[data-collapsed="1"] .c2c-ne-panel-body,
#${POPOVER_ID}.c2c-ne-panel-mode[data-collapsed="1"] .c2c-ne-edge.s,
#${POPOVER_ID}.c2c-ne-panel-mode[data-collapsed="1"] .c2c-ne-edge.sw,
#${POPOVER_ID}.c2c-ne-panel-mode[data-collapsed="1"] .c2c-ne-edge.se,
#${POPOVER_ID}.c2c-ne-panel-mode[data-collapsed="1"] .c2c-ne-edge.e,
#${POPOVER_ID}.c2c-ne-panel-mode[data-collapsed="1"] .c2c-ne-edge.w { display: none; }
/* Per-widget hover table inside the panel body */
#${POPOVER_ID} table.mec-ne-opts td { padding: 2px 8px 2px 0; vertical-align: top; }
#${POPOVER_ID} table.mec-ne-opts td:first-child { color:var(--c2c-yellow); font-weight:600; width: 28%; }
#${POPOVER_ID} table.mec-ne-opts tr.cur td { background: rgba(137,180,250,0.18); }
    `.trim();
    document.head.appendChild(style);
}

function _getPopover() {
    let el = document.getElementById(POPOVER_ID);
    if (!el) {
        el = document.createElement("div");
        el.id = POPOVER_ID;
        document.body.appendChild(el);

        el.addEventListener("mouseenter", () => {
            _popoverHovered = true;
            _clearHide();
        });
        el.addEventListener("mouseleave", () => {
            _popoverHovered = false;
            if (!_panelMode()) _scheduleHide();
        });
    }
    if (_panelMode()) _enablePanelMode(el);
    else _disablePanelMode(el);
    return el;
}

// ── panel mode (fixed inspector dock) ──────────────────────────────────────
const _PANEL_GEOM_KEY = "c2c.inspector.geom.v1";
function _panelMode() {
    return _getSetting("mec.node_explain.surface", "panel") === "panel";
}
function _innerTarget(el) {
    return el.classList.contains("c2c-ne-panel-mode")
        ? el.querySelector(".c2c-ne-panel-body") : el;
}
function _setPanelCtx(label) {
    const el = document.getElementById(POPOVER_ID);
    const ctx = el?.querySelector(".c2c-ne-panel-ctx");
    if (ctx) ctx.textContent = label || "";
}
function _loadGeom() {
    try { return JSON.parse(localStorage.getItem(_PANEL_GEOM_KEY)) || {}; }
    catch (_) { return {}; }
}
function _saveGeom(g) {
    try { localStorage.setItem(_PANEL_GEOM_KEY, JSON.stringify(g)); } catch (_) {}
}
function _applyGeom(el) {
    const g = _loadGeom();
    const s = el.style;
    if (typeof g.right  === "number") s.setProperty("--c2c-insp-right",  g.right  + "px");
    if (typeof g.bottom === "number") s.setProperty("--c2c-insp-bottom", g.bottom + "px");
    if (typeof g.w === "number") s.setProperty("--c2c-insp-w", Math.max(220, g.w) + "px");
    if (typeof g.h === "number") s.setProperty("--c2c-insp-h", Math.max(160, g.h) + "px");
    el.dataset.collapsed = g.collapsed ? "1" : "0";
}
function _disablePanelMode(el) {
    if (!el.classList.contains("c2c-ne-panel-mode")) return;
    el.removeAttribute("data-c2c-dock");
    el.classList.remove("c2c-ne-panel-mode", "visible");
    // Restore native innerHTML setter (delete the per-instance shadow)
    try { delete el.innerHTML; } catch (_) {}
    el.innerHTML = "";
    el._c2cPanelized = false;
}
function _enablePanelMode(el) {
    if (el._c2cPanelized) return;
    el._c2cPanelized = true;
    el.setAttribute("data-c2c-dock", "inspector");
    el.classList.add("c2c-ne-panel-mode", "visible");
    // Build chrome: head + body + 8 resize edges/corners.
    // Use the prototype setter ONCE, before redirecting innerHTML on this instance.
    const nativeSet = Object.getOwnPropertyDescriptor(Element.prototype, "innerHTML").set;
    nativeSet.call(el,
        `<header class="c2c-ne-panel-head">
            <span class="c2c-ne-panel-grip" title="Drag to move">\u22ee\u22ee</span>
            <span class="c2c-ne-panel-title">Inspector <span class="c2c-ne-panel-shortcut">(Ctrl+Shift+I)</span></span>
            <span class="c2c-ne-panel-ctx"></span>
            <button class="c2c-ne-panel-collapse" title="Collapse / expand">\u2013</button>
         </header>
         <div class="c2c-ne-panel-body"></div>
         <div class="c2c-ne-edge n"  data-dir="N"  title="Resize"></div>
         <div class="c2c-ne-edge s"  data-dir="S"  title="Resize"></div>
         <div class="c2c-ne-edge e"  data-dir="E"  title="Resize"></div>
         <div class="c2c-ne-edge w"  data-dir="W"  title="Resize"></div>
         <div class="c2c-ne-edge nw" data-dir="NW" title="Resize"></div>
         <div class="c2c-ne-edge ne" data-dir="NE" title="Resize"></div>
         <div class="c2c-ne-edge sw" data-dir="SW" title="Resize"></div>
         <div class="c2c-ne-edge se" data-dir="SE" title="Resize"></div>`);
    const bodyEl  = el.querySelector(".c2c-ne-panel-body");
    const headEl  = el.querySelector(".c2c-ne-panel-head");
    const collapseBtn = el.querySelector(".c2c-ne-panel-collapse");

    // Redirect future el.innerHTML writes to the body so existing renderers
    // don't need any changes — they keep doing `el.innerHTML = ...` and the
    // panel chrome stays intact.
    Object.defineProperty(el, "innerHTML", {
        configurable: true,
        get() { return bodyEl.innerHTML; },
        set(html) { bodyEl.innerHTML = html; },
    });

    _applyGeom(el);

    // ── drag (header) ──
    const startDrag = (e) => {
        if (e.target.closest(".c2c-ne-panel-collapse")) return;
        if (e.target.closest(".c2c-ne-edge"))           return;
        e.preventDefault();
        const startX = e.clientX, startY = e.clientY;
        const cs = getComputedStyle(el);
        const startRight  = parseFloat(cs.right)  || 16;
        const startBottom = parseFloat(cs.bottom) || 96;
        const onMove = (ev) => {
            const dx = ev.clientX - startX, dy = ev.clientY - startY;
            const w = el.offsetWidth, h = el.offsetHeight;
            let r = startRight - dx, b = startBottom - dy;
            r = Math.max(0, Math.min(window.innerWidth  - w, r));
            // bottom-anchored: clamp so top = innerHeight - bottom - h >= 60.
            b = Math.max(0, Math.min(window.innerHeight - h - 60, b));
            el.style.setProperty("--c2c-insp-right",  r + "px");
            el.style.setProperty("--c2c-insp-bottom", b + "px");
        };
        const onUp = () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup",   onUp);
            const g = _loadGeom();
            g.right  = parseFloat(getComputedStyle(el).right);
            g.bottom = parseFloat(getComputedStyle(el).bottom);
            _saveGeom(g);
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup",   onUp);
    };
    headEl.addEventListener("mousedown", startDrag);

    // ── resize (any edge or corner; panel anchored bottom-right) ──
    const attachResize = (edgeEl, dir) => {
        const hasN = dir.includes("N"), hasS = dir.includes("S");
        const hasE = dir.includes("E"), hasW = dir.includes("W");
        edgeEl.addEventListener("mousedown", (e) => {
            e.preventDefault(); e.stopPropagation();
            const startX = e.clientX, startY = e.clientY;
            const cs = getComputedStyle(el);
            const startW = el.offsetWidth, startH = el.offsetHeight;
            const startR = parseFloat(cs.right)  || 16;
            const startB = parseFloat(cs.bottom) || 96;
            const onMove = (ev) => {
                const dx = ev.clientX - startX;
                const dy = ev.clientY - startY;
                let w = startW, h = startH, r = startR, b = startB;
                // East edge drags right edge: keep LEFT edge in place.
                //   left = right + w  (in viewport-right coords)
                //   want left constant => (startR + startW) === (r + w)
                //   right edge moves dx right => r decreases by dx; w grows by dx.
                if (hasE) { w = startW + dx; r = startR - dx; }
                // West edge drags left edge: keep RIGHT edge in place.
                //   right unchanged; w shrinks by dx.
                if (hasW) { w = startW - dx; }
                // South edge drags bottom edge: keep TOP edge in place.
                //   top = bottom + h; want top constant => (startB + startH) === (b + h)
                //   bottom edge moves dy down => b decreases by dy; h grows by dy.
                if (hasS) { h = startH + dy; b = startB - dy; }
                // North edge drags top edge: keep BOTTOM edge in place.
                //   bottom unchanged; h shrinks by dy.
                if (hasN) { h = startH - dy; }
                w = Math.max(220, Math.min(900, w));
                h = Math.max(120, Math.min(window.innerHeight - 80, h));
                r = Math.max(0, Math.min(window.innerWidth  - w, r));
                // bottom-anchored: top = innerHeight - bottom - h must >= 60.
                b = Math.max(0, Math.min(window.innerHeight - h - 60, b));
                el.style.setProperty("--c2c-insp-w", w + "px");
                el.style.setProperty("--c2c-insp-h", h + "px");
                el.style.setProperty("--c2c-insp-right",  r + "px");
                el.style.setProperty("--c2c-insp-bottom", b + "px");
            };
            const onUp = () => {
                window.removeEventListener("mousemove", onMove);
                window.removeEventListener("mouseup",   onUp);
                const cs2 = getComputedStyle(el);
                const g = _loadGeom();
                g.w = el.offsetWidth; g.h = el.offsetHeight;
                g.right = parseFloat(cs2.right); g.bottom = parseFloat(cs2.bottom);
                _saveGeom(g);
            };
            window.addEventListener("mousemove", onMove);
            window.addEventListener("mouseup",   onUp);
        });
    };
    el.querySelectorAll(".c2c-ne-edge").forEach(eg => attachResize(eg, eg.dataset.dir));

    // ── collapse ──
    collapseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const wasCollapsed = el.dataset.collapsed === "1";
        const now = wasCollapsed ? "0" : "1";
        el.dataset.collapsed = now;
        collapseBtn.textContent = now === "1" ? "+" : "\u2013";
        const g = _loadGeom(); g.collapsed = now === "1"; _saveGeom(g);
        // EXPAND clamp: bottom-anchored, ensure top >= 60 after layout.
        if (wasCollapsed) {
            requestAnimationFrame(() => {
                const h = el.offsetHeight, w = el.offsetWidth;
                const cs = getComputedStyle(el);
                let r = parseFloat(cs.right)  || 16;
                let b = parseFloat(cs.bottom) || 96;
                r = Math.max(0, Math.min(window.innerWidth  - w,     r));
                b = Math.max(0, Math.min(window.innerHeight - h - 60, b));
                el.style.setProperty("--c2c-insp-right",  r + "px");
                el.style.setProperty("--c2c-insp-bottom", b + "px");
                const gg = _loadGeom(); gg.right = r; gg.bottom = b; _saveGeom(gg);
            });
        }
    });
    if (el.dataset.collapsed === "1") collapseBtn.textContent = "+";
}

/** Hit-test the FULL body of a node (any pixel inside its bounding box).
 *  Used in panel mode to make the inspector follow the cursor across the
 *  whole node, not just the title corner. */
function _nodeAtBody(gx, gy) {
    const g = app.canvas?.graph || app.graph;
    const nodes = g?._nodes;
    if (!nodes) return null;
    for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        if (!n || n.flags?.collapsed) continue;
        const nx = n.pos[0], ny = n.pos[1];
        const nw = n.size?.[0] || 0, nh = n.size?.[1] || 0;
        // Include the title bar above the body.
        if (gx >= nx && gx <= nx + nw && gy >= ny - TITLE_H && gy <= ny + nh) return n;
    }
    return null;
}

/** Hit-test which widget on a node the cursor is over.
 *  Robust to DOM widgets (textareas etc.) that don't set last_y reliably:
 *  - primary path: LiteGraph last_y + computed height + 2px tolerance
 *  - fallback path: when NO widget has a last_y (rare; happens before first
 *    draw or for custom node packs that override drawing), divide the node
 *    body equally across visible widgets and pick the matching band. */
function _widgetAt(node, gx, gy) {
    if (!node || !Array.isArray(node.widgets) || !node.widgets.length) return null;
    if (gx < node.pos[0] || gx > node.pos[0] + (node.size?.[0] || 0)) return null;
    const localY = gy - node.pos[1];
    const defaultH = (window.LiteGraph && LiteGraph.NODE_WIDGET_HEIGHT) || 20;
    const TOL = 2;

    // Visible widget set (don't skip computedDisabled — disabled widgets
    // still take screen space and the user can still hover them).
    const visible = [];
    for (let i = 0; i < node.widgets.length; i++) {
        const w = node.widgets[i];
        if (!w || w.hidden) continue;
        visible.push({ w, i });
    }
    if (!visible.length) return null;

    // Primary: last_y-based hit-test with computed height + tolerance.
    for (const { w, i } of visible) {
        if (typeof w.last_y !== "number") continue;
        let h = defaultH;
        try {
            const cs = w.computeSize ? w.computeSize(node.size?.[0] || 0) : null;
            if (Array.isArray(cs) && cs.length > 1 && typeof cs[1] === "number") h = cs[1];
        } catch (_) { /* keep default */ }
        if (localY >= w.last_y - TOL && localY < w.last_y + h + TOL) {
            return { widget: w, index: i };
        }
    }

    // Fallback: equal-share band scan over the node body (excluding title bar).
    // Triggers when NO widget has last_y yet (no draw pass, or DOM-widget pack).
    const anyLastY = visible.some(({ w }) => typeof w.last_y === "number");
    if (!anyLastY) {
        const TITLE_H = (window.LiteGraph && LiteGraph.NODE_TITLE_HEIGHT) || 30;
        const bodyTop = TITLE_H;
        const bodyBot = node.size?.[1] || (bodyTop + visible.length * defaultH);
        const bandH   = Math.max(1, (bodyBot - bodyTop) / visible.length);
        const idx     = Math.floor((localY - bodyTop) / bandH);
        if (idx >= 0 && idx < visible.length) return { widget: visible[idx].w, index: visible[idx].i };
    }
    return null;
}

/** Position the popover near the node title bar, clamped to viewport. */
function _positionPopover(el, node) {
    const ds     = app.canvas.ds;
    const rect   = app.canvas.canvas.getBoundingClientRect();

    const nx = node.pos[0];
    const ny = node.pos[1];
    const nw = node.size[0];

    // Top-left of title bar in screen coords
    const sx = (nx)           * ds.scale + ds.offset[0] + rect.left;
    const sy = (ny - TITLE_H) * ds.scale + ds.offset[1] + rect.top;
    // Node width in screen coords
    const sw = nw * ds.scale;

    const popH   = el.scrollHeight || 300;
    const margin = 8;
    const vw     = window.innerWidth;
    const vh     = window.innerHeight;

    // Prefer above title bar, fall back to below
    let top;
    if (sy - popH - margin > 0) {
        top = sy - popH - margin;
    } else {
        top = sy + TITLE_H * ds.scale + margin;
    }

    // Horizontal: align with left of node, clamp to viewport
    let left = sx + (sw - POPOVER_W) / 2;
    left = Math.max(margin, Math.min(left, vw - POPOVER_W - margin));
    top  = Math.max(margin, Math.min(top,  vh - popH      - margin));

    el.style.left = `${left + window.scrollX}px`;
    el.style.top  = `${top  + window.scrollY}px`;
}

/** Position popover next to a specific slot dot. Inputs anchor on the left
 *  side of the screen (popover appears to the left of the slot); outputs
 *  anchor on the right. Falls back to a 20 px grid if `getConnectionPos`
 *  isn't available. */
function _positionPopoverSlot(el, node, isInput, slotIndex) {
    const ds   = app.canvas.ds;
    const rect = app.canvas.canvas.getBoundingClientRect();
    let gp = null;
    if (node.getConnectionPos) {
        try { gp = node.getConnectionPos(isInput, slotIndex); } catch (_) { gp = null; }
    }
    if (!gp || !gp.length) {
        const nx = node.pos[0], ny = node.pos[1], nw = node.size?.[0] || 0;
        gp = [isInput ? nx : nx + nw, ny + 20 * (slotIndex + 0.5)];
    }
    const sx = gp[0] * ds.scale + ds.offset[0] + rect.left;
    const sy = gp[1] * ds.scale + ds.offset[1] + rect.top;

    const popH   = el.scrollHeight || 160;
    const margin = 8;
    const vw = window.innerWidth, vh = window.innerHeight;
    // Inputs: popover to the left of slot; outputs: to the right.
    let left = isInput ? (sx - POPOVER_W - margin) : (sx + margin);
    // Vertically centred on slot
    let top  = sy - popH / 2;
    // If pushed off-screen horizontally, flip to the other side
    if (left < margin) left = sx + margin;
    if (left + POPOVER_W + margin > vw) left = sx - POPOVER_W - margin;
    left = Math.max(margin, Math.min(left, vw - POPOVER_W - margin));
    top  = Math.max(margin, Math.min(top,  vh - popH      - margin));
    el.style.left = `${left + window.scrollX}px`;
    el.style.top  = `${top  + window.scrollY}px`;
}

// --------------------------------------------------------------------------
// Type glossary + suggestion cache (filled lazily from backend)
// --------------------------------------------------------------------------
const _TYPE_GLOSSARY = {
    // Minimal local fallback (covers the most common types when backend is offline).
    "MODEL":        "A diffusion / UNet model object — output of checkpoint loaders, consumed by KSampler-family nodes.",
    "CLIP":         "Text encoder (CLIP / T5). Feed into CLIPTextEncode to turn a prompt into conditioning.",
    "VAE":          "Variational Auto-Encoder. Decodes latents to pixels or vice versa.",
    "CONDITIONING": "Encoded prompt embeddings — connect into KSampler positive / negative.",
    "LATENT":       "Compressed image in latent space. VAEDecode turns it into pixels.",
    "IMAGE":        "RGB tensor (B,H,W,3), float32 in [0,1].",
    "MASK":         "Grayscale mask (B,H,W), float32 in [0,1].",
    "CONTROL_NET":  "ControlNet model — applied via ControlNetApply / Advanced.",
    "INT": "Integer scalar.", "FLOAT": "Floating-point scalar.", "STRING": "Text string.",
    "BOOLEAN": "True / False toggle.", "COMBO": "Drop-down chooser.",
    "*": "Wildcard — accepts any type.",
};
let _typeGlossaryFetched = false;

function _typeHint(t) {
    if (!t) return "";
    // Fire one-time fetch from backend to overwrite/extend the local glossary
    if (!_typeGlossaryFetched) {
        _typeGlossaryFetched = true;
        fetch("/c2c/autoconnect/type_glossary")
            .then(r => r.ok ? r.json() : null)
            .then(j => { if (j && j.glossary) Object.assign(_TYPE_GLOSSARY, j.glossary); })
            .catch(() => {});
    }
    return _TYPE_GLOSSARY[t] || _TYPE_GLOSSARY[String(t).toUpperCase()] || "";
}

// --------------------------------------------------------------------------
// Widget glossary — per-parameter explanations + per-COMBO-option blurbs.
// Generic by widget NAME (lowercased). Custom node packs reuse the same
// conventional names (seed, cfg, steps, denoise, sampler_name, scheduler,
// ckpt_name, lora_name, strength_*, vae_name, width, height, batch_size,
// filename_prefix, text...), so this covers the long tail too.
// --------------------------------------------------------------------------
const _WIDGET_GLOSSARY = {
    "seed": "Pseudo-random seed for the noise. Same seed + same inputs = same image. Set fixed for reproducibility; use control_after_generate=randomize for variation.",
    "control_after_generate": "What to do with the seed after each run: fixed | increment | decrement | randomize.",
    "noise_seed": "Seed for the noise tensor in advanced samplers. Same role as 'seed'.",
    "steps": "Denoising iterations. More steps = smoother but slower. Typical: SDXL 20-40, Lightning 4-8, Turbo 1-4, Flux 20-30.",
    "start_at_step": "First step index this sampler runs (advanced two-pass workflows). 0 = from full noise.",
    "end_at_step": "Last step index. Use with start_at_step to chain samplers without re-noising.",
    "cfg": "Classifier-Free Guidance scale. Higher = more prompt adherence (and saturation). SDXL 4-8, SD1.5 5-9, Turbo/Lightning 1-2, Flux 1 (CFG built-in).",
    "guidance": "Flux/Hunyuan guidance strength. Built-in equivalent of CFG; typical 3-5.",
    "denoise": "Fraction of latent to overwrite with noise before sampling. 1.0 = full txt2img. <1.0 = img2img / hires-fix strength.",
    "sampler_name": "Sampling algorithm that steps noise -> image. See option list for per-sampler character.",
    "scheduler": "How sigmas (noise levels) are distributed across the steps.",
    "ckpt_name": "Checkpoint to load (UNet + VAE + CLIP packed together).",
    "config_name": "Optional YAML config for the checkpoint (legacy SD1.x).",
    "unet_name": "UNet/DiT weights file.",
    "clip_name": "Text encoder weights (CLIP / T5).",
    "clip_name1": "First CLIP encoder (SDXL has two).",
    "clip_name2": "Second CLIP encoder (SDXL).",
    "vae_name": "VAE codec for latent <-> pixel conversion.",
    "lora_name": "LoRA adapter file to layer onto the base model.",
    "strength_model": "How strongly the LoRA modifies the UNet (0..2). 1.0 = full effect.",
    "strength_clip": "How strongly the LoRA modifies the text encoder. 0 = TI-only LoRA.",
    "width": "Output width (px). Multiple of 8 (often 64). SDXL native 1024, SD1.5 native 512/768.",
    "height": "Output height (px). Same multiples-of-8 rule.",
    "batch_size": "Images per queued run. VRAM-bound.",
    "text": "Prompt text (positive or negative).",
    "text_g": "SDXL CLIP-G (global) prompt — broader concepts.",
    "text_l": "SDXL CLIP-L (local) prompt — finer detail.",
    "filename_prefix": "Output filename prefix (without extension).",
    "image": "Source image tensor.",
    "title": "Node display title (cosmetic).",
    "ascore": "Aesthetic score (SDXL refiner conditioning).",
    "crop_w": "Crop x offset embedded into SDXL conditioning.",
    "crop_h": "Crop y offset embedded into SDXL conditioning.",
    "target_width": "Target width baked into SDXL conditioning.",
    "target_height": "Target height baked into SDXL conditioning.",
    "stop_at_clip_layer": "CLIP-skip. -1 = use last layer; -2 = skip last layer (Booru aesthetic), etc.",
    "amount": "Generic strength/amount slider.",
    "value": "Generic value field.",
    "speak_speed": "TTS playback speed multiplier.",
    "interpolation": "Resampling kernel: nearest | bilinear | bicubic | lanczos. Bilinear = soft fast; lanczos = sharp slow.",
    "upscale_method": "Upscale algorithm — same trade-offs as 'interpolation'.",
};

// Combo-option glossaries — what each value MEANS.
const _SAMPLER_GLOSSARY = {
    "euler":              "Simple, fast, deterministic. Solid baseline at any CFG.",
    "euler_ancestral":    "Euler with re-injected noise each step. More creative variation.",
    "heun":               "Higher-order Euler. Slower, slightly more accurate per step.",
    "dpm_2":              "DPM-Solver order-2. Good quality/step ratio.",
    "dpm_2_ancestral":    "DPM-2 with noise injection.",
    "lms":                "Linear multi-step. Smooth, classic.",
    "dpm_fast":           "Adaptive DPM. Fast, can be noisy at low steps.",
    "dpm_adaptive":       "Adaptive step size DPM. Accuracy-driven, ignores step count.",
    "dpmpp_2s_ancestral": "DPM++ 2S + noise. Stable for most styles.",
    "dpmpp_sde":          "DPM++ SDE. Stochastic — common SD default.",
    "dpmpp_sde_gpu":      "DPM++ SDE on GPU (faster).",
    "dpmpp_2m":           "DPM++ 2M — multistep. Very popular for SDXL.",
    "dpmpp_2m_sde":       "DPM++ 2M SDE — multistep + stochastic. Often sharper.",
    "dpmpp_2m_sde_gpu":   "DPM++ 2M SDE on GPU.",
    "dpmpp_3m_sde":       "DPM++ 3M SDE — three-step. Higher accuracy, slower.",
    "dpmpp_3m_sde_gpu":   "DPM++ 3M SDE on GPU.",
    "ddim":               "Denoising Diffusion Implicit Models. Deterministic, classic.",
    "uni_pc":             "UniPC — Unified Predictor-Corrector. Fast convergence at low steps.",
    "uni_pc_bh2":         "UniPC + B-spline higher-order.",
    "lcm":                "Latent Consistency Models — for LCM-distilled checkpoints (4-8 steps).",
    "ipndm":              "Improved PNDM.",
    "ipndm_v":            "Variance-corrected IPNDM.",
    "deis":               "DEIS — exponential-integrator solver.",
    "res_multistep":      "RES — re-sampling multistep.",
    "res_multistep_cfg_pp": "RES multistep + CFG++ for stronger prompt adherence.",
    "gradient_estimation":"Gradient-estimation sampler (experimental).",
};
const _SCHEDULER_GLOSSARY = {
    "normal":           "Linear sigma spacing. ComfyUI default.",
    "karras":           "Karras curve — denser sigmas near the start. Standard for SDXL.",
    "exponential":      "Exponentially spaced sigmas.",
    "sgm_uniform":      "SGM uniform spacing — used by some research samplers.",
    "simple":           "Simplest linear schedule.",
    "ddim_uniform":     "Uniform schedule that mimics original DDIM.",
    "beta":             "Beta-distribution-based schedule.",
    "linear_quadratic": "Linear-then-quadratic split.",
    "kl_optimal":       "KL-divergence-optimal spacing (research).",
};
const _CONTROL_AFTER_GLOSSARY = {
    "fixed":     "Keep the seed as-is across runs.",
    "increment": "Add 1 to the seed each run.",
    "decrement": "Subtract 1 from the seed each run.",
    "randomize": "Pick a new random seed each run.",
};

/** Return a human-readable widget-type label and combo-option glossary, if any. */
function _widgetTypeLabel(w) {
    const t = String(w?.type || "").toLowerCase();
    if (t === "combo")  return "drop-down (combo)";
    if (t === "number") return "number";
    if (t === "slider") return "slider";
    if (t === "toggle") return "toggle (boolean)";
    if (t === "text" || t === "string") return "text";
    if (t === "button") return "button";
    if (t === "customtext") return "multi-line text";
    return t || "?";
}
function _comboOptionGlossary(name) {
    const lower = String(name || "").toLowerCase();
    if (lower === "sampler_name") return _SAMPLER_GLOSSARY;
    if (lower === "scheduler")    return _SCHEDULER_GLOSSARY;
    if (lower === "control_after_generate") return _CONTROL_AFTER_GLOSSARY;
    return null;
}

// Cache for AI-generated widget blurbs (key: "<cls>|<widgetName>")
const _AI_WIDGET_CACHE = new Map();
let _aiInflight = null;

async function _aiExplainWidget(cls, widgetName, widgetType, sampleValue) {
    const key = `${cls}|${widgetName}`;
    if (_AI_WIDGET_CACHE.has(key)) return _AI_WIDGET_CACHE.get(key);
    if (_aiInflight) return null;  // serialise to avoid hammering backend
    _aiInflight = (async () => {
        try {
            // _c2c_ai_client.js exposes window.c2cAI.ask({prompt, system, ...})
            if (!window.c2cAI || typeof window.c2cAI.ask !== "function") return null;
            const prompt = `Explain the widget "${widgetName}" (type=${widgetType}, sample=${JSON.stringify(sampleValue)}) on the ComfyUI node "${cls}" in one sentence, plain English, for a non-technical user.`;
            const ans = await window.c2cAI.ask({ prompt, system: "You are ComfyUI's inline help. One sentence only. No markdown.", maxTokens: 80, timeoutMs: 6000 });
            const txt = String(ans?.text || ans || "").trim();
            if (txt) _AI_WIDGET_CACHE.set(key, txt);
            return txt;
        } catch (_) { return null; }
        finally { _aiInflight = null; }
    })();
    return _aiInflight;
}

/** Render the per-widget inspector view (panel mode). */
function _renderWidget(el, node, w, idx) {
    const name = String(w.name || "(unnamed)");
    const lower = name.toLowerCase();
    const cls = node.type || node.title || "?";
    const val = w.value;
    const opts = w.options || {};
    // Lookup priority: exact name → lowercase name → input-spec tooltip → generic-by-type → AI on-demand.
    let blurb = _WIDGET_GLOSSARY[name] || _WIDGET_GLOSSARY[lower] || "";
    let blurbSource = blurb ? "glossary" : "";
    if (!blurb) {
        const spec = _getInputSpec(node, name);
        if (spec?.opts?.tooltip) { blurb = String(spec.opts.tooltip); blurbSource = "node"; }
    }
    if (!blurb) {
        const tlow = String(w.type || "").toLowerCase();
        if (tlow === "combo")  blurb = "Drop-down selector — choose one of the listed options.";
        else if (tlow === "number" || tlow === "slider") blurb = "Numeric value. See range below.";
        else if (tlow === "toggle") blurb = "Boolean toggle (on/off).";
        else if (tlow === "text" || tlow === "string" || tlow === "customtext") blurb = "Text value.";
        if (blurb) blurbSource = "type";
    }
    // AI fallback: only on-demand for custom widgets we have nothing for.
    const aiKey = `${cls}|${name}`;
    if (!blurb && _AI_WIDGET_CACHE.has(aiKey)) { blurb = _AI_WIDGET_CACHE.get(aiKey); blurbSource = "ai"; }
    let aiPending = false;
    if (!blurb) {
        // Trigger lazy AI fetch (settings-gated). Cache fills in for next hover.
        let aiEnabled = true;
        try { aiEnabled = app.ui.settings.getSettingValue("c2c.inspector.ai_widget_blurb", true); } catch (_) {}
        if (aiEnabled && window.c2cAI && typeof window.c2cAI.ask === "function") {
            aiPending = true;
            _aiExplainWidget(cls, name, w.type, val).then(txt => {
                if (txt && _currentWidgetKey === `${node.id}|${idx}`) _renderWidget(el, node, w, idx);
            });
        }
        blurb = aiPending ? "… (asking the AI for an explanation, this will cache for next time)"
                          : "No description available for this widget yet.";
        blurbSource = aiPending ? "ai-pending" : "none";
    }

    const parts = [];
    parts.push(`<div class="mec-ne-headline">${_esc(name)} <span class="mec-ne-muted">— ${_esc(_widgetTypeLabel(w))}</span></div>`);
    parts.push(`<div class="mec-ne-muted">on <code>${_esc(cls)}</code> &middot; widget #${idx}</div>`);
    parts.push(`<div class="mec-ne-purpose" style="margin-top:8px"><b>Current value:</b> <code>${_esc(String(val))}</code></div>`);

    // Range / step / default.
    const rangeBits = [];
    if (typeof opts.min  === "number") rangeBits.push(`min ${opts.min}`);
    if (typeof opts.max  === "number") rangeBits.push(`max ${opts.max}`);
    if (typeof opts.step === "number") rangeBits.push(`step ${opts.step}`);
    if ("default" in opts && opts.default !== undefined) rangeBits.push(`default ${opts.default}`);
    if (rangeBits.length) {
        parts.push(`<div class="mec-ne-muted">Range: ${_esc(rangeBits.join(" · "))}</div>`);
    }

    if (blurb) parts.push(`<div style="margin-top:8px">${_esc(blurb)}</div>`);

    // Combo options table with per-option blurbs.
    const values = Array.isArray(opts.values)
        ? opts.values
        : (Array.isArray(w.options?.values) ? w.options.values : null);
    if (values && values.length) {
        const og = _comboOptionGlossary(name);
        parts.push(`<div class="mec-ne-section-title">Options (${values.length})</div>`);
        const rows = values.map(v => {
            const ov = String(v);
            const desc = og ? (og[ov] || og[ov.toLowerCase()] || "") : "";
            const cur = (String(val) === ov) ? ' class="cur"' : "";
            return `<tr${cur}><td><code>${_esc(ov)}</code></td><td>${_esc(desc)}</td></tr>`;
        }).join("");
        parts.push(`<table class="mec-ne-opts"><tbody>${rows}</tbody></table>`);
    }

    el.innerHTML = parts.join("");
}

/** Look up `(type, opts)` for a slot/widget on a node by name, using
 *  whatever spec the front-end has cached. Returns null if not found. */
function _getInputSpec(node, slotName) {
    if (!node || !slotName) return null;
    // ComfyUI puts the parsed INPUT_TYPES on the registered_node_types entry
    // under .nodeData.input (required + optional), keyed by slot name.
    const reg = window.LiteGraph?.registered_node_types?.[node.type];
    const inp = reg?.nodeData?.input;
    if (!inp) return null;
    const sec = inp.required?.[slotName] || inp.optional?.[slotName];
    if (!Array.isArray(sec) || !sec.length) return null;
    const type = sec[0];
    const opts = (sec.length >= 2 && typeof sec[1] === "object" && sec[1] !== null) ? sec[1] : {};
    return { type, opts };
}

/** Render a tiny key/value block describing the widget linked to a slot:
 *  current value, default, min/max/step, tooltip — whatever's known. */
function _renderWidgetMeta(node, slot) {
    if (!node || !slot || !slot.name) return "";
    // Find a matching widget on the node by name (most common path).
    const w = (Array.isArray(node.widgets) ? node.widgets : []).find(x => x?.name === slot.name);
    const spec = _getInputSpec(node, slot.name);
    if (!w && !spec) return "";

    const rows = [];
    if (w && "value" in w) rows.push(["value", String(w.value)]);
    if (spec?.opts) {
        if ("default" in spec.opts) rows.push(["default", String(spec.opts.default)]);
        if ("min"     in spec.opts) rows.push(["min",     String(spec.opts.min)]);
        if ("max"     in spec.opts) rows.push(["max",     String(spec.opts.max)]);
        if ("step"    in spec.opts) rows.push(["step",    String(spec.opts.step)]);
        if (spec.opts.tooltip)       rows.push(["info",   String(spec.opts.tooltip)]);
    }
    if (!rows.length) return "";
    const cells = rows.map(([k, v]) =>
        `<div class="k">${_esc(k)}</div><div class="v">${_esc(v)}</div>`
    ).join("");
    return `<div class="mec-ne-section-title">Widget</div><div class="mec-ne-kv">${cells}</div>`;
}

// Per-slot suggestion cache: key = "cls|dir|slot|type" -> array of {cls,slot,score,confidence,sources}
const _SUGGEST_CACHE = new Map();

async function _fillSuggestions(targetId, node, isInput, slotIndex, slotType) {
    const target = document.getElementById(targetId);
    if (!target) return;
    const slot = (isInput ? node.inputs : node.outputs)?.[slotIndex];
    if (!slot) { target.innerHTML = ""; return; }
    const cls = node.type || "";
    const dir = isInput ? "input" : "output";
    const slotName = slot.name || "";
    const cacheKey = `${cls}|${dir}|${slotName}|${slotType}`;

    let suggestions = _SUGGEST_CACHE.get(cacheKey);
    if (!suggestions) {
        try {
            const url = `/c2c/autoconnect/suggest?cls=${encodeURIComponent(cls)}&dir=${dir}&slot=${encodeURIComponent(slotName)}&type=${encodeURIComponent(slotType)}&limit=5`;
            const r = await fetch(url);
            if (r.ok) {
                const j = await r.json();
                suggestions = Array.isArray(j?.suggestions) ? j.suggestions : [];
            } else {
                suggestions = [];
            }
        } catch (_e) {
            suggestions = [];
        }
        _SUGGEST_CACHE.set(cacheKey, suggestions);
    }

    // The target may have been removed if the popover was hidden meanwhile.
    if (!document.getElementById(targetId)) return;

    if (!suggestions.length) {
        target.innerHTML = `<span class="mec-ne-muted" style="font-size:11px">No suggestions yet — connect once and I'll learn.</span>`;
        return;
    }

    const rows = suggestions.map((s, i) => {
        const conf = Math.max(0, Math.min(1, Number(s.confidence || 0)));
        const pct = Math.round(conf * 100);
        const sources = Array.isArray(s.sources) ? s.sources.join(",") : "";
        const label = `${_esc(s.cls)} · <code>${_esc(s.slot || "")}</code>`;
        return `<div class="mec-ne-row">
            <span class="mec-ne-rowtext" title="${_esc(sources)}">${label}</span>
            <span class="mec-ne-confbar" title="confidence ${pct}%"><i style="width:${pct}%"></i></span>
            <button class="mec-ne-insert" data-cls="${_esc(s.cls)}" data-slot="${_esc(s.slot || "")}"
                    data-srcid="${node.id}" data-srcslot="${slotIndex}" data-isinput="${isInput?1:0}"
                    data-rank="${i}">Insert</button>
        </div>`;
    }).join("");
    target.innerHTML = rows;

    // Wire Insert buttons → c2c_autoconnect.insertSuggestion
    target.querySelectorAll("button.mec-ne-insert").forEach(btn => {
        btn.addEventListener("click", (ev) => {
            ev.stopPropagation();
            const dstCls   = btn.getAttribute("data-cls");
            const dstSlot  = btn.getAttribute("data-slot");
            const srcId    = parseInt(btn.getAttribute("data-srcid"), 10);
            const srcSlot  = parseInt(btn.getAttribute("data-srcslot"), 10);
            const srcIsIn  = btn.getAttribute("data-isinput") === "1";
            try {
                window.c2c_autoconnect?.insertSuggestion?.({
                    sourceNodeId: srcId, sourceSlotIndex: srcSlot, sourceIsInput: srcIsIn,
                    destClass: dstCls, destSlotName: dstSlot,
                });
            } catch (e) { console.error("[C2C] insert failed:", e); }
        });
    });
}

/** Render a single slot's information: name, type, slot index, and what it's
 *  connected to (source for inputs, list of destinations for outputs). */
function _renderSingleSlot(el, node, isInput, slotIndex) {
    const slot = (isInput ? node.inputs : node.outputs)?.[slotIndex];
    if (!slot) { el.classList.remove("visible"); el.innerHTML = ""; return; }
    const kind = isInput ? "Input" : "Output";
    const tStr = Array.isArray(slot.type) ? slot.type.join(" | ")
               : (slot.type === -1 || slot.type === undefined || slot.type === null) ? "*"
               : String(slot.type);
    const primaryType = Array.isArray(slot.type) ? String(slot.type[0]) : tStr;
    const graph = node.graph || app.canvas?.graph || app.graph;

    let linkBlock = "";
    if (isInput) {
        if (slot.link != null && graph?.links) {
            const link = graph.links[slot.link];
            if (link) {
                const src = graph.getNodeById?.(link.origin_id);
                const srcName  = src?.outputs?.[link.origin_slot]?.name ?? `#${link.origin_slot}`;
                const srcTitle = src?.title || src?.type || "?";
                linkBlock = `
                    <div class="mec-ne-section-title">Connected from</div>
                    <table class="mec-ne-slots"><tbody>
                      <tr><td>${_esc(srcTitle)}</td><td><code>${_esc(srcName)}</code></td></tr>
                    </tbody></table>`;
            }
        } else {
            linkBlock = `<div class="mec-ne-muted" style="font-style:italic;margin-top:4px">Not connected.</div>`;
        }
        // Widget value (either converted-to-input or matched by name on node.widgets)
        const wBlock = _renderWidgetMeta(node, slot);
        if (wBlock) linkBlock += wBlock;
    } else {
        const links = Array.isArray(slot.links) ? slot.links : [];
        if (links.length) {
            const rows = links.slice(0, 8).map(lid => {
                const link = graph?.links?.[lid];
                if (!link) return "";
                const dst = graph.getNodeById?.(link.target_id);
                const dstName  = dst?.inputs?.[link.target_slot]?.name ?? `#${link.target_slot}`;
                const dstTitle = dst?.title || dst?.type || "?";
                return `<tr><td>${_esc(dstTitle)}</td><td><code>${_esc(dstName)}</code></td></tr>`;
            }).join("");
            const more = links.length > 8 ? `<div class="mec-ne-muted">…+${links.length - 8} more</div>` : "";
            linkBlock = `
                <div class="mec-ne-section-title">Fans out to (${links.length})</div>
                <table class="mec-ne-slots"><tbody>${rows}</tbody></table>${more}`;
        } else {
            linkBlock = `<div class="mec-ne-muted" style="font-style:italic;margin-top:4px">Not connected.</div>`;
        }
    }

    // Type hint blurb (from glossary cache; falls back to "(no hint)" silently)
    const hint = _typeHint(primaryType);
    const hintBlock = hint ? `<div class="mec-ne-hint"><b>${_esc(primaryType)}</b> — ${_esc(hint)}</div>` : "";

    // Async suggestions placeholder (filled in by _fillSuggestions)
    const suggestId = `mec-ne-sg-${node.id}-${isInput?1:0}-${slotIndex}`;
    const suggestBlock = `
        <div class="mec-ne-section-title">${isInput ? "Suggested source nodes" : "Suggested next nodes"}</div>
        <div id="${suggestId}" class="mec-ne-suggest"><span class="mec-ne-muted">Loading…</span></div>
        <div class="mec-ne-tip">Double-click slot to insert top pick · Shift = chain</div>`;

    el.innerHTML = `
        <div class="mec-ne-headline">${_esc(kind)} • ${_esc(slot.name || "(unnamed)")}</div>
        <div class="mec-ne-purpose mec-ne-muted">Type <code>${_esc(tStr)}</code> · Slot #${slotIndex} on <code>${_esc(node.type || node.title || "")}</code></div>
        ${hintBlock}
        ${linkBlock}
        ${suggestBlock}`.trim();
    el.classList.add("visible");

    // Kick off async suggestion fill (no await — fire-and-forget)
    _fillSuggestions(suggestId, node, isInput, slotIndex, primaryType);
}

function _showLoading(el, nodeName) {
    el.innerHTML = `
        <div class="mec-ne-loading">
            <div class="mec-ne-spinner"></div>
            <span>Loading explanation for <strong>${_esc(nodeName)}</strong>…</span>
        </div>`.trim();
    el.classList.add("visible");
}

/** Render slot info directly from the LGraphNode object — works for ANY node
 *  (ComfyUI native, third-party, our own) without needing a backend round-trip.
 *  This is shown immediately on dwell; if a richer LLM explanation arrives
 *  later, it overwrites this block. */
function _renderSlotsLocal(el, node) {
    const title = node.title || node.type || "";
    const cls   = node.type || "";
    const ins   = Array.isArray(node.inputs)  ? node.inputs  : [];
    const outs  = Array.isArray(node.outputs) ? node.outputs : [];
    const widgets = Array.isArray(node.widgets) ? node.widgets : [];

    const slotType = (s) => {
        const t = s?.type;
        if (t === undefined || t === null || t === -1) return "*";
        return String(t);
    };
    const inputRows = ins.map((s, i) => {
        const linked = s.link != null ? "— linked" : "";
        return `<tr><td>${i}</td><td>${_esc(s.name || "")}</td><td><code>${_esc(slotType(s))}</code></td><td class="mec-ne-muted">${linked}</td></tr>`;
    }).join("");
    const outputRows = outs.map((s, i) => {
        const fanout = Array.isArray(s.links) ? s.links.length : 0;
        return `<tr><td>${i}</td><td>${_esc(s.name || "")}</td><td><code>${_esc(slotType(s))}</code></td><td class="mec-ne-muted">${fanout ? `→ ${fanout}` : ""}</td></tr>`;
    }).join("");
    const widgetRows = widgets.slice(0, 24).map((w) => {
        let v = w.value;
        if (v !== undefined && v !== null && typeof v !== "object") {
            v = String(v); if (v.length > 40) v = v.slice(0, 37) + "…";
        } else if (v && typeof v === "object") {
            v = "[object]";
        } else { v = ""; }
        return `<tr><td>${_esc(w.name || "")}</td><td><code>${_esc(String(w.type || ""))}</code></td><td class="mec-ne-muted">${_esc(v)}</td></tr>`;
    }).join("");

    const inputsBlock = ins.length ? `
        <div class="mec-ne-section-title">Inputs (${ins.length})</div>
        <table class="mec-ne-slots"><thead><tr><th>#</th><th>name</th><th>type</th><th></th></tr></thead><tbody>${inputRows}</tbody></table>` : "";
    const outputsBlock = outs.length ? `
        <div class="mec-ne-section-title">Outputs (${outs.length})</div>
        <table class="mec-ne-slots"><thead><tr><th>#</th><th>name</th><th>type</th><th></th></tr></thead><tbody>${outputRows}</tbody></table>` : "";
    const widgetsBlock = widgets.length ? `
        <div class="mec-ne-section-title">Widgets (${widgets.length}${widgets.length > 24 ? " — first 24" : ""})</div>
        <table class="mec-ne-slots"><thead><tr><th>name</th><th>type</th><th>value</th></tr></thead><tbody>${widgetRows}</tbody></table>` : "";

    el.innerHTML = `
        <div class="mec-ne-headline">${_esc(title)}</div>
        <div class="mec-ne-purpose mec-ne-muted"><code>${_esc(cls)}</code></div>
        ${_capabilityBlock(cls)}
        ${_recommendBlock(cls, node)}
        ${inputsBlock}
        ${outputsBlock}
        ${widgetsBlock}`.trim();
    el.classList.add("visible");
}

/** Rule-based (NO AI) sampler/scheduler/cfg recommendation, rendered INSIDE the
 *  C2C node inspector for sampler-type nodes. Logic lives in
 *  c2c_settings_recommender.js (window.__C2C_REC); this just formats it. */
function _recommendBlock(cls, node) {
    try {
        const R = window.__C2C_REC;
        if (!R || !R.isSamplerNode(node)) return "";
        const rec = R.recFor(R.modelForSampler(node, R.detectedModels()));
        const dd = R.deepFor(rec.label) || {};
        const row = (k, v) => (v && v !== "—")
            ? `<tr><td class="mec-ne-muted" style="white-space:nowrap;vertical-align:top">${_esc(k)}</td><td>${_esc(v)}</td></tr>` : "";
        return `<div class="mec-ne-section-title">⚙ Recommended settings — ${_esc(rec.label)}</div>`
            + `<table class="mec-ne-slots"><tbody>`
            + row("sampler", rec.sampler) + row("scheduler", rec.scheduler)
            + row("cfg", rec.cfg) + row("steps", rec.steps)
            + row("resolution", dd.res) + row("VAE", dd.vae) + row("CLIP/encoder", dd.clip)
            + row("negative", dd.neg) + row("alternatives", dd.alts) + row("⚠ pitfalls", dd.pitfalls)
            + `</tbody></table>`;
    } catch (_) { return ""; }
}

/** Instant capability summary derived from the shared node taxonomy
 *  (js/c2c_node_taxonomy.js). Shows the node's category colour + the
 *  capability keywords used by the Workflow Library / Finder search, so the
 *  explanation surface and search stay in lock-step. */
function _capabilityBlock(cls) {
    const caps = capabilityFor(cls);
    if (!caps) return "";
    const col = nodeColor(cls);
    return `<div class="mec-ne-purpose" style="display:flex;gap:6px;align-items:center;margin-top:3px">`
         + `<span style="width:10px;height:10px;border-radius:2px;flex:0 0 auto;background:${col};border:1px solid ${lighten(col, 40)}"></span>`
         + `<span style="color:#9aa0d0;font-size:11px">${_esc(caps)}</span></div>`;
}

function _showError(el, msg) {
    el.innerHTML = `<div class="mec-ne-error">⚠ ${_esc(msg)}</div>`;
    el.classList.add("visible");
}

function _renderExplanation(el, data) {
    const headline   = data.headline    || "";
    const purpose    = data.purpose     || "";
    const inputs     = Array.isArray(data.inputs)  ? data.inputs  : [];
    const outputs    = Array.isArray(data.outputs) ? data.outputs : [];
    const when       = data.when_to_use || "";
    const tier       = data.tier        || "unknown";

    const tierClass  = tier.startsWith("cloud") ? "tier-cloud"
                     : tier.startsWith("gguf")  ? "tier-gguf"
                     : "tier-det";
    const tierLabel  = tier.startsWith("cloud") ? `☁ ${tier.split("/")[1] || "cloud"}`
                     : tier.startsWith("gguf")  ? "🤖 Local GGUF"
                     : "📋 Deterministic";

    const inputRows  = inputs.slice(0, 16).map(inp =>
        `<tr><td>${_esc(inp.name || "")}</td><td>${_esc(inp.what_for || "")}</td></tr>`
    ).join("");
    const outputRows = outputs.slice(0, 8).map(out =>
        `<tr><td>${_esc(out.name || "")}</td><td>${_esc(out.what_for || "")}</td></tr>`
    ).join("");

    const inputsBlock = inputs.length ? `
        <div class="mec-ne-section-title">Inputs</div>
        <table><tbody>${inputRows}</tbody></table>` : "";
    const outputsBlock = outputs.length ? `
        <div class="mec-ne-section-title">Outputs</div>
        <table><tbody>${outputRows}</tbody></table>` : "";

    el.innerHTML = `
        <div class="mec-ne-headline">${_esc(headline)}</div>
        <div class="mec-ne-purpose">${_esc(purpose)}</div>
        ${inputsBlock}
        ${outputsBlock}
        <div class="mec-ne-footer">
            <span class="mec-ne-badge ${tierClass}">${tierLabel}</span>
            <span>${_esc(when)}</span>
        </div>`.trim();
    el.classList.add("visible");
}

function _esc(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

// ── show / hide logic ──────────────────────────────────────────────────────
function _clearHide() {
    if (_hideTimer !== null) {
        clearTimeout(_hideTimer);
        _hideTimer = null;
    }
}

function _scheduleHide() {
    _clearHide();
    _hideTimer = setTimeout(() => {
        if (!_nodeHovered && !_popoverHovered) {
            _hidePopover();
        }
    }, HIDE_MS);
}

function _hidePopover() {
    const el = document.getElementById(POPOVER_ID);
    // In panel mode the panel is always present — don't tear it down,
    // just clear the content area.
    if (el && el.classList.contains("c2c-ne-panel-mode")) {
        const body = el.querySelector(".c2c-ne-panel-body");
        if (body) body.innerHTML = "";
        _setPanelCtx("");
        _popoverNode = null;
        _popoverSlot = null;
        _stopReanchorLoop();
        return;
    }
    if (el) {
        el.classList.remove("visible");
        el.innerHTML = "";
    }
    _popoverNode = null;
    _popoverSlot = null;
    _stopReanchorLoop();
}

// P0.11 — keep the popover glued to the node title while the user pans/zooms
// the canvas or jumps between subgraphs. Cheap because it only runs while the
// popover is visible; reposition only fires when ds.scale/offset or node.pos
// actually change.
function _startReanchorLoop() {
    if (_rafHandle !== null) return;
    const tick = () => {
        _rafHandle = null;
        const el = document.getElementById(POPOVER_ID);
        if (!el || !el.classList.contains("visible")) return;
        const node = _popoverSlot ? _popoverSlot.node : _popoverNode;
        if (!node) return;
        // If the node was removed (deleted, subgraph swap, etc.) close.
        const g = app.canvas?.graph || app.graph;
        if (!g || !(g._nodes || []).includes(node)) { _hidePopover(); return; }
        const ds = app.canvas?.ds;
        if (!ds) { _rafHandle = requestAnimationFrame(tick); return; }
        const sx = ds.scale, ox = ds.offset?.[0] ?? 0, oy = ds.offset?.[1] ?? 0;
        const nx = node.pos[0], ny = node.pos[1], nw = node.size?.[0] || 0;
        const gid = g.id ?? null;
        const sk = _popoverSlot ? `${_popoverSlot.isInput?1:0}:${_popoverSlot.slotIndex}` : "";
        if (sx !== _lastDs.sx || ox !== _lastDs.ox || oy !== _lastDs.oy ||
            nx !== _lastDs.nx || ny !== _lastDs.ny || nw !== _lastDs.nw ||
            gid !== _lastDs.gid || sk !== _lastDs.sk) {
            _lastDs = { sx, ox, oy, nx, ny, nw, gid, sk };
            if (_popoverSlot) {
                _positionPopoverSlot(el, node, _popoverSlot.isInput, _popoverSlot.slotIndex);
            } else {
                _positionPopover(el, node);
            }
        }
        _rafHandle = requestAnimationFrame(tick);
    };
    _lastDs = { sx: 0, ox: 0, oy: 0, nx: 0, ny: 0, nw: 0, gid: null, sk: "" };
    _rafHandle = requestAnimationFrame(tick);
}

function _stopReanchorLoop() {
    if (_rafHandle !== null) {
        cancelAnimationFrame(_rafHandle);
        _rafHandle = null;
    }
}

// ── dwell handling ─────────────────────────────────────────────────────────
function _clearDwell() {
    if (_dwellTimer !== null) {
        clearTimeout(_dwellTimer);
        _dwellTimer = null;
    }
}

async function _fetchAndShow(node) {
    const backend = _getSetting("mec.node_explain.backend", "auto");
    const quant   = _getSetting("mec.node_explain.gguf_quant", "Q4_K_M");
    const cls     = node.type;
    const el      = _getPopover();

    // If already showing for this node, do nothing
    if (_popoverNode === node && el.classList.contains("visible")) return;

    _popoverNode = node;
    _popoverSlot = null;
    const inPanel = el.classList.contains("c2c-ne-panel-mode");

    // ALWAYS render slot info first \u2014 works for any node, no backend needed.
    _renderSlotsLocal(el, node);
    if (inPanel) {
        _setPanelCtx((node.type || node.title || "?") + " · node");
    } else {
        _positionPopover(el, node);
        _startReanchorLoop();
        requestAnimationFrame(() => _positionPopover(el, node));
    }

    // If the LLM explanation is cached, swap it in.
    if (_CACHE.has(cls)) {
        _renderExplanation(el, _CACHE.get(cls));
        if (!inPanel) requestAnimationFrame(() => _positionPopover(el, node));
        return;
    }

    // Backend disabled / unsupported node — keep the local slot view.
    if (backend === "off") return;
    if (_PENDING.has(cls)) return;   // request already in-flight
    _PENDING.add(cls);

    try {
        const url = `/mec/node_explain/${encodeURIComponent(cls)}?backend=${backend}&quant=${quant}`;
        const resp = await fetch(url);
        const json = await resp.json();
        _PENDING.delete(cls);

        if (!json.success) {
            // Backend doesn't know this class — silently keep the local slot view.
            return;
        }

        const data = json.data;
        _CACHE.set(cls, data);

        if (_popoverNode === node && el.classList.contains("visible")) {
            _renderExplanation(el, data);
            requestAnimationFrame(() => _positionPopover(el, node));
        }
    } catch (err) {
        _PENDING.delete(cls);
        // Network error — keep the local slot view, no scary error toast.
    }
}

// ── canvas event handlers ──────────────────────────────────────────────────

/** Show slot popover immediately (synchronous, no backend needed). */
function _showSlot(node, isInput, slotIndex) {
    const el = _getPopover();
    _popoverNode = null;
    _popoverSlot = { node, isInput, slotIndex };
    _renderSingleSlot(el, node, isInput, slotIndex);
    if (el.classList.contains("c2c-ne-panel-mode")) {
        const slot = (isInput ? node.inputs : node.outputs)?.[slotIndex];
        _setPanelCtx(`${node.type || "?"} \u2022 ${isInput ? "in" : "out"} \u00b7 ${slot?.name || "#" + slotIndex}`);
        return;  // panel is fixed — no positioning, no reanchor loop
    }
    _positionPopoverSlot(el, node, isInput, slotIndex);
    _startReanchorLoop();
    requestAnimationFrame(() => _positionPopoverSlot(el, node, isInput, slotIndex));
}

function _slotKey(s) {
    return s ? `${s.node?.id ?? "?"}|${s.isInput ? 1 : 0}|${s.slotIndex}` : "";
}

function _onMouseMove(e) {
    // Ignore if user is dragging canvas (left button held)
    if (e.buttons & 1) {
        _clearDwell();
        _nodeHovered = false;
        if (!_panelMode()) _scheduleHide();
        return;
    }

    const { gx, gy } = _toGraph(e);
    const inPanel = _panelMode();

    // 1) Slot under cursor? (highest priority — small, precise)
    const slotHit = _slotHitTest(gx, gy);
    if (slotHit) {
        const newKey = _slotKey(slotHit);
        const curKey = _slotKey(_currentSlot);
        if (newKey === curKey) return;
        _clearDwell();
        _currentSlot = slotHit;
        _currentNode = null;
        _nodeHovered = true;
        _clearHide();
        const dwellMs = inPanel ? 30 : Math.max(50, parseInt(
            _getSetting("mec.node_explain.dwell_ms", DEFAULT_DWELL_MS), 10) || DEFAULT_DWELL_MS);
        _dwellTimer = setTimeout(() => {
            _dwellTimer = null;
            if (_slotKey(_currentSlot) === newKey) {
                _showSlot(slotHit.node, slotHit.isInput, slotHit.slotIndex);
            }
        }, dwellMs);
        return;
    }

    // No slot hit — clear any stale slot tracking and check title corner.
    if (_currentSlot) {
        _currentSlot = null;
        _clearDwell();
        _nodeHovered = false;
        if (_popoverSlot && !inPanel) _scheduleHide();
    }

    // 2) Widget under cursor? (panel mode only — instant, no dwell)
    if (inPanel) {
        const bodyNode = _nodeAtBody(gx, gy);
        if (bodyNode) {
            const wHit = _widgetAt(bodyNode, gx, gy);
            if (wHit) {
                const wKey = `${bodyNode.id}|${wHit.index}`;
                if (wKey !== _currentWidgetKey) {
                    _currentWidgetKey = wKey;
                    _currentNode      = null;
                    _nodeHovered      = true;
                    _clearDwell();
                    _clearHide();
                    const el = _getPopover();
                    _popoverNode = null; _popoverSlot = null;
                    _renderWidget(el, bodyNode, wHit.widget, wHit.index);
                    _setPanelCtx(`${bodyNode.type || "?"} \u2022 widget \u00b7 ${wHit.widget.name || "#" + wHit.index}`);
                }
                return;
            }
            // Inside node body but NOT on a widget — fall through to node-body rendering.
            if (_currentWidgetKey) _currentWidgetKey = null;
        } else {
            if (_currentWidgetKey) _currentWidgetKey = null;
        }
    }

    // 3) Body / title under cursor?
    //    Panel mode: anywhere on the node → show node info (sticky).
    //    Popover mode: only the top-left title corner (drag-friendly).
    const node = inPanel ? _nodeAtBody(gx, gy) : _nodeAtTitle(gx, gy);
    if (node === _currentNode) {
        if (!node && !_nodeHovered && !inPanel) _scheduleHide();
        return;
    }
    _clearDwell();
    _currentNode = node;

    if (!node) {
        _nodeHovered = false;
        // Panel mode is sticky: keep last content visible.
        if (!inPanel) _scheduleHide();
        return;
    }

    _nodeHovered = true;
    _clearHide();

    const dwellMs = inPanel ? 30 : Math.max(50, parseInt(
        _getSetting("mec.node_explain.dwell_ms", DEFAULT_DWELL_MS), 10) || DEFAULT_DWELL_MS);

    _dwellTimer = setTimeout(() => {
        _dwellTimer = null;
        if (_currentNode === node) {
            _fetchAndShow(node);
        }
    }, dwellMs);
}

function _onMouseLeave() {
    _clearDwell();
    _nodeHovered  = false;
    _currentNode  = null;
    _currentSlot  = null;
    _currentWidgetKey = null;
    // Sticky in panel mode — don't blank when cursor leaves the canvas.
    if (!_panelMode()) _scheduleHide();
}

// ── extension registration ─────────────────────────────────────────────────
app.registerExtension({
    name: "C2C.NodeExplain",

    settings: [
        {
            id:      "mec.node_explain.surface",
            name:    "Node Explain: surface",
            tooltip: "Where node/slot info is shown. 'panel' = fixed bottom-right dock (draggable, sticky). 'popover' = floating bubble next to node (legacy).",
            type:    "combo",
            options: ["panel", "popover"],
            default: "panel",
        },
        {
            id:      "mec.node_explain.backend",
            name:    "Node Explain: LLM backend",
            tooltip: "Which backend to use for 'What does this node do?' hover tooltips.",
            type:    "combo",
            options: ["auto", "api", "gguf", "off"],
            default: "auto",
        },
        {
            id:      "mec.node_explain.gguf_quant",
            name:    "Node Explain: GGUF quant",
            tooltip: "Which Qwen3.5-2B quantisation to use for local inference.",
            type:    "combo",
            options: ["Q4_K_M", "Q5_K_M", "Q8_0"],
            default: "Q4_K_M",
        },
        {
            id:      "mec.node_explain.dwell_ms",
            name:    "Node Explain: hover dwell (ms)",
            tooltip: "How long to hover a node title before the explanation card appears. Lower = snappier.",
            type:    "slider",
            attrs:   { min: 50, max: 1500, step: 50 },
            default: DEFAULT_DWELL_MS,
        },
    ],

    async setup() {
        _injectStyle();
        _getPopover();   // pre-create so listeners are attached

        const canvas = app.canvas?.canvas;
        if (!canvas) {
            console.warn("[MEC.NodeExplain] canvas not available at setup() — retrying…");
            // Retry once the graph is ready
            app.graph?.onAfterChange?.(() => this.setup());
            return;
        }

        canvas.addEventListener("mousemove",  _onMouseMove);
        canvas.addEventListener("mouseleave", _onMouseLeave);

        // Also hide when LiteGraph fires its own drag events
        canvas.addEventListener("mousedown",  () => {
            _clearDwell();
            _nodeHovered = false;
            if (!_panelMode()) _scheduleHide();
        });

        console.log("[MEC.NodeExplain] Loaded — hover a node title to explain it (default 250 ms dwell, configurable).");
    },
});
