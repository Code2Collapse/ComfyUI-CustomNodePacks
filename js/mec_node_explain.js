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

// ── constants ──────────────────────────────────────────────────────────────
const DWELL_MS   = 800;   // how long to hover before showing tooltip
const HIDE_MS    = 150;   // delay before hiding after mouse leaves
const TITLE_H    = 30;    // LiteGraph default title bar height in graph-coords
const POPOVER_W  = 360;   // popover width px
const POPOVER_ID = "mec-node-explain-popover";
const STYLE_ID   = "mec-node-explain-style";

// ── session cache ──────────────────────────────────────────────────────────
const _CACHE = new Map();   // Map<className, responseData>
const _PENDING = new Set(); // classes with a request in-flight

// ── state ──────────────────────────────────────────────────────────────────
let _dwellTimer   = null;
let _hideTimer    = null;
let _currentNode  = null;   // node whose title is under cursor
let _popoverNode  = null;   // node the open popover belongs to
let _nodeHovered  = false;
let _popoverHovered = false;

// ── helpers ────────────────────────────────────────────────────────────────
function _getSetting(id, def) {
    try {
        return app.ui.settings.getSettingValue(id, def);
    } catch (_) {
        return def;
    }
}

/** Convert canvas event coords to graph-space coords. */
function _toGraph(e) {
    const canvas = app.canvas;
    const rect   = canvas.canvas.getBoundingClientRect();
    const ds     = canvas.ds;
    const gx = (e.clientX - rect.left  - ds.offset[0]) / ds.scale;
    const gy = (e.clientY - rect.top   - ds.offset[1]) / ds.scale;
    return { gx, gy };
}

/** Find the node whose title bar contains the graph-space point (gx, gy). */
function _nodeAtTitle(gx, gy) {
    const nodes = app.graph._nodes;
    if (!nodes) return null;
    // Iterate in reverse so topmost (highest index) node wins
    for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        if (!n || n.flags?.collapsed) continue;
        const nx = n.pos[0];
        const ny = n.pos[1];
        const nw = n.size[0];
        // Title is the bar immediately above the node body
        const titleTop    = ny - TITLE_H;
        const titleBottom = ny;
        if (gx >= nx && gx <= nx + nw && gy >= titleTop && gy <= titleBottom) {
            return n;
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
    position: fixed;
    z-index: 99999;
    width: ${POPOVER_W}px;
    max-height: 420px;
    overflow-y: auto;
    background: #1e1e2e;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 12px 14px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.65);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
    line-height: 1.45;
    color: #cdd6f4;
    pointer-events: auto;
    display: none;
}
#${POPOVER_ID}.visible { display: block; }
#${POPOVER_ID} .mec-ne-headline {
    font-size: 14px;
    font-weight: 700;
    color: #89b4fa;
    margin-bottom: 6px;
}
#${POPOVER_ID} .mec-ne-purpose {
    margin-bottom: 8px;
    color: #cdd6f4;
}
#${POPOVER_ID} .mec-ne-section-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6c7086;
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
    color: #a6e3a1;
    white-space: nowrap;
    width: 30%;
}
#${POPOVER_ID} .mec-ne-footer {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #313244;
    font-size: 11px;
    color: #6c7086;
}
#${POPOVER_ID} .mec-ne-badge {
    padding: 1px 7px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    background: #313244;
    color: #89b4fa;
    white-space: nowrap;
}
#${POPOVER_ID} .mec-ne-badge.tier-cloud  { background:#1e3a5f; color:#89dceb; }
#${POPOVER_ID} .mec-ne-badge.tier-gguf   { background:#1a3a1a; color:#a6e3a1; }
#${POPOVER_ID} .mec-ne-badge.tier-det    { background:#3a2a1a; color:#fab387; }
#${POPOVER_ID} .mec-ne-loading {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    color: #6c7086;
}
#${POPOVER_ID} .mec-ne-spinner {
    width: 18px;
    height: 18px;
    border: 2px solid #313244;
    border-top-color: #89b4fa;
    border-radius: 50%;
    animation: mec-spin 0.7s linear infinite;
    flex-shrink: 0;
}
@keyframes mec-spin {
    to { transform: rotate(360deg); }
}
#${POPOVER_ID} .mec-ne-error {
    color: #f38ba8;
    font-size: 12px;
}
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
            _scheduleHide();
        });
    }
    return el;
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

    el.style.left = `${left}px`;
    el.style.top  = `${top}px`;
}

function _showLoading(el, nodeName) {
    el.innerHTML = `
        <div class="mec-ne-loading">
            <div class="mec-ne-spinner"></div>
            <span>Loading explanation for <strong>${_esc(nodeName)}</strong>…</span>
        </div>`.trim();
    el.classList.add("visible");
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
    if (el) {
        el.classList.remove("visible");
        el.innerHTML = "";
    }
    _popoverNode = null;
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
    if (backend === "off") return;

    const quant   = _getSetting("mec.node_explain.gguf_quant", "Q4_K_M");
    const cls     = node.type;
    const el      = _getPopover();

    // If already showing for this node, do nothing
    if (_popoverNode === node && el.classList.contains("visible")) return;

    _popoverNode = node;
    _positionPopover(el, node);

    // JS cache hit
    if (_CACHE.has(cls)) {
        _renderExplanation(el, _CACHE.get(cls));
        return;
    }

    // Show loading spinner
    _showLoading(el, node.title || cls);
    // Re-position after showing (now has height)
    requestAnimationFrame(() => _positionPopover(el, node));

    if (_PENDING.has(cls)) return;   // request already in-flight
    _PENDING.add(cls);

    try {
        const url = `/mec/node_explain/${encodeURIComponent(cls)}?backend=${backend}&quant=${quant}`;
        const resp = await fetch(url);
        const json = await resp.json();
        _PENDING.delete(cls);

        if (!json.success) {
            // Only show error if this node is still the one being hovered
            if (_popoverNode === node) _showError(el, json.message || "Failed to load explanation.");
            return;
        }

        const data = json.data;
        _CACHE.set(cls, data);

        if (_popoverNode === node && el.classList.contains("visible")) {
            _renderExplanation(el, data);
            _positionPopover(el, node);
        }
    } catch (err) {
        _PENDING.delete(cls);
        if (_popoverNode === node) _showError(el, `Network error: ${err.message}`);
    }
}

// ── canvas event handlers ──────────────────────────────────────────────────
function _onMouseMove(e) {
    // Ignore if user is dragging canvas (left button held)
    if (e.buttons & 1) {
        _clearDwell();
        _nodeHovered = false;
        _scheduleHide();
        return;
    }

    const { gx, gy } = _toGraph(e);
    const node = _nodeAtTitle(gx, gy);

    if (node === _currentNode) return;    // same node, no change

    // Node changed → reset dwell
    _clearDwell();
    _currentNode = node;

    if (!node) {
        _nodeHovered = false;
        _scheduleHide();
        return;
    }

    _nodeHovered = true;
    _clearHide();

    const backend = _getSetting("mec.node_explain.backend", "auto");
    if (backend === "off") return;

    _dwellTimer = setTimeout(() => {
        _dwellTimer = null;
        if (_currentNode === node) {
            _fetchAndShow(node);
        }
    }, DWELL_MS);
}

function _onMouseLeave() {
    _clearDwell();
    _nodeHovered  = false;
    _currentNode  = null;
    _scheduleHide();
}

// ── extension registration ─────────────────────────────────────────────────
app.registerExtension({
    name: "MEC.NodeExplain",

    settings: [
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
            _scheduleHide();
        });

        console.log("[MEC.NodeExplain] Loaded — hover a node title for 800 ms to explain it.");
    },
});
