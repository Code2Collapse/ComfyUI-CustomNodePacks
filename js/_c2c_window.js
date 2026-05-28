/**
 * _c2c_window.js — shared window-manager primitives for every C2C AI panel.
 *
 * What this gives every panel:
 *   • cursorAnchor(el, ev)   — position next to the mouse pointer with smart
 *                              edge-flip + a faint connector line so the user
 *                              always knows what triggered the popover.
 *   • makeDraggable(el, opts)— grab the header to move the panel; position
 *                              is persisted under localStorage[opts.key].
 *   • makeResizable(el, opts)— grab the bottom-right corner to resize; size
 *                              is persisted under localStorage[opts.key + ":size"].
 *   • makePinnable(el, btn)  — toggle button that locks the panel in place
 *                              and prevents auto-close timers from firing.
 *   • cascade(el, key)       — initial position cascades so multiple open
 *                              panels never sit on top of each other.
 *   • drawConnector(fromXY, toRect)
 *                            — thin SVG line from the trigger point to the
 *                              panel; auto-disposes when either disappears.
 *
 * Catppuccin tokens are centralised here so every panel renders with the
 * same palette.
 */
import { C } from "./_c2c_theme.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

// Re-export the theme palette so downstream panels can pull both window
// primitives and the colour tokens from a single import.
export { C };

// ── Reserved screen real estate ────────────────────────────────────────────
// ComfyUI's native top toolbar (logo, menus, workflow tabs) lives in roughly
// the first 56 px of the viewport. NO C2C overlay may intrude into this
// gutter — every helper below enforces this minimum top.
export const TOOLBAR_GUTTER = 60;

/** Clamp a top-pixel value so the overlay never covers the native toolbar. */
export function safeTop(top, height = 0) {
    const vh = window.innerHeight;
    const min = TOOLBAR_GUTTER;
    const max = Math.max(min, vh - height - 4);
    return Math.max(min, Math.min(top, max));
}

// ── Z-order / focus stack ─────────────────────────────────────────────────
// Classic OS behavior: any mousedown anywhere inside a C2C window raises
// it above all other C2C windows. The focused window also gets a subtle
// neon accent (see CSS [data-c2c-focused="true"]).
const _Z_BASE = 5000;
const _Z_CAP  = 8900;        // stay below OmniPill bar (9000) and toasts (10000)
let   _zTop   = _Z_BASE;
let   _focused = null;       // currently-focused .c2c-win element

/** Bring `el` to the top of the C2C z-stack and mark it focused. Idempotent. */
export function bringToFront(el) {
    if (!el || !el.classList || !el.classList.contains("c2c-win")) return;
    if (_focused === el && parseInt(el.style.zIndex || "0", 10) === _zTop) return;
    // Periodic normalize: if we ever climb close to the cap, reseat all
    // visible C2C panels in their current visual order back into [base, base+N].
    if (_zTop >= _Z_CAP) {
        const panels = Array.from(document.querySelectorAll(".c2c-win"))
            .sort((a, b) => (parseInt(a.style.zIndex||"0",10) - parseInt(b.style.zIndex||"0",10)));
        _zTop = _Z_BASE;
        panels.forEach(p => { p.style.zIndex = String(++_zTop); });
    }
    el.style.zIndex = String(++_zTop);
    if (_focused && _focused !== el) {
        try { _focused.dataset.c2cFocused = "false"; } catch (__e) { /* detached */ }
    }
    el.dataset.c2cFocused = "true";
    _focused = el;
}

/** Wire a panel so any mousedown inside it claims focus. Idempotent. */
export function attachFocusOnInteract(el) {
    if (!el || el.dataset.c2cFocusWired === "true") return;
    el.dataset.c2cFocusWired = "true";
    // Capture phase so we win before LiteGraph canvas listeners; but we do
    // NOT stopPropagation — buttons, inputs, drag/resize handlers must all
    // still receive the event normally.
    el.addEventListener("mousedown", () => bringToFront(el), true);
    el.addEventListener("focusin",   () => bringToFront(el), true);
}

// ── shared style sheet (injected once) ─────────────────────────────────────
const STYLE_ID = "c2c-window-style";
function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.c2c-win {
    position: fixed;
    /* Sit above ComfyUI's native Inspector (z<=1300) and body docks
       (z<=1001) but below toasts (10000) and the OmniPill bar (9000) so
       the launcher stays always-visible. bringToFront() bumps the focused
       window up via a per-session counter starting at 5000 (cap 8900). */
    z-index: var(--c2c-z-panel, 5000);
    background: var(--c2c-bg);
    color: var(--c2c-fg);
    border: 1px solid var(--c2c-border);
    border-radius: 10px;
    box-shadow:
        0 18px 48px color-mix(in srgb, var(--c2c-shadowBase) 65%, transparent),
        0 0 0 1px color-mix(in srgb, var(--c2c-shadowBase) 40%, transparent);
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    font-size: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-width: 280px;
    min-height: 120px;
    backdrop-filter: blur(6px);
}
.c2c-win-header {
    padding: 8px 12px;
    background: linear-gradient(180deg, var(--c2c-bg2) 0%, var(--c2c-bg3) 100%);
    border-bottom: 1px solid var(--c2c-border);
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: move;
    user-select: none;
    flex-shrink: 0;
}
.c2c-win-title {
    color: var(--c2c-mauve);
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 11px;
    flex-shrink: 0;
}
.c2c-win-subtitle {
    flex: 1;
    color: var(--c2c-sub);
    font-size: 11px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.c2c-win-btn {
    background: transparent;
    color: var(--c2c-sub);
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 2px 7px;
    cursor: pointer;
    font-size: 11px;
    line-height: 1.2;
    transition: background 0.12s, color 0.12s;
}
.c2c-win-btn:hover { background: var(--c2c-border); color: var(--c2c-fg); }
.c2c-win-btn.primary { background: var(--c2c-mauve); color: var(--c2c-bg); font-weight: 600; }
.c2c-win-btn.primary:hover { filter: brightness(1.1); }
.c2c-win-btn.success { background: var(--c2c-green); color: var(--c2c-bg); font-weight: 600; }
.c2c-win-btn.success:hover { filter: brightness(1.1); }
.c2c-win-btn.danger { background: var(--c2c-red); color: var(--c2c-bg); font-weight: 600; }
.c2c-win-btn.icon { padding: 2px 6px; font-size: 14px; line-height: 1; }
.c2c-win-btn[data-pinned="true"] { background: var(--c2c-peach); color: var(--c2c-bg); }
.c2c-win-tabs {
    display: flex;
    background: var(--c2c-bg2);
    border-bottom: 1px solid var(--c2c-border);
    overflow-x: auto;
    scrollbar-width: none;
    flex-shrink: 0;
}
.c2c-win-tabs::-webkit-scrollbar { display: none; }
.c2c-win-tab {
    padding: 6px 12px;
    color: var(--c2c-sub);
    cursor: pointer;
    border: none;
    background: transparent;
    font-size: 11px;
    font-weight: 500;
    border-bottom: 2px solid transparent;
    transition: color 0.12s, border-color 0.12s;
    white-space: nowrap;
}
.c2c-win-tab:hover { color: var(--c2c-fg); }
.c2c-win-tab.active { color: var(--c2c-mauve); border-bottom-color: var(--c2c-mauve); }
.c2c-win-body {
    flex: 1;
    overflow: auto;
    padding: 12px;
    line-height: 1.5;
}
.c2c-win-body code, .c2c-win-body pre {
    background: var(--c2c-bg3);
    border: 1px solid var(--c2c-border);
    border-radius: 4px;
    padding: 1px 5px;
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
    font-size: 11px;
    color: var(--c2c-peach);
}
.c2c-win-body pre {
    padding: 8px 10px;
    overflow-x: auto;
    white-space: pre;
}
.c2c-win-footer {
    padding: 5px 12px;
    background: var(--c2c-bg2);
    border-top: 1px solid var(--c2c-border);
    color: var(--c2c-dim);
    font-size: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
}
.c2c-win-resize {
    position: absolute;
    right: 0; bottom: 0;
    width: 14px; height: 14px;
    cursor: nwse-resize;
    background:
        linear-gradient(135deg, transparent 0 6px, var(--c2c-dim) 6px 7px, transparent 7px 9px, var(--c2c-dim) 9px 10px, transparent 10px);
    opacity: 0.7;
}
.c2c-win-resize:hover { opacity: 1; }
/* 8-edge resize handles. The corner grip above stays as a visual hint;
   the 4 edges + 4 corners listed here cover ALL drag directions. */
.c2c-win-edge { position: absolute; z-index: 5; }
.c2c-win-edge.n  { top:0;    left:8px;   right:8px;  height:5px; cursor: ns-resize; }
.c2c-win-edge.s  { bottom:0; left:8px;   right:8px;  height:5px; cursor: ns-resize; }
.c2c-win-edge.e  { right:0;  top:8px;    bottom:8px; width:5px;  cursor: ew-resize; }
.c2c-win-edge.w  { left:0;   top:8px;    bottom:8px; width:5px;  cursor: ew-resize; }
.c2c-win-edge.nw { top:0;    left:0;     width:10px; height:10px; cursor: nwse-resize; }
.c2c-win-edge.ne { top:0;    right:0;    width:10px; height:10px; cursor: nesw-resize; }
.c2c-win-edge.sw { bottom:0; left:0;     width:10px; height:10px; cursor: nesw-resize; }
.c2c-win-edge.se { bottom:0; right:0;    width:10px; height:10px; cursor: nwse-resize; }
.c2c-win-edge:hover { background: color-mix(in srgb, var(--c2c-mauve) 22%, transparent); }
/* Minimized state: hide everything below the header. The header keeps
   its drag handle so the user can still relocate the chip.
   Two complementary rule sets so this works for BOTH structures:
     A) buildPanel-style panels with explicit .c2c-win-body / -tabs / -footer
     B) attachWindowChrome retrofits where the panel has a raw <h3>/<div>
        header tagged with [data-c2c-header="true"] and free-form siblings. */
.c2c-win[data-minimized="true"] { min-height: 0 !important; height: auto !important; max-height: none !important; overflow: hidden !important; }
.c2c-win[data-minimized="true"] .c2c-win-tabs,
.c2c-win[data-minimized="true"] .c2c-win-body,
.c2c-win[data-minimized="true"] .c2c-win-footer { display: none !important; }
.c2c-win[data-minimized="true"] > *:not([data-c2c-header]):not(.c2c-win-header):not(.c2c-win-tabs):not(.c2c-win-body):not(.c2c-win-footer):not(.c2c-win-edge):not(.c2c-win-resize) {
    display: none !important;
}
.c2c-win[data-minimized="true"] .c2c-win-edge.s,
.c2c-win[data-minimized="true"] .c2c-win-edge.sw,
.c2c-win[data-minimized="true"] .c2c-win-edge.se,
.c2c-win[data-minimized="true"] .c2c-win-edge.e,
.c2c-win[data-minimized="true"] .c2c-win-edge.w,
.c2c-win[data-minimized="true"] .c2c-win-resize { display: none !important; }
/* attachWindowChrome injects a "–" button on a sibling header that was not
   originally a c2c-win-header. Style it so it visually matches the rest of
   the chrome instead of being a bare oversized minus sign. */
.c2c-win [data-c2c-min] {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--c2c-sub) !important;
    font-size: 13px !important;
    line-height: 1 !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
    cursor: pointer !important;
    transition: background 0.12s, color 0.12s;
    margin-left: 4px;
}
.c2c-win [data-c2c-min]:hover { background: var(--c2c-border) !important; color: var(--c2c-fg) !important; }
/* Mark the autodetected header as draggable visually. Raise z-index above
   the absolutely-positioned 5-px N edge handle (z-index:5) so clicks on
   the top of the header are caught by the header, not the resize strip. */
.c2c-win [data-c2c-header] {
    cursor: move !important;
    user-select: none;
    position: relative;
    z-index: var(--c2c-z-header, 10);
}
/* Pointer events live on the header itself — buttons inside still receive
   their own clicks because they bubble first. */

/* ── Focused-window accent ─────────────────────────────────────────────
   When a panel is the most-recently clicked C2C window, bringToFront()
   sets [data-c2c-focused="true"]. We highlight the border + add a subtle
   neon glow so the user can see which window owns input focus. */
.c2c-win[data-c2c-focused="true"] {
    border-color: color-mix(in srgb, var(--c2c-mauve) 55%, var(--c2c-border));
    box-shadow:
        0 18px 48px color-mix(in srgb, var(--c2c-shadowBase) 75%, transparent),
        0 0 0 1px color-mix(in srgb, var(--c2c-mauve) 35%, transparent),
        0 0 18px color-mix(in srgb, var(--c2c-mauve) 22%, transparent);
}
.c2c-win[data-c2c-focused="true"] .c2c-win-header,
.c2c-win[data-c2c-focused="true"] [data-c2c-header] {
    border-bottom-color: color-mix(in srgb, var(--c2c-mauve) 45%, var(--c2c-border));
}
.c2c-win[data-c2c-focused="true"] .c2c-win-title {
    color: color-mix(in srgb, var(--c2c-mauve) 80%, white);
    text-shadow: 0 0 6px color-mix(in srgb, var(--c2c-mauve) 45%, transparent);
}
/* Shortcut hint inline in the title (e.g. "WIZARD (CTRL+SHIFT+P)"). */
.c2c-win-shortcut {
    color: var(--c2c-sub);
    font-weight: 400;
    font-size: 10px;
    letter-spacing: 0.02em;
    margin-left: 6px;
    opacity: 0.8;
    text-transform: uppercase;
}
.c2c-win-chip {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 1px 7px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
    background: var(--c2c-bg3);
    color: var(--c2c-sub);
    border: 1px solid var(--c2c-border);
}
.c2c-win-chip.error   { color: var(--c2c-red);    border-color: var(--c2c-red); }
.c2c-win-chip.warn    { color: var(--c2c-yellow); border-color: var(--c2c-yellow); }
.c2c-win-chip.ok      { color: var(--c2c-green);  border-color: var(--c2c-green); }
.c2c-win-chip.info    { color: var(--c2c-blue);   border-color: var(--c2c-blue); }
.c2c-connector {
    position: fixed;
    pointer-events: none;
    z-index: var(--c2c-z-modal);
    overflow: visible;
    left: 0; top: 0;
    width: 0; height: 0;
}
.c2c-connector line {
    stroke: var(--c2c-mauve);
    stroke-width: 1.5;
    stroke-dasharray: 4 4;
    opacity: 0.55;
    animation: c2c-dash 0.6s linear infinite;
}
.c2c-connector circle.from { fill: var(--c2c-mauve); }
.c2c-connector circle.to   { fill: var(--c2c-mauve); opacity: 0.6; }
@keyframes c2c-dash { to { stroke-dashoffset: -16; } }
`;
    document.head.appendChild(s);
}
_injectStyle();

// ── persistent geometry ────────────────────────────────────────────────────
function _load(key) {
    try { return JSON.parse(localStorage.getItem(key) || "null"); } catch (_) { return null; }
}
function _save(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch (__c2cErr) { __c2cReport("_c2c_window", __c2cErr); }
}

// ── cursor-anchored positioning ────────────────────────────────────────────
/**
 * Position `el` near (clientX, clientY) with smart edge-flip and a small gap
 * so the cursor never overlaps the panel. Returns the chosen anchor side.
 */
export function cursorAnchor(el, clientX, clientY, { gap = 14, prefer = "right-below" } = {}) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const w  = el.offsetWidth  || 360;
    const h  = el.offsetHeight || 220;

    // Try preferred quadrant, flip if it would clip.
    let placeRight = !prefer.startsWith("left");
    let placeBelow = !prefer.endsWith("above");

    if (placeRight && clientX + gap + w > vw - 4) placeRight = false;
    if (!placeRight && clientX - gap - w < 4)     placeRight = true;
    if (placeBelow && clientY + gap + h > vh - 4) placeBelow = false;
    if (!placeBelow && clientY - gap - h < 4)     placeBelow = true;

    let left = placeRight ? clientX + gap : clientX - gap - w;
    let top  = placeBelow ? clientY + gap : clientY - gap - h;

    left = Math.max(4, Math.min(left, vw - w - 4));
    top  = safeTop(top, h);

    el.style.left = `${left}px`;
    el.style.top  = `${top}px`;
    return { left, top, w, h, placeRight, placeBelow };
}

// ── connector line (cursor → panel) ────────────────────────────────────────
const CONNECTOR_ID = "c2c-window-connector";
export function drawConnector(fromX, fromY, panelEl) {
    let svg = document.getElementById(CONNECTOR_ID);
    if (!svg) {
        svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.id = CONNECTOR_ID;
        svg.classList.add("c2c-connector");
        svg.setAttribute("width",  String(window.innerWidth));
        svg.setAttribute("height", String(window.innerHeight));
        svg.style.width  = "100vw";
        svg.style.height = "100vh";
        document.body.appendChild(svg);
    }
    svg.innerHTML = "";
    if (!panelEl) return;
    const r = panelEl.getBoundingClientRect();
    // Aim at the closest edge midpoint of the panel.
    const cx = r.left + r.width / 2;
    const cy = r.top  + r.height / 2;
    let toX, toY;
    if (Math.abs(fromX - cx) > Math.abs(fromY - cy)) {
        toX = fromX < cx ? r.left : r.right;
        toY = Math.max(r.top + 8, Math.min(r.bottom - 8, fromY));
    } else {
        toY = fromY < cy ? r.top : r.bottom;
        toX = Math.max(r.left + 8, Math.min(r.right - 8, fromX));
    }
    const NS = "http://www.w3.org/2000/svg";
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", fromX); line.setAttribute("y1", fromY);
    line.setAttribute("x2", toX);   line.setAttribute("y2", toY);
    svg.appendChild(line);
    const c1 = document.createElementNS(NS, "circle");
    c1.setAttribute("cx", fromX); c1.setAttribute("cy", fromY); c1.setAttribute("r", 3);
    c1.classList.add("from");
    svg.appendChild(c1);
    const c2 = document.createElementNS(NS, "circle");
    c2.setAttribute("cx", toX); c2.setAttribute("cy", toY); c2.setAttribute("r", 4);
    c2.classList.add("to");
    svg.appendChild(c2);
}
export function clearConnector() {
    document.getElementById(CONNECTOR_ID)?.remove();
    // Also detach any active node-anchor reflow loop.
    if (_anchorReflow) {
        try { _anchorReflow.detach(); } catch (__c2cErr) { __c2cReport("_c2c_window", __c2cErr); }
        _anchorReflow = null;
    }
}

// ── nodeAnchor: position a panel relative to a LiteGraph node ──────────────
//
// Why this exists:
//   Most C2C panels DESCRIBE a specific node (explainer, doctor finding,
//   error translator, face-overlay etc.). Anchoring to the cursor is wrong
//   because the cursor moves away; anchoring to a fixed screen corner is
//   wrong because the user has no visual binding. This helper computes the
//   node's body rect in client coords from graph coords (via the LiteGraph
//   canvas pan/zoom transform), picks the best of 4 sides (right > left >
//   below > above) that fits without clipping, draws a connector from the
//   chosen edge of the node to the panel, and re-fires on pan/zoom/resize
//   so the panel tracks the node until cleared.
//
// Usage:
//   import { nodeAnchor, clearConnector } from "./_c2c_window.js";
//   const handle = nodeAnchor(el, node, { gap: 12, prefer: "right" });
//   ...
//   handle.detach();  // or call clearConnector() to drop everything
let _anchorReflow = null;

export function nodeAnchor(el, node, opts = {}) {
    const { gap = 12, prefer = "right", track = true } = opts;
    if (!el || !node) return { detach() {} };

    const place = () => _placeRelativeToNode(el, node, gap, prefer);

    // First placement.
    place();

    // Re-place on canvas pan/zoom and window resize so the panel tracks.
    if (!track) return { detach() {} };

    const canvas = (typeof window !== "undefined" && window.app?.canvas) || null;
    const cvsEl = canvas?.canvas || null;
    let rafPending = false;
    const requestPlace = () => {
        if (rafPending) return;
        rafPending = true;
        requestAnimationFrame(() => { rafPending = false; place(); });
    };
    window.addEventListener("resize", requestPlace, { passive: true });
    if (cvsEl) {
        cvsEl.addEventListener("wheel", requestPlace, { passive: true });
        cvsEl.addEventListener("mousemove", requestPlace, { passive: true });
        cvsEl.addEventListener("mouseup", requestPlace, { passive: true });
    }

    const handle = {
        detach() {
            window.removeEventListener("resize", requestPlace);
            if (cvsEl) {
                cvsEl.removeEventListener("wheel", requestPlace);
                cvsEl.removeEventListener("mousemove", requestPlace);
                cvsEl.removeEventListener("mouseup", requestPlace);
            }
            if (_anchorReflow === handle) _anchorReflow = null;
        },
    };
    // Replace any prior anchor's listener set so clearConnector() drops it.
    if (_anchorReflow) { try { _anchorReflow.detach(); } catch (__c2cErr) { __c2cReport("_c2c_window", __c2cErr); } }
    _anchorReflow = handle;
    return handle;
}

/** Pure layout step used by nodeAnchor. Returns { side, fromX, fromY }. */
function _placeRelativeToNode(el, node, gap, prefer) {
    const canvas = window.app?.canvas;
    if (!canvas || !node) return null;
    const cvsEl = canvas.canvas;
    if (!cvsEl) return null;

    const rect = cvsEl.getBoundingClientRect();
    const ds = canvas.ds || { scale: 1, offset: [0, 0] };
    const s = ds.scale || 1;
    const ox = ds.offset?.[0] ?? 0;
    const oy = ds.offset?.[1] ?? 0;

    // Node body in graph-space. LiteGraph stores pos as the top-left of the
    // node BODY (under the title bar). The title bar lives in y ∈ [pos.y-30, pos.y].
    const TITLE = 30;
    const nx = node.pos[0];
    const ny = node.pos[1];
    const nw = (node.size && node.size[0]) || 200;
    const nh = (node.size && node.size[1]) || 60;
    const titleTop = ny - TITLE;
    const bodyBot  = ny + nh;

    // Graph → client transform.
    const toX = (gx) => gx * s + ox + rect.left;
    const toY = (gy) => gy * s + oy + rect.top;

    const nodeLeft   = toX(nx);
    const nodeRight  = toX(nx + nw);
    const nodeTop    = toY(titleTop);
    const nodeBottom = toY(bodyBot);
    const nodeMidX   = (nodeLeft + nodeRight) / 2;
    const nodeMidY   = (nodeTop + nodeBottom) / 2;

    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const w  = el.offsetWidth  || 360;
    const h  = el.offsetHeight || 240;

    // Candidate placements, in priority order based on `prefer`.
    const cands = {
        right: {
            left: nodeRight + gap,
            top:  Math.max(4, Math.min(nodeTop, vh - h - 4)),
            fits: () => nodeRight + gap + w <= vw - 4,
            from: () => ({ x: nodeRight, y: nodeMidY, side: "right" }),
        },
        left: {
            left: nodeLeft - gap - w,
            top:  Math.max(4, Math.min(nodeTop, vh - h - 4)),
            fits: () => nodeLeft - gap - w >= 4,
            from: () => ({ x: nodeLeft, y: nodeMidY, side: "left" }),
        },
        below: {
            left: Math.max(4, Math.min(nodeMidX - w / 2, vw - w - 4)),
            top:  nodeBottom + gap,
            fits: () => nodeBottom + gap + h <= vh - 4,
            from: () => ({ x: nodeMidX, y: nodeBottom, side: "below" }),
        },
        above: {
            left: Math.max(4, Math.min(nodeMidX - w / 2, vw - w - 4)),
            top:  nodeTop - gap - h,
            fits: () => nodeTop - gap - h >= 4,
            from: () => ({ x: nodeMidX, y: nodeTop, side: "above" }),
        },
    };

    const order = (() => {
        if (prefer === "left")  return ["left", "right", "below", "above"];
        if (prefer === "below") return ["below", "above", "right", "left"];
        if (prefer === "above") return ["above", "below", "right", "left"];
        return ["right", "left", "below", "above"];
    })();

    let chosen = order.find(k => cands[k].fits()) || order[0];
    const c = cands[chosen];
    let left = c.left;
    let top  = c.top;

    // Final viewport clamp (covers the no-fit fallback).
    left = Math.max(4, Math.min(left, vw - w - 4));
    top  = safeTop(top, h);

    el.style.position = "fixed";
    el.style.left = `${left}px`;
    el.style.top  = `${top}px`;
    el.dataset.anchorSide = chosen;

    const f = c.from();
    drawConnector(f.x, f.y, el);
    return { side: chosen, fromX: f.x, fromY: f.y };
}

// ── drag / resize / pin ────────────────────────────────────────────────────
export function makeDraggable(el, { handle, handleSelector, key } = {}) {
    // When `handleSelector` is given we bind mousedown to the panel root and
    // delegate via `e.target.closest(handleSelector)` so dynamic re-renders
    // (innerHTML wipes that replace the header element) keep working.
    const h = handleSelector ? el : (handle || el.querySelector(".c2c-win-header") || el);
    if (!h) return;
    if (key) {
        const saved = _load(`c2c.win.${key}`);
        if (saved && Number.isFinite(saved.left) && Number.isFinite(saved.top)) {
            // Clamp to current viewport so a smaller screen doesn't lose the panel.
            const vw = window.innerWidth, vh = window.innerHeight;
            el.style.left = `${Math.max(4, Math.min(saved.left, vw - 80))}px`;
            el.style.top  = `${Math.max(4, Math.min(saved.top,  vh - 80))}px`;
            el.style.right = ""; el.style.bottom = "";
            // Kill any CSS centering transform so the inline left/top wins.
            el.style.transform = "none";
        }
    }
    let sx = 0, sy = 0, ox = 0, oy = 0, dragging = false;
    let lockedW = 0, lockedH = 0;
    h.addEventListener("mousedown", (e) => {
        // ignore clicks on buttons inside the header
        if (e.target.closest("button, .c2c-win-tab, [data-c2c-min], input, select, textarea, a, label")) return;
        if (handleSelector && !e.target.closest(handleSelector)) return;
        // Also bail if pointer landed on a resize edge (those sit absolutely
        // and may overlap the very top/bottom strip of the header).
        if (e.target.classList?.contains?.("c2c-win-edge")) return;
        dragging = true;
        sx = e.clientX; sy = e.clientY;
        const r = el.getBoundingClientRect();
        ox = r.left; oy = r.top;
        // Lock current dimensions so any latent CSS bottom/right (from a
        // panel's own stylesheet) cannot stretch the box mid-drag.
        lockedW = r.width; lockedH = r.height;
        el.style.width  = `${lockedW}px`;
        el.style.height = `${lockedH}px`;
        el.style.left = `${ox}px`; el.style.top = `${oy}px`;
        // Use "auto" (not empty string) so CSS bottom/right cannot reassert.
        el.style.right = "auto"; el.style.bottom = "auto";
        // CRITICAL: kill any CSS centering transform (translate(-50%,-50%))
        // — getBoundingClientRect() already returns the post-transform visible
        // rect, so we anchored left/top to it. If transform stayed applied,
        // the panel would re-shift -W/2,-H/2 and snap to the upper-left corner
        // on the very first drag tick. Set inline so it wins over stylesheet.
        el.style.transform = "none";
        document.body.style.userSelect = "none";
        e.preventDefault();
        e.stopPropagation();
    }, true);
    window.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const nl = ox + (e.clientX - sx);
        const nt = oy + (e.clientY - sy);
        // Use the LOCKED dimensions captured at mousedown — never the live
        // offsetWidth/Height which can fluctuate if any stylesheet rule wins
        // over inline styles. lockedW/H stays constant for the whole drag.
        const w  = lockedW || el.offsetWidth  || 80;
        const h  = lockedH || el.offsetHeight || 80;
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        // Clamp by full bbox so the panel is always fully reachable and
        // never tucks behind ComfyUI's native top toolbar.
        el.style.left = `${Math.max(4, Math.min(nl, vw - w - 4))}px`;
        el.style.top  = `${Math.max(TOOLBAR_GUTTER, Math.min(nt, vh - h - 4))}px`;
        e.preventDefault();
    });
    window.addEventListener("mouseup", () => {
        if (!dragging) return;
        dragging = false;
        document.body.style.userSelect = "";
        if (key) {
            const r = el.getBoundingClientRect();
            _save(`c2c.win.${key}`, { left: r.left, top: r.top });
        }
    });
}

export function makeResizable(el, { key, minW = 280, minH = 160 } = {}) {
    if (key) {
        const saved = _load(`c2c.win.${key}:size`);
        if (saved && saved.w && saved.h) {
            el.style.width  = `${saved.w}px`;
            el.style.height = `${saved.h}px`;
        }
    }
    // Visual corner hint (purely decorative; the real handler is the SE edge).
    const grip = document.createElement("div");
    grip.className = "c2c-win-resize";
    el.appendChild(grip);
    // 8 invisible edge handles cover every drag direction.
    const DIRS = ["n", "s", "e", "w", "nw", "ne", "sw", "se"];
    DIRS.forEach(dir => {
        const e = document.createElement("div");
        e.className = `c2c-win-edge ${dir}`;
        e.dataset.dir = dir.toUpperCase();
        el.appendChild(e);
        _attachEdgeResize(e, el, dir.toUpperCase(), { minW, minH, key });
    });
}

function _attachEdgeResize(edgeEl, el, dir, { minW, minH, key }) {
    const hasN = dir.includes("N"), hasS = dir.includes("S");
    const hasE = dir.includes("E"), hasW = dir.includes("W");
    edgeEl.addEventListener("mousedown", (ev) => {
        if (el.dataset.minimized === "true") return; // can't resize collapsed panel
        ev.preventDefault(); ev.stopPropagation();
        const startX = ev.clientX, startY = ev.clientY;
        const r = el.getBoundingClientRect();
        const startW = r.width, startH = r.height;
        const startL = r.left,  startT  = r.top;
        // Lock to left/top so we can stretch from any edge without anchor drift.
        el.style.left = `${startL}px`; el.style.top = `${startT}px`;
        // "auto" — not empty — so the panel's own stylesheet `bottom`/`right`
        // cannot reassert and stretch the element while resizing.
        el.style.right = "auto"; el.style.bottom = "auto";
        // Kill CSS centering transform for the same reason as makeDraggable —
        // rect is already post-transform so transform must be cleared to avoid
        // a -W/2,-H/2 jump on the first resize tick.
        el.style.transform = "none";
        document.body.style.userSelect = "none";
        const onMove = (e2) => {
            const dx = e2.clientX - startX, dy = e2.clientY - startY;
            let w = startW, h = startH, l = startL, t = startT;
            if (hasE) { w = startW + dx; }
            if (hasW) { w = startW - dx; l = startL + dx; }
            if (hasS) { h = startH + dy; }
            if (hasN) { h = startH - dy; t = startT + dy; }
            w = Math.max(minW, Math.min(window.innerWidth  - 8, w));
            h = Math.max(minH, Math.min(window.innerHeight - 8, h));
            l = Math.max(4,             Math.min(window.innerWidth  - w - 4, l));
            t = Math.max(TOOLBAR_GUTTER, Math.min(window.innerHeight - h - 4, t));
            el.style.width  = `${w}px`;
            el.style.height = `${h}px`;
            el.style.left   = `${l}px`;
            el.style.top    = `${t}px`;
        };
        const onUp = () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup",   onUp);
            document.body.style.userSelect = "";
            if (key) {
                const r2 = el.getBoundingClientRect();
                _save(`c2c.win.${key}:size`, { w: r2.width, h: r2.height });
                _save(`c2c.win.${key}`,      { left: r2.left,  top: r2.top });
            }
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup",   onUp);
    });
}

/** Public helper: clamp `el` into the viewport, honouring TOOLBAR_GUTTER. */
export function clampToViewport(el) {
    const r = el.getBoundingClientRect();
    const vw = window.innerWidth, vh = window.innerHeight;
    const l = Math.max(4,             Math.min(r.left, vw - r.width  - 4));
    const t = Math.max(TOOLBAR_GUTTER, Math.min(r.top,  vh - r.height - 4));
    el.style.left = `${l}px`;
    el.style.top  = `${t}px`;
    el.style.right = ""; el.style.bottom = "";
}

/** Toggle minimized state on a buildPanel-built window. Persists under
 *  `c2c.win.<key>:min`. On EXPAND, reclamps the now-tall panel so nothing
 *  spills off-screen. */
export function toggleMinimize(el, key) {
    const wasMin = el.dataset.minimized === "true";
    el.dataset.minimized = wasMin ? "false" : "true";
    if (key) _save(`c2c.win.${key}:min`, !wasMin);
    if (wasMin) {
        // EXPAND: wait for layout to apply, then clamp.
        requestAnimationFrame(() => clampToViewport(el));
    }
}

export function makePinnable(el, btn, key) {
    btn.dataset.pinned = "false";
    btn.title = "Pin (prevent auto-close)";
    btn.textContent = "📌";
    // Restore pinned state from localStorage (if a storageKey was provided)
    if (key) {
        const saved = _load(`c2c.win.${key}:pinned`);
        if (saved === true) {
            btn.dataset.pinned = "true";
            el.dataset.pinned = "true";
            btn.title = "Unpin";
        }
    }
    btn.onclick = () => {
        const next = btn.dataset.pinned !== "true";
        btn.dataset.pinned = next ? "true" : "false";
        el.dataset.pinned = next ? "true" : "false";
        btn.title = next ? "Unpin" : "Pin (prevent auto-close)";
        if (key) _save(`c2c.win.${key}:pinned`, next);
    };
}

// ── cascade initial position when multiple panels are open ─────────────────
let _cascadeIndex = 0;
export function cascade(el, key) {
    const saved = key ? _load(`c2c.win.${key}`) : null;
    if (saved) return; // already-saved position wins
    const open = document.querySelectorAll("[data-c2c-panel='true']").length;
    const base = 64;
    const step = 32;
    const off  = base + (open % 6) * step;
    el.style.top   = `${off}px`;
    el.style.right = "";
    el.style.left  = `${window.innerWidth - (el.offsetWidth || 440) - off}px`;
    _cascadeIndex++;
}

// ── build a standard panel shell ───────────────────────────────────────────
/**
 * Build a fully wired panel (header w/ title, subtitle, action slot, pin,
 * close; optional tabs row; scrollable body; footer with status+meta).
 * Returns refs to every part so the caller doesn't have to query.
 */
export function buildPanel({
    id,
    title,
    shortcut = null,       // e.g. "Ctrl+Shift+P" — rendered as a muted hint
    width = 460,
    height = 480,
    tabs = null,           // [{ key, label }] or null for no tab bar
    onClose = null,
    actions = [],          // [{ label, className, title, onClick, id }]
    storageKey = null,     // also enables drag/resize persistence
} = {}) {
    const existing = document.getElementById(id);
    if (existing) return _refs(existing);
    const el = document.createElement("div");
    el.id = id;
    el.className = "c2c-win";
    el.dataset.c2cPanel = "true";
    el.style.width  = `${width}px`;
    el.style.height = `${height}px`;

    const tabBar = tabs && tabs.length
        ? `<div class="c2c-win-tabs">${tabs.map((t, i) => `<button class="c2c-win-tab${i === 0 ? " active" : ""}" data-tab="${t.key}">${t.label}</button>`).join("")}</div>`
        : "";
    const actionBtns = actions.map(a =>
        `<button class="c2c-win-btn ${a.className || ""}" data-action="${a.id || a.label}" title="${a.title || a.label}">${a.label}</button>`
    ).join("");
    const titleHTML = shortcut
        ? `${esc(title)} <span class="c2c-win-shortcut">(${esc(shortcut)})</span>`
        : esc(title);
    el.innerHTML = `
        <div class="c2c-win-header">
            <span class="c2c-win-title">${titleHTML}</span>
            <span class="c2c-win-subtitle"></span>
            ${actionBtns}
            <button class="c2c-win-btn icon" data-act="pin"></button>
            <button class="c2c-win-btn icon" data-act="min" title="Minimize / restore">–</button>
            <button class="c2c-win-btn icon" data-act="close" title="Close">×</button>
        </div>
        ${tabBar}
        <div class="c2c-win-body"></div>
        <div class="c2c-win-footer">
            <span class="status"></span>
            <span class="meta"></span>
        </div>`;
    document.body.appendChild(el);

    el.querySelector("[data-act='close']").onclick = () => {
        if (el.dataset.pinned === "true") return; // pinned: ignore close shortcut from outside
        el.remove();
        clearConnector();
        if (typeof onClose === "function") onClose();
    };
    el.querySelector("[data-act='min']").onclick = (ev) => {
        ev.stopPropagation();
        toggleMinimize(el, storageKey);
    };
    makePinnable(el, el.querySelector("[data-act='pin']"), storageKey);
    actions.forEach(a => {
        const b = el.querySelector(`[data-action="${a.id || a.label}"]`);
        if (b && typeof a.onClick === "function") b.onclick = () => a.onClick(el);
    });
    cascade(el, storageKey);
    makeDraggable(el, { key: storageKey });
    makeResizable(el, { key: storageKey });
    attachFocusOnInteract(el);
    bringToFront(el);  // newly-opened panel claims focus

    // Restore persisted minimized state (after drag/resize so saved
    // position takes precedence over cascade defaults).
    if (storageKey && _load(`c2c.win.${storageKey}:min`) === true) {
        el.dataset.minimized = "true";
    }

    // tab switching
    if (tabs && tabs.length) {
        el.querySelectorAll(".c2c-win-tab").forEach(btn => {
            btn.onclick = () => {
                el.querySelectorAll(".c2c-win-tab").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                el.dataset.activeTab = btn.dataset.tab;
                el.dispatchEvent(new CustomEvent("c2c:tab", { detail: { tab: btn.dataset.tab } }));
            };
        });
        el.dataset.activeTab = tabs[0].key;
    }
    return _refs(el);
}

function _refs(el) {
    return {
        el,
        header:   el.querySelector(".c2c-win-header"),
        title:    el.querySelector(".c2c-win-title"),
        subtitle: el.querySelector(".c2c-win-subtitle"),
        body:     el.querySelector(".c2c-win-body"),
        status:   el.querySelector(".c2c-win-footer .status"),
        meta:     el.querySelector(".c2c-win-footer .meta"),
        setSubtitle: (txt) => {
            // Accept either plain text (default) or an HTML string. Callers
            // that previously relied on chip markup landing in the subtitle
            // (e.g. workflow doctor's severity chips) would otherwise see
            // raw `<span class=…>` leak as text content.
            const node = el.querySelector(".c2c-win-subtitle");
            const s = txt || "";
            if (typeof s === "string" && /<[a-z!\/]/i.test(s)) {
                node.innerHTML = s;
            } else {
                node.textContent = s;
            }
        },
        setStatus:   (txt) => { el.querySelector(".c2c-win-footer .status").textContent = txt || ""; },
        setMeta:     (txt) => { el.querySelector(".c2c-win-footer .meta").textContent = txt || ""; },
    };
}

// HTML escape helper used by every panel.
export function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, c =>
        ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]));
}

/**
 * attachWindowChrome — additive retrofit for an EXISTING panel (modal,
 * aside, etc.) that gives it the same drag/resize/minimize/shortcut-hint
 * affordances as `buildPanel`, WITHOUT rewriting the panel's internal
 * layout.
 *
 * The panel can keep its modal overlay backdrop on first open. As soon
 * as the user grabs the header (drag), an edge (resize) or the minimize
 * button, the panel "undocks" from the overlay: backdrop turns
 * transparent + click-through, panel switches to `position:fixed` with
 * explicit coordinates, and from then on behaves like a free-floating
 * window. Geometry is persisted under `c2c.win.<storageKey>` (and
 * `:size`, `:min`) so the next open restores the floating state.
 *
 *   attachWindowChrome(panel, {
 *     storageKey: "model_browser",
 *     overlay:    overlayEl,         // optional outer dim layer
 *     header:     headerEl,          // drag handle row
 *     titleEl:    panel.querySelector(".my-title"),  // node to append (…) hint
 *     shortcut:   "Ctrl+Shift+M",
 *     minW: 360, minH: 220,
 *   });
 */
export function attachWindowChrome(panel, {
    storageKey = null,
    overlay    = null,
    header     = null,
    headerSelector = null,
    titleEl    = null,
    titleSelector = null,
    shortcut   = null,
    minW       = 320,
    minH       = 200,
} = {}) {
    if (!panel) return;
    const _isInit = panel.dataset.c2cChrome === "true";

    // 3) Undock helper: convert modal layout → free-floating window.
    const _undock = () => {
        // Always clear bottom/right anchors. Many OmniPill panels anchor in
        // their own stylesheet with `bottom:<n>px; right:<n>px;` AND we set
        // inline `top`/`left` — `position:fixed` then stretches the element
        // between top+bottom (and left+right), giving a huge phantom size.
        // Do this unconditionally so older sessions that undocked without
        // the fix still get healed on any subsequent interaction.
        if (panel.style.bottom !== "auto") panel.style.bottom = "auto";
        if (panel.style.right  !== "auto") panel.style.right  = "auto";
        if (panel.dataset.c2cUndocked === "true") return;
        panel.dataset.c2cUndocked = "true";
        // Measure FIRST — before mutating any layout state — so we capture
        // the panel's current on-screen position (flex-centered inside the
        // modal overlay, or wherever it lives).
        const r = panel.getBoundingClientRect();
        if (overlay) {
            overlay.style.background     = "transparent";
            overlay.style.pointerEvents  = "none";
            overlay.style.alignItems     = "";
            overlay.style.justifyContent = "";
        }
        panel.style.position      = "fixed";
        panel.style.pointerEvents = "auto";
        panel.style.maxHeight     = "";
        panel.style.maxWidth      = "";
        panel.style.margin        = "0";
        // Clamp into viewport (respect ComfyUI's top toolbar gutter).
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const left = Math.max(4, Math.min(r.left, vw - 80));
        const top  = Math.max(TOOLBAR_GUTTER, Math.min(r.top, vh - 80));
        panel.style.left = `${left}px`;
        panel.style.top  = `${top}px`;
    };

    // Header injection (re-callable on every panel re-render). Looks up the
    // header element either by static ref or by selector (re-queried each
    // time so innerHTML wipes don't strip our chrome).
    const _injectHeader = () => {
        const hdr = header || (headerSelector ? panel.querySelector(headerSelector) : null);
        if (!hdr) return;
        // Tag the autodetected header so the minimized CSS knows which child
        // to keep visible (survives innerHTML re-renders via the MO below).
        hdr.dataset.c2cHeader = "true";
        const tEl = titleEl || (titleSelector ? panel.querySelector(titleSelector) : hdr);
        if (shortcut && tEl && !tEl.querySelector(".c2c-win-shortcut")) {
            const span = document.createElement("span");
            span.className = "c2c-win-shortcut";
            span.textContent = ` (${shortcut})`;
            tEl.appendChild(span);
        }
        hdr.style.cursor     = hdr.style.cursor     || "move";
        hdr.style.userSelect = hdr.style.userSelect || "none";
        let minBtn = hdr.querySelector("[data-c2c-min]");
        if (!minBtn) {
            minBtn = document.createElement("button");
            minBtn.type = "button";
            minBtn.dataset.c2cMin = "true";
            minBtn.title = "Minimize / restore";
            minBtn.textContent = "–";
            // Styling comes from the global .c2c-win [data-c2c-min] rule so
            // it visually matches the rest of the chrome (no bare 18-px font).
            const closeBtn =
                  hdr.querySelector('[data-act="close"]')
               || hdr.querySelector('[id$="-close"]')
               || Array.from(hdr.querySelectorAll("button")).find(b =>
                       /^\s*[×x]\s*$/i.test(b.textContent || ""));
            if (closeBtn) closeBtn.before(minBtn);
            else hdr.appendChild(minBtn);
            minBtn.onclick = (e) => {
                e.stopPropagation();
                _undock();
                toggleMinimize(panel, storageKey);
            };
        }
    };

    // Ensure 8 resize edges + corner grip exist as direct children of the
    // panel. If panel.innerHTML wiped them, re-create them. Each edge is
    // wired with its own _attachEdgeResize handler. Pre-undock listener
    // is added on every fresh edge so resize starts from a fixed-pos rect.
    const _ensureEdges = () => {
        if (panel.querySelectorAll(":scope > .c2c-win-edge").length === 8) return;
        // Wipe stragglers first
        panel.querySelectorAll(":scope > .c2c-win-edge").forEach(e => e.remove());
        const oldGrip = panel.querySelector(":scope > .c2c-win-resize");
        if (oldGrip) oldGrip.remove();
        const grip = document.createElement("div");
        grip.className = "c2c-win-resize";
        panel.appendChild(grip);
        const DIRS = ["n", "s", "e", "w", "nw", "ne", "sw", "se"];
        DIRS.forEach(dir => {
            const e = document.createElement("div");
            e.className = `c2c-win-edge ${dir}`;
            e.dataset.dir = dir.toUpperCase();
            panel.appendChild(e);
            _attachEdgeResize(e, panel, dir.toUpperCase(), { minW, minH, key: storageKey });
            e.addEventListener("mousedown", _undock, true);
        });
    };

    // On re-entrant calls (panel already chrome'd), only re-inject the
    // header bits + ensure edges (panel-level drag/storage are already wired).
    if (_isInit) { _injectHeader(); _ensureEdges(); return; }
    panel.dataset.c2cChrome = "true";
    panel.dataset.c2cPanel  = "true";
    panel.classList.add("c2c-win");

    _injectHeader();

    // 4) Header-mousedown → undock at CAPTURE phase so makeDraggable's
    //    bubble-phase handler sees an already-fixed element with correct
    //    rect coords. Bail on inner controls so they keep working. Use
    //    delegated listener on the panel so dynamic header re-renders work.
    if (header || headerSelector) {
        panel.addEventListener("mousedown", (e) => {
            if (e.target.closest("button, input, select, textarea, a, label")) return;
            const inHdr = header
                ? header.contains(e.target)
                : !!e.target.closest(headerSelector);
            if (!inHdr) return;
            _undock();
        }, true);
    }

    // 5) Restore persisted geometry / minimized state. If anything was
    //    saved, undock immediately so the saved coords actually apply.
    if (storageKey) {
        const savedPos  = _load(`c2c.win.${storageKey}`);
        const savedSize = _load(`c2c.win.${storageKey}:size`);
        const savedMin  = _load(`c2c.win.${storageKey}:min`) === true;
        if (savedPos || savedSize || savedMin) {
            _undock();
            if (savedSize?.w && savedSize?.h) {
                panel.style.width  = `${savedSize.w}px`;
                panel.style.height = `${savedSize.h}px`;
            }
            if (savedPos && Number.isFinite(savedPos.left) && Number.isFinite(savedPos.top)) {
                const vw = window.innerWidth, vh = window.innerHeight;
                panel.style.left = `${Math.max(4,             Math.min(savedPos.left, vw - 80))}px`;
                panel.style.top  = `${Math.max(TOOLBAR_GUTTER, Math.min(savedPos.top,  vh - 80))}px`;
            }
            if (savedMin) panel.dataset.minimized = "true";
        }
    }

    // 6) Wire drag (resize edges are created/maintained by _ensureEdges).
    makeDraggable(panel, {
        handle: header || undefined,
        handleSelector: headerSelector || undefined,
        key: storageKey,
    });
    _ensureEdges();
    // 7) Z-order / focus: mousedown anywhere inside the panel raises it
    //    above all other C2C windows (classic OS behavior). Call once
    //    here so a freshly retrofit panel starts at the top of the stack.
    attachFocusOnInteract(panel);
    bringToFront(panel);

    // 8) Re-entrant header + edge upkeep — if `headerSelector` is used,
    //    watch the panel for childList changes so re-rendered headers AND
    //    wiped resize edges get re-injected automatically.
    if (headerSelector) {
        try {
            const mo = new MutationObserver(() => { _injectHeader(); _ensureEdges(); });
            mo.observe(panel, { childList: true, subtree: false });
        } catch (__c2cErr) { __c2cReport("_c2c_window", __c2cErr); }
    }
}
