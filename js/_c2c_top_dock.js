/* ────────────────────────────────────────────────────────────────────
 * _c2c_top_dock.js — shared top-row dock for C2C/MEC overlay buttons.
 *
 * Problem (user-reported 2026-05-21):
 *   The Auto-Checkpoint button, Workflow Doctor button and Bookmarks
 *   strip each pinned themselves with hardcoded `position:fixed; top:68px`
 *   in their own corner. When the user toggled the ComfyUI sidebar or the
 *   "Workflow tabs" row reflowed, our buttons stayed put and collided
 *   with native ComfyUI chrome (queue button, tab handles). They also
 *   blocked mouse clicks meant for the canvas/tabs underneath.
 *
 * This module provides a SINGLE managed flex strip that:
 *   • sits AT the very top of the viewport in a `pointer-events:none`
 *     wrapper so clicks pass through to anything not on a button,
 *   • measures the bottom edge of ComfyUI's top menu / Workflow-tabs row
 *     each tick and shifts itself down by that amount (no more overlap
 *     with native chrome),
 *   • re-measures on body resize and DOM mutation (sidebar toggle),
 *   • exposes `register(el, {side: "left"|"right", order: N})` so each
 *     overlay just hands over its DOM element.
 *
 * Z-index is capped at 2500 (below PrimeVue's modal layer at 3000) so
 * ComfyUI dialogs always win.
 *
 * Public API
 *   window.__c2cTopDock.register(el, {side, order})
 *   window.__c2cTopDock.unregister(el)
 *   window.__c2cTopDock.recompute()
 *
 * Licence: Apache-2.0
 * ──────────────────────────────────────────────────────────────────── */

import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const WRAP_ID_LEFT  = "c2c-top-dock-left";
const WRAP_ID_RIGHT = "c2c-top-dock-right";
const STYLE_ID      = "c2c-top-dock-style";
const FALLBACK_TOP  = 44;   // px below the very top if we can't measure
const MIN_GAP       = 6;    // px breathing room below native chrome
const SIDE_GAP      = 8;    // px breathing room beside left/right sidebars
const MIN_EDGE      = 12;   // never get closer than this to viewport edge
// Z-index is supplied via the locked CSS scale (`--c2c-z-dock`) defined in
// _c2c_theme.js — 2500, between hud(1000) and popover(9000). Kept as a
// fallback literal in case the stylesheet hasn't loaded yet on first paint.
const Z_INDEX       = 2500;

// Vue sidebar selectors — when these are open they push our strips inward.
// Modern ComfyUI (frontend 1.43+) renders the sidebar as
// `<div role="complementary" aria-label="Sidebar" class="side-bar-panel ...">`
// inside a `.p-splitter` flex layout. There is no `-left`/`-right` suffix on
// the class; the side is determined by the user's "Sidebar location" setting
// and shows up as the panel's position in the splitter. We therefore include
// the generic `.side-bar-panel` selector in BOTH lists and let the
// measurement functions decide based on the panel's actual viewport coords
// (`r.left > 8` rules out right-anchored panels for the left-sidebar measure,
// and `r.right < vw - 8` rules out left-anchored panels for the right-sidebar
// measure).
const LEFT_SIDEBAR_SEL = [
    ".side-tool-bar-container",
    ".side-bar-panel",
    ".side-bar-panel.side-bar-panel-left",   // legacy
    "[role='complementary'][aria-label='Sidebar']",
    "[data-testid='left-sidebar']",
    ".comfyui-body-left",
].join(",");
const RIGHT_SIDEBAR_SEL = [
    ".side-bar-panel",
    ".side-bar-panel.side-bar-panel-right",  // legacy
    "[role='complementary'][aria-label='Sidebar']",
    "[data-testid='right-sidebar']",
    ".comfyui-body-right",
].join(",");

const _members = new Map(); // el -> {side, order}

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
#${WRAP_ID_LEFT}, #${WRAP_ID_RIGHT} {
    position: fixed;
    z-index: var(--c2c-z-dock, ${Z_INDEX});
    display: flex;
    gap: 6px;
    pointer-events: none;          /* clicks fall through gaps to ComfyUI */
    align-items: center;
    flex-wrap: wrap;               /* extra items spill to a new row */
    max-width: 48vw;               /* never blanket the entire top */
}
#${WRAP_ID_LEFT}  { left:  12px; justify-content: flex-start; }
#${WRAP_ID_RIGHT} { right: 14px; justify-content: flex-end; }
#${WRAP_ID_LEFT}:empty,
#${WRAP_ID_RIGHT}:empty {
    /* Once OmniBar absorbs all our dock members the wrappers go empty;
     * hide them so they don't reserve a phantom top-row band above
     * OmniBar that would push the OmniBar measurer downward. */
    display: none;
}
#${WRAP_ID_LEFT}  > *,
#${WRAP_ID_RIGHT} > * {
    pointer-events: auto;          /* but each child IS clickable */
}
`;
    document.head.appendChild(s);
}

function _wrap(side) {
    const id = side === "left" ? WRAP_ID_LEFT : WRAP_ID_RIGHT;
    let el = document.getElementById(id);
    if (!el) {
        el = document.createElement("div");
        el.id = id;
        el.setAttribute("data-c2c-overlay", "1"); // ignored by our own measurers
        document.body.appendChild(el);
    }
    return el;
}

/* Measure the lowest "top chrome" element so we sit just below it. */
function _measureTopChromeBottom() {
    // Explicit native ComfyUI top-chrome selectors. Width threshold is kept
    // generous because the workflow-tabs tablist can be narrow (only ~200px
    // when a single tab is open), but its CONTAINER spans the full width.
    const SELECTORS = [
        ".comfyui-body-top",
        ".comfy-menu",
        ".comfyui-menu",
        ".comfyui-workflow-tabs",
        ".workflow-tabs-container",   // full-width wrapper (modern ComfyUI)
        ".workflow-tabs",              // inner tablist (may be narrow)
        ".actionbar-container",        // floating Run + ComfyUI-Manager row
        "[data-testid='top-menu']",
        ".p-menubar",
    ];
    let bottom = 0;
    for (const sel of SELECTORS) {
        const nodes = document.querySelectorAll(sel);
        for (const n of nodes) {
            if (n.hasAttribute("data-c2c-overlay")) continue; // ignore our own
            const r = n.getBoundingClientRect();
            // Accept any element anchored near the top of the viewport
            // (top <= 60 covers the case where workflow tabs sit BELOW
            // a thin 16px chrome strip). No width gate — the tablist
            // itself can be narrow but its bottom edge still matters.
            if (r.top <= 60 && r.height > 4 && r.bottom > bottom && r.bottom < 200) {
                bottom = r.bottom;
            }
        }
    }
    return bottom || FALLBACK_TOP;
}

/* Largest right-edge of any open LEFT sidebar (0 if none). */
function _measureLeftSidebarRight() {
    const vw = window.innerWidth;
    let right = 0;
    for (const n of document.querySelectorAll(LEFT_SIDEBAR_SEL)) {
        if (n.hasAttribute("data-c2c-overlay")) continue;
        const r = n.getBoundingClientRect();
        if (r.width < 20 || r.height < 100) continue;     // collapsed
        // Modern ComfyUI's sidebar panel sits at x≈72 (right of the icon
        // rail). Accept anything in the LEFT half whose right edge is
        // closer to the left side than the right side of the viewport.
        if (r.left > vw * 0.45) continue;                 // not a LEFT-anchored panel
        if (r.right > vw * 0.55) continue;                // spans too far right (modal)
        if (r.right > right) right = r.right;
    }
    return right;
}

/* Smallest left-edge of any open RIGHT sidebar (window.innerWidth if none). */
function _measureRightSidebarLeft() {
    const vw = window.innerWidth;
    let left = vw;
    for (const n of document.querySelectorAll(RIGHT_SIDEBAR_SEL)) {
        if (n.hasAttribute("data-c2c-overlay")) continue;
        const r = n.getBoundingClientRect();
        if (r.width < 20 || r.height < 100) continue;
        // Right-anchored if its left edge is past the viewport midline
        // (and the panel ends near/at the right edge).
        if (r.left < vw * 0.55) continue;
        if (r.right < vw * 0.65) continue;                // too narrow / not right-anchored
        if (r.left < left) left = r.left;
    }
    return left;
}

let _lastTop = -1;
let _lastLeft = -1;
let _lastRight = -1;
function _applyTop() {
    const t   = _measureTopChromeBottom() + MIN_GAP;
    const lsb = _measureLeftSidebarRight();
    const rsb = _measureRightSidebarLeft();
    const vw  = window.innerWidth;

    const leftPx  = Math.max(MIN_EDGE, lsb ? lsb + SIDE_GAP : 12);
    const rightPx = Math.max(MIN_EDGE, rsb < vw ? (vw - rsb) + SIDE_GAP : 14);

    if (t === _lastTop && leftPx === _lastLeft && rightPx === _lastRight) return;
    _lastTop = t; _lastLeft = leftPx; _lastRight = rightPx;

    const left  = document.getElementById(WRAP_ID_LEFT);
    const right = document.getElementById(WRAP_ID_RIGHT);
    if (left)  { left.style.top  = `${t}px`; left.style.left  = `${leftPx}px`; }
    if (right) { right.style.top = `${t}px`; right.style.right = `${rightPx}px`; }
}

function _sortChildren(wrap) {
    const arr = Array.from(wrap.children);
    arr.sort((a, b) => (Number(a.dataset.c2cOrder) || 0) - (Number(b.dataset.c2cOrder) || 0));
    for (const c of arr) wrap.appendChild(c);
}

function register(el, opts = {}) {
    if (!el) return;
    _injectStyle();
    const side  = opts.side === "left" ? "left" : "right";
    const order = typeof opts.order === "number" ? opts.order : 100;

    // ── OmniBar consolidation ────────────────────────────────────────
    // Per locked spec ("all of them should be there in omnibar only.
    // no need of duplication."), if the OmniBar is mounted we hand the
    // element off to its slot system instead of building a parallel
    // top-row strip. The dock wrappers are kept empty (hidden via the
    // is-empty rule below) so legacy callers stay functional even if
    // OmniBar is later disabled or torn down.
    if (window.C2COmniBar && typeof window.C2COmniBar.register === "function") {
        const section = side === "left" ? "tools" : "bookmarks";
        const slotId = (el.id || "c2c-dock-slot") + "-" + side;
        try {
            // Reset any positioning the caller set so the OmniBar host
            // controls layout entirely (matches the dock contract).
            el.style.position = "static";
            el.style.top = "";
            el.style.left = "";
            el.style.right = "";
            el.style.bottom = "";
            el.style.zIndex = "";
            el.dataset.c2cOrder = String(order);
            window.C2COmniBar.register({
                section,
                id: slotId,
                order,
                element: el,
            });
            _members.set(el, { side, order, proxied: true, slotId });
            return;
        } catch (__c2cErr) {
            __c2cReport("_c2c_top_dock:omnibar_proxy", __c2cErr);
            // fall through to legacy dock mount
        }
    }

    const wrap  = _wrap(side);

    // Strip any positioning the caller set; let flex handle it.
    el.style.position = "static";
    el.style.top    = "";
    el.style.left   = "";
    el.style.right  = "";
    el.style.bottom = "";
    el.style.zIndex = "";
    el.dataset.c2cOrder = String(order);
    el.dataset.c2cDock  = side;
    wrap.appendChild(el);
    _members.set(el, { side, order });
    _sortChildren(wrap);
    _applyTop();
}

function unregister(el) {
    if (!el) return;
    const meta = _members.get(el);
    _members.delete(el);
    // OmniBar-proxied members must be removed from the OmniBar slot list,
    // not from a dock wrapper they were never appended to.
    if (meta && meta.proxied && window.C2COmniBar?.register) {
        try {
            // OmniBar.register returns an unregister fn, but we didn't
            // capture it. Re-register with an empty placeholder element
            // so its detach loop runs, then immediately remove via the
            // hidden private slot list. Simpler: detach from DOM if it
            // still has a slot-id parent.
            const parent = el.parentElement;
            if (parent && parent.classList?.contains("c2c-omnibar-section")) {
                parent.removeChild(el);
            }
        } catch (__c2cErr) { __c2cReport("_c2c_top_dock:omnibar_unregister", __c2cErr); }
        return;
    }
    if (el.parentElement && (el.parentElement.id === WRAP_ID_LEFT
                          || el.parentElement.id === WRAP_ID_RIGHT)) {
        el.parentElement.removeChild(el);
    }
}

let _scheduled = false;
function _schedule() {
    if (_scheduled) return;
    _scheduled = true;
    requestAnimationFrame(() => { _scheduled = false; try { _applyTop(); _migrateToOmniBarIfReady(); } catch (__c2cErr) { __c2cReport("_c2c_top_dock", __c2cErr); } });
}

/* Late-migration: any element that was register()'ed before OmniBar mounted
 * (e.g. early bootstrap order) lives in the legacy dock wrappers. As soon
 * as OmniBar appears we move it into the appropriate slot so the user
 * never sees a transient parallel strip. */
function _migrateToOmniBarIfReady() {
    if (!window.C2COmniBar || typeof window.C2COmniBar.register !== "function") return;
    let migrated = false;
    for (const [el, meta] of _members) {
        if (meta.proxied) continue;
        const section = meta.side === "left" ? "tools" : "bookmarks";
        const slotId = (el.id || "c2c-dock-slot") + "-" + meta.side;
        try {
            window.C2COmniBar.register({
                section,
                id: slotId,
                order: meta.order,
                element: el,
            });
            meta.proxied = true;
            meta.slotId = slotId;
            migrated = true;
        } catch (__c2cErr) {
            __c2cReport("_c2c_top_dock:migrate", __c2cErr);
        }
    }
    if (migrated) {
        // OmniBar may need to re-measure now that its sections grew.
        try { window.C2COmniBar.refresh?.(); }
        catch (__c2cErr) { __c2cReport("_c2c_top_dock:omnibar_refresh", __c2cErr); }
    }
}

let _booted = false;
function _boot() {
    if (_booted) return;
    _booted = true;
    _injectStyle();
    _wrap("left");
    _wrap("right");
    _applyTop();
    try {
        const ro = new ResizeObserver(_schedule);
        ro.observe(document.body);
    } catch (__c2cErr) { __c2cReport("_c2c_top_dock", __c2cErr); }
    try {
        const mo = new MutationObserver(_schedule);
        mo.observe(document.body, { childList: true, subtree: false });
    } catch (__c2cErr) { __c2cReport("_c2c_top_dock", __c2cErr); }
    window.addEventListener("resize", _schedule, { passive: true });
    setInterval(_schedule, 1000);
}

window.__c2cTopDock = { register, unregister, recompute: _applyTop };

app.registerExtension({
    name: "C2C.TopDock",
    async setup() {
        _boot();
        console.log("[C2C.TopDock] Ready — top overlay buttons share a managed strip.");
    },
});
