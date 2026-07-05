/* ────────────────────────────────────────────────────────────────────
 * mec_dock_anchor.js — viewport bottom occlusion tracker
 *
 * Problem
 * -------
 * Several MEC/C2C HUDs (`mec_system_hud`, error-translator
 * prompt) are pinned to the bottom-right of the viewport with a small
 * fixed `bottom` offset (e.g. 64 px) to clear ComfyUI's native status bar.
 * When the user opens a docked panel at the bottom (queue tab, image-feed,
 * Vue side-bar tab pinned south, or a manually-resized node "feed"),
 * that panel extends UPWARD from the viewport bottom and our HUDs end up
 * either covered or visually crowding it. The user explicitly reported:
 *
 *   "If resize feed opens, the bottom right UI is not moving up. Mouse
 *    pointer have to moved 2-3 times onto node to see information."
 *
 * Solution
 * --------
 * Continuously measure the largest "bottom-fixed" occupier in the
 * right-half of the viewport and expose it as the CSS custom property
 * `--mec-bottom-occupied-px` on `:root`. Anchored HUDs read this and add
 * it to their baseline `bottom` offset.
 *
 * Detection strategy
 *   1. ResizeObserver on `document.body` (catches splitter resizes).
 *   2. MutationObserver on `<body>` (catches panel show/hide).
 *   3. 750 ms fallback interval (catches CSS-transitions that finish
 *      after the mutation event).
 *
 * Occluder candidates (queried each tick — cheap because the document
 * isn't huge and we early-exit on bounding box):
 *   - `.comfyui-body-bottom` / `.comfyui-bottom-panel`           (Vue UI)
 *   - `.p-splitter-panel:last-child:not(:empty)`                 (Vue UI)
 *   - `[data-testid="status-bar"]`                               (Vue UI)
 *   - Any `position:fixed` element whose bottom touches viewport
 *     bottom and whose right edge is past the viewport mid-line.
 *
 * Our own HUDs are explicitly skipped (they carry a `data-mec-dock`
 * attribute) to avoid feedback loops.
 *
 * Public API
 *   window.__mecDock.register(el, {baseBottom})  →  manages bottom px
 *   window.__mecDock.unregister(el)
 *   window.__mecDock.recompute()
 *
 * Licence: Apache-2.0
 * ──────────────────────────────────────────────────────────────────── */

import { app } from "../../scripts/app.js";
import { LITE } from "./_c2c_lite.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const NATIVE_STATUS_BAR_PX = 52;   // ComfyUI Vue status bar default
const RIGHT_HALF_THRESHOLD = 0.5;  // occluder must extend past this fraction
const POLL_MS              = 750;

/** Registered anchors → {el, baseBottom}. */
const _anchors = new Map();
let _lastOccupied = -1;

// 2026-05-20 PERF FIX: the old implementation did
// `document.querySelectorAll("body *")` plus a `getComputedStyle()` on
// every DOM element each tick AND on every mutation. On a real workflow
// (1000+ DOM nodes, attribute mutations on every drag/widget update)
// that produced thousands of style recalcs per minute and was the #1
// reason ComfyUI felt sluggish after installing ComfyUI-CustomNodePacks.
//
// The narrow selector below catches the documented occluder cases
// (Vue UI bottom panels, splitter, status bar) and known third-party
// pop-ups by attribute, with NO full-DOM walk and NO getComputedStyle.
const OCCLUDER_SELECTOR = [
    ".comfyui-body-bottom",
    ".comfyui-bottom-panel",
    ".p-splitter-panel.bottom-panel",
    "[data-testid='status-bar']",
    ".comfy-image-feed",                  // built-in image feed
    ".comfyui-queue-sidebar",
    "[data-mec-occluder='1']",            // explicit opt-in
].join(",");

function _measureOccupiedPx() {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const rightCut = vw * RIGHT_HALF_THRESHOLD;
    let occupied = 0;

    const candidates = document.querySelectorAll(OCCLUDER_SELECTOR);
    for (const el of candidates) {
        if (el.hasAttribute("data-mec-dock")) continue;
        if (el.hasAttribute("data-c2c-overlay")) continue; // our own top/bottom strips
        const r = el.getBoundingClientRect();
        if (r.width < 100 || r.height < 16) continue;
        if (r.bottom < vh - 4) continue;           // not anchored to bottom
        if (r.right < rightCut) continue;          // not over our region
        if (r.height > vh * 0.6) continue;         // modal-sized; skip
        if (r.height > occupied) occupied = r.height;
    }
    return occupied;
}

function _applyAll(px) {
    document.documentElement.style.setProperty(
        "--mec-bottom-occupied-px", `${px}px`);
    for (const [el, info] of _anchors) {
        if (!el.isConnected) continue;
        // Left-side anchors are NOT pushed up by right-side occluders;
        // they have their own measurement at register time. We still
        // honour the user-set baseBottom always.
        const push = info.side === "left" ? 0 : px;
        el.style.bottom = `${info.baseBottom + push}px`;
    }
}

function recompute() {
    const px = _measureOccupiedPx();
    if (px === _lastOccupied) return;
    _lastOccupied = px;
    _applyAll(px);
}

// 2026-05-20 PERF FIX: coalesce MutationObserver bursts. Without this,
// every node drag / widget input fires `subtree:true,attributes:true`
// callbacks dozens of times per second. Now we collapse all of them
// into one rAF-scheduled recompute.
let _scheduled = false;
function _scheduleRecompute() {
    if (_scheduled) return;
    _scheduled = true;
    requestAnimationFrame(() => {
        _scheduled = false;
        try { recompute(); } catch { /* */ }
    });
}

function register(el, opts = {}) {
    if (!el) return;
    const baseBottom = typeof opts.baseBottom === "number" ? opts.baseBottom : 64;
    const side = opts.side === "left" ? "left" : "right";
    el.setAttribute("data-mec-dock", "1");
    _anchors.set(el, { baseBottom, side });
    const occupiedPush = side === "left" ? 0 : (_lastOccupied > 0 ? _lastOccupied : 0);
    el.style.bottom = `${baseBottom + occupiedPush}px`;
    if (_lastOccupied < 0) recompute();
}

function unregister(el) {
    if (!el) return;
    _anchors.delete(el);
    el.removeAttribute("data-mec-dock");
}

let _booted = false;
function _boot() {
    if (_booted) return;
    _booted = true;

    // Adopt HUDs that were created before this extension booted
    // (file load order is alphabetical).
    const ADOPT = [
        { id: "mec-system-hud",   baseBottom: 64 },
    ];
    for (const a of ADOPT) {
        const el = document.getElementById(a.id);
        if (el && !_anchors.has(el)) register(el, { baseBottom: a.baseBottom });
    }
    // Watch for late-creation of those same HUDs.
    try {
        const adoptMo = new MutationObserver(() => {
            for (const a of ADOPT) {
                const el = document.getElementById(a.id);
                if (el && !_anchors.has(el)) register(el, { baseBottom: a.baseBottom });
            }
        });
        adoptMo.observe(document.body, { childList: true, subtree: false });
    } catch (__c2cErr) { __c2cReport("c2c_dock_anchor", __c2cErr); }

    try {
        const ro = new ResizeObserver(() => _scheduleRecompute());
        ro.observe(document.body);
    } catch (__c2cErr) { __c2cReport("c2c_dock_anchor", __c2cErr); }

    try {
        const mo = new MutationObserver(() => _scheduleRecompute());
        // PERF FIX 2026-05-20: removed `subtree:true` + `attributes:true`.
        // The old config fired on every node-drag style mutation across
        // the entire DOM (~50-100 events per second when interacting).
        // We only need to know when bottom panels appear/disappear, which
        // is a `childList` event near the document root.
        mo.observe(document.body, { childList: true, subtree: false });
    } catch (__c2cErr) { __c2cReport("c2c_dock_anchor", __c2cErr); }

    window.addEventListener("resize", _scheduleRecompute, { passive: true });
    setInterval(_scheduleRecompute, POLL_MS);
    recompute();
}

window.__mecDock = { register, unregister, recompute };

if (!LITE) app.registerExtension({
    name: "C2C.DockAnchor",
    async setup() {
        _boot();
        console.log("[MEC.DockAnchor] Ready — bottom-right HUD reposition active.");
    },
});
