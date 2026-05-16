/* ────────────────────────────────────────────────────────────────────
 * mec_dock_anchor.js — viewport bottom occlusion tracker
 *
 * Problem
 * -------
 * Several MEC/C2C HUDs (`mec_system_hud`, `c2c_minimap`, error-translator
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

const NATIVE_STATUS_BAR_PX = 52;   // ComfyUI Vue status bar default
const RIGHT_HALF_THRESHOLD = 0.5;  // occluder must extend past this fraction
const POLL_MS              = 750;

/** Registered anchors → {el, baseBottom}. */
const _anchors = new Map();
let _lastOccupied = -1;

function _measureOccupiedPx() {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const rightCut = vw * RIGHT_HALF_THRESHOLD;
    let occupied = 0;

    // Known Vue UI selectors first (fast, specific)
    const SEL = [
        ".comfyui-body-bottom",
        ".comfyui-bottom-panel",
        ".p-splitter-panel.bottom-panel",
        "[data-testid='status-bar']",
    ];
    for (const sel of SEL) {
        document.querySelectorAll(sel).forEach((el) => {
            if (el.hasAttribute("data-mec-dock")) return;
            const r = el.getBoundingClientRect();
            if (r.width < 100 || r.height < 16) return;
            if (r.bottom < vh - 4) return;          // not anchored to bottom
            if (r.right < rightCut) return;         // not over our region
            occupied = Math.max(occupied, r.height);
        });
    }

    // Generic sweep — only `position:fixed` to keep cost low.
    // This catches node-internal "image feed" pop-ups too.
    document.querySelectorAll("body *").forEach((el) => {
        if (el.hasAttribute("data-mec-dock")) return;
        if (el.id === "vue-app" || el.tagName === "CANVAS") return;
        const cs = el.ownerDocument.defaultView.getComputedStyle(el);
        if (cs.position !== "fixed") return;
        if (cs.visibility === "hidden" || cs.display === "none") return;
        const r = el.getBoundingClientRect();
        if (r.width < 100 || r.height < 16) return;
        if (r.bottom < vh - 4) return;
        if (r.right < rightCut) return;
        // Ignore overlays larger than half-viewport (modals, etc.)
        if (r.height > vh * 0.6) return;
        occupied = Math.max(occupied, r.height);
    });

    return occupied;
}

function _applyAll(px) {
    document.documentElement.style.setProperty(
        "--mec-bottom-occupied-px", `${px}px`);
    for (const [el, info] of _anchors) {
        if (!el.isConnected) continue;
        el.style.bottom = `${info.baseBottom + px}px`;
    }
}

function recompute() {
    const px = _measureOccupiedPx();
    if (px === _lastOccupied) return;
    _lastOccupied = px;
    _applyAll(px);
}

function register(el, opts = {}) {
    if (!el) return;
    const baseBottom = typeof opts.baseBottom === "number" ? opts.baseBottom : 64;
    el.setAttribute("data-mec-dock", "1");
    _anchors.set(el, { baseBottom });
    el.style.bottom = `${baseBottom + (_lastOccupied > 0 ? _lastOccupied : 0)}px`;
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
    // (file load order is alphabetical; c2c_minimap loads first).
    const ADOPT = [
        { id: "mec-system-hud",   baseBottom: 64 },
        { id: "c2c-minimap-root", baseBottom: 88 },
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
    } catch (_) { /* ignore */ }

    try {
        const ro = new ResizeObserver(() => recompute());
        ro.observe(document.body);
    } catch (_) { /* old browser, fallback to interval */ }

    try {
        const mo = new MutationObserver(() => recompute());
        mo.observe(document.body, {
            childList: true,
            subtree:   true,
            attributes:true,
            attributeFilter: ["style", "class", "hidden"],
        });
    } catch (_) { /* ignore */ }

    window.addEventListener("resize", recompute, { passive: true });
    setInterval(recompute, POLL_MS);
    recompute();
}

window.__mecDock = { register, unregister, recompute };

app.registerExtension({
    name: "MEC.DockAnchor",
    async setup() {
        _boot();
        console.log("[MEC.DockAnchor] Ready — bottom-right HUD reposition active.");
    },
});
