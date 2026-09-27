/**
 * c2c_ui/nodes2.js — ComfyUI Nodes 2.0 compatibility helpers.
 *
 * Adapted from ComfyUI-Pixaroma (MIT, https://github.com/pixaroma/ComfyUI-Pixaroma)
 * Copyright (c) 2026 pixaroma. MIT License: the full notice ships beside this
 * file as LICENSE-PIXAROMA.txt.
 * Sources: third_party/ComfyUI-Pixaroma/js/shared/nodes2.mjs,
 *          renderer_switch.mjs, resize_floor.mjs
 *
 * Visual components are original C2C design; only renderer-compat logic is ported.
 *
 * License: Apache-2.0
 */

const CANVAS_BACKING_CAP = 6000;
const RENDERER_POLL_MS = 300;

/** @returns {object|null} ComfyUI app instance, read at call time. */
function getApp() {
    return window.comfyAPI?.app?.app ?? window.app ?? null;
}

/**
 * True when ComfyUI's Nodes 2.0 (Vue) renderer is active.
 * Read live — do not cache.
 */
export function isVueNodes() {
    return !!window.LiteGraph?.vueNodesMode;
}

/**
 * Adaptive canvasOnly house rule for internal DOM widgets.
 * @param {object} widget
 * @returns {object}
 */
export function adaptiveCanvasOnly(widget) {
    if (!widget || !widget.options) return widget;
    try {
        Object.defineProperty(widget.options, "canvasOnly", {
            configurable: true,
            enumerable: true,
            get() {
                return !window.LiteGraph?.vueNodesMode;
            },
        });
    } catch (_e) {
        widget.options.canvasOnly = !window.LiteGraph?.vueNodesMode;
    }
    return widget;
}

const _rendererListeners = new Set();
let _rendererTimer = null;
let _rendererLast = null;

function _currentRendererMode() {
    return !!window.LiteGraph?.vueNodesMode;
}

function _rendererTick() {
    const now = _currentRendererMode();
    if (now === _rendererLast) return;
    _rendererLast = now;
    for (const cb of [..._rendererListeners]) {
        try {
            cb(now);
        } catch (err) {
            console.warn("[c2c_ui] renderer-change handler failed", err);
        }
    }
}

/**
 * Register a callback fired when the renderer flips. Returns unsubscribe.
 * @param {(vue: boolean) => void} cb
 * @returns {() => void}
 */
export function onRendererChange(cb) {
    if (typeof cb !== "function") return () => {};
    _rendererListeners.add(cb);
    if (_rendererTimer == null) {
        _rendererLast = _currentRendererMode();
        _rendererTimer = setInterval(_rendererTick, RENDERER_POLL_MS);
    }
    return () => {
        _rendererListeners.delete(cb);
        if (!_rendererListeners.size && _rendererTimer != null) {
            clearInterval(_rendererTimer);
            _rendererTimer = null;
            _rendererLast = null;
        }
    };
}

/**
 * Sum visible children heights + gaps + vertical padding.
 * @param {HTMLElement} root
 * @returns {number}
 */
export function measureRootContent(root) {
    if (!root) return 0;
    let h = 0;
    let count = 0;
    for (const child of root.children) {
        if (child.offsetParent === null) continue;
        h += child.offsetHeight;
        count += 1;
    }
    const cs = getComputedStyle(root);
    const gap = parseFloat(cs.rowGap || cs.gap) || 0;
    if (count > 1) h += gap * (count - 1);
    h += (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0);
    return h;
}

/**
 * Drag-time resize floor for Nodes 2.0 DOM widgets.
 * @param {HTMLElement} root
 * @param {(root: HTMLElement) => number} measureFn
 * @param {() => void} [onRelease]
 * @returns {() => void}
 */
export function installResizeFloor(root, measureFn, onRelease) {
    if (!root || typeof measureFn !== "function") return () => {};
    let armed = false;

    const clear = () => {
        if (!armed) return;
        armed = false;
        try { root.style.minHeight = ""; } catch (_e) { /* ignore */ }
        if (typeof onRelease === "function") {
            try { onRelease(); } catch (_e) { /* ignore */ }
        }
    };

    const onDown = (e) => {
        if (!isVueNodes() || !root.isConnected) return;
        if (e.target?.closest?.(".lg-node-widget")) return;
        let cur = "";
        try {
            cur = (e.target && window.getComputedStyle(e.target).cursor) || "";
        } catch (_e) { /* ignore */ }
        if (cur.indexOf("resize") === -1) return;
        const myNode = root.closest(".lg-node");
        const downNode = e.target.closest && e.target.closest(".lg-node");
        if (myNode && downNode && myNode !== downNode) return;
        let h = 0;
        try { h = measureFn(root); } catch (_e) { return; }
        if (!(h > 0)) return;
        try {
            root.style.minHeight = Math.round(h) + "px";
            armed = true;
        } catch (_e) { /* ignore */ }
    };

    window.addEventListener("pointerdown", onDown, true);
    window.addEventListener("pointerup", clear, true);
    window.addEventListener("pointercancel", clear, true);

    return () => {
        window.removeEventListener("pointerdown", onDown, true);
        window.removeEventListener("pointerup", clear, true);
        window.removeEventListener("pointercancel", clear, true);
        clear();
    };
}

/**
 * Effective backing-store scale for a DOM canvas in Nodes 2.0.
 * @param {number} cssW
 * @param {number} cssH
 * @returns {number}
 */
export function canvasBackingScale(cssW, cssH) {
    const dpr = window.devicePixelRatio || 1;
    const zoom = Math.max(1, getApp()?.canvas?.ds?.scale || 1);
    let s = dpr * zoom;
    const longCss = Math.max(cssW || 0, cssH || 0);
    if (longCss > 0 && longCss * s > CANVAS_BACKING_CAP) {
        s = CANVAS_BACKING_CAP / longCss;
    }
    return Math.max(dpr, s);
}

/**
 * Repaint when graph zoom changes (not every frame).
 * @param {object} node
 * @param {() => void} render
 * @param {string} rafKey
 * @returns {() => void}
 */
export function installZoomRepaint(node, render, rafKey) {
    let lastZoom = -1;
    const tick = () => {
        const zoom = Math.max(1, getApp()?.canvas?.ds?.scale || 1);
        if (Math.abs(zoom - lastZoom) > 0.005) {
            lastZoom = zoom;
            try { render(); } catch (_e) { /* ignore */ }
        }
        node[rafKey] = requestAnimationFrame(tick);
    };
    node[rafKey] = requestAnimationFrame(tick);
    return () => {
        try { cancelAnimationFrame(node[rafKey]); } catch (_e) { /* ignore */ }
        node[rafKey] = null;
    };
}
