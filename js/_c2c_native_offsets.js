/*
 * _c2c_native_offsets.js
 * ─────────────────────
 * Measures ComfyUI's native chrome (top body bar + workflow tabs, left tool
 * rail, bottom status strip) and publishes the measurements as CSS variables
 * on :root so OmniBar (P0.2) and any other C2C surface can position itself with
 *
 *     top:    calc(var(--c2c-native-top, 0px) + 8px)
 *     left:   calc(var(--c2c-native-left, 0px) + 8px)
 *     bottom: calc(var(--c2c-native-bottom, 0px) + 8px)
 *
 * Variables published on :root:
 *   --c2c-native-top    : pixels — bottom edge of native top chrome
 *                         (max of .comfyui-body-top, .workflow-tabs, .comfyui-menu, etc.)
 *   --c2c-native-left   : pixels — right edge of left rail (.side-tool-bar-container)
 *   --c2c-native-bottom : pixels — height of native bottom status strip
 *
 * Lifecycle:
 *   - ResizeObserver(document.body)            — DOM-level reflow / drawer toggles
 *   - MutationObserver(document.body, subtree) — sidebars docking/undocking, late mounts
 *   - window 'resize'                          — viewport changes
 *   - polling fallback @ 1 Hz                  — catches anything the observers miss
 *
 * Error policy (per user override 2026-05-25): NO empty catches. Every failure
 * is reported via `c2c:registry-failure` CustomEvent + fire-and-forget POST
 * /c2c/registry/failure (consumed by nodes/_c2c_registry.py — same pattern
 * used by _c2c_undo.js and _c2c_storage.js). The page is never crashed.
 *
 * Per MEGA_PLAN principle 12: purely additive. If selectors miss, the
 * variable falls back to 0 and consumers' additive padding still gives a
 * safe gap from the viewport edge.
 */

const COMPONENT = "c2c_native_offsets";

const TOP_SELECTORS = [
    ".comfyui-body-top",
    ".workflow-tabs",
    ".comfyui-menu",
    "#comfyui-menu",
    ".comfy-menu",
];

const LEFT_SELECTORS = [
    ".side-tool-bar-container",
];

const BOTTOM_SELECTORS = [
    ".workflow-status",
    ".comfy-status-strip",
    ".comfyui-status",
    ".comfyui-body-bottom",
];

/**
 * Centralised failure surfacer. Dispatches the same CustomEvent the rest of
 * the C2C codebase listens for, and best-effort POSTs to the registry so
 * Doctor / future telemetry tabs can show it. Never throws. Never silent.
 *
 * @param {string} where  short tag identifying the call site
 * @param {Error|string} err the underlying failure
 */
function _reportFailure(where, err) {
    const detail = {
        component: COMPONENT,
        where: String(where || ""),
        message: (err && err.message) ? err.message : String(err),
        stack: (err && err.stack) ? err.stack : null,
        ts: Date.now(),
    };
    // Console first — guaranteed visible during dev.
    try {
        // eslint-disable-next-line no-console
        console.error("[c2c_native_offsets]", where, err);
    } catch (consoleErr) {
        // If console itself throws, there is no surface left to log to —
        // fall through to the event + POST below which do not depend on it.
        void consoleErr;
    }
    // CustomEvent — listened to by _c2c_registry.js and the eventual Doctor tab.
    try {
        window.dispatchEvent(new CustomEvent("c2c:registry-failure", { detail }));
    } catch (dispatchErr) {
        // CustomEvent unsupported in the host? Only one remaining surface left.
        // eslint-disable-next-line no-console
        console.error("[c2c_native_offsets] dispatch failed", dispatchErr);
    }
    // Fire-and-forget POST. We do not await; we do not retry; we do not block.
    try {
        const body = JSON.stringify(detail);
        fetch("/c2c/registry/failure", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body,
            keepalive: true,
        }).catch((netErr) => {
            // eslint-disable-next-line no-console
            console.error("[c2c_native_offsets] POST /c2c/registry/failure failed", netErr);
        });
    } catch (fetchErr) {
        // fetch may not exist (server-side rendering, very old browser).
        // eslint-disable-next-line no-console
        console.error("[c2c_native_offsets] fetch unavailable", fetchErr);
    }
}

function _measureTop() {
    let bottom = 0;
    for (const sel of TOP_SELECTORS) {
        let el;
        try {
            el = document.querySelector(sel);
        } catch (qErr) {
            _reportFailure("querySelector:top:" + sel, qErr);
            continue;
        }
        if (!el) continue;
        let r;
        try {
            r = el.getBoundingClientRect();
        } catch (rectErr) {
            _reportFailure("getBoundingClientRect:top:" + sel, rectErr);
            continue;
        }
        if (r.height > 0 && r.width > 0 && r.bottom > bottom) bottom = r.bottom;
    }
    return Math.max(0, Math.round(bottom));
}

function _measureLeft() {
    let right = 0;
    for (const sel of LEFT_SELECTORS) {
        let el;
        try {
            el = document.querySelector(sel);
        } catch (qErr) {
            _reportFailure("querySelector:left:" + sel, qErr);
            continue;
        }
        if (!el) continue;
        let r;
        try {
            r = el.getBoundingClientRect();
        } catch (rectErr) {
            _reportFailure("getBoundingClientRect:left:" + sel, rectErr);
            continue;
        }
        if (r.height > 0 && r.width > 0 && r.right > right) right = r.right;
    }
    return Math.max(0, Math.round(right));
}

function _measureBottom() {
    // Height of strip(s) docked to viewport bottom, in pixels.
    let h = 0;
    const vh = (window.innerHeight || document.documentElement.clientHeight || 0);
    for (const sel of BOTTOM_SELECTORS) {
        let el;
        try {
            el = document.querySelector(sel);
        } catch (qErr) {
            _reportFailure("querySelector:bottom:" + sel, qErr);
            continue;
        }
        if (!el) continue;
        let r;
        try {
            r = el.getBoundingClientRect();
        } catch (rectErr) {
            _reportFailure("getBoundingClientRect:bottom:" + sel, rectErr);
            continue;
        }
        if (r.height <= 0 || r.width <= 0) continue;
        // Only count strips actually sitting near the viewport bottom.
        if (r.bottom < vh - 8) continue;
        const hh = Math.round(r.height);
        if (hh > h) h = hh;
    }
    return Math.max(0, h);
}

let _lastTop = -1;
let _lastLeft = -1;
let _lastBottom = -1;
let _resizeObserver = null;
let _mutationObserver = null;
let _pollTimer = 0;
let _started = false;

function _publish() {
    const t = _measureTop();
    const l = _measureLeft();
    const b = _measureBottom();
    if (t === _lastTop && l === _lastLeft && b === _lastBottom) return;
    _lastTop = t; _lastLeft = l; _lastBottom = b;
    const r = document.documentElement;
    try {
        r.style.setProperty("--c2c-native-top", t + "px");
    } catch (setErr) {
        _reportFailure("setProperty:--c2c-native-top", setErr);
    }
    try {
        r.style.setProperty("--c2c-native-left", l + "px");
    } catch (setErr) {
        _reportFailure("setProperty:--c2c-native-left", setErr);
    }
    try {
        r.style.setProperty("--c2c-native-bottom", b + "px");
    } catch (setErr) {
        _reportFailure("setProperty:--c2c-native-bottom", setErr);
    }
}

function _onResize() { _publish(); }

// rAF-coalesced variant for observer callbacks that fire in bursts
// (e.g. MutationObserver on body+subtree:true during sidebar layout).
// Multiple mutations within one frame collapse into a single _publish.
let _rafPending = false;
function _onResizeCoalesced() {
    if (_rafPending) return;
    _rafPending = true;
    const raf = (typeof requestAnimationFrame !== "undefined")
        ? requestAnimationFrame
        : (cb) => setTimeout(cb, 16);
    raf(() => { _rafPending = false; _publish(); });
}

function _pollTick() {
    _publish();
    try {
        _pollTimer = window.setTimeout(_pollTick, 1000);
    } catch (timerErr) {
        _reportFailure("setTimeout:poll", timerErr);
        _pollTimer = 0;
    }
}

/** Force a remeasure now. Useful after toggling sidebars programmatically. */
export function refreshNativeOffsets() { _publish(); }

/** Read the current cached top/left/bottom (px). */
export function getNativeOffsets() {
    return {
        top: _lastTop < 0 ? 0 : _lastTop,
        left: _lastLeft < 0 ? 0 : _lastLeft,
        bottom: _lastBottom < 0 ? 0 : _lastBottom,
    };
}

/** Install observers + start polling. Idempotent. */
export function startNativeOffsets() {
    if (_started) return;
    _started = true;
    // First publish so consumers can use the vars synchronously after import.
    _publish();
    if (typeof ResizeObserver !== "undefined") {
        try {
            _resizeObserver = new ResizeObserver(_onResizeCoalesced);
            _resizeObserver.observe(document.body);
        } catch (roErr) {
            _reportFailure("ResizeObserver.observe", roErr);
            _resizeObserver = null;
        }
    } else {
        _reportFailure("ResizeObserver", new Error("ResizeObserver unsupported"));
    }
    if (typeof MutationObserver !== "undefined") {
        try {
            _mutationObserver = new MutationObserver(_onResizeCoalesced);
            // PERF: subtree:false. This module is imported by _c2c_theme.js (i.e.
            // by EVERY C2C extension), so this observer is always live. With
            // subtree:true it fired on every descendant mutation across ComfyUI's
            // whole Vue/LiteGraph DOM (thousands/sec) → constant allocation →
            // young-gen GC pressure (a real OOM contributor). We only need to catch
            // native chrome mounting at the body level; the ResizeObserver on body +
            // the window resize listener + the 1s polling backstop cover the rest.
            _mutationObserver.observe(document.body, { childList: true, subtree: false });
        } catch (moErr) {
            _reportFailure("MutationObserver.observe", moErr);
            _mutationObserver = null;
        }
    } else {
        _reportFailure("MutationObserver", new Error("MutationObserver unsupported"));
    }
    try {
        window.addEventListener("resize", _onResize, { passive: true });
    } catch (addErr) {
        _reportFailure("addEventListener:resize", addErr);
    }
    // Polling backstop. Catches late-mounting native UI (workflow-tabs sometimes
    // mounts ~1.5s after boot) and anything the observers miss.
    if (!_pollTimer) {
        try {
            _pollTimer = window.setTimeout(_pollTick, 250);
        } catch (timerErr) {
            _reportFailure("setTimeout:initial-poll", timerErr);
            _pollTimer = 0;
        }
    }
}

/** Stop all observers (used in tests, or if a host wants to shut us down). */
export function stopNativeOffsets() {
    if (!_started) return;
    _started = false;
    if (_resizeObserver) {
        try {
            _resizeObserver.disconnect();
        } catch (discErr) {
            _reportFailure("ResizeObserver.disconnect", discErr);
        }
        _resizeObserver = null;
    }
    if (_mutationObserver) {
        try {
            _mutationObserver.disconnect();
        } catch (discErr) {
            _reportFailure("MutationObserver.disconnect", discErr);
        }
        _mutationObserver = null;
    }
    try {
        window.removeEventListener("resize", _onResize);
    } catch (rmErr) {
        _reportFailure("removeEventListener:resize", rmErr);
    }
    if (_pollTimer) {
        try {
            window.clearTimeout(_pollTimer);
        } catch (clearErr) {
            _reportFailure("clearTimeout:poll", clearErr);
        }
        _pollTimer = 0;
    }
}

// Auto-start on import. Defer to a microtask so callers can re-import without
// duplicate observers (idempotent), and so the DOM is ready in most cases.
if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
        try {
            document.addEventListener("DOMContentLoaded", startNativeOffsets, { once: true });
        } catch (listErr) {
            _reportFailure("addEventListener:DOMContentLoaded", listErr);
        }
    } else {
        // Defer one tick so :root style sets land after current script work.
        try {
            Promise.resolve().then(startNativeOffsets);
        } catch (promErr) {
            _reportFailure("Promise.resolve.then", promErr);
        }
    }
}
