// ============================================================================
// C2C — P10.5 Ctrl+F Search Restore
// ----------------------------------------------------------------------------
// Why this exists:
//   On a clean install of ComfyUI 1.43.x the LiteGraph node search box
//   (Ctrl+F) is technically functional — `app.canvas.showSearchBox` exists,
//   creates the dialog element, sizes it (482x265), but **leaves it hidden**.
//   Diagnostics on the live install showed:
//       - the .litegraph.litesearchbox element IS appended to <body>
//       - getComputedStyle().display === "none"  (ancestor or rule)
//       - el.offsetParent === null
//   so the user pressed Ctrl+F and nothing visible happened.
//
// What this does:
//   1. Registers a C2C command "c2c.search.openNodeSearch" that opens the
//      LiteGraph search box and FORCES visibility (display, opacity, z-index,
//      position) regardless of what CSS or competing extensions did.
//   2. Binds Ctrl+F (and Cmd+F on Mac) to that command via the standard Vue
//      keybinding system so it survives sidebar focus.
//   3. Installs a single capture-phase keydown fallback on `window` so the
//      shortcut still fires even if the Vue layer is bypassed (matches the
//      same defensive pattern used in mec_clipboard.js for Ctrl+C).
//   4. After opening, auto-focuses the search input so the user can type
//      immediately.
//
// Non-goals:
//   - We DO NOT replace the LiteGraph search UI. We just make it visible.
//   - We DO NOT steal Ctrl+F when the focused element is a text input
//     (textarea, input, contenteditable) — browser's native find/replace and
//     widget editing still win.
// ============================================================================

import { app } from "../../scripts/app.js";

const LOG = (...a) => console.debug("[c2c-search]", ...a);

// ---------------------------------------------------------- visibility forcer
function forceShow(el) {
    if (!el) return;
    // Some themes / Comfy versions hide the dialog via inherited CSS or via
    // a stale inline `display:none` left over from a previous open/close.
    el.style.setProperty("display", "block", "important");
    el.style.setProperty("visibility", "visible", "important");
    el.style.setProperty("opacity", "1", "important");
    el.style.setProperty("pointer-events", "auto", "important");
    // LiteGraph positions the dialog at the cursor; if no event was provided
    // it may end up at (0,0) and clip under the topbar — pin to viewport mid.
    const w = el.offsetWidth || 482;
    const h = el.offsetHeight || 265;
    const r = el.getBoundingClientRect();
    if (r.top < 8 || r.left < 8 ||
        r.right > window.innerWidth || r.bottom > window.innerHeight) {
        const x = Math.max(8, Math.round((window.innerWidth  - w) / 2));
        const y = Math.max(8, Math.round((window.innerHeight - h) / 3));
        el.style.setProperty("left", x + "px", "important");
        el.style.setProperty("top",  y + "px", "important");
    }
    // Z-index above the sidebars/topbar (Comfy sidebars use z-index 100-1000).
    el.style.setProperty("z-index", "10000", "important");
}

function findSearchBoxEl() {
    return document.querySelector(".litegraph.litesearchbox") ||
           document.querySelector(".litesearchbox");
}

function focusSearchInput(el) {
    const inp = el?.querySelector?.("input.value") || el?.querySelector?.("input[type=text]");
    if (inp) {
        try { inp.focus({ preventScroll: true }); } catch { inp.focus(); }
        try { inp.select(); } catch { /* not all inputs support select */ }
    }
}

// --------------------------------------------------------------- core action
function openNodeSearch(event) {
    const canvas = app?.canvas;
    if (!canvas || typeof canvas.showSearchBox !== "function") {
        console.warn("[c2c-search] app.canvas.showSearchBox is not available");
        return false;
    }
    // Remove a stale dialog first so we don't end up with two stacked.
    const prior = findSearchBoxEl();
    if (prior && prior.parentNode) {
        try { prior.parentNode.removeChild(prior); } catch { /* noop */ }
    }
    try {
        // LiteGraph's showSearchBox expects a MouseEvent-ish argument for
        // positioning. We pass a synthetic one centered on the viewport.
        const ev = event || new MouseEvent("mousedown", {
            clientX: Math.round(window.innerWidth / 2),
            clientY: Math.round(window.innerHeight / 3),
        });
        canvas.showSearchBox(ev);
    } catch (exc) {
        console.error("[c2c-search] showSearchBox threw:", exc);
        return false;
    }
    // Schedule visibility-forcing once the element is in the DOM. LiteGraph
    // creates it synchronously, but some patches attach it on rAF.
    const tryShow = (attempt = 0) => {
        const el = findSearchBoxEl();
        if (el) { forceShow(el); focusSearchInput(el); return; }
        if (attempt < 8) requestAnimationFrame(() => tryShow(attempt + 1));
    };
    tryShow();
    return true;
}

// --------------------------------------------------------- typing guard
function isUserTypingInField(target) {
    if (!target) return false;
    const tag = (target.tagName || "").toUpperCase();
    if (tag === "INPUT") {
        // Allow Ctrl+F to override only for *non-text* inputs like checkbox.
        const t = (target.type || "").toLowerCase();
        return !["checkbox", "radio", "button", "submit"].includes(t);
    }
    if (tag === "TEXTAREA") return true;
    if (target.isContentEditable) return true;
    return false;
}

// ----------------------------------------------------------- capture handler
function captureKeydown(e) {
    // Match Ctrl+F (Windows/Linux) and Cmd+F (Mac). Reject if shift/alt to
    // keep DevTools' "find in page" Cmd+Shift+F alone.
    const isF = e.key === "f" || e.key === "F" || e.code === "KeyF";
    if (!isF) return;
    if (!(e.ctrlKey || e.metaKey)) return;
    if (e.shiftKey || e.altKey) return;
    if (isUserTypingInField(e.target)) return;
    // Stop the browser's native page-find dialog AND any other listener.
    e.preventDefault();
    e.stopPropagation();
    if (typeof e.stopImmediatePropagation === "function") {
        e.stopImmediatePropagation();
    }
    openNodeSearch(e);
}

// =========================================================== Comfy extension
app.registerExtension({
    name: "c2c.search.ctrl_f_restore",
    commands: [
        {
            id: "c2c.search.openNodeSearch",
            label: "C2C: Open node search (Ctrl+F)",
            function: () => openNodeSearch(),
        },
    ],
    keybindings: [
        { combo: { key: "f", ctrl: true },  commandId: "c2c.search.openNodeSearch" },
        { combo: { key: "f", metaKey: true }, commandId: "c2c.search.openNodeSearch" },
    ],
    async setup() {
        // Capture phase so we beat ComfyUI 1.42+ Vue keybindings AND the
        // browser's native Ctrl+F dialog. This duplicates the registered
        // keybinding above on purpose — the Vue keybinding system silently
        // fails when input focus is in an unexpected element, and the global
        // capture listener guarantees the shortcut works everywhere.
        window.addEventListener("keydown", captureKeydown, { capture: true });
        LOG("Ctrl+F restore installed");
    },
});
