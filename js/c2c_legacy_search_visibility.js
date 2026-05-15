// c2c_legacy_search_visibility.js — keeps the stock LiteGraph add-node
// dialog visible. Companion to c2c_workflow_find.js.
// ---------------------------------------------------------------------
// Why:
//   On clean installs of ComfyUI 1.43.x+, the LiteGraph search popup
//   (`app.canvas.showSearchBox()`) is created hidden:
//     - element IS appended to <body>
//     - getComputedStyle().display === "none"
//     - offsetParent === null
//   so users click the canvas "+" button (or double-click the canvas)
//   and nothing visible happens.
//
//   c2c_workflow_find.js hijacks Ctrl+F for in-graph search; this file
//   handles every OTHER entry point (the "+" button, double-click,
//   Vue command "Comfy.NewSearch", drag-from-empty-slot) by forcing
//   the popup visible AFTER the engine creates it.
//
// Non-goals:
//   - DO NOT bind any keyboard shortcut here (Ctrl+F belongs to
//     c2c_workflow_find.js).
//   - DO NOT change WHAT the search lists; only ensure it appears.
//
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";

const TAG = "[C2C.LegacySearchVis]";

function forceShow(el) {
    if (!el) return;
    el.style.setProperty("display", "block", "important");
    el.style.setProperty("visibility", "visible", "important");
    el.style.setProperty("opacity", "1", "important");
    el.style.setProperty("pointer-events", "auto", "important");
    el.style.setProperty("z-index", "10000", "important");
    // Re-position if engine left it at (0,0) or off-screen.
    const w = el.offsetWidth || 482;
    const h = el.offsetHeight || 265;
    const r = el.getBoundingClientRect();
    if (r.top < 8 || r.left < 8
        || r.right > window.innerWidth || r.bottom > window.innerHeight) {
        const x = Math.max(8, Math.round((window.innerWidth  - w) / 2));
        const y = Math.max(8, Math.round((window.innerHeight - h) / 3));
        el.style.setProperty("left", x + "px", "important");
        el.style.setProperty("top",  y + "px", "important");
    }
    // Focus the input so user can type immediately.
    const inp = el.querySelector?.("input.value")
             || el.querySelector?.("input[type=text]");
    if (inp) {
        try { inp.focus({ preventScroll: true }); } catch { inp.focus(); }
        try { inp.select(); } catch { /* not all inputs support select */ }
    }
}

function findSearchBoxEl() {
    return document.querySelector(".litegraph.litesearchbox")
        || document.querySelector(".litesearchbox");
}

// Patch the canvas prototype the moment LiteGraph is available, so EVERY
// invocation route (canvas dbl-click, "+" button, drag-out-of-slot, the
// Vue command, etc.) benefits — without us listening to keystrokes.
function patchShowSearchBox() {
    try {
        const proto = (window.LGraphCanvas || app?.canvas?.constructor)?.prototype;
        if (!proto || proto._c2c_show_patched) return;
        const orig = proto.showSearchBox;
        if (typeof orig !== "function") return;
        proto.showSearchBox = function (...args) {
            const ret = orig.apply(this, args);
            // Engine may create the element synchronously or schedule
            // it; cover both with a microtask + a short re-check.
            queueMicrotask(() => forceShow(findSearchBoxEl()));
            setTimeout(() => forceShow(findSearchBoxEl()), 50);
            return ret;
        };
        proto._c2c_show_patched = true;
        console.log(`${TAG} patched LGraphCanvas.showSearchBox`);
    } catch (e) {
        console.warn(`${TAG} patch failed:`, e);
    }
}

// MutationObserver fallback — if some other extension creates the
// element via a different code path, we still force it visible.
function installMutationGuard() {
    const obs = new MutationObserver((muts) => {
        for (const m of muts) {
            for (const n of m.addedNodes || []) {
                if (!(n instanceof HTMLElement)) continue;
                if (n.classList?.contains?.("litesearchbox")
                    || n.querySelector?.(".litesearchbox")) {
                    const el = n.classList.contains("litesearchbox")
                        ? n : n.querySelector(".litesearchbox");
                    forceShow(el);
                }
            }
        }
    });
    obs.observe(document.body, { childList: true, subtree: false });
}

app.registerExtension({
    name: "C2C.LegacySearchVisibility",
    async setup() {
        patchShowSearchBox();
        installMutationGuard();
        // Some Comfy builds load LGraphCanvas after extensions; retry once.
        setTimeout(patchShowSearchBox, 250);
        setTimeout(patchShowSearchBox, 1500);
    },
});
