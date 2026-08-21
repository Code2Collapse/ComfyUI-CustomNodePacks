// _c2c_lite.js — C2C "Lite / Performance mode".
// ---------------------------------------------------------------------------
// This pack ships 100+ JS extensions. On a busy graph (1000s of nodes) the
// cumulative per-frame + per-event overhead of the *visual extras* (completion
// FX, animated noodles, always-on HUD pills, per-node badges, mood board, etc.)
// is what makes a loaded box feel sluggish/unresponsive. Lite mode lets the user
// switch those OFF so only the functional tools remain.
//
// HOW IT WORKS (load-order-proof): the flag lives in localStorage and is read
// SYNCHRONOUSLY at module-eval time, BEFORE any heavy extension registers. Each
// gated extension does `import { LITE } from "./_c2c_lite.js"` — the ES import
// guarantees this module evaluates first — and wraps its registration in
// `if (!LITE) …`. So in lite mode the heavy extension never registers: its draw
// hooks, DOM, and timers are never installed (true load reduction, not a flag
// checked every frame).
//
// Toggle: Settings → C2C → Performance → "Lite mode". Changing it writes
// localStorage and the value applies on the next page load (extensions are
// imported once at startup), so we offer a one-click reload.

import { app } from "/scripts/app.js";

const LS_KEY = "c2c.lite";

export const LITE = (() => {
    try { return localStorage.getItem(LS_KEY) === "1"; } catch (_) { return false; }
})();

// ── central LITE filter ────────────────────────────────────────────────────
// Gating one file at a time means editing every heavy extension. This instead
// filters at the single choke point every extension goes through, by NAME, so a
// new visual extra is opted out with one line here.
//
// LOAD ORDER IS THE WHOLE GAME. ComfyUI discovers extensions with a plain
// recursive glob (server.py get_extensions) whose order is filesystem-dependent
// and NOT guaranteed sorted. If a listed extension evaluates before this module,
// its registerExtension call has already happened and the filter is useless.
// Every file named below therefore carries `import "./_c2c_lite.js"` — an ES
// import forces this module to evaluate first, turning a coin flip into a
// guarantee. If you add a name here, add that import to its file too.
//
// Only extensions NO NODE DEPENDS ON belong here. Gating a node's own widget
// script leaves that node with no UI, which is a bug, not a saving.
const SKIP_WHEN_LITE = new Set([
    "C2C.StatsPill", "C2C.IntBadge", "C2C.StatusStrip", "C2C.TopDock",
    "C2C.UILayout", "C2C.MoodBoard", "C2C.FrameOverlay", "C2C.GraphHealth",
    "C2C.NodeBookmarks", "C2C.TokenCounter", "C2C.SurpriseMe",
    "C2C.CostEstimator", "C2C.MetadataInspector", "C2C.OverlayVisibility",
    "c2c.ai.statusBar", "MEC.IntegrityStatus",
]);

if (LITE && typeof app.registerExtension === "function"
        && !app.registerExtension._c2cLitePatched) {
    const _origReg = app.registerExtension.bind(app);
    // Rest args, not (ext): ComfyUI has added parameters to registerExtension
    // before, and swallowing them here would silently drop options on every
    // extension in the app, not just ours.
    const _filtered = function (ext, ...rest) {
        if (ext && SKIP_WHEN_LITE.has(ext.name)) {
            try { console.debug("[C2C.Lite] skipped", ext.name); } catch (_) {}
            return undefined;
        }
        return _origReg(ext, ...rest);
    };
    _filtered._c2cLitePatched = true;
    app.registerExtension = _filtered;
}

// Optional helper for gated files that prefer a function call.
export function liteSkip(label) {
    if (LITE && label) { try { console.debug(`[C2C.Lite] skipped ${label}`); } catch (_) {} }
    return LITE;
}

// localStorage is the SOLE source of truth (read at module-eval). ComfyUI fires
// the setting's onChange with its server-stored value during init, which must
// NOT be allowed to clobber localStorage — so onChange is ignored until the user
// can actually interact (after setup).
let _initDone = false;

if (!(app.extensions || []).some((e) => e?.name === "C2C.LiteMode")) app.registerExtension({
    name: "C2C.LiteMode",
    settings: [
        {
            id: "c2c.lite.enabled",
            name: "Lite mode — disable C2C visual extras (FX, animated noodles, HUD pills, badges) for performance",
            tooltip: "Recommended on heavy graphs / low-RAM machines. Keeps all functional tools; turns off "
                   + "eye-candy and always-on overlays. Applies after a page reload.",
            type: "boolean",
            defaultValue: LITE,
            category: ["c2c", "Performance", "Lite mode"],
            onChange: (v) => {
                if (!_initDone) return;          // ignore the init echo + our own sync (don't clobber localStorage)
                const on = !!v;
                let changed = false;
                try { changed = (localStorage.getItem(LS_KEY) === "1") !== on; localStorage.setItem(LS_KEY, on ? "1" : "0"); } catch (_) {}
                if (!changed) return;
                // Offer an immediate reload so the change takes effect.
                try {
                    const t = app.extensionManager?.toast;
                    if (t?.add) {
                        t.add({ severity: "info", summary: "C2C Lite mode",
                                detail: `Lite mode ${on ? "ON" : "OFF"} — reload to apply.`, life: 6000 });
                    }
                } catch (_) {}
                // A gentle confirm to reload now (skips if the host blocks dialogs).
                setTimeout(() => {
                    try {
                        if (window.confirm(`C2C Lite mode ${on ? "enabled" : "disabled"}.\nReload now to apply?`)) {
                            location.reload();
                        }
                    } catch (_) {}
                }, 50);
            },
        },
    ],
    async setup() {
        // Sync the checkbox to the real (localStorage) state without writing back
        // (onChange is still gated by _initDone), then allow user toggles.
        try { app.ui.settings.setSettingValue("c2c.lite.enabled", LITE); } catch (_) {}
        setTimeout(() => { _initDone = true; }, 800);
        if (LITE) { try { console.log("%c[C2C.Lite] active — visual extras disabled for performance", "color:#8cf"); } catch (_) {} }
    },
});
