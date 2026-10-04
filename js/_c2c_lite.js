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
// Toggle: Settings → C2C → Performance → "Performance mode". Changing it writes
// localStorage and the value applies on the next page load (extensions are
// imported once at startup), so we offer a one-click reload.

import { app } from "/scripts/app.js";
import { getRuntime, probeGpuSoftware } from "./_c2c_runtime.js";

const LS_MODE = "c2c.perf.mode";
const MODE_AUTO = "Auto (recommended)";
const MODE_FULL = "Full";
const MODE_LITE = "Lite";

const _rt = getRuntime();
export const TIER = _rt.tier();
export const LITE = TIER === "lite";

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
    "C2C.NodeBookmarks", "tokens",   // NOT "C2C.TokenCounter" — that name never existed
    "C2C.SurpriseMe",
    "C2C.CostEstimator", "C2C.MetadataInspector", "C2C.OverlayVisibility",
    "c2c.ai.statusBar", "MEC.IntegrityStatus",
    "C2C.NodeExplain", "C2C.ProgressHUD", "C2C.OmniBar", "Yellow",
    "C2C.CompatibilityHints", "C2C.DoctorV3", "C2C.DiagnosticsSidebar",
    "C2C.WorkflowWizard", "C2C.WSLogger",
    "C2C.ABSplit", "C2C.ColorspaceBadges", "C2C.CompletionFX",
    "C2C.ComplexityHUD", "C2C.DockAnchor", "C2C.FlameGraph",
    "C2C.NoodleStyles", "C2C.WireLabels",
]);

// Everything that makes an extension COST something at runtime. Lite mode
// strips exactly these and passes the rest through.
const BEHAVIOUR_KEYS = [
    "init", "setup", "aboutPageBadges", "commands", "keybindings", "menuCommands",
    "beforeRegisterNodeDef", "beforeRegisterVueAppNodeDefs", "registerCustomNodes",
    "loadedGraphNode", "nodeCreated", "beforeConfigureGraph", "afterConfigureGraph",
    "onNodeOutputsUpdated", "getCustomWidgets", "getSelectionToolboxCommands",
];

if (LITE && typeof app.registerExtension === "function"
        && !app.registerExtension._c2cLitePatched) {
    const _origReg = app.registerExtension.bind(app);
    // Rest args, not (ext): this replaces app.registerExtension for the WHOLE
    // app, and ComfyUI has added parameters to it before — swallowing them
    // would silently drop options for every other extension pack installed.
    const _filtered = function (ext, ...rest) {
        if (ext && SKIP_WHEN_LITE.has(ext.name)) {
            // Register a STRIPPED extension rather than dropping it.
            //
            // Dropping it outright is what made "all my settings vanished":
            // ComfyUI builds the settings panel from the `settings` array on the
            // registered extension, so an unregistered extension has no entries
            // — and the values the user had already chosen have nothing left to
            // attach to. Lite mode is supposed to turn BEHAVIOUR off, never to
            // take away the control that turns it back on.
            //
            // So keep name + settings (+ their onChange, which is how the user
            // re-enables things), and drop only the keys that install work:
            // timers, draw hooks, node hooks, DOM.
            const lean = { name: ext.name };
            if (ext.settings) lean.settings = ext.settings;
            for (const k of Object.keys(ext)) {
                if (k === "name" || k === "settings") continue;
                if (!BEHAVIOUR_KEYS.includes(k)) lean[k] = ext[k];   // inert data
            }
            try { console.debug("[C2C.Lite] stripped", ext.name); } catch (_) {}
            return _origReg(lean, ...rest);
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

function _readStoredMode() {
    try {
        const v = localStorage.getItem(LS_MODE);
        if (v === MODE_AUTO || v === MODE_FULL || v === MODE_LITE) return v;
        if (!v && localStorage.getItem("c2c.lite") === "1") return MODE_LITE;
    } catch (_) {}
    return MODE_AUTO;
}

function _writeMode(mode) {
    try { localStorage.setItem(LS_MODE, mode); } catch (_) {}
}

// `ask` only when the USER just changed the setting. Automatic detection must
// never open a blocking confirm() on page load (it freezes every script on the
// page until answered, and nobody asked); it only informs, and the
// "C2C: GPU & performance status" command carries the Reload button.
function _offerReload(summary, detail, { ask = true } = {}) {
    try {
        const t = app.extensionManager?.toast;
        if (t?.add) {
            t.add({
                severity: "info",
                summary,
                detail,
                life: 8000,
            });
        }
    } catch (_) {}
    if (!ask) return;
    setTimeout(() => {
        try {
            if (window.confirm(`${detail}\nReload now to apply?`)) {
                location.reload();
            }
        } catch (_) {}
    }, 50);
}

// localStorage is the SOLE source of truth (read at module-eval). ComfyUI fires
// the setting's onChange with its server-stored value during init, which must
// NOT be allowed to clobber localStorage — so onChange is ignored until the user
// can actually interact (after setup).
let _initDone = false;
let _gpuProbeDone = false;

if (!(app.extensions || []).some((e) => e?.name === "C2C.LiteMode")) app.registerExtension({
    name: "C2C.LiteMode",
    settings: [
        {
            id: "c2c.perf.mode",
            name: "Performance mode — Auto detects software rendering; Lite disables C2C visual extras",
            tooltip: "Auto (recommended) enables Lite when the browser renders without a GPU "
                   + "(SwiftShader, llvmpipe, etc.). Lite turns off eye-candy and always-on overlays "
                   + "while keeping functional tools. Applies after a page reload.",
            type: "combo",
            options: [MODE_AUTO, MODE_FULL, MODE_LITE],
            defaultValue: _readStoredMode(),
            category: ["c2c", "Performance", "Performance mode"],
            onChange: (v) => {
                if (!_initDone) return;
                const mode = (v === MODE_FULL || v === MODE_LITE) ? v : MODE_AUTO;
                let changed = false;
                try { changed = _readStoredMode() !== mode; _writeMode(mode); } catch (_) {}
                if (!changed) return;
                _offerReload(
                    "C2C Performance mode",
                    `Performance mode set to "${mode}" — reload to apply.`,
                );
            },
        },
    ],
    async setup() {
        try { app.ui.settings.setSettingValue("c2c.perf.mode", _readStoredMode()); } catch (_) {}
        setTimeout(() => { _initDone = true; }, 800);
        if (LITE) {
            try {
                const { C } = await import("./_c2c_theme.js");
                console.log("%c[C2C.Lite] active — visual extras disabled for performance", `color:${C.sky}`);
            } catch (_) {}
        }
        // Live GPU probe: once per page load, never at module eval.
        const _runGpuProbe = () => {
            if (_gpuProbeDone) return;
            _gpuProbeDone = true;
            try {
                const entry = probeGpuSoftware();
                const rt = getRuntime();
                const evalTier = TIER;
                const probedTier = rt.tierAfterProbe(entry);
                rt.refreshTierCache();
                if (entry.software) {
                    import("./c2c_gpu_status.js")
                        .then((m) => m.onSoftwareProbeResult(entry))
                        .catch(() => {});
                }
                if (_readStoredMode() === MODE_AUTO && probedTier !== evalTier && probedTier === "lite") {
                    _offerReload(
                        "C2C Performance mode",
                        "C2C: switched to Lite on next reload because this browser is rendering without the GPU.",
                        { ask: false },
                    );
                }
            } catch (_) {}
        };
        if (typeof requestIdleCallback === "function") {
            requestIdleCallback(_runGpuProbe, { timeout: 3000 });
        } else {
            setTimeout(_runGpuProbe, 500);
        }
    },
});
