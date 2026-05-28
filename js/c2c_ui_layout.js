/**
 * c2c_ui_layout.js — single source of truth for C2C quick-action surface.
 *
 * Why this exists:
 *   Pre-v2.1, C2C had two competing surfaces for the eight quick-action
 *   buttons (Workflow Wizard, Group Presets, Mood Board, Surprise Me,
 *   A/B Split, Cost Estimator, Flame Graph, Undo Panel):
 *     1) Per-module floating buttons on document.body (right-rail).
 *     2) The c2c.launcher sidebar tab (which delegates by .click()).
 *   The `c2c.launcher.consolidate` boolean toggled between the two but
 *   produced confusing in-between states (e.g. sidebar shown AND floats
 *   shown). This file replaces that boolean with a single tri-state
 *   `c2c.ui.layout` setting:
 *
 *     "mini-row"  — horizontal toolbar pinned under the top menu,
 *                   eight color-coded tiles (this file).
 *     "sidebar"   — sidebar tab only (legacy launcher behaviour).
 *     "floating"  — original right-rail floating buttons (legacy v1).
 *
 *   In all three modes the upstream per-module logic is preserved
 *   verbatim — we only re-skin the entry points and toggle visibility
 *   of the floating buttons.
 *
 * Setting:
 *     c2c.ui.layout  (combo, default "mini-row")
 *
 * Color coding (Catppuccin Mocha palette, by category):
 *     Workflow Wizard  blue       (workflow)
 *     Group Presets    mauve      (templates)
 *     Mood Board       pink       (creative)
 *     Surprise Me      yellow     (random)
 *     A/B Split        green      (compare)
 *     Cost Estimator   peach      (economics)
 *     Flame Graph      red        (perf)
 *     Undo Panel       sapphire   (history)
 */

import { app } from "../../scripts/app.js";

const SETTING_ID = "c2c.ui.layout";
const ROW_HOST_ID = "c2c-mini-row-host";
const HIDE_STYLE_ID = "c2c-ui-layout-hide-floats";
const ROW_STYLE_ID = "c2c-ui-layout-row-style";
const SIDEBAR_ID = "c2c.launcher";       // absorbed from former c2c_launcher.js
const SIDEBAR_STYLE_ID = "c2c-launcher-sidebar-style";

const TARGETS = [
    { btnId: "mec-wizard-btn",        icon: "🧙", label: "Wizard",   tip: "Step-through workflow builder", color: "var(--c2c-blue)" },
    { btnId: "mec-group-presets-btn", icon: "📚", label: "Presets",  tip: "Save & recall node-group templates", color: "var(--c2c-mauve)" },
    { btnId: "mec-mood-btn",          icon: "🎨", label: "Mood",     tip: "Reference image palette", color: "var(--c2c-pink)" },
    { btnId: "mec-surprise-btn",      icon: "🎰", label: "Surprise", tip: "Randomize seeds & queue", color: "var(--c2c-yellow)" },
    { btnId: "mec-ab-btn",            icon: "⚖",  label: "A/B",      tip: "Compare last two outputs", color: "var(--c2c-green)" },
    { btnId: "mec-cost-btn",          icon: "💰", label: "Cost",     tip: "Estimate render cost", color: "var(--c2c-peach)" },
    { btnId: "mec-flamegraph-btn",    icon: "⏱",  label: "Flame",    tip: "Per-node execution timing", color: "var(--c2c-red)" },
    { btnId: "mec-undo-btn",          icon: "↶",  label: "Undo",     tip: "Undo history viewer", color: "var(--c2c-sapphire)" },
];

const FLOAT_SEL = TARGETS.map(t => "#" + t.btnId).join(", ");

// ── Style injection helpers ──────────────────────────────────────────

function _ensureRowStyle() {
    if (document.getElementById(ROW_STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = ROW_STYLE_ID;
    s.textContent = `
#${ROW_HOST_ID} {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: linear-gradient(180deg, var(--c2c-bg) 0%, var(--c2c-bg2) 100%);
    border-bottom: 1px solid var(--c2c-border);
    font-family: system-ui, -apple-system, sans-serif;
    user-select: none;
    z-index: var(--c2c-z-panel);
}
#${ROW_HOST_ID} .c2c-mr-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--c2c-dim);
    margin-right: 4px;
}
#${ROW_HOST_ID} .c2c-mr-tile {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 9px;
    border-radius: 5px;
    border: 1px solid var(--c2c-border);
    background: var(--c2c-bg3);
    color: var(--c2c-fg);
    font-size: 11px;
    cursor: pointer;
    transition: transform 0.06s, border-color 0.12s, background 0.12s;
    line-height: 1.1;
}
#${ROW_HOST_ID} .c2c-mr-tile:hover {
    background: var(--c2c-bg2);
}
#${ROW_HOST_ID} .c2c-mr-tile:active { transform: translateY(1px); }
#${ROW_HOST_ID} .c2c-mr-tile.c2c-missing {
    opacity: 0.35;
    cursor: not-allowed;
}
#${ROW_HOST_ID} .c2c-mr-tile.c2c-missing:hover { background: var(--c2c-bg3); }
#${ROW_HOST_ID} .c2c-mr-ico {
    font-size: 13px;
    line-height: 1;
}
#${ROW_HOST_ID} .c2c-mr-txt {
    font-size: 10.5px;
    font-weight: 500;
}
    `.trim();
    document.head.appendChild(s);
}

function _ensureHideFloatsStyle(active) {
    let s = document.getElementById(HIDE_STYLE_ID);
    if (!active) {
        if (s) s.remove();
        return;
    }
    if (!s) {
        s = document.createElement("style");
        s.id = HIDE_STYLE_ID;
        document.head.appendChild(s);
    }
    s.textContent = `${FLOAT_SEL} { display: none !important; }`;
}

// ── Delegation (same trick as c2c_launcher) ──────────────────────────

function _delegate(btnId) {
    const t = document.getElementById(btnId);
    if (!t) return false;
    const prev = t.style.display;
    t.style.display = "flex";
    try { t.click(); }
    finally { requestAnimationFrame(() => { t.style.display = prev; }); }
    return true;
}

// ── Mini-row mount / unmount ─────────────────────────────────────────

function _findMenuAnchor() {
    // Deprecated DOM-insertion anchor lookup. Modern PrimeVue ComfyUI
    // uses absolutely-positioned top chrome that isn't in normal flow,
    // so DOM-after-insertion no longer translates to visual stacking.
    // Kept only so legacy callers don't crash; the mini-row mount path
    // now uses fixed positioning below the measured native chrome.
    return null;
}

// Measure how much vertical room native ComfyUI top chrome currently
// claims (top app menu strip + workflow-tabs row). Used to position the
// mini-row IMMEDIATELY BELOW that strip so we never overlap interactive
// native UI. Survives subgraph/canvas pan because we re-measure on every
// resize and DOM mutation tick.
function _measureTopChromeBottom() {
    let bottom = 0;
    // Include `.workflow-tabs-container` (full-width wrapper around the
    // narrow `.workflow-tabs` tablist) so we always pick up the deepest
    // visual edge of the native top chrome — even when only one workflow
    // is open and `.workflow-tabs` is only ~190px wide.
    for (const sel of [".comfyui-body-top", ".workflow-tabs-container", ".workflow-tabs", ".comfyui-menu", "#comfyui-menu", ".comfy-menu"]) {
        const el = document.querySelector(sel);
        if (!el) continue;
        const r = el.getBoundingClientRect();
        if (r.height > 0 && r.width > 0 && r.bottom > bottom) bottom = r.bottom;
    }
    // Also stack BELOW the shared __c2cTopDock strip so our buttons
    // never collide with autockpt/doctor pinned in the dock above.
    for (const id of ["c2c-top-dock-left", "c2c-top-dock-right"]) {
        const el = document.getElementById(id);
        if (!el) continue;
        const r = el.getBoundingClientRect();
        if (r.height > 0 && r.bottom > bottom) bottom = r.bottom;
    }
    return Math.max(0, Math.round(bottom));
}

// Measure how far the left rail (PrimeVue side-tool-bar) extends so we
// can clear it. Returns 0 if the rail is absent or hidden.
function _measureLeftRailRight() {
    const el = document.querySelector(".side-tool-bar-container");
    if (!el) return 0;
    const r = el.getBoundingClientRect();
    if (r.height <= 0 || r.width <= 0) return 0;
    return Math.max(0, Math.round(r.right));
}

function _mountMiniRow() {
    if (document.getElementById(ROW_HOST_ID)) return;
    _ensureRowStyle();

    // ── OmniBar-only consolidation (LOCKED) ──────────────────────────
    // Per locked user spec ("all of them should be there in omnibar
    // only. no need of duplication."), the eight C2C quick-action
    // tiles MUST live exclusively inside the OmniBar Tools section.
    // There is no legacy standalone strip — falling back to one would
    // double-render the tiles when OmniBar mounts after this module
    // (race condition observed live 2026-05-27).
    //
    // Strategy: register all tiles unconditionally with a deferred
    // registrar that waits for window.C2COmniBar.register to exist,
    // then registers each tile exactly once. A hidden sentinel host
    // satisfies any legacy code that still queries ROW_HOST_ID.
    const sentinel = document.createElement("div");
    sentinel.id = ROW_HOST_ID;
    sentinel.style.display = "none";
    sentinel.setAttribute("data-c2c-overlay", "1");
    sentinel.setAttribute("data-c2c-mini-row-mode", "omnibar");
    document.body.appendChild(sentinel);

    _miniRowOmniSlots = [];
    const tiles = [];
    let order = 100;
    for (const t of TARGETS) {
        const tile = document.createElement("button");
        tile.type = "button";
        tile.className = "c2c-mr-tile c2c-mr-tile--omnibar";
        tile.title = t.tip;
        tile.style.borderLeft = `3px solid ${t.color}`;
        tile.innerHTML = `<span class="c2c-mr-ico" style="color:${t.color}">${t.icon}</span>` +
                         `<span class="c2c-mr-txt">${t.label}</span>`;
        tile.addEventListener("click", (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            const ok = _delegate(t.btnId);
            if (!ok) {
                tile.classList.add("c2c-missing");
                tile.title = `${t.label} — module not loaded`;
            }
        });
        if (!document.getElementById(t.btnId)) {
            setTimeout(() => {
                if (!document.getElementById(t.btnId)) {
                    tile.classList.add("c2c-missing");
                    tile.title = `${t.label} — module not loaded`;
                }
            }, 2500);
        }
        tiles.push({ tile, target: t, order });
        _miniRowOmniSlots.push(tile);
        order += 10;
    }

    let _registered = false;
    const _tryRegister = () => {
        if (_registered) return true;
        const omni = window.C2COmniBar;
        if (!omni || typeof omni.register !== "function") return false;
        for (const { tile, target, order: ord } of tiles) {
            omni.register({
                section: "tools",
                id: "mini-row-" + target.btnId,
                order: ord,
                element: tile,
                onMode: (mode) => {
                    const txt = tile.querySelector(".c2c-mr-txt");
                    if (txt) txt.style.display = mode === "icon" ? "none" : "";
                },
            });
        }
        try { omni.refresh?.(); } catch (err) { void err; }
        _registered = true;
        return true;
    };

    if (!_tryRegister()) {
        // OmniBar not mounted yet — poll briefly (every 50 ms, up to
        // 60 s). Each tick is cheap; we stop the instant OmniBar
        // appears. We do NOT build a standalone strip in the meantime
        // because that's exactly what produced the duplicate floating
        // tiles. While waiting the tiles are simply parked off-DOM —
        // user sees an empty Tools section until OmniBar mounts (a
        // span of a few hundred ms at most in practice).
        let ticks = 0;
        const maxTicks = 1200; // 50 ms * 1200 = 60 s safety cap
        const iv = setInterval(() => {
            if (_tryRegister() || ++ticks >= maxTicks) {
                clearInterval(iv);
            }
        }, 50);
    }
}

let _miniRowRepositioner = null;
let _miniRowResizeObserver = null;
let _miniRowOmniSlots = null;

function _unmountMiniRow() {
    const h = document.getElementById(ROW_HOST_ID);
    if (h) h.remove();
    // Tear down the reposition listeners so we don't keep paying the
    // ResizeObserver cost while the row is dormant. Re-attached on the
    // next mount.
    if (_miniRowRepositioner) {
        window.removeEventListener("resize", _miniRowRepositioner);
        _miniRowRepositioner = null;
    }
    if (_miniRowResizeObserver) {
        try { _miniRowResizeObserver.disconnect(); } catch { /* already disconnected */ }
        _miniRowResizeObserver = null;
    }
}

// ── Sidebar tab (absorbed from c2c_launcher.js) ──────────────────────
// The sidebar tab mounts an 8-tile grid that delegates the same way
// the mini-row does. It is registered unconditionally so it is always
// available from the left rail regardless of `c2c.ui.layout` mode.

function _ensureSidebarStyle() {
    if (document.getElementById(SIDEBAR_STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = SIDEBAR_STYLE_ID;
    s.textContent = `
.c2c-launcher-root { padding:12px; color:var(--c2c-fg); font-family:system-ui,-apple-system,sans-serif; font-size:13px; height:100%; box-sizing:border-box; overflow-y:auto; }
.c2c-launcher-header { font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:var(--c2c-dim); margin:4px 0 10px 0; }
.c2c-launcher-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; }
.c2c-launcher-tile { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px; padding:14px 8px; background:var(--c2c-bg); border:1px solid var(--c2c-border); border-radius:8px; color:var(--c2c-fg); cursor:pointer; transition:border-color 0.12s,transform 0.08s,background 0.12s; user-select:none; }
.c2c-launcher-tile:hover { border-color:var(--c2c-blue); background:var(--c2c-bg2); }
.c2c-launcher-tile:active { transform:scale(0.97); }
.c2c-launcher-tile.c2c-missing { opacity:0.4; cursor:not-allowed; }
.c2c-launcher-tile.c2c-missing:hover { border-color:var(--c2c-border); background:var(--c2c-bg); }
.c2c-launcher-icon { font-size:22px; line-height:1; }
.c2c-launcher-label { font-size:11px; text-align:center; color:var(--c2c-subtext1); }
.c2c-launcher-foot { margin-top:14px; padding-top:10px; border-top:1px solid var(--c2c-border); font-size:11px; color:var(--c2c-dim); line-height:1.5; }
    `.trim();
    document.head.appendChild(s);
}

function _mountSidebar(el) {
    _ensureSidebarStyle();
    el.innerHTML = "";
    const root = document.createElement("div");
    root.className = "c2c-launcher-root";

    const header = document.createElement("div");
    header.className = "c2c-launcher-header";
    header.textContent = "Code2Collapse — Quick Actions";
    root.appendChild(header);

    const grid = document.createElement("div");
    grid.className = "c2c-launcher-grid";
    for (const t of TARGETS) {
        const tile = document.createElement("div");
        tile.className = "c2c-launcher-tile";
        tile.title = t.tip;
        tile.innerHTML =
            `<div class="c2c-launcher-icon" style="color:${t.color}">${t.icon}</div>` +
            `<div class="c2c-launcher-label">${t.label}</div>`;
        tile.addEventListener("click", () => {
            const ok = _delegate(t.btnId);
            if (!ok) {
                tile.classList.add("c2c-missing");
                tile.title = `${t.label} — module not loaded`;
            }
        });
        if (!document.getElementById(t.btnId)) {
            setTimeout(() => {
                if (!document.getElementById(t.btnId)) {
                    tile.classList.add("c2c-missing");
                    tile.title = `${t.label} — module not loaded`;
                }
            }, 2500);
        }
        grid.appendChild(tile);
    }
    root.appendChild(grid);

    const foot = document.createElement("div");
    foot.className = "c2c-launcher-foot";
    foot.textContent = "Tiles forward to the matching action module. " +
        "Change 'C2C: quick-action layout' in Settings to switch between mini-row, sidebar, or legacy floating buttons.";
    root.appendChild(foot);

    el.appendChild(root);
}

// ── Mode application ─────────────────────────────────────────────────

function _applyLayout(mode) {
    switch (mode) {
        case "mini-row":
            _mountMiniRow();
            _ensureHideFloatsStyle(true);
            break;
        case "sidebar":
            _unmountMiniRow();
            _ensureHideFloatsStyle(true);
            break;
        case "floating":
            _unmountMiniRow();
            _ensureHideFloatsStyle(false);
            break;
        default:
            _unmountMiniRow();
            _ensureHideFloatsStyle(true);
    }
}

app.registerExtension({
    name: "C2C.UILayout",
    settings: [
        {
            id: SETTING_ID,
            name: "C2C: quick-action layout",
            tooltip:
                "Where to show the eight C2C quick-action buttons. " +
                "'mini-row' pins a color-coded toolbar under the top menu, " +
                "'sidebar' uses the C2C sidebar tab only, " +
                "'floating' restores the legacy right-rail buttons.",
            type: "combo",
            options: [
                { value: "mini-row", text: "Mini-row (under top menu)" },
                { value: "sidebar",  text: "Sidebar tab only" },
                { value: "floating", text: "Floating buttons (legacy)" },
            ],
            defaultValue: "mini-row",
            onChange: (v) => _applyLayout(v),
        },
    ],
    async setup() {
        let mode = "mini-row";
        try {
            mode = app.ui.settings.getSettingValue(SETTING_ID, "mini-row");
        } catch { /* setting not yet registered */ }

        // Register the C2C sidebar tab unconditionally so it is always
        // discoverable from the left rail, regardless of layout mode.
        try {
            app.extensionManager?.registerSidebarTab?.({
                id: SIDEBAR_ID,
                icon: "pi pi-th-large",
                title: "C2C",
                tooltip: "Code2Collapse quick actions launcher",
                type: "custom",
                render(el) { _mountSidebar(el); },
            });
        } catch (e) {
            console.warn("[C2C.UILayout] sidebar register failed:", e);
        }

        // The mini-row needs the top menu DOM to exist; wait one frame
        // after setup() returns so ComfyUI has finished mounting.
        const apply = () => _applyLayout(mode);
        if (document.readyState === "complete") {
            requestAnimationFrame(apply);
        } else {
            window.addEventListener("load", () => requestAnimationFrame(apply), { once: true });
        }
    },
});
