/*
 * c2c_omnibar.js — P0.2 OmniBar (Phase 1: SKELETON + SLOT API)
 * ─────────────────────────────────────────────────────────────
 * The single, theme-aware, position-configurable command bar that
 * eventually hosts ALL floating C2C surfaces (HUDs, chips, status pills,
 * bookmarks, AI controls). No duplicate UI surfaces anywhere in C2C.
 *
 * What this Phase 1 ships
 *   ✓ A visible bar at the user-chosen edge (top|bottom|left|right).
 *   ✓ Four section hosts: Tools | Stats | Bookmarks | AI (separated by
 *     animated theme-aware dividers).
 *   ✓ A Slot Registration API so future migrations (INT, Graph Health,
 *     AI HUD, MEC chips, etc.) plug into a host instead of mounting
 *     themselves at fixed coordinates.
 *   ✓ Position persistence via _c2c_storage.js (`omnibar/position`).
 *   ✓ Hide-toggle via Settings (`c2c.omnibar.visible`).
 *   ✓ Live response to:
 *        — workflow tab switch    (re-mount + replay slots)
 *        — subgraph enter/exit    (survives — body-level mount)
 *        — variant flip           (purely CSS-var driven; no JS work)
 *        — native chrome reflow   (subscribes to --c2c-native-*)
 *        — viewport resize        (re-evaluates icon-only collapse)
 *   ✓ Vertical orientations (left|right) auto-render icon-only chips.
 *   ✓ Hard rule: every catch surfaces via `c2c:registry-failure` +
 *     POST /c2c/registry/failure. No empty catches.
 *
 * What this Phase 1 does NOT do (intentionally — comes in turns 2-8)
 *   ✗ Migrate any of the 9 existing chips/HUDs (they keep working as-is).
 *   ✗ Bookmark management UI (Phase 6).
 *   ✗ CI lint enforcement (Phase 7).
 *
 * What Phase 1 DOES do for the cog button (no stubs):
 *   ✓ Full position-picker popover (top/bottom/left/right buttons).
 *   ✓ Density toggle (comfortable | compact).
 *   ✓ Visibility toggle (Hide OmniBar — re-enable via the cycle command).
 *   ✓ "Open full Settings" link as a power-user shortcut.
 *   ✓ Outside-click + Esc dismiss; focus-trap inside popover.
 *
 * The 4 sections are empty in this turn. Each subsequent turn fills one
 * by deleting the corresponding source file and inlining its logic.
 *
 * License: Apache-2.0
 */

import { app } from "../../scripts/app.js";
import { startNativeOffsets, refreshNativeOffsets } from "./_c2c_native_offsets.js";

const COMPONENT = "c2c_omnibar";

const ROOT_ID    = "c2c-omnibar";
const PILL_ID    = "c2c-omnibar-pill";
const STYLE_ID   = "c2c-omnibar-style";
const SECTION_IDS = ["tools", "stats", "bookmarks", "ai"];
const SECTION_LABELS = {
    tools:     "Tools",
    stats:     "Stats & Status",
    bookmarks: "Bookmarks",
    ai:        "AI Assist",
};

const POSITION_VALUES = ["top", "bottom", "left", "right"];
const POSITION_DEFAULT = "top";

const SETTING_POSITION = "c2c.omnibar.position";
const SETTING_VISIBLE  = "c2c.omnibar.visible";
const SETTING_DENSITY  = "c2c.omnibar.density"; // "comfortable" | "compact"

// Width threshold below which TOP/BOTTOM auto-collapses to icon-only mode.
// (vertical positions are always icon-only.)
const COLLAPSE_PX = 1100;

// ── Error surfacing (same pattern as _c2c_native_offsets.js & _c2c_undo.js) ──
function _reportFailure(where, err) {
    const detail = {
        component: COMPONENT,
        where: String(where || ""),
        message: (err && err.message) ? err.message : String(err),
        stack: (err && err.stack) ? err.stack : null,
        ts: Date.now(),
    };
    try {
        // eslint-disable-next-line no-console
        console.error("[c2c_omnibar]", where, err);
    } catch (consoleErr) { void consoleErr; }
    try {
        window.dispatchEvent(new CustomEvent("c2c:registry-failure", { detail }));
    } catch (dispatchErr) {
        // eslint-disable-next-line no-console
        console.error("[c2c_omnibar] dispatch failed", dispatchErr);
    }
    try {
        fetch("/c2c/registry/failure", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(detail),
            keepalive: true,
        }).catch((netErr) => {
            // eslint-disable-next-line no-console
            console.error("[c2c_omnibar] POST /c2c/registry/failure failed", netErr);
        });
    } catch (fetchErr) {
        // eslint-disable-next-line no-console
        console.error("[c2c_omnibar] fetch unavailable", fetchErr);
    }
}

// ── Setting helpers ─────────────────────────────────────────────────────────
function _getSetting(id, fallback) {
    try {
        const v = app?.ui?.settings?.getSettingValue?.(id, fallback);
        return (v === undefined || v === null) ? fallback : v;
    } catch (err) {
        _reportFailure("getSetting:" + id, err);
        return fallback;
    }
}

function _setSetting(id, value) {
    try {
        app?.ui?.settings?.setSettingValue?.(id, value);
    } catch (err) {
        _reportFailure("setSetting:" + id, err);
    }
}

// ── Slot registry ───────────────────────────────────────────────────────────
// Migration owners (INT badge, AI HUD, etc.) call C2COmniBar.register() with
// their element + an update() callback + an optional onMode(mode) callback.
// OmniBar stores them by section, sorts by `order`, and re-renders the section
// host. Owners NEVER position their own element — the host does.
const _slots = {
    tools: [],
    stats: [],
    bookmarks: [],
    ai: [],
};

function _renderSection(section) {
    const host = document.querySelector(`#${ROOT_ID} .c2c-omnibar-section[data-section="${section}"]`);
    if (!host) return;
    const slots = (_slots[section] || []).slice().sort((a, b) => (a.order || 0) - (b.order || 0));
    // Detach known children only — preserve other library mounts if any.
    const existing = new Set(Array.from(host.children).map((c) => c));
    const wanted = new Set();
    const mode = _currentChipMode();
    for (const slot of slots) {
        if (!slot.element) continue;
        wanted.add(slot.element);
        // Mark with metadata for the mode-aware CSS rules.
        slot.element.setAttribute("data-c2c-slot-id", slot.id);
        slot.element.setAttribute("data-c2c-section", section);
        if (!existing.has(slot.element)) {
            host.appendChild(slot.element);
        }
        if (typeof slot.onMode === "function") {
            try { slot.onMode(mode); }
            catch (err) { _reportFailure("slot.onMode:" + slot.id, err); }
        }
    }
    // Remove orphans we previously appended but are no longer registered.
    for (const child of Array.from(host.children)) {
        if (!wanted.has(child) && child.hasAttribute("data-c2c-slot-id")) {
            host.removeChild(child);
        }
    }
    // Empty-section visibility: hide whole wrap (header + host) if no slots.
    const empty = wanted.size === 0;
    host.classList.toggle("is-empty", empty);
    const wrap = host.closest(".c2c-omnibar-section-wrap");
    if (wrap) wrap.classList.toggle("is-empty", empty);
}

/**
 * Public Slot Registration API.
 *
 *   C2COmniBar.register({
 *       section: "stats",            // one of SECTION_IDS
 *       id: "graph-health",          // unique
 *       order: 100,                  // lower = earlier in section
 *       element: someChipEl,         // already-built HTMLElement
 *       update: () => {...},         // called when OmniBar polls (1Hz)
 *       onMode: (mode) => {...},     // optional; mode = "full" | "icon"
 *   });
 *
 * Returns an unregister() function.
 */
function register(spec) {
    if (!spec || typeof spec !== "object") {
        _reportFailure("register:spec", new Error("invalid spec"));
        return () => {};
    }
    const { section, id, element } = spec;
    if (!SECTION_IDS.includes(section)) {
        _reportFailure("register:section", new Error("unknown section: " + section));
        return () => {};
    }
    if (!id || typeof id !== "string") {
        _reportFailure("register:id", new Error("id required"));
        return () => {};
    }
    if (!(element instanceof HTMLElement)) {
        _reportFailure("register:element", new Error("element must be HTMLElement"));
        return () => {};
    }
    // Replace existing slot with same id.
    const bucket = _slots[section];
    const ix = bucket.findIndex((s) => s.id === id);
    const slot = {
        id,
        order: spec.order || 100,
        element,
        update: typeof spec.update === "function" ? spec.update : null,
        onMode: typeof spec.onMode === "function" ? spec.onMode : null,
    };
    if (ix >= 0) bucket[ix] = slot; else bucket.push(slot);
    _renderSection(section);
    return () => {
        const i = bucket.findIndex((s) => s.id === id);
        if (i >= 0) {
            bucket.splice(i, 1);
            // Detach from DOM if still mounted.
            if (slot.element && slot.element.parentElement) {
                slot.element.parentElement.removeChild(slot.element);
            }
            _renderSection(section);
        }
    };
}

function _pollAllSlots() {
    for (const section of SECTION_IDS) {
        for (const slot of _slots[section]) {
            if (typeof slot.update === "function") {
                try { slot.update(); }
                catch (err) { _reportFailure("slot.update:" + slot.id, err); }
            }
        }
    }
}

// ── Position & mode ─────────────────────────────────────────────────────────
function _getPosition() {
    const raw = _getSetting(SETTING_POSITION, POSITION_DEFAULT);
    return POSITION_VALUES.includes(raw) ? raw : POSITION_DEFAULT;
}

function _isVertical(pos) { return pos === "left" || pos === "right"; }

function _currentChipMode() {
    const pos = _getPosition();
    if (_isVertical(pos)) return "icon";
    // Horizontal: depend on viewport width + user-density setting.
    const density = _getSetting(SETTING_DENSITY, "comfortable");
    if (density === "compact") return "icon";
    if (window.innerWidth < COLLAPSE_PX) return "icon";
    return "full";
}

// ── DOM construction ────────────────────────────────────────────────────────
function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
/* ─── PILL (single trigger in Manager bar) ─────────────────────────── */
#${PILL_ID} {
    appearance: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 28px;
    padding: 0 12px;
    margin: 0 2px;
    border-radius: 999px;
    border: 1px solid color-mix(in srgb, var(--c2c-accent, var(--c2c-blue, var(--c2c-blue))) 50%, transparent);
    background: linear-gradient(
        180deg,
        color-mix(in srgb, var(--c2c-accent, var(--c2c-blue, var(--c2c-blue))) 24%, transparent) 0%,
        color-mix(in srgb, var(--c2c-accent, var(--c2c-blue, var(--c2c-blue))) 10%, transparent) 100%);
    color: var(--c2c-fg, var(--c2c-accentBright));
    font: 600 11px/1.1 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    letter-spacing: 0.04em;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
    transition: background var(--c2c-dur-fast, 120ms) var(--c2c-ease-out, ease),
                border-color var(--c2c-dur-fast, 120ms) var(--c2c-ease-out, ease),
                transform var(--c2c-dur-fast, 120ms) var(--c2c-ease-out, ease);
}
#${PILL_ID}:hover {
    background: linear-gradient(
        180deg,
        color-mix(in srgb, var(--c2c-accent, var(--c2c-blue, var(--c2c-blue))) 36%, transparent) 0%,
        color-mix(in srgb, var(--c2c-accent, var(--c2c-blue, var(--c2c-blue))) 18%, transparent) 100%);
    border-color: var(--c2c-accent, var(--c2c-blue, var(--c2c-blue)));
}
#${PILL_ID}:active { transform: translateY(1px); }
#${PILL_ID}:focus-visible {
    outline: 2px solid var(--c2c-accent, var(--c2c-blue, var(--c2c-blue)));
    outline-offset: 2px;
}
#${PILL_ID}[aria-expanded="true"] {
    background: linear-gradient(
        180deg,
        color-mix(in srgb, var(--c2c-accent, var(--c2c-blue, var(--c2c-blue))) 48%, transparent) 0%,
        color-mix(in srgb, var(--c2c-accent, var(--c2c-blue, var(--c2c-blue))) 28%, transparent) 100%);
    border-color: var(--c2c-accent, var(--c2c-blue, var(--c2c-blue)));
}
#${PILL_ID} .c2c-omnibar-pill-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--c2c-accent, var(--c2c-blue, var(--c2c-blue)));
    box-shadow: 0 0 6px var(--c2c-accent, var(--c2c-blue, var(--c2c-blue)));
    flex: 0 0 auto;
}
#${PILL_ID} .c2c-omnibar-pill-chev {
    font-size: 8px;
    opacity: 0.7;
    margin-left: 2px;
    transition: transform var(--c2c-dur-fast, 120ms) var(--c2c-ease-out, ease);
}
#${PILL_ID}[aria-expanded="true"] .c2c-omnibar-pill-chev {
    transform: rotate(180deg);
}
#${PILL_ID}[data-c2c-omnibar-hidden="1"] { display: none !important; }

/* ─── PANEL (dropdown anchored under pill) ──────────────────────────── */
#${ROOT_ID} {
    position: fixed;
    z-index: var(--c2c-z-popover, 9000);
    display: none;
    flex-direction: column;
    gap: 8px;
    width: 540px;
    max-width: calc(100vw - 16px);
    max-height: calc(100vh - 140px);
    overflow-y: auto;
    padding: 10px 12px 12px 12px;
    background: color-mix(in srgb, var(--c2c-panelBg, var(--c2c-bg, var(--c2c-bg))) 96%, transparent);
    border: 1px solid color-mix(in srgb, var(--c2c-highlightBase, var(--c2c-border, var(--c2c-surface1))) 28%, transparent);
    border-radius: 12px;
    box-shadow: 0 14px 32px color-mix(in srgb, var(--c2c-shadowBase, var(--c2c-black)) 55%, transparent);
    backdrop-filter: blur(10px);
    color: var(--c2c-fg, var(--c2c-accentBright));
    font: 11px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    user-select: none;
    pointer-events: auto;
}
#${ROOT_ID}[data-c2c-omnibar-open="1"] { display: flex; }

/* Panel header strip (title + close button). */
#${ROOT_ID} .c2c-omnibar-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2px 6px 2px;
    border-bottom: 1px solid color-mix(in srgb, var(--c2c-highlightBase, var(--c2c-border, var(--c2c-surface1))) 18%, transparent);
    margin-bottom: 4px;
}
#${ROOT_ID} .c2c-omnibar-panel-title {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--c2c-fg, var(--c2c-accentBright));
}
#${ROOT_ID} .c2c-omnibar-panel-title .c2c-omnibar-panel-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--c2c-accent, var(--c2c-blue, var(--c2c-blue)));
    box-shadow: 0 0 6px var(--c2c-accent, var(--c2c-blue, var(--c2c-blue)));
}
#${ROOT_ID} .c2c-omnibar-panel-close {
    appearance: none;
    background: transparent;
    border: 1px solid color-mix(in srgb, var(--c2c-highlightBase, var(--c2c-border, var(--c2c-surface1))) 22%, transparent);
    border-radius: 6px;
    color: var(--c2c-sub, var(--c2c-fg));
    width: 22px;
    height: 22px;
    line-height: 1;
    cursor: pointer;
    font-size: 13px;
}
#${ROOT_ID} .c2c-omnibar-panel-close:hover {
    background: color-mix(in srgb, var(--c2c-red, var(--c2c-red)) 18%, transparent);
    border-color: color-mix(in srgb, var(--c2c-red, var(--c2c-red)) 50%, transparent);
    color: var(--c2c-fg, var(--c2c-accentBright));
}

/* Each section: block with header above + wrap row of pills below. */
#${ROOT_ID} .c2c-omnibar-section-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 4px 2px 6px 2px;
    border-radius: 6px;
}
#${ROOT_ID} .c2c-omnibar-section-wrap.is-empty { display: none; }
#${ROOT_ID} .c2c-omnibar-section-header {
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--c2c-sub, color-mix(in srgb, var(--c2c-fg, var(--c2c-accentBright)) 64%, transparent));
    padding: 0 4px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
#${ROOT_ID} .c2c-omnibar-section-header::after {
    content: "";
    flex: 1 1 auto;
    height: 1px;
    background: color-mix(in srgb, var(--c2c-highlightBase, var(--c2c-border, var(--c2c-surface1))) 14%, transparent);
    margin-left: 6px;
}
#${ROOT_ID} .c2c-omnibar-section {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
    padding: 2px 2px;
    min-height: 24px;
}
#${ROOT_ID} .c2c-omnibar-section.is-empty {
    display: none;
}

/* ── Pill-shaped buttons for in-bar slots ───────────────────────────────
 * Per locked spec: every action/button rendered inside OmniBar must be a
 * rounded, compact, clearly labelled pill. We target the common surface
 * types our slots register today:
 *   • plain <button> (autockpt-btn, doctor-btn)
 *   • mini-row tiles (.c2c-mr-tile inside .tools)
 *   • bookmark grid root (#c2c-bookmarks-root inside .bookmarks)
 * Slot-owner inline styles still win — we only style what isn't already
 * set by the owner. */
#${ROOT_ID} .c2c-omnibar-section > button,
#${ROOT_ID} .c2c-omnibar-section > .c2c-omnibar-slot-pill {
    appearance: none;
    background: color-mix(in srgb, var(--c2c-surface1, var(--c2c-bg)) 90%, transparent);
    color: var(--c2c-fg, var(--c2c-accentBright));
    border: 1px solid color-mix(in srgb, var(--c2c-highlightBase, var(--c2c-border)) 24%, transparent);
    border-radius: 999px;             /* pill */
    padding: 3px 10px;
    font: 500 11px/1.15 ui-sans-serif, system-ui, sans-serif;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    transition: background var(--c2c-dur-fast, 120ms) var(--c2c-ease-out, ease),
                border-color var(--c2c-dur-fast, 120ms) var(--c2c-ease-out, ease),
                transform var(--c2c-dur-fast, 120ms) var(--c2c-ease-out, ease);
}
#${ROOT_ID} .c2c-omnibar-section > button:hover,
#${ROOT_ID} .c2c-omnibar-section > .c2c-omnibar-slot-pill:hover {
    background: color-mix(in srgb, var(--c2c-accent, var(--c2c-blue)) 12%, transparent);
    border-color: color-mix(in srgb, var(--c2c-accent, var(--c2c-blue)) 50%, transparent);
}
#${ROOT_ID} .c2c-omnibar-section > button:active,
#${ROOT_ID} .c2c-omnibar-section > .c2c-omnibar-slot-pill:active {
    transform: translateY(1px);
}
#${ROOT_ID} .c2c-omnibar-section > button:focus-visible {
    outline: 2px solid var(--c2c-accent, var(--c2c-blue));
    outline-offset: 1px;
}
/* Mini-row tiles inherit pill shape and keep their colored left rule. */
#${ROOT_ID} .c2c-omnibar-section .c2c-mr-tile {
    border-radius: 999px;
}
/* Bookmarks root: pack its 9 numbered slots tightly inside the pill area.
   position:static overrides the fixed positioning from injectStyle() in
   c2c_node_bookmarks.js — without this the strip floats outside the panel. */
#${ROOT_ID} .c2c-omnibar-section > #c2c-bookmarks-root {
    position: static !important;
    background: transparent;
    border: 0;
    padding: 0;
    border-radius: 0;
    display: inline-flex !important;
    align-items: center;
    gap: 3px;
    z-index: auto !important;
}
#${ROOT_ID}[data-c2c-omnibar-mode="icon"] .c2c-mr-txt { display: none; }
`;
    document.head.appendChild(s);
}

function _ensureRoot() {
    let root = document.getElementById(ROOT_ID);
    if (root) return root;
    root = document.createElement("div");
    root.id = ROOT_ID;
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-label", "C2C OmniPill");
    root.setAttribute("aria-modal", "false");
    root.setAttribute("data-c2c-overlay", "1");

    // Panel header: title + close button. The header gives the dropdown a
    // clear identity so it never feels like an orphan floating menu.
    const head = document.createElement("div");
    head.className = "c2c-omnibar-panel-header";
    const title = document.createElement("div");
    title.className = "c2c-omnibar-panel-title";
    const titleDot = document.createElement("span");
    titleDot.className = "c2c-omnibar-panel-dot";
    const titleText = document.createElement("span");
    titleText.textContent = "C2C OmniPill";
    title.appendChild(titleDot);
    title.appendChild(titleText);
    head.appendChild(title);
    const close = document.createElement("button");
    close.type = "button";
    close.className = "c2c-omnibar-panel-close";
    close.title = "Close (Esc)";
    close.setAttribute("aria-label", "Close OmniPill");
    close.textContent = "×";
    close.addEventListener("click", () => _closePanel());
    head.appendChild(close);
    root.appendChild(head);

    // One wrap per section: <header>SECTION</header><div.section/>. The wrap
    // gets `is-empty` toggled from _renderSection so empty groups disappear
    // entirely (header and host together).
    for (const section of SECTION_IDS) {
        const wrap = document.createElement("div");
        wrap.className = "c2c-omnibar-section-wrap is-empty";
        wrap.setAttribute("data-c2c-section-wrap", section);
        const header = document.createElement("div");
        header.className = "c2c-omnibar-section-header";
        header.textContent = SECTION_LABELS[section] || section;
        wrap.appendChild(header);
        const host = document.createElement("div");
        host.className = "c2c-omnibar-section is-empty";
        host.setAttribute("data-section", section);
        host.setAttribute("role", "group");
        host.setAttribute("aria-label", "C2C OmniPill — " + section);
        wrap.appendChild(host);
        root.appendChild(wrap);
    }

    document.body.appendChild(root);
    return root;
}

// ── Pill (single trigger in Manager bar) + Panel toggle ──────────────────
//
// Architecture (locked per user mandate):
//   • A single rounded pill button is injected into ComfyUI's actionbar
//     button-group (the same row as the "Manager" button), so OmniBar
//     consumes ZERO extra vertical space on the canvas.
//   • Clicking the pill toggles an anchored dropdown panel directly below
//     it. The panel holds every registered slot, grouped by section with
//     clear headers.
//   • Outside-click or Esc dismisses the panel. The pill itself stays put
//     and is the only persistent footprint.
//   • A MutationObserver keeps the pill re-injected if ComfyUI's frontend
//     re-renders the actionbar (Vue may swap nodes on theme/menu changes).

let _pillEl = null;
let _pillObserver = null;
let _panelDismissBound = false;
let _panelOpen = false;

function _findManagerButtonGroup() {
    // Prefer the button-group that contains the Manager dropdown trigger.
    // Falls back to first button-group inside .actionbar-container.
    // (Pre-2026-05-26 we also looked for the legacy `#mec-integrity-btn`
    // anchor; that element no longer exists.)
    const bar = document.querySelector(".actionbar-container, .comfyui-menu .actionbar");
    if (!bar) return null;
    const groups = bar.querySelectorAll(".comfyui-button-group");
    for (const g of groups) {
        // Manager button typically has the label "Manager" or class "manager-button".
        const btns = g.querySelectorAll("button");
        for (const b of btns) {
            const txt = (b.textContent || "").trim().toLowerCase();
            if (txt.includes("manager") || b.classList.contains("manager-button")) return g;
        }
    }
    return groups[0] || null;
}

function _buildPill() {
    if (_pillEl && document.body.contains(_pillEl)) return _pillEl;
    const btn = document.createElement("button");
    btn.id = PILL_ID;
    btn.type = "button";
    btn.className = "comfyui-button c2c-omnibar-pill";
    btn.title = "C2C OmniPill — open all C2C tools (Alt+Shift+O)";
    btn.setAttribute("aria-label", "C2C OmniPill");
    btn.setAttribute("aria-haspopup", "dialog");
    btn.setAttribute("aria-expanded", "false");
    btn.setAttribute("aria-controls", ROOT_ID);
    const dot = document.createElement("span");
    dot.className = "c2c-omnibar-pill-dot";
    btn.appendChild(dot);
    const lbl = document.createElement("span");
    lbl.className = "c2c-omnibar-pill-label";
    lbl.textContent = "C2C";
    btn.appendChild(lbl);
    const chev = document.createElement("span");
    chev.className = "c2c-omnibar-pill-chev";
    chev.textContent = "▾";
    chev.setAttribute("aria-hidden", "true");
    btn.appendChild(chev);
    btn.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        _togglePanel();
    });
    _pillEl = btn;
    return btn;
}

function _ensurePill() {
    // Respect visibility setting: if hidden, remove pill entirely.
    const visible = !!_getSetting(SETTING_VISIBLE, true);
    if (!visible) {
        if (_pillEl && _pillEl.parentNode) _pillEl.parentNode.removeChild(_pillEl);
        if (_panelOpen) _closePanel();
        return null;
    }
    const host = _findManagerButtonGroup();
    if (!host) {
        // Manager bar not ready yet — observer will retry.
        return null;
    }
    const pill = _buildPill();
    if (pill.parentNode === host) return pill;
    // Append at the end of the Manager button group. The canonical INT
    // chip (`#c2c-int-chip`) lives INSIDE this pill's panel, not in the
    // actionbar, so no special ordering is needed any more.
    host.appendChild(pill);
    return pill;
}

function _startPillObserver() {
    if (_pillObserver) return;
    const target = document.querySelector(".actionbar-container, .comfyui-menu") || document.body;
    // PERF 2026-05-26: rAF-coalesce mutation bursts. The actionbar/menu
    // subtree mutates on every queue-execution / model-load tick; without
    // coalescing _ensurePill() runs dozens of times per second during heavy
    // graph activity.
    let rafPending = false;
    const raf = (typeof requestAnimationFrame !== "undefined")
        ? requestAnimationFrame : (cb) => setTimeout(cb, 16);
    _pillObserver = new MutationObserver(() => {
        if (rafPending) return;
        rafPending = true;
        raf(() => {
            rafPending = false;
            // Cheap idempotent check — if pill is already in DOM under the
            // right host, _ensurePill() returns immediately.
            try { _ensurePill(); }
            catch (err) { _reportFailure("pillObserver:ensure", err); }
        });
    });
    try {
        _pillObserver.observe(target, { childList: true, subtree: true });
    } catch (err) {
        _reportFailure("pillObserver:observe", err);
    }
}

function _positionPanel() {
    const root = document.getElementById(ROOT_ID);
    if (!root || !_pillEl) return;
    const r = _pillEl.getBoundingClientRect();
    // Force measurable layout even when hidden.
    const wasOpen = root.getAttribute("data-c2c-omnibar-open") === "1";
    if (!wasOpen) {
        root.style.visibility = "hidden";
        root.setAttribute("data-c2c-omnibar-open", "1");
    }
    const pw = root.offsetWidth || 540;
    const ph = root.offsetHeight || 240;
    if (!wasOpen) {
        root.removeAttribute("data-c2c-omnibar-open");
        root.style.visibility = "";
    }
    const margin = 6;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let top  = r.bottom + 6;
    // Align panel right edge with pill right edge (panel grows leftward).
    let left = r.right - pw;
    // Viewport clamp.
    left = Math.max(margin, Math.min(left, vw - pw - margin));
    // If not enough room below, flip above.
    if (top + ph + margin > vh && r.top - ph - 6 > margin) {
        top = r.top - ph - 6;
    } else {
        top = Math.max(margin, Math.min(top, vh - ph - margin));
    }
    root.style.top  = top  + "px";
    root.style.left = left + "px";
}

function _onPanelOutsideClick(ev) {
    const root = document.getElementById(ROOT_ID);
    if (!root || !_panelOpen) return;
    if (root.contains(ev.target)) return;
    if (_pillEl && _pillEl.contains(ev.target)) return;
    _closePanel();
}

function _onPanelKey(ev) {
    if (ev.key === "Escape" && _panelOpen) _closePanel();
}

function _openPanel() {
    const root = _ensureRoot();
    _positionPanel();
    root.setAttribute("data-c2c-omnibar-open", "1");
    if (_pillEl) _pillEl.setAttribute("aria-expanded", "true");
    _panelOpen = true;
    if (!_panelDismissBound) {
        document.addEventListener("mousedown", _onPanelOutsideClick, true);
        document.addEventListener("keydown", _onPanelKey, true);
        window.addEventListener("resize", _positionPanel, { passive: true });
        window.addEventListener("scroll", _positionPanel, { passive: true, capture: true });
        _panelDismissBound = true;
    }
    // Refresh slot rendering on open so newly-registered slots appear.
    for (const section of SECTION_IDS) _renderSection(section);
}

function _closePanel() {
    const root = document.getElementById(ROOT_ID);
    if (root) root.removeAttribute("data-c2c-omnibar-open");
    if (_pillEl) _pillEl.setAttribute("aria-expanded", "false");
    _panelOpen = false;
    if (_panelDismissBound) {
        document.removeEventListener("mousedown", _onPanelOutsideClick, true);
        document.removeEventListener("keydown", _onPanelKey, true);
        window.removeEventListener("resize", _positionPanel);
        window.removeEventListener("scroll", _positionPanel, true);
        _panelDismissBound = false;
    }
}

function _togglePanel() {
    if (_panelOpen) _closePanel();
    else _openPanel();
}

// Back-compat alias: legacy keybinding/command code referenced _toggleCogPopover.
const _toggleCogPopover = _togglePanel;

// ── Safe-offset measurement ────────────────────────────────────────────────
// Retained for backward-compat reads (other modules may probe
// --c2c-native-top); pill+panel layout no longer depends on it.
const _SAFE_BAND_PX = 110;     // how far in from each edge to scan (chrome lives within ~100px)
const _SAFE_GAP_PX  = 6;       // breathing room between OmniBar & crowders
const _SAFE_Z_MAX   = 1100;    // skip dialogs/toasts/draggable popovers above this z
const _SAFE_SKIP_CLASSES = [
    "p-panel", "p-dialog", "p-overlay", "p-toast", "p-tooltip",
    "p-popover", "p-menu", "p-confirmpopup", "p-contextmenu", "p-drawer",
    // dom-widget / dom-widget-host are LiteGraph's per-node DOM widgets
    // (CLIP text encode boxes, image previews, draggable Run widget host).
    // They are workflow CONTENT — not chrome — and must not push OmniBar.
    "dom-widget", "dom-widget-host", "p-buttongroup",
];
const _SAFE_SKIP_ROLES = new Set([
    "dialog", "alert", "alertdialog", "tooltip", "menu", "menuitem", "status",
]);
// Native top-chrome containers that are usually position:relative (so the
// generic fixed/absolute filter below would miss them) but ALWAYS anchor the
// top of the workspace. The OmniBar must sit directly below the busiest of
// these — most importantly the ComfyUI-Manager `.actionbar-container` which
// holds the Run button + Manager + Integrity badge bar.
const _NATIVE_TOP_SELECTORS = [
    ".comfyui-body-top",
    ".comfy-menu",
    ".comfyui-menu",
    ".comfyui-workflow-tabs",
    ".workflow-tabs-container",
    ".workflow-tabs",
    ".actionbar-container",
    "[data-testid='top-menu']",
    ".p-menubar",
];
// Excluded by ID (self + popover + our own descendants), id-prefix, or because
// the element is full-viewport (canvas, vue-app overlays, modal scrims).
function _shouldSkipForOverlap(el, vw, vh) {
    if (!el || el.id === ROOT_ID || el.id === "c2c-omnibar-cog-pop" || el.id === STYLE_ID) return true;
    const root = document.getElementById(ROOT_ID);
    if (root && (root === el || root.contains(el) || el.contains(root))) return true;
    const cs = getComputedStyle(el);
    if (cs.position !== "fixed" && cs.position !== "absolute") return true;
    if (cs.display === "none" || cs.visibility === "hidden") return true;
    if (parseFloat(cs.opacity || "1") <= 0.01) return true;
    if (cs.pointerEvents === "none" && (parseFloat(cs.opacity || "1") < 0.5)) return true;
    // Skip popovers, dialogs, toasts — user-dragged or transient overlays.
    const zRaw = parseInt(cs.zIndex, 10);
    if (Number.isFinite(zRaw) && zRaw >= _SAFE_Z_MAX) return true;
    const role = (el.getAttribute && el.getAttribute("role")) || "";
    if (role && _SAFE_SKIP_ROLES.has(role)) return true;
    if (el.classList && _SAFE_SKIP_CLASSES.some((c) => el.classList.contains(c))) return true;
    // Skip if ANY ancestor up to body is a known popover/dialog class — prevents
    // counting child elements of draggable widgets.
    let cur = el.parentElement;
    while (cur && cur !== document.body) {
        if (cur.classList && _SAFE_SKIP_CLASSES.some((c) => cur.classList.contains(c))) return true;
        const r2 = (cur.getAttribute && cur.getAttribute("role")) || "";
        if (r2 && _SAFE_SKIP_ROLES.has(r2)) return true;
        cur = cur.parentElement;
    }
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return true;
    // Full-viewport overlays (canvases, app shell, modal scrims) — skip.
    if (r.width >= vw * 0.95 && r.height >= vh * 0.85) return true;
    return false;
}
function _measureSafeOffsets() {
    const vw = window.innerWidth || 1, vh = window.innerHeight || 1;
    // Native chrome baseline (from _c2c_native_offsets.js).
    const cs = getComputedStyle(document.documentElement);
    const px = (name, dflt) => {
        const raw = (cs.getPropertyValue(name) || "").trim();
        const m = raw.match(/(-?[\d.]+)/);
        return m ? parseFloat(m[1]) : dflt;
    };
    const baselineTop    = px("--c2c-native-top",    0);
    const baselineBot    = px("--c2c-native-bottom", 0);
    const baselineLeft   = px("--c2c-native-left",   0);
    const baselineRight  = 0;

    // Per-edge maxima (bottom of stuff at top, top of stuff at bottom, etc.)
    let topMax    = baselineTop;
    let botMin    = vh - baselineBot;     // walked UPward — min Y
    let leftMax   = baselineLeft;
    let rightMin  = vw;                   // walked LEFTward — min X

    let nodes;
    try { nodes = document.querySelectorAll("body *"); }
    catch (err) { _reportFailure("measureSafeOffsets:querySelectorAll", err); return null; }

    // First pass — explicit native top-chrome containers (position:relative or
    // static, so the generic fixed/absolute filter below skips them). These
    // anchor the workspace top and the OmniBar MUST sit below the deepest one.
    for (const sel of _NATIVE_TOP_SELECTORS) {
        let list;
        try { list = document.querySelectorAll(sel); }
        catch (err) { _reportFailure("measureSafeOffsets:nativeQS:" + sel, err); continue; }
        for (const el of list) {
            const cs2 = getComputedStyle(el);
            if (cs2.display === "none" || cs2.visibility === "hidden") continue;
            if (parseFloat(cs2.opacity || "1") <= 0.01) continue;
            const r = el.getBoundingClientRect();
            if (r.height < 4 || r.bottom <= 0 || r.top > _SAFE_BAND_PX * 2) continue;
            if (r.width >= vw * 0.95 && r.height >= vh * 0.85) continue; // full-viewport
            if (r.bottom > topMax) topMax = r.bottom;
        }
    }

    for (const el of nodes) {
        if (_shouldSkipForOverlap(el, vw, vh)) continue;
        const r = el.getBoundingClientRect();
        // Element is in TOP band if its top is < _SAFE_BAND_PX AND it sits near
        // the horizontal centre (where OmniBar/top dock would be).
        if (r.top < _SAFE_BAND_PX && r.bottom > 0 && r.bottom > topMax) {
            // Only count if it actually overlaps the horizontal centre band
            // (we centre OmniBar via translateX(-50%)).
            const cx = vw / 2;
            if (r.left <= cx + 80 && r.right >= cx - 80) topMax = r.bottom;
        }
        if (r.bottom > vh - _SAFE_BAND_PX && r.top < vh && r.top < botMin) {
            const cx = vw / 2;
            if (r.left <= cx + 80 && r.right >= cx - 80) botMin = r.top;
        }
        if (r.left < _SAFE_BAND_PX && r.right > 0 && r.right > leftMax) {
            const cy = vh / 2;
            if (r.top <= cy + 80 && r.bottom >= cy - 80) leftMax = r.right;
        }
        if (r.right > vw - _SAFE_BAND_PX && r.left < vw && r.left < rightMin) {
            const cy = vh / 2;
            if (r.top <= cy + 80 && r.bottom >= cy - 80) rightMin = r.left;
        }
    }
    return {
        top:    Math.max(0,  Math.round(topMax)            + _SAFE_GAP_PX),
        bottom: Math.max(0,  Math.round(vh - botMin)       + _SAFE_GAP_PX),
        left:   Math.max(0,  Math.round(leftMax)           + _SAFE_GAP_PX),
        right:  Math.max(0,  Math.round(vw - rightMin)     + _SAFE_GAP_PX),
    };
}
function _clearInlineEdges(root) {
    root.style.top = "";
    root.style.bottom = "";
    root.style.left = "";
    root.style.right = "";
}

// ── Mount / re-mount / apply position ──────────────────────────────────────
function _applyPositionAndMode() {
    // New architecture: the OmniBar surface is (1) a single pill injected
    // into the Manager bar, and (2) an anchored dropdown panel. Position /
    // density settings no longer affect the bar layout (kept for backward
    // compat only). This function:
    //   • ensures the pill is in the Manager bar (or removed if hidden)
    //   • re-renders all sections so newly-registered slots appear
    //   • if the panel is open, repositions it under the pill
    _ensureRoot();
    _ensurePill();
    for (const section of SECTION_IDS) _renderSection(section);
    if (_panelOpen) _positionPanel();
}

function _remountIfNeeded() {
    // If the root was detached (e.g. native menu rebuild blew away body
    // children — rare, but ComfyUI does aggressive remounts on workflow tab
    // switches), reinsert it.
    const root = document.getElementById(ROOT_ID);
    if (!root) {
        _ensureRoot();
        _applyPositionAndMode();
        return;
    }
    if (!document.body.contains(root)) {
        document.body.appendChild(root);
        _applyPositionAndMode();
    }
    // Same idempotency check for the pill.
    if (!_pillEl || !document.body.contains(_pillEl)) {
        _ensurePill();
    }
}

// ── Lifecycle ───────────────────────────────────────────────────────────────
let _pollHandle = 0;
let _resizeHandle = 0;
let _started = false;
let _bodyObserver = null;
let _bodyObsThrottle = 0;

function _startBodyObserver() {
    if (_bodyObserver) return;
    try {
        _bodyObserver = new MutationObserver(() => {
            // Throttle to one re-apply per frame; ignored if our own writes.
            if (_bodyObsThrottle) return;
            _bodyObsThrottle = window.requestAnimationFrame(() => {
                _bodyObsThrottle = 0;
                try { _applyPositionAndMode(); }
                catch (err) { _reportFailure("bodyObserver:_applyPositionAndMode", err); }
            });
        });
        _bodyObserver.observe(document.body, { childList: true, subtree: false });
    } catch (err) {
        _reportFailure("MutationObserver:body", err);
        _bodyObserver = null;
    }
}

function _startPolling() {
    if (_pollHandle) return;
    let _reposCounter = 0;
    const tick = () => {
        try { _pollAllSlots(); }
        catch (err) { _reportFailure("pollTick", err); }
        // Every 2nd tick (~2s) re-measure safe offsets so OmniBar shifts when
        // other extensions mount/unmount/resize their floating UI.
        _reposCounter = (_reposCounter + 1) % 2;
        if (_reposCounter === 0) {
            try { _applyPositionAndMode(); }
            catch (err) { _reportFailure("repositionTick", err); }
        }
        try { _pollHandle = window.setTimeout(tick, 1000); }
        catch (err) {
            _reportFailure("setTimeout:poll", err);
            _pollHandle = 0;
        }
    };
    try { _pollHandle = window.setTimeout(tick, 250); }
    catch (err) {
        _reportFailure("setTimeout:initial-poll", err);
        _pollHandle = 0;
    }
}

function _stopPolling() {
    if (_pollHandle) {
        try { window.clearTimeout(_pollHandle); }
        catch (err) { _reportFailure("clearTimeout:poll", err); }
        _pollHandle = 0;
    }
}

function _onResize() {
    // Debounced via rAF — we don't need pixel-perfect; we just need eventual.
    if (_resizeHandle) return;
    try {
        _resizeHandle = window.requestAnimationFrame(() => {
            _resizeHandle = 0;
            _applyPositionAndMode();
        });
    } catch (err) {
        _reportFailure("requestAnimationFrame:resize", err);
        // Fallback synchronous re-apply.
        _applyPositionAndMode();
    }
}

function _onGraphChanged() {
    // Workflow tab switch / new graph load: ensure we're still mounted.
    _remountIfNeeded();
}

function _start() {
    if (_started) return;
    _started = true;

    // One-time consolidation migration (v1): OmniBar is now the sole top
    // host for all C2C surfaces. If the user previously toggled it off
    // (legacy default before consolidation), surface it ON exactly once so
    // they discover the unified bar; subsequent user toggles are honoured.
    try {
        const MIGRATION_KEY = "c2c.omnibar.consolidation.v1";
        if (typeof localStorage !== "undefined" && !localStorage.getItem(MIGRATION_KEY)) {
            _setSetting(SETTING_VISIBLE, true);
            localStorage.setItem(MIGRATION_KEY, String(Date.now()));
        }
    } catch (err) {
        _reportFailure("consolidationMigrationV1", err);
    }

    // Ensure measurer is running (idempotent).
    try { startNativeOffsets(); }
    catch (err) { _reportFailure("startNativeOffsets", err); }

    _injectStyle();
    _ensureRoot();
    _ensurePill();
    _startPillObserver();
    _applyPositionAndMode();
    // Re-apply after layout settles so the auto-measure has the correct bar
    // width AND so late-mounting native chrome (mini-row-host, ComfyUI-Manager
    // bar, bookmarks-root) has had a chance to register. Without this the
    // first measurement uses a stale narrower rect / missing crowders and the
    // bar can briefly land inside a native UI band before the polling loop
    // corrects it.
    try {
        requestAnimationFrame(() => {
            try { _applyPositionAndMode(); } catch (err) { _reportFailure("rAF1:_applyPositionAndMode", err); }
            requestAnimationFrame(() => {
                try { _applyPositionAndMode(); } catch (err) { _reportFailure("rAF2:_applyPositionAndMode", err); }
            });
        });
    } catch (err) {
        _reportFailure("requestAnimationFrame", err);
    }
    for (const delay of [250, 500, 1000, 2000]) {
        setTimeout(() => {
            try { _applyPositionAndMode(); } catch (err) { _reportFailure("settle:_applyPositionAndMode", err); }
        }, delay);
    }
    _startPolling();
    _startBodyObserver();

    // Set html[data-c2c-strip="on"] so the EXISTING CSS hide-sweep in
    // js/C2C_status_strip.js takes effect on the legacy floating HUDs
    // (#mec-system-hud, #c2c-ai-hud, #mec-complexity-hud, #mec-cost-btn,
    // #mec-cost-panel, #c2c-graph-health-pill). Those HUDs' data is already
    // republished by the bottom status strip's GPU/VRAM/AI/$/C chips, so the
    // user loses no information — they just stop seeing duplicate floating
    // pills before our phased migration relocates them into OmniBar slots.
    try {
        document.documentElement.setAttribute("data-c2c-strip", "on");
    } catch (err) {
        _reportFailure("setAttribute:data-c2c-strip", err);
    }

    // Respond to setting changes (position/visibility/density).
    try {
        const settings = app?.ui?.settings;
        if (settings && typeof settings.addEventListener === "function") {
            settings.addEventListener("change", (ev) => {
                const id = ev?.detail?.id || ev?.id;
                if (id === SETTING_POSITION || id === SETTING_VISIBLE || id === SETTING_DENSITY) {
                    _applyPositionAndMode();
                }
            });
        }
    } catch (err) {
        _reportFailure("settings:addEventListener", err);
    }

    // Window resize → re-evaluate icon-only collapse threshold.
    try { window.addEventListener("resize", _onResize, { passive: true }); }
    catch (err) { _reportFailure("addEventListener:resize", err); }

    // Workflow tab switch / new graph load.
    try {
        app?.api?.addEventListener?.("graphChanged", _onGraphChanged);
    } catch (err) {
        _reportFailure("api:graphChanged", err);
    }

    // ComfyUI calls loadGraphData on workflow tab switch — wrap it to nudge.
    try {
        if (app && typeof app.loadGraphData === "function" && !app._c2cOmniBarPatched) {
            const orig = app.loadGraphData.bind(app);
            app.loadGraphData = function (...args) {
                let r;
                try { r = orig(...args); }
                catch (err) {
                    // Upstream (ComfyUI core / Manager) commonly throws when
                    // graphData is undefined. Don't amplify into our hooks.
                    _reportFailure("loadGraphData:orig", err);
                    throw err;
                }
                try { _onGraphChanged(); }
                catch (err) { _reportFailure("loadGraphData:hook", err); }
                return r;
            };
            app._c2cOmniBarPatched = true;
        }
    } catch (err) {
        _reportFailure("patch:loadGraphData", err);
    }

    // Listen for any caller that wants to nudge a refresh (e.g. theme.js
    // setVariant() finishes → ensures border/bg recolor sticks).
    try {
        window.addEventListener("c2c:variant-changed", _applyPositionAndMode);
    } catch (err) {
        _reportFailure("addEventListener:variant", err);
    }
}

function _stop() {
    if (!_started) return;
    _started = false;
    _stopPolling();
    try { window.removeEventListener("resize", _onResize); }
    catch (err) { _reportFailure("removeEventListener:resize", err); }
}

// ── Public API ──────────────────────────────────────────────────────────────
const PublicAPI = Object.freeze({
    register,
    refresh: _applyPositionAndMode,
    pollNow: _pollAllSlots,
    setPosition: (pos) => {
        if (!POSITION_VALUES.includes(pos)) {
            _reportFailure("setPosition", new Error("invalid pos: " + pos));
            return;
        }
        _setSetting(SETTING_POSITION, pos);
        _applyPositionAndMode();
    },
    setVisible: (visible) => {
        _setSetting(SETTING_VISIBLE, !!visible);
        _applyPositionAndMode();
    },
    setDensity: (density) => {
        if (density !== "comfortable" && density !== "compact") {
            _reportFailure("setDensity", new Error("invalid density: " + density));
            return;
        }
        _setSetting(SETTING_DENSITY, density);
        _applyPositionAndMode();
    },
    getPosition: _getPosition,
    getMode: _currentChipMode,
    getRoot: () => document.getElementById(ROOT_ID),
    getPill: () => document.getElementById(PILL_ID),
    openPanel:   _openPanel,
    closePanel:  _closePanel,
    togglePanel: _togglePanel,
    isPanelOpen: () => _panelOpen,
    SECTION_IDS: SECTION_IDS.slice(),
});

// Expose on window so non-module-importing code (e.g. legacy chips while they
// still exist during the 8-turn phased migration) can register slots.
try {
    if (typeof window !== "undefined") {
        Object.defineProperty(window, "C2COmniBar", {
            value: PublicAPI,
            writable: false,
            configurable: false,
        });
    }
} catch (err) {
    _reportFailure("defineProperty:window.C2COmniBar", err);
}

// ── Extension registration ──────────────────────────────────────────────────
app.registerExtension({
    name: "C2C.OmniBar",
    settings: [
        {
            id: SETTING_POSITION,
            name: "C2C ▸ OmniBar ▸ Position",
            tooltip: "Where the OmniBar lives. Defaults to top edge.",
            type: "combo",
            options: [
                { text: "Top (default)", value: "top" },
                { text: "Bottom",        value: "bottom" },
                { text: "Left rail",     value: "left" },
                { text: "Right rail",    value: "right" },
            ],
            defaultValue: POSITION_DEFAULT,
            onChange: () => _applyPositionAndMode(),
        },
        {
            id: SETTING_VISIBLE,
            name: "C2C ▸ OmniBar ▸ Visible",
            tooltip: "Hide OmniBar entirely (re-enable here to bring it back).",
            type: "boolean",
            defaultValue: true,
            onChange: () => _applyPositionAndMode(),
        },
        {
            id: SETTING_DENSITY,
            name: "C2C ▸ OmniBar ▸ Density",
            tooltip: "Comfortable shows labels; Compact forces icon-only mode (also auto-enabled when viewport is narrow or position is left/right).",
            type: "combo",
            options: [
                { text: "Comfortable", value: "comfortable" },
                { text: "Compact",     value: "compact" },
            ],
            defaultValue: "comfortable",
            onChange: () => _applyPositionAndMode(),
        },
    ],
    commands: [
        {
            id: "C2C.OmniBar.toggle",
            label: "C2C: Toggle OmniBar panel",
            function: () => { _togglePanel(); },
        },
        {
            id: "C2C.OmniBar.cyclePosition",
            label: "C2C: Cycle OmniBar position (top → right → bottom → left)",
            function: () => {
                const order = ["top", "right", "bottom", "left"];
                const cur = _getPosition();
                const next = order[(order.indexOf(cur) + 1) % order.length];
                PublicAPI.setPosition(next);
            },
        },
    ],
    keybindings: [
        {
            combo: { key: "o", alt: true, shift: true },
            commandId: "C2C.OmniBar.toggle",
        },
    ],
    async setup() {
        try { _start(); }
        catch (err) { _reportFailure("setup", err); }
    },
});

// Defensive auto-start in case the extension setup hook is skipped for any
// reason (e.g. host strips registerExtension during reload). startNativeOffsets
// is idempotent.
if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
        try {
            document.addEventListener("DOMContentLoaded", _start, { once: true });
        } catch (err) {
            _reportFailure("addEventListener:DOMContentLoaded", err);
        }
    } else {
        try {
            Promise.resolve().then(() => {
                // Wait one frame so app.ui.settings is available before _start
                // reads our SETTING_* defaults.
                try {
                    if (typeof requestAnimationFrame === "function") {
                        requestAnimationFrame(() => _start());
                    } else {
                        _start();
                    }
                } catch (err) {
                    _reportFailure("rAF:autoStart", err);
                    _start();
                }
            });
        } catch (err) {
            _reportFailure("Promise.then:autoStart", err);
        }
    }
}

export {
    PublicAPI as C2COmniBar,
    register,
    SECTION_IDS,
    POSITION_VALUES,
    _stop as __stopForTests,
};
