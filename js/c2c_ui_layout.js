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

const TARGETS = [
    { btnId: "mec-wizard-btn",        icon: "🧙", label: "Wizard",   tip: "Step-through workflow builder", color: "#89b4fa" },
    { btnId: "mec-group-presets-btn", icon: "📚", label: "Presets",  tip: "Save & recall node-group templates", color: "#cba6f7" },
    { btnId: "mec-mood-btn",          icon: "🎨", label: "Mood",     tip: "Reference image palette", color: "#f5c2e7" },
    { btnId: "mec-surprise-btn",      icon: "🎰", label: "Surprise", tip: "Randomize seeds & queue", color: "#f9e2af" },
    { btnId: "mec-ab-btn",            icon: "⚖",  label: "A/B",      tip: "Compare last two outputs", color: "#a6e3a1" },
    { btnId: "mec-cost-btn",          icon: "💰", label: "Cost",     tip: "Estimate render cost", color: "#fab387" },
    { btnId: "mec-flamegraph-btn",    icon: "⏱",  label: "Flame",    tip: "Per-node execution timing", color: "#f38ba8" },
    { btnId: "mec-undo-btn",          icon: "↶",  label: "Undo",     tip: "Undo history viewer", color: "#74c7ec" },
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
    background: linear-gradient(180deg, #1e1e2e 0%, #181825 100%);
    border-bottom: 1px solid #313244;
    font-family: system-ui, -apple-system, sans-serif;
    user-select: none;
    z-index: 999;
}
#${ROW_HOST_ID} .c2c-mr-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6c7086;
    margin-right: 4px;
}
#${ROW_HOST_ID} .c2c-mr-tile {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 9px;
    border-radius: 5px;
    border: 1px solid #313244;
    background: #11111b;
    color: #cdd6f4;
    font-size: 11px;
    cursor: pointer;
    transition: transform 0.06s, border-color 0.12s, background 0.12s;
    line-height: 1.1;
}
#${ROW_HOST_ID} .c2c-mr-tile:hover {
    background: #181825;
}
#${ROW_HOST_ID} .c2c-mr-tile:active { transform: translateY(1px); }
#${ROW_HOST_ID} .c2c-mr-tile.c2c-missing {
    opacity: 0.35;
    cursor: not-allowed;
}
#${ROW_HOST_ID} .c2c-mr-tile.c2c-missing:hover { background: #11111b; }
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
    // ComfyUI's new (PrimeVue) top menu vs. the classic float menu — try both.
    return (
        document.querySelector(".comfyui-menu") ||
        document.querySelector("#comfyui-menu") ||
        document.querySelector(".comfy-menu") ||
        document.body
    );
}

function _mountMiniRow() {
    if (document.getElementById(ROW_HOST_ID)) return;
    _ensureRowStyle();
    const host = document.createElement("div");
    host.id = ROW_HOST_ID;

    const lbl = document.createElement("span");
    lbl.className = "c2c-mr-label";
    lbl.textContent = "C2C";
    host.appendChild(lbl);

    for (const t of TARGETS) {
        const tile = document.createElement("button");
        tile.type = "button";
        tile.className = "c2c-mr-tile";
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
            // Module may mount later; re-check after a beat.
            setTimeout(() => {
                if (!document.getElementById(t.btnId)) {
                    tile.classList.add("c2c-missing");
                    tile.title = `${t.label} — module not loaded`;
                }
            }, 2500);
        }
        host.appendChild(tile);
    }

    // Insert ABOVE the canvas, AFTER the top menu.
    const anchor = _findMenuAnchor();
    if (anchor === document.body) {
        // Fallback: pin to top of body.
        host.style.position = "fixed";
        host.style.top = "0";
        host.style.left = "0";
        host.style.right = "0";
        document.body.prepend(host);
    } else {
        // Insert immediately after the menu.
        anchor.parentNode?.insertBefore(host, anchor.nextSibling);
    }
}

function _unmountMiniRow() {
    const h = document.getElementById(ROW_HOST_ID);
    if (h) h.remove();
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
