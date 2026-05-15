/**
 * c2c_launcher.js — C2C Launcher sidebar tab.
 *
 * Consolidates the 8 right-rail floating action buttons from prior MEC
 * modules into a single sidebar grid:
 *
 *   🧙  Workflow Wizard       (mec-wizard-btn)
 *   📚  Group Presets         (mec-group-presets-btn)
 *   🎨  Mood Board            (mec-mood-btn)
 *   🎰  Surprise Me           (mec-surprise-btn)
 *   ⚖   A/B Split             (mec-ab-btn)
 *   💰  Cost Estimator        (mec-cost-btn)
 *   ⏱   Flame Graph           (mec-flamegraph-btn)
 *   ↶   Undo Panel            (mec-undo-btn)
 *
 * Strategy:
 *   - Each upstream module still mounts its floating button on document.body.
 *   - This launcher injects a CSS rule that hides those buttons when the
 *     consolidate setting is ON (default).
 *   - The launcher grid forwards click events by calling .click() on the
 *     hidden target — so each module's panel/state logic is preserved
 *     verbatim, no upstream surgery required.
 *
 * Setting:
 *   c2c.launcher.consolidate  (boolean, default true)
 *     OFF → legacy floating buttons reappear, launcher grid still works.
 *     ON  → floating buttons hidden, launcher is the only entry point.
 */

import { app } from "../../scripts/app.js";

const SIDEBAR_ID = "c2c.launcher";
const HIDE_STYLE_ID = "c2c-launcher-hide-floats";
const ROOT_STYLE_ID = "c2c-launcher-root-style";

const TARGETS = [
    { btnId: "mec-wizard-btn",        icon: "🧙", label: "Workflow Wizard", tip: "Step-through workflow builder" },
    { btnId: "mec-group-presets-btn", icon: "📚", label: "Group Presets",   tip: "Save & recall node-group templates" },
    { btnId: "mec-mood-btn",          icon: "🎨", label: "Mood Board",      tip: "Reference image palette" },
    { btnId: "mec-surprise-btn",      icon: "🎰", label: "Surprise Me",     tip: "Randomize seeds & queue" },
    { btnId: "mec-ab-btn",            icon: "⚖",  label: "A/B Split",       tip: "Compare last two outputs" },
    { btnId: "mec-cost-btn",          icon: "💰", label: "Cost Estimator",  tip: "Estimate render cost" },
    { btnId: "mec-flamegraph-btn",    icon: "⏱",  label: "Flame Graph",     tip: "Per-node execution timing" },
    { btnId: "mec-undo-btn",          icon: "↶",  label: "Undo Panel",      tip: "Undo history viewer" },
];

const FLOAT_IDS = TARGETS.map(t => "#" + t.btnId).join(",\n            ");

function _ensureHideStyle(active) {
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
    // Use !important — each module sets `display:flex` inline on its own onSetting
    // callback; we need to win against that.
    s.textContent = `
${FLOAT_IDS} {
    display: none !important;
}
    `.trim();
}

function _ensureRootStyle() {
    if (document.getElementById(ROOT_STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = ROOT_STYLE_ID;
    s.textContent = `
.c2c-launcher-root {
    padding: 12px;
    color: #cdd6f4;
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 13px;
    height: 100%;
    box-sizing: border-box;
    overflow-y: auto;
}
.c2c-launcher-header {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6c7086;
    margin: 4px 0 10px 0;
}
.c2c-launcher-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
}
.c2c-launcher-tile {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 14px 8px;
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 8px;
    color: #cdd6f4;
    cursor: pointer;
    transition: border-color 0.12s, transform 0.08s, background 0.12s;
    user-select: none;
}
.c2c-launcher-tile:hover {
    border-color: #89b4fa;
    background: #181825;
}
.c2c-launcher-tile:active { transform: scale(0.97); }
.c2c-launcher-tile.c2c-missing {
    opacity: 0.4;
    cursor: not-allowed;
}
.c2c-launcher-tile.c2c-missing:hover {
    border-color: #313244;
    background: #1e1e2e;
}
.c2c-launcher-icon {
    font-size: 22px;
    line-height: 1;
}
.c2c-launcher-label {
    font-size: 11px;
    text-align: center;
    color: #bac2de;
}
.c2c-launcher-foot {
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid #313244;
    font-size: 11px;
    color: #6c7086;
    line-height: 1.5;
}
.c2c-launcher-foot kbd {
    background: #313244;
    color: #cdd6f4;
    padding: 1px 5px;
    border-radius: 3px;
    font-family: ui-monospace, monospace;
    font-size: 10px;
}
    `.trim();
    document.head.appendChild(s);
}

function _delegate(btnId) {
    const t = document.getElementById(btnId);
    if (!t) {
        console.warn(`[C2C.Launcher] target #${btnId} not found in DOM`);
        return false;
    }
    // Temporarily un-hide for the click so any toggle logic that reads
    // element visibility still works, then re-hide on the next frame.
    const prevDisplay = t.style.display;
    t.style.display = "flex";
    try { t.click(); }
    finally {
        requestAnimationFrame(() => { t.style.display = prevDisplay; });
    }
    return true;
}

function _mountLauncher(el) {
    _ensureRootStyle();
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
        tile.innerHTML = `
            <div class="c2c-launcher-icon">${t.icon}</div>
            <div class="c2c-launcher-label">${t.label}</div>
        `;
        tile.addEventListener("click", () => {
            const ok = _delegate(t.btnId);
            if (!ok) {
                tile.classList.add("c2c-missing");
                tile.title = `${t.label} — module not loaded`;
            }
        });
        if (!document.getElementById(t.btnId)) {
            // Module hasn't mounted yet (race) — re-check shortly.
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
    foot.innerHTML = `
        Tiles forward to the matching action module. Toggle
        <kbd>C2C: consolidate floating buttons</kbd> in Settings to bring
        back the original right-rail buttons.
    `;
    root.appendChild(foot);

    el.appendChild(root);
}

app.registerExtension({
    name: "C2C.Launcher",
    settings: [
        {
            id: "c2c.launcher.consolidate",
            name: "C2C: consolidate floating buttons into sidebar",
            tooltip: "Hide the right-rail floating buttons and route them through the C2C sidebar tab.",
            type: "boolean",
            default: true,
            onChange: (v) => _ensureHideStyle(v),
        },
    ],
    async setup() {
        // Apply consolidation hide-rule from saved setting on startup.
        let consolidate = true;
        try {
            consolidate = app.ui.settings.getSettingValue("c2c.launcher.consolidate", true);
        } catch { /* setting not yet registered */ }
        _ensureHideStyle(consolidate);

        try {
            app.extensionManager?.registerSidebarTab?.({
                id: SIDEBAR_ID,
                icon: "pi pi-th-large",
                title: "C2C",
                tooltip: "Code2Collapse quick actions launcher",
                type: "custom",
                render(el) { _mountLauncher(el); },
            });
            console.log("[C2C.Launcher] sidebar registered (id=" + SIDEBAR_ID + ")");
        } catch (e) {
            console.warn("[C2C.Launcher] sidebar register failed:", e);
        }
    },
});
