/**
 * c2c_overlay_visibility.js — One-stop ON/OFF switch for every floating
 * C2C/MEC overlay, HUD, badge, and quick-action button.
 *
 * Why this exists:
 *   The user reported "bottom-right UI thing is covering the main part"
 *   and asked for a separate toggle per overlay so they can hide the
 *   ones they don't use.
 *
 *   Rather than patch every module's own `addSetting()` (some have none,
 *   some have a single master toggle), this single file:
 *
 *     1. Registers a Boolean setting per overlay under
 *        Settings → C2C Overlays.
 *     2. Maintains a single CSS rule that `display: none !important`s
 *        the overlays whose setting is OFF.
 *     3. Reapplies the rule on every settings-change and on document
 *        mutation (in case the overlay gets re-mounted later).
 *
 * Default = true (show) for every overlay, so existing users see no
 * behavioral change until they explicitly hide something.
 */

import { app } from "../../scripts/app.js";

const STYLE_ID = "c2c-overlay-visibility-style";
const SETTING_PREFIX = "c2c.overlays.";

/**
 * Registry of every togglable overlay.
 *   - id       : Settings key suffix (also used in CSS rule).
 *   - selector : CSS selector targeting the overlay DOM element(s).
 *   - label    : Human-readable name shown in Settings.
 *   - category : Sub-group inside Settings.
 *   - tip      : Tooltip shown next to the setting row.
 *   - defaultOn: Initial value (true = visible, the safe default).
 *
 * Groups roughly mirror visual position so the user can find them.
 */
const OVERLAYS = [
    // ── Always-on HUDs (chips / pills) ─────────────────────────────
    {
        id: "aiStatusBar",
        selector: "#c2c-ai-hud, #c2c-ai-hud-flyout",
        label: "AI Status Bar (top-right pills)",
        category: ["C2C Overlays", "Top-right HUDs"],
        tip: "⚡ active-backend / $ daily cost / ● online pills + flyout.",
        defaultOn: true,
    },
    {
        id: "systemHud",
        selector: "#mec-system-hud",
        label: "System HUD (queue / VRAM / cost)",
        category: ["C2C Overlays", "Bottom-right HUDs"],
        tip: "Bottom-right chip with queue depth + VRAM + AI $ today.",
        defaultOn: true,
    },
    {
        id: "complexityHud",
        selector: "#mec-complexity-hud",
        label: "Complexity HUD (top-center)",
        category: ["C2C Overlays", "Top-center HUDs"],
        tip: "Workflow complexity / node count badge.",
        defaultOn: true,
    },
    {
        id: "whatsWired",
        selector: "#mec-whats-wired",
        label: "What's Wired chip",
        category: ["C2C Overlays", "Top-left HUDs"],
        tip: "Live link / wire summary chip.",
        defaultOn: true,
    },
    // ── Floating action buttons (right rail) ───────────────────────
    {
        id: "wizardBtn",
        selector: "#mec-wizard-btn",
        label: "Workflow Wizard button",
        category: ["C2C Overlays", "Right-rail buttons"],
        tip: "🧙 floating wizard launcher (already in C2C Launcher sidebar).",
        defaultOn: false,
    },
    {
        id: "moodBoardBtn",
        selector: "#mec-mood-btn",
        label: "Mood Board button",
        category: ["C2C Overlays", "Right-rail buttons"],
        tip: "🎨 reference image palette.",
        defaultOn: false,
    },
    {
        id: "surpriseBtn",
        selector: "#mec-surprise-btn",
        label: "Surprise Me button",
        category: ["C2C Overlays", "Right-rail buttons"],
        tip: "🎰 randomize seeds & queue.",
        defaultOn: false,
    },
    {
        id: "abSplitBtn",
        selector: "#mec-ab-btn",
        label: "A/B Split button",
        category: ["C2C Overlays", "Right-rail buttons"],
        tip: "⚖ compare last two outputs.",
        defaultOn: false,
    },
    {
        id: "costBtn",
        selector: "#mec-cost-btn",
        label: "Cost Estimator button",
        category: ["C2C Overlays", "Right-rail buttons"],
        tip: "💰 estimate render cost.",
        defaultOn: false,
    },
    {
        id: "flamegraphBtn",
        selector: "#mec-flamegraph-btn",
        label: "Flame Graph button",
        category: ["C2C Overlays", "Right-rail buttons"],
        tip: "⏱ per-node execution timing.",
        defaultOn: false,
    },
    {
        id: "undoBtn",
        selector: "#mec-undo-btn",
        label: "Undo Panel button",
        category: ["C2C Overlays", "Right-rail buttons"],
        tip: "↶ Undo history viewer.",
        defaultOn: false,
    },
    {
        id: "groupPresetsBtn",
        selector: "#mec-group-presets-btn",
        label: "Group Presets button",
        category: ["C2C Overlays", "Right-rail buttons"],
        tip: "📚 save & recall node-group templates.",
        defaultOn: false,
    },
    // Legacy `integrityBtn` (#mec-integrity-btn) entry retired 2026-05-26.
    // The canonical INT chip now lives inside the OmniBar as #c2c-int-chip.
    {
        id: "doctorBtn",
        selector: "#c2c-doctor-btn",
        label: "Workflow Doctor button",
        category: ["C2C Overlays", "Right-rail buttons"],
        tip: "🩺 lint your workflow.",
        defaultOn: true,
    },
    {
        id: "autoCheckpointBtn",
        selector: "#c2c-autockpt-btn",
        label: "Auto-Checkpoint button",
        category: ["C2C Overlays", "Right-rail buttons"],
        tip: "💾 auto checkpoint picker.",
        defaultOn: true,
    },
    // ── In-canvas decorations ─────────────────────────────────────
    {
        id: "frameOverlay",
        selector: ".mec-frame-overlay-host",
        label: "Per-node frame overlay",
        category: ["C2C Overlays", "Canvas decorations"],
        tip: "Sample frame thumbnails drawn over video nodes.",
        defaultOn: true,
    },
    {
        id: "completionFx",
        selector: "#mec-confetti-canvas",
        label: "Completion confetti effect",
        category: ["C2C Overlays", "Canvas decorations"],
        tip: "🎉 burst when a queue finishes successfully.",
        defaultOn: false,
    },
    {
        id: "errorToast",
        selector: ".mec-error-toast-host",
        label: "Error toast popups",
        category: ["C2C Overlays", "Canvas decorations"],
        tip: "Bottom-center error toasts.",
        defaultOn: true,
    },
    {
        id: "miniRow",
        selector: "#c2c-mini-row-host",
        label: "C2C mini-row (color-coded launcher)",
        category: ["C2C Overlays", "Top-center HUDs"],
        tip: "Compact horizontal launcher row.",
        defaultOn: true,
    },
];

// ── Visibility state in-memory ─────────────────────────────────────
const STATE = Object.create(null);
for (const o of OVERLAYS) STATE[o.id] = o.defaultOn;

function _readSettings() {
    const s = app.ui?.settings;
    if (!s) return;
    for (const o of OVERLAYS) {
        const v = s.getSettingValue(SETTING_PREFIX + o.id);
        if (typeof v === "boolean") STATE[o.id] = v;
    }
}

function _composeCss() {
    const hidden = OVERLAYS.filter(o => STATE[o.id] === false);
    if (!hidden.length) return "";
    const selectors = hidden.map(o => o.selector).join(",\n");
    return `${selectors} {\n  display: none !important;\n}`;
}

function _applyCss() {
    let s = document.getElementById(STYLE_ID);
    const css = _composeCss();
    if (!css) {
        if (s) s.remove();
        return;
    }
    if (!s) {
        s = document.createElement("style");
        s.id = STYLE_ID;
        document.head.appendChild(s);
    }
    s.textContent = css;
}

function _onSettingChange(id, value) {
    STATE[id] = !!value;
    _applyCss();
}

app.registerExtension({
    name: "C2C.OverlayVisibility",
    async setup() {
        // Register one setting per overlay, grouped under "C2C Overlays".
        for (const o of OVERLAYS) {
            try {
                app.ui.settings.addSetting({
                    id: SETTING_PREFIX + o.id,
                    name: o.label,
                    category: o.category,
                    type: "boolean",
                    defaultValue: o.defaultOn,
                    tooltip: o.tip,
                    onChange: (v) => _onSettingChange(o.id, v),
                });
            } catch (err) {
                console.warn("[C2C.OverlayVisibility] addSetting failed:", o.id, err);
            }
        }
        _readSettings();
        _applyCss();

        // Re-apply when overlays get mounted later — covers modules that
        // create their DOM in response to user actions (e.g. flyouts).
        const mo = new MutationObserver(() => _applyCss());
        mo.observe(document.body, { childList: true, subtree: false });
    },
});
