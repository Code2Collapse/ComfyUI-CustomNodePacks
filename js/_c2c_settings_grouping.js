/**
 * _c2c_settings_grouping.js — unify all C2C/MEC settings under ONE "C2C" section.
 *
 * Why this exists:
 *   Our ~120 settings were registered across ~40 files with no `category`, so the
 *   ComfyUI settings panel grouped them by their ID prefix — producing THREE of our
 *   sections side by side: "c2c", "mec", and "C2C Overlays". To the user that reads
 *   as the same project's settings duplicated/spammed under multiple brand headers.
 *
 * What this does (non-breaking — IDs and saved values are untouched):
 *   - Forces `category[0] = "C2C"` on every setting whose id starts with c2c./mec.,
 *     keeping any meaningful sub-group as a nested level, otherwise deriving a tidy
 *     sub-group from the id's second segment. The panel then shows ONE "C2C" section.
 *   - The setting ID (e.g. `mec.noodle.style`) is the persistence key and is NOT
 *     changed, so no toggle is reset and no migration shim is needed.
 *
 * Robustness (order-independent):
 *   1. Patch `app.ui.settings.addSetting` so settings registered AFTER us are
 *      normalized at registration time.
 *   2. In setup() (after all declarative settings are registered) sweep the live
 *      reactive store — `app.extensionManager.setting.settings` — which is the same
 *      object the Vue panel renders from; mutating `.category` there re-groups live.
 *   3. A short delayed re-sweep catches any lazily/late-registered settings.
 */
import { app } from "../../scripts/app.js";

const OURS = /^(c2c|mec)\./i;

// Nicer sub-group labels for a few ids where Title-Case of the segment looks off.
const PRETTY = {
    omnibar: "OmniBar",
    system_hud: "System HUD",
    progress_hud: "Progress HUD",
    node_explain: "Node Explainer",
    ai: "AI",
    completion_fx: "Celebrations",
    noodle: "Noodle Styles",
    colorspace_badges: "Colorspace Badges",
    seed_sweep: "Seed Sweep",
    surprise_me: "Surprise Me",
    lora_scrubber: "LoRA Scrubber",
    token_counter: "Token Counter",
    wire_labels: "Wire Labels",
    sticky_notes: "Sticky Notes",
    batch_runner: "Batch Runner",
    cost_estimator: "Cost Estimator",
    complexity_hud: "Complexity HUD",
    compatibility_hints: "Compatibility Hints",
    style_presets: "Style Presets",
    frame_overlay: "Frame Overlay",
    insight_overlay: "Insight Overlay",
    isolate: "Isolate Subgraph",
    autolayout: "Auto-layout",
    autocheckpoint: "Auto-checkpoint",
    reset: "Reset to Defaults",
    overlays: "Overlays",
    theme: "Theme",
    linkedValues: "Linked Values",
};

function titleCase(s) {
    return String(s).replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()).trim();
}

/** Compute the normalized category array for one setting (or null if not ours). */
function normalizeCategory(setting) {
    const id = setting && typeof setting.id === "string" ? setting.id : "";
    if (!OURS.test(id)) return null;
    let sub;
    if (Array.isArray(setting.category) && setting.category.length) {
        // Keep their sub-grouping but strip any leading C2C/MEC brand token.
        let parts = setting.category.slice();
        parts[0] = String(parts[0]).replace(/^c2c\s*/i, "").replace(/^mec\s*/i, "").trim();
        parts = parts.filter(Boolean);
        if (parts[0] === "C2C") parts = parts.slice(1);
        sub = parts.length ? parts : null;
    }
    if (!sub) {
        const seg = id.split(".")[1] || "General";
        sub = [PRETTY[seg] || titleCase(seg)];
    }
    return ["C2C", ...sub];
}

let _patched = false;
function patchAddSetting() {
    const s = app.ui?.settings;
    if (!s || typeof s.addSetting !== "function" || s.addSetting.__c2c_grouped) return false;
    const orig = s.addSetting.bind(s);
    const wrapped = function (params) {
        try {
            const cat = normalizeCategory(params);
            if (cat) params = { ...params, category: cat };
        } catch (_) { /* never block a registration */ }
        return orig(params);
    };
    wrapped.__c2c_grouped = true;
    s.addSetting = wrapped;
    _patched = true;
    return true;
}

/** Sweep the live reactive store and re-group anything already registered. */
function sweepExisting() {
    const store = app.extensionManager?.setting?.settings;
    if (!store) return 0;
    let changed = 0;
    for (const setting of Object.values(store)) {
        const cat = normalizeCategory(setting);
        if (!cat) continue;
        const cur = Array.isArray(setting.category) ? setting.category.join("/") : "";
        if (cur === cat.join("/")) continue;
        try { setting.category = cat; changed++; } catch (_) { /* ignore frozen */ }
    }
    return changed;
}

// (1) Patch as early as possible so future registrations are normalized at add-time.
patchAddSetting();

app.registerExtension({
    name: "C2C.SettingsGrouping",
    async init() {
        // app.ui.settings may not have existed at module-eval; ensure the patch lands.
        if (!_patched) patchAddSetting();
    },
    async setup() {
        if (!_patched) patchAddSetting();
        const n = sweepExisting();
        console.log(`[C2C.SettingsGrouping] unified ${n} setting(s) under one "C2C" section`);
        // Catch any settings registered lazily after first paint.
        setTimeout(sweepExisting, 1500);
        setTimeout(sweepExisting, 4000);
    },
});
