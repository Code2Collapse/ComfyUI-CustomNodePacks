// ============================================================================
// C2C — P10.2 Reset Widgets to Defaults
// ----------------------------------------------------------------------------
// Why this exists:
//   A power-user often tweaks a dozen sliders to debug a workflow and then
//   wants to snap one or more nodes back to "stock" values without deleting
//   and re-adding them. Vanilla ComfyUI has no such shortcut.
//
// What this does:
//   - Right-click a node > "Reset to ORIGINAL", the canvas menu, or the
//     command "c2c.reset.selectedDefaults": reset the selected nodes (all
//     nodes only after a confirmation) to the defaults declared in
//     /object_info. One undo step.
//   - Ctrl+R does the same ONLY when "c2c.reset.enabled" is switched on
//     (default off since A9: it is the browser's reload key).
//   - Settings > C2C > Reset to Defaults: a button that resets the C2C
//     settings themselves (secrets kept, confirmation first); also the
//     command "c2c.reset.allSettings".
//   - On every node, for each widget, looks up the matching INPUT spec by
//     name in /object_info, reads `default` (or the third element of the
//     legacy [type, opts] tuple), and assigns it.
//   - Skips combos / files / inputs without a default to avoid clobbering
//     loader selections.
//   - SKIPS rgthree Get/Set nodes (Set anything, Get anything,
//     SetNode/GetNode, comfyClass starts with rgthree.) because they have no
//     editable inputs and toggling would break their reroute behavior.
//
// /object_info schema (1.43+):
//     {
//       "NodeName": {
//         "input": {
//           "required": { "image": ["IMAGE"], "steps": ["INT", { "default": 20, ... }] },
//           "optional": { ... }
//         },
//         ...
//       }
//     }
// ============================================================================

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { legacyCanvasMenu } from "./_c2c_compat.js";
import { c2cConfirm } from "./_c2c_dialog.js";
import { asOneUndoStep } from "./_c2c_undo_scope.js";

const LOG = (...a) => console.debug("[c2c-reset]", ...a);

// ---------------------------------------------------- object_info one-shot
let _objectInfoP = null;
function getObjectInfo() {
    if (_objectInfoP) return _objectInfoP;
    _objectInfoP = api.fetchApi("/object_info")
        .then(r => r.json())
        .catch(exc => {
            console.warn("[c2c-reset] /object_info fetch failed:", exc);
            return {};
        });
    return _objectInfoP;
}

// ----------------------------------------------------------------- skip rules
function shouldSkipNode(node) {
    if (!node || !node.type) return true;
    const t = String(node.type);
    if (t.startsWith("Set ") || t.startsWith("Get ")) return true;
    if (t === "SetNode" || t === "GetNode") return true;
    const cc = String(node.comfyClass || "");
    if (cc.toLowerCase().startsWith("rgthree.")) return true;
    if (cc === "SetNode" || cc === "GetNode") return true;
    return false;
}

// File/model combos whose options are filenames — resetting these would clobber
// the user's selected checkpoint/LoRA/VAE, so we never auto-reset them.
const _FILE_EXT_RE = /\.(safetensors|ckpt|pt|pth|bin|gguf|onnx|sft|vae|yaml|yml|json|pkl|npz|png|jpe?g|webp|gif|mp4|webm)$/i;
function _looksLikeFileCombo(opts) {
    return opts.some((o) => typeof o === "string" && (_FILE_EXT_RE.test(o) || o.includes("/") || o.includes("\\")));
}

// ---------------------------------------- pull default from /object_info entry
function extractDefault(spec) {
    // Modern shape:  ["INT", { "default": 20, "min": 1, ... }]
    // Tuple shape:   ["INT", 20]                  (legacy)
    // Combo shape:   [[ "opt1","opt2","opt3" ], { "default": "opt2" }?]
    if (!Array.isArray(spec)) return { has: false };
    const t = spec[0];
    const meta = spec[1];
    if (Array.isArray(t)) {
        // COMBO. The node-author "original" is the explicit `default` if given,
        // otherwise the FIRST option — which is exactly the value a freshly
        // created node shows (sampler_name→euler, scheduler→simple, etc.).
        // We still skip dynamic file/model lists so a reset never silently
        // re-points a loader at the first file on disk.
        if (meta && typeof meta === "object" && "default" in meta && meta.default !== undefined) {
            return { has: true, value: meta.default, isCombo: true };
        }
        if (t.length && !_looksLikeFileCombo(t)) {
            return { has: true, value: t[0], isCombo: true };
        }
        return { has: false, isCombo: true };
    }
    if (spec.length < 2) return { has: false };
    if (meta && typeof meta === "object" && "default" in meta) {
        return { has: true, value: meta.default };
    }
    if (typeof meta === "number" || typeof meta === "string" || typeof meta === "boolean") {
        return { has: true, value: meta };
    }
    return { has: false };
}

// --------------------------------------------------------------- reset core
function resetNode(node, objectInfo) {
    if (shouldSkipNode(node)) return { skipped: true, reason: "skip-rule" };
    const def = objectInfo?.[node.type] || objectInfo?.[node.comfyClass];
    if (!def) return { skipped: true, reason: "no /object_info entry" };
    const inputs = { ...(def.input?.required || {}), ...(def.input?.optional || {}) };
    let changed = 0;
    for (const w of node.widgets || []) {
        if (!w?.name) continue;
        if (w.name.startsWith("_")) continue;       // hidden/internal C2C widgets
        const spec = inputs[w.name];
        if (!spec) continue;
        const d = extractDefault(spec);
        if (!d.has) continue;
        if (w.value === d.value) continue;
        try {
            w.value = d.value;
            if (typeof w.callback === "function") w.callback(d.value);
            changed++;
        } catch (exc) {
            console.warn("[c2c-reset] widget", w.name, "on", node.type, "failed:", exc);
        }
    }
    if (changed > 0) {
        node.setDirtyCanvas?.(true, true);
    }
    return { changed };
}

/**
 * Reset the selected nodes, or - only after a confirmation - every node of the graph on screen. ONE undo step.
 * (A9: Ctrl+R used to reset every widget of every node at once, silently, when nothing was selected.)
 */
async function resetSelectedOrAll() {
    const objectInfo = await getObjectInfo();
    const canvas = app.canvas;
    const graph = canvas?.graph || app?.graph;
    if (!graph) return;
    const selected = Object.values(canvas?.selected_nodes || {});
    const targets = selected.length > 0
        ? selected
        : (graph._nodes || graph.nodes || []);
    if (!targets.length) return;
    if (!selected.length) {
        const ok = await c2cConfirm(
            `Reset the widgets of ALL ${targets.length} nodes in this graph to their defaults?\n\n` +
            "Select nodes first to reset only those. Ctrl+Z undoes the reset in one step.",
            { title: "Reset to defaults", okLabel: "Reset all nodes", cancelLabel: "Cancel" });
        if (!ok) return;
    }
    let resetCount = 0;
    let skipCount = 0;
    let nodesTouched = 0;
    asOneUndoStep(canvas, () => {
        for (const n of targets) {
            const r = resetNode(n, objectInfo);
            if (r.skipped) { skipCount++; continue; }
            if (r.changed > 0) { nodesTouched++; resetCount += r.changed; }
        }
    });
    const scope = selected.length > 0 ? `${selected.length} selected node(s)` : "ALL nodes";
    const msg = `Reset ${resetCount} widget(s) on ${nodesTouched} node(s) (${scope}).` +
                (skipCount > 0 ? ` Skipped ${skipCount}.` : "");
    LOG(msg);
    try {
        app.extensionManager?.toast?.add?.({
            severity: nodesTouched > 0 ? "success" : "info",
            summary: "C2C reset to defaults",
            detail: msg,
            life: 3500,
        });
    } catch { /* toast best-effort */ }
}

// --------------------------------------------------------- typing guard
function isUserTypingInField(target) {
    if (!target) return false;
    const tag = (target.tagName || "").toUpperCase();
    if (tag === "INPUT") {
        const t = (target.type || "").toLowerCase();
        return !["checkbox", "radio", "button", "submit"].includes(t);
    }
    if (tag === "TEXTAREA") return true;
    if (target.isContentEditable) return true;
    return false;
}

// ---- capture-phase keybinding (must beat the browser's Ctrl+R reload) -----
function captureKeydown(e) {
    if (e.key !== "r" && e.key !== "R" && e.code !== "KeyR") return;
    if (!(e.ctrlKey || e.metaKey)) return;
    if (e.shiftKey) return;                  // leave Ctrl+Shift+R (hard-reload) alone
    if (isUserTypingInField(e.target)) return;
    e.preventDefault();
    e.stopPropagation();
    if (typeof e.stopImmediatePropagation === "function") e.stopImmediatePropagation();
    resetSelectedOrAll();
}

// ------------------------------------------------- reset the C2C SETTINGS
// "Reset to defaults" in Settings > C2C used to hold only the Ctrl+R toggle above: there was no way to put the
// C2C settings themselves back (A9, "reset to defaults is not working"). Secrets (API keys, tokens, passwords)
// are never reset, and nothing is written without a confirmation listing what will change.
const OUR_SETTING = /^(c2c|mec)\./i;
const SECRET_SETTING = /(api[_.-]?key|apikey|token|secret|password|passphrase|credential)/i;
const RESET_SETTINGS_ID = "c2c.reset.settingsButton";

function _settingDefault(param) {
    try { return typeof param.defaultValue === "function" ? param.defaultValue() : param.defaultValue; }
    catch { return undefined; }
}

/** Every registered C2C setting whose value differs from its default (secrets excluded). */
function changedC2CSettings() {
    const store = app.extensionManager?.setting?.settings || {};
    const out = [];
    for (const [id, param] of Object.entries(store)) {
        if (!OUR_SETTING.test(id) || SECRET_SETTING.test(id) || id === RESET_SETTINGS_ID) continue;
        if (param?.type === "hidden" || typeof param?.type === "function") continue;
        const def = _settingDefault(param);
        if (def === undefined) continue;
        let cur;
        try { cur = app.ui.settings.getSettingValue(id); } catch { continue; }
        if (cur === undefined || JSON.stringify(cur) === JSON.stringify(def)) continue;
        out.push({ id, name: param.name || id, def });
    }
    return out;
}

async function resetC2CSettings() {
    const changed = changedC2CSettings();
    if (!changed.length) {
        try {
            app.extensionManager?.toast?.add?.({ severity: "info", summary: "C2C settings",
                detail: "Every C2C setting is already at its default.", life: 3500 });
        } catch { /* best-effort */ }
        return 0;
    }
    const shown = changed.slice(0, 12).map((c) => "  - " + c.name).join("\n");
    const more = changed.length > 12 ? `\n  ... and ${changed.length - 12} more` : "";
    const ok = await c2cConfirm(
        `Reset ${changed.length} C2C setting(s) to their defaults?\n\n${shown}${more}\n\n` +
        "API keys, tokens and passwords are kept. Some settings take effect after a page reload.",
        { title: "Reset C2C settings", okLabel: "Reset settings", cancelLabel: "Cancel" });
    if (!ok) return 0;
    let done = 0;
    const S = app.ui.settings;
    for (const c of changed) {
        // one at a time: rapid parallel writes to /api/settings returned 500s (server write contention)
        try {
            if (typeof S.setSettingValueAsync === "function") await S.setSettingValueAsync(c.id, c.def);
            else await S.setSettingValue(c.id, c.def);
            done++;
        } catch (exc) { console.warn("[c2c-reset] setting", c.id, "not reset:", exc); }
    }
    try {
        app.extensionManager?.toast?.add?.({ severity: done === changed.length ? "success" : "warn",
            summary: "C2C settings", detail: `Reset ${done} of ${changed.length} setting(s) to defaults.`, life: 4000 });
    } catch { /* best-effort */ }
    return done;
}

/** Settings-panel control: a button (core renders a function `type` as a custom element). */
function _renderResetSettingsButton() {
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:flex;align-items:center;gap:8px;";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "Reset C2C settings…";
    btn.className = "p-button p-component p-button-sm p-button-secondary";
    btn.addEventListener("click", async () => {
        btn.disabled = true;
        try { await resetC2CSettings(); } finally { btn.disabled = false; }
    });
    wrap.appendChild(btn);
    return wrap;
}

// ================================================================ extension
function _canvasMenuItems() {
    return [null, {
        content: "\u21BA C2C: Reset selection to defaults",
        callback: () => resetSelectedOrAll(),
    }];
}

function _mergeCanvasMenuItems(opts) {
    opts.push(..._canvasMenuItems());
    return opts;
}

app.registerExtension({
    name: "c2c.reset_to_defaults",
    settings: [
        {
            id: RESET_SETTINGS_ID,
            name: "Reset every C2C setting to its default (API keys are kept)",
            tooltip: "Lists the C2C settings that differ from their defaults and asks before changing anything.",
            type: _renderResetSettingsButton,
            defaultValue: "",
        },
        {
            id: "c2c.reset.enabled",
            name: "Ctrl+R resets nodes to their defaults (instead of reloading the page)",
            // OFF by default (A9): Ctrl+R is the browser's reload key, and with nothing selected it reset every
            // widget of every node at once. When on, an all-nodes reset now asks first and is one undo step.
            tooltip: "When ON, Ctrl+R resets the selected nodes (or, after a confirmation, all nodes) to their " +
                     "defaults. OFF: Ctrl+R reloads the page as usual. Right-click a node for 'Reset to ORIGINAL'.",
            type: "boolean",
            defaultValue: false,
        },
    ],
    commands: [
        {
            id: "c2c.reset.selectedDefaults",
            label: "C2C: Reset selected nodes to defaults",
            function: resetSelectedOrAll,
        },
        {
            id: "c2c.reset.allSettings",
            label: "C2C: Reset C2C settings to defaults",
            function: resetC2CSettings,
        },
    ],
    // Per-node right-click: reset THIS node to the ORIGINAL defaults declared by
    // the node author in /object_info (NOT the values baked into a copied
    // workflow). This is the "reset to original" the user asked for — it reads
    // the node's own INPUT_TYPES defaults from the live ComfyUI registry, so it
    // works for KSampler or any custom node, independent of the loaded workflow.
    getNodeMenuItems(node) {
        if (!node || shouldSkipNode(node)) return [];
        return [{
            content: "↺ Reset to ORIGINAL (node author / ComfyUI defaults)",
            callback: async () => {
                const oi = await getObjectInfo();
                let r = { skipped: true, reason: "not run" };
                asOneUndoStep(app.canvas, () => { r = resetNode(node, oi); });
                const det = r.skipped
                    ? `Skipped: ${r.reason}`
                    : (r.changed > 0 ? `Reset ${r.changed} widget(s) to ${node.type} originals.`
                                     : "Already at the node's original defaults.");
                try {
                    app.extensionManager?.toast?.add?.({
                        severity: r.changed > 0 ? "success" : "info",
                        summary: "Reset to original",
                        detail: det, life: 3500,
                    });
                } catch { /* best-effort */ }
            },
        }];
    },
    getCanvasMenuItems() {
        return _canvasMenuItems();
    },
    // NOTE: No `keybindings:` entry — ComfyUI's registry conflict-checks
    // by `key` alone (ignoring modifiers), so { key:"r", ctrl:true }
    // collides with Comfy.RefreshNodeDefinitions' bare `r` and spams a
    // toast. The capture-phase listener below handles Ctrl+R directly.
    async setup() {
        // Capture-phase fallback so we beat the browser's reload-page.
        window.addEventListener("keydown", (e) => {
            const enabled = app.ui?.settings?.getSettingValue?.("c2c.reset.enabled", false);
            if (enabled !== true) return;
            captureKeydown(e);
        }, { capture: true });
        legacyCanvasMenu("reset_to_defaults", _mergeCanvasMenuItems);
        LOG("Reset-to-defaults installed");
    },
});
