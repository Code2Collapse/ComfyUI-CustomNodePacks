// ============================================================================
// C2C — P10.2 Reset Widgets to Defaults
// ----------------------------------------------------------------------------
// Why this exists:
//   A power-user often tweaks a dozen sliders to debug a workflow and then
//   wants to snap one or more nodes back to "stock" values without deleting
//   and re-adding them. Vanilla ComfyUI has no such shortcut.
//
// What this does:
//   - Ctrl+R: reset the currently selected nodes (or all nodes if no
//     selection) to the defaults declared in /object_info.
//   - Command "c2c.reset.selectedDefaults" registered for the command
//     palette.
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

// ---------------------------------------- pull default from /object_info entry
function extractDefault(spec) {
    // Modern shape:  ["INT", { "default": 20, "min": 1, ... }]
    // Tuple shape:   ["INT", 20]                  (legacy)
    // Combo shape:   [[ "opt1","opt2","opt3" ]]   — we do NOT reset combos
    //                                              because the first option
    //                                              isn't always the "default"
    //                                              that the loader wants.
    if (!Array.isArray(spec)) return { has: false };
    const t = spec[0];
    if (Array.isArray(t)) return { has: false, isCombo: true };
    if (spec.length < 2) return { has: false };
    const meta = spec[1];
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

async function resetSelectedOrAll() {
    const objectInfo = await getObjectInfo();
    const graph = app?.graph;
    if (!graph) return;
    const selected = Object.values(app.canvas?.selected_nodes || {});
    const targets = selected.length > 0
        ? selected
        : (graph._nodes || graph.nodes || []);
    let resetCount = 0;
    let skipCount = 0;
    let nodesTouched = 0;
    for (const n of targets) {
        const r = resetNode(n, objectInfo);
        if (r.skipped) { skipCount++; continue; }
        if (r.changed > 0) { nodesTouched++; resetCount += r.changed; }
    }
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

// ================================================================ extension
app.registerExtension({
    name: "c2c.reset_to_defaults",
    settings: [
        {
            id: "c2c.reset.enabled",
            name: "C2C \u25B8 Reset-to-defaults \u25B8 Capture Ctrl+R",
            tooltip: "When ON, Ctrl+R resets selected (or all) nodes to /object_info defaults instead of reloading the browser. Turn OFF to restore browser reload.",
            type: "boolean",
            default: true,
        },
    ],
    commands: [
        {
            id: "c2c.reset.selectedDefaults",
            label: "C2C: Reset selected nodes to defaults",
            function: resetSelectedOrAll,
        },
    ],
    keybindings: [
        // Vue-layer binding (works inside the canvas). Capture-listener below
        // catches the rest.
        { combo: { key: "r", ctrl: true },  commandId: "c2c.reset.selectedDefaults" },
        { combo: { key: "r", metaKey: true }, commandId: "c2c.reset.selectedDefaults" },
    ],
    async setup() {
        // Capture-phase fallback so we beat the browser's reload-page.
        window.addEventListener("keydown", (e) => {
            const enabled = app.ui?.settings?.getSettingValue?.("c2c.reset.enabled", true);
            if (enabled === false) return;
            captureKeydown(e);
        }, { capture: true });
        // Right-click → "Reset to defaults" on selection.
        const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
        if (orig && !orig.__c2c_reset_patched) {
            const patched = function (...args) {
                const opts = orig.apply(this, args) || [];
                opts.push(null);
                opts.push({
                    content: "\u21BA C2C: Reset selection to defaults",
                    callback: () => resetSelectedOrAll(),
                });
                return opts;
            };
            patched.__c2c_reset_patched = true;
            LGraphCanvas.prototype.getCanvasMenuOptions = patched;
        }
        LOG("Reset-to-defaults installed");
    },
});
