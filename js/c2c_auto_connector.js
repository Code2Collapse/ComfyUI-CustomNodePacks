/**
 * c2c_auto_connector.js — auto-connect newly added nodes.
 *
 * When the user adds a new node (palette, drag, paste, dblclick add-menu),
 * try to wire its first compatible input to the most-recently-selected
 * node's first compatible free output. Opt-in via setting.
 *
 * Apache-2.0 © Code2Collapse.
 */

import { app } from "../../scripts/app.js";

const SETTING_ID = "c2c.autoConnector.enabled";
let _lastNode = null;          // last node the user clicked / created.
let _lastAddTs = 0;             // dedupe rapid double-add.

function enabled() {
    try { return app.ui.settings.getSettingValue(SETTING_ID, false); }
    catch { return false; }
}

function typeMatches(srcType, dstType) {
    if (!srcType || !dstType) return false;
    if (srcType === "*" || dstType === "*") return true;
    if (srcType === dstType) return true;
    // LiteGraph allows comma-separated alternative types.
    const srcList = String(srcType).split(",").map(s => s.trim());
    const dstList = String(dstType).split(",").map(s => s.trim());
    return srcList.some(s => dstList.includes(s));
}

function firstFreeOutput(srcNode, requiredType) {
    if (!srcNode?.outputs) return -1;
    for (let i = 0; i < srcNode.outputs.length; i++) {
        const o = srcNode.outputs[i];
        if (!typeMatches(o.type, requiredType)) continue;
        // Don't reuse "links" — outputs can be re-fanned, but prefer free first.
        if (!o.links || o.links.length === 0) return i;
    }
    // Fallback: allow re-fanning a busy output.
    for (let i = 0; i < srcNode.outputs.length; i++) {
        if (typeMatches(srcNode.outputs[i].type, requiredType)) return i;
    }
    return -1;
}

function autoWire(newNode) {
    if (!newNode || !_lastNode || _lastNode === newNode) return;
    if (!_lastNode.outputs?.length || !newNode.inputs?.length) return;

    // Try every input of newNode (first compatible wins).
    for (let i = 0; i < newNode.inputs.length; i++) {
        const inp = newNode.inputs[i];
        if (inp.link != null) continue;
        const outIdx = firstFreeOutput(_lastNode, inp.type);
        if (outIdx < 0) continue;
        try {
            _lastNode.connect(outIdx, newNode, i);
            return; // one auto-link is enough.
        } catch (e) { console.warn("[C2C.AutoConnector] connect failed:", e); }
    }
}

app.registerExtension({
    name: "C2C.AutoConnector",
    async setup() {
        try {
            app.ui.settings.addSetting({
                id: SETTING_ID,
                name: "Auto-connect newly added nodes",
                tooltip: "When you add a new node to the canvas, automatically wire it to the previously-selected node's first compatible output. Opt-in.",
                type: "boolean", defaultValue: false,
                category: ["c2c", "Productivity", "Auto Connector"],
            });
        } catch {}

        // Track last clicked/selected node.
        const c = app.canvas;
        if (c) {
            const _origSel = c.onNodeSelected;
            c.onNodeSelected = function (n) {
                _lastNode = n || _lastNode;
                return _origSel?.apply(this, arguments);
            };
        }

        // Hook .add on the instance (ComfyUI replaces it with its own wrapper)
        // AND on the prototype (in case future code calls super-style).
        const installHook = (target) => {
            if (!target || target._c2c_ac_patched) return;
            const _orig = target.add;
            if (typeof _orig !== "function") return;
            target.add = function (node, ...rest) {
                const r = _orig.call(this, node, ...rest);
                queueMicrotask(() => {
                    try {
                        if (!enabled()) return;
                        if (Date.now() - _lastAddTs < 80) return;
                        _lastAddTs = Date.now();
                        if (node && node !== _lastNode) {
                            autoWire(node);
                            _lastNode = node;
                            app.graph?.setDirtyCanvas?.(true, true);
                        }
                    } catch (e) { console.warn("[C2C.AutoConnector]", e); }
                });
                return r;
            };
            target._c2c_ac_patched = true;
        };
        installHook(app.graph);                                       // instance
        installHook(LiteGraph?.LGraph?.prototype);                    // prototype
        console.log("[C2C.AutoConnector] ready.");
    },
});
