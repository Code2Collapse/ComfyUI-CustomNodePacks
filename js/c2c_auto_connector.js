/**
 * c2c_auto_connector.js — auto-connect newly added nodes.
 *
 * When the user adds a new node on its own (search box, node library, add menu), wire it to the NEAREST node on
 * its left: every free input whose type has exactly one matching output there gets a wire (owner A9, 2026-10-09:
 * "Nearest node, exact types" + "Every matching input"). Exact type strings only, never "*", never a widget input.
 * Nothing is wired when no node matches, when two candidates are about equally near, or when core already wired the
 * new node (a wire released on the canvas opens the search box and core connects that wire itself). The wires are
 * one undo step. Opt-in via setting.
 *
 * Apache-2.0 © Code2Collapse.
 */

import { app } from "../../scripts/app.js";
import { asOneUndoStep } from "./_c2c_undo_scope.js";

const SETTING_ID = "c2c.autoConnector.enabled";
let _lastAddTs = 0;             // dedupe rapid double-add.

// Only a node the user just added ON ITS OWN may be auto-wired. Graph load and ComfyUI's undo (which reloads
// the whole graph) add every node again; paste and Alt-drag duplicate add copies. Wiring those rewired whole
// workflows on every Ctrl+Z (A9, 2026-10-08: "undo is damaging my entire workflow", "connections damaged
// because of autoconnect").
let _altPointerDown = false;    // Alt held while a pointer is down = core's clone-drag
let _pasting = 0;               // depth of paste calls in progress
let _tick = null;               // adds in the current tick; several = paste / load / subgraph, never wire

function graphLoading() {
    try {
        const CT = window.comfyAPI?.changeTracker?.ChangeTracker;
        if (CT && CT.isLoadingGraph) return true;
    } catch { /* ignore */ }
    return false;
}

function enabled() {
    try { return app.ui.settings.getSettingValue(SETTING_ID, false); }
    catch { return false; }
}

const SEARCH_RADIUS = 900;     // graph px from the new node's left edge
const TIE_RATIO = 1.15;         // a second candidate within 15% of the nearest = ambiguous, wire nothing

function wireable(inp) {
    return inp && inp.link == null && !inp.widget && inp.type && inp.type !== "*";
}

/** Inputs of `node` that `src` can feed by EXACT type with exactly one candidate output each. Only the FIRST free
 *  input of a type is wired (as core's own drop-on-node picks): feeding one IMAGE into both "destination" and
 *  "source" of a compositor would be nonsense. */
function plan(src, node) {
    const wires = [];
    const seen = new Set();
    (node.inputs || []).forEach((inp, i) => {
        if (!wireable(inp) || seen.has(inp.type)) return;
        seen.add(inp.type);
        const outs = [];
        (src.outputs || []).forEach((o, k) => { if (o && o.type === inp.type) outs.push(k); });
        if (outs.length === 1) wires.push([outs[0], i]);
    });
    return wires;
}

/** The nearest node on the new node's left (its right edge left of our left edge), by edge-to-edge distance. */
function candidates(node) {
    const g = node.graph;
    const left = node.pos[0], midY = node.pos[1] + (node.size?.[1] || 0) / 2;
    const out = [];
    for (const n of g?._nodes || []) {
        if (n === node || !n.outputs?.length) continue;
        const right = n.pos[0] + (n.size?.[0] || 0);
        if (right > left + 20) continue;                    // not upstream
        const dy = Math.max(0, Math.abs((n.pos[1] + (n.size?.[1] || 0) / 2) - midY) - (n.size?.[1] || 0) / 2);
        const d = Math.hypot(left - right, dy);
        if (d <= SEARCH_RADIUS) out.push({ n, d });
    }
    return out.sort((a, b) => a.d - b.d);
}

function alreadyWired(node) {
    return (node.inputs || []).some((i) => i.link != null)
        || (node.outputs || []).some((o) => o.links?.length);
}

function autoWire(newNode) {
    if (!newNode?.graph || !newNode.inputs?.some(wireable) || alreadyWired(newNode)) return;
    const ranked = candidates(newNode).map((c) => ({ ...c, wires: plan(c.n, newNode) })).filter((c) => c.wires.length);
    if (!ranked.length) return;
    if (ranked.length > 1 && ranked[1].d <= ranked[0].d * TIE_RATIO + 4) return;   // two about equally near
    const { n: src, wires } = ranked[0];
    asOneUndoStep(app.canvas, () => {
        for (const [outIdx, inIdx] of wires) {
            try { src.connect(outIdx, newNode, inIdx); }
            catch (e) { console.warn("[C2C.AutoConnector] connect failed:", e); }
        }
    });
}

app.registerExtension({
    name: "C2C.AutoConnector",
    settings: [{
        id: SETTING_ID,
        name: "Auto-connect newly added nodes",
        tooltip: "When you add a node on its own, wire it to the nearest node on its left: every free input whose "
            + "type has exactly one matching output there. Exact types only; nothing is wired when two nodes are "
            + "about equally near or when ComfyUI already wired the new node. One Ctrl+Z removes the wires. Opt-in.",
        type: "boolean", defaultValue: false,
        category: ["c2c", "Productivity", "Auto Connector"],
    }],
    async setup() {
        // Hook .add on the instance (ComfyUI replaces it with its own wrapper)
        // AND on the prototype (in case future code calls super-style).
        const installHook = (target) => {
            if (!target || target._c2c_ac_patched) return;
            const _orig = target.add;
            if (typeof _orig !== "function") return;
            target.add = function (node, ...rest) {
                const r = _orig.call(this, node, ...rest);
                if (!enabled()) return r;
                // rest[0] === true is LiteGraph's skip_compute_order: configure()/load, never a user add.
                const loading = rest[0] === true || graphLoading();
                const copying = _altPointerDown || _pasting > 0;
                // One record per tick, held by reference: every add of the same tick (paste, template, load)
                // sees the final count when its check runs.
                if (!_tick) { _tick = { count: 0 }; setTimeout(() => { _tick = null; }, 0); }
                const tick = _tick;
                tick.count += 1;
                if (loading || copying) return r;
                // After core had its chance: a wire released on the canvas opens the search box, and core
                // connects that wire to the node it creates (inputs OR outputs) after adding it.
                setTimeout(() => {
                    try {
                        if (tick.count > 1) return;                       // a batch add, never wire
                        if (Date.now() - _lastAddTs < 80) return;
                        _lastAddTs = Date.now();
                        autoWire(node);
                        node?.graph?.setDirtyCanvas?.(true, true);
                    } catch (e) { console.warn("[C2C.AutoConnector]", e); }
                }, 60);
                return r;
            };
            target._c2c_ac_patched = true;
        };
        installHook(app.graph);                                       // instance
        installHook(LiteGraph?.LGraph?.prototype);                    // prototype

        window.addEventListener("pointerdown", (e) => { _altPointerDown = !!e.altKey; }, { capture: true, passive: true });
        const up = () => { _altPointerDown = false; };
        window.addEventListener("pointerup", up, { capture: true, passive: true });
        window.addEventListener("pointercancel", up, { capture: true, passive: true });
        const LGC = window.LiteGraph?.LGraphCanvas?.prototype;
        for (const name of ["pasteFromClipboard", "_pasteFromClipboard"]) {
            const orig = LGC?.[name];
            if (typeof orig !== "function" || orig.__c2cAcPaste) continue;
            const wrapped = function (...args) {
                _pasting += 1;
                try {
                    const r = orig.apply(this, args);
                    if (r && typeof r.then === "function") return r.finally(() => { _pasting -= 1; });
                    _pasting -= 1;
                    return r;
                } catch (e) { _pasting -= 1; throw e; }
            };
            wrapped.__c2cAcPaste = true;
            LGC[name] = wrapped;
        }
        console.log("[C2C.AutoConnector] ready.");
    },
});
