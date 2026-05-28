/**
 * c2c_slot_getset.js — fast wire ↔ Get/Set on slots.
 *
 * Adds two new interactions on LiteGraph slots:
 *
 *   • Double-click an OUTPUT slot:
 *       → spawns a kjnodes `SetNode` right next to it, connects the
 *         output into the SetNode input, and names the variable after
 *         the slot.  (Quick "register a variable here" gesture.)
 *
 *   • Double-click an INPUT slot:
 *       → spawns a kjnodes `GetNode` next to it, picks the most-recent
 *         matching SetNode variable (or creates one with the slot's
 *         name), and wires it into this input.
 *
 *   • Right-click a slot:
 *       → adds extra entries "Set Variable…" / "Get Variable…" beside
 *         LiteGraph's default slot menu (Disconnect Links, Remove…).
 *
 * Requires kjnodes' SetNode / GetNode (auto-detected).  If they are
 * missing, the extension stays inert.
 *
 * Apache-2.0 © Code2Collapse.
 */

import { app } from "../../scripts/app.js";

const SETTING_ID = "c2c.slotGetSet.enabled";

function enabled() {
    try { return app.ui.settings.getSettingValue(SETTING_ID, true); }
    catch { return true; }
}

function hasKJSetGet() {
    return !!(LiteGraph?.registered_node_types?.["SetNode"]
           && LiteGraph?.registered_node_types?.["GetNode"]);
}

function suggestName(base, slotType) {
    const t = (slotType || "ANY").replace(/[^A-Za-z0-9]+/g, "");
    const b = String(base || "var").replace(/[^A-Za-z0-9_]+/g, "_");
    return `${b}_${t}`;
}

function findSetByName(name) {
    const g = app.graph;
    if (!g) return null;
    for (const n of g._nodes || []) {
        if (n.type !== "SetNode") continue;
        const w = (n.widgets || []).find(w => w.name === "Constant");
        if (w && String(w.value) === String(name)) return n;
    }
    return null;
}

function listSetNames(matchType) {
    const out = [];
    for (const n of app.graph?._nodes || []) {
        if (n.type !== "SetNode") continue;
        const w = (n.widgets || []).find(w => w.name === "Constant");
        if (!w) continue;
        const inSlot = n.inputs?.[0];
        if (matchType && inSlot && inSlot.type !== "*" && inSlot.type !== matchType) continue;
        out.push(String(w.value || ""));
    }
    return out;
}

function spawnSetForOutput(srcNode, outSlot) {
    if (!hasKJSetGet()) return null;
    const out = srcNode.outputs?.[outSlot];
    if (!out) return null;
    const sn = LiteGraph.createNode("SetNode");
    app.graph.add(sn);
    // Place a touch to the right of the source slot.
    const sp = srcNode.getConnectionPos(false, outSlot);
    sn.pos = [sp[0] + 30, sp[1] - 10];
    const w = (sn.widgets || []).find(w => w.name === "Constant");
    if (w) {
        const name = suggestName(out.name || srcNode.title || "var", out.type);
        w.value = name;
        w.callback?.(name);
    }
    try { srcNode.connect(outSlot, sn, 0); } catch (e) { console.warn(e); }
    app.graph.setDirtyCanvas(true, true);
    return sn;
}

function spawnGetForInput(dstNode, inSlot) {
    if (!hasKJSetGet()) return null;
    const inp = dstNode.inputs?.[inSlot];
    if (!inp) return null;
    // If a matching Set exists, prefer it; otherwise we need a source first.
    const names = listSetNames(inp.type);
    if (!names.length) {
        // Can't synthesize a Set without a source — just notify the user.
        if (window.app?.extensionManager?.toast?.add) {
            app.extensionManager.toast.add({
                severity: "info",
                summary: "No matching Set Node",
                detail: `No SetNode publishes a '${inp.type}' value yet. Double-click an output first.`,
                life: 3000,
            });
        }
        return null;
    }
    const gn = LiteGraph.createNode("GetNode");
    app.graph.add(gn);
    const dp = dstNode.getConnectionPos(true, inSlot);
    gn.pos = [dp[0] - 180, dp[1] - 10];
    const w = (gn.widgets || []).find(w => w.name === "Constant");
    if (w) {
        // Pick the most-recently-defined matching name (last entry in graph).
        const choice = names[names.length - 1];
        w.value = choice;
        w.callback?.(choice);
    }
    try { gn.connect(0, dstNode, inSlot); } catch (e) { console.warn(e); }
    app.graph.setDirtyCanvas(true, true);
    return gn;
}

// ── Slot hit-test ────────────────────────────────────────────────────
// Returns {node, slot, isInput} or null for the slot under canvas-local
// (graphspace) coords (x, y).
function slotUnder(x, y) {
    const nodes = app.graph?._nodes || [];
    for (let i = nodes.length - 1; i >= 0; i--) {
        const n = nodes[i];
        if (!n.flags || n.flags.collapsed) continue;
        // Check inputs
        for (let s = 0; s < (n.inputs?.length || 0); s++) {
            const p = n.getConnectionPos(true, s);
            const dx = x - p[0], dy = y - p[1];
            if (dx*dx + dy*dy < 100) return { node: n, slot: s, isInput: true };
        }
        for (let s = 0; s < (n.outputs?.length || 0); s++) {
            const p = n.getConnectionPos(false, s);
            const dx = x - p[0], dy = y - p[1];
            if (dx*dx + dy*dy < 100) return { node: n, slot: s, isInput: false };
        }
    }
    return null;
}

app.registerExtension({
    name: "C2C.SlotGetSet",
    async setup() {
        try {
            app.ui.settings.addSetting({
                id: SETTING_ID,
                name: "Double-click slot to spawn Get/Set Node (requires KJNodes)",
                tooltip: "Double-click an output slot → creates a SetNode and registers a variable named after the slot. Double-click an input → spawns a GetNode wired to the most-recent matching Set.",
                type: "boolean", defaultValue: true,
                category: ["c2c", "Productivity", "Slot Get/Set"],
            });
        } catch {}

        // Hook double-click on canvas.
        const c = app.canvas;
        if (!c) return;
        const _origDbl = c.processMouseDouble || c.onMouseDoubleClick;
        // LGraphCanvas dblclick goes through ad-hoc handling; safer route:
        // attach native dblclick on the canvas element.
        const cvsEl = c.canvas;
        if (cvsEl && !cvsEl._c2c_getset_hooked) {
            cvsEl._c2c_getset_hooked = true;
            cvsEl.addEventListener("dblclick", (e) => {
                if (!enabled() || !hasKJSetGet()) return;
                const rect = cvsEl.getBoundingClientRect();
                const cx = (e.clientX - rect.left - c.ds.offset[0]) / c.ds.scale;
                const cy = (e.clientY - rect.top  - c.ds.offset[1]) / c.ds.scale;
                const hit = slotUnder(cx, cy);
                if (!hit) return;
                // Don't fire if user is double-clicking to add a node (no slot under).
                e.preventDefault(); e.stopPropagation();
                if (hit.isInput) spawnGetForInput(hit.node, hit.slot);
                else             spawnSetForOutput(hit.node, hit.slot);
            }, true);  // capture
        }

        // Augment per-node slot context menu.
        const augmentMenu = (proto) => {
            if (!proto || proto._c2c_getset_menu_hooked) return;
            proto._c2c_getset_menu_hooked = true;
            const _orig = proto.getSlotMenuOptions;
            proto.getSlotMenuOptions = function (slot) {
                const items = _orig ? _orig.call(this, slot) || [] : [];
                if (!enabled() || !hasKJSetGet()) return items;
                const isInput  = slot?.input != null;
                const isOutput = slot?.output != null;
                const slotIdx  = (slot?.slot ?? slot?.index ?? -1);
                const node = this;
                if (isOutput && slotIdx >= 0) {
                    items.push(null);
                    items.push({
                        content: "Set Variable…",
                        callback: () => spawnSetForOutput(node, slotIdx),
                    });
                }
                if (isInput && slotIdx >= 0) {
                    items.push(null);
                    items.push({
                        content: "Get Variable…",
                        callback: () => spawnGetForInput(node, slotIdx),
                    });
                }
                return items;
            };
        };
        // Patch base LGraphNode prototype (covers every node type).
        augmentMenu(window.LGraphNode?.prototype);
        console.log("[C2C.SlotGetSet] ready. KJNodes present:", hasKJSetGet());
    },
});
