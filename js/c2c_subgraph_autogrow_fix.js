/**
 * Convert to Subgraph keeps every link on its input (L9.96) - a core frontend fix, behind a setting.
 *
 * Converting a node that has growing input groups (autogrow: core MiniMax H3 Reference to Video, TextEncodeQwenImage21,
 * Bernini Conditioning, ...) into a subgraph puts its links on the wrong inputs. The copy inside the subgraph gets its
 * inputs in schema order with the grown slots moved to the end, while every link keeps its old slot INDEX. Measured
 * on frontend 1.53.10, 1.55.14 and 1.57.0 with core alone (--disable-all-custom-nodes): a reference video landed on
 * an AUDIO input, width's link on length (docs/evidence/L9.96).
 *
 * Fix: note each node's input names just before the conversion, then put every link back on the input with the same
 * name. This goes through the frontend's own disconnect / connect calls, because 1.53+ derives input.link from a store
 * and ignores assignments. It also removes the empty autogrow slots the mistake grew. Nodes that were not affected
 * are not touched. The MiniMax Suite ships the same fix; whichever loads first marks the prototype and the other
 * stands down.
 */
import { app } from "../../scripts/app.js";
import { getRuntime } from "./_c2c_runtime.js";

const SETTING_ID = "c2c.coreFix.subgraphAutogrowLinks";
const FLAG = "__c2cAutogrowSubgraphFix";
let _enabled = true;

function linkOf(graph, id) {
    return graph?.links?.get ? graph.links.get(id) : graph?.links?.[id];
}

const slotIndex = (node, name) => (node.inputs || []).findIndex((s) => s.name === name);

function inputLinkOf(graph, node, i) {
    return node.getInputLink ? node.getInputLink(i) : linkOf(graph, node.inputs[i]?.link);
}

export function relinkByName(graph, node, oldNames) {
    const now = (node?.inputs || []).map((s) => s.name);
    if (!oldNames || !now.length || now.every((n, i) => n === oldNames[i])) return 0;
    const moves = [];
    now.forEach((name, i) => {
        const l = inputLinkOf(graph, node, i);
        if (!l) return;
        const want = oldNames[l.target_slot ?? i];
        if (want == null || want === name || slotIndex(node, want) < 0) return;
        moves.push({ originId: l.origin_id, originSlot: l.origin_slot, wrong: name, want });
    });
    for (const m of moves) {
        const i = slotIndex(node, m.wrong);
        if (i >= 0) node.disconnectInput(i);
    }
    let done = 0;
    for (const m of moves) {
        const j = slotIndex(node, m.want);
        if (j < 0) continue;
        const inputNode = graph.inputNode;
        if (inputNode && m.originId === inputNode.id) {
            const sin = inputNode.slots?.[m.originSlot] ?? graph.inputs?.[m.originSlot];
            sin?.connect(node.inputs[j], node);
        } else {
            graph.getNodeById(m.originId)?.connect(m.originSlot, node, j);
        }
        done += inputLinkOf(graph, node, slotIndex(node, m.want)) ? 1 : 0;
    }
    if (moves.length) trimGrownSlots(graph, node);
    return done;
}

// Each autogrow group keeps its connected slots and ONE empty slot after the last of them (at least its first slot).
export function trimGrownSlots(graph, node) {
    const groups = new Map();
    (node.inputs || []).forEach((s, i) => {
        const m = /^(.+\.[^.]*_)(\d+)$/.exec(s.name || "");
        if (!m) return;
        const g = groups.get(m[1]) ?? [];
        g.push({ i, n: Number(m[2]), linked: !!inputLinkOf(graph, node, i) });
        groups.set(m[1], g);
    });
    const drop = [];
    for (const slots of groups.values()) {
        const first = Math.min(...slots.map((s) => s.n));
        const last = Math.max(first - 1, ...slots.filter((s) => s.linked).map((s) => s.n));
        for (const s of slots) if (!s.linked && s.n > last + 1) drop.push(s.i);
    }
    for (const i of drop.sort((a, b) => b - a)) node.removeInput(i);
    return drop.length;
}

function install() {
    const g = app.rootGraph ?? app.graph;
    let proto = g && Object.getPrototypeOf(g);
    while (proto && !Object.prototype.hasOwnProperty.call(proto, "convertToSubgraph")) proto = Object.getPrototypeOf(proto);
    if (!proto || proto[FLAG]) return;
    Object.defineProperty(proto, FLAG, { value: true });
    getRuntime().safePatch(proto, "convertToSubgraph", (orig) => function (items, ...rest) {
        if (!_enabled) return orig.call(this, items, ...rest);
        const before = new Map();
        try {
            for (const it of items || []) if (it && Array.isArray(it.inputs)) before.set(String(it.id), it.inputs.map((s) => s.name));
        } catch { /* an unusual selection: convert without the repair */ }
        const res = orig.call(this, items, ...rest);
        try {
            const sg = res?.subgraph ?? res?.node?.subgraph;
            let moved = 0;
            for (const n of sg?.nodes ?? sg?._nodes ?? []) moved += relinkByName(sg, n, before.get(String(n.id)));
            if (moved) {
                console.info(`[C2C] Convert to Subgraph: ${moved} link(s) put back on the inputs they were on (autogrow slot order).`);
                app.canvas?.setDirty?.(true, true);
            }
        } catch (e) {
            console.warn("[C2C] subgraph link repair skipped:", e);
        }
        return res;
    }, { id: "subgraphAutogrowFix.convertToSubgraph" });
}

app.registerExtension({
    name: "C2C.SubgraphAutogrowFix",
    settings: [{
        id: SETTING_ID,
        name: "Convert to Subgraph keeps links on their inputs",
        tooltip: "ComfyUI moves the links of nodes with growing input lists (for example MiniMax H3 Reference to Video) "
            + "onto the wrong inputs when they are converted into a subgraph. This puts each link back where it was. "
            + "Off = ComfyUI's own behaviour.",
        type: "boolean",
        defaultValue: true,
        category: ["c2c", "Canvas", "Subgraphs"],
        onChange: (v) => { _enabled = v !== false; },
    }],
    async setup() {
        _enabled = app.ui.settings.getSettingValue(SETTING_ID) !== false;
        install();
    },
});
