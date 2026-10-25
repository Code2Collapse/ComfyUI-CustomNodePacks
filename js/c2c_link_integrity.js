/**
 * Link integrity for Code2Collapse nodes (L9.52, owner log 2026-10-25: "VideoComparerC2C 184: Return type mismatch
 * between linked nodes: bit_depth, received_type(IMAGE,LATENT) mismatch input_type(['8', '10', ...])").
 *
 * A saved workflow can hold a wire whose type cannot feed the socket it sits on - there, Load Video's IMAGE on the
 * comparer's bit_depth list. The frontend type-checks a wire the user draws, never one read from a file, so the
 * graph looked fine and the server refused the whole prompt. Measured: frontends 1.53.10 / 1.55.14 / 1.57.0 place
 * fresh, reloaded and older-layout wires correctly; how this wire got there was not reproducible, so the guard fixes
 * the state itself, wherever it came from.
 *
 * On a workflow load (and when a wire is made to one of our nodes), every wire into a Code2Collapse node whose type
 * the socket cannot take is moved to the first FREE socket of that node that takes it (never a widget's), or removed
 * when there is none. One toast says what moved where. Nothing polls; other packs' nodes are never touched.
 * Off switch: setting "C2C › Graph › Repair mistyped wires" (on by default).
 */
import { app } from "../../scripts/app.js";

const SETTING = "c2c.graph.repairMistypedLinks";
const PACK_MODULES = ["ComfyUI-CustomNodePacks", "ComfyUI-GLM_Image", "ComfyUI-NukeMaxNodes", "ComfyUI-MiniMaxSuite",
    "ComfyUI-WanAnimalPreprocessor", "ComfyUI-WanAnimatePreprocessV2", "ComfyUI-WanNodeExperiments",
    "ComfyUI-LTX-Enhanced", "ComfyUI-OmniScale"];
const PACK_TYPES = new Set();

function enabled() {
    try { return app.ui?.settings?.getSettingValue?.(SETTING, true) !== false; } catch (_e) { return true; }
}

function linkOf(graph, id) {
    if (id == null || !graph) return null;
    const L = graph.links ?? graph._links;
    return (L instanceof Map ? L.get(id) : L?.[id]) ?? null;
}

function valid(from, to) {
    try { return LiteGraph.isValidConnection(from, to); } catch (_e) { return true; }
}

function label(node) {
    return node?.title || node?.type || "node";
}

/** Fix one node's inputs; returns human-readable notes. */
export function repairNode(node) {
    const graph = node?.graph;
    const notes = [];
    if (!graph || !PACK_TYPES.has(node.type)) return notes;
    (node.inputs || []).forEach((inp, i) => {
        const l = linkOf(graph, inp?.link);
        if (!l) return;
        const origin = graph.getNodeById(l.origin_id);
        const outType = origin?.outputs?.[l.origin_slot]?.type ?? l.type;
        if (valid(outType, inp.type)) return;
        const name = inp.label || inp.widget?.name || inp.name;
        const to = (node.inputs || []).findIndex((s, j) => j !== i && s && s.link == null && !s.widget
            && valid(outType, s.type));
        node.disconnectInput(i);
        if (origin && to >= 0) {
            origin.connect(l.origin_slot, node, to);
            notes.push(`${label(node)}: the wire from ${label(origin)} sat on "${name}" (it takes ${inp.type}, not `
                + `${outType}). Moved to "${node.inputs[to].label || node.inputs[to].name}".`);
        } else {
            notes.push(`${label(node)}: removed a wire from ${label(origin)} that sat on "${name}" (it takes `
                + `${inp.type}, not ${outType}) - no free input takes it; reconnect it where you meant.`);
        }
    });
    return notes;
}

function walk(graph, fn, seen = new Set()) {
    if (!graph || seen.has(graph)) return;
    seen.add(graph);
    for (const n of graph.nodes || graph._nodes || []) {
        fn(n);
        if (n?.subgraph) walk(n.subgraph, fn, seen);
    }
}

export function repairGraph(root) {
    const notes = [];
    walk(root, (n) => notes.push(...repairNode(n)));
    return notes;
}

function report(notes) {
    if (!notes.length) return;
    console.warn("[C2C] link integrity:", notes);
    try {
        app.extensionManager?.toast?.add({ severity: "warn", summary: "Wires repaired", life: 12000,
            detail: notes.slice(0, 4).join("\n") + (notes.length > 4 ? `\n(+${notes.length - 4} more, see console)` : "") });
    } catch (_e) { /* the console has it */ }
}

if (!globalThis.__c2cLinkIntegrityExt) {
    globalThis.__c2cLinkIntegrityExt = true;
    app.registerExtension({
        name: "C2C.LinkIntegrity",
        settings: [{
            id: SETTING, name: "Repair mistyped wires", type: "boolean", defaultValue: true,
            category: ["c2c", "Graph", "Repair mistyped wires"],
            tooltip: "When a workflow loads (or a wire is made) and a wire into a Code2Collapse node carries a type its "
                + "socket cannot take (an IMAGE on a list widget, ...), move it to the node's first free socket that "
                + "takes it, or remove it. A toast says what moved.",
        }],
        async beforeRegisterNodeDef(nodeType, nodeData) {
            const py = String(nodeData?.python_module || "");
            if (!PACK_MODULES.some((m) => py.includes(m))) return;
            PACK_TYPES.add(nodeData.name);
            const orig = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, slot, connected, ...rest) {
                const r = orig?.apply(this, [type, slot, connected, ...rest]);
                if (connected && type === LiteGraph.INPUT && enabled() && !app.configuringGraph && !this.__c2cRepairing) {
                    this.__c2cRepairing = true;
                    try { report(repairNode(this)); } finally { this.__c2cRepairing = false; }
                }
                return r;
            };
        },
        afterConfigureGraph() {
            if (!enabled()) return;
            report(repairGraph(app.rootGraph ?? app.graph));
        },
    });
}
