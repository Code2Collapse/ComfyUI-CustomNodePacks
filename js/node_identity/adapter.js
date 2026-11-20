// C2C Node Identity - LiteGraph adapter (frontend 1.53.10 APIs, see docs/research/node_identity_design.md s0).
// Builds the engine's plain snapshot from the live graph (root + every subgraph definition) and writes names.
// WRITES ONLY node.title and node.properties.c2c_identity. Never type, comfyClass, id, inputs, outputs, links,
// widget values, size or pos. No imports: the caller passes the root graph.
// License: Apache-2.0

export const PROP = "c2c_identity";
// litegraph constants (fe 1.53.10 constants.ts): node ids are STRINGS there (toNodeId), numbers in older builds -
// always compare as strings
const SUBGRAPH_INPUT_ID = "-10";   // link origin inside a subgraph: its input boundary
const SUBGRAPH_OUTPUT_ID = "-20";  // link target inside a subgraph: its output boundary

const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

export function identityOf(node) {
    const v = node?.properties?.[PROP];
    return v && typeof v === "object" ? v : null;
}

export function defaultTitleOf(node) {
    const c = node?.constructor || {};
    return String(c.title || c.nodeData?.display_name || node?.type || "");
}

export function isSubgraphNode(node) {
    try {
        if (typeof node?.isSubgraphNode === "function") return !!node.isSubgraphNode();
    } catch (_e) { /* older builds */ }
    return !!(node?.subgraph && typeof node.subgraph === "object");
}

/** Every graph of the workflow: the root and each subgraph definition (shared by all its instances). */
export function allGraphs(root) {
    const out = [];
    const seen = new Set();
    const add = (g, path) => {
        if (!g || seen.has(g)) return;
        seen.add(g);
        out.push({ graph: g, path });
        for (const n of g.nodes || g._nodes || []) {
            if (isSubgraphNode(n)) add(n.subgraph, `${path}/${n.subgraph?.name || n.title || n.id}`);
        }
    };
    add(root, "");
    try {
        for (const sg of root?.subgraphs?.values?.() || []) add(sg, `/${sg.name || sg.id}`);
    } catch (_e) { /* no subgraph map */ }
    return out;
}

const keyOf = (root, graph, node) => (graph === root ? String(node.id) : `${graph.id}:${node.id}`);

/** [{key, node, graph, path}] for every node of the workflow (each node object once). */
export function walk(root) {
    const out = [];
    const seen = new Set();
    for (const { graph, path } of allGraphs(root)) {
        for (const node of graph.nodes || graph._nodes || []) {
            if (seen.has(node)) continue;
            seen.add(node);
            out.push({ key: keyOf(root, graph, node), node, graph, path });
        }
    }
    return out;
}

function linkOf(graph, id) {
    if (id == null) return null;
    try {
        if (typeof graph.getLink === "function") return graph.getLink(id) || null;
    } catch (_e) { /* fall through */ }
    const L = graph.links || graph._links;
    return (L?.get ? L.get(id) : L?.[id]) || null;
}

function groupTitleOf(graph, node) {
    const x = (+node.pos?.[0] || 0) + 1;
    const y = (+node.pos?.[1] || 0) + 1;
    let best = null;
    let bestArea = Infinity;
    for (const g of graph.groups || graph._groups || []) {
        const b = g._bounding || g.bounding || [g.pos?.[0], g.pos?.[1], g.size?.[0], g.size?.[1]];
        if (!b || !(x >= b[0] && y >= b[1] && x <= b[0] + b[2] && y <= b[1] + b[3])) continue;
        const area = b[2] * b[3];
        if (area < bestArea || (area === bestArea && String(g.title) < String(best?.title))) { best = g; bestArea = area; }
    }
    return best ? String(best.title || "") : "";
}

function widgetValues(node) {
    const w = {};
    for (const x of node.widgets || []) {
        if (!x?.name || own(w, x.name)) continue;
        const v = x.value;
        if (typeof v === "string") w[x.name] = v.length > 4000 ? v.slice(0, 4000) : v;
        else if (typeof v === "number" || typeof v === "boolean") w[x.name] = v;
    }
    return w;
}

/** The engine snapshot plus a key -> {node, graph} map for writing back. */
export function snapshot(root) {
    const entries = walk(root);
    const byNode = new Map(entries.map((e) => [e.node, e.key]));
    // the first instance of each subgraph definition (walk order) stands for its boundary: a link from the
    // subgraph's input / to its output continues at that instance's input / output in the parent graph
    const wrapper = new Map();
    for (const e of entries) if (isSubgraphNode(e.node) && e.node.subgraph && !wrapper.has(e.node.subgraph)) wrapper.set(e.node.subgraph, e);
    const nodes = [];
    const refs = new Map();
    for (const { key, node, graph, path } of entries) {
        const nd = node.constructor?.nodeData || {};
        const keyIn = (g, id) => {
            const n = id == null ? null : g.getNodeById?.(id);
            return n ? (byNode.get(n) ?? null) : null;
        };
        const ref = (id, slot) => {
            if (id == null) return null;
            if (String(id) === SUBGRAPH_INPUT_ID) {                       // fed from outside the subgraph
                const w = wrapper.get(graph);
                const L = w ? linkOf(w.graph, w.node.inputs?.[slot]?.link) : null;
                return L ? keyIn(w.graph, L.origin_id) : null;
            }
            return keyIn(graph, id);
        };
        const targets = (L) => {                                  // [[key, inputName]] for one output link
            if (String(L.target_id) === SUBGRAPH_OUTPUT_ID) {
                const w = wrapper.get(graph);
                if (!w) return [];
                return (w.node.outputs?.[L.target_slot]?.links || []).map((id) => linkOf(w.graph, id)).filter(Boolean)
                    .map((PL) => [keyIn(w.graph, PL.target_id), w.graph.getNodeById?.(PL.target_id)?.inputs?.[PL.target_slot]?.name]);
            }
            return [[keyIn(graph, L.target_id), graph.getNodeById?.(L.target_id)?.inputs?.[L.target_slot]?.name]];
        };
        nodes.push({
            key, id: node.id, type: String(node.comfyClass || node.type || ""), title: String(node.title ?? ""),
            defaultTitle: defaultTitleOf(node), category: String(nd.category ?? node.category ?? ""),
            outputNode: !!nd.output_node, isVirtual: !!node.isVirtualNode, isSubgraph: isSubgraphNode(node),
            widgets: widgetValues(node),
            inputs: (node.inputs || []).map((i) => {
                const L = linkOf(graph, i.link);
                return { name: String(i.name ?? ""), type: String(i.type ?? ""), from: L ? ref(L.origin_id, L.origin_slot) : null };
            }),
            outputs: (node.outputs || []).map((o) => {
                const pairs = (o.links || []).map((id) => linkOf(graph, id)).filter(Boolean).flatMap(targets)
                    .filter(([k]) => k != null);
                return { name: String(o.name ?? ""), type: String(o.type ?? ""), to: pairs.map(([k]) => k),
                         toInputs: pairs.map(([, n]) => (n == null ? null : String(n))) };
            }),
            pos: [+node.pos?.[0] || 0, +node.pos?.[1] || 0], group: groupTitleOf(graph, node), path,
            srName: node.properties?.["Node name for S&R"] ?? null, identity: identityOf(node),
        });
        refs.set(key, { node, graph });
    }
    return { snap: { nodes }, refs };
}

/** Keys of the selected nodes (canvas selection of the graph on screen). */
export function selectedKeys(app, refs) {
    const want = new Set();
    const c = app.canvas;
    for (const n of Object.values(c?.selected_nodes || {})) want.add(n);
    for (const it of c?.selectedItems || []) if (it && it.id != null && it.type != null && !it.recomputeInsideNodes) want.add(it);
    const keys = new Set();
    for (const [k, r] of refs) if (want.has(r.node)) keys.add(k);
    return keys;
}

/** Keys of the nodes inside the selected group(s); with no group selected, the groups holding the selection. */
export function groupKeys(app, refs) {
    const c = app.canvas;
    let groups = [...(c?.selectedItems || [])].filter((it) => typeof it?.recomputeInsideNodes === "function");
    if (!groups.length) {
        const sel = [...selectedKeys(app, refs)].map((k) => refs.get(k)?.node).filter(Boolean);
        const g = c?.graph;
        for (const grp of g?.groups || g?._groups || []) {
            try { grp.recomputeInsideNodes(); } catch (_e) { /* keep going */ }
            if ((grp.nodes || []).some((n) => sel.includes(n))) groups.push(grp);
        }
    }
    const inside = new Set();
    for (const grp of groups) {
        try { grp.recomputeInsideNodes(); } catch (_e) { /* keep going */ }
        for (const n of grp.nodes || []) inside.add(n);
    }
    const keys = new Set();
    for (const [k, r] of refs) if (inside.has(r.node)) keys.add(k);
    return { keys, groups: groups.length };
}

/** Write one alias with provenance. source: "auto" | "manual". */
export function setAlias(node, alias, source) {
    node.properties ||= {};
    const prev = identityOf(node) || {};
    const original = own(prev, "original") ? prev.original : String(node.title ?? defaultTitleOf(node));
    const next = { v: 1, original, alias: String(alias), source };
    if (prev.title === "show" || prev.title === "hide") next.title = prev.title;
    node.properties[PROP] = next;
    node.title = String(alias);
}

/** Back to the title before the first Node Identity change; the per-node visibility override is kept. */
export function restore(node) {
    const id = identityOf(node);
    if (!id || !own(id, "original")) return false;
    node.title = String(id.original || defaultTitleOf(node));
    if (id.title === "show" || id.title === "hide") node.properties[PROP] = { v: 1, title: id.title };
    else delete node.properties[PROP];
    return true;
}

export function setTitleOverride(node, value) {
    node.properties ||= {};
    const id = { ...(identityOf(node) || { v: 1 }) };
    if (value === "show" || value === "hide") id.title = value;
    else delete id.title;
    if (Object.keys(id).filter((k) => k !== "v").length) node.properties[PROP] = id;
    else delete node.properties[PROP];
}

/** One undo step around fn (ChangeTracker before/after-change), then redraw. */
export function oneStep(app, fn) {
    app.canvas?.emitBeforeChange?.();
    try {
        return fn();
    } finally {
        app.canvas?.emitAfterChange?.();
        app.canvas?.setDirty?.(true, true);
    }
}
