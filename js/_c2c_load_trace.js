/**
 * Which Load Video a node's frames come from, on the LIVE graph (L9.55, owner 2026-10-25) - the front-end twin of
 * nodes/c2c_video/handles_io.py resolve_follow(), so Save Video can show what "auto (from Load)" will resolve to
 * before a run. Same rule: the primary Load is reached through each node's FIRST linked image input (never execution
 * order); every Load reachable through any image input is a candidate. Crosses into subgraph nodes and out through a
 * subgraph's input node. Ids are execution ids ("12", or "45:12" inside subgraph node 45), as in the API prompt, so a
 * pick made here names the same Load the server follows.
 */

export const LOADER_TYPES = new Set(["LoadVideoC2C", "LoadVideoPathC2C"]);
const MAX_HOPS = 256;

function linkOf(graph, id) {
    if (id == null || !graph) return null;
    const L = graph.links ?? graph._links;
    return (L instanceof Map ? L.get(id) : L?.[id]) ?? null;
}

const hasKind = (t, kind) => String(t ?? "").split(",").map((s) => s.trim()).includes(kind);

/** Input indices of `kind` (then wildcard), in order - widget sockets excluded. */
function typedInputs(node, kind) {
    const typed = [], wild = [];
    (node.inputs || []).forEach((inp, i) => {
        if (!inp || inp.widget) return;
        if (hasKind(inp.type, kind)) typed.push(i);
        else if (String(inp.type).trim() === "*") wild.push(i);
    });
    return typed.concat(wild);
}

/** subgraph -> {graph, node} (its first instance), for every subgraph under the root. */
function parentsOf(root) {
    const map = new Map();
    const walk = (g, depth) => {
        if (!g || depth > 16) return;
        for (const n of g.nodes || g._nodes || []) {
            if (n?.subgraph && !map.has(n.subgraph)) {
                map.set(n.subgraph, { graph: g, node: n });
                walk(n.subgraph, depth + 1);
            }
        }
    };
    walk(root, 0);
    return map;
}

function execId(graph, node, parents) {
    const parts = [String(node.id)];
    let g = graph;
    for (let d = 0; d < 16 && parents.get(g); d++) {
        const p = parents.get(g);
        parts.unshift(String(p.node.id));
        g = p.graph;
    }
    return parts.join(":");
}

/** The node + output slot feeding `node.inputs[i]`, through subgraph boundaries; null when unwired. */
function origin(graph, node, i, parents) {
    const l = linkOf(graph, node.inputs?.[i]?.link);
    if (!l) return null;
    return resolve(graph, l.origin_id, l.origin_slot, parents, 0);
}

function resolve(graph, originId, slot, parents, depth) {
    if (depth > 32) return null;
    const io = graph?.inputNode;
    if (io && String(originId) === String(io.id)) {               // a subgraph's input: out to the parent
        const p = parents.get(graph);
        if (!p) return null;
        return origin(p.graph, p.node, slot, parents);
    }
    const n = graph?.getNodeById?.(originId);
    if (!n) return null;
    if (n.subgraph) {                                             // a subgraph node: in to what feeds that output
        const sub = n.subgraph;
        const lid = sub.outputs?.[slot]?.linkIds?.[0];
        const l = linkOf(sub, lid);
        if (!l) return null;
        if (!parents.has(sub)) parents.set(sub, { graph, node: n });
        return resolve(sub, l.origin_id, l.origin_slot, parents, depth + 1);
    }
    return { graph, node: n, slot };
}

const describe = (o, parents) => ({
    id: execId(o.graph, o.node, parents),
    node: o.node,
    title: o.node.title || o.node.type,
    video: o.node.widgets?.find((w) => w.name === "video")?.value ?? "",
});

/**
 * -> {primary: {id, node, title, video} | null, candidates: [...], ambiguous, why}
 * `inputName` = the node's frames input ("images"; Save Video's stream path uses "latent").
 */
export function traceLoads(app, node, inputName = "images", kind = "IMAGE") {
    const root = app.rootGraph ?? app.graph;
    const parents = parentsOf(root);
    const graph = node.graph ?? root;
    const start = (node.inputs || []).findIndex((x) => x.name === inputName);
    const out = { primary: null, candidates: [], ambiguous: false, why: "" };
    if (start < 0) return out;
    // primary: first linked image input at every node
    let o = origin(graph, node, start, parents), why = `${inputName} is not wired`;
    for (let hop = 0; o && hop < MAX_HOPS; hop++) {
        if (LOADER_TYPES.has(o.node.type)) {
            if (o.slot === 0) out.primary = describe(o, parents);
            else why = `Load Video output ${o.slot} is not its frames`;
            break;
        }
        const next = typedInputs(o.node, kind).find((i) => linkOf(o.graph, o.node.inputs[i]?.link));
        if (next === undefined) { why = `no Load Video upstream (stopped at ${o.node.type})`; break; }
        o = origin(o.graph, o.node, next, parents);
    }
    // candidates: any image input
    const seen = new Set(), found = new Map();
    const stack = [origin(graph, node, start, parents)];
    while (stack.length && seen.size < MAX_HOPS) {
        const c = stack.pop();
        if (!c) continue;
        const key = execId(c.graph, c.node, parents);
        if (seen.has(key)) continue;
        seen.add(key);
        if (LOADER_TYPES.has(c.node.type)) {
            if (c.slot === 0 && !found.has(key)) found.set(key, describe(c, parents));
            continue;
        }
        const ins = typedInputs(c.node, kind).map((i) => origin(c.graph, c.node, i, parents)).filter(Boolean);
        stack.push(...ins.reverse());
    }
    out.candidates = [...found.values()];
    if (out.primary && !found.has(out.primary.id)) out.candidates.unshift(out.primary);
    out.ambiguous = out.candidates.length > 1;
    out.why = out.primary ? "" : why;
    return out;
}

/** What Save follows for a `follow_load` value: the pick when it is still upstream, else the automatic choice. */
export function followed(trace, follow) {
    const want = String(follow ?? "auto").trim();
    if (want && want !== "auto") {
        const hit = trace.candidates.find((c) => c.id === want);
        if (hit) return { load: hit, chosen: true, stale: false };
        return { load: autoLoad(trace), chosen: false, stale: true };
    }
    return { load: autoLoad(trace), chosen: false, stale: false };
}

function autoLoad(trace) {
    if (trace.primary) return trace.primary;
    return trace.candidates.length === 1 ? trace.candidates[0] : null;
}
