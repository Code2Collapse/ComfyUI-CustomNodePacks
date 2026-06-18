// FILE: js/_subgraph_walk.js
// Shared helper: subgraph-aware node lookup.
//
// Why
// ───
//   ComfyUI 0.2x introduced inline subgraphs. The frontend exposes the
//   ROOT graph as `app.graph`, but each SubgraphNode wrapper carries its
//   own nested LGraph at `node.subgraph`. Server-side execution events
//   (`progress`, `executing`, `executed`, custom events that include a
//   `node` / `node_id` field) refer to nodes by either:
//     - a plain integer id (top-level node in the root graph), or
//     - a composite path id `"7:5"`, `"7:5:3"`, ... (parent : child : ...)
//       when the node lives inside one or more subgraph instances.
//
//   `app.graph.getNodeById(id)` only walks the root graph and returns
//   `null` for any node nested inside a subgraph. Overlays that key off
//   server-emitted ids therefore silently disappear when the user (or
//   anyone) drops the same node into a subgraph.
//
// What this exports
// ─────────────────
//   findNodeAnywhere(idLike)
//       Resolves an id (string, number, or composite "a:b:c") to
//       { node, graph, path: [int...], wrapperChain: [SubgraphNode...] }
//       or null. `graph` is the LGraph that owns `node`. `wrapperChain`
//       is the list of SubgraphNode wrappers from root → leaf parent
//       (empty for top-level nodes); useful for bubbling overlays up to
//       the wrapper visible in whichever canvas the user is looking at.
//
//   forAllNodes(fn)
//       Iterates every node across the root graph and every nested
//       subgraph. Calls fn(node, graph). Stops if fn returns `true`.
//
//   dirtyAllGraphs()
//       Marks the root graph AND every loaded subgraph dirty so the
//       canvas redraws regardless of which one the user has open.
//
//   isSubgraphNode(node)
//       Robust subgraph-wrapper detection across frontend versions.

import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

export function isSubgraphNode(node) {
    if (!node) return false;
    try { if (typeof node.isSubgraphNode === "function" && node.isSubgraphNode()) return true; } catch (__c2cErr) { __c2cReport("_subgraph_walk", __c2cErr); }
    const t = node.type || node.comfyClass || "";
    if (t === "Subgraph" || t === "SubgraphNode" || t === "graph/subgraph") return true;
    if (node.subgraph && typeof node.subgraph === "object") return true;
    if (node.subgraphId != null || node.subgraph_id != null) return true;
    return false;
}

function _innerGraph(node) {
    // The actual nested LGraph — different builds use different attrs.
    if (!node) return null;
    if (node.subgraph && typeof node.subgraph === "object" && node.subgraph._nodes !== undefined) {
        return node.subgraph;
    }
    if (node.graph_data && node.graph_data._nodes !== undefined) return node.graph_data;
    // Newer frontend: subgraph definitions live on the workflow; the
    // node carries a `subgraphId` and the resolved instance is cached
    // on `node._subgraphInstance` once the canvas has drawn it.
    if (node._subgraphInstance && node._subgraphInstance._nodes !== undefined) {
        return node._subgraphInstance;
    }
    return null;
}

function _nodesOf(graph) {
    if (!graph) return [];
    if (Array.isArray(graph._nodes)) return graph._nodes;
    if (Array.isArray(graph.nodes))  return graph.nodes;
    if (graph._nodes_by_id) return Object.values(graph._nodes_by_id);
    return [];
}

function _getById(graph, id) {
    if (!graph) return null;
    const numId = Number(id);
    if (typeof graph.getNodeById === "function") {
        const n = graph.getNodeById(numId);
        if (n) return n;
    }
    if (graph._nodes_by_id && graph._nodes_by_id[numId]) return graph._nodes_by_id[numId];
    if (graph._nodes_by_id && graph._nodes_by_id[String(id)]) return graph._nodes_by_id[String(id)];
    for (const n of _nodesOf(graph)) {
        if (n && (n.id === numId || String(n.id) === String(id))) return n;
    }
    return null;
}

export function findNodeAnywhere(idLike) {
    if (idLike == null || idLike === "") return null;
    const root = app.graph;
    if (!root) return null;

    const raw = String(idLike);
    // Composite id like "7:5:3" — walk parent → child.
    if (raw.includes(":")) {
        const parts = raw.split(":");
        let graph = root;
        const wrapperChain = [];
        let node = null;
        for (let i = 0; i < parts.length; i++) {
            const piece = parts[i];
            node = _getById(graph, piece);
            if (!node) return null;
            if (i < parts.length - 1) {
                const inner = _innerGraph(node);
                if (!inner) return null;
                wrapperChain.push(node);
                graph = inner;
                node = null;
            }
        }
        if (!node) return null;
        return { node, graph, path: parts.map(Number), wrapperChain };
    }

    // Plain id — try root first, then DFS into every subgraph.
    const direct = _getById(root, raw);
    if (direct) return { node: direct, graph: root, path: [Number(raw)], wrapperChain: [] };

    const stack = [{ graph: root, chain: [] }];
    while (stack.length) {
        const { graph, chain } = stack.pop();
        for (const n of _nodesOf(graph)) {
            if (!n) continue;
            if (String(n.id) === raw) {
                return { node: n, graph, path: [...chain.map(c => c.id), n.id], wrapperChain: chain };
            }
            const inner = _innerGraph(n);
            if (inner) stack.push({ graph: inner, chain: [...chain, n] });
        }
    }
    return null;
}

export function forAllNodes(fn) {
    const root = app.graph;
    if (!root) return;
    const stack = [root];
    const seen = new Set();
    while (stack.length) {
        const g = stack.pop();
        if (!g || seen.has(g)) continue;
        seen.add(g);
        for (const n of _nodesOf(g)) {
            if (!n) continue;
            if (fn(n, g) === true) return;
            const inner = _innerGraph(n);
            if (inner) stack.push(inner);
        }
    }
}

export function dirtyAllGraphs() {
    const seen = new Set();
    const mark = (g) => {
        if (!g || seen.has(g)) return;
        seen.add(g);
        try { g.setDirtyCanvas?.(true, true); } catch (__c2cErr) { __c2cReport("_subgraph_walk", __c2cErr); }
    };
    mark(app.graph);
    forAllNodes((n) => {
        const inner = _innerGraph(n);
        if (inner) mark(inner);
    });
    // Also redraw the active canvas explicitly, in case the user is
    // currently inside a subgraph view that isn't `app.graph`.
    try { app.canvas?.setDirty?.(true, true); } catch (__c2cErr) { __c2cReport("_subgraph_walk", __c2cErr); }
    try {
        const list = app.graph?.list_of_graphcanvas || app.canvas?.constructor?.active_canvas
            ? (app.graph.list_of_graphcanvas || [app.canvas])
            : [];
        for (const c of list || []) c?.setDirty?.(true, true);
    } catch (__c2cErr) { __c2cReport("_subgraph_walk", __c2cErr); }
}
