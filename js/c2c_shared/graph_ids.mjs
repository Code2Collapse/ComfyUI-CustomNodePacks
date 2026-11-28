// PORTED FROM: ComfyUI-AusBoss js/shared/graph_ids.mjs (https://github.com/ausboss/ComfyUI-AusBoss), MIT License,
// Copyright (c) 2026 ausboss. Licence text: js/c2c_shared/LICENSE-AUSBOSS.txt. Code unchanged at port (2026-11-28,
// agy review Phase 3 item 12: one shared UI foundation for the C2C packs).
// Resolving ComfyUI execution ids to graph nodes.
//
// Kept free of /scripts/app.js imports so node:test can cover it
// (tests/graph_ids.test.mjs); callers pass the root graph in.

// ComfyUI identifies a node inside a subgraph by its execution id: the chain
// of subgraph node ids that reaches it, colon-joined ("12:3" is node 3 inside
// the subgraph node 12 holds). Only the last segment is a real graph node id,
// and ids are numbered independently inside each subgraph, so looking the
// bare tail up on the root graph does not merely miss - it can return a
// completely unrelated top-level node that happens to share the number.
// Walking the chain is the only way to answer correctly; an id that does not
// resolve returns null, because acting on the wrong node is worse than doing
// nothing at all.
export function nodeByExecutionId(rootGraph, executionId) {
  const parts = String(executionId ?? "")
    .split(":")
    .filter((part) => part !== "");
  if (!parts.length) return null;
  let graph = rootGraph ?? null;
  for (const part of parts.slice(0, -1)) {
    const holder = graph?.getNodeById?.(Number(part)) ?? graph?.getNodeById?.(part);
    graph = holder?.subgraph ?? null;
    if (!graph) return null;
  }
  const tail = parts[parts.length - 1];
  return graph?.getNodeById?.(Number(tail)) ?? graph?.getNodeById?.(tail) ?? null;
}

// The other way round: a node's execution id, found by walking from the root
// graph down through the boxes (subgraph nodes) that hold it. This is what a
// run of one node alone has to name. A box that is used twice gives its first
// use; a node that is on no graph gives null.
export function executionIdOf(rootGraph, node) {
  if (!rootGraph || !node) return null;
  const walk = (graph, prefix, depth) => {
    const nodes = graph?.nodes ?? graph?._nodes ?? [];
    if (nodes.includes(node)) return `${prefix}${node.id}`;
    if (depth > 16) return null;
    for (const holder of nodes) {
      if (!holder?.subgraph) continue;
      const found = walk(holder.subgraph, `${prefix}${holder.id}:`, depth + 1);
      if (found) return found;
    }
    return null;
  };
  return walk(rootGraph, "", 0);
}

// Whether ComfyUI leaves a node out of every run: it is bypassed or muted
// (modes 4 and 2), or a box (subgraph node) on the way to it is. Such a node
// can not be run alone either: a prompt of it has nothing left to run, and
// ComfyUI refuses it with "Prompt has no outputs". A node that is on no graph
// counts only its own mode.
const OFF_MODES = new Set([2, 4]);
export function isSwitchedOff(rootGraph, node) {
  if (!node) return false;
  if (OFF_MODES.has(node.mode)) return true;
  if (!rootGraph) return false;
  const holdersOf = (graph, depth) => {
    const nodes = graph?.nodes ?? graph?._nodes ?? [];
    if (nodes.includes(node)) return [];
    if (depth > 16) return null;
    for (const holder of nodes) {
      if (!holder?.subgraph) continue;
      const found = holdersOf(holder.subgraph, depth + 1);
      if (found) return [holder, ...found];
    }
    return null;
  };
  return (holdersOf(rootGraph, 0) ?? []).some((holder) => OFF_MODES.has(holder.mode));
}
