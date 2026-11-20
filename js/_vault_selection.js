// _vault_selection.js - what "Lock selection as C2C Vault" protects, and how its progress and failures read
// (V1, owner 2026-11-20: "lock a selection in one step", docs/h2vault/V1_cnp_vault_ux_spec.md).
//
// Pure logic: no DOM, no import from the ComfyUI frontend (tests run it under node). The frontend 1.53.10 selection
// (`canvas.selectedItems`) is a Set of nodes, groups (LGraphGroup: recomputeInsideNodes() + a `children` Set) and
// native link reroutes; only real nodes may reach the protect transaction.
//
// License: Apache-2.0

const VAULT_TYPES = new Set(["C2C_SecureSubgraph", "C2C_SecureSubgraphOutput", "C2C_VaultLocked", "C2C_VaultSealed"]);

export const PROBLEM_EMPTY = "Select the nodes (or a group) to lock first.";
export const PROBLEM_VAULT = "The selection contains a vault. Lock the nodes around it separately.";
export const PROBLEM_GRAPHS = "Select nodes from one graph (the one you are looking at).";

function isGroupOf(LG) {
  return (x) => {
    if (!x || typeof x !== "object") return false;
    if (LG?.LGraphGroup) return x instanceof LG.LGraphGroup;
    return typeof x.recomputeInsideNodes === "function";
  };
}

function isNodeOf(LG, isGroup) {
  return (x) => {
    if (!x || typeof x !== "object" || isGroup(x)) return false;
    if (LG?.LGraphNode) return x instanceof LG.LGraphNode;
    return typeof x.type === "string" && typeof x.connect === "function"
      && (Array.isArray(x.inputs) || Array.isArray(x.outputs));
  };
}

const isSubgraphNode = (n) => {
  try { return typeof n?.isSubgraphNode === "function" && !!n.isSubgraphNode(); } catch (_) { return false; }
};

/**
 * The nodes a lock acts on.
 * @param {Iterable<object>|undefined} items  canvas.selectedItems (or any iterable of canvas items)
 * @param {{LiteGraph?: object}} [opts]       window.LiteGraph; without it the shapes are duck-typed
 * @returns {{kind: "nodes"|"subgraph"|"empty", nodes: object[], groups: object[], title: string|null,
 *            hasSubgraph: boolean, problems: string[]}}
 */
export function resolveLockSelection(items, { LiteGraph } = {}) {
  const isGroup = isGroupOf(LiteGraph);
  const isNode = isNodeOf(LiteGraph, isGroup);
  const nodes = new Map();
  const groups = [];
  const looseNodes = [];                  // selected directly, not only through a group
  const seenGroups = new Set();
  const addNode = (n, loose) => {
    if (!isNode(n) || n.id == null || Number(n.id) < 0) return;
    const key = String(n.id) + "@" + String(n.graph?.id ?? "");
    if (!nodes.has(key)) nodes.set(key, n);
    if (loose) looseNodes.push(n);
  };
  const walkGroup = (g, depth = 0) => {
    if (seenGroups.has(g) || depth > 32) return;
    seenGroups.add(g);
    try { g.recomputeInsideNodes?.(); } catch (_) { /* keep the last known children */ }
    for (const c of g.children ?? g._children ?? []) {
      if (isGroup(c)) walkGroup(c, depth + 1);
      else addNode(c, false);
    }
  };
  for (const it of items ?? []) {
    if (isGroup(it)) { groups.push(it); walkGroup(it); } else addNode(it, true);
  }
  const list = [...nodes.values()];
  const hasSubgraph = list.some(isSubgraphNode);
  const problems = [];
  if (!list.length) problems.push(PROBLEM_EMPTY);
  else if (list.some((n) => VAULT_TYPES.has(n.type))) problems.push(PROBLEM_VAULT);
  else if (new Set(list.map((n) => n.graph)).size > 1) problems.push(PROBLEM_GRAPHS);
  const kind = !list.length ? "empty" : (list.length === 1 && hasSubgraph ? "subgraph" : "nodes");
  let title = null;
  if (kind === "subgraph") title = list[0].title || null;
  else if (groups.length === 1 && list.length) {
    const inside = new Set();
    for (const c of groups[0].children ?? []) inside.add(c);
    if (looseNodes.every((n) => inside.has(n))) title = groups[0].title || null;
  }
  return { kind, nodes: list, groups, title, hasSubgraph, problems };
}

/**
 * The dialog's selection capsule: titles (max 4) and how many sockets the vault will get.
 * `boundary` is _vault_protect_tx.js captureBoundary() output, or null for one existing subgraph node.
 */
export function summarizeSelection(nodes, boundary) {
  const list = [...(nodes || [])];
  const names = list.map((n) => String(n.title || n.type || "Node"));
  const distinct = (recs) => new Set((recs || []).map((r) => `${r.origin_id}#${r.origin_slot}`)).size;
  let inputs = 0, outputs = 0;
  if (boundary) {
    inputs = distinct(boundary.entering);
    outputs = distinct(boundary.leaving);
  } else if (list.length === 1) {
    inputs = (list[0].inputs || []).length;
    outputs = (list[0].outputs || []).length;
  }
  return { count: list.length, chips: names.slice(0, 4), more: Math.max(0, names.length - 4), inputs, outputs,
    hasSubgraph: list.some(isSubgraphNode) };
}

export const STAGE_LABELS = Object.freeze({
  SELECTION_VALIDATION: "Checking the selection",
  GRAPH_SNAPSHOT: "Saving a restore point",
  BOUNDARY_CAPTURE: "Recording the wires",
  NATIVE_SUBGRAPH_CONVERSION: "Grouping the nodes",
  SUBGRAPH_INTERFACE_VALIDATION: "Checking every wire",
  INNER_GRAPH_EXTRACTION: "Reading the inside",
  VAULT_PAYLOAD_VALIDATION: "Checking the inside",
  ENCRYPTION_AND_SEALING: "Encrypting",
  VAULT_NODE_INSERTION: "Placing the vault",
  EXTERNAL_WIRE_RESTORATION: "Reconnecting the wires",
  FINAL_GRAPH_VALIDATION: "Final check",
  COMMIT: "Done",
});

export function stageLabel(stage) {
  return STAGE_LABELS[stage] ?? "Locking";
}

/** A failed protect in words: what failed, why, and that nothing was changed. */
export function protectFailureText(err) {
  const why = String(err?.message || err || "").trim();
  if (err?.busy) return why;
  return `Not locked. ${stageLabel(err?.stage)} failed: ${why}${/[.!?]$/.test(why) ? "" : "."} ` +
    "Your nodes are back exactly as they were.";
}
