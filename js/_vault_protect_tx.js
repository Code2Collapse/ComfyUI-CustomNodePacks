// _vault_protect_tx.js - "Protect selection" as an atomic transaction (owner directive A26, ledger L9.203).
//
// The old flow converted the selection to a native subgraph, then protected it; any failure after the conversion left
// an ordinary subgraph in the workflow ("nothing lost"), and wires coming from a containing subgraph's input node were
// re-attached by "first unlinked input of the same type" - the frontend's own convertToSubgraph does the same
// (`findInputSlotByType`, frontend 1.53.10) and silently skips links it cannot resolve (console.warn only).
//
// Here every stage is named; the whole workflow is snapshotted first and restored on ANY failure with the same call
// the frontend's undo uses (ChangeTracker.updateState -> loadGraphData(state, false, false, workflow, ...)). Boundary
// links are captured as a manifest of stable identities BEFORE conversion and checked after it: a link is repaired
// only when the manifest names exactly one place for it, otherwise the transaction aborts. Nothing is inferred from
// socket types or order, and inner nodes are found by their exact execution path (the frontend's own algorithm,
// findPartialExecutionPathToGraph / createExecutionIdFromPath in 1.53.10), never by the last segment of an id.

export const STAGES = Object.freeze([
  "SELECTION_VALIDATION", "GRAPH_SNAPSHOT", "BOUNDARY_CAPTURE", "NATIVE_SUBGRAPH_CONVERSION",
  "SUBGRAPH_INTERFACE_VALIDATION", "INNER_GRAPH_EXTRACTION", "VAULT_PAYLOAD_VALIDATION", "ENCRYPTION_AND_SEALING",
  "VAULT_NODE_INSERTION", "EXTERNAL_WIRE_RESTORATION", "FINAL_GRAPH_VALIDATION", "COMMIT",
]);
export const ADAPTER = "fe-1.53 execution-path adapter v1";

export class ProtectError extends Error {
  constructor(stage, message) {
    super(message);
    this.name = "ProtectError";
    this.stage = stage;
  }
}

const isSub = (n) => { try { return typeof n?.isSubgraphNode === "function" && n.isSubgraphNode(); } catch (_) { return false; } };
const nodesOf = (g) => g?.nodes || g?._nodes || [];

/** Run `fn` with `graph` open on the canvas: frontend 1.53.10 records nothing for connects on a subgraph that is not
 *  the open graph (ledger L9.100). Returns to the previous graph afterwards. */
function inGraph(graph, fn) {
  const canvas = globalThis.comfyAPI?.app?.app?.canvas ?? globalThis.LGraphCanvas?.active_canvas;
  const prev = canvas?.graph;
  const swap = canvas && prev && prev !== graph;
  if (swap) canvas.setGraph(graph);
  try { return fn(); } finally { if (swap) canvas.setGraph(prev); }
}

export function linkOf(graph, id) {
  if (id == null || !graph) return null;
  const L = graph.links ?? graph._links;
  return (L instanceof Map ? L.get(id) : L?.[id]) ?? null;
}

// ── execution paths (mirrors frontend 1.53.10 getExecutionIdByNode) ─────────────────────────────────────────────
function pathToGraph(target, from) {
  for (const n of nodesOf(from)) {
    if (!isSub(n)) continue;
    if (n.subgraph === target) return `${n.id}`;
    const deeper = pathToGraph(target, n.subgraph);
    if (deeper !== undefined) return `${n.id}:${deeper}`;
  }
  return undefined;
}

/** The execution id of `node` as graphToPrompt names it ("7", "12:7", ...), or null. */
export function execIdOf(node, root) {
  const g = node?.graph;
  if (!g) return null;
  if (g === root || g.isRootGraph) return String(node.id);
  const p = pathToGraph(g, root);
  return p === undefined ? null : `${p}:${node.id}`;
}

/** Where an output (node, slot) really comes from, as an execution path RELATIVE to the graph holding `node`
 *  ("7", or "12:7" through subgraph node 12), following subgraph outputs inward. Null when not exactly resolvable. */
export function relSource(node, slot, depth = 0) {
  if (!node || depth > 16) return null;
  if (!isSub(node)) return { id: String(node.id), slot };
  let r = null;
  try { r = node.resolveSubgraphOutputLink?.(slot); } catch (_) { r = null; }
  const inner = r?.outputNode ?? r?.node ?? null;
  const innerSlot = r?.link?.origin_slot ?? r?.slot ?? r?.outputSlot;
  if (!inner || innerSlot == null) return null;
  const d = relSource(inner, innerSlot, depth + 1);
  return d ? { id: `${node.id}:${d.id}`, slot: d.slot } : null;
}

/** The node at an execution path ("12:7") starting from the root graph. */
export function nodeByExecPath(root, path) {
  let g = root, n = null;
  const parts = String(path).split(":").filter((x) => x.length);
  for (let i = 0; i < parts.length; i++) {
    n = g?.getNodeById?.(parts[i]) ?? nodesOf(g).find((x) => String(x.id) === parts[i]) ?? null;
    if (!n) return null;
    if (i < parts.length - 1) g = isSub(n) ? n.subgraph : null;
  }
  return n;
}

/**
 * The absolute execution source (graphToPrompt key + slot) of a link `l` inside `graph`. `graphPath` is the
 * execution path of the subgraph node whose inside `graph` is ("" = the root graph). A link from the graph's own
 * input node is followed out through THAT subgraph node - never through "the first" instance of a shared definition.
 */
export function absSource(root, graph, l, graphPath, depth = 0) {
  if (!l || depth > 16) return null;
  const base = graphPath ? `${graphPath}:` : "";
  const ioIn = graph?.inputNode?.id;
  if (ioIn != null && String(l.origin_id) === String(ioIn)) {
    if (!graphPath) return null;
    const holder = nodeByExecPath(root, graphPath);              // the subgraph node whose inside `graph` is
    const outer = holder && linkOf(holder.graph, holder.inputs?.[l.origin_slot]?.link);
    const outerPath = graphPath.split(":").slice(0, -1).join(":");
    return outer ? absSource(root, holder.graph, outer, outerPath, depth + 1) : null;
  }
  let origin = graph.getNodeById?.(l.origin_id) ?? nodesOf(graph).find((x) => String(x.id) === String(l.origin_id));
  let slot = l.origin_slot;
  // virtual nodes (Reroute, Set/Get-style nodes) are resolved through by graphToPrompt: follow them the same way
  if (origin?.isVirtualNode && !isSub(origin)) {
    const rv = origin.resolveVirtualOutput?.(slot);
    if (rv?.node) {
      origin = rv.node;
      slot = rv.slot ?? 0;
    } else {
      const up = origin.getInputLink?.(0);
      return up ? absSource(root, graph, up, graphPath, depth + 1) : null;
    }
  }
  const rs = relSource(origin, slot);
  return rs ? { id: `${base}${rs.id}`, slot: rs.slot } : null;
}

// ── snapshot / restore ───────────────────────────────────────────────────────────────────────────────────────
export function snapshot(app) {
  const root = app.rootGraph ?? app.graph;
  const canvas = app.canvas;
  const inside = canvas?.graph && canvas.graph !== root ? canvas.graph.id : null;
  return { state: JSON.parse(JSON.stringify(root.serialize())), inside };
}

export async function restore(app, snap) {
  const wf = app.extensionManager?.workflow?.activeWorkflow ?? null;
  await app.loadGraphData(JSON.parse(JSON.stringify(snap.state)), false, false, wf,
    { checkForRerouteMigration: false, silentAssetErrors: true });
  if (snap.inside) {                                           // back into the subgraph the user was editing
    const root = app.rootGraph ?? app.graph;
    const sub = root.subgraphs?.get?.(snap.inside);
    if (sub) { try { app.canvas.setGraph(sub); } catch (_) { /* stays at root: the graph itself is restored */ } }
  }
}

// ── boundary manifest ────────────────────────────────────────────────────────────────────────────────────────
/** Every link that crosses the selection, by stable identity (graph, node id, slot index + name + type). */
export function captureBoundary(graph, nodes) {
  const sel = new Set(nodes.map((n) => String(n.id)));
  const io = { in: graph.inputNode?.id, out: graph.outputNode?.id };
  const entering = [], leaving = [], internal = [];
  for (const n of nodes) {
    (n.inputs || []).forEach((inp, i) => {
      const l = linkOf(graph, inp?.link);
      if (!l) return;
      const rec = { target_id: String(n.id), target_input: i, target_name: inp.name, type: String(inp.type ?? l.type ?? "*"),
        widget: inp.widget?.name ?? null, origin_id: String(l.origin_id), origin_slot: l.origin_slot,
        origin_is_io: io.in != null && String(l.origin_id) === String(io.in) };
      if (sel.has(String(l.origin_id))) internal.push(rec); else entering.push(rec);
    });
    (n.outputs || []).forEach((out, j) => (out?.links || []).forEach((lid) => {
      const l = linkOf(graph, lid);
      if (!l || sel.has(String(l.target_id))) return;
      leaving.push({ origin_id: String(n.id), origin_slot: j, origin_name: out.name, type: String(out.type ?? l.type ?? "*"),
        target_id: String(l.target_id), target_input: l.target_slot,
        target_is_io: io.out != null && String(l.target_id) === String(io.out) });
    }));
  }
  // nested subgraph nodes: how many of their definition's output links resolve now (frontend 1.53.10 loses them when
  // a selection that CONTAINS a subgraph node is converted - ledger L9.100; checked after conversion)
  const nested = nodes.filter(isSub).map((n) => ({ id: String(n.id), title: n.title || n.type,
    outs: (n.subgraph?.outputs || []).map((so) => (so.linkIds || []).filter((lid) => linkOf(n.subgraph, lid)).length) }));
  return { graph_id: graph.id ?? "root", selected: [...sel], types: Object.fromEntries(nodes.map((n) => [String(n.id), n.type])),
    entering, leaving, internal, nested };
}

/**
 * Frontend issue #23101 (reproduced on 1.53.10 with ResizeImageMaskNode, 2026-11-12): converting a node with
 * DynamicCombo sub-inputs moves their inner links one slot down (resize_type.width -> resize_type.height, height ->
 * crop). The sub-input NAME is a stable identity, so a link is put back on the inner input of the recorded name -
 * but only when exactly one subgraph input carries that name and it feeds the same inner node; otherwise nothing is
 * touched and the caller aborts. Returns the number of links repaired.
 */
export function repairShiftedInnerLinks(sg, manifest) {
  const sub = sg?.subgraph;
  if (!sub) return 0;
  const plan = [];
  const toNode = (si, id) => (si.linkIds || []).map((lid) => linkOf(sub, lid)).filter((L) => L && String(L.target_id) === id);
  for (const rec of manifest.entering) {
    const inner = sub.getNodeById?.(rec.target_id);
    const want = inner ? (inner.inputs || []).findIndex((q) => q.name === rec.target_name) : -1;
    if (want < 0) return 0;
    // the subgraph input that belongs to this record: by its NAME (stable), else the one wired to the exact slot
    const named = (sub.inputs || []).map((si, k) => [si, k]).filter(([si]) => si.name === rec.target_name
      && toNode(si, rec.target_id).length);
    const bySlot = (sub.inputs || []).map((si, k) => [si, k]).filter(([si]) => toNode(si, rec.target_id)
      .some((L) => L.target_slot === want));
    const pick = named.length === 1 ? named[0] : (named.length === 0 && bySlot.length === 1 ? bySlot[0] : null);
    if (!pick) return 0;
    const [si, k] = pick;
    const links = toNode(si, rec.target_id);
    if (links.length !== 1) return 0;
    if (links[0].target_slot !== want) plan.push({ k, inner, want, wrongSlot: links[0].target_slot });
  }
  // disconnect every wrong link first (they occupy each other's slots), then connect by name - with the subgraph open
  if (plan.length) inGraph(sub, () => {
    for (const p of plan) p.inner.disconnectInput(p.wrongSlot, true);
    for (const p of plan) {
      if (p.inner.inputs[p.want]?.link != null) p.inner.disconnectInput(p.want, true);
      sub.inputNode.slots[p.k].connect(p.inner.inputs[p.want], p.inner);
    }
  });
  return plan.length;
}

/** #23101 for links between two selected nodes: the link from (origin, slot) into the target node sits on the wrong
 *  slot; move each one to the inner input of the recorded NAME (two phases, as above). Ambiguity: untouched. */
function repairShiftedInternalLinks(sub, manifest) {
  const plan = [];
  for (const rec of manifest.internal || []) {
    const n = sub.getNodeById?.(rec.target_id);
    const origin = sub.getNodeById?.(rec.origin_id);
    if (!n || !origin) continue;
    const want = (n.inputs || []).findIndex((q) => q.name === rec.target_name);
    if (want < 0) continue;
    const from = (n.inputs || []).map((q, i) => [i, linkOf(sub, q.link)])
      .filter(([, L]) => L && String(L.origin_id) === rec.origin_id && L.origin_slot === rec.origin_slot);
    if (from.length !== 1 || from[0][0] === want) continue;
    plan.push({ n, origin, slot: rec.origin_slot, wrong: from[0][0], want });
  }
  if (plan.length) inGraph(sub, () => {
    for (const p of plan) p.n.disconnectInput(p.wrong, true);
    for (const p of plan) p.origin.connect(p.slot, p.n, p.want);
  });
  return plan.length;
}

/** Index of the subgraph input / output that the manifest record maps to: exactly one, or a ProtectError. */
function subgraphInputFor(sg, rec) {
  const sub = sg.subgraph;
  const hits = [];
  (sub?.inputs || []).forEach((si, k) => {
    if ((si.linkIds || []).some((lid) => {
      const L = linkOf(sub, lid);
      return L && String(L.target_id) === rec.target_id && L.target_slot === rec.target_input;
    })) hits.push(k);
  });
  if (hits.length !== 1) {
    throw new ProtectError("SUBGRAPH_INTERFACE_VALIDATION", `The input "${rec.target_name}" of node ${rec.target_id} ` +
      `maps to ${hits.length} subgraph inputs after conversion (expected exactly one); nothing was changed.`);
  }
  return hits[0];
}

function subgraphOutputFor(sg, rec) {
  const sub = sg.subgraph;
  const hits = [];
  (sub?.outputs || []).forEach((so, j) => {
    if ((so.linkIds || []).some((lid) => {
      const L = linkOf(sub, lid);
      return L && String(L.origin_id) === rec.origin_id && L.origin_slot === rec.origin_slot;
    })) hits.push(j);
  });
  if (hits.length !== 1) {
    throw new ProtectError("SUBGRAPH_INTERFACE_VALIDATION", `The output "${rec.origin_name}" of node ${rec.origin_id} ` +
      `maps to ${hits.length} subgraph outputs after conversion (expected exactly one); nothing was changed.`);
  }
  return hits[0];
}

/**
 * After convertToSubgraph: every selected node is inside with its id and type, and every boundary link of the
 * manifest is wired to the ONE subgraph socket that leads to its original inner socket. A missing wire from the
 * containing subgraph's own input / output node (frontend 1.53 drops or mis-places those) is re-made on that exact
 * socket; any other difference aborts. Returns the {entering: [k...], leaving: [j...]} socket map.
 */
export function verifyConversion(graph, sg, manifest) {
  const sub = sg?.subgraph;
  if (!sub) throw new ProtectError("NATIVE_SUBGRAPH_CONVERSION", "The selection did not become a subgraph.");
  for (const id of manifest.selected) {
    const n = sub.getNodeById?.(id) ?? nodesOf(sub).find((x) => String(x.id) === id);
    if (!n || n.type !== manifest.types[id]) {
      throw new ProtectError("SUBGRAPH_INTERFACE_VALIDATION", `Node ${id} (${manifest.types[id]}) is not inside the new ` +
        "subgraph with the same id and type; nothing was changed.");
    }
  }
  for (const rec of manifest.nested || []) {
    const inner = sub.getNodeById?.(rec.id);
    const now = (inner?.subgraph?.outputs || []).map((so) => (so.linkIds || []).filter((lid) => linkOf(inner.subgraph, lid)).length);
    if (rec.outs.some((c, j) => (now[j] ?? 0) < c)) {
      throw new ProtectError("SUBGRAPH_INTERFACE_VALIDATION", `This ComfyUI frontend lost links inside the nested subgraph ` +
        `"${rec.title}" while converting the selection (a known frontend issue with selections that contain a subgraph). ` +
        "Protect that subgraph on its own, or unpack it first (right-click > Unpack Subgraph) and protect again.");
    }
  }
  // links between selected nodes must all be inside the new subgraph (shifted dynamic inputs put back by name first)
  repairShiftedInternalLinks(sub, manifest);
  for (const rec of manifest.internal || []) {
    const n = sub.getNodeById?.(rec.target_id);
    const want = n ? (n.inputs || []).findIndex((q) => q.name === rec.target_name) : -1;
    const l = n && linkOf(sub, n.inputs?.[want]?.link);
    if (!l || String(l.origin_id) !== rec.origin_id || l.origin_slot !== rec.origin_slot) {
      throw new ProtectError("SUBGRAPH_INTERFACE_VALIDATION", `The wire into "${rec.target_name}" of node ${rec.target_id} ` +
        "inside the selection did not survive the conversion; nothing was changed.");
    }
  }
  const repaired = repairShiftedInnerLinks(sg, manifest);
  if (repaired) console.info(`[C2C Vault] put ${repaired} shifted dynamic-input link(s) back by name (frontend #23101)`);
  const inMap = [], outMap = [];
  for (const rec of manifest.entering) {
    const k = subgraphInputFor(sg, rec);
    inMap.push(k);
    const l = linkOf(graph, sg.inputs?.[k]?.link);
    const ok = l && String(l.origin_id) === rec.origin_id && l.origin_slot === rec.origin_slot;
    if (ok) continue;
    if (!l && rec.origin_is_io) {                                 // the 1.53 drop: re-make it on the exact socket
      graph.inputNode?.slots?.[rec.origin_slot]?.connect(sg.inputs[k], sg);
      const again = linkOf(graph, sg.inputs?.[k]?.link);
      if (again && String(again.origin_id) === rec.origin_id && again.origin_slot === rec.origin_slot) continue;
    }
    throw new ProtectError("SUBGRAPH_INTERFACE_VALIDATION", `The wire into "${rec.target_name}" (node ${rec.target_id}) ` +
      "did not survive the conversion and cannot be restored unambiguously; nothing was changed.");
  }
  for (const rec of manifest.leaving) {
    const j = subgraphOutputFor(sg, rec);
    outMap.push(j);
    const linked = (sg.outputs?.[j]?.links || []).some((lid) => {
      const L = linkOf(graph, lid);
      return L && String(L.target_id) === rec.target_id && L.target_slot === rec.target_input;
    });
    if (linked) continue;
    if (rec.target_is_io) {
      graph.outputNode?.slots?.[rec.target_input]?.connect(sg.outputs[j], sg);
      const again = (sg.outputs?.[j]?.links || []).some((lid) => {
        const L = linkOf(graph, lid);
        return L && String(L.target_id) === rec.target_id && L.target_slot === rec.target_input;
      });
      if (again) continue;
    }
    throw new ProtectError("SUBGRAPH_INTERFACE_VALIDATION", `The wire from "${rec.origin_name}" (node ${rec.origin_id}) ` +
      "did not survive the conversion and cannot be restored unambiguously; nothing was changed.");
  }
  return { entering: inMap, leaving: outMap };
}

/** The sockets of `node` as {origin|target, slot} lists, for comparing a subgraph node with its vault. */
export function outerWires(graph, node) {
  const ins = (node.inputs || []).map((inp, i) => {
    const l = linkOf(graph, inp?.link);
    return l ? { i, origin: String(l.origin_id), slot: l.origin_slot } : null;
  }).filter(Boolean);
  const outs = [];
  (node.outputs || []).forEach((o, j) => (o?.links || []).forEach((lid) => {
    const l = linkOf(graph, lid);
    if (l) outs.push({ j, target: String(l.target_id), slot: l.target_slot });
  }));
  return { ins, outs };
}

/**
 * FINAL_GRAPH_VALIDATION: the vault carries exactly the outer wires the subgraph had - each subgraph input wire on
 * the vault input that `vaultSlotOf` names (promoted inputs become controls and carry no wire), each output wire on
 * the same output index - and nothing else.
 */
export function verifyVaultWiring(graph, vault, before, vaultSlotOf, promotedIdx = new Set()) {
  const after = outerWires(graph, vault);
  const key = (w) => `${w.i ?? w.j}|${w.origin ?? w.target}|${w.slot}`;
  const expectIns = before.ins.filter((w) => !promotedIdx.has(w.i)).map((w) => ({ ...w, i: vaultSlotOf(w.i) }));
  const want = new Set([...expectIns.map(key), ...before.outs.map(key)]);
  const have = new Set([...after.ins.map(key), ...after.outs.map(key)]);
  const missing = [...want].filter((k) => !have.has(k));
  const extra = [...have].filter((k) => !want.has(k));
  if (expectIns.some((w) => w.i < 0) || missing.length || extra.length) {
    throw new ProtectError("FINAL_GRAPH_VALIDATION", `The vault's wires differ from the subgraph's (missing ` +
      `${missing.length}, unexpected ${extra.length}); the original graph was restored.`);
  }
}

/**
 * Run `steps` (in STAGES order) with the graph snapshotted first; any throw restores the snapshot and rethrows a
 * ProtectError naming the stage. `ctx.stage` is updated as stages start so errors from helpers are attributed.
 */
export async function runTransaction(app, steps) {
  const ctx = { stage: "GRAPH_SNAPSHOT" };
  const snap = snapshot(app);
  try {
    for (const [stage, fn] of steps) {
      ctx.stage = stage;
      await fn(ctx);
    }
    ctx.stage = "COMMIT";
    return ctx;
  } catch (e) {
    const stage = e?.stage || ctx.stage;
    try {
      await restore(app, snap);
    } catch (re) {
      throw new ProtectError(stage, `${e?.message || e} - and restoring the original graph FAILED (${re?.message || re}); ` +
        "use Undo or reload the workflow.");
    }
    throw e instanceof ProtectError ? e : new ProtectError(stage, String(e?.message || e));
  }
}
