// C2C Vault: turn a canvas selection into the vault payload (nodes, links, boundary) - including selections made
// INSIDE a subgraph and selections that CONTAIN subgraph nodes (owner 2026-10-14: everything must work in subgraphs,
// ledger L2.45 / L2.46).
//
// Pure: no app, no DOM. It reads only the LiteGraph objects it is given, so tests run it in node against mock
// graphs (tests/test_vault_flatten.py).
//
// What it handles that the old deriveBoundary did not:
//   - links from a subgraph's INPUT node (origin "-10") and to its OUTPUT node (target "-20"): the old code looked
//     those ids up with getNodeById, found nothing, and the vault lost the wire (measured: a vault locked inside a
//     subgraph had no output link and the run never started);
//   - subgraph nodes inside the selection: their inner nodes are flattened into the payload (ids "<sg>:<inner>",
//     recursively), because the server cannot run a subgraph-definition UUID as a class;
//   - frontend-only nodes (Reroute, PrimitiveNode): a Reroute is followed through, a Primitive's value is already in
//     its targets' widgets, so neither enters the payload;
//   - boundary names are unique (the executor keys boundary inputs by name: two "image" inputs overwrote each other).

export const IO_IN = "-10";
export const IO_OUT = "-20";

export function allLinks(graph) {
  const L = graph?.links;
  if (!L) return [];
  if (typeof L.values === "function") return [...L.values()].filter(Boolean);
  if (Array.isArray(L)) return L.filter(Boolean);
  return Object.keys(L).map((k) => L[k]).filter(Boolean);
}

export function linkById(graph, id) {
  if (id == null) return null;
  const L = graph?.links;
  return (L instanceof Map ? L.get(id) : L?.[id]) ?? graph?._links?.get?.(id) ?? null;
}

function nodesOf(graph) {
  return (graph?._nodes || graph?.nodes || []).filter((n) => n && Number(n.id) >= 0);
}

export function isSubgraphNode(n) {
  try { if (typeof n?.isSubgraphNode === "function") return !!n.isSubgraphNode(); } catch (_) { /* ignore */ }
  return !!(n && n.subgraph && Array.isArray(n.subgraph._nodes || n.subgraph.nodes));
}

export function isVirtualNode(n) {
  return !!(n && !isSubgraphNode(n) && (n.isVirtualNode === true || n.constructor?.isVirtualNode === true));
}

function isPrimitive(n) {
  return isVirtualNode(n) && !(n.inputs || []).some((i) => i && !i.widget);
}

/** Widget values a node really sends: no display widgets ("$$..."), buttons or serialize:false widgets. */
export function widgetValues(n) {
  return Object.fromEntries((n.widgets || [])
    .filter((w) => w?.name && !String(w.name).startsWith("$$") && w.type !== "button"
      && w.serialize !== false && w.options?.serialize !== false)
    .map((w) => [w.name, w.value]));
}

/**
 * @param {object[]} sel   the selected LiteGraph nodes (IO nodes and ids < 0 are ignored)
 * @param {object} graph   the graph the selection lives in (the CURRENT canvas graph, root or subgraph)
 * @returns {{nodes, links, boundary_in, boundary_out, externalSources, externalTargets, inTypes, outTypes,
 *            problems: string[], flattenedSubgraphs: number}}
 */
export function flattenSelection(sel, graph) {
  const ctx = { nodes: [], byId: new Map(), problems: [], flattenedSubgraphs: 0 };
  const members = sel.filter((n) => n && Number(n.id) >= 0);
  const expanded = new Map();
  for (const n of members) expanded.set(String(n.id), expand(n, "", ctx));

  const links = [];
  const bIn = [];      // { name, to, to_slot, to_name, type, source }
  const bOut = [];     // { name, from, from_slot, type, targets: [] }
  resolveLinks(graph, members, expanded, ctx, {
    internal: (src, t) => links.push(payloadLink(src, t)),
    fromIo: (k, targets, type) => {
      for (const t of targets) bIn.push({ name: t.to_name, to: t.to, to_slot: t.to_slot, to_name: t.to_name, type,
        source: { kind: "io", slot: k } });
    },
    toIo: (k, src, type, name) => pushOut(bOut, src, type, name, { kind: "io", slot: k }),
    fromExternal: (lk, targets, type) => {
      for (const t of targets) bIn.push({ name: t.to_name, to: t.to, to_slot: t.to_slot, to_name: t.to_name, type,
        source: { kind: "node", id: lk.origin_id, slot: lk.origin_slot } });
    },
    toExternal: (lk, src, type, name) => pushOut(bOut, src, type, name,
      { kind: "node", id: lk.target_id, slot: lk.target_slot }),
  });

  // Nothing leaves the selection: offer every output nothing inside consumes (the old behaviour).
  if (!bOut.length) {
    const consumed = new Set(links.map((l) => `${l.from}:${l.from_slot}`));
    for (const n of members) {
      const e = expanded.get(String(n.id));
      (n.outputs || []).forEach((out, slot) => {
        const src = srcOf(e, slot);
        if (!src || consumed.has(`${src.from}:${src.from_slot}`)) return;
        pushOut(bOut, src, out.type || "*", out.name, null);
      });
    }
  }

  uniqueNames(bIn);
  uniqueNames(bOut);
  return {
    nodes: ctx.nodes,
    links,
    boundary_in: bIn.map(({ name, to, to_slot, to_name }) => ({ name, to, to_slot, to_name })),
    boundary_out: bOut.map(({ name, from, from_slot }) => ({ name, from, from_slot })),
    externalSources: bIn.map((b) => b.source),
    externalTargets: bOut.map((b) => b.targets),
    inTypes: bIn.map((b) => b.type || "*"),
    outTypes: bOut.map((b) => b.type || "*"),
    problems: ctx.problems,
    flattenedSubgraphs: ctx.flattenedSubgraphs,
  };
}

function payloadLink(src, t) {
  return { from: src.from, from_slot: src.from_slot, to: t.to, to_slot: t.to_slot, to_name: t.to_name };
}

function pushOut(bOut, src, type, name, target) {
  // one vault output per produced value, however many places outside consume it
  let o = bOut.find((b) => b.from === src.from && b.from_slot === src.from_slot);
  if (!o) {
    o = { name: name || `out_${bOut.length}`, from: src.from, from_slot: src.from_slot, type: type || "*", targets: [] };
    bOut.push(o);
  }
  if (target) o.targets.push(target);
}

function uniqueNames(list) {
  const seen = new Map();
  for (const b of list) {
    const base = String(b.name || "value");
    const k = (seen.get(base) || 0) + 1;
    seen.set(base, k);
    if (k > 1) b.name = `${base}_${k}`;
  }
}

/** One node's endpoints: ins.get(slot) = flat targets fed by that input; outs.get(slot) = flat producer. */
function expand(node, prefix, ctx) {
  // subgraph nodes report isVirtualNode too (they are expanded at queue time): test for them FIRST
  const sub = isSubgraphNode(node);
  if (!sub && isVirtualNode(node)) return { virtual: true, primitive: isPrimitive(node), node };
  if (!sub) {
    const id = prefix + String(node.id);
    const rec = { id, class_type: node.comfyClass || node.type, widgets: widgetValues(node) };
    ctx.nodes.push(rec);
    ctx.byId.set(id, rec);
    const ins = new Map();
    (node.inputs || []).forEach((inp, j) => ins.set(j, [{ to: id, to_slot: j, to_name: inp.name, type: inp.type }]));
    const outs = new Map();
    (node.outputs || []).forEach((_o, m) => outs.set(m, { from: id, from_slot: m }));
    return { id, ins, outs, node };
  }
  // a subgraph node: flatten its definition
  ctx.flattenedSubgraphs += 1;
  const sg = node.subgraph;
  const p2 = `${prefix}${node.id}:`;
  const inner = nodesOf(sg);
  const children = new Map();
  for (const n of inner) children.set(String(n.id), expand(n, p2, ctx));
  const ins = new Map();
  const outs = new Map();
  resolveLinks(sg, inner, children, ctx, {
    internal: null,                                    // inner=true collects these on ctx.pendingLinks
    fromIo: (k, targets) => ins.set(k, [...(ins.get(k) || []), ...targets]),
    toIo: (k, src) => outs.set(k, src),
    fromExternal: () => ctx.problems.push("a link inside a subgraph has an unknown source"),
    toExternal: () => ctx.problems.push("a link inside a subgraph has an unknown target"),
  }, true);
  // a promoted widget: the value typed on the subgraph node feeds the inner widget when nothing is wired to it
  (node.inputs || []).forEach((inp, k) => {
    if (!inp || inp.link != null) return;
    const w = (node.widgets || []).find((x) => x.name === inp.name || x.name === inp.widget?.name);
    if (!w || w.value === undefined) return;
    for (const t of ins.get(k) || []) {
      const rec = ctx.byId.get(t.to);
      if (rec) rec.widgets[t.to_name] = w.value;
    }
  });
  return { id: null, ins, outs, node, subgraph: true };
}

function srcOf(e, slot) {
  if (!e || e.virtual) return null;
  return e.outs.get(slot) || null;
}

function targetsOf(e, slot) {
  if (!e || e.virtual) return [];
  return e.ins.get(slot) || [];
}

/**
 * Walk every link of `graph` that touches `members` and report it to the handlers, with Reroutes followed and
 * Primitive links dropped. `inner` = the graph is a subgraph definition being flattened (IO ids are its own IO).
 */
function resolveLinks(graph, members, expanded, ctx, h, inner = false) {
  const memberIds = new Set(members.map((n) => String(n.id)));
  // inner links of a flattened subgraph are internal payload links: collect them on the context
  if (inner && !ctx.pendingLinks) ctx.pendingLinks = [];
  const internal = inner ? (src, t) => ctx.pendingLinks.push(payloadLink(src, t)) : h.internal;
  const sourceOf = (lk, depth = 0) => {
    // follow Reroutes upstream; returns {kind: "member"|"io"|"external"|"primitive", ...}
    const oid = String(lk.origin_id);
    if (oid === IO_IN) return { kind: "io", slot: lk.origin_slot };
    if (!memberIds.has(oid)) return { kind: "external", lk };
    const e = expanded.get(oid);
    if (e?.virtual) {
      if (e.primitive) return { kind: "primitive" };
      const up = linkById(graph, e.node.inputs?.[0]?.link);
      if (!up || depth > 64) return { kind: "none" };
      return sourceOf(up, depth + 1);
    }
    const src = srcOf(e, lk.origin_slot);
    return src ? { kind: "member", src, type: e.node.outputs?.[lk.origin_slot]?.type } : { kind: "none" };
  };
  for (const lk of allLinks(graph)) {
    const tid = String(lk.target_id);
    const tIsMember = memberIds.has(tid);
    const tIsIo = tid === IO_OUT;
    if (!tIsMember && !tIsIo && !memberIds.has(String(lk.origin_id))) continue;   // not ours
    const te = tIsMember ? expanded.get(tid) : null;
    if (te?.virtual) continue;                         // a Reroute's own input: handled from its consumers
    const s = sourceOf(lk);
    if (s.kind === "primitive" || s.kind === "none") continue;
    if (tIsMember) {
      const targets = targetsOf(te, lk.target_slot);
      if (!targets.length) continue;
      if (s.kind === "member") for (const t of targets) internal(s.src, t);
      else if (s.kind === "io") h.fromIo(s.slot, targets, ioType(graph, "in", s.slot) || lk.type);
      else h.fromExternal(s.lk, targets, outType(graph, s.lk) || lk.type);
    } else if (tIsIo) {
      if (s.kind === "member") h.toIo(lk.target_slot, s.src, s.type || lk.type, ioName(graph, "out", lk.target_slot));
      else ctx.problems.push(inner
        ? "a subgraph passes an input straight to an output; vault the nodes around it instead"
        : "the selection passes a subgraph input straight to its output");
    } else if (s.kind === "member") {
      const origin = expanded.get(String(lk.origin_id));
      const name = origin?.node?.outputs?.[lk.origin_slot]?.name;
      h.toExternal(lk, s.src, s.type || lk.type, name);
    } else {
      // only a Reroute in the selection gets here: it carries an outside value to an outside node
      ctx.problems.push("a Reroute in the selection passes a value straight through; leave it outside the selection");
    }
  }
  if (!inner && ctx.pendingLinks) {
    for (const l of ctx.pendingLinks) h.internal({ from: l.from, from_slot: l.from_slot }, l);   // already payload-shaped
    ctx.pendingLinks = null;
  }
}

function ioType(graph, side, slot) {
  const node = side === "in" ? graph?.inputNode : graph?.outputNode;
  return node?.slots?.[slot]?.type;
}

function ioName(graph, side, slot) {
  const node = side === "in" ? graph?.inputNode : graph?.outputNode;
  const s = node?.slots?.[slot];
  return s?.label || s?.name;
}

function outType(graph, lk) {
  const n = graph?.getNodeById?.(lk.origin_id);
  return n?.outputs?.[lk.origin_slot]?.type;
}
