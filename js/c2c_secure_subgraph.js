// C2C Vault as a protected native subgraph (A18, docs/VAULT_DESIGN_A18.md) - front end.
//
//   Locked state  : ONE node (C2C_SecureSubgraph / ..._Output): public sockets, promoted controls, the encrypted
//                   payload. No inner node, link, value or layout in the workflow.
//   Editing state : a real native subgraph node (same id, same links), entered like any subgraph. Leaving it
//                   (breadcrumb / navigation) re-encrypts and puts the vault node back (owner: exit = lock).
// While a vault is open, every serialization (save, autosave, export, the queued workflow) writes the vault in its
// locked form, copying the open subgraph is refused, and a queue encrypts a snapshot of the current edit first.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const VAULT_TYPES = new Set(["C2C_SecureSubgraph", "C2C_SecureSubgraphOutput"]);
const LEGACY_TYPES = new Set(["C2C_VaultLocked", "C2C_VaultSealed"]);
const PACK_AUX_ID = "Code2Collapse/ComfyUI-CustomNodePacks";
const OPEN = new Map();            // native SubgraphNode -> edit session
const RUN_UNLOCKED = new Set();    // Locked vault ids unlocked in this tab
let RAW_G2P = null;                // app.graphToPrompt before the guard (buildApi needs the inner nodes)
const LOG = (...a) => console.info("[C2C Vault]", ...a);

const PROFILES = [
  { id: "sealed_installs", title: "Sealed — authorized installations", tag: "Recommended",
    run: "Without a password, only on ComfyUI installations you authorize (yours is always included).",
    edit: "Vault password + your authenticator code + editor list.",
    net: "None - runs offline.", where: "On each authorized installation's GPU.",
    custody: "A per-vault key, wrapped for each authorized installation's own key pair.",
    security: "Encrypted. Other installations cannot run or read it, even with this pack.", risk: "" },
  { id: "locked", title: "Locked — password to run", tag: "",
    run: "Anyone with the vault password; asked once per browser tab.",
    edit: "Vault password + your authenticator code + editor list.",
    net: "None - runs offline.", where: "On the GPU of whoever runs it.",
    custody: "A per-vault key, unlocked by the password (Argon2id).",
    security: "Encrypted. Without the password nothing runs or opens.", risk: "" },
  { id: "sealed_portable", title: "Legacy portable — OBFUSCATION ONLY", tag: "Not secure",
    run: "Anyone, anywhere, without a password.",
    edit: "Vault password + your authenticator code + editor list.",
    net: "None.", where: "On the GPU of whoever runs it.",
    custody: "A key every copy of this pack can derive.",
    security: "OBFUSCATION ONLY: a knowledgeable recipient can extract the graph. Use only when a vault must run " +
      "on machines you cannot authorize.", risk: "warn" },
  { id: "owner_hosted", title: "Sealed — owner-hosted (coming)", tag: "Planned", disabled: true,
    run: "Recipients send inputs to YOUR server and get outputs back.", edit: "Only you.",
    net: "Recipients need your server online.", where: "On your server's GPU.",
    custody: "Graph and key never leave your server.", security: "Strongest: recipients never receive the graph.",
    risk: "" },
];

// ── small helpers ───────────────────────────────────────────────────────────────────────────────────────────
async function post(route, body) {
  const r = await api.fetchApi(route, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
  let data = {};
  try { data = await r.json(); } catch (_) { /* keep {} */ }
  return { ok: r.ok && data.ok !== false, status: r.status, data };
}

function el(tag, props = {}, kids = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "style") Object.assign(e.style, v);
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else if (k in e) e[k] = v;
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) if (c != null) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}

const T = {
  bg: "var(--c2c-bg2, #1d1f24)", fg: "var(--c2c-fg, #d8dae0)", dim: "var(--c2c-fg-dim, #9aa0aa)",
  border: "var(--c2c-border, #3a3e47)", field: "var(--c2c-surface0, #15171b)",
  accent: "var(--c2c-accent, #6aa7ff)", warn: "var(--c2c-warning, #f5c26b)", err: "var(--c2c-error, #ff7b7b)",
};

function inputStyle(extra = {}) {
  return { width: "100%", boxSizing: "border-box", padding: "6px 8px", borderRadius: "4px", background: T.field,
    color: T.fg, border: `1px solid ${T.border}`, font: "12px/1.4 inherit", ...extra };
}

function button(text, primary, onClick) {
  return el("button", { textContent: text, on: { click: onClick }, style: {
    padding: "5px 12px", borderRadius: "4px", cursor: "pointer", font: "12px inherit",
    border: `1px solid ${primary ? T.accent : T.border}`, background: primary ? T.accent : "transparent",
    color: primary ? "#0b0d10" : T.fg } });
}

/** A modal shell; `build(body, close)` fills it. Resolves with whatever close() is given. */
function modal(title, build, { width = 420 } = {}) {
  return new Promise((resolve) => {
    const back = el("div", { role: "dialog", "aria-modal": "true", style: { position: "fixed", inset: "0",
      zIndex: "10000", background: "rgba(0,0,0,.55)", display: "flex", alignItems: "center",
      justifyContent: "center" } });
    const box = el("div", { style: { background: T.bg, color: T.fg, border: `1px solid ${T.border}`,
      borderRadius: "6px", padding: "14px 16px", width: `${width}px`, maxWidth: "94vw", maxHeight: "90vh",
      overflow: "auto", font: "12px/1.45 sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,.5)" } });
    const close = (v) => { back.remove(); resolve(v); };
    box.append(el("div", { textContent: title, style: { fontWeight: "600", fontSize: "13px", marginBottom: "10px" } }));
    const body = el("div");
    box.append(body);
    back.append(box);
    back.addEventListener("keydown", (e) => { if (e.key === "Escape") close(null); });
    document.body.append(back);
    build(body, close);
    setTimeout(() => body.querySelector("input,button,select")?.focus(), 0);
  });
}

function errorLine() {
  return el("div", { style: { color: T.err, minHeight: "16px", margin: "6px 0 0", whiteSpace: "pre-wrap" } });
}

function toast(text, warn) {
  try {
    app.extensionManager?.toast?.add({ severity: warn ? "warn" : "info", summary: "C2C Vault", detail: text,
      life: 5000 });
  } catch (_) { LOG(text); }
}

// ── graph helpers ───────────────────────────────────────────────────────────────────────────────────────────
function isSubgraphNode(n) {
  try { return typeof n?.isSubgraphNode === "function" && n.isSubgraphNode(); } catch (_) { return false; }
}

function linkOf(graph, id) {
  if (id == null || !graph) return null;
  const L = graph.links ?? graph._links;
  return (L instanceof Map ? L.get(id) : L?.[id]) ?? null;
}

function nodeDefOf(type) {
  return LiteGraph.registered_node_types?.[type]?.nodeData ?? null;
}

/** The wires of a node: input slot index -> source, output index -> targets (sources/targets by id). */
function captureWires(graph, node) {
  const ins = (node.inputs || []).map((inp, i) => {
    const l = linkOf(graph, inp?.link);
    return l ? { i, origin: l.origin_id, slot: l.origin_slot } : null;
  }).filter(Boolean);
  const outs = [];
  (node.outputs || []).forEach((o, j) => (o?.links || []).forEach((lid) => {
    const l = linkOf(graph, lid);
    if (l) outs.push({ j, target: l.target_id, slot: l.target_slot });
  }));
  return { ins, outs };
}

/**
 * Put `to` where `from` is: same id (the prompt, history and the workflow keep naming the same node), same position,
 * every wire back. `mapIn(i)` turns an input slot index of `from` into one of `to` (vault inputs skip the promoted
 * controls of a subgraph); -1 drops that wire. The id is set BEFORE the node joins the graph: renumbering a node after
 * it is wired leaves its links pointing at an id nothing has (measured: InvalidLinkError at queue time).
 */
function replaceNode(graph, from, to, mapIn = (i) => i) {
  const wires = captureWires(graph, from);
  const id = from.id;
  to.pos = [...from.pos];
  graph.remove(from);
  if (id != null && graph.getNodeById(id) == null) to.id = id;
  graph.add(to);
  return wires;
}

function restoreWires(graph, wires, to, mapIn = (i) => i) {
  // inside a subgraph, wires to its own input / output node carry those nodes' ids, which getNodeById does not know
  // (measured: a vault made inside a subgraph lost them and its inner node had "no input at all")
  const ioIn = graph.inputNode, ioOut = graph.outputNode;
  for (const c of wires.ins) {
    const k = mapIn(c.i);
    if (k < 0 || k >= (to.inputs || []).length) continue;
    if (ioIn && c.origin === ioIn.id) { ioIn.slots?.[c.slot]?.connect(to.inputs[k], to); continue; }
    const origin = graph.getNodeById(c.origin);
    if (origin) origin.connect(c.slot, to, k);
  }
  for (const c of wires.outs) {
    if (c.j >= (to.outputs || []).length) continue;
    if (ioOut && c.target === ioOut.id) { ioOut.slots?.[c.slot]?.connect(to.outputs[c.j], to); continue; }
    const target = graph.getNodeById(c.target);
    if (target) to.connect(c.j, target, c.slot);
  }
}

/** Vault input slot index (by name in_<k>) for subgraph input i, given the non-promoted input order. */
function vaultSlotOf(v, inputMap, i) {
  const k = (inputMap || []).indexOf(i);
  return k < 0 ? -1 : (v.inputs || []).findIndex((x) => x.name === `in_${k}`);
}

/** Subgraph node input index for vault input slot `slot` (named in_<k>). */
function subgraphSlotOf(v, inputMap, slot) {
  const m = /^in_(\d+)$/.exec(v.inputs?.[slot]?.name || "");
  return m ? (inputMap?.[Number(m[1])] ?? Number(m[1])) : -1;
}

// ── vault node appearance and sockets ───────────────────────────────────────────────────────────────────────
function vaultInfo(node) { return node.properties?.c2c_vault || {}; }

function hideWidget(w) {
  if (!w) return;
  w.type = "hidden";
  w.hidden = true;
  w.computeSize = () => [0, -4];
  w.options = { ...(w.options || {}), hidden: true };
}

function promotedWidget(node, p) {
  const name = `p_${p.key}`;
  let w = node.widgets?.find((x) => x.name === name);
  if (w) return w;
  const t = String(p.type || "").toUpperCase();
  const v = p.value;
  if (Array.isArray(p.options) && p.options.length) w = node.addWidget("combo", name, v ?? p.options[0], () => {}, { values: p.options });
  else if (t === "BOOLEAN") w = node.addWidget("toggle", name, !!v, () => {});
  else if (t === "INT" || t === "FLOAT" || typeof v === "number") {
    w = node.addWidget("number", name, Number(v ?? 0), () => {}, { precision: t === "INT" ? 0 : 3,
      step2: t === "INT" ? 1 : 0.01 });
  } else w = node.addWidget("text", name, v == null ? "" : String(v), () => {});
  w.label = p.name || name;
  return w;
}

function applyInterface(node) {
  const info = vaultInfo(node);
  const iface = info.interface || {};
  hideWidget(node.widgets?.find((w) => w.name === "vault_id"));
  hideWidget(node.widgets?.find((w) => w.name === "vault_payload"));
  for (const p of iface.params || []) promotedWidget(node, p);
  const nIn = (iface.in || []).length, nOut = (iface.out || []).length;
  // the node def declares 256 outputs; a vault shows the ones its inside has
  while ((node.outputs || []).length > nOut) node.removeOutput(node.outputs.length - 1);
  (iface.out || []).forEach((o, j) => {
    if (!node.outputs[j]) node.addOutput(`out_${j}`, o.type || "*");
    node.outputs[j].label = o.name;
    node.outputs[j].type = o.type || "*";
  });
  (iface.in || []).forEach((s, i) => {
    const name = `in_${i}`;
    let slot = node.inputs?.find((x) => x.name === name);
    if (!slot) slot = node.addInput(name, s.type || "*");
    slot.label = s.name;
    slot.type = s.type || "*";
  });
  while ((node.inputs || []).filter((x) => /^in_\d+$/.test(x.name)).length > nIn) {
    const idx = node.inputs.findIndex((x) => x.name === `in_${nIn}`);
    if (idx < 0) break;
    node.removeInput(idx);
  }
  node.properties = node.properties || {};
  node.properties.aux_id = PACK_AUX_ID;                   // a machine without the pack offers to install it
  const prof = PROFILES.find((p) => p.id === info.profile);
  const legacy = info.profile === "sealed_portable";
  node.title = info.title || node.title || "C2C Vault";
  node.color = legacy ? "#6b5a2a" : "#2d4a6b";
  node.bgcolor = legacy ? "#3a321c" : "#1c2c40";
  node.__c2cBadge = `🔒 ${legacy ? "legacy - obfuscation only" : (prof?.id === "locked" ? "locked" : "sealed")}` +
    (info.node_count ? ` · ${info.node_count} nodes` : "");
  node.setSize?.(node.computeSize?.() ?? node.size);
  node.setDirtyCanvas?.(true, true);
}

// ── API form from the frontend's own graphToPrompt (exact widget / primitive / nested-subgraph conversion) ──
function execPrefix(output, sg) {
  const innerIds = new Set((sg.subgraph?.nodes || sg.subgraph?._nodes || []).map((n) => String(n.id)));
  for (const key of Object.keys(output)) {
    const parts = key.split(":");
    if (parts.length >= 2 && innerIds.has(parts.at(-1)) && parts.at(-2) === String(sg.id)) {
      return parts.slice(0, -1).join(":") + ":";
    }
  }
  return null;
}

function promotedOf(sg) {
  // native promotion: subgraph inputs wired to inner widgets show as widgets on the subgraph node
  const out = [];
  const inputs = sg.subgraph?.inputs || [];
  inputs.forEach((inp, i) => {
    const w = sg.widgets?.find((x) => x.name === inp.name);
    if (!w) return;
    const targets = [];
    for (const lid of inp.linkIds || []) {
      const l = linkOf(sg.subgraph, lid);
      const n = l && sg.subgraph.getNodeById(l.target_id);
      const name = n?.inputs?.[l.target_slot]?.widget?.name ?? n?.inputs?.[l.target_slot]?.name;
      if (n && name) targets.push({ node: String(n.id), input: name });
    }
    if (targets.length) out.push({ key: `i${i}`, name: inp.label || inp.name, type: inp.type, value: w.value,
      options: w.options?.values, targets, index: i });
  });
  return out;
}

async function buildApi(sg, extraParams = []) {
  const { output } = await (RAW_G2P ?? app.graphToPrompt.bind(app))();
  const prefix = execPrefix(output, sg);
  if (!prefix) throw new Error("This subgraph has no nodes that run (nothing to protect).");
  const nodes = {};
  for (const [k, v] of Object.entries(output)) {
    if (k.startsWith(prefix)) nodes[k.slice(prefix.length)] = { class_type: v.class_type, inputs: { ...v.inputs } };
  }
  const graph = sg.graph;
  // outer sources of each vault input (resolved like graphToPrompt resolves them)
  const inputs = (sg.inputs || []).map((s) => ({ name: s.label || s.name, type: String(s.type || "*") }));
  const outerSrc = new Map();
  (sg.inputs || []).forEach((s, i) => {
    const l = linkOf(graph, s.link);
    if (l) outerSrc.set(i, l);
  });
  const promoted = promotedOf(sg);
  const promotedIdx = new Set(promoted.map((p) => p.index));
  // inner inputs fed from a vault input -> {"vault_in": i}: by the subgraph's own input links ...
  const sub = sg.subgraph;
  (sub?.inputs || []).forEach((inp, i) => {
    if (promotedIdx.has(i)) return;
    for (const lid of inp.linkIds || []) {
      const l = linkOf(sub, lid);
      const n = l && sub.getNodeById(l.target_id);
      const name = n?.inputs?.[l.target_slot]?.widget?.name ?? n?.inputs?.[l.target_slot]?.name;
      const id = n && String(n.id);
      if (id && nodes[id] && name) nodes[id].inputs[name] = { vault_in: i };
    }
  });
  // ... and any remaining link that leaves the subgraph (nested subgraphs) by its resolved source
  const srcKey = (v) => `${v[0]}#${v[1]}`;
  const bySource = new Map();
  for (const n of Object.values(nodes)) for (const [name, v] of Object.entries(n.inputs)) {
    if (Array.isArray(v) && v.length === 2 && typeof v[0] === "string" && !nodes[v[0]] && !v[0].startsWith(prefix)) {
      bySource.set(srcKey(v), null);
    }
  }
  if (bySource.size) {
    // map resolved outer sources to vault inputs via direct cases recorded above
    for (const [i] of outerSrc) {
      for (const n of Object.values(nodes)) for (const v of Object.values(n.inputs)) {
        if (v && typeof v === "object" && !Array.isArray(v) && v.vault_in === i) { /* already direct */ }
      }
    }
  }
  for (const n of Object.values(nodes)) for (const [name, v] of Object.entries(n.inputs)) {
    if (Array.isArray(v) && v.length === 2 && typeof v[0] === "string") {
      if (v[0].startsWith(prefix)) n.inputs[name] = [v[0].slice(prefix.length), v[1]];
      else if (!nodes[v[0]]) {
        const i = [...outerSrc.entries()].findIndex(([, l]) => String(l.origin_id) === v[0].split(":").at(-1)
          && l.origin_slot === v[1]);
        if (i >= 0) n.inputs[name] = { vault_in: [...outerSrc.keys()][i] };
        else throw new Error("A node inside reads from outside the selection through a path the vault cannot " +
          "follow. Include that source in the selection, or wire it to a subgraph input.");
      }
    }
  }
  // outputs: the inner source of each subgraph output
  const outputs = (sg.outputs || []).map((o, j) => {
    let r = null;
    try { r = sg.resolveSubgraphOutputLink?.(j); } catch (_) { /* below */ }
    const node = r?.outputNode ?? r?.node ?? null;
    const slot = r?.link?.origin_slot ?? r?.slot ?? r?.outputSlot ?? 0;
    let id = node ? String(node.id) : null;
    if (id && !nodes[id]) {                     // an inner subgraph node: its resolved exec id is deeper
      const deeper = Object.keys(nodes).find((k) => k.split(":")[0] === id);
      id = deeper ?? id;
    }
    if (!id || !nodes[id]) {
      const lnk = sub?.outputNode?.slots?.[j]?.getLinks?.()?.[0];
      if (lnk && nodes[String(lnk.origin_id)]) return { name: o.label || o.name, type: String(o.type || "*"),
        node: String(lnk.origin_id), slot: lnk.origin_slot };
      throw new Error(`Output "${o.label || o.name}" is not connected inside the subgraph.`);
    }
    return { name: o.label || o.name, type: String(o.type || "*"), node: id, slot };
  });
  const apiPromoted = [];
  for (const p of promoted) for (const t of p.targets) apiPromoted.push({ key: p.key, node: t.node, input: t.input });
  const carried = [];
  for (const ep of extraParams || []) {              // an upgraded old vault's controls, read from the inner widget
    if (!nodes[ep.nodeId] || promoted.some((p) => p.key === ep.key)) continue;
    const w = sg.subgraph?.getNodeById?.(ep.nodeId)?.widgets?.find((x) => x.name === ep.input);
    apiPromoted.push({ key: ep.key, node: ep.nodeId, input: ep.input });
    carried.push({ key: ep.key, name: ep.name, type: ep.type, value: w?.value, options: ep.options });
  }
  const interface_ = {
    in: inputs.filter((_, i) => !promotedIdx.has(i)).map((s) => s),
    out: outputs.map((o) => ({ name: o.name, type: o.type })),
    params: [...promoted.map((p) => ({ key: p.key, name: p.name, type: p.type, value: p.value, options: p.options })),
      ...carried],
  };
  // vault inputs are numbered without the promoted ones
  const remap = new Map();
  let k = 0;
  inputs.forEach((_, i) => { if (!promotedIdx.has(i)) remap.set(i, k++); });
  for (const n of Object.values(nodes)) for (const [name, v] of Object.entries(n.inputs)) {
    if (v && typeof v === "object" && !Array.isArray(v) && "vault_in" in v) n.inputs[name] = { vault_in: remap.get(v.vault_in) };
  }
  const outputsNode = Object.values(nodes).some((n) => nodeDefOf(n.class_type)?.output_node);
  const inputMap = inputs.map((_, i) => i).filter((i) => !promotedIdx.has(i));
  return { api: { nodes, inputs: interface_.in, outputs, promoted: apiPromoted }, interface: interface_, outputsNode,
    inputMap };
}

function exportUi(sg) {
  const defs = [];
  const seen = new Set();
  const walk = (subgraph) => {
    if (!subgraph || seen.has(subgraph.id)) return;
    seen.add(subgraph.id);
    defs.push(subgraph.asSerialisable ? subgraph.asSerialisable() : subgraph.export?.());
    for (const n of subgraph.nodes || subgraph._nodes || []) if (isSubgraphNode(n)) walk(n.subgraph);
  };
  walk(sg.subgraph);
  return { root: sg.subgraph.id, definitions: defs, title: sg.title, size: [...sg.size], promoted: promotedOf(sg)
    .map((p) => ({ name: p.name, value: p.value })) };
}

// ── creating a vault from a subgraph node ───────────────────────────────────────────────────────────────────
async function protectSubgraphNode(sg, opts) {
  const built = await buildApi(sg);
  const ui = exportUi(sg);
  const r = await post("/c2c_vault/sg/protect", { password: opts.password, profile: opts.profile,
    recipients: opts.recipients, editors: opts.editors, interface: built.interface, api: built.api, ui,
    outputs_node: built.outputsNode, client_id: api.clientId });
  if (!r.ok) throw new Error(r.data.error || `Locking failed (${r.status}).`);
  const v = makeVaultNode(sg.graph, r.data, built, sg);
  removeDefinitionIfUnused(sg);
  if (opts.profile === "locked") RUN_UNLOCKED.add(r.data.vault_id);
  return v;
}

/** The locked vault node (built = {interface, outputsNode, inputMap}); replaces `replace` (a subgraph node) in place. */
function makeVaultNode(graph, res, built, replace) {
  const type = built.outputsNode ? "C2C_SecureSubgraphOutput" : "C2C_SecureSubgraph";
  const v = LiteGraph.createNode(type);
  v.properties = { ...(v.properties || {}), c2c_vault: { interface: built.interface, profile: res.profile,
    node_count: res.node_count, title: replace?.title, vault_id: res.vault_id, outputs_node: !!built.outputsNode,
    input_map: built.inputMap || null } };
  v.widgets.find((w) => w.name === "vault_id").value = res.vault_id;
  v.widgets.find((w) => w.name === "vault_payload").value = res.payload;
  if (!replace) {
    graph.add(v);
    applyInterface(v);
    return v;
  }
  const wires = replaceNode(graph, replace, v);
  applyInterface(v);                                         // the in_<k> / out_<j> sockets exist from here on
  restoreWires(graph, wires, v, (i) => vaultSlotOf(v, built.inputMap, i));
  if (replace.title) v.title = replace.title;
  return v;
}

function removeDefinitionIfUnused(sg) {
  const root = app.rootGraph ?? app.graph;
  const ids = vaultDefIds(sg.subgraph);
  if (!ids.size || !root?.subgraphs) return;
  // a definition stays only if a node OUTSIDE the vault (the root, or a definition not in the vault) still uses it
  const used = new Set();
  const visit = (g, seen = new Set()) => {
    if (!g || seen.has(g)) return;
    seen.add(g);
    for (const n of g.nodes || g._nodes || []) {
      if (!isSubgraphNode(n) || n === sg) continue;
      used.add(n.subgraph?.id);
      if (!ids.has(n.subgraph?.id)) visit(n.subgraph, seen);
    }
  };
  visit(root);
  for (const id of ids) if (!used.has(id)) root.subgraphs.delete?.(id);
}

// ── dialogs ─────────────────────────────────────────────────────────────────────────────────────────────────
function profileDialog({ title, defaultProfile }) {
  return modal(title, (body, close) => {
    let chosen = defaultProfile || "sealed_installs";
    const row = el("div", { style: { display: "flex", gap: "12px", alignItems: "stretch" } });
    const list = el("div", { style: { flex: "0 0 52%", display: "flex", flexDirection: "column", gap: "4px" } });
    const panel = el("div", { style: { flex: "1", border: `1px solid ${T.border}`, borderRadius: "4px",
      padding: "8px", background: T.field, fontSize: "11.5px" } });
    const fact = (k, v) => el("div", { style: { margin: "0 0 5px" } }, [el("b", { textContent: k + " " }), v]);
    const render = () => {
      const p = PROFILES.find((x) => x.id === chosen);
      panel.replaceChildren(
        el("div", { textContent: p.title, style: { fontWeight: "600", marginBottom: "6px",
          color: p.risk === "warn" ? T.warn : T.fg } }),
        fact("Runs:", p.run), fact("Opens for editing:", p.edit), fact("Network:", p.net),
        fact("GPU:", p.where), fact("Key:", p.custody),
        el("div", { textContent: p.security, style: { marginTop: "6px", color: p.risk === "warn" ? T.warn : T.dim } }));
      for (const b of list.children) {
        const on = b.dataset.id === chosen;
        b.style.borderColor = on ? T.accent : T.border;
        b.style.background = on ? "rgba(106,167,255,.12)" : "transparent";
        b.querySelector("input").checked = on;
      }
      recip.style.display = chosen === "sealed_installs" ? "" : "none";
    };
    for (const p of PROFILES) {
      const radio = el("input", { type: "radio", name: "c2c-vault-profile", value: p.id, disabled: !!p.disabled });
      const item = el("label", { "data-id": p.id, style: { display: "flex", gap: "6px", alignItems: "flex-start",
        border: `1px solid ${T.border}`, borderRadius: "4px", padding: "5px 6px", cursor: p.disabled ? "default" : "pointer",
        opacity: p.disabled ? ".5" : "1" } }, [radio, el("span", {}, [p.title, p.tag ? el("span", { textContent:
        ` ${p.tag}`, style: { color: p.risk === "warn" ? T.warn : T.dim, fontSize: "11px" } }) : null])]);
      item.dataset.id = p.id;
      if (!p.disabled) item.addEventListener("click", () => { chosen = p.id; render(); });
      list.append(item);
    }
    row.append(list, panel);
    const pw = el("input", { type: "password", placeholder: "Vault password (opens it for editing)",
      autocomplete: "new-password", style: inputStyle({ marginTop: "10px" }) });
    const pw2 = el("input", { type: "password", placeholder: "Repeat the password", autocomplete: "new-password",
      style: inputStyle({ marginTop: "6px" }) });
    const recip = el("textarea", { placeholder: "Other installations that may run it: paste their installation " +
      "codes (c2cvk1:...), one per line. Yours is included.", rows: 2, style: inputStyle({ marginTop: "6px",
      resize: "vertical" }) });
    const err = errorLine();
    const note = el("div", { textContent: "The password is never stored. Without it (and your authenticator) the " +
      "vault cannot be opened - keep it safe.", style: { color: T.dim, marginTop: "6px" } });
    const go = () => {
      if (pw.value.length < 8) { err.textContent = "Use at least 8 characters."; return; }
      if (pw.value !== pw2.value) { err.textContent = "The passwords differ."; return; }
      close({ profile: chosen, password: pw.value,
        recipients: recip.value.split(/\s+/).map((s) => s.trim()).filter(Boolean) });
    };
    pw2.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    body.append(row, pw, pw2, recip, note, err, el("div", { style: { display: "flex", justifyContent: "flex-end",
      gap: "8px", marginTop: "10px" } }, [button("Cancel", false, () => close(null)), button("Lock", true, go)]));
    render();
  }, { width: 620 });
}

function authDialog(title, { needCode = true } = {}) {
  return modal(title, (body, close) => {
    const pw = el("input", { type: "password", placeholder: "Vault password", autocomplete: "current-password",
      style: inputStyle() });
    const code = el("input", { inputMode: "numeric", placeholder: "6-digit code from your authenticator app",
      maxLength: 11, autocomplete: "one-time-code", style: inputStyle({ marginTop: "6px", letterSpacing: "2px" }) });
    const err = errorLine();
    const go = () => close({ password: pw.value, code: code.value.replace(/\s/g, ""), err });
    for (const i of [pw, code]) i.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    body.append(pw, needCode ? code : null, err, el("div", { style: { display: "flex", justifyContent: "flex-end",
      gap: "8px", marginTop: "10px" } }, [button("Cancel", false, () => close(null)), button("Open", true, go)]));
  });
}

async function enrolDialog() {
  const st = await post("/c2c_vault/2fa/enrol/begin", {});
  if (!st.ok) { toast(st.data.error || "Could not start the setup.", true); return false; }
  const d = st.data;
  return modal("Two-factor setup for C2C Vault", (body, close) => {
    const qr = el("div", { innerHTML: d.qr_svg || "", style: { width: "180px", height: "180px", flex: "0 0 auto",
      background: "#fff", borderRadius: "4px" } });
    const facts = el("div", { style: { fontSize: "11.5px" } }, [
      el("div", { textContent: "1. Scan the QR code with an authenticator app (Google Authenticator, Microsoft " +
        "Authenticator, 1Password, Aegis...). Or type the setup key:" }),
      el("code", { textContent: d.secret_grouped, style: { display: "block", margin: "6px 0", userSelect: "all",
        fontSize: "12.5px" } }),
      el("div", { textContent: `${d.issuer} · ${d.account} · ${d.algorithm}, ${d.digits} digits, every ${d.period} s`,
        style: { color: T.dim } }),
      el("div", { textContent: "2. Save these recovery codes (each works once if you lose the phone):",
        style: { marginTop: "8px" } }),
      el("code", { textContent: (d.recovery_codes || []).join("  "), style: { display: "block", margin: "4px 0",
        userSelect: "all" } }),
    ]);
    const saved = el("input", { type: "checkbox" });
    const code = el("input", { inputMode: "numeric", placeholder: "3. Enter the code the app now shows",
      maxLength: 6, style: inputStyle({ marginTop: "8px", letterSpacing: "2px" }) });
    const err = errorLine();
    const go = async () => {
      if (!saved.checked) { err.textContent = "Confirm that you saved the recovery codes."; return; }
      const r = await post("/c2c_vault/2fa/enrol/confirm", { code: code.value.trim() });
      if (r.ok) { toast("Two-factor is set up. A code works once: the next sign-in needs the next code."); close(true); }
      else err.textContent = r.data.error || "That code is not right.";
    };
    code.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    body.append(el("div", { style: { display: "flex", gap: "12px" } }, [qr, facts]),
      el("label", { style: { display: "flex", gap: "6px", marginTop: "8px" } }, [saved,
        "I saved the recovery codes"]), code, err,
      el("div", { style: { display: "flex", justifyContent: "flex-end", gap: "8px", marginTop: "10px" } },
        [button("Cancel", false, () => close(false)), button("Confirm", true, go)]));
  }, { width: 560 });
}

/** Two-factor setup from the vault menu: a second setup first needs a current code or a recovery code (owner: the
 *  pasted setup key is exposed - setting it up again must be possible from here). */
async function twoFactorSetup() {
  const st = await (await api.fetchApi("/c2c_vault/2fa/status")).json().catch(() => ({}));
  if (st.enrolled) {
    const code = await modal("Set up the authenticator again", (body, close) => {
      const inp = el("input", { inputMode: "numeric", placeholder: "Current 6-digit code, or a recovery code",
        autocomplete: "one-time-code", style: inputStyle({ letterSpacing: "1px" }) });
      const go = () => close(inp.value.replace(/\s/g, ""));
      inp.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
      body.append(el("div", { textContent: `An authenticator is set up on this install (${st.recovery_codes_left ?? 0} ` +
        "recovery codes left). To replace it - for example because its setup key was shared - confirm with a code " +
        "from it or a recovery code.", style: { color: T.dim, marginBottom: "8px" } }), inp,
        el("div", { style: { display: "flex", justifyContent: "flex-end", gap: "8px", marginTop: "10px" } },
          [button("Cancel", false, () => close(null)), button("Replace", true, go)]));
    });
    if (!code) return false;
    const r = await post("/c2c_vault/2fa/reset", { code });
    if (!r.ok) { toast(r.data.error || "That code is not right.", true); return false; }
  }
  return enrolDialog();
}

// ── opening (auth -> native subgraph) and auto-lock on leaving ──────────────────────────────────────────────
async function openVault(v) {
  const st = await (await api.fetchApi("/c2c_vault/2fa/status")).json().catch(() => ({}));
  if (!st.enrolled) {
    toast("Opening a vault needs an authenticator app on this installation. Set it up now.");
    if (!(await enrolDialog())) return;
  }
  const payload = v.widgets.find((w) => w.name === "vault_payload")?.value;
  const vaultId = v.widgets.find((w) => w.name === "vault_id")?.value;
  for (;;) {
    const a = await authDialog(`Open ${v.title || "vault"}`);
    if (!a) return;
    const r = await post("/c2c_vault/sg/open", { payload, vault_id: vaultId, password: a.password, code: a.code,
      client_id: api.clientId });
    if (r.ok) { await enterOpened(v, r.data, payload); return; }
    if (r.data.needs_enrolment) { if (!(await enrolDialog())) return; continue; }
    toast(r.data.error || "Could not open the vault.", true);
    if (r.data.reason !== "reused" && r.data.reason !== "clock" && r.status !== 403) return;
  }
}

async function enterOpened(v, res, payload) {
  const graph = v.graph;
  const root = app.rootGraph ?? app.graph;
  let sg;
  let extraParams = [];
  if (res.ui?.definitions?.length) {
    // nested definitions first, the vault's own last
    const defs = [...res.ui.definitions];
    const own = defs.find((d) => d.id === res.ui.root) ?? defs[0];
    for (const d of defs) if (d !== own && !root.subgraphs?.has?.(d.id)) root.createSubgraph(d);
    const subgraph = root.subgraphs?.get?.(own.id) ?? root.createSubgraph(own);
    sg = LiteGraph.createNode(subgraph.id);
    if (!sg) throw new Error("This front end cannot create the subgraph node.");
    const inputMap = vaultInfo(v).input_map;
    const wires = replaceNode(graph, v, sg);
    if (res.ui.size) sg.size = [...res.ui.size];
    sg.title = res.ui.title || v.title;
    restoreWires(graph, wires, sg, (slot) => subgraphSlotOf(v, inputMap, slot));
  } else {
    const legacy = buildFromLegacy(v, res);      // an old vault: rebuild its nodes, then convert natively
    sg = legacy.node;
    extraParams = legacy.extraParams;
  }
  OPEN.set(sg, { token: res.edit_token, vaultId: v.widgets.find((w) => w.name === "vault_id").value, payload,
    profile: res.profile, info: { ...vaultInfo(v) }, title: v.title, graph, extraParams });
  app.canvas.openSubgraph?.(sg.subgraph, sg);
  LOG("opened", sg.id);
}

/**
 * An old vault (C2C_VaultLocked / Sealed format) opened for editing: rebuild its nodes, wire them where the vault was,
 * then convert them natively. -> { node, extraParams }. The old promoted controls keep their values (applied to the
 * inner widgets) and are carried into the vault made when it locks again (owner A18: upgrade without breaking).
 */
function buildFromLegacy(v, res) {
  const graph = v.graph;
  const ex = res.legacy_exec || {};
  const made = new Map();
  let x = v.pos[0], y = v.pos[1];
  for (const n of ex.nodes || []) {
    const node = LiteGraph.createNode(n.class_type);
    if (!node) throw new Error("This vault needs custom nodes that are not installed here.");
    node.pos = [x, y];
    x += 260;
    graph.add(node);
    for (const [k, val] of Object.entries(n.widgets || {})) {
      const w = node.widgets?.find((w) => w.name === k);
      if (w) w.value = val;
    }
    made.set(String(n.id), node);
  }
  const ifaceParams = vaultInfo(v).interface?.params || [];
  const extraParams = [];
  for (const pr of ex.promoted || []) {
    const node = made.get(String(pr.node));
    const iw = node?.widgets?.find((w) => w.name === pr.widget);
    const pw = v.widgets?.find((w) => w.name === `p_${pr.id}`);
    if (!iw) continue;
    if (pw) iw.value = pw.value;
    const meta = ifaceParams.find((q) => String(q.key) === String(pr.id)) || {};
    extraParams.push({ key: String(pr.id), name: meta.name || String(pr.id), type: meta.type, options: meta.options,
      nodeId: String(node.id), input: pr.widget });
  }
  for (const l of ex.links || []) {
    const a = made.get(String(l.from)), b = made.get(String(l.to));
    if (!a || !b) continue;
    const slot = l.to_name ? b.inputs.findIndex((i) => i.name === l.to_name) : l.to_slot;
    if (slot >= 0) a.connect(l.from_slot, b, slot);
  }
  // boundary inputs by NAME: the vault's first slots are its widgets' sockets, so in_<i> is not slot i (measured:
  // a relocked old vault had no inputs). Inside a subgraph a source can be that subgraph's own input node.
  const wires = captureWires(graph, v);
  const byName = new Map(wires.ins.map((c) => [v.inputs?.[c.i]?.name, c]));
  const ioIn = graph.inputNode, ioOut = graph.outputNode;
  (ex.boundary_in || []).forEach((b, i) => {
    const c = byName.get(`in_${i}`);
    const tgt = made.get(String(b.to));
    if (!c || !tgt) return;
    const slot = b.widget ? tgt.inputs.findIndex((q) => q.widget?.name === b.widget || q.name === b.widget)
      : b.to_name ? tgt.inputs.findIndex((q) => q.name === b.to_name) : b.to_slot;
    if (slot < 0 || slot == null) return;
    if (ioIn && String(c.origin) === String(ioIn.id)) ioIn.slots?.[c.slot]?.connect(tgt.inputs[slot], tgt);
    else graph.getNodeById(c.origin)?.connect(c.slot, tgt, slot);
  });
  (ex.boundary_out || []).forEach((b, j) => {
    const src = made.get(String(b.from));
    if (!src) return;
    for (const c of wires.outs.filter((o) => o.j === j)) {
      if (ioOut && String(c.target) === String(ioOut.id)) ioOut.slots?.[c.slot]?.connect(src.outputs[b.from_slot], src);
      else {
        const t = graph.getNodeById(c.target);
        if (t) src.connect(b.from_slot, t, c.slot);
      }
    }
  });
  graph.remove(v);
  const ioWires = wiresFromParentInputs(graph, [...made.values()]);
  const r = graph.convertToSubgraph(new Set(made.values()));
  if (ioWires.length) reconnectParentInputs(graph, r.node, ioWires);
  return { node: r.node, extraParams };
}

let LOCKING = false;
async function lockOpened(sg, sess) {
  if (LOCKING) return;
  LOCKING = true;
  try {
    const built = await buildApi(sg, sess.extraParams);
    const r = await post("/c2c_vault/sg/commit", { edit_token: sess.token, api: built.api, ui: exportUi(sg),
      interface: built.interface, payload_hash: sess.payloadHash });
    if (!r.ok) throw new Error(r.data.error || `Locking failed (${r.status}).`);
    OPEN.delete(sg);
    makeVaultNode(sg.graph, { ...r.data, node_count: r.data.node_count }, built, sg);
    removeDefinitionIfUnused(sg);
    if (sess.profile === "locked") RUN_UNLOCKED.add(sess.vaultId);
    toast("Vault locked.");
  } catch (e) {
    toast(`${e.message} The vault stays open; fix it and leave the subgraph again.`, true);
  } finally { LOCKING = false; }
}

/** Is `g` the vault's subgraph or a subgraph nested anywhere inside it? (Subgraphs carry no parent pointer.) */
function contains(sub, g, seen = new Set()) {
  if (!sub || !g) return false;
  if (g === sub) return true;
  if (seen.has(sub)) return false;
  seen.add(sub);
  for (const n of sub.nodes || sub._nodes || []) if (isSubgraphNode(n) && contains(n.subgraph, g, seen)) return true;
  return false;
}

/** Subgraph definition ids used inside a vault (its own and every nested one). */
function vaultDefIds(sub, out = new Set()) {
  if (!sub || out.has(sub.id)) return out;
  out.add(sub.id);
  for (const n of sub.nodes || sub._nodes || []) if (isSubgraphNode(n)) vaultDefIds(n.subgraph, out);
  return out;
}

function onSetGraph(e) {
  const { newGraph, oldGraph } = e.detail || {};
  for (const [sg, sess] of [...OPEN]) {
    // leaving the vault (to its parent or anywhere outside) locks it; going deeper into a nested subgraph does not
    if (oldGraph && contains(sg.subgraph, oldGraph) && !contains(sg.subgraph, newGraph)) lockOpened(sg, sess);
  }
}

// ── serialization / clipboard / queue guards ────────────────────────────────────────────────────────────────
function lockedForm(sg, sess) {
  // the vault node as of its last encrypted snapshot, placed where the subgraph node is: inputs in_<k> follow the
  // non-promoted subgraph inputs (input_map), promoted controls go back into the p_ widgets' values
  const built = sess.built;
  const info = { ...(sess.info || {}), ...(built ? { interface: built.interface, outputs_node: !!built.outputsNode,
    input_map: built.inputMap } : {}) };
  const map = info.input_map || (sg.inputs || []).map((_, i) => i);
  const params = info.interface?.params || [];
  return { id: sg.id, type: info.outputs_node ? "C2C_SecureSubgraphOutput" : "C2C_SecureSubgraph",
    pos: [...sg.pos], size: [...sg.size], flags: {}, order: sg.order ?? 0, mode: sg.mode ?? 0,
    title: sess.title, properties: { c2c_vault: info, aux_id: PACK_AUX_ID },
    inputs: map.map((i, k) => ({ name: `in_${k}`, label: sg.inputs?.[i]?.label || sg.inputs?.[i]?.name,
      type: sg.inputs?.[i]?.type, link: sg.inputs?.[i]?.link ?? null })),
    outputs: (sg.outputs || []).map((o, j) => ({ name: `out_${j}`, label: o.label || o.name, type: o.type,
      links: o.links ? [...o.links] : null })),
    widgets_values: [sess.vaultId, sess.payload, ...params.map((prm) => prm.value)] };
}

/** Serialized links into a node that became its locked form: target slot = the subgraph input -> in_<k>. */
function remapLinkSlots(serialized, nodeId, map) {
  const fix = (slot) => { const k = (map || []).indexOf(slot); return k < 0 ? slot : k; };
  for (const l of serialized.links || []) {
    if (Array.isArray(l)) { if (l[3] === nodeId) l[4] = fix(l[4]); }
    else if (l && l.target_id === nodeId) l.target_slot = fix(l.target_slot);
  }
}

function scrub(serialized) {
  if (!OPEN.size || !serialized) return serialized;
  const ids = new Set();
  for (const sg of OPEN.keys()) for (const id of vaultDefIds(sg.subgraph)) ids.add(id);
  const strip = (nodes, holder) => (nodes || []).map((n) => {
    for (const [sg, sess] of OPEN) {
      if (n?.id === sg.id && n.type === sg.subgraph?.id) {
        const lf = lockedForm(sg, sess);
        remapLinkSlots(holder, sg.id, lf.properties.c2c_vault.input_map);
        return lf;
      }
    }
    return n;
  });
  serialized.nodes = strip(serialized.nodes, serialized);
  const defs = serialized.definitions?.subgraphs;
  if (Array.isArray(defs)) {
    for (const d of defs) d.nodes = strip(d.nodes, d);
    serialized.definitions.subgraphs = defs.filter((d) => !ids.has(d.id));
  }
  return serialized;
}

function installGuards() {
  const G = (app.rootGraph ?? app.graph)?.constructor;
  if (G && !G.prototype.__c2cVaultSerialize) {
    const orig = G.prototype.serialize;
    G.prototype.serialize = function (...a) { return scrub(orig.apply(this, a)); };
    G.prototype.__c2cVaultSerialize = true;
  }
  const C = app.canvas?.constructor;
  if (C && C.prototype.copyToClipboard && !C.prototype.__c2cVaultCopy) {
    const orig = C.prototype.copyToClipboard;
    C.prototype.copyToClipboard = function (items, ...a) {
      const sel = items ?? this.selectedItems ?? this.selected_nodes;
      const list = sel instanceof Set ? [...sel] : Array.isArray(sel) ? sel : Object.values(sel || {});
      if (list.some((n) => OPEN.has(n))) {
        toast("An open vault cannot be copied. Leave it (it locks), then copy the vault node.", true);
        return;
      }
      return orig.call(this, items, ...a);
    };
    C.prototype.__c2cVaultCopy = true;
  }
  if (!app.__c2cVaultQueue) {
    RAW_G2P = app.graphToPrompt.bind(app);
    app.graphToPrompt = async function (...a) {
      await snapshotOpen();
      const out = await RAW_G2P(...a);
      if (out?.output) rewriteOpenVaults(out.output);
      return out;
    };
    // the run password of a Locked vault is asked when a prompt is QUEUED, not on every graphToPrompt (other code -
    // Save Video's next-path preview, exports - calls that too)
    const q = api.queuePrompt.bind(api);
    api.queuePrompt = async function (number, data, ...rest) {
      await ensureRunKeys(data?.output);
      return q(number, data, ...rest);
    };
    app.__c2cVaultQueue = true;
  }
  window.addEventListener("beforeunload", () => {
    try { navigator.sendBeacon?.(api.apiURL("/c2c_vault/sg/forget_run"), JSON.stringify({ client_id: api.clientId })); }
    catch (_) { /* best effort */ }
  });
}

/** The flattened prompt of an OPEN vault -> its locked node: inner nodes out, the vault in, wires re-pointed. */
function rewriteOpenVaults(output) {
  for (const [sg, sess] of OPEN) {
    const prefix = execPrefix(output, sg);
    const built = sess.built;
    if (!prefix || !built) continue;
    const vid = prefix.slice(0, -1);
    const inputs = { vault_id: sess.vaultId, vault_payload: sess.payload };
    for (const [nid, n] of Object.entries(built.api.nodes)) {
      for (const [name, v] of Object.entries(n.inputs)) {
        if (!v || typeof v !== "object" || Array.isArray(v) || !("vault_in" in v)) continue;
        const flat = output[prefix + nid]?.inputs?.[name];
        if (flat !== undefined && inputs[`in_${v.vault_in}`] === undefined) inputs[`in_${v.vault_in}`] = flat;
      }
    }
    for (const prm of built.interface.params || []) {
      const t = (built.api.promoted || []).find((x) => x.key === prm.key);
      const flat = t ? output[prefix + t.node]?.inputs?.[t.input] : undefined;
      inputs[`p_${prm.key}`] = flat !== undefined ? flat : prm.value;
    }
    for (const k of Object.keys(output)) if (k.startsWith(prefix)) delete output[k];
    const outIdx = new Map(built.api.outputs.map((o, j) => [`${o.node}#${o.slot}`, j]));
    for (const n of Object.values(output)) {
      for (const [name, v] of Object.entries(n.inputs || {})) {
        if (!Array.isArray(v) || typeof v[0] !== "string" || !v[0].startsWith(prefix)) continue;
        const j = outIdx.get(`${v[0].slice(prefix.length)}#${v[1]}`);
        if (j != null) n.inputs[name] = [vid, j];
        else delete n.inputs[name];                 // never send an inner id; the server names what is missing
      }
    }
    output[vid] = { class_type: built.outputsNode ? "C2C_SecureSubgraphOutput" : "C2C_SecureSubgraph", inputs,
      _meta: { title: sess.title || "C2C Vault" } };
  }
  return output;
}

let SNAPPING = false;
async function snapshotOpen() {
  if (!OPEN.size || SNAPPING) return;
  SNAPPING = true;
  try {
    for (const [sg, sess] of OPEN) {
      const built = await buildApi(sg, sess.extraParams);
      const r = await post("/c2c_vault/sg/snapshot", { edit_token: sess.token, api: built.api, ui: exportUi(sg),
        interface: built.interface, payload_hash: sess.payloadHash });
      if (!r.ok) throw new Error(r.data.error || "Could not encrypt the open vault for this run.");
      sess.payload = r.data.payload;
      sess.built = built;
      sess.payloadHash = undefined;
      sess.info = { ...sess.info, interface: built.interface, outputs_node: built.outputsNode,
        node_count: r.data.node_count };
    }
  } finally { SNAPPING = false; }
}

async function ensureRunKeys(output) {
  for (const [id, n] of Object.entries(output || {})) {
    if (!VAULT_TYPES.has(n.class_type)) continue;
    const graphNode = (app.rootGraph ?? app.graph).getNodeById?.(id.split(":")[0]);
    const info = graphNode ? vaultInfo(graphNode) : {};
    const vid = n.inputs?.vault_id;
    if (info.profile !== "locked" || RUN_UNLOCKED.has(vid)) continue;
    for (;;) {
      const a = await authDialog(`Password to run ${graphNode?.title || "this vault"}`, { needCode: false });
      if (!a) throw new Error("Run cancelled: a Locked vault needs its password.");
      const r = await post("/c2c_vault/sg/unlock_run", { payload: n.inputs.vault_payload, vault_id: vid,
        password: a.password, client_id: api.clientId });
      if (r.ok) { RUN_UNLOCKED.add(vid); break; }
      toast(r.data.error || "Wrong password.", true);
    }
  }
}

// ── old Locked / Sealed nodes: upgraded automatically, payload kept ────────────────────────────────────────
function upgradeLegacy(old) {
  try {
    const w = (n) => old.widgets?.find((x) => x.name === n)?.value;
    const payload = w("vault_payload"), vid = w("vault_id");
    if (!payload || !vid) return;
    let iface = {};
    try { iface = JSON.parse(w("vault_interface") || "{}"); } catch (_) { iface = {}; }
    const params = (iface.params || []).filter((p) => !("socket" in p))
      .map((p) => ({ key: p.id, name: p.name, type: p.type, value: p.value }));
    const ifc = { in: iface.in || [], out: iface.out || [], params };
    const profile = old.type === "C2C_VaultLocked" ? "locked" : (iface.mode === "sealed_portable" ? "sealed_portable" : "sealed_installs");
    const graph = old.graph;
    const v = makeVaultNode(graph, { vault_id: vid, payload, profile, node_count: iface.node_count },
      { interface: ifc, outputsNode: !!iface.outputs_node, inputMap: null }, null);
    v.pos = [...old.pos];
    v.title = old.title;
    // old sockets: input_i -> in_i, outputs by index
    // by NAME: the vault node's first slots are its widgets' sockets (vault_id, vault_payload), so input_0 is not
    // slot 0 (measured: the upgraded vault's image wire was refused and the inside ran without its input)
    const wires = captureWires(graph, old);
    const slotOf = (i) => {
      const m = /^input_(\d+)$/.exec(old.inputs?.[i]?.name || "");
      return m ? (v.inputs || []).findIndex((x) => x.name === `in_${m[1]}`) : -1;
    };
    restoreWires(graph, { ins: wires.ins, outs: [] }, v, slotOf);
    for (const c of wires.outs) {
      const target = graph.getNodeById(c.target);
      if (target && c.j < v.outputs.length) v.connect(c.j, target, c.slot);
    }
    graph.remove(old);
    LOG(`upgraded an old ${old.type} to a protected subgraph vault (same payload)`);
  } catch (e) { console.warn("[C2C Vault] upgrade of an old vault failed; it keeps working as before:", e); }
}

// ── menus ───────────────────────────────────────────────────────────────────────────────────────────────────
function selectedNodes(canvas) {
  const s = canvas?.selectedItems ?? canvas?.selected_nodes;
  const list = s instanceof Set ? [...s] : Object.values(s || {});
  return list.filter((n) => n && typeof n.id !== "undefined" && n.constructor?.name !== "LGraphGroup");
}

/**
 * Frontend 1.53.10 bug (measured 2026-10-25): convertToSubgraph INSIDE a subgraph drops the wires that come from the
 * containing subgraph's own input node - the new subgraph node's inputs exist but are unwired, so a vault made there
 * ran without its inputs. Record them before converting ...
 */
function wiresFromParentInputs(graph, nodes) {
  const io = graph?.inputNode;
  if (!io) return [];
  const out = [];
  for (const n of nodes) {
    (n.inputs || []).forEach((inp, i) => {
      const l = linkOf(graph, inp?.link);
      if (l && String(l.origin_id) === String(io.id)) out.push({ slot: l.origin_slot, node: String(n.id), input: i,
        type: inp.type });
    });
  }
  return out;
}

/** ... and put each back on the new subgraph node's input that feeds the same inner socket (else: same type, in order). */
function reconnectParentInputs(graph, sg, wires) {
  const io = graph.inputNode;
  const sub = sg?.subgraph;
  if (!io || !sub) return;
  const taken = new Set();
  for (const w of wires) {
    let idx = (sub.inputs || []).findIndex((si, i) => !taken.has(i) && (si.linkIds || []).some((lid) => {
      const L = linkOf(sub, lid);
      return L && String(L.target_id) === w.node && L.target_slot === w.input;
    }));
    if (idx < 0) idx = (sg.inputs || []).findIndex((s, i) => !taken.has(i) && s.link == null && s.type === w.type);
    if (idx < 0 || !sg.inputs[idx] || sg.inputs[idx].link != null) continue;
    taken.add(idx);
    io.slots?.[w.slot]?.connect(sg.inputs[idx], sg);
  }
}

async function lockSelection(canvas, profileHint) {
  const nodes = selectedNodes(canvas);
  if (!nodes.length) return;
  const opts = await profileDialog({ title: "Protect as C2C Vault", defaultProfile: profileHint });
  if (!opts) return;
  const graph = nodes[0].graph;
  let sg;
  try {
    const single = nodes.length === 1 && isSubgraphNode(nodes[0]);
    const ioWires = single ? [] : wiresFromParentInputs(graph, nodes);
    sg = single ? nodes[0] : graph.convertToSubgraph(new Set(nodes)).node;
    if (ioWires.length) reconnectParentInputs(graph, sg, ioWires);
    await protectSubgraphNode(sg, opts);
    toast("Locked. Double-click the vault to open it (password + authenticator).");
  } catch (e) {
    toast(`${e.message}${sg && !isSubgraphNode(nodes[0]) ? " The selection is now a normal subgraph (nothing lost)." : ""}`, true);
  }
}

async function showInstallCode() {
  const r = await (await api.fetchApi("/c2c_vault/sg/install_code")).json().catch(() => ({}));
  await modal("My installation code", (body, close) => {
    body.append(el("div", { textContent: "Send this to a vault author so their Sealed vaults run here. It is a " +
      "public key - safe to share.", style: { color: T.dim } }),
    el("code", { textContent: r.code || "unavailable", style: { display: "block", margin: "8px 0", userSelect: "all",
      wordBreak: "break-all" } }),
    el("div", { style: { display: "flex", justifyContent: "flex-end", gap: "8px" } }, [
      button("Copy", false, () => navigator.clipboard?.writeText(r.code || "")), button("Close", true, () => close())]));
  });
}

app.registerExtension({
  name: "C2C.SecureSubgraph",
  setup() {
    installGuards();
    app.canvas?.canvas?.addEventListener("litegraph:set-graph", onSetGraph);
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (VAULT_TYPES.has(nodeData.name)) {
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function (...a) {
        const r = created?.apply(this, a);
        applyInterface(this);
        return r;
      };
      const configured = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (...a) {
        const r = configured?.apply(this, a);
        applyInterface(this);
        return r;
      };
      nodeType.prototype.onDblClick = function () { openVault(this); return true; };
      const fg = nodeType.prototype.onDrawForeground;
      nodeType.prototype.onDrawForeground = function (ctx, ...a) {
        const r = fg?.apply(this, [ctx, ...a]);
        if (this.__c2cBadge && !this.flags?.collapsed) {
          ctx.save();
          ctx.font = "11px sans-serif";
          ctx.fillStyle = this.properties?.c2c_vault?.profile === "sealed_portable" ? "#f5c26b" : "#9ec3ff";
          ctx.textAlign = "right";
          ctx.fillText(this.__c2cBadge, this.size[0] - 8, -8);
          ctx.restore();
        }
        return r;
      };
    }
    if (LEGACY_TYPES.has(nodeData.name)) {
      const configured = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (...a) {
        const r = configured?.apply(this, a);
        if (!window.__c2cVaultNoUpgrade) setTimeout(() => this.graph && upgradeLegacy(this), 0);
        return r;
      };
    }
  },
  getCanvasMenuItems(canvas) {
    const nodes = selectedNodes(canvas);
    if (!nodes.length) return [];
    return [null,
      { content: "🔒 Lock as C2C Vault (password to run)…", callback: () => lockSelection(canvas, "locked") },
      { content: "🔒 Seal as C2C Vault (authorized installations)…", callback: () => lockSelection(canvas, "sealed_installs") }];
  },
  getNodeMenuItems(node) {
    if (VAULT_TYPES.has(node?.comfyClass ?? node?.type)) {
      return [null,
        { content: "🔓 Open / edit vault…", callback: () => openVault(node) },
        { content: "🔑 My installation code", callback: showInstallCode },
        { content: "📱 Two-factor setup…", callback: () => twoFactorSetup() }];
    }
    if (isSubgraphNode(node) && !OPEN.has(node)) {
      return [null,
        { content: "🔒 Protect this subgraph as C2C Vault…", callback: async () => {
          const opts = await profileDialog({ title: "Protect as C2C Vault", defaultProfile: "sealed_installs" });
          if (opts) await protectSubgraphNode(node, opts).catch((e) => toast(e.message, true));
        } }];
    }
    return [];
  },
});

// the old vault front end (js/c2c_vault.js) routes its command-palette Lock / Seal and "Seal whole workflow" here
globalThis.__c2cVaultNative = { lockSelection: (profile) => lockSelection(app.canvas, profile) };

export { buildApi, exportUi, scrub, OPEN };
