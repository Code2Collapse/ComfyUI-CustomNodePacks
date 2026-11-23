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
import { BUSY_MESSAGE, ProtectError, absSource, captureBoundary, ensureReroutes, execIdOf, outerWires, relSource,
  rerouteChain, runTransaction, verifyConversion, verifyVaultWiring } from "./_vault_protect_tx.js";
// V1 (owner 2026-11-20): one-step "Lock selection as C2C Vault" + the Vault UI overhaul (docs/h2vault/V1_cnp_vault_ux_spec.md)
import { protectFailureText, resolveLockSelection, stageLabel, summarizeSelection } from "./_vault_selection.js";
import { block, capsule, choiceCards, codeField, copyRow, disclosure, el as vel, factList, noticeEl, passwordField,
  statusCard, stepSection, switchRow, textArea, textField, vaultDialog, vbutton } from "./_vault_ui.js";
import { tagDomWidget } from "./_vue_dom_rebind.js";

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
    custody: "Graph and key never leave your server.", security: "Strongest: recipients never receive the graph. " +
      "Available today for whole workflows: run the C2C App Gateway and use 📤 Publish as C2C App " +
      "(docs/APP_GATEWAY_DEPLOY.md).",
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

/** `warn`: true -> a warning; or a severity string ("success" | "info" | "warn" | "error"). */
function toast(text, warn, summary = "C2C Vault") {
  const severity = typeof warn === "string" ? warn : (warn ? "warn" : "info");
  try {
    app.extensionManager?.toast?.add({ severity, summary, detail: text, life: severity === "error" ? 8000 : 5000 });
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
    return l ? { i, origin: l.origin_id, slot: l.origin_slot, parent: l.parentId ?? null } : null;
  }).filter(Boolean);
  const outs = [];
  (node.outputs || []).forEach((o, j) => (o?.links || []).forEach((lid) => {
    const l = linkOf(graph, lid);
    if (l) outs.push({ j, target: l.target_id, slot: l.target_slot, parent: l.parentId ?? null });
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
  // L9.249 V3: a wire keeps its native link reroutes (its parent reroute id) when that reroute still exists
  const via = (c) => (c.parent != null && graph.reroutes?.get?.(c.parent) ? c.parent : undefined);
  for (const c of wires.ins) {
    const k = mapIn(c.i);
    if (k < 0 || k >= (to.inputs || []).length) continue;
    if (ioIn && c.origin === ioIn.id) { ioIn.slots?.[c.slot]?.connect(to.inputs[k], to, via(c)); continue; }
    const origin = graph.getNodeById(c.origin);
    if (origin) origin.connect(c.slot, to, k, via(c));
  }
  for (const c of wires.outs) {
    if (c.j >= (to.outputs || []).length) continue;
    if (ioOut && c.target === ioOut.id) { ioOut.slots?.[c.slot]?.connect(to.outputs[c.j], to, via(c)); continue; }
    const target = graph.getNodeById(c.target);
    if (target) to.connect(c.j, target, c.slot, via(c));
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
  node.color = legacy ? "#5a4a22" : "#2b2552";
  node.bgcolor = legacy ? "#2e2716" : "#191632";
  node.__c2cBadge = `🔒 ${legacy ? "legacy - obfuscation only" : (prof?.id === "locked" ? "locked" : "sealed")}` +
    (info.node_count ? ` · ${info.node_count} nodes` : "");
  ensureStatusCard(node);
  if (!node.__vaultMinWidth && typeof node.computeSize === "function") {
    // DOM widget computeSize gives a height only: the status card's minimum width comes from the node (memory:
    // frontend-node-width-traps)
    const cs = node.computeSize;
    node.computeSize = function (...a) { const r = cs.apply(this, a); if (Array.isArray(r)) r[0] = Math.max(r[0], 340); return r; };
    node.__vaultMinWidth = true;
  }
  node.setSize?.(node.computeSize?.() ?? node.size);
  node.setDirtyCanvas?.(true, true);
}

// ── the vault node's body: a status card (V1) ──────────────────────────────────────────────────────────────────
// A display-only DOM widget, in both renderers (the old badge was canvas text above the title that Nodes 2.0 never
// drew). It is never saved (widget.serialize = false: frontend 1.53.10 saves any widget whose own .serialize is not
// false, L9.237) and it is always the LAST widget, so the saved order [vault_id, vault_payload, p_* ...] never moves.
// It shows only public facts: profile, inner node COUNT, this tab's run state - never an inner name or type.
const STATUS_WIDGET = "vault_status";
const STATUS_H = 60;

function ensureStatusCard(node) {
  if (typeof document === "undefined" || typeof node?.addDOMWidget !== "function") return;
  let w = node.widgets?.find((x) => x.name === STATUS_WIDGET);
  if (!w) {
    const card = statusCard();
    w = node.addDOMWidget(STATUS_WIDGET, STATUS_WIDGET, card.el, { serialize: false, margin: 4,
      getMinHeight: () => STATUS_H, getHeight: () => STATUS_H });
    w.serialize = false;
    w.computeSize = (width) => [width, STATUS_H + 8];
    w.__vaultCard = card;
    try { tagDomWidget(w); } catch (_) { /* the Nodes 2.0 rebind tag is a nicety */ }
  }
  const ws = node.widgets || [];
  const at = ws.indexOf(w);
  if (at >= 0 && at !== ws.length - 1) { ws.splice(at, 1); ws.push(w); }
  updateStatusCard(node);
}

function updateStatusCard(node) {
  const card = node?.widgets?.find((x) => x.name === STATUS_WIDGET)?.__vaultCard;
  if (!card) return;
  const info = vaultInfo(node);
  const vid = node.widgets?.find((x) => x.name === "vault_id")?.value;
  const n = Number(info.node_count) || 0;
  const detail = n ? `${n} node${n === 1 ? "" : "s"} inside` : "Protected subgraph";
  if (!vid) {
    card.update({ tone: "warn", title: "Empty vault", detail: "Lock a selection to fill it",
      meta: "Right-click nodes › Lock as C2C Vault" });
    return;
  }
  if (info.profile === "sealed_portable") {
    card.update({ tone: "warn", title: "Portable · obfuscation only", detail, meta: "Runs anywhere without a password",
      action: { label: "Open…", ariaLabel: `Open ${node.title || "vault"}`, onClick: () => openVault(node) } });
    return;
  }
  const locked = info.profile === "locked";
  const unlocked = locked && RUN_UNLOCKED.has(vid);
  card.update({ tone: unlocked ? "ok" : "accent",
    title: locked ? "Locked · password to run" : "Sealed · authorized installations", detail,
    meta: locked ? (unlocked ? "Unlocked in this tab" : "Asks for the password at run time")
      : "No password on authorized installs", metaTone: unlocked ? "ok" : "",
    action: { label: "Open…", ariaLabel: `Open ${node.title || "vault"}`, onClick: () => openVault(node) } });
}

/** Refresh the cards of every vault with this id (after a run unlock). */
function refreshCards(vaultId) {
  const walk = (g, seen = new Set()) => {
    if (!g || seen.has(g)) return;
    seen.add(g);
    for (const n of g.nodes || g._nodes || []) {
      if (VAULT_TYPES.has(n.type) && (!vaultId || n.widgets?.find((w) => w.name === "vault_id")?.value === vaultId)) {
        updateStatusCard(n);
      }
      if (isSubgraphNode(n)) walk(n.subgraph, seen);
    }
  };
  try { walk(app.rootGraph ?? app.graph); } catch (_) { /* display only */ }
}

function markRunUnlocked(vaultId) {
  RUN_UNLOCKED.add(vaultId);
  refreshCards(vaultId);
}

// ── API form from the frontend's own graphToPrompt (exact widget / primitive / nested-subgraph conversion) ──
// A26 L9.203: the subgraph node's exact execution path (the frontend's own algorithm, see _vault_protect_tx.js) -
// never "a key whose last two segments look right", which matched the wrong instance of a shared definition.
function execPrefix(output, sg) {
  const id = execIdOf(sg, app.rootGraph ?? app.graph);
  if (!id) return null;
  const prefix = `${id}:`;
  return Object.keys(output).some((k) => k.startsWith(prefix)) ? prefix : null;
}

function promotedOf(sg) {
  // native promotion: subgraph inputs wired to inner widgets show as widgets on the subgraph node
  const out = [];
  const inputs = sg.subgraph?.inputs || [];
  inputs.forEach((inp, i) => {
    const w = sg.widgets?.find((x) => x.name === inp.name);
    if (!w) return;
    // A26 L9.203: a promoted widget input that an OUTER wire feeds is data, not a control - treating it as a control
    // left that inner input without a source ("vault input undefined"; the old flow sealed it unwired)
    if (sg.inputs?.[i]?.link != null) return;
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
  // ... and any remaining link that leaves the subgraph (nested subgraphs): matched by the EXACT resolved execution
  // source of each vault input's outer wire (A26 L9.203 - it used to compare only the last segment of the id)
  const root = app.rootGraph ?? app.graph;
  const sgPath = prefix.slice(0, -1);
  const graphPath = sgPath.split(":").slice(0, -1).join(":");
  const bySource = new Map();
  for (const [i, l] of outerSrc) {
    const src = absSource(root, graph, l, graphPath);
    if (src) bySource.set(`${src.id}#${src.slot}`, i);
  }
  for (const n of Object.values(nodes)) for (const [name, v] of Object.entries(n.inputs)) {
    if (Array.isArray(v) && v.length === 2 && typeof v[0] === "string") {
      if (v[0].startsWith(prefix)) n.inputs[name] = [v[0].slice(prefix.length), v[1]];
      else if (!nodes[v[0]]) {
        const i = bySource.get(`${v[0]}#${v[1]}`);
        if (i != null) n.inputs[name] = { vault_in: i };
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
    const innerSlot = r?.link?.origin_slot ?? r?.slot ?? r?.outputSlot ?? 0;
    // an inner subgraph node resolves to its exact inner source (A26: no "first key starting with that id")
    const rs = node ? relSource(node, innerSlot) : null;
    const id = rs ? rs.id : null;
    const slot = rs ? rs.slot : innerSlot;
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
const MAX_VAULT_OUTPUTS = 256;                     // nodes/secure_subgraph.py MAX_OUTPUTS

/** VAULT_PAYLOAD_VALIDATION (A26 L9.203): the extracted graph is complete and consistent before anything is sealed. */
function validateBuilt(built) {
  const fail = (m) => { throw new ProtectError("VAULT_PAYLOAD_VALIDATION", m); };
  const { nodes, inputs, outputs, promoted } = built.api;
  if (!Object.keys(nodes).length) fail("This subgraph has no nodes that run (nothing to protect).");
  if (outputs.length > MAX_VAULT_OUTPUTS) fail(`A vault can have at most ${MAX_VAULT_OUTPUTS} outputs (this has ${outputs.length}).`);
  const outCount = (id) => (nodeDefOf(nodes[id]?.class_type)?.output || []).length;
  for (const [id, n] of Object.entries(nodes)) {
    if (!nodeDefOf(n.class_type)) fail(`Node ${id} is a "${n.class_type}", which this ComfyUI does not have.`);
    for (const [name, v] of Object.entries(n.inputs || {})) {
      if (Array.isArray(v) && v.length === 2 && typeof v[0] === "string") {
        if (!nodes[v[0]]) fail(`Input "${name}" of node ${id} reads from node ${v[0]}, which is not inside.`);
        if (!(Number.isInteger(v[1]) && v[1] >= 0 && v[1] < outCount(v[0]))) fail(`Input "${name}" of node ${id} reads output ${v[1]} of node ${v[0]}, which does not exist.`);
      } else if (v && typeof v === "object" && "vault_in" in v) {
        if (!(Number.isInteger(v.vault_in) && v.vault_in >= 0 && v.vault_in < inputs.length)) fail(`Input "${name}" of node ${id} names vault input ${v.vault_in}, which does not exist.`);
      }
    }
  }
  outputs.forEach((o, j) => {
    if (!nodes[o.node] || !(o.slot >= 0 && o.slot < outCount(o.node))) fail(`Vault output ${j} ("${o.name}") does not point at an existing output inside.`);
  });
  for (const p of promoted || []) if (!nodes[p.node]) fail(`The control "${p.key}" targets node ${p.node}, which is not inside.`);
}

/**
 * "Protect selection" / "Protect this subgraph" as ONE transaction (A26 L9.203): stages per _vault_protect_tx.js
 * STAGES; any failure restores the exact original graph (never a leftover ordinary subgraph) and names the stage.
 */
async function protectTransaction(nodes, opts) {
  const graph = nodes[0].graph;
  const single = nodes.length === 1 && isSubgraphNode(nodes[0]);
  let sg = single ? nodes[0] : null, manifest = null, built = null, res = null, vault = null, before = null, wires = null;
  let chains = [];
  try {
    await runTransaction(app, [
      ["SELECTION_VALIDATION", () => {
        if (nodes.some((n) => n.graph !== graph)) throw new ProtectError("SELECTION_VALIDATION", "Select nodes from one graph.");
        if (nodes.some((n) => VAULT_TYPES.has(n.type) || LEGACY_TYPES.has(n.type))) {
          throw new ProtectError("SELECTION_VALIDATION", "The selection contains a vault; protect the nodes around it separately.");
        }
      }],
      ["BOUNDARY_CAPTURE", () => { manifest = single ? null : captureBoundary(graph, nodes); }],
      ["NATIVE_SUBGRAPH_CONVERSION", () => {
        if (!single) sg = graph.convertToSubgraph(new Set(nodes))?.node ?? null;
        if (!sg) throw new ProtectError("NATIVE_SUBGRAPH_CONVERSION", "The selection could not be turned into a subgraph.");
      }],
      ["SUBGRAPH_INTERFACE_VALIDATION", () => {
        if (manifest) verifyConversion(graph, sg, manifest);
        if ((sg.outputs || []).length > MAX_VAULT_OUTPUTS) {
          throw new ProtectError("SUBGRAPH_INTERFACE_VALIDATION", `A vault can have at most ${MAX_VAULT_OUTPUTS} outputs.`);
        }
      }],
      ["INNER_GRAPH_EXTRACTION", async () => { built = await buildApi(sg); }],
      ["VAULT_PAYLOAD_VALIDATION", () => validateBuilt(built)],
      ["ENCRYPTION_AND_SEALING", async () => {
        const r = await post("/c2c_vault/sg/protect", { password: opts.password, profile: opts.profile,
          recipients: opts.recipients, editors: opts.editors, interface: built.interface, api: built.api, ui: exportUi(sg),
          outputs_node: built.outputsNode, client_id: api.clientId });
        if (!r.ok) throw new ProtectError("ENCRYPTION_AND_SEALING", r.data?.error || `Locking failed (${r.status}).`);
        res = r.data;
      }],
      ["VAULT_NODE_INSERTION", () => {
        before = outerWires(graph, sg);
        const type = built.outputsNode ? "C2C_SecureSubgraphOutput" : "C2C_SecureSubgraph";
        vault = LiteGraph.createNode(type);
        if (!vault) throw new ProtectError("VAULT_NODE_INSERTION", `${type} is not registered (is the pack loaded?).`);
        vault.properties = { ...(vault.properties || {}), c2c_vault: { interface: built.interface, profile: res.profile,
          node_count: res.node_count, title: sg.title, vault_id: res.vault_id, outputs_node: !!built.outputsNode,
          input_map: built.inputMap || null } };
        vault.widgets.find((w) => w.name === "vault_id").value = res.vault_id;
        vault.widgets.find((w) => w.name === "vault_payload").value = res.payload;
        // a FRESH id: Nodes 2.0 keys a node's view by id and kept showing the subgraph's view ("New Subgraph",
        // "Enter subgraph") on the vault; nothing outside refers to this subgraph node's id (A26 L9.203)
        wires = captureWires(graph, sg);
        // removing the subgraph node can delete reroutes only its wires used: keep their chains to re-make them
        chains = [...wires.ins, ...wires.outs].filter((c) => c.parent != null).map((c) => rerouteChain(graph, c.parent))
          .filter(Boolean);
        vault.pos = [...sg.pos];
        graph.remove(sg);
        graph.add(vault);
        applyInterface(vault);
        // the frontend names every converted subgraph "New Subgraph": a vault made from one is called "C2C Vault"
        const named = typeof opts.title === "string" ? opts.title.trim().slice(0, 80) : "";
        const t = named || (sg.title && sg.title !== "New Subgraph" ? sg.title : "C2C Vault");
        vault.title = t;
        vault.properties.c2c_vault.title = t;
      }],
      ["EXTERNAL_WIRE_RESTORATION", () => {
        for (const ch of chains) {
          if (!ensureReroutes(graph, ch)) {
            throw new ProtectError("EXTERNAL_WIRE_RESTORATION", "A wire's reroute dots could not be re-made on the vault.");
          }
        }
        restoreWires(graph, wires, vault, (i) => vaultSlotOf(vault, built.inputMap, i));
      }],
      ["FINAL_GRAPH_VALIDATION", () => {
        const kept = new Set(built.inputMap || []);
        const promotedIdx = new Set((sg.inputs || []).map((_, i) => i).filter((i) => !kept.has(i)));
        verifyVaultWiring(graph, vault, before, (i) => vaultSlotOf(vault, built.inputMap, i), promotedIdx);
        removeDefinitionIfUnused(sg);
      }],
    ], { onStage: opts.onStage });
  } catch (e) {
    // L9.249 V6: sealing already stored this tab's run key for the new vault (Locked profile); the vault no longer
    // exists in the graph, so drop that server session too. Best effort: the graph is already restored.
    if (res?.vault_id) await dropProtectSession(res.vault_id);
    throw e;
  }
  if (opts.profile === "locked") markRunUnlocked(res.vault_id);
  updateStatusCard(vault);
  graph.setDirtyCanvas?.(true, true);
  return vault;
}

/** V6: ask the server to forget the run session a failed protect left behind (POST /c2c_vault/sg/protect/abort). */
async function dropProtectSession(vaultId) {
  try {
    const r = await post("/c2c_vault/sg/protect/abort", { vault_id: vaultId, client_id: api.clientId });
    LOG("dropped the run session of the vault a failed protect made", vaultId, r.data?.dropped);
  } catch (err) { console.warn("[C2C Vault] could not drop the failed protect's run session:", err); }
}

async function protectSubgraphNode(sg, opts) {
  const built = await buildApi(sg);
  const ui = exportUi(sg);
  const r = await post("/c2c_vault/sg/protect", { password: opts.password, profile: opts.profile,
    recipients: opts.recipients, editors: opts.editors, interface: built.interface, api: built.api, ui,
    outputs_node: built.outputsNode, client_id: api.clientId });
  if (!r.ok) throw new Error(r.data.error || `Locking failed (${r.status}).`);
  const v = makeVaultNode(sg.graph, r.data, built, sg);
  removeDefinitionIfUnused(sg);
  if (opts.profile === "locked") markRunUnlocked(r.data.vault_id);
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

// ── dialogs (V1: js/_vault_ui.js design system; docs/h2vault/V1_cnp_vault_ux_spec.md) ─────────────────────────
const SHORT = {
  sealed_installs: { title: "Sealed", tag: "Recommended", tone: "ok" },
  locked: { title: "Locked", tag: "Password to run", tone: "" },
  sealed_portable: { title: "Portable", tag: "Obfuscation only", tone: "warn" },
};
const PROTECT_FOOT = "The password is never stored. Keep it safe.";

function profileFacts(p) {
  return factList([["Opens for editing", p.edit], ["Network", p.net], ["Runs on", p.where],
    ["Key", p.custody], ["Protection", p.security]]);
}

const splitCodes = (text) => String(text || "").split(/\s+/).map((s) => s.trim()).filter(Boolean);

/**
 * THE protect dialog (spec 4.1): one dialog for "lock a selection" and "protect this subgraph". `run(opts)` performs
 * the lock (protectTransaction) while the dialog shows its stages; a failure is shown inside the dialog, which stays
 * open. Resolves with the vault node, or null when cancelled.
 */
function protectDialog({ count, kind, summary, title, defaultProfile, run }) {
  const many = count === 1 ? "" : "s";
  const d = vaultDialog({
    title: kind === "subgraph" ? `Lock "${title || "subgraph"}" as a vault` : `Lock ${count} node${many} as a vault`,
    subtitle: kind === "subgraph" ? "The subgraph becomes one node whose inside is encrypted."
      : `${count === 1 ? "It becomes" : "They become"} one node whose inside is encrypted. Everything outside stays as it is.`,
    width: 860 });
  const left = vel("div", { class: "c2c-vault-col" });
  const right = vel("div", { class: "c2c-vault-col" });
  d.body.append(vel("div", { class: "c2c-vault-cols" }, [left, right]));
  left.append(capsule(summary));
  if (summary.hasSubgraph && kind !== "subgraph") {
    left.append(noticeEl("info", "The selection contains a subgraph. If ComfyUI loses one of its wires while " +
      "grouping, nothing is changed and you are told - then unpack it (right-click › Unpack Subgraph) and lock again."));
  }
  const name = textField({ label: "Vault name", value: title || "C2C Vault", maxLength: 80,
    hint: "Shown on the node. Anyone who receives the workflow can read the name." });
  const facts = vel("div");
  const choice = choiceCards({ label: "Who can run it", value: defaultProfile || "sealed_installs",
    options: PROFILES.map((p) => ({ id: p.id, title: SHORT[p.id]?.title ?? p.title, tag: SHORT[p.id]?.tag,
      tone: SHORT[p.id]?.tone, summary: p.run, disabled: p.disabled })),
    onChange: () => render() });
  const hosted = PROFILES.find((p) => p.id === "owner_hosted");
  const pw2 = passwordField({ label: "Repeat the password" });
  const pw = passwordField({ label: "Vault password", strength: true, generate: true,
    hint: "Opens the vault for editing, together with your authenticator app.", onGenerate: (v) => pw2.setValue(v) });
  const recip = textArea({ label: "Installation codes, one per line", placeholder: "c2cvk1:...", rows: 3,
    hint: "Each person finds theirs under right-click on a vault › My installation code. Yours is always included." });
  const others = disclosure("Other installations that may run it (optional)");
  others.body.append(recip.el, vel("div", { class: "c2c-vault-row" }, [vbutton("Import key file…", { small: true,
    onClick: async () => {
      const k = await importKeyFile();
      if (k?.error) { d.notice("danger", k.error); return; }
      if (k && !recip.input.value.includes(k.code)) recip.input.value = `${recip.input.value.trim()}\n${k.code}`.trim();
    } })]));
  function render() {
    const p = PROFILES.find((x) => x.id === choice.value());
    facts.replaceChildren(profileFacts(p));
    if (p.risk === "warn") facts.append(vel("div", { style: { marginTop: "8px" } }, [noticeEl("warn", p.security)]));
    others.el.style.display = p.id === "sealed_installs" ? "" : "none";
  }
  left.append(name.el, pw.el, pw2.el, others.el);
  right.append(choice.el, facts);
  if (hosted) {
    right.append(vel("p", { class: "c2c-vault-dim", text: "Owner-hosted vaults, where recipients send inputs to your " +
      "server and never receive the graph, are coming. For whole workflows, use the C2C App Gateway." }));
  }
  render();
  d.status(PROTECT_FOOT);
  d.addAction("Cancel", { onClick: () => d.close(null) });
  const go = async () => {
    if (d.busy) return;
    d.notice(null);
    if (pw.value().length < 8) { d.notice("danger", "Use at least 8 characters for the vault password."); pw.input.focus(); return; }
    if (pw.value() !== pw2.value()) { d.notice("danger", "The two passwords differ."); pw2.input.focus(); return; }
    d.setBusy(true, "Locking…");
    d.status(`${stageLabel("SELECTION_VALIDATION")}…`);
    try {
      const vault = await run({ profile: choice.value(), password: pw.value(), recipients: splitCodes(recip.input.value),
        title: name.input.value, onStage: (s) => d.status(s === "COMMIT" ? "Locked." : `${stageLabel(s)}…`) });
      d.close(vault);
    } catch (e) {
      LOG("protect failed", e?.stage, e?.message);
      d.setBusy(false);
      d.status(PROTECT_FOOT);
      d.notice(e?.busy ? "warn" : "danger", protectFailureText(e));
    }
  };
  d.addAction(kind === "subgraph" ? "Lock subgraph" : `Lock ${count} node${many}`, { primary: true, onClick: go });
  d.onSubmit(go);
  return d.result;
}

/**
 * Password (+ authenticator code) dialog that STAYS OPEN until `submit` succeeds (spec 4.2 / 4.3 / 4.5).
 * submit({password, code}) -> {ok: true, ...} | {enrol: true} | {error: string}. Resolves with that result or null.
 */
function credentialDialog({ title, subtitle, needCode = true, primary = "Open vault", submit }) {
  const d = vaultDialog({ title, subtitle, width: 440 });
  const pw = passwordField({ label: "Vault password", autocomplete: "current-password" });
  const code = needCode ? codeField({ allowRecovery: true, hint: "Or one of your recovery codes. A code works once: " +
    "if you just used one, wait for the next." }) : null;
  if (code) d.onCleanup(code.destroy);
  d.body.append(pw.el);
  if (code) d.body.append(code.el);
  d.addAction("Cancel", { onClick: () => d.close(null) });
  const go = async () => {
    if (d.busy) return;
    d.notice(null);
    if (!pw.value()) { d.notice("danger", "Type the vault password."); pw.input.focus(); return; }
    if (code && !code.value()) { d.notice("danger", "Type the code your authenticator app shows now."); code.input.focus(); return; }
    d.setBusy(true, "Checking…");
    let out;
    try { out = await submit({ password: pw.value(), code: code ? code.value() : "" }); }
    catch (e) { out = { error: String(e?.message || e) }; }
    if (out?.ok || out?.enrol) { d.close(out); return; }
    d.setBusy(false);
    d.notice("danger", out?.error || "That did not work. Check the password and try again.");
    if (code) { code.clear(); code.input.focus(); } else pw.input.select();
  };
  d.addAction(primary, { primary: true, onClick: go });
  d.onSubmit(go);
  return d.result;
}

/** A server answer for a failed open / access check, in words (reused code, clock drift, wrong password ...). */
function authError(r, fallback) {
  const d = r?.data || {};
  if (d.reason === "reused") return `That code was already used. Wait for the next one${d.next_in ? ` (${d.next_in} s)` : ""}, then type it.`;
  if (d.reason === "clock") return d.error || "The code is for another time: this computer's or the phone's clock is off. Set both to automatic time.";
  if (r?.status === 429) return d.error || "Too many attempts. Wait a few minutes, then try again.";
  return d.error || fallback;
}

async function enrolDialog() {
  const st = await post("/c2c_vault/2fa/enrol/begin", {});
  if (!st.ok) { toast(st.data.error || "Could not start the setup.", "error"); return false; }
  const data = st.data;
  const d = vaultDialog({ title: "Set up two-factor for C2C Vault",
    subtitle: "Opening a vault on this installation needs a code from an authenticator app, as well as the password.",
    width: 640 });
  const qr = vel("div", { class: "c2c-vault-qr" }, [vel("img", { alt: "QR code for your authenticator app",
    src: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(data.qr_svg || "")}` })]);
  const scan = stepSection(1, "Scan the code with your authenticator app",
    vel("div", { class: "c2c-vault-row", style: { alignItems: "flex-start", gap: "16px", flexWrap: "wrap" } }, [qr,
      vel("div", { style: { flex: "1", minWidth: "220px", display: "flex", flexDirection: "column", gap: "8px" } }, [
        vel("p", { class: "c2c-vault-dim", text: "Google Authenticator, Microsoft Authenticator, 1Password, Aegis and " +
          "others work. No camera? Type the setup key instead:" }),
        copyRow({ label: "setup key", value: data.secret_grouped }),
        vel("p", { class: "c2c-vault-dim", text: `${data.issuer} · ${data.account} · ${data.algorithm}, ${data.digits} ` +
          `digits, every ${data.period} s` })])]));
  const codes = data.recovery_codes || [];
  const saved = vel("input", { type: "checkbox" });
  const download = () => {
    const url = URL.createObjectURL(new Blob([`C2C Vault recovery codes (each works once)\n\n${codes.join("\n")}\n`],
      { type: "text/plain" }));
    const a = vel("a", { href: url, download: "c2c-vault-recovery-codes.txt" });
    document.body.append(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
  };
  const copyAll = vbutton("Copy all", { small: true, onClick: async () => {
    try { await navigator.clipboard.writeText(codes.join("\n")); copyAll.querySelector(".c2c-vault-btn__label").textContent = "Copied"; }
    catch (_) { /* the grid is selectable */ }
  } });
  const keep = stepSection(2, "Save your recovery codes",
    vel("p", { class: "c2c-vault-dim", text: "Each works once instead of a code if you lose the phone. They are not shown again." }),
    vel("div", { class: "c2c-vault-codes" }, codes.map((c) => vel("span", { text: c }))),
    vel("div", { class: "c2c-vault-row", style: { justifyContent: "space-between", flexWrap: "wrap" } }, [
      vel("label", { class: "c2c-vault-check" }, [saved, vel("span", { text: "I saved the recovery codes" })]),
      vel("div", { class: "c2c-vault-row" }, [copyAll, vbutton("Download .txt", { small: true, onClick: download })])]));
  const code = codeField({ label: "Code from the app" });
  d.onCleanup(code.destroy);
  const confirm = stepSection(3, "Type the code your app shows now", code.el);
  d.body.append(scan, keep, confirm);
  d.addAction("Cancel", { onClick: () => d.close(false) });
  const go = async () => {
    if (d.busy) return;
    d.notice(null);
    if (!saved.checked) { d.notice("danger", "Confirm that you saved the recovery codes first."); saved.focus(); return; }
    if (!/^\d{6}$/.test(code.value())) { d.notice("danger", "Type the 6-digit code your app shows now."); code.input.focus(); return; }
    d.setBusy(true, "Confirming…");
    const r = await post("/c2c_vault/2fa/enrol/confirm", { code: code.value() });
    if (r.ok) {
      d.close(true);
      toast("Two-factor is set up. A code works once: the next sign-in needs the next code.", "success", "Two-factor on");
      return;
    }
    d.setBusy(false);
    d.notice("danger", authError(r, "That code is not right. Check the phone's time and type the current code."));
    code.clear();
    code.input.focus();
  };
  d.addAction("Confirm", { primary: true, onClick: go });
  d.onSubmit(go);
  // the dialog takes focus on the first text field, which is the code at the bottom: start at the top instead
  setTimeout(() => d.panel.focus(), 0);
  return d.result;
}

/** Two-factor setup from the vault menu: a second setup first needs a current code or a recovery code (owner: the
 *  pasted setup key is exposed - setting it up again must be possible from here). */
async function twoFactorSetup() {
  const st = await (await api.fetchApi("/c2c_vault/2fa/status")).json().catch(() => ({}));
  if (st.enrolled) {
    const d = vaultDialog({ title: "Set up the authenticator again",
      subtitle: "Use this when the setup key was shared or the phone changed. The old key stops working.", width: 460 });
    d.body.append(noticeEl("info", `An authenticator is set up on this installation (${st.recovery_codes_left ?? 0} ` +
      "recovery codes left). Confirm with a code from it, or with a recovery code."));
    const code = codeField({ label: "Current code or a recovery code", allowRecovery: true });
    d.onCleanup(code.destroy);
    d.body.append(code.el);
    d.addAction("Cancel", { onClick: () => d.close(false) });
    const go = async () => {
      if (d.busy) return;
      d.notice(null);
      if (!code.value()) { d.notice("danger", "Type a current code or a recovery code."); return; }
      d.setBusy(true, "Checking…");
      const r = await post("/c2c_vault/2fa/reset", { code: code.value() });
      if (r.ok) { d.close(true); return; }
      d.setBusy(false);
      d.notice("danger", authError(r, "That code is not right."));
      code.clear();
      code.input.focus();
    };
    d.addAction("Replace", { primary: true, onClick: go });
    d.onSubmit(go);
    if (!(await d.result)) return false;
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
  const title = v.title || "vault";
  let opened = null;
  const out = await credentialDialog({ title: `Open "${title}"`, primary: "Open vault",
    subtitle: "Type the vault password and the code your authenticator app shows.",
    submit: async ({ password, code }) => {
      const r = await post("/c2c_vault/sg/open", { payload, vault_id: vaultId, password, code, client_id: api.clientId });
      if (r.ok) { opened = r.data; return { ok: true }; }
      if (r.data.needs_enrolment) return { enrol: true };
      return { error: authError(r, "Could not open the vault.") };
    } });
  if (!out) return;
  if (out.enrol) {
    if (await enrolDialog()) await openVault(v);
    return;
  }
  try {
    await enterOpened(v, opened, payload);
  } catch (e) {
    toast(`The vault opened but could not be shown: ${e?.message || e}`, "error");
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
    // The vault node's CURRENT name wins: res.ui.title is the name at protect time (often 'New Subgraph'), so a vault
    // renamed later came back under its old name on every open (found by the H2Vault review, 2026-11-21).
    sg.title = v.title || res.ui.title;
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
    if (sess.profile === "locked") markRunUnlocked(sess.vaultId);
    toast("Vault locked again: what you changed is encrypted.", "success", "Vault locked");
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
    const ok = await credentialDialog({ title: `Run "${graphNode?.title || "this vault"}"`, needCode: false,
      primary: "Run", subtitle: "This vault is Locked: it needs its password once per browser tab.",
      submit: async ({ password }) => {
        const r = await post("/c2c_vault/sg/unlock_run", { payload: n.inputs.vault_payload, vault_id: vid, password,
          client_id: api.clientId });
        return r.ok ? { ok: true } : { error: authError(r, "Wrong password.") };
      } });
    if (!ok) throw new Error("Run cancelled: a Locked vault needs its password.");
    markRunUnlocked(vid);
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
/** The canvas selection as canvas items (nodes, groups, reroutes); _vault_selection.js decides what is locked. */
function selectedItemsOf(canvas) {
  const s = canvas?.selectedItems ?? canvas?.selected_nodes;
  return s instanceof Set ? [...s] : Object.values(s || {});
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

/** ... and put each back on the ONE subgraph input that feeds the same inner socket. A26 L9.203: the old fallback
 *  ("first unlinked input of the same type, in order") could swap two INT inputs; an unplaceable wire now throws. */
function reconnectParentInputs(graph, sg, wires) {
  const io = graph.inputNode;
  const sub = sg?.subgraph;
  if (!io || !sub) return;
  for (const w of wires) {
    const hits = [];
    (sub.inputs || []).forEach((si, i) => {
      if ((si.linkIds || []).some((lid) => {
        const L = linkOf(sub, lid);
        return L && String(L.target_id) === w.node && L.target_slot === w.input;
      })) hits.push(i);
    });
    if (hits.length !== 1) {
      throw new Error(`A wire from the surrounding subgraph into node ${w.node} cannot be restored unambiguously ` +
        `(${hits.length} matching inputs).`);
    }
    const idx = hits[0];
    if (sg.inputs[idx]?.link != null) continue;                 // the frontend kept it
    io.slots?.[w.slot]?.connect(sg.inputs[idx], sg);
  }
}

/**
 * V1 one-step lock: whatever is selected (nodes, a group, or one subgraph node) -> ONE dialog -> the A26 transaction
 * (convert to a native subgraph + protect, one undo step, exact rollback). `items` overrides the canvas selection
 * (the node menu on an unselected node). Resolves with the vault node or null.
 */
/** V6-A: a node (or a subgraph holding one) whose parameter values are locked (js/c2c_lock_params.js). */
function hasLockedValues(n, seen = new Set()) {
  if (!n || seen.has(n)) return false;
  seen.add(n);
  if (n.properties?.c2c_locked_params) return true;
  return isSubgraphNode(n) && (n.subgraph?.nodes || []).some((c) => hasLockedValues(c, seen));
}

async function lockSelection(canvas, profileHint, items = null) {
  const r = resolveLockSelection(items ?? selectedItemsOf(canvas), { LiteGraph: globalThis.LiteGraph });
  if (r.problems.length) { toast(r.problems[0], true); return null; }
  if (r.nodes.some((n) => hasLockedValues(n))) {
    toast("Some selected nodes have locked parameter values. Remove that lock first (right-click › Remove value " +
      "lock…): a vault encrypts their values anyway.", true);
    return null;
  }
  let boundary = null;
  if (r.kind === "nodes") {
    try { boundary = captureBoundary(r.nodes[0].graph, r.nodes); } catch (_) { boundary = { entering: [], leaving: [] }; }
  }
  const summary = summarizeSelection(r.nodes, boundary);
  // A failed attempt restores the graph by reloading it (new node objects, same ids) and the dialog stays open for a
  // retry: each attempt therefore takes the nodes by id from the graph on screen, never the stale objects.
  const ids = r.nodes.map((n) => n.id);
  const live = () => {
    if (r.nodes.every((n) => n.graph && n.graph.getNodeById?.(n.id) === n)) return r.nodes;
    const g = app.canvas?.graph;
    const again = ids.map((id) => g?.getNodeById?.(id)).filter(Boolean);
    if (again.length !== ids.length) {
      throw new ProtectError("SELECTION_VALIDATION", "The selected nodes are no longer in the graph on screen. " +
        "Close this dialog and select them again.");
    }
    return again;
  };
  const vault = await protectDialog({ count: r.nodes.length, kind: r.kind, summary, title: r.title,
    defaultProfile: profileHint || "sealed_installs", run: async (opts) => protectTransaction(live(), opts) });
  if (!vault) return null;
  try { canvas?.deselectAll?.(); canvas?.select?.(vault); } catch (_) { /* selection is a nicety */ }
  const what = r.kind === "subgraph" ? "the subgraph" : `${r.nodes.length} node${r.nodes.length === 1 ? "" : "s"}`;
  toast(`Locked ${what} into "${vault.title}". Double-click it to open (password + authenticator).`, "success",
    "Vault locked");
  return vault;
}

/** Protect without a dialog (browser tests, A26 L9.203). True / false; the outcome is told in a toast. */
async function protectWithFeedback(nodes, opts) {
  try {
    await protectTransaction(nodes, opts);
    toast("Locked. Double-click the vault to open it (password + authenticator).", "success", "Vault locked");
    return true;
  } catch (e) {
    // never a success message for a half-done protect: the original graph is back
    LOG("protect failed", e?.stage, e?.message);
    if (e?.busy) { toast(BUSY_MESSAGE, true); return false; }       // L9.249 V2: the running one is untouched
    toast(protectFailureText(e), "error", "Not locked");
    return false;
  }
}

async function showInstallCode() {
  const r = await (await api.fetchApi("/c2c_vault/sg/install_code")).json().catch(() => ({}));
  const d = vaultDialog({ title: "My installation code",
    subtitle: "Send this to a vault author so their Sealed vaults run here. It is a public key: safe to share.",
    width: 520 });
  const name = textField({ label: "Name the author will see (optional)", placeholder: "e.g. Studio B", maxLength: 60 });
  if (r.code) d.body.append(copyRow({ label: "installation code", value: r.code }));
  else d.body.append(noticeEl("danger", "This installation's code is not available. Check that the C2C Vault server part loaded."));
  d.body.append(name.el);
  if (r.code) d.addAction("Save key file", { onClick: () => saveKeyFile(r.code, name.input.value.trim()) });
  d.addAction("Done", { primary: true, onClick: () => d.close(true) });
  d.onSubmit(() => d.close(true));
  await d.result;
}

// ── access management (L9.48): installations, editors, password ─────────────────────────────────────────────
const KEYFILE_FORMAT = "c2c-install-key";
const CODE_PREFIX = "c2cvk1:";

/** <name>.c2ckey: this installation's PUBLIC key as a file (docs/VAULT_DESIGN_A18.md, Access management). */
function saveKeyFile(code, label) {
  const data = { format: KEYFILE_FORMAT, version: 1, code, label: label || "", created: new Date().toISOString() };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: "application/json" }));
  const a = el("a", { href: url, download: `${(label || "c2c-installation").replace(/[^\w.-]+/g, "_").slice(0, 40)}.c2ckey` });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
}

/** Pick a key file -> {code, label}, {error} or null (cancelled). A text file holding just a code works too. */
function importKeyFile() {
  return new Promise((resolve) => {
    const inp = el("input", { type: "file", accept: ".c2ckey,.json,.txt", style: { display: "none" } });
    const done = (v) => { inp.remove(); resolve(v); };
    inp.addEventListener("cancel", () => done(null));
    inp.addEventListener("change", async () => {
      const f = inp.files?.[0];
      if (!f) return done(null);
      const text = (await f.text().catch(() => "")).trim();
      try {
        const d = JSON.parse(text);
        if (d?.format === KEYFILE_FORMAT && typeof d.code === "string") {
          return done({ code: d.code.trim(), label: String(d.label || "") });
        }
      } catch (_) { /* not JSON: maybe a bare code */ }
      if (text.startsWith(CODE_PREFIX)) return done({ code: text.split(/\s+/)[0], label: f.name.replace(/\.[^.]+$/, "") });
      done({ error: `${f.name} is not a C2C installation key file.` });
    });
    document.body.append(inp);
    inp.click();
  });
}

function shortCode(code) {
  return code.length > 30 ? `${code.slice(0, 18)}…${code.slice(-8)}` : code;
}

/** The access dialog (spec 4.5). `apply(choice)` -> {ok} or {error}; on an error the dialog stays open with the
 *  message. Resolves with apply's result, or null when cancelled. */
function accessDialog(v, res, focus, apply, issue) {
  const sealed = res.profile === "sealed_installs";
  const d = vaultDialog({ title: `${focus === "password" ? "Change the password of" : "Access to"} "${v.title || "vault"}"`,
    subtitle: sealed ? "Who may run this vault, who may open it for editing, and its password."
      : "Who may open this vault for editing, and its password.", width: 640 });
  const lists = { editors: res.editors.map((r) => ({ ...r })), recipients: res.recipients.map((r) => ({ ...r })) };
  const setLabel = (code, label) => {
    for (const r of [...lists.editors, ...lists.recipients]) if (r.code === code) r.label = label;
  };
  const section = (key, title, hint) => {
    const rows = vel("div", { class: "c2c-vault-list" });
    const draw = () => rows.replaceChildren(...lists[key].map((r, i) => {
      const name = vel("input", { type: "text", class: "c2c-vault-input", value: r.label || "", maxLength: 60,
        placeholder: r.me ? "This installation" : "Name (optional)", "aria-label": `Name for installation ${shortCode(r.code)}`,
        on: { input: (e) => setLabel(r.code, e.target.value) } });
      return vel("div", { class: "c2c-vault-keyrow" }, [name,
        vel("span", { class: "c2c-vault-keycode", text: shortCode(r.code), title: r.code }),
        r.me ? vel("span", { class: "c2c-vault-badge", text: "This installation" })
          : vbutton("Remove", { small: true, ariaLabel: `Remove installation ${shortCode(r.code)}`,
            onClick: () => { lists[key].splice(i, 1); draw(); } })]);
    }));
    const code = vel("input", { type: "text", class: "c2c-vault-input", placeholder: "Paste an installation code (c2cvk1:...)",
      "aria-label": `Add to ${title.toLowerCase()}`, spellcheck: false });
    const add = (c, label) => {
      c = String(c || "").trim();
      if (!c) return;
      if (!c.startsWith(CODE_PREFIX)) { d.notice("danger", `An installation code starts with ${CODE_PREFIX}`); return; }
      if (lists[key].some((r) => r.code === c)) { d.notice("warn", "That installation is already in the list."); return; }
      const known = [...lists.editors, ...lists.recipients].find((r) => r.code === c);
      lists[key].push({ code: c, label: label || known?.label || "", me: false });
      code.value = "";
      d.notice(null);
      draw();
    };
    code.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); add(code.value); } });
    draw();
    return block(title, hint, rows, vel("div", { class: "c2c-vault-row" }, [
      vel("div", { style: { flex: "1", minWidth: "0" } }, [code]),
      vbutton("Add", { onClick: () => add(code.value) }),
      vbutton("Import key file…", { onClick: async () => {
        const k = await importKeyFile();
        if (k?.error) d.notice("danger", k.error);
        else if (k) add(k.code, k.label);
      } })]));
  };
  if (sealed) {
    d.body.append(section("recipients", "Installations that may run it",
      "Without a password, only on these ComfyUI installations. This one is always included."));
  } else {
    d.body.append(noticeEl("info", res.profile === "locked" ? "Locked vault: whoever has the password can run it, so " +
      "there are no installations to manage here." : "Portable vault: it runs anywhere (obfuscation only)."));
  }
  d.body.append(section("editors", "Editors", "Installations that may open it for editing - each with the password " +
    "and its own authenticator. This one stays an editor."));
  // V6-B: offline access keys with an expiry date (Sealed vaults only)
  const ak = res.access_keys || {};
  let requireKey = null, revoke = null;
  if (ak.available) {
    requireKey = switchRow("Require an access key to run", { checked: !!ak.required });
    revoke = vel("input", { type: "checkbox" });
    const kids = [requireKey.el, vel("p", { class: "c2c-vault-block__hint", text: "Installations then run it only with " +
      "a key you issue, valid until a date you choose. Editors never need one. Keys work offline: a key someone already " +
      "has keeps opening the copies they have until it expires; revoking stops it opening new copies." })];
    if (ak.required) {
      kids.push(vel("label", { class: "c2c-vault-check" }, [revoke,
        vel("span", { text: `Revoke every key issued so far (key generation ${ak.gen} → ${ak.gen + 1})` })]));
      if (ak.issuer) {
        kids.push(vel("div", { class: "c2c-vault-row" }, [vbutton("Issue an access key…", {
          onClick: () => issueKeyDialog(lists.recipients.filter((r) => !r.me), issue) })]));
      } else {
        kids.push(noticeEl("info", "Keys for this vault are issued by the installation that turned them on."));
      }
    } else {
      kids.push(vel("p", { class: "c2c-vault-dim", text: "Turn it on and apply; then open this dialog again to issue keys." }));
    }
    d.body.append(vel("section", { class: "c2c-vault-block" }, [
      vel("h3", { class: "c2c-vault-block__title", text: "Access keys" }), ...kids]));
  }
  const pw = passwordField({ label: "New vault password", strength: true });
  const pw2 = passwordField({ label: "Repeat the new password" });
  const pwBox = vel("div", { class: "c2c-vault-grid2" }, [pw.el, pw2.el]);
  const change = switchRow("Change the password", { checked: focus === "password",
    onChange: (on) => { pwBox.style.display = on ? "" : "none"; if (on) pw.input.focus(); } });
  pwBox.style.display = change.checked() ? "" : "none";
  d.body.append(vel("div", { class: "c2c-vault-block" }, [change.el, pwBox]));
  d.body.append(noticeEl("info", (sealed ? "Removing an installation protects new copies only: a copy it already has " +
    "still runs there. " : "") + "Applying re-encrypts the vault with fresh keys; save the workflow afterwards so the " +
    "new access goes with it."));
  d.addAction("Cancel", { onClick: () => d.close(null) });
  const go = async () => {
    if (d.busy) return;
    d.notice(null);
    let newPassword = null;
    if (change.checked()) {
      if (pw.value().length < 8) { d.notice("danger", "Use at least 8 characters for the new password."); pw.input.focus(); return; }
      if (pw.value() !== pw2.value()) { d.notice("danger", "The two new passwords differ."); pw2.input.focus(); return; }
      newPassword = pw.value();
    }
    const labels = {};
    for (const r of [...lists.editors, ...lists.recipients]) if (String(r.label || "").trim()) labels[r.code] = r.label.trim();
    d.setBusy(true, "Applying…");
    d.status("Re-encrypting the vault…");
    const out = await apply({ editors: lists.editors.map((r) => r.code), recipients: lists.recipients.map((r) => r.code),
      labels, new_password: newPassword,
      ...(requireKey ? { require_access_key: requireKey.checked(), revoke_access_keys: !!revoke?.checked } : {}) });
    if (out?.error) { d.setBusy(false); d.status(""); d.notice("danger", out.error); return; }
    d.close(out);
  };
  d.addAction("Apply", { primary: true, onClick: go });
  if (focus === "password") setTimeout(() => pw.input.focus(), 0);
  return d.result;
}

// ── V6-B access keys: issue (author) and enter (recipient) ───────────────────────────────────────────────────
const ACCESS_KEY_RE = /c2cak1\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/;

function isoDay(date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function saveAccessKeyFile(key, name) {
  const url = URL.createObjectURL(new Blob([key + "\n"], { type: "text/plain" }));
  const a = el("a", { href: url, download: `${(name || "vault-access-key").replace(/[^\w.-]+/g, "_").slice(0, 40)}.c2caccess` });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
}

/** Author: sign a key for some of the vault's installations, until a date. `issue(choice)` -> {data} or {error}. */
async function issueKeyDialog(installs, issue) {
  const d = vaultDialog({ title: "Issue an access key", width: 520,
    subtitle: "A signed key that lets the installations you choose run this vault until a date." });
  if (!installs.length) {
    d.body.append(noticeEl("warn", "This vault has no other installations yet. Add one under \"Installations that may " +
      "run it\", apply, then issue its key."));
    d.addAction("Close", { primary: true, onClick: () => d.close(null) });
    return d.result;
  }
  const boxes = installs.map((r) => {
    const box = vel("input", { type: "checkbox", checked: installs.length === 1 });
    const row = vel("label", { class: "c2c-vault-check" }, [box,
      vel("span", { text: r.label ? `${r.label} · ${shortCode(r.code)}` : shortCode(r.code), title: r.code })]);
    return { code: r.code, box, row };
  });
  const soon = new Date(Date.now() + 30 * 86400000);
  const date = vel("input", { type: "date", class: "c2c-vault-input", value: isoDay(soon), min: isoDay(new Date(Date.now() + 86400000)) });
  const note = textField({ label: "Note (optional, inside the key)", placeholder: "e.g. Client review, March", maxLength: 60 });
  d.body.append(block("Installations", "The key works only on these.", vel("div", { class: "c2c-vault-list" }, boxes.map((b) => b.row))),
    vel("div", { class: "c2c-vault-grid2" }, [vel("div", { class: "c2c-vault-field" }, [
      vel("label", { class: "c2c-vault-label", text: "Valid until (end of day, UTC)", for: "c2c-ak-date" }), date]), note.el]));
  date.id = "c2c-ak-date";
  const result = vel("div");
  d.body.append(result);
  d.addAction("Close", { onClick: () => d.close(null) });
  const go = async () => {
    if (d.busy) return;
    d.notice(null);
    const chosen = boxes.filter((b) => b.box.checked).map((b) => b.code);
    if (!chosen.length) { d.notice("danger", "Choose at least one installation."); return; }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date.value)) { d.notice("danger", "Choose the last day the key works."); date.focus(); return; }
    d.setBusy(true, "Signing…");
    const out = await issue({ installs: chosen, expires: date.value, note: note.input.value.trim() });
    d.setBusy(false);
    if (out?.error) { d.notice("danger", out.error); return; }
    const k = out.data;
    result.replaceChildren(copyRow({ label: "access key", value: k.key }),
      noticeEl("ok", `Key for ${k.installs} installation${k.installs === 1 ? "" : "s"}, valid until the end of ${k.expires} ` +
        "(UTC). Send it with the workflow; they enter it once (right-click the vault › Enter access key)."),
      vel("div", { class: "c2c-vault-row" }, [vbutton("Save key file", { onClick: () => saveAccessKeyFile(k.key,
        note.input.value.trim() || `vault-key-${k.expires}`) })]));
  };
  d.addAction("Issue key", { primary: true, onClick: go });
  d.onSubmit(go);
  return d.result;
}

let ACCESS_OFFER_OPEN = false;

/** Recipient: enter (paste or open a file) the access key its author sent. `reason`: why a run was refused. */
async function enterAccessKey(v, { reason = "" } = {}) {
  if (ACCESS_OFFER_OPEN) return;
  ACCESS_OFFER_OPEN = true;
  try {
    const W = (name) => v.widgets.find((w) => w.name === name);
    const d = vaultDialog({ title: `Access key for "${v.title || "vault"}"`, width: 520,
      subtitle: "Paste the key the vault's author sent you, or open the key file. It is checked on this installation." });
    if (reason) d.body.append(noticeEl("warn", reason));
    const ta = textArea({ label: "Access key", placeholder: "c2cak1.…", rows: 4 });
    d.body.append(ta.el, vel("div", { class: "c2c-vault-row" }, [vbutton("Open key file…", { onClick: async () => {
      const text = await pickTextFile(".c2caccess,.txt");
      if (text == null) return;
      const m = text.match(ACCESS_KEY_RE);
      if (m) { ta.input.value = m[0]; d.notice(null); } else d.notice("danger", "That file does not hold a C2C vault access key.");
    } })]));
    d.addAction("Cancel", { onClick: () => d.close(null) });
    const go = async () => {
      if (d.busy) return;
      d.notice(null);
      const m = ta.input.value.match(ACCESS_KEY_RE);
      if (!m) { d.notice("danger", "Paste the whole key: it starts with c2cak1."); ta.input.focus(); return; }
      d.setBusy(true, "Checking…");
      const r = await post("/c2c_vault/sg/access_key/add", { token: m[0], payload: W("vault_payload")?.value,
        vault_id: W("vault_id")?.value, client_id: api.clientId });
      d.setBusy(false);
      if (!r.ok) { d.notice("danger", r.data.error || `The key was not accepted (${r.status}).`); return; }
      d.close(r.data);
    };
    d.addAction("Use this key", { primary: true, onClick: go });
    const out = await d.result;
    if (out) toast(`This installation may run "${v.title || "the vault"}" until the end of ${out.expires} (UTC). Run it again.`,
      "success", "Access key accepted");
  } finally { ACCESS_OFFER_OPEN = false; }
}

function pickTextFile(accept) {
  return new Promise((resolve) => {
    const inp = el("input", { type: "file", accept, style: { display: "none" } });
    const done = (v) => { inp.remove(); resolve(v); };
    inp.addEventListener("cancel", () => done(null));
    inp.addEventListener("change", async () => {
      const f = inp.files?.[0];
      done(f && f.size < 65536 ? await f.text().catch(() => null) : null);
    });
    document.body.append(inp);
    inp.click();
  });
}

// ── missing custom nodes (L9.50): one click, names never shown (owner A18) ─────────────────────────────────
let INSTALL_OFFER_OPEN = false;
const MISSING_PREFIX = "This workflow needs custom nodes that are not installed";

/** Offer to install what this vault needs here. `auto`: from a failed run (asked on every run attempt, owner). */
async function installMissing(v, { auto = false } = {}) {
  if (INSTALL_OFFER_OPEN) return;
  INSTALL_OFFER_OPEN = true;
  try {
    const W = (name) => v.widgets.find((w) => w.name === name);
    const body = { payload: W("vault_payload")?.value, vault_id: W("vault_id")?.value, client_id: api.clientId };
    const m = await post("/c2c_vault/sg/missing", body);
    if (!m.ok) { toast(m.data.error || "Could not check this vault.", true); return; }
    const s = m.data;
    if (!s.missing) {
      if (!auto) toast("Nothing is missing: this vault's custom nodes are all installed here.");
      return;
    }
    const n = s.packs, many = n !== 1;
    const ask = vaultDialog({ title: "Custom nodes needed",
      subtitle: "This workflow's vault uses custom nodes that this ComfyUI does not have.", width: 500 });
    if (s.installable) {
      ask.body.append(vel("p", { style: { margin: "0" }, text: `It needs ${n} custom node pack${many ? "s" : ""} that ` +
        `${many ? "are" : "is"} not installed here (from ${s.hosts.join(", ")}). Without ${many ? "them" : "it"} the ` +
        "workflow will not run." }));
      if (s.installable < n || s.unknown_classes) {
        ask.body.append(noticeEl("warn", "Some of them cannot be installed from here - ask the vault's author for those."));
      }
      ask.body.append(noticeEl("warn", `Installing downloads ${many ? "them" : "it"} and runs ${many ? "their" : "its"} ` +
        "code when ComfyUI restarts. Install only if you trust where this workflow comes from."));
      ask.addAction("Cancel", { onClick: () => ask.close(false) });
      ask.addAction("Install", { primary: true, onClick: () => ask.close(true) });
    } else {
      ask.body.append(vel("p", { style: { margin: "0" }, text: "The vault does not say where to get them. Ask the " +
        "vault's author which ones to install." }));
      ask.addAction("Close", { primary: true, onClick: () => ask.close(false) });
    }
    const go = await ask.result;
    if (!go) return;
    toast("Installing the custom nodes this workflow needs…");
    const r = await post("/c2c_vault/sg/install_missing", { ...body, consent: true });
    const d = r.data || {};
    const lines = [];
    if (!r.ok) lines.push(["danger", d.error || `Installing failed (${r.status}).`]);
    if (d.installed) lines.push(["ok", `Installed ${d.installed} pack${d.installed !== 1 ? "s" : ""}. Restart ComfyUI to ` +
      `load ${d.installed !== 1 ? "them" : "it"}, then run the workflow again.`]);
    if (d.skipped) lines.push(["info", `${d.skipped} folder${d.skipped !== 1 ? "s" : ""} already existed and ` +
      `${d.skipped !== 1 ? "were" : "was"} left as ${d.skipped !== 1 ? "they were" : "it was"}.`]);
    for (const f of d.failed || []) lines.push(["danger", `Not installed: ${f}.`]);
    if (d.requirements?.note) lines.push(["info", `Their Python requirements: ${d.requirements.note}.`]);
    if (!lines.length) lines.push(["info", "Nothing was installed."]);
    const done = vaultDialog({ title: "Custom nodes", tone: r.ok ? "ok" : "danger", width: 500 });
    done.body.append(...lines.map(([tone, t]) => noticeEl(tone, t)));
    done.addAction("OK", { primary: true, onClick: () => done.close(true) });
    done.onSubmit(() => done.close(true));
    await done.result;
  } finally { INSTALL_OFFER_OPEN = false; }
}

/** The node an execution id names ("5", or "12:5" inside a subgraph). */
function nodeByExecId(id) {
  let g = app.rootGraph ?? app.graph, n = null;
  for (const part of String(id).split(":")) {
    n = g?.getNodeById?.(Number(part)) ?? g?.getNodeById?.(part) ?? null;
    if (!n) return null;
    g = n.subgraph;
  }
  return n;
}

function onExecutionError(e) {
  const d = e?.detail || {};
  if (VAULT_TYPES.has(d.node_type) && /access key/i.test(String(d.exception_message || ""))) {     // V6-B
    const v = nodeByExecId(d.node_id);
    if (v) setTimeout(() => enterAccessKey(v, { reason: String(d.exception_message) }), 0);
    return;
  }
  if (!VAULT_TYPES.has(d.node_type) || !String(d.exception_message || "").startsWith(MISSING_PREFIX)) return;
  const v = nodeByExecId(d.node_id);
  if (v) setTimeout(() => installMissing(v, { auto: true }), 0);
}

async function manageAccess(v, focus) {
  const st = await (await api.fetchApi("/c2c_vault/2fa/status")).json().catch(() => ({}));
  if (!st.enrolled) {
    toast("Managing a vault's access needs an authenticator app on this installation. Set it up now.");
    if (!(await enrolDialog())) return;
  }
  const W = (name) => v.widgets.find((w) => w.name === name);
  const payload = W("vault_payload")?.value;
  const vaultId = W("vault_id")?.value;
  let res = null;
  const verb = focus === "password" ? "Change the password of" : "Manage access to";
  const ok = await credentialDialog({ title: `${verb} "${v.title || "vault"}"`, primary: "Continue",
    subtitle: "Confirm with the vault password and your authenticator code. The inside stays closed.",
    submit: async ({ password, code }) => {
      const r = await post("/c2c_vault/sg/access/open", { payload, vault_id: vaultId, password, code,
        client_id: api.clientId });
      if (r.ok) { res = r.data; return { ok: true }; }
      if (r.data.needs_enrolment) return { enrol: true };
      return { error: authError(r, "Could not unlock the vault's access.") };
    } });
  if (!ok) return;
  if (ok.enrol) {
    if (await enrolDialog()) await manageAccess(v, focus);
    return;
  }
  const out = await accessDialog(v, res, focus, async (choice) => {
    const r = await post("/c2c_vault/sg/access/apply", { access_token: res.access_token, payload, ...choice });
    if (!r.ok) return { error: r.data.error || `Could not apply the change (${r.status}).` };
    W("vault_payload").value = r.data.payload;                  // swapped only now: a failure left the old vault
    if (res.profile === "locked") markRunUnlocked(vaultId);
    v.setDirtyCanvas?.(true, true);
    try { app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.(); } catch (_) {}
    return { ok: true, data: r.data };
  }, async (choice) => {                                        // V6-B: issue a key inside the same access session
    const r = await post("/c2c_vault/sg/access/issue_key", { access_token: res.access_token, payload, ...choice });
    return r.ok ? { data: r.data } : { error: r.data.error || `Could not issue the key (${r.status}).` };
  });
  if (!out) { post("/c2c_vault/sg/access/cancel", { access_token: res.access_token }); return; }
  toast(out.data.password_changed
    ? "Password changed and the vault re-encrypted. Save the workflow - copies saved before still open with the old password."
    : "Access updated and the vault re-encrypted. Save the workflow so the new access goes with it.", "success",
  out.data.password_changed ? "Password changed" : "Access updated");
}

const LOCK_CMD = "c2c.vault.lockSelectionAsVault";
const LOCK_LABEL = "🔒 Lock as C2C Vault (password to run)…";
const SEAL_LABEL = "🔒 Seal as C2C Vault (authorized installations)…";

/** The two lock entries (canvas and node menus) for `items`; [] when nothing lockable is in them. */
function lockMenuItems(items) {
  const r = resolveLockSelection(items, { LiteGraph: globalThis.LiteGraph });
  if (r.kind === "empty") return [];
  if (r.kind === "subgraph") {
    return [null, { content: "🔒 Protect this subgraph as C2C Vault…",
      callback: () => lockSelection(app.canvas, "sealed_installs", items) }];
  }
  return [null,
    { content: LOCK_LABEL, callback: () => lockSelection(app.canvas, "locked", items) },
    { content: SEAL_LABEL, callback: () => lockSelection(app.canvas, "sealed_installs", items) }];
}

app.registerExtension({
  name: "C2C.SecureSubgraph",
  // V1: one command for the palette, the selection toolbox (js/c2c_vault.js returns it) and Alt+Shift+L ("L" for lock).
  // Not Ctrl+Shift+L: KJNodes binds it (ToggleForceShowSetGetLinks) and the frontend refuses the duplicate. Not
  // Ctrl+Alt+<key>: that is AltGr on many European Windows layouts, and Ctrl+Alt+V is the explicit MEC paste
  // (js/c2c_clipboard.js). Alt+Shift+L is unused by core 1.53.10, KJNodes and every C2C pack (scanned 2026-11-23).
  commands: [{ id: LOCK_CMD, label: "Lock selection as C2C Vault…", icon: "pi pi-lock",
    function: () => lockSelection(app.canvas, null) }],
  keybindings: [{ combo: { key: "l", alt: true, shift: true }, commandId: LOCK_CMD }],
  setup() {
    installGuards();
    app.canvas?.canvas?.addEventListener("litegraph:set-graph", onSetGraph);
    api.addEventListener("execution_error", onExecutionError);      // L9.50: offer the install on every attempt
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
      // V1: the profile / node count / run state moved from canvas text above the title (classic only) into the
      // status card in the body (both renderers); node.__c2cBadge is still set for anything that reads it.
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
    return lockMenuItems(selectedItemsOf(canvas));
  },
  getNodeMenuItems(node) {
    const type = node?.comfyClass ?? node?.type;
    if (VAULT_TYPES.has(type)) {
      return [null,
        { content: "🔓 Open / edit vault…", callback: () => openVault(node) },
        { content: "👥 Manage installations & editors…", callback: () => manageAccess(node, "access") },
        { content: "🔐 Change password…", callback: () => manageAccess(node, "password") },
        { content: "📦 Install missing custom nodes…", callback: () => installMissing(node) },
        { content: "🎫 Enter access key…", callback: () => enterAccessKey(node) },
        { content: "🔑 My installation code", callback: showInstallCode },
        { content: "📱 Two-factor setup…", callback: () => twoFactorSetup() }];
    }
    if (LEGACY_TYPES.has(type) || OPEN.has(node) || node?.id == null || Number(node.id) < 0) return [];
    // V1: right-clicking a node is where people look. It acts on the selection when the node is part of it,
    // otherwise on that node alone; one subgraph node keeps "Protect this subgraph".
    const sel = selectedItemsOf(app.canvas);
    return lockMenuItems(sel.includes(node) ? sel : [node]);
  },
});

// the old vault front end (js/c2c_vault.js) routes its command-palette Lock / Seal and "Seal whole workflow" here
globalThis.__c2cVaultNative = { lockSelection: (profile) => lockSelection(app.canvas, profile),
  // browser tests (A26 L9.203): protect given nodes without the dialog
  protectNodes: (nodes, opts) => protectWithFeedback(nodes, opts), protectTransaction,
  // V6-A (js/c2c_lock_params.js): parameter values are never locked inside a vault being edited
  inOpenVault: (graph) => [...OPEN.keys()].some((sg) => contains(sg.subgraph, graph)) };

export { buildApi, exportUi, scrub, OPEN };
