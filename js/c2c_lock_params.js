// c2c_lock_params.js - "Lock parameter values" (V6-A, owner 2026-11-21). A node, a selection or a group keeps its
// place, sockets and wires; the values of its widgets are encrypted IN the workflow with a password that never
// leaves this browser. Runs ask for the password once per tab. Design: docs/h2vault/V6_cnp_vault_enhancements.md.
//
// Where the real values live: only in this tab's memory after an unlock (UNLOCKED, by envelope id) and in the one
// request that queues a prompt (api.queuePrompt is wrapped; graphToPrompt is NOT, so Save, autosave, undo, Export,
// Export (API), copy-paste and the PNG workflow all carry placeholders + ciphertext). The server
// (nodes/lock_params.py) keeps them out of history, PNG metadata and the queue views.
//
// License: Apache-2.0
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { LOCK_PROP, STATUS_WIDGET, LockError, deriveKey, envelopeOf, injectLockedValues, lockableWidgets, newSalt,
  placeholderFor, sealValues, unlockEnvelopes, PBKDF2_ITERATIONS } from "./_lockparams_core.js";
import { capsule, el, noticeEl, passwordField, statusCard, vaultDialog } from "./_vault_ui.js";
import { resolveLockSelection } from "./_vault_selection.js";
import { setWidgetVisible, vueFitNodeHeight, vueNodesActive, vueSyncNodeWidgets } from "./_widget_visibility.js";
import { tagDomWidget } from "./_vue_dom_rebind.js";

const UNLOCKED = new Map();                    // envelope id -> {name: value}, this tab only
const VAULT_TYPES = new Set(["C2C_SecureSubgraph", "C2C_SecureSubgraphOutput", "C2C_VaultLocked", "C2C_VaultSealed"]);
const CARD_H = 56;
const MIN_W = 260;
const CMD = "c2c.lockParams.lockSelection";

function toast(text, severity = "info", summary = "Locked values") {
  try {
    app.extensionManager?.toast?.add({ severity, summary, detail: text, life: severity === "error" ? 8000 : 5000 });
  } catch (_) { /* a toast is a nicety */ }
}

const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
const isSubgraphNode = (n) => { try { return !!n?.isSubgraphNode?.(); } catch (_) { return false; } };
const rootGraph = () => app.rootGraph ?? app.graph;

function checkState() {
  try { app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.(); } catch (_) { /* undo nicety */ }
}

/** Every node of a graph, nested subgraphs included. */
function* allNodes(g, seen = new Set()) {
  if (!g || seen.has(g)) return;
  seen.add(g);
  for (const n of g.nodes || g._nodes || []) {
    yield n;
    if (isSubgraphNode(n)) yield* allNodes(n.subgraph, seen);
  }
}

/** The node an execution id names ("5", or "12:5" inside a subgraph). */
function nodeByExecId(id) {
  let g = rootGraph(), n = null;
  for (const part of String(id).split(":")) {
    n = g?.getNodeById?.(Number(part)) ?? g?.getNodeById?.(part) ?? null;
    if (!n) return null;
    g = n.subgraph;
  }
  return n;
}

function inOpenVault(node) {
  try { return !!globalThis.__c2cVaultNative?.inOpenVault?.(node?.graph); } catch (_) { return false; }
}

// ── the locked look: hidden widgets + a status card (both renderers) ──────────────────────────────────────────────
function cardOf(node) { return node?.widgets?.find((w) => w.name === STATUS_WIDGET) ?? null; }

function ensureCard(node) {
  if (typeof document === "undefined" || typeof node?.addDOMWidget !== "function") return;
  let w = cardOf(node);
  if (!w) {
    const card = statusCard();
    w = node.addDOMWidget(STATUS_WIDGET, STATUS_WIDGET, card.el, { serialize: false, margin: 4,
      getMinHeight: () => CARD_H, getHeight: () => CARD_H });
    w.serialize = false;                       // fe 1.53 saves any widget whose own .serialize is not false (L9.237)
    w.computeSize = (width) => [width, CARD_H + 8];
    w.__lockCard = card;
    try { tagDomWidget(w); } catch (_) { /* the Nodes 2.0 rebind tag is a nicety */ }
  }
  const ws = node.widgets || [];
  const at = ws.indexOf(w);
  if (at >= 0 && at !== ws.length - 1) { ws.splice(at, 1); ws.push(w); }   // always last: saved positions never move
  if (!node.__lockMinWidth && typeof node.computeSize === "function") {
    const cs = node.computeSize;
    node.computeSize = function (...a) {
      const r = cs.apply(this, a);
      if (Array.isArray(r) && envelopeOf(this)) r[0] = Math.max(r[0], MIN_W);
      return r;
    };
    node.__lockMinWidth = true;
  }
  updateCard(node);
}

function updateCard(node) {
  const card = cardOf(node)?.__lockCard;
  const env = envelopeOf(node);
  if (!card || !env) return;
  const open = UNLOCKED.has(env.id);
  card.update({ tone: open ? "ok" : "accent", title: open ? "Values locked · unlocked here" : "Values locked",
    detail: `${plural(env.names.length, "value")} encrypted in the workflow`,
    meta: open ? "Runs in this tab without asking" : "Asks for the password at run time", metaTone: open ? "ok" : "",
    action: open ? null : { label: "Unlock…", ariaLabel: `Unlock the values of ${node.title || "this node"}`,
      onClick: () => unlockDialog([node]) } });
}

function relayout(node) {
  vueSyncNodeWidgets(node);
  if (vueNodesActive()) vueFitNodeHeight(node);
  else {
    try { node.setSize?.([Math.max(node.size?.[0] ?? 0, MIN_W), node.computeSize?.()[1] ?? node.size?.[1]]); }
    catch (_) { /* size is cosmetic */ }
  }
  node.setDirtyCanvas?.(true, true);
}

/** Show a node as locked: its locked widgets keep their placeholder and are hidden; the card says what happened. */
function applyLocked(node) {
  const env = envelopeOf(node);
  if (!env) return;
  for (const name of env.names) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w) continue;
    const ph = placeholderFor(w);
    if (w.value !== ph) w.value = ph;
    setWidgetVisible(w, false);
  }
  ensureCard(node);
  relayout(node);
}

function removeCard(node) {
  const w = cardOf(node);
  if (!w) return;
  try {
    if (typeof node.removeWidget === "function") node.removeWidget(w);
    else { node.widgets.splice(node.widgets.indexOf(w), 1); w.onRemove?.(); }
  } catch (_) { /* already gone */ }
}

/** Back to a normal node: the real values in the widgets, widgets shown, no envelope. */
function applyUnlocked(node, values) {
  const env = envelopeOf(node);
  for (const name of env?.names || []) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w) continue;
    setWidgetVisible(w, true);
    if (values && name in values) w.value = values[name];
  }
  if (node.properties) delete node.properties[LOCK_PROP];
  removeCard(node);
  relayout(node);
}

function refreshCards() {
  for (const n of allNodes(rootGraph())) if (envelopeOf(n)) updateCard(n);
}

// ── what a lock acts on ─────────────────────────────────────────────────────────────────────────────────────────
function selectedItems(canvas) {
  const s = canvas?.selectedItems ?? canvas?.selected_nodes;
  return s instanceof Set ? [...s] : Object.values(s || {});
}

/** The nodes of a selection (groups and subgraph contents included), split into what can lock and what cannot. */
function collect(items) {
  const r = resolveLockSelection(items, { LiteGraph: globalThis.LiteGraph });
  const nodes = [];
  const seen = new Set();
  const add = (n) => {
    if (!n || seen.has(n)) return;
    seen.add(n);
    if (isSubgraphNode(n)) { for (const c of allNodes(n.subgraph)) if (!isSubgraphNode(c)) add(c); return; }
    nodes.push(n);
  };
  for (const n of r.nodes) add(n);
  const out = { lockable: [], locked: [], refused: [], skipped: [] };
  for (const n of nodes) {
    const type = n.comfyClass ?? n.type;
    if (VAULT_TYPES.has(type) || inOpenVault(n)) { out.refused.push(n); continue; }
    if (envelopeOf(n)) { out.locked.push(n); continue; }
    const { lock, skipped } = lockableWidgets(n);
    if (skipped.length) out.skipped.push(`${n.title || type}: ${skipped.join(", ")}`);
    if (lock.length) out.lockable.push({ node: n, widgets: lock });
  }
  return out;
}

// ── dialogs ─────────────────────────────────────────────────────────────────────────────────────────────────────
async function lockDialog(items) {
  const c = collect(items);
  if (!c.lockable.length) {
    toast(c.locked.length ? "These values are already locked." : c.refused.length
      ? "Vaults and nodes inside an open vault keep their values encrypted already." : "Nothing here has values to lock.",
    "warn");
    return false;
  }
  const count = c.lockable.reduce((s, x) => s + x.widgets.length, 0);
  const d = vaultDialog({ title: "Lock parameter values",
    subtitle: "The nodes stay where they are. Their values are encrypted in the workflow with your password.",
    width: 520 });
  const titles = c.lockable.map((x) => x.node.title || x.node.type);
  const cap = capsule({ chips: titles.slice(0, 6), more: Math.max(0, titles.length - 6) });
  cap.querySelector(".c2c-vault-capsule__meta")?.remove();
  cap.append(el("div", { class: "c2c-vault-capsule__meta",
    text: `${plural(count, "value")} in ${plural(c.lockable.length, "node")} will be locked` }));
  d.body.append(cap);
  if (c.locked.length) d.body.append(noticeEl("info", `${plural(c.locked.length, "node")} already locked stay as they are.`));
  if (c.refused.length) d.body.append(noticeEl("info", `${plural(c.refused.length, "vault node")} skipped: a vault encrypts its inside already.`));
  if (c.skipped.length) d.body.append(noticeEl("warn", `Not lockable (custom editors keep their value visible):\n${c.skipped.join("\n")}`));
  const pw = passwordField({ label: "Password", strength: true, generate: true });
  const pw2 = passwordField({ label: "Repeat the password" });
  d.body.append(el("div", { class: "c2c-vault-grid2" }, [pw.el, pw2.el]));
  d.body.append(noticeEl("info", "Each browser tab asks for this password once, at the first run. Without it the " +
    "values cannot be recovered. A locked seed does not change between runs."));
  d.addAction("Cancel", { onClick: () => d.close(false) });
  const go = async () => {
    if (d.busy) return;
    d.notice(null);
    const p = pw.value();
    if (p.length < 8) { d.notice("danger", "Use at least 8 characters."); pw.input.focus(); return; }
    if (p !== pw2.value()) { d.notice("danger", "The two passwords differ."); pw2.input.focus(); return; }
    d.setBusy(true, "Locking…");
    d.status("Deriving the key…");
    try {
      const salt = newSalt();
      const key = await deriveKey(p, salt, { iterations: PBKDF2_ITERATIONS });
      d.status("Encrypting…");
      const sealed = [];
      for (const { node, widgets } of c.lockable) {
        const values = Object.fromEntries(widgets.map((w) => [w.name, w.value]));
        sealed.push({ node, values, env: await sealValues({ key, salt, iterations: PBKDF2_ITERATIONS,
          type: node.comfyClass ?? node.type, values }) });
      }
      for (const { node, values, env } of sealed) {       // nothing changes until every node is sealed
        node.properties = node.properties || {};
        node.properties[LOCK_PROP] = env;
        UNLOCKED.set(env.id, values);                       // the author's tab can run it right away
        applyLocked(node);
      }
      checkState();
      d.close(true);
      toast(`Locked ${plural(count, "value")} in ${plural(sealed.length, "node")}. Save the workflow to keep the locked form.`,
        "success", "Values locked");
    } catch (e) {
      d.setBusy(false);
      d.status("");
      d.notice("danger", `Nothing was locked: ${e?.message || e}`);
    }
  };
  d.onSubmit(go);
  d.addAction(`Lock ${plural(count, "value")}`, { primary: true, onClick: go });
  return d.result;
}

/** Ask the password for these nodes' envelopes (and try it on every other locked node of the workflow). */
async function unlockDialog(nodes, { forRun = false } = {}) {
  const wanted = nodes.map(envelopeOf).filter((e) => e && !UNLOCKED.has(e.id));
  if (!wanted.length) return true;
  const d = vaultDialog({ title: forRun ? "Run with locked values" : "Unlock locked values",
    subtitle: forRun ? `${plural(nodes.length, "node")} in this run ${nodes.length === 1 ? "has" : "have"} locked values. ` +
      "Enter the password once for this browser tab." : "This browser tab will run them without asking again. The values stay hidden.",
    width: 460 });
  const pw = passwordField({ label: "Password", autocomplete: "current-password" });
  d.body.append(pw.el);
  d.addAction("Cancel", { onClick: () => d.close(false) });
  const go = async () => {
    if (d.busy) return;
    d.notice(null);
    d.setBusy(true, "Unlocking…");
    const others = [...allNodes(rootGraph())].map(envelopeOf).filter((e) => e && !UNLOCKED.has(e.id));
    const envs = [...new Map([...wanted, ...others].map((e) => [e.id, e])).values()];
    const { opened } = await unlockEnvelopes(envs, pw.value());
    for (const [id, values] of opened) UNLOCKED.set(id, values);
    const still = wanted.filter((e) => !UNLOCKED.has(e.id));
    d.setBusy(false);
    if (still.length === wanted.length) { d.notice("danger", "Wrong password."); pw.input.select?.(); return; }
    refreshCards();
    if (still.length) {
      d.notice("warn", `${plural(still.length, "node")} use${still.length === 1 ? "s" : ""} another password. Enter it too.`);
      pw.clear();
      return;
    }
    d.close(true);
  };
  d.onSubmit(go);
  d.addAction(forRun ? "Unlock and run" : "Unlock", { primary: true, onClick: go });
  return d.result;
}

async function removeLockDialog(nodes) {
  const targets = nodes.filter((n) => envelopeOf(n));
  if (!targets.length) return false;
  const d = vaultDialog({ title: "Remove the value lock",
    subtitle: `The values of ${plural(targets.length, "node")} go back into ${targets.length === 1 ? "its" : "their"} ` +
      "widgets, visible and saved in plain text again.", width: 460, tone: "warn" });
  const pw = passwordField({ label: "Password", autocomplete: "current-password" });
  d.body.append(pw.el);
  d.addAction("Cancel", { onClick: () => d.close(false) });
  const go = async () => {
    if (d.busy) return;
    d.notice(null);
    d.setBusy(true, "Checking…");
    const { opened, failed } = await unlockEnvelopes(targets.map(envelopeOf), pw.value());
    d.setBusy(false);
    if (!opened.size) { d.notice("danger", "Wrong password."); pw.input.select?.(); return; }
    for (const n of targets) {
      const env = envelopeOf(n);
      if (opened.has(env.id)) { UNLOCKED.delete(env.id); applyUnlocked(n, opened.get(env.id)); }
    }
    checkState();
    d.close(true);
    toast(failed.length ? `Removed the lock from ${plural(opened.size, "node")}; ${plural(failed.length, "node")} ` +
      "use another password and stay locked." : `Removed the lock from ${plural(opened.size, "node")}.`,
    failed.length ? "warn" : "success", "Lock removed");
  };
  d.onSubmit(go);
  d.addAction("Remove lock", { primary: true, onClick: go });
  return d.result;
}

// ── the run ─────────────────────────────────────────────────────────────────────────────────────────────────────
/** A copy of the prompt with this tab's real values in it; asks for a password when a locked node is not unlocked. */
async function withRealValues(output) {
  if (!output || ![...Object.keys(output)].some((id) => envelopeOf(nodeByExecId(id)))) return output;
  const copy = structuredClone(output);         // the caller's object keeps placeholders (it may be reused / shown)
  const fill = () => injectLockedValues(copy, { nodeFor: nodeByExecId, valuesFor: (env) => UNLOCKED.get(env.id) ?? null });
  let r = fill();
  if (r.mismatched.length) {                     // checked before any password is asked; never names a value
    const n = nodeByExecId(r.mismatched[0].execId);
    throw new Error(`Run cancelled: locked values sit on a node of another type (${n?.title || r.mismatched[0].nodeType}). ` +
      "Remove that node or its lock.");
  }
  if (r.missing.length) {
    const nodes = [...new Set(r.missing.map((m) => nodeByExecId(m.execId)).filter(Boolean))];
    if (!(await unlockDialog(nodes, { forRun: true }))) throw new Error("Run cancelled: locked values need their password.");
    r = fill();
    if (r.missing.length) throw new Error("Run cancelled: some locked values are still locked.");
  }
  return copy;
}

function installQueueHook() {
  if (api.__c2cLockParams) return;
  const q = api.queuePrompt.bind(api);
  api.queuePrompt = async function (number, data, ...rest) {
    if (data?.output) data = { ...data, output: await withRealValues(data.output) };
    return q(number, data, ...rest);
  };
  api.__c2cLockParams = true;
}

/** Every load / paste / undo / convert-to-subgraph goes through configure: show locked nodes as locked. */
function installConfigureHook() {
  const N = globalThis.LiteGraph?.LGraphNode;
  if (!N || N.prototype.__c2cLockParams) return;
  const orig = N.prototype.configure;
  N.prototype.configure = function (...a) {
    const r = orig.apply(this, a);
    try { if (this.properties?.[LOCK_PROP]) applyLocked(this); } catch (_) { /* never break loading */ }
    return r;
  };
  N.prototype.__c2cLockParams = true;
}

// ── menus ───────────────────────────────────────────────────────────────────────────────────────────────────────
function menuFor(items) {
  const c = collect(items);
  const out = [];
  if (c.lockable.length) out.push({ content: "🔒 Lock parameter values…", callback: () => lockDialog(items) });
  const locked = c.locked;
  if (locked.some((n) => !UNLOCKED.has(envelopeOf(n).id))) {
    out.push({ content: "🔓 Unlock locked values (this tab)…", callback: () => unlockDialog(locked) });
  }
  if (locked.length) out.push({ content: "🔓 Remove value lock…", callback: () => removeLockDialog(locked) });
  return out.length ? [null, ...out] : [];
}

app.registerExtension({
  name: "C2C.LockParams",
  commands: [{ id: CMD, label: "Lock parameter values of the selection…", icon: "pi pi-lock",
    function: () => lockDialog(selectedItems(app.canvas)) }],
  setup() {
    installQueueHook();
    installConfigureHook();
    // already on the canvas when this extension loaded (the startup workflow)
    for (const n of allNodes(rootGraph())) if (envelopeOf(n)) applyLocked(n);
    app.canvas?.canvas?.addEventListener("litegraph:set-graph", (e) => {
      for (const n of e?.detail?.newGraph?.nodes || []) if (envelopeOf(n)) setTimeout(() => applyLocked(n), 0);
    });
  },
  getCanvasMenuItems(canvas) {
    const items = selectedItems(canvas);
    return items.length ? menuFor(items) : [];
  },
  getNodeMenuItems(node) {
    if (!node || node.id == null || Number(node.id) < 0) return [];
    const sel = selectedItems(app.canvas);
    return menuFor(sel.includes(node) ? sel : [node]);
  },
});

// browser tests: lock / unlock without the dialogs (the password still never leaves the page)
globalThis.__c2cLockParams = {
  async lock(nodes, password) {
    const salt = newSalt();
    const key = await deriveKey(password, salt);
    for (const n of nodes) {
      const { lock } = lockableWidgets(n);
      if (!lock.length) continue;
      const values = Object.fromEntries(lock.map((w) => [w.name, w.value]));
      const env = await sealValues({ key, salt, type: n.comfyClass ?? n.type, values });
      n.properties = n.properties || {};
      n.properties[LOCK_PROP] = env;
      UNLOCKED.set(env.id, values);
      applyLocked(n);
    }
    checkState();
  },
  forget() { UNLOCKED.clear(); refreshCards(); },
  unlockedCount: () => UNLOCKED.size,
  LockError,
};
