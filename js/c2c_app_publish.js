// C2C App Gateway — "Publish as C2C App" (A26 L9.206 S3).
//
// Works when this ComfyUI is opened as the owner Studio THROUGH the C2C App Gateway (https://<gateway>/studio/):
// the page then shares the gateway's origin, so the owner's gateway session (cookie + CSRF token from /api/v1/me)
// publishes directly. Opened any other way, the command explains how to reach the Studio and does nothing else.
//
// The dialog turns the current workflow into a release: the ComfyUI API prompt (app.graphToPrompt) + bindings from
// chosen widget inputs to logical ids stay PRIVATE on the gateway; only the manifest (logical ids, types, limits)
// reaches runners. Input types and limits come from the worker's /object_info for each node class.
// License: Apache-2.0
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const GW = { me: null, checked: false };
const MAX_SEED = 2 ** 53 - 1;

// ── small helpers (same look as the Vault dialogs) ──────────────────────────────────────────────────────────
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

const inputStyle = (extra = {}) => ({ boxSizing: "border-box", padding: "4px 6px", borderRadius: "4px",
  background: T.field, color: T.fg, border: `1px solid ${T.border}`, font: "12px/1.4 inherit", ...extra });

function button(text, primary, onClick) {
  return el("button", { textContent: text, type: "button", on: { click: onClick }, style: {
    padding: "5px 12px", borderRadius: "4px", cursor: "pointer", font: "12px inherit",
    border: `1px solid ${primary ? T.accent : T.border}`, background: primary ? T.accent : "transparent",
    color: primary ? "#0b0d10" : T.fg } });
}

function modal(title, build, { width = 760 } = {}) {
  return new Promise((resolve) => {
    const back = el("div", { role: "dialog", "aria-modal": "true", "aria-label": title, "data-c2c-app-publish": "1",
      style: { position: "fixed", inset: "0", zIndex: "10000", background: "rgba(0,0,0,.55)", display: "flex",
        alignItems: "center", justifyContent: "center" } });
    const box = el("div", { style: { background: T.bg, color: T.fg, border: `1px solid ${T.border}`,
      borderRadius: "6px", padding: "14px 16px", width: `${width}px`, maxWidth: "96vw", maxHeight: "92vh",
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

function toast(text, warn) {
  try {
    app.extensionManager?.toast?.add({ severity: warn ? "warn" : "info", summary: "C2C App", detail: text, life: 6000 });
  } catch (_) { console.log("[C2C App]", text); }
}

const section = (text) => el("div", { textContent: text, style: { fontWeight: "600", margin: "12px 0 6px",
  color: T.fg, borderBottom: `1px solid ${T.border}`, paddingBottom: "3px" } });
const hint = (text) => el("div", { textContent: text, style: { color: T.dim, margin: "2px 0 6px" } });

// ── gateway session (same origin only) ──────────────────────────────────────────────────────────────────────
async function gatewayMe() {
  if (GW.checked) return GW.me;
  GW.checked = true;
  if (!location.pathname.startsWith("/studio")) return null;          // not opened through the gateway
  try {
    const r = await fetch("/api/v1/me", { credentials: "same-origin" });
    const me = r.ok ? await r.json() : null;
    GW.me = me && (me.scopes || []).includes("publish") && me.csrf ? me : null;
  } catch (_) { GW.me = null; }
  return GW.me;
}

async function gw(method, path, body) {
  const h = { "X-C2C-CSRF": GW.me?.csrf || "" };
  if (body !== undefined) h["Content-Type"] = "application/json";
  const r = await fetch(path, { method, credentials: "same-origin", headers: h,
    body: body === undefined ? undefined : JSON.stringify(body) });
  let data = {};
  try { data = await r.json(); } catch (_) { /* keep {} */ }
  return { ok: r.ok, status: r.status, data };
}

// ── workflow -> candidate inputs / outputs ──────────────────────────────────────────────────────────────────
const _defs = new Map();
async function nodeDef(cls) {
  if (!_defs.has(cls)) {
    _defs.set(cls, (async () => {
      try {
        const r = await api.fetchApi(`/object_info/${encodeURIComponent(cls)}`);
        return r.ok ? (await r.json())[cls] || null : null;
      } catch (_) { return null; }
    })());
  }
  return _defs.get(cls);
}

const snake = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").replace(/^(\d)/, "n$1")
  .slice(0, 40) || "input";

/** The public type + constraints a widget input can have, from its /object_info spec; null = not publishable. */
function describe(name, spec, value) {
  if (!Array.isArray(spec)) return null;
  const [t, o = {}] = spec;
  if ((Array.isArray(t) || t === "COMBO") && o.image_upload) return { type: "image", max_mb: 20 };
  if (Array.isArray(t)) return { type: "choice", options: t.map(String) };
  if (t === "COMBO") return { type: "choice", options: (o.options || []).map(String) };
  if (t === "INT") {
    if (/seed/i.test(name)) return { type: "seed" };
    const lo = Math.max(Number.isFinite(o.min) ? o.min : -MAX_SEED, -MAX_SEED);
    const hi = Math.min(Number.isFinite(o.max) ? o.max : MAX_SEED, MAX_SEED);
    return { type: "int", min: lo, max: hi, step: o.step || 1 };
  }
  if (t === "FLOAT") {
    return { type: "number", min: Number.isFinite(o.min) ? o.min : -1e9, max: Number.isFinite(o.max) ? o.max : 1e9,
      step: o.step || 0.01 };
  }
  if (t === "STRING") return { type: "text", max_len: o.multiline ? 10000 : 2000 };
  if (t === "BOOLEAN") return { type: "boolean" };
  return null;
}

async function candidates(prompt) {
  const inputs = [], outputs = [];
  for (const [id, node] of Object.entries(prompt)) {
    const def = await nodeDef(node.class_type);
    if (!def) continue;
    const title = node._meta?.title || def.display_name || node.class_type;
    const specs = { ...(def.input?.optional || {}), ...(def.input?.required || {}) };
    for (const [name, value] of Object.entries(node.inputs || {})) {
      if (Array.isArray(value)) continue;                               // a link, not a value
      const d = describe(name, specs[name], value);
      if (d) inputs.push({ node: id, input: name, title, value, ...d });
    }
    if (def.output_node) {
      const video = /video|webm|mp4|gif|animat/i.test(node.class_type);
      const audio = /audio/i.test(node.class_type);
      outputs.push({ node: id, title, cls: node.class_type, type: audio ? "audio" : video ? "video" : "image",
        kind: audio ? "audio" : video ? "gifs" : "images" });
    }
  }
  return { inputs, outputs };
}

// ── dialog ──────────────────────────────────────────────────────────────────────────────────────────────────
async function publishDialog() {
  const me = await gatewayMe();
  if (!me) {
    toast("Publishing needs the C2C App Gateway: open ComfyUI as the owner Studio at https://<your gateway>/studio/ " +
      "(signed in as owner or publisher), then run this again.", true);
    return null;
  }
  const { output: prompt } = await app.graphToPrompt();
  if (!prompt || !Object.keys(prompt).length) { toast("This workflow has no nodes that would run.", true); return null; }
  const { inputs, outputs } = await candidates(prompt);
  if (!outputs.length) { toast("Add an output node (for example Save Image) before publishing.", true); return null; }
  const apps = (await gw("GET", "/api/v1/admin/apps")).data.apps || [];

  return modal("Publish as C2C App", (body, close) => {
    const err = el("div", { "data-role": "error", style: { color: T.err, minHeight: "16px", margin: "8px 0 0",
      whiteSpace: "pre-wrap" } });
    // app identity
    body.append(hint("Runners see only the inputs and outputs you tick here. The workflow itself stays on the gateway."));
    const appSel = el("select", { "data-role": "app", style: inputStyle({ width: "100%" }) },
      [el("option", { value: "", textContent: "New app…" }),
        ...apps.map((a) => el("option", { value: a.app_id, textContent: `${a.name} (${a.app_id}, release ${a.current ?? "–"})` }))]);
    const appId = el("input", { "data-role": "app-id", placeholder: "app id, e.g. portrait-relight",
      style: inputStyle({ width: "100%" }) });
    const name = el("input", { "data-role": "name", placeholder: "Name runners see",
      value: app.extensionManager?.workflow?.activeWorkflow?.filename || "", style: inputStyle({ width: "100%" }) });
    const desc = el("textarea", { "data-role": "description", rows: 2, placeholder: "What the app does (optional)",
      style: inputStyle({ width: "100%", resize: "vertical" }) });
    const syncApp = () => {
      const a = apps.find((x) => x.app_id === appSel.value);
      appId.disabled = !!a;
      if (a) { appId.value = a.app_id; name.value = a.name; }
    };
    appSel.addEventListener("change", syncApp);
    body.append(section("App"), appSel,
      el("div", { style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px", marginTop: "6px" } },
        [appId, name]), el("div", { style: { marginTop: "6px" } }, [desc]));

    // inputs
    body.append(section("Public inputs"),
      hint("Ids and labels are public. Name them for your runners, not after the nodes inside."));
    if (!inputs.length) body.append(hint("No widget in this workflow can be made public."));
    const used = new Set();
    const rows = inputs.map((c) => {
      const tick = el("input", { type: "checkbox", "aria-label": `Make ${c.title} ${c.input} public` });
      let id = snake(c.input === "image" && c.type === "image" ? "source_media" : c.input);
      for (let k = 2; used.has(id); k++) id = `${snake(c.input)}_${k}`;
      used.add(id);
      const idIn = el("input", { value: id, "aria-label": "Logical id", style: inputStyle({ width: "130px" }) });
      const label = el("input", { value: c.input.replace(/_/g, " "), "aria-label": "Label",
        style: inputStyle({ width: "130px" }) });
      const extra = el("span", { style: { display: "inline-flex", gap: "4px", alignItems: "center" } });
      let lo, hi, opts;
      if (c.type === "number" || c.type === "int") {
        lo = el("input", { type: "number", value: c.min, "aria-label": "Minimum", style: inputStyle({ width: "72px" }) });
        hi = el("input", { type: "number", value: c.max, "aria-label": "Maximum", style: inputStyle({ width: "72px" }) });
        extra.append("min", lo, "max", hi);
      } else if (c.type === "choice") {
        opts = el("textarea", { value: c.options.join("\n"), rows: 2, "aria-label": "Allowed options, one per line",
          style: inputStyle({ width: "160px", resize: "vertical" }) });
        extra.append(opts);
      }
      const shown = typeof c.value === "string" && c.value.length > 28 ? c.value.slice(0, 28) + "…" : String(c.value);
      const row = el("div", { "data-node": c.node, "data-input": c.input, style: { display: "flex", gap: "6px",
        alignItems: "center", flexWrap: "wrap", padding: "4px 0", borderBottom: `1px dashed ${T.border}` } }, [
        tick, el("span", { style: { minWidth: "200px", flex: "1 1 200px" } }, [
          el("b", { textContent: c.title }), ` › ${c.input} `,
          el("span", { textContent: `(${c.type}) = ${shown}`, style: { color: T.dim } })]),
        idIn, label, extra]);
      return { c, tick, idIn, label, lo, hi, opts, row };
    });
    body.append(...rows.map((r) => r.row));

    // outputs
    body.append(section("Public outputs"));
    const orows = outputs.map((o, i) => {
      const tick = el("input", { type: "checkbox", checked: i === 0, "aria-label": `Make ${o.title} public` });
      const idIn = el("input", { value: i ? `result_${i + 1}` : "result", "aria-label": "Output id",
        style: inputStyle({ width: "130px" }) });
      const row = el("div", { "data-node": o.node, style: { display: "flex", gap: "6px", alignItems: "center",
        padding: "4px 0" } }, [tick, el("span", { style: { flex: "1" } }, [el("b", { textContent: o.title }),
        el("span", { textContent: ` ${o.cls} (${o.type})`, style: { color: T.dim } })]), idIn]);
      return { o, tick, idIn, row };
    });
    body.append(...orows.map((r) => r.row));

    // limits + sharing
    const num = (v, label, w = "80px") => el("input", { type: "number", value: v, min: 1, "aria-label": label,
      style: inputStyle({ width: w }) });
    const runtime = num(600, "Longest run, seconds"), conc = num(1, "Runs at once per person"),
      perHour = num(30, "Runs per hour per person");
    const cancel = el("input", { type: "checkbox", checked: true, "aria-label": "Runners may cancel" });
    const share = el("input", { "data-role": "share", placeholder: "runner names, comma separated (optional)",
      style: inputStyle({ width: "100%" }) });
    body.append(section("Limits and access"),
      el("div", { style: { display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "center" } }, [
        "Longest run (s)", runtime, "At once", conc, "Per hour", perHour,
        el("label", { style: { display: "inline-flex", gap: "4px", alignItems: "center" } }, [cancel, "Runners may cancel"])]),
      el("div", { style: { marginTop: "6px" } }, [share]));

    const go = button("Publish", true, async () => {
      err.textContent = "";
      const chosen = rows.filter((r) => r.tick.checked);
      const outs = orows.filter((r) => r.tick.checked);
      const id = (appSel.value || appId.value).trim();
      if (!id) { err.textContent = "Give the app an id (lower-case letters, digits and '-')."; return; }
      if (!outs.length) { err.textContent = "Tick at least one public output."; return; }
      const manifest = { name: name.value.trim() || id, description: desc.value.trim(), inputs: [], outputs: [],
        limits: { max_runtime_s: +runtime.value, max_concurrent_per_user: +conc.value, runs_per_hour: +perHour.value },
        cancel: cancel.checked ? "allowed" : "never" };
      const bindings = {}, outBind = {};
      for (const r of chosen) {
        const lid = r.idIn.value.trim();
        const e = { id: lid, type: r.c.type, label: r.label.value.trim() || lid };
        const v = r.c.value;
        if (r.c.type === "image") Object.assign(e, { required: true, max_mb: r.c.max_mb });
        if (r.c.type === "number" || r.c.type === "int") {
          Object.assign(e, { min: +r.lo.value, max: +r.hi.value });
          if (r.c.type === "number") e.step = r.c.step;
          if (typeof v === "number" && v >= e.min && v <= e.max) e.default = v;
        }
        if (r.c.type === "choice") {
          e.options = r.opts.value.split("\n").map((s) => s.trim()).filter(Boolean);
          if (e.options.includes(String(v))) e.default = String(v);
        }
        if (r.c.type === "text") { e.max_len = r.c.max_len; if (typeof v === "string" && v.length <= e.max_len) e.default = v; }
        if (r.c.type === "boolean" && typeof v === "boolean") e.default = v;
        if (r.c.type === "seed" && Number.isInteger(v) && v >= 0 && v <= MAX_SEED) e.default = v;
        if (bindings[lid]) {                                             // same id twice = one value, several inputs
          bindings[lid].push({ node: r.c.node, input: r.c.input });
          continue;
        }
        manifest.inputs.push(e);
        bindings[lid] = [{ node: r.c.node, input: r.c.input }];
      }
      for (const r of outs) {
        const lid = r.idIn.value.trim();
        manifest.outputs.push({ id: lid, type: r.o.type });
        outBind[lid] = { node: r.o.node, kind: r.o.kind };
      }
      go.disabled = true;
      try {
        if (!appSel.value) {
          const c = await gw("POST", "/api/v1/admin/apps", { app_id: id, name: manifest.name });
          if (!c.ok) { err.textContent = c.data.message || `Could not create the app (${c.status}).`; return; }
        }
        const p = await gw("POST", `/api/v1/admin/apps/${encodeURIComponent(id)}/releases`,
          { manifest, private: { prompt, bindings, outputs: outBind } });
        if (!p.ok) { err.textContent = p.data.message || `Publishing failed (${p.status}).`; return; }
        const people = share.value.split(",").map((s) => s.trim()).filter(Boolean);
        if (people.length) {
          const a = await gw("POST", `/api/v1/admin/apps/${encodeURIComponent(id)}/acl`, { add: people });
          if (!a.ok) { err.textContent = `Published release ${p.data.release}, but sharing failed: ${a.data.message || a.status}`; return; }
        }
        toast(`Published ${manifest.name} release ${p.data.release}. Runners open it at /app/${id}.`);
        close({ app_id: id, release: p.data.release });
      } finally { go.disabled = false; }
    });
    body.append(err, el("div", { style: { display: "flex", gap: "8px", justifyContent: "flex-end", marginTop: "10px" } },
      [button("Cancel", false, () => close(null)), go]));
  });
}

app.registerExtension({
  name: "C2C.AppPublish",
  commands: [{ id: "C2C.App.Publish", label: "C2C: Publish this workflow as an App…", icon: "pi pi-send",
    function: () => publishDialog().catch((e) => toast(`Publishing failed: ${e?.message || e}`, true)) }],
  async setup() { await gatewayMe(); },
  getCanvasMenuItems() {
    if (!GW.me) return [];
    return [null, { content: "📤 Publish as C2C App…", callback: () => publishDialog().catch((e) =>
      toast(`Publishing failed: ${e?.message || e}`, true)) }];
  },
});

// browser tests (A26 L9.206 S3)
globalThis.__c2cAppPublish = { open: publishDialog, candidates, describe, gatewayMe };
