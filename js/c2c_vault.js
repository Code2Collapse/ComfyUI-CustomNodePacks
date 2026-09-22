import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * C2C Vault — password-locked subgraph.
 *
 * Password is typed into a transient modal and POSTed to /c2c_vault/unlock or
 * /open. It is never written to a widget and never reaches the queued prompt.
 *
 * Edit view uses a floating overlay — NEVER convertToSubgraph. ComfyUI's
 * native subgraph API exists, but a native subgraph serialises its inner nodes
 * into workflow JSON on save, which would write the decrypted graph to disk.
 */

const NODES = ["C2C_VaultLocked", "C2C_VaultSealed"];
const MAX_VAULT_INPUTS = 10;
const MAX_VAULT_OUTPUTS = 8;
const MAX_VAULT_PARAMS = 8;
const SLOT_H = 20;
const PROMOTED_VALUE_TYPES = new Set(["INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"]);

/** @type {Map<string, { subgraph: object, panel: HTMLElement, dispose: () => void }>} */
const vaultEditSessions = new Map();

/** @type {Set<string>} vault_ids with an active unlock session (locked mode). */
const unlockedSessions = new Set();

function css(el, s) { Object.assign(el.style, s); }

function headless() {
  return typeof document === "undefined" || typeof window === "undefined";
}

function generateStrongPassword(len = 20) {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%&*";
  const buf = new Uint8Array(len);
  crypto.getRandomValues(buf);
  let out = "";
  for (let i = 0; i < len; i++) out += chars[buf[i] % chars.length];
  return out;
}

function parseInterface(raw) {
  try {
    const v = typeof raw === "string" ? JSON.parse(raw || "{}") : (raw || {});
    const params = [];
    if (Array.isArray(v.params)) {
      for (const p of v.params) {
        if (!p || typeof p !== "object") continue;
        const id = String(p.id || "").trim();
        if (!id) continue;
        const entry = {
          id,
          name: String(p.name || id),
          type: String(p.type || "STRING").toUpperCase(),
          value: p.value,
        };
        if (p.socket !== undefined && p.socket !== null) {
          const sock = Number(p.socket);
          if (Number.isFinite(sock)) entry.socket = sock;
        }
        params.push(entry);
      }
    }
    return {
      mode: v.mode || "locked",
      node_count: Number(v.node_count) || 0,
      in: Array.isArray(v.in) ? v.in : [],
      out: Array.isArray(v.out) ? v.out : [],
      params,
    };
  } catch (_) {
    return { mode: "locked", node_count: 0, in: [], out: [], params: [] };
  }
}

function widget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

async function post(route, body) {
  const r = await api.fetchApi(route, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = {};
  try { data = await r.json(); } catch (_) { /* keep {} */ }
  return { ok: r.ok && data.ok, data, status: r.status };
}

function shakeEl(el) {
  if (!el) return;
  el.animate([
    { transform: "translateX(0)" },
    { transform: "translateX(-6px)" },
    { transform: "translateX(6px)" },
    { transform: "translateX(-4px)" },
    { transform: "translateX(4px)" },
    { transform: "translateX(0)" },
  ], { duration: 380, easing: "ease-in-out" });
}

function modalPasswordStep({ title, note, sealed }) {
  if (headless()) return Promise.resolve(null);
  return new Promise((resolve) => {
    const back = document.createElement("div");
    css(back, {
      position: "fixed", inset: "0", zIndex: "10000",
      background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center",
    });
    const box = document.createElement("div");
    css(box, {
      background: "var(--comfy-menu-bg, #353535)",
      color: "var(--fg-color, #ddd)",
      border: "1px solid var(--border-color, #4a4a4a)",
      borderRadius: "6px", padding: "18px 20px",
      minWidth: "360px", maxWidth: "480px", font: "13px sans-serif",
      boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    });

    const h = document.createElement("div");
    h.textContent = title;
    css(h, { fontWeight: "600", marginBottom: "10px" });

    const p = document.createElement("div");
    p.textContent = note || "";
    css(p, { opacity: "0.75", marginBottom: "12px", lineHeight: "1.45", fontSize: "12px" });

    const pw = document.createElement("input");
    pw.type = "password";
    pw.placeholder = "Vault password";
    pw.autocomplete = "new-password";
    css(pw, {
      width: "100%", boxSizing: "border-box", marginBottom: "8px",
      padding: "7px 9px", borderRadius: "4px",
      background: "var(--comfy-input-bg, #222)",
      color: "var(--input-text, #ccc)",
      border: "1px solid var(--border-color, #4a4a4a)",
    });

    const pw2 = document.createElement("input");
    pw2.type = "password";
    pw2.placeholder = "Confirm password";
    pw2.autocomplete = "new-password";
    css(pw2, {
      width: "100%", boxSizing: "border-box", marginBottom: "8px",
      padding: "7px 9px", borderRadius: "4px",
      background: "var(--comfy-input-bg, #222)",
      color: "var(--input-text, #ccc)",
      border: "1px solid var(--border-color, #4a4a4a)",
    });

    const genRow = document.createElement("div");
    css(genRow, { display: "flex", gap: "8px", marginBottom: "8px", alignItems: "center" });
    const genBtn = document.createElement("button");
    genBtn.textContent = "Generate strong password";
    const genShow = document.createElement("span");
    genShow.textContent = "";
    css(genShow, { fontFamily: "ui-monospace, monospace", fontSize: "11px", color: "var(--c2c-green, #a6e3a1)", flex: "1", wordBreak: "break-all" });
    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy";
    copyBtn.style.display = "none";
    for (const b of [genBtn, copyBtn]) {
      css(b, {
        padding: "4px 10px", borderRadius: "4px", cursor: "pointer", fontSize: "11px",
        background: "var(--comfy-input-bg, #222)",
        color: "var(--input-text, #ccc)",
        border: "1px solid var(--border-color, #4a4a4a)",
      });
    }
    let shownOnce = "";
    genBtn.onclick = () => {
      shownOnce = generateStrongPassword(20);
      pw.value = shownOnce;
      pw2.value = shownOnce;
      genShow.textContent = shownOnce;
      copyBtn.style.display = "";
    };
    copyBtn.onclick = async () => {
      try { await navigator.clipboard.writeText(shownOnce); copyBtn.textContent = "Copied"; } catch (_) {}
    };
    genRow.append(genBtn, copyBtn, genShow);

    const err = document.createElement("div");
    css(err, { color: "#f87171", minHeight: "16px", marginBottom: "8px" });

    const row = document.createElement("div");
    css(row, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const next = document.createElement("button");
    next.textContent = "Next →";
    for (const b of [cancel, next]) {
      css(b, {
        padding: "6px 14px", borderRadius: "4px", cursor: "pointer",
        background: "var(--comfy-input-bg, #222)",
        color: "var(--input-text, #ccc)",
        border: "1px solid var(--border-color, #4a4a4a)",
      });
    }

    const done = (v) => { back.remove(); resolve(v); };
    cancel.onclick = () => done(null);
    next.onclick = () => {
      if (!pw.value) { err.textContent = "Enter a password."; shakeEl(box); return; }
      if (pw.value !== pw2.value) { err.textContent = "Passwords do not match."; shakeEl(box); return; }
      const v = pw.value;
      pw.value = ""; pw2.value = "";
      done(v);
    };
    back.onkeydown = (e) => {
      if (e.key === "Escape") done(null);
      if (e.key === "Enter") next.click();
    };

    row.append(cancel, next);
    box.append(h, p, pw, pw2, genRow, err, row);
    back.append(box);
    document.body.append(back);
    pw.focus();
  });
}

function modalBoundaryStep({ boundary, nodeCount, sealed }) {
  if (headless()) return Promise.resolve(null);
  return new Promise((resolve) => {
    const back = document.createElement("div");
    css(back, {
      position: "fixed", inset: "0", zIndex: "10000",
      background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center",
    });
    const box = document.createElement("div");
    css(box, {
      background: "var(--comfy-menu-bg, #353535)",
      color: "var(--fg-color, #ddd)",
      border: "1px solid var(--border-color, #4a4a4a)",
      borderRadius: "6px", padding: "18px 20px",
      minWidth: "420px", maxWidth: "560px", maxHeight: "80vh", overflow: "auto",
      font: "13px sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    });

    const h = document.createElement("div");
    h.textContent = `Boundary preview — ${nodeCount} node(s)`;
    css(h, { fontWeight: "600", marginBottom: "10px" });

    const note = document.createElement("div");
    note.textContent = sealed
      ? "These are the wires that cross the vault edge. Rename them now — recipients see these socket names, not the encrypted internals."
      : "Rename boundary sockets before locking. Wrong names are painful to fix without reopening the vault.";
    css(note, { opacity: "0.75", marginBottom: "12px", lineHeight: "1.45", fontSize: "12px" });

    const mkTable = (title, rows, key) => {
      const sec = document.createElement("div");
      css(sec, { marginBottom: "12px" });
      const th = document.createElement("div");
      th.textContent = title;
      css(th, { fontWeight: "600", marginBottom: "6px", fontSize: "12px" });
      sec.append(th);
      rows.forEach((row, i) => {
        const line = document.createElement("div");
        css(line, { display: "flex", gap: "8px", marginBottom: "4px", alignItems: "center" });
        const typeEl = document.createElement("span");
        typeEl.textContent = row.type || "*";
        css(typeEl, { width: "72px", fontFamily: "ui-monospace, monospace", fontSize: "11px", opacity: "0.7" });
        const nameIn = document.createElement("input");
        nameIn.value = row.name;
        css(nameIn, {
          flex: "1", padding: "4px 8px", borderRadius: "4px",
          background: "var(--comfy-input-bg, #222)", color: "var(--input-text, #ccc)",
          border: "1px solid var(--border-color, #4a4a4a)",
        });
        nameIn.oninput = () => { row.name = nameIn.value; };
        line.append(typeEl, nameIn);
        sec.append(line);
      });
      return sec;
    };

    box.append(h, note);
    if (boundary.boundary_in.length) box.append(mkTable("Inputs (wires entering)", boundary.boundary_in, "in"));
    else {
      const none = document.createElement("div");
      none.textContent = "No external inputs.";
      css(none, { opacity: "0.6", marginBottom: "8px", fontSize: "12px" });
      box.append(none);
    }
    if (boundary.boundary_out.length) box.append(mkTable("Outputs (wires leaving)", boundary.boundary_out, "out"));
    else {
      const none = document.createElement("div");
      none.textContent = "No outputs — vault cannot run.";
      css(none, { color: "#f87171", marginBottom: "8px", fontSize: "12px" });
      box.append(none);
    }

    const row = document.createElement("div");
    css(row, { display: "flex", gap: "8px", justifyContent: "flex-end", marginTop: "12px" });
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const ok = document.createElement("button");
    ok.textContent = sealed ? "Seal" : "Lock";
    for (const b of [cancel, ok]) {
      css(b, {
        padding: "6px 14px", borderRadius: "4px", cursor: "pointer",
        background: "var(--comfy-input-bg, #222)",
        color: "var(--input-text, #ccc)",
        border: "1px solid var(--border-color, #4a4a4a)",
      });
    }
    const done = (v) => { back.remove(); resolve(v); };
    cancel.onclick = () => done(null);
    ok.onclick = () => {
      if (!boundary.boundary_out.length) { shakeEl(box); return; }
      done(boundary);
    };
    row.append(cancel, ok);
    box.append(row);
    back.append(box);
  });
}

function inferPromotedWidgetType(w) {
  const t = String(w?.type || "").toUpperCase();
  if (PROMOTED_VALUE_TYPES.has(t)) return t;
  const lt = String(w?.type || "").toLowerCase();
  const opt = w?.options || {};
  if (lt === "combo") return "COMBO";
  if (lt === "toggle" || lt === "bool" || lt === "boolean") return "BOOLEAN";
  if (lt === "number") {
    if (opt.round === true || opt.precision === 0 || opt.step === 1) return "INT";
    return "FLOAT";
  }
  if (lt === "text" || lt === "string") return "STRING";
  return t || "STRING";
}

function isPromotableWidget(w) {
  if (!w || !w.name) return false;
  if (w.type === "button") return false;
  if (w.name.startsWith("vault_") || w.name.startsWith("__vault_p_")) return false;
  const sz = w.computeSize?.();
  if (sz && sz[1] <= 0) return false;
  return true;
}

function modalPromoteStep({ selection }) {
  if (headless()) return { promoted: [], params: [] };

  const groups = [];
  for (const n of selection) {
    const widgets = (n.widgets || []).filter(isPromotableWidget);
    if (!widgets.length) continue;
    groups.push({
      nodeId: String(n.id),
      title: `${n.comfyClass || n.type} (id ${n.id})`,
      rows: widgets.map((w) => {
        const type = inferPromotedWidgetType(w);
        const widgetOk = PROMOTED_VALUE_TYPES.has(type);
        return {
          widgetName: w.name,
          type,
          value: w.value,
          comboOptions: w.type === "combo" ? (w.options?.values || [w.value]) : null,
          checked: false,
          label: w.name,
          mode: widgetOk ? "widget" : "socket",
          widgetOk,
        };
      }),
    });
  }

  return new Promise((resolve) => {
    const back = document.createElement("div");
    css(back, {
      position: "fixed", inset: "0", zIndex: "10000",
      background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center",
    });
    const box = document.createElement("div");
    css(box, {
      background: "var(--comfy-menu-bg, #353535)",
      color: "var(--fg-color, #ddd)",
      border: "1px solid var(--border-color, #4a4a4a)",
      borderRadius: "6px", padding: "18px 20px",
      minWidth: "460px", maxWidth: "620px", maxHeight: "80vh", overflow: "auto",
      font: "13px sans-serif", boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    });

    const h = document.createElement("div");
    h.textContent = "Promote parameters";
    css(h, { fontWeight: "600", marginBottom: "10px" });

    const note = document.createElement("div");
    note.textContent =
      "Promoted parameters stay editable from outside the vault while the contents stay sealed — the label you choose is public; which internal widget it drives is not.";
    css(note, { opacity: "0.75", marginBottom: "12px", lineHeight: "1.45", fontSize: "12px" });

    box.append(h, note);

    if (!groups.length) {
      const none = document.createElement("div");
      none.textContent = "No promotable widgets in this selection.";
      css(none, { opacity: "0.6", marginBottom: "8px", fontSize: "12px" });
      box.append(none);
    }

    for (const group of groups) {
      const sec = document.createElement("div");
      css(sec, { marginBottom: "12px" });
      const gh = document.createElement("div");
      gh.textContent = group.title;
      css(gh, { fontWeight: "600", marginBottom: "6px", fontSize: "12px" });
      sec.append(gh);

      for (const row of group.rows) {
        const line = document.createElement("div");
        css(line, {
          display: "grid", gridTemplateColumns: "auto 1fr 1fr auto",
          gap: "8px", marginBottom: "6px", alignItems: "center",
          opacity: row.checked ? "1" : "0.55",
        });

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.title = `Promote ${row.widgetName} from inside the vault`;
        cb.onchange = () => {
          row.checked = cb.checked;
          line.style.opacity = row.checked ? "1" : "0.55";
        };

        const wname = document.createElement("span");
        wname.textContent = row.widgetName;
        css(wname, { fontFamily: "ui-monospace, monospace", fontSize: "11px", opacity: "0.8" });

        const labelIn = document.createElement("input");
        labelIn.value = row.label;
        labelIn.placeholder = "Public label";
        labelIn.title = "Name shown on the vault node — not the internal widget name";
        css(labelIn, {
          padding: "4px 8px", borderRadius: "4px",
          background: "var(--comfy-input-bg, #222)", color: "var(--input-text, #ccc)",
          border: "1px solid var(--border-color, #4a4a4a)",
        });
        labelIn.oninput = () => { row.label = labelIn.value; };

        const modeSel = document.createElement("select");
        css(modeSel, {
          padding: "4px 6px", borderRadius: "4px",
          background: "var(--comfy-input-bg, #222)", color: "var(--input-text, #ccc)",
          border: "1px solid var(--border-color, #4a4a4a)",
        });
        const optWidget = document.createElement("option");
        optWidget.value = "widget";
        optWidget.textContent = "Widget on vault";
        const optSocket = document.createElement("option");
        optSocket.value = "socket";
        optSocket.textContent = "Input socket";
        modeSel.append(optWidget, optSocket);
        modeSel.value = row.mode;
        if (!row.widgetOk) {
          optWidget.disabled = true;
          optWidget.title = `${row.type} has no vault widget — wire it via param_N instead`;
          modeSel.value = "socket";
          row.mode = "socket";
        }
        modeSel.title = row.widgetOk
          ? "Expose as an editable widget, or as a param_N socket"
          : `${row.type} cannot be typed on the vault — only a param_N wire`;
        modeSel.onchange = () => { row.mode = modeSel.value; };

        line.append(cb, wname, labelIn, modeSel);
        sec.append(line);
      }
      box.append(sec);
    }

    const row = document.createElement("div");
    css(row, { display: "flex", gap: "8px", justifyContent: "flex-end", marginTop: "12px" });
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const skip = document.createElement("button");
    skip.textContent = "Skip";
    const ok = document.createElement("button");
    ok.textContent = "Continue";
    for (const b of [cancel, skip, ok]) {
      css(b, {
        padding: "6px 14px", borderRadius: "4px", cursor: "pointer",
        background: "var(--comfy-input-bg, #222)",
        color: "var(--input-text, #ccc)",
        border: "1px solid var(--border-color, #4a4a4a)",
      });
    }

    const finish = (v) => { back.remove(); resolve(v); };
    cancel.onclick = () => finish(null);
    skip.onclick = () => finish({ promoted: [], params: [] });
    ok.onclick = () => {
      const promoted = [];
      const params = [];
      const socketRows = [];
      let nextId = 0;

      for (const group of groups) {
        for (const row of group.rows) {
          if (!row.checked) continue;
          const id = `p${nextId++}`;
          promoted.push({ id, node: group.nodeId, widget: row.widgetName });
          const entry = {
            id,
            name: (row.label || row.widgetName).trim() || row.widgetName,
            type: row.type,
            value: row.value,
            _comboOptions: row.comboOptions,
          };
          if (row.mode === "socket") socketRows.push(entry);
          else params.push(entry);
        }
      }

      const used = new Set(params.filter((p) => p.socket !== undefined).map((p) => p.socket));
      for (const entry of socketRows) {
        let k = 0;
        while (used.has(k) && k < MAX_VAULT_PARAMS) k++;
        if (k >= MAX_VAULT_PARAMS) {
          shakeEl(box);
          return;
        }
        entry.socket = k;
        used.add(k);
        params.push(entry);
      }

      finish({ promoted, params });
    };

    row.append(cancel, skip, ok);
    box.append(row);
    back.append(box);
    document.body.append(back);
  });
}

function modalAlert(title, note) {
  if (headless()) return Promise.resolve();
  return new Promise((resolve) => {
    const back = document.createElement("div");
    css(back, {
      position: "fixed", inset: "0", zIndex: "10000",
      background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center",
    });
    const box = document.createElement("div");
    css(box, {
      background: "var(--comfy-menu-bg, #353535)", color: "var(--fg-color, #ddd)",
      border: "1px solid var(--border-color, #4a4a4a)", borderRadius: "6px",
      padding: "18px 20px", minWidth: "300px", font: "13px sans-serif",
    });
    const ok = document.createElement("button");
    ok.textContent = "OK";
    css(ok, { marginTop: "12px", padding: "6px 14px", cursor: "pointer" });
    ok.onclick = () => { back.remove(); resolve(); };
    box.append(
      Object.assign(document.createElement("div"), { textContent: title, style: { fontWeight: "600", marginBottom: "8px" } }),
      Object.assign(document.createElement("div"), { textContent: note, style: { opacity: "0.8", lineHeight: "1.45" } }),
      ok,
    );
    back.append(box);
    document.body.append(back);
  });
}

function modalPassword({ title, note, confirmLabel }) {
  if (headless()) return Promise.resolve(null);
  return new Promise((resolve) => {
    const back = document.createElement("div");
    css(back, {
      position: "fixed", inset: "0", zIndex: "10000",
      background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center",
    });
    const box = document.createElement("div");
    css(box, {
      background: "var(--comfy-menu-bg, #353535)", color: "var(--fg-color, #ddd)",
      border: "1px solid var(--border-color, #4a4a4a)", borderRadius: "6px",
      padding: "18px 20px", minWidth: "340px", font: "13px sans-serif",
    });
    const pw = document.createElement("input");
    pw.type = "password";
    pw.autocomplete = "current-password";
    css(pw, {
      width: "100%", boxSizing: "border-box", margin: "10px 0",
      padding: "7px 9px", borderRadius: "4px",
      background: "var(--comfy-input-bg, #222)", color: "var(--input-text, #ccc)",
      border: "1px solid var(--border-color, #4a4a4a)",
    });
    const err = document.createElement("div");
    css(err, { color: "#f87171", minHeight: "16px" });
    const row = document.createElement("div");
    css(row, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const ok = document.createElement("button");
    ok.textContent = confirmLabel || "OK";
    for (const b of [cancel, ok]) css(b, { padding: "6px 14px", cursor: "pointer" });
    const done = (v) => { back.remove(); resolve(v); };
    cancel.onclick = () => done(null);
    ok.onclick = () => {
      if (!pw.value) { err.textContent = "Enter a password."; shakeEl(box); return; }
      const v = pw.value; pw.value = ""; done(v);
    };
    box.append(
      Object.assign(document.createElement("div"), { textContent: title, style: { fontWeight: "600" } }),
      Object.assign(document.createElement("div"), { textContent: note, style: { opacity: "0.75", fontSize: "12px", marginTop: "6px" } }),
      pw, err, row,
    );
    row.append(cancel, ok);
    back.append(box);
    document.body.append(back);
    pw.focus();
  });
}

/** Classify selection links — mirrors nodes/vault_boundary.py */
function deriveBoundary(sel, graph) {
  const idOf = (n) => String(n.id);
  const ids = new Set(sel.map(idOf));
  const links = [];
  const boundary_in = [];
  const externalSources = [];
  const inTypes = [];

  for (const n of sel) {
    (n.inputs || []).forEach((inp, slot) => {
      const lk = inp.link != null ? graph.links[inp.link] : null;
      if (!lk) return;
      if (ids.has(String(lk.origin_id))) {
        links.push({
          from: String(lk.origin_id), from_slot: lk.origin_slot,
          to: idOf(n), to_slot: slot,
        });
      } else {
        const origin = graph.getNodeById?.(lk.origin_id);
        const outType = origin?.outputs?.[lk.origin_slot]?.type || "*";
        boundary_in.push({
          name: inp.name || `in_${boundary_in.length}`,
          to: idOf(n), to_slot: slot, type: outType,
        });
        externalSources.push({ id: lk.origin_id, slot: lk.origin_slot });
        inTypes.push(outType);
      }
    });
  }

  const boundary_out = [];
  const externalTargets = [];
  const outTypes = [];

  for (const n of sel) {
    (n.outputs || []).forEach((out, slot) => {
      (out.links || []).forEach((lid) => {
        const lk = graph.links[lid];
        if (!lk || ids.has(String(lk.target_id))) return;
        boundary_out.push({
          name: out.name || `out_${boundary_out.length}`,
          from: idOf(n), from_slot: slot, type: out.type || "*",
        });
        externalTargets.push({ id: lk.target_id, slot: lk.target_slot });
        outTypes.push(out.type || "*");
      });
    });
  }

  if (boundary_out.length === 0) {
    const consumed = new Set(links.map((l) => `${l.from}:${l.from_slot}`));
    for (const n of sel) {
      (n.outputs || []).forEach((out, slot) => {
        if (consumed.has(`${idOf(n)}:${slot}`)) return;
        boundary_out.push({
          name: out.name || `out_${boundary_out.length}`,
          from: idOf(n), from_slot: slot, type: out.type || "*",
        });
        externalTargets.push(null);
        outTypes.push(out.type || "*");
      });
    }
  }

  return {
    links, boundary_in, boundary_out,
    externalSources, externalTargets, inTypes, outTypes,
  };
}

function hideSlot(slot, hidden) {
  if (!slot) return;
  if (hidden) {
    if (!slot._vaultStash) slot._vaultStash = { name: slot.name, type: slot.type };
    slot._vaultHidden = true;
    slot.name = "";
  } else {
    slot._vaultHidden = false;
    if (slot._vaultStash) {
      slot.name = slot._vaultStash.name;
      slot.type = slot._vaultStash.type;
    }
  }
}

function inputSlotByName(node, name) {
  return (node.inputs || []).find((inp) => inp.name === name);
}

function readNodeInterface(node) {
  return parseInterface(widget(node, "vault_interface")?.value);
}

function writeNodeInterface(node, iface) {
  const w = widget(node, "vault_interface");
  if (w) w.value = JSON.stringify(iface);
}

function lowestFreeParamSocket(params) {
  const used = new Set(
    (params || []).filter((p) => p.socket !== undefined && p.socket !== null).map((p) => Number(p.socket)),
  );
  for (let k = 0; k < MAX_VAULT_PARAMS; k++) {
    if (!used.has(k)) return k;
  }
  return null;
}

function manifestTypeToSlotType(t) {
  switch (String(t || "").toUpperCase()) {
    case "INT": return "INT";
    case "FLOAT": return "FLOAT";
    case "STRING": return "STRING";
    case "BOOLEAN": return "BOOLEAN";
    default: return "*";
  }
}

function updatePromotedParamValue(node, paramId, value) {
  const iface = readNodeInterface(node);
  const p = (iface.params || []).find((x) => x.id === paramId);
  if (!p || p.socket !== undefined) return;
  p.value = value;
  writeNodeInterface(node, iface);
}

function removePromotedWidget(node, w) {
  if (!w) return;
  if (typeof node.removeWidget === "function") node.removeWidget(w);
  else {
    const idx = node.widgets?.indexOf(w);
    if (idx >= 0) node.widgets.splice(idx, 1);
  }
}

function installPromotedWidgetMenu(w, node, isSealed) {
  const origMouse = w.mouse;
  w.mouse = function (event, pos, nodeRef) {
    if (event?.button === 2 || event?.which === 3) {
      const menu = new LiteGraph.ContextMenu([
        {
          content: "Convert widget to input",
          callback: () => convertPromotedParamToInput(nodeRef || node, w._vaultParamId, isSealed),
        },
      ], { event, parentMenu: null, node: nodeRef || node });
      return true;
    }
    return origMouse?.call(this, event, pos, nodeRef);
  };
}

function createPromotedWidget(node, param, comboOptions, isSealed) {
  const label = param.name || param.id;
  const onChange = (v) => updatePromotedParamValue(node, param.id, v);
  let w;
  switch (param.type) {
    case "INT":
      w = node.addWidget("number", label, param.value ?? 0, onChange, { round: true, step: 10, precision: 0 });
      break;
    case "FLOAT":
      w = node.addWidget("number", label, param.value ?? 0, onChange, { step: 0.01 });
      break;
    case "BOOLEAN":
      w = node.addWidget("toggle", label, !!param.value, onChange);
      break;
    case "COMBO": {
      const values = comboOptions?.length ? comboOptions : [String(param.value ?? "")];
      w = node.addWidget("combo", label, param.value ?? values[0], onChange, { values });
      break;
    }
    default:
      w = node.addWidget("text", label, String(param.value ?? ""), onChange);
      break;
  }
  w._vaultParamId = param.id;
  w._vaultComboOptions = comboOptions || null;
  w.serializeValue = () => undefined;
  installPromotedWidgetMenu(w, node, isSealed);
  return w;
}

function reconcilePromotedWidgets(node, params, isSealed) {
  const active = new Map();
  for (const p of params || []) {
    if (p.socket === undefined && PROMOTED_VALUE_TYPES.has(p.type)) active.set(p.id, p);
  }
  for (const w of [...(node.widgets || [])]) {
    if (!w._vaultParamId) continue;
    if (!active.has(w._vaultParamId)) removePromotedWidget(node, w);
  }
  for (const p of active.values()) {
    let w = (node.widgets || []).find((x) => x._vaultParamId === p.id);
    const comboOptions = node._vaultPromotedCombo?.[p.id] || w?._vaultComboOptions || null;
    if (!w) {
      w = createPromotedWidget(node, p, comboOptions, isSealed);
    } else {
      const label = p.name || p.id;
      if (w.name !== label) w.name = label;
      if (w.value !== p.value) w.value = p.value;
      if (!w._vaultMenuInstalled) {
        installPromotedWidgetMenu(w, node, isSealed);
        w._vaultMenuInstalled = true;
      }
    }
  }
}

function convertPromotedParamToInput(node, paramId, isSealed) {
  const iface = readNodeInterface(node);
  const p = (iface.params || []).find((x) => x.id === paramId);
  if (!p || p.socket !== undefined) return;
  const k = lowestFreeParamSocket(iface.params);
  if (k === null) {
    modalAlert("No sockets free", `All param_0..param_${MAX_VAULT_PARAMS - 1} sockets are in use.`);
    return;
  }
  p.socket = k;
  writeNodeInterface(node, iface);
  applyVaultInterface(node, iface, isSealed);
  node.setDirtyCanvas?.(true, true);
}

function convertPromotedParamToWidget(node, paramId, isSealed) {
  const iface = readNodeInterface(node);
  const p = (iface.params || []).find((x) => x.id === paramId);
  if (!p || p.socket === undefined) return;
  if (!PROMOTED_VALUE_TYPES.has(p.type)) {
    modalAlert("Cannot convert", `${p.type} parameters can only be driven by a wire.`);
    return;
  }
  delete p.socket;
  writeNodeInterface(node, iface);
  applyVaultInterface(node, iface, isSealed);
  node.setDirtyCanvas?.(true, true);
}

function applyVaultInterface(node, iface, isSealed) {
  if (!node || !iface) return;
  const mIn = iface.in?.length || 0;
  const mOut = iface.out?.length || 0;
  const params = iface.params || [];

  for (let i = 0; i < MAX_VAULT_INPUTS; i++) {
    const slot = inputSlotByName(node, `input_${i}`);
    if (!slot) continue;
    if (i < mIn) {
      hideSlot(slot, false);
      slot.name = iface.in[i].name || `in_${i}`;
      slot.type = iface.in[i].type || "*";
    } else {
      hideSlot(slot, true);
    }
  }

  for (let i = 0; i < MAX_VAULT_OUTPUTS; i++) {
    const slot = node.outputs?.[i];
    if (!slot) continue;
    if (i < mOut) {
      hideSlot(slot, false);
      slot.name = iface.out[i].name || `out_${i}`;
      slot.type = iface.out[i].type || "*";
    } else {
      hideSlot(slot, true);
    }
  }

  const claimedSockets = new Set();
  for (const p of params) {
    if (p.socket === undefined || p.socket === null) continue;
    const k = Number(p.socket);
    if (!Number.isFinite(k) || k < 0 || k >= MAX_VAULT_PARAMS) continue;
    claimedSockets.add(k);
    const slot = inputSlotByName(node, `param_${k}`);
    if (!slot) continue;
    hideSlot(slot, false);
    slot.name = p.name || p.id;
    slot.type = manifestTypeToSlotType(p.type);
  }

  for (let k = 0; k < MAX_VAULT_PARAMS; k++) {
    const slot = inputSlotByName(node, `param_${k}`);
    if (!slot) continue;
    if (!claimedSockets.has(k)) hideSlot(slot, true);
  }

  reconcilePromotedWidgets(node, params, isSealed);

  node.properties = node.properties || {};
  node.properties.vault_slots = { in: mIn, out: mOut, params: params.length };

  const badge = isSealed ? "📦 SEALED" : (unlockedSessions.has(widget(node, "vault_id")?.value) ? "🔓 UNLOCKED" : "🔒 LOCKED");
  const base = isSealed ? "C2C Vault — Sealed" : "C2C Vault — Locked";
  node.title = `${badge}  ${base}`;

  const summary = node._vaultSummaryEl;
  if (summary) {
    const mParams = params.length;
    summary.textContent = `${iface.node_count || 0} nodes · ${mIn} inputs · ${mOut} outputs · ${mParams} params`;
  }

  const origCompute = node._vaultOrigComputeSize || node.computeSize?.bind(node);
  if (!node._vaultOrigComputeSize) node._vaultOrigComputeSize = origCompute;
  const hiddenParams = MAX_VAULT_PARAMS - claimedSockets.size;
  node.computeSize = function (outW) {
    const sz = origCompute ? origCompute(outW) : [node.size?.[0] || 200, 120];
    const hiddenIn = MAX_VAULT_INPUTS - mIn;
    const hiddenOut = MAX_VAULT_OUTPUTS - mOut;
    const hiddenSlots = Math.max(hiddenIn, hiddenOut, hiddenParams);
    sz[1] = Math.max(60, sz[1] - hiddenSlots * SLOT_H);
    return sz;
  };
  node.setSize(node.computeSize());
  node.setDirtyCanvas?.(true, true);
}

function buildInterfaceManifest(mode, nodeCount, boundary, inTypes, outTypes, params = []) {
  return {
    mode,
    node_count: nodeCount,
    in: boundary.boundary_in.map((b, i) => ({
      name: b.name, type: inTypes[i] || b.type || "*",
    })),
    out: boundary.boundary_out.map((b, i) => ({
      name: b.name, type: outTypes[i] || b.type || "*",
    })),
    params: (params || []).map((p) => {
      const entry = {
        id: p.id,
        name: p.name,
        type: p.type,
        value: p.value,
      };
      if (p.socket !== undefined && p.socket !== null) entry.socket = p.socket;
      return entry;
    }),
  };
}

function selectionCentroid(sel) {
  let x = 0, y = 0;
  for (const n of sel) { x += n.pos[0]; y += n.pos[1]; }
  return [x / sel.length, y / sel.length];
}

function openVaultOverlay(node, subgraph, isSealed) {
  if (headless()) return;
  const vaultId = widget(node, "vault_id")?.value || "";
  if (vaultEditSessions.has(vaultId)) {
    vaultEditSessions.get(vaultId).panel?.focus?.();
    return;
  }

  const panel = document.createElement("div");
  panel.className = "c2c-win";
  css(panel, {
    position: "fixed", top: "80px", left: "80px", zIndex: "8500",
    width: "520px", maxHeight: "70vh", overflow: "auto",
    background: "var(--comfy-menu-bg, #353535)",
    color: "var(--fg-color, #ddd)",
    border: "1px solid var(--border-color, #4a4a4a)",
    borderRadius: "8px", padding: "12px",
    boxShadow: "0 12px 40px rgba(0,0,0,0.55)",
  });

  const hdr = document.createElement("div");
  hdr.textContent = `Vault editor — ${vaultId}`;
  css(hdr, { fontWeight: "600", marginBottom: "8px" });

  const warn = document.createElement("div");
  warn.textContent = "Read-only preview of decrypted contents. Edits here are NOT saved to the workflow — close before Ctrl+S. Native subgraph was rejected because it would serialise plaintext.";
  css(warn, { fontSize: "11px", opacity: "0.75", marginBottom: "10px", lineHeight: "1.4" });

  const list = document.createElement("div");
  for (const n of (subgraph.nodes || [])) {
    const line = document.createElement("div");
    line.textContent = `• ${n.class_type} (id ${n.id})`;
    css(line, { fontFamily: "ui-monospace, monospace", fontSize: "12px", marginBottom: "2px" });
    list.append(line);
  }

  const closeBtn = document.createElement("button");
  closeBtn.textContent = "Close editor";
  css(closeBtn, { marginTop: "12px", padding: "6px 14px", cursor: "pointer" });

  const dispose = () => {
    vaultEditSessions.delete(vaultId);
    panel.remove();
  };
  closeBtn.onclick = dispose;

  panel.append(hdr, warn, list, closeBtn);
  document.body.append(panel);
  vaultEditSessions.set(vaultId, { subgraph, panel, dispose });
}

function vaultSaveGuardMessage() {
  return "A vault editor is open. Close it before saving — decrypted contents live only in memory and must never be written to workflow JSON. (Native subgraph was rejected for this reason.)";
}

function installSaveGuard() {
  if (headless() || app._c2cVaultSaveGuard) return;
  app._c2cVaultSaveGuard = true;

  // Native convertToSubgraph EXISTS in ComfyUI, but we deliberately do NOT use it:
  // a native subgraph serialises its inner nodes into workflow JSON on save, so
  // Ctrl+S while editing would write the decrypted graph to disk. Overlay only.
  const origGTP = app.graphToPrompt?.bind(app);
  if (origGTP) {
    app.graphToPrompt = async function (...args) {
      if (vaultEditSessions.size > 0) {
        await modalAlert("Save blocked", vaultSaveGuardMessage());
        throw new Error("C2C Vault: save blocked while editor open");
      }
      return origGTP(...args);
    };
  }

  document.addEventListener("keydown", (e) => {
    if (vaultEditSessions.size === 0) return;
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault();
      e.stopPropagation();
      modalAlert("Save blocked", vaultSaveGuardMessage());
    }
  }, true);
}

async function handleUnlockOrOpen(node, isSealed, forEdit = false) {
  const id = widget(node, "vault_id")?.value || "";
  const payload = widget(node, "vault_payload")?.value || "";
  if (!payload) { await modalAlert("No payload", "This vault has no payload yet. Lock a selection first."); return; }

  const pw = await modalPassword({
    title: forEdit || isSealed ? "Open vault for editing" : "Unlock vault",
    note: isSealed
      ? "Sealed vaults run without a password. The password only unlocks the ciphertext for editing."
      : "Password is sent once to the server and never stored in the workflow.",
    confirmLabel: forEdit || isSealed ? "Open" : "Unlock",
  });
  if (pw === null) return;

  const route = (forEdit || isSealed) ? "/c2c_vault/open" : "/c2c_vault/unlock";
  const { ok, data } = await post(route, { vault_id: id, password: pw, payload });

  if (!ok) {
    await modalAlert("Could not open vault", data.error || "Wrong password or modified payload.");
    return;
  }

  if (!isSealed && route === "/c2c_vault/unlock") unlockedSessions.add(id);

  if (forEdit || isSealed) {
    openVaultOverlay(node, data.subgraph, isSealed);
  } else {
    await modalAlert("Vault unlocked", "Session unlocked for this ComfyUI process. Queue the workflow to run.");
  }

  const iface = parseInterface(widget(node, "vault_interface")?.value);
  applyVaultInterface(node, iface, isSealed);
}

app.registerExtension({
  name: "Code2Collapse.CustomNodePacks.Vault",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData.name)) return;
    const isSealed = nodeData.name === "C2C_VaultSealed";

    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const node = this;

      for (const wn of ["vault_payload", "vault_interface"]) {
        const w = widget(node, wn);
        if (w) {
          w.computeSize = () => [0, -4];
          if (w.inputEl) w.inputEl.hidden = true;
        }
      }

      const host = document.createElement("div");
      css(host, {
        width: "100%", padding: "4px 0", fontSize: "11px",
        opacity: "0.8", textAlign: "center",
        color: "var(--fg-color, #ddd)",
      });
      node._vaultSummaryEl = host;
      host.textContent = "0 nodes · 0 inputs · 0 outputs · 0 params";
      node.addDOMWidget("vault_summary", "summary", host, { serialize: false });

      node.addWidget("button", isSealed ? "Open for editing…" : "Unlock…", null, () => {
        handleUnlockOrOpen(node, isSealed, isSealed);
      });

      node.addWidget("button", "Lock session", null, async () => {
        const id = widget(node, "vault_id")?.value || "";
        await post("/c2c_vault/lock_session", { vault_id: id });
        unlockedSessions.delete(id);
        const iface = parseInterface(widget(node, "vault_interface")?.value);
        applyVaultInterface(node, iface, isSealed);
        await modalAlert("Session locked", "Vault session cleared. Enter the password again to queue.");
      });

      setTimeout(() => {
        const iface = parseInterface(widget(node, "vault_interface")?.value);
        if (iface.in.length || iface.out.length || iface.node_count || iface.params.length) {
          applyVaultInterface(node, iface, isSealed);
        }
      }, 0);

      return r;
    };

    const origSlotMenu = nodeType.prototype.getSlotMenuOptions;
    nodeType.prototype.getSlotMenuOptions = function (slot) {
      const items = origSlotMenu?.call(this, slot) || [];
      const inp = slot?.input;
      if (!inp?.name?.startsWith("param_")) return items;
      const k = Number(inp.name.slice(6));
      const iface = readNodeInterface(this);
      const param = (iface.params || []).find((p) => Number(p.socket) === k);
      if (!param || !PROMOTED_VALUE_TYPES.has(param.type)) return items;
      items.push(null);
      items.push({
        content: "Convert to widget",
        callback: () => convertPromotedParamToWidget(this, param.id, isSealed),
      });
      return items;
    };

    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = configured?.apply(this, arguments);
      const iface = parseInterface(widget(this, "vault_interface")?.value);
      applyVaultInterface(this, iface, isSealed);
      return r;
    };

    const dbl = nodeType.prototype.onDblClick;
    nodeType.prototype.onDblClick = function (...args) {
      const r = dbl?.apply(this, arguments);
      handleUnlockOrOpen(this, isSealed, isSealed);
      return r;
    };

    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function (...args) {
      const id = widget(this, "vault_id")?.value || "";
      const sess = vaultEditSessions.get(id);
      if (sess) sess.dispose();
      unlockedSessions.delete(id);
      return removed?.apply(this, arguments);
    };
  },

  setup() {
    installSaveGuard();

    const orig = app.canvas.getCanvasMenuOptions;
    app.canvas.getCanvasMenuOptions = function () {
      const opts = orig?.apply(this, arguments) || [];
      opts.push({
        content: "C2C Vault → Lock selection (password to run)",
        callback: () => lockSelection(false),
      });
      opts.push({
        content: "C2C Vault → Seal selection (runs without password)",
        callback: () => lockSelection(true),
      });
      return opts;
    };

    async function lockSelection(sealed) {
      if (headless()) return;
      const sel = Object.values(app.canvas.selected_nodes || {});
      if (sel.length < 1) {
        await modalAlert("Nothing selected", "Select the nodes to lock first.");
        return;
      }

      const pw = await modalPasswordStep({
        title: `${sealed ? "Seal" : "Lock"} ${sel.length} node(s) into a vault`,
        note: sealed
          ? "Sealed: recipient runs with no password but cannot read the graph. Your password is still required to edit — store it safely; there is no recovery."
          : "Locked: recipient needs this password to run at all. Store it safely — there is no recovery, by design.",
        sealed,
      });
      if (pw === null) return;

      const derived = deriveBoundary(sel, app.graph);
      if (!derived.boundary_out.length) {
        await modalAlert("No outputs", "These nodes produce no output. Include the node whose result you want.");
        return;
      }
      if (derived.boundary_in.length > MAX_VAULT_INPUTS) {
        await modalAlert("Too many inputs", `Selection needs ${derived.boundary_in.length} external inputs but vault supports ${MAX_VAULT_INPUTS}. Include more upstream nodes inside the selection.`);
        return;
      }
      if (derived.boundary_out.length > MAX_VAULT_OUTPUTS) {
        await modalAlert("Too many outputs", `Selection needs ${derived.boundary_out.length} outputs but vault supports ${MAX_VAULT_OUTPUTS}.`);
        return;
      }

      const confirmed = await modalBoundaryStep({
        boundary: derived, nodeCount: sel.length, sealed,
      });
      if (!confirmed) return;

      const promotion = await modalPromoteStep({ selection: sel });
      if (promotion === null) return;

      const idOf = (n) => String(n.id);
      const nodes = sel.map((n) => ({
        id: idOf(n),
        class_type: n.comfyClass || n.type,
        widgets: Object.fromEntries((n.widgets || []).map((w) => [w.name, w.value])),
      }));

      const boundary_in = derived.boundary_in.map((b) => ({
        name: b.name, to: b.to, to_slot: b.to_slot,
      }));
      const boundary_out = derived.boundary_out.map((b) => ({
        name: b.name, from: b.from, from_slot: b.from_slot,
      }));

      const vault_id = `vault-${Math.random().toString(36).slice(2, 10)}`;
      const { ok, data } = await post("/c2c_vault/lock", {
        vault_id, password: pw, mode: sealed ? "sealed" : "locked",
        subgraph: {
          nodes, links: derived.links, boundary_in, boundary_out,
          promoted: promotion.promoted || [],
        },
      });
      if (!ok) {
        await modalAlert("Lock failed", data.error || "Could not lock the selection.");
        return;
      }

      const iface = buildInterfaceManifest(
        sealed ? "sealed" : "locked", sel.length,
        { boundary_in, boundary_out },
        derived.inTypes, derived.outTypes,
        promotion.params || [],
      );

      const [cx, cy] = selectionCentroid(sel);
      const vault = LiteGraph.createNode(sealed ? "C2C_VaultSealed" : "C2C_VaultLocked");
      vault.pos = [cx, cy];
      app.graph.add(vault);
      widget(vault, "vault_id").value = vault_id;
      widget(vault, "vault_payload").value = data.payload;
      const ifaceW = widget(vault, "vault_interface");
      if (ifaceW) ifaceW.value = JSON.stringify(iface);
      vault._vaultPromotedCombo = {};
      for (const p of promotion.params || []) {
        if (p._comboOptions) vault._vaultPromotedCombo[p.id] = p._comboOptions;
      }
      applyVaultInterface(vault, iface, sealed);

      derived.externalSources.forEach((src, i) => {
        const srcNode = app.graph.getNodeById(src.id);
        if (srcNode) srcNode.connect(src.slot, vault, i);
      });
      derived.externalTargets.forEach((tgt, i) => {
        if (!tgt) return;
        const tgtNode = app.graph.getNodeById(tgt.id);
        if (tgtNode) vault.connect(i, tgtNode, tgt.slot);
      });

      for (const n of sel) app.graph.remove(n);
      app.graph.setDirtyCanvas(true, true);
      await modalAlert(
        sealed ? "Sealed" : "Locked",
        `${sealed ? "Sealed" : "Locked"} ${sel.length} node(s) into a vault. Original nodes removed.`,
      );
    }
  },
});
