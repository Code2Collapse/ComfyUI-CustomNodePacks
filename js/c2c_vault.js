import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { legacyCanvasMenu } from "./_c2c_compat.js";
import { asOneUndoStep } from "./_c2c_undo_scope.js";
import { setHidden } from "./_widget_visibility.js";

/**
 * C2C Vault — password-locked subgraph.
 *
 * The password is typed into the field ON the node (or the double-click modal) and POSTed to
 * /c2c_vault/unlock or /open. It lives only in that <input>: the widget's value is always "" and is not
 * serialised, so it never reaches the workflow JSON, the queued prompt, undo history or the clipboard.
 *
 * Edit view uses a floating overlay — NEVER convertToSubgraph. ComfyUI's
 * native subgraph API exists, but a native subgraph serialises its inner nodes
 * into workflow JSON on save, which would write the decrypted graph to disk.
 */

const NODES = ["C2C_VaultLocked", "C2C_VaultSealed"];
const MAX_VAULT_INPUTS = 10;
const MAX_VAULT_OUTPUTS = 8;
const MAX_VAULT_PARAMS = 8;
const PROMOTED_VALUE_TYPES = new Set(["INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"]);

const SETTING_SCOPE = "c2c.vault.scope";
const SCOPE_SELECTED = "Selected nodes";
const SCOPE_WHOLE = "Whole workflow";
const WORKFLOW_VAULT_NOTE =
  "C2C Vault encrypted workflow: open it in ComfyUI with ComfyUI-CustomNodePacks and the password.";
const EXPORT_CMD_IDS = ["Comfy.ExportWorkflow", "Comfy.ExportWorkflowAPI"];

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
    back.setAttribute("role", "dialog");
    back.setAttribute("aria-modal", "true");
    css(back, {
      position: "fixed", inset: "0", zIndex: "10000",
      background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center",
    });
    const box = document.createElement("div");
    css(box, {
      background: "var(--c2c-bg2)",
      color: "var(--c2c-fg)",
      border: "1px solid var(--c2c-border)",
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
      background: "var(--c2c-surface0)",
      color: "var(--c2c-fg)",
      border: "1px solid var(--c2c-border)",
    });

    const pw2 = document.createElement("input");
    pw2.type = "password";
    pw2.placeholder = "Confirm password";
    pw2.autocomplete = "new-password";
    css(pw2, {
      width: "100%", boxSizing: "border-box", marginBottom: "8px",
      padding: "7px 9px", borderRadius: "4px",
      background: "var(--c2c-surface0)",
      color: "var(--c2c-fg)",
      border: "1px solid var(--c2c-border)",
    });

    const genRow = document.createElement("div");
    css(genRow, { display: "flex", gap: "8px", marginBottom: "8px", alignItems: "center" });
    const genBtn = document.createElement("button");
    genBtn.textContent = "Generate strong password";
    const genShow = document.createElement("span");
    genShow.textContent = "";
    css(genShow, { fontFamily: "ui-monospace, monospace", fontSize: "11px", color: "var(--c2c-green)", flex: "1", wordBreak: "break-all" });
    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy";
    copyBtn.style.display = "none";
    for (const b of [genBtn, copyBtn]) {
      css(b, {
        padding: "4px 10px", borderRadius: "4px", cursor: "pointer", fontSize: "11px",
        background: "var(--c2c-surface0)",
        color: "var(--c2c-fg)",
        border: "1px solid var(--c2c-border)",
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
    css(err, { color: "var(--c2c-dangerStrong)", minHeight: "16px", marginBottom: "8px" });

    const row = document.createElement("div");
    css(row, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const next = document.createElement("button");
    next.textContent = "Next →";
    for (const b of [cancel, next]) {
      css(b, {
        padding: "6px 14px", borderRadius: "4px", cursor: "pointer",
        background: "var(--c2c-surface0)",
        color: "var(--c2c-fg)",
        border: "1px solid var(--c2c-border)",
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
    back.setAttribute("role", "dialog");
    back.setAttribute("aria-modal", "true");
    css(back, {
      position: "fixed", inset: "0", zIndex: "10000",
      background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center",
    });
    const box = document.createElement("div");
    css(box, {
      background: "var(--c2c-bg2)",
      color: "var(--c2c-fg)",
      border: "1px solid var(--c2c-border)",
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
          background: "var(--c2c-surface0)", color: "var(--c2c-fg)",
          border: "1px solid var(--c2c-border)",
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
      css(none, { color: "var(--c2c-dangerStrong)", marginBottom: "8px", fontSize: "12px" });
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
        background: "var(--c2c-surface0)",
        color: "var(--c2c-fg)",
        border: "1px solid var(--c2c-border)",
      });
    }
    const done = (v) => { back.remove(); resolve(v); };
    cancel.onclick = () => done(null);
    ok.onclick = () => {
      if (!boundary.boundary_out.length) { shakeEl(box); return; }
      done(boundary);
    };
    back.onkeydown = (e) => {
      if (e.key === "Escape") done(null);
      if (e.key === "Enter") ok.click();
    };
    row.append(cancel, ok);
    box.append(row);
    back.append(box);
    // This step was built but never added to the page, so Lock and Seal waited forever after the password
    // and the vault could never be created (A9 "vault is not working", docs/evidence/L2.24).
    document.body.append(back);
    back.tabIndex = -1;
    ok.focus();
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
    back.setAttribute("role", "dialog");
    back.setAttribute("aria-modal", "true");
    css(back, {
      position: "fixed", inset: "0", zIndex: "10000",
      background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center",
    });
    const box = document.createElement("div");
    css(box, {
      background: "var(--c2c-bg2)",
      color: "var(--c2c-fg)",
      border: "1px solid var(--c2c-border)",
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
          background: "var(--c2c-surface0)", color: "var(--c2c-fg)",
          border: "1px solid var(--c2c-border)",
        });
        labelIn.oninput = () => { row.label = labelIn.value; };

        const modeSel = document.createElement("select");
        css(modeSel, {
          padding: "4px 6px", borderRadius: "4px",
          background: "var(--c2c-surface0)", color: "var(--c2c-fg)",
          border: "1px solid var(--c2c-border)",
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
        background: "var(--c2c-surface0)",
        color: "var(--c2c-fg)",
        border: "1px solid var(--c2c-border)",
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
    back.setAttribute("role", "dialog");
    back.setAttribute("aria-modal", "true");
    css(back, {
      position: "fixed", inset: "0", zIndex: "10000",
      background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center",
    });
    const box = document.createElement("div");
    css(box, {
      background: "var(--c2c-bg2)", color: "var(--c2c-fg)",
      border: "1px solid var(--c2c-border)", borderRadius: "6px",
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
    back.setAttribute("role", "dialog");
    back.setAttribute("aria-modal", "true");
    css(back, {
      position: "fixed", inset: "0", zIndex: "10000",
      background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center",
    });
    const box = document.createElement("div");
    css(box, {
      background: "var(--c2c-bg2)", color: "var(--c2c-fg)",
      border: "1px solid var(--c2c-border)", borderRadius: "6px",
      padding: "18px 20px", minWidth: "340px", font: "13px sans-serif",
    });
    const pw = document.createElement("input");
    pw.type = "password";
    pw.autocomplete = "current-password";
    css(pw, {
      width: "100%", boxSizing: "border-box", margin: "10px 0",
      padding: "7px 9px", borderRadius: "4px",
      background: "var(--c2c-surface0)", color: "var(--c2c-fg)",
      border: "1px solid var(--c2c-border)",
    });
    const err = document.createElement("div");
    css(err, { color: "var(--c2c-dangerStrong)", minHeight: "16px" });
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
  // Frontend 1.52 keeps links in a Map: an index lookup on graph.links returned undefined for every wire, so the vault saw no
  // internal links and no inputs.
  const linkOf = (id) => {
    if (id == null) return null;
    const L = graph.links;
    return (L instanceof Map ? L.get(id) : L?.[id]) ?? graph._links?.get?.(id) ?? null;
  };
  const ids = new Set(sel.map(idOf));
  const links = [];
  const boundary_in = [];
  const externalSources = [];
  const inTypes = [];

  for (const n of sel) {
    (n.inputs || []).forEach((inp, slot) => {
      const lk = linkOf(inp.link);
      if (!lk) return;
      if (ids.has(String(lk.origin_id))) {
        links.push({
          from: String(lk.origin_id), from_slot: lk.origin_slot,
          to: idOf(n), to_slot: slot, to_name: inp.name,
        });
      } else {
        const origin = graph.getNodeById?.(lk.origin_id);
        const outType = origin?.outputs?.[lk.origin_slot]?.type || "*";
        boundary_in.push({
          name: inp.name || `in_${boundary_in.length}`,
          to: idOf(n), to_slot: slot, to_name: inp.name, type: outType,
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
        const lk = linkOf(lid);
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

// A socket's NAME is the key ComfyUI sends in the prompt (input_0, param_3, ...), so it is never changed: the
// boundary name the user sees is the slot's LABEL. Renaming `name` sent "destination" instead of "input_0", and
// the server answered "input 0 ('destination') is not connected" (A9, docs/evidence/L2.24).
const WIRE_SOCKET = /^(input|param)_\d+$/;

/** Vaults saved while the names were overwritten still carry all 18 wire sockets (input_0..9, then param_0..7,
 *  in declaration order): give them their canonical names back. Wire sockets are the non-widget inputs. */
function canonicalizeVaultSockets(node) {
  const sockets = (node.inputs || []).filter((inp) => !inp.widget);
  if (sockets.length !== MAX_VAULT_INPUTS + MAX_VAULT_PARAMS) return;   // current saves: names already canonical
  if (sockets.every((inp) => WIRE_SOCKET.test(inp.name || ""))) return;
  sockets.forEach((inp, k) => {
    const want = k < MAX_VAULT_INPUTS ? `input_${k}` : `param_${k - MAX_VAULT_INPUTS}`;
    if (inp.name === want) return;
    if (inp.label == null && inp.name) inp.label = inp.name;
    inp.name = want;
  });
}

/** Show exactly the vault's boundary: the sockets the interface uses exist (canonical name, boundary label);
 *  unused, unconnected ones are removed instead of drawn as rows of unlabelled dots. */
function syncVaultSockets(node, iface, params) {
  const want = new Map();   // socket name -> { label, type }
  (iface.in || []).slice(0, MAX_VAULT_INPUTS).forEach((b, i) => {
    want.set(`input_${i}`, { label: b.name || `in_${i}`, type: b.type || "*" });
  });
  for (const p of params) {
    const k = Number(p.socket);
    if (p.socket === undefined || p.socket === null || !Number.isFinite(k) || k < 0 || k >= MAX_VAULT_PARAMS) continue;
    want.set(`param_${k}`, { label: p.name || p.id, type: manifestTypeToSlotType(p.type) });
  }
  for (let i = (node.inputs || []).length - 1; i >= 0; i--) {
    const inp = node.inputs[i];
    if (inp.widget || !WIRE_SOCKET.test(inp.name || "")) continue;
    const w = want.get(inp.name);
    if (w) { inp.label = w.label; inp.type = w.type; }
    else if (inp.link == null) node.removeInput(i);
  }
  for (const [name, w] of want) {
    if (inputSlotByName(node, name)) continue;
    node.addInput(name, w.type);
    const slot = inputSlotByName(node, name);
    if (slot) slot.label = w.label;
  }
  const mOut = Math.min(iface.out?.length || 0, MAX_VAULT_OUTPUTS);
  for (let i = (node.outputs || []).length - 1; i >= mOut; i--) {
    if ((node.outputs[i].links || []).length) break;   // a wired output keeps every output before it
    node.removeOutput(i);
  }
  for (let i = 0; i < mOut; i++) {
    if (!node.outputs?.[i]) node.addOutput(`output_${i}`, iface.out[i].type || "*");
    node.outputs[i].label = iface.out[i].name || `out_${i}`;
    node.outputs[i].type = iface.out[i].type || "*";
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
  canonicalizeVaultSockets(node);
  const mIn = iface.in?.length || 0;
  const mOut = iface.out?.length || 0;
  const params = iface.params || [];
  syncVaultSockets(node, iface, params);

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

  // the sockets match the boundary now, so the node's own size is right (no hidden-row arithmetic)
  if (node._vaultOrigComputeSize) { node.computeSize = node._vaultOrigComputeSize; delete node._vaultOrigComputeSize; }
  const sz = node.computeSize?.();
  if (sz) node.setSize([Math.max(node.size?.[0] || 0, sz[0]), sz[1]]);
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
    background: "var(--c2c-bg2)",
    color: "var(--c2c-fg)",
    border: "1px solid var(--c2c-border)",
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
  node._vaultAccess?.refresh();
}

/** Double-click: a Locked vault whose session is open opens for editing at once (the key is already held);
 *  otherwise the password is asked first. */
async function openOnDoubleClick(node, isSealed) {
  const id = widget(node, "vault_id")?.value || "";
  if (!isSealed && unlockedSessions.has(id)) {
    const { ok, data } = await post("/c2c_vault/open", { vault_id: id, payload: widget(node, "vault_payload")?.value || "" });
    if (ok) { openVaultOverlay(node, data.subgraph, false); return; }
    unlockedSessions.delete(id);       // expired on the server: fall through to the password
    node._vaultAccess?.refresh();
  }
  return handleUnlockOrOpen(node, isSealed, isSealed);
}

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

  const graph = app.canvas?.graph || app.graph;
  const derived = deriveBoundary(sel, graph);
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
    // real inputs only: display widgets ("$$canvas-image-preview", buttons, serialize:false) are not node inputs
    widgets: Object.fromEntries((n.widgets || [])
      .filter((w) => w?.name && !String(w.name).startsWith("$$") && w.type !== "button"
        && w.serialize !== false && w.options?.serialize !== false)
      .map((w) => [w.name, w.value])),
  }));

  const boundary_in = derived.boundary_in.map((b) => ({
    name: b.name, to: b.to, to_slot: b.to_slot, to_name: b.to_name,
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
  if (!sealed && data.session_open) unlockedSessions.add(vault_id);

  const iface = buildInterfaceManifest(
    sealed ? "sealed" : "locked", sel.length,
    { boundary_in, boundary_out },
    derived.inTypes, derived.outTypes,
    promotion.params || [],
  );

  const [cx, cy] = selectionCentroid(sel);
  const vault = LiteGraph.createNode(sealed ? "C2C_VaultSealed" : "C2C_VaultLocked");
  vault.pos = [cx, cy];
  // one undo step: the vault node, its wiring and the removal of the originals
  asOneUndoStep(app.canvas, () => {
    graph.add(vault);
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
      const srcNode = graph.getNodeById(src.id);
      const slot = (vault.inputs || []).findIndex((inp) => inp.name === `input_${i}`);
      if (srcNode && slot >= 0) srcNode.connect(src.slot, vault, slot);
    });
    derived.externalTargets.forEach((tgt, i) => {
      if (!tgt) return;
      const tgtNode = graph.getNodeById(tgt.id);
      if (tgtNode) vault.connect(i, tgtNode, tgt.slot);
    });

    for (const n of sel) graph.remove(n);
  });
  vault._vaultAccess?.refresh();      // the field now shows "Unlocked" (the creator's session) or "Sealed"
  graph.setDirtyCanvas(true, true);
  await modalAlert(
    sealed ? "Sealed" : "Locked",
    `${sealed ? "Sealed" : "Locked"} ${sel.length} node(s) into a vault. Original nodes removed.`,
  );
}

function vaultScope() {
  try {
    return app.ui?.settings?.getSettingValue?.(SETTING_SCOPE, SCOPE_SELECTED) || SCOPE_SELECTED;
  } catch (_) {
    return SCOPE_SELECTED;
  }
}

function getWorkflowVaultMarker() {
  const m = app.graph?.extra?.c2c_vault_workflow;
  if (m?.vault_id && m?.payload) return m;
  return null;
}

function buildWorkflowFileWrapper(marker) {
  return {
    c2c_encrypted_workflow: 1,
    vault_id: marker.vault_id,
    mode: "locked",
    payload: marker.payload,
    note: WORKFLOW_VAULT_NOTE,
  };
}

function stampWorkflowVaultMarker(file) {
  if (!app.graph) return;
  app.graph.extra = app.graph.extra || {};
  app.graph.extra.c2c_vault_workflow = { vault_id: file.vault_id, payload: file.payload };
}

function sanitizeWorkflowFilename(raw) {
  let name = String(raw || "").trim();
  name = name.replace(/^workflows\//i, "").replace(/\.json$/i, "");
  name = name.replace(/[/\\:]/g, "_").slice(0, 200).trim();
  return name;
}

function modalPromptFilename() {
  if (headless()) return Promise.resolve(null);
  return new Promise((resolve) => {
    const back = document.createElement("div");
    back.setAttribute("role", "dialog");
    back.setAttribute("aria-modal", "true");
    css(back, {
      position: "fixed", inset: "0", zIndex: "10000",
      background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center",
    });
    const box = document.createElement("div");
    css(box, {
      background: "var(--c2c-bg2)", color: "var(--c2c-fg)",
      border: "1px solid var(--c2c-border)", borderRadius: "6px",
      padding: "18px 20px", minWidth: "340px", font: "13px sans-serif",
    });
    const inp = document.createElement("input");
    inp.type = "text";
    inp.placeholder = "Workflow name";
    css(inp, {
      width: "100%", boxSizing: "border-box", margin: "10px 0",
      padding: "7px 9px", borderRadius: "4px",
      background: "var(--c2c-surface0)", color: "var(--c2c-fg)",
      border: "1px solid var(--c2c-border)",
    });
    const row = document.createElement("div");
    css(row, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const ok = document.createElement("button");
    ok.textContent = "Save";
    for (const b of [cancel, ok]) css(b, { padding: "6px 14px", cursor: "pointer" });
    const done = (v) => { back.remove(); resolve(v); };
    cancel.onclick = () => done(null);
    ok.onclick = () => {
      const n = sanitizeWorkflowFilename(inp.value);
      if (!n) { shakeEl(box); return; }
      done(n);
    };
    box.append(
      Object.assign(document.createElement("div"), {
        textContent: "Save encrypted workflow",
        style: { fontWeight: "600" },
      }),
      Object.assign(document.createElement("div"), {
        textContent: "Enter a name for this workflow file.",
        style: { opacity: "0.75", fontSize: "12px", marginTop: "6px" },
      }),
      inp, row,
    );
    row.append(cancel, ok);
    back.append(box);
    document.body.append(back);
    inp.focus();
  });
}

async function resolveWorkflowFilename() {
  const aw = app.extensionManager?.workflow?.activeWorkflow;
  const raw = aw?.filename ?? aw?.name ?? aw?.path ?? "";
  const name = sanitizeWorkflowFilename(raw);
  if (!name || /^unsaved/i.test(name)) return modalPromptFilename();
  return name;
}

function deepCloneWorkflow(obj) {
  return JSON.parse(JSON.stringify(obj));
}

function workflowForRelock(workflow) {
  const copy = deepCloneWorkflow(workflow);
  if (copy.extra?.c2c_vault_workflow) {
    copy.extra = { ...copy.extra };
    delete copy.extra.c2c_vault_workflow;
  }
  return copy;
}

function parseStoredWorkflowData(data) {
  if (data == null) return null;
  if (typeof data === "string") {
    try { return JSON.parse(data); } catch (_) { return null; }
  }
  if (typeof data === "object") return data;
  return null;
}

function isWorkflowSavePath(path) {
  return typeof path === "string" && /^workflows\/.+\.json$/i.test(path);
}

function isComfyWorkflowObject(obj) {
  return !!obj && typeof obj === "object" && Array.isArray(obj.nodes)
    && obj.c2c_encrypted_workflow !== 1;
}

function workflowVaultMarkerFromData(parsed, fallback) {
  const m = parsed?.extra?.c2c_vault_workflow;
  if (m?.vault_id && m?.payload) return m;
  return fallback;
}

async function unlockEncryptedWorkflowFile(file) {
  while (true) {
    const password = await modalPassword({
      title: "Open encrypted workflow",
      note: "Enter the password for this workflow file.",
      confirmLabel: "Open",
    });
    if (password === null) return null;
    const { ok, data, status } = await post("/c2c_vault/workflow/unlock", { file, password });
    if (ok) return data.workflow;
    if (status === 429) {
      await modalAlert("Too many attempts", data.error || "Try again later.");
      return null;
    }
    await modalAlert("Could not open workflow", data.error || "Wrong password or modified file.");
  }
}

async function relockWorkflowForSave(workflow, marker) {
  const wf = workflowForRelock(workflow);
  let { ok, data, status } = await post("/c2c_vault/workflow/relock", {
    vault_id: marker.vault_id,
    payload: marker.payload,
    workflow: wf,
  });
  if (ok) return data.file;
  if (status === 403) {
    const password = await modalPassword({
      title: "Session expired",
      note: "Re-enter the workflow password to save it encrypted.",
      confirmLabel: "Unlock",
    });
    if (password === null) throw new Error("C2C Vault: save cancelled — session expired.");
    const file = buildWorkflowFileWrapper(marker);
    const unlock = await post("/c2c_vault/workflow/unlock", { file, password });
    if (!unlock.ok) {
      throw new Error(unlock.data?.error || "C2C Vault: unlock failed.");
    }
    marker = { vault_id: file.vault_id, payload: file.payload };
    ({ ok, data, status } = await post("/c2c_vault/workflow/relock", {
      vault_id: marker.vault_id,
      payload: marker.payload,
      workflow: wf,
    }));
    if (ok) return data.file;
  }
  throw new Error(data?.error || "C2C Vault: could not re-encrypt the workflow.");
}

async function saveWorkflowEncrypted() {
  const pw = await modalPasswordStep({
    title: "Save workflow encrypted",
    note: "The saved file will be ciphertext. Nothing runs until the password is entered.",
    sealed: false,
  });
  if (pw === null) return;

  const workflow = app.graph?.serialize?.() ?? app.canvas?.graph?.serialize?.();
  if (!workflow) {
    await modalAlert("No workflow", "Nothing to encrypt.");
    return;
  }

  const { ok, data } = await post("/c2c_vault/workflow/lock", { password: pw, workflow });
  if (!ok) {
    await modalAlert("Encrypt failed", data.error || "Could not encrypt the workflow.");
    return;
  }

  const name = await resolveWorkflowFilename();
  if (!name) return;

  try {
    await api.storeUserData(`workflows/${name}.json`, data.file, { overwrite: true, stringify: true });
  } catch (e) {
    await modalAlert("Save failed", String(e?.message || e));
    return;
  }

  stampWorkflowVaultMarker(data.file);
  await modalAlert("Saved encrypted", `Workflow saved as workflows/${name}.json`);
}

/** Output nodes (Preview / Save ...) stay OUTSIDE a whole-workflow seal: a graph whose only top-level node is the
 *  vault has no output, so ComfyUI refuses to queue it ("Prompt has no outputs", measured), and the results would be
 *  invisible anyway. Everything else goes inside and is wired to them through the vault's outputs. */
function isOutputNode(n) {
  return !!(n?.constructor?.nodeData?.output_node || n?.constructor?.nodeData?.output_node === "true");
}

function selectNodesForWholeSeal() {
  const canvas = app.canvas;
  const graph = canvas?.graph || app.graph;
  if (!canvas || !graph) return 0;
  canvas.selected_nodes = {};
  let count = 0;
  for (const n of graph._nodes || []) {
    if (n && !isOutputNode(n) && !NODES.includes(n.type)) { canvas.selected_nodes[n.id] = n; count++; }
  }
  return count;
}

async function sealWholeWorkflow() {
  if (!selectNodesForWholeSeal()) {
    await modalAlert("Nothing to seal", "The workflow has no nodes other than its outputs.");
    return;
  }
  await lockSelection(true);
}

function installLoadGraphDataHook() {
  if (app._c2cVaultLoadHooked) return;
  const orig = app.loadGraphData?.bind(app);
  if (!orig) return;
  app._c2cVaultLoadHooked = true;
  app.loadGraphData = async function (graphData, ...rest) {
    if (graphData?.c2c_encrypted_workflow === 1) {
      const workflow = await unlockEncryptedWorkflowFile(graphData);
      if (!workflow) return undefined;
      workflow.extra = workflow.extra || {};
      workflow.extra.c2c_vault_workflow = {
        vault_id: graphData.vault_id,
        payload: graphData.payload,
      };
      return orig(workflow, ...rest);
    }
    return orig(graphData, ...rest);
  };
}

function installStoreUserDataHook() {
  if (api._c2cVaultStoreHooked) return;
  const orig = api.storeUserData?.bind(api);
  if (!orig) return;
  api._c2cVaultStoreHooked = true;
  api.storeUserData = async function (path, data, opts) {
    if (!isWorkflowSavePath(path)) return orig(path, data, opts);

    const parsed = parseStoredWorkflowData(data);
    if (!parsed) return orig(path, data, opts);

    const marker = workflowVaultMarkerFromData(parsed, getWorkflowVaultMarker());
    if (!marker || !isComfyWorkflowObject(parsed)) return orig(path, data, opts);

    const file = await relockWorkflowForSave(parsed, marker);
    stampWorkflowVaultMarker(file);
    return orig(path, file, { overwrite: true, stringify: true });
  };
}

/** A workflow object that was opened from an encrypted file (it carries the public marker). */
function isMarkedWorkflow(o) {
  const m = o?.extra?.c2c_vault_workflow;
  return !!(o && typeof o === "object" && Array.isArray(o.nodes) && m?.vault_id && m?.payload);
}

/** Replace every marked workflow inside a stored value with its ciphertext wrapper, keeping the container: ComfyUI's
 *  draft store keeps several workflows under one key (`Comfy.Workflow.Drafts`) and in envelopes whose workflow is a
 *  JSON STRING, so the value is walked, never swapped whole (that would wipe the other drafts). */
function rewriteMarkedWorkflows(v, depth = 0) {
  if (depth > 6) return v;
  if (typeof v === "string") {
    if (!v.includes("c2c_vault_workflow")) return v;
    try {
      const parsed = JSON.parse(v);
      const out = rewriteMarkedWorkflows(parsed, depth + 1);
      return out === parsed ? v : JSON.stringify(out);
    } catch (_) { return v; }
  }
  if (!v || typeof v !== "object") return v;
  if (isMarkedWorkflow(v)) return buildWorkflowFileWrapper(v.extra.c2c_vault_workflow);
  let changed = false;
  const copy = Array.isArray(v) ? [] : {};
  for (const [k, val] of Object.entries(v)) {
    const nv = rewriteMarkedWorkflows(val, depth + 1);
    if (nv !== val) changed = true;
    copy[k] = nv;
  }
  return changed ? copy : v;
}

function installStoragePersistenceHook() {
  if (Storage.prototype._c2cVaultSetItemHooked) return;
  Storage.prototype._c2cVaultSetItemHooked = true;
  const orig = Storage.prototype.setItem;
  Storage.prototype.setItem = function (key, value) {
    // Tab restore and drafts keep the open workflow in browser storage: an encrypted workflow is kept there as its
    // ciphertext wrapper, so a reload asks for the password instead of restoring plaintext.
    if (typeof value === "string" && value.includes("c2c_vault_workflow")) {
      let out = value;
      try { out = rewriteMarkedWorkflows(value); } catch (_) { out = null; }
      if (out === null || (out === value && /"nodes"\s*:/.test(value))) return;   // never write a marked plaintext workflow
      return orig.call(this, key, out);
    }
    return orig.call(this, key, value);
  };
}

function installExportGuard() {
  if (window._c2cVaultExportGuard) return;
  window._c2cVaultExportGuard = true;

  const refuse = async () => {
    await modalAlert(
      "This workflow is encrypted",
      "Use C2C Vault: Save workflow encrypted. Exporting would write it as plain JSON.",
    );
  };
  const wrapOne = (cmd) => {
    if (!cmd || cmd._c2cVaultExportWrapped || typeof cmd.function !== "function") return;
    const orig = cmd.function;
    cmd.function = async (...args) => (getWorkflowVaultMarker() ? refuse() : orig.apply(cmd, args));
    cmd._c2cVaultExportWrapped = true;
  };

  let tries = 0;
  const tick = () => {
    const mgr = app.extensionManager?.command;
    if (!mgr) {
      if (tries++ < 80) setTimeout(tick, 250);
      return;
    }
    // The command store's execute() is what the menu, keybindings and the palette call (frontend 1.52.7); its
    // `commands` is an ARRAY of the command objects, so the old Map/object lookup never found Export at all.
    if (typeof mgr.execute === "function" && !mgr._c2cVaultExecWrapped) {
      const origExec = mgr.execute;
      mgr.execute = function (id, ...rest) {
        if (EXPORT_CMD_IDS.includes(id) && getWorkflowVaultMarker()) return refuse();
        return origExec.call(this, id, ...rest);
      };
      mgr._c2cVaultExecWrapped = true;
    }
    const list = mgr.commands;
    const all = Array.isArray(list) ? list : (list instanceof Map ? [...list.values()] : Object.values(list || {}));
    for (const cmd of all) if (EXPORT_CMD_IDS.includes(cmd?.id)) wrapOne(cmd);   // a caller that runs .function()
  };
  tick();
}

function _vaultCanvasMenuItems() {
  if (vaultScope() === SCOPE_WHOLE) {
    return [
      {
        content: "C2C Vault → Save workflow encrypted…",
        callback: () => saveWorkflowEncrypted(),
      },
      {
        content: "C2C Vault → Seal whole workflow",
        callback: () => sealWholeWorkflow(),
      },
    ];
  }
  return [
    {
      content: "C2C Vault → Lock selection (password to run)",
      callback: () => lockSelection(false),
    },
    {
      content: "C2C Vault → Seal selection (runs without password)",
      callback: () => lockSelection(true),
    },
  ];
}

function _mergeVaultCanvasMenuItems(opts) {
  opts.push(..._vaultCanvasMenuItems());
  return opts;
}

/** The vault's access row ON the node (owner A9: "there is no option to ask for password ... vault id? tf is
 *  that?"). Locked: status + password field + Unlock, then "Lock again". Sealed: status + password field + "Open for
 *  editing". The session state comes from the server (/c2c_vault/status), so it is right after a restart or expiry. */
function buildAccessWidget(node, isSealed) {
  const host = document.createElement("div");
  css(host, { display: "flex", flexDirection: "column", gap: "4px", padding: "2px 2px 4px",
    fontSize: "11px", color: "var(--c2c-fg)", boxSizing: "border-box", width: "100%" });
  const status = document.createElement("div");
  css(status, { opacity: "0.9", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" });
  const row = document.createElement("div");
  css(row, { display: "flex", gap: "4px", alignItems: "center" });
  const pw = document.createElement("input");
  pw.type = "password";
  pw.autocomplete = "off";
  pw.spellcheck = false;
  pw.placeholder = isSealed ? "Password to open for editing" : "Password";
  pw.setAttribute("aria-label", isSealed ? "Vault password (opens the vault for editing)" : "Vault password");
  css(pw, { flex: "1", minWidth: "0", padding: "3px 6px", borderRadius: "4px", font: "inherit",
    background: "var(--c2c-surface0)", color: "var(--c2c-fg)", border: "1px solid var(--c2c-border)" });
  const go = document.createElement("button");
  go.type = "button";
  go.textContent = isSealed ? "Open for editing" : "Unlock";
  const relock = document.createElement("button");
  relock.type = "button";
  relock.textContent = "Lock again";
  relock.title = "Forget this vault's key now; the password is needed again to run it.";
  for (const b of [go, relock]) css(b, { padding: "3px 8px", cursor: "pointer", font: "inherit", whiteSpace: "nowrap" });
  const err = document.createElement("div");
  css(err, { color: "var(--c2c-dangerStrong)", minHeight: "0", whiteSpace: "normal" });
  row.append(pw, go, relock);
  for (const el of [status, row, err]) el.style.flexShrink = "0";
  host.append(status, row, err);

  const vaultId = () => widget(node, "vault_id")?.value || "";
  const hasPayload = () => !!(widget(node, "vault_payload")?.value || "").trim();
  let open = false;
  const fit = () => {           // the error line adds a row: grow the node instead of spilling out of it
    const sz = node.computeSize?.();
    if (sz) node.setSize([Math.max(node.size?.[0] || 0, sz[0]), sz[1]]);
    node.setDirtyCanvas?.(true, true);
  };
  const showError = (msg) => { err.textContent = msg; fit(); };
  const retitle = () => {
    if (isSealed) return;
    const badge = open ? "🔓 UNLOCKED" : "🔒 LOCKED";
    node.title = `${badge}  C2C Vault — Locked`;
  };
  const render = () => {
    const hadError = !!err.textContent;
    err.textContent = "";
    if (hadError) fit();
    retitle();
    if (!hasPayload()) {
      status.textContent = "Empty vault: select nodes, right-click the canvas, C2C Vault → Lock or Seal selection.";
      pw.style.display = go.style.display = relock.style.display = "none";
      return;
    }
    if (isSealed) {
      status.textContent = "Sealed · runs without a password";
      relock.style.display = "none";
      pw.style.display = go.style.display = "";
      return;
    }
    status.textContent = open ? "Unlocked for this ComfyUI session" : "Locked · type the password to run it";
    pw.style.display = go.style.display = open ? "none" : "";
    relock.style.display = open ? "" : "none";
  };
  const refresh = async () => {
    if (!isSealed && hasPayload()) {
      const { ok, data } = await post("/c2c_vault/status", { vault_id: vaultId() });
      open = !!(ok && data.open);
      if (open) unlockedSessions.add(vaultId()); else unlockedSessions.delete(vaultId());
    }
    render();
  };
  const submit = async () => {
    const password = pw.value;
    if (!password) { showError("Type the password first."); shakeEl(host); pw.focus(); return; }
    go.disabled = true;
    try {
      const route = isSealed ? "/c2c_vault/open" : "/c2c_vault/unlock";
      const { ok, data, status } = await post(route, { vault_id: vaultId(), password,
        payload: widget(node, "vault_payload")?.value || "" });
      if (!ok) {
        // 403 = wrong password or modified payload (deliberately one message); 429 = locked out for a while
        showError(status === 429 ? (data.error || "Too many attempts. Wait a few minutes.")
          : "Wrong password (or the vault data was changed).");
        shakeEl(host); pw.select(); return;
      }
      pw.value = "";
      if (isSealed) openVaultOverlay(node, data.subgraph, true);
      else { open = true; unlockedSessions.add(vaultId()); }
      applyVaultInterface(node, parseInterface(widget(node, "vault_interface")?.value), isSealed);
      render();
    } finally {
      go.disabled = false;
    }
  };
  go.addEventListener("click", submit);
  pw.addEventListener("keydown", (e) => {
    e.stopPropagation();               // canvas shortcuts must not fire while typing a password
    if (e.key === "Enter") { e.preventDefault(); submit(); }
  });
  relock.addEventListener("click", async () => {
    await post("/c2c_vault/lock_session", { vault_id: vaultId() });
    open = false;
    unlockedSessions.delete(vaultId());
    applyVaultInterface(node, parseInterface(widget(node, "vault_interface")?.value), isSealed);
    render();
  });
  render();
  node._vaultAccess = { refresh, focus: () => pw.focus() };
  node.addDOMWidget("vault_access", "vault_access", host, {
    serialize: false,
    margin: 2,
    getValue: () => "",
    setValue: () => {},
    getMinHeight: () => (err.textContent ? 70 : 52),
    getHeight: () => (err.textContent ? 70 : 52),
  });
}

app.registerExtension({
  name: "Code2Collapse.CustomNodePacks.Vault",

  // Declarative: frontend 1.52 has no extensionManager.registerCommand (the call was silently skipped).
  commands: [
    { id: "c2c.vault.saveWorkflowEncrypted", label: "C2C Vault: Save workflow encrypted", function: () => saveWorkflowEncrypted() },
    { id: "c2c.vault.sealWholeWorkflow", label: "C2C Vault: Seal whole workflow", function: () => sealWholeWorkflow() },
  ],

  settings: [{
    id: SETTING_SCOPE,
    name: "Vault scope",
    tooltip: "Selected nodes: lock/seal a canvas selection into a vault node.\n"
      + "Whole workflow: encrypt the entire saved file, or seal every node into one vault.",
    type: "combo",
    options: [SCOPE_SELECTED, SCOPE_WHOLE],
    defaultValue: SCOPE_SELECTED,
    category: ["c2c", "Vault", "Scope"],
  }],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData.name)) return;
    const isSealed = nodeData.name === "C2C_VaultSealed";

    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const node = this;

      // Internal: the ciphertext and the public manifest travel in the workflow but are not for editing. Hidden in
      // both renderers (the old `inputEl.hidden` only worked on the classic canvas and is deprecated, L7.35).
      // vault_id is a public handle the UI matches sessions with; it means nothing to a person (A9), so it is
      // hidden with the other two.
      for (const wn of ["vault_id", "vault_payload", "vault_interface"]) setHidden(widget(node, wn), true);

      const host = document.createElement("div");
      css(host, {
        width: "100%", padding: "4px 0", fontSize: "11px",
        opacity: "0.8", textAlign: "center",
        color: "var(--c2c-fg)",
      });
      node._vaultSummaryEl = host;
      host.textContent = "0 nodes · 0 inputs · 0 outputs · 0 params";
      // margin 2: ComfyUI draws a DOM element at computedHeight - 2*margin, and the default margin of 10 crushed this
      // row to 10px and the access row below to 40 (its status line shrank to 2px).
      node.addDOMWidget("vault_summary", "summary", host, { serialize: false, margin: 2, getMinHeight: () => 18, getHeight: () => 18 });

      buildAccessWidget(node, isSealed);

      setTimeout(() => {
        // Also for an EMPTY vault: its 18 wire sockets and 8 outputs mean nothing until a selection is locked
        // into it. A saved vault restores its own sockets in configure().
        applyVaultInterface(node, parseInterface(widget(node, "vault_interface")?.value), isSealed);
        node._vaultAccess?.refresh();
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
      this._vaultAccess?.refresh();
      return r;
    };

    const dbl = nodeType.prototype.onDblClick;
    nodeType.prototype.onDblClick = function (...args) {
      const r = dbl?.apply(this, arguments);
      openOnDoubleClick(this, isSealed);
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

  getCanvasMenuItems() {
    return _vaultCanvasMenuItems();
  },

  setup() {
    installSaveGuard();
    installLoadGraphDataHook();
    installStoreUserDataHook();
    installStoragePersistenceHook();
    installExportGuard();
    legacyCanvasMenu("vault", _mergeVaultCanvasMenuItems);
  },
});
