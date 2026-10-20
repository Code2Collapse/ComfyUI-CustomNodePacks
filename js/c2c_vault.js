import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { legacyCanvasMenu } from "./_c2c_compat.js";
import { getRuntime } from "./_c2c_runtime.js";
import { asOneUndoStep } from "./_c2c_undo_scope.js";
import { setHidden } from "./_widget_visibility.js";
import { flattenSelection, isSubgraphNode, isVirtualNode, linkById } from "./_vault_flatten.js";

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
const EDIT_GUARD_CMD_IDS = [...EXPORT_CMD_IDS, "Comfy.SaveWorkflow", "Comfy.SaveWorkflowAs", "Comfy.DuplicateWorkflow",
  "Comfy.QueuePrompt", "Comfy.QueuePromptFront"];

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

function modalPasswordStep({ title, note, sealed, confirmLabel }) {
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
    next.textContent = confirmLabel || "Next →";
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
  if (isEmptyVault(node)) ensureCaptureSocket(node); else removeCaptureSocket(node);

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

// ─────────────────────────── editing a vault (A17: password + second factor -> the REAL graph) ───────────────────
//
// Open for editing = password (server decrypts, keeps the plaintext) -> authenticator code (only now the graph comes
// back) -> the inner nodes in their own workflow tab, the vault's inputs / outputs as placeholder nodes. Nothing of
// that tab is ever persisted: drafts and the node clipboard are kept out of browser storage, Save / Export / Queue are
// refused there. Lock & close rebuilds the vault from the tab, the server re-encrypts with fresh keys material and
// reads it back before the vault node's payload is replaced. A reload or crash loses only the unlocked edits; the
// workflow keeps its last locked payload.

const EDIT_MARK = "c2c_vault_edit";
const EDIT_IN = "C2C_VaultEditInput";
const EDIT_OUT = "C2C_VaultEditOutput";
let editClipboard = null;            // node copies made inside an edit tab: memory only

function activeEditSession() {
  const root = app.rootGraph || app.graph;
  const mark = root?.extra?.[EDIT_MARK];
  return mark ? vaultEditSessions.get(mark.vault_id) || null : null;
}

function modalText({ title, note, placeholder, confirmLabel, mono }) {
  if (headless()) return Promise.resolve(null);
  return new Promise((resolve) => {
    const back = document.createElement("div");
    back.setAttribute("role", "dialog");
    back.setAttribute("aria-modal", "true");
    css(back, { position: "fixed", inset: "0", zIndex: "10000", background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center" });
    const box = document.createElement("div");
    css(box, { background: "var(--c2c-bg2)", color: "var(--c2c-fg)", border: "1px solid var(--c2c-border)",
      borderRadius: "6px", padding: "18px 20px", minWidth: "340px", maxWidth: "560px", font: "13px sans-serif" });
    const input = document.createElement("input");
    input.type = "text";
    input.autocomplete = "one-time-code";
    input.inputMode = mono ? "numeric" : "text";
    input.placeholder = placeholder || "";
    css(input, { width: "100%", boxSizing: "border-box", margin: "10px 0", padding: "7px 9px", borderRadius: "4px",
      background: "var(--c2c-surface0)", color: "var(--c2c-fg)", border: "1px solid var(--c2c-border)",
      fontFamily: mono ? "ui-monospace, monospace" : "inherit", letterSpacing: mono ? "0.2em" : "normal" });
    const row = document.createElement("div");
    css(row, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const ok = document.createElement("button");
    ok.textContent = confirmLabel || "OK";
    for (const b of [cancel, ok]) css(b, { padding: "6px 14px", cursor: "pointer" });
    const done = (v) => { back.remove(); resolve(v); };
    cancel.onclick = () => done(null);
    ok.onclick = () => done(input.value.trim());
    input.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") ok.onclick(); });
    box.append(
      Object.assign(document.createElement("div"), { textContent: title, style: { fontWeight: "600" } }),
      Object.assign(document.createElement("div"), { textContent: note, style: { opacity: "0.8", fontSize: "12px",
        marginTop: "6px", lineHeight: "1.45", whiteSpace: "pre-wrap" } }),
      input, row,
    );
    row.append(cancel, ok);
    back.append(box);
    document.body.append(back);
    input.focus();
  });
}

function modalSecretSheet({ title, lines, confirmLabel }) {
  if (headless()) return Promise.resolve();
  return new Promise((resolve) => {
    const back = document.createElement("div");
    back.setAttribute("role", "dialog");
    css(back, { position: "fixed", inset: "0", zIndex: "10000", background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center" });
    const box = document.createElement("div");
    css(box, { background: "var(--c2c-bg2)", color: "var(--c2c-fg)", border: "1px solid var(--c2c-border)",
      borderRadius: "6px", padding: "18px 20px", width: "min(560px, 92vw)", font: "13px sans-serif" });
    box.append(Object.assign(document.createElement("div"), { textContent: title,
      style: { fontWeight: "600", marginBottom: "8px" } }));
    for (const [label, value, copyable] of lines) {
      const l = document.createElement("div");
      l.textContent = label;
      css(l, { opacity: "0.8", fontSize: "12px", marginTop: "8px" });
      const v = document.createElement("div");
      v.textContent = value;
      css(v, { fontFamily: "ui-monospace, monospace", fontSize: "12px", wordBreak: "break-all", whiteSpace: "pre-wrap",
        background: "var(--c2c-surface0)", padding: "6px 8px", borderRadius: "4px", marginTop: "4px" });
      box.append(l, v);
      if (copyable) {
        const c = document.createElement("button");
        c.textContent = "Copy";
        css(c, { marginTop: "4px", padding: "3px 10px", cursor: "pointer" });
        c.onclick = async () => { try { await navigator.clipboard.writeText(value); c.textContent = "Copied"; } catch (_) {} };
        box.append(c);
      }
    }
    const ok = document.createElement("button");
    ok.textContent = confirmLabel || "Done";
    css(ok, { marginTop: "14px", padding: "6px 14px", cursor: "pointer", float: "right" });
    ok.onclick = () => { back.remove(); resolve(); };
    box.append(ok);
    back.append(box);
    document.body.append(back);
  });
}

/** First use: set up the authenticator for THIS install (per person, per install - owner D0.25). */
async function ensureSecondFactor() {
  const s = await api.fetchApi("/c2c_vault/2fa/status").then((r) => r.json()).catch(() => ({}));
  if (s.enrolled) return true;
  const { ok, data } = await post("/c2c_vault/2fa/enrol/begin", {});
  if (!ok) { await modalAlert("Second factor", data.error || "Could not start the setup."); return false; }
  await modalSecretSheet({
    title: "Set up your authenticator for vault editing (this install only)",
    lines: [
      ["Add this key to an authenticator app (manual entry, time-based, 6 digits):", data.secret_base32, true],
      ["Or this setup link:", data.otpauth_uri, true],
      ["Recovery codes - each works once instead of a code. Keep them somewhere safe; they are not shown again:",
        data.recovery_codes.join("\n"), true],
    ],
    confirmLabel: "I have saved them",
  });
  for (;;) {
    const code = await modalText({ title: "Confirm the setup", note: "Type the 6-digit code your app shows now.",
      placeholder: "123456", confirmLabel: "Confirm", mono: true });
    if (code === null) return false;
    const r = await post("/c2c_vault/2fa/enrol/confirm", { code });
    if (r.ok) return true;
    await modalAlert("Not confirmed", r.data.error || "That code is not right.");
  }
}

/** Password -> code -> the real graph in an edit tab. */
async function beginEdit(node, isSealed, typedPassword = null) {
  if (headless()) return;
  const vaultId = widget(node, "vault_id")?.value || "";
  if (vaultEditSessions.has(vaultId)) {
    await modalAlert("Already open", "This vault is already open in an edit tab.");
    return;
  }
  const payload = widget(node, "vault_payload")?.value || "";
  const password = typedPassword || await modalPassword({ title: "Open the vault for editing",
    note: "Step 1 of 2: the vault's password. Step 2: the code from your authenticator.", confirmLabel: "Next" });
  if (!password) return;
  let begin = await post("/c2c_vault/edit/begin", { vault_id: vaultId, payload, password });
  if (!begin.ok && begin.data.needs_enrolment) {
    if (!(await ensureSecondFactor())) return;
    begin = await post("/c2c_vault/edit/begin", { vault_id: vaultId, payload, password });
  }
  if (!begin.ok) {
    await modalAlert("Could not open the vault", begin.data.error || "Wrong password, or the vault data was changed.");
    return;
  }
  for (;;) {
    const code = await modalText({ title: "Open the vault for editing",
      note: "Step 2 of 2: the 6-digit code from your authenticator (or a recovery code).",
      placeholder: "123456", confirmLabel: "Open", mono: true });
    if (code === null) return;
    const v = await post("/c2c_vault/edit/verify", { edit_id: begin.data.edit_id, code });
    if (v.ok) { await openEditTab(node, v.data); return; }
    await modalAlert("Not opened", v.data.error || "That code is not right.");
    if (/expired/i.test(v.data.error || "")) return;
  }
}

function graphPathOf(node) {
  const g = node.graph;
  const root = app.rootGraph || app.graph;
  return { graphId: g && g !== root ? g.id : null, nodeId: node.id };
}

function findVaultNode(path) {
  const root = app.rootGraph || app.graph;
  let g = root;
  if (path.graphId) g = root.subgraphs?.get?.(path.graphId) || null;
  return g?.getNodeById?.(path.nodeId) || null;
}

/** The edit tab's graph: the layout the vault was locked with (or one laid out from the graph), no outside links. */
function editGraphData(result, vaultId) {
  const lay = result.layout;
  const data = { last_node_id: 0, last_link_id: 0, nodes: [], links: [], groups: [], config: {},
    extra: { [EDIT_MARK]: { vault_id: vaultId } }, version: 0.4 };
  if (!lay || !Array.isArray(lay.nodes)) return data;
  const known = new Set((lay.links || []).map((l) => l[0]));
  data.nodes = lay.nodes.map((n) => {
    const c = JSON.parse(JSON.stringify(n));
    for (const inp of c.inputs || []) if (inp.link != null && !known.has(inp.link)) inp.link = null;
    for (const out of c.outputs || []) out.links = (out.links || []).filter((id) => known.has(id));
    return c;
  });
  data.links = JSON.parse(JSON.stringify(lay.links || []));
  data.last_node_id = Math.max(0, ...data.nodes.map((n) => Number(n.id) || 0));
  data.last_link_id = Math.max(0, ...data.links.map((l) => Number(l[0]) || 0));
  return data;
}

/** No layout (an old vault, or one made from subgraphs): the nodes are created from the graph and laid out in
 *  columns by dependency. Returns {execId -> node}. */
function buildFromExec(graph, ex) {
  const byExec = new Map();
  const depth = new Map();
  const incoming = new Map();
  for (const l of ex.links || []) incoming.set(String(l.to), [...(incoming.get(String(l.to)) || []), String(l.from)]);
  const d = (id, seen = new Set()) => {
    if (depth.has(id)) return depth.get(id);
    if (seen.has(id)) return 0;
    seen.add(id);
    const v = Math.max(-1, ...(incoming.get(id) || []).map((s) => d(s, seen))) + 1;
    depth.set(id, v);
    return v;
  };
  const rows = new Map();
  for (const spec of ex.nodes || []) {
    const n = LiteGraph.createNode(spec.class_type);
    if (!n) throw new Error(`This vault needs a node type that is not installed: ${spec.class_type}`);
    const col = d(String(spec.id));
    const row = rows.get(col) || 0;
    rows.set(col, row + 1);
    n.pos = [80 + col * 380, 80 + row * 260];
    graph.add(n);
    for (const [k, val] of Object.entries(spec.widgets || {})) {
      const w = (n.widgets || []).find((x) => x.name === k);
      if (w) w.value = val;
    }
    byExec.set(String(spec.id), n);
  }
  for (const l of ex.links || []) {
    const a = byExec.get(String(l.from));
    const b = byExec.get(String(l.to));
    if (!a || !b) continue;
    const slot = l.to_name ? (b.inputs || []).findIndex((i) => i.name === l.to_name) : Number(l.to_slot);
    if (slot >= 0) a.connect(Number(l.from_slot), b, slot);
  }
  return byExec;
}

async function openEditTab(node, result) {
  const ws = app.extensionManager?.workflow;
  if (!ws?.createTemporary) {
    await modalAlert("Cannot open", "This ComfyUI front end has no workflow tabs API (needs frontend 1.53+).");
    await post("/c2c_vault/edit/cancel", { edit_token: result.edit_token });
    return;
  }
  const vaultId = widget(node, "vault_id")?.value || "";
  const iface = parseInterface(widget(node, "vault_interface")?.value);
  const outer = ws.activeWorkflow || null;
  const session = { vaultId, token: result.edit_token, result, iface, isSealed: node.type === "C2C_VaultSealed",
    path: graphPathOf(node), outer, tmp: null, idMap: null, bar: null, timer: null };
  vaultEditSessions.set(vaultId, session);
  const label = String(vaultId || "vault").replace(/[^\w.-]/g, "").slice(0, 40);   // the id, not the status badge
  const data = editGraphData(result, vaultId);
  session.tmp = ws.createTemporary(`Vault edit - ${label}.json`, data);
  await app.loadGraphData(data, true, true, session.tmp);
  const graph = app.rootGraph || app.graph;
  const ex = result.exec;
  let byExec;
  if (!result.layout || !Array.isArray(result.layout.nodes)) {
    byExec = buildFromExec(graph, ex);
    session.idMap = new Map([...byExec.entries()].map(([k, n]) => [k, String(n.id)]));
  } else {
    byExec = new Map((graph._nodes || []).map((n) => [String(n.id), n]));
  }
  // the vault's inputs / outputs as placeholders, wired exactly as the vault wires them
  const xs = [...byExec.values()].map((n) => n.pos[0]);
  const minX = Math.min(80, ...xs) - 320;
  const maxX = Math.max(80, ...xs.map((x, i) => x + ([...byExec.values()][i].size?.[0] || 200))) + 80;
  (ex.boundary_in || []).forEach((b, i) => {
    const target = byExec.get(String(b.to));
    const p = LiteGraph.createNode(EDIT_IN);
    p.title = `Vault input ${i}: ${iface.in?.[i]?.name || b.name}`;
    p.pos = [minX, 80 + i * 110];
    graph.add(p);
    const w = (p.widgets || []).find((x) => x.name === "index");
    if (w) w.value = i;
    if (p.outputs?.[0]) { p.outputs[0].label = iface.in?.[i]?.name || b.name; }
    if (target) {
      const slot = b.to_name ? (target.inputs || []).findIndex((x) => x.name === b.to_name) : Number(b.to_slot);
      if (slot >= 0) p.connect(0, target, slot);
    }
  });
  (ex.boundary_out || []).forEach((b, j) => {
    const source = byExec.get(String(b.from));
    const p = LiteGraph.createNode(EDIT_OUT);
    p.title = `Vault output ${j}: ${iface.out?.[j]?.name || b.name}`;
    p.pos = [maxX, 80 + j * 110];
    graph.add(p);
    const w = (p.widgets || []).find((x) => x.name === "index");
    if (w) w.value = j;
    if (source) source.connect(Number(b.from_slot), p, 0);
  });
  graph.setDirtyCanvas?.(true, true);
  showEditBar(session);
}

function isEditTabActive(session) {
  // the workflow store hands out reactive proxies: compare paths, never objects
  const active = app.extensionManager?.workflow?.activeWorkflow;
  return !!(active && session.tmp && active.path === session.tmp.path);
}

function showEditBar(session) {
  const bar = document.createElement("div");
  bar.setAttribute("role", "toolbar");
  css(bar, { position: "fixed", top: "118px", left: "50%", transform: "translateX(-50%)", zIndex: "9000",
    display: "flex", gap: "8px", alignItems: "center", padding: "6px 10px", borderRadius: "6px",
    background: "var(--c2c-bg2)", color: "var(--c2c-fg)", border: "1px solid var(--c2c-border)",
    font: "12px sans-serif", boxShadow: "0 6px 20px rgba(0,0,0,0.45)" });
  const text = document.createElement("span");
  text.textContent = "🔓 Vault edit view - nothing here is saved until you lock it";
  const lockBtn = document.createElement("button");
  lockBtn.textContent = "Lock & close";
  const discard = document.createElement("button");
  discard.textContent = "Discard edits";
  const pwBtn = document.createElement("button");
  pwBtn.textContent = "Change password…";
  pwBtn.title = "Lock & close then encrypts the vault under a new password; the old one stops opening it.";
  for (const b of [lockBtn, discard, pwBtn]) css(b, { padding: "4px 10px", cursor: "pointer" });
  bar.append(text, lockBtn, discard, pwBtn);
  pwBtn.onclick = async () => {
    const np = await modalPasswordStep({ title: "New vault password", confirmLabel: "Use this password",
      note: "It applies when you press Lock & close; until then nothing changes. The old password then stops "
        + "opening this vault. There is no password recovery." });
    if (np === null) return;
    session.newPassword = np;                    // in memory only, sent once with Lock & close
    text.textContent = "🔓 Vault edit view - the new password applies when you lock it";
  };
  document.body.append(bar);
  session.bar = bar;
  lockBtn.onclick = async () => { lockBtn.disabled = true; try { await commitEdit(session); } finally { lockBtn.disabled = false; } };
  discard.onclick = () => discardEdit(session);
  // only visible while the edit tab is the active one (not ambient: it must stay right while a queue runs)
  session.timer = getRuntime().every(`vault.editbar.${session.vaultId}`, 400, () => {
    bar.style.display = isEditTabActive(session) ? "flex" : "none";
  }, { ambient: false });
}

function closeEditSession(session) {
  session.timer?.cancel();
  session.newPassword = null;
  session.bar?.remove();
  vaultEditSessions.delete(session.vaultId);
  if (!vaultEditSessions.size) editClipboard = null;
}

/** The edit tab -> the vault's graph again. The interface stays fixed: every input placeholder feeds exactly the
 *  one node input it fed, every output placeholder is fed by one output. */
function execFromEditGraph(session) {
  const graph = app.rootGraph || app.graph;
  const all = graph._nodes || [];
  const ph = (n) => n.type === EDIT_IN || n.type === EDIT_OUT;
  const inner = all.filter((n) => !ph(n));
  const idx = (n) => Number((n.widgets || []).find((w) => w.name === "index")?.value);
  const placeholders = new Map(all.filter(ph).map((n) => [String(n.id), n]));
  const derived = flattenSelection(inner, graph);
  if (derived.problems.length) throw new Error(derived.problems.join("\n"));
  const orig = session.result.exec;
  const nIn = (orig.boundary_in || []).length;
  const nOut = (orig.boundary_out || []).length;
  const bIn = new Array(nIn).fill(null);
  derived.boundary_in.forEach((b, k) => {
    const src = derived.externalSources[k];
    const p = src && placeholders.get(String(src.id));
    if (!p || p.type !== EDIT_IN) throw new Error(`An input inside the vault (${b.to_name}) is wired from outside the `
      + "vault's inputs. Connect it to a node inside, or to a Vault input placeholder.");
    const i = idx(p);
    if (bIn[i]) throw new Error(`Vault input ${i} feeds more than one node input. A vault input feeds exactly one: `
      + "the vault's interface cannot change while it is edited.");
    bIn[i] = { name: orig.boundary_in[i].name, to: b.to, to_slot: b.to_slot, to_name: b.to_name };
  });
  const missingIn = bIn.findIndex((b) => !b);
  if (missingIn >= 0) throw new Error(`Vault input ${missingIn} (${session.iface.in?.[missingIn]?.name || ""}) is not `
    + "connected to anything inside. Connect it: the vault's interface cannot change while it is edited.");
  const bOut = new Array(nOut).fill(null);
  derived.boundary_out.forEach((b, k) => {
    for (const t of derived.externalTargets[k] || []) {
      const p = t && placeholders.get(String(t.id));
      if (!p || p.type !== EDIT_OUT) continue;
      const j = idx(p);
      if (bOut[j]) throw new Error(`Vault output ${j} is fed twice.`);
      bOut[j] = { name: orig.boundary_out[j].name, from: b.from, from_slot: b.from_slot };
    }
  });
  const missingOut = bOut.findIndex((b) => !b);
  if (missingOut >= 0) throw new Error(`Vault output ${missingOut} (${session.iface.out?.[missingOut]?.name || ""}) `
    + "is not fed by any node. Connect it: the vault's interface cannot change while it is edited.");
  let promoted = JSON.parse(JSON.stringify(orig.promoted || []));
  if (session.idMap) {
    promoted = promoted.map((p) => (p && p.node != null && session.idMap.has(String(p.node))
      ? { ...p, node: session.idMap.get(String(p.node)) } : p));
  }
  const ids = new Set(derived.nodes.map((n) => String(n.id)));
  for (const p of promoted) {
    if (p && p.node != null && !ids.has(String(p.node))) {
      throw new Error("A promoted parameter's node was deleted. Put it back, or lock the vault again from scratch "
        + "to change its parameters.");
    }
  }
  return {
    exec: { nodes: derived.nodes, links: derived.links, boundary_in: bIn, boundary_out: bOut, promoted },
    layout: selectionLayout(inner, graph),
  };
}

async function backToOuter(session) {
  const ws = app.extensionManager?.workflow;
  if (session.outer && ws) await app.loadGraphData(session.outer.activeState, true, true, session.outer);
  try { if (ws && session.tmp) await ws.closeWorkflow(session.tmp); } catch (_) { /* already closed */ }
}

async function commitEdit(session) {
  if (!isEditTabActive(session)) {
    await modalAlert("Switch to the edit tab", "Lock the vault from its edit tab.");
    return;
  }
  let built;
  try { built = execFromEditGraph(session); } catch (e) { await modalAlert("Cannot lock yet", String(e.message || e)); return; }
  const { ok, data } = await post("/c2c_vault/edit/commit", { edit_token: session.token, exec: built.exec,
    layout: built.layout, interface: session.iface, new_password: session.newPassword || undefined });
  if (!ok) { await modalAlert("Not locked", (data.error || "The vault could not be locked.") + " The edit tab stays open; the vault keeps its last locked version."); return; }
  await backToOuter(session);
  const vault = findVaultNode(session.path);
  if (!vault) {
    await modalAlert("Locked, but the vault node is gone", "The new encrypted payload could not be put back: the "
      + "vault node was deleted from the workflow while it was being edited.");
    closeEditSession(session);
    return;
  }
  const w = widget(vault, "vault_payload");
  if (w) w.value = data.payload;
  vault.graph?.setDirtyCanvas?.(true, true);
  try { app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.(); } catch (_) {}
  if (!session.isSealed) unlockedSessions.add(session.vaultId);
  vault._vaultAccess?.refresh();
  closeEditSession(session);
  await modalAlert("Locked", "The vault's new contents are encrypted with fresh keys and were read back before "
    + "the old payload was replaced. Save the workflow to keep them.");
}

async function discardEdit(session) {
  await post("/c2c_vault/edit/cancel", { edit_token: session.token });
  await backToOuter(session);
  closeEditSession(session);
}

/** C2C Vault menu: this install's public key, for authors who seal vaults for it. */
async function showInstallKey() {
  const r = await api.fetchApi("/c2c_vault/keys/public").then((x) => x.json()).catch(() => ({}));
  if (!r.ok) { await modalAlert("Install key", r.error || "Could not read this install's key."); return; }
  await modalSecretSheet({ title: "This install's vault key (public - safe to share)", lines: [
    ["Send this to a vault's author: they seal the vault for this install, and it then runs here without a password.",
      r.public_key, true],
    ["Key id", r.key_id, false],
  ] });
}

async function manageSecondFactor() {
  const s = await api.fetchApi("/c2c_vault/2fa/status").then((r) => r.json()).catch(() => ({}));
  if (!s.enrolled) { if (await ensureSecondFactor()) await modalAlert("Second factor", "Set up on this install."); return; }
  const code = await modalText({ title: "Reset the second factor",
    note: `Set up on this install (${s.recovery_codes_left} recovery codes left). To set it up again, type a current `
      + "code or a recovery code.", placeholder: "123456", confirmLabel: "Reset", mono: true });
  if (code === null) return;
  const r = await post("/c2c_vault/2fa/reset", { code });
  await modalAlert("Second factor", r.ok ? "Removed. Set it up again the next time you open a vault for editing."
    : (r.data.error || "That code is not right."));
}

/** Seal: who may run it without the password. */
function modalSealTarget() {
  if (headless()) return Promise.resolve({ mode: "sealed", recipients: [] });
  return new Promise((resolve) => {
    const back = document.createElement("div");
    back.setAttribute("role", "dialog");
    css(back, { position: "fixed", inset: "0", zIndex: "10000", background: "rgba(0,0,0,0.55)", display: "flex",
      alignItems: "center", justifyContent: "center" });
    const box = document.createElement("div");
    css(box, { background: "var(--c2c-bg2)", color: "var(--c2c-fg)", border: "1px solid var(--c2c-border)",
      borderRadius: "6px", padding: "18px 20px", width: "min(560px, 92vw)", font: "13px sans-serif" });
    const mk = (value, label, note, checked) => {
      const l = document.createElement("label");
      css(l, { display: "block", margin: "8px 0", cursor: "pointer" });
      const r = Object.assign(document.createElement("input"), { type: "radio", name: "c2c-seal", value, checked });
      l.append(r, ` ${label}`);
      const n = document.createElement("div");
      n.textContent = note;
      css(n, { opacity: "0.75", fontSize: "12px", marginLeft: "20px", lineHeight: "1.4" });
      l.append(n);
      return l;
    };
    const keys = document.createElement("textarea");
    keys.placeholder = "Recipient install keys (c2cvk1:...), one per line - optional";
    keys.rows = 3;
    css(keys, { width: "100%", boxSizing: "border-box", fontFamily: "ui-monospace, monospace", fontSize: "11px",
      background: "var(--c2c-surface0)", color: "var(--c2c-fg)", border: "1px solid var(--c2c-border)" });
    keys.addEventListener("keydown", (e) => e.stopPropagation());
    const row = document.createElement("div");
    css(row, { display: "flex", gap: "8px", justifyContent: "flex-end", marginTop: "12px" });
    const cancel = Object.assign(document.createElement("button"), { textContent: "Cancel" });
    const ok = Object.assign(document.createElement("button"), { textContent: "Seal" });
    for (const b of [cancel, ok]) css(b, { padding: "6px 14px", cursor: "pointer" });
    cancel.onclick = () => { back.remove(); resolve(null); };
    ok.onclick = () => {
      const mode = box.querySelector("input[name=c2c-seal]:checked")?.value || "sealed";
      const recipients = keys.value.split(/\s+/).map((s) => s.trim()).filter(Boolean);
      back.remove();
      resolve({ mode, recipients: mode === "sealed" ? recipients : [] });
    };
    box.append(
      Object.assign(document.createElement("div"), { textContent: "Who may run it without the password?",
        style: { fontWeight: "600" } }),
      mk("sealed", "This install, and the installs whose keys I paste", "Each install has its own key pair. The "
        + "vault runs passwordless only there; elsewhere it needs the password. A recipient sends you their key "
        + "from C2C Vault -> This install's key.", true),
      keys,
      mk("sealed_portable", "Anywhere (portable - obfuscation only)", "Runs on any ComfyUI with this pack, like the "
        + "old sealed vaults. NOT secure: anyone with the pack can read it. It only keeps the graph out of the "
        + "workflow file.", false),
      row,
    );
    row.append(cancel, ok);
    back.append(box);
    document.body.append(back);
  });
}

let guardOpen = false;
/** One refusal dialog at a time: a keybinding and the command wrapper can both catch the same Ctrl+S. */
async function guardAlert() {
  if (guardOpen) return;
  guardOpen = true;
  try { await modalAlert("Not in the edit view", vaultSaveGuardMessage()); } finally { guardOpen = false; }
}

function vaultSaveGuardMessage() {
  return "This is a vault's edit view: it is never saved, exported or queued. Press 'Lock & close' (the vault's new "
    + "contents are encrypted into the vault node), then save or queue the workflow that holds the vault.";
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
      if (activeEditSession()) {
        await guardAlert();
        throw new Error("C2C Vault: the edit view is never queued or saved");
      }
      return origGTP(...args);
    };
  }

  document.addEventListener("keydown", (e) => {
    if (!activeEditSession()) return;
    if ((e.ctrlKey || e.metaKey) && (e.key === "s" || e.key === "S")) {
      e.preventDefault();
      e.stopPropagation();
      guardAlert();
    }
  }, true);
  window.addEventListener("beforeunload", (e) => {
    if (vaultEditSessions.size) { e.preventDefault(); e.returnValue = ""; }      // a reminder, not a safeguard
  });
}

async function handleUnlockOrOpen(node, isSealed, forEdit = false) {
  const id = widget(node, "vault_id")?.value || "";
  const payload = widget(node, "vault_payload")?.value || "";
  if (!payload) { await modalAlert("No payload", "This vault has no payload yet. Lock a selection first."); return; }

  if (forEdit || isSealed) { await beginEdit(node, isSealed); return; }
  const pw = await modalPassword({
    title: "Unlock vault",
    note: "Password is sent once to the server and never stored in the workflow.",
    confirmLabel: "Unlock",
  });
  if (pw === null) return;

  const route = "/c2c_vault/unlock";
  const { ok, data } = await post(route, { vault_id: id, password: pw, payload });

  if (!ok) {
    await modalAlert("Could not open vault", data.error || "Wrong password or modified payload.");
    return;
  }

  if (!isSealed && route === "/c2c_vault/unlock") unlockedSessions.add(id);

  await modalAlert("Vault unlocked", "Session unlocked for this ComfyUI process. Queue the workflow to run.");

  const iface = parseInterface(widget(node, "vault_interface")?.value);
  applyVaultInterface(node, iface, isSealed);
  node._vaultAccess?.refresh();
}

/** Double-click: a Locked vault whose session is open opens for editing at once (the key is already held);
 *  otherwise the password is asked first. */
async function openOnDoubleClick(node, isSealed) {
  const id = widget(node, "vault_id")?.value || "";
  // a run session is never enough to read a vault (A17): editing always asks for the password and the code
  if (isSealed || unlockedSessions.has(id)) return beginEdit(node, isSealed);
  return handleUnlockOrOpen(node, isSealed, false);
}

/** The real nodes of the current selection (groups, IO nodes and anything without an id are not lockable). */
function selectedLockableNodes() {
  const c = app.canvas;
  const out = new Map();
  const push = (n) => {
    if (n && n.id != null && Number(n.id) >= 0 && typeof n.serialize === "function") out.set(String(n.id), n);
  };
  try { c?.selectedItems?.forEach?.(push); } catch (_) { /* older frontends */ }
  Object.values(c?.selected_nodes || {}).forEach(push);
  return [...out.values()];
}

async function lockSelection(sealed) {
  return lockNodes(selectedLockableNodes().filter((n) => !NODES.includes(n.type)), sealed, null);
}

/** Lock / seal `sel` into a vault: a new node at the selection, or `into` (an EMPTY vault already on the canvas).
 *  Works on the root graph and inside subgraphs, and flattens subgraph nodes in the selection (L2.45 / L2.46). */
async function lockNodes(sel, sealed, into) {
  if (headless()) return;
  if (sel.length < 1) {
    await modalAlert("Nothing to lock", into
      ? "Select the nodes to protect, or connect the output you want to protect to this vault's 'capture' input, first."
      : "Select the nodes to lock first.");
    return;
  }
  // the graph the selection lives in: the subgraph the user is looking at, not the root
  const graph = sel[0].graph || app.canvas?.graph || app.graph;
  const derived = flattenSelection(sel, graph);
  if (derived.problems.length) {
    await modalAlert("Cannot lock this selection", derived.problems.join("\n"));
    return;
  }
  if (!derived.nodes.length) {
    await modalAlert("Nothing to lock", "The selection holds only front-end helpers (Reroute / Primitive).");
    return;
  }
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

  const count = derived.nodes.length;
  const pw = await modalPasswordStep({
    title: `${sealed ? "Seal" : "Lock"} ${count} node(s) into a vault`,
    note: (sealed
      ? "Sealed: recipient runs with no password but cannot read the graph. Your password is still required to edit — store it safely; there is no recovery."
      : "Locked: recipient needs this password to run at all. Store it safely — there is no recovery, by design.")
      + (derived.flattenedSubgraphs
        ? ` ${derived.flattenedSubgraphs} subgraph(s) in the selection are locked with their contents.` : ""),
    sealed,
  });
  if (pw === null) return;

  const confirmed = await modalBoundaryStep({ boundary: derived, nodeCount: count, sealed });
  if (!confirmed) return;

  // promotion addresses a node by its id in the payload: only plain nodes (a subgraph node is flattened away)
  const promotion = await modalPromoteStep({ selection: sel.filter((n) => !isSubgraphNode(n) && !isVirtualNode(n)) });
  if (promotion === null) return;

  const { boundary_in, boundary_out } = derived;
  const vault_id = `vault-${Math.random().toString(36).slice(2, 10)}`;
  // the public interface is built BEFORE locking: its structure (socket names / types, promoted ids / names /
  // types) is authenticated inside the ciphertext header, so the node can tell if it is changed later (A17)
  const iface = buildInterfaceManifest(
    sealed ? "sealed" : "locked", count,
    { boundary_in, boundary_out },
    derived.inTypes, derived.outTypes,
    promotion.params || [],
  );
  let sealTarget = { mode: sealed ? "sealed" : "locked", recipients: [] };
  if (sealed) {
    sealTarget = await modalSealTarget();
    if (sealTarget === null) return;
  }
  const { ok, data } = await post("/c2c_vault/lock", {
    vault_id, password: pw, mode: sealTarget.mode, recipients: sealTarget.recipients,
    subgraph: {
      nodes: derived.nodes, links: derived.links, boundary_in, boundary_out,
      promoted: promotion.promoted || [],
    },
    interface: iface,
    layout: selectionLayout(sel, graph),
  });
  if (!ok) {
    await modalAlert("Lock failed", data.error || "Could not lock the selection.");
    return;
  }
  if (!sealed && data.session_open) unlockedSessions.add(vault_id);

  const vault = into || LiteGraph.createNode(sealed ? "C2C_VaultSealed" : "C2C_VaultLocked");
  if (!into) vault.pos = selectionCentroid(sel);
  const lost = [];
  // one undo step: the vault node, its wiring and the removal of the originals
  asOneUndoStep(app.canvas, () => {
    if (!into) graph.add(vault);
    removeCaptureSocket(vault);
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
      const slot = (vault.inputs || []).findIndex((inp) => inp.name === `input_${i}`);
      if (slot < 0 || !connectSourceToVault(graph, src, vault, slot)) lost.push(`input ${boundary_in[i]?.name}`);
    });
    derived.externalTargets.forEach((targets, i) => {
      for (const tgt of targets || []) {
        if (!connectVaultToTarget(graph, tgt, vault, i)) lost.push(`output ${boundary_out[i]?.name}`);
      }
    });

    for (const n of sel) graph.remove(n);
  });
  vault._vaultAccess?.refresh();      // the field now shows "Unlocked" (the creator's session) or "Sealed"
  graph.setDirtyCanvas?.(true, true);
  await modalAlert(
    sealed ? "Sealed" : "Locked",
    `${sealed ? "Sealed" : "Locked"} ${count} node(s) into a vault. Original nodes removed.`
      + (lost.length ? ` Could not reconnect: ${lost.join(", ")} - connect them by hand.` : ""),
  );
}

/** The selected nodes as ComfyUI serializes them (positions, sizes, titles, colours, widget values) and the links
 *  between them: what the vault editor shows again (A17). Encrypted with the graph, never in the clear. A selection
 *  with subgraph nodes is flattened for execution and carries no layout (the editor lays it out from the graph). */
function selectionLayout(sel, graph) {
  if (sel.some((n) => isSubgraphNode(n))) return null;
  const ids = new Set(sel.map((n) => n.id));
  const nodes = sel.map((n) => JSON.parse(JSON.stringify(n.serialize())));
  const links = [];
  for (const n of sel) {
    for (const inp of n.inputs || []) {
      if (inp.link == null) continue;
      const l = linkById(graph, inp.link);
      if (l && ids.has(l.origin_id) && ids.has(l.target_id)) {
        links.push([l.id, l.origin_id, l.origin_slot, l.target_id, l.target_slot, l.type]);
      }
    }
  }
  return { nodes, links };
}

/** A boundary source -> the vault's input: an outside node, or the subgraph's own input (IO node "-10"). */
function connectSourceToVault(graph, src, vault, slot) {
  if (src?.kind === "io") {
    const io = graph.inputNode?.slots?.[src.slot];
    if (!io?.connect) return false;
    io.connect(vault.inputs[slot], vault);
    return true;
  }
  const n = graph.getNodeById?.(src?.id);
  if (!n) return false;
  n.connect(src.slot, vault, slot);
  return true;
}

/** The vault's output -> a boundary target: an outside node, or the subgraph's own output (IO node "-20"). */
function connectVaultToTarget(graph, tgt, vault, outIndex) {
  if (tgt?.kind === "io") {
    const io = graph.outputNode?.slots?.[tgt.slot];
    if (!io?.connect || !vault.outputs?.[outIndex]) return false;
    io.connect(vault.outputs[outIndex], vault);
    return true;
  }
  const n = graph.getNodeById?.(tgt?.id);
  if (!n) return false;
  vault.connect(outIndex, n, tgt.slot);
  return true;
}

// ── the empty vault's own way in: a "capture" port (owner 2026-10-14: "how tf can i connect and lock/seal any node?")
const CAPTURE = "capture";

function isEmptyVault(node) {
  return NODES.includes(node?.type) && !(widget(node, "vault_payload")?.value || "").trim();
}

function ensureCaptureSocket(node) {
  if ((node.inputs || []).some((i) => i.name === CAPTURE)) return;
  node.addInput(CAPTURE, "*");
  const s = (node.inputs || []).find((i) => i.name === CAPTURE);
  if (s) s.label = "capture: connect the output to protect";
}

function removeCaptureSocket(node) {
  const i = (node.inputs || []).findIndex((x) => x.name === CAPTURE);
  if (i < 0) return;
  if (node.inputs[i].link != null) node.disconnectInput(i);
  node.removeInput(i);
}

/** What "capture" protects: the node wired to it plus every upstream node that feeds ONLY that set (a loader also
 *  used elsewhere stays outside and becomes a vault input). */
function captureSet(vault) {
  const graph = vault.graph;
  const slot = (vault.inputs || []).find((i) => i.name === CAPTURE);
  const lk = slot?.link != null ? linkById(graph, slot.link) : null;
  const start = lk ? graph.getNodeById(lk.origin_id) : null;
  if (!start || NODES.includes(start.type)) return [];
  const set = new Map([[String(start.id), start]]);
  const capId = lk.id;
  const feedsOnlySet = (u) => (u.outputs || []).every((o) => (o.links || []).every((id) => {
    if (id === capId) return true;
    const l = linkById(graph, id);
    return !l || set.has(String(l.target_id));
  }));
  for (let changed = true; changed;) {
    changed = false;
    for (const x of [...set.values()]) {
      for (const inp of x.inputs || []) {
        const l = linkById(graph, inp.link);
        if (!l) continue;
        const u = graph.getNodeById(l.origin_id);
        if (!u || set.has(String(u.id)) || Number(u.id) < 0 || NODES.includes(u.type)) continue;
        if (feedsOnlySet(u)) { set.set(String(u.id), u); changed = true; }
      }
    }
  }
  return [...set.values()];
}

async function lockCaptured(vault, sealed) {
  const sel = captureSet(vault);
  if (!sel.length) {
    await modalAlert("Nothing connected", "Connect the output you want to protect to this vault's 'capture' input first.");
    return;
  }
  // the capture wire is only a pointer: drop it so the vault does not wire its own output back into itself
  const i = (vault.inputs || []).findIndex((x) => x.name === CAPTURE);
  if (i >= 0 && vault.inputs[i].link != null) vault.disconnectInput(i);
  return lockNodes(sel, sealed, vault);
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
    if (typeof data === "string" ? data.includes(EDIT_MARK) : JSON.stringify(data || {}).includes(EDIT_MARK)) {
      await modalAlert("Not saved", vaultSaveGuardMessage());
      throw new Error("C2C Vault: the edit view is never saved");
    }
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
  const origGet = Storage.prototype.getItem;
  Storage.prototype.getItem = function (key) {
    if (key === "litegrapheditor_clipboard" && activeEditSession() && editClipboard !== null) return editClipboard;
    return origGet.call(this, key);
  };
  Storage.prototype.setItem = function (key, value) {
    // A vault's edit view (A17): its drafts are never stored, and a node copy made inside it stays in memory (a paste
    // outside the edit view gets the ordinary clipboard, so plaintext nodes cannot leak into a saved workflow).
    if (activeEditSession() && key === "litegrapheditor_clipboard") { editClipboard = String(value); return; }
    if (typeof value === "string" && value.includes(EDIT_MARK)) return;
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
    cmd.function = async (...args) => {
      if (activeEditSession()) return guardAlert();
      if (EXPORT_CMD_IDS.includes(cmd.id) && getWorkflowVaultMarker()) return refuse();
      return orig.apply(cmd, args);
    };
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
        if (EDIT_GUARD_CMD_IDS.includes(id) && activeEditSession()) return guardAlert();
        if (EXPORT_CMD_IDS.includes(id) && getWorkflowVaultMarker()) return refuse();
        return origExec.call(this, id, ...rest);
      };
      mgr._c2cVaultExecWrapped = true;
    }
    const list = mgr.commands;
    const all = Array.isArray(list) ? list : (list instanceof Map ? [...list.values()] : Object.values(list || {}));
    // a keybinding or the menu may run .function() directly: wrap every guarded command, not only Export
    for (const cmd of all) if (EDIT_GUARD_CMD_IDS.includes(cmd?.id)) wrapOne(cmd);
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
    {
      content: "C2C Vault → This install's key…",
      callback: () => showInstallKey(),
    },
    {
      content: "C2C Vault → Second factor (authenticator)…",
      callback: () => manageSecondFactor(),
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
  const edit = document.createElement("button");
  edit.type = "button";
  edit.textContent = "Edit…";
  edit.title = "Open the vault's real graph for editing: needs the password and your authenticator code.";
  edit.addEventListener("click", () => beginEdit(node, false));
  for (const b of [go, relock, edit]) css(b, { padding: "3px 8px", cursor: "pointer", font: "inherit", whiteSpace: "nowrap" });
  const err = document.createElement("div");
  css(err, { color: "var(--c2c-dangerStrong)", minHeight: "0", whiteSpace: "normal" });
  row.append(pw, go, relock, edit);
  // an EMPTY vault: two ways in, both on the node (owner 2026-10-14: "how tf can i connect and lock/seal any node?")
  const emptyRow = document.createElement("div");
  css(emptyRow, { display: "flex", gap: "4px", alignItems: "center" });
  const verb = isSealed ? "Seal" : "Lock";
  const lockSel = document.createElement("button");
  lockSel.type = "button";
  lockSel.textContent = `${verb} selected nodes`;
  lockSel.title = "Select the nodes to protect on the canvas, then press this: they move into this vault.";
  const lockCap = document.createElement("button");
  lockCap.type = "button";
  lockCap.textContent = `${verb} what's connected`;
  lockCap.title = "Wire the output you want to protect into 'capture': that node and everything upstream that feeds only it move into this vault.";
  for (const b of [lockSel, lockCap]) css(b, { flex: "1", padding: "3px 6px", cursor: "pointer", font: "inherit", whiteSpace: "nowrap" });
  emptyRow.append(lockSel, lockCap);
  lockSel.addEventListener("click", () => lockNodes(
    selectedLockableNodes().filter((n) => n !== node && !NODES.includes(n.type)), isSealed, node));
  lockCap.addEventListener("click", () => lockCaptured(node, isSealed));
  for (const el of [status, row, emptyRow, err]) el.style.flexShrink = "0";
  host.append(status, row, emptyRow, err);

  const vaultId = () => widget(node, "vault_id")?.value || "";
  const hasPayload = () => !!(widget(node, "vault_payload")?.value || "").trim();
  let open = false;
  let sealStatus = "";
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
      status.textContent = "Empty vault: wire what to protect into 'capture', or select nodes, then:";
      pw.style.display = go.style.display = relock.style.display = "none";
      emptyRow.style.display = "";
      return;
    }
    emptyRow.style.display = "none";
    if (isSealed) {
      status.textContent = sealStatus || "Sealed";
      relock.style.display = edit.style.display = "none";
      pw.style.display = go.style.display = "";
      return;
    }
    status.textContent = open ? "Unlocked for this ComfyUI session" : "Locked · type the password to run it";
    pw.style.display = go.style.display = open ? "none" : "";
    relock.style.display = edit.style.display = open ? "" : "none";
  };
  const refresh = async () => {
    if (isSealed && hasPayload()) {
      const { ok, data } = await post("/c2c_vault/info", { payload: widget(node, "vault_payload")?.value || "" });
      sealStatus = !ok ? "Sealed" : data.legacy ? "Legacy seal · obfuscation only (open and lock it to upgrade)"
        : data.mode === "portable" ? "Portable seal · runs anywhere · obfuscation only"
        : data.runs_here ? `Sealed for ${data.installs} install(s) · runs here without a password`
        : "Sealed for other installs · needs the password here";
    }
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
      if (isSealed) { pw.value = ""; await beginEdit(node, true, password); return; }
      const route = "/c2c_vault/unlock";
      const { ok, data, status } = await post(route, { vault_id: vaultId(), password,
        payload: widget(node, "vault_payload")?.value || "" });
      if (!ok) {
        // 403 = wrong password or modified payload (deliberately one message); 429 = locked out for a while
        showError(status === 429 ? (data.error || "Too many attempts. Wait a few minutes.")
          : "Wrong password (or the vault data was changed).");
        shakeEl(host); pw.select(); return;
      }
      pw.value = "";
      open = true;
      unlockedSessions.add(vaultId());
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
    { id: "c2c.vault.lockSelection", label: "C2C Vault: Lock selected nodes (password to run)", icon: "pi pi-lock",
      function: () => lockSelection(false) },
    { id: "c2c.vault.sealSelection", label: "C2C Vault: Seal selected nodes (runs without password)", icon: "pi pi-box",
      function: () => lockSelection(true) },
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
      // onRemoved also fires when a graph switch clears the canvas (opening a vault's edit tab does exactly that):
      // the edit session and the run session belong to the vault, not to this canvas node. Lock & close reports a
      // vault node that was really deleted.
      return removed?.apply(this, arguments);
    };
  },

  getCanvasMenuItems() {
    return _vaultCanvasMenuItems();
  },

  // right-clicking a node is where people look (the canvas menu alone hid the vault, L2.45)
  getNodeMenuItems(node) {
    if (NODES.includes(node?.type)) {
      if (!isEmptyVault(node)) return [];
      const sealed = node.type === "C2C_VaultSealed";
      const verb = sealed ? "Seal" : "Lock";
      return [
        { content: `C2C Vault → ${verb} selected nodes into this vault`,
          callback: () => lockNodes(selectedLockableNodes().filter((n) => n !== node && !NODES.includes(n.type)), sealed, node) },
        { content: `C2C Vault → ${verb} what is connected to 'capture'`, callback: () => lockCaptured(node, sealed) },
      ];
    }
    const nodesFor = () => {
      const sel = selectedLockableNodes().filter((n) => !NODES.includes(n.type));
      return sel.some((n) => n === node) ? sel : [node];
    };
    return [
      { content: "C2C Vault → Lock selected nodes (password to run)", callback: () => lockNodes(nodesFor(), false, null) },
      { content: "C2C Vault → Seal selected nodes (runs without password)", callback: () => lockNodes(nodesFor(), true, null) },
    ];
  },

  getSelectionToolboxCommands(item) {
    if (!item || typeof item.serialize !== "function" || NODES.includes(item.type) || Number(item.id) < 0) return [];
    return ["c2c.vault.lockSelection", "c2c.vault.sealSelection"];
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
