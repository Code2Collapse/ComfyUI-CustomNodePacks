import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * C2C Vault — password-locked subgraph.
 *
 * The password is typed into a transient modal and POSTed straight to
 * /c2c_vault/unlock. It is never written to a widget, never held in the node,
 * and never reaches the queued prompt — a widget value would be serialised into
 * the workflow JSON, which is the very file the vault protects.
 */

const NODES = ["C2C_VaultLocked", "C2C_VaultSealed"];

function css(el, s) { Object.assign(el.style, s); }

function modal({ title, note, confirm = "OK", withConfirmField = false }) {
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
      minWidth: "340px", font: "13px sans-serif",
      boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    });

    const h = document.createElement("div");
    h.textContent = title;
    css(h, { fontWeight: "600", marginBottom: "10px" });

    const p = document.createElement("div");
    p.textContent = note || "";
    css(p, { opacity: "0.75", marginBottom: "12px", lineHeight: "1.45" });

    const mk = (ph) => {
      const i = document.createElement("input");
      i.type = "password";
      i.placeholder = ph;
      i.autocomplete = "new-password";
      css(i, {
        width: "100%", boxSizing: "border-box", marginBottom: "8px",
        padding: "7px 9px", borderRadius: "4px",
        background: "var(--comfy-input-bg, #222)",
        color: "var(--input-text, #ccc)",
        border: "1px solid var(--border-color, #4a4a4a)",
      });
      return i;
    };
    const pw = mk("Vault password");
    const pw2 = withConfirmField ? mk("Confirm password") : null;

    const err = document.createElement("div");
    css(err, { color: "#f87171", minHeight: "16px", marginBottom: "8px" });

    const row = document.createElement("div");
    css(row, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const cancel = document.createElement("button");
    cancel.textContent = "Cancel";
    const ok = document.createElement("button");
    ok.textContent = confirm;
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
      if (!pw.value) { err.textContent = "Enter a password."; return; }
      if (pw2 && pw.value !== pw2.value) { err.textContent = "Passwords do not match."; return; }
      // Read the value, then let the field go out of scope with the modal.
      const v = pw.value;
      pw.value = ""; if (pw2) pw2.value = "";
      done(v);
    };
    back.onkeydown = (e) => {
      if (e.key === "Escape") done(null);
      if (e.key === "Enter") ok.click();
    };

    row.append(cancel, ok);
    box.append(h, p, pw); if (pw2) box.append(pw2);
    box.append(err, row);
    back.append(box);
    document.body.append(back);
    pw.focus();
  });
}

async function post(route, body) {
  const r = await api.fetchApi(route, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = {};
  try { data = await r.json(); } catch (_) { /* keep {} */ }
  return { ok: r.ok && data.ok, data };
}

function widget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

app.registerExtension({
  name: "Code2Collapse.CustomNodePacks.Vault",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData.name)) return;

    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      const node = this;

      // The payload is ciphertext, but it is long and unreadable — showing the
      // raw base64 in a multiline box is noise. Keep the widget (it must still
      // serialise) and collapse its height to zero.
      const pay = widget(node, "vault_payload");
      if (pay) {
        pay.computeSize = () => [0, -4];
        if (pay.inputEl) pay.inputEl.hidden = true;
      }

      const isSealed = nodeData.name === "C2C_VaultSealed";

      // A sealed vault already runs; the only thing a password buys is EDITING.
      // Offering "Unlock to run" there would imply it was blocked, which it is not.
      node.addWidget("button", isSealed ? "Open for editing…" : "Unlock…", null, async () => {
        const id = widget(node, "vault_id")?.value || "";
        const payload = pay?.value || "";
        if (!payload) { alert("This vault has no payload yet."); return; }
        const pw = await modal({
          title: "Unlock vault",
          note: isSealed
            ? "This vault already runs without a password. The password is only "
              + "needed to read its contents for editing."
            : "The password is sent once to unlock this session and is never stored "
              + "in the workflow. The unlock lasts for the rest of the session.",
          confirm: isSealed ? "Open" : "Unlock",
        });
        if (pw === null) return;
        const route = isSealed ? "/c2c_vault/open" : "/c2c_vault/unlock";
        const { ok, data } = await post(route, { vault_id: id, password: pw, payload });
        alert(ok
          ? (isSealed ? "Vault opened — contents returned for editing."
                      : "Vault unlocked for this session.")
          : (data.error || "Could not open the vault."));
      });

      node.addWidget("button", "Lock session", null, async () => {
        await post("/c2c_vault/lock_session", { vault_id: widget(node, "vault_id")?.value || "" });
        alert("Vault re-locked.");
      });

      return r;
    };
  },

  // Lock-from-selection lives on the canvas menu, not the node: you are turning
  // a set of EXISTING nodes into a vault, so there is no vault node to click yet.
  setup() {
    const orig = app.canvas.getCanvasMenuOptions;
    app.canvas.getCanvasMenuOptions = function () {
      const opts = orig?.apply(this, arguments) || [];
      opts.push({
        content: "C2C: Lock selection into a Vault (password to RUN)",
        callback: () => lockSelection(false),
      });
      opts.push({
        content: "C2C: Seal selection into a Vault (runs without password)",
        callback: () => lockSelection(true),
      });
      return opts;
    };

    // Mirrors input_0..input_2 declared in nodes/vault_node.py.
    const MAX_VAULT_INPUTS = 3;

    async function lockSelection(sealed) {
        {
          const sel = Object.values(app.canvas.selected_nodes || {});
          if (sel.length < 1) { alert("Select the nodes to lock first."); return; }
          const pw = await modal({
            title: `${sealed ? "Seal" : "Lock"} ${sel.length} node(s) into a vault`,
            note: sealed
              ? "Sealed: the recipient RUNS this with no password, and cannot read "
                + "it. Your password is still needed to open it for editing — keep "
                + "it, or you lose the ability to edit this vault."
              : "Locked: the recipient needs this password to run it at all. Store "
                + "it somewhere safe — there is no recovery, by design.",
            confirm: sealed ? "Seal" : "Lock",
            withConfirmField: true,
          });
          if (pw === null) return;

          const idOf = (n) => String(n.id);
          const ids = new Set(sel.map(idOf));
          const nodes = sel.map((n) => ({
            id: idOf(n),
            class_type: n.comfyClass || n.type,
            widgets: Object.fromEntries(
              (n.widgets || []).map((w) => [w.name, w.value])),
          }));
          // Every link is one of three kinds: wholly INSIDE the selection (it
          // travels with the vault), entering from OUTSIDE (a boundary input),
          // or leaving to OUTSIDE (a boundary output). This used to keep only
          // the inside ones and send empty boundary lists, which produced a
          // vault that could never run - execute_subgraph rejects a subgraph
          // declaring no outputs, so every UI-locked vault failed at runtime.
          const links = [];
          const boundary_in = [];
          const externalSources = [];        // parallel to boundary_in
          for (const n of sel) {
            (n.inputs || []).forEach((inp, slot) => {
              const lk = inp.link != null ? app.graph.links[inp.link] : null;
              if (!lk) return;
              if (ids.has(String(lk.origin_id))) {
                links.push({ from: String(lk.origin_id), from_slot: lk.origin_slot,
                             to: idOf(n), to_slot: slot });
              } else {
                boundary_in.push({ name: `in_${boundary_in.length}`,
                                   to: idOf(n), to_slot: slot });
                externalSources.push({ id: lk.origin_id, slot: lk.origin_slot });
              }
            });
          }

          const boundary_out = [];
          const externalTargets = [];        // parallel to boundary_out
          for (const n of sel) {
            (n.outputs || []).forEach((out, slot) => {
              (out.links || []).forEach((lid) => {
                const lk = app.graph.links[lid];
                if (!lk || ids.has(String(lk.target_id))) return;
                boundary_out.push({ name: `out_${boundary_out.length}`,
                                    from: idOf(n), from_slot: slot });
                externalTargets.push({ id: lk.target_id, slot: lk.target_slot });
              });
            });
          }

          // A self-contained selection feeds nothing downstream, so no link
          // crosses outward. Fall back to its sinks - the outputs nothing
          // inside consumes - so it still has something to return.
          if (boundary_out.length === 0) {
            const consumed = new Set(links.map((l) => `${l.from}:${l.from_slot}`));
            for (const n of sel) {
              (n.outputs || []).forEach((out, slot) => {
                if (consumed.has(`${idOf(n)}:${slot}`)) return;
                boundary_out.push({ name: `out_${boundary_out.length}`,
                                    from: idOf(n), from_slot: slot });
                externalTargets.push(null);
              });
            }
          }

          if (boundary_out.length === 0) {
            alert("These nodes produce no output, so a vault built from them could "
                + "never run. Include the node whose result you actually want.");
            return;
          }
          if (boundary_in.length > MAX_VAULT_INPUTS) {
            alert(`This selection needs ${boundary_in.length} inputs from outside, but `
                + `a vault node has only ${MAX_VAULT_INPUTS}. Select the upstream nodes `
                + "too, so fewer wires cross the boundary.");
            return;
          }
          if (boundary_out.length > 1) {
            const go = confirm(`This selection produces ${boundary_out.length} outputs, `
                + `but a vault node has one, so only ${boundary_out[0].name} will be `
                + "wired up. Continue?");
            if (!go) return;
          }

          const vault_id = `vault-${Math.random().toString(36).slice(2, 10)}`;
          const { ok, data } = await post("/c2c_vault/lock", {
            vault_id, password: pw, mode: sealed ? "sealed" : "locked",
            subgraph: { nodes, links, boundary_in, boundary_out },
          });
          if (!ok) { alert(data.error || "Could not lock the selection."); return; }

          const vault = LiteGraph.createNode(sealed ? "C2C_VaultSealed" : "C2C_VaultLocked");
          vault.pos = [sel[0].pos[0], sel[0].pos[1]];
          app.graph.add(vault);
          widget(vault, "vault_id").value = vault_id;
          widget(vault, "vault_payload").value = data.payload;

          // Put the vault where the selection sat in the graph, so the wires
          // that crossed the boundary now cross into and out of the vault.
          externalSources.forEach((src, i) => {
            const srcNode = app.graph.getNodeById(src.id);
            if (srcNode) srcNode.connect(src.slot, vault, i);
          });
          if (externalTargets[0]) {
            const tgtNode = app.graph.getNodeById(externalTargets[0].id);
            if (tgtNode) vault.connect(0, tgtNode, externalTargets[0].slot);
          }
          app.graph.setDirtyCanvas(true, true);
          alert(`${sealed ? "Sealed" : "Locked"} ${sel.length} node(s). The `
              + "originals are left in place — delete them once you have verified "
              + "the vault runs.");
        }
    }
  },
});
