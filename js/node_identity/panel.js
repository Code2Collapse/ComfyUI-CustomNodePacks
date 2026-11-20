// C2C Node Identity - floating panel in the OmniPill (house rule: no extra sidebar tab). Preview every proposed
// name before anything is written; Apply = one undo step, Cancel writes nothing, Undo = core Comfy.Undo.
// The admin section shows only for the admin (the server says who that is). Plain DOM, --c2c-* colours.
// License: Apache-2.0
import { mountOmniTool } from "../_c2c_omni_tool.js";
import { render } from "./engine.js";

const BTN = "background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:5px;"
    + "padding:4px 9px;cursor:pointer;font-size:12px";
const INPUT = "background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:4px;"
    + "padding:3px 6px;font-size:12px;min-width:0";
const DIM = "color:var(--c2c-fg-dim,#9aa)";
const STATUS_TEXT = {
    rename: "new name", unchanged: "same", "keep-manual": "kept (yours)", "keep-auto": "kept (named)",
    excluded: "skipped (type)", referenced: "skipped (used in %…%)", edit: "your name",
};

function el(tag, css, text) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text != null) e.textContent = text;
    return e;
}

function button(text, onClick, title) {
    const b = el("button", BTN, text);
    b.type = "button";
    if (title) { b.title = title; b.dataset.tip = title; }
    b.onclick = onClick;
    return b;
}

function checkbox(label, checked, title) {
    const wrap = el("label", "display:flex;align-items:center;gap:6px;cursor:pointer");
    const c = el("input");
    c.type = "checkbox";
    c.checked = !!checked;
    wrap.append(c, el("span", null, label));
    if (title) wrap.title = title;
    return { wrap, c };
}

export function mountIdentityPanel(ctx) {
    const { app, server, titles } = ctx;
    let view = { scope: "workflow", mode: "rename", restore: false, admin: false };
    let current = null;          // last preview {rows, refs}
    let refs = null;

    function build(body) {
        body.style.cssText += ";display:flex;flex-direction:column;gap:8px;padding:10px;font-size:12px;overflow:hidden";
        const status = el("div", "min-height:16px;line-height:1.35");
        status.setAttribute("role", "status");
        const titleRow = el("div", "display:flex;gap:8px;align-items:center;flex-wrap:wrap");
        const hideBtn = button("", () => ctx.toggleTitles(), "Hide or show every node title (Alt+Shift+T)");
        titleRow.append(el("span", "font-weight:600", "Titles"), hideBtn,
            el("span", DIM, "Alt+Shift+T · per node: right-click → Node Identity"));

        const scopeRow = el("div", "display:flex;gap:6px;align-items:center;flex-wrap:wrap");
        const scopeSel = el("select", INPUT);
        scopeSel.setAttribute("aria-label", "Which nodes to rename");
        for (const [v, t] of [["selected", "Selected nodes"], ["group", "Selected group"], ["workflow", "Entire workflow"]]) {
            const o = el("option", null, t);
            o.value = v;
            scopeSel.appendChild(o);
        }
        const regen = checkbox("Regenerate names it made before", false,
            "Off: nodes that already carry a Node Identity name keep it. Names you typed are always kept.");
        const previewBtn = button("Preview", () => doPreview(), "Show the proposed names; nothing is written yet");
        scopeRow.append(el("span", "font-weight:600", "Rename"), scopeSel, regen.wrap, previewBtn);

        const note = el("div", `${DIM};min-height:16px`);
        note.setAttribute("role", "status");
        note.setAttribute("aria-live", "polite");
        const table = el("div", "display:flex;flex-direction:column;gap:2px;overflow:auto;flex:1;min-height:90px;"
            + "border:1px solid var(--c2c-border);border-radius:5px;padding:4px");
        const foot = el("div", "display:flex;gap:6px;flex-wrap:wrap");
        const applyBtn = button("Apply", () => doApply(), "Write the ticked names (one undo step)");
        const cancelBtn = button("Cancel", () => { current = null; renderRows(); note.textContent = "Nothing was changed."; });
        const undoBtn = button("Undo", () => app.extensionManager?.command?.execute?.("Comfy.Undo"), "Ctrl+Z");
        const restoreBtn = button("Restore original titles", () => doRestore(),
            "Back to the titles from before Node Identity, for the chosen scope");
        foot.append(applyBtn, cancelBtn, undoBtn, restoreBtn);

        // admin section: only for the admin
        const admin = el("details", "border-top:1px solid var(--c2c-border);padding-top:6px");
        const sum = el("summary", "cursor:pointer;font-weight:600", "Naming rules (admin)");
        const form = el("div", "display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;margin-top:6px");
        const f = {
            enabled: checkbox("Feature on", true, "Off: nobody renames and titles are always shown"),
            locked: checkbox("Lock names", false, "Names are read-only for everyone except you"),
            others_may_rename: checkbox("Others may rename", false, "Clients other than this machine may rename"),
            others_may_toggle: checkbox("Others may hide titles", true, "Clients other than this machine may hide titles"),
            keep_manual: checkbox("Keep names people typed", true),
            use_prompt_words: checkbox("Use prompt words as detail", true),
            ai_naming: checkbox("AI naming hook (experimental)", false,
                "Only used when another extension registers a namer; the rules stay the fallback"),
        };
        for (const k of Object.keys(f)) form.appendChild(f[k].wrap);
        const tpl = el("input", `${INPUT};grid-column:1 / span 2`);
        tpl.setAttribute("aria-label", "Name template");
        tpl.placeholder = "{stage} · {function} · {detail}";
        const sep = el("input", INPUT);
        sep.maxLength = 5;
        sep.setAttribute("aria-label", "Separator");
        const example = el("div", DIM);
        const excl = el("textarea", `${INPUT};grid-column:1 / span 2;height:44px;resize:vertical`);
        excl.placeholder = "Node types never renamed, one per line (e.g. MyNodeType)";
        excl.setAttribute("aria-label", "Node types never renamed");
        const saveBtn = button("Save rules", () => doSave());
        form.append(el("span", `grid-column:1 / span 2;${DIM}`, "Template: {stage} {function} {detail} {type}"), tpl,
            el("span", DIM, "Separator"), sep, example, el("span"), excl, saveBtn);
        const updateExample = () => {
            example.textContent = "e.g. " + render(tpl.value || "{stage} · {function}",
                { stage: "H3", function: "Detail Refinement", detail: "Upscale", type: "KSampler" }, sep.value || " · ");
        };
        tpl.oninput = updateExample;
        sep.oninput = updateExample;
        admin.append(sum, form);

        body.append(status, titleRow, scopeRow, note, table, foot, admin);

        function renderStatus() {
            const p = server.policy || {};
            const parts = [];
            if (!server.reachable) parts.push("Server rules unavailable: renaming is off.");
            else if (p.enabled === false) parts.push("Turned off by the admin.");
            else if (server.is_admin) parts.push("You are the admin (this machine).");
            else if (!server.may?.rename) parts.push(p.locked ? "Names are locked by the admin (read-only)."
                : "Read-only: the admin has not allowed renaming from this client.");
            if (p.locked && server.is_admin) parts.push("Names are locked for other clients.");
            status.textContent = parts.join(" ");
            status.style.color = server.may?.rename ? "var(--c2c-fg)" : "var(--c2c-amber,#d9a441)";
            hideBtn.textContent = titles.state.hide ? "Show titles" : "Hide titles";
            hideBtn.disabled = p.enabled === false || !server.may?.toggle;
            for (const b of [applyBtn, restoreBtn]) b.disabled = !server.may?.rename;
            for (const b of [hideBtn, applyBtn, restoreBtn]) {       // a disabled control must look disabled
                b.style.opacity = b.disabled ? "0.45" : "1";
                b.style.cursor = b.disabled ? "not-allowed" : "pointer";
                b.title = b.disabled ? (status.textContent || "Not allowed for this client.") : (b.dataset.tip || b.title);
            }
            admin.style.display = server.is_admin ? "" : "none";
            if (server.is_admin) {
                for (const k of Object.keys(f)) f[k].c.checked = !!p[k];
                tpl.value = p.template || "";
                sep.value = p.separator || " · ";
                excl.value = (p.excluded_types || []).join("\n");
                updateExample();
            }
        }

        function renderRows() {
            table.textContent = "";
            if (!current) {
                table.appendChild(el("div", `${DIM};padding:8px`, "Choose a scope and press Preview."));
                return;
            }
            if (!current.rows.length) {
                table.appendChild(el("div", `${DIM};padding:8px`, current.note || "No node in this scope."));
                return;
            }
            const head = el("div", `display:grid;grid-template-columns:18px 1fr 1fr 92px;gap:6px;${DIM};padding:0 2px`);
            head.append(el("span"), el("span", null, "Now"), el("span", null, "Proposed (editable)"), el("span", null, "Status"));
            table.appendChild(head);
            for (const r of current.rows) {
                const row = el("div", "display:grid;grid-template-columns:18px 1fr 1fr 92px;gap:6px;align-items:center;"
                    + "padding:1px 2px");
                const tick = el("input");
                tick.type = "checkbox";
                tick.checked = r.status === "rename";
                tick.disabled = !server.may?.rename;
                tick.setAttribute("aria-label", `Rename ${r.current || "node"}`);
                const now = el("span", "overflow:hidden;text-overflow:ellipsis;white-space:nowrap", r.current || "(empty)");
                now.title = `${r.current}\nnode #${r.id} · ${r.type}${r.path ? `\nin ${r.path}` : ""}`;
                const inp = el("input", INPUT);
                inp.value = r.proposed;
                inp.readOnly = !server.may?.rename;
                inp.setAttribute("aria-label", `New name for ${r.current || "node"} (node ${r.id})`);
                inp.title = r.reason || "";
                inp.oninput = () => { tick.checked = true; };
                const st = el("span", `${DIM};font-size:11px`, STATUS_TEXT[r.status] || r.status);
                st.title = r.reason || "";
                row.append(tick, now, inp, st);
                row._c2c = { r, tick, inp };
                table.appendChild(row);
            }
        }

        function doPreview() {
            view.scope = scopeSel.value;
            view.mode = regen.c.checked ? "regenerate" : "rename";
            current = ctx.preview(view.scope, view.mode);
            refs = current.refs;
            const n = current.rows.filter((r) => r.status === "rename").length;
            note.textContent = current.note || `${n} new name(s) proposed, ${current.rows.length - n} kept or skipped. `
                + "Nothing is written until Apply.";
            renderRows();
        }

        async function doApply() {
            if (!current) { note.textContent = "Press Preview first."; return; }
            const rows = [];
            const edits = {};
            for (const row of table.children) {
                const d = row._c2c;
                if (!d || !d.tick.checked) continue;
                const value = d.inp.value;
                if (value !== d.r.proposed || (d.r.status !== "rename" && value !== d.r.current)) edits[d.r.key] = value;
                else if (d.r.status === "rename") rows.push(d.r);
                if (Object.prototype.hasOwnProperty.call(edits, d.r.key)) rows.push({ ...d.r, status: "edit" });
            }
            const res = await ctx.applyRows(rows, refs, edits);
            note.textContent = res.changed ? `${res.changed} node(s) renamed. Undo: Ctrl+Z or the Undo button.`
                : (res.problems?.length ? res.problems.join(" ") : "Nothing changed.");
            if (res.changed) doPreview();
        }

        async function doRestore() {
            const n = await ctx.restoreScope(scopeSel.value);
            note.textContent = n ? `${n} title(s) restored.` : "Nothing to restore in this scope.";
            if (current) doPreview();
        }

        async function doSave() {
            const p = {};
            for (const k of Object.keys(f)) p[k] = f[k].c.checked;
            p.template = tpl.value.trim() || "{stage} · {function}";
            p.separator = sep.value || " · ";
            p.excluded_types = excl.value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
            try {
                await ctx.savePolicy(p);
                note.textContent = "Rules saved.";
            } catch (e) {
                note.textContent = `Not saved: ${e.message || e}`;
            }
            renderStatus();
        }

        scopeSel.value = view.scope;
        renderStatus();
        renderRows();
        return {
            refresh() { renderStatus(); },
            applyView(v) {
                if (v.scope) scopeSel.value = v.scope;
                regen.c.checked = v.mode === "regenerate";
                if (v.admin && server.is_admin) admin.open = true;
                renderStatus();
                if (v.restore) { note.textContent = "Choose the scope, then press Restore original titles."; return; }
                if (v.scope) doPreview();
            },
        };
    }

    let refsUI = null;
    const handle = mountOmniTool({
        id: "node-identity", title: "C2C Node Identity", label: "Names", icon: "🏷",
        section: "tools", order: 45, width: 560, height: 520,
        build: (body) => { refsUI = build(body); return refsUI; },
        onOpen: () => refsUI?.refresh?.(),
    });
    return {
        open(v = {}) {
            view = { ...view, ...v };
            handle.open();
            refsUI?.applyView?.(v);
        },
        refresh() { refsUI?.refresh?.(); },
    };
}
