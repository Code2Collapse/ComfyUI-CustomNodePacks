// C2C Bypass Manager panel (L4.03 slice 3): a floating tool in the OmniPill (house rule: no extra sidebar tab).
// One row per node type in the whole workflow (subgraphs included): count + a tri-state button (normal -> bypassed ->
// muted -> normal; "mixed" goes to normal). A filter by type / title / pattern, "restore all", and named sets saved
// in the workflow (graph.extra.c2c_bypass_sets). Every change is one undo step; pinned / locked nodes are kept.
import { app } from "../../scripts/app.js";
import { mountOmniTool } from "./_c2c_omni_tool.js";
import { MODE, walk, match, apply, summary, nextMode, saveSet, applySet } from "./c2c_bypass/engine.js";

const STATE_STYLE = {
    normal: "background:var(--c2c-bg2);color:var(--c2c-fg)",
    bypassed: "background:#5b2a7a;color:#fff",
    muted: "background:#3a3f4a;color:#cfd3dc",
    mixed: "background:repeating-linear-gradient(45deg,#5b2a7a 0 6px,#3a3f4a 6px 12px);color:#fff",
};

function rootGraph() {
    return app.rootGraph ?? app.graph;
}

function oneStep(fn) {
    app.canvas?.emitBeforeChange?.();
    try {
        return fn();
    } finally {
        app.canvas?.emitAfterChange?.();
        app.canvas?.setDirty?.(true, true);
    }
}

function el(tag, css, text) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text != null) e.textContent = text;
    return e;
}

function build(body) {
    body.style.cssText += ";display:flex;flex-direction:column;gap:8px;padding:10px;font-size:12px";
    const bar = el("div", "display:flex;gap:6px;align-items:center");
    const by = el("select", "background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);"
        + "border-radius:4px;padding:3px");
    for (const [v, t] of [["type", "type contains"], ["regex", "pattern (regex)"]]) {
        const o = el("option", null, t);
        o.value = v;
        by.appendChild(o);
    }
    const q = el("input", "flex:1;min-width:0;background:var(--c2c-bg2);color:var(--c2c-fg);"
        + "border:1px solid var(--c2c-border);border-radius:4px;padding:4px 6px");
    q.placeholder = "filter node types...";
    bar.append(by, q);
    const note = el("div", "color:var(--c2c-fg-dim,#9aa);min-height:16px");
    const list = el("div", "display:flex;flex-direction:column;gap:3px;overflow:auto;flex:1;min-height:80px");
    const foot = el("div", "display:flex;gap:6px;flex-wrap:wrap");
    const btn = (t, f) => {
        const b = el("button", "background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);"
            + "border-radius:5px;padding:4px 9px;cursor:pointer", t);
        b.onclick = f;
        return b;
    };

    function entries() {
        const all = walk(rootGraph());
        const v = q.value.trim();
        if (!v) return all;
        if (by.value === "regex") return match(all, { by: "regex", value: v });
        const low = v.toLowerCase();
        return all.filter((e) => String(e.node.comfyClass ?? e.node.type ?? "").toLowerCase().includes(low));
    }

    function render() {
        list.textContent = "";
        let rows;
        try {
            rows = summary(entries());
            note.textContent = "";
        } catch (e) {
            note.textContent = e.message;
            return;
        }
        if (!rows.length) {
            list.appendChild(el("div", "color:var(--c2c-fg-dim,#9aa);padding:8px", "No node matches."));
            return;
        }
        for (const r of rows) {
            const row = el("div", "display:flex;align-items:center;gap:8px;padding:2px 0");
            const b = el("button", `min-width:84px;border:1px solid var(--c2c-border);border-radius:5px;padding:3px 6px;`
                + `cursor:pointer;${STATE_STYLE[r.state]}`, r.state);
            b.title = "Click: normal -> bypassed -> muted -> normal (mixed -> normal)";
            b.onclick = () => {
                const res = oneStep(() => apply(match(walk(rootGraph()), { by: "type", value: r.type }),
                                                nextMode(r.state)));
                note.textContent = `${r.type}: ${res.changed.length} changed`
                    + (res.skipped.length ? `, ${res.skipped.length} pinned / locked kept` : "");
                render();
            };
            row.append(b, el("span", "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap", r.type),
                       el("span", "color:var(--c2c-fg-dim,#9aa);font-variant-numeric:tabular-nums", `x${r.count}`));
            list.appendChild(row);
        }
    }

    foot.append(
        btn("Restore all", () => {
            const res = oneStep(() => apply(walk(rootGraph()), MODE.NORMAL));
            note.textContent = `${res.changed.length} node(s) back to normal`;
            render();
        }),
        btn("Save set...", () => {
            const name = window.prompt("Name of this bypass set (saved in the workflow):", "draft");
            if (!name) return;
            const n = saveSet(rootGraph(), name.trim(), walk(rootGraph()));
            app.canvas?.setDirty?.(true, true);
            note.textContent = `Set '${name.trim()}' saved (${n} nodes).`;
        }),
        btn("Apply set...", () => {
            const sets = Object.keys(rootGraph()?.extra?.c2c_bypass_sets ?? {});
            if (!sets.length) {
                note.textContent = "No set saved in this workflow yet.";
                return;
            }
            const name = window.prompt(`Apply which set? (${sets.join(", ")})`, sets[0]);
            if (!name) return;
            try {
                const res = oneStep(() => applySet(rootGraph(), name.trim()));
                note.textContent = `Set '${name.trim()}': ${res.changed.length} changed`
                    + (res.missing.length ? `, ${res.missing.length} node(s) no longer in the workflow` : "");
            } catch (e) {
                note.textContent = e.message;
            }
            render();
        }),
        btn("Refresh", render),
    );
    q.oninput = render;
    by.onchange = render;
    body.append(bar, note, list, foot);
    render();
    return { refresh: render };
}

let refs = null;
let handle = null;
app.registerExtension({
    name: "C2C.BypassManager.Panel",
    // fe 1.53.10 has no extensionManager.registerCommand (the OmniPill helper's fallback is skipped there), so the
    // panel's command is declared here, where every frontend version reads it
    commands: [{ id: "c2c.bypass.panel", label: "C2C: Open Bypass Manager", function: () => handle?.open?.() }],
    setup() {
        try {
            handle = mountOmniTool({
                id: "bypass-manager", title: "C2C Bypass Manager", label: "Bypass", icon: "⏸",
                section: "tools", order: 40, width: 420, height: 380,
                build: (body) => { refs = build(body); return refs; },
                onOpen: () => refs?.refresh?.(),
            });
        } catch (e) {
            console.warn("[C2C] Bypass Manager panel not mounted:", e);
        }
    },
});
