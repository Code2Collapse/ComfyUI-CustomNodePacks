// c2c_ai_diagnose.js — graph-aware error diagnosis card with apply-fix.
//
// On execution_error: read the FAILING NODE's live configuration from the
// graph (widget values, upstream node types), POST /c2c/ai/diagnose, and
// show a body-mounted card (popover rule: in-node UI gets eclipsed by
// third-party overlays) with the plain-English cause + fix. When the backend
// returns a mechanically-validated {widget, value} suggestion, an "Apply
// fix" button sets it after one confirmation. Additive — does not touch the
// existing error translator toast.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function gatherContext(nodeId) {
    const n = app.graph?.getNodeById?.(Number(nodeId)) ||
              (app.graph?._nodes || []).find(x => String(x.id) === String(nodeId));
    if (!n) return { node: null, ctx: {} };
    const widgets = {};
    for (const w of n.widgets || []) {
        if (w.type === "hidden" || typeof w.value === "object") continue;
        const v = w.value;
        if (typeof v === "string" && v.length > 300) continue;   // json blobs
        widgets[w.name] = v;
    }
    const upstream = [];
    for (const inp of n.inputs || []) {
        if (inp.link == null) continue;
        const link = (n.graph || app.graph)?.links?.[inp.link];
        const src = link && (n.graph || app.graph)?.getNodeById?.(link.origin_id);
        if (src) upstream.push({ id: String(src.id), type: src.type });
    }
    return { node: n, ctx: { widgets, upstream } };
}

function showCard(node, diag) {
    document.getElementById("__c2c_diag_card")?.remove();
    const card = document.createElement("div");
    card.id = "__c2c_diag_card";
    card.style.cssText = `
        position:fixed;right:16px;bottom:16px;z-index:2147483000;width:min(420px,90vw);
        background:var(--c2c-bg,#1e1e2e);color:var(--c2c-fg,#cdd6f4);
        border:1px solid var(--c2c-red,#f38ba8);border-radius:10px;padding:14px;
        font:12.5px system-ui,sans-serif;box-shadow:0 10px 40px rgba(0,0,0,0.6);
    `;
    const title = node ? `${node.title || node.type} failed` : "Workflow error";
    const applyBtn = diag.apply
        ? `<button data-apply style="padding:5px 12px;border-radius:6px;border:1px solid var(--c2c-green,#a6e3a1);
             background:var(--c2c-green,#a6e3a1);color:#11111b;font-weight:600;cursor:pointer;">
             Apply fix: ${diag.apply.widget} → ${JSON.stringify(diag.apply.value)}</button>`
        : "";
    card.innerHTML = `
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span style="font-size:15px;">🩺</span>
            <span style="font-weight:600;">${title}</span>
            <span style="margin-left:auto;color:var(--c2c-overlay1,#7f849c);font-size:10.5px;">via ${diag.provider}</span>
            <button data-x style="background:none;border:none;color:inherit;cursor:pointer;font-size:14px;">✕</button>
        </div>
        <div style="margin-bottom:4px;">${diag.cause || ""}</div>
        <div style="color:var(--c2c-overlay1,#a6adc8);margin-bottom:${diag.apply ? "10px" : "0"};">${diag.fix || ""}</div>
        <div style="display:flex;gap:8px;">${applyBtn}
            ${node ? `<button data-goto style="padding:5px 12px;border-radius:6px;border:1px solid var(--c2c-surface1,#45475a);
                background:transparent;color:inherit;cursor:pointer;">Show node</button>` : ""}
        </div>
    `;
    document.body.appendChild(card);
    card.querySelector("[data-x]").onclick = () => card.remove();
    card.querySelector("[data-goto]")?.addEventListener("click", () => {
        try {
            app.canvas.centerOnNode(node);
            app.canvas.selectNode(node);
        } catch (_) {}
    });
    const btn = card.querySelector("[data-apply]");
    if (btn && diag.apply) {
        btn.onclick = () => {
            const { widget, value } = diag.apply;
            if (!confirm(`Set ${node?.type}.${widget} = ${JSON.stringify(value)}?`)) return;
            const w = (node?.widgets || []).find(x => x.name === widget);
            if (!w) { btn.textContent = "Widget not found"; return; }
            const old = w.value;
            w.value = value;
            try { w.callback?.(value, app.canvas, node, [0, 0], null); } catch (_) {}
            node.setDirtyCanvas?.(true, true);
            btn.textContent = `Applied (was ${JSON.stringify(old)}) — run again`;
            btn.disabled = true;
            btn.style.opacity = "0.7";
            app.extensionManager?.toast?.add?.({
                severity: "success", summary: "Fix applied",
                detail: `${node.type}.${widget} = ${JSON.stringify(value)}`, life: 4000,
            });
        };
    }
    // Errors deserve attention but not forever.
    setTimeout(() => { try { card.remove(); } catch (_) {} }, 90_000);
}

async function runDiagnosis(detail) {
    try {
        const nodeId = detail?.node_id;
        const { node, ctx } = gatherContext(nodeId);
        const body = {
            exc_type: detail?.exception_type || "",
            message: detail?.exception_message || "",
            traceback: (detail?.traceback || []).join ? (detail.traceback || []).join("\n") : String(detail?.traceback || ""),
            node_id: String(nodeId ?? ""),
            node_type: detail?.node_type || node?.type || "",
            ...ctx,
        };
        const r = await (await fetch("/c2c/ai/diagnose", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        })).json();
        if (r?.ok) showCard(node, r);
    } catch (e) {
        console.warn("[C2C diagnose] failed", e);
    }
}

app.registerExtension({
    name: "C2C.AIDiagnose",
    setup() {
        api.addEventListener("execution_error", (ev) => runDiagnosis(ev.detail));
    },
});

// Test hook: run the full pipeline (context → route → card → apply) without
// a real execution failure.
window.__c2cDiagnoseTest = (detail) => runDiagnosis(detail);
