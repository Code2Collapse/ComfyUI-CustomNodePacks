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
import { findNodeAnywhere } from "./_subgraph_walk.js";

function gatherContext(nodeId) {
    // server ids of nodes inside subgraphs are composite ("12:5"): Number() of them is NaN (L2.47)
    const n = findNodeAnywhere(nodeId)?.node ||
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

// L6.07: the last diagnoses, kept in THIS browser only (localStorage); never sent anywhere
const HISTORY_KEY = "c2c.diagnose.history";
function _history() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch (_) { return []; }
}
function _remember(node, diag) {
    try {
        const h = _history();
        h.unshift({ at: new Date().toISOString(), node: node ? (node.title || node.type) : "Workflow",
                    cause: diag.cause || "", fix: diag.fix || "", provider: diag.provider || "rules",
                    report: diag.report || "" });
        localStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(0, 20)));
    } catch (_) { /* storage full or blocked: history is a convenience */ }
}

function showCard(node, diag, body) {
    document.getElementById("__c2c_diag_card")?.remove();
    const card = document.createElement("div");
    card.id = "__c2c_diag_card";
    card.style.cssText = `
        position:fixed;right:16px;bottom:16px;z-index:2147483000;width:min(420px,90vw);
        background:var(--c2c-bg);color:var(--c2c-fg);
        border:1px solid var(--c2c-red);border-radius:10px;padding:14px;
        font:12.5px system-ui,sans-serif;box-shadow:0 10px 40px rgba(0,0,0,0.6);
    `;
    const title = node ? `${node.title || node.type} failed` : "Workflow error";
    const applyBtn = diag.apply
        ? `<button data-apply style="padding:5px 12px;border-radius:6px;border:1px solid var(--c2c-green);
             background:var(--c2c-green);color:var(--c2c-onAccent);font-weight:600;cursor:pointer;">
             Apply fix: ${diag.apply.widget} → ${JSON.stringify(diag.apply.value)}</button>`
        : "";
    card.innerHTML = `
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span style="font-size:15px;">🩺</span>
            <span style="font-weight:600;">${title}</span>
            <span style="margin-left:auto;color:var(--c2c-overlay1);font-size:10.5px;">via ${diag.provider}</span>
            <button data-x style="background:none;border:none;color:inherit;cursor:pointer;font-size:14px;">✕</button>
        </div>
        <div data-cause style="margin-bottom:4px;"></div>
        <div data-fix style="color:var(--c2c-sub);margin-bottom:${diag.apply || diag.consent ? "10px" : "0"};"></div>
        ${diag.consent ? `<div data-consent style="margin-bottom:10px;padding:8px;border:1px solid var(--c2c-surface1);
                border-radius:6px;">
            <div data-consent-line style="margin-bottom:6px;"></div>
            <details><summary style="cursor:pointer;color:var(--c2c-sub);">What will be sent</summary>
                <pre data-consent-text style="max-height:180px;overflow:auto;white-space:pre-wrap;margin:6px 0 0;
                     font:11px ui-monospace,monospace;"></pre></details>
            <button data-ask-cloud style="margin-top:8px;padding:5px 12px;border-radius:6px;border:1px solid var(--c2c-surface1);
                background:transparent;color:inherit;cursor:pointer;"></button>
        </div>` : ""}
        <div style="display:flex;gap:8px;">${applyBtn}
            ${node ? `<button data-goto style="padding:5px 12px;border-radius:6px;border:1px solid var(--c2c-surface1);
                background:transparent;color:inherit;cursor:pointer;">Show node</button>` : ""}
            ${diag.report ? `<button data-copy style="padding:5px 12px;border-radius:6px;border:1px solid var(--c2c-surface1);
                background:transparent;color:inherit;cursor:pointer;" title="Copies the error, cause and fix with keys and file paths removed">Copy report</button>` : ""}
            <button data-history style="padding:5px 12px;border-radius:6px;border:1px solid var(--c2c-surface1);
                background:transparent;color:inherit;cursor:pointer;">History</button>
        </div>
    `;
    // model text goes in as TEXT, never as HTML (L6.01 S2)
    card.querySelector("[data-cause]").textContent = diag.cause || "";
    card.querySelector("[data-fix]").textContent = diag.fix || "";
    if (diag.consent) {
        const c = diag.consent;
        const bits = [`${c.chars ?? (c.text || "").length} characters`];
        if (c.redacted) bits.push("keys, paths and e-mail addresses removed");
        if (c.est_cost_usd_max != null) bits.push(`at most $${Number(c.est_cost_usd_max).toFixed(4)}`);
        card.querySelector("[data-consent-line]").textContent =
            `No model on this computer answered. ${c.backend}${c.model ? ` (${c.model})` : ""} can - ` +
            `it is an online service, so this error's details would leave your computer (${bits.join(", ")}).`;
        card.querySelector("[data-consent-text]").textContent = c.text || "";
        const ask = card.querySelector("[data-ask-cloud]");
        ask.textContent = `Ask ${c.backend}`;
        ask.onclick = async () => {
            ask.disabled = true;
            ask.textContent = "Asking…";
            try {
                const r = await (await fetch("/c2c/ai/diagnose", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ ...body, consent_cloud: true }),
                })).json();
                if (r?.ok) showCard(node, { ...r, consent: null }, body);
            } catch (e) {
                ask.disabled = false;
                ask.textContent = `Ask ${c.backend} (failed - try again)`;
            }
        };
    }
    document.body.appendChild(card);
    if (!diag.consent) _remember(node, diag);
    card.querySelector("[data-copy]")?.addEventListener("click", async (e) => {
        const btn = e.currentTarget;            // currentTarget is null again after the await
        try { await navigator.clipboard.writeText(diag.report); btn.textContent = "Copied"; }
        catch (_) { btn.textContent = "Copy failed"; }
    });
    card.querySelector("[data-history]").addEventListener("click", (e) => {
        let box = card.querySelector("[data-history-list]");
        if (box) { box.remove(); return; }
        box = document.createElement("div");
        box.dataset.historyList = "1";
        box.style.cssText = "margin-top:10px;max-height:200px;overflow:auto;border-top:1px solid var(--c2c-surface1);" +
                            "padding-top:8px;display:flex;flex-direction:column;gap:6px;";
        const items = _history();
        if (!items.length) box.textContent = "No earlier diagnoses on this browser.";
        for (const it of items) {
            const row = document.createElement("div");
            row.style.cssText = "font-size:11.5px;";
            const t = document.createElement("div");
            t.style.cssText = "color:var(--c2c-sub);";
            t.textContent = `${new Date(it.at).toLocaleString()} · ${it.node} · via ${it.provider}`;
            const c = document.createElement("div");
            c.textContent = it.cause;
            row.append(t, c);
            box.appendChild(row);
        }
        card.appendChild(box);
    });
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
        if (r?.ok) showCard(node, r, body);
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
