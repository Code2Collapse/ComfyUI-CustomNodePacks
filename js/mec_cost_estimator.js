/**
 * mec_cost_estimator.js — Phase 13: Render Cost Estimator
 *
 * Adds a "💰 Estimate" chip near the Queue button. Click → calls
 * /mec/cost_estimate with the current workflow JSON and shows a panel
 * with predicted time / VRAM / per-node breakdown.
 *
 * Setting:
 *   mec.cost_estimator.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

const BTN_ID    = "mec-cost-btn";
const PANEL_ID  = "mec-cost-panel";
const STYLE_ID  = "mec-cost-style";

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${BTN_ID} {
    position: fixed;
    bottom: 156px;
    right: 16px;
    z-index: 99996;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #1e1e2e;
    border: 1px solid #45475a;
    color: #fab387;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
}
#${BTN_ID}:hover { border-color: #fab387; }
#${PANEL_ID} {
    position: fixed;
    bottom: 156px;
    right: 64px;
    z-index: 99996;
    width: 400px;
    max-height: 60vh;
    overflow-y: auto;
    background: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 12px;
    color: #cdd6f4;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.65);
    display: none;
}
#${PANEL_ID}.visible { display: block; }
#${PANEL_ID} h3 { margin: 0 0 8px 0; color: #fab387; font-size: 13px; display: flex; justify-content: space-between; }
#${PANEL_ID} .ce-close { background: none; border: none; color: #6c7086; cursor: pointer; font-size: 14px; }
#${PANEL_ID} .ce-summary {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin-bottom: 10px;
}
#${PANEL_ID} .ce-card {
    background: #313244;
    border-radius: 5px;
    padding: 6px 8px;
    text-align: center;
}
#${PANEL_ID} .ce-card .v {
    font-size: 16px;
    font-weight: 700;
    color: #a6e3a1;
    font-family: monospace;
}
#${PANEL_ID} .ce-card .l { font-size: 10px; color: #6c7086; text-transform: uppercase; }
#${PANEL_ID} table { width: 100%; border-collapse: collapse; font-size: 11px; }
#${PANEL_ID} th, #${PANEL_ID} td {
    text-align: left;
    padding: 4px 6px;
    border-bottom: 1px solid #313244;
}
#${PANEL_ID} th { color: #6c7086; font-weight: 600; }
#${PANEL_ID} td.right { text-align: right; font-family: monospace; }
#${PANEL_ID} td.bar { width: 60px; padding: 0 6px; }
#${PANEL_ID} .ce-bar { height: 4px; background: #f9e2af; border-radius: 2px; }
#${PANEL_ID} .ce-warn {
    background: #3a311e;
    border-left: 3px solid #f9e2af;
    padding: 4px 8px;
    margin-bottom: 8px;
    font-size: 11px;
    color: #f9e2af;
}
#${PANEL_ID} .ce-cold { color: #fab387; font-style: italic; }
    `.trim();
    document.head.appendChild(style);
}

function _ensureUi() {
    if (!document.getElementById(BTN_ID)) {
        const b = document.createElement("button");
        b.id = BTN_ID;
        b.title = "Estimate render cost";
        b.textContent = "💰";
        b.addEventListener("click", _toggle);
        document.body.appendChild(b);
    }
    if (!document.getElementById(PANEL_ID)) {
        const p = document.createElement("div");
        p.id = PANEL_ID;
        document.body.appendChild(p);
    }
}

let _open = false;
function _toggle() {
    _open = !_open;
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    if (_open) {
        p.classList.add("visible");
        _run();
    } else {
        p.classList.remove("visible");
    }
}

async function _getWorkflowApi() {
    // app.graphToPrompt() returns {output, workflow}. `output` is the API-format
    // dict { "<id>": {class_type, inputs}, ... } we need.
    try {
        const r = await app.graphToPrompt();
        return r?.output || null;
    } catch (e) {
        console.warn("[MEC.CostEstimator] graphToPrompt failed:", e);
        return null;
    }
}

function _fmtMs(ms) {
    if (ms < 1000) return `${ms.toFixed(0)} ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(2)} s`;
    return `${Math.floor(ms / 60000)}m ${((ms % 60000) / 1000).toFixed(1)}s`;
}

async function _run() {
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    p.innerHTML = `
        <h3>💰 Render cost estimate <button class="ce-close">×</button></h3>
        <div>Calculating…</div>
    `;
    p.querySelector(".ce-close").addEventListener("click", _toggle);

    const wf = await _getWorkflowApi();
    if (!wf || Object.keys(wf).length === 0) {
        p.innerHTML = `
            <h3>💰 Render cost estimate <button class="ce-close">×</button></h3>
            <div class="ce-warn">No workflow on canvas.</div>`;
        p.querySelector(".ce-close").addEventListener("click", _toggle);
        return;
    }

    let data;
    try {
        const r = await fetch("/mec/cost_estimate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ workflow: wf }),
        });
        const j = await r.json();
        if (!j.success) throw new Error(j.error || "fail");
        data = j.data;
    } catch (e) {
        console.warn("[MEC.CostEstimator] estimate failed:", e);
        p.innerHTML = `
            <h3>💰 Render cost estimate <button class="ce-close">×</button></h3>
            <div class="ce-warn">Estimate failed — see console.</div>`;
        p.querySelector(".ce-close").addEventListener("click", _toggle);
        return;
    }

    const maxMs = Math.max(1, ...data.per_node.map(r => r.ms));
    const rowsHtml = data.per_node
        .sort((a, b) => b.ms - a.ms)
        .map(r => {
            const pct = (r.ms / maxMs) * 100;
            const cls = r.samples === 0 ? "ce-cold" : "";
            return `
                <tr>
                    <td class="${cls}">${r.class_type}</td>
                    <td class="right">${_fmtMs(r.ms)}</td>
                    <td class="right">${r.vram_mb ? r.vram_mb.toFixed(0) + " MB" : "—"}</td>
                    <td class="bar"><div class="ce-bar" style="width:${pct}%"></div></td>
                    <td class="right">${r.samples}</td>
                </tr>
            `;
        }).join("");

    p.innerHTML = `
        <h3>💰 Render cost estimate <button class="ce-close">×</button></h3>
        <div class="ce-summary">
            <div class="ce-card"><div class="v">${_fmtMs(data.total_ms)}</div><div class="l">Total time</div></div>
            <div class="ce-card"><div class="v">${data.peak_vram_mb.toFixed(0)} MB</div><div class="l">Peak VRAM</div></div>
            <div class="ce-card"><div class="v">${data.per_node.length}</div><div class="l">Nodes</div></div>
        </div>
        ${data.warnings.map(w => `<div class="ce-warn">⚠ ${w}</div>`).join("")}
        <table>
            <thead><tr><th>Node class</th><th class="right">Time</th><th class="right">VRAM</th><th>Δ</th><th class="right">n</th></tr></thead>
            <tbody>${rowsHtml}</tbody>
        </table>
        <div style="margin-top:8px;font-size:10px;color:#6c7086;">History: ${data.samples_in_history} samples · italic = no prior runs (fallback default).</div>
    `;
    p.querySelector(".ce-close").addEventListener("click", _toggle);
}

app.registerExtension({
    name: "MEC.CostEstimator",
    settings: [
        {
            id: "mec.cost_estimator.enabled",
            name: "Cost Estimator: enabled",
            tooltip: "Show 💰 button to predict render time before queueing.",
            type: "boolean",
            defaultValue: true,
            onChange: (v) => {
                const b = document.getElementById(BTN_ID);
                if (b) b.style.display = v ? "flex" : "none";
            },
        },
    ],
    async setup() {
        _injectStyle();
        _ensureUi();
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("mec.cost_estimator.enabled", true); }
            catch { return true; }
        })();
        const b = document.getElementById(BTN_ID);
        if (b) b.style.display = enabled ? "flex" : "none";
        console.log("[MEC.CostEstimator] Loaded.");
    },
});
