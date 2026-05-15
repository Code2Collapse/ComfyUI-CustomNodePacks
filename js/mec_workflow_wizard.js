/**
 * mec_workflow_wizard.js — Phase 15: Workflow Wizard / Guided Mode
 *
 * Wand button → template picker → step-by-step guided overlay that
 * highlights the next node-type to configure with a pulsing border and
 * shows the step's hint as a floating tip.
 *
 * Setting:
 *   mec.workflow_wizard.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

const BTN_ID   = "mec-wizard-btn";
const PANEL_ID = "mec-wizard-panel";
const STYLE_ID = "mec-wizard-style";

let _activeWizard = null;  // { steps, idx, template_id }
let _rafTok = null;

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${BTN_ID} {
    position: fixed;
    bottom: 202px;
    right: 16px;
    z-index: 99996;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #1e1e2e;
    border: 1px solid #45475a;
    color: #f9e2af;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
}
#${BTN_ID}:hover { border-color: #f9e2af; }
#${PANEL_ID} {
    position: fixed;
    bottom: 202px;
    right: 64px;
    z-index: 99996;
    width: 380px;
    max-height: 70vh;
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
#${PANEL_ID} h3 { margin: 0 0 8px 0; color: #f9e2af; font-size: 13px; display: flex; justify-content: space-between; }
#${PANEL_ID} .wz-close { background: none; border: none; color: #6c7086; cursor: pointer; font-size: 14px; }
#${PANEL_ID} .wz-card {
    border: 1px solid #313244;
    border-radius: 5px;
    padding: 8px 10px;
    margin-bottom: 6px;
    cursor: pointer;
}
#${PANEL_ID} .wz-card:hover { border-color: #f9e2af; background: #313244; }
#${PANEL_ID} .wz-card .wz-title { font-weight: 600; color: #f9e2af; }
#${PANEL_ID} .wz-card .wz-desc  { color: #cdd6f4; margin-top: 2px; }
#${PANEL_ID} .wz-card .wz-meta  { color: #6c7086; font-size: 10px; margin-top: 4px; }

#${PANEL_ID} .wz-step-title { color: #f9e2af; font-weight: 700; font-size: 14px; }
#${PANEL_ID} .wz-step-hint  { margin: 8px 0; line-height: 1.4; }
#${PANEL_ID} .wz-step-meta  { color: #6c7086; font-size: 10px; margin-bottom: 10px; font-family: monospace; }
#${PANEL_ID} .wz-actions { display: flex; justify-content: space-between; gap: 6px; margin-top: 10px; }
#${PANEL_ID} button.wz-btn {
    background: #313244;
    border: 1px solid #45475a;
    color: #cdd6f4;
    border-radius: 4px;
    padding: 4px 12px;
    cursor: pointer;
    font-size: 11px;
}
#${PANEL_ID} button.wz-btn.primary { border-color: #f9e2af; color: #f9e2af; }
#${PANEL_ID} button.wz-btn:hover   { background: #45475a; }
#${PANEL_ID} .wz-progress {
    height: 4px;
    background: #313244;
    border-radius: 2px;
    overflow: hidden;
    margin: 6px 0 10px 0;
}
#${PANEL_ID} .wz-progress .bar { height: 100%; background: #f9e2af; transition: width 0.2s; }
    `.trim();
    document.head.appendChild(style);
}

function _ensureUi() {
    if (!document.getElementById(BTN_ID)) {
        const b = document.createElement("button");
        b.id = BTN_ID;
        b.title = "Workflow Wizard";
        b.textContent = "🧙";
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
    if (_open) { p.classList.add("visible"); _renderHome(); }
    else       { p.classList.remove("visible"); }
}

async function _renderHome() {
    const p = document.getElementById(PANEL_ID);
    p.innerHTML = `
        <h3>🧙 Workflow Wizard <button class="wz-close">×</button></h3>
        <div class="wz-list"><div style="color:#6c7086">Loading templates…</div></div>
    `;
    p.querySelector(".wz-close").addEventListener("click", _toggle);
    let templates = [];
    try {
        const r = await fetch("/mec/wizard/templates");
        const j = await r.json();
        if (j.success) templates = j.data.templates || [];
    } catch (e) {
        console.warn("[MEC.Wizard] list failed:", e);
    }
    const list = p.querySelector(".wz-list");
    if (!templates.length) {
        list.innerHTML = `<div style="color:#6c7086">No wizard templates installed.</div>`;
        return;
    }
    list.innerHTML = templates.map(t => `
        <div class="wz-card" data-id="${t.id}">
            <div class="wz-title">${t.title}</div>
            <div class="wz-desc">${t.description}</div>
            <div class="wz-meta">${t.step_count} steps · ${t.id}</div>
        </div>
    `).join("");
    list.querySelectorAll(".wz-card").forEach(el => {
        el.addEventListener("click", () => _startWizard(el.getAttribute("data-id")));
    });
}

async function _startWizard(id) {
    let tpl;
    try {
        const r = await fetch(`/mec/wizard/templates/${id}`);
        const j = await r.json();
        if (!j.success) throw new Error(j.error);
        tpl = j.data;
    } catch (e) {
        console.warn("[MEC.Wizard] start failed:", e);
        alert("Could not load wizard template.");
        return;
    }
    _activeWizard = { steps: tpl.steps || [], idx: 0, template_id: tpl.id, title: tpl.title };
    _renderStep();
    _startHighlightLoop();
}

function _endWizard() {
    _activeWizard = null;
    _stopHighlightLoop();
    _renderHome();
}

function _renderStep() {
    const p = document.getElementById(PANEL_ID);
    if (!p || !_activeWizard) return;
    const { steps, idx, title } = _activeWizard;
    const step = steps[idx];
    if (!step) { _endWizard(); return; }
    const pct = ((idx + 1) / steps.length) * 100;
    p.innerHTML = `
        <h3>🧙 ${title.replace(/</g, "&lt;")} <button class="wz-close">×</button></h3>
        <div class="wz-progress"><div class="bar" style="width:${pct}%"></div></div>
        <div class="wz-step-title">${step.title.replace(/</g, "&lt;")}</div>
        <div class="wz-step-meta">node: ${step.node_type || "—"}${step.widget ? "  widget: " + step.widget : ""}</div>
        <div class="wz-step-hint">${(step.hint || "").replace(/</g, "&lt;")}</div>
        <div class="wz-actions">
            <button class="wz-btn wz-prev"  ${idx === 0 ? "disabled" : ""}>← Back</button>
            <button class="wz-btn wz-exit">Exit</button>
            <button class="wz-btn primary wz-next">${idx === steps.length - 1 ? "Finish ✓" : "Next →"}</button>
        </div>
    `;
    p.querySelector(".wz-close").addEventListener("click", () => { _toggle(); _stopHighlightLoop(); _activeWizard = null; });
    p.querySelector(".wz-prev").addEventListener("click", () => { _activeWizard.idx = Math.max(0, idx - 1); _renderStep(); });
    p.querySelector(".wz-exit").addEventListener("click", _endWizard);
    p.querySelector(".wz-next").addEventListener("click", () => {
        if (idx >= steps.length - 1) _endWizard();
        else { _activeWizard.idx = idx + 1; _renderStep(); }
    });
}

function _findTargetNodes() {
    if (!_activeWizard) return [];
    const step = _activeWizard.steps[_activeWizard.idx];
    if (!step || !step.node_type) return [];
    const g = app.graph;
    if (!g || !g._nodes) return [];
    const want = step.node_type.toLowerCase();
    return g._nodes.filter(n => (n.type || "").toLowerCase() === want);
}

function _drawHighlight() {
    if (!_activeWizard) return;
    const canvas = app.canvas;
    if (!canvas || !canvas.ctx) return;
    const targets = _findTargetNodes();
    if (!targets.length) return;
    const ctx = canvas.ctx;
    ctx.save();
    const t = (performance.now() % 1500) / 1500;
    const alpha = 0.4 + 0.5 * Math.abs(Math.sin(t * Math.PI * 2));
    ctx.lineWidth = 4;
    ctx.strokeStyle = `rgba(249, 226, 175, ${alpha.toFixed(3)})`;
    for (const n of targets) {
        const [x, y] = n.pos;
        const [w, h] = n.size;
        ctx.strokeRect(x - 6, y - 6 - (n.flags?.collapsed ? 0 : 0), w + 12, (n.flags?.collapsed ? 30 : h) + 12);
    }
    ctx.restore();
}

function _startHighlightLoop() {
    _stopHighlightLoop();
    const tick = () => {
        if (!_activeWizard) return;
        app.canvas?.setDirty?.(true, true);
        _rafTok = requestAnimationFrame(tick);
    };
    _rafTok = requestAnimationFrame(tick);
}

function _stopHighlightLoop() {
    if (_rafTok) { cancelAnimationFrame(_rafTok); _rafTok = null; }
}

function _patchDrawForeground() {
    const orig = LGraphCanvas.prototype.onDrawForeground;
    LGraphCanvas.prototype.onDrawForeground = function (ctx) {
        if (orig) orig.call(this, ctx);
        try { _drawHighlight(); }
        catch (e) { /* never break drawing */ }
    };
    LGraphCanvas.prototype.onDrawForeground._mecWizardPatched = true;
}

app.registerExtension({
    name: "MEC.WorkflowWizard",
    settings: [
        {
            id: "mec.workflow_wizard.enabled",
            name: "Workflow Wizard: enabled",
            tooltip: "Show 🧙 button to launch step-by-step workflow wizards.",
            type: "boolean",
            default: true,
            onChange: (v) => {
                const b = document.getElementById(BTN_ID);
                if (b) b.style.display = v ? "flex" : "none";
            },
        },
    ],
    async setup() {
        _injectStyle();
        _ensureUi();
        if (!LGraphCanvas.prototype.onDrawForeground?._mecWizardPatched) {
            _patchDrawForeground();
        }
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("mec.workflow_wizard.enabled", true); }
            catch { return true; }
        })();
        const b = document.getElementById(BTN_ID);
        if (b) b.style.display = enabled ? "flex" : "none";
        console.log("[MEC.WorkflowWizard] Loaded.");
    },
});
