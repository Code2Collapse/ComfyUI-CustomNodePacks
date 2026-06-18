/**
 * c2c_cost_estimator.js — Cost Estimator (god-level rebuild, 2026-05-27)
 *
 * Floating 💰 button. Side panel with a render-cost estimate AND a real
 * "cheaper alternative" advisor that calls streamAI with the workflow
 * profile and surfaces concrete swap suggestions (model / sampler / steps
 * / resolution / scheduler).
 *
 * Features:
 *   1) Summary cards (total time, peak VRAM, node count)
 *   2) Per-node breakdown table sorted by time, with cold-run italics
 *   3) Warnings from backend
 *   4) Top-3 bottleneck highlighter
 *   5) "What-if" simulator: steps × res × batch sliders → recomputed total
 *   6) AI cheaper-alt suggestions (streamAI feature="cost_optimize")
 *   7) Export markdown / JSON report
 *   8) Body-only re-render preserves chrome; listener registry
 *   9) Refresh button re-pulls cost estimate without closing panel
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { attachWindowChrome } from "./_c2c_window.js";
import { streamAI } from "./_c2c_ai_client.js";

const BTN_ID    = "mec-cost-btn";
const PANEL_ID  = "mec-cost-panel";
const STYLE_ID  = "mec-cost-style";

const VRAM_HINTS = {
    KSampler: 2400, KSamplerAdvanced: 2400, SamplerCustom: 2400, SamplerCustomAdvanced: 2400,
    CheckpointLoaderSimple: 4000, UNETLoader: 3500, VAEDecode: 800, VAEEncode: 600,
    CLIPTextEncode: 200, ControlNetApplyAdvanced: 1200, LoraLoader: 400,
};

const _state = {
    open: false,
    data: null,
    workflow: null,
    busy: false,
    aiOutput: "",
    aiBusy: false,
    whatIf: { stepsMul: 1.0, resMul: 1.0, batchMul: 1.0 },
    live: {
        active: false,
        startTs: 0,
        elapsed: 0,
        promptId: null,
        nodeTimings: [],
        currentNode: null,
        nodeStartTs: 0,
        timerHandle: null,
    },
    runHistory: [],
};
const _listeners = [];
function _on(t, e, fn, opts) { t.addEventListener(e, fn, opts); _listeners.push(() => t.removeEventListener(e, fn, opts)); }
function _clear() { while (_listeners.length) { try { _listeners.pop()(); } catch {} } }

function _onExecStart(ev) {
    const L = _state.live;
    L.active = true;
    L.startTs = performance.now();
    L.elapsed = 0;
    L.promptId = ev?.detail?.prompt_id || null;
    L.nodeTimings = [];
    L.currentNode = null;
    L.nodeStartTs = 0;
    clearInterval(L.timerHandle);
    L.timerHandle = setInterval(() => {
        if (!L.active) { clearInterval(L.timerHandle); return; }
        L.elapsed = performance.now() - L.startTs;
        if (_state.open) _renderLiveBar();
    }, 250);
    if (_state.open) _renderBody();
}

function _onExecuted(ev) {
    const L = _state.live;
    if (!L.active) return;
    const d = ev?.detail;
    const nodeId = d?.node;
    if (L.currentNode) {
        const dur = performance.now() - L.nodeStartTs;
        L.nodeTimings.push({ ...L.currentNode, ms: dur });
    }
    const g = app.graph;
    const gNode = g?.getNodeById?.(nodeId);
    const classType = gNode?.type || gNode?.comfyClass || `node_${nodeId}`;
    L.currentNode = { nodeId, classType, vram: VRAM_HINTS[classType] || 0 };
    L.nodeStartTs = performance.now();
    if (_state.open) _renderLiveBar();
}

function _onExecDone() {
    const L = _state.live;
    if (!L.active) return;
    if (L.currentNode) {
        const dur = performance.now() - L.nodeStartTs;
        L.nodeTimings.push({ ...L.currentNode, ms: dur });
        L.currentNode = null;
    }
    L.elapsed = performance.now() - L.startTs;
    L.active = false;
    clearInterval(L.timerHandle);
    _state.runHistory.push({
        ts: Date.now(),
        total_ms: L.elapsed,
        node_count: L.nodeTimings.length,
        peak_vram: Math.max(0, ...L.nodeTimings.map(n => n.vram)),
        nodes: L.nodeTimings.slice(),
    });
    if (_state.runHistory.length > 50) _state.runHistory.shift();
    if (_state.open) _renderBody();
}

function _renderLiveBar() {
    const panel = document.getElementById(PANEL_ID);
    if (!panel) return;
    let bar = panel.querySelector('[data-role="live-bar"]');
    if (!bar) return;
    const L = _state.live;
    const elapsed = L.active ? (performance.now() - L.startTs) : L.elapsed;
    const done = L.nodeTimings.length;
    const cur = L.currentNode ? ` → ${L.currentNode.classType}` : "";
    const vramEst = L.nodeTimings.reduce((s, n) => s + n.vram, 0) + (L.currentNode?.vram || 0);
    bar.innerHTML = `<div style="display:flex;align-items:center;gap:8px;padding:6px 0;">
        <span style="color:var(--c2c-okSoft);font-weight:700;">${L.active ? "⏱ LIVE" : "✓ Done"}</span>
        <span>${_fmtMs(elapsed)}</span>
        <span style="color:var(--c2c-overlay0);">${done} node(s)${cur}</span>
        <span style="color:var(--c2c-yellow);">~${(vramEst / 1024).toFixed(1)} GB VRAM</span>
    </div>`;
}

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${BTN_ID} {
    position: fixed; bottom: 156px; right: 16px;
    z-index: var(--c2c-z-hud, 1000);
    width: 38px; height: 38px; border-radius: 50%;
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface1); color: var(--c2c-peach);
    font-size: 16px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex; align-items: center; justify-content: center;
}
#${BTN_ID}:hover { border-color: var(--c2c-peach); }

#${PANEL_ID} {
    position: fixed; top: 80px; right: 80px;
    width: min(94vw, 540px); height: min(82vh, 740px);
    background: var(--c2c-bg2); border: 1px solid var(--c2c-surface1); border-radius: 8px;
    color: var(--c2c-fg); font-family: -apple-system, "Segoe UI", sans-serif; font-size: 12px;
    box-shadow: 0 6px 32px rgba(0,0,0,0.7);
    display: none; flex-direction: column; overflow: hidden;
}
#${PANEL_ID}.visible { display: flex; }
#${PANEL_ID} h3 {
    margin: 0; padding: 8px 12px; color: var(--c2c-peach); font-size: 14px;
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(180deg,var(--c2c-bg),var(--c2c-bg2)); border-bottom: 1px solid var(--c2c-surface0);
}
#${PANEL_ID} .ce-body { flex: 1 1 auto; min-height: 0; overflow: auto; padding: 10px; }
#${PANEL_ID} .ce-close { background: none; border: none; color: var(--c2c-overlay0); cursor: pointer; font-size: 16px; padding: 0 4px; }

#${PANEL_ID} .ce-toolbar { display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
#${PANEL_ID} .ce-toolbar button {
    background: var(--c2c-surface0); border: 1px solid var(--c2c-surface1); color: var(--c2c-fg); border-radius: 4px;
    padding: 3px 8px; font-size: 11px; cursor: pointer;
}
#${PANEL_ID} .ce-toolbar button:hover { border-color: var(--c2c-peach); }
#${PANEL_ID} .ce-toolbar button[disabled] { opacity: 0.5; cursor: not-allowed; }

#${PANEL_ID} .ce-summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 10px; }
#${PANEL_ID} .ce-card { background: var(--c2c-surface0); border-radius: 5px; padding: 6px 8px; text-align: center; }
#${PANEL_ID} .ce-card .v { font-size: 16px; font-weight: 700; color: var(--c2c-okSoft); font-family: monospace; }
#${PANEL_ID} .ce-card .l { font-size: 10px; color: var(--c2c-overlay0); text-transform: uppercase; }

#${PANEL_ID} .ce-warn {
    background: var(--c2c-warnBg2); border-left: 3px solid var(--c2c-yellow);
    padding: 4px 8px; margin-bottom: 6px; font-size: 11px; color: var(--c2c-yellow);
}

#${PANEL_ID} .ce-section { margin-top: 12px; }
#${PANEL_ID} .ce-section h4 { margin: 0 0 6px 0; color: var(--c2c-mauve); font-size: 12px; }

#${PANEL_ID} table { width: 100%; border-collapse: collapse; font-size: 11px; }
#${PANEL_ID} th, #${PANEL_ID} td { text-align: left; padding: 4px 6px; border-bottom: 1px solid var(--c2c-surface0); }
#${PANEL_ID} th { color: var(--c2c-overlay0); font-weight: 600; }
#${PANEL_ID} td.right { text-align: right; font-family: monospace; }
#${PANEL_ID} td.bar { width: 60px; padding: 0 6px; }
#${PANEL_ID} .ce-bar { height: 4px; background: var(--c2c-yellow); border-radius: 2px; }
#${PANEL_ID} tr.bottleneck td:first-child { color: var(--c2c-red); font-weight: 600; }
#${PANEL_ID} .ce-cold { color: var(--c2c-peach); font-style: italic; }

#${PANEL_ID} .ce-whatif {
    background: var(--c2c-bg3); border: 1px solid var(--c2c-surface0); border-radius: 4px;
    padding: 8px; display: grid; grid-template-columns: auto 1fr auto; gap: 4px 8px; align-items: center;
}
#${PANEL_ID} .ce-whatif label { font-size: 11px; color: var(--c2c-fg); }
#${PANEL_ID} .ce-whatif input[type="range"] { width: 100%; }
#${PANEL_ID} .ce-whatif .ce-wf-val { color: var(--c2c-okSoft); font-family: monospace; font-size: 11px; min-width: 60px; text-align: right; }
#${PANEL_ID} .ce-whatif .ce-wf-result {
    grid-column: 1 / -1; margin-top: 6px; padding: 6px; background: var(--c2c-bg);
    border-radius: 3px; color: var(--c2c-mauve); font-family: monospace; font-size: 12px; text-align: center;
}

#${PANEL_ID} .ce-ai-box {
    margin-top: 6px; padding: 8px 10px; background: var(--c2c-bg);
    border: 1px solid var(--c2c-surface0); border-radius: 4px; font-size: 11px;
    line-height: 1.5; white-space: pre-wrap;
}
#${PANEL_ID} .ce-ai-box.empty { color: var(--c2c-overlay0); font-style: italic; }
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
        p.innerHTML = `
            <h3><span>💰 Render cost</span><button class="ce-close" title="Close">×</button></h3>
            <div class="ce-body" data-role="body"></div>
        `;
        document.body.appendChild(p);
        p.querySelector(".ce-close").addEventListener("click", _toggle);
        attachWindowChrome(p, { storageKey: "cost_estimator", headerSelector: "h3", titleSelector: "h3 > span", minW: 360, minH: 320 });
    }
}

function _toggle() {
    _state.open = !_state.open;
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    if (_state.open) { p.classList.add("visible"); _refresh(); }
    else { p.classList.remove("visible"); _clear(); }
}

async function _getWorkflowApi() {
    try {
        const r = await app.graphToPrompt();
        return r?.output || null;
    } catch (e) {
        console.warn("[C2C.CostEstimator] graphToPrompt failed:", e);
        return null;
    }
}

function _fmtMs(ms) {
    if (ms < 1000) return `${ms.toFixed(0)} ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(2)} s`;
    return `${Math.floor(ms / 60000)}m ${((ms % 60000) / 1000).toFixed(1)}s`;
}

function _esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));
}

async function _refresh() {
    _state.busy = true;
    _renderBody();
    const wf = await _getWorkflowApi();
    _state.workflow = wf;
    if (!wf || Object.keys(wf).length === 0) {
        _state.data = null;
        _state.busy = false;
        _renderBody();
        return;
    }
    try {
        const r = await fetch("/mec/cost_estimate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ workflow: wf }),
        });
        const j = await r.json();
        if (!j.success) throw new Error(j.error || "fail");
        _state.data = j.data;
    } catch (e) {
        console.warn("[C2C.CostEstimator] estimate failed:", e);
        _state.data = { _error: String(e?.message || e) };
    }
    _state.busy = false;
    _renderBody();
}

function _renderBody() {
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    const body = p.querySelector('[data-role="body"]');
    if (!body) return;
    _clear();

    const L = _state.live;
    const hist = _state.runHistory;
    const avgMs = hist.length ? hist.reduce((s, r) => s + r.total_ms, 0) / hist.length : 0;

    const liveSection = `<div data-role="live-bar" style="border-bottom:1px solid var(--c2c-surface0);margin-bottom:6px;">${
        L.active
            ? `<div style="display:flex;align-items:center;gap:8px;padding:6px 0;">
                <span style="color:var(--c2c-okSoft);font-weight:700;">⏱ LIVE</span>
                <span>${_fmtMs(L.elapsed)}</span>
                <span style="color:var(--c2c-overlay0);">${L.nodeTimings.length} node(s)${L.currentNode ? ` → ${L.currentNode.classType}` : ""}</span>
               </div>`
            : hist.length
                ? `<div style="display:flex;align-items:center;gap:8px;padding:6px 0;flex-wrap:wrap;">
                    <span style="color:var(--c2c-peach);font-weight:600;">📊 ${hist.length} run(s)</span>
                    <span>avg: ${_fmtMs(avgMs)}</span>
                    <span style="color:var(--c2c-overlay0);">last: ${_fmtMs(hist[hist.length - 1].total_ms)} · ${hist[hist.length - 1].node_count} nodes</span>
                   </div>`
                : `<div style="padding:6px 0;color:var(--c2c-overlay0);font-style:italic;">Queue a prompt to start live tracking…</div>`
    }</div>`;

    const tb = `
        ${liveSection}
        <div class="ce-toolbar">
            <button data-act="refresh">🔄 Refresh</button>
            <button data-act="ai" ${_state.data?._error || !_state.data ? "disabled" : ""}>${_state.aiBusy ? "⌛ AI…" : "✨ Cheaper-alt"}</button>
            <button data-act="export-md" ${!_state.data || _state.data._error ? "disabled" : ""}>⇩ Markdown</button>
            <button data-act="export-json" ${!_state.data || _state.data._error ? "disabled" : ""}>⇩ JSON</button>
        </div>`;

    if (_state.busy) { body.innerHTML = tb + `<div style="color:var(--c2c-overlay0);padding:20px;text-align:center;">Computing estimate…</div>`; _bindToolbar(body); return; }
    if (!_state.workflow) { body.innerHTML = tb + `<div class="ce-warn">No workflow on canvas.</div>`; _bindToolbar(body); return; }
    if (_state.data?._error) { body.innerHTML = tb + `<div class="ce-warn">Estimate failed: ${_esc(_state.data._error)}</div>`; _bindToolbar(body); return; }

    const d = _state.data;
    if (!d) { body.innerHTML = tb + `<div class="ce-warn">No data.</div>`; _bindToolbar(body); return; }

    const sorted = (d.per_node || []).slice().sort((a, b) => b.ms - a.ms);
    const maxMs = Math.max(1, ...sorted.map((r) => r.ms));
    const bottleneckIds = new Set(sorted.slice(0, 3).map((r) => r.class_type));

    const rows = sorted.map((r) => {
        const pct = (r.ms / maxMs) * 100;
        const cls = r.samples === 0 ? "ce-cold" : "";
        const bt = bottleneckIds.has(r.class_type) ? "bottleneck" : "";
        return `<tr class="${bt}">
            <td class="${cls}">${_esc(r.class_type)}</td>
            <td class="right">${_fmtMs(r.ms)}</td>
            <td class="right">${r.vram_mb ? r.vram_mb.toFixed(0) + " MB" : "—"}</td>
            <td class="bar"><div class="ce-bar" style="width:${pct}%"></div></td>
            <td class="right">${r.samples}</td>
        </tr>`;
    }).join("");

    const wf = _state.whatIf;
    const factor = wf.stepsMul * wf.resMul * wf.resMul * wf.batchMul;
    const projTotal = d.total_ms * factor;
    const projVram = d.peak_vram_mb * wf.resMul * wf.resMul * wf.batchMul;

    body.innerHTML = tb + `
        <div class="ce-summary">
            <div class="ce-card"><div class="v">${_fmtMs(d.total_ms)}</div><div class="l">Total time</div></div>
            <div class="ce-card"><div class="v">${d.peak_vram_mb.toFixed(0)} MB</div><div class="l">Peak VRAM</div></div>
            <div class="ce-card"><div class="v">${(d.per_node || []).length}</div><div class="l">Nodes</div></div>
        </div>
        ${(d.warnings || []).map((w) => `<div class="ce-warn">⚠ ${_esc(w)}</div>`).join("")}

        <div class="ce-section">
            <h4>Per-node breakdown</h4>
            <table>
                <thead><tr><th>Node class</th><th class="right">Time</th><th class="right">VRAM</th><th>Δ</th><th class="right">n</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
            <div style="margin-top:6px;font-size:10px;color:var(--c2c-overlay0);">History: ${d.samples_in_history} samples · italic = no prior runs · pink = bottleneck.</div>
        </div>

        <div class="ce-section">
            <h4>What-if simulator</h4>
            <div class="ce-whatif">
                <label>steps ×</label>
                <input type="range" min="0.25" max="4" step="0.05" value="${wf.stepsMul}" data-wf="stepsMul">
                <div class="ce-wf-val" data-role="v-stepsMul">${wf.stepsMul.toFixed(2)}×</div>

                <label>res  ×</label>
                <input type="range" min="0.25" max="2" step="0.05" value="${wf.resMul}" data-wf="resMul">
                <div class="ce-wf-val" data-role="v-resMul">${wf.resMul.toFixed(2)}×</div>

                <label>batch×</label>
                <input type="range" min="0.25" max="8" step="0.25" value="${wf.batchMul}" data-wf="batchMul">
                <div class="ce-wf-val" data-role="v-batchMul">${wf.batchMul.toFixed(2)}×</div>

                <div class="ce-wf-result">Projected: <b>${_fmtMs(projTotal)}</b> · VRAM ≈ <b>${projVram.toFixed(0)} MB</b></div>
            </div>
        </div>

        <div class="ce-section">
            <h4>✨ AI cheaper-alt suggestions</h4>
            <div class="ce-ai-box ${_state.aiOutput ? "" : "empty"}" data-role="ai">${_state.aiOutput ? _esc(_state.aiOutput) : "Click ✨ Cheaper-alt to analyze."}</div>
        </div>
    `;
    _bindToolbar(body);
    _bindWhatIf(body);
}

function _bindToolbar(body) {
    const r = body.querySelector('[data-act="refresh"]');
    if (r) _on(r, "click", () => _refresh());
    const ai = body.querySelector('[data-act="ai"]');
    if (ai) _on(ai, "click", () => _aiSuggest(body));
    const md = body.querySelector('[data-act="export-md"]');
    if (md) _on(md, "click", () => _exportMarkdown());
    const js = body.querySelector('[data-act="export-json"]');
    if (js) _on(js, "click", () => _exportJson());
}

function _bindWhatIf(body) {
    for (const inp of body.querySelectorAll('input[data-wf]')) {
        const key = inp.dataset.wf;
        _on(inp, "input", () => {
            _state.whatIf[key] = parseFloat(inp.value);
            const v = body.querySelector(`[data-role="v-${key}"]`);
            if (v) v.textContent = `${_state.whatIf[key].toFixed(2)}×`;
            const d = _state.data;
            if (d && !d._error) {
                const wf = _state.whatIf;
                const factor = wf.stepsMul * wf.resMul * wf.resMul * wf.batchMul;
                const projTotal = d.total_ms * factor;
                const projVram = d.peak_vram_mb * wf.resMul * wf.resMul * wf.batchMul;
                const res = body.querySelector(".ce-wf-result");
                if (res) res.innerHTML = `Projected: <b>${_fmtMs(projTotal)}</b> · VRAM ≈ <b>${projVram.toFixed(0)} MB</b>`;
            }
        });
    }
}

async function _aiSuggest(body) {
    if (_state.aiBusy || !_state.data || _state.data._error) return;
    _state.aiBusy = true;
    _state.aiOutput = "";
    const aiEl = body.querySelector('[data-role="ai"]');
    if (aiEl) { aiEl.classList.remove("empty"); aiEl.textContent = "Analyzing workflow for cheaper alternatives…"; }

    const top = _state.data.per_node.slice().sort((a, b) => b.ms - a.ms).slice(0, 8);
    const profile = top.map((r) => `${r.class_type}: ${_fmtMs(r.ms)}${r.vram_mb ? ` / ${r.vram_mb.toFixed(0)}MB` : ""}`).join("\n");
    const ksamplers = [];
    for (const [id, n] of Object.entries(_state.workflow || {})) {
        if (/Sampler|KSampler/i.test(n.class_type)) {
            const inp = n.inputs || {};
            ksamplers.push(`${n.class_type}: steps=${inp.steps} cfg=${inp.cfg} sampler=${inp.sampler_name} sched=${inp.scheduler} denoise=${inp.denoise}`);
        }
    }

    try {
        await streamAI({
            feature: "cost_optimize",
            sensitivity: "normal",
            max_tokens: 500,
            temperature: 0.4,
            messages: [
                { role: "system", content: "You are a ComfyUI optimization advisor. Given a workflow profile, suggest 3-5 concrete CHEAPER alternatives: smaller model, fewer steps, lower CFG, different sampler/scheduler, lower res, batch tweaks. Each suggestion: 1 short sentence + expected % savings. Be concrete." },
                { role: "user", content: `Total: ${_fmtMs(_state.data.total_ms)}, peak VRAM ${_state.data.peak_vram_mb.toFixed(0)} MB.\n\nTop nodes by time:\n${profile}\n\nSamplers:\n${ksamplers.join("\n") || "(none)"}` },
            ],
            onChunk: (c) => { _state.aiOutput += c; if (aiEl) aiEl.textContent = _state.aiOutput; },
            onError: (e) => { _state.aiOutput += `\n[error: ${e}]`; if (aiEl) aiEl.textContent = _state.aiOutput; },
            onDone: () => {},
        });
    } catch (e) {
        _state.aiOutput += `\n[exception: ${e?.message || e}]`;
        if (aiEl) aiEl.textContent = _state.aiOutput;
    } finally {
        _state.aiBusy = false;
        const btn = document.querySelector(`#${PANEL_ID} [data-act="ai"]`);
        if (btn) btn.textContent = "✨ Cheaper-alt";
    }
}

function _exportMarkdown() {
    if (!_state.data || _state.data._error) return;
    const d = _state.data;
    const sorted = d.per_node.slice().sort((a, b) => b.ms - a.ms);
    let md = `# Render cost report\n\n- **Total time:** ${_fmtMs(d.total_ms)}\n- **Peak VRAM:** ${d.peak_vram_mb.toFixed(0)} MB\n- **Nodes:** ${d.per_node.length}\n- **History samples:** ${d.samples_in_history}\n\n`;
    if (d.warnings?.length) md += `## Warnings\n${d.warnings.map(w => `- ⚠ ${w}`).join("\n")}\n\n`;
    md += `## Per-node breakdown\n\n| Class | Time | VRAM | Samples |\n|---|---:|---:|---:|\n`;
    for (const r of sorted) md += `| ${r.class_type} | ${_fmtMs(r.ms)} | ${r.vram_mb ? r.vram_mb.toFixed(0) + ' MB' : '—'} | ${r.samples} |\n`;
    if (_state.aiOutput) md += `\n## AI suggestions\n\n${_state.aiOutput}\n`;
    const blob = new Blob([md], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `cost_report_${Date.now()}.md`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function _exportJson() {
    if (!_state.data || _state.data._error) return;
    const blob = new Blob([JSON.stringify({ estimate: _state.data, whatIf: _state.whatIf, ai: _state.aiOutput }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `cost_report_${Date.now()}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

app.registerExtension({
    name: "C2C.CostEstimator",
    settings: [
        {
            id: "mec.cost_estimator.enabled",
            name: "Cost Estimator: enabled",
            tooltip: "Show 💰 button to predict render time before queueing.",
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
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("mec.cost_estimator.enabled", true); }
            catch { return true; }
        })();
        const b = document.getElementById(BTN_ID);
        if (b) b.style.display = enabled ? "flex" : "none";
        api.addEventListener("execution_start", _onExecStart);
        api.addEventListener("executed", _onExecuted);
        api.addEventListener("execution_success", _onExecDone);
        api.addEventListener("execution_error", _onExecDone);
        console.log("[C2C.CostEstimator] live-tracking loaded.");
    },
});
