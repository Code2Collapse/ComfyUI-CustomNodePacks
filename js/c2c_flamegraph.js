/**
 * c2c_flamegraph.js — Execution Flame Graph (god-level rebuild, 2026-05-27)
 *
 * Floating ⏱ button. buildPanel-backed window showing a per-node flame
 * graph for the latest run AND a real two-run comparison (snapshot A vs
 * current B) with per-node delta highlighting.
 *
 * Features:
 *   1) Latest-run flame bars (heat-colored by relative time)
 *   2) Search/filter rows by class or id
 *   3) Sort by time / class / vram
 *   4) Snapshot current run to slot A or B
 *   5) Compare mode: aligned A↔B with Δ ms + Δ% column, red=slower / green=faster
 *   6) AI insight via streamAI (feature="flame_compare") — explains regressions
 *   7) Copy CSV / Copy comparison-CSV
 *   8) Auto-open setting + auto-refresh on every execution_success
 *   9) buildPanel-backed (chrome safe); explicit body-only render + listener registry
 */

import { app } from "../../scripts/app.js";
import { LITE } from "./_c2c_lite.js";
import { api } from "../../scripts/api.js";
import { buildPanel } from "./_c2c_window.js";
import { streamAI } from "./_c2c_ai_client.js";

const STYLE_ID = "mec-flamegraph-style";
const PANEL_ID = "c2c-flame";
const BTN_ID   = "mec-flamegraph-btn";

const _state = {
    open: false,
    current: null,
    A: null,
    B: null,
    mode: "current",
    sortBy: "time",
    filter: "",
    aiOutput: "",
    aiBusy: false,
    live: {
        active: false,
        promptId: null,
        startTs: 0,
        currentNodeId: null,
        currentNodeClass: null,
        nodeStartTs: 0,
        completedRows: [],
        progress: { node: null, value: 0, max: 0 },
        timerHandle: null,
    },
};
const _listeners = [];
function _on(t, e, fn, opts) { t.addEventListener(e, fn, opts); _listeners.push(() => t.removeEventListener(e, fn, opts)); }
function _clear() { while (_listeners.length) { try { _listeners.pop()(); } catch {} } }

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
#${BTN_ID} {
    position: fixed; bottom: 60px; right: 16px;
    z-index: var(--c2c-z-hud, 1000);
    width: 38px; height: 38px; border-radius: 50%;
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface1); color: var(--c2c-peach);
    font-size: 18px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex; align-items: center; justify-content: center;
}
#${BTN_ID}:hover { border-color: var(--c2c-blue); color: var(--c2c-blue); }

#${PANEL_ID} .fg-tools {
    display: flex; gap: 6px; flex-wrap: wrap; align-items: center;
    padding: 6px 0; margin-bottom: 6px; border-bottom: 1px solid var(--c2c-surface0);
}
#${PANEL_ID} .fg-tools button, #${PANEL_ID} .fg-tools select, #${PANEL_ID} .fg-tools input {
    background: var(--c2c-surface0); border: 1px solid var(--c2c-surface1); color: var(--c2c-fg);
    border-radius: 4px; padding: 3px 7px; font-size: 11px;
}
#${PANEL_ID} .fg-tools button { cursor: pointer; }
#${PANEL_ID} .fg-tools button:hover { border-color: var(--c2c-blue); color: var(--c2c-blue); }
#${PANEL_ID} .fg-tools button[disabled] { opacity: 0.4; cursor: not-allowed; }
#${PANEL_ID} .fg-tools input { width: 100px; }
#${PANEL_ID} .fg-tools .fg-slot {
    font-size: 10px; color: var(--c2c-overlay0); padding: 2px 5px; border-radius: 3px; background: var(--c2c-bg);
}
#${PANEL_ID} .fg-tools .fg-slot.set { color: var(--c2c-okSoft); }

#${PANEL_ID} .fg-row {
    display: grid; grid-template-columns: 1fr 70px;
    align-items: center; gap: 8px; padding: 3px 0;
    border-bottom: 1px solid var(--c2c-surface0);
}
#${PANEL_ID} .fg-bar {
    position: relative; height: 20px; background: var(--c2c-surface0);
    border-radius: 3px; overflow: hidden;
}
#${PANEL_ID} .fg-bar-fill { height: 100%; border-radius: 3px; }
#${PANEL_ID} .fg-bar-label {
    position: absolute; top: 0; left: 6px; line-height: 20px;
    font-size: 11px; color: var(--c2c-fg); white-space: nowrap;
    text-shadow: 0 0 3px rgba(0,0,0,0.7);
}
#${PANEL_ID} .fg-time { font-size: 11px; color: var(--c2c-okSoft); text-align: right; font-variant-numeric: tabular-nums; }
#${PANEL_ID} .fg-row.fg-error .fg-bar-fill { background: var(--c2c-red) !important; }
#${PANEL_ID} .fg-row.fg-error .fg-time { color: var(--c2c-red); }
#${PANEL_ID} .fg-empty {
    color: var(--c2c-overlay0); font-style: italic; padding: 12px 0; text-align: center;
}

#${PANEL_ID} .fg-cmp-row {
    display: grid; grid-template-columns: 1fr 60px 60px 70px;
    align-items: center; gap: 6px; padding: 3px 0;
    border-bottom: 1px solid var(--c2c-surface0); font-size: 11px;
}
#${PANEL_ID} .fg-cmp-row .fg-cmp-cls { color: var(--c2c-fg); }
#${PANEL_ID} .fg-cmp-row .fg-cmp-A { color: var(--c2c-sapphire); text-align: right; font-family: monospace; }
#${PANEL_ID} .fg-cmp-row .fg-cmp-B { color: var(--c2c-mauve); text-align: right; font-family: monospace; }
#${PANEL_ID} .fg-cmp-row .fg-cmp-D { text-align: right; font-family: monospace; font-weight: 700; }
#${PANEL_ID} .fg-cmp-row .fg-cmp-D.up   { color: var(--c2c-red); }
#${PANEL_ID} .fg-cmp-row .fg-cmp-D.down { color: var(--c2c-okSoft); }
#${PANEL_ID} .fg-cmp-row .fg-cmp-D.flat { color: var(--c2c-overlay0); }
#${PANEL_ID} .fg-cmp-row .fg-cmp-D.new  { color: var(--c2c-yellow); }
#${PANEL_ID} .fg-cmp-row .fg-cmp-D.gone { color: var(--c2c-overlay0); font-style: italic; }

#${PANEL_ID} .fg-ai {
    margin-top: 8px; padding: 8px 10px; background: var(--c2c-bg);
    border: 1px solid var(--c2c-surface0); border-radius: 4px; font-size: 11px;
    line-height: 1.5; white-space: pre-wrap;
}
#${PANEL_ID} .fg-ai.empty { color: var(--c2c-overlay0); font-style: italic; }
    `.trim();
    document.head.appendChild(s);
}

function _heatColor(pct) {
    const r = Math.round(166 + (243 - 166) * pct);
    const g = Math.round(227 + (139 - 227) * pct);
    const b = Math.round(161 + (168 - 161) * pct);
    return `rgb(${r},${g},${b})`;
}

function _esc(s) {
    return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function _sortRows(rows) {
    const cmp = {
        time:  (a, b) => b.elapsed_ms - a.elapsed_ms,
        class: (a, b) => String(a.node_class).localeCompare(String(b.node_class)),
        vram:  (a, b) => (b.vram_peak_mb || 0) - (a.vram_peak_mb || 0),
    }[_state.sortBy] || ((a, b) => b.elapsed_ms - a.elapsed_ms);
    return rows.slice().sort(cmp);
}

function _filterRows(rows) {
    const q = _state.filter.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter((r) =>
        String(r.node_class).toLowerCase().includes(q) ||
        String(r.node_id).toLowerCase().includes(q)
    );
}

function _liveExecStart(ev) {
    const L = _state.live;
    L.active = true;
    L.promptId = ev?.detail?.prompt_id || null;
    L.startTs = performance.now();
    L.completedRows = [];
    L.currentNodeId = null;
    L.currentNodeClass = null;
    L.nodeStartTs = 0;
    L.progress = { node: null, value: 0, max: 0 };
    clearInterval(L.timerHandle);
    L.timerHandle = setInterval(() => {
        if (!L.active) { clearInterval(L.timerHandle); return; }
        if (_state.open) _renderPanel();
    }, 500);
    if (_state.open) _renderPanel();
}

function _liveExecuted(ev) {
    const L = _state.live;
    if (!L.active) return;
    const d = ev?.detail;
    const nodeId = d?.node;
    if (L.currentNodeId != null) {
        const dur = performance.now() - L.nodeStartTs;
        L.completedRows.push({
            node_id: String(L.currentNodeId),
            node_class: L.currentNodeClass,
            elapsed_ms: dur,
            vram_peak_mb: 0,
            error: false,
        });
    }
    const g = app.graph;
    const gNode = g?.getNodeById?.(nodeId);
    L.currentNodeId = nodeId;
    L.currentNodeClass = gNode?.type || gNode?.comfyClass || `node_${nodeId}`;
    L.nodeStartTs = performance.now();
    L.progress = { node: nodeId, value: 0, max: 0 };
    if (_state.open) _renderPanel();
}

function _liveProgress(ev) {
    const d = ev?.detail;
    if (!d || !_state.live.active) return;
    _state.live.progress = { node: d.node, value: d.value || 0, max: d.max || 0 };
}

function _liveExecDone(ev) {
    const L = _state.live;
    if (!L.active) return;
    if (L.currentNodeId != null) {
        const dur = performance.now() - L.nodeStartTs;
        L.completedRows.push({
            node_id: String(L.currentNodeId),
            node_class: L.currentNodeClass,
            elapsed_ms: dur,
            vram_peak_mb: 0,
            error: !!ev?.detail?.exception_message,
        });
    }
    const totalMs = performance.now() - L.startTs;
    L.active = false;
    clearInterval(L.timerHandle);
    _state.current = {
        rows: L.completedRows.slice(),
        total_ms: totalMs,
        node_count: L.completedRows.length,
        prompt_id: L.promptId,
    };
    if (_state.open) _renderPanel();
}

function _renderToolbar() {
    return `
        <div class="fg-tools">
            <button data-act="refresh">🔄</button>
            <input data-role="filter" placeholder="filter…" value="${_esc(_state.filter)}">
            <select data-role="sort">
                <option value="time"${_state.sortBy==="time"?" selected":""}>sort: time</option>
                <option value="class"${_state.sortBy==="class"?" selected":""}>sort: class</option>
                <option value="vram"${_state.sortBy==="vram"?" selected":""}>sort: vram</option>
            </select>
            <span style="color:var(--c2c-overlay0);">|</span>
            <button data-act="snap-a">⇩ A</button>
            <button data-act="snap-b">⇩ B</button>
            <span class="fg-slot ${_state.A?"set":""}" title="${_state.A?_state.A.prompt_id||"":""}">A:${_state.A?(_state.A.total_ms/1000).toFixed(2)+"s":"—"}</span>
            <span class="fg-slot ${_state.B?"set":""}" title="${_state.B?_state.B.prompt_id||"":""}">B:${_state.B?(_state.B.total_ms/1000).toFixed(2)+"s":"—"}</span>
            <button data-act="mode" ${(!_state.A||!_state.B)?"disabled":""}>${_state.mode==="compare"?"current":"compare A↔B"}</button>
            <span style="color:var(--c2c-overlay0);">|</span>
            <button data-act="ai" ${(_state.mode!=="compare"||!_state.A||!_state.B)?"disabled":""}>${_state.aiBusy?"⌛":"✨ Explain"}</button>
            <button data-act="csv">CSV</button>
        </div>
    `;
}

function _renderCurrent(body, data) {
    const L = _state.live;
    if (L.active) {
        const elapsed = performance.now() - L.startTs;
        const done = L.completedRows;
        const allRows = [...done];
        if (L.currentNodeId != null) {
            allRows.push({
                node_id: String(L.currentNodeId),
                node_class: L.currentNodeClass,
                elapsed_ms: performance.now() - L.nodeStartTs,
                vram_peak_mb: 0,
                error: false,
                _live: true,
            });
        }
        const maxMs = Math.max(1, ...allRows.map(r => r.elapsed_ms));
        const prgPct = L.progress.max > 0 ? (L.progress.value / L.progress.max * 100).toFixed(0) : "";
        let html = `<div style="padding:6px 0;border-bottom:1px solid var(--c2c-surface0);margin-bottom:6px;">
            <span style="color:var(--c2c-okSoft);font-weight:700;">⏱ LIVE</span>
            <span style="margin-left:8px;">${(elapsed / 1000).toFixed(1)}s elapsed · ${done.length} done${
                L.currentNodeClass ? ` · executing: <b style="color:var(--c2c-yellow);">${_esc(L.currentNodeClass)}</b>${prgPct ? ` (${prgPct}%)` : ""}` : ""
            }</span>
        </div>`;
        html += allRows.map((r) => {
            const pct = Math.min(1, r.elapsed_ms / maxMs);
            const isLive = r._live;
            const color = isLive ? "var(--c2c-yellow)" : _heatColor(pct);
            const label = `${_esc(r.node_class)} (#${_esc(r.node_id)})`;
            const progBar = isLive && L.progress.max > 0
                ? `<div style="position:absolute;top:0;left:0;height:100%;width:${(L.progress.value/L.progress.max*100).toFixed(1)}%;background:var(--c2c-mauve);opacity:0.3;border-radius:3px;"></div>`
                : "";
            return `<div class="fg-row" data-nid="${_esc(r.node_id)}">
                <div class="fg-bar">
                    <div class="fg-bar-fill" style="width:${(pct * 100).toFixed(1)}%;background:${color};${isLive ? "animation:pulse 1s infinite;" : ""}"></div>
                    ${progBar}
                    <div class="fg-bar-label">${label}${isLive ? " ⏳" : ""}</div>
                </div>
                <div class="fg-time">${r.elapsed_ms.toFixed(0)} ms</div>
            </div>`;
        }).join("");
        body.insertAdjacentHTML("beforeend", `<div class="fg-rows">${html}</div>`);
        return;
    }
    if (!data || !data.rows || data.rows.length === 0) {
        body.insertAdjacentHTML("beforeend", `<div class="fg-empty">No completed runs yet — queue a prompt first.</div>`);
        return;
    }
    const rows = _sortRows(_filterRows(data.rows));
    if (!rows.length) {
        body.insertAdjacentHTML("beforeend", `<div class="fg-empty">No nodes match the filter.</div>`);
        return;
    }
    const maxMs = Math.max(1, ...rows.map((r) => r.elapsed_ms));
    const rowsHtml = rows.map((r) => {
        const pct = Math.min(1, r.elapsed_ms / maxMs);
        const color = r.error ? "var(--c2c-red)" : _heatColor(pct);
        const label = `${_esc(r.node_class)} (#${_esc(r.node_id)})`;
        return `<div class="fg-row ${r.error ? "fg-error" : ""}" data-nid="${_esc(r.node_id)}">
            <div class="fg-bar" title="${r.error ? "Errored: " + _esc(r.exc_type || "") : ""}">
                <div class="fg-bar-fill" style="width:${(pct * 100).toFixed(1)}%;background:${color};"></div>
                <div class="fg-bar-label">${label}</div>
            </div>
            <div class="fg-time">${r.elapsed_ms.toFixed(0)} ms</div>
        </div>`;
    }).join("");
    body.insertAdjacentHTML("beforeend", `<div class="fg-rows">${rowsHtml}</div>`);
}

function _renderCompare(body) {
    if (!_state.A || !_state.B) {
        body.insertAdjacentHTML("beforeend", `<div class="fg-empty">Snapshot two runs (A and B) to compare.</div>`);
        return;
    }
    const mapA = new Map(_state.A.rows.map((r) => [String(r.node_id) + ":" + r.node_class, r]));
    const mapB = new Map(_state.B.rows.map((r) => [String(r.node_id) + ":" + r.node_class, r]));
    const keys = new Set([...mapA.keys(), ...mapB.keys()]);
    const rows = [...keys].map((k) => {
        const a = mapA.get(k); const b = mapB.get(k);
        const aMs = a?.elapsed_ms ?? null;
        const bMs = b?.elapsed_ms ?? null;
        let delta = null, pct = null, status = "flat";
        if (aMs == null) { status = "new"; }
        else if (bMs == null) { status = "gone"; }
        else { delta = bMs - aMs; pct = aMs > 0 ? (delta / aMs) * 100 : 0; status = Math.abs(pct) < 3 ? "flat" : (delta > 0 ? "up" : "down"); }
        return { key: k, cls: (a || b).node_class, nid: (a || b).node_id, aMs, bMs, delta, pct, status };
    });
    // Sort by abs delta desc, then status (new/gone at bottom)
    rows.sort((x, y) => {
        const rank = (s) => ({ up: 0, down: 0, flat: 1, new: 2, gone: 3 }[s] ?? 4);
        const rx = rank(x.status), ry = rank(y.status);
        if (rx !== ry) return rx - ry;
        return Math.abs(y.delta || 0) - Math.abs(x.delta || 0);
    });
    const filtered = rows.filter((r) => {
        const q = _state.filter.trim().toLowerCase();
        return !q || String(r.cls).toLowerCase().includes(q) || String(r.nid).toLowerCase().includes(q);
    });
    const header = `<div class="fg-cmp-row" style="font-weight:600;color:var(--c2c-overlay0);border-bottom:1px solid var(--c2c-surface1);">
        <div>node</div><div style="text-align:right;">A ms</div><div style="text-align:right;">B ms</div><div style="text-align:right;">Δ</div>
    </div>`;
    const html = filtered.map((r) => {
        const aTxt = r.aMs == null ? "—" : r.aMs.toFixed(0);
        const bTxt = r.bMs == null ? "—" : r.bMs.toFixed(0);
        let dTxt = "—";
        if (r.status === "new") dTxt = "NEW";
        else if (r.status === "gone") dTxt = "gone";
        else dTxt = `${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(0)}ms / ${r.pct >= 0 ? "+" : ""}${r.pct.toFixed(1)}%`;
        return `<div class="fg-cmp-row">
            <div class="fg-cmp-cls">${_esc(r.cls)} <span style="color:var(--c2c-overlay0);">#${_esc(r.nid)}</span></div>
            <div class="fg-cmp-A">${aTxt}</div>
            <div class="fg-cmp-B">${bTxt}</div>
            <div class="fg-cmp-D ${r.status}">${dTxt}</div>
        </div>`;
    }).join("");
    const dT = _state.B.total_ms - _state.A.total_ms;
    const dPct = _state.A.total_ms > 0 ? (dT / _state.A.total_ms) * 100 : 0;
    const cls = Math.abs(dPct) < 3 ? "flat" : (dT > 0 ? "up" : "down");
    body.insertAdjacentHTML("beforeend", `
        <div style="margin:6px 0;padding:6px 8px;background:var(--c2c-bg3);border:1px solid var(--c2c-surface0);border-radius:4px;font-size:11px;">
            Total: A=<b style="color:var(--c2c-sapphire);">${(_state.A.total_ms/1000).toFixed(2)}s</b> · B=<b style="color:var(--c2c-mauve);">${(_state.B.total_ms/1000).toFixed(2)}s</b>
            · Δ=<b class="fg-cmp-D ${cls}" style="display:inline;">${dT>=0?"+":""}${dT.toFixed(0)}ms / ${dPct>=0?"+":""}${dPct.toFixed(1)}%</b>
        </div>
        ${header}${html}
        <div class="fg-ai ${_state.aiOutput ? "" : "empty"}" data-role="ai">${_state.aiOutput ? _esc(_state.aiOutput) : "Click ✨ Explain for an AI regression analysis."}</div>
    `);
}

function _renderPanel() {
    const panel = document.getElementById(PANEL_ID);
    if (!panel) return;
    const body = panel.querySelector(".c2c-win-body");
    if (!body) return;
    _clear();
    body.innerHTML = _renderToolbar();
    const subtitle = panel.querySelector(".c2c-win-subtitle");
    if (subtitle) {
        const d = _state.current;
        subtitle.textContent = _state.mode === "compare"
            ? `compare A ↔ B`
            : (d ? `${d.node_count} nodes · total ${d.total_ms.toFixed(0)} ms${d.prompt_id ? ` · ${String(d.prompt_id).slice(0,8)}…` : ""}` : "no data yet");
    }
    if (_state.mode === "compare") _renderCompare(body);
    else _renderCurrent(body, _state.current);
    _bindToolbar(body);
}

function _bindToolbar(body) {
    const tb = body.querySelector(".fg-tools");
    if (!tb) return;
    _on(tb.querySelector('[data-act="refresh"]'), "click", () => _refresh());
    _on(tb.querySelector('[data-role="filter"]'), "input", (e) => { _state.filter = e.target.value; _renderPanel(); });
    _on(tb.querySelector('[data-role="sort"]'), "change", (e) => { _state.sortBy = e.target.value; _renderPanel(); });
    _on(tb.querySelector('[data-act="snap-a"]'), "click", () => { if (_state.current) { _state.A = JSON.parse(JSON.stringify(_state.current)); _renderPanel(); } });
    _on(tb.querySelector('[data-act="snap-b"]'), "click", () => { if (_state.current) { _state.B = JSON.parse(JSON.stringify(_state.current)); _renderPanel(); } });
    const mb = tb.querySelector('[data-act="mode"]');
    if (mb) _on(mb, "click", () => { _state.mode = _state.mode === "compare" ? "current" : "compare"; _renderPanel(); });
    const ai = tb.querySelector('[data-act="ai"]');
    if (ai) _on(ai, "click", () => _aiExplain(body));
    _on(tb.querySelector('[data-act="csv"]'), "click", () => _copyCsv());
}

async function _refresh() {
    try {
        const resp = await fetch("/mec/diagnostics/flamegraph?limit=50");
        const json = await resp.json();
        if (json.success) {
            _state.current = json.data;
            _renderPanel();
        }
    } catch (e) {
        console.warn("[C2C.FlameGraph] fetch failed:", e);
    }
}

function _copyCsv() {
    const src = _state.mode === "compare"
        ? _buildCompareCsv()
        : _buildCurrentCsv();
    if (!src) return;
    navigator.clipboard?.writeText(src).then(() => console.log("[C2C.FlameGraph] CSV copied."));
}

function _buildCurrentCsv() {
    const d = _state.current;
    if (!d?.rows) return null;
    const header = "node_id,node_class,elapsed_ms,cpu_ms,vram_delta_mb,vram_peak_mb,error";
    const lines = d.rows.map((r) => `${r.node_id},${r.node_class},${r.elapsed_ms},${r.cpu_ms},${r.vram_delta_mb},${r.vram_peak_mb},${r.error}`);
    return [header, ...lines].join("\n");
}

function _buildCompareCsv() {
    if (!_state.A || !_state.B) return null;
    const mapA = new Map(_state.A.rows.map((r) => [String(r.node_id), r]));
    const mapB = new Map(_state.B.rows.map((r) => [String(r.node_id), r]));
    const keys = new Set([...mapA.keys(), ...mapB.keys()]);
    const header = "node_id,node_class,A_ms,B_ms,delta_ms,delta_pct";
    const lines = [...keys].map((k) => {
        const a = mapA.get(k); const b = mapB.get(k);
        const aMs = a?.elapsed_ms ?? "";
        const bMs = b?.elapsed_ms ?? "";
        const cls = (a || b).node_class;
        const d = (typeof aMs === "number" && typeof bMs === "number") ? (bMs - aMs).toFixed(2) : "";
        const p = (typeof aMs === "number" && typeof bMs === "number" && aMs > 0) ? ((bMs - aMs) / aMs * 100).toFixed(2) : "";
        return `${k},${cls},${aMs},${bMs},${d},${p}`;
    });
    return [header, ...lines].join("\n");
}

async function _aiExplain(body) {
    if (_state.aiBusy || !_state.A || !_state.B) return;
    _state.aiBusy = true;
    _renderPanel();
    const aiEl = document.querySelector(`#${PANEL_ID} [data-role="ai"]`);
    if (aiEl) { aiEl.classList.remove("empty"); aiEl.textContent = "Analyzing regression…"; }
    const csv = _buildCompareCsv() || "";
    _state.aiOutput = "";
    try {
        await streamAI({
            feature: "flame_compare",
            sensitivity: "normal",
            max_tokens: 400,
            temperature: 0.3,
            messages: [
                { role: "system", content: "You are a performance analyst. Given two flame-graph runs (CSV) explain WHAT got slower/faster, by how much, and the most likely cause (model swap, batch change, sampler/scheduler, IO, VRAM swap). Be concise: bullet points." },
                { role: "user", content: `Total A=${(_state.A.total_ms/1000).toFixed(2)}s, B=${(_state.B.total_ms/1000).toFixed(2)}s.\n\n${csv}` },
            ],
            onChunk: (c) => { _state.aiOutput += c; const el = document.querySelector(`#${PANEL_ID} [data-role="ai"]`); if (el) el.textContent = _state.aiOutput; },
            onError: (e) => { _state.aiOutput += `\n[error: ${e}]`; },
            onDone: () => {},
        });
    } catch (e) {
        _state.aiOutput += `\n[exception: ${e?.message || e}]`;
    } finally {
        _state.aiBusy = false;
        const btn = document.querySelector(`#${PANEL_ID} [data-act="ai"]`);
        if (btn) btn.textContent = "✨ Explain";
    }
}

function _openPanel() {
    _state.open = true;
    _ensurePanel();
    _refresh();
}

function _closePanel() {
    _state.open = false;
    _clear();
    document.getElementById(PANEL_ID)?.remove();
}

function _togglePanel() { _state.open ? _closePanel() : _openPanel(); }

function _ensurePanel() {
    if (document.getElementById(PANEL_ID)) return;
    buildPanel({
        id: PANEL_ID,
        title: "⏱ Execution Flame Graph",
        shortcut: "Ctrl+Shift+G",
        width: 560,
        height: 520,
        storageKey: "flame",
        onClose: () => { _state.open = false; _clear(); },
        actions: [],
    });
}

function _ensureUi() {
    if (!document.getElementById(BTN_ID)) {
        const btn = document.createElement("button");
        btn.id = BTN_ID;
        btn.title = "Execution Flame Graph (Ctrl+Shift+G)";
        btn.textContent = "⏱";
        btn.addEventListener("click", _togglePanel);
        document.body.appendChild(btn);
    }
}

if (!LITE) app.registerExtension({
    name: "C2C.FlameGraph",
    settings: [
        {
            id: "c2c.flamegraph.enabled",
            name: "Flame Graph: enabled",
            tooltip: "Show the ⏱ button for the execution flame graph.",
            type: "boolean",
            default: true,
            onChange: (v) => {
                const btn = document.getElementById(BTN_ID);
                if (btn) btn.style.display = v ? "flex" : "none";
            },
        },
        {
            id: "c2c.flamegraph.auto_show",
            name: "Flame Graph: auto-open after run",
            tooltip: "Automatically open the flame graph panel when a prompt finishes.",
            type: "boolean",
            default: false,
        },
    ],
    async setup() {
        _injectStyle();
        _ensureUi();

        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("c2c.flamegraph.enabled", true); }
            catch { return true; }
        })();
        const btn = document.getElementById(BTN_ID);
        if (btn) btn.style.display = enabled ? "flex" : "none";

        api.addEventListener("execution_start", _liveExecStart);
        api.addEventListener("executed", _liveExecuted);
        api.addEventListener("progress", _liveProgress);
        api.addEventListener("execution_success", (ev) => {
            _liveExecDone(ev);
            const auto = (() => {
                try { return app.ui.settings.getSettingValue("c2c.flamegraph.auto_show", false); }
                catch { return false; }
            })();
            if (!_state.open && auto) _openPanel();
        });
        api.addEventListener("execution_error", (ev) => {
            _liveExecDone(ev);
        });

        console.log("[C2C.FlameGraph] live-tracking loaded.");
    },
});
