/**
 * c2c_workflow_wizard.js — Workflow Wizard (god-level rebuild, 2026-05-27)
 *
 * 9 features:
 *   1) Template browser with categories, search box, favorites (localStorage)
 *   2) Step navigator sidebar — every step clickable, status-coded
 *   3) Pulsing multi-node highlight on the active step's target node_type
 *   4) Auto-zoom-and-center on the target node when entering a step
 *   5) Inline widget editor — widget-typed steps render the widget value
 *      with a live <input>/<select> bound to the actual node widget
 *   6) Step validation — gate "Next" until the node exists (and widget is set
 *      for widget-typed steps); shows red/green badge
 *   7) AI "why this step?" via streamAI (feature: "wizard_explain")
 *   8) Skip / mark-done per step with persistence across resumes
 *   9) Resume: progress per-template-id stored in localStorage; offered on
 *      template open if a saved state exists
 *
 * Structural rules:
 *   - Body-only re-render — chrome (8 resize edges + header) survives every
 *     view change. _renderBody() targets [data-role="body"], never the panel
 *     root.
 *   - z-index inherits .c2c-win (5000 base) via attachWindowChrome; no
 *     --c2c-z-hud overrides on the panel.
 *   - Listeners owned by _listeners[], cleared before every re-bind.
 */

import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { attachWindowChrome } from "./_c2c_window.js";
import { streamAI } from "./_c2c_ai_client.js";
import { c2cConfirm, c2cAlert } from "./_c2c_dialog.js";

const BTN_ID   = "mec-wizard-btn";
const PANEL_ID = "mec-wizard-panel";
const STYLE_ID = "mec-wizard-style";
const LS_FAVS  = "c2c.wizard.favs";
const LS_RESUME = "c2c.wizard.resume";       // { [template_id]: { idx, skipped:[], done:[], ts } }

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

let _open = false;
let _view = "home";                          // "home" | "step"
let _templates = [];                         // cached list
let _query = "";
let _favs = new Set();
let _active = null;                          // { template_id, title, steps, idx, skipped:Set, done:Set }
let _aiBusy = false;
let _aiOutput = "";
let _drawPatched = false;
let _rafTok = null;
const _listeners = [];

function _on(t, e, fn, opts) { t.addEventListener(e, fn, opts); _listeners.push(() => t.removeEventListener(e, fn, opts)); }
function _clearListeners() { while (_listeners.length) { try { _listeners.pop()(); } catch {} } }

function _loadFavs() {
    try { _favs = new Set(JSON.parse(localStorage.getItem(LS_FAVS) || "[]")); }
    catch { _favs = new Set(); }
}
function _saveFavs() { localStorage.setItem(LS_FAVS, JSON.stringify([..._favs])); }

function _readResume() {
    try { return JSON.parse(localStorage.getItem(LS_RESUME) || "{}"); }
    catch { return {}; }
}
function _writeResume(map) { localStorage.setItem(LS_RESUME, JSON.stringify(map)); }
function _saveActiveResume() {
    if (!_active) return;
    const m = _readResume();
    m[_active.template_id] = {
        idx: _active.idx,
        skipped: [..._active.skipped],
        done: [..._active.done],
        ts: Date.now(),
    };
    _writeResume(m);
}
function _clearActiveResume(id) {
    const m = _readResume();
    delete m[id];
    _writeResume(m);
}

// ─────────────────────────────────────────────────────────────────────────────
// Style
// ─────────────────────────────────────────────────────────────────────────────

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${BTN_ID} {
    position: fixed; bottom: 202px; right: 16px;
    z-index: var(--c2c-z-hud, 1000);
    width: 38px; height: 38px; border-radius: 50%;
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface1); color: var(--c2c-yellow);
    font-size: 16px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex; align-items: center; justify-content: center;
}
#${BTN_ID}:hover { border-color: var(--c2c-yellow); }

#${PANEL_ID} {
    position: fixed; top: 80px; right: 80px;
    width: min(92vw, 720px); height: min(80vh, 620px);
    background: var(--c2c-bg2); border: 1px solid var(--c2c-surface1); border-radius: 8px;
    color: var(--c2c-fg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px; box-shadow: 0 6px 32px rgba(0,0,0,0.7);
    display: none; flex-direction: column; overflow: hidden;
}
#${PANEL_ID}.visible { display: flex; }
#${PANEL_ID} h3 {
    margin: 0; padding: 8px 12px; color: var(--c2c-yellow); font-size: 14px;
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(180deg, var(--c2c-bg) 0%, var(--c2c-bg2) 100%);
    border-bottom: 1px solid var(--c2c-surface0);
}
#${PANEL_ID} .wz-body { flex: 1 1 auto; min-height: 0; display: flex; }
#${PANEL_ID} .wz-close { background: none; border: none; color: var(--c2c-overlay0); cursor: pointer; font-size: 16px; padding: 0 4px; }

/* Home view */
#${PANEL_ID} .wz-home { flex: 1; display: flex; flex-direction: column; padding: 10px; gap: 8px; }
#${PANEL_ID} .wz-search-row { display: flex; gap: 6px; align-items: center; }
#${PANEL_ID} .wz-search-row input {
    flex: 1; background: var(--c2c-bg); border: 1px solid var(--c2c-surface0); color: var(--c2c-fg);
    border-radius: 4px; padding: 5px 8px; font-size: 12px;
}
#${PANEL_ID} .wz-cats { display: flex; flex-wrap: wrap; gap: 4px; }
#${PANEL_ID} .wz-cats .wz-cat {
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface0); color: var(--c2c-fg);
    border-radius: 12px; padding: 2px 10px; cursor: pointer; font-size: 11px;
}
#${PANEL_ID} .wz-cats .wz-cat.active { background: var(--c2c-surface0); border-color: var(--c2c-yellow); color: var(--c2c-yellow); }
#${PANEL_ID} .wz-list { flex: 1; overflow: auto; display: flex; flex-direction: column; gap: 6px; }
#${PANEL_ID} .wz-card {
    border: 1px solid var(--c2c-surface0); border-radius: 5px; padding: 8px 10px;
    cursor: pointer; transition: border-color 120ms, background 120ms;
    position: relative;
}
#${PANEL_ID} .wz-card:hover { border-color: var(--c2c-yellow); background: var(--c2c-bg); }
#${PANEL_ID} .wz-card .wz-title { font-weight: 600; color: var(--c2c-yellow); }
#${PANEL_ID} .wz-card .wz-desc { color: var(--c2c-fg); margin-top: 2px; font-size: 11px; }
#${PANEL_ID} .wz-card .wz-meta { color: var(--c2c-overlay0); font-size: 10px; margin-top: 4px; }
#${PANEL_ID} .wz-card .wz-fav {
    position: absolute; top: 6px; right: 6px; background: none; border: none;
    color: var(--c2c-surface1); cursor: pointer; font-size: 14px;
}
#${PANEL_ID} .wz-card .wz-fav.on { color: var(--c2c-yellow); }
#${PANEL_ID} .wz-card .wz-resume {
    position: absolute; bottom: 6px; right: 6px; font-size: 9px;
    color: var(--c2c-sapphire); background: var(--c2c-bg); padding: 1px 5px; border-radius: 8px;
    border: 1px solid var(--c2c-surface0);
}

/* Step view */
#${PANEL_ID} .wz-step-wrap { flex: 1; display: flex; min-height: 0; }
#${PANEL_ID} .wz-nav {
    width: 200px; flex: 0 0 200px; overflow: auto;
    border-right: 1px solid var(--c2c-surface0); background: var(--c2c-bg3);
    padding: 6px 0;
}
#${PANEL_ID} .wz-nav .wz-nav-item {
    padding: 5px 8px 5px 14px; cursor: pointer; font-size: 11px;
    border-left: 3px solid transparent; position: relative;
    color: var(--c2c-subtext1); line-height: 1.3;
}
#${PANEL_ID} .wz-nav .wz-nav-item:hover { background: var(--c2c-bg); }
#${PANEL_ID} .wz-nav .wz-nav-item.active {
    border-left-color: var(--c2c-yellow); background: var(--c2c-bg); color: var(--c2c-yellow); font-weight: 600;
}
#${PANEL_ID} .wz-nav .wz-nav-item .wz-nav-num {
    display: inline-block; width: 18px; text-align: center; color: var(--c2c-overlay0);
    font-family: monospace; margin-right: 4px;
}
#${PANEL_ID} .wz-nav .wz-nav-item.done   { color: var(--c2c-okSoft); }
#${PANEL_ID} .wz-nav .wz-nav-item.done   .wz-nav-num::after { content: " ✓"; color: var(--c2c-okSoft); }
#${PANEL_ID} .wz-nav .wz-nav-item.skipped { color: var(--c2c-overlay0); text-decoration: line-through; }

#${PANEL_ID} .wz-step { flex: 1; padding: 10px 12px; overflow: auto; display: flex; flex-direction: column; }
#${PANEL_ID} .wz-progress {
    height: 4px; background: var(--c2c-surface0); border-radius: 2px; overflow: hidden;
    margin: 0 0 10px 0;
}
#${PANEL_ID} .wz-progress .bar { height: 100%; background: var(--c2c-yellow); transition: width 0.2s; }
#${PANEL_ID} .wz-step-title { color: var(--c2c-yellow); font-weight: 700; font-size: 14px; }
#${PANEL_ID} .wz-step-meta {
    color: var(--c2c-overlay0); font-size: 10px; margin: 4px 0 8px 0; font-family: monospace;
    display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
}
#${PANEL_ID} .wz-step-meta .wz-valid {
    padding: 1px 6px; border-radius: 8px; font-size: 10px;
}
#${PANEL_ID} .wz-step-meta .wz-valid.ok   { background: var(--c2c-okBgDark); color: var(--c2c-okSoft); }
#${PANEL_ID} .wz-step-meta .wz-valid.bad  { background: var(--c2c-panelBg); color: var(--c2c-red); }
#${PANEL_ID} .wz-step-hint  { margin: 4px 0 10px 0; line-height: 1.5; }

#${PANEL_ID} .wz-widget-edit {
    background: var(--c2c-bg3); border: 1px solid var(--c2c-surface0); border-radius: 4px;
    padding: 8px 10px; margin: 8px 0;
}
#${PANEL_ID} .wz-widget-edit h4 {
    margin: 0 0 6px 0; color: var(--c2c-sapphire); font-size: 11px; font-weight: 600;
}
#${PANEL_ID} .wz-widget-edit .wz-w-row { display: flex; gap: 6px; align-items: center; margin: 4px 0; }
#${PANEL_ID} .wz-widget-edit .wz-w-row label { color: var(--c2c-overlay0); min-width: 80px; font-family: monospace; font-size: 10px; }
#${PANEL_ID} .wz-widget-edit input,
#${PANEL_ID} .wz-widget-edit select {
    flex: 1; background: var(--c2c-bg); border: 1px solid var(--c2c-surface0); color: var(--c2c-fg);
    border-radius: 3px; padding: 3px 6px; font-size: 11px;
}

#${PANEL_ID} .wz-ai-box {
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface0); border-radius: 4px;
    padding: 8px 10px; margin: 8px 0; white-space: pre-wrap;
    font-size: 11px; line-height: 1.45; max-height: 200px; overflow: auto;
}
#${PANEL_ID} .wz-ai-box.empty { color: var(--c2c-overlay0); font-style: italic; }

#${PANEL_ID} .wz-actions {
    display: flex; gap: 6px; margin-top: auto; padding-top: 8px;
    border-top: 1px solid var(--c2c-surface0);
}
#${PANEL_ID} .wz-actions button {
    background: var(--c2c-surface0); border: 1px solid var(--c2c-surface1); color: var(--c2c-fg);
    border-radius: 4px; padding: 4px 12px; cursor: pointer; font-size: 11px;
}
#${PANEL_ID} .wz-actions button:hover:not(:disabled) { border-color: var(--c2c-yellow); }
#${PANEL_ID} .wz-actions button:disabled { opacity: 0.45; cursor: not-allowed; }
#${PANEL_ID} .wz-actions button.primary { border-color: var(--c2c-yellow); color: var(--c2c-yellow); }
#${PANEL_ID} .wz-actions .spacer { flex: 1; }
    `.trim();
    document.head.appendChild(style);
}

// ─────────────────────────────────────────────────────────────────────────────
// Panel lifecycle (chrome-preserving)
// ─────────────────────────────────────────────────────────────────────────────

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
        p.innerHTML = `
            <h3>
                <span>🧙 Workflow Wizard</span>
                <button class="wz-browse" title="Browse live workflows (OpenArt + Civitai)" style="background:none;border:none;color:var(--c2c-mauve);cursor:pointer;font-size:13px;padding:0 6px">🔍</button>
                <button class="wz-close" title="Close">×</button>
            </h3>
            <div class="wz-body" data-role="body"></div>
        `;
        document.body.appendChild(p);
        p.querySelector(".wz-close").addEventListener("click", _toggle);
        p.querySelector(".wz-browse").addEventListener("click", () => window.__C2C_PRESET_HUB__?.open({ tab: "openart" }));
        attachWindowChrome(p, {
            storageKey: "workflow_wizard",
            headerSelector: "h3",
            titleSelector: "h3 > span",
            minW: 480, minH: 360,
        });
    }
}

function _toggle() {
    _open = !_open;
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    if (_open) {
        p.classList.add("visible");
        _loadFavs();
        _view = _active ? "step" : "home";
        _renderBody();
        if (_view === "home" && !_templates.length) _fetchTemplates().then(() => _renderBody());
    } else {
        p.classList.remove("visible");
        _clearListeners();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Templates fetch (cached)
// ─────────────────────────────────────────────────────────────────────────────

async function _fetchTemplates() {
    try {
        const r = await fetch("/mec/wizard/templates");
        const j = await r.json();
        if (j.success) _templates = j.data.templates || [];
    } catch (e) {
        console.warn("[C2C.Wizard] list failed:", e);
        _templates = [];
    }
}

async function _fetchTemplate(id) {
    const r = await fetch(`/mec/wizard/templates/${encodeURIComponent(id)}`);
    const j = await r.json();
    if (!j.success) throw new Error(j.error || "fetch failed");
    return j.data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Render entry
// ─────────────────────────────────────────────────────────────────────────────

function _renderBody() {
    const p = document.getElementById(PANEL_ID);
    if (!p) return;
    const body = p.querySelector('[data-role="body"]');
    if (!body) return;
    _clearListeners();
    if (_view === "step" && _active) _renderStep(body);
    else _renderHome(body);
}

// ── Home view ────────────────────────────────────────────────────────────────

let _catFilter = "all";

function _renderHome(body) {
    const cats = [...new Set(_templates.map((t) => t.category || "general"))];
    const resumes = _readResume();
    const q = _query.toLowerCase();
    const filtered = _templates.filter((t) => {
        if (_catFilter !== "all" && (t.category || "general") !== _catFilter) return false;
        if (q && !`${t.title} ${t.description} ${t.id}`.toLowerCase().includes(q)) return false;
        return true;
    });
    // favorites first
    filtered.sort((a, b) => (_favs.has(b.id) ? 1 : 0) - (_favs.has(a.id) ? 1 : 0));

    body.innerHTML = `
        <div class="wz-home">
            <div class="wz-search-row">
                <input data-role="search" placeholder="Search templates…" value="${_esc(_query)}">
            </div>
            <div class="wz-cats">
                <span class="wz-cat ${_catFilter === "all" ? "active" : ""}" data-cat="all">all (${_templates.length})</span>
                ${cats.map((c) => {
                    const n = _templates.filter((t) => (t.category || "general") === c).length;
                    return `<span class="wz-cat ${_catFilter === c ? "active" : ""}" data-cat="${_esc(c)}">${_esc(c)} (${n})</span>`;
                }).join("")}
            </div>
            <div class="wz-list" data-role="list">
                ${filtered.length === 0 ? `<div style="color:var(--c2c-overlay0);">No templates match.</div>` : filtered.map((t) => {
                    const fav = _favs.has(t.id);
                    const r = resumes[t.id];
                    return `
                        <div class="wz-card" data-id="${_esc(t.id)}">
                            <button class="wz-fav ${fav ? "on" : ""}" data-role="fav" data-id="${_esc(t.id)}" title="Favorite">★</button>
                            <div class="wz-title">${_esc(t.title)}</div>
                            <div class="wz-desc">${_esc(t.description || "")}</div>
                            <div class="wz-meta">${t.step_count ?? "?"} steps · ${_esc(t.category || "general")} · ${_esc(t.id)}</div>
                            ${r ? `<div class="wz-resume">resume at step ${r.idx + 1}</div>` : ""}
                        </div>
                    `;
                }).join("")}
            </div>
        </div>
    `;

    const search = body.querySelector('[data-role="search"]');
    _on(search, "input", (e) => { _query = e.target.value; _renderHome(body); });
    for (const c of body.querySelectorAll(".wz-cat")) {
        _on(c, "click", () => { _catFilter = c.dataset.cat; _renderHome(body); });
    }
    for (const card of body.querySelectorAll(".wz-card")) {
        _on(card, "click", (e) => {
            if (e.target.closest('[data-role="fav"]')) return;
            const id = card.dataset.id;
            _startWizard(id);
        });
    }
    for (const fav of body.querySelectorAll('[data-role="fav"]')) {
        _on(fav, "click", (e) => {
            e.stopPropagation();
            const id = fav.dataset.id;
            if (_favs.has(id)) _favs.delete(id); else _favs.add(id);
            _saveFavs();
            _renderHome(body);
        });
    }
}

// ── Wizard activation ────────────────────────────────────────────────────────

async function _startWizard(id) {
    let tpl;
    try { tpl = await _fetchTemplate(id); }
    catch (e) { console.warn("[C2C.Wizard] start failed:", e); c2cAlert("Could not load template."); return; }
    const saved = _readResume()[id];
    let resumeIdx = 0;
    let skipped = new Set();
    let done = new Set();
    if (saved && (saved.idx > 0 || (saved.done && saved.done.length))) {
        if (await c2cConfirm(`Resume "${tpl.title}" from step ${saved.idx + 1}?`)) {
            resumeIdx = Math.min(saved.idx, (tpl.steps?.length || 1) - 1);
            skipped = new Set(saved.skipped || []);
            done = new Set(saved.done || []);
        } else {
            _clearActiveResume(id);
        }
    }
    _active = {
        template_id: tpl.id,
        title: tpl.title,
        steps: tpl.steps || [],
        idx: resumeIdx,
        skipped,
        done,
    };
    _aiOutput = "";
    _view = "step";
    _renderBody();
    _startHighlightLoop();
    _autoCenterOnTarget();
    _ensureDrawPatch();
}

function _endWizard({ commit = false } = {}) {
    if (_active && commit) _saveActiveResume();
    if (_active) _clearActiveResume(_active.template_id);
    _active = null;
    _aiOutput = "";
    _view = "home";
    _stopHighlightLoop();
    _renderBody();
}

// ── Step view ────────────────────────────────────────────────────────────────

function _stepNodeMatches(step) {
    const g = app.graph;
    if (!g || !g._nodes || !step?.node_type) return [];
    const want = step.node_type.toLowerCase();
    return g._nodes.filter((n) => (n.type || "").toLowerCase() === want);
}

function _stepValidity(step) {
    const matches = _stepNodeMatches(step);
    if (!step.node_type) return { ok: true, msg: "no target" };
    if (!matches.length) return { ok: false, msg: `no ${step.node_type} in graph` };
    if (step.widget) {
        const node = matches[0];
        const w = (node.widgets || []).find((x) => x.name === step.widget);
        if (!w) return { ok: false, msg: `widget "${step.widget}" missing` };
        if (w.value === undefined || w.value === null || w.value === "") {
            return { ok: false, msg: `widget "${step.widget}" empty` };
        }
        return { ok: true, msg: `${step.widget} = ${String(w.value).slice(0, 40)}` };
    }
    return { ok: true, msg: `found ${matches.length}` };
}

function _renderStep(body) {
    if (!_active) return;
    const { steps, idx, title, skipped, done } = _active;
    const step = steps[idx];
    if (!step) { _endWizard(); return; }
    const pct = ((idx + 1) / steps.length) * 100;
    const v = _stepValidity(step);

    body.innerHTML = `
        <div class="wz-step-wrap">
            <div class="wz-nav" data-role="nav">
                ${steps.map((s, i) => {
                    const cls = [
                        "wz-nav-item",
                        i === idx ? "active" : "",
                        done.has(i) ? "done" : "",
                        skipped.has(i) ? "skipped" : "",
                    ].filter(Boolean).join(" ");
                    return `<div class="${cls}" data-i="${i}">
                        <span class="wz-nav-num">${i + 1}</span>${_esc(s.title)}
                    </div>`;
                }).join("")}
            </div>
            <div class="wz-step">
                <div class="wz-progress"><div class="bar" style="width:${pct}%"></div></div>
                <div class="wz-step-title">${_esc(step.title)}</div>
                <div class="wz-step-meta">
                    <span>node: ${_esc(step.node_type || "—")}</span>
                    ${step.widget ? `<span>widget: ${_esc(step.widget)}</span>` : ""}
                    <span class="wz-valid ${v.ok ? "ok" : "bad"}">${v.ok ? "✓" : "✗"} ${_esc(v.msg)}</span>
                </div>
                <div class="wz-step-hint">${_esc(step.hint || "")}</div>
                ${step.widget ? _renderWidgetEditor(step) : ""}
                ${(_aiOutput || _aiBusy) ? `<div class="wz-ai-box${_aiOutput ? "" : " empty"}" data-role="ai">${_esc(_aiOutput) || "Waiting for AI…"}</div>` : ""}
                <div class="wz-actions">
                    <button data-act="prev" ${idx === 0 ? "disabled" : ""}>← Back</button>
                    <button data-act="skip">Skip</button>
                    <button data-act="done">Mark done</button>
                    <span class="spacer"></span>
                    <button data-act="ai" ${_aiBusy ? "disabled" : ""}>${_aiBusy ? "⌛ AI…" : "✨ Why?"}</button>
                    <button data-act="exit">Exit</button>
                    <button data-act="next" class="primary" ${(!v.ok && !skipped.has(idx)) ? "disabled" : ""}>
                        ${idx === steps.length - 1 ? "Finish ✓" : "Next →"}
                    </button>
                </div>
            </div>
        </div>
    `;

    // Sidebar nav
    for (const item of body.querySelectorAll(".wz-nav-item")) {
        _on(item, "click", () => {
            _active.idx = parseInt(item.dataset.i, 10);
            _aiOutput = "";
            _saveActiveResume();
            _renderBody();
            _autoCenterOnTarget();
        });
    }
    // Widget editor bindings
    if (step.widget) _bindWidgetEditor(body, step);

    // Actions
    const act = (name) => body.querySelector(`[data-act="${name}"]`);
    _on(act("prev"), "click", () => { _active.idx = Math.max(0, idx - 1); _aiOutput = ""; _saveActiveResume(); _renderBody(); _autoCenterOnTarget(); });
    _on(act("skip"), "click", () => { _active.skipped.add(idx); _active.done.delete(idx); _saveActiveResume(); _renderBody(); });
    _on(act("done"), "click", () => { _active.done.add(idx); _active.skipped.delete(idx); _saveActiveResume(); _renderBody(); });
    _on(act("exit"), "click", () => _endWizard({ commit: true }));
    _on(act("next"), "click", () => {
        if (idx >= steps.length - 1) { _endWizard(); return; }
        if (v.ok) _active.done.add(idx);
        _active.idx = idx + 1;
        _aiOutput = "";
        _saveActiveResume();
        _renderBody();
        _autoCenterOnTarget();
    });
    _on(act("ai"), "click", () => _aiExplain(step));
}

// ── Inline widget editor ─────────────────────────────────────────────────────

function _renderWidgetEditor(step) {
    return `
        <div class="wz-widget-edit">
            <h4>Set widget</h4>
            <div data-role="wedit"></div>
        </div>
    `;
}

function _bindWidgetEditor(body, step) {
    const slot = body.querySelector('[data-role="wedit"]');
    if (!slot) return;
    const nodes = _stepNodeMatches(step);
    if (!nodes.length) {
        slot.innerHTML = `<div style="color:var(--c2c-red);">Node "${_esc(step.node_type)}" not in graph yet — add it on the canvas first.</div>`;
        return;
    }
    const node = nodes[0];
    const w = (node.widgets || []).find((x) => x.name === step.widget);
    if (!w) {
        slot.innerHTML = `<div style="color:var(--c2c-red);">Widget "${_esc(step.widget)}" not found on node.</div>`;
        return;
    }
    const t = w.type || (typeof w.value === "number" ? "number" : (typeof w.value === "boolean" ? "boolean" : "text"));
    let html = "";
    if (w.options?.values && Array.isArray(w.options.values)) {
        html = `<select data-role="winput">${w.options.values.map((opt) =>
            `<option value="${_esc(opt)}"${String(opt) === String(w.value) ? " selected" : ""}>${_esc(opt)}</option>`
        ).join("")}</select>`;
    } else if (t === "number" || typeof w.value === "number") {
        html = `<input type="number" data-role="winput" value="${_esc(w.value)}" step="${w.options?.step ?? "any"}">`;
    } else if (t === "boolean" || typeof w.value === "boolean") {
        html = `<input type="checkbox" data-role="winput" ${w.value ? "checked" : ""}>`;
    } else {
        html = `<input type="text" data-role="winput" value="${_esc(w.value ?? "")}">`;
    }
    slot.innerHTML = `<div class="wz-w-row"><label>${_esc(step.widget)}</label>${html}</div>`;
    const input = slot.querySelector('[data-role="winput"]');
    _on(input, "change", () => {
        let v;
        if (input.type === "checkbox") v = input.checked;
        else if (input.type === "number") v = parseFloat(input.value);
        else v = input.value;
        w.value = v;
        if (typeof w.callback === "function") {
            try { w.callback(v, app.canvas, node); } catch (e) { console.warn(e); }
        }
        node.setDirtyCanvas?.(true, true);
        // Re-render to refresh validity badge
        _renderBody();
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Highlight + zoom-to-node
// ─────────────────────────────────────────────────────────────────────────────

function _drawHighlight() {
    if (!_active) return;
    const canvas = app.canvas;
    if (!canvas || !canvas.ctx) return;
    const step = _active.steps[_active.idx];
    if (!step) return;
    const targets = _stepNodeMatches(step);
    if (!targets.length) return;
    const ctx = canvas.ctx;
    ctx.save();
    const t = (performance.now() % 1500) / 1500;
    const alpha = 0.35 + 0.55 * Math.abs(Math.sin(t * Math.PI * 2));
    ctx.lineWidth = 4;
    ctx.strokeStyle = `rgba(249, 226, 175, ${alpha.toFixed(3)})`;
    for (const n of targets) {
        const [x, y] = n.pos;
        const [w, h] = n.size;
        const hh = n.flags?.collapsed ? 30 : h;
        ctx.strokeRect(x - 6, y - 6, w + 12, hh + 12);
    }
    ctx.restore();
}

function _startHighlightLoop() {
    _stopHighlightLoop();
    const tick = () => {
        if (!_active) return;
        app.canvas?.setDirty?.(true, true);
        _rafTok = requestAnimationFrame(tick);
    };
    _rafTok = requestAnimationFrame(tick);
}

function _stopHighlightLoop() {
    if (_rafTok) { cancelAnimationFrame(_rafTok); _rafTok = null; }
}

function _ensureDrawPatch() {
    if (_drawPatched) return;
    if (typeof LGraphCanvas === "undefined") return;
    const orig = LGraphCanvas.prototype.onDrawForeground;
    LGraphCanvas.prototype.onDrawForeground = function (ctx) {
        if (orig) orig.call(this, ctx);
        try { _drawHighlight(); }
        catch (e) { __c2cReport("c2c_workflow_wizard", e); }
    };
    LGraphCanvas.prototype.onDrawForeground._c2cWizardPatched = true;
    _drawPatched = true;
}

function _autoCenterOnTarget() {
    if (!_active) return;
    const step = _active.steps[_active.idx];
    if (!step) return;
    const targets = _stepNodeMatches(step);
    if (!targets.length) return;
    const n = targets[0];
    const canvas = app.canvas;
    if (!canvas) return;
    const cx = n.pos[0] + (n.size[0] / 2);
    const cy = n.pos[1] + ((n.flags?.collapsed ? 30 : n.size[1]) / 2);
    try {
        if (typeof canvas.centerOnNode === "function") {
            canvas.centerOnNode(n);
        } else if (canvas.ds) {
            // LiteGraph DragAndScale: set offset so node center aligns with canvas center
            const rect = canvas.canvas.getBoundingClientRect();
            const scale = canvas.ds.scale || 1;
            canvas.ds.offset[0] = (rect.width / (2 * scale)) - cx;
            canvas.ds.offset[1] = (rect.height / (2 * scale)) - cy;
        }
        canvas.setDirty?.(true, true);
    } catch (e) { /* ignore */ }
}

// ─────────────────────────────────────────────────────────────────────────────
// AI explain
// ─────────────────────────────────────────────────────────────────────────────

async function _aiExplain(step) {
    if (_aiBusy) return;
    _aiBusy = true;
    _aiOutput = "";
    _renderBody();
    const ai = document.querySelector(`#${PANEL_ID} [data-role="ai"]`);
    const onChunk = (chunk) => {
        _aiOutput += chunk;
        if (ai) {
            ai.classList.remove("empty");
            ai.textContent = _aiOutput;
        }
    };
    const user = [
        `I'm following the workflow wizard "${_active?.title}", currently on step ${_active.idx + 1} of ${_active.steps.length}.`,
        `Step title: ${step.title}`,
        `Target node type: ${step.node_type || "(any)"}`,
        step.widget ? `Target widget: ${step.widget}` : "",
        `Step hint (already shown): ${step.hint || "(none)"}`,
        "",
        "In ≤5 short bullet points, explain WHY this step matters in the wider workflow — what role this node/widget plays, common values, and what breaks if we skip or mis-set it. Be concrete.",
    ].filter(Boolean).join("\n");
    try {
        await streamAI({
            feature: "wizard_explain",
            sensitivity: "normal",
            max_tokens: 320,
            temperature: 0.3,
            messages: [
                { role: "system", content: "You are a ComfyUI workflow tutor. Be terse and concrete." },
                { role: "user", content: user },
            ],
            onChunk,
            onError: (e) => { _aiOutput += `\n[error: ${e}]`; },
            onDone: () => { /* finalised */ },
        });
    } catch (e) {
        _aiOutput += `\n[exception: ${e?.message || e}]`;
    } finally {
        _aiBusy = false;
        _renderBody();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function _esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
}

// ─────────────────────────────────────────────────────────────────────────────
// Extension entry
// ─────────────────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "C2C.WorkflowWizard",
    settings: [
        {
            id: "c2c.workflow_wizard.enabled",
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
        _loadFavs();
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("c2c.workflow_wizard.enabled", true); }
            catch { return true; }
        })();
        const b = document.getElementById(BTN_ID);
        if (b) b.style.display = enabled ? "flex" : "none";
        console.log("[C2C.WorkflowWizard] godlevel-rebuild loaded.");
    },
});

// ── P7: Live validator + auto-fix (Face-Director plan, Wizard tier 1) ────────
// Scans the graph on every change for the two highest-value classes of silent
// breakage: (a) COMBO widget values that no longer exist in the live options
// (stale model filenames after a workflow reload), and (b) required link
// inputs left unconnected. Issues surface as a bottom-left "⚠ N issues" pill;
// clicking it opens a fix-it list with one-click actions.
(function () {
    const PILL_ID = "c2c-wizard-validator-pill";
    const LIST_ID = "c2c-wizard-validator-list";
    let _objInfo = null;
    let _issues = [];
    let _timer = 0;

    async function _defs() {
        if (_objInfo) return _objInfo;
        try {
            const r = await fetch("/object_info");
            _objInfo = await r.json();
        } catch (_) { _objInfo = {}; }
        return _objInfo;
    }

    function _isLinkType(spec) {
        const t = Array.isArray(spec) ? spec[0] : spec;
        return typeof t === "string" && /^[A-Z][A-Z0-9_]*$/.test(t) &&
            !["INT", "FLOAT", "STRING", "BOOLEAN"].includes(t);
    }

    async function _scan() {
        const defs = await _defs();
        const out = [];
        for (const n of (app.graph?._nodes || [])) {
            const def = defs[n.comfyClass || n.type];
            if (!def) continue;
            const req = (def.input && def.input.required) || {};
            // (a) stale combo values
            for (const w of (n.widgets || [])) {
                const spec = req[w.name];
                if (!spec || !Array.isArray(spec) || !Array.isArray(spec[0])) continue;
                const opts = spec[0];
                if (opts.length && w.value != null && !opts.includes(w.value)) {
                    out.push({
                        kind: "stale_combo", node: n, widget: w, opts,
                        msg: `${n.title || n.type}: "${w.name}" is set to a file that no longer exists (${String(w.value).slice(0, 40)})`,
                        fix: `Set to ${String(opts[0]).slice(0, 32)}`,
                        apply: () => { w.value = opts[0]; try { w.callback?.(opts[0], app.canvas, n); } catch (_) {} n.setDirtyCanvas?.(true, true); },
                    });
                }
            }
            // (b) unconnected required link inputs
            for (const [name, spec] of Object.entries(req)) {
                if (!_isLinkType(spec)) continue;
                const slot = (n.inputs || []).find((i) => i.name === name);
                if (slot && slot.link == null) {
                    out.push({
                        kind: "missing_link", node: n,
                        msg: `${n.title || n.type}: required input "${name}" is not connected`,
                        fix: "Show node",
                        apply: () => { try { app.canvas.centerOnNode(n); app.canvas.selectNode(n); } catch (_) {} },
                    });
                }
            }
        }
        _issues = out;
        _render();
    }

    function _pill() {
        let p = document.getElementById(PILL_ID);
        if (p) return p;
        p = document.createElement("button");
        p.id = PILL_ID;
        p.style.cssText =
            "position:fixed;left:74px;bottom:14px;z-index:9000;display:none;" +
            "padding:4px 12px;border-radius:999px;border:1px solid var(--c2c-dangerBg,#3b2222);" +
            "background:var(--c2c-bg2,#1a1a23);color:var(--c2c-red,#f38ba8);" +
            "font:600 11px ui-sans-serif,system-ui;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.4);";
        p.addEventListener("click", _toggleList);
        document.body.appendChild(p);
        return p;
    }

    function _toggleList() {
        let l = document.getElementById(LIST_ID);
        if (l) { l.remove(); return; }
        l = document.createElement("div");
        l.id = LIST_ID;
        l.style.cssText =
            "position:fixed;left:74px;bottom:44px;z-index:9001;width:380px;max-height:46vh;" +
            "overflow:auto;background:var(--c2c-bg,#1a1a22);border:1px solid var(--c2c-surface2,#45475a);" +
            "border-radius:10px;padding:8px;box-shadow:0 8px 28px rgba(0,0,0,.5);" +
            "font:11px ui-sans-serif,system-ui;color:var(--c2c-fg,#cdd6f4);";
        for (const iss of _issues) {
            const row = document.createElement("div");
            row.style.cssText = "display:flex;gap:8px;align-items:center;padding:6px;border-bottom:1px solid var(--c2c-surface0,#2a2a35);";
            const txt = document.createElement("div");
            txt.style.cssText = "flex:1;line-height:1.4;";
            txt.textContent = iss.msg;
            const btn = document.createElement("button");
            btn.textContent = iss.fix;
            btn.style.cssText =
                "flex:0 0 auto;padding:3px 10px;border-radius:6px;cursor:pointer;border:none;" +
                "background:var(--c2c-blue,#89b4fa);color:var(--c2c-bg3,#11111b);font:600 10px ui-sans-serif;";
            btn.addEventListener("click", () => { try { iss.apply(); } catch (_) {} setTimeout(_scan, 250); });
            row.append(txt, btn);
            l.appendChild(row);
        }
        if (!_issues.length) {
            l.textContent = "No issues — workflow looks healthy.";
            l.style.padding = "14px";
        }
        document.body.appendChild(l);
    }

    function _render() {
        const p = _pill();
        if (_issues.length) {
            p.style.display = "block";
            p.textContent = `⚠ ${_issues.length} workflow issue${_issues.length > 1 ? "s" : ""}`;
        } else {
            p.style.display = "none";
            document.getElementById(LIST_ID)?.remove();
        }
    }

    function _schedule() {
        if (_timer) clearTimeout(_timer);
        _timer = setTimeout(() => { _timer = 0; _scan().catch(() => {}); }, 700);
    }

    const _arm = () => {
        try {
            if (typeof app?.api?.addEventListener === "function") {
                app.api.addEventListener("graphChanged", _schedule);
            }
        } catch (_) {}
        // Safety-net poll: cheap (object_info cached; scan is O(nodes)).
        setInterval(_schedule, 5000);
        _schedule();
    };
    if (window.app?.graph) _arm(); else setTimeout(_arm, 1500);
})();
