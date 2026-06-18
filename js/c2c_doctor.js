/**
 * c2c_doctor.js - 13-tab Doctor mega-panel (P0.5).
 *
 * Right-side slide-in overlay. Resizable via the left-edge handle.
 * Auto-runs the Workflow Lint tab on every Queue Prompt.
 *
 * Tabs (13):
 *   1.  Workflow Lint        -> /c2c/doctor/analyze
 *   2.  Model Inventory      -> /object_info (extract model enums)
 *   3.  Performance          -> /c2c/int/runs
 *   4.  Custom Node Versions -> /c2c/depcheck/snapshot
 *   5.  Python Env           -> /c2c/doctor/pyenv
 *   6.  Disk Usage           -> /c2c/doctor/disk
 *   7.  GPU Curves           -> /system_stats (polled, sparkline)
 *   8.  Error Log            -> in-memory ring (window error + console.error
 *                              + 'c2c:registry-failure' events)
 *   9.  Version Status       -> /c2c/registry/status
 *  10.  Settings             -> ComfyUI settings store, filter c2c.* / mec.*
 *  11.  History              -> /c2c/int/runs (timeline grouped)
 *  12.  Metadata Integrity   -> POST /c2c/doctor/scan_file (drag-drop)
 *  13.  Damaged Package      -> POST /c2c/doctor/scan_file (zip/whl/.ckpt)
 *
 * Every tab reads live data. There are no static placeholders.
 * Errors from a route are rendered in-place rather than silently swallowed.
 *
 * Hook points:
 *   - window.__C2C_DOCTOR__.open()   -- open & focus panel
 *   - window.__C2C_DOCTOR__.run()    -- open & re-run lint
 *   - window.__C2C_DOCTOR__.status() -- {sev: 'ok'|'warn'|'err'|'crit', count}
 *   - emits CustomEvent('c2c:doctor-status', {detail: status()}) on body
 *     so the INT badge / OmniBar can subscribe.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { attachWindowChrome } from "./_c2c_window.js";

// ---------------------------------------------------------------------------
// constants & helpers
// ---------------------------------------------------------------------------
const PANEL_ID  = "c2c-doctor-v3-panel";
const STYLE_ID  = "c2c-doctor-v3-style";
const BTN_ID    = "c2c-doctor-v3-btn";
const LS_TAB    = "c2c.doctor.v3.tab";

const TABS = [
    { key: "lint",     label: "Workflow Lint",        icon: "L" },
    { key: "models",   label: "Model Inventory",      icon: "M" },
    { key: "perf",     label: "Performance",          icon: "P" },
    { key: "deps",     label: "Custom Node Versions", icon: "V" },
    { key: "pyenv",    label: "Python Env",           icon: "Y" },
    { key: "disk",     label: "Disk Usage",           icon: "D" },
    { key: "gpu",      label: "GPU Curves",           icon: "G" },
    { key: "errors",   label: "Error Log",            icon: "E" },
    { key: "versions", label: "Version Status",       icon: "S" },
    { key: "settings", label: "Settings",             icon: "*" },
    { key: "history",  label: "History",              icon: "H" },
    { key: "meta",     label: "Metadata Integrity",   icon: "I" },
    { key: "pkg",      label: "Damaged Package",      icon: "Z" },
];

const ESC_MAP = { "&": "&amp;", "<": "&lt;", ">": "&gt;",
                  '"': "&quot;", "'": "&#39;" };
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ESC_MAP[c]);

function tok(name, fallback = "") {
    try {
        const v = getComputedStyle(document.documentElement)
            .getPropertyValue(name).trim();
        return v || fallback;
    } catch { return fallback; }
}

function fmtBytes(n) {
    if (n == null || isNaN(n)) return "-";
    const u = ["B","KB","MB","GB","TB","PB"];
    let i = 0; let v = Number(n);
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(v >= 100 || i === 0 ? 0 : v >= 10 ? 1 : 2)} ${u[i]}`;
}

function fmtMs(ms) {
    if (ms == null) return "-";
    if (ms < 1000) return `${Math.round(ms)} ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(2)} s`;
    return `${(ms / 60000).toFixed(1)} min`;
}

function fmtTs(ts) {
    if (!ts) return "-";
    const d = new Date(typeof ts === "number" ? (ts < 1e12 ? ts * 1000 : ts) : ts);
    if (isNaN(+d)) return String(ts);
    return d.toLocaleString();
}

async function fetchJSON(url, opts) {
    try {
        const r = await fetch(url, opts);
        const t = await r.text();
        let j; try { j = JSON.parse(t); } catch { return { _httpStatus: r.status, _raw: t }; }
        j._httpStatus = r.status;
        return j;
    } catch (exc) {
        return { _fetchError: String(exc && exc.message || exc) };
    }
}

// ---------------------------------------------------------------------------
// Error ring (Error Log tab data source)
// ---------------------------------------------------------------------------
const ERR_RING_MAX = 250;
const _errRing = [];

function _pushErr(kind, payload) {
    const entry = { ts: Date.now(), kind, ...payload };
    _errRing.push(entry);
    if (_errRing.length > ERR_RING_MAX) _errRing.splice(0, _errRing.length - ERR_RING_MAX);
}

(function installErrorHooks() {
    if (window.__C2C_DOCTOR_ERR_HOOKED__) return;
    window.__C2C_DOCTOR_ERR_HOOKED__ = true;

    const origErr = console.error.bind(console);
    console.error = function (...args) {
        try {
            const msg = args.map((a) =>
                a instanceof Error ? `${a.message}\n${a.stack || ""}`
                                   : typeof a === "string" ? a
                                   : JSON.stringify(a)).join(" ");
            // Suppress our own re-entrancy
            if (!String(msg).includes("[c2c-doctor]")) {
                _pushErr("console.error", { message: String(msg).slice(0, 2000) });
            }
        } catch {}
        return origErr(...args);
    };
    window.addEventListener("error", (e) => {
        _pushErr("window.error", {
            message: String(e.message || ""),
            source: String(e.filename || ""),
            line: e.lineno, col: e.colno,
        });
    });
    window.addEventListener("unhandledrejection", (e) => {
        let msg = "";
        try { msg = (e.reason && (e.reason.stack || e.reason.message)) || String(e.reason); }
        catch { msg = "<unserializable>"; }
        _pushErr("unhandledrejection", { message: String(msg).slice(0, 2000) });
    });
    document.addEventListener("c2c:registry-failure", (e) => {
        const d = e.detail || {};
        _pushErr("registry-failure", {
            message: d.message || d.error || JSON.stringify(d).slice(0, 500),
            module: d.module || d.feature || "",
        });
    });
    // B2 — Server-side execution errors (ComfyUI native event).
    // `execution_error` fires once per failed prompt with the offending node
    // id/type + Python traceback. We push it into the same ring so the
    // Doctor's Error Log tab shows it alongside client-side errors, and the
    // user can hit "Explain" to get an AI-powered plain-English breakdown.
    try {
        if (api && typeof api.addEventListener === "function") {
            api.addEventListener("execution_error", (ev) => {
                const d = (ev && ev.detail) || {};
                _pushErr("execution_error", {
                    message:    String(d.exception_message || d.message || "<no message>"),
                    traceback:  Array.isArray(d.traceback)
                                  ? d.traceback.join("")
                                  : String(d.traceback || ""),
                    nodeId:     d.node_id || d.node || null,
                    nodeType:   d.node_type || "",
                    excType:    d.exception_type || "",
                });
            });
        }
    } catch { /* api may be unavailable in some bootstraps */ }
})();

// ---------------------------------------------------------------------------
// GPU samples ring (GPU Curves tab data source)
// ---------------------------------------------------------------------------
const GPU_RING_MAX = 180;  // 6 min at 2s cadence
const _gpuRing = [];
let _gpuPollTimer = null;

function _gpuPollOnce() {
    fetchJSON("/system_stats").then((j) => {
        if (!j || j._fetchError) return;
        const dev = (j.devices && j.devices[0]) || {};
        const samp = {
            ts: Date.now(),
            vram_used: Number(dev.vram_total) - Number(dev.vram_free || 0),
            vram_total: Number(dev.vram_total || 0),
            vram_free: Number(dev.vram_free || 0),
            ram_used: Number((j.system && j.system.ram_total) || 0) - Number((j.system && j.system.ram_free) || 0),
            ram_total: Number((j.system && j.system.ram_total) || 0),
        };
        _gpuRing.push(samp);
        if (_gpuRing.length > GPU_RING_MAX) _gpuRing.shift();
    });
}

function _ensureGpuPolling() {
    if (_gpuPollTimer != null) return;
    _gpuPollOnce();
    _gpuPollTimer = setInterval(_gpuPollOnce, 2000);
}

// ---------------------------------------------------------------------------
// Last lint result + status broadcast
// ---------------------------------------------------------------------------
let _lastLint = null;

function _doctorStatus() {
    const f = (_lastLint && _lastLint.report && _lastLint.report.findings) || [];
    let err = 0, warn = 0, info = 0;
    for (const x of f) {
        const s = (x.severity || "info").toLowerCase();
        if (s === "error" || s === "critical") err++;
        else if (s === "warning" || s === "warn") warn++;
        else info++;
    }
    const sev = err > 0 ? "err" : warn > 0 ? "warn" : "ok";
    return { sev, error: err, warning: warn, info, total: f.length };
}

function _broadcastStatus() {
    const s = _doctorStatus();
    try {
        document.body.dispatchEvent(new CustomEvent("c2c:doctor-status", { detail: s }));
    } catch {}
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------
function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const el = document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = `
#${PANEL_ID} {
    position: fixed; top: 0; right: 0; height: 100vh;
    width: 540px; min-width: 380px; max-width: 95vw;
    background: var(--c2c-bg, var(--c2c-bg2)); color: var(--c2c-fg, var(--c2c-accentBright));
    border-left: 1px solid var(--c2c-border, var(--c2c-gray800));
    box-shadow: -8px 0 32px rgba(0,0,0,.4);
    display: grid; grid-template-rows: auto auto 1fr auto;
    z-index: var(--c2c-z-popover);
    transform: translateX(100%);
    transition: transform .22s ease-out;
    font: 12px/1.4 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
#${PANEL_ID}.open { transform: translateX(0); }
#${PANEL_ID} .c2c-doc-hdr {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 12px;
    border-bottom: 1px solid var(--c2c-border, var(--c2c-gray800));
    background: var(--c2c-bg-elev, var(--c2c-bg));
}
#${PANEL_ID} .c2c-doc-title { font-size: 14px; font-weight: 600; flex: 1; }
#${PANEL_ID} .c2c-doc-iconbtn {
    background: transparent; border: 1px solid var(--c2c-border, var(--c2c-gray700));
    color: var(--c2c-fg, var(--c2c-accentBright)); padding: 3px 8px; border-radius: 4px;
    cursor: pointer; font: inherit;
}
#${PANEL_ID} .c2c-doc-iconbtn:hover { background: var(--c2c-bg-hover, var(--c2c-panelBg)); }
#${PANEL_ID} .c2c-doc-tabs {
    display: flex; flex-wrap: wrap; gap: 2px;
    padding: 6px 8px;
    border-bottom: 1px solid var(--c2c-border, var(--c2c-gray800));
    background: var(--c2c-bg-elev, var(--c2c-bg));
}
#${PANEL_ID} .c2c-doc-tab {
    padding: 4px 8px; border-radius: 4px;
    background: transparent; border: 1px solid transparent;
    color: var(--c2c-fg-dim, var(--c2c-gray300)); cursor: pointer; font: inherit;
    white-space: nowrap;
}
#${PANEL_ID} .c2c-doc-tab:hover { background: var(--c2c-bg-hover, var(--c2c-panelBg)); color: var(--c2c-fg, var(--c2c-accentBright)); }
#${PANEL_ID} .c2c-doc-tab.active {
    background: var(--c2c-accent-bg, var(--c2c-panelBg));
    color: var(--c2c-accent, var(--c2c-accentSoft));
    border-color: var(--c2c-accent, var(--c2c-accentSoft));
}
#${PANEL_ID} .c2c-doc-body {
    overflow: auto; padding: 12px;
}
#${PANEL_ID} .c2c-doc-footer {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 12px;
    border-top: 1px solid var(--c2c-border, var(--c2c-gray800));
    background: var(--c2c-bg-elev, var(--c2c-bg));
    font-size: 11px; color: var(--c2c-fg-dim, var(--c2c-gray400));
}
#${PANEL_ID} h3 { font-size: 12px; margin: 0 0 8px; color: var(--c2c-accent, var(--c2c-accentSoft));
                   text-transform: uppercase; letter-spacing: .04em; }
#${PANEL_ID} .c2c-doc-section { margin-bottom: 16px; }
#${PANEL_ID} table { width: 100%; border-collapse: collapse; font-size: 11px; }
#${PANEL_ID} th, #${PANEL_ID} td {
    text-align: left; padding: 4px 8px;
    border-bottom: 1px solid var(--c2c-border-soft, var(--c2c-panelBg));
    vertical-align: top;
}
#${PANEL_ID} th { color: var(--c2c-fg-dim, var(--c2c-gray360)); font-weight: 600; }
#${PANEL_ID} .c2c-doc-pill {
    display: inline-block; padding: 1px 6px; border-radius: 10px;
    font-size: 10px; font-weight: 600;
}
#${PANEL_ID} .pill-err  { background: var(--c2c-panelBg); color: var(--c2c-dangerTint); }
#${PANEL_ID} .pill-warn { background: var(--c2c-surface0); color: var(--c2c-yellow); }
#${PANEL_ID} .pill-info { background: var(--c2c-surface0); color: var(--c2c-accentLink); }
#${PANEL_ID} .pill-ok   { background: var(--c2c-surface0); color: var(--c2c-teal); }
#${PANEL_ID} .c2c-doc-finding {
    border: 1px solid var(--c2c-border, var(--c2c-gray800)); border-radius: 5px;
    padding: 8px 10px; margin-bottom: 6px;
    background: var(--c2c-bg-elev, var(--c2c-bg));
}
#${PANEL_ID} .c2c-doc-finding .fhead {
    display: flex; gap: 8px; align-items: center; margin-bottom: 4px;
}
#${PANEL_ID} .c2c-doc-finding .ftitle { font-weight: 600; flex: 1; }
#${PANEL_ID} .c2c-doc-finding .fbody { color: var(--c2c-fg-dim, var(--c2c-gray250)); font-size: 11px; }
#${PANEL_ID} .c2c-doc-finding .ffix {
    margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap;
}
#${PANEL_ID} .c2c-doc-drop {
    border: 2px dashed var(--c2c-border, var(--c2c-gray600)); border-radius: 6px;
    padding: 24px; text-align: center; cursor: pointer;
    color: var(--c2c-fg-dim, var(--c2c-gray300));
}
#${PANEL_ID} .c2c-doc-drop.over { border-color: var(--c2c-accent, var(--c2c-accentSoft)); background: rgba(80,180,255,.06); }
#${PANEL_ID} .c2c-doc-empty { color: var(--c2c-fg-dim, var(--c2c-gray400)); font-style: italic; padding: 12px; text-align: center; }
#${PANEL_ID} .c2c-doc-spark { height: 60px; background: var(--c2c-bg-elev, var(--c2c-bg));
                              border: 1px solid var(--c2c-border-soft, var(--c2c-panelBg)); border-radius: 4px;
                              margin-bottom: 4px; display: block; width: 100%; }
#${PANEL_ID} .c2c-doc-kv { display: grid; grid-template-columns: 160px 1fr; gap: 2px 12px; font-size: 11px; }
#${PANEL_ID} .c2c-doc-kv > div:nth-child(odd) { color: var(--c2c-fg-dim, var(--c2c-gray360)); }
#${PANEL_ID} pre.c2c-doc-pre {
    background: var(--c2c-bg-elev, var(--c2c-bg)); border: 1px solid var(--c2c-border-soft, var(--c2c-panelBg));
    border-radius: 4px; padding: 8px; font-size: 11px; line-height: 1.4;
    max-height: 360px; overflow: auto; white-space: pre-wrap; word-break: break-word;
}
#${BTN_ID} {
    position: fixed;
    top: calc(var(--c2c-native-top, 48px) + 4px);
    right: 8px;
    background: var(--c2c-bg-elev, var(--c2c-bg)); color: var(--c2c-fg, var(--c2c-accentBright));
    border: 1px solid var(--c2c-border, var(--c2c-gray700)); border-radius: 4px;
    padding: 4px 10px; font: 12px/1 ui-sans-serif, sans-serif;
    cursor: pointer; z-index: var(--c2c-z-dock, 2500);
    /* Fallback only — hidden when OmniBar is present */
}
#${BTN_ID}.sev-warn { border-color: var(--c2c-warnBright); color: var(--c2c-warnSoft); }
#${BTN_ID}.sev-err  { border-color: var(--c2c-danger); color: var(--c2c-dangerTint); }
`;
    document.head.appendChild(el);
}

// ---------------------------------------------------------------------------
// Panel construction
// ---------------------------------------------------------------------------
let _panel = null;
let _panelRefs = null;

function buildPanel() {
    injectStyles();
    const root = document.createElement("aside");
    root.id = PANEL_ID;
    root.setAttribute("role", "complementary");
    root.setAttribute("aria-label", "C2C Doctor");

    root.innerHTML = `
<div class="c2c-doc-hdr">
    <div class="c2c-doc-title">C2C Doctor</div>
    <button class="c2c-doc-iconbtn" data-act="rerun"   title="Re-run lint">Re-run</button>
    <button class="c2c-doc-iconbtn" data-act="refresh" title="Refresh active tab">Refresh</button>
    <button class="c2c-doc-iconbtn" data-act="close"   title="Close">X</button>
</div>
<div class="c2c-doc-tabs">
    ${TABS.map((t) => `<button class="c2c-doc-tab" data-tab="${t.key}" title="${esc(t.label)}">${esc(t.label)}</button>`).join("")}
</div>
<div class="c2c-doc-body" data-role="body">
    <div class="c2c-doc-empty">Loading...</div>
</div>
<div class="c2c-doc-footer">
    <span data-role="status">ready</span>
    <span style="flex:1"></span>
    <span data-role="counts"></span>
</div>
`;
    document.body.appendChild(root);

    const $ = (sel) => root.querySelector(sel);
    const refs = {
        root,
        body: $('[data-role="body"]'),
        status: $('[data-role="status"]'),
        counts: $('[data-role="counts"]'),
        tabs: Array.from(root.querySelectorAll(".c2c-doc-tab")),
        active: null,
    };

    // tab clicks
    refs.tabs.forEach((b) => {
        b.addEventListener("click", () => setActiveTab(b.dataset.tab));
    });
    // header buttons
    root.querySelector('[data-act="close"]').addEventListener("click", closePanel);
    root.querySelector('[data-act="rerun"]').addEventListener("click", runLint);
    root.querySelector('[data-act="refresh"]').addEventListener("click", () => refreshActive());
    // Esc closes
    root.addEventListener("keydown", (e) => { if (e.key === "Escape") closePanel(); });

    // Window chrome (drag / 8-edge resize / minimize). Doctor has no
    // overlay backdrop — it's a right-edge aside, so omit `overlay`.
    const _hdr   = root.querySelector(".c2c-doc-hdr");
    const _title = root.querySelector(".c2c-doc-title");
    attachWindowChrome(root, {
        storageKey: "doctor",
        header: _hdr,
        titleEl: _title,
        minW: 380, minH: 320,
    });

    _panel = root;
    _panelRefs = refs;
    return refs;
}

function ensurePanel() {
    if (_panelRefs) return _panelRefs;
    return buildPanel();
}

function openPanel() {
    const refs = ensurePanel();
    refs.root.classList.add("open");
    refs.root.tabIndex = -1;
    refs.root.focus({ preventScroll: true });
    const last = localStorage.getItem(LS_TAB) || "lint";
    if (!refs.active) setActiveTab(last);
}

function closePanel() {
    if (_panel) _panel.classList.remove("open");
}

function setActiveTab(key) {
    const refs = ensurePanel();
    if (!TABS.find((t) => t.key === key)) key = "lint";
    refs.active = key;
    localStorage.setItem(LS_TAB, key);
    refs.tabs.forEach((b) => b.classList.toggle("active", b.dataset.tab === key));
    refs.body.innerHTML = '<div class="c2c-doc-empty">Loading...</div>';
    refs.status.textContent = "loading " + key + "...";
    Promise.resolve().then(() => renderTab(key, refs))
        .then(() => { refs.status.textContent = "ok"; })
        .catch((exc) => {
            refs.body.innerHTML =
                `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">render error: ${esc(exc && exc.message || exc)}</div>`;
            refs.status.textContent = "error";
            // eslint-disable-next-line no-console
            console.error("[c2c-doctor] tab render", key, exc);
        });
}

function refreshActive() {
    const refs = ensurePanel();
    if (refs.active) setActiveTab(refs.active);
}

// ---------------------------------------------------------------------------
// Tab dispatcher
// ---------------------------------------------------------------------------
async function renderTab(key, refs) {
    const map = {
        lint: renderLint, models: renderModels, perf: renderPerf,
        deps: renderDeps, pyenv: renderPyEnv, disk: renderDisk,
        gpu: renderGpu, errors: renderErrors, versions: renderVersions,
        settings: renderSettings, history: renderHistory,
        meta: renderMeta, pkg: renderPkg,
    };
    const fn = map[key];
    if (!fn) {
        refs.body.innerHTML = `<div class="c2c-doc-empty">unknown tab: ${esc(key)}</div>`;
        return;
    }
    await fn(refs.body, refs);
}

// ---------------------------------------------------------------------------
// Tab 1: Workflow Lint
// ---------------------------------------------------------------------------
async function runLint() {
    const wf = app.graph && app.graph.serialize && app.graph.serialize();
    if (!wf) {
        _lastLint = { report: { success: false, error: "no_graph" } };
        _broadcastStatus();
        return _lastLint;
    }
    const j = await fetchJSON("/c2c/doctor/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ workflow: wf }),
    });
    _lastLint = { ts: Date.now(), report: j };
    _broadcastStatus();
    if (_panelRefs && _panelRefs.active === "lint") refreshActive();
    if (_panelRefs) updateCounts();
    return _lastLint;
}

function updateCounts() {
    if (!_panelRefs) return;
    const s = _doctorStatus();
    _panelRefs.counts.innerHTML =
        `<span class="c2c-doc-pill pill-err">err ${s.error}</span> ` +
        `<span class="c2c-doc-pill pill-warn">warn ${s.warning}</span> ` +
        `<span class="c2c-doc-pill pill-info">info ${s.info}</span>`;
    // Toolbar button colour
    const btn = document.getElementById(BTN_ID);
    if (btn) {
        btn.classList.remove("sev-warn", "sev-err");
        if (s.error > 0) btn.classList.add("sev-err");
        else if (s.warning > 0) btn.classList.add("sev-warn");
    }
}

function _focusNode(nid) {
    const n = app.graph && app.graph.getNodeById && app.graph.getNodeById(nid);
    if (!n) return;
    try {
        app.canvas.deselectAllNodes && app.canvas.deselectAllNodes();
        app.canvas.selectNode && app.canvas.selectNode(n);
        if (app.canvas.centerOnNode) app.canvas.centerOnNode(n);
        app.canvas.setDirty && app.canvas.setDirty(true, true);
    } catch {}
}

async function _applyFix(fix) {
    if (!fix || !fix.node_id) return;
    const n = app.graph.getNodeById(fix.node_id);
    if (!n) return;
    try {
        if (fix.kind === "set_widget" && fix.widget) {
            const w = (n.widgets || []).find((w) => w.name === fix.widget);
            if (w) { w.value = fix.value; w.callback && w.callback(w.value); }
        } else if (fix.kind === "set_widget_many" && Array.isArray(fix.changes)) {
            for (const ch of fix.changes) {
                const w = (n.widgets || []).find((w) => w.name === ch.widget);
                if (w) { w.value = ch.value; w.callback && w.callback(w.value); }
            }
        } else if (fix.kind === "set_mode") {
            n.mode = fix.value;
        }
        app.graph.setDirtyCanvas(true, true);
    } catch (exc) {
        // eslint-disable-next-line no-console
        console.error("[c2c-doctor] autofix failed", exc);
    }
}

async function renderLint(body) {
    if (!_lastLint) await runLint();
    const rep = _lastLint && _lastLint.report;
    if (!rep) { body.innerHTML = '<div class="c2c-doc-empty">no graph</div>'; return; }
    if (rep.success === false) {
        body.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">analyze failed: ${esc(rep.error || rep._fetchError || "unknown")}</div>`;
        return;
    }
    const findings = rep.findings || [];
    updateCounts();
    if (findings.length === 0) {
        body.innerHTML =
            `<div class="c2c-doc-section">` +
            `<h3>Workflow Lint</h3>` +
            `<div class="c2c-doc-empty">All clear. ${esc(rep.rules_ran || "")} rules checked.</div>` +
            `</div>`;
        return;
    }
    const bySev = { error: [], warning: [], info: [] };
    for (const f of findings) {
        const k = (f.severity || "info").toLowerCase();
        (bySev[k === "critical" ? "error" : k === "warn" ? "warning" : k] || bySev.info).push(f);
    }
    const sevName = { error: "Errors", warning: "Warnings", info: "Info" };
    const sevPill = { error: "err", warning: "warn", info: "info" };
    let html = `<div class="c2c-doc-section"><h3>Workflow Lint</h3>`;
    for (const sev of ["error", "warning", "info"]) {
        const arr = bySev[sev];
        if (!arr.length) continue;
        html += `<div style="margin: 6px 0 8px;"><b>${esc(sevName[sev])}</b> <span class="c2c-doc-pill pill-${sevPill[sev]}">${arr.length}</span></div>`;
        for (const f of arr) {
            const nid = f.node_id != null ? f.node_id : (f.nodeId != null ? f.nodeId : null);
            const rule = f.rule || f.code || "rule";
            html += `<div class="c2c-doc-finding">
              <div class="fhead">
                <span class="c2c-doc-pill pill-${sevPill[sev]}">${esc(rule)}</span>
                <span class="ftitle">${esc(f.message || f.title || "")}</span>
                ${nid != null ? `<button class="c2c-doc-iconbtn" data-jump="${esc(nid)}" title="Focus node">Node #${esc(nid)}</button>` : ""}
              </div>
              ${f.detail ? `<div class="fbody">${esc(f.detail)}</div>` : ""}
              ${Array.isArray(f.fixes) && f.fixes.length ? `<div class="ffix">${
                  f.fixes.map((fx, i) =>
                      `<button class="c2c-doc-iconbtn" data-fix="${i}" data-rule="${esc(rule)}">${esc(fx.label || ("Apply fix " + (i+1)))}</button>`
                  ).join("")
              }</div>` : ""}
            </div>`;
            // attach fixes via dataset workaround below
            f._fixes = f.fixes || [];
        }
    }
    html += `</div>`;
    body.innerHTML = html;
    // wire up jump + autofix
    body.querySelectorAll("[data-jump]").forEach((b) => {
        b.addEventListener("click", () => _focusNode(Number(b.dataset.jump)));
    });
    body.querySelectorAll("[data-fix]").forEach((b) => {
        b.addEventListener("click", async () => {
            const card = b.closest(".c2c-doc-finding");
            if (!card) return;
            // find this finding by index in flat order
            const allCards = Array.from(body.querySelectorAll(".c2c-doc-finding"));
            const idx = allCards.indexOf(card);
            const flat = [...bySev.error, ...bySev.warning, ...bySev.info];
            const f = flat[idx];
            const fix = f && f._fixes && f._fixes[Number(b.dataset.fix)];
            if (!fix) return;
            b.disabled = true; b.textContent = "applying...";
            await _applyFix(fix);
            b.textContent = "applied";
            setTimeout(runLint, 300);
        });
    });
}

// ---------------------------------------------------------------------------
// Tab 2: Model Inventory
// ---------------------------------------------------------------------------
// Pull the live /object_info and extract model-enum widget options.
// Different nodes own the canonical list for each folder.
const MODEL_SOURCES = [
    { folder: "checkpoints",     node: "CheckpointLoaderSimple", input: "ckpt_name"    },
    { folder: "loras",           node: "LoraLoader",             input: "lora_name"    },
    { folder: "vae",             node: "VAELoader",              input: "vae_name"     },
    { folder: "clip",            node: "CLIPLoader",             input: "clip_name"    },
    { folder: "clip_vision",     node: "CLIPVisionLoader",       input: "clip_name"    },
    { folder: "controlnet",      node: "ControlNetLoader",       input: "control_net_name" },
    { folder: "upscale_models",  node: "UpscaleModelLoader",     input: "model_name"   },
    { folder: "embeddings",      node: "LoadLatent",             input: "latent"       }, // not used; placeholder
    { folder: "diffusion_models",node: "UNETLoader",             input: "unet_name"    },
    { folder: "style_models",    node: "StyleModelLoader",       input: "style_model_name" },
    { folder: "gligen",          node: "GLIGENLoader",           input: "gligen_name"  },
    { folder: "hypernetworks",   node: "HypernetworkLoader",     input: "hypernetwork_name" },
];

async function renderModels(body) {
    const oi = await fetchJSON("/object_info");
    if (!oi || oi._fetchError) {
        body.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">/object_info failed: ${esc(oi && oi._fetchError || "unknown")}</div>`;
        return;
    }
    const rows = [];
    let total = 0;
    for (const src of MODEL_SOURCES) {
        const node = oi[src.node];
        if (!node || !node.input) continue;
        const inp = (node.input.required && node.input.required[src.input])
                 || (node.input.optional && node.input.optional[src.input]);
        if (!inp || !Array.isArray(inp[0])) continue;
        const names = inp[0].filter((x) => typeof x === "string");
        total += names.length;
        rows.push({
            folder: src.folder, count: names.length,
            samples: names.slice(0, 3), all: names,
        });
    }
    if (!rows.length) {
        body.innerHTML = `<div class="c2c-doc-empty">No model folders detected in /object_info.</div>`;
        return;
    }
    rows.sort((a, b) => b.count - a.count);
    body.innerHTML =
        `<div class="c2c-doc-section">
          <h3>Model Inventory</h3>
          <div style="margin-bottom:8px">Total entries across ${rows.length} folders: <b>${total}</b></div>
          <table>
            <thead><tr><th>Folder</th><th>Count</th><th>Sample</th><th></th></tr></thead>
            <tbody>
              ${rows.map((r, i) => `
                <tr>
                  <td><code>${esc(r.folder)}</code></td>
                  <td>${r.count}</td>
                  <td>${esc(r.samples.join(", ") || "-")}</td>
                  <td><button class="c2c-doc-iconbtn" data-expand="${i}">List...</button></td>
                </tr>
                <tr data-detail="${i}" style="display:none">
                  <td colspan="4"><pre class="c2c-doc-pre">${esc(r.all.join("\n"))}</pre></td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>`;
    body.querySelectorAll("[data-expand]").forEach((b) => {
        b.addEventListener("click", () => {
            const row = body.querySelector(`[data-detail="${b.dataset.expand}"]`);
            if (!row) return;
            const showing = row.style.display !== "none";
            row.style.display = showing ? "none" : "";
            b.textContent = showing ? "List..." : "Hide";
        });
    });
}

// ---------------------------------------------------------------------------
// Tab 3: Performance
// ---------------------------------------------------------------------------
async function renderPerf(body) {
    const j = await fetchJSON("/c2c/int/runs?n=200");
    if (!j || j._fetchError) {
        body.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">/c2c/int/runs failed: ${esc(j && j._fetchError || "unknown")}</div>`;
        return;
    }
    const items = Array.isArray(j.items) ? j.items : (Array.isArray(j) ? j : []);
    if (!items.length) {
        body.innerHTML = `<div class="c2c-doc-empty">No execution telemetry yet. Run a workflow to populate.</div>`;
        return;
    }
    // aggregate per node type-ish via node_id frequency
    const byNode = new Map();
    let totalMs = 0; let totalVram = 0; let nVram = 0;
    for (const e of items) {
        if (typeof e.elapsed_ms === "number") totalMs += e.elapsed_ms;
        if (typeof e.vram_peak_mb === "number") { totalVram += e.vram_peak_mb; nVram++; }
        const k = String(e.node_id || "?");
        const cur = byNode.get(k) || { node_id: k, n: 0, ms: 0, peak: 0 };
        cur.n++;
        cur.ms += Number(e.elapsed_ms) || 0;
        cur.peak = Math.max(cur.peak, Number(e.vram_peak_mb) || 0);
        byNode.set(k, cur);
    }
    const top = Array.from(byNode.values()).sort((a, b) => b.ms - a.ms).slice(0, 30);

    body.innerHTML =
        `<div class="c2c-doc-section">
          <h3>Performance</h3>
          <div class="c2c-doc-kv">
            <div>samples</div><div>${items.length}</div>
            <div>total elapsed</div><div>${fmtMs(totalMs)}</div>
            <div>avg vram peak</div><div>${nVram ? (totalVram/nVram).toFixed(0)+" MB" : "-"}</div>
          </div>
        </div>
        <div class="c2c-doc-section">
          <h3>Hottest nodes</h3>
          <table>
            <thead><tr><th>node_id</th><th>runs</th><th>total</th><th>peak vram</th></tr></thead>
            <tbody>
              ${top.map((r) => `
                <tr>
                  <td><button class="c2c-doc-iconbtn" data-jump="${esc(r.node_id)}">#${esc(r.node_id)}</button></td>
                  <td>${r.n}</td>
                  <td>${fmtMs(r.ms)}</td>
                  <td>${r.peak ? r.peak.toFixed(0)+" MB" : "-"}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>`;
    body.querySelectorAll("[data-jump]").forEach((b) => {
        b.addEventListener("click", () => _focusNode(Number(b.dataset.jump)));
    });
}

// ---------------------------------------------------------------------------
// Tab 4: Custom Node Versions
// ---------------------------------------------------------------------------
async function renderDeps(body) {
    const j = await fetchJSON("/c2c/depcheck/snapshot");
    if (!j || j._fetchError || j.success === false) {
        body.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">/c2c/depcheck/snapshot failed: ${esc(j && (j._fetchError || j.error) || "unknown")}</div>`;
        return;
    }
    const dirs = (j.data && j.data.dirs) || [];
    body.innerHTML =
        `<div class="c2c-doc-section">
          <h3>Custom Node Packs (${dirs.length})</h3>
          <table>
            <thead><tr><th>Path</th></tr></thead>
            <tbody>
              ${dirs.map((d) => `<tr><td><code>${esc(d)}</code></td></tr>`).join("")}
            </tbody>
          </table>
        </div>
        <div class="c2c-doc-section">
          <h3>Scan for new packs</h3>
          <button class="c2c-doc-iconbtn" data-act="scan">Scan against current as baseline</button>
          <pre class="c2c-doc-pre" data-role="scan-out" style="display:none"></pre>
        </div>`;
    body.querySelector('[data-act="scan"]').addEventListener("click", async (e) => {
        const out = body.querySelector('[data-role="scan-out"]');
        out.style.display = ""; out.textContent = "scanning...";
        const r = await fetchJSON("/c2c/depcheck/scan_new", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ baseline: dirs }),
        });
        out.textContent = JSON.stringify(r, null, 2);
    });
}

// ---------------------------------------------------------------------------
// Tab 5: Python Env
// ---------------------------------------------------------------------------
async function renderPyEnv(body) {
    const j = await fetchJSON("/c2c/doctor/pyenv");
    if (!j || j._fetchError || j.success === false) {
        body.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">/c2c/doctor/pyenv failed: ${esc(j && (j._fetchError || j.error) || "unknown")}</div>`;
        return;
    }
    const py = j.python || {}; const cv = j.comfy || {}; const dv = j.device || {};
    const pkgs = j.packages || [];
    const installed = pkgs.filter((p) => p.installed);
    const missing = pkgs.filter((p) => !p.installed);

    let devHtml = "";
    if (dv.cuda_available) {
        devHtml = `<div class="c2c-doc-kv">
            <div>torch</div><div>${esc(dv.torch)}</div>
            <div>CUDA</div><div>${esc(dv.cuda)}</div>
            <div>cuDNN</div><div>${esc(dv.cudnn)}</div>
            <div>devices</div><div>${dv.device_count}</div>
          </div>
          <table style="margin-top:6px">
            <thead><tr><th>idx</th><th>name</th><th>VRAM</th><th>capability</th></tr></thead>
            <tbody>${(dv.devices || []).map((d) => `
              <tr><td>${d.index}</td><td>${esc(d.name)}</td>
                  <td>${fmtBytes(d.total_mem)}</td><td>${esc(d.capability)}</td></tr>
            `).join("")}</tbody>
          </table>`;
    } else {
        devHtml = `<div class="c2c-doc-empty">torch.cuda not available${dv.error ? ` (${esc(dv.error)})` : ""}</div>`;
    }

    body.innerHTML =
        `<div class="c2c-doc-section">
          <h3>Runtime</h3>
          <div class="c2c-doc-kv">
            <div>Python</div><div>${esc(py.version)} on ${esc(py.platform)}</div>
            <div>executable</div><div><code>${esc(py.executable)}</code></div>
            <div>ComfyUI version</div><div>${esc(cv.comfyui_version || "(not exposed)")}</div>
          </div>
        </div>
        <div class="c2c-doc-section">
          <h3>Device</h3>
          ${devHtml}
        </div>
        <div class="c2c-doc-section">
          <h3>Packages (${installed.length} installed, ${missing.length} missing)</h3>
          <table>
            <thead><tr><th>name</th><th>version</th></tr></thead>
            <tbody>
              ${installed.map((p) => `<tr><td>${esc(p.name)}</td><td>${esc(p.version)}</td></tr>`).join("")}
              ${missing.map((p) => `<tr><td>${esc(p.name)}</td><td><span class="c2c-doc-pill pill-warn">missing</span></td></tr>`).join("")}
            </tbody>
          </table>
        </div>`;
}

// ---------------------------------------------------------------------------
// Tab 6: Disk Usage
// ---------------------------------------------------------------------------
async function renderDisk(body) {
    body.innerHTML = `<div class="c2c-doc-empty">scanning disk... (this can take a few seconds on first run)</div>`;
    const j = await fetchJSON("/c2c/doctor/disk");
    if (!j || j._fetchError || j.success === false) {
        body.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">/c2c/doctor/disk failed: ${esc(j && (j._fetchError || j.error) || "unknown")}</div>`;
        return;
    }
    const drive = j.drive || {};
    const sections = j.sections || {};
    const breakdown = (j.models_breakdown || []).sort((a, b) => b.bytes - a.bytes);
    const usedPct = drive.total ? Math.round(100 * drive.used / drive.total) : 0;

    body.innerHTML =
        `<div class="c2c-doc-section">
          <h3>Drive ${j.cached ? `(cached ${j.age_s}s)` : "(fresh)"}</h3>
          <div class="c2c-doc-kv">
            <div>root</div><div><code>${esc(j.root)}</code></div>
            <div>total</div><div>${fmtBytes(drive.total)}</div>
            <div>used</div><div>${fmtBytes(drive.used)} (${usedPct}%)</div>
            <div>free</div><div>${fmtBytes(drive.free)}</div>
          </div>
          <div style="margin-top:6px">
            <button class="c2c-doc-iconbtn" data-act="refresh-disk">Recompute (bust cache)</button>
          </div>
        </div>
        <div class="c2c-doc-section">
          <h3>ComfyUI sections</h3>
          <table>
            <thead><tr><th>section</th><th>files</th><th>size</th></tr></thead>
            <tbody>
              ${Object.entries(sections).map(([k, v]) => `
                <tr>
                  <td><code>${esc(k)}</code></td>
                  <td>${v.missing ? "-" : v.files}</td>
                  <td>${v.missing ? "<span class=\"c2c-doc-pill pill-info\">missing</span>" : fmtBytes(v.bytes)}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
        <div class="c2c-doc-section">
          <h3>models/ breakdown</h3>
          <table>
            <thead><tr><th>folder</th><th>files</th><th>size</th></tr></thead>
            <tbody>
              ${breakdown.map((b) => `
                <tr><td><code>${esc(b.name)}</code></td><td>${b.files}</td><td>${fmtBytes(b.bytes)}</td></tr>
              `).join("")}
            </tbody>
          </table>
        </div>`;
    body.querySelector('[data-act="refresh-disk"]').addEventListener("click", async () => {
        body.innerHTML = `<div class="c2c-doc-empty">recomputing...</div>`;
        await fetchJSON("/c2c/doctor/disk?refresh=1");
        refreshActive();
    });
}

// ---------------------------------------------------------------------------
// Tab 7: GPU Curves
// ---------------------------------------------------------------------------
function _spark(ctx, w, h, data, max, color) {
    if (!data.length) return;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.beginPath();
    const step = w / Math.max(1, data.length - 1);
    for (let i = 0; i < data.length; i++) {
        const v = max > 0 ? data[i] / max : 0;
        const x = i * step;
        const y = h - v * (h - 4) - 2;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    // fill
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = color + "22"; ctx.fill();
}

async function renderGpu(body, refs) {
    _ensureGpuPolling();
    body.innerHTML =
        `<div class="c2c-doc-section">
          <h3>VRAM usage (60 min window, polled every 2s)</h3>
          <canvas class="c2c-doc-spark" data-role="vram"></canvas>
          <div class="c2c-doc-kv" data-role="vram-kv"></div>
        </div>
        <div class="c2c-doc-section">
          <h3>System RAM</h3>
          <canvas class="c2c-doc-spark" data-role="ram"></canvas>
          <div class="c2c-doc-kv" data-role="ram-kv"></div>
        </div>
        <div style="color:var(--c2c-fg-dim,var(--c2c-gray400));font-size:11px">samples buffered: <span data-role="n">0</span></div>`;
    const vramC = body.querySelector('[data-role="vram"]');
    const ramC  = body.querySelector('[data-role="ram"]');

    const draw = () => {
        if (!_panelRefs || _panelRefs.active !== "gpu") return;
        const W = vramC.clientWidth, H = vramC.clientHeight;
        vramC.width = W; vramC.height = H;
        ramC.width  = W; ramC.height  = H;
        const vctx = vramC.getContext("2d");
        const rctx = ramC.getContext("2d");
        const vramData = _gpuRing.map((s) => s.vram_used);
        const ramData  = _gpuRing.map((s) => s.ram_used);
        const vramMax  = _gpuRing.length ? _gpuRing[_gpuRing.length - 1].vram_total : 0;
        const ramMax   = _gpuRing.length ? _gpuRing[_gpuRing.length - 1].ram_total : 0;
        _spark(vctx, W, H, vramData, vramMax, "var(--c2c-accentSoft)");
        _spark(rctx, W, H, ramData, ramMax, "var(--c2c-warn)");
        const last = _gpuRing[_gpuRing.length - 1];
        if (last) {
            body.querySelector('[data-role="vram-kv"]').innerHTML =
                `<div>used</div><div>${fmtBytes(last.vram_used)} / ${fmtBytes(last.vram_total)}</div>
                 <div>free</div><div>${fmtBytes(last.vram_free)}</div>`;
            body.querySelector('[data-role="ram-kv"]').innerHTML =
                `<div>used</div><div>${fmtBytes(last.ram_used)} / ${fmtBytes(last.ram_total)}</div>`;
        }
        body.querySelector('[data-role="n"]').textContent = String(_gpuRing.length);
    };
    draw();
    const iv = setInterval(() => {
        if (!_panelRefs || _panelRefs.active !== "gpu" || !document.body.contains(body)) {
            clearInterval(iv); return;
        }
        draw();
    }, 1000);
}

// ---------------------------------------------------------------------------
// Tab 8: Error Log
// ---------------------------------------------------------------------------
async function renderErrors(body) {
    const items = _errRing.slice().reverse();
    if (!items.length) {
        body.innerHTML = `<div class="c2c-doc-section">
          <h3>Error Log</h3>
          <div class="c2c-doc-empty">No errors captured this session.</div>
        </div>`;
        return;
    }
    // B2 — each row gets a stable index so the AI Explain card has a
    // place to attach. We render newest-first but index against the
    // original ring position so Clear/refresh stays stable until next
    // tab switch.
    const pillFor = (kind) => {
        if (kind === "execution_error") return "err";
        if (kind === "console.error")   return "warn";
        return "err";
    };
    body.innerHTML = `<div class="c2c-doc-section">
      <h3>Error Log (${items.length}, newest first)</h3>
      <button class="c2c-doc-iconbtn" data-act="clear">Clear</button>
      <table style="margin-top:8px">
        <thead><tr><th>time</th><th>kind</th><th>message</th><th style="width:80px">explain</th></tr></thead>
        <tbody>
          ${items.map((e, i) => {
            const isExec = e.kind === "execution_error";
            const nodeLabel = isExec
              ? `<br><small>node ${esc(String(e.nodeId||"?"))} \u00b7 ${esc(e.nodeType||"")}</small>`
              : "";
            const srcLabel = e.source
              ? `<br><small>${esc(e.source)}:${e.line||0}</small>`
              : "";
            // Any captured error is explainable — message alone is enough.
            return `
            <tr data-row="${i}">
              <td style="white-space:nowrap">${fmtTs(e.ts)}</td>
              <td><span class="c2c-doc-pill pill-${pillFor(e.kind)}">${esc(e.kind)}</span></td>
              <td><div style="max-width:340px;word-break:break-word">${esc(e.message || "")}${nodeLabel}${srcLabel}</div></td>
              <td><button class="c2c-doc-iconbtn" data-explain="${i}" title="AI explainer">Explain</button></td>
            </tr>
            <tr data-card="${i}" style="display:none"><td colspan="4"><div class="c2c-doc-explain-host" style="padding:8px 4px"></div></td></tr>`;
          }).join("")}
        </tbody>
      </table>
    </div>`;
    body.querySelector('[data-act="clear"]').addEventListener("click", () => {
        _errRing.length = 0;
        refreshActive();
    });
    body.querySelectorAll('[data-explain]').forEach((btn) => {
        btn.addEventListener("click", async () => {
            const i = parseInt(btn.getAttribute("data-explain"), 10);
            const entry = items[i];
            if (!entry) return;
            const cardRow = body.querySelector(`[data-card="${i}"]`);
            const host = cardRow.querySelector(".c2c-doc-explain-host");
            // Toggle if already shown and not loading
            if (cardRow.style.display !== "none" && !btn._busy) {
                cardRow.style.display = "none";
                return;
            }
            cardRow.style.display = "";
            host.innerHTML = `<div class="c2c-doc-empty">Asking the Doctor\u2026</div>`;
            btn._busy = true; btn.disabled = true;
            try {
                const r = await fetch("/c2c/doctor/explain", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        kind:      entry.kind,
                        message:   entry.message || "",
                        traceback: entry.traceback || "",
                        node_id:   entry.nodeId,
                        node_type: entry.nodeType,
                    }),
                });
                const j = await r.json();
                if (!j || !j.success) {
                    const err = (j && j.error) || `HTTP ${r.status}`;
                    host.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">Explain failed: ${esc(String(err))}</div>`;
                    return;
                }
                const ex = j.data.explanation || {};
                const sevCls = ex.severity === "critical" ? "err"
                             : ex.severity === "error"    ? "err"
                             : ex.severity === "warning"  ? "warn" : "ok";
                const fixes = Array.isArray(ex.fixes) ? ex.fixes : [];
                host.innerHTML = `<div class="c2c-doc-section" style="margin:0;border-left:3px solid var(--c2c-accent,var(--c2c-accentSoft));padding-left:10px">
                  <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:6px">
                    <span class="c2c-doc-pill pill-${sevCls}">${esc(ex.severity || "unknown")}</span>
                    <span class="c2c-doc-pill">${esc(ex.category || "unknown")}</span>
                    <small style="opacity:.65">${esc(j.data.backend_id || "")} \u00b7 ${esc(j.data.model || "")} \u00b7 ${j.data.latency_ms||0}ms</small>
                  </div>
                  <div style="font-weight:600;margin-bottom:4px">${esc(ex.summary || "")}</div>
                  ${ex.root_cause ? `<div style="margin-bottom:6px">${esc(ex.root_cause)}</div>` : ""}
                  ${fixes.length ? `<div><b>Fixes:</b><ol style="margin:4px 0 4px 18px">${
                      fixes.map((f) => `<li>${esc(String(f))}</li>`).join("")
                  }</ol></div>` : ""}
                  ${ex.preventive ? `<div style="opacity:.85"><b>Prevent:</b> ${esc(ex.preventive)}</div>` : ""}
                  ${ex._raw_output ? `<details style="margin-top:6px"><summary style="cursor:pointer;opacity:.7">raw model output</summary><pre style="white-space:pre-wrap;font-size:11px">${esc(ex._raw_output)}</pre></details>` : ""}
                </div>`;
            } catch (e) {
                host.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">Explain failed: ${esc(String(e && e.message || e))}</div>`;
            } finally {
                btn._busy = false; btn.disabled = false;
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Tab 9: Version Status (registry)
// ---------------------------------------------------------------------------
async function renderVersions(body) {
    const j = await fetchJSON("/c2c/registry/status");
    if (!j || j._fetchError) {
        body.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">/c2c/registry/status failed: ${esc(j && j._fetchError || "unknown")}</div>`;
        return;
    }
    // Real payload shape from nodes/_c2c_registry.py:
    //   { failures:[{group,key,error,traceback,...}], statuses:[{group,key,status,...}], counts:{...} }
    const failures = Array.isArray(j.failures) ? j.failures : [];
    const statuses = Array.isArray(j.statuses) ? j.statuses : [];
    const counts   = j.counts || {};
    const byStatus = {};
    for (const s of statuses) {
        const k = s.status || "unknown";
        (byStatus[k] = byStatus[k] || []).push(s);
    }
    const fmtKey = (e) => {
        const g = e.group || ""; const k = e.key || e.name || e.feature || "";
        return g && k ? `${g}/${k}` : (k || g || JSON.stringify(e));
    };
    const tableOf = (arr, detailFn) => arr.length === 0
      ? `<div class="c2c-doc-empty">none</div>`
      : `<table><thead><tr><th>module</th><th>detail</th></tr></thead><tbody>${
            arr.map((e) => `<tr><td><code>${esc(fmtKey(e))}</code></td><td>${esc(detailFn(e))}</td></tr>`).join("")
        }</tbody></table>`;
    const countsHtml = Object.keys(counts).length
      ? `<div class="c2c-doc-section"><h3>Summary</h3><div>${
            Object.entries(counts).map(([k, v]) =>
                `<span class="c2c-doc-pill" style="margin-right:6px">${esc(k)}: ${esc(String(v))}</span>`
            ).join("")
        }</div></div>`
      : "";
    const statusOrder = ["ready", "experimental", "missing-deps", "missing-weights", "unknown"];
    const statusKeys = statusOrder.filter((k) => byStatus[k] && byStatus[k].length);
    for (const k of Object.keys(byStatus)) if (!statusKeys.includes(k)) statusKeys.push(k);
    const sections = statusKeys.map((k) => {
        const arr = byStatus[k];
        const detail = (e) => [e.reason, e.detail, e.hint, e.version, e.message]
                              .filter(Boolean).join(" \u00b7 ");
        return `<div class="c2c-doc-section"><h3>${esc(k)} (${arr.length})</h3>${tableOf(arr, detail)}</div>`;
    }).join("");
    const failHtml = `<div class="c2c-doc-section"><h3>Failures (${failures.length})</h3>${
        tableOf(failures, (e) => (e.error || e.message || "").toString().slice(0, 240))
    }</div>`;
    body.innerHTML = countsHtml + failHtml + sections;
}

// ---------------------------------------------------------------------------
// Tab 10: Settings
// ---------------------------------------------------------------------------
async function renderSettings(body) {
    const ss = app.ui && app.ui.settings;
    // ComfyUI settings store keeps registered settings in .settingsLookup
    // and current values in localStorage under "Comfy.Settings.<id>"
    const reg = (ss && (ss.settingsLookup || ss.settings)) || {};
    const ids = Object.keys(reg).filter((k) => /^(c2c\.|mec\.)/i.test(k)).sort();
    const rows = ids.map((id) => {
        const def = reg[id] || {};
        let val;
        try { val = ss.getSettingValue ? ss.getSettingValue(id) : undefined; }
        catch { val = undefined; }
        if (val === undefined) {
            try { val = JSON.parse(localStorage.getItem("Comfy.Settings." + id)); }
            catch { val = localStorage.getItem("Comfy.Settings." + id); }
        }
        return { id, name: def.name || id, type: def.type || typeof val, value: val };
    });
    if (!rows.length) {
        body.innerHTML = `<div class="c2c-doc-empty">No registered c2c.* or mec.* settings.</div>`;
        return;
    }
    body.innerHTML = `<div class="c2c-doc-section">
      <h3>C2C / MEC settings (${rows.length})</h3>
      <table>
        <thead><tr><th>id</th><th>name</th><th>value</th></tr></thead>
        <tbody>${rows.map((r) => `
          <tr>
            <td><code>${esc(r.id)}</code></td>
            <td>${esc(r.name)}</td>
            <td><code>${esc(JSON.stringify(r.value))}</code></td>
          </tr>
        `).join("")}</tbody>
      </table>
    </div>`;
}

// ---------------------------------------------------------------------------
// Tab 11: History
// ---------------------------------------------------------------------------
async function renderHistory(body) {
    const j = await fetchJSON("/c2c/int/runs?n=500");
    if (!j || j._fetchError) {
        body.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">/c2c/int/runs failed: ${esc(j && j._fetchError || "unknown")}</div>`;
        return;
    }
    const items = Array.isArray(j.items) ? j.items : (Array.isArray(j) ? j : []);
    if (!items.length) {
        body.innerHTML = `<div class="c2c-doc-empty">No history yet.</div>`;
        return;
    }
    // group by day
    const byDay = new Map();
    for (const e of items) {
        const ts = typeof e.ts === "number" ? (e.ts < 1e12 ? e.ts * 1000 : e.ts) : Date.parse(e.ts);
        const d = new Date(ts);
        const key = isNaN(+d) ? "unknown" : d.toISOString().slice(0, 10);
        if (!byDay.has(key)) byDay.set(key, []);
        byDay.get(key).push(e);
    }
    const days = Array.from(byDay.entries()).sort((a, b) => (a[0] < b[0] ? 1 : -1));
    body.innerHTML = days.map(([day, arr]) => `
      <div class="c2c-doc-section">
        <h3>${esc(day)} (${arr.length})</h3>
        <table>
          <thead><tr><th>time</th><th>type</th><th>node</th><th>elapsed</th><th>vram peak</th><th>exc</th></tr></thead>
          <tbody>${arr.slice(0, 100).map((e) => `
            <tr>
              <td style="white-space:nowrap">${fmtTs(e.ts)}</td>
              <td>${esc(e.type || "")}</td>
              <td>#${esc(e.node_id || "")}</td>
              <td>${fmtMs(e.elapsed_ms)}</td>
              <td>${e.vram_peak_mb != null ? e.vram_peak_mb.toFixed(0) + " MB" : "-"}</td>
              <td>${e.exc_type ? `<span class="c2c-doc-pill pill-err">${esc(e.exc_type)}</span> ${esc(e.exc_msg || "")}` : ""}</td>
            </tr>
          `).join("")}</tbody>
        </table>
      </div>
    `).join("");
}

// ---------------------------------------------------------------------------
// Tabs 12 & 13: drag-drop file scan
// ---------------------------------------------------------------------------
function _scanDropZone(body, intro) {
    body.innerHTML = `
    <div class="c2c-doc-section">
      <h3>${esc(intro.title)}</h3>
      <div style="color:var(--c2c-fg-dim,var(--c2c-gray360));margin-bottom:8px">${esc(intro.help)}</div>
      <div class="c2c-doc-drop" data-role="drop">
        Drop file here or <u>click to pick</u> (max 64 MiB)
      </div>
      <input type="file" data-role="file" hidden>
      <div data-role="result"></div>
    </div>`;
    const drop = body.querySelector('[data-role="drop"]');
    const file = body.querySelector('[data-role="file"]');
    const out  = body.querySelector('[data-role="result"]');
    drop.addEventListener("click", () => file.click());
    file.addEventListener("change", (e) => {
        const f = e.target.files && e.target.files[0];
        if (f) _doScan(f, out);
    });
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => {
        e.preventDefault(); drop.classList.remove("over");
        const f = e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) _doScan(f, out);
    });
}

async function _doScan(f, out) {
    if (f.size > 64 * 1024 * 1024) {
        out.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">File too large (${fmtBytes(f.size)}). Server cap is 64 MiB.</div>`;
        return;
    }
    out.innerHTML = `<div class="c2c-doc-empty">scanning ${esc(f.name)} (${fmtBytes(f.size)})...</div>`;
    const fd = new FormData();
    fd.append("file", f, f.name);
    let j;
    try {
        const r = await fetch("/c2c/doctor/scan_file", { method: "POST", body: fd });
        j = await r.json();
    } catch (exc) {
        out.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">scan failed: ${esc(exc.message)}</div>`;
        return;
    }
    if (j.success === false && j.error) {
        out.innerHTML = `<div class="c2c-doc-empty" style="color:var(--c2c-dangerSoft)">${esc(j.error)}: ${esc(j.detail || "")}</div>`;
        return;
    }
    const issues = j.issues || [];
    const sev = (s) => `<span class="c2c-doc-pill pill-${s === "error" ? "err" : s === "warning" ? "warn" : "info"}">${esc(s)}</span>`;
    const issuesHtml = issues.length === 0 ?
        `<div class="c2c-doc-empty">No issues found.</div>` :
        `<table>
          <thead><tr><th>sev</th><th>kind</th><th>detail</th></tr></thead>
          <tbody>${issues.map((i) => `
            <tr><td>${sev(i.severity || "info")}</td><td><code>${esc(i.kind)}</code></td>
                <td>${esc(i.where || i.detail || JSON.stringify(Object.fromEntries(Object.entries(i).filter(([k]) => !["severity","kind","where","detail"].includes(k)))))}</td></tr>
          `).join("")}</tbody>
        </table>`;
    out.innerHTML = `
      <div class="c2c-doc-section" style="margin-top:12px">
        <h3>Result</h3>
        <div class="c2c-doc-kv">
          <div>file</div><div><code>${esc(j.filename)}</code></div>
          <div>format</div><div>${esc(j.format || "-")}</div>
          <div>size</div><div>${fmtBytes(j.size)}</div>
          <div>SHA-256</div><div><code style="word-break:break-all">${esc(j.sha256)}</code></div>
          ${j.workflow_node_count ? `<div>workflow nodes</div><div>${j.workflow_node_count}</div>` : ""}
          ${j.tensor_count ? `<div>tensors</div><div>${j.tensor_count}</div>` : ""}
          ${j.entries ? `<div>zip entries</div><div>${j.entries}</div>` : ""}
        </div>
      </div>
      <div class="c2c-doc-section">
        <h3>Issues (${issues.length})</h3>
        ${issuesHtml}
      </div>
      ${j.embedded_metadata ? `<div class="c2c-doc-section">
        <h3>Embedded metadata</h3>
        <pre class="c2c-doc-pre">${esc(JSON.stringify(j.embedded_metadata, null, 2))}</pre>
      </div>` : ""}
      ${j.png_chunks ? `<div class="c2c-doc-section">
        <h3>PNG chunks</h3>
        <div><code>${esc(j.png_chunks.join(" "))}</code></div>
      </div>` : ""}
    `;
}

async function renderMeta(body) {
    _scanDropZone(body, {
        title: "Metadata Integrity",
        help: "Drag a .json workflow, .png screenshot with embedded workflow, " +
              "or .safetensors model. The server validates structure and " +
              "extracts embedded metadata. Files are inspected in-memory only.",
    });
}

async function renderPkg(body) {
    _scanDropZone(body, {
        title: "Damaged / Suspicious Package",
        help: "Drag a custom-node .zip, .whl, or legacy .ckpt/.pt. " +
              "The scanner checks the archive header, lists entries, and " +
              "flags risky Python patterns. Files are never executed.",
    });
}

// ---------------------------------------------------------------------------
// Toolbar button & extension registration
// ---------------------------------------------------------------------------
function ensureToolbarButton() {
    if (document.getElementById(BTN_ID)) return;
    const b = document.createElement("button");
    b.id = BTN_ID;
    b.textContent = "Doctor";
    b.title = "C2C Doctor (P0.5)";
    b.addEventListener("click", () => { openPanel(); runLint(); });
    document.body.appendChild(b);
    updateCounts();
}

function _hookOmniBar() {
    // Register Doctor as an OmniBar "tools" slot using the public
    // window.C2COmniBar.register() API. Falls back to the standalone
    // fixed-position button if the OmniBar hasn't loaded yet or is absent.
    const tryReg = () => {
        const ob = window.C2COmniBar;
        if (!ob || typeof ob.register !== "function") return false;
        try {
            // Build a pill element matching OmniBar slot conventions.
            const pill = document.createElement("button");
            // Stable DOM id for testability. Distinct from the legacy
            // standalone btn `c2c-doctor-btn` (handled by c2c_workflow_doctor.js
            // and removed by the cleanup line below) — using a unique id
            // prevents the cleanup from accidentally removing our pill.
            pill.id = "c2c-doctor-pill";
            pill.className = "c2c-omnibar-slot-pill";
            pill.title = "C2C Doctor — workflow lint & diagnostics";
            pill.style.cssText = "display:flex;align-items:center;gap:5px;";
            const dot = document.createElement("span");
            dot.style.cssText = "width:6px;height:6px;border-radius:50%;background:var(--c2c-okSoft);flex-shrink:0;";
            const label = document.createElement("span");
            label.textContent = "Doctor";
            pill.append(dot, label);
            pill.addEventListener("click", () => { openPanel(); runLint(); });

            // Keep dot color in sync with current Doctor severity.
            const _sync = () => {
                const s = _doctorStatus();
                dot.style.background = s.errors > 0 ? "var(--c2c-red)"
                                     : s.warnings > 0 ? "var(--c2c-warn)"
                                     : "var(--c2c-okSoft)";
            };
            document.addEventListener("c2c:doctor-status", _sync);

            const unreg = ob.register({
                section: "tools",
                // Use the same slot-id that _c2c_top_dock.js assigns when
                // c2c_workflow_doctor.js proxies #c2c-doctor-btn to OmniBar
                // (format: "<el.id>-<side>"). Registering the same id replaces
                // that older, lower-fidelity slot with V3's status-dot pill.
                id: "c2c-doctor-btn-left",
                order: 20,
                element: pill,
                onMode(mode) {
                    label.style.display = mode === "icon" ? "none" : "";
                },
                update: _sync,
            });
            if (typeof unreg === "function") {
                window.__C2C_DOCTOR_V3_OMNIBAR_UNREG__ = unreg;
            }

            // Remove standalone fallback buttons now that OmniBar hosts Doctor.
            document.getElementById(BTN_ID)?.remove();           // c2c-doctor-v3-btn
            document.getElementById("c2c-doctor-btn")?.remove(); // legacy workflow-doctor btn
            return true;
        } catch (exc) {
            // eslint-disable-next-line no-console
            console.error("[c2c-doctor] omnibar register", exc);
            return false;
        }
    };
    if (tryReg()) return;
    // Poll briefly while the OmniBar extension finishes booting.
    let n = 0;
    const iv = setInterval(() => {
        if (tryReg() || ++n > 40) clearInterval(iv);
    }, 500);
}

// Public hook (namespaced V3 to avoid clobbering the older
// c2c_workflow_doctor.js popup, which also owns window.__C2C_DOCTOR__).
window.__C2C_DOCTOR_V3__ = {
    open:   () => { openPanel(); },
    run:    () => { openPanel(); return runLint(); },
    status: () => _doctorStatus(),
};

app.registerExtension({
    name: "C2C.DoctorV3",
    async setup() {
        injectStyles();
        ensureToolbarButton();
        _hookOmniBar();
        // Remove the legacy v2 Workflow Doctor button (c2c_workflow_doctor.js).
        // It may load/register after V3 due to settings.onChange firing early;
        // sweep three times to cover any creation race.
        const rmV2 = () => document.getElementById("c2c-doctor-btn")?.remove();
        rmV2(); setTimeout(rmV2, 600); setTimeout(rmV2, 1500);
        // Auto-run lint on every Queue Prompt (execution_start fires
        // when the server accepts the queued prompt).
        try {
            api.addEventListener("execution_start", () => { runLint(); });
        } catch (exc) {
            // eslint-disable-next-line no-console
            console.error("[c2c-doctor] api.addEventListener", exc);
        }
        // First-paint lint (non-blocking)
        setTimeout(runLint, 1500);
    },
});
