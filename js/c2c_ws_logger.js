// c2c_ws_logger.js — WebSocket Event Ring Buffer + Sidebar (C2C v2.0 §8.2)
// ---------------------------------------------------------------------
// What it does:
//   • Subscribes to `app.api` events (status, executing, progress,
//     executed, execution_start, execution_error, execution_cached, b_preview).
//   • Keeps the last N=500 events in a ring buffer with monotonic counter.
//   • Renders a sidebar tab "WS Log" with:
//       - filter dropdown (event type)
//       - search box (substring across JSON)
//       - live tail toggle
//       - pause / resume
//       - clear
//       - copy-as-JSON button (selected or all)
//   • Hotkey Ctrl+Shift+W toggles the sidebar.
//   • Settings: c2c.wsLogger.enabled, c2c.wsLogger.buffer_size.
//
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";

const SETTING_ENABLED = "c2c.wsLogger.enabled";
const SETTING_BUFFER = "c2c.wsLogger.buffer_size";
const TAB_ID = "c2c.wsLogger";

const EVENT_TYPES = [
    "status", "executing", "progress", "executed",
    "execution_start", "execution_error", "execution_cached",
    "execution_interrupted", "b_preview",
];

let _enabled = true;
let _bufSize = 500;
let _seq = 0;
const _ring = []; // newest at end
let _paused = false;
let _attached = false;
let _root = null;

function pushEvent(type, detail) {
    _seq++;
    const entry = { seq: _seq, t: Date.now(), type, detail };
    _ring.push(entry);
    while (_ring.length > _bufSize) _ring.shift();
    if (_root && !_paused) renderRows();
}

function attachListeners() {
    if (_attached || !app.api) return;
    _attached = true;
    for (const t of EVENT_TYPES) {
        try {
            app.api.addEventListener(t, (ev) => {
                if (!_enabled) return;
                // Strip non-serializable / heavy keys (b_preview Blob).
                let d;
                try {
                    d = ev && ev.detail !== undefined ? ev.detail : null;
                    if (d instanceof Blob) d = { _blob: true, size: d.size, type: d.type };
                } catch { d = null; }
                pushEvent(t, d);
            });
        } catch (e) {
            console.warn("[c2c_ws_logger] failed to attach", t, e);
        }
    }
}

// ── Sidebar UI ────────────────────────────────────────────────────────
let _filterType = "all";
let _searchText = "";
let _selected = new Set();

function renderTabBody(el) {
    _root = el;
    el.style.cssText = "display:flex; flex-direction:column; height:100%; font:12px ui-sans-serif, system-ui, sans-serif; color:var(--c2c-accentText);";

    const toolbar = document.createElement("div");
    toolbar.style.cssText = "display:flex; flex-wrap:wrap; gap:6px; padding:8px; border-bottom:1px solid rgba(255,255,255,0.06); align-items:center;";

    const sel = document.createElement("select");
    sel.style.cssText = "background:rgba(255,255,255,0.05); color:var(--c2c-accentText); border:1px solid rgba(255,255,255,0.1); padding:3px 6px; border-radius:4px;";
    for (const v of ["all", ...EVENT_TYPES]) {
        const o = document.createElement("option");
        o.value = v; o.textContent = v; sel.appendChild(o);
    }
    sel.value = _filterType;
    sel.addEventListener("change", () => { _filterType = sel.value; renderRows(); });
    toolbar.appendChild(sel);

    const search = document.createElement("input");
    search.type = "text"; search.placeholder = "search…";
    search.style.cssText = "flex:1; min-width:80px; background:rgba(255,255,255,0.05); color:var(--c2c-accentText); border:1px solid rgba(255,255,255,0.1); padding:3px 6px; border-radius:4px;";
    search.addEventListener("input", () => { _searchText = search.value; renderRows(); });
    toolbar.appendChild(search);

    const pause = mkBtn(_paused ? "▶" : "❚❚", () => {
        _paused = !_paused; pause.textContent = _paused ? "▶" : "❚❚";
        pause.title = _paused ? "Resume" : "Pause";
        if (!_paused) renderRows();
    });
    pause.title = "Pause";
    toolbar.appendChild(pause);

    toolbar.appendChild(mkBtn("Clear", () => { _ring.length = 0; _selected.clear(); renderRows(); }));
    toolbar.appendChild(mkBtn("Copy", async () => {
        const data = _selected.size ? _ring.filter(e => _selected.has(e.seq)) : filterRing();
        try { await navigator.clipboard.writeText(JSON.stringify(data, null, 2)); } catch {}
    }));

    el.appendChild(toolbar);

    const list = document.createElement("div");
    list.id = "c2c-ws-log-list";
    list.style.cssText = "overflow:auto; flex:1; font:11px ui-monospace, monospace;";
    el.appendChild(list);

    const status = document.createElement("div");
    status.id = "c2c-ws-log-status";
    status.style.cssText = "padding:4px 8px; border-top:1px solid rgba(255,255,255,0.06); color:var(--c2c-accentMuted2); font-size:10px;";
    el.appendChild(status);

    renderRows();
}

function filterRing() {
    let out = _ring;
    if (_filterType !== "all") out = out.filter(e => e.type === _filterType);
    if (_searchText) {
        const q = _searchText.toLowerCase();
        out = out.filter(e => {
            try { return JSON.stringify(e.detail).toLowerCase().includes(q) || e.type.includes(q); }
            catch { return e.type.includes(q); }
        });
    }
    return out;
}

function renderRows() {
    if (!_root) return;
    const list = _root.querySelector("#c2c-ws-log-list");
    const status = _root.querySelector("#c2c-ws-log-status");
    if (!list) return;
    const filt = filterRing();
    // Render newest first for tail readability.
    const view = filt.slice(-200).reverse();
    list.innerHTML = "";
    for (const e of view) {
        const row = document.createElement("div");
        const color = colorForType(e.type);
        const selected = _selected.has(e.seq);
        row.style.cssText = `padding:3px 8px; cursor:pointer; border-left:3px solid ${color}; ${selected ? "background:rgba(91,141,239,0.18);" : ""}`;
        const time = new Date(e.t).toISOString().slice(11, 23);
        const detail = summarize(e.detail);
        row.innerHTML = `<span style="color:var(--c2c-accentMuted2);">${time}</span> <span style="color:${color}; font-weight:600;">${e.type}</span> <span style="color:var(--c2c-accentLight2);">${escapeHtml(detail)}</span>`;
        row.addEventListener("click", (ev) => {
            if (ev.shiftKey) {
                if (_selected.has(e.seq)) _selected.delete(e.seq);
                else _selected.add(e.seq);
            } else {
                _selected.clear(); _selected.add(e.seq);
                showDetail(e);
            }
            renderRows();
        });
        list.appendChild(row);
    }
    if (status) status.textContent = `${filt.length} shown · ${_ring.length}/${_bufSize} buffered · ${_paused ? "PAUSED" : "live"}`;
}

function colorForType(t) {
    return {
        status: "var(--c2c-accentMuted2)",
        executing: "var(--c2c-accentSoft2)",
        progress: "var(--c2c-cyanMid)",
        executed: "var(--c2c-okMute)",
        execution_start: "var(--c2c-violetMid)",
        execution_error: "var(--c2c-dangerMid)",
        execution_cached: "var(--c2c-amberDim)",
        execution_interrupted: "var(--c2c-amberMid)",
        b_preview: "var(--c2c-gray400)",
    }[t] || "var(--c2c-accentLight2)";
}

function summarize(d) {
    if (d === null || d === undefined) return "";
    try {
        if (typeof d === "string") return d;
        if (d.node !== undefined) return `node=${d.node}` + (d.value !== undefined ? ` value=${d.value}` : "") + (d.max !== undefined ? `/${d.max}` : "");
        if (d.prompt_id !== undefined) return `prompt=${String(d.prompt_id).slice(0, 8)}`;
        const j = JSON.stringify(d);
        return j.length > 160 ? j.slice(0, 157) + "…" : j;
    } catch { return ""; }
}

function showDetail(entry) {
    const root = document.createElement("div");
    root.style.cssText = "position:fixed; inset:0; z-index: var(--c2c-z-modal, 10000); background:rgba(0,0,0,0.55); display:flex; align-items:center; justify-content:center;";
    const dlg = document.createElement("div");
    dlg.style.cssText = "width:min(720px,90vw); max-height:80vh; overflow:auto; background:var(--c2c-panelDeep3); color:var(--c2c-accentText); padding:14px 18px; border:1px solid rgba(255,255,255,0.14); border-radius:8px; font:12px ui-sans-serif;";
    dlg.innerHTML = `<div style="display:flex;justify-content:space-between;margin-bottom:8px;"><b>${entry.type}</b> <span style="color:var(--c2c-accentMuted2);">seq ${entry.seq} · ${new Date(entry.t).toISOString()}</span></div>`;
    const pre = document.createElement("pre");
    pre.style.cssText = "background:var(--c2c-panelDeep4); padding:10px; border-radius:6px; overflow:auto; max-height:60vh; font:11px ui-monospace, monospace;";
    pre.textContent = JSON.stringify(entry.detail, null, 2);
    dlg.appendChild(pre);
    root.appendChild(dlg);
    document.body.appendChild(root);
    const close = () => { root.remove(); document.removeEventListener("keydown", onKey); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    root.addEventListener("click", (e) => { if (e.target === root) close(); });
    document.addEventListener("keydown", onKey);
}

function mkBtn(label, onClick) {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = "background:rgba(255,255,255,0.06); color:var(--c2c-accentLight2); border:1px solid rgba(255,255,255,0.1); border-radius:4px; padding:2px 8px; cursor:pointer;";
    b.addEventListener("click", onClick);
    return b;
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]);
}

function toggleSidebar() {
    try {
        if (app.extensionManager && app.extensionManager.toggleSidebarTab) {
            app.extensionManager.toggleSidebarTab(TAB_ID);
        }
    } catch (e) { console.warn("[c2c_ws_logger] toggle failed", e); }
}

app.registerExtension({
    name: "C2C.WSLogger",
    async setup() {
        app.ui.settings.addSetting({
            id: SETTING_ENABLED, name: "C2C ▸ WebSocket Logger: enabled",
            type: "boolean", defaultValue: true,
            onChange: v => { _enabled = !!v; if (_enabled) attachListeners(); },
        });
        app.ui.settings.addSetting({
            id: SETTING_BUFFER, name: "C2C ▸ WebSocket Logger: buffer size",
            type: "slider", attrs: { min: 50, max: 5000, step: 50 },
            defaultValue: 500,
            onChange: v => { _bufSize = Math.max(50, parseInt(v) || 500); while (_ring.length > _bufSize) _ring.shift(); renderRows(); },
        });
        _enabled = app.ui.settings.getSettingValue(SETTING_ENABLED, true);
        _bufSize = app.ui.settings.getSettingValue(SETTING_BUFFER, 500);

        if (app.extensionManager && app.extensionManager.registerSidebarTab) {
            app.extensionManager.registerSidebarTab({
                id: TAB_ID,
                icon: "pi pi-history",
                title: "WS Log",
                tooltip: "WebSocket Event Log (Ctrl+Shift+W)",
                type: "custom",
                render: (el) => renderTabBody(el),
            });
        }

        if (_enabled) attachListeners();

        window.addEventListener("keydown", (e) => {
            if (!_enabled) return;
            if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === "w") {
                e.preventDefault(); toggleSidebar();
            }
        }, true);
    },
});
