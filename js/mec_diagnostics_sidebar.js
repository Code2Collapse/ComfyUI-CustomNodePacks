// MEC Diagnostics — sidebar tab.
//
// Coexists with rookiestar28/ComfyUI-Doctor. We register our OWN sidebar
// entry under id `mec.diagnostics` with a distinct icon, so both panels
// are visible in ComfyUI's left sidebar.
//
// Tabs:
//   1. Diagnostics      Live error feed from /mec/diagnostics/recent.
//   2. Statistics       Pattern + category counters from /mec/diagnostics/statistics.
//   3. Clipboard        History of MEC-clipboard payloads in this tab + replay.
//   4. Settings         error_assistant tier mode + reload patterns button.
//   5. Patterns         Read-only listing of loaded patterns (id/category/priority/source).

import { app } from "../../scripts/app.js";

const STYLE_TAG_ID = "mec-diagnostics-style";

// -----------------------------------------------------------------------
// Stylesheet (injected once)
// -----------------------------------------------------------------------
function _injectStyle() {
    if (document.getElementById(STYLE_TAG_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_TAG_ID;
    s.textContent = `
    .mec-diag-root {
        display:flex; flex-direction:column; height:100%;
        font-family: var(--p-font-family, system-ui), sans-serif;
        font-size: 12px; color: var(--fg-color, #cdd6f4);
        background: var(--bg-color, #1e1e2e);
    }
    .mec-diag-tabs {
        display:flex; gap:0; border-bottom:1px solid var(--border-color, #313244);
        flex-shrink:0;
    }
    .mec-diag-tab {
        flex:1; padding:6px 4px; text-align:center; cursor:pointer;
        background:transparent; border:none; color: inherit;
        border-bottom:2px solid transparent;
        font-size:11px;
    }
    .mec-diag-tab.active {
        border-bottom-color: var(--p-primary-color, #89b4fa);
        color: var(--p-primary-color, #89b4fa);
        font-weight:600;
    }
    .mec-diag-body {
        flex:1; overflow:auto; padding:8px;
    }
    .mec-diag-empty {
        color: var(--descriptions-text-color, #a6adc8);
        text-align:center; padding:24px 8px; font-style:italic;
    }
    .mec-diag-card {
        border:1px solid var(--border-color, #313244);
        border-radius:6px; padding:8px; margin-bottom:6px;
        background: var(--p-content-background, #181825);
    }
    .mec-diag-card.error  { border-left:3px solid #f38ba8; }
    .mec-diag-card.warn   { border-left:3px solid #fab387; }
    .mec-diag-card.info   { border-left:3px solid #89b4fa; }
    .mec-diag-card.ok     { border-left:3px solid #a6e3a1; }
    .mec-diag-row { display:flex; justify-content:space-between; gap:6px;
                    align-items:baseline; }
    .mec-diag-id  { font-weight:600; color: var(--fg-color, #cdd6f4); }
    .mec-diag-meta { color: var(--descriptions-text-color, #a6adc8);
                     font-size:10px; }
    .mec-diag-msg {
        font-family: ui-monospace, "Cascadia Mono", monospace;
        font-size:11px; word-break: break-word;
        color: var(--fg-color, #cdd6f4); margin-top:4px;
    }
    .mec-diag-tag {
        display:inline-block; padding:1px 6px; border-radius:10px;
        font-size:10px; background:#313244; color:#cdd6f4;
        margin-right:4px;
    }
    .mec-diag-tag.cat-memory   { background:#f38ba8; color:#11111b; }
    .mec-diag-tag.cat-cuda     { background:#cba6f7; color:#11111b; }
    .mec-diag-tag.cat-dtype    { background:#fab387; color:#11111b; }
    .mec-diag-tag.cat-model_loading { background:#94e2d5; color:#11111b; }
    .mec-diag-tag.cat-environment   { background:#f9e2af; color:#11111b; }
    .mec-diag-tag.cat-dataflow      { background:#74c7ec; color:#11111b; }
    .mec-diag-tag.cat-workflow      { background:#89b4fa; color:#11111b; }
    .mec-diag-btn {
        padding:4px 10px; border-radius:4px;
        border:1px solid var(--border-color, #45475a);
        background: var(--p-button-secondary-bg, #313244);
        color: var(--fg-color, #cdd6f4);
        cursor:pointer; font-size:11px;
    }
    .mec-diag-btn:hover { background: var(--p-button-secondary-hover-bg, #45475a); }
    .mec-diag-btn.primary {
        background: var(--p-primary-color, #89b4fa);
        color: var(--p-primary-color-text, #11111b);
        border-color: transparent; font-weight:600;
    }
    .mec-diag-toolbar {
        display:flex; gap:6px; padding-bottom:6px; flex-wrap:wrap;
    }
    .mec-diag-bar {
        height:4px; border-radius:2px; background:#313244; overflow:hidden;
        margin-top:3px;
    }
    .mec-diag-bar > div { height:100%; background:var(--p-primary-color, #89b4fa); }
    .mec-diag-kv { display:grid; grid-template-columns: 1fr auto; gap:4px 8px; }
    .mec-diag-kv .k { color: var(--descriptions-text-color, #a6adc8); }
    .mec-diag-kv .v { font-family: ui-monospace, monospace; }
    .mec-diag-input, .mec-diag-select {
        width:100%; padding:4px 6px; border-radius:4px;
        border:1px solid var(--border-color, #45475a);
        background: var(--p-content-background, #181825);
        color: var(--fg-color, #cdd6f4); font-size:11px;
    }
    `;
    document.head.appendChild(s);
}

// -----------------------------------------------------------------------
// Backend helpers
// -----------------------------------------------------------------------
async function _api(path, opts = {}) {
    try {
        const r = await fetch(path, opts);
        const j = await r.json().catch(() => ({ success: false, error: "bad_json", message: "Non-JSON reply" }));
        return j;
    } catch (e) {
        return { success: false, error: "network", message: String(e) };
    }
}

function _fmtTs(ts) {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString();
}

function _toast(msg, severity = "info") {
    try {
        app.extensionManager?.toast?.add({
            severity, summary: "MEC Diagnostics", detail: msg, life: 3000,
        });
    } catch (_) { /* sidebar may render before toast service is ready */ }
}

// -----------------------------------------------------------------------
// Tab renderers
// -----------------------------------------------------------------------
async function _renderDiagnostics(body) {
    body.innerHTML = "";
    const toolbar = document.createElement("div");
    toolbar.className = "mec-diag-toolbar";
    const refresh = document.createElement("button");
    refresh.className = "mec-diag-btn primary";
    refresh.textContent = "Refresh";
    const clear = document.createElement("button");
    clear.className = "mec-diag-btn";
    clear.textContent = "Clear";
    const onlyErr = document.createElement("label");
    onlyErr.style.cssText = "display:flex;align-items:center;gap:4px;font-size:11px;";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    onlyErr.appendChild(cb);
    onlyErr.appendChild(document.createTextNode("Errors only"));
    toolbar.append(refresh, clear, onlyErr);
    body.appendChild(toolbar);

    const list = document.createElement("div");
    body.appendChild(list);

    async function load() {
        list.innerHTML = `<div class="mec-diag-empty">Loading…</div>`;
        const j = await _api(
            `/mec/diagnostics/recent?limit=200${cb.checked ? "&kind=error" : ""}`);
        list.innerHTML = "";
        if (!j.success) {
            list.innerHTML = `<div class="mec-diag-empty">Backend unavailable: ${j.message || j.error}</div>`;
            return;
        }
        const items = j.data || [];
        if (!items.length) {
            list.innerHTML = `<div class="mec-diag-empty">No events yet — run a workflow.</div>`;
            return;
        }
        for (const ev of items) {
            const card = document.createElement("div");
            const sev = (ev.type === "node_error" || ev.severity === "error") ? "error"
                      : (ev.severity === "warn" ? "warn" : "info");
            card.className = `mec-diag-card ${sev}`;
            const head = document.createElement("div");
            head.className = "mec-diag-row";
            const id = document.createElement("span");
            id.className = "mec-diag-id";
            id.textContent = ev.exc_type || ev.kind || ev.type || "event";
            const meta = document.createElement("span");
            meta.className = "mec-diag-meta";
            meta.textContent = `${_fmtTs(ev.ts)}${ev.node_id ? " · node " + ev.node_id : ""}`;
            head.append(id, meta);
            card.appendChild(head);
            if (ev.category || ev.pattern_id) {
                const tags = document.createElement("div");
                tags.style.marginTop = "3px";
                if (ev.category) {
                    const t = document.createElement("span");
                    t.className = `mec-diag-tag cat-${ev.category}`;
                    t.textContent = ev.category;
                    tags.appendChild(t);
                }
                if (ev.pattern_id) {
                    const t2 = document.createElement("span");
                    t2.className = "mec-diag-tag";
                    t2.textContent = ev.pattern_id;
                    tags.appendChild(t2);
                }
                card.appendChild(tags);
            }
            const msg = ev.exc_msg || ev.message || ev.hint;
            if (msg) {
                const m = document.createElement("div");
                m.className = "mec-diag-msg";
                m.textContent = msg;
                card.appendChild(m);
            }
            list.appendChild(card);
        }
    }
    refresh.onclick = load;
    cb.onchange = load;
    clear.onclick = async () => {
        const j = await _api("/mec/diagnostics/clear", { method: "POST" });
        if (j.success) { _toast("Cleared", "success"); load(); }
        else _toast("Clear failed: " + j.message, "error");
    };
    load();
}

async function _renderStatistics(body) {
    body.innerHTML = `<div class="mec-diag-empty">Loading…</div>`;
    const j = await _api("/mec/diagnostics/statistics");
    if (!j.success) {
        body.innerHTML = `<div class="mec-diag-empty">Backend unavailable: ${j.message || j.error}</div>`;
        return;
    }
    const d = j.data || { patterns: [], categories: [], total_events: 0 };
    body.innerHTML = "";
    const total = document.createElement("div");
    total.className = "mec-diag-card info";
    total.innerHTML = `<div class="mec-diag-row"><b>Total matched events</b><span>${d.total_events}</span></div>`;
    body.appendChild(total);

    if (d.categories.length) {
        const h = document.createElement("h4");
        h.textContent = "By category";
        h.style.cssText = "margin:8px 0 4px 0;";
        body.appendChild(h);
        const max = Math.max(...d.categories.map(c => c.count));
        for (const c of d.categories) {
            const card = document.createElement("div");
            card.className = "mec-diag-card";
            card.innerHTML = `<div class="mec-diag-row">
                <span><span class="mec-diag-tag cat-${c.category}">${c.category}</span></span>
                <span>${c.count}</span></div>
                <div class="mec-diag-bar"><div style="width:${(c.count / max * 100).toFixed(1)}%;"></div></div>`;
            body.appendChild(card);
        }
    }
    if (d.patterns.length) {
        const h2 = document.createElement("h4");
        h2.textContent = "Top patterns";
        h2.style.cssText = "margin:10px 0 4px 0;";
        body.appendChild(h2);
        for (const p of d.patterns.slice(0, 20)) {
            const card = document.createElement("div");
            card.className = "mec-diag-card";
            card.innerHTML = `<div class="mec-diag-row">
                <span class="mec-diag-id">${p.pattern_id}</span>
                <span>${p.count}×</span></div>
                <div class="mec-diag-meta">first ${_fmtTs(p.first_seen)} · last ${_fmtTs(p.last_seen)}</div>`;
            body.appendChild(card);
        }
    }
    if (!d.patterns.length && !d.categories.length) {
        body.appendChild(Object.assign(document.createElement("div"),
            { className: "mec-diag-empty", textContent: "No events recorded yet." }));
    }
}

async function _renderPatterns(body) {
    body.innerHTML = `<div class="mec-diag-empty">Loading…</div>`;
    const tb = document.createElement("div");
    tb.className = "mec-diag-toolbar";
    const reload = document.createElement("button");
    reload.className = "mec-diag-btn primary";
    reload.textContent = "Hot-reload patterns";
    tb.appendChild(reload);
    const j = await _api("/mec/diagnostics/patterns");
    body.innerHTML = "";
    body.appendChild(tb);
    if (!j.success) {
        body.appendChild(Object.assign(document.createElement("div"),
            { className: "mec-diag-empty", textContent: `Backend unavailable: ${j.message || j.error}` }));
        return;
    }
    const total = document.createElement("div");
    total.className = "mec-diag-card info";
    total.innerHTML = `<div class="mec-diag-row"><b>Loaded</b><span>${j.data.count} patterns</span></div>`;
    body.appendChild(total);
    for (const p of j.data.patterns) {
        const card = document.createElement("div");
        card.className = "mec-diag-card";
        card.innerHTML = `<div class="mec-diag-row">
            <span class="mec-diag-id">${p.id}</span>
            <span class="mec-diag-meta">prio ${p.priority} · conf ${p.confidence}</span></div>
            <div style="margin-top:3px;">
                <span class="mec-diag-tag cat-${p.category}">${p.category}</span>
                ${p.exc_types.length ? p.exc_types.map(t => `<span class="mec-diag-tag">${t}</span>`).join("") : ""}
            </div>
            <div class="mec-diag-meta" style="margin-top:3px;">${p.source}</div>`;
        body.appendChild(card);
    }
    reload.onclick = async () => {
        reload.disabled = true; reload.textContent = "Reloading…";
        const r = await _api("/mec/diagnostics/reload_patterns", { method: "POST" });
        reload.disabled = false; reload.textContent = "Hot-reload patterns";
        if (r.success) { _toast(`Reloaded ${r.data.count} patterns`, "success"); _renderPatterns(body); }
        else _toast("Reload failed: " + r.message, "error");
    };
}

async function _renderSettings(body) {
    body.innerHTML = `<div class="mec-diag-empty">Loading…</div>`;
    const j = await _api("/mec/diagnostics/settings");
    if (!j.success) {
        body.innerHTML = `<div class="mec-diag-empty">Backend unavailable: ${j.message || j.error}</div>`;
        return;
    }
    const s = j.data;
    body.innerHTML = "";
    const card = document.createElement("div");
    card.className = "mec-diag-card";
    card.innerHTML = `
        <h4 style="margin:0 0 8px 0;">Error Assistant</h4>
        <div class="mec-diag-kv">
            <span class="k">Tier mode</span>
            <select class="mec-diag-select" data-k="mode">
                <option value="auto">auto</option>
                <option value="deterministic_only">deterministic only (Tier 1)</option>
                <option value="local_only">local only (Tier 1+2)</option>
                <option value="cloud_only">cloud only (Tier 1+3)</option>
            </select>
            <span class="k">Cloud provider</span>
            <select class="mec-diag-select" data-k="cloud_provider">
                <option value="openai">openai</option>
                <option value="anthropic">anthropic</option>
                <option value="gemini">gemini</option>
                <option value="openrouter">openrouter</option>
            </select>
            <span class="k">Cloud model</span>
            <input class="mec-diag-input" data-k="cloud_model"/>
            <span class="k">Local model</span>
            <input class="mec-diag-input" data-k="local_model"/>
            <span class="k">Max tokens</span>
            <input class="mec-diag-input" data-k="max_tokens" type="number"/>
        </div>
        <div class="mec-diag-toolbar" style="padding-top:8px;">
            <button class="mec-diag-btn primary" data-action="save">Save</button>
            <button class="mec-diag-btn" data-action="reload">Reload patterns</button>
        </div>
    `;
    body.appendChild(card);
    for (const el of card.querySelectorAll("[data-k]")) {
        const k = el.dataset.k;
        if (s[k] !== undefined) el.value = s[k];
    }
    card.querySelector('[data-action="save"]').onclick = async () => {
        const payload = {};
        for (const el of card.querySelectorAll("[data-k]")) {
            let v = el.value;
            if (el.type === "number") v = Number(v);
            payload[el.dataset.k] = v;
        }
        const r = await _api("/mec/diagnostics/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (r.success) _toast("Saved", "success");
        else _toast("Save failed: " + r.message, "error");
    };
    card.querySelector('[data-action="reload"]').onclick = async () => {
        const r = await _api("/mec/diagnostics/reload_patterns", { method: "POST" });
        if (r.success) _toast(`Reloaded ${r.data.count} patterns`, "success");
        else _toast("Reload failed: " + r.message, "error");
    };
}

// Clipboard tab — purely client-side history of payloads our clipboard
// has copied during this tab's lifetime (mec_clipboard.js writes here).
window.__MEC_CLIPBOARD_HISTORY__ = window.__MEC_CLIPBOARD_HISTORY__ || [];

async function _renderClipboard(body) {
    body.innerHTML = "";
    const tb = document.createElement("div");
    tb.className = "mec-diag-toolbar";
    const refresh = document.createElement("button");
    refresh.className = "mec-diag-btn primary";
    refresh.textContent = "Refresh";
    const clear = document.createElement("button");
    clear.className = "mec-diag-btn";
    clear.textContent = "Clear history";
    tb.append(refresh, clear);
    body.appendChild(tb);
    const list = document.createElement("div");
    body.appendChild(list);

    function render() {
        list.innerHTML = "";
        const h = window.__MEC_CLIPBOARD_HISTORY__ || [];
        if (!h.length) {
            list.innerHTML = `<div class="mec-diag-empty">No copies yet. Press Ctrl+C on selected nodes.</div>`;
            return;
        }
        for (let i = h.length - 1; i >= 0; i--) {
            const entry = h[i];
            const card = document.createElement("div");
            card.className = "mec-diag-card info";
            const node_count = entry.payload?.nodes?.length || 0;
            const pack_count = entry.payload?.packs?.length || 0;
            card.innerHTML = `<div class="mec-diag-row">
                <span class="mec-diag-id">${node_count} node(s)</span>
                <span class="mec-diag-meta">${new Date(entry.ts).toLocaleTimeString()}</span></div>
                <div class="mec-diag-meta">${pack_count} pack(s) · v${entry.payload?.version || "?"}</div>`;
            const replay = document.createElement("button");
            replay.className = "mec-diag-btn";
            replay.style.marginTop = "4px";
            replay.textContent = "Copy this payload to clipboard";
            replay.onclick = async () => {
                try {
                    await navigator.clipboard.writeText(entry.text);
                    _toast("Re-copied", "success");
                } catch (e) { _toast("Clipboard write denied", "error"); }
            };
            card.appendChild(replay);
            list.appendChild(card);
        }
    }
    refresh.onclick = render;
    clear.onclick = () => { window.__MEC_CLIPBOARD_HISTORY__ = []; render(); };
    render();
}

// -----------------------------------------------------------------------
// Tab manager
// -----------------------------------------------------------------------
const TABS = [
    { id: "diag",  label: "Diagnostics", render: _renderDiagnostics },
    { id: "stats", label: "Statistics",  render: _renderStatistics },
    { id: "clip",  label: "Clipboard",   render: _renderClipboard  },
    { id: "set",   label: "Settings",    render: _renderSettings   },
    { id: "pat",   label: "Patterns",    render: _renderPatterns   },
];

function _mountSidebar(el) {
    _injectStyle();
    el.innerHTML = "";
    const root = document.createElement("div");
    root.className = "mec-diag-root";
    const tabBar = document.createElement("div");
    tabBar.className = "mec-diag-tabs";
    const body = document.createElement("div");
    body.className = "mec-diag-body";

    let active = TABS[0].id;
    const buttons = {};
    function setActive(id) {
        active = id;
        for (const t of TABS) buttons[t.id].classList.toggle("active", t.id === id);
        const def = TABS.find(t => t.id === id);
        if (def) def.render(body);
    }
    for (const t of TABS) {
        const b = document.createElement("button");
        b.className = "mec-diag-tab";
        b.textContent = t.label;
        b.onclick = () => setActive(t.id);
        tabBar.appendChild(b);
        buttons[t.id] = b;
    }

    root.appendChild(tabBar);
    root.appendChild(body);
    el.appendChild(root);
    setActive(active);
}

// -----------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------
app.registerExtension({
    name: "MEC.DiagnosticsSidebar",
    async setup() {
        try {
            app.extensionManager?.registerSidebarTab?.({
                id: "mec.diagnostics",
                icon: "pi pi-shield",
                title: "MEC Diagnostics",
                tooltip: "MaskEditControl diagnostics, statistics, clipboard history & error patterns",
                type: "custom",
                render(el) { _mountSidebar(el); },
            });
            console.log("[MEC.diagnostics] sidebar registered (id=mec.diagnostics)");
        } catch (e) {
            console.warn("[MEC.diagnostics] sidebar register failed:", e);
        }
    },
});
