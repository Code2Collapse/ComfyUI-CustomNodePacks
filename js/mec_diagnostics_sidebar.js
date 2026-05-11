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

// Curated suggestion list for the local-model input. Free-typing is still
// allowed (the backend resolves any GGUF stub or absolute path). Each entry
// shows approx VRAM/RAM footprint at Q4 quant + one-line "best for" hint.
const _LOCAL_MODEL_SUGGESTIONS = [
    { id: "qwen3-0.6b-instruct-q4_k_m",            ram: "~450 MB",  note: "tiny, fastest" },
    { id: "qwen3-1.7b-instruct-q4_k_m",            ram: "~1.1 GB",  note: "balanced default" },
    { id: "qwen3-4b-instruct-q4_k_m",              ram: "~2.5 GB",  note: "strongest small local" },
    { id: "qwen2.5-0.5b-instruct-q4_k_m",          ram: "~400 MB",  note: "legacy tiny" },
    { id: "qwen2.5-1.5b-instruct-q4_k_m",          ram: "~1.0 GB",  note: "legacy balanced" },
    { id: "qwen2.5-3b-instruct-q4_k_m",            ram: "~2.0 GB",  note: "legacy mid" },
    { id: "qwen2.5-coder-1.5b-instruct-q4_k_m",    ram: "~1.0 GB",  note: "code-aware" },
    { id: "phi-3.5-mini-instruct-q4_k_m",          ram: "~2.3 GB",  note: "MS, strong reasoning" },
    { id: "gemma-2-2b-it-q4_k_m",                  ram: "~1.6 GB",  note: "Google, multilingual" },
    { id: "llama-3.2-1b-instruct-q4_k_m",          ram: "~770 MB",  note: "Meta, fast" },
    { id: "llama-3.2-3b-instruct-q4_k_m",          ram: "~2.0 GB",  note: "Meta, good" },
    { id: "smollm2-1.7b-instruct-q4_k_m",          ram: "~1.1 GB",  note: "HF, fast" },
    { id: "tinyllama-1.1b-chat-q4_k_m",            ram: "~700 MB",  note: "tiniest, weak" },
    { id: "deepseek-r1-distill-qwen-1.5b-q4_k_m",  ram: "~1.0 GB",  note: "chain-of-thought" },
    { id: "internlm2.5-1.8b-chat-q4_k_m",          ram: "~1.2 GB",  note: "Shanghai AI Lab" },
];

// Curated Ollama tag suggestions. The actual list comes from /api/tags but
// these are shown as quick-pick chips when the daemon isn't reachable yet.
const _OLLAMA_MODEL_SUGGESTIONS = [
    "qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b",
    "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b",
    "llama3.2:1b", "llama3.2:3b", "llama3.1:8b",
    "phi3.5:3.8b", "gemma2:2b", "gemma2:9b",
    "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:8b",
    "mistral:7b", "codellama:7b",
];

// Rough cost estimate per 1 invocation (~512 in-tokens + 512 out) per model.
// Numbers are illustrative only — pulled from public pricing as of 2026-05.
const _CLOUD_COST_HINT = {
    "openai/gpt-4o-mini":            "~$0.0002 / error",
    "openai/gpt-4o":                 "~$0.005 / error",
    "openai/gpt-4.1-mini":           "~$0.0003 / error",
    "anthropic/claude-3-5-haiku":    "~$0.0008 / error",
    "anthropic/claude-3-5-sonnet":   "~$0.0095 / error",
    "anthropic/claude-3-7-sonnet":   "~$0.012 / error",
    "gemini/gemini-1.5-flash":       "~$0.0001 / error",
    "gemini/gemini-1.5-pro":         "~$0.0035 / error",
    "gemini/gemini-2.0-flash":       "~$0.0002 / error",
    "openrouter/auto":               "varies by route",
    "groq/llama-3.3-70b-versatile":  "FREE tier available",
    "groq/llama-3.1-8b-instant":     "FREE tier available",
    "groq/mixtral-8x7b-32768":       "FREE tier available",
    "deepseek/deepseek-chat":        "~$0.00007 / error",
    "deepseek/deepseek-reasoner":    "~$0.0006 / error",
};

// Curated Tier-3 model suggestions per provider (filled into the model
// <input>'s datalist when the user picks a provider).
const _CLOUD_MODEL_SUGGESTIONS = {
    openai:     ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    anthropic:  ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"],
    gemini:     ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-2.0-flash-exp"],
    openrouter: ["openai/gpt-4o-mini", "anthropic/claude-3-5-haiku", "google/gemini-flash-1.5", "auto"],
    groq:       ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    deepseek:   ["deepseek-chat", "deepseek-reasoner"],
};

function _statusPill(state /* 'ready' | 'warn' | 'error' | 'idle' */, text) {
    const colors = {
        ready: { bg: "#16331f", fg: "#7be089", dot: "#3ecf5a" },
        warn:  { bg: "#3a2e15", fg: "#f0c764", dot: "#e3a93b" },
        error: { bg: "#3a1818", fg: "#ff8b8b", dot: "#e25151" },
        idle:  { bg: "#252525", fg: "#999",    dot: "#666"   },
    }[state] || { bg: "#252525", fg: "#999", dot: "#666" };
    return `<span style="display:inline-flex;align-items:center;gap:6px;background:${colors.bg};color:${colors.fg};padding:2px 8px;border-radius:10px;font-size:11px;font-weight:500;">
        <span style="width:7px;height:7px;border-radius:50%;background:${colors.dot};"></span>${text}
    </span>`;
}

function _tierCardHTML(opts) {
    const { tierNum, title, subtitle, color, bodyHTML } = opts;
    return `
    <div class="mec-diag-card" data-tier="${tierNum}" style="border-left:3px solid ${color};margin-bottom:10px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <label style="display:flex;align-items:center;gap:8px;cursor:pointer;flex:1;">
                <input type="checkbox" data-k="tier${tierNum}_enabled" style="margin:0;"/>
                <span style="font-weight:600;color:${color};font-size:13px;">Tier ${tierNum}</span>
                <span style="opacity:0.85;">— ${title}</span>
            </label>
            <span class="mec-diag-tier-status" data-tier-status="${tierNum}">${_statusPill("idle", "checking…")}</span>
            <button class="mec-diag-btn" data-action="test-tier" data-tier="${tierNum}" style="font-size:11px;padding:3px 10px;">Test</button>
        </div>
        <div class="mec-diag-meta" style="font-size:11px;opacity:0.7;margin-bottom:8px;padding-left:24px;">${subtitle}</div>
        <div style="padding-left:24px;">${bodyHTML}</div>
    </div>`;
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

    const wrap = document.createElement("div");
    wrap.innerHTML = `
        <div style="font-size:13px;font-weight:600;margin-bottom:4px;">Error Assistant</div>
        <div class="mec-diag-meta" style="font-size:11px;opacity:0.7;margin-bottom:12px;">
            Pick which tiers run when an error fires. Multiple tiers can be enabled — Tier 1 always runs first to provide context, then the highest-priority enabled LLM tier (Tier 3 if available, else Tier 2).
        </div>

        ${_tierCardHTML({
            tierNum: 1,
            title: "Patterns (offline, instant)",
            subtitle: "Regex/heuristic matcher. Zero VRAM, &lt;1 ms. Always recommended ON — feeds context to the LLM tiers.",
            color: "#3ecf5a",
            bodyHTML: `
                <div class="mec-diag-toolbar" style="margin:0;padding:0;gap:6px;">
                    <button class="mec-diag-btn" data-action="reload" style="font-size:11px;padding:3px 10px;">Reload patterns</button>
                    <button class="mec-diag-btn" data-action="toggle-custom" style="font-size:11px;padding:3px 10px;">Custom patterns ▾</button>
                </div>
                <div data-custom-patterns style="display:none;margin-top:8px;border-top:1px dashed #333;padding-top:8px;">
                    <div style="font-size:11px;font-weight:600;margin-bottom:4px;">Add your own pattern</div>
                    <div class="mec-diag-kv" style="grid-template-columns:90px 1fr;">
                        <span class="k">ID</span>
                        <input class="mec-diag-input" data-cp="id" placeholder="my_custom_oom" style="font-family:monospace;"/>
                        <span class="k">Regex</span>
                        <input class="mec-diag-input" data-cp="regex" placeholder="(?i)my custom error message" style="font-family:monospace;"/>
                        <span class="k">Category</span>
                        <select class="mec-diag-select" data-cp="category">
                            <option value="user">user</option>
                            <option value="vram">vram</option>
                            <option value="shape">shape</option>
                            <option value="missing">missing</option>
                            <option value="network">network</option>
                            <option value="config">config</option>
                            <option value="uncategorized">uncategorized</option>
                        </select>
                        <span class="k">Cause</span>
                        <input class="mec-diag-input" data-cp="cause" placeholder="One-sentence explanation"/>
                        <span class="k">Fixes</span>
                        <textarea class="mec-diag-input" data-cp="fixes" rows="3" placeholder="One fix per line"></textarea>
                    </div>
                    <div class="mec-diag-toolbar" style="margin:6px 0 0 0;padding:0;justify-content:flex-end;">
                        <button class="mec-diag-btn primary" data-action="add-pattern" style="font-size:11px;padding:3px 10px;">Add pattern</button>
                    </div>
                    <div data-custom-list style="margin-top:10px;font-size:11px;"></div>
                </div>`,
        })}

        ${_tierCardHTML({
            tierNum: 2,
            title: "Local LLM (Ollama or llama.cpp)",
            subtitle: "Runs on your machine. Private, free, no API key. Ollama is recommended — easier setup, more models.",
            color: "#5b9bd5",
            bodyHTML: `
                <div class="mec-diag-kv">
                    <span class="k">Backend</span>
                    <select class="mec-diag-select" data-k="tier2_backend">
                        <option value="ollama">ollama (recommended)</option>
                        <option value="llamacpp">llama.cpp + GGUF</option>
                    </select>
                </div>
                <div data-tier2-ollama style="margin-top:6px;">
                    <div class="mec-diag-kv">
                        <span class="k">Server URL</span>
                        <input class="mec-diag-input" data-k="ollama_url" placeholder="http://localhost:11434"/>
                        <span class="k">Model</span>
                        <input class="mec-diag-input" data-k="ollama_model" list="mec-ollama-model-list" placeholder="qwen3:4b"/>
                    </div>
                    <datalist id="mec-ollama-model-list">
                        ${_OLLAMA_MODEL_SUGGESTIONS.map(m => `<option value="${m}"></option>`).join("")}
                    </datalist>
                    <div class="mec-diag-toolbar" style="margin:6px 0 0 0;padding:0;gap:6px;">
                        <button class="mec-diag-btn" data-action="ollama-refresh" type="button" style="font-size:11px;padding:3px 10px;">Refresh installed</button>
                        <span data-ollama-info style="font-size:10.5px;opacity:0.7;align-self:center;">—</span>
                    </div>
                    <div data-ollama-chips style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px;"></div>
                    <div class="mec-diag-meta" style="font-size:10.5px;opacity:0.6;margin-top:6px;">
                        Install Ollama: <a href="https://ollama.com/download" target="_blank" style="color:#7bb6f4;">ollama.com/download</a> · then run e.g. <code>ollama pull qwen3:4b</code>
                    </div>
                </div>
                <div data-tier2-llamacpp style="margin-top:6px;display:none;">
                    <div class="mec-diag-kv">
                        <span class="k">GGUF stub</span>
                        <input class="mec-diag-input" data-k="local_model" list="mec-local-model-list" placeholder="qwen3-1.7b-instruct-q4_k_m"/>
                    </div>
                    <datalist id="mec-local-model-list">
                        ${_LOCAL_MODEL_SUGGESTIONS.map(m => `<option value="${m.id}"></option>`).join("")}
                    </datalist>
                    <details style="margin-top:6px;">
                        <summary style="cursor:pointer;font-size:11px;opacity:0.8;">Curated GGUF catalog (${_LOCAL_MODEL_SUGGESTIONS.length} models)</summary>
                        <table style="width:100%;font-size:10.5px;border-collapse:collapse;margin-top:4px;">
                            <thead><tr style="opacity:0.7;text-align:left;"><th style="padding:2px 4px;">Model</th><th style="padding:2px 4px;">VRAM</th><th style="padding:2px 4px;">Note</th></tr></thead>
                            <tbody>
                            ${_LOCAL_MODEL_SUGGESTIONS.map(m => `
                                <tr data-pick="${m.id}" style="cursor:pointer;border-top:1px solid #2a2a2a;">
                                    <td style="padding:2px 4px;font-family:monospace;">${m.id}</td>
                                    <td style="padding:2px 4px;opacity:0.8;">${m.ram}</td>
                                    <td style="padding:2px 4px;opacity:0.7;">${m.note}</td>
                                </tr>`).join("")}
                            </tbody>
                        </table>
                    </details>
                    <div class="mec-diag-meta" style="font-size:10.5px;opacity:0.6;margin-top:6px;">
                        Place GGUF in <code>ComfyUI/models/llm/</code> · install: <code>pip install llama-cpp-python</code>
                    </div>
                </div>`,
        })}

        ${_tierCardHTML({
            tierNum: 3,
            title: "Cloud LLM (OpenAI / Anthropic / Gemini / OpenRouter / Groq / DeepSeek)",
            subtitle: "Fastest and strongest. Groq has a free tier; OpenRouter and DeepSeek are cheapest paid. Keys are stored encrypted.",
            color: "#d58a3e",
            bodyHTML: `
                <div class="mec-diag-kv">
                    <span class="k">Provider</span>
                    <select class="mec-diag-select" data-k="cloud_provider">
                        <option value="openai">openai</option>
                        <option value="anthropic">anthropic</option>
                        <option value="gemini">gemini</option>
                        <option value="openrouter">openrouter</option>
                        <option value="groq">groq (free tier)</option>
                        <option value="deepseek">deepseek (cheap)</option>
                    </select>
                    <span class="k">Model</span>
                    <input class="mec-diag-input" data-k="cloud_model" list="mec-cloud-model-list" placeholder="gpt-4o-mini"/>
                    <span class="k">API key</span>
                    <span style="display:flex;gap:4px;align-items:center;">
                        <input class="mec-diag-input" data-k-secret="api_key" type="password"
                               style="flex:1;font-family:monospace;" placeholder="(not set)" autocomplete="off"/>
                        <button class="mec-diag-btn" data-action="show-key" type="button" title="Show/hide" style="font-size:11px;padding:3px 8px;">👁</button>
                        <button class="mec-diag-btn" data-action="save-key" type="button" style="font-size:11px;padding:3px 10px;">Save key</button>
                    </span>
                </div>
                <datalist id="mec-cloud-model-list"></datalist>
                <div class="mec-diag-meta" data-cost-hint style="font-size:10.5px;opacity:0.6;margin-top:4px;">
                    Cost: —
                </div>`,
        })}

        <div class="mec-diag-card" style="margin-top:4px;">
            <div style="font-weight:600;font-size:12px;margin-bottom:6px;">Shared</div>
            <div class="mec-diag-kv">
                <span class="k">Max tokens</span>
                <input class="mec-diag-input" data-k="max_tokens" type="number" min="64" max="4096" step="64"/>
            </div>
            <div class="mec-diag-meta" style="font-size:10.5px;opacity:0.6;margin-top:4px;">
                Output cap for Tier 2 / Tier 3. 512 is a good default.
            </div>
        </div>

        <div class="mec-diag-toolbar" style="padding-top:10px;justify-content:flex-end;">
            <button class="mec-diag-btn primary" data-action="save">Save settings</button>
        </div>
    `;
    body.appendChild(wrap);

    // Hydrate non-secret fields.
    for (const el of wrap.querySelectorAll("[data-k]")) {
        const k = el.dataset.k;
        if (s[k] === undefined) continue;
        if (el.type === "checkbox") el.checked = !!s[k];
        else el.value = s[k];
    }

    // Update cost-hint helper.
    const costHint = wrap.querySelector("[data-cost-hint]");
    const updateCostHint = () => {
        const prov = wrap.querySelector('[data-k="cloud_provider"]').value;
        const model = (wrap.querySelector('[data-k="cloud_model"]').value || "").trim();
        const k = `${prov}/${model}`;
        const direct = _CLOUD_COST_HINT[k];
        const fallback = Object.entries(_CLOUD_COST_HINT).find(([key]) => key.startsWith(prov + "/"));
        costHint.textContent = "Cost: " + (direct || (fallback ? fallback[1] + " (closest match)" : "varies"));
    };
    wrap.querySelector('[data-k="cloud_provider"]').addEventListener("change", updateCostHint);
    wrap.querySelector('[data-k="cloud_model"]').addEventListener("input", updateCostHint);
    updateCostHint();

    // ----- Cloud model datalist updates with provider -----
    const cloudModelDL = wrap.querySelector("#mec-cloud-model-list");
    const cloudModelInput = wrap.querySelector('[data-k="cloud_model"]');
    const refreshCloudDatalist = () => {
        const prov = wrap.querySelector('[data-k="cloud_provider"]').value;
        const opts = _CLOUD_MODEL_SUGGESTIONS[prov] || [];
        cloudModelDL.innerHTML = opts.map(m => `<option value="${m}"></option>`).join("");
    };
    wrap.querySelector('[data-k="cloud_provider"]').addEventListener("change", refreshCloudDatalist);
    refreshCloudDatalist();

    // ----- Tier 2 backend toggle (ollama vs llama.cpp) -----
    const backendSel = wrap.querySelector('[data-k="tier2_backend"]');
    const ollamaPane = wrap.querySelector("[data-tier2-ollama]");
    const llamaPane  = wrap.querySelector("[data-tier2-llamacpp]");
    const syncTier2Pane = () => {
        const isOl = backendSel.value === "ollama";
        ollamaPane.style.display = isOl ? "" : "none";
        llamaPane.style.display  = isOl ? "none" : "";
    };
    backendSel.addEventListener("change", syncTier2Pane);
    syncTier2Pane();

    // ----- Tier 2 / llama.cpp: clickable GGUF rows -----
    for (const tr of llamaPane.querySelectorAll("tr[data-pick]")) {
        tr.addEventListener("click", () => {
            llamaPane.querySelector('[data-k="local_model"]').value = tr.dataset.pick;
        });
    }

    // ----- Tier 2 / Ollama: refresh installed models -----
    const ollamaInfo = ollamaPane.querySelector("[data-ollama-info]");
    const ollamaChips = ollamaPane.querySelector("[data-ollama-chips]");
    const ollamaModelInput = ollamaPane.querySelector('[data-k="ollama_model"]');
    const refreshOllama = async () => {
        const url = ollamaPane.querySelector('[data-k="ollama_url"]').value || "http://localhost:11434";
        ollamaInfo.textContent = "checking…";
        ollamaChips.innerHTML = "";
        const r = await _api(`/mec/diagnostics/ollama/tags?url=${encodeURIComponent(url)}`);
        if (!r.success) { ollamaInfo.textContent = "error: " + (r.message || r.error); return; }
        if (!r.data.available) { ollamaInfo.textContent = "daemon unreachable"; return; }
        const models = r.data.models || [];
        ollamaInfo.textContent = models.length ? `${models.length} installed` : "no models pulled";
        for (const m of models) {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "mec-diag-btn";
            chip.style.cssText = "font-size:10.5px;padding:2px 8px;background:#2a3a4a;";
            chip.textContent = m;
            chip.onclick = () => { ollamaModelInput.value = m; };
            ollamaChips.appendChild(chip);
        }
    };
    ollamaPane.querySelector('[data-action="ollama-refresh"]').onclick = refreshOllama;
    // Auto-probe once on first render (cheap).
    setTimeout(() => { if (backendSel.value === "ollama") refreshOllama(); }, 50);

    // ----- Tier 1: custom-pattern toggle + list + add/remove -----
    const customWrap = wrap.querySelector("[data-custom-patterns]");
    const customList = wrap.querySelector("[data-custom-list]");
    wrap.querySelector('[data-action="toggle-custom"]').onclick = async () => {
        const showing = customWrap.style.display !== "none";
        customWrap.style.display = showing ? "none" : "";
        if (!showing) await refreshCustomList();
    };
    const refreshCustomList = async () => {
        customList.innerHTML = "loading…";
        const r = await _api("/mec/diagnostics/patterns/custom");
        if (!r.success) { customList.textContent = "error: " + r.message; return; }
        const items = (r.data && r.data.patterns) || [];
        if (!items.length) { customList.innerHTML = `<em style="opacity:0.6;">No custom patterns yet.</em>`; return; }
        customList.innerHTML = `<div style="font-weight:600;margin-bottom:4px;">Current custom patterns (${items.length})</div>` +
            items.map(p => `
                <div style="display:flex;gap:6px;align-items:center;padding:3px 0;border-top:1px dashed #2a2a2a;">
                    <code style="flex:0 0 auto;">${p.id}</code>
                    <span style="flex:1;opacity:0.75;font-size:10px;font-family:monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${p.regex}</span>
                    <span style="opacity:0.6;font-size:10px;">${p.category}</span>
                    <button class="mec-diag-btn" data-rm="${p.id}" style="font-size:10.5px;padding:1px 6px;">✕</button>
                </div>`).join("");
        for (const btn of customList.querySelectorAll("[data-rm]")) {
            btn.onclick = async () => {
                const id = btn.dataset.rm;
                if (!confirm(`Remove custom pattern "${id}"?`)) return;
                const r2 = await _api(`/mec/diagnostics/patterns/custom?id=${encodeURIComponent(id)}`, { method: "DELETE" });
                if (r2.success) { _toast(`Removed "${id}"`, "success"); await refreshCustomList(); }
                else _toast("Remove failed: " + r2.message, "error");
            };
        }
    };
    wrap.querySelector('[data-action="add-pattern"]').onclick = async () => {
        const get = (name) => (wrap.querySelector(`[data-cp="${name}"]`).value || "").trim();
        const payload = {
            id: get("id"),
            regex: get("regex"),
            category: get("category") || "user",
            cause: get("cause"),
            fixes: get("fixes").split("\n").map(s => s.trim()).filter(Boolean),
        };
        if (!payload.id || !payload.regex) { _toast("ID and regex are required", "warn"); return; }
        const r = await _api("/mec/diagnostics/patterns/custom", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (r.success) {
            _toast(`Added "${payload.id}". Total patterns: ${r.data.total_patterns}`, "success");
            for (const cp of wrap.querySelectorAll("[data-cp]")) cp.value = cp.tagName === "SELECT" ? cp.options[0].value : "";
            await refreshCustomList();
            await refreshStatus();
        } else { _toast("Add failed: " + r.message, "error"); }
    };

    // Status pill loader.
    const setStatus = (n, html) => {
        const el = wrap.querySelector(`[data-tier-status="${n}"]`);
        if (el) el.innerHTML = html;
    };
    const refreshStatus = async () => {
        for (const n of [1, 2, 3]) setStatus(n, _statusPill("idle", "checking…"));
        const r = await _api("/mec/diagnostics/error_assistant/status");
        if (!r.success) {
            for (const n of [1, 2, 3]) setStatus(n, _statusPill("error", "backend down"));
            return;
        }
        for (const n of [1, 2, 3]) {
            const t = r.data[`tier${n}`] || {};
            const state = t.ready ? "ready" : (n === 1 ? "error" : "warn");
            const txt = t.ready ? "ready" : (t.detail || "not ready");
            setStatus(n, _statusPill(state, txt));
        }
    };
    refreshStatus();

    // API-key field: load masked preview, show/hide, save.
    const keyInput = wrap.querySelector('[data-k-secret="api_key"]');
    let _keyVisible = false;
    let _keyDirty = false;
    keyInput.addEventListener("input", () => { _keyDirty = true; });
    const loadKeyPreview = async () => {
        const prov = wrap.querySelector('[data-k="cloud_provider"]').value;
        const r = await _api(`/mec/diagnostics/error_assistant/secrets?provider=${encodeURIComponent(prov)}`);
        if (r.success) {
            keyInput.value = r.data.set ? r.data.preview : "";
            keyInput.placeholder = r.data.set ? "(stored — type to replace)" : "(not set)";
            _keyDirty = false;
        }
    };
    wrap.querySelector('[data-k="cloud_provider"]').addEventListener("change", loadKeyPreview);
    loadKeyPreview();
    wrap.querySelector('[data-action="show-key"]').onclick = () => {
        _keyVisible = !_keyVisible;
        keyInput.type = _keyVisible ? "text" : "password";
    };
    wrap.querySelector('[data-action="save-key"]').onclick = async () => {
        if (!_keyDirty) {
            _toast("Type a key first (current value is the masked preview)", "warn");
            return;
        }
        const prov = wrap.querySelector('[data-k="cloud_provider"]').value;
        const r = await _api("/mec/diagnostics/error_assistant/secrets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ provider: prov, api_key: keyInput.value }),
        });
        if (r.success) {
            _toast(r.data.set ? `API key saved for ${prov}` : `API key cleared for ${prov}`, "success");
            await loadKeyPreview();
            await refreshStatus();
        } else {
            _toast("Save failed: " + r.message, "error");
        }
    };

    // Per-tier Test buttons.
    for (const btn of wrap.querySelectorAll('[data-action="test-tier"]')) {
        btn.onclick = async () => {
            const n = Number(btn.dataset.tier);
            btn.disabled = true; const orig = btn.textContent; btn.textContent = "Testing…";
            const r = await _api("/mec/diagnostics/error_assistant/test_tier", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ tier: n }),
            });
            btn.disabled = false; btn.textContent = orig;
            if (!r.success) { _toast(`Tier ${n} test failed: ${r.message}`, "error"); return; }
            const ok = r.data.ok;
            _toast(
                `Tier ${n} ${ok ? "OK" : "fell back to T" + r.data.tier_returned} — ${r.data.elapsed_ms}ms`,
                ok ? "success" : "warn",
            );
        };
    }

    // Save settings (non-secret fields).
    wrap.querySelector('[data-action="save"]').onclick = async () => {
        const payload = {};
        for (const el of wrap.querySelectorAll("[data-k]")) {
            let v;
            if (el.type === "checkbox") v = el.checked;
            else if (el.type === "number") v = Number(el.value);
            else v = el.value;
            payload[el.dataset.k] = v;
        }
        const r = await _api("/mec/diagnostics/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (r.success) { _toast("Saved", "success"); await refreshStatus(); }
        else _toast("Save failed: " + r.message, "error");
    };

    // Reload patterns (Tier 1).
    wrap.querySelector('[data-action="reload"]').onclick = async () => {
        const r = await _api("/mec/diagnostics/reload_patterns", { method: "POST" });
        if (r.success) { _toast(`Reloaded ${r.data.count} patterns`, "success"); await refreshStatus(); }
        else _toast("Reload failed: " + r.message, "error");
    };
}

// Clipboard tab — purely client-side history of payloads our clipboard
// has copied during this tab's lifetime (mec_clipboard.js writes here).
window.__MEC_CLIPBOARD_HISTORY__ = window.__MEC_CLIPBOARD_HISTORY__ || [];

async function _renderClipboard(body) {
    body.innerHTML = "";

    // Auto-copy toggle (mirrors the ComfyUI setting + localStorage flag
    // owned by mec_clipboard.js). Same key, same event \u2014 either control
    // updates the other.
    const AUTOCOPY_KEY = "mec.clipboard.autoCopy";
    const isOn = () => {
        try {
            const v = localStorage.getItem(AUTOCOPY_KEY);
            return v === null ? true : v === "1" || v === "true";
        } catch (_) { return true; }
    };
    const setOn = (on) => {
        try { localStorage.setItem(AUTOCOPY_KEY, on ? "1" : "0"); } catch (_) {}
        window.dispatchEvent(new CustomEvent("mec-clipboard-autocopy-changed", { detail: { enabled: !!on } }));
        try { app.ui?.settings?.setSettingValue?.("MEC.Clipboard.AutoCopy", !!on); } catch (_) {}
    };
    const toggleRow = document.createElement("div");
    toggleRow.className = "mec-diag-card info";
    toggleRow.style.cssText = "margin-bottom:8px;display:flex;align-items:center;gap:8px;";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = "mec-autocopy-toggle";
    cb.checked = isOn();
    cb.onchange = () => setOn(cb.checked);
    const lbl = document.createElement("label");
    lbl.htmlFor = "mec-autocopy-toggle";
    lbl.textContent = "Auto-copy on Ctrl+C";
    lbl.style.cssText = "cursor:pointer;flex:1;";
    const hint = document.createElement("div");
    hint.className = "mec-diag-meta";
    hint.style.cssText = "flex-basis:100%;font-size:11px;opacity:0.75;";
    hint.textContent = "When ON, plain Ctrl+C also writes the portable JSON payload to your OS clipboard. When OFF, only LiteGraph's native in-tab clipboard runs \u2014 use the buttons below to re-copy any past payload.";
    toggleRow.append(cb, lbl, hint);
    body.appendChild(toggleRow);
    // Sync with the ComfyUI settings panel.
    window.addEventListener("mec-clipboard-autocopy-changed", (e) => {
        cb.checked = !!(e.detail?.enabled);
    });

    const tb = document.createElement("div");
    tb.className = "mec-diag-toolbar";
    const refresh = document.createElement("button");
    refresh.className = "mec-diag-btn primary";
    refresh.textContent = "Refresh";
    const pasteOS = document.createElement("button");
    pasteOS.className = "mec-diag-btn";
    pasteOS.textContent = "Paste node from clipboard";
    pasteOS.title = "Read the current OS clipboard and, if it contains a MEC payload, paste those nodes into the graph (same as Ctrl+Alt+V).";
    pasteOS.onclick = async () => {
        const api = window.__MEC_CLIPBOARD_API__;
        if (!api?.pasteFromOSClipboard) {
            _toast("MEC clipboard module not loaded", "error");
            return;
        }
        try { await api.pasteFromOSClipboard(); }
        catch (e) { _toast("Paste failed: " + e, "error"); }
    };
    const clear = document.createElement("button");
    clear.className = "mec-diag-btn";
    clear.textContent = "Clear history";
    tb.append(refresh, pasteOS, clear);
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
            const pasteBtn = document.createElement("button");
            pasteBtn.className = "mec-diag-btn primary";
            pasteBtn.style.cssText = "margin-top:4px;margin-left:6px;";
            pasteBtn.textContent = "Paste into graph";
            pasteBtn.title = "Drop this payload's nodes into the current workflow without touching the OS clipboard.";
            pasteBtn.onclick = async () => {
                const api = window.__MEC_CLIPBOARD_API__;
                if (!api?.pasteFromText) {
                    _toast("MEC clipboard module not loaded", "error");
                    return;
                }
                try { await api.pasteFromText(entry.text); }
                catch (e) { _toast("Paste failed: " + e, "error"); }
            };
            card.appendChild(replay);
            card.appendChild(pasteBtn);
            list.appendChild(card);
        }
    }
    refresh.onclick = render;
    clear.onclick = () => { window.__MEC_CLIPBOARD_HISTORY__ = []; render(); };
    render();
}

// -----------------------------------------------------------------------
// Integrity tab — mirrors the topbar button's modal but inline.
// Reads live state from window.__MEC_INTEGRITY__ (set by
// js/nukenodemax/integrity_badges.js). Survives if that script
// hasn't loaded yet by falling back to a direct /nukenodemax/
// integrity_report fetch.
// -----------------------------------------------------------------------
function _extractPkg(msg) {
    let m = msg.match(/^The package [`"]([A-Za-z0-9_.\-]+)[`"]/);
    if (m) return m[1];
    m = msg.match(/^([A-Za-z0-9_.\-]+)\s+\S+\s+(?:has requirement|requires|depends)/i);
    if (m) return m[1];
    return null;
}

async function _renderIntegrity(body) {
    body.innerHTML = "";

    const toolbar = document.createElement("div");
    toolbar.className = "mec-diag-toolbar";
    const refresh = document.createElement("button");
    refresh.className = "mec-diag-btn primary";
    refresh.textContent = "Refresh";
    const openDlg = document.createElement("button");
    openDlg.className = "mec-diag-btn";
    openDlg.textContent = "Open full dialog";
    openDlg.onclick = () => window.__MEC_INTEGRITY__?.open?.();
    const muteLabel = document.createElement("label");
    muteLabel.style.cssText = "display:flex;align-items:center;gap:4px;font-size:11px;margin-left:auto;";
    const muteCb = document.createElement("input");
    muteCb.type = "checkbox";
    muteCb.checked = !!window.__MEC_INTEGRITY__?.isMuted?.();
    muteCb.onchange = () => {
        window.__MEC_INTEGRITY__?.setMuted?.(muteCb.checked);
        _toast(muteCb.checked ? "Integrity warnings muted" : "Integrity warnings un-muted");
    };
    muteLabel.append(muteCb, document.createTextNode("Mute warnings"));
    toolbar.append(refresh, openDlg, muteLabel);
    body.appendChild(toolbar);

    const meta = document.createElement("div");
    meta.className = "mec-diag-meta";
    meta.style.cssText = "padding:0 2px 6px;";
    body.appendChild(meta);

    const list = document.createElement("div");
    body.appendChild(list);

    function render(state) {
        const events = state?.events || [];
        meta.textContent =
            `pip_check: ${state?.pipOk ? "ok" : "conflicts"} · ` +
            `checksum drift: ${state?.drift ?? 0} · ` +
            `backend: ${state?.usedUv ? "uv" : "pip"}` +
            (state?.fromCache ? " (cached)" : " (fresh)") +
            (state?.lastUpdated ? ` · ${new Date(state.lastUpdated).toLocaleTimeString()}` : "");

        list.innerHTML = "";
        if (!events.length) {
            list.innerHTML = `<div class="mec-diag-empty">No integrity events. Environment looks clean.</div>`;
            return;
        }

        // Group by kind for readability.
        const groups = {};
        for (const e of events) (groups[e.kind] = groups[e.kind] || []).push(e);

        for (const [kind, items] of Object.entries(groups)) {
            const h = document.createElement("div");
            h.style.cssText = "font-weight:600;opacity:0.8;margin:8px 0 4px;font-size:11px;text-transform:uppercase;letter-spacing:0.4px;";
            h.textContent = `${kind.replace(/_/g, " ")} (${items.length})`;
            list.appendChild(h);

            for (const e of items) {
                const sev = (e.severity || "warn").toLowerCase();
                const card = document.createElement("div");
                card.className = `mec-diag-card ${sev === "error" ? "error" : "warn"}`;
                const row = document.createElement("div");
                row.className = "mec-diag-row";
                const id = document.createElement("span");
                id.className = "mec-diag-id";
                id.textContent = sev;
                const metaR = document.createElement("span");
                metaR.className = "mec-diag-meta";
                metaR.textContent = e.file || "";
                row.append(id, metaR);
                card.appendChild(row);
                const msg = document.createElement("div");
                msg.className = "mec-diag-msg";
                msg.textContent = e.message;
                card.appendChild(msg);
                const pkg = e.kind === "dependency_conflict" ? _extractPkg(e.message) : null;
                if (pkg) {
                    const act = document.createElement("button");
                    act.className = "mec-diag-btn primary";
                    act.style.cssText = "margin-top:6px;";
                    act.textContent = `Reinstall ${pkg}`;
                    act.onclick = () => window.__MEC_INTEGRITY__?.reinstall?.(pkg);
                    card.appendChild(act);
                }
                list.appendChild(card);
            }
        }
    }

    // Subscribe to live state if available; otherwise fetch once.
    let unsub = null;
    if (window.__MEC_INTEGRITY__) {
        unsub = window.__MEC_INTEGRITY__.subscribe(render);
    } else {
        const j = await _api("/nukenodemax/integrity_report");
        render({
            events: j.events || [],
            pipOk: j.pip_check ? !!j.pip_check.ok : true,
            drift: (j.checksum_drift || []).length,
            usedUv: !!j.used_uv,
            fromCache: !!j.from_cache,
            lastUpdated: Date.now(),
        });
    }

    refresh.onclick = async () => {
        if (window.__MEC_INTEGRITY__) {
            await window.__MEC_INTEGRITY__.refresh();
            _toast("Integrity report refreshed");
        } else {
            const j = await _api("/nukenodemax/integrity_report");
            render({
                events: j.events || [],
                pipOk: j.pip_check ? !!j.pip_check.ok : true,
                drift: (j.checksum_drift || []).length,
                usedUv: !!j.used_uv,
                fromCache: !!j.from_cache,
                lastUpdated: Date.now(),
            });
        }
    };

    // Tear down the subscription when this body is re-rendered (tab swap).
    const teardownObs = new MutationObserver(() => {
        if (!document.body.contains(list)) {
            if (unsub) { try { unsub(); } catch {} }
            teardownObs.disconnect();
        }
    });
    teardownObs.observe(body.parentNode || document.body, { childList: true });
}

// -----------------------------------------------------------------------
// Tab manager
// -----------------------------------------------------------------------
const TABS = [
    { id: "diag",  label: "Diagnostics", render: _renderDiagnostics },
    { id: "stats", label: "Statistics",  render: _renderStatistics },
    { id: "clip",  label: "Clipboard",   render: _renderClipboard  },
    { id: "intg",  label: "Integrity",   render: _renderIntegrity  },
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
