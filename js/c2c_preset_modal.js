/**
 * C2C Preset Hub — full-screen modal (P0.5, locked spec 2026-05-25).
 *
 * A live preset/prompt aggregator across 9 sources with a filter sidebar,
 * results grid, side-preview, a native Promptomania builder tab, and a local
 * Image to Prompt (CLIP-Interrogator) tab. Talks only to the backend at
 * /c2c/presets/* (no direct upstream calls from the browser).
 *
 * Public API (consumed by the 6 wizard wire-ups + OmniBar):
 *   window.__C2C_PRESET_HUB__.open({ tab, q, filters })
 *
 * Theme: uses var(--c2c-*) tokens only (no frozen palette) so it flips across
 * Mocha / Latte / OLED.
 */

import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { forAllNodes } from "./_subgraph_walk.js";

const MODAL_ID = "c2c-preset-hub-modal";

// Tabs map 1:1 to backend sources + the two local tabs.
const TABS = [
    { key: "lexica", label: "Lexica" },
    { key: "civitai", label: "Civitai" },
    { key: "huggingface", label: "HF" },
    { key: "openart", label: "OpenArt" },
    { key: "promptdexter", label: "Pdexter" },
    { key: "c2c_doctor", label: "GitHub Issues" },
    { key: "promptomania", label: "Promptomania Builder" },
    { key: "image2prompt", label: "Image → Prompt" },
];

let _state = {
    tab: "lexica",
    q: "",
    filters: {},        // { checkpoint, loras:[], sampler, scheduler, cfg, steps, width, height, prompt }
    useFlags: { ckpt: true, loras: true, dims: false, prompt: true },
    results: [],
    selected: null,     // currently previewed card
};

// ----------------------------------------------------------- styles
function box() { return `background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:6px`; }
function btnPrimary() { return `background:var(--c2c-mauve);color:var(--c2c-bg);border:none;border-radius:5px;padding:6px 14px;cursor:pointer;font-size:12px`; }
function btnGhost() { return `background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:5px;padding:6px 12px;cursor:pointer;font-size:12px`; }
function inputStyle() { return `width:100%;box-sizing:border-box;background:var(--c2c-bg);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:5px;padding:6px;font-size:12px`; }

// ----------------------------------------------------------- graph scan (settings-aware prefill, ALL 7)
function scanGraphFilters() {
    const out = { checkpoint: "", loras: [], sampler: "", scheduler: "", cfg: null, steps: null, width: null, height: null, prompt: "" };
    try {
        forAllNodes((n) => {
            const t = n.type || "";
            const w = (name) => (n.widgets || []).find(x => x.name === name)?.value;
            if (/CheckpointLoaderSimple|CheckpointLoader|UNETLoader/i.test(t)) {
                out.checkpoint = out.checkpoint || w("ckpt_name") || w("unet_name") || "";
            }
            if (/LoraLoader/i.test(t)) {
                const ln = w("lora_name");
                if (ln && !out.loras.includes(ln)) out.loras.push(ln);
            }
            if (/KSampler/i.test(t)) {
                out.sampler = out.sampler || w("sampler_name") || "";
                out.scheduler = out.scheduler || w("scheduler") || "";
                if (out.cfg == null && w("cfg") != null) out.cfg = w("cfg");
                if (out.steps == null && w("steps") != null) out.steps = w("steps");
            }
            if (/EmptyLatentImage|EmptySD3LatentImage/i.test(t)) {
                if (out.width == null && w("width") != null) out.width = w("width");
                if (out.height == null && w("height") != null) out.height = w("height");
            }
        });
        // Positive prompt = CLIPTextEncode feeding KSampler.positive (best-effort).
        out.prompt = findPositivePromptText() || "";
    } catch (exc) {
        __c2cReport("c2c_preset_modal:scanGraph", exc);
    }
    return out;
}

function findPositivePromptText() {
    let positiveText = "";
    forAllNodes((s, g) => {
        if (!/KSampler/i.test(s.type || "")) return;
        const pos = (s.inputs || []).find(i => i.name === "positive");
        if (pos?.link == null) return;
        const ln = g?.links?.[pos.link];
        const src = ln && g?.getNodeById?.(ln.origin_id);
        const txt = src && (src.widgets || []).find(w => w.name === "text")?.value;
        if (txt && !positiveText) positiveText = txt;
    });
    return positiveText;
}

// ----------------------------------------------------------- query assembly per source
function buildQueryForSource(source) {
    const f = _state.filters;
    const use = _state.useFlags;
    const tokens = [];
    if (_state.q) tokens.push(_state.q);
    else if (use.prompt && f.prompt) tokens.push(f.prompt.slice(0, 200));
    if (use.ckpt && f.checkpoint) tokens.push(stripExt(f.checkpoint));
    if (use.loras && f.loras?.length) tokens.push(...f.loras.map(stripExt));
    return tokens.join(" ").trim();
}
function stripExt(s) { return String(s || "").replace(/\.(safetensors|ckpt|pt|bin)$/i, "").replace(/[\\/]/g, " "); }

// ----------------------------------------------------------- backend calls
async function fetchSources() {
    try {
        const r = await fetch("/c2c/presets/sources");
        return (await r.json()).sources || [];
    } catch (exc) { __c2cReport("c2c_preset_modal:sources", exc); return []; }
}
async function fetchSearch(source, q) {
    const filtersJson = encodeURIComponent(JSON.stringify(_state.filters));
    const url = `/c2c/presets/search?source=${encodeURIComponent(source)}&q=${encodeURIComponent(q)}&filters=${filtersJson}&page=0`;
    const r = await fetch(url);
    return await r.json();
}
async function fetchRefresh(source, q) {
    const r = await fetch("/c2c/presets/refresh", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, q, filters: _state.filters }),
    });
    return await r.json();
}
async function fetchTaxonomy() {
    const r = await fetch("/c2c/presets/taxonomy?source=promptomania");
    return await r.json();
}
async function fetchInterrogate(imageB64) {
    const r = await fetch("/c2c/presets/interrogate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: imageB64 }),
    });
    return await r.json();
}

// ----------------------------------------------------------- modal scaffold
function ensureModal() {
    let modal = document.getElementById(MODAL_ID);
    if (modal) return modal;

    modal = document.createElement("div");
    modal.id = MODAL_ID;
    modal.style.cssText = `position:fixed;inset:0;z-index:1000000;display:none;
        background:rgba(0,0,0,0.55);backdrop-filter:blur(2px);
        font-family:ui-sans-serif,system-ui,sans-serif;`;
    modal.innerHTML = `
      <div style="position:absolute;inset:3% 4%;${box()};display:flex;flex-direction:column;overflow:hidden;box-shadow:0 12px 48px rgba(0,0,0,0.5)">
        <div style="display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid var(--c2c-border)">
          <span style="color:var(--c2c-mauve);font-weight:600;letter-spacing:0.5px;text-transform:uppercase;font-size:13px">Preset Hub</span>
          <div id="ph-tabs" style="display:flex;gap:4px;flex-wrap:wrap;flex:1"></div>
          <button id="ph-close" title="Close" style="${btnGhost()};padding:4px 10px">✕</button>
        </div>
        <div style="display:flex;flex:1;min-height:0">
          <div id="ph-sidebar" style="width:240px;border-right:1px solid var(--c2c-border);padding:12px;overflow:auto"></div>
          <div id="ph-main" style="flex:1;display:flex;min-width:0">
            <div id="ph-grid" style="flex:1;padding:12px;overflow:auto"></div>
            <div id="ph-preview" style="width:320px;border-left:1px solid var(--c2c-border);padding:12px;overflow:auto;display:none"></div>
          </div>
        </div>
        <div id="ph-footer" style="padding:6px 14px;border-top:1px solid var(--c2c-border);font-size:11px;color:var(--c2c-sub);display:flex;justify-content:space-between;align-items:center"></div>
      </div>`;
    document.body.appendChild(modal);

    modal.querySelector("#ph-close").onclick = () => close();
    modal.addEventListener("mousedown", (e) => { if (e.target === modal) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && modal.style.display !== "none") close(); });
    return modal;
}

function renderTabs(modal) {
    const host = modal.querySelector("#ph-tabs");
    host.innerHTML = "";
    for (const t of TABS) {
        const b = document.createElement("button");
        b.textContent = t.label;
        const active = t.key === _state.tab;
        b.style.cssText = `padding:4px 10px;border-radius:5px;cursor:pointer;font-size:11px;border:1px solid var(--c2c-border);
            background:${active ? "var(--c2c-mauve)" : "var(--c2c-bg2)"};
            color:${active ? "var(--c2c-bg)" : "var(--c2c-fg)"}`;
        b.onclick = () => { _state.tab = t.key; _state.selected = null; renderAll(modal); };
        host.appendChild(b);
    }
}

function renderSidebar(modal) {
    const host = modal.querySelector("#ph-sidebar");
    const f = _state.filters;
    const builderOrLocal = _state.tab === "promptomania" || _state.tab === "image2prompt";
    if (builderOrLocal) { host.innerHTML = `<div style="font-size:11px;color:var(--c2c-sub)">This tab has no search filters.</div>`; return; }
    host.innerHTML = `
      <div style="font-size:11px;color:var(--c2c-sub);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px">Filters</div>
      <label style="display:flex;gap:6px;align-items:center;font-size:12px;margin:4px 0"><input type="checkbox" id="ph-use-ckpt" ${_state.useFlags.ckpt ? "checked" : ""}/> Use my checkpoint</label>
      <div style="font-size:10px;color:var(--c2c-sub);margin:0 0 6px 22px">${f.checkpoint ? stripExt(f.checkpoint) : "(none detected)"}</div>
      <label style="display:flex;gap:6px;align-items:center;font-size:12px;margin:4px 0"><input type="checkbox" id="ph-use-loras" ${_state.useFlags.loras ? "checked" : ""}/> Use my LoRAs (${f.loras?.length || 0})</label>
      <label style="display:flex;gap:6px;align-items:center;font-size:12px;margin:4px 0"><input type="checkbox" id="ph-use-prompt" ${_state.useFlags.prompt ? "checked" : ""}/> Seed from graph prompt</label>
      <label style="display:flex;gap:6px;align-items:center;font-size:12px;margin:4px 0"><input type="checkbox" id="ph-use-dims" ${_state.useFlags.dims ? "checked" : ""}/> Dims ${f.width && f.height ? `${f.width}×${f.height}` : ""}</label>
      <label style="display:block;font-size:10px;color:var(--c2c-sub);margin-top:10px">Query</label>
      <input id="ph-query" type="text" value="${_esc(_state.q)}" placeholder="search terms" style="${inputStyle()}"/>
      <button id="ph-apply" style="${btnPrimary()};margin-top:10px;width:100%">Apply →</button>
      <button id="ph-rescan" style="${btnGhost()};margin-top:6px;width:100%">Re-scan graph</button>`;

    host.querySelector("#ph-use-ckpt").onchange = (e) => { _state.useFlags.ckpt = e.target.checked; };
    host.querySelector("#ph-use-loras").onchange = (e) => { _state.useFlags.loras = e.target.checked; };
    host.querySelector("#ph-use-prompt").onchange = (e) => { _state.useFlags.prompt = e.target.checked; };
    host.querySelector("#ph-use-dims").onchange = (e) => { _state.useFlags.dims = e.target.checked; };
    host.querySelector("#ph-query").oninput = (e) => { _state.q = e.target.value; };
    host.querySelector("#ph-apply").onclick = () => runSearch(modal);
    host.querySelector("#ph-rescan").onclick = () => { _state.filters = scanGraphFilters(); renderSidebar(modal); };
}

function renderGrid(modal, opts = {}) {
    const grid = modal.querySelector("#ph-grid");
    if (_state.tab === "promptomania") { renderBuilder(modal, grid); return; }
    if (_state.tab === "image2prompt") { renderImage2Prompt(modal, grid); return; }

    if (opts.loading) {
        grid.innerHTML = `<div style="color:var(--c2c-sub);font-size:12px">Fetching live from ${tabLabel()}…</div>`;
        return;
    }
    const results = _state.results || [];
    if (!results.length) {
        grid.innerHTML = `<div style="color:var(--c2c-sub);font-size:12px">No results. Try a different query or press Apply.</div>`;
        return;
    }
    grid.style.display = "grid";
    grid.style.gridTemplateColumns = "repeat(auto-fill, minmax(150px, 1fr))";
    grid.style.gap = "10px";
    grid.style.alignContent = "start";
    grid.innerHTML = "";
    for (const card of results) {
        const c = document.createElement("div");
        c.style.cssText = `${box()};overflow:hidden;cursor:pointer;display:flex;flex-direction:column`;
        const img = card.thumb || card.image;
        c.innerHTML = `
          ${img ? `<img src="${_esc(img)}" loading="lazy" style="width:100%;height:120px;object-fit:cover;background:var(--c2c-bg)"/>`
                : `<div style="height:120px;display:flex;align-items:center;justify-content:center;color:var(--c2c-sub);font-size:10px">${_esc(card.kind || "prompt")}</div>`}
          <div style="padding:6px;font-size:10px;color:var(--c2c-fg);max-height:54px;overflow:hidden">${_esc((card.prompt || card.model || card.id || "").slice(0, 90))}</div>`;
        c.onclick = () => { _state.selected = card; renderPreview(modal); };
        grid.appendChild(c);
    }
}

function renderPreview(modal) {
    const p = modal.querySelector("#ph-preview");
    const card = _state.selected;
    if (!card) { p.style.display = "none"; return; }
    p.style.display = "block";
    const params = [];
    if (card.model) params.push(`model: ${_esc(card.model)}`);
    if (card.sampler) params.push(`sampler: ${_esc(card.sampler)}`);
    if (card.cfg != null) params.push(`cfg: ${card.cfg}`);
    if (card.steps != null) params.push(`steps: ${card.steps}`);
    if (card.seed) params.push(`seed: ${_esc(card.seed)}`);
    if (card.width && card.height) params.push(`${card.width}×${card.height}`);
    p.innerHTML = `
      ${card.image ? `<img src="${_esc(card.image)}" style="width:100%;border-radius:6px;background:var(--c2c-bg)"/>` : ""}
      <div style="margin-top:8px;font-size:11px;color:var(--c2c-sub)">Prompt</div>
      <div style="font-size:11px;white-space:pre-wrap;border:1px solid var(--c2c-border);border-radius:5px;padding:6px;margin-top:2px;max-height:160px;overflow:auto">${_esc(card.prompt || "(none)")}</div>
      ${card.negative ? `<div style="margin-top:6px;font-size:11px;color:var(--c2c-sub)">Negative</div>
        <div style="font-size:11px;white-space:pre-wrap;border:1px solid var(--c2c-border);border-radius:5px;padding:6px;margin-top:2px;max-height:100px;overflow:auto">${_esc(card.negative)}</div>` : ""}
      ${params.length ? `<div style="margin-top:6px;font-size:10px;color:var(--c2c-sub)">${params.join(" · ")}</div>` : ""}
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px">
        <button id="ph-apply-graph" style="${btnPrimary()}">Apply to graph</button>
        <button id="ph-copy" style="${btnGhost()}">Copy</button>
        ${card.permalink ? `<button id="ph-open" style="${btnGhost()}">Open original ↗</button>` : ""}
      </div>
      <div id="ph-apply-msg" style="margin-top:8px;font-size:10px;color:var(--c2c-sub)"></div>`;

    p.querySelector("#ph-apply-graph").onclick = () => applyCardToGraph(modal, card);
    p.querySelector("#ph-copy").onclick = () => {
        navigator.clipboard?.writeText(card.prompt || "");
        p.querySelector("#ph-apply-msg").textContent = "Copied prompt.";
    };
    const openBtn = p.querySelector("#ph-open");
    if (openBtn) openBtn.onclick = () => window.open(card.permalink, "_blank", "noopener");
}

// ----------------------------------------------------------- apply-to-graph per card type
function applyCardToGraph(modal, card) {
    const msg = modal.querySelector("#ph-apply-msg");
    try {
        const kind = card.kind || "prompt";
        if (kind === "model") {
            // Enqueue a Manager download if available; otherwise copy id.
            const id = card.model || card.id;
            navigator.clipboard?.writeText(id);
            msg.textContent = `Model id copied: ${id}. Use Manager → Install Models to download.`;
            return;
        }
        if (kind === "workflow") {
            msg.textContent = "Workflow card — open the original to import (graph replace requires the source JSON).";
            if (card.permalink) window.open(card.permalink, "_blank", "noopener");
            return;
        }
        if (kind === "issue") {
            if (card.permalink) window.open(card.permalink, "_blank", "noopener");
            msg.textContent = "Opened issue in a new tab.";
            return;
        }
        // Default: prompt card → ask positive or negative, write to CLIPTextEncode.
        const target = window.confirm("Apply as POSITIVE prompt?\n\nOK = positive, Cancel = negative") ? "positive" : "negative";
        const enc = findEncoder(target);
        if (!enc) { msg.textContent = "No CLIPTextEncode found for " + target + "."; return; }
        const w = (enc.widgets || []).find(x => x.name === "text");
        if (!w) { msg.textContent = "Encoder has no text widget."; return; }
        w.value = target === "negative" ? (card.negative || card.prompt || "") : (card.prompt || "");
        enc.setDirtyCanvas?.(true, true);
        msg.textContent = `Applied to ${target} encoder #${enc.id}.`;
    } catch (exc) {
        __c2cReport("c2c_preset_modal:applyCard", exc);
        msg.textContent = "Apply failed: " + (exc?.message || exc);
    }
}

function findEncoder(role) {
    // Prefer the encoder wired to KSampler[role]; else first CLIPTextEncode.
    let wired = null, first = null;
    forAllNodes((s, g) => {
        if (/KSampler/i.test(s.type || "")) {
            const slot = (s.inputs || []).find(i => i.name === role);
            if (slot?.link != null) {
                const ln = g?.links?.[slot.link];
                const src = ln && g?.getNodeById?.(ln.origin_id);
                if (src && /CLIPTextEncode/.test(src.type || "")) wired = wired || src;
            }
        }
    });
    if (wired) return wired;
    forAllNodes((n) => { if (!first && /CLIPTextEncode/.test(n.type || "")) first = n; });
    return first;
}

// ----------------------------------------------------------- Promptomania builder
let _taxonomyCache = null;
async function renderBuilder(modal, grid) {
    grid.style.display = "block";
    grid.innerHTML = `<div style="color:var(--c2c-sub);font-size:12px">Loading builder taxonomy…</div>`;
    if (!_taxonomyCache) {
        const j = await fetchTaxonomy();
        if (!j.ok) { grid.innerHTML = `<div style="color:var(--c2c-red);font-size:12px">Taxonomy unavailable: ${_esc(j.message || "")}</div>`; return; }
        _taxonomyCache = j.taxonomy;
    }
    const tax = _taxonomyCache;
    const rows = (tax.categories || []).map(cat => `
        <label style="display:block;font-size:10px;color:var(--c2c-sub);margin-top:8px">${_esc(cat.label)}</label>
        <select data-cat="${_esc(cat.key)}" style="${inputStyle()}">
          <option value="">—</option>
          ${(cat.options || []).map(o => `<option value="${_esc(o)}">${_esc(o)}</option>`).join("")}
        </select>`).join("");
    const negs = (tax.negative_presets || []).map(n => `<option value="${_esc(n.value)}">${_esc(n.label)}</option>`).join("");
    grid.innerHTML = `
      <div style="max-width:520px">
        <div style="font-size:11px;color:var(--c2c-sub);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px">Promptomania Builder (native)</div>
        ${rows}
        <label style="display:block;font-size:10px;color:var(--c2c-sub);margin-top:10px">Negative preset</label>
        <select id="ph-b-neg" style="${inputStyle()}"><option value="">—</option>${negs}</select>
        <label style="display:block;font-size:10px;color:var(--c2c-sub);margin-top:10px">Composed positive</label>
        <textarea id="ph-b-pos" rows="3" style="${inputStyle()};font-family:ui-monospace,monospace"></textarea>
        <div style="display:flex;gap:6px;margin-top:10px">
          <button id="ph-b-apply" style="${btnPrimary()}">Apply to graph</button>
          <button id="ph-b-copy" style="${btnGhost()}">Copy</button>
        </div>
        <div id="ph-b-msg" style="margin-top:8px;font-size:10px;color:var(--c2c-sub)"></div>
      </div>`;

    const recompose = () => {
        const toks = [];
        grid.querySelectorAll("select[data-cat]").forEach(s => { if (s.value) toks.push(s.value); });
        grid.querySelector("#ph-b-pos").value = toks.join(", ");
    };
    grid.querySelectorAll("select[data-cat]").forEach(s => s.onchange = recompose);
    grid.querySelector("#ph-b-apply").onclick = () => {
        const pos = grid.querySelector("#ph-b-pos").value;
        const neg = grid.querySelector("#ph-b-neg").value;
        const pe = findEncoder("positive");
        const ne = findEncoder("negative");
        let done = [];
        if (pe) { const w = (pe.widgets || []).find(x => x.name === "text"); if (w) { w.value = pos; pe.setDirtyCanvas?.(true, true); done.push("positive"); } }
        if (neg && ne) { const w = (ne.widgets || []).find(x => x.name === "text"); if (w) { w.value = neg; ne.setDirtyCanvas?.(true, true); done.push("negative"); } }
        grid.querySelector("#ph-b-msg").textContent = done.length ? "Applied: " + done.join(", ") : "No CLIPTextEncode found.";
    };
    grid.querySelector("#ph-b-copy").onclick = () => {
        navigator.clipboard?.writeText(grid.querySelector("#ph-b-pos").value);
        grid.querySelector("#ph-b-msg").textContent = "Copied positive.";
    };
}

// ----------------------------------------------------------- Image → Prompt
function renderImage2Prompt(modal, grid) {
    grid.style.display = "block";
    grid.innerHTML = `
      <div style="max-width:520px">
        <div style="font-size:11px;color:var(--c2c-sub);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">Image → Prompt (local CLIP-Interrogator)</div>
        <input type="file" id="ph-i-file" accept="image/*" style="${inputStyle()}"/>
        <div id="ph-i-thumb" style="margin-top:8px"></div>
        <button id="ph-i-go" style="${btnPrimary()};margin-top:8px" disabled>Interrogate</button>
        <label style="display:block;font-size:10px;color:var(--c2c-sub);margin-top:10px">Result prompt</label>
        <textarea id="ph-i-out" rows="4" style="${inputStyle()};font-family:ui-monospace,monospace"></textarea>
        <div style="display:flex;gap:6px;margin-top:8px">
          <button id="ph-i-apply" style="${btnPrimary()}">Apply to positive</button>
          <button id="ph-i-copy" style="${btnGhost()}">Copy</button>
        </div>
        <div id="ph-i-msg" style="margin-top:8px;font-size:10px;color:var(--c2c-sub)">Note: first run downloads CLIP/BLIP weights.</div>
      </div>`;
    let b64 = "";
    grid.querySelector("#ph-i-file").onchange = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
            b64 = reader.result;
            grid.querySelector("#ph-i-thumb").innerHTML = `<img src="${b64}" style="max-width:200px;border-radius:6px"/>`;
            grid.querySelector("#ph-i-go").disabled = false;
        };
        reader.readAsDataURL(file);
    };
    grid.querySelector("#ph-i-go").onclick = async () => {
        const msg = grid.querySelector("#ph-i-msg");
        msg.textContent = "Interrogating (this may take a moment)…";
        const j = await fetchInterrogate(b64);
        if (j.ok) { grid.querySelector("#ph-i-out").value = j.prompt; msg.textContent = "Done."; }
        else { msg.textContent = "Failed: " + (j.message || ""); }
    };
    grid.querySelector("#ph-i-apply").onclick = () => {
        const pe = findEncoder("positive");
        const txt = grid.querySelector("#ph-i-out").value;
        if (pe) { const w = (pe.widgets || []).find(x => x.name === "text"); if (w) { w.value = txt; pe.setDirtyCanvas?.(true, true); grid.querySelector("#ph-i-msg").textContent = "Applied to positive #" + pe.id; return; } }
        grid.querySelector("#ph-i-msg").textContent = "No CLIPTextEncode found.";
    };
    grid.querySelector("#ph-i-copy").onclick = () => {
        navigator.clipboard?.writeText(grid.querySelector("#ph-i-out").value);
        grid.querySelector("#ph-i-msg").textContent = "Copied.";
    };
}

// ----------------------------------------------------------- footer / cache badge
function renderFooter(modal, info) {
    const f = modal.querySelector("#ph-footer");
    if (_state.tab === "promptomania" || _state.tab === "image2prompt") { f.innerHTML = `<span>Local tab — no network fetch.</span><span></span>`; return; }
    const age = info?.age_seconds;
    const ageStr = age == null ? "" : age < 90 ? "just now" : age < 3600 ? `${Math.round(age / 60)}m ago` : `${Math.round(age / 3600)}h ago`;
    const badge = info?.cached ? `${tabLabel()} · fetched ${ageStr}` : `${tabLabel()} · live`;
    f.innerHTML = `<span>${_esc(badge)}</span>
      <span><button id="ph-refresh" style="${btnGhost()};padding:3px 10px">Refresh ↻</button></span>`;
    const rb = f.querySelector("#ph-refresh");
    if (rb) rb.onclick = () => runSearch(modal, { force: true });
}

// ----------------------------------------------------------- orchestration
async function runSearch(modal, opts = {}) {
    if (_state.tab === "promptomania" || _state.tab === "image2prompt") { renderGrid(modal); renderFooter(modal); return; }
    const q = buildQueryForSource(_state.tab);
    renderGrid(modal, { loading: true });
    let j;
    try {
        j = opts.force ? await fetchRefresh(_state.tab, q) : await fetchSearch(_state.tab, q);
    } catch (exc) {
        __c2cReport("c2c_preset_modal:search", exc);
        modal.querySelector("#ph-grid").innerHTML = `<div style="color:var(--c2c-red);font-size:12px">Search failed: ${_esc(exc?.message || exc)}</div>`;
        return;
    }
    _state.results = j.results || [];
    renderGrid(modal);
    renderFooter(modal, j);
    if (!j.ok && j.message) {
        const f = modal.querySelector("#ph-footer");
        f.insertAdjacentHTML("afterbegin", `<span style="color:var(--c2c-red)">⚠ ${_esc(j.message)} (showing cached if any) </span>`);
    }
}

function renderAll(modal) {
    renderTabs(modal);
    renderSidebar(modal);
    renderPreview(modal);
    runSearch(modal);
}

function tabLabel() { return (TABS.find(t => t.key === _state.tab) || {}).label || _state.tab; }
function _esc(s) { return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }

// ----------------------------------------------------------- public open/close
function open(optsArg = {}) {
    const modal = ensureModal();
    if (optsArg.tab && TABS.some(t => t.key === optsArg.tab)) _state.tab = optsArg.tab;
    if (typeof optsArg.q === "string") _state.q = optsArg.q;
    // Always re-scan the graph on open (settings-aware prefill).
    _state.filters = scanGraphFilters();
    if (optsArg.filters) _state.filters = Object.assign({}, _state.filters, optsArg.filters);
    _state.selected = null;
    modal.style.display = "block";
    renderAll(modal);
}
function close() {
    const modal = document.getElementById(MODAL_ID);
    if (modal) modal.style.display = "none";
}

window.__C2C_PRESET_HUB__ = { open, close };

// ----------------------------------------------------------- register (command + menu)
app.registerExtension({
    name: "c2c.preset.hub",
    async setup() {
        try {
            app.extensionManager?.registerCommand?.({
                id: "c2c.presetHub.open",
                label: "C2C: Open Preset Hub",
                function: () => open({ tab: "lexica" }),
            });
        } catch (exc) {
            console.warn("[c2c.preset.hub] command registration failed:", exc);
        }
        // OmniBar AI-section shortcut chip (public register API).
        try {
            const chip = document.createElement("button");
            chip.textContent = "🔍 Presets";
            chip.title = "Open the C2C Preset Hub";
            chip.style.cssText = "background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:5px;padding:3px 8px;cursor:pointer;font-size:11px";
            chip.onclick = () => open({ tab: "lexica" });
            window.C2COmniBar?.register?.({
                section: "ai",
                id: "preset-hub",
                order: 60,
                element: chip,
                onMode: (mode) => { chip.textContent = mode === "icon" ? "🔍" : "🔍 Presets"; },
            });
        } catch (exc) {
            console.warn("[c2c.preset.hub] omnibar chip registration failed:", exc);
        }
    },
});
