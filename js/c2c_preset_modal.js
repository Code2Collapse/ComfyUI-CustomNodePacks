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
import { c2cConfirm, c2cPrompt } from "./_c2c_dialog.js";
import { capabilityFor, nodeColor } from "./c2c_node_taxonomy.js";
import { renderGraphPreview, legendHTML } from "./c2c_graph_preview.js";
import { scoreLocal } from "./c2c_workflow_library.js";

const MODAL_ID = "c2c-preset-hub-modal";

// Tabs map 1:1 to backend sources + the two local tabs.
const TABS = [
    { key: "workflows", label: "\ud83d\udcc2 Workflows" },
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
    // 3-state NSFW filter (Track A.1, 2026-05-30):
    //   "sfw"  → drop any card with nsfw || nsfw_unknown.
    //   "both" → render everything; blur nsfw || nsfw_unknown (DEFAULT).
    //   "nsfw" → drop cards with nsfw === false && nsfw_unknown === false.
    // Persisted in localStorage under "c2c.presetHub.nsfwMode".
    nsfwMode: "both",
    showNsfw: false,    // legacy alias: true iff nsfwMode === "nsfw".
};

// Restore persisted NSFW mode on module load (best-effort; storage may be
// unavailable in some embedded contexts).
try {
    const _persisted = localStorage.getItem("c2c.presetHub.nsfwMode");
    if (_persisted === "sfw" || _persisted === "both" || _persisted === "nsfw") {
        _state.nsfwMode = _persisted;
        _state.showNsfw = (_persisted === "nsfw");
    }
} catch (_e) { /* localStorage unavailable — keep defaults */ }

// Workflows tab (merged local + online) sub-state.
const _wf = {
    scope: "both",      // "local" | "online" | "both"
    dirs: [],           // scan locations
    dirsLoaded: false,
    scanned: false,
    scanning: false,
    fingerprints: [],   // last scan fingerprints
    local: [],          // scored local results
    online: [],         // online (OpenArt) workflow cards
    status: "",
};

// ----------------------------------------------------------- styles
function box() { return `background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:6px`; }
function btnPrimary() { return `background:var(--c2c-mauve);color:var(--c2c-bg);border:none;border-radius:5px;padding:6px 14px;cursor:pointer;font-size:12px`; }
function btnGhost() { return `background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:5px;padding:6px 12px;cursor:pointer;font-size:12px`; }
function inputStyle() { return `width:100%;box-sizing:border-box;background:var(--c2c-bg);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:5px;padding:6px;font-size:12px`; }

// ----------------------------------------------------------- NSFW thumbnails
// Returns the markup for a card thumbnail cell. NSFW-flagged images are
// blurred with a click-to-reveal overlay unless the user has toggled
// "show NSFW" on. The cell is position:relative so the overlay can fill it.
function isBlurred(card) {
    // In "nsfw" mode the user has explicitly asked for NSFW content — no blur.
    if (_state.nsfwMode === "nsfw") return false;
    // In "sfw" mode the card never reaches the renderer (filtered upstream),
    // so the only mode where blur applies is "both".
    return !!card.nsfw || !!card.nsfw_unknown;
}

// Track A.1 — return true iff the card is allowed under the current NSFW
// filter mode. Used by every render path that builds a result list.
function passesNsfwFilter(card) {
    const flagged = !!(card && (card.nsfw || card.nsfw_unknown));
    if (_state.nsfwMode === "sfw") return !flagged;
    if (_state.nsfwMode === "nsfw") return flagged;
    return true; // "both"
}

// Normalise a card returned by an online workflow source so the NSFW
// fields are always defined. OpenArt and other community feeds have no
// explicit NSFW flag; older server builds did not stamp nsfw_unknown on
// every record, so we conservatively default to "unverified" (blurred)
// whenever the source did not provide either field.
const _UNVERIFIED_SOURCES = new Set(["openart"]);
function normalizeOnlineWorkflowCard(card) {
    if (card == null || typeof card !== "object") return card;
    if (card.nsfw === undefined) card.nsfw = false;
    if (card.nsfw_unknown === undefined) {
        // Default-blur for known-unverified sources, default-clear otherwise.
        card.nsfw_unknown = _UNVERIFIED_SOURCES.has(String(card.source || ""));
    }
    return card;
}

// Track A.2 — collapse cards that represent the same workflow across (or
// within) sources. Dedupe key: lowercased title + first-token of author,
// falling back to the thumbnail URL. The first occurrence wins and gets an
// `also_on` array listing the other source labels the work appeared in.
function dedupeWorkflows(cards) {
    if (!Array.isArray(cards) || cards.length < 2) return cards || [];
    const _firstToken = (s) => String(s || "").trim().toLowerCase().split(/\s+/)[0] || "";
    const _norm = (s) => String(s || "").toLowerCase().replace(/<[^>]*>/g, " ")
        .replace(/[^a-z0-9]+/g, " ").trim();
    const _key = (c) => {
        const ex = c.extra || {};
        const title = _norm(ex.title || c.prompt || c.id || "");
        const author = _firstToken(ex.author);
        if (title) return `t:${title}|a:${author}`;
        const thumb = String(c.thumb || c.image || "").trim().toLowerCase();
        return thumb ? `u:${thumb}` : "";
    };
    const seen = new Map();
    const out = [];
    for (const c of cards) {
        const k = _key(c);
        if (!k) { out.push(c); continue; }
        const prior = seen.get(k);
        if (prior) {
            // Record the additional source on the survivor.
            const src = String(c.source || c.__source || "").trim();
            if (src && src !== prior.source) {
                prior.also_on = prior.also_on || [];
                if (!prior.also_on.includes(src)) prior.also_on.push(src);
            }
            continue;
        }
        seen.set(k, c);
        out.push(c);
    }
    return out;
}
function thumbCellHtml(card, { h = 120, fallback = "prompt" } = {}) {
    const img = card.thumb || card.image;
    const blurred = isBlurred(card);
    const unverified = !card.nsfw && !!card.nsfw_unknown;
    const label = unverified ? "unverified \u00b7 click to reveal" : "NSFW \u00b7 click to reveal";
    const imgTag = img
        ? `<img src="${_esc(img)}" loading="lazy" style="width:100%;height:${h}px;object-fit:cover;background:var(--c2c-bg);${blurred ? "filter:blur(20px)" : ""}"/>`
        : `<div style="height:${h}px;display:flex;align-items:center;justify-content:center;color:var(--c2c-sub);font-size:10px">${_esc(fallback)}</div>`;
    const overlay = blurred
        ? `<div class="ph-nsfw-reveal" style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;background:rgba(0,0,0,0.45);cursor:pointer">
             <span style="font-size:18px">\ud83d\udd1e</span>
             <span style="font-size:9px;color:#fff;letter-spacing:0.5px">${label}</span>
           </div>`
        : "";
    return `<div style="position:relative;width:100%">${imgTag}${overlay}</div>`;
}

// Wire reveal overlays inside a freshly-rendered container. Clicking an
// overlay unblurs only that card and stops the click from selecting it.
function wireNsfwReveals(root) {
    root.querySelectorAll(".ph-nsfw-reveal").forEach((ov) => {
        ov.onclick = (e) => {
            e.stopPropagation();
            const cell = ov.parentElement;
            const im = cell && cell.querySelector("img");
            if (im) im.style.filter = "";
            ov.remove();
        };
    });
}

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

// Retry-with-jitter helper for optional backend fetches that may race the
// server boot. Returns the eventual successful response, or throws the LAST
// error if all attempts fail. Delays in ms; jitter is ±20%.
async function _retry(fn, label, attempts = 3, baseDelays = [250, 500, 1000]) {
    let lastErr;
    for (let i = 0; i < attempts; i++) {
        try {
            return await fn();
        } catch (exc) {
            lastErr = exc;
            if (i === attempts - 1) break;
            const base = baseDelays[Math.min(i, baseDelays.length - 1)] || 1000;
            const jitter = base * (0.8 + Math.random() * 0.4);
            // Intermediate-retry noise stays at "warn" → console only, no toast.
            __c2cReport(`c2c_preset_modal:${label}:retry${i + 1}`, exc, { level: "warn" });
            await new Promise((r) => setTimeout(r, jitter));
        }
    }
    throw lastErr;
}

// One-time backend-readiness probe. Returns true if /c2c/presets/sources is
// alive (server-side preset hub is mounted). Cached on the modal until reload.
let _backendReady = null;
async function probePresetBackend() {
    if (_backendReady !== null) return _backendReady;
    try {
        const r = await fetch("/c2c/presets/sources");
        _backendReady = r.ok;
    } catch (_e) {
        _backendReady = false;
    }
    return _backendReady;
}

async function fetchSources() {
    try {
        const r = await fetch("/c2c/presets/sources");
        return (await r.json()).sources || [];
    } catch (exc) { __c2cReport("c2c_preset_modal:sources", exc); return []; }
}
async function fetchSearch(source, q) {
    const filtersJson = encodeURIComponent(JSON.stringify(_state.filters));
    const url = `/c2c/presets/search?source=${encodeURIComponent(source)}&q=${encodeURIComponent(q)}&filters=${filtersJson}&page=0`;
    return await _retry(async () => {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
    }, "search");
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

// ----------------------------------------------------------- library (local workflows) backend
async function libLocations() {
    return await _retry(async () => {
        const r = await fetch("/c2c/library/locations");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
    }, "libLocations");
}
async function libSaveLocations(directories) {
    const r = await fetch("/c2c/library/locations", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ directories }),
    });
    return await r.json();
}
async function libScan(directories) {
    return await _retry(async () => {
        const r = await fetch("/c2c/library/scan", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ directories }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
    }, "libScan");
}
async function libLoad(path) {
    const r = await fetch("/c2c/library/load?path=" + encodeURIComponent(path));
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
    if (_state.tab === "workflows") { renderWorkflowSidebar(modal, host); return; }
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
    if (_state.tab === "workflows") { renderWorkflows(modal, grid, opts); return; }
    if (_state.tab === "promptomania") { renderBuilder(modal, grid); return; }
    if (_state.tab === "image2prompt") { renderImage2Prompt(modal, grid); return; }

    if (opts.loading) {
        grid.innerHTML = `<div style="color:var(--c2c-sub);font-size:12px">Fetching live from ${tabLabel()}…</div>`;
        return;
    }
    const results = (_state.results || []).filter(passesNsfwFilter);
    if (!results.length) {
        // Differentiate "no results from server" vs "all filtered out by NSFW mode".
        const allCount = (_state.results || []).length;
        let hint = "No results. Try a different query or press Apply.";
        if (allCount > 0 && _state.nsfwMode !== "both") {
            const mode = _state.nsfwMode === "sfw" ? "SFW only" : "NSFW only";
            hint = `No results match the “${mode}” filter (${allCount} hidden). Switch the NSFW filter to “Both” to see them.`;
        }
        // Track A.4 — Civitai-specific hint: anonymous Civitai requests are
        // server-capped to SFW. If the user picks "NSFW only" and gets zero
        // results back from the server (allCount === 0), surface the API-key
        // requirement so they don't think the filter is broken.
        if (_state.tab === "civitai" && _state.nsfwMode === "nsfw" && allCount === 0) {
            hint = `Civitai requires an API key to return NSFW results. ` +
                   `Add CIVITAI_API_KEY in your environment (or in Settings → Secrets) and restart ComfyUI.`;
        }
        grid.innerHTML = `<div style="color:var(--c2c-sub);font-size:12px">${_esc(hint)}</div>`;
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
        c.innerHTML = `
          ${_thumbWithOverlays(card, { fallback: card.kind || "prompt" })}
          <div style="padding:6px;display:flex;flex-direction:column;gap:2px">
            <div style="font-size:10px;color:var(--c2c-fg);max-height:54px;overflow:hidden">${_esc((card.prompt || card.model || card.id || "").slice(0, 90))}</div>
            ${_cardBadgesHtml(card)}
          </div>`;
        c.onclick = () => { _state.selected = card; renderPreview(modal); };
        grid.appendChild(c);
    }
    wireNsfwReveals(grid);
    _wirePermalinkClicks(grid);
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
    const previewBlurred = isBlurred(card);
    p.innerHTML = `
      ${card.image ? `<div style="position:relative">
          <img src="${_esc(card.image)}" style="width:100%;border-radius:6px;background:var(--c2c-bg);${previewBlurred ? "filter:blur(24px)" : ""}"/>
          ${previewBlurred ? `<div class="ph-nsfw-reveal" style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;background:rgba(0,0,0,0.45);cursor:pointer;border-radius:6px">
              <span style="font-size:22px">\u{1F51E}</span>
              <span style="font-size:10px;color:#fff;letter-spacing:0.5px">NSFW \u00b7 click to reveal</span>
            </div>` : ""}
        </div>` : ""}
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
    wireNsfwReveals(p);
}

// ----------------------------------------------------------- apply-to-graph per card type
function applyCardToGraph(modal, card) {
    const msg = modal.querySelector("#ph-apply-msg");
    return (async () => {
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
        const target = (await c2cConfirm("Apply as POSITIVE prompt?\n\nOK = positive, Cancel = negative")) ? "positive" : "negative";
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
    })();
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

// ----------------------------------------------------------- Workflows (merged local + online)
function renderWorkflowSidebar(modal, host) {
    const scopeBtn = (key, label) => {
        const active = _wf.scope === key;
        return `<button data-scope="${key}" style="flex:1;padding:5px 0;border-radius:5px;cursor:pointer;font-size:11px;border:1px solid var(--c2c-border);
            background:${active ? "var(--c2c-mauve)" : "var(--c2c-bg2)"};color:${active ? "var(--c2c-bg)" : "var(--c2c-fg)"}">${label}</button>`;
    };
    const locRows = (_wf.dirs || []).length
        ? _wf.dirs.map((d, i) => `
            <div style="display:flex;align-items:center;gap:6px;font-size:11px;margin:3px 0">
              <span data-lib-tog="${i}" style="cursor:pointer;width:16px;text-align:center;color:${d.enabled ? "var(--c2c-green)" : "var(--c2c-sub)"}">${d.enabled ? "\u2611" : "\u2610"}</span>
              <span title="${_esc(d.path)}" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;${d.exists === false ? "color:var(--c2c-red)" : ""}">${_esc(d.path)}${d.exists === false ? " [missing]" : ""}</span>
              <span data-lib-rm="${i}" style="cursor:pointer;color:var(--c2c-sub)">\u00d7</span>
            </div>`).join("")
        : `<div style="font-size:10px;color:var(--c2c-sub)">No folders yet.</div>`;

    host.innerHTML = `
      <div style="font-size:11px;color:var(--c2c-sub);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px">Search scope</div>
      <div style="display:flex;gap:4px">${scopeBtn("local", "Local")}${scopeBtn("online", "Online")}${scopeBtn("both", "Both")}</div>
      <label style="display:block;font-size:10px;color:var(--c2c-sub);margin-top:12px">Query</label>
      <input id="ph-wf-query" type="text" value="${_esc(_state.q)}" placeholder="e.g. generate video from an image" style="${inputStyle()}"/>
      <button id="ph-wf-apply" style="${btnPrimary()};margin-top:10px;width:100%">Search \u2192</button>
      <div style="font-size:11px;color:var(--c2c-sub);text-transform:uppercase;letter-spacing:0.5px;margin:16px 0 6px">Local folders</div>
      <div id="ph-wf-locs">${locRows}</div>
      <div style="display:flex;gap:6px;margin-top:8px">
        <button id="ph-wf-addloc" style="${btnGhost()};flex:1">+ Folder</button>
        <button id="ph-wf-scan" style="${btnGhost()};flex:1">Rescan</button>
      </div>`;

    host.querySelectorAll("[data-scope]").forEach((b) => {
        b.onclick = () => { _wf.scope = b.dataset.scope; renderWorkflowSidebar(modal, host); runSearch(modal); };
    });
    host.querySelector("#ph-wf-query").oninput = (e) => { _state.q = e.target.value; };
    host.querySelector("#ph-wf-query").onkeydown = (e) => { if (e.key === "Enter") runSearch(modal); };
    host.querySelector("#ph-wf-apply").onclick = () => runSearch(modal);
    host.querySelector("#ph-wf-scan").onclick = () => { _wf.scanned = false; runSearch(modal, { force: true }); };
    host.querySelector("#ph-wf-addloc").onclick = async () => {
        const p = await c2cPrompt("Workflow folder path (absolute):", "");
        if (!p) return;
        _wf.dirs = _wf.dirs || [];
        _wf.dirs.push({ path: p, enabled: true });
        const r = await libSaveLocations(_wf.dirs);
        if (r.success) { _wf.dirs = r.directories; _wf.scanned = false; }
        renderWorkflowSidebar(modal, host);
        runSearch(modal, { force: true });
    };
    host.querySelectorAll("[data-lib-tog]").forEach((el) => {
        el.onclick = async () => {
            const i = +el.dataset.libTog;
            _wf.dirs[i].enabled = !_wf.dirs[i].enabled;
            const r = await libSaveLocations(_wf.dirs);
            if (r.success) _wf.dirs = r.directories;
            _wf.scanned = false;
            renderWorkflowSidebar(modal, host);
            runSearch(modal, { force: true });
        };
    });
    host.querySelectorAll("[data-lib-rm]").forEach((el) => {
        el.onclick = async () => {
            _wf.dirs.splice(+el.dataset.libRm, 1);
            const r = await libSaveLocations(_wf.dirs);
            if (r.success) _wf.dirs = r.directories;
            _wf.scanned = false;
            renderWorkflowSidebar(modal, host);
            runSearch(modal, { force: true });
        };
    });
}

async function searchWorkflows(modal, opts = {}) {
    const q = _state.q || "";
    _wf.local = [];
    _wf.online = [];

    // ── local ──
    if (_wf.scope === "local" || _wf.scope === "both") {
        try {
            if (!_wf.dirsLoaded) {
                const loc = await libLocations();
                _wf.dirs = loc.success ? (loc.directories || []) : [];
                _wf.dirsLoaded = true;
                renderSidebar(modal);
            }
            if (!_wf.scanned || opts.force) {
                _wf.scanning = true;
                _wf.status = "Scanning local folders\u2026";
                renderGrid(modal, { loading: true });
                renderFooter(modal);
                const scan = await libScan(_wf.dirs);
                _wf.fingerprints = scan.success ? (scan.workflows || []) : [];
                _wf.scanned = true;
                _wf.scanning = false;
            }
            const pool = _wf.fingerprints;
            _wf.local = (q.trim()
                ? pool.map((fp) => scoreLocal(fp, q)).filter((fp) => fp.score > 0)
                : pool.map((fp) => ({ ...fp, score: 0, matched_terms: [] })))
                .sort((a, b) => (b.score || 0) - (a.score || 0))
                .slice(0, 200);
        } catch (exc) {
            _wf.scanning = false;
            // Optional feature — workflow library backend may not be mounted
            // on slim ComfyUI installs. Log to console only; do NOT toast.
            __c2cReport("c2c_preset_modal:wfLocal", exc, { level: "info" });
            _wf.status = "Local workflow library unavailable (backend not loaded).";
        }
    }

    // ── online (OpenArt workflow community) ──
    if (_wf.scope === "online" || _wf.scope === "both") {
        try {
            const j = await (opts.force ? fetchRefresh("openart", q) : fetchSearch("openart", q));
            _wf.online = (j.results || []).map((c) =>
                normalizeOnlineWorkflowCard({ ...c, __online: true }));
            // Track A.2 — cross-source dedupe. Currently only OpenArt is
            // wired, but dedupe also collapses intra-source duplicates
            // (the public feed occasionally returns the same workflow more
            // than once) and is forward-compatible with future Civitai/HF
            // workflow sources.
            _wf.online = dedupeWorkflows(_wf.online);
            if (!j.ok && j.message) _wf.status = "Online: " + j.message;
        } catch (exc) {
            // Optional — online source may be down. Console only, no toast.
            __c2cReport("c2c_preset_modal:wfOnline", exc, { level: "info" });
        }
    }

    _wf.status = `Local ${_wf.local.length} \u00b7 Online ${_wf.online.length}`;
    renderGrid(modal);
    renderFooter(modal);
}

function renderWorkflows(modal, grid, opts = {}) {
    if (opts.loading || _wf.scanning) {
        grid.style.display = "block";
        grid.innerHTML = `<div style="color:var(--c2c-sub);font-size:12px">${_esc(_wf.status || "Loading\u2026")}</div>`;
        return;
    }
    grid.style.display = "block";
    grid.innerHTML = "";

    const sectionCss = "font-size:11px;color:var(--c2c-sub);text-transform:uppercase;letter-spacing:0.5px;margin:4px 0 8px";
    const gridCss = "display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:18px";

    // Local section
    if (_wf.scope === "local" || _wf.scope === "both") {
        const head = document.createElement("div");
        head.style.cssText = sectionCss;
        head.textContent = `\ud83d\udcc2 Local workflows (${_wf.local.length})`;
        grid.appendChild(head);
        if (!_wf.local.length) {
            const empty = document.createElement("div");
            empty.style.cssText = "color:var(--c2c-sub);font-size:12px;margin-bottom:18px";
            empty.textContent = _wf.dirs.length ? "No local matches. Add folders or try another query." : "No scan folders yet \u2014 add one in the sidebar.";
            grid.appendChild(empty);
        } else {
            const wrap = document.createElement("div");
            wrap.style.cssText = gridCss;
            for (const fp of _wf.local) {
                const chips = (fp.nodes || []).slice(0, 6).map((nt) =>
                    `<span style="display:inline-block;margin:2px;padding:1px 6px;border-radius:3px;font-size:9px;color:#fff;background:${nodeColor(nt)}" title="${_esc(capabilityFor(nt))}">${_esc(nt)}</span>`).join("");
                const card = document.createElement("div");
                card.style.cssText = `${box()};padding:8px;display:flex;flex-direction:column;gap:4px`;
                card.innerHTML = `
                  <div style="font-size:12px;font-weight:600;color:var(--c2c-fg);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${_esc(fp.path)}">${_esc(fp.filename)}</div>
                  <div style="font-size:10px;color:var(--c2c-sub)">${fp.node_count} nodes${fp.score ? ` \u00b7 score ${fp.score.toFixed(1)}` : ""}</div>
                  <div>${chips}</div>
                  <div style="display:flex;gap:6px;margin-top:4px">
                    <button class="wf-open" style="${btnPrimary()};padding:4px 10px;font-size:11px">Open in canvas</button>
                    <button class="wf-prev" style="${btnGhost()};padding:4px 10px;font-size:11px">Preview</button>
                  </div>`;
                card.querySelector(".wf-open").onclick = () => openLocalWorkflow(modal, fp);
                card.querySelector(".wf-prev").onclick = () => previewLocalWorkflow(modal, fp);
                wrap.appendChild(card);
            }
            grid.appendChild(wrap);
        }
    }

    // Online section (reuse prompt-card preview pane via _state.selected)
    if (_wf.scope === "online" || _wf.scope === "both") {
        const onlineFiltered = _wf.online.filter(passesNsfwFilter);
        const hiddenByFilter = _wf.online.length - onlineFiltered.length;
        const head = document.createElement("div");
        head.style.cssText = sectionCss;
        const hiddenStr = hiddenByFilter > 0 ? ` \u00b7 ${hiddenByFilter} hidden by NSFW filter` : "";
        head.textContent = `\ud83c\udf10 Online \u2014 OpenArt (${onlineFiltered.length})${hiddenStr}`;
        grid.appendChild(head);
        if (!onlineFiltered.length) {
            const empty = document.createElement("div");
            empty.style.cssText = "color:var(--c2c-sub);font-size:12px";
            empty.textContent = _wf.online.length
                ? "All online results were hidden by the NSFW filter."
                : "No online results. Try a different query.";
            grid.appendChild(empty);
        } else {
            const wrap = document.createElement("div");
            wrap.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px";
            for (const c of onlineFiltered) {
                const el = document.createElement("div");
                el.style.cssText = `${box()};overflow:hidden;cursor:pointer;display:flex;flex-direction:column`;
                const ex = c.extra || {};
                const title = String(ex.title || c.prompt || c.id || "")
                    .replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
                const sub = ex.node_count ? `${ex.node_count} nodes${ex.author ? ` \u00b7 ${ex.author}` : ""}` : (ex.author || "");
                el.innerHTML = `
                  ${_thumbWithOverlays(c, { fallback: "workflow" })}
                  <div style="padding:6px;display:flex;flex-direction:column;gap:2px">
                    <div style="font-size:11px;font-weight:600;color:var(--c2c-fg);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${_esc(title)}">${_esc(title.slice(0, 70) || "workflow")}</div>
                    ${sub ? `<div style="font-size:9px;color:var(--c2c-sub)">${_esc(sub)}</div>` : ""}
                    ${_cardBadgesHtml(c)}
                  </div>`;
                el.onclick = () => { _state.selected = c; renderPreview(modal); };
                wrap.appendChild(el);
            }
            grid.appendChild(wrap);
            wireNsfwReveals(wrap);
            _wirePermalinkClicks(wrap);
        }
    }
}

let _wfPreview = null;
function previewLocalWorkflow(modal, fp) {
    const p = modal.querySelector("#ph-preview");
    p.style.display = "block";
    p.innerHTML = `
      <div style="font-size:11px;color:var(--c2c-sub)">Graph preview</div>
      <div style="font-size:12px;font-weight:600;margin:2px 0 6px">${_esc(fp.filename)}</div>
      <div id="ph-wf-prev-host" style="font-size:11px;color:var(--c2c-sub)">Loading\u2026</div>
      <div id="ph-wf-prev-legend" style="margin-top:6px;display:none"></div>
      <button id="ph-wf-prev-open" style="${btnPrimary()};margin-top:10px">Open in canvas</button>`;
    p.querySelector("#ph-wf-prev-open").onclick = () => openLocalWorkflow(modal, fp);
    libLoad(fp.path).then((data) => {
        const host = p.querySelector("#ph-wf-prev-host");
        if (!host) return;
        if (!data.success) { host.innerHTML = `<span style="color:var(--c2c-red)">Preview error: ${_esc(data.error || "?")}</span>`; return; }
        host.innerHTML = "";
        if (_wfPreview) { _wfPreview.destroy?.(); _wfPreview = null; }
        _wfPreview = renderGraphPreview(host, data.workflow, { height: 280 });
        const lg = p.querySelector("#ph-wf-prev-legend");
        if (lg) { lg.innerHTML = legendHTML(); lg.style.display = "block"; }
    }).catch((exc) => {
        const host = p.querySelector("#ph-wf-prev-host");
        if (host) host.innerHTML = `<span style="color:var(--c2c-red)">Preview failed: ${_esc(String(exc))}</span>`;
    });
}

async function openLocalWorkflow(modal, fp) {
    _wf.status = "Loading " + fp.filename + "\u2026";
    renderFooter(modal);
    try {
        const data = await libLoad(fp.path);
        if (!data.success) { _wf.status = "Load error: " + (data.error || "?"); renderFooter(modal); return; }
        await app.loadGraphData(data.workflow);
        _wf.status = "Loaded " + fp.filename;
        renderFooter(modal);
        close();
    } catch (exc) {
        __c2cReport("c2c_preset_modal:wfOpen", exc);
        _wf.status = "Open failed: " + (exc?.message || exc);
        renderFooter(modal);
    }
}

// ----------------------------------------------------------- card chip helpers (Track A.3)

// Per-source display label + colour used for the corner "source" badge.
// Themed via --c2c-* CSS vars; falls back to safe hex if a var is missing.
const _SOURCE_META = {
    lexica:      { label: "Lexica",      icon: "L" },
    civitai:     { label: "Civitai",     icon: "C" },
    huggingface: { label: "HF",          icon: "H" },
    openart:     { label: "OpenArt",     icon: "O" },
    promptdexter:{ label: "Pdexter",     icon: "P" },
    c2c_doctor:  { label: "Issues",      icon: "G" },
    workflows:   { label: "Workflow",    icon: "W" },
};
function _sourceLabel(src) {
    const k = String(src || "").toLowerCase();
    return (_SOURCE_META[k] && _SOURCE_META[k].label) || (k || "?");
}

// Tiny themed pill helper. Returns inline-styled HTML; safe to insert with
// innerHTML (caller pre-escapes the label).
function _chipHtml(label, opts = {}) {
    const {
        bg = "var(--c2c-bg)",
        fg = "var(--c2c-fg)",
        border = "var(--c2c-border)",
        title = "",
        extra = "",
    } = opts;
    return (
        `<span title="${_esc(title || label)}" style="display:inline-block;` +
            `padding:1px 6px;border-radius:3px;font-size:9px;font-weight:500;` +
            `background:${bg};color:${fg};border:1px solid ${border};` +
            `margin:1px 2px 0 0;line-height:1.4;${extra}">` +
            _esc(label) +
        `</span>`
    );
}

// Build the cluster of badges that decorate a card (source, NSFW, tag chips,
// "also on" list). Returns a string suitable for inline insertion under the
// card thumbnail.
function _cardBadgesHtml(card, { maxTags = 3 } = {}) {
    const parts = [];
    if (card.source) {
        parts.push(_chipHtml(_sourceLabel(card.source), {
            bg: "var(--c2c-mauve)", fg: "var(--c2c-bg)", border: "var(--c2c-mauve)",
            title: `Source: ${_sourceLabel(card.source)}`,
        }));
    }
    if (card.nsfw === true) {
        parts.push(_chipHtml("NSFW", {
            bg: "var(--c2c-red,#ef4444)", fg: "#fff", border: "var(--c2c-red,#ef4444)",
            title: "Source flagged this card as NSFW.",
        }));
    } else if (card.nsfw_unknown === true) {
        parts.push(_chipHtml("?", {
            bg: "var(--c2c-bg2)", fg: "var(--c2c-sub)", border: "var(--c2c-border)",
            title: "Source does not flag NSFW — thumbnail blurred to be safe.",
        }));
    }
    const tags = Array.isArray(card.tags) ? card.tags.filter(Boolean).slice(0, maxTags) : [];
    for (const t of tags) {
        parts.push(_chipHtml(String(t).slice(0, 14), { title: `Tag: ${t}` }));
    }
    if (Array.isArray(card.also_on) && card.also_on.length) {
        parts.push(_chipHtml(`also on ${card.also_on.length}`, {
            title: `Also published on: ${card.also_on.map(_sourceLabel).join(", ")}`,
        }));
    }
    if (!parts.length) return "";
    return `<div style="display:flex;flex-wrap:wrap;align-items:center;margin-top:2px">${parts.join("")}</div>`;
}

// Width × height pill rendered on top of the thumbnail when dims are known.
function _dimsBadgeHtml(card) {
    if (!card || !card.width || !card.height) return "";
    return (
        `<span style="position:absolute;right:4px;bottom:4px;background:rgba(0,0,0,0.55);` +
            `color:#fff;font-size:9px;padding:1px 5px;border-radius:3px;line-height:1.4">` +
            `${card.width}\u00d7${card.height}` +
        `</span>`
    );
}

// "Open in source" icon button (top-right of thumbnail). Stops click
// propagation so it does not also trigger card-select.
function _permalinkButtonHtml(card) {
    if (!card || !card.permalink) return "";
    return (
        `<a class="c2c-ph-permalink" href="${_esc(card.permalink)}" target="_blank" rel="noopener noreferrer" ` +
            `title="Open original on ${_esc(_sourceLabel(card.source))}" ` +
            `style="position:absolute;right:4px;top:4px;background:rgba(0,0,0,0.55);color:#fff;` +
            `font-size:11px;line-height:1;padding:2px 5px;border-radius:3px;text-decoration:none">\u2197</a>`
    );
}

// Glue the dims + permalink overlays onto a thumbnail cell.
function _thumbWithOverlays(card, opts) {
    const inner = thumbCellHtml(card, opts);
    return inner.replace(
        /<\/div>$/,
        `${_dimsBadgeHtml(card)}${_permalinkButtonHtml(card)}</div>`,
    );
}

// Wire the permalink-button click handlers inside a freshly rendered container
// so a click on the "open in source" button does not also select the card.
function _wirePermalinkClicks(root) {
    if (!root) return;
    root.querySelectorAll("a.c2c-ph-permalink").forEach((a) => {
        a.addEventListener("click", (e) => e.stopPropagation());
    });
}

// ----------------------------------------------------------- footer / cache badge
// 3-state SFW/NSFW segmented control. Modes:
//   sfw  → drop nsfw + nsfw_unknown
//   both → blur nsfw + nsfw_unknown (default)
//   nsfw → keep only nsfw + nsfw_unknown
function _nsfwSegmentedHtml() {
    const mode = _state.nsfwMode || "both";
    const segCss = (active) => {
        const base = `border:1px solid var(--c2c-border);background:var(--c2c-bg2);color:var(--c2c-fg);` +
                     `padding:3px 9px;font-size:11px;cursor:pointer;line-height:1.4`;
        const on = `background:var(--c2c-mauve);color:var(--c2c-bg);border-color:var(--c2c-mauve)`;
        return base + (active ? `;${on}` : "");
    };
    return (
        `<span class="c2c-nsfw-seg" role="group" aria-label="NSFW filter" ` +
              `title="Filter results by NSFW status" ` +
              `style="display:inline-flex;border-radius:5px;overflow:hidden;line-height:1">` +
            `<button data-mode="sfw"  style="${segCss(mode==='sfw' )};border-radius:5px 0 0 5px">SFW only</button>` +
            `<button data-mode="both" style="${segCss(mode==='both')};border-left:none;border-right:none">Both (blur)</button>` +
            `<button data-mode="nsfw" style="${segCss(mode==='nsfw')};border-radius:0 5px 5px 0">NSFW only</button>` +
        `</span>`
    );
}
function _wireNsfwSegmented(rootEl, modal, info) {
    const seg = rootEl.querySelector(".c2c-nsfw-seg");
    if (!seg) return;
    seg.querySelectorAll("button[data-mode]").forEach((b) => {
        b.onclick = () => {
            const m = b.getAttribute("data-mode");
            if (m !== "sfw" && m !== "both" && m !== "nsfw") return;
            _state.nsfwMode = m;
            _state.showNsfw = (m === "nsfw"); // keep legacy alias in sync
            try { localStorage.setItem("c2c.presetHub.nsfwMode", m); }
            catch (_e) { /* storage unavailable */ }
            renderGrid(modal);
            renderPreview(modal);
            renderFooter(modal, info);
        };
    });
}

function renderFooter(modal, info) {
    const f = modal.querySelector("#ph-footer");
    const nsfwBtn = _nsfwSegmentedHtml();
    const wireNsfwBtn = () => _wireNsfwSegmented(f, modal, info);
    if (_state.tab === "workflows") {
        const scopeLbl = _wf.scope === "local" ? "Local only" : _wf.scope === "online" ? "Online only" : "Local + Online";
        f.innerHTML = `<span>${_esc(_wf.status || scopeLbl)}</span>
          <span style="display:flex;gap:6px">${nsfwBtn}<button id="ph-wf-refresh" style="${btnGhost()};padding:3px 10px">Refresh ↻</button></span>`;
        const rb = f.querySelector("#ph-wf-refresh");
        if (rb) rb.onclick = () => { _wf.scanned = false; runSearch(modal, { force: true }); };
        wireNsfwBtn();
        return;
    }
    if (_state.tab === "promptomania" || _state.tab === "image2prompt") { f.innerHTML = `<span>Local tab — no network fetch.</span><span></span>`; return; }
    const age = info?.age_seconds;
    const ageStr = age == null ? "" : age < 90 ? "just now" : age < 3600 ? `${Math.round(age / 60)}m ago` : `${Math.round(age / 3600)}h ago`;
    const badge = info?.cached ? `${tabLabel()} · fetched ${ageStr}` : `${tabLabel()} · live`;
    f.innerHTML = `<span>${_esc(badge)}</span>
      <span style="display:flex;gap:6px">${nsfwBtn}<button id="ph-refresh" style="${btnGhost()};padding:3px 10px">Refresh ↻</button></span>`;
    const rb = f.querySelector("#ph-refresh");
    if (rb) rb.onclick = () => runSearch(modal, { force: true });
    wireNsfwBtn();
}

// ----------------------------------------------------------- orchestration
async function runSearch(modal, opts = {}) {
    if (_state.tab === "workflows") { await searchWorkflows(modal, opts); return; }
    if (_state.tab === "promptomania" || _state.tab === "image2prompt") { renderGrid(modal); renderFooter(modal); return; }
    const q = buildQueryForSource(_state.tab);
    renderGrid(modal, { loading: true });
    let j;
    try {
        j = opts.force ? await fetchRefresh(_state.tab, q) : await fetchSearch(_state.tab, q);
    } catch (exc) {
        // User-initiated search failed after retries. The inline red banner
        // already informs the user — log at "warn" so it surfaces in DevTools
        // but does NOT add to the registry-failure toast pile.
        __c2cReport("c2c_preset_modal:search", exc, { level: "warn" });
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
    // Bust in-memory online caches so stale records (e.g. from a session
    // where the backend hadn't yet stamped `nsfw_unknown`) cannot bleed
    // back into the freshly-opened modal with the wrong blur state.
    _wf.online = [];
    _state.results = [];
    modal.style.display = "block";
    // Fire-and-forget readiness probe so we have a sane diagnostic if the
    // preset backend isn't mounted (e.g. extension load order race).
    probePresetBackend().then((ok) => {
        if (!ok) {
            __c2cReport(
                "c2c_preset_modal:probe",
                new Error("/c2c/presets/sources unreachable; preset backend not mounted"),
                { level: "info" },
            );
            const grid = modal.querySelector("#ph-grid");
            if (grid && !grid.innerHTML.trim()) {
                grid.innerHTML =
                    `<div style="color:var(--c2c-sub);font-size:12px;padding:8px">` +
                    `Preset backend not loaded yet — restart ComfyUI if this persists.` +
                    `</div>`;
            }
        }
    });
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
