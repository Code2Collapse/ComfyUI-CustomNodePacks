/**
 * c2c_model_browser.js — Model Browser (P2.1)
 * Search Civitai and HuggingFace for models/LoRAs/VAEs, plus a LOCAL
 * source that lists models already on disk (always works, no network).
 * Download directly into ComfyUI model directories via backend.
 * Open: Ctrl+Alt+M or OmniBar.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { attachWindowChrome } from "./_c2c_window.js";

const SEARCH_DEBOUNCE_MS = 400;
// "local" is FIRST so it's the default — it lists on-disk models and never
// needs the network, so opening the browser shows results immediately
// instead of a Civitai/HF "loading error" when those are unreachable.
const SOURCES = ["local", "civitai", "huggingface"];

let _panel  = null;
let _isOpen = false;
let _debounceTimer = null;
let _currentSource = "local";
let _page = 1;
let _slot = null;

/* ─── backend API calls ─── */

/* ─── Civitai filters (L5.03): only sent for the Civitai source ─── */
const _filters = { sort: "", period: "", base: "", tag: "", user: "", nsfw: false, favorites: false };
let _optionsLoaded = false;

function _filterQuery() {
  if (_currentSource !== "civitai") return "";
  const p = new URLSearchParams();
  for (const k of ["sort", "period", "base", "tag", "user"]) if (_filters[k]) p.set(k, _filters[k]);
  if (_filters.nsfw) p.set("nsfw", "1");
  if (_filters.favorites) p.set("favorites", "1");
  const q = p.toString();
  return q ? "&" + q : "";
}

async function _loadCivitaiOptions() {
  if (_optionsLoaded) return;
  try {
    const r = await api.fetchApi("/c2c/models/civitai_options");
    if (!r.ok) return;
    const o = await r.json();
    const fill = (id, values, first) => {
      const sel = document.getElementById(id);
      if (!sel) return;
      sel.innerHTML = `<option value="">${first}</option>` + values.map(v => `<option>${_esc(v)}</option>`).join("");
    };
    fill("c2c-mb-f-sort", o.sorts || [], "Sort: Most Downloaded");
    fill("c2c-mb-f-period", o.periods || [], "Period: all time");
    fill("c2c-mb-f-base", o.base_models || [], "Any base model");
    const fav = document.getElementById("c2c-mb-f-fav");
    if (fav && !o.favorites_available) {
      fav.disabled = true;
      fav.parentElement.title = "Favorites needs your Civitai API key (CIVITAI_API_KEY)";
      fav.parentElement.style.opacity = "0.5";
    }
    _optionsLoaded = true;
  } catch (_) { /* offline: the filters stay at their defaults */ }
}

function _syncFilterBar() {
  const bar = document.getElementById("c2c-mb-filters");
  if (!bar) return;
  const on = _currentSource === "civitai";
  bar.style.display = on ? "flex" : "none";
  if (on) _loadCivitaiOptions();
}

async function _searchModels(source, query, type, page) {
  const resp = await api.fetchApi(`/c2c/models/search?source=${source}&q=${encodeURIComponent(query)}&type=${encodeURIComponent(type)}&page=${page}${_filterQuery()}`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function _downloadModel(source, modelId, fileId, destDir, sha256) {
  const resp = await api.fetchApi("/c2c/models/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, modelId, fileId, destDir, ...(sha256 ? { sha256 } : {}) })
  });
  if (!resp.ok) {
    let msg = await resp.text();
    try { msg = JSON.parse(msg).error || msg; } catch (_) { /* plain text */ }
    throw new Error(msg);
  }
  return resp.json();
}

/* ─── downloads (L5.05): polled only while the browser is open and a job is queued or running ─── */

let _dlTimer = null;

function _fmtBytes(n) {
  if (!n) return "0 MB";
  return n >= 1 << 30 ? `${(n / (1 << 30)).toFixed(2)} GB` : `${(n / (1 << 20)).toFixed(1)} MB`;
}

function _fmtEta(s) {
  if (s == null) return "";
  if (s < 60) return `${s}s left`;
  if (s < 3600) return `${Math.round(s / 60)} min left`;
  return `${(s / 3600).toFixed(1)} h left`;
}

async function _dlAction(id, action) {
  try { await api.fetchApi(`/c2c/models/downloads/${id}/${action}`, { method: "POST" }); } catch (_) { /* next poll shows it */ }
  _pollDownloads();
}

function _renderDownloads(jobs) {
  const box = document.getElementById("c2c-mb-downloads");
  if (!box) return;
  box.style.display = jobs.length ? "block" : "none";
  box.innerHTML = "";
  const head = document.createElement("div");
  head.style.cssText = "display:flex;align-items:center;gap:8px;font-size:12px;font-weight:600;margin-bottom:4px;";
  head.textContent = `Downloads (${jobs.length})`;
  if (jobs.some(j => ["done", "error", "cancelled"].includes(j.status))) {
    const clr = document.createElement("button");
    clr.textContent = "Clear finished";
    clr.style.cssText = "margin-left:auto;font-size:11px;padding:1px 8px;background:rgba(255,255,255,.07);" +
      "border:1px solid rgba(255,255,255,.12);border-radius:4px;cursor:pointer;color:inherit;";
    clr.addEventListener("click", () => _dlAction("all", "clear"));
    head.appendChild(clr);
  }
  box.appendChild(head);
  for (const j of jobs) {
    const row = document.createElement("div");
    row.dataset.job = j.id;
    row.style.cssText = "display:grid;grid-template-columns:1fr auto;gap:2px 10px;padding:5px 0;" +
      "border-top:1px solid rgba(255,255,255,.06);font-size:11px;";
    const name = document.createElement("div");
    name.textContent = j.label;
    name.title = j.dest;
    name.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const acts = document.createElement("div");
    acts.style.cssText = "display:flex;gap:4px;justify-content:flex-end;";
    const btn = (label, action) => {
      const b = document.createElement("button");
      b.textContent = label; b.dataset.action = action;
      b.style.cssText = "font-size:10px;padding:0 7px;background:rgba(255,255,255,.07);" +
        "border:1px solid rgba(255,255,255,.12);border-radius:4px;cursor:pointer;color:inherit;";
      b.addEventListener("click", () => _dlAction(j.id, action));
      acts.appendChild(b);
    };
    if (j.status === "running" || j.status === "queued") { btn("Pause", "pause"); btn("Cancel", "cancel"); }
    else if (j.status === "paused") { btn("Resume", "resume"); btn("Cancel", "cancel"); }
    else if (j.status === "error") { btn("Retry", "resume"); }
    const bar = document.createElement("div");
    bar.style.cssText = "grid-column:1 / -1;height:4px;border-radius:2px;background:rgba(255,255,255,.08);overflow:hidden;";
    const fill = document.createElement("div");
    const pct = j.progress != null ? Math.round(j.progress * 100) : (j.status === "done" ? 100 : 0);
    const tone = j.status === "error" ? "var(--c2c-dangerSoft)" : j.status === "done" ? "var(--c2c-okSoft2)"
      : "var(--c2c-accentLink, #89b4fa)";
    fill.style.cssText = `height:100%;width:${pct}%;background:${tone};`;
    bar.appendChild(fill);
    const info = document.createElement("div");
    info.style.cssText = "grid-column:1 / -1;color:var(--c2c-gray450);";
    const size = j.total ? `${_fmtBytes(j.done)} / ${_fmtBytes(j.total)}` : _fmtBytes(j.done);
    const parts = {
      queued: ["Queued"],
      running: [size, j.speed ? `${_fmtBytes(j.speed)}/s` : "", _fmtEta(j.eta_s)],
      paused: ["Paused", size],
      done: ["Done", _fmtBytes(j.total), j.verified ? "SHA-256 verified" : "not verified (no published hash)"],
      cancelled: ["Cancelled"],
      error: [j.error || "Failed"],
    }[j.status] || [j.status];
    info.textContent = parts.filter(Boolean).join("  ·  ");
    if (j.status === "error") info.style.color = "var(--c2c-dangerSoft)";
    row.append(name, acts, bar, info);
    box.appendChild(row);
  }
}

async function _pollDownloads() {
  if (_dlTimer) { clearTimeout(_dlTimer); _dlTimer = null; }
  if (!_isOpen) return;
  let jobs = [];
  try {
    const r = await api.fetchApi("/c2c/models/downloads");
    if (r.ok) jobs = (await r.json()).jobs || [];
  } catch (_) { /* server busy: try again on the next tick */ }
  _renderDownloads(jobs);
  if (_isOpen && jobs.some(j => j.status === "queued" || j.status === "running")) {
    _dlTimer = setTimeout(_pollDownloads, 1000);
  }
}

async function _listDestDirs() {
  const resp = await api.fetchApi("/c2c/models/dest_dirs");
  if (!resp.ok) return { dirs: ["models/checkpoints","models/loras","models/vae","models/controlnet","models/embeddings"] };
  return resp.json();
}

/* ─── panel ─── */

function _buildPanel() {
  if (_panel) return _panel;

  const overlay = document.createElement("div");
  overlay.id = "c2c-model-browser-overlay";
  overlay.style.cssText = `
    position: fixed; inset: 0;
    background: rgba(0,0,0,.7);
    z-index: var(--c2c-z-modal);
    display: flex;
    align-items: center;
    justify-content: center;
  `;
  overlay.addEventListener("click", (e) => { if (e.target === overlay) _close(); });

  const panel = document.createElement("div");
  panel.id = "c2c-model-browser";
  panel.style.cssText = `
    background: var(--c2c-surface, var(--c2c-neutral950));
    border: 1px solid var(--c2c-border, rgba(255,255,255,.15));
    border-radius: 10px;
    width: 780px;
    max-height: 85vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 16px 48px rgba(0,0,0,.7);
    color: var(--c2c-fg, var(--c2c-gray100));
    font-family: var(--c2c-font, system-ui, sans-serif);
    font-size: 13px;
  `;

  panel.innerHTML = `
    <div style="padding:14px 16px;border-bottom:1px solid rgba(255,255,255,.1);
                display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
      <span style="font-weight:600;font-size:14px;">🔎 Model Browser</span>
      <div style="display:flex;gap:4px;margin-left:4px;" id="c2c-mb-source-tabs"></div>
      <select id="c2c-mb-type" style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);
                border-radius:5px;padding:3px 6px;color:inherit;font-size:12px;">
        <option value="Checkpoint">Checkpoint</option>
        <option value="LORA">LoRA</option>
        <option value="VAE">VAE</option>
        <option value="ControlNet">ControlNet</option>
        <option value="TextualInversion">Embedding</option>
        <option value="Upscaler">Upscaler</option>
        <option value="">All</option>
      </select>
      <input id="c2c-mb-search" type="text" placeholder="Search models…"
        style="flex:1;min-width:160px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);
               border-radius:5px;padding:5px 10px;color:inherit;font-size:13px;">
      <button id="c2c-mb-close"
        style="background:none;border:none;font-size:18px;cursor:pointer;color:var(--c2c-gray300);padding:0 4px;">×</button>
    </div>
    <div id="c2c-mb-filters" style="display:none;align-items:center;gap:6px;flex-wrap:wrap;padding:8px 16px;
                border-bottom:1px solid rgba(255,255,255,.08);font-size:11px;">
      <select id="c2c-mb-f-sort" data-f="sort" style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:5px;padding:3px 6px;color:inherit;font-size:11px;"><option value="">Sort: Most Downloaded</option></select>
      <select id="c2c-mb-f-period" data-f="period" style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:5px;padding:3px 6px;color:inherit;font-size:11px;"><option value="">Period: all time</option></select>
      <select id="c2c-mb-f-base" data-f="base" style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:5px;padding:3px 6px;color:inherit;font-size:11px;max-width:150px;"><option value="">Any base model</option></select>
      <input id="c2c-mb-f-tag" data-f="tag" placeholder="Tag" style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:5px;padding:3px 6px;color:inherit;font-size:11px;width:90px;">
      <input id="c2c-mb-f-user" data-f="user" placeholder="Creator" style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:5px;padding:3px 6px;color:inherit;font-size:11px;width:100px;">
      <label style="display:flex;align-items:center;gap:3px;cursor:pointer;"><input id="c2c-mb-f-nsfw" data-f="nsfw" type="checkbox">NSFW</label>
      <label style="display:flex;align-items:center;gap:3px;cursor:pointer;"><input id="c2c-mb-f-fav" data-f="favorites" type="checkbox">Favorites</label>
    </div>
    <div id="c2c-mb-results" style="overflow-y:auto;flex:1;padding:10px 14px;"></div>
    <div id="c2c-mb-downloads" style="display:none;max-height:190px;overflow-y:auto;padding:8px 16px;
                border-top:1px solid rgba(255,255,255,.08);"></div>
    <div style="display:flex;align-items:center;gap:8px;padding:8px 16px;
                border-top:1px solid rgba(255,255,255,.08);">
      <button id="c2c-mb-prev"
        style="padding:4px 10px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);
               border-radius:5px;cursor:pointer;color:var(--c2c-gray300);font-size:12px;">◀ Prev</button>
      <span id="c2c-mb-page-info" style="font-size:12px;color:var(--c2c-gray450);flex:1;text-align:center;">Page 1</span>
      <button id="c2c-mb-next"
        style="padding:4px 10px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);
               border-radius:5px;cursor:pointer;color:var(--c2c-gray300);font-size:12px;">Next ▶</button>
    </div>
  `;

  overlay.appendChild(panel);
  document.body.appendChild(overlay);
  _panel = overlay;

  // source tabs
  for (const el of panel.querySelectorAll("#c2c-mb-filters [data-f]")) {
    const ev = el.tagName === "INPUT" && el.type === "text" ? "input" : "change";
    el.addEventListener(ev, () => {
      _filters[el.dataset.f] = el.type === "checkbox" ? el.checked : el.value.trim();
      _page = 1;
      clearTimeout(_debounceTimer);
      _debounceTimer = setTimeout(_doSearch, ev === "input" ? SEARCH_DEBOUNCE_MS : 0);
    });
  }
  const tabsEl = panel.querySelector("#c2c-mb-source-tabs");
  const _labels = { local: "Local", civitai: "Civitai", huggingface: "HuggingFace" };
  for (const src of SOURCES) {
    const tab = document.createElement("button");
    tab.textContent = _labels[src] || src;
    tab.dataset.src = src;
    tab.style.cssText = `
      padding:3px 10px;border-radius:5px;cursor:pointer;font-size:12px;
      background:${src === _currentSource ? "var(--c2c-blueBg)" : "var(--c2c-overlay-hover)"};
      border:1px solid ${src === _currentSource ? "var(--c2c-blue)" : "var(--c2c-border)"};
      color:${src === _currentSource ? "var(--c2c-blue)" : "var(--c2c-gray300)"};
    `;
    tab.addEventListener("click", () => {
      _currentSource = src;
      _page = 1;
      _updateTabStyles();
      _syncFilterBar();
      _doSearch();
    });
    tabsEl.appendChild(tab);
  }

  panel.querySelector("#c2c-mb-close").addEventListener("click", _close);
  panel.querySelector("#c2c-mb-search").addEventListener("input", () => {
    clearTimeout(_debounceTimer);
    _debounceTimer = setTimeout(() => { _page = 1; _doSearch(); }, SEARCH_DEBOUNCE_MS);
  });
  panel.querySelector("#c2c-mb-type").addEventListener("change", () => { _page = 1; _doSearch(); });
  panel.querySelector("#c2c-mb-prev").addEventListener("click", () => {
    if (_page > 1) { _page--; _doSearch(); }
  });
  panel.querySelector("#c2c-mb-next").addEventListener("click", () => {
    _page++;
    _doSearch();
  });

  // Window chrome: drag / 8-edge resize / minimize / shortcut hint.
  const header  = panel.firstElementChild;
  const titleEl = header?.querySelector("span");
  attachWindowChrome(panel, {
    storageKey: "model_browser",
    overlay,
    header,
    titleEl,
    shortcut: "Ctrl+Alt+M",
    minW: 480, minH: 320,
  });

  return overlay;
}

function _updateTabStyles() {
  const tabs = document.querySelectorAll("#c2c-mb-source-tabs [data-src]");
  for (const tab of tabs) {
    const active = tab.dataset.src === _currentSource;
    tab.style.background = active ? "var(--c2c-blueBg)" : "var(--c2c-overlay-hover)";
    tab.style.border      = `1px solid ${active ? "var(--c2c-blue)" : "var(--c2c-border)"}`;
    tab.style.color       = active ? "var(--c2c-blue)" : "var(--c2c-gray300)";
  }
}

async function _doSearch() {
  const query = document.querySelector("#c2c-mb-search")?.value?.trim() || "";
  const type  = document.querySelector("#c2c-mb-type")?.value || "";
  const results = document.getElementById("c2c-mb-results");
  const pageInfo = document.getElementById("c2c-mb-page-info");
  if (!results) return;

  results.innerHTML = `<div style="text-align:center;padding:30px;color:var(--c2c-gray500);">Searching…</div>`;

  try {
    const data = await _searchModels(_currentSource, query, type, _page);
    pageInfo && (pageInfo.textContent = `Page ${_page}`);

    const items = data.items || data.results || [];
    if (items.length === 0) {
      results.innerHTML = `<div style="text-align:center;padding:30px;color:var(--c2c-gray500);">No results</div>`;
      return;
    }

    const destData = await _listDestDirs();
    const destDirs = destData.dirs || [];

    results.innerHTML = "";
    for (const item of items) {
      results.appendChild(_buildCard(item, destDirs));
    }
  } catch (err) {
    // The backend replies {"error": "<plain-English reason>"} — surface THAT,
    // not raw JSON braces or a blank line (a network fail has no message).
    let msg = err?.message || "";
    try { const j = JSON.parse(msg); if (j && j.error) msg = j.error; } catch (_) {}
    if (!msg.trim()) msg = "Could not reach the ComfyUI server (network error).";
    const esc = msg.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    results.innerHTML = `<div style="text-align:center;padding:30px 40px;color:var(--c2c-dangerSoft);line-height:1.55;">
      ${esc}<br><small style="color:var(--c2c-gray500);">Details are in the ComfyUI console.</small></div>`;
  }
}

/* ─── Civitai detail view (L5.04). The description arrives SANITISED by the server (nodes/civitai_detail.py
       allow-list: no scripts, attributes or non-http links); everything else is built with DOM APIs. ─── */

function _el(tag, css, text) {
  const e = document.createElement(tag);
  if (css) e.style.cssText = css;
  if (text != null) e.textContent = text;
  return e;
}

const _BTN = "font-size:11px;padding:2px 9px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);" +
  "border-radius:4px;cursor:pointer;color:inherit;";

async function _copy(text, btn) {
  try { await navigator.clipboard.writeText(text); btn.textContent = "Copied"; }
  catch (_) { btn.textContent = "Copy failed"; }
  setTimeout(() => { btn.textContent = "Copy"; }, 1200);
}

async function _getJson(url) {
  const r = await api.fetchApi(url);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

function _renderVersion(box, model, v, destDirs) {
  box.innerHTML = "";
  const meta = _el("div", "display:flex;gap:6px;flex-wrap:wrap;align-items:center;font-size:11px;margin:6px 0;");
  if (v.base_model) meta.append(_el("span", "background:rgba(255,255,255,.08);padding:1px 6px;border-radius:3px;", v.base_model));
  if (v.downloads) meta.append(_el("span", "color:var(--c2c-gray450);", `⬇ ${_fmtNum(v.downloads)}`));
  if (v.early_access) {
    const b = _el("span", "background:var(--c2c-warnBg, rgba(250,179,135,.15));color:var(--c2c-warnSoft, #fab387);" +
      "padding:1px 6px;border-radius:3px;font-weight:600;", v.early_access);
    b.title = "Downloads open when early access ends (on civitai.com you can buy access). The browser never works around it.";
    meta.append(b);
  }
  box.append(meta);
  if (v.trained_words?.length) {
    const tw = _el("div", "display:flex;gap:5px;flex-wrap:wrap;align-items:center;margin:4px 0 8px;font-size:11px;");
    tw.append(_el("span", "color:var(--c2c-gray450);", "Trigger words:"));
    for (const w of v.trained_words) {
      const chip = _el("span", "background:rgba(255,255,255,.07);padding:1px 6px;border-radius:3px;font-family:monospace;", w);
      tw.append(chip);
    }
    const all = _el("button", _BTN, "Copy");
    all.dataset.copyWords = "1";
    all.title = "Copy all trigger words";
    all.addEventListener("click", () => _copy(v.trained_words.join(", "), all));
    tw.append(all);
    box.append(tw);
  }
  // files
  const table = _el("div", "display:grid;grid-template-columns:1fr auto auto auto auto;gap:4px 10px;align-items:center;" +
    "font-size:11px;margin:6px 0;");
  for (const h of ["File", "Size", "Format", "Hash", ""]) table.append(_el("span", "color:var(--c2c-gray450);font-weight:600;", h));
  const dest = _el("select", "background:var(--c2c-overlay-hover);border:1px solid var(--c2c-border);border-radius:4px;" +
    "padding:2px 5px;color:inherit;font-size:10px;max-width:160px;");
  const def = _defaultDest(model.type, destDirs);
  for (const d of destDirs) { const o = _el("option", null, d); if (d === def) o.selected = true; dest.append(o); }
  for (const f of v.files || []) {
    const name = _el("span", "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", (f.primary ? "★ " : "") + f.name);
    name.title = [f.name, f.type, f.scan?.pickle && `pickle scan: ${f.scan.pickle}`, f.scan?.virus && `virus scan: ${f.scan.virus}`]
      .filter(Boolean).join("\n");
    const size = _el("span", null, f.size_kb ? _fmtBytes(f.size_kb * 1024) : "");
    const fmt = _el("span", null, [f.format, f.precision, f.pruned].filter(Boolean).join(" · "));
    const hash = _el("span", "font-family:monospace;", f.sha256 ? f.sha256.slice(0, 10) + "…" : "—");
    if (f.sha256) hash.title = `SHA-256 ${f.sha256}`;
    const dl = _el("button", _BTN + "background:var(--c2c-okBg2);border-color:var(--c2c-okMute);color:var(--c2c-okSoft2);", "⬇");
    dl.dataset.fileId = String(f.id);
    if (!f.downloadable) {
      dl.disabled = true; dl.style.opacity = "0.4"; dl.style.cursor = "not-allowed";
      dl.title = v.early_access ? `${v.early_access}: not downloadable yet` : "Not downloadable";
    } else {
      dl.title = `Download into the folder chosen below${f.sha256 ? " (SHA-256 checked)" : ""}`;
      dl.addEventListener("click", async () => {
        dl.disabled = true; dl.textContent = "⏳";
        try {
          await _downloadModel("civitai", String(model.id), String(f.id), dest.value, f.sha256 || "");
          dl.textContent = "✓";
          _pollDownloads();
        } catch (err) {
          dl.textContent = "✗"; dl.title = err.message; dl.disabled = false;
        }
      });
    }
    table.append(name, size, fmt, hash, dl);
  }
  box.append(table);
  const destRow = _el("div", "display:flex;gap:6px;align-items:center;font-size:11px;margin-bottom:8px;");
  destRow.append(_el("span", "color:var(--c2c-gray450);", "Save into:"), dest);
  box.append(destRow);
  if (v.images?.length) {
    const grid = _el("div", "display:flex;gap:6px;flex-wrap:wrap;margin:6px 0;");
    for (const im of v.images) {
      const a = _el("a");
      a.href = im.url; a.target = "_blank"; a.rel = "noopener noreferrer";
      const img = _el("img", "width:110px;height:110px;object-fit:cover;border-radius:4px;background:var(--c2c-neutral920);");
      img.loading = "lazy"; img.src = im.url; img.alt = "example image";
      a.append(img); grid.append(a);
    }
    box.append(grid);
  }
  if (v.description) {
    const d = _el("div", "font-size:12px;line-height:1.5;color:var(--c2c-gray300);margin-top:6px;");
    d.innerHTML = v.description;                       // sanitised server-side (allow-list)
    box.append(d);
  }
}

async function _openDetail(modelId) {
  const results = document.getElementById("c2c-mb-results");
  if (!results) return;
  const scroll = results.scrollTop;
  const saved = [...results.childNodes];
  const restore = () => { results.replaceChildren(...saved); results.scrollTop = scroll; };
  results.innerHTML = `<div style="text-align:center;padding:30px;color:var(--c2c-gray500);">Loading model…</div>`;
  const nsfw = _filters.nsfw ? "?nsfw=1" : "";
  let model, destDirs;
  try {
    [model, destDirs] = await Promise.all([_getJson(`/c2c/models/civitai/model/${modelId}${nsfw}`),
                                           _listDestDirs().then(d => d.dirs || [])]);
  } catch (err) {
    results.innerHTML = "";
    const back = _el("button", _BTN, "◀ Back");
    back.addEventListener("click", restore);
    results.append(back, _el("div", "color:var(--c2c-dangerSoft);padding:20px;", err.message));
    return;
  }
  results.innerHTML = "";
  const wrap = _el("div", "padding:2px 2px 10px;");
  wrap.id = "c2c-mb-detail";
  const top = _el("div", "display:flex;gap:8px;align-items:center;flex-wrap:wrap;");
  const back = _el("button", _BTN, "◀ Back");
  back.addEventListener("click", restore);
  top.append(back, _el("span", "font-weight:600;font-size:14px;", model.name));
  if (model.type) top.append(_el("span", "font-size:11px;background:rgba(255,255,255,.08);padding:1px 6px;border-radius:3px;", model.type));
  const sub = [model.creator && `by ${model.creator}`, model.stats?.downloadCount && `⬇ ${_fmtNum(model.stats.downloadCount)}`,
               model.stats?.rating && `⭐ ${Number(model.stats.rating).toFixed(1)}`].filter(Boolean).join("  ·  ");
  wrap.append(top, _el("div", "font-size:11px;color:var(--c2c-gray450);margin:4px 0;", sub));
  if (model.tags?.length) wrap.append(_el("div", "font-size:10px;color:var(--c2c-gray500);margin-bottom:6px;", model.tags.join(" · ")));
  const vsel = _el("select", "background:var(--c2c-overlay-hover);border:1px solid var(--c2c-border);border-radius:4px;" +
    "padding:2px 6px;color:inherit;font-size:11px;");
  vsel.id = "c2c-mb-version";
  for (const v of model.versions || []) {
    const o = _el("option", null, `${v.name}${v.base_model ? " · " + v.base_model : ""}${v.early_access ? " · early access" : ""}`);
    o.value = String(v.id); vsel.append(o);
  }
  const vrow = _el("div", "display:flex;gap:6px;align-items:center;font-size:11px;");
  vrow.append(_el("span", "color:var(--c2c-gray450);", "Version:"), vsel);
  const vbox = _el("div");
  wrap.append(vrow, vbox);
  if (model.first_version) _renderVersion(vbox, model, model.first_version, destDirs);
  vsel.addEventListener("change", async () => {
    vbox.innerHTML = `<div style="color:var(--c2c-gray500);padding:10px;">Loading version…</div>`;
    try { _renderVersion(vbox, model, await _getJson(`/c2c/models/civitai/version/${vsel.value}${nsfw}`), destDirs); }
    catch (err) { vbox.replaceChildren(_el("div", "color:var(--c2c-dangerSoft);padding:10px;", err.message)); }
  });
  if (model.description) {
    const d = _el("div", "font-size:12px;line-height:1.5;color:var(--c2c-gray300);margin-top:10px;border-top:1px solid " +
      "rgba(255,255,255,.08);padding-top:8px;");
    d.innerHTML = model.description;                   // sanitised server-side (allow-list)
    wrap.append(d);
  }
  results.append(wrap);
  results.scrollTop = 0;
}

// L5.03: the folder a model of this type belongs in (a LoRA went to the alphabetically first folder, "datasets").
const _TYPE_DIRS = { checkpoint: "checkpoints", lora: "loras", locon: "loras", dora: "loras", vae: "vae",
  controlnet: "controlnet", textualinversion: "embeddings", upscaler: "upscale_models", motionmodule: "animatediff_models" };
function _defaultDest(type, destDirs) {
  const want = _TYPE_DIRS[String(type || "").toLowerCase()] || _TYPE_DIRS[String(document.querySelector("#c2c-mb-type")?.value || "").toLowerCase()];
  if (!want) return destDirs.find(d => /(^|\/)checkpoints$/.test(d)) || destDirs[0];
  return destDirs.find(d => d === `models/${want}`) || destDirs.find(d => d.split("/").pop() === want) || destDirs[0];
}

function _buildCard(item, destDirs) {
  const card = document.createElement("div");
  card.style.cssText = `
    display: flex;
    gap: 10px;
    padding: 10px;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 7px;
    margin-bottom: 7px;
    background: rgba(255,255,255,.03);
  `;

  const thumb = item.modelVersions?.[0]?.images?.[0]?.url || item.image || "";
  const name  = item.name || item.id || "Model";
  const type  = item.type || item.model_type || "";
  const downloads = item.stats?.downloadCount || item.downloads || 0;
  const rating    = item.stats?.rating?.toFixed(1) || "";
  const files     = item.modelVersions?.[0]?.files || item.siblings || [];
  const modelId   = item.id || item.modelId || "";
  const versionId = item.modelVersions?.[0]?.id || "";

  card.innerHTML = `
    ${thumb ? `<img src="${thumb}" style="width:70px;height:70px;object-fit:cover;border-radius:4px;flex-shrink:0;background:var(--c2c-neutral990);">` : `<div style="width:70px;height:70px;background:var(--c2c-neutral920);border-radius:4px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:24px;">📦</div>`}
    <div style="flex:1;min-width:0;">
      <div style="font-weight:600;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
        ${item._source === "civitai" || (_currentSource === "civitai" && item._source !== "local")
          ? `<a href="#" data-detail style="color:inherit;text-decoration:underline dotted;" title="Versions, files, trigger words, images">${_esc(name)}</a>`
          : _esc(name)}</div>
      <div style="font-size:11px;color:var(--c2c-gray400);margin-top:2px;">
        ${type ? `<span style="background:rgba(255,255,255,.08);padding:1px 5px;border-radius:3px;">${_esc(type)}</span> ` : ""}
        ${downloads ? `⬇ ${_fmtNum(downloads)} ` : ""}
        ${rating ? `⭐ ${rating}` : ""}
      </div>
      <div style="font-size:11px;color:var(--c2c-gray500);margin-top:3px;">
        ${item.description ? _esc(item.description.replace(/<[^>]+>/g,"")).slice(0,120) + "…" : ""}
      </div>
      <div style="display:flex;gap:6px;margin-top:7px;flex-wrap:wrap;" data-model-id="${modelId}" data-version-id="${versionId}">
        ${files.slice(0,3).map((f, i) => {
          const fname = f.name || f.rfilename || `file-${i}`;
          const fsize = f.sizeKB ? `${(f.sizeKB/1024).toFixed(1)}GB` : f.size ? `${(f.size/1e9).toFixed(1)}GB` : "";
          return `<span style="font-size:10px;background:rgba(255,255,255,.06);padding:2px 7px;border-radius:3px;white-space:nowrap;cursor:default;" title="${_esc(fname)}">
            ${_esc(fname.slice(0,22))} ${fsize ? `<span style="color:var(--c2c-gray500)">(${fsize})</span>` : ""}
          </span>`;
        }).join("")}
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:5px;align-items:flex-end;flex-shrink:0;">
      ${item._source === "local"
        ? `<span style="font-size:11px;background:var(--c2c-okBg);border:1px solid var(--c2c-okMute);
                padding:4px 10px;border-radius:5px;color:var(--c2c-okSoft2);font-weight:600;">✓ On disk</span>`
        : `<select data-dest style="background:var(--c2c-overlay-hover);border:1px solid var(--c2c-border);
              border-radius:4px;padding:2px 5px;color:inherit;font-size:10px;max-width:140px;">
          ${destDirs.map(d => `<option${d === _defaultDest(type, destDirs) ? " selected" : ""}>${_esc(d)}</option>`).join("")}
        </select>
        <button data-dl
          style="padding:4px 12px;background:var(--c2c-okBg2);border:1px solid var(--c2c-okMute);
                 border-radius:5px;cursor:pointer;color:var(--c2c-okSoft2);font-size:12px;font-weight:600;">
          ⬇ Download
        </button>`
      }
    </div>
  `;

  card.querySelector("[data-detail]")?.addEventListener("click", (e) => { e.preventDefault(); _openDetail(modelId); });
  if (item._source !== "local") {
    card.querySelector("[data-dl]").addEventListener("click", async (e) => {
      const btn     = e.currentTarget;
      const destSel = card.querySelector("[data-dest]");
      const destDir = destSel?.value || "models/checkpoints";
      // L5.05: prefer a .safetensors file (no pickled code) when the version lists several
      const pick = files.find(f => /\.safetensors$/i.test(f.name || f.rfilename || "")) || files[0];
      const fileId = pick?.id || pick?.rfilename || "";
      const sha256 = pick?.hashes?.SHA256 || pick?.hashes?.sha256 || "";
      btn.textContent = "⏳";
      btn.disabled = true;
      try {
        const res = await _downloadModel(_currentSource, String(modelId), String(fileId), destDir, sha256);
        btn.textContent = "✓ Queued";
        btn.title = res.verify ? "Will be checked against the published SHA-256" : "No published hash: not verified";
        btn.style.background = "var(--c2c-okBg)";
        _pollDownloads();
      } catch (err) {
        btn.textContent = "✗ Error";
        btn.style.background = "var(--c2c-dangerBg)";
        btn.title = err.message;
        setTimeout(() => { btn.textContent = "⬇ Download"; btn.disabled = false; btn.style.background = ""; }, 2000);
      }
    });
  }

  return card;
}

function _esc(s) {
  return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function _fmtNum(n) {
  if (n >= 1e6) return (n/1e6).toFixed(1)+"M";
  if (n >= 1e3) return (n/1e3).toFixed(0)+"K";
  return String(n);
}

/* ─── open / close ─── */

function _open() {
  _buildPanel();
  _panel.style.display = "flex";
  _isOpen = true;
  _syncFilterBar();
  document.querySelector("#c2c-mb-search")?.focus();
  if (!document.querySelector("#c2c-mb-results")?.children.length) _doSearch();
  _pollDownloads();
}

function _close() {
  if (_panel) _panel.style.display = "none";
  _isOpen = false;
  if (_dlTimer) { clearTimeout(_dlTimer); _dlTimer = null; }   // downloads keep running on the server
}

/* ─── keyboard ─── */

document.addEventListener("keydown", (e) => {
  // Ctrl+Alt+M (was Ctrl+Shift+M — moved so it no longer collides with any
  // core/custom Ctrl+M binding). Alt+mouseselect-style collisions are also
  // avoided because Alt alone never opens this.
  if (e.ctrlKey && e.altKey && e.key === "M") {
    e.preventDefault();
    _isOpen ? _close() : _open();
  }
});

/* ─── OmniBar slot ─── */

function _buildSlot() {
  const btn = document.createElement("button");
  btn.id        = "c2c-model-browser-btn";
  btn.className = "c2c-omnibar-slot-pill";
  btn.textContent = "📦 Models";
  btn.title     = "Model Browser (Ctrl+Alt+M)";
  btn.style.cssText = `
    font-size: 11px;
    padding: 2px 7px;
    cursor: pointer;
    background: var(--c2c-pill-bg, rgba(255,255,255,.07));
    color: var(--c2c-fg, var(--c2c-gray200));
    border: 1px solid var(--c2c-border, rgba(255,255,255,.12));
    border-radius: 10px;
  `;
  btn.addEventListener("click", () => _isOpen ? _close() : _open());
  _slot = btn;
  return btn;
}

/* ─── extension ─── */

app.registerExtension({
  name: "C2C.ModelBrowser",

  async setup() {
    const tryRegister = () => {
      if (!window.C2COmniBar) return setTimeout(tryRegister, 200);
      window.C2COmniBar.register({
        section : "tools",
        id      : "c2c-model-browser",
        order   : 20,
        element : _buildSlot(),
      });
    };
    tryRegister();

    window.C2CModelBrowser = { open: _open, close: _close };
  }
});
