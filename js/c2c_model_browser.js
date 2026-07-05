/**
 * c2c_model_browser.js — Model Browser (P2.1)
 * Search Civitai and HuggingFace for models/LoRAs/VAEs.
 * Download directly into ComfyUI model directories via backend.
 * Open: Ctrl+Shift+M or OmniBar.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { attachWindowChrome } from "./_c2c_window.js";

const SEARCH_DEBOUNCE_MS = 400;
const SOURCES = ["civitai", "huggingface"];

let _panel  = null;
let _isOpen = false;
let _debounceTimer = null;
let _currentSource = "civitai";
let _page = 1;
let _slot = null;

/* ─── backend API calls ─── */

async function _searchModels(source, query, type, page) {
  const resp = await api.fetchApi(`/c2c/models/search?source=${source}&q=${encodeURIComponent(query)}&type=${encodeURIComponent(type)}&page=${page}`);
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function _downloadModel(source, modelId, fileId, destDir) {
  const resp = await api.fetchApi("/c2c/models/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, modelId, fileId, destDir })
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
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
    <div id="c2c-mb-results" style="overflow-y:auto;flex:1;padding:10px 14px;"></div>
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
  const tabsEl = panel.querySelector("#c2c-mb-source-tabs");
  for (const src of SOURCES) {
    const tab = document.createElement("button");
    tab.textContent = src === "civitai" ? "Civitai" : "HuggingFace";
    tab.dataset.src = src;
    tab.style.cssText = `
      padding:3px 10px;border-radius:5px;cursor:pointer;font-size:12px;
      background:${src === _currentSource ? "rgba(100,160,255,.2)" : "rgba(255,255,255,.07)"};
      border:1px solid ${src === _currentSource ? "rgba(100,160,255,.4)" : "rgba(255,255,255,.12)"};
      color:${src === _currentSource ? "var(--c2c-blue)" : "var(--c2c-gray300)"};
    `;
    tab.addEventListener("click", () => {
      _currentSource = src;
      _page = 1;
      _updateTabStyles();
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
    shortcut: "Ctrl+Shift+M",
    minW: 480, minH: 320,
  });

  return overlay;
}

function _updateTabStyles() {
  const tabs = document.querySelectorAll("#c2c-mb-source-tabs [data-src]");
  for (const tab of tabs) {
    const active = tab.dataset.src === _currentSource;
    tab.style.background = active ? "rgba(100,160,255,.2)" : "rgba(255,255,255,.07)";
    tab.style.border      = `1px solid ${active ? "rgba(100,160,255,.4)" : "rgba(255,255,255,.12)"}`;
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
        ${_esc(name)}</div>
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
      <select data-dest style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);
              border-radius:4px;padding:2px 5px;color:inherit;font-size:10px;max-width:140px;">
        ${destDirs.map(d => `<option>${_esc(d)}</option>`).join("")}
      </select>
      <button data-dl
        style="padding:4px 12px;background:rgba(0,200,100,.18);border:1px solid rgba(0,200,100,.3);
               border-radius:5px;cursor:pointer;color:var(--c2c-okSoft2);font-size:12px;font-weight:600;">
        ⬇ Download
      </button>
    </div>
  `;

  card.querySelector("[data-dl]").addEventListener("click", async (e) => {
    const btn     = e.currentTarget;
    const destSel = card.querySelector("[data-dest]");
    const destDir = destSel?.value || "models/checkpoints";
    const firstFile = files[0];
    const fileId = firstFile?.id || firstFile?.rfilename || "";
    btn.textContent = "⏳";
    btn.disabled = true;
    try {
      const res = await _downloadModel(_currentSource, String(modelId), String(fileId), destDir);
      btn.textContent = "✓ Queued";
      btn.style.background = "rgba(0,255,0,.1)";
    } catch (err) {
      btn.textContent = "✗ Error";
      btn.style.background = "rgba(255,0,0,.1)";
      btn.title = err.message;
      setTimeout(() => { btn.textContent = "⬇ Download"; btn.disabled = false; btn.style.background = ""; }, 2000);
    }
  });

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
  document.querySelector("#c2c-mb-search")?.focus();
  if (!document.querySelector("#c2c-mb-results")?.children.length) _doSearch();
}

function _close() {
  if (_panel) _panel.style.display = "none";
  _isOpen = false;
}

/* ─── keyboard ─── */

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.shiftKey && e.key === "M") {
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
  btn.title     = "Model Browser (Ctrl+Shift+M)";
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
