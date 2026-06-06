/**
 * c2c_workflow_library.js - Workflow Library panel (C2C).
 *
 * A library browser for your saved ComfyUI workflow .json files, modelled on
 * gregowahoo/comfyui-workflow-finder (MIT). Where the in-graph finder
 * (c2c_workflow_find.js, Ctrl+F) searches the CURRENT graph, this panel
 * searches a LIBRARY of workflow files on disk:
 *
 *   • Scan locations  — add/enable/disable folders; persisted server-side via
 *                       /c2c/library/locations.
 *   • Semantic search — ranks workflows by the *capabilities* of the nodes
 *                       they contain (js/c2c_node_taxonomy.js), so a plain
 *                       query like "make a video from an image" matches.
 *   • Name filter     — instant filename substring match.
 *   • Package filter  — show only workflows whose required custom-node packages
 *                       are all enabled (or "core only").
 *   • Open / Copy     — load a workflow into the canvas, or copy its path.
 *
 * Backend: nodes/c2c_workflow_library.py. Scoring is done here so the node
 * capability map (taxonomy) has a single source of truth.
 *
 * License: Apache-2.0
 */
import { app } from "../../scripts/app.js";
import { attachWindowChrome } from "./_c2c_window.js";
import { capabilityFor, nodeColor } from "./c2c_node_taxonomy.js";
import { renderGraphPreview, legendHTML } from "./c2c_graph_preview.js";
import { c2cPrompt } from "./_c2c_dialog.js";

const PANEL_ID = "c2c-library-panel";
const STYLE_ID = "c2c-library-style";
const BTN_ID   = "c2c-library-btn";
const LS_TAB   = "c2c.library.lastSort";

// ── Suggested searches (compact; categories mirror gregowahoo) ────────
const SUGGESTIONS = [
    "generate video from an image",
    "text to image with Flux",
    "inpaint to fill or replace an area",
    "face swap with identity preservation",
    "caption an image with a vision model",
    "upscale and enhance an image",
    "segment and mask an object",
    "image generation with ControlNet pose",
    "style transfer with IP-Adapter",
    "remove the background from an image",
    "batch process images from a folder",
    "Qwen image edit by text instruction",
];

let _panel = null;
let _refs = null;
let _state = {
    workflows: [],     // fingerprints from last scan
    packages: [],      // all detected package names
    pkgFilter: null,   // Set of enabled pkgs, or null = no filter
    coreOnly: false,
    sortCol: "score",
    sortAsc: false,
    lastResults: [],
};

// ── helpers ───────────────────────────────────────────────────────────
function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
}

function tok(text) {
    const m = String(text || "").toLowerCase().match(/[a-z0-9]+/g);
    return new Set(m || []);
}
export { tok };

function fmtTs(ts) {
    if (!ts) return "";
    const d = new Date(ts * 1000);
    const p = (n) => String(n).padStart(2, "0");
    return `${p(d.getMonth() + 1)}/${p(d.getDate())}/${String(d.getFullYear()).slice(2)} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

async function jget(url) {
    const r = await fetch(url);
    return r.json();
}
async function jpost(url, body) {
    const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
    });
    return r.json();
}

// ── Semantic scoring (mirrors gregowahoo score_local, in JS) ──────────
function scoreLocal(fp, query) {
    const q = tok(query);
    if (!q.size) return { ...fp, score: 0, matched_terms: [] };
    const matched = new Set();
    let score = 0;
    for (const nt of fp.nodes || []) {
        const caps = capabilityFor(nt);
        const hits = [...tok(caps)].filter((t) => q.has(t));
        if (hits.length) {
            score += hits.length * 2;
            hits.forEach((h) => matched.add(h));
        }
    }
    for (const t of fp.titles || []) {
        const hits = [...tok(t)].filter((x) => q.has(x));
        if (hits.length) {
            score += hits.length;
            hits.forEach((h) => matched.add(h));
        }
    }
    for (const s of fp.text_snippets || []) {
        const low = s.toLowerCase();
        for (const t of q) {
            if (t.length > 3 && low.includes(t)) { score += 0.5; break; }
        }
    }
    const fnHits = [...tok(fp.filename)].filter((t) => q.has(t));
    if (fnHits.length) { score += fnHits.length * 1.5; fnHits.forEach((h) => matched.add(h)); }
    return { ...fp, score, matched_terms: [...matched].sort() };
}
export { scoreLocal };

function passesPkgFilter(fp) {
    const req = fp.required_packages || [];
    if (_state.coreOnly) return req.length === 0;
    if (_state.pkgFilter) return req.every((p) => _state.pkgFilter.has(p));
    return true;
}

function computeResults(query) {
    const pool = _state.workflows.filter(passesPkgFilter);
    let res;
    if (query && query.trim()) {
        res = pool.map((fp) => scoreLocal(fp, query)).filter((fp) => fp.score > 0);
    } else {
        res = pool.map((fp) => ({ ...fp, score: 0, matched_terms: [] }));
    }
    return sortResults(res);
}

function sortResults(res) {
    const col = _state.sortCol, asc = _state.sortAsc;
    const num = col === "score" || col === "nodes";
    const val = (fp) => {
        switch (col) {
            case "score": return fp.score || 0;
            case "nodes": return fp.node_count || 0;
            case "filename": return (fp.filename || "").toLowerCase();
            case "modified": return fp.modified || 0;
            default: return 0;
        }
    };
    res.sort((a, b) => {
        const va = val(a), vb = val(b);
        const cmp = num ? (va - vb) : (va < vb ? -1 : va > vb ? 1 : 0);
        return asc ? cmp : -cmp;
    });
    return res;
}

// ── Styles ────────────────────────────────────────────────────────────
function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const el = document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = `
#${PANEL_ID}{position:fixed;top:60px;right:16px;width:760px;height:560px;
  display:none;flex-direction:column;background:#0c0c1e;color:#d4d4f0;
  border:1px solid #2d2d5e;border-radius:8px;z-index:1200;
  box-shadow:0 8px 32px rgba(0,0,0,.55);font:13px Consolas,monospace;overflow:hidden;}
#${PANEL_ID}.open{display:flex;}
#${PANEL_ID} .c2c-lib-hdr{display:flex;align-items:center;gap:8px;padding:8px 10px;
  background:#111128;border-bottom:1px solid #2d2d5e;cursor:move;}
#${PANEL_ID} .c2c-lib-title{font-weight:bold;color:#8a90ff;flex:0 0 auto;}
#${PANEL_ID} .c2c-lib-hdr .sp{flex:1;}
#${PANEL_ID} button{background:#26264e;color:#d4d4f0;border:0;border-radius:4px;
  padding:4px 9px;cursor:pointer;font:12px Consolas,monospace;}
#${PANEL_ID} button:hover{background:#3a3a7e;}
#${PANEL_ID} button.acc{background:#6c72ff;color:#0a0a1a;font-weight:bold;}
#${PANEL_ID} button.acc:hover{background:#8a90ff;}
#${PANEL_ID} .c2c-lib-body{flex:1;overflow:auto;padding:8px 10px;}
#${PANEL_ID} .c2c-lib-foot{padding:5px 10px;background:#080818;border-top:1px solid #2d2d5e;
  color:#5a5a9a;font-size:11px;}
#${PANEL_ID} .sec{margin-bottom:10px;}
#${PANEL_ID} .sec h4{margin:0 0 5px;color:#6c72ff;font-size:11px;letter-spacing:.5px;}
#${PANEL_ID} .locrow{display:flex;align-items:center;gap:6px;padding:2px 0;}
#${PANEL_ID} .locrow .chk{cursor:pointer;width:20px;text-align:center;}
#${PANEL_ID} .locrow .path{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
#${PANEL_ID} .locrow .miss{color:#cc6666;}
#${PANEL_ID} input[type=text]{background:#080818;color:#e0e0ff;border:1px solid #2d2d5e;
  border-radius:4px;padding:5px 8px;font:13px Consolas,monospace;flex:1;}
#${PANEL_ID} select{background:#080818;color:#d4d4f0;border:1px solid #2d2d5e;
  border-radius:4px;padding:4px;font:12px Consolas,monospace;}
#${PANEL_ID} .qrow{display:flex;gap:6px;margin-bottom:5px;align-items:center;}
#${PANEL_ID} table{width:100%;border-collapse:collapse;font-size:12px;}
#${PANEL_ID} th{position:sticky;top:0;background:#1a1a3e;color:#8a90ff;text-align:left;
  padding:4px 6px;cursor:pointer;user-select:none;border-bottom:1px solid #2d2d5e;}
#${PANEL_ID} td{padding:3px 6px;border-bottom:1px solid #18183a;}
#${PANEL_ID} tr:hover td{background:#16163200;background:#161632;}
#${PANEL_ID} tr.sel td{background:#282860;}
#${PANEL_ID} .score{color:#66cc88;text-align:right;}
#${PANEL_ID} .matched{color:#8888bb;}
#${PANEL_ID} .pkgs{color:#cc88cc;font-size:11px;}
#${PANEL_ID} .nodechip{display:inline-block;margin:2px;padding:1px 6px;border-radius:3px;
  font-size:11px;color:#e8e8ff;}
#${BTN_ID}{position:fixed;bottom:8px;right:96px;z-index:1100;background:#26264e;
  color:#d4d4f0;border:1px solid #3a3a7e;border-radius:5px;padding:5px 10px;
  cursor:pointer;font:12px Consolas,monospace;}
#${BTN_ID}:hover{background:#3a3a7e;}
`;
    document.head.appendChild(el);
}

// ── Panel construction ────────────────────────────────────────────────
function buildPanel() {
    injectStyles();
    const root = document.createElement("aside");
    root.id = PANEL_ID;
    root.setAttribute("role", "complementary");
    root.setAttribute("aria-label", "C2C Workflow Library");
    root.innerHTML = `
<div class="c2c-lib-hdr">
  <span class="c2c-lib-title">Workflow Library</span>
  <button data-act="scan" class="acc" title="Scan enabled locations">Scan</button>
  <button data-act="addloc" title="Add a workflow folder">+ Folder</button>
  <span class="sp"></span>
  <button data-act="close" title="Close">X</button>
</div>
<div class="c2c-lib-body">
  <div class="sec" data-role="locs"><h4>SCAN LOCATIONS</h4><div data-role="locrows"></div></div>
  <div class="sec">
    <h4>SEARCH</h4>
    <div class="qrow">
      <input type="text" data-role="q" placeholder="Describe what you want — e.g. generate video from an image" />
      <select data-role="sugg" title="Suggested searches"><option value="">— suggestions —</option>${SUGGESTIONS.map((s) => `<option>${esc(s)}</option>`).join("")}</select>
    </div>
    <div class="qrow">
      <input type="text" data-role="name" placeholder="Name filter (instant filename match)" />
      <button data-act="pkg" title="Filter by required node packages">Pkg Filter</button>
      <button data-act="clear">Clear</button>
    </div>
  </div>
  <div class="sec" data-role="results"></div>
  <div class="sec" data-role="detail"></div>
</div>
<div class="c2c-lib-foot" data-role="status">Not scanned yet — click Scan.</div>
`;
    document.body.appendChild(root);

    const $ = (s) => root.querySelector(s);
    _refs = {
        root,
        locrows: $('[data-role="locrows"]'),
        q: $('[data-role="q"]'),
        name: $('[data-role="name"]'),
        sugg: $('[data-role="sugg"]'),
        results: $('[data-role="results"]'),
        detail: $('[data-role="detail"]'),
        status: $('[data-role="status"]'),
    };

    $('[data-act="close"]').addEventListener("click", closePanel);
    $('[data-act="scan"]').addEventListener("click", doScan);
    $('[data-act="addloc"]').addEventListener("click", addLocation);
    $('[data-act="pkg"]').addEventListener("click", openPkgFilter);
    $('[data-act="clear"]').addEventListener("click", () => {
        _refs.q.value = ""; _refs.name.value = ""; renderResults();
    });
    _refs.q.addEventListener("keydown", (e) => { if (e.key === "Enter") renderResults(); });
    _refs.q.addEventListener("input", debounce(renderResults, 220));
    _refs.name.addEventListener("input", debounce(renderResults, 160));
    _refs.sugg.addEventListener("change", () => {
        if (_refs.sugg.value) { _refs.q.value = _refs.sugg.value; _refs.sugg.value = ""; renderResults(); }
    });
    root.addEventListener("keydown", (e) => { if (e.key === "Escape") closePanel(); });

    attachWindowChrome(root, {
        storageKey: "library",
        header: root.querySelector(".c2c-lib-hdr"),
        titleEl: root.querySelector(".c2c-lib-title"),
        minW: 520, minH: 360,
    });

    _panel = root;
    return _refs;
}

function ensurePanel() { return _refs || buildPanel(); }

// Opens the merged Preset Hub on the Workflows tab (local + online in one
// panel). Falls back to the standalone Library panel if the hub module did
// not load.
function openMerged() {
    const hub = window.__C2C_PRESET_HUB__;
    if (hub && typeof hub.open === "function") { hub.open({ tab: "workflows" }); return; }
    openPanel();
}

function debounce(fn, ms) {
    let t = null;
    return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

function openPanel() {
    const refs = ensurePanel();
    refs.root.classList.add("open");
    refs.root.tabIndex = -1;
    refs.root.focus({ preventScroll: true });
    if (!_state.workflows.length) loadLocations();
}
function closePanel() { if (_panel) _panel.classList.remove("open"); }

// ── Locations ─────────────────────────────────────────────────────────
async function loadLocations() {
    try {
        const data = await jget("/c2c/library/locations");
        if (data.success) renderLocations(data.directories);
    } catch (e) {
        _refs.status.textContent = "Could not load locations: " + e;
    }
}

function renderLocations(dirs) {
    _state.dirs = dirs;
    _refs.locrows.innerHTML = dirs.length
        ? dirs.map((d, i) => `
<div class="locrow">
  <span class="chk" data-i="${i}" style="color:${d.enabled ? "#66cc88" : "#505070"}">${d.enabled ? "\u2611" : "\u2610"}</span>
  <span class="path ${d.exists ? "" : "miss"}">${esc(d.path)}${d.exists ? "" : "  [not found]"}</span>
  <button data-rm="${i}" title="Remove">\u00d7</button>
</div>`).join("")
        : '<div style="color:#5a5a9a">No locations. Click "+ Folder" to add one.</div>';
    _refs.locrows.querySelectorAll(".chk").forEach((el) => {
        el.addEventListener("click", () => {
            const i = +el.dataset.i;
            _state.dirs[i].enabled = !_state.dirs[i].enabled;
            saveLocations();
        });
    });
    _refs.locrows.querySelectorAll("[data-rm]").forEach((el) => {
        el.addEventListener("click", () => {
            _state.dirs.splice(+el.dataset.rm, 1);
            saveLocations();
        });
    });
}

async function saveLocations() {
    const data = await jpost("/c2c/library/locations", { directories: _state.dirs });
    if (data.success) renderLocations(data.directories);
}

async function addLocation() {
    const p = await c2cPrompt("Workflow folder path (absolute):", "");
    if (!p) return;
    _state.dirs = _state.dirs || [];
    _state.dirs.push({ path: p, enabled: true });
    saveLocations();
}

// ── Scan ──────────────────────────────────────────────────────────────
async function doScan() {
    _refs.status.textContent = "Scanning…";
    try {
        const data = await jpost("/c2c/library/scan", { directories: _state.dirs });
        if (!data.success) { _refs.status.textContent = "Scan error: " + (data.error || "?"); return; }
        _state.workflows = data.workflows || [];
        _state.packages = data.packages || [];
        _state.pkgFilter = null;
        _state.coreOnly = false;
        _refs.status.textContent =
            `\u2713 ${data.workflow_count} workflow(s)  (${data.file_count} files)  · ${data.packages.length} custom package(s)`;
        renderResults();
    } catch (e) {
        _refs.status.textContent = "Scan failed: " + e;
    }
}

// ── Results ───────────────────────────────────────────────────────────
function renderResults() {
    if (!_state.workflows.length) {
        _refs.results.innerHTML = '<div style="color:#5a5a9a">Scan a location to begin.</div>';
        return;
    }
    const namePat = _refs.name.value.trim().toLowerCase();
    let res = computeResults(_refs.q.value);
    if (namePat) res = res.filter((fp) => fp.filename.toLowerCase().includes(namePat));
    _state.lastResults = res;

    const arrow = (c) => _state.sortCol === c ? (_state.sortAsc ? " \u25b2" : " \u25bc") : "";
    const head = `
<table><thead><tr>
  <th data-sort="score">Score${arrow("score")}</th>
  <th data-sort="filename">Workflow${arrow("filename")}</th>
  <th data-sort="nodes">#${arrow("nodes")}</th>
  <th data-sort="modified">Modified${arrow("modified")}</th>
  <th>Matched</th>
  <th>Packages</th>
</tr></thead><tbody>`;
    const rows = res.slice(0, 300).map((fp, i) => `
<tr data-i="${i}">
  <td class="score">${fp.score ? fp.score.toFixed(1) : "\u2014"}</td>
  <td>${esc(fp.filename)}</td>
  <td style="text-align:center">${fp.node_count}</td>
  <td>${esc(fmtTs(fp.modified))}</td>
  <td class="matched">${esc((fp.matched_terms || []).join(", "))}</td>
  <td class="pkgs">${esc((fp.required_packages || []).join(", "))}</td>
</tr>`).join("");
    _refs.results.innerHTML = head + rows + "</tbody></table>" +
        `<div style="color:#5a5a9a;margin-top:4px">${res.length} result(s)${res.length > 300 ? " (showing 300)" : ""}</div>`;

    _refs.results.querySelectorAll("th[data-sort]").forEach((th) => {
        th.addEventListener("click", () => {
            const c = th.dataset.sort;
            if (_state.sortCol === c) _state.sortAsc = !_state.sortAsc;
            else { _state.sortCol = c; _state.sortAsc = c === "filename"; }
            renderResults();
        });
    });
    _refs.results.querySelectorAll("tr[data-i]").forEach((tr) => {
        tr.addEventListener("click", () => {
            _refs.results.querySelectorAll("tr.sel").forEach((x) => x.classList.remove("sel"));
            tr.classList.add("sel");
            showDetail(_state.lastResults[+tr.dataset.i]);
        });
    });
}

let _preview = null;

function showDetail(fp) {
    if (_preview) { _preview.destroy(); _preview = null; }
    if (!fp) { _refs.detail.innerHTML = ""; return; }
    const chips = (fp.nodes || []).map((nt) => {
        const cap = capabilityFor(nt);
        return `<span class="nodechip" style="background:${nodeColor(nt)}" title="${esc(cap)}">${esc(nt)}</span>`;
    }).join("");
    _refs.detail.innerHTML = `
<h4>${esc(fp.filename)}</h4>
<div style="color:#8888bb;font-size:11px;margin-bottom:4px">${esc(fp.path)}</div>
<div class="qrow">
  <button class="acc" data-act="open">Open in canvas</button>
  <button data-act="preview">Preview graph</button>
  <button data-act="copy">Copy path</button>
  <span class="sp"></span>
  <span style="color:#5a5a9a">${fp.node_count} nodes</span>
</div>
<div data-role="preview" style="margin-top:6px"></div>
<div data-role="legend" style="margin-top:4px;display:none"></div>
<div style="margin-top:6px">${chips}</div>`;
    _refs.detail.querySelector('[data-act="open"]').addEventListener("click", () => openWorkflow(fp));
    _refs.detail.querySelector('[data-act="preview"]').addEventListener("click", () => previewWorkflow(fp));
    _refs.detail.querySelector('[data-act="copy"]').addEventListener("click", () => {
        navigator.clipboard?.writeText(fp.path);
        _refs.status.textContent = "Copied: " + fp.path;
    });
}

async function previewWorkflow(fp) {
    const host = _refs.detail.querySelector('[data-role="preview"]');
    const legend = _refs.detail.querySelector('[data-role="legend"]');
    if (!host) return;
    host.innerHTML = '<div style="color:#5a5a9a">Loading preview\u2026</div>';
    try {
        const data = await jget("/c2c/library/load?path=" + encodeURIComponent(fp.path));
        if (!data.success) { host.innerHTML = '<div style="color:#cc6666">Preview error: ' + esc(data.error || "?") + "</div>"; return; }
        if (_preview) { _preview.destroy(); _preview = null; }
        _preview = renderGraphPreview(host, data.workflow, { height: 300 });
        if (legend) { legend.innerHTML = legendHTML(); legend.style.display = "block"; }
    } catch (e) {
        host.innerHTML = '<div style="color:#cc6666">Preview failed: ' + esc(String(e)) + "</div>";
    }
}

async function openWorkflow(fp) {
    _refs.status.textContent = "Loading " + fp.filename + "…";
    try {
        const data = await jget("/c2c/library/load?path=" + encodeURIComponent(fp.path));
        if (!data.success) { _refs.status.textContent = "Load error: " + (data.error || "?"); return; }
        await app.loadGraphData(data.workflow);
        _refs.status.textContent = "Loaded " + fp.filename;
        closePanel();
    } catch (e) {
        _refs.status.textContent = "Open failed: " + e;
    }
}

// ── Package filter dialog ─────────────────────────────────────────────
function openPkgFilter() {
    if (!_state.packages.length && !_state.coreOnly) {
        _refs.status.textContent = "No custom packages detected — scan first.";
        return;
    }
    const counts = {};
    for (const fp of _state.workflows) {
        for (const p of fp.required_packages || []) counts[p] = (counts[p] || 0) + 1;
    }
    const enabled = _state.pkgFilter || new Set(_state.packages);
    const wrap = document.createElement("div");
    wrap.style.cssText = "position:fixed;inset:0;z-index:1300;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.5);";
    wrap.innerHTML = `
<div style="background:#0c0c1e;border:1px solid #2d2d5e;border-radius:8px;width:420px;max-height:70vh;
  display:flex;flex-direction:column;color:#d4d4f0;font:13px Consolas,monospace;">
  <div style="padding:8px 10px;border-bottom:1px solid #2d2d5e;color:#8a90ff;font-weight:bold;">Node Pack Filter</div>
  <label style="padding:8px 10px;display:flex;gap:6px;align-items:center;">
    <input type="checkbox" data-role="core" ${_state.coreOnly ? "checked" : ""}/> Core nodes only (no custom packages)
  </label>
  <div data-role="list" style="flex:1;overflow:auto;padding:4px 10px;border-top:1px solid #18183a;"></div>
  <div style="padding:8px 10px;border-top:1px solid #2d2d5e;display:flex;gap:6px;">
    <button data-act="all">All</button><button data-act="none">None</button>
    <span style="flex:1"></span>
    <button data-act="clear">Clear filter</button>
    <button class="acc" data-act="apply" style="background:#6c72ff;color:#0a0a1a;font-weight:bold;border:0;border-radius:4px;padding:4px 9px;cursor:pointer;">Apply</button>
  </div>
</div>`;
    const list = wrap.querySelector('[data-role="list"]');
    list.innerHTML = _state.packages.map((p) => `
<label style="display:flex;gap:6px;align-items:center;padding:2px 0;">
  <input type="checkbox" data-pkg="${esc(p)}" ${enabled.has(p) ? "checked" : ""}/>
  ${esc(p)} <span style="color:#5a5a9a">(${counts[p] || 0})</span>
</label>`).join("") || '<div style="color:#5a5a9a">No custom packages.</div>';
    document.body.appendChild(wrap);
    const close = () => wrap.remove();
    wrap.addEventListener("click", (e) => { if (e.target === wrap) close(); });
    wrap.querySelector('[data-act="all"]').addEventListener("click", () =>
        list.querySelectorAll("input[data-pkg]").forEach((c) => { c.checked = true; }));
    wrap.querySelector('[data-act="none"]').addEventListener("click", () =>
        list.querySelectorAll("input[data-pkg]").forEach((c) => { c.checked = false; }));
    wrap.querySelector('[data-act="clear"]').addEventListener("click", () => {
        _state.pkgFilter = null; _state.coreOnly = false; renderResults(); close();
    });
    wrap.querySelector('[data-act="apply"]').addEventListener("click", () => {
        _state.coreOnly = wrap.querySelector('[data-role="core"]').checked;
        if (_state.coreOnly) { _state.pkgFilter = null; }
        else {
            const checked = new Set(
                [...list.querySelectorAll("input[data-pkg]:checked")].map((c) => c.dataset.pkg));
            _state.pkgFilter = checked.size === _state.packages.length ? null : checked;
        }
        renderResults(); close();
    });
}

// ── Toolbar button (standalone fallback only) ─────────────────────────
// Used only if the OmniBar extension never loads. When OmniBar is present
// the Library lives inside it as a "tools" slot (see _hookOmniBar below) and
// this fixed-position button is removed.
function ensureButton() {
    if (document.getElementById(BTN_ID)) return;
    const b = document.createElement("button");
    b.id = BTN_ID;
    b.textContent = "Library";
    b.title = "C2C Workflow Library — search local + online workflows";
    b.addEventListener("click", openMerged);
    document.body.appendChild(b);
}

// ── OmniBar integration ───────────────────────────────────────────────
// Register the Library as an OmniBar "tools" slot using the public
// window.C2COmniBar.register() API. Falls back to the standalone button if
// OmniBar hasn't loaded yet, retrying briefly while it boots.
function _hookOmniBar() {
    const tryReg = () => {
        const ob = window.C2COmniBar;
        if (!ob || typeof ob.register !== "function") return false;
        try {
            const pill = document.createElement("button");
            pill.id = "c2c-library-pill";
            pill.className = "c2c-omnibar-slot-pill";
            pill.title = "C2C Workflow Library — search local + online workflows";
            pill.style.cssText = "display:flex;align-items:center;gap:5px;";
            const icon = document.createElement("span");
            icon.textContent = "📚";
            icon.style.flexShrink = "0";
            const label = document.createElement("span");
            label.textContent = "Library";
            pill.append(icon, label);
            pill.addEventListener("click", openMerged);

            ob.register({
                section: "tools",
                id: "c2c-library",
                order: 40,
                element: pill,
                onMode(mode) {
                    label.style.display = mode === "icon" ? "none" : "";
                },
            });
            // OmniBar now hosts the Library — drop the standalone button.
            document.getElementById(BTN_ID)?.remove();
            return true;
        } catch (exc) {
            // eslint-disable-next-line no-console
            console.error("[c2c-library] omnibar register", exc);
            return false;
        }
    };
    if (tryReg()) return;
    let n = 0;
    const iv = setInterval(() => {
        if (tryReg() || ++n > 40) clearInterval(iv);
    }, 500);
}

app.registerExtension({
    name: "C2C.WorkflowLibrary",
    async setup() {
        injectStyles();
        ensureButton();
        _hookOmniBar();
        window.__C2C_LIBRARY__ = { open: openPanel, close: closePanel };
    },
});
