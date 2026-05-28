/**
 * c2c_group_presets.js — Node Group Presets (god-level rebuild, 2026-05-27)
 *
 * Right-click selection → "💾 Save selection as preset…". Floating 📚 button
 * opens a gallery with import/export, civitai-format share, tag search,
 * AI naming suggestions, and one-click recall onto the canvas.
 *
 * Features:
 *   1) Save selection as preset (with thumbnail)
 *   2) Gallery (lazy thumbs, search, tag filter)
 *   3) Recall preset → spawn nodes at canvas centre with link rewire
 *   4) Delete preset
 *   5) Import a .c2cpreset.json file (or paste JSON)
 *   6) Export a single preset as civitai-compatible blob (workflow + meta + thumb)
 *   7) Bulk export ALL presets as a single .zip-style JSON pack
 *   8) AI naming/tag suggestion via streamAI on save
 *   9) Body-only re-render preserves chrome; listener registry; chrome-safe z-index
 */

import { app } from "../../scripts/app.js";
import { attachWindowChrome } from "./_c2c_window.js";
import { streamAI } from "./_c2c_ai_client.js";

const STYLE_ID    = "mec-group-presets-style";
const GALLERY_ID  = "mec-group-presets-gallery";
const BTN_ID      = "mec-group-presets-btn";

const _state = {
    open: false,
    presets: [],
    filter: "",
    busy: false,
};
const _listeners = [];
function _on(t, e, fn, opts) { t.addEventListener(e, fn, opts); _listeners.push(() => t.removeEventListener(e, fn, opts)); }
function _clear() { while (_listeners.length) { try { _listeners.pop()(); } catch {} } }

function _esc(s) { return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c])); }

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${BTN_ID} {
    position: fixed; bottom: 110px; right: 16px;
    z-index: var(--c2c-z-hud, 1000);
    width: 38px; height: 38px; border-radius: 50%;
    background: var(--c2c-bg); border: 1px solid var(--c2c-surface1); color: var(--c2c-mauve);
    font-size: 16px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex; align-items: center; justify-content: center;
}
#${BTN_ID}:hover { border-color: var(--c2c-mauve); }

#${GALLERY_ID} {
    position: fixed; top: 80px; right: 80px;
    width: min(94vw, 480px); height: min(82vh, 640px);
    background: var(--c2c-bg2); border: 1px solid var(--c2c-surface1); border-radius: 8px;
    color: var(--c2c-fg); font-family: -apple-system, "Segoe UI", sans-serif; font-size: 12px;
    box-shadow: 0 6px 32px rgba(0,0,0,0.7);
    display: none; flex-direction: column; overflow: hidden;
}
#${GALLERY_ID}.visible { display: flex; }
#${GALLERY_ID} .gp-header {
    margin: 0; padding: 8px 12px; color: var(--c2c-mauve); font-size: 14px;
    display: flex; justify-content: space-between; align-items: center;
    background: linear-gradient(180deg,var(--c2c-bg),var(--c2c-bg2)); border-bottom: 1px solid var(--c2c-surface0);
}
#${GALLERY_ID} .gp-close {
    background: none; border: none; color: var(--c2c-overlay0); cursor: pointer; font-size: 16px; padding: 0 4px;
}
#${GALLERY_ID} .gp-body { flex: 1 1 auto; min-height: 0; overflow: auto; padding: 10px; }
#${GALLERY_ID} .gp-toolbar { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; align-items: center; }
#${GALLERY_ID} .gp-toolbar button, #${GALLERY_ID} .gp-toolbar input {
    background: var(--c2c-surface0); border: 1px solid var(--c2c-surface1); color: var(--c2c-fg); border-radius: 4px;
    padding: 3px 7px; font-size: 11px;
}
#${GALLERY_ID} .gp-toolbar button { cursor: pointer; }
#${GALLERY_ID} .gp-toolbar button:hover { border-color: var(--c2c-mauve); }
#${GALLERY_ID} .gp-toolbar input { flex: 1; min-width: 80px; }

#${GALLERY_ID} .gp-list { display: flex; flex-direction: column; gap: 6px; }
#${GALLERY_ID} .gp-item {
    display: flex; align-items: center; gap: 8px; padding: 6px;
    border: 1px solid var(--c2c-surface0); border-radius: 5px; cursor: pointer;
}
#${GALLERY_ID} .gp-item:hover { background: var(--c2c-surface0); border-color: var(--c2c-mauve); }
#${GALLERY_ID} .gp-thumb { width: 48px; height: 48px; flex-shrink: 0; background: var(--c2c-surface0); border-radius: 4px; object-fit: cover; }
#${GALLERY_ID} .gp-info { flex: 1; min-width: 0; }
#${GALLERY_ID} .gp-name { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#${GALLERY_ID} .gp-meta { font-size: 10px; color: var(--c2c-overlay0); }
#${GALLERY_ID} .gp-tags { font-size: 10px; color: var(--c2c-yellow); margin-top: 2px; }
#${GALLERY_ID} .gp-actions { display: flex; gap: 4px; }
#${GALLERY_ID} .gp-actions button {
    background: transparent; border: 1px solid var(--c2c-surface1); border-radius: 4px;
    padding: 2px 6px; cursor: pointer; font-size: 11px;
}
#${GALLERY_ID} .gp-actions .gp-share { color: var(--c2c-blue); }
#${GALLERY_ID} .gp-actions .gp-share:hover { background: var(--c2c-panelTint); }
#${GALLERY_ID} .gp-actions .gp-del { color: var(--c2c-red); }
#${GALLERY_ID} .gp-actions .gp-del:hover { background: var(--c2c-dangerBg); }
#${GALLERY_ID} .gp-empty { color: var(--c2c-overlay0); text-align: center; font-style: italic; padding: 30px 0; }

#${GALLERY_ID} .gp-import-box {
    background: var(--c2c-bg3); border: 1px solid var(--c2c-surface0); border-radius: 4px;
    padding: 8px; margin-bottom: 8px;
}
#${GALLERY_ID} .gp-import-box textarea {
    width: 100%; min-height: 80px; background: var(--c2c-bg); border: 1px solid var(--c2c-surface0);
    color: var(--c2c-fg); font-family: monospace; font-size: 10px; padding: 4px; box-sizing: border-box;
}
    `.trim();
    document.head.appendChild(style);
}

function _ensureUi() {
    if (!document.getElementById(BTN_ID)) {
        const b = document.createElement("button");
        b.id = BTN_ID;
        b.title = "Node Group Presets";
        b.textContent = "📚";
        b.addEventListener("click", _toggleGallery);
        document.body.appendChild(b);
    }
    if (!document.getElementById(GALLERY_ID)) {
        const g = document.createElement("div");
        g.id = GALLERY_ID;
        g.innerHTML = `
            <div class="gp-header"><span>📚 Preset Library</span><button class="gp-close">×</button></div>
            <div class="gp-body" data-role="body"></div>
        `;
        document.body.appendChild(g);
        g.querySelector(".gp-close").addEventListener("click", _toggleGallery);
        attachWindowChrome(g, { storageKey: "group_presets", headerSelector: ".gp-header", titleSelector: ".gp-header > span", minW: 360, minH: 320 });
    }
}

function _toggleGallery() {
    _state.open = !_state.open;
    const g = document.getElementById(GALLERY_ID);
    if (!g) return;
    if (_state.open) { g.classList.add("visible"); _refreshGallery(); }
    else { g.classList.remove("visible"); _clear(); }
}

async function _fetchPresets() {
    try {
        const r = await fetch("/mec/presets");
        const j = await r.json();
        if (j.success) return j.data.presets || [];
    } catch (e) { console.warn("[C2C.GroupPresets] list failed:", e); }
    return [];
}

async function _refreshGallery() {
    _state.busy = true;
    _renderGallery();
    _state.presets = await _fetchPresets();
    _state.busy = false;
    _renderGallery();
}

function _renderGallery() {
    const g = document.getElementById(GALLERY_ID);
    if (!g) return;
    const body = g.querySelector('[data-role="body"]');
    if (!body) return;
    _clear();
    const tb = `
        <div class="gp-toolbar">
            <input data-role="filter" placeholder="search…" value="${_esc(_state.filter)}">
            <button data-act="import">⇧ Import</button>
            <button data-act="export-all" ${_state.presets.length?"":"disabled"}>⇩ Export all</button>
            <button data-act="refresh">🔄</button>
        </div>
        <div data-role="import-box"></div>
        <div class="gp-list" data-role="list"></div>
    `;
    body.innerHTML = tb;
    _bindToolbar(body);
    _renderList(body);
}

function _bindToolbar(body) {
    _on(body.querySelector('[data-role="filter"]'), "input", (e) => { _state.filter = e.target.value; _renderList(body); });
    _on(body.querySelector('[data-act="import"]'), "click", () => _renderImportBox(body));
    _on(body.querySelector('[data-act="export-all"]'), "click", () => _exportAll());
    _on(body.querySelector('[data-act="refresh"]'), "click", () => _refreshGallery());
}

function _renderImportBox(body) {
    const slot = body.querySelector('[data-role="import-box"]');
    if (!slot) return;
    slot.innerHTML = `
        <div class="gp-import-box">
            <div style="margin-bottom:4px;color:var(--c2c-mauve);font-size:11px;">Paste preset JSON (civitai or .c2cpreset.json) and click Import:</div>
            <textarea data-role="import-ta" placeholder='{"name":"...","subgraph":{"nodes":[...]}}'></textarea>
            <div style="margin-top:6px;display:flex;gap:4px;">
                <button data-act="import-go" style="background:var(--c2c-surface0);border:1px solid var(--c2c-surface1);color:var(--c2c-fg);border-radius:4px;padding:3px 8px;cursor:pointer;font-size:11px;">Import</button>
                <button data-act="import-file" style="background:var(--c2c-surface0);border:1px solid var(--c2c-surface1);color:var(--c2c-fg);border-radius:4px;padding:3px 8px;cursor:pointer;font-size:11px;">From file…</button>
                <button data-act="import-cancel" style="background:transparent;border:1px solid var(--c2c-surface0);color:var(--c2c-overlay0);border-radius:4px;padding:3px 8px;cursor:pointer;font-size:11px;">Cancel</button>
            </div>
        </div>
    `;
    _on(slot.querySelector('[data-act="import-go"]'), "click", async () => {
        const txt = slot.querySelector('[data-role="import-ta"]').value.trim();
        if (!txt) return;
        await _importJson(txt);
        slot.innerHTML = "";
        _refreshGallery();
    });
    _on(slot.querySelector('[data-act="import-file"]'), "click", () => {
        const inp = document.createElement("input");
        inp.type = "file";
        inp.accept = ".json,.c2cpreset,application/json";
        inp.addEventListener("change", async () => {
            const f = inp.files?.[0];
            if (!f) return;
            const text = await f.text();
            await _importJson(text);
            slot.innerHTML = "";
            _refreshGallery();
        });
        inp.click();
    });
    _on(slot.querySelector('[data-act="import-cancel"]'), "click", () => { slot.innerHTML = ""; });
}

function _renderList(body) {
    const list = body.querySelector('[data-role="list"]');
    if (!list) return;
    if (_state.busy) { list.innerHTML = `<div class="gp-empty">Loading…</div>`; return; }
    const q = _state.filter.trim().toLowerCase();
    const filtered = _state.presets.filter((p) => {
        if (!q) return true;
        if ((p.name || "").toLowerCase().includes(q)) return true;
        if ((p.tags || []).some((t) => String(t).toLowerCase().includes(q))) return true;
        return false;
    });
    if (!filtered.length) {
        list.innerHTML = `<div class="gp-empty">${_state.presets.length ? "No presets match." : "No presets yet — select nodes and right-click → \"Save as preset\"."}</div>`;
        return;
    }
    list.innerHTML = filtered.map((p) => `
        <div class="gp-item" data-id="${_esc(p.id)}">
            ${p.has_thumb ? `<img class="gp-thumb" data-id="${_esc(p.id)}">` : `<div class="gp-thumb"></div>`}
            <div class="gp-info">
                <div class="gp-name">${_esc(p.name || p.id)}</div>
                <div class="gp-meta">${p.node_count} nodes · ${p.created ? new Date(p.created * 1000).toLocaleString() : ""}</div>
                ${(p.tags && p.tags.length) ? `<div class="gp-tags">#${p.tags.map(_esc).join(" #")}</div>` : ""}
            </div>
            <div class="gp-actions">
                <button class="gp-share" data-act="share" data-id="${_esc(p.id)}" title="Export (civitai-share)">⇩</button>
                <button class="gp-del"   data-act="del"   data-id="${_esc(p.id)}" title="Delete">×</button>
            </div>
        </div>
    `).join("");
    // Lazy thumbs
    list.querySelectorAll("img.gp-thumb").forEach(async (img) => {
        const id = img.getAttribute("data-id");
        try {
            const r = await fetch(`/mec/presets/${id}`);
            const j = await r.json();
            if (j.success && j.data.thumbnail) img.src = j.data.thumbnail;
        } catch {}
    });
    list.querySelectorAll(".gp-item").forEach((el) => {
        _on(el, "click", (ev) => {
            const act = ev.target.getAttribute?.("data-act");
            const id = ev.target.getAttribute?.("data-id") || el.getAttribute("data-id");
            if (act === "share") { ev.stopPropagation(); _exportOne(id); return; }
            if (act === "del")   { ev.stopPropagation(); _deletePreset(id); return; }
            _loadPreset(id);
        });
    });
}

async function _deletePreset(id) {
    if (!confirm("Delete this preset?")) return;
    try {
        await fetch(`/mec/presets/${id}`, { method: "DELETE" });
        _refreshGallery();
    } catch (e) {
        console.warn("[C2C.GroupPresets] delete failed:", e);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Import / Export (civitai-compatible blob)
// ─────────────────────────────────────────────────────────────────────────────

function _civitaiBlob(name, sub, thumbnail, tags = [], meta = {}) {
    // civitai workflow JSON convention: meta.workflow holds the inline subgraph.
    return {
        _schema: "c2c.preset/1",
        _civitai: true,
        name,
        type: "ComfyUI Workflow Snippet",
        tags,
        thumbnail,
        meta: {
            createdAt: new Date().toISOString(),
            nodeCount: sub?.nodes?.length || 0,
            ...meta,
        },
        workflow: sub,
    };
}

async function _exportOne(id) {
    try {
        const r = await fetch(`/mec/presets/${id}`);
        const j = await r.json();
        if (!j.success) throw new Error(j.error || "load");
        const d = j.data;
        const blob = _civitaiBlob(d.name || id, d.subgraph, d.thumbnail || null, d.tags || []);
        const text = JSON.stringify(blob, null, 2);
        await navigator.clipboard.writeText(text);
        // Also offer a download
        const f = new Blob([text], { type: "application/json" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(f);
        a.download = `${(d.name || id).replace(/[^a-z0-9_\-]+/gi, "_")}.c2cpreset.json`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 5000);
        console.log("[C2C.GroupPresets] Exported preset (civitai-format) & copied to clipboard.");
    } catch (e) {
        console.warn("[C2C.GroupPresets] export failed:", e);
        alert("Export failed — see console.");
    }
}

async function _exportAll() {
    const detailed = [];
    for (const p of _state.presets) {
        try {
            const r = await fetch(`/mec/presets/${p.id}`);
            const j = await r.json();
            if (j.success) detailed.push(_civitaiBlob(j.data.name || p.id, j.data.subgraph, j.data.thumbnail || null, j.data.tags || []));
        } catch {}
    }
    const pack = { _schema: "c2c.preset.pack/1", count: detailed.length, presets: detailed, exportedAt: new Date().toISOString() };
    const text = JSON.stringify(pack, null, 2);
    const f = new Blob([text], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(f);
    a.download = `c2c_presets_pack_${Date.now()}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

async function _importJson(text) {
    let obj;
    try { obj = JSON.parse(text); }
    catch (e) { alert("Invalid JSON."); return; }
    const items = obj?._schema === "c2c.preset.pack/1" && Array.isArray(obj.presets)
        ? obj.presets
        : [obj];
    let ok = 0, fail = 0;
    for (const it of items) {
        try {
            const name = it.name || `Imported ${Date.now()}`;
            const sub = it.workflow || it.subgraph || null;
            if (!sub || !sub.nodes) { fail++; continue; }
            const thumbnail = it.thumbnail || null;
            const tags = it.tags || [];
            const resp = await fetch("/mec/presets", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, subgraph: sub, thumbnail, tags }),
            });
            const j = await resp.json();
            if (j.success) ok++; else fail++;
        } catch { fail++; }
    }
    alert(`Imported ${ok} preset(s)${fail ? `, ${fail} failed` : ""}.`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Save / Recall
// ─────────────────────────────────────────────────────────────────────────────

function _selectedNodes() {
    const c = app.canvas;
    if (!c) return [];
    return Object.values(c.selected_nodes || {});
}

function _serializeSubgraph(nodes) {
    const idSet = new Set(nodes.map((n) => n.id));
    const ser = nodes.map((n) => n.serialize ? n.serialize() : null).filter(Boolean);
    const g = app.graph;
    const links = [];
    if (g && g.links) {
        const all = Array.isArray(g.links) ? g.links : Object.values(g.links);
        for (const l of all) {
            if (!l) continue;
            if (idSet.has(l.origin_id) && idSet.has(l.target_id)) {
                links.push(Array.isArray(l) ? l : [l.id, l.origin_id, l.origin_slot, l.target_id, l.target_slot, l.type]);
            }
        }
    }
    return { nodes: ser, links };
}

function _thumbnailFromCanvas() {
    try {
        const canvas = app.canvas?.canvas;
        if (!canvas) return null;
        const small = document.createElement("canvas");
        const W = 96, H = 96;
        small.width = W; small.height = H;
        const ctx = small.getContext("2d");
        ctx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, W, H);
        const url = small.toDataURL("image/jpeg", 0.6);
        if (url.length > 60000) return null;
        return url;
    } catch { return null; }
}

async function _aiSuggestNameAndTags(subgraph) {
    try {
        const classes = (subgraph.nodes || []).map((n) => n.type).slice(0, 30);
        let out = "";
        await streamAI({
            feature: "preset_name",
            sensitivity: "normal",
            max_tokens: 80,
            temperature: 0.5,
            messages: [
                { role: "system", content: "You name ComfyUI workflow snippets. Reply with JSON only: {\"name\":\"…\", \"tags\":[\"…\",\"…\"]}. Name ≤ 5 words; 2–4 short kebab-case tags." },
                { role: "user", content: `Node classes:\n${classes.join("\n")}` },
            ],
            onChunk: (c) => { out += c; },
            onError: () => {},
            onDone: () => {},
        });
        const m = out.match(/\{[\s\S]*\}/);
        if (!m) return null;
        return JSON.parse(m[0]);
    } catch { return null; }
}

async function _savePreset() {
    const nodes = _selectedNodes();
    if (nodes.length === 0) { alert("Select one or more nodes first."); return; }
    const subgraph = _serializeSubgraph(nodes);
    let defaultName = "My preset";
    let defaultTags = [];
    const useAI = confirm(`Save ${nodes.length} node(s) as preset.\n\nUse AI to suggest a name + tags?`);
    if (useAI) {
        const ai = await _aiSuggestNameAndTags(subgraph);
        if (ai) { defaultName = ai.name || defaultName; defaultTags = ai.tags || []; }
    }
    const name = prompt(`Preset name?`, defaultName);
    if (!name) return;
    const tagsStr = prompt(`Tags (comma-separated, optional)?`, defaultTags.join(", "));
    const tags = (tagsStr || "").split(",").map((s) => s.trim()).filter(Boolean);
    const thumbnail = _thumbnailFromCanvas();
    try {
        const resp = await fetch("/mec/presets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, subgraph, thumbnail, tags }),
        });
        const json = await resp.json();
        if (!json.success) { alert(`Save failed: ${json.error || "unknown"}`); return; }
        console.log("[C2C.GroupPresets] Saved:", json.data);
        if (_state.open) _refreshGallery();
    } catch (e) {
        console.warn("[C2C.GroupPresets] save failed:", e);
        alert("Save failed — see console.");
    }
}

async function _loadPreset(id) {
    let data;
    try {
        const r = await fetch(`/mec/presets/${id}`);
        const j = await r.json();
        if (!j.success) throw new Error(j.error || "load_failed");
        data = j.data;
    } catch (e) {
        console.warn("[C2C.GroupPresets] load failed:", e);
        alert("Could not load preset.");
        return;
    }
    const sub = data.subgraph;
    if (!sub || !sub.nodes) return;

    const g = app.graph;
    const canvas = app.canvas;
    if (!g || !canvas) return;

    const oldToNew = new Map();
    let baseX = 0, baseY = 0, count = 0;
    for (const n of sub.nodes) if (n.pos) { baseX += n.pos[0]; baseY += n.pos[1]; count++; }
    if (count) { baseX /= count; baseY /= count; }
    const target = canvas.convertCanvasToOffset
        ? canvas.convertCanvasToOffset([canvas.canvas.width / 2, canvas.canvas.height / 2])
        : [0, 0];
    const dx = target[0] - baseX + 40;
    const dy = target[1] - baseY + 40;

    const created = [];
    for (const ns of sub.nodes) {
        try {
            const node = LiteGraph.createNode(ns.type);
            if (!node) continue;
            node.configure(ns);
            oldToNew.set(ns.id, node);
            node.pos = [(ns.pos?.[0] || 0) + dx, (ns.pos?.[1] || 0) + dy];
            node.id = -1;
            g.add(node);
            created.push(node);
        } catch (e) {
            console.warn("[C2C.GroupPresets] node add failed:", ns.type, e);
        }
    }
    for (const link of (sub.links || [])) {
        const [, oOrig, oSlot, oTarget, oTargetSlot] = link;
        const src = oldToNew.get(oOrig);
        const dst = oldToNew.get(oTarget);
        if (src && dst) { try { src.connect(oSlot, dst, oTargetSlot); } catch {} }
    }
    canvas.setDirty(true, true);
    console.log("[C2C.GroupPresets] Loaded", created.length, "nodes from preset", id);
}

function _patchCanvasMenu() {
    const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
    if (!orig || orig._mecPresetsPatched) return;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        const opts = orig.call(this);
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("c2c.group_presets.enabled", true); }
            catch { return true; }
        })();
        if (!enabled) return opts;
        opts.push(null);
        opts.push({ content: "📚 Preset library…", callback: () => _toggleGallery() });
        const sel = _selectedNodes();
        if (sel.length > 0) {
            opts.push({ content: `💾 Save ${sel.length} selected as preset…`, callback: () => _savePreset() });
        }
        return opts;
    };
    LGraphCanvas.prototype.getCanvasMenuOptions._mecPresetsPatched = true;
}

app.registerExtension({
    name: "C2C.GroupPresets",
    settings: [
        {
            id: "c2c.group_presets.enabled",
            name: "Group Presets: enabled",
            tooltip: "Save selections as reusable presets and recall via the 📚 button.",
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
        _patchCanvasMenu();
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("c2c.group_presets.enabled", true); }
            catch { return true; }
        })();
        const b = document.getElementById(BTN_ID);
        if (b) b.style.display = enabled ? "flex" : "none";
        console.log("[C2C.GroupPresets] godlevel-rebuild loaded.");
    },
});
