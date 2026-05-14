/**
 * mec_group_presets.js — Phase 11: Node Group Presets ("Macro Nodes")
 *
 * Right-click on a selection of nodes → "💾 Save selection as preset…".
 * Adds a floating gallery (📚 button) listing saved presets — click one to
 * paste it at the canvas cursor position.
 *
 * Settings:
 *   mec.group_presets.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

const STYLE_ID    = "mec-group-presets-style";
const GALLERY_ID  = "mec-group-presets-gallery";
const BTN_ID      = "mec-group-presets-btn";

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${BTN_ID} {
    position: fixed;
    bottom: 110px;
    right: 16px;
    z-index: 99996;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #1e1e2e;
    border: 1px solid #45475a;
    color: #cba6f7;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
}
#${BTN_ID}:hover { border-color: #cba6f7; }
#${GALLERY_ID} {
    position: fixed;
    bottom: 110px;
    right: 64px;
    z-index: 99996;
    width: 360px;
    max-height: 60vh;
    overflow-y: auto;
    background: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 10px 12px;
    color: #cdd6f4;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.65);
    display: none;
}
#${GALLERY_ID}.visible { display: block; }
#${GALLERY_ID} .gp-header {
    font-weight: 700;
    color: #cba6f7;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
#${GALLERY_ID} .gp-close {
    background: transparent;
    border: none;
    color: #6c7086;
    cursor: pointer;
    font-size: 14px;
}
#${GALLERY_ID} .gp-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
#${GALLERY_ID} .gp-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px;
    border: 1px solid #313244;
    border-radius: 5px;
    cursor: pointer;
}
#${GALLERY_ID} .gp-item:hover { background: #313244; border-color: #cba6f7; }
#${GALLERY_ID} .gp-thumb {
    width: 48px;
    height: 48px;
    flex-shrink: 0;
    background: #313244;
    border-radius: 4px;
    object-fit: cover;
}
#${GALLERY_ID} .gp-info { flex: 1; min-width: 0; }
#${GALLERY_ID} .gp-name {
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
#${GALLERY_ID} .gp-meta {
    font-size: 10px;
    color: #6c7086;
}
#${GALLERY_ID} .gp-del {
    background: transparent;
    border: 1px solid #45475a;
    color: #f38ba8;
    border-radius: 4px;
    padding: 2px 6px;
    cursor: pointer;
    font-size: 11px;
}
#${GALLERY_ID} .gp-del:hover { background: #3a1e29; }
#${GALLERY_ID} .gp-empty {
    color: #6c7086;
    text-align: center;
    font-style: italic;
    padding: 12px 0;
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
        document.body.appendChild(g);
    }
}

let _galleryOpen = false;
function _toggleGallery() {
    _galleryOpen = !_galleryOpen;
    const g = document.getElementById(GALLERY_ID);
    if (!g) return;
    if (_galleryOpen) {
        g.classList.add("visible");
        _refreshGallery();
    } else {
        g.classList.remove("visible");
    }
}

async function _refreshGallery() {
    const g = document.getElementById(GALLERY_ID);
    if (!g) return;
    g.innerHTML = `
        <div class="gp-header">
            <span>📚 Preset Library</span>
            <button class="gp-close">×</button>
        </div>
        <div class="gp-list"><div class="gp-empty">Loading…</div></div>
    `;
    g.querySelector(".gp-close").addEventListener("click", _toggleGallery);

    let presets = [];
    try {
        const resp = await fetch("/mec/presets");
        const json = await resp.json();
        if (json.success) presets = json.data.presets || [];
    } catch (e) {
        console.warn("[MEC.GroupPresets] list failed:", e);
    }

    const list = g.querySelector(".gp-list");
    if (!presets.length) {
        list.innerHTML = `<div class="gp-empty">No presets yet — select nodes and right-click → "Save as preset".</div>`;
        return;
    }

    list.innerHTML = presets.map(p => `
        <div class="gp-item" data-id="${p.id}">
            ${p.has_thumb
                ? `<img class="gp-thumb" data-id="${p.id}">`
                : `<div class="gp-thumb"></div>`}
            <div class="gp-info">
                <div class="gp-name">${(p.name || p.id).replace(/</g, "&lt;")}</div>
                <div class="gp-meta">${p.node_count} nodes · ${p.created ? new Date(p.created * 1000).toLocaleString() : ""}</div>
            </div>
            <button class="gp-del" data-id="${p.id}">×</button>
        </div>
    `).join("");

    // Lazy-load thumbs
    list.querySelectorAll("img.gp-thumb").forEach(async (img) => {
        const id = img.getAttribute("data-id");
        try {
            const r = await fetch(`/mec/presets/${id}`);
            const j = await r.json();
            if (j.success && j.data.thumbnail) img.src = j.data.thumbnail;
        } catch { /* ignore */ }
    });

    list.querySelectorAll(".gp-item").forEach(el => {
        el.addEventListener("click", async (ev) => {
            if (ev.target.classList.contains("gp-del")) return;
            const id = el.getAttribute("data-id");
            await _loadPreset(id);
        });
    });
    list.querySelectorAll(".gp-del").forEach(b => {
        b.addEventListener("click", async (ev) => {
            ev.stopPropagation();
            const id = b.getAttribute("data-id");
            if (!confirm("Delete this preset?")) return;
            try {
                await fetch(`/mec/presets/${id}`, { method: "DELETE" });
                _refreshGallery();
            } catch (e) {
                console.warn("[MEC.GroupPresets] delete failed:", e);
            }
        });
    });
}

function _selectedNodes() {
    const c = app.canvas;
    if (!c) return [];
    return Object.values(c.selected_nodes || {});
}

function _serializeSubgraph(nodes) {
    const idSet = new Set(nodes.map(n => n.id));
    const ser = nodes.map(n => n.serialize ? n.serialize() : null).filter(Boolean);
    const g = app.graph;
    let links = [];
    if (g && g.links) {
        const all = Array.isArray(g.links) ? g.links : Object.values(g.links);
        for (const l of all) {
            if (!l) continue;
            if (idSet.has(l.origin_id) && idSet.has(l.target_id)) {
                links.push(Array.isArray(l)
                    ? l
                    : [l.id, l.origin_id, l.origin_slot, l.target_id, l.target_slot, l.type]);
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

async function _savePreset() {
    const nodes = _selectedNodes();
    if (nodes.length === 0) {
        alert("Select one or more nodes first.");
        return;
    }
    const name = prompt(`Save ${nodes.length} node(s) as preset — name?`, "My preset");
    if (!name) return;
    const subgraph = _serializeSubgraph(nodes);
    const thumbnail = _thumbnailFromCanvas();
    try {
        const resp = await fetch("/mec/presets", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, subgraph, thumbnail }),
        });
        const json = await resp.json();
        if (!json.success) {
            alert(`Save failed: ${json.error || "unknown"}`);
            return;
        }
        console.log("[MEC.GroupPresets] Saved:", json.data);
        if (_galleryOpen) _refreshGallery();
    } catch (e) {
        console.warn("[MEC.GroupPresets] save failed:", e);
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
        console.warn("[MEC.GroupPresets] load failed:", e);
        alert("Could not load preset.");
        return;
    }
    const sub = data.subgraph;
    if (!sub || !sub.nodes) return;

    // Compute paste offset: bring centroid of subgraph to current canvas center
    const g = app.graph;
    const canvas = app.canvas;
    if (!g || !canvas) return;

    // ID remap: each node needs a fresh id to avoid collision.
    const oldToNew = new Map();
    let baseX = 0, baseY = 0, count = 0;
    for (const n of sub.nodes) {
        if (n.pos) { baseX += n.pos[0]; baseY += n.pos[1]; count++; }
    }
    if (count) { baseX /= count; baseY /= count; }
    const target = canvas.convertOffsetToCanvas
        ? canvas.convertCanvasToOffset
            ? canvas.convertCanvasToOffset([canvas.canvas.width / 2, canvas.canvas.height / 2])
            : [0, 0]
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
            node.pos = [ (ns.pos?.[0] || 0) + dx, (ns.pos?.[1] || 0) + dy ];
            // Force a fresh id assignment
            node.id = -1;
            g.add(node);
            created.push(node);
        } catch (e) {
            console.warn("[MEC.GroupPresets] node add failed:", ns.type, e);
        }
    }
    // Re-wire links between newly created nodes
    for (const link of (sub.links || [])) {
        const [_id, oOrig, oSlot, oTarget, oTargetSlot] = link;
        const src = oldToNew.get(oOrig);
        const dst = oldToNew.get(oTarget);
        if (src && dst) {
            try { src.connect(oSlot, dst, oTargetSlot); } catch { /* ignore */ }
        }
    }
    canvas.setDirty(true, true);
    console.log("[MEC.GroupPresets] Loaded", created.length, "nodes from preset", id);
}

function _patchCanvasMenu() {
    const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
    if (!orig || orig._mecPresetsPatched) return;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        const opts = orig.call(this);
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("mec.group_presets.enabled", true); }
            catch { return true; }
        })();
        if (!enabled) return opts;
        opts.push(null);
        opts.push({
            content: "📚 Preset library…",
            callback: () => _toggleGallery(),
        });
        const sel = _selectedNodes();
        if (sel.length > 0) {
            opts.push({
                content: `💾 Save ${sel.length} selected as preset…`,
                callback: () => _savePreset(),
            });
        }
        return opts;
    };
    LGraphCanvas.prototype.getCanvasMenuOptions._mecPresetsPatched = true;
}

app.registerExtension({
    name: "MEC.GroupPresets",
    settings: [
        {
            id: "mec.group_presets.enabled",
            name: "Group Presets: enabled",
            tooltip: "Save selections as reusable presets and recall via the 📚 button.",
            type: "boolean",
            defaultValue: true,
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
            try { return app.ui.settings.getSettingValue("mec.group_presets.enabled", true); }
            catch { return true; }
        })();
        const b = document.getElementById(BTN_ID);
        if (b) b.style.display = enabled ? "flex" : "none";
        console.log("[MEC.GroupPresets] Loaded.");
    },
});
