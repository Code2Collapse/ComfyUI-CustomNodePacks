// c2c_node_bookmarks.js — Ctrl+B toggle bookmark, Ctrl+Shift+1..9 jump (C2C)
// ---------------------------------------------------------------------
// What it does:
//   • Ctrl+B on a selected node → adds/removes a bookmark slot (1..9).
//     Bookmark slot is auto-assigned to the lowest free index 1..9.
//   • Ctrl+Shift+1..9 → pan + zoom to that bookmark, pulse the node.
//   • Bookmarks persist in workflow JSON under
//     `extra.c2c.bookmarks = { "<slot>": <node_id>, ... }` (round-trips
//     through Save/Load).
//   • A small 9-cell strip top-right shows current bookmarks (digit +
//     node name). Click a cell to jump.
//
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";
import { findNodeAnywhere, dirtyAllGraphs } from "./_subgraph_walk.js";

const ROOT_ID = "c2c-bookmarks-root";
const SETTING_ID = "c2c.bookmarks.enabled";
const PULSE_MS = 900;

function injectStyle() {
    if (document.getElementById("c2c-bookmarks-style")) return;
    const s = document.createElement("style");
    s.id = "c2c-bookmarks-style";
    s.textContent = `
#${ROOT_ID} {
    position: fixed; top: 56px; right: 14px;
    z-index: var(--c2c-z-hud, 1000); display: flex; gap: 4px;
    background: color-mix(in srgb, var(--c2c-bg) 78%, transparent);
    border: 1px solid color-mix(in srgb, var(--c2c-fg) 8%, transparent);
    border-radius: 7px; padding: 4px;
    font: 11px ui-sans-serif, system-ui, sans-serif;
    color: var(--c2c-fg); backdrop-filter: blur(6px);
}
#${ROOT_ID} .cell {
    min-width: 22px; height: 22px;
    border-radius: 4px;
    background: color-mix(in srgb, var(--c2c-fg) 4%, transparent);
    border: 1px solid color-mix(in srgb, var(--c2c-fg) 6%, transparent);
    display:flex; align-items:center; justify-content:center;
    cursor: pointer; color: var(--c2c-sub, var(--c2c-fg));
    font-weight: 600;
}
#${ROOT_ID} .cell.filled {
    background: color-mix(in srgb, var(--c2c-blue) 16%, transparent);
    border-color: color-mix(in srgb, var(--c2c-blue) 45%, transparent);
    color: var(--c2c-fg);
}
#${ROOT_ID} .cell:hover {
    border-color: color-mix(in srgb, var(--c2c-yellow) 60%, transparent);
}`;
    document.head.appendChild(s);
}

function getMap() {
    const g = app.graph;
    if (!g) return {};
    g.extra = g.extra || {};
    g.extra.c2c = g.extra.c2c || {};
    g.extra.c2c.bookmarks = g.extra.c2c.bookmarks || {};
    return g.extra.c2c.bookmarks;
}

function findNode(id) {
    const hit = findNodeAnywhere(id);
    return hit ? hit.node : null;
}

function pulseFocus(node) {
    const c = app.canvas; if (!c || !node) return;
    const ds = c.ds;
    if (ds) {
        const cx = node.pos[0] + node.size[0] * 0.5;
        const cy = node.pos[1] + node.size[1] * 0.5;
        const dpr = window.devicePixelRatio || 1;
        ds.scale = Math.min(1.2, Math.max(0.55, ds.scale || 1));
        ds.offset[0] = -cx + (c.canvas.width  / dpr) / (2 * ds.scale);
        ds.offset[1] = -cy + (c.canvas.height / dpr) / (2 * ds.scale);
    }
    c.selectNode?.(node, false);
    node._c2c_pulse_until = performance.now() + PULSE_MS;
    c.setDirty(true, true);
}

function selectedNode() {
    const sel = app.canvas?.selected_nodes;
    if (!sel) return null;
    const keys = Object.keys(sel);
    return keys.length ? sel[keys[0]] : null;
}

function nextFreeSlot(map) {
    for (let i = 1; i <= 9; i++) if (!map[String(i)]) return String(i);
    return null;
}

function toggleBookmark() {
    const n = selectedNode();
    if (!n) return;
    const map = getMap();
    // Already bookmarked?
    for (const [slot, id] of Object.entries(map)) {
        if (String(id) === String(n.id)) { delete map[slot]; render(); return; }
    }
    const slot = nextFreeSlot(map);
    if (!slot) return;
    map[slot] = n.id;
    render();
    pulseFocus(n);
}

function jumpToSlot(slot) {
    const map = getMap();
    const id = map[String(slot)];
    if (id == null) return;
    const n = findNode(id);
    if (!n) { delete map[String(slot)]; render(); return; }
    pulseFocus(n);
}

let _root = null;
let _omnibarUnreg = null;

function ensureRoot() {
    if (_root) return _root;
    injectStyle();
    _root = document.createElement("div");
    _root.id = ROOT_ID;
    // Register with OmniBar bookmarks section. Retry for up to 4 s (40×100ms)
    // because OmniBar extension loads after us alphabetically. If OmniBar never
    // arrives, fall back to legacy body-appended fixed strip.
    let _tries = 0;
    const _tryOmniBar = () => {
        if (window.C2COmniBar?.register) {
            _omnibarUnreg = window.C2COmniBar.register({
                section: "bookmarks",
                id: "bookmarks",
                order: 10,
                element: _root,
                update: render,
            });
        } else if (++_tries < 40) {
            setTimeout(_tryOmniBar, 100);
        } else {
            // OmniBar unavailable — mount as legacy fixed strip
            if (!document.body.contains(_root)) document.body.appendChild(_root);
        }
    };
    _tryOmniBar();
    return _root;
}

function render() {
    ensureRoot();
    try {
        if (app.ui?.settings?.getSettingValue?.(SETTING_ID, true) === false) {
            _root.style.display = "none"; return;
        }
    } catch { /* settings not ready */ }
    _root.style.display = "flex";
    const map = getMap();
    let html = "";
    for (let i = 1; i <= 9; i++) {
        const id = map[String(i)];
        const n = id != null ? findNode(id) : null;
        const cls = n ? "cell filled" : "cell";
        const tip = n ? `${i} · ${n.title || n.type}` : `Bookmark ${i} (empty)`;
        html += `<div class="${cls}" data-slot="${i}" title="${tip.replace(/"/g, "&quot;")}">${i}</div>`;
    }
    _root.innerHTML = html;
    [..._root.children].forEach((el) => {
        el.addEventListener("click", () => jumpToSlot(el.dataset.slot));
    });
}

function isEditingField() {
    const ae = document.activeElement;
    if (!ae) return false;
    const tag = ae.tagName?.toLowerCase();
    return tag === "input" || tag === "textarea" || ae.isContentEditable;
}

function onKey(ev) {
    if (isEditingField()) return;
    if (!(ev.ctrlKey || ev.metaKey)) return;
    // Ctrl+B → toggle bookmark on selected node
    if ((ev.key === "b" || ev.key === "B") && !ev.shiftKey) {
        ev.preventDefault();
        toggleBookmark();
        return;
    }
    // Ctrl+Shift+1..9 → jump to slot
    if (ev.shiftKey && /^[1-9]$/.test(ev.key)) {
        ev.preventDefault();
        jumpToSlot(ev.key);
    }
}

app.registerExtension({
    name: "C2C.NodeBookmarks",
    async setup() {
        try {
            app.ui.settings.addSetting({
                id: SETTING_ID,
                name: "Node bookmarks strip (Ctrl+B, Ctrl+Shift+1..9)",
                type: "boolean", defaultValue: true,
                category: ["c2c", "Overlays", "Bookmarks"],
                onChange: render,
            });
        } catch { /* */ }
        window.addEventListener("keydown", onKey, true);
        // Render after each graph load/change.
        const _origLoad = app.loadGraphData?.bind(app);
        if (typeof _origLoad === "function") {
            app.loadGraphData = function (...args) {
                let r;
                try { r = _origLoad(...args); }
                catch (err) {
                    log.warn("[node_bookmarks] loadGraphData upstream threw", err);
                    throw err;
                }
                setTimeout(render, 50);
                return r;
            };
        }
        setInterval(render, 1500);  // cheap; only re-renders 9 cells
        render();
        console.log("[C2C.NodeBookmarks] ready.");
    },
});
