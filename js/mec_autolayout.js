/**
 * mec_autolayout.js — Phase 18b: Auto-layout (graph "Tidy")
 *
 * Adds a "🧹 Tidy layout" entry to the canvas right-click menu. Performs a
 * simple longest-path layered (Sugiyama-style) layout over the directed
 * link graph. Pure JS, no dagre dependency.
 *
 * Setting:
 *   mec.autolayout.enabled — bool (default true)
 *   mec.autolayout.col_gap — number (default 60)
 *   mec.autolayout.row_gap — number (default 40)
 */

import { app } from "../../scripts/app.js";

function _settingsEnabled() {
    try { return app.ui.settings.getSettingValue("mec.autolayout.enabled", true); }
    catch { return true; }
}
function _gaps() {
    let col = 60, row = 40;
    try { col = app.ui.settings.getSettingValue("mec.autolayout.col_gap", 60); } catch { /* ignore */ }
    try { row = app.ui.settings.getSettingValue("mec.autolayout.row_gap", 40); } catch { /* ignore */ }
    return { col: Math.max(10, +col || 60), row: Math.max(10, +row || 40) };
}

/**
 * Returns Map<nodeId, layer>. Layer 0 = no inputs from in-graph nodes.
 */
function _computeLayers(nodes, links) {
    const incoming = new Map();   // nodeId -> Set of upstream nodeIds
    const nodeIds = new Set(nodes.map(n => n.id));
    for (const n of nodes) incoming.set(n.id, new Set());
    for (const lid of Object.keys(links || {})) {
        const l = links[lid];
        if (!l) continue;
        const { origin_id, target_id } = l;
        if (!nodeIds.has(origin_id) || !nodeIds.has(target_id)) continue;
        if (origin_id === target_id) continue;  // self-loop guard
        incoming.get(target_id).add(origin_id);
    }
    const layer = new Map();
    // Iteratively assign layer = 1 + max(layer of upstream); start with sources.
    let changed = true;
    let iter = 0;
    while (changed && iter < nodes.length + 5) {
        changed = false;
        iter++;
        for (const n of nodes) {
            const ups = incoming.get(n.id);
            if (!ups.size) {
                if (!layer.has(n.id) || layer.get(n.id) !== 0) {
                    layer.set(n.id, 0);
                    changed = true;
                }
                continue;
            }
            let maxUp = -1;
            let allKnown = true;
            for (const u of ups) {
                if (!layer.has(u)) { allKnown = false; break; }
                maxUp = Math.max(maxUp, layer.get(u));
            }
            if (allKnown) {
                const cand = maxUp + 1;
                if (!layer.has(n.id) || layer.get(n.id) !== cand) {
                    layer.set(n.id, cand);
                    changed = true;
                }
            }
        }
    }
    // Fallback for cycles: assign any unassigned to 0.
    for (const n of nodes) if (!layer.has(n.id)) layer.set(n.id, 0);
    return layer;
}

function _tidy() {
    const g = app.graph;
    if (!g || !g._nodes || !g._nodes.length) return;
    const nodes = g._nodes.slice();
    const layer = _computeLayers(nodes, g.links || {});
    const { col, row } = _gaps();

    // Group by layer.
    const layers = new Map();
    for (const n of nodes) {
        const L = layer.get(n.id) ?? 0;
        if (!layers.has(L)) layers.set(L, []);
        layers.get(L).push(n);
    }
    const sortedLayers = Array.from(layers.keys()).sort((a, b) => a - b);

    // Within each layer, preserve current vertical order to minimize churn.
    for (const L of sortedLayers) {
        layers.get(L).sort((a, b) => (a.pos?.[1] ?? 0) - (b.pos?.[1] ?? 0));
    }

    // Place left-to-right.
    let xCursor = 0;
    for (const L of sortedLayers) {
        const col_nodes = layers.get(L);
        const maxW = Math.max(...col_nodes.map(n => n.size?.[0] || 200));
        let yCursor = 0;
        for (const n of col_nodes) {
            n.pos = [xCursor, yCursor];
            yCursor += (n.size?.[1] || 100) + row;
        }
        xCursor += maxW + col;
    }
    app.canvas?.setDirty?.(true, true);
    console.log(`[MEC.Autolayout] Tidied ${nodes.length} nodes across ${sortedLayers.length} layers.`);
}

function _patchMenu() {
    if (LGraphCanvas.prototype._mecAutolayoutPatched) return;
    LGraphCanvas.prototype._mecAutolayoutPatched = true;
    const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        const opts = orig ? orig.apply(this, arguments) : [];
        if (!_settingsEnabled()) return opts;
        opts.push(null);
        opts.push({ content: "🧹 Tidy layout", callback: _tidy });
        return opts;
    };
}

app.registerExtension({
    name: "MEC.Autolayout",
    settings: [
        {
            id: "mec.autolayout.enabled",
            name: "Auto-layout: right-click canvas → Tidy",
            type: "boolean",
            default: true,
        },
        {
            id: "mec.autolayout.col_gap",
            name: "Auto-layout: horizontal gap (px)",
            type: "number",
            default: 60,
        },
        {
            id: "mec.autolayout.row_gap",
            name: "Auto-layout: vertical gap (px)",
            type: "number",
            default: 40,
        },
    ],
    commands: [
        { id: "mec.autolayout.tidy", label: "🧹 MEC: Tidy layout", function: _tidy },
    ],
    async setup() {
        _patchMenu();
        console.log("[MEC.Autolayout] Loaded.");
    },
});
