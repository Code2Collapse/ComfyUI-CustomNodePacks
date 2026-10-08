/**
 * mec_autolayout.js — Phase 18b: Auto-layout (graph "Tidy")
 *
 * Adds a "🧹 Tidy layout" entry to the canvas right-click menu. Performs a
 * simple longest-path layered (Sugiyama-style) layout over the directed
 * link graph. Pure JS, no dagre dependency. Runs only when asked: on the
 * selection, or on the whole graph after a confirmation; one undo step.
 *
 * Setting:
 *   mec.autolayout.enabled — bool (default true)
 *   mec.autolayout.col_gap — number (default 60)
 *   mec.autolayout.row_gap — number (default 40)
 */

import { app } from "../../scripts/app.js";
import { legacyCanvasMenu } from "./_c2c_compat.js";
import { c2cConfirm } from "./_c2c_dialog.js";
import { asOneUndoStep } from "./_c2c_undo_scope.js";

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

/** Every link of `g` as plain objects. LiteGraph 1.52 keeps links in a Map; `Object.keys()` of a Map is empty,
 *  which put every node in layer 0 and stacked the whole graph into one column. */
function _linkList(g) {
    const L = g?.links ?? g?._links;
    if (!L) return [];
    if (L instanceof Map) return [...L.values()].filter(Boolean);
    return Object.values(L).filter(Boolean);
}

/**
 * Returns Map<nodeId, layer>. Layer 0 = no inputs from the nodes being laid out.
 */
function _computeLayers(nodes, links) {
    const incoming = new Map();   // nodeId -> Set of upstream nodeIds
    const nodeIds = new Set(nodes.map(n => n.id));
    for (const n of nodes) incoming.set(n.id, new Set());
    for (const l of links) {
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

function _groupKids(group) {
    try { group.recomputeInsideNodes?.(); } catch { /* ignore */ }
    return Array.from(group._children ?? group.children ?? []).filter(c => c && c.id != null && c.pos && c.size);
}

const _titleH = () => LiteGraph?.NODE_TITLE_HEIGHT || 30;

/** [x, y, w, h] of a node including its title bar, in graph space. */
function _nodeBox(n) {
    const th = n.flags?.collapsed ? 0 : _titleH();
    return [n.pos[0], n.pos[1] - th, n.size?.[0] || 200, (n.size?.[1] || 100) + th];
}

/**
 * Layout units: every group whose members are all being tidied is ONE block (its members keep their arrangement
 * and move with it); every other node is its own unit. Laying out nodes one by one and refitting the group
 * frame afterwards pulled unrelated nodes inside the frame, so the next group drag carried them along.
 */
function _units(g, nodes) {
    const moving = new Set(nodes);
    const owner = new Map();               // node -> group unit
    const units = [];
    const groups = (g._groups || g.groups || []).filter(gr => !gr.pinned);
    // outer groups first, so a node in nested groups belongs to the outermost one
    groups.sort((p, q) => (q.size[0] * q.size[1]) - (p.size[0] * p.size[1]));
    for (const gr of groups) {
        const kids = _groupKids(gr).filter(k => g._nodes.includes(k));
        if (!kids.length || !kids.every(k => moving.has(k) && !owner.has(k))) continue;
        const u = { group: gr, nodes: kids, box: () => [gr.pos[0], gr.pos[1], gr.size[0], gr.size[1]] };
        for (const k of kids) owner.set(k, u);
        units.push(u);
    }
    for (const n of nodes) if (!owner.has(n)) {
        const u = { node: n, nodes: [n], box: () => _nodeBox(n) };
        owner.set(n, u);
        units.push(u);
    }
    return { units, owner };
}

/** Move a unit so its box's top-left lands on (x, y). */
function _placeUnit(u, x, y) {
    const [bx, by] = u.box();
    const dx = x - bx, dy = y - by;
    if (!dx && !dy) return;
    if (u.group) {
        u.group.move(dx, dy, true);                       // the frame only; members are moved below
        for (const n of u.nodes) n.pos = [n.pos[0] + dx, n.pos[1] + dy];
    } else {
        u.node.pos = [u.node.pos[0] + dx, u.node.pos[1] + dy];
    }
}

/** Lay out `nodes` left to right by data flow, keeping the top-left corner they already occupy. */
function _layout(g, nodes) {
    const { units, owner } = _units(g, nodes);
    const ids = new Map(units.map((u, i) => [u, i]));
    const pseudo = units.map((u, i) => ({ id: i }));
    const links = [];
    for (const l of _linkList(g)) {
        const a = owner.get(g.getNodeById(l.origin_id)), b = owner.get(g.getNodeById(l.target_id));
        if (a && b && a !== b) links.push({ origin_id: ids.get(a), target_id: ids.get(b) });
    }
    const layer = _computeLayers(pseudo, links);
    const { col, row } = _gaps();
    const boxes = units.map(u => u.box());
    const x0 = Math.min(...boxes.map(b => b[0]));
    const y0 = Math.min(...boxes.map(b => b[1]));
    const layers = new Map();
    units.forEach((u, i) => {
        const L = layer.get(i) ?? 0;
        if (!layers.has(L)) layers.set(L, []);
        layers.get(L).push(i);
    });
    const sorted = Array.from(layers.keys()).sort((a, b) => a - b);
    let xCursor = x0;
    for (const L of sorted) {
        // within a column keep the current vertical order, to minimise churn
        const column = layers.get(L).sort((a, b) => boxes[a][1] - boxes[b][1]);
        let yCursor = y0;
        let maxW = 0;
        for (const i of column) {
            _placeUnit(units[i], xCursor, yCursor);
            yCursor += boxes[i][3] + row;
            maxW = Math.max(maxW, boxes[i][2]);
        }
        xCursor += maxW + col;
    }
    return { layers: sorted.length, groups: units.filter(u => u.group).length };
}

/**
 * Tidy the selected nodes (two or more), or - after a confirmation - every node of the graph on screen.
 * Explicit only: nothing calls this except the menu entry and the command. Pinned nodes and pinned groups stay
 * where they are; a group moves as one block. The whole operation is ONE undo step. (A9: the old version moved
 * every node of the ROOT graph to (0,0) without asking, dropped nodes out of their groups and read links as a
 * plain object, which put the whole graph into a single column.)
 */
async function _tidy() {
    const canvas = app.canvas;
    const g = canvas?.graph || app.graph;
    if (!g?._nodes?.length) return;
    const selected = Object.values(canvas?.selected_nodes || {}).filter(n => g._nodes.includes(n));
    let nodes = selected.length >= 2 ? selected : g._nodes.slice();
    nodes = nodes.filter(n => !n.pinned && n.pos && n.size);
    if (selected.length < 2) {
        // whole graph: a group that cannot move as one block (pinned, or holding a pinned node) stays where it
        // is with all its members, instead of losing them to the layout
        const fixed = new Set();
        for (const gr of (g._groups || g.groups || [])) {
            const kids = _groupKids(gr).filter(k => g._nodes.includes(k));
            if (gr.pinned || kids.some(k => k.pinned)) for (const k of kids) fixed.add(k);
        }
        nodes = nodes.filter(n => !fixed.has(n));
    }
    if (nodes.length < 2) return;
    if (selected.length < 2) {
        const ok = await c2cConfirm(
            `Tidy moves all ${nodes.length} unpinned nodes in this graph into columns by data flow. ` +
            "Each group moves as one block.\n\n" +
            "To tidy only part of the graph, select those nodes first. Ctrl+Z undoes the tidy in one step.",
            { title: "Tidy layout", okLabel: "Tidy all nodes", cancelLabel: "Cancel" });
        if (!ok) return;
    }
    let r = { layers: 0, groups: 0 };
    asOneUndoStep(canvas, () => { r = _layout(g, nodes); });
    canvas?.setDirty?.(true, true);
    console.log(`[C2C.Autolayout] Tidied ${nodes.length} nodes (${r.groups} groups as blocks) across ${r.layers} columns.`);
}

function _canvasMenuItems() {
    if (!_settingsEnabled()) return [];
    const n = Object.keys(app.canvas?.selected_nodes || {}).length;
    return [null, { content: n >= 2 ? `🧹 Tidy layout (${n} selected nodes)` : "🧹 Tidy layout (all nodes…)", callback: _tidy }];
}

function _mergeCanvasMenuItems(opts) {
    const items = _canvasMenuItems();
    if (items.length) opts.push(...items);
    return opts;
}

app.registerExtension({
    name: "C2C.Autolayout",
    settings: [
        {
            id: "mec.autolayout.enabled",
            name: "Auto-layout: show Tidy in the canvas right-click menu (never runs on its own)",
            type: "boolean",
            defaultValue: true,
        },
        {
            id: "mec.autolayout.col_gap",
            name: "Auto-layout: horizontal gap (px)",
            type: "number",
            defaultValue: 60,
        },
        {
            id: "mec.autolayout.row_gap",
            name: "Auto-layout: vertical gap (px)",
            type: "number",
            defaultValue: 40,
        },
    ],
    commands: [
        { id: "mec.autolayout.tidy", label: "🧹 MEC: Tidy layout", function: _tidy },
    ],
    getCanvasMenuItems() {
        return _canvasMenuItems();
    },
    async setup() {
        legacyCanvasMenu("autolayout", _mergeCanvasMenuItems);
        console.log("[MEC.Autolayout] Loaded.");
    },
});
