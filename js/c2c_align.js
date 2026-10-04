/**
 * c2c_align.js — C2C Align: smart align commands, distribute, equal size, drag guides.
 */
import { app } from "../../scripts/app.js";
import {
    smartAlign,
    alignUnits,
    distribute,
    equalSize,
    snapMove,
    unionRect,
    nodeRect,
    nodePosFromRect,
    groupRect,
    nodeTitleHeight,
} from "./c2c_align_core.js";

const GUIDE_COLOR = "#b494ff";
const SETTING_GUIDES = "c2c.align.guides";
const SETTING_SNAP_PX = "c2c.align.snapPx";
const SETTING_MIN_GAP = "c2c.align.minGap";

const CMD = {
    SMART: "C2C.Align.Smart",
    LEFT: "C2C.Align.Left",
    RIGHT: "C2C.Align.Right",
    TOP: "C2C.Align.Top",
    BOTTOM: "C2C.Align.Bottom",
    CENTER_X: "C2C.Align.CenterX",
    CENTER_Y: "C2C.Align.CenterY",
    DIST_H: "C2C.Align.DistributeH",
    DIST_V: "C2C.Align.DistributeV",
    EQ_W: "C2C.Align.EqualWidth",
    EQ_H: "C2C.Align.EqualHeight",
    TOGGLE_GUIDES: "C2C.Align.ToggleGuides",
};

function setting(id, fallback) {
    try { return app.ui?.settings?.getSettingValue?.(id, fallback); }
    catch { return fallback; }
}

function guidesEnabled() {
    return setting(SETTING_GUIDES, true) !== false;
}

function snapPx() {
    const v = +setting(SETTING_SNAP_PX, 8);
    return Number.isFinite(v) && v > 0 ? v : 8;
}

function minGap() {
    const v = +setting(SETTING_MIN_GAP, 30);
    return Number.isFinite(v) && v >= 0 ? v : 30;
}

function lgOpts() {
    const LG = window.LiteGraph ?? {};
    return { noTitle: LG.NO_TITLE ?? 1, titleHeight: LG.NODE_TITLE_HEIGHT ?? 30 };
}

// Duck-typed on purpose. node.pos is a typed-array view in this frontend, not
// an Array (Array.isArray was false for every node, so every command silently
// did nothing), and class names are not a contract across frontend builds.
function isGroup(item) {
    return !!item && typeof item.recomputeInsideNodes === "function" && "_children" in item;
}

function isReroute(item) {
    return !!item && ("posInternal" in item || "linkIds" in item) && !isGroup(item);
}

function isNode(item) {
    return !!item && !isGroup(item) && !isReroute(item)
        && item.pos?.length >= 2 && item.size?.length >= 2 && typeof item.computeSize === "function";
}

function unitKey(item) {
    return isGroup(item) ? `g:${item.id}` : `n:${item.id}`;
}

function rectOf(item) {
    if (isGroup(item)) return groupRect(item);
    return nodeRect(item, lgOpts());
}

function groupContainsNode(group, node) {
    try { group.recomputeInsideNodes?.(); } catch { /* ignore */ }
    const kids = group._nodes ?? group._children ?? group.children ?? [];
    return kids.includes?.(node) ?? kids.some?.((c) => c === node);
}

function collectUnits() {
    const canvas = app.canvas;
    if (!canvas?.selectedItems?.size) return [];

    const selected = Array.from(canvas.selectedItems);
    const selectedGroups = selected.filter(isGroup);
    const units = [];

    for (const item of selected) {
        if (isReroute(item)) continue;
        if (!isGroup(item) && !isNode(item)) continue;
        if (item.pinned) continue;
        if (isNode(item) && selectedGroups.some((g) => groupContainsNode(g, item))) continue;
        if (isGroup(item)) {
            try { item.recomputeInsideNodes?.(); } catch { /* ignore */ }
        }
        const r = rectOf(item);
        units.push({
            id: unitKey(item),
            x: r.x, y: r.y, w: r.w, h: r.h,
            ref: item,
        });
    }
    return units;
}

function unitsToRects(units) {
    return units.map(({ id, x, y, w, h }) => ({ id, x, y, w, h }));
}

/** Move a group and everything inside it. `_children` already holds every
 *  descendant (nested groups' nodes included), so one flat pass moves each
 *  item once. Nodes go through the pos setter (layout store); groups and
 *  reroutes through move(..., skipChildren). */
function moveGroupDeep(group, dx, dy) {
    if (!dx && !dy) return;
    try { group.recomputeInsideNodes?.(); } catch { /* ignore */ }
    const kids = Array.from(group._children ?? group.children ?? []);
    group.move(dx, dy, true);
    for (const c of kids) {
        if (c?.pinned) continue;
        if (isNode(c)) c.pos = [c.pos[0] + dx, c.pos[1] + dy];
        else if (typeof c?.move === "function") c.move(dx, dy, true);
    }
}

function applyMoves(units, moves) {
    const byId = new Map(moves.map((m) => [m.id, m]));
    for (const u of units) {
        const m = byId.get(u.id);
        if (!m || (!m.dx && !m.dy)) continue;
        const item = u.ref;
        if (isGroup(item)) moveGroupDeep(item, m.dx, m.dy);
        else item.pos = [item.pos[0] + m.dx, item.pos[1] + m.dy];
    }
}

function applyEqualSizes(units, sizes) {
    const byId = new Map(sizes.map((s) => [s.id, s]));
    const opts = lgOpts();
    for (const u of units) {
        const s = byId.get(u.id);
        if (!s) continue;
        const item = u.ref;
        if (isGroup(item)) {
            if (item.size?.length >= 2) item.size = [s.w, s.h];   // typed view, not an Array
        } else {
            const titleH = nodeTitleHeight(item, opts);
            const collapsed = !!item.flags?.collapsed;
            const bodyH = collapsed ? 0 : Math.max(0, s.h - titleH);
            const w = collapsed ? (item._collapsed_width ?? s.w) : s.w;
            const size = [w, collapsed ? (item.size?.[1] ?? 0) : bodyH];
            if (typeof item.setSize === "function") item.setSize(size);
            else if (item.size?.length >= 2) item.size = size;
        }
    }
}

function changeTracker() {
    return app.extensionManager?.workflow?.activeWorkflow?.changeTracker ?? null;
}

/** One undo step. The tracker's "before" is whatever it captured last, which
 *  may predate edits made without a key or mouse event - so capture it here
 *  first, then again after the change. */
function withUndo(fn) {
    const graph = app.graph;
    if (!graph) return;
    const ct = changeTracker();
    try { ct?.captureCanvasState?.(); } catch { /* ignore */ }
    try { graph.beforeChange?.(); } catch { /* ignore */ }
    try { fn(); } finally {
        try { graph.afterChange?.(); } catch { /* ignore */ }
        try { ct?.captureCanvasState?.(); } catch { /* ignore */ }
        try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
    }
}

function runAlign(mode) {
    const units = collectUnits();
    if (units.length < 2) return;
    const rects = unitsToRects(units);
    let result;
    if (mode === "smart") result = smartAlign(rects, { minGap: minGap() });
    else if (mode === "distH") result = distribute(rects, "x", { minGap: minGap() });
    else if (mode === "distV") result = distribute(rects, "y", { minGap: minGap() });
    else result = alignUnits(rects, mode);
    withUndo(() => applyMoves(units, result.moves));
}

function runEqual(dim) {
    const units = collectUnits();
    if (!units.length) return;
    const sizes = equalSize(unitsToRects(units), dim);
    withUndo(() => applyEqualSizes(units, sizes));
}

function toggleGuides() {
    const cur = guidesEnabled();
    try {
        app.extensionManager?.setting?.set?.(SETTING_GUIDES, !cur);
    } catch {
        try { app.ui?.settings?.setSettingValue?.(SETTING_GUIDES, !cur); } catch { /* ignore */ }
    }
}

function vueDraggingFlag() {
    const ls = app.extensionManager?.layoutStore
        ?? window.comfyAPI?.layoutStore;
    const flag = ls?.isDraggingVueNodes;
    if (flag && typeof flag === "object" && "value" in flag) return !!flag.value;
    return !!flag;
}

function classicDragging() {
    const c = app.canvas;
    return !!(c?.state?.draggingItems ?? c?.isDragging);
}

function pointerToGraph(e) {
    const canvas = app.canvas;
    if (!canvas) return { x: 0, y: 0 };
    if (typeof canvas.convertEventToCanvasOffset === "function") {
        const p = canvas.convertEventToCanvasOffset(e);
        return { x: p[0], y: p[1] };
    }
    const r = canvas.canvas.getBoundingClientRect();
    const ds = canvas.ds;
    return {
        x: (e.clientX - r.left) / ds.scale - ds.offset[0] / ds.scale,
        y: (e.clientY - r.top) / ds.scale - ds.offset[1] / ds.scale,
    };
}

function visibleGraphBounds() {
    const canvas = app.canvas;
    if (!canvas?.ds) return null;
    const ds = canvas.ds;
    const el = canvas.canvas;
    const w = el.clientWidth / ds.scale;
    const h = el.clientHeight / ds.scale;
    return {
        x: -ds.offset[0] / ds.scale,
        y: -ds.offset[1] / ds.scale,
        w, h,
    };
}

function rectsOverlap(a, b) {
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

function collectGroupDescendantKeys(group, excluded) {
    try { group.recomputeInsideNodes?.(); } catch { /* ignore */ }
    const kids = group._nodes ?? group._children ?? group.children ?? [];
    for (const child of kids) {
        if (isGroup(child)) {
            excluded.add(unitKey(child));
            collectGroupDescendantKeys(child, excluded);
        } else if (isNode(child)) {
            excluded.add(unitKey(child));
        }
    }
}

function excludedTargetKeys(movingUnits) {
    const excluded = new Set(movingUnits.map((u) => u.id));
    for (const u of movingUnits) {
        if (isGroup(u.ref)) collectGroupDescendantKeys(u.ref, excluded);
    }
    return excluded;
}

function collectTargetRects(movingUnits) {
    const vis = visibleGraphBounds();
    const excluded = excludedTargetKeys(movingUnits);
    const out = [];
    const graph = app.graph;
    if (!graph) return out;

    const tryAdd = (item) => {
        if (!item) return;
        const key = unitKey(item);
        if (excluded.has(key)) return;
        if (!isNode(item) && !isGroup(item)) return;
        const r = rectOf(item);
        const rect = { id: key, x: r.x, y: r.y, w: r.w, h: r.h };
        if (vis && !rectsOverlap(rect, vis)) return;
        out.push(rect);
    };

    for (const n of graph._nodes ?? []) tryAdd(n);
    for (const g of graph._groups ?? []) tryAdd(g);
    return out;
}

function isLgNodeHeaderTarget(target, clientY) {
    const el = target?.closest?.(".lg-node");
    if (!el) return false;
    const titleH = window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
    const rect = el.getBoundingClientRect();
    return (clientY - rect.top) <= titleH + 2;
}

function guidesBypassed(e) {
    // Shift = core snap-to-grid while dragging, so guides step aside. (Not
    // Ctrl: a Ctrl-drag does not move nodes in ComfyUI at all.) Otherwise
    // guides are switched with the setting / C2C.Align.ToggleGuides.
    return e.shiftKey || !guidesEnabled();
}

/** @type {null | { pointerId: number, ox: number, oy: number, units: object[] | null, starts: Map<string,{x:number,y:number,w:number,h:number}>, targets: object[], vueHeader: boolean, moved: boolean, beforeChangeDone: boolean }} */
let _drag = null;
/** @type {object[]} */
let _guides = [];

function clearDrag() {
    if (_writeRaf) { cancelAnimationFrame(_writeRaf); _writeRaf = 0; _pendingWrite = null; }
    _drag = null;
    _guides = [];
    try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
}

function setUnitAbsolute(u, x, y) {
    const item = u.ref;
    if (isGroup(item)) {
        const cur = groupRect(item);
        const dx = x - cur.x;
        const dy = y - cur.y;
        if (!dx && !dy) return;
        moveGroupDeep(item, dx, dy);
    } else {
        const titleH = nodeTitleHeight(item, lgOpts());
        const pos = nodePosFromRect({ x, y, w: u.w, h: u.h }, titleH);
        if (item.pos[0] === pos[0] && item.pos[1] === pos[1]) return;
        item.pos = pos;
    }
    u.x = x;
    u.y = y;
}

function onGraphSurface(target) {
    const c = app.canvas?.canvas;
    return !!target && (target === c || !!target.closest?.(".lg-node"));
}

function snapshotPositions() {
    const snap = new Map();
    const graph = app.graph;
    for (const n of graph?._nodes ?? []) snap.set(unitKey(n), rectOf(n));
    for (const g of graph?._groups ?? []) snap.set(unitKey(g), rectOf(g));
    return snap;
}

function itemUnderPointer(e, p) {
    const el = e.target?.closest?.(".lg-node[data-node-id]");
    const graph = app.graph;
    if (el) return graph?.getNodeById?.(el.dataset.nodeId) ?? graph?.getNodeById?.(Number(el.dataset.nodeId)) ?? null;
    return graph?.getNodeOnPos?.(p.x, p.y) ?? graph?.getGroupOnPos?.(p.x, p.y) ?? null;
}

function onPointerDown(e) {
    if (e.button !== 0 || !onGraphSurface(e.target)) return;
    const p = pointerToGraph(e);
    _drag = {
        snapshot: snapshotPositions(),
        downItem: itemUnderPointer(e, p),
        pointerId: e.pointerId,
        ox: p.x,
        oy: p.y,
        units: null,
        starts: new Map(),
        targets: [],
        vueHeader: isLgNodeHeaderTarget(e.target, e.clientY),
        moved: false,
        beforeChangeDone: false,
    };
}

function ensureDragStarted(totalDx, totalDy) {
    if (!_drag || _drag.starts.size) return;

    if (!_drag.units) {
        let units = collectUnits();
        // Dragging an unselected node moves that node alone.
        const down = _drag.downItem;
        if (down && (isNode(down) || isGroup(down)) && !down.pinned
            && !units.some((u) => u.ref === down)
            && !units.some((u) => isGroup(u.ref) && groupContainsNode(u.ref, down))) {
            const r = rectOf(down);
            units = [{ id: unitKey(down), x: r.x, y: r.y, w: r.w, h: r.h, ref: down }];
        }
        _drag.units = units;
        if (!_drag.units.length) return;
    }

    for (const u of _drag.units) {
        const r = rectOf(u.ref);
        u.x = r.x;
        u.y = r.y;
        u.w = r.w;
        u.h = r.h;
        // Where it was at pointerdown. (Current minus the pointer delta was
        // wrong in Nodes 2.0, which starts moving only after its own drag
        // threshold - the node got pinned in place.)
        const snap = _drag.snapshot?.get(u.id);
        _drag.starts.set(u.id, snap
            ? { x: snap.x, y: snap.y, w: r.w, h: r.h }
            : { x: r.x - totalDx, y: r.y - totalDy, w: r.w, h: r.h });
    }
    _drag.targets = collectTargetRects(_drag.units);
    if (!_drag.beforeChangeDone) {
        _drag.beforeChangeDone = true;
        try { app.graph?.beforeChange?.(); } catch { /* ignore */ }
    }
}

function dragActive(e) {
    if (!_drag || e.pointerId !== _drag.pointerId) return false;
    if (classicDragging()) return true;
    if (vueDraggingFlag()) return true;
    if (_drag.vueHeader) {
        const p = pointerToGraph(e);
        const dx = (p.x - _drag.ox) * (app.canvas?.ds?.scale ?? 1);
        const dy = (p.y - _drag.oy) * (app.canvas?.ds?.scale ?? 1);
        if (Math.hypot(dx, dy) > 3) _drag.moved = true;
        return _drag.moved;
    }
    return false;
}

function onPointerMove(e) {
    if (!_drag || e.pointerId !== _drag.pointerId) return;

    if (guidesBypassed(e)) {
        if (_guides.length) {
            _guides = [];
            try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
        }
        return;
    }

    if (!dragActive(e)) return;

    const canvas = app.canvas;
    const ds = canvas?.ds;
    if (!ds) return;

    const p = pointerToGraph(e);
    const totalDx = p.x - _drag.ox;
    const totalDy = p.y - _drag.oy;
    ensureDragStarted(totalDx, totalDy);
    if (!_drag.units?.length || !_drag.starts.size) return;

    const startRects = _drag.units.map((u) => {
        const s = _drag.starts.get(u.id);
        return { id: u.id, x: s.x + totalDx, y: s.y + totalDy, w: s.w, h: s.h };
    });
    const rawUnion = unionRect(startRects);
    const threshold = snapPx() / ds.scale;
    const { dx, dy, guides } = snapMove(rawUnion, _drag.targets, { threshold });
    _guides = guides;

    const targetsNow = _drag.units.map((u) => {
        const s = _drag.starts.get(u.id);
        return [u, s.x + totalDx + dx, s.y + totalDy + dy];
    });
    _drag.last = targetsNow;
    applyDragPositions(targetsNow);
    try { canvas.setDirty?.(true, true); } catch { /* ignore */ }
}

/** Classic canvas: write now - LiteGraph moved the items synchronously in its
 *  own handler, which already ran. Nodes 2.0: Vue applies drag positions in
 *  its next animation frame, after this handler, and would overwrite ours;
 *  so write in a one-shot frame queued after Vue's (not a loop: one per move,
 *  coalesced). */
function applyDragPositions(list) {
    if (!window.LiteGraph?.vueNodesMode) {
        for (const [u, x, y] of list) setUnitAbsolute(u, x, y);
        return;
    }
    _pendingWrite = list;
    if (_writeRaf) return;
    _writeRaf = requestAnimationFrame(() => {
        _writeRaf = 0;
        const l = _pendingWrite;
        _pendingWrite = null;
        if (l) for (const [u, x, y] of l) setUnitAbsolute(u, x, y);
        try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
    });
}
let _writeRaf = 0;
let _pendingWrite = null;

function onPointerUp(e) {
    if (!_drag || e.pointerId !== _drag.pointerId) return;
    const last = _drag.last;
    const vue = !!window.LiteGraph?.vueNodesMode;
    if (vue && last && !guidesBypassed(e)) {
        // Vue commits its own (unsnapped) final position on release; put the
        // snapped one back after it, then close the undo step.
        const began = _drag.beforeChangeDone;
        setTimeout(() => requestAnimationFrame(() => {
            for (const [u, x, y] of last) setUnitAbsolute(u, x, y);
            if (began) { try { app.graph?.afterChange?.(); } catch { /* ignore */ } }
            try { changeTracker()?.captureCanvasState?.(); } catch { /* ignore */ }
            try { app.canvas?.setDirty?.(true, true); } catch { /* ignore */ }
        }), 0);
        clearDrag();
        return;
    }
    if (_drag.beforeChangeDone) {
        try { app.graph?.afterChange?.(); } catch { /* ignore */ }
    }
    clearDrag();
}

function drawGuides(ctx, _visibleArea) {
    if (!_guides.length) return;
    const scale = app.canvas?.ds?.scale ?? 1;
    ctx.save();
    ctx.strokeStyle = GUIDE_COLOR;
    ctx.fillStyle = GUIDE_COLOR;
    ctx.lineWidth = 1.5 / scale;   // 1 px vanished on the dark canvas
    const fontSize = 11 / scale;
    const tick = 4 / scale;

    for (const g of _guides) {
        if (g.kind === "align") {
            ctx.beginPath();
            if (g.axis === "x") {
                ctx.moveTo(g.at, g.from);
                ctx.lineTo(g.at, g.to);
                // end markers, so a short guide still reads as a guide
                ctx.moveTo(g.at - tick, g.from); ctx.lineTo(g.at + tick, g.from);
                ctx.moveTo(g.at - tick, g.to); ctx.lineTo(g.at + tick, g.to);
            } else {
                ctx.moveTo(g.from, g.at);
                ctx.lineTo(g.to, g.at);
                ctx.moveTo(g.from, g.at - tick); ctx.lineTo(g.from, g.at + tick);
                ctx.moveTo(g.to, g.at - tick); ctx.lineTo(g.to, g.at + tick);
            }
            ctx.stroke();
        } else if (g.kind === "spacing") {
            ctx.beginPath();
            if (g.axis === "x") {
                ctx.moveTo(g.from, g.at);
                ctx.lineTo(g.to, g.at);
            } else {
                ctx.moveTo(g.at, g.from);
                ctx.lineTo(g.at, g.to);
            }
            ctx.stroke();
            if (g.label != null) {
                const mx = g.axis === "x" ? (g.from + g.to) / 2 : g.at;
                const my = g.axis === "y" ? (g.from + g.to) / 2 : g.at;
                const text = String(Math.round(g.label));
                ctx.font = `${fontSize}px sans-serif`;
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(text, mx, my - 6 / scale);
            }
        }
    }
    ctx.restore();
}

function chainDrawForeground() {
    const canvas = app.canvas;
    if (!canvas || canvas._c2cAlignDrawPatched) return;
    const prev = canvas.onDrawForeground;
    canvas.onDrawForeground = function (ctx, visibleArea) {
        if (typeof prev === "function") prev.call(this, ctx, visibleArea);
        try { drawGuides(ctx, visibleArea); } catch { /* ignore */ }
    };
    canvas._c2cAlignDrawPatched = true;
}

function menuEntries() {
    return [
        { content: "Smart align (auto)", callback: () => runAlign("smart") },
        null,
        { content: "Align left", callback: () => runAlign("left") },
        { content: "Align right", callback: () => runAlign("right") },
        { content: "Align top", callback: () => runAlign("top") },
        { content: "Align bottom", callback: () => runAlign("bottom") },
        { content: "Centre vertically (line)", callback: () => runAlign("centerX") },
        { content: "Centre horizontally (line)", callback: () => runAlign("centerY") },
        null,
        { content: "Distribute horizontally", callback: () => runAlign("distH") },
        { content: "Distribute vertically", callback: () => runAlign("distV") },
        null,
        { content: "Equal width", callback: () => runEqual("width") },
        { content: "Equal height", callback: () => runEqual("height") },
        null,
        { content: "Toggle smart guides", callback: () => toggleGuides() },
    ];
}

function alignSubmenu() {
    if (collectUnits().length < 2) return [];
    return [{
        content: "C2C Align",
        submenu: { options: menuEntries() },
    }];
}

app.registerExtension({
    name: "C2C.Align",
    settings: [
        {
            id: SETTING_GUIDES,
            name: "C2C › Align › Smart guides while dragging",
            type: "boolean",
            defaultValue: true,
            category: ["c2c", "Canvas", "Align"],
        },
        {
            id: SETTING_SNAP_PX,
            name: "C2C › Align › Snap distance (screen px)",
            type: "number",
            defaultValue: 8,
            category: ["c2c", "Canvas", "Align"],
        },
        {
            id: SETTING_MIN_GAP,
            name: "C2C › Align › Minimum gap (graph units)",
            type: "number",
            defaultValue: 30,
            category: ["c2c", "Canvas", "Align"],
        },
    ],
    commands: [
        { id: CMD.SMART, label: "C2C: Smart align", function: () => runAlign("smart") },
        { id: CMD.LEFT, label: "C2C: Align left", function: () => runAlign("left") },
        { id: CMD.RIGHT, label: "C2C: Align right", function: () => runAlign("right") },
        { id: CMD.TOP, label: "C2C: Align top", function: () => runAlign("top") },
        { id: CMD.BOTTOM, label: "C2C: Align bottom", function: () => runAlign("bottom") },
        { id: CMD.CENTER_X, label: "C2C: Centre on vertical line", function: () => runAlign("centerX") },
        { id: CMD.CENTER_Y, label: "C2C: Centre on horizontal line", function: () => runAlign("centerY") },
        { id: CMD.DIST_H, label: "C2C: Distribute horizontally", function: () => runAlign("distH") },
        { id: CMD.DIST_V, label: "C2C: Distribute vertically", function: () => runAlign("distV") },
        { id: CMD.EQ_W, label: "C2C: Equal width", function: () => runEqual("width") },
        { id: CMD.EQ_H, label: "C2C: Equal height", function: () => runEqual("height") },
        { id: CMD.TOGGLE_GUIDES, label: "C2C: Toggle smart guides", function: () => toggleGuides() },
    ],
    keybindings: [
        { combo: { key: "a", alt: true }, commandId: CMD.SMART },
        { combo: { key: "ArrowLeft", alt: true, shift: true }, commandId: CMD.LEFT },
        { combo: { key: "ArrowRight", alt: true, shift: true }, commandId: CMD.RIGHT },
        { combo: { key: "ArrowUp", alt: true, shift: true }, commandId: CMD.TOP },
        { combo: { key: "ArrowDown", alt: true, shift: true }, commandId: CMD.BOTTOM },
        { combo: { key: "x", alt: true, shift: true }, commandId: CMD.CENTER_X },
        { combo: { key: "y", alt: true, shift: true }, commandId: CMD.CENTER_Y },
        { combo: { key: "h", alt: true, shift: true }, commandId: CMD.DIST_H },
        { combo: { key: "v", alt: true, shift: true }, commandId: CMD.DIST_V },
        { combo: { key: "g", alt: true, shift: true }, commandId: CMD.TOGGLE_GUIDES },
    ],
    getCanvasMenuItems() {
        return alignSubmenu();
    },
    getNodeMenuItems(_node) {
        return alignSubmenu();
    },
    getSelectionToolboxCommands() {
        if (collectUnits().length < 2) return [];
        return [CMD.SMART, CMD.DIST_H, CMD.DIST_V];
    },
    async setup() {
        chainDrawForeground();
        window.addEventListener("pointerdown", onPointerDown, { capture: true, passive: true });
        window.addEventListener("pointermove", onPointerMove, { passive: true });
        window.addEventListener("pointerup", onPointerUp, { passive: true });
        window.addEventListener("pointercancel", onPointerUp, { passive: true });
    },
});
