/**
 * SplineMaskEditorMEC – interactive spline drawing widget.
 *
 * Clean rewrite (May 2026):
 *   - HTML toolbar (no canvas-painted buttons).
 *   - Multiple paths supported (one active at a time).
 *   - spline_data JSON contract preserved:
 *       [ {points:[{x,y}], type, closed, handles?}, ... ]
 *
 * Interaction:
 *   Left click          add control point to active path
 *   Drag a point        move it
 *   Shift+click point   delete it
 *   Right-click point   delete it
 *   Alt / Middle drag   pan
 *   Wheel               zoom (cursor-anchored)
 *   N                   new path
 *   C                   toggle closed
 *   Delete              clear active path
 *   Ctrl+Z / Ctrl+Y     undo / redo
 *   F                   fit view
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "SplineMaskEditorMEC";

const COLOR = {
    bg: "#181825",
    border: "#313244",
    text: "#cdd6f4",
    sub: "#7f849c",
    paths: ["#a6e3a1", "#89b4fa", "#f9e2af", "#fab387", "#f5c2e7", "#94e2d5"],
};

const HISTORY_LIMIT = 80;
const POINT_HIT_PX = 8;

class Editor {
    constructor(node) {
        this.node = node;
        this.shapes = []; // {points:[{x,y}], type, closed, handles?}
        this.active = -1;

        this.canvasW = 512;
        this.canvasH = 512;

        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this._fitted = false;

        this.refImg = null;
        this.refUrl = null;

        this.hover = { shape: -1, point: -1 };
        this.drag = null;

        this.undoStack = [];
        this.redoStack = [];

        this.cursor = { x: 0, y: 0, visible: false };

        this.splineType = "catmull_rom";
        this.closedDefault = true;
        this.samplesPerSegment = 20;
    }

    snapshot() { return JSON.stringify({ s: this.shapes, a: this.active }); }
    pushUndo() {
        this.undoStack.push(this.snapshot());
        if (this.undoStack.length > HISTORY_LIMIT) this.undoStack.shift();
        this.redoStack.length = 0;
    }
    restore(s) {
        const o = JSON.parse(s);
        this.shapes = o.s || [];
        this.active = o.a ?? -1;
    }
    undo() {
        if (!this.undoStack.length) return false;
        this.redoStack.push(this.snapshot());
        this.restore(this.undoStack.pop()); return true;
    }
    redo() {
        if (!this.redoStack.length) return false;
        this.undoStack.push(this.snapshot());
        this.restore(this.redoStack.pop()); return true;
    }

    viewToCanvas(vx, vy) {
        return { x: (vx - this.panX) / this.zoom, y: (vy - this.panY) / this.zoom };
    }
    minZoom(vw, vh) {
        if (vw <= 0 || vh <= 0) return 0.05;
        return Math.min(vw / this.canvasW, vh / this.canvasH);
    }
    clampPan(vw, vh) {
        const dW = this.canvasW * this.zoom;
        const dH = this.canvasH * this.zoom;
        if (dW <= vw) this.panX = (vw - dW) / 2;
        else this.panX = Math.min(0, Math.max(vw - dW, this.panX));
        if (dH <= vh) this.panY = (vh - dH) / 2;
        else this.panY = Math.min(0, Math.max(vh - dH, this.panY));
    }
    setZoomAround(newZ, ax, ay, vw, vh) {
        const minZ = this.minZoom(vw, vh);
        newZ = Math.max(minZ, Math.min(8, newZ));
        const k = newZ / this.zoom;
        this.panX = ax - (ax - this.panX) * k;
        this.panY = ay - (ay - this.panY) * k;
        this.zoom = newZ;
        this.clampPan(vw, vh);
    }
    fitView(vw, vh) {
        if (vw <= 0 || vh <= 0) return;
        const pad = 8;
        const sx = (vw - pad * 2) / this.canvasW;
        const sy = (vh - pad * 2) / this.canvasH;
        this.zoom = Math.max(0.05, Math.min(sx, sy, 8));
        this.panX = (vw - this.canvasW * this.zoom) / 2;
        this.panY = (vh - this.canvasH * this.zoom) / 2;
        this._fitted = true;
    }

    newPath() {
        const closed = this.closedDefault;
        this.shapes.push({ points: [], type: this.splineType, closed });
        this.active = this.shapes.length - 1;
    }

    findPoint(cx, cy) {
        const r = POINT_HIT_PX / this.zoom;
        const r2 = r * r;
        for (let s = this.shapes.length - 1; s >= 0; s--) {
            const sh = this.shapes[s];
            for (let i = sh.points.length - 1; i >= 0; i--) {
                const p = sh.points[i];
                const d = (p.x - cx) ** 2 + (p.y - cy) ** 2;
                if (d < r2) return { shape: s, point: i };
            }
        }
        return { shape: -1, point: -1 };
    }

    save() {
        const w = this.node.widgets?.find(w => w.name === "spline_data");
        if (!w) return;
        const data = this.shapes.map(sh => ({
            points: sh.points.map(p => ({ x: +p.x.toFixed(2), y: +p.y.toFixed(2) })),
            type: sh.type || this.splineType,
            closed: !!sh.closed,
            ...(sh.handles ? { handles: sh.handles } : {}),
        }));
        w.value = JSON.stringify(data);
        if (this.node.graph) this.node.graph.setDirtyCanvas(true, false);
    }
    load() {
        const w = this.node.widgets?.find(w => w.name === "spline_data");
        if (!w?.value) return;
        try {
            const o = JSON.parse(w.value);
            if (Array.isArray(o)) {
                this.shapes = o.map(sh => ({
                    points: (sh.points || []).map(p => ({ x: +p.x, y: +p.y })),
                    type: sh.type || this.splineType,
                    closed: !!sh.closed,
                    ...(sh.handles ? { handles: sh.handles } : {}),
                }));
                this.active = this.shapes.length ? 0 : -1;
            }
        } catch (_) {}
    }

    setRefImage(url, ow, oh) {
        if (url === this.refUrl && this.refImg && this.refImg.complete) return;
        this.refUrl = url;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            this.refImg = img;
            const w = ow || img.naturalWidth;
            const h = oh || img.naturalHeight;
            if (w > 0 && h > 0) {
                this.canvasW = w; this.canvasH = h;
                const wW = this.node.widgets?.find(x => x.name === "width");
                const hW = this.node.widgets?.find(x => x.name === "height");
                if (wW && +wW.value !== w) wW.value = w;
                if (hW && +hW.value !== h) hW.value = h;
            }
            this._fitted = false;
            this.onLoaded?.();
        };
        img.src = url;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Spline sampling (Catmull-Rom, polyline) – used only for preview rendering
// ─────────────────────────────────────────────────────────────────────
function catmullRom(points, samplesPerSeg, closed) {
    const n = points.length;
    if (n < 2) return points.slice();
    if (n === 2) {
        const [a, b] = points;
        const out = [];
        for (let i = 0; i <= samplesPerSeg; i++) {
            const t = i / samplesPerSeg;
            out.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
        }
        return out;
    }
    const ext = closed
        ? [points[n - 1], ...points, points[0], points[1]]
        : [points[0], ...points, points[n - 1]];

    const out = [];
    const segs = closed ? n : n - 1;
    for (let i = 0; i < segs; i++) {
        const p0 = ext[i], p1 = ext[i + 1], p2 = ext[i + 2], p3 = ext[i + 3];
        // Centripetal alpha=0.5
        const a = 0.5;
        const t01 = Math.pow(Math.hypot(p1.x - p0.x, p1.y - p0.y), a) || 1e-6;
        const t12 = Math.pow(Math.hypot(p2.x - p1.x, p2.y - p1.y), a) || 1e-6;
        const t23 = Math.pow(Math.hypot(p3.x - p2.x, p3.y - p2.y), a) || 1e-6;
        const m1x = (p2.x - p0.x) - (p1.x - p0.x) * t12 / (t01 + t12) + (p2.x - p1.x) * t01 / (t01 + t12);
        const m1y = (p2.y - p0.y) - (p1.y - p0.y) * t12 / (t01 + t12) + (p2.y - p1.y) * t01 / (t01 + t12);
        const m2x = (p3.x - p1.x) - (p2.x - p1.x) * t23 / (t12 + t23) + (p3.x - p2.x) * t12 / (t12 + t23);
        const m2y = (p3.y - p1.y) - (p2.y - p1.y) * t23 / (t12 + t23) + (p3.y - p2.y) * t12 / (t12 + t23);
        const last = i === segs - 1;
        const steps = samplesPerSeg + (last ? 1 : 0);
        for (let s = 0; s < steps; s++) {
            const t = s / samplesPerSeg;
            const t2 = t * t, t3 = t2 * t;
            const h00 = 2 * t3 - 3 * t2 + 1;
            const h10 = t3 - 2 * t2 + t;
            const h01 = -2 * t3 + 3 * t2;
            const h11 = t3 - t2;
            out.push({
                x: h00 * p1.x + h10 * m1x * 0.5 + h01 * p2.x + h11 * m2x * 0.5,
                y: h00 * p1.y + h10 * m1y * 0.5 + h01 * p2.y + h11 * m2y * 0.5,
            });
        }
    }
    return out;
}

function sampleShape(sh, samplesPerSeg) {
    const pts = sh.points;
    if (pts.length < 2) return pts.slice();
    if (sh.type === "polyline") return pts.slice();
    return catmullRom(pts, samplesPerSeg, sh.closed);
}

function draw(ed, ctx, vw, vh) {
    ctx.save();
    ctx.fillStyle = COLOR.bg;
    ctx.fillRect(0, 0, vw, vh);

    const z = ed.zoom;
    ctx.translate(ed.panX, ed.panY);
    ctx.scale(z, z);

    if (ed.refImg && ed.refImg.complete) {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        try { ctx.drawImage(ed.refImg, 0, 0, ed.canvasW, ed.canvasH); } catch (_) {}
    } else {
        ctx.fillStyle = "#11111b";
        ctx.fillRect(0, 0, ed.canvasW, ed.canvasH);
    }
    ctx.strokeStyle = "#45475a";
    ctx.lineWidth = 1 / z;
    ctx.strokeRect(0, 0, ed.canvasW, ed.canvasH);

    for (let s = 0; s < ed.shapes.length; s++) {
        const sh = ed.shapes[s];
        const c = COLOR.paths[s % COLOR.paths.length];
        const isActive = s === ed.active;

        if (sh.points.length >= 2) {
            const sampled = sampleShape(sh, ed.samplesPerSegment);
            ctx.strokeStyle = c;
            ctx.lineWidth = (isActive ? 2.0 : 1.5) / z;
            ctx.fillStyle = c + "22";
            ctx.beginPath();
            ctx.moveTo(sampled[0].x, sampled[0].y);
            for (let i = 1; i < sampled.length; i++) ctx.lineTo(sampled[i].x, sampled[i].y);
            if (sh.closed) {
                ctx.closePath();
                ctx.fill();
            }
            ctx.stroke();
        }

        for (let i = 0; i < sh.points.length; i++) {
            const p = sh.points[i];
            const isHov = ed.hover.shape === s && ed.hover.point === i;
            const r = (isHov ? 6 : 4) / z;
            ctx.fillStyle = c;
            ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = "#000a";
            ctx.lineWidth = 1.5 / z;
            ctx.stroke();
            if (isActive && i === 0) {
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 1 / z;
                ctx.beginPath(); ctx.arc(p.x, p.y, r + 2 / z, 0, Math.PI * 2); ctx.stroke();
            }
        }
    }

    ctx.restore();
}

function findRefImage(node) {
    if (!node.inputs) return null;
    const linkedInput = node.inputs.find(i => i.name === "image" && i.link != null);
    if (!linkedInput) return null;
    const link = app.graph.links[linkedInput.link];
    if (!link) return null;

    const visited = new Set();
    const queue = [{ id: link.origin_id, depth: 0 }];
    while (queue.length) {
        const { id, depth } = queue.shift();
        if (visited.has(id)) continue;
        visited.add(id);
        const n = app.graph.getNodeById(id);
        if (!n) continue;
        if (n.imgs?.length) return { url: n.imgs[0].src };
        const w = n.widgets?.find(w => w.name === "image");
        if (w?.value && typeof w.value === "string") {
            const parts = w.value.split("/");
            const sub = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
            const fn = parts[parts.length - 1];
            return { url: `/view?filename=${encodeURIComponent(fn)}&subfolder=${encodeURIComponent(sub)}&type=input` };
        }
        if (depth < 3 && n.inputs) {
            for (const inp of n.inputs) {
                if (inp.link == null) continue;
                const li = app.graph.links[inp.link];
                if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
            }
        }
    }
    return null;
}

function installEditor(node) {
    const ed = new Editor(node);

    const hideWidget = (w) => {
        if (!w) return;
        w.type = "hidden";
        w.computeSize = () => [0, -4];
        w.draw = () => {};
        const el = w.element;
        if (el) {
            el.hidden = true;
            el.style.display = "none";
            const wrap = el.parentElement;
            if (wrap?.classList?.contains("dom-widget")) wrap.style.display = "none";
        }
    };
    hideWidget(node.widgets?.find(w => w.name === "spline_data"));

    const syncFromWidgets = () => {
        const t = node.widgets?.find(x => x.name === "spline_type");
        const c = node.widgets?.find(x => x.name === "closed");
        const sps = node.widgets?.find(x => x.name === "samples_per_segment");
        const w = node.widgets?.find(x => x.name === "width");
        const h = node.widgets?.find(x => x.name === "height");
        if (t) ed.splineType = t.value;
        if (c) ed.closedDefault = !!c.value;
        if (sps) ed.samplesPerSegment = +sps.value || 20;
        if (w && +w.value > 0) ed.canvasW = +w.value;
        if (h && +h.value > 0) ed.canvasH = +h.value;
    };
    syncFromWidgets();
    ed.load();

    const root = document.createElement("div");
    // Inset from node body — leaves a hit area for LiteGraph border / resize
    // corner so the user can grab edges and the bottom-right grip.
    root.style.cssText = `
        display:flex;flex-direction:column;
        width:calc(100% - 12px);height:calc(100% - 18px);
        margin:2px 6px 16px 6px;
        background:${COLOR.bg};border:1px solid ${COLOR.border};border-radius:6px;
        overflow:hidden;font-family:Inter,system-ui,sans-serif;color:${COLOR.text};
        box-sizing:border-box;user-select:none;
        pointer-events:none;
    `;

    const tb = document.createElement("div");
    tb.style.cssText = `
        display:flex;align-items:center;gap:4px;padding:4px 6px;
        background:linear-gradient(#22223a,#1a1a2e);
        border-bottom:1px solid ${COLOR.border};
        flex:0 0 auto;font-size:11px;line-height:1;
        pointer-events:auto;
    `;
    root.appendChild(tb);

    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText = "position:relative;flex:1 1 auto;min-height:0;overflow:hidden;cursor:crosshair;background:#11111b;pointer-events:auto;";
    root.appendChild(canvasWrap);

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;width:100%;height:100%;outline:none;";
    canvas.tabIndex = 0;
    canvasWrap.appendChild(canvas);

    const status = document.createElement("div");
    status.style.cssText = `
        position:absolute;left:6px;bottom:6px;padding:3px 7px;
        background:#1e1e2ed8;border:1px solid ${COLOR.border};border-radius:4px;
        font-size:10px;color:${COLOR.sub};pointer-events:none;
        font-family:ui-monospace,Menlo,monospace;letter-spacing:.2px;
    `;
    canvasWrap.appendChild(status);

    const ctx = canvas.getContext("2d");

    const mkIcon = (label, title, onClick) => {
        const b = document.createElement("button");
        b.textContent = label; b.title = title;
        b.style.cssText = `
            min-width:26px;height:24px;padding:0 7px;
            border:1px solid ${COLOR.border};border-radius:4px;
            background:#313244;color:${COLOR.text};
            font-size:11px;font-weight:500;cursor:pointer;
            display:inline-flex;align-items:center;justify-content:center;
            white-space:nowrap;line-height:1;flex:0 0 auto;
            transition:background .12s,border-color .12s;
        `;
        b.onmouseenter = () => { b.style.background = "#45475a"; b.style.borderColor = "#585b70"; };
        b.onmouseleave = () => { b.style.background = "#313244"; b.style.borderColor = COLOR.border; };
        b.onmousedown = (e) => e.stopPropagation();
        b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); onClick(b); render(); };
        return b;
    };

    const counter = document.createElement("span");
    counter.style.cssText = `
        color:${COLOR.text};font-size:10px;flex:1 1 auto;
        font-family:ui-monospace,Menlo,monospace;
        padding:3px 8px;background:#11111b;border:1px solid ${COLOR.border};
        border-radius:4px;letter-spacing:.3px;
        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
    `;
    tb.appendChild(counter);

    tb.appendChild(mkIcon("+ Path", "Add new path (N)", () => {
        ed.pushUndo(); ed.newPath(); ed.save();
    }));
    const closedBtn = mkIcon("◯", "Toggle closed (C)", () => {
        if (ed.active < 0) return;
        ed.pushUndo();
        ed.shapes[ed.active].closed = !ed.shapes[ed.active].closed;
        ed.save();
    });
    tb.appendChild(closedBtn);
    tb.appendChild(mkIcon("↶", "Undo (Ctrl+Z)", () => { if (ed.undo()) ed.save(); }));
    tb.appendChild(mkIcon("↷", "Redo (Ctrl+Y)", () => { if (ed.redo()) ed.save(); }));
    tb.appendChild(mkIcon("\u2796", "Zoom out (-)", () => {
        const r = canvasWrap.getBoundingClientRect();
        ed.setZoomAround(ed.zoom * 0.85, r.width / 2, r.height / 2, r.width, r.height);
    }));
    tb.appendChild(mkIcon("\u2795", "Zoom in (+)", () => {
        const r = canvasWrap.getBoundingClientRect();
        ed.setZoomAround(ed.zoom * 1.15, r.width / 2, r.height / 2, r.width, r.height);
    }));
    tb.appendChild(mkIcon("\u2B1B", "Fit image (0 / F)", () => {
        const r = canvasWrap.getBoundingClientRect(); ed.fitView(r.width, r.height);
    }));

    const menuBtn = mkIcon("≡", "More actions", () => toggleMenu());
    tb.appendChild(menuBtn);

    const menu = document.createElement("div");
    menu.style.cssText = `
        position:absolute;top:34px;right:6px;z-index:50;
        background:#1e1e2e;border:1px solid ${COLOR.border};border-radius:6px;
        box-shadow:0 6px 20px #000a;padding:4px;display:none;
        min-width:200px;font-size:11px;
    `;
    root.style.position = "relative";
    root.appendChild(menu);

    const mkMenuItem = (label, onClick, danger) => {
        const it = document.createElement("div");
        it.textContent = label;
        it.style.cssText = `
            padding:7px 12px;border-radius:4px;cursor:pointer;
            color:${danger ? "#f38ba8" : COLOR.text};white-space:nowrap;
        `;
        it.onmouseenter = () => it.style.background = "#313244";
        it.onmouseleave = () => it.style.background = "";
        it.onclick = (e) => { e.stopPropagation(); menu.style.display = "none"; onClick(); render(); };
        return it;
    };
    const mkMenuSep = () => {
        const s = document.createElement("div");
        s.style.cssText = `height:1px;background:${COLOR.border};margin:4px 2px;`;
        return s;
    };

    menu.appendChild(mkMenuItem("Reset zoom (100%)", () => { ed.zoom = 1; ed.panX = 0; ed.panY = 0; }));
    menu.appendChild(mkMenuItem("Fit to view", () => {
        const r = canvasWrap.getBoundingClientRect(); ed.fitView(r.width, r.height);
    }));
    menu.appendChild(mkMenuSep());
    menu.appendChild(mkMenuItem("Clear active path", () => {
        if (ed.active < 0) return;
        ed.pushUndo(); ed.shapes[ed.active].points = []; ed.save();
    }, true));
    menu.appendChild(mkMenuItem("Delete active path", () => {
        if (ed.active < 0) return;
        ed.pushUndo();
        ed.shapes.splice(ed.active, 1);
        ed.active = ed.shapes.length ? Math.min(ed.active, ed.shapes.length - 1) : -1;
        ed.save();
    }, true));
    menu.appendChild(mkMenuItem("Clear all paths", () => {
        if (!ed.shapes.length) return;
        ed.pushUndo(); ed.shapes = []; ed.active = -1; ed.save();
    }, true));

    function toggleMenu() {
        const open = menu.style.display === "none" || !menu.style.display;
        menu.style.display = open ? "block" : "none";
        if (open) {
            const off = (e) => {
                if (!menu.contains(e.target) && e.target !== menuBtn) {
                    menu.style.display = "none";
                    document.removeEventListener("mousedown", off, true);
                }
            };
            setTimeout(() => document.addEventListener("mousedown", off, true), 0);
        }
    }

    const widgetH = 460;
    node.addDOMWidget("spline_editor", "canvas", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => widgetH,
        getHeight: () => widgetH,
    });

    if (!node.size || node.size[0] < 600) {
        const h = node.size?.[1] || 640;
        node.setSize?.([600, Math.max(h, 640)]);
    }

    let lastW = 0, lastH = 0;
    function syncCanvasPx() {
        const r = canvasWrap.getBoundingClientRect();
        // Parent ComfyUI canvas applies CSS transform: scale(zoom) to DOM
        // widgets. Use offsetWidth/Height (unscaled layout pixels) to keep
        // the canvas resolution stable when the user zooms the graph.
        const cssW = canvasWrap.offsetWidth || r.width;
        const cssH = canvasWrap.offsetHeight || r.height;
        const w = Math.max(1, Math.round(cssW));
        const h = Math.max(1, Math.round(cssH));
        const dpr = window.devicePixelRatio || 1;
        const needPx = (canvas.width !== w * dpr) || (canvas.height !== h * dpr);
        if (w === lastW && h === lastH && !needPx) return false;
        lastW = w; lastH = h;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (!ed._fitted) ed.fitView(w, h);
        return true;
    }

    // Reflect "closed" state on the icon button
    function updateClosedBtn() {
        const sh = ed.active >= 0 ? ed.shapes[ed.active] : null;
        const isClosed = sh?.closed;
        closedBtn.textContent = isClosed ? "◉" : "◯";
        closedBtn.title = isClosed ? "Active path is CLOSED — click to open (C)"
                                   : "Active path is OPEN — click to close (C)";
    }

    let raf = 0;
    function render() {
        if (raf) return;
        raf = requestAnimationFrame(() => {
            raf = 0;
            syncCanvasPx();
            draw(ed, ctx, lastW, lastH);
            const totalPts = ed.shapes.reduce((a, s) => a + s.points.length, 0);
            const activePts = ed.active >= 0 ? ed.shapes[ed.active].points.length : 0;
            const closed = ed.active >= 0 ? (ed.shapes[ed.active].closed ? "closed" : "open") : "—";
            counter.textContent = `paths ${ed.shapes.length} · active ${ed.active >= 0 ? ed.active + 1 : "—"}/${ed.shapes.length} · pts ${activePts}/${totalPts} · ${closed}`;
            status.textContent = `${ed.canvasW}×${ed.canvasH} · zoom ${(ed.zoom * 100).toFixed(0)}% · ${ed.splineType}`;
            updateClosedBtn();
        });
    }
    ed.onLoaded = () => { ed._fitted = false; render(); };

    const ro = new ResizeObserver(() => render());
    ro.observe(canvasWrap);

    // ComfyUI's outer canvas applies a CSS scale transform when the user
    // zooms the graph. getBoundingClientRect() returns the *visual*
    // (post-transform) box; the canvas's logical pixels are unscaled.
    // Multiply mouse delta by (offsetWidth / rect.width) to undo the
    // parent scale so clicks land exactly under the cursor at any zoom.
    function _scale(r, el) {
        return {
            sx: (el.offsetWidth  || r.width)  / Math.max(1, r.width),
            sy: (el.offsetHeight || r.height) / Math.max(1, r.height),
        };
    }
    function eventCanvas(e) {
        const r = canvas.getBoundingClientRect();
        const { sx, sy } = _scale(r, canvas);
        return ed.viewToCanvas((e.clientX - r.left) * sx, (e.clientY - r.top) * sy);
    }
    function eventClient(e) {
        const r = canvas.getBoundingClientRect();
        const { sx, sy } = _scale(r, canvas);
        return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
    }

    canvas.addEventListener("pointerdown", (e) => {
        e.preventDefault(); canvas.focus();
        canvas.setPointerCapture(e.pointerId);
        const c = eventCanvas(e);

        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            const ec = eventClient(e);
            ed.drag = { kind: "pan", startX: ec.x - ed.panX, startY: ec.y - ed.panY };
            canvas.style.cursor = "grabbing";
            return;
        }
        const hit = ed.findPoint(c.x, c.y);

        if (e.button === 2) {
            // right-click → delete pt under cursor
            if (hit.shape >= 0) {
                ed.pushUndo();
                ed.shapes[hit.shape].points.splice(hit.point, 1);
                if (ed.shapes[hit.shape].points.length === 0) {
                    ed.shapes.splice(hit.shape, 1);
                    ed.active = ed.shapes.length ? Math.min(ed.active, ed.shapes.length - 1) : -1;
                }
                ed.save(); render();
            }
            return;
        }
        if (e.shiftKey && hit.shape >= 0) {
            ed.pushUndo();
            ed.shapes[hit.shape].points.splice(hit.point, 1);
            ed.save(); render(); return;
        }
        if (hit.shape >= 0) {
            ed.pushUndo();
            ed.active = hit.shape;
            ed.drag = { kind: "point", shape: hit.shape, point: hit.point };
            return;
        }
        // Add new point on click (creates path if none exists)
        ed.pushUndo();
        if (ed.active < 0 || ed.active >= ed.shapes.length) ed.newPath();
        ed.shapes[ed.active].points.push({ x: +c.x.toFixed(2), y: +c.y.toFixed(2) });
        ed.save(); render();
    });

    canvas.addEventListener("pointermove", (e) => {
        const c = eventCanvas(e);
        ed.cursor = { x: c.x, y: c.y, visible: true };

        if (ed.drag?.kind === "pan") {
            const ec = eventClient(e);
            ed.panX = ec.x - ed.drag.startX;
            ed.panY = ec.y - ed.drag.startY;
            ed.clampPan(lastW, lastH);
            render(); return;
        }
        if (ed.drag?.kind === "point") {
            const sh = ed.shapes[ed.drag.shape];
            if (sh) {
                const p = sh.points[ed.drag.point];
                if (p) { p.x = +c.x.toFixed(2); p.y = +c.y.toFixed(2); render(); }
            }
            return;
        }
        const hit = ed.findPoint(c.x, c.y);
        if (hit.shape !== ed.hover.shape || hit.point !== ed.hover.point) {
            ed.hover = hit;
            canvas.style.cursor = hit.shape >= 0 ? "pointer" : "crosshair";
        }
        // Always render so the cursor indicator follows smoothly.
        // render() coalesces via RAF, so this stays cheap.
        render();
    });

    canvas.addEventListener("pointerup", (e) => {
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
        if (ed.drag?.kind === "pan") {
            ed.drag = null; canvas.style.cursor = "crosshair"; return;
        }
        if (ed.drag?.kind === "point") {
            ed.drag = null; ed.save(); render(); return;
        }
    });

    canvas.addEventListener("pointerleave", () => {
        ed.cursor.visible = false;
        ed.hover = { shape: -1, point: -1 };
        render();
    });

    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    // Mouse-wheel zoom is intentionally disabled. Zoom via toolbar buttons
    // or +/-/0 keys. Swallow the wheel so the parent ComfyUI graph canvas
    // does not zoom while the user hovers our editor.
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        e.stopPropagation();
    }, { passive: false });

    canvas.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
            e.preventDefault();
            (e.shiftKey ? ed.redo() : ed.undo()) && ed.save(); render();
        } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
            e.preventDefault(); ed.redo() && ed.save(); render();
        } else if (e.key === "Delete" || e.key === "Backspace") {
            e.preventDefault();
            if (ed.hover.shape >= 0) {
                ed.pushUndo();
                ed.shapes[ed.hover.shape].points.splice(ed.hover.point, 1);
                ed.hover = { shape: -1, point: -1 };
                ed.save(); render();
            }
        } else if (e.key.toLowerCase() === "n") {
            ed.pushUndo(); ed.newPath(); ed.save(); render();
        } else if (e.key.toLowerCase() === "c") {
            if (ed.active >= 0) {
                ed.pushUndo();
                ed.shapes[ed.active].closed = !ed.shapes[ed.active].closed;
                ed.save(); render();
            }
        } else if (e.key.toLowerCase() === "f" || e.key === "0") {
            e.preventDefault();
            const r = canvasWrap.getBoundingClientRect();
            ed.fitView(r.width, r.height); render();
        } else if (e.key === "+" || e.key === "=") {
            e.preventDefault();
            const r = canvasWrap.getBoundingClientRect();
            ed.setZoomAround(ed.zoom * 1.15, r.width / 2, r.height / 2, r.width, r.height);
            render();
        } else if (e.key === "-" || e.key === "_") {
            e.preventDefault();
            const r = canvasWrap.getBoundingClientRect();
            ed.setZoomAround(ed.zoom * 0.85, r.width / 2, r.height / 2, r.width, r.height);
            render();
        }
    });

    // React to widget edits
    for (const wn of ["spline_type", "closed", "samples_per_segment", "width", "height"]) {
        const w = node.widgets?.find(x => x.name === wn);
        if (!w) continue;
        const orig = w.callback;
        w.callback = function (v) {
            orig?.call(this, v);
            syncFromWidgets();
            // Apply spline_type / closed change to ALL existing shapes
            if (wn === "spline_type") {
                for (const sh of ed.shapes) sh.type = ed.splineType;
                ed.save();
            } else if (wn === "closed") {
                // do not clobber per-shape closed flag automatically; only newly added paths use it
            } else if (wn === "width" || wn === "height") {
                ed._fitted = false;
            }
            render();
        };
    }

    function tryDiscoverRef() {
        const r = findRefImage(node);
        if (r?.url && r.url !== ed.refUrl) ed.setRefImage(r.url);
    }
    const origConn = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        origConn?.apply(this, args);
        setTimeout(tryDiscoverRef, 200);
    };
    const origExec = node.onExecuted;
    node.onExecuted = function (out) {
        origExec?.apply(this, arguments);
        if (out?.preview_b64?.[0]) {
            ed.setRefImage("data:image/jpeg;base64," + out.preview_b64[0]);
        } else {
            tryDiscoverRef();
        }
    };
    const refPoll = setInterval(() => { if (!ed.refImg) tryDiscoverRef(); }, 1500);

    const origRemoved = node.onRemoved;
    node.onRemoved = function () {
        origRemoved?.apply(this, arguments);
        clearInterval(refPoll); ro.disconnect();
        if (raf) cancelAnimationFrame(raf);
    };

    setTimeout(render, 50);
    setTimeout(tryDiscoverRef, 100);

    node._mecEditor = ed;
    node._mecRender = render;
}

app.registerExtension({
    name: "Comfy.MEC.SplineMaskEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const orig = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            orig?.apply(this, arguments);
            if (this._mecEditor) { this._mecEditor.load(); this._mecRender?.(); }
        };
    },
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME) return;
        installEditor(node);
    },
});
