/**
 * PointsMaskEditor / SAMMaskGeneratorMEC editor widget.
 *
 * Clean rewrite (May 2026):
 *   - HTML/CSS toolbar (no canvas-painted buttons that get cut off).
 *   - Canvas is image-only; annotations overlaid.
 *   - editor_data JSON contract preserved:
 *       {points:[{x,y,label,radius}], bboxes:[[x1,y1,x2,y2,label]]}
 *
 * Interaction model (no mode switch):
 *   Left click          add positive point
 *   Right click         add negative point
 *   Left  drag (empty)  draw positive bbox
 *   Right drag (empty)  draw negative bbox
 *   Drag a point        move it
 *   Shift+click element delete it
 *   Middle / Alt+drag   pan
 *   Wheel               adjust point radius
 *   Ctrl+wheel          zoom (cursor-anchored)
 *   Ctrl+Z / Ctrl+Y     undo / redo
 *   Delete / Backspace  delete hovered element
 *   F                   fit view
 */
import { app } from "../../scripts/app.js";

const TARGET_NODES = ["PointsMaskEditor", "SAMMaskGeneratorMEC"];

const COLOR = {
    bg: "#181825",
    border: "#313244",
    pos: "#22d65a",
    neg: "#ff4466",
    posFill: "rgba(34,214,90,0.12)",
    negFill: "rgba(255,68,102,0.12)",
    text: "#cdd6f4",
    sub: "#7f849c",
};

const HISTORY_LIMIT = 80;
const POINT_HIT_PX = 8;
const BBOX_EDGE_PX = 6;
const MIN_BBOX_PX = 4;
const CLICK_THRESHOLD_PX = 4;

class Editor {
    constructor(node) {
        this.node = node;
        this.points = [];
        this.bboxes = [];
        this.canvasW = 512;
        this.canvasH = 512;
        this.radius = 3.0;

        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this._fitted = false;

        this.refImg = null;
        this.refUrl = null;

        this.hover = { kind: null, idx: -1 };
        this.drag = null;

        this.undoStack = [];
        this.redoStack = [];

        this.cursor = { x: 0, y: 0, visible: false };
    }

    snapshot() { return JSON.stringify({ p: this.points, b: this.bboxes }); }
    pushUndo() {
        this.undoStack.push(this.snapshot());
        if (this.undoStack.length > HISTORY_LIMIT) this.undoStack.shift();
        this.redoStack.length = 0;
    }
    restore(s) { const o = JSON.parse(s); this.points = o.p || []; this.bboxes = o.b || []; }
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

    fitView(viewW, viewH) {
        if (viewW <= 0 || viewH <= 0) return;
        const pad = 16;
        const sx = (viewW - pad * 2) / this.canvasW;
        const sy = (viewH - pad * 2) / this.canvasH;
        this.zoom = Math.max(0.05, Math.min(sx, sy, 8));
        this.panX = (viewW - this.canvasW * this.zoom) / 2;
        this.panY = (viewH - this.canvasH * this.zoom) / 2;
        this._fitted = true;
    }

    counts() {
        let pos = 0, neg = 0, bp = 0, bn = 0;
        for (const p of this.points) p.label === 1 ? pos++ : neg++;
        for (const b of this.bboxes) b[4] === 1 ? bp++ : bn++;
        return { pos, neg, bp, bn };
    }

    findPoint(cx, cy) {
        const rPx = POINT_HIT_PX / this.zoom;
        let best = -1, bestD = rPx * rPx;
        for (let i = this.points.length - 1; i >= 0; i--) {
            const p = this.points[i];
            const d = (p.x - cx) ** 2 + (p.y - cy) ** 2;
            if (d < bestD) { bestD = d; best = i; }
        }
        return best;
    }
    findBbox(cx, cy) {
        const tol = BBOX_EDGE_PX / this.zoom;
        for (let i = this.bboxes.length - 1; i >= 0; i--) {
            const [x1, y1, x2, y2] = this.bboxes[i];
            const onV = (Math.abs(cx - x1) < tol || Math.abs(cx - x2) < tol) && cy >= y1 - tol && cy <= y2 + tol;
            const onH = (Math.abs(cy - y1) < tol || Math.abs(cy - y2) < tol) && cx >= x1 - tol && cx <= x2 + tol;
            if (onV || onH) return i;
        }
        for (let i = this.bboxes.length - 1; i >= 0; i--) {
            const [x1, y1, x2, y2] = this.bboxes[i];
            if (cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2) return i;
        }
        return -1;
    }

    save() {
        const w = this.node.widgets?.find(w => w.name === "editor_data");
        if (!w) return;
        const data = {
            points: this.points.map(p => ({
                x: +p.x.toFixed(2), y: +p.y.toFixed(2),
                label: p.label | 0,
                radius: +(+p.radius || this.radius).toFixed(2),
            })),
            bboxes: this.bboxes.map(b => [b[0] | 0, b[1] | 0, b[2] | 0, b[3] | 0, b[4] | 0]),
        };
        w.value = JSON.stringify(data);
        if (this.node.graph) this.node.graph.setDirtyCanvas(true, false);
    }

    load() {
        const w = this.node.widgets?.find(w => w.name === "editor_data");
        if (!w?.value) return;
        try {
            const o = JSON.parse(w.value);
            if (Array.isArray(o)) {
                this.points = o; this.bboxes = [];
            } else {
                this.points = (o.points || []).map(p => ({
                    x: +p.x, y: +p.y, label: p.label | 0,
                    radius: +(p.radius || this.radius),
                }));
                this.bboxes = (o.bboxes || []).map(b => [+b[0], +b[1], +b[2], +b[3], (b[4] != null ? b[4] | 0 : 1)]);
            }
        } catch (_) { /* ignore */ }
    }

    setRefImage(url, origW, origH) {
        if (url === this.refUrl && this.refImg && this.refImg.complete) return;
        this.refUrl = url;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            this.refImg = img;
            const w = origW || img.naturalWidth;
            const h = origH || img.naturalHeight;
            if (w > 0 && h > 0 && (w !== this.canvasW || h !== this.canvasH)) {
                this.canvasW = w; this.canvasH = h;
                const wW = this.node.widgets?.find(x => x.name === "width");
                const hW = this.node.widgets?.find(x => x.name === "height");
                if (wW) wW.value = w;
                if (hW) hW.value = h;
            }
            this._fitted = false;
            this.onLoaded?.();
        };
        img.src = url;
    }
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

    for (let i = 0; i < ed.bboxes.length; i++) {
        const [x1, y1, x2, y2, lbl] = ed.bboxes[i];
        const c = lbl === 1 ? COLOR.pos : COLOR.neg;
        ctx.fillStyle = lbl === 1 ? COLOR.posFill : COLOR.negFill;
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        ctx.strokeStyle = c;
        ctx.lineWidth = (ed.hover.kind === "bbox" && ed.hover.idx === i ? 2.5 : 1.5) / z;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const t = 6 / z;
        ctx.beginPath();
        ctx.moveTo(x1, y1 + t); ctx.lineTo(x1, y1); ctx.lineTo(x1 + t, y1);
        ctx.moveTo(x2 - t, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + t);
        ctx.moveTo(x2, y2 - t); ctx.lineTo(x2, y2); ctx.lineTo(x2 - t, y2);
        ctx.moveTo(x1 + t, y2); ctx.lineTo(x1, y2); ctx.lineTo(x1, y2 - t);
        ctx.stroke();
    }

    if (ed.drag?.kind === "bbox-new") {
        const { sx, sy, ex, ey, label } = ed.drag;
        const x1 = Math.min(sx, ex), y1 = Math.min(sy, ey);
        const x2 = Math.max(sx, ex), y2 = Math.max(sy, ey);
        ctx.fillStyle = label === 1 ? COLOR.posFill : COLOR.negFill;
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        ctx.strokeStyle = label === 1 ? COLOR.pos : COLOR.neg;
        ctx.setLineDash([6 / z, 4 / z]);
        ctx.lineWidth = 1.5 / z;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.setLineDash([]);
    }

    for (let i = 0; i < ed.points.length; i++) {
        const p = ed.points[i];
        const c = p.label === 1 ? COLOR.pos : COLOR.neg;
        ctx.strokeStyle = c + "66";
        ctx.lineWidth = 1 / z;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.stroke();
        const isHov = ed.hover.kind === "point" && ed.hover.idx === i;
        const dotR = (isHov ? 6 : 4) / z;
        ctx.fillStyle = c;
        ctx.beginPath(); ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "#000a";
        ctx.lineWidth = 1.5 / z;
        ctx.stroke();
    }

    if (ed.cursor.visible && !ed.drag) {
        ctx.strokeStyle = "#ffffff66";
        ctx.lineWidth = 1 / z;
        ctx.beginPath();
        ctx.arc(ed.cursor.x, ed.cursor.y, ed.radius, 0, Math.PI * 2);
        ctx.stroke();
    }

    ctx.restore();
}

function findRefImage(node) {
    if (!node.inputs) return null;
    const linkedInput = node.inputs.find(i =>
        (i.name === "reference_image" || i.name === "image") && i.link != null);
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
    hideWidget(node.widgets?.find(w => w.name === "editor_data"));

    const syncDims = () => {
        const w = node.widgets?.find(x => x.name === "width");
        const h = node.widgets?.find(x => x.name === "height");
        if (w) ed.canvasW = +w.value || ed.canvasW;
        if (h) ed.canvasH = +h.value || ed.canvasH;
    };
    const syncRadius = () => {
        const r = node.widgets?.find(x => x.name === "default_radius");
        if (r) ed.radius = +r.value || ed.radius;
    };
    syncDims(); syncRadius();
    ed.load();

    const root = document.createElement("div");
    root.style.cssText = `
        display:flex;flex-direction:column;width:100%;height:100%;
        background:${COLOR.bg};border:1px solid ${COLOR.border};border-radius:6px;
        overflow:hidden;font-family:Inter,system-ui,sans-serif;color:${COLOR.text};
        box-sizing:border-box;user-select:none;
    `;

    // --- Compact toolbar (icon buttons + dropdown menu) ----------------
    const tb = document.createElement("div");
    tb.style.cssText = `
        display:flex;align-items:center;gap:4px;padding:4px 6px;
        background:linear-gradient(#22223a,#1a1a2e);
        border-bottom:1px solid ${COLOR.border};
        flex:0 0 auto;font-size:11px;line-height:1;
    `;
    root.appendChild(tb);

    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText = "position:relative;flex:1 1 auto;min-height:0;overflow:hidden;cursor:crosshair;background:#11111b;";
    root.appendChild(canvasWrap);

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;width:100%;height:100%;outline:none;";
    canvas.tabIndex = 0;
    canvasWrap.appendChild(canvas);

    // status pill (image res + zoom only — no live cursor coords)
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
        b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); onClick(); render(); };
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

    tb.appendChild(mkIcon("↶", "Undo (Ctrl+Z)", () => { if (ed.undo()) ed.save(); }));
    tb.appendChild(mkIcon("↷", "Redo (Ctrl+Y)", () => { if (ed.redo()) ed.save(); }));
    tb.appendChild(mkIcon("⛶", "Fit view (F)", () => {
        const r = canvasWrap.getBoundingClientRect();
        ed.fitView(r.width, r.height);
    }));

    // Dropdown menu (≡)
    const menuBtn = mkIcon("≡", "More actions", () => toggleMenu());
    tb.appendChild(menuBtn);

    const menu = document.createElement("div");
    menu.style.cssText = `
        position:absolute;top:34px;right:6px;z-index:50;
        background:#1e1e2e;border:1px solid ${COLOR.border};border-radius:6px;
        box-shadow:0 6px 20px #000a;padding:4px;display:none;
        min-width:180px;font-size:11px;
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
    menu.appendChild(mkMenuItem("Clear points", () => {
        if (!ed.points.length) return;
        ed.pushUndo(); ed.points = []; ed.save();
    }, true));
    menu.appendChild(mkMenuItem("Clear bounding boxes", () => {
        if (!ed.bboxes.length) return;
        ed.pushUndo(); ed.bboxes = []; ed.save();
    }, true));
    menu.appendChild(mkMenuItem("Clear everything", () => {
        if (!ed.points.length && !ed.bboxes.length) return;
        ed.pushUndo(); ed.points = []; ed.bboxes = []; ed.save();
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
    node.addDOMWidget("points_editor", "canvas", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => widgetH,
        getHeight: () => widgetH,
    });

    // Ensure a sensible default node width so the toolbar fits without scrolling.
    if (!node.size || node.size[0] < 600) {
        const h = node.size?.[1] || 620;
        node.setSize?.([600, Math.max(h, 620)]);
    }

    let lastW = 0, lastH = 0;
    function syncCanvasPx() {
        const r = canvasWrap.getBoundingClientRect();
        const w = Math.max(1, Math.round(r.width));
        const h = Math.max(1, Math.round(r.height));
        if (w === lastW && h === lastH) return false;
        // Scale pan/zoom proportionally so the visual view stays put under
        // ComfyUI canvas zoom (which resizes our container without us moving).
        if (lastW > 0 && lastH > 0 && ed._fitted) {
            const sx = w / lastW;
            ed.zoom *= sx;
            ed.panX *= sx;
            ed.panY *= (h / lastH);
        }
        lastW = w; lastH = h;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        if (!ed._fitted) ed.fitView(w, h);
        return true;
    }

    let raf = 0;
    function render() {
        if (raf) return;
        raf = requestAnimationFrame(() => {
            raf = 0;
            syncCanvasPx();
            draw(ed, ctx, lastW, lastH);
            const c = ed.counts();
            counter.textContent = `pts +${c.pos}/-${c.neg}   box +${c.bp}/-${c.bn}`;
            status.textContent = `${ed.canvasW}×${ed.canvasH} · zoom ${(ed.zoom * 100).toFixed(0)}% · r ${ed.radius.toFixed(1)}`;
        });
    }
    ed.onLoaded = () => { ed._fitted = false; render(); };

    const ro = new ResizeObserver(() => render());
    ro.observe(canvasWrap);

    // Mouse coords MUST be scaled by (canvas.width / r.width) because the
    // <canvas> element is CSS-scaled by ComfyUI's outer-canvas zoom while
    // its internal pixel buffer (canvas.width) stays fixed. Without this,
    // every click drifts proportional to LiteGraph's zoom level.
    function eventCanvas(e) {
        const r = canvas.getBoundingClientRect();
        const sx = canvas.width  / (r.width  || 1);
        const sy = canvas.height / (r.height || 1);
        return ed.viewToCanvas((e.clientX - r.left) * sx, (e.clientY - r.top) * sy);
    }

    canvas.addEventListener("pointerdown", (e) => {
        e.preventDefault(); canvas.focus();
        canvas.setPointerCapture(e.pointerId);
        const c = eventCanvas(e);

        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            const r0 = canvas.getBoundingClientRect();
            const sx0 = canvas.width / (r0.width || 1), sy0 = canvas.height / (r0.height || 1);
            ed.drag = {
                kind: "pan",
                startX: (e.clientX - r0.left) * sx0 - ed.panX,
                startY: (e.clientY - r0.top)  * sy0 - ed.panY,
            };
            canvas.style.cursor = "grabbing";
            return;
        }
        const pi = ed.findPoint(c.x, c.y);
        const bi = pi < 0 ? ed.findBbox(c.x, c.y) : -1;

        if (e.shiftKey) {
            if (pi >= 0) { ed.pushUndo(); ed.points.splice(pi, 1); ed.save(); render(); return; }
            if (bi >= 0) { ed.pushUndo(); ed.bboxes.splice(bi, 1); ed.save(); render(); return; }
            return;
        }
        if (e.button === 0 && pi >= 0) {
            ed.pushUndo();
            ed.drag = { kind: "point", idx: pi };
            return;
        }
        if (e.button === 0 || e.button === 2) {
            ed.drag = {
                kind: "maybe-bbox",
                button: e.button,
                startX: c.x, startY: c.y,
                screenStartX: e.clientX, screenStartY: e.clientY,
            };
        }
    });

    canvas.addEventListener("pointermove", (e) => {
        const c = eventCanvas(e);
        ed.cursor = { x: c.x, y: c.y, visible: true };

        if (ed.drag?.kind === "pan") {
            const r1 = canvas.getBoundingClientRect();
            const sx1 = canvas.width / (r1.width || 1), sy1 = canvas.height / (r1.height || 1);
            ed.panX = (e.clientX - r1.left) * sx1 - ed.drag.startX;
            ed.panY = (e.clientY - r1.top)  * sy1 - ed.drag.startY;
            render(); return;
        }
        if (ed.drag?.kind === "point") {
            const p = ed.points[ed.drag.idx];
            if (p) { p.x = +c.x.toFixed(2); p.y = +c.y.toFixed(2); render(); }
            return;
        }
        if (ed.drag?.kind === "maybe-bbox") {
            const dx = e.clientX - ed.drag.screenStartX;
            const dy = e.clientY - ed.drag.screenStartY;
            if (Math.hypot(dx, dy) > CLICK_THRESHOLD_PX) {
                ed.drag = {
                    kind: "bbox-new",
                    sx: ed.drag.startX, sy: ed.drag.startY,
                    ex: c.x, ey: c.y,
                    label: ed.drag.button === 0 ? 1 : 0,
                    button: ed.drag.button,
                };
                render();
            }
            return;
        }
        if (ed.drag?.kind === "bbox-new") {
            ed.drag.ex = c.x; ed.drag.ey = c.y; render(); return;
        }

        const pi = ed.findPoint(c.x, c.y);
        let kind = null, idx = -1;
        if (pi >= 0) { kind = "point"; idx = pi; }
        else {
            const bi = ed.findBbox(c.x, c.y);
            if (bi >= 0) { kind = "bbox"; idx = bi; }
        }
        if (kind !== ed.hover.kind || idx !== ed.hover.idx) {
            ed.hover = { kind, idx };
            canvas.style.cursor = kind ? "pointer" : "crosshair";
            render();
        }
    });

    canvas.addEventListener("pointerup", (e) => {
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
        const c = eventCanvas(e);

        if (ed.drag?.kind === "pan") {
            ed.drag = null; canvas.style.cursor = "crosshair"; return;
        }
        if (ed.drag?.kind === "point") {
            ed.drag = null; ed.save(); render(); return;
        }
        if (ed.drag?.kind === "maybe-bbox") {
            const label = ed.drag.button === 0 ? 1 : 0;
            ed.drag = null;
            ed.pushUndo();
            ed.points.push({
                x: +c.x.toFixed(2), y: +c.y.toFixed(2),
                label, radius: ed.radius,
            });
            ed.save(); render(); return;
        }
        if (ed.drag?.kind === "bbox-new") {
            const { sx, sy, ex, ey, label } = ed.drag;
            ed.drag = null;
            const x1 = Math.min(sx, ex), y1 = Math.min(sy, ey);
            const x2 = Math.max(sx, ex), y2 = Math.max(sy, ey);
            if (x2 - x1 >= MIN_BBOX_PX && y2 - y1 >= MIN_BBOX_PX) {
                ed.pushUndo();
                ed.bboxes.push([Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2), label]);
                ed.save();
            }
            render();
        }
    });

    canvas.addEventListener("pointerleave", () => {
        ed.cursor.visible = false;
        ed.hover = { kind: null, idx: -1 };
        render();
    });

    canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    canvas.addEventListener("wheel", (e) => {
        e.preventDefault(); e.stopPropagation();
        const r = canvas.getBoundingClientRect();
        const sx = canvas.width  / (r.width  || 1);
        const sy = canvas.height / (r.height || 1);
        const mx = (e.clientX - r.left) * sx, my = (e.clientY - r.top) * sy;
        if (e.ctrlKey || e.metaKey) {
            const f = e.deltaY < 0 ? 1.15 : 0.87;
            const newZ = Math.max(0.05, Math.min(40, ed.zoom * f));
            const k = newZ / ed.zoom;
            ed.panX = mx - (mx - ed.panX) * k;
            ed.panY = my - (my - ed.panY) * k;
            ed.zoom = newZ;
        } else {
            ed.radius = Math.max(0.5, Math.min(256, ed.radius + (e.deltaY < 0 ? 0.5 : -0.5)));
            const rW = node.widgets?.find(x => x.name === "default_radius");
            if (rW) rW.value = ed.radius;
        }
        render();
    }, { passive: false });

    canvas.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
            e.preventDefault();
            (e.shiftKey ? ed.redo() : ed.undo()) && ed.save(); render();
        } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
            e.preventDefault(); ed.redo() && ed.save(); render();
        } else if (e.key === "Delete" || e.key === "Backspace") {
            e.preventDefault();
            if (ed.hover.kind === "point") { ed.pushUndo(); ed.points.splice(ed.hover.idx, 1); ed.save(); }
            else if (ed.hover.kind === "bbox") { ed.pushUndo(); ed.bboxes.splice(ed.hover.idx, 1); ed.save(); }
            ed.hover = { kind: null, idx: -1 }; render();
        } else if (e.key.toLowerCase() === "f") {
            const r = canvasWrap.getBoundingClientRect();
            ed.fitView(r.width, r.height); render();
        }
    });

    for (const wn of ["width", "height"]) {
        const w = node.widgets?.find(x => x.name === wn);
        if (!w) continue;
        const orig = w.callback;
        w.callback = function (v) { orig?.call(this, v); syncDims(); ed._fitted = false; render(); };
    }
    const rW = node.widgets?.find(x => x.name === "default_radius");
    if (rW) {
        const orig = rW.callback;
        rW.callback = function (v) { orig?.call(this, v); ed.radius = +v; render(); };
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
        if (out?.bg_image?.[0]) {
            const w = out.bg_image_width?.[0];
            const h = out.bg_image_height?.[0];
            ed.setRefImage("data:image/jpeg;base64," + out.bg_image[0], w, h);
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
    name: "MaskEditControl.PointsBBoxEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TARGET_NODES.includes(nodeData.name)) return;
        const orig = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            orig?.apply(this, arguments);
            if (this._mecEditor) { this._mecEditor.load(); this._mecRender?.(); }
        };
    },
    async nodeCreated(node) {
        if (!TARGET_NODES.includes(node.comfyClass)) return;
        installEditor(node);
    },
});
