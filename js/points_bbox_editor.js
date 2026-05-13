/**
 * PointsMaskEditor / SAMMaskGeneratorMEC editor widget.
 *
 * Clean rewrite (May 2026):
 *   - HTML/CSS toolbar (no canvas-painted buttons that get cut off).
 *   - Canvas is image-only; annotations overlaid.
 *   - editor_data JSON contract preserved:
 *       {points:[{x,y,label,radius}], bboxes:[[x1,y1,x2,y2,label]]}
 *
 * Interaction model (May 2026 v2):
 *   Left click               add positive point
 *   Right click              add negative point
 *   Shift + LMB drag         draw positive bbox (multi-bbox)
 *   Shift + RMB drag         draw negative bbox (multi-bbox)
 *   Ctrl  + LMB drag         draw positive bbox (single — replaces all)
 *   Ctrl  + RMB drag         draw negative bbox (single — replaces all)
 *   Drag a point             move it
 *   Alt + click element      delete it
 *   Middle / Alt + drag      pan
 *   Delete / Backspace       delete hovered element
 *   Ctrl+Z / Ctrl+Y          undo / redo
 *   F  / 0                   fit view
 *   + / -                    zoom in/out
 */
import { app } from "../../scripts/app.js";
import { installModeGated } from "./_mode_gate.js";

// Targets the unified MaskEditMEC (mode=points_bbox) plus legacy classes.
const TARGET_NODES = ["MaskEditMEC", "PointsMaskEditor", "SAMMaskGeneratorMEC"];

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

// Reduce a reference-image URL to a stable identity key that survives
// graph re-runs but flips when the user actually swaps the input image.
//   /view?filename=foo.png&subfolder=&type=input  →  "view:input/foo.png"
//   data:image/jpeg;base64,/9j/4AAQ…             →  "data:<len>:<head>:<tail>"
//   anything else (raw path / blob / http URL)   →  the URL itself
// For data URLs we sample length + head + tail of the base64 payload —
// identical content + identical JPEG/PNG encoder settings produce
// identical bytes, so this catches "same image, re-run" without paying
// a full hash. Real image swaps virtually always change the length.
function _refIdentityKey(url) {
    if (typeof url !== "string" || !url) return "";
    if (url.startsWith("data:")) {
        const comma = url.indexOf(",");
        const payload = comma >= 0 ? url.slice(comma + 1) : url;
        const head = payload.slice(0, 24);
        const tail = payload.slice(-24);
        return `data:${payload.length}:${head}:${tail}`;
    }
    const qIdx = url.indexOf("?");
    if (qIdx >= 0 && url.slice(0, qIdx).endsWith("/view")) {
        try {
            const params = new URLSearchParams(url.slice(qIdx + 1));
            const fn = params.get("filename") || "";
            const sub = params.get("subfolder") || "";
            const type = params.get("type") || "";
            return `view:${type}/${sub}/${fn}`;
        } catch (_) {
            // fall through
        }
    }
    return url;
}

class Editor {
    constructor(node) {
        this.node = node;
        this.points = [];
        this.bboxes = [];
        this.canvasW = 512;
        this.canvasH = 512;
        this.radius = 3.0;
        this.singleBbox = false;

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

    // Smallest zoom that still fits the image inside the viewport (no
    // scrolling past the borders). Used to clamp every zoom-out.
    minZoom(viewW, viewH) {
        if (viewW <= 0 || viewH <= 0) return 0.05;
        return Math.min(viewW / this.canvasW, viewH / this.canvasH);
    }

    // Stop pan from moving the image off-screen. Anchors:
    //   - if the image is wider than the viewport, allow X pan only inside
    //     the overflow region; otherwise centre it horizontally.
    //   - same for Y.
    clampPan(viewW, viewH) {
        const dW = this.canvasW * this.zoom;
        const dH = this.canvasH * this.zoom;
        if (dW <= viewW) {
            this.panX = (viewW - dW) / 2;
        } else {
            this.panX = Math.min(0, Math.max(viewW - dW, this.panX));
        }
        if (dH <= viewH) {
            this.panY = (viewH - dH) / 2;
        } else {
            this.panY = Math.min(0, Math.max(viewH - dH, this.panY));
        }
    }

    setZoomAround(newZ, anchorX, anchorY, viewW, viewH) {
        const minZ = this.minZoom(viewW, viewH);
        newZ = Math.max(minZ, Math.min(8, newZ));
        const k = newZ / this.zoom;
        this.panX = anchorX - (anchorX - this.panX) * k;
        this.panY = anchorY - (anchorY - this.panY) * k;
        this.zoom = newZ;
        this.clampPan(viewW, viewH);
    }

    fitView(viewW, viewH) {
        if (viewW <= 0 || viewH <= 0) return;
        // No padding: image touches the canvas edges, never overflows.
        const sx = viewW / this.canvasW;
        const sy = viewH / this.canvasH;
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
        const newKey = _refIdentityKey(url);
        const prevKey = this.refKey;
        const isReplacement = prevKey != null && newKey !== prevKey;
        this.refUrl = url;
        this.refKey = newKey;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            this.refImg = img;
            const w = origW || img.naturalWidth;
            const h = origH || img.naturalHeight;
            const dimsChanged = (w > 0 && h > 0 && (w !== this.canvasW || h !== this.canvasH));
            if (dimsChanged) {
                this.canvasW = w; this.canvasH = h;
                const wW = this.node.widgets?.find(x => x.name === "width");
                const hW = this.node.widgets?.find(x => x.name === "height");
                if (wW) wW.value = w;
                if (hW) hW.value = h;
            }
            // Only wipe annotations when the *identity* of the upstream
            // image changes (filename for /view URLs, content fingerprint
            // for data URLs). Re-running the graph with the same image
            // keeps the user's points/bboxes intact.
            if (isReplacement && (this.points.length || this.bboxes.length)) {
                this.pushUndo();
                this.points = [];
                this.bboxes = [];
                this.save?.();
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
    // Inset the editor from the node body so LiteGraph keeps a hit area for
    // its border / resize corner. Without this the DOM widget swallows every
    // pointer event and the user cannot grab the node edge or the bottom-
    // right resize grip — that's the "mouse doesn't work near borders" bug.
    root.style.cssText = `
        display:flex;flex-direction:column;
        width:calc(100% - 12px);height:calc(100% - 18px);
        margin:2px 6px 16px 6px;
        background:${COLOR.bg};border:1px solid ${COLOR.border};border-radius:6px;
        overflow:hidden;font-family:Inter,system-ui,sans-serif;color:${COLOR.text};
        box-sizing:border-box;user-select:none;
        pointer-events:none;
    `;

    // --- Compact toolbar (icon buttons + dropdown menu) ----------------
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
    tb.appendChild(mkIcon("\u2796", "Zoom out (-)", () => {
        const r = canvasWrap.getBoundingClientRect();
        ed.setZoomAround(ed.zoom * 0.85, r.width / 2, r.height / 2, r.width, r.height);
    }));
    tb.appendChild(mkIcon("\u2795", "Zoom in (+)", () => {
        const r = canvasWrap.getBoundingClientRect();
        ed.setZoomAround(ed.zoom * 1.15, r.width / 2, r.height / 2, r.width, r.height);
    }));
    tb.appendChild(mkIcon("\u2B1B", "Fit image (0 / F)", () => {
        const r = canvasWrap.getBoundingClientRect();
        ed.fitView(r.width, r.height);
    }));

    // Single-bbox toggle: when on, drawing a new bbox replaces the
    // existing one(s). Useful for SAM2/SAM3 single-region prompting.
    // (Ctrl+drag temporarily forces this regardless of toggle state.)
    const singleBtn = mkIcon("1\u25A1", "Single bbox mode (replace previous). Ctrl+drag does this once without toggling.", () => {
        ed.singleBbox = !ed.singleBbox;
        singleBtn.style.background = ed.singleBbox ? "#7c5cff" : "#313244";
        singleBtn.style.borderColor = ed.singleBbox ? "#7c5cff" : COLOR.border;
        singleBtn.style.color = ed.singleBbox ? "#ffffff" : COLOR.text;
        // If toggling ON with multiple boxes already, keep only the latest.
        if (ed.singleBbox && ed.bboxes.length > 1) {
            ed.pushUndo();
            ed.bboxes = [ed.bboxes[ed.bboxes.length - 1]];
            ed.save();
        }
    });
    tb.appendChild(singleBtn);

    // Visible Clear button (always-accessible, bypasses dropdown menu).
    const clearBtn = mkIcon("\u2715", "Clear all points + bboxes", () => {
        if (!ed.points.length && !ed.bboxes.length) return;
        ed.pushUndo();
        ed.points = [];
        ed.bboxes = [];
        ed.save();
    });
    clearBtn.style.color = "#f38ba8";
    clearBtn.style.borderColor = "#583444";
    tb.appendChild(clearBtn);

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

    const TOOLBAR_H = 36;
    const STATUS_PAD = 8;
    const MIN_WIDGET_H = 240;
    const MAX_WIDGET_H = 1100;
    let widgetH = 460;
    node.addDOMWidget("points_editor", "canvas", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => widgetH,
        getHeight: () => widgetH,
    });

    // Stash host root for mode-gate hide/show on the unified node.
    node._mecPointsHost = root;

    // Resize the widget so the image canvas matches the input image's
    // aspect ratio. Called after a new image loads.
    function resizeForImage() {
        const nodeW = (node.size?.[0] || 600);
        const innerW = Math.max(64, nodeW - 24); // node padding
        const aspect = ed.canvasH / Math.max(1, ed.canvasW); // h/w
        const targetCanvasH = Math.round(innerW * aspect);
        const target = Math.max(MIN_WIDGET_H, Math.min(MAX_WIDGET_H,
                                                       targetCanvasH + TOOLBAR_H + STATUS_PAD));
        if (Math.abs(target - widgetH) < 4) return;
        widgetH = target;
        // Recompute total node height from the bottom-up so other widgets
        // (height, width, default_radius...) stay visible above ours.
        const cur = node.size || [nodeW, 620];
        const delta = target - 460;
        node.setSize?.([cur[0], Math.max(620 + delta, target + 220)]);
        node.graph?.setDirtyCanvas(true, true);
    }

    // Ensure a sensible default node width so the toolbar fits without scrolling.
    if (!node.size || node.size[0] < 600) {
        const h = node.size?.[1] || 620;
        node.setSize?.([600, Math.max(h, 620)]);
    }

    let lastW = 0, lastH = 0;
    function syncCanvasPx() {
        const r = canvasWrap.getBoundingClientRect();
        // Parent ComfyUI canvas applies CSS transform: scale(zoom) to DOM
        // widgets. Divide by the layout-vs-visual ratio to recover the
        // unscaled CSS pixel size of the widget.
        const cssW = canvasWrap.offsetWidth || r.width;
        const cssH = canvasWrap.offsetHeight || r.height;
        const w = Math.max(1, Math.round(cssW));
        const h = Math.max(1, Math.round(cssH));
        const dpr = window.devicePixelRatio || 1;
        const needPx = (canvas.width !== w * dpr) || (canvas.height !== h * dpr);
        const sizeChanged = (w !== lastW || h !== lastH);
        if (!sizeChanged && !needPx) return false;
        lastW = w; lastH = h;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // Re-fit on every actual size change so the image always touches
        // the canvas edges and the user never needs to manually zoom.
        if (sizeChanged || !ed._fitted) ed.fitView(w, h);
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
    ed.onLoaded = () => { ed._fitted = false; resizeForImage(); render(); };

    const ro = new ResizeObserver(() => render());
    ro.observe(canvasWrap);

    // Mouse coords -> canvas CSS-pixel coords. ComfyUI's outer canvas
    // applies a CSS scale transform when the user zooms the graph. That
    // makes getBoundingClientRect() return the *visual* (post-transform)
    // size, while the canvas's internal logical pixels stay unscaled.
    // Multiply by (offsetWidth / rect.width) to undo the parent scale —
    // without this every click lands at click_x * litegraph_zoom.
    function eventCanvas(e) {
        const r = canvas.getBoundingClientRect();
        const sx = (canvas.offsetWidth  || r.width)  / Math.max(1, r.width);
        const sy = (canvas.offsetHeight || r.height) / Math.max(1, r.height);
        return ed.viewToCanvas((e.clientX - r.left) * sx, (e.clientY - r.top) * sy);
    }

    canvas.addEventListener("pointerdown", (e) => {
        e.preventDefault(); canvas.focus();
        canvas.setPointerCapture(e.pointerId);
        const c = eventCanvas(e);

        // Pan: middle button or Alt+drag (without Shift/Ctrl)
        if (e.button === 1 || (e.button === 0 && e.altKey && !e.shiftKey && !e.ctrlKey && !e.metaKey)) {
            const r0 = canvas.getBoundingClientRect();
            const sx = (canvas.offsetWidth  || r0.width)  / Math.max(1, r0.width);
            const sy = (canvas.offsetHeight || r0.height) / Math.max(1, r0.height);
            ed.drag = {
                kind: "pan",
                startX: (e.clientX - r0.left) * sx - ed.panX,
                startY: (e.clientY - r0.top)  * sy - ed.panY,
                sx, sy,
            };
            canvas.style.cursor = "grabbing";
            return;
        }

        const pi = ed.findPoint(c.x, c.y);
        const bi = pi < 0 ? ed.findBbox(c.x, c.y) : -1;

        // Alt+click on an element deletes it (was shift+click; shift now reserved for bbox draw)
        if (e.altKey && !e.shiftKey && !e.ctrlKey) {
            if (pi >= 0) { ed.pushUndo(); ed.points.splice(pi, 1); ed.save(); render(); return; }
            if (bi >= 0) { ed.pushUndo(); ed.bboxes.splice(bi, 1); ed.save(); render(); return; }
            return;
        }

        // Shift / Ctrl + drag = draw bbox (any button picks polarity).
        // Ctrl variant replaces existing boxes (single-bbox mode).
        if (e.shiftKey || e.ctrlKey || e.metaKey) {
            ed.drag = {
                kind: "maybe-bbox",
                button: e.button,
                replace: !!(e.ctrlKey || e.metaKey),
                startX: c.x, startY: c.y,
                screenStartX: e.clientX, screenStartY: e.clientY,
            };
            return;
        }

        // Plain LMB on existing point → move it
        if (e.button === 0 && pi >= 0) {
            ed.pushUndo();
            ed.drag = { kind: "point", idx: pi };
            return;
        }

        // Plain LMB / RMB on empty space → click-only (commits to a point on pointerup)
        if (e.button === 0 || e.button === 2) {
            ed.drag = {
                kind: "maybe-point",
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
            const sx = ed.drag.sx, sy = ed.drag.sy;
            ed.panX = (e.clientX - r1.left) * sx - ed.drag.startX;
            ed.panY = (e.clientY - r1.top)  * sy - ed.drag.startY;
            ed.clampPan(lastW, lastH);
            render(); return;
        }
        if (ed.drag?.kind === "point") {
            const p = ed.points[ed.drag.idx];
            if (p) { p.x = +c.x.toFixed(2); p.y = +c.y.toFixed(2); render(); }
            return;
        }
        if (ed.drag?.kind === "maybe-point") {
            // Plain LMB/RMB drag has no bbox effect anymore; if user drags
            // far enough we just abandon the click (treat as no-op).
            const dx = e.clientX - ed.drag.screenStartX;
            const dy = e.clientY - ed.drag.screenStartY;
            if (Math.hypot(dx, dy) > CLICK_THRESHOLD_PX) {
                ed.drag = { kind: "noop" };
            }
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
                    replace: ed.drag.replace,
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
        }
        // Always render so the cursor radius circle follows the mouse.
        // render() coalesces via RAF — cheap when nothing else changed.
        render();
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
        if (ed.drag?.kind === "noop") {
            ed.drag = null; return;
        }
        if (ed.drag?.kind === "maybe-point") {
            const label = ed.drag.button === 0 ? 1 : 0;
            ed.drag = null;
            ed.pushUndo();
            ed.points.push({
                x: +c.x.toFixed(2), y: +c.y.toFixed(2),
                label, radius: ed.radius,
            });
            ed.save(); render(); return;
        }
        if (ed.drag?.kind === "maybe-bbox") {
            // Modifier was held but user didn't actually drag — ignore.
            ed.drag = null; return;
        }
        if (ed.drag?.kind === "bbox-new") {
            const { sx, sy, ex, ey, label, replace } = ed.drag;
            ed.drag = null;
            const x1 = Math.min(sx, ex), y1 = Math.min(sy, ey);
            const x2 = Math.max(sx, ex), y2 = Math.max(sy, ey);
            if (x2 - x1 >= MIN_BBOX_PX && y2 - y1 >= MIN_BBOX_PX) {
                ed.pushUndo();
                if (replace || ed.singleBbox) {
                    // Single-bbox mode (Ctrl-drag, or toggle on): replace any existing boxes.
                    ed.bboxes = [];
                }
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

    // Mouse wheel inside the editor: SWALLOW it so the parent ComfyUI
    // graph canvas does NOT zoom while the user is hovering our editor.
    // Without this, scrolling over the editor would re-fit-on-resize and
    // create the "image rescaling on canvas zoom" symptom.
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
            if (ed.hover.kind === "point") { ed.pushUndo(); ed.points.splice(ed.hover.idx, 1); ed.save(); }
            else if (ed.hover.kind === "bbox") { ed.pushUndo(); ed.bboxes.splice(ed.hover.idx, 1); ed.save(); }
            ed.hover = { kind: null, idx: -1 }; render();
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
        if (node.comfyClass === "MaskEditMEC") {
            installModeGated(node, {
                activeWhen: "points_bbox",
                installerKey: "maskEditPoints",
                installer: (n) => installEditor(n),
                hostFinder: (n) => n._mecPointsHost || null,
            });
        } else {
            installEditor(node);
        }
    },
});
