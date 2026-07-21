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
import { installModeGated } from "./_mode_gate.js";
import { C, bg3, border, peach } from "./_c2c_theme.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { drawEditorEmptyState } from "./_editor_empty_state.js";
import { ensureC2CKit } from "./_c2c_ui_kit.js";

// Targets both the unified SplineMaskMEC (mode=edit) node and any
// legacy SplineMaskEditorMEC references that may still live on saved
// graphs. The unified node is gated on mode === "edit".
const NODE_NAMES = ["SplineMaskMEC", "SplineMaskEditorMEC"];
const NODE_NAME = "SplineMaskMEC";

const COLOR = {
    bg: "var(--c2c-bg2)",
    border: "var(--c2c-border)",
    text: "var(--c2c-fg)",
    sub: "var(--c2c-overlay1)",
    paths: ["var(--c2c-green)", "var(--c2c-blue)", "var(--c2c-yellow)", "var(--c2c-peach)", "var(--c2c-pink)", "var(--c2c-teal)"],
};

const HISTORY_LIMIT = 80;
const POINT_HIT_PX = 8;

// Module-level decoded-image cache keyed by URL.
// Fixes "slow image load": when the user pans the graph or re-opens the
// node, we re-use the already-decoded HTMLImageElement instead of issuing
// a fresh fetch + decode every time the URL is re-set.
const IMG_CACHE = new Map(); // url -> {img, w, h}
const IMG_CACHE_MAX = 32;
function _cacheGet(url) { return IMG_CACHE.get(url) || null; }
function _cachePut(url, entry) {
    IMG_CACHE.set(url, entry);
    if (IMG_CACHE.size > IMG_CACHE_MAX) {
        const firstKey = IMG_CACHE.keys().next().value;
        if (firstKey !== undefined) IMG_CACHE.delete(firstKey);
    }
}

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
        this.centripetalAlpha = 0.5;
    }

    snapshot() { return JSON.stringify({ s: this.shapes, a: this.active }); }
    pushUndo() {
        this.undoStack.push(this.snapshot());
        if (this.undoStack.length > HISTORY_LIMIT) this.undoStack.shift();
        this.redoStack.length = 0;
    }
    restore(s) {
        // Defensive: corrupted snapshot would crash entire canvas.
        let o;
        try { o = JSON.parse(s); }
        catch (e) { console.warn("[MEC.SplineMaskEditor] restore: malformed snapshot, ignored", e); return; }
        if (!o) return;
        this.shapes = Array.isArray(o.s) ? o.s : [];
        this.active = Number.isFinite(o.a) ? o.a : -1;
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
    // Hit-test Bezier handle endpoints on the active shape only.
    // Returns {shape, point, side:"in"|"out"} or null.
    findHandle(cx, cy) {
        if (this.active < 0) return null;
        const sh = this.shapes[this.active];
        if (!sh || sh.type !== "bezier" || !Array.isArray(sh.handles)) return null;
        const r = POINT_HIT_PX / this.zoom;
        const r2 = r * r;
        for (let i = 0; i < sh.points.length; i++) {
            const h = sh.handles[i];
            if (!h) continue;
            const p = sh.points[i];
            for (const side of ["in", "out"]) {
                const hv = h[side];
                if (!hv) continue;
                const hx = p.x + hv.x, hy = p.y + hv.y;
                const d = (hx - cx) ** 2 + (hy - cy) ** 2;
                if (d < r2) return { shape: this.active, point: i, side };
            }
        }
        return null;
    }
    // Ensure shape.handles is sized to match shape.points; create symmetric
    // tangents from neighbours when missing. Idempotent.
    ensureHandles(sh) {
        if (sh.type !== "bezier") return;
        if (!Array.isArray(sh.handles)) sh.handles = [];
        const n = sh.points.length;
        for (let i = 0; i < n; i++) {
            if (sh.handles[i] && sh.handles[i].in && sh.handles[i].out) continue;
            const p = sh.points[i];
            const prev = sh.points[i - 1] || (sh.closed ? sh.points[n - 1] : null);
            const next = sh.points[i + 1] || (sh.closed ? sh.points[0]     : null);
            let tx = 0, ty = 0;
            if (prev && next) { tx = (next.x - prev.x) * 0.25; ty = (next.y - prev.y) * 0.25; }
            else if (prev)    { tx = (p.x - prev.x) * 0.33;    ty = (p.y - prev.y) * 0.33; }
            else if (next)    { tx = (next.x - p.x) * 0.33;    ty = (next.y - p.y) * 0.33; }
            else              { tx = 40; ty = 0; }
            sh.handles[i] = { in: { x: -tx, y: -ty }, out: { x: tx, y: ty } };
        }
        sh.handles.length = n;
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
                for (const sh of this.shapes) if (sh.type === "bezier") this.ensureHandles(sh);
            }
        } catch (__c2cErr) { __c2cReport("spline_mask_editor", __c2cErr); }
    }

    setRefImage(url, ow, oh) {
        if (url === this.refUrl && this.refImg && this.refImg.complete) return;
        const sameUrl = (url === this.refUrl);
        this.refUrl = url;

        // Cache hit: zero-cost re-use; do NOT reset zoom/pan when the URL
        // didn't actually change (was the source of the bounce bug — every
        // re-render set refImg again, clearing _fitted, snapping the view).
        const cached = _cacheGet(url);
        if (cached?.img?.complete) {
            this.refImg = cached.img;
            const w = ow || cached.w;
            const h = oh || cached.h;
            if (w > 0 && h > 0 && (this.canvasW !== w || this.canvasH !== h)) {
                this.canvasW = w; this.canvasH = h;
                const wW = this.node.widgets?.find(x => x.name === "width");
                const hW = this.node.widgets?.find(x => x.name === "height");
                if (wW && +wW.value !== w) wW.value = w;
                if (hW && +hW.value !== h) hW.value = h;
                this._fitted = false; // dims changed — a fresh fit is correct
            }
            if (!sameUrl) this._fitted = false;
            this.onLoaded?.();
            return;
        }

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
            _cachePut(url, { img, w, h });
            this._fitted = false;
            this.onLoaded?.();
        };
        img.src = url;
    }
}

function catmullRom(points, samplesPerSeg, closed, alpha) {
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
    // Centripetal Catmull-Rom — alpha is now configurable (0=uniform,
    // 0.5=centripetal, 1=chordal). 0.5 is the default and avoids cusps.
    const a = (typeof alpha === "number") ? Math.max(0, Math.min(1, alpha)) : 0.5;
    for (let i = 0; i < segs; i++) {
        const p0 = ext[i], p1 = ext[i + 1], p2 = ext[i + 2], p3 = ext[i + 3];
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

function bezierCubic(points, samplesPerSeg, closed, handles) {
    const n = points.length;
    if (n < 2 || !Array.isArray(handles) || handles.length < n) return points.slice();
    const out = [];
    const segs = closed ? n : n - 1;
    for (let i = 0; i < segs; i++) {
        const p1 = points[i];
        const p2 = points[(i + 1) % n];
        const h1 = handles[i];
        const h2 = handles[(i + 1) % n];
        if (!h1 || !h2) continue;
        const c1x = p1.x + (h1.out?.x ?? 0), c1y = p1.y + (h1.out?.y ?? 0);
        const c2x = p2.x + (h2.in?.x  ?? 0), c2y = p2.y + (h2.in?.y  ?? 0);
        const last = i === segs - 1;
        const steps = samplesPerSeg + (last ? 1 : 0);
        for (let s = 0; s < steps; s++) {
            const t = s / samplesPerSeg;
            const u = 1 - t;
            const b0 = u * u * u, b1 = 3 * u * u * t, b2 = 3 * u * t * t, b3 = t * t * t;
            out.push({
                x: b0 * p1.x + b1 * c1x + b2 * c2x + b3 * p2.x,
                y: b0 * p1.y + b1 * c1y + b2 * c2y + b3 * p2.y,
            });
        }
    }
    return out;
}

// ── new accurate curve families (mirror nodes/_spline_curves.py exactly so
//    the on-canvas preview matches the rasterized mask) ──────────────────
const BSPLINE_M = [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]
    .map((r) => r.map((v) => v / 6));

function bsplineLike(points, spb, closed, weights) {
    const n = points.length;
    if (n < 3) return points.slice();
    const W = (Array.isArray(weights) && weights.length === n)
        ? weights.map((w) => Math.max(1e-6, +w || 1)) : points.map(() => 1);
    let idx, segs;
    if (closed) { idx = [n - 1]; for (let i = 0; i < n; i++) idx.push(i); idx.push(0, 1); segs = n; }
    else { idx = [0, 0]; for (let i = 0; i < n; i++) idx.push(i); idx.push(n - 1, n - 1); segs = idx.length - 3; }
    const out = [], lastSeg = segs - 1;
    for (let s = 0; s < segs; s++) {
        const c = [idx[s], idx[s + 1], idx[s + 2], idx[s + 3]];
        const steps = spb + ((!closed && s === lastSeg) ? 1 : 0);
        for (let k = 0; k < steps; k++) {
            const t = k / spb, tv = [t * t * t, t * t, t, 1];
            const basis = [0, 0, 0, 0];
            for (let a = 0; a < 4; a++) { let su = 0; for (let b = 0; b < 4; b++) su += tv[b] * BSPLINE_M[b][a]; basis[a] = su; }
            let nx = 0, ny = 0, den = 0;
            for (let a = 0; a < 4; a++) { const P = points[c[a]], wb = basis[a] * W[c[a]]; nx += wb * P.x; ny += wb * P.y; den += wb; }
            out.push({ x: nx / den, y: ny / den });
        }
    }
    return out;
}

function solveLinear(A, b) {  // small dense Gaussian elimination (n<=~64)
    const n = b.length, M = A.map((r, i) => r.concat(b[i]));
    for (let col = 0; col < n; col++) {
        let piv = col; for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
        [M[col], M[piv]] = [M[piv], M[col]];
        const d = M[col][col] || 1e-12;
        for (let r = 0; r < n; r++) { if (r === col) continue; const f = M[r][col] / d; for (let cc = col; cc <= n; cc++) M[r][cc] -= f * M[col][cc]; }
    }
    return M.map((r, i) => r[n] / (r[i] || 1e-12));
}

function naturalCubic1d(y, closed) {
    const n = y.length, k = new Array(n).fill(0);
    if (closed) {
        const A = Array.from({ length: n }, () => new Array(n).fill(0)), rhs = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            A[i][(i - 1 + n) % n] = 1; A[i][i] = 4; A[i][(i + 1) % n] = 1;
            rhs[i] = 6 * (y[(i + 1) % n] - 2 * y[i] + y[(i - 1 + n) % n]);
        }
        return solveLinear(A, rhs);
    }
    if (n >= 3) {
        const m = n - 2, A = Array.from({ length: m }, () => new Array(m).fill(0)), rhs = new Array(m).fill(0);
        for (let i = 0; i < m; i++) { A[i][i] = 4; if (i > 0) A[i][i - 1] = 1; if (i < m - 1) A[i][i + 1] = 1; rhs[i] = 6 * (y[i + 2] - 2 * y[i + 1] + y[i]); }
        const sol = solveLinear(A, rhs); for (let i = 0; i < m; i++) k[i + 1] = sol[i];
    }
    return k;
}

function naturalCubic(points, spb, closed) {
    const n = points.length;
    if (n < 3) return points.slice();
    const kx = naturalCubic1d(points.map((p) => p.x), closed);
    const ky = naturalCubic1d(points.map((p) => p.y), closed);
    const segs = closed ? n : n - 1, out = [], lastSeg = segs - 1;
    for (let i = 0; i < segs; i++) {
        const j = (i + 1) % n, steps = spb + ((!closed && i === lastSeg) ? 1 : 0);
        for (let s = 0; s < steps; s++) {
            const t = s / spb, u = 1 - t;
            const cx = u * points[i].x + t * points[j].x + ((u ** 3 - u) * kx[i] + (t ** 3 - t) * kx[j]) / 6;
            const cy = u * points[i].y + t * points[j].y + ((u ** 3 - u) * ky[i] + (t ** 3 - t) * ky[j]) / 6;
            out.push({ x: cx, y: cy });
        }
    }
    return out;
}

function cardinalSpline(points, spb, closed, tension) {
    const n = points.length;
    if (n < 3) return points.slice();
    const c = 1 - Math.max(0, Math.min(1, +tension || 0));
    let ext;
    if (closed) ext = [points[n - 1], ...points, points[0], points[1]];
    else ext = [{ x: 2 * points[0].x - points[1].x, y: 2 * points[0].y - points[1].y }, ...points,
                { x: 2 * points[n - 1].x - points[n - 2].x, y: 2 * points[n - 1].y - points[n - 2].y }];
    const segs = closed ? n : n - 1, out = [], lastSeg = segs - 1;
    for (let i = 0; i < segs; i++) {
        const p0 = ext[i], p1 = ext[i + 1], p2 = ext[i + 2], p3 = ext[i + 3];
        const m1x = c * (p2.x - p0.x) * 0.5, m1y = c * (p2.y - p0.y) * 0.5;
        const m2x = c * (p3.x - p1.x) * 0.5, m2y = c * (p3.y - p1.y) * 0.5;
        const steps = spb + ((!closed && i === lastSeg) ? 1 : 0);
        for (let s = 0; s < steps; s++) {
            const t = s / spb, t2 = t * t, t3 = t2 * t;
            const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
            out.push({ x: h00 * p1.x + h10 * m1x + h01 * p2.x + h11 * m2x,
                       y: h00 * p1.y + h10 * m1y + h01 * p2.y + h11 * m2y });
        }
    }
    return out;
}

function sampleFamily(pts, type, spb, closed, weights, tension) {
    if (type === "b_spline") return bsplineLike(pts, spb, closed);
    if (type === "nurbs") return bsplineLike(pts, spb, closed, weights);
    if (type === "natural_cubic") return naturalCubic(pts, spb, closed);
    if (type === "cardinal") return cardinalSpline(pts, spb, closed, tension);
    return catmullRom(pts, spb, closed, 0.5);
}

function sampleWithCusps(sh, type, spb) {
    const pts = sh.points, n = pts.length;
    if (n < 3) return pts.slice();
    const cusps = sh.cusps;
    if (!Array.isArray(cusps) || !cusps.some(Boolean))
        return sampleFamily(pts, type, spb, sh.closed, sh.weights, sh.tension);
    const corner = pts.map((_, i) => !!cusps[i]);
    const order = pts.map((_, i) => i).concat(sh.closed ? [0] : []);
    const runs = []; let cur = [];
    for (const i of order) { cur.push(i); if (corner[i] && cur.length > 1) { runs.push(cur); cur = [i]; } }
    if (cur.length > 1) runs.push(cur);
    const out = [];
    for (const run of runs) {
        const rp = run.map((i) => pts[i]);
        const rw = sh.weights ? run.map((i) => sh.weights[i] ?? 1) : null;
        let seg = sampleFamily(rp, type, spb, false, rw, sh.tension);
        if (out.length && seg.length && Math.abs(out[out.length - 1].x - seg[0].x) < 1e-6
            && Math.abs(out[out.length - 1].y - seg[0].y) < 1e-6) seg = seg.slice(1);
        out.push(...seg);
    }
    return out;
}

// Parametric primitives (mirror _spline_curves.make_primitive) for preview.
const PRIMITIVE_TYPES = ["circle", "ellipse", "rectangle", "rounded_rect", "polygon", "star", "arc"];
function makePrimitive(type, pts, samples, params) {
    params = params || {};
    if (pts.length < 2) return pts.slice();
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const x0 = Math.min(...xs), y0 = Math.min(...ys), x1 = Math.max(...xs), y1 = Math.max(...ys);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const rx = Math.max(1e-6, (x1 - x0) / 2), ry = Math.max(1e-6, (y1 - y0) / 2);
    const rot = +params.rotation || 0, cr = Math.cos(rot), sr = Math.sin(rot);
    const place = (dx, dy) => ({ x: cx + dx * cr - dy * sr, y: cy + dx * sr + dy * cr });
    const n = Math.max(3, samples | 0), TAU = Math.PI * 2;
    if (type === "circle" || type === "ellipse") {
        const r = type === "circle" ? Math.min(rx, ry) : null;
        return Array.from({ length: n }, (_, i) => place((r || rx) * Math.cos(TAU * i / n), (r || ry) * Math.sin(TAU * i / n)));
    }
    if (type === "rectangle") return [place(-rx, -ry), place(rx, -ry), place(rx, ry), place(-rx, ry)];
    if (type === "rounded_rect") {
        let rad = params.corner_radius != null ? +params.corner_radius : Math.min(rx, ry) * 0.25;
        rad = Math.max(0, Math.min(rad, Math.min(rx, ry)));
        const per = Math.max(2, n >> 2), out = [];
        [[rx - rad, ry - rad, 0], [-rx + rad, ry - rad, Math.PI / 2], [-rx + rad, -ry + rad, Math.PI], [rx - rad, -ry + rad, 1.5 * Math.PI]]
            .forEach(([ccx, ccy, a0]) => { for (let k = 0; k <= per; k++) { const a = a0 + (Math.PI / 2) * (k / per); out.push(place(ccx + rad * Math.cos(a), ccy + rad * Math.sin(a))); } });
        return out;
    }
    if (type === "polygon") {
        const sides = Math.max(3, params.sides | 0 || 6), a0 = params.start_angle != null ? +params.start_angle : -Math.PI / 2;
        return Array.from({ length: sides }, (_, i) => place(rx * Math.cos(a0 + TAU * i / sides), ry * Math.sin(a0 + TAU * i / sides)));
    }
    if (type === "star") {
        const sides = Math.max(2, params.sides | 0 || 5), inner = Math.max(0.05, Math.min(1, params.inner_ratio != null ? +params.inner_ratio : 0.5));
        const a0 = params.start_angle != null ? +params.start_angle : -Math.PI / 2, out = [];
        for (let i = 0; i < sides * 2; i++) { const rr = i % 2 === 0 ? 1 : inner, a = a0 + Math.PI * i / sides; out.push(place(rx * rr * Math.cos(a), ry * rr * Math.sin(a))); }
        return out;
    }
    if (type === "arc") {
        const a0 = +params.start_angle || 0, a1 = params.end_angle != null ? +params.end_angle : Math.PI, r = Math.min(rx, ry);
        return Array.from({ length: n }, (_, i) => place(r * Math.cos(a0 + (a1 - a0) * i / (n - 1)), r * Math.sin(a0 + (a1 - a0) * i / (n - 1))));
    }
    return pts.slice();
}

function sampleShape(sh, samplesPerSeg, alpha) {
    const pts = sh.points;
    if (pts.length < 2) return pts.slice();
    if (PRIMITIVE_TYPES.includes(sh.type))
        return makePrimitive(sh.type, pts, Math.max(3, samplesPerSeg * 3), sh.params);
    if (sh.type === "polyline") return pts.slice();
    if (sh.type === "bezier")   return bezierCubic(pts, samplesPerSeg, sh.closed, sh.handles);
    if (sh.type === "b_spline" || sh.type === "nurbs" || sh.type === "natural_cubic" || sh.type === "cardinal")
        return sampleWithCusps(sh, sh.type, samplesPerSeg);
    return catmullRom(pts, samplesPerSeg, sh.closed, alpha);
}

function draw(ed, ctx, vw, vh) {
    ctx.save();
    // Outer fill MUST be a resolved literal — Canvas2D cannot parse the raw
    // `var(--c2c-bg2)` that COLOR.bg holds, so it silently rendered BLACK: that
    // was the "dead space" letterbox the user saw. With no backdrop image the
    // whole area IS the drawing surface (bg3); with an image we letterbox
    // against a soft neutral (bg2) instead of harsh black.
    ctx.fillStyle = ed.refImg ? C.bg2 : C.bg3;
    ctx.fillRect(0, 0, vw, vh);

    const z = ed.zoom;
    ctx.translate(ed.panX, ed.panY);
    ctx.scale(z, z);

    if (ed.refImg && ed.refImg.complete) {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        try { ctx.drawImage(ed.refImg, 0, 0, ed.canvasW, ed.canvasH); } catch (__c2cErr) { __c2cReport("spline_mask_editor", __c2cErr); }
    } else if (!ed.shapes.some(s => s.points.length > 0)) {
        drawEditorEmptyState(ctx, ed.canvasW, ed.canvasH, z, "✎", [
            "Click to place spline points",
            "Drag points to shape the mask · + Path starts another shape",
            "Connect an image for a reference backdrop",
        ]);
    } else {
        ctx.fillStyle = C.bg3;
        ctx.fillRect(0, 0, ed.canvasW, ed.canvasH);
    }
    ctx.strokeStyle = C.surface1;
    ctx.lineWidth = 1 / z;
    ctx.strokeRect(0, 0, ed.canvasW, ed.canvasH);

    for (let s = 0; s < ed.shapes.length; s++) {
        const sh = ed.shapes[s];
        // Resolved literals — COLOR.paths holds var(--…) strings which canvas
        // cannot parse (strokeStyle/fillStyle silently keep their previous
        // value). C.* resolves to real hex from the active theme.
        const PATHC = [C.green, C.blue, C.yellow, C.peach, C.pink, C.teal];
        const c = PATHC[s % PATHC.length];
        const isActive = s === ed.active;

        if (sh.points.length >= 2) {
            const sampled = sampleShape(sh, ed.samplesPerSegment, ed.centripetalAlpha);
            ctx.strokeStyle = c;
            ctx.lineWidth = (isActive ? 2.0 : 1.5) / z;
            ctx.beginPath();
            ctx.moveTo(sampled[0].x, sampled[0].y);
            for (let i = 1; i < sampled.length; i++) ctx.lineTo(sampled[i].x, sampled[i].y);
            if (sh.closed) {
                ctx.closePath();
                // globalAlpha instead of `c + "22"`: COLOR.paths entries were
                // var(--…) strings, so the concatenated fillStyle was INVALID
                // and canvas silently kept the LAST valid fill — the dark
                // backdrop — filling the whole shape opaque black over the
                // image (the reported bug). Nuke-style: image stays visible.
                ctx.fillStyle = c;
                ctx.globalAlpha = 0.14;
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }
            ctx.stroke();
        }

        for (let i = 0; i < sh.points.length; i++) {
            const p = sh.points[i];
            const isHov = ed.hover.shape === s && ed.hover.point === i;
            const r = (isHov ? 6 : 4) / z;
            ctx.fillStyle = c;
            ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = C.black;
            ctx.lineWidth = 1.5 / z;
            ctx.stroke();
            if (isActive && i === 0) {
                ctx.strokeStyle = C.white;
                ctx.lineWidth = 1 / z;
                ctx.beginPath(); ctx.arc(p.x, p.y, r + 2 / z, 0, Math.PI * 2); ctx.stroke();
            }
        }

        // Bezier handle endpoints + connector lines — Nuke behaviour: the
        // tangents always shape the curve, but the handle UI only appears
        // for the point you are working with (hovered or mid-drag), not
        // splayed across every vertex.
        if (isActive && sh.type === "bezier" && Array.isArray(sh.handles)) {
            for (let i = 0; i < sh.points.length; i++) {
                const engaged = (ed.hover.shape === s && ed.hover.point === i)
                    || (ed.drag && ed.drag.point === i);
                if (!engaged) continue;
                const p = sh.points[i];
                const h = sh.handles[i];
                if (!h) continue;
                for (const side of ["in", "out"]) {
                    const hv = h[side];
                    if (!hv) continue;
                    const hx = p.x + hv.x, hy = p.y + hv.y;
                    // Canvas can't parse var(); use the theme-resolved color.
                    // (was "var(--c2c-peach)aa" → silently rendered as default/black.)
                    ctx.save();
                    ctx.globalAlpha = 0.67;             // the intended ~"aa" alpha
                    ctx.strokeStyle = C.peach;
                    ctx.lineWidth = 1 / z;
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(hx, hy);
                    ctx.stroke();
                    ctx.restore();
                    ctx.fillStyle = C.peach;
                    ctx.beginPath();
                    ctx.arc(hx, hy, 3.5 / z, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = C.black;
                    ctx.lineWidth = 1 / z;
                    ctx.stroke();
                }
            }
        }
    }

    ctx.restore();
}

function findRefImage(node) {
    if (!node.inputs) return null;
    const linkedInput = node.inputs.find(i => i.name === "image" && i.link != null);
    if (!linkedInput) return null;
    // Resolve links within the node's owning graph (subgraph-safe).
    const ownerGraph = node.graph || app.graph;
    const link = ownerGraph?.links?.[linkedInput.link];
    if (!link) return null;

    const visited = new Set();
    const queue = [{ id: link.origin_id, depth: 0 }];
    while (queue.length) {
        const { id, depth } = queue.shift();
        if (visited.has(id)) continue;
        visited.add(id);
        const n = ownerGraph.getNodeById?.(id);
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
                const li = ownerGraph.links?.[inp.link];
                if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
            }
        }
    }
    return null;
}

function installEditor(node) {
    if (node._mecSplineEditHost) return;
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
        const ca = node.widgets?.find(x => x.name === "centripetal_alpha");
        if (t) ed.splineType = t.value;
        if (c) ed.closedDefault = !!c.value;
        if (sps) ed.samplesPerSegment = +sps.value || 20;
        if (w && +w.value > 0) ed.canvasW = +w.value;
        if (h && +h.value > 0) ed.canvasH = +h.value;
        if (ca && typeof ca.value === "number") ed.centripetalAlpha = ca.value;
    };
    syncFromWidgets();
    ed.load();

    ensureC2CKit();
    const root = document.createElement("div");
    root.className = "c2ck";
    // Inset from node body — leaves a hit area for LiteGraph border / resize
    // corner so the user can grab edges and the bottom-right grip.
    root.style.cssText = `
        position:relative;
        display:flex;flex-direction:column;
        width:calc(100% - 12px);height:calc(100% - 18px);
        margin:2px 6px 16px 6px;
        background:#161616;border:1px solid #111;border-radius:7px;
        overflow:hidden;color:#e6e6e6;
        box-sizing:border-box;user-select:none;
        pointer-events:none;
    `;
    // ^ position:relative is REQUIRED: the ≡ actions menu is position:absolute
    //   (top:34px;right:6px) and must anchor to THIS root — without it the menu
    //   resolved against a higher positioned ancestor and appeared OFFSET from
    //   the button (worse at graph zoom).

    const tb = document.createElement("div");
    tb.className = "c2ck-toolbar";
    tb.style.cssText = `
        display:flex;align-items:center;gap:4px;padding:5px 7px;
        background:#1e1e1e;border-bottom:1px solid #111;
        flex:0 0 auto;font-size:11px;line-height:1;
        pointer-events:auto;
    `;
    root.appendChild(tb);

    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText = "position:relative;flex:1 1 auto;min-height:0;overflow:hidden;cursor:crosshair;background:var(--c2c-bg3);pointer-events:auto;";
    root.appendChild(canvasWrap);

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;width:100%;height:100%;outline:none;";
    canvas.tabIndex = 0;
    canvasWrap.appendChild(canvas);

    const status = document.createElement("div");
    status.style.cssText = `
        position:absolute;left:6px;bottom:6px;padding:3px 7px;
        background:var(--c2c-bg)d8;border:1px solid ${COLOR.border};border-radius:4px;
        font-size:10px;color:${COLOR.sub};pointer-events:none;
        font-family:ui-monospace,Menlo,monospace;letter-spacing:.2px;
    `;
    canvasWrap.appendChild(status);

    const ctx = canvas.getContext("2d");

    const mkIcon = (label, title, onClick) => {
        const b = document.createElement("button");
        b.textContent = label; b.title = title;
        b.className = "c2ck-btn c2ck-btn-icon";
        b.onmousedown = (e) => e.stopPropagation();
        b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); onClick(b); render(); };
        return b;
    };

    const counter = document.createElement("span");
    counter.style.cssText = `
        color:${COLOR.text};font-size:10px;flex:1 1 auto;
        font-family:ui-monospace,Menlo,monospace;
        padding:3px 8px;background:var(--c2c-bg3);border:1px solid ${COLOR.border};
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

    // ≡ menu — mounted on document.body with position:fixed and a top-level
    // z-index. Mounted inside the node it sat UNDER third-party overlays
    // (rgthree's <rgthree-progress-bar> intercepted every click at the menu's
    // coordinates — the reported "menu buttons do nothing"). Items act on
    // pointerdown so nothing downstream can swallow the interaction.
    root.style.position = "relative";
    const MENU_ITEMS = () => [
        { label: "Reset zoom (100%)", run: () => { ed.zoom = 1; ed.panX = 0; ed.panY = 0; } },
        { label: "Fit to view", run: () => {
            const r = canvasWrap.getBoundingClientRect(); ed.fitView(r.width, r.height);
        } },
        { sep: true },
        { label: "Clear active path", danger: true, run: () => {
            if (ed.active < 0) return;
            ed.pushUndo(); ed.shapes[ed.active].points = []; ed.save();
        } },
        { label: "Delete active path", danger: true, run: () => {
            if (ed.active < 0) return;
            ed.pushUndo();
            ed.shapes.splice(ed.active, 1);
            ed.active = ed.shapes.length ? Math.min(ed.active, ed.shapes.length - 1) : -1;
            ed.save();
        } },
        { label: "Clear all paths", danger: true, run: () => {
            if (!ed.shapes.length) return;
            ed.pushUndo(); ed.shapes = []; ed.active = -1; ed.save();
        } },
    ];

    let _openMenuEl = null;
    function _closeMenu() {
        if (_openMenuEl) { try { _openMenuEl.remove(); } catch (_) {} _openMenuEl = null; }
    }
    function toggleMenu() {
        if (_openMenuEl) { _closeMenu(); return; }
        const menu = document.createElement("div");
        const br = menuBtn.getBoundingClientRect();
        menu.style.cssText = `
            position:fixed;top:${Math.round(br.bottom + 4)}px;z-index:2147483000;
            background:var(--c2c-bg);border:1px solid ${COLOR.border};border-radius:6px;
            box-shadow:0 6px 20px rgba(0,0,0,0.6);padding:4px;
            min-width:200px;font-size:11px;
        `;
        // right-align to the button, clamped to the viewport
        menu.style.left = Math.max(8, Math.round(br.right - 208)) + "px";
        for (const item of MENU_ITEMS()) {
            if (item.sep) {
                const s = document.createElement("div");
                s.style.cssText = `height:1px;background:${COLOR.border};margin:4px 2px;`;
                menu.appendChild(s);
                continue;
            }
            const it = document.createElement("div");
            it.textContent = item.label;
            it.style.cssText = `
                padding:7px 12px;border-radius:4px;cursor:pointer;
                color:${item.danger ? "var(--c2c-red)" : COLOR.text};white-space:nowrap;
            `;
            it.onmouseenter = () => it.style.background = C.border;
            it.onmouseleave = () => it.style.background = "";
            it.onpointerdown = (e) => {
                e.stopPropagation(); e.preventDefault();
                _closeMenu();
                item.run(); render();
            };
            menu.appendChild(it);
        }
        document.body.appendChild(menu);
        _openMenuEl = menu;
        const off = (e) => {
            if (!menu.contains(e.target) && e.target !== menuBtn) {
                _closeMenu();
                document.removeEventListener("pointerdown", off, true);
            }
        };
        setTimeout(() => document.addEventListener("pointerdown", off, true), 0);
    }

    let widgetH = 300;
    const EDIT_MIN_H = 200, EDIT_MAX_H = 860;
    const canvasWidget = node.addDOMWidget("spline_editor", "canvas", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => widgetH,
        getHeight: () => widgetH,
        getMaxHeight: () => EDIT_MAX_H,
    });
    // Size the DOM widget the proven way (mirrors the working FaceController3D
    // editor): modern ComfyUI lays DOM widgets out via `computeLayoutSize`
    // (min/max) + `getHeight`, and the element's OWN style.height must be set
    // directly — otherwise the widget stays pinned to its min height and the node
    // leaves empty body below it (the "tiny image + dead space" bug). `widgetH`
    // is the single source of truth, driven by the image aspect in resizeForImage.
    canvasWidget.computeLayoutSize = () => ({ minHeight: EDIT_MIN_H, maxHeight: EDIT_MAX_H });
    canvasWidget.computeSize = function (width) {
        return [width, Math.max(EDIT_MIN_H, widgetH)];
    };
    try { root.style.height = widgetH + "px"; } catch (_) {}

    node._mecSplineEditHost = root;
    node._mecSplineEditWidget = canvasWidget;
    node._mecSplineEditWidgetH = () => widgetH;

    // Keep the node wide enough for a USABLE editor canvas while an editing mode
    // is active. The mode-gate calls `node.setSize(node.computeSize())`, and the
    // node's own computeSize returns the narrow widget-driven width (~280px) —
    // which snapped the node back from 600→280 and collapsed the canvas to
    // ~186px / "zoom 21%" (the "spline editor doesn't work" report). Enforce a
    // minimum width via computeSize so BOTH our setSize and the mode-gate's keep
    // the editor roomy. Only in editor modes, so non-editing modes stay compact.
    const MIN_EDIT_W = 600;
    const _editorModeActive = () => {
        const mv = node.widgets?.find?.((x) => x && x.name === "mode")?.value ?? "";
        // All three modes embed a full-width canvas editor (edit = spline draw,
        // track = keyframe frame-viewer, flow_path = flow editor) — each needs
        // the roomy width or its canvas collapses to ~186px. "track" was missing,
        // which is why track mode snapped back to ~281px / dead space.
        return mv === "edit" || mv === "flow_path" || mv === "track";
    };
    if (!node._mecSplineCSPatched) {
        node._mecSplineCSPatched = true;
        const _origCS = typeof node.computeSize === "function" ? node.computeSize.bind(node) : null;
        node.computeSize = function (w) {
            const sz = _origCS ? _origCS(w) : [MIN_EDIT_W, 380];
            try { if (_editorModeActive()) sz[0] = Math.max(sz[0] || 0, MIN_EDIT_W); } catch (_) {}
            return sz;
        };
    }

    node.setSize?.([Math.max(node.size?.[0] || 0, 600), Math.max(node.size?.[1] || 0, 380)]);

    // Grow the node so the editor's aspect matches the backdrop image → the photo
    // FILLS the editor (contain-fit fills both dims, no big margins). Uses the
    // proven FaceController3D height-sync: set the widget height + the element's
    // own style.height, then setSize the node to (top chrome + editor), guarded
    // against the onResize→setSize→onResize runaway.
    let _syncInFlight = false;
    function _syncEditorHeight() {
        if (_syncInFlight || !root) return;
        _syncInFlight = true;
        try {
            const cur = node.size || [0, 0];
            // Node height ABOVE the editor = current total minus the editor's
            // current footprint. Measure BEFORE we change the element height so the
            // math is stable (no circular reflow read).
            const topPx = Math.max(0, (cur[1] || 0) - (root.offsetHeight || widgetH));
            const h = Math.max(EDIT_MIN_H, Math.min(EDIT_MAX_H, Math.round(widgetH)));
            widgetH = h;
            root.style.height = h + "px";
            const target = Math.round(topPx + h + 18);   // +18 = root's 2px top / 16px bottom margin
            if (Math.abs(target - (cur[1] || 0)) > 3) {
                node.__mecSplineSyncing = true;
                try { node.setSize([cur[0], target]); } finally { node.__mecSplineSyncing = false; }
            }
            node.setDirtyCanvas?.(true, true);
        } catch (_) { /* never break the editor */ }
        finally { _syncInFlight = false; }
        // ComfyUI's DOM-widget wrapper grows a frame or two AFTER setSize; a single
        // immediate re-fit locks the image into the small INTERMEDIATE size (the
        // "zoom 37%" report). Staggered re-fits once the wrapper settles — same
        // proven pattern as the points editor. Idempotent + cheap.
        for (const d of [0, 60, 200, 400]) setTimeout(() => { ed._fitted = false; render(); }, d);
    }
    function resizeForImage() {
        if (!ed.canvasW || !ed.canvasH) return;
        const baseW = (node.size?.[0] || 600) - 24;
        if (baseW <= 0) return;
        const aspect = ed.canvasH / ed.canvasW;
        // editor height that makes the editor aspect == image aspect at full width
        // (+40 for the toolbar strip). Clamped so an extreme portrait stays sane.
        widgetH = Math.max(EDIT_MIN_H, Math.min(EDIT_MAX_H, Math.round(baseW * aspect) + 40));
        _syncEditorHeight();
    }

    let lastW = 0, lastH = 0;
    function syncCanvasPx() {
        // canvasWrap is flex:1 1 auto inside the flex-column root, but ComfyUI's
        // DOM-widget layout does NOT reliably distribute height to a <canvas>
        // child — the wrap collapses to the canvas's intrinsic height, so the
        // image auto-fit ran against a ~266px viewport ("zoom 37%") even though
        // the editor root was 530px+. Size it EXPLICITLY = root minus toolbar,
        // exactly like the points editor / tracker fix (see memory: the flex-fill
        // bug). Uses unscaled layout px (offsetHeight), transform-safe.
        try {
            const availH = (root.offsetHeight || 0) - (tb.offsetHeight || 0);
            if (availH > 40 && Math.abs((canvasWrap.offsetHeight || 0) - availH) > 2) {
                canvasWrap.style.height = availH + "px";
                canvasWrap.style.flex = "0 0 auto";
            }
        } catch (_) {}
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
        const sizeChanged = (w !== lastW) || (h !== lastH);
        if (!sizeChanged && !needPx) return false;
        lastW = w; lastH = h;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // No backdrop image + auto output dims + nothing drawn yet → make the
        // logical drawing surface EQUAL the viewport so it FILLS the node body
        // instead of letterboxing a fixed 512² into a wide, short strip. Frozen
        // the moment a shape exists so the user's points never rescale.
        if (!ed.refImg && ed.shapes.length === 0) {
            const wW = node.widgets?.find(x => x.name === "width");
            const hW = node.widgets?.find(x => x.name === "height");
            if ((!wW || +wW.value <= 0) && (!hW || +hW.value <= 0)) {
                ed.canvasW = w; ed.canvasH = h;
                ed.zoom = 1; ed.panX = 0; ed.panY = 0; ed._fitted = true;
            }
        }
        // Auto-fit ONLY on first sizing or when a new image set _fitted=false.
        // Previously we re-fit on every resize → any container reflow snapped
        // the user's zoom/pan back to "fit", which is the "bouncy image"
        // complaint. Now we just clamp pan into the new viewport instead.
        if (!ed._fitted) {
            ed.fitView(w, h);
        } else {
            ed.clampPan(w, h);
        }
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
    ed.onLoaded = () => { ed._fitted = false; resizeForImage(); render(); };

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
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {} // pointer may already be inactive (pen/touch/synthetic)
        const c = eventCanvas(e);

        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            const ec = eventClient(e);
            ed.drag = { kind: "pan", startX: ec.x - ed.panX, startY: ec.y - ed.panY };
            canvas.style.cursor = "grabbing";
            return;
        }
        // Bezier handle drag takes priority over point hit, on active shape only.
        const handleHit = (e.button === 0 && !e.shiftKey) ? ed.findHandle(c.x, c.y) : null;
        if (handleHit) {
            ed.pushUndo();
            ed.drag = { kind: "handle", shape: handleHit.shape, point: handleHit.point, side: handleHit.side,
                        mirror: !e.altKey };
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
        ed.ensureHandles(ed.shapes[ed.active]);
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
        if (ed.drag?.kind === "handle") {
            const sh = ed.shapes[ed.drag.shape];
            if (sh && Array.isArray(sh.handles)) {
                const p = sh.points[ed.drag.point];
                const h = sh.handles[ed.drag.point];
                if (p && h) {
                    const nx = +(c.x - p.x).toFixed(2);
                    const ny = +(c.y - p.y).toFixed(2);
                    h[ed.drag.side] = { x: nx, y: ny };
                    // Symmetric mirror unless Alt held when drag started.
                    if (ed.drag.mirror) {
                        const other = ed.drag.side === "in" ? "out" : "in";
                        h[other] = { x: -nx, y: -ny };
                    }
                    render();
                }
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
        try { canvas.releasePointerCapture(e.pointerId); } catch (__c2cErr) { __c2cReport("spline_mask_editor.releaseCapture", __c2cErr, { level: "info" }); } // expected when pointer already inactive
        if (ed.drag?.kind === "pan") {
            ed.drag = null; canvas.style.cursor = "crosshair"; return;
        }
        if (ed.drag?.kind === "point") {
            ed.drag = null; ed.save(); render(); return;
        }
        if (ed.drag?.kind === "handle") {
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
    for (const wn of ["spline_type", "closed", "samples_per_segment", "width", "height", "centripetal_alpha"]) {
        const w = node.widgets?.find(x => x.name === wn);
        if (!w) continue;
        const orig = w.callback;
        w.callback = function (v) {
            orig?.call(this, v);
            syncFromWidgets();
            // Apply spline_type / closed change to ALL existing shapes
            if (wn === "spline_type") {
                for (const sh of ed.shapes) {
                    sh.type = ed.splineType;
                    if (sh.type === "bezier") ed.ensureHandles(sh);
                }
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
        // Server may emit either key depending on version; accept both.
        const b64 = out?.preview_b64?.[0] ?? out?.preview?.[0];
        if (b64) {
            // Bare base64 → prepend data-url prefix; full data URL → pass through.
            const url = b64.startsWith("data:") ? b64 : ("data:image/jpeg;base64," + b64);
            ed.setRefImage(url);
        } else {
            tryDiscoverRef();
        }
    };
    const refPoll = setInterval(() => { if (!ed.refImg) tryDiscoverRef(); }, 1500);

    const origResize = node.onResize;
    node.onResize = function(...args) {
        origResize?.apply(this, args);
        render();
        node.setDirtyCanvas(true, true);
        node.graph?.setDirtyCanvas(true, true);
    };

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
        if (!NODE_NAMES.includes(nodeData.name)) return;
        const orig = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            orig?.apply(this, arguments);
            if (this._mecEditor) { this._mecEditor.load(); this._mecRender?.(); }
        };
    },
    async nodeCreated(node) {
        if (!NODE_NAMES.includes(node.comfyClass)) return;
        if (node.comfyClass === "SplineMaskMEC") {
            // Unified node: only install the editor when mode === "edit".
            installModeGated(node, {
                activeWhen: "edit",
                installerKey: "splineEdit",
                installer: (n) => installEditor(n),
                hostFinder: (n) => n._mecSplineEditHost || null,
                widgetFinder: (n) => n._mecSplineEditWidget || null,
                widgetHeight: (n) => (typeof n._mecSplineEditWidgetH === "function" ? n._mecSplineEditWidgetH() : 460),
            });
        } else {
            // Legacy direct binding (kept for backward graph loading only).
            installEditor(node);
        }
    },
});

// Export for reuse by sibling editors (e.g. SplinePathFlowMaskMEC).
// Stash on window so a separate module file can pick it up without a build step.
window.__MEC_SPLINE_EDITOR__ = { installEditor, IMG_CACHE };

