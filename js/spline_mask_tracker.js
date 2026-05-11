/**
 * SplineMaskTrackerMEC — multi-keyframe spline editor.
 *
 * UX:
 *   - Frame scrubber along the bottom: scrub through every frame in the
 *     upstream IMAGE batch (read from node.imgs[] of the source after the
 *     graph has been executed at least once; before that we show a
 *     prompt asking the user to Queue once so we can see the frames).
 *   - Click on the canvas adds a control point to the active keyframe.
 *     Drag points to move them. Shift / right-click deletes.
 *   - "Set keyframe" pins the current frame's control points into
 *     keyframes_json with frame=<current_frame>. Pinned frames are shown
 *     as bright marks on the scrubber.
 *   - "Clear keyframe" removes the pin for the current frame.
 *   - When scrubbing to a NON-pinned frame, we show a LINEARLY
 *     INTERPOLATED preview between the surrounding keyframes (so the
 *     user can see roughly what the tracker will do; the Python node
 *     does the real LK tracking at execute time).
 *   - "Onion skin": light overlay of adjacent keyframes' shapes for
 *     reference.
 *
 * On widget value change we serialize:
 *   [{"frame": <int>, "points": [[x,y],...]}, ...]
 * sorted by frame, into the hidden keyframes_json widget.
 */

import { app } from "../../scripts/app.js";

const NODE_NAME = "SplineMaskTrackerMEC";

const COLOR = {
    bg:     "#181825",
    border: "#313244",
    text:   "#cdd6f4",
    sub:    "#7f849c",
    accent: "#a6e3a1",   // pinned keyframe color
    interp: "#89b4fa",   // interpolated preview color
    onion:  "#f9e2af",
    pin:    "#fab387",
    danger: "#f38ba8",
};

const POINT_HIT_PX  = 8;
const SCRUB_HEIGHT  = 28;
const HISTORY_LIMIT = 80;

// Module-level decoded-image cache (URL -> {img,w,h}). Same idea as
// spline_mask_editor.js — avoids re-decoding every frame on every scrub.
const IMG_CACHE = new Map();
const IMG_CACHE_MAX = 256;
function _cacheGet(url) { return IMG_CACHE.get(url) || null; }
function _cachePut(url, entry) {
    IMG_CACHE.set(url, entry);
    if (IMG_CACHE.size > IMG_CACHE_MAX) {
        const k = IMG_CACHE.keys().next().value;
        if (k !== undefined) IMG_CACHE.delete(k);
    }
}


// ─── Tracker state ──────────────────────────────────────────────────
class TrackerState {
    constructor(node) {
        this.node = node;
        // {frame -> [{x,y},...]}
        this.keyframes = new Map();
        // currently-scrubbed frame index
        this.curFrame = 0;
        // canvas (image) dims — set after we load first frame
        this.canvasW = 512;
        this.canvasH = 512;
        // pixel viewport state
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this._fitted = false;
        // image frames: array of {url, img}
        this.frames = [];
        this.frameUrls = [];
        // editing
        this.hover = { point: -1 };
        this.drag = null;
        this.cursor = { x: 0, y: 0, visible: false };
        // settings (mirror from widgets)
        this.closed = true;
        this.samplesPerSegment = 24;
        // history
        this.undoStack = [];
        this.redoStack = [];
        // onion skin
        this.onionEnabled = true;
    }

    snapshot() {
        return JSON.stringify({
            k: [...this.keyframes.entries()],
            f: this.curFrame,
        });
    }
    pushUndo() {
        this.undoStack.push(this.snapshot());
        if (this.undoStack.length > HISTORY_LIMIT) this.undoStack.shift();
        this.redoStack.length = 0;
    }
    restore(s) {
        const o = JSON.parse(s);
        this.keyframes = new Map(o.k.map(([f, pts]) =>
            [Number(f), pts.map(p => ({ x: +p.x, y: +p.y }))]
        ));
        this.curFrame = o.f ?? 0;
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

    /** Get the displayed control points for the current frame.
     * If the frame is pinned, return those. Otherwise return a linear
     * interpolation between the surrounding keyframes (or the nearest
     * one if we're outside the range). Returns null if no keyframes. */
    pointsAtFrame(f) {
        if (this.keyframes.has(f)) return this.keyframes.get(f);
        const sorted = [...this.keyframes.keys()].sort((a, b) => a - b);
        if (!sorted.length) return null;
        if (f <= sorted[0]) return this.keyframes.get(sorted[0]);
        if (f >= sorted[sorted.length - 1]) return this.keyframes.get(sorted[sorted.length - 1]);
        let prev = sorted[0], next = sorted[sorted.length - 1];
        for (let i = 0; i < sorted.length - 1; i++) {
            if (sorted[i] <= f && sorted[i + 1] >= f) {
                prev = sorted[i]; next = sorted[i + 1]; break;
            }
        }
        const a = this.keyframes.get(prev);
        const b = this.keyframes.get(next);
        // Enforce same length — if mismatched, fall back to nearest.
        if (a.length !== b.length) {
            return Math.abs(f - prev) <= Math.abs(f - next) ? a : b;
        }
        const t = (f - prev) / (next - prev);
        return a.map((p, i) => ({
            x: p.x + (b[i].x - p.x) * t,
            y: p.y + (b[i].y - p.y) * t,
        }));
    }

    isPinned(f) { return this.keyframes.has(f); }

    /** Pin or update the current frame's control points. */
    pinCurrent(points) {
        this.keyframes.set(this.curFrame,
            points.map(p => ({ x: +p.x.toFixed(2), y: +p.y.toFixed(2) })));
    }
    unpinCurrent() { this.keyframes.delete(this.curFrame); }

    viewToCanvas(vx, vy) {
        return { x: (vx - this.panX) / this.zoom, y: (vy - this.panY) / this.zoom };
    }
    minZoom(vw, vh) {
        if (vw <= 0 || vh <= 0) return 0.05;
        return Math.min(vw / this.canvasW, vh / this.canvasH);
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
    setZoomAround(newZ, ax, ay, vw, vh) {
        const minZ = this.minZoom(vw, vh);
        newZ = Math.max(minZ, Math.min(8, newZ));
        const k = newZ / this.zoom;
        this.panX = ax - (ax - this.panX) * k;
        this.panY = ay - (ay - this.panY) * k;
        this.zoom = newZ;
    }

    findPoint(cx, cy) {
        const pts = this.pointsAtFrame(this.curFrame);
        if (!pts || !this.keyframes.has(this.curFrame)) {
            // Only allow hitting points when current frame is pinned —
            // otherwise the points are an interpolated preview.
            return { point: -1 };
        }
        const r = POINT_HIT_PX / this.zoom; const r2 = r * r;
        for (let i = pts.length - 1; i >= 0; i--) {
            const p = pts[i];
            if ((p.x - cx) ** 2 + (p.y - cy) ** 2 < r2) return { point: i };
        }
        return { point: -1 };
    }

    save() {
        const w = this.node.widgets?.find(w => w.name === "keyframes_json");
        if (!w) return;
        const sorted = [...this.keyframes.entries()].sort((a, b) => a[0] - b[0]);
        const out = sorted.map(([f, pts]) => ({
            frame: f,
            points: pts.map(p => [+p.x.toFixed(2), +p.y.toFixed(2)]),
        }));
        w.value = JSON.stringify(out);
        if (this.node.graph) this.node.graph.setDirtyCanvas(true, false);
    }
    load() {
        const w = this.node.widgets?.find(w => w.name === "keyframes_json");
        if (!w?.value) return;
        try {
            const o = JSON.parse(w.value);
            if (!Array.isArray(o)) return;
            this.keyframes = new Map();
            for (const kf of o) {
                if (!kf || typeof kf.frame !== "number" || !Array.isArray(kf.points)) continue;
                const pts = kf.points
                    .filter(p => Array.isArray(p) && p.length >= 2)
                    .map(p => ({ x: +p[0], y: +p[1] }));
                if (pts.length) this.keyframes.set(kf.frame | 0, pts);
            }
        } catch (_) {}
    }
}


// ─── Catmull-Rom (mirrors python centripetal alpha=0.5) ─────────────
function catmullRom(points, samplesPerSeg, closed) {
    const n = points.length;
    if (n < 2) return points.slice();
    if (n === 2) {
        const [a, b] = points; const out = [];
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
    const a = 0.5;
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


// ─── Upstream frame discovery ──────────────────────────────────────
/** Walk the input chain from `node` looking for a node with imgs[].length
 *  > 0 (LoadImage / VHS_LoadVideo / SaveImage etc populate this after a
 *  graph execution). Return an array of URLs. */
function findUpstreamFrames(node) {
    if (!node.inputs) return [];
    const linkedInput = node.inputs.find(i => i.name === "image" && i.link != null);
    if (!linkedInput) return [];
    const visited = new Set();
    const queue = [{ id: app.graph.links[linkedInput.link]?.origin_id, depth: 0 }];
    while (queue.length) {
        const { id, depth } = queue.shift();
        if (id == null || visited.has(id)) continue;
        visited.add(id);
        const n = app.graph.getNodeById(id);
        if (!n) continue;
        if (n.imgs?.length) return n.imgs.map(im => im.src);
        // Some video loaders use videos[] instead.
        const w = n.widgets?.find(w => w.name === "image" || w.name === "video");
        if (w?.value && typeof w.value === "string") {
            const parts = w.value.split("/");
            const sub = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
            const fn = parts[parts.length - 1];
            return [`/view?filename=${encodeURIComponent(fn)}&subfolder=${encodeURIComponent(sub)}&type=input`];
        }
        if (depth < 4 && n.inputs) {
            for (const inp of n.inputs) {
                if (inp.link == null) continue;
                const li = app.graph.links[inp.link];
                if (li) queue.push({ id: li.origin_id, depth: depth + 1 });
            }
        }
    }
    return [];
}


// ─── Painter ────────────────────────────────────────────────────────
function draw(state, ctx, vw, vh) {
    ctx.save();
    ctx.fillStyle = COLOR.bg;
    ctx.fillRect(0, 0, vw, vh);

    const z = state.zoom;
    ctx.translate(state.panX, state.panY);
    ctx.scale(z, z);

    // Background frame.
    const frame = state.frames[state.curFrame];
    if (frame?.img?.complete) {
        ctx.imageSmoothingEnabled = true;
        try { ctx.drawImage(frame.img, 0, 0, state.canvasW, state.canvasH); } catch (_) {}
    } else {
        ctx.fillStyle = "#11111b";
        ctx.fillRect(0, 0, state.canvasW, state.canvasH);
    }
    ctx.strokeStyle = "#45475a";
    ctx.lineWidth = 1 / z;
    ctx.strokeRect(0, 0, state.canvasW, state.canvasH);

    // Onion skin: nearest pinned keyframes (one before, one after).
    if (state.onionEnabled && state.keyframes.size >= 2) {
        const sorted = [...state.keyframes.keys()].sort((a, b) => a - b);
        let prev = null, next = null;
        for (const f of sorted) {
            if (f < state.curFrame) prev = f;
            else if (f > state.curFrame && next === null) next = f;
        }
        for (const f of [prev, next]) {
            if (f == null) continue;
            const pts = state.keyframes.get(f);
            if (!pts || pts.length < 2) continue;
            const sampled = catmullRom(pts, state.samplesPerSegment, state.closed);
            ctx.strokeStyle = COLOR.onion + "70";
            ctx.lineWidth = 1.5 / z;
            ctx.setLineDash([4 / z, 3 / z]);
            ctx.beginPath();
            ctx.moveTo(sampled[0].x, sampled[0].y);
            for (let i = 1; i < sampled.length; i++) ctx.lineTo(sampled[i].x, sampled[i].y);
            if (state.closed) ctx.closePath();
            ctx.stroke();
            ctx.setLineDash([]);
            // tiny point markers
            for (const p of pts) {
                ctx.fillStyle = COLOR.onion + "60";
                ctx.beginPath(); ctx.arc(p.x, p.y, 2 / z, 0, Math.PI * 2); ctx.fill();
            }
        }
    }

    // Current frame's points (real or interpolated).
    const points = state.pointsAtFrame(state.curFrame);
    const pinned = state.isPinned(state.curFrame);
    if (points && points.length >= 2) {
        const sampled = catmullRom(points, state.samplesPerSegment, state.closed);
        ctx.strokeStyle = pinned ? COLOR.accent : COLOR.interp;
        ctx.lineWidth = (pinned ? 2.2 : 1.6) / z;
        if (!pinned) ctx.setLineDash([6 / z, 4 / z]);
        ctx.fillStyle = (pinned ? COLOR.accent : COLOR.interp) + "20";
        ctx.beginPath();
        ctx.moveTo(sampled[0].x, sampled[0].y);
        for (let i = 1; i < sampled.length; i++) ctx.lineTo(sampled[i].x, sampled[i].y);
        if (state.closed) { ctx.closePath(); ctx.fill(); }
        ctx.stroke();
        ctx.setLineDash([]);
    }
    if (points) {
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            const hov = state.hover.point === i && pinned;
            const r = (hov ? 6 : 4) / z;
            ctx.fillStyle = pinned ? COLOR.accent : COLOR.interp;
            ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = "#000a";
            ctx.lineWidth = 1.5 / z;
            ctx.stroke();
            if (pinned && i === 0) {
                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 1 / z;
                ctx.beginPath(); ctx.arc(p.x, p.y, r + 2 / z, 0, Math.PI * 2); ctx.stroke();
            }
        }
    }

    ctx.restore();
}


// ─── DOM build ──────────────────────────────────────────────────────
function installEditor(node) {
    const state = new TrackerState(node);

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
    hideWidget(node.widgets?.find(w => w.name === "keyframes_json"));

    const syncFromWidgets = () => {
        const c = node.widgets?.find(x => x.name === "closed");
        const sps = node.widgets?.find(x => x.name === "samples_per_segment");
        if (c) state.closed = !!c.value;
        if (sps) state.samplesPerSegment = +sps.value || 24;
    };
    syncFromWidgets();
    state.load();

    // Layout.
    const root = document.createElement("div");
    root.style.cssText = `
        display:flex;flex-direction:column;
        width:calc(100% - 12px);height:calc(100% - 18px);
        margin:2px 6px 16px 6px;
        background:${COLOR.bg};border:1px solid ${COLOR.border};border-radius:6px;
        overflow:hidden;font-family:Inter,system-ui,sans-serif;color:${COLOR.text};
        box-sizing:border-box;user-select:none;position:relative;
        pointer-events:none;
    `;

    const tb = document.createElement("div");
    tb.style.cssText = `
        display:flex;align-items:center;gap:4px;padding:4px 6px;flex-wrap:wrap;
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
        position:absolute;left:6px;bottom:${SCRUB_HEIGHT + 6}px;padding:3px 7px;
        background:#1e1e2ed8;border:1px solid ${COLOR.border};border-radius:4px;
        font-size:10px;color:${COLOR.sub};pointer-events:none;
        font-family:ui-monospace,Menlo,monospace;letter-spacing:.2px;
    `;
    canvasWrap.appendChild(status);

    // Scrubber strip across the bottom of canvasWrap.
    const scrub = document.createElement("div");
    scrub.style.cssText = `
        position:absolute;left:0;right:0;bottom:0;height:${SCRUB_HEIGHT}px;
        background:#11111be0;border-top:1px solid ${COLOR.border};
        cursor:pointer;
    `;
    canvasWrap.appendChild(scrub);

    const ctx = canvas.getContext("2d");
    const scrubCtx = (() => {
        const c2 = document.createElement("canvas");
        c2.style.cssText = "display:block;width:100%;height:100%;";
        scrub.appendChild(c2);
        return c2.getContext("2d");
    })();

    const mkBtn = (label, title, onClick, opts = {}) => {
        const b = document.createElement("button");
        b.textContent = label; b.title = title;
        b.style.cssText = `
            min-width:26px;height:24px;padding:0 9px;
            border:1px solid ${COLOR.border};border-radius:4px;
            background:${opts.bg || "#313244"};color:${opts.fg || COLOR.text};
            font-size:11px;font-weight:500;cursor:pointer;
            display:inline-flex;align-items:center;justify-content:center;
            white-space:nowrap;line-height:1;flex:0 0 auto;
            transition:background .12s,border-color .12s;
        `;
        b.onmouseenter = () => { b.style.background = opts.hover || "#45475a"; };
        b.onmouseleave = () => { b.style.background = opts.bg || "#313244"; };
        b.onmousedown = (e) => e.stopPropagation();
        b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); onClick(b); render(); };
        return b;
    };

    const counter = document.createElement("span");
    counter.style.cssText = `
        color:${COLOR.text};font-size:10px;flex:0 0 auto;
        font-family:ui-monospace,Menlo,monospace;
        padding:3px 8px;background:#11111b;border:1px solid ${COLOR.border};
        border-radius:4px;letter-spacing:.3px;min-width:130px;
    `;
    tb.appendChild(counter);

    const btnPin = mkBtn("📌 Set Keyframe", "Pin the current control points to this frame (K)", () => {
        const pts = state.pointsAtFrame(state.curFrame);
        if (!pts || pts.length < 2) {
            flashStatus("Add at least 2 control points before pinning a keyframe");
            return;
        }
        state.pushUndo();
        state.pinCurrent(pts);
        state.save();
    }, { bg: "#2d4a3e", hover: "#3a5f50", fg: COLOR.accent });
    tb.appendChild(btnPin);

    const btnUnpin = mkBtn("✖ Clear Keyframe", "Remove the keyframe at this frame", () => {
        if (!state.isPinned(state.curFrame)) {
            flashStatus("This frame is not a keyframe");
            return;
        }
        state.pushUndo();
        state.unpinCurrent();
        state.save();
    }, { bg: "#4a2d2d", hover: "#5f3a3a", fg: COLOR.danger });
    tb.appendChild(btnUnpin);

    tb.appendChild(mkBtn("↶", "Undo (Ctrl+Z)", () => { if (state.undo()) state.save(); }));
    tb.appendChild(mkBtn("↷", "Redo (Ctrl+Y)", () => { if (state.redo()) state.save(); }));

    const btnOnion = mkBtn("🧅 Onion", "Toggle onion-skin overlay", () => {
        state.onionEnabled = !state.onionEnabled;
        btnOnion.style.background = state.onionEnabled ? "#3a5f50" : "#313244";
    });
    btnOnion.style.background = "#3a5f50";
    tb.appendChild(btnOnion);

    tb.appendChild(mkBtn("\u2B1B Fit", "Fit image to view (F)", () => {
        const r = canvasWrap.getBoundingClientRect();
        state.fitView(r.width, r.height - SCRUB_HEIGHT);
    }));

    tb.appendChild(mkBtn("\u21BB Reload frames", "Re-scan upstream node for frames (after Queue)",
        () => { loadFrames(); }));

    const btnClearAll = mkBtn("🗑 Clear All", "Remove all keyframes", () => {
        if (!state.keyframes.size) return;
        if (!confirm(`Delete all ${state.keyframes.size} keyframes?`)) return;
        state.pushUndo();
        state.keyframes.clear();
        state.save();
    }, { fg: COLOR.danger });
    tb.appendChild(btnClearAll);

    let widgetH = 540;
    node.addDOMWidget("tracker_editor", "canvas", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => widgetH,
        getHeight: () => widgetH,
    });

    if (!node.size || node.size[0] < 620) {
        const h = node.size?.[1] || 720;
        node.setSize?.([620, Math.max(h, 720)]);
    }

    // ── Frame loading ───────────────────────────────────────────────
    function loadFrames() {
        const urls = findUpstreamFrames(node);
        if (!urls.length) {
            flashStatus("No frames found upstream. Queue Prompt once, then click ↻ Reload frames.");
            state.frames = [];
            state.frameUrls = [];
            return;
        }
        state.frameUrls = urls;
        state.frames = urls.map(u => ({ url: u, img: null }));
        // Load each (cached).
        urls.forEach((u, i) => {
            const cached = _cacheGet(u);
            if (cached?.img?.complete) {
                state.frames[i].img = cached.img;
                if (i === 0) setCanvasDims(cached.w, cached.h);
                render();
                return;
            }
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                state.frames[i].img = img;
                _cachePut(u, { img, w: img.naturalWidth, h: img.naturalHeight });
                if (i === 0) setCanvasDims(img.naturalWidth, img.naturalHeight);
                render();
            };
            img.onerror = () => { console.warn("[SplineTracker] frame load failed:", u); };
            img.src = u;
        });
        // Clamp current frame in case batch shrank.
        state.curFrame = Math.min(state.curFrame, urls.length - 1);
        flashStatus(`Loaded ${urls.length} frame${urls.length === 1 ? "" : "s"}`);
        render();
    }

    function setCanvasDims(w, h) {
        if (!w || !h || (state.canvasW === w && state.canvasH === h)) return;
        state.canvasW = w; state.canvasH = h;
        state._fitted = false;
    }

    let statusTimer = null;
    function flashStatus(msg) {
        status.textContent = msg;
        if (statusTimer) clearTimeout(statusTimer);
        statusTimer = setTimeout(() => { statusTimer = null; }, 1500);
    }

    // ── Render ──────────────────────────────────────────────────────
    function render() {
        const r = canvasWrap.getBoundingClientRect();
        const vw = Math.max(1, r.width);
        const vh = Math.max(1, r.height - SCRUB_HEIGHT);
        if (canvas.width !== vw || canvas.height !== vh) {
            canvas.width = vw; canvas.height = vh;
        }
        if (!state._fitted) state.fitView(vw, vh);
        draw(state, ctx, vw, vh);

        // Status text.
        const pin = state.isPinned(state.curFrame) ? "📌 KEY" : "interp";
        const pts = state.pointsAtFrame(state.curFrame);
        const nPts = pts ? pts.length : 0;
        const total = state.frames.length;
        status.textContent = `frame ${state.curFrame}/${Math.max(0, total - 1)} · ${pin} · ${nPts} pts · ${state.keyframes.size} keyframes`;

        // Counter pill.
        counter.textContent = `Frame ${state.curFrame + 1} / ${Math.max(1, total)}`;

        // Scrubber.
        renderScrubber();
    }

    function renderScrubber() {
        const sr = scrub.getBoundingClientRect();
        const sc = scrubCtx.canvas;
        const w = Math.max(1, sr.width);
        const h = Math.max(1, sr.height);
        if (sc.width !== w || sc.height !== h) { sc.width = w; sc.height = h; }
        scrubCtx.fillStyle = "#11111b";
        scrubCtx.fillRect(0, 0, w, h);
        const n = state.frames.length || 1;
        // Track.
        scrubCtx.fillStyle = "#313244";
        scrubCtx.fillRect(8, h / 2 - 2, w - 16, 4);
        // Frame ticks.
        if (n <= 100) {
            scrubCtx.fillStyle = "#45475a";
            for (let i = 0; i < n; i++) {
                const x = 8 + (w - 16) * (i / Math.max(1, n - 1));
                scrubCtx.fillRect(x - 0.5, h / 2 - 4, 1, 8);
            }
        }
        // Pinned keyframe markers.
        for (const f of state.keyframes.keys()) {
            const x = 8 + (w - 16) * (f / Math.max(1, n - 1));
            scrubCtx.fillStyle = COLOR.accent;
            scrubCtx.beginPath();
            scrubCtx.moveTo(x, 2);
            scrubCtx.lineTo(x - 4, 10);
            scrubCtx.lineTo(x + 4, 10);
            scrubCtx.closePath();
            scrubCtx.fill();
        }
        // Current frame indicator.
        const cx = 8 + (w - 16) * (state.curFrame / Math.max(1, n - 1));
        scrubCtx.strokeStyle = "#fab387";
        scrubCtx.lineWidth = 2;
        scrubCtx.beginPath();
        scrubCtx.moveTo(cx, 4);
        scrubCtx.lineTo(cx, h - 4);
        scrubCtx.stroke();
        scrubCtx.fillStyle = "#fab387";
        scrubCtx.beginPath();
        scrubCtx.arc(cx, h / 2, 5, 0, Math.PI * 2);
        scrubCtx.fill();
    }

    function scrubXToFrame(x) {
        const sr = scrub.getBoundingClientRect();
        const t = (x - sr.left - 8) / Math.max(1, sr.width - 16);
        const n = Math.max(1, state.frames.length);
        return Math.max(0, Math.min(n - 1, Math.round(t * (n - 1))));
    }

    // ── Interaction ─────────────────────────────────────────────────
    let scrubbing = false;
    scrub.addEventListener("mousedown", (e) => {
        e.preventDefault(); e.stopPropagation();
        scrubbing = true;
        state.curFrame = scrubXToFrame(e.clientX);
        render();
    });
    window.addEventListener("mousemove", (e) => {
        if (!scrubbing) return;
        state.curFrame = scrubXToFrame(e.clientX);
        render();
    });
    window.addEventListener("mouseup", () => {
        if (scrubbing) { scrubbing = false; render(); }
    });

    // Canvas pointer.
    function canvasPos(e) {
        const r = canvas.getBoundingClientRect();
        return state.viewToCanvas(e.clientX - r.left, e.clientY - r.top);
    }

    canvas.addEventListener("mousedown", (e) => {
        e.preventDefault();
        canvas.focus();
        const { x, y } = canvasPos(e);
        if (e.button === 2) {
            // Right click: delete point under cursor.
            const hit = state.findPoint(x, y);
            if (hit.point >= 0) {
                state.pushUndo();
                const pts = state.keyframes.get(state.curFrame);
                pts.splice(hit.point, 1);
                if (!pts.length) state.keyframes.delete(state.curFrame);
                state.save();
                render();
            }
            return;
        }
        if (e.button !== 0) return;
        const hit = state.findPoint(x, y);
        if (hit.point >= 0) {
            if (e.shiftKey) {
                // Shift-click deletes
                state.pushUndo();
                const pts = state.keyframes.get(state.curFrame);
                pts.splice(hit.point, 1);
                if (!pts.length) state.keyframes.delete(state.curFrame);
                state.save();
                render();
                return;
            }
            // Drag existing point.
            state.pushUndo();
            state.drag = { point: hit.point, ox: x, oy: y };
            return;
        }
        // Add a new control point.
        // If current frame is not pinned, we promote it: first
        // materialise the (interpolated) points as a brand-new keyframe,
        // then append the new point.
        const current = state.pointsAtFrame(state.curFrame);
        const newPts = current ? current.map(p => ({ x: p.x, y: p.y })) : [];
        newPts.push({ x, y });
        state.pushUndo();
        state.pinCurrent(newPts);
        state.save();
        render();
    });

    canvas.addEventListener("mousemove", (e) => {
        const { x, y } = canvasPos(e);
        state.cursor = { x, y, visible: true };
        if (state.drag) {
            const pts = state.keyframes.get(state.curFrame);
            if (pts) {
                pts[state.drag.point] = { x, y };
                render();
            }
            return;
        }
        const hit = state.findPoint(x, y);
        if (hit.point !== state.hover.point) {
            state.hover = hit;
            canvas.style.cursor = hit.point >= 0 ? "grab" : "crosshair";
            render();
        }
    });

    canvas.addEventListener("mouseup", () => {
        if (state.drag) {
            state.save();
            state.drag = null;
            render();
        }
    });
    canvas.addEventListener("mouseleave", () => {
        state.cursor.visible = false;
        if (state.drag) { state.save(); state.drag = null; }
    });
    canvas.addEventListener("contextmenu", (e) => { e.preventDefault(); });
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const r = canvas.getBoundingClientRect();
        const ax = e.clientX - r.left, ay = e.clientY - r.top;
        const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
        state.setZoomAround(state.zoom * factor, ax, ay, r.width, r.height);
        render();
    }, { passive: false });

    canvas.addEventListener("keydown", (e) => {
        const n = state.frames.length;
        if (e.key === "ArrowLeft") {
            state.curFrame = Math.max(0, state.curFrame - 1);
            e.preventDefault(); render();
        } else if (e.key === "ArrowRight") {
            state.curFrame = Math.min(Math.max(0, n - 1), state.curFrame + 1);
            e.preventDefault(); render();
        } else if (e.key === "k" || e.key === "K") {
            btnPin.click(); e.preventDefault();
        } else if (e.key === "f" || e.key === "F") {
            const r = canvasWrap.getBoundingClientRect();
            state.fitView(r.width, r.height - SCRUB_HEIGHT);
            e.preventDefault(); render();
        } else if (e.key === "z" && (e.ctrlKey || e.metaKey)) {
            if (state.undo()) state.save();
            e.preventDefault(); render();
        } else if ((e.key === "y" && (e.ctrlKey || e.metaKey)) ||
                   (e.key === "Z" && e.ctrlKey && e.shiftKey)) {
            if (state.redo()) state.save();
            e.preventDefault(); render();
        }
    });

    // Refresh frames on every executed event (so user's Queue Prompt
    // automatically populates the editor without clicking ↻).
    const onExecuted = ({ detail }) => {
        // Detail isn't strictly needed — any execution might've updated
        // the upstream node's imgs[]. Re-scan.
        loadFrames();
    };
    app.api.addEventListener("executed", onExecuted);
    node.onRemoved = (orig => function () {
        try { app.api.removeEventListener("executed", onExecuted); } catch (_) {}
        orig?.call(this);
    })(node.onRemoved);

    // Sync widget value changes (when user edits widgets from the panel
    // we need to redraw).
    const widgetSync = () => { syncFromWidgets(); render(); };
    for (const wn of ["closed", "samples_per_segment"]) {
        const w = node.widgets?.find(x => x.name === wn);
        if (!w) continue;
        const orig = w.callback;
        w.callback = function (v) { orig?.call(this, v); widgetSync(); };
    }

    // Initial paint & try to find frames now (in case graph already ran).
    setTimeout(() => {
        loadFrames();
        render();
    }, 50);

    // Periodic render so canvas resize keeps up.
    const ro = new ResizeObserver(() => render());
    ro.observe(canvasWrap);

    state.render = render;
    state.loadFrames = loadFrames;
}


// ─── Extension registration ─────────────────────────────────────────
app.registerExtension({
    name: "MEC.SplineMaskTracker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            // Defer until widgets are populated.
            setTimeout(() => installEditor(this), 0);
        };
    },
});
