/**
 * VideoMaskEditorMEC - video mask editor (rewritten 2026-10-23, owner report; ledger L9.35 / L9.36).
 *
 * What changed against the old editor, and why:
 *  - Frames are drawn at the clip's own resolution (it used to draw 384 px JPEG thumbnails stretched over the
 *    frame, the blurry plate the owner saw).
 *  - Every frame shows the mask the node will output there: painted, interpolated between painted frames, held,
 *    the input mask, or empty (the backend's /effective route). Unpainted frames used to look blank while the node
 *    held the first painted mask over the whole clip ("the mask sticks throughout").
 *  - Painting on an unpainted frame starts from that effective mask and makes the frame a keyframe.
 *  - The working mask is one alpha byte per pixel, drawn through a canvas that is updated only where a stroke
 *    touched it. The old editor built a new full-frame canvas on every mouse move (79 ms per move at 4K).
 *  - Painted frames are saved as you go (the node keeps them on disk), Revert undoes this session's changes.
 *  - Interpolation between and around painted frames is chosen in the editor and written to the node.
 */

import { app } from "../../scripts/app.js";
import { findUpstreamFrameSource } from "./_frame_finder.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { c2cAlert, c2cConfirm } from "./_c2c_dialog.js";
import { setHidden } from "./_widget_visibility.js";
// NOTE: do NOT import { C } from './_c2c_theme.js': this module defines its own `C` (a duplicate declaration once
// made the browser drop the whole extension).

const NODE_NAME = "VideoMaskEditorMEC";

// ─── Theme ──────────────────────────────────────────────────────────
// Canvas2D ignores var(...) colours (it keeps the previous colour, usually black), so every value here resolves to a
// literal colour at paint time.
const _C_FALLBACK = {
    bg: "#14141f", bg2: "#1b1b29", bg3: "#10101a", panel: "#222233", raised: "#2a2a3d",
    border: "#36364d", text: "#dcdff0", sub: "#9396ad", faint: "#6c6f85",
    accent: "#a6e3a1", accent2: "#89b4fa", warn: "#f9e2af", danger: "#f38ba8", white: "#ffffff",
};
const _C_TOKEN = {
    bg: "--c2c-bg", bg2: "--c2c-bg2", bg3: "--c2c-bg3", panel: "--c2c-surface0", raised: "--c2c-surface1",
    border: "--c2c-surface2", text: "--c2c-fg", sub: "--c2c-sub", accent: "--c2c-green", accent2: "--c2c-blue",
    warn: "--c2c-yellow", danger: "--c2c-red",
};
const C = new Proxy(_C_FALLBACK, {
    get(target, key) {
        const tok = _C_TOKEN[key];
        if (tok) {
            try {
                const v = getComputedStyle(document.documentElement).getPropertyValue(tok).trim();
                if (v && !v.startsWith("var(")) return v;
            } catch (err) {
                __c2cReport?.("video_mask_editor.palette", err, { token: tok, severity: "info" });
            }
        }
        return target[key];
    },
});

const OVERLAY_COLOURS = [
    ["green", "#7ee787"], ["red", "#ff5c7a"], ["blue", "#5aa9ff"], ["yellow", "#ffd75e"], ["white", "#ffffff"],
];
const TWEENS = [
    ["distance_transform", "Morph the shape"], ["linear", "Blend"], ["hold", "Hold the previous"],
    ["none", "None - frame by frame"],
];
const OUTSIDES = [["empty", "Empty"], ["hold", "Hold the nearest"]];
const VIEW_MODES = [["overlay", "Overlay"], ["mask", "Mask"], ["matte", "Matte"]];
const HISTORY_BYTES = 192 * 1024 * 1024;
const EFFECTIVE_MAX = 2048;              // long side of the effective-mask previews of unpainted frames
const SAVE_DELAY_MS = 500;

function uuid() {
    if (crypto?.randomUUID) return crypto.randomUUID();
    return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c =>
        (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & 15) >> (c / 4)).toString(16));
}

function _setting(id, fallback, min, max) {
    try {
        const v = Number(app.ui?.settings?.getSettingValue?.(id, fallback));
        if (!Number.isFinite(v)) return fallback;
        return Math.max(min, Math.min(max, Math.round(v)));
    } catch { return fallback; }
}

function _hexRgb(hex) {
    const h = String(hex || "#ffffff").replace("#", "");
    const f = h.length === 3 ? h.split("").map(c => c + c).join("") : h;
    return { r: parseInt(f.slice(0, 2), 16) || 0, g: parseInt(f.slice(2, 4), 16) || 0, b: parseInt(f.slice(4, 6), 16) || 0 };
}

function _el(tag, css = "", props = {}) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    for (const [k, v] of Object.entries(props)) {
        if (k === "text") e.textContent = v;
        else if (k === "html") e.innerHTML = v;
        else e[k] = v;
    }
    return e;
}

// 18 px line icons (stroke = currentColor)
const ICON = {
    brush: '<path d="M4 20c2.5 0 4-1.5 4-3.5S6.5 13 5 14.5 4 18 4 20z"/><path d="M9.5 15.5 20 5l-1-1L8.5 14.5"/>',
    erase: '<path d="M7 20h12"/><path d="m5 15 9-9 5 5-8 8H8z"/><path d="m10 10 5 5"/>',
    fill: '<path d="M5 12 12 5l7 7-7 7z"/><path d="M19 16c0 1.2.8 2 1.5 2S22 17.2 22 16s-1.5-3-1.5-3S19 14.8 19 16z"/>',
    lasso: '<path d="M7 17c-2.8-1-4-3-4-5 0-4 4-7 9-7s9 3 9 7-4 7-9 7c-1.2 0-2.3-.2-3.3-.5"/><path d="M7 17c0 2-1 3-2.5 3.5"/>',
    rect: '<rect x="4" y="6" width="16" height="12" rx="1"/>',
    ellipse: '<ellipse cx="12" cy="12" rx="8.5" ry="6.5"/>',
    polygon: '<path d="M12 3.5 20.5 10l-3.2 10H6.7L3.5 10z"/>',
    star: '<path d="m12 3.5 2.6 5.6 6 .7-4.5 4.1 1.2 6-5.3-3-5.3 3 1.2-6L3.4 9.8l6-.7z"/>',
    line: '<path d="M5 19 19 5"/>',
    pen: '<path d="M4 20 8 8l8-4 4 4-4 8z"/><circle cx="11" cy="13" r="1.6"/><path d="M4 20l5.8-5.8"/>',
    undo: '<path d="M9 7 4 12l5 5"/><path d="M4 12h10a6 6 0 0 1 0 12h-1"/>',
    redo: '<path d="m15 7 5 5-5 5"/><path d="M20 12H10a6 6 0 0 0 0 12h1"/>',
    add: '<path d="M12 5v14M5 12h14"/>',
    sub: '<path d="M5 12h14"/>',
    play: '<path d="M8 5v14l11-7z"/>',
    pause: '<path d="M8 5v14M16 5v14"/>',
    prev: '<path d="m15 6-6 6 6 6"/>',
    next: '<path d="m9 6 6 6-6 6"/>',
    first: '<path d="M7 6v12"/><path d="m17 6-6 6 6 6"/>',
    last: '<path d="M17 6v12"/><path d="m7 6 6 6-6 6"/>',
    prevKey: '<path d="m14 6-6 6 6 6"/><path d="M19 9l-3 3 3 3"/>',
    nextKey: '<path d="m10 6 6 6-6 6"/><path d="M5 9l3 3-3 3"/>',
    onion: '<circle cx="9" cy="12" r="5"/><circle cx="15" cy="12" r="5"/>',
    fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    key: '<path d="m12 3 9 9-9 9-9-9z"/>',
    trash: '<path d="M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13"/>',
    invert: '<circle cx="12" cy="12" r="8"/><path d="M12 4v16a8 8 0 0 0 0-16z" fill="currentColor" stroke="none"/>',
    square: '<rect x="4" y="4" width="16" height="16" rx="1" fill="currentColor" stroke="none" opacity=".55"/>',
    clear: '<rect x="4" y="4" width="16" height="16" rx="1"/><path d="m8 8 8 8M16 8l-8 8"/>',
    copy: '<rect x="8" y="8" width="12" height="12" rx="1"/><path d="M4 16V4h12"/>',
};
function _icon(name, size = 18) {
    return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" ` +
        `stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[name] || ""}</svg>`;
}

function _loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.decoding = "async";
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
    });
}

/** Bounded cache of decoded frames; evicts the frames farthest from the current one. */
class FrameCache {
    constructor(source, limit) {
        this.source = source;
        this.limit = Math.max(2, limit | 0);
        this.cache = new Map();
        this.pending = new Map();
        this.cur = 0;
    }
    _evict() {
        while (this.cache.size > this.limit) {
            let worst = null, dist = -1;
            for (const k of this.cache.keys()) {
                const d = Math.abs(k - this.cur);
                if (d > dist) { dist = d; worst = k; }
            }
            if (worst == null) break;
            this.cache.delete(worst);
        }
    }
    ensure(i) {
        i |= 0;
        this.cur = i;
        if (this.cache.has(i)) return Promise.resolve(this.cache.get(i));
        if (this.pending.has(i)) return this.pending.get(i);
        const p = _loadImage(this.source.url(i)).then(img => {
            this.cache.set(i, img);
            this.pending.delete(i);
            this._evict();
            return img;
        }).catch(err => { this.pending.delete(i); throw err; });
        this.pending.set(i, p);
        return p;
    }
    get(i) { return this.cache.get(i | 0) || null; }
    prefetch(i, span = 2) {
        for (let d = 1; d <= span; d++) {
            for (const j of [i + d, i - d]) {
                if (j < 0 || j >= this.source.count || this.cache.has(j) || this.pending.has(j)) continue;
                this.ensure(j).catch(() => {});
            }
        }
        this.cur = i;
    }
}

// ─── Editor model ───────────────────────────────────────────────────
class VideoMaskEditor {
    constructor(node) {
        this.node = node;
        this.sessionId = VideoMaskEditor.sessionOf(node);
        this.source = null;
        this.frames = null;
        this.count = 0;
        this.W = 0;
        this.H = 0;
        this.cur = 0;
        this.keyframes = new Map();      // frame -> PNG Blob (8-bit grey) as saved on the server
        this.original = new Map();       // the keyframes when the editor opened, for Revert
        this.touched = new Set();        // frames changed in this session
        this.thumbs = new Map();         // frame -> small tinted canvas (onion skin, keyframe list)
        this.alpha = null;               // Uint8Array W*H: the current frame's working mask, null = not editable yet
        this.dirty = false;              // alpha differs from the saved keyframe
        this.effective = null;           // {frame, canvas, source} the node's mask on an unpainted frame
        this.effCache = new Map();
        this.rev = 0;                    // bumps on every keyframe change (invalidates effective previews)
        this.tool = "brush";
        this.size = 40;
        this.hardness = 0.7;
        this.opacity = 1.0;
        this.subtract = false;
        this.tolerance = 24;
        this.sides = 5;
        this.starInner = 0.5;
        this.rotation = 0;
        this.viewMode = "overlay";
        this.colour = OVERLAY_COLOURS[0][1];
        this.maskOpacity = 0.55;
        this.onion = false;
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.undo = new Map();           // frame -> [{x, y, w, h, before, after}]
        this.redo = new Map();
        this.undoBytes = 0;
        this.saving = 0;
        this.saveState = "saved";
        this.saveTimer = null;
        this.lasso = [];
        this.pen = [];
        this.penDrag = -1;
        this.penHover = null;
        this.penSmooth = true;
        this.shapeStart = null;
        this.shapeCur = null;
        this.cursor = null;
        this.playing = false;
        this.ui = {};
        this._raf = 0;
        this._nav = 0;
        this.target = null;
        this._inflight = Promise.resolve(true);
        this._effPending = -1;
    }

    static sessionOf(node) {
        const w = node.widgets?.find(x => x.name === "session_id");
        if (!w) return "";
        if (!w.value || typeof w.value !== "string" || w.value.length < 8) w.value = uuid();
        return w.value;
    }

    widget(name) { return this.node.widgets?.find(w => w.name === name); }
    get tween() { return this.widget("tween_mode")?.value || "distance_transform"; }
    get outside() { return this.widget("outside_keys")?.value || "empty"; }
    setWidget(name, value) {
        const w = this.widget(name);
        if (!w) return;
        w.value = value;
        try { w.callback?.(value); } catch (e) { __c2cReport("video_mask_editor.widget", e); }
        this.node.setDirtyCanvas?.(true, true);
    }

    // ── loading ─────────────────────────────────────────────────────
    async load() {
        const maxPx = _setting("c2c.vme.previewMaxPx", 4096, 0, 16384);
        const src = await findUpstreamFrameSource(this.node, { planMaxPx: maxPx });
        if (!src.count) {
            c2cAlert("Video Mask Editor\n\nNo frames found for the image input. Run the workflow once (Queue) so the " +
                     "video upstream has frames to show, then open the editor again.");
            return false;
        }
        this.source = src;
        this.count = src.count;
        this.frames = new FrameCache(src, _setting("c2c.vme.cacheFrames", 48, 4, 2000));
        const im0 = await this.frames.ensure(0);
        this.W = src.width > 0 ? src.width : im0.naturalWidth;
        this.H = src.height > 0 ? src.height : im0.naturalHeight;
        this.maskCanvas = _el("canvas");
        this.maskCanvas.width = this.W;
        this.maskCanvas.height = this.H;
        this.maskCtx = this.maskCanvas.getContext("2d");
        this.maskImg = this.maskCtx.createImageData(this.W, this.H);
        this._tintImage();
        await fetch(`/mec/video_mask_editor/init?session=${encodeURIComponent(this.sessionId)}`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ h: this.H, w: this.W }),
        });
        const st = await (await fetch(`/mec/video_mask_editor/state?session=${encodeURIComponent(this.sessionId)}`)).json();
        for (const f of st.keyframes || []) {
            const r = await fetch(`/mec/video_mask_editor/keyframe?session=${encodeURIComponent(this.sessionId)}&frame=${f}`);
            if (!r.ok) continue;
            const blob = await r.blob();
            this.keyframes.set(f, blob);
            this.original.set(f, blob);
        }
        return true;
    }

    // ── frames ──────────────────────────────────────────────────────
    async goto(i) {
        if (!this.frames) return;                               // still loading
        i = Math.max(0, Math.min(this.count - 1, i | 0));
        if (i === this.cur && (this.alpha || this.effective?.frame === i)) return;
        // Every step gets a token: a slow decode of a frame the user has already left must not land on the frame
        // shown now (measured: stepping 0 -> 5 past keyframe 2 put frame 2's mask on frame 5).
        const nav = ++this._nav;
        this.target = i;
        await this.commit();
        if (nav !== this._nav) return;
        this.cur = i;
        this.alpha = null;
        this.dirty = false;
        this.effective = null;
        this.lasso = []; this.pen = []; this.penHover = null;
        const blob = this.keyframes.get(i);
        if (blob) await this._alphaFromBlob(blob, i, nav);
        else this._fetchEffective(i);
        if (nav !== this._nav) return;
        this.target = null;
        this.frames.ensure(i).then(() => this.requestDraw()).catch(() => this.requestDraw());
        this.frames.prefetch(i, this.playing ? 6 : 2);
        this.requestDraw();
        this.refreshPanels();
    }

    /** Step relative to where the user is GOING: presses during a save or decode must add up, not repeat. */
    step(d) { return this.goto((this.target ?? this.cur) + d); }
    jumpKey(dir) {
        const at = this.target ?? this.cur;
        const k = [...this.keyframes.keys()].sort((p, q) => p - q);
        const t = dir < 0 ? k.filter(f => f < at).pop() : k.find(f => f > at);
        if (t != null) this.goto(t);
    }

    async _alphaFromBlob(blob, frame = this.cur, nav = this._nav) {
        const bm = await createImageBitmap(blob);
        const c = _el("canvas");
        c.width = this.W; c.height = this.H;
        const cx = c.getContext("2d", { willReadFrequently: true });
        cx.imageSmoothingEnabled = true;
        cx.drawImage(bm, 0, 0, this.W, this.H);
        bm.close?.();
        const d = cx.getImageData(0, 0, this.W, this.H).data;
        const a = new Uint8Array(this.W * this.H);
        for (let p = 0, q = 0; q < a.length; p += 4, q++) a[q] = d[p];
        if (frame !== this.cur || nav !== this._nav) return false;        // the user moved on meanwhile
        this.alpha = a;
        this._flush(0, 0, this.W, this.H);
        return true;
    }

    _effKey(i) { return `${i}|${this.rev}|${this.tween}|${this.outside}`; }

    async _fetchEffective(i) {
        const key = this._effKey(i);
        const hit = this.effCache.get(key);
        if (hit) { if (this.cur === i && !this.alpha) { this.effective = hit; this.requestDraw(); this.refreshPanels(); } return; }
        this._effPending = i;
        const qs = new URLSearchParams({ session: this.sessionId, frame: String(i), n: String(this.count),
            tween: this.tween, outside: this.outside, max: String(EFFECTIVE_MAX) });
        try {
            const r = await fetch(`/mec/video_mask_editor/effective?${qs}`);
            const sourceTag = r.headers.get("X-C2C-Source") || (r.status === 204 ? "empty" : "");
            let canvas = null;
            if (r.status === 200) {
                const bm = await createImageBitmap(await r.blob());
                canvas = this._tintBitmap(bm, bm.width, bm.height);
                bm.close?.();
            }
            const entry = { frame: i, canvas, source: sourceTag || "empty" };
            if (this.effCache.size > 64) this.effCache.clear();
            this.effCache.set(key, entry);
            // an older answer (keyframes or modes changed since it was asked) never replaces a newer one
            if (this.cur === i && !this.alpha && key === this._effKey(i)) { this.effective = entry; this.requestDraw(); this.refreshPanels(); }
        } catch (e) { __c2cReport("video_mask_editor.effective", e); }
        if (this._effPending === i) this._effPending = -1;
    }

    /** False while this frame's starting mask is still on its way (painting now would start from empty). */
    get ready() {
        return !!this.alpha || this.effective?.frame === this.cur || this._effPending !== this.cur;
    }

    /** Grey mask bitmap -> canvas tinted with the overlay colour, alpha = grey. */
    _tintBitmap(bm, w, h) {
        const c = _el("canvas");
        c.width = w; c.height = h;
        const cx = c.getContext("2d", { willReadFrequently: true });
        cx.drawImage(bm, 0, 0, w, h);
        const id = cx.getImageData(0, 0, w, h);
        const { r, g, b } = _hexRgb(this.colour);
        const d = id.data;
        for (let p = 0; p < d.length; p += 4) { const a = d[p]; d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = a; }
        cx.putImageData(id, 0, 0);
        return c;
    }

    /** Fill maskImg's colour channels with the overlay colour once; strokes then touch alpha only. */
    _tintImage() {
        const { r, g, b } = _hexRgb(this.colour);
        const d = this.maskImg.data;
        for (let p = 0; p < d.length; p += 4) { d[p] = r; d[p + 1] = g; d[p + 2] = b; }
    }

    setColour(hex) {
        this.colour = hex;
        this._tintImage();
        if (this.alpha) this._flush(0, 0, this.W, this.H);
        this.effCache.clear();
        this.thumbs.clear();
        if (!this.alpha) this._fetchEffective(this.cur);
        this.requestDraw();
        this.refreshPanels();
    }

    /** Copy alpha into the overlay canvas, only inside the rectangle that changed. */
    _flush(x0, y0, x1, y1) {
        x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
        x1 = Math.min(this.W, Math.ceil(x1)); y1 = Math.min(this.H, Math.ceil(y1));
        if (x1 <= x0 || y1 <= y0 || !this.alpha) return;
        const d = this.maskImg.data, a = this.alpha, W = this.W;
        for (let y = y0; y < y1; y++) {
            let q = y * W + x0, p = q * 4 + 3;
            for (let x = x0; x < x1; x++, q++, p += 4) d[p] = a[q];
        }
        this.maskCtx.putImageData(this.maskImg, 0, 0, x0, y0, x1 - x0, y1 - y0);
    }

    /** Make the current frame editable: start from the mask the node outputs there (interpolated, held, ...). */
    ensureEditable() {
        if (this.alpha) return;
        const a = new Uint8Array(this.W * this.H);
        const eff = this.effective?.frame === this.cur ? this.effective.canvas : null;
        if (eff) {
            const c = _el("canvas");
            c.width = this.W; c.height = this.H;
            const cx = c.getContext("2d", { willReadFrequently: true });
            cx.drawImage(eff, 0, 0, this.W, this.H);
            const d = cx.getImageData(0, 0, this.W, this.H).data;
            for (let p = 3, q = 0; q < a.length; p += 4, q++) a[q] = d[p];
        }
        this.alpha = a;
        this._flush(0, 0, this.W, this.H);
    }

    // ── undo (rectangle patches, bounded by bytes) ───────────────────
    beginEdit() {
        this.ensureEditable();
        if (!this._before || this._before.length !== this.alpha.length) this._before = new Uint8Array(this.alpha.length);
        this._before.set(this.alpha);
        this._edit = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
    }
    touch(x0, y0, x1, y1) {
        const e = this._edit;
        if (!e) return;
        e.x0 = Math.min(e.x0, x0); e.y0 = Math.min(e.y0, y0); e.x1 = Math.max(e.x1, x1); e.y1 = Math.max(e.y1, y1);
    }
    endEdit() {
        const e = this._edit;
        this._edit = null;
        if (!e || e.x1 <= e.x0) return;
        const x = Math.max(0, Math.floor(e.x0)), y = Math.max(0, Math.floor(e.y0));
        const w = Math.min(this.W, Math.ceil(e.x1)) - x, h = Math.min(this.H, Math.ceil(e.y1)) - y;
        if (w <= 0 || h <= 0) return;
        const before = new Uint8Array(w * h), after = new Uint8Array(w * h);
        for (let r = 0; r < h; r++) {
            const off = (y + r) * this.W + x;
            before.set(this._before.subarray(off, off + w), r * w);
            after.set(this.alpha.subarray(off, off + w), r * w);
        }
        const stack = this.undo.get(this.cur) || [];
        stack.push({ x, y, w, h, before, after });
        this.undo.set(this.cur, stack);
        this.redo.set(this.cur, []);
        this.undoBytes += 2 * w * h;
        while (this.undoBytes > HISTORY_BYTES) {
            let dropped = false;
            for (const st of this.undo.values()) {
                if (st.length) { const o = st.shift(); this.undoBytes -= 2 * o.w * o.h; dropped = true; break; }
            }
            if (!dropped) break;
        }
        this.markDirty();
    }
    _applyPatch(pt, data) {
        for (let r = 0; r < pt.h; r++) this.alpha.set(data.subarray(r * pt.w, (r + 1) * pt.w), (pt.y + r) * this.W + pt.x);
        this._flush(pt.x, pt.y, pt.x + pt.w, pt.y + pt.h);
        this.markDirty();
    }
    undoOp() {
        const st = this.undo.get(this.cur);
        if (!st?.length || !this.alpha) return;
        const pt = st.pop();
        (this.redo.get(this.cur) || this.redo.set(this.cur, []).get(this.cur)).push(pt);
        this._applyPatch(pt, pt.before);
    }
    redoOp() {
        const st = this.redo.get(this.cur);
        if (!st?.length || !this.alpha) return;
        const pt = st.pop();
        (this.undo.get(this.cur) || this.undo.set(this.cur, []).get(this.cur)).push(pt);
        this._applyPatch(pt, pt.after);
    }

    // ── saving ───────────────────────────────────────────────────────
    markDirty() {
        const was = this.dirty;
        this.dirty = true;
        if (!was) { this.drawTimeline(); this.refreshPanels(); }
        this.setSaveState("pending");
        clearTimeout(this.saveTimer);
        this.saveTimer = setTimeout(() => this.commit(), SAVE_DELAY_MS);
        this.requestDraw();
    }

    setSaveState(s) {
        this.saveState = s;
        const el = this.ui.saveChip;
        if (!el) return;
        const map = {
            saved: ["All changes saved", C.accent], pending: ["Unsaved changes", C.warn],
            saving: ["Saving...", C.warn], failed: ["Save failed - retrying on the next edit", C.danger],
        };
        const [txt, col] = map[s] || map.saved;
        el.textContent = txt;
        el.style.color = col;
        el.style.borderColor = col + "66";
    }

    async _pngFromAlpha(a) {
        const c = _el("canvas");
        c.width = this.W; c.height = this.H;
        const cx = c.getContext("2d");
        const id = cx.createImageData(this.W, this.H);
        const d = id.data;
        for (let p = 0, q = 0; q < a.length; p += 4, q++) { const v = a[q]; d[p] = v; d[p + 1] = v; d[p + 2] = v; d[p + 3] = 255; }
        cx.putImageData(id, 0, 0);
        return await new Promise(res => c.toBlob(res, "image/png"));
    }

    /** Save the current frame if it changed; an all-empty painted frame stays a keyframe (an explicit "no mask"). */
    async commit() {
        clearTimeout(this.saveTimer);
        // A save already on its way finishes first: until the server has it, the masks it shows for unpainted
        // frames are still the old ones (measured: frame 4 read "No mask" right after keyframe 6 was painted).
        await this._inflight;
        if (!this.dirty || !this.alpha) return true;
        const job = this._commitNow();
        this._inflight = job.catch(() => false);
        return await job;
    }

    /** After any keyframe change: forget the effective masks and re-ask for the one on screen. */
    _keyframesChanged() {
        this.rev++;
        this.effCache.clear();
        if (!this.alpha) this._fetchEffective(this.cur);
    }

    async _commitNow() {
        const frame = this.cur;
        const snapshot = this.alpha.slice();
        this.dirty = false;
        this.saving++;
        this.setSaveState("saving");
        try {
            const blob = await this._pngFromAlpha(snapshot);
            const r = await fetch(`/mec/video_mask_editor/keyframe?session=${encodeURIComponent(this.sessionId)}&frame=${frame}`,
                { method: "POST", headers: { "Content-Type": "image/png" }, body: blob });
            if (!r.ok) throw new Error(`server answered ${r.status}`);
            this.keyframes.set(frame, blob);
            this.touched.add(frame);
            this.thumbs.delete(frame);
            this._keyframesChanged();
            this.saving--;
            if (!this.saving) this.setSaveState(this.dirty ? "pending" : "saved");
            this.refreshPanels();
            return true;
        } catch (e) {
            this.saving--;
            this.dirty = true;
            this.setSaveState("failed");
            __c2cReport("video_mask_editor.save", e);
            return false;
        }
    }

    async deleteKeyframe(f) {
        await fetch(`/mec/video_mask_editor/keyframe?session=${encodeURIComponent(this.sessionId)}&frame=${f}`, { method: "DELETE" });
        this.keyframes.delete(f);
        this.thumbs.delete(f);
        this.touched.add(f);
        if (f === this.cur) {
            this.alpha = null;
            this.dirty = false;
        }
        this._keyframesChanged();
        this.requestDraw();
        this.refreshPanels();
    }

    async makeKeyframe() {
        this.beginEdit();
        this.touch(0, 0, 0, 0);
        this._edit = null;
        this.markDirty();
        await this.commit();
    }

    /** Put every frame changed in this session back the way it was when the editor opened. */
    async revert() {
        clearTimeout(this.saveTimer);
        this.dirty = false;
        for (const f of this.touched) {
            const orig = this.original.get(f);
            const url = `/mec/video_mask_editor/keyframe?session=${encodeURIComponent(this.sessionId)}&frame=${f}`;
            if (orig) await fetch(url, { method: "POST", headers: { "Content-Type": "image/png" }, body: orig });
            else await fetch(url, { method: "DELETE" });
        }
    }

    // ── painting ─────────────────────────────────────────────────────
    stamp(x, y, prev) {
        const a = this.alpha, W = this.W, H = this.H;
        const r = Math.max(0.5, this.size / 2), r2 = r * r;
        const inner = r * Math.max(0, Math.min(1, this.hardness));
        const amt = 255 * this.opacity;
        const erase = this.tool === "erase" || (this.tool === "brush" && this.subtract);
        const dist = prev ? Math.hypot(x - prev.x, y - prev.y) : 0;
        const steps = prev ? Math.max(1, Math.ceil(dist / Math.max(1, r * 0.3))) : 0;
        let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
        for (let s = prev ? 1 : 0; s <= steps; s++) {
            const t = steps ? s / steps : 0;
            const cx = prev ? prev.x + (x - prev.x) * t : x;
            const cy = prev ? prev.y + (y - prev.y) * t : y;
            const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(W - 1, Math.ceil(cx + r));
            const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(H - 1, Math.ceil(cy + r));
            for (let yy = y0; yy <= y1; yy++) {
                const dy = yy + 0.5 - cy;
                let q = yy * W + x0;
                for (let xx = x0; xx <= x1; xx++, q++) {
                    const dx = xx + 0.5 - cx;
                    const d2 = dx * dx + dy * dy;
                    if (d2 > r2) continue;
                    const d = Math.sqrt(d2);
                    const f = d <= inner ? 1 : 1 - (d - inner) / Math.max(1e-6, r - inner);
                    const v = amt * f;
                    a[q] = erase ? Math.max(0, a[q] - v) : Math.max(a[q], Math.min(255, v));
                }
            }
            bx0 = Math.min(bx0, x0); by0 = Math.min(by0, y0); bx1 = Math.max(bx1, x1 + 1); by1 = Math.max(by1, y1 + 1);
        }
        if (bx1 > bx0) { this._flush(bx0, by0, bx1, by1); this.touch(bx0, by0, bx1, by1); }
    }

    /** Rasterise a closed outline (image coords) into alpha, only over its bounding box. */
    fillPath(pts, { line = false } = {}) {
        if (pts.length < (line ? 2 : 3)) return;
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
        const pad = line ? this.size / 2 + 2 : 2;
        x0 = Math.max(0, Math.floor(x0 - pad)); y0 = Math.max(0, Math.floor(y0 - pad));
        x1 = Math.min(this.W, Math.ceil(x1 + pad)); y1 = Math.min(this.H, Math.ceil(y1 + pad));
        const w = x1 - x0, h = y1 - y0;
        if (w <= 0 || h <= 0) return;
        const c = _el("canvas");
        c.width = w; c.height = h;
        const cx = c.getContext("2d", { willReadFrequently: true });
        cx.translate(-x0, -y0);
        cx.fillStyle = "#ffffff"; cx.strokeStyle = "#ffffff";
        cx.beginPath();
        cx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) cx.lineTo(pts[i][0], pts[i][1]);
        if (line) { cx.lineWidth = Math.max(1, this.size); cx.lineCap = "round"; cx.lineJoin = "round"; cx.stroke(); }
        else { cx.closePath(); cx.fill(); }
        const d = cx.getImageData(0, 0, w, h).data;
        const a = this.alpha, amt = this.opacity, sub = this.subtract;
        for (let yy = 0; yy < h; yy++) {
            let q = (y0 + yy) * this.W + x0, p = yy * w * 4 + 3;
            for (let xx = 0; xx < w; xx++, q++, p += 4) {
                const v = d[p] * amt;
                if (!v) continue;
                a[q] = sub ? Math.max(0, a[q] - v) : Math.max(a[q], v);
            }
        }
        this._flush(x0, y0, x1, y1);
        this.touch(x0, y0, x1, y1);
    }

    floodFill(x, y) {
        x |= 0; y |= 0;
        const W = this.W, H = this.H, a = this.alpha;
        if (x < 0 || y < 0 || x >= W || y >= H) return;
        const start = a[y * W + x], target = this.subtract ? 0 : 255, tol = this.tolerance;
        if (start === target) return;
        const seen = new Uint8Array(W * H);
        const stack = [y * W + x];
        let x0 = x, y0 = y, x1 = x, y1 = y;
        while (stack.length) {
            const q = stack.pop();
            if (seen[q]) continue;
            seen[q] = 1;
            if (Math.abs(a[q] - start) > tol) continue;
            a[q] = target;
            const qx = q % W, qy = (q / W) | 0;
            if (qx < x0) x0 = qx; if (qx > x1) x1 = qx; if (qy < y0) y0 = qy; if (qy > y1) y1 = qy;
            if (qx > 0) stack.push(q - 1);
            if (qx < W - 1) stack.push(q + 1);
            if (qy > 0) stack.push(q - W);
            if (qy < H - 1) stack.push(q + W);
        }
        this._flush(x0, y0, x1 + 1, y1 + 1);
        this.touch(x0, y0, x1 + 1, y1 + 1);
    }

    wholeFrame(op) {
        this.beginEdit();
        const a = this.alpha;
        if (op === "clear") a.fill(0);
        else if (op === "fill") a.fill(255);
        else if (op === "invert") for (let q = 0; q < a.length; q++) a[q] = 255 - a[q];
        this._flush(0, 0, this.W, this.H);
        this.touch(0, 0, this.W, this.H);
        this.endEdit();
    }

    async copyFromNearestKeyframe() {
        const keys = [...this.keyframes.keys()].filter(f => f !== this.cur).sort((p, q) => Math.abs(p - this.cur) - Math.abs(q - this.cur));
        if (!keys.length) { c2cAlert("There is no other painted frame to copy from yet."); return; }
        this.beginEdit();
        const keep = this._before.slice();
        await this._alphaFromBlob(this.keyframes.get(keys[0]));
        this._before = keep;
        this.touch(0, 0, this.W, this.H);
        this.endEdit();
    }

    shapePoints(kind, a, b, square) {
        let x0 = Math.min(a.x, b.x), y0 = Math.min(a.y, b.y), x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
        if (square) { const s = Math.min(x1 - x0, y1 - y0); x1 = x0 + s; y1 = y0 + s; }
        const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, rx = Math.max(0.5, (x1 - x0) / 2), ry = Math.max(0.5, (y1 - y0) / 2);
        const rot = this.rotation, cs = Math.cos(rot), sn = Math.sin(rot);
        const R = (px, py) => rot ? [cx + (px - cx) * cs - (py - cy) * sn, cy + (px - cx) * sn + (py - cy) * cs] : [px, py];
        if (kind === "rect") return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].map(([px, py]) => R(px, py));
        if (kind === "ellipse") return Array.from({ length: 96 }, (_, i) => R(cx + Math.cos(i / 96 * 2 * Math.PI) * rx, cy + Math.sin(i / 96 * 2 * Math.PI) * ry));
        const n = Math.max(3, this.sides | 0), star = kind === "star", cnt = star ? n * 2 : n;
        return Array.from({ length: cnt }, (_, i) => {
            const ang = -Math.PI / 2 + i / cnt * 2 * Math.PI, rr = star && i % 2 ? this.starInner : 1;
            return R(cx + Math.cos(ang) * rx * rr, cy + Math.sin(ang) * ry * rr);
        });
    }

    penCurve(closed) {
        const P = this.pen, n = P.length;
        if (n < 3 || !this.penSmooth) return P.map(p => [p.x, p.y]);
        const at = i => closed ? P[((i % n) + n) % n] : P[Math.max(0, Math.min(n - 1, i))];
        const out = [];
        for (let i = 0; i < (closed ? n : n - 1); i++) {
            const p0 = at(i - 1), p1 = at(i), p2 = at(i + 1), p3 = at(i + 2);
            for (let s = 0; s < 16; s++) {
                const t = s / 16, t2 = t * t, t3 = t2 * t;
                out.push([
                    0.5 * (2 * p1.x + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
                    0.5 * (2 * p1.y + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
                ]);
            }
        }
        return out;
    }

    commitPen() {
        if (this.pen.length >= 3) {
            this.beginEdit();
            this.fillPath(this.penCurve(true));
            this.endEdit();
        }
        this.pen = []; this.penHover = null; this.penDrag = -1;
        this.requestDraw();
    }

    // ── view ─────────────────────────────────────────────────────────
    toImage(vx, vy) { return { x: (vx - this.panX) / this.zoom, y: (vy - this.panY) / this.zoom }; }
    fit() {
        const r = this.ui.viewport.getBoundingClientRect();
        const pad = 24;
        this.zoom = Math.max(0.02, Math.min((r.width - 2 * pad) / this.W, (r.height - 2 * pad) / this.H, 8));
        this.panX = (r.width - this.W * this.zoom) / 2;
        this.panY = (r.height - this.H * this.zoom) / 2;
        this.requestDraw();
    }
    zoomAt(vx, vy, factor) {
        const z = Math.max(0.02, Math.min(32, this.zoom * factor)), k = z / this.zoom;
        this.panX = vx - (vx - this.panX) * k;
        this.panY = vy - (vy - this.panY) * k;
        this.zoom = z;
        this.requestDraw();
    }

    requestDraw() {
        if (this._raf) return;
        this._raf = requestAnimationFrame(() => { this._raf = 0; this.draw(); });
    }

    _onionCanvas(f) {
        let c = this.thumbs.get(f);
        if (c) return c;
        const blob = this.keyframes.get(f);
        if (!blob) return null;
        createImageBitmap(blob).then(bm => {
            const k = Math.min(1, 512 / Math.max(bm.width, bm.height));
            this.thumbs.set(f, this._tintBitmap(bm, Math.max(1, Math.round(bm.width * k)), Math.max(1, Math.round(bm.height * k))));
            bm.close?.();
            this.requestDraw();
            this.refreshPanels();
        }).catch(() => {});
        return null;
    }

    draw() {
        const cnv = this.ui.canvas, vp = this.ui.viewport;
        if (!cnv || !vp) return;
        const r = vp.getBoundingClientRect(), dpr = Math.max(1, window.devicePixelRatio || 1);
        const cw = Math.max(1, Math.round(r.width * dpr)), ch = Math.max(1, Math.round(r.height * dpr));
        if (cnv.width !== cw || cnv.height !== ch) { cnv.width = cw; cnv.height = ch; }
        const ctx = cnv.getContext("2d");
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, cw, ch);
        ctx.setTransform(dpr * this.zoom, 0, 0, dpr * this.zoom, dpr * this.panX, dpr * this.panY);
        ctx.imageSmoothingEnabled = this.zoom < 3;
        ctx.imageSmoothingQuality = "high";
        const frame = this.frames?.get(this.cur);
        const maskSrc = this.alpha ? this.maskCanvas : (this.effective?.frame === this.cur ? this.effective.canvas : null);
        if (this.viewMode === "mask") {
            ctx.fillStyle = "#000";
            ctx.fillRect(0, 0, this.W, this.H);
            if (maskSrc) {
                ctx.save();
                ctx.filter = "brightness(0) invert(1)";
                ctx.drawImage(maskSrc, 0, 0, this.W, this.H);
                ctx.restore();
            }
        } else {
            if (frame) ctx.drawImage(frame, 0, 0, this.W, this.H);
            else { ctx.fillStyle = C.bg2; ctx.fillRect(0, 0, this.W, this.H); }
            if (this.viewMode === "matte") {
                ctx.save();
                ctx.globalCompositeOperation = "destination-in";
                if (maskSrc) ctx.drawImage(maskSrc, 0, 0, this.W, this.H);
                else ctx.clearRect(0, 0, this.W, this.H);
                ctx.restore();
            } else {
                if (this.onion) {
                    const keys = [...this.keyframes.keys()].sort((p, q) => p - q);
                    const prev = keys.filter(f => f < this.cur).pop(), next = keys.find(f => f > this.cur);
                    for (const f of [prev, next]) {
                        if (f == null) continue;
                        const oc = this._onionCanvas(f);
                        if (!oc) continue;
                        ctx.globalAlpha = 0.22;
                        ctx.drawImage(oc, 0, 0, this.W, this.H);
                    }
                    ctx.globalAlpha = 1;
                }
                if (maskSrc) {
                    ctx.globalAlpha = this.alpha ? this.maskOpacity : this.maskOpacity * 0.75;
                    ctx.drawImage(maskSrc, 0, 0, this.W, this.H);
                    ctx.globalAlpha = 1;
                }
            }
        }
        // frame edge
        ctx.lineWidth = 1 / this.zoom;
        ctx.strokeStyle = C.border;
        ctx.strokeRect(0, 0, this.W, this.H);
        // tool previews
        const px = 1 / this.zoom;
        ctx.lineWidth = 1.5 * px;
        if (this.tool === "lasso" && this.lasso.length) {
            ctx.strokeStyle = C.accent2;
            ctx.setLineDash([6 * px, 4 * px]);
            ctx.beginPath();
            ctx.moveTo(this.lasso[0].x, this.lasso[0].y);
            for (const p of this.lasso) ctx.lineTo(p.x, p.y);
            if (this.cursorImg) ctx.lineTo(this.cursorImg.x, this.cursorImg.y);
            ctx.stroke();
            ctx.setLineDash([]);
        }
        if (this.shapeStart && this.shapeCur) {
            ctx.strokeStyle = this.subtract ? C.danger : C.accent2;
            ctx.setLineDash([6 * px, 4 * px]);
            ctx.beginPath();
            if (this.tool === "line") {
                ctx.moveTo(this.shapeStart.x, this.shapeStart.y);
                ctx.lineTo(this.shapeCur.x, this.shapeCur.y);
            } else {
                const pts = this.shapePoints(this.tool, this.shapeStart, this.shapeCur, this._shapeSquare);
                ctx.moveTo(pts[0][0], pts[0][1]);
                for (const [x, y] of pts) ctx.lineTo(x, y);
                ctx.closePath();
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }
        if (this.tool === "pen" && this.pen.length) {
            const curve = this.penCurve(this.pen.length >= 3);
            ctx.strokeStyle = C.accent2;
            ctx.beginPath();
            ctx.moveTo(curve[0][0], curve[0][1]);
            for (const [x, y] of curve) ctx.lineTo(x, y);
            if (this.pen.length >= 3) ctx.closePath();
            ctx.stroke();
            if (this.penHover && this.penDrag < 0) {
                const lp = this.pen[this.pen.length - 1];
                ctx.setLineDash([4 * px, 4 * px]);
                ctx.beginPath(); ctx.moveTo(lp.x, lp.y); ctx.lineTo(this.penHover.x, this.penHover.y); ctx.stroke();
                ctx.setLineDash([]);
            }
            for (let i = 0; i < this.pen.length; i++) {
                ctx.beginPath();
                ctx.arc(this.pen[i].x, this.pen[i].y, 4.5 * px, 0, Math.PI * 2);
                ctx.fillStyle = i === 0 ? C.accent2 : "#fff";
                ctx.fill();
                ctx.stroke();
            }
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        if ((this.tool === "brush" || this.tool === "erase") && this.cursor) {
            ctx.strokeStyle = "rgba(0,0,0,.55)";
            ctx.lineWidth = 3;
            ctx.beginPath(); ctx.arc(this.cursor.x, this.cursor.y, Math.max(1, this.size / 2 * this.zoom), 0, Math.PI * 2); ctx.stroke();
            ctx.strokeStyle = this.tool === "erase" || this.subtract ? C.danger : "#fff";
            ctx.lineWidth = 1.2;
            ctx.stroke();
        }
        this.updateStatus();
    }

    /** Frame state used by the timeline and the frame card: painted / interpolated / held / none, client side. */
    stateOf(i, sorted = null) {
        if (this.keyframes.has(i) || (i === this.cur && this.dirty)) return "painted";
        const keys = sorted || [...this.keyframes.keys()].sort((p, q) => p - q);
        if (!keys.length) return "none";
        if (keys[0] < i && keys[keys.length - 1] > i) {
            return this.tween === "none" ? "none" : (this.tween === "hold" ? "held" : "interp");
        }
        return this.outside === "hold" ? "held" : "none";
    }

    drawTimeline() {
        const c = this.ui.timeline;
        if (!c) return;
        const r = c.getBoundingClientRect(), dpr = Math.max(1, window.devicePixelRatio || 1);
        const W = Math.max(1, Math.round(r.width * dpr)), H = Math.max(1, Math.round(r.height * dpr));
        if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
        const ctx = c.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const w = r.width, h = r.height, pad = 14, n = Math.max(1, this.count);
        ctx.fillStyle = C.bg3;
        ctx.fillRect(0, 0, w, h);
        const fx = i => pad + (w - 2 * pad) * ((i + 0.5) / n);
        const cell = (w - 2 * pad) / n;
        // span bands
        const bandY = h * 0.42, bandH = h * 0.26;
        const sorted = [...this.keyframes.keys()].sort((p, q) => p - q);
        for (let i = 0; i < n; i++) {
            const s = this.stateOf(i, sorted);
            if (s === "none") continue;
            const x = pad + i * cell;
            if (s === "interp") { ctx.fillStyle = this.colour + "55"; ctx.fillRect(x, bandY, Math.max(1, cell), bandH); }
            else if (s === "held") { ctx.fillStyle = this.colour + "2e"; ctx.fillRect(x, bandY, Math.max(1, cell), bandH); }
            else if (s === "painted") { ctx.fillStyle = this.colour + "99"; ctx.fillRect(x, bandY, Math.max(1, cell), bandH); }
        }
        // ticks + labels
        const step = Math.max(1, Math.ceil(n / Math.max(1, (w - 2 * pad) / 56)));
        ctx.fillStyle = C.faint;
        ctx.font = "10px system-ui, sans-serif";
        ctx.textAlign = "center";
        for (let i = 0; i < n; i++) {
            const x = fx(i);
            const major = i % step === 0;
            ctx.fillRect(Math.round(x), 4, 1, major ? 7 : 3);
            if (major) ctx.fillText(String(i), x, h - 5);
        }
        // keyframe diamonds
        for (const f of this.keyframes.keys()) {
            const x = fx(f), y = bandY + bandH / 2;
            ctx.beginPath();
            ctx.moveTo(x, y - 6); ctx.lineTo(x + 6, y); ctx.lineTo(x, y + 6); ctx.lineTo(x - 6, y); ctx.closePath();
            ctx.fillStyle = this.colour; ctx.fill();
            ctx.strokeStyle = C.bg3; ctx.lineWidth = 1.5; ctx.stroke();
        }
        // playhead
        const px = fx(this.cur);
        ctx.fillStyle = C.warn;
        ctx.fillRect(Math.round(px) - 1, 2, 2, h - 18);
        ctx.beginPath(); ctx.moveTo(px - 5, 2); ctx.lineTo(px + 5, 2); ctx.lineTo(px, 8); ctx.closePath(); ctx.fill();
    }

    updateStatus() {
        const s = this.ui;
        if (!s.frameField) return;
        if (document.activeElement !== s.frameField) s.frameField.value = String(this.cur);
        s.frameTotal.textContent = `of ${this.count - 1}`;
        if (s.zoomLabel) s.zoomLabel.textContent = `${Math.round(this.zoom * 100)}%`;
    }

    refreshPanels() {
        if (this._panelRaf) return;
        this._panelRaf = requestAnimationFrame(() => { this._panelRaf = 0; this._refreshPanels(); });
    }

    _refreshPanels() {
        const s = this.ui;
        if (!s.stateChip) return;
        const st = this.stateOf(this.cur);
        let label, col;
        if (this.alpha) { label = this.keyframes.has(this.cur) || this.dirty ? "Painted keyframe" : "Painted"; col = this.colour; }
        else {
            const src = this.effective?.frame === this.cur ? this.effective.source : "";
            if (src.startsWith("interpolated")) label = `Interpolated (${src.replace("interpolated ", "frames ")})`;
            else if (src.startsWith("held")) label = `Held (${src.replace("held from ", "from frame ")})`;
            else if (src === "input mask") label = "Input mask";
            else if (src === "empty") label = "No mask";
            else label = st === "interp" ? "Interpolated" : st === "held" ? "Held" : "No mask";
            col = C.sub;
        }
        s.stateChip.textContent = label;
        s.stateChip.style.color = col;
        s.stateChip.style.borderColor = (col.startsWith("#") ? col : C.border) + "66";
        s.btnDelKey.disabled = !this.keyframes.has(this.cur);
        s.btnMakeKey.disabled = this.keyframes.has(this.cur);
        s.hint.textContent = this.alpha
            ? "This frame is a keyframe: its mask is used as painted."
            : "Paint here to turn this frame into a keyframe. It starts from the mask shown.";
        // keyframe list
        const list = s.kfList;
        list.replaceChildren();
        const keys = [...this.keyframes.keys()].sort((p, q) => p - q);
        if (!keys.length) {
            list.appendChild(_el("div", `color:${C.faint};font-size:12px;padding:10px 4px;line-height:1.5;`,
                { text: "No keyframes yet. Paint on any frame; it is saved as you go." }));
        }
        for (const f of keys) {
            const row = _el("div", `display:grid;grid-template-columns:52px 1fr auto;gap:8px;align-items:center;padding:5px 6px;
                border-radius:7px;cursor:pointer;border:1px solid ${f === this.cur ? this.colour + "88" : "transparent"};
                background:${f === this.cur ? this.colour + "14" : "transparent"};`);
            const th = _el("canvas", `width:52px;height:30px;border-radius:4px;background:#000;display:block;`);
            th.width = 104; th.height = 60;
            const oc = this._onionCanvas(f);
            if (oc) {
                const tc = th.getContext("2d");
                const k = Math.min(104 / oc.width, 60 / oc.height);
                tc.drawImage(oc, (104 - oc.width * k) / 2, (60 - oc.height * k) / 2, oc.width * k, oc.height * k);
            }
            const lbl = _el("div", `font-size:12px;color:${C.text};font-variant-numeric:tabular-nums;`, { text: `Frame ${f}` });
            const del = _el("button", `border:none;background:transparent;color:${C.faint};cursor:pointer;padding:4px;border-radius:5px;display:flex;`,
                { html: _icon("trash", 15), title: `Delete keyframe ${f}` });
            del.onmouseenter = () => { del.style.color = C.danger; };
            del.onmouseleave = () => { del.style.color = C.faint; };
            del.onclick = (e) => { e.stopPropagation(); this.deleteKeyframe(f); };
            row.onclick = () => this.goto(f);
            row.append(th, lbl, del);
            list.appendChild(row);
        }
        s.kfCount.textContent = String(keys.length);
        this.drawTimeline();
    }

    // ── playback ─────────────────────────────────────────────────────
    togglePlay() {
        this.playing = !this.playing;
        this.ui.btnPlay.innerHTML = _icon(this.playing ? "pause" : "play");
        if (!this.playing) return;
        const fps = _setting("c2c.vme.playFps", 12, 1, 60);
        const tick = async () => {
            if (!this.playing) return;
            const t0 = performance.now();
            await this.goto(((this.target ?? this.cur) + 1) % this.count);
            setTimeout(tick, Math.max(0, 1000 / fps - (performance.now() - t0)));
        };
        tick();
    }
}

// ─── Modal ──────────────────────────────────────────────────────────
function openEditor(node) {
    if (document.querySelector("[data-vme-modal]")) return;
    const ed = new VideoMaskEditor(node);
    const font = "font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;";
    const overlay = _el("div", `position:fixed;inset:0;z-index:var(--c2c-z-modal,10000);display:flex;flex-direction:column;
        background:${C.bg};color:${C.text};${font}font-size:13px;`);
    overlay.dataset.vmeModal = "1";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Video mask editor");

    const btnCss = (opts = {}) => `display:inline-flex;align-items:center;justify-content:center;gap:6px;height:32px;
        min-width:32px;padding:0 ${opts.icon ? 0 : 12}px;border-radius:8px;border:1px solid ${opts.border || C.border};
        background:${opts.bg || C.panel};color:${opts.fg || C.text};cursor:pointer;font-size:12.5px;font-weight:500;
        transition:background .12s,border-color .12s,color .12s;`;
    const mkBtn = (label, opts = {}) => {
        const b = _el("button", btnCss(opts), { html: label, title: opts.title || "", type: "button" });
        if (opts.id) b.dataset.vmeAction = opts.id;
        b.addEventListener("mouseenter", () => { if (!b.dataset.on) b.style.background = opts.hover || C.raised; });
        b.addEventListener("mouseleave", () => { if (!b.dataset.on) b.style.background = opts.bg || C.panel; });
        return b;
    };
    const select = (options, value, onChange, title) => {
        const s = _el("select", `height:32px;border-radius:8px;border:1px solid ${C.border};background:${C.panel};color:${C.text};
            padding:0 8px;font-size:12.5px;cursor:pointer;width:100%;${font}`, { title });
        for (const [v, t] of options) s.appendChild(_el("option", "", { value: v, text: t }));
        s.value = value;
        s.onchange = () => onChange(s.value);
        return s;
    };
    const range = (label, min, max, step, value, fmt, onInput) => {
        const wrap = _el("label", `display:inline-flex;align-items:center;gap:8px;color:${C.sub};font-size:12px;`);
        const cap = _el("span", "min-width:54px;", { text: label });
        const inp = _el("input", `width:110px;accent-color:${C.accent2};`, { type: "range", min, max, step, value });
        const out = _el("span", `min-width:42px;color:${C.text};font-variant-numeric:tabular-nums;font-size:12px;`, { text: fmt(value) });
        inp.oninput = () => { out.textContent = fmt(+inp.value); onInput(+inp.value); };
        wrap.append(cap, inp, out);
        wrap.set = (v) => { inp.value = v; out.textContent = fmt(v); };
        return wrap;
    };

    // header
    const header = _el("div", `display:flex;align-items:center;gap:14px;padding:10px 16px;border-bottom:1px solid ${C.border};
        background:${C.bg2};flex:0 0 auto;`);
    const title = _el("div", "display:flex;flex-direction:column;gap:1px;min-width:0;");
    title.append(_el("div", `font-size:15px;font-weight:600;letter-spacing:.1px;`, { text: "Video Mask Editor" }));
    const meta = _el("div", `font-size:12px;color:${C.sub};font-variant-numeric:tabular-nums;`, { text: "loading frames..." });
    title.appendChild(meta);
    const saveChip = _el("span", `margin-left:6px;font-size:11.5px;padding:4px 10px;border-radius:999px;border:1px solid ${C.border};white-space:nowrap;`);
    const spacer = _el("div", "flex:1;");
    const viewSeg = _el("div", `display:inline-flex;border:1px solid ${C.border};border-radius:9px;overflow:hidden;`);
    const viewBtns = {};
    for (const [v, t] of VIEW_MODES) {
        const b = _el("button", `height:30px;padding:0 12px;border:none;background:transparent;color:${C.sub};cursor:pointer;font-size:12.5px;${font}`,
            { text: t, type: "button", title: `View: ${t.toLowerCase()} (V cycles)` });
        b.onclick = () => setView(v);
        viewBtns[v] = b;
        viewSeg.appendChild(b);
    }
    const swatches = _el("div", "display:inline-flex;gap:6px;align-items:center;");
    const swBtns = {};
    for (const [name, hex] of OVERLAY_COLOURS) {
        const b = _el("button", `width:18px;height:18px;border-radius:50%;border:2px solid transparent;background:${hex};cursor:pointer;padding:0;`,
            { type: "button", title: `Mask colour: ${name}` });
        b.onclick = () => { ed.setColour(hex); syncSwatches(); };
        swBtns[hex] = b;
        swatches.appendChild(b);
    }
    const opacityCtl = range("Opacity", 0.1, 1, 0.05, ed.maskOpacity, v => `${Math.round(v * 100)}%`, v => { ed.maskOpacity = v; ed.requestDraw(); });
    const btnRevert = mkBtn("Revert", { title: "Undo every change made since the editor opened, and close" });
    const btnDone = mkBtn("Done", { bg: C.accent2, fg: "#0c1020", border: C.accent2, hover: C.accent2, title: "Close (changes are already saved)" });
    btnDone.style.fontWeight = "600";
    header.append(title, saveChip, spacer, viewSeg, swatches, opacityCtl, btnRevert, btnDone);

    // tool options bar
    const optsBar = _el("div", `display:flex;align-items:center;gap:16px;padding:8px 16px;border-bottom:1px solid ${C.border};
        background:${C.bg};flex:0 0 auto;min-height:48px;flex-wrap:wrap;`);
    const toolName = _el("div", `font-size:12px;font-weight:600;color:${C.text};min-width:76px;`);
    const sizeCtl = range("Size", 1, 600, 1, ed.size, v => `${v | 0} px`, v => { ed.size = v; ed.requestDraw(); });
    const hardCtl = range("Hardness", 0, 1, 0.05, ed.hardness, v => `${Math.round(v * 100)}%`, v => { ed.hardness = v; });
    const opCtl = range("Strength", 0.05, 1, 0.05, ed.opacity, v => `${Math.round(v * 100)}%`, v => { ed.opacity = v; });
    const tolCtl = range("Tolerance", 0, 128, 1, ed.tolerance, v => String(v | 0), v => { ed.tolerance = v; });
    const sidesCtl = range("Sides", 3, 24, 1, ed.sides, v => String(v | 0), v => { ed.sides = v | 0; ed.requestDraw(); });
    const pointCtl = range("Point", 0.1, 0.9, 0.05, ed.starInner, v => `${Math.round(v * 100)}%`, v => { ed.starInner = v; ed.requestDraw(); });
    const rotCtl = range("Rotate", 0, 360, 1, 0, v => `${v | 0}°`, v => { ed.rotation = v * Math.PI / 180; ed.requestDraw(); });
    const smoothLbl = _el("label", `display:inline-flex;gap:6px;align-items:center;color:${C.sub};font-size:12px;cursor:pointer;`);
    const smoothInp = _el("input", `accent-color:${C.accent2};`, { type: "checkbox", checked: true });
    smoothInp.onchange = () => { ed.penSmooth = smoothInp.checked; ed.requestDraw(); };
    smoothLbl.append(smoothInp, document.createTextNode("Smooth curve"));
    const toolHint = _el("div", `font-size:12px;color:${C.faint};margin-left:auto;`);
    optsBar.append(toolName, sizeCtl, hardCtl, opCtl, tolCtl, sidesCtl, pointCtl, rotCtl, smoothLbl, toolHint);

    // body: rail | viewport | side panel
    const body = _el("div", "display:flex;flex:1;min-height:0;");
    const rail = _el("div", `display:flex;flex-direction:column;gap:4px;padding:10px 8px;border-right:1px solid ${C.border};
        background:${C.bg2};flex:0 0 auto;overflow-y:auto;`);
    const TOOLS = [
        ["brush", "Brush", "B", "Paint. Hold Alt (or switch to Subtract) to remove."], ["erase", "Eraser", "E", "Remove mask."],
        ["fill", "Fill", "G", "Fill the connected area under the click."], ["lasso", "Lasso", "L", "Click points, double-click or Enter to fill."],
        ["rect", "Rectangle", "R", "Drag. Shift keeps it square."], ["ellipse", "Ellipse", "C", "Drag. Shift keeps it round."],
        ["polygon", "Polygon", "P", "Drag a regular polygon."], ["star", "Star", "T", "Drag a star."],
        ["line", "Line", "N", "Drag a line, as wide as the brush."], ["pen", "Roto spline", "Y", "Click points, click the first one or Enter to close."],
    ];
    const toolBtns = {};
    const railBtn = (icon, title) => {
        const b = _el("button", `width:38px;height:38px;border-radius:9px;border:1px solid transparent;background:transparent;
            color:${C.sub};cursor:pointer;display:flex;align-items:center;justify-content:center;`, { html: _icon(icon, 19), title, type: "button" });
        b.addEventListener("mouseenter", () => { if (!b.dataset.on) b.style.background = C.panel; });
        b.addEventListener("mouseleave", () => { if (!b.dataset.on) b.style.background = "transparent"; });
        return b;
    };
    for (const [id, name, key, hint] of TOOLS) {
        const b = railBtn(id === "erase" ? "erase" : id, `${name} (${key})`);
        b.onclick = () => setTool(id);
        b.dataset.hint = hint;
        b.dataset.name = name;
        toolBtns[id] = b;
        rail.appendChild(b);
        if (id === "lasso" || id === "line") rail.appendChild(_el("div", `height:1px;background:${C.border};margin:4px 2px;`));
    }
    rail.appendChild(_el("div", `height:1px;background:${C.border};margin:4px 2px;`));
    const btnAddSub = railBtn("add", "Add / Subtract (X)");
    btnAddSub.onclick = () => { ed.subtract = !ed.subtract; syncAddSub(); };
    rail.appendChild(btnAddSub);
    const btnUndo = railBtn("undo", "Undo (Ctrl+Z)");
    btnUndo.onclick = () => ed.undoOp();
    const btnRedo = railBtn("redo", "Redo (Ctrl+Shift+Z / Ctrl+Y)");
    btnRedo.onclick = () => ed.redoOp();
    rail.append(btnUndo, btnRedo);

    const viewport = _el("div", `flex:1;position:relative;overflow:hidden;cursor:crosshair;
        background-color:${C.bg3};background-image:linear-gradient(45deg,#ffffff08 25%,transparent 25%,transparent 75%,#ffffff08 75%),
        linear-gradient(45deg,#ffffff08 25%,transparent 25%,transparent 75%,#ffffff08 75%);background-size:24px 24px;background-position:0 0,12px 12px;`);
    const canvas = _el("canvas", "position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;outline:none;");
    canvas.tabIndex = 0;
    canvas.dataset.vmeCanvas = "1";
    const zoomLabel = _el("div", `position:absolute;right:12px;bottom:10px;font-size:11.5px;color:${C.sub};background:${C.bg2}cc;
        padding:3px 8px;border-radius:6px;border:1px solid ${C.border};pointer-events:none;font-variant-numeric:tabular-nums;`);
    const banner = _el("div", `position:absolute;left:12px;top:10px;right:12px;font-size:12px;color:${C.warn};background:${C.bg2}e6;
        padding:6px 10px;border-radius:7px;border:1px solid ${C.warn}55;pointer-events:none;`);
    banner.hidden = true;
    const loading = _el("div", `position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;
        gap:10px;color:${C.sub};font-size:13px;pointer-events:none;`);
    loading.innerHTML = `<div style="width:28px;height:28px;border-radius:50%;border:3px solid ${C.border};border-top-color:${C.accent2};
        animation:c2c-vme-spin .8s linear infinite"></div><div>Loading frames...</div>
        <style>@keyframes c2c-vme-spin{to{transform:rotate(360deg)}}@media (prefers-reduced-motion:reduce){[style*=c2c-vme-spin]{animation:none}}</style>`;
    viewport.append(canvas, zoomLabel, banner, loading);

    const side = _el("div", `width:296px;flex:0 0 296px;border-left:1px solid ${C.border};background:${C.bg2};display:flex;
        flex-direction:column;min-height:0;`);
    const section = (titleText, extra) => {
        const sec = _el("div", `padding:14px 16px;border-bottom:1px solid ${C.border};display:flex;flex-direction:column;gap:10px;`);
        const head = _el("div", "display:flex;align-items:center;gap:8px;");
        head.appendChild(_el("div", `font-size:11px;font-weight:600;letter-spacing:.8px;text-transform:uppercase;color:${C.sub};`, { text: titleText }));
        if (extra) head.appendChild(extra);
        sec.appendChild(head);
        return sec;
    };
    // frame card
    const frameSec = section("This frame");
    const stateChip = _el("div", `align-self:flex-start;font-size:12.5px;font-weight:600;padding:5px 11px;border-radius:999px;border:1px solid ${C.border};`);
    const hint = _el("div", `font-size:12px;color:${C.sub};line-height:1.5;`);
    const frameOps = _el("div", "display:grid;grid-template-columns:1fr 1fr;gap:6px;");
    const btnMakeKey = mkBtn(`${_icon("key", 15)} Add key`, { title: "Keep the mask shown here as a keyframe (K)" });
    const btnDelKey = mkBtn(`${_icon("trash", 15)} Delete key`, { title: "Delete this keyframe: the frame goes back to the interpolated / held mask" });
    const btnCopy = mkBtn(`${_icon("copy", 15)} Copy nearest`, { title: "Copy the nearest keyframe's mask onto this frame" });
    const btnClear = mkBtn(`${_icon("clear", 15)} Clear`, { title: "Empty this frame's mask (Delete)" });
    const btnFill = mkBtn(`${_icon("square", 15)} Fill frame`, { title: "Mask the whole frame" });
    const btnInvert = mkBtn(`${_icon("invert", 15)} Invert`, { title: "Invert this frame's mask (I)" });
    frameOps.append(btnMakeKey, btnDelKey, btnCopy, btnClear, btnFill, btnInvert);
    stateChip.dataset.vme = "state";
    frameSec.append(stateChip, hint, frameOps);
    // interpolation card
    const interpSec = section("Unpainted frames");
    const capCss = `font-size:12px;color:${C.sub};margin-bottom:-4px;`;
    const tweenSel = select(TWEENS, ed.tween, v => { ed.setWidget("tween_mode", v); onModeChange(); }, "Frames between two painted frames");
    const outsideSel = select(OUTSIDES, ed.outside, v => { ed.setWidget("outside_keys", v); onModeChange(); }, "Frames before the first and after the last painted frame");
    tweenSel.dataset.vme = "tween";
    outsideSel.dataset.vme = "outside";
    interpSec.append(_el("div", capCss, { text: "Between two painted frames" }), tweenSel,
        _el("div", capCss, { text: "Before the first / after the last" }), outsideSel,
        _el("div", `font-size:11.5px;color:${C.faint};line-height:1.5;`,
            { text: "The node outputs exactly what you see on each frame. With an input mask connected, frames no rule fills keep the input mask." }));
    // keyframes
    const kfCount = _el("span", `font-size:11px;color:${C.faint};font-variant-numeric:tabular-nums;`);
    const kfSec = section("Keyframes", kfCount);
    kfSec.style.flex = "1";
    kfSec.style.minHeight = "0";
    const kfList = _el("div", "display:flex;flex-direction:column;gap:2px;overflow-y:auto;min-height:0;flex:1;margin:0 -6px;");
    kfSec.appendChild(kfList);
    const keys = _el("details", `padding:10px 16px;font-size:11.5px;color:${C.sub};line-height:1.7;`);
    keys.innerHTML = `<summary style="cursor:pointer;color:${C.text};font-weight:600;font-size:12px;">Keyboard</summary>
        <div>B E G L R C P T N Y &middot; tools &nbsp; X &middot; add / subtract</div>
        <div>[ ] &middot; size &nbsp; Alt+drag &middot; subtract while painting</div>
        <div>&larr; &rarr; &middot; frame &nbsp; Shift+&larr;/&rarr; &middot; previous / next keyframe</div>
        <div>Home / End &middot; first / last frame &nbsp; / &middot; play</div>
        <div>K &middot; make keyframe &nbsp; Delete &middot; clear &nbsp; I &middot; invert</div>
        <div>V &middot; view &nbsp; O &middot; onion skin &nbsp; Shift+F &middot; fit</div>
        <div>Ctrl+Z / Ctrl+Shift+Z &middot; undo / redo &nbsp; wheel &middot; zoom &nbsp; Space / middle-drag &middot; pan</div>`;
    side.append(frameSec, interpSec, kfSec, keys);
    body.append(rail, viewport, side);

    // transport + timeline
    const bottom = _el("div", `display:flex;align-items:center;gap:10px;padding:8px 16px;border-top:1px solid ${C.border};background:${C.bg2};flex:0 0 auto;`);
    const tbtn = (icon, title, fn) => { const b = mkBtn(_icon(icon), { icon: true, title }); b.onclick = fn; return b; };
    const btnFirst = tbtn("first", "First frame (Home)", () => ed.goto(0));
    const btnPrevKey = tbtn("prevKey", "Previous keyframe (Shift+Left)", () => ed.jumpKey(-1));
    const btnPrev = tbtn("prev", "Previous frame (Left)", () => ed.step(-1));
    const btnPlay = tbtn("play", "Play / pause (/)", () => ed.togglePlay());
    const btnNext = tbtn("next", "Next frame (Right)", () => ed.step(1));
    const btnNextKey = tbtn("nextKey", "Next keyframe (Shift+Right)", () => ed.jumpKey(1));
    const btnLast = tbtn("last", "Last frame (End)", () => ed.goto(ed.count - 1));
    const frameField = _el("input", `width:56px;height:30px;border-radius:7px;border:1px solid ${C.border};background:${C.bg};color:${C.text};
        text-align:center;font-variant-numeric:tabular-nums;font-size:12.5px;${font}`, { type: "number", min: 0, value: 0, title: "Frame" });
    frameField.onchange = () => ed.goto(+frameField.value || 0);
    const frameTotal = _el("span", `font-size:12px;color:${C.sub};font-variant-numeric:tabular-nums;min-width:44px;`);
    const timeline = _el("canvas", "flex:1;height:46px;min-width:120px;cursor:pointer;border-radius:8px;display:block;");
    const btnOnion = tbtn("onion", "Onion skin: show the neighbouring keyframes (O)", () => { ed.onion = !ed.onion; syncToggles(); ed.requestDraw(); });
    const btnFit = tbtn("fit", "Fit the frame (Shift+F)", () => ed.fit());
    bottom.append(btnFirst, btnPrevKey, btnPrev, btnPlay, btnNext, btnNextKey, btnLast, frameField, frameTotal, timeline, btnOnion, btnFit);

    overlay.append(header, optsBar, body, bottom);
    document.body.appendChild(overlay);
    ed.ui = { canvas, viewport, timeline, saveChip, stateChip, hint, kfList, kfCount, btnDelKey, btnMakeKey, frameField, frameTotal,
              zoomLabel, btnPlay };
    ed.setSaveState("saved");

    // ── wiring ──────────────────────────────────────────────────────
    function setTool(id) {
        if (ed.tool === "pen" && id !== "pen" && ed.pen.length >= 3) ed.commitPen();
        ed.tool = id;
        ed.lasso = []; ed.pen = []; ed.penHover = null; ed.shapeStart = ed.shapeCur = null;
        for (const [k, b] of Object.entries(toolBtns)) {
            const on = k === id;
            b.dataset.on = on ? "1" : "";
            b.style.background = on ? C.accent2 + "26" : "transparent";
            b.style.color = on ? C.text : C.sub;
            b.style.borderColor = on ? C.accent2 + "88" : "transparent";
        }
        const b = toolBtns[id];
        toolName.textContent = b.dataset.name;
        toolHint.textContent = b.dataset.hint;
        const paint = id === "brush" || id === "erase", shape = ["rect", "ellipse", "polygon", "star"].includes(id);
        sizeCtl.style.display = paint || id === "line" ? "" : "none";
        hardCtl.style.display = paint ? "" : "none";
        opCtl.style.display = id === "fill" ? "none" : "";
        tolCtl.style.display = id === "fill" ? "" : "none";
        sidesCtl.style.display = id === "polygon" || id === "star" ? "" : "none";
        pointCtl.style.display = id === "star" ? "" : "none";
        rotCtl.style.display = shape ? "" : "none";
        smoothLbl.style.display = id === "pen" ? "" : "none";
        canvas.style.cursor = paint ? "none" : "crosshair";
        ed.requestDraw();
    }
    function syncAddSub() {
        btnAddSub.innerHTML = _icon(ed.subtract ? "sub" : "add", 19);
        btnAddSub.title = ed.subtract ? "Subtract: tools remove mask (X)" : "Add: tools add mask (X)";
        btnAddSub.style.color = ed.subtract ? C.danger : C.accent;
        btnAddSub.style.borderColor = ed.subtract ? C.danger + "66" : C.accent + "55";
        ed.requestDraw();
    }
    function setView(v) {
        ed.viewMode = v;
        for (const [k, b] of Object.entries(viewBtns)) {
            b.style.background = k === v ? C.raised : "transparent";
            b.style.color = k === v ? C.text : C.sub;
        }
        ed.requestDraw();
    }
    function syncSwatches() {
        for (const [hex, b] of Object.entries(swBtns)) b.style.borderColor = hex === ed.colour ? C.text : "transparent";
    }
    function syncToggles() {
        btnOnion.dataset.on = ed.onion ? "1" : "";
        btnOnion.style.background = ed.onion ? C.accent2 + "26" : C.panel;
        btnOnion.style.borderColor = ed.onion ? C.accent2 + "88" : C.border;
    }
    function onModeChange() {
        ed.effCache.clear();
        if (!ed.alpha) ed._fetchEffective(ed.cur);
        ed.refreshPanels();
        ed.requestDraw();
    }
    btnMakeKey.onclick = () => ed.makeKeyframe();
    btnDelKey.onclick = () => ed.deleteKeyframe(ed.cur);
    btnCopy.onclick = () => ed.copyFromNearestKeyframe();
    btnClear.onclick = () => ed.wholeFrame("clear");
    btnFill.onclick = () => ed.wholeFrame("fill");
    btnInvert.onclick = () => ed.wholeFrame("invert");

    // pointer
    let panning = null, painting = false, spaceDown = false;
    const viewXY = (e) => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    canvas.addEventListener("pointerdown", (e) => {
        canvas.focus();
        canvas.setPointerCapture?.(e.pointerId);
        const v = viewXY(e);
        if (e.button === 1 || (e.button === 0 && spaceDown)) { panning = { ...v, px: ed.panX, py: ed.panY }; e.preventDefault(); return; }
        if (e.button !== 0 || !ed.W || !ed.ready) return;
        const p = ed.toImage(v.x, v.y);
        if (ed.playing) ed.togglePlay();
        if (ed.tool === "brush" || ed.tool === "erase") {
            ed._altSub = e.altKey && ed.tool === "brush" && !ed.subtract;
            if (ed._altSub) ed.subtract = true;
            ed.beginEdit();
            painting = true;
            ed.stamp(p.x, p.y, null);
            ed._last = p;
            ed.requestDraw();
        } else if (ed.tool === "fill") {
            ed.beginEdit(); ed.floodFill(p.x, p.y); ed.endEdit(); ed.requestDraw();
        } else if (ed.tool === "lasso") {
            ed.lasso.push(p); ed.requestDraw();
        } else if (ed.tool === "pen") {
            const hit = ed.pen.findIndex(q => Math.hypot(q.x - p.x, q.y - p.y) <= 8 / ed.zoom);
            if (hit === 0 && ed.pen.length >= 3) { ed.commitPen(); return; }
            if (hit >= 0) ed.penDrag = hit; else ed.pen.push(p);
            ed.requestDraw();
        } else {
            ed.shapeStart = p; ed.shapeCur = p; ed._shapeSquare = e.shiftKey; ed.requestDraw();
        }
        e.preventDefault();
    });
    canvas.addEventListener("pointermove", (e) => {
        const v = viewXY(e);
        ed.cursor = v;
        const p = ed.toImage(v.x, v.y);
        ed.cursorImg = p;
        if (panning) { ed.panX = panning.px + v.x - panning.x; ed.panY = panning.py + v.y - panning.y; ed.requestDraw(); return; }
        if (painting) {
            const evs = e.getCoalescedEvents?.() || [e];
            for (const ce of evs) {
                const cv = viewXY(ce), cp = ed.toImage(cv.x, cv.y);
                ed.stamp(cp.x, cp.y, ed._last);
                ed._last = cp;
            }
        } else if (ed.shapeStart) { ed.shapeCur = p; ed._shapeSquare = e.shiftKey; }
        else if (ed.tool === "pen") { if (ed.penDrag >= 0) ed.pen[ed.penDrag] = p; ed.penHover = p; }
        ed.requestDraw();
    });
    const endPointer = () => {
        if (panning) panning = null;
        if (painting) {
            painting = false;
            ed.endEdit();
            if (ed._altSub) { ed.subtract = false; ed._altSub = false; }
        }
        if (ed.penDrag >= 0) ed.penDrag = -1;
        if (ed.shapeStart && ed.shapeCur) {
            const a = ed.shapeStart, b = ed.shapeCur;
            ed.shapeStart = ed.shapeCur = null;
            if (Math.hypot(b.x - a.x, b.y - a.y) > 1.5) {
                ed.beginEdit();
                if (ed.tool === "line") ed.fillPath([[a.x, a.y], [b.x, b.y]], { line: true });
                else ed.fillPath(ed.shapePoints(ed.tool, a, b, ed._shapeSquare));
                ed.endEdit();
            }
        }
        ed.requestDraw();
    };
    canvas.addEventListener("pointerup", endPointer);
    canvas.addEventListener("pointercancel", endPointer);
    canvas.addEventListener("pointerleave", () => { ed.cursor = null; ed.requestDraw(); });
    const finishLasso = () => {
        if (ed.tool === "lasso" && ed.lasso.length >= 3) {
            ed.beginEdit(); ed.fillPath(ed.lasso.map(p => [p.x, p.y])); ed.endEdit(); ed.lasso = []; ed.requestDraw();
        }
    };
    canvas.addEventListener("dblclick", () => { finishLasso(); if (ed.tool === "pen") ed.commitPen(); });
    canvas.addEventListener("contextmenu", (e) => { e.preventDefault(); finishLasso(); });
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const v = viewXY(e);
        ed.zoomAt(v.x, v.y, Math.exp(-e.deltaY * 0.0015));
    }, { passive: false });

    // timeline scrub
    let scrubbing = false;
    const scrubTo = (e) => {
        const r = timeline.getBoundingClientRect(), pad = 14;
        const t = (e.clientX - r.left - pad) / Math.max(1, r.width - 2 * pad);
        ed.goto(Math.max(0, Math.min(ed.count - 1, Math.floor(t * ed.count))));
    };
    timeline.addEventListener("pointerdown", (e) => { scrubbing = true; timeline.setPointerCapture?.(e.pointerId); if (ed.playing) ed.togglePlay(); scrubTo(e); });
    timeline.addEventListener("pointermove", (e) => { if (scrubbing) scrubTo(e); });
    timeline.addEventListener("pointerup", () => { scrubbing = false; });

    // keyboard (capture phase: the editor sees keys before ComfyUI's window shortcuts; L2.15)
    const TOOL_KEYS = { b: "brush", e: "erase", g: "fill", l: "lasso", r: "rect", c: "ellipse", p: "polygon", t: "star", n: "line", y: "pen" };
    const onKey = (e) => {
        if (e.key === " ") { spaceDown = e.type === "keydown"; if (e.target === canvas || e.target === document.body) e.preventDefault(); return; }
        if (e.type !== "keydown") return;
        if (!ed.loaded) { if (e.key === "Escape") { close(false); e.preventDefault(); } return; }
        const tag = e.target?.tagName;
        if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") { if (e.key === "Escape") e.target.blur(); return; }
        const mod = e.ctrlKey || e.metaKey, k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
        let done = true;
        if (k === "Escape") {
            if (ed.pen.length || ed.lasso.length) { ed.pen = []; ed.lasso = []; ed.requestDraw(); } else close(false);
        } else if (k === "Enter") { finishLasso(); if (ed.tool === "pen") ed.commitPen(); }
        else if (k === "Backspace" && ed.tool === "pen" && ed.pen.length) { ed.pen.pop(); ed.requestDraw(); }
        else if (mod && k === "z" && !e.shiftKey) ed.undoOp();
        else if (mod && (k === "y" || (k === "z" && e.shiftKey))) ed.redoOp();
        else if (mod) done = false;
        else if (TOOL_KEYS[k]) setTool(TOOL_KEYS[k]);
        else if (k === "x") { ed.subtract = !ed.subtract; syncAddSub(); }
        else if (k === "[") { ed.size = Math.max(1, Math.round(ed.size / 1.15)); sizeCtl.set(ed.size); ed.requestDraw(); }
        else if (k === "]") { ed.size = Math.min(600, Math.round(ed.size * 1.15) + 1); sizeCtl.set(ed.size); ed.requestDraw(); }
        else if (k === "ArrowLeft") e.shiftKey ? ed.jumpKey(-1) : ed.step(-1);
        else if (k === "ArrowRight") e.shiftKey ? ed.jumpKey(1) : ed.step(1);
        else if (k === "Home") ed.goto(0);
        else if (k === "End") ed.goto(ed.count - 1);
        else if (k === "/") ed.togglePlay();
        else if (k === "k") ed.makeKeyframe();
        else if (k === "Delete") ed.wholeFrame("clear");
        else if (k === "i") ed.wholeFrame("invert");
        else if (k === "o") { ed.onion = !ed.onion; syncToggles(); ed.requestDraw(); }
        else if (k === "v") { const i = VIEW_MODES.findIndex(([m]) => m === ed.viewMode); setView(VIEW_MODES[(i + 1) % VIEW_MODES.length][0]); }
        else if (k === "f" && e.shiftKey) ed.fit();
        else done = false;
        if (done) { e.preventDefault(); e.stopPropagation(); }
    };
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("keyup", onKey, true);
    const ro = new ResizeObserver(() => { ed.requestDraw(); ed.drawTimeline(); });
    ro.observe(viewport);
    ro.observe(timeline);

    let closing = false;
    async function close(revert) {
        if (closing) return;
        if (revert && ed.touched.size + (ed.dirty ? 1 : 0)) {
            const ok = await c2cConfirm("Revert every change made since the editor opened?");
            if (!ok) return;
        }
        closing = true;
        ed.playing = false;
        window.removeEventListener("keydown", onKey, true);
        window.removeEventListener("keyup", onKey, true);
        ro.disconnect();
        try {
            if (revert) await ed.revert();
            else if (!(await ed.commit())) {
                closing = false;
                window.addEventListener("keydown", onKey, true);
                window.addEventListener("keyup", onKey, true);
                c2cAlert("The last change could not be saved to the server. Check that ComfyUI is running, then press Done again.");
                return;
            }
        } catch (e) { __c2cReport("video_mask_editor.close", e); }
        overlay.remove();
        node.setDirtyCanvas?.(true, true);
        node.__vmeRefresh?.();
    }
    btnDone.onclick = () => close(false);
    btnRevert.onclick = () => close(true);

    setTool("brush");
    syncAddSub();
    setView("overlay");
    syncSwatches();
    syncToggles();
    ed.load().then(async (ok) => {
        if (!ok) { close(false); return; }
        meta.textContent = `${ed.W} × ${ed.H} · ${ed.count} frame${ed.count === 1 ? "" : "s"}`;
        if (ed.source.sampled || ed.source.kind === "single") {
            banner.hidden = false;
            banner.textContent = ed.source.note || "These frames are a preview and may not match the batch this node receives.";
        }
        ed.cur = -1;
        await ed.goto(0);
        ed.fit();
        ed.refreshPanels();
        ed.loaded = true;
        loading.remove();
        overlay.dataset.ready = "1";
        canvas.focus();
    }).catch((e) => {
        __c2cReport("video_mask_editor.load", e);
        c2cAlert("Video Mask Editor could not load its frames: " + (e?.message || e));
        close(false);
    });
}

// ─── Extension ──────────────────────────────────────────────────────
app.registerExtension({
    name: "MEC.VideoMaskEditor",
    settings: [
        { id: "c2c.vme.previewMaxPx", name: "Video Mask Editor: frame size (long side, 0 = the clip's own)", type: "number",
          defaultValue: 4096, attrs: { min: 0, max: 16384, step: 64 },
          tooltip: "Frames larger than this are shown scaled down. The mask is always painted at the clip's own resolution." },
        { id: "c2c.vme.sampledFrames", name: "Video Mask Editor: sampled video frame count", type: "number", defaultValue: 32,
          attrs: { min: 2, max: 2000, step: 1 },
          tooltip: "When no exact frame plan is available, sample this many evenly spaced frames from the video preview element." },
        { id: "c2c.vme.cacheFrames", name: "Video Mask Editor: decoded frame cache size", type: "number", defaultValue: 48,
          attrs: { min: 4, max: 2000, step: 1 }, tooltip: "Decoded frames kept in memory while the editor is open." },
        { id: "c2c.vme.playFps", name: "Video Mask Editor: playback frame rate", type: "number", defaultValue: 12,
          attrs: { min: 1, max: 60, step: 1 }, tooltip: "Frames per second when the editor plays the clip." },
    ],
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            const self = this;
            this.addWidget("button", "Open Video Mask Editor", null, () => openEditor(self), { serialize: false });
            // the session id is state, not a setting (L9.17)
            setTimeout(() => {
                const w = self.widgets?.find(x => x.name === "session_id");
                if (!w) return;
                if (!w.value || w.value.length < 8) w.value = uuid();
                try { setHidden(w, true); } catch (e) { __c2cReport("video_mask_editor.hide_session", e); }
                self.setSize?.([Math.max(self.size?.[0] || 0, 300), self.computeSize?.()[1] ?? self.size?.[1]]);
                self.setDirtyCanvas?.(true, true);
            }, 0);
        };
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origConfigure?.apply(this, arguments);
            const w = this.widgets?.find(x => x.name === "session_id");
            if (w) { try { setHidden(w, true); } catch (_) { /* older frontend */ } }
            return r;
        };
    },
});
