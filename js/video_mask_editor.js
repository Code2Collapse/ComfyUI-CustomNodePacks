/**
 * VideoMaskEditorMEC — fullscreen video mask editor.
 *
 * Tools: brush · erase · fill · lasso · onion-skin · history.
 * Storage: per-pinned-frame ImageData kept client-side, POSTed to the
 *   server-side session on Save (PNG-compressed per frame).
 * UX target: high-contrast dark UI, GPU-light (one canvas per frame
 *   on demand; off-screen mask buffer only for the current frame).
 */

import { app } from "../../scripts/app.js";
import { findUpstreamFramesAsync } from "./_frame_finder.js";

const NODE_NAME = "VideoMaskEditorMEC";

const C = {
    bg:      "#0e0e16",
    panel:   "#181825",
    border:  "#313244",
    text:    "#cdd6f4",
    sub:     "#7f849c",
    accent:  "#a6e3a1",
    accent2: "#89b4fa",
    warn:    "#fab387",
    danger:  "#f38ba8",
    onion:   "#f9e2af",
};

const HISTORY_LIMIT = 60;

// ─── UUID generator (RFC4122 v4, crypto if available) ────────────────
function uuid() {
    if (crypto?.randomUUID) return crypto.randomUUID();
    return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c =>
        (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & 15) >> (c / 4)).toString(16));
}

// ─── Upstream frame discovery (shared finder) ──────────────────────
// Implementation lives in _frame_finder.js and handles video sources,
// sibling preview scan, and single-frame fallback.

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const cached = loadImage.cache?.get(url);
        if (cached?.complete) return resolve(cached);
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            if (!loadImage.cache) loadImage.cache = new Map();
            loadImage.cache.set(url, img);
            resolve(img);
        };
        img.onerror = reject;
        img.src = url;
    });
}


// ─── Editor session class ───────────────────────────────────────────
class VMEEditor {
    constructor(node) {
        this.node = node;
        this.sessionId = this._getOrCreateSession();
        this.frameUrls = [];
        this.frameImgs = [];       // loaded HTMLImageElements
        this.frameCount = 0;
        this.frameW = 0;
        this.frameH = 0;
        this.curFrame = 0;
        // Pinned keyframes: Map<frame:int, ImageData>
        this.keyframes = new Map();
        this.curMask = null;       // ImageData for current frame's working buffer
        this.curDirty = false;
        // Tool state
        this.tool = "brush";       // brush|erase|fill|lasso
        this.brushRadius = 30;
        this.brushOpacity = 1.0;
        this.brushFeather = 0.6;
        this.onion = true;
        this.painting = false;
        this.lastPaint = null;
        this.lasso = [];           // polygon points in canvas coords
        // View
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        // Undo (per-frame stack of ImageData)
        this.history = new Map();  // frame -> [ImageData,...]
        this.redo = new Map();
        // DOM refs filled by mount()
        this.dom = {};
    }

    _getOrCreateSession() {
        const w = this.node.widgets?.find(w => w.name === "session_id");
        if (!w) return "";
        if (!w.value || typeof w.value !== "string" || w.value.length < 8) {
            w.value = uuid();
        }
        return w.value;
    }

    // ── frame init / switch ──────────────────────────────────────────
    async loadFrames() {
        const urls = await findUpstreamFramesAsync(this.node, { maxVideoFrames: 32 });
        if (!urls.length) {
            alert("[VideoMaskEditor]\n\nNo frames found upstream. " +
                  "Queue Prompt once so the upstream image source has " +
                  "previewable frames, then re-open the editor.");
            return false;
        }
        this.frameUrls = urls;
        this.frameCount = urls.length;
        this.frameImgs = new Array(urls.length).fill(null);
        // Load first frame to learn dimensions.
        const im0 = await loadImage(urls[0]);
        this.frameW = im0.naturalWidth;
        this.frameH = im0.naturalHeight;
        this.frameImgs[0] = im0;
        // Lazy-load others in background.
        urls.slice(1).forEach((u, i) => {
            loadImage(u).then(img => { this.frameImgs[i + 1] = img; this.draw(); })
                        .catch(() => {});
        });
        // Init session on server.
        try {
            await fetch(`/mec/video_mask_editor/init?session=${encodeURIComponent(this.sessionId)}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ h: this.frameH, w: this.frameW }),
            });
        } catch (e) { console.warn("[VME] init failed:", e); }
        // Fetch existing server-side keyframes (in case we re-opened a session).
        try {
            const r = await fetch(`/mec/video_mask_editor/state?session=${encodeURIComponent(this.sessionId)}`);
            const j = await r.json();
            for (const f of j.keyframes || []) await this._fetchExistingKeyframe(f);
        } catch (e) { /* fine */ }
        this._switchFrame(0, { fresh: true });
        return true;
    }

    async _fetchExistingKeyframe(f) {
        try {
            const r = await fetch(`/mec/video_mask_editor/keyframe?session=${encodeURIComponent(this.sessionId)}&frame=${f}`);
            if (!r.ok) return;
            const blob = await r.blob();
            const bm = await createImageBitmap(blob);
            const c = document.createElement("canvas");
            c.width = this.frameW; c.height = this.frameH;
            const cx = c.getContext("2d");
            cx.drawImage(bm, 0, 0, this.frameW, this.frameH);
            // Convert grayscale L to RGBA red-tint mask.
            const id = cx.getImageData(0, 0, this.frameW, this.frameH);
            // Original L is in R==G==B; we want alpha = R, color = accent.
            const accent = this._hexToRgb(C.accent);
            for (let i = 0; i < id.data.length; i += 4) {
                const a = id.data[i];
                id.data[i] = accent.r;
                id.data[i + 1] = accent.g;
                id.data[i + 2] = accent.b;
                id.data[i + 3] = a;
            }
            this.keyframes.set(f, id);
        } catch (e) { /* ignore */ }
    }

    _hexToRgb(hex) {
        const h = hex.replace("#", "");
        return {
            r: parseInt(h.slice(0, 2), 16),
            g: parseInt(h.slice(2, 4), 16),
            b: parseInt(h.slice(4, 6), 16),
        };
    }

    _emptyMask() {
        return new ImageData(this.frameW, this.frameH);
    }

    /** Save current working mask back into keyframes (if dirty) and
     *  load the new frame's mask. */
    _switchFrame(idx, { fresh = false } = {}) {
        idx = Math.max(0, Math.min(this.frameCount - 1, idx | 0));
        if (!fresh) this._commitCurrent();
        this.curFrame = idx;
        // Working buffer: start from pinned keyframe at this frame
        // if any, else from an empty mask. We do NOT auto-tween in the
        // editor — user is editing keyframes explicitly. (Tweening
        // happens server-side at execute time.)
        const kf = this.keyframes.get(idx);
        if (kf) {
            this.curMask = new ImageData(
                new Uint8ClampedArray(kf.data), kf.width, kf.height);
        } else {
            this.curMask = this._emptyMask();
        }
        this.curDirty = false;
        this.draw();
    }

    /** If user painted on this frame, store as a keyframe. */
    _commitCurrent() {
        if (!this.curDirty || !this.curMask) return;
        this.keyframes.set(this.curFrame, this.curMask);
        this.curDirty = false;
    }

    // ── Undo / redo (per-frame) ──────────────────────────────────────
    _pushUndo() {
        if (!this.curMask) return;
        const stk = this.history.get(this.curFrame) || [];
        stk.push(new ImageData(new Uint8ClampedArray(this.curMask.data),
                               this.curMask.width, this.curMask.height));
        if (stk.length > HISTORY_LIMIT) stk.shift();
        this.history.set(this.curFrame, stk);
        this.redo.set(this.curFrame, []);
    }
    undo() {
        const stk = this.history.get(this.curFrame);
        if (!stk?.length) return;
        const cur = new ImageData(new Uint8ClampedArray(this.curMask.data),
                                  this.curMask.width, this.curMask.height);
        const r = this.redo.get(this.curFrame) || [];
        r.push(cur);
        this.redo.set(this.curFrame, r);
        this.curMask = stk.pop();
        this.curDirty = true;
        this.draw();
    }
    redoOp() {
        const r = this.redo.get(this.curFrame);
        if (!r?.length) return;
        const stk = this.history.get(this.curFrame) || [];
        stk.push(new ImageData(new Uint8ClampedArray(this.curMask.data),
                               this.curMask.width, this.curMask.height));
        this.history.set(this.curFrame, stk);
        this.curMask = r.pop();
        this.curDirty = true;
        this.draw();
    }

    // ── Painting ─────────────────────────────────────────────────────
    _stamp(x, y, prev) {
        // Draw soft brush onto curMask. We use straight per-pixel
        // alpha-modify so brush size doesn't depend on canvas zoom.
        const data = this.curMask.data;
        const W = this.curMask.width, H = this.curMask.height;
        const r = this.brushRadius;
        const r2 = r * r;
        const isErase = this.tool === "erase";
        const op = this.brushOpacity;
        const featherStart = r * (1 - this.brushFeather);
        const accent = this._hexToRgb(C.accent);
        const lineCount = prev
            ? Math.max(1, Math.ceil(Math.hypot(x - prev.x, y - prev.y) / Math.max(1, r * 0.35)))
            : 1;
        for (let step = 0; step <= lineCount; step++) {
            const t = lineCount === 0 ? 0 : step / lineCount;
            const cx = prev ? prev.x + (x - prev.x) * t : x;
            const cy = prev ? prev.y + (y - prev.y) * t : y;
            const x0 = Math.max(0, Math.floor(cx - r));
            const y0 = Math.max(0, Math.floor(cy - r));
            const x1 = Math.min(W - 1, Math.ceil(cx + r));
            const y1 = Math.min(H - 1, Math.ceil(cy + r));
            for (let yy = y0; yy <= y1; yy++) {
                const dy = yy - cy;
                for (let xx = x0; xx <= x1; xx++) {
                    const dx = xx - cx;
                    const d2 = dx * dx + dy * dy;
                    if (d2 > r2) continue;
                    const d = Math.sqrt(d2);
                    let falloff = 1;
                    if (d > featherStart) {
                        const t2 = (d - featherStart) / Math.max(1e-6, r - featherStart);
                        falloff = 1 - t2;
                    }
                    const add = Math.round(255 * op * falloff);
                    const i = (yy * W + xx) * 4;
                    if (isErase) {
                        const cur = data[i + 3];
                        const newA = Math.max(0, cur - add);
                        data[i + 3] = newA;
                    } else {
                        const cur = data[i + 3];
                        const newA = Math.min(255, cur + add);
                        data[i + 3] = newA;
                        data[i] = accent.r;
                        data[i + 1] = accent.g;
                        data[i + 2] = accent.b;
                    }
                }
            }
        }
    }

    _floodFill(x, y) {
        // Scanline flood-fill on alpha channel; fills connected region
        // of similar alpha with full opacity.
        x = x | 0; y = y | 0;
        const W = this.curMask.width, H = this.curMask.height;
        if (x < 0 || y < 0 || x >= W || y >= H) return;
        const data = this.curMask.data;
        const accent = this._hexToRgb(C.accent);
        const startIdx = (y * W + x) * 4 + 3;
        const startA = data[startIdx];
        const isErase = this.tool === "erase";
        const target = isErase ? 0 : 255;
        if (startA === target) return;
        const tol = 32;
        const stack = [[x, y]];
        const seen = new Uint8Array(W * H);
        while (stack.length) {
            const [cx, cy] = stack.pop();
            if (cx < 0 || cy < 0 || cx >= W || cy >= H) continue;
            const fIdx = cy * W + cx;
            if (seen[fIdx]) continue;
            const a = data[fIdx * 4 + 3];
            if (Math.abs(a - startA) > tol) continue;
            seen[fIdx] = 1;
            data[fIdx * 4 + 3] = target;
            if (!isErase) {
                data[fIdx * 4] = accent.r;
                data[fIdx * 4 + 1] = accent.g;
                data[fIdx * 4 + 2] = accent.b;
            }
            stack.push([cx + 1, cy], [cx - 1, cy], [cx, cy + 1], [cx, cy - 1]);
        }
    }

    _fillLasso() {
        if (this.lasso.length < 3) return;
        const W = this.curMask.width, H = this.curMask.height;
        // Build off-screen path → bitmask via canvas.
        const off = document.createElement("canvas");
        off.width = W; off.height = H;
        const oc = off.getContext("2d");
        oc.fillStyle = "#ffffff";
        oc.beginPath();
        oc.moveTo(this.lasso[0].x, this.lasso[0].y);
        for (let i = 1; i < this.lasso.length; i++) oc.lineTo(this.lasso[i].x, this.lasso[i].y);
        oc.closePath();
        oc.fill();
        const id = oc.getImageData(0, 0, W, H);
        const src = id.data;
        const dst = this.curMask.data;
        const isErase = this.tool === "erase";
        const accent = this._hexToRgb(C.accent);
        const op = Math.round(255 * this.brushOpacity);
        for (let i = 0; i < dst.length; i += 4) {
            if (src[i] === 0) continue;  // outside polygon
            if (isErase) {
                dst[i + 3] = Math.max(0, dst[i + 3] - op);
            } else {
                dst[i + 3] = Math.min(255, dst[i + 3] + op);
                dst[i] = accent.r;
                dst[i + 1] = accent.g;
                dst[i + 2] = accent.b;
            }
        }
    }

    // ── Coord transforms ─────────────────────────────────────────────
    _viewToFrame(vx, vy) {
        return { x: (vx - this.panX) / this.zoom, y: (vy - this.panY) / this.zoom };
    }
    fitView() {
        const r = this.dom.viewport.getBoundingClientRect();
        const pad = 20;
        const sx = (r.width - pad * 2) / this.frameW;
        const sy = (r.height - pad * 2) / this.frameH;
        this.zoom = Math.max(0.05, Math.min(sx, sy, 8));
        this.panX = (r.width - this.frameW * this.zoom) / 2;
        this.panY = (r.height - this.frameH * this.zoom) / 2;
    }

    // ── Render ───────────────────────────────────────────────────────
    draw() {
        const cnv = this.dom.canvas;
        const r = this.dom.viewport.getBoundingClientRect();
        const W = Math.max(1, Math.floor(r.width));
        const H = Math.max(1, Math.floor(r.height));
        if (cnv.width !== W || cnv.height !== H) {
            cnv.width = W; cnv.height = H;
        }
        const ctx = cnv.getContext("2d");
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, W, H);
        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);

        // Underlay: current video frame.
        const frame = this.frameImgs[this.curFrame];
        if (frame?.complete) {
            ctx.imageSmoothingEnabled = true;
            try { ctx.drawImage(frame, 0, 0, this.frameW, this.frameH); }
            catch (_) {}
        } else {
            ctx.fillStyle = "#11111b";
            ctx.fillRect(0, 0, this.frameW, this.frameH);
            ctx.fillStyle = C.sub;
            ctx.font = "14px Inter, system-ui";
            ctx.fillText("loading frame…", 12, 20);
        }

        // Onion skin: nearest pinned keyframes (one back, one forward).
        if (this.onion && this.keyframes.size) {
            const sorted = [...this.keyframes.keys()].sort((a, b) => a - b);
            let prev = null, next = null;
            for (const f of sorted) {
                if (f < this.curFrame) prev = f;
                else if (f > this.curFrame && next === null) next = f;
            }
            const onionColors = [
                [prev, "rgba(249,226,175,0.35)"],
                [next, "rgba(137,180,250,0.35)"],
            ];
            for (const [f, tint] of onionColors) {
                if (f == null) continue;
                const id = this.keyframes.get(f);
                if (!id) continue;
                // Render via a temporary canvas tinted with composite.
                const tmp = document.createElement("canvas");
                tmp.width = id.width; tmp.height = id.height;
                tmp.getContext("2d").putImageData(id, 0, 0);
                ctx.globalAlpha = 0.45;
                ctx.drawImage(tmp, 0, 0);
                ctx.globalAlpha = 1.0;
            }
        }

        // Mask overlay.
        if (this.curMask) {
            const tmp = document.createElement("canvas");
            tmp.width = this.curMask.width; tmp.height = this.curMask.height;
            tmp.getContext("2d").putImageData(this.curMask, 0, 0);
            ctx.globalAlpha = 0.65;
            ctx.drawImage(tmp, 0, 0);
            ctx.globalAlpha = 1.0;
        }

        // Frame border.
        ctx.strokeStyle = C.border;
        ctx.lineWidth = 2 / this.zoom;
        ctx.strokeRect(0, 0, this.frameW, this.frameH);

        // Lasso preview.
        if (this.tool === "lasso" && this.lasso.length) {
            ctx.strokeStyle = C.accent2;
            ctx.fillStyle = C.accent2 + "30";
            ctx.lineWidth = 2 / this.zoom;
            ctx.setLineDash([6 / this.zoom, 4 / this.zoom]);
            ctx.beginPath();
            ctx.moveTo(this.lasso[0].x, this.lasso[0].y);
            for (let i = 1; i < this.lasso.length; i++) ctx.lineTo(this.lasso[i].x, this.lasso[i].y);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        ctx.restore();

        // Brush cursor outline.
        if ((this.tool === "brush" || this.tool === "erase") && this._cursor) {
            ctx.strokeStyle = this.tool === "erase" ? C.danger : C.accent;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.arc(this._cursor.vx, this._cursor.vy,
                    this.brushRadius * this.zoom, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Update status / sidebar.
        this._updateStatus();
        this._updateSidebar();
        this._drawScrubber();
    }

    _updateStatus() {
        const pinned = this.keyframes.has(this.curFrame) || this.curDirty;
        this.dom.status.textContent =
            `${this.frameW}×${this.frameH} · frame ${this.curFrame + 1}/${this.frameCount} · ` +
            `${pinned ? "📌 keyframe" : "blank"} · ${this.keyframes.size} pinned` +
            (this.curDirty ? " · ●" : "");
    }

    _updateSidebar() {
        const list = this.dom.kfList;
        list.innerHTML = "";
        const sorted = [...this.keyframes.keys()].sort((a, b) => a - b);
        for (const f of sorted) {
            const row = document.createElement("div");
            row.style.cssText = `
                display:flex;align-items:center;gap:6px;padding:4px 8px;
                border-radius:4px;background:${f === this.curFrame ? C.accent + "22" : "transparent"};
                cursor:pointer;font-size:11px;border:1px solid transparent;
                border-color:${f === this.curFrame ? C.accent + "55" : "transparent"};
            `;
            row.onmouseenter = () => { if (f !== this.curFrame) row.style.background = "#222234"; };
            row.onmouseleave = () => { row.style.background = f === this.curFrame ? C.accent + "22" : "transparent"; };
            const dot = document.createElement("span");
            dot.textContent = "📌";
            dot.style.fontSize = "10px";
            row.appendChild(dot);
            const lbl = document.createElement("span");
            lbl.textContent = `frame ${f}`;
            lbl.style.flex = "1";
            lbl.style.color = C.text;
            lbl.onclick = () => this._switchFrame(f);
            row.appendChild(lbl);
            const del = document.createElement("button");
            del.textContent = "✕";
            del.title = "Delete this keyframe";
            del.style.cssText = `
                width:18px;height:18px;border:none;border-radius:3px;
                background:transparent;color:${C.danger};cursor:pointer;font-size:11px;
            `;
            del.onmouseenter = () => { del.style.background = C.danger + "33"; };
            del.onmouseleave = () => { del.style.background = "transparent"; };
            del.onclick = (e) => {
                e.stopPropagation();
                this.keyframes.delete(f);
                if (f === this.curFrame) {
                    this.curMask = this._emptyMask();
                    this.curDirty = false;
                }
                this.draw();
            };
            row.appendChild(del);
            list.appendChild(row);
        }
        if (!sorted.length) {
            const empty = document.createElement("div");
            empty.textContent = "no keyframes yet — paint then Pin";
            empty.style.cssText = `color:${C.sub};font-size:11px;padding:6px;font-style:italic;`;
            list.appendChild(empty);
        }
    }

    _drawScrubber() {
        const c = this.dom.scrub;
        const r = c.getBoundingClientRect();
        const W = Math.max(1, Math.floor(r.width));
        const H = Math.max(1, Math.floor(r.height));
        if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
        const ctx = c.getContext("2d");
        ctx.fillStyle = "#11111b";
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = C.border;
        ctx.fillRect(8, H / 2 - 2, W - 16, 4);
        const n = Math.max(1, this.frameCount);
        // Frame ticks.
        if (n <= 200) {
            ctx.fillStyle = "#45475a";
            for (let i = 0; i < n; i++) {
                const x = 8 + (W - 16) * (i / Math.max(1, n - 1));
                ctx.fillRect(x - 0.5, H / 2 - 4, 1, 8);
            }
        }
        // Keyframe pins.
        for (const f of this.keyframes.keys()) {
            const x = 8 + (W - 16) * (f / Math.max(1, n - 1));
            ctx.fillStyle = C.accent;
            ctx.beginPath();
            ctx.moveTo(x, 2); ctx.lineTo(x - 4, 10); ctx.lineTo(x + 4, 10);
            ctx.closePath(); ctx.fill();
        }
        // Current frame indicator.
        const cx = 8 + (W - 16) * (this.curFrame / Math.max(1, n - 1));
        ctx.strokeStyle = C.warn;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(cx, 2); ctx.lineTo(cx, H - 2); ctx.stroke();
        ctx.fillStyle = C.warn;
        ctx.beginPath(); ctx.arc(cx, H / 2, 5, 0, Math.PI * 2); ctx.fill();
    }

    // ── Pin / Unpin ──────────────────────────────────────────────────
    pin() {
        if (!this.curMask) return;
        // Empty mask = nothing to pin.
        let any = false;
        const d = this.curMask.data;
        for (let i = 3; i < d.length; i += 4) {
            if (d[i] > 0) { any = true; break; }
        }
        if (!any) {
            alert("Nothing to pin — paint a mask first.");
            return;
        }
        this.keyframes.set(this.curFrame,
            new ImageData(new Uint8ClampedArray(this.curMask.data),
                          this.curMask.width, this.curMask.height));
        this.curDirty = false;
        this.draw();
    }
    unpin() {
        this.keyframes.delete(this.curFrame);
        this.curMask = this._emptyMask();
        this.curDirty = false;
        this.draw();
    }

    // ── Save to server ───────────────────────────────────────────────
    async save() {
        this._commitCurrent();
        // Init (idempotent) just in case shape changed.
        await fetch(`/mec/video_mask_editor/init?session=${encodeURIComponent(this.sessionId)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ h: this.frameH, w: this.frameW }),
        });
        // Clear and re-POST every keyframe (simple & correct).
        await fetch(`/mec/video_mask_editor/clear?session=${encodeURIComponent(this.sessionId)}`, {
            method: "POST",
        });
        let saved = 0;
        const sorted = [...this.keyframes.entries()].sort((a, b) => a[0] - b[0]);
        for (const [f, id] of sorted) {
            const blob = await this._maskToPngBlob(id);
            const r = await fetch(`/mec/video_mask_editor/keyframe?session=${encodeURIComponent(this.sessionId)}&frame=${f}`, {
                method: "POST",
                headers: { "Content-Type": "image/png" },
                body: blob,
            });
            if (r.ok) saved++;
        }
        return saved;
    }

    async _maskToPngBlob(id) {
        // Reduce RGBA → grayscale L (use alpha channel as luma) for compact PNG.
        const W = id.width, H = id.height;
        const c = document.createElement("canvas");
        c.width = W; c.height = H;
        const cx = c.getContext("2d");
        const out = cx.createImageData(W, H);
        for (let i = 0; i < id.data.length; i += 4) {
            const a = id.data[i + 3];
            out.data[i] = out.data[i + 1] = out.data[i + 2] = a;
            out.data[i + 3] = 255;
        }
        cx.putImageData(out, 0, 0);
        return new Promise(resolve => c.toBlob(resolve, "image/png"));
    }
}


// ─── Modal mount ────────────────────────────────────────────────────
function openModal(node) {
    const ed = new VMEEditor(node);

    const overlay = document.createElement("div");
    overlay.style.cssText = `
        position:fixed;inset:0;z-index:10000;background:${C.bg}f0;
        display:flex;flex-direction:column;color:${C.text};
        font-family:Inter,system-ui,sans-serif;
    `;
    document.body.appendChild(overlay);

    // ── Top bar ────────────────────────────────────────────────────
    const top = document.createElement("div");
    top.style.cssText = `
        display:flex;align-items:center;gap:8px;padding:8px 14px;
        background:linear-gradient(${C.panel},#101018);
        border-bottom:1px solid ${C.border};flex:0 0 auto;flex-wrap:wrap;
    `;
    overlay.appendChild(top);

    const title = document.createElement("div");
    title.innerHTML = `<span style="color:${C.accent};font-weight:600">🎨 Video Mask Editor</span>` +
                      `<span style="color:${C.sub};font-size:11px;margin-left:10px;">session: <code style="color:${C.text}">${ed.sessionId.slice(0,8)}…</code></span>`;
    top.appendChild(title);

    const spacer = document.createElement("div");
    spacer.style.flex = "1";
    top.appendChild(spacer);

    const mkBtn = (label, opts = {}) => {
        const b = document.createElement("button");
        b.innerHTML = label;
        b.title = opts.title || "";
        b.style.cssText = `
            min-width:32px;height:30px;padding:0 12px;
            background:${opts.bg || C.panel};color:${opts.fg || C.text};
            border:1px solid ${opts.border || C.border};border-radius:5px;
            cursor:pointer;font-size:12px;font-weight:500;line-height:1;
            display:inline-flex;align-items:center;justify-content:center;gap:6px;
            transition:background .12s,border-color .12s,transform .05s;
        `;
        b.onmouseenter = () => { b.style.background = opts.hover || "#2a2a3e"; };
        b.onmouseleave = () => { b.style.background = opts.bg || C.panel; };
        b.onmousedown = () => { b.style.transform = "scale(0.97)"; };
        b.onmouseup = () => { b.style.transform = "scale(1)"; };
        return b;
    };

    // ── Tool buttons ───────────────────────────────────────────────
    const tools = [
        { id: "brush",  label: "🖌 Brush",  hot: "B" },
        { id: "erase",  label: "🧽 Erase",  hot: "E" },
        { id: "fill",   label: "🪣 Fill",   hot: "G" },
        { id: "lasso",  label: "✂ Lasso",  hot: "L" },
    ];
    const toolBtns = {};
    const setTool = (id) => {
        ed.tool = id;
        for (const k in toolBtns) {
            const sel = k === id;
            toolBtns[k].style.background = sel ? C.accent : C.panel;
            toolBtns[k].style.color = sel ? "#181825" : C.text;
            toolBtns[k].style.borderColor = sel ? C.accent : C.border;
        }
        ed.lasso = [];
        ed.draw();
    };
    for (const t of tools) {
        const b = mkBtn(t.label, { title: `${t.label} (${t.hot})` });
        b.onclick = () => setTool(t.id);
        toolBtns[t.id] = b;
        top.appendChild(b);
    }

    // ── Brush controls ─────────────────────────────────────────────
    const sliderWrap = (lbl, min, max, val, step, onChange, suffix = "") => {
        const wrap = document.createElement("div");
        wrap.style.cssText = `display:flex;align-items:center;gap:6px;padding:0 4px;font-size:11px;color:${C.sub};`;
        const ttl = document.createElement("span");
        ttl.textContent = lbl;
        ttl.style.minWidth = "44px";
        wrap.appendChild(ttl);
        const inp = document.createElement("input");
        inp.type = "range";
        inp.min = min; inp.max = max; inp.value = val; inp.step = step;
        inp.style.cssText = `width:80px;accent-color:${C.accent};`;
        const out = document.createElement("span");
        out.textContent = val + suffix;
        out.style.cssText = `min-width:36px;color:${C.text};font-family:ui-monospace,monospace;font-size:10px;`;
        inp.oninput = () => {
            const v = +inp.value;
            out.textContent = v + suffix;
            onChange(v);
        };
        wrap.appendChild(inp);
        wrap.appendChild(out);
        return wrap;
    };
    top.appendChild(sliderWrap("Size", 2, 200, ed.brushRadius, 1,
        v => { ed.brushRadius = v; ed.draw(); }, "px"));
    top.appendChild(sliderWrap("Opacity", 5, 100, Math.round(ed.brushOpacity * 100), 1,
        v => { ed.brushOpacity = v / 100; }, "%"));
    top.appendChild(sliderWrap("Feather", 0, 100, Math.round(ed.brushFeather * 100), 1,
        v => { ed.brushFeather = v / 100; }, "%"));

    const sep = () => { const d = document.createElement("div"); d.style.cssText = `width:1px;height:22px;background:${C.border};margin:0 4px;`; return d; };
    top.appendChild(sep());

    const btnUndo = mkBtn("↶", { title: "Undo (Ctrl+Z)" });
    btnUndo.onclick = () => ed.undo();
    top.appendChild(btnUndo);
    const btnRedo = mkBtn("↷", { title: "Redo (Ctrl+Y)" });
    btnRedo.onclick = () => ed.redoOp();
    top.appendChild(btnRedo);

    const btnOnion = mkBtn("🧅 Onion", { title: "Toggle onion-skin (O)", bg: ed.onion ? C.accent + "44" : C.panel });
    btnOnion.onclick = () => {
        ed.onion = !ed.onion;
        btnOnion.style.background = ed.onion ? C.accent + "44" : C.panel;
        ed.draw();
    };
    top.appendChild(btnOnion);

    const btnFit = mkBtn("⬛ Fit", { title: "Fit to view (F)" });
    btnFit.onclick = () => { ed.fitView(); ed.draw(); };
    top.appendChild(btnFit);

    top.appendChild(sep());

    const btnPin = mkBtn("📌 Pin Keyframe", { title: "Pin current mask (K)", bg: "#2d4a3e", hover: "#3a5f50", fg: C.accent });
    btnPin.onclick = () => ed.pin();
    top.appendChild(btnPin);

    const btnUnpin = mkBtn("✕ Unpin", { title: "Remove keyframe", bg: "#4a2d2d", hover: "#5f3a3a", fg: C.danger });
    btnUnpin.onclick = () => ed.unpin();
    top.appendChild(btnUnpin);

    top.appendChild(sep());

    const btnSave = mkBtn("💾 Save & Close", { title: "Persist keyframes to server & close",
        bg: C.accent, fg: "#0f0f17", border: C.accent, hover: "#7fd17a" });
    top.appendChild(btnSave);

    const btnCancel = mkBtn("✕ Cancel", { title: "Discard changes & close",
        bg: C.panel, hover: "#332233", fg: C.danger });
    top.appendChild(btnCancel);

    // ── Body: canvas + sidebar ─────────────────────────────────────
    const body = document.createElement("div");
    body.style.cssText = `display:flex;flex:1;min-height:0;`;
    overlay.appendChild(body);

    const viewport = document.createElement("div");
    viewport.style.cssText = `flex:1;position:relative;background:${C.bg};overflow:hidden;cursor:crosshair;`;
    body.appendChild(viewport);

    const canvas = document.createElement("canvas");
    canvas.style.cssText = `display:block;width:100%;height:100%;`;
    canvas.tabIndex = 0;
    viewport.appendChild(canvas);

    const status = document.createElement("div");
    status.style.cssText = `
        position:absolute;left:10px;bottom:48px;padding:4px 10px;
        background:${C.panel}d8;border:1px solid ${C.border};border-radius:5px;
        font-size:11px;color:${C.sub};font-family:ui-monospace,monospace;
        pointer-events:none;letter-spacing:.2px;
    `;
    viewport.appendChild(status);

    const scrubCanvas = document.createElement("canvas");
    scrubCanvas.style.cssText = `
        position:absolute;left:0;right:0;bottom:0;height:36px;width:100%;
        background:#11111b;border-top:1px solid ${C.border};cursor:pointer;
    `;
    viewport.appendChild(scrubCanvas);

    // Sidebar
    const sidebar = document.createElement("div");
    sidebar.style.cssText = `
        width:240px;background:${C.panel};border-left:1px solid ${C.border};
        display:flex;flex-direction:column;flex:0 0 240px;
    `;
    body.appendChild(sidebar);
    const sbHead = document.createElement("div");
    sbHead.textContent = "KEYFRAMES";
    sbHead.style.cssText = `
        padding:10px 12px;font-size:10px;font-weight:600;letter-spacing:1px;
        color:${C.sub};border-bottom:1px solid ${C.border};
    `;
    sidebar.appendChild(sbHead);
    const kfList = document.createElement("div");
    kfList.style.cssText = `flex:1;overflow-y:auto;padding:6px;display:flex;flex-direction:column;gap:2px;`;
    sidebar.appendChild(kfList);

    const sbHelp = document.createElement("div");
    sbHelp.style.cssText = `
        padding:10px 12px;font-size:10px;color:${C.sub};line-height:1.6;
        border-top:1px solid ${C.border};font-family:ui-monospace,monospace;
    `;
    sbHelp.innerHTML = `
        <div style="color:${C.text};font-weight:600;margin-bottom:6px;">SHORTCUTS</div>
        <div>B / E / G / L · tools</div>
        <div>[ ] · brush size</div>
        <div>← / → · step frame</div>
        <div>K · pin · O · onion · F · fit</div>
        <div>Ctrl+Z / Y · undo / redo</div>
        <div>Mouse wheel · zoom</div>
        <div>Middle / Space-drag · pan</div>
    `;
    sidebar.appendChild(sbHelp);

    // ── Hook up editor DOM refs ────────────────────────────────────
    ed.dom = { viewport, canvas, status, scrub: scrubCanvas, kfList };

    // ── Pointer handling ───────────────────────────────────────────
    let panning = false; let panStart = null;
    let spaceDown = false;
    canvas.addEventListener("mousedown", (e) => {
        e.preventDefault();
        canvas.focus();
        const r = canvas.getBoundingClientRect();
        const vx = e.clientX - r.left, vy = e.clientY - r.top;
        if (e.button === 1 || (e.button === 0 && spaceDown)) {
            panning = true; panStart = { vx, vy, px: ed.panX, py: ed.panY };
            return;
        }
        if (e.button !== 0) return;
        const fp = ed._viewToFrame(vx, vy);
        if (ed.tool === "fill") {
            ed._pushUndo();
            ed._floodFill(fp.x, fp.y);
            ed.curDirty = true;
            ed.draw();
            return;
        }
        if (ed.tool === "lasso") {
            ed.lasso.push(fp);
            ed.draw();
            return;
        }
        ed._pushUndo();
        ed.painting = true;
        ed._stamp(fp.x, fp.y, null);
        ed.lastPaint = fp;
        ed.curDirty = true;
        ed.draw();
    });

    canvas.addEventListener("mousemove", (e) => {
        const r = canvas.getBoundingClientRect();
        const vx = e.clientX - r.left, vy = e.clientY - r.top;
        ed._cursor = { vx, vy };
        if (panning && panStart) {
            ed.panX = panStart.px + (vx - panStart.vx);
            ed.panY = panStart.py + (vy - panStart.vy);
            ed.draw();
            return;
        }
        if (ed.painting) {
            const fp = ed._viewToFrame(vx, vy);
            ed._stamp(fp.x, fp.y, ed.lastPaint);
            ed.lastPaint = fp;
            ed.draw();
            return;
        }
        if (ed.tool === "brush" || ed.tool === "erase") {
            ed.draw();
        }
    });

    canvas.addEventListener("mouseup", (e) => {
        if (panning) { panning = false; panStart = null; }
        if (ed.painting) { ed.painting = false; ed.lastPaint = null; }
    });

    canvas.addEventListener("dblclick", (e) => {
        if (ed.tool === "lasso" && ed.lasso.length >= 3) {
            ed._pushUndo();
            ed._fillLasso();
            ed.lasso = [];
            ed.curDirty = true;
            ed.draw();
        }
    });

    canvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        if (ed.tool === "lasso" && ed.lasso.length >= 3) {
            ed._pushUndo();
            ed._fillLasso();
            ed.lasso = [];
            ed.curDirty = true;
            ed.draw();
        }
    });

    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const r = canvas.getBoundingClientRect();
        const ax = e.clientX - r.left, ay = e.clientY - r.top;
        const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        const newZ = Math.max(0.05, Math.min(16, ed.zoom * factor));
        const k = newZ / ed.zoom;
        ed.panX = ax - (ax - ed.panX) * k;
        ed.panY = ay - (ay - ed.panY) * k;
        ed.zoom = newZ;
        ed.draw();
    }, { passive: false });

    // Keyboard.
    const onKey = (e) => {
        if (e.key === " ") { spaceDown = (e.type === "keydown"); e.preventDefault(); return; }
        if (e.type !== "keydown") return;
        if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
        if (e.key === "Escape") { close(false); e.preventDefault(); return; }
        if (e.key === "b" || e.key === "B") { setTool("brush"); e.preventDefault(); return; }
        if (e.key === "e" || e.key === "E") { setTool("erase"); e.preventDefault(); return; }
        if (e.key === "g" || e.key === "G") { setTool("fill"); e.preventDefault(); return; }
        if (e.key === "l" || e.key === "L") { setTool("lasso"); e.preventDefault(); return; }
        if (e.key === "k" || e.key === "K") { ed.pin(); e.preventDefault(); return; }
        if (e.key === "o" || e.key === "O") { btnOnion.click(); e.preventDefault(); return; }
        // NOTE: bare F is reserved by KJNodes.fillConnectSelected. Use Shift+F here.
        if ((e.key === "F" || e.key === "f") && e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) {
            ed.fitView(); ed.draw(); e.preventDefault(); e.stopImmediatePropagation(); return;
        }
        if (e.key === "[") { ed.brushRadius = Math.max(2, ed.brushRadius - 4); ed.draw(); e.preventDefault(); return; }
        if (e.key === "]") { ed.brushRadius = Math.min(200, ed.brushRadius + 4); ed.draw(); e.preventDefault(); return; }
        if (e.key === "ArrowLeft") { ed._switchFrame(ed.curFrame - 1); e.preventDefault(); return; }
        if (e.key === "ArrowRight") { ed._switchFrame(ed.curFrame + 1); e.preventDefault(); return; }
        if ((e.key === "z" || e.key === "Z") && (e.ctrlKey || e.metaKey) && !e.shiftKey) { ed.undo(); e.preventDefault(); return; }
        if ((e.key === "y" || e.key === "Y") && (e.ctrlKey || e.metaKey)) { ed.redoOp(); e.preventDefault(); return; }
        if (e.key === "Z" && (e.ctrlKey || e.metaKey) && e.shiftKey) { ed.redoOp(); e.preventDefault(); return; }
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);

    // Scrubber.
    let scrubbing = false;
    const scrubXToFrame = (clientX) => {
        const r = scrubCanvas.getBoundingClientRect();
        const t = (clientX - r.left - 8) / Math.max(1, r.width - 16);
        const n = Math.max(1, ed.frameCount);
        return Math.max(0, Math.min(n - 1, Math.round(t * (n - 1))));
    };
    scrubCanvas.addEventListener("mousedown", (e) => {
        scrubbing = true;
        ed._switchFrame(scrubXToFrame(e.clientX));
    });
    window.addEventListener("mousemove", (e) => {
        if (scrubbing) ed._switchFrame(scrubXToFrame(e.clientX));
    });
    window.addEventListener("mouseup", () => { scrubbing = false; });

    // Resize observer.
    const ro = new ResizeObserver(() => ed.draw());
    ro.observe(viewport);

    // ── Save / Cancel handlers ─────────────────────────────────────
    let closing = false;
    const close = async (commit) => {
        if (closing) return;
        closing = true;
        window.removeEventListener("keydown", onKey);
        window.removeEventListener("keyup", onKey);
        ro.disconnect();
        if (commit) {
            btnSave.disabled = true;
            btnSave.textContent = "saving…";
            try {
                const n = await ed.save();
                console.log(`[VME] saved ${n} keyframes for session ${ed.sessionId}`);
            } catch (e) {
                console.error("[VME] save failed:", e);
                alert("Save failed:\n" + (e?.message || e));
                closing = false;
                btnSave.disabled = false;
                btnSave.innerHTML = "💾 Save & Close";
                return;
            }
            // Trigger a redraw of the node so the session_id field shows.
            ed.node.graph?.setDirtyCanvas(true, true);
        }
        document.body.removeChild(overlay);
    };
    btnSave.onclick = () => close(true);
    btnCancel.onclick = () => close(false);

    // ── Init ───────────────────────────────────────────────────────
    setTool("brush");
    ed.loadFrames().then(ok => {
        if (!ok) { close(false); return; }
        ed.fitView();
        ed.draw();
        canvas.focus();
    });
}


// ─── Extension registration ─────────────────────────────────────────
app.registerExtension({
    name: "MEC.VideoMaskEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            const self = this;
            // Ensure session_id is populated.
            setTimeout(() => {
                const w = self.widgets?.find(w => w.name === "session_id");
                if (w && (!w.value || w.value.length < 8)) {
                    w.value = uuid();
                }
            }, 0);
            // Add an "Open Editor" big button widget at the top.
            this.addWidget(
                "button",
                "🎨 Open Video Mask Editor",
                null,
                () => openModal(self),
                { serialize: false }
            );
        };
    },
});
