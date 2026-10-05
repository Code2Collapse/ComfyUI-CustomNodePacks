/**
 * IMEEditor — mask editing state for one modal session.
 */
import { resolveEditorSource } from "./source.js";
import { C, hexToRgb } from "./palette.js";
import { fitView, zoomAtCursor } from "./coords.js";
import { UndoStack, captureRect, applyRect, pushUndoEntry } from "./undo.js";
import * as tools from "./tools.js";
import * as colour from "./colour.js";
import * as api from "./api.js";

const MP_LIMIT = 16_000_000;
const OVERLAY_ALPHA = 0.45;
const PREVIEW_ALPHA = 0.45;
const CLEAN_MASK_CAP = 8;

export class IMEEditor {
    constructor(node, editorId) {
        this.node = node;
        this.editorId = editorId;
        this.frameSource = null;
        this.frameCount = 0;
        this.nativeW = 0;
        this.nativeH = 0;
        this.editW = 0;
        this.editH = 0;
        this.editScale = 1;
        this.curFrame = 0;
        this.masks = new Map();
        this.storedFrames = new Set();
        this.dirtyFrames = new Set();
        this.cleanLru = [];
        this.imageBitmap = null;
        this.imageData = null;
        this.mask = null;
        this.overlayCanvas = null;
        this.overlayCtx = null;
        this.matteCanvas = null;
        this.matteCtx = null;
        this._overlayTint = hexToRgb(C.accent);
        this.undo = new UndoStack();
        this.tool = "brush";
        this.brushSize = 24;
        this.brushHardness = 0.6;
        this.brushOpacity = 1;
        this.brushSpacing = 0.15;      // stamp distance as a fraction of the brush DIAMETER
        this.hover = null;             // pointer position in image space, for the brush outline
        this.bucketTolerance = 32;
        this.selectionMode = "add";
        this.subtract = false;
        this.colourSamples = [];
        this.colourSamplePixels = [];
        this.colourSpace = "rgb";
        this.colourTolerance = 32;
        this.colourSoftness = 0;
        this.colourContiguous = false;
        this.colourDist = null;
        this.colourDistKey = "";
        this.colourPreview = null;
        this.previewCanvas = null;
        this.previewCtx = null;
        this._previewTint = hexToRgb(C.accent2);
        this.viewMode = 0;
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.painting = false;
        this.shapeStart = null;
        this.shapeCur = null;
        this.polyPts = [];
        this.lassoPts = [];
        this.lastPt = null;
        this.strokeStart = null;
        this.coverage = null;
        this._strokeBounds = null;
        this._strokeCommitted = false;
        this.spacePan = false;
        this.panning = false;
        this.panAnchor = null;
        this.digest = "";
        this._urls = [];
        this._raf = 0;
        this.dom = {};
        this.onChange = null;
        this.onToolChange = null;
    }

    _releaseBitmap() {
        try { this.imageBitmap?.close?.(); } catch (_) { /* ignore */ }
        this.imageBitmap = null;
        this.imageData = null;
        for (const u of this._urls) {
            try { URL.revokeObjectURL(u); } catch (_) { /* ignore */ }
        }
        this._urls = [];
    }

    _ensureBuffers() {
        if (!this.overlayCanvas || this.overlayCanvas.width !== this.editW) {
            this.overlayCanvas = document.createElement("canvas");
            this.overlayCanvas.width = this.editW;
            this.overlayCanvas.height = this.editH;
            this.overlayCtx = this.overlayCanvas.getContext("2d");
            this.previewCanvas = document.createElement("canvas");
            this.previewCanvas.width = this.editW;
            this.previewCanvas.height = this.editH;
            this.previewCtx = this.previewCanvas.getContext("2d");
            this.matteCanvas = document.createElement("canvas");
            this.matteCanvas.width = this.editW;
            this.matteCanvas.height = this.editH;
            this.matteCtx = this.matteCanvas.getContext("2d");
        }
    }

    _syncOverlayRect(x, y, w, h) {
        if (!this.mask || !this.overlayCtx) return;
        const x2 = Math.min(this.editW, x + w);
        const y2 = Math.min(this.editH, y + h);
        const rw = x2 - x;
        const rh = y2 - y;
        if (rw <= 0 || rh <= 0) return;
        const id = this.overlayCtx.createImageData(rw, rh);
        const ac = this._overlayTint;
        for (let yy = 0; yy < rh; yy++) {
            for (let xx = 0; xx < rw; xx++) {
                const pi = (y + yy) * this.editW + (x + xx);
                const a = this.mask[pi];
                const oi = (yy * rw + xx) * 4;
                id.data[oi] = ac.r;
                id.data[oi + 1] = ac.g;
                id.data[oi + 2] = ac.b;
                id.data[oi + 3] = Math.round(a * OVERLAY_ALPHA);
            }
        }
        this.overlayCtx.putImageData(id, x, y);
        if (this.viewMode === 1) this._syncMatteRect(x, y, rw, rh);
    }

    _syncMatteRect(x, y, w, h) {
        if (!this.mask || !this.matteCtx) return;
        const id = this.matteCtx.createImageData(w, h);
        for (let yy = 0; yy < h; yy++) {
            for (let xx = 0; xx < w; xx++) {
                const pi = (y + yy) * this.editW + (x + xx);
                const g = this.mask[pi];
                const oi = (yy * w + xx) * 4;
                id.data[oi] = id.data[oi + 1] = id.data[oi + 2] = g;
                id.data[oi + 3] = 255;
            }
        }
        this.matteCtx.putImageData(id, x, y);
    }

    _rebuildOverlayFull() {
        if (!this.mask) return;
        this._ensureBuffers();
        this._syncOverlayRect(0, 0, this.editW, this.editH);
    }

    _syncPreviewRect(x, y, w, h) {
        if (!this.colourPreview || !this.previewCtx) return;
        const x2 = Math.min(this.editW, x + w);
        const y2 = Math.min(this.editH, y + h);
        const rw = x2 - x;
        const rh = y2 - y;
        if (rw <= 0 || rh <= 0) return;
        const id = this.previewCtx.createImageData(rw, rh);
        const ac = this._previewTint;
        for (let yy = 0; yy < rh; yy++) {
            for (let xx = 0; xx < rw; xx++) {
                const pi = (y + yy) * this.editW + (x + xx);
                const a = this.colourPreview[pi];
                const oi = (yy * rw + xx) * 4;
                id.data[oi] = ac.r;
                id.data[oi + 1] = ac.g;
                id.data[oi + 2] = ac.b;
                id.data[oi + 3] = Math.round(a * PREVIEW_ALPHA);
            }
        }
        this.previewCtx.putImageData(id, x, y);
    }

    _rebuildPreviewFull() {
        if (!this.colourPreview) return;
        this._ensureBuffers();
        this.previewCtx.clearRect(0, 0, this.editW, this.editH);
        this._syncPreviewRect(0, 0, this.editW, this.editH);
    }

    _rebuildMatteFull() {
        if (!this.mask || this.viewMode !== 1) return;
        this._ensureBuffers();
        this.matteCtx.clearRect(0, 0, this.editW, this.editH);
        this._syncMatteRect(0, 0, this.editW, this.editH);
    }

    _maskDirtyRect(x0, y0, x1, y1) {
        const x = Math.max(0, x0 | 0);
        const y = Math.max(0, y0 | 0);
        const x2 = Math.min(this.editW, (x1 | 0) + 1);
        const y2 = Math.min(this.editH, (y1 | 0) + 1);
        this._syncOverlayRect(x, y, x2 - x, y2 - y);
    }

    requestDraw() {
        if (this._raf) return;
        this._raf = requestAnimationFrame(() => {
            this._raf = 0;
            this._drawNow();
        });
    }

    draw() {
        this.requestDraw();
    }

    maskCount() {
        const s = new Set(this.storedFrames);
        for (const f of this.dirtyFrames) s.add(f);
        return s.size;
    }

    frameHasMask(i) {
        const f = i | 0;
        return this.storedFrames.has(f) || this.dirtyFrames.has(f);
    }

    _touchCleanLru(idx) {
        idx |= 0;
        const p = this.cleanLru.indexOf(idx);
        if (p >= 0) this.cleanLru.splice(p, 1);
        this.cleanLru.push(idx);
    }

    _removeCleanLru(idx) {
        const p = this.cleanLru.indexOf(idx);
        if (p >= 0) this.cleanLru.splice(p, 1);
    }

    _trimCleanCache() {
        while (this.cleanLru.length > 0) {
            let cleanN = 0;
            for (const f of this.masks.keys()) {
                if (!this.dirtyFrames.has(f)) cleanN++;
            }
            if (cleanN <= CLEAN_MASK_CAP) break;
            let victim = null;
            for (const f of this.cleanLru) {
                if (!this.dirtyFrames.has(f)) {
                    victim = f;
                    break;
                }
            }
            if (victim == null) break;
            this.masks.delete(victim);
            this._removeCleanLru(victim);
        }
    }

    _evictLeavingFrame(idx) {
        if (this.dirtyFrames.has(idx)) {
            this.masks.set(idx, this.mask);
            return;
        }
        if (!this.storedFrames.has(idx)) {
            this.masks.delete(idx);
            this._removeCleanLru(idx);
            return;
        }
        if (this.masks.has(idx)) {
            this._touchCleanLru(idx);
            this._trimCleanCache();
        }
    }

    async loadFrames() {
        const source = await resolveEditorSource(this.node);
        if (!source.count) {
            return { ok: false, message: "Connect an image source and queue once, or wire a C2C/VHS loader directly." };
        }
        this.frameSource = source;
        this.frameCount = source.count;
        this.nativeW = source.width || 0;
        this.nativeH = source.height || 0;
        if (!this.nativeW || !this.nativeH) {
            const url = source.url(0);
            const im = await this._loadUrl(url);
            this.nativeW = im.naturalWidth || im.width;
            this.nativeH = im.naturalHeight || im.height;
        }
        const pixels = this.nativeW * this.nativeH;
        if (pixels > MP_LIMIT) {
            this.editScale = Math.sqrt(MP_LIMIT / pixels);
        } else {
            this.editScale = 1;
        }
        this.editW = Math.max(1, Math.round(this.nativeW * this.editScale));
        this.editH = Math.max(1, Math.round(this.nativeH * this.editScale));
        this._ensureBuffers();
        try {
            const st = await api.fetchState(this.editorId);
            this.digest = st.digest || "";
            this.storedFrames = new Set((st.frames || []).map((f) => f | 0));
        } catch (e) {
            console.warn("[IME] state load:", e);
            this.storedFrames = new Set();
        }
        await this._switchFrame(0, true);
        return { ok: true, banner: pixels > MP_LIMIT };
    }

    async _loadUrl(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });
    }

    frameUrl(idx, native = true) {
        const src = this.frameSource;
        if (!src) return "";
        if (src.kind === "plan" && native) {
            const tok = src.url(0).match(/token=([^&]+)/);
            if (tok) {
                return `/c2c/frames/thumb?token=${encodeURIComponent(tok[1])}&i=${idx | 0}&max=0&fmt=png`;
            }
        }
        return src.url(idx);
    }

    async _loadMaskFrame(f) {
        const blob = await api.fetchFramePng(this.editorId, f);
        if (!blob) return;
        const bm = await createImageBitmap(blob);
        const c = document.createElement("canvas");
        c.width = this.editW; c.height = this.editH;
        const cx = c.getContext("2d");
        cx.drawImage(bm, 0, 0, this.editW, this.editH);
        bm.close();
        const id = cx.getImageData(0, 0, this.editW, this.editH);
        const buf = new Uint8Array(this.editW * this.editH);
        for (let i = 0, p = 0; p < buf.length; i += 4, p++) buf[p] = id.data[i];
        const fi = f | 0;
        this.masks.set(fi, buf);
        this._touchCleanLru(fi);
    }

    _emptyMask() {
        return new Uint8Array(this.editW * this.editH);
    }

    async _switchFrame(idx, fresh = false) {
        idx = Math.max(0, Math.min(this.frameCount - 1, idx | 0));
        if (!fresh) {
            this._commitFrame();
            this._commitBrushStroke();
            this.clearColourPreview();
            this._evictLeavingFrame(this.curFrame);
        }
        this.curFrame = idx;
        if (this.storedFrames.has(idx) && !this.masks.has(idx)) {
            await this._loadMaskFrame(idx);
        }
        this.mask = this.masks.get(idx) || this._emptyMask();
        this.undo = new UndoStack();
        await this._loadImageFrame(idx);
        this._rebuildOverlayFull();
        if (this.viewMode === 1) this._rebuildMatteFull();
        this._notify();
        this.requestDraw();
    }

    _commitFrame() {
        if (this.dirtyFrames.has(this.curFrame)) {
            this.masks.set(this.curFrame, this.mask);
        }
    }

    async _loadImageFrame(idx) {
        this._releaseBitmap();
        const url = this.frameUrl(idx, true);
        try {
            const r = await fetch(url);
            const blob = await r.blob();
            this.imageBitmap = await createImageBitmap(blob);
            const c = document.createElement("canvas");
            c.width = this.editW; c.height = this.editH;
            const cx = c.getContext("2d");
            cx.drawImage(this.imageBitmap, 0, 0, this.editW, this.editH);
            this.imageData = cx.getImageData(0, 0, this.editW, this.editH).data;
        } catch (e) {
            console.warn("[IME] image load:", e);
        }
    }

    _notify() {
        this.onChange?.();
    }

    _commitUndoRect(x0, y0, x1, y1, beforeSrc) {
        const entry = pushUndoEntry(
            this.undo, this.mask, this.editW, this.editH,
            x0, y0, x1, y1, beforeSrc,
        );
        this.dirtyFrames.add(this.curFrame);
        this._maskDirtyRect(entry.x, entry.y, entry.x + entry.w - 1, entry.y + entry.h - 1);
        this._notify();
        return entry;
    }

    undoOp() {
        const e = this.undo.popUndo();
        if (!e) return;
        this.undo.pushRedo(e);
        applyRect(this.mask, this.editW, { x: e.x, y: e.y, w: e.w, h: e.h, data: e.before });
        this.dirtyFrames.add(this.curFrame);
        this._maskDirtyRect(e.x, e.y, e.x + e.w - 1, e.y + e.h - 1);
        this.requestDraw();
        this._notify();
    }

    redoOp() {
        const e = this.undo.popRedo();
        if (!e) return;
        this.undo.push(e);
        applyRect(this.mask, this.editW, { x: e.x, y: e.y, w: e.w, h: e.h, data: e.after });
        this.dirtyFrames.add(this.curFrame);
        this._maskDirtyRect(e.x, e.y, e.x + e.w - 1, e.y + e.h - 1);
        this.requestDraw();
        this._notify();
    }

    async copyPrevFrame() {
        const prev = this.curFrame - 1;
        if (prev < 0) return;
        if (this.storedFrames.has(prev) && !this.masks.has(prev)) {
            await this._loadMaskFrame(prev);
        }
        const src = this.masks.get(prev);
        if (!src) return;
        const before = new Uint8Array(this.mask);
        this.mask.set(src);
        this._commitUndoRect(0, 0, this.editW - 1, this.editH - 1, before);
        this.requestDraw();
    }

    setTool(t) {
        this.cancelShapeInProgress();
        this.clearColourPreview();
        this.tool = t;
        this.onToolChange?.();
    }

    setSelectionMode(m) {
        this.selectionMode = m;
        this.requestDraw();
    }

    shapeInProgress() {
        return (this.tool === "polygon" && this.polyPts.length > 0) ||
            (this.tool === "lasso" && this.lassoPts.length > 0) ||
            ((this.tool === "rect" || this.tool === "ellipse") && this.shapeStart) ||
            this.colourPreviewActive();
    }

    cancelShapeInProgress() {
        this.polyPts = [];
        this.lassoPts = [];
        this.shapeStart = null;
        this.shapeCur = null;
        this.painting = false;
        this.requestDraw();
    }

    colourPreviewActive() {
        return this.colourPreview != null && this.colourSamples.length > 0;
    }

    clearColourPreview() {
        this.colourSamples = [];
        this.colourSamplePixels = [];
        this.colourDist = null;
        this.colourDistKey = "";
        this.colourPreview = null;
        if (this.previewCtx) {
            this.previewCtx.clearRect(0, 0, this.editW, this.editH);
        }
        this.requestDraw();
    }

    _colourDistKey() {
        return JSON.stringify(this.colourSamples) + "|" + this.colourSpace;
    }

    _ensureColourDist() {
        const key = this._colourDistKey();
        if (this.colourDist && this.colourDistKey === key) return;
        if (!this.imageData || !this.colourSamples.length) {
            this.colourDist = null;
            this.colourDistKey = key;
            return;
        }
        this.colourDist = colour.distanceMap(
            this.imageData, this.editW, this.editH,
            this.colourSamples, this.colourSpace,
        );
        this.colourDistKey = key;
    }

    _rebuildColourPreview() {
        if (!this.colourSamples.length) {
            this.colourPreview = null;
            if (this.previewCtx) this.previewCtx.clearRect(0, 0, this.editW, this.editH);
            this.requestDraw();
            return;
        }
        this._ensureColourDist();
        if (!this.colourDist) return;
        let sel = colour.selectionFromDistance(
            this.colourDist, this.colourTolerance, this.colourSoftness,
        );
        if (this.colourContiguous && this.colourSamplePixels.length) {
            sel = colour.keepConnected(sel, this.editW, this.editH, this.colourSamplePixels);
        }
        this.colourPreview = sel;
        this._rebuildPreviewFull();
        this.requestDraw();
    }

    addColourSample(ix, iy, shiftKey) {
        if (!this.imageData) return;
        const x = ix | 0, y = iy | 0;
        if (x < 0 || y < 0 || x >= this.editW || y >= this.editH) return;
        const i = (y * this.editW + x) * 4;
        const sample = [this.imageData[i], this.imageData[i + 1], this.imageData[i + 2]];
        if (shiftKey) {
            this.colourSamples.push(sample);
            this.colourSamplePixels.push({ x, y });
        } else {
            this.colourSamples = [sample];
            this.colourSamplePixels = [{ x, y }];
        }
        this._rebuildColourPreview();
    }

    applyColourPreview() {
        if (!this.colourPreviewActive()) return;
        const bounds = { x0: 0, y0: 0, x1: this.editW - 1, y1: this.editH - 1 };
        this._commitSelection(this.colourPreview, bounds, false);
        this.clearColourPreview();
    }

    _effectiveSelectionMode(altKey) {
        if (altKey) {
            if (this.selectionMode === "add") return "subtract";
            if (this.selectionMode === "subtract") return "add";
        }
        return this.selectionMode;
    }

    _commitSelection(sel, bounds, altKey) {
        const mode = this._effectiveSelectionMode(altKey);
        const ub = mode === "intersect"
            ? { x0: 0, y0: 0, x1: this.editW - 1, y1: this.editH - 1 }
            : bounds;
        const before = captureRect(this.mask, this.editW, this.editH, ub.x0, ub.y0, ub.x1, ub.y1);
        const rb = tools.composeSelection(this.mask, sel, this.editW, this.editH, mode, bounds);
        const after = captureRect(this.mask, this.editW, this.editH, rb.x0, rb.y0, rb.x1, rb.y1);
        this.undo.push({
            x: after.x, y: after.y, w: after.w, h: after.h,
            before: before.data, after: after.data,
        });
        this.dirtyFrames.add(this.curFrame);
        this._maskDirtyRect(rb.x0, rb.y0, rb.x1, rb.y1);
        this._notify();
    }

    _beginBrushStroke(ix, iy) {
        this.strokeStart = new Uint8Array(this.mask);
        this.coverage = new Uint8Array(this.mask.length);
        this._strokeBounds = null;
        this._strokeCommitted = false;
        this._stampAt(ix, iy);
    }

    _stampAt(cx, cy) {
        const erase = this.tool === "eraser" || this.subtract;
        const b = tools.stampBrushCoverage(
            this.mask, this.strokeStart, this.coverage,
            this.editW, this.editH, cx, cy,
            this.brushSize / 2, this.brushHardness, this.brushOpacity, erase,   // brushSize is the diameter
        );
        if (b) {
            this._strokeBounds = this._strokeBounds
                ? {
                    x0: Math.min(this._strokeBounds.x0, b.x0),
                    y0: Math.min(this._strokeBounds.y0, b.y0),
                    x1: Math.max(this._strokeBounds.x1, b.x1),
                    y1: Math.max(this._strokeBounds.y1, b.y1),
                }
                : b;
            this._maskDirtyRect(b.x0, b.y0, b.x1, b.y1);
        }
    }

    _commitBrushStroke() {
        if (this._strokeCommitted || !this.strokeStart || !this._strokeBounds) {
            this._clearBrushState();
            return;
        }
        const { x0, y0, x1, y1 } = this._strokeBounds;
        this._commitUndoRect(x0, y0, x1, y1, this.strokeStart);
        this._strokeCommitted = true;
        this._clearBrushState();
    }

    _clearBrushState() {
        this.strokeStart = null;
        this.coverage = null;
        this._strokeBounds = null;
        this.painting = false;
    }

    finishPointerStroke(ix, iy, e) {
        if (this.panning) {
            this.panning = false;
            this.panAnchor = null;
            return;
        }
        if (!this.painting && !this.shapeStart && !this.strokeStart) return;
        const sq = e?.shiftKey;
        const altKey = this.subtract;              // Alt decided at pointerdown, like the brush
        if (this.tool === "brush" || this.tool === "eraser") {
            this._commitBrushStroke();
        } else if (this.tool === "rect" && this.shapeStart) {
            const sel = new Uint8Array(this.editW * this.editH);
            const b = tools.fillRect(
                sel, this.editW, this.editH,
                this.shapeStart.x, this.shapeStart.y, ix, iy, 255, sq,
            );
            this._commitSelection(sel, b, altKey);
            this.shapeStart = null;
            this.shapeCur = null;
            this.painting = false;
        } else if (this.tool === "ellipse" && this.shapeStart) {
            const sel = new Uint8Array(this.editW * this.editH);
            const b = tools.fillEllipse(
                sel, this.editW, this.editH,
                this.shapeStart.x, this.shapeStart.y, ix, iy, 255, sq,
            );
            this._commitSelection(sel, b, altKey);
            this.shapeStart = null;
            this.shapeCur = null;
            this.painting = false;
        } else if (this.tool === "lasso" && this.lassoPts.length > 2) {
            const sel = new Uint8Array(this.editW * this.editH);
            const b = tools.fillPolygon(sel, this.editW, this.editH, this.lassoPts, 255);
            if (b) this._commitSelection(sel, b, altKey);
            this.lassoPts = [];
            this.painting = false;
        } else if (this.tool === "lasso") {
            this.lassoPts = [];
            this.painting = false;
        }
        this.requestDraw();
    }

    pointerDown(ix, iy, e) {
        if (this.spacePan || e.button === 1) {
            this.panning = true;
            this.panAnchor = { x: e.clientX - this.panX, y: e.clientY - this.panY };
            return;
        }
        if (this.tool === "colour") {
            this.addColourSample(ix, iy, e.shiftKey);
            return;
        }
        this.painting = true;
        this.lastPt = { x: ix, y: iy };
        if (this.tool === "brush" || this.tool === "eraser") {
            this._beginBrushStroke(ix, iy);
        } else if (this.tool === "rect" || this.tool === "ellipse") {
            this.shapeStart = { x: ix, y: iy };
            this.shapeCur = { x: ix, y: iy };
        } else if (this.tool === "polygon") {
            this.polyPts.push({ x: ix, y: iy });
        } else if (this.tool === "lasso") {
            this.lassoPts = [{ x: ix, y: iy }];
        } else if (this.tool === "bucket") {
            const sel = new Uint8Array(this.editW * this.editH);
            const b = tools.floodFillScratch(
                sel, this.imageData, this.editW, this.editH,
                ix | 0, iy | 0, this.bucketTolerance, 255,
            );
            if (b) this._commitSelection(sel, b, e.altKey);
            this.painting = false;
            this.requestDraw();
        }
    }

    pointerMove(ix, iy, e) {
        this.hover = { x: ix, y: iy };
        if (this.tool === "colour") {
            this.requestDraw();
            return;
        }
        if (!this.painting && (this.tool === "brush" || this.tool === "eraser")) this.requestDraw();
        if (this.panning && this.panAnchor) {
            this.panX = e.clientX - this.panAnchor.x;
            this.panY = e.clientY - this.panAnchor.y;
            this.requestDraw();
            return;
        }
        if (!this.painting) return;
        if (this.tool === "brush" || this.tool === "eraser") {
            const b = tools.lineBrush(
                (cx, cy) => this._stampAt(cx, cy),
                this.lastPt.x, this.lastPt.y, ix, iy,
                this.brushSize / 2, Math.max(1, this.brushSize * this.brushSpacing),
            );
            if (b) {
                this._strokeBounds = this._strokeBounds
                    ? {
                        x0: Math.min(this._strokeBounds.x0, b.x0),
                        y0: Math.min(this._strokeBounds.y0, b.y0),
                        x1: Math.max(this._strokeBounds.x1, b.x1),
                        y1: Math.max(this._strokeBounds.y1, b.y1),
                    }
                    : b;
            }
            this.lastPt = { x: ix, y: iy };
            this.requestDraw();
        } else if (this.tool === "rect" || this.tool === "ellipse") {
            this.shapeCur = { x: ix, y: iy };
            this.requestDraw();
        } else if (this.tool === "lasso") {
            this.lassoPts.push({ x: ix, y: iy });
            this.requestDraw();
        }
    }

    pointerUp(ix, iy, e) {
        this.finishPointerStroke(ix, iy, e);
    }

    closePolygon(e) {
        if (this.tool !== "polygon" || this.polyPts.length < 3) return;
        const sel = new Uint8Array(this.editW * this.editH);
        const b = tools.fillPolygon(sel, this.editW, this.editH, this.polyPts, 255);
        if (b) this._commitSelection(sel, b, e?.altKey ?? this.subtract);
        this.polyPts = [];
        this.painting = false;
        this.requestDraw();
    }

    popPolyPoint() {
        this.polyPts.pop();
        this.requestDraw();
    }

    wheelZoom(cx, cy, deltaY) {
        const f = deltaY < 0 ? 1.1 : 1 / 1.1;
        const z = zoomAtCursor(this.zoom, this.panX, this.panY, cx, cy, f);
        this.zoom = z.zoom; this.panX = z.panX; this.panY = z.panY;
        this.requestDraw();
    }

    fit() {
        const c = this.dom.canvas;
        if (!c) return;
        const v = fitView(c.clientWidth, c.clientHeight, this.editW, this.editH);
        this.zoom = v.zoom; this.panX = v.panX; this.panY = v.panY;
        this.viewTop = c.getBoundingClientRect().top;
        this.requestDraw();
    }

    zoom100() {
        const c = this.dom.canvas;
        if (!c) return;
        this.zoom = 1;
        this.panX = (c.clientWidth - this.editW) / 2;
        this.panY = (c.clientHeight - this.editH) / 2;
        this.viewTop = c.getBoundingClientRect().top;
        this.requestDraw();
    }

    cycleView() {
        this.viewMode = (this.viewMode + 1) % 3;
        if (this.viewMode === 1) this._rebuildMatteFull();
        this.requestDraw();
    }

    _drawNow() {
        const c = this.dom.canvas;
        if (!c) return;
        const ctx = c.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        const vw = c.clientWidth, vh = c.clientHeight;
        if (c.width !== Math.round(vw * dpr) || c.height !== Math.round(vh * dpr)) {
            c.width = Math.round(vw * dpr);
            c.height = Math.round(vh * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, vw, vh);
        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);
        // Zoomed in, show real pixels: the mask from 100% (its edge is what is being judged), the plate
        // from 400%. Zoomed out, smoothing avoids aliasing.
        ctx.imageSmoothingEnabled = this.zoom < 4;
        if (this.viewMode !== 1 && this.imageBitmap) {
            ctx.drawImage(this.imageBitmap, 0, 0, this.editW, this.editH);
        }
        ctx.imageSmoothingEnabled = this.zoom < 1;
        if (this.viewMode === 1 && this.matteCanvas) {
            ctx.drawImage(this.matteCanvas, 0, 0, this.editW, this.editH);
        } else if (this.viewMode !== 2 && this.overlayCanvas) {
            ctx.drawImage(this.overlayCanvas, 0, 0, this.editW, this.editH);
        }
        if (this.viewMode !== 2 && this.colourPreview && this.previewCanvas) {
            ctx.drawImage(this.previewCanvas, 0, 0, this.editW, this.editH);
        }
        ctx.imageSmoothingEnabled = true;
        if (this.shapeStart && this.shapeCur) {
            ctx.strokeStyle = C.accent2;
            ctx.setLineDash([4, 4]);
            const x0 = this.shapeStart.x, y0 = this.shapeStart.y;
            const x1 = this.shapeCur.x, y1 = this.shapeCur.y;
            if (this.tool === "ellipse") {
                const ax0 = Math.min(x0, x1), ay0 = Math.min(y0, y1);
                const ax1 = Math.max(x0, x1), ay1 = Math.max(y0, y1);
                const cx = (ax0 + ax1) / 2, cy = (ay0 + ay1) / 2;
                const rx = Math.max(0.5, (ax1 - ax0) / 2), ry = Math.max(0.5, (ay1 - ay0) / 2);
                ctx.beginPath();
                ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
                ctx.stroke();
            } else {
                ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
            }
            ctx.setLineDash([]);
        }
        if (this.polyPts.length > 1) {
            ctx.strokeStyle = C.accent2;
            ctx.beginPath();
            ctx.moveTo(this.polyPts[0].x, this.polyPts[0].y);
            for (let i = 1; i < this.polyPts.length; i++) {
                ctx.lineTo(this.polyPts[i].x, this.polyPts[i].y);
            }
            ctx.stroke();
        }
        if (this.lassoPts.length > 1) {
            ctx.strokeStyle = C.accent2;
            ctx.beginPath();
            ctx.moveTo(this.lassoPts[0].x, this.lassoPts[0].y);
            for (let i = 1; i < this.lassoPts.length; i++) {
                ctx.lineTo(this.lassoPts[i].x, this.lassoPts[i].y);
            }
            ctx.stroke();
        }
        if (this.tool === "colour" && this.colourSamplePixels.length) {
            const arm = 5 / this.zoom;
            for (const p of this.colourSamplePixels) {
                const cx = p.x + 0.5, cy = p.y + 0.5;
                ctx.beginPath();
                ctx.moveTo(cx - arm, cy); ctx.lineTo(cx + arm, cy);
                ctx.moveTo(cx, cy - arm); ctx.lineTo(cx, cy + arm);
                ctx.lineWidth = 3 / this.zoom;
                ctx.strokeStyle = C.black;
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(cx - arm, cy); ctx.lineTo(cx + arm, cy);
                ctx.moveTo(cx, cy - arm); ctx.lineTo(cx, cy + arm);
                ctx.lineWidth = 1.5 / this.zoom;
                ctx.strokeStyle = C.white;
                ctx.stroke();
            }
            ctx.lineWidth = 1;
        }
        if (this.hover && (this.tool === "brush" || this.tool === "eraser")) {
            // Brush outline, two-tone so it reads on any plate; line widths are screen pixels.
            const r = Math.max(0.5, this.brushSize / 2);
            ctx.beginPath();
            ctx.arc(this.hover.x, this.hover.y, r, 0, Math.PI * 2);
            ctx.lineWidth = 3 / this.zoom;
            ctx.strokeStyle = C.black;
            ctx.stroke();
            ctx.lineWidth = 1.5 / this.zoom;
            ctx.strokeStyle = this.tool === "eraser" || this.subtract ? C.danger : C.white;
            ctx.stroke();
            ctx.lineWidth = 1;
        }
        ctx.restore();
    }

    async save() {
        this._commitBrushStroke();
        this._commitFrame();
        let digest = this.digest;
        const uploaded = [];
        for (const f of this.dirtyFrames) {
            let buf = this.masks.get(f);
            if (!buf && f === this.curFrame) buf = this.mask;
            if (!buf) continue;
            const blob = await this._maskToPngBlob(buf);
            const j = await api.putFramePng(this.editorId, f, blob);
            digest = j.digest;
            uploaded.push(f);
        }
        this.dirtyFrames.clear();
        for (const f of uploaded) this.storedFrames.add(f);
        this.digest = digest;
        this._notify();
        return digest;
    }

    async _maskToPngBlob(buf) {
        const small = document.createElement("canvas");
        small.width = this.editW;
        small.height = this.editH;
        const scx = small.getContext("2d");
        const id = scx.createImageData(this.editW, this.editH);
        for (let p = 0, i = 0; p < buf.length; p++, i += 4) {
            id.data[i] = id.data[i + 1] = id.data[i + 2] = buf[p];
            id.data[i + 3] = 255;
        }
        scx.putImageData(id, 0, 0);
        const out = document.createElement("canvas");
        out.width = this.nativeW;
        out.height = this.nativeH;
        out.getContext("2d").drawImage(small, 0, 0, this.nativeW, this.nativeH);
        return new Promise((res) => out.toBlob(res, "image/png"));
    }

    dispose() {
        if (this._raf) cancelAnimationFrame(this._raf);
        this._raf = 0;
        this._releaseBitmap();
        this.masks.clear();
        this.storedFrames.clear();
        this.dirtyFrames.clear();
        this.cleanLru = [];
        this.mask = null;
        this.overlayCanvas = null;
        this.previewCanvas = null;
        this.matteCanvas = null;
        this.node = null;
    }
}
