/**
 * IMEEditor — mask editing state for one modal session.
 */
import { resolveEditorSource } from "./source.js";
import { C, hexToRgb } from "./palette.js";
import { fitView, zoomAtCursor } from "./coords.js";
import { FrameHistory, captureRect, applyRect, pushUndoEntry } from "./undo.js";
import * as tools from "./tools.js";
import * as colour from "./colour.js";
import * as api from "./api.js";
import { c2cAlert } from "../_c2c_dialog.js";
import {
    rubylithRGBA, gridLines,
    overlayRGBA, matteRGBA, previewRGBA, outlineTileRGBA, outlineBufRect,
    overlayTileLOD, matteTileLOD, rubylithTileLOD, previewTileLOD, outlineTileLOD,
} from "./view.js";
import { TileCache, tilesForRect, pickLodLevel, DEFAULT_BYTE_BUDGET } from "./tiles.js";

const MP_LIMIT = 40_000_000;
const SAM_MP_LIMIT = 4_000_000;
const OVERLAY_ALPHA = 0.45;
const PREVIEW_ALPHA = 0.45;
const CLEAN_MASK_CAP = 8;
const APPLY_ALL_MEM_CAP = 512 * 1024 * 1024;
const VIEW_MODE_KEY = "c2c.ime.viewMode";

function _loadViewMode() {
    try {
        const v = parseInt(localStorage.getItem(VIEW_MODE_KEY), 10);
        if (Number.isInteger(v) && v >= 0 && v <= 4) return v;
    } catch (_) { /* ignore */ }
    return 0;
}

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
        this._tileCache = new TileCache({ byteBudget: DEFAULT_BYTE_BUDGET });
        this._outlineBuf = null;
        this._outlineBufStale = false;
        this._outlineDirty = null;
        this._overlayTint = hexToRgb(C.accent);
        this._gridTint = hexToRgb(C.sub);
        this.history = new FrameHistory();
        this.pressureSize = false;
        this.pressureOpacity = false;
        this._lastPressure = 1;
        this._maskClipboard = null;
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
        this.candidatePreview = null;
        this._previewTint = hexToRgb(C.accent2);
        this.samPoints = [];
        this.samBox = null;
        this.samSeq = 0;
        this.samAbort = null;
        this.samModel = "";
        this.samDragStart = null;
        this.samScore = 0;
        this.frameKey = "";
        this.frameKeyFor = -1;
        this.refineBand = null;
        this.refineSeq = 0;
        this.refineAbort = null;
        this._refineStrokeStart = null;
        this._refineCoverage = null;
        this._refineStrokeBounds = null;
        this.toolStatus = "";
        this.toolStatusKind = "";
        this.viewMode = _loadViewMode();
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
        this.onStatus = null;
    }

    _reportStatus(text, kind = "info") {
        this.toolStatus = text || "";
        this.toolStatusKind = kind;
        this.onStatus?.(this.toolStatus, kind);
    }

    getToolStatusLine() {
        return this.toolStatus || "";
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

    _viewName(vm) {
        return ["overlay", "matte", "rubylith", "outline"][vm];
    }

    _ensureOutlineBuf() {
        const n = this.editW * this.editH;
        if (!this._outlineBuf || this._outlineBuf.length !== n) {
            this._outlineBuf = new Uint8Array(n);
            this._outlineBufStale = true;
            this._outlineDirty = null;
            this._tileCache.setImageSize(this.editW, this.editH);
        }
    }

    _rebuildOutlineBufFull() {
        if (!this.mask || !this._outlineBuf) return;
        this._outlineBuf.fill(0);
        outlineBufRect(
            this.mask, this.editW, this.editH,
            0, 0, this.editW - 1, this.editH - 1,
            this._outlineBuf,
        );
    }

    _updateOutlineBufRect(x0, y0, x1, y1) {
        if (!this.mask || !this._outlineBuf) return;
        outlineBufRect(this.mask, this.editW, this.editH, x0, y0, x1, y1, this._outlineBuf);
    }

    _flushOutlineBuf() {
        const vm = this.viewMode;
        if (vm === 3) {
            if (this._outlineBufStale) {
                this._rebuildOutlineBufFull();
                this._outlineBufStale = false;
                this._outlineDirty = null;
            } else if (this._outlineDirty) {
                const d = this._outlineDirty;
                this._outlineDirty = null;
                const x0 = Math.max(0, d.x0 - 1);
                const y0 = Math.max(0, d.y0 - 1);
                const x1 = Math.min(this.editW - 1, d.x1 + 1);
                const y1 = Math.min(this.editH - 1, d.y1 + 1);
                this._updateOutlineBufRect(x0, y0, x1, y1);
            }
        } else if (this._outlineDirty) {
            this._outlineBufStale = true;
            this._outlineDirty = null;
        }
    }

    _createTileCanvas(w, h) {
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        return { canvas, ctx: canvas.getContext("2d"), w, h, dirty: null, bytes: w * h * 4 };
    }

    _fillTileRegion(viewKey, t, u0, v0, u1, v1, id, srcBuf = null) {
        const { x, y, w, h, level } = t;
        const rw = u1 - u0 + 1;
        const rh = v1 - v0 + 1;
        if (viewKey === "overlay" && this.mask) {
            if (level === 0) {
                overlayRGBA(
                    this.mask, this.editW, this.editH, x + u0, y + v0, x + u1, y + v1,
                    id.data, rw, this._overlayTint, OVERLAY_ALPHA,
                );
            } else {
                overlayTileLOD(
                    this.mask, this.editW, this.editH, level, x, y,
                    u0, v0, u1, v1, id.data, rw, this._overlayTint, OVERLAY_ALPHA,
                );
            }
        } else if (viewKey === "matte" && this.mask) {
            if (level === 0) {
                matteRGBA(
                    this.mask, this.editW, this.editH, x + u0, y + v0, x + u1, y + v1,
                    id.data, rw,
                );
            } else {
                matteTileLOD(
                    this.mask, this.editW, this.editH, level, x, y,
                    u0, v0, u1, v1, id.data, rw,
                );
            }
        } else if (viewKey === "rubylith" && this.mask) {
            if (level === 0) {
                rubylithRGBA(
                    this.mask, this.editW, this.editH, x + u0, y + v0, x + u1, y + v1,
                    id.data, rw,
                );
            } else {
                rubylithTileLOD(
                    this.mask, this.editW, this.editH, level, x, y,
                    u0, v0, u1, v1, id.data, rw,
                );
            }
        } else if (viewKey === "outline" && this.mask) {
            if (level === 0) {
                outlineTileRGBA(this.mask, this.editW, this.editH, x, y, w, h, id.data, rw);
            } else {
                outlineTileLOD(
                    this.mask, this.editW, this.editH, level, x, y, w, h,
                    u0, v0, u1, v1, id.data, rw,
                );
            }
        } else if (viewKey === "preview") {
            const buf = srcBuf || this.candidatePreview;
            if (buf) {
                if (level === 0) {
                    previewRGBA(
                        buf, this.editW, this.editH, x + u0, y + v0, x + u1, y + v1,
                        id.data, rw, this._previewTint, PREVIEW_ALPHA,
                    );
                } else {
                    previewTileLOD(
                        buf, this.editW, this.editH, level, x, y,
                        u0, v0, u1, v1, id.data, rw, this._previewTint, PREVIEW_ALPHA,
                    );
                }
            }
        }
    }

    _fillTile(viewKey, t, srcBuf = null, dirtyRect = null) {
        const { w, h, tx, ty, level } = t;
        let u0 = 0;
        let v0 = 0;
        let u1 = w - 1;
        let v1 = h - 1;
        if (dirtyRect) {
            u0 = dirtyRect.x0;
            v0 = dirtyRect.y0;
            u1 = dirtyRect.x1;
            v1 = dirtyRect.y1;
        }
        if (viewKey === "outline") {
            u0 = Math.max(0, u0 - 1);
            v0 = Math.max(0, v0 - 1);
            u1 = Math.min(w - 1, u1 + 1);
            v1 = Math.min(h - 1, v1 + 1);
        }
        let tile = this._tileCache.get(viewKey, level, tx, ty);
        if (!tile) tile = this._createTileCanvas(w, h);
        const rw = u1 - u0 + 1;
        const rh = v1 - v0 + 1;
        const id = tile.ctx.createImageData(rw, rh);
        if (viewKey === "outline" && level === 0 && dirtyRect) {
            const tmp = tile.ctx.createImageData(w, h);
            outlineTileRGBA(this.mask, this.editW, this.editH, t.x, t.y, w, h, tmp.data, w);
            for (let v = v0; v <= v1; v++) {
                for (let u = u0; u <= u1; u++) {
                    const si = (v * w + u) * 4;
                    const di = ((v - v0) * rw + (u - u0)) * 4;
                    id.data[di] = tmp.data[si];
                    id.data[di + 1] = tmp.data[si + 1];
                    id.data[di + 2] = tmp.data[si + 2];
                    id.data[di + 3] = tmp.data[si + 3];
                }
            }
        } else {
            this._fillTileRegion(viewKey, t, u0, v0, u1, v1, id, srcBuf);
        }
        tile.ctx.putImageData(id, u0, v0);
        this._tileCache.set(viewKey, level, tx, ty, tile);
    }

    _drawViewTiles(ctx, viewKey, srcBuf = null, vw, vh) {
        const { ix0, iy0, ix1, iy1 } = this._visibleImageRect(vw, vh);
        const level = pickLodLevel(
            this.zoom, ix0, iy0, ix1, iy1,
            this.editW, this.editH, DEFAULT_BYTE_BUDGET,
        );
        // Tiles are placed in screen space with edges snapped to whole pixels, so neighbours abut exactly.
        const z = this.zoom;
        const px = this.panX;
        const py = this.panY;
        const dpr = window.devicePixelRatio || 1;
        ctx.save();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        for (const t of tilesForRect(ix0, iy0, ix1, iy1, this.editW, this.editH, level)) {
            let tile = this._tileCache.get(viewKey, level, t.tx, t.ty);
            if (!tile) {
                this._fillTile(viewKey, t, srcBuf, null);
                tile = this._tileCache.get(viewKey, level, t.tx, t.ty);
            } else if (tile.dirty) {
                this._fillTile(viewKey, t, srcBuf, tile.dirty);
                tile = this._tileCache.get(viewKey, level, t.tx, t.ty);
            }
            if (!tile?.canvas) continue;
            const dx0 = Math.round(px + t.x * z);
            const dy0 = Math.round(py + t.y * z);
            const dx1 = Math.round(px + (t.x + t.imageW) * z);
            const dy1 = Math.round(py + (t.y + t.imageH) * z);
            if (dx1 > dx0 && dy1 > dy0) {
                ctx.drawImage(tile.canvas, 0, 0, t.w, t.h, dx0, dy0, dx1 - dx0, dy1 - dy0);
            }
        }
        ctx.restore();
    }

    _visibleImageRect(vw, vh) {
        const ix0 = Math.max(0, Math.floor((0 - this.panX) / this.zoom));
        const iy0 = Math.max(0, Math.floor((0 - this.panY) / this.zoom));
        const ix1 = Math.min(this.editW - 1, Math.ceil((vw - this.panX) / this.zoom) - 1);
        const iy1 = Math.min(this.editH - 1, Math.ceil((vh - this.panY) / this.zoom) - 1);
        return { ix0, iy0, ix1, iy1 };
    }

    _setCandidatePreview(buf) {
        this.candidatePreview = buf;
        this._tileCache.markViewStale("preview");
        this.requestDraw();
    }

    _clearCandidatePreview() {
        this.candidatePreview = null;
        this._tileCache.markViewStale("preview");
        this.requestDraw();
    }

    candidatePreviewActive() {
        if (!this.candidatePreview) return false;
        if (this.colourSamples.length > 0) return true;
        if (this.tool === "sam" && (this.samPoints.length > 0 || this.samBox)) return true;
        return false;
    }

    applyCandidatePreview() {
        if (!this.candidatePreviewActive()) return;
        const bounds = { x0: 0, y0: 0, x1: this.editW - 1, y1: this.editH - 1 };
        this._commitSelection(this.candidatePreview, bounds, false);
        if (this.colourSamples.length > 0) {
            this.clearColourPreview();
        } else {
            this.clearSamState(false);
        }
        this._reportStatus("");
    }

    applySamPreview() {
        this.applyCandidatePreview();
    }

    // Mask edits mark tiles stale; outlineBuf is updated once per frame in _drawNow when view 3 is shown.
    _maskDirtyRect(x0, y0, x1, y1) {
        const x = Math.max(0, x0 | 0);
        const y = Math.max(0, y0 | 0);
        const x2 = Math.min(this.editW, (x1 | 0) + 1) - 1;
        const y2 = Math.min(this.editH, (y1 | 0) + 1) - 1;
        if (x2 < x || y2 < y) return;
        this._tileCache.markStaleRect(x, y, x2, y2);
        const od = this._outlineDirty;
        this._outlineDirty = od
            ? { x0: Math.min(od.x0, x), y0: Math.min(od.y0, y), x1: Math.max(od.x1, x2), y1: Math.max(od.y1, y2) }
            : { x0: x, y0: y, x1: x2, y1: y2 };
        this.requestDraw();
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
        this._ensureOutlineBuf();
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
            this.clearSamState(false);
            this.clearRefineState(false);
            this._evictLeavingFrame(this.curFrame);
        }
        this.curFrame = idx;
        if (this.storedFrames.has(idx) && !this.masks.has(idx)) {
            await this._loadMaskFrame(idx);
        }
        this.mask = this.masks.get(idx) || this._emptyMask();
        await this._loadImageFrame(idx);
        this._tileCache.markAllStale();
        this._outlineBufStale = true;
        this._outlineDirty = null;
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
            this.frameKey = "";
            this.frameKeyFor = -1;
        } catch (e) {
            console.warn("[IME] image load:", e);
        }
    }

    _notify() {
        this.onChange?.();
    }

    _commitUndoRect(x0, y0, x1, y1, beforeSrc) {
        const entry = pushUndoEntry(
            this.history, this.curFrame, this.mask, this.editW, this.editH,
            x0, y0, x1, y1, beforeSrc,
        );
        this.dirtyFrames.add(this.curFrame);
        this._maskDirtyRect(entry.x, entry.y, entry.x + entry.w - 1, entry.y + entry.h - 1);
        this._notify();
        return entry;
    }

    undoOp() {
        const e = this.history.undo(this.curFrame);
        if (!e) return;
        this.history.pushRedo(this.curFrame, e);
        applyRect(this.mask, this.editW, { x: e.x, y: e.y, w: e.w, h: e.h, data: e.before });
        this.dirtyFrames.add(this.curFrame);
        this._maskDirtyRect(e.x, e.y, e.x + e.w - 1, e.y + e.h - 1);
        this.requestDraw();
        this._notify();
    }

    redoOp() {
        const e = this.history.redo(this.curFrame);
        if (!e) return;
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

    copyMask() {
        this._maskClipboard = new Uint8Array(this.mask);
        this._reportStatus("Mask copied");
    }

    pasteMask(altKey = false) {
        if (!this._maskClipboard) {
            this._reportStatus("Nothing to paste", "error");
            return;
        }
        const bounds = { x0: 0, y0: 0, x1: this.editW - 1, y1: this.editH - 1 };
        this._commitSelection(this._maskClipboard, bounds, altKey);
        this._reportStatus(`Pasted mask (${this._effectiveSelectionMode(altKey)})`);
        this.requestDraw();
    }

    clearFrame() {
        const before = new Uint8Array(this.mask);
        this.mask.fill(0);
        this._commitUndoRect(0, 0, this.editW - 1, this.editH - 1, before);
        this._reportStatus("Frame cleared");
        this.requestDraw();
    }

    exportMaskPng() {
        const c = document.createElement("canvas");
        c.width = this.editW;
        c.height = this.editH;
        const cx = c.getContext("2d");
        const id = cx.createImageData(this.editW, this.editH);
        for (let p = 0, i = 0; p < this.mask.length; p++, i += 4) {
            id.data[i] = id.data[i + 1] = id.data[i + 2] = this.mask[p];
            id.data[i + 3] = 255;
        }
        cx.putImageData(id, 0, 0);
        c.toBlob((blob) => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `${this.editorId}_frame${this.curFrame + 1}.png`;
            a.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        }, "image/png");
        this._reportStatus("Mask exported");
    }

    async importMaskPng(file) {
        if (!file) return;
        const blob = await file.arrayBuffer();
        const bm = await createImageBitmap(new Blob([blob]));
        const sw = bm.width;
        const sh = bm.height;
        const c = document.createElement("canvas");
        c.width = this.editW;
        c.height = this.editH;
        const cx = c.getContext("2d");
        const src = document.createElement("canvas");
        src.width = sw;
        src.height = sh;
        src.getContext("2d").drawImage(bm, 0, 0);
        bm.close();
        const sdata = src.getContext("2d").getImageData(0, 0, sw, sh).data;
        let useAlpha = false;
        for (let i = 3; i < sdata.length; i += 4) {
            if (sdata[i] < 255) {
                useAlpha = true;
                break;
            }
        }
        const sample = (i) => useAlpha
            ? sdata[i + 3]
            : Math.round(0.299 * sdata[i] + 0.587 * sdata[i + 1] + 0.114 * sdata[i + 2]);
        let binary = true;
        for (let i = 0; i < sdata.length; i += 4) {
            const v = sample(i);
            if (v !== 0 && v !== 255) {
                binary = false;
                break;
            }
        }
        cx.imageSmoothingEnabled = !binary;
        cx.drawImage(src, 0, 0, this.editW, this.editH);
        const id = cx.getImageData(0, 0, this.editW, this.editH);
        const sel = new Uint8Array(this.editW * this.editH);
        for (let p = 0, i = 0; p < sel.length; p++, i += 4) {
            sel[p] = useAlpha
                ? id.data[i + 3]
                : Math.round(0.299 * id.data[i] + 0.587 * id.data[i + 1] + 0.114 * id.data[i + 2]);
        }
        const bounds = { x0: 0, y0: 0, x1: this.editW - 1, y1: this.editH - 1 };
        this._commitSelection(sel, bounds, false);
        this._reportStatus(`Imported mask (${this.selectionMode})`);
        this.requestDraw();
    }

    async _ensureMaskFrame(f) {
        const fi = f | 0;
        if (this.storedFrames.has(fi) && !this.masks.has(fi)) {
            await this._loadMaskFrame(fi);
        }
        if (!this.masks.has(fi)) {
            this.masks.set(fi, this._emptyMask());
        }
    }

    async applyToAllFrames() {
        if (this.frameCount <= 1) {
            this._reportStatus("Only one frame", "error");
            return;
        }
        let newHeld = 0;
        for (let f = 0; f < this.frameCount; f++) {
            if (f === this.curFrame) continue;
            if (!this.masks.has(f)) newHeld++;
        }
        const bytes = newHeld * this.editW * this.editH;
        if (bytes > APPLY_ALL_MEM_CAP) {
            const gb = bytes / (1024 ** 3);
            const gbText = gb >= 10 ? Math.round(gb) : gb.toFixed(1);
            const msg =
                `Applying to all ${this.frameCount} frames would hold about ${gbText} GB of masks in the browser. `
                + "Set the node's frame_mode to 'shared' instead - it uses this frame's mask for every frame without copying it.";
            this._reportStatus(msg, "error");
            c2cAlert(msg);
            return;
        }
        const sel = new Uint8Array(this.mask);
        const bounds = { x0: 0, y0: 0, x1: this.editW - 1, y1: this.editH - 1 };
        let n = 0;
        for (let f = 0; f < this.frameCount; f++) {
            if (f === this.curFrame) continue;
            await this._ensureMaskFrame(f);
            this._commitSelectionOnFrame(f, sel, bounds, false);
            n++;
        }
        this._reportStatus(`Applied to ${n} frame${n === 1 ? "" : "s"}`);
        this._notify();
        this.requestDraw();
    }

    setTool(t) {
        this.cancelShapeInProgress();
        this.clearColourPreview();
        this.clearSamState(false);
        this.clearRefineState(false);
        this.tool = t;
        this._reportStatus("");
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
            (this.tool === "sam" && (this.samPoints.length > 0 || this.samBox)) ||
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
        return this.candidatePreview != null && this.colourSamples.length > 0;
    }

    clearColourPreview() {
        this.colourSamples = [];
        this.colourSamplePixels = [];
        this.colourDist = null;
        this.colourDistKey = "";
        if (this.tool !== "sam") {
            this._clearCandidatePreview();
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
            if (this.tool !== "sam") this._clearCandidatePreview();
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
        this._setCandidatePreview(sel);
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
        this.applyCandidatePreview();
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
        this.history.push(this.curFrame, {
            x: after.x, y: after.y, w: after.w, h: after.h,
            before: before.data, after: after.data,
        });
        this.dirtyFrames.add(this.curFrame);
        this._maskDirtyRect(rb.x0, rb.y0, rb.x1, rb.y1);
        this._notify();
    }

    _commitSelectionOnFrame(frame, sel, bounds, altKey) {
        const f = frame | 0;
        let mask = this.masks.get(f);
        if (!mask) {
            mask = this._emptyMask();
            this.masks.set(f, mask);
        }
        const mode = this._effectiveSelectionMode(altKey);
        const ub = mode === "intersect"
            ? { x0: 0, y0: 0, x1: this.editW - 1, y1: this.editH - 1 }
            : bounds;
        const before = captureRect(mask, this.editW, this.editH, ub.x0, ub.y0, ub.x1, ub.y1);
        const rb = tools.composeSelection(mask, sel, this.editW, this.editH, mode, bounds);
        const after = captureRect(mask, this.editW, this.editH, rb.x0, rb.y0, rb.x1, rb.y1);
        this.history.push(f, {
            x: after.x, y: after.y, w: after.w, h: after.h,
            before: before.data, after: after.data,
        });
        this.dirtyFrames.add(f);
        if (f === this.curFrame) {
            this._maskDirtyRect(rb.x0, rb.y0, rb.x1, rb.y1);
        }
    }

    async _ensureFrameKey() {
        if (this.frameKeyFor === this.curFrame && this.frameKey) return this.frameKey;
        if (!this.imageData) return "";
        const digest = await crypto.subtle.digest("SHA-1", this.imageData);
        this.frameKey = Array.from(new Uint8Array(digest))
            .map((b) => b.toString(16).padStart(2, "0")).join("");
        this.frameKeyFor = this.curFrame;
        return this.frameKey;
    }

    _samUploadScale() {
        const px = this.editW * this.editH;
        return px > SAM_MP_LIMIT ? Math.sqrt(SAM_MP_LIMIT / px) : 1;
    }

    async _rgbPngBlob() {
        const s = this._samUploadScale();
        const c = document.createElement("canvas");
        if (s === 1) {
            c.width = this.editW;
            c.height = this.editH;
            const cx = c.getContext("2d");
            const id = cx.createImageData(this.editW, this.editH);
            for (let p = 0, i = 0; p < this.editW * this.editH; p++, i += 4) {
                id.data[i] = this.imageData[i];
                id.data[i + 1] = this.imageData[i + 1];
                id.data[i + 2] = this.imageData[i + 2];
                id.data[i + 3] = 255;
            }
            cx.putImageData(id, 0, 0);
            return new Promise((res) => c.toBlob(res, "image/png"));
        }
        const w = Math.max(1, Math.round(this.editW * s));
        const h = Math.max(1, Math.round(this.editH * s));
        c.width = w;
        c.height = h;
        const cx = c.getContext("2d");
        cx.imageSmoothingEnabled = true;
        if (this.imageBitmap) {
            cx.drawImage(this.imageBitmap, 0, 0, this.editW, this.editH, 0, 0, w, h);
        } else if (this.imageData) {
            const full = document.createElement("canvas");
            full.width = this.editW;
            full.height = this.editH;
            const fcx = full.getContext("2d");
            const id = fcx.createImageData(this.editW, this.editH);
            for (let p = 0, i = 0; p < this.editW * this.editH; p++, i += 4) {
                id.data[i] = this.imageData[i];
                id.data[i + 1] = this.imageData[i + 1];
                id.data[i + 2] = this.imageData[i + 2];
                id.data[i + 3] = 255;
            }
            fcx.putImageData(id, 0, 0);
            cx.drawImage(full, 0, 0, this.editW, this.editH, 0, 0, w, h);
        }
        return new Promise((res) => c.toBlob(res, "image/png"));
    }

    _upscaleMaskUint8(src, sw, sh, dw, dh) {
        const c = document.createElement("canvas");
        c.width = sw;
        c.height = sh;
        const cx = c.getContext("2d");
        const id = cx.createImageData(sw, sh);
        for (let p = 0, i = 0; p < src.length; p++, i += 4) {
            const v = src[p];
            id.data[i] = id.data[i + 1] = id.data[i + 2] = v;
            id.data[i + 3] = 255;
        }
        cx.putImageData(id, 0, 0);
        const out = document.createElement("canvas");
        out.width = dw;
        out.height = dh;
        const ocx = out.getContext("2d");
        ocx.imageSmoothingEnabled = true;
        ocx.drawImage(c, 0, 0, sw, sh, 0, 0, dw, dh);
        const oid = ocx.getImageData(0, 0, dw, dh);
        const mask = new Uint8Array(dw * dh);
        for (let p = 0, i = 0; p < mask.length; p++, i += 4) mask[p] = oid.data[i];
        return mask;
    }

    clearSamState(report = true) {
        this.samAbort?.abort();
        this.samAbort = null;
        this.samPoints = [];
        this.samBox = null;
        this.samDragStart = null;
        this.samScore = 0;
        if (this.tool === "sam" || !this.colourSamples.length) {
            this._clearCandidatePreview();
        }
        if (report) this._reportStatus("");
        this.requestDraw();
    }

    clearRefineState(report = true) {
        this.refineAbort?.abort();
        this.refineAbort = null;
        this.refineBand = null;
        this._refineStrokeStart = null;
        this._refineCoverage = null;
        this._refineStrokeBounds = null;
        if (this.tool === "refine") {
            this._tileCache.markViewStale("preview");
        }
        if (report) this._reportStatus("");
        this.requestDraw();
    }

    _samMeta(seq) {
        const s = this._samUploadScale();
        const w = Math.max(1, Math.round(this.editW * s));
        const h = Math.max(1, Math.round(this.editH * s));
        return {
            editor_id: this.editorId,
            frame_key: this.frameKey,
            seq,
            width: w,
            height: h,
            model: this.samModel || "",
            points: this.samPoints.map((p) => [p.x * s, p.y * s, p.label]),
            box: this.samBox ? this.samBox.map((v) => v * s) : null,
        };
    }

    async _requestSam() {
        if (this.tool !== "sam" || !this.imageData) return;
        if (!this.samModel) {
            this._reportStatus(this.samUnavailable || "SAM: no model installed - pick one marked [download] to fetch it.", "error");
            return;
        }
        if (!this.samPoints.length && !this.samBox) {
            this._clearCandidatePreview();
            this._reportStatus("");
            return;
        }
        const seq = ++this.samSeq;
        this.samAbort?.abort();
        const ac = new AbortController();
        this.samAbort = ac;
        this._reportStatus("SAM: computing image embedding…");
        try {
            await this._ensureFrameKey();
            const meta = this._samMeta(seq);
            let result;
            try {
                result = await api.samPredict(meta);
            } catch (e) {
                if (e.needImage) {
                    this._reportStatus("SAM: uploading frame…");
                    const blob = await this._rgbPngBlob();
                    if (seq !== this.samSeq) return;
                    result = await api.samPredict(meta, blob);
                } else if (e.superseded) {
                    return;
                } else {
                    throw e;
                }
            }
            if (seq !== this.samSeq || ac.signal.aborted) return;
            const s = this._samUploadScale();
            const expectW = Math.max(1, Math.round(this.editW * s));
            const expectH = Math.max(1, Math.round(this.editH * s));
            if (result.width !== expectW || result.height !== expectH) {
                this._reportStatus("SAM returned an unexpected mask size.", "error");
                return;
            }
            this.samScore = result.score;
            const mask = s === 1
                ? result.mask
                : this._upscaleMaskUint8(result.mask, expectW, expectH, this.editW, this.editH);
            this._setCandidatePreview(mask);
            this._reportStatus(`SAM score: ${result.score.toFixed(2)}`);
        } catch (e) {
            if (seq !== this.samSeq || ac.signal.aborted || e.superseded) return;
            this._reportStatus(e?.message || "SAM request failed.", "error");
        }
    }

    _ensureRefineBand() {
        if (!this.refineBand || this.refineBand.length !== this.editW * this.editH) {
            this.refineBand = new Uint8Array(this.editW * this.editH);
        }
        return this.refineBand;
    }

    _beginRefineStroke(ix, iy) {
        const band = this._ensureRefineBand();
        this._refineStrokeStart = new Uint8Array(band);
        this._refineCoverage = new Uint8Array(band.length);
        this._refineStrokeBounds = null;
        this._stampRefineAt(ix, iy);
    }

    _stampRefineAt(cx, cy) {
        const band = this._ensureRefineBand();
        const b = tools.stampBrushCoverage(
            band, this._refineStrokeStart, this._refineCoverage,
            this.editW, this.editH, cx, cy,
            this.brushSize / 2, this.brushHardness, 1, false,
        );
        if (b) {
            this._refineStrokeBounds = this._refineStrokeBounds
                ? {
                    x0: Math.min(this._refineStrokeBounds.x0, b.x0),
                    y0: Math.min(this._refineStrokeBounds.y0, b.y0),
                    x1: Math.max(this._refineStrokeBounds.x1, b.x1),
                    y1: Math.max(this._refineStrokeBounds.y1, b.y1),
                }
                : b;
            this._syncRefineBandPreview();
        }
    }

    _syncRefineBandPreview() {
        if (!this.refineBand) return;
        this._tileCache.markViewStale("preview");
        this.requestDraw();
    }

    _bandBBox(margin = 16) {
        const band = this.refineBand;
        if (!band) return null;
        let x0 = this.editW, y0 = this.editH, x1 = -1, y1 = -1;
        for (let y = 0; y < this.editH; y++) {
            for (let x = 0; x < this.editW; x++) {
                if (band[y * this.editW + x] > 0) {
                    x0 = Math.min(x0, x);
                    y0 = Math.min(y0, y);
                    x1 = Math.max(x1, x);
                    y1 = Math.max(y1, y);
                }
            }
        }
        if (x1 < x0) return null;
        x0 = Math.max(0, x0 - margin);
        y0 = Math.max(0, y0 - margin);
        x1 = Math.min(this.editW - 1, x1 + margin);
        y1 = Math.min(this.editH - 1, y1 + margin);
        return { x0, y0, x1, y1 };
    }

    async _cropPngRgb(x0, y0, x1, y1) {
        const w = x1 - x0 + 1;
        const h = y1 - y0 + 1;
        const c = document.createElement("canvas");
        c.width = w;
        c.height = h;
        const cx = c.getContext("2d");
        const id = cx.createImageData(w, h);
        for (let yy = 0; yy < h; yy++) {
            for (let xx = 0; xx < w; xx++) {
                const si = ((y0 + yy) * this.editW + (x0 + xx)) * 4;
                const di = (yy * w + xx) * 4;
                id.data[di] = this.imageData[si];
                id.data[di + 1] = this.imageData[si + 1];
                id.data[di + 2] = this.imageData[si + 2];
                id.data[di + 3] = 255;
            }
        }
        cx.putImageData(id, 0, 0);
        return new Promise((res) => c.toBlob(res, "image/png"));
    }

    async _cropPngL(buf, x0, y0, x1, y1) {
        const w = x1 - x0 + 1;
        const h = y1 - y0 + 1;
        const c = document.createElement("canvas");
        c.width = w;
        c.height = h;
        const cx = c.getContext("2d");
        const id = cx.createImageData(w, h);
        for (let yy = 0; yy < h; yy++) {
            for (let xx = 0; xx < w; xx++) {
                const v = buf[(y0 + yy) * this.editW + (x0 + xx)];
                const di = (yy * w + xx) * 4;
                id.data[di] = id.data[di + 1] = id.data[di + 2] = v;
                id.data[di + 3] = 255;
            }
        }
        cx.putImageData(id, 0, 0);
        return new Promise((res) => c.toBlob(res, "image/png"));
    }

    async _runRefine() {
        if (this.tool !== "refine" || !this.refineBand || !this.imageData) return;
        const bb = this._bandBBox(16);
        if (!bb) {
            this.clearRefineState(false);
            return;
        }
        const seq = ++this.refineSeq;
        this.refineAbort?.abort();
        const ac = new AbortController();
        this.refineAbort = ac;
        this._reportStatus("Refine: computing edge matte…");
        try {
            const { x0, y0, x1, y1 } = bb;
            const [imgB, maskB, bandB] = await Promise.all([
                this._cropPngRgb(x0, y0, x1, y1),
                this._cropPngL(this.mask, x0, y0, x1, y1),
                this._cropPngL(this.refineBand, x0, y0, x1, y1),
            ]);
            if (seq !== this.refineSeq || ac.signal.aborted) return;
            const meta = { editor_id: this.editorId, seq, bbox: [x0, y0, x1, y1] };
            const alphaBlob = await api.refine(meta, imgB, maskB, bandB);
            if (seq !== this.refineSeq || ac.signal.aborted) return;
            const bm = await createImageBitmap(alphaBlob);
            const cw = bm.width;
            const ch = bm.height;
            const c = document.createElement("canvas");
            c.width = cw;
            c.height = ch;
            const cx = c.getContext("2d");
            cx.drawImage(bm, 0, 0);
            bm.close();
            const id = cx.getImageData(0, 0, cw, ch);
            const alpha = new Uint8Array(cw * ch);
            for (let p = 0, i = 0; p < alpha.length; p++, i += 4) alpha[p] = id.data[i];
            this._applyRefineCrop(x0, y0, alpha, cw, ch);
            this.clearRefineState(false);
            this._reportStatus("Refine applied.");
        } catch (e) {
            if (seq !== this.refineSeq || ac.signal.aborted || e.superseded) return;
            this._reportStatus(e?.message || "Refine failed.", e.needWeights ? "need_weights" : "error");
        }
    }

    _applyRefineCrop(x0, y0, alphaCrop, cw, ch) {
        const before = captureRect(
            this.mask, this.editW, this.editH,
            x0, y0, x0 + cw - 1, y0 + ch - 1,
        );
        const band = this.refineBand;
        for (let yy = 0; yy < ch; yy++) {
            for (let xx = 0; xx < cw; xx++) {
                const fx = x0 + xx;
                const fy = y0 + yy;
                if (fx < 0 || fy < 0 || fx >= this.editW || fy >= this.editH) continue;
                const i = fy * this.editW + fx;
                const b = band[i] / 255;
                if (b <= 0) continue;
                const a = alphaCrop[yy * cw + xx];
                const m = this.mask[i];
                this.mask[i] = Math.round(b * a + (1 - b) * m);
            }
        }
        // `before` is a crop; _commitUndoRect expects a full-frame source, so push the entry directly.
        const after = captureRect(this.mask, this.editW, this.editH, x0, y0, x0 + cw - 1, y0 + ch - 1);
        this.history.push(this.curFrame, {
            x: before.x, y: before.y, w: before.w, h: before.h,
            before: before.data, after: after.data,
        });
        this.dirtyFrames.add(this.curFrame);
        this._maskDirtyRect(x0, y0, x0 + cw - 1, y0 + ch - 1);
        this._notify();
        this.requestDraw();
    }

    _eventPressure(e) {
        if (!e || e.pointerType === "mouse") return 1;
        return e.pressure > 0 ? e.pressure : 0.5;
    }

    _pointerImageXY(ev) {
        const c = this.dom.canvas;
        if (!c) return { x: 0, y: 0 };
        const r = c.getBoundingClientRect();
        return {
            x: (ev.clientX - r.left - this.panX) / this.zoom,
            y: (ev.clientY - r.top - this.panY) / this.zoom,
        };
    }

    _beginBrushStroke(ix, iy, e) {
        this.strokeStart = new Uint8Array(this.mask);
        this.coverage = new Uint8Array(this.mask.length);
        this._strokeBounds = null;
        this._strokeCommitted = false;
        this._strokePointerType = e?.pointerType || "mouse";
        this._lastPressure = this._eventPressure(e);
        this._stampAt(ix, iy, this._lastPressure, this._strokePointerType);
    }

    _stampAt(cx, cy, pressure, pointerType = "mouse") {
        const erase = this.tool === "eraser" || this.subtract;
        let radius = this.brushSize / 2;
        let opacity = this.brushOpacity;
        if (pointerType !== "mouse" && (this.pressureSize || this.pressureOpacity)) {
            const s = tools.stampPressure(
                pressure, this.brushSize, this.brushOpacity,
                this.pressureSize, this.pressureOpacity,
            );
            radius = s.radius;
            opacity = s.opacity;
        }
        const b = tools.stampBrushCoverage(
            this.mask, this.strokeStart, this.coverage,
            this.editW, this.editH, cx, cy,
            radius, this.brushHardness, opacity, erase,
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
        } else if (this.tool === "sam" && this.samDragStart) {
            // Point or box is only known once the pointer is released: a press that travelled more than 4 image
            // px is a box and adds NO point (a point at the box corner would select the background around it).
            const start = this.samDragStart;
            if (Math.hypot(ix - start.x, iy - start.y) > 4) {
                this.samBox = [Math.min(start.x, ix), Math.min(start.y, iy), Math.max(start.x, ix), Math.max(start.y, iy)];
            } else {
                this.samPoints.push({ x: start.x, y: start.y, label: start.alt ? 0 : 1 });
            }
            this.samDragStart = null;
            this.shapeCur = null;
            this.painting = false;
            this._requestSam();
        } else if (this.tool === "refine") {
            this._runRefine();
            this._refineStrokeStart = null;
            this._refineCoverage = null;
            this._refineStrokeBounds = null;
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
        if (this.tool === "sam") {
            // Alt is read here, at press time (released-before-pointerup must not flip a negative click).
            this.painting = true;
            this.samDragStart = { x: ix, y: iy, alt: !!e.altKey };
            this.shapeCur = null;
            return;
        }
        this.painting = true;
        this.lastPt = { x: ix, y: iy };
        if (this.tool === "brush" || this.tool === "eraser") {
            this._beginBrushStroke(ix, iy, e);
        } else if (this.tool === "refine") {
            this._beginRefineStroke(ix, iy);
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
        if (this.tool === "sam" && this.painting && this.samDragStart) {
            this.shapeCur = { x: ix, y: iy };       // live box outline while dragging
        }
        if (this.tool === "colour" || this.tool === "sam") {
            this.requestDraw();
            return;
        }
        if (!this.painting && (this.tool === "brush" || this.tool === "eraser" || this.tool === "refine")) {
            this.requestDraw();
        }
        if (this.panning && this.panAnchor) {
            this.panX = e.clientX - this.panAnchor.x;
            this.panY = e.clientY - this.panAnchor.y;
            this.requestDraw();
            return;
        }
        if (!this.painting) return;
        if (this.tool === "brush" || this.tool === "eraser") {
            const events = e.getCoalescedEvents?.() ?? [e];
            let prev = this.lastPt;
            let prevP = this._lastPressure;
            const pt = this._strokePointerType || "mouse";
            for (const ev of events) {
                const p = this._pointerImageXY(ev);
                const pr = this._eventPressure(ev);
                if (prev) {
                    const b = tools.lineBrushPressure(
                        (cx, cy, pressure) => this._stampAt(cx, cy, pressure, pt),
                        prev.x, prev.y, prevP, p.x, p.y, pr,
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
                } else {
                    this._stampAt(p.x, p.y, pr, pt);
                }
                prev = p;
                prevP = pr;
            }
            this.lastPt = prev;
            this._lastPressure = prevP;
            this.requestDraw();
        } else if (this.tool === "refine") {
            tools.lineBrush(
                (cx, cy) => this._stampRefineAt(cx, cy),
                this.lastPt.x, this.lastPt.y, ix, iy,
                this.brushSize / 2, Math.max(1, this.brushSize * this.brushSpacing),
            );
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
        this.viewMode = (this.viewMode + 1) % 5;
        try { localStorage.setItem(VIEW_MODE_KEY, String(this.viewMode)); } catch (_) { /* ignore */ }
        this.requestDraw();
    }

    _drawNow() {
        const c = this.dom.canvas;
        if (!c) return;
        this._flushOutlineBuf();
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
        const vm = this.viewMode;
        if (vm !== 1 && this.imageBitmap) {
            ctx.drawImage(this.imageBitmap, 0, 0, this.editW, this.editH);
        }
        ctx.imageSmoothingEnabled = this.zoom < 1;
        if (vm === 1) {
            this._drawViewTiles(ctx, "matte", null, vw, vh);
        } else if (vm === 0) {
            this._drawViewTiles(ctx, "overlay", null, vw, vh);
        } else if (vm === 2) {
            this._drawViewTiles(ctx, "rubylith", null, vw, vh);
        } else if (vm === 3 && this.zoom < 2) {
            ctx.imageSmoothingEnabled = false;
            this._drawViewTiles(ctx, "outline", null, vw, vh);
        }
        if (vm !== 4 && this.candidatePreview) {
            this._drawViewTiles(ctx, "preview", this.candidatePreview, vw, vh);
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
        if (this.tool === "sam" && this.samBox) {
            const [x0, y0, x1, y1] = this.samBox;
            ctx.strokeStyle = C.accent2;
            ctx.setLineDash([4 / this.zoom, 4 / this.zoom]);
            ctx.lineWidth = 1.5 / this.zoom;
            ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
            ctx.setLineDash([]);
            ctx.lineWidth = 1;
        }
        if (this.tool === "sam" && this.samDragStart && this.shapeCur) {
            const x0 = this.samDragStart.x;
            const y0 = this.samDragStart.y;
            const x1 = this.shapeCur.x;
            const y1 = this.shapeCur.y;
            if (Math.hypot(x1 - x0, y1 - y0) > 4) {
                ctx.strokeStyle = C.accent2;
                ctx.setLineDash([4 / this.zoom, 4 / this.zoom]);
                ctx.lineWidth = 1.5 / this.zoom;
                ctx.strokeRect(
                    Math.min(x0, x1), Math.min(y0, y1),
                    Math.abs(x1 - x0), Math.abs(y1 - y0),
                );
                ctx.setLineDash([]);
                ctx.lineWidth = 1;
            }
        }
        if (this.tool === "sam" && this.samPoints.length) {
            const arm = 5 / this.zoom;
            for (const p of this.samPoints) {
                const cx = p.x + 0.5;
                const cy = p.y + 0.5;
                const fg = p.label === 1;
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
                ctx.strokeStyle = fg ? "#6bdc6b" : C.danger;
                ctx.stroke();
            }
            ctx.lineWidth = 1;
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
        if (this.hover && (this.tool === "brush" || this.tool === "eraser" || this.tool === "refine")) {
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
        if (vm === 3 && this.zoom >= 2 && this._outlineBuf && this.mask) {
            const { ix0, iy0, ix1, iy1 } = this._visibleImageRect(vw, vh);
            const z = this.zoom;
            const px = this.panX;
            const py = this.panY;
            const iw = this.editW;
            const ih = this.editH;
            const mask = this.mask;
            const buf = this._outlineBuf;
            // Pixel edges are snapped to whole screen pixels (neighbours share the exact same edge), so the line
            // is continuous and crisp at fractional zoom; fractional rects blended to pink/brown at 2.3x.
            const white = [];
            const black = [];
            for (let y = iy0; y <= iy1; y++) {
                const row = y * iw;
                const T = Math.round(y * z + py);
                const B = Math.round((y + 1) * z + py);
                for (let x = ix0; x <= ix1; x++) {
                    if (!buf[row + x]) continue;
                    const L = Math.round(x * z + px);
                    const R = Math.round((x + 1) * z + px);
                    if (y <= 0 || mask[row - iw + x] < 128) { white.push(L, T, R - L, 1); black.push(L, T - 1, R - L, 1); }
                    if (y >= ih - 1 || mask[row + iw + x] < 128) { white.push(L, B - 1, R - L, 1); black.push(L, B, R - L, 1); }
                    if (x <= 0 || mask[row + x - 1] < 128) { white.push(L, T, 1, B - T); black.push(L - 1, T, 1, B - T); }
                    if (x >= iw - 1 || mask[row + x + 1] < 128) { white.push(R - 1, T, 1, B - T); black.push(R, T, 1, B - T); }
                }
            }
            ctx.fillStyle = "#000000";
            for (let i = 0; i < black.length; i += 4) ctx.fillRect(black[i], black[i + 1], black[i + 2], black[i + 3]);
            ctx.fillStyle = "#ffffff";
            for (let i = 0; i < white.length; i += 4) ctx.fillRect(white[i], white[i + 1], white[i + 2], white[i + 3]);
        }
        const grid = gridLines(this.zoom, this.panX, this.panY, vw, vh);
        if (grid.xs.length) {
            const g = this._gridTint;
            ctx.strokeStyle = `rgba(${g.r},${g.g},${g.b},0.35)`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (const x of grid.xs) {
                ctx.moveTo(x, 0);
                ctx.lineTo(x, vh);
            }
            for (const y of grid.ys) {
                ctx.moveTo(0, y);
                ctx.lineTo(vw, y);
            }
            ctx.stroke();
        }
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
        this.samAbort?.abort();
        this.refineAbort?.abort();
        if (this.editorId) api.release(this.editorId);
        this._releaseBitmap();
        this.masks.clear();
        this.storedFrames.clear();
        this.dirtyFrames.clear();
        this.cleanLru = [];
        this.mask = null;
        this._tileCache?.dispose();
        this._tileCache = null;
        this._outlineBuf = null;
        this._outlineDirty = null;
        this._outlineBufStale = false;
        this.history = null;
        this._maskClipboard = null;
        this.node = null;
    }
}
