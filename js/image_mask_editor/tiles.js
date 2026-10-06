/**
 * Tiled LRU cache for Image Mask Editor view layers (DOM-free, Node-testable).
 */

export const TILE = 512;
export const DEFAULT_BYTE_BUDGET = 64 * TILE * TILE * 4;
const LOD_MARGIN = 4;

/** @typedef {{ tx: number, ty: number, x: number, y: number, w: number, h: number, level: number, scale: number, imageW: number, imageH: number }} TileDesc */

export function lodLevelForZoom(zoom) {
    if (!(zoom > 0)) return 0;
    const inv = 1 / zoom;
    let l = 0;
    while ((1 << (l + 1)) <= inv) l++;
    return l;
}

export function tileSpan(level) {
    return TILE << (level | 0);
}

export function tileCanvasSize(imageW, imageH, level) {
    const scale = 1 << (level | 0);
    return {
        w: Math.min(TILE, Math.ceil(imageW / scale)),
        h: Math.min(TILE, Math.ceil(imageH / scale)),
    };
}

/**
 * List tile descriptors intersecting an inclusive image rect at LOD level.
 * @returns {TileDesc[]}
 */
export function tilesForRect(x0, y0, x1, y1, w, h, level = 0) {
    const rx0 = Math.max(0, x0 | 0);
    const ry0 = Math.max(0, y0 | 0);
    const rx1 = Math.min((w | 0) - 1, x1 | 0);
    const ry1 = Math.min((h | 0) - 1, y1 | 0);
    if (rx1 < rx0 || ry1 < ry0 || w <= 0 || h <= 0) return [];

    const L = level | 0;
    const span = tileSpan(L);
    const scale = 1 << L;
    const tx0 = Math.floor(rx0 / span);
    const tx1 = Math.floor(rx1 / span);
    const ty0 = Math.floor(ry0 / span);
    const ty1 = Math.floor(ry1 / span);
    const out = [];
    for (let ty = ty0; ty <= ty1; ty++) {
        const y = ty * span;
        const imageH = Math.min(span, h - y);
        for (let tx = tx0; tx <= tx1; tx++) {
            const x = tx * span;
            const imageW = Math.min(span, w - x);
            const { w: tw, h: th } = tileCanvasSize(imageW, imageH, L);
            out.push({
                tx, ty, x, y, w: tw, h: th,
                level: L, scale, imageW, imageH,
            });
        }
    }
    return out;
}

export function tileKey(view, level, tx, ty) {
    return `${view}:${level | 0}:${tx | 0}:${ty | 0}`;
}

export function maxTilesForBudget(byteBudget, margin = LOD_MARGIN) {
    return Math.max(1, Math.floor(byteBudget / (TILE * TILE * 4)) - margin);
}

/** Raise LOD until visible tile count fits the byte budget (minus margin). */
export function pickLodLevel(zoom, ix0, iy0, ix1, iy1, w, h, byteBudget, margin = LOD_MARGIN) {
    let L = lodLevelForZoom(zoom);
    const cap = maxTilesForBudget(byteBudget, margin);
    while (L < 24) {
        if (tilesForRect(ix0, iy0, ix1, iy1, w, h, L).length <= cap) return L;
        L++;
    }
    return L;
}

function _tileBytes(tile) {
    return (tile.w | 0) * (tile.h | 0) * 4;
}

function _unionDirty(tile, x0, y0, x1, y1) {
    const d = tile.dirty;
    if (!d) {
        tile.dirty = { x0, y0, x1, y1 };
        return;
    }
    d.x0 = Math.min(d.x0, x0);
    d.y0 = Math.min(d.y0, y0);
    d.x1 = Math.max(d.x1, x1);
    d.y1 = Math.max(d.y1, y1);
}

function _fullDirty(tile) {
    tile.dirty = { x0: 0, y0: 0, x1: tile.w - 1, y1: tile.h - 1 };
}

/** Image dirty rect -> tile-local pixel rect (inclusive) at level L. */
export function dirtyTileRect(x0, y0, x1, y1, t) {
    const scale = t.scale;
    const ix0 = t.x;
    const iy0 = t.y;
    const ix1 = ix0 + t.imageW - 1;
    const iy1 = iy0 + t.imageH - 1;
    const cx0 = Math.max(x0, ix0);
    const cy0 = Math.max(y0, iy0);
    const cx1 = Math.min(x1, ix1);
    const cy1 = Math.min(y1, iy1);
    if (cx1 < cx0 || cy1 < cy0) return null;
    const u0 = Math.max(0, Math.floor((cx0 - ix0) / scale));
    const v0 = Math.max(0, Math.floor((cy0 - iy0) / scale));
    const u1 = Math.min(t.w - 1, Math.floor((cx1 - ix0) / scale));
    const v1 = Math.min(t.h - 1, Math.floor((cy1 - iy0) / scale));
    if (u1 < u0 || v1 < v0) return null;
    return { x0: u0, y0: v0, x1: u1, y1: v1 };
}

export class TileCache {
    /**
     * @param {{ tileSize?: number, byteBudget?: number, createTile?: (w:number,h:number)=>object, imgW?: number, imgH?: number }} [opts]
     */
    constructor(opts = {}) {
        this.tileSize = opts.tileSize ?? TILE;
        this.byteBudget = opts.byteBudget ?? DEFAULT_BYTE_BUDGET;
        this._createTile = opts.createTile ?? null;
        this.imgW = opts.imgW ?? 0;
        this.imgH = opts.imgH ?? 0;
        /** @type {Map<string, object>} */
        this._map = new Map();
        /** @type {string[]} LRU oldest-first */
        this._lru = [];
        this._bytes = 0;
        /** @type {Set<number>} */
        this._levels = new Set();
    }

    setImageSize(w, h) {
        this.dispose();
        this.imgW = w | 0;
        this.imgH = h | 0;
    }

    bytesUsed() {
        return this._bytes;
    }

    cachedLevels() {
        return [...this._levels];
    }

    _touch(key) {
        const i = this._lru.indexOf(key);
        if (i >= 0) this._lru.splice(i, 1);
        this._lru.push(key);
    }

    _evict() {
        while (this._bytes > this.byteBudget && this._lru.length > 0) {
            const key = this._lru.shift();
            const tile = this._map.get(key);
            if (!tile) continue;
            this._releaseTile(tile);
            this._map.delete(key);
            this._bytes -= _tileBytes(tile);
            if (this._map.size === 0) this._levels.clear();
        }
    }

    _releaseTile(tile) {
        const c = tile.canvas;
        if (c) {
            c.width = 0;
            c.height = 0;
        }
    }

    get(view, level, tx, ty) {
        const key = tileKey(view, level, tx, ty);
        const tile = this._map.get(key);
        if (tile) this._touch(key);
        return tile ?? null;
    }

    set(view, level, tx, ty, tile) {
        const key = tileKey(view, level, tx, ty);
        const prev = this._map.get(key);
        if (prev) {
            this._bytes -= _tileBytes(prev);
            // A tile refreshed in place is stored again under its own key: releasing it would zero the canvas
            // that is about to be drawn ("drawImage ... width or height of 0", seen in the 8K e2e).
            if (prev !== tile) this._releaseTile(prev);
            const i = this._lru.indexOf(key);
            if (i >= 0) this._lru.splice(i, 1);
        }
        tile.dirty = null;
        tile.bytes = _tileBytes(tile);
        this._map.set(key, tile);
        this._levels.add(level | 0);
        this._bytes += tile.bytes;
        this._touch(key);
        this._evict();
    }

    markStaleRect(x0, y0, x1, y1) {
        if (!this.imgW || !this.imgH) return;
        const levels = new Set([0, ...this._levels]);
        for (const L of levels) {
            const tiles = tilesForRect(x0, y0, x1, y1, this.imgW, this.imgH, L);
            if (!tiles.length) continue;
            const hit = new Set(tiles.map((t) => `${t.tx}:${t.ty}`));
            for (const [key, tile] of this._map.entries()) {
                const parts = key.split(":");
                if ((parts[1] | 0) !== L) continue;
                const suf = `${parts[2]}:${parts[3]}`;
                if (!hit.has(suf)) continue;
                const t = tiles.find((tt) => tt.tx === (parts[2] | 0) && tt.ty === (parts[3] | 0));
                if (!t) continue;
                const dr = dirtyTileRect(x0, y0, x1, y1, t);
                if (dr) _unionDirty(tile, dr.x0, dr.y0, dr.x1, dr.y1);
            }
        }
    }

    markAllStale() {
        for (const tile of this._map.values()) _fullDirty(tile);
    }

    markViewStale(view) {
        const prefix = `${view}:`;
        for (const [key, tile] of this._map.entries()) {
            if (key.startsWith(prefix)) _fullDirty(tile);
        }
    }

    dispose() {
        for (const tile of this._map.values()) this._releaseTile(tile);
        this._map.clear();
        this._lru = [];
        this._bytes = 0;
        this._levels.clear();
    }
}
