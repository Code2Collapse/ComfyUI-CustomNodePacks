/** Virtualized horizontal frame strip for multi-frame mask editing. */
import { C } from "./palette.js";

const CELL_W = 96;
const CELL_H = 54;
const PAD = 2;
const VISIBLE_PAD = 2;
const BITMAP_CAP = 96;

export class FrameStrip {
    constructor() {
        this.host = null;
        this.ed = null;
        this.onUpdate = null;
        this.scrollEl = null;
        this.inner = null;
        this.cells = new Map();
        this._bitmaps = new Map();
        this._lru = [];
        this._inflight = new Map();
        this._failed = new Set();
        this._ro = null;
        this._onScroll = null;
    }

    mount(host, ed, onUpdate) {
        this.host = host;
        this.ed = ed;
        this.onUpdate = onUpdate;
        this.scrollEl = document.createElement("div");
        this.scrollEl.style.cssText = [
            `overflow-x:auto;overflow-y:hidden;white-space:nowrap;padding:6px 8px;`,
            `background:${C.panel};border-top:1px solid ${C.border};`,
        ].join("");
        this.inner = document.createElement("div");
        this.inner.style.cssText = "position:relative;height:54px;";
        this.scrollEl.appendChild(this.inner);
        host.appendChild(this.scrollEl);
        this._onScroll = () => this._render();
        this.scrollEl.addEventListener("scroll", this._onScroll);
        this._ro = new ResizeObserver(() => this._render());
        this._ro.observe(this.scrollEl);
        this._render();
    }

    refresh() {
        this._render();
        this._scrollCurrentIntoView();
    }

    dispose() {
        try { this._ro?.disconnect(); } catch (_) { /* ignore */ }
        if (this._onScroll && this.scrollEl) {
            this.scrollEl.removeEventListener("scroll", this._onScroll);
        }
        for (const bm of this._bitmaps.values()) {
            try { bm.close(); } catch (_) { /* ignore */ }
        }
        this._bitmaps.clear();
        this._lru = [];
        this._inflight.clear();
        this._failed.clear();
        this.cells.clear();
        try { this.scrollEl?.remove(); } catch (_) { /* ignore */ }
        this.host = null;
        this.ed = null;
    }

    _thumbUrl(i) {
        const src = this.ed?.frameSource;
        if (!src) return "";
        if (typeof src.thumbUrl === "function") return src.thumbUrl(i);
        if (src.kind === "plan") {
            const tok = src.url(0).match(/token=([^&]+)/);
            if (tok) {
                return `/c2c/frames/thumb?token=${encodeURIComponent(tok[1])}&i=${i | 0}&max=128&fmt=jpeg`;
            }
        }
        return src.url(i);
    }

    _visibleRange(count, scrollLeft, viewW) {
        const first = Math.max(0, Math.floor(scrollLeft / (CELL_W + PAD)) - VISIBLE_PAD);
        const last = Math.min(
            count - 1,
            Math.ceil((scrollLeft + viewW) / (CELL_W + PAD)) + VISIBLE_PAD,
        );
        return { first, last };
    }

    _evictBitmap(i) {
        const bm = this._bitmaps.get(i);
        if (bm) {
            try { bm.close(); } catch (_) { /* ignore */ }
            this._bitmaps.delete(i);
        }
        const p = this._lru.indexOf(i);
        if (p >= 0) this._lru.splice(p, 1);
    }

    _touchBitmap(i, bm) {
        this._evictBitmap(i);
        this._bitmaps.set(i, bm);
        this._lru.push(i);
        while (this._lru.length > BITMAP_CAP) {
            this._evictBitmap(this._lru[0]);
        }
    }

    async _loadBitmap(i) {
        if (this._bitmaps.has(i)) return this._bitmaps.get(i);
        if (this._failed.has(i)) return null;
        const pending = this._inflight.get(i);
        if (pending) return pending;

        const p = (async () => {
            const url = this._thumbUrl(i);
            if (!url) {
                this._failed.add(i);
                return null;
            }
            try {
                const r = await fetch(url);
                if (!r.ok) {
                    this._failed.add(i);
                    return null;
                }
                const bm = await createImageBitmap(await r.blob(), {
                    resizeWidth: CELL_W,
                    resizeQuality: "low",
                });
                this._touchBitmap(i, bm);
                return bm;
            } catch (_) {
                this._failed.add(i);
                return null;
            } finally {
                this._inflight.delete(i);
            }
        })();
        this._inflight.set(i, p);
        return p;
    }

    _scrollCurrentIntoView() {
        if (!this.scrollEl || !this.ed) return;
        const i = this.ed.curFrame | 0;
        const x = i * (CELL_W + PAD);
        const left = this.scrollEl.scrollLeft;
        const w = this.scrollEl.clientWidth;
        if (x < left) this.scrollEl.scrollLeft = x;
        else if (x + CELL_W > left + w) this.scrollEl.scrollLeft = x + CELL_W - w;
    }

    _render() {
        const ed = this.ed;
        if (!ed || !this.inner || !this.scrollEl) return;
        const count = ed.frameCount | 0;
        this.inner.style.width = `${Math.max(0, count * (CELL_W + PAD) - PAD)}px`;
        const { first, last } = this._visibleRange(
            count,
            this.scrollEl.scrollLeft,
            this.scrollEl.clientWidth,
        );
        const need = new Set();
        for (let i = first; i <= last; i++) need.add(i);
        for (const [idx, el] of this.cells) {
            if (!need.has(idx)) {
                el.remove();
                this.cells.delete(idx);
            }
        }
        for (let i = first; i <= last; i++) {
            let cell = this.cells.get(i);
            if (!cell) {
                cell = this._makeCell(i);
                this.cells.set(i, cell);
                this.inner.appendChild(cell);
            }
            this._styleCell(cell, i);
            this._loadBitmap(i).then((bm) => {
                if (!cell.isConnected || this.cells.get(i) !== cell) return;
                const cv = cell.querySelector("canvas");
                if (cv && bm) {
                    const cx = cv.getContext("2d");
                    cx.clearRect(0, 0, CELL_W, CELL_H);
                    const scale = Math.min(CELL_W / bm.width, CELL_H / bm.height);
                    const dw = bm.width * scale;
                    const dh = bm.height * scale;
                    cx.drawImage(bm, (CELL_W - dw) / 2, (CELL_H - dh) / 2, dw, dh);
                }
            });
        }
    }

    _makeCell(i) {
        const cell = document.createElement("button");
        cell.type = "button";
        cell.dataset.frame = String(i);
        cell.style.cssText = [
            `position:absolute;top:0;width:${CELL_W}px;height:${CELL_H}px;`,
            `padding:0;border:1px solid ${C.border};background:${C.bg};cursor:pointer;`,
            `border-radius:3px;overflow:hidden;`,
        ].join("");
        const cv = document.createElement("canvas");
        cv.width = CELL_W;
        cv.height = CELL_H;
        cv.style.cssText = "display:block;width:100%;height:100%;";
        const label = document.createElement("span");
        label.style.cssText = [
            `position:absolute;left:3px;bottom:2px;font:10px/1 sans-serif;color:${C.white};`,
            `text-shadow:0 0 2px ${C.black};pointer-events:none;`,
        ].join("");
        label.textContent = String(i + 1);
        const dot = document.createElement("span");
        dot.className = "ime-strip-dot";
        dot.style.cssText = [
            `position:absolute;right:4px;top:4px;width:6px;height:6px;border-radius:50%;`,
            `background:${C.accent};display:none;pointer-events:none;`,
        ].join("");
        cell.append(cv, label, dot);
        cell.onclick = async () => {
            if (!this.ed) return;
            await this.ed._switchFrame(i);
            this.onUpdate?.();
            this.refresh();
        };
        return cell;
    }

    _styleCell(cell, i) {
        const ed = this.ed;
        cell.style.left = `${i * (CELL_W + PAD)}px`;
        cell.style.outline = i === ed.curFrame ? `2px solid ${C.accent}` : "";
        const dot = cell.querySelector(".ime-strip-dot");
        if (dot) dot.style.display = ed.frameHasMask(i) ? "block" : "none";
    }
}
