/** Dirty-rect undo history capped at 128 MB (undo + redo). */

const MAX_BYTES = 128 * 1024 * 1024;

export class UndoStack {
    constructor() {
        this.undo = [];
        this.redo = [];
        this.bytes = 0;
    }

    _entryBytes(e) {
        return e.before.byteLength + e.after.byteLength;
    }

    _evict() {
        while (this.bytes > MAX_BYTES && this.undo.length > 1) {
            const e = this.undo.shift();
            if (e) this.bytes -= this._entryBytes(e);
        }
        if (this.bytes > MAX_BYTES) {
            for (const e of this.redo) {
                this.bytes -= this._entryBytes(e);
            }
            this.redo = [];
        }
    }

    push(entry) {
        const cost = this._entryBytes(entry);
        this.undo.push(entry);
        this.bytes += cost;
        for (const e of this.redo) {
            this.bytes -= this._entryBytes(e);
        }
        this.redo = [];
        this._evict();
    }

    popUndo() {
        const e = this.undo.pop();
        if (e) this.bytes -= this._entryBytes(e);
        return e || null;
    }

    pushRedo(entry) {
        this.redo.push(entry);
        this.bytes += this._entryBytes(entry);
        this._evict();
    }

    popRedo() {
        const e = this.redo.pop();
        if (e) this.bytes -= this._entryBytes(e);
        return e || null;
    }
}

export function captureRect(mask, w, h, x0, y0, x1, y1) {
    const x = Math.max(0, Math.min(x0, x1) | 0);
    const y = Math.max(0, Math.min(y0, y1) | 0);
    const x2 = Math.min(w, (Math.max(x0, x1) | 0) + 1);
    const y2 = Math.min(h, (Math.max(y0, y1) | 0) + 1);
    const rw = Math.max(0, x2 - x);
    const rh = Math.max(0, y2 - y);
    const buf = new Uint8Array(rw * rh);
    for (let yy = 0; yy < rh; yy++) {
        const src = (y + yy) * w + x;
        buf.set(mask.subarray(src, src + rw), yy * rw);
    }
    return { x, y, w: rw, h: rh, data: buf };
}

export function applyRect(mask, w, entry) {
    const data = entry.data ?? entry.before ?? entry.after;
    if (!data) return;
    for (let yy = 0; yy < entry.h; yy++) {
        const row = (entry.y + yy) * w + entry.x;
        for (let xx = 0; xx < entry.w; xx++) {
            mask[row + xx] = data[yy * entry.w + xx];
        }
    }
}

export function pushUndoEntry(undo, mask, w, h, x0, y0, x1, y1, beforeSrc) {
    const after = captureRect(mask, w, h, x0, y0, x1, y1);
    const before = beforeSrc
        ? captureRect(beforeSrc, w, h, x0, y0, x1, y1)
        : captureRect(mask, w, h, x0, y0, x1, y1);
    undo.push({
        x: after.x,
        y: after.y,
        w: after.w,
        h: after.h,
        before: before.data,
        after: after.data,
    });
    return after;
}
