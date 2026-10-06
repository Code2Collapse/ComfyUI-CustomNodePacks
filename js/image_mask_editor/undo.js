/** Dirty-rect undo history capped at 128 MB (undo + redo). */

const MAX_BYTES = 128 * 1024 * 1024;
const GLOBAL_MAX_BYTES = 128 * 1024 * 1024;

export class UndoStack {
    constructor(maxBytes = MAX_BYTES) {
        this.maxBytes = maxBytes;
        this.undo = [];
        this.redo = [];
        this.bytes = 0;
    }

    _entryBytes(e) {
        return e.before.byteLength + e.after.byteLength;
    }

    _evict() {
        if (!Number.isFinite(this.maxBytes)) return;
        while (this.bytes > this.maxBytes && this.undo.length > 1) {
            const e = this.undo.shift();
            if (e) this.bytes -= this._entryBytes(e);
        }
        if (this.bytes > this.maxBytes) {
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

    reinstateUndo(entry) {
        this.undo.push(entry);
        this.bytes += this._entryBytes(entry);
    }
}

/** One UndoStack per frame under a single global byte budget. */
export class FrameHistory {
    constructor(maxBytes = GLOBAL_MAX_BYTES) {
        this.maxBytes = maxBytes;
        this.stacks = new Map();
        this.lastPushOrder = new Map();
        this._pushSeq = 0;
        this.bytes = 0;
    }

    _stack(frame) {
        const f = frame | 0;
        if (!this.stacks.has(f)) {
            this.stacks.set(f, new UndoStack(Infinity));
        }
        return this.stacks.get(f);
    }

    _recalcBytes() {
        let b = 0;
        for (const s of this.stacks.values()) b += s.bytes;
        this.bytes = b;
    }

    push(frame, entry) {
        const f = frame | 0;
        this._stack(f).push(entry);
        this.lastPushOrder.set(f, ++this._pushSeq);
        this._recalcBytes();
        this._evict(f);
    }

    _evict(currentFrame) {
        const cur = currentFrame | 0;
        while (this.bytes > this.maxBytes) {
            let victim = null;
            let oldest = Infinity;
            for (const [f, s] of this.stacks) {
                if (f === cur || s.undo.length === 0) continue;
                const order = this.lastPushOrder.get(f) ?? 0;
                if (order < oldest) {
                    oldest = order;
                    victim = f;
                }
            }
            if (victim != null) {
                const vs = this.stacks.get(victim);
                const e = vs.undo.shift();
                if (e) vs.bytes -= vs._entryBytes(e);
                this._recalcBytes();
                continue;
            }
            const cs = this._stack(cur);
            if (cs.undo.length > 1) {
                const e = cs.undo.shift();
                if (e) cs.bytes -= cs._entryBytes(e);
                this._recalcBytes();
                continue;
            }
            break;
        }
    }

    undo(frame) {
        const e = this._stack(frame).popUndo();
        this._recalcBytes();
        return e;
    }

    redo(frame) {
        const s = this._stack(frame);
        const e = s.popRedo();
        if (!e) return null;
        s.reinstateUndo(e);
        this.lastPushOrder.set(frame | 0, ++this._pushSeq);
        this._recalcBytes();
        this._evict(frame | 0);
        return e;
    }

    pushRedo(frame, entry) {
        this._stack(frame).pushRedo(entry);
        this._recalcBytes();
        this._evict(frame | 0);
    }

    pushUndo(frame, entry) {
        this._stack(frame).push(entry);
        this.lastPushOrder.set(frame | 0, ++this._pushSeq);
        this._recalcBytes();
        this._evict(frame | 0);
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

export function pushUndoEntry(a, b, c, d, e, f, g, h, i, j) {
    if (a instanceof FrameHistory) {
        const frame = b;
        const mask = c;
        const w = d;
        const ht = e;
        const x0 = f;
        const y0 = g;
        const x1 = h;
        const y1 = i;
        const beforeSrc = j;
        const after = captureRect(mask, w, ht, x0, y0, x1, y1);
        const before = beforeSrc
            ? captureRect(beforeSrc, w, ht, x0, y0, x1, y1)
            : captureRect(mask, w, ht, x0, y0, x1, y1);
        a.push(frame, {
            x: after.x,
            y: after.y,
            w: after.w,
            h: after.h,
            before: before.data,
            after: after.data,
        });
        return after;
    }
    const undo = a;
    const mask = b;
    const w = c;
    const ht = d;
    const x0 = e;
    const y0 = f;
    const x1 = g;
    const y1 = h;
    const beforeSrc = i;
    const after = captureRect(mask, w, ht, x0, y0, x1, y1);
    const before = beforeSrc
        ? captureRect(beforeSrc, w, ht, x0, y0, x1, y1)
        : captureRect(mask, w, ht, x0, y0, x1, y1);
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
