/**
 * Raster tools operating on Uint8 mask buffers.
 * Image coordinates are continuous (pixel k spans [k, k+1)); every kernel samples at pixel CENTRES
 * (k + 0.5), so a shape covers the pixels whose centre it contains and stays centred where it was drawn.
 */

export function stampBrushCoverage(
    mask, strokeStart, coverage, w, h, cx, cy, radius, hardness, opacity, erase,
) {
    const r2 = radius * radius;
    // hardness 1 = hard disc, 0 = linear falloff from the centre (was inverted: 1 painted the softest cone)
    const featherStart = radius * Math.max(0, Math.min(1, hardness));
    const x0 = Math.max(0, Math.floor(cx - radius));
    const y0 = Math.max(0, Math.floor(cy - radius));
    const x1 = Math.min(w - 1, Math.ceil(cx + radius));
    const y1 = Math.min(h - 1, Math.ceil(cy + radius));
    const op = Math.max(0, Math.min(1, opacity));
    for (let yy = y0; yy <= y1; yy++) {
        const dy = yy + 0.5 - cy;
        for (let xx = x0; xx <= x1; xx++) {
            const dx = xx + 0.5 - cx;
            const d2 = dx * dx + dy * dy;
            if (d2 > r2) continue;
            const d = Math.sqrt(d2);
            let falloff = 1;
            if (d > featherStart) {
                falloff = 1 - (d - featherStart) / Math.max(1e-6, radius - featherStart);
            }
            const stamp = Math.round(255 * falloff);
            const i = yy * w + xx;
            if (stamp > coverage[i]) coverage[i] = stamp;
            const cov = coverage[i];
            if (erase) {
                mask[i] = Math.round(strokeStart[i] * (1 - (cov * op) / 255));
            } else {
                const painted = Math.round(cov * op);
                mask[i] = painted > strokeStart[i] ? painted : strokeStart[i];
            }
        }
    }
    return { x0, y0, x1, y1 };
}

export function lineBrush(stampFn, x0, y0, x1, y1, radius, spacing) {
    const dist = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(1, Math.ceil(dist / Math.max(1, spacing)));
    let bx0 = Infinity, by0 = Infinity, bx1 = -1, by1 = -1;
    for (let s = 0; s <= steps; s++) {
        const t = steps === 0 ? 0 : s / steps;
        const cx = x0 + (x1 - x0) * t;
        const cy = y0 + (y1 - y0) * t;
        const b = stampFn(cx, cy);
        if (b) {
            bx0 = Math.min(bx0, b.x0); by0 = Math.min(by0, b.y0);
            bx1 = Math.max(bx1, b.x1); by1 = Math.max(by1, b.y1);
        }
    }
    if (bx1 < bx0) return null;
    return { x0: bx0, y0: by0, x1: bx1, y1: by1 };
}

export function fillRect(mask, w, h, x0, y0, x1, y1, value, square) {
    let ax0 = Math.min(x0, x1), ay0 = Math.min(y0, y1);
    let ax1 = Math.max(x0, x1), ay1 = Math.max(y0, y1);
    if (square) {
        const s = Math.min(ax1 - ax0, ay1 - ay0);
        ax1 = ax0 + s; ay1 = ay0 + s;
    }
    const ix0 = Math.max(0, Math.ceil(ax0 - 0.5));
    const iy0 = Math.max(0, Math.ceil(ay0 - 0.5));
    const ix1 = Math.min(w, Math.ceil(ax1 - 0.5));
    const iy1 = Math.min(h, Math.ceil(ay1 - 0.5));
    for (let yy = iy0; yy < iy1; yy++) {
        const row = yy * w;
        for (let xx = ix0; xx < ix1; xx++) mask[row + xx] = value;
    }
    return { x0: ix0, y0: iy0, x1: Math.max(ix0, ix1 - 1), y1: Math.max(iy0, iy1 - 1) };
}

export function fillEllipse(mask, w, h, x0, y0, x1, y1, value, square) {
    let ax0 = Math.min(x0, x1), ay0 = Math.min(y0, y1);
    let ax1 = Math.max(x0, x1), ay1 = Math.max(y0, y1);
    if (square) {
        const s = Math.min(ax1 - ax0, ay1 - ay0);
        ax1 = ax0 + s; ay1 = ay0 + s;
    }
    const cx = (ax0 + ax1) / 2, cy = (ay0 + ay1) / 2;
    const rx = Math.max(0.5, (ax1 - ax0) / 2), ry = Math.max(0.5, (ay1 - ay0) / 2);
    const ix0 = Math.max(0, Math.floor(cx - rx));
    const iy0 = Math.max(0, Math.floor(cy - ry));
    const ix1 = Math.min(w, Math.ceil(cx + rx));
    const iy1 = Math.min(h, Math.ceil(cy + ry));
    for (let yy = iy0; yy < iy1; yy++) {
        for (let xx = ix0; xx < ix1; xx++) {
            const dx = (xx + 0.5 - cx) / rx, dy = (yy + 0.5 - cy) / ry;
            if (dx * dx + dy * dy <= 1) mask[yy * w + xx] = value;
        }
    }
    return { x0: ix0, y0: iy0, x1: ix1 - 1, y1: iy1 - 1 };
}

export function fillPolygon(mask, w, h, pts, value) {
    if (pts.length < 3) return null;
    let minY = h, maxY = 0, minX = w, maxX = 0;
    for (const p of pts) {
        minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
        minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    }
    const iy0 = Math.max(0, Math.floor(minY));
    const iy1 = Math.min(h, Math.ceil(maxY));
    const ix0 = Math.max(0, Math.floor(minX));
    const ix1 = Math.min(w, Math.ceil(maxX));
    for (let yy = iy0; yy < iy1; yy++) {
        const cy = yy + 0.5;
        const xs = [];
        for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
            const yi = pts[i].y, yj = pts[j].y;
            if ((yi > cy) !== (yj > cy)) {
                const xi = pts[i].x, xj = pts[j].x;
                xs.push(((xj - xi) * (cy - yi)) / (yj - yi + 1e-9) + xi);
            }
        }
        xs.sort((a, b) => a - b);
        for (let k = 0; k + 1 < xs.length; k += 2) {
            const xa = Math.max(ix0, Math.ceil(xs[k] - 0.5));
            const xb = Math.min(ix1, Math.ceil(xs[k + 1] - 0.5));
            for (let xx = xa; xx < xb; xx++) mask[yy * w + xx] = value;
        }
    }
    return { x0: ix0, y0: iy0, x1: ix1 - 1, y1: iy1 - 1 };
}

function _match(px, imageData, tr, tg, tb, tol) {
    const i = px * 4;
    return Math.abs(imageData[i] - tr) <= tol &&
        Math.abs(imageData[i + 1] - tg) <= tol &&
        Math.abs(imageData[i + 2] - tb) <= tol;
}

/** Scanline flood fill with a growable Int32Array stack. Mutates mask in place. */
export function floodFill(mask, imageData, w, h, sx, sy, tolerance, value) {
    if (!imageData || sx < 0 || sy < 0 || sx >= w || sy >= h) return null;
    const seed = sy * w + sx;
    const idx0 = seed * 4;
    const tr = imageData[idx0], tg = imageData[idx0 + 1], tb = imageData[idx0 + 2];
    const tol = tolerance | 0;
    const visited = new Uint8Array(w * h);
    let stack = new Int32Array(256);
    let sp = 0;
    const push = (x, y) => {
        if (sp + 2 > stack.length) {
            const bigger = new Int32Array(stack.length * 2);
            bigger.set(stack.subarray(0, sp));
            stack = bigger;
        }
        stack[sp++] = x;
        stack[sp++] = y;
    };
    push(sx, sy);
    let x0 = sx, y0 = sy, x1 = sx, y1 = sy;
    while (sp > 0) {
        const y = stack[--sp];
        const x = stack[--sp];
        let xL = x;
        while (xL >= 0) {
            const pi = y * w + xL;
            if (visited[pi] || !_match(pi, imageData, tr, tg, tb, tol)) break;
            xL--;
        }
        xL++;
        let spanUp = false;
        let spanDown = false;
        for (let xx = xL; xx < w; xx++) {
            const pi = y * w + xx;
            if (visited[pi] || !_match(pi, imageData, tr, tg, tb, tol)) break;
            visited[pi] = 1;
            mask[pi] = value;
            x0 = Math.min(x0, xx); x1 = Math.max(x1, xx);
            y0 = Math.min(y0, y); y1 = Math.max(y1, y);
            if (y > 0) {
                const up = pi - w;
                if (!visited[up] && _match(up, imageData, tr, tg, tb, tol)) {
                    if (!spanUp) { push(xx, y - 1); spanUp = true; }
                } else spanUp = false;
            }
            if (y < h - 1) {
                const dn = pi + w;
                if (!visited[dn] && _match(dn, imageData, tr, tg, tb, tol)) {
                    if (!spanDown) { push(xx, y + 1); spanDown = true; }
                } else spanDown = false;
            }
        }
    }
    return { x0, y0, x1, y1 };
}

/** Flood fill on a scratch buffer; returns bounds without touching the live mask. */
export function floodFillScratch(scratch, imageData, w, h, sx, sy, tolerance, value) {
    return floodFill(scratch, imageData, w, h, sx, sy, tolerance, value);
}

/**
 * Compose a scratch selection (0–255, may be soft) into the live mask.
 * mode "add":     mask = max(mask, sel) inside bounds
 * mode "subtract": mask = min(mask, 255 - sel) inside bounds
 * mode "intersect": mask = min(mask, sel) over the WHOLE frame (outside sel → 0)
 * Returns undo bounds {x0,y0,x1,y1}; intersect always returns the full frame.
 */
export function composeSelection(mask, sel, w, h, mode, bounds) {
    if (mode === "intersect") {
        const n = w * h;
        for (let i = 0; i < n; i++) {
            mask[i] = Math.min(mask[i], sel[i]);
        }
        return { x0: 0, y0: 0, x1: w - 1, y1: h - 1 };
    }
    const x0 = bounds.x0 | 0, y0 = bounds.y0 | 0;
    const x1 = bounds.x1 | 0, y1 = bounds.y1 | 0;
    if (mode === "add") {
        for (let yy = y0; yy <= y1; yy++) {
            const row = yy * w;
            for (let xx = x0; xx <= x1; xx++) {
                const i = row + xx;
                const s = sel[i];
                if (s > mask[i]) mask[i] = s;
            }
        }
    } else {
        for (let yy = y0; yy <= y1; yy++) {
            const row = yy * w;
            for (let xx = x0; xx <= x1; xx++) {
                const i = row + xx;
                const v = 255 - sel[i];
                if (v < mask[i]) mask[i] = v;
            }
        }
    }
    return { x0, y0, x1, y1 };
}

/** Linear interpolation; t in [0, 1]. */
export function pressureAt(p0, p1, t) {
    return p0 + (p1 - p0) * t;
}

/** Stamp radius and opacity from pen pressure and toolbar toggles. */
export function stampPressure(pressure, baseSize, baseOpacity, useSize, useOpacity) {
    let p = pressure;
    if (useSize || useOpacity) {
        p = Math.max(0.05, Math.min(1, pressure));
    }
    const size = useSize ? baseSize * p : baseSize;
    const opacity = useOpacity ? baseOpacity * p : baseOpacity;
    return { radius: size / 2, opacity };
}

/** Like lineBrush but interpolates pressure between endpoints for each stamp. */
export function lineBrushPressure(stampFn, x0, y0, p0, x1, y1, p1, baseRadius, spacing) {
    const dist = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(1, Math.ceil(dist / Math.max(1, spacing)));
    let bx0 = Infinity, by0 = Infinity, bx1 = -1, by1 = -1;
    for (let s = 0; s <= steps; s++) {
        const t = steps === 0 ? 0 : s / steps;
        const cx = x0 + (x1 - x0) * t;
        const cy = y0 + (y1 - y0) * t;
        const pr = pressureAt(p0, p1, t);
        const b = stampFn(cx, cy, pr);
        if (b) {
            bx0 = Math.min(bx0, b.x0); by0 = Math.min(by0, b.y0);
            bx1 = Math.max(bx1, b.x1); by1 = Math.max(by1, b.y1);
        }
    }
    if (bx1 < bx0) return null;
    return { x0: bx0, y0: by0, x1: bx1, y1: by1 };
}
