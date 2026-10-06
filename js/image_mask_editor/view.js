/**
 * Pure view kernels for the Image Mask Editor (DOM-free, Node-testable).
 */

/** 1 where mask>=128 and any 4-neighbour is <128 or OOB; 0 elsewhere; only writes inside rect. */
export function outlineMask(mask, w, h, x0, y0, x1, y1) {
    const out = new Uint8Array(w * h);
    const rx0 = Math.max(0, x0 | 0);
    const ry0 = Math.max(0, y0 | 0);
    const rx1 = Math.min(w - 1, x1 | 0);
    const ry1 = Math.min(h - 1, y1 | 0);
    for (let y = ry0; y <= ry1; y++) {
        const row = y * w;
        for (let x = rx0; x <= rx1; x++) {
            if (mask[row + x] < 128) continue;
            let edge = false;
            if (x <= 0 || mask[row + x - 1] < 128) edge = true;
            else if (x >= w - 1 || mask[row + x + 1] < 128) edge = true;
            else if (y <= 0 || mask[row - w + x] < 128) edge = true;
            else if (y >= h - 1 || mask[row + w + x] < 128) edge = true;
            if (edge) out[row + x] = 1;
        }
    }
    return out;
}

/** mask<128 → red 50% in out; mask>=128 → transparent. Only writes inside rect. */
export function rubylithRGBA(mask, w, h, x0, y0, x1, y1, rgba, outW = w) {
    const rx0 = Math.max(0, x0 | 0);
    const ry0 = Math.max(0, y0 | 0);
    const rx1 = Math.min(w - 1, x1 | 0);
    const ry1 = Math.min(h - 1, y1 | 0);
    for (let y = ry0; y <= ry1; y++) {
        const row = y * w;
        for (let x = rx0; x <= rx1; x++) {
            const oi = ((y - ry0) * outW + (x - rx0)) * 4;
            if (mask[row + x] < 128) {
                rgba[oi] = 255;
                rgba[oi + 1] = 0;
                rgba[oi + 2] = 0;
                rgba[oi + 3] = 128;
            } else {
                rgba[oi + 3] = 0;
            }
        }
    }
}

/** Screen-space pixel-edge coordinates visible in the viewport. Empty when zoom < 8. */
export function gridLines(zoom, panX, panY, viewW, viewH) {
    if (zoom < 8) return { xs: [], ys: [] };
    const kx0 = Math.ceil((0 - panX) / zoom);
    const kx1 = Math.floor((viewW - panX) / zoom);
    const ky0 = Math.ceil((0 - panY) / zoom);
    const ky1 = Math.floor((viewH - panY) / zoom);
    const xs = [];
    const ys = [];
    for (let k = kx0; k <= kx1; k++) xs.push(k * zoom + panX);
    for (let k = ky0; k <= ky1; k++) ys.push(k * zoom + panY);
    return { xs, ys };
}
