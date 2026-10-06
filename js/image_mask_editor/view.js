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

/** Mean mask value over a scale×scale image block (partial blocks at edges). */
export function blockMean(mask, w, h, bx, by, scale) {
    const x0 = bx | 0;
    const y0 = by | 0;
    const x1 = Math.min(w, x0 + (scale | 0));
    const y1 = Math.min(h, y0 + (scale | 0));
    if (x1 <= x0 || y1 <= y0) return 0;
    let sum = 0;
    let n = 0;
    for (let y = y0; y < y1; y++) {
        const row = y * w;
        for (let x = x0; x < x1; x++) {
            sum += mask[row + x];
            n++;
        }
    }
    return n > 0 ? sum / n : 0;
}

/** LOD block means on the downsampled grid (level L, scale = 2^L). */
export function lodBlockMeans(mask, w, h, level) {
    const scale = 1 << (level | 0);
    const gw = Math.ceil(w / scale);
    const gh = Math.ceil(h / scale);
    const out = new Float32Array(gw * gh);
    for (let gy = 0; gy < gh; gy++) {
        for (let gx = 0; gx < gw; gx++) {
            out[gy * gw + gx] = blockMean(mask, w, h, gx * scale, gy * scale, scale);
        }
    }
    return { means: out, gw, gh, scale };
}

function _lodFillRect(fn, mask, imgW, imgH, level, tileImgX, tileImgY, u0, v0, u1, v1, rgba, outW, extra) {
    const scale = 1 << (level | 0);
    for (let v = v0; v <= v1; v++) {
        for (let u = u0; u <= u1; u++) {
            const bx = tileImgX + u * scale;
            const by = tileImgY + v * scale;
            const mean = blockMean(mask, imgW, imgH, bx, by, scale);
            const oi = ((v - v0) * outW + (u - u0)) * 4;
            fn(mean, rgba, oi, extra);
        }
    }
}

/** Accent overlay tint into rgba (tile-sized buffer). */
export function overlayRGBA(mask, w, h, x0, y0, x1, y1, rgba, outW, tint, alpha) {
    const rx0 = Math.max(0, x0 | 0);
    const ry0 = Math.max(0, y0 | 0);
    const rx1 = Math.min(w - 1, x1 | 0);
    const ry1 = Math.min(h - 1, y1 | 0);
    for (let y = ry0; y <= ry1; y++) {
        const row = y * w;
        for (let x = rx0; x <= rx1; x++) {
            const oi = ((y - ry0) * outW + (x - rx0)) * 4;
            const a = mask[row + x];
            rgba[oi] = tint.r;
            rgba[oi + 1] = tint.g;
            rgba[oi + 2] = tint.b;
            rgba[oi + 3] = Math.round(a * alpha);
        }
    }
}

/** Grayscale matte into rgba (tile-sized buffer). */
export function matteRGBA(mask, w, h, x0, y0, x1, y1, rgba, outW) {
    const rx0 = Math.max(0, x0 | 0);
    const ry0 = Math.max(0, y0 | 0);
    const rx1 = Math.min(w - 1, x1 | 0);
    const ry1 = Math.min(h - 1, y1 | 0);
    for (let y = ry0; y <= ry1; y++) {
        const row = y * w;
        for (let x = rx0; x <= rx1; x++) {
            const oi = ((y - ry0) * outW + (x - rx0)) * 4;
            const g = mask[row + x];
            rgba[oi] = rgba[oi + 1] = rgba[oi + 2] = g;
            rgba[oi + 3] = 255;
        }
    }
}

/** Candidate / refine preview tint into rgba (tile-sized buffer). */
export function previewRGBA(buf, w, h, x0, y0, x1, y1, rgba, outW, tint, alpha) {
    const rx0 = Math.max(0, x0 | 0);
    const ry0 = Math.max(0, y0 | 0);
    const rx1 = Math.min(w - 1, x1 | 0);
    const ry1 = Math.min(h - 1, y1 | 0);
    for (let y = ry0; y <= ry1; y++) {
        const row = y * w;
        for (let x = rx0; x <= rx1; x++) {
            const oi = ((y - ry0) * outW + (x - rx0)) * 4;
            const a = buf[row + x];
            rgba[oi] = tint.r;
            rgba[oi + 1] = tint.g;
            rgba[oi + 2] = tint.b;
            rgba[oi + 3] = Math.round(a * alpha);
        }
    }
}

export function overlayTileLOD(mask, imgW, imgH, level, tileImgX, tileImgY, u0, v0, u1, v1, rgba, outW, tint, alpha) {
    _lodFillRect((mean, rgba, oi, { tint, alpha }) => {
        rgba[oi] = tint.r;
        rgba[oi + 1] = tint.g;
        rgba[oi + 2] = tint.b;
        rgba[oi + 3] = Math.round(mean * alpha);
    }, mask, imgW, imgH, level, tileImgX, tileImgY, u0, v0, u1, v1, rgba, outW, { tint, alpha });
}

export function matteTileLOD(mask, imgW, imgH, level, tileImgX, tileImgY, u0, v0, u1, v1, rgba, outW) {
    _lodFillRect((mean, rgba, oi) => {
        const g = Math.round(mean);
        rgba[oi] = rgba[oi + 1] = rgba[oi + 2] = g;
        rgba[oi + 3] = 255;
    }, mask, imgW, imgH, level, tileImgX, tileImgY, u0, v0, u1, v1, rgba, outW, null);
}

export function rubylithTileLOD(mask, imgW, imgH, level, tileImgX, tileImgY, u0, v0, u1, v1, rgba, outW) {
    _lodFillRect((mean, rgba, oi) => {
        if (mean < 128) {
            rgba[oi] = 255;
            rgba[oi + 1] = 0;
            rgba[oi + 2] = 0;
            rgba[oi + 3] = 128;
        } else {
            rgba[oi + 3] = 0;
        }
    }, mask, imgW, imgH, level, tileImgX, tileImgY, u0, v0, u1, v1, rgba, outW, null);
}

export function previewTileLOD(buf, imgW, imgH, level, tileImgX, tileImgY, u0, v0, u1, v1, rgba, outW, tint, alpha) {
    _lodFillRect((mean, rgba, oi, { tint, alpha }) => {
        rgba[oi] = tint.r;
        rgba[oi + 1] = tint.g;
        rgba[oi + 2] = tint.b;
        rgba[oi + 3] = Math.round(mean * alpha);
    }, buf, imgW, imgH, level, tileImgX, tileImgY, u0, v0, u1, v1, rgba, outW, { tint, alpha });
}

/** Outline tile RGBA from mask; halo ±1 px for seam-free tiles; does not touch outlineBuf. */
export function outlineTileRGBA(mask, w, h, x, y, tileW, tileH, rgba, outW) {
    const hx0 = Math.max(0, (x | 0) - 1);
    const hy0 = Math.max(0, (y | 0) - 1);
    const hx1 = Math.min(w - 1, (x | 0) + (tileW | 0));
    const hy1 = Math.min(h - 1, (y | 0) + (tileH | 0));
    const edges = outlineMask(mask, w, h, hx0, hy0, hx1, hy1);
    for (let yy = 0; yy < tileH; yy++) {
        for (let xx = 0; xx < tileW; xx++) {
            const pi = (y + yy) * w + (x + xx);
            const oi = (yy * outW + xx) * 4;
            if (edges[pi]) {
                rgba[oi] = rgba[oi + 1] = rgba[oi + 2] = 255;
                rgba[oi + 3] = 255;
            } else {
                rgba[oi + 3] = 0;
            }
        }
    }
}

/** Outline tile at LOD: threshold block means at 128, 4-neighbour edges on LOD grid; no outlineBuf. */
export function outlineTileLOD(mask, imgW, imgH, level, tileImgX, tileImgY, tileW, tileH, u0, v0, u1, v1, rgba, outW) {
    const hu0 = Math.max(0, u0 - 1);
    const hv0 = Math.max(0, v0 - 1);
    const hu1 = Math.min(tileW - 1, u1 + 1);
    const hv1 = Math.min(tileH - 1, v1 + 1);
    const gw = hu1 - hu0 + 1;
    const gh = hv1 - hv0 + 1;
    const scale = 1 << (level | 0);
    const lod = new Uint8Array(gw * gh);
    for (let v = 0; v < gh; v++) {
        for (let u = 0; u < gw; u++) {
            const lu = hu0 + u;
            const lv = hv0 + v;
            const mean = blockMean(mask, imgW, imgH, tileImgX + lu * scale, tileImgY + lv * scale, scale);
            lod[v * gw + u] = mean >= 128 ? 255 : 0;
        }
    }
    const edges = outlineMask(lod, gw, gh, u0 - hu0, v0 - hv0, u1 - hu0, v1 - hv0);
    for (let v = v0; v <= v1; v++) {
        for (let u = u0; u <= u1; u++) {
            const oi = ((v - v0) * outW + (u - u0)) * 4;
            if (edges[(v - hv0) * gw + (u - hu0)]) {
                rgba[oi] = rgba[oi + 1] = rgba[oi + 2] = 255;
                rgba[oi + 3] = 255;
            } else {
                rgba[oi + 3] = 0;
            }
        }
    }
}

/** Write edge flags into outlineBuf for an inclusive rect (clears non-edges inside rect). */
export function outlineBufRect(mask, w, h, x0, y0, x1, y1, outlineBuf) {
    const edges = outlineMask(mask, w, h, x0, y0, x1, y1);
    const rx0 = Math.max(0, x0 | 0);
    const ry0 = Math.max(0, y0 | 0);
    const rx1 = Math.min(w - 1, x1 | 0);
    const ry1 = Math.min(h - 1, y1 | 0);
    for (let y = ry0; y <= ry1; y++) {
        const row = y * w;
        for (let x = rx0; x <= rx1; x++) {
            const pi = row + x;
            outlineBuf[pi] = edges[pi] ? 1 : 0;
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
