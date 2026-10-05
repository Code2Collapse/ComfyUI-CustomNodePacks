/**
 * Colour-range distance kernels (pure functions, no imports — Node-testable).
 *
 * Distance formulas (test contract):
 *
 * RGB: d = sqrt(dr² + dg² + db²) with r,g,b in 0–255.
 *
 * HSV (OpenCV 8-bit): H ∈ [0,180), S,V ∈ [0,255].
 *   dh = circular hue difference on the 180-circle (wrap, max 90°), × 255/90,
 *        weighted × min(s1,s2)/255 (grey → no hue).
 *   d = max(dh_weighted, |ds|, |dv|).
 *
 * LAB: sRGB γ-decode → linear RGB → XYZ (D65) → CIE L*a*b* (exact CIE formulas, as skimage.color.rgb2lab).
 *   d = 2.55 × ΔE76 (Euclidean in Lab).
 */

function _srgbToLinear(c) {
    const x = c / 255;
    return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

function _rgbToLab(r, g, b) {
    const R = _srgbToLinear(r), G = _srgbToLinear(g), B = _srgbToLinear(b);
    const X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B;
    const Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B;
    const Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B;
    const Xn = 0.95047, Yn = 1, Zn = 1.08883;
    const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
    const fy = f(Y / Yn);
    return {
        L: 116 * fy - 16,
        a: 500 * (f(X / Xn) - fy),
        b: 200 * (fy - f(Z / Zn)),
    };
}

/** OpenCV 8-bit HSV: H ∈ [0,180), S,V ∈ [0,255]. */
function _rgbToHsv8(r, g, b) {
    const rf = r / 255, gf = g / 255, bf = b / 255;
    const mx = Math.max(rf, gf, bf), mn = Math.min(rf, gf, bf);
    const d = mx - mn;
    let h = 0;
    if (d > 1e-9) {
        if (mx === rf) h = 60 * (((gf - bf) / d) % 6);
        else if (mx === gf) h = 60 * ((bf - rf) / d + 2);
        else h = 60 * ((rf - gf) / d + 4);
        if (h < 0) h += 360;
    }
    const s = mx <= 1e-9 ? 0 : (d / mx) * 255;
    const v = mx * 255;
    return { h: h / 2, s, v }; // H scaled to [0,180)
}

function _hueDiff180(h1, h2) {
    let d = Math.abs(h1 - h2);
    if (d > 90) d = 180 - d;
    return d;
}

function _distRgb(r1, g1, b1, r2, g2, b2) {
    const dr = r1 - r2, dg = g1 - g2, db = b1 - b2;
    return Math.sqrt(dr * dr + dg * dg + db * db);
}

function _distHsv(r1, g1, b1, r2, g2, b2) {
    const a = _rgbToHsv8(r1, g1, b1), b = _rgbToHsv8(r2, g2, b2);
    const dh = _hueDiff180(a.h, b.h) * (255 / 90) * (Math.min(a.s, b.s) / 255);
    const ds = Math.abs(a.s - b.s);
    const dv = Math.abs(a.v - b.v);
    return Math.max(dh, ds, dv);
}

function _distLab(r1, g1, b1, r2, g2, b2) {
    const a = _rgbToLab(r1, g1, b1), b = _rgbToLab(r2, g2, b2);
    const dL = a.L - b.L, da = a.a - b.a, db = a.b - b.b;
    return 2.55 * Math.sqrt(dL * dL + da * da + db * db);
}

function _pixelDist(r, g, b, samples, space) {
    let minD = Infinity;
    for (const s of samples) {
        let d;
        if (space === "hsv") d = _distHsv(r, g, b, s[0], s[1], s[2]);
        else if (space === "lab") d = _distLab(r, g, b, s[0], s[1], s[2]);
        else d = _distRgb(r, g, b, s[0], s[1], s[2]);
        if (d < minD) minD = d;
    }
    return minD;
}

/** MIN distance from each pixel to any sample colour. */
export function distanceMap(rgba, w, h, samples, space) {
    const out = new Float32Array(w * h);
    if (!samples.length) return out;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 4;
            out[y * w + x] = _pixelDist(rgba[i], rgba[i + 1], rgba[i + 2], samples, space);
        }
    }
    return out;
}

/** Threshold distance map into a soft selection mask. */
export function selectionFromDistance(dist, tol, softness) {
    const n = dist.length;
    const out = new Uint8Array(n);
    const t = tol, soft = softness;
    for (let i = 0; i < n; i++) {
        const d = dist[i];
        if (d <= t) out[i] = 255;
        else if (soft > 0 && d < t + soft) {
            out[i] = Math.round(255 * (1 - (d - t) / soft));
        }
    }
    return out;
}

/** Keep only 4-connected components of sel>0 that contain a seed pixel. */
export function keepConnected(sel, w, h, seeds) {
    const out = new Uint8Array(sel.length);
    if (!seeds.length) return out;
    const visited = new Uint8Array(sel.length);
    const stack = new Int32Array(sel.length);
    for (const seed of seeds) {
        const sx = seed.x | 0, sy = seed.y | 0;
        if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
        const si = sy * w + sx;
        if (visited[si] || sel[si] === 0) continue;
        let sp = 0;
        stack[sp++] = sx;
        stack[sp++] = sy;
        visited[si] = 1;
        while (sp > 0) {
            const y = stack[--sp];
            const x = stack[--sp];
            const pi = y * w + x;
            out[pi] = sel[pi];
            if (x > 0) {
                const l = pi - 1;
                if (!visited[l] && sel[l] > 0) { visited[l] = 1; stack[sp++] = x - 1; stack[sp++] = y; }
            }
            if (x + 1 < w) {
                const r = pi + 1;
                if (!visited[r] && sel[r] > 0) { visited[r] = 1; stack[sp++] = x + 1; stack[sp++] = y; }
            }
            if (y > 0) {
                const u = pi - w;
                if (!visited[u] && sel[u] > 0) { visited[u] = 1; stack[sp++] = x; stack[sp++] = y - 1; }
            }
            if (y + 1 < h) {
                const d = pi + w;
                if (!visited[d] && sel[d] > 0) { visited[d] = 1; stack[sp++] = x; stack[sp++] = y + 1; }
            }
        }
    }
    return out;
}
