/** Zoom / pan coordinate transforms. */

export function screenToImage(sx, sy, zoom, panX, panY) {
    return { x: (sx - panX) / zoom, y: (sy - panY) / zoom };
}

export function imageToScreen(ix, iy, zoom, panX, panY) {
    return { x: ix * zoom + panX, y: iy * zoom + panY };
}

export function zoomAtCursor(zoom, panX, panY, cx, cy, factor, minZ = 0.05, maxZ = 32) {
    const nz = Math.max(minZ, Math.min(maxZ, zoom * factor));
    const scale = nz / zoom;
    return {
        zoom: nz,
        panX: cx - (cx - panX) * scale,
        panY: cy - (cy - panY) * scale,
    };
}

export function fitView(viewW, viewH, imgW, imgH, pad = 16) {
    if (!imgW || !imgH) return { zoom: 1, panX: pad, panY: pad };
    const zw = (viewW - pad * 2) / imgW;
    const zh = (viewH - pad * 2) / imgH;
    const zoom = Math.min(zw, zh, 8);
    const panX = (viewW - imgW * zoom) / 2;
    const panY = (viewH - imgH * zoom) / 2;
    return { zoom, panX, panY };
}
