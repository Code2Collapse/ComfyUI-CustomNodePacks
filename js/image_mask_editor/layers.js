/**
 * Mask layer stack merge (DOM-free, Node-testable).
 */

export const MAX_LAYERS = 8;
export const LAYER_MEM_CAP = 512 * 1024 * 1024;

export function defaultManifest() {
    return {
        version: 1,
        layers: [{
            id: "base",
            name: "Layer 1",
            mode: "add",
            visible: true,
            locked: false,
        }],
        active: "base",
    };
}

export function needsSidecar(manifest) {
    const layers = manifest?.layers || [];
    if (layers.length > 1) return true;
    if (layers.length !== 1) return false;
    const layer = layers[0];
    return !(layer.visible !== false && layer.mode === "add");
}

/**
 * Layer that receives legacy merged PNG content when a frame has merged PNG but no layer PNGs yet.
 * Bottom→top: first visible add; else first add (even hidden); else null.
 * @returns {string|null}
 */
export function legacyFillLayer(manifest) {
    const layers = manifest?.layers || [];
    for (let i = 0; i < layers.length; i++) {
        const layer = layers[i];
        if (layer.mode === "add" && layer.visible !== false) return layer.id;
    }
    for (let i = 0; i < layers.length; i++) {
        const layer = layers[i];
        if (layer.mode === "add") return layer.id;
    }
    return null;
}

export function makeLayerId() {
    const raw = (crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`).replace(/[^a-z0-9-]/gi, "").toLowerCase();
    return raw.slice(0, 16) || `l${Date.now().toString(36).slice(-8)}`;
}

function _buf(buffers, id) {
    if (buffers instanceof Map) return buffers.get(id);
    return buffers[id];
}

/**
 * Merge layer stack into out for pixels inside [x0,y0]-[x1,y1].
 * @param {Map<string,Uint8Array>|Record<string,Uint8Array>} buffers
 * @param {Array<{id:string,mode:string,visible:boolean}>} layers bottom→top
 * @param {Uint8Array} out full-frame merged buffer
 */
export function mergeLayers(buffers, layers, x0, y0, x1, y1, w, out) {
    const rx0 = Math.max(0, x0 | 0);
    const ry0 = Math.max(0, y0 | 0);
    const rx1 = Math.min((w | 0) - 1, x1 | 0);
    const ry1 = Math.min(((out.length / w) | 0) - 1, y1 | 0);
    if (rx1 < rx0 || ry1 < ry0) return;
    for (let y = ry0; y <= ry1; y++) {
        const row = y * w;
        for (let x = rx0; x <= rx1; x++) {
            const idx = row + x;
            let m = 0;
            for (let li = 0; li < layers.length; li++) {
                const layer = layers[li];
                if (layer.visible === false) continue;
                const buf = _buf(buffers, layer.id);
                const L = buf ? buf[idx] : 0;
                const mode = layer.mode || "add";
                if (mode === "add") {
                    if (L > m) m = L;
                } else if (mode === "subtract") {
                    const v = 255 - L;
                    if (v < m) m = v;
                } else if (mode === "intersect") {
                    if (L < m) m = L;
                }
            }
            out[idx] = m;
        }
    }
}
