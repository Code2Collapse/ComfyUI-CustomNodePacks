/**
 * c2c_live_preview.js — live sampling preview drawn ON the sampler node.
 *
 * Renders the server's `b_preview` latent-preview frames directly inside the
 * body of whichever node is currently executing — i.e. on the KSampler /
 * SamplerCustom / KSampler (Efficient) / KijaiSampler / ClownsharKSampler /
 * any sampler — exactly like native ComfyUI's in-node preview. NOT a separate
 * floating window.
 *
 * Why it still works when "ComfyUI preview doesn't work":
 *   - It decodes the b_preview Blob ITSELF and draws it in onDrawForeground,
 *     so it does not depend on native's node-preview display path or any
 *     "show preview" setting being on.
 *   - It pairs with nodes/_c2c_preview_guard.py, which forces the server to
 *     emit previews even when launched with --preview-method none.
 *
 * Why it does not damage native:
 *   - It never touches node.imgs, node.imageIndex, or any native handler. It
 *     only CHAINS onto the node's onDrawForeground + computeSize (calling the
 *     originals first). If native's own preview also shows, ours sits in its
 *     own reserved strip below the widgets; remove our strip and the node is
 *     byte-for-byte the native node again.
 *
 * Robust / update-proof: listens only to the long-stable public api events
 * (executing, b_preview, progress, execution_*). Every handler is wrapped; a
 * future event-shape change no-ops instead of throwing.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NS = "C2C.LivePreview";
const S = {
    enabled: "c2c.livePreview.enabled",
    maxH: "c2c.livePreview.maxHeight",
    opacity: "c2c.livePreview.opacity",
};

let _runningId = null;          // node id currently executing
let _haveMeta = false;          // saw b_preview_with_metadata (supersedes b_preview)

function _get(id, def) {
    try { const v = app.ui?.settings?.getSettingValue?.(id, def); return v === undefined ? def : v; }
    catch { return def; }
}

function _node(id) {
    if (id == null) return null;
    try { return app.graph?.getNodeById?.(Number(id)) || app.graph?.getNodeById?.(id) || null; }
    catch { return null; }
}

// ── per-node preview install (only on nodes that actually stream previews) ──
function _install(node) {
    if (!node || node._c2cPrevInstalled) return;
    node._c2cPrevInstalled = true;

    // Chain computeSize so the reserved preview strip is part of the node's
    // natural height — survives any setSize(computeSize()) from other code.
    const origCompute = node.computeSize?.bind(node);
    node._c2cOrigCompute = origCompute;
    node.computeSize = function (out) {
        const sz = origCompute ? origCompute(out) : [this.size[0], this.size[1]];
        if (this._c2cPrev && _get(S.enabled, true) && !this.flags?.collapsed) {
            const w = Math.max(32, this.size[0] - 12);
            const maxH = Number(_get(S.maxH, 320)) || 320;
            const ph = Math.min(maxH, w / (this._c2cPrevAR || 1));
            sz[1] += ph + 10;
        }
        return sz;
    };

    // Chain onDrawForeground to paint the preview below the widgets.
    const origDraw = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        origDraw?.apply(this, arguments);
        try { _draw(this, ctx); } catch { /* never break canvas paint */ }
    };
}

function _draw(node, ctx) {
    const bmp = node._c2cPrev;
    if (!bmp || node.flags?.collapsed || !_get(S.enabled, true)) return;
    const pad = 6;
    const w = Math.max(32, node.size[0] - pad * 2);
    const maxH = Number(_get(S.maxH, 320)) || 320;
    const ar = node._c2cPrevAR || (bmp.width / bmp.height) || 1;
    const ph = Math.min(maxH, w / ar);
    // Top of the preview strip = the node's natural (widgets-only) height.
    const naturalH = node._c2cOrigCompute ? node._c2cOrigCompute()[1] : 0;
    const y = Math.max(naturalH + 2, node.size[1] - ph - pad);

    ctx.save();
    ctx.globalAlpha = Number(_get(S.opacity, 1)) || 1;
    ctx.fillStyle = "#000";
    ctx.fillRect(pad, y, w, ph);
    try { ctx.drawImage(bmp, pad, y, w, ph); } catch { /* bitmap gone */ }
    ctx.globalAlpha = 1;

    // Progress bar across the bottom of the strip.
    const pr = node._c2cProg;
    if (pr && pr.max) {
        const f = Math.max(0, Math.min(1, pr.value / pr.max));
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        ctx.fillRect(pad, y + ph - 4, w, 4);
        ctx.fillStyle = "#89b4fa";
        ctx.fillRect(pad, y + ph - 4, w * f, 4);
    }
    ctx.restore();
}

function _setPreview(blob, explicitId) {
    if (!_get(S.enabled, true)) return;
    const node = _node(explicitId != null ? explicitId : _runningId);
    if (!node) return;
    // Decode off the main thread; swap the bitmap in atomically when ready.
    createImageBitmap(blob).then((bmp) => {
        if (!bmp) return;
        const prev = node._c2cPrev;
        const firstFrame = !prev;
        node._c2cPrev = bmp;
        node._c2cPrevAR = (bmp.width / bmp.height) || 1;
        try { prev?.close?.(); } catch {}
        _install(node);
        // Grow the node once so the strip has room (keeps last frame after).
        if (firstFrame) {
            try {
                const want = node.computeSize();
                if (node.size[1] < want[1]) node.setSize([node.size[0], want[1]]);
            } catch {}
        }
        node.setDirtyCanvas?.(true, true);
    }).catch(() => { /* keep last frame */ });
}

function _setProgress(value, max, nodeId) {
    const node = _node(nodeId != null ? nodeId : _runningId);
    if (!node) return;
    node._c2cProg = { value, max };
    node.setDirtyCanvas?.(true, false);
}

function _install_listeners() {
    api.addEventListener("executing", (e) => {
        try {
            const d = e?.detail;
            _runningId = (d && typeof d === "object") ? (d.node ?? d.id ?? null) : (d ?? null);
        } catch { _runningId = null; }
    });
    // Newer ComfyUI: carries the node id directly (correct in subgraphs, no
    // executing-event race). Preferred when present.
    api.addEventListener("b_preview_with_metadata", (e) => {
        try {
            const d = e?.detail;
            if (!d) return;
            const blob = d.blob instanceof Blob ? d.blob
                : (d.image instanceof Blob ? d.image : null);
            if (!blob) return;
            _haveMeta = true;
            const id = d.displayNodeId ?? d.nodeId ?? d.node ?? _runningId;
            _setPreview(blob, id);
        } catch {}
    });
    // Classic event (the metadata variant above supersedes it where present).
    api.addEventListener("b_preview", (e) => {
        try {
            if (_haveMeta) return;          // already handled with a node id
            const d = e?.detail;
            if (d instanceof Blob) _setPreview(d);
            else if (d?.image instanceof Blob) _setPreview(d.image);
        } catch {}
    });
    api.addEventListener("progress", (e) => {
        try { const d = e?.detail || {}; _setProgress(d.value, d.max, d.node); } catch {}
    });
    // Resilience: keep the last frame through errors/interruptions.
    api.addEventListener("execution_interrupted", () => { _runningId = null; });
    api.addEventListener("execution_error", () => { _runningId = null; });
    api.addEventListener("execution_success", () => { _runningId = null; });
}

app.registerExtension({
    name: NS,
    settings: [
        { id: S.enabled, name: "C2C ▸ Live Preview ▸ On-node preview", type: "boolean", default: true,
          tooltip: "Draw the live sampling preview inside the sampler node (works even if ComfyUI's own preview display is off)." },
        { id: S.maxH, name: "C2C ▸ Live Preview ▸ Max height (px)", type: "slider",
          attrs: { min: 128, max: 640, step: 16 }, default: 320 },
        { id: S.opacity, name: "C2C ▸ Live Preview ▸ Opacity", type: "slider",
          attrs: { min: 0.3, max: 1, step: 0.02 }, default: 1.0 },
    ],
    async setup() {
        _install_listeners();
        console.log("[C2C.LivePreview] ready — on-node sampler preview from b_preview.");
    },
});
