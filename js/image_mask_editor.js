/**
 * ImageMaskEditorC2C — native-resolution mask editor node UI.
 */
import { app } from "../../scripts/app.js";
import { resolveEditorSource } from "./image_mask_editor/source.js";
import { fetchState } from "./image_mask_editor/api.js";
import { ensureEditorId, dedupeOnConfigure } from "./image_mask_editor/id.js";
import { openModal } from "./image_mask_editor/modal.js";
import { C, hexToRgb } from "./image_mask_editor/palette.js";

const NODE_NAME = "ImageMaskEditorC2C";
const EXT_NAME = "C2C.ImageMaskEditor";
const THUMB_H = 200;

function _vueActive() {
    try {
        return app.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled") === true;
    } catch (_) {
        return false;
    }
}

class ThumbController {
    constructor(node) {
        this.node = node;
        this.frame = 0;
        this.digest = "";
        this.lastW = 0;
        this.plate = null;
        this._maskBm = null;
        this._tintCanvas = null;
        this._tintKey = "";
        this.frameCount = 1;
        this.nativeW = 0;
        this.nativeH = 0;
        this._plateUrl = "";
        this._vueCanvas = null;
        this._vueCtx = null;
        this._statusEl = null;
    }

    markDirty() {
        this._scheduleDraw();
    }

    _scheduleDraw() {
        if (!this.node?.graph) return;
        try {
            this.node.graph.setDirtyCanvas?.(true, true);
        } catch (_) { /* node removed */ }
        this._drawVue();
    }

    async refresh() {
        if (!this.node?.graph) return;
        const eid = ensureEditorId(this.node);
        try {
            const src = await resolveEditorSource(this.node);
            this.frameCount = Math.max(1, src.count || 1);
            this.nativeW = src.width || 0;
            this.nativeH = src.height || 0;
            if (src.count) {
                let url = src.thumbUrl?.(0) ?? src.url(0);
                if (src.kind === "plan") {
                    const m = url.match(/token=([^&]+)/);
                    if (m) url = `/c2c/frames/thumb?token=${encodeURIComponent(m[1])}&i=${this.frame | 0}&max=384&fmt=jpeg`;
                }
                this._plateUrl = url;
                this.plate = await _loadImg(url);
                if (!this.nativeW) {
                    this.nativeW = this.plate.naturalWidth;
                    this.nativeH = this.plate.naturalHeight;
                }
            }
            if (eid) {
                const st = await fetchState(eid);
                this.digest = st.digest || "";
                const frames = st.frames || [];
                if (frames.length) {
                    const f = frames.includes(this.frame) ? this.frame : frames[0];
                    this.frame = f;
                    await this._loadMaskThumb(eid, f);
                } else {
                    this._clearMaskTint();
                }
            }
        } catch (e) {
            console.warn("[IME] thumb refresh:", e);
        }
        this._updateStatus();
        this._scheduleDraw();
    }

    _clearMaskTint() {
        try { this._maskBm?.close?.(); } catch (_) { /* ignore */ }
        this._maskBm = null;
        this._tintCanvas = null;
        this._tintKey = "";
    }

    async _loadMaskThumb(eid, frame) {
        const r = await fetch(`/c2c/image_mask_editor/frame?id=${encodeURIComponent(eid)}&frame=${frame | 0}`);
        if (!r.ok) { this._clearMaskTint(); return; }
        try { this._maskBm?.close?.(); } catch (_) { /* ignore */ }
        this._maskBm = await createImageBitmap(await r.blob());
        this._tintCanvas = null;
        this._tintKey = "";
    }

    _ensureTint(dw, dh) {
        if (!this._maskBm || dw < 1 || dh < 1) return null;
        const key = `${this.digest}|${this.frame}|${Math.round(dw)}x${Math.round(dh)}`;
        if (this._tintCanvas && this._tintKey === key) return this._tintCanvas;
        const c = document.createElement("canvas");
        c.width = Math.max(1, Math.round(dw));
        c.height = Math.max(1, Math.round(dh));
        const cx = c.getContext("2d");
        cx.drawImage(this._maskBm, 0, 0, c.width, c.height);
        const id = cx.getImageData(0, 0, c.width, c.height);
        const ac = hexToRgb(C.accent);
        for (let i = 0; i < id.data.length; i += 4) {
            const lum = id.data[i];
            id.data[i] = ac.r;
            id.data[i + 1] = ac.g;
            id.data[i + 2] = ac.b;
            id.data[i + 3] = Math.round(lum * 0.45);
        }
        cx.putImageData(id, 0, 0);
        this._tintCanvas = c;
        this._tintKey = key;
        return c;
    }

    _updateStatus() {
        if (!this._statusEl) return;
        const n = this.digest ? "…" : "0";
        const eid = ensureEditorId(this.node);
        fetchState(eid).then((st) => {
            const cnt = (st.frames || []).length;
            this._statusEl.textContent =
                `frame ${this.frame + 1}/${this.frameCount} · ${cnt} masks · ${this.nativeW || "?"}×${this.nativeH || "?"}`;
        }).catch(() => {
            this._statusEl.textContent =
                `frame ${this.frame + 1}/${this.frameCount} · ${this.nativeW || "?"}×${this.nativeH || "?"}`;
        });
    }

    draw(ctx, w, h) {
        if (!ctx) return;
        const W = w || this.lastW || 320;
        this.lastW = W;
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, W, THUMB_H);
        if (!this.plate) {
            ctx.fillStyle = C.sub;
            ctx.font = "12px sans-serif";
            ctx.fillText("No preview — connect image & queue once", 8, THUMB_H / 2);
            return;
        }
        const iw = this.plate.naturalWidth || this.nativeW || 1;
        const ih = this.plate.naturalHeight || this.nativeH || 1;
        const scale = Math.min((W - 8) / iw, (THUMB_H - 8) / ih);
        const dw = iw * scale, dh = ih * scale;
        const ox = (W - dw) / 2, oy = (THUMB_H - dh) / 2;
        ctx.drawImage(this.plate, ox, oy, dw, dh);
        const tint = this._ensureTint(dw, dh);
        if (tint) ctx.drawImage(tint, ox, oy, dw, dh);
    }

    _drawVue() {
        if (!this._vueCanvas || !this._vueCtx) return;
        const r = this._vueCanvas.getBoundingClientRect();
        const w = Math.max(1, Math.round(r.width || 320));
        const dpr = window.devicePixelRatio || 1;
        this._vueCanvas.width = Math.round(w * dpr);
        this._vueCanvas.height = Math.round(THUMB_H * dpr);
        this._vueCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.draw(this._vueCtx, w, THUMB_H);
    }

    installVueCanvas(host) {
        const cvs = document.createElement("canvas");
        cvs.style.cssText = "width:100%;height:100%;display:block;";
        host.appendChild(cvs);
        this._vueCanvas = cvs;
        this._vueCtx = cvs.getContext("2d");
        const ro = new ResizeObserver(() => {
            if (this.node?.graph) this._drawVue();
        });
        ro.observe(host);
        this._ro = ro;
    }

    dispose() {
        try { this._ro?.disconnect(); } catch (_) { /* ignore */ }
        this._clearMaskTint();
        this.plate = null;
        this.node = null;
    }
}

function _loadImg(url) {
    return new Promise((resolve, reject) => {
        const im = new Image();
        im.crossOrigin = "anonymous";
        im.onload = () => resolve(im);
        im.onerror = reject;
        im.src = url;
    });
}

function installNodeUI(node) {
    const thumb = new ThumbController(node);
    node._imeThumb = thumb;

    const widget = {
        name: "ime_preview",
        type: "custom",
        // Frontend 1.52 calls computeSize() with NO argument (LGraphNode._arrangeWidgets) and with a
        // width elsewhere (getWidgetOnPos) - never with the node, so read the width from the closure.
        computeSize(width) {
            return [typeof width === "number" ? width : (node.size?.[0] ?? 320), THUMB_H];
        },
        // LiteGraph calls draw(ctx, node, width, y, H): y is this widget's own slot, so paint there -
        // painting at 0 covered the title and the input widgets and left this slot empty.
        draw(ctx, n, w, y) {
            if (thumb.lastW !== w) thumb.lastW = w;
            ctx.save();
            ctx.translate(0, y);
            thumb.draw(ctx, w, THUMB_H);
            ctx.restore();
        },
    };
    node.addCustomWidget(widget);

    if (_vueActive()) {
        widget.options = { canvasOnly: true };
        widget.computeSize = () => [0, -4];
        const host = document.createElement("div");
        host.style.cssText = `width:100%;height:${THUMB_H}px;overflow:hidden;`;
        thumb.installVueCanvas(host);
        try {
            node.addDOMWidget("ime_preview_vue", "canvas", host, {
                serialize: false,
                getMinHeight: () => THUMB_H,
                getHeight: () => THUMB_H,
            });
        } catch (_) { /* classic only */ }
    }

    const status = document.createElement("div");
    status.style.cssText = `padding:2px 4px;font:11px/14px sans-serif;color:${C.sub};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`;
    thumb._statusEl = status;
    try {
        // One 18 px text line. Frontend 1.52.7 gives a DOM widget without explicit heights a 50 px
        // minimum and no maximum (DOMWidgetImpl.computeLayoutSize: dead space under the status) and a
        // 10 px margin on every side (BaseDOMWidgetImpl.DEFAULT_MARGIN: the text spilled onto the button).
        node.addDOMWidget("ime_status", "div", status, {
            serialize: false, margin: 2, getMinHeight: () => 22, getMaxHeight: () => 22,
        });
    } catch (_) {
        const sw = node.addWidget("text", "status", "", () => {}, { serialize: false });
        sw.disabled = true;
        thumb._statusEl = { set textContent(t) { sw.value = t; } };
    }

    node.addWidget("button", "✏️ Edit mask", null, () => {
        if (!node.graph) return;
        const eid = ensureEditorId(node);
        openModal(node, eid, (digest) => {
            thumb.digest = digest || "";
            thumb.refresh();
        });
    }, { serialize: false });

    const origResize = node.onResize;
    node.onResize = function (...a) {
        const r = origResize?.apply(this, a);
        if (this.size?.[0] !== thumb.lastW) thumb._scheduleDraw();
        return r;
    };

    const origRemoved = node.onRemoved;
    node.onRemoved = function (...a) {
        thumb.dispose();
        return origRemoved?.apply(this, a);
    };

    setTimeout(() => thumb.refresh(), 50);
}

if (!(app.extensions || []).some((e) => e?.name === EXT_NAME)) {
    app.registerExtension({
        name: EXT_NAME,
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== NODE_NAME) return;
            const origExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (output) {
                const r = origExecuted?.apply(this, arguments);
                this._imeRun = output?.c2c_ime_frames?.[0] ?? null;
                this._imeThumb?.refresh();
                return r;
            };
            const origConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const r = origConfigure?.apply(this, arguments);
                dedupeOnConfigure(this).then(() => {
                    this._imeThumb?.refresh();
                }).catch(() => {});
                return r;
            };
        },
        async nodeCreated(node) {
            if (node.comfyClass !== NODE_NAME) return;
            ensureEditorId(node);
            installNodeUI(node);
        },
    });
}
