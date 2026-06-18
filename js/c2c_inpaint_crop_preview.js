// c2c_inpaint_crop_preview.js
// P0.4 — live-preview overlay for InpaintCropProMEC.
//
// Adds a custom canvas widget at the bottom of the node body that draws:
//   - the source image bounds (uses real upstream IMAGE dims when a
//     connected loader has executed; otherwise a 16:9 synthetic frame),
//   - a synthetic mask bbox (centered, 30% of source — until we wire a
//     real preview channel from the backend),
//   - the resulting crop rectangle after `context_from_mask_extend_factor`
//     (or AUTO) and `aspect_preset`,
//   - text labels for the resolved factor, target W/H and the preset name.
//
// Pure additive. Every draw call is wrapped in try/catch — if it throws,
// the node still works exactly as before.

import { app } from "../../scripts/app.js";
import { C } from './_c2c_theme.js';

const EXT_NAME = "C2C.InpaintCropPreview";
const NODE_NAME = "InpaintCropProMEC";
const WIDGET_NAME = "_c2c_crop_preview";
const BAND_HEIGHT = 150;

const ASPECT_RATIOS = {
    "Square": 1.0,
    "16:9":   16.0 / 9.0,
    "9:16":   9.0 / 16.0,
    "4:3":    4.0 / 3.0,
    "3:4":    3.0 / 4.0,
};
const ASPECT_PRESETS = ["Custom", "Auto", "Square", "16:9", "9:16", "4:3", "3:4"];

function _widgetValue(node, name, fallback) {
    if (!node || !node.widgets) return fallback;
    const w = node.widgets.find(w => w && w.name === name);
    return (w && w.value !== undefined && w.value !== null) ? w.value : fallback;
}

function _resolveAspect(preset, manualW, manualH, maskAR) {
    if (preset === "Custom" || !ASPECT_PRESETS.includes(preset)) {
        return [Math.round(manualW), Math.round(manualH)];
    }
    let targetAR;
    if (preset === "Auto") {
        if (!maskAR || maskAR <= 0) return [Math.round(manualW), Math.round(manualH)];
        let best = null, bestD = Infinity;
        for (const k of Object.keys(ASPECT_RATIOS)) {
            const d = Math.abs(Math.log(ASPECT_RATIOS[k]) - Math.log(Math.max(maskAR, 1e-6)));
            if (d < bestD) { bestD = d; best = k; }
        }
        targetAR = ASPECT_RATIOS[best];
    } else {
        targetAR = ASPECT_RATIOS[preset];
    }
    const short = Math.max(64, Math.min(manualW, manualH));
    if (targetAR >= 1.0) return [Math.round(short * targetAR), short];
    return [short, Math.round(short / targetAR)];
}

function _autoFactor(maskFill /* 0..1 */, targetFill = 0.70) {
    if (!maskFill || maskFill <= 0) return 1.0;
    return Math.max(0.30, Math.min(10.0, Math.sqrt(maskFill / Math.max(targetFill, 1e-3))));
}

function _readUpstreamImageSize(node) {
    try {
        if (!node || !node.inputs) return null;
        const slot = node.inputs.findIndex(i => i && i.name === "image");
        if (slot < 0) return null;
        const link_id = node.inputs[slot].link;
        if (link_id == null) return null;
        const link = node.graph?.links?.[link_id];
        if (!link) return null;
        const src = node.graph.getNodeById(link.origin_id);
        if (!src) return null;
        if (src.imgs && src.imgs.length > 0) {
            const im = src.imgs[0];
            if (im.naturalWidth > 0 && im.naturalHeight > 0) {
                return { w: im.naturalWidth, h: im.naturalHeight };
            }
        }
        if (Array.isArray(src.images) && src.images.length > 0) {
            const im = src.images[0];
            if (im && im.naturalWidth > 0 && im.naturalHeight > 0) {
                return { w: im.naturalWidth, h: im.naturalHeight };
            }
        }
    } catch (e) { /* swallow */ }
    return null;
}

function _drawPreview(node, ctx, x, y, w, h) {
    try {
        const factorManual = Number(_widgetValue(node, "context_from_mask_extend_factor", 1.2));
        const autoFactor   = !!_widgetValue(node, "auto_context_factor", false);
        const preset       = String(_widgetValue(node, "aspect_preset", "Custom"));
        const tgtWManual   = Math.max(64, Math.round(Number(_widgetValue(node, "output_target_width",  512))));
        const tgtHManual   = Math.max(64, Math.round(Number(_widgetValue(node, "output_target_height", 512))));

        const upstream = _readUpstreamImageSize(node);
        const srcW = upstream ? upstream.w : 1920;
        const srcH = upstream ? upstream.h : 1080;

        const mW = Math.round(srcW * 0.30);
        const mH = Math.round(srcH * 0.30);
        const mX = Math.round((srcW - mW) / 2);
        const mY = Math.round((srcH - mH) / 2);
        const maskAR = mW / Math.max(1, mH);
        const synthFill = 0.85;
        const factor = autoFactor ? _autoFactor(synthFill) : Math.max(0.10, Math.min(100.0, factorManual));

        let cropW, cropH, cropX, cropY;
        if (factor >= 1.0) {
            const gx = Math.round(mW * (factor - 1.0) / 2.0);
            const gy = Math.round(mH * (factor - 1.0) / 2.0);
            cropX = Math.max(0, mX - gx);
            cropY = Math.max(0, mY - gy);
            cropW = Math.min(srcW, mX + mW + gx) - cropX;
            cropH = Math.min(srcH, mY + mH + gy) - cropY;
        } else {
            const sx = Math.round(mW * (1.0 - factor) / 2.0);
            const sy = Math.round(mH * (1.0 - factor) / 2.0);
            cropX = mX + sx;
            cropY = mY + sy;
            cropW = Math.max(1, mW - 2 * sx);
            cropH = Math.max(1, mH - 2 * sy);
        }

        const [tgtW, tgtH] = _resolveAspect(preset, tgtWManual, tgtHManual, maskAR);
        const tgtAR = tgtW / Math.max(1, tgtH);
        const cropAR = cropW / Math.max(1, cropH);
        const shrinkMode = factor < 1.0;
        if (shrinkMode) {
            if (cropAR > tgtAR) {
                const newW = Math.max(1, Math.round(cropH * tgtAR));
                cropX = cropX + Math.round((cropW - newW) / 2);
                cropW = newW;
            } else if (cropAR < tgtAR) {
                const newH = Math.max(1, Math.round(cropW / tgtAR));
                cropY = cropY + Math.round((cropH - newH) / 2);
                cropH = newH;
            }
        } else {
            if (cropAR < tgtAR) {
                const newW = Math.round(cropH * tgtAR);
                cropX = Math.max(0, Math.round(cropX - (newW - cropW) / 2));
                cropW = newW;
            } else if (cropAR > tgtAR) {
                const newH = Math.round(cropW / tgtAR);
                cropY = Math.max(0, Math.round(cropY - (newH - cropH) / 2));
                cropH = newH;
            }
        }
        cropX = Math.max(0, Math.min(srcW - 1, cropX));
        cropY = Math.max(0, Math.min(srcH - 1, cropY));
        cropW = Math.max(1, Math.min(srcW - cropX, cropW));
        cropH = Math.max(1, Math.min(srcH - cropY, cropH));

        const PAD = 6;
        const headerH = 16;
        const labelH = 14;
        const innerX = x + PAD;
        const innerY = y + PAD + headerH;
        const innerW = w - 2 * PAD;
        const innerH = h - 2 * PAD - headerH - labelH;

        ctx.save();
        ctx.fillStyle = "rgba(20, 22, 30, 0.95)";
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = "rgba(255,255,255,0.18)";
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);

        ctx.fillStyle = C.fg;
        ctx.font = "11px ui-sans-serif, system-ui, -apple-system, sans-serif";
        ctx.textBaseline = "middle";
        ctx.fillText("Crop preview" + (upstream ? "  \u2022  live image" : "  \u2022  synthetic"),
                     x + PAD, y + PAD + headerH / 2);

        const scale = Math.min(innerW / srcW, innerH / srcH);
        const dispW = Math.max(2, srcW * scale);
        const dispH = Math.max(2, srcH * scale);
        const dispX = innerX + (innerW - dispW) / 2;
        const dispY = innerY + (innerH - dispH) / 2;

        ctx.fillStyle = C.bg;
        ctx.fillRect(dispX, dispY, dispW, dispH);
        ctx.strokeStyle = C.overlay0;
        ctx.lineWidth = 1;
        ctx.strokeRect(dispX + 0.5, dispY + 0.5, dispW - 1, dispH - 1);

        const mxd = dispX + mX * scale;
        const myd = dispY + mY * scale;
        ctx.strokeStyle = C.yellow;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 2]);
        ctx.strokeRect(mxd + 0.5, myd + 0.5, mW * scale - 1, mH * scale - 1);

        const cxd = dispX + cropX * scale;
        const cyd = dispY + cropY * scale;
        ctx.setLineDash([]);
        ctx.strokeStyle = shrinkMode ? C.red : C.okSoft;
        ctx.lineWidth = 2;
        ctx.strokeRect(cxd + 0.5, cyd + 0.5, cropW * scale - 1, cropH * scale - 1);

        ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = C.subtext1;
        ctx.textBaseline = "top";
        const labelY = innerY + innerH + 2;
        const factorLabel = autoFactor
            ? `factor=AUTO ${factor.toFixed(2)}`
            : `factor=${factor.toFixed(2)}`;
        ctx.fillText(`${factorLabel}  preset=${preset}  \u2192  target ${tgtW}\u00d7${tgtH}`,
                     innerX, labelY);
        ctx.restore();
    } catch (e) {
        console.warn("[C2C.InpaintCropPreview] draw failed:", e);
    }
}

function _attachPreviewWidget(node) {
    try {
        if (!node || !Array.isArray(node.widgets)) return;
        if (node.widgets.find(w => w?.name === WIDGET_NAME)) return;
        const widget = {
            type: "custom",
            name: WIDGET_NAME,
            value: null,
            options: { serialize: false },
            serializeValue: () => undefined,
            draw(ctx, n, widget_width, y, _h) {
                _drawPreview(n, ctx, 6, y, Math.max(80, widget_width - 12), BAND_HEIGHT - 4);
            },
            computeSize(width) { return [width, BAND_HEIGHT]; },
        };
        node.widgets.push(widget);

        const wantedNames = [
            "context_from_mask_extend_factor",
            "auto_context_factor",
            "aspect_preset",
            "output_target_width",
            "output_target_height",
        ];
        for (const w of node.widgets) {
            if (!w || !wantedNames.includes(w.name)) continue;
            const orig = w.callback;
            w.callback = function () {
                try { orig?.apply(this, arguments); } catch (e) { /* upstream */ }
                try { app.graph?.setDirtyCanvas?.(true, true); } catch (e) { /* */ }
            };
        }

        try {
            if (typeof node.computeSize === "function") {
                const newSize = node.computeSize();
                if (newSize && Array.isArray(newSize) && newSize[1] > (node.size?.[1] || 0)) {
                    node.size[1] = newSize[1];
                }
            } else if (Array.isArray(node.size)) {
                node.size[1] = (node.size[1] || 200) + BAND_HEIGHT;
            }
        } catch (e) { /* */ }
        try { app.graph?.setDirtyCanvas?.(true, true); } catch (e) { /* */ }
    } catch (e) {
        console.warn("[C2C.InpaintCropPreview] attach failed:", e);
    }
}

app.registerExtension({
    name: EXT_NAME,
    async nodeCreated(node) {
        if (node?.comfyClass !== NODE_NAME) return;
        _attachPreviewWidget(node);
    },
    async loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_NAME) return;
        _attachPreviewWidget(node);
    },
});
