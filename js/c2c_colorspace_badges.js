/**
 * mec_colorspace_badges.js — Phase 19: OCIO / Colorspace badges
 *
 * Draws a small colored badge in the top-right of IMAGE-typed nodes
 * indicating an inferred colorspace tag. Heuristics:
 *   - Node type contains "OCIO", "ACES", "ACEScg", "Linear", "sRGB",
 *     "Rec.709", "Log", "Linearize", "ColorMatch" → use that.
 *   - Node type contains "Save"/"Preview"/"Load" with "exr" hint → "linear".
 *   - VAEDecode-style nodes → "sRGB" output.
 *   - Otherwise no badge.
 *
 * Setting:
 *   mec.colorspace_badges.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";
import { C } from "./_c2c_theme.js";

// Canvas 2D cannot resolve CSS var() strings (renders black/invisible), so we
// map each colorspace tag to the resolved-hex value from the theme palette.
const CS_COLORS = {
    "sRGB":     C.okSoft,
    "Linear":   C.blue,
    "ACEScg":   C.mauve,
    "ACES":     C.mauve,
    "Rec.709":  C.sapphire,
    "Log":      C.yellow,
    "Raw":      C.overlay0,
    "OCIO":     C.pink,
};

// Perf: this enabled flag is read by _drawBadge(), which runs per node per
// frame. Cache it (seeded at setup() + onChange) so getSettingValue is not a
// synchronous storage hit in the render loop.
let _enabled = true;

function _settingsEnabled() {
    try { return app.ui.settings.getSettingValue("mec.colorspace_badges.enabled", true); }
    catch { return true; }
}

function _inferTag(node) {
    const t = (node?.type || "") + " " + (node?.title || "");
    const lower = t.toLowerCase();
    if (/acescg/i.test(t)) return "ACEScg";
    if (/aces/i.test(t)) return "ACES";
    if (/linearize|to[\s_-]?linear|toscenelinear/i.test(t)) return "Linear";
    if (/\blinear\b/i.test(t)) return "Linear";
    if (/rec\.?709/i.test(t)) return "Rec.709";
    if (/(?:^|[^a-z])log(?:[^a-z]|$)|tolog|alexa[_\s]?log|slog/i.test(t)) return "Log";
    if (/\bsrgb\b/i.test(t) || /tosrgb|srgb_out/i.test(t)) return "sRGB";
    if (/ocio/i.test(t)) return "OCIO";
    if (/\braw\b/i.test(t)) return "Raw";
    // VAEDecode outputs sRGB in standard SD/SDXL/Flux/SD3 pipelines.
    if (/^vaedecode/i.test(t)) return "sRGB";
    // EXR savers/loaders → linear by convention.
    if (/exr/i.test(lower) && /(save|load|writer|reader)/i.test(lower)) return "Linear";
    return null;
}

function _hasImagePort(node) {
    const has = (arr) => Array.isArray(arr) && arr.some(p => (p?.type || "").toUpperCase() === "IMAGE");
    return has(node?.outputs) || has(node?.inputs);
}

function _drawBadge(node, ctx) {
    if (!_enabled) return;
    if (!_hasImagePort(node)) return;
    const tag = _inferTag(node);
    if (!tag) return;
    const color = CS_COLORS[tag] || C.blue;
    const w = node.size?.[0] || 200;
    const padX = 6, padY = -16;  // sit above the node title bar
    // Measure
    ctx.save();
    ctx.font = "bold 10px -apple-system, Segoe UI, sans-serif";
    const text = tag;
    const metrics = ctx.measureText(text);
    const boxW = metrics.width + 10;
    const boxH = 14;
    const x = w - boxW - padX;
    const y = padY;
    ctx.fillStyle = "rgba(30, 30, 46, 0.95)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y, boxW, boxH, 4);
        ctx.fill();
        ctx.stroke();
    } else {
        ctx.fillRect(x, y, boxW, boxH);
        ctx.strokeRect(x, y, boxW, boxH);
    }
    ctx.fillStyle = color;
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + 5, y + boxH / 2);
    ctx.restore();
}

function _patch() {
    if (LGraphCanvas.prototype._mecColorspacePatched) return;
    LGraphCanvas.prototype._mecColorspacePatched = true;
    // Hook into per-node draw via LGraphCanvas.prototype.drawNode by wrapping
    // the original to draw the badge AFTER the node frame.
    const orig = LGraphCanvas.prototype.drawNode;
    LGraphCanvas.prototype.drawNode = function (node, ctx) {
        const r = orig.apply(this, arguments);
        try {
            if (node && !node.flags?.collapsed) _drawBadge(node, ctx);
        } catch { /* never break drawing */ }
        return r;
    };
}

app.registerExtension({
    name: "C2C.ColorspaceBadges",
    settings: [
        {
            id: "mec.colorspace_badges.enabled",
            name: "Colorspace Badges: show inferred OCIO/colorspace on IMAGE nodes",
            type: "boolean",
            default: true,
            onChange: (v) => { _enabled = (v !== false); },
        },
    ],
    async setup() {
        _enabled = _settingsEnabled();
        _patch();
        console.log("[MEC.ColorspaceBadges] Loaded.");
    },
});
