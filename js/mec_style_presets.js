/**
 * mec_style_presets.js — Phase 17c: Quick Style Presets
 *
 * Adds a "🎨 Apply style preset" item to the right-click context menu of
 * CLIPTextEncode-style nodes. Submenu lists curated style modifiers that
 * are appended (or replace existing trailing style block) to the prompt.
 *
 * Setting:
 *   mec.style_presets.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

const PRESETS = {
    "Cinematic":      "cinematic lighting, shallow depth of field, anamorphic lens flare, film grain",
    "Anime":          "anime style, vibrant colors, clean lineart, cel shading, studio ghibli",
    "Oil Painting":   "oil painting, thick brush strokes, impasto, classical art, rich colors",
    "Watercolor":     "watercolor, soft washes, paper texture, light bleeds, hand painted",
    "Photoreal":      "photorealistic, high detail, 50mm lens, natural skin texture, dslr",
    "Cyberpunk":      "cyberpunk, neon lights, rain reflections, holographic billboards, dystopian",
    "Studio Ghibli":  "studio ghibli style, soft palette, painted backgrounds, whimsical",
    "Dark Fantasy":   "dark fantasy, dramatic lighting, painterly, intricate detail, mythic atmosphere",
    "Vintage Film":   "vintage 35mm film, kodachrome, faded colors, light leaks, 1970s",
    "Concept Art":    "concept art, matte painting, dramatic perspective, trending on artstation",
    "Pixel Art":      "pixel art, 16-bit, limited palette, retro game aesthetic",
    "Minimalist":     "minimalist, clean composition, negative space, muted palette",
};

const TAG = "[mec-style]";  // marker so we can replace previously inserted block

function _applyPreset(node, name) {
    const widgets = node.widgets || [];
    const w = widgets.find(x => x && x.name === "text" && (x.type === "customtext" || x.type === "string" || x.type === "STRING"));
    if (!w) {
        console.warn("[MEC.StylePresets] No text widget on", node.type);
        return;
    }
    const suffix = PRESETS[name];
    if (!suffix) return;
    let current = String(w.value || "").trim();
    // Strip any previous MEC style block.
    const markerIdx = current.lastIndexOf(TAG);
    if (markerIdx !== -1) current = current.slice(0, markerIdx).replace(/[,\s]+$/, "");
    const sep = current ? ", " : "";
    w.value = `${current}${sep}${suffix} ${TAG}`;
    if (typeof w.callback === "function") {
        try { w.callback(w.value, app.canvas, node); } catch { /* ignore */ }
    }
    app.canvas?.setDirty?.(true, true);
    console.log(`[MEC.StylePresets] Applied "${name}" to node ${node.id}`);
}

function _isPromptNode(node) {
    return /cliptextencode/i.test(node?.type || "");
}

app.registerExtension({
    name: "MEC.StylePresets",
    settings: [
        {
            id: "mec.style_presets.enabled",
            name: "Style Presets: right-click → Apply style preset",
            tooltip: "Adds a curated style-tag palette to the prompt node context menu.",
            type: "boolean",
            default: true,
        },
    ],
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData?.name) return;
        if (!/CLIPTextEncode/i.test(nodeData.name)) return;
        const orig = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            if (orig) orig.apply(this, arguments);
            const enabled = (() => {
                try { return app.ui.settings.getSettingValue("mec.style_presets.enabled", true); }
                catch { return true; }
            })();
            if (!enabled) return;
            const self = this;
            const submenu = Object.keys(PRESETS).map(name => ({
                content: name,
                callback: () => _applyPreset(self, name),
            }));
            submenu.push(null);
            submenu.push({
                content: "Clear style block",
                callback: () => {
                    const w = (self.widgets || []).find(x => x && x.name === "text");
                    if (!w) return;
                    let v = String(w.value || "");
                    const idx = v.lastIndexOf(TAG);
                    if (idx !== -1) v = v.slice(0, idx).replace(/[,\s]+$/, "");
                    w.value = v;
                    if (typeof w.callback === "function") {
                        try { w.callback(w.value, app.canvas, self); } catch { /* ignore */ }
                    }
                    app.canvas?.setDirty?.(true, true);
                },
            });
            options.unshift({
                content: "🎨 Apply style preset",
                has_submenu: true,
                submenu: { options: submenu },
            });
        };
    },
    async setup() {
        console.log("[MEC.StylePresets] Loaded.");
    },
});
