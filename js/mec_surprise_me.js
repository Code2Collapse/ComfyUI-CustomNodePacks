/**
 * mec_surprise_me.js — Phase 17a: "Surprise Me" button
 *
 * Floating 🎰 button. On click:
 *   1. Randomize every "seed"-named INT widget in the graph.
 *   2. Optionally append a random suffix tag from a small style palette
 *      to the FIRST positive CLIPTextEncode found (controlled by setting).
 *   3. Queue one prompt.
 *
 * Settings:
 *   mec.surprise_me.enabled         — bool (default true)
 *   mec.surprise_me.spice_prompts   — bool (default false) — append style tag
 */

import { app } from "../../scripts/app.js";

const BTN_ID = "mec-surprise-btn";
const STYLE_ID = "mec-surprise-style";

const STYLE_TAGS = [
    "cinematic lighting", "golden hour", "rim light", "soft focus",
    "moody atmosphere", "vibrant colors", "neon accents",
    "studio lighting", "shallow depth of field", "high contrast",
    "watercolor texture", "film grain",
];

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${BTN_ID} {
    position: fixed;
    bottom: 294px;
    right: 16px;
    z-index: 99996;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #1e1e2e;
    border: 1px solid #45475a;
    color: #f5c2e7;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.15s;
}
#${BTN_ID}:hover { border-color: #f5c2e7; transform: rotate(15deg); }
#${BTN_ID}.spinning { animation: mec-surprise-spin 0.8s ease-out; }
@keyframes mec-surprise-spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(720deg); }
}
    `.trim();
    document.head.appendChild(style);
}

function _randomizeSeeds() {
    const g = app.graph;
    if (!g || !g._nodes) return 0;
    let count = 0;
    for (const node of g._nodes) {
        const widgets = node.widgets || [];
        for (const w of widgets) {
            if (!w || !w.name) continue;
            if (!/seed/i.test(w.name)) continue;
            if (typeof w.value !== "number") continue;
            // 32-bit unsigned range — same as ComfyUI's default seed widget.
            const newVal = Math.floor(Math.random() * 0xFFFFFFFF);
            w.value = newVal;
            if (typeof w.callback === "function") {
                try { w.callback(newVal, app.canvas, node); }
                catch { /* ignore */ }
            }
            count++;
        }
    }
    app.canvas?.setDirty?.(true, true);
    return count;
}

function _maybeSpicePrompt() {
    const enable = (() => {
        try { return app.ui.settings.getSettingValue("mec.surprise_me.spice_prompts", false); }
        catch { return false; }
    })();
    if (!enable) return null;
    const g = app.graph;
    if (!g || !g._nodes) return null;
    // First CLIPTextEncode-like node with non-empty text we don't already own.
    for (const node of g._nodes) {
        if (!/cliptextencode/i.test(node.type || "")) continue;
        const w = (node.widgets || []).find(x => x && (x.type === "customtext" || x.type === "string" || x.type === "STRING") && x.name === "text");
        if (!w) continue;
        const original = String(w.value || "").trim();
        if (!original) continue;
        // Skip if it's a negative prompt heuristic.
        if (/\b(worst|bad|low quality|ugly|blurry|nsfw)\b/i.test(original)) continue;
        const tag = STYLE_TAGS[Math.floor(Math.random() * STYLE_TAGS.length)];
        if (original.toLowerCase().includes(tag.toLowerCase())) return null;
        w.value = `${original}, ${tag}`;
        if (typeof w.callback === "function") {
            try { w.callback(w.value, app.canvas, node); } catch { /* ignore */ }
        }
        return { tag, node: node.id };
    }
    return null;
}

async function _surprise() {
    const btn = document.getElementById(BTN_ID);
    if (btn) {
        btn.classList.remove("spinning");
        void btn.offsetWidth;  // restart animation
        btn.classList.add("spinning");
    }
    const seeds = _randomizeSeeds();
    const spice = _maybeSpicePrompt();
    console.log(`[MEC.SurpriseMe] Randomized ${seeds} seed(s)`,
                spice ? `· added "${spice.tag}" to node ${spice.node}` : "");
    try { await app.queuePrompt(0, 1); }
    catch (e) { console.warn("[MEC.SurpriseMe] queue failed:", e); }
}

app.registerExtension({
    name: "MEC.SurpriseMe",
    settings: [
        {
            id: "mec.surprise_me.enabled",
            name: "Surprise Me: enabled",
            tooltip: "Show 🎰 button that randomizes seeds and queues a fresh run.",
            type: "boolean",
            defaultValue: true,
            onChange: (v) => {
                const b = document.getElementById(BTN_ID);
                if (b) b.style.display = v ? "flex" : "none";
            },
        },
        {
            id: "mec.surprise_me.spice_prompts",
            name: "Surprise Me: also add a random style tag to the prompt",
            tooltip: "Appends a small lighting/style tag (e.g. 'cinematic lighting') to the first CLIPTextEncode prompt.",
            type: "boolean",
            defaultValue: false,
        },
    ],
    async setup() {
        _injectStyle();
        if (!document.getElementById(BTN_ID)) {
            const b = document.createElement("button");
            b.id = BTN_ID;
            b.title = "Surprise me!";
            b.textContent = "🎰";
            b.addEventListener("click", _surprise);
            document.body.appendChild(b);
        }
        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("mec.surprise_me.enabled", true); }
            catch { return true; }
        })();
        document.getElementById(BTN_ID).style.display = enabled ? "flex" : "none";
        console.log("[MEC.SurpriseMe] Loaded.");
    },
});
