/**
 * mec_frame_overlay.js — Phase 14: Frame Range Overlay
 *
 * Detects when a node's preview holds multiple images (a batch / animation)
 * and overlays a slim frame-scrubber underneath the preview. Plays through
 * frames in-place at the chosen FPS.
 *
 * Works for ANY node whose execution result has `images: [{filename,
 * subfolder, type}, ...]` with length > 1 (PreviewImage, SaveImage,
 * VHS_VideoCombine images, etc.). The existing VideoFramePlayerMEC keeps
 * its own UI; this is the generic embed for everyone else.
 *
 * Setting:
 *   mec.frame_overlay.enabled — bool (default true)
 *   mec.frame_overlay.fps      — number (default 12)
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE_ID = "mec-frame-overlay-style";
const STATE_KEY = "_mecFrameOverlay";

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.mec-frame-overlay {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 6px;
    background: rgba(24, 24, 37, 0.85);
    border-top: 1px solid var(--c2c-surface1);
    font-family: monospace;
    font-size: 10px;
    color: var(--c2c-fg);
}
.mec-frame-overlay button {
    background: var(--c2c-surface0);
    border: 1px solid var(--c2c-surface1);
    color: var(--c2c-fg);
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 11px;
    cursor: pointer;
}
.mec-frame-overlay button:hover { border-color: var(--c2c-blue); }
.mec-frame-overlay input[type=range] {
    flex: 1;
    accent-color: var(--c2c-blue);
}
.mec-frame-overlay .fo-counter {
    min-width: 56px;
    text-align: right;
    color: var(--c2c-okSoft);
}
    `.trim();
    document.head.appendChild(style);
}

function _urlForImage(img) {
    if (!img || !img.filename) return null;
    const params = new URLSearchParams({
        filename:  img.filename,
        subfolder: img.subfolder || "",
        type:      img.type      || "output",
    });
    return `${api.api_base || ""}/view?${params.toString()}`;
}

function _imageElForNode(node) {
    // ComfyUI stores preview elements differently across releases; probe known
    // places without breaking if missing.
    if (node.imgs && node.imgs[0]) return node.imgs[0];
    if (node.preview_img) return node.preview_img;
    return null;
}

function _ensureState(node) {
    if (!node[STATE_KEY]) {
        node[STATE_KEY] = {
            frames:   [],
            cursor:   0,
            playing:  false,
            interval: null,
            domAttached: false,
        };
    }
    return node[STATE_KEY];
}

function _stopPlayback(state) {
    if (state.interval) {
        clearInterval(state.interval);
        state.interval = null;
    }
    state.playing = false;
}

function _showFrame(node, state, idx) {
    if (!state.frames.length) return;
    state.cursor = ((idx % state.frames.length) + state.frames.length) % state.frames.length;
    const url = _urlForImage(state.frames[state.cursor]);
    if (!url) return;
    if (!node.imgs) node.imgs = [];
    if (!node.imgs[0] || !(node.imgs[0] instanceof Image)) {
        const im = new Image();
        node.imgs[0] = im;
    }
    node.imgs[0].src = url;
    node.imgs[0].onload = () => node.setDirtyCanvas?.(true, true);
    node.setDirtyCanvas?.(true, true);
}

function _renderControl(node, state) {
    const enabled = (() => {
        try { return app.ui.settings.getSettingValue("mec.frame_overlay.enabled", true); }
        catch { return true; }
    })();
    if (!enabled || state.frames.length <= 1) {
        if (state.controlEl) {
            state.controlEl.remove();
            state.controlEl = null;
        }
        return;
    }
    if (state.controlEl) return; // already rendered

    const root = document.createElement("div");
    root.className = "mec-frame-overlay";
    root.innerHTML = `
        <button class="fo-prev"  title="Previous frame">◀</button>
        <button class="fo-play"  title="Play / pause">▶</button>
        <button class="fo-next"  title="Next frame">▶|</button>
        <input type="range" min="0" max="${state.frames.length - 1}" value="${state.cursor}">
        <span class="fo-counter">${state.cursor + 1} / ${state.frames.length}</span>
    `;
    const slider  = root.querySelector("input[type=range]");
    const counter = root.querySelector(".fo-counter");

    const update = () => {
        slider.max = String(state.frames.length - 1);
        slider.value = String(state.cursor);
        counter.textContent = `${state.cursor + 1} / ${state.frames.length}`;
    };

    slider.addEventListener("input", () => {
        _stopPlayback(state);
        _showFrame(node, state, parseInt(slider.value, 10));
        update();
    });
    root.querySelector(".fo-prev").addEventListener("click", () => {
        _stopPlayback(state);
        _showFrame(node, state, state.cursor - 1);
        update();
    });
    root.querySelector(".fo-next").addEventListener("click", () => {
        _stopPlayback(state);
        _showFrame(node, state, state.cursor + 1);
        update();
    });
    const playBtn = root.querySelector(".fo-play");
    playBtn.addEventListener("click", () => {
        if (state.playing) {
            _stopPlayback(state);
            playBtn.textContent = "▶";
            return;
        }
        const fps = (() => {
            try { return app.ui.settings.getSettingValue("mec.frame_overlay.fps", 12); }
            catch { return 12; }
        })();
        const delay = Math.max(20, 1000 / Math.max(1, fps));
        state.playing = true;
        playBtn.textContent = "⏸";
        state.interval = setInterval(() => {
            _showFrame(node, state, state.cursor + 1);
            update();
        }, delay);
    });

    // Attach as a DOM widget so it lives inside the node body.
    try {
        node.addDOMWidget(
            "c2c_frame_overlay",
            "c2c_frame_overlay",
            root,
            { serialize: false, hideOnZoom: false },
        );
    } catch (e) {
        // Older ComfyUI without addDOMWidget — fallback: just leave the
        // arrow keys functional via the patched event hook below.
        console.debug("[MEC.FrameOverlay] addDOMWidget unsupported:", e);
    }
    state.controlEl = root;
    state.update = update;
}

function _onExecuted(node, output) {
    if (!output || !Array.isArray(output.images)) return;
    if (output.images.length <= 1) {
        // Single image — clear any previous batch state.
        if (node[STATE_KEY]) {
            _stopPlayback(node[STATE_KEY]);
            node[STATE_KEY].frames = [];
            if (node[STATE_KEY].controlEl) {
                node[STATE_KEY].controlEl.remove();
                node[STATE_KEY].controlEl = null;
            }
        }
        return;
    }
    const state = _ensureState(node);
    _stopPlayback(state);
    state.frames = output.images.slice();
    state.cursor = 0;
    _renderControl(node, state);
    if (state.update) state.update();
}

app.registerExtension({
    name: "C2C.FrameOverlay",
    settings: [
        {
            id: "mec.frame_overlay.enabled",
            name: "Frame Overlay: scrubber on multi-image previews",
            type: "boolean",
            default: true,
            onChange: () => app.canvas?.setDirty?.(true, true),
        },
        {
            id: "mec.frame_overlay.fps",
            name: "Frame Overlay: playback FPS",
            type: "number",
            default: 12,
            attrs: { min: 1, max: 60, step: 1 },
        },
    ],
    async beforeRegisterNodeDef(nodeType, _nodeData, _appRef) {
        const orig = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const r = orig?.apply(this, arguments);
            try { _onExecuted(this, output); }
            catch (e) { console.warn("[MEC.FrameOverlay] onExecuted error:", e); }
            return r;
        };
    },
    async setup() {
        _injectStyle();
        console.log("[MEC.FrameOverlay] Loaded.");
    },
});
