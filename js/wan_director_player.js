// wan_director_player.js — live preview player embedded in WanDirectorC2C.
//
// Responsibilities (kept deliberately small):
//   1. A DOM widget rendering the timeline's image clips as a scrubbable
//      thumbnail strip (reads `timeline_data` JSON, parses `segments` of
//      type "image", decodes imageB64 / imageFile to <img> tags).
//   2. A <video> preview area: when the node executes and the upstream
//      VHS_VideoCombine (or any node that writes a file to /output and
//      reports it via `ui.gifs` / `ui.images` / `ui.videos`) returns a
//      url-shaped result, we plug that url into the <video> tag.
//   3. Status text fed from the node's `info` string output (parsed JSON).
//   4. Play / pause / step buttons; arrow-key scrub when focused.
//
// No external deps. Works on any browser ComfyUI runs in.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PLAYER_H = 280;          // total player widget height (px)
const STRIP_H  = 64;            // thumbnail strip height
const BTN_H    = 28;
const FPS_DEFAULT = 16;

function _b64Url(s) {
    if (!s) return null;
    if (s.startsWith("data:")) return s;
    if (s.startsWith("/view?")) return s;
    return "data:image/png;base64," + s;
}

function _resolveImage(seg) {
    if (seg.imageFile) {
        // ComfyUI serves /input via /view
        return `/view?filename=${encodeURIComponent(seg.imageFile)}&type=input&subfolder=`;
    }
    if (seg.imageB64) return _b64Url(seg.imageB64);
    return null;
}

function makePlayerDOM(node) {
    const root = document.createElement("div");
    root.style.cssText = `
        position: relative; width: 100%; height: ${PLAYER_H}px;
        background: var(--c2c-gray950); border: 1px solid var(--c2c-gray700); border-radius: 4px;
        display: flex; flex-direction: column; font-family: system-ui, sans-serif;
        color: var(--c2c-gray150); box-sizing: border-box; overflow: hidden;
    `;

    // ── Video / image preview area ────────────────────────────────
    const stage = document.createElement("div");
    stage.style.cssText = `
        flex: 1 1 auto; min-height: 0; position: relative;
        background: var(--c2c-black); display: flex; align-items: center; justify-content: center;
    `;
    const stageImg = document.createElement("img");
    stageImg.style.cssText = "max-width: 100%; max-height: 100%; display: none;";
    const stageVideo = document.createElement("video");
    stageVideo.controls = true;
    stageVideo.style.cssText = "max-width: 100%; max-height: 100%; display: none;";
    const stageMsg = document.createElement("div");
    stageMsg.style.cssText = "color:var(--c2c-gray400); font-size:12px; padding:8px; text-align:center;";
    stageMsg.textContent = "No preview yet — add image/video clips or run the workflow.";
    stage.append(stageImg, stageVideo, stageMsg);

    // ── Controls row ──────────────────────────────────────────────
    const controls = document.createElement("div");
    controls.style.cssText = `
        flex: 0 0 ${BTN_H}px; display: flex; align-items: center; gap: 6px;
        padding: 0 6px; background: var(--c2c-neutral910); border-top: 1px solid var(--c2c-gray800);
    `;
    const mkBtn = (label, title) => {
        const b = document.createElement("button");
        b.textContent = label; b.title = title;
        b.style.cssText = "background:var(--c2c-gray800);color:var(--c2c-gray150);border:1px solid var(--c2c-gray600);border-radius:3px;padding:2px 8px;cursor:pointer;font-size:12px;";
        return b;
    };
    const btnPrev = mkBtn("⏮", "Previous clip");
    const btnPlay = mkBtn("▶", "Play / pause");
    const btnNext = mkBtn("⏭", "Next clip");
    const status  = document.createElement("span");
    status.style.cssText = "flex:1 1 auto;font-size:11px;color:var(--c2c-gray300);margin-left:8px;";
    status.textContent = "idle";
    controls.append(btnPrev, btnPlay, btnNext, status);

    // ── Thumbnail strip ───────────────────────────────────────────
    const strip = document.createElement("div");
    strip.style.cssText = `
        flex: 0 0 ${STRIP_H}px; display: flex; gap: 4px;
        padding: 4px; overflow-x: auto; background: var(--c2c-neutral940);
        border-top: 1px solid var(--c2c-gray800);
    `;

    root.append(stage, controls, strip);

    // ── State ─────────────────────────────────────────────────────
    const state = {
        clips: [],
        active: -1,
        playing: false,
        timer: null,
        fps: FPS_DEFAULT,
    };

    function showImage(url) {
        stageVideo.pause();
        stageVideo.style.display = "none";
        stageMsg.style.display = "none";
        stageImg.src = url;
        stageImg.style.display = "block";
    }
    function showVideo(url) {
        stageImg.style.display = "none";
        stageMsg.style.display = "none";
        stageVideo.src = url;
        stageVideo.style.display = "block";
        stageVideo.play().catch(() => {});
    }
    function showMessage(msg) {
        stageImg.style.display = "none";
        stageVideo.pause();
        stageVideo.style.display = "none";
        stageMsg.style.display = "block";
        stageMsg.textContent = msg;
    }

    function selectClip(i) {
        if (state.clips.length === 0) { showMessage("No image clips on the timeline."); return; }
        i = ((i % state.clips.length) + state.clips.length) % state.clips.length;
        state.active = i;
        const clip = state.clips[i];
        showImage(clip.url);
        status.textContent = `clip ${i + 1}/${state.clips.length} · frame ${clip.start}\u2013${clip.start + clip.length}`;
        // highlight thumb
        strip.querySelectorAll("[data-thumb]").forEach((el, idx) => {
            el.style.outline = (idx === i) ? "2px solid var(--c2c-cyanBright2)" : "1px solid var(--c2c-gray700)";
        });
    }

    function rebuildStrip() {
        strip.replaceChildren();
        state.clips.forEach((clip, i) => {
            const tile = document.createElement("div");
            tile.setAttribute("data-thumb", String(i));
            tile.style.cssText = `
                flex: 0 0 ${Math.max(48, STRIP_H - 16)}px; height: ${STRIP_H - 16}px;
                background: var(--c2c-black) center/contain no-repeat url("${clip.url}");
                border-radius: 2px; cursor: pointer; outline: 1px solid var(--c2c-gray700);
            `;
            tile.title = `${clip.start}–${clip.start + clip.length}f${clip.text ? ` · ${clip.text}` : ""}`;
            tile.onclick = () => selectClip(i);
            strip.appendChild(tile);
        });
    }

    function rescan() {
        const tdW = node.widgets?.find(w => w.name === "timeline_data");
        const fpsW = node.widgets?.find(w => w.name === "frame_rate");
        state.fps = (fpsW?.value && fpsW.value > 0) ? fpsW.value : FPS_DEFAULT;
        let td = {};
        try { td = JSON.parse(tdW?.value || "{}"); } catch { td = {}; }
        const segs = (td.segments || [])
            .filter(s => (s.type || "image") === "image")
            .filter(s => s.imageFile || s.imageB64)
            .sort((a, b) => (a.start || 0) - (b.start || 0));
        state.clips = segs.map(s => ({
            url: _resolveImage(s),
            start: parseInt(s.start || 0, 10),
            length: parseInt(s.length || 1, 10),
            text: (s.text || "").slice(0, 64),
        })).filter(c => c.url);
        rebuildStrip();
        if (state.clips.length > 0) {
            const keep = Math.min(state.active, state.clips.length - 1);
            selectClip(Math.max(0, keep));
        } else {
            showMessage("No image clips on the timeline.");
            status.textContent = "idle";
        }
    }

    function play() {
        if (state.playing || state.clips.length < 2) return;
        state.playing = true;
        btnPlay.textContent = "⏸";
        const tick = () => {
            if (!state.playing) return;
            const cur = state.clips[state.active] || state.clips[0];
            const dwellMs = Math.max(120, (cur.length / Math.max(1, state.fps)) * 1000);
            state.timer = setTimeout(() => {
                selectClip(state.active + 1);
                tick();
            }, dwellMs);
        };
        tick();
    }
    function pause() {
        state.playing = false;
        btnPlay.textContent = "▶";
        if (state.timer) { clearTimeout(state.timer); state.timer = null; }
    }

    btnPrev.onclick = () => { pause(); selectClip(state.active - 1); };
    btnNext.onclick = () => { pause(); selectClip(state.active + 1); };
    btnPlay.onclick = () => { state.playing ? pause() : play(); };

    return { root, rescan, showVideo, showImage, showMessage,
             setStatus: (s) => status.textContent = s };
}

app.registerExtension({
    name: "C2C.WanDirector.Player",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "WanDirectorC2C") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);

            const player = makePlayerDOM(this);
            // Register a DOM widget so ComfyUI lays it out with the node.
            const widget = this.addDOMWidget("wan_director_live_player", "div", player.root, {
                serialize: false,
                hideOnZoom: false,
            });
            // Use the inset `width` LiteGraph passes — never `this.size[0]`,
            // which over-reserves the column and leaks the node bgcolor as
            // dark gutters on both edges of the widget.
            widget.computeSize = (width) => [width, PLAYER_H];
            this._wd_player = player;

            // Refresh thumbnails when timeline_data is updated.
            const tdW = this.widgets?.find(w => w.name === "timeline_data");
            if (tdW) {
                const orig = tdW.callback;
                tdW.callback = (...args) => {
                    const v = orig?.apply(tdW, args);
                    try { player.rescan(); } catch (e) { console.warn("WanDirector player rescan failed", e); }
                    return v;
                };
            }
            // Initial scan after the rest of the node is wired.
            setTimeout(() => { try { player.rescan(); } catch {} }, 50);

            return r;
        };

        // Hook executed events to display final video / image / info text.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (!this._wd_player) return;
            try {
                // ComfyUI bundles UI-side output under message; we look for
                // a video/image url plus the `info` text from RETURN_NAMES[7].
                const videos = message?.videos || message?.gifs;
                if (Array.isArray(videos) && videos.length) {
                    const v = videos[0];
                    const url = `/view?filename=${encodeURIComponent(v.filename)}` +
                                `&type=${encodeURIComponent(v.type || "output")}` +
                                `&subfolder=${encodeURIComponent(v.subfolder || "")}`;
                    this._wd_player.showVideo(url);
                } else if (Array.isArray(message?.images) && message.images.length) {
                    const im = message.images[0];
                    const url = `/view?filename=${encodeURIComponent(im.filename)}` +
                                `&type=${encodeURIComponent(im.type || "output")}` +
                                `&subfolder=${encodeURIComponent(im.subfolder || "")}`;
                    this._wd_player.showImage(url);
                }
                // Plain text outputs (e.g. our `info` JSON) come in `text`
                // when ShowText-style preview nodes are present; we surface
                // a short status if available.
                if (Array.isArray(message?.text) && message.text.length) {
                    try {
                        const j = JSON.parse(message.text[0]);
                        const s = `${j.backend} · ${j.label} · ${j.frames}f@${j.fps}fps` +
                                  (j.prompt_relay?.applied ? ` · PR(${j.prompt_relay.note})` : "");
                        this._wd_player.setStatus(s);
                    } catch {
                        this._wd_player.setStatus(String(message.text[0]).slice(0, 200));
                    }
                }
            } catch (e) {
                console.warn("WanDirector onExecuted handler:", e);
            }
        };
    },
});
