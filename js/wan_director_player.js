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

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const PLAYER_H = 168;          // total player widget height (px) — was 280 (node-stack bloat); 168 keeps stage usable + transport
const STRIP_H  = 56;            // thumbnail strip height
const SCRUB_H  = 12;            // scrubber/seek-bar row height
const BTN_H    = 28;
const FPS_DEFAULT = 16;
const SPEEDS   = [0.5, 1, 2, 4];

function _fmtTime(sec) {
    if (!isFinite(sec) || sec < 0) sec = 0;
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
}

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
    root.className = "wd-live-player-root";
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
    stageVideo.controls = false;        // we provide our own transport (scrubber/step/loop/speed/in-out)
    stageVideo.playsInline = true;
    stageVideo.style.cssText = "max-width: 100%; max-height: 100%; display: none;";
    const stageMsg = document.createElement("div");
    stageMsg.style.cssText = "color:var(--c2c-gray400); font-size:12px; padding:8px; text-align:center;";
    stageMsg.textContent = "No preview yet — add image/video clips or run the workflow.";
    stage.append(stageImg, stageVideo, stageMsg);

    // ── Scrubber / seek bar (click + drag to seek; shows in/out region) ──
    const scrub = document.createElement("div");
    scrub.style.cssText = `
        flex: 0 0 ${SCRUB_H}px; position: relative; cursor: pointer;
        background: var(--c2c-gray900); border-top: 1px solid var(--c2c-gray800);
    `;
    const scrubRegion = document.createElement("div");   // shaded in→out region
    scrubRegion.style.cssText = "position:absolute;top:0;bottom:0;left:0;right:0;background:var(--c2c-cyanBright2);opacity:0.14;pointer-events:none;";
    const scrubFill = document.createElement("div");      // played portion
    scrubFill.style.cssText = "position:absolute;top:0;bottom:0;left:0;width:0%;background:var(--c2c-cyanBright2);opacity:0.55;pointer-events:none;";
    const scrubHead = document.createElement("div");      // playhead
    scrubHead.style.cssText = "position:absolute;top:-2px;bottom:-2px;left:0;width:2px;background:var(--c2c-gray50);box-shadow:0 0 3px var(--c2c-black);pointer-events:none;";
    const mkMark = (color) => { const m = document.createElement("div"); m.style.cssText = `position:absolute;top:0;bottom:0;width:2px;background:${color};display:none;pointer-events:none;`; return m; };
    const markIn  = mkMark("var(--c2c-green, #4ade80)");
    const markOut = mkMark("var(--c2c-amber, #fbbf24)");
    scrub.append(scrubRegion, scrubFill, scrubHead, markIn, markOut);

    // ── Controls row ──────────────────────────────────────────────
    const controls = document.createElement("div");
    controls.style.cssText = `
        flex: 0 0 ${BTN_H}px; display: flex; align-items: center; gap: 4px;
        padding: 0 6px; background: var(--c2c-neutral910); border-top: 1px solid var(--c2c-gray800);
    `;
    const mkBtn = (label, title) => {
        const b = document.createElement("button");
        b.textContent = label; b.title = title;
        b.style.cssText = "background:var(--c2c-gray800);color:var(--c2c-gray150);border:1px solid var(--c2c-gray600);border-radius:3px;padding:2px 6px;cursor:pointer;font-size:12px;line-height:1;";
        return b;
    };
    const btnPrev  = mkBtn("⏮", "Previous clip");
    const btnFprev = mkBtn("◀ǀ", "Step back 1 frame  (←)");
    const btnPlay  = mkBtn("▶", "Play / pause  (Space)");
    const btnFnext = mkBtn("ǀ▶", "Step forward 1 frame  (→)");
    const btnNext  = mkBtn("⏭", "Next clip");
    const btnLoop  = mkBtn("🔁", "Loop  (off)");
    btnLoop.style.opacity = "0.45";
    const btnSpeed = mkBtn("1×", "Playback speed");
    const btnIn    = mkBtn("[", "Set IN point  (i)");
    const btnOut   = mkBtn("]", "Set OUT point  (o)");
    const btnClr   = mkBtn("⌫", "Clear IN/OUT trim");
    const tc = document.createElement("span");      // timecode + frame readout
    tc.style.cssText = "font-size:11px;color:var(--c2c-gray200);margin-left:4px;font-variant-numeric:tabular-nums;white-space:nowrap;";
    tc.textContent = "0:00";
    const status  = document.createElement("span");
    status.style.cssText = "flex:1 1 auto;font-size:11px;color:var(--c2c-gray300);margin-left:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    status.textContent = "idle";
    controls.append(btnPrev, btnFprev, btnPlay, btnFnext, btnNext, btnLoop, btnSpeed, btnIn, btnOut, btnClr, tc, status);

    // ── Thumbnail strip ───────────────────────────────────────────
    const strip = document.createElement("div");
    strip.style.cssText = `
        flex: 0 0 ${STRIP_H}px; display: flex; gap: 4px;
        padding: 4px; overflow-x: auto; background: var(--c2c-neutral940);
        border-top: 1px solid var(--c2c-gray800);
    `;

    root.tabIndex = 0;                          // focusable → keyboard transport
    root.style.outline = "none";
    root.append(stage, scrub, controls, strip);

    // ── State ─────────────────────────────────────────────────────
    const state = {
        clips: [],
        active: -1,
        playing: false,
        timer: null,
        fps: FPS_DEFAULT,
        mode: "clips",        // "clips" (image strip) | "video" (executed render) | "image" (single)
        loop: false,
        speedIdx: 1,          // index into SPEEDS, default 1×
        inFrac: null,         // trim IN  as 0..1 fraction of timeline
        outFrac: null,        // trim OUT as 0..1 fraction of timeline
    };

    function showImage(url) {
        stageVideo.pause();
        stageVideo.style.display = "none";
        stageMsg.style.display = "none";
        stageImg.src = url;
        stageImg.style.display = "block";
    }
    function showVideo(url) {            // executed render → full video transport
        state.mode = "video";
        state.inFrac = state.outFrac = null;     // fresh source → clear trim
        stageImg.style.display = "none";
        stageMsg.style.display = "none";
        stageVideo.src = url;
        stageVideo.style.display = "block";
        syncSpeed();
        stageVideo.play().catch(() => {});
    }
    function showMessage(msg) {
        stageImg.style.display = "none";
        stageVideo.pause();
        stageVideo.style.display = "none";
        stageMsg.style.display = "block";
        stageMsg.textContent = msg;
    }

    // ── transport core (works for both video + image-clip modes) ──
    const vidDur = () => (state.mode === "video" && isFinite(stageVideo.duration)) ? stageVideo.duration : 0;
    function curFrac() {
        if (state.mode === "video") { const d = vidDur(); return d > 0 ? Math.min(1, stageVideo.currentTime / d) : 0; }
        const n = state.clips.length; return n > 0 ? Math.max(0, state.active) / n : 0;
    }
    function updateScrub() {
        const frac = curFrac();
        scrubFill.style.width = (frac * 100) + "%";
        scrubHead.style.left  = (frac * 100) + "%";
        const setMark = (m, f) => { if (f == null) m.style.display = "none"; else { m.style.display = "block"; m.style.left = (f * 100) + "%"; } };
        setMark(markIn, state.inFrac);
        setMark(markOut, state.outFrac);
        const lo = state.inFrac == null ? 0 : state.inFrac, hi = state.outFrac == null ? 1 : state.outFrac;
        scrubRegion.style.left = (lo * 100) + "%";
        scrubRegion.style.right = ((1 - hi) * 100) + "%";
        scrubRegion.style.display = (state.inFrac != null || state.outFrac != null) ? "block" : "none";
        if (state.mode === "video") {
            tc.textContent = `${_fmtTime(stageVideo.currentTime)} / ${_fmtTime(vidDur())} · f${Math.round(stageVideo.currentTime * state.fps)}`;
        } else if (state.clips.length) {
            tc.textContent = `clip ${state.active + 1}/${state.clips.length}`;
        } else {
            tc.textContent = "0:00";
        }
    }
    function seekToFrac(frac) {
        frac = Math.min(1, Math.max(0, frac));
        if (state.mode === "video") { const d = vidDur(); if (d > 0) stageVideo.currentTime = frac * d; }
        else if (state.clips.length) selectClip(Math.min(state.clips.length - 1, Math.floor(frac * state.clips.length)));
        updateScrub();
    }
    function frameStep(dir) {
        pause();
        if (state.mode === "video") {
            const d = vidDur();
            if (d > 0) stageVideo.currentTime = Math.min(d, Math.max(0, stageVideo.currentTime + dir / Math.max(1, state.fps)));
            updateScrub();
        } else {
            selectClip(state.active + dir);
        }
    }
    function syncSpeed() {
        const spd = SPEEDS[state.speedIdx] || 1;
        btnSpeed.textContent = spd + "×";
        if (state.mode === "video") stageVideo.playbackRate = spd;
    }
    function cycleSpeed() { state.speedIdx = (state.speedIdx + 1) % SPEEDS.length; syncSpeed(); }
    function toggleLoop() {
        state.loop = !state.loop;
        btnLoop.style.opacity = state.loop ? "1" : "0.45";
        btnLoop.title = "Loop  (" + (state.loop ? "on" : "off") + ")";
    }
    function setIn()  { state.inFrac  = curFrac(); if (state.outFrac != null && state.outFrac <= state.inFrac) state.outFrac = null; updateScrub(); }
    function setOut() { state.outFrac = curFrac(); if (state.inFrac  != null && state.inFrac  >= state.outFrac) state.inFrac  = null; updateScrub(); }
    function clearTrim() { state.inFrac = state.outFrac = null; updateScrub(); }

    // Video-driven scrubber + in/out + loop enforcement.
    stageVideo.addEventListener("timeupdate", () => {
        if (state.mode !== "video") return;
        const d = vidDur();
        if (d > 0) {
            const hi = (state.outFrac == null ? 1 : state.outFrac) * d;
            const lo = (state.inFrac  == null ? 0 : state.inFrac)  * d;
            if (stageVideo.currentTime >= hi - 1e-3) {
                if (state.loop) { stageVideo.currentTime = lo; stageVideo.play().catch(() => {}); }
                else { stageVideo.pause(); }
            }
        }
        updateScrub();
    });
    stageVideo.addEventListener("play",  () => { state.playing = true;  btnPlay.textContent = "⏸"; });
    stageVideo.addEventListener("pause", () => { state.playing = false; btnPlay.textContent = "▶"; });
    stageVideo.addEventListener("loadedmetadata", () => { syncSpeed(); updateScrub(); });

    // Scrubber pointer interaction (click + drag to seek).
    function scrubToEvent(ev) {
        const r = scrub.getBoundingClientRect();
        seekToFrac((ev.clientX - r.left) / Math.max(1, r.width));
    }
    let scrubbing = false;
    scrub.addEventListener("pointerdown", (ev) => { scrubbing = true; try { scrub.setPointerCapture(ev.pointerId); } catch {} pause(); scrubToEvent(ev); });
    scrub.addEventListener("pointermove", (ev) => { if (scrubbing) scrubToEvent(ev); });
    scrub.addEventListener("pointerup",   (ev) => { scrubbing = false; try { scrub.releasePointerCapture(ev.pointerId); } catch {} });

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
        updateScrub();
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
            // Only drop into clip-preview mode if we're not already showing an
            // executed video (rescan fires on timeline edits, not just init).
            if (state.mode !== "video") state.mode = "clips";
            const keep = Math.min(state.active, state.clips.length - 1);
            selectClip(Math.max(0, keep));
        } else {
            if (state.mode !== "video") showMessage("No image clips on the timeline.");
            status.textContent = "idle";
        }
        updateScrub();
    }

    function play() {
        if (state.mode === "video") {
            const d = vidDur();
            if (d > 0) {                          // resume inside the [in,out] region
                const hi = (state.outFrac == null ? 1 : state.outFrac) * d;
                const lo = (state.inFrac  == null ? 0 : state.inFrac)  * d;
                if (stageVideo.currentTime >= hi - 1e-3 || stageVideo.currentTime < lo) stageVideo.currentTime = lo;
            }
            stageVideo.play().catch(() => {});
            return;
        }
        if (state.playing || state.clips.length < 2) return;
        state.playing = true;
        btnPlay.textContent = "⏸";
        const tick = () => {
            if (!state.playing) return;
            const cur = state.clips[state.active] || state.clips[0];
            const spd = SPEEDS[state.speedIdx] || 1;
            const dwellMs = Math.max(60, (cur.length / Math.max(1, state.fps)) * 1000 / spd);
            state.timer = setTimeout(() => {
                const n = state.clips.length;
                const loIdx = state.inFrac  == null ? 0     : Math.floor(state.inFrac  * n);
                const hiIdx = state.outFrac == null ? n - 1 : Math.min(n - 1, Math.floor(state.outFrac * n));
                let nxt = state.active + 1;
                if (nxt > hiIdx) { if (state.loop) nxt = loIdx; else { pause(); return; } }
                selectClip(nxt);
                tick();
            }, dwellMs);
        };
        tick();
    }
    function pause() {
        if (state.mode === "video") { stageVideo.pause(); return; }
        state.playing = false;
        btnPlay.textContent = "▶";
        if (state.timer) { clearTimeout(state.timer); state.timer = null; }
    }

    btnPrev.onclick  = () => { pause(); state.mode === "video" ? seekToFrac(state.inFrac == null ? 0 : state.inFrac) : selectClip(state.active - 1); root.focus(); };
    btnNext.onclick  = () => { pause(); state.mode === "video" ? seekToFrac(state.outFrac == null ? 1 : state.outFrac) : selectClip(state.active + 1); root.focus(); };
    btnFprev.onclick = () => { frameStep(-1); root.focus(); };
    btnFnext.onclick = () => { frameStep(+1); root.focus(); };
    btnPlay.onclick  = () => { state.playing ? pause() : play(); root.focus(); };
    btnLoop.onclick  = () => toggleLoop();
    btnSpeed.onclick = () => cycleSpeed();
    btnIn.onclick    = () => setIn();
    btnOut.onclick   = () => setOut();
    btnClr.onclick   = () => clearTrim();

    // Keyboard transport when the player has focus (does not leak to the canvas).
    root.addEventListener("keydown", (ev) => {
        let handled = true;
        switch (ev.key) {
            case " ": case "k": state.playing ? pause() : play(); break;
            case "ArrowLeft":  frameStep(-1); break;
            case "ArrowRight": frameStep(+1); break;
            case "i": case "I": setIn(); break;
            case "o": case "O": setOut(); break;
            case "l": case "L": toggleLoop(); break;
            default: handled = false;
        }
        if (handled) { ev.preventDefault(); ev.stopPropagation(); }
    });

    syncSpeed();
    updateScrub();

    return { root, rescan, showVideo, showImage, showMessage,
             setStatus: (s) => status.textContent = s };
}

// Guard against double-registration when CustomNodePacks ships the same extension.
if (!(app.extensions || []).some(e => e?.name === "C2C.WanDirector.Player")) app.registerExtension({
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
            // Clip to the widget slot — prevents the player DOM spilling
            // outside the node bounds (same fix as the timeline widget).
            // Retried: the Vue layer mounts the wrapper after onNodeCreated.
            const _wdClipPlayer = () => {
                try {
                    const wrap = player.root.closest?.(".dom-widget");
                    if (wrap && wrap !== player.root) { wrap.style.overflow = "hidden"; return true; }
                } catch (_) {}
                return false;
            };
            requestAnimationFrame(_wdClipPlayer);
            setTimeout(_wdClipPlayer, 500);
            setTimeout(_wdClipPlayer, 1500);
            // Liveness self-clean: when the node is removed WITHOUT onRemoved
            // firing (some graph.clear/workflow-load paths), the dead player UI
            // would linger and swallow clicks. Poll cheaply; remove ourselves.
            const _self = this;
            const _aliveTimer = setInterval(() => {
                if (_self.graph == null) {
                    try { player.root.remove(); } catch (_) {}
                    try { player.destroy?.(); } catch (_) {}
                    clearInterval(_aliveTimer);
                }
            }, 2000);
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
