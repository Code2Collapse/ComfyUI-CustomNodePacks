/**
 * Interactive player for the C2C video loaders (L2.50, extended L2.53; owner 2026-10-14/15: "I can't pause preview,
 * forward backward preview or do whatever I want with the preview like WhatDreamsCost and other people's video node").
 *
 * - Transport: first / previous / play-pause / next / last, play backwards, speed 0.25x-4x, loop mode
 *   (loop the selection, loop the whole clip, play once, ping-pong), select all, set in / set out.
 * - A scrubber over the WHOLE clip with a ruler (frames or time): click or drag to seek; hovering shows that frame in
 *   a thumbnail without moving the playhead. The shaded range is the node's frame selection and its two handles ARE
 *   the selection: releasing one writes skip_first_frames / frame_load_cap (images: skip_first_images /
 *   image_load_cap) through the widgets' own callbacks, so the node, its saved values and the server probe agree.
 * - The frame box takes a frame number or a timecode (click it, type, Enter).
 * - Keys while the player has focus: Space play / pause, Left / Right step (Shift: 10), Home / End, I / O in / out,
 *   J / K / L shuttle (backwards / stop / forwards, press again for 2x, 4x), Up / Down speed, F fullscreen.
 * - Frame numbers are exact source (or force_rate) frames. Long clips preview every k-th frame (at most 2000 in the
 *   preview); the label then says so.
 * - Cost: a frame callback runs only while the video plays; nothing polls while paused (owner's UI-load rule). The
 *   hover thumbnail has its own small decoder, made on the first hover and released 8 s after the pointer leaves.
 *   Works the same inside subgraphs and in both renderers: it only touches its own DOM and its own node's widgets.
 */
import { fmtNum } from "./c2c_video_player_ui.js";
import {
    LOOP_LABELS, formatSeconds, loopStep, nextLoopMode, nextSpeed, parseFrameInput, rulerTicks, shuttle, timecode,
    selectionFromWidgets, widgetsFromSelection,
} from "./_c2c_player_core.js";

export { timecode, selectionFromWidgets, widgetsFromSelection };

const STYLE_ID = "c2c-vplayer-v2";
const UNIT_KEY = "c2c.vplayer.unit";
const THUMB_RELEASE_MS = 8000;

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    document.getElementById("c2c-vplayer-v1")?.remove();
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.c2c-vplayer { display: flex; flex-direction: column; gap: 3px; outline: none; position: relative; }
.c2c-vplayer:focus-visible { box-shadow: 0 0 0 1px var(--cu-accent); border-radius: 6px; }
.c2c-vplayer__bar { display: flex; align-items: center; gap: 2px; flex-wrap: wrap; }
.c2c-vplayer__group { display: flex; align-items: center; gap: 2px; }
.c2c-vplayer__sep { width: 1px; height: 16px; margin: 0 3px; background: var(--cu-edge-strong); }
.c2c-vplayer__btn { min-width: 24px; height: 22px; padding: 0 5px; border-radius: 4px; border: 1px solid var(--cu-edge-strong);
  background: var(--cu-raised); color: var(--cu-ink); font-size: 11px; line-height: 20px; cursor: pointer;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.c2c-vplayer__btn:hover { border-color: var(--cu-accent); }
.c2c-vplayer__btn:focus-visible { outline: 1px solid var(--cu-accent); outline-offset: 1px; }
.c2c-vplayer__btn[aria-pressed="true"] { background: var(--cu-accent); color: var(--cu-ground); border-color: var(--cu-accent); }
.c2c-vplayer__btn:disabled { opacity: 0.4; cursor: default; }
.c2c-vplayer__frame { margin-left: auto; font-size: 10px; color: var(--cu-ink-soft); font-variant-numeric: tabular-nums;
  white-space: nowrap; cursor: text; border-radius: 3px; padding: 1px 3px; }
.c2c-vplayer__frame:hover { background: var(--cu-raised); color: var(--cu-ink); }
.c2c-vplayer__frame-input { margin-left: auto; width: 96px; height: 18px; font-size: 11px; padding: 0 4px;
  border-radius: 3px; border: 1px solid var(--cu-accent); background: var(--cu-ground); color: var(--cu-ink);
  font-variant-numeric: tabular-nums; }
.c2c-vplayer__track { position: relative; height: 18px; border-radius: 4px; background: var(--cu-edge); cursor: pointer;
  touch-action: none; }
.c2c-vplayer__sel { position: absolute; top: 3px; bottom: 3px; background: var(--cu-accent); opacity: 0.55; border-radius: 2px; }
.c2c-vplayer__handle { position: absolute; top: 0; bottom: 0; width: 8px; margin-left: -4px; border-radius: 3px;
  background: var(--cu-accent); cursor: ew-resize; box-shadow: 0 0 0 1px var(--cu-ground); }
.c2c-vplayer__head { position: absolute; top: -1px; bottom: -1px; width: 2px; margin-left: -1px; background: var(--cu-ink);
  pointer-events: none; }
.c2c-vplayer__hover { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--cu-ink-soft); opacity: 0.7;
  pointer-events: none; display: none; }
.c2c-vplayer__ruler { position: relative; height: 13px; margin: 0 0 1px; pointer-events: none; }
.c2c-vplayer__tick { position: absolute; top: 0; width: 1px; background: var(--cu-edge-strong); }
.c2c-vplayer__tick--major { height: 5px; }
.c2c-vplayer__tick--minor { height: 3px; opacity: 0.7; }
.c2c-vplayer__tick-label { position: absolute; top: 4px; font-size: 9px; line-height: 9px; color: var(--cu-ink-soft);
  transform: translateX(-50%); white-space: nowrap; font-variant-numeric: tabular-nums; }
.c2c-vplayer__labels { display: flex; justify-content: space-between; gap: 8px; font-size: 10px; color: var(--cu-ink-soft);
  font-variant-numeric: tabular-nums; }
.c2c-vplayer__thumb { position: absolute; bottom: 100%; margin-bottom: 4px; width: 132px; padding: 3px; border-radius: 5px;
  background: var(--cu-ground); border: 1px solid var(--cu-edge-strong); box-shadow: 0 4px 12px rgba(0,0,0,0.35);
  pointer-events: none; display: none; z-index: 5; transform: translateX(-50%); }
.c2c-vplayer__thumb video { display: block; width: 100%; height: auto; border-radius: 3px; background: #000; }
.c2c-vplayer__thumb span { display: block; text-align: center; font-size: 10px; color: var(--cu-ink);
  font-variant-numeric: tabular-nums; margin-top: 2px; }
.c2c-vloader:fullscreen { background: var(--cu-ground, #111); padding: 16px; box-sizing: border-box; overflow: auto;
  display: flex; flex-direction: column; justify-content: center; }
.c2c-vloader:fullscreen .c2c-ui-stage__viewport {
  width: min(100%, calc((100vh - 190px) * var(--c2c-aspect, 1.7778))); margin: 0 auto; }
`;
    document.head.appendChild(s);
}

/** A button in the player's style (the crop tool uses it too). */
export function playerButton(text, title, onClick) {
    ensureStyles();
    return btn(text, title, onClick);
}

function btn(text, title, onClick) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "c2c-vplayer__btn";
    b.textContent = text;
    b.title = title;
    b.setAttribute("aria-label", title);
    b.addEventListener("click", (e) => { e.stopPropagation(); onClick(); });
    return b;
}

function group(...els) {
    const g = document.createElement("div");
    g.className = "c2c-vplayer__group";
    g.append(...els);
    return g;
}

function sep() {
    const s = document.createElement("span");
    s.className = "c2c-vplayer__sep";
    return s;
}

function readUnit() {
    try { return localStorage.getItem(UNIT_KEY) === "time" ? "time" : "frames"; } catch (_e) { return "frames"; }
}

function writeUnit(u) {
    try { localStorage.setItem(UNIT_KEY, u); } catch (_e) { /* private window: per-session only */ }
}

/**
 * Build the player inside ``host`` (the loader's timeline container) and drive ``video``.
 * opts.onSelect(inFrame, outFrame) is called when a handle is released or I/O / select-all is pressed.
 * opts.fullscreenTarget: the element shown full screen (the whole loader, so the controls come along).
 */
export function attachPlayer(host, video, opts = {}) {
    ensureStyles();
    const root = document.createElement("div");
    root.className = "c2c-vplayer";
    root.tabIndex = 0;
    root.title = "Click to focus. Space play / pause, Left / Right step (Shift: 10), J / K / L shuttle, "
        + "Up / Down speed, Home / End, I / O set in / out, F full screen";

    const state = {
        frames: 1, fps: 24, step: 1, previewFrames: 1, previewFps: 24, in: 0, out: 0, count: 1, nth: 1,
        loop: "selection", speed: 1, rate: 0, frame: 0, ready: false, cb: 0, revRaf: 0, revLast: 0, revAcc: 0,
        unit: readUnit(),
    };

    // ── transport bar ──
    const bar = document.createElement("div");
    bar.className = "c2c-vplayer__bar";
    const bFirst = btn("⏮", "First frame of the selection (Home)", () => { stop(); seek(state.in); });
    const bPrev = btn("◀|", "Previous frame (Left; Shift+Left: 10 frames)", () => { stop(); seek(state.frame - state.step); });
    const bBack = btn("◀", "Play backwards (J; again: faster)", () => playAt(-state.speed));
    const bPlay = btn("▶", "Play / pause (Space)", () => togglePlay());
    const bNext = btn("|▶", "Next frame (Right; Shift+Right: 10 frames)", () => { stop(); seek(state.frame + state.step); });
    const bLast = btn("⏭", "Last frame of the selection (End)", () => { stop(); seek(state.out); });
    const bSpeed = btn("1×", "Playback speed (Up / Down). Click: faster; Shift+click: slower", () => {});
    bSpeed.addEventListener("click", (e) => setSpeed(nextSpeed(state.speed, e.shiftKey ? -1 : 1)), { capture: true });
    bSpeed.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); setSpeed(nextSpeed(state.speed, -1)); });
    const bLoop = btn(LOOP_LABELS.selection, "Loop mode: loop the selection, loop the whole clip, play once, ping-pong",
        () => { state.loop = nextLoopMode(state.loop); renderButtons(); });
    const bAll = btn("all", "Select the whole clip", () => { state.in = 0; state.out = state.frames - 1; commit(); });
    const bIn = btn("[ in", "Set the first frame to load here (I)", () => setIn(state.frame));
    const bOut = btn("out ]", "Set the last frame to load here (O)", () => setOut(state.frame));
    const bUnit = btn("frames", "Show frames or time on the ruler and labels", () => {
        state.unit = state.unit === "time" ? "frames" : "time";
        writeUnit(state.unit);
        renderButtons(); renderRuler(); renderSelection(); renderFrame();
    });
    const bFull = btn("⛶", "Full screen (F)", () => toggleFullscreen());
    const frameLabel = document.createElement("span");
    frameLabel.className = "c2c-vplayer__frame";
    frameLabel.title = "Click to type a frame number or a timecode (hh:mm:ss:ff)";
    bar.append(group(bFirst, bPrev, bBack, bPlay, bNext, bLast), sep(), group(bSpeed, bLoop), sep(),
               group(bIn, bOut, bAll), sep(), group(bUnit, bFull), frameLabel);

    // ── scrubber ──
    const ruler = document.createElement("div");
    ruler.className = "c2c-vplayer__ruler";
    const track = document.createElement("div");
    track.className = "c2c-vplayer__track";
    track.title = "Drag to scrub; drag the handles to choose the frames to load; hover to preview a frame";
    const sel = document.createElement("div");
    sel.className = "c2c-vplayer__sel";
    const hIn = document.createElement("div");
    hIn.className = "c2c-vplayer__handle";
    hIn.title = "First frame to load (skip_first_frames)";
    const hOut = document.createElement("div");
    hOut.className = "c2c-vplayer__handle";
    hOut.title = "Last frame to load (sets frame_load_cap)";
    const head = document.createElement("div");
    head.className = "c2c-vplayer__head";
    const hoverLine = document.createElement("div");
    hoverLine.className = "c2c-vplayer__hover";
    track.append(sel, hIn, hOut, hoverLine, head);

    const thumb = document.createElement("div");
    thumb.className = "c2c-vplayer__thumb";
    const thumbLabel = document.createElement("span");
    thumb.append(thumbLabel);

    const labels = document.createElement("div");
    labels.className = "c2c-vplayer__labels";
    const lSel = document.createElement("span");
    const lCount = document.createElement("span");
    labels.append(lSel, lCount);

    root.append(bar, track, ruler, labels, thumb);
    host.appendChild(root);

    let editingFrame = false;
    const pct = (f) => (state.frames > 1 ? (f / (state.frames - 1)) * 100 : 0);
    const clamp = (f) => Math.max(0, Math.min(state.frames - 1, Math.round(f)));
    const fmtFrame = (f) => (state.unit === "time" ? timecode(f, state.fps) : fmtNum(f));

    function renderButtons() {
        bSpeed.textContent = `${state.speed}×`;
        bLoop.textContent = LOOP_LABELS[state.loop];
        bLoop.setAttribute("aria-pressed", String(state.loop !== "once"));
        bUnit.textContent = state.unit === "time" ? "time" : "frames";
        bPlay.textContent = state.rate > 0 ? "⏸" : "▶";
        bBack.textContent = state.rate < 0 ? "⏸" : "◀";
        bBack.setAttribute("aria-pressed", String(state.rate < 0));
    }

    function renderSelection() {
        sel.style.left = `${pct(state.in)}%`;
        sel.style.width = `${Math.max(0.4, pct(state.out) - pct(state.in))}%`;
        hIn.style.left = `${pct(state.in)}%`;
        hOut.style.left = `${pct(state.out)}%`;
        if (state.unit === "time") {
            const a = state.in / state.fps, b = state.out / state.fps;
            lSel.textContent = `${formatSeconds(a)} – ${formatSeconds(b)} of ${formatSeconds((state.frames - 1) / state.fps)}`;
        } else {
            lSel.textContent = `frames ${fmtNum(state.in)}–${fmtNum(state.out)} of ${fmtNum(state.frames)}`;
        }
        const every = state.nth > 1 ? ` (every ${state.nth}${state.nth === 2 ? "nd" : state.nth === 3 ? "rd" : "th"})` : "";
        const span = formatSeconds((state.out - state.in + 1) / state.fps);
        lCount.textContent = `${fmtNum(state.count)} will load${every} · ${span}`;
    }

    function renderFrame() {
        head.style.left = `${pct(state.frame)}%`;
        if (!editingFrame) {
            const sub = state.step > 1 ? ` · preview every ${state.step}` : "";
            frameLabel.textContent = state.unit === "time"
                ? `${timecode(state.frame, state.fps)} / ${timecode(state.frames - 1, state.fps)}${sub}`
                : `frame ${fmtNum(state.frame)} / ${fmtNum(state.frames - 1)} · ${timecode(state.frame, state.fps)}${sub}`;
        }
    }

    let rulerWidth = 0;
    function renderRuler() {
        const w = track.clientWidth;
        rulerWidth = w;
        ruler.replaceChildren();
        if (!w || state.frames < 2) return;
        const frag = document.createDocumentFragment();
        let lastLabelRight = -Infinity;
        for (const t of rulerTicks(state.frames, state.fps, w, state.unit)) {
            const x = pct(t.frame);
            const tick = document.createElement("div");
            tick.className = `c2c-vplayer__tick c2c-vplayer__tick--${t.major ? "major" : "minor"}`;
            tick.style.left = `${x}%`;
            frag.appendChild(tick);
            if (t.major && t.label) {
                const px = (x / 100) * w;
                if (px - lastLabelRight < 8 || px > w - 12) continue;     // no overlapping or clipped labels
                const lab = document.createElement("span");
                lab.className = "c2c-vplayer__tick-label";
                lab.style.left = `${x}%`;
                lab.textContent = t.label;
                frag.appendChild(lab);
                lastLabelRight = px + t.label.length * 3.2;
            }
        }
        ruler.appendChild(frag);
    }
    const ro = typeof ResizeObserver === "function"
        ? new ResizeObserver(() => { if (Math.abs(track.clientWidth - rulerWidth) > 2) renderRuler(); })
        : null;
    ro?.observe(track);

    // ── video <-> frame mapping (the preview holds domain frames 0, k, 2k, ...) ──
    const frameAt = (t) => clamp(Math.floor(t * state.previewFps + 1e-3) * state.step);
    const timeOf = (f) => (Math.round(clamp(f) / state.step) + 0.5) / state.previewFps;

    function seek(f) {
        state.frame = clamp(f);
        renderFrame();
        if (video && state.ready) {
            try { video.currentTime = Math.max(0, timeOf(state.frame)); } catch (_e) { /* not seekable yet */ }
        }
    }

    // ── forward playback: the browser plays; each presented frame is checked against the loop bounds ──
    function onFrameCallback() {
        state.cb = 0;
        if (!(state.rate > 0) || !video) return;
        const f = frameAt(video.currentTime);
        const r = loopStep({ frame: f, dir: 1, mode: state.loop, inF: state.in, outF: state.out, frames: state.frames });
        if (r.stop) { stop(); seek(r.frame); return; }
        if (r.dir < 0) { playAt(-Math.abs(state.rate), r.frame); return; }    // ping-pong turns around
        if (r.frame !== f) { seek(r.frame); } else { state.frame = f; renderFrame(); }
        schedule();
    }

    function schedule() {
        if (!video || state.cb) return;
        state.cb = typeof video.requestVideoFrameCallback === "function"
            ? video.requestVideoFrameCallback(onFrameCallback)
            : requestAnimationFrame(onFrameCallback);
    }

    function cancelSchedule() {
        if (!state.cb) return;
        if (video?.cancelVideoFrameCallback && typeof video.requestVideoFrameCallback === "function") {
            video.cancelVideoFrameCallback(state.cb);
        } else {
            cancelAnimationFrame(state.cb);
        }
        state.cb = 0;
    }

    // ── backward playback: browsers cannot play backwards, so it steps by seeking, one seek in flight at a time ──
    let seekBusy = false;
    let seekSince = 0;
    function reverseTick(now) {
        state.revRaf = 0;
        if (!(state.rate < 0)) return;
        const dt = state.revLast ? Math.min(0.25, (now - state.revLast) / 1000) : 0;
        state.revLast = now;
        state.revAcc += dt * state.fps * Math.abs(state.rate);
        if (seekBusy && now - seekSince > 500) seekBusy = false;    // a seek that never reported back
        const stepFrames = Math.floor(state.revAcc / state.step) * state.step;
        if (stepFrames >= state.step && !seekBusy) {
            state.revAcc -= stepFrames;
            const target = state.frame - stepFrames;
            const r = loopStep({ frame: target, dir: -1, mode: state.loop, inF: state.in, outF: state.out, frames: state.frames });
            if (r.stop) { stop(); seek(r.frame); return; }
            if (r.dir > 0) { seek(r.frame); playAt(Math.abs(state.rate)); return; }   // ping-pong turns around
            seekBusy = true;
            seekSince = now;
            seek(r.frame);
        }
        state.revRaf = requestAnimationFrame(reverseTick);
    }

    /** Play at a signed rate: > 0 forwards (the browser plays), < 0 backwards (stepped seeks), 0 stops. */
    async function playAt(rate, fromFrame = null) {
        if (!video || !state.ready) return;
        if (rate === state.rate && fromFrame == null) { stop(); return; }   // the same button again pauses
        cancelSchedule();
        if (state.revRaf) { cancelAnimationFrame(state.revRaf); state.revRaf = 0; }
        state.rate = rate;
        if (fromFrame != null) seek(fromFrame);
        if (rate > 0) {
            const r = loopStep({ frame: state.frame, dir: 1, mode: state.loop, inF: state.in, outF: state.out, frames: state.frames });
            if (state.loop !== "clip" && state.loop !== "once" && (state.frame >= state.out || state.frame < state.in)) seek(state.in);
            else if (r.frame !== state.frame) seek(r.frame);
            video.playbackRate = Math.abs(rate);
            try { await video.play(); } catch (_e) { /* autoplay policy: the next click works */ }
            schedule();
        } else if (rate < 0) {
            if (!video.paused) video.pause();
            if ((state.loop === "selection" || state.loop === "pingpong") && (state.frame <= state.in || state.frame > state.out)) {
                seek(state.out);
            }
            state.revLast = 0;
            state.revAcc = 0;
            state.revRaf = requestAnimationFrame(reverseTick);
        } else if (!video.paused) {
            video.pause();
        }
        renderButtons();
    }

    function stop() {
        cancelSchedule();
        if (state.revRaf) { cancelAnimationFrame(state.revRaf); state.revRaf = 0; }
        const wasPlaying = state.rate !== 0;
        state.rate = 0;
        if (video && !video.paused) video.pause();
        if (wasPlaying) renderButtons();
    }

    function togglePlay() {
        if (state.rate !== 0) stop();
        else playAt(state.speed);
    }

    function setSpeed(s) {
        state.speed = s;
        if (state.rate !== 0) {
            state.rate = Math.sign(state.rate) * s;
            if (state.rate > 0 && video) video.playbackRate = s;
        }
        renderButtons();
    }

    function setIn(f) {
        state.in = clamp(f);
        if (state.out < state.in) state.out = state.in;
        commit();
    }

    function setOut(f) {
        state.out = clamp(f);
        if (state.in > state.out) state.in = state.out;
        commit();
    }

    function commit() {
        renderSelection();
        try { opts.onSelect?.(state.in, state.out); } catch (_e) { /* the widget write reports its own errors */ }
    }

    function toggleFullscreen() {
        const target = opts.fullscreenTarget || host;
        try {
            if (document.fullscreenElement) document.exitFullscreen();
            else target.requestFullscreen?.().then(() => root.focus({ preventScroll: true })).catch(() => {});
        } catch (_e) { /* not allowed here */ }
    }
    const onFsChange = () => requestAnimationFrame(renderRuler);
    document.addEventListener("fullscreenchange", onFsChange);

    // ── frame box: type a frame or a timecode ──
    frameLabel.addEventListener("click", (e) => {
        e.stopPropagation();
        stop();
        const input = document.createElement("input");
        input.className = "c2c-vplayer__frame-input";
        input.value = String(state.frame);
        input.setAttribute("aria-label", "Go to frame or timecode");
        let finished = false;
        const finish = (go) => {
            if (finished) return;                   // removing the input blurs it: the blur must not finish twice
            finished = true;
            if (go) {
                const f = parseFrameInput(input.value, state.fps);
                if (f != null) seek(f);
            }
            editingFrame = false;
            input.replaceWith(frameLabel);
            renderFrame();
            root.focus({ preventScroll: true });
        };
        input.addEventListener("keydown", (ev) => {
            ev.stopPropagation();
            if (ev.key === "Enter") finish(true);
            else if (ev.key === "Escape") finish(false);
        });
        input.addEventListener("blur", () => finish(true));
        editingFrame = true;
        frameLabel.replaceWith(input);
        input.focus();
        input.select();
    });

    // ── hover thumbnail: a second, small decoder on the same preview ──
    let thumbVideo = null;
    let thumbBusy = false;
    let thumbPending = null;
    let thumbRelease = 0;
    const thumbTimeOf = (f) => timeOf(f);
    function thumbSeek(f) {
        if (!video?.currentSrc) return;
        if (!thumbVideo) {
            thumbVideo = document.createElement("video");
            thumbVideo.muted = true;
            thumbVideo.playsInline = true;
            thumbVideo.preload = "auto";
            thumbVideo.addEventListener("seeked", () => {
                thumbBusy = false;
                if (thumbPending != null) { const p = thumbPending; thumbPending = null; thumbSeek(p); }
            });
            thumbVideo.addEventListener("error", () => { thumbBusy = false; });
            thumb.prepend(thumbVideo);
        }
        if (thumbVideo.getAttribute("src") !== video.currentSrc) {
            thumbVideo.src = video.currentSrc;
            thumbBusy = false;
        }
        if (thumbBusy) { thumbPending = f; return; }
        thumbBusy = true;
        try { thumbVideo.currentTime = Math.max(0, thumbTimeOf(f)); } catch (_e) { thumbBusy = false; }
    }
    function releaseThumb() {
        if (!thumbVideo) return;
        thumbVideo.removeAttribute("src");
        try { thumbVideo.load(); } catch (_e) { /* ignore */ }
        thumbVideo.remove();
        thumbVideo = null;
        thumbBusy = false;
        thumbPending = null;
    }
    function showHover(e) {
        if (!state.ready || state.frames < 2) return;
        const r = track.getBoundingClientRect();
        const x = Math.max(0, Math.min(1, (e.clientX - r.left) / Math.max(1, r.width)));
        const f = clamp(Math.round((x * (state.frames - 1)) / state.step) * state.step);
        hoverLine.style.display = "block";
        hoverLine.style.left = `${pct(f)}%`;
        const rootR = root.getBoundingClientRect();
        const half = 66;
        const left = Math.max(half, Math.min(rootR.width - half, (r.left - rootR.left) + x * r.width));
        thumb.style.left = `${left}px`;
        thumb.style.display = "block";
        thumbLabel.textContent = state.unit === "time" ? timecode(f, state.fps) : `frame ${fmtNum(f)}`;
        if (thumbRelease) { clearTimeout(thumbRelease); thumbRelease = 0; }
        thumbSeek(f);
    }
    function hideHover() {
        hoverLine.style.display = "none";
        thumb.style.display = "none";
        if (!thumbRelease) thumbRelease = setTimeout(() => { thumbRelease = 0; releaseThumb(); }, THUMB_RELEASE_MS);
    }

    // ── pointer: scrub the track, drag the handles ──
    let drag = null;
    const frameFromEvent = (e) => {
        const r = track.getBoundingClientRect();
        const x = Math.max(0, Math.min(1, (e.clientX - r.left) / Math.max(1, r.width)));
        return clamp(x * (state.frames - 1));
    };
    track.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        e.preventDefault();
        root.focus({ preventScroll: true });
        drag = e.target === hIn ? "in" : e.target === hOut ? "out" : "seek";
        try { track.setPointerCapture(e.pointerId); } catch (_e) { /* ignore */ }
        hideHover();
        stop();
        if (drag === "seek") seek(frameFromEvent(e));
    });
    track.addEventListener("pointermove", (e) => {
        if (!drag) { if (e.pointerType !== "touch") showHover(e); return; }
        e.stopPropagation();
        const f = frameFromEvent(e);
        if (drag === "seek") seek(f);
        else if (drag === "in") { state.in = Math.min(f, state.out); renderSelection(); seek(state.in); }
        else { state.out = Math.max(f, state.in); renderSelection(); seek(state.out); }
    });
    track.addEventListener("pointerleave", () => { if (!drag) hideHover(); });
    const endDrag = (e) => {
        if (!drag) return;
        e.stopPropagation();
        const was = drag;
        drag = null;
        try { track.releasePointerCapture(e.pointerId); } catch (_e) { /* ignore */ }
        if (was !== "seek") commit();
    };
    track.addEventListener("pointerup", endDrag);
    track.addEventListener("pointercancel", endDrag);

    // ── keys ──
    root.addEventListener("keydown", (e) => {
        const k = e.key;
        const big = e.shiftKey ? 10 : 1;
        const map = {
            " ": () => togglePlay(),
            ArrowLeft: () => { stop(); seek(state.frame - state.step * big); },
            ArrowRight: () => { stop(); seek(state.frame + state.step * big); },
            ArrowUp: () => setSpeed(nextSpeed(state.speed, 1)),
            ArrowDown: () => setSpeed(nextSpeed(state.speed, -1)),
            Home: () => { stop(); seek(state.in); }, End: () => { stop(); seek(state.out); },
            i: () => setIn(state.frame), o: () => setOut(state.frame),
            j: () => shuttleKey("j"), k: () => shuttleKey("k"), l: () => shuttleKey("l"),
            f: () => toggleFullscreen(),
        };
        const fn = map[k] || map[k.toLowerCase?.()];
        if (!fn || e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();
        e.stopPropagation();
        fn();
    });
    function shuttleKey(key) {
        const r = shuttle(state.rate, key);
        if (r === 0) { stop(); return; }
        state.speed = Math.abs(r);
        if (r === state.rate) { renderButtons(); return; }     // already at 4x: keep playing
        playAt(r);
    }

    // ── the video element ──
    const onPlay = () => { if (!(state.rate > 0)) { state.rate = state.speed; video.playbackRate = state.speed; schedule(); renderButtons(); } };
    const onPause = () => { if (state.rate > 0) { cancelSchedule(); state.rate = 0; renderButtons(); } };
    const onEnded = () => {
        if (!(state.rate > 0)) return;
        const r = loopStep({ frame: state.frames, dir: 1, mode: state.loop, inF: state.in, outF: state.out, frames: state.frames });
        if (r.stop) { stop(); seek(r.frame); return; }
        if (r.dir < 0) { playAt(-Math.abs(state.rate), r.frame); return; }
        seek(r.frame);
        video.play().catch(() => {});
    };
    const onMeta = () => { state.ready = true; seek(state.frame); renderRuler(); };
    const onEmptied = () => { state.ready = false; stop(); releaseThumb(); };
    const onSeeked = () => { seekBusy = false; if (state.rate <= 0) renderFrame(); };
    if (video) {
        video.loop = false;
        video.autoplay = false;
        video.addEventListener("play", onPlay);
        video.addEventListener("pause", onPause);
        video.addEventListener("ended", onEnded);
        video.addEventListener("loadedmetadata", onMeta);
        video.addEventListener("emptied", onEmptied);
        video.addEventListener("seeked", onSeeked);
    }

    renderButtons();
    renderSelection();
    renderFrame();

    return {
        root,
        /** Whole-clip domain (frames after force_rate) and how the preview samples it. */
        setDomain({ frames, fps, step = 1, previewFrames, previewFps }) {
            state.frames = Math.max(1, frames | 0);
            state.fps = fps > 0 ? fps : 24;
            state.step = Math.max(1, step | 0);
            state.previewFrames = Math.max(1, previewFrames | 0);
            state.previewFps = previewFps > 0 ? previewFps : state.fps / state.step;
            state.frame = clamp(state.frame);
            renderRuler();
            renderSelection();
            renderFrame();
        },
        /** The node's current selection (from its widgets and the server's exact count). */
        setSelection({ in: a, out: b, count, nth }) {
            if (drag === "in" || drag === "out") return;    // never fight the user's drag
            state.in = clamp(a);
            state.out = Math.max(state.in, clamp(b));
            state.count = Math.max(0, count | 0);
            state.nth = Math.max(1, nth | 0);
            if (state.rate === 0 && (state.frame < state.in || state.frame > state.out)) state.frame = state.in;
            renderSelection();
            renderFrame();
        },
        pause() { stop(); },
        get state() { return { ...state }; },
        destroy() {
            stop();
            releaseThumb();
            if (thumbRelease) clearTimeout(thumbRelease);
            ro?.disconnect();
            document.removeEventListener("fullscreenchange", onFsChange);
            if (video) {
                video.removeEventListener("play", onPlay);
                video.removeEventListener("pause", onPause);
                video.removeEventListener("ended", onEnded);
                video.removeEventListener("loadedmetadata", onMeta);
                video.removeEventListener("emptied", onEmptied);
                video.removeEventListener("seeked", onSeeked);
            }
            root.remove();
        },
    };
}
