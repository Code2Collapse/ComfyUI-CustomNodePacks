/**
 * Interactive player for the C2C video loaders (L2.50; owner 2026-10-14: "Load Video C2c has no player, frame
 * selector preview or any of the features i asked previously").
 *
 * - Transport: first / previous frame / play-pause / next frame / last, loop the selection, set in / set out.
 * - A scrubber over the WHOLE clip: click or drag to seek; the shaded range is the node's frame selection and its two
 *   handles ARE the selection: releasing one writes skip_first_frames / frame_load_cap (images: skip_first_images /
 *   image_load_cap) through the widgets' own callbacks, so the node, its saved values and the server probe agree.
 * - Frame numbers are exact source (or force_rate) frames. Long clips preview every k-th frame (at most 2000 in the
 *   preview); the label then says so.
 * - Keys while the player has focus: Space play/pause, Left/Right step, Home/End, I/O set in/out.
 * - Cost: a frame callback runs only while the video plays; nothing polls while paused (owner's UI-load rule).
 *   Works the same inside subgraphs and in both renderers: it only touches its own DOM and its own node's widgets.
 */
import { fmtNum } from "./c2c_video_player_ui.js";

const STYLE_ID = "c2c-vplayer-v1";

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.c2c-vplayer { display: flex; flex-direction: column; gap: 4px; outline: none; }
.c2c-vplayer:focus-visible { box-shadow: 0 0 0 1px var(--cu-accent); border-radius: 6px; }
.c2c-vplayer__bar { display: flex; align-items: center; gap: 2px; flex-wrap: wrap; }
.c2c-vplayer__btn { min-width: 24px; height: 22px; padding: 0 5px; border-radius: 4px; border: 1px solid var(--cu-edge-strong);
  background: var(--cu-raised); color: var(--cu-ink); font-size: 11px; line-height: 20px; cursor: pointer; }
.c2c-vplayer__btn:hover { border-color: var(--cu-accent); }
.c2c-vplayer__btn[aria-pressed="true"] { background: var(--cu-accent); color: var(--cu-ground); border-color: var(--cu-accent); }
.c2c-vplayer__btn:disabled { opacity: 0.4; cursor: default; }
.c2c-vplayer__frame { margin-left: auto; font-size: 10px; color: var(--cu-ink-soft); font-variant-numeric: tabular-nums;
  white-space: nowrap; }
.c2c-vplayer__track { position: relative; height: 18px; border-radius: 4px; background: var(--cu-edge); cursor: pointer;
  touch-action: none; }
.c2c-vplayer__sel { position: absolute; top: 3px; bottom: 3px; background: var(--cu-accent); opacity: 0.55; border-radius: 2px; }
.c2c-vplayer__handle { position: absolute; top: 0; bottom: 0; width: 8px; margin-left: -4px; border-radius: 3px;
  background: var(--cu-accent); cursor: ew-resize; box-shadow: 0 0 0 1px var(--cu-ground); }
.c2c-vplayer__head { position: absolute; top: -1px; bottom: -1px; width: 2px; margin-left: -1px; background: var(--cu-ink);
  pointer-events: none; }
.c2c-vplayer__labels { display: flex; justify-content: space-between; gap: 8px; font-size: 10px; color: var(--cu-ink-soft);
  font-variant-numeric: tabular-nums; }
`;
    document.head.appendChild(s);
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

export function timecode(frame, fps) {
    if (!(fps > 0)) return "";
    const f = Math.max(0, Math.round(frame));
    const whole = Math.max(1, Math.round(fps));
    const ff = f % whole;
    const total = Math.floor(f / whole);
    const s = total % 60, m = Math.floor(total / 60) % 60, h = Math.floor(total / 3600);
    const p = (n) => String(n).padStart(2, "0");
    return `${p(h)}:${p(m)}:${p(s)}:${p(ff)}`;
}

/** Selection from the node's widgets -> [in, out] in domain frames. ``count`` = frames that will load. */
export function selectionFromWidgets(skip, cap, nth, domainFrames) {
    const n = Math.max(1, domainFrames | 0);
    const start = Math.min(n - 1, Math.max(0, skip | 0));
    const every = Math.max(1, nth | 0);
    const available = Math.floor((n - 1 - start) / every) + 1;
    const count = (cap | 0) > 0 ? Math.min(cap | 0, available) : available;
    return { in: start, out: start + (count - 1) * every, count };
}

/** [in, out] -> widget values. Selecting the whole clip writes cap 0 ("all"). */
export function widgetsFromSelection(inF, outF, nth, domainFrames) {
    const n = Math.max(1, domainFrames | 0);
    const a = Math.max(0, Math.min(n - 1, Math.round(Math.min(inF, outF))));
    const b = Math.max(a, Math.min(n - 1, Math.round(Math.max(inF, outF))));
    const every = Math.max(1, nth | 0);
    const cap = (a === 0 && b === n - 1) ? 0 : Math.floor((b - a) / every) + 1;
    return { skip: a, cap };
}

/**
 * Build the player inside ``host`` (the loader's timeline container) and drive ``video``.
 * opts.onSelect(inFrame, outFrame) is called when a handle is released or I/O is pressed.
 */
export function attachPlayer(host, video, opts) {
    ensureStyles();
    const root = document.createElement("div");
    root.className = "c2c-vplayer";
    root.tabIndex = 0;
    root.title = "Click to focus: Space play / pause, Left / Right step, Home / End, I / O set in / out";

    const state = { frames: 1, fps: 24, step: 1, previewFrames: 1, previewFps: 24, in: 0, out: 0, count: 1, nth: 1,
                    loopSel: true, playing: false, frame: 0, ready: false, cb: 0 };

    const bar = document.createElement("div");
    bar.className = "c2c-vplayer__bar";
    const bFirst = btn("⏮", "First frame of the selection (Home)", () => seek(state.in));
    const bPrev = btn("◀", "Previous frame (Left)", () => seek(state.frame - state.step));
    const bPlay = btn("▶", "Play / pause (Space)", () => togglePlay());
    const bNext = btn("▶|", "Next frame (Right)", () => seek(state.frame + state.step));
    const bLast = btn("⏭", "Last frame of the selection (End)", () => seek(state.out));
    const bLoop = btn("⟲ sel", "Loop inside the selection (off = play the whole clip)", () => {
        state.loopSel = !state.loopSel;
        bLoop.setAttribute("aria-pressed", String(state.loopSel));
    });
    bLoop.setAttribute("aria-pressed", "true");
    const bIn = btn("[ in", "Set the first frame to load here (I)", () => setIn(state.frame));
    const bOut = btn("out ]", "Set the last frame to load here (O)", () => setOut(state.frame));
    const frameLabel = document.createElement("span");
    frameLabel.className = "c2c-vplayer__frame";
    bar.append(bFirst, bPrev, bPlay, bNext, bLast, bLoop, bIn, bOut, frameLabel);

    const track = document.createElement("div");
    track.className = "c2c-vplayer__track";
    track.title = "Drag to scrub; drag the handles to choose the frames to load";
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
    track.append(sel, hIn, hOut, head);

    const labels = document.createElement("div");
    labels.className = "c2c-vplayer__labels";
    const lSel = document.createElement("span");
    const lCount = document.createElement("span");
    labels.append(lSel, lCount);

    root.append(bar, track, labels);
    host.appendChild(root);

    const pct = (f) => (state.frames > 1 ? (f / (state.frames - 1)) * 100 : 0);
    const clamp = (f) => Math.max(0, Math.min(state.frames - 1, Math.round(f)));

    function renderSelection() {
        sel.style.left = `${pct(state.in)}%`;
        sel.style.width = `${Math.max(0.4, pct(state.out) - pct(state.in))}%`;
        hIn.style.left = `${pct(state.in)}%`;
        hOut.style.left = `${pct(state.out)}%`;
        lSel.textContent = `frames ${fmtNum(state.in)}–${fmtNum(state.out)} of ${fmtNum(state.frames)}`;
        const every = state.nth > 1 ? ` (every ${state.nth}${state.nth === 2 ? "nd" : state.nth === 3 ? "rd" : "th"})` : "";
        lCount.textContent = `${fmtNum(state.count)} will load${every}`;
    }

    function renderFrame() {
        head.style.left = `${pct(state.frame)}%`;
        const sub = state.step > 1 ? ` · preview every ${state.step} frames` : "";
        frameLabel.textContent = `frame ${fmtNum(state.frame)} / ${fmtNum(state.frames - 1)} · ${timecode(state.frame, state.fps)}${sub}`;
    }

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

    function onFrameCallback() {
        state.cb = 0;
        if (!state.playing || !video) return;
        const f = frameAt(video.currentTime);
        if (state.loopSel && (f > state.out || f < state.in)) {
            seek(state.in);
        } else {
            state.frame = f;
            renderFrame();
        }
        schedule();
    }

    function schedule() {
        if (!video || state.cb) return;
        if (typeof video.requestVideoFrameCallback === "function") {
            state.cb = video.requestVideoFrameCallback(onFrameCallback);
        } else {
            state.cb = requestAnimationFrame(onFrameCallback);
        }
    }

    function setPlaying(on) {
        state.playing = on;
        bPlay.textContent = on ? "⏸" : "▶";
        if (on) schedule();
    }

    async function togglePlay() {
        if (!video || !state.ready) return;
        if (video.paused) {
            if (state.loopSel && (state.frame < state.in || state.frame >= state.out)) seek(state.in);
            try { await video.play(); } catch (_e) { /* autoplay policy: the next click works */ }
        } else {
            video.pause();
        }
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
        if (drag === "seek") {
            if (video && !video.paused) video.pause();
            seek(frameFromEvent(e));
        }
    });
    track.addEventListener("pointermove", (e) => {
        if (!drag) return;
        e.stopPropagation();
        const f = frameFromEvent(e);
        if (drag === "seek") seek(f);
        else if (drag === "in") { state.in = Math.min(f, state.out); renderSelection(); seek(state.in); }
        else { state.out = Math.max(f, state.in); renderSelection(); seek(state.out); }
    });
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

    root.addEventListener("keydown", (e) => {
        const k = e.key;
        const map = {
            " ": () => togglePlay(), ArrowLeft: () => seek(state.frame - state.step),
            ArrowRight: () => seek(state.frame + state.step), Home: () => seek(state.in), End: () => seek(state.out),
            i: () => setIn(state.frame), I: () => setIn(state.frame), o: () => setOut(state.frame), O: () => setOut(state.frame),
        };
        if (!map[k]) return;
        e.preventDefault();
        e.stopPropagation();
        map[k]();
    });

    if (video) {
        video.loop = false;
        video.autoplay = false;
        video.addEventListener("play", () => setPlaying(true));
        video.addEventListener("pause", () => setPlaying(false));
        video.addEventListener("ended", () => {
            if (state.loopSel) { seek(state.in); video.play().catch(() => {}); } else setPlaying(false);
        });
        video.addEventListener("loadedmetadata", () => { state.ready = true; seek(state.frame); });
        video.addEventListener("emptied", () => { state.ready = false; setPlaying(false); });
        video.addEventListener("seeked", () => { if (!state.playing) { renderFrame(); } });
    }

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
            renderFrame();
        },
        /** The node's current selection (from its widgets and the server's exact count). */
        setSelection({ in: a, out: b, count, nth }) {
            if (drag === "in" || drag === "out") return;    // never fight the user's drag
            state.in = clamp(a);
            state.out = Math.max(state.in, clamp(b));
            state.count = Math.max(0, count | 0);
            state.nth = Math.max(1, nth | 0);
            if (!state.playing && (state.frame < state.in || state.frame > state.out)) state.frame = state.in;
            renderSelection();
            renderFrame();
        },
        get state() { return { ...state }; },
        destroy() {
            if (state.cb && video?.cancelVideoFrameCallback) video.cancelVideoFrameCallback(state.cb);
            else if (state.cb) cancelAnimationFrame(state.cb);
            root.remove();
        },
    };
}
