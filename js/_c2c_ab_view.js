/**
 * A/B view over a video viewport (L9.40, docs/save-video-v2.md): Save Video (C2C) compares the saved file with the
 * frames that came in; Video Comparer (C2C) builds on the same view.
 *
 * - B follows A frame by frame (A is the clock: the player drives it), with an offset in frames. A frame of A that
 *   has no B frame (outside the written range) shows A only and says so.
 * - Modes: A, B, wipe (vertical / horizontal / diagonal, drag the divider), difference (|A - B|, CSS
 *   mix-blend-mode, gain = brightness of the composite), onion (B at 50 %), flicker (A and B in turn).
 * - Zoom (wheel, around the pointer), pan (drag while zoomed, or middle button), double-click = fit, 1:1 = one video
 *   pixel per screen pixel. Loupe: 8x magnifier under the pointer with the pixel's RGB (0-1 and 8-bit).
 * - Cost: nothing runs while A is paused except one seek of B per A seek; while A plays, one B seek per presented
 *   frame (requestVideoFrameCallback), never two in flight. No polling.
 */

const STYLE_ID = "c2c-abview-v1";
export const AB_MODES = ["A", "B", "wipe", "difference", "onion", "flicker"];
export const WIPE_SHAPES = ["vertical", "horizontal", "diagonal"];

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.c2c-ab { position: absolute; inset: 0; overflow: hidden; }
.c2c-ab [hidden] { display: none !important; }
.c2c-ab__stage { position: absolute; inset: 0; transform-origin: 0 0; }
.c2c-ab__stage > video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
.c2c-ab__b { pointer-events: none; }
.c2c-ab__divider { position: absolute; z-index: 3; background: var(--cu-accent, #6cf); opacity: 0.9; cursor: ew-resize; }
.c2c-ab__divider::after { content: ""; position: absolute; left: 50%; top: 50%; width: 14px; height: 14px;
  margin: -7px 0 0 -7px; border-radius: 50%; background: var(--cu-accent, #6cf); box-shadow: 0 0 0 2px var(--cu-ground, #000); }
.c2c-ab__badge { position: absolute; z-index: 4; left: 6px; top: 6px; font-size: 10px; padding: 1px 6px; border-radius: 4px;
  background: var(--cu-raised, #222); color: var(--cu-ink-soft, #aaa); border: 1px solid var(--cu-edge-strong, #444);
  pointer-events: none; }
.c2c-ab__label { position: absolute; z-index: 4; bottom: 6px; font-size: 10px; padding: 1px 5px; border-radius: 3px;
  background: var(--cu-raised, #222); color: var(--cu-ink, #eee); pointer-events: none; opacity: 0.85; }
.c2c-ab__loupe { position: absolute; z-index: 5; width: 168px; height: 188px; pointer-events: none; border-radius: 6px;
  border: 1px solid var(--cu-edge-strong, #444); background: var(--cu-ground, #000); overflow: hidden; }
.c2c-ab__loupe canvas { display: block; width: 168px; height: 168px; image-rendering: pixelated; }
.c2c-ab__loupe span { display: block; font-size: 10px; line-height: 20px; text-align: center; color: var(--cu-ink, #eee);
  font-variant-numeric: tabular-nums; }
`;
    document.head.appendChild(s);
}

/** The rectangle (client px) a contained video's picture occupies inside its element. */
function contentRect(video) {
    const r = video.getBoundingClientRect();
    const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
    const s = Math.min(r.width / vw, r.height / vh);
    const w = vw * s, h = vh * s;
    return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, width: w, height: h, scale: s };
}

/**
 * Mount on ``viewport`` (position: relative) around the existing ``a`` video. Returns the controller.
 * opts.fps: A's frame rate; opts.labels: [labelA, labelB].
 */
export function attachABView(viewport, a, opts = {}) {
    ensureStyles();
    const wrap = document.createElement("div");
    wrap.className = "c2c-ab";
    const stage = document.createElement("div");
    stage.className = "c2c-ab__stage";
    const b = document.createElement("video");
    b.className = "c2c-ab__b";
    b.muted = true;
    b.playsInline = true;
    b.preload = "auto";
    b.hidden = true;
    const divider = document.createElement("div");
    divider.className = "c2c-ab__divider";
    divider.hidden = true;
    const badge = document.createElement("div");
    badge.className = "c2c-ab__badge";
    badge.hidden = true;
    const labA = document.createElement("div");
    labA.className = "c2c-ab__label";
    labA.style.left = "6px";
    const labB = document.createElement("div");
    labB.className = "c2c-ab__label";
    labB.style.right = "6px";
    const loupe = document.createElement("div");
    loupe.className = "c2c-ab__loupe";
    loupe.hidden = true;
    const lc = document.createElement("canvas");
    lc.width = 21;
    lc.height = 21;
    const lText = document.createElement("span");
    loupe.append(lc, lText);

    a.parentElement?.insertBefore(wrap, a);
    stage.append(a, b);
    wrap.append(stage, divider, badge, labA, labB, loupe);
    if (getComputedStyle(viewport).position === "static") viewport.style.position = "relative";

    const st = {
        mode: "A", shape: "vertical", pos: 0.5, gain: 1, offset: 0, fpsA: opts.fps || 24, fpsB: opts.fps || 24,
        framesB: 0, hasB: false, zoom: 1, tx: 0, ty: 0, flickerOn: false, flickerT: 0, loupe: false,
        seekBusy: false, wantB: -1, vfc: 0,
    };
    labA.textContent = (opts.labels || ["A"])[0];
    labB.textContent = (opts.labels || ["A", "B"])[1];

    // ---- B follows A ----------------------------------------------------------------------------------------
    const frameA = () => Math.max(0, Math.floor((a.currentTime || 0) * st.fpsA + 1e-3));
    function syncB() {
        if (!st.hasB || st.mode === "A") return;
        const f = frameA() - st.offset;
        // frames 0 = not told: B's own length once its metadata is in, else assume the frame exists
        const nB = st.framesB || (Number.isFinite(b.duration) ? Math.floor(b.duration * st.fpsB + 1e-3) : Infinity);
        const inside = f >= 0 && f < nB;
        b.style.visibility = inside ? "visible" : "hidden";
        badge.hidden = inside;
        if (!inside) {
            badge.textContent = "no B frame here (outside the written range)";
            return;
        }
        if (f === st.wantB && !st.seekBusy) return;
        st.wantB = f;
        if (st.seekBusy) return;
        st.seekBusy = true;
        b.currentTime = (f + 0.5) / st.fpsB;
    }
    b.addEventListener("seeked", () => {
        st.seekBusy = false;
        const f = frameA() - st.offset;
        if (f !== Math.floor(b.currentTime * st.fpsB + 1e-3) && f >= 0) syncB();
    });
    const onAFrame = () => {
        syncB();
        if (st.mode === "flicker") flickerTick();
        if (!a.paused && typeof a.requestVideoFrameCallback === "function") st.vfc = a.requestVideoFrameCallback(onAFrame);
    };
    const onPlay = () => { if (typeof a.requestVideoFrameCallback === "function") st.vfc = a.requestVideoFrameCallback(onAFrame); };
    a.addEventListener("play", onPlay);
    a.addEventListener("seeked", syncB);
    a.addEventListener("loadeddata", syncB);

    // ---- modes ----------------------------------------------------------------------------------------------
    function clipFor() {
        const p = st.pos * 100;
        if (st.shape === "horizontal") return `inset(${p}% 0 0 0)`;
        if (st.shape === "diagonal") return `polygon(${p * 2}% 0, 100% 0, 100% 100%, ${p * 2 - 100}% 100%)`;
        return `inset(0 0 0 ${p}%)`;
    }
    function placeDivider() {
        divider.hidden = st.mode !== "wipe";
        if (divider.hidden) return;
        divider.style.transform = "";
        if (st.shape === "horizontal") {
            Object.assign(divider.style, { left: "0", right: "0", width: "", height: "2px", top: `${st.pos * 100}%`,
                                           bottom: "", cursor: "ns-resize" });
        } else {
            // diagonal: the clip line runs from (2p, 0) to (2p - 1, 1) in unit coordinates, so at mid-height it is
            // at x = 2p - 0.5, with a slant of atan(width / height)
            const x = st.shape === "diagonal" ? 2 * st.pos - 0.5 : st.pos;
            Object.assign(divider.style, { top: "0", bottom: "0", height: "", width: "2px", left: `${x * 100}%`,
                                           right: "", cursor: "ew-resize" });
            if (st.shape === "diagonal") {
                const r = wrap.getBoundingClientRect();
                divider.style.transform = `skewX(${-Math.atan2(r.width || 16, r.height || 9)}rad)`;
            }
        }
    }
    function flickerTick() {
        const now = performance.now();
        if (now - st.flickerT > 500) { st.flickerOn = !st.flickerOn; st.flickerT = now; }
        b.style.opacity = st.flickerOn ? "1" : "0";
    }
    function render() {
        const showB = st.hasB && st.mode !== "A";
        b.hidden = !showB;
        a.style.opacity = st.mode === "B" ? "0" : "1";
        b.style.clipPath = st.mode === "wipe" ? clipFor() : "none";
        b.style.mixBlendMode = st.mode === "difference" ? "difference" : "normal";
        b.style.opacity = st.mode === "onion" ? "0.5" : "1";
        stage.style.filter = st.mode === "difference" && st.gain !== 1 ? `brightness(${st.gain})` : "none";
        labA.hidden = !showB || st.mode === "B" || st.mode === "difference";
        labB.hidden = !showB || st.mode === "difference";
        if (st.mode === "difference") badge.textContent = `|A - B| x ${st.gain}`;
        badge.hidden = st.mode !== "difference";
        placeDivider();
        stage.style.transform = `translate(${st.tx}px, ${st.ty}px) scale(${st.zoom})`;
        if (showB) syncB();
    }

    // ---- interaction: wipe divider, zoom, pan, loupe ----------------------------------------------------------
    let drag = null;
    divider.addEventListener("pointerdown", (e) => {
        e.preventDefault(); e.stopPropagation();
        drag = "wipe";
        divider.setPointerCapture(e.pointerId);
    });
    divider.addEventListener("pointermove", (e) => {
        if (drag !== "wipe") return;
        const r = wrap.getBoundingClientRect();
        const ux = (e.clientX - r.left) / r.width, uy = (e.clientY - r.top) / r.height;
        st.pos = st.shape === "horizontal" ? uy : st.shape === "diagonal" ? (ux + uy) / 2 : ux;
        st.pos = Math.max(0, Math.min(1, st.pos));
        render();
    });
    divider.addEventListener("pointerup", () => { drag = null; });
    wrap.addEventListener("wheel", (e) => {
        e.preventDefault(); e.stopPropagation();
        const r = wrap.getBoundingClientRect();
        const px = e.clientX - r.left, py = e.clientY - r.top;
        const z = Math.max(1, Math.min(64, st.zoom * Math.exp(-e.deltaY * 0.0015)));
        st.tx = px - (px - st.tx) * (z / st.zoom);
        st.ty = py - (py - st.ty) * (z / st.zoom);
        st.zoom = z;
        if (z === 1) { st.tx = 0; st.ty = 0; }
        render();
    }, { passive: false });
    let panFrom = null;
    wrap.addEventListener("pointerdown", (e) => {
        if (e.target === divider) return;
        if (e.button === 1 || (e.button === 0 && st.zoom > 1)) {
            e.preventDefault(); e.stopPropagation();
            panFrom = { x: e.clientX - st.tx, y: e.clientY - st.ty };
            wrap.setPointerCapture(e.pointerId);
        }
    });
    wrap.addEventListener("pointermove", (e) => {
        if (panFrom) {
            st.tx = e.clientX - panFrom.x;
            st.ty = e.clientY - panFrom.y;
            render();
        }
        if (st.loupe) drawLoupe(e);
    });
    wrap.addEventListener("pointerup", () => { panFrom = null; });
    wrap.addEventListener("pointerleave", () => { loupe.hidden = true; });
    wrap.addEventListener("dblclick", (e) => { e.stopPropagation(); fit(); });

    const probe = document.createElement("canvas");
    probe.width = 21;
    probe.height = 21;
    function drawLoupe(e) {
        const src = (st.hasB && st.mode !== "A" && !b.hidden && b.style.visibility !== "hidden"
                     && (st.mode === "B" || (st.mode === "wipe" && pointerOnB(e)))) ? b : a;
        if (!src.videoWidth) return;
        const cr = contentRect(src);
        const ux = (e.clientX - cr.left) / cr.width, uy = (e.clientY - cr.top) / cr.height;
        if (ux < 0 || uy < 0 || ux > 1 || uy > 1) { loupe.hidden = true; return; }
        const x = Math.floor(ux * src.videoWidth), y = Math.floor(uy * src.videoHeight);
        const ctx = probe.getContext("2d", { willReadFrequently: true });
        ctx.imageSmoothingEnabled = false;
        ctx.clearRect(0, 0, 21, 21);
        try {
            ctx.drawImage(src, x - 10, y - 10, 21, 21, 0, 0, 21, 21);
            const px = ctx.getImageData(10, 10, 1, 1).data;
            lc.getContext("2d").drawImage(probe, 0, 0);
            const c = lc.getContext("2d");
            c.strokeStyle = "rgba(255,255,255,0.8)";
            c.lineWidth = 0.1;
            c.strokeRect(10, 10, 1, 1);
            lText.textContent = `${src === b ? "B" : "A"} x${x} y${y}  ${px[0]} ${px[1]} ${px[2]}  `
                + `(${(px[0] / 255).toFixed(3)} ${(px[1] / 255).toFixed(3)} ${(px[2] / 255).toFixed(3)})`;
        } catch (_e) {
            lText.textContent = "pixel values unavailable";
        }
        const r = wrap.getBoundingClientRect();
        const lx = Math.min(r.width - 172, e.clientX - r.left + 16), ly = Math.min(r.height - 192, e.clientY - r.top + 16);
        loupe.style.left = `${Math.max(4, lx)}px`;
        loupe.style.top = `${Math.max(4, ly)}px`;
        loupe.hidden = false;
    }
    function pointerOnB(e) {
        const r = wrap.getBoundingClientRect();
        if (st.shape === "horizontal") return (e.clientY - r.top) / r.height > st.pos;
        if (st.shape === "diagonal") return (e.clientX - r.left) / r.width + (e.clientY - r.top) / r.height > 2 * st.pos;
        return (e.clientX - r.left) / r.width > st.pos;
    }
    function fit() { st.zoom = 1; st.tx = 0; st.ty = 0; render(); }
    function oneToOne() {
        const cr = contentRect(a);
        if (!a.videoWidth || !cr.scale) return;
        const z = Math.max(1, 1 / (cr.scale / st.zoom));
        const r = wrap.getBoundingClientRect();
        st.tx = r.width / 2 - (r.width / 2) * z;
        st.ty = r.height / 2 - (r.height / 2) * z;
        st.zoom = z;
        render();
    }

    render();
    return {
        /** B source: url (or "" to clear), its frame rate, frame count, and the A frame where B's frame 0 sits. */
        setB(url, { fps = st.fpsA, frames = 0, offset = 0 } = {}) {
            st.hasB = !!url;
            st.fpsB = fps > 0 ? fps : st.fpsA;
            st.framesB = Math.max(0, frames | 0);
            st.offset = offset | 0;
            st.wantB = -1;
            if (url) { if (b.src !== url) b.src = url; } else { b.removeAttribute("src"); try { b.load(); } catch (_e) { /* */ } }
            render();
        },
        setFpsA(fps) { st.fpsA = fps > 0 ? fps : 24; },
        setMode(m) { if (AB_MODES.includes(m)) { st.mode = m; render(); } },
        setShape(s) { if (WIPE_SHAPES.includes(s)) { st.shape = s; render(); } },
        setGain(g) { st.gain = Math.max(1, Math.min(64, Number(g) || 1)); render(); },
        setLoupe(on) { st.loupe = !!on; if (!on) loupe.hidden = true; },
        setLabels(la, lb) { labA.textContent = la; labB.textContent = lb; },
        fit,
        oneToOne,
        get state() { return { ...st }; },
        destroy() {
            a.removeEventListener("play", onPlay);
            a.removeEventListener("seeked", syncB);
            a.removeEventListener("loadeddata", syncB);
            if (st.vfc && typeof a.cancelVideoFrameCallback === "function") a.cancelVideoFrameCallback(st.vfc);
            b.removeAttribute("src");
            try { b.load(); } catch (_e) { /* */ }
            wrap.parentElement?.insertBefore(a, wrap);
            wrap.remove();
        },
    };
}
