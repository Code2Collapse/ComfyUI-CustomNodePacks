/**
 * Video Comparer (C2C) v2 player (L9.40, owner 2026-10-23: "give more power to video comparer also. its weak").
 *
 * - Transport: Load Video's player (play / pause, backwards, J / K / L, step, loop, in / out, timecode, full screen)
 *   on A; every other source follows A frame by frame, B shifted by b_offset frames.
 * - Views: A, B, wipe (vertical / horizontal / diagonal), difference with gain, onion, flicker (js/_c2c_ab_view.js),
 *   side by side, and a 2 x 2 grid of A / B / C / D; zoom, pan, 1:1, loupe with pixel values.
 * - Metrics: the run's per-frame PSNR / SSIM / colour difference as a strip under the scrubber; hover reads the
 *   values, click goes to that frame, the worst frame of each is marked.
 * - Nothing runs while paused; followers seek once per presented frame of A while it plays.
 */
import { attachPlayer, playerButton } from "./c2c_video_player.js";
import { attachABView, WIPE_SHAPES } from "./_c2c_ab_view.js";

const STYLE_ID = "c2c-cmpplayer-v1";
const METRIC_COLOURS = { psnr: "#58c4dc", ssim: "#9bd66f", delta_e: "#f08a5d" };

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.c2c-cmp { display: flex; flex-direction: column; gap: 4px; width: 100%; }
.c2c-cmp [hidden] { display: none !important; }   /* display: grid / flex below would beat [hidden] */
.c2c-cmp__view { position: relative; width: 100%; background: var(--cu-sunken, #111); border-radius: 6px; overflow: hidden; }
.c2c-cmp__view > video { width: 100%; height: 100%; display: block; object-fit: contain; }
.c2c-cmp__grid { position: absolute; inset: 0; display: grid; gap: 2px; background: var(--cu-ground, #000); z-index: 6; }
.c2c-cmp__grid[data-n="2"] { grid-template-columns: 1fr 1fr; }
.c2c-cmp__grid[data-n="4"] { grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; }
.c2c-cmp__cell { position: relative; overflow: hidden; background: var(--cu-sunken, #111); }
.c2c-cmp__cell video { width: 100%; height: 100%; object-fit: contain; display: block; }
.c2c-cmp__cell span { position: absolute; left: 4px; top: 4px; font-size: 10px; padding: 0 5px; border-radius: 3px;
  background: var(--cu-raised, #222); color: var(--cu-ink, #eee); opacity: 0.85; }
.c2c-cmp__tools { display: flex; flex-wrap: wrap; gap: 2px; align-items: center; }
.c2c-cmp__offset { width: 54px; font-size: 10px; background: var(--cu-raised, #222); color: var(--cu-ink, #eee);
  border: 1px solid var(--cu-edge-strong, #444); border-radius: 4px; padding: 1px 4px; }
.c2c-cmp__metrics { position: relative; height: 46px; width: 100%; cursor: crosshair; }
.c2c-cmp__metrics canvas { width: 100%; height: 100%; display: block; }
.c2c-cmp__readout { font-size: 10px; color: var(--cu-ink-soft, #aaa); font-variant-numeric: tabular-nums; min-height: 13px; }
`;
    document.head.appendChild(s);
}

/** A <video> that shows frame (frameOfMaster - offset) of its own clip, following ``master``. */
function follower(video, master, getFps, getOffset) {
    let busy = false, want = -1;
    const frames = () => Math.max(0, Math.floor((video.duration || 0) * getFps() + 1e-3));
    const sync = () => {
        if (!video.src) return;
        const f = Math.floor((master.currentTime || 0) * getFps() + 1e-3) - getOffset();
        const inside = f >= 0 && (frames() === 0 || f < frames());
        video.style.visibility = inside ? "visible" : "hidden";
        if (!inside || (f === want && !busy)) return;
        want = f;
        if (busy) return;
        busy = true;
        video.currentTime = (f + 0.5) / getFps();
    };
    video.addEventListener("seeked", () => { busy = false; sync(); });
    return sync;
}

export function createComparePlayer(opts = {}) {
    ensureStyles();
    const root = document.createElement("div");
    root.className = "c2c-cmp";
    root.dataset.c2cCompare = "player";
    const view = document.createElement("div");
    view.className = "c2c-cmp__view";
    view.style.aspectRatio = "16 / 9";
    const vA = document.createElement("video");
    vA.muted = true;
    vA.playsInline = true;
    vA.preload = "auto";
    view.append(vA);
    const grid = document.createElement("div");
    grid.className = "c2c-cmp__grid";
    grid.hidden = true;
    view.append(grid);
    const timeline = document.createElement("div");
    const tools = document.createElement("div");
    tools.className = "c2c-cmp__tools";
    const metricsBox = document.createElement("div");
    metricsBox.className = "c2c-cmp__metrics";
    metricsBox.hidden = true;
    const mc = document.createElement("canvas");
    metricsBox.append(mc);
    const readout = document.createElement("div");
    readout.className = "c2c-cmp__readout";
    root.append(view, timeline, metricsBox, readout, tools);

    const st = { fps: 24, offset: 0, sources: {}, labels: {}, metrics: null, mode: "wipe", frames: 1, active: false };
    const player = attachPlayer(timeline, vA, { verb: "loop", fullscreenTarget: root,
                                                inTitle: "Loop start", outTitle: "Loop end" });
    const ab = attachABView(view, vA, { labels: ["A", "B"] });

    // grid / side-by-side cells: A is shown by a follower too, so all cells are equal
    const cells = {};
    for (const k of ["a", "b", "c", "d"]) {
        const cell = document.createElement("div");
        cell.className = "c2c-cmp__cell";
        const v = document.createElement("video");
        v.muted = true;
        v.playsInline = true;
        v.preload = "auto";
        const lab = document.createElement("span");
        cell.append(v, lab);
        cells[k] = { cell, v, lab, sync: follower(v, vA, () => st.fps, () => (k === "b" ? st.offset : 0)) };
    }
    const syncCells = () => { if (!grid.hidden) for (const c of Object.values(cells)) c.sync(); };
    let vfc = 0;
    const onFrame = () => {
        syncCells();
        drawMetrics();
        if (!vA.paused && typeof vA.requestVideoFrameCallback === "function") vfc = vA.requestVideoFrameCallback(onFrame);
    };
    vA.addEventListener("play", () => { if (typeof vA.requestVideoFrameCallback === "function") vfc = vA.requestVideoFrameCallback(onFrame); });
    vA.addEventListener("seeked", () => { syncCells(); drawMetrics(); });

    // ── tools ──
    const btns = {};
    const setMode = (m) => {
        st.mode = m;
        const gridMode = m === "side" || m === "grid";
        grid.hidden = !gridMode;
        if (gridMode) {
            const keys = m === "side" ? ["a", "b"] : ["a", "b", "c", "d"];
            grid.dataset.n = String(keys.length);
            grid.replaceChildren(...keys.map((k) => cells[k].cell));
            syncCells();
        } else {
            ab.setMode(m);
        }
        for (const [k, b] of Object.entries(btns)) b.setAttribute("aria-pressed", String(k === m));
    };
    const add = (m, label, title) => { btns[m] = playerButton(label, title, () => setMode(m)); return btns[m]; };
    let shapeIdx = 0;
    const bShape = playerButton("|", "Wipe direction: vertical, horizontal, diagonal", () => {
        shapeIdx = (shapeIdx + 1) % WIPE_SHAPES.length;
        ab.setShape(WIPE_SHAPES[shapeIdx]);
        bShape.textContent = ["|", "—", "/"][shapeIdx];
    });
    let gain = 1;
    const bGain = playerButton("×1", "Difference gain (click: x2, shift-click: /2)", () => {});
    bGain.addEventListener("click", (e) => {
        gain = e.shiftKey ? Math.max(1, gain / 2) : Math.min(64, gain * 2);
        ab.setGain(gain);
        bGain.textContent = `×${gain}`;
    }, { capture: true });
    let loupe = false;
    const bLoupe = playerButton("loupe", "Magnifier with pixel values under the pointer", () => {
        loupe = !loupe;
        ab.setLoupe(loupe);
        bLoupe.setAttribute("aria-pressed", String(loupe));
    });
    const offset = document.createElement("input");
    offset.type = "number";
    offset.className = "c2c-cmp__offset";
    offset.title = "B starts this many frames after A (b_offset; the metrics use it too after a run)";
    offset.addEventListener("change", () => {
        st.offset = Number(offset.value) | 0;
        ab.setB(st.sources.b || "", { fps: st.fps, frames: 0, offset: st.offset });
        opts.onOffset?.(st.offset);
        syncCells();
    });
    const offLabel = document.createElement("span");
    offLabel.className = "c2c-cmp__readout";
    offLabel.textContent = "B offset";
    tools.append(add("A", "A", "A only"), add("B", "B", "B only"), add("wipe", "A|B", "Wipe (drag the line)"), bShape,
                 add("difference", "diff", "|A - B|"), bGain, add("onion", "onion", "B over A at 50 %"),
                 add("flicker", "flicker", "A and B in turn"), add("side", "A B", "Side by side, in step"),
                 add("grid", "2×2", "A, B, C, D in step"),
                 playerButton("fit", "Fit (or double-click)", () => ab.fit()),
                 playerButton("1:1", "One video pixel per screen pixel", () => ab.oneToOne()), bLoupe,
                 offLabel, offset);

    // ── metrics strip ──
    function drawMetrics() {
        const m = st.metrics;
        if (!m || metricsBox.hidden) return;
        const w = Math.max(1, metricsBox.clientWidth), h = Math.max(1, metricsBox.clientHeight);
        const dpr = window.devicePixelRatio || 1;
        if (mc.width !== Math.round(w * dpr) || mc.height !== Math.round(h * dpr)) {
            mc.width = Math.round(w * dpr);
            mc.height = Math.round(h * dpr);
        }
        const c = mc.getContext("2d");
        c.setTransform(dpr, 0, 0, dpr, 0, 0);
        c.clearRect(0, 0, w, h);
        const css = getComputedStyle(root);
        c.fillStyle = css.getPropertyValue("--cu-sunken").trim() || "#111";
        c.fillRect(0, 0, w, h);
        const n = m.frames || 1;
        const x = (i) => (n > 1 ? (i / (n - 1)) * (w - 2) + 1 : w / 2);
        const curve = (vals, norm, colour) => {
            c.strokeStyle = colour;
            c.lineWidth = 1.25;
            c.beginPath();
            let pen = false;
            vals.forEach((v, i) => {
                if (v == null) { pen = false; return; }
                const y = h - 3 - norm(v) * (h - 6);
                if (!pen) { c.moveTo(x(i), y); pen = true; } else c.lineTo(x(i), y);
            });
            c.stroke();
        };
        // higher = better for all three lines: PSNR 20..60 dB, SSIM 0.5..1, dE 10..0
        curve(m.psnr || [], (v) => Math.max(0, Math.min(1, (v - 20) / 40)), METRIC_COLOURS.psnr);
        curve(m.ssim || [], (v) => Math.max(0, Math.min(1, (v - 0.5) / 0.5)), METRIC_COLOURS.ssim);
        curve(m.delta_e || [], (v) => Math.max(0, Math.min(1, 1 - v / 10)), METRIC_COLOURS.delta_e);
        for (const [k, i] of Object.entries(m.worst || {})) {
            c.fillStyle = METRIC_COLOURS[k];
            c.fillRect(x(i) - 1, 0, 2, 5);
        }
        const f = Math.floor((vA.currentTime || 0) * st.fps + 1e-3);
        c.fillStyle = css.getPropertyValue("--cu-ink").trim() || "#eee";
        c.fillRect(x(Math.min(n - 1, f)) - 0.5, 0, 1, h);
    }
    function frameAt(ev) {
        const r = metricsBox.getBoundingClientRect();
        const n = st.metrics?.frames || 1;
        return Math.max(0, Math.min(n - 1, Math.round(((ev.clientX - r.left) / r.width) * (n - 1))));
    }
    function readoutFor(i) {
        const m = st.metrics;
        if (!m) return "";
        const p = m.psnr?.[i], s = m.ssim?.[i], d = m.delta_e?.[i];
        if (p == null) return `frame ${i}: no B frame at this offset`;
        return `frame ${i}: PSNR ${p.toFixed(2)} dB · SSIM ${s.toFixed(4)} · ΔE ${d.toFixed(3)}`;
    }
    metricsBox.addEventListener("pointermove", (e) => { readout.textContent = readoutFor(frameAt(e)); });
    metricsBox.addEventListener("pointerleave", () => { readout.textContent = summaryText(); });
    metricsBox.addEventListener("click", (e) => player.seekFrame(frameAt(e)));
    function summaryText() {
        const m = st.metrics;
        if (!m?.summary) return m?.note || "";
        const s = m.summary;
        return `A vs B over ${m.pairs} frames: PSNR ${s.psnr.mean.toFixed(2)} dB (worst ${s.psnr.min.toFixed(2)} at `
            + `${m.worst.psnr}) · SSIM ${s.ssim.mean.toFixed(4)} · ΔE ${s.delta_e.mean.toFixed(3)} `
            + `(worst ${s.delta_e.max.toFixed(3)} at ${m.worst.delta_e})`;
    }

    setMode("wipe");
    return {
        root,
        /** sources: {a, b, c, d} urls (video or still), labels, fps, frames (of A), offset. */
        setSources({ sources = {}, labels = {}, fps = 24, frames = 0, offset: off = 0, aspect }) {
            st.sources = sources;
            st.labels = labels;
            st.fps = fps > 0 ? fps : 24;
            st.offset = off | 0;
            offset.value = String(st.offset);
            if (aspect) view.style.aspectRatio = String(aspect);
            if (sources.a && vA.src !== sources.a) vA.src = sources.a;
            ab.setFpsA(st.fps);
            ab.setLabels(labels.a || "A", labels.b || "B");
            ab.setB(sources.b || "", { fps: st.fps, frames: 0, offset: st.offset });
            for (const k of ["a", "b", "c", "d"]) {
                const c = cells[k];
                c.lab.textContent = labels[k] || k.toUpperCase();
                if (sources[k]) { if (c.v.src !== sources[k]) c.v.src = sources[k]; } else c.v.removeAttribute("src");
            }
            btns.grid.disabled = !(sources.c || sources.d);
            st.frames = Math.max(1, frames | 0);
            player.setDomain({ frames: st.frames, fps: st.fps, step: 1, previewFrames: st.frames, previewFps: st.fps });
            player.setSelection({ in: 0, out: st.frames - 1, count: st.frames, nth: 1 });
            syncCells();
        },
        setMetrics(m) {
            st.metrics = m && (m.psnr || m.note) ? m : null;
            metricsBox.hidden = !st.metrics?.psnr;
            readout.textContent = summaryText();
            requestAnimationFrame(drawMetrics);
        },
        pause() { player.pause(); },
        destroy() {
            if (vfc && typeof vA.cancelVideoFrameCallback === "function") vA.cancelVideoFrameCallback(vfc);
            player.destroy();
            ab.destroy();
            for (const c of Object.values(cells)) { c.v.removeAttribute("src"); }
            vA.removeAttribute("src");
            root.remove();
        },
    };
}
