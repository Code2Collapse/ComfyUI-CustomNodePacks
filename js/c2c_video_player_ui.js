/**
 * Shared C2C video player DOM — timeline, chips, preview stage.
 * Used by c2c_video_loader.js and c2c_save_video.js.
 */

import { ensureStyles, emptyState, statusLine } from "./c2c_ui/index.js";

export const STYLE_ID = "c2c-vloader-v1";
export const MARGIN = 2;
export const TIMELINE_H = 28;
export const INFO_H = 26;
export const GAP = 6;
export const PREVIEW_MIN_H = 120;
export const DEFAULT_ASPECT = 16 / 9;

export function ensureVloaderStyles() {
    ensureStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.c2c-vloader { display: flex; flex-direction: column; gap: ${GAP}px; width: 100%; }
.c2c-vloader__preview .c2c-ui-stage__viewport { position: relative; width: 100%; background: var(--cu-sunken); border-radius: 6px; overflow: hidden; }
.c2c-vloader__video { width: 100%; height: 100%; display: block; object-fit: contain; background: var(--cu-ground); }
.c2c-vloader__loading { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  background: var(--cu-sunken); }
.c2c-vloader__loading::after { content: ""; width: 28px; height: 28px; border-radius: 50%;
  border: 2px solid var(--cu-edge); border-top-color: var(--cu-accent); animation: c2c-vloader-spin 0.8s linear infinite; }
@keyframes c2c-vloader-spin { to { transform: rotate(360deg); } }
.c2c-vloader [hidden] { display: none !important; }
.c2c-vloader__timeline { display: flex; flex-direction: column; gap: 4px; }
.c2c-vloader__timeline-track { position: relative; height: 10px; border-radius: 4px; background: var(--cu-edge); overflow: hidden; }
.c2c-vloader__timeline-sel { position: absolute; top: 0; bottom: 0; background: var(--cu-accent); border-radius: 3px; opacity: 0.85; }
.c2c-vloader__timeline-ticks { position: absolute; top: 0; bottom: 0; pointer-events: none; opacity: 0.55;
  background-image: linear-gradient(90deg, var(--cu-ground) 0 1px, transparent 1px); background-repeat: repeat-x; }
.c2c-vloader__timeline-labels { display: flex; justify-content: space-between; font-size: 10px; color: var(--cu-ink-soft); }
.c2c-vloader__info { display: flex; flex-wrap: wrap; gap: 4px; }
.c2c-vloader__chip { font-size: 10px; line-height: 1.3; padding: 2px 6px; border-radius: 4px;
  background: var(--cu-raised); border: 1px solid var(--cu-edge-strong); color: var(--cu-ink-soft); white-space: nowrap; }
`;
    document.head.appendChild(style);
}

export function fmtNum(n) {
    return Number(n).toLocaleString("en-US");
}

export function previewHeight(width, aspect, hasPreview) {
    if (!hasPreview) return 0;
    return Math.max(PREVIEW_MIN_H, Math.round(width / aspect));
}

export function restHeight(ui) {
    let h = 0;
    let n = 0;
    for (const el of [ui.timeline, ui.info, ui.status?.el]) {
        if (!el || el.offsetParent === null) continue;
        h += el.offsetHeight;
        n += 1;
    }
    return n ? h + GAP * (n - 1) : TIMELINE_H + INFO_H + GAP;
}

export function reservedHeight(width, hasPreview, aspect, ui) {
    let h = restHeight(ui);
    if (hasPreview) h += previewHeight(Math.max(1, width - 2 * MARGIN), aspect, true) + GAP;
    return h + 2 * MARGIN;
}

export function buildDom(cfg) {
    ensureVloaderStyles();
    const root = document.createElement("div");
    root.className = "c2c-ui c2c-vloader";

    let previewWrap = null;
    let viewport = null;
    let video = null;
    let loadingEl = null;

    if (cfg.hasPreview) {
        previewWrap = document.createElement("div");
        previewWrap.className = "c2c-vloader__preview c2c-ui-stage";
        viewport = document.createElement("div");
        viewport.className = "c2c-ui-stage__viewport";
        viewport.style.aspectRatio = String(DEFAULT_ASPECT);
        loadingEl = document.createElement("div");
        loadingEl.className = "c2c-vloader__loading";
        loadingEl.hidden = true;
        video = document.createElement("video");
        video.className = "c2c-vloader__video";
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.autoplay = true;
        viewport.appendChild(video);
        viewport.appendChild(loadingEl);
        previewWrap.appendChild(viewport);
        root.appendChild(previewWrap);
    }

    const timeline = document.createElement("div");
    timeline.className = "c2c-vloader__timeline";
    const track = document.createElement("div");
    track.className = "c2c-vloader__timeline-track";
    const sel = document.createElement("div");
    sel.className = "c2c-vloader__timeline-sel";
    const ticks = document.createElement("div");
    ticks.className = "c2c-vloader__timeline-ticks";
    track.appendChild(sel);
    track.appendChild(ticks);
    const labels = document.createElement("div");
    labels.className = "c2c-vloader__timeline-labels";
    const labelLeft = document.createElement("span");
    const labelRight = document.createElement("span");
    labels.appendChild(labelLeft);
    labels.appendChild(labelRight);
    timeline.appendChild(track);
    timeline.appendChild(labels);
    root.appendChild(timeline);

    const info = document.createElement("div");
    info.className = "c2c-vloader__info";
    root.appendChild(info);

    const status = statusLine();
    root.appendChild(status.el);

    return {
        root,
        previewWrap,
        viewport,
        video,
        loadingEl,
        timeline,
        track,
        sel,
        ticks,
        labelLeft,
        labelRight,
        info,
        status,
    };
}

export function setEmpty(ui, cfg) {
    if (!ui.viewport) return;
    ui.video?.pause();
    if (ui.video) ui.video.removeAttribute("src");
    ui.viewport.querySelector(".c2c-ui-empty")?.remove();
    ui.viewport.insertBefore(
        emptyState({ title: cfg.emptyTitle, hint: cfg.emptyHint }),
        ui.loadingEl,
    );
    ui.loadingEl.hidden = true;
}

export function setLoading(ui, on, aspect) {
    if (!ui.viewport) return;
    ui.viewport.style.aspectRatio = String(aspect || DEFAULT_ASPECT);
    ui.loadingEl.hidden = !on;
    if (on) ui.viewport.querySelector(".c2c-ui-empty")?.remove();
}

export function renderChips(infoEl, data) {
    infoEl.innerHTML = "";
    const sel = data.selection || {};
    const chips = [];
    if (sel.count != null) {
        const dur = sel.count && sel.fps ? (sel.count / sel.fps).toFixed(1) : "0";
        chips.push(`${fmtNum(sel.count)} f · ${Number(sel.fps || 0).toFixed(2)} fps · ${dur} s`);
    } else if (data.frames != null) {
        const dur = data.frames && data.fps ? (data.frames / data.fps).toFixed(1) : "0";
        chips.push(`${fmtNum(data.frames)} f · ${Number(data.fps || 0).toFixed(2)} fps · ${dur} s`);
    }
    if (data.width && data.height) chips.push(`${data.width}×${data.height}`);
    const codec = (data.codec || data.format || "") ? String(data.codec || data.format).toUpperCase() : "";
    const colour = data.colour || {};
    const MATRIX = { bt709: "BT.709", bt601: "BT.601", bt2020: "BT.2020", smpte240m: "SMPTE 240M", fcc: "FCC" };
    const RANGE = { tv: "limited", pc: "full" };
    const TRANSFER = { linear: "linear", srgb: "sRGB", pq: "PQ", hlg: "HLG", log: "log" };
    const colourName = data.colorspace
        || (colour.matrix && colour.matrix !== "rgb"
            ? [MATRIX[colour.matrix] || colour.matrix, RANGE[colour.range] || colour.range].filter(Boolean).join(" ")
            : (TRANSFER[colour.transfer] || colour.transfer || ""));
    const colourBits = [codec, data.bit_depth ? `${data.bit_depth}-bit` : "", colourName]
        .filter(Boolean).join(" · ");
    if (colourBits) chips.push(colourBits);
    if (data.audio) chips.push(`audio ${data.audio.sample_rate} Hz`);
    if (data.rotation) chips.push(`${data.rotation}°`);
    if (data.has_alpha || data.alpha) chips.push("alpha");
    for (const text of chips) {
        const chip = document.createElement("span");
        chip.className = "c2c-vloader__chip";
        chip.textContent = text;
        infoEl.appendChild(chip);
    }
}

export function renderTimeline(ui, data) {
    const srcCount = data.frame_count || data.frames || data.selection?.source_frame_count || 1;
    const sel = data.selection || {};
    const first = sel.first_source ?? 0;
    const last = sel.last_source ?? (data.frames ? data.frames - 1 : 0);
    const span = Math.max(1, last - first + 1);
    const leftPct = (first / Math.max(1, srcCount - 1)) * 100;
    const widthPct = (span / Math.max(1, srcCount)) * 100;
    const left = `${Math.min(100, leftPct)}%`;
    const width = `${Math.max(0.5, Math.min(100 - leftPct, widthPct))}%`;
    ui.sel.style.left = left;
    ui.sel.style.width = width;
    const tickCount = sel.count ?? data.frames;
    const showTicks = tickCount != null && tickCount > 1 && tickCount <= 200;
    ui.ticks.style.display = showTicks ? "block" : "none";
    if (showTicks) {
        ui.ticks.style.left = left;
        ui.ticks.style.width = width;
        ui.ticks.style.backgroundSize = `calc(100% / ${tickCount}) 100%`;
    }
    ui.labelLeft.textContent = `frame ${fmtNum(first)}`;
    ui.labelRight.textContent = `of ${fmtNum(srcCount)}`;
}

/** Playback scrub timeline (full span, click-to-seek). */
export function renderPlaybackTimeline(ui, data, video) {
    const frames = data.frames || 1;
    const fps = data.fps || 24;
    ui.sel.style.left = "0%";
    ui.sel.style.width = "100%";
    const showTicks = frames > 1 && frames <= 200;
    ui.ticks.style.display = showTicks ? "block" : "none";
    if (showTicks) {
        ui.ticks.style.left = "0%";
        ui.ticks.style.width = "100%";
        ui.ticks.style.backgroundSize = `calc(100% / ${frames}) 100%`;
    }
    const updateLabels = () => {
        const t = video?.currentTime || 0;
        const idx = Math.min(frames, Math.max(1, Math.floor(t * fps) + 1));
        ui.labelLeft.textContent = `frame ${fmtNum(idx)}`;
        ui.labelRight.textContent = `of ${fmtNum(frames)}`;
        const pct = data.frames > 1 ? (idx - 1) / (frames - 1) : 0;
        ui.sel.style.left = "0%";
        ui.sel.style.width = `${Math.max(0.5, pct * 100)}%`;
    };
    updateLabels();
    return updateLabels;
}

export function attachPauseObservers(node, video) {
    if (!video) return () => {};
    const pause = () => { try { video.pause(); } catch (_e) { /* ignore */ } };
    const io = new IntersectionObserver((entries) => {
        if (!entries.some((e) => e.isIntersecting)) pause();
    }, { threshold: 0.05 });
    io.observe(video);
    const onVis = () => { if (document.hidden) pause(); };
    document.addEventListener("visibilitychange", onVis);
    const origCollapse = node.collapse;
    node.collapse = function (...args) {
        const r = origCollapse?.apply(this, args);
        if (this.flags?.collapsed) pause();
        return r;
    };
    return () => {
        io.disconnect();
        document.removeEventListener("visibilitychange", onVis);
        node.collapse = origCollapse;
        pause();
        video.removeAttribute("src");
        try { video.load(); } catch (_e) { /* ignore */ }
    };
}
