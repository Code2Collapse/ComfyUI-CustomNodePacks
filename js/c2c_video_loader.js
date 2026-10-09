/**
 * C2C video loader nodes — probe info strip, selection timeline, H.264 preview.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ensureStyles, emptyState, statusLine, measureRootContent, installResizeFloor } from "./c2c_ui/index.js";
import { setHidden } from "./_widget_visibility.js";

const STYLE_ID = "c2c-vloader-v1";
const DEBOUNCE_MS = 350;
const MARGIN = 2;
const TIMELINE_H = 28;
const INFO_H = 26;
const GAP = 6;
const PREVIEW_MIN_H = 120;
const DEFAULT_ASPECT = 16 / 9;

const LOADERS = {
    LoadVideoC2C: {
        // Our server-side preview, not core's: core's upload preview plays the ORIGINAL file in the browser, and
        // Chrome cannot decode ProRes, DNxHR, HEVC 10-bit or FFV1/MKV - those showed no preview at all (L2.27,
        // measured; the server transcodes each in under a second). Core's own preview widget is hidden.
        hasPreview: true,
        hideCorePreview: true,
        mode: "upload",
        emptyTitle: "Load Video (C2C)",
        emptyHint: "Choose a video from the input folder.",
    },
    // OmniScale's loader (ComfyUI-OmniScale) uses core's upload preview too, with the same blind spot for pro
    // formats (owner A9: "Load Video (C2C) and OmniScale Load Video"). With CustomNodePacks installed it gets our
    // transcoded preview; without it, core's stays. Nothing is imported across packs.
    OmniScaleLoadVideo: {
        hasPreview: true,
        hideCorePreview: true,
        mode: "upload",
        emptyTitle: "OmniScale Load Video",
        emptyHint: "Choose a video from the input folder.",
    },
    LoadVideoPathC2C: {
        hasPreview: true,
        mode: "video",
        emptyTitle: "Load Video Path (C2C)",
        emptyHint: "Paste a video path, a folder of images, or shot.####.exr",
    },
    LoadImagesPathC2C: {
        hasPreview: true,
        mode: "images",
        emptyTitle: "Load Images Path (C2C)",
        emptyHint: "Paste a directory of sequential images.",
    },
};

const VIDEO_WIDGETS = [
    "force_rate", "skip_first_frames", "select_every_nth",
    "frame_load_cap", "format", "sequence_fps",
];
const IMAGE_WIDGETS = ["image_load_cap", "skip_first_images", "select_every_nth"];
const PATH_WIDGETS = ["video", "directory"];

function ensureVloaderStyles() {
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
/* an author "display" beats the [hidden] attribute - restate it */
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

function fmtNum(n) {
    return Number(n).toLocaleString("en-US");
}

function widgetValue(node, name) {
    const w = node.widgets?.find((x) => x.name === name);
    return w != null ? w.value : undefined;
}

function collectParams(node, cfg) {
    const p = new URLSearchParams();
    if (cfg.mode === "upload") {
        const video = widgetValue(node, "video");
        if (!video) return null;
        p.set("filename", String(video));
        p.set("type", "input");
    } else if (cfg.mode === "images") {
        const dir = widgetValue(node, "directory");
        if (!dir || !String(dir).trim()) return null;
        p.set("directory", String(dir).trim());
        p.set("skip_first_images", String(widgetValue(node, "skip_first_images") ?? 0));
        p.set("image_load_cap", String(widgetValue(node, "image_load_cap") ?? 0));
        p.set("select_every_nth", String(widgetValue(node, "select_every_nth") ?? 1));
    } else {
        const path = widgetValue(node, "video");
        if (!path || !String(path).trim()) return null;
        p.set("path", String(path).trim());
        p.set("force_rate", String(widgetValue(node, "force_rate") ?? 0));
        p.set("skip_first_frames", String(widgetValue(node, "skip_first_frames") ?? 0));
        p.set("select_every_nth", String(widgetValue(node, "select_every_nth") ?? 1));
        p.set("frame_load_cap", String(widgetValue(node, "frame_load_cap") ?? 0));
        p.set("format", String(widgetValue(node, "format") ?? "None"));
        const sf = widgetValue(node, "sequence_fps");
        if (sf != null) p.set("sequence_fps", String(sf));
    }
    if (cfg.mode === "upload") {
        p.set("force_rate", String(widgetValue(node, "force_rate") ?? 0));
        p.set("skip_first_frames", String(widgetValue(node, "skip_first_frames") ?? 0));
        p.set("select_every_nth", String(widgetValue(node, "select_every_nth") ?? 1));
        p.set("frame_load_cap", String(widgetValue(node, "frame_load_cap") ?? 0));
        p.set("format", String(widgetValue(node, "format") ?? "None"));
    }
    return p;
}

function previewHeight(width, aspect, hasPreview) {
    if (!hasPreview) return 0;
    return Math.max(PREVIEW_MIN_H, Math.round(width / aspect));
}

// Everything under the preview, measured when laid out: chips wrap to more
// rows on a narrow node and the status line appears with an error, so a
// fixed estimate would let them hang past the node's bottom edge.
function restHeight(ui) {
    let h = 0;
    let n = 0;
    for (const el of [ui.timeline, ui.info, ui.status.el]) {
        if (!el || el.offsetParent === null) continue;
        h += el.offsetHeight;
        n += 1;
    }
    return n ? h + GAP * (n - 1) : TIMELINE_H + INFO_H + GAP;
}

function reservedHeight(width, cfg, aspect, ui) {
    let h = restHeight(ui);
    // the preview width is the widget width minus the DOM margin on each side
    if (cfg.hasPreview) h += previewHeight(Math.max(1, width - 2 * MARGIN), aspect, true) + GAP;
    return h + 2 * MARGIN;
}

function buildDom(cfg) {
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
        sel,
        ticks,
        labelLeft,
        labelRight,
        info,
        status,
    };
}

function setEmpty(ui, cfg) {
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

function setLoading(ui, on, aspect) {
    if (!ui.viewport) return;
    ui.viewport.style.aspectRatio = String(aspect || DEFAULT_ASPECT);
    ui.loadingEl.hidden = !on;
    if (on) ui.viewport.querySelector(".c2c-ui-empty")?.remove();
}

function renderChips(infoEl, data) {
    infoEl.innerHTML = "";
    const sel = data.selection || {};
    const chips = [];
    if (sel.count != null) {
        const dur = sel.count && sel.fps ? (sel.count / sel.fps).toFixed(1) : "0";
        chips.push(`${fmtNum(sel.count)} f · ${Number(sel.fps || 0).toFixed(2)} fps · ${dur} s`);
    }
    if (data.width && data.height) chips.push(`${data.width}×${data.height}`);
    const codec = data.codec ? String(data.codec).toUpperCase() : "";
    const colour = data.colour || {};
    // YUV sources: the matrix and range the pixels were decoded with.
    // RGB sources (image sequences): what the values mean - linear or sRGB.
    const MATRIX = { bt709: "BT.709", bt601: "BT.601", bt2020: "BT.2020", smpte240m: "SMPTE 240M", fcc: "FCC" };
    const RANGE = { tv: "limited", pc: "full" };
    const TRANSFER = { linear: "linear", srgb: "sRGB", pq: "PQ", hlg: "HLG", log: "log" };
    const colourName = colour.matrix && colour.matrix !== "rgb"
        ? [MATRIX[colour.matrix] || colour.matrix, RANGE[colour.range] || colour.range].filter(Boolean).join(" ")
        : (TRANSFER[colour.transfer] || colour.transfer || "");
    const colourBits = [codec, data.bit_depth ? `${data.bit_depth}-bit` : "", colourName]
        .filter(Boolean).join(" · ");
    if (colourBits) chips.push(colourBits);
    if (data.audio) chips.push(`audio ${data.audio.sample_rate} Hz`);
    if (data.rotation) chips.push(`${data.rotation}°`);
    if (data.has_alpha) chips.push("alpha");
    for (const text of chips) {
        const chip = document.createElement("span");
        chip.className = "c2c-vloader__chip";
        chip.textContent = text;
        infoEl.appendChild(chip);
    }
}

function renderTimeline(ui, data) {
    const srcCount = data.frame_count || data.selection?.source_frame_count || 1;
    const sel = data.selection || {};
    const first = sel.first_source ?? 0;
    const last = sel.last_source ?? 0;
    const span = Math.max(1, last - first + 1);
    const leftPct = (first / Math.max(1, srcCount - 1)) * 100;
    const widthPct = (span / Math.max(1, srcCount)) * 100;
    const left = `${Math.min(100, leftPct)}%`;
    const width = `${Math.max(0.5, Math.min(100 - leftPct, widthPct))}%`;
    ui.sel.style.left = left;
    ui.sel.style.width = width;
    // one tick per LOADED frame, inside the selected span only
    const showTicks = sel.count != null && sel.count > 1 && sel.count <= 200;
    ui.ticks.style.display = showTicks ? "block" : "none";
    if (showTicks) {
        ui.ticks.style.left = left;
        ui.ticks.style.width = width;
        ui.ticks.style.backgroundSize = `calc(100% / ${sel.count}) 100%`;
    }
    ui.labelLeft.textContent = `frame ${fmtNum(first)}`;
    ui.labelRight.textContent = `of ${fmtNum(srcCount)}`;
}

function attachPauseObservers(node, video) {
    if (!video) return () => {};
    const pause = () => { try { video.pause(); } catch (_e) { /* ignore */ } };
    const io = new IntersectionObserver((entries) => {
        if (!entries.some((e) => e.isIntersecting)) pause();
    }, { threshold: 0.05 });
    io.observe(video);
    const onVis = () => { if (document.hidden) pause(); };
    document.addEventListener("visibilitychange", onVis);
    // no polling timer: pause from the node's own collapse (both renderers
    // route collapsing through LGraphNode.collapse)
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

function mountLoaderWidget(node, cfg) {
    const ui = buildDom(cfg);
    let aspect = DEFAULT_ASPECT;
    let abortCtrl = null;
    let debounceTimer = null;
    let cleanupVideo = () => {};

    const refresh = async () => {
        if (abortCtrl) abortCtrl.abort();
        abortCtrl = new AbortController();
        const params = collectParams(node, cfg);
        if (!params) {
            setEmpty(ui, cfg);
            ui.status.setText("", "default");
            if (cfg.hasPreview) ui.video?.removeAttribute("src");
            return;
        }

        setLoading(ui, cfg.hasPreview, aspect);
        ui.status.setText("Loading preview…", "default");
        if (cfg.hideCorePreview) hideCorePreview(node);

        try {
            const probeUrl = api.apiURL("/c2c/video/probe?" + params.toString());
            const probeResp = await fetch(probeUrl, { signal: abortCtrl.signal });
            const probeJson = await probeResp.json();
            if (!probeResp.ok) {
                throw new Error(probeJson.error || "Could not read video information.");
            }

            aspect = (probeJson.width && probeJson.height)
                ? probeJson.width / probeJson.height
                : DEFAULT_ASPECT;
            if (ui.viewport) ui.viewport.style.aspectRatio = String(aspect);

            renderTimeline(ui, probeJson);
            renderChips(ui.info, probeJson);
            ui.status.setText("", "default");

            if (cfg.hasPreview && ui.video) {
                const previewUrl = api.apiURL("/c2c/video/preview?" + params.toString());
                ui.viewport.querySelector(".c2c-ui-empty")?.remove();
                setLoading(ui, true, aspect);
                ui.video.src = previewUrl;
                ui.video.onloadeddata = () => setLoading(ui, false, aspect);
                ui.video.onerror = () => {
                    setLoading(ui, false, aspect);
                    ui.status.setText("Could not play the preview.", "danger");
                };
                try { await ui.video.play(); } catch (_e) { /* autoplay blocked */ }
            } else {
                setLoading(ui, false, aspect);
            }

            fitNode();
        } catch (err) {
            if (err.name === "AbortError") return;
            setLoading(ui, false, aspect);
            if (cfg.hasPreview && !ui.video?.src) setEmpty(ui, cfg);
            ui.status.setText(err.message || "Something went wrong.", "danger");
            fitNode();
        }
    };

    const scheduleRefresh = () => {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => refresh(), DEBOUNCE_MS);
    };

    const nodeWidth = () => node.size?.[0] || 340;
    const getReserved = (w) => reservedHeight(w || nodeWidth(), cfg, aspect, ui);
    // height follows content (the aspect of the clip, how the chips wrapped);
    // width stays the user's. One frame later, so the DOM has laid out.
    const fitNode = () => requestAnimationFrame(() => {
        const need = node.computeSize?.();
        if (!need || !node.setSize) return;
        node.setSize([node.size[0], need[1]]);
        node.setDirtyCanvas?.(true, true);
    });
    const widgetRef = node.addDOMWidget("c2c_preview", "div", ui.root, {
        serialize: false,
        margin: MARGIN,
        getMinHeight: () => getReserved(nodeWidth()),
    });
    widgetRef.computeSize = (w) => [w, getReserved(w)];

    const uninstallFloor = installResizeFloor(ui.root, measureRootContent);

    const hookWidget = (name) => {
        const w = node.widgets?.find((x) => x.name === name);
        if (!w) return;
        const orig = w.callback;
        w.callback = function (...args) {
            const out = orig?.apply(this, args);
            scheduleRefresh();
            return out;
        };
    };

    for (const name of [...PATH_WIDGETS, ...VIDEO_WIDGETS, ...IMAGE_WIDGETS]) {
        hookWidget(name);
    }

    if (cfg.hasPreview && ui.video) {
        cleanupVideo = attachPauseObservers(node, ui.video);
    }

    const origRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        if (abortCtrl) abortCtrl.abort();
        if (debounceTimer) clearTimeout(debounceTimer);
        cleanupVideo();
        uninstallFloor();
        return origRemoved?.apply(this, args);
    };

    setEmpty(ui, cfg);
    scheduleRefresh();

    return widgetRef;
}

/** Hide core's upload preview (a <video> of the original file) so the node shows one preview, ours. Core adds it
 *  with the upload widget, possibly after onNodeCreated, so this runs again a few times and on every refresh. */
function hideCorePreview(node) {
    // Nodes 2.0 draws core's player from the node-output store, not from a widget: scope a CSS rule to this
    // node's root (Vue leaves attributes it does not render alone, and every refresh re-applies it).
    if (globalThis.LiteGraph?.vueNodesMode) {
        if (!document.getElementById("c2c-no-core-video-preview")) {
            const s = document.createElement("style");
            s.id = "c2c-no-core-video-preview";
            s.textContent = "[data-c2c-no-core-preview] .video-preview { display: none !important; }";
            document.head.appendChild(s);
        }
        document.querySelector(`[data-node-id="${node.id}"]`)?.setAttribute("data-c2c-no-core-preview", "1");
    }
    const w = node.widgets?.find((x) => x.name === "video-preview" || (x.type === "video" && x.name !== "c2c_preview"));
    if (!w) return false;
    const v = w.element?.querySelector?.("video") || (w.element?.tagName === "VIDEO" ? w.element : null);
    try { v?.pause(); } catch (_e) { /* ignore */ }
    if (!w.__c2cHidden) {
        w.__c2cHidden = true;
        setHidden(w, true);
        const need = node.computeSize?.();
        if (need) node.setSize([node.size[0], need[1]]);
        node.setDirtyCanvas?.(true, true);
    }
    return true;
}

function setupLoaderNode(node, cfg) {
    if (node._c2cVideoLoaderMounted) return;
    node._c2cVideoLoaderMounted = true;
    mountLoaderWidget(node, cfg);
    if (cfg.hideCorePreview) {
        for (const ms of [0, 250, 1000, 3000]) setTimeout(() => hideCorePreview(node), ms);
        const cb = node.widgets?.find((x) => x.name === "video");
        if (cb) {
            const orig = cb.callback;
            cb.callback = function (...a) { const r = orig?.apply(this, a); setTimeout(() => hideCorePreview(node), 0); return r; };
        }
    }
}

if (!globalThis.__c2cVideoLoadersExt) {
    globalThis.__c2cVideoLoadersExt = true;
    app.registerExtension({
        name: "C2C.VideoLoaders",
        async beforeRegisterNodeDef(nodeType, nodeData, _app) {
            const cfg = LOADERS[nodeData.name];
            if (!cfg) return;
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...args) {
                const r = onCreated?.apply(this, args);
                setupLoaderNode(this, cfg);
                return r;
            };
        },
    });
}
