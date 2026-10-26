/**
 * C2C video loader nodes — probe info strip, H.264 preview, and a real player (L2.50): transport, a scrubber over the
 * whole clip, and in / out handles that write the node's frame selection (c2c_video_player.js).
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { measureRootContent, installResizeFloor } from "./c2c_ui/index.js";
import { setHidden } from "./_widget_visibility.js";
import {
    MARGIN,
    DEFAULT_ASPECT,
    buildDom,
    setEmpty,
    setLoading,
    renderChips,
    renderTimeline,
    attachPauseObservers,
    reservedHeight,
} from "./c2c_video_player_ui.js";
import {
    attachPlayer, holdPlayerWidth, playerButton, selectionFromWidgets, widgetsFromSelection,
} from "./c2c_video_player.js";
import { attachCrop } from "./c2c_video_crop.js";
import { afterRender, rebindNode, tagDomWidget } from "./_vue_dom_rebind.js";
import {
    RESETTABLE, SETTINGS, capEdit, capStep, describe, hints, rateNote, sizeEdit, snapCap, FORMAT_GRID,
} from "./_c2c_loader_hints.js";

const DEBOUNCE_MS = 350;
// L2.53 inputs (owner A14). They change what loads, so they refresh the probe; the preview (the whole clip) ignores
// them - kept out of its URL so moving the crop or a size never reloads the preview.
const LOAD_ONLY_WIDGETS = [
    "custom_width", "custom_height", "seconds_cap", "resolution", "divisible_by", "decoder", "audio_when_missing",
    "crop_x", "crop_y", "crop_w", "crop_h",
];
// L9.52: the handle settings go with the probe, so the node says what a run will add BEFORE it runs
const HANDLE_WIDGETS = [
    "handles", "handle_frames", "handle_target", "handle_side", "handle_grid", "handle_custom_step",
    "handle_custom_plus", "extra_handles",          // L9.56: whole model steps on top, shown in the details line
];
// L9.62: the colour settings go with the probe, so the node says what colour management will do BEFORE a run
const COLOUR_WIDGETS = ["colorspace_in", "working_space"];

const LOADERS = {
    LoadVideoC2C: {
        // Our server-side preview, not core's: core's upload preview plays the ORIGINAL file in the browser, and
        // Chrome cannot decode ProRes, DNxHR, HEVC 10-bit or FFV1/MKV - those showed no preview at all (L2.27,
        // measured; the server transcodes each in under a second). Core's own preview widget is hidden.
        hasPreview: true,
        hideCorePreview: true,
        mode: "upload",
        settingsUx: true,          // L2.53: grey real values, reset arrows, reset all, details (our nodes only)
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
        settingsUx: true,
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
const PREVIEW_FRAME_LIMIT = 2000;   // the preview route's cap; longer clips preview every k-th frame
const CROP_WIDGETS = ["crop_x", "crop_y", "crop_w", "crop_h"];   // fractions of the frame, set by the crop tool

function readCrop(node) {
    const [x, y, w, h] = CROP_WIDGETS.map((n) => Number(widgetValue(node, n)));
    return { x: Number.isFinite(x) ? x : 0, y: Number.isFinite(y) ? y : 0,
             w: Number.isFinite(w) ? w : 1, h: Number.isFinite(h) ? h : 1 };
}

/** The selection widgets of a loader: image folders use their own names. */
function selectionNames(cfg) {
    return cfg.mode === "images"
        ? { skip: "skip_first_images", cap: "image_load_cap", nth: "select_every_nth" }
        : { skip: "skip_first_frames", cap: "frame_load_cap", nth: "select_every_nth" };
}

/** The same request with no selection: the whole clip (after force_rate), optionally sampled every ``k`` frames. */
function domainParams(params, cfg, k = 1, maxFrames = null) {
    const d = new URLSearchParams(params.toString());
    const n = selectionNames(cfg);
    d.set(n.skip, "0");
    d.set(n.cap, "0");
    d.set(n.nth, String(Math.max(1, k)));
    for (const name of [...LOAD_ONLY_WIDGETS, ...HANDLE_WIDGETS, ...COLOUR_WIDGETS]) d.delete(name);   // the whole clip
    if (maxFrames) d.set("max_frames", String(maxFrames));
    return d;
}

function setWidget(node, name, value) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w || w.value === value) return;
    w.value = value;
    try { w.callback?.(value, app.canvas, node); } catch (_e) { /* a downstream hook's error is not ours */ }
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
    if (cfg.mode !== "images") {
        // L2.53: everything else a run uses, so the probe answers with exactly what will load
        for (const name of [...LOAD_ONLY_WIDGETS, ...HANDLE_WIDGETS, ...COLOUR_WIDGETS]) {
            const v = widgetValue(node, name);
            if (v !== undefined && v !== null) p.set(name, String(v));
        }
    }
    return p;
}

function mountLoaderWidget(node, cfg) {
    const ui = buildDom(cfg);
    let aspect = DEFAULT_ASPECT;
    let abortCtrl = null;
    let debounceTimer = null;
    let cleanupVideo = () => {};
    let domain = { frames: 1, fps: 24 };
    let previewSrc = "";
    let ux = null;

    // The player replaces the static selection bar: it shows the whole clip, the playhead and the selection, and its
    // handles write the selection widgets (skip / cap), which then refresh the probe through their callbacks.
    let player = null;
    if (cfg.hasPreview && ui.video) {
        ui.track.hidden = true;
        ui.labelLeft.parentElement.hidden = true;
        player = attachPlayer(ui.timeline, ui.video, {
            fullscreenTarget: ui.root,
            onSelect: (inF, outF) => {
                const n = selectionNames(cfg);
                const nth = Number(widgetValue(node, n.nth) ?? 1) || 1;
                const { skip, cap } = widgetsFromSelection(inF, outF, nth, domain.frames);
                // the handles set frames: a seconds cap would cut them short, so it goes back to "follow"
                const sec = node.widgets?.find((x) => x.name === "seconds_cap");
                if (sec && Number(sec.value) > 0) { sec.value = 0; scheduleRefresh(); }
                setWidget(node, n.skip, skip);
                setWidget(node, n.cap, cap);
                node.setDirtyCanvas?.(true, true);
            },
        });
    }

    // Crop tool (Load Video / Load Video Path): drawn over the preview, writes the four crop widgets, which it hides.
    let crop = null;
    if (player && ui.viewport && node.widgets?.some((w) => w.name === "crop_x")) {
        for (const name of CROP_WIDGETS) {
            const w = node.widgets.find((x) => x.name === name);
            if (w) setHidden(w, true);
        }
        crop = attachCrop(ui.viewport, ui.timeline, {
            makeButton: playerButton,
            onCommit: (r) => {
                const round = (v) => Math.round(v * 10000) / 10000;
                setWidget(node, "crop_x", round(r.x));
                setWidget(node, "crop_y", round(r.y));
                setWidget(node, "crop_w", round(r.w));
                setWidget(node, "crop_h", round(r.h));
                crop?.setRect(readCrop(node));          // show exactly what is stored
                node.setDirtyCanvas?.(true, true);
                fitNode();
            },
        });
    }

    const refresh = async () => {
        if (abortCtrl) abortCtrl.abort();
        abortCtrl = new AbortController();
        const params = collectParams(node, cfg);
        if (!params) {
            setEmpty(ui, cfg);
            ui.status.setText("", "default");
            if (cfg.hasPreview) ui.video?.removeAttribute("src");
            previewSrc = "";
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
            ui.root.style.setProperty("--c2c-aspect", String(aspect));      // the full-screen view keeps it

            renderTimeline(ui, probeJson);
            renderChips(ui.info, probeJson);
            ux?.update(probeJson);
            if (crop) {
                crop.setFrameSize(probeJson.width || 0, probeJson.height || 0);
                crop.setRect(readCrop(node));
            }
            ui.status.setText("", "default");

            if (cfg.hasPreview && ui.video && player) {
                // whole clip (after force_rate) for the scrubber; the exact count of this selection from the probe
                const dParams = domainParams(params, cfg);
                const dResp = await fetch(api.apiURL("/c2c/video/probe?" + dParams.toString()), { signal: abortCtrl.signal });
                const dJson = await dResp.json();
                if (!dResp.ok) throw new Error(dJson.error || "Could not read video information.");
                const frames = Math.max(1, dJson.selection?.count ?? dJson.frame_count ?? 1);
                const fps = dJson.selection?.fps || dJson.fps || 24;
                domain = { frames, fps };
                const k = Math.max(1, Math.ceil(frames / PREVIEW_FRAME_LIMIT));
                const pFrames = Math.ceil(frames / k);
                const pParams = domainParams(params, cfg, k, Math.min(PREVIEW_FRAME_LIMIT, pFrames));
                let pFps = fps / k;
                if (k > 1) {
                    const pResp = await fetch(api.apiURL("/c2c/video/probe?" + pParams.toString()), { signal: abortCtrl.signal });
                    const pJson = await pResp.json();
                    if (pResp.ok && pJson.selection?.fps) pFps = pJson.selection.fps;
                }
                player.setDomain({ frames, fps, step: k, previewFrames: pFrames, previewFps: pFps });
                const n = selectionNames(cfg);
                const nth = Number(widgetValue(node, n.nth) ?? 1) || 1;
                const selW = selectionFromWidgets(Number(widgetValue(node, n.skip) ?? 0), Number(widgetValue(node, n.cap) ?? 0),
                                                  nth, frames);
                const count = probeJson.selection?.count ?? selW.count;
                player.setSelection({ in: selW.in, out: selW.in + Math.max(0, count - 1) * nth, count, nth });

                const previewUrl = api.apiURL("/c2c/video/preview?" + pParams.toString());
                if (previewUrl !== previewSrc) {        // a new clip or rate; moving a handle never re-encodes
                    previewSrc = previewUrl;
                    ui.viewport.querySelector(".c2c-ui-empty")?.remove();
                    setLoading(ui, true, aspect);
                    ui.video.src = previewUrl;
                    ui.video.onloadeddata = () => setLoading(ui, false, aspect);
                    ui.video.onerror = () => {
                        setLoading(ui, false, aspect);
                        ui.status.setText("Could not play the preview.", "danger");
                    };
                } else {
                    setLoading(ui, false, aspect);
                }
            } else if (cfg.hasPreview && ui.video) {
                const previewUrl = api.apiURL("/c2c/video/preview?" + params.toString());
                ui.viewport.querySelector(".c2c-ui-empty")?.remove();
                setLoading(ui, true, aspect);
                ui.video.src = previewUrl;
                ui.video.onloadeddata = () => setLoading(ui, false, aspect);
                ui.video.onerror = () => {
                    setLoading(ui, false, aspect);
                    ui.status.setText("Could not play the preview.", "danger");
                };
            } else {
                setLoading(ui, false, aspect);
            }

            fitNode();
        } catch (err) {
            if (err.name === "AbortError") return;
            ux?.update(null);
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
    const getReserved = (w) => reservedHeight(w || nodeWidth(), cfg.hasPreview, aspect, ui);
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
    if (player) holdPlayerWidth(node, player);          // the transport on ONE line (L9.52)
    if (cfg.hideCorePreview) muteCorePreviewSize(node);
    tagDomWidget(widgetRef);       // Nodes 2.0 inside a new subgraph shows a stale copy; _vue_dom_rebind swaps ours in

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

    for (const name of [...PATH_WIDGETS, ...VIDEO_WIDGETS, ...IMAGE_WIDGETS, ...LOAD_ONLY_WIDGETS, ...HANDLE_WIDGETS,
                        ...COLOUR_WIDGETS]) {
        hookWidget(name);
    }
    // L2.53 settings UX: grey real values, reset arrows, reset all, details line, frame-grid stepping
    ux = cfg.settingsUx ? installSettingsUx(node, cfg, ui, scheduleRefresh, fitNode) : null;

    if (cfg.hasPreview && ui.video) {
        cleanupVideo = attachPauseObservers(node, ui.video);
    }

    // Removal is not always deletion: "Convert to subgraph" removes the node from its graph and adds the SAME node
    // to the subgraph (L2.50: the player and the preview vanished inside subgraphs in Nodes 2.0). So removal only
    // pauses and cancels work; if the node is still in no graph a moment later it was deleted, and everything is
    // released. Re-adding it refreshes the preview.
    let teardownTimer = null;
    let tornDown = false;
    const origRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        if (abortCtrl) abortCtrl.abort();
        if (debounceTimer) clearTimeout(debounceTimer);
        try { ui.video?.pause(); } catch (_e) { /* ignore */ }
        if (teardownTimer) clearTimeout(teardownTimer);
        teardownTimer = setTimeout(() => {
            teardownTimer = null;
            if (node.graph) return;                     // moved (into or out of a subgraph), not deleted
            tornDown = true;
            cleanupVideo();
            player?.destroy();
            crop?.destroy();
            uninstallFloor();
        }, 2000);
        return origRemoved?.apply(this, args);
    };
    const origAdded = node.onAdded;
    node.onAdded = function (...args) {
        const r = origAdded?.apply(this, args);
        if (teardownTimer) { clearTimeout(teardownTimer); teardownTimer = null; }
        if (!tornDown) {
            previewSrc = "";                            // reload the preview in its new place
            scheduleRefresh();
        }
        afterRender(() => rebindNode(node));
        return r;
    };

    setEmpty(ui, cfg);
    scheduleRefresh();

    return widgetRef;
}

// ── L2.53 settings UX (owner A14, from ComfyUI-4K-Video-Save-Load, MIT): grey real values, reset arrows, reset all,
// the details line, frame-grid stepping, a new clip resets the settings. Numbers come from the probe (the server
// builds the same handle a run builds), so the hints, the player's "N will load" and the run agree. Classic renderer:
// the grey value and the arrow sit beside the widget. Nodes 2.0 draws widgets itself, so there they sit in the panel.
const RESET_GLYPH = "↺";
const BUTTON_X = 54;         // a number widget steps on clicks within ~40 px of its edges: the arrow sits clear of that
const BUTTON_ROOM = 20;

function widgetValues(node) {
    const out = {};
    for (const w of node.widgets || []) if (w?.name) out[w.name] = w.value;
    return out;
}

function vueMode() {
    return !!globalThis.LiteGraph?.vueNodesMode;
}

function installSettingsUx(node, cfg, ui, scheduleRefresh, fitNode) {
    let probe = null;
    let hintMap = {};
    const W = (name) => node.widgets?.find((x) => x.name === name);

    // a quiet write: our own follow-up changes must not fire the user-edit side effects below
    const quiet = (name, value) => {
        const w = W(name);
        if (!w || w.value === value) return false;
        w.value = value;
        return true;
    };

    // ── the panel block: details lines, (Nodes 2.0) hint chips, reset all ──
    const box = document.createElement("div");
    box.className = "c2c-vloader__summary";
    const lines = document.createElement("div");
    lines.className = "c2c-vloader__summary-lines";
    const chips = document.createElement("div");
    chips.className = "c2c-vloader__summary-chips";
    const resetAll = document.createElement("button");
    resetAll.type = "button";
    resetAll.className = "c2c-vplayer__btn c2c-vloader__reset-all";
    resetAll.textContent = "reset all to defaults";
    resetAll.title = "Every setting back to its default; the clip stays";
    resetAll.addEventListener("click", (e) => { e.stopPropagation(); doResetAll(); });
    chips.appendChild(resetAll);
    box.append(lines, chips);
    ui.root.insertBefore(box, ui.status?.el ?? null);
    ui.summary = box;

    function reset(name) {
        if (name === "custom_width" || name === "custom_height") {
            node._c2cApplying = true;
            try {
                setWidget(node, "resolution", "custom");
                setWidget(node, "custom_width", 0);
                setWidget(node, "custom_height", 0);
            } finally { node._c2cApplying = false; }
        } else if (name in RESETTABLE) {
            node._c2cApplying = true;
            try { setWidget(node, name, RESETTABLE[name]); } finally { node._c2cApplying = false; }
        }
        scheduleRefresh();
        node.setDirtyCanvas?.(true, true);
    }

    function doResetAll() {
        const defaults = cfg.defaults || {};
        node._c2cApplying = true;
        try {
            for (const name of SETTINGS) {
                if (name in defaults && W(name)) setWidget(node, name, defaults[name]);
            }
        } finally { node._c2cApplying = false; }
        applyGrid();
        scheduleRefresh();
        node.setDirtyCanvas?.(true, true);
    }

    function renderBox() {
        const values = widgetValues(node);
        const text = describe(probe);
        const note = probe?.load ? rateNote(values.format, probe.load, values.select_every_nth) : "";
        if (note) text.push(note);
        const hp = probe?.load?.handles;                 // L9.52: what the run's handles add, before it runs
        if (hp?.note) text.push(`Handles: ${hp.note}${hp.error ? "" : " - Save Video (C2C) trims them"}`);
        const cm = probe?.cm;                            // L9.62: what colour management will do, before a run
        if (cm) text.push(colourLine(cm));
        lines.replaceChildren(...text.map((t) => {
            const d = document.createElement("div");
            d.textContent = t;
            return d;
        }));
        // Nodes 2.0: a chip per widget whose own value is not what loads, with its reset arrow
        for (const el of [...chips.querySelectorAll(".c2c-vloader__hint")]) el.remove();
        if (vueMode()) {
            for (const [name, h] of Object.entries(hintMap)) {
                if (!h.differs && !h.canReset) continue;
                const chip = document.createElement("span");
                chip.className = "c2c-vloader__chip c2c-vloader__hint";
                const v = values[name];
                chip.textContent = h.differs ? `${name} ${h.shown} ← ${v}` : `${name} ${v}`;
                const b = document.createElement("button");
                b.type = "button";
                b.className = "c2c-vloader__hint-reset";
                b.textContent = RESET_GLYPH;
                b.disabled = !h.canReset;
                b.title = h.canReset ? `Reset ${name}` : "Nothing to reset";
                b.addEventListener("click", (e) => { e.stopPropagation(); reset(name); });
                chip.appendChild(b);
                chips.insertBefore(chip, resetAll);
            }
        }
        fitNode();
    }

    // ── classic renderer: the grey value and the arrow beside the widget (the 4K pack's way) ──
    function decorate(w) {
        if (!w || w.__c2cDecorated || typeof w.drawTruncatingText !== "function") return;
        w.__c2cDecorated = true;
        const drawText = w.drawTruncatingText;
        w.drawTruncatingText = function (options) {
            const rightPadding = (options.rightPadding ?? 20) + BUTTON_ROOM;
            drawText.call(this, { ...options, rightPadding });
            if (!this.computedDisabled) drawExtras(options.ctx, this, options.width, rightPadding);
        };
        const prevDown = w.onPointerDown;
        w.onPointerDown = function (pointer, n, canvas) {
            const x = canvas.graph_mouse[0] - node.pos[0];
            if (Math.abs(x - ((this.width || node.size[0]) - BUTTON_X)) <= 10) {
                if (hintMap[this.name]?.canReset) pointer.onClick = () => reset(this.name);
                return true;
            }
            return prevDown ? prevDown.call(this, pointer, n, canvas) : false;
        };
    }

    function drawExtras(ctx, w, width, rightPadding) {
        const h = hintMap[w.name];
        if (!h) return;
        const y = w.y + w.height * 0.7;
        ctx.save();
        ctx.textAlign = "center";
        ctx.fillStyle = h.canReset ? "#aaa" : "#555";            // literal colours: var() paints black in a canvas
        ctx.fillText(RESET_GLYPH, width - BUTTON_X, y);
        if (h.differs) {
            const text = `${h.shown}←`;
            const margin = 15;
            const right = width - 2 * margin - rightPadding - ctx.measureText(w._displayValue ?? String(w.value)).width - 6;
            const labelEnd = 2 * margin + 10 + ctx.measureText(w.label || w.name).width;
            if (right - ctx.measureText(text).width > labelEnd) {
                ctx.textAlign = "right";
                ctx.fillStyle = "#8a8a8a";
                ctx.fillText(text, right, y);
            }
        }
        ctx.restore();
    }

    for (const name of Object.keys(RESETTABLE)) decorate(W(name));

    // ── side effects of the user's own edits (never of ours: node._c2cApplying) ──
    const after = (name, fn) => {
        const w = W(name);
        if (!w) return;
        const orig = w.callback;
        let prev = w.value;
        w.callback = function (...args) {
            const before = prev;
            prev = w.value;
            const r = orig?.apply(this, args);
            if (!node._c2cApplying && !app.configuringGraph) {
                try { fn(w, before); } catch (_e) { /* a hint never breaks the widget */ }
            }
            prev = w.value;
            return r;
        };
        // a loaded workflow sets values without callbacks: re-read the starting point then
        const origConfigure = node.onConfigure;
        node.onConfigure = function (...a) { const r = origConfigure?.apply(this, a); prev = w.value; return r; };
    };

    function applyGrid() {
        const w = W("frame_load_cap");
        if (!w) return;
        const fmt = widgetValue(node, "format") ?? "None";
        w.options = w.options || {};
        w.options.step2 = capStep(fmt);
        const grid = FORMAT_GRID[fmt];
        if (grid && Number(w.value) > 0) w.value = snapCap(w.value, -1, grid);   // nearest legal count
    }

    after("frame_load_cap", (w, before) => {
        const grid = FORMAT_GRID[widgetValue(node, "format") ?? "None"];
        if (grid) w.value = snapCap(w.value, before, grid);
        const other = capEdit("frame_load_cap", w.value);
        if (other) quiet("seconds_cap", other.seconds_cap);
    });
    after("seconds_cap", (w) => {
        const other = capEdit("seconds_cap", w.value);
        if (other) quiet("frame_load_cap", other.frame_load_cap);
    });
    for (const name of ["custom_width", "custom_height"]) {
        after(name, () => {
            const change = sizeEdit(name, widgetValues(node));
            if (change) for (const [k, v] of Object.entries(change)) quiet(k, v);
        });
    }
    after("format", () => applyGrid());
    if (cfg.mode === "upload") {
        // picking, uploading or dropping a new clip loads it with default settings; opening a workflow does not
        after("video", (w, before) => { if (before !== undefined && before !== w.value) doResetAll(); });
    }
    applyGrid();

    return {
        update(p) {
            probe = p;
            node._c2cColourPlan = p?.cm ?? null;         // L9.62: Save Video (C2C) reads it to say what it reverses
            for (const g of new Set([node.graph, app.rootGraph ?? app.graph])) {
                for (const s of g?.nodes || []) if (s.type === "SaveVideoC2C") s._c2cSave?.refreshLink?.();
            }
            const values = widgetValues(node);
            hintMap = hints(values, p?.load ?? null);
            // outside "custom" the size fields show the size the clip loads at (written quietly: no switch to custom)
            if (p?.load && (values.resolution ?? "custom") !== "custom") {
                quiet("custom_width", p.load.width);
                quiet("custom_height", p.load.height);
                hintMap = hints(widgetValues(node), p.load);
            }
            renderBox();
            node.setDirtyCanvas?.(true, true);
        },
    };
}

/** Hide core's upload preview (a <video> of the original file) so the node shows one preview, ours. Core adds it
 *  with the upload widget, possibly after onNodeCreated, so this runs again a few times and on every refresh. */
/**
 * Core's video upload adds its own "video-preview" DOM widget (useNodeVideo) whose computeLayoutSize min width is fitted
 * from the clip's aspect AND the node's current width - while it exists, a wider node asks for a wider minimum, and the
 * frontend resized new loaders to that (measured L9.52: 399 -> 503 px in Nodes 2.0). We hide that preview anyway, so its
 * size never counts: core assigns computeLayoutSize right after addDOMWidget returns, hence an accessor that keeps ours.
 */
function muteCorePreviewSize(node) {
    if (node.__c2cMutedCorePreview) return;
    node.__c2cMutedCorePreview = true;
    const zero = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
    const mute = (w) => {
        try {
            Object.defineProperty(w, "computeLayoutSize", { configurable: true, get: () => zero, set: () => {} });
        } catch (_e) { /* a sealed widget keeps core's size */ }
    };
    for (const w of node.widgets || []) if (w?.name === "video-preview") mute(w);
    const add = node.addDOMWidget;
    if (typeof add !== "function") return;
    node.addDOMWidget = function (name, ...rest) {
        const w = add.call(this, name, ...rest);
        if (name === "video-preview" && w) mute(w);
        return w;
    };
}

function hideCorePreview(node) {
    // Nodes 2.0 draws core's player from the node-output store, not from a widget: scope a CSS rule to this
    // node's root (Vue leaves attributes it does not render alone, and every refresh re-applies it).
    if (globalThis.LiteGraph?.vueNodesMode) {
        if (!document.getElementById("c2c-no-core-video-preview")) {
            const s = document.createElement("style");
            s.id = "c2c-no-core-video-preview";
            // frontend 1.53 draws the upload's preview in an .image-preview panel ("Image failed to load" for a
            // video); older releases used .video-preview. A loader is not an output node, so neither shows outputs.
            // :has() also covers a node view that never got the attribute (a new subgraph instance, L2.50)
            s.textContent = "[data-c2c-no-core-preview] .video-preview, [data-c2c-no-core-preview] .image-preview, "
                + ".lg-node:has(.c2c-vloader) .video-preview, .lg-node:has(.c2c-vloader) .image-preview "
                + "{ display: none !important; }";
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

/** L9.62: one line for the details box - the space, what said so, and what happens to it. */
function colourLine(cm) {
    if (cm.error) return `Colour: ${cm.error}`;
    if (!cm.source) return `Colour: unknown - ${(cm.notes || []).slice(-1)[0] || "pick colorspace_in"}`;
    const from = cm.mode === "manual" ? "set by hand" : `${cm.detected_from}${cm.confidence ? ", " + cm.confidence : ""}`;
    const what = cm.transform ? `-> ${cm.working} (${cm.transform.kind === "view" ? "view " + cm.transform.view : "conversion"})`
        : "- not converted";
    const note = (cm.notes || []).find((n) => !/^OpenColorIO .* works/.test(n));
    return `Colour: ${cm.source} (${from}) ${what}${note ? " · " + note : ""} · config ${cm.config_source || cm.config}`;
}

// L9.53 colour management (owner 2026-10-25 / 26). The Python default is "off (as before)", so saved workflows and
// API prompts keep their pixels; a node made now starts on auto. working_space only matters when colour is on.
const CS_OFF = "off (as before)";
const CS_AUTO = "auto (detect)";

function setupColour(node, fresh) {
    const csW = node.widgets?.find((w) => w.name === "colorspace_in");
    const wsW = node.widgets?.find((w) => w.name === "working_space");
    if (!csW) return;
    const sync = () => {
        // a value this list does not hold (a shifted older workflow) means what the loader always did
        if (!csW.options?.values?.includes(csW.value)) csW.value = CS_OFF;
        if (wsW) setHidden(wsW, csW.value === CS_OFF);
    };
    const prev = csW.callback;
    csW.callback = function (...a) {
        const r = prev?.apply(this, a);
        sync();
        return r;
    };
    setTimeout(() => {                                // a loaded / pasted node's values are restored by then
        // new node only: one configured from saved data (a workflow, a paste) keeps what it had
        if (fresh && !node._c2cColourConfigured && csW.value === CS_OFF) csW.value = CS_AUTO;
        sync();
    }, 0);
}

if (!globalThis.__c2cVideoLoadersExt) {
    globalThis.__c2cVideoLoadersExt = true;
    // L9.62 (owner 2026-10-25: "$OCIO, else setting, else the built-in ACES studio config"): changing one makes the
    // server load the config again; the colour-space lists follow after a page reload (R)
    // the new value travels with the request: the settings file may be written a moment after onChange
    const reloadColour = (id) => (value) => {
        api.fetchApi("/c2c/video/cm/reload", { method: "POST", body: JSON.stringify({ id, value }) }).catch(() => {});
    };
    app.registerExtension({
        name: "C2C.VideoLoaders",
        settings: [
            { id: "c2c.video.cm.ocioConfig", name: "OCIO config file (used when $OCIO is not set)", type: "text",
              defaultValue: "", category: ["c2c", "Video", "Colour Management", "config"], onChange: reloadColour("c2c.video.cm.ocioConfig"),
              tooltip: "Path to your own .ocio config (e.g. the studio's Nuke config). $OCIO wins when it is set; "
                + "empty = the built-in config below. Reload the page (R) so the colour-space lists follow." },
            { id: "c2c.video.cm.builtinConfig", name: "Built-in OCIO config", type: "combo",
              options: ["studio (latest)", "cg (latest)", "studio ACES 1.3", "cg ACES 1.3"], defaultValue: "studio (latest)",
              category: ["c2c", "Video", "Colour Management", "builtin"], onChange: reloadColour("c2c.video.cm.builtinConfig"),
              tooltip: "The config OpenColorIO ships (nothing to download), used without $OCIO or a config file." },
            { id: "c2c.video.cm.view", name: "Display view (empty = the config's default)", type: "text",
              defaultValue: "", category: ["c2c", "Video", "Colour Management", "view"], onChange: reloadColour("c2c.video.cm.view"),
              tooltip: "The view Load Video uses to bring scene-referred footage to sRGB display (and Save reverses), "
                + "e.g. 'ACES 2.0 - SDR 100 nits (Rec.709)' or 'Un-tone-mapped'. A name the config lacks falls back "
                + "to its default, and the node says so." },
        ],
        async beforeRegisterNodeDef(nodeType, nodeData, _app) {
            const cfg = LOADERS[nodeData.name];
            if (!cfg) return;
            // the node's own defaults, for "reset all to defaults" (combo without a default: its first choice)
            cfg.defaults = {};
            for (const group of [nodeData.input?.required, nodeData.input?.optional]) {
                for (const [name, spec] of Object.entries(group || {})) {
                    const opts = Array.isArray(spec) ? spec[1] : null;
                    if (opts && "default" in opts) cfg.defaults[name] = opts.default;
                    else if (Array.isArray(spec?.[0]) && spec[0].length) cfg.defaults[name] = spec[0][0];
                }
            }
            if ("colorspace_in" in cfg.defaults) cfg.defaults.colorspace_in = CS_AUTO;   // what a new node starts on
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...args) {
                const r = onCreated?.apply(this, args);
                setupLoaderNode(this, cfg);
                setupColour(this, !app.configuringGraph);
                return r;
            };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (...args) {
                this._c2cColourConfigured = true;              // saved data: no new-node colour default
                return onConfigure?.apply(this, args);
            };
        },
    });
}
