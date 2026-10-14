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
import { attachPlayer, selectionFromWidgets, widgetsFromSelection } from "./c2c_video_player.js";
import { afterRender, rebindNode, tagDomWidget } from "./_vue_dom_rebind.js";

const DEBOUNCE_MS = 350;

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
const PREVIEW_FRAME_LIMIT = 2000;   // the preview route's cap; longer clips preview every k-th frame

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

    // The player replaces the static selection bar: it shows the whole clip, the playhead and the selection, and its
    // handles write the selection widgets (skip / cap), which then refresh the probe through their callbacks.
    let player = null;
    if (cfg.hasPreview && ui.video) {
        ui.track.hidden = true;
        ui.labelLeft.parentElement.hidden = true;
        player = attachPlayer(ui.timeline, ui.video, {
            onSelect: (inF, outF) => {
                const n = selectionNames(cfg);
                const nth = Number(widgetValue(node, n.nth) ?? 1) || 1;
                const { skip, cap } = widgetsFromSelection(inF, outF, nth, domain.frames);
                setWidget(node, n.skip, skip);
                setWidget(node, n.cap, cap);
                node.setDirtyCanvas?.(true, true);
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

            renderTimeline(ui, probeJson);
            renderChips(ui.info, probeJson);
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

    for (const name of [...PATH_WIDGETS, ...VIDEO_WIDGETS, ...IMAGE_WIDGETS]) {
        hookWidget(name);
    }

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

/** Hide core's upload preview (a <video> of the original file) so the node shows one preview, ours. Core adds it
 *  with the upload widget, possibly after onNodeCreated, so this runs again a few times and on every refresh. */
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
