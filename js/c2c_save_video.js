/**
 * Save Video (C2C) — player, timeline scrub, format chips, naming setting.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { measureRootContent, installResizeFloor } from "./c2c_ui/index.js";
import { setHidden, vueSyncNodeWidgets } from "./_widget_visibility.js";
import {
    MARGIN,
    DEFAULT_ASPECT,
    buildDom,
    setLoading,
    renderChips,
    renderPlaybackTimeline,
    attachPauseObservers,
    reservedHeight,
} from "./c2c_video_player_ui.js";

const NODE = "SaveVideoC2C";
const SETTING_NAMING = "c2c.saveVideo.naming";

const NO_QUALITY_FORMATS = new Set([
    "MOV DNxHR HQ",
    "MOV DNxHR HQX",
    "MOV DNxHR 444",
    "MKV FFV1 (lossless)",
    "PNG sequence 16-bit",
    "EXR sequence (half)",
    "EXR sequence (float)",
]);

function formatBase(label) {
    return String(label || "").replace(/\s+\[unavailable\]$/, "").trim();
}

function formatHasQuality(label) {
    return !NO_QUALITY_FORMATS.has(formatBase(label));
}

function viewUrl(meta) {
    if (!meta?.filename) return "";
    return api.apiURL(
        `/view?filename=${encodeURIComponent(meta.filename)}` +
        `&type=${encodeURIComponent(meta.type || "output")}` +
        `&subfolder=${encodeURIComponent(meta.subfolder || "")}`,
    );
}

function resolveNaming(node) {
    const namingW = node.widgets?.find((w) => w.name === "naming");
    const resolvedW = node.widgets?.find((w) => w.name === "naming_resolved");
    if (!resolvedW) return;
    const naming = namingW?.value ?? "default";
    if (naming === "default") {
        try {
            resolvedW.value = app.ui?.settings?.getSettingValue?.(SETTING_NAMING, "ComfyUI counter")
                ?? "ComfyUI counter";
        } catch {
            resolvedW.value = "ComfyUI counter";
        }
    } else {
        resolvedW.value = naming;
    }
}

function applyQualityVisibility(node) {
    const fmtW = node.widgets?.find((w) => w.name === "format");
    const qW = node.widgets?.find((w) => w.name === "quality");
    if (!qW) return;
    const show = formatHasQuality(fmtW?.value);
    setHidden(qW, !show);
    vueSyncNodeWidgets(node);
    const sz = node.computeSize?.();
    if (sz) node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.setDirtyCanvas?.(true, true);
}

function hookWidget(node, name, fn) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w || w.__c2cSvHooked) return;
    w.__c2cSvHooked = true;
    const orig = w.callback;
    w.callback = function (...args) {
        const r = orig?.apply(this, args);
        fn?.();
        return r;
    };
    const prevBq = w.beforeQueued;
    w.beforeQueued = function (...args) {
        fn?.();
        return prevBq?.apply(this, args);
    };
}

function mountSaveWidget(node) {
    const cfg = {
        hasPreview: true,
        emptyTitle: "Save Video (C2C)",
        emptyHint: "Queue to write and preview the saved file.",
    };
    const ui = buildDom(cfg);
    let aspect = DEFAULT_ASPECT;
    let cleanupVideo = () => {};
    let labelUpdater = null;

    const nodeWidth = () => node.size?.[0] || 340;
    const getReserved = (w) => reservedHeight(w || nodeWidth(), true, aspect, ui);

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

    if (ui.video) {
        cleanupVideo = attachPauseObservers(node, ui.video);
        ui.track.style.cursor = "pointer";
        ui.track.addEventListener("click", (ev) => {
            if (!ui.video?.duration) return;
            const rect = ui.track.getBoundingClientRect();
            const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
            ui.video.currentTime = frac * ui.video.duration;
        });
        ui.video.addEventListener("timeupdate", () => labelUpdater?.());
    }

    const showResult = (data) => {
        if (!data) return;
        aspect = (data.width && data.height) ? data.width / data.height : DEFAULT_ASPECT;
        if (ui.viewport) ui.viewport.style.aspectRatio = String(aspect);
        renderChips(ui.info, data);
        const prev = data.preview || data;
        const url = viewUrl(prev);
        if (ui.video && url) {
            setLoading(ui, true, aspect);
            ui.viewport?.querySelector(".c2c-ui-empty")?.remove();
            ui.video.src = url;
            ui.video.onloadeddata = () => {
                setLoading(ui, false, aspect);
                labelUpdater = renderPlaybackTimeline(ui, data, ui.video);
                labelUpdater();
            };
            ui.video.onerror = () => {
                setLoading(ui, false, aspect);
                ui.status.setText("Could not play the preview.", "danger");
            };
            try { ui.video.play(); } catch { /* autoplay blocked */ }
        }
        const warns = data.warnings;
        if (Array.isArray(warns) && warns.length) {
            ui.status.setText(warns.join(" "), "warn");
        } else {
            ui.status.setText("", "default");
        }
        fitNode();
    };

    node._c2cSaveShowResult = showResult;

    const origRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        cleanupVideo();
        uninstallFloor();
        return origRemoved?.apply(this, args);
    };

    return widgetRef;
}

if (!globalThis.__c2cSaveVideoExt) {
    globalThis.__c2cSaveVideoExt = true;
    app.registerExtension({
        name: "C2C.SaveVideo",
        settings: [{
            id: SETTING_NAMING,
            name: "Save Video naming mode",
            type: "combo",
            options: ["ComfyUI counter", "Folder Version"],
            defaultValue: "ComfyUI counter",
            category: ["c2c", "Video", "Save Video"],
            tooltip: "Default naming for Save Video (C2C) when the node's naming input is 'default'.",
        }],
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== NODE) return;

            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...args) {
                const r = onCreated?.apply(this, args);
                if (!this._c2cSaveMounted) {
                    this._c2cSaveMounted = true;
                    mountSaveWidget(this);
                    const nr = this.widgets?.find((w) => w.name === "naming_resolved");
                    if (nr) setHidden(nr, true);
                    hookWidget(this, "format", () => applyQualityVisibility(this));
                    hookWidget(this, "naming", () => resolveNaming(this));
                    const node = this;
                    for (const name of ["naming", "format"]) {
                        const w = this.widgets?.find((x) => x.name === name);
                        if (w) {
                            const prev = w.beforeQueued;
                            w.beforeQueued = function (...a) {
                                resolveNaming(node);   // `this` here is the widget, not the node
                                return prev?.apply(this, a);
                            };
                        }
                    }
                    setTimeout(() => {
                        applyQualityVisibility(this);
                        resolveNaming(this);
                    }, 0);
                }
                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecuted?.apply(this, arguments);
                const data = message?.c2c_save_video?.[0];
                this._c2cSaveShowResult?.(data);
                return r;
            };
        },
    });
}
