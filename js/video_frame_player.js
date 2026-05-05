import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * VideoFramePlayerMEC - frame scrubber + drag-crop + aspect-lock overlay.
 *
 *  - Scrubber: drag/click timeline to scrub; arrows = step frames; space = play.
 *  - Crop: when crop_enabled is true, a drag-rectangle is drawn on the
 *    preview. 4 corner + 4 edge handles resize; drag inside to move.
 *    Aspect-lock kicks in when aspect_ratio != 'free'.
 *  - All crop math is clamped to the preview rect so the rectangle CAN'T
 *    scroll outside the canvas border.
 *
 * UX inspiration (no code copied - clean-room implementation, see NOTICE.md):
 *   - Olm DragCrop by Olli Sorjonen (source-available, not OSS):
 *     https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop
 *     popularised the in-node drag-rectangle crop UX. Its licence
 *     prohibits redistribution, so its code is NOT used here; this
 *     overlay is written independently using standard HTML5 canvas
 *     drag-handle patterns (8-handle hit-test, aspect-anchor opposite
 *     corner, dim overlay, rule-of-thirds guides).
 *   - WhatDreamsCost-ComfyUI 'Load Video UI' by Jonathan Watkins (GPL-3.0):
 *     https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI
 *     inspired the trim/timeline+resize widget layout. No GPL source
 *     was copied (would be incompatible with this pack's MIT licence).
 */

const TL_H = 36;       // timeline bar height
const TICK_H = 6;
const HANDLE_R = 7;
const PLAY_BTN_W = 28;
const STATUS_H = 18;
const CROP_HANDLE_S = 8;       // crop handle hit size (half-edge)
const CROP_LINE_W = 1.5;
const TRIM_HANDLE_W = 6;       // trim marker hit width on timeline

const ASPECT_VALUES = {
    "free":     null,
    "original": null,    // dynamic -> srcW/srcH
    "1:1":      1.0,
    "4:3":      4.0 / 3.0,
    "3:4":      3.0 / 4.0,
    "16:9":     16.0 / 9.0,
    "9:16":     9.0 / 16.0,
    "2:1":      2.0,
    "21:9":     21.0 / 9.0,
    "custom":   null,    // dynamic -> custom_aspect_w/h
};

function getAspect(node, S) {
    const wAR = node.widgets?.find(w => w.name === "aspect_ratio");
    const ar = wAR?.value ?? "free";
    if (ar === "free") return null;
    if (ar === "original") {
        return (S.width > 0 && S.height > 0) ? S.width / S.height : null;
    }
    if (ar === "custom") {
        const cw = node.widgets?.find(w => w.name === "custom_aspect_w")?.value ?? 16;
        const ch = node.widgets?.find(w => w.name === "custom_aspect_h")?.value ?? 9;
        if (cw > 0 && ch > 0) return cw / ch;
        return null;
    }
    return ASPECT_VALUES[ar] ?? null;
}

function snapToAspect(cx, cy, cw, ch, aspect) {
    if (!aspect || aspect <= 0) return [cx, cy, cw, ch];
    const cur = cw / Math.max(ch, 1e-6);
    const centreX = cx + cw / 2;
    const centreY = cy + ch / 2;
    if (cur > aspect) cw = ch * aspect;
    else              ch = cw / aspect;
    cx = Math.max(0, Math.min(1 - cw, centreX - cw / 2));
    cy = Math.max(0, Math.min(1 - ch, centreY - ch / 2));
    return [cx, cy, cw, ch];
}

function clamp01Rect(cx, cy, cw, ch) {
    cw = Math.max(0.001, Math.min(1, cw));
    ch = Math.max(0.001, Math.min(1, ch));
    cx = Math.max(0, Math.min(1 - cw, cx));
    cy = Math.max(0, Math.min(1 - ch, cy));
    return [cx, cy, cw, ch];
}

app.registerExtension({
    name: "MEC.VideoFramePlayer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VideoFramePlayerMEC") return;

        const _created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _created?.apply(this, arguments);
            const node = this;

            const el = document.createElement("div");
            el.style.cssText =
                "position:relative;width:calc(100% - 12px);min-height:240px;margin:2px 6px 16px 6px;background:#0c0c0c;pointer-events:auto;" +
                "border-radius:4px;overflow:hidden;";   // overflow:hidden = no scroll-out
            el.setAttribute("role", "group");
            el.setAttribute("aria-label",
                "Video frame player. Drag timeline to scrub. Drag rectangle to crop. " +
                "Arrow keys step frames, space toggles play.");

            const cvs = document.createElement("canvas");
            cvs.style.cssText = "display:block;width:100%;cursor:default;";
            cvs.setAttribute("tabindex", "0");
            el.appendChild(cvs);

            const S = {
                cvs,
                ctx: cvs.getContext("2d"),
                el,
                images: [],
                frameCount: 0,
                idx: 0,
                width: 0,
                height: 0,
                playing: false,
                fps: 24,
                loopMode: "loop",
                _ppDir: 1,         // ping-pong direction
                trimStart: 0,
                trimEnd: 0,        // resolved later
                _rafId: null,
                _lastT: 0,
                drag: null,        // null | "tl" | "crop-move" | "crop-N/S/E/W/NE/NW/SE/SW"
                _dragStart: null,  // { mx, my, cx, cy, cw, ch }
                _hover: null,
                // current preview rect (px on canvas) updated each render
                preview: { x: 0, y: 0, w: 0, h: 0 },
            };
            node._S = S;

            const canvasXY = (e) => {
                const r = cvs.getBoundingClientRect();
                return [
                    (e.clientX - r.left) * (cvs.width / (r.width || 1)),
                    (e.clientY - r.top) * (cvs.height / (r.height || 1)),
                ];
            };

            const setIdx = (i) => {
                if (S.frameCount <= 0) return;
                const lo = Math.max(0, S.trimStart | 0);
                const hi = Math.min(S.frameCount - 1, S.trimEnd | 0);
                S.idx = Math.max(lo, Math.min(hi, Math.round(i)));
                const w = node.widgets?.find((w) => w.name === "frame_index");
                if (w) {
                    w.value = S.idx;
                    if (w.callback) try { w.callback(S.idx); } catch (_) {}
                }
                node._render();
                app.graph.setDirtyCanvas(true);
            };

            // trim widgets helpers
            const getTrim = () => {
                const fs = Number(node.widgets?.find(w => w.name === "frame_start")?.value ?? 0);
                let fe = Number(node.widgets?.find(w => w.name === "frame_end")?.value ?? -1);
                const total = Math.max(1, S.frameCount);
                const lo = Math.max(0, Math.min(total - 1, fs | 0));
                const hi = (fe < 0) ? (total - 1) : Math.max(lo, Math.min(total - 1, fe | 0));
                return [lo, hi];
            };
            const setTrim = (lo, hi) => {
                const total = Math.max(1, S.frameCount);
                lo = Math.max(0, Math.min(total - 1, lo | 0));
                hi = Math.max(lo, Math.min(total - 1, hi | 0));
                const wS = node.widgets?.find(w => w.name === "frame_start");
                const wE = node.widgets?.find(w => w.name === "frame_end");
                if (wS) { wS.value = lo; try { wS.callback?.(lo); } catch (_) {} }
                if (wE) { wE.value = hi; try { wE.callback?.(hi); } catch (_) {} }
                S.trimStart = lo;
                S.trimEnd = hi;
                if (S.idx < lo || S.idx > hi) setIdx(Math.max(lo, Math.min(hi, S.idx)));
            };
            node._setTrim = setTrim;

            // Read crop widgets (normalized [0..1])
            const getCropNorm = () => {
                const get = (n, def) => node.widgets?.find(w => w.name === n)?.value ?? def;
                return [
                    Number(get("crop_x", 0)),
                    Number(get("crop_y", 0)),
                    Number(get("crop_w", 1)),
                    Number(get("crop_h", 1)),
                ];
            };
            const setCropNorm = (cx, cy, cw, ch) => {
                [cx, cy, cw, ch] = clamp01Rect(cx, cy, cw, ch);
                const set = (n, v) => {
                    const w = node.widgets?.find(w => w.name === n);
                    if (!w) return;
                    w.value = +v.toFixed(4);
                    if (w.callback) try { w.callback(w.value); } catch (_) {}
                };
                set("crop_x", cx); set("crop_y", cy);
                set("crop_w", cw); set("crop_h", ch);
            };
            const isCropEnabled = () =>
                !!(node.widgets?.find(w => w.name === "crop_enabled")?.value);
            const isCropLocked = () =>
                !!(node.widgets?.find(w => w.name === "crop_locked")?.value);

            node._setCropNorm = setCropNorm;
            node._getCropNorm = getCropNorm;
            node._isCropEnabled = () => isCropEnabled();
            node._isCropLocked = () => isCropLocked();

            // Crop pixel rect on canvas (intersects preview rect)
            const cropPxRect = () => {
                const p = S.preview;
                if (p.w <= 0 || p.h <= 0) return null;
                const [cx, cy, cw, ch] = getCropNorm();
                return {
                    x: p.x + cx * p.w,
                    y: p.y + cy * p.h,
                    w: cw * p.w,
                    h: ch * p.h,
                };
            };

            // Hit-test crop handles. Returns drag mode or null.
            const handleHit = (mx, my) => {
                if (!isCropEnabled() || isCropLocked()) return null;
                const cr = cropPxRect();
                if (!cr) return null;
                const hs = CROP_HANDLE_S;
                const corners = [
                    ["NW", cr.x,         cr.y],
                    ["NE", cr.x + cr.w,  cr.y],
                    ["SW", cr.x,         cr.y + cr.h],
                    ["SE", cr.x + cr.w,  cr.y + cr.h],
                ];
                for (const [k, hx, hy] of corners) {
                    if (Math.abs(mx - hx) <= hs && Math.abs(my - hy) <= hs)
                        return "crop-" + k;
                }
                const edges = [
                    ["N", cr.x + cr.w / 2, cr.y],
                    ["S", cr.x + cr.w / 2, cr.y + cr.h],
                    ["W", cr.x,            cr.y + cr.h / 2],
                    ["E", cr.x + cr.w,     cr.y + cr.h / 2],
                ];
                for (const [k, hx, hy] of edges) {
                    if (Math.abs(mx - hx) <= hs && Math.abs(my - hy) <= hs)
                        return "crop-" + k;
                }
                if (mx >= cr.x && mx <= cr.x + cr.w &&
                    my >= cr.y && my <= cr.y + cr.h) return "crop-move";
                return null;
            };

            // Update crop based on drag
            const applyCropDrag = (mode, mx, my) => {
                const ds = S._dragStart;
                if (!ds) return;
                const p = S.preview;
                if (p.w <= 0 || p.h <= 0) return;
                const dx = (mx - ds.mx) / p.w;
                const dy = (my - ds.my) / p.h;
                let [cx, cy, cw, ch] = [ds.cx, ds.cy, ds.cw, ds.ch];

                if (mode === "crop-move") {
                    cx = ds.cx + dx;
                    cy = ds.cy + dy;
                } else {
                    // edge / corner resize
                    let nx = cx, ny = cy, nx2 = cx + cw, ny2 = cy + ch;
                    if (mode.includes("W")) nx = ds.cx + dx;
                    if (mode.includes("E")) nx2 = ds.cx + ds.cw + dx;
                    if (mode.includes("N")) ny = ds.cy + dy;
                    if (mode.includes("S")) ny2 = ds.cy + ds.ch + dy;
                    if (nx2 < nx + 0.005) {
                        if (mode.includes("W")) nx = nx2 - 0.005;
                        else nx2 = nx + 0.005;
                    }
                    if (ny2 < ny + 0.005) {
                        if (mode.includes("N")) ny = ny2 - 0.005;
                        else ny2 = ny + 0.005;
                    }
                    cx = nx; cy = ny; cw = nx2 - nx; ch = ny2 - ny;

                    const aspect = getAspect(node, S);
                    if (aspect && aspect > 0 && mode !== "crop-move") {
                        // Anchor opposite corner for aspect snap
                        const ax = mode.includes("W") ? ds.cx + ds.cw : ds.cx;
                        const ay = mode.includes("N") ? ds.cy + ds.ch : ds.cy;
                        let aw = cw, ah = ch;
                        const cur = aw / Math.max(ah, 1e-6);
                        if (cur > aspect) aw = ah * aspect;
                        else              ah = aw / aspect;
                        let nx_  = mode.includes("W") ? ax - aw : ax;
                        let ny_  = mode.includes("N") ? ay - ah : ay;
                        cx = nx_; cy = ny_; cw = aw; ch = ah;
                    }
                }
                setCropNorm(cx, cy, cw, ch);
                node._render();
            };

            cvs.addEventListener("pointerdown", (e) => {
                cvs.focus();
                const [mx, my] = canvasXY(e);
                // crop handle / drag inside crop
                const hit = handleHit(mx, my);
                if (hit) {
                    cvs.setPointerCapture(e.pointerId);
                    S.drag = hit;
                    const [cx, cy, cw, ch] = getCropNorm();
                    S._dragStart = { mx, my, cx, cy, cw, ch };
                    return;
                }
                const r = tlRect();
                // Play/pause
                if (mx >= r.playBtnX && mx <= r.playBtnX + r.playBtnW &&
                    my >= r.y && my <= r.y + r.h) {
                    S.playing = !S.playing;
                    if (S.playing) startPlay(); else stopPlay();
                    node._render();
                    return;
                }
                // Trim handles (priority over scrub)
                if (my >= r.y && my <= r.y + r.h && mx >= r.barX) {
                    const [lo, hi] = getTrim();
                    const t0 = S.frameCount > 1 ? lo / (S.frameCount - 1) : 0;
                    const t1 = S.frameCount > 1 ? hi / (S.frameCount - 1) : 1;
                    const x0 = r.barX + t0 * r.barW;
                    const x1 = r.barX + t1 * r.barW;
                    if (Math.abs(mx - x0) <= TRIM_HANDLE_W) {
                        cvs.setPointerCapture(e.pointerId); S.drag = "trim-start"; return;
                    }
                    if (Math.abs(mx - x1) <= TRIM_HANDLE_W) {
                        cvs.setPointerCapture(e.pointerId); S.drag = "trim-end"; return;
                    }
                }
                // Timeline scrub
                if (my >= r.y && my <= r.y + r.h && mx >= r.barX) {
                    cvs.setPointerCapture(e.pointerId);
                    S.drag = "tl";
                    const t = (mx - r.barX) / Math.max(1, r.barW);
                    setIdx(t * (S.frameCount - 1));
                }
            });

            cvs.addEventListener("pointermove", (e) => {
                const [mx, my] = canvasXY(e);
                if (S.drag === "tl") {
                    const r = tlRect();
                    const t = (mx - r.barX) / Math.max(1, r.barW);
                    setIdx(t * (S.frameCount - 1));
                    return;
                }
                if (S.drag === "trim-start" || S.drag === "trim-end") {
                    const r = tlRect();
                    const t = (mx - r.barX) / Math.max(1, r.barW);
                    const f = Math.max(0, Math.min(S.frameCount - 1, Math.round(t * (S.frameCount - 1))));
                    const [lo, hi] = getTrim();
                    if (S.drag === "trim-start") setTrim(Math.min(f, hi), hi);
                    else                          setTrim(lo, Math.max(f, lo));
                    node._render();
                    return;
                }
                if (S.drag && S.drag.startsWith("crop")) {
                    applyCropDrag(S.drag, mx, my);
                    return;
                }
                // hover for cursor
                const hit = handleHit(mx, my);
                let cur = "default";
                if (isCropEnabled() && isCropLocked()) cur = "not-allowed";
                else if (hit === "crop-move") cur = "move";
                else if (hit) {
                    const m = hit.replace("crop-", "");
                    cur = ({
                        N: "ns-resize", S: "ns-resize", E: "ew-resize", W: "ew-resize",
                        NE: "nesw-resize", SW: "nesw-resize",
                        NW: "nwse-resize", SE: "nwse-resize",
                    })[m] || "default";
                } else {
                    // trim-handle hover
                    const r = tlRect();
                    if (my >= r.y && my <= r.y + r.h) {
                        const [lo, hi] = getTrim();
                        const t0 = S.frameCount > 1 ? lo / (S.frameCount - 1) : 0;
                        const t1 = S.frameCount > 1 ? hi / (S.frameCount - 1) : 1;
                        const x0 = r.barX + t0 * r.barW;
                        const x1 = r.barX + t1 * r.barW;
                        if (Math.abs(mx - x0) <= TRIM_HANDLE_W ||
                            Math.abs(mx - x1) <= TRIM_HANDLE_W) cur = "ew-resize";
                    }
                }
                cvs.style.cursor = cur;
            });

            const up = (e) => {
                S.drag = null;
                S._dragStart = null;
                try { cvs.releasePointerCapture(e.pointerId); } catch (_) {}
            };
            cvs.addEventListener("pointerup", up);
            cvs.addEventListener("pointercancel", up);

            cvs.addEventListener("keydown", (e) => {
                let handled = true;
                const step = e.shiftKey ? 10 : 1;
                if (e.key === "ArrowLeft")       setIdx(S.idx - step);
                else if (e.key === "ArrowRight") setIdx(S.idx + step);
                else if (e.key === "Home")       setIdx(S.trimStart);
                else if (e.key === "End")        setIdx(S.trimEnd);
                else if (e.key === " ") { S.playing = !S.playing; if (S.playing) startPlay(); else stopPlay(); }
                else if (e.key === "r" || e.key === "R") {
                    if (!isCropLocked()) setCropNorm(0, 0, 1, 1);
                }
                else if (e.key === "i" || e.key === "I") setTrim(S.idx, getTrim()[1]);  // mark IN
                else if (e.key === "o" || e.key === "O") setTrim(getTrim()[0], S.idx);  // mark OUT
                else handled = false;
                if (handled) { e.preventDefault(); node._render(); }
            });

            const tlRect = () => {
                const cw = cvs.width, ch = cvs.height;
                const pad = 12;
                const x = pad, y = ch - TL_H - 6;
                const w = cw - pad * 2, h = TL_H;
                return { x, y, w, h, playBtnX: x, playBtnW: PLAY_BTN_W,
                         barX: x + PLAY_BTN_W + 8, barW: w - PLAY_BTN_W - 8 };
            };
            node._tlRect = tlRect;

            const startPlay = () => {
                S._lastT = performance.now();
                S._ppDir = 1;
                const tick = (now) => {
                    if (!S.playing) return;
                    const dt = (now - S._lastT) / 1000;
                    const advance = dt * Math.max(0.1, S.fps);
                    if (advance >= 1) {
                        S._lastT = now;
                        const lo = S.trimStart | 0;
                        const hi = S.trimEnd | 0;
                        const span = Math.max(1, hi - lo + 1);
                        const inc = Math.floor(advance) * (S.loopMode === "ping-pong" ? S._ppDir : 1);
                        let next = S.idx + inc;
                        if (S.loopMode === "once") {
                            if (next > hi) { next = hi; S.playing = false; stopPlay(); }
                            else if (next < lo) next = lo;
                        } else if (S.loopMode === "ping-pong") {
                            if (next > hi) { next = hi - (next - hi); S._ppDir = -1; if (next < lo) next = lo; }
                            else if (next < lo) { next = lo + (lo - next); S._ppDir = 1; if (next > hi) next = hi; }
                        } else { // loop
                            if (next > hi) next = lo + ((next - lo) % span);
                            else if (next < lo) next = lo;
                        }
                        setIdx(next);
                    }
                    S._rafId = requestAnimationFrame(tick);
                };
                S._rafId = requestAnimationFrame(tick);
            };
            const stopPlay = () => {
                if (S._rafId) cancelAnimationFrame(S._rafId);
                S._rafId = null;
            };
            node._stopPlay = stopPlay;

            node.addDOMWidget("video_player_view", "VFPLAYER", el, { serialize: false });
            node.setSize([520, 380]);

            let _lockedW = 520, _lockedH = 380;
            const _origCompute = node.computeSize;
            node.computeSize = function () {
                if (_lockedW > 0 && _lockedH > 0) return [_lockedW, _lockedH];
                return _origCompute?.apply(this, arguments) ?? [520, 380];
            };
            node._lockSize = (w, h) => { _lockedW = w; _lockedH = h; };

            // Re-render whenever a crop-related widget changes from the panel
            const watch = ["crop_enabled", "crop_locked", "aspect_ratio",
                           "custom_aspect_w", "custom_aspect_h",
                           "crop_x", "crop_y", "crop_w", "crop_h",
                           "frame_start", "frame_end", "frame_stride",
                           "playback_fps", "loop_mode"];
            requestAnimationFrame(() => {
                for (const wn of watch) {
                    const w = node.widgets?.find(x => x.name === wn);
                    if (!w) continue;
                    const _cb = w.callback;
                    w.callback = function (v) {
                        try { _cb?.call(this, v); } catch (_) {}
                        // When aspect_ratio changes, snap the current crop.
                        if (wn === "aspect_ratio" || wn === "custom_aspect_w" || wn === "custom_aspect_h") {
                            const a = getAspect(node, S);
                            const [cx, cy, cw, ch] = getCropNorm();
                            const [nx, ny, nw, nh] = snapToAspect(cx, cy, cw, ch, a);
                            setCropNorm(nx, ny, nw, nh);
                        }
                        if (wn === "playback_fps") S.fps = Math.max(0.1, Number(v) || 24);
                        if (wn === "loop_mode") S.loopMode = String(v || "loop");
                        if (wn === "frame_start" || wn === "frame_end") {
                            const [lo, hi] = getTrim();
                            S.trimStart = lo; S.trimEnd = hi;
                        }
                        node._render?.();
                        app.graph.setDirtyCanvas(true);
                    };
                }
            });
        };

        const _exec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (msg) {
            _exec?.apply(this, arguments);
            const S = this._S;
            if (!S || !msg?.frames) return;

            const frames = msg.frames;
            S.frameCount = msg.frame_count?.[0] ?? frames.length;
            S.idx = Math.max(0, Math.min(S.frameCount - 1, msg.current_index?.[0] ?? S.idx));
            S.width = msg.width?.[0] ?? 0;
            S.height = msg.height?.[0] ?? 0;
            S.trimStart = msg.frame_start?.[0] ?? 0;
            S.trimEnd = msg.frame_end?.[0] ?? (S.frameCount - 1);
            S.fps = Math.max(0.1, Number(msg.playback_fps?.[0] ?? 24));
            S.loopMode = String(msg.loop_mode?.[0] ?? "loop");

            // Sync server-snapped crop back to widgets
            if (typeof msg.crop_x?.[0] === "number") {
                this._setCropNorm?.(msg.crop_x[0], msg.crop_y[0], msg.crop_w[0], msg.crop_h[0]);
            }

            const mkURL = (info) =>
                api.apiURL(
                    `/view?filename=${encodeURIComponent(info.filename)}` +
                    `&type=${info.type}` +
                    `&subfolder=${encodeURIComponent(info.subfolder || "")}`
                );

            S.images = new Array(frames.length);
            const node = this;
            let loaded = 0;
            const onLoad = () => {
                if (++loaded >= frames.length) {
                    if (S.width > 0 && S.height > 0) {
                        const w = Math.max(node.size[0], 360);
                        const aspect = S.height / S.width;
                        const previewArea = (w - 24) * aspect;
                        const h = Math.round(previewArea + TL_H + STATUS_H + 24);
                        node._lockSize?.(w, h);
                        node.setSize([w, h]);
                    }
                    node._render();
                    app.graph.setDirtyCanvas(true);
                }
            };
            for (let i = 0; i < frames.length; i++) {
                const im = new Image();
                im.onload = onLoad;
                im.onerror = onLoad;
                im.src = mkURL(frames[i]);
                S.images[i] = im;
            }
        };

        nodeType.prototype._render = function () {
            const S = this._S;
            if (!S) return;
            const cvs = S.cvs, ctx = S.ctx;
            const cw = S.el.clientWidth || 480;
            const ch = Math.max(160, S.el.clientHeight || 320);
            if (cvs.width !== cw || cvs.height !== ch) {
                cvs.width = cw;
                cvs.height = ch;
            }
            ctx.clearRect(0, 0, cw, ch);

            const tl = this._tlRect();
            const previewMaxH = tl.y - 12 - STATUS_H;
            const previewMaxW = cw - 24;
            const aspect = (S.height || 1) / (S.width || 1);
            let pw = previewMaxW, ph = pw * aspect;
            if (ph > previewMaxH) { ph = previewMaxH; pw = ph / aspect; }
            const px = (cw - pw) / 2;
            const py = STATUS_H + 6;
            S.preview = { x: px, y: py, w: pw, h: ph };

            ctx.fillStyle = "#000";
            ctx.fillRect(px, py, pw, ph);
            const cur = S.images?.[S.idx];
            if (cur && cur.complete && cur.naturalWidth > 0) {
                ctx.drawImage(cur, px, py, pw, ph);
            } else {
                ctx.fillStyle = "#444";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("loading...", px + pw / 2, py + ph / 2);
            }

            // Crop overlay
            if (this._isCropEnabled?.()) {
                const [ncx, ncy, ncw, nch] = this._getCropNorm();
                const cr = {
                    x: px + ncx * pw, y: py + ncy * ph,
                    w: ncw * pw,       h: nch * ph,
                };
                // Dim outside crop
                ctx.save();
                ctx.fillStyle = "rgba(0,0,0,0.55)";
                // top
                ctx.fillRect(px, py, pw, cr.y - py);
                // bottom
                ctx.fillRect(px, cr.y + cr.h, pw, (py + ph) - (cr.y + cr.h));
                // left
                ctx.fillRect(px, cr.y, cr.x - px, cr.h);
                // right
                ctx.fillRect(cr.x + cr.w, cr.y, (px + pw) - (cr.x + cr.w), cr.h);
                ctx.restore();

                // Border
                ctx.save();
                ctx.strokeStyle = this._isCropLocked?.() ? "#f80" : "#5bf";
                ctx.lineWidth = CROP_LINE_W;
                if (this._isCropLocked?.()) ctx.setLineDash([6, 4]);
                ctx.strokeRect(cr.x + 0.5, cr.y + 0.5, cr.w - 1, cr.h - 1);
                ctx.setLineDash([]);

                // Rule-of-thirds guides
                ctx.strokeStyle = "rgba(91,255,255,0.35)";
                ctx.lineWidth = 1;
                for (let i = 1; i <= 2; i++) {
                    const tx = cr.x + (cr.w * i) / 3;
                    const ty = cr.y + (cr.h * i) / 3;
                    ctx.beginPath(); ctx.moveTo(tx, cr.y); ctx.lineTo(tx, cr.y + cr.h); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(cr.x, ty); ctx.lineTo(cr.x + cr.w, ty); ctx.stroke();
                }

                // Handles (suppressed when locked)
                if (!this._isCropLocked?.()) {
                    ctx.fillStyle = "#fff";
                    ctx.strokeStyle = "#222";
                    ctx.lineWidth = 1;
                    const drawHandle = (hx, hy) => {
                        ctx.fillRect(hx - CROP_HANDLE_S/2, hy - CROP_HANDLE_S/2, CROP_HANDLE_S, CROP_HANDLE_S);
                        ctx.strokeRect(hx - CROP_HANDLE_S/2 + 0.5, hy - CROP_HANDLE_S/2 + 0.5, CROP_HANDLE_S - 1, CROP_HANDLE_S - 1);
                    };
                    drawHandle(cr.x, cr.y);
                    drawHandle(cr.x + cr.w, cr.y);
                    drawHandle(cr.x, cr.y + cr.h);
                    drawHandle(cr.x + cr.w, cr.y + cr.h);
                    drawHandle(cr.x + cr.w / 2, cr.y);
                    drawHandle(cr.x + cr.w / 2, cr.y + cr.h);
                    drawHandle(cr.x, cr.y + cr.h / 2);
                    drawHandle(cr.x + cr.w, cr.y + cr.h / 2);
                }
                ctx.restore();

                // Crop dim text
                ctx.save();
                ctx.fillStyle = "rgba(0,0,0,0.6)";
                const cwPx = Math.round(ncw * (S.width || 0));
                const chPx = Math.round(nch * (S.height || 0));
                const txt = `${cwPx} x ${chPx}`;
                ctx.font = "11px sans-serif";
                const tw = ctx.measureText(txt).width + 8;
                ctx.fillRect(cr.x + 2, cr.y + 2, tw, 16);
                ctx.fillStyle = "#fff";
                ctx.textAlign = "left";
                ctx.textBaseline = "top";
                ctx.fillText(txt, cr.x + 6, cr.y + 4);
                ctx.restore();
            }

            // Status line
            ctx.save();
            ctx.fillStyle = "#bbb";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            const stride = Number(this.widgets?.find(w => w.name === "frame_stride")?.value ?? 1);
            const trimSpan = Math.max(0, S.trimEnd - S.trimStart + 1);
            const emit = Math.max(1, Math.ceil(trimSpan / Math.max(1, stride)));
            const statusTxt = `Frame ${S.idx + 1} / ${S.frameCount}   ${S.width}x${S.height}   ` +
                              `${S.fps.toFixed(1)} fps   ${S.loopMode}   ` +
                              `trim ${S.trimStart}-${S.trimEnd}` +
                              (stride > 1 ? ` /${stride}` : "") +
                              `  -> ${emit}f`;
            ctx.fillText(statusTxt, 12, 4);
            ctx.restore();

            // Timeline bar
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(tl.x, tl.y, tl.w, tl.h);

            // Play/pause button
            ctx.fillStyle = "#2a2a2a";
            ctx.fillRect(tl.playBtnX, tl.y, tl.playBtnW, tl.h);
            ctx.fillStyle = "#fff";
            ctx.beginPath();
            const cxp = tl.playBtnX + tl.playBtnW / 2;
            const cyp = tl.y + tl.h / 2;
            if (S.playing) {
                ctx.fillRect(cxp - 6, cyp - 7, 4, 14);
                ctx.fillRect(cxp + 2, cyp - 7, 4, 14);
            } else {
                ctx.moveTo(cxp - 5, cyp - 7);
                ctx.lineTo(cxp + 7, cyp);
                ctx.lineTo(cxp - 5, cyp + 7);
                ctx.closePath();
                ctx.fill();
            }

            // Track
            const trackY = tl.y + tl.h / 2;
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(tl.barX, trackY);
            ctx.lineTo(tl.barX + tl.barW, trackY);
            ctx.stroke();

            // Ticks
            ctx.strokeStyle = "#444";
            ctx.lineWidth = 1;
            const tickEvery = S.frameCount <= 20 ? 1 : Math.max(1, Math.floor(S.frameCount / 20));
            for (let i = 0; i < S.frameCount; i += tickEvery) {
                const tx = tl.barX + (i / Math.max(1, S.frameCount - 1)) * tl.barW;
                ctx.beginPath();
                ctx.moveTo(tx, trackY - TICK_H / 2);
                ctx.lineTo(tx, trackY + TICK_H / 2);
                ctx.stroke();
            }

            // Progress fill
            const progT = S.frameCount > 1 ? S.idx / (S.frameCount - 1) : 0;
            ctx.strokeStyle = "#5bf";
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(tl.barX, trackY);
            ctx.lineTo(tl.barX + tl.barW * progT, trackY);
            ctx.stroke();

            // Handle
            const hxp = tl.barX + tl.barW * progT;
            ctx.beginPath();
            ctx.arc(hxp, trackY, HANDLE_R, 0, Math.PI * 2);
            ctx.fillStyle = "#fff";
            ctx.fill();
            ctx.strokeStyle = "#5bf";
            ctx.lineWidth = 2;
            ctx.stroke();

            // Trim region overlay + markers
            if (S.frameCount > 0) {
                const t0 = S.frameCount > 1 ? S.trimStart / (S.frameCount - 1) : 0;
                const t1 = S.frameCount > 1 ? S.trimEnd / (S.frameCount - 1) : 1;
                const x0 = tl.barX + t0 * tl.barW;
                const x1 = tl.barX + t1 * tl.barW;
                // Dim outside trim
                ctx.save();
                ctx.fillStyle = "rgba(0,0,0,0.5)";
                if (x0 > tl.barX) ctx.fillRect(tl.barX, tl.y, x0 - tl.barX, tl.h);
                if (x1 < tl.barX + tl.barW) ctx.fillRect(x1, tl.y, (tl.barX + tl.barW) - x1, tl.h);
                // Trim band on track
                ctx.fillStyle = "rgba(120,220,120,0.18)";
                ctx.fillRect(x0, trackY - 6, x1 - x0, 12);
                // Start marker (green) + end marker (red)
                ctx.fillStyle = "#5d5";
                ctx.fillRect(x0 - 2, tl.y + 2, 3, tl.h - 4);
                ctx.fillStyle = "#e55";
                ctx.fillRect(x1 - 1, tl.y + 2, 3, tl.h - 4);
                // tiny grip dots
                ctx.fillStyle = "#fff";
                ctx.fillRect(x0 - 1, tl.y + tl.h / 2 - 1, 1, 2);
                ctx.fillRect(x1, tl.y + tl.h / 2 - 1, 1, 2);
                // Stride visualisation
                const stride = Number(this.widgets?.find(w => w.name === "frame_stride")?.value ?? 1);
                if (stride > 1 && S.frameCount > 1) {
                    ctx.fillStyle = "#fc6";
                    for (let f = S.trimStart; f <= S.trimEnd; f += stride) {
                        const tx = tl.barX + (f / (S.frameCount - 1)) * tl.barW;
                        ctx.fillRect(tx - 0.5, trackY - 8, 1, 4);
                    }
                }
                ctx.restore();
            }
        };

        const _onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try { this._stopPlay?.(); } catch (_) {}
            _onRemoved?.apply(this, arguments);
        };
    },
});
