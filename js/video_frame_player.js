import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * VideoFramePlayerMEC – lightweight OpenRV-style frame scrubber.
 *
 * Renders a canvas + timeline bar. Drag/click on timeline to scrub frames;
 * arrow keys step ±1 frame; space = play/pause. Selected frame index is
 * mirrored back to the `frame_index` widget so the next graph run emits
 * that frame as the IMAGE output.
 */

const TL_H = 36;       // timeline bar height
const TICK_H = 6;      // tick mark height
const HANDLE_R = 7;
const PLAY_BTN_W = 28;
const STATUS_H = 18;

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
                "position:relative;width:100%;min-height:240px;background:#0c0c0c;border-radius:4px;overflow:hidden;";
            el.setAttribute("role", "group");
            el.setAttribute("aria-label", "Video frame scrubber. Drag timeline to scrub. Left/Right step frames. Space toggles play.");

            const cvs = document.createElement("canvas");
            cvs.style.cssText = "display:block;width:100%;cursor:default;";
            cvs.setAttribute("tabindex", "0");
            el.appendChild(cvs);

            const S = {
                cvs,
                ctx: cvs.getContext("2d"),
                el,
                images: [],          // Image[] preloaded thumbnails
                frameCount: 0,
                idx: 0,               // current frame
                width: 0,             // source width
                height: 0,            // source height
                playing: false,
                fps: 24,
                _rafId: null,
                _lastT: 0,
                drag: false,
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
                S.idx = Math.max(0, Math.min(S.frameCount - 1, Math.round(i)));
                // Mirror to frame_index widget
                const w = node.widgets?.find((w) => w.name === "frame_index");
                if (w) {
                    w.value = S.idx;
                    if (w.callback) try { w.callback(S.idx); } catch (_) {}
                }
                node._render();
                app.graph.setDirtyCanvas(true);
            };

            const tlRect = () => {
                const cw = cvs.width, ch = cvs.height;
                const pad = 12;
                const x = pad, y = ch - TL_H - 6;
                const w = cw - pad * 2, h = TL_H;
                return { x, y, w, h, playBtnX: x, playBtnW: PLAY_BTN_W,
                         barX: x + PLAY_BTN_W + 8, barW: w - PLAY_BTN_W - 8 };
            };

            cvs.addEventListener("pointerdown", (e) => {
                const [cx, cy] = canvasXY(e);
                const r = tlRect();
                // Play/pause button
                if (cx >= r.playBtnX && cx <= r.playBtnX + r.playBtnW &&
                    cy >= r.y && cy <= r.y + r.h) {
                    S.playing = !S.playing;
                    if (S.playing) startPlay(); else stopPlay();
                    node._render();
                    return;
                }
                // Timeline bar
                if (cy >= r.y && cy <= r.y + r.h && cx >= r.barX) {
                    cvs.setPointerCapture(e.pointerId);
                    S.drag = true;
                    const t = (cx - r.barX) / Math.max(1, r.barW);
                    setIdx(t * (S.frameCount - 1));
                }
            });
            cvs.addEventListener("pointermove", (e) => {
                if (!S.drag) return;
                const [cx] = canvasXY(e);
                const r = tlRect();
                const t = (cx - r.barX) / Math.max(1, r.barW);
                setIdx(t * (S.frameCount - 1));
            });
            const up = () => { S.drag = false; };
            cvs.addEventListener("pointerup", up);
            cvs.addEventListener("pointercancel", up);

            cvs.addEventListener("keydown", (e) => {
                let handled = true;
                const step = e.shiftKey ? 10 : 1;
                if (e.key === "ArrowLeft")       setIdx(S.idx - step);
                else if (e.key === "ArrowRight") setIdx(S.idx + step);
                else if (e.key === "Home")       setIdx(0);
                else if (e.key === "End")        setIdx(S.frameCount - 1);
                else if (e.key === " ") { S.playing = !S.playing; if (S.playing) startPlay(); else stopPlay(); }
                else handled = false;
                if (handled) { e.preventDefault(); node._render(); }
            });

            const startPlay = () => {
                S._lastT = performance.now();
                const tick = (now) => {
                    if (!S.playing) return;
                    const dt = (now - S._lastT) / 1000;
                    const advance = dt * S.fps;
                    if (advance >= 1) {
                        S._lastT = now;
                        let next = S.idx + Math.floor(advance);
                        if (next >= S.frameCount) next = 0;  // loop
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
            node.setSize([520, 360]);

            // Lock compute size to prevent jitter
            let _lockedW = 520, _lockedH = 360;
            const _origCompute = node.computeSize;
            node.computeSize = function() {
                if (_lockedW > 0 && _lockedH > 0) return [_lockedW, _lockedH];
                return _origCompute?.apply(this, arguments) ?? [520, 360];
            };
            node._lockSize = (w, h) => { _lockedW = w; _lockedH = h; };
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

            const mkURL = (info) =>
                api.apiURL(
                    `/view?filename=${encodeURIComponent(info.filename)}` +
                    `&type=${info.type}` +
                    `&subfolder=${encodeURIComponent(info.subfolder || "")}`
                );

            // Preload all preview frames
            S.images = new Array(frames.length);
            const node = this;
            let loaded = 0;
            const onLoad = () => {
                if (++loaded >= frames.length) {
                    // Resize node to fit aspect ratio
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

            // Draw current frame fitted to the available area (above timeline)
            const tl = (function () {
                const pad = 12;
                const x = pad, y = ch - TL_H - 6;
                return { x, y, w: cw - pad * 2, h: TL_H,
                         playBtnX: x, playBtnW: PLAY_BTN_W,
                         barX: x + PLAY_BTN_W + 8, barW: cw - pad * 2 - PLAY_BTN_W - 8 };
            })();
            const previewMaxH = tl.y - 12 - STATUS_H;
            const previewMaxW = cw - 24;
            const aspect = (S.height || 1) / (S.width || 1);
            let pw = previewMaxW, ph = pw * aspect;
            if (ph > previewMaxH) { ph = previewMaxH; pw = ph / aspect; }
            const px = (cw - pw) / 2;
            const py = STATUS_H + 6;

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
                ctx.fillText("loading…", px + pw / 2, py + ph / 2);
            }

            // Status line (top)
            ctx.save();
            ctx.fillStyle = "#bbb";
            ctx.font = "11px sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            const statusTxt = `Frame ${S.idx + 1} / ${S.frameCount}   ${S.width}x${S.height}   ${S.fps} fps`;
            ctx.fillText(statusTxt, 12, 4);
            ctx.restore();

            // Timeline bar background
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(tl.x, tl.y, tl.w, tl.h);

            // Play/pause button
            ctx.fillStyle = "#2a2a2a";
            ctx.fillRect(tl.playBtnX, tl.y, tl.playBtnW, tl.h);
            ctx.fillStyle = "#fff";
            ctx.beginPath();
            const cx = tl.playBtnX + tl.playBtnW / 2;
            const cy = tl.y + tl.h / 2;
            if (S.playing) {
                // Pause icon: two bars
                ctx.fillRect(cx - 6, cy - 7, 4, 14);
                ctx.fillRect(cx + 2, cy - 7, 4, 14);
            } else {
                // Play icon: triangle
                ctx.moveTo(cx - 5, cy - 7);
                ctx.lineTo(cx + 7, cy);
                ctx.lineTo(cx - 5, cy + 7);
                ctx.closePath();
                ctx.fill();
            }

            // Timeline track
            const trackY = tl.y + tl.h / 2;
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(tl.barX, trackY);
            ctx.lineTo(tl.barX + tl.barW, trackY);
            ctx.stroke();

            // Tick marks every 10% (or every frame if <=20 frames)
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
            const hx = tl.barX + tl.barW * progT;
            ctx.beginPath();
            ctx.arc(hx, trackY, HANDLE_R, 0, Math.PI * 2);
            ctx.fillStyle = "#fff";
            ctx.fill();
            ctx.strokeStyle = "#5bf";
            ctx.lineWidth = 2;
            ctx.stroke();
        };

        // Cleanup playback on node removal
        const _onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try { this._stopPlay?.(); } catch (_) {}
            _onRemoved?.apply(this, arguments);
        };
    },
});
