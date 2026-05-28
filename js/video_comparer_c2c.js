// VideoComparerC2C — frontend upload + REAL-TIME A/B comparer widget.
// (formerly VideoComparerMEC; legacy node key kept as alias)
//
// Modes that render LIVE in the browser (no Queue needed):
//   wipe, onion, diff, side_by_side, per_channel, false_color, bit_depth_crush
//
// Modes that require a server Queue (numpy / librosa / pyloudnorm):
//   waveform_scope, parade_scope, vectorscope, histogram_scope,
//   audio_waveform, audio_spectro, audio_loudness
//
// Sources resolved live in-browser:
//   - file_a / file_b combo (uploaded via the upload buttons or pre-existing
//     files in ComfyUI/input/). Fetched as /view?filename=...&type=input.
//   - Tensor inputs (image_a/image_b) cannot be previewed without a Queue.
//
// Drag the canvas horizontally to scrub the wipe / onion. For videos use the
// frame_index slider OR ←/→ arrow keys on the canvas to scrub.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { C } from './_c2c_theme.js';

// Registered under both "VideoComparerC2C" (new) and "VideoComparerMEC"
// (legacy alias) so old saved workflows load with the new widget.
const NODE_IDS = new Set(["VideoComparerC2C", "VideoComparerMEC"]);

const LIVE_MODES = new Set([
    "wipe", "onion", "diff", "side_by_side",
    "per_channel", "false_color", "bit_depth_crush",
    // synced_player is technically not "canvas-rendered" — it uses two
    // real <video> elements — but it IS fully client-side, so we mark
    // it LIVE so the LIVE/Queue badge reads correctly.
    "synced_player",
]);

// 256-entry color LUTs (matplotlib-style approximations) for false_color /
// bit_depth_crush.
function _lut(name) {
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
        const t = i / 255;
        let r, g, b;
        switch (name) {
            case "viridis": r = 68 + (253 - 68) * t; g = 1 + (231 - 1) * t; b = 84 + (37 - 84) * t; break;
            case "plasma":  r = 13 + (240 - 13) * t; g = 8 + (249 - 8) * t; b = 135 + (33 - 135) * t; break;
            case "inferno": r = 0 + (252 - 0) * t; g = 0 + (255 - 0) * t; b = 4 + (164 - 4) * t; break;
            case "magma":   r = 0 + (252 - 0) * t; g = 0 + (253 - 0) * t; b = 4 + (191 - 4) * t; break;
            case "hot":     r = Math.min(255, t * 3 * 255); g = Math.min(255, Math.max(0, (t - 0.33) * 3 * 255)); b = Math.min(255, Math.max(0, (t - 0.66) * 3 * 255)); break;
            case "coolwarm": r = 59 + (180 - 59) * t; g = 76 + (4 - 76) * t; b = 192 + (38 - 192) * t; break;
            case "turbo":
            default:
                r = Math.round(34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05)))));
                g = Math.round(23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (1073.77 + t * 707.56)))));
                b = Math.round(27.2 + t * (3211.1 + t * (-15327.97 + t * (27814.0 + t * (-22569.18 + t * 6838.66)))));
                break;
        }
        lut[i * 3]     = Math.max(0, Math.min(255, r));
        lut[i * 3 + 1] = Math.max(0, Math.min(255, g));
        lut[i * 3 + 2] = Math.max(0, Math.min(255, b));
    }
    return lut;
}

async function uploadFile(file) {
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("type", "input");
    body.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!resp.ok) throw new Error(`upload failed: ${resp.status}`);
    const data = await resp.json();
    const sub = data.subfolder ? data.subfolder + "/" : "";
    return sub + (data.name || file.name);
}

const VIDEO_EXT = /\.(mp4|mov|mkv|webm|m4v|avi|gif)$/i;
const AUDIO_EXT = /\.(wav|mp3|flac|ogg|aac|m4a)$/i;

function mediaURL(filename) {
    if (!filename) return null;
    return api.apiURL(
        `/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=`,
    );
}

/** Load a file path as Image or Video element. Returns {kind, el, w, h, dur}. */
function loadSource(filename) {
    return new Promise((resolve) => {
        if (!filename) { resolve(null); return; }
        const url = mediaURL(filename);
        if (VIDEO_EXT.test(filename)) {
            const v = document.createElement("video");
            v.src = url;
            v.muted = true;
            v.crossOrigin = "anonymous";
            v.preload = "auto";
            v.playsInline = true;
            v.addEventListener("loadeddata", () => {
                resolve({ kind: "video", el: v, w: v.videoWidth, h: v.videoHeight, dur: v.duration });
            }, { once: true });
            v.addEventListener("error", () => resolve(null), { once: true });
        } else if (AUDIO_EXT.test(filename)) {
            resolve({ kind: "audio", el: null, w: 0, h: 0 });
        } else {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => resolve({ kind: "image", el: img, w: img.naturalWidth, h: img.naturalHeight });
            img.onerror = () => resolve(null);
            img.src = url;
        }
    });
}

function getW(node, name) { return node.widgets?.find((w) => w.name === name); }
function getVal(node, name, def) { const w = getW(node, name); return w ? w.value : def; }

function makeUploadButton(node, slot, label, onUploaded) {
    const widgetName = slot === "a" ? "file_a" : "file_b";
    const btn = node.addWidget("button", `📁 Upload ${label}`, null, async () => {
        const inp = document.createElement("input");
        inp.type = "file";
        inp.accept =
            "image/*,video/*,audio/*,.exr,.hdr,.tif,.tiff,.webp,.mp4,.mov,.mkv,.webm,.gif,.wav,.mp3,.flac,.ogg,.aac,.m4a";
        inp.style.display = "none";
        document.body.appendChild(inp);
        inp.addEventListener("change", async () => {
            const f = inp.files && inp.files[0];
            inp.remove();
            if (!f) return;
            try {
                btn.name = `⏳ Uploading ${label}…`;
                node.setDirtyCanvas(true, true);
                const uploaded = await uploadFile(f);
                const w = getW(node, widgetName);
                if (w) {
                    if (Array.isArray(w.options?.values) && !w.options.values.includes(uploaded)) {
                        w.options.values.push(uploaded);
                    }
                    w.value = uploaded;
                }
                btn.name = `✅ ${label}: ${f.name.length > 22 ? f.name.slice(0, 19) + "…" : f.name}`;
                if (onUploaded) await onUploaded(slot, uploaded);
            } catch (e) {
                console.error("[VideoComparerC2C] upload error", e);
                btn.name = `❌ Upload ${label} failed`;
            }
            node.setDirtyCanvas(true, true);
        });
        inp.click();
    });
    btn.serialize = false;
    return btn;
}

// ── Legacy-type migration: saved workflows that reference the old
// `VideoComparerMEC` key would otherwise fail to instantiate because we
// no longer register that key in NODE_CLASS_MAPPINGS (it caused a
// duplicate entry in the node search palette). We hook
// `loadGraphData` / `app.graph.configure` and rewrite the node `type`
// in-place BEFORE LiteGraph tries to find the registered class.
// Idempotent — safe to run on every graph load.
function _migrateLegacyVideoComparerTypes(graphData) {
    try {
        const nodes = graphData?.nodes;
        if (!Array.isArray(nodes)) return;
        let migrated = 0;
        for (const n of nodes) {
            if (n && n.type === "VideoComparerMEC") {
                n.type = "VideoComparerC2C";
                migrated += 1;
            }
        }
        if (migrated > 0) {
            console.log(
                `[VideoComparerC2C] migrated ${migrated} legacy ` +
                `"VideoComparerMEC" node(s) to "VideoComparerC2C" on load.`
            );
        }
    } catch (e) {
        console.warn("[VideoComparerC2C] legacy-type migration failed:", e);
    }
}

app.registerExtension({
    name: "MEC.VideoComparer",
    async setup() {
        // Wrap loadGraphData so EVERY graph load (file open, paste, undo,
        // template, examples) is run through the migration.
        const _origLoad = app.loadGraphData?.bind(app);
        if (typeof _origLoad === "function") {
            // Defensive: ComfyUI's loadDefaultWorkflow path can invoke
            // app.loadGraphData() with no arguments (or null) when there is
            // no previously-saved workflow. Some upstream wrappers (Manager,
            // rgthree) dereference graphData.extra and crash on undefined,
            // which cascades through ALL c2c loadGraphData wrappers and
            // leaves the OmniPill ecosystem in a half-initialised state
            // (buttons render but downstream init never completes). Coalesce
            // to a benign empty workflow so every wrapper downstream has a
            // safe object to read from.
            const _emptyGraph = () => ({
                extra: {}, nodes: [], links: [], groups: [],
                version: 0.4, last_node_id: 0, last_link_id: 0,
                config: {},
            });
            app.loadGraphData = function (graphData, ...rest) {
                if (graphData == null) graphData = _emptyGraph();
                try { _migrateLegacyVideoComparerTypes(graphData); }
                catch (err) {
                    console.warn("[video_comparer] migrate failed", err);
                }
                return _origLoad(graphData, ...rest);
            };
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_IDS.has(nodeData.name)) return;

        const _onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _onCreated?.apply(this, arguments);
            const node = this;

            const S = {
                srcA: null, srcB: null,
                serverPreview: null,
                drag: false,
                wipePos: 0.5,
                onionAlpha: 0.5,
                rafToken: 0,
            };
            node._VC = S;

            // ── Build DOM widget ─────────────────────────────
            const wrap = document.createElement("div");
            wrap.style.cssText = "position:relative;width:calc(100% - 12px);margin:6px;background:var(--c2c-scrimDark2);border:1px solid #1f1f2a;border-radius:6px;overflow:hidden;min-height:240px;";
            wrap.setAttribute("role", "group");
            wrap.setAttribute("aria-label", "A/B media comparer canvas");

            const cvs = document.createElement("canvas");
            cvs.style.cssText = "display:block;width:100%;cursor:col-resize;outline:none;";
            cvs.tabIndex = 0;
            cvs.setAttribute("role", "img");
            wrap.appendChild(cvs);

            const overlay = document.createElement("div");
            overlay.style.cssText = "position:absolute;top:0;left:0;right:0;padding:4px 8px;display:flex;justify-content:space-between;align-items:center;pointer-events:none;font:11px system-ui,sans-serif;color:var(--c2c-fg);text-shadow:0 1px 2px rgba(0,0,0,0.8);";
            const modeBadge = document.createElement("div");
            modeBadge.textContent = "wipe";
            modeBadge.style.cssText = "background:rgba(0,0,0,0.55);padding:2px 8px;border-radius:10px;";
            const liveBadge = document.createElement("div");
            liveBadge.textContent = "● LIVE";
            liveBadge.style.cssText = "background:rgba(40,180,80,0.85);color:var(--c2c-white);padding:2px 8px;border-radius:10px;font-weight:600;";
            overlay.appendChild(modeBadge);
            overlay.appendChild(liveBadge);
            wrap.appendChild(overlay);

            const hint = document.createElement("div");
            hint.style.cssText = "position:absolute;bottom:6px;left:8px;right:8px;font:10px system-ui;color:#9aa0b8;pointer-events:none;text-shadow:0 1px 2px rgba(0,0,0,0.7);";
            hint.textContent = "Upload A/B (or pick from combo). Drag canvas to wipe. ←/→ to scrub video frames. Space to play/pause.";
            wrap.appendChild(hint);

            const ctx = cvs.getContext("2d", { willReadFrequently: true });

            // ── Synced dual-video player (mode = "synced_player") ──
            // Two side-by-side <video> elements sharing a single transport.
            // Hidden by default; activated by the render() switch.
            const playerHost = document.createElement("div");
            playerHost.style.cssText = "display:none;flex-direction:column;width:100%;background:#0a0a10;";
            const playerRow = document.createElement("div");
            playerRow.style.cssText = "display:flex;width:100%;gap:2px;background:#0a0a10;";
            const vidA = document.createElement("video");
            const vidB = document.createElement("video");
            for (const v of [vidA, vidB]) {
                // NOTE: parent wrapA/wrapB already provide the 50/50 split via
                // `flex:1 1 50%`. The <video> itself must fill its wrapper
                // (width:100%) — using width:50% here caused each video to
                // render at 25% of total width (50% of 50%), making the
                // player look "broken" / "can't see it play". (P1.2 regression fix)
                v.style.cssText = "width:100%;height:auto;display:block;background:var(--c2c-black);";
                v.playsInline = true;
                v.preload = "auto";
                v.muted = true;          // dual-audio would echo; user can unmute the master
                v.controls = false;
            }
            vidA.title = "A";
            vidB.title = "B";
            const vidALabel = document.createElement("div");
            vidALabel.textContent = "A";
            vidALabel.style.cssText = "position:absolute;top:6px;left:6px;background:rgba(0,0,0,0.6);color:var(--c2c-okSoft);padding:1px 6px;border-radius:8px;font:11px system-ui;font-weight:600;pointer-events:none;";
            const vidBLabel = document.createElement("div");
            vidBLabel.textContent = "B";
            vidBLabel.style.cssText = "position:absolute;top:6px;right:6px;background:rgba(0,0,0,0.6);color:var(--c2c-pink);padding:1px 6px;border-radius:8px;font:11px system-ui;font-weight:600;pointer-events:none;";
            const wrapA = document.createElement("div");
            wrapA.style.cssText = "position:relative;flex:1 1 50%;min-width:0;";
            wrapA.appendChild(vidA);
            wrapA.appendChild(vidALabel);
            const wrapB = document.createElement("div");
            wrapB.style.cssText = "position:relative;flex:1 1 50%;min-width:0;";
            wrapB.appendChild(vidB);
            wrapB.appendChild(vidBLabel);
            playerRow.appendChild(wrapA);
            playerRow.appendChild(wrapB);
            playerHost.appendChild(playerRow);

            // Shared transport: play/pause button + seekbar + time readout.
            const transport = document.createElement("div");
            transport.style.cssText = "display:flex;align-items:center;gap:6px;padding:6px 8px;background:var(--c2c-bg3);border-top:1px solid #1f1f2a;color:var(--c2c-fg);font:11px system-ui;";
            const playBtn = document.createElement("button");
            playBtn.type = "button";
            playBtn.textContent = "▶";
            playBtn.style.cssText = "background:var(--c2c-bg);border:1px solid var(--c2c-surface0);color:var(--c2c-fg);border-radius:4px;padding:2px 10px;cursor:pointer;font-size:13px;line-height:1;";
            const seek = document.createElement("input");
            seek.type = "range";
            seek.min = "0";
            seek.max = "10000";
            seek.value = "0";
            seek.step = "1";
            seek.style.cssText = "flex:1 1 auto;accent-color:var(--c2c-blue);";
            const tReadout = document.createElement("span");
            tReadout.textContent = "0.00 / 0.00";
            tReadout.style.cssText = "font-family:ui-monospace,monospace;font-size:10.5px;color:var(--c2c-sub);min-width:84px;text-align:right;";
            const muteBtn = document.createElement("button");
            muteBtn.type = "button";
            muteBtn.textContent = "🔇 A";
            muteBtn.title = "Toggle audio (unmute A — B stays muted to avoid echo)";
            muteBtn.style.cssText = "background:var(--c2c-bg);border:1px solid var(--c2c-surface0);color:var(--c2c-fg);border-radius:4px;padding:2px 8px;cursor:pointer;font-size:10.5px;";
            // Frame-step buttons for real per-frame A/B comparison.
            const mkStep = (label, title) => {
                const b = document.createElement("button");
                b.type = "button"; b.textContent = label; b.title = title;
                b.style.cssText = "background:var(--c2c-bg);border:1px solid var(--c2c-surface0);color:var(--c2c-fg);border-radius:4px;padding:2px 8px;cursor:pointer;font-family:ui-monospace,monospace;font-size:11px;line-height:1;";
                return b;
            };
            const btnFirst = mkStep("⏮", "Jump to first frame");
            const btnStepB = mkStep("◀ 1f", "Step back 1 frame (key: ,)");
            const btnStepF = mkStep("1f ▶", "Step forward 1 frame (key: .)");
            const btnLast  = mkStep("⏭", "Jump to last frame");
            transport.appendChild(playBtn);
            transport.appendChild(btnFirst);
            transport.appendChild(btnStepB);
            transport.appendChild(btnStepF);
            transport.appendChild(btnLast);
            transport.appendChild(seek);
            transport.appendChild(tReadout);
            transport.appendChild(muteBtn);
            playerHost.appendChild(transport);
            wrap.appendChild(playerHost);

            // Synced-player state — _sp.master is whichever <video> drives time;
            // we always seek both to master.currentTime per RAF tick. Master is
            // picked as whichever has the longer duration (so the shorter one
            // pauses at its end naturally).
            const _sp = { active: false, raf: 0, master: vidA, slave: vidB, fps: 30, rvfcA: 0, rvfcB: 0 };

            const _spDetectFps = () => {
                // Best-effort fps: prefer rVFC-reported processingDuration, else fall back to 30.
                // We can also derive it lazily from successive timestamps.
                return _sp.fps;
            };

            const _spPickMaster = () => {
                const dA = isFinite(vidA.duration) ? vidA.duration : 0;
                const dB = isFinite(vidB.duration) ? vidB.duration : 0;
                if (dB > dA) { _sp.master = vidB; _sp.slave = vidA; }
                else         { _sp.master = vidA; _sp.slave = vidB; }
            };

            const _spLoadSources = () => {
                const a = S.srcA, b = S.srcB;
                if (a && a.kind === "video" && vidA.src !== a.el.currentSrc && a.el.currentSrc) {
                    vidA.src = a.el.currentSrc;
                }
                if (b && b.kind === "video" && vidB.src !== b.el.currentSrc && b.el.currentSrc) {
                    vidB.src = b.el.currentSrc;
                }
            };

            const _spSync = () => {
                // Push master.currentTime onto slave (with rate-matched tolerance)
                if (!_sp.master.duration || !_sp.slave.duration) return;
                const t = _sp.master.currentTime;
                if (Math.abs(_sp.slave.currentTime - t) > 0.05) {
                    try { _sp.slave.currentTime = Math.min(_sp.slave.duration, t); } catch {}
                }
                if (_sp.master.paused !== _sp.slave.paused) {
                    if (_sp.master.paused) _sp.slave.pause();
                    else _sp.slave.play().catch(() => {});
                }
                // Update transport UI
                const dur = _sp.master.duration || 0;
                const fps = _spDetectFps();
                const f = Math.round(t * fps);
                const fEnd = Math.round(dur * fps);
                tReadout.textContent = `f ${f}/${fEnd} · ${t.toFixed(2)}s / ${dur.toFixed(2)}s`;
                seek.value = String(Math.round((dur > 0 ? t / dur : 0) * 10000));
                playBtn.textContent = _sp.master.paused ? "▶" : "❚❚";
            };

            // Frame-step API — exposed for keyboard + button handlers.
            const _spStepFrames = (n) => {
                _spPickMaster();
                if (!_sp.master.duration) return;
                if (!_sp.master.paused) _sp.master.pause();
                if (!_sp.slave.paused)  _sp.slave.pause();
                const fps = _spDetectFps();
                const dt = n / Math.max(1, fps);
                const t = Math.max(0, Math.min((_sp.master.duration || 0) - 1e-3,
                                                _sp.master.currentTime + dt));
                try {
                    _sp.master.currentTime = t;
                    _sp.slave.currentTime  = Math.min(_sp.slave.duration || t, t);
                } catch {}
                _spSync();
            };
            const _spJump = (where) => {
                _spPickMaster();
                if (!_sp.master.duration) return;
                if (!_sp.master.paused) _sp.master.pause();
                if (!_sp.slave.paused)  _sp.slave.pause();
                const t = (where === "end")
                    ? Math.max(0, (_sp.master.duration || 0) - (1 / Math.max(1, _spDetectFps())))
                    : 0;
                try {
                    _sp.master.currentTime = t;
                    _sp.slave.currentTime  = Math.min(_sp.slave.duration || t, t);
                } catch {}
                _spSync();
            };

            // Per-decoded-frame sync via requestVideoFrameCallback (Chromium, Safari 16+).
            const _spInstallRVFC = () => {
                const setup = (vid, key) => {
                    if (!vid.requestVideoFrameCallback) return;
                    const cb = (now, meta) => {
                        if (!_sp.active) return;
                        // Derive an fps estimate from successive expectedDisplayTimes.
                        if (meta && meta.expectedDisplayTime && vid._lastEDT) {
                            const dt = (meta.expectedDisplayTime - vid._lastEDT) / 1000;
                            if (dt > 1e-4 && dt < 1) {
                                const inst = 1 / dt;
                                _sp.fps = _sp.fps ? (_sp.fps * 0.8 + inst * 0.2) : inst;
                            }
                        }
                        if (meta) vid._lastEDT = meta.expectedDisplayTime;
                        if (vid === _sp.master) _spSync();
                        try { vid[key] = vid.requestVideoFrameCallback(cb); } catch {}
                    };
                    try { vid[key] = vid.requestVideoFrameCallback(cb); } catch {}
                };
                setup(vidA, "rvfcA");
                setup(vidB, "rvfcB");
            };
            const _spUninstallRVFC = () => {
                if (vidA.cancelVideoFrameCallback && vidA.rvfcA) {
                    try { vidA.cancelVideoFrameCallback(vidA.rvfcA); } catch {}
                }
                if (vidB.cancelVideoFrameCallback && vidB.rvfcB) {
                    try { vidB.cancelVideoFrameCallback(vidB.rvfcB); } catch {}
                }
                vidA.rvfcA = 0; vidB.rvfcB = 0;
            };

            const _spRaf = () => {
                _sp.raf = 0;
                if (!_sp.active) return;
                _spSync();
                if (!_sp.master.paused) _sp.raf = requestAnimationFrame(_spRaf);
            };
            const _spKick = () => { if (!_sp.raf) _sp.raf = requestAnimationFrame(_spRaf); };

            playBtn.addEventListener("click", () => {
                _spPickMaster();
                if (_sp.master.paused) {
                    _sp.master.play().catch(() => {});
                    _sp.slave.play().catch(() => {});
                    _spKick();
                } else {
                    _sp.master.pause();
                    _sp.slave.pause();
                    _spSync();
                }
            });
            seek.addEventListener("input", () => {
                _spPickMaster();
                const dur = _sp.master.duration || 0;
                const t = (parseInt(seek.value, 10) / 10000) * dur;
                try { _sp.master.currentTime = t; _sp.slave.currentTime = Math.min(_sp.slave.duration || t, t); } catch {}
                _spSync();
            });
            muteBtn.addEventListener("click", () => {
                vidA.muted = !vidA.muted;
                muteBtn.textContent = vidA.muted ? "🔇 A" : "🔊 A";
            });
            btnFirst.addEventListener("click", () => _spJump("start"));
            btnLast .addEventListener("click", () => _spJump("end"));
            btnStepB.addEventListener("click", () => _spStepFrames(-1));
            btnStepF.addEventListener("click", () => _spStepFrames(+1));
            // Make the player div focusable so it can capture keyboard events.
            playerHost.tabIndex = 0;
            playerHost.style.outline = "none";
            playerHost.addEventListener("keydown", (e) => {
                if (!_sp.active) return;
                if (e.key === "," || e.key === "ArrowLeft")  { _spStepFrames(-1); e.preventDefault(); }
                else if (e.key === "." || e.key === "ArrowRight") { _spStepFrames(+1); e.preventDefault(); }
                else if (e.key === "Home") { _spJump("start"); e.preventDefault(); }
                else if (e.key === "End")  { _spJump("end");   e.preventDefault(); }
                else if (e.code === "Space") {
                    playBtn.click();
                    e.preventDefault();
                }
            });
            // Keep the loop running while master is playing.
            vidA.addEventListener("play",  () => _spKick());
            vidB.addEventListener("play",  () => _spKick());
            vidA.addEventListener("pause", () => _spSync());
            vidB.addEventListener("pause", () => _spSync());


            // ── Frame scrubbing for videos ───────────────────
            const applyFrameToVideos = () => {
                const fIdx = Math.max(0, getVal(node, "frame_index", 0) | 0);
                const fps = 24;
                const t = fIdx / fps;
                for (const s of [S.srcA, S.srcB]) {
                    if (s && s.kind === "video" && Math.abs(s.el.currentTime - t) > 0.02) {
                        try { s.el.currentTime = Math.min(s.dur || t, t); } catch {}
                    }
                }
            };
            const seekToFrame = (idx) => {
                const fW = getW(node, "frame_index");
                if (fW) {
                    fW.value = Math.max(0, Math.round(idx));
                    fW.callback?.(fW.value);
                }
                applyFrameToVideos();
                render();
            };

            // ── Render core ──────────────────────────────────
            const luma = (r, g, b) => 0.299 * r + 0.587 * g + 0.114 * b;
            const quantize = (v, bits) => {
                if (bits >= 8) return v;
                const levels = (1 << bits) - 1;
                return Math.round((v / 255) * levels) / Math.max(1, levels) * 255;
            };
            const drawSource = (s, dx, dy, dw, dh) => {
                if (!s || !s.el) return;
                ctx.drawImage(s.el, dx, dy, dw, dh);
            };

            const render = () => {
                const cw = wrap.clientWidth || 400;
                const mode = getVal(node, "mode", "wipe");
                const isLive = LIVE_MODES.has(mode);

                // ── synced_player mode: swap canvas for dual <video> ──
                if (mode === "synced_player") {
                    cvs.style.display = "none";
                    overlay.style.display = "none";
                    hint.style.display = "none";
                    playerHost.style.display = "flex";
                    const wasActive = _sp.active;
                    _sp.active = true;
                    _spLoadSources();
                    _spPickMaster();
                    if (!wasActive) _spInstallRVFC();
                    // If neither A nor B is a video, show a placeholder.
                    const aIsVid = S.srcA && S.srcA.kind === "video";
                    const bIsVid = S.srcB && S.srcB.kind === "video";
                    if (!aIsVid && !bIsVid) {
                        tReadout.textContent = "(no videos)";
                        seek.disabled = true;
                        playBtn.disabled = true;
                        for (const b of [btnFirst, btnLast, btnStepB, btnStepF]) b.disabled = true;
                    } else {
                        seek.disabled = false;
                        playBtn.disabled = false;
                        for (const b of [btnFirst, btnLast, btnStepB, btnStepF]) b.disabled = false;
                    }
                    modeBadge.textContent = mode;
                    return;
                } else if (_sp.active) {
                    // Coming back from synced_player → restore canvas view.
                    _sp.active = false;
                    if (_sp.raf) { cancelAnimationFrame(_sp.raf); _sp.raf = 0; }
                    _spUninstallRVFC();
                    try { vidA.pause(); vidB.pause(); } catch {}
                    playerHost.style.display = "none";
                    cvs.style.display = "block";
                    overlay.style.display = "flex";
                    hint.style.display = "block";
                }

                if (!S.srcA && !S.srcB && !S.serverPreview) {
                    const ch = 240;
                    if (cvs.width !== cw || cvs.height !== ch) { cvs.width = cw; cvs.height = ch; }
                    ctx.fillStyle = C.scrimDark2; ctx.fillRect(0, 0, cw, ch);
                    ctx.fillStyle = "#5a5a78";
                    ctx.font = "12px system-ui,sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Upload A and B to begin (live for wipe/onion/diff/per-channel/false-color/crush)", cw / 2, ch / 2);
                    modeBadge.textContent = mode;
                    liveBadge.textContent = isLive ? "● LIVE" : "○ Queue";
                    liveBadge.style.background = isLive ? "rgba(40,180,80,0.85)" : "rgba(200,140,40,0.85)";
                    return;
                }

                // Server-only mode + we have a server preview: just blit it.
                if (!isLive && S.serverPreview) {
                    const img = S.serverPreview;
                    const ar = img.h / img.w;
                    const ch = Math.max(180, Math.min(720, Math.round(cw * ar)));
                    if (cvs.width !== cw || cvs.height !== ch) { cvs.width = cw; cvs.height = ch; }
                    ctx.clearRect(0, 0, cw, ch);
                    ctx.drawImage(img.el, 0, 0, cw, ch);
                    modeBadge.textContent = mode;
                    liveBadge.textContent = "✓ server";
                    liveBadge.style.background = "rgba(80,140,220,0.85)";
                    return;
                }

                const a = S.srcA, b = S.srcB;
                const refA = a || b, refB = b || a;
                if (!refA) return;
                const baseW = Math.max(refA.w || 1, refB?.w || 1);
                const baseH = Math.max(refA.h || 1, refB?.h || 1);
                const aspect = baseH / baseW;
                const ch = Math.max(180, Math.min(720, Math.round(cw * aspect)));
                if (cvs.width !== cw || cvs.height !== ch) { cvs.width = cw; cvs.height = ch; }

                modeBadge.textContent = mode;
                liveBadge.textContent = isLive ? "● LIVE" : "○ Queue to render";
                liveBadge.style.background = isLive ? "rgba(40,180,80,0.85)" : "rgba(200,140,40,0.85)";

                ctx.clearRect(0, 0, cw, ch);

                if (!isLive) {
                    if (refA) drawSource(refA, 0, 0, cw, ch);
                    ctx.fillStyle = "rgba(0,0,0,0.55)";
                    ctx.fillRect(0, 0, cw, ch);
                    ctx.fillStyle = C.warnTint;
                    ctx.font = "13px system-ui,sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText(`'${mode}' renders on the server — Queue to update`, cw / 2, ch / 2);
                    return;
                }

                if (mode === "wipe" && a && b) {
                    const dx = Math.round(cw * S.wipePos);
                    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, dx, ch); ctx.clip();
                    drawSource(a, 0, 0, cw, ch);
                    ctx.restore();
                    ctx.save(); ctx.beginPath(); ctx.rect(dx, 0, cw - dx, ch); ctx.clip();
                    drawSource(b, 0, 0, cw, ch);
                    ctx.restore();
                    ctx.fillStyle = "rgba(255,230,80,0.95)";
                    ctx.fillRect(dx - 1, 0, 2, ch);
                    ctx.beginPath();
                    ctx.arc(dx, ch / 2, 10, 0, Math.PI * 2);
                    ctx.fillStyle = "rgba(255,230,80,0.95)"; ctx.fill();
                    ctx.fillStyle = C.black; ctx.font = "bold 12px system-ui";
                    ctx.textAlign = "center"; ctx.textBaseline = "middle";
                    ctx.fillText("⇆", dx, ch / 2);
                    ctx.textBaseline = "alphabetic";
                    return;
                }

                if (mode === "onion" && a && b) {
                    drawSource(a, 0, 0, cw, ch);
                    ctx.globalAlpha = S.onionAlpha;
                    drawSource(b, 0, 0, cw, ch);
                    ctx.globalAlpha = 1;
                    return;
                }

                if (mode === "side_by_side" && a && b) {
                    const halfW = cw / 2;
                    drawSource(a, 0, 0, halfW, ch);
                    drawSource(b, halfW, 0, halfW, ch);
                    ctx.fillStyle = "rgba(255,230,80,0.9)";
                    ctx.fillRect(halfW - 1, 0, 2, ch);
                    return;
                }

                if (a && b && (mode === "diff" || mode === "per_channel" || mode === "false_color" || mode === "bit_depth_crush")) {
                    const off = document.createElement("canvas");
                    off.width = cw; off.height = ch;
                    const octx = off.getContext("2d", { willReadFrequently: true });
                    octx.drawImage(a.el, 0, 0, cw, ch);
                    const idA = octx.getImageData(0, 0, cw, ch);
                    octx.clearRect(0, 0, cw, ch);
                    octx.drawImage(b.el, 0, 0, cw, ch);
                    const idB = octx.getImageData(0, 0, cw, ch);
                    const outImg = ctx.createImageData(cw, ch);
                    const pA = idA.data, pB = idB.data, p = outImg.data;
                    const gain = Math.max(1, +getVal(node, "diff_gain", 16));
                    const gamma = Math.max(0.1, +getVal(node, "diff_gamma", 1));
                    const thr = Math.max(0, +getVal(node, "diff_threshold", 0)) * 255;
                    const diffMode = getVal(node, "diff_mode", "absolute");
                    const lutName = getVal(node, "false_color_lut", "turbo");
                    const bits = parseInt(getVal(node, "bit_depth", "32"), 10) || 32;
                    const lut = (mode === "false_color" || mode === "bit_depth_crush") ? _lut(lutName) : null;
                    const invG = 1 / gamma;

                    for (let i = 0; i < pA.length; i += 4) {
                        if (mode === "diff") {
                            let dr, dg, db;
                            if (diffMode === "signed") {
                                dr = Math.min(255, Math.max(0, ((pB[i]   - pA[i])   * 0.5) * gain + 127.5));
                                dg = Math.min(255, Math.max(0, ((pB[i+1] - pA[i+1]) * 0.5) * gain + 127.5));
                                db = Math.min(255, Math.max(0, ((pB[i+2] - pA[i+2]) * 0.5) * gain + 127.5));
                            } else if (diffMode === "luminance") {
                                let d = Math.abs(luma(pA[i], pA[i+1], pA[i+2]) - luma(pB[i], pB[i+1], pB[i+2]));
                                if (d < thr) d = 0;
                                d = Math.min(255, Math.pow(d * gain / 255, invG) * 255);
                                dr = dg = db = d;
                            } else { // absolute
                                let r = Math.abs(pA[i] - pB[i]);
                                let g = Math.abs(pA[i+1] - pB[i+1]);
                                let bb = Math.abs(pA[i+2] - pB[i+2]);
                                if (r < thr) r = 0; if (g < thr) g = 0; if (bb < thr) bb = 0;
                                dr = Math.min(255, Math.pow(r * gain / 255, invG) * 255);
                                dg = Math.min(255, Math.pow(g * gain / 255, invG) * 255);
                                db = Math.min(255, Math.pow(bb * gain / 255, invG) * 255);
                            }
                            p[i] = dr; p[i+1] = dg; p[i+2] = db; p[i+3] = 255;
                        } else if (mode === "per_channel") {
                            p[i]   = Math.min(255, Math.abs(pA[i]   - pB[i])   * gain);
                            p[i+1] = Math.min(255, Math.abs(pA[i+1] - pB[i+1]) * gain);
                            p[i+2] = Math.min(255, Math.abs(pA[i+2] - pB[i+2]) * gain);
                            p[i+3] = 255;
                        } else if (mode === "false_color") {
                            const d = (Math.abs(pA[i] - pB[i]) + Math.abs(pA[i+1] - pB[i+1]) + Math.abs(pA[i+2] - pB[i+2])) / 3;
                            const idx = Math.min(255, Math.max(0, Math.round(d * gain)));
                            p[i] = lut[idx*3]; p[i+1] = lut[idx*3+1]; p[i+2] = lut[idx*3+2]; p[i+3] = 255;
                        } else if (mode === "bit_depth_crush") {
                            const ar = quantize(pA[i],   bits), ag = quantize(pA[i+1], bits), ab = quantize(pA[i+2], bits);
                            const br = quantize(pB[i],   bits), bg = quantize(pB[i+1], bits), bb = quantize(pB[i+2], bits);
                            const d = (Math.abs(ar - br) + Math.abs(ag - bg) + Math.abs(ab - bb)) / 3;
                            const idx = Math.min(255, Math.max(0, Math.round(d * gain)));
                            p[i] = lut[idx*3]; p[i+1] = lut[idx*3+1]; p[i+2] = lut[idx*3+2]; p[i+3] = 255;
                        }
                    }
                    ctx.putImageData(outImg, 0, 0);
                    return;
                }

                // Only one source available
                const only = a || b;
                if (only) drawSource(only, 0, 0, cw, ch);
                if (!a || !b) {
                    ctx.fillStyle = "rgba(0,0,0,0.45)";
                    ctx.fillRect(0, ch - 28, cw, 28);
                    ctx.fillStyle = C.warnTint;
                    ctx.font = "11px system-ui,sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText(a ? "Waiting for B…" : "Waiting for A…", cw / 2, ch - 10);
                }
            };

            // RAF loop while any video is playing OR while drag is active.
            const rafTick = () => {
                S.rafToken = 0;
                const playing = (S.srcA && S.srcA.kind === "video" && !S.srcA.el.paused) ||
                                (S.srcB && S.srcB.kind === "video" && !S.srcB.el.paused);
                render();
                if (playing || S.drag) S.rafToken = requestAnimationFrame(rafTick);
            };
            const kick = () => { if (!S.rafToken) S.rafToken = requestAnimationFrame(rafTick); };

            // ── Pointer interaction ─────────────────────────
            const xy = (e) => {
                const r = cvs.getBoundingClientRect();
                return [
                    (e.clientX - r.left) * (cvs.width / Math.max(1, r.width)),
                    (e.clientY - r.top) * (cvs.height / Math.max(1, r.height)),
                ];
            };
            cvs.addEventListener("pointerdown", (e) => {
                const mode = getVal(node, "mode", "wipe");
                if (mode !== "wipe" && mode !== "onion") { cvs.focus(); return; }
                cvs.setPointerCapture(e.pointerId);
                cvs.focus();
                S.drag = true;
                const [cx] = xy(e);
                const nx = Math.max(0, Math.min(1, cx / cvs.width));
                if (mode === "wipe") { S.wipePos = nx; const w = getW(node, "wipe_position"); if (w) w.value = nx; }
                else                 { S.onionAlpha = nx; const w = getW(node, "onion_alpha"); if (w) w.value = nx; }
                kick();
            });
            cvs.addEventListener("pointermove", (e) => {
                if (!S.drag) return;
                const mode = getVal(node, "mode", "wipe");
                const [cx] = xy(e);
                const nx = Math.max(0, Math.min(1, cx / cvs.width));
                if (mode === "wipe") { S.wipePos = nx; const w = getW(node, "wipe_position"); if (w) w.value = nx; }
                else if (mode === "onion") { S.onionAlpha = nx; const w = getW(node, "onion_alpha"); if (w) w.value = nx; }
                render();
            });
            const endDrag = () => {
                if (!S.drag) return;
                S.drag = false;
                const mode = getVal(node, "mode", "wipe");
                const w = getW(node, mode === "wipe" ? "wipe_position" : "onion_alpha");
                if (w) w.callback?.(w.value);
                render();
            };
            cvs.addEventListener("pointerup", endDrag);
            cvs.addEventListener("pointercancel", endDrag);
            cvs.addEventListener("keydown", (e) => {
                if (e.key === "ArrowLeft")  { seekToFrame((getVal(node, "frame_index", 0) | 0) - 1); e.preventDefault(); }
                else if (e.key === "ArrowRight") { seekToFrame((getVal(node, "frame_index", 0) | 0) + 1); e.preventDefault(); }
                else if (e.key === " ") {
                    for (const s of [S.srcA, S.srcB]) {
                        if (s && s.kind === "video") { if (s.el.paused) s.el.play(); else s.el.pause(); }
                    }
                    kick();
                    e.preventDefault();
                }
            });

            // ── Mount DOM widget + lock size ─────────────────
            node.addDOMWidget("comparer_view", "COMPARER", wrap, { serialize: false });
            // Default size: wide enough so each video gets ~360px in the
            // dual-player view (was 480 → each video ~224px, felt cramped).
            // User can still resize via the corner handle.
            const LOCKED_W = 760, LOCKED_H = 600;
            node.setSize([LOCKED_W, LOCKED_H]);

            // ── Upload buttons (wired to live load) ─────────
            const onUploaded = async (slot, filename) => {
                if (slot === "a") S.srcA = await loadSource(filename);
                else              S.srcB = await loadSource(filename);
                applyFrameToVideos();
                render();
            };
            makeUploadButton(node, "a", "A", onUploaded);
            makeUploadButton(node, "b", "B", onUploaded);

            // ── Hook every relevant widget for instant redraw ─
            const liveHooks = [
                "mode", "wipe_position", "onion_alpha",
                "diff_gain", "diff_gamma", "diff_threshold", "diff_mode",
                "false_color_lut", "bit_depth", "frame_index",
                "file_a", "file_b",
            ];
            const installCallback = (w) => {
                if (!w) return;
                const orig = w.callback;
                w.callback = async (v) => {
                    orig?.call(w, v);
                    if (w.name === "file_a") S.srcA = await loadSource(v);
                    if (w.name === "file_b") S.srcB = await loadSource(v);
                    if (w.name === "wipe_position") S.wipePos = +v;
                    if (w.name === "onion_alpha")   S.onionAlpha = +v;
                    if (w.name === "frame_index")   applyFrameToVideos();
                    render();
                };
            };
            for (const n of liveHooks) installCallback(getW(node, n));

            S.wipePos = +getVal(node, "wipe_position", 0.5);
            S.onionAlpha = +getVal(node, "onion_alpha", 0.5);

            queueMicrotask(async () => {
                const fa = getVal(node, "file_a", "");
                const fb = getVal(node, "file_b", "");
                if (fa) S.srcA = await loadSource(fa);
                if (fb) S.srcB = await loadSource(fb);
                applyFrameToVideos();
                render();
            });

            const ro = new ResizeObserver(() => render());
            ro.observe(wrap);
            node._VC_ro = ro;

            node.setDirtyCanvas(true, true);
        };

        // ── Server-side output (after Queue) — show server preview when mode is server-only.
        const _exec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (msg) {
            _exec?.apply(this, arguments);
            const S = this._VC;
            if (!S) return;
            const previews = msg?.images;
            if (!previews || !previews[0]) return;
            const info = previews[0];
            const url = api.apiURL(
                `/view?filename=${encodeURIComponent(info.filename)}&type=${info.type || "temp"}&subfolder=${encodeURIComponent(info.subfolder || "")}`,
            );
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                S.serverPreview = { kind: "image", el: img, w: img.naturalWidth, h: img.naturalHeight };
                this.setDirtyCanvas(true, true);
                // Trigger a redraw — find the canvas inside the DOM widget.
                try {
                    const wrap = this.widgets?.find((w) => w.type === "COMPARER")?.element
                              || this.widgets?.find((w) => w.name === "comparer_view")?.element;
                    const c = wrap?.querySelector?.("canvas");
                    if (c) {
                        const ctx = c.getContext("2d");
                        const cw = c.width;
                        const ar = img.naturalHeight / img.naturalWidth;
                        const ch = Math.max(180, Math.min(720, Math.round(cw * ar)));
                        if (c.height !== ch) c.height = ch;
                        ctx.clearRect(0, 0, cw, ch);
                        ctx.drawImage(img, 0, 0, cw, ch);
                    }
                } catch {}
            };
            img.src = url;
        };

        const _removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            _removed?.apply(this, arguments);
            try { this._VC_ro?.disconnect?.(); } catch {}
            if (this._VC?.rafToken) cancelAnimationFrame(this._VC.rafToken);
        };
    },
});
