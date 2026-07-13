/**
 * wan_director_timeline.js — interactive timeline authoring UI for
 * WanDirectorC2C.
 *
 * Clean-room implementation. Schema-compatible with the LTX Director
 * (WhatDreamsCost/ComfyUI, GPL-3) so existing user-saved timelines can
 * be loaded, but NO source code was copied. Written from scratch against
 * a behavioural spec extracted by reading the upstream JS for
 * interoperability purposes only.
 *
 * Apache-2.0 © 2025-2026 Code2Collapse (Likhith).
 *
 * Responsibilities:
 *   - Render a dual-track (image/text + audio) canvas timeline
 *   - Mouse drag: select / move / left-resize / right-resize segments
 *   - Right-click context menu: add text seg, add image, add audio, delete
 *   - File drag-drop: image → /upload/image, audio → /upload/audio + decode
 *   - Per-segment prompt textarea + guide-strength slider (image only)
 *   - Ruler with adaptive ticks, frames|seconds display mode
 *   - Playhead scrubbing, play / pause, frame-stepping
 *   - Persistence: writes hidden widgets timeline_data / local_prompts /
 *     segment_lengths / guide_strength on every edit
 *   - Restore from saved timeline_data on graph load
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { C, reducedMotion } from './_c2c_theme.js';
import { reportFailure } from './_c2c_report.js';
import {
    WD_DEFAULT_W,
    capWdNode,
    installWanDirectorPrototype,
    wdEnsureDomWidgetsAttached,
    wdHideWidget,
    wdMaxH,
    wdVueNudge,
} from "./_wan_director_ui.js";

// ── Constants ───────────────────────────────────────────────────────
const RULER_H = 22;
const IMG_TRACK_H = 96;
const AUD_TRACK_H = 48;
const VID_TRACK_H = 76;          // Control-Video track (filmstrip of imported clips)
const PROPS_MIN_H = 88;
const TOOLBAR_H = 28;
const PLAYER_BAR_H = 24;
const PAD = 4;
const MIN_SEG_FRAMES = 6;
const HANDLE_PX = 12;
const WAVE_PEAKS = 200;

// ── v2 automation tracks (LoRA / camera / seed / pose) ──────────────
// The backend (director_node.py schema v2) already parses + validates
// these four arrays and emits them in `tracks_program`; this surfaces
// them in the timeline so the "6-track" program can be authored here.
const V2_LANE_H = 26;
const V2_DEFS = [
    {
        key: "loraSegments", label: "LoRA", color: "#b4befe",
        make: (s, l) => ({ id: nid(), name: "", strength: 1.0, start: s, length: l }),
        summary: (x) => `${x.name || "lora"} · ${(+x.strength || 0).toFixed(2)}`,
    },
    {
        key: "cameraSegments", label: "Cam", color: "#94e2d5",
        make: (s, l) => ({ id: nid(), type: "static", start: s, length: l, params: {} }),
        summary: (x) => x.type || "static",
    },
    {
        key: "seedSegments", label: "Seed", color: "#f9e2af",
        make: (s, l) => ({ id: nid(), seed: 0, mode: "fixed", start: s, length: l }),
        summary: (x) => `${x.seed ?? 0} · ${x.mode || "fixed"}`,
    },
    {
        key: "poseSegments", label: "Pose", color: "#f5c2e7",
        make: (s, l) => ({ id: nid(), poseFile: "", strength: 1.0, interpolation: "linear", start: s, length: l }),
        summary: (x) => `${x.poseFile || "pose"} · ${(+x.strength || 0).toFixed(2)}`,
    },
];
const V2_KEYS = V2_DEFS.map(d => d.key);

// Camera motion presets (LTX-Director-style one-click moves). type must be
// one of the backend's _CAMERA_TYPES (static/pan/zoom/orbit/dolly); params
// is the free-form dict the downstream applier interprets.
const CAMERA_PRESETS = [
    { label: "Static",     type: "static", params: {} },
    { label: "Pan ←",      type: "pan",    params: { direction: "left",  speed: 1.0 } },
    { label: "Pan →",      type: "pan",    params: { direction: "right", speed: 1.0 } },
    { label: "Dolly In",   type: "dolly",  params: { direction: "in",    speed: 1.0 } },
    { label: "Dolly Out",  type: "dolly",  params: { direction: "out",   speed: 1.0 } },
    { label: "Zoom In",    type: "zoom",   params: { direction: "in",    amount: 1.25 } },
    { label: "Zoom Out",   type: "zoom",   params: { direction: "out",   amount: 1.25 } },
    { label: "Orbit ↺",    type: "orbit",  params: { direction: "ccw",   speed: 1.0 } },
    { label: "Orbit ↻",    type: "orbit",  params: { direction: "cw",    speed: 1.0 } },
];
const V2_TOTAL_H = V2_LANE_H * V2_DEFS.length;
const V2_LABEL_PX = 38;

// Left sidebar width. The canvas reserves this via _frameToX/_xToFrame; a
// crisp DOM sidebar (track labels + eye toggles + sub-status pills, LTX-style)
// is overlaid on top of it so the labels render as real text, not canvas paint.
const LANE_X0 = 110;
const TRACK_LABELS = {
    image:          { name: "SCENE",   glyph: "🎬", color: "#89b4fa" },
    audio:          { name: "AUDIO",   glyph: "♪",  color: "#a6e3a1" },
    video:          { name: "CONTROL", glyph: "🎞", color: "#f9e2af" },
    loraSegments:   { name: "LoRA",    glyph: "◆" },
    cameraSegments: { name: "Cam",     glyph: "🎥" },
    seedSegments:   { name: "Seed",    glyph: "⬡" },
    poseSegments:   { name: "Pose",    glyph: "🕺" },
};
const CAMERA_TYPES = ["static", "pan", "zoom", "orbit", "dolly"];
const SEED_MODES = ["fixed", "increment", "random_per_frame"];
const POSE_INTERP = ["nearest", "linear"];

// Total canvas height for all stacked tracks (ruler + image + audio + video +
// the four v2 automation lanes). Single source of truth so a new track can't
// be added in one layout site and forgotten in another.
const TRACKS_CANVAS_H = RULER_H + IMG_TRACK_H + AUD_TRACK_H + VID_TRACK_H + V2_TOTAL_H;
const VID_THUMBS = 8;            // filmstrip thumbnails decoded per video clip

const HIDDEN_NAMES = new Set([
    "timeline_data", "local_prompts", "negative_prompts",
    "segment_lengths", "guide_strength",
]);

// ── Helpers ─────────────────────────────────────────────────────────
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const nid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

function hideWidget(w) {
    wdHideWidget(w);
}

function readWidget(node, name, fallback) {
    const w = node.widgets?.find(x => x.name === name);
    return w ? w.value : fallback;
}
function writeWidget(node, name, value) {
    const w = node.widgets?.find(x => x.name === name);
    if (w) {
        w.value = value;
        w.callback?.(value);
    }
}

function safeParseJSON(s, fallback) {
    if (!s || typeof s !== "string") return fallback;
    try { return JSON.parse(s); } catch { return fallback; }
}

// Adaptive ruler step picker — keep label spacing >= 60 px.
function pickRulerStep(durFrames, fps, pxPerFrame, mode) {
    const secs = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800];
    const frms = [1, 2, 5, 10, 24, 48, 120, 240, 480, 960, 1920];
    const target = 60;
    const arr = (mode === "seconds")
        ? secs.map(s => s * fps)
        : frms;
    for (const f of arr) if (f * pxPerFrame >= target) return f;
    return arr[arr.length - 1];
}

function fmtTime(frames, fps, mode) {
    if (mode === "frames") return `${Math.round(frames)}f`;
    const s = frames / Math.max(1, fps);
    return s >= 10 ? `${s.toFixed(1)}s` : `${s.toFixed(2)}s`;
}

// ── Upload helper using ComfyUI /upload/image and /upload/audio ─────
async function uploadFile(file, type) {
    const fd = new FormData();
    fd.append("image", file);     // ComfyUI's /upload/image accepts "image"
    fd.append("overwrite", "true");
    fd.append("type", "input");
    // BUG FIX (2026-07): ComfyUI has NO /upload/audio route (POST → HTTP 405),
    // so "+ Audio" uploads always failed silently and nothing appeared on the
    // track. /upload/image stores ANY file type into input/ (verified: it 200s
    // for .wav/.mp3 too), so route BOTH audio and video through it.
    try {
        const r = await api.fetchApi("/upload/image", { method: "POST", body: fd });
        if (!r.ok) {
            reportFailure("WanDirector.uploadFile",
                new Error(`upload HTTP ${r.status} for ${file && file.name}`),
                "wan_director_timeline");
            _uploadToast(file, `the server refused the upload (HTTP ${r.status})`);
            return null;
        }
        const j = await r.json();
        // returns { name, subfolder, type }
        return j;
    } catch (e) {
        reportFailure("WanDirector.uploadFile", e, "wan_director_timeline");
        _uploadToast(file, String((e && e.message) || e || "network error"));
        return null;
    }
}

// A failed import must NEVER be silent — that reads as "the button is broken".
function _uploadToast(file, why) {
    try {
        app.extensionManager.toast.add({
            severity: "error",
            summary: "WanDirector — import failed",
            detail: `${file && file.name ? file.name : "file"}: ${why}. ` +
                    "The clip was not added to the timeline.",
            life: 7000,
        });
    } catch (_) { /* toast API absent on very old frontends — console has it */ }
}

// Decode audio to peaks + duration. Returns {durFrames, peaks[]}.
async function decodeAudioPeaks(blob, fps) {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await blob.arrayBuffer();
    const audio = await ac.decodeAudioData(buf);
    const ch = audio.getChannelData(0);
    const peaks = new Array(WAVE_PEAKS).fill(0);
    const step = ch.length / WAVE_PEAKS;
    for (let i = 0; i < WAVE_PEAKS; i++) {
        const s = Math.floor(i * step);
        const e = Math.floor((i + 1) * step);
        let mx = 0;
        for (let j = s; j < e; j++) {
            const v = Math.abs(ch[j]);
            if (v > mx) mx = v;
        }
        peaks[i] = mx;
    }
    const durFrames = Math.round(audio.duration * fps);
    return { durFrames, peaks, audio };
}

// ── TimelineEditor ──────────────────────────────────────────────────
// ── Design system ───────────────────────────────────────────────────
// A dedicated stylesheet (injected once) is what gives LTX Director its crisp,
// "authentic" feel vs ad-hoc inline styles: consistent radii, transitions,
// hover states, floating uppercase labels and sub-status pills. We mirror that
// language here with a neutral-dark palette so the node reads as a finished
// pro tool, not a canvas sketch.
function _ensureWdStyles() {
    if (document.getElementById("wd-timeline-styles")) return;
    const el = document.createElement("style");
    el.id = "wd-timeline-styles";
    el.textContent = `
.wd-root{--wd-bg:#161616;--wd-panel:#1e1e1e;--wd-panel2:#222;--wd-line:#111;
  --wd-line2:#2c2c2c;--wd-fg:#e6e6e6;--wd-dim:#8a8a8a;--wd-dim2:#666;--wd-acc:#5b9dd9;
  font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;color:var(--wd-fg);
  background:var(--wd-bg);border-radius:8px;padding:8px;display:flex;flex-direction:column;
  gap:7px;width:100%;height:100%;box-sizing:border-box;overflow:hidden;min-height:0;}
.wd-toolbar{display:flex;gap:5px;align-items:center;flex-wrap:wrap;flex:0 0 auto;}
.wd-btn{background:var(--wd-panel2);color:var(--wd-fg);border:1px solid var(--wd-line);
  border-radius:5px;padding:5px 11px;font-size:11px;font-weight:500;cursor:pointer;
  display:inline-flex;align-items:center;gap:5px;transition:background .15s ease,border-color .15s ease,transform .05s ease;}
.wd-btn:hover{background:#2e2e2e;border-color:#4a4a4a;}
.wd-btn:active{transform:translateY(1px);}
.wd-btn.on{background:#1c2733;border-color:#2f4a63;color:#cfe6ff;}
.wd-btn-danger:hover{background:#3a1717;border-color:#a44;color:#ffb4b4;}
.wd-btn-icon{padding:5px 8px;font-size:12px;}
.wd-sep{width:1px;align-self:stretch;background:var(--wd-line2);margin:2px 3px;}
.wd-select{background:var(--wd-panel2);color:var(--wd-fg);border:1px solid var(--wd-line);
  border-radius:5px;padding:4px 6px;font-size:11px;cursor:pointer;}
.wd-status{margin-left:auto;font:11px ui-monospace,monospace;color:var(--wd-dim);letter-spacing:.2px;}
.wd-canvas-wrap{position:relative;width:100%;flex:0 0 auto;background:#141414;
  border:1px solid var(--wd-line);border-radius:7px;overflow:hidden;}
.wd-canvas{display:block;width:100%;outline:none;cursor:default;}
/* DOM track sidebar overlaid on the canvas's reserved left column */
.wd-sb{position:absolute;left:0;top:0;height:100%;background:#171717;border-right:1px solid #0b0b0b;
  display:flex;flex-direction:column;z-index:3;overflow:hidden;box-sizing:border-box;}
.wd-sb-ruler{display:flex;align-items:center;padding:0 8px;font-size:9px;color:#666;letter-spacing:.6px;
  border-bottom:1px solid #0b0b0b;box-sizing:border-box;flex:0 0 auto;}
.wd-sb-row{position:relative;display:flex;flex-direction:column;justify-content:center;gap:3px;
  padding:0 7px 0 12px;border-bottom:1px solid #0b0b0b;box-sizing:border-box;flex:0 0 auto;overflow:hidden;}
.wd-sb-row.sm{flex-direction:row;align-items:center;gap:6px;padding:0 6px 0 12px;}
.wd-sb-row::before{content:"";position:absolute;left:0;top:2px;bottom:2px;width:3px;background:var(--acc);opacity:.85;}
.wd-sb-row.muted{opacity:.4;}
.wd-sb-top{display:flex;align-items:center;gap:6px;}
.wd-sb-eye{background:none;border:none;color:var(--acc);cursor:pointer;font-size:12px;padding:0;line-height:1;
  width:16px;text-align:left;flex:0 0 auto;transition:opacity .12s ease;}
.wd-sb-eye:hover{opacity:.7;}
.wd-sb-name{display:flex;align-items:center;gap:5px;font-size:10.5px;font-weight:600;color:#d4d4d4;letter-spacing:.3px;}
.wd-sb-row.sm .wd-sb-name{font-size:9.5px;font-weight:500;}
.wd-sb-g{color:var(--acc);font-size:13px;}
.wd-sb-row.sm .wd-sb-g{font-size:11px;}
.wd-sb-pill{align-self:flex-start;display:inline-flex;align-items:center;font-size:8.5px;font-weight:600;
  padding:1px 7px;border-radius:999px;background:#242424;border:1px solid #303030;color:#9a9a9a;letter-spacing:.2px;
  max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;box-sizing:border-box;}
.wd-playerbar{display:flex;align-items:center;gap:9px;background:var(--wd-panel);
  border:1px solid var(--wd-line);border-radius:7px;padding:6px 10px;flex:0 0 auto;}
.wd-transport{display:flex;align-items:center;gap:3px;}
.wd-tbtn{background:none;border:none;color:#cfcfcf;cursor:pointer;font-size:13px;
  width:26px;height:22px;border-radius:4px;display:inline-flex;align-items:center;justify-content:center;
  transition:background .12s ease,color .12s ease;}
.wd-tbtn:hover{background:#2c2c2c;color:#fff;}
.wd-seek{flex:1 1 auto;accent-color:var(--wd-acc);height:4px;}
.wd-readout{font:11px ui-monospace,monospace;color:var(--wd-dim);white-space:nowrap;letter-spacing:.3px;}
.wd-panel{background:var(--wd-panel);border:1px solid var(--wd-line);border-radius:7px;
  box-sizing:border-box;position:relative;}
.wd-props{display:flex;flex-direction:column;gap:7px;flex:1 1 auto;min-height:0;overflow-y:auto;padding:2px;}
.wd-prompt-wrap{position:relative;width:100%;background:var(--wd-panel);border:1px solid var(--wd-line);
  border-radius:7px;box-sizing:border-box;overflow:hidden;transition:border-color .2s ease;min-height:74px;}
.wd-prompt-wrap:focus-within{border-color:#4d6a86;}
.wd-plabel{position:absolute;top:6px;left:9px;font-size:9px;font-weight:700;color:var(--wd-dim2);
  text-transform:uppercase;letter-spacing:.6px;pointer-events:none;user-select:none;z-index:2;}
.wd-pmeta{position:absolute;top:6px;right:9px;font:9px ui-monospace,monospace;color:var(--wd-dim2);
  pointer-events:none;z-index:2;}
.wd-parea{width:100%;box-sizing:border-box;background:transparent;color:var(--wd-fg);border:none;
  padding:22px 9px 9px;resize:none;font-size:12px;line-height:1.45;outline:none;min-height:74px;}
.wd-parea::placeholder{color:#555;}
.wd-info{background:#191919;color:#bcbcbc;border:1px solid var(--wd-line);border-radius:7px;
  padding:10px 11px;font-size:11.5px;line-height:1.6;}
.wd-info b,.wd-info span{color:#fff;font-weight:600;}
.wd-itag{display:block;font-size:9px;font-weight:700;color:var(--wd-dim2);text-transform:uppercase;
  letter-spacing:.6px;margin-bottom:6px;}
.wd-gsrow{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--wd-dim);}
.wd-gsrow input[type=range]{flex:1 1 auto;accent-color:var(--wd-acc);}
.wd-field{flex:1;min-width:0;background:#171717;color:var(--wd-fg);border:1px solid var(--wd-line);
  border-radius:5px;padding:4px 7px;font-size:11px;}
.wd-field:focus{outline:none;border-color:#4d6a86;}
.wd-menu{position:fixed;z-index:2147483000;background:#1c1c1c;border:1px solid #333;border-radius:9px;
  padding:5px;box-shadow:0 10px 30px rgba(0,0,0,.6);display:flex;flex-direction:column;gap:1px;min-width:176px;}
.wd-menu-head{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:#666;padding:5px 10px 3px;}
.wd-menu-btn{display:flex;align-items:center;gap:9px;background:none;border:none;color:#e6e6e6;
  font:12px ui-sans-serif,system-ui;text-align:left;padding:7px 10px;border-radius:6px;cursor:pointer;
  transition:background .12s ease;width:100%;box-sizing:border-box;}
.wd-menu-btn:hover:not(:disabled){background:#2c2c2c;}
.wd-menu-btn:disabled{opacity:.4;cursor:not-allowed;}
.wd-menu-btn .g{width:16px;text-align:center;flex:0 0 auto;color:#9aa;}
.wd-menu-sep{height:1px;background:#2c2c2c;margin:3px 4px;}
`;
    document.head.appendChild(el);
}

class TimelineEditor {
    constructor(node, container) {
        this.node = node;
        this.container = container;
        this.segments = [];        // {id,type,start,length,prompt,imageFile,imageB64,guideStrength}
        this.audioSegments = [];   // {id,start,length,trimStart,audioDurationFrames,audioFile,fileName,waveformPeaks}
        this.motionSegments = [];  // {id,type:"video",start,length,trimStart,videoFile,fileName,srcDurationFrames}; backend → control_video
        // v2 automation tracks (see V2_DEFS). Each is an array on `this`
        // keyed by its schema name so _segArr() can resolve it generically.
        this.loraSegments = [];
        this.cameraSegments = [];
        this.seedSegments = [];
        this.poseSegments = [];
        this.selection = { type: null, idx: -1 };
        this.playhead = 0;          // frames
        this.playing = false;
        // LTX-Director-style editing state: loop/work region (I/O keys),
        // per-track mutes (persisted), and Shift-to-bypass clip snapping.
        this.inPoint = null;
        this.outPoint = null;
        this.trackMuted = {};
        this._snapDisabled = false;
        this.audioCtx = null;
        this.audioBuffers = new Map(); // file → AudioBuffer
        this.activeSources = [];
        this.playStartedAt = 0;
        this.playStartFrame = 0;
        this.images = new Map();    // url → HTMLImageElement
        this.lastDraw = 0;
        this.zoom = 1.0;
        this.dragState = null;
        this.hoverState = null;
        this.clipboard = null;
        this.suppressCommit = false;
        this._buildDOM();
        this._loadFromWidgets();
        this._wireEvents();
        this._wireWidgetSync();   // live-refresh when duration/fps/variant widgets change
        this.render();
    }

    /** Re-render the timeline whenever a driving litegraph widget changes, so
     *  values like duration_frames / frame_rate / variant update in real time
     *  (the getters are live; this just triggers the repaint). */
    _wireWidgetSync() {
        const names = ["duration_frames", "frame_rate", "duration_seconds",
                       "model_variant", "display_mode"];
        for (const name of names) {
            const w = (this.node.widgets || []).find((x) => x.name === name);
            if (!w || w.__wdSynced) continue;
            const orig = w.callback;
            const self = this;
            w.callback = function (...a) {
                const r = orig ? orig.apply(this, a) : undefined;
                (self.scheduleRender ? self.scheduleRender() : self.render());
                return r;
            };
            w.__wdSynced = true;
        }
    }

    // ── Convenience widget accessors ──
    get fps()      { return parseFloat(readWidget(this.node, "frame_rate", 16)) || 16; }
    get durFrames(){ return parseInt(readWidget(this.node, "duration_frames", 81)) || 81; }
    get displayMode() { return readWidget(this.node, "display_mode", "seconds"); }
    get visualDurFrames() {
        let mx = 0;
        for (const s of this.segments)      mx = Math.max(mx, s.start + s.length);
        for (const s of this.audioSegments) mx = Math.max(mx, s.start + s.length);
        for (const s of this.motionSegments) mx = Math.max(mx, s.start + s.length);
        return Math.max(mx, Math.round(this.durFrames * 1.30));
    }

    // ── DOM ─────────────────────────────────────────────────────────
    _buildDOM() {
        _ensureWdStyles();
        const root = this.container;
        root.classList.add("wd-root");
        root.style.cssText = "width:100%;height:100%;";
        // Toolbar
        const tb = document.createElement("div");
        tb.className = "wd-toolbar";
        const mkBtn = (label, title, extra) => {
            const b = document.createElement("button");
            b.type = "button";
            b.textContent = label;
            b.title = title;
            b.className = "wd-btn" + (extra ? " " + extra : "");
            return b;
        };
        this.btnAddText  = mkBtn("＋ Text",  "Add a text-only segment");
        this.btnAddImage = mkBtn("＋ Image", "Upload an image segment");
        this.btnAddVideo = mkBtn("＋ Video", "Import a video clip onto the Control-Video track (mp4/mov/webm…)");
        this.btnAddAudio = mkBtn("＋ Audio", "Upload an audio segment");
        const sep1 = document.createElement("div"); sep1.className = "wd-sep";
        this.btnSplit    = mkBtn("Split ✂", "Split the selected clip at the playhead (S)");
        this.btnDelete   = mkBtn("Delete",  "Delete selected segment (Delete key)", "wd-btn-danger");
        const sep2 = document.createElement("div"); sep2.className = "wd-sep";
        this.btnPlay     = mkBtn("▶",       "Play / Pause (Space)", "wd-btn-icon");
        this.btnZoomOut  = mkBtn("−",       "Zoom out", "wd-btn-icon");
        this.btnZoomIn   = mkBtn("+",       "Zoom in", "wd-btn-icon");
        this.btnFit      = mkBtn("Fit",     "Fit timeline to view");
        const modeSel = document.createElement("select");
        modeSel.className = "wd-select";
        modeSel.innerHTML = '<option value="seconds">Seconds</option><option value="frames">Frames</option>';
        modeSel.value = this.displayMode;
        modeSel.onchange = () => { writeWidget(this.node, "display_mode", modeSel.value); this.render(); };
        this.modeSel = modeSel;
        const status = document.createElement("span");
        status.className = "wd-status";
        this.statusEl = status;
        for (const el of [this.btnAddText, this.btnAddImage, this.btnAddVideo, this.btnAddAudio, sep1,
                          this.btnSplit, this.btnDelete, sep2,
                          this.btnPlay, this.btnZoomOut, this.btnZoomIn, this.btnFit, modeSel, status]) {
            tb.appendChild(el);
        }
        root.appendChild(tb);

        // Canvas wrap (for drag-drop overlay)
        const wrap = document.createElement("div");
        wrap.className = "wd-canvas-wrap";
        this.cvs = document.createElement("canvas");
        this.cvs.tabIndex = 0;
        this.cvs.className = "wd-canvas";
        this.cvs.style.height = TRACKS_CANVAS_H + "px";
        wrap.appendChild(this.cvs);
        const dropHint = document.createElement("div");
        dropHint.style.cssText = "position:absolute;inset:0;pointer-events:none;display:none;align-items:center;justify-content:center;background:rgba(80,140,220,0.22);color:#fff;font-size:13px;font-weight:600;border:2px dashed #5b9dd9;border-radius:7px;";
        dropHint.textContent = "Drop video, image, or audio file here";
        this.dropHint = dropHint;
        wrap.appendChild(dropHint);
        // Crisp DOM track sidebar (labels + eye toggles + status pills) overlaid
        // on the canvas's reserved left column — the LTX-Director look, real text.
        wrap.appendChild(this._buildSidebarOverlay());
        root.appendChild(wrap);
        this.cvsWrap = wrap;

        // Player bar (seekbar + readout)
        const pbar = document.createElement("div");
        pbar.className = "wd-playerbar";
        this.seek = document.createElement("input");
        this.seek.type = "range";
        this.seek.min = "0";
        this.seek.max = "10000";
        this.seek.value = "0";
        this.seek.className = "wd-seek";
        this.timeReadout = document.createElement("span");
        this.timeReadout.className = "wd-readout";
        this.timeReadout.textContent = "f 0 / 0";
        pbar.appendChild(this.seek);
        pbar.appendChild(this.timeReadout);
        root.appendChild(pbar);

        // Properties panel
        const props = document.createElement("div");
        props.className = "wd-props";
        // Floating-label prompt wrapper (LTX-style): SEGMENT PROMPT label +
        // bounds meta pinned in the corners, textarea fills the box.
        const promptWrap = document.createElement("div");
        promptWrap.className = "wd-prompt-wrap";
        const plabel = document.createElement("div");
        plabel.className = "wd-plabel";
        plabel.textContent = "SEGMENT PROMPT";
        this.propTitle = plabel;
        const pmeta = document.createElement("div");
        pmeta.className = "wd-pmeta";
        this.propBounds = pmeta;
        const ta = document.createElement("textarea");
        ta.className = "wd-parea";
        ta.placeholder = "Select an image or text clip, then describe it here…";
        ta.oninput = () => {
            const seg = this._selSeg();
            if (seg && (seg.type === "image" || seg.type === "text")) {
                seg.prompt = ta.value;
                this.commitChanges();
                this.render();
            }
        };
        this.promptArea = ta;
        promptWrap.appendChild(plabel);
        promptWrap.appendChild(pmeta);
        promptWrap.appendChild(ta);
        props.appendChild(promptWrap);
        this.promptWrap = promptWrap;

        const gsRow = document.createElement("div");
        gsRow.className = "wd-gsrow";
        const gsLabel = document.createElement("span");
        gsLabel.textContent = "Guide strength";
        this.gsSlider = document.createElement("input");
        this.gsSlider.type = "range";
        this.gsSlider.min = "0"; this.gsSlider.max = "200"; this.gsSlider.value = "100";
        this.gsVal = document.createElement("span");
        this.gsVal.style.cssText = "font-family:ui-monospace,monospace;color:#e6e6e6;min-width:40px;text-align:right;";
        this.gsVal.textContent = "1.00";
        this.gsSlider.oninput = () => {
            const v = parseInt(this.gsSlider.value) / 100;
            this.gsVal.textContent = v.toFixed(2);
            const seg = this._selSeg();
            if (seg && seg.type === "image") {
                seg.guideStrength = v;
                this.commitChanges();
            }
        };
        gsRow.appendChild(gsLabel);
        gsRow.appendChild(this.gsSlider);
        gsRow.appendChild(this.gsVal);
        props.appendChild(gsRow);

        // Audio info box (visible only when audio segment selected)
        this.audioInfo = document.createElement("div");
        this.audioInfo.className = "wd-info";
        this.audioInfo.style.display = "none";
        props.appendChild(this.audioInfo);

        // Global prompt (LTX parity) — mirrors the node's global_prompt widget,
        // which we hide from the widget stack so it lives here instead.
        const gWrap = document.createElement("div");
        gWrap.className = "wd-prompt-wrap";
        gWrap.style.flex = "0 0 auto";
        const gLabel = document.createElement("div");
        gLabel.className = "wd-plabel";
        gLabel.textContent = "GLOBAL PROMPT";
        const gTa = document.createElement("textarea");
        gTa.className = "wd-parea";
        gTa.placeholder = "Prompt applied to the whole clip…";
        gTa.value = readWidget(this.node, "global_prompt", "") || "";
        gTa.oninput = () => { writeWidget(this.node, "global_prompt", gTa.value); };
        gWrap.append(gLabel, gTa);
        props.appendChild(gWrap);
        this.globalPromptTa = gTa;
        try {
            const gw = (this.node.widgets || []).find((w) => w.name === "global_prompt");
            if (gw) { gw.hidden = true; gw.computeSize = () => [0, -4]; gw.type = "hidden"; }
        } catch (_) { /* best-effort */ }

        root.appendChild(props);

        // Hidden file inputs
        this.fileImg = document.createElement("input");
        this.fileImg.type = "file"; this.fileImg.accept = "image/*"; this.fileImg.style.display = "none";
        this.fileVid = document.createElement("input");
        this.fileVid.type = "file";
        // Any format: web video + pro/cinema (EXR, DPX, ProRes, DNxHD, MXF/DCP,
        // JPEG2000). image/* is included because EXR/DPX carry no video MIME type.
        this.fileVid.accept = "video/*,image/*,.mp4,.mov,.webm,.mkv,.avi,.m4v,.gif,.mxf,.mts,.m2ts,.exr,.dpx,.hdr,.dnxhd,.prores,.j2k,.jp2,.jpc";
        this.fileVid.style.display = "none";
        this.fileAud = document.createElement("input");
        this.fileAud.type = "file"; this.fileAud.accept = "audio/*"; this.fileAud.style.display = "none";
        root.appendChild(this.fileImg);
        root.appendChild(this.fileVid);
        root.appendChild(this.fileAud);
    }

    _wireEvents() {
        this.btnAddText.onclick  = () => this.addTextSegment();
        this.btnAddImage.onclick = () => this.fileImg.click();
        this.btnAddVideo.onclick = () => this.fileVid.click();
        this.btnAddAudio.onclick = () => this.fileAud.click();
        this.btnSplit.onclick    = () => this.splitSelectedAtPlayhead();
        this.btnDelete.onclick   = () => this.deleteSelected();
        this.btnPlay.onclick     = () => this.togglePlay();
        this.btnZoomIn.onclick   = () => this.setZoom(this.zoom * 1.4);
        this.btnZoomOut.onclick  = () => this.setZoom(this.zoom / 1.4);
        this.btnFit.onclick      = () => this.setZoom(1.0);

        this.fileImg.onchange = async () => {
            for (const f of this.fileImg.files) await this.addImageSegmentFromFile(f);
            this.fileImg.value = "";
        };
        this.fileVid.onchange = async () => {
            for (const f of this.fileVid.files) await this.addVideoSegmentFromFile(f);
            this.fileVid.value = "";
        };
        this.fileAud.onchange = async () => {
            for (const f of this.fileAud.files) await this.addAudioSegmentFromFile(f);
            this.fileAud.value = "";
        };

        // Drag-drop: accept drops ANYWHERE on the timeline (was canvas-only,
        // which made the node feel like it "takes nothing" because users
        // dropped on the props/textbox area). Also stop propagation so
        // ComfyUI's document-level drop handler (which interprets images as
        // workflow files or auto-adds LoadImage nodes) doesn't steal the
        // event. (P1.2 regression fix)
        const dropZone = this.container;
        const isFileDrag = (e) => {
            const t = e.dataTransfer?.types;
            return t && (t.includes("Files") || Array.from(t).includes("Files"));
        };
        dropZone.addEventListener("dragover", (e) => {
            if (!isFileDrag(e)) return;
            e.preventDefault();
            e.stopPropagation();
            this.dropHint.style.display = "flex";
        });
        dropZone.addEventListener("dragleave", (e) => {
            // Only hide when we actually leave the container, not when
            // moving between its children.
            if (e.target === dropZone || !dropZone.contains(e.relatedTarget)) {
                this.dropHint.style.display = "none";
            }
        });
        dropZone.addEventListener("drop", async (e) => {
            if (!isFileDrag(e)) return;
            e.preventDefault();
            e.stopPropagation();
            this.dropHint.style.display = "none";
            const files = Array.from(e.dataTransfer?.files || []);
            let accepted = 0;
            for (const f of files) {
                if (f.type.startsWith("video/") || /\.(mp4|mov|webm|mkv|avi|m4v|mxf|mts|m2ts|exr|dpx|hdr|dnxhd|prores|j2k|jp2|jpc)$/i.test(f.name)) {
                    await this.addVideoSegmentFromFile(f); accepted += 1;
                } else if (f.type.startsWith("image/")) { await this.addImageSegmentFromFile(f); accepted += 1; }
                else if (f.type.startsWith("audio/")) { await this.addAudioSegmentFromFile(f); accepted += 1; }
            }
            if (accepted === 0 && files.length > 0) {
                // Give visible feedback that the drop was received but rejected.
                const orig = this.dropHint.textContent;
                this.dropHint.textContent = `Unsupported (${files.map(f => f.type || "?").join(", ")})`;
                this.dropHint.style.display = "flex";
                this.dropHint.style.background = "rgba(220,80,80,0.25)";
                setTimeout(() => {
                    this.dropHint.style.display = "none";
                    this.dropHint.textContent = orig;
                    this.dropHint.style.background = "rgba(80,140,220,0.25)";
                }, 1600);
            }
        });

        // Canvas mouse
        this.cvs.addEventListener("mousedown", (e) => this._onMouseDown(e));
        this.cvs.addEventListener("mousemove", (e) => this._onMouseMove(e));
        window.addEventListener("mouseup",  (e) => this._onMouseUp(e));
        this.cvs.addEventListener("dblclick",  (e) => this._onDblClick(e));
        this.cvs.addEventListener("contextmenu", (e) => this._onContextMenu(e));
        this.cvs.addEventListener("wheel", (e) => {
            if (e.ctrlKey) {
                e.preventDefault();
                this.setZoom(this.zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
            }
        }, { passive: false });

        // Keyboard (canvas focused)
        this.cvs.addEventListener("keydown", (e) => {
            const sel = this.selection;
            if (e.key === "Delete" || e.key === "Backspace") { this.deleteSelected(); e.preventDefault(); }
            else if ((e.key === "c" || e.key === "C") && (e.ctrlKey || e.metaKey)) {
                if (sel.type && sel.idx >= 0) this._copySegment({ segType: sel.type, idx: sel.idx });
                e.preventDefault();
            }
            else if ((e.key === "v" || e.key === "V") && (e.ctrlKey || e.metaKey)) {
                if (this.clipboard) this._pasteAt(this.playhead, this.clipboard.track);
                e.preventDefault();
            }
            else if (e.key === "d" || e.key === "D") {
                // Duplicate the selected clip right after itself.
                if (sel.type && sel.idx >= 0) {
                    const src = this._segArr(sel.type)[sel.idx];
                    if (src) {
                        this._copySegment({ segType: sel.type, idx: sel.idx });
                        this._pasteAt(src.start + src.length, sel.type);
                    }
                }
                e.preventDefault();
            }
            else if (e.key === "s" || e.key === "S") { this.splitSelectedAtPlayhead(); e.preventDefault(); }
            else if (e.key === "i" || e.key === "I") {
                this.inPoint = (this.inPoint === this.playhead) ? null : this.playhead;
                if (this.inPoint != null && this.outPoint != null && this.outPoint <= this.inPoint) this.outPoint = null;
                this.commitChanges(); e.preventDefault();
            }
            else if (e.key === "o" || e.key === "O") {
                this.outPoint = (this.outPoint === this.playhead) ? null : this.playhead;
                if (this.inPoint != null && this.outPoint != null && this.outPoint <= this.inPoint) this.inPoint = null;
                this.commitChanges(); e.preventDefault();
            }
            else if (e.key === "x" || e.key === "X") {
                if (this.inPoint != null || this.outPoint != null) { this.inPoint = null; this.outPoint = null; this.commitChanges(); }
                e.preventDefault();
            }
            else if (e.key === "+" || e.key === "=") { this.setZoom(this.zoom * 1.4); e.preventDefault(); }
            else if (e.key === "-" || e.key === "_") { this.setZoom(this.zoom / 1.4); e.preventDefault(); }
            else if (e.code === "Space") { this.togglePlay(); e.preventDefault(); }
            else if (e.key === "ArrowLeft")  { this.stepPlayhead(-1); e.preventDefault(); }
            else if (e.key === "ArrowRight") { this.stepPlayhead(+1); e.preventDefault(); }
            else if (e.key === "Home") { this.playhead = 0; this.render(); e.preventDefault(); }
            else if (e.key === "End")  { this.playhead = this.durFrames; this.render(); e.preventDefault(); }
        });

        // Seek bar
        this.seek.oninput = () => {
            const v = parseInt(this.seek.value) / 10000;
            this.playhead = Math.round(v * this.visualDurFrames);
            this.render();
        };

        // Resize observer to redraw on container resize.
        const ro = new ResizeObserver(() => this._renderRaf());
        ro.observe(this.container);
        this._resizeObs = ro;
    }

    // ── Save / load ─────────────────────────────────────────────────
    _loadFromWidgets() {
        this.suppressCommit = true;
        const tl = safeParseJSON(readWidget(this.node, "timeline_data", ""), null);
        if (tl) {
            this.segments = Array.isArray(tl.segments) ? tl.segments : [];
            this.audioSegments = Array.isArray(tl.audioSegments) ? tl.audioSegments : [];
            this.motionSegments = Array.isArray(tl.motionSegments) ? tl.motionSegments : [];
            for (const s of this.segments)      if (!s.id) s.id = nid();
            for (const s of this.audioSegments) if (!s.id) s.id = nid();
            for (const s of this.motionSegments) { if (!s.id) s.id = nid(); s.type = "video"; }
            // v2 automation tracks (load, tolerate missing/legacy v1 docs).
            for (const k of V2_KEYS) {
                this[k] = Array.isArray(tl[k]) ? tl[k] : [];
                for (const s of this[k]) if (!s.id) s.id = nid();
            }
            this.inPoint  = Number.isFinite(tl.inPoint)  ? tl.inPoint  : null;
            this.outPoint = Number.isFinite(tl.outPoint) ? tl.outPoint : null;
            this.trackMuted = (tl.trackMuted && typeof tl.trackMuted === "object") ? tl.trackMuted : {};
        }
        this.suppressCommit = false;
        // Async image preload
        for (const s of this.segments) this._preloadImage(s);
        // Thumbnails for video clips aren't stored in the JSON (would bloat the
        // workflow); regenerate them from the uploaded file on load.
        for (const s of this.motionSegments) this._regenVideoThumbs(s);
    }

    commitChanges() {
        if (this.suppressCommit) return;
        // Bracket the widget-value writes with LiteGraph's change
        // notifications so each persisted timeline state becomes a
        // proper undo step in the native Ctrl+Z stack.
        try { app.graph?.beforeChange?.(); } catch (_) {}
        // timeline_data JSON
        const segOut = this.segments.map(s => ({
            id: s.id, type: s.type, start: s.start, length: s.length,
            prompt: s.prompt || "",
            imageFile: s.imageFile || "",
            imageB64:  s.imageB64  || "",
            guideStrength: typeof s.guideStrength === "number" ? s.guideStrength : 1.0,
        }));
        const audOut = this.audioSegments.map(s => ({
            id: s.id, type: "audio", start: s.start, length: s.length,
            trimStart: s.trimStart || 0,
            audioDurationFrames: s.audioDurationFrames || 0,
            audioFile: s.audioFile || "",
            fileName:  s.fileName  || "",
            waveformPeaks: s.waveformPeaks || [],
        }));
        // Control-Video track: the backend's _build_control_video reads
        // `motionSegments` (videoFile + start/length/trimStart in frames) and
        // decodes them into the control_video output. Thumbnails are NOT
        // persisted (they're regenerated from the file on load) to keep the
        // workflow JSON small.
        const motionOut = this.motionSegments.map(s => ({
            id: s.id, type: "video", start: s.start, length: s.length,
            trimStart: s.trimStart || 0,
            videoFile: s.videoFile || "",
            fileName:  s.fileName  || "",
            srcDurationFrames: s.srcDurationFrames || 0,
            prompt: s.prompt || "",
        }));
        // schema v2: preserve the four automation tracks so they survive an
        // edit (and reach the backend's tracks_program). Without this, any
        // image/audio edit would silently wipe the LoRA/camera/seed/pose data.
        const tlOut = { schema_version: 2, segments: segOut, audioSegments: audOut, motionSegments: motionOut };
        for (const k of V2_KEYS) tlOut[k] = (this[k] || []).map(s => ({ ...s }));
        // Work region + track mutes (additive keys; backend tolerates extras
        // and skips muted tracks — see director_node timeline parse).
        if (this.inPoint  != null) tlOut.inPoint  = this.inPoint;
        if (this.outPoint != null) tlOut.outPoint = this.outPoint;
        if (Object.values(this.trackMuted || {}).some(Boolean)) tlOut.trackMuted = { ...this.trackMuted };
        writeWidget(this.node, "timeline_data", JSON.stringify(tlOut));
        // local_prompts: pipe-delimited, image+text segments sorted by start
        const sorted = [...this.segments].sort((a, b) => a.start - b.start);
        writeWidget(this.node, "local_prompts",
            sorted.map(s => (s.prompt || "").replace(/\|/g, "/")).join(" | "));
        // segment_lengths: comma-delimited lengths clipped at durFrames
        const dur = this.durFrames;
        writeWidget(this.node, "segment_lengths",
            sorted.map(s => Math.max(1, Math.min(s.length, dur))).join(","));
        // guide_strength: image segments only
        writeWidget(this.node, "guide_strength",
            sorted.filter(s => s.type === "image")
                  .map(s => (typeof s.guideStrength === "number" ? s.guideStrength : 1.0).toFixed(2))
                  .join(","));
        try { app.graph?.afterChange?.(); } catch (_) {}
        this.render();
    }

    // ── Segment creation ────────────────────────────────────────────
    _findFreeSlot(length, track) {
        // Try to place at the end of the existing segments, else fall back to 0.
        const arr = this._segArr(track);
        let cursor = 0;
        for (const s of arr.sort((a, b) => a.start - b.start)) {
            cursor = Math.max(cursor, s.start + s.length);
        }
        return cursor;
    }

    addTextSegment() {
        const length = Math.round(this.fps); // 1 second default
        const start = this._findFreeSlot(length, "image");
        const seg = {
            id: nid(), type: "text", start, length,
            prompt: "New text segment", guideStrength: 1.0,
        };
        this.segments.push(seg);
        this.selection = { type: "image", idx: this.segments.length - 1 };
        this.commitChanges();
        this._updatePropsPanel();
        try { this.promptArea?.focus(); } catch (_) {}
    }

    async addImageSegmentFromFile(file) {
        const res = await uploadFile(file, "image");
        if (!res) return;
        const filename = res.subfolder ? `${res.subfolder}/${res.name}` : res.name;
        const length = Math.round(this.fps * 2); // 2 seconds default
        const start = this._findFreeSlot(length, "image");
        const seg = {
            id: nid(), type: "image", start, length,
            prompt: "", imageFile: filename, imageB64: "",
            guideStrength: 1.0,
        };
        this.segments.push(seg);
        this._preloadImage(seg);
        this.selection = { type: "image", idx: this.segments.length - 1 };
        this.commitChanges();
        this._updatePropsPanel();
    }

    async addAudioSegmentFromFile(file) {
        let peaksInfo = null;
        try { peaksInfo = await decodeAudioPeaks(file, this.fps); }
        catch (e) { reportFailure("WanDirector.addAudioSegmentFromFile", e, "wan_director_timeline"); return; }
        const res = await uploadFile(file, "audio");
        if (!res) return;
        const filename = res.subfolder ? `${res.subfolder}/${res.name}` : res.name;
        const length = Math.min(peaksInfo.durFrames, this.durFrames);
        const start = this._findFreeSlot(length, "audio");
        const seg = {
            id: nid(), type: "audio", start, length, trimStart: 0,
            audioDurationFrames: peaksInfo.durFrames,
            audioFile: filename, fileName: file.name,
            waveformPeaks: peaksInfo.peaks,
        };
        this.audioBuffers.set(filename, peaksInfo.audio);
        this.audioSegments.push(seg);
        this.selection = { type: "audio", idx: this.audioSegments.length - 1 };
        this.commitChanges();
        this._updatePropsPanel();
    }

    // ── Control-Video clips ─────────────────────────────────────────
    /** Load a video src (object URL or /view URL) just far enough to read its
     *  duration/size and grab a strip of poster thumbnails. */
    _decodeVideoMeta(src) {
        return new Promise((resolve, reject) => {
            const v = document.createElement("video");
            v.muted = true; v.crossOrigin = "anonymous"; v.preload = "auto";
            v.playsInline = true; v.src = src;
            let done = false;
            const fail = (e) => { if (!done) { done = true; reject(e || new Error("video decode failed")); } };
            v.onerror = () => fail(new Error("video load error"));
            v.onloadedmetadata = async () => {
                const durationSec = isFinite(v.duration) ? v.duration : 0;
                const width = v.videoWidth, height = v.videoHeight;
                let thumbs = [];
                try { thumbs = await this._captureThumbs(v, VID_THUMBS); } catch (_) {}
                if (!done) { done = true; resolve({ durationSec, width, height, thumbs }); }
            };
            setTimeout(() => fail(new Error("video decode timeout")), 15000);
        });
    }

    /** Seek a loaded <video> to `count` evenly-spaced times and snapshot each
     *  frame to a small JPEG <img> for the timeline filmstrip. */
    _captureThumbs(v, count) {
        return new Promise((resolve) => {
            const dur = (isFinite(v.duration) && v.duration > 0) ? v.duration : 0;
            if (!dur) { resolve([]); return; }
            const ch = 64;
            const ar = (v.videoWidth && v.videoHeight) ? v.videoWidth / v.videoHeight : 16 / 9;
            const cw = Math.max(1, Math.round(ch * ar));
            const cvs = document.createElement("canvas");
            cvs.width = cw; cvs.height = ch;
            const ctx = cvs.getContext("2d");
            const times = [];
            for (let k = 0; k < count; k++) times.push(Math.min(dur - 0.01, (dur * (k + 0.5)) / count));
            const thumbs = [];
            let i = 0, guard = null;
            const cleanup = () => { v.removeEventListener("seeked", onSeeked); if (guard) clearTimeout(guard); };
            const arm = () => { guard = setTimeout(() => { cleanup(); resolve(thumbs); }, 4000); };
            const grab = () => {
                try { ctx.drawImage(v, 0, 0, cw, ch); const img = new Image(); img.src = cvs.toDataURL("image/jpeg", 0.6); thumbs.push(img); } catch (_) {}
                i += 1;
                if (i < times.length) { v.currentTime = times[i]; }
                else { cleanup(); resolve(thumbs); }
            };
            const onSeeked = () => { if (guard) { clearTimeout(guard); guard = null; } grab(); arm(); };
            v.addEventListener("seeked", onSeeked);
            v.currentTime = times[0];
            arm();
        });
    }

    /** Probe an uploaded media file on the SERVER (ffmpeg / OpenCV) — works for
     *  ANY format incl. EXR / DPX / ProRes / DNxHD / MXF-DCP / 4K, returning
     *  metadata + downscaled thumbnails the browser could never decode itself.
     *  This is the universal path; the client <video> decode is only a fallback
     *  for web-friendly formats when this endpoint is unavailable. */
    async _probeMediaServer(inputName, opts = {}) {
        try {
            const q = new URLSearchParams({ file: inputName, count: String(VID_THUMBS), height: "80" });
            if (opts.exrView) q.set("exr_view", opts.exrView);
            const r = await api.fetchApi("/wne/media_probe?" + q.toString());
            if (!r.ok) return null;
            const j = await r.json();
            if (!j || !j.ok) return null;
            const thumbs = (j.thumbs || []).map((d) => { const im = new Image(); im.src = d; return im; });
            return { durationSec: j.durationSec || 0, fps: j.fps || 0, frameCount: j.frameCount || 0,
                     width: j.width || 0, height: j.height || 0, thumbs, kind: j.kind };
        } catch (e) { return null; }
    }

    /** Re-decode thumbnails for a saved clip from its uploaded input file. */
    _regenVideoThumbs(seg) {
        if (!seg.videoFile) return;
        this._probeMediaServer(seg.videoFile).then((meta) => {
            if (meta && meta.thumbs && meta.thumbs.length) {
                seg._thumbs = meta.thumbs;
                if (!seg.srcDurationFrames && meta.frameCount > 1) seg.srcDurationFrames = meta.frameCount;
                seg._thumbs.forEach((im) => { im.onload = () => this.render(); });
                this.render();
                return;
            }
            // fallback: client <video> decode for web-friendly formats only
            if (/\.(mp4|webm|m4v|mov|ogg|gif)$/i.test(seg.videoFile)) {
                const parts = seg.videoFile.split("/");
                const name = parts.pop(); const sub = parts.join("/");
                const url = `/view?filename=${encodeURIComponent(name)}&subfolder=${encodeURIComponent(sub)}&type=input`;
                this._decodeVideoMeta(url).then((m) => { seg._thumbs = m.thumbs || []; this.render(); }).catch(() => {});
            }
        });
    }

    async addVideoSegmentFromFile(file) {
        // Upload first (the server probe reads from input/), then decode on the
        // SERVER — that path handles EVERY format (EXR/DCP/ProRes/DNxHD/4K). A
        // client <video> decode is only a fallback for web-friendly formats when
        // the endpoint is missing, so we never hang on an undecodable pro file.
        const res = await uploadFile(file, "image");   // /upload/image stores ANY file in input/
        if (!res) return;
        const filename = res.subfolder ? `${res.subfolder}/${res.name}` : res.name;
        const fps = this.fps;
        let meta = await this._probeMediaServer(filename);
        if (!meta && /\.(mp4|webm|m4v|mov|ogg|gif)$/i.test(file.name)) {
            const objUrl = URL.createObjectURL(file);
            try { meta = await this._decodeVideoMeta(objUrl); } catch (_) {}
            URL.revokeObjectURL(objUrl);
        }
        meta = meta || { durationSec: 0, fps: 0, frameCount: 0, width: 0, height: 0, thumbs: [] };
        // A single still (EXR/DPX with no duration) is holdable for any length, so
        // leave srcDurationFrames=0 (the resize clamp only applies when it's >0).
        const still = (meta.frameCount || 0) <= 1 && (meta.durationSec || 0) <= 0;
        const srcFrames = still ? 0
            : (meta.frameCount > 0 ? meta.frameCount
               : Math.max(1, Math.round((meta.durationSec || 2) * (meta.fps || fps))));
        const length = still
            ? Math.min(this.durFrames, Math.max(MIN_SEG_FRAMES, Math.round(fps * 2)))
            : Math.max(MIN_SEG_FRAMES, Math.min(srcFrames, this.durFrames));
        const start = this._findFreeSlot(length, "video");
        const seg = {
            id: nid(), type: "video", start, length, trimStart: 0,
            videoFile: filename, fileName: file.name,
            srcDurationFrames: srcFrames, srcDurationSec: meta.durationSec || 0,
            srcW: meta.width || 0, srcH: meta.height || 0, kind: meta.kind || "video",
            prompt: "", _thumbs: meta.thumbs || [],
        };
        this.motionSegments.push(seg);
        this.selection = { type: "video", idx: this.motionSegments.length - 1 };
        this.commitChanges();
        this._updatePropsPanel();
        (seg._thumbs || []).forEach((im) => { im.onload = () => this.render(); });
    }

    /** Split the selected clip at the playhead into two clips. For media clips
     *  (video/audio) the second half's source in-point (trimStart) advances so
     *  playback is continuous across the cut. */
    splitSelectedAtPlayhead() {
        const seg = this._selSeg();
        if (!seg) return;
        const { type } = this.selection;
        const ph = this.playhead;
        if (ph <= seg.start + MIN_SEG_FRAMES || ph >= seg.start + seg.length - MIN_SEG_FRAMES) return;
        const arr = this._segArr(type);
        const firstLen = ph - seg.start;
        const secondLen = seg.length - firstLen;
        const clone = { ...seg };                       // shallow → shares _thumbs/_imgUrl for display
        clone.id = nid();
        clone.start = ph;
        clone.length = secondLen;
        if (type === "video" || type === "audio") clone.trimStart = (seg.trimStart || 0) + firstLen;
        seg.length = firstLen;
        arr.push(clone);
        if (type === "image") this._preloadImage(clone);
        this.commitChanges();
        this._updatePropsPanel();
    }

    deleteSelected() {
        const { type, idx } = this.selection;
        if (idx < 0) return;
        const arr = this._segArr(type);
        if (idx < arr.length) {
            arr.splice(idx, 1);
            this.selection = { type: null, idx: -1 };
            this.commitChanges();
            if (this._isV2(type)) this._updateV2Props(); else this._updatePropsPanel();
        }
    }

    _selSeg() {
        const { type, idx } = this.selection;
        if (idx < 0) return null;
        const arr = this._segArr(type);
        return arr[idx] || null;
    }

    _preloadImage(seg) {
        let url = null;
        if (seg.imageFile) {
            const parts = seg.imageFile.split("/");
            const name = parts.pop();
            const sub = parts.join("/");
            url = `/view?filename=${encodeURIComponent(name)}&subfolder=${encodeURIComponent(sub)}&type=input`;
        } else if (seg.imageB64) {
            url = seg.imageB64.startsWith("data:") ? seg.imageB64 : `data:image/png;base64,${seg.imageB64}`;
        }
        if (!url) return;
        if (this.images.has(url)) return;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload  = () => this.render();
        img.onerror = () => {};
        img.src = url;
        this.images.set(url, img);
        seg._imgUrl = url;
    }

    // ── Hit testing & coords ────────────────────────────────────────
    _pxPerFrame() {
        const w = this.cvs.width / (window.devicePixelRatio || 1);
        // Timeline occupies the area to the RIGHT of the label sidebar.
        return (Math.max(1, w - LANE_X0) * this.zoom) / Math.max(1, this.visualDurFrames);
    }
    _frameToX(f) { return f * this._pxPerFrame() + LANE_X0; }
    _xToFrame(x) { return Math.round((x - LANE_X0) / Math.max(1e-3, this._pxPerFrame())); }

    /** Mouse event → canvas CSS-pixel coords, undoing the LiteGraph zoom.
     *  getBoundingClientRect() is post-transform (visual size); offsetWidth is
     *  the unscaled layout size every draw/hit-test math uses. Without this
     *  ratio, clicks at zoom≠1 land offset by 1/zoom — the "timeline doesn't
     *  work unless at 100%" bug. (Same correction as points_bbox_editor.) */
    _evtXY(e) {
        const rect = this.cvs.getBoundingClientRect();
        const kx = (this.cvs.offsetWidth  || rect.width)  / Math.max(1, rect.width);
        const ky = (this.cvs.offsetHeight || rect.height) / Math.max(1, rect.height);
        return [(e.clientX - rect.left) * kx, (e.clientY - rect.top) * ky];
    }

    // Generic segment-array resolver: "image" → segments, "audio" →
    // audioSegments, otherwise a v2 track key (loraSegments, …).
    _segArr(segType) {
        if (segType === "audio") return this.audioSegments;
        if (segType === "image") return this.segments;
        if (segType === "video") return this.motionSegments;
        return this[segType] || [];
    }
    _isV2(segType) { return V2_KEYS.includes(segType); }
    _vidTop() { return RULER_H + IMG_TRACK_H + AUD_TRACK_H; }
    _v2Base() { return RULER_H + IMG_TRACK_H + AUD_TRACK_H + VID_TRACK_H; }
    _v2LaneTop(li) { return this._v2Base() + li * V2_LANE_H; }
    _v2DefForType(segType) { return V2_DEFS.find(d => d.key === segType) || null; }

    _hitTest(mx, my) {
        // Returns {kind, segType, idx, edge}
        if (my < RULER_H) return { kind: "ruler" };
        const v2Base = this._v2Base();
        // Left sidebar (labels + mute toggles) — one row per lane.
        if (mx < LANE_X0) {
            const imgB = RULER_H + IMG_TRACK_H, audB = imgB + AUD_TRACK_H, vidB = audB + VID_TRACK_H;
            let track = null;
            if (my < imgB) track = "image";
            else if (my < audB) track = "audio";
            else if (my < vidB) track = "video";
            else {
                const li = Math.floor((my - v2Base) / V2_LANE_H);
                if (li >= 0 && li < V2_DEFS.length) track = V2_DEFS[li].key;
            }
            if (track) return { kind: "mute", track };
        }
        // v2 automation lanes (below the audio track).
        if (my >= v2Base) {
            const li = Math.max(0, Math.min(V2_DEFS.length - 1, Math.floor((my - v2Base) / V2_LANE_H)));
            const segType = V2_DEFS[li].key, arr = this._segArr(segType);
            for (let i = 0; i < arr.length; i++) {
                const s = arr[i];
                const x1 = this._frameToX(s.start), x2 = this._frameToX(s.start + s.length);
                if (mx >= x1 - HANDLE_PX/2 && mx <= x2 + HANDLE_PX/2) {
                    if (Math.abs(mx - x1) <= HANDLE_PX/2) return { kind: "seg", segType, idx: i, edge: "left" };
                    if (Math.abs(mx - x2) <= HANDLE_PX/2) return { kind: "seg", segType, idx: i, edge: "right" };
                    if (mx >= x1 && mx <= x2) return { kind: "seg", segType, idx: i, edge: "mid" };
                }
            }
            return { kind: "track", track: segType };
        }
        const imgBot = RULER_H + IMG_TRACK_H;
        const audBot = imgBot + AUD_TRACK_H;          // === _vidTop()
        const track = (my < imgBot) ? "image" : (my < audBot) ? "audio" : "video";
        const arr = this._segArr(track);
        for (let i = 0; i < arr.length; i++) {
            const s = arr[i];
            const x1 = this._frameToX(s.start);
            const x2 = this._frameToX(s.start + s.length);
            if (mx >= x1 - HANDLE_PX/2 && mx <= x2 + HANDLE_PX/2) {
                if (Math.abs(mx - x1) <= HANDLE_PX/2) return { kind: "seg", segType: track, idx: i, edge: "left" };
                if (Math.abs(mx - x2) <= HANDLE_PX/2) return { kind: "seg", segType: track, idx: i, edge: "right" };
                if (mx >= x1 && mx <= x2) return { kind: "seg", segType: track, idx: i, edge: "mid" };
            }
        }
        return { kind: "track", track };
    }

    // ── Mouse handlers ──────────────────────────────────────────────
    _onMouseDown(e) {
        this.cvs.focus();
        const [mx, my] = this._evtXY(e);
        if (e.button === 2) return; // handled by contextmenu
        const hit = this._hitTest(mx, my);
        if (hit.kind === "ruler") {
            this.playhead = clamp(this._xToFrame(mx), 0, this.visualDurFrames);
            this.dragState = { type: "playhead" };
            this.render();
            return;
        }
        if (hit.kind === "mute") {
            this.trackMuted[hit.track] = !this.trackMuted[hit.track];
            if (hit.track === "audio" && this.playing) { this._stopAudio(); if (!this.trackMuted.audio) this._startAudio(); }
            this.commitChanges();
            return;
        }
        if (hit.kind === "seg") {
            this.selection = { type: hit.segType === "audio" ? "audio" : hit.segType === "image" ? "image" : hit.segType, idx: hit.idx };
            const arr = this._segArr(hit.segType);
            const seg = arr[hit.idx];
            this.dragState = {
                type: "seg", edge: hit.edge, segType: hit.segType, idx: hit.idx,
                startFrame: this._xToFrame(mx),
                origStart: seg.start, origLength: seg.length, origTrim: seg.trimStart || 0,
            };
            if (this._isV2(hit.segType)) this._updateV2Props(); else this._updatePropsPanel();
            this.render();
            return;
        }
        // Empty-track "+" add-affordance click → open an add MENU (LTX-style):
        // Scene → Text / Image / Paste / Video, Audio → upload, Control → Video.
        if (this._addHints) {
            const img = this._addHints.image, aud = this._addHints.audio, vid = this._addHints.video;
            if (img && Math.hypot(mx - img.cx, my - img.cy) <= img.r) { this._showAddMenu(e.clientX, e.clientY, "image"); return; }
            if (vid && Math.hypot(mx - vid.cx, my - vid.cy) <= vid.r) { this._showAddMenu(e.clientX, e.clientY, "video"); return; }
            if (aud && Math.hypot(mx - aud.cx, my - aud.cy) <= aud.r) { this._showAddMenu(e.clientX, e.clientY, "audio"); return; }
        }
        // Empty track click — clear selection
        this.selection = { type: null, idx: -1 };
        this._updatePropsPanel();
        this.render();
    }

    // ── Add menu (LTX-style "+" popover) ────────────────────────────
    _showAddMenu(clientX, clientY, track) {
        this._dismissAddMenu();
        const menu = document.createElement("div");
        menu.className = "wd-menu";
        const head = document.createElement("div");
        head.className = "wd-menu-head";
        head.textContent = track === "audio" ? "Add to Audio"
                         : track === "video" ? "Add to Control video" : "Add to Scene";
        menu.appendChild(head);
        const mkItem = (label, glyph, fn, opts = {}) => {
            const b = document.createElement("button");
            b.className = "wd-menu-btn";
            b.innerHTML = `<span class="g">${glyph}</span>${label}`;
            if (opts.disabled) { b.disabled = true; if (opts.title) b.title = opts.title; }
            else b.addEventListener("pointerdown", (ev) => {
                ev.preventDefault(); ev.stopPropagation(); this._dismissAddMenu();
                try { fn(); } catch (err) { __c2cReport("wanDirector.addMenu", err); }
            });
            menu.appendChild(b);
        };
        if (track === "image") {
            mkItem("Text segment", "T", () => this.addTextSegment());
            mkItem("Image — upload", "🖼", () => this.fileImg?.click());
            mkItem("Paste image", "📋", () => this._pasteImage());
            const sep = document.createElement("div"); sep.className = "wd-menu-sep"; menu.appendChild(sep);
            mkItem("Video — upload", "🎞", () => this.fileVid?.click());
        } else if (track === "audio") {
            mkItem("Audio — upload", "♪", () => this.fileAud?.click());
        } else if (track === "video") {
            mkItem("Video — upload", "🎞", () => this.fileVid?.click());
        }
        document.body.appendChild(menu);
        // Clamp to viewport.
        const vw = window.innerWidth, vh = window.innerHeight, r = menu.getBoundingClientRect();
        let px = clientX + 4, py = clientY - 6;
        if (px + r.width > vw - 6) px = vw - r.width - 6;
        if (py + r.height > vh - 6) py = vh - r.height - 6;
        menu.style.left = Math.max(6, px) + "px";
        menu.style.top = Math.max(6, py) + "px";
        this._addMenu = menu;
        setTimeout(() => {
            this._addMenuDismiss = (ev) => { if (!menu.contains(ev.target)) this._dismissAddMenu(); };
            document.addEventListener("pointerdown", this._addMenuDismiss, true);
            document.addEventListener("wheel", this._addMenuDismiss, true);
        }, 0);
    }
    _dismissAddMenu() {
        if (this._addMenu) { try { this._addMenu.remove(); } catch (_) {} this._addMenu = null; }
        if (this._addMenuDismiss) {
            document.removeEventListener("pointerdown", this._addMenuDismiss, true);
            document.removeEventListener("wheel", this._addMenuDismiss, true);
            this._addMenuDismiss = null;
        }
    }
    async _pasteImage() {
        try {
            const items = await navigator.clipboard.read();
            for (const item of items) {
                const t = item.types.find((x) => x.startsWith("image/"));
                if (t) {
                    const blob = await item.getType(t);
                    const file = new File([blob], "clipboard.png", { type: blob.type });
                    await this.addImageSegmentFromFile(file);
                    break;
                }
            }
        } catch (_) { _uploadToast?.({ name: "clipboard" }, "Clipboard image paste was blocked by the browser."); }
    }

    _onMouseMove(e) {
        const [mx, my] = this._evtXY(e);
        // Cursor feedback
        if (!this.dragState) {
            const hit = this._hitTest(mx, my);
            if (hit.kind === "seg") {
                this.cvs.style.cursor = (hit.edge === "left" || hit.edge === "right") ? "ew-resize" : "grab";
            } else if (hit.kind === "ruler") {
                this.cvs.style.cursor = "pointer";
            } else {
                this.cvs.style.cursor = "default";
            }
            return;
        }
        const ds = this.dragState;
        if (ds.type === "playhead") {
            this.playhead = clamp(this._xToFrame(mx), 0, this.visualDurFrames);
            this._renderRaf();
            return;
        }
        if (ds.type === "seg") {
            const arr = this._segArr(ds.segType);
            const seg = arr[ds.idx];
            const cur = this._xToFrame(mx);
            const delta = cur - ds.startFrame;
            this._snapDisabled = !!e.shiftKey;
            if (ds.edge === "mid") {
                const raw = Math.max(0, ds.origStart + delta);
                // Snap whichever end actually hits a target; when both do,
                // take the closer snap. (Comparing raw distances alone would
                // always prefer the unsnapped candidate at distance 0.)
                const byStart = this._snapFrame(raw, seg);
                const byEnd = this._snapFrame(raw + seg.length, seg) - seg.length;
                const sHit = byStart !== raw, eHit = byEnd !== raw;
                let v = raw;
                if (sHit && eHit) v = (Math.abs(byStart - raw) <= Math.abs(byEnd - raw)) ? byStart : byEnd;
                else if (sHit) v = byStart;
                else if (eHit) v = byEnd;
                seg.start = Math.max(0, v);
                this._resolveCollisions(ds.segType, ds.idx);
            } else if (ds.edge === "right") {
                const rawEnd = seg.start + Math.max(MIN_SEG_FRAMES, ds.origLength + delta);
                seg.length = Math.max(MIN_SEG_FRAMES, this._snapFrame(rawEnd, seg) - seg.start);
                if (ds.segType === "audio" && seg.audioDurationFrames) {
                    seg.length = Math.min(seg.length, seg.audioDurationFrames - (seg.trimStart || 0));
                } else if (ds.segType === "video" && seg.srcDurationFrames) {
                    seg.length = Math.min(seg.length, seg.srcDurationFrames - (seg.trimStart || 0));
                }
                this._resolveCollisions(ds.segType, ds.idx);
            } else if (ds.edge === "left") {
                const newStart = Math.max(0, this._snapFrame(ds.origStart + delta, seg));
                const trim = newStart - ds.origStart;
                const newLen = ds.origLength - trim;
                if (newLen >= MIN_SEG_FRAMES) {
                    seg.start = newStart;
                    seg.length = newLen;
                    if (ds.segType === "audio" || ds.segType === "video") {
                        seg.trimStart = Math.max(0, ds.origTrim + trim);
                    }
                }
                this._resolveCollisions(ds.segType, ds.idx);
            }
            this._renderRaf();
        }
    }

    _onMouseUp(e) {
        if (this.dragState) {
            const ds = this.dragState;
            this.dragState = null;
            if (ds.type === "seg") this.commitChanges();
        }
    }

    _onDblClick(e) {
        const [mx, my] = this._evtXY(e);
        const hit = this._hitTest(mx, my);
        if (hit.kind === "seg" && this._isV2(hit.segType)) {
            this.selection = { type: hit.segType, idx: hit.idx };
            this._updateV2Props(); this.render();
            try { this.audioInfo.querySelector("input,select")?.focus(); } catch (_) {}
        } else if (hit.kind === "track" && this._isV2(hit.track)) {
            this.addV2Segment(hit.track, this._xToFrame(mx));
        } else if (hit.kind === "seg") {
            this.promptArea.focus();
        }
    }

    _onContextMenu(e) {
        e.preventDefault();
        const [mx, my] = this._evtXY(e);
        const hit = this._hitTest(mx, my);
        const items = [];
        if (hit.kind === "seg" && this._isV2(hit.segType)) {
            const def = this._v2DefForType(hit.segType);
            items.push({ label: `Edit ${def?.label || "segment"}…`, action: () => { this.selection = { type: hit.segType, idx: hit.idx }; this._updateV2Props(); this.render(); } });
            if (hit.segType === "cameraSegments") {
                for (const p of CAMERA_PRESETS) {
                    items.push({ label: `🎥 ${p.label}`, action: () => {
                        const s = this._segArr(hit.segType)[hit.idx];
                        if (!s) return;
                        s.type = p.type;
                        s.params = { ...p.params };
                        this.selection = { type: hit.segType, idx: hit.idx };
                        this.commitChanges();
                        this._updateV2Props();
                    } });
                }
            }
            items.push({ label: "Delete", action: () => { this.selection = { type: hit.segType, idx: hit.idx }; this.deleteSelected(); } });
        } else if (hit.kind === "seg") {
            items.push({ label: "Copy", action: () => this._copySegment(hit) });
            items.push({ label: "Split at playhead", action: () => { this.selection = { type: hit.segType, idx: hit.idx }; this.splitSelectedAtPlayhead(); } });
            if (hit.segType === "image" || hit.segType === "video") {
                items.push({ label: "🎲 Retake span (new seed)", action: () => this._retakeSpan(hit) });
            }
            items.push({ label: "Delete", action: () => { this.selection = { type: hit.segType, idx: hit.idx }; this.deleteSelected(); } });
        } else if (hit.kind === "track" && this._isV2(hit.track)) {
            const def = this._v2DefForType(hit.track);
            items.push({ label: `+ ${def?.label || "segment"} here`, action: () => this.addV2Segment(hit.track, this._xToFrame(mx)) });
        } else if (hit.kind === "track") {
            if (hit.track === "video") {
                items.push({ label: "+ Video…", action: () => this.fileVid.click() });
            } else {
                items.push({ label: "+ Text segment", action: () => this.addTextSegment() });
                items.push({ label: "+ Image…",       action: () => this.fileImg.click() });
                items.push({ label: "+ Video…",       action: () => this.fileVid.click() });
                items.push({ label: "+ Audio…",       action: () => this.fileAud.click() });
            }
            if (this.clipboard) items.push({ label: "Paste", action: () => this._pasteAt(this._xToFrame(mx), hit.track) });
        }
        if (!items.length) return;
        this._showContextMenu(e.clientX, e.clientY, items);
    }

    _showContextMenu(x, y, items) {
        const old = document.getElementById("__wd_ctx");
        if (old) old.remove();
        const m = document.createElement("div");
        m.id = "__wd_ctx";
        m.style.cssText = `
            position:fixed;left:${x}px;top:${y}px;z-index: var(--c2c-z-popover, 9000);
            background:var(--c2c-bg);color:var(--c2c-fg);border:1px solid var(--c2c-surface1);
            border-radius:4px;padding:4px;font:11px system-ui;min-width:140px;
            box-shadow:0 4px 16px rgba(0,0,0,0.5);
        `;
        for (const it of items) {
            const row = document.createElement("div");
            row.textContent = it.label;
            row.style.cssText = "padding:5px 10px;cursor:pointer;border-radius:3px;";
            row.onmouseenter = () => row.style.background = "var(--c2c-surface0)";
            row.onmouseleave = () => row.style.background = "transparent";
            row.onclick = () => { m.remove(); it.action(); };
            m.appendChild(row);
        }
        document.body.appendChild(m);
        const dismiss = (ev) => {
            if (!m.contains(ev.target)) { m.remove(); document.removeEventListener("mousedown", dismiss); }
        };
        setTimeout(() => document.addEventListener("mousedown", dismiss), 0);
    }

    _copySegment(hit) {
        const arr = this._segArr(hit.segType);
        const src = arr[hit.idx];
        if (!src) return;
        // Strip runtime-only underscore keys (decoded thumbs/images aren't JSON-safe).
        const clean = {};
        for (const k in src) if (!k.startsWith("_")) clean[k] = src[k];
        this.clipboard = { track: hit.segType, seg: JSON.parse(JSON.stringify(clean)) };
    }
    _pasteAt(frame, track) {
        if (!this.clipboard || this.clipboard.track !== track) return;
        const seg = JSON.parse(JSON.stringify(this.clipboard.seg));
        seg.id = nid();
        seg.start = Math.max(0, frame);
        const arr = this._segArr(track);
        arr.push(seg);
        this._resolveCollisions(track, arr.length - 1);
        if (track === "video") this._regenVideoThumbs(seg);
        else if (track === "image") this._preloadImage(seg);
        this.commitChanges();
    }

    // Avoid overlap on the same track. Push later segments rightwards.
    _resolveCollisions(track, idx) {
        const arr = this._segArr(track);
        const sorted = [...arr].sort((a, b) => a.start - b.start);
        let cursor = -Infinity;
        for (const s of sorted) {
            if (s.start < cursor) s.start = cursor;
            cursor = s.start + s.length;
        }
    }

    // ── Snapping (LTX-style) ────────────────────────────────────────
    // Snap a frame to the playhead, region marks, clip edges on every track,
    // and the timeline bounds — within an 8-CSS-px tolerance. Hold Shift
    // while dragging to bypass.
    _snapFrame(f, skipSeg) {
        if (this._snapDisabled) return f;
        const tol = 8 / Math.max(1e-3, this._pxPerFrame());
        const targets = [0, this.durFrames, this.playhead];
        if (this.inPoint  != null) targets.push(this.inPoint);
        if (this.outPoint != null) targets.push(this.outPoint);
        for (const t of ["image", "audio", "video", ...V2_KEYS]) {
            for (const s of this._segArr(t)) {
                if (s === skipSeg) continue;
                targets.push(s.start, s.start + s.length);
            }
        }
        let best = f, bd = tol;
        for (const t of targets) {
            const d = Math.abs(t - f);
            if (d < bd) { bd = d; best = t; }
        }
        return best;
    }

    // ── Retake (LTX-2-style): re-roll the seed for one clip's span ──
    _retakeSpan(hit) {
        const seg = this._segArr(hit.segType)[hit.idx];
        if (!seg) return;
        const a = seg.start, b = seg.start + seg.length;
        // Drop seed segments whose midpoint falls inside the span, then lay
        // a fresh fixed seed exactly over it.
        this.seedSegments = this.seedSegments.filter(s => {
            const mid = s.start + s.length / 2;
            return mid < a || mid >= b;
        });
        const fresh = {
            id: nid(), seed: Math.floor(Math.random() * 0x7fffffff),
            mode: "fixed", start: a, length: seg.length,
        };
        this.seedSegments.push(fresh);
        this.selection = { type: "seedSegments", idx: this.seedSegments.indexOf(fresh) };
        this.commitChanges();
        this._updateV2Props();
    }

    // ── v2 automation tracks (LoRA / camera / seed / pose) ───────────
    addV2Segment(key, atFrame) {
        const def = V2_DEFS.find(d => d.key === key);
        if (!def) return;
        const length = Math.round(this.fps); // 1s default
        const start = Math.max(0, atFrame == null ? this._findFreeSlot(length, key) : atFrame);
        const seg = def.make(start, length);
        this[key].push(seg);
        this._resolveCollisions(key, this[key].length - 1);
        this.selection = { type: key, idx: this[key].indexOf(seg) };
        this.commitChanges();
        this._updateV2Props();
    }

    _drawV2Lanes(ctx, cssW) {
        const base = this._v2Base();
        const durX = this._frameToX(this.durFrames);
        for (let li = 0; li < V2_DEFS.length; li++) {
            const def = V2_DEFS[li], top = base + li * V2_LANE_H, arr = this._segArr(def.key);
            // lane background (alternating) + out-of-duration shadow
            ctx.fillStyle = li % 2 ? C.scrimDark : C.bg3;
            ctx.fillRect(0, top, cssW, V2_LANE_H);
            if (durX < cssW) { ctx.fillStyle = "rgba(0,0,0,0.5)"; ctx.fillRect(durX, top, cssW - durX, V2_LANE_H); }
            // (label + mute toggle are drawn by _drawSidebar)
            ctx.strokeStyle = C.surface1Alt; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(0, top + V2_LANE_H - 0.5); ctx.lineTo(cssW, top + V2_LANE_H - 0.5); ctx.stroke();
            // segments
            for (let i = 0; i < arr.length; i++) {
                const s = arr[i];
                const x1 = this._frameToX(s.start), x2 = this._frameToX(s.start + s.length);
                const w = Math.max(3, x2 - x1), y = top + 3, h = V2_LANE_H - 6;
                const selected = this.selection.type === def.key && this.selection.idx === i;
                ctx.fillStyle = def.color;
                ctx.globalAlpha = selected ? 0.95 : 0.7;
                this._roundRect(ctx, x1, y, w, h, 3); ctx.fill();
                ctx.globalAlpha = 1;
                if (selected) { ctx.strokeStyle = C.fg || "#fff"; ctx.lineWidth = 1.5; this._roundRect(ctx, x1, y, w, h, 3); ctx.stroke(); }
                if (w > 26) {
                    ctx.save(); ctx.beginPath(); ctx.rect(x1 + 2, y, w - 4, h); ctx.clip();
                    ctx.fillStyle = C.scrimDark; ctx.font = "8px ui-monospace,monospace"; ctx.textBaseline = "middle"; ctx.textAlign = "left";
                    ctx.fillText(def.summary(s), x1 + 4, y + h / 2);
                    ctx.restore();
                }
            }
        }
    }
    _roundRect(ctx, x, y, w, h, r) {
        r = Math.min(r, w / 2, h / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    }

    _updateV2Props() {
        const seg = this._selSeg(), def = this._v2DefForType(this.selection.type);
        if (!seg || !def) { this._updatePropsPanel(); return; }
        this.propBounds.textContent = `${fmtTime(seg.start, this.fps, this.displayMode)} → ${fmtTime(seg.start + seg.length, this.fps, this.displayMode)} · ${seg.length}f`;
        this.promptWrap.style.display = "none";
        this.gsSlider.parentElement.style.display = "none";
        this.audioInfo.style.display = "block";
        this.audioInfo.innerHTML = "";
        const _vtag = document.createElement("span");
        _vtag.className = "wd-itag";
        _vtag.textContent = `${def.label} · ${def.summary(seg)}`;
        this.audioInfo.appendChild(_vtag);
        this.audioInfo.appendChild(this._buildV2Editor(seg, def));
    }

    _buildV2Editor(seg, def) {
        const wrap = document.createElement("div");
        wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;font:11px system-ui;";
        const row = (label) => {
            const r = document.createElement("div");
            r.style.cssText = "display:flex;align-items:center;gap:6px;";
            const l = document.createElement("span");
            l.textContent = label; l.style.cssText = "width:72px;color:var(--c2c-dim,#9399b2);flex:0 0 auto;";
            r.appendChild(l); return r;
        };
        const styleInput = (el) => { el.style.cssText = "flex:1;min-width:0;background:var(--c2c-surface0,#313244);color:var(--c2c-fg,#cdd6f4);border:1px solid var(--c2c-surface1,#45475a);border-radius:3px;padding:3px 5px;font:11px system-ui;"; return el; };
        const commit = () => { this.commitChanges(); this.propTitle.textContent = `${def.label}: ${def.summary(seg)}`; };
        const textIn = (val, on) => { const i = styleInput(document.createElement("input")); i.type = "text"; i.value = val ?? ""; i.oninput = () => { on(i.value); commit(); }; return i; };
        const numIn = (val, min, max, step, on) => { const i = styleInput(document.createElement("input")); i.type = "number"; i.value = String(val); i.min = min; i.max = max; i.step = step; i.oninput = () => { on(parseFloat(i.value)); commit(); }; return i; };
        const selIn = (val, opts, on) => { const s = styleInput(document.createElement("select")); for (const o of opts) { const op = document.createElement("option"); op.value = o; op.textContent = o; if (o === val) op.selected = true; s.appendChild(op); } s.onchange = () => { on(s.value); commit(); }; return s; };
        const add = (label, input) => { const r = row(label); r.appendChild(input); wrap.appendChild(r); };

        if (def.key === "loraSegments") {
            add("Name", textIn(seg.name, v => seg.name = v));
            add("Strength", numIn(seg.strength ?? 1.0, -2, 2, 0.05, v => seg.strength = isNaN(v) ? 0 : v));
        } else if (def.key === "cameraSegments") {
            add("Type", selIn(seg.type || "static", CAMERA_TYPES, v => seg.type = v));
            const pj = textIn(JSON.stringify(seg.params || {}), v => { try { seg.params = JSON.parse(v || "{}"); } catch (_) {} });
            add("Params", pj);
            const hint = document.createElement("div");
            hint.style.cssText = "font:9px ui-sans-serif;color:var(--c2c-dim,#9399b2);padding-left:78px;";
            hint.textContent = "pan→{dx,dy} · zoom→{from,to} · orbit→{radius,deg} · dolly→{dz}";
            wrap.appendChild(hint);
        } else if (def.key === "seedSegments") {
            add("Seed", numIn(seg.seed ?? 0, 0, 4294967295, 1, v => seg.seed = isNaN(v) ? 0 : Math.round(v)));
            add("Mode", selIn(seg.mode || "fixed", SEED_MODES, v => seg.mode = v));
        } else if (def.key === "poseSegments") {
            add("Pose file", textIn(seg.poseFile, v => seg.poseFile = v));
            add("Strength", numIn(seg.strength ?? 1.0, 0, 2, 0.05, v => seg.strength = isNaN(v) ? 0 : v));
            add("Interp", selIn(seg.interpolation || "linear", POSE_INTERP, v => seg.interpolation = v));
        }
        return wrap;
    }

    // ── Playback ────────────────────────────────────────────────────
    setZoom(z) { this.zoom = clamp(z, 0.1, 32); this.render(); }
    stepPlayhead(n) { this.playhead = clamp(this.playhead + n, 0, this.visualDurFrames); this.render(); }
    togglePlay() {
        if (this.playing) { this._stopAudio(); this.playing = false; this.btnPlay.textContent = "▶"; }
        else {
            // With a work region set, start playback from its in-point when
            // the playhead sits outside the region.
            const li = this.inPoint, lo = this.outPoint;
            if (li != null && lo != null && lo > li && (this.playhead < li || this.playhead >= lo)) this.playhead = li;
            this._startAudio(); this.playing = true; this.btnPlay.textContent = "❚❚"; this._tick();
        }
    }
    _ensureAudioCtx() {
        if (!this.audioCtx) this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        return this.audioCtx;
    }
    async _startAudio() {
        this._stopAudio();
        const ac = this._ensureAudioCtx();
        if (ac.state === "suspended") await ac.resume();
        this.playStartedAt = ac.currentTime;
        this.playStartFrame = this.playhead;
        const fps = this.fps;
        if (this.trackMuted?.audio) return;   // muted lane: silent preview
        for (const seg of this.audioSegments) {
            const buf = this.audioBuffers.get(seg.audioFile);
            if (!buf) continue;
            const segStartT = seg.start / fps;
            const phT = this.playhead / fps;
            const offsetInSeg = phT - segStartT;
            // Skip if playhead is past the segment
            const segEndT = (seg.start + seg.length) / fps;
            if (phT >= segEndT) continue;
            const startDelay = Math.max(0, segStartT - phT);
            const offset = (seg.trimStart || 0) / fps + Math.max(0, offsetInSeg);
            const duration = (seg.length / fps) - Math.max(0, offsetInSeg);
            if (duration <= 0) continue;
            const src = ac.createBufferSource();
            src.buffer = buf;
            src.connect(ac.destination);
            try { src.start(ac.currentTime + startDelay, offset, duration); } catch {}
            this.activeSources.push(src);
        }
    }
    _stopAudio() {
        for (const s of this.activeSources) { try { s.stop(); } catch {} }
        this.activeSources = [];
    }
    _tick() {
        if (!this.playing) return;
        const ac = this.audioCtx;
        if (ac) {
            const dt = ac.currentTime - this.playStartedAt;
            this.playhead = this.playStartFrame + Math.round(dt * this.fps);
            // Work region (I/O points): loop playback inside [in, out).
            const loopIn = this.inPoint, loopOut = this.outPoint;
            if (loopIn != null && loopOut != null && loopOut > loopIn && this.playhead >= loopOut) {
                this._stopAudio();
                this.playhead = loopIn;
                this._startAudio();
            } else if (this.playhead >= this.visualDurFrames) {
                this.playhead = this.visualDurFrames;
                this.togglePlay();
                return;
            }
            this.render();
        }
        // Honour OS prefers-reduced-motion: drop playhead refresh from rAF
        // (~60fps) down to ~10fps via setTimeout. Audio scheduling is
        // unaffected; only the visual playhead update slows.
        if (reducedMotion()) {
            setTimeout(() => this._tick(), 100);
        } else {
            requestAnimationFrame(() => this._tick());
        }
    }

    // ── Properties panel sync ───────────────────────────────────────
    _updatePropsPanel() {
        const seg = this._selSeg();
        if (!seg) {
            this.propTitle.textContent = "SEGMENT PROMPT";
            this.propBounds.textContent = "";
            this.promptWrap.style.display = "";
            this.promptArea.value = "";
            this.promptArea.disabled = true;
            this.gsSlider.parentElement.style.display = "none";
            this.gsSlider.disabled = true;
            this.audioInfo.style.display = "none";
            return;
        }
        const fps = this.fps;
        this.propBounds.textContent = `${fmtTime(seg.start, fps, this.displayMode)} → ${fmtTime(seg.start + seg.length, fps, this.displayMode)} · ${seg.length}f`;
        if (seg.type === "audio") {
            this.propTitle.textContent = `Audio: ${seg.fileName || seg.audioFile || "(unknown)"}`;
            this.promptWrap.style.display = "none";
            this.gsSlider.parentElement.style.display = "none";
            this.audioInfo.style.display = "block";
            const trimIn = (seg.trimStart || 0) / fps;
            const trimOut = ((seg.audioDurationFrames || 0) - (seg.trimStart || 0) - seg.length) / fps;
            this.audioInfo.innerHTML =
                `<span class="wd-itag">Audio clip</span>` +
                `File: <b>${seg.fileName || seg.audioFile}</b><br>` +
                `Source length: ${(seg.audioDurationFrames / fps).toFixed(2)}s<br>` +
                `Output length: ${(seg.length / fps).toFixed(2)}s<br>` +
                `Trim-in: ${trimIn.toFixed(2)}s · Trim-out: ${Math.max(0, trimOut).toFixed(2)}s`;
        } else if (seg.type === "video") {
            this.propTitle.textContent = `Video: ${seg.fileName || seg.videoFile || "(clip)"}`;
            this.promptWrap.style.display = "none";
            this.gsSlider.parentElement.style.display = "none";
            this.audioInfo.style.display = "block";
            const srcSec = (seg.srcDurationFrames || 0) / fps;
            const trimInF = seg.trimStart || 0;
            this.audioInfo.innerHTML =
                `<span class="wd-itag">Control video</span>` +
                `File: <b>${seg.fileName || seg.videoFile}</b><br>` +
                `Source: ${srcSec.toFixed(2)}s · ${seg.srcDurationFrames || "?"}f` +
                    (seg.srcW ? ` · ${seg.srcW}×${seg.srcH}` : "") + `<br>` +
                `On timeline: ${fmtTime(seg.start, fps, this.displayMode)} → ${fmtTime(seg.start + seg.length, fps, this.displayMode)} · ${seg.length}f<br>` +
                `Trim-in: ${trimInF}f → feeds <b>control_video</b>`;
        } else {
            this.propTitle.textContent = (seg.type === "text" ? "TEXT SEGMENT" : "IMAGE SEGMENT");
            this.promptWrap.style.display = "";
            this.promptArea.disabled = false;
            this.promptArea.value = seg.prompt || "";
            this.gsSlider.parentElement.style.display = "flex";
            this.audioInfo.style.display = "none";
            const dis = (seg.type !== "image");
            this.gsSlider.disabled = dis;
            this.gsSlider.value = String(Math.round((seg.guideStrength ?? 1.0) * 100));
            this.gsVal.textContent = (seg.guideStrength ?? 1.0).toFixed(2);
        }
    }

    // ── Render ──────────────────────────────────────────────────────
    render() {
        // Liveness self-clean: if this node was removed WITHOUT onRemoved firing
        // (some graph.clear/workflow-load paths), the dead UI would linger in the
        // DOM and swallow the user's clicks. Detect and self-remove.
        if (this._destroyed) return;
        if (this.node && this.node.graph == null) { this.destroy(); return; }
        const dpr = window.devicePixelRatio || 1;
        const cssW = this.cvs.clientWidth || 600;
        const cssH = TRACKS_CANVAS_H;
        const w = Math.round(cssW * dpr);
        const h = Math.round(cssH * dpr);
        if (this.cvs.width !== w || this.cvs.height !== h) {
            this.cvs.width = w; this.cvs.height = h;
        }
        const ctx = this.cvs.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cssW, cssH);

        // Background tracks
        const vidTop = this._vidTop();
        ctx.fillStyle = C.scrimDark;
        ctx.fillRect(0, 0, cssW, cssH);
        ctx.fillStyle = C.bg3;
        ctx.fillRect(0, RULER_H, cssW, IMG_TRACK_H);
        ctx.fillStyle = C.scrimDark;
        ctx.fillRect(0, RULER_H + IMG_TRACK_H, cssW, AUD_TRACK_H);
        ctx.fillStyle = C.bg3;                                  // Control-Video band (media tone)
        ctx.fillRect(0, vidTop, cssW, VID_TRACK_H);

        // Out-of-duration shadow (covers image + audio + video bands)
        const durX = this._frameToX(this.durFrames);
        if (durX < cssW) {
            ctx.fillStyle = "rgba(0,0,0,0.5)";
            ctx.fillRect(durX, RULER_H, cssW - durX, IMG_TRACK_H + AUD_TRACK_H + VID_TRACK_H);
        }

        // Ruler
        this._drawRuler(ctx, cssW);
        // Segments
        for (let i = 0; i < this.segments.length; i++) this._drawSegment(ctx, this.segments[i], i, "image");
        for (let i = 0; i < this.audioSegments.length; i++) this._drawSegment(ctx, this.audioSegments[i], i, "audio");
        for (let i = 0; i < this.motionSegments.length; i++) this._drawSegment(ctx, this.motionSegments[i], i, "video");
        // Empty-track "+" add-affordances (LTX-style): inviting, clickable hot-spots.
        this._addHints = {};
        if (this.segments.length === 0) this._drawAddHint(ctx, cssW, RULER_H, IMG_TRACK_H, "image");
        if (this.audioSegments.length === 0) this._drawAddHint(ctx, cssW, RULER_H + IMG_TRACK_H, AUD_TRACK_H, "audio");
        if (this.motionSegments.length === 0) this._drawAddHint(ctx, cssW, vidTop, VID_TRACK_H, "video");
        // v2 automation lanes
        this._drawV2Lanes(ctx, cssW);
        // Muted-lane dimming (over the timeline area)
        this._drawMuteDots(ctx, cssW);
        // Sync the DOM sidebar (labels/pills/mute state) — the crisp overlay
        // replaces the old canvas-drawn column.
        this._syncSidebar();
        // Playhead
        this._drawPlayhead(ctx, cssH);

        // Update seekbar + readout
        const vd = this.visualDurFrames;
        this.seek.value = String(Math.round((this.playhead / Math.max(1, vd)) * 10000));
        this.timeReadout.textContent = `f ${this.playhead}/${this.durFrames} · ${fmtTime(this.playhead, this.fps, this.displayMode)}`;
        const v2n = V2_KEYS.reduce((a, k) => a + (this[k]?.length || 0), 0);
        this.statusEl.textContent = `${this.segments.length} clips · ${this.motionSegments.length} video · ${this.audioSegments.length} audio` +
            (v2n ? ` · ${v2n} automation` : "") + ` · ${this.durFrames}f @ ${this.fps}fps`;
    }

    _drawRuler(ctx, cssW) {
        // Ruler bar + a slightly darker sidebar corner so the label column
        // reads as one continuous gutter from the ruler down.
        ctx.fillStyle = "#1c1e28";
        ctx.fillRect(0, 0, cssW, RULER_H);
        ctx.fillStyle = "#16171f";
        ctx.fillRect(0, 0, LANE_X0, RULER_H);
        // Corner label.
        ctx.fillStyle = C.slateMute || "#7f849c";
        ctx.font = "9px ui-sans-serif,system-ui";
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(this.displayMode === "frames" ? "FRAME" : "TIME", 8, RULER_H / 2);

        const pxF = this._pxPerFrame();
        const step = pickRulerStep(this.visualDurFrames, this.fps, pxF, this.displayMode);
        const minorStep = step / (this.displayMode === "frames" ? 5 : 4);
        // Minor ticks — short, faint.
        ctx.strokeStyle = "rgba(255,255,255,0.07)"; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let f = 0; f <= this.visualDurFrames + step; f += minorStep) {
            const x = this._frameToX(f);
            if (x < LANE_X0 - 1) continue;
            if (x > cssW + 2) break;
            ctx.moveTo(x, RULER_H - 4); ctx.lineTo(x, RULER_H);
        }
        ctx.stroke();
        // Major ticks + labels.
        ctx.fillStyle = C.dim || "#a6adc8";
        ctx.font = "10px ui-monospace,monospace";
        ctx.strokeStyle = "rgba(255,255,255,0.16)"; ctx.lineWidth = 1;
        for (let f = 0; f <= this.visualDurFrames + step; f += step) {
            const x = this._frameToX(f);
            if (x < LANE_X0 - 1) continue;
            if (x > cssW + 50) break;
            ctx.beginPath(); ctx.moveTo(x, RULER_H - 8); ctx.lineTo(x, RULER_H); ctx.stroke();
            ctx.fillText(fmtTime(f, this.fps, this.displayMode), x + 3, RULER_H / 2);
        }
        // Crisp baseline under the ruler.
        ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, RULER_H - 0.5); ctx.lineTo(cssW, RULER_H - 0.5); ctx.stroke();

        // Work region (I/O points): amber band on the ruler + edge flags.
        const li = this.inPoint, lo = this.outPoint;
        if (li != null || lo != null) {
            const x1 = this._frameToX(li != null ? li : 0);
            const x2 = this._frameToX(lo != null ? lo : this.visualDurFrames);
            ctx.fillStyle = "rgba(249,226,175,0.22)";
            ctx.fillRect(x1, 0, Math.max(2, x2 - x1), RULER_H);
            ctx.fillStyle = "#f9e2af"; ctx.textAlign = "left";
            if (li != null) { ctx.fillRect(x1, 0, 2, RULER_H); ctx.fillText("I", x1 + 4, RULER_H / 2); }
            if (lo != null) { ctx.fillRect(x2 - 2, 0, 2, RULER_H); ctx.textAlign = "right"; ctx.fillText("O", x2 - 4, RULER_H / 2); ctx.textAlign = "left"; }
        }
    }

    // Build the crisp DOM sidebar (real text labels + eye toggles + pills).
    _buildSidebarOverlay() {
        const sb = document.createElement("div");
        sb.className = "wd-sb";
        sb.style.width = LANE_X0 + "px";
        const rc = document.createElement("div");
        rc.className = "wd-sb-ruler";
        rc.textContent = "TIME";
        rc.style.height = RULER_H + "px";
        sb.appendChild(rc);
        this._sbRows = {};
        const defs = [
            { key: "image", h: IMG_TRACK_H, big: true },
            { key: "audio", h: AUD_TRACK_H, big: true },
            { key: "video", h: VID_TRACK_H, big: true },
            ...V2_DEFS.map((d) => ({ key: d.key, h: V2_LANE_H, big: false, color: d.color })),
        ];
        for (const d of defs) {
            const meta = TRACK_LABELS[d.key] || { name: d.key, glyph: "•" };
            const acc = meta.color || d.color || "#cdd6f4";
            const row = document.createElement("div");
            row.className = "wd-sb-row" + (d.big ? "" : " sm");
            row.style.height = d.h + "px";
            row.style.setProperty("--acc", acc);
            const eye = document.createElement("button");
            eye.type = "button";
            eye.className = "wd-sb-eye";
            eye.title = "Mute / unmute this track";
            eye.onmousedown = (e) => e.stopPropagation();
            eye.onclick = (e) => {
                e.stopPropagation();
                this.trackMuted[d.key] = !this.trackMuted[d.key];
                if (d.key === "audio" && this.playing) { this._stopAudio(); if (!this.trackMuted.audio) this._startAudio(); }
                this._syncSidebar(); this.render();
            };
            const name = document.createElement("div");
            name.className = "wd-sb-name";
            name.innerHTML = `<span class="wd-sb-g">${meta.glyph}</span>${meta.name}`;
            if (d.big) {
                const top = document.createElement("div");
                top.className = "wd-sb-top";
                top.append(eye, name);
                row.appendChild(top);
                const pill = document.createElement("div");
                pill.className = "wd-sb-pill";
                pill.style.display = "none";
                row.appendChild(pill);
                this._sbRows[d.key] = { row, eye, pill, big: true };
            } else {
                row.append(eye, name);
                this._sbRows[d.key] = { row, eye, big: false };
            }
            sb.appendChild(row);
        }
        this._sidebarEl = sb;
        return sb;
    }
    // Reflect mute state + status pills onto the DOM sidebar.
    _syncSidebar() {
        if (!this._sbRows) return;
        for (const key in this._sbRows) {
            const r = this._sbRows[key];
            const muted = !!this.trackMuted?.[key];
            r.row.classList.toggle("muted", muted);
            r.eye.textContent = muted ? "⊘" : (r.big ? "👁" : "●");
        }
        const setPill = (key, text) => {
            const r = this._sbRows[key];
            if (!r || !r.pill) return;
            r.pill.textContent = text || "";
            r.pill.style.display = text ? "inline-flex" : "none";
        };
        const n = (a) => (a && a.length) ? `${a.length} clip${a.length > 1 ? "s" : ""}` : "";
        setPill("image", n(this.segments));
        setPill("audio", readWidget(this.node, "audio_target", "") || "");
        setPill("video", n(this.motionSegments));
    }

    // (legacy) canvas sidebar — replaced by the DOM overlay; kept as a no-op-safe
    // helper in case an old render path calls it.
    _drawSidebar(ctx, cssW, cssH) {
        // Column background + right divider.
        ctx.fillStyle = "#16171f";
        ctx.fillRect(0, RULER_H, LANE_X0, cssH - RULER_H);
        ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(LANE_X0 - 0.5, RULER_H); ctx.lineTo(LANE_X0 - 0.5, cssH); ctx.stroke();

        const rows = [
            { key: "image", top: RULER_H, h: IMG_TRACK_H, big: true },
            { key: "audio", top: RULER_H + IMG_TRACK_H, h: AUD_TRACK_H, big: true },
            { key: "video", top: this._vidTop(), h: VID_TRACK_H, big: true },
            ...V2_DEFS.map((d, i) => ({ key: d.key, top: this._v2LaneTop(i), h: V2_LANE_H, big: false, color: d.color })),
        ];
        for (const r of rows) {
            const meta = TRACK_LABELS[r.key] || { name: r.key, glyph: "•" };
            const color = meta.color || r.color || "#cdd6f4";
            const muted = !!this.trackMuted?.[r.key];
            const cy = r.top + r.h / 2;
            // Accent tab on the left edge of the track (its identity colour).
            ctx.fillStyle = color;
            ctx.globalAlpha = muted ? 0.3 : 0.9;
            ctx.fillRect(0, r.top + 1, 3, r.h - 2);
            ctx.globalAlpha = 1;
            // Mute toggle dot.
            const mx = 13, my = r.big ? r.top + 13 : cy;
            ctx.beginPath(); ctx.arc(mx, my, 4, 0, Math.PI * 2);
            if (muted) {
                ctx.strokeStyle = "rgba(200,205,220,0.8)"; ctx.lineWidth = 1.3; ctx.stroke();
                ctx.beginPath(); ctx.moveTo(mx - 4, my + 4); ctx.lineTo(mx + 4, my - 4); ctx.stroke();
            } else {
                ctx.fillStyle = color; ctx.fill();
            }
            // Glyph + name.
            ctx.globalAlpha = muted ? 0.45 : 1;
            if (r.big) {
                ctx.fillStyle = color; ctx.font = "13px ui-sans-serif,system-ui";
                ctx.textAlign = "left"; ctx.textBaseline = "middle";
                ctx.fillText(meta.glyph, 24, r.top + 13);
                ctx.fillStyle = "#cdd6f4"; ctx.font = "600 10px ui-sans-serif,system-ui";
                ctx.fillText(meta.name, 8, r.top + 30);
            } else {
                ctx.fillStyle = color; ctx.font = "10px ui-sans-serif,system-ui";
                ctx.textAlign = "left"; ctx.textBaseline = "middle";
                ctx.fillText(meta.glyph, 24, cy);
                ctx.fillStyle = "#bac2de"; ctx.font = "9px ui-sans-serif,system-ui";
                ctx.fillText(meta.name, 34, cy);
            }
            ctx.globalAlpha = 1;
        }
    }

    // Dim overlay on muted lanes (the mute TOGGLE + label live in _drawSidebar).
    _drawMuteDots(ctx, cssW) {
        const lanes = [
            { key: "image", top: RULER_H, h: IMG_TRACK_H },
            { key: "audio", top: RULER_H + IMG_TRACK_H, h: AUD_TRACK_H },
            { key: "video", top: this._vidTop(), h: VID_TRACK_H },
            ...V2_DEFS.map((d, i) => ({ key: d.key, top: this._v2LaneTop(i), h: V2_LANE_H })),
        ];
        for (const ln of lanes) {
            if (this.trackMuted?.[ln.key]) {
                ctx.fillStyle = "rgba(0,0,0,0.45)";
                ctx.fillRect(LANE_X0, ln.top, cssW - LANE_X0, ln.h);
            }
        }
    }

    _drawAddHint(ctx, cssW, top, trackH, kind) {
        // LTX-style centered "+" inviting a clip into an empty track. The hot-zone
        // is recorded in this._addHints[kind] for click-to-add in _onMouseDown.
        const durX = this._frameToX(this.durFrames);
        const cx = Math.max(LANE_X0 + 36, Math.min(cssW - 36, (LANE_X0 + Math.min(cssW, durX)) / 2));
        const cy = top + trackH / 2 - 6;
        const r = 14;
        ctx.save();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "rgba(150,165,210,0.55)";
        ctx.fillStyle = "rgba(255,255,255,0.045)";
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy);
        ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6);
        ctx.stroke();
        ctx.fillStyle = "rgba(160,172,205,0.7)";
        ctx.font = "10px ui-sans-serif,system-ui,sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(kind === "audio" ? "+ add audio  ·  or drop a file"
                   : kind === "video" ? "+ add video  ·  or drop a clip"
                                      : "+ add image  ·  or drop a file", cx, cy + r + 11);
        ctx.restore();
        this._addHints[kind] = { cx, cy, r: r + 6 };
    }

    _drawSegment(ctx, seg, idx, track) {
        const x1 = this._frameToX(seg.start);
        const x2 = this._frameToX(seg.start + seg.length);
        let y, h, selType;
        if (track === "audio")      { y = RULER_H + IMG_TRACK_H + 4; h = AUD_TRACK_H - 8; selType = "audio"; }
        else if (track === "video") { y = this._vidTop() + 4;        h = VID_TRACK_H - 8; selType = "video"; }
        else                        { y = RULER_H + 4;               h = IMG_TRACK_H - 8; selType = "image"; }
        const w = Math.max(2, x2 - x1);
        const selected = this.selection.type === selType && this.selection.idx === idx;

        // Body
        if (seg.type === "image") {
            const img = this.images.get(seg._imgUrl);
            ctx.save();
            ctx.beginPath();
            ctx.rect(x1, y, w, h);
            ctx.clip();
            if (img && img.complete && img.naturalWidth > 0) {
                // Fit cover
                const ar = img.naturalWidth / img.naturalHeight;
                const tr = w / h;
                let dx = x1, dy = y, dw = w, dh = h;
                if (ar > tr) { dw = h * ar; dx = x1 + (w - dw) / 2; }
                else { dh = w / ar; dy = y + (h - dh) / 2; }
                ctx.drawImage(img, dx, dy, dw, dh);
            } else {
                ctx.fillStyle = C.surface1Alt;
                ctx.fillRect(x1, y, w, h);
                ctx.fillStyle = C.slateMute;
                ctx.font = "10px ui-sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("(loading)", x1 + w / 2, y + h / 2);
            }
            // Prompt overlay
            if (seg.prompt && w > 40) {
                const oh = Math.max(14, Math.min(28, h * 0.18));
                ctx.fillStyle = "rgba(0,0,0,0.6)";
                ctx.fillRect(x1, y + h - oh, w, oh);
                ctx.fillStyle = C.fg;
                ctx.font = `${Math.round(oh * 0.55)}px ui-sans-serif`;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                let txt = seg.prompt;
                const maxW = w - 8;
                if (ctx.measureText(txt).width > maxW) {
                    while (txt.length > 1 && ctx.measureText(txt + "…").width > maxW) txt = txt.slice(0, -1);
                    txt = txt + "…";
                }
                ctx.fillText(txt, x1 + 4, y + h - oh / 2);
            }
            ctx.restore();
        } else if (seg.type === "text") {
            // Mauve-tint background for text-segment fills. Uses the
            // C.violetBgAlt palette token so the variant switcher
            // (mocha/latte/oled) can recolour it.
            ctx.fillStyle = C.violetBgAlt;
            ctx.fillRect(x1, y, w, h);
            ctx.fillStyle = C.pink;
            ctx.font = "11px ui-sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            const txt = (seg.prompt || "Text").slice(0, 40);
            ctx.fillText(txt, x1 + w / 2, y + h / 2);
        } else if (seg.type === "audio") {
            // Dark-green tint for audio-segment fills via C.okBgDark3
            // so it tracks the user's Catppuccin variant.
            ctx.fillStyle = C.okBgDark3;
            ctx.fillRect(x1, y, w, h);
            // Waveform
            const peaks = seg.waveformPeaks || [];
            if (peaks.length > 0) {
                const totalFrames = seg.audioDurationFrames || seg.length;
                const trim = seg.trimStart || 0;
                const cx = w;
                const mid = y + h / 2;
                ctx.strokeStyle = C.okSoft;
                ctx.lineWidth = 1;
                ctx.beginPath();
                const steps = Math.min(peaks.length, Math.max(20, Math.floor(w / 2)));
                for (let i = 0; i < steps; i++) {
                    const fInSeg = (i / steps) * seg.length;
                    const fAbs = trim + fInSeg;
                    const peakIdx = Math.min(peaks.length - 1, Math.max(0, Math.floor((fAbs / totalFrames) * peaks.length)));
                    const amp = peaks[peakIdx] * (h * 0.45);
                    const px = x1 + (i / steps) * w;
                    ctx.moveTo(px, mid - amp);
                    ctx.lineTo(px, mid + amp);
                }
                ctx.stroke();
            }
            // Filename label
            if (w > 40) {
                ctx.fillStyle = C.fg;
                ctx.font = "10px ui-sans-serif";
                ctx.textAlign = "left";
                ctx.textBaseline = "top";
                let n = seg.fileName || seg.audioFile || "audio";
                const maxW = w - 6;
                if (ctx.measureText(n).width > maxW) {
                    while (n.length > 1 && ctx.measureText(n + "…").width > maxW) n = n.slice(0, -1);
                    n = n + "…";
                }
                ctx.fillText(n, x1 + 3, y + 3);
            }
        } else if (seg.type === "video") {
            ctx.save();
            ctx.beginPath(); ctx.rect(x1, y, w, h); ctx.clip();
            const thumbs = (seg._thumbs || []).filter(t => t.complete && t.naturalWidth > 0);
            if (thumbs.length) {
                // Tile the poster strip across the clip so it reads as a filmstrip;
                // pick which thumb by horizontal position within the clip.
                const ar = thumbs[0].naturalWidth / thumbs[0].naturalHeight || (16 / 9);
                const tileW = Math.max(10, h * ar);
                for (let dx = x1; dx < x1 + w; dx += tileW) {
                    const frac = (dx - x1) / Math.max(1, w);
                    const t = thumbs[Math.min(thumbs.length - 1, Math.floor(frac * thumbs.length))];
                    ctx.drawImage(t, dx, y, tileW, h);
                }
            } else {
                ctx.fillStyle = C.surface1Alt;
                ctx.fillRect(x1, y, w, h);
                ctx.fillStyle = C.slateMute;
                ctx.font = "10px ui-sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText("decoding video…", x1 + w / 2, y + h / 2);
            }
            // Title bar (film glyph + filename)
            if (w > 28) {
                ctx.fillStyle = "rgba(0,0,0,0.55)";
                ctx.fillRect(x1, y, w, 14);
                ctx.fillStyle = C.fg;
                ctx.font = "10px ui-sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
                let n = "🎞 " + (seg.fileName || seg.videoFile || "video");
                const maxW = w - 6;
                if (ctx.measureText(n).width > maxW) {
                    while (n.length > 1 && ctx.measureText(n + "…").width > maxW) n = n.slice(0, -1);
                    n = n + "…";
                }
                ctx.fillText(n, x1 + 3, y + 7);
            }
            // Trim-in indicator (amber underline)
            if (seg.trimStart) {
                ctx.fillStyle = C.amber || "#fbbf24";
                ctx.fillRect(x1, y + h - 3, w, 3);
            }
            ctx.restore();
        }

        // Border. Canvas can't resolve CSS variables on strokeStyle so use
        // the literal hex from the C palette proxy (C.blue / C.panelHi2)
        // instead of `var(--c2c-blue)` — previously stroked transparent.
        ctx.strokeStyle = selected ? C.blue : C.panelHi2;
        ctx.lineWidth = selected ? 2 : 1;
        ctx.strokeRect(x1 + 0.5, y + 0.5, w - 1, h - 1);
    }

    _drawPlayhead(ctx, cssH) {
        const x = this._frameToX(this.playhead);
        ctx.strokeStyle = C.red;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, cssH);
        ctx.stroke();
        // Triangle handle
        ctx.fillStyle = C.red;
        ctx.beginPath();
        ctx.moveTo(x - 6, 0);
        ctx.lineTo(x + 6, 0);
        ctx.lineTo(x, 8);
        ctx.closePath();
        ctx.fill();
    }

    destroy() {
        this._stopAudio();
        this._dismissAddMenu();
        try { this._resizeObs?.disconnect(); } catch {}
        if (this.audioCtx) try { this.audioCtx.close(); } catch {}
        this.audioCtx = null;
        if (this._renderRafId) {
            try { cancelAnimationFrame(this._renderRafId); } catch {}
            this._renderRafId = 0;
        }
        // CRITICAL: remove the UI from the DOM. Without this, workflow loads
        // (graph.clear) leave a DEAD timeline stacked on the page — its "+ Video"
        // button still catches clicks and uploads into a node that no longer
        // exists = the user's "clicking does nothing gets uploaded".
        try { this.container?.remove(); } catch {}
        this._destroyed = true;
    }

    /**
     * Coalesce repeated render() calls into a single rAF tick. Cheap
     * hot-loop callers (mousemove during drag, ResizeObserver bursts,
     * image-load chains) should use this instead of render() directly.
     */
    _renderRaf() {
        if (this._renderRafId) return;
        this._renderRafId = requestAnimationFrame(() => {
            this._renderRafId = 0;
            try { this.render(); } catch (e) {
                reportFailure("WanDirector._renderRaf", e, "wan_director_timeline");
            }
        });
    }
}

// ── Extension registration ──────────────────────────────────────────
// Tracks every live WanDirector node so Ctrl+K palette actions can
// pick the most-recently-created one as the target (matches user
// expectation: "act on the thing I'm currently editing").
const _wdInstances = new Set();

function _wdCapNode(node) {
    capWdNode(node);
}

// Guard: CustomNodePacks ships the same WanDirector extensions. Whichever pack
// loads first registers; the other skips silently (no "already registered" error).
if (!(app.extensions || []).some(e => e?.name === "C2C.WanDirectorTimeline")) app.registerExtension({
    name: "C2C.WanDirectorTimeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WanDirectorC2C") return;
        installWanDirectorPrototype(nodeType);

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origCreated?.apply(this, arguments);
            // Hide the timeline-managed string widgets so they don't clutter the node.
            for (const w of (this.widgets || [])) {
                if (HIDDEN_NAMES.has(w.name)) hideWidget(w);
            }
            wdVueNudge(this);
            // Moderate default width; height capped after DOM widgets register.
            if (!this._wdTimelineSizeApplied) {
                this.size[0] = Math.max(this.size[0] || 0, WD_DEFAULT_W);
                this._wdTimelineSizeApplied = true;
            }
            _wdCapNode(this);
            const host = document.createElement("div");
            // overflow:hidden is load-bearing: when wdMaxH caps the node
            // shorter than the timeline stack, the DOM used to spill out of
            // the node over the graph (the "damaged node" screenshots).
            host.style.cssText = "width:100%;height:100%;overflow:hidden;min-height:0;";
            const widget = this.addDOMWidget("wd_timeline", "wd_timeline", host, {
                getValue: () => "",
                setValue: () => {},
                serialize: false,
            });
            // Clip to the widget slot — without this the timeline DOM spills
            // OUTSIDE the node bounds over neighbouring nodes whenever its
            // content height exceeds the node's layout slot (zoom/restore).
            // Retried because the Vue layer mounts the .dom-widget wrapper
            // AFTER onNodeCreated (a single rAF fires too early).
            const _wdClipTimeline = () => {
                try {
                    const wrap = host.closest?.(".dom-widget");
                    if (wrap && wrap !== host) { wrap.style.overflow = "hidden"; return true; }
                } catch (_) {}
                return false;
            };
            requestAnimationFrame(_wdClipTimeline);
            setTimeout(_wdClipTimeline, 500);
            setTimeout(_wdClipTimeline, 1500);
            const self = this;
            // Full fixed height. Node height is no longer viewport-capped
            // (see _wan_director_ui capWd* — the cap made LiteGraph's slot
            // stack exceed the node and the timeline hung over the graph),
            // so the slot always fits the content by construction.
            widget.computeSize = function (width) {
                return [width, TRACKS_CANVAS_H + TOOLBAR_H + PLAYER_BAR_H + PROPS_MIN_H + 40];
            };
            setTimeout(() => {
                try { self._wdTimeline = new TimelineEditor(self, host); }
                catch (err) { reportFailure("WanDirector.timelineInit", err, "wan_director_timeline"); }
                // Liveness self-clean (idle-safe): a dead-idle timeline never
                // calls render(), so poll cheaply and remove the UI when the
                // node has left the graph without onRemoved firing.
                const _aliveTimer = setInterval(() => {
                    if (self.graph == null) {
                        try { self._wdTimeline?.destroy(); } catch (_) {}
                        try { host.remove(); } catch (_) {}
                        clearInterval(_aliveTimer);
                        return;
                    }
                    // Nodes 2.0 self-heal: re-register DOM widgets Vue lost
                    // during the first-instance mount race (timeline + player).
                    try { wdEnsureDomWidgetsAttached(self); } catch (_) {}
                }, 2000);
                // Re-flow once after all DOM widgets (timeline + player) have
                // registered so the player's computeSize sees a stable width
                // and the node height fits both DOM widgets. Preserve the
                // 820-min width clamp from above.
                try {
                    _wdCapNode(self);
                    const cs = self.computeSize();
                    self.setSize([
                        Math.max(cs[0] || 0, WD_DEFAULT_W),
                        Math.min(cs[1] || wdMaxH(), wdMaxH()),
                    ]);
                    _wdCapNode(self);
                } catch {}
                self.setDirtyCanvas?.(true, true);
            }, 0);
            _wdInstances.add(self);
            const _capLoop = () => { _wdCapNode(self); };
            setTimeout(_capLoop, 0);
            setTimeout(_capLoop, 200);
            setTimeout(_capLoop, 800);
            return r;
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = origConfigure?.apply(this, arguments);
            setTimeout(() => {
                _wdCapNode(this);
                // Workflows SAVED while the old viewport-height cap was live
                // carry the truncated size (e.g. 640x879) in their JSON, and
                // LiteGraph keeps the saved size on load — so the timeline
                // kept hanging out of the node for existing workflows even
                // after the cap was removed. Height is dictated by the
                // widget stack for this node; re-assert it (keep user width).
                try {
                    const need = this.computeSize();
                    if (Array.isArray(need) && this.size[1] < need[1] - 4) {
                        this.setSize([Math.max(this.size[0], need[0]), need[1]]);
                    }
                } catch (_) {}
                if (this._wdTimeline) {
                    this._wdTimeline._loadFromWidgets();
                    this._wdTimeline._updatePropsPanel();
                    this._wdTimeline.render();
                }
            }, 0);
            return r;
        };

        const origRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try { this._wdTimeline?.destroy(); } catch {}
            _wdInstances.delete(this);
            return origRemoved?.apply(this, arguments);
        };
    },
});

// ── Ctrl+K command-palette actions ──────────────────────────────────
// Registered via the global window.__C2C_ACTIONS__ contract from
// _c2c_actions.js (loaded by c2c_command_palette.js). The palette is
// disabled if no WanDirector node is on the graph.
function _wdActive() {
    // Most-recently-added instance wins; the Set preserves insertion order.
    let last = null;
    for (const n of _wdInstances) last = n;
    return last?._wdTimeline || null;
}
function _wdRegisterActions() {
    const reg = window.__C2C_ACTIONS__?.register;
    if (typeof reg !== "function") return;
    const enabled = () => _wdInstances.size > 0;
    const actions = [
        { id: "c2c.wanDirector.playToggle",   title: "WanDirector: Play / Pause",        icon: "▶︎", keywords: ["timeline","play","pause","preview"],     run: () => _wdActive()?.togglePlay()        },
        { id: "c2c.wanDirector.deleteSel",    title: "WanDirector: Delete selected segment", icon: "✕", keywords: ["timeline","remove","delete"],         run: () => _wdActive()?.deleteSelected()    },
        { id: "c2c.wanDirector.addText",      title: "WanDirector: Add text segment",     icon: "T",  keywords: ["timeline","text","prompt"],            run: () => _wdActive()?.addTextSegment()    },
        { id: "c2c.wanDirector.addImage",     title: "WanDirector: Add image segment…",   icon: "🖼", keywords: ["timeline","image","picture","upload"], run: () => _wdActive()?.fileImg?.click()    },
        { id: "c2c.wanDirector.addAudio",     title: "WanDirector: Add audio segment…",   icon: "♪", keywords: ["timeline","audio","sound","music","upload"], run: () => _wdActive()?.fileAud?.click() },
    ];
    for (const a of actions) {
        try {
            reg({ ...a, kind: "command", scope: "graph", enabled });
        } catch (e) {
            reportFailure("WanDirector.registerAction:" + a.id, e, "wan_director_timeline");
        }
    }
}
// Defer registration so the palette extension has time to expose
// window.__C2C_ACTIONS__. Two passes in case load order differs.
setTimeout(_wdRegisterActions, 0);
setTimeout(_wdRegisterActions, 1000);
