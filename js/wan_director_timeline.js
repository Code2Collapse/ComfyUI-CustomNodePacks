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

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { C, reducedMotion } from './_c2c_theme.js';
import { reportFailure } from './_c2c_report.js';
import {
    WD_DEFAULT_W,
    capWdNode,
    installWanDirectorPrototype,
    wdHideWidget,
    wdMaxH,
} from "./_wan_director_ui.js";

// ── Constants ───────────────────────────────────────────────────────
const RULER_H = 22;
const IMG_TRACK_H = 96;
const AUD_TRACK_H = 48;
const PROPS_MIN_H = 88;
const TOOLBAR_H = 28;
const PLAYER_BAR_H = 24;
const PAD = 4;
const MIN_SEG_FRAMES = 6;
const HANDLE_PX = 12;
const WAVE_PEAKS = 200;

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
    const url = type === "audio" ? "/upload/audio" : "/upload/image";
    try {
        const r = await api.fetchApi(url, { method: "POST", body: fd });
        if (!r.ok) return null;
        const j = await r.json();
        // returns { name, subfolder, type }
        return j;
    } catch (e) {
        reportFailure("WanDirector.uploadFile", e, "wan_director_timeline");
        return null;
    }
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
class TimelineEditor {
    constructor(node, container) {
        this.node = node;
        this.container = container;
        this.segments = [];        // {id,type,start,length,prompt,imageFile,imageB64,guideStrength}
        this.audioSegments = [];   // {id,start,length,trimStart,audioDurationFrames,audioFile,fileName,waveformPeaks}
        this.selection = { type: null, idx: -1 };
        this.playhead = 0;          // frames
        this.playing = false;
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
        this.render();
    }

    // ── Convenience widget accessors ──
    get fps()      { return parseFloat(readWidget(this.node, "frame_rate", 16)) || 16; }
    get durFrames(){ return parseInt(readWidget(this.node, "duration_frames", 81)) || 81; }
    get displayMode() { return readWidget(this.node, "display_mode", "seconds"); }
    get visualDurFrames() {
        let mx = 0;
        for (const s of this.segments)      mx = Math.max(mx, s.start + s.length);
        for (const s of this.audioSegments) mx = Math.max(mx, s.start + s.length);
        return Math.max(mx, Math.round(this.durFrames * 1.30));
    }

    // ── DOM ─────────────────────────────────────────────────────────
    _buildDOM() {
        const root = this.container;
        root.classList.add("wd-timeline-root");
        root.style.cssText = `
            display:flex;flex-direction:column;gap:4px;width:100%;height:100%;
            box-sizing:border-box;font-family:ui-sans-serif,system-ui,sans-serif;
            color:var(--c2c-gray100);background:var(--c2c-scrimDark7);border-radius:6px;padding:6px;
        `;
        // Toolbar
        const tb = document.createElement("div");
        tb.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:wrap;height:auto;";
        const mkBtn = (label, title) => {
            const b = document.createElement("button");
            b.type = "button";
            b.textContent = label;
            b.title = title;
            b.style.cssText = `
                background:var(--c2c-panelBg);color:var(--c2c-fg);border:1px solid var(--c2c-surface1Alt);
                border-radius:4px;padding:4px 10px;font-size:11px;cursor:pointer;
            `;
            b.onmouseenter = () => { b.style.background = "var(--c2c-surface1Alt)"; };
            b.onmouseleave = () => { b.style.background = "var(--c2c-panelBg)"; };
            return b;
        };
        this.btnAddText  = mkBtn("+ Text",  "Add a text-only segment");
        this.btnAddImage = mkBtn("+ Image", "Upload an image segment");
        this.btnAddAudio = mkBtn("+ Audio", "Upload an audio segment");
        this.btnDelete   = mkBtn("Delete",  "Delete selected segment (Delete key)");
        this.btnPlay     = mkBtn("▶",       "Play / Pause (Space)");
        this.btnZoomOut  = mkBtn("−",       "Zoom out");
        this.btnZoomIn   = mkBtn("+",       "Zoom in");
        this.btnFit      = mkBtn("Fit",     "Fit timeline to view");
        const modeSel = document.createElement("select");
        modeSel.style.cssText = "background:var(--c2c-panelBg);color:var(--c2c-fg);border:1px solid var(--c2c-surface1Alt);border-radius:4px;padding:3px;font-size:11px;";
        modeSel.innerHTML = '<option value="seconds">Seconds</option><option value="frames">Frames</option>';
        modeSel.value = this.displayMode;
        modeSel.onchange = () => { writeWidget(this.node, "display_mode", modeSel.value); this.render(); };
        this.modeSel = modeSel;
        const status = document.createElement("span");
        status.style.cssText = "margin-left:auto;font-size:10.5px;color:var(--c2c-sub);font-family:ui-monospace,monospace;";
        this.statusEl = status;
        for (const el of [this.btnAddText, this.btnAddImage, this.btnAddAudio, this.btnDelete,
                          this.btnPlay, this.btnZoomOut, this.btnZoomIn, this.btnFit, modeSel, status]) {
            tb.appendChild(el);
        }
        root.appendChild(tb);

        // Canvas wrap (for drag-drop overlay)
        const wrap = document.createElement("div");
        wrap.style.cssText = "position:relative;width:100%;background:var(--c2c-scrimDark3);border-radius:4px;border:1px solid var(--c2c-panelBg);";
        this.cvs = document.createElement("canvas");
        this.cvs.tabIndex = 0;
        this.cvs.style.cssText = "display:block;width:100%;height:" + (RULER_H + IMG_TRACK_H + AUD_TRACK_H) + "px;outline:none;cursor:default;";
        wrap.appendChild(this.cvs);
        const dropHint = document.createElement("div");
        dropHint.style.cssText = "position:absolute;inset:0;pointer-events:none;display:none;align-items:center;justify-content:center;background:rgba(80,140,220,0.25);color:var(--c2c-white);font-size:14px;border:2px dashed var(--c2c-blue);border-radius:4px;";
        dropHint.textContent = "Drop image or audio file here";
        this.dropHint = dropHint;
        wrap.appendChild(dropHint);
        root.appendChild(wrap);
        this.cvsWrap = wrap;

        // Player bar (seekbar + readout)
        const pbar = document.createElement("div");
        pbar.style.cssText = `
            display:flex;align-items:center;gap:6px;background:var(--c2c-bg3);
            border:1px solid var(--c2c-panelBg);border-radius:4px;padding:2px 6px;height:${PLAYER_BAR_H}px;
        `;
        this.seek = document.createElement("input");
        this.seek.type = "range";
        this.seek.min = "0";
        this.seek.max = "10000";
        this.seek.value = "0";
        this.seek.style.cssText = "flex:1 1 auto;accent-color:var(--c2c-blue);";
        this.timeReadout = document.createElement("span");
        this.timeReadout.style.cssText = "font-family:ui-monospace,monospace;font-size:10.5px;color:var(--c2c-sub);min-width:120px;text-align:right;";
        this.timeReadout.textContent = "f 0 / 0";
        pbar.appendChild(this.seek);
        pbar.appendChild(this.timeReadout);
        root.appendChild(pbar);

        // Properties panel
        const props = document.createElement("div");
        props.style.cssText = `
            display:flex;flex-direction:column;gap:4px;background:var(--c2c-bg3);
            border:1px solid var(--c2c-panelBg);border-radius:4px;padding:6px;min-height:${PROPS_MIN_H}px;
        `;
        const labelRow = document.createElement("div");
        labelRow.style.cssText = "display:flex;gap:8px;align-items:center;font-size:11px;color:var(--c2c-sub);";
        this.propTitle = document.createElement("span");
        this.propTitle.textContent = "No segment selected";
        this.propBounds = document.createElement("span");
        this.propBounds.style.cssText = "margin-left:auto;color:var(--c2c-slateMute);font-family:ui-monospace,monospace;";
        labelRow.appendChild(this.propTitle);
        labelRow.appendChild(this.propBounds);
        props.appendChild(labelRow);

        const ta = document.createElement("textarea");
        ta.placeholder = "Prompt for this segment (select an image or text clip)";
        ta.style.cssText = `
            background:var(--c2c-scrimDark3);color:var(--c2c-gray100);border:1px solid var(--c2c-panelBg);border-radius:4px;
            padding:6px;font-size:12px;font-family:ui-sans-serif,system-ui,sans-serif;
            resize:vertical;min-height:60px;outline:none;width:100%;box-sizing:border-box;
        `;
        ta.onfocus = () => { ta.style.borderColor = "var(--c2c-blue)"; };
        ta.onblur  = () => { ta.style.borderColor = "var(--c2c-panelBg)"; };
        ta.oninput = () => {
            const seg = this._selSeg();
            if (seg && (seg.type === "image" || seg.type === "text")) {
                seg.prompt = ta.value;
                this.commitChanges();
                this.render();
            }
        };
        this.promptArea = ta;
        props.appendChild(ta);

        const gsRow = document.createElement("div");
        gsRow.style.cssText = "display:flex;gap:6px;align-items:center;font-size:11px;color:var(--c2c-sub);";
        const gsLabel = document.createElement("span");
        gsLabel.textContent = "Guide strength:";
        this.gsSlider = document.createElement("input");
        this.gsSlider.type = "range";
        this.gsSlider.min = "0"; this.gsSlider.max = "200"; this.gsSlider.value = "100";
        this.gsSlider.style.cssText = "flex:1 1 auto;accent-color:var(--c2c-okSoft);";
        this.gsVal = document.createElement("span");
        this.gsVal.style.cssText = "font-family:ui-monospace,monospace;color:var(--c2c-fg);min-width:40px;text-align:right;";
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
        this.audioInfo.style.cssText = "display:none;background:var(--c2c-scrimDark3);border:1px solid var(--c2c-panelBg);border-radius:4px;padding:6px;font-size:11px;color:var(--c2c-sub);line-height:1.5;font-family:ui-monospace,monospace;";
        props.appendChild(this.audioInfo);

        root.appendChild(props);

        // Hidden file inputs
        this.fileImg = document.createElement("input");
        this.fileImg.type = "file"; this.fileImg.accept = "image/*"; this.fileImg.style.display = "none";
        this.fileAud = document.createElement("input");
        this.fileAud.type = "file"; this.fileAud.accept = "audio/*"; this.fileAud.style.display = "none";
        root.appendChild(this.fileImg);
        root.appendChild(this.fileAud);
    }

    _wireEvents() {
        this.btnAddText.onclick  = () => this.addTextSegment();
        this.btnAddImage.onclick = () => this.fileImg.click();
        this.btnAddAudio.onclick = () => this.fileAud.click();
        this.btnDelete.onclick   = () => this.deleteSelected();
        this.btnPlay.onclick     = () => this.togglePlay();
        this.btnZoomIn.onclick   = () => this.setZoom(this.zoom * 1.4);
        this.btnZoomOut.onclick  = () => this.setZoom(this.zoom / 1.4);
        this.btnFit.onclick      = () => this.setZoom(1.0);

        this.fileImg.onchange = async () => {
            for (const f of this.fileImg.files) await this.addImageSegmentFromFile(f);
            this.fileImg.value = "";
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
                if (f.type.startsWith("image/")) { await this.addImageSegmentFromFile(f); accepted += 1; }
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
            if (e.key === "Delete" || e.key === "Backspace") { this.deleteSelected(); e.preventDefault(); }
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
            for (const s of this.segments)      if (!s.id) s.id = nid();
            for (const s of this.audioSegments) if (!s.id) s.id = nid();
        }
        this.suppressCommit = false;
        // Async image preload
        for (const s of this.segments) this._preloadImage(s);
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
        writeWidget(this.node, "timeline_data", JSON.stringify({ segments: segOut, audioSegments: audOut }));
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
        const arr = track === "audio" ? this.audioSegments : this.segments;
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

    deleteSelected() {
        const { type, idx } = this.selection;
        if (idx < 0) return;
        const arr = type === "audio" ? this.audioSegments : this.segments;
        if (idx < arr.length) {
            arr.splice(idx, 1);
            this.selection = { type: null, idx: -1 };
            this.commitChanges();
            this._updatePropsPanel();
        }
    }

    _selSeg() {
        const { type, idx } = this.selection;
        if (idx < 0) return null;
        const arr = type === "audio" ? this.audioSegments : this.segments;
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
        return (w * this.zoom) / Math.max(1, this.visualDurFrames);
    }
    _frameToX(f) { return f * this._pxPerFrame() + PAD; }
    _xToFrame(x) { return Math.round((x - PAD) / Math.max(1e-3, this._pxPerFrame())); }

    _hitTest(mx, my) {
        // Returns {kind, segType, idx, edge}
        if (my < RULER_H) return { kind: "ruler" };
        const imgBot = RULER_H + IMG_TRACK_H;
        const track = (my < imgBot) ? "image" : "audio";
        const arr = track === "audio" ? this.audioSegments : this.segments;
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
        const rect = this.cvs.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        if (e.button === 2) return; // handled by contextmenu
        const hit = this._hitTest(mx, my);
        if (hit.kind === "ruler") {
            this.playhead = clamp(this._xToFrame(mx), 0, this.visualDurFrames);
            this.dragState = { type: "playhead" };
            this.render();
            return;
        }
        if (hit.kind === "seg") {
            this.selection = { type: hit.segType === "audio" ? "audio" : "image", idx: hit.idx };
            const arr = hit.segType === "audio" ? this.audioSegments : this.segments;
            const seg = arr[hit.idx];
            this.dragState = {
                type: "seg", edge: hit.edge, segType: hit.segType, idx: hit.idx,
                startFrame: this._xToFrame(mx),
                origStart: seg.start, origLength: seg.length, origTrim: seg.trimStart || 0,
            };
            this._updatePropsPanel();
            this.render();
            return;
        }
        // Empty track click — clear selection
        this.selection = { type: null, idx: -1 };
        this._updatePropsPanel();
        this.render();
    }

    _onMouseMove(e) {
        const rect = this.cvs.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
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
            const arr = ds.segType === "audio" ? this.audioSegments : this.segments;
            const seg = arr[ds.idx];
            const cur = this._xToFrame(mx);
            const delta = cur - ds.startFrame;
            if (ds.edge === "mid") {
                seg.start = Math.max(0, ds.origStart + delta);
                this._resolveCollisions(ds.segType, ds.idx);
            } else if (ds.edge === "right") {
                seg.length = Math.max(MIN_SEG_FRAMES, ds.origLength + delta);
                if (ds.segType === "audio" && seg.audioDurationFrames) {
                    seg.length = Math.min(seg.length, seg.audioDurationFrames - (seg.trimStart || 0));
                }
                this._resolveCollisions(ds.segType, ds.idx);
            } else if (ds.edge === "left") {
                const newStart = Math.max(0, ds.origStart + delta);
                const trim = newStart - ds.origStart;
                const newLen = ds.origLength - trim;
                if (newLen >= MIN_SEG_FRAMES) {
                    seg.start = newStart;
                    seg.length = newLen;
                    if (ds.segType === "audio") {
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
        const rect = this.cvs.getBoundingClientRect();
        const my = e.clientY - rect.top;
        const hit = this._hitTest(e.clientX - rect.left, my);
        if (hit.kind === "seg") {
            this.promptArea.focus();
        }
    }

    _onContextMenu(e) {
        e.preventDefault();
        const rect = this.cvs.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const hit = this._hitTest(mx, my);
        const items = [];
        if (hit.kind === "seg") {
            items.push({ label: "Copy", action: () => this._copySegment(hit) });
            items.push({ label: "Delete", action: () => { this.selection = { type: hit.segType === "audio" ? "audio" : "image", idx: hit.idx }; this.deleteSelected(); } });
        } else if (hit.kind === "track") {
            items.push({ label: "+ Text segment", action: () => this.addTextSegment() });
            items.push({ label: "+ Image…",       action: () => this.fileImg.click() });
            items.push({ label: "+ Audio…",       action: () => this.fileAud.click() });
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
        const arr = hit.segType === "audio" ? this.audioSegments : this.segments;
        this.clipboard = { track: hit.segType, seg: JSON.parse(JSON.stringify(arr[hit.idx])) };
    }
    _pasteAt(frame, track) {
        if (!this.clipboard || this.clipboard.track !== track) return;
        const seg = JSON.parse(JSON.stringify(this.clipboard.seg));
        seg.id = nid();
        seg.start = Math.max(0, frame);
        const arr = track === "audio" ? this.audioSegments : this.segments;
        arr.push(seg);
        this._resolveCollisions(track, arr.length - 1);
        this.commitChanges();
    }

    // Avoid overlap on the same track. Push later segments rightwards.
    _resolveCollisions(track, idx) {
        const arr = track === "audio" ? this.audioSegments : this.segments;
        const sorted = [...arr].sort((a, b) => a.start - b.start);
        let cursor = -Infinity;
        for (const s of sorted) {
            if (s.start < cursor) s.start = cursor;
            cursor = s.start + s.length;
        }
    }

    // ── Playback ────────────────────────────────────────────────────
    setZoom(z) { this.zoom = clamp(z, 0.1, 32); this.render(); }
    stepPlayhead(n) { this.playhead = clamp(this.playhead + n, 0, this.visualDurFrames); this.render(); }
    togglePlay() {
        if (this.playing) { this._stopAudio(); this.playing = false; this.btnPlay.textContent = "▶"; }
        else { this._startAudio(); this.playing = true; this.btnPlay.textContent = "❚❚"; this._tick(); }
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
            if (this.playhead >= this.visualDurFrames) {
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
            this.propTitle.textContent = "No segment selected";
            this.propBounds.textContent = "";
            this.promptArea.value = "";
            this.promptArea.disabled = true;
            this.gsSlider.disabled = true;
            this.audioInfo.style.display = "none";
            return;
        }
        const fps = this.fps;
        this.propBounds.textContent = `${fmtTime(seg.start, fps, this.displayMode)} → ${fmtTime(seg.start + seg.length, fps, this.displayMode)} · ${seg.length}f`;
        if (seg.type === "audio") {
            this.propTitle.textContent = `Audio: ${seg.fileName || seg.audioFile || "(unknown)"}`;
            this.promptArea.style.display = "none";
            this.gsSlider.parentElement.style.display = "none";
            this.audioInfo.style.display = "block";
            const trimIn = (seg.trimStart || 0) / fps;
            const trimOut = ((seg.audioDurationFrames || 0) - (seg.trimStart || 0) - seg.length) / fps;
            this.audioInfo.innerHTML =
                `File: ${seg.fileName || seg.audioFile}<br>` +
                `Source length: ${(seg.audioDurationFrames / fps).toFixed(2)}s<br>` +
                `Output length: ${(seg.length / fps).toFixed(2)}s<br>` +
                `Trim-in: ${trimIn.toFixed(2)}s · Trim-out: ${Math.max(0, trimOut).toFixed(2)}s`;
        } else {
            this.propTitle.textContent = (seg.type === "text" ? "Text segment" : "Image segment");
            this.promptArea.style.display = "block";
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
        const dpr = window.devicePixelRatio || 1;
        const cssW = this.cvs.clientWidth || 600;
        const cssH = RULER_H + IMG_TRACK_H + AUD_TRACK_H;
        const w = Math.round(cssW * dpr);
        const h = Math.round(cssH * dpr);
        if (this.cvs.width !== w || this.cvs.height !== h) {
            this.cvs.width = w; this.cvs.height = h;
        }
        const ctx = this.cvs.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cssW, cssH);

        // Background tracks
        ctx.fillStyle = C.scrimDark;
        ctx.fillRect(0, 0, cssW, cssH);
        ctx.fillStyle = C.bg3;
        ctx.fillRect(0, RULER_H, cssW, IMG_TRACK_H);
        ctx.fillStyle = C.scrimDark;
        ctx.fillRect(0, RULER_H + IMG_TRACK_H, cssW, AUD_TRACK_H);

        // Out-of-duration shadow
        const durX = this._frameToX(this.durFrames);
        if (durX < cssW) {
            ctx.fillStyle = "rgba(0,0,0,0.5)";
            ctx.fillRect(durX, RULER_H, cssW - durX, IMG_TRACK_H + AUD_TRACK_H);
        }

        // Ruler
        this._drawRuler(ctx, cssW);
        // Segments
        for (let i = 0; i < this.segments.length; i++) this._drawSegment(ctx, this.segments[i], i, "image");
        for (let i = 0; i < this.audioSegments.length; i++) this._drawSegment(ctx, this.audioSegments[i], i, "audio");
        // Playhead
        this._drawPlayhead(ctx, cssH);

        // Update seekbar + readout
        const vd = this.visualDurFrames;
        this.seek.value = String(Math.round((this.playhead / Math.max(1, vd)) * 10000));
        this.timeReadout.textContent = `f ${this.playhead}/${this.durFrames} · ${fmtTime(this.playhead, this.fps, this.displayMode)}`;
        this.statusEl.textContent = `${this.segments.length} clips · ${this.audioSegments.length} audio · ${this.durFrames}f @ ${this.fps}fps`;
    }

    _drawRuler(ctx, cssW) {
        ctx.fillStyle = C.panelHi2;
        ctx.fillRect(0, 0, cssW, RULER_H);
        ctx.fillStyle = C.dim;
        ctx.font = "10px ui-monospace,monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        const pxF = this._pxPerFrame();
        const step = pickRulerStep(this.visualDurFrames, this.fps, pxF, this.displayMode);
        for (let f = 0; f <= this.visualDurFrames + step; f += step) {
            const x = this._frameToX(f);
            if (x > cssW + 50) break;
            ctx.strokeStyle = C.panelHi2;
            ctx.beginPath();
            ctx.moveTo(x, RULER_H - 6);
            ctx.lineTo(x, RULER_H);
            ctx.stroke();
            ctx.fillStyle = C.slateMute;
            ctx.fillText(fmtTime(f, this.fps, this.displayMode), x + 3, RULER_H / 2);
        }
    }

    _drawSegment(ctx, seg, idx, track) {
        const x1 = this._frameToX(seg.start);
        const x2 = this._frameToX(seg.start + seg.length);
        const y = track === "audio" ? RULER_H + IMG_TRACK_H + 4 : RULER_H + 4;
        const h = (track === "audio" ? AUD_TRACK_H : IMG_TRACK_H) - 8;
        const w = Math.max(2, x2 - x1);
        const selected = this.selection.type === (track === "audio" ? "audio" : "image") && this.selection.idx === idx;

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
        try { this._resizeObs?.disconnect(); } catch {}
        if (this.audioCtx) try { this.audioCtx.close(); } catch {}
        this.audioCtx = null;
        if (this._renderRafId) {
            try { cancelAnimationFrame(this._renderRafId); } catch {}
            this._renderRafId = 0;
        }
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

app.registerExtension({
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
            // Moderate default width; height capped after DOM widgets register.
            if (!this._wdTimelineSizeApplied) {
                this.size[0] = Math.max(this.size[0] || 0, WD_DEFAULT_W);
                this._wdTimelineSizeApplied = true;
            }
            _wdCapNode(this);
            const host = document.createElement("div");
            host.style.cssText = "width:100%;height:100%;";
            const widget = this.addDOMWidget("wd_timeline", "wd_timeline", host, {
                getValue: () => "",
                setValue: () => {},
                serialize: false,
            });
            widget.computeSize = function (width) {
                return [width, RULER_H + IMG_TRACK_H + AUD_TRACK_H + TOOLBAR_H + PLAYER_BAR_H + PROPS_MIN_H + 40];
            };
            const self = this;
            setTimeout(() => {
                try { self._wdTimeline = new TimelineEditor(self, host); }
                catch (err) { reportFailure("WanDirector.timelineInit", err, "wan_director_timeline"); }
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
