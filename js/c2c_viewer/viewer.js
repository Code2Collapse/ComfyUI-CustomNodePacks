/**
 * The Matte viewer panel on Mask + Matting (L7.114, owner A14; design docs/design/matte_viewer.md).
 *
 * Plate -> Subject -> Render on the node itself:
 *  - Load plate: the nodes feeding the image input + the Plate Proxy helper -> proxy frames to scrub.
 *  - Find: words on the current frame -> every object found, tinted, one hit map each (Detect helper, ComfyUI's own
 *    SAM 3). Click = subject (the others found are excluded), Shift-click = add a part, Alt-click = cut a part,
 *    Ctrl-click = remove the nearest marker. A click with nothing found asks Detect about the point itself.
 *  - Render: a preview chunk from the pick frame (N frames, node.properties), then the full shot. The Matte node runs
 *    with the selection in `picks`; the Viewer Result helper sends back proxies + QC numbers.
 *  - Result: wipe between plate and result, comp / alpha / checker / overlay, scrub, QC badges.
 *
 * The selection is the node's `picks` widget (serialised; the panel itself is not). Canvas colours are literal values
 * read from the --cu tokens: a var() in a canvas fillStyle paints black. No polling while idle; the transport polls
 * /history only while a helper job is pending (transport.js).
 *
 * This module takes ComfyUI's `app` and `api` as arguments (mask_matting.js passes them), so it has no import that
 * depends on where the pack is installed.
 *
 * License: Apache-2.0
 */

import { button, ensureStyles, pillBar } from "../c2c_ui/index.js";
import { canvasBackingScale, onZoomChange } from "../c2c_ui/nodes2.js";
import { parseFrameInput, rulerTicks } from "../_c2c_player_core.js";
import {
    DEFAULT_PREVIEW_FRAMES, QC_NAMES, addPart, clickSubject, cutPart, emptySelection, fitRect, frameLabel, fromPicks,
    hitTest, maskFromRGBA, nearestFrame, previewRange, qcLevel, qcTooltip, qcValueText, removeNearest,
    selectionMask, stepper, toPicks, viewToPlate,
} from "./core.js";
import { HELPERS, createRunner, executionIdOf, findNodeId, helperPrompt, viewUrl } from "./transport.js";

export const MATTE_CLASS = "MaskOpsMEC";
const PROP_PREVIEW = "c2c_viewer_preview_frames";
const PROP_ADVANCED = "c2c_viewer_advanced";
const IMAGE_CACHE = 48;                      // decoded proxy images kept per viewer (plate + result each)
const MISSING_PREFIX = "MISSING: ";

// fixed rows (px): head, scrub, ruler, find, actions, status, tools, qc, settings
const ROWS = [24, 24, 14, 26, 28, 16, 24, 22, 24];
const GAP = 6;
const PAD = 4;
const FIXED = ROWS.reduce((a, b) => a + b, 0) + GAP * ROWS.length + PAD * 2 + 2;   // + the view's 1 px border
const VIEW_MIN = 140;
const VIEW_MAX = 640;

const STYLE_ID = "c2c-viewer-v1";
const CSS = `
.c2cv { display:flex; flex-direction:column; gap:${GAP}px; padding:${PAD}px; height:100%; box-sizing:border-box;
  min-width:0; user-select:none; }
.c2cv-head { display:flex; align-items:center; gap:6px; height:${ROWS[0]}px; min-width:0; flex:none; }
.c2cv-steps { display:flex; gap:4px; flex:none; }
.c2cv-step { display:inline-flex; align-items:center; gap:4px; height:20px; padding:0 8px; border-radius:10px;
  font:inherit; font-size:11px; border:1px solid var(--cu-edge-strong); color:var(--cu-ink-dim); white-space:nowrap;
  background:transparent; margin:0; }
.c2cv-step:hover:not(:disabled) { border-color:var(--cu-accent-hover); }
.c2cv-step--done { color:var(--cu-ink-soft); }
.c2cv-step--current { border-color:var(--cu-accent); color:var(--cu-ink);
  background:color-mix(in srgb, var(--cu-accent) 22%, transparent); }
.c2cv-select { flex:1 1 auto; min-width:0; height:22px; border-radius:5px; border:1px solid var(--cu-edge-strong);
  background:var(--cu-sunken); color:var(--cu-ink); font:inherit; font-size:11px; padding:0 4px; }
.c2cv-select--missing { border-color:var(--cu-warn); color:var(--cu-warn); }
.c2cv-view { position:relative; flex:1 0 auto; min-height:${VIEW_MIN}px; border-radius:6px; overflow:hidden;
  background:var(--cu-sunken); border:1px solid var(--cu-edge); }
.c2cv-view canvas { position:absolute; inset:0; width:100%; height:100%; display:block; outline:none; cursor:crosshair; }
.c2cv-view canvas:focus-visible { box-shadow:inset 0 0 0 1px var(--cu-accent); }
.c2cv-msg { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; text-align:center;
  padding:16px; color:var(--cu-ink-soft); font-size:12px; line-height:1.45; pointer-events:none; }
.c2cv-tag { position:absolute; left:6px; top:6px; font-size:10px; padding:2px 7px; border-radius:8px;
  background:color-mix(in srgb, var(--cu-ground) 78%, transparent); color:var(--cu-ink-soft); pointer-events:none;
  white-space:nowrap; max-width:calc(100% - 12px); overflow:hidden; text-overflow:ellipsis; }
.c2cv-scrub { display:flex; align-items:center; gap:4px; height:${ROWS[1]}px; flex:none; }
.c2cv-scrub input[type=range] { flex:1 1 auto; min-width:0; accent-color:var(--cu-accent); }
.c2cv-icon { width:26px; height:22px; padding:0 !important; flex:none; }
.c2cv-ruler { height:${ROWS[2]}px; flex:none; width:100%; display:block; }
.c2cv-row { display:flex; gap:4px; align-items:center; flex:none; min-width:0; }
.c2cv-input { flex:1 1 auto; min-width:0; height:24px; box-sizing:border-box; border-radius:5px;
  border:1px solid var(--cu-edge-strong); background:var(--cu-sunken); color:var(--cu-ink); padding:0 8px;
  font:inherit; font-size:12px; user-select:text; }
.c2cv-input:focus { outline:none; border-color:var(--cu-accent); }
.c2cv-input:disabled { opacity:.55; }
.c2cv-frame { flex:none; width:58px; text-align:center; padding:0 4px; }
.c2cv-n { flex:none; width:46px; text-align:center; padding:0 2px; }
.c2cv .c2c-ui-btn { height:24px; padding:0 10px; font-size:11.5px; white-space:nowrap; }
.c2cv-actions .c2c-ui-btn { height:28px; }
.c2cv-primary { flex:2 1 0; min-width:0; }
.c2cv-secondary { flex:1 1 0; min-width:0; }
.c2cv-status { height:${ROWS[5]}px; flex:none; font-size:11px; line-height:${ROWS[5]}px; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; color:var(--cu-ink-soft); }
.c2cv-status--warn { color:var(--cu-warn); }
.c2cv-status--danger { color:var(--cu-danger); }
.c2cv-status--ok { color:var(--cu-ok); }
.c2cv-tools .c2c-ui-pillbar { flex:none; }
.c2cv-tools .c2c-ui-pill { font-size:11px; padding:2px 8px; }
.c2cv-dim { color:var(--cu-ink-dim); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  min-width:0; }
.c2cv-chip { display:inline-flex; align-items:center; gap:5px; height:20px; padding:0 8px; border-radius:10px;
  border:1px solid var(--cu-edge-strong); font-size:11px; color:var(--cu-ink-soft); white-space:nowrap; flex:none; }
.c2cv-dot { width:7px; height:7px; border-radius:50%; background:var(--cu-ink-dim); }
.c2cv-dot--green { background:var(--cu-ok); }
.c2cv-dot--amber { background:var(--cu-warn); }
.c2cv-dot--red { background:var(--cu-danger); }
.c2cv-spacer { flex:1 1 auto; }
.c2cv [hidden] { display:none !important; }
`;

function ensureViewerStyles() {
    ensureStyles();
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = CSS;
    document.head.appendChild(s);
}

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

/** Stop the graph from reading pointer / key events meant for the panel (dragging the node, shortcuts). */
function keepEvents(root) {
    for (const t of ["pointerdown", "mousedown", "dblclick", "contextmenu"]) {
        root.addEventListener(t, (e) => e.stopPropagation());
    }
    root.addEventListener("keydown", (e) => {
        if (e.target?.tagName === "INPUT" || e.target?.tagName === "SELECT") e.stopPropagation();
    });
}

/** A literal colour for a canvas (resolves var() chains through the computed style; never paints black). */
function tokens(root) {
    const cs = getComputedStyle(root);
    const get = (name, fallback) => (cs.getPropertyValue(name) || "").trim() || fallback;
    return {
        accent: get("--cu-accent", "#b494ff"), danger: get("--cu-danger", "#f27a92"), ok: get("--cu-ok", "#7fe0ab"),
        warn: get("--cu-warn", "#f3d288"), ink: get("--cu-ink", "#e8e6f7"), dim: get("--cu-ink-dim", "#6f6d9b"),
        ground: get("--cu-ground", "#07081a"), sunken: get("--cu-sunken", "#0c0d23"),
        edge: get("--cu-edge-strong", "#3b3a68"),
        series: [1, 2, 3, 4, 5, 6].map((i) => get(`--cu-series-${i}`, "#76dccb")),
    };
}

const _rgbCache = new Map();
function rgbOf(color) {
    if (_rgbCache.has(color)) return _rgbCache.get(color);
    const c = document.createElement("canvas");
    c.width = c.height = 1;
    const g = c.getContext("2d");
    g.fillStyle = "#000";
    g.fillStyle = color;
    g.fillRect(0, 0, 1, 1);
    const d = g.getImageData(0, 0, 1, 1).data;
    const rgb = [d[0], d[1], d[2]];
    _rgbCache.set(color, rgb);
    return rgb;
}

/** A hit map as a tinted canvas (built once per detection and colour). */
function tintCanvas(mask, w, h, color, alpha = 1) {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const g = c.getContext("2d");
    const img = g.createImageData(w, h);
    const [r, gg, b] = rgbOf(color);
    const a = Math.round(255 * alpha);
    for (let i = 0; i < w * h; i += 1) {
        if (!mask[i]) continue;
        img.data[i * 4] = r; img.data[i * 4 + 1] = gg; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = a;
    }
    g.putImageData(img, 0, 0);
    return c;
}

/** Small LRU of decoded images by URL. */
function imageCache(limit, onLoad) {
    const map = new Map();
    return {
        get(url) {
            if (!url) return null;
            let e = map.get(url);
            if (e) { map.delete(url); map.set(url, e); return e.ready ? e.img : null; }
            const img = new Image();
            e = { img, ready: false };
            img.onload = () => { e.ready = true; onLoad?.(); };
            img.onerror = () => { map.delete(url); };
            img.src = url;
            map.set(url, e);
            while (map.size > limit) map.delete(map.keys().next().value);
            return null;
        },
        clear() { map.clear(); },
    };
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("A viewer image could not be loaded (the temp file is gone). Run the step again."));
        img.src = url;
    });
}

/** A Detect object's PNG -> {mask, w, h, count}. */
async function hitMapOf(url, w, h) {
    const img = await loadImage(url);
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(img, 0, 0, w, h);
    const mask = maskFromRGBA(g.getImageData(0, 0, w, h).data, w, h);
    let count = 0;
    for (let i = 0; i < mask.length; i += 1) count += mask[i];
    return { mask, w, h, count };
}

/** The Detect helper's model slot, fetched once per page for every viewer; Load plate fetches it again, so a
 *  checkpoint placed while ComfyUI runs shows up (after R). */
let _detectInfo = null;
function detectInfo(api, refresh = false) {
    if (!_detectInfo || refresh) {
        _detectInfo = (async () => {
            try {
                const r = await api.fetchApi(`/object_info/${HELPERS.detect}`);
                const spec = (await r.json())?.[HELPERS.detect]?.input?.required?.model;
                return { models: Array.isArray(spec?.[0]) ? spec[0] : [], missing: spec?.[1]?.c2c_missing || null };
            } catch (_e) {
                _detectInfo = null;                      // try again next time
                return { models: [], missing: null };
            }
        })();
    }
    return _detectInfo;
}

function humanError(e) {
    const raw = String(e?.message || e || "");
    if (!raw) return "Something went wrong.";
    if (/CUDA out of memory|OutOfMemory/i.test(raw)) return "Not enough GPU memory. Lower the resolution or the preview length.";
    if (/ModuleNotFoundError/i.test(raw)) return "A required Python package is missing (see the ComfyUI console).";
    const line = raw.split("\n").find((l) => l.trim()) || raw;
    return line.length > 400 ? line.slice(0, 400) + "…" : line;
}

/**
 * Build the viewer for one Mask + Matting node.
 *  opts.app, opts.api        ComfyUI's app and api
 *  opts.getPicks()/setPicks(json)   the node's picks widget
 *  opts.getBackend()         the node's backend widget value
 *  opts.onAdvanced(open)     the "All settings" switch (mask_matting.js folds the regular widgets)
 *  opts.onLayout()           the panel wants a new height (the plate's aspect changed)
 * Returns {root, heightFor(width), syncFromWidget(), destroy()}.
 */
export function createViewer(node, opts) {
    ensureViewerStyles();
    const { app, api } = opts;
    const runner = createRunner(api);

    const st = {
        plate: null, frame: 0, objects: [], detFrame: null, sel: null, text: "", model: "", models: [], missing: null,
        tab: "pick", result: null, resultStale: false, previewDone: false, view: "comp", wipe: 1, busy: false,
        wrapper: "", selVer: 0,
    };

    // ── DOM ─────────────────────────────────────────────────────────────────────────────────────────────────────
    const root = el("div", "c2c-ui c2cv");          // c2c-ui: the --cu tokens and the primitives' styles live there
    keepEvents(root);

    const head = el("div", "c2cv-head");
    const steps = el("div", "c2cv-steps");
    const modelSel = el("select", "c2cv-select");
    modelSel.title = "SAM checkpoint for Find and clicks (ComfyUI's own SAM 3), strongest first.";
    head.append(steps, modelSel);

    const view = el("div", "c2cv-view");
    const canvas = el("canvas");
    canvas.tabIndex = 0;
    canvas.setAttribute("aria-label", "Plate frame. Click the subject; Shift-click adds a part, Alt-click cuts one, Ctrl-click removes the nearest marker.");
    const msg = el("div", "c2cv-msg");
    const tag = el("div", "c2cv-tag");
    tag.hidden = true;
    view.append(canvas, msg, tag);

    const scrub = el("div", "c2cv-scrub");
    const prevBtn = button("◀", { onClick: () => setFrame(st.frame - stepSize()) });
    prevBtn.classList.add("c2cv-icon");
    prevBtn.title = "Previous frame (←)";
    const slider = el("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "0";
    slider.value = "0";
    slider.setAttribute("aria-label", "Frame");
    const nextBtn = button("▶", { onClick: () => setFrame(st.frame + stepSize()) });
    nextBtn.classList.add("c2cv-icon");
    nextBtn.title = "Next frame (→)";
    const frameBox = el("input", "c2cv-input c2cv-frame");
    frameBox.type = "text";
    frameBox.title = "Frame number (or timecode)";
    frameBox.setAttribute("aria-label", "Frame number");
    scrub.append(prevBtn, slider, nextBtn, frameBox);
    const ruler = el("canvas", "c2cv-ruler");

    const findRow = el("div", "c2cv-row");
    findRow.style.height = `${ROWS[3]}px`;
    const findBox = el("input", "c2cv-input");
    findBox.type = "text";
    findBox.placeholder = "What to find: person, dog";
    findBox.setAttribute("aria-label", "What to find, comma-separated");
    const findBtn = button("Find", { onClick: () => find() });
    findBtn.title = "Find every match on this frame (Enter)";
    const clearBtn = button("Clear", { onClick: () => clearSelection() });
    clearBtn.title = "Forget the subject and every marker";
    findRow.append(findBox, findBtn, clearBtn);

    const actions = el("div", "c2cv-row c2cv-actions");
    actions.style.height = `${ROWS[4]}px`;
    const primary = button("Load plate", { primary: true, onClick: () => runAction(primaryAction) });
    primary.classList.add("c2cv-primary");
    const secondary = button("Render full shot", { onClick: () => runAction(secondaryAction) });
    secondary.classList.add("c2cv-secondary");
    const nBox = el("input", "c2cv-input c2cv-n");
    nBox.type = "number";
    nBox.min = "1";
    nBox.max = "999";
    nBox.title = "Frames in the preview chunk (from the pick frame)";
    nBox.setAttribute("aria-label", "Preview frames");
    actions.append(primary, secondary, nBox);

    const status = el("div", "c2cv-status");

    const tools = el("div", "c2cv-row c2cv-tools");
    tools.style.height = `${ROWS[6]}px`;
    const tabBar = pillBar([{ value: "pick", label: "Pick" }, { value: "result", label: "Result" }],
        { value: "pick", onChange: (v) => { st.tab = v; update(); } });
    const viewBar = pillBar(["comp", "alpha", "checker", "overlay"].map((v) => ({ value: v, label: v[0].toUpperCase() + v.slice(1) })),
        { value: "comp", onChange: (v) => { st.view = v; draw(); } });
    viewBar.title = "Comp: the cut-out on the ground. Alpha: the matte. Checker: the cut-out on a checkerboard. Overlay: the matte tinted on the plate. Drag on the frame to wipe to the plate.";
    const resultInfo = el("span", "c2cv-dim");
    tools.append(tabBar, viewBar, resultInfo);

    const qcRow = el("div", "c2cv-row");
    qcRow.style.height = `${ROWS[7]}px`;
    const qcChips = {};
    for (const name of Object.keys(QC_NAMES)) {
        const chip = el("span", "c2cv-chip");
        const dot = el("span", "c2cv-dot");
        const label = el("span", "", `${QC_NAMES[name]} n/a`);
        chip.append(dot, label);
        qcChips[name] = { chip, dot, label };
        qcRow.append(chip);
    }

    const advRow = el("div", "c2cv-row");
    advRow.style.height = `${ROWS[8]}px`;
    const advLabel = el("span", "c2cv-dim", "");
    const advBtn = button("All settings ▸", { onClick: () => setAdvanced(!advancedOpen()) });
    advBtn.title = "Show every setting of the node (they stay saved either way)";
    advRow.append(advLabel, el("span", "c2cv-spacer"), advBtn);

    root.append(head, view, scrub, ruler, findRow, actions, status, tools, qcRow, advRow);

    // ── small state helpers ─────────────────────────────────────────────────────────────────────────────────────
    const plateFrames = () => st.plate?.count || 0;
    const stepSize = () => 1;
    const previewN = () => {
        const v = Number(node.properties?.[PROP_PREVIEW]);
        return Number.isFinite(v) && v >= 1 ? Math.min(999, Math.round(v)) : DEFAULT_PREVIEW_FRAMES;
    };
    const advancedOpen = () => !!node.properties?.[PROP_ADVANCED];
    const pickStep = () => !/remove background/i.test(String(opts.getBackend?.() || ""));
    let tok = null;
    const colours = () => (tok ||= tokens(root));

    function setStatus(text, kind = "") {
        status.textContent = text || "";
        status.title = text || "";
        status.className = "c2cv-status" + (kind ? ` c2cv-status--${kind}` : "");
    }

    function setAdvanced(open) {
        node.properties = node.properties || {};
        node.properties[PROP_ADVANCED] = !!open;
        advBtn.lastChild.textContent = open ? "All settings ▾" : "All settings ▸";
        advBtn.setAttribute("aria-expanded", String(!!open));
        opts.onAdvanced?.(!!open);
    }

    function writePicks() {
        const p = st.plate;
        const json = st.sel?.subject ? toPicks(st.sel, {
            text: st.text, model: st.model, plate: p ? { frames: p.count, w: p.width, h: p.height } : null,
        }) : "";
        opts.setPicks?.(json);
    }

    function selectionChanged() {
        st.selVer += 1;
        st.previewDone = false;
        if (st.result) st.resultStale = true;
        writePicks();
        update();
    }

    // ── the model slot (the Detect helper's definition) ─────────────────────────────────────────────────────────
    let modelsPromise = null;
    function loadModels(refresh = false) {
        if (!modelsPromise || refresh) modelsPromise = fillModels(refresh);
        return modelsPromise;
    }
    async function fillModels(refresh) {
        const info = await detectInfo(api, refresh);
        st.models = info.models;
        st.missing = info.missing;
        modelSel.replaceChildren();
        for (const m of st.models) {
            const o = el("option", "", m.startsWith(MISSING_PREFIX) ? `Missing: ${m.slice(MISSING_PREFIX.length)}` : m);
            o.value = m;
            modelSel.append(o);
        }
        if (!st.model || !st.models.includes(st.model)) st.model = st.models[0] || "";
        modelSel.value = st.model;
        modelSel.classList.toggle("c2cv-select--missing", st.model.startsWith(MISSING_PREFIX));
        if (st.model.startsWith(MISSING_PREFIX) && st.missing) {
            modelSel.title = `No SAM 3 checkpoint. Put ${st.missing.file} in ${st.missing.folder} (download: ${st.missing.url}), then press R in ComfyUI.`;
        }
        update();
    }
    modelSel.addEventListener("change", () => {
        st.model = modelSel.value;
        modelSel.classList.toggle("c2cv-select--missing", st.model.startsWith(MISSING_PREFIX));
        st.objects = [];
        st.detFrame = null;
        if (st.sel?.subject) writePicks();
        draw();
    });

    // ── jobs ────────────────────────────────────────────────────────────────────────────────────────────────────
    async function job(kind, o, label) {
        const graph = await app.graphToPrompt();
        const prompt = graph?.output || {};
        const rootGraph = app.rootGraph ?? app.graph;
        const execId = executionIdOf(rootGraph, node);
        const id = findNodeId(prompt, MATTE_CLASS, execId, node.id);
        if (!id) throw new Error("This Mask + Matting node is not in the graph that would run (is it muted or bypassed?).");
        st.wrapper = "";
        if (execId && execId.includes(":")) {
            const first = rootGraph?.getNodeById?.(execId.split(":")[0]);
            st.wrapper = first ? `${first.title || first.type} #${first.id}` : `#${execId.split(":")[0]}`;
        }
        const { prompt: hp, helperId } = helperPrompt(prompt, id, kind, o);
        st.busy = true;
        setStatus(label);
        update();
        try {
            return await runner.run(hp, helperId, (_s, text) => setStatus(`${label} ${text}`.trim()));
        } finally {
            st.busy = false;
        }
    }

    async function loadPlate() {
        loadModels(true);                               // a checkpoint placed since (and R pressed) shows up
        try {
            const p = await job("plate", {}, "Loading the plate…");
            st.plate = p;
            plateImgs.clear();
            st.frame = Math.min(Math.max(0, st.sel?.frame ?? st.frame), p.count - 1);
            setStatus(`Plate: ${p.width} × ${p.height}, ${p.count} frame${p.count === 1 ? "" : "s"}`
                + (p.step > 1 ? ` (every ${p.step}th shown)` : "") + (st.wrapper ? ` · in subgraph ${st.wrapper}` : "")
                + ". " + (pickStep() ? "Type what to find, or click the subject." : "Render when ready."), "ok");
            opts.onLayout?.();
        } catch (e) {
            setStatus(humanError(e), "danger");
        }
        update();
    }

    function checkModel() {
        if (!st.model || st.model.startsWith(MISSING_PREFIX)) {
            const m = st.missing;
            throw new Error(m ? `No SAM 3 checkpoint is installed: put ${m.file} in ${m.folder} (download: ${m.url}), then press R in ComfyUI.`
                : "No SAM 3 checkpoint is installed, so the viewer cannot find objects.");
        }
    }

    /** Detect on the current frame -> objects with hit maps. */
    async function detect(o, label) {
        await loadModels();
        checkModel();
        const p = await job("detect", { frame: st.frame, model: st.model, ...o }, label);
        const [pw, ph] = p.proxy;
        const objects = [];
        for (const ob of p.objects || []) {
            const hm = await hitMapOf(viewUrl(api, ob.mask), pw, ph);
            objects.push({ ...ob, ...hm });
        }
        return { objects, message: p.message || "", frame: p.frame };
    }

    async function find() {
        const text = findBox.value.trim();
        if (!st.plate) { setStatus("Load the plate first.", "warn"); return; }
        if (!text) { setStatus("Type what to find (for example: person), or click the subject on the frame.", "warn"); findBox.focus(); return; }
        st.text = text;
        try {
            const r = await detect({ text }, "Finding…");
            st.objects = r.objects;
            st.detFrame = r.frame;
            tintCache.clear();
            if (!r.objects.length) setStatus(r.message || `Nothing found for '${text}'.`, "warn");
            else setStatus(`${r.objects.length} found: ${r.objects.map((o) => `${o.label} ${Math.round(o.score * 100)}%`).join(", ")}. Click the subject.`, "ok");
        } catch (e) {
            setStatus(humanError(e), "danger");
        }
        update();
    }

    /** The region SAM gives for one point (the part preview for Shift / Alt). null when Detect cannot run. */
    async function regionAt(x, y, label) {
        try {
            const r = await detect({ points: [{ x, y, label: 1 }] }, label);
            return r.objects[0] || null;
        } catch (e) {
            setStatus(humanError(e), "warn");
            return null;
        }
    }

    async function onPick(x, y, mods) {
        if (!st.plate || !pickStep() || st.busy) return;
        const onDetFrame = st.detFrame === st.frame;
        const objs = onDetFrame ? st.objects : [];
        if (mods.ctrl) {
            if (!st.sel) return;
            const r = removeNearest(st.sel, x, y, st.plate.width / st.plate.height);
            if (!r.removed) { setStatus("No marker near that point.", "warn"); return; }
            st.sel = r.sel.subject ? r.sel : null;
            setStatus(`Removed the ${r.removed === "excluded" ? "excluded object" : r.removed} marker.`);
            selectionChanged();
            return;
        }
        if (mods.shift || mods.alt) {
            if (!st.sel?.subject) { setStatus(mods.shift ? "Click the subject first (no modifier key), then Shift-click to add parts."
                : "Click the subject first, then Alt-click to cut parts away.", "warn"); return; }
            if (st.sel.frame !== st.frame) { setStatus(`Parts are picked on the pick frame (${st.sel.frame}). Go back to it first.`, "warn"); return; }
            const hit = mods.shift ? hitTest(objs, x, y) : null;
            const reg = await regionAt(x, y, mods.shift ? "Finding the part…" : "Finding the part to cut…");
            const region = reg ? reg.mask : null;
            const r = mods.shift ? addPart(st.sel, objs, x, y, region) : cutPart(st.sel, x, y, region);
            if (!r.ok) { setStatus(r.reason, "warn"); update(); return; }
            st.sel = r.sel;
            setStatus(mods.shift ? `Part added${hit ? ` (${hit.label})` : ""}${region ? "" : " (point only: no part preview)"}.`
                : `Part cut${region ? "" : " (point only: no part preview)"}.`, "ok");
            selectionChanged();
            return;
        }
        // a plain click: the subject
        let pool = objs;
        let fromPoint = false;
        if (!hitTest(objs, x, y)) {
            const reg = await regionAt(x, y, "Finding the subject…");
            if (reg) {
                reg.id = objs.length ? Math.max(...objs.map((o) => o.id)) + 1 : 0;
                reg.label = "clicked";
                pool = [...objs, reg];
                st.objects = pool;
                st.detFrame = st.frame;
                tintCache.clear();
                fromPoint = true;
            }
        }
        const r = clickSubject(pool, x, y, st.frame);
        st.sel = r.sel;
        setStatus(fromPoint ? "Subject set from the click. A single click finds the part SAM is surest of; type what it is and press Find for the whole object, or Shift-click to add parts."
            : r.message, "ok");
        selectionChanged();
    }

    async function render(preview) {
        if (!st.plate) { setStatus("Load the plate first.", "warn"); return; }
        if (pickStep() && !st.sel?.subject) { setStatus("Click the subject first.", "warn"); return; }
        const p = st.plate;
        const plate = { frames: p.count, w: p.width, h: p.height };
        const range = preview ? previewRange(st.sel?.frame ?? st.frame, p.count, previewN()) : null;
        const sel = st.sel?.subject ? st.sel : { ...emptySelection(st.frame), subject: { x: 0.5, y: 0.5, id: null } };
        const picks = toPicks(sel, { text: st.text, model: st.model, plate, range });
        writePicks();
        try {
            const res = await job("render", { picks, start: range ? range.start : 0 },
                preview ? `Rendering the preview (${range.count} frames)…` : "Rendering the full shot…");
            st.result = { ...res, preview: !!preview };
            st.resultStale = false;
            resultImgs.clear();
            if (preview) st.previewDone = true;
            st.tab = "result";
            if (st.frame < res.start || st.frame >= res.start + res.count) st.frame = res.start;
            const empty = res.empty ? ` Frames ${res.empty[0]}–${res.empty[1] - 1} are empty: SeC + MatAnyone2 starts on the pick frame.` : "";
            setStatus(`${preview ? "Preview" : "Full shot"} rendered: frames ${res.start}–${res.start + res.count - 1}.${empty}`, empty ? "warn" : "ok");
        } catch (e) {
            setStatus(humanError(e), /Stopped/.test(String(e?.message)) ? "warn" : "danger");
        }
        update();
    }

    function clearSelection() {
        st.sel = null;
        setStatus("Selection cleared.");
        selectionChanged();
    }

    const STEP_TIPS = {
        plate: "Load the plate again (after a change upstream)",
        subject: "Back to picking: the frame with the found objects and the markers",
        render: "The rendered matte",
    };
    function stepClicked(id) {
        if (st.busy) return;
        if (id === "plate") loadPlate();
        else if (id === "subject") { st.tab = "pick"; update(); }
        else if (id === "render") { st.tab = "result"; update(); }
    }

    let primaryAction = "load";
    let secondaryAction = null;
    function runAction(id) {
        if (id === "wait") { runner.cancel(); return; }
        if (id === "load") loadPlate();
        else if (id === "find") find();
        else if (id === "preview") render(true);
        else if (id === "full") render(false);
    }

    // ── frames ──────────────────────────────────────────────────────────────────────────────────────────────────
    function setFrame(f) {
        const n = plateFrames();
        if (!n) return;
        st.frame = Math.min(n - 1, Math.max(0, Math.round(f)));
        update();
    }
    slider.addEventListener("input", () => setFrame(Number(slider.value)));
    frameBox.addEventListener("change", () => {
        const f = parseFrameInput(frameBox.value, 24);
        if (f === null) { frameBox.value = String(st.frame); return; }
        setFrame(f);
    });
    findBox.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); find(); } });
    nBox.addEventListener("change", () => {
        const v = Math.min(999, Math.max(1, Math.round(Number(nBox.value) || DEFAULT_PREVIEW_FRAMES)));
        node.properties = node.properties || {};
        node.properties[PROP_PREVIEW] = v;
        st.previewDone = false;
        update();
    });
    canvas.addEventListener("keydown", (e) => {
        if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
            e.preventDefault();
            e.stopPropagation();
            setFrame(st.frame + (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 10 : 1));
        }
    });

    // ── canvas interaction ──────────────────────────────────────────────────────────────────────────────────────
    let wiping = false;
    const platePoint = (e) => {
        const p = st.plate;
        return p ? viewToPlate(canvas.getBoundingClientRect(), p.width, p.height, e.clientX, e.clientY) : null;
    };
    canvas.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        canvas.focus({ preventScroll: true });
        const pt = platePoint(e);
        if (!pt) return;
        if (st.tab === "result") {
            if (!pt.inside && !st.result) return;
            wiping = true;
            canvas.setPointerCapture?.(e.pointerId);
            st.wipe = pt.x;
            draw();
            return;
        }
        if (!pt.inside || e.button !== 0) return;
        onPick(pt.x, pt.y, { shift: e.shiftKey, alt: e.altKey, ctrl: e.ctrlKey || e.metaKey });
    });
    canvas.addEventListener("pointermove", (e) => {
        if (!wiping) { hoverTag(e); return; }
        const pt = platePoint(e);
        if (pt) { st.wipe = pt.x; draw(); }
    });
    const endWipe = (e) => { if (wiping) { wiping = false; canvas.releasePointerCapture?.(e.pointerId); } };
    canvas.addEventListener("pointerup", endWipe);
    canvas.addEventListener("pointercancel", endWipe);
    canvas.addEventListener("pointerleave", () => { if (!wiping) updateTag(); });
    canvas.addEventListener("dblclick", (e) => { if (st.tab === "result") { st.wipe = 1; draw(); e.stopPropagation(); } });

    function hoverTag(e) {
        if (st.tab !== "pick" || st.detFrame !== st.frame || !st.objects.length) return;
        const pt = platePoint(e);
        const o = pt?.inside ? hitTest(st.objects, pt.x, pt.y) : null;
        if (o) { tag.hidden = false; tag.textContent = `${o.label} #${o.id + 1} · ${Math.round(o.score * 100)}%`; }
        else updateTag();
    }

    // ── drawing ─────────────────────────────────────────────────────────────────────────────────────────────────
    const plateImgs = imageCache(IMAGE_CACHE, () => draw());
    const resultImgs = imageCache(IMAGE_CACHE * 2, () => draw());
    const tintCache = new Map();
    let checker = null;
    let rafId = 0;

    function draw() {
        if (rafId) return;
        rafId = requestAnimationFrame(() => { rafId = 0; paint(); });
    }

    function sizeCanvas(c) {
        const cw = Math.max(1, c.clientWidth), ch = Math.max(1, c.clientHeight);
        const s = canvasBackingScale(cw, ch);
        const W = Math.max(1, Math.round(cw * s)), H = Math.max(1, Math.round(ch * s));
        if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
        return { cw, ch, s };
    }

    function tinted(key, mask, w, h, color, alpha) {
        const k = `${key}|${color}|${alpha}`;
        let c = tintCache.get(k);
        if (!c) { c = tintCanvas(mask, w, h, color, alpha); tintCache.set(k, c); }
        return c;
    }

    function paint() {
        const { cw, ch, s } = sizeCanvas(canvas);
        const g = canvas.getContext("2d");
        const T = colours();
        g.setTransform(1, 0, 0, 1, 0, 0);
        g.fillStyle = T.sunken;
        g.fillRect(0, 0, canvas.width, canvas.height);
        paintRuler();
        const p = st.plate;
        if (!p) return;
        g.setTransform(s, 0, 0, s, 0, 0);
        const r = fitRect(cw, ch, p.width, p.height);
        if (st.tab === "result" && st.result) paintResult(g, r, T);
        else paintPick(g, r, T);
    }

    function plateImageAt(index) {
        const f = nearestFrame(st.plate?.frames, index);
        return f ? plateImgs.get(viewUrl(api, f)) : null;
    }

    function paintPick(g, r, T) {
        const img = plateImageAt(st.frame);
        if (img) g.drawImage(img, r.x, r.y, r.w, r.h);
        const onDet = st.detFrame === st.frame && st.objects.length;
        const sel = st.sel;
        if (onDet) {
            const o0 = st.objects[0];
            if (sel?.subject && sel.frame === st.frame) {
                const key = `sel${st.selVer}`;
                if (!tintCache.has(`${key}|${T.accent}|0.5`)) {
                    for (const k of tintCache.keys()) if (k.startsWith("sel")) tintCache.delete(k);
                }
                const m = tintCache.has(`${key}|${T.accent}|0.5`) ? null : selectionMask(st.objects, sel, o0.w, o0.h);
                g.drawImage(tinted(key, m, o0.w, o0.h, T.accent, 0.5), r.x, r.y, r.w, r.h);
                for (const e of sel.excluded) {
                    const o = st.objects.find((x) => x.id === e.id);
                    if (o) g.drawImage(tinted(`o${o.id}`, o.mask, o.w, o.h, T.danger, 0.28), r.x, r.y, r.w, r.h);
                }
            } else {
                st.objects.forEach((o, i) => {
                    g.drawImage(tinted(`o${o.id}`, o.mask, o.w, o.h, T.series[i % T.series.length], 0.38), r.x, r.y, r.w, r.h);
                });
            }
        }
        if (sel?.subject && sel.frame === st.frame) paintMarkers(g, r, T, sel);
    }

    function paintMarkers(g, r, T, sel) {
        const at = (m) => [r.x + m.x * r.w, r.y + m.y * r.h];
        const ring = (x, y, fill, rad) => {
            g.beginPath();
            g.arc(x, y, rad, 0, Math.PI * 2);
            g.fillStyle = fill;
            g.fill();
            g.lineWidth = 1.5;
            g.strokeStyle = T.ground;
            g.stroke();
        };
        const glyph = (x, y, color, kind) => {
            ring(x, y, color, 6);
            g.strokeStyle = T.ground;
            g.lineWidth = 1.6;
            g.beginPath();
            if (kind === "+") { g.moveTo(x - 3, y); g.lineTo(x + 3, y); g.moveTo(x, y - 3); g.lineTo(x, y + 3); }
            else if (kind === "-") { g.moveTo(x - 3, y); g.lineTo(x + 3, y); }
            else { g.moveTo(x - 2.5, y - 2.5); g.lineTo(x + 2.5, y + 2.5); g.moveTo(x + 2.5, y - 2.5); g.lineTo(x - 2.5, y + 2.5); }
            g.stroke();
        };
        for (const e of sel.excluded) glyph(...at(e), T.danger, "x");
        for (const a of sel.add) glyph(...at(a), T.ok, "+");
        for (const c of sel.cut) glyph(...at(c), T.warn, "-");
        const [sx, sy] = at(sel.subject);
        ring(sx, sy, T.accent, 7);
        ring(sx, sy, T.ink, 2.5);
    }

    function resultFrame(index) {
        const fr = st.result?.frames;
        if (!fr?.length) return null;
        const f = nearestFrame(fr, index);
        return f;
    }

    function paintResult(g, r, T) {
        const res = st.result;
        const inRange = st.frame >= res.start && st.frame < res.start + res.count;
        const plateImg = plateImageAt(st.frame);
        if (!inRange) {
            if (plateImg) { g.globalAlpha = 0.35; g.drawImage(plateImg, r.x, r.y, r.w, r.h); g.globalAlpha = 1; }
            return;
        }
        const f = resultFrame(st.frame);
        const pImg = f ? resultImgs.get(viewUrl(api, f.plate)) : null;
        const aImg = f ? resultImgs.get(viewUrl(api, f.alpha)) : null;
        if (!pImg || !aImg) { if (plateImg) g.drawImage(plateImg, r.x, r.y, r.w, r.h); return; }
        // the cut-out: plate x alpha, at proxy size
        const cut = document.createElement("canvas");
        cut.width = pImg.naturalWidth;
        cut.height = pImg.naturalHeight;
        const cg = cut.getContext("2d");
        if (st.view === "alpha") {
            cg.fillStyle = "#000";
            cg.fillRect(0, 0, cut.width, cut.height);
            cg.drawImage(whiteOf(aImg), 0, 0, cut.width, cut.height);
        } else if (st.view === "overlay") {
            cg.drawImage(pImg, 0, 0);
            const tint = document.createElement("canvas");
            tint.width = cut.width;
            tint.height = cut.height;
            const tg = tint.getContext("2d");
            tg.drawImage(aImg, 0, 0, cut.width, cut.height);
            tg.globalCompositeOperation = "source-in";
            tg.fillStyle = T.accent;
            tg.fillRect(0, 0, cut.width, cut.height);
            cg.globalAlpha = 0.55;
            cg.drawImage(tint, 0, 0);
            cg.globalAlpha = 1;
        } else {
            cg.drawImage(pImg, 0, 0);
            cg.globalCompositeOperation = "destination-in";
            cg.drawImage(aImg, 0, 0, cut.width, cut.height);
            cg.globalCompositeOperation = "source-over";
        }
        // ground under the cut-out
        if (st.view === "checker" || st.view === "comp") {
            if (st.view === "checker") {
                g.fillStyle = checkerPattern(g);
            } else {
                g.fillStyle = T.ground;
            }
            g.fillRect(r.x, r.y, r.w, r.h);
        }
        g.drawImage(cut, r.x, r.y, r.w, r.h);
        // the wipe: the plate left of the line
        if (st.wipe < 1) {
            const wx = r.x + st.wipe * r.w;
            g.save();
            g.beginPath();
            g.rect(r.x, r.y, wx - r.x, r.h);
            g.clip();
            g.drawImage(pImg, r.x, r.y, r.w, r.h);
            g.restore();
            g.strokeStyle = T.ink;
            g.lineWidth = 1;
            g.beginPath();
            g.moveTo(wx, r.y);
            g.lineTo(wx, r.y + r.h);
            g.stroke();
        }
    }

    const _white = new WeakMap();
    function whiteOf(aImg) {                 // the alpha PNG as white-on-transparent (its colour channels may be anything)
        let c = _white.get(aImg);
        if (c) return c;
        c = document.createElement("canvas");
        c.width = aImg.naturalWidth;
        c.height = aImg.naturalHeight;
        const g = c.getContext("2d");
        g.drawImage(aImg, 0, 0);
        g.globalCompositeOperation = "source-in";
        g.fillStyle = "#fff";
        g.fillRect(0, 0, c.width, c.height);
        _white.set(aImg, c);
        return c;
    }

    function checkerPattern(g) {
        if (!checker) {
            checker = document.createElement("canvas");
            checker.width = checker.height = 16;
            const cg = checker.getContext("2d");
            cg.fillStyle = "#6b6b75";
            cg.fillRect(0, 0, 16, 16);
            cg.fillStyle = "#9a9aa6";
            cg.fillRect(0, 0, 8, 8);
            cg.fillRect(8, 8, 8, 8);
        }
        return g.createPattern(checker, "repeat");
    }

    function paintRuler() {
        const { cw, ch, s } = sizeCanvas(ruler);
        const g = ruler.getContext("2d");
        const T = colours();
        g.setTransform(s, 0, 0, s, 0, 0);
        g.clearRect(0, 0, cw, ch);
        const n = plateFrames();
        if (!n) return;
        const x0 = 4, x1 = cw - 4;
        const xOf = (f) => x0 + (n > 1 ? f / (n - 1) : 0) * (x1 - x0);
        const res = st.result;
        if (res) {                                         // the rendered range (and the empty part of it)
            g.fillStyle = T.accent;
            g.globalAlpha = st.resultStale ? 0.25 : 0.5;
            g.fillRect(xOf(res.start), 0, Math.max(2, xOf(res.start + res.count - 1) - xOf(res.start)), 3);
            if (res.empty) {
                g.fillStyle = T.warn;
                g.fillRect(xOf(res.empty[0]), 0, Math.max(2, xOf(res.empty[1] - 1) - xOf(res.empty[0])), 3);
            }
            g.globalAlpha = 1;
        }
        g.fillStyle = T.dim;
        g.font = "9px system-ui, sans-serif";
        g.textBaseline = "bottom";
        for (const t of rulerTicks(n, 24, x1 - x0)) {
            const x = xOf(t.frame);
            g.fillRect(Math.round(x), t.major ? 4 : 6, 1, t.major ? 4 : 2);
            if (t.label) g.fillText(t.label, Math.min(x + 2, cw - 24), ch);
        }
        if (st.sel?.subject) {                             // the pick frame
            const x = xOf(st.sel.frame);
            g.fillStyle = T.accent;
            g.beginPath();
            g.moveTo(x - 4, 3); g.lineTo(x + 4, 3); g.lineTo(x, 9); g.closePath();
            g.fill();
        }
        g.fillStyle = T.ink;                               // the current frame
        g.fillRect(Math.round(xOf(st.frame)) - 1, 0, 2, 10);
    }

    // ── the rest of the panel ───────────────────────────────────────────────────────────────────────────────────
    function updateTag() {
        if (!st.plate) { tag.hidden = true; return; }
        let t = frameLabel(st.frame, st.plate.count);
        if (st.tab === "result" && st.result) {
            const res = st.result;
            if (st.frame < res.start || st.frame >= res.start + res.count) t += " · not rendered";
            else if (res.empty && st.frame >= res.empty[0] && st.frame < res.empty[1]) t += " · empty (before the pick frame)";
            if (st.resultStale) t += " · selection changed since";
        } else if (st.sel?.subject && st.sel.frame !== st.frame) {
            t += ` · pick frame ${st.sel.frame}`;
        }
        tag.hidden = false;
        tag.textContent = t;
    }

    function setPill(bar, value) {
        for (const b of bar.querySelectorAll(".c2c-ui-pill")) {
            const on = b.dataset.value === value;
            b.classList.toggle("c2c-ui-pill--active", on);
            b.setAttribute("aria-pressed", on ? "true" : "false");
        }
    }

    function update() {
        const p = st.plate;
        const pick = pickStep();
        const sp = stepper({ plate: !!p, subject: !!st.sel?.subject, preview: st.previewDone, result: !!st.result,
            busy: st.busy, pickStep: pick }, previewN());
        steps.replaceChildren(...sp.steps.map((s, i) => {
            const chip = el("button", `c2cv-step c2cv-step--${s.state}`, `${s.state === "done" ? "✓" : i + 1} ${s.label}`);
            chip.type = "button";
            chip.style.cursor = st.busy ? "default" : "pointer";
            chip.disabled = st.busy;
            chip.title = STEP_TIPS[s.id] || "";
            chip.setAttribute("aria-current", s.state === "current" ? "step" : "false");
            chip.addEventListener("click", () => stepClicked(s.id));
            return chip;
        }));
        // actions: the stepper's one next action; Cancel while busy
        primaryAction = sp.action.id;
        primary.lastChild.textContent = st.busy ? "Cancel" : sp.action.label;
        primary.classList.toggle("c2c-ui-btn--danger", st.busy);
        primary.classList.toggle("c2c-ui-btn--primary", !st.busy);
        primary.disabled = false;
        primary.title = "";
        if (sp.action.id === "find") {
            // the Find button sits next to the box: the prominent button shows what comes next, once there is a subject
            primary.lastChild.textContent = `Preview ${previewN()} frames`;
            primary.disabled = true;
            primary.title = "Find the subject (type what it is) or click it on the frame first.";
        }
        if (st.busy) { primaryAction = "wait"; primary.disabled = false; primary.lastChild.textContent = "Cancel"; }
        secondaryAction = null;
        if (!st.busy && p && sp.current === "render" && sp.action.id === "preview") secondaryAction = "full";
        else if (!st.busy && p && sp.action.id === "find") secondaryAction = null;
        else if (!st.busy && p && sp.action.id === "full") secondaryAction = "preview";
        secondary.hidden = !secondaryAction;
        if (secondaryAction === "full") secondary.lastChild.textContent = "Render full shot";
        else if (secondaryAction === "preview") secondary.lastChild.textContent = `Preview ${previewN()} again`;
        secondary.disabled = st.busy;
        nBox.value = String(previewN());
        nBox.disabled = st.busy;
        // scrubber
        const n = plateFrames();
        slider.max = String(Math.max(0, n - 1));
        slider.value = String(st.frame);
        slider.disabled = !n;
        prevBtn.disabled = !n || st.frame <= 0;
        nextBtn.disabled = !n || st.frame >= n - 1;
        frameBox.value = n ? String(st.frame) : "";
        frameBox.disabled = !n;
        // find
        findRow.hidden = false;
        findBox.disabled = !p || !pick || st.busy;
        findBtn.disabled = !p || !pick || st.busy;
        clearBtn.disabled = !st.sel || st.busy;
        if (document.activeElement !== findBox && !findBox.value && st.text) findBox.value = st.text;
        modelSel.disabled = st.busy || !pick;
        // message on the frame
        if (!p) {
            msg.hidden = false;
            msg.textContent = "Mask + Matting viewer. Connect a plate to the image input, then press Load plate.";
        } else if (st.tab === "result" && !st.result) {
            msg.hidden = false;
            msg.textContent = "No render yet. Preview a few frames from the pick frame, then render the full shot.";
        } else {
            msg.hidden = true;
        }
        // tools + qc
        setPill(tabBar, st.tab);
        setPill(viewBar, st.view);
        viewBar.style.visibility = st.tab === "result" ? "visible" : "hidden";
        const res = st.result;
        resultInfo.textContent = res ? `${res.preview ? "Preview" : "Full shot"} ${res.start}–${res.start + res.count - 1}${st.resultStale ? " · selection changed" : ""}` : "";
        const q = res?.qc;
        for (const [name, c] of Object.entries(qcChips)) {
            const metric = q?.[name];
            const t = q?.thresholds?.[name];
            const lvl = qcLevel(metric?.value, t);
            c.dot.className = `c2cv-dot${lvl === "n/a" ? "" : ` c2cv-dot--${lvl}`}`;
            c.label.textContent = `${QC_NAMES[name]} ${metric ? qcValueText(name, metric.value) : "n/a"}`;
            c.chip.title = metric ? qcTooltip(name, metric, t, q?.qc_side) : `${QC_NAMES[name]}: shown after a render.`;
        }
        advBtn.lastChild.textContent = advancedOpen() ? "All settings ▾" : "All settings ▸";
        advLabel.textContent = st.wrapper ? `Driving the copy in subgraph ${st.wrapper}` : "";
        updateTag();
        syncViewHeight();
        draw();
    }

    // ── size, zoom, lifetime ────────────────────────────────────────────────────────────────────────────────────
    /** The frame view's height for an inner width: the plate's aspect (16:9 before a plate), clamped. */
    function viewHeight(inner) {
        const p = st.plate;
        const aspect = p ? p.width / Math.max(1, p.height) : 16 / 9;
        return Math.min(VIEW_MAX, Math.max(VIEW_MIN, Math.round(Math.max(120, inner) / aspect)));
    }
    function heightFor(width) {
        return FIXED + viewHeight((width || 400) - 8 - PAD * 2 - 2);
    }
    /** The view's minimum height follows the panel's own width: Nodes 2.0 fits the node to its content, so a view
     *  that only flexed into the space shrank to VIEW_MIN there (live check 2026-10-29). */
    function syncViewHeight() {
        const inner = root.clientWidth - PAD * 2 - 2;
        if (inner <= 0) return;
        const h = `${viewHeight(inner)}px`;
        if (view.style.minHeight === h) return;
        view.style.minHeight = h;
        opts.onLayout?.();                               // the node refits (a taller view must not clip the rows)
    }

    const offZoom = onZoomChange(() => draw());
    let ro = null;
    if (typeof ResizeObserver !== "undefined") {
        ro = new ResizeObserver(() => { syncViewHeight(); draw(); });
        ro.observe(view);
        ro.observe(root);
    }

    function syncFromWidget() {
        const got = fromPicks(opts.getPicks?.() || "");
        if (got) {
            st.sel = got.sel;
            st.text = got.text;
            findBox.value = got.text;
            if (got.model) st.model = got.model;
            st.frame = got.sel.frame;
        } else {
            st.sel = null;
        }
        update();
    }

    function destroy() {
        try { runner.cancel(); } catch (_e) { /* ignore */ }
        try { offZoom(); } catch (_e) { /* ignore */ }
        try { ro?.disconnect(); } catch (_e) { /* ignore */ }
        if (rafId) cancelAnimationFrame(rafId);
        plateImgs.clear();
        resultImgs.clear();
        tintCache.clear();
    }

    setStatus("");
    loadModels();
    syncFromWidget();
    return { root, heightFor, syncFromWidget, destroy, get busy() { return st.busy; } };
}
