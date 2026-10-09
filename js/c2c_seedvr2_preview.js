/**
 * SeedVR2 preview (L7.57) - a live view on the numz "SeedVR2 Video Upscaler" node; its code is not changed.
 *
 * nodes/seedvr2_preview.py sends the last frame of every decoded batch with the input frame at the same index
 * ("c2c.seedvr2.preview") and, when the run ends, keeps a small JPEG strip of every frame ("c2c.seedvr2.done",
 * GET /c2c/seedvr2/frame). This widget shows them: a before/after wipe while it runs, then a frame slider to scrub;
 * views Wipe / After / Before / Difference (|after - before|, amplified) and Alpha (RGBA input).
 * Setting: Settings > C2C > Video > SeedVR2 preview.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ensureStyles, pillBar, statusLine, emptyState, measureRootContent } from "./c2c_ui/index.js";

const NODE = "SeedVR2VideoUpscaler";
const SETTING = "c2c.seedvr2.preview";
const STYLE_ID = "c2c-seedvr2-preview-v1";
const MARGIN = 2;
const BAR_H = 28;
const STATUS_H = 22;
const SCRUB_H = 30;
const GAP = 6;
let _enabled = true;

function ensureLocalStyles() {
    ensureStyles();
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.c2c-s2 { display:flex; flex-direction:column; gap:${GAP}px; width:100%; height:100%; box-sizing:border-box;
  pointer-events:auto; font: 12px/1.3 var(--c2c-font, system-ui, sans-serif); }
.c2c-s2__view { position:relative; flex:0 0 auto; width:100%; aspect-ratio: var(--c2c-s2-aspect, 1.7778);
  overflow:hidden; border-radius:6px;
  background: var(--c2c-neutral990, #050608); touch-action:none; cursor: ew-resize; outline:none; }
.c2c-s2__view:focus-visible { box-shadow: 0 0 0 2px var(--c2c-accent, #719CB5); }
.c2c-s2__view img, .c2c-s2__view canvas { position:absolute; inset:0; width:100%; height:100%; object-fit:contain;
  user-select:none; -webkit-user-drag:none; pointer-events:none; }
.c2c-s2__handle { position:absolute; top:0; bottom:0; width:2px; margin-left:-1px; background: var(--c2c-accent, #719CB5);
  pointer-events:none; box-shadow: 0 0 0 1px rgba(5,6,8,.6); }
.c2c-s2__handle::after { content:""; position:absolute; top:50%; left:50%; width:14px; height:14px; margin:-7px 0 0 -7px;
  border-radius:50%; background: var(--c2c-accent, #719CB5); box-shadow: 0 0 0 2px rgba(5,6,8,.7); }
.c2c-s2__tag { position:absolute; top:6px; padding:1px 6px; border-radius:4px; font-size:10px; letter-spacing:.04em;
  text-transform:uppercase; background: rgba(5,6,8,.72); color: var(--c2c-text-muted, #8995A1); pointer-events:none; }
.c2c-s2__scrub { display:flex; align-items:center; gap:8px; height:${SCRUB_H - 6}px; }
.c2c-s2__scrub input { flex:1 1 auto; min-width:0; }
.c2c-s2__scrub span { min-width:72px; text-align:right; font-variant-numeric: tabular-nums;
  color: var(--c2c-text-muted, #8995A1); }
.c2c-s2 [hidden] { display:none !important; }
`;
    document.head.appendChild(s);
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.decoding = "async";
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("image failed to load"));
        img.src = src;
    });
}

/** |after - before| amplified 4x, as a cool ramp (black -> Ice Moon -> white), at the after frame's size. */
function differenceCanvas(before, after) {
    const w = after.naturalWidth, h = after.naturalHeight;
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(before, 0, 0, w, h);
    const b = g.getImageData(0, 0, w, h).data;
    g.drawImage(after, 0, 0, w, h);
    const out = g.getImageData(0, 0, w, h);
    const a = out.data;
    for (let i = 0; i < a.length; i += 4) {
        const d = Math.min(1, (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2])) / 765 * 4);
        a[i] = 113 * d + 142 * d * d; a[i + 1] = 156 * d + 99 * d * d; a[i + 2] = 181 * d + 74 * d * d; a[i + 3] = 255;
    }
    g.putImageData(out, 0, 0);
    return c;
}

/** The after frame with the transparent parts (alpha < 1) tinted, so the matte reads at a glance. */
function alphaCanvas(after, alpha) {
    const w = after.naturalWidth, h = after.naturalHeight;
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(alpha, 0, 0, w, h);
    const m = g.getImageData(0, 0, w, h).data;
    g.drawImage(after, 0, 0, w, h);
    const out = g.getImageData(0, 0, w, h);
    const p = out.data;
    for (let i = 0; i < p.length; i += 4) {
        const t = (1 - m[i] / 255) * 0.65;                         // tint strength = how transparent
        p[i] = p[i] * (1 - t) + 101 * t; p[i + 1] = p[i + 1] * (1 - t) + 11 * t; p[i + 2] = p[i + 2] * (1 - t) + 18 * t;
    }
    g.putImageData(out, 0, 0);
    return c;
}

function buildView(node) {
    ensureLocalStyles();
    const root = document.createElement("div");
    root.className = "c2c-s2";

    const state = { mode: "wipe", pos: 0.5, aspect: 16 / 9, before: null, after: null, alpha: null,
                    run: null, live: false, frames: [], hasBefore: false, hasAlpha: false, loadToken: 0 };

    const modes = [
        { value: "wipe", label: "Wipe" }, { value: "after", label: "After" }, { value: "before", label: "Before" },
        { value: "diff", label: "Difference" }, { value: "alpha", label: "Alpha" },
    ];
    const bar = pillBar(modes, { value: "wipe", onChange: (v) => { state.mode = v; render(); } });
    const alphaPill = bar.querySelector('[data-value="alpha"]');
    alphaPill.hidden = true;

    const status = statusLine();
    status.setText("Runs live while SeedVR2 decodes.", "default");

    const view = document.createElement("div");
    view.className = "c2c-s2__view";
    view.tabIndex = 0;
    view.setAttribute("role", "slider");
    view.setAttribute("aria-label", "Before / after wipe (drag, or arrow keys)");
    view.setAttribute("aria-valuemin", "0");
    view.setAttribute("aria-valuemax", "100");
    const empty = emptyState({ title: "SeedVR2 preview", hint: "Queue the workflow to watch the upscale frame by frame." });
    view.appendChild(empty);

    const scrub = document.createElement("div");
    scrub.className = "c2c-s2__scrub";
    scrub.hidden = true;
    const range = document.createElement("input");
    range.type = "range"; range.min = "0"; range.step = "1"; range.value = "0";
    range.setAttribute("aria-label", "Frame");
    const label = document.createElement("span");
    scrub.append(range, label);

    root.append(bar, status.el, view, scrub);

    // ── drawing ──
    function layer(el, clipLeftPct) {
        if (clipLeftPct != null) el.style.clipPath = `inset(0 0 0 ${clipLeftPct}%)`;
        view.appendChild(el);
    }
    function tag(text, side) {
        const t = document.createElement("div");
        t.className = "c2c-s2__tag";
        t.textContent = text;
        t.style[side] = "6px";
        view.appendChild(t);
    }
    function render() {
        if (!state.after) return;
        view.replaceChildren();
        const pct = Math.round(state.pos * 1000) / 10;
        view.setAttribute("aria-valuenow", String(Math.round(pct)));
        const mode = state.mode === "alpha" && !state.alpha ? "after" : state.mode;
        if (mode === "wipe" && state.before) {
            layer(state.before.cloneNode());
            layer(state.after.cloneNode(), pct);
            const h = document.createElement("div");
            h.className = "c2c-s2__handle";
            h.style.left = `${pct}%`;
            view.appendChild(h);
            tag("Before", "left"); tag("After", "right");
        } else if (mode === "before" && state.before) {
            layer(state.before.cloneNode()); tag("Before", "left");
        } else if (mode === "diff" && state.before) {
            layer(differenceCanvas(state.before, state.after)); tag("Difference x4", "left");
        } else if (mode === "alpha" && state.alpha) {
            layer(alphaCanvas(state.after, state.alpha)); tag("Alpha (tinted = transparent)", "left");
        } else {
            layer(state.after.cloneNode()); tag("After", "right");
        }
    }
    function setAspect(w, h) {
        const a = w && h ? w / h : state.aspect;
        if (Math.abs(a - state.aspect) < 1e-3) return;
        state.aspect = a;
        view.style.setProperty("--c2c-s2-aspect", String(a));   // the view takes the frame's shape: no letterbox
        fit();
    }

    // ── interaction: drag / arrows move the wipe ──
    const setPos = (clientX) => {
        const r = view.getBoundingClientRect();
        if (r.width <= 0) return;
        state.pos = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
        if (state.mode !== "wipe") return;
        render();
    };
    view.addEventListener("pointerdown", (e) => {
        if (!state.after) return;
        e.stopPropagation(); e.preventDefault();
        view.setPointerCapture?.(e.pointerId);
        view.focus({ preventScroll: true });
        setPos(e.clientX);
    });
    view.addEventListener("pointermove", (e) => {
        if (view.hasPointerCapture?.(e.pointerId)) { e.stopPropagation(); setPos(e.clientX); }
    });
    view.addEventListener("pointerup", (e) => { view.releasePointerCapture?.(e.pointerId); e.stopPropagation(); });
    view.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
        e.preventDefault(); e.stopPropagation();
        state.pos = Math.min(1, Math.max(0, state.pos + (e.key === "ArrowLeft" ? -0.05 : 0.05)));
        render();
    });

    // ── scrub ──
    const frameURL = (i, which) => api.apiURL(
        `/c2c/seedvr2/frame?key=${encodeURIComponent(state.run.key)}&i=${i}&which=${which}`);
    async function showFrame(i) {
        const token = ++state.loadToken;
        label.textContent = `frame ${state.frames[i] + 1} / ${state.frames[state.frames.length - 1] + 1}`;
        try {
            const jobs = [loadImage(frameURL(i, "after")),
                          state.hasBefore ? loadImage(frameURL(i, "before")) : Promise.resolve(null),
                          state.hasAlpha ? loadImage(frameURL(i, "alpha")) : Promise.resolve(null)];
            const [after, before, alpha] = await Promise.all(jobs);
            if (token !== state.loadToken) return;          // a newer frame was asked for meanwhile
            Object.assign(state, { after, before, alpha });
            render();
        } catch {
            if (token === state.loadToken) status.setText("That preview is no longer kept; run the node again.", "warn");
        }
    }
    range.addEventListener("input", () => showFrame(Number(range.value)));
    range.addEventListener("pointerdown", (e) => e.stopPropagation());

    // ── events from the backend ──
    async function onLive(d) {
        if (state.run?.prompt_id !== d.prompt_id) {
            state.run = { prompt_id: d.prompt_id, key: `${d.prompt_id}:${d.node}` };
            scrub.hidden = true;
            alphaPill.hidden = true;
        }
        state.live = true;
        status.setText(`Decoding batch ${d.batch} of ${d.batches} · frame ${d.frame + 1} of ${d.total_frames}`, "default");
        try {
            const [after, before] = await Promise.all([loadImage(d.after), d.before ? loadImage(d.before) : null]);
            Object.assign(state, { after, before, alpha: null });
            setAspect(after.naturalWidth, after.naturalHeight);
            render();
        } catch { /* a dropped live frame is replaced by the next one */ }
    }
    function onDone(d) {
        state.live = false;
        state.run = { prompt_id: d.prompt_id, key: d.key };
        state.frames = d.frames || [];
        state.hasBefore = !!d.before;
        state.hasAlpha = !!d.alpha;
        alphaPill.hidden = !state.hasAlpha;
        setAspect(d.width, d.height);
        if (state.frames.length > 0) {
            range.max = String(state.frames.length - 1);
            range.value = String(state.frames.length - 1);
            scrub.hidden = false;
            showFrame(state.frames.length - 1);
        }
        status.setText(`Done · ${d.count} frame${d.count === 1 ? "" : "s"} kept for scrubbing`
            + (state.hasBefore ? " · drag to compare" : ""), "ok");
        fit();
    }

    // ── size ──
    function height(width) {
        // the view's own layout width (offsetWidth ignores canvas zoom); before it is laid out, the node's width
        const inner = view.offsetWidth || Math.max(160, (width || node.size?.[0] || 360) - 2 * MARGIN);
        const viewH = Math.round(inner / state.aspect);
        if (root.offsetParent !== null && view.offsetHeight > 0) {
            // laid out: the rows as they really render, with the view at this width's height
            return Math.ceil(measureRootContent(root) - view.offsetHeight + viewH) + 2 * MARGIN;
        }
        const rows = scrub.hidden ? 3 : 4;              // before layout: the heights the styles aim for
        return BAR_H + STATUS_H + viewH + (scrub.hidden ? 0 : SCRUB_H) + (rows - 1) * GAP + 2 * MARGIN;
    }
    function fit() {
        // the node follows its content when the frame shape or the scrub row changes (not on every frame)
        requestAnimationFrame(() => {
            const need = node.computeSize?.();
            if (need && Math.abs(node.size[1] - need[1]) > 1) node.setSize([node.size[0], need[1]]);
            node.setDirtyCanvas?.(true, true);
        });
    }

    return { root, onLive, onDone, height };
}

function mount(node) {
    const ctl = buildView(node);
    const w = node.addDOMWidget("c2c_seedvr2_preview", "div", ctl.root, {
        serialize: false, margin: MARGIN, getMinHeight: () => ctl.height(node.size?.[0]),
    });
    w.computeSize = (width) => [width, ctl.height(width)];
    node.__c2cSeedVR2View = ctl;         // on the node: its id is only assigned when it is added to a graph
}

/** The widget of the node the backend names ("12", or "5:12" inside a subgraph): the graph on screen first. */
function viewFor(d) {
    const id = String(d?.node ?? "").split(":").pop();
    if (!id) return null;
    for (const g of [app.canvas?.graph, app.graph]) {
        const n = g?.getNodeById?.(id) ?? g?.getNodeById?.(Number(id));
        if (n?.__c2cSeedVR2View) return n.__c2cSeedVR2View;
    }
    return null;
}

if (!globalThis.__c2cSeedVR2PreviewExt) {
    globalThis.__c2cSeedVR2PreviewExt = true;
    app.registerExtension({
        name: "C2C.SeedVR2Preview",
        settings: [{
            id: SETTING,
            name: "SeedVR2 preview",
            tooltip: "A live before/after view on the SeedVR2 Video Upscaler node while it runs, then a frame slider. "
                + "Off = the node as it ships.",
            type: "boolean",
            defaultValue: true,
            category: ["c2c", "Video", "SeedVR2 preview"],
            onChange: (v) => { _enabled = v !== false; },
        }],
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData?.name !== NODE) return;
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...a) {
                const r = created?.apply(this, a);
                if (_enabled && !this.__c2cSeedVR2Preview) {
                    this.__c2cSeedVR2Preview = true;
                    mount(this);
                }
                return r;
            };
        },
        async setup() {
            _enabled = app.ui.settings.getSettingValue(SETTING) !== false;
            api.addEventListener("c2c.seedvr2.preview", (e) => { if (_enabled) viewFor(e.detail)?.onLive(e.detail); });
            api.addEventListener("c2c.seedvr2.done", (e) => { if (_enabled) viewFor(e.detail)?.onDone(e.detail); });
        },
    });
}
