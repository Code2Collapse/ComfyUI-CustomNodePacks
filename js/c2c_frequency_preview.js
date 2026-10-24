/**
 * Frequency Separate (MEC) preview (L9.39 - the live preview had been dropped).
 *
 * After a run the node sends its first frame's input, base (low frequency) and detail (high frequency) as small
 * PNGs. While the settings still match that run, the panel shows those exact images. Change radius / mode / detail
 * and it recomputes a live approximation in the browser - a Gaussian base with the same radius (sigma = radius / 2,
 * as the node's Gaussian method), the detail divided or subtracted the way the node encodes it - labelled as such,
 * until the next run brings the exact result of the chosen method (Guided, Rolling Guidance, Median) back.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "FrequencySeparateMEC";
const VIEWS = [["input", "Input"], ["base", "Base"], ["detail", "Detail"], ["split", "Base | Detail"]];
const HEIGHT = 260;

const STYLE = `
.c2c-freq{display:flex;flex-direction:column;gap:4px;height:100%;box-sizing:border-box;padding:4px;
  font:11px var(--c2c-font,system-ui,sans-serif);color:var(--c2c-fg,#ddd)}
.c2c-freq [hidden]{display:none!important}
.c2c-freq .bar{display:flex;gap:3px;flex-wrap:wrap}
.c2c-freq button{font:inherit;padding:2px 7px;border-radius:4px;cursor:pointer;color:inherit;
  background:color-mix(in srgb,var(--c2c-surface0,#333) 70%,transparent);border:1px solid var(--c2c-border,#444)}
.c2c-freq button[aria-pressed=true]{border-color:var(--c2c-blue,#6aa0ff);color:var(--c2c-fg,#fff)}
.c2c-freq .stage{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;
  background:var(--c2c-bg3,#111);border-radius:4px;overflow:hidden}
.c2c-freq canvas{max-width:100%;max-height:100%;image-rendering:auto}
.c2c-freq .note{color:var(--c2c-gray400,#999);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.c2c-freq .empty{color:var(--c2c-gray500,#888);text-align:center;padding:12px}
`;

function ensureStyle() {
    if (document.getElementById("c2c-freq-style")) return;
    const s = document.createElement("style");
    s.id = "c2c-freq-style";
    s.textContent = STYLE;
    document.head.appendChild(s);
}

function viewUrl(f) {
    const q = new URLSearchParams({ filename: f.filename, subfolder: f.subfolder || "", type: f.type || "temp" });
    return api.apiURL(`/view?${q}`);
}

function loadImg(url) {
    return new Promise((res, rej) => {
        const im = new Image();
        im.onload = () => res(im);
        im.onerror = rej;
        im.src = url;
    });
}

function settingsOf(node) {
    const W = (n) => node.widgets?.find((w) => w.name === n)?.value;
    return { method: W("method"), radius: Number(W("radius")), mode: W("mode"), detail: W("detail") };
}

/** Live approximation: Gaussian base (canvas blur, sigma = radius/2 scaled to the preview) and the detail. */
function approximate(src, s, scale) {
    const w = src.naturalWidth, h = src.naturalHeight;
    const sigma = Math.max(0.5, (s.radius / 2) * scale);
    // edge-replicated pad (the node pads by reflection; transparent borders would fade the base at the edges)
    const m = Math.ceil(3 * sigma) + 1;
    const pad = document.createElement("canvas");
    pad.width = w + 2 * m; pad.height = h + 2 * m;
    const px = pad.getContext("2d");
    px.drawImage(src, 0, 0, w, h, m, m, w, h);
    px.drawImage(src, 0, 0, 1, h, 0, m, m, h);                 // left
    px.drawImage(src, w - 1, 0, 1, h, m + w, m, m, h);         // right
    px.drawImage(pad, 0, m, pad.width, 1, 0, 0, pad.width, m);              // top (incl. corners)
    px.drawImage(pad, 0, m + h - 1, pad.width, 1, 0, m + h, pad.width, m);  // bottom
    const blurred = document.createElement("canvas");
    blurred.width = pad.width; blurred.height = pad.height;
    const blx = blurred.getContext("2d");
    blx.filter = `blur(${sigma}px)`;
    blx.drawImage(pad, 0, 0);
    blx.filter = "none";
    const base = document.createElement("canvas");
    base.width = w; base.height = h;
    const bx = base.getContext("2d", { willReadFrequently: true });
    bx.drawImage(blurred, m, m, w, h, 0, 0, w, h);
    const org = document.createElement("canvas");
    org.width = w; org.height = h;
    const ox = org.getContext("2d", { willReadFrequently: true });
    ox.drawImage(src, 0, 0);
    const a = ox.getImageData(0, 0, w, h), b = bx.getImageData(0, 0, w, h);
    const det = ox.createImageData(w, h);
    const luma = s.detail === "Luminance";
    for (let i = 0; i < a.data.length; i += 4) {
        const ar = a.data[i] / 255, ag = a.data[i + 1] / 255, ab = a.data[i + 2] / 255;
        const br = b.data[i] / 255, bg = b.data[i + 1] / 255, bb = b.data[i + 2] / 255;
        let r, g, bl;
        if (luma) {
            const la = 0.2126 * ar + 0.7152 * ag + 0.0722 * ab, lb = 0.2126 * br + 0.7152 * bg + 0.0722 * bb;
            const v = s.mode === "Divide" ? 0.5 * la / (lb + 1e-4) : la - lb + 0.5;
            r = g = bl = v;
        } else if (s.mode === "Divide") {
            r = 0.5 * ar / (br + 1e-4); g = 0.5 * ag / (bg + 1e-4); bl = 0.5 * ab / (bb + 1e-4);
        } else {
            r = ar - br + 0.5; g = ag - bg + 0.5; bl = ab - bb + 0.5;
        }
        det.data[i] = Math.max(0, Math.min(255, r * 255));
        det.data[i + 1] = Math.max(0, Math.min(255, g * 255));
        det.data[i + 2] = Math.max(0, Math.min(255, bl * 255));
        det.data[i + 3] = 255;
    }
    const detail = document.createElement("canvas");
    detail.width = w; detail.height = h;
    detail.getContext("2d").putImageData(det, 0, 0);
    return { base, detail };
}

function attach(node) {
    if (node.__c2cFreq) return;
    ensureStyle();
    const root = document.createElement("div");
    root.className = "c2c-freq";
    root.innerHTML = `<div class="bar"></div><div class="stage"><div class="empty">Run the node once to see its
        input split into base and detail. Then drag radius or switch mode: the preview follows live.</div>
        <canvas hidden></canvas></div><div class="note"></div>`;
    const bar = root.querySelector(".bar"), canvas = root.querySelector("canvas");
    const empty = root.querySelector(".empty"), note = root.querySelector(".note");
    const st = { view: "split", imgs: null, run: null, approx: null, key: "" };
    for (const [v, label] of VIEWS) {
        const b = document.createElement("button");
        b.textContent = label;
        b.dataset.view = v;
        b.addEventListener("click", () => { st.view = v; draw(); });
        bar.appendChild(b);
    }

    function draw() {
        for (const b of bar.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.view === st.view));
        if (!st.imgs) { empty.hidden = false; canvas.hidden = true; return; }
        empty.hidden = true; canvas.hidden = false;
        const s = settingsOf(node);
        const exact = st.run && s.radius === st.run.radius && s.mode === st.run.mode && s.detail === st.run.detail
            && s.method === st.run.method;
        let base = st.imgs.base, detail = st.imgs.detail;
        if (!exact) {
            const key = `${s.radius}|${s.mode}|${s.detail}`;
            if (st.key !== key) {
                const scale = st.run ? st.run.preview_width / Math.max(1, st.run.source_width) : 1;
                st.approx = approximate(st.imgs.input, s, scale);
                st.key = key;
            }
            base = st.approx.base; detail = st.approx.detail;
        }
        const src = st.imgs.input;
        const w = src.naturalWidth, h = src.naturalHeight;
        canvas.width = w; canvas.height = h;
        const cx = canvas.getContext("2d");
        if (st.view === "input") cx.drawImage(src, 0, 0);
        else if (st.view === "base") cx.drawImage(base, 0, 0, w, h);
        else if (st.view === "detail") cx.drawImage(detail, 0, 0, w, h);
        else {
            cx.drawImage(base, 0, 0, w, h);
            cx.drawImage(detail, w / 2, 0, w / 2, h, w / 2, 0, w / 2, h);
            cx.fillStyle = "#ffffff";
            cx.fillRect(Math.floor(w / 2), 0, 1, h);
        }
        note.textContent = exact
            ? `${st.run.method}, radius ${st.run.radius}, ${st.run.mode}, ${st.run.detail}: the node's own result (frame 1)`
            : `Live approximation (Gaussian base, radius ${s.radius}, ${s.mode}). Run for the exact ${s.method} result.`;
    }

    const widget = node.addDOMWidget("frequency_preview", "c2c_freq_preview", root, {
        serialize: false, hideOnZoom: false, getMinHeight: () => HEIGHT, getHeight: () => HEIGHT,
    });
    widget.computeSize = (w) => [w, HEIGHT];
    for (const name of ["radius", "mode", "detail", "method"]) {
        const w = node.widgets?.find((x) => x.name === name);
        if (!w || w.__c2cFreqHooked) continue;
        w.__c2cFreqHooked = true;
        const orig = w.callback;
        w.callback = function (...a) { const r = orig?.apply(this, a); draw(); return r; };
    }
    node.__c2cFreq = {
        async show(files, settings) {
            try {
                const by = Object.fromEntries(files.map((f) => [f.kind, f]));
                const [input, base, detail] = await Promise.all(["input", "base", "detail"].map((k) => loadImg(viewUrl(by[k]))));
                st.imgs = { input, base, detail };
                st.run = settings || null;
                st.key = "";
                draw();
            } catch (err) {
                note.textContent = "Preview images could not be loaded.";
            }
        },
        draw,
    };
    draw();
}

if (!globalThis.__c2cFrequencyPreviewExt) {
    globalThis.__c2cFrequencyPreviewExt = true;
    app.registerExtension({
        name: "C2C.FrequencyPreview",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== NODE) return;
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...args) {
                const r = onCreated?.apply(this, args);
                attach(this);
                return r;
            };
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecuted?.apply(this, arguments);
                const files = message?.c2c_freq;
                if (Array.isArray(files) && files.length) {
                    attach(this);
                    this.__c2cFreq.show(files, message.c2c_freq_settings?.[0]);
                }
                return r;
            };
        },
    });
}
