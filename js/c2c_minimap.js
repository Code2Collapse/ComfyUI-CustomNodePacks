// c2c_minimap.js — bottom-right canvas minimap (C2C)
// ---------------------------------------------------------------------
// What it does:
//   • Draws a live thumbnail of the entire graph in a fixed-position
//     canvas at bottom-right (≈220×140 px, resizable in 4 sizes).
//   • Shows current viewport as a yellow rectangle.
//   • Click anywhere in minimap → pan the main canvas so that point
//     becomes the centre. Drag = continuous pan.
//   • Mouse wheel inside minimap = zoom main canvas around that point.
//   • Auto-hides when graph has <6 nodes (not useful at small scale).
//   • Toggle via Settings: "c2c.minimap.enabled" + "c2c.minimap.size".
//
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";

const ROOT_ID = "c2c-minimap-root";
const SETTING_ENABLE = "c2c.minimap.enabled";
const SETTING_SIZE   = "c2c.minimap.size";
const SIZES = { S: [160, 100], M: [220, 140], L: [300, 190], XL: [400, 250] };

let _root = null, _canvas = null, _ctx = null;
let _dragging = false;
let _rafToken = 0;

function injectStyle() {
    if (document.getElementById("c2c-minimap-style")) return;
    const s = document.createElement("style");
    s.id = "c2c-minimap-style";
    s.textContent = `
#${ROOT_ID} {
    position: fixed; right: 14px; bottom: 88px;
    z-index: 9000;
    background: rgba(18, 20, 24, 0.86);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 8px;
    padding: 4px 4px 6px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.45);
    backdrop-filter: blur(8px);
    font: 10px ui-sans-serif, system-ui, sans-serif;
    color: #9ec1ff;
    user-select: none;
}
#${ROOT_ID} .c2c-mini-canvas { display:block; cursor: crosshair; border-radius: 4px; }
#${ROOT_ID} .c2c-mini-label  { padding: 2px 4px 0; text-align:right; opacity:.7; font-size: 9px; }`;
    document.head.appendChild(s);
}

function settingsBool(id, def) {
    try { const v = app.ui?.settings?.getSettingValue(id, def); return v !== false && v !== "false"; }
    catch { return def; }
}
function settingsStr(id, def) {
    try { return app.ui?.settings?.getSettingValue(id, def) || def; }
    catch { return def; }
}

function ensureRoot() {
    if (_root) return _root;
    injectStyle();
    _root = document.createElement("div");
    _root.id = ROOT_ID;
    const [w, h] = SIZES[settingsStr(SETTING_SIZE, "M")] || SIZES.M;
    _root.innerHTML = `<canvas class="c2c-mini-canvas" width="${w}" height="${h}"></canvas>
                       <div class="c2c-mini-label">minimap · click to pan</div>`;
    document.body.appendChild(_root);
    // Auto-shift up when a bottom panel (image feed / queue tab / splitter)
    // extends over our base 88 px gap. See `mec_dock_anchor.js`.
    try { window.__mecDock?.register?.(_root, { baseBottom: 88 }); } catch (_) {}
    _canvas = _root.querySelector("canvas");
    _ctx = _canvas.getContext("2d");
    _canvas.addEventListener("mousedown", onMouseDown);
    _canvas.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup",   () => _dragging = false);
    return _root;
}

function graphBounds() {
    const nodes = app.graph?._nodes || [];
    if (!nodes.length) return null;
    let x0 =  Infinity, y0 =  Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of nodes) {
        const [x, y] = n.pos || [0, 0];
        const [w, h] = n.size || [100, 60];
        if (x < x0) x0 = x;
        if (y < y0) y0 = y;
        if (x + w > x1) x1 = x + w;
        if (y + h > y1) y1 = y + h;
    }
    const pad = 80;
    return { x0: x0 - pad, y0: y0 - pad, x1: x1 + pad, y1: y1 + pad };
}

function viewportBounds() {
    const c = app.canvas;
    if (!c) return null;
    const ds = c.ds; if (!ds) return null;
    const dpr = window.devicePixelRatio || 1;
    const W = c.canvas.width  / dpr;
    const H = c.canvas.height / dpr;
    const scale = ds.scale || 1;
    return {
        x: -ds.offset[0],
        y: -ds.offset[1],
        w: W / scale,
        h: H / scale,
    };
}

function fitTransform(b, cw, ch) {
    const bw = b.x1 - b.x0, bh = b.y1 - b.y0;
    const k = Math.min(cw / bw, ch / bh);
    const offx = (cw - bw * k) * 0.5 - b.x0 * k;
    const offy = (ch - bh * k) * 0.5 - b.y0 * k;
    return { k, offx, offy };
}

function draw() {
    if (!_canvas || !_ctx) return;
    const enabled = settingsBool(SETTING_ENABLE, true);
    const nodes = app.graph?._nodes || [];
    if (!enabled || nodes.length < 6) { _root.style.display = "none"; return; }
    _root.style.display = "block";

    const b = graphBounds();
    if (!b) return;
    const cw = _canvas.width, ch = _canvas.height;
    const { k, offx, offy } = fitTransform(b, cw, ch);
    _canvas._t = { k, offx, offy, b };

    _ctx.fillStyle = "rgba(28,30,36,0.92)";
    _ctx.fillRect(0, 0, cw, ch);

    // nodes
    for (const n of nodes) {
        const [x, y] = n.pos || [0, 0];
        const [w, h] = n.size || [100, 60];
        const rx = x * k + offx, ry = y * k + offy;
        const rw = Math.max(1, w * k), rh = Math.max(1, h * k);
        _ctx.fillStyle = n.bgcolor || "#3a4554";
        _ctx.fillRect(rx, ry, rw, rh);
        if (n._c2c_pulse_until && performance.now() < n._c2c_pulse_until) {
            _ctx.strokeStyle = "#ffd166"; _ctx.lineWidth = 1.5;
            _ctx.strokeRect(rx - 1, ry - 1, rw + 2, rh + 2);
        }
    }
    // viewport
    const vp = viewportBounds();
    if (vp) {
        _ctx.strokeStyle = "#ffd166";
        _ctx.lineWidth = 1.4;
        _ctx.strokeRect(
            vp.x * k + offx,
            vp.y * k + offy,
            Math.max(2, vp.w * k),
            Math.max(2, vp.h * k),
        );
    }
}

function panTo(mx, my) {
    const t = _canvas._t; if (!t) return;
    const c = app.canvas; if (!c || !c.ds) return;
    const gx = (mx - t.offx) / t.k;
    const gy = (my - t.offy) / t.k;
    const dpr = window.devicePixelRatio || 1;
    const W = c.canvas.width  / dpr;
    const H = c.canvas.height / dpr;
    const scale = c.ds.scale || 1;
    c.ds.offset[0] = -gx + (W / scale) * 0.5;
    c.ds.offset[1] = -gy + (H / scale) * 0.5;
    c.setDirty(true, true);
}

function onMouseDown(ev) {
    _dragging = true;
    const r = _canvas.getBoundingClientRect();
    panTo(ev.clientX - r.left, ev.clientY - r.top);
    ev.preventDefault();
}
function onMouseMove(ev) {
    if (!_dragging) return;
    const r = _canvas.getBoundingClientRect();
    panTo(ev.clientX - r.left, ev.clientY - r.top);
}
function onWheel(ev) {
    const c = app.canvas; if (!c?.ds) return;
    ev.preventDefault();
    const delta = ev.deltaY > 0 ? 0.9 : 1.1;
    c.ds.scale = Math.max(0.05, Math.min(4, c.ds.scale * delta));
    c.setDirty(true, true);
}

function tick() {
    draw();
    _rafToken = requestAnimationFrame(tick);
}

app.registerExtension({
    name: "C2C.Minimap",
    async setup() {
        try {
            app.ui.settings.addSetting({
                id: SETTING_ENABLE,
                name: "Canvas minimap (bottom-right)",
                type: "boolean", defaultValue: true,
                category: ["c2c", "Overlays", "Minimap"],
            });
            app.ui.settings.addSetting({
                id: SETTING_SIZE,
                name: "Minimap size",
                type: "combo", options: ["S", "M", "L", "XL"], defaultValue: "M",
                category: ["c2c", "Overlays", "Minimap"],
                onChange: () => {
                    if (!_root) return;
                    const [w, h] = SIZES[settingsStr(SETTING_SIZE, "M")] || SIZES.M;
                    _canvas.width = w; _canvas.height = h;
                },
            });
        } catch { /* settings API not ready */ }
        ensureRoot();
        tick();
        console.log("[C2C.Minimap] ready.");
    },
});
