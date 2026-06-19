/**
 * mec_noodle_styles.js — custom link rendering styles for ComfyUI.
 *
 * Patches `LGraphCanvas.prototype.renderLink` with a dispatcher that
 * looks at the `mec.noodle.style` setting and either calls the original
 * renderer (for `default`) or one of the styles defined below.
 *
 * Adds a "🍝 Noodle Style →" submenu to the canvas right-click menu.
 *
 * Styles:
 *   default       — ComfyUI built-in (unmodified).
 *   spider-web    — Tobey 🕷️ sprite rides the midpoint of every link.
 *   lightsaber    — white core + colored glow, type-keyed:
 *                       MODEL→red, IMAGE→blue, MASK→green, default→purple.
 *   dna-helix     — two phase-shifted sines, link-type tinted.
 *   rainbow-flow  — animated HSL gradient flowing output→input.
 *   dashed-march  — marching ants (setLineDash + lineDashOffset tick).
 *   lightning     — perturbed polyline, redraws every 3rd frame.
 *   pulse-packet  — bezier + a moving "data packet" dot.
 *   neon-tube     — bright thin core + dim wide glow (no animation).
 *
 * The setting key `mec.noodle.style` is also exposed in the ComfyUI
 * settings panel so the user can change it without using the menu.
 */

import { app } from "../../scripts/app.js";
import { C } from './_c2c_theme.js';
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const STYLES = [
    "default","spider-web","lightsaber","dna-helix","rainbow-flow",
    "dashed-march","lightning","pulse-packet","neon-tube",
];
const SETTING_ID = "mec.noodle.style";

let _orig = null;

// CRITICAL: Canvas2D CANNOT parse CSS var() — assigning "var(--x)" to
// fillStyle/strokeStyle silently leaves it BLACK (the historic black-confetti /
// black-dot bug). So every colour handed to the canvas MUST be a resolved
// literal. _safeColor() resolves var() → the computed hex (cached), with
// Catppuccin literal fallbacks if the var is unset.
const _C2C_FALLBACK = {
    "--c2c-red": "#f38ba8", "--c2c-blue": "#89b4fa", "--c2c-green": "#a6e3a1",
    "--c2c-mauve": "#cba6f7", "--c2c-yellow": "#f9e2af", "--c2c-peach": "#fab387",
    "--c2c-teal": "#94e2d5", "--c2c-lavender": "#b4befe",
};
const _cssCache = {};
function _safeColor(c) {
    if (!c || typeof c !== "string") return c || null;
    const m = c.match(/var\(\s*(--[a-z0-9-]+)/i);
    if (!m) return c;                       // already a literal
    const name = m[1];
    if (_cssCache[name]) return _cssCache[name];
    let v = "";
    try { v = getComputedStyle(document.documentElement).getPropertyValue(name).trim(); } catch (_) {}
    if (!v) { try { v = getComputedStyle(document.body).getPropertyValue(name).trim(); } catch (_) {} }
    if (!v) v = _C2C_FALLBACK[name] || "#b4befe";
    _cssCache[name] = v;
    return v;
}

// Catppuccin-keyed default link colors per type (RESOLVED to literal hex —
// canvas-safe; returning var() here rendered black links/glows/dots).
function _typeColor(linkType) {
    const t = (linkType || "").toUpperCase();
    if (t.includes("MODEL"))      return _safeColor("var(--c2c-red)");
    if (t.includes("IMAGE"))      return _safeColor("var(--c2c-blue)");
    if (t.includes("MASK"))       return _safeColor("var(--c2c-green)");
    if (t.includes("LATENT"))     return _safeColor("var(--c2c-mauve)");
    if (t.includes("CLIP"))       return _safeColor("var(--c2c-yellow)");
    if (t.includes("VAE"))        return _safeColor("var(--c2c-peach)");
    if (t.includes("CONDITION"))  return _safeColor("var(--c2c-teal)");
    return _safeColor("var(--c2c-lavender)");
}

function _bezierPoints(a, b) {
    const dist = Math.max(20, Math.abs(a[0] - b[0]) * 0.5);
    return [[a[0]+dist, a[1]], [b[0]-dist, b[1]]];
}
function _bezierAt(t, a, cp1, cp2, b) {
    const u = 1 - t;
    const x = u*u*u*a[0] + 3*u*u*t*cp1[0] + 3*u*t*t*cp2[0] + t*t*t*b[0];
    const y = u*u*u*a[1] + 3*u*u*t*cp1[1] + 3*u*t*t*cp2[1] + t*t*t*b[1];
    return [x, y];
}

// ── Style implementations ────────────────────────────────────────────
// Clean full-body Spider-Man with web strands already in both hands (transparent
// PNG at js/assets/spiderman_cutout.png, from the user's "cutout_new" render —
// arms outstretched, the web extends horizontally to frame edges). Lazy-loaded
// once; until it loads (or if it's missing) the style degrades to webbing-only.
let _spideyImg = null;   // ImageBitmap once loaded
let _spideyTried = false;
function _spidey() {
    if (_spideyTried) return _spideyImg;
    _spideyTried = true;
    // fetch→createImageBitmap instead of new Image(): in this app shell the
    // HTMLImageElement load stalled indefinitely while fetch() of the same
    // URL returned the full JPEG — bitmap decoding sidesteps whatever
    // intercepts element loads, and drawImage accepts ImageBitmap directly.
    try {
        const url = new URL("./assets/spiderman_cutout.png", import.meta.url).href;
        fetch(url, { cache: "force-cache" })
            .then((r) => (r.ok ? r.blob() : Promise.reject(r.status)))
            .then((b) => createImageBitmap(b))
            .then((bmp) => {
                _spideyImg = bmp;
                try { app.graph?.setDirtyCanvas(true, true); } catch (_) {}
            })
            .catch(() => { _spideyImg = null; });
    } catch (_) { _spideyImg = null; }
    return _spideyImg;
}
// In the cutout the web strands run horizontally through his fists on the line
// y≈19.4% of the image — anchoring that line ON the noodle puts the in-image web
// (and his hands) right on the rope at any link angle.
const _SPIDEY_W = 150;          // drawn width in graph units (image is wide: web spans edge-to-edge)
const _SPIDEY_HANDS_Y = 0.194;  // hands/web line, fraction of image height (measured from cutout)

function _renderSpiderWeb(ctx, a, b /*, color */) {
    // Raimi train-scene WEBBING (user spec 2026-06-12, with reference still):
    // a braided white web-rope — two wavy rails around a core filament,
    // bound by X cross-ties, with splayed anchor strands at both ends.
    // (Replaces the old single line + 🕷️ emoji, which read as "a spider on
    // a wire", not Spider-Man webbing.)
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 26;
    const pts = [], nrm = [];
    for (let i = 0; i <= N; i++) {
        const t = i / N;
        const p = _bezierAt(t, a, cp1, cp2, b);
        pts.push(p);
        if (i > 0) {
            const dx = p[0] - pts[i - 1][0], dy = p[1] - pts[i - 1][1];
            const L = Math.hypot(dx, dy) || 1;
            nrm.push([-dy / L, dx / L]);
        }
    }
    nrm.push(nrm[nrm.length - 1] || [0, 1]);
    const railAt = (i, side) => {
        const wiggle = Math.sin(i * 1.25 + side * Math.PI) * 1.2;
        const off = side * 3 + wiggle;
        return [pts[i][0] + nrm[i][0] * off, pts[i][1] + nrm[i][1] * off];
    };
    ctx.save();
    ctx.lineCap = "round";
    ctx.strokeStyle = "#f4f6ff";
    // core filament
    ctx.globalAlpha = 0.95; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]);
    ctx.bezierCurveTo(cp1[0], cp1[1], cp2[0], cp2[1], b[0], b[1]); ctx.stroke();
    // two wavy rails
    ctx.lineWidth = 1.3; ctx.globalAlpha = 0.85;
    for (const side of [-1, 1]) {
        ctx.beginPath();
        for (let i = 0; i <= N; i++) {
            const r = railAt(i, side);
            i ? ctx.lineTo(r[0], r[1]) : ctx.moveTo(r[0], r[1]);
        }
        ctx.stroke();
    }
    // X cross-ties binding the rails (the braided web-rope look)
    ctx.lineWidth = 1; ctx.globalAlpha = 0.7;
    for (let i = 1; i < N - 1; i += 3) {
        const a1 = railAt(i, -1), b1 = railAt(i + 1, 1);
        const a2 = railAt(i, 1),  b2 = railAt(i + 1, -1);
        ctx.beginPath(); ctx.moveTo(a1[0], a1[1]); ctx.lineTo(b1[0], b1[1]); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(a2[0], a2[1]); ctx.lineTo(b2[0], b2[1]); ctx.stroke();
    }
    // anchor splats: 3 short splayed strands where the web grips each socket
    ctx.lineWidth = 1.2; ctx.globalAlpha = 0.8;
    for (const [end, dirIdx] of [[a, 1], [b, N - 1]]) {
        const base = nrm[dirIdx];
        for (const spread of [-0.6, 0, 0.6]) {
            const c = Math.cos(spread), s = Math.sin(spread);
            const vx = base[0] * c - base[1] * s, vy = base[0] * s + base[1] * c;
            ctx.beginPath(); ctx.moveTo(end[0], end[1]);
            ctx.lineTo(end[0] + vx * 9, end[1] + vy * 9); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(end[0], end[1]);
            ctx.lineTo(end[0] - vx * 9, end[1] - vy * 9); ctx.stroke();
        }
    }
    // THE MEME: Tobey rides the midpoint of the link, the strand running
    // through his fists. Rotated to the link tangent, flipped so he is
    // never upside-down. Skipped gracefully until the still is loaded.
    const img = _spidey();
    const iw = img ? (img.width || img.naturalWidth || 0) : 0;
    const ih = img ? (img.height || img.naturalHeight || 0) : 0;
    if (img && iw > 0) {
        const [mx, my] = _bezierAt(0.5, a, cp1, cp2, b);
        const p1 = _bezierAt(0.46, a, cp1, cp2, b);
        const p2 = _bezierAt(0.54, a, cp1, cp2, b);
        let ang = Math.atan2(p2[1] - p1[1], p2[0] - p1[0]);
        if (ang > Math.PI / 2) ang -= Math.PI;
        else if (ang < -Math.PI / 2) ang += Math.PI;
        const w = _SPIDEY_W;
        const h = w * (ih / iw);
        ctx.globalAlpha = 1;
        ctx.translate(mx, my);
        ctx.rotate(ang);
        // anchor the HANDS LINE (not the image centre) on the noodle
        ctx.drawImage(img, -w / 2, -h * _SPIDEY_HANDS_Y, w, h);
        ctx.rotate(-ang);
        ctx.translate(-mx, -my);
    }
    ctx.restore();
}
function _renderLightsaber(ctx, a, b, color, linkType) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const glow = _typeColor(linkType);
    ctx.save();
    // Outer glow
    ctx.strokeStyle = glow; ctx.lineWidth = 8; ctx.globalAlpha = 0.45;
    ctx.shadowBlur = 16; ctx.shadowColor = glow;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    // White core
    ctx.globalAlpha = 1; ctx.shadowBlur = 0;
    ctx.strokeStyle = C.white; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    ctx.restore();
}
function _renderDnaHelix(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 40;
    ctx.save();
    const phase = (performance.now()/300) % (Math.PI*2);
    for (let strand=0; strand<2; strand++) {
        ctx.strokeStyle = strand===0 ? color : _typeColor("");
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i=0;i<=N;i++) {
            const t = i/N;
            const [bx,by] = _bezierAt(t, a, cp1, cp2, b);
            // perpendicular offset
            const [bx2,by2] = _bezierAt(Math.min(1,t+0.01), a, cp1, cp2, b);
            const dx = bx2-bx, dy = by2-by, len = Math.hypot(dx,dy)||1;
            const nx = -dy/len, ny = dx/len;
            const off = Math.sin(t*Math.PI*6 + phase + strand*Math.PI) * 8;
            const px = bx + nx*off, py = by + ny*off;
            if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
        }
        ctx.stroke();
    }
    ctx.restore();
}
function _renderRainbowFlow(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const t = performance.now()/30;
    const grad = ctx.createLinearGradient(a[0],a[1],b[0],b[1]);
    for (let i=0;i<=6;i++) {
        const hue = ((i*60)+t) % 360;
        grad.addColorStop(i/6, `hsl(${hue}, 80%, 65%)`);
    }
    ctx.save();
    ctx.strokeStyle = grad; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    ctx.restore();
}
function _renderDashedMarch(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.setLineDash([8,6]);
    ctx.lineDashOffset = -((performance.now()/40) % 14);
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    ctx.setLineDash([]); ctx.restore();
}
function _renderLightning(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 10;
    // Only re-jitter every 3rd frame to avoid epilepsy + perf hit.
    const phase = Math.floor(performance.now()/100);
    const rand = (i) => {
        const s = Math.sin(i*12.9898 + phase*78.233) * 43758.5453;
        return s - Math.floor(s);
    };
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.shadowBlur = 8; ctx.shadowColor = color;
    ctx.beginPath();
    for (let i=0;i<=N;i++) {
        const t = i/N;
        const [bx,by] = _bezierAt(t, a, cp1, cp2, b);
        const jx = (rand(i)-0.5)*16, jy = (rand(i+100)-0.5)*16;
        if (i===0) ctx.moveTo(bx,by); else ctx.lineTo(bx+jx, by+jy);
    }
    ctx.stroke(); ctx.restore();
}
function _renderPulsePacket(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.globalAlpha = 0.7;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    ctx.globalAlpha = 1;
    const t = ((performance.now()/1500) % 1);
    const [px, py] = _bezierAt(t, a, cp1, cp2, b);
    ctx.fillStyle = C.white; ctx.shadowBlur = 12; ctx.shadowColor = color;
    ctx.beginPath(); ctx.arc(px,py,4,0,Math.PI*2); ctx.fill();
    ctx.restore();
}
function _renderNeonTube(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save();
    // wide dim glow
    ctx.strokeStyle = color; ctx.lineWidth = 6; ctx.globalAlpha = 0.25;
    ctx.shadowBlur = 18; ctx.shadowColor = color;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    // bright thin core
    ctx.globalAlpha = 1; ctx.shadowBlur = 0;
    ctx.strokeStyle = C.white; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(a[0],a[1]);
    ctx.bezierCurveTo(cp1[0],cp1[1], cp2[0],cp2[1], b[0],b[1]); ctx.stroke();
    ctx.restore();
}

const _RENDER = {
    "spider-web":   _renderSpiderWeb,
    "lightsaber":   _renderLightsaber,
    "dna-helix":    _renderDnaHelix,
    "rainbow-flow": _renderRainbowFlow,
    "dashed-march": _renderDashedMarch,
    "lightning":    _renderLightning,
    "pulse-packet": _renderPulsePacket,
    "neon-tube":    _renderNeonTube,
};

// PERF: renderLink runs PER LINK, PER FRAME (145 links × 60fps = ~8,700/s).
// _currentStyle() used to call getSettingValue() there — 8,700 reactive
// settings-store reads/sec, AND (because it passed the deprecated 2nd arg)
// 8,700 "defaultValue is deprecated" warnings/sec, which c2c_doctor's patched
// console.warn then processed 8,700×/sec. That pegged a core for ANY style,
// because it ran before the `default` early-return. Cache it; refresh only on
// change. getSettingValue is now called ~once, not per link per frame.
let _styleCache = "default";
function _refreshStyleCache() {
    try {
        const v = app.ui.settings.getSettingValue(SETTING_ID);   // no deprecated 2nd arg
        _styleCache = (v === undefined || v === null) ? "default" : v;
    } catch (_) { _styleCache = "default"; }
}
function _currentStyle() { return _styleCache; }

function _installRenderPatch() {
    if (_orig || !window.LGraphCanvas) return;
    _orig = LGraphCanvas.prototype.renderLink;
    LGraphCanvas.prototype.renderLink = function (ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, num_sublines) {
        const style = _currentStyle();
        if (style === "default" || !_RENDER[style]) {
            return _orig.call(this, ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, num_sublines);
        }
        try {
            const linkType = link?.type || "";
            // Resolve the incoming link colour too — ComfyUI may hand us a
            // CSS var() which would render the noodle black on canvas.
            const effColor = _safeColor(color) || _typeColor(linkType);
            _RENDER[style](ctx, a, b, effColor, linkType);
            // ── Keep "the dot" working on EVERY custom noodle ──────────────
            // The native renderLink stores link._pos (the bezier centre) and
            // ComfyUI's separate link-marker pass draws + hit-tests the dot
            // against it. Replacing renderLink dropped link._pos, so the dot
            // "disappeared" on custom styles. Restoring it brings back the
            // NATIVE dot (correct theme colour, clickable, menu) on every style
            // — including pressing Spidey, who rides the centre.
            // We draw a small affordance dot using a RESOLVED colour (never a
            // raw var() → that was the black-dot bug). Skipped for spider-web
            // (the figure IS the dot).
            if (link) {
                const [cp1m, cp2m] = _bezierPoints(a, b);
                const mid = _bezierAt(0.5, a, cp1m, cp2m, b);
                if (!link._pos) link._pos = new Float32Array(2);
                link._pos[0] = mid[0]; link._pos[1] = mid[1];
                if (style !== "spider-web") {
                    ctx.save();
                    ctx.beginPath(); ctx.arc(mid[0], mid[1], 5, 0, Math.PI * 2);
                    ctx.fillStyle = effColor || "#cdd6f4";   // resolved literal, never var()
                    ctx.globalAlpha = 0.95; ctx.fill();
                    ctx.globalAlpha = 1; ctx.lineWidth = 1.5;
                    ctx.strokeStyle = "rgba(20,20,28,0.65)"; ctx.stroke();
                    ctx.restore();
                }
            }
            // Force a continuous redraw for animated styles — THROTTLED (~20fps)
            // and PAUSED while the tab is hidden. Previously this set
            // dirty_canvas=true on every render, forcing a full-canvas repaint of
            // EVERY link at max FPS forever — the idle "FPS:44" + ~46% CPU the user
            // hit (each repaint re-runs these heavy per-link renderers ×N links).
            // 50ms throttle ≈ 20fps: smooth shimmer at a fraction of the cost, and
            // zero CPU in the background.
            if (style === "dna-helix" || style === "rainbow-flow" || style === "dashed-march" ||
                style === "lightning" || style === "pulse-packet") {
                const _now = performance.now();
                if (!document.hidden && _now - (this._c2cNoodleAnimTick || 0) > 50) {
                    this._c2cNoodleAnimTick = _now;
                    this.dirty_canvas = true;
                }
            }
        } catch (e) {
            console.warn("[MEC.NoodleStyles] render error, falling back:", e);
            return _orig.call(this, ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, num_sublines);
        }
    };
}

function _installMenuPatch() {
    if (!window.LGraphCanvas) return;
    const origMenu = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        const out = origMenu ? origMenu.call(this) : [];
        const cur = _currentStyle();
        const submenu = STYLES.map(s => ({
            content: `${s === cur ? "✓ " : "  "}${s}`,
            callback: () => {
                _styleCache = s;   // update cache immediately (don't rely on onChange firing)
                try { app.ui.settings.setSettingValue(SETTING_ID, s); } catch (__c2cErr) { __c2cReport("c2c_noodle_styles", __c2cErr); }
                this.setDirty(true, true);
            },
        }));
        out.push(null); // separator
        out.push({
            content: "🍝 Noodle Style",
            has_submenu: true,
            submenu: { options: submenu },
        });
        out.push({
            content: "— also configurable in Settings → mec.noodle.style",
            disabled: true,
        });
        return out;
    };
}

app.registerExtension({
    name: "C2C.NoodleStyles",
    settings: [
        {
            id: SETTING_ID,
            name: "Noodle style",
            tooltip:
                "Custom rendering for node links. `default` keeps ComfyUI's built-in. " +
                "Also reachable from the canvas right-click menu (🍝 Noodle Style) and " +
                "from the reroute-dot context menu.",
            type: "combo",
            options: STYLES,
            defaultValue: "default",
            onChange: (v) => { _styleCache = (v === undefined || v === null) ? "default" : v; },
        },
    ],
    async setup() {
        _refreshStyleCache();   // seed the per-frame style cache once at startup
        // LiteGraph is loaded before extensions, but be defensive.
        const tryInstall = () => {
            if (window.LGraphCanvas) {
                _installRenderPatch();
                _installMenuPatch();
                console.log("[MEC.NoodleStyles] Installed:", STYLES.filter(s=>s!=="default").join(", "));
            } else {
                setTimeout(tryInstall, 100);
            }
        };
        tryInstall();
    },
});

// Expose to other modules (e.g. universal_reroute.js menu mirror).
export const NOODLE_STYLES = STYLES;
export const NOODLE_SETTING_ID = SETTING_ID;
