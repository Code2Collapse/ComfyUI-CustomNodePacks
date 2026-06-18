/**
 * c2c_completion_fx.js — Completion FX: themed celebration animations.
 *
 * On `execution_success`, plays the user-selected celebration style on a
 * full-viewport overlay canvas. Pure Canvas2D, zero dependencies. The full
 * 14-style catalogue from ideas_report.md §2 (user spec 2026-06-12:
 * "star wars saber hitting and flashing", "tron legacy animation", + more):
 *
 *   confetti · fireworks · starwars · tron · spiderman · matrix · portal ·
 *   mario · saber-clash · stranger · dragonball · sparkles · balloons · random
 *
 * Settings (ids keep the legacy `mec.` prefix — renaming ids loses stored
 * user values; labels say C2C):
 *   mec.completion_fx.confetti    — bool master (legacy id, default true)
 *   mec.completion_fx.style      — combo, default "confetti"
 *   mec.completion_fx.intensity  — low | normal | high
 *   mec.completion_fx.chime      — bool (default false)
 *   mec.completion_fx.chime_style— arpeggio | mario | tron | saber
 *   mec.completion_fx.min_runtime_ms — skip celebrating runs shorter than N
 *
 * Exposes `window.__C2C_FX = { play(style?), styles }` so c2c_celebrations.js
 * delegates here instead of double-firing its own confetti, and so a manual
 * trigger (command palette / OmniPill) can fire any style on demand.
 *
 * NOTE: every canvas color below is a LITERAL hex on purpose — Canvas2D
 * cannot parse CSS var() (the historic "all-black confetti" bug class).
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const CANVAS_ID = "mec-confetti-canvas";
const STYLES = ["confetti", "fireworks", "starwars", "tron", "spiderman",
    "spiderman-swing", "matrix", "portal", "mario", "saber-clash", "saber-duel",
    "stranger", "dragonball", "trophy", "level-up", "sparkles", "balloons", "random"];

// Catppuccin-mocha literals (canvas-safe).
const PAL = { red: "#f38ba8", peach: "#fab387", yellow: "#f9e2af", green: "#a6e3a1",
    blue: "#89b4fa", mauve: "#cba6f7", pink: "#f5c2e7", sky: "#89dceb",
    teal: "#94e2d5", text: "#cdd6f4", gold: "#ffd866" };
const CONFETTI_COLORS = [PAL.red, PAL.peach, PAL.yellow, PAL.green, PAL.blue, PAL.mauve, PAL.pink];

// The transparent Spider-Man cutout (shared with the noodle style). Lazy-loaded
// once via fetch→createImageBitmap (plain new Image() stalled in this app shell);
// until it resolves, spiderman-swing falls back to the 🕷 glyph.
let _spideyFxImg = null, _spideyFxTried = false;
function _spideyFx() {
    if (_spideyFxTried) return _spideyFxImg;
    _spideyFxTried = true;
    try {
        const url = new URL("./assets/spiderman_cutout.png", import.meta.url).href;
        fetch(url, { cache: "force-cache" })
            .then((r) => (r.ok ? r.blob() : Promise.reject(r.status)))
            .then((b) => createImageBitmap(b))
            .then((bmp) => { _spideyFxImg = bmp; })
            .catch(() => { _spideyFxImg = null; });
    } catch (_) { _spideyFxImg = null; }
    return _spideyFxImg;
}

let _running = false;
let _lastRunStart = 0;

// Settings live under c2c.* (so they file under the C2C section). Legacy
// installs stored values under mec.completion_fx.* — read those as fallback
// and migrate once at setup so nothing the user configured is lost.
function _setting(id, dflt) {
    try {
        const v = app.ui.settings.getSettingValue("c2c.completion_fx." + id);
        if (v !== undefined && v !== null && v !== "") return v;
        const legacy = app.ui.settings.getSettingValue("mec.completion_fx." + id);
        if (legacy !== undefined && legacy !== null && legacy !== "") return legacy;
        return dflt;
    } catch (_) { return dflt; }
}
function _intensity() {
    const v = _setting("intensity", "normal");
    return v === "low" ? 0.5 : v === "high" ? 1.6 : 1.0;
}
const _rand = (a, b) => a + Math.random() * (b - a);
const _pick = (arr) => arr[(Math.random() * arr.length) | 0];

function _ensureCanvas() {
    let c = document.getElementById(CANVAS_ID);
    if (c) return c;
    c = document.createElement("canvas");
    c.id = CANVAS_ID;
    // cssText mirrors c2c_celebrations.js — the PROVEN-visible overlay recipe.
    // (Object.assign(style,{inset,width:"100%"…}) produced a canvas that
    // painted internally but never composited on screen in this app shell.)
    c.style.cssText = [
        "position:fixed", "inset:0", "width:100vw", "height:100vh",
        "pointer-events:none", "display:none",
        "z-index:var(--c2c-z-toast, 100002)",
    ].join(";");
    document.body.appendChild(c);
    return c;
}

function _begin() {
    if (_running || document.hidden) return null;
    const c = _ensureCanvas();
    c.width = window.innerWidth;
    c.height = window.innerHeight;
    c.style.display = "block";
    _running = true;
    return c;
}
function _end(c) {
    try {
        const ctx = c.getContext("2d");
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = "source-over";
        ctx.clearRect(0, 0, c.width, c.height);
        c.style.display = "none";
    } catch (_) {}
    _running = false;
}

/**
 * Shared runner: every style supplies { dur, init(s,W,H,I), frame(s,ctx,W,H,p,now) }.
 * The runner owns begin/end, the rAF loop, and the frame-budget guard
 * (>32ms on 6 frames → finish as a short sparkle pass instead of janking).
 */
function _run(style) {
    const def = _MAP[style];
    if (!def) return;
    const c = _begin();
    if (!c) return;
    const ctx = c.getContext("2d");
    const W = c.width, H = c.height, I = _intensity();
    const s = {};
    try { def.init?.(s, W, H, I); } catch (e) { __c2cReport("completion_fx." + style, e); _end(c); return; }
    const t0 = performance.now();
    let last = t0, slow = 0, degraded = false;
    const tick = (now) => {
        const p = Math.min(1, (now - t0) / def.dur);
        if (now - last > 32) slow++;
        last = now;
        if (slow > 6 && !degraded && style !== "sparkles") {
            degraded = true;            // frame budget blown — fade out early
        }
        try {
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.globalAlpha = 1;
            ctx.globalCompositeOperation = "source-over";
            ctx.clearRect(0, 0, W, H);
            if (degraded) ctx.globalAlpha = Math.max(0, 1 - (p * 4 - 3));
            def.frame(s, ctx, W, H, p, now);
        } catch (e) { __c2cReport("completion_fx." + style, e); _end(c); return; }
        if (p < 1 && !(degraded && p > 0.9)) requestAnimationFrame(tick);
        else _end(c);
    };
    requestAnimationFrame(tick);
}

/* ════════════════════ style implementations ════════════════════ */
const _MAP = {};

// 1 · confetti — classic colored squares rain.
_MAP["confetti"] = {
    dur: 2400,
    init(s, W, H, I) {
        s.ps = Array.from({ length: Math.round(90 * I) }, () => ({
            x: Math.random() * W, y: -20 - Math.random() * 220,
            vx: _rand(-1.5, 1.5), vy: _rand(2, 6),
            rot: _rand(0, Math.PI * 2), vr: _rand(-0.2, 0.2),
            size: _rand(6, 12), color: _pick(CONFETTI_COLORS),
        }));
    },
    frame(s, ctx, W, H) {
        for (const p of s.ps) {
            p.vy += 0.08; p.x += p.vx; p.y += p.vy; p.rot += p.vr;
            ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot);
            ctx.fillStyle = p.color;
            ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
            ctx.restore();
        }
    },
};

// 2 · fireworks — five staggered radial bursts with fading trails.
_MAP["fireworks"] = {
    dur: 2800,
    init(s, W, H, I) {
        s.bursts = Array.from({ length: 5 }, (_, i) => ({
            at: i * 0.16, x: _rand(W * 0.15, W * 0.85), y: _rand(H * 0.15, H * 0.5),
            color: _pick(CONFETTI_COLORS), ps: null,
        }));
        s.n = Math.round(60 * I);
    },
    frame(s, ctx, W, H, p) {
        for (const b of s.bursts) {
            if (p < b.at) continue;
            if (!b.ps) b.ps = Array.from({ length: s.n }, (_, i) => {
                const a = (i / s.n) * Math.PI * 2 + _rand(-0.05, 0.05);
                const v = _rand(2.2, 5.5);
                return { x: b.x, y: b.y, px: b.x, py: b.y, vx: Math.cos(a) * v, vy: Math.sin(a) * v };
            });
            const k = Math.min(1, (p - b.at) / 0.45);
            ctx.globalAlpha = Math.max(0, 1 - k);
            ctx.strokeStyle = b.color; ctx.lineWidth = 2;
            for (const q of b.ps) {
                q.px = q.x; q.py = q.y;
                q.vy += 0.05; q.x += q.vx; q.y += q.vy;
                ctx.beginPath(); ctx.moveTo(q.px, q.py); ctx.lineTo(q.x, q.y); ctx.stroke();
            }
            ctx.globalAlpha = 1;
        }
    },
};

// 3 · starwars — perspective starfield + scrolling crawl.
_MAP["starwars"] = {
    dur: 3200,
    init(s, W, H, I) {
        s.stars = Array.from({ length: Math.round(160 * I) }, () => ({
            x: _rand(-1, 1), y: _rand(-1, 1), z: _rand(0.05, 1),
        }));
        s.lines = ["EXECUTION COMPLETE", "Episode ∞", "THE WORKFLOW AWAKENS"];
    },
    frame(s, ctx, W, H, p) {
        ctx.fillStyle = "rgba(0,0,0,0.55)"; ctx.fillRect(0, 0, W, H);
        const cx = W / 2, cy = H / 2;
        ctx.fillStyle = "#ffffff";
        for (const st of s.stars) {
            st.z -= 0.012; if (st.z <= 0.02) st.z = 1;
            const sx = cx + (st.x / st.z) * cx, sy = cy + (st.y / st.z) * cy;
            const r = Math.max(0.4, 1.6 * (1 - st.z));
            ctx.globalAlpha = Math.min(1, 1.2 - st.z);
            ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1;
        // crawl: rises from bottom, shrinking into the distance
        ctx.save();
        ctx.textAlign = "center"; ctx.fillStyle = "#ffe81f";
        s.lines.forEach((ln, i) => {
            const y = H * (1.15 - p * 1.1) + i * 64;
            const k = Math.max(0.25, 1 - (H - y) / H);   // smaller as it climbs
            ctx.font = `bold ${Math.round(40 * k)}px sans-serif`;
            ctx.globalAlpha = Math.max(0, Math.min(1, k * 1.4 - 0.2));
            ctx.fillText(ln, cx, y);
        });
        ctx.restore();
    },
};

// 4 · tron — glowing grid floor + two light cycles + verdict text.
_MAP["tron"] = {
    dur: 3000,
    init(s, W, H) {
        const mk = (color, y, dir) => ({ color, trail: [{ x: dir > 0 ? -40 : W + 40, y }], dir, y, seg: 0 });
        s.cycles = [mk("#6be5ff", H * 0.42, 1), mk("#ff9a3c", H * 0.58, -1)];
    },
    frame(s, ctx, W, H, p) {
        ctx.fillStyle = "rgba(2,8,18,0.78)"; ctx.fillRect(0, 0, W, H);
        // perspective grid
        ctx.strokeStyle = "rgba(107,229,255,0.35)"; ctx.lineWidth = 1;
        const horizon = H * 0.30;
        for (let i = 0; i <= 14; i++) {
            const x = (i / 14) * W;
            ctx.beginPath(); ctx.moveTo(W / 2 + (x - W / 2) * 0.25, horizon); ctx.lineTo(x, H); ctx.stroke();
        }
        for (let i = 0; i <= 8; i++) {
            const t = i / 8, y = horizon + (H - horizon) * t * t;
            ctx.globalAlpha = 0.15 + 0.3 * t;
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        }
        ctx.globalAlpha = 1;
        // light cycles with hard-turn trails
        for (const cy of s.cycles) {
            const head = cy.trail[cy.trail.length - 1];
            const speed = (W / 130) * (cy.dir);
            let nx = head.x + speed, ny = head.y;
            if (Math.random() < 0.035 && cy.trail.length < 24) {            // 90° jink
                ny = head.y + _rand(-1, 1) > 0 ? head.y + _rand(30, 70) : head.y - _rand(30, 70);
                cy.trail.push({ x: head.x, y: ny });
            }
            cy.trail.push({ x: nx, y: ny });
            if (cy.trail.length > 26) cy.trail.shift();
            ctx.save();
            ctx.shadowColor = cy.color; ctx.shadowBlur = 14;
            ctx.strokeStyle = cy.color; ctx.lineWidth = 3; ctx.lineJoin = "miter";
            ctx.beginPath();
            cy.trail.forEach((q, i) => i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y));
            ctx.stroke();
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(nx - 4, ny - 4, 8, 8);
            ctx.restore();
        }
        if (p > 0.25) {
            ctx.save();
            ctx.textAlign = "center"; ctx.font = "bold 34px monospace";
            ctx.shadowColor = "#6be5ff"; ctx.shadowBlur = 18;
            ctx.fillStyle = "#dff7ff";
            ctx.globalAlpha = Math.min(1, (p - 0.25) * 3) * (p > 0.85 ? (1 - p) / 0.15 : 1);
            ctx.fillText("EXECUTION COMPLETE", W / 2, H * 0.2);
            ctx.restore();
        }
    },
};

// 5 · spiderman — radiating web strands + 🕷 swinging through, closing quote.
_MAP["spiderman"] = {
    dur: 3000,
    init(s, W, H, I) {
        s.webs = Array.from({ length: Math.round(7 * I) }, () => {
            const corner = _pick([[0, 0], [W, 0], [0, H], [W, H]]);
            return { x: corner[0], y: corner[1], a: Math.atan2(H / 2 - corner[1], W / 2 - corner[0]) + _rand(-0.5, 0.5), len: 0 };
        });
    },
    frame(s, ctx, W, H, p) {
        ctx.strokeStyle = "rgba(240,240,250,0.85)"; ctx.lineWidth = 1.6;
        for (const w of s.webs) {
            w.len = Math.min(Math.hypot(W, H) * 0.5, w.len + 26);
            const ex = w.x + Math.cos(w.a) * w.len, ey = w.y + Math.sin(w.a) * w.len;
            ctx.beginPath(); ctx.moveTo(w.x, w.y); ctx.lineTo(ex, ey); ctx.stroke();
            for (let i = 1; i <= 4; i++) {                       // cross-spokes
                const t = i / 5, jx = w.x + Math.cos(w.a) * w.len * t, jy = w.y + Math.sin(w.a) * w.len * t;
                ctx.beginPath(); ctx.arc(jx, jy, 7 * i * 0.6, w.a - 0.7, w.a + 0.7); ctx.stroke();
            }
        }
        // spider swings across on a rope (pendulum on a moving anchor)
        const ax = W * (0.15 + p * 0.7), ay = -10;
        const swing = Math.sin(p * Math.PI * 3) * 0.9;
        const rl = H * 0.34;
        const sx = ax + Math.sin(swing) * rl, sy = ay + Math.cos(swing) * rl;
        ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(sx, sy); ctx.stroke();
        ctx.font = "30px serif"; ctx.textAlign = "center";
        ctx.fillText("\u{1F577}", sx, sy + 10);
        if (p > 0.6) {
            ctx.font = "italic 22px Georgia, serif";
            ctx.fillStyle = PAL.red;
            ctx.globalAlpha = Math.min(1, (p - 0.6) * 4);
            ctx.fillText("With great power comes great responsibility.", W / 2, H * 0.85);
            ctx.globalAlpha = 1;
        }
    },
};

// 6 · matrix — green code-rain with bright leads and fading tails (overlay-safe).
_MAP["matrix"] = {
    dur: 3000,
    init(s, W, H, I) {
        s.fs = 18;
        s.cols = Math.floor(W / s.fs);
        s.drops = Array.from({ length: s.cols }, () => ({ y: Math.random() * -H, hist: [] }));
        s.glyphs = "ｱｲｳｴｵｶｷｸｹｺ0123456789ABCDEF<>".split("");
        s.skip = I < 1 ? 2 : 1;                       // low intensity: every 2nd column
    },
    frame(s, ctx, W, H, p) {
        ctx.fillStyle = "rgba(0,10,2,0.62)"; ctx.fillRect(0, 0, W, H);
        ctx.font = `${s.fs}px monospace`;
        for (let i = 0; i < s.cols; i += s.skip) {
            const d = s.drops[i];
            d.hist.unshift({ y: d.y, g: _pick(s.glyphs) });
            if (d.hist.length > 12) d.hist.pop();
            d.hist.forEach((h, k) => {
                ctx.fillStyle = k === 0 ? "#eafff0" : `rgba(0,255,102,${Math.max(0, 0.9 - k * 0.08)})`;
                ctx.fillText(h.g, i * s.fs, h.y);
            });
            d.y += s.fs;
            if (d.y > H + 60 && Math.random() > 0.96) { d.y = -20; d.hist.length = 0; }
        }
        if (p > 0.35) {
            ctx.textAlign = "center"; ctx.font = "bold 26px monospace";
            ctx.fillStyle = "#00ff66";
            ctx.globalAlpha = Math.min(1, (p - 0.35) * 3) * (p > 0.85 ? (1 - p) / 0.15 : 1);
            ctx.fillText("EXECUTION COMPLETE :: SYSTEM ONLINE", W / 2, H / 2);
            ctx.globalAlpha = 1; ctx.textAlign = "left";
        }
    },
};

// 7 · portal — orange↘blue rings, particle stream between, whoomp close.
_MAP["portal"] = {
    dur: 2600,
    init(s, W, H, I) {
        s.a = { x: W * 0.12, y: H * 0.78, color: "#ff9a3c" };
        s.b = { x: W * 0.88, y: H * 0.2, color: "#37c3f2" };
        s.ps = Array.from({ length: Math.round(42 * I) }, () => ({ t: Math.random(), v: _rand(0.004, 0.012), r: _rand(2, 4.5) }));
    },
    frame(s, ctx, W, H, p) {
        const easeOutBack = (t) => 1 + 2.7 * Math.pow(t - 1, 3) + 1.7 * Math.pow(t - 1, 2);
        const open = p < 0.75 ? easeOutBack(Math.min(1, p * 4)) : Math.max(0, 1 - (p - 0.75) * 4);
        const R = 56 * open;
        for (const g of [s.a, s.b]) {
            ctx.save();
            ctx.shadowColor = g.color; ctx.shadowBlur = 26;
            ctx.strokeStyle = g.color; ctx.lineWidth = 7;
            ctx.beginPath(); ctx.ellipse(g.x, g.y, R * 0.62, R, -0.5, 0, Math.PI * 2); ctx.stroke();
            ctx.restore();
        }
        if (open > 0.1) {
            const mx = (s.a.x + s.b.x) / 2 + 80, my = (s.a.y + s.b.y) / 2 - 120;   // curved path
            for (const q of s.ps) {
                q.t += q.v; if (q.t > 1) q.t = 0;
                const t = q.t, it = 1 - t;
                const x = it * it * s.a.x + 2 * it * t * mx + t * t * s.b.x;
                const y = it * it * s.a.y + 2 * it * t * my + t * t * s.b.y;
                ctx.fillStyle = t < 0.5 ? s.a.color : s.b.color;
                ctx.globalAlpha = 0.85 * open;
                ctx.beginPath(); ctx.arc(x, y, q.r * (1 - Math.abs(t - 0.5)), 0, Math.PI * 2); ctx.fill();
            }
            ctx.globalAlpha = 1;
        }
    },
};

// 8 · mario — coins burst from the bottom + "1-UP!".
_MAP["mario"] = {
    dur: 2600,
    init(s, W, H, I) {
        s.coins = Array.from({ length: Math.round(18 * I) }, (_, i) => ({
            x: _rand(W * 0.1, W * 0.9), y: H + 20, vy: _rand(-13, -8), vx: _rand(-1.2, 1.2),
            at: (i / (18 * I)) * 0.4, spin: _rand(0, Math.PI),
        }));
    },
    frame(s, ctx, W, H, p) {
        for (const c of s.coins) {
            if (p < c.at) continue;
            c.vy += 0.22; c.x += c.vx; c.y += c.vy; c.spin += 0.18;
            const squish = Math.abs(Math.cos(c.spin));            // coin spin = x-scale
            ctx.save(); ctx.translate(c.x, c.y); ctx.scale(Math.max(0.15, squish), 1);
            ctx.fillStyle = PAL.gold; ctx.beginPath(); ctx.arc(0, 0, 11, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = "#b8860b"; ctx.lineWidth = 2.5; ctx.stroke();
            ctx.fillStyle = "#b8860b"; ctx.fillRect(-2, -6, 4, 12);
            ctx.restore();
        }
        if (p > 0.3 && p < 0.85) {
            ctx.font = "bold 30px monospace"; ctx.textAlign = "center";
            ctx.fillStyle = "#3ddc54";
            const rise = (p - 0.3) * 120;
            ctx.fillText("1-UP!", W / 2, H * 0.4 - rise);
        }
    },
};

// 9 · saber-clash — two blades ignite, meet center, spark flash, retract.
_MAP["saber-clash"] = {
    dur: 2800,
    init(s, W, H, I) {
        const cx = W / 2, cy = H / 2;
        s.A = { x0: -60, y0: -60, color: "#66ccff" };                 // from top-left
        s.B = { x0: W + 60, y0: H + 60, color: "#ff5566" };           // from bottom-right
        s.cx = cx; s.cy = cy;
        s.sparks = Array.from({ length: Math.round(46 * I) }, () => {
            const a = _rand(0, Math.PI * 2), v = _rand(3, 9);
            return { x: cx, y: cy, vx: Math.cos(a) * v, vy: Math.sin(a) * v, hue: _pick(["#ffffff", "#ffe9a8", "#ffd866"]) };
        });
    },
    frame(s, ctx, W, H, p) {
        // blade extension: 0→1 over first 40%, hold, retract last 20%
        const ext = p < 0.4 ? p / 0.4 : p > 0.8 ? Math.max(0, 1 - (p - 0.8) / 0.2) : 1;
        const drawBlade = (b) => {
            const ex = b.x0 + (s.cx - b.x0) * ext, ey = b.y0 + (s.cy - b.y0) * ext;
            ctx.save();
            ctx.lineCap = "round";
            ctx.shadowColor = b.color; ctx.shadowBlur = 22;
            ctx.strokeStyle = b.color; ctx.lineWidth = 11; ctx.globalAlpha = 0.55;
            ctx.beginPath(); ctx.moveTo(b.x0, b.y0); ctx.lineTo(ex, ey); ctx.stroke();
            ctx.globalAlpha = 1; ctx.shadowBlur = 8;
            ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 4.5;
            ctx.beginPath(); ctx.moveTo(b.x0, b.y0); ctx.lineTo(ex, ey); ctx.stroke();
            ctx.restore();
        };
        drawBlade(s.A); drawBlade(s.B);
        // clash window: flash + sparks
        if (p >= 0.4 && p < 0.8) {
            const k = (p - 0.4) / 0.4;
            const flash = Math.max(0, Math.sin(Math.min(1, k * 2.2) * Math.PI));
            const g = ctx.createRadialGradient(s.cx, s.cy, 0, s.cx, s.cy, 190);
            g.addColorStop(0, `rgba(255,255,255,${0.85 * flash})`);
            g.addColorStop(1, "rgba(255,255,255,0)");
            ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
            ctx.lineWidth = 2;
            for (const q of s.sparks) {
                const px = q.x, py = q.y;
                q.x += q.vx; q.y += q.vy; q.vy += 0.12;
                ctx.strokeStyle = q.hue; ctx.globalAlpha = Math.max(0, 1 - k * 1.2);
                ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(q.x, q.y); ctx.stroke();
            }
            ctx.globalAlpha = 1;
        }
    },
};

// 5b · spiderman-swing — Spidey (the cutout) swings L↔R on a web from the top
//      middle, holding a COMPLETED board. (user idea 2026-06-15)
_MAP["spiderman-swing"] = {
    dur: 3600,
    init(s, W, H) { s.anchor = [W / 2, -8]; s.ropeLen = H * 0.30; },
    frame(s, ctx, W, H, p) {
        const [ax, ay] = s.anchor;
        const ang = Math.sin(p * Math.PI * 2.4) * 0.7;          // pendulum swing
        const bx = ax + Math.sin(ang) * s.ropeLen;
        const by = ay + Math.cos(ang) * s.ropeLen;
        // web rope from anchor to one hand
        ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        // a couple of background web strands from the top corners for flavour
        ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(bx, by); ctx.moveTo(W, 0); ctx.lineTo(bx, by); ctx.stroke();
        // Spider-Man at the bob, leaning into the swing
        const img = _spideyFx();
        ctx.save();
        ctx.translate(bx, by);
        ctx.rotate(ang * 0.5);
        if (img) {
            const iw = img.width || img.naturalWidth || 1, ih = img.height || img.naturalHeight || 1;
            const w = 118, h = w * (ih / iw);
            ctx.drawImage(img, -w / 2, -h * 0.26, w, h);        // hands line near top
        } else {
            ctx.font = "44px serif"; ctx.textAlign = "center"; ctx.fillText("\u{1F577}", 0, 14);
        }
        ctx.restore();
        // COMPLETED board held below, swaying with the swing
        ctx.save();
        ctx.translate(bx, by + 78);
        ctx.rotate(ang * 0.3);
        const bw = 196, bh = 46;
        ctx.fillStyle = "#f9e2af"; ctx.strokeStyle = "#7a5d12"; ctx.lineWidth = 3;
        ctx.fillRect(-bw / 2, -bh / 2, bw, bh); ctx.strokeRect(-bw / 2, -bh / 2, bw, bh);
        ctx.fillStyle = "#1a1a2e"; ctx.font = "bold 26px Georgia, serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("COMPLETED", 0, 1);
        ctx.restore();
        ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    },
};

// 9b · saber-duel — two blades swing in from both sides, clash centre, and
//      COMPLETED bursts out of the flash. (user idea 2026-06-15)
_MAP["saber-duel"] = {
    dur: 3000,
    init(s, W, H, I) {
        s.cx = W / 2; s.cy = H * 0.46;
        s.lHilt = [W * 0.16, s.cy + 130]; s.rHilt = [W * 0.84, s.cy + 130];
        s.sparks = Array.from({ length: Math.round(54 * I) }, () => {
            const a = _rand(0, Math.PI * 2), v = _rand(4, 12);
            return { x: s.cx, y: s.cy, vx: Math.cos(a) * v, vy: Math.sin(a) * v, c: _pick(["#ffffff", "#ffe9a8", "#89b4fa", "#f38ba8"]) };
        });
    },
    frame(s, ctx, W, H, p) {
        const reach = p < 0.45 ? p / 0.45 : 1;                  // blades meet by 45%
        const blade = (hilt, color) => {
            const tx = hilt[0] + (s.cx - hilt[0]) * reach, ty = hilt[1] + (s.cy - hilt[1]) * reach;
            ctx.save(); ctx.lineCap = "round";
            ctx.shadowColor = color; ctx.shadowBlur = 24;
            ctx.strokeStyle = color; ctx.lineWidth = 12; ctx.globalAlpha = 0.5;
            ctx.beginPath(); ctx.moveTo(hilt[0], hilt[1]); ctx.lineTo(tx, ty); ctx.stroke();
            ctx.globalAlpha = 1; ctx.shadowBlur = 10;
            ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 5;
            ctx.beginPath(); ctx.moveTo(hilt[0], hilt[1]); ctx.lineTo(tx, ty); ctx.stroke();
            ctx.restore();
        };
        blade(s.lHilt, "#89b4fa"); blade(s.rHilt, "#f38ba8");
        if (p >= 0.42) {
            const k = (p - 0.42) / 0.58;
            const flash = Math.max(0, Math.sin(Math.min(1, k * 2) * Math.PI));
            const g = ctx.createRadialGradient(s.cx, s.cy, 0, s.cx, s.cy, 250);
            g.addColorStop(0, `rgba(255,255,255,${0.9 * flash})`);
            g.addColorStop(1, "rgba(255,255,255,0)");
            ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
            ctx.lineWidth = 2;
            for (const q of s.sparks) {
                const px = q.x, py = q.y; q.x += q.vx; q.y += q.vy; q.vy += 0.14;
                ctx.strokeStyle = q.c; ctx.globalAlpha = Math.max(0, 1 - k * 1.1);
                ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(q.x, q.y); ctx.stroke();
            }
            ctx.globalAlpha = Math.min(1, k * 2) * (p > 0.9 ? (1 - p) / 0.1 : 1);
            ctx.textAlign = "center"; ctx.fillStyle = "#ffd866";
            ctx.font = "bold 42px Georgia, serif";
            ctx.fillText("COMPLETED", s.cx, s.cy - 26 - k * 22);
            ctx.globalAlpha = 1; ctx.textAlign = "left";
        }
    },
};

// 13b · trophy — a golden cup rises from the bottom amid twinkles + COMPLETED!
_MAP["trophy"] = {
    dur: 2800,
    init(s, W, H, I) {
        s.tw = Array.from({ length: Math.round(46 * I) }, () => ({ x: _rand(0, W), y: _rand(0, H), r: _rand(1, 3), ph: _rand(0, 6.28) }));
    },
    frame(s, ctx, W, H, p) {
        const cx = W / 2;
        const y = p < 0.5 ? (H + 90) - (p / 0.5) * (H + 90 - H * 0.52) : H * 0.52;
        for (const t of s.tw) {
            const a = (Math.sin(p * 9 + t.ph) + 1) / 2;
            ctx.fillStyle = `rgba(255,230,150,${a})`;
            ctx.beginPath(); ctx.arc(t.x, t.y, t.r, 0, 6.28); ctx.fill();
        }
        ctx.save(); ctx.translate(cx, y);
        ctx.fillStyle = "#ffd866"; ctx.strokeStyle = "#b8860b"; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(-42, -52); ctx.quadraticCurveTo(0, 44, 42, -52); ctx.closePath(); ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.arc(-46, -42, 16, Math.PI * 0.5, Math.PI * 1.5); ctx.stroke();
        ctx.beginPath(); ctx.arc(46, -42, 16, Math.PI * 1.5, Math.PI * 0.5, true); ctx.stroke();
        ctx.fillRect(-6, 8, 12, 26); ctx.fillRect(-32, 34, 64, 12);
        ctx.restore();
        if (p > 0.45) {
            ctx.textAlign = "center"; ctx.fillStyle = "#f9e2af";
            ctx.font = "bold 40px Georgia, serif";
            ctx.globalAlpha = Math.min(1, (p - 0.45) * 3) * (p > 0.9 ? (1 - p) / 0.1 : 1);
            ctx.fillText("COMPLETED!", cx, H * 0.52 - 96);
            ctx.globalAlpha = 1; ctx.textAlign = "left";
        }
    },
};

// 14b · level-up — RPG light pillars + rising motes + "LEVEL UP / COMPLETE".
_MAP["level-up"] = {
    dur: 2600,
    init(s, W, H, I) {
        s.cols = Array.from({ length: Math.round(9 * I) }, () => ({ x: _rand(W * 0.18, W * 0.82), w: _rand(14, 34), ph: _rand(0, 6.28) }));
        s.motes = Array.from({ length: Math.round(52 * I) }, () => ({ x: _rand(0, W), y: _rand(0, H), v: _rand(1.2, 4) }));
    },
    frame(s, ctx, W, H, p) {
        for (const m of s.motes) {
            m.y -= m.v; if (m.y < -6) { m.y = H + 6; m.x = _rand(0, W); }
            ctx.fillStyle = "rgba(137,180,250,0.7)"; ctx.fillRect(m.x, m.y, 2, 6);
        }
        for (const c of s.cols) {
            const h = H * (0.3 + 0.5 * ((Math.sin(p * 6 + c.ph) + 1) / 2));
            const g = ctx.createLinearGradient(0, H, 0, H - h);
            g.addColorStop(0, "rgba(203,166,247,0)");
            g.addColorStop(1, "rgba(203,166,247,0.5)");
            ctx.fillStyle = g; ctx.fillRect(c.x - c.w / 2, H - h, c.w, h);
        }
        if (p > 0.3) {
            ctx.textAlign = "center";
            ctx.globalAlpha = Math.min(1, (p - 0.3) * 3) * (p > 0.88 ? (1 - p) / 0.12 : 1);
            ctx.fillStyle = "#cba6f7"; ctx.font = "bold 46px 'Trebuchet MS', sans-serif";
            ctx.fillText("LEVEL UP", W / 2, H * 0.42);
            ctx.fillStyle = "#a6e3a1"; ctx.font = "bold 26px 'Trebuchet MS', sans-serif";
            ctx.fillText("WORKFLOW COMPLETE", W / 2, H * 0.42 + 40);
            ctx.globalAlpha = 1; ctx.textAlign = "left";
        }
    },
};

// 10 · stranger — red veins crawl in from the edges + CRT flicker + subtitle.
_MAP["stranger"] = {
    dur: 2800,
    init(s, W, H, I) {
        const mkVein = () => {
            const side = (Math.random() * 4) | 0;
            const x = side === 0 ? 0 : side === 1 ? W : _rand(0, W);
            const y = side === 2 ? 0 : side === 3 ? H : _rand(0, H);
            const toward = Math.atan2(H / 2 - y, W / 2 - x);
            const pts = [{ x, y }];
            let a = toward, cx = x, cy = y;
            for (let i = 0; i < 14; i++) {
                a += _rand(-0.55, 0.55);
                cx += Math.cos(a) * _rand(14, 30); cy += Math.sin(a) * _rand(14, 30);
                pts.push({ x: cx, y: cy });
            }
            return pts;
        };
        s.veins = Array.from({ length: Math.round(10 * I) }, mkVein);
    },
    frame(s, ctx, W, H, p) {
        ctx.fillStyle = "rgba(8,0,2,0.45)"; ctx.fillRect(0, 0, W, H);
        const reveal = Math.min(1, p * 1.6);
        ctx.strokeStyle = "#e63946"; ctx.lineWidth = 2.2;
        ctx.shadowColor = "#e63946"; ctx.shadowBlur = 9;
        for (const v of s.veins) {
            const n = Math.max(2, Math.floor(v.length * reveal));
            ctx.beginPath();
            for (let i = 0; i < n; i++) i ? ctx.lineTo(v[i].x, v[i].y) : ctx.moveTo(v[i].x, v[i].y);
            ctx.stroke();
        }
        ctx.shadowBlur = 0;
        // CRT scanlines with random flicker
        const flick = Math.random() < 0.12 ? 0.3 : 0.13;
        ctx.fillStyle = `rgba(0,0,0,${flick})`;
        for (let y = 0; y < H; y += 4) ctx.fillRect(0, y, W, 1.4);
        if (p > 0.45) {
            ctx.font = "26px Georgia, serif"; ctx.textAlign = "center";
            ctx.fillStyle = "#ff4d5e";
            ctx.globalAlpha = (Math.random() < 0.08 ? 0.55 : 1) * Math.min(1, (p - 0.45) * 4);
            ctx.fillText("F R I E N D S   D O N ' T   L I E", W / 2, H / 2);
            ctx.globalAlpha = 1;
        }
    },
};

// 11 · dragonball — charging aura ball, pulse, radial-ray release.
_MAP["dragonball"] = {
    dur: 2800,
    init(s) { s.rays = 24; },
    frame(s, ctx, W, H, p) {
        const cx = W / 2, cy = H / 2;
        const charge = Math.min(1, p / 0.6);
        const pulse = 1 + Math.sin(p * 26) * 0.08 * charge;
        const R = (30 + 90 * charge) * pulse;
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
        g.addColorStop(0, "rgba(255,255,255,0.95)");
        g.addColorStop(0.45, "rgba(255,224,102,0.85)");
        g.addColorStop(1, "rgba(255,224,102,0)");
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.fill();
        if (p > 0.6) {                                            // release rays
            const k = (p - 0.6) / 0.4;
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            ctx.translate(cx, cy); ctx.rotate(k * 0.5);
            for (let i = 0; i < s.rays; i++) {
                const a = (i / s.rays) * Math.PI * 2;
                const len = Math.hypot(W, H) * 0.6 * k;
                const grad = ctx.createLinearGradient(0, 0, Math.cos(a) * len, Math.sin(a) * len);
                grad.addColorStop(0, "rgba(255,236,150,0.8)");
                grad.addColorStop(1, "rgba(255,236,150,0)");
                ctx.strokeStyle = grad; ctx.lineWidth = 7;
                ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(Math.cos(a) * len, Math.sin(a) * len); ctx.stroke();
            }
            ctx.restore();
            ctx.font = "bold 26px monospace"; ctx.textAlign = "center";
            ctx.fillStyle = PAL.gold; ctx.globalAlpha = Math.min(1, k * 2);
            ctx.fillText("POWER LEVEL: 9000+", cx, cy - R - 24);
            ctx.globalAlpha = 1;
        }
    },
};

// 12 · sparkles — minimal gold twinkles (also the frame-budget fallback).
_MAP["sparkles"] = {
    dur: 1200,
    init(s, W, H, I) {
        s.ps = Array.from({ length: Math.round(60 * I) }, () => ({
            x: Math.random() * W, y: Math.random() * H,
            ph: _rand(0, Math.PI * 2), r: _rand(2, 5), sp: _rand(5, 9),
        }));
    },
    frame(s, ctx, W, H, p) {
        for (const q of s.ps) {
            const a = Math.max(0, Math.sin(q.ph + p * q.sp));
            ctx.save();
            ctx.translate(q.x, q.y); ctx.rotate(q.ph + p * 2);
            ctx.globalAlpha = a * (1 - p * 0.5);
            ctx.fillStyle = PAL.gold;
            ctx.beginPath();                                       // 4-point star
            for (let i = 0; i < 8; i++) {
                const rr = i % 2 ? q.r * 0.35 : q.r;
                const aa = (i / 8) * Math.PI * 2;
                i ? ctx.lineTo(Math.cos(aa) * rr, Math.sin(aa) * rr) : ctx.moveTo(rr, 0);
            }
            ctx.closePath(); ctx.fill();
            ctx.restore();
        }
    },
};

// 13 · balloons — rise from the bottom, bob, strings dangle, pop at top.
_MAP["balloons"] = {
    dur: 3000,
    init(s, W, H, I) {
        s.bs = Array.from({ length: Math.round(12 * I) }, (_, i) => ({
            x: _rand(W * 0.06, W * 0.94), y: H + _rand(20, 240),
            vy: _rand(-2.6, -1.7), bobA: _rand(0, Math.PI * 2), bobW: _rand(2, 4),
            r: _rand(16, 26), color: _pick(CONFETTI_COLORS), popped: 0,
        }));
    },
    frame(s, ctx, W, H, p) {
        for (const b of s.bs) {
            if (b.popped) {
                if (b.popped < 8) {                                // brief pop burst
                    ctx.strokeStyle = b.color; ctx.lineWidth = 2;
                    for (let i = 0; i < 6; i++) {
                        const a = (i / 6) * Math.PI * 2;
                        ctx.beginPath();
                        ctx.moveTo(b.x + Math.cos(a) * b.popped * 2, b.y + Math.sin(a) * b.popped * 2);
                        ctx.lineTo(b.x + Math.cos(a) * (b.popped * 2 + 7), b.y + Math.sin(a) * (b.popped * 2 + 7));
                        ctx.stroke();
                    }
                    b.popped++;
                }
                continue;
            }
            b.y += b.vy;
            const bx = b.x + Math.sin(p * Math.PI * 2 * b.bobW + b.bobA) * 14;
            if (b.y < b.r + 6) { b.popped = 1; b.x = bx; continue; }
            ctx.strokeStyle = "rgba(220,220,230,0.7)"; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(bx, b.y + b.r);
            ctx.quadraticCurveTo(bx + 6, b.y + b.r + 18, bx - 3, b.y + b.r + 34); ctx.stroke();
            ctx.fillStyle = b.color;
            ctx.beginPath(); ctx.ellipse(bx, b.y, b.r * 0.82, b.r, 0, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = "rgba(255,255,255,0.45)";
            ctx.beginPath(); ctx.ellipse(bx - b.r * 0.3, b.y - b.r * 0.35, b.r * 0.2, b.r * 0.28, -0.5, 0, Math.PI * 2); ctx.fill();
        }
    },
};

/* ════════════════════ chimes ════════════════════ */
let _audioCtx = null;
function _tone(type, freq, t, dur, vol) {
    const ctx = _audioCtx;
    const osc = ctx.createOscillator(), gain = ctx.createGain();
    osc.type = type; osc.frequency.value = freq;
    osc.connect(gain).connect(ctx.destination);
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(vol, t + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    osc.start(t); osc.stop(t + dur + 0.05);
}
function _chime() {
    try {
        _audioCtx = _audioCtx || new (window.AudioContext || window.webkitAudioContext)();
        const now = _audioCtx.currentTime;
        const style = _setting("chime_style", "arpeggio");
        if (style === "mario") {              // square-wave C-E-G-C
            [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => _tone("square", f, now + i * 0.09, 0.18, 0.08));
        } else if (style === "tron") {        // low saw pulse rising
            [82.4, 110, 164.8].forEach((f, i) => _tone("sawtooth", f, now + i * 0.12, 0.4, 0.10));
        } else if (style === "saber") {       // detuned square swell + clash
            _tone("square", 90, now, 0.5, 0.07); _tone("square", 93, now, 0.5, 0.07);
            _tone("triangle", 740, now + 0.42, 0.22, 0.12);
        } else {                              // arpeggio (classic sine C5-E5-G5)
            [523.25, 659.25, 783.99].forEach((f, i) => _tone("sine", f, now + i * 0.08, 0.35, 0.15));
        }
    } catch (e) { __c2cReport("c2c_completion_fx", e); }
}

/* ════════════════════ trigger + registration ════════════════════ */
function _play(styleOverride) {
    let style = styleOverride || _setting("style", "confetti");
    try {
        if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) style = "sparkles";
    } catch (_) {}
    if (style === "random") style = _pick(STYLES.filter((x) => x !== "random"));
    _run(style);
}

function _onSuccess() {
    try {
        const minMs = Number(_setting("min_runtime_ms", 0)) || 0;
        if (minMs > 0 && _lastRunStart && performance.now() - _lastRunStart < minMs) return;
        if (_setting("confetti", true)) _play();
        if (_setting("chime", false)) _chime();
    } catch { /* ignore */ }
}

app.registerExtension({
    name: "C2C.CompletionFX",
    // NOTE: ComfyUI's settings panel reads `defaultValue` — registering with
    // only `default` rendered every combo EMPTY (the reported "celebration
    // options not given"). Ids live under c2c.* so they file in the C2C
    // section; legacy mec.* values are migrated once in setup().
    settings: [
        { id: "c2c.completion_fx.confetti", name: "C2C Completion FX: enabled (celebrate on success)",
          type: "boolean", defaultValue: true, default: true },
        { id: "c2c.completion_fx.style", name: "C2C Completion FX: style",
          type: "combo", options: STYLES, defaultValue: "confetti", default: "confetti",
          tooltip: "Which celebration plays on workflow success. 'random' picks one each run." },
        { id: "c2c.completion_fx.intensity", name: "C2C Completion FX: intensity",
          type: "combo", options: ["low", "normal", "high"], defaultValue: "normal", default: "normal",
          tooltip: "Scales particle counts. 'low' for low-end laptops." },
        { id: "c2c.completion_fx.min_runtime_ms", name: "C2C Completion FX: skip if run shorter than (ms)",
          type: "number", defaultValue: 0, default: 0,
          tooltip: "Don't celebrate a 200ms test run. 0 = always celebrate." },
        { id: "c2c.completion_fx.chime", name: "C2C Completion FX: play chime",
          type: "boolean", defaultValue: false, default: false },
        { id: "c2c.completion_fx.chime_style", name: "C2C Completion FX: chime style",
          type: "combo", options: ["arpeggio", "mario", "tron", "saber"],
          defaultValue: "arpeggio", default: "arpeggio" },
    ],
    commands: [
        { id: "C2C.CompletionFX.test", label: "Completion FX: preview current style",
          function: () => _play() },
    ],
    async setup() {
        // One-time migration: copy any stored legacy mec.completion_fx.* value
        // to its c2c.completion_fx.* id (skip if the new id already has one).
        try {
            for (const k of ["confetti", "style", "intensity", "min_runtime_ms", "chime", "chime_style"]) {
                const cur = app.ui.settings.getSettingValue("c2c.completion_fx." + k);
                const old = app.ui.settings.getSettingValue("mec.completion_fx." + k);
                if ((cur === undefined || cur === null || cur === "") &&
                    old !== undefined && old !== null && old !== "") {
                    await app.ui.settings.setSettingValueAsync?.("c2c.completion_fx." + k, old)
                        ?? app.ui.settings.setSettingValue("c2c.completion_fx." + k, old);
                }
            }
        } catch (_) { /* migration is best-effort */ }
        api.addEventListener("execution_start", () => { _lastRunStart = performance.now(); });
        api.addEventListener("execution_success", _onSuccess);
        // Public hook: manual triggers + c2c_celebrations.js delegates here
        // instead of double-firing its own confetti.
        window.__C2C_FX = { play: _play, styles: STYLES.slice() };
        console.log("[C2C.CompletionFX] Loaded — styles:", STYLES.join(", "));
    },
});
