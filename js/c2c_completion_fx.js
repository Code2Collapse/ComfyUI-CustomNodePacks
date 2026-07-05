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
import { LITE } from "./_c2c_lite.js";
import { api } from "../../scripts/api.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const CANVAS_ID = "mec-confetti-canvas";
const STYLES = ["confetti", "fireworks", "starwars", "tron", "spiderman",
    "spiderman-swing", "matrix", "portal", "mario", "saber-clash", "saber-duel",
    "stranger", "dragonball", "trophy", "level-up", "sparkles", "balloons",
    // franchise FX (2026-07)
    "pokemon", "minecraft", "pacman", "sonic", "zelda",
    // game FX batch 2 — arcade + console (2026-07)
    "halo", "gta", "tetris", "space-invaders", "street-fighter",
    "mortal-kombat", "doom", "metroid", "elden-ring", "among-us",
    "random"];

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

// 19 · pokemon — Pikachu-yellow electric bolts + a bouncing Poké Ball.
_MAP["pokemon"] = {
    dur: 2600,
    init(s, W, H, I) {
        s.bolts = Array.from({ length: Math.round(16 * I) }, () => ({
            x: _rand(0, W), y: _rand(0, H * 0.6),
            len: _rand(40, 100), ang: _rand(0, Math.PI * 2), seed: Math.random() * 99,
        }));
        s.ball = { x: W / 2, y: H * 0.35, vy: -2, vx: _rand(-3, 3) };
    },
    frame(s, ctx, W, H, p, now) {
        for (const b of s.bolts) {
            if (((now / 90 + b.seed) % 1) > 0.6) continue;          // flicker
            ctx.save();
            ctx.strokeStyle = "#ffe14d"; ctx.lineWidth = 2.4;
            ctx.shadowColor = "#ffd400"; ctx.shadowBlur = 10;
            ctx.globalAlpha = 0.9 * (1 - p * 0.5);
            ctx.beginPath();
            let x = b.x, y = b.y; ctx.moveTo(x, y);
            for (let k = 0; k < 5; k++) { x += Math.cos(b.ang) * b.len / 5 + _rand(-9, 9); y += Math.sin(b.ang) * b.len / 5 + _rand(-9, 9); ctx.lineTo(x, y); }
            ctx.stroke(); ctx.restore();
        }
        const bl = s.ball; bl.vy += 0.35; bl.x += bl.vx; bl.y += bl.vy;
        if (bl.y > H * 0.72) { bl.y = H * 0.72; bl.vy *= -0.62; }
        ctx.save(); ctx.translate(bl.x, bl.y); ctx.globalAlpha = 1 - p * 0.25;
        const r = 24;
        ctx.beginPath(); ctx.arc(0, 0, r, Math.PI, 0); ctx.fillStyle = "#ee3b3b"; ctx.fill();
        ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI); ctx.fillStyle = "#f2f2f2"; ctx.fill();
        ctx.fillStyle = "#1a1a1a"; ctx.fillRect(-r, -3, 2 * r, 6);
        ctx.beginPath(); ctx.arc(0, 0, 7, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
        ctx.strokeStyle = "#1a1a1a"; ctx.lineWidth = 2.5; ctx.stroke();
        ctx.restore();
    },
};

// 20 · minecraft — pixelated blocks (grass/dirt/gold/diamond) tumble down.
_MAP["minecraft"] = {
    dur: 2600,
    init(s, W, H, I) {
        const COLS = ["#7CB342", "#8D6E63", "#FDD835", "#4DD0E1", "#B0BEC5", "#E53935"];
        s.bs = Array.from({ length: Math.round(46 * I) }, () => ({
            x: Math.random() * W, y: -20 - Math.random() * 260,
            vy: _rand(2, 5), size: Math.round(_rand(10, 22) / 2) * 2, // even → crisp pixels
            color: _pick(COLS),
        }));
    },
    frame(s, ctx, W, H) {
        ctx.imageSmoothingEnabled = false;
        for (const b of s.bs) {
            b.vy += 0.06; b.y += b.vy;
            const g = b.size;
            ctx.fillStyle = b.color;
            ctx.fillRect(Math.round(b.x), Math.round(b.y), g, g);
            // simple pixel shading (darker bottom-right quarter)
            ctx.fillStyle = "rgba(0,0,0,0.22)";
            ctx.fillRect(Math.round(b.x) + g / 2, Math.round(b.y) + g / 2, g / 2, g / 2);
            ctx.fillStyle = "rgba(255,255,255,0.18)";
            ctx.fillRect(Math.round(b.x), Math.round(b.y), g / 2, g / 2);
        }
    },
};

// 21 · pacman — Pac-Man chomps across the screen eating a row of pellets.
_MAP["pacman"] = {
    dur: 2600,
    init(s, W, H) {
        s.y = H * 0.5;
        s.pellets = Array.from({ length: 16 }, (_, i) => ({ x: W * (0.08 + i * 0.058), eaten: false }));
    },
    frame(s, ctx, W, H, p, now) {
        const px = W * (-0.05 + p * 1.12);
        for (const pel of s.pellets) {
            if (pel.x < px) pel.eaten = true;
            if (pel.eaten) continue;
            ctx.fillStyle = "#ffd54a"; ctx.beginPath(); ctx.arc(pel.x, s.y, 5, 0, Math.PI * 2); ctx.fill();
        }
        const mouth = (Math.abs(Math.sin(now / 90)) * 0.5 + 0.06) * Math.PI;
        ctx.save(); ctx.translate(px, s.y);
        ctx.fillStyle = "#ffe14d"; ctx.shadowColor = "#ffe14d"; ctx.shadowBlur = 14;
        ctx.beginPath(); ctx.moveTo(0, 0);
        ctx.arc(0, 0, 26, mouth, Math.PI * 2 - mouth); ctx.closePath(); ctx.fill();
        ctx.shadowBlur = 0; ctx.fillStyle = "#1a1a1a";
        ctx.beginPath(); ctx.arc(4, -11, 3.2, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
    },
};

// 22 · sonic — spinning gold rings + blue speed streaks.
_MAP["sonic"] = {
    dur: 2600,
    init(s, W, H, I) {
        s.rings = Array.from({ length: Math.round(14 * I) }, () => ({
            x: _rand(W * 0.1, W * 0.9), y: _rand(H * 0.15, H * 0.85),
            r: _rand(10, 20), ph: _rand(0, Math.PI * 2), sp: _rand(4, 8), vy: _rand(-1, -3),
        }));
        s.streaks = Array.from({ length: Math.round(10 * I) }, () => ({ y: _rand(0, H), x: _rand(0, W), len: _rand(60, 160) }));
    },
    frame(s, ctx, W, H, p, now) {
        for (const st of s.streaks) {
            st.x -= 26; if (st.x < -st.len) st.x = W + _rand(0, 80);
            const grd = ctx.createLinearGradient(st.x, st.y, st.x + st.len, st.y);
            grd.addColorStop(0, "rgba(41,120,255,0)"); grd.addColorStop(1, "rgba(120,190,255,0.7)");
            ctx.strokeStyle = grd; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(st.x, st.y); ctx.lineTo(st.x + st.len, st.y); ctx.stroke();
        }
        for (const rg of s.rings) {
            rg.y += rg.vy; rg.ph += 0.25;
            const sx = Math.abs(Math.cos(rg.ph)) * rg.r + 3;      // spin → ellipse squash
            ctx.save(); ctx.translate(rg.x, rg.y); ctx.globalAlpha = 1 - p * 0.4;
            ctx.strokeStyle = "#ffd23f"; ctx.lineWidth = 4; ctx.shadowColor = "#ffd23f"; ctx.shadowBlur = 8;
            ctx.beginPath(); ctx.ellipse(0, 0, sx, rg.r, 0, 0, Math.PI * 2); ctx.stroke();
            ctx.restore();
        }
    },
};

// 23 · zelda — the golden Triforce rises with a burst of green sparkles.
_MAP["zelda"] = {
    dur: 2800,
    init(s, W, H, I) {
        s.sp = Array.from({ length: Math.round(40 * I) }, () => ({
            x: W / 2 + _rand(-40, 40), y: H * 0.6 + _rand(-20, 20),
            a: _rand(0, Math.PI * 2), sp: _rand(1.5, 5), life: _rand(0.5, 1),
        }));
    },
    frame(s, ctx, W, H, p, now) {
        // triforce (3 stacked triangles) rising + shimmering
        const cy = H * 0.62 - p * H * 0.18, cx = W / 2, T = 46 * (0.6 + p * 0.6);
        const glow = 0.6 + 0.4 * Math.sin(now / 120);
        ctx.save(); ctx.globalAlpha = Math.min(1, p * 2);
        ctx.fillStyle = "#ffe14d"; ctx.strokeStyle = "#fff6c2"; ctx.lineWidth = 2;
        ctx.shadowColor = "#ffd400"; ctx.shadowBlur = 20 * glow;
        const tri = (ox, oy) => { ctx.beginPath(); ctx.moveTo(cx + ox, cy + oy - T); ctx.lineTo(cx + ox - T * 0.87, cy + oy + T * 0.5); ctx.lineTo(cx + ox + T * 0.87, cy + oy + T * 0.5); ctx.closePath(); ctx.fill(); ctx.stroke(); };
        tri(0, -T * 0.5); tri(-T * 0.87, T * 0.5); tri(T * 0.87, T * 0.5);
        ctx.restore();
        // green + gold sparkles
        for (const q of s.sp) {
            q.x += Math.cos(q.a) * q.sp; q.y += Math.sin(q.a) * q.sp - 0.4;
            const al = Math.max(0, q.life - p);
            ctx.fillStyle = Math.random() < 0.5 ? "#8cff66" : "#ffe14d";
            ctx.globalAlpha = al; ctx.shadowColor = "#8cff66"; ctx.shadowBlur = 6;
            ctx.beginPath(); ctx.arc(q.x, q.y, 2.2, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = 1; ctx.shadowBlur = 0;
    },
};

// 24 · halo — an energy shockwave + Halo ring arc + plasma sparks.
_MAP["halo"] = {
    dur: 2600,
    init(s, W, H, I) {
        s.sparks = Array.from({ length: Math.round(50 * I) }, () => ({ a: _rand(0, Math.PI * 2), sp: _rand(3, 9), life: _rand(0.6, 1) }));
    },
    frame(s, ctx, W, H, p, now) {
        p = p < 0 ? 0 : p > 1 ? 1 : p;   // driver can pass a lead-in p<0 → negative arc radius
        const cx = W / 2, cy = H / 2;
        ctx.save();
        ctx.strokeStyle = "#39d0ff"; ctx.lineWidth = Math.max(0.5, 6 * (1 - p)); ctx.globalAlpha = 1 - p;
        ctx.shadowColor = "#39d0ff"; ctx.shadowBlur = 24;
        ctx.beginPath(); ctx.arc(cx, cy, p * Math.min(W, H) * 0.55, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = Math.min(1, p * 1.5) * 0.8; ctx.strokeStyle = "#bff4ff"; ctx.lineWidth = 10; ctx.shadowBlur = 12;
        ctx.beginPath(); ctx.arc(cx, cy + H * 0.5, W * 0.7, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke();
        for (const q of s.sparks) {
            const r = q.sp * p * 40, al = Math.max(0, q.life - p);
            ctx.globalAlpha = al; ctx.fillStyle = "#8ff0ff"; ctx.shadowColor = "#39d0ff"; ctx.shadowBlur = 8;
            ctx.beginPath(); ctx.arc(cx + Math.cos(q.a) * r, cy + Math.sin(q.a) * r, 2.4, 0, Math.PI * 2); ctx.fill();
        }
        ctx.restore();
    },
};

// 25 · gta — "MISSION PASSED" in gold with neon + wanted stars lighting up.
_MAP["gta"] = {
    dur: 3000,
    init(s) { s.n = 6; },
    frame(s, ctx, W, H, p, now) {
        const cx = W / 2, cy = H * 0.42;
        ctx.save(); ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
        ctx.font = "800 42px Impact, system-ui, sans-serif";
        ctx.shadowColor = "#ff43a4"; ctx.shadowBlur = 18; ctx.globalAlpha = Math.min(1, p * 2);
        ctx.fillStyle = "#ffe14d"; ctx.fillText("MISSION PASSED", cx, cy);
        ctx.font = "700 20px Impact, system-ui, sans-serif"; ctx.shadowColor = "#39c0ff"; ctx.shadowBlur = 10;
        ctx.fillStyle = "#8fe9ff"; ctx.fillText("RESPECT ++", cx, cy + 30);
        const cy2 = cy + 64;
        for (let i = 0; i < s.n; i++) {
            const lit = p > 0.3 + i * 0.09, x = cx + (i - (s.n - 1) / 2) * 34;
            ctx.fillStyle = lit ? "#ffe14d" : "#3a3a3a"; ctx.shadowBlur = lit ? 12 : 0; ctx.shadowColor = "#fff2a0";
            ctx.beginPath();
            for (let k = 0; k < 10; k++) { const r = k % 2 ? 4 : 9, ang = Math.PI / 5 * k - Math.PI / 2, px = x + Math.cos(ang) * r, py = cy2 + Math.sin(ang) * r; k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
            ctx.closePath(); ctx.fill();
        }
        ctx.restore();
    },
};

// 26 · tetris — tetrominoes rain down, the bottom row flashes then clears.
_MAP["tetris"] = {
    dur: 2600,
    init(s, W, H) {
        s.cell = Math.max(16, Math.floor(W / 26));
        s.cols = Math.max(6, Math.floor(W / s.cell));
        const C = ["#00f0f0", "#f0f000", "#a000f0", "#00f000", "#f00000", "#0000f0", "#f0a000"];
        s.blocks = Array.from({ length: Math.round(s.cols * 1.6) }, () => ({ c: (Math.random() * s.cols) | 0, col: _pick(C), delay: _rand(0, 0.5) }));
    },
    frame(s, ctx, W, H, p, now) {
        const cell = s.cell, floorY = H - cell;
        for (const bl of s.blocks) {
            const fp = Math.min(1, Math.max(0, (p - bl.delay) / 0.5));
            ctx.fillStyle = bl.col; ctx.shadowColor = bl.col; ctx.shadowBlur = 6;
            ctx.fillRect(bl.c * cell + 1, -cell + fp * (floorY + cell) + 1, cell - 2, cell - 2);
        }
        ctx.shadowBlur = 0;
        if (p > 0.6) {
            ctx.fillStyle = `rgba(255,255,255,${0.3 + 0.5 * ((Math.sin(now / 60) + 1) / 2)})`;
            ctx.fillRect(0, floorY, (p - 0.6) / 0.4 * W, cell);
        }
    },
};

// 27 · space-invaders — a squadron descends, then bursts into pixel confetti.
_MAP["space-invaders"] = {
    dur: 2800,
    init(s) { s.rows = 3; s.cols = 8; s.exploded = false; s.bits = []; },
    frame(s, ctx, W, H, p, now) {
        const unit = Math.min(W, H) * 0.045, frame = Math.floor(now / 300) % 2;
        const rows = frame ? ["01110", "11111", "10101", "01010"] : ["01110", "11111", "10101", "10001"];
        if (p < 0.6) {
            const descend = p / 0.6;
            ctx.fillStyle = "#4dff5a"; ctx.shadowColor = "#39ff7a"; ctx.shadowBlur = 8;
            for (let r = 0; r < s.rows; r++) for (let c = 0; c < s.cols; c++) {
                const ox = W * 0.5 + (c - (s.cols - 1) / 2) * unit * 6.5;
                const oy = H * 0.14 + r * unit * 6 + descend * H * 0.24;
                for (let br = 0; br < 4; br++) for (let bc = 0; bc < 5; bc++) if (rows[br][bc] === "1")
                    ctx.fillRect(ox + (bc - 2.5) * unit, oy + (br - 2) * unit, unit, unit);
            }
        } else {
            if (!s.exploded) {
                s.exploded = true;
                for (let i = 0; i < 130; i++) s.bits.push({ x: _rand(W * 0.2, W * 0.8), y: _rand(H * 0.3, H * 0.7), vx: _rand(-6, 6), vy: _rand(-9, 2), col: _pick(["#4dff5a", "#ffffff", "#39ff7a"]) });
            }
            for (const bt of s.bits) { bt.x += bt.vx; bt.y += bt.vy; bt.vy += 0.4; ctx.fillStyle = bt.col; ctx.fillRect(bt.x, bt.y, unit, unit); }
        }
        ctx.shadowBlur = 0;
    },
};

// 28 · street-fighter — a Hadouken shockwave detonates a big "K.O.".
_MAP["street-fighter"] = {
    dur: 2600,
    init() {},
    frame(s, ctx, W, H, p, now) {
        const cx = W / 2, cy = H / 2;
        ctx.save();
        for (let k = 0; k < 3; k++) {
            const rp = p * 1.4 - k * 0.18; if (rp < 0 || rp > 1) continue;
            ctx.globalAlpha = (1 - rp) * 0.8; ctx.strokeStyle = "#39c0ff"; ctx.lineWidth = 8 * (1 - rp);
            ctx.shadowColor = "#39c0ff"; ctx.shadowBlur = 20;
            ctx.beginPath(); ctx.arc(cx, cy, rp * Math.min(W, H) * 0.5, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.globalAlpha = Math.min(1, (p - 0.2) * 3);
        const sc = 1 + Math.max(0, 0.4 - p) * 2;
        ctx.translate(cx, cy); ctx.scale(sc, sc); ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.font = "900 90px Impact, system-ui, sans-serif";
        ctx.lineWidth = 5; ctx.strokeStyle = "#0a1a3a"; ctx.strokeText("K.O.", 0, 0);
        ctx.fillStyle = "#ffe14d"; ctx.shadowColor = "#ff8c1a"; ctx.shadowBlur = 22; ctx.fillText("K.O.", 0, 0);
        ctx.restore();
    },
};

// 29 · mortal-kombat — red lightning bolts + "FLAWLESS VICTORY".
_MAP["mortal-kombat"] = {
    dur: 3000,
    init(s) { s.bolts = Array.from({ length: 5 }, () => ({ x: _rand(0.15, 0.85) })); },
    frame(s, ctx, W, H, p, now) {
        const phase = Math.floor(now / 60);
        const rnd = (i) => { const v = Math.sin(i * 12.9 + phase * 7.1) * 43758.5; return v - Math.floor(v); };
        ctx.save(); ctx.strokeStyle = "#ff2b1f"; ctx.lineWidth = 2; ctx.shadowColor = "#ff3b2f"; ctx.shadowBlur = 10;
        s.bolts.forEach((bl, bi) => {
            ctx.beginPath(); let x = bl.x * W;
            for (let y = 0; y <= H; y += 24) { x += (rnd(bi * 10 + y) - 0.5) * 26; y ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
            ctx.stroke();
        });
        ctx.globalAlpha = Math.min(1, p * 2); ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.font = "900 46px 'Times New Roman', Georgia, serif";
        ctx.fillStyle = "#c1121f"; ctx.shadowColor = "#ff5a4d"; ctx.shadowBlur = 20;
        ctx.fillText("FLAWLESS", W / 2, H / 2 - 26);
        ctx.fillText("VICTORY", W / 2, H / 2 + 26);
        ctx.restore();
    },
};

// 30 · doom — green plasma burst + blood-red flash + "RIP AND TEAR".
_MAP["doom"] = {
    dur: 2600,
    init(s, W, H, I) { s.ps = Array.from({ length: Math.round(60 * I) }, () => ({ a: _rand(0, Math.PI * 2), sp: _rand(4, 11), life: _rand(0.5, 1) })); },
    frame(s, ctx, W, H, p, now) {
        p = p < 0 ? 0 : p > 1 ? 1 : p;   // guard: p<0 lead-in would make plasma radii negative
        const cx = W / 2, cy = H / 2;
        ctx.save();
        ctx.fillStyle = `rgba(140,10,10,${Math.max(0, 0.5 - p)})`; ctx.fillRect(0, 0, W, H);
        for (const q of s.ps) {
            const r = q.sp * p * 42, al = Math.max(0, q.life - p);
            ctx.globalAlpha = al; ctx.fillStyle = "#7cff4d"; ctx.shadowColor = "#39ff14"; ctx.shadowBlur = 10;
            ctx.beginPath(); ctx.arc(cx + Math.cos(q.a) * r, cy + Math.sin(q.a) * r, 3, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = Math.min(1, p * 2); ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.font = "900 44px Impact, system-ui, sans-serif";
        ctx.fillStyle = "#ff3b1f"; ctx.shadowColor = "#ff7a1e"; ctx.shadowBlur = 18;
        ctx.fillText("RIP AND TEAR", cx, cy);
        ctx.restore();
    },
};

// 31 · metroid — Samus's charge blast rings out in orange energy.
_MAP["metroid"] = {
    dur: 2600,
    init(s) { s.rings = 3; },
    frame(s, ctx, W, H, p, now) {
        p = p < 0 ? 0 : p > 1 ? 1 : p;   // guard: p<0 lead-in would make ring/orb radii negative
        const cx = W / 2, cy = H / 2;
        ctx.save();
        for (let k = 0; k < s.rings; k++) {
            const rp = p * 1.3 - k * 0.15; if (rp < 0 || rp > 1) continue;
            ctx.globalAlpha = 1 - rp; ctx.strokeStyle = "#ff9a3a"; ctx.lineWidth = 7 * (1 - rp);
            ctx.shadowColor = "#ffb03a"; ctx.shadowBlur = 20;
            ctx.beginPath(); ctx.arc(cx, cy, rp * Math.min(W, H) * 0.5, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.globalAlpha = 1; ctx.fillStyle = "#fff0c8"; ctx.shadowColor = "#ffb03a"; ctx.shadowBlur = 26;
        ctx.beginPath(); ctx.arc(cx, cy, 16 * (0.6 + 0.4 * Math.sin(now / 90)) * (1 - p * 0.5), 0, Math.PI * 2); ctx.fill();
        ctx.restore();
    },
};

// 32 · elden-ring — golden "ENEMY FELLED" with rising erdtree embers.
_MAP["elden-ring"] = {
    dur: 3200,
    init(s, W, H, I) { s.emb = Array.from({ length: Math.round(60 * I) }, () => ({ x: _rand(0, W), y: _rand(H * 0.5, H), sp: _rand(0.4, 1.4), ph: _rand(0, 6) })); },
    frame(s, ctx, W, H, p, now) {
        for (const e of s.emb) {
            e.y -= e.sp * 2.2;
            ctx.globalAlpha = 0.8; ctx.fillStyle = "#ffdf7e"; ctx.shadowColor = "#ffcf4d"; ctx.shadowBlur = 8;
            ctx.beginPath(); ctx.arc(e.x + Math.sin(now / 400 + e.ph) * 6, e.y, 1.8, 0, Math.PI * 2); ctx.fill();
        }
        ctx.globalAlpha = Math.min(1, p * 1.6); ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.font = "600 40px 'Times New Roman', Georgia, serif";
        ctx.fillStyle = "#e8c14a"; ctx.shadowColor = "#ffe27a"; ctx.shadowBlur = 22;
        ctx.fillText("ENEMY FELLED", W / 2, H / 2);
        ctx.globalAlpha = Math.min(1, p * 1.6) * 0.8; ctx.strokeStyle = "#c9a227"; ctx.lineWidth = 1.5; ctx.shadowBlur = 0;
        ctx.beginPath(); ctx.moveTo(W * 0.3, H / 2 + 26); ctx.lineTo(W * 0.7, H / 2 + 26); ctx.stroke();
        ctx.globalAlpha = 1;
    },
};

// 33 · among-us — a crewmate parade slides across under "VICTORY".
_MAP["among-us"] = {
    dur: 2800,
    init(s, W, H) {
        const C = ["#c51111", "#132ed1", "#117f2d", "#ed54ba", "#f07d0d", "#f6f657", "#3f474e", "#d6e0f0"];
        s.crew = Array.from({ length: 6 }, (_, i) => ({ x: -_rand(0, W * 0.5) - i * 60, y: H * (0.55 + (i % 2) * 0.12), col: C[i % C.length], sp: _rand(3, 5) }));
    },
    frame(s, ctx, W, H, p, now) {
        for (const cm of s.crew) {
            cm.x += cm.sp; const wob = Math.sin(now / 130 + cm.x / 40) * 2, R = Math.min(W, H) * 0.05;
            ctx.save(); ctx.translate(cm.x, cm.y + wob);
            ctx.fillStyle = cm.col; ctx.shadowColor = "rgba(0,0,0,0.4)"; ctx.shadowBlur = 6;
            ctx.fillRect(-R * 1.3, -R * 0.4, R * 0.7, R * 1.2);
            ctx.beginPath(); ctx.arc(0, -R * 0.3, R, Math.PI, 0); ctx.lineTo(R, R); ctx.lineTo(-R, R); ctx.closePath(); ctx.fill();
            ctx.fillStyle = "#a9d3e8"; ctx.shadowBlur = 0;
            ctx.beginPath(); ctx.ellipse(R * 0.35, -R * 0.4, R * 0.55, R * 0.4, 0, 0, Math.PI * 2); ctx.fill();
            ctx.restore();
        }
        ctx.globalAlpha = Math.min(1, p * 2); ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.font = "800 46px system-ui, sans-serif";
        ctx.fillStyle = "#f6f657"; ctx.shadowColor = "#fff7a0"; ctx.shadowBlur = 16;
        ctx.fillText("VICTORY", W / 2, H * 0.28);
        ctx.globalAlpha = 1; ctx.shadowBlur = 0;
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

if (!LITE) app.registerExtension({
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
