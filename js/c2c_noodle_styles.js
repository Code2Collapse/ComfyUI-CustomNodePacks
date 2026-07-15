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
import { LITE } from "./_c2c_lite.js";
import { C } from './_c2c_theme.js';
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { api } from "/scripts/api.js";

const STYLES = [
    "default","spider-web","lightsaber","dna-helix","rainbow-flow",
    "dashed-march","lightning","pulse-packet","neon-tube",
    // new (2026-06-30)
    "rgb-spectrum","gradient","fire","comet","candy","aurora","heartbeat",
    // franchise styles (2026-06-30)
    "matrix","tron","portal","pacman","hyperspace","upside-down","ki-blast","rainbow-road",
    // franchise games/movies (batch 2 — implemented earlier, now exposed 2026-07)
    "pokemon","minecraft","sonic","zelda","halo","among-us",
    "avatar-bending","john-wick","ghostbusters","jurassic",
    // game classics (batch 3 — arcade + console, 2026-07)
    "gta","space-invaders","tetris","street-fighter","mega-man",
    "mortal-kombat","metroid","doom","elden-ring","galaga",
];
const SETTING_ID = "mec.noodle.style";

// Styles that need a continuous (throttled ~20fps) redraw to animate. Static
// styles (neon-tube, gradient, minecraft voxel trail, spider-web figure) are
// intentionally absent so they don't force idle repaints.
// ── UNIQUE per-skin animation overlays ─────────────────────────────────
// Every animated skin gets its OWN signature motion (user spec 2026-07-15),
// drawn as a light overlay AFTER the skin base render. Positions sample
// _bezierAt, so overlays follow whatever pipe SHAPE is active. All overlays
// ride the existing 20fps dirty-throttle and pause when the tab is hidden.
function _apt(t, a, b) {
    const [c1, c2] = _bezierPoints(a, b);
    return _bezierAt(Math.max(0, Math.min(1, t)), a, c1, c2, b);
}
const _TAU = Math.PI * 2;
function _aDot(ctx, p, r, col, glow) {
    ctx.save(); if (glow) { ctx.shadowColor = col; ctx.shadowBlur = glow; }
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, _TAU); ctx.fill(); ctx.restore();
}
function _aRing(ctx, p, r, col, w, glow) {
    ctx.save(); if (glow) { ctx.shadowColor = col; ctx.shadowBlur = glow; }
    ctx.strokeStyle = col; ctx.lineWidth = w || 2;
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, _TAU); ctx.stroke(); ctx.restore();
}
function _aGlyph(ctx, p, txt, size, col, rot) {
    ctx.save(); ctx.translate(p[0], p[1]); if (rot) ctx.rotate(rot);
    ctx.fillStyle = col; ctx.font = size + "px monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(txt, 0, 0); ctx.restore();
}
// Identity-disc: a ring with a bright rim + dark core (tron).
function _aDisc(ctx, p, r, col) {
    ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = 10;
    ctx.strokeStyle = col; ctx.lineWidth = 3.5;
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, _TAU); ctx.stroke();
    ctx.shadowBlur = 0; ctx.fillStyle = "rgba(10,12,18,0.9)";
    ctx.beginPath(); ctx.arc(p[0], p[1], r - 2.5, 0, _TAU); ctx.fill(); ctx.restore();
}
const _ANIM_OVERLAYS = {
    "tron": (ctx, a, b, col, now) => {           // red & blue identity discs fly, clash white
        const ph = (now / 1400) % 1;
        const pr = _apt(ph * 0.5, a, b), pb = _apt(1 - ph * 0.5, a, b);
        _aDisc(ctx, pr, 6, "#ff3b30"); _aDisc(ctx, pb, 6, "#4da6ff");
        if (ph > 0.9) { const m = _apt(0.5, a, b); _aRing(ctx, m, 6 + (ph - 0.9) * 90, "#ffffff", 2.5, 14); }
    },
    "matrix": (ctx, a, b, col, now) => {         // glyphs rain OFF the wire
        for (let i = 0; i < 5; i++) {
            const t = (i * 0.19 + 0.08), p = _apt(t, a, b);
            const drop = ((now / 700) + i * 0.37) % 1;
            _aGlyph(ctx, [p[0], p[1] + drop * 26], "01"[i % 2], 9, "rgba(57,255,20," + (1 - drop) + ")");
        }
    },
    "pacman": (ctx, a, b, col, now) => {         // ghosts chase the pac
        const ph = (now / 1600) % 1;
        for (const gc of [[0.10, "#ff4d4d"], [0.18, "#4dc3ff"]]) {
            const p = _apt(ph - gc[0], a, b);
            ctx.save(); ctx.fillStyle = gc[1]; ctx.beginPath();
            ctx.arc(p[0], p[1] - 1, 5, Math.PI, 0);
            ctx.lineTo(p[0] + 5, p[1] + 4); ctx.lineTo(p[0] - 5, p[1] + 4); ctx.closePath(); ctx.fill();
            ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(p[0] - 2, p[1] - 2, 1.4, 0, _TAU); ctx.arc(p[0] + 2, p[1] - 2, 1.4, 0, _TAU); ctx.fill();
            ctx.restore();
        }
    },
    "sonic": (ctx, a, b, col, now) => {          // gold rings pop + spin along the run
        for (let i = 0; i < 3; i++) {
            const t = ((now / 1100) + i * 0.33) % 1, p = _apt(t, a, b);
            const sx = Math.abs(Math.cos(now / 180 + i));
            ctx.save(); ctx.translate(p[0], p[1]); ctx.scale(Math.max(0.15, sx), 1);
            _aRing(ctx, [0, 0], 5.5, "#ffd54a", 2.5, 8); ctx.restore();
        }
    },
    "portal": (ctx, a, b, col, now) => {         // dot enters orange portal, exits blue
        const pA = _apt(0.22, a, b), pB = _apt(0.78, a, b);
        const ell = (p, c) => { ctx.save(); ctx.translate(p[0], p[1]); ctx.scale(0.45, 1);
            _aRing(ctx, [0, 0], 9, c, 3, 10); ctx.restore(); };
        ell(pA, "#ff9f1c"); ell(pB, "#4da6ff");
        const ph = (now / 1300) % 1;
        if (ph < 0.4) _aDot(ctx, _apt(ph * 0.55, a, b), 3.5, "#fff", 6);
        else if (ph > 0.6) _aDot(ctx, _apt(0.78 + (ph - 0.6) * 0.55, a, b), 3.5, "#fff", 6);
    },
    "lightning": (ctx, a, b, col, now) => {      // side-forks crackle off the bolt
        const seed = Math.floor(now / 160);
        for (let i = 0; i < 2; i++) {
            const t = (((seed * 37 + i * 53) % 89) / 89), p = _apt(t, a, b);
            const ang = ((seed * 61 + i * 97) % 360) / 180 * Math.PI, L = 10 + ((seed + i) % 3) * 5;
            ctx.save(); ctx.strokeStyle = "#e8f4ff"; ctx.lineWidth = 1.2; ctx.shadowColor = "#7cc4ff"; ctx.shadowBlur = 8;
            ctx.beginPath(); ctx.moveTo(p[0], p[1]);
            ctx.lineTo(p[0] + Math.cos(ang) * L * 0.6, p[1] + Math.sin(ang) * L * 0.6);
            ctx.lineTo(p[0] + Math.cos(ang + 0.5) * L, p[1] + Math.sin(ang + 0.5) * L); ctx.stroke(); ctx.restore();
        }
    },
    "heartbeat": (ctx, a, b, col, now) => {      // an ECG blip sweeps the line
        const t = (now / 1500) % 1, p = _apt(t, a, b);
        ctx.save(); ctx.strokeStyle = "#ff4d6d"; ctx.lineWidth = 2; ctx.shadowColor = "#ff4d6d"; ctx.shadowBlur = 8;
        ctx.beginPath(); ctx.moveTo(p[0] - 10, p[1]);
        ctx.lineTo(p[0] - 4, p[1]); ctx.lineTo(p[0] - 1, p[1] - 9); ctx.lineTo(p[0] + 2, p[1] + 7); ctx.lineTo(p[0] + 5, p[1]); ctx.lineTo(p[0] + 11, p[1]);
        ctx.stroke(); ctx.restore();
    },
    "dna-helix": (ctx, a, b, col, now) => {      // base-pair rungs light up in sequence
        const idx = Math.floor(now / 260) % 5;
        const p = _apt(0.14 + idx * 0.18, a, b);
        _aRing(ctx, p, 6, "#7cf5d4", 1.8, 9);
    },
    "rainbow-flow": (ctx, a, b, col, now) => {   // white comet rides the rainbow
        const t = (now / 1000) % 1, p = _apt(t, a, b);
        _aDot(ctx, p, 3.2, "#ffffff", 10);
        _aDot(ctx, _apt(t - 0.04, a, b), 2.2, "rgba(255,255,255,0.5)", 0);
    },
    "dashed-march": (ctx, a, b, col, now) => {   // counter-marching ticks both ways
        const t1 = (now / 900) % 1, t2 = 1 - ((now / 1300) % 1);
        _aGlyph(ctx, _apt(t1, a, b), ">", 11, col);
        _aGlyph(ctx, _apt(t2, a, b), "<", 11, "rgba(200,210,235,0.8)");
    },
    "pulse-packet": (ctx, a, b, col, now) => {   // packet drops expanding data ripples
        const t = (now / 1200) % 1, p = _apt(t, a, b);
        const rip = (now / 400) % 1;
        _aRing(ctx, p, 4 + rip * 10, "rgba(120,200,255," + (1 - rip) + ")", 1.5, 0);
    },
    "rgb-spectrum": (ctx, a, b, col, now) => {   // R/G/B convoy in file
        const base = (now / 1100) % 1;
        for (const co of [["#ff3b30", 0], ["#34c759", 0.06], ["#4da6ff", 0.12]])
            _aDot(ctx, _apt(base - co[1], a, b), 2.6, co[0], 7);
    },
    "fire": (ctx, a, b, col, now) => {           // embers rise off the flame
        for (let i = 0; i < 4; i++) {
            const t = (i * 0.23 + 0.1), p = _apt(t, a, b);
            const rise = ((now / 800) + i * 0.41) % 1;
            _aDot(ctx, [p[0] + Math.sin(rise * 9 + i) * 3, p[1] - rise * 20], 1.8, "rgba(255," + (140 + i * 20) + ",40," + (1 - rise) + ")", 0);
        }
    },
    "comet": (ctx, a, b, col, now) => {          // meteor + spark shards
        const t = (now / 1000) % 1, p = _apt(t, a, b);
        _aDot(ctx, p, 3.5, "#fff8e1", 12);
        for (let i = 1; i <= 3; i++) _aDot(ctx, _apt(t - i * 0.03, a, b), 2.5 - i * 0.6, "rgba(255,220,150," + (0.8 - i * 0.22) + ")", 0);
    },
    "candy": (ctx, a, b, col, now) => {          // pastel gumballs bobbing
        const cols = ["#ffb3d9", "#b3e5ff", "#fff3b0"];
        for (let i = 0; i < 3; i++) {
            const t = ((now / 1600) + i * 0.33) % 1, p = _apt(t, a, b);
            _aDot(ctx, [p[0], p[1] + Math.sin(now / 200 + i * 2) * 3], 3, cols[i], 5);
        }
    },
    "aurora": (ctx, a, b, col, now) => {         // soft curtains shimmer above
        for (let i = 0; i < 3; i++) {
            const t = 0.2 + i * 0.3, p = _apt(t, a, b);
            const h = 8 + Math.sin(now / 500 + i * 1.7) * 5;
            ctx.save(); ctx.strokeStyle = "rgba(" + (120 + i * 40) + ",255," + (200 - i * 30) + ",0.35)"; ctx.lineWidth = 4;
            ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(p[0], p[1] - h); ctx.stroke(); ctx.restore();
        }
    },
    "hyperspace": (ctx, a, b, col, now) => {     // star-streaks overtake at warp
        for (let i = 0; i < 3; i++) {
            const t = ((now / 500) + i * 0.33) % 1;
            const p = _apt(t, a, b), q = _apt(t - 0.06, a, b);
            ctx.save(); ctx.strokeStyle = "rgba(220,235,255," + (0.9 - i * 0.25) + ")"; ctx.lineWidth = 1.6;
            ctx.beginPath(); ctx.moveTo(q[0], q[1]); ctx.lineTo(p[0], p[1]); ctx.stroke(); ctx.restore();
        }
    },
    "upside-down": (ctx, a, b, col, now) => {    // spores drift down, flashlight flickers
        for (let i = 0; i < 4; i++) {
            const t = i * 0.23 + 0.1, p = _apt(t, a, b);
            const dr = ((now / 1400) + i * 0.31) % 1;
            _aDot(ctx, [p[0] + Math.sin(dr * 5 + i) * 4, p[1] + dr * 18], 1.5, "rgba(220,200,255," + (0.7 - dr * 0.6) + ")", 0);
        }
        if (Math.floor(now / 90) % 7 === 0) _aDot(ctx, _apt(0.5, a, b), 4, "rgba(255,240,200,0.85)", 12);
    },
    "ki-blast": (ctx, a, b, col, now) => {       // charge at source, then RELEASE
        const ph = (now / 1600) % 1;
        if (ph < 0.45) _aRing(ctx, _apt(0.05, a, b), 3 + ph * 16, "rgba(120,220,255," + (0.9 - ph) + ")", 2, 10);
        else _aDot(ctx, _apt((ph - 0.45) / 0.55, a, b), 4.5, "#9bf6ff", 14);
    },
    "rainbow-road": (ctx, a, b, col, now) => {   // star sparkle convoy
        for (let i = 0; i < 2; i++) {
            const t = ((now / 1400) + i * 0.5) % 1, p = _apt(t, a, b);
            _aGlyph(ctx, p, "*", 12, ["#fff34d", "#ffffff"][i], now / 300 + i);
        }
    },
    "pokemon": (ctx, a, b, col, now) => {        // pokeball rolls the wire
        const t = (now / 1500) % 1, p = _apt(t, a, b), rot = now / 150;
        ctx.save(); ctx.translate(p[0], p[1]); ctx.rotate(rot);
        ctx.fillStyle = "#ff3b30"; ctx.beginPath(); ctx.arc(0, 0, 5, Math.PI, 0); ctx.fill();
        ctx.fillStyle = "#f5f5f5"; ctx.beginPath(); ctx.arc(0, 0, 5, 0, Math.PI); ctx.fill();
        ctx.strokeStyle = "#1a1a22"; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(-5, 0); ctx.lineTo(5, 0); ctx.stroke();
        ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(0, 0, 1.6, 0, _TAU); ctx.fill(); ctx.stroke();
        ctx.restore();
    },
    "zelda": (ctx, a, b, col, now) => {          // triforce glints
        const t = (now / 1800) % 1, p = _apt(t, a, b);
        const tri = (x, y, r) => { ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x - r, y + r); ctx.lineTo(x + r, y + r); ctx.closePath(); ctx.fill(); };
        ctx.save(); ctx.fillStyle = "#ffd54a"; ctx.shadowColor = "#ffd54a"; ctx.shadowBlur = 8;
        tri(p[0], p[1] - 3, 3); tri(p[0] - 3, p[1] + 3, 3); tri(p[0] + 3, p[1] + 3, 3); ctx.restore();
    },
    "halo": (ctx, a, b, col, now) => {           // plasma bolt with heat trail
        const t = (now / 800) % 1, p = _apt(t, a, b);
        _aDot(ctx, p, 3.4, "#7cf5ff", 12);
        _aDot(ctx, _apt(t - 0.05, a, b), 2.2, "rgba(124,245,255,0.4)", 0);
    },
    "among-us": (ctx, a, b, col, now) => {       // impostor chases crew bean
        const ph = (now / 1700) % 1;
        const bean = (p, c) => { ctx.save(); ctx.fillStyle = c;
            ctx.beginPath(); ctx.ellipse(p[0], p[1], 4, 5, 0, 0, _TAU); ctx.fill();
            ctx.fillStyle = "#aee3f2"; ctx.beginPath(); ctx.ellipse(p[0] + 1.5, p[1] - 1.5, 2.2, 1.4, 0, 0, _TAU); ctx.fill(); ctx.restore(); };
        bean(_apt(ph, a, b), "#4dc3ff"); bean(_apt(ph - 0.12, a, b), "#ff4d4d");
    },
    "avatar-bending": (ctx, a, b, col, now) => { // four elements orbit the flow
        const t = (now / 1600) % 1, p = _apt(t, a, b);
        const cols = ["#7cc4ff", "#ff9f1c", "#a0f0a0", "#e8e4d8"];
        for (let i = 0; i < 4; i++) {
            const ang = now / 400 + i * Math.PI / 2;
            _aDot(ctx, [p[0] + Math.cos(ang) * 8, p[1] + Math.sin(ang) * 8], 2, cols[i], 5);
        }
    },
    "john-wick": (ctx, a, b, col, now) => {      // tracer round + muzzle flash
        const ph = (now / 700) % 1;
        if (ph < 0.12) _aDot(ctx, _apt(0.02, a, b), 4, "rgba(255,240,180,0.9)", 12);
        const p = _apt(ph, a, b), q = _apt(ph - 0.04, a, b);
        ctx.save(); ctx.strokeStyle = "#ffe9b0"; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(q[0], q[1]); ctx.lineTo(p[0], p[1]); ctx.stroke(); ctx.restore();
    },
    "ghostbusters": (ctx, a, b, col, now) => {   // proton stream wiggle
        ctx.save(); ctx.strokeStyle = "rgba(255,159,28,0.85)"; ctx.lineWidth = 1.6;
        ctx.shadowColor = "#ff9f1c"; ctx.shadowBlur = 6;
        ctx.beginPath();
        for (let i = 0; i <= 16; i++) {
            const t = i / 16, p = _apt(t, a, b);
            const off = Math.sin(t * 22 + now / 90) * 4 * Math.sin(t * Math.PI);
            if (i === 0) ctx.moveTo(p[0], p[1] + off); else ctx.lineTo(p[0], p[1] + off);
        }
        ctx.stroke(); ctx.restore();
    },
    "jurassic": (ctx, a, b, col, now) => {       // footsteps stamp along, tremor
        const stepIdx = Math.floor(now / 500) % 4;
        for (let i = 0; i <= stepIdx; i++) {
            const p = _apt(0.15 + i * 0.22, a, b);
            _aGlyph(ctx, [p[0], p[1] + (i % 2 ? 5 : -5)], "●", 6, "rgba(160,150,120,0.7)");
        }
    },
    "gta": (ctx, a, b, col, now) => {            // pursuit lights strobe the wire
        const t = (now / 1200) % 1, p = _apt(t, a, b);
        const red = Math.floor(now / 180) % 2 === 0;
        _aDot(ctx, [p[0] - 4, p[1]], 3, red ? "#ff3b30" : "rgba(255,59,48,0.25)", red ? 10 : 0);
        _aDot(ctx, [p[0] + 4, p[1]], 3, red ? "rgba(77,166,255,0.25)" : "#4da6ff", red ? 0 : 10);
    },
    "space-invaders": (ctx, a, b, col, now) => { // invader steps down, zap shoots up
        const step = Math.floor(now / 400) % 8;
        const p = _apt(0.12 + step * 0.1, a, b);
        _aGlyph(ctx, [p[0], p[1] - 6], "▓", 8, "#a6e3a1");
        const zt = (now / 600) % 1, zp = _apt(1 - zt * 0.9, a, b);
        ctx.save(); ctx.strokeStyle = "#e8f4ff"; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(zp[0], zp[1] + 6); ctx.lineTo(zp[0], zp[1] - 2); ctx.stroke(); ctx.restore();
    },
    "tetris": (ctx, a, b, col, now) => {         // tetromino cells tumble along
        const cols = ["#4dd2ff", "#ffd24d", "#c77dff", "#8cff66"];
        for (let i = 0; i < 3; i++) {
            const t = ((now / 1800) + i * 0.33) % 1, p = _apt(t, a, b);
            ctx.save(); ctx.translate(p[0], p[1]); ctx.rotate((now / 500 + i) % _TAU);
            ctx.fillStyle = cols[(i + Math.floor(now / 900)) % 4];
            ctx.fillRect(-3, -3, 6, 6);
            ctx.strokeStyle = "rgba(0,0,0,0.5)"; ctx.lineWidth = 1; ctx.strokeRect(-3, -3, 6, 6);
            ctx.restore();
        }
    },
    "street-fighter": (ctx, a, b, col, now) => { // hadouken orb, burst on arrival
        const ph = (now / 1100) % 1;
        const p = _apt(ph, a, b);
        _aDot(ctx, p, 4.5, "#7cc4ff", 12);
        _aDot(ctx, _apt(ph - 0.05, a, b), 3, "rgba(124,196,255,0.45)", 0);
        if (ph > 0.93) _aRing(ctx, _apt(1, a, b), (ph - 0.93) * 120, "rgba(180,220,255,0.8)", 2, 10);
    },
    "mega-man": (ctx, a, b, col, now) => {       // triple buster pellets
        const base = (now / 900) % 1;
        for (let i = 0; i < 3; i++) {
            const p = _apt(base - i * 0.07, a, b);
            _aDot(ctx, p, 2.6 - i * 0.5, "#9bf6ff", 8);
        }
    },
    "mortal-kombat": (ctx, a, b, col, now) => {  // dragon-fire pulse sweeps + flash
        const ph = (now / 1400) % 1, p = _apt(ph, a, b);
        _aDot(ctx, p, 3.5, "#ffd24d", 10);
        if (ph > 0.9) _aGlyph(ctx, _apt(0.5, a, b), "✦", 14, "rgba(255,210,77," + ((ph - 0.9) * 10) + ")");
    },
    "metroid": (ctx, a, b, col, now) => {        // morph ball rolls with band highlight
        const t = (now / 1600) % 1, p = _apt(t, a, b), rot = now / 130;
        ctx.save(); ctx.translate(p[0], p[1]); ctx.rotate(rot);
        ctx.fillStyle = "#ff9f1c"; ctx.beginPath(); ctx.arc(0, 0, 4.5, 0, _TAU); ctx.fill();
        ctx.strokeStyle = "#7a3c00"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(-4.5, 0); ctx.lineTo(4.5, 0); ctx.stroke();
        ctx.restore();
    },
    "doom": (ctx, a, b, col, now) => {           // fireball + smoke puffs
        const t = (now / 900) % 1, p = _apt(t, a, b);
        _aDot(ctx, p, 3.6, "#ff6b1c", 12);
        for (let i = 1; i <= 2; i++) {
            const q = _apt(t - i * 0.05, a, b);
            _aDot(ctx, [q[0], q[1] - i * 2], 2, "rgba(120,110,100," + (0.5 - i * 0.18) + ")", 0);
        }
    },
    "elden-ring": (ctx, a, b, col, now) => {     // grace wisp drifts, slow flicker
        const t = (now / 2600) % 1, p = _apt(t, a, b);
        const fl = 0.6 + 0.4 * Math.sin(now / 220);
        _aDot(ctx, [p[0], p[1] + Math.sin(now / 380) * 3], 3, "rgba(255,225,150," + fl + ")", 14);
    },
    "galaga": (ctx, a, b, col, now) => {         // alien zigzags, ship missile intercepts
        const t = (now / 1500) % 1;
        const p = _apt(t, a, b);
        _aGlyph(ctx, [p[0] + Math.sin(now / 160) * 5, p[1]], "¤", 9, "#ff77c8");
        const mt = (now / 700) % 1, m = _apt(1 - mt, a, b);
        _aDot(ctx, m, 1.8, "#e8f4ff", 5);
    },
};

const _ANIMATED = new Set([
    "dna-helix", "rainbow-flow", "dashed-march", "lightning", "pulse-packet",
    "rgb-spectrum", "fire", "comet", "candy", "aurora", "heartbeat",
    "matrix", "tron", "portal", "pacman", "hyperspace", "upside-down", "ki-blast", "rainbow-road",
    "pokemon", "sonic", "zelda", "halo", "among-us", "avatar-bending", "john-wick", "ghostbusters", "jurassic",
    "gta", "space-invaders", "tetris", "street-fighter", "mega-man", "mortal-kombat", "metroid", "doom", "elden-ring", "galaga",
]);

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

// ── Pipe SHAPE (separate setting from the visual skin — any skin rides any
// shape). "auto" = dead-straight when the nodes are roughly aligned, gentle
// spline when offset (the Nuke read). angular/diag45 are piecewise polylines:
// skins that SAMPLE via _bezierAt follow them exactly; skins that stroke raw
// bezierCurveTo(cp1,cp2) render the rounded approximation of the same elbow.
const SHAPE_SETTING_ID = "mec.noodle.shape";
const SHAPES = ["auto", "spline", "straight", "angular", "rounded", "diag45", "arc", "s-curve"];
let _shapeCache = "auto";
function _refreshShapeCache() {
    try {
        const v = app.ui.settings.getSettingValue(SHAPE_SETTING_ID);
        _shapeCache = (v === undefined || v === null) ? "auto" : v;
    } catch (_) { _shapeCache = "auto"; }
}
function _currentShape() { return _shapeCache; }
const _lerp2 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
function _alignedStraight(a, b) {
    const dx = Math.abs(b[0] - a[0]), dy = Math.abs(b[1] - a[1]);
    return (dx >= dy ? dy : dx) <= 40;     // near-horizontal OR near-vertical
}
// Elbow/diag waypoints for the piecewise shapes.
function _shapeWaypoints(a, b, shape) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const horiz = Math.abs(dx) >= Math.abs(dy);
    if (shape === "angular") {
        return horiz
            ? [a, [a[0] + dx / 2, a[1]], [a[0] + dx / 2, b[1]], b]
            : [a, [a[0], a[1] + dy / 2], [b[0], a[1] + dy / 2], b];
    }
    // diag45 (Houdini): 45° run first, then straight to the target.
    const sx = Math.sign(dx) || 1, sy = Math.sign(dy) || 1;
    if (horiz) { const e = [a[0] + sx * Math.abs(dy), b[1]]; return [a, e, b]; }
    const e = [b[0], a[1] + sy * Math.abs(dx)]; return [a, e, b];
}
function _polyAt(t, pts) {
    let total = 0; const seg = [];
    for (let i = 0; i < pts.length - 1; i++) {
        const l = Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]);
        seg.push(l); total += l;
    }
    if (total <= 0) return pts[0].slice();
    let d = t * total;
    for (let i = 0; i < seg.length; i++) {
        if (d <= seg[i] || i === seg.length - 1) {
            return _lerp2(pts[i], pts[i + 1], seg[i] ? d / seg[i] : 0);
        }
        d -= seg[i];
    }
    return pts[pts.length - 1].slice();
}

function _bezierPoints(a, b) {
    const shape = _currentShape();
    if (shape === "straight" || (shape === "auto" && _alignedStraight(a, b))) {
        return [_lerp2(a, b, 1 / 3), _lerp2(a, b, 2 / 3)];
    }
    if (shape === "rounded" || shape === "angular" || shape === "diag45") {
        // Elbow tangents (angular/diag45 sampled exactly in _bezierAt; raw
        // bezier consumers get the rounded approximation of the same elbow).
        const w = _shapeWaypoints(a, b, shape === "diag45" ? "diag45" : "angular");
        return [w[1].slice(), w[w.length - 2].slice()];
    }
    if (shape === "arc") {
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
        const n = [-(b[1] - a[1]) / len, (b[0] - a[0]) / len];
        const off = Math.min(len * 0.28, 120);
        const c1 = _lerp2(a, b, 0.25), c2 = _lerp2(a, b, 0.75);
        return [[c1[0] + n[0] * off, c1[1] + n[1] * off], [c2[0] + n[0] * off, c2[1] + n[1] * off]];
    }
    if (shape === "s-curve") {
        const dx = b[0] - a[0], dy = b[1] - a[1];
        if (Math.abs(dx) >= Math.abs(dy)) {
            const d = Math.max(30, Math.abs(dx) * 0.9) * (dx >= 0 ? 1 : -1);
            return [[a[0] + d, a[1]], [b[0] - d, b[1]]];
        }
        const d = Math.max(30, Math.abs(dy) * 0.9) * (dy >= 0 ? 1 : -1);
        return [[a[0], a[1] + d], [b[0], b[1] - d]];
    }
    // spline (and auto when offset) — the classic geometry-aware tangents.
    // Geometry-aware control points. The classic form (cp1 = a+right, cp2 =
    // b-left) only reads well for normal left→right flow; when the endpoints are
    // vertically stacked or the target sits to the LEFT (e.g. with floating
    // ports sliding the exit to a node's bottom/right edge), forcing horizontal
    // tangents makes the curve loop back on itself. So extend the tangents along
    // whichever axis actually dominates, sign following the direction to the
    // other endpoint. Normal forward (a left of b) wires are unchanged.
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const adx = Math.abs(dx), ady = Math.abs(dy);
    if (adx >= ady) {
        const d = Math.max(20, adx * 0.5) * (dx >= 0 ? 1 : -1);
        return [[a[0] + d, a[1]], [b[0] - d, b[1]]];
    }
    const d = Math.max(20, ady * 0.5) * (dy >= 0 ? 1 : -1);
    return [[a[0], a[1] + d], [b[0], b[1] - d]];
}
function _bezierAt(t, a, cp1, cp2, b) {
    const _sh = _currentShape();
    if (_sh === "angular" || _sh === "diag45") {
        return _polyAt(t, _shapeWaypoints(a, b, _sh));
    }
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

// ── NEW unique noodle designs (2026-06-30) ──────────────────────────────────
// Full-RGB spectrum flowing along the wire — "millions of colours". Each segment
// gets its own hue; the whole band scrolls so it reads as living RGB.
function _renderRgbSpectrum(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 48, off = (performance.now() / 1000) % 1;
    ctx.save(); ctx.lineCap = "round"; ctx.lineWidth = 3;
    let prev = _bezierAt(0, a, cp1, cp2, b);
    for (let i = 1; i <= N; i++) {
        const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
        ctx.strokeStyle = `hsl(${((t + off) * 360) % 360}, 100%, 56%)`;
        ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
        prev = p;
    }
    ctx.restore();
}
// Smooth two-stop gradient from the OUTPUT type colour to the INPUT type colour,
// with a soft glow — clean and "pro" (think Nuke pipe colour-coding).
function _renderGradient(ctx, a, b, color, linkType) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const grad = ctx.createLinearGradient(a[0], a[1], b[0], b[1]);
    grad.addColorStop(0, color);
    grad.addColorStop(1, _typeColor(linkType));
    ctx.save(); ctx.lineCap = "round"; ctx.lineWidth = 3.5; ctx.strokeStyle = grad;
    ctx.shadowBlur = 6; ctx.shadowColor = color;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]);
    ctx.bezierCurveTo(cp1[0], cp1[1], cp2[0], cp2[1], b[0], b[1]); ctx.stroke();
    ctx.restore();
}
// Flame: red→orange→yellow gradient that flickers + tapers, with ember glow.
function _renderFire(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 40, time = performance.now() / 120;
    ctx.save(); ctx.lineCap = "round"; ctx.shadowBlur = 10; ctx.shadowColor = "#ff6a00";
    let prev = _bezierAt(0, a, cp1, cp2, b);
    for (let i = 1; i <= N; i++) {
        const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
        const flick = 0.5 + 0.5 * Math.sin(time + i * 0.6);
        ctx.strokeStyle = `hsl(${8 + t * 48 + flick * 8}, 100%, ${50 + flick * 10}%)`;
        ctx.lineWidth = 4 - t * 1.6;
        ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
        prev = p;
    }
    ctx.restore();
}
// Comet: a glowing head races along a faint base line, leaving a fading trail.
function _renderComet(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    ctx.strokeStyle = color; ctx.globalAlpha = 0.22; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]);
    ctx.bezierCurveTo(cp1[0], cp1[1], cp2[0], cp2[1], b[0], b[1]); ctx.stroke();
    const head = (performance.now() / 900) % 1, TRAIL = 14;
    for (let i = 0; i < TRAIL; i++) {
        const t = head - i * 0.025; if (t < 0) continue;
        const p = _bezierAt(t, a, cp1, cp2, b), k = 1 - i / TRAIL;
        ctx.globalAlpha = k; ctx.fillStyle = color; ctx.shadowBlur = 12; ctx.shadowColor = color;
        ctx.beginPath(); ctx.arc(p[0], p[1], 3.6 * (1 - i / TRAIL * 0.7), 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
}
// Candy cane: white rope with marching coloured diagonal stripes.
function _renderCandy(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const off = (performance.now() / 60) % 16;
    ctx.save(); ctx.lineCap = "butt"; ctx.lineWidth = 4.5;
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath(); ctx.moveTo(a[0], a[1]);
    ctx.bezierCurveTo(cp1[0], cp1[1], cp2[0], cp2[1], b[0], b[1]); ctx.stroke();
    ctx.strokeStyle = color; ctx.lineWidth = 4.5; ctx.setLineDash([8, 8]); ctx.lineDashOffset = -off;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]);
    ctx.bezierCurveTo(cp1[0], cp1[1], cp2[0], cp2[1], b[0], b[1]); ctx.stroke();
    ctx.setLineDash([]); ctx.restore();
}
// Aurora: soft green↔blue↔purple ribbon that breathes along the wire.
function _renderAurora(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 40, t0 = (performance.now() / 1400) % 1;
    ctx.save(); ctx.lineCap = "round"; ctx.lineWidth = 3.5; ctx.shadowBlur = 9;
    let prev = _bezierAt(0, a, cp1, cp2, b);
    for (let i = 1; i <= N; i++) {
        const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
        const col = `hsl(${180 + 70 * Math.sin((t + t0) * Math.PI * 2)}, 85%, 62%)`;
        ctx.strokeStyle = col; ctx.shadowColor = col;
        ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
        prev = p;
    }
    ctx.restore();
}
// Heartbeat: an ECG spike races along the wire on a glowing baseline.
function _renderHeartbeat(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 60, beat = (performance.now() / 700) % 1;
    ctx.save(); ctx.lineCap = "round"; ctx.lineWidth = 2;
    ctx.strokeStyle = color; ctx.shadowBlur = 6; ctx.shadowColor = color;
    ctx.beginPath();
    let pnrm = [0, 1];
    for (let i = 0; i <= N; i++) {
        const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
        if (i > 0) { const q = _bezierAt((i - 1) / N, a, cp1, cp2, b); const dx = p[0] - q[0], dy = p[1] - q[1], L = Math.hypot(dx, dy) || 1; pnrm = [-dy / L, dx / L]; }
        const d = Math.abs(((t - beat + 1) % 1));
        const spike = d < 0.05 ? (d < 0.025 ? Math.sin(d / 0.025 * Math.PI) * 16 : -Math.sin((d - 0.025) / 0.025 * Math.PI) * 9) : 0;
        const x = p[0] + pnrm[0] * spike, y = p[1] + pnrm[1] * spike;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke(); ctx.restore();
}

// ─────────────────────── Franchise styles (2026-06-30) ───────────────────────
// All Canvas2D-literal colours (no var()). Lean per-link work; animated ones
// piggy-back the existing 20fps throttle. Each is instantly recognisable.

// Helper: draw the bezier trunk once.
function _trunk(ctx, a, cp1, cp2, b, color, w, glow) {
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.bezierCurveTo(cp1[0], cp1[1], cp2[0], cp2[1], b[0], b[1]);
    if (glow) { ctx.shadowBlur = glow; ctx.shadowColor = color; }
    ctx.lineWidth = w; ctx.strokeStyle = color; ctx.stroke();
    ctx.shadowBlur = 0;
}

// THE MATRIX — green digital rain cascading output→input.
function _renderMatrix(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#0c3", 1.5, 0);
    const N = 16, head = (performance.now() / 650) % 1;
    for (let i = 0; i < N; i++) {
        const t = (i / N + head) % 1;
        const p = _bezierAt(t, a, cp1, cp2, b);
        const lead = i === N - 1;
        ctx.fillStyle = lead ? "#dcffe0" : "#39ff5a";
        ctx.globalAlpha = lead ? 1 : (0.25 + 0.55 * (i / N));
        ctx.shadowBlur = lead ? 8 : 3; ctx.shadowColor = "#39ff5a";
        ctx.fillRect(p[0] - 1.5, p[1] - 2.5, 3, 5);
    }
    ctx.restore();
}

// TRON — neon cyan light-trace with a racing light cycle + glow.
function _renderTron(ctx, a, b, color) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const cyan = "#7df9ff", orange = "#ffae42";
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#0a2a30", 5, 0);          // dark conduit
    _trunk(ctx, a, cp1, cp2, b, cyan, 1.6, 10);             // bright trace
    const t = (performance.now() / 900) % 1;
    const p = _bezierAt(t, a, cp1, cp2, b);
    ctx.shadowBlur = 14; ctx.shadowColor = orange;
    ctx.fillStyle = orange;
    ctx.fillRect(p[0] - 3, p[1] - 1.5, 6, 3);               // the light cycle
    ctx.restore();
}

// PORTAL — blue exit, orange entry, with a shimmering energy core.
function _renderPortal(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 40;
    ctx.save(); ctx.lineCap = "round"; ctx.lineWidth = 3;
    let prev = a;
    for (let i = 1; i <= N; i++) {
        const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
        // orange (entry, near a) → blue (exit, near b)
        const r = Math.round(0x05 + (0xff - 0x05) * (1 - t));
        const g = Math.round(0x99 + (0x5e - 0x99) * Math.abs(0.5 - t) * 0);
        const bl = Math.round(0xff * t + 0x42 * (1 - t));
        ctx.strokeStyle = `rgb(${r},${110 + Math.round(60 * Math.sin(t * 6 + performance.now() / 200))},${bl})`;
        ctx.shadowBlur = 6; ctx.shadowColor = t > 0.5 ? "#3b8dff" : "#ff8a1e";
        ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
        prev = p;
    }
    ctx.restore();
}

// PAC-MAN — yellow chomper eats a trail of pellets along the wire.
function _renderPacman(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save();
    // pellet trail
    const N = 12;
    ctx.fillStyle = "#ffd54a"; ctx.globalAlpha = 0.8;
    for (let i = 1; i < N; i++) {
        const p = _bezierAt(i / N, a, cp1, cp2, b);
        ctx.beginPath(); ctx.arc(p[0], p[1], 1.6, 0, Math.PI * 2); ctx.fill();
    }
    // pac-man travelling, mouth chomping
    const t = (performance.now() / 1400) % 1;
    const p = _bezierAt(t, a, cp1, cp2, b);
    const q = _bezierAt(Math.min(1, t + 0.01), a, cp1, cp2, b);
    const ang = Math.atan2(q[1] - p[1], q[0] - p[0]);
    const mouth = (Math.abs(Math.sin(performance.now() / 120)) * 0.5 + 0.05) * Math.PI;
    ctx.globalAlpha = 1; ctx.fillStyle = "#ffe14d";
    ctx.shadowBlur = 8; ctx.shadowColor = "#ffe14d";
    ctx.beginPath();
    ctx.moveTo(p[0], p[1]);
    ctx.arc(p[0], p[1], 6, ang + mouth, ang - mouth + Math.PI * 2);
    ctx.closePath(); ctx.fill();
    ctx.restore();
}

// HYPERSPACE — Star Wars jump: blue-white streaks racing to the target.
function _renderHyperspace(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#0b1b3a", 4, 0);
    const streaks = 5, base = performance.now() / 350;
    for (let s = 0; s < streaks; s++) {
        const head = ((base + s / streaks) % 1);
        const tail = Math.max(0, head - 0.18);
        const ph = _bezierAt(head, a, cp1, cp2, b);
        const pt = _bezierAt(tail, a, cp1, cp2, b);
        const grd = ctx.createLinearGradient(pt[0], pt[1], ph[0], ph[1]);
        grd.addColorStop(0, "rgba(120,170,255,0)");
        grd.addColorStop(1, "#eaf2ff");
        ctx.strokeStyle = grd; ctx.lineWidth = 2; ctx.shadowBlur = 8; ctx.shadowColor = "#9ec5ff";
        ctx.beginPath(); ctx.moveTo(pt[0], pt[1]); ctx.lineTo(ph[0], ph[1]); ctx.stroke();
    }
    ctx.restore();
}

// THE UPSIDE DOWN — Stranger Things: dark-red tendril, drifting spores, red glow.
function _renderUpsideDown(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#7a0d12", 2.2, 9);
    _trunk(ctx, a, cp1, cp2, b, "#ff3b3b", 0.8, 6);
    const N = 9, now = performance.now();
    for (let i = 0; i < N; i++) {
        const t = (i + 0.5) / N;
        const p = _bezierAt(t, a, cp1, cp2, b);
        const fx = Math.sin(now / 700 + i * 1.7) * 6;
        const fy = Math.cos(now / 900 + i * 2.1) * 5;
        ctx.fillStyle = "#ffd0d0"; ctx.globalAlpha = 0.45 + 0.4 * Math.abs(Math.sin(now / 600 + i));
        ctx.shadowBlur = 5; ctx.shadowColor = "#ff5a5a";
        ctx.beginPath(); ctx.arc(p[0] + fx, p[1] + fy, 1.3, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
}

// KI BLAST — Dragon Ball: thick golden aura that flickers/charges.
function _renderKi(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const flick = 0.7 + 0.3 * Math.abs(Math.sin(performance.now() / 90));
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#ffcc33", 6 * flick, 18);   // outer aura
    _trunk(ctx, a, cp1, cp2, b, "#fff6c2", 2, 8);            // hot core
    // sparks
    const N = 6, now = performance.now();
    for (let i = 0; i < N; i++) {
        const t = ((i / N) + (now / 1000)) % 1;
        const p = _bezierAt(t, a, cp1, cp2, b);
        ctx.fillStyle = "#fff2a8"; ctx.globalAlpha = 0.8;
        ctx.shadowBlur = 10; ctx.shadowColor = "#ffd14d";
        ctx.beginPath(); ctx.arc(p[0], p[1], 1.4, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
}

// RAINBOW ROAD — Mario Kart: rainbow band with travelling star sparkles.
function _renderRainbowRoad(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 48, shift = performance.now() / 600;
    ctx.save(); ctx.lineCap = "round"; ctx.lineWidth = 4;
    let prev = a;
    for (let i = 1; i <= N; i++) {
        const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
        const hue = ((t * 360 + shift * 360) % 360);
        ctx.strokeStyle = `hsl(${hue},95%,62%)`;
        ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
        prev = p;
    }
    // travelling sparkle star
    const ts = (performance.now() / 900) % 1;
    const sp = _bezierAt(ts, a, cp1, cp2, b);
    ctx.fillStyle = "#ffffff"; ctx.shadowBlur = 10; ctx.shadowColor = "#fff";
    ctx.beginPath();
    for (let k = 0; k < 10; k++) {
        const ang = (Math.PI / 5) * k - Math.PI / 2;
        const rr = k % 2 ? 1.8 : 4.2;
        const x = sp[0] + Math.cos(ang) * rr, y = sp[1] + Math.sin(ang) * rr;
        k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.closePath(); ctx.fill();
    ctx.restore();
}

// ─────────────────── More franchise styles (2026-06-30 batch 2) ───────────────────
// All Canvas2D-literal colours. Lean per-link work; animated ones piggy-back the
// existing 20fps throttle. Each is an instantly-recognisable franchise.

// POKEMON — Pikachu yellow rope crackling with electric arcs (Thunderbolt).
function _renderPokemon(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#ffcb05", 3, 8);            // Pikachu yellow core
    // jagged electric arcs branching off the rope (re-jitter every 3rd frame)
    const N = 12, phase = Math.floor(performance.now() / 90);
    const rand = (i) => { const s = Math.sin(i * 91.7 + phase * 53.3) * 43758.5; return s - Math.floor(s); };
    ctx.strokeStyle = "#fff6b0"; ctx.lineWidth = 1.2; ctx.shadowBlur = 6; ctx.shadowColor = "#ffe14d";
    for (let i = 1; i < N; i += 2) {
        const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
        const q = _bezierAt((i - 1) / N, a, cp1, cp2, b);
        const dx = p[0] - q[0], dy = p[1] - q[1], L = Math.hypot(dx, dy) || 1;
        const nx = -dy / L, ny = dx / L, side = rand(i) > 0.5 ? 1 : -1;
        ctx.beginPath(); ctx.moveTo(p[0], p[1]);
        ctx.lineTo(p[0] + nx * 5 * side, p[1] + ny * 5 * side);
        ctx.lineTo(p[0] + nx * 9 * side - dx * 0.4, p[1] + ny * 9 * side - dy * 0.4);
        ctx.stroke();
    }
    ctx.restore();
}

// MINECRAFT — blocky, pixelated stair-stepped trail of cubes (no smooth curve).
function _renderMinecraft(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 26, S = 5;   // S = block grid size
    ctx.save();
    const COLS = ["#5d8a40", "#7caa4d", "#8b5a2b", "#6b6b6b"];  // grass/dirt/stone
    let last = null;
    for (let i = 0; i <= N; i++) {
        const p = _bezierAt(i / N, a, cp1, cp2, b);
        // snap to a block grid so the trail looks voxel/pixelated
        const gx = Math.round(p[0] / S) * S, gy = Math.round(p[1] / S) * S;
        if (last && gx === last[0] && gy === last[1]) continue;
        last = [gx, gy];
        ctx.fillStyle = COLS[i % COLS.length];
        ctx.fillRect(gx - S / 2, gy - S / 2, S, S);
        ctx.strokeStyle = "rgba(0,0,0,0.35)"; ctx.lineWidth = 0.6;
        ctx.strokeRect(gx - S / 2, gy - S / 2, S, S);
    }
    ctx.restore();
}

// SONIC — blue speed-blur streak with spinning gold rings travelling along it.
function _renderSonic(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#1a6dff", 4, 10);          // Sonic blue blur
    _trunk(ctx, a, cp1, cp2, b, "#bcd8ff", 1.2, 4);         // speed highlight
    const N = 4, head = (performance.now() / 700) % 1;
    for (let i = 0; i < N; i++) {
        const t = (i / N + head) % 1, p = _bezierAt(t, a, cp1, cp2, b);
        const wob = Math.abs(Math.cos(performance.now() / 100 + i));  // ring spin
        ctx.strokeStyle = "#ffd54a"; ctx.lineWidth = 1.6;
        ctx.shadowBlur = 6; ctx.shadowColor = "#ffd54a";
        ctx.beginPath(); ctx.ellipse(p[0], p[1], 4.5 * (0.25 + 0.75 * wob), 4.5, 0, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.restore();
}

// ZELDA — green Hero rope with a golden Triforce that sparkles at the midpoint.
function _renderZelda(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#2e7d32", 3, 6);           // Link green
    _trunk(ctx, a, cp1, cp2, b, "#7bd17f", 1, 3);
    // golden Triforce (3 small triangles) pulsing at the centre
    const m = _bezierAt(0.5, a, cp1, cp2, b);
    const pulse = 0.8 + 0.2 * Math.sin(performance.now() / 250);
    const R = 6 * pulse;
    ctx.fillStyle = "#ffd700"; ctx.shadowBlur = 10; ctx.shadowColor = "#ffe97a";
    const tri = (ox, oy) => {
        ctx.beginPath();
        ctx.moveTo(m[0] + ox, m[1] + oy - R);
        ctx.lineTo(m[0] + ox - R * 0.866, m[1] + oy + R * 0.5);
        ctx.lineTo(m[0] + ox + R * 0.866, m[1] + oy + R * 0.5);
        ctx.closePath(); ctx.fill();
    };
    tri(0, -R * 0.5); tri(-R * 0.866, R * 0.5); tri(R * 0.866, R * 0.5);
    ctx.restore();
}

// HALO — energy-sword cyan plasma: twin tapered blades + bright white core.
function _renderHalo(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const cyan = "#2ad4ff";
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, cyan, 6, 16);               // plasma glow
    _trunk(ctx, a, cp1, cp2, b, "#bff4ff", 3, 8);           // inner plasma
    _trunk(ctx, a, cp1, cp2, b, "#ffffff", 1.2, 0);         // white-hot core
    // shimmering energy flecks
    const N = 5, now = performance.now();
    for (let i = 0; i < N; i++) {
        const t = ((i / N) + (now / 1300)) % 1, p = _bezierAt(t, a, cp1, cp2, b);
        ctx.fillStyle = "#dffaff"; ctx.globalAlpha = 0.7 + 0.3 * Math.sin(now / 120 + i);
        ctx.shadowBlur = 8; ctx.shadowColor = cyan;
        ctx.beginPath(); ctx.arc(p[0], p[1], 1.3, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
}

// AVATAR-BENDING — half water (blue), half fire (orange): the rope transitions
// from cool bending energy to roaring flame along its length.
function _renderAvatarBending(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 40, time = performance.now() / 150;
    ctx.save(); ctx.lineCap = "round";
    let prev = _bezierAt(0, a, cp1, cp2, b);
    for (let i = 1; i <= N; i++) {
        const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
        // hue 200 (water blue) → 25 (fire orange); flicker on the fire half
        const flick = t > 0.5 ? 0.5 + 0.5 * Math.sin(time + i * 0.7) : 0;
        const hue = 200 - t * 175;
        ctx.strokeStyle = `hsl(${hue}, 95%, ${52 + flick * 12}%)`;
        ctx.lineWidth = 3.4; ctx.shadowBlur = 7;
        ctx.shadowColor = t > 0.5 ? "#ff7a1e" : "#3aa0ff";
        ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
        prev = p;
    }
    ctx.restore();
}

// JOHN-WICK — a sleek gold coin spins along a dark continental rope.
function _renderJohnWick(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#1c1c22", 3, 0);           // dark suit-black rope
    _trunk(ctx, a, cp1, cp2, b, "#c9a227", 0.8, 4);         // thin gold pinstripe
    // the gold coin, spinning (x-scale = |cos|) as it travels
    const t = (performance.now() / 1500) % 1, p = _bezierAt(t, a, cp1, cp2, b);
    const spin = Math.abs(Math.cos(performance.now() / 200));
    ctx.save(); ctx.translate(p[0], p[1]); ctx.scale(Math.max(0.12, spin), 1);
    ctx.fillStyle = "#e8c14a"; ctx.shadowBlur = 9; ctx.shadowColor = "#ffe27a";
    ctx.beginPath(); ctx.arc(0, 0, 5.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#8a6d1b"; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.restore();
    ctx.restore();
}

// GHOSTBUSTERS — green/red proton stream: a wild crackling beam that wobbles.
function _renderGhostbusters(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 24, phase = Math.floor(performance.now() / 70);
    const rand = (i) => { const s = Math.sin(i * 17.3 + phase * 61.1) * 43758.5; return s - Math.floor(s); };
    ctx.save(); ctx.lineCap = "round";
    // two intertwined streams: proton green + containment red
    for (const [col, seed, w] of [["#39ff7a", 0, 2.4], ["#ff4d4d", 50, 1.6]]) {
        ctx.strokeStyle = col; ctx.lineWidth = w; ctx.shadowBlur = 9; ctx.shadowColor = col;
        ctx.beginPath();
        for (let i = 0; i <= N; i++) {
            const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
            const q = _bezierAt(Math.min(1, t + 0.01), a, cp1, cp2, b);
            const dx = q[0] - p[0], dy = q[1] - p[1], L = Math.hypot(dx, dy) || 1;
            const nx = -dy / L, ny = dx / L;
            const j = (rand(i + seed) - 0.5) * 9 * Math.sin(t * Math.PI);  // 0 at ends
            i ? ctx.lineTo(p[0] + nx * j, p[1] + ny * j) : ctx.moveTo(p[0] + nx * j, p[1] + ny * j);
        }
        ctx.stroke();
    }
    ctx.restore();
}

// JURASSIC — fossilised amber rope with a frozen DNA double-helix inside.
function _renderJurassic(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#c97a16", 5, 8);           // amber resin
    _trunk(ctx, a, cp1, cp2, b, "#ffd98a", 2, 4);           // honey highlight
    // suspended DNA strands inside the amber
    const N = 36, phase = (performance.now() / 600) % (Math.PI * 2);
    for (let strand = 0; strand < 2; strand++) {
        ctx.strokeStyle = "#5a3408"; ctx.lineWidth = 1; ctx.globalAlpha = 0.85;
        ctx.beginPath();
        for (let i = 0; i <= N; i++) {
            const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
            const q = _bezierAt(Math.min(1, t + 0.01), a, cp1, cp2, b);
            const dx = q[0] - p[0], dy = q[1] - p[1], L = Math.hypot(dx, dy) || 1;
            const nx = -dy / L, ny = dx / L;
            const off = Math.sin(t * Math.PI * 5 + phase + strand * Math.PI) * 3.5;
            i ? ctx.lineTo(p[0] + nx * off, p[1] + ny * off) : ctx.moveTo(p[0] + nx * off, p[1] + ny * off);
        }
        ctx.stroke();
    }
    ctx.globalAlpha = 1; ctx.restore();
}

// AMONG-US — a little crewmate (with visor + backpack) sliding along the wire.
function _renderAmongUs(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const COLS = ["#c51111", "#132ed1", "#117f2d", "#ed54ba", "#f07d0d", "#f6f657"];
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#3a3f55", 2, 0);           // wire / vent line
    const t = (performance.now() / 1700) % 1, p = _bezierAt(t, a, cp1, cp2, b);
    const q = _bezierAt(Math.min(1, t + 0.01), a, cp1, cp2, b);
    const dir = q[0] - p[0] >= 0 ? 1 : -1;                  // face travel direction
    const col = COLS[Math.floor(t * COLS.length) % COLS.length];
    ctx.save(); ctx.translate(p[0], p[1]);
    const wob = Math.sin(performance.now() / 130) * 0.6;    // little waddle
    ctx.shadowBlur = 5; ctx.shadowColor = "rgba(0,0,0,0.5)";
    // backpack
    ctx.fillStyle = col;
    ctx.fillRect(-dir * 6.5, -1 + wob, 3.5, 6);
    // body (rounded capsule)
    ctx.beginPath();
    ctx.arc(0, -1.5 + wob, 5, Math.PI, 0);
    ctx.lineTo(5, 5 + wob); ctx.lineTo(-5, 5 + wob); ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#2a2f44"; ctx.fillRect(-1.5, 4.5 + wob, 1.6, 2.5);   // legs
    ctx.fillRect(0.4, 4.5 + wob, 1.6, 2.5);
    // visor
    ctx.fillStyle = "#a9d3e8"; ctx.shadowBlur = 0;
    ctx.beginPath(); ctx.ellipse(dir * 1.5, -2 + wob, 2.6, 1.8, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.beginPath(); ctx.ellipse(dir * 2.3, -2.6 + wob, 0.8, 0.6, 0, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    ctx.restore();
}

// ── GAME BATCH 3 (arcade + console classics) ─────────────────────────

// GTA — Rockstar Vice City neon: a hot-pink→cyan rope with a spinning
// wanted-level star pulsing at the midpoint.
function _renderGta(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 30, time = performance.now() / 200;
    ctx.save(); ctx.lineCap = "round";
    let prev = _bezierAt(0, a, cp1, cp2, b);
    for (let i = 1; i <= N; i++) {
        const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
        const hue = 320 - t * 130;                            // pink → cyan
        ctx.strokeStyle = `hsl(${hue}, 100%, ${58 + 8 * Math.sin(time + i)}%)`;
        ctx.lineWidth = 3.2; ctx.shadowBlur = 8; ctx.shadowColor = ctx.strokeStyle;
        ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
        prev = p;
    }
    const m = _bezierAt(0.5, a, cp1, cp2, b);
    ctx.translate(m[0], m[1]); ctx.rotate(performance.now() / 500);
    ctx.fillStyle = "#ffe14d"; ctx.shadowBlur = 10; ctx.shadowColor = "#fff2a0";
    ctx.beginPath();
    for (let k = 0; k < 10; k++) {
        const r = k % 2 ? 3 : 7, ang = (Math.PI / 5) * k - Math.PI / 2;
        const x = Math.cos(ang) * r, y = Math.sin(ang) * r;
        k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.closePath(); ctx.fill();
    ctx.restore();
}

// SPACE INVADERS — a row of blocky green aliens marching along the wire,
// legs twitching between two frames like the arcade original.
function _renderSpaceInvaders(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save();
    _trunk(ctx, a, cp1, cp2, b, "#0f3d12", 1.5, 0);          // faint scan rail
    const N = 5, frame = Math.floor(performance.now() / 300) % 2;
    const head = (performance.now() / 2600) % 1, s = 1.4;
    const rows = frame ? ["01110", "11111", "10101", "01010"]
                       : ["01110", "11111", "10101", "10001"];
    ctx.fillStyle = "#4dff5a"; ctx.shadowBlur = 6; ctx.shadowColor = "#39ff7a";
    for (let i = 0; i < N; i++) {
        const t = (i / N + head) % 1, p = _bezierAt(t, a, cp1, cp2, b);
        for (let r = 0; r < rows.length; r++)
            for (let c = 0; c < 5; c++)
                if (rows[r][c] === "1")
                    ctx.fillRect(p[0] + (c - 2.5) * s, p[1] + (r - 2) * s, s, s);
    }
    ctx.restore();
}

// TETRIS — coloured tetromino clusters tumbling (rotating) along the wire.
function _renderTetris(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const COLS = ["#00f0f0", "#f0f000", "#a000f0", "#00f000", "#f00000", "#0000f0", "#f0a000"];
    const N = 7, S = 4, head = (performance.now() / 1400) % 1;
    ctx.save();
    for (let i = 0; i < N; i++) {
        const t = (i / N + head) % 1, p = _bezierAt(t, a, cp1, cp2, b);
        const col = COLS[i % COLS.length];
        ctx.save(); ctx.translate(p[0], p[1]);
        ctx.rotate((Math.floor(performance.now() / 200 + i) % 4) * Math.PI / 2);
        ctx.fillStyle = col; ctx.shadowBlur = 5; ctx.shadowColor = col;
        ctx.strokeStyle = "rgba(255,255,255,0.4)"; ctx.lineWidth = 0.6;
        for (const [ox, oy] of [[-S, -S], [0, -S], [-S, 0]]) {   // simple L cluster
            ctx.fillRect(ox, oy, S, S); ctx.strokeRect(ox, oy, S, S);
        }
        ctx.restore();
    }
    ctx.restore();
}

// STREET FIGHTER — a Hadouken: white-hot core in a cyan energy orb hurtling
// output→input, leaving a fading motion streak.
function _renderStreetFighter(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#1e90ff", 2, 6);            // ki trail
    const t = (performance.now() / 900) % 1, p = _bezierAt(t, a, cp1, cp2, b);
    for (let k = 1; k <= 5; k++) {
        const q = _bezierAt(Math.max(0, t - k * 0.03), a, cp1, cp2, b);
        ctx.globalAlpha = 0.5 - k * 0.08; ctx.fillStyle = "#9fd8ff";
        ctx.beginPath(); ctx.arc(q[0], q[1], 4 - k * 0.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 14; ctx.shadowColor = "#39c0ff"; ctx.fillStyle = "#39c0ff";
    ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0; ctx.fillStyle = "#ffffff";
    ctx.beginPath(); ctx.arc(p[0], p[1], 2.6, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
}

// MEGA MAN — the Mega Buster: a stream of pale-cyan plasma pellets firing.
function _renderMegaMan(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#0a3a6b", 2, 0);            // armour-blue rail
    const N = 4, head = (performance.now() / 650) % 1;
    for (let i = 0; i < N; i++) {
        const t = (i / N + head) % 1, p = _bezierAt(t, a, cp1, cp2, b);
        ctx.fillStyle = "#8ff0ff"; ctx.shadowBlur = 8; ctx.shadowColor = "#28c8ff";
        ctx.beginPath(); ctx.arc(p[0], p[1], 3.4, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#ffffff"; ctx.shadowBlur = 0;
        ctx.beginPath(); ctx.arc(p[0], p[1], 1.3, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
}

// MORTAL KOMBAT — a jagged blood-red lightning arc crackling down the wire.
function _renderMortalKombat(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    const N = 22, phase = Math.floor(performance.now() / 60);
    const rand = (i) => { const s = Math.sin(i * 33.7 + phase * 47.1) * 43758.5; return s - Math.floor(s); };
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#3a0000", 4, 4);            // dark aura
    for (const [col, w, amp] of [["#ff2b1f", 2.2, 10], ["#ffd24d", 1, 5]]) {
        ctx.strokeStyle = col; ctx.lineWidth = w; ctx.shadowBlur = 9; ctx.shadowColor = "#ff3b2f";
        ctx.beginPath();
        for (let i = 0; i <= N; i++) {
            const t = i / N, p = _bezierAt(t, a, cp1, cp2, b);
            const q = _bezierAt(Math.min(1, t + 0.01), a, cp1, cp2, b);
            const dx = q[0] - p[0], dy = q[1] - p[1], L = Math.hypot(dx, dy) || 1;
            const nx = -dy / L, ny = dx / L, j = (rand(i) - 0.5) * amp * Math.sin(t * Math.PI);
            i ? ctx.lineTo(p[0] + nx * j, p[1] + ny * j) : ctx.moveTo(p[0] + nx * j, p[1] + ny * j);
        }
        ctx.stroke();
    }
    ctx.restore();
}

// METROID — Samus's charge beam: an orange energy tube with a pulsing charge
// orb + orbiting particles near the tip.
function _renderMetroid(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#ff7a18", 4, 10);           // beam glow
    _trunk(ctx, a, cp1, cp2, b, "#ffe08a", 1.4, 4);          // hot core
    const m = _bezierAt(0.82, a, cp1, cp2, b), now = performance.now();
    ctx.shadowBlur = 12; ctx.shadowColor = "#ffb03a"; ctx.fillStyle = "#fff0c8";
    ctx.beginPath(); ctx.arc(m[0], m[1], 4 * (0.7 + 0.3 * Math.sin(now / 160)), 0, Math.PI * 2); ctx.fill();
    for (let k = 0; k < 3; k++) {
        const ang = now / 120 + k * (Math.PI * 2 / 3);
        ctx.beginPath(); ctx.arc(m[0] + Math.cos(ang) * 7, m[1] + Math.sin(ang) * 7, 1.6, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
}

// DOOM — hellfire plasma: green BFG bolts racing over a molten-red rail.
function _renderDoom(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#7a1010", 4, 8);            // molten rail
    _trunk(ctx, a, cp1, cp2, b, "#ff5a2a", 1.4, 4);          // lava highlight
    const N = 3, head = (performance.now() / 500) % 1;
    for (let i = 0; i < N; i++) {
        const t = (i / N + head) % 1, p = _bezierAt(t, a, cp1, cp2, b);
        ctx.fillStyle = "#7cff4d"; ctx.shadowBlur = 12; ctx.shadowColor = "#39ff14";
        ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#eaffea"; ctx.shadowBlur = 0;
        ctx.beginPath(); ctx.arc(p[0], p[1], 1.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
}

// ELDEN RING — the Erdtree's grace: a gilded rope shedding golden embers.
function _renderEldenRing(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#b8860b", 3, 6);            // gilded rope
    _trunk(ctx, a, cp1, cp2, b, "#ffe9a8", 1, 3);            // grace highlight
    const N = 10, now = performance.now();
    for (let i = 0; i < N; i++) {
        const base = _bezierAt((i + 0.5) / N, a, cp1, cp2, b);
        const life = ((now / 1400) + i * 0.13) % 1;          // rise + fade
        ctx.globalAlpha = (1 - life) * 0.9;
        ctx.fillStyle = "#ffdf7e"; ctx.shadowBlur = 7; ctx.shadowColor = "#ffcf4d";
        ctx.beginPath();
        ctx.arc(base[0] + Math.sin(now / 300 + i) * 4, base[1] - life * 14, 1.6 * (1 - life * 0.5), 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.globalAlpha = 1; ctx.restore();
}

// GALAGA — a white arcade fighter flies the wire firing twin blue bolts.
function _renderGalaga(ctx, a, b) {
    const [cp1, cp2] = _bezierPoints(a, b);
    ctx.save(); ctx.lineCap = "round";
    _trunk(ctx, a, cp1, cp2, b, "#12123a", 1.5, 0);          // starfield rail
    const t = (performance.now() / 1500) % 1, p = _bezierAt(t, a, cp1, cp2, b);
    const q = _bezierAt(Math.min(1, t + 0.01), a, cp1, cp2, b);
    const ang = Math.atan2(q[1] - p[1], q[0] - p[0]);
    const boltT = (performance.now() / 250) % 1;
    ctx.strokeStyle = "#66e0ff"; ctx.lineWidth = 1.4; ctx.shadowBlur = 6; ctx.shadowColor = "#66e0ff";
    for (const s of [-2.5, 2.5]) {
        const bp = _bezierAt(Math.min(1, t + 0.04 + boltT * 0.08), a, cp1, cp2, b);
        ctx.beginPath();
        ctx.moveTo(bp[0] - Math.sin(ang) * s, bp[1] + Math.cos(ang) * s);
        ctx.lineTo(bp[0] - Math.sin(ang) * s + Math.cos(ang) * 4, bp[1] + Math.cos(ang) * s + Math.sin(ang) * 4);
        ctx.stroke();
    }
    ctx.save(); ctx.translate(p[0], p[1]); ctx.rotate(ang);
    ctx.fillStyle = "#eef4ff"; ctx.shadowBlur = 5; ctx.shadowColor = "#cfe4ff";
    ctx.beginPath(); ctx.moveTo(6, 0); ctx.lineTo(-4, -4); ctx.lineTo(-2, 0); ctx.lineTo(-4, 4); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#ff4d4d"; ctx.beginPath(); ctx.arc(-1, 0, 1.2, 0, Math.PI * 2); ctx.fill();  // cockpit
    ctx.restore();
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
    "rgb-spectrum": _renderRgbSpectrum,
    "gradient":     _renderGradient,
    "fire":         _renderFire,
    "comet":        _renderComet,
    "candy":        _renderCandy,
    "aurora":       _renderAurora,
    "heartbeat":    _renderHeartbeat,
    // franchise styles
    "matrix":       _renderMatrix,
    "tron":         _renderTron,
    "portal":       _renderPortal,
    "pacman":       _renderPacman,
    "hyperspace":   _renderHyperspace,
    "upside-down":  _renderUpsideDown,
    "ki-blast":     _renderKi,
    "rainbow-road": _renderRainbowRoad,
    // franchise styles (batch 2)
    "pokemon":          _renderPokemon,
    "minecraft":        _renderMinecraft,
    "sonic":            _renderSonic,
    "zelda":            _renderZelda,
    "halo":             _renderHalo,
    "avatar-bending":   _renderAvatarBending,
    "john-wick":        _renderJohnWick,
    "ghostbusters":     _renderGhostbusters,
    "jurassic":         _renderJurassic,
    "among-us":         _renderAmongUs,
    // game classics (batch 3)
    "gta":              _renderGta,
    "space-invaders":   _renderSpaceInvaders,
    "tetris":           _renderTetris,
    "street-fighter":   _renderStreetFighter,
    "mega-man":         _renderMegaMan,
    "mortal-kombat":    _renderMortalKombat,
    "metroid":          _renderMetroid,
    "doom":             _renderDoom,
    "elden-ring":       _renderEldenRing,
    "galaga":           _renderGalaga,
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

/* ════════════════ COMPLETION FX — game sprites ride the wires ════════════
 * On `execution_success` a game-themed sprite travels every link output→input
 * once (~1.4s), then clears. Independent of the noodle STYLE (works on
 * `default` too). Reuses _bezierPoints/_bezierAt so the sprite follows the
 * exact wire path. Setting: mec.noodle.completion_fx (off/pacman/coin/ring/
 * star/invader/random). A rAF pump forces redraws for the duration.
 * ────────────────────────────────────────────────────────────────────── */
const FX_SETTING_ID = "mec.noodle.completion_fx";
const FX_DUR_MS = 1400;
const _FX_THEMES = ["pacman", "coin", "ring", "star", "invader"];
const _fx = { active: false, start: 0, base: "random" };

function _fxSetting() {
    try {
        const v = app.ui.settings.getSettingValue(FX_SETTING_ID);
        return (v === undefined || v === null) ? "random" : v;
    } catch (_) { return "random"; }
}
function _fxThemeFor(link) {
    if (_fx.base !== "random") return _fx.base;
    const id = Math.abs((link && (link.id ?? 0)) | 0);
    return _FX_THEMES[id % _FX_THEMES.length];
}

// ── sprites (drawn centred at x,y; dir=+1/-1 travel; ts=seconds for phase) ──
function _fxPacman(ctx, x, y, dir, ts) {
    const r = 11, m = 0.30 * (0.5 + 0.5 * Math.sin(ts * 16));
    ctx.save(); ctx.translate(x, y); ctx.scale(dir, 1);
    ctx.fillStyle = "#ffd54a";
    ctx.beginPath(); ctx.moveTo(0, 0);
    ctx.arc(0, 0, r, m * Math.PI, (2 - m) * Math.PI); ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#1a1a22";
    ctx.beginPath(); ctx.arc(2, -r * 0.45, 1.7, 0, 7); ctx.fill();
    ctx.restore();
}
function _fxCoin(ctx, x, y, dir, ts) {
    const r = 10, sx = Math.abs(Math.cos(ts * 6)) * 0.82 + 0.18;
    ctx.save(); ctx.translate(x, y); ctx.scale(sx, 1);
    ctx.fillStyle = "#ffcf33"; ctx.beginPath(); ctx.arc(0, 0, r, 0, 7); ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = "#c9971a"; ctx.stroke();
    if (sx > 0.5) { ctx.fillStyle = "#a8790f"; ctx.font = "bold 12px ui-sans-serif,system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("$", 0, 1); }
    ctx.restore();
}
function _fxRing(ctx, x, y, dir, ts) {
    const r = 10, sx = Math.abs(Math.cos(ts * 6)) * 0.9 + 0.1;
    ctx.save(); ctx.translate(x, y); ctx.scale(sx, 1);
    ctx.lineWidth = 3.6; ctx.strokeStyle = "#ffd54a"; ctx.beginPath(); ctx.arc(0, 0, r, 0, 7); ctx.stroke();
    ctx.lineWidth = 1.2; ctx.strokeStyle = "#fff3b0"; ctx.beginPath(); ctx.arc(0, 0, r * 0.7, 0, 7); ctx.stroke();
    ctx.restore();
}
function _fxStar(ctx, x, y, dir, ts) {
    const R = 12; ctx.save(); ctx.translate(x, y); ctx.rotate(ts * 3);
    ctx.fillStyle = "#ffe34a"; ctx.strokeStyle = "#e0a815"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < 5; i++) {
        const a1 = -Math.PI / 2 + i * 2 * Math.PI / 5, a2 = a1 + Math.PI / 5;
        ctx.lineTo(Math.cos(a1) * R, Math.sin(a1) * R);
        ctx.lineTo(Math.cos(a2) * R * 0.45, Math.sin(a2) * R * 0.45);
    }
    ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#1a1a22";
    ctx.beginPath(); ctx.arc(-3, -1, 1.4, 0, 7); ctx.arc(3, -1, 1.4, 0, 7); ctx.fill();
    ctx.restore();
}
const _INV_A = ["..X.X..", "..XXX..", ".XXXXX.", "X.XXX.X", "X.X.X.X"];
const _INV_B = ["..X.X..", "X.XXX.X", "XXXXXXX", ".XXXXX.", ".X...X."];
function _fxInvader(ctx, x, y, dir, ts) {
    const s = 2.4, g = (Math.sin(ts * 9) > 0) ? _INV_A : _INV_B, w = 7, h = 5;
    ctx.save(); ctx.fillStyle = "#a6e3a1"; ctx.translate(x - w * s / 2, y - h * s / 2);
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) if (g[r][c] === "X") ctx.fillRect(c * s, r * s, s - 0.3, s - 0.3);
    ctx.restore();
}
const _FX_SPRITES = { pacman: _fxPacman, coin: _fxCoin, ring: _fxRing, star: _fxStar, invader: _fxInvader };

function _triggerFxAnim() {
    const theme = _fxSetting();
    if (theme === "off") return;
    _fx.base = theme; _fx.active = true; _fx.start = performance.now();
    // Links are drawn on the background canvas and the graph loop may be idle
    // after a run finishes, so setting dirty flags alone won't repaint. Force a
    // real draw each frame for the (short) animation window; one final draw
    // clears the last sprite. Paused while the tab is hidden.
    const pump = () => {
        if (!_fx.active) return;
        const done = performance.now() - _fx.start >= FX_DUR_MS;
        if (done) _fx.active = false;
        try { if (app.canvas && !document.hidden) app.canvas.draw(true, true); } catch (_) {}
        if (!done) requestAnimationFrame(pump);
    };
    requestAnimationFrame(pump);
}

// Draw the travelling sprite for one link (called from renderLink, any style).
function _maybeFx(canvas, ctx, a, b, link) {
    if (!_fx.active) return;
    const p = (performance.now() - _fx.start) / FX_DUR_MS;
    if (p >= 1) { _fx.active = false; return; }
    const [cp1, cp2] = _bezierPoints(a, b);
    const t = p;                                   // linear head; sprite at the front
    const pos = _bezierAt(t, a, cp1, cp2, b);
    const ah = _bezierAt(Math.min(1, t + 0.02), a, cp1, cp2, b);
    const dir = (ah[0] - pos[0]) >= 0 ? 1 : -1;
    // short fading trail
    ctx.save();
    for (let k = 1; k <= 3; k++) {
        const tt = t - k * 0.035; if (tt < 0) break;
        const tp = _bezierAt(tt, a, cp1, cp2, b);
        ctx.globalAlpha = 0.18 * (3 - k) / 3;
        ctx.fillStyle = "#fff3b0";
        ctx.beginPath(); ctx.arc(tp[0], tp[1], 3.2, 0, 7); ctx.fill();
    }
    ctx.globalAlpha = 1;
    (_FX_SPRITES[_fxThemeFor(link)] || _fxStar)(ctx, pos[0], pos[1], dir, performance.now() / 1000);
    ctx.restore();
}

function _installRenderPatch() {
    if (_orig || !window.LGraphCanvas) return;
    _orig = LGraphCanvas.prototype.renderLink;
    LGraphCanvas.prototype.renderLink = function (ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, num_sublines) {
        const style = _currentStyle();
        if (style === "default" || !_RENDER[style]) {
            // Default skin: core renders its spline — UNLESS a custom pipe
            // SHAPE is active, in which case we stroke the shape ourselves
            // (same colour/width as core would use).
            const _sh = _currentShape();
            const _core = _sh === "spline" || (_sh === "auto" && !_alignedStraight(a, b));
            if (_core) {
                const _r = _orig.call(this, ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, num_sublines);
                if (_fx.active) { try { _maybeFx(this, ctx, a, b, link); } catch (_) {} }
                return _r;
            }
            try {
                const linkType = link?.type || "";
                const effColor = _safeColor(color) || _typeColor(linkType);
                ctx.save();
                ctx.strokeStyle = effColor;
                ctx.lineWidth = this.connections_width || 3;
                ctx.lineJoin = "round"; ctx.lineCap = "round";
                ctx.beginPath();
                if (_sh === "angular" || _sh === "diag45") {
                    const pts = _shapeWaypoints(a, b, _sh);
                    ctx.moveTo(pts[0][0], pts[0][1]);
                    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
                } else {
                    const [c1, c2] = _bezierPoints(a, b);
                    ctx.moveTo(a[0], a[1]);
                    ctx.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], b[0], b[1]);
                }
                ctx.stroke();
                ctx.restore();
                if (link) {
                    const mid = _bezierAt(0.5, a, ..._bezierPoints(a, b), b);
                    if (!link._pos) link._pos = new Float32Array(2);
                    link._pos[0] = mid[0]; link._pos[1] = mid[1];
                    ctx.save();
                    ctx.beginPath(); ctx.arc(mid[0], mid[1], 5, 0, Math.PI * 2);
                    ctx.fillStyle = effColor; ctx.globalAlpha = 0.95; ctx.fill();
                    ctx.globalAlpha = 1; ctx.lineWidth = 1.5;
                    ctx.strokeStyle = "rgba(20,20,28,0.65)"; ctx.stroke();
                    ctx.restore();
                }
                if (_fx.active) { try { _maybeFx(this, ctx, a, b, link); } catch (_) {} }
                return;
            } catch (_e2) {
                return _orig.call(this, ctx, a, b, link, skip_border, flow, color, start_dir, end_dir, num_sublines);
            }
        }
        try {
            const linkType = link?.type || "";
            // Resolve the incoming link colour too — ComfyUI may hand us a
            // CSS var() which would render the noodle black on canvas.
            const effColor = _safeColor(color) || _typeColor(linkType);
            _RENDER[style](ctx, a, b, effColor, linkType);
            // Unique per-skin animation overlay (shape-aware via _bezierAt).
            const _ov = _ANIM_OVERLAYS[style];
            if (_ov && !document.hidden) { try { _ov(ctx, a, b, effColor, performance.now()); } catch (_) {} }
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
            if (_ANIMATED.has(style)) {
                const _now = performance.now();
                if (!document.hidden && _now - (this._c2cNoodleAnimTick || 0) > 50) {
                    this._c2cNoodleAnimTick = _now;
                    this.dirty_canvas = true;
                }
            }
            // completion FX sprite rides this wire too
            if (_fx.active) { try { _maybeFx(this, ctx, a, b, link); } catch (_) {} }
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

if (!LITE) app.registerExtension({
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
        {
            id: SHAPE_SETTING_ID,
            name: "Pipe shape",
            tooltip: "The PATH the wires take — independent of the visual style, any skin rides "
                   + "any shape. auto = dead straight when nodes are aligned, gentle curve when "
                   + "offset (Nuke-like). angular = 90° elbows, diag45 = Houdini diagonals, "
                   + "arc = circular bow, s-curve = deep ease.",
            type: "combo",
            options: SHAPES,
            defaultValue: "auto",
            onChange: (v) => { _shapeCache = (v === undefined || v === null) ? "auto" : v; try { app.graph?.setDirtyCanvas(true, true); } catch (_) {} },
        },
        {
            id: FX_SETTING_ID,
            name: "Noodle completion FX",
            tooltip:
                "When a run finishes, a game sprite travels every wire once. " +
                "`random` gives each wire a different sprite. `off` disables it.",
            type: "combo",
            options: ["off", "random", ..._FX_THEMES],
            defaultValue: "random",
        },
    ],
    async setup() {
        _refreshStyleCache();   // seed the per-frame style cache once at startup
        _refreshShapeCache();   // seed the pipe-shape cache too
        // Game-sprite completion animation: ride every wire when a run finishes.
        try { api.addEventListener("execution_success", () => { try { _triggerFxAnim(); } catch (_) {} }); } catch (_) {}
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
