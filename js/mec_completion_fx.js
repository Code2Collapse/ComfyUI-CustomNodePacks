/**
 * mec_completion_fx.js — Phase 17e: Confetti / Vibe FX on completion
 *
 * On `execution_success`, briefly rains lightweight confetti from the top
 * of the viewport. Pure canvas2D, no dependencies. Optional success chime.
 *
 * Settings:
 *   mec.completion_fx.confetti — bool (default true)
 *   mec.completion_fx.chime    — bool (default false)
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CANVAS_ID = "mec-confetti-canvas";
const COLORS = ["#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#89b4fa", "#cba6f7", "#f5c2e7"];
const NUM_PIECES = 80;
const DURATION_MS = 2200;

let _running = false;

function _ensureCanvas() {
    let c = document.getElementById(CANVAS_ID);
    if (c) return c;
    c = document.createElement("canvas");
    c.id = CANVAS_ID;
    Object.assign(c.style, {
        position: "fixed",
        inset: "0",
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: "99999",
        display: "none",
    });
    document.body.appendChild(c);
    return c;
}

function _resizeCanvas(c) {
    c.width  = window.innerWidth;
    c.height = window.innerHeight;
}

function _spawnPieces() {
    const w = window.innerWidth;
    const pieces = [];
    for (let i = 0; i < NUM_PIECES; i++) {
        pieces.push({
            x: Math.random() * w,
            y: -20 - Math.random() * 200,
            vx: (Math.random() - 0.5) * 3,
            vy: 2 + Math.random() * 4,
            rot: Math.random() * Math.PI * 2,
            vr:  (Math.random() - 0.5) * 0.2,
            size: 6 + Math.random() * 6,
            color: COLORS[Math.floor(Math.random() * COLORS.length)],
        });
    }
    return pieces;
}

function _confetti() {
    if (_running) return;
    _running = true;
    const c = _ensureCanvas();
    _resizeCanvas(c);
    c.style.display = "block";
    const ctx = c.getContext("2d");
    const pieces = _spawnPieces();
    const start = performance.now();
    const tick = (now) => {
        const elapsed = now - start;
        ctx.clearRect(0, 0, c.width, c.height);
        for (const p of pieces) {
            p.vy += 0.08;  // gravity
            p.x  += p.vx;
            p.y  += p.vy;
            p.rot += p.vr;
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rot);
            ctx.fillStyle = p.color;
            ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
            ctx.restore();
        }
        if (elapsed < DURATION_MS) {
            requestAnimationFrame(tick);
        } else {
            ctx.clearRect(0, 0, c.width, c.height);
            c.style.display = "none";
            _running = false;
        }
    };
    requestAnimationFrame(tick);
}

let _audioCtx = null;
function _chime() {
    try {
        _audioCtx = _audioCtx || new (window.AudioContext || window.webkitAudioContext)();
        const ctx = _audioCtx;
        const now = ctx.currentTime;
        const notes = [523.25, 659.25, 783.99]; // C5 E5 G5
        notes.forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = "sine";
            osc.frequency.value = freq;
            osc.connect(gain).connect(ctx.destination);
            const t = now + i * 0.08;
            gain.gain.setValueAtTime(0.0001, t);
            gain.gain.exponentialRampToValueAtTime(0.15, t + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.35);
            osc.start(t);
            osc.stop(t + 0.4);
        });
    } catch (e) {
        // Browser likely blocked audio until user gesture.
    }
}

function _onSuccess() {
    try {
        if (app.ui.settings.getSettingValue("mec.completion_fx.confetti", true)) _confetti();
        if (app.ui.settings.getSettingValue("mec.completion_fx.chime", false))   _chime();
    } catch { /* ignore */ }
}

app.registerExtension({
    name: "MEC.CompletionFX",
    settings: [
        {
            id: "mec.completion_fx.confetti",
            name: "Completion FX: confetti on success",
            type: "boolean",
            defaultValue: true,
        },
        {
            id: "mec.completion_fx.chime",
            name: "Completion FX: subtle chime on success",
            type: "boolean",
            defaultValue: false,
        },
    ],
    async setup() {
        api.addEventListener("execution_success", _onSuccess);
        window.addEventListener("resize", () => {
            const c = document.getElementById(CANVAS_ID);
            if (c) _resizeCanvas(c);
        });
        console.log("[MEC.CompletionFX] Loaded.");
    },
});
