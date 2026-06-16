/**
 * c2c_celebrations.js — P2.2 Celebrations (confetti + sound + easter eggs)
 *
 * Defaults per MEGA_PLAN.md §P2.2:
 *   - confetti     ON  (settings: c2c.celebrations.confetti)
 *   - sound        OFF (settings: c2c.celebrations.sound)
 *   - easter eggs  ON  (settings: c2c.celebrations.easterEggs) — rare variants <2%
 *   - master kill  ON  (settings: c2c.celebrations.enabled)
 *
 * Always honours `prefers-reduced-motion: reduce` — no animation, no sound.
 *
 * Uses public LiteGraph / app.api channels only.  Pure canvas implementation —
 * no third-party libraries.  Z-index uses the c2c theme token scale
 * (`var(--c2c-z-toast)`) so confetti renders above modals/palette/toasts.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NS = "C2C.Celebrations";

/* ─── settings ─── */
const SETTING = {
  master    : "c2c.celebrations.enabled",
  confetti  : "c2c.celebrations.confetti",
  sound     : "c2c.celebrations.sound",
  easter    : "c2c.celebrations.easterEggs",
};

/* ─── state ─── */
let _canvas       = null;
let _ctx          = null;
let _particles    = [];
let _rafId        = 0;
let _audioCtx     = null;
let _lastSuccessAt = 0;            // dedupe — execution_success can fire twice
let _reduceMotion = false;
let _enabled      = { master: true, confetti: true, sound: false, easter: true };

/* ─── reduced-motion ─── */
function _watchMotionPreference() {
  try {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    _reduceMotion = mq.matches;
    mq.addEventListener?.("change", (e) => { _reduceMotion = e.matches; });
  } catch (_e) {
    _reduceMotion = false;
  }
}

/* ─── overlay canvas (lazy) ─── */
function _ensureCanvas() {
  if (_canvas) return _canvas;
  const c = document.createElement("canvas");
  c.id = "c2c-celebrations-canvas";
  c.style.cssText = [
    "position:fixed",
    "inset:0",
    "width:100vw",
    "height:100vh",
    "pointer-events:none",
    "z-index:var(--c2c-z-toast, 100002)",
    "display:block",
  ].join(";");
  document.body.appendChild(c);
  _canvas = c;
  _ctx    = c.getContext("2d");
  _resizeCanvas();
  window.addEventListener("resize", _resizeCanvas, { passive: true });
  return c;
}

function _resizeCanvas() {
  if (!_canvas) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  _canvas.width  = Math.floor(window.innerWidth  * dpr);
  _canvas.height = Math.floor(window.innerHeight * dpr);
  _ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

/* ─── particle factory ─── */
const PALETTE_DEFAULT = ["var(--c2c-warn)","var(--c2c-blue)","var(--c2c-violetSoft)","var(--c2c-dangerStrong)","var(--c2c-green)","var(--c2c-sky)","var(--c2c-danger)"];
const PALETTE_RAINBOW = ["var(--c2c-danger)","var(--c2c-warnBright)","#8AC926","#1982C4","var(--c2c-overlay0)","var(--c2c-red)","#7DDF64"];
const PALETTE_GHOST   = ["var(--c2c-accentBright)","var(--c2c-fg)","#A3BE8C","var(--c2c-sapphire)","var(--c2c-overlay1)"]; // rare easter egg
const SHAPES = ["sq","rect","circle","tri"];

function _rand(a, b) { return a + Math.random() * (b - a); }

// Canvas 2D `fillStyle` CANNOT parse `var(--x)` — assigning it is silently
// ignored and the context keeps its previous color (black by default), which
// made every default-palette particle render BLACK ("all black confetti").
// Resolve CSS custom properties to concrete colors before they hit the canvas.
function _resolveColor(c) {
  if (typeof c === "string" && c.startsWith("var(")) {
    const name = c.slice(4, -1).split(",")[0].trim();
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      if (v) return v;
    } catch (_) { /* fall through to fallback */ }
    return "#89b4fa";  // legible fallback so a particle is never invisible/black
  }
  return c;
}

function _makeBurst(originX, originY, variant) {
  const paletteRaw =
    variant === "rainbow" ? PALETTE_RAINBOW :
    variant === "ghost"   ? PALETTE_GHOST   : PALETTE_DEFAULT;
  // Resolve once per burst (≈7 entries) rather than per particle.
  const palette = paletteRaw.map(_resolveColor);
  const count = variant === "rainbow" ? 220 : 140;
  const ps = [];
  for (let i = 0; i < count; i++) {
    const angle = _rand(-Math.PI * 0.85, -Math.PI * 0.15); // upward fan
    const speed = _rand(4.5, 11.5);
    ps.push({
      x  : originX,
      y  : originY,
      vx : Math.cos(angle) * speed,
      vy : Math.sin(angle) * speed,
      g  : _rand(0.18, 0.32),                              // gravity per frame
      drag: _rand(0.985, 0.998),
      size : _rand(5, 11),
      rot  : _rand(0, Math.PI * 2),
      vr   : _rand(-0.25, 0.25),
      life : 0,
      ttl  : _rand(80, 130),                               // frames
      color: palette[(Math.random() * palette.length) | 0],
      shape: SHAPES[(Math.random() * SHAPES.length) | 0],
    });
  }
  _particles.push(...ps);
}

/* ─── render loop ─── */
function _step() {
  if (!_ctx) { _rafId = 0; return; }
  _ctx.clearRect(0, 0, _canvas.width, _canvas.height);

  for (let i = _particles.length - 1; i >= 0; i--) {
    const p = _particles[i];
    p.life += 1;
    p.vy += p.g;
    p.vx *= p.drag;
    p.vy *= p.drag;
    p.x  += p.vx;
    p.y  += p.vy;
    p.rot += p.vr;

    const k = 1 - (p.life / p.ttl);
    if (k <= 0 || p.y > window.innerHeight + 40) {
      _particles.splice(i, 1);
      continue;
    }

    _ctx.save();
    _ctx.globalAlpha = Math.max(0, Math.min(1, k));
    _ctx.translate(p.x, p.y);
    _ctx.rotate(p.rot);
    _ctx.fillStyle = p.color;
    switch (p.shape) {
      case "circle":
        _ctx.beginPath();
        _ctx.arc(0, 0, p.size * 0.55, 0, Math.PI * 2);
        _ctx.fill();
        break;
      case "rect":
        _ctx.fillRect(-p.size, -p.size * 0.4, p.size * 2, p.size * 0.8);
        break;
      case "tri":
        _ctx.beginPath();
        _ctx.moveTo(0, -p.size);
        _ctx.lineTo(p.size, p.size);
        _ctx.lineTo(-p.size, p.size);
        _ctx.closePath();
        _ctx.fill();
        break;
      default: // sq
        _ctx.fillRect(-p.size * 0.5, -p.size * 0.5, p.size, p.size);
    }
    _ctx.restore();
  }

  if (_particles.length > 0) {
    _rafId = requestAnimationFrame(_step);
  } else {
    _rafId = 0;
  }
}

/* ─── sound (deferred AudioContext to avoid autoplay-policy errors) ─── */
function _playChime() {
  try {
    if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const ctx = _audioCtx;
    if (ctx.state === "suspended") { ctx.resume?.(); }
    const now = ctx.currentTime;
    // 3-note major triad arpeggio: C5, E5, G5
    const freqs = [523.25, 659.25, 783.99];
    freqs.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, now + i * 0.09);
      gain.gain.setValueAtTime(0, now + i * 0.09);
      gain.gain.linearRampToValueAtTime(0.18, now + i * 0.09 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.09 + 0.4);
      osc.connect(gain).connect(ctx.destination);
      osc.start(now + i * 0.09);
      osc.stop(now + i * 0.09 + 0.45);
    });
  } catch (e) {
    console.warn("[c2c.celebrations] audio failed:", e?.message || e);
  }
}

/* ─── public trigger ─── */
function _trigger(reason) {
  if (!_enabled.master) return;
  if (_reduceMotion) return;

  // Confetti burst. When the Completion-FX style engine is loaded
  // (c2c_completion_fx.js — 14 themed styles incl. saber-clash/tron/matrix),
  // DELEGATE to it: both extensions listening to execution_success used to
  // double-fire two confetti rains at once. Easter eggs map to 'random'.
  if (_enabled.confetti) {
    if (window.__C2C_FX?.play) {
      const easter = _enabled.easter && Math.random() < 0.018;
      window.__C2C_FX.play(easter ? "random" : undefined);
    } else {
      _ensureCanvas();
      // Two-origin sympathetic burst feels richer than a single point.
      const w = window.innerWidth, h = window.innerHeight;
      let variant = "default";
      if (_enabled.easter && Math.random() < 0.018) {
        variant = Math.random() < 0.5 ? "rainbow" : "ghost";
      }
      _makeBurst(w * 0.30, h * 0.92, variant);
      _makeBurst(w * 0.70, h * 0.92, variant);
      if (!_rafId) _rafId = requestAnimationFrame(_step);
    }
  }

  // Sound
  if (_enabled.sound) _playChime();

  // Expose hook for tests
  try {
    window.dispatchEvent(new CustomEvent("c2c:celebration", { detail: { reason } }));
  } catch (_e) { /* ignore */ }
}

/* ─── settings binding ─── */
function _readSetting(id, fallback) {
  try { return app.ui.settings.getSettingValue(id, fallback); }
  catch (_e) { return fallback; }
}

function _registerSettings() {
  const defs = [
    [SETTING.master  , "C2C › Celebrations › Enabled"      , true ],
    [SETTING.confetti, "C2C › Celebrations › Confetti"     , true ],
    [SETTING.sound   , "C2C › Celebrations › Sound chime"  , false],
    [SETTING.easter  , "C2C › Celebrations › Easter eggs"  , true ],
  ];
  for (const [id, name, def] of defs) {
    try {
      app.ui.settings.addSetting({
        id, name, type: "boolean", defaultValue: def,
        onChange: (v) => { _refreshFlags(); },
      });
    } catch (e) {
      console.warn(`[c2c.celebrations] addSetting failed for ${id}:`, e?.message || e);
    }
  }
}

function _refreshFlags() {
  _enabled.master   = !!_readSetting(SETTING.master   , true );
  _enabled.confetti = !!_readSetting(SETTING.confetti , true );
  _enabled.sound    = !!_readSetting(SETTING.sound    , false);
  _enabled.easter   = !!_readSetting(SETTING.easter   , true );
}

/* ─── event wiring ─── */
function _onSuccess(_evt) {
  // The Completion-FX style engine (c2c_completion_fx.js) ALSO listens to
  // execution_success and calls the same _play(). If it is present, let it
  // OWN the success celebration — otherwise both fire _play() once each and
  // the user gets a double confetti burst (+ possible double chime) per run.
  // The manual window.C2CCelebrations.trigger() path is unaffected, and the
  // fallback below still runs when Completion-FX is not loaded.
  if (window.__C2C_FX?.play) return;
  const now = performance.now();
  if (now - _lastSuccessAt < 400) return;     // dedupe twin events
  _lastSuccessAt = now;
  _trigger("execution_success");
}

function _wireEvents() {
  // ComfyUI fires `execution_success` on api when the queue finishes; some
  // builds also expose `executed` (per-node). We listen to both but dedupe.
  try {
    if (typeof api?.addEventListener === "function") {
      api.addEventListener("execution_success", _onSuccess);
      // 'status' carries queue_remaining transitions; treat finished-from-busy
      // as celebration trigger fallback when execution_success is missing.
      let prevQueueRemaining = null;
      api.addEventListener("status", (evt) => {
        const q = evt?.detail?.exec_info?.queue_remaining;
        if (typeof q !== "number") return;
        if (prevQueueRemaining != null && prevQueueRemaining > 0 && q === 0) {
          _onSuccess(evt);
        }
        prevQueueRemaining = q;
      });
    }
  } catch (e) {
    console.warn("[c2c.celebrations] event wiring failed:", e?.message || e);
  }
}

/* ─── extension ─── */
app.registerExtension({
  name: NS,

  async setup() {
    _watchMotionPreference();
    _registerSettings();
    _refreshFlags();
    _wireEvents();

    // Public manual trigger for other extensions / palette commands.
    window.C2CCelebrations = Object.freeze({
      trigger : (reason = "manual") => _trigger(reason),
      isReducedMotion: () => _reduceMotion,
      flags   : () => ({ ..._enabled }),
    });
  },
});
