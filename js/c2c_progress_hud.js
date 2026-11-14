// FILE: js/mec_progress_hud.js
// FEATURE: Per-node progress HUD overlay with ETA, rate, and animated bar.
//
// What this does
// ──────────────
//   ComfyUI core emits `progress` websocket events for every node that
//   calls `comfy.utils.ProgressBar.update_absolute()`. The default
//   frontend paints a thin green fill at the top of the running node
//   but with NO text — on small/zoomed nodes the user can't tell whether
//   the bar is at 1 % or 99 %, and there is no ETA.
//
//   This extension hooks every node's draw pipeline and renders:
//     • a smooth animated bar across the node header
//     • a centred label  "  42 / 100 · 42% · ETA 00:18 · 4.2/s  "
//   The node TITLE is never written (L9.20 / L9.118): a title prefix leaked
//   into saved workflows, queued prompts (_meta.title) and marked every run
//   as a workflow change. Titles saved by older builds with the leaked
//   "⏳ NN% │ " prefix are repaired once on workflow load.
//
//   ETA is computed from a per-node rolling-window average of step time
//   (last 8 progress events). Rate (items/sec) is the inverse. Both are
//   recomputed only when a new progress event arrives, never per draw,
//   so the HUD stays cheap.
//
//   Works for ALL nodes (not only MEC) because every progress-aware
//   ComfyUI node uses the same websocket channel.
//
// Settings
// ────────
//   mec.progress_hud.enabled    — master toggle (default true)
//   mec.progress_hud.eta        — show ETA + rate (default true)
//   (mec.progress_hud.title_pct was removed with the title prefix; a stored
//    value is ignored.)
//
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { findNodeAnywhere, forAllNodes } from "./_subgraph_walk.js";
import { C, onThemeChange } from "./_c2c_theme.js";
import { getRuntime } from "./_c2c_runtime.js";
// Lite mode: this is an AMBIENT extension (no node depends on it), so in lite
// mode it must never register at all — its rAF loops, timers and draw hooks are
// then never installed. See _c2c_lite.js.
import { LITE } from "./_c2c_lite.js";

function _requestFgRedraw() {
    getRuntime().requestRedraw();
}

const SETTINGS = {
    enabled: true,
    eta: true,
};

// Canvas 2D fillStyle cannot parse `var(--x)` — it silently keeps the
// previous color, so the progress FILL was rendering as the dark track
// color (invisible / "damaged bar"). Resolve theme vars to concrete colors,
// cached so getComputedStyle isn't hit on every draw.
const _cssColorCache = new Map();
function _cssColor(varName, fallbackKey) {
    if (_cssColorCache.has(varName)) return _cssColorCache.get(varName);
    let out = C[fallbackKey] || "";
    try {
        const v = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
        if (v) out = v;
    } catch (_) { /* keep fallback */ }
    _cssColorCache.set(varName, out);
    return out;
}
onThemeChange(() => _cssColorCache.clear());

// Per-node WeakMap so lookups inside onDrawForeground are O(1) and don't
// depend on whether the user is currently viewing the root graph or a
// subgraph interior. Each entry is the same shape as the legacy
// PROGRESS map, plus an optional `bubble` flag set on ancestor wrapper
// SubgraphNodes so the bar shows up on the wrapper too.
const PROGRESS_BY_NODE = new WeakMap();

// node_id (string) -> {
//   value, max,                       // latest progress
//   started_ts, last_ts,              // ms timestamps
//   samples: [{ts, value}, ...],      // rolling window (max 16)
//   smoothed,                         // smoothed displayed value
// }
const PROGRESS = new Map();
let EXEC_NODE = null;

const SAMPLE_WINDOW = 16;

function _now() { return performance.now(); }

function _record(id, value, max) {
    const t = _now();
    let p = PROGRESS.get(id);
    if (!p) {
        p = { value: 0, max: max || 1, started_ts: t, last_ts: t,
              samples: [{ ts: t, value: 0 }], smoothed: 0 };
        PROGRESS.set(id, p);
    }
    p.value = value;
    p.max   = max || p.max || 1;
    p.last_ts = t;
    p.samples.push({ ts: t, value });
    if (p.samples.length > SAMPLE_WINDOW) p.samples.shift();

    // Mirror into the per-node WeakMap so the draw hook can find this
    // entry regardless of which graph (root or subgraph) is being
    // rendered. Also project a "bubble" entry onto every ancestor
    // SubgraphNode wrapper so the user sees progress on the wrapper
    // when they are at the parent level.
    const resolved = findNodeAnywhere(id);
    if (resolved) {
        PROGRESS_BY_NODE.set(resolved.node, p);
        for (const wrapper of resolved.wrapperChain) {
            // Use the same progress object reference so updates are
            // shared, but tag a property so the draw code can render a
            // subtler indicator on the wrapper.
            PROGRESS_BY_NODE.set(wrapper, p);
        }
    }
}

function _rate(p) {
    // items per second over the rolling window.
    if (!p || p.samples.length < 2) return 0;
    const a = p.samples[0];
    const b = p.samples[p.samples.length - 1];
    const dt = (b.ts - a.ts) / 1000;
    const dv = b.value - a.value;
    if (dt <= 0 || dv <= 0) return 0;
    return dv / dt;
}

function _eta_seconds(p) {
    const r = _rate(p);
    if (!r) return null;
    const remaining = Math.max(0, (p.max || 1) - p.value);
    return remaining / r;
}

function _fmt_time(secs) {
    if (secs == null || !isFinite(secs) || secs < 0) return "--:--";
    secs = Math.round(secs);
    if (secs < 3600) {
        const m = Math.floor(secs / 60);
        const s = secs % 60;
        return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    return `${h}:${String(m).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;
}

function _fmt_rate(r) {
    if (!r) return null;
    if (r >= 100) return `${r.toFixed(0)}/s`;
    if (r >= 10)  return `${r.toFixed(1)}/s`;
    if (r >= 1)   return `${r.toFixed(2)}/s`;
    return `${(60 * r).toFixed(1)}/min`;
}

// Older builds wrote "⏳ NN% │ " into node.title while a node ran; a save or
// autosave mid-run persisted it. Strip exactly that prefix (repeated, if a
// title was prefixed twice) and nothing else.
const LEAKED_TITLE_PREFIX = /^(?:⏳ \d{1,3}% │ )+/;
function _repairLeakedTitles() {
    try {
        forAllNodes((node) => {
            const t = node?.title;
            if (typeof t === "string" && LEAKED_TITLE_PREFIX.test(t)) {
                node.title = t.replace(LEAKED_TITLE_PREFIX, "");
            }
        });
    } catch (_) { /* never block a workflow load */ }
}

function _clear_progress(node_id) {
    const resolved = findNodeAnywhere(node_id);
    if (!resolved) return;
    for (const wrapper of resolved.wrapperChain) PROGRESS_BY_NODE.delete(wrapper);
    if (resolved.node) PROGRESS_BY_NODE.delete(resolved.node);
}

// ── Socket wiring ────────────────────────────────────────────────────
api.addEventListener("progress", (ev) => {
    if (!SETTINGS.enabled) return;
    const d = ev.detail || ev;
    if (!d || d.value == null || d.max == null) return;
    const id = String(d.node ?? d.node_id ?? EXEC_NODE ?? "");
    if (!id) return;
    _record(id, +d.value, +d.max || 1);
    _requestFgRedraw();
});

api.addEventListener("executing", (ev) => {
    const d = ev.detail || ev;
    const id = d?.node ?? d;
    EXEC_NODE = (id == null || id === "") ? null : String(id);
    if (EXEC_NODE === null) {
        // Workflow finished — clear overlays: drop each node's (and its
        // subgraph wrappers') PROGRESS_BY_NODE entry so no stale bar stays drawn.
        for (const id of PROGRESS.keys()) _clear_progress(id);
        PROGRESS.clear();
        _requestFgRedraw();
    }
});

api.addEventListener("executed", (ev) => {
    const d = ev.detail || ev;
    const id = String(d?.node ?? "");
    if (id) {
        PROGRESS.delete(id);
        _clear_progress(id);
    }
    _requestFgRedraw();
});

api.addEventListener("execution_error", () => {
    for (const id of PROGRESS.keys()) _clear_progress(id);
    PROGRESS.clear();
    _requestFgRedraw();
});

// Drive a steady redraw while ≥1 node is in progress so smoothing + ETA
// counter visibly tick down between socket events.
// PERF: every frame of this loop redraws the whole canvas. It used to run at
// 60 fps for the entire generation - while the GPU is busiest - even when the
// bar had long caught up and only the shimmer moved. Now: full rate only while
// a bar is gliding to a new value, ~12 fps otherwise (shimmer + ETA).
const IDLE_FRAME_MS = 83;
let _raf_handle = null;
let _idle_timer = null;
function _animate_loop() {
    _raf_handle = null;
    if (document.hidden) return;
    if (PROGRESS.size === 0) return;
    // Smooth interpolate displayed value toward true value.
    let moving = false;
    for (const p of PROGRESS.values()) {
        const target = p.value;
        const diff   = target - p.smoothed;
        if (Math.abs(diff) > 0.01) { p.smoothed += diff * 0.18; moving = true; }
        else p.smoothed = target;
    }
    _requestFgRedraw();
    if (moving) {
        _raf_handle = requestAnimationFrame(_animate_loop);
    } else {
        _idle_timer = setTimeout(() => {
            _idle_timer = null;
            _raf_handle = requestAnimationFrame(_animate_loop);
        }, IDLE_FRAME_MS);
    }
}
function _ensure_anim() {
    if (document.hidden) return;
    if (_raf_handle == null && _idle_timer == null && PROGRESS.size > 0) {
        _raf_handle = requestAnimationFrame(_animate_loop);
    }
}

try {
    document.addEventListener("visibilitychange", () => {
        if (!document.hidden && PROGRESS.size > 0) _ensure_anim();
    });
} catch (_) { /* headless */ }

app.registerExtension({
    name: "C2C.ProgressHUD",
    settings: [
        {
            id: "mec.progress_hud.enabled",
            name: "Progress HUD — enable per-node overlay",
            type: "boolean",
            defaultValue: true,
            onChange: (v) => {
                SETTINGS.enabled = !!v;
                if (!v) { for (const id of PROGRESS.keys()) _clear_progress(id); PROGRESS.clear(); }
            },
        },
        {
            id: "mec.progress_hud.eta",
            name: "Progress HUD — show ETA + rate",
            type: "boolean",
            defaultValue: true,
            onChange: (v) => { SETTINGS.eta = !!v; },
        },
    ],

    // Repair titles saved by older builds with the leaked progress prefix.
    afterConfigureGraph() { _repairLeakedTitles(); },

    async beforeRegisterNodeDef(nodeType /*, nodeData, app */) {
        const origDrawFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origDrawFg) origDrawFg.apply(this, arguments);
            if (!SETTINGS.enabled) return;
            if (this.flags?.collapsed) return;
            // Per-node WeakMap lookup works for top-level AND subgraph
            // children AND wrapper SubgraphNodes (mirrored on record).
            const p = PROGRESS_BY_NODE.get(this);
            if (!p) return;
            const v = (p.smoothed > 0 ? p.smoothed : p.value);
            const pct = Math.max(0, Math.min(100, (v / p.max) * 100));
            if (pct >= 100) return;

            _ensure_anim();

            // ── Animated bar across the title bar (above core green fill) ──
            const W = this.size[0];
            const barH = 4;
            const barY = -barH - 2;
            ctx.save();
            // Track
            ctx.fillStyle = "rgba(0,0,0,0.45)";
            ctx.fillRect(0, barY, W, barH);
            // Fill colour: green → yellow → orange by percentage. Theme vars
            // resolved to concrete colors (canvas can't read var()).
            const fillCol = pct < 50 ? _cssColor("--c2c-okSoft", "okSoft")
                          : pct < 85 ? _cssColor("--c2c-yellow", "yellow")
                          :            _cssColor("--c2c-peach",  "peach");
            ctx.fillStyle = fillCol;
            ctx.fillRect(0, barY, W * (pct / 100), barH);
            // Subtle moving shimmer ─ purely cosmetic.
            const shimmerX = ((_now() / 8) % W);
            const grd = ctx.createLinearGradient(shimmerX - 30, 0, shimmerX + 30, 0);
            grd.addColorStop(0, "rgba(255,255,255,0)");
            grd.addColorStop(0.5, "rgba(255,255,255,0.35)");
            grd.addColorStop(1, "rgba(255,255,255,0)");
            ctx.fillStyle = grd;
            ctx.fillRect(0, barY, W * (pct / 100), barH);

            // ── Centred label in title bar ───────────────────────────────
            ctx.font = "bold 11px ui-monospace, 'Cascadia Mono', monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";

            const parts = [`${Math.round(v)} / ${p.max}`, `${pct.toFixed(0)}%`];
            if (SETTINGS.eta) {
                const eta = _eta_seconds(p);
                if (eta != null) parts.push(`ETA ${_fmt_time(eta)}`);
                const r = _rate(p);
                const rs = _fmt_rate(r);
                if (rs) parts.push(rs);
            }
            const text = parts.join("  \u00b7  ");

            const cx = W / 2;
            const cy = -16;
            const w = ctx.measureText(text).width + 14;
            const h = 16;
            ctx.fillStyle = C.bg;
            ctx.globalAlpha = 0.85;
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(cx - w / 2, cy - h / 2, w, h, 6);
            else ctx.rect(cx - w / 2, cy - h / 2, w, h);
            ctx.fill();
            ctx.globalAlpha = 1;
            // Border
            ctx.strokeStyle = C.surface1;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = fillCol;
            ctx.fillText(text, cx, cy);
            ctx.restore();
        };
    },
});
