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
//     • a node-title prefix while running ("⏳ 42% ")
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
//   mec.progress_hud.title_pct  — prefix node title with " ⏳ NN% "
//                                 while executing (default true)
//
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SETTINGS = {
    enabled: true,
    eta: true,
    title_pct: true,
};

// node_id (string) -> {
//   value, max,                       // latest progress
//   started_ts, last_ts,              // ms timestamps
//   samples: [{ts, value}, ...],      // rolling window (max 16)
//   smoothed,                         // smoothed displayed value
// }
const PROGRESS = new Map();
let EXEC_NODE = null;
// node_id -> original title (so we can restore it cleanly)
const ORIG_TITLES = new Map();

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

// Title prefix management.
function _apply_title_prefix(node, pct) {
    if (!SETTINGS.title_pct) return;
    if (!ORIG_TITLES.has(node.id)) ORIG_TITLES.set(node.id, node.title);
    const orig = ORIG_TITLES.get(node.id);
    const tag  = `\u23F3 ${pct.toFixed(0)}% \u2502 `;
    if (!node.title.startsWith(tag.slice(0, 2))) {
        node.title = tag + orig;
    } else {
        node.title = tag + orig;
    }
}

function _clear_title_prefix(node_id) {
    const g = app.graph;
    if (!g) return;
    const node = g.getNodeById?.(Number(node_id)) || null;
    if (!node) { ORIG_TITLES.delete(node_id); return; }
    if (ORIG_TITLES.has(node_id)) {
        node.title = ORIG_TITLES.get(node_id);
        ORIG_TITLES.delete(node_id);
    }
}

function _clear_all_title_prefixes() {
    for (const id of Array.from(ORIG_TITLES.keys())) _clear_title_prefix(id);
}

// ── Socket wiring ────────────────────────────────────────────────────
api.addEventListener("progress", (ev) => {
    if (!SETTINGS.enabled) return;
    const d = ev.detail || ev;
    if (!d || d.value == null || d.max == null) return;
    const id = String(d.node ?? d.node_id ?? EXEC_NODE ?? "");
    if (!id) return;
    _record(id, +d.value, +d.max || 1);
    if (app.graph) app.graph.setDirtyCanvas(true, true);
});

api.addEventListener("executing", (ev) => {
    const d = ev.detail || ev;
    const id = d?.node ?? d;
    EXEC_NODE = (id == null || id === "") ? null : String(id);
    if (EXEC_NODE === null) {
        // Workflow finished — clear overlays + title prefixes.
        PROGRESS.clear();
        _clear_all_title_prefixes();
        if (app.graph) app.graph.setDirtyCanvas(true, true);
    }
});

api.addEventListener("executed", (ev) => {
    const d = ev.detail || ev;
    const id = String(d?.node ?? "");
    if (id) {
        PROGRESS.delete(id);
        _clear_title_prefix(id);
    }
    if (app.graph) app.graph.setDirtyCanvas(true, true);
});

api.addEventListener("execution_error", () => {
    PROGRESS.clear();
    _clear_all_title_prefixes();
    if (app.graph) app.graph.setDirtyCanvas(true, true);
});

// Drive a steady redraw while ≥1 node is in progress so smoothing + ETA
// counter visibly tick down between socket events.
let _raf_handle = null;
function _animate_loop() {
    _raf_handle = null;
    if (PROGRESS.size === 0) return;
    // Smooth interpolate displayed value toward true value.
    const t = _now();
    for (const p of PROGRESS.values()) {
        const target = p.value;
        const diff   = target - p.smoothed;
        if (Math.abs(diff) > 0.01) p.smoothed += diff * 0.18;
        else p.smoothed = target;
    }
    if (app.graph) app.graph.setDirtyCanvas(true, true);
    _raf_handle = requestAnimationFrame(_animate_loop);
}
function _ensure_anim() {
    if (_raf_handle == null && PROGRESS.size > 0) {
        _raf_handle = requestAnimationFrame(_animate_loop);
    }
}

app.registerExtension({
    name: "MEC.ProgressHUD",
    settings: [
        {
            id: "mec.progress_hud.enabled",
            name: "Progress HUD — enable per-node overlay",
            type: "boolean",
            default: true,
            onChange: (v) => { SETTINGS.enabled = !!v; if (!v) PROGRESS.clear(); },
        },
        {
            id: "mec.progress_hud.eta",
            name: "Progress HUD — show ETA + rate",
            type: "boolean",
            default: true,
            onChange: (v) => { SETTINGS.eta = !!v; },
        },
        {
            id: "mec.progress_hud.title_pct",
            name: "Progress HUD — prefix node title with % while running",
            type: "boolean",
            default: true,
            onChange: (v) => {
                SETTINGS.title_pct = !!v;
                if (!v) _clear_all_title_prefixes();
            },
        },
    ],

    async beforeRegisterNodeDef(nodeType /*, nodeData, app */) {
        const origDrawFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origDrawFg) origDrawFg.apply(this, arguments);
            if (!SETTINGS.enabled) return;
            if (this.flags?.collapsed) return;
            const id = String(this.id);
            const p = PROGRESS.get(id);
            if (!p) return;
            const v = (p.smoothed > 0 ? p.smoothed : p.value);
            const pct = Math.max(0, Math.min(100, (v / p.max) * 100));
            if (pct >= 100) return;

            _ensure_anim();
            _apply_title_prefix(this, pct);

            // ── Animated bar across the title bar (above core green fill) ──
            const W = this.size[0];
            const barH = 4;
            const barY = -barH - 2;
            ctx.save();
            // Track
            ctx.fillStyle = "rgba(0,0,0,0.45)";
            ctx.fillRect(0, barY, W, barH);
            // Fill colour: green → yellow → orange by percentage.
            const fillCol = pct < 50 ? "#a6e3a1" : pct < 85 ? "#f9e2af" : "#fab387";
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
            ctx.fillStyle = "rgba(30,30,46,0.85)";  // Catppuccin base
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(cx - w / 2, cy - h / 2, w, h, 6);
            else ctx.rect(cx - w / 2, cy - h / 2, w, h);
            ctx.fill();
            // Border
            ctx.strokeStyle = "rgba(69,71,90,0.9)";  // surface1
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = fillCol;
            ctx.fillText(text, cx, cy);
            ctx.restore();
        };
    },
});
