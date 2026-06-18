/**
 * mec_system_hud.js — P10.3 Workspace System HUD
 *
 * Floating chip pinned to the bottom-right of the canvas that surfaces
 * three workspace-wide metrics the existing per-node and complexity HUDs
 * don't cover:
 *
 *   1. Queue depth     — items waiting + running, from /queue.
 *   2. VRAM usage      — from /system_stats (refreshed every 3s).
 *   3. AI cost today   — USD spent against c2c_ai router, from /c2c/ai/cost.
 *
 * Click the chip to expand it into a fuller card listing per-device
 * VRAM and the daily cost cap. The chip is intentionally low-traffic
 * (3s polling) so it costs ~zero perf even on small machines.
 *
 * Settings:
 *   mec.system_hud.enabled  — bool (default true)
 *   mec.system_hud.vram     — bool (default true)
 *   mec.system_hud.cost     — bool (default true)
 *
 * The HUD coexists with mec_complexity_hud.js (top-center) and
 * mec_progress_hud.js (per-node title bar).
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const HUD_ID   = "mec-system-hud";
const STYLE_ID = "mec-system-hud-style";
const POLL_MS  = 3000;

const SETTINGS = { enabled: true, vram: true, cost: true };

let _hud = null;
let _expanded = false;
let _timer = null;
const STATE = {
    queue_running: 0,
    queue_pending: 0,
    vram_used_gb: 0,
    vram_total_gb: 0,
    vram_label: "",
    cost_today_usd: 0,
    cost_cap_usd: null,
    devices: [],
    ai_enabled: false,
    // iter-4 extended metrics
    gpu_util_pct: null,        // primary GPU utilisation % (0-100)
    gpu_temp_c:   null,        // primary GPU temperature °C
    gpu_name:     "",
    cpu_util_pct: 0,           // CPU utilisation %
    cpu_logical:  0,
    ram_used_gb:  0,
    ram_total_gb: 0,
    ram_pct:      0,
    metrics_available: false,
    metrics_source: "none",    // pynvml | nvidia-smi | none
    last_run_sec: null,        // seconds for the most recent finished run
    last_run_node_count: 0,
    _run_started_at: 0,        // perf.now() of execution_start
    ws_lat_ms: null,           // round-trip latency to /queue
    graph_node_count: 0,
    graph_link_count: 0,
};

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
#${HUD_ID} {
    position: fixed;
    right: 10px;
    /* Raised well above ComfyUI's bottom status bar (FPS + VRAM/RAM
       badges + queue indicator). Native bar is ~52px on the modern Vue
       UI; keep a 12px safety gap so we never overlap. */
    bottom: 64px;
    z-index: var(--c2c-z-hud, 1000);
    background: var(--c2c-bg);
    color: var(--c2c-fg);
    border: 1px solid var(--c2c-surface0);
    border-radius: 8px;
    font-family: ui-monospace, "Cascadia Mono", monospace;
    font-size: 11px;
    line-height: 1.35;
    padding: 5px 10px;
    cursor: pointer;
    user-select: none;
    box-shadow: 0 2px 6px rgba(0,0,0,0.45);
    display: flex;
    gap: 10px;
    align-items: center;
    pointer-events: auto;
}
#${HUD_ID}.expanded {
    flex-direction: column;
    align-items: stretch;
    padding: 8px 12px;
    min-width: 220px;
    gap: 6px;
    font-size: 12px;
}
#${HUD_ID} .sys-cell { white-space: nowrap; }
#${HUD_ID} .sys-cell b { color: var(--c2c-blue); }
#${HUD_ID} .sys-warn   { color: var(--c2c-peach); }
#${HUD_ID} .sys-danger { color: var(--c2c-red); }
#${HUD_ID} .sys-ok     { color: var(--c2c-okSoft); }
#${HUD_ID} h4 {
    margin: 0 0 2px 0; font-size: 11px; color: var(--c2c-slate400);
    text-transform: uppercase; letter-spacing: 0.4px;
}
#${HUD_ID} .sys-row {
    display: flex; justify-content: space-between; gap: 12px;
}
    `.trim();
    document.head.appendChild(s);
}

function _ensureHud() {
    if (_hud) return _hud;
    _injectStyle();
    _hud = document.createElement("div");
    _hud.id = HUD_ID;
    _hud.title = "Click to expand / collapse";
    _hud.addEventListener("click", () => {
        _expanded = !_expanded;
        _hud.classList.toggle("expanded", _expanded);
        _render();
    });
    document.body.appendChild(_hud);
    // Register with the dock-anchor so the chip auto-shifts up when a
    // bottom panel (image feed, queue tab, splitter) extends over our
    // base offset. See `mec_dock_anchor.js`.
    try { window.__mecDock?.register?.(_hud, { baseBottom: 64 }); } catch (__c2cErr) { __c2cReport("c2c_system_hud", __c2cErr); }
    return _hud;
}

function _hide() {
    if (_hud) {
        try { window.__mecDock?.unregister?.(_hud); } catch (__c2cErr) { __c2cReport("c2c_system_hud", __c2cErr); }
        _hud.remove();
        _hud = null;
    }
}

function _vramTone(used, total) {
    if (!total) return "sys-cell";
    const r = used / total;
    if (r > 0.92) return "sys-cell sys-danger";
    if (r > 0.78) return "sys-cell sys-warn";
    return "sys-cell sys-ok";
}

function _costTone(spent, cap) {
    if (!cap || cap <= 0) return "sys-cell";
    const r = spent / cap;
    if (r > 0.95) return "sys-cell sys-danger";
    if (r > 0.75) return "sys-cell sys-warn";
    return "sys-cell sys-ok";
}

function _render() {
    if (!_hud) return;
    const qd = STATE.queue_running + STATE.queue_pending;
    const qStr = STATE.queue_running > 0
        ? `<b>Q</b> ${STATE.queue_running}\u25B6 +${STATE.queue_pending}`
        : `<b>Q</b> ${qd}`;
    const vramStr = SETTINGS.vram && STATE.vram_total_gb > 0
        ? `<span class="${_vramTone(STATE.vram_used_gb, STATE.vram_total_gb)}">
             <b>VRAM</b> ${STATE.vram_used_gb.toFixed(1)} / ${STATE.vram_total_gb.toFixed(1)}G
           </span>`
        : "";
    const costStr = SETTINGS.cost && STATE.ai_enabled
        ? `<span class="${_costTone(STATE.cost_today_usd, STATE.cost_cap_usd)}">
             <b>AI</b> $${STATE.cost_today_usd.toFixed(3)}${STATE.cost_cap_usd != null ? "/$" + STATE.cost_cap_usd.toFixed(2) : ""}
           </span>`
        : "";

    if (!_expanded) {
        _hud.innerHTML = `
            <span class="sys-cell">${qStr}</span>
            ${vramStr}
            ${costStr}
        `;
        return;
    }
    // Expanded card
    const deviceRows = (STATE.devices || []).map(d => {
        const u = d.vram_used_gb || 0, t = d.vram_total_gb || 0;
        const cls = _vramTone(u, t);
        return `<div class="sys-row"><span>${d.name}</span>
                <span class="${cls}">${u.toFixed(1)} / ${t.toFixed(1)} GB</span></div>`;
    }).join("") || `<div class="sys-row"><span style="color:var(--c2c-overlay0)">No device data</span></div>`;
    _hud.innerHTML = `
        <h4>Queue</h4>
        <div class="sys-row">
            <span>Running</span><span><b>${STATE.queue_running}</b></span>
        </div>
        <div class="sys-row">
            <span>Pending</span><span><b>${STATE.queue_pending}</b></span>
        </div>
        <h4 style="margin-top:6px">VRAM</h4>
        ${deviceRows}
        ${SETTINGS.cost && STATE.ai_enabled ? `
        <h4 style="margin-top:6px">AI Cost (today)</h4>
        <div class="sys-row">
            <span class="${_costTone(STATE.cost_today_usd, STATE.cost_cap_usd)}">$${STATE.cost_today_usd.toFixed(4)}</span>
            <span style="color:var(--c2c-overlay0)">${STATE.cost_cap_usd != null ? "cap $" + STATE.cost_cap_usd.toFixed(2) : "no cap"}</span>
        </div>
        ` : ""}
    `;
}

async function _pollMetrics() {
    try {
        const r = await api.fetchApi("/c2c/sys/metrics");
        if (!r.ok) return;
        const d = await r.json();
        if (!d || d.ok === false) return;
        const g = d.gpu || {};
        STATE.metrics_available = !!g.available;
        STATE.metrics_source = String(d.source || "none");
        if (g.available) {
            STATE.gpu_util_pct = (g.util_pct ?? null);
            STATE.gpu_temp_c   = (g.temp_c   ?? null);
            STATE.gpu_name     = String(g.name || "");
            // Refresh VRAM via metrics endpoint too — fresher than /system_stats
            if (typeof g.vram_used_gb === "number" && typeof g.vram_total_gb === "number") {
                STATE.vram_used_gb  = g.vram_used_gb;
                STATE.vram_total_gb = g.vram_total_gb;
            }
            if (Array.isArray(g.devices) && g.devices.length > 0) {
                STATE.devices = g.devices.map(x => ({
                    name: x.name || "GPU",
                    vram_used_gb: +x.vram_used_gb || 0,
                    vram_total_gb: +x.vram_total_gb || 0,
                    util_pct: x.util_pct,
                    temp_c: x.temp_c,
                }));
            }
        }
        const c = d.cpu || {}, m = d.ram || {};
        STATE.cpu_util_pct = +c.util_pct || 0;
        STATE.cpu_logical  = +c.logical  || 0;
        STATE.ram_used_gb  = +m.used_gb  || 0;
        STATE.ram_total_gb = +m.total_gb || 0;
        STATE.ram_pct      = +m.pct      || 0;
    } catch { /* metrics endpoint unavailable */ }
}

// Round-trip latency to the ComfyUI server (used for the ws_lat chip).
async function _pollLatency() {
    const t0 = (typeof performance !== "undefined") ? performance.now() : Date.now();
    try {
        const r = await api.fetchApi("/queue");
        if (r.ok) {
            await r.text();
            const t1 = (typeof performance !== "undefined") ? performance.now() : Date.now();
            STATE.ws_lat_ms = Math.round(t1 - t0);
        }
    } catch { /* offline */ }
}

function _refreshGraphSize() {
    try {
        const g = app?.graph;
        if (!g) return;
        STATE.graph_node_count = Array.isArray(g._nodes) ? g._nodes.length : 0;
        STATE.graph_link_count = (g.links && typeof g.links === "object")
            ? Object.keys(g.links).length
            : 0;
    } catch { /* graph not ready */ }
}

async function _pollQueue() {
    try {
        const r = await api.fetchApi("/queue");
        if (!r.ok) return;
        const data = await r.json();
        STATE.queue_running = (data.queue_running || []).length;
        STATE.queue_pending = (data.queue_pending || []).length;
    } catch { /* network noise — silent */ }
}

async function _pollVram() {
    if (!SETTINGS.vram) return;
    try {
        const r = await api.fetchApi("/system_stats");
        if (!r.ok) return;
        const data = await r.json();
        const devs = data.devices || [];
        STATE.devices = devs.map(d => ({
            name: d.name || "GPU",
            vram_used_gb: (d.vram_total - d.vram_free) / 1073741824,
            vram_total_gb: d.vram_total / 1073741824,
        }));
        if (STATE.devices.length > 0) {
            // headline = primary GPU (#0)
            STATE.vram_used_gb  = STATE.devices[0].vram_used_gb;
            STATE.vram_total_gb = STATE.devices[0].vram_total_gb;
        }
    } catch { /* not all backends expose /system_stats */ }
}

async function _pollCost() {
    if (!SETTINGS.cost) return;
    try {
        const r = await api.fetchApi("/c2c/ai/cost");
        if (!r.ok) { STATE.ai_enabled = false; return; }
        const data = await r.json();
        STATE.ai_enabled    = true;
        STATE.cost_today_usd = +data.spent_today_usd || +data.today_usd || 0;
        STATE.cost_cap_usd   = data.daily_cap_usd != null
            ? +data.daily_cap_usd
            : (data.cap_usd != null ? +data.cap_usd : null);
    } catch { STATE.ai_enabled = false; }
}

async function _tick() {
    if (!SETTINGS.enabled) return;
    _refreshGraphSize();
    await Promise.all([_pollQueue(), _pollVram(), _pollCost(), _pollMetrics(), _pollLatency()]);
    _render();
    _publishToStrip();
}

// Iteration 2: publish polled values into window.C2CStatusStrip so the
// OmniPill Stats section's all-stats slot can render them. The legacy
// bottom-right #mec-system-hud element is CSS-hidden by _c2c_theme.js;
// keeping its poll loop alive makes data flow through the registry.
function _publishToStrip() {
    const strip = (typeof window !== "undefined") ? window.C2CStatusStrip : null;
    if (!strip || typeof strip.register !== "function") return;
    try {
        const qTotal = (STATE.queue_running | 0) + (STATE.queue_pending | 0);
        strip.register({
            id: "queue",
            label: "Q",
            value: String(qTotal),
            state: qTotal === 0 ? "idle" : (qTotal > 5 ? "warn" : "ok"),
            tooltip: `Queue — ${STATE.queue_running} running / ${STATE.queue_pending} pending`,
            priority: 20,
        });
        if (SETTINGS.vram && STATE.vram_total_gb > 0) {
            const u = STATE.vram_used_gb;
            const t = STATE.vram_total_gb;
            const ratio = u / Math.max(t, 0.001);
            strip.register({
                id: "vram",
                label: "VRAM",
                value: `${u.toFixed(1)}/${t.toFixed(1)}G`,
                state: ratio > 0.92 ? "err" : ratio > 0.78 ? "warn" : "ok",
                tooltip: `Primary GPU VRAM — ${(ratio * 100).toFixed(0)}%`,
                priority: 30,
            });
        }
        if (SETTINGS.cost && STATE.ai_enabled) {
            const v = STATE.cost_today_usd || 0;
            const cap = STATE.cost_cap_usd;
            const ratio = cap ? v / Math.max(cap, 0.000001) : 0;
            strip.register({
                id: "cost",
                label: "AI $",
                value: `$${v.toFixed(3)}`,
                state: ratio > 0.95 ? "err" : ratio > 0.75 ? "warn" : "ok",
                tooltip: cap
                    ? `AI cost today — $${v.toFixed(4)} / cap $${cap.toFixed(2)}`
                    : `AI cost today — $${v.toFixed(4)} (no cap)`,
                priority: 40,
            });
        }
        // iter-4: GPU utilisation %
        if (STATE.metrics_available && STATE.gpu_util_pct != null) {
            const u = STATE.gpu_util_pct;
            strip.register({
                id: "gpu",
                label: "GPU",
                value: `${u}%`,
                state: u >= 95 ? "warn" : u >= 60 ? "ok" : "idle",
                tooltip: `${STATE.gpu_name || "Primary GPU"} utilisation — ${u}%`,
                priority: 32,
            });
        }
        // iter-4: GPU temperature °C
        if (STATE.metrics_available && STATE.gpu_temp_c != null) {
            const t = STATE.gpu_temp_c;
            strip.register({
                id: "gpu_temp",
                label: "GPU °C",
                value: `${t}°`,
                state: t >= 88 ? "err" : t >= 80 ? "warn" : "ok",
                tooltip: `${STATE.gpu_name || "Primary GPU"} temperature — ${t} °C`,
                priority: 34,
            });
        }
        // iter-4: RAM used/total
        if (STATE.ram_total_gb > 0) {
            strip.register({
                id: "ram",
                label: "RAM",
                value: `${STATE.ram_used_gb.toFixed(1)}/${STATE.ram_total_gb.toFixed(1)}G`,
                state: STATE.ram_pct >= 92 ? "err" : STATE.ram_pct >= 80 ? "warn" : "ok",
                tooltip: `System RAM — ${STATE.ram_pct}%`,
                priority: 36,
            });
        }
        // iter-4: CPU %
        if (STATE.cpu_logical > 0) {
            const c = STATE.cpu_util_pct;
            strip.register({
                id: "cpu",
                label: "CPU",
                value: `${c.toFixed(0)}%`,
                state: c >= 95 ? "err" : c >= 80 ? "warn" : "ok",
                tooltip: `CPU — ${c.toFixed(1)}% over ${STATE.cpu_logical} logical cores`,
                priority: 38,
            });
        }
        // iter-4: last run duration
        if (STATE.last_run_sec != null) {
            const s = STATE.last_run_sec;
            const v = s >= 60 ? `${Math.floor(s/60)}m${(s%60).toFixed(0)}s` : `${s.toFixed(1)}s`;
            strip.register({
                id: "last_run",
                label: "RUN",
                value: v,
                state: "idle",
                tooltip: `Last finished run — ${s.toFixed(2)}s` +
                    (STATE.last_run_node_count ? ` over ${STATE.last_run_node_count} nodes` : ""),
                priority: 50,
            });
        }
        // iter-4: graph size (node + link count)
        if (STATE.graph_node_count > 0) {
            strip.register({
                id: "graph_size",
                label: "GRAPH",
                value: `${STATE.graph_node_count}n/${STATE.graph_link_count}l`,
                state: "idle",
                tooltip: `Workflow size — ${STATE.graph_node_count} nodes, ${STATE.graph_link_count} links`,
                priority: 60,
            });
        }
        // iter-4: WS uplink latency
        if (STATE.ws_lat_ms != null) {
            const l = STATE.ws_lat_ms;
            strip.register({
                id: "ws_lat",
                label: "LAT",
                value: `${l}ms`,
                state: l >= 400 ? "err" : l >= 150 ? "warn" : "ok",
                tooltip: `Round-trip latency to ComfyUI — ${l} ms`,
                priority: 70,
            });
        }
    } catch (e) {
        console.warn("[C2CSystemHUD] publish to strip failed:", e);
    }
}

// Hook ComfyUI execution events to time the most recent run.
function _hookRunTimer() {
    if (window.__C2C_RUN_TIMER_HOOKED) return;
    window.__C2C_RUN_TIMER_HOOKED = true;
    try {
        api.addEventListener("execution_start", () => {
            STATE._run_started_at = (typeof performance !== "undefined") ? performance.now() : Date.now();
        });
        const _end = () => {
            const t0 = STATE._run_started_at;
            if (!t0) return;
            const t1 = (typeof performance !== "undefined") ? performance.now() : Date.now();
            STATE.last_run_sec = Math.max(0, (t1 - t0) / 1000);
            STATE._run_started_at = 0;
            try {
                const g = app?.graph;
                STATE.last_run_node_count = (g && Array.isArray(g._nodes)) ? g._nodes.length : 0;
            } catch { /* */ }
            _publishToStrip();
        };
        api.addEventListener("execution_success", _end);
        api.addEventListener("execution_error", _end);
    } catch (e) {
        console.warn("[C2CSystemHUD] could not hook run timer:", e);
    }
}

function _start() {
    if (_timer) return;
    _ensureHud();
    _hookRunTimer();
    _tick();
    _timer = setInterval(_tick, POLL_MS);
    if (typeof window !== "undefined") {
        // Cleanup on unload so reload doesn't leak the interval.
        window.__MEC_SYSTEM_HUD_INTERVAL = _timer;
        window.addEventListener("beforeunload", _stop, { once: true });
    }
}

function _stop() {
    if (_timer) { clearInterval(_timer); _timer = null; }
    _hide();
}

app.registerExtension({
    name: "C2C.SystemHUD",
    settings: [
        {
            id: "mec.system_hud.enabled",
            name: "System HUD \u2014 show queue/VRAM/cost chip",
            type: "boolean",
            default: true,
            onChange: (v) => {
                SETTINGS.enabled = !!v;
                if (v) _start(); else _stop();
            },
        },
        {
            id: "mec.system_hud.vram",
            name: "System HUD \u2014 include VRAM usage",
            type: "boolean",
            default: true,
            onChange: (v) => { SETTINGS.vram = !!v; },
        },
        {
            id: "mec.system_hud.cost",
            name: "System HUD \u2014 include AI cost",
            type: "boolean",
            default: true,
            onChange: (v) => { SETTINGS.cost = !!v; },
        },
    ],
    async setup() {
        // Hydrate from settings (in case user toggled off then reloaded).
        try {
            SETTINGS.enabled = app.ui?.settings?.getSettingValue?.("mec.system_hud.enabled", true) !== false;
            SETTINGS.vram    = app.ui?.settings?.getSettingValue?.("mec.system_hud.vram",    true) !== false;
            SETTINGS.cost    = app.ui?.settings?.getSettingValue?.("mec.system_hud.cost",    true) !== false;
        } catch { /* settings unavailable mid-init */ }
        if (SETTINGS.enabled) _start();
    },
});
