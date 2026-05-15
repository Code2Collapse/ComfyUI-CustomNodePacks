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
    z-index: 99996;
    background: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #313244;
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
#${HUD_ID} .sys-cell b { color: #89b4fa; }
#${HUD_ID} .sys-warn   { color: #fab387; }
#${HUD_ID} .sys-danger { color: #f38ba8; }
#${HUD_ID} .sys-ok     { color: #a6e3a1; }
#${HUD_ID} h4 {
    margin: 0 0 2px 0; font-size: 11px; color: #94a3b8;
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
    return _hud;
}

function _hide() {
    if (_hud) { _hud.remove(); _hud = null; }
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
    }).join("") || `<div class="sys-row"><span style="color:#6c7086">No device data</span></div>`;
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
            <span style="color:#6c7086">${STATE.cost_cap_usd != null ? "cap $" + STATE.cost_cap_usd.toFixed(2) : "no cap"}</span>
        </div>
        ` : ""}
    `;
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
    await Promise.all([_pollQueue(), _pollVram(), _pollCost()]);
    _render();
}

function _start() {
    if (_timer) return;
    _ensureHud();
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
    name: "MEC.SystemHUD",
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
