/**
 * c2c_ai_status_bar.js — Always-visible AI HUD (top-right).
 *
 * Pills:
 *   ⚡ active backend (id + tier color)   click → flyout
 *   $ daily cost / cap                    color shifts at 80%, red over cap
 *   ● online indicator (green ok / red broken / gray paused)
 *
 * Flyout panel (click any pill):
 *   - Per-backend health table
 *   - Per-feature current routing
 *   - "Refresh now" button
 *   - "Pause AI" toggle (stores c2c.ai.paused setting)
 *   - "Open Settings" link
 *
 * Polls /c2c/ai/status every 10s, and immediately after any /c2c/ai/* POST.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HUD_ID = "c2c-ai-hud";
const FLYOUT_ID = "c2c-ai-hud-flyout";
const STYLE_ID = "c2c-ai-hud-style";
const POLL_MS = 10_000;
const SETTING_PAUSE = "c2c.ai.paused";
const SETTING_HUD = "c2c.ai.hudVisible";

// Catppuccin Mocha palette
const C = {
    bg:    "#1e1e2e",
    bg2:   "#181825",
    fg:    "#cdd6f4",
    sub:   "#a6adc8",
    border:"#313244",
    red:   "#f38ba8",
    green: "#a6e3a1",
    yellow:"#f9e2af",
    blue:  "#89b4fa",
    mauve: "#cba6f7",
    teal:  "#94e2d5",
};

let _lastStatus = null;
let _pollTimer = null;
let _flyoutOpen = false;

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
#${HUD_ID} {
  position: fixed; top: 6px; right: 12px; z-index: 9999;
  display: flex; gap: 6px; align-items: center;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 11px; color: ${C.fg};
  user-select: none;
  pointer-events: auto;
}
#${HUD_ID} .pill {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 8px; border-radius: 10px;
  background: ${C.bg}; border: 1px solid ${C.border};
  cursor: pointer; line-height: 1.2;
  transition: background 0.15s, border-color 0.15s;
}
#${HUD_ID} .pill:hover { background: ${C.bg2}; border-color: ${C.mauve}; }
#${HUD_ID} .pill .dot {
  width: 7px; height: 7px; border-radius: 50%; background: ${C.sub};
  display: inline-block;
}
#${HUD_ID} .pill.cloud .dot { background: ${C.blue}; }
#${HUD_ID} .pill.local .dot { background: ${C.green}; }
#${HUD_ID} .pill.warn   { border-color: ${C.yellow}; color: ${C.yellow}; }
#${HUD_ID} .pill.danger { border-color: ${C.red};    color: ${C.red}; }
#${HUD_ID} .pill.paused { opacity: 0.55; border-style: dashed; }

#${FLYOUT_ID} {
  position: fixed; top: 36px; right: 12px; z-index: 9998;
  width: 360px; max-height: 70vh; overflow: auto;
  background: ${C.bg}; color: ${C.fg};
  border: 1px solid ${C.border}; border-radius: 8px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px;
  padding: 10px;
}
#${FLYOUT_ID} h4 {
  margin: 0 0 6px 0; color: ${C.mauve};
  font-size: 12px; font-weight: 600; letter-spacing: 0.4px;
  text-transform: uppercase;
}
#${FLYOUT_ID} table { width: 100%; border-collapse: collapse; font-size: 11px; }
#${FLYOUT_ID} td { padding: 3px 4px; border-bottom: 1px solid ${C.border}; }
#${FLYOUT_ID} td.ok { color: ${C.green}; }
#${FLYOUT_ID} td.bad { color: ${C.red}; }
#${FLYOUT_ID} button {
  background: ${C.bg2}; color: ${C.fg};
  border: 1px solid ${C.border}; border-radius: 4px;
  padding: 4px 10px; cursor: pointer; font-size: 11px;
  margin-right: 6px;
}
#${FLYOUT_ID} button:hover { border-color: ${C.mauve}; color: ${C.mauve}; }
#${FLYOUT_ID} .row { display: flex; gap: 6px; align-items: center; margin: 6px 0; }
#${FLYOUT_ID} .muted { color: ${C.sub}; }
`;
    document.head.appendChild(s);
}

function _hudRoot() {
    let root = document.getElementById(HUD_ID);
    if (!root) {
        root = document.createElement("div");
        root.id = HUD_ID;
        document.body.appendChild(root);
    }
    return root;
}

async function _fetchStatus() {
    try {
        const r = await fetch("/c2c/ai/status", { headers: { Accept: "application/json" }});
        if (!r.ok) return null;
        const j = await r.json();
        return j?.data || null;
    } catch (_) { return null; }
}

function _render(status) {
    const root = _hudRoot();
    root.innerHTML = "";

    const paused = !!app.ui?.settings?.getSettingValue(SETTING_PAUSE, false);
    const hudVisible = app.ui?.settings?.getSettingValue(SETTING_HUD, true);
    if (!hudVisible) { root.style.display = "none"; return; }
    root.style.display = "flex";

    if (!status || !Array.isArray(status.backends) || status.backends.length === 0) {
        const pill = document.createElement("div");
        pill.className = "pill warn";
        pill.title = "No AI backends configured — open Settings → C2C → AI Backends";
        pill.textContent = "AI: not configured";
        pill.onclick = () => app.ui?.settings?.show?.("C2C ▸ AI Backends");
        root.appendChild(pill);
        return;
    }

    // pick "active" = first healthy non-disabled backend
    const enabledHealthy = status.backends.filter(b => b.enabled && b.health?.ok);
    const active = enabledHealthy[0] || status.backends[0];

    // Backend pill
    const bPill = document.createElement("div");
    bPill.className = "pill " + (active.tier === "cloud" ? "cloud" : "local") + (paused ? " paused" : "");
    bPill.title = `${active.display_name}\nmodel: ${active.model}\nrtt: ${active.health?.last_rtt_ms ?? "?"}ms` +
                  (active.health?.last_error ? `\nerror: ${active.health.last_error}` : "");
    const dot = document.createElement("span"); dot.className = "dot";
    bPill.appendChild(dot);
    bPill.appendChild(document.createTextNode((paused ? "⏸ " : "") + active.display_name));
    bPill.onclick = _toggleFlyout;
    root.appendChild(bPill);

    // Cost pill
    const cost = status.cost || {};
    const frac = cost.fraction_used ?? 0;
    const cPill = document.createElement("div");
    cPill.className = "pill" + (cost.over_cap ? " danger" : (cost.over_warn ? " warn" : ""));
    cPill.title = `today ${(cost.today_cost_usd ?? 0).toFixed(4)} USD / cap ${(cost.cap_usd ?? 0).toFixed(2)} USD across ${cost.today_calls ?? 0} calls`;
    cPill.textContent = `$${(cost.today_cost_usd ?? 0).toFixed(3)} / ${(cost.cap_usd ?? 0).toFixed(2)}`;
    cPill.onclick = _toggleFlyout;
    root.appendChild(cPill);
}

function _toggleFlyout() {
    if (_flyoutOpen) { _closeFlyout(); return; }
    _openFlyout();
}

function _closeFlyout() {
    const f = document.getElementById(FLYOUT_ID);
    if (f) f.remove();
    _flyoutOpen = false;
    document.removeEventListener("mousedown", _outsideClose, true);
}

function _outsideClose(ev) {
    const f = document.getElementById(FLYOUT_ID);
    const h = document.getElementById(HUD_ID);
    if (!f) return;
    if (f.contains(ev.target) || (h && h.contains(ev.target))) return;
    _closeFlyout();
}

function _openFlyout() {
    _closeFlyout();
    const status = _lastStatus;
    const f = document.createElement("div");
    f.id = FLYOUT_ID;

    f.innerHTML = `<h4>AI Backends</h4>`;
    const tbl = document.createElement("table");
    (status?.backends || []).forEach(b => {
        const tr = document.createElement("tr");
        tr.innerHTML =
            `<td>${b.tier === "cloud" ? "☁" : "💻"}</td>` +
            `<td>${b.display_name}<div class="muted" style="font-size:10px">${b.model}</div></td>` +
            `<td class="${b.health?.ok ? "ok" : "bad"}">${b.health?.ok ? "ok" : "down"}</td>` +
            `<td class="muted">${b.health?.last_rtt_ms ?? "?"}ms</td>`;
        tbl.appendChild(tr);
    });
    f.appendChild(tbl);

    const cost = status?.cost || {};
    const costH = document.createElement("h4"); costH.textContent = "Cost (today)";
    costH.style.marginTop = "10px";
    f.appendChild(costH);
    const costDiv = document.createElement("div");
    costDiv.innerHTML =
        `<div>$${(cost.today_cost_usd ?? 0).toFixed(4)} / cap $${(cost.cap_usd ?? 0).toFixed(2)} ` +
        `<span class="muted">(${cost.today_calls ?? 0} calls)</span></div>` +
        `<div class="muted" style="font-size:10px">day: ${cost.day_key ?? "?"}</div>`;
    f.appendChild(costDiv);

    const row = document.createElement("div"); row.className = "row";
    const btnRefresh = document.createElement("button");
    btnRefresh.textContent = "Refresh";
    btnRefresh.onclick = async () => {
        btnRefresh.disabled = true; btnRefresh.textContent = "…";
        await fetch("/c2c/ai/probe", { method: "POST" });
        await refresh();
        _closeFlyout(); _openFlyout();
    };
    row.appendChild(btnRefresh);

    const paused = !!app.ui?.settings?.getSettingValue(SETTING_PAUSE, false);
    const btnPause = document.createElement("button");
    btnPause.textContent = paused ? "Resume AI" : "Pause AI";
    btnPause.onclick = () => {
        app.ui?.settings?.setSettingValue(SETTING_PAUSE, !paused);
        _closeFlyout();
        refresh();
    };
    row.appendChild(btnPause);

    const btnSettings = document.createElement("button");
    btnSettings.textContent = "Settings";
    btnSettings.onclick = () => {
        app.ui?.settings?.show?.("C2C ▸ AI Backends");
        _closeFlyout();
    };
    row.appendChild(btnSettings);

    f.appendChild(row);
    document.body.appendChild(f);
    _flyoutOpen = true;
    setTimeout(() => document.addEventListener("mousedown", _outsideClose, true), 0);
}

async function refresh() {
    _lastStatus = await _fetchStatus();
    _render(_lastStatus);
}

app.registerExtension({
    name: "c2c.ai.statusBar",
    settings: [
        { id: SETTING_HUD, name: "C2C ▸ AI ▸ Show top-right status HUD",
          type: "boolean", default: true,
          onChange: () => refresh() },
        { id: SETTING_PAUSE, name: "C2C ▸ AI ▸ Pause all AI calls",
          type: "boolean", default: false,
          onChange: () => refresh() },
    ],
    async setup() {
        _injectStyle();
        await refresh();
        if (_pollTimer) clearInterval(_pollTimer);
        _pollTimer = setInterval(refresh, POLL_MS);
    },
});

// Expose for other modules to nudge the HUD after they do something
window.__C2C_AI_HUD__ = { refresh };
