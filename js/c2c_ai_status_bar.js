/**
 * c2c_ai_status_bar.js — AI status surface, hosted in OmniBar `ai` slot.
 *
 * Phase 4 (P0.2) migration (2026-05-25):
 *   • The legacy top-right floating HUD (#c2c-ai-hud + #c2c-ai-hud-flyout)
 *     has been retired. The AI chip is now a single OmniBar slot pill
 *     registered into section "ai" via window.C2COmniBar.register().
 *   • One chip (collapsed = dot; expanded = backend + cost). Clicking the
 *     chip opens an anchored popover containing the per-backend table,
 *     cost row, Refresh / Pause / Settings actions.
 *
 * Polls /c2c/ai/status every 10s, and immediately after any user-driven
 * /c2c/ai/probe POST. Exposes window.__C2C_AI_HUD__.refresh() so sibling
 * modules (router, prompts, error translator) can nudge the chip after a
 * call lands.
 *
 * License: Apache-2.0
 */

import { app } from "../../scripts/app.js";

const CHIP_ID    = "c2c-ai-chip";
const POPOVER_ID = "c2c-ai-popover";
const STYLE_ID   = "c2c-ai-hud-style";
const POLL_MS    = 10_000;
const SETTING_PAUSE = "c2c.ai.paused";
const SETTING_HUD   = "c2c.ai.hudVisible";

let _lastStatus = null;
let _pollTimer = null;
let _popoverOpen = false;
let _chipEl = null;
let _chipDot = null;
let _chipLabel = null;
let _chipCost = null;
let _slotMode = "full";
let _unregisterSlot = null;

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
#${CHIP_ID} {
  display: inline-flex; align-items: center; gap: 6px;
  height: 22px; padding: 0 9px; border-radius: 999px;
  cursor: pointer; user-select: none;
  background: color-mix(in srgb, var(--c2c-bg, var(--c2c-bg2)) 88%, transparent);
  border: 1px solid color-mix(in srgb, var(--c2c-border, var(--c2c-surface0)) 70%, transparent);
  color: var(--c2c-fg, var(--c2c-accentBright));
  font: 11px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  transition: background 0.12s, border-color 0.12s;
  white-space: nowrap;
}
#${CHIP_ID}:hover {
  background: color-mix(in srgb, var(--c2c-bg2, var(--c2c-panelTint)) 92%, transparent);
  border-color: color-mix(in srgb, var(--c2c-mauve, var(--c2c-violetSoft)) 50%, transparent);
}
#${CHIP_ID} .ai-dot {
  width: 7px; height: 7px; border-radius: 50%; flex: 0 0 7px;
  background: var(--c2c-sub, var(--c2c-gray400));
}
#${CHIP_ID}[data-tier="cloud"] .ai-dot { background: var(--c2c-blue, var(--c2c-accentSoft)); }
#${CHIP_ID}[data-tier="local"] .ai-dot { background: var(--c2c-green, var(--c2c-okSoft2)); }
#${CHIP_ID}[data-state="warn"]  { color: var(--c2c-yellow, var(--c2c-warn)); border-color: color-mix(in srgb, var(--c2c-yellow, var(--c2c-warn)) 45%, transparent); }
#${CHIP_ID}[data-state="danger"]{ color: var(--c2c-red,    var(--c2c-danger)); border-color: color-mix(in srgb, var(--c2c-red,    var(--c2c-danger)) 50%, transparent); }
#${CHIP_ID}[data-state="paused"]{ opacity: 0.55; border-style: dashed; }
#${CHIP_ID} .ai-label { font-weight: 500; }
#${CHIP_ID} .ai-cost  { opacity: 0.78; font-variant-numeric: tabular-nums; }
#${CHIP_ID}[data-c2c-mode="icon"] .ai-label,
#${CHIP_ID}[data-c2c-mode="icon"] .ai-cost { display: none; }
#${CHIP_ID}[data-c2c-mode="icon"] { padding: 0 6px; gap: 0; }

#${POPOVER_ID} {
  position: fixed; z-index: var(--c2c-z-popover, 9000);
  min-width: 320px; max-width: 380px; max-height: 70vh; overflow: auto;
  background: color-mix(in srgb, var(--c2c-bg, var(--c2c-bg2)) 96%, transparent);
  color: var(--c2c-fg, var(--c2c-accentBright));
  border: 1px solid color-mix(in srgb, var(--c2c-border, var(--c2c-surface0)) 80%, transparent);
  border-radius: 10px;
  box-shadow: 0 10px 32px color-mix(in srgb, var(--c2c-shadowBase, var(--c2c-black)) 55%, transparent);
  font: 12px ui-sans-serif, system-ui, sans-serif;
  padding: 10px 12px;
  display: none;
  backdrop-filter: blur(8px);
}
#${POPOVER_ID}.open { display: block; }
#${POPOVER_ID} h4 {
  margin: 0 0 6px 0; color: var(--c2c-mauve, var(--c2c-violetSoft));
  font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
  text-transform: uppercase;
}
#${POPOVER_ID} table { width: 100%; border-collapse: collapse; font-size: 11px; }
#${POPOVER_ID} td { padding: 4px 5px; border-bottom: 1px solid color-mix(in srgb, var(--c2c-border, var(--c2c-surface0)) 40%, transparent); vertical-align: top; }
#${POPOVER_ID} td.ok  { color: var(--c2c-green, var(--c2c-okSoft2)); }
#${POPOVER_ID} td.bad { color: var(--c2c-red,   var(--c2c-danger)); }
#${POPOVER_ID} td.tier { width: 18px; text-align: center; opacity: 0.85; }
#${POPOVER_ID} td.rtt  { color: var(--c2c-sub, var(--c2c-gray400)); text-align: right; font-variant-numeric: tabular-nums; }
#${POPOVER_ID} .muted  { color: var(--c2c-sub, var(--c2c-gray400)); }
#${POPOVER_ID} .cost-line {
  display: flex; justify-content: space-between; align-items: baseline;
  margin: 6px 0 2px; font-variant-numeric: tabular-nums;
}
#${POPOVER_ID} .cost-line.warn   { color: var(--c2c-yellow, var(--c2c-warn)); }
#${POPOVER_ID} .cost-line.danger { color: var(--c2c-red,    var(--c2c-danger)); }
#${POPOVER_ID} .actions {
  display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap;
}
#${POPOVER_ID} button {
  flex: 1 1 0; min-width: 80px;
  background: color-mix(in srgb, var(--c2c-bg2, var(--c2c-panelTint)) 90%, transparent);
  color: var(--c2c-fg, var(--c2c-accentBright));
  border: 1px solid color-mix(in srgb, var(--c2c-border, var(--c2c-surface0)) 80%, transparent);
  border-radius: 5px; padding: 5px 8px; cursor: pointer; font-size: 11px;
}
#${POPOVER_ID} button:hover {
  border-color: var(--c2c-mauve, var(--c2c-violetSoft));
  color: var(--c2c-mauve, var(--c2c-violetSoft));
}
#${POPOVER_ID} button:disabled { opacity: 0.5; cursor: progress; }
`;
    document.head.appendChild(s);
}

function _buildChip() {
    const el = document.createElement("button");
    el.id = CHIP_ID;
    el.type = "button";
    el.className = "c2c-omnibar-slot-pill";
    el.setAttribute("aria-haspopup", "dialog");
    el.setAttribute("aria-expanded", "false");
    el.setAttribute("data-state", "ok");
    el.setAttribute("data-tier", "local");

    const dot = document.createElement("span");
    dot.className = "ai-dot";
    el.appendChild(dot);

    const label = document.createElement("span");
    label.className = "ai-label";
    label.textContent = "AI…";
    el.appendChild(label);

    const cost = document.createElement("span");
    cost.className = "ai-cost";
    cost.textContent = "";
    el.appendChild(cost);

    el.addEventListener("click", _togglePopover);

    _chipEl = el;
    _chipDot = dot;
    _chipLabel = label;
    _chipCost = cost;
    return el;
}

async function _fetchStatus() {
    try {
        const r = await fetch("/c2c/ai/status", { headers: { Accept: "application/json" } });
        if (!r.ok) return null;
        const j = await r.json();
        return j?.data || null;
    } catch (_) { return null; }
}

function _renderChip(status) {
    if (!_chipEl) return;
    const hudVisible = app.ui?.settings?.getSettingValue?.(SETTING_HUD, true);
    if (hudVisible === false) {
        _chipEl.style.display = "none";
        return;
    }
    _chipEl.style.display = "inline-flex";

    if (!status || !Array.isArray(status.backends) || status.backends.length === 0) {
        _chipEl.setAttribute("data-state", "warn");
        _chipEl.setAttribute("data-tier", "local");
        _chipLabel.textContent = "AI: not configured";
        _chipCost.textContent = "";
        _chipEl.title = "No AI backends configured — open Settings → C2C → AI Backends";
        return;
    }

    const paused = !!app.ui?.settings?.getSettingValue?.(SETTING_PAUSE, false);
    const enabledHealthy = status.backends.filter((b) => b.enabled && b.health?.ok);
    const active = enabledHealthy[0] || status.backends[0];
    const cost = status.cost || {};
    const tier = active.tier === "cloud" ? "cloud" : "local";
    let state = "ok";
    if (paused) state = "paused";
    else if (cost.over_cap) state = "danger";
    else if (cost.over_warn || !active.health?.ok) state = "warn";

    _chipEl.setAttribute("data-tier", tier);
    _chipEl.setAttribute("data-state", state);
    _chipLabel.textContent = (paused ? "⏸ " : "") + (active.display_name || "AI");
    const today = (cost.today_cost_usd ?? 0);
    const cap   = (cost.cap_usd ?? 0);
    _chipCost.textContent = `$${today.toFixed(3)}/${cap.toFixed(2)}`;

    const rtt = active.health?.last_rtt_ms;
    const err = active.health?.last_error;
    _chipEl.title = `${active.display_name}\nmodel: ${active.model}` +
                    `\nrtt: ${rtt ?? "?"}ms` +
                    `\ntoday: $${today.toFixed(4)} / cap $${cap.toFixed(2)}` +
                    (err ? `\nerror: ${err}` : "");
}

function _ensurePopover() {
    let pop = document.getElementById(POPOVER_ID);
    if (pop) return pop;
    pop = document.createElement("div");
    pop.id = POPOVER_ID;
    pop.setAttribute("role", "dialog");
    pop.setAttribute("aria-label", "AI status");
    document.body.appendChild(pop);
    return pop;
}

function _renderPopover() {
    const pop = _ensurePopover();
    const status = _lastStatus;
    if (!status || !Array.isArray(status.backends) || status.backends.length === 0) {
        pop.innerHTML = `<h4>AI</h4>
            <div class="muted">No AI backends configured.</div>
            <div class="actions">
              <button data-act="settings">Open Settings</button>
            </div>`;
        pop.querySelector('[data-act="settings"]').addEventListener("click", () => {
            try { app.ui?.settings?.show?.("C2C ▸ AI Backends"); } catch { /* */ }
            _closePopover();
        });
        return;
    }

    const cost = status.cost || {};
    const today = (cost.today_cost_usd ?? 0);
    const cap   = (cost.cap_usd ?? 0);
    const calls = cost.today_calls ?? 0;
    const costCls = cost.over_cap ? "danger" : (cost.over_warn ? "warn" : "");
    const paused = !!app.ui?.settings?.getSettingValue?.(SETTING_PAUSE, false);

    const rows = status.backends.map((b) => {
        const tierIcon = b.tier === "cloud" ? "☁" : "💻";
        const healthCls = b.health?.ok ? "ok" : "bad";
        const healthText = b.enabled ? (b.health?.ok ? "ok" : "down") : "off";
        const rtt = b.health?.last_rtt_ms;
        return `<tr>
            <td class="tier">${tierIcon}</td>
            <td>${_escape(b.display_name || b.id || "?")}
                <div class="muted" style="font-size:10px;">${_escape(b.model || "")}</div></td>
            <td class="${healthCls}">${healthText}</td>
            <td class="rtt">${rtt != null ? `${rtt}ms` : "—"}</td>
        </tr>`;
    }).join("");

    pop.innerHTML = `
        <h4>AI Backends</h4>
        <table>${rows}</table>
        <h4 style="margin-top:10px;">Cost (today)</h4>
        <div class="cost-line ${costCls}">
            <span>$${today.toFixed(4)} / cap $${cap.toFixed(2)}</span>
            <span class="muted">${calls} call${calls === 1 ? "" : "s"}</span>
        </div>
        <div class="muted" style="font-size:10px;">day: ${_escape(cost.day_key || "?")}</div>
        <div class="actions">
            <button data-act="refresh">Refresh</button>
            <button data-act="pause">${paused ? "Resume AI" : "Pause AI"}</button>
            <button data-act="settings">Settings</button>
        </div>
    `;

    pop.querySelector('[data-act="refresh"]').addEventListener("click", async (ev) => {
        const btn = ev.currentTarget;
        btn.disabled = true; btn.textContent = "…";
        try { await fetch("/c2c/ai/probe", { method: "POST" }); } catch { /* */ }
        await refresh();
        if (_popoverOpen) _renderPopover();
    });
    pop.querySelector('[data-act="pause"]').addEventListener("click", () => {
        try { app.ui?.settings?.setSettingValue?.(SETTING_PAUSE, !paused); } catch { /* */ }
        refresh().then(() => { if (_popoverOpen) _renderPopover(); });
    });
    pop.querySelector('[data-act="settings"]').addEventListener("click", () => {
        try { app.ui?.settings?.show?.("C2C ▸ AI Backends"); } catch { /* */ }
        _closePopover();
    });
}

function _positionPopover() {
    const pop = document.getElementById(POPOVER_ID);
    if (!pop || !_popoverOpen) return;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    pop.style.visibility = "hidden";
    pop.style.left = "0px";
    pop.style.top  = "0px";
    const popW = pop.offsetWidth;
    const popH = pop.offsetHeight;

    let anchorRect = null;
    if (_chipEl && _chipEl.isConnected && _chipEl.offsetWidth > 0) {
        anchorRect = _chipEl.getBoundingClientRect();
    } else {
        const pill = window.C2COmniBar?.getPill?.();
        if (pill) anchorRect = pill.getBoundingClientRect();
    }

    let left, top;
    if (anchorRect) {
        left = Math.min(Math.max(8, anchorRect.left), vw - popW - 8);
        top  = anchorRect.bottom + 6;
        if (top + popH > vh - 8) {
            top = Math.max(8, anchorRect.top - popH - 6);
        }
    } else {
        left = Math.max(8, vw - popW - 16);
        top  = 60;
    }

    pop.style.left = `${Math.round(left)}px`;
    pop.style.top  = `${Math.round(top)}px`;
    pop.style.visibility = "visible";
}

function _togglePopover() {
    const pop = _ensurePopover();
    if (_popoverOpen) { _closePopover(); return; }
    _renderPopover();
    pop.classList.add("open");
    _popoverOpen = true;
    if (_chipEl) _chipEl.setAttribute("aria-expanded", "true");
    requestAnimationFrame(_positionPopover);
    window.addEventListener("resize", _positionPopover, { passive: true });
    window.addEventListener("scroll", _positionPopover, { passive: true, capture: true });
    setTimeout(() => document.addEventListener("mousedown", _outsideClose, true), 0);
    document.addEventListener("keydown", _onEscape, true);
}

function _closePopover() {
    const pop = document.getElementById(POPOVER_ID);
    if (pop) pop.classList.remove("open");
    _popoverOpen = false;
    if (_chipEl) _chipEl.setAttribute("aria-expanded", "false");
    window.removeEventListener("resize", _positionPopover);
    window.removeEventListener("scroll", _positionPopover, { capture: true });
    document.removeEventListener("mousedown", _outsideClose, true);
    document.removeEventListener("keydown", _onEscape, true);
}

function _outsideClose(ev) {
    const pop = document.getElementById(POPOVER_ID);
    if (!pop) return;
    if (pop.contains(ev.target)) return;
    if (_chipEl && _chipEl.contains(ev.target)) return;
    _closePopover();
}

function _onEscape(ev) {
    if (ev.key === "Escape") _closePopover();
}

function _onSlotMode(mode) {
    _slotMode = (mode === "icon") ? "icon" : "full";
    if (_chipEl) _chipEl.setAttribute("data-c2c-mode", _slotMode);
    if (_popoverOpen) requestAnimationFrame(_positionPopover);
}

function _escape(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
}

async function refresh() {
    _lastStatus = await _fetchStatus();
    _renderChip(_lastStatus);
    if (_popoverOpen) _renderPopover();
}

function _registerChip() {
    if (_unregisterSlot) return;
    const ob = window.C2COmniBar;
    if (!ob || typeof ob.register !== "function") return;
    _injectStyle();
    const el = _buildChip();
    try {
        _unregisterSlot = ob.register({
            section: "ai",
            id: "ai",
            order: 10,
            element: el,
            update: () => { /* polled via setInterval; chip re-renders on each refresh */ },
            onMode: _onSlotMode,
        });
    } catch (e) {
        console.warn("[c2c_ai_status_bar] OmniBar registration failed:", e);
    }
}

app.registerExtension({
    name: "c2c.ai.statusBar",
    settings: [
        { id: SETTING_HUD,   name: "C2C ▸ AI ▸ Show OmniBar AI chip",
          type: "boolean", default: true,  onChange: () => refresh() },
        { id: SETTING_PAUSE, name: "C2C ▸ AI ▸ Pause all AI calls",
          type: "boolean", default: false, onChange: () => refresh() },
    ],
    async setup() {
        // Wait for OmniBar host (up to ~4 s).
        for (let i = 0; i < 40; i++) {
            if (window.C2COmniBar && typeof window.C2COmniBar.register === "function") break;
            await new Promise((r) => setTimeout(r, 100));
        }
        _registerChip();
        // Initial paint + poll loop.
        await refresh();
        if (_pollTimer) clearInterval(_pollTimer);
        _pollTimer = setInterval(refresh, POLL_MS);
        // Sweep any stale legacy DOM left behind by older builds.
        document.getElementById("c2c-ai-hud")?.remove();
        document.getElementById("c2c-ai-hud-flyout")?.remove();
    },
});

// Public API for sibling modules.
window.__C2C_AI_HUD__ = {
    refresh,
    open:  () => { if (!_popoverOpen) _togglePopover(); },
    close: _closePopover,
};
