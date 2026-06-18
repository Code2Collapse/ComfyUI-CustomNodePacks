// c2c_stats_pill.js
// ─────────────────────────────────────────────────────────────────────
// Stats Pill — second pill in the ComfyUI Manager button group, right
// next to the C2C OmniPill. Shows ONE live stat at a time; scroll-wheel
// cycles through (INT, GPU, VRAM, Q, AI, $, C). Click toggles a horizontal
// "navbar" rendered just below the Manager bar showing ALL stats inline.
//
// Data source: window.C2CStatusStrip (see C2C_status_strip.js). We never
// poll endpoints ourselves — that registry already polls /system_stats,
// /queue, /c2c/ai/status, /c2c/ai/cost every 3s and pushes updates here
// via the subscribe() callback. INT/graph-health chip is also published
// into the same registry by c2c_int_badge.js.
//
// Public API (window.C2CStatsPill):
//   getActive()    — id of currently visible chip
//   setActive(id)  — programmatically switch
//   list()         — chip ids in display order
//   openPanel()    — open the all-stats navbar
//   closePanel()   — close it
//   togglePanel()  — toggle
//   isPanelOpen()  — boolean
//   getPill()      — the pill DOM node (or null)
//
// License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────

import { app } from "../../scripts/app.js";
import { z as Z } from "./_c2c_theme.js";

const PILL_ID    = "c2c-stats-pill";
const PANEL_ID   = "c2c-stats-pill-panel";
const STYLE_ID   = "c2c-stats-pill-style";
const STORAGE_K  = "c2c.statsPill.active";
const MGR_SEL    = ".actionbar-container .comfyui-button-group";
const OMNI_SEL   = "#c2c-omnibar-pill";

// Built-in chip display order for the System HUD pill.
// graph_health is owned by c2c_int_badge.js (Doctor button).
// cmplx is owned by #mec-complexity-hud (top-center bar).
// Both are excluded from the SysHUD cycle to keep domain separation:
// SysHUD pill = live MACHINE state only (Crystools-style scroll cycle).
// Order is the user-facing scroll order; chips that don't have data yet
// are skipped automatically.
const ORDER = [
    "vram",       // VRAM used/total GB
    "gpu",        // GPU utilisation %
    "gpu_temp",   // GPU temperature °C
    "ram",        // RAM used/total GB
    "cpu",        // CPU %
    "queue",      // Queue depth
    "cost",       // AI $ today
    "last_run",   // Last run time (sec)
    "graph_size", // Workflow node + link count
    "ws_lat",     // Round-trip latency ms
    "ai",
];
const EXCLUDE = new Set(["graph_health", "cmplx"]);

const CSS = `
#${PILL_ID} {
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  height: 28px; padding: 0 10px;
  margin: 0 0 0 4px;
  background: linear-gradient(180deg,
              color-mix(in srgb, var(--c2c-blue) 18%, var(--c2c-bg)),
              color-mix(in srgb, var(--c2c-blue) 6%, var(--c2c-bg)));
  color: var(--c2c-fg);
  border: 1px solid color-mix(in srgb, var(--c2c-blue) 35%, var(--c2c-border));
  border-radius: 14px;
  font: 11px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-weight: 600;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  transition: background 0.15s, border-color 0.15s, transform 0.08s;
  position: relative;
  text-align: center;
  /* min-width is set dynamically by _lockPillWidth() so the pill grows
     to the widest chip it has ever shown and never shrinks back (no
     resize flicker as values cycle). */
}
#${PILL_ID}:hover {
  background: linear-gradient(180deg,
              color-mix(in srgb, var(--c2c-blue) 26%, var(--c2c-bg)),
              color-mix(in srgb, var(--c2c-blue) 12%, var(--c2c-bg)));
  border-color: color-mix(in srgb, var(--c2c-blue) 55%, var(--c2c-border));
}
#${PILL_ID}:active { transform: scale(0.97); }
#${PILL_ID} .c2c-sp-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--c2c-sub);
  flex: 0 0 7px;
  transition: background 0.2s, box-shadow 0.2s;
}
#${PILL_ID} .c2c-sp-lbl {
  opacity: 0.95;
  letter-spacing: 0.02em;
  font-size: 10px;
  text-transform: uppercase;
}
#${PILL_ID} .c2c-sp-val {
  opacity: 0.85;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  /* No max-width clipping anymore — the pill grows to fit; lock keeps
     it stable across cycles. */
  overflow: hidden;
  text-overflow: ellipsis;
}
#${PILL_ID} .c2c-sp-cyc {
  opacity: 0.45;
  font-size: 9px;
  margin-left: 2px;
}
#${PILL_ID}[data-state="ok"]   .c2c-sp-dot { background: var(--c2c-green);
  box-shadow: 0 0 4px color-mix(in srgb, var(--c2c-green) 60%, transparent); }
#${PILL_ID}[data-state="warn"] .c2c-sp-dot { background: var(--c2c-warnBright, var(--c2c-yellow));
  box-shadow: 0 0 4px color-mix(in srgb, var(--c2c-yellow) 60%, transparent); }
#${PILL_ID}[data-state="err"]  .c2c-sp-dot { background: var(--c2c-red);
  box-shadow: 0 0 5px color-mix(in srgb, var(--c2c-red) 60%, transparent); }
#${PILL_ID}[data-state="crit"] .c2c-sp-dot { background: var(--c2c-mauve, var(--c2c-violetSoft, var(--c2c-mauve)));
  box-shadow: 0 0 6px color-mix(in srgb, var(--c2c-mauve, var(--c2c-mauve)) 70%, transparent); }
#${PILL_ID}[data-state="idle"] .c2c-sp-dot { background: var(--c2c-sub); box-shadow: none; }
#${PILL_ID}[aria-expanded="true"] {
  background: linear-gradient(180deg,
              color-mix(in srgb, var(--c2c-blue) 34%, var(--c2c-bg)),
              color-mix(in srgb, var(--c2c-blue) 18%, var(--c2c-bg)));
}

#${PANEL_ID} {
  position: fixed;
  display: none;
  z-index: var(--c2c-z-popover, 9000);
  background: var(--c2c-panelBg, var(--c2c-bg));
  color: var(--c2c-fg);
  border: 1px solid var(--c2c-border);
  border-radius: 10px;
  box-shadow: 0 8px 24px var(--c2c-shadowBase, rgba(0,0,0,0.4));
  padding: 8px 10px;
  font: 11px/1.2 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: calc(100vw - 16px);
  overflow-x: auto;
}
#${PANEL_ID}[data-c2c-stats-pill-open="1"] { display: flex; }
#${PANEL_ID} .c2c-sp-row {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
}
#${PANEL_ID} .c2c-sp-chip,
.c2c-sp-omni-row .c2c-sp-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 10px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--c2c-highlightBase, var(--c2c-bg)) 65%, transparent);
  border: 1px solid var(--c2c-border);
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, transform 0.08s;
  font-weight: 600;
  white-space: nowrap;
}
#${PANEL_ID} .c2c-sp-chip:hover,
.c2c-sp-omni-row .c2c-sp-chip:hover {
  background: color-mix(in srgb, var(--c2c-highlightBase, var(--c2c-bg)) 85%, transparent);
  border-color: color-mix(in srgb, var(--c2c-blue) 45%, var(--c2c-border));
}
#${PANEL_ID} .c2c-sp-chip.active {
  background: color-mix(in srgb, var(--c2c-blue) 22%, var(--c2c-bg));
  border-color: color-mix(in srgb, var(--c2c-blue) 55%, var(--c2c-border));
}
#${PANEL_ID} .c2c-sp-chip:active,
.c2c-sp-omni-row .c2c-sp-chip:active { transform: scale(0.97); }
#${PANEL_ID} .c2c-sp-chip .c2c-sp-dot,
.c2c-sp-omni-row .c2c-sp-chip .c2c-sp-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--c2c-sub);
  flex: 0 0 7px;
}
#${PANEL_ID} .c2c-sp-chip[data-state="ok"]   .c2c-sp-dot,
.c2c-sp-omni-row .c2c-sp-chip[data-state="ok"]   .c2c-sp-dot { background: var(--c2c-green); box-shadow: 0 0 3px var(--c2c-green); }
#${PANEL_ID} .c2c-sp-chip[data-state="warn"] .c2c-sp-dot,
.c2c-sp-omni-row .c2c-sp-chip[data-state="warn"] .c2c-sp-dot { background: var(--c2c-yellow); box-shadow: 0 0 3px var(--c2c-yellow); }
#${PANEL_ID} .c2c-sp-chip[data-state="err"]  .c2c-sp-dot,
.c2c-sp-omni-row .c2c-sp-chip[data-state="err"]  .c2c-sp-dot { background: var(--c2c-red);    box-shadow: 0 0 4px var(--c2c-red); }
#${PANEL_ID} .c2c-sp-chip[data-state="crit"] .c2c-sp-dot,
.c2c-sp-omni-row .c2c-sp-chip[data-state="crit"] .c2c-sp-dot { background: var(--c2c-mauve, var(--c2c-mauve)); box-shadow: 0 0 5px var(--c2c-mauve, var(--c2c-mauve)); }
#${PANEL_ID} .c2c-sp-chip .c2c-sp-lbl,
.c2c-sp-omni-row .c2c-sp-chip .c2c-sp-lbl {
  font-size: 10px; opacity: 0.95; text-transform: uppercase; letter-spacing: 0.02em;
}
#${PANEL_ID} .c2c-sp-chip .c2c-sp-val,
.c2c-sp-omni-row .c2c-sp-chip .c2c-sp-val {
  opacity: 0.85; font-variant-numeric: tabular-nums; font-weight: 700;
}
`;

let _pill = null;
let _panel = null;
let _styleEl = null;
let _activeId = null;
let _mutObserver = null;
let _unsubStrip = null;
let _outsideHandler = null;
let _escHandler = null;
let _wheelAcc = 0;
// Auto-grow-and-lock: track the widest natural width the pill has ever
// needed, and pin the pill at that width so cycling chips never causes
// a layout shift (user-reported annoyance).
let _maxPillWidth = 0;

function _lockPillWidth() {
    if (!_pill) return;
    try {
        // Measure the unconstrained natural width. We temporarily clear
        // any previously-set min-width so we get a clean reading, then
        // restore the lock at max(previous, current).
        const prev = _pill.style.minWidth;
        _pill.style.minWidth = "";
        // scrollWidth ignores the min-width we just cleared and reflects
        // the actual content size including padding.
        const w = _pill.scrollWidth;
        if (w > _maxPillWidth) _maxPillWidth = w;
        const px = Math.max(_maxPillWidth, w);
        if (px > 0) _pill.style.minWidth = px + "px";
        else if (prev) _pill.style.minWidth = prev;
    } catch { /* offscreen / not attached — ignore */ }
}

function _injectStyle() {
    if (_styleEl && document.head.contains(_styleEl)) return;
    _styleEl = document.createElement("style");
    _styleEl.id = STYLE_ID;
    _styleEl.textContent = CSS;
    document.head.appendChild(_styleEl);
}

function _getOrderedChips() {
    const strip = window.C2CStatusStrip;
    if (!strip || typeof strip.getAll !== "function") return [];
    const all = strip.getAll(); // sorted by priority
    // Apply our preferred display order on top: ORDER ids first (if present),
    // then remaining chips in registry order. EXCLUDE filters out chips that
    // belong to other UI surfaces (INT/Doctor, complexity bar).
    const map = new Map(all.filter(c => !EXCLUDE.has(c.id)).map(c => [c.id, c]));
    const out = [];
    for (const id of ORDER) {
        if (map.has(id)) { out.push(map.get(id)); map.delete(id); }
    }
    for (const c of map.values()) out.push(c);
    return out;
}

function _loadActive() {
    try {
        const saved = localStorage.getItem(STORAGE_K);
        if (saved) return saved;
    } catch { /* */ }
    return null;
}
function _saveActive(id) {
    try { localStorage.setItem(STORAGE_K, id); } catch { /* */ }
}

function _resolveActive() {
    const chips = _getOrderedChips();
    if (chips.length === 0) return null;
    // Prefer the user's saved choice if it still exists.
    if (_activeId && chips.find(c => c.id === _activeId)) return _activeId;
    const saved = _loadActive();
    if (saved && chips.find(c => c.id === saved)) {
        _activeId = saved;
        return _activeId;
    }
    // Otherwise pick the first chip with non-idle state, else first chip.
    const interesting = chips.find(c => c.state && c.state !== "idle");
    _activeId = (interesting || chips[0]).id;
    return _activeId;
}

function _renderPill() {
    if (!_pill) return;
    const chips = _getOrderedChips();
    const activeId = _resolveActive();
    const chip = chips.find(c => c.id === activeId);
    _pill.dataset.state = chip?.state || "idle";
    _pill.title = chip
        ? `${chip.label || chip.id}: ${chip.value ?? "—"}${chip.tooltip ? "\n" + chip.tooltip : ""}\n\nScroll to cycle · Click for all stats`
        : "C2C Stats — no data yet\nScroll to cycle · Click for all stats";
    _pill.innerHTML = "";
    const dot = document.createElement("span");
    dot.className = "c2c-sp-dot";
    _pill.appendChild(dot);
    const lbl = document.createElement("span");
    lbl.className = "c2c-sp-lbl";
    lbl.textContent = chip ? (chip.label || chip.id) : "…";
    _pill.appendChild(lbl);
    const val = document.createElement("span");
    val.className = "c2c-sp-val";
    val.textContent = chip ? (chip.value != null && chip.value !== "" ? String(chip.value) : "—") : "—";
    _pill.appendChild(val);
    // small cycle indicator hint (▾) so the user knows it's interactive
    const cyc = document.createElement("span");
    cyc.className = "c2c-sp-cyc";
    cyc.textContent = "▾";
    _pill.appendChild(cyc);
    // Update the locked width AFTER content is in the DOM so scrollWidth
    // reflects the new chip; this guarantees the pill grows to the
    // widest value ever observed and never shrinks below it.
    _lockPillWidth();
}

function _renderPanel() {
    if (!_panel) return;
    const chips = _getOrderedChips();
    const activeId = _resolveActive();
    _panel.innerHTML = "";
    const row = document.createElement("div");
    row.className = "c2c-sp-row";
    if (chips.length === 0) {
        const empty = document.createElement("div");
        empty.style.opacity = "0.7";
        empty.style.padding = "4px 8px";
        empty.textContent = "No live stats yet (still warming up the polling loop)…";
        row.appendChild(empty);
    }
    for (const c of chips) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "c2c-sp-chip" + (c.id === activeId ? " active" : "");
        chip.dataset.state = c.state || "idle";
        chip.dataset.chipId = c.id;
        chip.title = c.tooltip || "";
        chip.setAttribute("aria-label", `${c.label || c.id}: ${c.value ?? "—"}`);
        const dot = document.createElement("span");
        dot.className = "c2c-sp-dot";
        chip.appendChild(dot);
        const lbl = document.createElement("span");
        lbl.className = "c2c-sp-lbl";
        lbl.textContent = c.label || c.id;
        chip.appendChild(lbl);
        const val = document.createElement("span");
        val.className = "c2c-sp-val";
        val.textContent = c.value != null && c.value !== "" ? String(c.value) : "—";
        chip.appendChild(val);
        chip.addEventListener("click", (ev) => {
            ev.stopPropagation();
            _activeId = c.id;
            _saveActive(c.id);
            _renderPill();
            _renderPanel();
            // If the source chip exposed an onClick, also fire it (e.g.
            // INT badge → open Doctor at the right tab).
            try {
                const raw = window.C2CStatusStrip?.get?.(c.id);
                if (raw && typeof raw.onClick === "function") raw.onClick(raw);
            } catch { /* */ }
        });
        row.appendChild(chip);
    }
    _panel.appendChild(row);
}

function _positionPanel() {
    if (!_panel || !_pill) return;
    const pr = _pill.getBoundingClientRect();
    _panel.style.top = (pr.bottom + 6) + "px";
    // anchor to the pill's right edge, but clamp to viewport
    const desiredWidth = Math.min(window.innerWidth - 16, _panel.scrollWidth + 22);
    let left = pr.right - desiredWidth;
    if (left < 8) left = 8;
    if (left + desiredWidth > window.innerWidth - 8) left = window.innerWidth - desiredWidth - 8;
    _panel.style.left = left + "px";
    _panel.style.maxWidth = (window.innerWidth - 16) + "px";
}

function _openPanel() {
    if (!_panel) _ensurePanel();
    _renderPanel();
    _panel.style.display = "flex";
    _panel.setAttribute("data-c2c-stats-pill-open", "1");
    _positionPanel();
    if (_pill) _pill.setAttribute("aria-expanded", "true");
    if (!_outsideHandler) {
        _outsideHandler = (ev) => {
            if (!_panel) return;
            if (_panel.contains(ev.target)) return;
            if (_pill && _pill.contains(ev.target)) return;
            _closePanel();
        };
        document.addEventListener("mousedown", _outsideHandler, true);
    }
    if (!_escHandler) {
        _escHandler = (ev) => {
            if (ev.key === "Escape" && _isPanelOpen()) { _closePanel(); }
        };
        document.addEventListener("keydown", _escHandler, true);
    }
    window.addEventListener("resize", _positionPanel, { passive: true });
    window.addEventListener("scroll", _positionPanel, { passive: true, capture: true });
}

function _closePanel() {
    if (!_panel) return;
    _panel.style.display = "none";
    _panel.removeAttribute("data-c2c-stats-pill-open");
    if (_pill) _pill.setAttribute("aria-expanded", "false");
    if (_outsideHandler) {
        document.removeEventListener("mousedown", _outsideHandler, true);
        _outsideHandler = null;
    }
    if (_escHandler) {
        document.removeEventListener("keydown", _escHandler, true);
        _escHandler = null;
    }
    window.removeEventListener("resize", _positionPanel);
    window.removeEventListener("scroll", _positionPanel, true);
}

function _isPanelOpen() {
    return !!(_panel && _panel.getAttribute("data-c2c-stats-pill-open") === "1");
}

function _togglePanel() {
    if (_isPanelOpen()) _closePanel(); else _openPanel();
}

function _cycle(delta) {
    const chips = _getOrderedChips();
    if (chips.length === 0) return;
    const activeId = _resolveActive();
    const idx = chips.findIndex(c => c.id === activeId);
    const n = chips.length;
    const next = chips[((idx < 0 ? 0 : idx) + delta + n) % n];
    _activeId = next.id;
    _saveActive(next.id);
    _renderPill();
    if (_isPanelOpen()) _renderPanel();
}

function _ensurePill() {
    if (_pill && document.body.contains(_pill)) return _pill;
    _injectStyle();
    _pill = document.createElement("button");
    _pill.type = "button";
    _pill.id = PILL_ID;
    _pill.setAttribute("aria-label", "C2C Stats — scroll to cycle, click to expand");
    _pill.setAttribute("aria-haspopup", "dialog");
    _pill.setAttribute("aria-expanded", "false");
    _pill.dataset.state = "idle";

    _pill.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        _togglePanel();
    });

    // Scroll-wheel cycles. We accumulate fractional deltas so trackpad
    // micro-movements don't fire 12× per swipe.
    _pill.addEventListener("wheel", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        _wheelAcc += ev.deltaY;
        // ~40px threshold (one mouse-wheel notch ≈ 100, trackpad ≈ 4-20)
        while (_wheelAcc >= 40)  { _cycle(+1); _wheelAcc -= 40; }
        while (_wheelAcc <= -40) { _cycle(-1); _wheelAcc += 40; }
    }, { passive: false });

    // Middle-click on the pill toggles the floating legacy strip (handy
    // diagnostic — also reachable from Settings).
    _pill.addEventListener("auxclick", (ev) => {
        if (ev.button !== 1) return;
        ev.preventDefault();
        try {
            const host = document.getElementById("c2c-status-strip");
            if (!host) return;
            const showing = host.style.display !== "none";
            window.C2CStatusStrip?.setVisible?.(!showing);
        } catch { /* */ }
    });
    return _pill;
}

function _ensurePanel() {
    if (_panel && document.body.contains(_panel)) return _panel;
    _injectStyle();
    _panel = document.createElement("div");
    _panel.id = PANEL_ID;
    _panel.setAttribute("role", "dialog");
    _panel.setAttribute("aria-label", "C2C Stats — all live values");
    document.body.appendChild(_panel);
    return _panel;
}

function _injectIntoManagerBar() {
    // Always anchor next to the OmniPill so we sit in the SAME button
    // group (ComfyUI's actionbar has multiple .comfyui-button-group
    // siblings; picking the first one lands us on the wrong side of
    // the bar). Fall back to the group containing the canonical INT
    // chip (`#c2c-int-chip` inside OmniBar), then to the first
    // actionbar group.
    const existing = document.getElementById(PILL_ID);
    const omni = document.querySelector(OMNI_SEL);
    let group = null;
    if (omni && omni.parentElement) {
        group = omni.parentElement;
    } else {
        const intChip = document.getElementById("c2c-int-chip");
        if (intChip && intChip.closest(".comfyui-button-group")) {
            group = intChip.closest(".comfyui-button-group");
        } else {
            group = document.querySelector(MGR_SEL);
        }
    }
    if (!group) return false;
    if (existing && existing.parentElement === group) return true;
    _ensurePill();
    if (omni && omni.parentElement === group) {
        omni.insertAdjacentElement("afterend", _pill);
    } else {
        group.appendChild(_pill);
    }
    _renderPill();
    return true;
}

// Iteration 3: docked placement — float the pill at bottom-right above
// the existing dock anchor (legacy #mec-system-hud is hidden so the
// position is free). Registers with window.__mecDock so other HUD
// elements (e.g. legacy popovers) auto-shift up when present.
function _dockToBottomRight() {
    _ensurePill();
    // Strip any Manager-bar styling and float bottom-right.
    _pill.style.position = "fixed";
    _pill.style.right = "12px";
    _pill.style.bottom = "12px";
    _pill.style.left = "auto";
    _pill.style.top = "auto";
    _pill.style.margin = "0";
    _pill.style.zIndex = String(Z.hud);
    if (!document.body.contains(_pill)) {
        document.body.appendChild(_pill);
    } else if (_pill.parentElement !== document.body) {
        document.body.appendChild(_pill);
    }
    try { window.__mecDock?.register?.(_pill, { baseBottom: 12 }); } catch { /* */ }
    _renderPill();
    return true;
}

function _clearPlacementStyles() {
    if (!_pill) return;
    _pill.style.position = "";
    _pill.style.right = "";
    _pill.style.bottom = "";
    _pill.style.left = "";
    _pill.style.top = "";
    _pill.style.margin = "";
    _pill.style.zIndex = "";
    try { window.__mecDock?.unregister?.(_pill); } catch { /* */ }
}

function _getPlacement() {
    try {
        const v = app.ui?.settings?.getSettingValue?.(
            "c2c.statsPill.placement", "manager");
        return v === "dock" ? "dock" : "manager";
    } catch { return "manager"; }
}

function _mountByPlacement() {
    const placement = _getPlacement();
    if (placement === "dock") {
        // Detach from any Manager-bar group first.
        if (_pill && _pill.parentElement &&
            _pill.parentElement !== document.body) {
            _pill.remove();
        }
        return _dockToBottomRight();
    }
    // Manager placement: clear any docked styles + inject into bar.
    _clearPlacementStyles();
    if (_pill && _pill.parentElement === document.body) _pill.remove();
    return _injectIntoManagerBar();
}

function _attachMutObserver() {
    if (_mutObserver) return;
    let rafPending = false;
    const tick = () => {
        rafPending = false;
        if (!_pill) return;
        // Dock placement: nothing to re-locate, just ensure the pill is
        // still attached. Avoid bouncing it back into the Manager bar.
        if (_getPlacement() === "dock") {
            if (!document.body.contains(_pill)) _dockToBottomRight();
            return;
        }
        if (!document.body.contains(_pill)) {
            _injectIntoManagerBar();
            return;
        }
        // Re-locate if OmniPill just appeared (or moved) in a different
        // button group than we currently sit in. Without this, we stay
        // stuck in the wrong group if we mounted before OmniPill did.
        const omni = document.querySelector(OMNI_SEL);
        if (omni && omni.parentElement && _pill.parentElement !== omni.parentElement) {
            omni.insertAdjacentElement("afterend", _pill);
            _renderPill();
        }
    };
    _mutObserver = new MutationObserver(() => {
        if (rafPending) return;
        rafPending = true;
        const raf = (typeof requestAnimationFrame !== "undefined")
            ? requestAnimationFrame
            : (cb) => setTimeout(cb, 16);
        raf(tick);
    });
    // childList only (subtree:false) — we only care about top-level mounts
    // and the pill/omni reparenting events, which all happen at body level
    // or are caught by polling elsewhere. This is ~30× cheaper than
    // subtree:true on a 5k-node DOM.
    _mutObserver.observe(document.body, { childList: true, subtree: false });
}

function _subscribeStrip() {
    if (_unsubStrip) return;
    const strip = window.C2CStatusStrip;
    if (!strip || typeof strip.subscribe !== "function") return;
    _unsubStrip = strip.subscribe(() => {
        _renderPill();
        if (_isPanelOpen()) _renderPanel();
    });
}

function _waitForStripAndStart() {
    if (window.C2CStatusStrip?.subscribe) {
        _subscribeStrip();
        _renderPill();
        return;
    }
    let ticks = 0;
    const iv = setInterval(() => {
        if (window.C2CStatusStrip?.subscribe) {
            clearInterval(iv);
            _subscribeStrip();
            _renderPill();
            return;
        }
        if (++ticks > 200) clearInterval(iv); // 200 * 50ms = 10s
    }, 50);
}

// ─── Public API ──────────────────────────────────────────────────────
const PublicAPI = {
    getActive()      { return _resolveActive(); },
    setActive(id)    {
        const chips = _getOrderedChips();
        if (!chips.find(c => c.id === id)) return false;
        _activeId = id; _saveActive(id);
        _renderPill();
        if (_isPanelOpen()) _renderPanel();
        return true;
    },
    list()           { return _getOrderedChips().map(c => c.id); },
    openPanel()      { _openPanel(); },
    closePanel()     { _closePanel(); },
    togglePanel()    { _togglePanel(); },
    isPanelOpen()    { return _isPanelOpen(); },
    getPill()        { return _pill; },
};

if (typeof window !== "undefined") {
    Object.defineProperty(window, "C2CStatsPill", {
        value: PublicAPI,
        writable: false,
        configurable: false,
    });
}

app.registerExtension({
    name: "C2C.StatsPill",
    settings: [
        {
            id: "c2c.statsPill.enabled",
            name: "System HUD pill (Crystools-style)",
            type: "boolean",
            defaultValue: true,
            tooltip: "Single live-stat pill (VRAM/Queue/AI$). Scroll-wheel cycles between them; click expands a navbar with all of them.",
            onChange: (v) => {
                if (v) {
                    _mountByPlacement();
                } else if (_pill && _pill.parentElement) {
                    _pill.remove();
                    _closePanel();
                }
            },
        },
        {
            id: "c2c.statsPill.placement",
            name: "System HUD pill placement",
            type: "combo",
            options: [
                { text: "Beside OmniPill (Manager bar)", value: "manager" },
                { text: "Bottom-right dock", value: "dock" },
            ],
            defaultValue: "manager",
            tooltip: "Where the System HUD pill lives. 'manager' = top-right next to OmniPill; 'dock' = floating bottom-right above other HUD chips.",
            onChange: () => _mountByPlacement(),
        },
    ],
    async setup() {
        try {
            _injectStyle();
            // Wait briefly for Manager bar to exist (it mounts late in
            // ComfyUI's Vue boot), then mount per placement setting.
            // MutationObserver handles any subsequent re-renders.
            let tries = 0;
            const iv = setInterval(() => {
                const enabled = app.ui?.settings?.getSettingValue?.(
                    "c2c.statsPill.enabled", true) ?? true;
                if (!enabled) { clearInterval(iv); return; }
                if (_mountByPlacement() || ++tries > 200) {
                    clearInterval(iv);
                    _attachMutObserver();
                    _waitForStripAndStart();
                }
            }, 50);
        } catch (e) {
            console.warn("[C2CStatsPill] setup failed:", e);
        }
    },
});
