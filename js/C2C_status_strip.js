// C2C_status_strip.js
// ─────────────────────────────────────────────────────────────────────
// Legacy host for third-party status chips ("HUD pills"). Mounts on
// document.body with position:fixed as a backward-compat surface.
//
// P0.2 Phase 5 (2026-05-30): All six built-in default chips (gpu, vram,
// queue, ai, cost, cmplx) and their polling loops have been RETIRED.
// The OmniBar now hosts the canonical stats surface:
//   • INT badge     → OmniBar "stats" section (c2c_int_badge.js)
//   • AI status     → OmniBar "ai" section    (c2c_ai_status_bar.js)
// Third-party code may still call window.C2CStatusStrip.register() to
// append chips to the bottom-right strip; the strip is hidden by default.
//
// Public API (window.C2CStatusStrip):
//   register({id, label, value?, state?, tooltip?, onClick?, priority?})
//   update(id, patch)         — merge patch into existing chip + rerender
//   unregister(id)            — remove a chip
//   setVisible(bool)          — global toggle (settings hook)
//   list()                    — array of currently registered ids
//   get(id)                   — shallow-copy of chip opts
//   getAll()                  — sorted array of shallow-copy opts
//   subscribe(cb)             — notify on any change; returns unsubscribe fn
// ─────────────────────────────────────────────────────────────────────

import { app } from "../../scripts/app.js";
import { LITE } from "./_c2c_lite.js";

const HOST_ID = "c2c-status-strip";

const CSS = `
#${HOST_ID} {
  position: fixed !important;
  right: 8px; bottom: 8px;
  display: flex; gap: 6px; align-items: center;
  z-index: var(--c2c-z-hud, 1000);
  pointer-events: none;          /* chips opt back in individually */
  font: 11px/1.2 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  /* zoom-immune: no transform on this element or any ancestor */
}
#${HOST_ID} .c2c-chip {
  pointer-events: auto;
  background: color-mix(in srgb, var(--c2c-surface0) 92%, transparent);
  color: var(--c2c-fg);
  border: 1px solid var(--c2c-surface1);
  border-radius: 999px;
  padding: 3px 9px;
  display: inline-flex; align-items: center; gap: 5px;
  white-space: nowrap;
  user-select: none;
  transition: background 0.15s, border-color 0.15s;
}
#${HOST_ID} .c2c-chip:hover {
  background: color-mix(in srgb, var(--c2c-surface1) 95%, transparent);
  border-color: var(--c2c-surface2);
}
#${HOST_ID} .c2c-chip.clickable { cursor: pointer; }
#${HOST_ID} .c2c-chip .c2c-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--c2c-sub);
}
#${HOST_ID} .c2c-chip .c2c-val { opacity: 0.85; }
#${HOST_ID} .c2c-chip[data-state="ok"]   .c2c-dot { background: var(--c2c-green); box-shadow: 0 0 4px var(--c2c-green); }
#${HOST_ID} .c2c-chip[data-state="warn"] .c2c-dot { background: var(--c2c-warnBright); box-shadow: 0 0 4px var(--c2c-warnBright); }
#${HOST_ID} .c2c-chip[data-state="err"]  .c2c-dot { background: var(--c2c-dangerStrong); box-shadow: 0 0 4px var(--c2c-dangerStrong); }
#${HOST_ID} .c2c-chip[data-state="crit"] .c2c-dot { background: var(--c2c-violetSoft); box-shadow: 0 0 6px var(--c2c-violetSoft); }
#${HOST_ID} .c2c-chip[data-state="crit"] { border-color: color-mix(in srgb, var(--c2c-violetSoft) 55%, transparent); }
#${HOST_ID} .c2c-chip[data-state="idle"] .c2c-dot { background: var(--c2c-sub); }
`;

const _chips = new Map(); // id -> { el, opts }
const _subscribers = new Set(); // (chipId|null, opts) => void
let _host = null;
let _styleEl = null;

function _notify(chipId) {
    for (const cb of _subscribers) {
        try { cb(chipId, chipId ? _chips.get(chipId)?.opts : null); }
        catch (e) { console.warn("[C2CStatusStrip] subscriber failed:", e); }
    }
}

function _ensureStyle() {
    if (_styleEl && document.head.contains(_styleEl)) return;
    _styleEl = document.createElement("style");
    _styleEl.dataset.c2cStatusStrip = "1";
    _styleEl.textContent = CSS;
    document.head.appendChild(_styleEl);
}

function _ensureHost() {
    _ensureStyle();
    if (_host && document.body.contains(_host)) return _host;
    // Reuse pre-existing host if HMR / extension reload left one behind.
    const found = document.getElementById(HOST_ID);
    if (found) { _host = found; return _host; }
    _host = document.createElement("div");
    _host.id = HOST_ID;
    document.body.appendChild(_host);
    return _host;
}

function _renderChip(id) {
    const c = _chips.get(id);
    if (!c) return;
    c.el.textContent = "";
    const dot = document.createElement("span");
    dot.className = "c2c-dot";
    c.el.appendChild(dot);
    const lbl = document.createElement("span");
    lbl.className = "c2c-lbl";
    lbl.textContent = c.opts.label || id;
    c.el.appendChild(lbl);
    if (c.opts.value != null && c.opts.value !== "") {
        const v = document.createElement("span");
        v.className = "c2c-val";
        v.textContent = String(c.opts.value);
        c.el.appendChild(v);
    }
    c.el.dataset.state = c.opts.state || "idle";
    c.el.title = c.opts.tooltip || "";
    c.el.classList.toggle("clickable", typeof c.opts.onClick === "function");
}

function _reorder() {
    const host = _ensureHost();
    const sorted = [..._chips.values()].sort(
        (a, b) => (a.opts.priority ?? 100) - (b.opts.priority ?? 100),
    );
    for (const c of sorted) host.appendChild(c.el);
}

const API = {
    register(opts) {
        if (!opts || !opts.id) {
            console.warn("[C2CStatusStrip] register: opts.id is required");
            return;
        }
        _ensureHost();
        if (_chips.has(opts.id)) {
            API.update(opts.id, opts);
            return;
        }
        const el = document.createElement("div");
        el.className = "c2c-chip";
        el.dataset.chipId = opts.id;
        el.addEventListener("click", () => {
            const c = _chips.get(opts.id);
            if (c && typeof c.opts.onClick === "function") {
                try { c.opts.onClick(c.opts); }
                catch (e) { console.warn("[C2CStatusStrip] onClick failed:", e); }
            }
        });
        _chips.set(opts.id, { el, opts: { ...opts } });
        _renderChip(opts.id);
        _reorder();
        _notify(opts.id);
    },
    update(id, patch) {
        const c = _chips.get(id);
        if (!c) return;
        Object.assign(c.opts, patch);
        _renderChip(id);
        _notify(id);
    },
    unregister(id) {
        const c = _chips.get(id);
        if (!c) return;
        c.el.remove();
        _chips.delete(id);
        _notify(id);
    },
    setVisible(v) {
        const host = _ensureHost();
        host.style.display = v ? "flex" : "none";
    },
    list() { return [..._chips.keys()]; },
    // ── New (2026-05-25): expose live chip data to other modules so the
    // Stats Pill (next to OmniPill) can render without duplicating the
    // polling loop. Returns a shallow copy of opts; mutations don't
    // affect the registry.
    get(id) {
        const c = _chips.get(id);
        return c ? { ...c.opts } : null;
    },
    getAll() {
        const sorted = [..._chips.values()].sort(
            (a, b) => (a.opts.priority ?? 100) - (b.opts.priority ?? 100),
        );
        return sorted.map(c => ({ ...c.opts }));
    },
    subscribe(cb) {
        if (typeof cb !== "function") return () => {};
        _subscribers.add(cb);
        return () => _subscribers.delete(cb);
    },
};

// Expose globally so legacy HUD scripts can migrate incrementally.
if (typeof window !== "undefined") window.C2CStatusStrip = API;

function _start() {
    _ensureHost();
}

if (!LITE) app.registerExtension({
    name: "C2C.StatusStrip",
    settings: [
        {
            id: "c2c.statusStrip.enabled",
            name: "Show bottom-right status strip (legacy)",
            type: "boolean",
            defaultValue: false,
            tooltip: "Legacy floating strip — off by default. The OmniBar (C2C pill in the Manager bar) is the canonical stats surface.",
            onChange: (v) => { try { API.setVisible(!!v); } catch { /* */ } },
        },
    ],
    async setup() {
        try {
            _start();
            const enabled = app.ui?.settings?.getSettingValue?.(
                "c2c.statusStrip.enabled", false) ?? false;
            API.setVisible(!!enabled);
        }
        catch (e) { console.warn("[C2CStatusStrip] setup failed:", e); }
    },
});
