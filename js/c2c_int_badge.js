// c2c_int_badge.js — P0.6 unified INT (Integrity) badge
// ─────────────────────────────────────────────────────────────────────
// P0.2 Phase 2 migration (2026-05-29):
//   This badge previously registered onto window.C2CStatusStrip (the
//   bottom-right floating chip strip). It now registers as a native slot
//   on window.C2COmniBar in the "stats" section, rendering as a pill
//   button alongside other stats chips (graph health, cost, complexity).
//   Status-strip wiring is fully removed — the chip element is owned by
//   this module and inserted into OmniBar's stats host by the slot
//   registry's renderer.
//
// Registers a single "INT" pill on the C2C OmniBar that merges:
//   • Workflow lint            — POST /c2c/int/health {workflow}
//   • Runtime telemetry        — node_error events from the diagnostics ring
//   • Package integrity        — events / pip_check / checksum drift
//   • Live graph health        — window.__C2C_GRAPH_HEALTH__ (cycles, dead,
//                                dangling) from c2c_graph_health.js
//
// Chip colour (4 levels):
//   green  ok    — clean
//   yellow warn  — warnings present
//   red    err   — errors present
//   purple crit  — pip-check failed OR >=2 OOMs in window OR last run failed
//
// Click → opens a breakdown popover with per-section counts and "Open Doctor",
// "Open Integrity", "Open Graph Health" deep-links.
//
// Settings:
//   c2c.int.windowSeconds (default 300)
//   c2c.int.pollMs        (default 4000)
//
// License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CHIP_ID = "int";
const CHIP_EL_ID = "c2c-int-chip";
const POPOVER_ID = "c2c-int-popover";
const STYLE_ID = "c2c-int-popover-style";

const SETTING_WINDOW = "c2c.int.windowSeconds";
const SETTING_POLL = "c2c.int.pollMs";

const DEFAULT_WINDOW_S = 300;
const DEFAULT_POLL_MS = 4000;
const MIN_POLL_MS = 1500;

let _timer = null;
let _lastHealth = null;
let _busy = false;
let _popoverOpen = false;
let _chipEl = null;
let _chipDot = null;
let _chipVal = null;
let _chipLbl = null;
let _slotMode = "full";
let _unregisterSlot = null;

function _getSetting(id, fallback) {
    try {
        const v = app?.ui?.settings?.getSettingValue?.(id, fallback);
        return (v === undefined || v === null) ? fallback : v;
    } catch { return fallback; }
}

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
/* INT chip inside the OmniBar stats section (slot-pill styling is
   provided by c2c_omnibar.js — we only add state colours + value). */
#${CHIP_EL_ID} {
    /* Slot-pill base from OmniBar already applied via .c2c-omnibar-slot-pill;
       these only fine-tune the INT-specific markers. */
    gap: 6px;
}
#${CHIP_EL_ID} .c2c-int-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--c2c-sub, color-mix(in srgb, var(--c2c-fg, var(--c2c-accentBright)) 40%, transparent));
    flex: 0 0 auto;
    transition: background 120ms ease, box-shadow 120ms ease;
}
#${CHIP_EL_ID} .c2c-int-lbl { font-weight: 600; letter-spacing: 0.04em; }
#${CHIP_EL_ID} .c2c-int-val {
    opacity: 0.85;
    font-variant-numeric: tabular-nums;
    min-width: 14px;
    text-align: right;
}
#${CHIP_EL_ID}[data-state="ok"]   .c2c-int-dot { background: var(--c2c-green,       var(--c2c-okSoft)); box-shadow: 0 0 5px var(--c2c-green,       var(--c2c-okSoft)); }
#${CHIP_EL_ID}[data-state="warn"] .c2c-int-dot { background: var(--c2c-warnBright,  var(--c2c-yellow)); box-shadow: 0 0 5px var(--c2c-warnBright,  var(--c2c-yellow)); }
#${CHIP_EL_ID}[data-state="err"]  .c2c-int-dot { background: var(--c2c-dangerStrong,var(--c2c-red)); box-shadow: 0 0 5px var(--c2c-dangerStrong,var(--c2c-red)); }
#${CHIP_EL_ID}[data-state="crit"] .c2c-int-dot { background: var(--c2c-violetSoft,  var(--c2c-mauve)); box-shadow: 0 0 7px var(--c2c-violetSoft,  var(--c2c-mauve)); }
#${CHIP_EL_ID}[data-state="crit"] {
    border-color: color-mix(in srgb, var(--c2c-violetSoft, var(--c2c-mauve)) 55%, transparent) !important;
}
/* Icon-mode collapse: hide label + value, only the dot remains. */
#${CHIP_EL_ID}[data-c2c-mode="icon"] .c2c-int-lbl,
#${CHIP_EL_ID}[data-c2c-mode="icon"] .c2c-int-val {
    display: none;
}
#${CHIP_EL_ID}[data-c2c-mode="icon"] { padding: 3px 8px; }

#${POPOVER_ID} {
    position: fixed;
    width: 360px; max-height: 460px;
    background: color-mix(in srgb, var(--c2c-panelBg) 97%, transparent);
    border: 1px solid color-mix(in srgb, var(--c2c-highlightBase) 14%, transparent);
    border-radius: 10px;
    color: var(--c2c-accentNeutral);
    font: 12px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    z-index: calc(var(--c2c-z-popover, 9000) + 2);
    display: none; flex-direction: column;
    box-shadow: 0 12px 36px color-mix(in srgb, var(--c2c-shadowBase) 55%, transparent);
    overflow: hidden;
}
#${POPOVER_ID}.open { display: flex; }
#${POPOVER_ID} .int-hdr {
    padding: 8px 12px;
    display: flex; align-items: center; justify-content: space-between;
    background: color-mix(in srgb, var(--c2c-accentSoft2) 10%, transparent);
    border-bottom: 1px solid color-mix(in srgb, var(--c2c-accentSoft2) 20%, transparent);
    font-weight: 600;
}
#${POPOVER_ID} .int-lvl-ok    { color: var(--c2c-okSoft2); }
#${POPOVER_ID} .int-lvl-warn  { color: var(--c2c-warnSoft); }
#${POPOVER_ID} .int-lvl-err   { color: var(--c2c-dangerTint); }
#${POPOVER_ID} .int-lvl-crit  { color: var(--c2c-violetTint); }
#${POPOVER_ID} .int-body { padding: 8px 12px; overflow-y: auto; }
#${POPOVER_ID} .int-sec {
    border-top: 1px dashed color-mix(in srgb, var(--c2c-highlightBase) 8%, transparent);
    padding: 7px 0 9px;
}
#${POPOVER_ID} .int-sec:first-child { border-top: 0; padding-top: 0; }
#${POPOVER_ID} .int-sec h4 {
    margin: 0 0 4px; font-size: 11px; font-weight: 600;
    color: var(--c2c-slate300); text-transform: uppercase; letter-spacing: 0.04em;
}
#${POPOVER_ID} .int-row {
    display: flex; justify-content: space-between; gap: 8px;
    padding: 1px 0; opacity: 0.92;
}
#${POPOVER_ID} .int-row .v { font-variant-numeric: tabular-nums; opacity: 0.85; }
#${POPOVER_ID} .int-row.warn .v { color: var(--c2c-warnSoft); }
#${POPOVER_ID} .int-row.err  .v { color: var(--c2c-dangerTint); }
#${POPOVER_ID} .int-row.crit .v { color: var(--c2c-violetTint); }
#${POPOVER_ID} .int-top {
    margin-top: 4px; padding: 4px 7px; border-radius: 5px;
    background: color-mix(in srgb, var(--c2c-highlightBase) 4%, transparent);
    font-size: 11px;
}
#${POPOVER_ID} .int-top + .int-top { margin-top: 3px; }
#${POPOVER_ID} .int-top .sev-error   { color: var(--c2c-dangerTint); }
#${POPOVER_ID} .int-top .sev-warning { color: var(--c2c-warnSoft); }
#${POPOVER_ID} .int-actions {
    display: flex; gap: 6px; padding: 8px 12px;
    border-top: 1px solid color-mix(in srgb, var(--c2c-highlightBase) 8%, transparent);
    background: color-mix(in srgb, var(--c2c-shadowBase) 18%, transparent);
}
#${POPOVER_ID} .int-actions button {
    flex: 1 1 0; cursor: pointer; padding: 5px 8px;
    border-radius: 5px; border: 1px solid color-mix(in srgb, var(--c2c-highlightBase) 18%, transparent);
    background: color-mix(in srgb, var(--c2c-highlightBase) 6%, transparent); color: var(--c2c-accentNeutral);
    font-size: 11px;
}
#${POPOVER_ID} .int-actions button:hover { background: color-mix(in srgb, var(--c2c-highlightBase) 14%, transparent); }
#${POPOVER_ID} .int-meta {
    font-size: 10px; opacity: 0.55; padding: 4px 12px 8px;
}
#${POPOVER_ID} .int-gh-rows { display: flex; flex-direction: column; gap: 2px; margin-top: 4px; }
#${POPOVER_ID} .int-gh-row {
    display: flex; align-items: center; gap: 6px;
    padding: 4px 6px; cursor: pointer; border-radius: 4px;
    background: color-mix(in srgb, var(--c2c-highlightBase) 4%, transparent);
    border: 1px solid color-mix(in srgb, var(--c2c-highlightBase) 6%, transparent);
    font-size: 11px;
}
#${POPOVER_ID} .int-gh-row:hover {
    background: color-mix(in srgb, var(--c2c-accentSoft2) 14%, transparent);
    border-color: color-mix(in srgb, var(--c2c-accentSoft2) 28%, transparent);
}
#${POPOVER_ID} .int-gh-tag {
    flex: 0 0 auto; font-size: 9.5px; padding: 1px 6px; border-radius: 9px;
    text-transform: uppercase; letter-spacing: 0.4px;
    background: color-mix(in srgb, var(--c2c-highlightBase) 8%, transparent);
}
#${POPOVER_ID} .int-gh-tag.err   { color: var(--c2c-danger); }
#${POPOVER_ID} .int-gh-tag.warn  { color: var(--c2c-warn); }
#${POPOVER_ID} .int-gh-tag.crit  { color: var(--c2c-purple, var(--c2c-violetSoft)); }
#${POPOVER_ID} .int-gh-text {
    flex: 1 1 auto; min-width: 0; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
    color: var(--c2c-accentText);
}
`;
    document.head.appendChild(s);
}

function _currentWorkflow() {
    try {
        const g = app?.graph;
        if (!g || typeof g.serialize !== "function") return null;
        return g.serialize();
    } catch { return null; }
}

function _graphHealthCounts() {
    try {
        const gh = window.__C2C_GRAPH_HEALTH__;
        if (!gh) return { available: false };
        return {
            available: true,
            dead: (gh.dead || []).length,
            cycles: (gh.cycles || []).length,
            dangling: (gh.dangling || []).length,
        };
    } catch { return { available: false }; }
}

function _mergeGraphHealthIntoLevel(level, gh) {
    if (!gh || !gh.available) return level;
    const order = { ok: 0, warn: 1, err: 2, crit: 3 };
    let bumped = level;
    if (gh.cycles > 0) {
        // Cycles → the graph cannot execute → error.
        if (order[bumped] < order.err) bumped = "err";
    }
    if (gh.dead > 0 || gh.dangling > 0) {
        if (order[bumped] < order.warn) bumped = "warn";
    }
    return bumped;
}

// Plain-English error translator for the Doctor popover. Maps common
// Python exception types to a one-line user-facing explanation.
//
// Track D.1 — now server-augmented: the local switch is the *instant*
// fallback used while the popover renders, and a background fetch to
// /mec/translate_error replaces it with the server's rule-pack answer
// (which can match Flux/Wan/SD3/GGUF/etc. rules that the JS switch
// cannot). Result is cached per-(exc_type, exc_msg) so reopening the
// popover for the same error is instant on subsequent renders.
const _PLAIN_ENGLISH_CACHE = new Map();   // key -> {text, ts}
const _PLAIN_ENGLISH_INFLIGHT = new Map(); // key -> Promise

function _plainEnglishCacheKey(err) {
    const t = String(err?.exc_type || "");
    const m = String(err?.exc_msg  || "").slice(0, 160);
    return `${t}|${m}`;
}

function _plainEnglishLocal(err) {
    try {
        const t = String(err?.exc_type || "").trim();
        const m = String(err?.exc_msg  || "").toLowerCase();
        const T = {
            "OutOfMemoryError":       "GPU ran out of VRAM. Try a smaller batch or lower resolution.",
            "torch.cuda.OutOfMemoryError": "GPU ran out of VRAM. Try a smaller batch or lower resolution.",
            "RuntimeError":           "Something went wrong while a node was running.",
            "FileNotFoundError":      "A file the workflow asked for is missing on disk.",
            "PermissionError":        "ComfyUI is not allowed to read or write that path.",
            "ModuleNotFoundError":    "A Python package is missing. Install it (pip install ...) and restart.",
            "ImportError":            "A Python package failed to import \u2014 likely version mismatch.",
            "ValueError":             "A node received an input that isn't a valid value.",
            "TypeError":              "A node received an input of the wrong type.",
            "KeyError":               "A dictionary lookup failed (often a missing config key).",
            "IndexError":             "A list/tensor index is out of range.",
            "ConnectionError":        "A network call to an external service failed.",
            "TimeoutError":           "Something took too long and was cancelled.",
            "JSONDecodeError":        "A JSON file is corrupt or has trailing/missing commas.",
            "AssertionError":         "A safety check inside a node failed \u2014 see the message below.",
            "NotImplementedError":    "This node feature is not implemented yet.",
            "ZeroDivisionError":      "A divide-by-zero happened \u2014 likely an empty mask or zero-length input.",
        };
        if (T[t]) {
            if ((t === "OutOfMemoryError" || t.endsWith("OutOfMemoryError")) && m.includes("gib")) {
                return T[t] + " (saw a GiB allocation request in the error).";
            }
            return T[t];
        }
        if (m.includes("out of memory") || m.includes("oom")) return T["OutOfMemoryError"];
        if (m.includes("no such file")) return T["FileNotFoundError"];
        if (m.includes("permission denied")) return T["PermissionError"];
        return t ? `(${t}) \u2014 see message below.` : "See message below.";
    } catch { return ""; }
}

async function _plainEnglishFromServer(err) {
    const exc_type = String(err?.exc_type || "").trim();
    const message  = String(err?.exc_msg  || "");
    if (!message && !exc_type) return "";
    try {
        const resp = await fetch("/mec/translate_error", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ exc_type, message }),
        });
        if (!resp.ok) return "";
        const j = await resp.json();
        // /mec/translate_error returns either {success:true,data:{cause,fixes,...}}
        // or {explanation:{...}}; handle both robustly.
        const payload = (j && j.data) || j || {};
        const cause = payload.cause || payload.summary || payload.root_cause || "";
        return String(cause || "").trim();
    } catch { return ""; }
}

function _plainEnglishError(err) {
    if (!err) return "";
    const key = _plainEnglishCacheKey(err);
    const cached = _PLAIN_ENGLISH_CACHE.get(key);
    if (cached && cached.text) return cached.text;
    // Kick off the upgrade-to-server-answer once per error fingerprint.
    if (!_PLAIN_ENGLISH_INFLIGHT.has(key)) {
        const p = _plainEnglishFromServer(err).then((serverText) => {
            if (serverText) {
                _PLAIN_ENGLISH_CACHE.set(key, { text: serverText, ts: Date.now() });
                _refreshPopoverPlainEnglish(err, serverText);
            }
            _PLAIN_ENGLISH_INFLIGHT.delete(key);
            return serverText;
        });
        _PLAIN_ENGLISH_INFLIGHT.set(key, p);
    }
    return _plainEnglishLocal(err);
}

// Update any visible Doctor popover plain-English line in place when the
// server response arrives. Scoped to the badge's own popover so we never
// touch unrelated DOM.
function _refreshPopoverPlainEnglish(err, text) {
    try {
        if (!text) return;
        const root = document.getElementById(POPOVER_ID);
        if (!root) return;
        const slot = root.querySelector(".int-top div[data-c2c-plain]");
        if (slot && slot.dataset.c2cKey === _plainEnglishCacheKey(err)) {
            // textContent for XSS safety
            slot.textContent = text;
        }
    } catch { /* defensive */ }
}

function _buildChip() {
    if (_chipEl) return _chipEl;
    const btn = document.createElement("button");
    btn.id = CHIP_EL_ID;
    btn.type = "button";
    btn.className = "c2c-omnibar-slot-pill";
    btn.setAttribute("aria-haspopup", "dialog");
    btn.setAttribute("aria-expanded", "false");
    btn.setAttribute("aria-label", "Doctor (workspace health)");
    btn.dataset.state = "idle";
    btn.title = "Doctor — unified workspace health (click for breakdown)";
    const dot = document.createElement("span");
    dot.className = "c2c-int-dot";
    btn.appendChild(dot);
    const lbl = document.createElement("span");
    lbl.className = "c2c-int-lbl";
    lbl.textContent = "🩺 Doctor";
    btn.appendChild(lbl);
    const val = document.createElement("span");
    val.className = "c2c-int-val";
    val.textContent = "…";
    btn.appendChild(val);
    btn.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        _togglePopover();
    });
    _chipEl = btn;
    _chipDot = dot;
    _chipLbl = lbl;
    _chipVal = val;
    return btn;
}

function _renderChip(health) {
    if (!_chipEl) return;
    const lvl = (health && health.level) || "idle";
    const lbl = (health && health.label) || "—";
    const c = (health && health.counts) || {};
    const totalIssues = (c.doctor_errors || 0) + (c.doctor_warnings || 0)
        + (c.runtime_errors || 0) + (c.integrity_events || 0)
        + (c.registry_failures || 0);
    const tooltipLines = [
        `Status: ${lbl}`,
        `Doctor: ${c.doctor_errors || 0} err / ${c.doctor_warnings || 0} warn / ${c.doctor_infos || 0} info`,
        `Runtime: ${c.runtime_errors || 0} err in last ${health?.window_s ?? "—"}s (${c.runtime_total || 0} total)`,
        `Integrity: ${c.integrity_events || 0} events, drift=${c.checksum_drift || 0}`,
        `Registry: ${c.registry_failures || 0} failures`,
        `OOMs: ${c.ooms_recent || 0}`,
        `VRAM peak: ${(health?.vram?.peak_mb_recent || 0).toFixed(1)} MB`,
    ];
    const value = totalIssues > 0 ? String(totalIssues) : (lvl === "idle" ? "…" : lbl);
    const state = lvl === "idle" ? "idle" : lvl;
    _chipEl.dataset.state = state;
    _chipEl.title = tooltipLines.join("\n");
    if (_chipVal) _chipVal.textContent = value;
}

function _onSlotMode(mode) {
    _slotMode = (mode === "icon") ? "icon" : "full";
    if (_chipEl) _chipEl.dataset.c2cMode = _slotMode;
}

function _ensurePopover() {
    _injectStyle();
    let pop = document.getElementById(POPOVER_ID);
    if (pop) return pop;
    pop = document.createElement("div");
    pop.id = POPOVER_ID;
    pop.setAttribute("role", "dialog");
    pop.setAttribute("aria-label", "Doctor breakdown");
    document.body.appendChild(pop);
    // Click-away closes — ignore clicks on the chip itself (which has its
    // own toggle handler) and inside the popover.
    document.addEventListener("mousedown", (ev) => {
        if (!_popoverOpen) return;
        if (pop.contains(ev.target)) return;
        if (_chipEl && _chipEl.contains(ev.target)) return;
        _closePopover();
    });
    document.addEventListener("keydown", (ev) => {
        if (_popoverOpen && ev.key === "Escape") _closePopover();
    });
    return pop;
}

function _positionPopover() {
    const pop = document.getElementById(POPOVER_ID);
    if (!pop || !_popoverOpen) return;
    const margin = 8;
    const vw = window.innerWidth || 1;
    const vh = window.innerHeight || 1;
    const pw = pop.offsetWidth || 360;
    const ph = pop.offsetHeight || 320;
    // Anchor: chip rect when chip is visible in DOM (inside OmniBar panel),
    // else fall back to the OmniBar pill, else fall back to top-right.
    let anchor = null;
    if (_chipEl && _chipEl.isConnected && _chipEl.offsetParent !== null) {
        anchor = _chipEl.getBoundingClientRect();
    } else {
        try {
            const pill = window.C2COmniBar?.getPill?.();
            if (pill) anchor = pill.getBoundingClientRect();
        } catch { /* */ }
    }
    let top, left;
    if (anchor) {
        // Place below the anchor by default; flip above when clipped.
        top = anchor.bottom + 6;
        // Align right edges of popover and anchor.
        left = anchor.right - pw;
        if (top + ph + margin > vh && anchor.top - ph - 6 > margin) {
            top = anchor.top - ph - 6;
        }
    } else {
        top = margin + 40;
        left = vw - pw - margin;
    }
    left = Math.max(margin, Math.min(left, vw - pw - margin));
    top  = Math.max(margin, Math.min(top,  vh - ph - margin));
    pop.style.top  = top  + "px";
    pop.style.left = left + "px";
}

function _closePopover() {
    const pop = document.getElementById(POPOVER_ID);
    if (pop) pop.classList.remove("open");
    if (_chipEl) _chipEl.setAttribute("aria-expanded", "false");
    _popoverOpen = false;
    window.removeEventListener("resize", _positionPopover);
    window.removeEventListener("scroll", _positionPopover, true);
}

function _openDoctorPanel() {
    try {
        const d = window.__C2C_DOCTOR__;
        if (d && typeof d.refresh === "function") {
            d.refresh();
        }
    } catch { /* */ }
    // Try button in topbar
    const sel = [
        "button#c2c-doctor-btn",
        "button[data-c2c-doctor]",
        'button[title*="Doctor" i]',
    ];
    for (const s of sel) {
        const el = document.querySelector(s);
        if (el && typeof el.click === "function") { el.click(); return; }
    }
}

function _openIntegrityPanel() {
    try {
        const i = window.__MEC_INTEGRITY__;
        if (i && typeof i.open === "function") { i.open(); return; }
    } catch { /* */ }
    const el = document.getElementById("mec-integrity-btn");
    if (el && typeof el.click === "function") el.click();
}

function _openGraphHealthPanel() {
    // Migrated to inline panel inside the INT popover. Just scroll the
    // Graph-health section into view + ensure the popover is open.
    try {
        if (!_popoverOpen) _togglePopover();
        const pop = document.getElementById(POPOVER_ID);
        const sec = pop?.querySelector('[data-int-section="graph-health"]');
        if (sec && typeof sec.scrollIntoView === "function") {
            sec.scrollIntoView({ block: "nearest", behavior: "smooth" });
        }
    } catch { /* */ }
}

function _renderPopover(health) {
    const pop = _ensurePopover();
    const lvl = (health && health.level) || "idle";
    const lbl = (health && health.label) || "—";
    const c = (health && health.counts) || {};
    const s = (health && health.sections) || {};
    // Graph health no longer rendered inside the Doctor popover — it now
    // lives in the CX bar popover (top-center) to keep Doctor focused on
    // health/integrity/lint/env.

    const sevRows = [
        { k: "doctor_errors",     l: "Workflow errors",      cls: "err",  v: c.doctor_errors || 0 },
        { k: "doctor_warnings",   l: "Workflow warnings",    cls: "warn", v: c.doctor_warnings || 0 },
        { k: "runtime_errors",    l: "Runtime errors (recent)", cls: "err",  v: c.runtime_errors || 0 },
        { k: "ooms_recent",       l: "OOM signals (recent)", cls: "crit", v: c.ooms_recent || 0 },
        { k: "integrity_errors",  l: "Integrity errors",     cls: "err",  v: c.integrity_errors || 0 },
        { k: "integrity_events",  l: "Integrity events",     cls: "warn", v: c.integrity_events || 0 },
        { k: "checksum_drift",    l: "Checksum drift",       cls: "warn", v: c.checksum_drift || 0 },
        { k: "registry_failures", l: "Component failures",   cls: "err",  v: c.registry_failures || 0 },
    ];
    // Aggregate count for the Summary header bar.
    const totalErr  = (c.doctor_errors||0) + (c.runtime_errors||0) + (c.integrity_errors||0) + (c.registry_failures||0) + (c.env_errors||0);
    const totalWarn = (c.doctor_warnings||0) + (c.integrity_events||0) + (c.checksum_drift||0) + (c.env_warnings||0);
    const totalCrit = (c.ooms_recent||0);

    const escape = (str) => String(str ?? "").replace(/[&<>]/g, (m) => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m]));

    const topRows = (s.doctor && s.doctor.top || []).map((f) => `
        <div class="int-top">
            <span class="sev-${escape(f.severity)}">●</span>
            <b>${escape(f.rule || "?")}</b>
            ${f.node_type ? `<span style="opacity:0.65;">\u00B7 ${escape(f.node_type)}#${escape(f.node_id ?? "?")}</span>` : ""}
            <div style="opacity:0.85; margin-top:2px;">${escape(f.detail || "")}</div>
        </div>`).join("");

    const ig = s.integrity || {};
    const rg = s.registry || {};
    const rt = s.runtime || {};
    const en = s.environment || {};
    const pyenvRaw = en.pyenv || {};
    const diskRaw = en.disk || {};
    // Normalise nested aggregator shape to flat fields used by the renderer.
    const pyenv = {
        python_version: pyenvRaw.python_version
            || (pyenvRaw.python && pyenvRaw.python.version) || "",
        torch_version: pyenvRaw.torch_version
            || (pyenvRaw.device && pyenvRaw.device.torch) || "",
        cuda_available: pyenvRaw.cuda_available != null
            ? pyenvRaw.cuda_available
            : !!(pyenvRaw.device && pyenvRaw.device.cuda_available),
    };
    // Disk: aggregator returns {drive:{total,used,free}, sections:{models:{...}}}.
    // Derive flat models_total_gb and free_gb if not provided.
    let _models_gb = diskRaw.models_total_gb;
    if (_models_gb == null && diskRaw.sections && diskRaw.sections.models) {
        const m = diskRaw.sections.models;
        const bytes = (typeof m.bytes === "number") ? m.bytes
            : (typeof m.total_bytes === "number") ? m.total_bytes
            : (typeof m.size_bytes === "number") ? m.size_bytes : null;
        if (bytes != null) _models_gb = bytes / (1024 ** 3);
    }
    let _free_gb = diskRaw.free_gb;
    if (_free_gb == null && diskRaw.drive && typeof diskRaw.drive.free === "number") {
        _free_gb = diskRaw.drive.free / (1024 ** 3);
    }
    const disk = { models_total_gb: _models_gb, free_gb: _free_gb };

    pop.innerHTML = `
        <div class="int-hdr">
            <span>🩺 Doctor · <span class="int-lvl-${escape(lvl)}">${escape(lbl)}</span></span>
            <span style="font-size:10px; opacity:0.6;">${(health?.window_s ?? DEFAULT_WINDOW_S)}s window</span>
        </div>
        <div class="int-body">
            <div class="int-sec">
                <h4>Summary</h4>
                <div class="int-row ${totalErr ? "err" : (totalWarn ? "warn" : "")}">
                    <span>Health snapshot</span>
                    <span class="v">${totalErr} err · ${totalWarn} warn${totalCrit ? " · " + totalCrit + " crit" : ""}</span>
                </div>
                ${sevRows.map((r) => `
                    <div class="int-row ${r.v > 0 ? r.cls : ""}">
                        <span>${escape(r.l)}</span>
                        <span class="v">${r.v}</span>
                    </div>`).join("")}
            </div>

            ${(s.doctor && s.doctor.ran) ? `
            <div class="int-sec">
                <h4>Workflow lint</h4>
                <div class="int-row"><span>Total findings</span><span class="v">${s.doctor.total || 0}</span></div>
                ${topRows || `<div style="opacity:0.55; font-size:11px;">No errors or warnings.</div>`}
            </div>` : ""}

            <div class="int-sec">
                <h4>Package integrity</h4>
                ${ig.available ? `
                    <div class="int-row ${ig.pip_check_ok === false ? "crit" : ""}">
                        <span>pip check</span>
                        <span class="v">${ig.pip_check_ok === false ? "FAIL" : "ok"}</span>
                    </div>
                    ${ig.pip_check_ok === false && ig.pip_check_detail ? `
                        <div class="int-top"><b>Conflicts</b>
                            <div style="opacity:0.85; margin-top:2px; white-space:pre-wrap; font-family:ui-monospace,monospace; font-size:10px;">${escape(String(ig.pip_check_detail).slice(0, 800))}</div>
                        </div>` : ""}
                    <div class="int-row"><span>Scan ready</span><span class="v">${ig.ready ? "yes" : "scanning…"}</span></div>
                    <div class="int-row"><span>Events</span><span class="v">${ig.events_total || 0}</span></div>
                    ${(ig.suspicious_files && ig.suspicious_files > 0) ? `
                        <div class="int-row crit"><span>⚠ Suspicious files</span><span class="v">${ig.suspicious_files}</span></div>` : ""}
                    ${(ig.checksum_drift && ig.checksum_drift > 0) ? `
                        <div class="int-row warn"><span>Checksum drift</span><span class="v">${ig.checksum_drift}</span></div>` : ""}
                ` : `<div style="opacity:0.55; font-size:11px;">Integrity scan unavailable.</div>`}
            </div>

            <div class="int-sec">
                <h4>Runtime</h4>
                ${rt.available ? `
                    <div class="int-row"><span>Buffer size</span><span class="v">${rt.buffer_len || 0}</span></div>
                    <div class="int-row"><span>Recent total</span><span class="v">${rt.recent_total || 0}</span></div>
                    <div class="int-row ${rt.recent_errors ? "err" : ""}"><span>Recent errors</span><span class="v">${rt.recent_errors || 0}</span></div>
                    ${rt.last_error ? `<div class="int-top"><span class="sev-error">●</span>
                        <b>${escape(rt.last_error.exc_type || "Error")}</b>
                        <div data-c2c-plain data-c2c-key="${escape(_plainEnglishCacheKey(rt.last_error))}" style="opacity:0.95; margin-top:4px; font-weight:600;">${escape(_plainEnglishError(rt.last_error))}</div>
                        <div style="opacity:0.7; margin-top:2px; font-size:10px; font-family:ui-monospace,monospace;">${escape((rt.last_error.exc_msg || "").slice(0, 200))}</div>
                    </div>` : ""}
                ` : `<div style="opacity:0.55; font-size:11px;">Diagnostics ring unavailable.</div>`}
            </div>

            <div class="int-sec">
                <h4>Component registry</h4>
                ${rg.available ? `
                    <div class="int-row ${rg.failures ? "err" : ""}"><span>Failures</span><span class="v">${rg.failures}</span></div>
                    <div class="int-row"><span>Missing deps</span><span class="v">${rg.missing_deps}</span></div>
                    <div class="int-row"><span>Missing weights</span><span class="v">${rg.missing_weights}</span></div>
                    <div class="int-row"><span>Ready</span><span class="v">${rg.ready}</span></div>
                ` : `<div style="opacity:0.55; font-size:11px;">Registry unavailable.</div>`}
            </div>

            <div class="int-sec" data-int-section="environment">
                <h4>Environment</h4>
                ${en.available ? `
                    <div class="int-row ${(en.py_errors||0) ? "err" : ((en.py_warnings||0) ? "warn" : "")}">
                        <span>Python packages</span>
                        <span class="v">${(en.py_errors||0)} err / ${(en.py_warnings||0)} warn</span>
                    </div>
                    ${pyenv.python_version ? `<div class="int-row"><span>Python</span><span class="v">${escape(pyenv.python_version)}</span></div>` : ""}
                    ${pyenv.torch_version ? `<div class="int-row"><span>Torch</span><span class="v">${escape(pyenv.torch_version)}${pyenv.cuda_available ? " (CUDA)" : ""}</span></div>` : ""}
                    ${disk.models_total_gb != null ? `<div class="int-row"><span>Models on disk</span><span class="v">${Number(disk.models_total_gb).toFixed(1)} GB</span></div>` : ""}
                    ${disk.free_gb != null ? `<div class="int-row ${Number(disk.free_gb) < 20 ? "warn" : ""}"><span>Free disk</span><span class="v">${Number(disk.free_gb).toFixed(1)} GB</span></div>` : ""}
                ` : `<div style="opacity:0.55; font-size:11px;">Environment diagnostics unavailable.</div>`}
            </div>
        </div>

        <div class="int-actions">
            <button data-act="doctor">Doctor</button>
            <button data-act="integrity">Integrity</button>
            <button data-act="graph">Graph</button>
            <button data-act="refresh">Refresh</button>
        </div>
        <div class="int-meta">updated ${new Date((health?.ts || Date.now()/1000) * 1000).toLocaleTimeString()}</div>
    `;
    pop.querySelector('[data-act="doctor"]').addEventListener("click",
        () => { _closePopover(); _openDoctorPanel(); });
    pop.querySelector('[data-act="integrity"]').addEventListener("click",
        () => { _closePopover(); _openIntegrityPanel(); });
    pop.querySelector('[data-act="graph"]').addEventListener("click",
        () => { _openGraphHealthPanel(); });
    pop.querySelector('[data-act="refresh"]').addEventListener("click",
        () => { _refreshNow(true); });

    _hydrateGraphHealthRows(pop);
}

function _hydrateGraphHealthRows(pop) {
    try {
        const host = pop.querySelector(".int-gh-rows");
        if (!host) return;
        const gh = window.__C2C_GRAPH_HEALTH__;
        const items = (gh && Array.isArray(gh.items)) ? gh.items : [];
        if (!items.length) {
            host.innerHTML = "";
            return;
        }
        host.innerHTML = items.slice(0, 24).map((r) => {
            const tagCls = r.tag === "cycle" ? "err" : (r.tag === "dead" ? "warn" : "crit");
            const txt = escape(r.text || "");
            return `<div class="int-gh-row" data-id="${escape(String(r.id))}" data-tag="${escape(r.tag)}">
                        <span class="int-gh-tag ${tagCls}">${escape(r.tag)}</span>
                        <span class="int-gh-text">${txt}</span>
                    </div>`;
        }).join("");
        host.querySelectorAll(".int-gh-row").forEach((el) => {
            el.addEventListener("click", () => {
                const id = el.dataset.id;
                try { window.__C2C_GRAPH_HEALTH__?.focus?.(id); } catch { /* */ }
            });
        });
    } catch { /* */ }
}

function _togglePopover() {
    const pop = _ensurePopover();
    if (_popoverOpen) { _closePopover(); return; }
    _renderPopover(_lastHealth);
    pop.classList.add("open");
    _popoverOpen = true;
    if (_chipEl) _chipEl.setAttribute("aria-expanded", "true");
    // Measure after the dialog flips to display:flex.
    requestAnimationFrame(_positionPopover);
    window.addEventListener("resize", _positionPopover, { passive: true });
    window.addEventListener("scroll", _positionPopover, { passive: true, capture: true });
}

async function _refreshNow(force = false) {
    if (_busy && !force) return;
    _busy = true;
    try {
        const window_s = +(_getSetting(SETTING_WINDOW, DEFAULT_WINDOW_S)) || DEFAULT_WINDOW_S;
        const wf = _currentWorkflow();
        let r;
        if (wf) {
            r = await api.fetchApi("/c2c/int/health", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ workflow: wf, window_s }),
            });
        } else {
            r = await api.fetchApi(`/c2c/int/health?window_s=${window_s}`);
        }
        if (!r.ok) {
            _lastHealth = null;
            _renderChip({ level: "idle", label: `HTTP ${r.status}`, counts: {} });
            return;
        }
        const j = await r.json();
        // Bump level with live graph-health that the JS already has cached.
        const gh = _graphHealthCounts();
        j.level = _mergeGraphHealthIntoLevel(j.level || "ok", gh);
        if (gh.available) {
            j.counts = j.counts || {};
            j.counts.graph_cycles = gh.cycles;
            j.counts.graph_dead = gh.dead;
            j.counts.graph_dangling = gh.dangling;
            if (gh.cycles + gh.dead + gh.dangling > 0 && j.label === "Healthy") {
                j.label = (j.level === "err") ? "Errors" : "Degraded";
            }
        }
        _lastHealth = j;
        _renderChip(j);
        if (_popoverOpen) _renderPopover(j);
    } catch (e) {
        console.warn("[c2c_int_badge] refresh failed:", e);
    } finally {
        _busy = false;
    }
}

function _startPolling() {
    if (_timer) clearInterval(_timer);
    const ms = Math.max(MIN_POLL_MS, +(_getSetting(SETTING_POLL, DEFAULT_POLL_MS)) || DEFAULT_POLL_MS);
    _timer = setInterval(_refreshNow, ms);
}

function _wireBackgroundEvents() {
    // Refresh shortly after the graph stops executing.
    try {
        api.addEventListener("execution_success", () => setTimeout(_refreshNow, 250));
        api.addEventListener("execution_error",   () => setTimeout(_refreshNow, 250));
        api.addEventListener("status",            (ev) => {
            const q = ev?.detail?.exec_info?.queue_remaining;
            if (q === 0) setTimeout(_refreshNow, 300);
        });
    } catch { /* */ }
    // Refresh when graph health publishes new counts.
    try {
        window.addEventListener("c2c:graph-health-updated", () => {
            if (_lastHealth) {
                const gh = _graphHealthCounts();
                _lastHealth.level = _mergeGraphHealthIntoLevel(_lastHealth.level || "ok", gh);
                _renderChip(_lastHealth);
                if (_popoverOpen) _renderPopover(_lastHealth);
            }
        });
    } catch { /* */ }
}

function _registerChip() {
    if (!window.C2COmniBar || typeof window.C2COmniBar.register !== "function") {
        return false;
    }
    if (_unregisterSlot) return true; // already registered
    _injectStyle();
    const el = _buildChip();
    try {
        _unregisterSlot = window.C2COmniBar.register({
            section: "stats",
            id: CHIP_ID,
            order: 10,
            element: el,
            // OmniBar polls update() ~1 Hz when the panel is open. Use it as
            // a cheap "tick" — _refreshNow is rate-limited by _busy + the
            // server-side poll interval, so this won't spam.
            update: () => { /* polling owned by _startPolling; no-op here */ },
            onMode: _onSlotMode,
        });
    } catch (e) {
        console.warn("[c2c_int_badge] OmniBar register failed:", e);
        return false;
    }
    return true;
}

app.registerExtension({
    name: "C2C.IntBadge",
    settings: [
        {
            id: SETTING_WINDOW,
            name: "INT badge: runtime window (seconds)",
            type: "number",
            defaultValue: DEFAULT_WINDOW_S,
            tooltip: "How far back to look for runtime errors / OOMs / VRAM peaks.",
        },
        {
            id: SETTING_POLL,
            name: "INT badge: poll interval (ms)",
            type: "number",
            defaultValue: DEFAULT_POLL_MS,
            tooltip: `Minimum ${MIN_POLL_MS} ms.`,
            onChange: () => _startPolling(),
        },
    ],
    async setup() {
        // Wait briefly for C2COmniBar to be defined — its module may load
        // a tick later than ours depending on file ordering.
        for (let i = 0; i < 40; i++) {
            if (_registerChip()) break;
            await new Promise((r) => setTimeout(r, 100));
        }
        _wireBackgroundEvents();
        _startPolling();
        // First refresh in a moment so other extensions have time to mount.
        setTimeout(_refreshNow, 600);
        if (typeof window !== "undefined") {
            window.__C2C_INT__ = {
                refresh: _refreshNow,
                state: () => _lastHealth,
                open: _togglePopover,
                close: _closePopover,
                element: () => _chipEl,
            };
        }
    },
});
