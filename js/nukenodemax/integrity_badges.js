// FILE: js/nukenodemax/integrity_badges.js
// FEATURE: W2 — Integrity status surfaced via a topbar button + modal
//                (the old corner bar was annoying & easy to miss).
//
// Design goals:
//   - NO modifications to ComfyUI core code.
//   - NO fixed-position corner popup.
//   - One small button next to the Comfy-Manager button in the same
//     top toolbar. Click -> modal dialog with the full event list.
//   - User can "Mute warnings" so the badge stays neutral even when
//     events exist (decision persists across reloads via localStorage).
//   - Per-node checksum-drift dot badge stays (it's a tiny, in-context
//     visual cue, not a popup).
//   - Live state exposed on window.__MEC_INTEGRITY__ so the MEC
//     Diagnostics sidebar can render the same data inline.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// ── State ───────────────────────────────────────────────────────────
const STATE = {
    events: [],
    pipOk: true,
    drift: 0,
    usedUv: false,
    fromCache: false,
    lastUpdated: null,
};

const LS_MUTE_KEY = "MEC.integrity.muted";
function isMuted() { return localStorage.getItem(LS_MUTE_KEY) === "1"; }
function setMuted(v) {
    if (v) localStorage.setItem(LS_MUTE_KEY, "1");
    else   localStorage.removeItem(LS_MUTE_KEY);
}

// Subscribers (sidebar uses this).
const SUBSCRIBERS = new Set();
function notify() {
    for (const cb of SUBSCRIBERS) {
        try { cb(STATE); } catch (e) { /* ignore */ }
    }
}

// Expose for the sidebar + external scripts.
window.__MEC_INTEGRITY__ = {
    get state() { return { ...STATE }; },
    subscribe(cb) { SUBSCRIBERS.add(cb); cb(STATE); return () => SUBSCRIBERS.delete(cb); },
    async refresh() { return fetchReport(); },
    open() { openDialog(); },
    isMuted, setMuted,
    reinstall: (pkg) => reinstall(pkg),
};

// ── Network ─────────────────────────────────────────────────────────
async function fetchReport() {
    try {
        const r = await fetch("/nukenodemax/integrity_report");
        const j = await r.json();
        ingest(j);
        return j;
    } catch (e) {
        console.warn("[MEC.integrity] fetch failed:", e);
        return null;
    }
}

function ingest(d) {
    if (!d) return;
    STATE.events = d.events || [];
    STATE.pipOk = d.pip_check ? !!d.pip_check.ok : true;
    STATE.drift = (d.checksum_drift || []).length;
    STATE.usedUv = !!d.used_uv;
    STATE.backend = d.backend || (d.used_uv ? "uv" : "pip");
    STATE.fromCache = !!d.from_cache;
    STATE.ready = d.ready !== false;
    STATE.status = d.status || (STATE.ready ? "ok" : "scanning");
    STATE.envKind = d.env_kind || "?";
    STATE.platform = d.platform || "?";
    STATE.python = d.python || "";
    STATE.pythonVersion = d.python_version || "";
    STATE.lastUpdated = Date.now();
    refreshButton();
    refreshDialogIfOpen();
    if (app.graph) app.graph.setDirtyCanvas(true, true);
    notify();
}

async function reinstall(pkg) {
    if (!confirm(`Reinstall package "${pkg}"?\nThis runs uv/pip install --reinstall.`)) return;
    const r = await fetch(`/nukenodemax/reinstall?package=${encodeURIComponent(pkg)}&confirm=yes`,
        { method: "POST" });
    const j = await r.json();
    if (j.ok) {
        toast(`✓ Reinstalled ${pkg}`, "success");
        await fetchReport();
    } else {
        toast(`✗ Reinstall failed: ${j.error || j.stderr || "unknown"}`, "error");
    }
}

function toast(msg, severity = "info") {
    try {
        app.extensionManager?.toast?.add({
            severity,
            summary: "MEC Integrity",
            detail: msg,
            life: 4000,
        });
    } catch {
        console.log("[MEC.integrity]", msg);
    }
}

// ── Styles ──────────────────────────────────────────────────────────
const STYLE_ID = "mec-integrity-style";
function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
    .mec-integ-btn {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 0 9px; margin: 0 4px;
        height: 28px;
        box-sizing: border-box;
        flex: 0 0 auto;
        align-self: center;
        border-radius: 14px;
        background: #313244;
        color: #cdd6f4;
        border: 1px solid #45475a;
        cursor: pointer;
        font-size: 11px; font-weight: 700;
        letter-spacing: 0.4px;
        font-family: var(--p-font-family, system-ui), sans-serif;
        line-height: 1;
        white-space: nowrap;
        user-select: none;
        transition: background 120ms ease, border-color 120ms ease, color 120ms ease, box-shadow 120ms ease;
    }
    .mec-integ-btn:hover {
        background: #45475a;
        box-shadow: 0 0 0 1px #585b70 inset, 0 2px 6px rgba(0,0,0,0.35);
    }
    .mec-integ-btn .mec-integ-icon { font-size: 14px; line-height: 1; }
    .mec-integ-btn .mec-integ-label {
        font-size: 10px; letter-spacing: 0.6px;
    }
    .mec-integ-btn.ok {
        border-color: #a6e3a1; color: #a6e3a1;
        background: linear-gradient(180deg, #1e2e26 0%, #1e1e2e 100%);
    }
    .mec-integ-btn.warn {
        border-color: #fab387; color: #fab387;
        background: linear-gradient(180deg, #3a2a1e 0%, #1e1e2e 100%);
    }
    .mec-integ-btn.error {
        border-color: #f38ba8; color: #f38ba8;
        background: linear-gradient(180deg, #3a1e26 0%, #1e1e2e 100%);
        animation: mec-integ-pulse 2.4s ease-in-out infinite;
    }
    .mec-integ-btn.scan {
        border-color: #89b4fa; color: #89b4fa;
        background: linear-gradient(180deg, #1e273a 0%, #1e1e2e 100%);
        animation: mec-integ-scan 1.4s ease-in-out infinite;
    }
    .mec-integ-btn.muted { opacity: 0.55; filter: grayscale(0.6); }
    @keyframes mec-integ-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(243,139,168,0.0); }
        50%      { box-shadow: 0 0 0 4px rgba(243,139,168,0.18); }
    }
    @keyframes mec-integ-scan {
        0%, 100% { box-shadow: 0 0 0 0 rgba(137,180,250,0.0); }
        50%      { box-shadow: 0 0 0 4px rgba(137,180,250,0.25); }
    }
    .mec-integ-dot {
        display:inline-flex; align-items:center; justify-content:center;
        min-width:18px; height:18px;
        padding: 0 5px; border-radius: 9px;
        background:#f38ba8; color:#11111b;
        font-size:10px; font-weight:800; text-align:center;
        line-height:1;
        box-shadow: 0 1px 2px rgba(0,0,0,0.35);
    }
    .mec-integ-dot.warn  { background:#fab387; }
    .mec-integ-dot.ok    { background:#a6e3a1; }
    .mec-integ-dot.error { background:#f38ba8; }

    .mec-integ-mask {
        position: fixed; inset: 0; z-index: 99998;
        background: rgba(0,0,0,0.55);
        display: flex; align-items: center; justify-content: center;
        font-family: var(--p-font-family, system-ui), sans-serif;
    }
    .mec-integ-dlg {
        width: min(720px, 92vw); max-height: 80vh;
        background: var(--comfy-menu-bg, #1e1e2e);
        color: var(--fg-color, #cdd6f4);
        border: 1px solid var(--border-color, #45475a);
        border-radius: 8px;
        box-shadow: 0 12px 48px rgba(0,0,0,0.5);
        display: flex; flex-direction: column;
        overflow: hidden;
    }
    .mec-integ-dlg-header {
        display: flex; align-items: center; gap: 8px;
        padding: 10px 14px;
        border-bottom: 1px solid var(--border-color, #313244);
        background: var(--p-content-background, #181825);
    }
    .mec-integ-dlg-title {
        flex: 1; font-weight: 600; font-size: 14px;
    }
    .mec-integ-dlg-meta {
        font-size: 11px; opacity: 0.7;
    }
    .mec-integ-dlg-body {
        flex: 1; overflow: auto; padding: 12px 14px;
        font-size: 12px;
    }
    .mec-integ-dlg-footer {
        display: flex; gap: 8px; align-items: center;
        padding: 8px 14px;
        border-top: 1px solid var(--border-color, #313244);
    }
    .mec-integ-dlg-footer .spacer { flex: 1; }
    .mec-integ-x {
        background: transparent; border: none; color: inherit;
        font-size: 18px; cursor: pointer; padding: 0 4px;
    }
    .mec-integ-x:hover { color: #f38ba8; }
    .mec-integ-row {
        display: flex; gap: 8px; align-items: flex-start;
        padding: 8px 10px; margin-bottom: 6px;
        border-radius: 4px;
        background: var(--p-content-background, #181825);
        border-left: 3px solid #89b4fa;
    }
    .mec-integ-row.warn  { border-left-color: #fab387; }
    .mec-integ-row.error { border-left-color: #f38ba8; }
    .mec-integ-kind {
        font-size: 10px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px;
        opacity: 0.7; min-width: 130px;
    }
    .mec-integ-msg {
        flex: 1;
        font-family: ui-monospace, "Cascadia Mono", monospace;
        font-size: 11px; word-break: break-word;
    }
    .mec-integ-act {
        padding: 3px 10px; border-radius: 4px;
        background: var(--p-primary-color, #89b4fa);
        color: var(--p-primary-color-text, #11111b);
        border: none; cursor: pointer; font-size: 11px;
        font-weight: 600;
    }
    .mec-integ-act:hover { filter: brightness(1.1); }
    .mec-integ-btn-sec {
        padding: 4px 12px; border-radius: 4px;
        background: var(--p-button-secondary-bg, #313244);
        color: var(--fg-color, #cdd6f4);
        border: 1px solid var(--border-color, #45475a);
        cursor: pointer; font-size: 12px;
    }
    .mec-integ-btn-sec:hover { background: var(--p-button-secondary-hover-bg, #45475a); }
    .mec-integ-empty {
        text-align: center; padding: 30px 10px;
        color: var(--descriptions-text-color, #a6adc8);
        font-style: italic;
    }
    `;
    document.head.appendChild(s);
}

// ── Topbar button ───────────────────────────────────────────────────
let BTN_EL = null;

function summary() {
    // Compact pill with a clear text label so users can see the
    // integrity state at a glance. Tooltip + modal still carry full detail.
    const muted = isMuted();
    const issues = STATE.events.length;
    const scanning = STATE.ready === false || STATE.status === "scanning";
    if (scanning) {
        return { tone: "scan", text: "⛨", label: "SCAN", dotClass: "", dot: "" };
    }
    if (muted) {
        return { tone: "muted", text: "⛨", label: "INT", dotClass: "ok", dot: "" };
    }
    if (!STATE.pipOk || issues > 0) {
        const tone = STATE.pipOk ? "warn" : "error";
        return { tone, text: "⛨", label: "INT", dotClass: tone, dot: String(issues || "!") };
    }
    return { tone: "ok", text: "⛨", label: "OK", dotClass: "ok", dot: "" };
}

function refreshButton() {
    const s = summary();
    if (!BTN_EL) return;
    BTN_EL.className = "mec-integ-btn " + s.tone;
    BTN_EL.title = (STATE.lastUpdated
        ? `MEC Integrity — ${STATE.events.length} event(s) — last scan ${new Date(STATE.lastUpdated).toLocaleTimeString()}. Click for details.`
        : `MEC Integrity — ${STATE.events.length} event(s). Click for details.`);
    BTN_EL.innerHTML = "";
    const icon = document.createElement("span");
    icon.className = "mec-integ-icon";
    icon.textContent = s.text;
    BTN_EL.appendChild(icon);
    const lbl = document.createElement("span");
    lbl.className = "mec-integ-label";
    lbl.textContent = s.label;
    BTN_EL.appendChild(lbl);
    if (s.dot !== "") {
        const dot = document.createElement("span");
        dot.className = "mec-integ-dot " + s.dotClass;
        dot.textContent = s.dot;
        BTN_EL.appendChild(dot);
    }
}

function findManagerButton() {
    // Try several known selectors. ComfyUI-Manager versions differ; the
    // common denominator is a top-bar button whose label/title/aria-label
    // contains "Manager".
    const sels = [
        'button#comfyui-manager-button',
        'button[aria-label*="Manager" i]',
        'button[title*="Manager" i]',
        '.comfyui-menu button[label*="Manager" i]',
    ];
    for (const sel of sels) {
        const el = document.querySelector(sel);
        if (el) return el;
    }
    // Text-content scan as last resort.
    for (const b of document.querySelectorAll(".comfyui-menu button, .comfy-menu button, header button")) {
        const t = (b.textContent || "").trim();
        if (/manager/i.test(t)) return b;
    }
    return null;
}

function findFallbackContainer() {
    return (
        document.querySelector(".comfyui-menu .comfyui-menu-right") ||
        document.querySelector(".comfyui-menu") ||
        document.querySelector(".comfy-menu") ||
        null
    );
}

function mountButton() {
    if (BTN_EL && document.body.contains(BTN_EL)) return;
    const btn = document.createElement("button");
    btn.id = "mec-integrity-btn";
    btn.className = "mec-integ-btn";
    btn.type = "button";
    btn.addEventListener("click", openDialog);
    BTN_EL = btn;
    refreshButton();

    const mgr = findManagerButton();
    if (mgr && mgr.parentNode) {
        mgr.parentNode.insertBefore(btn, mgr.nextSibling);
        return;
    }
    const container = findFallbackContainer();
    if (container) {
        container.appendChild(btn);
    } else {
        // Last resort: subtle pin to top-right (only if no menu exists).
        Object.assign(btn.style, {
            position: "fixed", top: "6px", right: "6px", zIndex: 999,
        });
        document.body.appendChild(btn);
    }
}

function startMountObserver() {
    // ComfyUI-Manager may register its button after our setup() runs.
    // Watch the DOM until we find it; stop once stable.
    let tries = 0;
    const observer = new MutationObserver(() => {
        tries++;
        if (BTN_EL && document.body.contains(BTN_EL)) {
            const mgr = findManagerButton();
            if (mgr && BTN_EL.previousSibling !== mgr) {
                mgr.parentNode?.insertBefore(BTN_EL, mgr.nextSibling);
            }
            if (tries > 40) observer.disconnect();
            return;
        }
        mountButton();
        if (tries > 80) observer.disconnect();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    mountButton();
    setTimeout(() => observer.disconnect(), 15000);
}

// ── (Floating fallback widget removed — user-mandated 2026-05) ──────
// Was a draggable pill duplicating the topbar button; deforming the
// navbar layout. Now only the topbar button remains.
function removeStaleFloater() {
    document.getElementById("mec-integrity-float")?.remove();
}

// ── Modal dialog ────────────────────────────────────────────────────
let DLG_EL = null;

function openDialog() {
    if (DLG_EL) return;
    fetchReport();

    const mask = document.createElement("div");
    mask.className = "mec-integ-mask";
    mask.addEventListener("click", (e) => {
        if (e.target === mask) closeDialog();
    });

    const dlg = document.createElement("div");
    dlg.className = "mec-integ-dlg";
    dlg.tabIndex = -1;

    const header = document.createElement("div");
    header.className = "mec-integ-dlg-header";
    header.innerHTML = `
        <span class="mec-integ-dlg-title">MEC Integrity Report</span>
        <span class="mec-integ-dlg-meta" data-role="meta"></span>
        <button class="mec-integ-x" data-role="x" title="Close">×</button>`;
    header.querySelector('[data-role="x"]').addEventListener("click", closeDialog);

    const body = document.createElement("div");
    body.className = "mec-integ-dlg-body";
    body.dataset.role = "body";

    const footer = document.createElement("div");
    footer.className = "mec-integ-dlg-footer";
    footer.innerHTML = `
        <label style="display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;">
            <input type="checkbox" data-role="mute" ${isMuted() ? "checked" : ""}>
            Mute warnings (button stays neutral)
        </label>
        <span class="spacer"></span>
        <button class="mec-integ-btn-sec" data-role="refresh">Refresh</button>
        <button class="mec-integ-btn-sec" data-role="close">Close</button>`;
    footer.querySelector('[data-role="mute"]').addEventListener("change", (e) => {
        setMuted(e.target.checked);
        refreshButton();
    });
    footer.querySelector('[data-role="refresh"]').addEventListener("click", async () => {
        await fetchReport();
        toast("Integrity report refreshed");
    });
    footer.querySelector('[data-role="close"]').addEventListener("click", closeDialog);

    dlg.appendChild(header);
    dlg.appendChild(body);
    dlg.appendChild(footer);
    mask.appendChild(dlg);
    document.body.appendChild(mask);
    DLG_EL = mask;

    renderDialogBody();
    setTimeout(() => dlg.focus(), 0);
    document.addEventListener("keydown", onEsc);
}

function closeDialog() {
    if (!DLG_EL) return;
    DLG_EL.remove();
    DLG_EL = null;
    document.removeEventListener("keydown", onEsc);
}

function onEsc(e) { if (e.key === "Escape") closeDialog(); }

function refreshDialogIfOpen() {
    if (DLG_EL) renderDialogBody();
}

function extractPackageName(msg) {
    // uv: "The package `<name>` has multiple installed distributions"
    let m = msg.match(/^The package [`"]([A-Za-z0-9_.\-]+)[`"]/);
    if (m) return m[1];
    // pip: "<pkg> X.Y has requirement <dep>..." or "<pkg> requires <dep>"
    m = msg.match(/^([A-Za-z0-9_.\-]+)\s+\S+\s+(?:has requirement|requires|depends)/i);
    if (m) return m[1];
    return null;
}

function renderDialogBody() {
    if (!DLG_EL) return;
    const body = DLG_EL.querySelector('[data-role="body"]');
    const meta = DLG_EL.querySelector('[data-role="meta"]');
    const events = STATE.events || [];

    if (meta) {
        const parts = [];
        parts.push(STATE.backend || (STATE.usedUv ? "uv" : "pip"));
        parts.push(STATE.fromCache ? "cached" : "fresh");
        if (STATE.status && STATE.status !== "ok") parts.push(STATE.status);
        if (STATE.lastUpdated) parts.push(new Date(STATE.lastUpdated).toLocaleTimeString());
        meta.textContent = parts.join(" · ");
    }

    body.innerHTML = "";
    if (STATE.ready === false || STATE.status === "scanning") {
        body.innerHTML = `<div class="mec-integ-empty">
            <div style="font-size:20px;margin-bottom:6px;">⏳</div>
            Integrity scan running… results will appear here shortly.<br/>
            <span style="opacity:0.7;font-size:11px;">First scan after startup takes ~5–30s depending on installed package count.</span>
        </div>`;
        return;
    }
    if (events.length === 0) {
        body.innerHTML = `<div class="mec-integ-empty">
            ✓ No integrity events. Environment looks clean.<br/>
            <span style="opacity:0.7;font-size:11px;">Backend: ${STATE.backend || "pip"} · Env: ${STATE.envKind || "?"} (${STATE.platform || "?"}) · Python ${STATE.pythonVersion || "?"} · Last scan: ${STATE.lastUpdated ? new Date(STATE.lastUpdated).toLocaleTimeString() : "n/a"}</span>
        </div>`;
        return;
    }

    // Group by kind.
    const groups = {};
    for (const e of events) (groups[e.kind] = groups[e.kind] || []).push(e);

    for (const [kind, list] of Object.entries(groups)) {
        const h = document.createElement("div");
        h.style.cssText = "font-weight:600;opacity:0.75;margin:6px 0 4px;font-size:11px;text-transform:uppercase;letter-spacing:0.4px;";
        h.textContent = `${kind.replace(/_/g, " ")} (${list.length})`;
        body.appendChild(h);

        for (const e of list) {
            const row = document.createElement("div");
            const sev = (e.severity || "warn").toLowerCase();
            row.className = "mec-integ-row " + (sev === "error" ? "error" : sev === "warn" ? "warn" : "");
            const kindEl = document.createElement("div");
            kindEl.className = "mec-integ-kind";
            kindEl.textContent = sev;
            const msgEl = document.createElement("div");
            msgEl.className = "mec-integ-msg";
            msgEl.textContent = e.message;
            row.appendChild(kindEl);
            row.appendChild(msgEl);
            const pkg = e.kind === "dependency_conflict" ? extractPackageName(e.message) : null;
            if (pkg) {
                const act = document.createElement("button");
                act.className = "mec-integ-act";
                act.textContent = `Reinstall ${pkg}`;
                act.addEventListener("click", () => reinstall(pkg));
                row.appendChild(act);
            }
            body.appendChild(row);
        }
    }
}

// ── Per-node checksum-drift badge (kept; tiny, in-context) ──────────
function installNodeBadge() {
    if (typeof LGraphCanvas === "undefined") return;
    if (LGraphCanvas.prototype.__MEC_INTEG_PATCHED__) return;
    LGraphCanvas.prototype.__MEC_INTEG_PATCHED__ = true;
    const origDraw = LGraphCanvas.prototype.drawNodeShape;
    LGraphCanvas.prototype.drawNodeShape = function (node /*, ctx, size, fg, bg, selected, mouseOver*/) {
        origDraw.apply(this, arguments);
        if (isMuted() || !STATE.events.length || !node?.type) return;
        const drifted = STATE.events.find(
            (e) => e.kind === "checksum_drift" && e.file &&
                   e.file.toLowerCase().includes(String(node.type).toLowerCase())
        );
        if (drifted) {
            const ctx = arguments[1];
            ctx.save();
            ctx.fillStyle = "#fab387";
            ctx.beginPath();
            ctx.arc(8, -8, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = "#11111b";
            ctx.font = "bold 9px monospace";
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            ctx.fillText("!", 8, -8);
            ctx.restore();
        }
    };
}

// ── Extension registration ──────────────────────────────────────────
app.registerExtension({
    name: "MEC.IntegrityStatus",
    setup() {
        injectStyle();
        installNodeBadge();
        api.addEventListener("nukenodemax.integrity", (ev) => ingest(ev.detail || ev));

        removeStaleFloater();
        startMountObserver();
        fetchReport();

        // Light periodic poll in case the socket misses an event.
        setInterval(fetchReport, 60_000);
    },
});
