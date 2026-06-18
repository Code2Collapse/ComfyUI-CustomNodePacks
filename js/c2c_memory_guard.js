/**
 * c2c_memory_guard.js — keep the ComfyUI tab from dying of a JS-heap OOM.
 *
 * Why this exists:
 *   The browser renderer has a hard per-tab V8 heap ceiling (~2 GB in
 *   Chromium/Brave). A long session with our many extensions — growing caches,
 *   preview blobs, repaint loops — plus a big graph can allocate faster than GC
 *   reclaims, ending in "Aw, Snap! (Error code 5)" / the journal line
 *   "V8 javascript OOM (MarkCompactCollector: young object promotion failed)".
 *   This guard makes that the pack's own responsibility: it watches the heap and
 *   proactively frees what WE retain before V8 hits the wall, so no user of these
 *   nodes inherits the crash.
 *
 * What it does (all guarded; no-ops where an API is missing):
 *   - Central registry every C2C module can opt into:
 *       window.__C2C_MEM__.register(name, clearFn)   // a cache it can flush
 *       window.__C2C_MEM__.trackBlob(url)            // an object URL to recycle
 *   - Polls performance.memory (Chromium/Brave). Paused while the tab is hidden.
 *       • ≥ warn%  : flush registered caches, set window.__C2C_LOW_MEM__ = true
 *                    (cheap signal any repaint/poll loop can read to back off),
 *                    one throttled toast.
 *       • ≥ 90%    : also revoke all tracked blob URLs + request GC; loud toast.
 *       • < 60%    : clear the low-memory flag (recovered).
 *   - Caps tracked blob URLs at 64 (FIFO-revokes the oldest) so preview/AB/image
 *     streams can never pile up object URLs.
 *   - Command "C2C: Free memory now" + a setting to disable.
 *
 * It is itself minimal: ONE 8 s interval, skipped when the tab is hidden, and a
 * visibilitychange listener. No per-frame work.
 *
 * License: Apache-2.0
 */
import { app } from "../../scripts/app.js";

const NS = "C2C.MemoryGuard";
const POLL_MS = 8000;
const S = { enabled: "c2c.memGuard.enabled", warnPct: "c2c.memGuard.warnPct" };

// Public, shared registry — created here, consumed pack-wide.
const MEM = (window.__C2C_MEM__ = window.__C2C_MEM__ || {
    caches: new Set(),   // { name, clear: () => void }
    blobs: new Set(),    // tracked object URLs (insertion-ordered → FIFO recycle)
    low: false,
});
MEM.register = (name, clear) => {
    if (typeof clear === "function") MEM.caches.add({ name: String(name || "?"), clear });
};
MEM.trackBlob = (url) => {
    if (!url) return url;
    MEM.blobs.add(url);
    if (MEM.blobs.size > 64) {
        const oldest = MEM.blobs.values().next().value;
        try { URL.revokeObjectURL(oldest); } catch (_) { /* ignore */ }
        MEM.blobs.delete(oldest);
    }
    return url;
};

let _timer = 0;
let _warnedAt = 0;

function _setting(id, def) {
    try { const v = app?.ui?.settings?.getSettingValue?.(id); return (v === undefined || v === null) ? def : v; }
    catch { return def; }
}

function _heap() {
    const m = performance && performance.memory;
    if (!m || !m.jsHeapSizeLimit) return null;            // non-Chromium → no signal
    return { used: m.usedJSHeapSize, limit: m.jsHeapSizeLimit, pct: m.usedJSHeapSize / m.jsHeapSizeLimit };
}

function _clearCaches() {
    let n = 0;
    for (const c of MEM.caches) { try { c.clear(); n++; } catch (_) { /* keep going */ } }
    return n;
}
function _revokeBlobs() {
    let n = 0;
    for (const u of MEM.blobs) { try { URL.revokeObjectURL(u); n++; } catch (_) { /* ignore */ } }
    MEM.blobs.clear();
    return n;
}
function _toast(detail, severity = "warn") {
    try { app.extensionManager?.toast?.add?.({ severity, summary: "C2C Memory Guard", detail, life: 5000 }); }
    catch { console.warn("[C2C.MemoryGuard]", detail); }
}
function _setLow(v) {
    MEM.low = !!v;
    window.__C2C_LOW_MEM__ = !!v;   // loops elsewhere can read this and back off
}

function _tick() {
    if (document.hidden) return;
    if (!_setting(S.enabled, true)) return;
    const h = _heap();
    if (!h) return;
    const warn = (Number(_setting(S.warnPct, 75)) || 75) / 100;
    if (h.pct >= 0.90) {
        _setLow(true);
        const c = _clearCaches();
        const b = _revokeBlobs();
        if (window.gc) { try { window.gc(); } catch (_) { /* not exposed */ } }
        _toast(`Heap at ${Math.round(h.pct * 100)}% — freed ${c} cache(s) + ${b} preview blob(s). Reload the tab if this repeats.`, "error");
    } else if (h.pct >= warn) {
        _setLow(true);
        _clearCaches();
        const now = Date.now();
        if (now - _warnedAt > 60000) { _warnedAt = now; _toast(`Heap at ${Math.round(h.pct * 100)}% — trimmed C2C caches to stay under the limit.`); }
    } else if (h.pct < 0.60) {
        _setLow(false);
    }
}

function _freeNow() {
    const c = _clearCaches();
    const b = _revokeBlobs();
    if (window.gc) { try { window.gc(); } catch (_) { /* ignore */ } }
    _toast(`Freed ${c} cache(s) + ${b} preview blob(s).`, "info");
}

app.registerExtension({
    name: NS,
    settings: [
        { id: S.enabled, name: "C2C ▸ Memory Guard ▸ Enabled", type: "boolean", defaultValue: true,
          tooltip: "Auto-frees C2C caches/preview blobs before the browser tab runs out of JS heap (prevents 'Aw, Snap' OOM crashes)." },
        { id: S.warnPct, name: "C2C ▸ Memory Guard ▸ Trim at heap %", type: "slider",
          attrs: { min: 50, max: 90, step: 5 }, defaultValue: 75,
          tooltip: "Heap usage at which to start flushing C2C caches." },
    ],
    commands: [
        { id: "c2c.memGuard.freeNow", label: "C2C: Free memory now", function: _freeNow },
    ],
    async setup() {
        if (_timer) clearInterval(_timer);
        _timer = setInterval(() => { try { _tick(); } catch (_) { /* never throw from the guard */ } }, POLL_MS);
        document.addEventListener("visibilitychange", () => { if (!document.hidden) { try { _tick(); } catch (_) {} } });
        const h = _heap();
        console.log(`[C2C.MemoryGuard] armed (poll ${POLL_MS}ms; heap API ${h ? "available" : "unavailable"}).`);
    },
});
