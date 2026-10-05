// _c2c_runtime.js — process-wide C2C frontend runtime singleton.
// ---------------------------------------------------------------------------
// One instance per page on globalThis.__c2cRuntime (create if absent, else
// reuse). Identical copies may ship in other packs — never import across packs.
//
// Provides: coalesced canvas redraw, generation run-state, guarded hooks,
// safe core patches, adaptive scheduling, graph-change callbacks, perf tier.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { reportFailure } from "./_c2c_report.js";

const LS_MODE = "c2c.perf.mode";
const LS_GPU = "c2c.perf.gpu";
const LS_LITE_LEGACY = "c2c.lite";
const MODE_AUTO = "Auto (recommended)";
const MODE_FULL = "Full";
const MODE_LITE = "Lite";
const SOFTWARE_RE = /swiftshader|llvmpipe|softpipe|lavapipe|software|basic render|mesa offscreen/i;
const REDRAW_CAP_PER_SEC = 15;
const REDRAW_CAP_WINDOW_MS = 1000;
const _OUR_ANIM_RE = /^(c2c|mec)-/;

function _readGpuCache() {
    try {
        const raw = localStorage.getItem(LS_GPU);
        if (!raw) return null;
        const o = JSON.parse(raw);
        if (o && typeof o === "object") return o;
    } catch (_) { /* keep null */ }
    return null;
}

function _resolveTierFromMode(mode, gpuCache) {
    if (mode === MODE_LITE) return "lite";
    if (mode === MODE_FULL) return "full";
    // Auto (recommended) or unknown -> the cached GPU probe decides. No probe
    // yet (first load on this browser) means FULL: a fresh install must look
    // and behave normally; the idle-time probe corrects it for the next load.
    if (gpuCache && gpuCache.software === true) return "lite";
    return "full";
}

function _readPerfMode() {
    try {
        const v = localStorage.getItem(LS_MODE);
        if (v === MODE_AUTO || v === MODE_FULL || v === MODE_LITE) return v;
        if (!v && localStorage.getItem(LS_LITE_LEGACY) === "1") return MODE_LITE;
    } catch (_) { /* fall through */ }
    return MODE_AUTO;
}

const GPU_PROBE_MAX_AGE_MS = 7 * 24 * 3600 * 1000;

/** The cached probe when it is under a week old and was taken by this browser build (UA): a GPU or
 *  driver change shows up as a new UA or an expired entry, and only then is a WebGL context created. */
export function freshGpuProbe() {
    const c = _readGpuCache();
    if (!c || typeof c.ts !== "number" || Date.now() - c.ts > GPU_PROBE_MAX_AGE_MS) return null;
    if (c.ua !== String(navigator?.userAgent || "")) return null;
    return c;
}

/** Live WebGL probe — call only from requestIdleCallback, never at module eval. */
export function probeGpuSoftware() {
    let software = true;
    let renderer = "";
    try {
        const canvas = document.createElement("canvas");
        canvas.__c2cProbe = true;              // the browser guard leaves this context alone
        const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
        if (!gl) {
            software = true;
        } else {
            const dbg = gl.getExtension("WEBGL_debug_renderer_info");
            renderer = String(
                dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)
            );
            software = SOFTWARE_RE.test(renderer);
            try {
                const lose = gl.getExtension("WEBGL_lose_context");
                canvas.__c2cIntentionalLoss = true;
                if (lose) lose.loseContext();
            } catch (_) { /* best effort */ }
        }
        canvas.width = 0;
        canvas.height = 0;
    } catch (_) {
        software = true;
    }
    const entry = { software, renderer, ua: String(navigator?.userAgent || ""), ts: Date.now() };
    try { localStorage.setItem(LS_GPU, JSON.stringify(entry)); } catch (_) {}
    return entry;
}

function _effectiveTier(mode, gpuCache) {
    return _resolveTierFromMode(mode, gpuCache);
}

function _createRuntime() {
    const _patchRegistry = new Map();
    let _running = false;
    let _queueRemaining = 0;
    const _runListeners = new Set();

    // ── redraw coalescing ──────────────────────────────────────────────────
    let _rafPending = false;
    let _wantFg = false;
    let _wantBg = false;
    let _redrawTimes = [];

    function _notifyRunState() {
        for (const cb of _runListeners) {
            try { cb(_running); } catch (e) { reportFailure("_c2c_runtime:runState", e); }
        }
        _syncDocClasses();
    }

    // ── quiet controller (CSS animation pause while running / hidden / lite) ─
    const _pausedAnims = new Set();
    const _quietListeners = new Set();
    let _quietActive = false;
    let _animStartListener = null;

    function _animationsSupported() {
        try {
            return typeof document !== "undefined"
                && typeof document.getAnimations === "function"
                && typeof CSSAnimation !== "undefined";
        } catch (_) {
            return false;
        }
    }

    function _pauseOurAnim(anim) {
        if (!_animationsSupported()) return;
        try {
            if (!(anim instanceof CSSAnimation)) return;
            const name = anim.animationName;
            if (!name || !_OUR_ANIM_RE.test(name)) return;
            if (anim.playState === "paused") return;
            anim.pause();
            _pausedAnims.add(anim);
        } catch (_) { /* old browser / detached animation */ }
    }

    function _pauseAllOurAnimations() {
        if (!_animationsSupported()) return;
        try {
            for (const anim of document.getAnimations()) _pauseOurAnim(anim);
        } catch (_) { /* inert on old browsers */ }
    }

    function _resumeOurAnimations() {
        if (!_animationsSupported()) return;
        for (const anim of _pausedAnims) {
            try {
                if (anim.playState === "paused") anim.play();
            } catch (_) { /* detached */ }
        }
        _pausedAnims.clear();
    }

    function _onAnimationStart(ev) {
        if (!_quietActive || !ev?.animation) return;
        _pauseOurAnim(ev.animation);
    }

    function _notifyQuiet(quiet) {
        for (const cb of _quietListeners) {
            try { cb(quiet); } catch (e) { reportFailure("_c2c_runtime:onQuiet", e); }
        }
    }

    function _updateQuietState(quiet) {
        const next = !!quiet;
        if (next === _quietActive) return;
        _quietActive = next;
        if (_animationsSupported()) {
            try {
                if (next) {
                    _pauseAllOurAnimations();
                    if (!_animStartListener) {
                        _animStartListener = (ev) => _onAnimationStart(ev);
                        document.addEventListener("animationstart", _animStartListener, true);
                    }
                } else {
                    // resume FIRST and on its own: if removing the listener
                    // ever failed, our animations must not stay frozen
                    try { _resumeOurAnimations(); } catch (_) { /* detached */ }
                    if (_animStartListener) {
                        const l = _animStartListener;
                        _animStartListener = null;
                        document.removeEventListener("animationstart", l, true);
                    }
                }
            } catch (_) { /* inert on old browsers */ }
        }
        _notifyQuiet(next);
    }

    function onQuiet(cb) {
        if (typeof cb !== "function") return () => {};
        _quietListeners.add(cb);
        try { cb(_quietActive); } catch (e) { reportFailure("_c2c_runtime:onQuiet", e); }
        return () => { _quietListeners.delete(cb); };
    }

    function _syncDocClasses() {
        try {
            const root = document.documentElement;
            if (!root) return;
            root.classList.toggle("c2c-running", _running);
            root.classList.toggle("c2c-lite", tier() === "lite");
            const quiet = _running || document.hidden || tier() === "lite";
            root.classList.toggle("c2c-quiet", quiet);
            _updateQuietState(quiet);
        } catch (_) { /* headless */ }
    }

    function _flushRedraw() {
        _rafPending = false;
        const bg = _wantBg;
        _wantFg = false;
        _wantBg = false;
        try {
            const canvas = app?.canvas;
            if (!canvas || typeof canvas.setDirty !== "function") return;
            canvas.setDirty(true, bg);
        } catch (e) {
            reportFailure("_c2c_runtime:requestRedraw", e);
        }
    }

    function _redrawAllowedNow() {
        if (!_running) return true;
        const now = Date.now();
        _redrawTimes = _redrawTimes.filter((t) => now - t < REDRAW_CAP_WINDOW_MS);
        return _redrawTimes.length < REDRAW_CAP_PER_SEC;
    }

    let _retryPending = false;

    function _scheduleRedraw() {
        if (_rafPending) return;
        if (!_redrawAllowedNow()) {
            // ONE pending retry however many callers ask while capped: a retry
            // per request piled up 125 timers a second (measured).
            if (_retryPending) return;
            _retryPending = true;
            try {
                setTimeout(() => { _retryPending = false; _scheduleRedraw(); },
                           Math.ceil(REDRAW_CAP_WINDOW_MS / REDRAW_CAP_PER_SEC));
            } catch (_) { _retryPending = false; }
            return;
        }
        _rafPending = true;
        try {
            requestAnimationFrame(() => {
                _rafPending = false;
                if (!_redrawAllowedNow()) {
                    _scheduleRedraw();
                    return;
                }
                if (_running) _redrawTimes.push(Date.now());
                _flushRedraw();
            });
        } catch (_) {
            _rafPending = false;
        }
    }

    function requestRedraw({ bg = false } = {}) {
        _wantFg = true;
        if (bg) _wantBg = true;
        _scheduleRedraw();
    }

    // ── run state ──────────────────────────────────────────────────────────
    function isRunning() { return _running; }

    function onRunState(cb) {
        if (typeof cb !== "function") return () => {};
        _runListeners.add(cb);
        return () => { _runListeners.delete(cb); };
    }

    function _setRunning(v) {
        const next = !!v;
        if (_running === next) return;
        _running = next;
        _notifyRunState();
    }

    function _wireRunState() {
        try {
            api.addEventListener("execution_start", () => _setRunning(true));
            api.addEventListener("execution_success", () => _setRunning(false));
            api.addEventListener("execution_error", () => _setRunning(false));
            api.addEventListener("execution_interrupted", () => _setRunning(false));
            api.addEventListener("executing", (ev) => {
                const d = ev?.detail ?? ev;
                const node = (d && typeof d === "object" && "node" in d) ? d.node : d;
                if (node == null && _queueRemaining === 0) _setRunning(false);
            });
            api.addEventListener("status", (ev) => {
                const q = ev?.detail?.exec_info?.queue_remaining;
                if (typeof q === "number") {
                    _queueRemaining = q;
                    if (q === 0 && _running) _setRunning(false);
                }
            });
        } catch (e) {
            reportFailure("_c2c_runtime:wireRunState", e);
        }
        try {
            document.addEventListener("visibilitychange", () => _syncDocClasses());
        } catch (_) { /* headless */ }
    }

    // ── guard / safePatch ──────────────────────────────────────────────────
    function guard(fn, where, { maxErrors = 5, windowMs = 10000 } = {}) {
        const errors = [];
        let tripped = false;
        return function guarded(...args) {
            if (tripped) return undefined;
            try {
                return fn.apply(this, args);
            } catch (err) {
                const now = Date.now();
                errors.push(now);
                while (errors.length && errors[0] < now - windowMs) errors.shift();
                if (errors.length >= maxErrors) {
                    tripped = true;
                    reportFailure(where, new Error("disabled after repeated errors"));
                    return undefined;
                }
                reportFailure(where, err);
                return undefined;
            }
        };
    }

    function patches() { return _patchRegistry; }

    function safePatch(obj, name, makeWrapper, { id, maxErrors = 5, windowMs = 10000 } = {}) {
        if (!id) return () => {};
        if (_patchRegistry.has(id)) return _patchRegistry.get(id).uninstall;
        if (!obj || typeof obj[name] !== "function") return () => {};
        const orig = obj[name];
        // Per-call record of whether OUR wrapper already reached the original.
        // If the wrapper throws after calling it (the usual "call core, then
        // draw our extra" shape), core already ran: return its result instead
        // of running core's method a second time.
        const calls = [];
        const origTracked = function (...a) {
            const rec = calls[calls.length - 1];
            const r = orig.apply(this, a);
            if (rec) { rec.called = true; rec.result = r; }
            return r;
        };
        const wrapped = makeWrapper(origTracked);
        const errors = [];
        let uninstall = () => {};
        const safe = function (...args) {
            const rec = { called: false, result: undefined };
            calls.push(rec);
            try {
                return wrapped.apply(this, args);
            } catch (err) {
                const now = Date.now();
                errors.push(now);
                while (errors.length && errors[0] < now - windowMs) errors.shift();
                if (errors.length >= maxErrors) {
                    reportFailure(`safePatch:${id}`, new Error("patch removed after repeated errors; core behaviour restored"));
                    uninstall();
                } else {
                    reportFailure(`safePatch:${id}`, err);
                }
                return rec.called ? rec.result : orig.apply(this, args);
            } finally {
                calls.pop();
            }
        };
        obj[name] = safe;
        uninstall = () => {
            if (obj[name] === safe) obj[name] = orig;
            _patchRegistry.delete(id);
        };
        _patchRegistry.set(id, { id, obj, name, orig, wrapper: safe, uninstall });
        return uninstall;
    }

    // ── every (lazy — no timer until first call) ───────────────────────────
    function every(id, ms, fn, { ambient = true, maxMs = ms * 8 } = {}) {
        let period = ms;
        let timer = null;
        let cancelled = false;
        const durations = [];
        // ONE guard for the task's lifetime: its error window must persist
        // across ticks, or the breaker never trips.
        const run = guard(fn, `every:${id}`);

        function _shouldPause() {
            if (document.hidden) return true;
            if (ambient && (_running || tier() === "lite")) return true;
            return false;
        }

        function _schedule() {
            if (cancelled) return;
            timer = setTimeout(_tick, period);
        }

        function _tick() {
            if (cancelled) return;
            if (_shouldPause()) {
                _schedule();
                return;
            }
            const t0 = (typeof performance !== "undefined") ? performance.now() : Date.now();
            run();
            const elapsed = ((typeof performance !== "undefined") ? performance.now() : Date.now()) - t0;
            durations.push(elapsed);
            if (durations.length > 3) durations.shift();
            if (durations.length === 3 && durations.every((d) => d > period * 0.1)) {
                period = Math.min(maxMs, period * 2);
                durations.length = 0;
            }
            _schedule();
        }

        // First timer starts here — nothing runs at module eval.
        _schedule();

        return {
            cancel() {
                cancelled = true;
                if (timer != null) {
                    clearTimeout(timer);
                    timer = null;
                }
            },
        };
    }

    // ── onGraphChange ──────────────────────────────────────────────────────
    const _graphCbs = new Set();
    let _graphTimer = null;
    let _graphDebounce = 250;

    function _deliverGraphChange() {
        _graphTimer = null;
        const run = () => {
            for (const cb of _graphCbs) {
                try { cb(); } catch (e) { reportFailure("_c2c_runtime:onGraphChange", e); }
            }
        };
        if (typeof requestIdleCallback === "function") {
            requestIdleCallback(run, { timeout: 1000 });
        } else {
            setTimeout(run, 0);
        }
    }

    function _scheduleGraphChange(debounceMs) {
        if (_graphTimer != null) clearTimeout(_graphTimer);
        _graphTimer = setTimeout(_deliverGraphChange, debounceMs);
    }

    function onGraphChange(cb, { debounceMs = 250 } = {}) {
        if (typeof cb !== "function") return () => {};
        _graphCbs.add(cb);
        _graphDebounce = Math.max(_graphDebounce, Number(debounceMs) || 0);
        return () => { _graphCbs.delete(cb); };
    }

    function _wireGraphChange() {
        try {
            api.addEventListener("graphChanged", () => _scheduleGraphChange(_graphDebounce));
        } catch (e) {
            reportFailure("_c2c_runtime:graphChanged", e);
        }
    }

    // ── tier (sync at eval — GPU cache only, no live probe) ────────────────
    let _cachedMode = _readPerfMode();
    let _cachedGpu = _readGpuCache();

    function tier() {
        return _effectiveTier(_cachedMode, _cachedGpu);
    }

    function _refreshTierCache() {
        _cachedMode = _readPerfMode();
        _cachedGpu = _readGpuCache();
    }

    /** After live GPU probe: returns effective tier; does not mutate eval-time tier. */
    function tierAfterProbe(gpuEntry) {
        return _resolveTierFromMode(_readPerfMode(), gpuEntry);
    }

    function readPerfMode() { return _readPerfMode(); }

    // Our <html> classes must survive whatever else writes that attribute
    // during boot (measured: set at eval, gone by the time the app is up) or
    // after a future core update. Re-apply ONLY on a mismatch, so our own
    // writes cannot loop.
    function _classesMismatch() {
        try {
            const c = document.documentElement.classList;
            const lite = tier() === "lite";
            const quiet = _running || document.hidden || lite;
            return c.contains("c2c-running") !== _running || c.contains("c2c-lite") !== lite
                || c.contains("c2c-quiet") !== quiet;
        } catch (_) { return false; }
    }
    try {
        if (typeof MutationObserver === "function" && document.documentElement) {
            new MutationObserver(() => { if (_classesMismatch()) _syncDocClasses(); })
                .observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
        }
    } catch (_) { /* headless */ }

    _wireRunState();
    _wireGraphChange();
    _syncDocClasses();

    // Minimal extension hook for afterConfigureGraph (workflow load).
    try {
        if (!(app.extensions || []).some((e) => e?.name === "C2C.RuntimeHooks")) {
            app.registerExtension({
                name: "C2C.RuntimeHooks",
                setup() { _syncDocClasses(); },
                afterConfigureGraph() { _scheduleGraphChange(_graphDebounce); },
            });
        }
    } catch (e) {
        reportFailure("_c2c_runtime:registerHooks", e);
    }

    return {
        runState: { isRunning, onRunState },
        requestRedraw,
        guard,
        safePatch,
        patches,
        every,
        onGraphChange,
        tier,
        tierAfterProbe,
        readPerfMode,
        refreshTierCache: _refreshTierCache,
        probeGpuSoftware,
        onQuiet,
    };
}

export function getRuntime() {
    if (!globalThis.__c2cRuntime) {
        globalThis.__c2cRuntime = _createRuntime();
    }
    return globalThis.__c2cRuntime;
}

getRuntime();
