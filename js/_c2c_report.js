/**
 * _c2c_report.js — shared failure-reporting helper for all C2C / MEC JS modules.
 *
 * Per locked policy (2026-05-25, user mandate "Strict"):
 *   - Every catch must route through reportFailure().
 *   - No silent empty `catch (_) {}` blocks anywhere in C2C/MEC code.
 *
 * The reporter:
 *   - Logs to console (level-aware: info|warn|error) so DevTools surfaces it.
 *   - Dispatches a window CustomEvent "c2c:registry-failure" with structured
 *     detail so the registry-status HUD and diagnostics sidebar can aggregate.
 *   - Best-effort POSTs to /c2c/registry/failure for the server-side audit log
 *     using fetch keepalive, so unload-time errors still make it.
 *
 * Severity model (2026-05-30, Track A.5):
 *   - level: "error"  (default) — full pipeline: console.error + dispatch + POST.
 *           The registry status HUD will surface this as a toast.
 *   - level: "warn"            — console.warn + dispatch only; NO server POST.
 *           Used for recoverable issues (one failed retry of N).
 *   - level: "info"            — console.info only; NO dispatch, NO POST.
 *           Used for optional-feature absences (missing optional route, etc.)
 *           where the user must NOT see a red toast.
 *
 * Rate limits (Slice A): per dedup key (where|message) — console first 3 then
 * every 100th; window event max 1/10s; POST max 1/5min; global POST cap 20/min.
 *
 * The implementation MUST itself be bullet-proof: it cannot throw, because
 * throwing inside an error handler would create an infinite loop or hide the
 * original error. Any internal failure is swallowed silently as a last resort
 * (with one console.error fallback).
 */

const _C2C_REPORT_ENDPOINT = "/c2c/registry/failure";
const _VALID_LEVELS = new Set(["error", "warn", "info"]);
const _EVENT_INTERVAL_MS = 10_000;
const _POST_INTERVAL_MS = 5 * 60_000;
const _GLOBAL_POST_CAP = 20;
const _GLOBAL_POST_WINDOW_MS = 60_000;

const _perKey = new Map();
const _globalPosts = [];

function _dedupKey(where, message) {
    return `${where}|${message}`;
}

const _MAX_KEYS = 500;

function _keyState(key) {
    let s = _perKey.get(key);
    if (!s) {
        // messages that embed ids or times make endless distinct keys: keep
        // the newest 500 (Map iterates oldest-first)
        if (_perKey.size >= _MAX_KEYS) _perKey.delete(_perKey.keys().next().value);
        s = { count: 0, lastEvent: 0, lastPost: 0 };
        _perKey.set(key, s);
    }
    return s;
}

function _globalPostAllowed(now) {
    while (_globalPosts.length && _globalPosts[0] < now - _GLOBAL_POST_WINDOW_MS) {
        _globalPosts.shift();
    }
    return _globalPosts.length < _GLOBAL_POST_CAP;
}

/**
 * Report a non-fatal failure from a C2C/MEC module.
 *
 * @param {string} where     Free-form scope label: "filename:functionName"
 *                           or "filename:callsite". Required.
 * @param {*}      err       The caught error/exception. Optional but
 *                           strongly recommended.
 * @param {string|object} [componentOrOpts]
 *                           Either a component-name string (legacy 3-arg
 *                           positional form) OR an options object:
 *                             { component?: string, level?: "error"|"warn"|"info" }
 *                           Default level is "error" (back-compat).
 */
export function reportFailure(where, err, componentOrOpts) {
    // Normalise the 3rd arg into {component, level}. Back-compat: a bare
    // string is still treated as component name with level="error".
    let component = "c2c";
    let level = "error";
    if (typeof componentOrOpts === "string") {
        component = componentOrOpts;
    } else if (componentOrOpts && typeof componentOrOpts === "object") {
        if (componentOrOpts.component) component = String(componentOrOpts.component);
        if (componentOrOpts.level && _VALID_LEVELS.has(componentOrOpts.level)) {
            level = componentOrOpts.level;
        }
    }
    let detail;
    try {
        detail = {
            component: String(component || "c2c"),
            where: String(where || "(unknown)"),
            message: (err && err.message) ? String(err.message) : String(err),
            stack: (err && err.stack) ? String(err.stack) : null,
            name: (err && err.name) ? String(err.name) : null,
            level,
            ts: Date.now(),
        };
    } catch (buildErr) {
        // Building the detail object should never fail, but if it does we
        // still want SOMETHING in the console. Use a literal fallback string.
        try {
            // eslint-disable-next-line no-console
            console.error("[c2c-report] detail-build-failed", buildErr, where, err);
        } catch (innerConsoleErr) {
            void innerConsoleErr;
        }
        return;
    }

    const now = detail.ts;
    const dkey = _dedupKey(detail.where, detail.message);
    let ks;
    try {
        ks = _keyState(dkey);
        ks.count += 1;
    } catch (_) {
        ks = { count: 1, lastEvent: 0, lastPost: 0 };
    }

    // 1) Console — first 3 per key, then every 100th occurrence.
    try {
        const n = ks.count;
        const logFull = n <= 3 || (n % 100 === 0);
        if (logFull) {
            const prefix = n > 3
                ? `[${detail.component}] ${detail.where}: repeated ${n} times`
                : `[${detail.component}] ${detail.where}:`;
            // eslint-disable-next-line no-console
            if (level === "info") console.info(prefix, err);
            // eslint-disable-next-line no-console
            else if (level === "warn") console.warn(prefix, err);
            // eslint-disable-next-line no-console
            else console.error(prefix, err);
        }
    } catch (consoleErr) {
        void consoleErr;
    }

    // 2) Window CustomEvent — max once per key per 10 s.
    if (level !== "info") {
        try {
            if (now - ks.lastEvent >= _EVENT_INTERVAL_MS) {
                ks.lastEvent = now;
                if (typeof window !== "undefined" && typeof window.dispatchEvent === "function") {
                    window.dispatchEvent(new CustomEvent("c2c:registry-failure", { detail }));
                }
            }
        } catch (dispatchErr) {
            try {
                // eslint-disable-next-line no-console
                console.error("[c2c-report] dispatch-failed", dispatchErr);
            } catch (innerDispatchErr) {
                void innerDispatchErr;
            }
        }
    }

    // 3) Best-effort server POST — max once per key per 5 min; global 20/min.
    if (level !== "error") return;
    try {
        if (now - ks.lastPost < _POST_INTERVAL_MS) return;
        if (!_globalPostAllowed(now)) return;
        ks.lastPost = now;
        _globalPosts.push(now);
        if (typeof fetch === "function") {
            fetch(_C2C_REPORT_ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(detail),
                keepalive: true,
            }).catch((netErr) => {
                try {
                    // eslint-disable-next-line no-console
                    console.debug("[c2c-report] net-post-failed", netErr);
                } catch (innerNetErr) {
                    void innerNetErr;
                }
            });
        }
    } catch (fetchErr) {
        try {
            // eslint-disable-next-line no-console
            console.error("[c2c-report] fetch-init-failed", fetchErr);
        } catch (innerFetchErr) {
            void innerFetchErr;
        }
    }
}

// Convenience default export so callers can do either:
//   import { reportFailure } from "./_c2c_report.js";
//   import reportFailure from "./_c2c_report.js";
export default reportFailure;
