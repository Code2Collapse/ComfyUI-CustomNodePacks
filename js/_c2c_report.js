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
 * The implementation MUST itself be bullet-proof: it cannot throw, because
 * throwing inside an error handler would create an infinite loop or hide the
 * original error. Any internal failure is swallowed silently as a last resort
 * (with one console.error fallback).
 */

const _C2C_REPORT_ENDPOINT = "/c2c/registry/failure";
const _VALID_LEVELS = new Set(["error", "warn", "info"]);

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

    // 1) Console — primary developer-facing channel. Level-aware so we don't
    //    spam DevTools with red noise for optional-feature absences.
    try {
        const prefix = `[${detail.component}] ${detail.where}:`;
        // eslint-disable-next-line no-console
        if (level === "info") console.info(prefix, err);
        // eslint-disable-next-line no-console
        else if (level === "warn") console.warn(prefix, err);
        // eslint-disable-next-line no-console
        else console.error(prefix, err);
    } catch (consoleErr) {
        // If even console.error throws (e.g. console mocked away), do nothing.
        void consoleErr;
    }

    // 2) Window CustomEvent — picked up by c2c_registry_status.js and the
    //    diagnostics sidebar. Skipped for level="info" so optional-feature
    //    absences don't accumulate in the registry-failure log.
    if (level !== "info") {
        try {
            if (typeof window !== "undefined" && typeof window.dispatchEvent === "function") {
                window.dispatchEvent(new CustomEvent("c2c:registry-failure", { detail }));
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

    // 3) Best-effort server POST. Keepalive lets it survive page unload.
    //    Only fires for level="error" so optional-failure noise does not
    //    surface as a red toast via /c2c/registry/status.
    if (level !== "error") return;
    try {
        if (typeof fetch === "function") {
            fetch(_C2C_REPORT_ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(detail),
                keepalive: true,
            }).catch((netErr) => {
                // Net errors here are expected when the server endpoint is
                // not mounted (older builds). Log once, do not re-throw.
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
