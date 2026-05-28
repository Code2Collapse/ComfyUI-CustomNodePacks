/**
 * _c2c_report.js — shared failure-reporting helper for all C2C / MEC JS modules.
 *
 * Per locked policy (2026-05-25, user mandate "Strict"):
 *   - Every catch must route through reportFailure().
 *   - No silent empty `catch (_) {}` blocks anywhere in C2C/MEC code.
 *
 * The reporter:
 *   - Logs to console (via console.error) so DevTools surfaces it.
 *   - Dispatches a window CustomEvent "c2c:registry-failure" with structured
 *     detail so the registry-status HUD and diagnostics sidebar can aggregate.
 *   - Best-effort POSTs to /c2c/registry/failure for the server-side audit log
 *     using fetch keepalive, so unload-time errors still make it.
 *
 * The implementation MUST itself be bullet-proof: it cannot throw, because
 * throwing inside an error handler would create an infinite loop or hide the
 * original error. Any internal failure is swallowed silently as a last resort
 * (with one console.error fallback).
 */

const _C2C_REPORT_ENDPOINT = "/c2c/registry/failure";

/**
 * Report a non-fatal failure from a C2C/MEC module.
 *
 * @param {string} where     Free-form scope label: "filename:functionName"
 *                           or "filename:callsite". Required.
 * @param {*}      err       The caught error/exception. Optional but
 *                           strongly recommended.
 * @param {string} [component] Optional component override; defaults to
 *                             "c2c" when not provided.
 */
export function reportFailure(where, err, component) {
    let detail;
    try {
        detail = {
            component: String(component || "c2c"),
            where: String(where || "(unknown)"),
            message: (err && err.message) ? String(err.message) : String(err),
            stack: (err && err.stack) ? String(err.stack) : null,
            name: (err && err.name) ? String(err.name) : null,
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

    // 1) Console — primary developer-facing channel.
    try {
        // eslint-disable-next-line no-console
        console.error(`[${detail.component}] ${detail.where}:`, err);
    } catch (consoleErr) {
        // If even console.error throws (e.g. console mocked away), do nothing.
        void consoleErr;
    }

    // 2) Window CustomEvent — picked up by c2c_registry_status.js and the
    //    diagnostics sidebar.
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

    // 3) Best-effort server POST. Keepalive lets it survive page unload.
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
