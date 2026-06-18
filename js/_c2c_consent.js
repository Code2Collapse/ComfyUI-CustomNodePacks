/**
 * _c2c_consent.js — explicit user consent gate for any network/AI action.
 *
 * §0.4 of ideas.md: nothing leaves the user's machine without an explicit
 * tap-through that shows WHAT, WHERE, and at WHAT COST. Granted scopes are
 * stored locally (per-machine) via the storage helper so users aren't
 * pestered every keystroke.
 *
 *   import { consent } from "./_c2c_consent.js";
 *
 *   if (!consent.has("ai.cloud.openai")) {
 *       const ok = await consent.request("ai.cloud.openai", {
 *           title:    t("consent.title.cloud_ai", "Send this to OpenAI?"),
 *           provider: "OpenAI (api.openai.com)",
 *           preview:  redactedPromptText,
 *           costUsd:  0.012,
 *           bytes:    redactedPromptText.length,
 *       });
 *       if (!ok) return;
 *   }
 *
 * Persistence:
 *   - Grants live under storage scope "consent" via _c2c_storage.
 *   - Duration choices: once | session | day | forever.
 *   - "once" never persists; "session" lives in window.sessionStorage; the
 *     rest go in localStorage via the storage helper (TTL when applicable).
 *   - revoke(scope) drops the grant. clearAll() wipes every grant.
 *
 * License: Apache-2.0
 */

import { storage } from "./_c2c_storage.js";
// NOTE: `C` (palette snapshot) is intentionally NOT imported. Every color
// below must resolve through live `var(--c2c-X)` CSS custom properties so
// the modal repaints when setVariant() flips the theme. Fallbacks must also
// be variant-aware (chain to another --c2c-X) — never a raw literal.
import { T, z, reducedMotion } from "./_c2c_theme.js";
import { t } from "./_c2c_i18n.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

// ── Grant store ───────────────────────────────────────────────────────────
const SCOPE = storage.scope("consent");
const SESSION_KEY_PREFIX = "c2c.consent.session.";

function _now() { return Date.now(); }

function _readSessionGrant(scope) {
    try {
        const raw = window.sessionStorage.getItem(SESSION_KEY_PREFIX + scope);
        if (!raw) return null;
        return JSON.parse(raw);
    } catch (_) { return null; }
}

function _writeSessionGrant(scope, grant) {
    try { window.sessionStorage.setItem(SESSION_KEY_PREFIX + scope, JSON.stringify(grant)); }
    catch (__c2cErr) { __c2cReport("_c2c_consent", __c2cErr); }
}

function _readGrant(scope) {
    // Persistent first (forever / day), session fallback.
    const persistent = SCOPE.get(scope);
    if (persistent && typeof persistent === "object") return persistent;
    return _readSessionGrant(scope);
}

/** True if the scope has a non-expired grant. */
export function has(scope) {
    if (typeof scope !== "string" || !scope) return false;
    const g = _readGrant(scope);
    if (!g) return false;
    if (typeof g.expiresAt === "number" && _now() > g.expiresAt) {
        revoke(scope);
        return false;
    }
    return true;
}

export function revoke(scope) {
    if (typeof scope !== "string" || !scope) return false;
    SCOPE.remove(scope);
    try { window.sessionStorage.removeItem(SESSION_KEY_PREFIX + scope); } catch (__c2cErr) { __c2cReport("_c2c_consent", __c2cErr); }
    return true;
}

export function list() {
    const out = [];
    for (const k of SCOPE.keys()) {
        const g = SCOPE.get(k);
        if (g && typeof g === "object") out.push({ scope: k, ...g });
    }
    try {
        for (let i = 0; i < window.sessionStorage.length; i++) {
            const k = window.sessionStorage.key(i);
            if (k && k.startsWith(SESSION_KEY_PREFIX)) {
                const g = JSON.parse(window.sessionStorage.getItem(k) || "null");
                if (g) out.push({ scope: k.slice(SESSION_KEY_PREFIX.length), ...g, duration: "session" });
            }
        }
    } catch (__c2cErr) { __c2cReport("_c2c_consent", __c2cErr); }
    return out;
}

export function clearAll() {
    let n = 0;
    for (const k of SCOPE.keys()) { SCOPE.remove(k); n++; }
    try {
        const keys = [];
        for (let i = 0; i < window.sessionStorage.length; i++) {
            const k = window.sessionStorage.key(i);
            if (k && k.startsWith(SESSION_KEY_PREFIX)) keys.push(k);
        }
        for (const k of keys) { window.sessionStorage.removeItem(k); n++; }
    } catch (__c2cErr) { __c2cReport("_c2c_consent", __c2cErr); }
    return n;
}

// ── Modal UI ──────────────────────────────────────────────────────────────
const STYLE_ID = "c2c-consent-style";
function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.c2c-consent-overlay {
    position: fixed; inset: 0; z-index: var(--c2c-z-modal, ${z.modal});
    background: var(--c2c-overlay-scrim, color-mix(in srgb, var(--c2c-shadowBase) 55%, transparent));
    display: flex; align-items: center; justify-content: center;
    animation: c2c-consent-fadein var(--c2c-dur-fast, 120ms) var(--c2c-ease-out);
}
@keyframes c2c-consent-fadein { from { opacity: 0; } to { opacity: 1; } }
.c2c-consent-card {
    background: var(--c2c-bg); color: var(--c2c-fg);
    border: 1px solid var(--c2c-border);
    border-radius: var(--c2c-radius-lg, ${T.radius.lg});
    box-shadow: var(--c2c-shadow-lg, ${T.shadow.lg});
    width: min(560px, 90vw); max-height: 80vh; overflow: hidden;
    display: flex; flex-direction: column;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 13px;
}
.c2c-consent-header {
    padding: ${T.pad.lg} ${T.pad.xl}; border-bottom: 1px solid var(--c2c-border);
    font-weight: 600; font-size: 14px;
}
.c2c-consent-body {
    padding: ${T.pad.lg} ${T.pad.xl}; overflow-y: auto; flex: 1;
    display: flex; flex-direction: column; gap: ${T.gap.md};
}
.c2c-consent-row {
    display: flex; gap: ${T.gap.md}; align-items: baseline;
}
.c2c-consent-row > .k {
    color: var(--c2c-sub); min-width: 86px; font-size: 12px;
}
.c2c-consent-row > .v { color: var(--c2c-fg); flex: 1; word-break: break-word; }
.c2c-consent-preview {
    background: var(--c2c-bg2); border: 1px solid var(--c2c-border);
    border-radius: var(--c2c-radius-sm, ${T.radius.sm});
    padding: ${T.pad.md}; font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 11px; white-space: pre-wrap; max-height: 220px; overflow: auto;
    color: var(--c2c-fg);
}
.c2c-consent-redacted-note {
    color: var(--c2c-status-info, var(--c2c-blue)); font-size: 11px;
}
.c2c-consent-footer {
    padding: ${T.pad.md} ${T.pad.xl}; border-top: 1px solid var(--c2c-border);
    display: flex; align-items: center; gap: ${T.gap.md}; flex-wrap: wrap;
}
.c2c-consent-footer .spacer { flex: 1; }
.c2c-consent-btn {
    padding: 6px 14px; border-radius: var(--c2c-radius-sm, ${T.radius.sm});
    border: 1px solid var(--c2c-border);
    background: var(--c2c-bg2); color: var(--c2c-fg);
    cursor: pointer; font-size: 13px; font-family: inherit;
    transition: background var(--c2c-dur-fast, 120ms) var(--c2c-ease-out);
}
.c2c-consent-btn:hover { background: var(--c2c-overlay-hover, color-mix(in srgb, var(--c2c-highlightBase) 6%, transparent)); }
.c2c-consent-btn:focus-visible { outline: none; box-shadow: var(--c2c-focus-ring, 0 0 0 2px var(--c2c-blue)); }
.c2c-consent-btn.primary {
    background: var(--c2c-blue); color: var(--c2c-bg); border-color: transparent;
}
.c2c-consent-btn.primary:hover { filter: brightness(1.1); }
.c2c-consent-btn.danger { color: var(--c2c-status-danger, var(--c2c-red)); }
.c2c-consent-duration {
    display: flex; gap: ${T.gap.xs}; align-items: center;
    color: var(--c2c-sub); font-size: 12px;
}
.c2c-consent-duration select {
    background: var(--c2c-bg2); color: var(--c2c-fg);
    border: 1px solid var(--c2c-border);
    border-radius: var(--c2c-radius-sm, ${T.radius.sm});
    padding: 3px 6px; font: inherit;
}
`;
    document.head.appendChild(s);
}

function _formatBytes(n) {
    if (typeof n !== "number" || !Number.isFinite(n)) return "—";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

function _formatUsd(n) {
    if (typeof n !== "number" || !Number.isFinite(n) || n <= 0) return "free";
    if (n < 0.01) return `~$${n.toFixed(4)}`;
    return `~$${n.toFixed(3)}`;
}

const DURATIONS = [
    { value: "once",    label: () => t("consent.duration.once",    "Just this once") },
    { value: "session", label: () => t("consent.duration.session", "For this session") },
    { value: "day",     label: () => t("consent.duration.day",     "For 24 hours") },
    { value: "forever", label: () => t("consent.duration.forever", "Always (until I revoke)") },
];

function _ttlMs(duration) {
    if (duration === "day") return 24 * 60 * 60 * 1000;
    return null; // forever: no expiry; once/session don't use this
}

/**
 * Show the consent modal and return a Promise resolving to true if the user
 * approves, false if they cancel or close. A granted scope is persisted
 * according to the duration the user picked.
 *
 * Opts:
 *   title       — modal heading (string)
 *   provider    — human-readable destination ("OpenAI (api.openai.com)")
 *   preview     — string shown verbatim in a code box (already redacted by caller)
 *   bytes       — payload size for the "Size" row
 *   costUsd     — estimated cost in USD
 *   feature     — short feature id ("error_translator") shown for context
 *   defaultDuration — "once" | "session" | "day" | "forever" (default "session")
 *   denyLabel   — override the cancel button text
 *   allowLabel  — override the primary button text
 *
 * Returns Promise<boolean>.
 */
export function request(scope, opts = {}) {
    if (typeof scope !== "string" || !scope) {
        return Promise.reject(new TypeError("consent.request: scope must be a non-empty string"));
    }
    if (has(scope)) return Promise.resolve(true);

    _injectStyle();

    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.className = "c2c-consent-overlay";
        overlay.setAttribute("role", "dialog");
        overlay.setAttribute("aria-modal", "true");

        const card = document.createElement("div");
        card.className = "c2c-consent-card";
        overlay.appendChild(card);

        const header = document.createElement("div");
        header.className = "c2c-consent-header";
        header.textContent = opts.title || t("consent.title.default", "Allow this network request?");
        card.appendChild(header);

        const body = document.createElement("div");
        body.className = "c2c-consent-body";
        card.appendChild(body);

        const rows = [
            ["consent.row.feature",  t("consent.row.feature",  "Feature"),  opts.feature  || scope],
            ["consent.row.provider", t("consent.row.provider", "Provider"), opts.provider || "(unspecified)"],
            ["consent.row.size",     t("consent.row.size",     "Size"),     _formatBytes(opts.bytes)],
            ["consent.row.cost",     t("consent.row.cost",     "Cost est."), _formatUsd(opts.costUsd)],
        ];
        for (const [, k, v] of rows) {
            const row = document.createElement("div");
            row.className = "c2c-consent-row";
            const ke = document.createElement("div"); ke.className = "k"; ke.textContent = k;
            const ve = document.createElement("div"); ve.className = "v"; ve.textContent = String(v);
            row.appendChild(ke); row.appendChild(ve);
            body.appendChild(row);
        }

        if (opts.preview) {
            const note = document.createElement("div");
            note.className = "c2c-consent-redacted-note";
            note.textContent = t(
                "consent.preview.note",
                "Below is exactly what will be sent (already redacted):"
            );
            body.appendChild(note);
            const pre = document.createElement("div");
            pre.className = "c2c-consent-preview";
            pre.textContent = String(opts.preview);
            body.appendChild(pre);
        }

        const footer = document.createElement("div");
        footer.className = "c2c-consent-footer";
        card.appendChild(footer);

        const dur = document.createElement("label");
        dur.className = "c2c-consent-duration";
        dur.appendChild(document.createTextNode(t("consent.duration.label", "Remember:")));
        const sel = document.createElement("select");
        for (const d of DURATIONS) {
            const o = document.createElement("option");
            o.value = d.value; o.textContent = d.label();
            sel.appendChild(o);
        }
        sel.value = ["once","session","day","forever"].includes(opts.defaultDuration)
            ? opts.defaultDuration : "session";
        dur.appendChild(sel);
        footer.appendChild(dur);

        const spacer = document.createElement("div");
        spacer.className = "spacer";
        footer.appendChild(spacer);

        const deny = document.createElement("button");
        deny.className = "c2c-consent-btn";
        deny.textContent = opts.denyLabel || t("action.cancel", "Cancel");
        footer.appendChild(deny);

        const allow = document.createElement("button");
        allow.className = "c2c-consent-btn primary";
        allow.textContent = opts.allowLabel || t("consent.action.send", "Send");
        footer.appendChild(allow);

        let settled = false;
        const close = (result) => {
            if (settled) return;
            settled = true;
            try { overlay.remove(); } catch (__c2cErr) { __c2cReport("_c2c_consent", __c2cErr); }
            document.removeEventListener("keydown", onKey, true);
            resolve(result);
        };
        const onKey = (e) => {
            if (e.key === "Escape") { e.preventDefault(); close(false); }
            if (e.key === "Enter" && document.activeElement === allow) { /* default */ }
        };

        deny.addEventListener("click", () => close(false));
        overlay.addEventListener("click", (e) => { if (e.target === overlay) close(false); });
        document.addEventListener("keydown", onKey, true);

        allow.addEventListener("click", () => {
            const duration = sel.value;
            const grant = {
                grantedAt: _now(),
                duration,
                provider: opts.provider || null,
                feature:  opts.feature  || null,
            };
            if (duration === "session") {
                _writeSessionGrant(scope, grant);
            } else if (duration === "day") {
                grant.expiresAt = _now() + _ttlMs("day");
                SCOPE.set(scope, grant, { ttl: _ttlMs("day") });
            } else if (duration === "forever") {
                SCOPE.set(scope, grant);
            }
            // "once": don't persist anything; caller still gets true.
            close(true);
        });

        document.body.appendChild(overlay);
        // Default focus on the primary action so Enter approves.
        setTimeout(() => { try { allow.focus({ preventScroll: true }); } catch (__c2cErr) { __c2cReport("_c2c_consent", __c2cErr); } },
                   reducedMotion() ? 0 : 60);
    });
}

export const consent = Object.freeze({ has, request, revoke, list, clearAll });
