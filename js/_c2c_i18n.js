/**
 * _c2c_i18n.js — minimal i18n runtime for every C2C panel.
 *
 * §0.5 of ideas.md: panels must be translatable without each one rolling
 * its own string table. This module exposes:
 *
 *   import { t, setLocale, getLocale, onLocaleChange, addBundle } from "./_c2c_i18n.js";
 *
 *   t("panel.title.explainer")                       // "AI Node Explainer"
 *   t("status.loading", "Loading…")                  // fallback if missing
 *   t("greeting.hello", "Hello, {name}", {name: "X"})
 *   t("items", "{n} item|{n} items", { n: 3 })       // pipe = singular|plural
 *
 * Resolution order for a key:
 *   1. active locale bundle (e.g. `es`)
 *   2. English bundle (`en`)
 *   3. fallback string passed as 2nd arg
 *   4. the key itself
 *
 * Variable interpolation uses `{name}` placeholders. A `{n}` value of 1
 * (or absolute value 1) selects the first pipe-arm, anything else the
 * second. Anything more complex should resolve to two distinct keys; this
 * is a deliberately small surface.
 *
 * Late-bound bundles: panels can ship their own strings via
 *   addBundle("en", { "myPanel.foo": "Foo" });
 *
 * Locale source order:
 *   1. ComfyUI setting `c2c.i18n.locale` if set to a concrete code.
 *   2. Stored c2c.i18n.locale (storage helper) if "auto" or unset.
 *   3. `navigator.language` two-letter code.
 *   4. "en".
 *
 * License: Apache-2.0
 */

import { app } from "../../scripts/app.js";
import { storage } from "./_c2c_storage.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

// ── Built-in English bundle (seed; panels add more via addBundle) ────────
const EN = {
    // Generic actions
    "action.ok":            "OK",
    "action.cancel":        "Cancel",
    "action.close":         "Close",
    "action.copy":          "Copy",
    "action.save":          "Save",
    "action.delete":        "Delete",
    "action.retry":         "Retry",
    "action.regenerate":    "Regenerate",
    "action.pin":           "Pin",
    "action.unpin":         "Unpin",

    // Generic status
    "status.idle":          "Idle",
    "status.loading":       "Loading…",
    "status.success":       "Done",
    "status.error":         "Error",
    "status.empty":         "Nothing to show",
    "status.streaming":     "Streaming…",

    // Common labels
    "label.search":         "Search",
    "label.tokens":         "tokens",
    "label.cost":           "cost",
    "label.elapsed":        "elapsed",

    // Errors
    "error.generic":        "Something went wrong",
    "error.network":        "Network error — check your connection",
    "error.no_api_key":     "No API key configured — open Settings → C2C AI",
    "error.quota":          "API quota exceeded",

    // Panels
    "panel.title.explainer":     "AI Node Explainer",
    "panel.title.errors":        "AI Error Translator",
    "panel.title.doctor":        "Workflow Doctor",
    "panel.title.palette":       "Command Palette",
    "panel.title.find":          "Find in Workflow",

    // Confirmations
    "confirm.discard_changes":   "Discard unsaved changes?",
    "confirm.delete_n_nodes":    "Delete {n} node|Delete {n} nodes",
};

const BUNDLES = { en: { ...EN } };

// ── Locale state ──────────────────────────────────────────────────────────
function _normaliseCode(code) {
    if (typeof code !== "string" || !code) return null;
    // Accept "en", "en-US", "en_GB"; emit the primary two-letter subtag.
    const m = /^([a-z]{2,3})(?:[-_]|$)/i.exec(code);
    return m ? m[1].toLowerCase() : null;
}

function _detectLocale() {
    // 1. explicit setting if not "auto"
    try {
        const s = app?.ui?.settings?.getSettingValue?.("c2c.i18n.locale");
        if (s && s !== "auto") {
            const n = _normaliseCode(s);
            if (n) return n;
        }
    } catch (__c2cErr) { __c2cReport("_c2c_i18n", __c2cErr); }
    // 2. persisted choice via storage
    try {
        const v = storage.get("i18n.locale");
        if (typeof v === "string" && v !== "auto") {
            const n = _normaliseCode(v);
            if (n) return n;
        }
    } catch (__c2cErr) { __c2cReport("_c2c_i18n", __c2cErr); }
    // 3. browser hint
    const nav = _normaliseCode(navigator?.language || "");
    if (nav) return nav;
    // 4. fallback
    return "en";
}

let _locale = _detectLocale();

export function getLocale() { return _locale; }

export function setLocale(code) {
    const n = _normaliseCode(code) || "en";
    if (n === _locale) return false;
    _locale = n;
    try { storage.set("i18n.locale", n); } catch (__c2cErr) { __c2cReport("_c2c_i18n", __c2cErr); }
    try { window.dispatchEvent(new CustomEvent("c2c:locale-changed", { detail: { locale: n } })); } catch (__c2cErr) { __c2cReport("_c2c_i18n", __c2cErr); }
    return true;
}

export function listLocales() { return Object.keys(BUNDLES); }

/**
 * Merge entries into a locale bundle. Creates the bundle if missing.
 * Existing keys are overwritten.
 */
export function addBundle(locale, entries) {
    const n = _normaliseCode(locale);
    if (!n || !entries || typeof entries !== "object") return false;
    BUNDLES[n] = Object.assign(BUNDLES[n] || {}, entries);
    return true;
}

// ── Subscription helper ───────────────────────────────────────────────────
export function onLocaleChange(cb) {
    if (typeof cb !== "function") return () => {};
    const handler = (ev) => {
        try { cb(ev?.detail || { locale: _locale }); }
        catch (err) { console.warn("[c2c-i18n] onLocaleChange handler threw", err); }
    };
    window.addEventListener("c2c:locale-changed", handler);
    return () => window.removeEventListener("c2c:locale-changed", handler);
}

// ── Interpolation + pluralisation ─────────────────────────────────────────
function _interpolate(str, vars) {
    if (!vars || typeof vars !== "object") return str;
    return str.replace(/\{([a-zA-Z0-9_]+)\}/g, (_m, k) =>
        Object.prototype.hasOwnProperty.call(vars, k) ? String(vars[k]) : `{${k}}`
    );
}

function _pluralise(str, vars) {
    if (!str.includes("|")) return str;
    const n = vars && Object.prototype.hasOwnProperty.call(vars, "n") ? Number(vars.n) : NaN;
    if (!Number.isFinite(n)) return str.split("|")[0];
    const arms = str.split("|");
    const idx = Math.abs(n) === 1 ? 0 : 1;
    return arms[idx] !== undefined ? arms[idx] : arms[0];
}

/**
 * Translate `key` for the active locale.
 *
 * @param {string}             key       dotted path, e.g. "panel.title.explainer"
 * @param {string=}            fallback  emitted if the key is missing in every bundle
 * @param {Record<string,any>=} vars     interpolation map; `n` drives plural selection
 * @returns {string}
 */
export function t(key, fallback, vars) {
    if (typeof key !== "string" || !key) return fallback ?? "";
    const active = BUNDLES[_locale] || null;
    const en = BUNDLES.en;
    const raw =
        (active && Object.prototype.hasOwnProperty.call(active, key) && active[key]) ||
        (Object.prototype.hasOwnProperty.call(en, key) && en[key]) ||
        fallback ||
        key;
    return _interpolate(_pluralise(String(raw), vars), vars);
}

// Aggregate export so consumers can do `import { i18n } from "./_c2c_i18n.js"`.
export const i18n = Object.freeze({
    t, getLocale, setLocale, listLocales, addBundle, onLocaleChange,
});

// ── ComfyUI setting registration ──────────────────────────────────────────
try {
    app.registerExtension({
        name: "C2C.I18n",
        settings: [
            {
                id: "c2c.i18n.locale",
                name: "C2C → Language",
                tooltip: "Language used for C2C panels and labels. 'auto' follows the browser.",
                type: "combo",
                options: [
                    { value: "auto", text: "Auto (browser)" },
                    { value: "en",   text: "English" },
                    { value: "es",   text: "Español" },
                    { value: "fr",   text: "Français" },
                    { value: "de",   text: "Deutsch" },
                    { value: "pt",   text: "Português" },
                    { value: "zh",   text: "中文" },
                    { value: "ja",   text: "日本語" },
                    { value: "ko",   text: "한국어" },
                    { value: "hi",   text: "हिन्दी" },
                    { value: "ar",   text: "العربية" },
                    { value: "ru",   text: "Русский" },
                ],
                defaultValue: "auto",
                onChange: (v) => {
                    if (v === "auto") {
                        // Re-run detection (browser → en fallback) and broadcast.
                        const old = _locale;
                        _locale = _detectLocale();
                        try { storage.remove("i18n.locale"); } catch (__c2cErr) { __c2cReport("_c2c_i18n", __c2cErr); }
                        if (old !== _locale) {
                            try { window.dispatchEvent(new CustomEvent("c2c:locale-changed", { detail: { locale: _locale } })); } catch (__c2cErr) { __c2cReport("_c2c_i18n", __c2cErr); }
                        }
                    } else {
                        setLocale(v);
                    }
                },
            },
        ],
    });
} catch (__c2cErr) { __c2cReport("_c2c_i18n", __c2cErr); }
