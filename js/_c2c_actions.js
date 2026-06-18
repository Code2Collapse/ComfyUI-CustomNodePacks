/**
 * _c2c_actions.js — global action registry consumed by the command palette
 * and any other surface that wants to list/run cross-feature actions.
 *
 * §12.5 of ideas.md + §13.5 of the audit gap matrix: every C2C feature
 * should register its top-level user actions through ONE public contract,
 * so they appear in Ctrl+K, in context menus, and (eventually) in the
 * omnibar without each panel reinventing discovery.
 *
 *   import { registerAction, unregisterAction, listActions, runAction,
 *            onActionsChanged } from "./_c2c_actions.js";
 *
 *   registerAction({
 *       id:       "c2c.ai.openErrorTranslator",
 *       title:    "Open AI Error Translator",
 *       titleKey: "actions.openErrorTranslator",   // resolved via i18n.t()
 *       hint:     "Paste a Python traceback and get a plain-English fix",
 *       hintKey:  "actions.openErrorTranslator.hint",
 *       kind:     "navigation",
 *       scope:    "global",
 *       icon:     "🩹",
 *       keywords: ["error", "traceback", "ai"],
 *       enabled:  () => true,
 *       run:      (ctx) => window.__C2C_AI_ERROR_TRANSLATOR__?.open?.(ctx),
 *   });
 *
 * Action shape (validated on register, extra keys preserved):
 *   id         string (required, unique). dotted scope.feature.verb.
 *   title      string fallback used when titleKey not set / missing
 *   titleKey   i18n key; resolved at list time via i18n.t(key, title)
 *   hint       string secondary line shown in palette
 *   hintKey    i18n key for hint
 *   kind       "command" | "navigation" | "toggle" | "generator"
 *              (purely informational; palette groups by it)
 *   scope      "global" | "graph" | "node" | "panel"  (palette filter aid)
 *   icon       single glyph
 *   keywords   string[] additional fuzzy-match seeds
 *   shortcut   display-only key combo string ("Ctrl+Shift+E")
 *   enabled    () => boolean — predicate, omit for always-on
 *   run        (ctx?) => void|Promise<void>  REQUIRED
 *
 * License: Apache-2.0
 */

import { i18n } from "./_c2c_i18n.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const _registry = new Map();   // id -> action
const _listeners = new Set();  // change subscribers

const _VALID_KINDS  = new Set(["command", "navigation", "toggle", "generator"]);
const _VALID_SCOPES = new Set(["global", "graph", "node", "panel"]);

function _validate(action) {
    if (!action || typeof action !== "object") {
        throw new TypeError("registerAction: action must be an object");
    }
    if (typeof action.id !== "string" || !action.id) {
        throw new TypeError("registerAction: action.id required (non-empty string)");
    }
    if (typeof action.run !== "function") {
        throw new TypeError(`registerAction(${action.id}): action.run must be a function`);
    }
    if (action.kind != null && !_VALID_KINDS.has(action.kind)) {
        throw new RangeError(`registerAction(${action.id}): invalid kind ${action.kind}`);
    }
    if (action.scope != null && !_VALID_SCOPES.has(action.scope)) {
        throw new RangeError(`registerAction(${action.id}): invalid scope ${action.scope}`);
    }
    if (action.keywords != null && !Array.isArray(action.keywords)) {
        throw new TypeError(`registerAction(${action.id}): keywords must be array`);
    }
    if (action.enabled != null && typeof action.enabled !== "function") {
        throw new TypeError(`registerAction(${action.id}): enabled must be a function`);
    }
}

function _emit() {
    for (const cb of _listeners) {
        try { cb(); } catch (e) { console.warn("[c2c-actions] listener threw", e); }
    }
}

/**
 * Register an action. Overwrites any prior registration with the same id
 * (so panels can hot-reload safely). Returns an unregister function.
 */
export function registerAction(action) {
    _validate(action);
    const frozen = Object.freeze({
        kind:  "command",
        scope: "global",
        ...action,
        keywords: Array.isArray(action.keywords) ? [...action.keywords] : [],
    });
    _registry.set(frozen.id, frozen);
    _emit();
    return () => unregisterAction(frozen.id);
}

/** Register many actions in one shot (atomic emit). */
export function registerActions(actions) {
    if (!Array.isArray(actions)) {
        throw new TypeError("registerActions: expected an array");
    }
    const handles = [];
    // Validate all up front so partial registration can't happen.
    for (const a of actions) _validate(a);
    for (const a of actions) {
        const frozen = Object.freeze({
            kind:  "command",
            scope: "global",
            ...a,
            keywords: Array.isArray(a.keywords) ? [...a.keywords] : [],
        });
        _registry.set(frozen.id, frozen);
        handles.push(frozen.id);
    }
    _emit();
    return () => { for (const id of handles) _registry.delete(id); _emit(); };
}

export function unregisterAction(id) {
    if (typeof id !== "string" || !id) return false;
    const removed = _registry.delete(id);
    if (removed) _emit();
    return removed;
}

/**
 * Return registered actions, with i18n-resolved title/hint, optionally
 * filtered by scope and/or `enabled()` predicate.
 *
 * @param {{scope?: string, includeDisabled?: boolean}} [opts]
 * @returns {Array<{
 *   id: string, title: string, hint: string, kind: string, scope: string,
 *   icon: string, keywords: string[], shortcut: string, enabled: boolean,
 *   run: Function,
 * }>}
 */
export function listActions(opts = {}) {
    const wantScope = opts.scope || null;
    const includeDisabled = !!opts.includeDisabled;
    const out = [];
    for (const a of _registry.values()) {
        if (wantScope && a.scope !== wantScope) continue;
        let enabled = true;
        if (typeof a.enabled === "function") {
            try { enabled = !!a.enabled(); }
            catch (e) { console.warn(`[c2c-actions] ${a.id} enabled() threw`, e); enabled = false; }
        }
        if (!enabled && !includeDisabled) continue;
        const title = a.titleKey ? i18n.t(a.titleKey, a.title || a.id) : (a.title || a.id);
        const hint  = a.hintKey  ? i18n.t(a.hintKey,  a.hint  || "")    : (a.hint  || "");
        out.push({
            id: a.id,
            title,
            hint,
            kind:     a.kind || "command",
            scope:    a.scope || "global",
            icon:     a.icon || "",
            keywords: a.keywords || [],
            shortcut: a.shortcut || "",
            enabled,
            run:      a.run,
        });
    }
    return out;
}

/** Look up a single action by id. Returns the resolved form or null. */
export function getAction(id) {
    const a = _registry.get(id);
    if (!a) return null;
    return listActions({ includeDisabled: true }).find((x) => x.id === id) || null;
}

/**
 * Execute an action by id. Returns the action's return value (which may
 * be a Promise). Throws if the action is missing; rejects/throws upstream
 * if the action itself throws.
 */
export function runAction(id, ctx) {
    const a = _registry.get(id);
    if (!a) throw new Error(`runAction: unknown action id "${id}"`);
    if (typeof a.enabled === "function") {
        try {
            if (!a.enabled()) throw new Error(`runAction: action "${id}" is disabled`);
        } catch (e) {
            // If enabled() throws, treat as disabled for safety
            throw new Error(`runAction: action "${id}" is disabled (${e.message})`);
        }
    }
    return a.run(ctx);
}

/**
 * Subscribe to registry changes (register / unregister). Returns an
 * unsubscribe function.
 */
export function onActionsChanged(cb) {
    if (typeof cb !== "function") return () => {};
    _listeners.add(cb);
    return () => _listeners.delete(cb);
}

// Aggregate for namespace-style imports.
export const actions = Object.freeze({
    register:     registerAction,
    registerMany: registerActions,
    unregister:   unregisterAction,
    list:         listActions,
    get:          getAction,
    run:          runAction,
    onChange:     onActionsChanged,
});

// Make it accessible from devtools / cross-extension code.
try { window.__C2C_ACTIONS__ = actions; } catch (__c2cErr) { __c2cReport("_c2c_actions", __c2cErr); }
