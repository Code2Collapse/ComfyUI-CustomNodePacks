/**
 * _c2c_storage.js — namespaced persistence wrapper around localStorage.
 *
 * §0.7 of ideas.md: every C2C panel must persist user state under a single
 * `c2c.` key prefix using a typed JSON envelope so we can version-migrate
 * without touching consumer code. This module is that contract.
 *
 * API:
 *   import { storage } from "./_c2c_storage.js";
 *   storage.get("foo")            // → any | null   (auto JSON.parse)
 *   storage.set("foo", { a: 1 })  // → boolean      (false on quota fail)
 *   storage.remove("foo")
 *   storage.keys()                // → string[]     (just the C2C-prefixed keys, prefix stripped)
 *   storage.clear()               // → number       (count removed)
 *
 *   // namespaced sub-scopes for per-panel isolation:
 *   const s = storage.scope("ai.explainer");
 *   s.set("geom", { x, y, w, h });
 *   s.get("geom");
 *
 *   // optional TTL (ms) — value transparently expires:
 *   storage.set("cached", obj, { ttl: 60_000 });
 *
 * Resilience:
 *   - Private-mode browsers where localStorage throws on access:
 *     wrapper degrades to an in-memory Map, panels keep working in-session.
 *   - QuotaExceededError: set() returns false; never throws.
 *   - Corrupt JSON: get() returns null, key is auto-pruned.
 *
 * License: Apache-2.0
 */
import { reportFailure as __c2cReport } from "./_c2c_report.js";


const PREFIX = "c2c.";

// ── Backing store ─────────────────────────────────────────────────────────
// In some environments (Safari private mode pre-2022, sandboxed iframes,
// over-quota), `window.localStorage` either throws on access or all writes
// throw. Probe once at module-load and fall back to an in-memory Map.
const _mem = new Map();
let _backend = (() => {
    try {
        const k = "__c2c_probe__";
        window.localStorage.setItem(k, "1");
        window.localStorage.removeItem(k);
        return window.localStorage;
    } catch (_) {
        console.warn("[c2c-storage] localStorage unavailable; using in-memory fallback");
        return {
            getItem:    (k) => (_mem.has(k) ? _mem.get(k) : null),
            setItem:    (k, v) => { _mem.set(k, String(v)); },
            removeItem: (k) => { _mem.delete(k); },
            key:        (i) => Array.from(_mem.keys())[i] ?? null,
            get length() { return _mem.size; },
        };
    }
})();

// ── JSON envelope ─────────────────────────────────────────────────────────
// Every stored value is wrapped as `{ v: <payload>, e?: <expiresAtMs> }`.
// Bare legacy strings (e.g. "1"/"0") are still readable for backward compat:
// they round-trip through JSON.parse and we expose them unchanged.
function _wrap(payload, ttlMs) {
    const env = { v: payload };
    if (Number.isFinite(ttlMs) && ttlMs > 0) env.e = Date.now() + ttlMs;
    return JSON.stringify(env);
}

function _unwrap(raw, fullKey) {
    if (raw === null || raw === undefined) return null;
    let parsed;
    try { parsed = JSON.parse(raw); }
    catch (_) {
        // Not a JSON envelope — return the raw string so legacy
        // `localStorage.setItem(k, "1")` callers keep working.
        return raw;
    }
    if (parsed && typeof parsed === "object" && Object.prototype.hasOwnProperty.call(parsed, "v")) {
        if (typeof parsed.e === "number" && Date.now() > parsed.e) {
            try { _backend.removeItem(fullKey); } catch (__c2cErr) { __c2cReport("_c2c_storage", __c2cErr); }
            return null;
        }
        return parsed.v;
    }
    // Foreign JSON value (someone wrote raw JSON, no envelope) — return as-is.
    return parsed;
}

// ── Core API ──────────────────────────────────────────────────────────────
function _fullKey(key) {
    if (typeof key !== "string" || key.length === 0) throw new TypeError("storage key must be a non-empty string");
    return key.startsWith(PREFIX) ? key : PREFIX + key;
}

export function get(key) {
    const fk = _fullKey(key);
    let raw;
    try { raw = _backend.getItem(fk); }
    catch (_) { return null; }
    try { return _unwrap(raw, fk); }
    catch (_) {
        try { _backend.removeItem(fk); } catch (__c2cErr) { __c2cReport("_c2c_storage", __c2cErr); }
        return null;
    }
}

export function set(key, value, { ttl } = {}) {
    const fk = _fullKey(key);
    try {
        _backend.setItem(fk, _wrap(value, ttl));
        return true;
    } catch (err) {
        // QuotaExceeded or SecurityError — caller decides what to do.
        return false;
    }
}

export function remove(key) {
    const fk = _fullKey(key);
    try { _backend.removeItem(fk); return true; }
    catch (_) { return false; }
}

export function has(key) {
    const fk = _fullKey(key);
    try { return _backend.getItem(fk) !== null; }
    catch (_) { return false; }
}

/** List all C2C-prefixed keys (prefix stripped). */
export function keys() {
    const out = [];
    try {
        const n = _backend.length;
        for (let i = 0; i < n; i++) {
            const k = _backend.key(i);
            if (typeof k === "string" && k.startsWith(PREFIX)) out.push(k.slice(PREFIX.length));
        }
    } catch (__c2cErr) { __c2cReport("_c2c_storage", __c2cErr); }
    return out;
}

/** Remove every C2C-prefixed key. Returns the count removed. */
export function clear() {
    const found = keys();
    let n = 0;
    for (const k of found) {
        if (remove(k)) n++;
    }
    return n;
}

/**
 * Sub-scope. Every call on the returned object is auto-prefixed with
 * `<scope>.`, so panels never collide:
 *   const s = scope("ai.explainer");
 *   s.set("geom", {...})  → stored under  c2c.ai.explainer.geom
 */
export function scope(ns) {
    if (typeof ns !== "string" || ns.length === 0) throw new TypeError("scope name must be a non-empty string");
    const pre = ns.endsWith(".") ? ns : ns + ".";
    return Object.freeze({
        get:    (k)            => get(pre + k),
        set:    (k, v, opts)   => set(pre + k, v, opts),
        remove: (k)            => remove(pre + k),
        has:    (k)            => has(pre + k),
        keys:   ()             => keys().filter(k => k.startsWith(pre)).map(k => k.slice(pre.length)),
        clear:  ()             => {
            let n = 0;
            for (const k of keys()) if (k.startsWith(pre) && remove(k)) n++;
            return n;
        },
        scope:  (sub)          => scope(pre + sub),
    });
}

/** Whether the backend is real localStorage (false ⇒ in-memory fallback). */
export function isPersistent() {
    return _backend === window.localStorage;
}

// Aggregate export so consumers can do `import { storage } from "./_c2c_storage.js"`.
export const storage = Object.freeze({ get, set, remove, has, keys, clear, scope, isPersistent });
