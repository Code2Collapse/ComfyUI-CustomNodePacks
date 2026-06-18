/**
 * _c2c_prompts.js — client-side fetcher for the versioned server-rendered
 * prompt library at /c2c/ai/prompts.
 *
 * All AI features in C2C now resolve their system prompts through this
 * helper instead of hard-coding them inline. The server holds the .j2
 * templates + frozen golden hashes; the JS asks "give me the rendered
 * text for template X with these vars".
 *
 * API:
 *   await getPrompt(name, vars?)                 → string (cached)
 *   await listPrompts({ refresh? })              → array of metadata records
 *   await verifyPrompts()                        → array of {name, ok, ...}
 *   clearCache()                                 → wipe in-memory cache
 *
 * Cache: keyed by `${name}\u0000${JSON.stringify(vars||{})}`. Persists
 * for the lifetime of the page. The server returns version+sha256 so
 * upstream features can pin against drift if they ever need to.
 */

const _cache = new Map();            // key → { text, version, golden_sha256 }
let _listCache = null;               // last list() response

const _BASE = "/c2c/ai/prompts";

async function _json(url, init) {
    const r = await fetch(url, init);
    let body;
    try { body = await r.json(); }
    catch (e) { throw new Error(`prompt fetch: invalid JSON (${url}): ${e.message}`); }
    if (!body || body.success !== true) {
        const msg = body?.message || body?.error || `HTTP ${r.status}`;
        throw new Error(`prompt fetch failed (${url}): ${msg}`);
    }
    return body.data;
}

export async function getPrompt(name, vars) {
    if (typeof name !== "string" || !name) {
        throw new Error("getPrompt: name (string) required");
    }
    const v = (vars && typeof vars === "object") ? vars : {};
    const key = `${name}\u0000${JSON.stringify(v)}`;
    const cached = _cache.get(key);
    if (cached) return cached.text;

    const data = await _json(`${_BASE}/render`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name, vars: v }),
    });
    _cache.set(key, { text: data.text, version: data.version, golden_sha256: data.golden_sha256 });
    return data.text;
}

export async function getPromptMeta(name, vars) {
    // Same key shape as getPrompt — but returns the full envelope.
    if (typeof name !== "string" || !name) throw new Error("getPromptMeta: name required");
    const v = (vars && typeof vars === "object") ? vars : {};
    const key = `${name}\u0000${JSON.stringify(v)}`;
    const cached = _cache.get(key);
    if (cached) return { text: cached.text, version: cached.version, golden_sha256: cached.golden_sha256, name };
    await getPrompt(name, v);
    const after = _cache.get(key);
    return { ...after, name };
}

export async function listPrompts({ refresh } = {}) {
    if (!refresh && _listCache) return _listCache;
    const data = await _json(`${_BASE}`);
    _listCache = data.templates || [];
    return _listCache;
}

export async function verifyPrompts() {
    const data = await _json(`${_BASE}/verify`);
    return data.results || [];
}

export function clearCache() {
    _cache.clear();
    _listCache = null;
}

// Side-effect: expose on window for power users + smoke tests.
if (typeof window !== "undefined") {
    window.__C2C_PROMPTS__ = { getPrompt, getPromptMeta, listPrompts, verifyPrompts, clearCache };
}
