// C2C settings registry, front end (L3.01 slice S3; backend nodes/_c2c_settings.py).
//
// Fetches GET /c2c/settings/schema at module load (top-level await, 3 s cap) and DECLARES the settings on
// registerExtension({ settings: [...] }) like every other C2C setting (tests/test_settings_declarative.py: Lite mode
// strips setup() of heavy extensions but keeps declared settings, so no imperative registration). Ids a pack already
// declares statically are skipped (first registration wins in the store anyway; this avoids the duplicate warning),
// so the registry only fills in settings that exist on the backend alone. Nothing here stores a value.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPE = { boolean: "boolean", number: "number", slider: "slider", combo: "combo", text: "text" };

function registered(id) {
    try {
        const s = app.extensionManager?.setting?.settings;
        if (s && typeof s === "object") return id in s;
    } catch (_e) { /* fall through */ }
    return false;
}

/** The core setting definition for one schema entry (pure; tests/test_l301_settings_registry.py runs it). */
export function toSettingDef(s) {
    const def = {
        id: s.id,
        name: s.name || s.id,
        type: TYPE[s.type] || "text",
        defaultValue: s.defaultValue,
        category: Array.isArray(s.category) && s.category.length ? s.category : ["C2C", ...String(s.id).split(".")],
        tooltip: [s.tooltip, s.env ? `The environment variable ${s.env} overrides this when it is set.` : "",
                  s.restart ? "Takes effect after a restart." : ""].filter(Boolean).join(" "),
    };
    if (Array.isArray(s.options)) def.options = s.options;
    if (s.attrs && typeof s.attrs === "object") def.attrs = { ...s.attrs };
    return def;
}

async function fetchSchema() {
    const ctl = typeof AbortController === "function" ? new AbortController() : null;
    const timer = setTimeout(() => ctl?.abort(), 3000);       // never hold up the page on a slow server
    try {
        const r = await api.fetchApi("/c2c/settings/schema", ctl ? { signal: ctl.signal } : {});
        return r.ok ? ((await r.json())?.settings || []) : [];
    } catch (_e) {
        return [];                                              // no backend registry: static declarations still work
    } finally {
        clearTimeout(timer);
    }
}

const _schema = await fetchSchema();

app.registerExtension({
    name: "C2C.SettingsRegistry",
    settings: _schema.filter((s) => s?.id && !registered(s.id)).map(toSettingDef),
});
