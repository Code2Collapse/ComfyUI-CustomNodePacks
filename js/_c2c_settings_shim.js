/**
 * _c2c_settings_shim.js — restore the historic
 *   `settings.getSettingValue(id, defaultValue)`
 * contract that newer ComfyUI front-ends silently dropped (they now
 * return `undefined` when the user hasn't touched the setting, even if
 * the registration declared a default). The deprecation warning
 *     "Parameter defaultValue is deprecated. The default value in
 *      settings definition will be used instead."
 * appears in console for every call.
 *
 * That change broke ~60 call sites across the c2c / mec extension fleet
 * that pass a sane fallback as the 2nd argument and expect it to be
 * returned. Patching every call would be invasive and easy to regress;
 * instead we wrap the store ONCE here and let the fallback take effect
 * whenever the registry-side default isn't surfaced.
 *
 * Idempotent: marks the wrapped function with `__c2c_default_shim__`.
 */
import { app } from "../../scripts/app.js";

function installShim() {
    const s = app?.ui?.settings;
    if (!s || typeof s.getSettingValue !== "function") return false;
    if (s.getSettingValue.__c2c_default_shim__) return true;
    const orig = s.getSettingValue.bind(s);
    function shimmed(id, def) {
        let v;
        try { v = orig(id); } catch { v = undefined; }
        if (v !== undefined && v !== null) return v;
        return def;
    }
    shimmed.__c2c_default_shim__ = true;
    s.getSettingValue = shimmed;
    return true;
}

app.registerExtension({
    name: "c2c.settings.shim",
    // `init` runs before any other extension's `setup`, giving downstream
    // c2c/mec code the patched function for its enable-checks.
    async init() {
        if (installShim()) return;
        // settings store may not be ready yet — retry briefly.
        for (let i = 0; i < 20; i++) {
            await new Promise(r => setTimeout(r, 50));
            if (installShim()) return;
        }
    },
    async setup() { installShim(); },
});
