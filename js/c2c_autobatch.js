/**
 * c2c_autobatch.js — automatic frame batching: Off / Internal / Universal (owner D0.14, A9).
 *
 *   Internal (default) — C2C's own image / mask nodes, plus the measured core list, split long batches into
 *                        memory-sized chunks automatically when a call would not fit.
 *   Universal          — any author's image / mask nodes, each proven frame-independent first (experimental).
 *   Off                — never split.
 * A saved "curated" (the old core-only mode) shows as Internal, which includes it.
 */
import { app } from "../../scripts/app.js";
import { legacyNodeMenu } from "./_c2c_compat.js";

const EXT_NAME = "c2c.autobatch";
const SETTING_MODE = "c2c.autobatch.mode";
const ROUTE_CONFIG = "/c2c/autobatch/config";

const MODE_OFF = "off";
const MODE_INTERNAL = "internal";
const MODE_CURATED = "curated";      // legacy: the measured core list only; Internal includes it
const MODE_UNIVERSAL = "universal";

const MODE_LABELS = {
    [MODE_OFF]: "Off",
    [MODE_INTERNAL]: "Internal",
    [MODE_UNIVERSAL]: "Universal",
};
const MODE_HINTS = {
    [MODE_OFF]: "Never split a batch.",
    [MODE_INTERNAL]: "C2C's own image / mask nodes (and the measured core list) split long batches automatically.",
    [MODE_UNIVERSAL]: "Any author's image / mask nodes, each proven frame-independent first (experimental).",
};

/** Settings-panel control: three buttons (owner D0.14: "3 buttons would be better"). */
function renderModeButtons(_name, setter, value) {
    const shown = value === MODE_CURATED ? MODE_INTERNAL : (value || MODE_INTERNAL);
    const wrap = document.createElement("div");
    wrap.setAttribute("role", "radiogroup");
    wrap.setAttribute("aria-label", "Image batching mode");
    wrap.style.cssText = "display:inline-flex;gap:0;border:1px solid var(--p-content-border-color, #555);"
        + "border-radius:6px;overflow:hidden;";
    for (const mode of [MODE_OFF, MODE_INTERNAL, MODE_UNIVERSAL]) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = MODE_LABELS[mode];
        b.title = MODE_HINTS[mode];
        b.setAttribute("role", "radio");
        b.setAttribute("aria-checked", String(mode === shown));
        const on = mode === shown;
        b.style.cssText = "padding:4px 12px;border:0;cursor:pointer;font:inherit;"
            + (on ? "background:var(--p-primary-color, #4a7dff);color:var(--p-primary-contrast-color, #fff);"
                  : "background:transparent;color:inherit;");
        b.addEventListener("click", () => {
            if (mode === shown) return;
            setter(mode);
            _setMode(mode);
        });
        wrap.appendChild(b);
    }
    return wrap;
}

let _syncingFromServer = false;

async function apiGet(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} returned ${r.status}`);
    return r.json();
}

async function apiPost(url, body) {
    const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!r.ok) {
        const text = await r.text();
        throw new Error(`${url} returned ${r.status}: ${text.slice(0, 200)}`);
    }
    return r.json();
}

function _nodeClass(node) {
    return node?.comfyClass || node?.type || "";
}

function _hasFrameInput(node) {
    return Array.isArray(node?.inputs) && node.inputs.some(
        (i) => i && (i.type === "IMAGE" || i.type === "MASK"),
    );
}

async function _loadConfig() {
    return apiGet(ROUTE_CONFIG);
}

async function _saveConfigPatch(patch) {
    const current = await _loadConfig();
    const merged = { ...current, ...patch };
    if (patch.allow || patch.never) {
        merged.allow = Array.isArray(patch.allow) ? patch.allow : (current.allow || []);
        merged.never = Array.isArray(patch.never) ? patch.never : (current.never || []);
    }
    return apiPost(ROUTE_CONFIG, merged);
}

async function _setMode(mode) {
    if (_syncingFromServer) return;
    try {
        await _saveConfigPatch({ mode });
    } catch (exc) {
        console.warn("[c2c.autobatch] could not save mode:", exc);
    }
}

async function _setNodeOverride(classId, choice) {
    if (!classId) return;
    try {
        const cfg = await _loadConfig();
        const allow = new Set(Array.isArray(cfg.allow) ? cfg.allow : []);
        const never = new Set(Array.isArray(cfg.never) ? cfg.never : []);
        allow.delete(classId);
        never.delete(classId);
        if (choice === "always") allow.add(classId);
        else if (choice === "never") never.add(classId);
        await _saveConfigPatch({
            allow: [...allow],
            never: [...never],
        });
    } catch (exc) {
        console.warn("[c2c.autobatch] could not save node override:", exc);
    }
}

function _nodeMenuItems(node) {
    if (!_hasFrameInput(node)) return [];
    const classId = _nodeClass(node);
    return [null, {
        content: "Image batching for this node type",
        submenu: {
            options: [
                {
                    content: "Always",
                    callback: () => { _setNodeOverride(classId, "always"); },
                },
                {
                    content: "Never",
                    callback: () => { _setNodeOverride(classId, "never"); },
                },
                {
                    content: "Default",
                    callback: () => { _setNodeOverride(classId, "default"); },
                },
            ],
        },
    }];
}

function _mergeNodeMenuItems(opts, node) {
    if (!Array.isArray(opts)) return opts;
    if (opts.some((o) => o && /(Auto-batch|Image batching (for)?) this node type/.test(o.content || ""))) return opts;
    const items = _nodeMenuItems(node);
    if (items.length) opts.push(...items);
    return opts;
}

async function _syncModeFromServer() {
    try {
        const cfg = await _loadConfig();
        const mode = (cfg && cfg.mode) || MODE_INTERNAL;
        _syncingFromServer = true;
        try {
            app.ui?.settings?.setSettingValue?.(SETTING_MODE, mode);
        } finally {
            _syncingFromServer = false;
        }
    } catch (exc) {
        console.warn("[c2c.autobatch] could not load config from server:", exc);
    }
}

if (!(app.extensions || []).some((e) => e?.name === EXT_NAME)) {
    app.registerExtension({
        name: EXT_NAME,
        settings: [
            {
                id: SETTING_MODE,
                // Named the way the owner asks for it ("image batching"): the Settings search matches the name.
                name: "Image batching (auto-batch long image / mask / video frame batches)",
                type: renderModeButtons,
                defaultValue: MODE_INTERNAL,
                category: ["c2c", "Image batching", "Mode"],
                tooltip:
                    "Internal (default): C2C's own image / mask nodes and the measured core list split a long batch "
                    + "into memory-sized chunks when it would not fit, and give the same result. "
                    + "Universal: any author's image / mask nodes, each checked frame-independent first "
                    + "(experimental). Off: never split. Per node type: right-click a node > Image batching for this node type. "
                    + "Env C2C_AUTOBATCH=0 overrides everything.",
                // Frontend 1.52 calls onChange(value, undefined) when the setting is REGISTERED, at every
                // page load; posting then overwrote the server file with the browser's stored value. Only
                // a real change (old value known) and not our own server sync writes the file.
                onChange: (v, old) => {
                    if (old === undefined || _syncingFromServer) return;
                    _setMode(v);
                },
            },
        ],
        commands: [MODE_OFF, MODE_INTERNAL, MODE_UNIVERSAL].map((mode) => ({
            id: `c2c.autobatch.mode.${mode}`,
            label: `C2C: Image batching ${MODE_LABELS[mode]}`,
            function: () => {
                _syncingFromServer = true;
                try { app.ui?.settings?.setSettingValue?.(SETTING_MODE, mode); } finally { _syncingFromServer = false; }
                _setMode(mode);
                app.extensionManager?.toast?.add?.({ severity: "info", summary: "Image batching",
                    detail: `${MODE_LABELS[mode]}: ${MODE_HINTS[mode]}`, life: 4000 });
            },
        })),
        getNodeMenuItems(node) {
            return _nodeMenuItems(node);
        },
        async setup() {
            await _syncModeFromServer();
            legacyNodeMenu("autobatch", _mergeNodeMenuItems);
        },
    });
}
