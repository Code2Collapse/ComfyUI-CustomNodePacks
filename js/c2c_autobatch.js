/**
 * c2c_autobatch.js — universal / curated automatic frame batching controls.
 */
import { app } from "../../scripts/app.js";
import { legacyNodeMenu } from "./_c2c_compat.js";

const EXT_NAME = "c2c.autobatch";
const SETTING_MODE = "c2c.autobatch.mode";
const ROUTE_CONFIG = "/c2c/autobatch/config";

const MODE_OFF = "off";
const MODE_CURATED = "curated";
const MODE_UNIVERSAL = "universal";

const MODE_LABELS = {
    [MODE_OFF]: "Off",
    [MODE_CURATED]: "Curated",
    [MODE_UNIVERSAL]: "Universal (experimental)",
};

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
        content: "Auto-batch this node type",
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
    if (opts.some((o) => o && /Auto-batch this node type/.test(o.content || ""))) return opts;
    const items = _nodeMenuItems(node);
    if (items.length) opts.push(...items);
    return opts;
}

async function _syncModeFromServer() {
    try {
        const cfg = await _loadConfig();
        const mode = (cfg && cfg.mode) || MODE_OFF;
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
                name: "C2C ▸ Performance ▸ Auto-batch mode",
                type: "combo",
                options: [
                    { value: MODE_OFF, text: MODE_LABELS[MODE_OFF] },
                    { value: MODE_CURATED, text: MODE_LABELS[MODE_CURATED] },
                    { value: MODE_UNIVERSAL, text: MODE_LABELS[MODE_UNIVERSAL] },
                ],
                defaultValue: MODE_OFF,
                category: ["c2c", "Performance", "Auto-batch"],
                tooltip:
                    "Off — no automatic chunking. Curated — measured core nodes only. "
                    + "Universal (experimental) — auto-detect image nodes and probe frame "
                    + "independence before chunking.",
                // Frontend 1.52 calls onChange(value, undefined) when the setting is REGISTERED, at every
                // page load; posting then overwrote the server file with the browser's stored value. Only
                // a real change (old value known) and not our own server sync writes the file.
                onChange: (v, old) => {
                    if (old === undefined || _syncingFromServer) return;
                    _setMode(v);
                },
            },
        ],
        getNodeMenuItems(node) {
            return _nodeMenuItems(node);
        },
        async setup() {
            await _syncModeFromServer();
            legacyNodeMenu("autobatch", _mergeNodeMenuItems);
        },
    });
}
