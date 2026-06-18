// c2c_linked_values.js — Linked Value Groups (C2C v2.0 §6.3)
// ---------------------------------------------------------------------
// What it does:
//   • Right-click a numeric/string/boolean widget → "Link with group..."
//     prompts for a group name; multiple widgets bound to the same group
//     update in lock-step whenever any of them changes.
//   • Right-click again → "Unlink from group X".
//   • Bottom-right status pill: "N links" (click → manager modal listing
//     groups, members, master value; click row → pulse member node).
//   • Persisted under `graph.extra.c2c.linkedValues = { "<group>": {
//       master: { node:<id>, widget:<name> },
//       members: [ { node, widget }, ... ],
//       last: <value>
//     } }`
//   • Settings: c2c.linkedValues.enabled (default true),
//               c2c.linkedValues.propagate_master_only (default false).
//   • Conflict policy: when value changes on a non-master member, the
//     change still propagates (so any member can drive). Master node is
//     used purely as the originator-of-record in the modal.
//
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";
import { forAllNodes, findNodeAnywhere } from "./_subgraph_walk.js";

const SETTING_ENABLED = "c2c.linkedValues.enabled";
const SETTING_MASTER_ONLY = "c2c.linkedValues.propagate_master_only";
const PULSE_MS = 900;
const PILL_ID = "c2c-linked-values-pill";

let _propagating = false;       // re-entrancy guard
let _enabled = true;
let _masterOnly = false;

function getStore() {
    const g = app.graph;
    if (!g) return null;
    g.extra = g.extra || {};
    g.extra.c2c = g.extra.c2c || {};
    g.extra.c2c.linkedValues = g.extra.c2c.linkedValues || {};
    return g.extra.c2c.linkedValues;
}

function saveStore() {
    // No-op: we mutate graph.extra directly so it serializes with the
    // workflow automatically.
}

function groupOfWidget(nodeId, wName) {
    const s = getStore(); if (!s) return null;
    for (const [name, grp] of Object.entries(s)) {
        if (grp.master && grp.master.node === nodeId && grp.master.widget === wName) return name;
        if ((grp.members || []).some(m => m.node === nodeId && m.widget === wName)) return name;
    }
    return null;
}

function listAllNumericLikeWidgets(node) {
    return (node.widgets || []).filter(w =>
        ["number", "slider", "combo", "toggle", "string", "text"].includes(w.type)
        || typeof w.value === "number" || typeof w.value === "string"
        || typeof w.value === "boolean"
    );
}

function propagate(groupName, value, originNodeId, originWidgetName) {
    if (_propagating) return;
    const s = getStore(); if (!s) return;
    const grp = s[groupName]; if (!grp) return;
    grp.last = value;
    _propagating = true;
    try {
        const targets = [grp.master, ...(grp.members || [])].filter(Boolean);
        for (const t of targets) {
            if (!t) continue;
            if (t.node === originNodeId && t.widget === originWidgetName) continue;
            if (_masterOnly && !(grp.master && grp.master.node === originNodeId && grp.master.widget === originWidgetName)) continue;
            const node = findNodeAnywhere(t.node)?.node;
            if (!node) continue;
            const w = (node.widgets || []).find(x => x.name === t.widget);
            if (!w) continue;
            // Use the widget's existing callback if it has clamping logic,
            // otherwise assign directly + setDirtyCanvas.
            const prev = w.value;
            try {
                w.value = value;
                if (typeof w.callback === "function") w.callback(value);
            } catch (e) {
                w.value = prev;
            }
        }
        app.graph.setDirtyCanvas(true, true);
    } finally {
        _propagating = false;
    }
    refreshPill();
}

function hookWidget(node, widget) {
    if (widget.__c2c_lv_hooked) return;
    widget.__c2c_lv_hooked = true;
    const origCb = widget.callback;
    widget.callback = function (v, ...rest) {
        const ret = origCb ? origCb.call(this, v, ...rest) : undefined;
        const g = groupOfWidget(node.id, widget.name);
        if (g && !_propagating && _enabled) {
            propagate(g, widget.value, node.id, widget.name);
        }
        return ret;
    };
}

function rehookAll() {
    if (!app.graph) return;
    forAllNodes((n) => {
        for (const w of n.widgets || []) hookWidget(n, w);
    });
}

// ── Pill UI ───────────────────────────────────────────────────────────
function ensurePill() {
    let el = document.getElementById(PILL_ID);
    if (el) return el;
    el = document.createElement("div");
    el.id = PILL_ID;
    el.style.cssText = `
        position: fixed; right: 14px; bottom: 158px; z-index: var(--c2c-z-hud);
        background: color-mix(in srgb, var(--c2c-panelBg) 84%, transparent);
        color: var(--c2c-accentLight2); font: 11px ui-sans-serif, system-ui, sans-serif;
        padding: 4px 10px; border-radius: 13px;
        border: 1px solid var(--c2c-border);
        cursor: pointer; user-select: none; backdrop-filter: blur(6px);
    `;
    el.title = "Linked Value Groups — click to manage";
    el.addEventListener("click", openManager);
    document.body.appendChild(el);
    // Register with dock anchor if present so toasts don't overlap.
    try {
        if (window.__mecDock && typeof window.__mecDock.register === "function") {
            window.__mecDock.register(el, { baseBottom: 158 });
        }
    } catch {}
    return el;
}

function refreshPill() {
    if (!_enabled) {
        const el = document.getElementById(PILL_ID);
        if (el) el.style.display = "none";
        return;
    }
    const el = ensurePill();
    const s = getStore() || {};
    const n = Object.keys(s).length;
    el.style.display = n > 0 ? "block" : "none";
    el.textContent = `${n} link${n === 1 ? "" : "s"}`;
}

function pulseNode(nodeId) {
    const n = findNodeAnywhere(nodeId)?.node;
    if (!n) return;
    const orig = n.bgcolor;
    // Read live accent from the active theme so the pulse flips with mocha/latte/oled.
    // If the theme stylesheet has not been injected yet, skip the pulse rather than
    // burn in a palette-specific literal (would break latte/oled tint).
    const accent = (getComputedStyle(document.documentElement).getPropertyValue("--c2c-accentSoft2") || "").trim();
    if (!accent) return;
    n.bgcolor = accent;
    app.canvas.centerOnNode(n);
    app.graph.setDirtyCanvas(true, true);
    setTimeout(() => { n.bgcolor = orig; app.graph.setDirtyCanvas(true, true); }, PULSE_MS);
}

function openManager() {
    const s = getStore() || {};
    const dlg = document.createElement("div");
    // bg2 + accentText both flip across mocha/latte/oled; panelDeep2 does NOT.
    dlg.style.cssText = `
        position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%);
        z-index: var(--c2c-z-modal); width: 540px; max-height: 70vh; overflow: auto;
        background: var(--c2c-bg2); color: var(--c2c-accentText);
        border: 1px solid var(--c2c-border); border-radius: 8px;
        padding: 14px 18px; font: 12px ui-sans-serif, system-ui, sans-serif;
        box-shadow: 0 12px 36px color-mix(in srgb, var(--c2c-shadowBase) 65%, transparent);
    `;
    const close = () => dlg.remove();
    const h = document.createElement("div");
    h.style.cssText = "font-weight:600;font-size:13px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;";
    h.innerHTML = `<span>Linked Value Groups</span><span style="cursor:pointer;color:var(--c2c-accentMuted2)" title="Close">✕</span>`;
    h.lastChild.addEventListener("click", close);
    dlg.appendChild(h);

    const entries = Object.entries(s);
    if (entries.length === 0) {
        const e = document.createElement("div");
        e.style.cssText = "padding:24px 8px;color:var(--c2c-accentMuted2);text-align:center;";
        e.textContent = "No groups yet. Right-click any widget → \"Link with group...\"";
        dlg.appendChild(e);
    } else {
        for (const [name, grp] of entries) {
            const card = document.createElement("div");
            card.style.cssText = "margin-bottom:10px;padding:8px 10px;background:color-mix(in srgb, var(--c2c-surface0) 30%, transparent);border-radius:6px;";
            const title = document.createElement("div");
            title.style.cssText = "font-weight:600;color:var(--c2c-accentLight);margin-bottom:4px;display:flex;justify-content:space-between;";
            title.innerHTML = `<span>${escapeHtml(name)} <span style="color:var(--c2c-accentMuted2);font-weight:400">· last = ${escapeHtml(String(grp.last))}</span></span>`;
            const del = document.createElement("span");
            del.textContent = "Delete";
            del.style.cssText = "cursor:pointer;color:var(--c2c-dangerSoft2);font-size:11px;font-weight:400;";
            del.addEventListener("click", () => { delete s[name]; refreshPill(); close(); openManager(); });
            title.appendChild(del);
            card.appendChild(title);
            const members = [grp.master, ...(grp.members || [])].filter(Boolean);
            for (const m of members) {
                const isMaster = grp.master && m.node === grp.master.node && m.widget === grp.master.widget;
                const node = findNodeAnywhere(m.node)?.node;
                const row = document.createElement("div");
                row.style.cssText = "padding:3px 6px;cursor:pointer;border-radius:3px;display:flex;justify-content:space-between;";
                row.innerHTML = `<span>${isMaster ? "★ " : "  "}#${m.node} · ${escapeHtml(node ? node.title || node.type : "missing")} · <code>${escapeHtml(m.widget)}</code></span>`;
                row.addEventListener("mouseenter", () => row.style.background = "color-mix(in srgb, var(--c2c-accentSoft2) 16%, transparent)");
                row.addEventListener("mouseleave", () => row.style.background = "");
                row.addEventListener("click", () => { pulseNode(m.node); close(); });
                card.appendChild(row);
            }
            dlg.appendChild(card);
        }
    }
    document.body.appendChild(dlg);
    const onKey = (e) => { if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); } };
    document.addEventListener("keydown", onKey);
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]);
}

// ── Right-click menu integration ──────────────────────────────────────
function widgetMenuItems(node) {
    const items = [];
    for (const w of (node.widgets || [])) {
        const existing = groupOfWidget(node.id, w.name);
        if (existing) {
            items.push({
                content: `Unlink "${w.name}" from group "${existing}"`,
                callback: () => {
                    const s = getStore();
                    const grp = s[existing];
                    if (!grp) return;
                    if (grp.master && grp.master.node === node.id && grp.master.widget === w.name) {
                        // Promote first member to master.
                        const next = (grp.members || []).shift();
                        grp.master = next || null;
                    } else {
                        grp.members = (grp.members || []).filter(m => !(m.node === node.id && m.widget === w.name));
                    }
                    if (!grp.master && (!grp.members || grp.members.length === 0)) delete s[existing];
                    refreshPill();
                },
            });
        } else {
            items.push({
                content: `Link "${w.name}" with group...`,
                callback: async () => {
                    const groups = Object.keys(getStore() || {});
                    const promptText = groups.length
                        ? `Group name (existing: ${groups.join(", ")}):`
                        : "Group name for the new link:";
                    const name = prompt(promptText, groups[0] || "group_1");
                    if (!name) return;
                    const s = getStore();
                    if (!s[name]) {
                        s[name] = { master: { node: node.id, widget: w.name }, members: [], last: w.value };
                    } else {
                        // Don't double-add.
                        const already = (s[name].members || []).some(m => m.node === node.id && m.widget === w.name);
                        if (!already) {
                            s[name].members = s[name].members || [];
                            s[name].members.push({ node: node.id, widget: w.name });
                            // Sync new member to current group value.
                            try { w.value = s[name].last; if (w.callback) w.callback(s[name].last); } catch {}
                        }
                    }
                    hookWidget(node, w);
                    refreshPill();
                },
            });
        }
    }
    return items;
}

// ── Extension wiring ──────────────────────────────────────────────────
app.registerExtension({
    name: "C2C.LinkedValues",
    async setup() {
        app.ui.settings.addSetting({
            id: SETTING_ENABLED, name: "C2C ▸ Linked Values: enabled",
            type: "boolean", defaultValue: true,
            onChange: v => { _enabled = !!v; refreshPill(); },
        });
        app.ui.settings.addSetting({
            id: SETTING_MASTER_ONLY, name: "C2C ▸ Linked Values: master-only propagation",
            type: "boolean", defaultValue: false,
            onChange: v => { _masterOnly = !!v; },
        });
        _enabled = app.ui.settings.getSettingValue(SETTING_ENABLED, true);
        _masterOnly = app.ui.settings.getSettingValue(SETTING_MASTER_ONLY, false);

        // Re-hook on graph load and node add.
        const origLoad = app.loadGraphData ? app.loadGraphData.bind(app) : null;
        if (origLoad) {
            app.loadGraphData = function (...args) {
                let r;
                try { r = origLoad(...args); }
                catch (err) {
                    console.warn("[linked_values] loadGraphData upstream threw", err);
                    throw err;
                }
                setTimeout(() => { rehookAll(); refreshPill(); }, 100);
                return r;
            };
        }
        const origAdd = app.graph.add ? app.graph.add.bind(app.graph) : null;
        if (origAdd) {
            app.graph.add = function (node, ...rest) {
                const r = origAdd(node, ...rest);
                setTimeout(() => {
                    for (const w of (node.widgets || [])) hookWidget(node, w);
                }, 0);
                return r;
            };
        }
        rehookAll();
        refreshPill();
    },
    async beforeRegisterNodeDef(nodeType /*, nodeData, app */) {
        const orig = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const r = orig ? orig.call(this, canvas, options) : undefined;
            if (!_enabled) return r;
            const items = widgetMenuItems(this);
            if (items.length) {
                options.push(null, { content: "── Linked Values ──", disabled: true }, ...items);
            }
            return r;
        };
    },
});
