/**
 * _c2c_slot_dblclick.js — the ONE setting that decides what a double-click on a node slot does.
 *
 * Two C2C features used to listen for slot double-clicks on the same canvas element (Get/Set spawning and the
 * auto-connect predictor). Both ran on every double-click: a Get/Set node appeared AND a predicted node was
 * inserted or wired to a nearby node (owner, 2026-10-08, ORDERS A9). Each feature now checks this setting and
 * only the selected one acts.
 *
 * Apache-2.0 © Code2Collapse.
 */

import { app } from "../../scripts/app.js";

export const DBL_SETTING_ID = "c2c.slotDoubleClick";
export const DBL_NOTHING = "Nothing";
export const DBL_GETSET = "Get/Set variable";
export const DBL_SUGGEST = "Suggest and connect a node (experimental)";
export const DBL_OPTIONS = [DBL_NOTHING, DBL_GETSET, DBL_SUGGEST];

/** The selected behaviour; Get/Set when the setting is unset or unreadable. */
export function slotDoubleClickMode() {
    try {
        const v = app.ui?.settings?.getSettingValue?.(DBL_SETTING_ID, DBL_GETSET);
        return DBL_OPTIONS.includes(v) ? v : DBL_GETSET;
    } catch {
        return DBL_GETSET;
    }
}

// ── Nodes 2.0: slots are DOM elements ──────────────────────────────────────
// The features above listen for dblclick on the canvas element, but in Nodes 2.0 a slot is a DOM element above the
// canvas, so a double-click on it never reached them (L2.15 link_integrity_probe --nodes2, ledger L2.35). One
// document-level capture listener resolves the slot from its `data-slot-key` ("<nodeId>-in-<i>" / "<nodeId>-out-<i>",
// frontend 1.52.7) and offers it to the registered handlers; the first that returns true handles it.
const _handlers = [];
const SLOT_KEY = /^(.+)-(in|out)-(\d+)$/;

function _onDomDblClick(e) {
    if (!globalThis.LiteGraph?.vueNodesMode || !_handlers.length) return;
    const el = e.target instanceof Element ? e.target.closest(".lg-slot") : null;
    const key = el?.querySelector?.("[data-slot-key]")?.dataset?.slotKey || el?.dataset?.slotKey;
    const m = key && SLOT_KEY.exec(key);
    if (!m) return;
    const node = app.canvas?.graph?.getNodeById?.(m[1]);
    if (!node) return;
    const hit = { node, isInput: m[2] === "in", slot: Number(m[3]), event: e };
    for (const h of _handlers) {
        let done = false;
        try { done = h(hit) === true; } catch (err) { console.warn("[C2C.SlotDblClick]", err); }
        if (done) { e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation(); return; }
    }
}

/** Register a Nodes 2.0 slot double-click handler: ({ node, isInput, slot, event }) => true when handled. */
export function onDomSlotDoubleClick(handler) {
    if (typeof handler !== "function") return;
    _handlers.push(handler);
    if (!globalThis.__c2cDomSlotDblClick) {
        globalThis.__c2cDomSlotDblClick = true;
        document.addEventListener("dblclick", _onDomDblClick, true);
    }
}
