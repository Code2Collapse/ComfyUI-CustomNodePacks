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
