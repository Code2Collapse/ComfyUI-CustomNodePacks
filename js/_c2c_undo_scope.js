/**
 * _c2c_undo_scope.js — Ctrl+Z / Ctrl+Y pressed inside a C2C editor undo the EDITOR, never the whole graph.
 *
 * ComfyUI's ChangeTracker (frontend 1.52) listens for Ctrl+Z on `window` in the CAPTURE phase, registered at app
 * start, so it runs before any editor's own listener and ignores stopPropagation / preventDefault. It skips the
 * key only when a text field has focus or a modal dialog (`[role=dialog][aria-modal=true]`) is open. An inline
 * editor (a canvas inside a node) therefore had every Ctrl+Z ALSO undo the graph: the whole workflow reloaded from
 * the previous state, positions reverted and every node object was replaced (measured, ORDERS A9 / L2.15,
 * docs/evidence/L2.15/editor_key_isolation.*.json).
 *
 * `claimUndo(el)` marks an editor element. While keyboard focus is inside a marked element, the tracker's
 * `undoRedo` reports the key as handled without touching the graph; the editor's own keydown listener does the
 * editor undo. Anywhere else, core behaviour is unchanged. Modal editors use role=dialog + aria-modal instead.
 *
 * Self-contained on purpose: the same file ships in every C2C pack that has an inline editor, and whichever pack
 * loads first installs the one guard (globalThis flag), so packs never stack wrappers.
 *
 * Apache-2.0 © Code2Collapse.
 */

export const UNDO_SCOPE_ATTR = "data-c2c-own-undo";

/** Mark `el` (an editor canvas or its root) as owning Ctrl+Z / Ctrl+Y while it has focus. Returns `el`. */
export function claimUndo(el) {
    try { el?.setAttribute?.(UNDO_SCOPE_ATTR, ""); } catch { /* detached or not an element */ }
    installUndoScope();
    return el;
}

function focusInClaimedEditor() {
    const a = document.activeElement;
    return !!(a && typeof a.closest === "function" && a.closest(`[${UNDO_SCOPE_ATTR}]`));
}

function isUndoRedoKey(e) {
    return !!e && (e.ctrlKey || e.metaKey) && !e.altKey && /^[zy]$/i.test(String(e.key || ""));
}

/**
 * Run `fn` as ONE undo step. Core's tracker counts before/after-change through the canvas events
 * (`emitBeforeChange` / `emitAfterChange`); `graph.beforeChange()` alone never reaches it. Nested calls are fine:
 * the tracker captures once, when the outermost step ends.
 */
export function asOneUndoStep(canvas, fn) {
    const g = canvas?.graph;
    try { canvas?.emitBeforeChange?.(); g?.beforeChange?.(); } catch { /* older frontend: no step grouping */ }
    try {
        return fn();
    } finally {
        try { g?.afterChange?.(); canvas?.emitAfterChange?.(); } catch { /* ignore */ }
    }
}

/** Install the guard once per page. Returns true when it is (or already was) installed. */
export function installUndoScope() {
    const G = globalThis;
    if (G.__c2cUndoScopeInstalled) return true;
    const CT = G.comfyAPI?.changeTracker?.ChangeTracker;
    const proto = CT?.prototype;
    if (!proto || typeof proto.undoRedo !== "function") return false;
    const orig = proto.undoRedo;
    let errors = 0;
    proto.undoRedo = function c2cScopedUndoRedo(e, ...rest) {
        try {
            if (errors < 5 && isUndoRedoKey(e) && focusInClaimedEditor()) return Promise.resolve(true);
        } catch (err) {
            errors += 1;   // after 5 failures the guard stays out of the way: core behaviour only
            console.warn("[C2C] undo scope check failed:", err);
        }
        return orig.call(this, e, ...rest);
    };
    G.__c2cUndoScopeInstalled = true;
    return true;
}

installUndoScope();
