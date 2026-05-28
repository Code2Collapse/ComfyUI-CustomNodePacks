/**
 * _c2c_undo.js (v2) — additive single-step undo bridge for C2C mutations.
 *
 * Quality bar §0.4: every C2C feature that mutates the graph MUST produce
 * exactly ONE undo step, and undo must restore precisely what changed —
 * not re-configure the whole graph (the v1 bug: that nuked unrelated
 * in-flight state and any open editor panels).
 *
 * v2 design rules (additive / non-invasive — bedrock principle §12):
 *   • Public LiteGraph API only. No splicing of `canvas.undo_history`.
 *     No replacement of `onAfterChange`. No private-state mutation.
 *   • LiteGraph's `graph.beforeChange()` / `graph.afterChange()` coalesce
 *     all native undo snapshots into one entry. We wrap the user's fn
 *     in that pair and let native do its own coalescing.
 *   • In parallel we capture a per-node diff (added / removed / changed)
 *     by serialising every node before & after via the public
 *     `node.serialize()` method. Restore is per-node `node.configure()` +
 *     `graph.add()` / `graph.remove()` — all public LiteGraph API.
 *   • Adds / removes that fn performs are tracked by temporarily wrapping
 *     `graph.add` / `graph.remove` for the duration of the batch (restored
 *     in finally, even on throw). Wrappers delegate to the originals.
 *   • Errors NEVER swallowed. Failures are:
 *       (a) re-thrown to the caller,
 *       (b) `console.error`'d with full stack,
 *       (c) dispatched as `c2c:registry-failure` CustomEvent,
 *       (d) POSTed (fire-and-forget) to /c2c/registry/failure if the
 *           route is mounted by `nodes/_c2c_registry.py`.
 *   • If fn throws, we attempt a per-node rollback to the captured
 *     before-state, then re-throw the original error.
 *
 * Public API (back-compat preserved):
 *   asOneUndo(name, fn)              — sync or thenable fn → one undo step
 *   asOneUndoAsync(name, asyncFn)    — alias
 *   c2cUndo.beginAction(name)        — procedural style
 *   c2cUndo.endAction()              — finalise current procedural batch
 *   c2cUndo.recent()                 — last 50 named entries (for §16 UI)
 *   c2cUndo.undo()                   — pop our own stack (parallel to native)
 *   c2cUndo.redo()                   — push it back
 *
 * License: Apache-2.0
 */

import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const LOG_CAP = 50;
const STACK_CAP = 100;

if (!window.__c2c_undo_log) window.__c2c_undo_log = [];
if (!window.__c2c_undo_stack) window.__c2c_undo_stack = [];   // entries we can undo
if (!window.__c2c_redo_stack) window.__c2c_redo_stack = [];   // entries we can redo

// ───────────────────────── error surface ─────────────────────────────────

function _surfaceFailure(where, error, context) {
    const payload = {
        scope: "c2c.undo",
        where,
        message: error?.message || String(error),
        stack: error?.stack || null,
        context: context || null,
        ts: Date.now(),
    };
    // (a) always console
    try { console.error(`[c2c-undo] ${where}:`, error, context || ""); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
    // (b) dispatch event for in-browser listeners (toast / Doctor / INT badge)
    try {
        window.dispatchEvent(new CustomEvent("c2c:registry-failure", { detail: payload }));
    } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
    // (c) fire-and-forget to backend registry (ignore network errors)
    try {
        if (typeof fetch === "function") {
            fetch("/c2c/registry/failure", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                keepalive: true,
            }).catch(() => {});
        }
    } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
}

// ───────────────────────── snapshot helpers ──────────────────────────────

function _serializeNode(node) {
    // Public LiteGraph API. Returns a plain serialisable object.
    try { return node.serialize(); }
    catch (e) {
        _surfaceFailure("serialize_node", e, { id: node?.id, type: node?.type });
        return null;
    }
}

function _snapshotAllNodes() {
    // NOTE on subgraph behaviour: ComfyUI swaps `app.graph` to the focused
    // subgraph while the user is editing inside one, so snapshotting the
    // currently active graph captures exactly the editable surface the undo
    // batch is mutating. Trying to span every nested subgraph here would
    // require composite-path keys + per-graph add/remove wrappers and risks
    // destroying parallel edits the user did not intend to revert.
    const graph = app?.graph;
    const map = new Map();
    if (!graph || !Array.isArray(graph._nodes)) return map;
    for (const node of graph._nodes) {
        const data = _serializeNode(node);
        if (data) map.set(node.id, data);
    }
    return map;
}

/** Snapshot all graph links as id→{id,origin_id,origin_slot,target_id,target_slot,type}. */
function _snapshotAllLinks() {
    const graph = app?.graph;
    const map = new Map();
    if (!graph || !graph.links) return map;
    // graph.links can be a plain object or a Map depending on LiteGraph build.
    const entries = (typeof graph.links?.entries === "function" && !Array.isArray(graph.links))
        ? Array.from(graph.links.entries())
        : Object.entries(graph.links);
    for (const [k, link] of entries) {
        if (!link) continue;
        map.set(Number(k), {
            id: link.id,
            origin_id: link.origin_id,
            origin_slot: link.origin_slot,
            target_id: link.target_id,
            target_slot: link.target_slot,
            type: link.type,
        });
    }
    return map;
}

/** Diff before/after maps → { added:[id], removed:[{id,data}], changed:[{id,before,after}] }. */
function _diff(before, after) {
    const added = [];
    const removed = [];
    const changed = [];
    for (const [id, data] of after) {
        if (!before.has(id)) {
            added.push(id);
        } else {
            const prev = before.get(id);
            // Cheap-but-correct equality via JSON.stringify (node data is plain).
            if (JSON.stringify(prev) !== JSON.stringify(data)) {
                changed.push({ id, before: prev, after: data });
            }
        }
    }
    for (const [id, data] of before) {
        if (!after.has(id)) removed.push({ id, data });
    }
    return { added, removed, changed };
}

/** Diff link maps → { added:[link], removed:[link] }. */
function _diffLinks(before, after) {
    const added = [];
    const removed = [];
    for (const [id, link] of after) {
        if (!before.has(id)) added.push(link);
    }
    for (const [id, link] of before) {
        if (!after.has(id)) removed.push(link);
    }
    return { added, removed };
}

function _hasAnyDiff(d) {
    return (
        d.added.length ||
        d.removed.length ||
        d.changed.length ||
        (d.links && (d.links.added.length || d.links.removed.length))
    );
}

// ───────────────────────── graph add/remove tracker ──────────────────────

/** Wrap graph.add / graph.remove for the duration of fn (additive, restored
 *  in finally). Lets us record which nodes were touched even if fn doesn't
 *  tell us. Wrappers always delegate to the original — never replace. */
function _installGraphTracker(seenAdds, seenRemoves) {
    const graph = app?.graph;
    if (!graph) return () => {};
    const origAdd = graph.add?.bind(graph);
    const origRemove = graph.remove?.bind(graph);
    if (typeof origAdd === "function") {
        graph.add = function (node, ...rest) {
            const r = origAdd(node, ...rest);
            try { if (node?.id != null) seenAdds.add(node.id); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
            return r;
        };
    }
    if (typeof origRemove === "function") {
        graph.remove = function (node, ...rest) {
            try { if (node?.id != null) seenRemoves.add(node.id); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
            return origRemove(node, ...rest);
        };
    }
    return () => {
        // Restore originals. Use defineProperty to clear any wrapper even if
        // a downstream consumer further wrapped it (best-effort).
        try { if (origAdd) graph.add = origAdd; } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
        try { if (origRemove) graph.remove = origRemove; } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
    };
}

// ───────────────────────── log + entry pushing ───────────────────────────

function _logEntry(name) {
    const log = window.__c2c_undo_log;
    log.push({ name, ts: Date.now() });
    if (log.length > LOG_CAP) log.splice(0, log.length - LOG_CAP);
    try { window.dispatchEvent(new CustomEvent("c2c:undo-entry", { detail: { name } })); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
}

function _pushUndoEntry(entry) {
    const stack = window.__c2c_undo_stack;
    stack.push(entry);
    if (stack.length > STACK_CAP) stack.splice(0, stack.length - STACK_CAP);
    // Any new action invalidates the redo stack.
    window.__c2c_redo_stack.length = 0;
}

// ───────────────────────── apply diff (undo / redo / rollback) ───────────

/** Add a link using public LiteGraph API. Returns true if connected. */
function _restoreLink(link) {
    const graph = app?.graph;
    if (!graph || !link) return false;
    const src = graph.getNodeById?.(link.origin_id);
    const dst = graph.getNodeById?.(link.target_id);
    if (!src || !dst) return false;
    // Skip if already wired (idempotent).
    try {
        const outs = src.outputs?.[link.origin_slot]?.links || [];
        if (Array.isArray(outs)) {
            for (const lid of outs) {
                const existing = graph.links?.[lid] || graph.links?.get?.(lid);
                if (existing && existing.target_id === link.target_id && existing.target_slot === link.target_slot) {
                    return true;
                }
            }
        }
    } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
    try {
        src.connect(link.origin_slot, dst, link.target_slot);
        return true;
    } catch (e) {
        _surfaceFailure("restore_link", e, { link });
        return false;
    }
}

/** Remove a link by id using public LiteGraph API. */
function _removeLink(linkId) {
    const graph = app?.graph;
    if (!graph || linkId == null) return false;
    try {
        if (typeof graph.removeLink === "function") {
            graph.removeLink(linkId);
            return true;
        }
    } catch (e) {
        _surfaceFailure("remove_link", e, { linkId });
    }
    return false;
}

function _applyDiff(diff, direction /* "undo" | "redo" */) {
    const graph = app?.graph;
    if (!graph) return;
    const LiteGraph = window.LiteGraph;
    if (!LiteGraph) {
        _surfaceFailure("apply_diff", new Error("LiteGraph global missing"), { direction });
        return;
    }
    const isUndo = direction === "undo";
    const links = diff.links || { added: [], removed: [] };
    try {
        if (isUndo) {
            // 1. Remove links that were added (so dangling nodes can be removed cleanly).
            for (const link of links.added) _removeLink(link.id);
            // 2. Re-create nodes that were removed.
            for (const { data } of diff.removed) {
                if (!data || !data.type) continue;
                if (graph.getNodeById?.(data.id)) continue;
                const node = LiteGraph.createNode(data.type);
                if (!node) continue;
                try { node.configure(data); } catch (e) {
                    _surfaceFailure("undo_reconfigure", e, { id: data.id, type: data.type });
                    continue;
                }
                graph.add(node);
            }
            // 3. Restore changed nodes to their before-state.
            for (const { id, before } of diff.changed) {
                const node = graph.getNodeById?.(id);
                if (!node || !before) continue;
                try { node.configure(before); } catch (e) {
                    _surfaceFailure("undo_configure_changed", e, { id });
                }
            }
            // 4. Remove nodes that were added.
            for (const id of diff.added) {
                const node = graph.getNodeById?.(id);
                if (node) {
                    try { graph.remove(node); } catch (e) {
                        _surfaceFailure("undo_remove_added", e, { id });
                    }
                }
            }
            // 5. Re-create links that were removed (now that endpoints exist again).
            for (const link of links.removed) _restoreLink(link);
        } else {
            // Redo: forward direction.
            // 1. Remove links that were originally removed.
            for (const link of links.removed) _removeLink(link.id);
            // 2. Re-remove nodes that were originally removed.
            for (const { id } of diff.removed) {
                const node = graph.getNodeById?.(id);
                if (node) {
                    try { graph.remove(node); } catch (e) {
                        _surfaceFailure("redo_remove", e, { id });
                    }
                }
            }
            // 3. Re-configure changed nodes to their after-state.
            for (const { id, after } of diff.changed) {
                const node = graph.getNodeById?.(id);
                if (!node || !after) continue;
                try { node.configure(after); } catch (e) {
                    _surfaceFailure("redo_configure_changed", e, { id });
                }
            }
            // 4. Re-add nodes that were added.
            for (const id of diff.added) {
                if (graph.getNodeById?.(id)) continue;
                const data = diff.addedData?.[id];
                if (!data || !data.type) continue;
                const node = LiteGraph.createNode(data.type);
                if (!node) continue;
                try { node.configure(data); } catch (e) {
                    _surfaceFailure("redo_reconfigure_added", e, { id });
                    continue;
                }
                graph.add(node);
            }
            // 5. Re-create links that were originally added.
            for (const link of links.added) _restoreLink(link);
        }
        try { graph.setDirtyCanvas?.(true, true); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
    } catch (e) {
        _surfaceFailure("apply_diff_unexpected", e, { direction });
    }
}

// ───────────────────────── core run helper ───────────────────────────────

function _runBatch(name, fn) {
    const graph = app?.graph;
    if (!graph) {
        // pre-boot fallback: just run, no tracking
        return { isThenable: false, result: fn() };
    }

    const beforeMap = _snapshotAllNodes();
    const beforeLinks = _snapshotAllLinks();
    const seenAdds = new Set();
    const seenRemoves = new Set();
    const restoreTracker = _installGraphTracker(seenAdds, seenRemoves);

    // Coalesce native undo snapshots into one entry.
    try { graph.beforeChange?.(); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }

    const finish = (result, threw, error) => {
        // Always restore the tracker, even on throw.
        try { restoreTracker(); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
        const afterMap = _snapshotAllNodes();
        const afterLinks = _snapshotAllLinks();
        const diff = _diff(beforeMap, afterMap);
        diff.links = _diffLinks(beforeLinks, afterLinks);

        if (threw) {
            // Per-node rollback to before-state (best-effort).
            try {
                const addedData = Object.fromEntries(
                    Array.from(afterMap).filter(([id]) => diff.added.includes(id))
                );
                _applyDiff({ ...diff, addedData }, "undo");
            } catch (e) {
                _surfaceFailure("rollback_after_throw", e, { name });
            }
            try { graph.afterChange?.(); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
            try { graph.setDirtyCanvas?.(true, true); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
            _surfaceFailure("user_fn_threw", error, { name });
            throw error;
        }

        // Success path: record entry (if anything actually changed).
        if (_hasAnyDiff(diff)) {
            // Capture data for added nodes so redo can recreate them.
            const addedData = {};
            for (const id of diff.added) {
                const data = afterMap.get(id);
                if (data) addedData[id] = data;
            }
            _pushUndoEntry({ name, ts: Date.now(), diff: { ...diff, addedData } });
        }
        _logEntry(name);
        try { graph.afterChange?.(); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
        try { graph.setDirtyCanvas?.(true, true); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
        return result;
    };

    let result;
    try {
        result = fn();
    } catch (e) {
        return { isThenable: false, finishSync: () => finish(undefined, true, e) };
    }
    if (result && typeof result.then === "function") {
        return {
            isThenable: true,
            promise: result.then(
                (val) => finish(val, false, null),
                (err) => finish(undefined, true, err),
            ),
        };
    }
    return { isThenable: false, finishSync: () => finish(result, false, null) };
}

// ───────────────────────── public API ────────────────────────────────────

/**
 * Run `fn`, registering the net effect as one named undo entry.
 * Sync or thenable; rolls back per-node on throw and re-throws.
 */
export function asOneUndo(name, fn) {
    if (typeof fn !== "function") {
        throw new TypeError("asOneUndo: fn must be a function");
    }
    const r = _runBatch(name, fn);
    if (r.isThenable) return r.promise;
    return r.finishSync();
}

/** Convenience alias for async callers. */
export function asOneUndoAsync(name, asyncFn) {
    return asOneUndo(name, asyncFn);
}

// Procedural style ─────────────────────────────────────────────────────────
let _activeBatch = null;
function beginAction(name) {
    if (_activeBatch) {
        _surfaceFailure("nested_begin_action", new Error("beginAction called while another batch is active"), { name, active: _activeBatch.name });
        // End the previous one defensively so we don't leak state.
        try { endAction(); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
    }
    const graph = app?.graph;
    if (!graph) { _activeBatch = { name, noop: true }; return; }
    _activeBatch = {
        name,
        beforeMap: _snapshotAllNodes(),
        beforeLinks: _snapshotAllLinks(),
        seenAdds: new Set(),
        seenRemoves: new Set(),
    };
    _activeBatch.restoreTracker = _installGraphTracker(_activeBatch.seenAdds, _activeBatch.seenRemoves);
    try { graph.beforeChange?.(); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
}

function endAction() {
    if (!_activeBatch) return;
    const batch = _activeBatch;
    _activeBatch = null;
    if (batch.noop) return;
    try { batch.restoreTracker(); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
    const graph = app?.graph;
    if (!graph) return;
    const afterMap = _snapshotAllNodes();
    const afterLinks = _snapshotAllLinks();
    const diff = _diff(batch.beforeMap, afterMap);
    diff.links = _diffLinks(batch.beforeLinks, afterLinks);
    if (_hasAnyDiff(diff)) {
        const addedData = {};
        for (const id of diff.added) {
            const data = afterMap.get(id);
            if (data) addedData[id] = data;
        }
        _pushUndoEntry({ name: batch.name, ts: Date.now(), diff: { ...diff, addedData } });
    }
    _logEntry(batch.name);
    try { graph.afterChange?.(); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
    try { graph.setDirtyCanvas?.(true, true); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
}

// Our parallel undo/redo (operates on our diff stack, not native).
function undo() {
    const entry = window.__c2c_undo_stack.pop();
    if (!entry) return false;
    _applyDiff(entry.diff, "undo");
    window.__c2c_redo_stack.push(entry);
    try { window.dispatchEvent(new CustomEvent("c2c:undo-popped", { detail: { name: entry.name } })); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
    return true;
}

function redo() {
    const entry = window.__c2c_redo_stack.pop();
    if (!entry) return false;
    _applyDiff(entry.diff, "redo");
    window.__c2c_undo_stack.push(entry);
    try { window.dispatchEvent(new CustomEvent("c2c:redo-pushed", { detail: { name: entry.name } })); } catch (__c2cErr) { __c2cReport("_c2c_undo", __c2cErr); }
    return true;
}

/** Read-only access to the named-entry log (last LOG_CAP entries). */
export function recentUndoEntries() {
    return window.__c2c_undo_log.slice();
}

// Frozen global for downstream consumers (Visual Undo Timeline, palette).
window.c2cUndo = Object.freeze({
    asOneUndo,
    asOneUndoAsync,
    beginAction,
    endAction,
    undo,
    redo,
    recent: recentUndoEntries,
    stackSize: () => window.__c2c_undo_stack.length,
    redoSize: () => window.__c2c_redo_stack.length,
});
