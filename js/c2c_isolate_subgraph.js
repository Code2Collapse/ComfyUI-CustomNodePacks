/**
 * mec_isolate_subgraph.js — Phase 18d: Right-click → Isolate subgraph
 *
 * Selecting one or more nodes and right-clicking → "🔍 Isolate" dims every
 * non-selected node and every link not connecting two selected nodes. Click
 * "↺ Reset isolation" (canvas menu) or press Esc to restore.
 *
 * Setting:
 *   mec.isolate.enabled — bool (default true)
 *   mec.isolate.dim     — number 0..1 (default 0.18) — opacity of dimmed nodes
 */

import { app } from "../../scripts/app.js";

let _isolatedIds = null;  // Set<number> or null

function _settingsEnabled() {
    try { return app.ui.settings.getSettingValue("mec.isolate.enabled", true); }
    catch { return true; }
}
function _dimAlpha() {
    try { return app.ui.settings.getSettingValue("mec.isolate.dim", 0.18); }
    catch { return 0.18; }
}

function _isolate() {
    const sel = app.canvas?.selected_nodes;
    if (!sel) return;
    const ids = new Set();
    if (sel instanceof Map) {
        for (const k of sel.keys()) ids.add(typeof k === "number" ? k : parseInt(k, 10));
    } else {
        for (const k of Object.keys(sel)) ids.add(parseInt(k, 10));
    }
    if (!ids.size) return;
    _isolatedIds = ids;
    app.canvas?.setDirty?.(true, true);
    console.log(`[MEC.Isolate] Isolating ${ids.size} nodes`);
}

function _reset() {
    _isolatedIds = null;
    app.canvas?.setDirty?.(true, true);
}

function _patch() {
    if (LGraphCanvas.prototype._mecIsolatePatched) return;
    LGraphCanvas.prototype._mecIsolatePatched = true;

    // Dim non-isolated nodes by wrapping LGraphCanvas.drawNode.
    //
    // NOTE: ComfyUI's extension API exposes `getCanvasMenuOptions` and
    // `getNodeMenuOptions` (handled below via `app.registerExtension`) but
    // does NOT expose a per-node alpha-modulation hook. To dim a node
    // during normal canvas paints we have to wrap `drawNode` directly.
    // The guard above ensures this runs exactly once.
    const origDrawNode = LGraphCanvas.prototype.drawNode;
    LGraphCanvas.prototype.drawNode = function (node, ctx) {
        if (!_isolatedIds || !_settingsEnabled() || _isolatedIds.has(node.id)) {
            return origDrawNode.apply(this, arguments);
        }
        const prev = ctx.globalAlpha;
        ctx.globalAlpha = prev * _dimAlpha();
        try {
            return origDrawNode.apply(this, arguments);
        } finally {
            ctx.globalAlpha = prev;
        }
    };

    // Esc key to reset.
    window.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && _isolatedIds) {
            _reset();
            e.stopPropagation();
        }
    });
}

app.registerExtension({
    name: "C2C.IsolateSubgraph",
    settings: [
        {
            id: "mec.isolate.enabled",
            name: "Isolate Subgraph: right-click node → Isolate",
            type: "boolean",
            default: true,
        },
        {
            id: "mec.isolate.dim",
            name: "Isolate Subgraph: dimmed alpha (0..1)",
            type: "number",
            default: 0.18,
        },
    ],
    // Extension-API menu hooks (no LGraphCanvas.prototype patching).
    getCanvasMenuOptions(opts /*, canvas */) {
        if (_isolatedIds && _settingsEnabled()) {
            opts.push(null);
            opts.push({ content: "↺ Reset isolation", callback: _reset });
        }
        return opts;
    },
    getNodeMenuOptions(opts /*, node */) {
        if (!_settingsEnabled()) return opts;
        opts.push(null);
        opts.push({
            content: _isolatedIds ? "↺ Reset isolation" : "🔍 Isolate selection",
            callback: () => { _isolatedIds ? _reset() : _isolate(); },
        });
        return opts;
    },
    async setup() {
        _patch();
        console.log("[MEC.IsolateSubgraph] Loaded.");
    },
});
