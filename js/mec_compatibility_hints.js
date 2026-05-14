/**
 * mec_compatibility_hints.js — Phase 6: Connection Compatibility Hints
 *
 * While the user is dragging a link out of an output slot, all input slots
 * across the graph that accept the source type are highlighted with a pulsing
 * green ring. Incompatible slots get a faded red ring. Helps new users learn
 * the data-type system at a glance.
 *
 * Type matching uses LiteGraph's compatibility rules:
 *  - Exact string match (case-insensitive).
 *  - Wildcard "*" matches anything.
 *  - Comma-separated lists ("IMAGE,LATENT") match any.
 *
 * Setting:
 *   mec.compatibility_hints.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

let _enabled = true;

function _normalizeType(t) {
    if (t === null || t === undefined) return [];
    if (typeof t !== "string") {
        if (Array.isArray(t)) return t.map(s => String(s).toUpperCase().trim());
        return [String(t).toUpperCase().trim()];
    }
    return t.split(",").map(s => s.toUpperCase().trim()).filter(Boolean);
}

function _typesCompatible(srcType, dstType) {
    const a = _normalizeType(srcType);
    const b = _normalizeType(dstType);
    if (a.includes("*") || b.includes("*")) return true;
    for (const x of a) for (const y of b) {
        if (x === y) return true;
    }
    return false;
}

function _patchCanvas() {
    if (!LGraphCanvas || LGraphCanvas.prototype._mecHintsPatched) return;

    const origDraw = LGraphCanvas.prototype.drawNode;
    if (typeof origDraw !== "function") return;

    LGraphCanvas.prototype.drawNode = function (node, ctx) {
        const result = origDraw.call(this, node, ctx);

        if (!_enabled) return result;
        const ci = this.connecting_node ? this : null;
        // LiteGraph exposes the connection-in-progress through:
        //   this.connecting_node, this.connecting_output, this.connecting_slot
        // (older versions use connecting_pos / connecting_input).
        const src      = this.connecting_node;
        const srcOut   = this.connecting_output;
        if (!src || !srcOut || src === node) return result;

        const srcType = srcOut.type;
        const inputs = node.inputs || [];
        if (inputs.length === 0) return result;

        for (let i = 0; i < inputs.length; i++) {
            const inp = inputs[i];
            if (!inp) continue;
            const compatible = _typesCompatible(srcType, inp.type);

            // Compute slot position in node-local coords, then transform.
            const pos = node.getConnectionPos
                ? node.getConnectionPos(true, i)
                : [node.pos[0], node.pos[1] + 10 + i * LiteGraph.NODE_SLOT_HEIGHT];

            ctx.save();
            // ctx is already in graph coords; pos is graph-coords from getConnectionPos.
            // Convert to ctx-local by subtracting node origin (because drawNode
            // translated ctx to node).
            const localX = pos[0] - node.pos[0];
            const localY = pos[1] - node.pos[1];

            const radius = compatible ? 9 : 7;
            ctx.beginPath();
            ctx.arc(localX, localY, radius, 0, Math.PI * 2);
            ctx.lineWidth = 2;
            if (compatible) {
                const t = (Date.now() % 800) / 800;          // 0..1 pulse
                const alpha = 0.5 + 0.5 * Math.sin(t * Math.PI * 2);
                ctx.strokeStyle = `rgba(166, 227, 161, ${alpha.toFixed(2)})`;
                ctx.shadowColor = "rgba(166, 227, 161, 0.8)";
                ctx.shadowBlur = 6;
            } else {
                ctx.strokeStyle = "rgba(243, 139, 168, 0.35)";
            }
            ctx.stroke();
            ctx.restore();
        }
        void ci;  // reserved for future
        return result;
    };

    LGraphCanvas.prototype._mecHintsPatched = true;
}

function _setupRedrawWhileConnecting() {
    // While a link drag is in progress we want continuous redraws so the
    // pulse animates. Hook onto the global animation loop already running
    // inside LiteGraph by triggering dirty_canvas every frame while
    // connecting_node is set.
    const tick = () => {
        try {
            const canvas = app.canvas;
            if (canvas && canvas.connecting_node && canvas.connecting_output && _enabled) {
                canvas.setDirty(true, true);
            }
        } catch { /* swallow */ }
        requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}

app.registerExtension({
    name: "MEC.CompatibilityHints",
    settings: [
        {
            id: "mec.compatibility_hints.enabled",
            name: "Compatibility Hints: highlight matching slots",
            tooltip: "Pulse green rings around input slots that accept the link being dragged.",
            type: "boolean",
            defaultValue: true,
            onChange: (v) => { _enabled = !!v; },
        },
    ],
    async setup() {
        try {
            _enabled = app.ui.settings.getSettingValue("mec.compatibility_hints.enabled", true);
        } catch { _enabled = true; }

        _patchCanvas();
        _setupRedrawWhileConnecting();
        console.log("[MEC.CompatibilityHints] Loaded — drag a link to see compatible slots glow.");
    },
});
