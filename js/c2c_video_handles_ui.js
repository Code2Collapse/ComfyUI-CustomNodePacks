/**
 * Built-in handles on Load Video (C2C) (L9.40): short labels for the handle widgets (the grid names are long, and a
 * long value clips its widget's label). WHICH handle widgets show (off: none; fit: target, side, grid; add: frames,
 * side, grid; custom step / plus only for the custom grid; extra handles with fit and add) is declared next to the
 * inputs in Python and applied by js/_c2c_node_layout.js (A28: one visibility owner per node).
 * The count the run used is in the node's info output and in video_info, where Save Video (C2C) trims it.
 */
import { app } from "../../scripts/app.js";

const NODES = new Set(["LoadVideoC2C"]);

// short labels: the grid names are long ("MiniMax H3 (17n+5)"), and a long value clips its widget's label
const LABELS = { handle_frames: "handle frames", handle_target: "target frames", handle_side: "handle side",
                 handle_grid: "grid", handle_custom_step: "grid step", handle_custom_plus: "grid plus",
                 extra_handles: "extra handles (model steps)" };

function label(node) {
    for (const [n, text] of Object.entries(LABELS)) {
        const w = node.widgets?.find((x) => x.name === n);
        // the frontend pre-fills label = name: "!w.label" alone never applied the short labels (L9.52, measured)
        if (w && (!w.label || w.label === w.name)) w.label = text;
    }
}

if (!globalThis.__c2cVideoHandlesUiExt) {
    globalThis.__c2cVideoHandlesUiExt = true;
    app.registerExtension({
        name: "C2C.VideoLoaderHandles",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (!NODES.has(nodeData.name)) return;
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...args) {
                const r = onCreated?.apply(this, args);
                label(this);
                return r;
            };
        },
    });
}
