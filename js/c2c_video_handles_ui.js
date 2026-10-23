/**
 * Built-in handles on Load Video (C2C) / Load Video Path (C2C) (L9.40): show only the handle widgets the chosen mode
 * uses - off: none; fit: target, side, grid; add: frames, side, grid; custom step / plus only for the custom grid.
 * The count the run used is in the node's info output and in video_info, where Save Video (C2C) trims it.
 */
import { app } from "../../scripts/app.js";
import { setHidden, vueSyncNodeWidgets } from "./_widget_visibility.js";

const NODES = new Set(["LoadVideoC2C", "LoadVideoPathC2C"]);
const CUSTOM = "custom (step / plus)";
const HANDLE_WIDGETS = ["handle_frames", "handle_target", "handle_side", "handle_grid", "handle_custom_step",
                        "handle_custom_plus"];

function apply(node) {
    const W = (n) => node.widgets?.find((w) => w.name === n);
    const mode = W("handles")?.value ?? "off";
    const grid = W("handle_grid")?.value;
    const show = {
        handle_frames: mode === "add",
        handle_target: mode === "fit",
        handle_side: mode !== "off",
        handle_grid: mode !== "off",
        handle_custom_step: mode !== "off" && grid === CUSTOM,
        handle_custom_plus: mode !== "off" && grid === CUSTOM,
    };
    let changed = false;
    for (const n of HANDLE_WIDGETS) {
        const w = W(n);
        if (!w) continue;
        const hide = !show[n];
        if (!!w.__mec_hidden !== hide) changed = true;
        setHidden(w, hide);
    }
    if (!changed) return;
    vueSyncNodeWidgets(node);
    const sz = node.computeSize?.();
    if (sz) node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.setDirtyCanvas?.(true, true);
}

// short labels: the grid names are long ("MiniMax H3 (17n+5)"), and a long value clips its widget's label
const LABELS = { handle_frames: "handle frames", handle_target: "target frames", handle_side: "handle side",
                 handle_grid: "grid", handle_custom_step: "grid step", handle_custom_plus: "grid plus" };

function label(node) {
    for (const [n, text] of Object.entries(LABELS)) {
        const w = node.widgets?.find((x) => x.name === n);
        if (w && !w.label) w.label = text;
    }
}

function hook(node, name) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w || w.__c2cHandlesHooked) return;
    w.__c2cHandlesHooked = true;
    const orig = w.callback;
    w.callback = function (...a) {
        const r = orig?.apply(this, a);
        apply(node);
        return r;
    };
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
                hook(this, "handles");
                hook(this, "handle_grid");
                label(this);
                setTimeout(() => apply(this), 0);
                return r;
            };
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (...args) {
                const r = onConfigure?.apply(this, args);
                setTimeout(() => apply(this), 0);
                return r;
            };
        },
    });
}
