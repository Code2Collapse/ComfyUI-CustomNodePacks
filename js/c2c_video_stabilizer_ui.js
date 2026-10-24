/**
 * Video Stabilizer (L9.39): show only the widgets the current choice uses - the manual settings (framing,
 * transform, camera lock, strength, smooth, keep FOV) only for preset = manual, and the RAFT settings (iterations,
 * fp16, allow download) only for method = raft_flow; with motion_meta wired (restoring the camera motion) only
 * padding_color is used, so the rest hides. Hidden widgets keep their values and still serialise.
 */
import { app } from "../../scripts/app.js";
import { setHidden, vueFitNodeHeight, vueSyncNodeWidgets } from "./_widget_visibility.js";

const NODE = "VideoStabilizerMEC";
const MANUAL = ["framing_mode", "transform_mode", "camera_lock", "strength", "smooth", "keep_fov"];
const RAFT = ["raft_iters", "use_half", "allow_download"];
const STABILISE = ["method", "preset", "frame_rate"];

function apply(node) {
    const W = (n) => node.widgets?.find((w) => w.name === n);
    const restoring = node.inputs?.some((i) => i.name === "motion_meta" && i.link != null) ?? false;
    const manual = !restoring && (W("preset")?.value ?? "manual") === "manual";
    const raft = !restoring && (W("method")?.value ?? "auto") === "raft_flow";
    let changed = false;
    for (const [names, show] of [[MANUAL, manual], [RAFT, raft], [STABILISE, !restoring]]) {
        for (const n of names) {
            const w = W(n);
            if (!w) continue;
            if (!!w.__mec_hidden !== !show) changed = true;
            setHidden(w, !show);
        }
    }
    if (!changed) return;
    vueSyncNodeWidgets(node);
    vueFitNodeHeight(node);
    const sz = node.computeSize?.();
    if (sz) node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
    node.setDirtyCanvas?.(true, true);
}

function hook(node, name) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w || w.__c2cStabHooked) return;
    w.__c2cStabHooked = true;
    const orig = w.callback;
    w.callback = function (...a) {
        const r = orig?.apply(this, a);
        apply(node);
        return r;
    };
}

if (!globalThis.__c2cVideoStabilizerUiExt) {
    globalThis.__c2cVideoStabilizerUiExt = true;
    app.registerExtension({
        name: "C2C.VideoStabilizerUI",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== NODE) return;
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...args) {
                const r = onCreated?.apply(this, args);
                hook(this, "method");
                hook(this, "preset");
                setTimeout(() => apply(this), 0);
                return r;
            };
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (...args) {
                const r = onConnectionsChange?.apply(this, args);
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
