// VAEMergeMEC: per-block sliders only visible when use_blocks=True AND auto_alpha=False.
import { app } from "../../scripts/app.js";
import { setHidden } from "./_widget_visibility.js";

const BLOCKS = [
    "block_conv_in", "block_conv_out", "block_norm_out",
    "block_0", "block_1", "block_2", "block_3", "block_mid",
];

function applyVisibility(node) {
    const get = (n) => node.widgets?.find(w => w.name === n);
    const useBlocks = !!get("use_blocks")?.value;
    const autoAlpha = !!get("auto_alpha")?.value;
    const showBlocks = useBlocks && !autoAlpha;
    for (const name of BLOCKS) {
        const w = get(name);
        if (w) setHidden(w, !showBlocks);
    }
    const sz = node.computeSize();
    node.size[0] = Math.max(node.size[0], sz[0]);
    node.size[1] = sz[1];
    node.setDirtyCanvas(true, true);
}

function hookWidget(node, name) {
    const w = node.widgets?.find(x => x.name === name);
    if (!w) return;
    const orig = w.callback;
    w.callback = (v, ...rest) => {
        const r = orig?.call(w, v, ...rest);
        applyVisibility(node);
        return r;
    };
}

app.registerExtension({
    name: "MEC.VAEMerge.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VAEMergeMEC") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            for (const name of ["use_blocks", "auto_alpha"]) hookWidget(this, name);
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => applyVisibility(this), 0);
            return r;
        };
    },
});
