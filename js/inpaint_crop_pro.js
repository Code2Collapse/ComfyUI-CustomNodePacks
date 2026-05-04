// InpaintCropProMEC: conditional widget visibility based on size_mode, aspect_ratio,
// downscale_factor, and video_stab_strength. Pattern: KJNodes-style hide via computeSize → [0,-4].
import { app } from "../../scripts/app.js";

const ALWAYS = new Set([
    "context_expand", "inpaint_mask_mode", "stitch_blend_mode", "blend_radius",
    "size_mode", "padding_multiple", "fill_masked_area", "downscale_factor",
    "mask_blur", "mask_grow", "mask_invert", "mask_fill_holes", "mask_hipass_filter",
    "aspect_ratio", "video_stable_crop", "video_stab_strength", "mask_temporal_smooth",
]);

function setHidden(w, hidden) {
    if (hidden) {
        if (!w.__origComputeSize) {
            w.__origComputeSize = w.computeSize;
            w.__origType = w.type;
        }
        w.computeSize = () => [0, -4];
        w.hidden = true;
        w.type = "hidden";
    } else {
        if (w.__origComputeSize) {
            w.computeSize = w.__origComputeSize;
            delete w.__origComputeSize;
        }
        w.hidden = false;
        w.type = w.__origType || w.type;
    }
}

function applyVisibility(node) {
    const get = (name) => node.widgets?.find(w => w.name === name);
    const sizeMode    = get("size_mode")?.value;
    const aspect      = get("aspect_ratio")?.value;
    const downscale   = get("downscale_factor")?.value;
    const stabStrength = get("video_stab_strength")?.value;

    const showForced  = sizeMode === "forced_size";
    const showRanged  = sizeMode === "ranged_size";
    const showCustom  = aspect === "custom";
    const showDown    = (typeof downscale === "number") ? downscale < 0.999 : false;
    const showStab    = (typeof stabStrength === "number") ? stabStrength > 0.0001 : false;

    const conditional = {
        forced_width:       showForced,
        forced_height:      showForced,
        min_size:           showRanged,
        max_size:           showRanged,
        custom_aspect_w:    showCustom,
        custom_aspect_h:    showCustom,
        downscale_method:   showDown,
        upscale_method:     showDown,
        video_stab_fps:     showStab,
        video_stab_padding: showStab,
    };

    for (const w of node.widgets) {
        if (ALWAYS.has(w.name)) { setHidden(w, false); continue; }
        if (w.name in conditional) { setHidden(w, !conditional[w.name]); continue; }
        // Unknown widgets — keep visible
        setHidden(w, false);
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
    name: "MEC.InpaintCropProMEC.ConditionalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "InpaintCropProMEC") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            for (const name of ["size_mode", "aspect_ratio", "downscale_factor", "video_stab_strength"]) {
                hookWidget(this, name);
            }
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
