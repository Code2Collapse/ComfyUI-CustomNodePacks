import { app } from "../../scripts/app.js";
import { setWidgetVisible } from "./_widget_visibility.js";

/**
 * InpaintCropProMEC — conditional widget visibility.
 *
 * Three boolean master toggles gate their related parameters. When the
 * master is OFF the dependent widgets are hidden (they keep their values
 * and continue to serialize, so backend defaults remain consistent).
 * When ON, they reveal again.
 *
 *   preresize                      → preresize_mode + min/max width/height
 *   extend_for_outpainting         → extend_up/down/left/right_factor
 *   output_resize_to_target_size   → output_target_width/height + output_padding
 */

const GROUPS = {
    preresize: [
        "preresize_mode",
        "preresize_min_width",
        "preresize_min_height",
        "preresize_max_width",
        "preresize_max_height",
    ],
    extend_for_outpainting: [
        "extend_up_factor",
        "extend_down_factor",
        "extend_left_factor",
        "extend_right_factor",
    ],
    output_resize_to_target_size: [
        "output_target_width",
        "output_target_height",
        "output_padding",
    ],
};

// Combo-master groups: dependents revealed when master.value === triggerValue.
const COMBO_GROUPS = {
    stitch_blend_mode: {
        triggerValue: "video_stable",
        dependents: [
            "video_stable_temporal_sigma",
            "video_stable_dilate_px",
            "video_stable_blur_sigma",
        ],
    },
};

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function applyGroup(node, masterName, value) {
    const dependents = GROUPS[masterName];
    if (!dependents) return;
    const visible = !!value;
    for (const depName of dependents) {
        const w = findWidget(node, depName);
        if (w) setWidgetVisible(w, visible);
    }
}

function applyComboGroup(node, masterName, value) {
    const cfg = COMBO_GROUPS[masterName];
    if (!cfg) return;
    const visible = value === cfg.triggerValue;
    for (const depName of cfg.dependents) {
        const w = findWidget(node, depName);
        if (w) setWidgetVisible(w, visible);
    }
}

function applyAll(node) {
    for (const masterName of Object.keys(GROUPS)) {
        const master = findWidget(node, masterName);
        if (master) applyGroup(node, masterName, master.value);
    }
    for (const masterName of Object.keys(COMBO_GROUPS)) {
        const master = findWidget(node, masterName);
        if (master) applyComboGroup(node, masterName, master.value);
    }
    // Force re-layout
    const sz = node.computeSize();
    node.size[1] = sz[1];
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "MEC.InpaintCropPro",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "InpaintCropProMEC") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);

            for (const masterName of Object.keys(GROUPS)) {
                const master = findWidget(this, masterName);
                if (!master) continue;
                const origCallback = master.callback;
                master.callback = (value) => {
                    const r = origCallback?.call(master, value);
                    applyGroup(this, masterName, value);
                    const sz = this.computeSize();
                    this.size[1] = sz[1];
                    this.setDirtyCanvas(true, true);
                    return r;
                };
            }
            for (const masterName of Object.keys(COMBO_GROUPS)) {
                const master = findWidget(this, masterName);
                if (!master) continue;
                const origCallback = master.callback;
                master.callback = (value) => {
                    const r = origCallback?.call(master, value);
                    applyComboGroup(this, masterName, value);
                    const sz = this.computeSize();
                    this.size[1] = sz[1];
                    this.setDirtyCanvas(true, true);
                    return r;
                };
            }
            applyAll(this);
            return ret;
        };

        // Re-apply after a workflow is loaded (widget values restored).
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const ret = onConfigure?.apply(this, arguments);
            applyAll(this);
            return ret;
        };
    },
});
