/**
 * wan_director_variant_gate.js — show/hide WanDirectorC2C widgets based
 * on the selected `model_variant`.
 *
 * The Python side accepts ALL widgets unconditionally (it picks the
 * relevant ones based on VARIANT_TABLE), but the UI was static, so
 * users were confused by irrelevant knobs (e.g. cfg_low_noise visible
 * for wan2.1_t2v, which doesn't use dual-cfg).
 *
 * Apache-2.0 © Code2Collapse.
 */

import { app } from "../../scripts/app.js";

// Mirror of nodes/wan_director/director_node.py:VARIANT_TABLE flags
// (only the ones that drive UI visibility).
const VARIANT_FLAGS = {
    "wan2.1_t2v":      { dual_cfg: false, ref_image: false, needs_image: false, everanimate_compatible: false },
    "wan2.1_i2v":      { dual_cfg: false, ref_image: false, needs_image: true,  everanimate_compatible: false },
    "wan2.2_t2v":      { dual_cfg: true,  ref_image: false, needs_image: false, everanimate_compatible: false },
    "wan2.2_i2v":      { dual_cfg: true,  ref_image: false, needs_image: true,  everanimate_compatible: false },
    "wan_fun_inp":     { dual_cfg: false, ref_image: false, needs_image: true,  everanimate_compatible: false },
    "wan_fun_control": { dual_cfg: false, ref_image: false, needs_image: false, everanimate_compatible: false },
    "wan_animate":     { dual_cfg: false, ref_image: true,  needs_image: false, everanimate_compatible: true  },
};

const EVERANIMATE_WIDGETS = [
    "everanimate_stage",
    "everanimate_num_chunks",
    "everanimate_overlap_frames",
    "everanimate_lora_strength",
    "everanimate_anchor_strategy",
];

function hideWidget(w) {
    if (!w || w._wd_hidden) return;
    w._wd_orig_computeSize = w.computeSize;
    w._wd_orig_type = w.type;
    w.hidden = true;
    if (!w.options) w.options = {};
    w.options.hidden = true;
    w.type = "hidden_" + (w._wd_orig_type || "");
    w.computeSize = () => [0, -4];
    w._wd_hidden = true;
}

function showWidget(w) {
    if (!w || !w._wd_hidden) return;
    w.hidden = false;
    if (w.options) w.options.hidden = false;
    if (w._wd_orig_type !== undefined) w.type = w._wd_orig_type;
    if (w._wd_orig_computeSize) w.computeSize = w._wd_orig_computeSize;
    else delete w.computeSize;
    w._wd_hidden = false;
}

function findW(node, name) {
    return (node.widgets || []).find(w => w.name === name);
}

function applyVariant(node, variant) {
    const flags = VARIANT_FLAGS[variant];
    if (!flags) return;
    const wCfgHi = findW(node, "cfg_high_noise");
    const wCfgLo = findW(node, "cfg_low_noise");
    const wRef   = findW(node, "ref_strength");

    // Rename cfg_high_noise → "cfg" label for single-cfg variants.
    if (wCfgHi) {
        if (!wCfgHi._wd_orig_label) wCfgHi._wd_orig_label = wCfgHi.label || wCfgHi.name;
        wCfgHi.label = flags.dual_cfg ? "cfg_high_noise" : "cfg";
    }

    if (flags.dual_cfg) showWidget(wCfgLo); else hideWidget(wCfgLo);
    if (flags.ref_image) showWidget(wRef);  else hideWidget(wRef);

    // EverAnimate toggle is visible only for compatible variants (wan_animate).
    // The 5 EA settings are visible only when toggle is also ON.
    const wEnableEA = findW(node, "enable_everanimate");
    if (flags.everanimate_compatible) showWidget(wEnableEA); else hideWidget(wEnableEA);
    const eaOn = flags.everanimate_compatible && !!(wEnableEA && wEnableEA.value);
    for (const wname of EVERANIMATE_WIDGETS) {
        const w = findW(node, wname);
        if (eaOn) showWidget(w); else hideWidget(w);
    }

    // Recompute node size and redraw.
    if (typeof node.setSize === "function" && node.computeSize) {
        const min = node.computeSize();
        const cur = node.size || [0, 0];
        node.setSize([Math.max(cur[0], min[0]), Math.max(cur[1], min[1])]);
    }
    app.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "C2C.WanDirector.VariantGate",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "WanDirectorC2C") return;

        const _onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = _onCreated?.apply(this, arguments);
            const node = this;
            const wVariant = findW(node, "model_variant");
            if (!wVariant) return r;

            // Hook callback (preserve any existing one — linked_values may chain).
            const orig = wVariant.callback;
            wVariant.callback = function (v, ...rest) {
                const ret = orig ? orig.call(this, v, ...rest) : undefined;
                try { applyVariant(node, v); } catch (e) { console.warn("[WanDirector] variant gate:", e); }
                return ret;
            };

            // Hook the EverAnimate toggle so flipping it reactively shows/hides
            // the 5 EA setting widgets without needing to re-select the variant.
            const wEnableEA = findW(node, "enable_everanimate");
            if (wEnableEA) {
                const origEA = wEnableEA.callback;
                wEnableEA.callback = function (v, ...rest) {
                    const ret = origEA ? origEA.call(this, v, ...rest) : undefined;
                    try { applyVariant(node, wVariant.value); } catch (e) { console.warn("[WanDirector] EA toggle:", e); }
                    return ret;
                };
            }

            // Apply once on creation (after widget order is finalised).
            queueMicrotask(() => {
                try { applyVariant(node, wVariant.value); } catch (e) { console.warn("[WanDirector] variant gate init:", e); }
            });
            return r;
        };

        // Also re-apply when graph is configured (load from JSON).
        const _onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = _onConfigure?.apply(this, arguments);
            const node = this;
            queueMicrotask(() => {
                const w = findW(node, "model_variant");
                if (w) { try { applyVariant(node, w.value); } catch {} }
            });
            return r;
        };
    },
});
