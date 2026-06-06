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
import {
    capWdNode,
    findWdWidget,
    installWanDirectorPrototype,
    wdHideWidget,
    wdShowWidget,
} from "./_wan_director_ui.js";

// Mirror of nodes/wan_director/director_node.py:VARIANT_TABLE flags
// (only the ones that drive UI visibility).
const VARIANT_FLAGS = {
    "wan2.1_t2v":      { dual_cfg: false, ref_image: false, needs_image: false, everanimate: false },
    "wan2.1_i2v":      { dual_cfg: false, ref_image: false, needs_image: true,  everanimate: false },
    "wan2.2_t2v":      { dual_cfg: true,  ref_image: false, needs_image: false, everanimate: false },
    "wan2.2_i2v":      { dual_cfg: true,  ref_image: false, needs_image: true,  everanimate: false },
    "wan_fun_inp":     { dual_cfg: false, ref_image: false, needs_image: true,  everanimate: false },
    "wan_fun_control": { dual_cfg: false, ref_image: false, needs_image: false, everanimate: false },
    "wan_animate":     { dual_cfg: false, ref_image: true,  needs_image: false, everanimate: false },
    "wan2.2_animate_everanimate": { dual_cfg: true,  ref_image: true, needs_image: false, everanimate: true },
};

const EVERANIMATE_WIDGETS = [
    "everanimate_stage",
    "everanimate_num_chunks",
    "everanimate_overlap_frames",
    "everanimate_lora_strength",
    "everanimate_anchor_strategy",
];

function hideWidget(w) {
    wdHideWidget(w);
    w._wd_hidden = true;
}

function showWidget(w) {
    wdShowWidget(w);
    w._wd_hidden = false;
}

function applyVariant(node, variant) {
    const flags = VARIANT_FLAGS[variant];
    if (!flags) return;
    const wCfgHi = findWdWidget(node, "cfg_high_noise");
    const wCfgLo = findWdWidget(node, "cfg_low_noise");
    const wRef   = findWdWidget(node, "ref_strength");

    // Rename cfg_high_noise → "cfg" label for single-cfg variants.
    if (wCfgHi) {
        if (!wCfgHi._wd_orig_label) wCfgHi._wd_orig_label = wCfgHi.label || wCfgHi.name;
        wCfgHi.label = flags.dual_cfg ? "cfg_high_noise" : "cfg";
    }

    if (flags.dual_cfg) showWidget(wCfgLo); else hideWidget(wCfgLo);
    if (flags.ref_image) showWidget(wRef);  else hideWidget(wRef);

    // needs_image: hide VAE-related inputs for t2v / non-image variants.
    const IMAGE_WIDGETS = ["vae_fp32_decode"];
    for (const wn of IMAGE_WIDGETS) {
        const w = findWdWidget(node, wn);
        if (flags.needs_image) showWidget(w); else hideWidget(w);
    }

    // EverAnimate widgets: only shown for the everanimate variant.
    for (const wname of EVERANIMATE_WIDGETS) {
        const w = findWdWidget(node, wname);
        if (flags.everanimate) showWidget(w); else hideWidget(w);
    }

    capWdNode(node);
    app.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "C2C.WanDirector.VariantGate",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "WanDirectorC2C") return;
        installWanDirectorPrototype(nodeType);

        const _onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = _onCreated?.apply(this, arguments);
            const node = this;
            const wVariant = findWdWidget(node, "model_variant");
            if (!wVariant) return r;

            // Hook callback (preserve any existing one — linked_values may chain).
            const orig = wVariant.callback;
            wVariant.callback = function (v, ...rest) {
                const ret = orig ? orig.call(this, v, ...rest) : undefined;
                try { applyVariant(node, v); } catch (e) { console.warn("[WanDirector] variant gate:", e); }
                return ret;
            };

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
                const w = findWdWidget(node, "model_variant");
                if (w) { try { applyVariant(node, w.value); } catch {} }
            });
            return r;
        };
    },
});
