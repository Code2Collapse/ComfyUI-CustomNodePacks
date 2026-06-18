/**
 * wan_director_compact.js — collapse WanDirectorC2C to a usable default height.
 * Hides gated advanced widgets until their enable_* toggle is true; caps node height.
 */
import { app } from "../../scripts/app.js";
import {
    capWdNode,
    findWdWidget,
    getWdAdvancedOpen,
    installWanDirectorPrototype,
    setWdAdvancedOpen,
    wdHideWidget,
    wdShowWidget,
} from "./_wan_director_ui.js";

/** enable_* parent → child widget names hidden when parent is false. */
const GATED = {
    enable_prompt_relay: ["prompt_relay_epsilon"],
    enable_dynamic_cfg: ["guidance_rescale_phi", "pag_scale"],
    enable_phase_shift: ["phase_shift_pct"],
    enable_multi_clip: ["structure_prompt", "detail_prompt"],
    enable_nag: ["nag_scale"],
    enable_asymflow: ["asymflow_shift"],
    enable_slg: ["slg_layers", "slg_scale"],
    enable_feta: ["feta_scale"],
    enable_riflex: ["riflex_k"],
};

/** Always hidden on create — managed by timeline/player DOM or JSON blobs. */
const ALWAYS_HIDDEN = new Set([
    "timeline_data", "local_prompts", "negative_prompts",
    "segment_lengths", "guide_strength",
]);

/** Collapsed until user expands "Advanced" (optional widgets). */
const ADVANCED_COLLAPSE = new Set([
    "cache_type", "cache_threshold",
    "custom_width", "custom_height", "resize_method",
    "audio_target",
    "everanimate_stage", "everanimate_num_chunks", "everanimate_overlap_frames",
    "everanimate_lora_strength", "everanimate_anchor_strategy",
    "duration_seconds", "display_mode",
    "cfg_high_noise", "cfg_low_noise", "ref_strength",
    "vae_fp32_decode",
]);

/** Quality-stack toggles — hidden until Advanced (compact node chrome). */
const QUALITY_COLLAPSE = new Set([
    "enable_prompt_relay", "prompt_relay_epsilon",
    "enable_dynamic_cfg", "guidance_rescale_phi", "pag_scale",
    "enable_phase_shift", "phase_shift_pct",
    "enable_multi_clip", "structure_prompt", "detail_prompt",
    "enable_nag", "nag_scale",
    "enable_asymflow", "asymflow_shift",
    "enable_slg", "slg_layers", "slg_scale",
    "enable_feta", "feta_scale",
    "enable_riflex", "riflex_k",
]);

function hideW(w) {
    wdHideWidget(w);
    w._wd_compact_hidden = true;
}

function showW(w) {
    wdShowWidget(w);
    w._wd_compact_hidden = false;
}

function applyGates(node) {
    for (const [gate, children] of Object.entries(GATED)) {
        const gw = findWdWidget(node, gate);
        const on = gw ? !!gw.value : false;
        for (const cn of children) {
            const cw = findWdWidget(node, cn);
            if (!cw) continue;
            if (on) showW(cw);
            else hideW(cw);
        }
    }
    for (const cn of QUALITY_COLLAPSE) {
        const cw = findWdWidget(node, cn);
        if (!cw) continue;
        if (getWdAdvancedOpen(node)) showW(cw);
        else hideW(cw);
    }
    if (!getWdAdvancedOpen(node)) {
        for (const cn of ADVANCED_COLLAPSE) {
            const cw = findWdWidget(node, cn);
            if (cw) hideW(cw);
        }
    } else {
        for (const cn of ADVANCED_COLLAPSE) {
            const cw = findWdWidget(node, cn);
            if (cw) showW(cw);
        }
    }
}

function wireGate(node, gateName) {
    const gw = findWdWidget(node, gateName);
    if (!gw || gw._wd_compact_wired) return;
    gw._wd_compact_wired = true;
    const orig = gw.callback;
    gw.callback = function (v, ...rest) {
        const ret = orig ? orig.call(this, v, ...rest) : undefined;
        applyGates(node);
        capWdNode(node);
        return ret;
    };
}

function compactNode(node) {
    for (const w of node.widgets || []) {
        if (ALWAYS_HIDDEN.has(w.name)) hideW(w);
    }
    for (const gate of Object.keys(GATED)) wireGate(node, gate);

    if (!findWdWidget(node, "show_advanced")) {
        const btn = node.addWidget("button", "show_advanced", "Show advanced ▼", () => {
            setWdAdvancedOpen(node, !getWdAdvancedOpen(node));
            btn.value = getWdAdvancedOpen(node) ? "Hide advanced ▲" : "Show advanced ▼";
            applyGates(node);
            capWdNode(node);
        });
    }

    applyGates(node);
    capWdNode(node);
}

app.registerExtension({
    name: "C2C.WanDirector.Compact",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "WanDirectorC2C") return;
        installWanDirectorPrototype(nodeType);

        const _onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = _onCreated?.apply(this, arguments);
            const node = this;
            queueMicrotask(() => compactNode(node));
            setTimeout(() => compactNode(node), 0);
            setTimeout(() => compactNode(node), 300);
            setTimeout(() => compactNode(node), 1200);
            return r;
        };

        const _onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = _onConfigure?.apply(this, arguments);
            queueMicrotask(() => compactNode(this));
            return r;
        };
    },
});
