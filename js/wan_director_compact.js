/**
 * wan_director_compact.js — collapse WanDirectorC2C to a usable default height.
 * Hides gated advanced widgets until their enable_* toggle is true; caps node height.
 */
import { app } from "/scripts/app.js";
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
    // The advanced quality stack was split out to the WanDirectorExtraArgs node,
    // so everything that REMAINS on WanDirector is now visible at all times
    // (user 2026-06-29: "show-advanced params must be visible at all times,
    // remove the advanced button"). Only the timeline / JSON state blobs that the
    // timeline editor owns stay hidden.
    for (const w of node.widgets || []) {
        if (ALWAYS_HIDDEN.has(w.name)) hideW(w);
        else if (w._wd_compact_hidden) showW(w);
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
    // The "Show advanced" button was removed — its remaining params are always
    // visible now, and the quality stack lives on the WanDirectorExtraArgs node.
    // Remove any stale button from graphs saved before this change.
    const stale = (node.widgets || []).find(w => w.name === "show_advanced");
    if (stale) {
        const i = node.widgets.indexOf(stale);
        if (i >= 0) node.widgets.splice(i, 1);
    }

    applyGates(node);
    capWdNode(node);
}

// Guard against double-registration when CustomNodePacks ships the same extension.
if (!(app.extensions || []).some(e => e?.name === "C2C.WanDirector.Compact")) app.registerExtension({
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
