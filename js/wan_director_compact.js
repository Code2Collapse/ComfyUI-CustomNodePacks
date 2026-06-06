/**
 * wan_director_compact.js — collapse WanDirectorC2C to a usable default height.
 * Hides gated advanced widgets until their enable_* toggle is true; caps node height.
 */
import { app } from "../../scripts/app.js";

const WD_MAX_H = () => Math.max(520, Math.floor((window.innerHeight || 1080) * 0.62));
const WD_DEFAULT_W = 640;

function _writeNodeSize(node, w, h) {
    if (!node) return;
    if (!node.size) node.size = [w, h];
    else {
        node.size[0] = w;
        node.size[1] = h;
    }
}

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
]);

let _advancedOpen = false;

function hideW(w) {
    if (!w || w._wd_compact_hidden) return;
    w._wd_compact_orig_cs = w.computeSize;
    w._wd_compact_orig_type = w.type;
    w.hidden = true;
    w.type = "hidden";
    w.computeSize = () => [0, -4];
    w._wd_compact_hidden = true;
}

function showW(w) {
    if (!w || !w._wd_compact_hidden) return;
    w.hidden = false;
    if (w._wd_compact_orig_type) w.type = w._wd_compact_orig_type;
    if (w._wd_compact_orig_cs) w.computeSize = w._wd_compact_orig_cs;
    else delete w.computeSize;
    w._wd_compact_hidden = false;
}

function findW(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

function applyGates(node) {
    for (const [gate, children] of Object.entries(GATED)) {
        const gw = findW(node, gate);
        const on = gw ? !!gw.value : false;
        for (const cn of children) {
            const cw = findW(node, cn);
            if (!cw) continue;
            if (on) showW(cw);
            else hideW(cw);
        }
    }
    if (!_advancedOpen) {
        for (const cn of ADVANCED_COLLAPSE) {
            const cw = findW(node, cn);
            if (cw) hideW(cw);
        }
    }
}

function capNode(node) {
    const cap = WD_MAX_H();
    const w = Math.max(node.size?.[0] || WD_DEFAULT_W, WD_DEFAULT_W);
    let h = node.size?.[1] || cap;
    if (typeof node.computeSize === "function") {
        try {
            const cs = node.computeSize();
            h = Math.min(cs[1] || h, cap);
        } catch (_) {}
    }
    h = Math.min(h, cap);
    _writeNodeSize(node, w, h);
    node.setDirtyCanvas?.(true, true);
}

function wireGate(node, gateName) {
    const gw = findW(node, gateName);
    if (!gw || gw._wd_compact_wired) return;
    gw._wd_compact_wired = true;
    const orig = gw.callback;
    gw.callback = function (v, ...rest) {
        const ret = orig ? orig.call(this, v, ...rest) : undefined;
        applyGates(node);
        capNode(node);
        return ret;
    };
}

function compactNode(node) {
    for (const w of node.widgets || []) {
        if (ALWAYS_HIDDEN.has(w.name)) hideW(w);
    }
    for (const gate of Object.keys(GATED)) wireGate(node, gate);

    if (!findW(node, "show_advanced")) {
        const btn = node.addWidget("button", "show_advanced", "Show advanced ▼", () => {
            _advancedOpen = !_advancedOpen;
            btn.value = _advancedOpen ? "Hide advanced ▲" : "Show advanced ▼";
            applyGates(node);
            capNode(node);
        });
    }

    applyGates(node);
    capNode(node);
}

app.registerExtension({
    name: "C2C.WanDirector.Compact",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "WanDirectorC2C") return;

        const _origSetSize = nodeType.prototype.setSize;
        if (typeof _origSetSize === "function") {
            nodeType.prototype.setSize = function (size) {
                const cap = WD_MAX_H();
                return _origSetSize.call(this, [
                    Math.max(size?.[0] || WD_DEFAULT_W, WD_DEFAULT_W),
                    Math.min(size?.[1] || cap, cap),
                ]);
            };
        }

        const _origCS = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (outW) {
            const cap = WD_MAX_H();
            const sz = _origCS ? _origCS.call(this, outW) : [this.size?.[0] || WD_DEFAULT_W, cap];
            return [Math.max(sz[0] || 0, WD_DEFAULT_W), Math.min(sz[1] || cap, cap)];
        };

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
