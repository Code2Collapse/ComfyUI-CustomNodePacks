/**
 * Fold away the MaskOps widgets the chosen models do not read.
 *
 * MaskOpsMEC has 68 parameters and most of them belong to one backend or one
 * optional feature. All 68 sit on the node looking like they do something,
 * which is worse than clutter: a visible control that is ignored teaches you
 * the wrong thing about the node, and you spend an afternoon adjusting
 * `trimap_dilate` on a matter that has no trimap.
 *
 * THE SPEC COMES FROM THE SERVER, not from a table in here. The backends
 * declare what they read in their own PARAMS, and a second copy in the
 * front-end is exactly how the folder incrementer's loader list went stale —
 * it was a hardcoded mirror of something the registry already knew. It also
 * has to be per-install: a machine without SeC must not be offered SeC's
 * controls.
 *
 * WHAT IS NEVER HIDDEN. Sockets. A hidden input that is WIRED is a connection
 * the user can neither see nor undo, so only widgets fold; anything carrying
 * data down a wire stays. And if the spec cannot be fetched, everything is
 * shown — the same honest default a backend that has not declared gets.
 */

import { app } from "../../scripts/app.js";
import { setWidgetVisible, vueSyncNodeWidgets } from "./_widget_visibility.js";

const NODE_ID = "MaskOpsMEC";
const ROUTE = "/c2c/mask/visibility";

let SPEC = null;
let SPEC_TRIED = false;

async function getSpec() {
    if (SPEC || SPEC_TRIED) return SPEC;
    SPEC_TRIED = true;
    try {
        const r = await fetch(ROUTE);
        if (r.ok) SPEC = await r.json();
    } catch (_e) {
        // Unreachable route: leave SPEC null so every widget stays visible.
        // Hiding a control because a fetch failed would be the worst outcome.
        SPEC = null;
    }
    return SPEC;
}

/** The value of another widget, for the toggle rules. */
function valueOf(node, name) {
    const w = node.widgets?.find((x) => x.name === name);
    return w ? w.value : undefined;
}

/** Whether this widget's feature switch is currently on. */
function toggleAllows(node, name, spec) {
    const rule = spec.toggles?.[name];
    if (rule === undefined) return true;
    if (Array.isArray(rule)) {
        // ["post_refine", "none"] -> visible while post_refine is NOT "none"
        return String(valueOf(node, rule[0])) !== String(rule[1]);
    }
    const v = valueOf(node, rule);
    return !(v === false || v === 0 || v === undefined);
}

/** ViTMatte's tile controls (spec.vitmatte_tiling, _visibility.py tiling_visible): only for the vitmatte
 *  matter; on ONYX only matte_tile (its tiling is internal), on the cascade all three. */
function tilingAllows(node, name, spec) {
    const t = spec.vitmatte_tiling;
    if (!t || !(t.widgets || []).includes(name)) return true;
    if (stripBadge(String(valueOf(node, "matter") ?? "")) !== "vitmatte") return false;
    const pipeline = String(valueOf(node, "pipeline") ?? "cascade (legacy)");
    if (pipeline === "onyx") return !(t.cascade_only || []).includes(name);
    return pipeline === "cascade (legacy)";
}

/** Which widgets this segmenter/matter pair actually reads. */
function relevant(node, spec) {
    const seg = String(valueOf(node, "segmenter") ?? "");
    const mat = String(valueOf(node, "matter") ?? "");
    const segInfo = spec.segmenters?.[stripBadge(seg)];
    const matInfo = spec.matters?.[stripBadge(mat)];

    // pipeline="onyx" is one fixed pipeline (SAM 3.1 video + tiled ViTMatte):
    // the segmenter picker decides nothing there, so it cannot be the key.
    // Core + ONYX's own reads + the matter's params, minus what the cascade
    // alone uses (spec.onyx, from _visibility.py).
    if (spec.onyx && String(valueOf(node, "pipeline") ?? "") === "onyx") {
        const keep = new Set([...(spec.always || []), ...(spec.feature || []),
                              ...(spec.sockets || []), ...(spec.onyx.keeps || [])]);
        if (matInfo) (matInfo.params || []).forEach((p) => keep.add(p));
        (spec.onyx.ignores || []).forEach((n) => keep.delete(n));
        return keep;
    }

    // "Has not said" is not a claim: show everything rather than hide a
    // control that might matter.
    if (!segInfo || segInfo.params === undefined) return null;

    // `feature` carries the pipeline-level widgets — luma key, trimap,
    // despill, diagnostics. No backend names them in PARAMS because they
    // belong to the node rather than to a model, so without this they could
    // never be shown at all and their switches would reveal nothing.
    const keep = new Set([...(spec.always || []), ...(spec.feature || []),
                          ...(spec.sockets || [])]);
    (segInfo.params || []).forEach((p) => keep.add(p));
    for (const [mode, names] of Object.entries(spec.by_mode || {})) {
        if ((segInfo.modes || []).includes(mode)) names.forEach((n) => keep.add(n));
    }
    if (segInfo.needs_video) (spec.video_window || []).forEach((n) => keep.add(n));
    if (matInfo) (matInfo.params || []).forEach((p) => keep.add(p));
    return keep;
}

/** Backend names can carry a status badge. The node builds it as "sam3  [missing-deps]" (two spaces + bracket,
 *  node.py _segmenter_choices / _strip_badge); the older " (missing deps)" form is still accepted. Splitting
 *  on " (" alone left the bracket form unmatched, so every missing-deps backend fell back to showing all. */
function stripBadge(v) {
    return String(v).split("  [")[0].split(" (")[0].trim();
}

function apply(node) {
    const spec = SPEC;
    if (!spec || !node.widgets) return;

    const keep = relevant(node, spec);
    const sockets = new Set(spec.sockets || []);
    let hidden = 0;

    for (const w of node.widgets) {
        // Helper widgets (the status strip) are not inputs and carry no
        // value: folding one zeroed its layout height while its element
        // stayed on screen, so it sat on top of the next parameter row.
        if (w.options?.serialize === false || w.serialize === false) continue;
        const isSocket = sockets.has(w.name);
        const wanted = isSocket
            || keep === null                       // undeclared: show all
            || (keep.has(w.name) && toggleAllows(node, w.name, spec)
                && tilingAllows(node, w.name, spec));

        // The shared helper folds BOTH layers: the LiteGraph row (type + zero height) and, for DOM-backed
        // widgets such as the scene_prompts textarea, the element and its .dom-widget wrapper - a bare type
        // swap left that textarea on screen on top of the next row. It also sets widget.hidden /
        // options.hidden, which is what Nodes 2.0 reads.
        setWidgetVisible(w, wanted);
        if (!wanted) hidden += 1;
    }

    node.__c2cHiddenCount = hidden;
    // Nodes 2.0 rebuilds its rows from options.hidden; mask_matting.js syncs after ITS refresh, which can
    // run before this fold, leaving empty rows where folded widgets were (ledger L9.19). Sync here too.
    vueSyncNodeWidgets(node);
    const size = node.computeSize();
    // Only ever GROW to the computed height: shrinking a node the user has
    // deliberately made taller is its own annoyance.
    node.setSize([Math.max(node.size[0], size[0]),
                  Math.max(node.size[1], size[1])]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "C2C.MaskOps.Visibility",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (String(nodeData?.name || "") !== NODE_ID) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;

            getSpec().then(() => {
                try { apply(node); } catch (_e) { /* never break the node */ }
            });

            // Re-apply whenever anything a rule reads changes. Wrapping the
            // callback rather than polling: a timer here would fight the
            // canvas redraw on a graph with many of these.
            const watched = new Set([
                "segmenter", "matter", "pipeline",
                ...Object.values(SPEC?.toggles || {}).flatMap(
                    (r) => (Array.isArray(r) ? [r[0]] : [r])),
                "enable_luma_key", "enable_advanced_trimap", "enable_diagnose",
                "robust_propagation", "auto_quality", "post_refine", "despill",
                "lightwrap_strength", "edge_mode",
            ]);
            for (const w of node.widgets || []) {
                if (!watched.has(w.name)) continue;
                const prev = w.callback;
                w.callback = function (...a) {
                    const out = prev?.apply(this, a);
                    try { apply(node); } catch (_e) { /* ignore */ }
                    return out;
                };
            }
            return r;
        };

        // A loaded workflow restores widget values after creation, so the
        // rules have to run again or a saved graph opens with the wrong set
        // folded.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...a) {
            const r = onConfigure?.apply(this, a);
            const node = this;
            getSpec().then(() => {
                try { apply(node); } catch (_e) { /* ignore */ }
            });
            return r;
        };
    },
});
