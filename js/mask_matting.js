/**
 * MaskMattingMEC — dynamic widget visibility.
 *
 * Hides / shows widgets based on the selected segmenter + matter + input_mode
 * so the node panel stays uncluttered. Widgets are hidden by setting their
 * type to "hidden" (LiteGraph respects this) and width to 0; restoring is
 * symmetric. The original type is cached on the widget under
 * ``__mec_origType``.
 */
import { app } from "../../scripts/app.js";

// Maps each widget to a predicate (segmenter,matter,mode,supports,vals) -> bool.
// ``vals`` is a flat {widgetName: value} snapshot, allowing toggle-gated widgets.
const PREDICATES = {
    // pos_points / neg_points / text_prompt are forceInput slots now (no widgets) —
    // not hidden via JS anymore. Keep entries removed so we don't try to
    // toggle non-existent widgets.

    tracking_direction:   (s, m, mode, sup) => sup.has("video") && (mode === "auto" || mode === "video"),
    frame_annotation:     (s, m, mode, sup) => sup.has("video") && (mode === "auto" || mode === "video"),
    object_id:            (s, m, mode, sup) => sup.has("video") && (mode === "auto" || mode === "video"),
    max_frames_to_track:  (s, m, mode, sup) => sup.has("video") && (mode === "auto" || mode === "video"),
    memory_size:          (s, m, mode, sup) => sup.has("video") && (mode === "auto" || mode === "video"),
    start_frame:          (s, m, mode, sup) => sup.has("video") && (mode === "auto" || mode === "video"),
    end_frame:            (s, m, mode, sup) => sup.has("video") && (mode === "auto" || mode === "video"),
    individual_objects:   (s, m, mode, sup) => sup.has("video"),

    // Manual trimap knobs are hidden when subject_preset != "custom"
    // (preset auto-fills dilate/erode/edge) AND when no matter is selected.
    trimap_dilate:        (s, m, mode, sup, v) => m && m !== "none" && (v.subject_preset === "custom"),
    trimap_erode:         (s, m, mode, sup, v) => m && m !== "none" && (v.subject_preset === "custom"),
    edge_radius:          (s, m, mode, sup, v) => m && m !== "none" && (v.subject_preset === "custom"),
    subject_preset:       (s, m) => m && m !== "none",
    matter_model:         (s, m) => m && m !== "none",

    // post_refine dependents
    refine_radius:        (s, m, mode, sup, v) => v.post_refine && v.post_refine !== "none",
    refine_iterations:    (s, m, mode, sup, v) => v.post_refine === "crf" || v.post_refine === "crf+guided",

    // despill dependents
    despill_strength:     (s, m, mode, sup, v) => v.despill && v.despill !== "off",
    preserve_skin:        (s, m, mode, sup, v) => v.despill && v.despill !== "off",

    // lightwrap dependents
    lightwrap_radius:     (s, m, mode, sup, v) => Number(v.lightwrap_strength || 0) > 0,

    // luma-key dependents
    luma_mode:            (s, m, mode, sup, v) => !!v.enable_luma_key,
    luma_low:             (s, m, mode, sup, v) => !!v.enable_luma_key && v.luma_mode === "custom",
    luma_high:            (s, m, mode, sup, v) => !!v.enable_luma_key && v.luma_mode === "custom",
    luma_gamma:           (s, m, mode, sup, v) => !!v.enable_luma_key,
    luma_falloff:         (s, m, mode, sup, v) => !!v.enable_luma_key,
    luma_invert:          (s, m, mode, sup, v) => !!v.enable_luma_key,
    luma_mix:             (s, m, mode, sup, v) => !!v.enable_luma_key,

    // advanced trimap dependents
    trimap_inner_scale:   (s, m, mode, sup, v) => !!v.enable_advanced_trimap,
    trimap_outer_scale:   (s, m, mode, sup, v) => !!v.enable_advanced_trimap,
    trimap_smooth:        (s, m, mode, sup, v) => !!v.enable_advanced_trimap,
    trimap_threshold:     (s, m, mode, sup, v) => !!v.enable_advanced_trimap,

    // auto-quality dependents
    quality_mode:         (s, m, mode, sup, v) => !!v.auto_quality,

    // diagnose dependents
    diag_ring_width:           (s, m, mode, sup, v) => !!v.enable_diagnose,
    diag_blur_threshold:       (s, m, mode, sup, v) => !!v.enable_diagnose,
    diag_brightness_threshold: (s, m, mode, sup, v) => !!v.enable_diagnose,

    // robust propagation dependents — only relevant in video mode too
    robust_propagation:          (s, m, mode, sup) => sup.has("video"),
    robust_confidence_threshold: (s, m, mode, sup, v) => !!v.robust_propagation && sup.has("video"),
    robust_reanchor_method:      (s, m, mode, sup, v) => !!v.robust_propagation && sup.has("video"),
    robust_blend_alpha:          (s, m, mode, sup, v) => !!v.robust_propagation && sup.has("video") && (v.robust_reanchor_method === "blend"),
};

// Coarse capability table mirrors segmenters/*.SUPPORTS_MODES on the Python side.
// Keep in sync if you wire new backends.
const SEGMENTER_MODES = {
    "sam2.1":         new Set(["points", "bbox", "auto", "video"]),
    "sam3":           new Set(["points", "bbox", "text", "auto"]),
    "sam3.1":         new Set(["points", "bbox", "text", "auto"]),
    "sec":            new Set(["points", "bbox", "video", "auto"]),
    "grounding-dino": new Set(["text"]),
    "birefnet":       new Set(["auto"]),
    "rmbg":           new Set(["auto"]),
    "videomama":      new Set(["text", "video"]),
    "inspyrenet":     new Set(["auto"]),
    "cutie":          new Set(["points", "bbox", "video"]),
    "dis":            new Set(["auto"]),
    "xmem":           new Set(["points", "bbox", "video"]),
    "person-mask":    new Set(["auto"]),
};

// Map the user-facing segmenter / matter name to its folder_paths key.
// Mirrors *.MODELS_KEY on the Python side.
const SEGMENTER_TO_KEY = {
    "sam2.1":         "sam2",
    "sam3":           "sam3",
    "sam3.1":         "sam3.1",
    "sec":            "sec",
    "grounding-dino": "grounding-dino",
    "birefnet":       "birefnet",
    "rmbg":           "rmbg",
    "videomama":      "videomama",
    "inspyrenet":     "inspyrenet",
    "cutie":          "cutie",
    "dis":            "dis",
    "xmem":           "xmem",
    "person-mask":    "person-mask",
};
const MATTER_TO_KEY = {
    "vitmatte":   "vitmatte",
    "rvm":        "rvm",
    "matanyone":  "matanyone",
    "bgmattingv2":"bgmattingv2",
};

// Filter a flat dropdown list (sam2/foo.pt, [preset:sam3] x.safetensors, ...)
// down to entries that belong to ``backendKey``. ``(auto)`` is always kept.
function filterChoicesForBackend(allChoices, backendKey) {
    if (!backendKey) return allChoices.slice();
    const out = [];
    const localPrefix  = `${backendKey}/`;
    const presetPrefix = `[preset:${backendKey}] `;
    for (const c of allChoices) {
        if (c === "(auto)") { out.push(c); continue; }
        if (c.startsWith(localPrefix) || c.startsWith(presetPrefix)) out.push(c);
    }
    if (!out.includes("(auto)")) out.unshift("(auto)");
    return out;
}

// Apply the filtered list to a combo widget. Caches the original full
// list on widget.__mec_allChoices so we can re-filter on every change.
function applyFilteredChoices(widget, allChoices, backendKey) {
    if (!widget) return;
    if (!widget.__mec_allChoices) widget.__mec_allChoices = allChoices.slice();
    const filtered = filterChoicesForBackend(widget.__mec_allChoices, backendKey);
    if (widget.options) widget.options.values = filtered;
    // Keep the user's selection if it's still in the filtered list,
    // otherwise snap to the first installed weight (or "(auto)").
    if (!filtered.includes(widget.value)) {
        // Prefer a real local weight over (auto) so we don't leave the
        // user staring at an empty selection when files exist.
        const realPick = filtered.find(c => c !== "(auto)" && !c.startsWith("[preset:"));
        widget.value = realPick || filtered[0] || "(auto)";
        widget.callback?.(widget.value);
    }
}

function stripBadge(s) {
    return (s || "").split("  [")[0].trim();
}

function setHidden(widget, hide) {
    if (!widget) return;
    if (widget.__mec_origType === undefined) {
        widget.__mec_origType = widget.type;
        widget.__mec_origComputeSize = widget.computeSize;
        widget.__mec_origDraw = widget.draw;
    }
    if (hide) {
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
    } else {
        widget.type = widget.__mec_origType;
        widget.computeSize = widget.__mec_origComputeSize;
        widget.draw = widget.__mec_origDraw;
    }
    // Multiline STRING widgets (text_prompt, pos_points, neg_points) and
    // any other DOM-backed widget mount their own <textarea>/<div> that
    // LiteGraph positions independently of the canvas widget list.
    // Toggling widget.type alone leaves that element visible at the bottom
    // of the node and prevents it from shrinking. Hide the DOM node too.
    const el = widget.element;
    if (el) {
        el.hidden = !!hide;
        el.style.display = hide ? "none" : "";
        const wrap = el.parentElement;
        if (wrap?.classList?.contains("dom-widget")) {
            wrap.style.display = hide ? "none" : "";
        }
    }
}

function refreshVisibility(node) {
    const widgetMap = {};
    for (const w of node.widgets || []) widgetMap[w.name] = w;
    const segWidget   = widgetMap.segmenter;
    const matWidget   = widgetMap.matter;
    if (!segWidget || !matWidget) return;
    const seg  = stripBadge(segWidget.value);
    const mat  = stripBadge(matWidget.value);
    const mode = "auto";
    const sup  = SEGMENTER_MODES[seg] || new Set(["auto"]);

    // Per-backend filtered model dropdowns. The Python side ships ONE big
    // list (sam2/..., sam3/..., [preset:sam2] ..., etc.); here we keep
    // only the entries whose backend matches the current segmenter /
    // matter so the user never sees "vitmatte" weights when picking SAM2.
    const segKey = SEGMENTER_TO_KEY[seg] || seg;
    const matKey = (mat && mat !== "none") ? (MATTER_TO_KEY[mat] || mat) : "";
    const modelW = widgetMap.model;
    const matterModelW = widgetMap.matter_model;
    if (modelW)  applyFilteredChoices(modelW,        modelW.options?.values        || [], segKey);
    if (matterModelW) applyFilteredChoices(matterModelW, matterModelW.options?.values || [], matKey);
    // Hide matter_model entirely when matter == none.
    if (!matKey) setHidden(matterModelW, true);

    for (const [name, pred] of Object.entries(PREDICATES)) {
        const w = widgetMap[name];
        if (!w) continue;
        // Build a flat snapshot of widget values so predicates can
        // gate on sibling toggles (enable_luma_key, robust_propagation, …).
        const vals = {};
        for (const ww of node.widgets || []) vals[ww.name] = ww.value;
        const visible = !!pred(seg, mat, mode, sup, vals);
        setHidden(w, !visible);
    }
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "MEC.MaskMatting.DynamicWidgets",
    async beforeRegisterNodeDef(nodeType, nodeData, _appRef) {
        if (nodeData?.name !== "MaskOpsMEC") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            // Hook each control widget so any change refreshes visibility.
            const node = this;
            const triggers = [
                "segmenter", "matter", "subject_preset",
                "enable_luma_key", "luma_mode",
                "enable_advanced_trimap",
                "enable_diagnose",
                "despill", "post_refine",
                "lightwrap_strength",
                "auto_quality",
                "robust_propagation", "robust_reanchor_method",
            ];
            for (const w of node.widgets || []) {
                if (!triggers.includes(w.name)) continue;
                const orig = w.callback;
                w.callback = function (...args) {
                    const out = orig?.apply(this, args);
                    refreshVisibility(node);
                    return out;
                };
            }
            // Initial pass after the next tick (widgets fully populated).
            setTimeout(() => refreshVisibility(node), 0);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => refreshVisibility(this), 0);
            return r;
        };
    },
});
