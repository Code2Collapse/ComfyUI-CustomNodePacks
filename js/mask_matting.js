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
import { vueSyncNodeWidgets } from "./_widget_visibility.js";
import { ensureC2CKit } from "./_c2c_ui_kit.js";

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
    "locate_anything": new Set(["text", "auto"]),
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
    "locate_anything": "locate_anything",
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

/** The server-driven folding (mask_visibility.js, spec from _visibility.py)
 *  is the authority when it is installed: two systems hiding the same widgets
 *  undo each other, depending on which callback ran last. The PREDICATES
 *  table below is kept only as the fallback for an install without it. */
function serverFoldingActive() {
    try { return (app.extensions || []).some((e) => e?.name === "C2C.MaskOps.Visibility"); }
    catch (_e) { return false; }
}

/** Objects in the scene_prompts JSON, 0 when it is empty, -1 when not JSON. */
function sceneObjectCount(raw) {
    if (!raw || !String(raw).trim()) return 0;
    try {
        const d = JSON.parse(raw);
        return Array.isArray(d?.objects) ? d.objects.filter((o) => o?.enabled !== false).length : 0;
    } catch (_e) { return -1; }
}

function refreshVisibility(node) {
    const widgetMap = {};
    for (const w of node.widgets || []) widgetMap[w.name] = w;
    const segWidget   = widgetMap.segmenter;
    const matWidget   = widgetMap.matter;
    if (!segWidget || !matWidget) return;
    const onyx = String(widgetMap.pipeline?.value || "") === "onyx";
    // ONYX always tracks with SAM 3.1, whatever the (folded) segmenter says.
    const seg  = onyx ? "sam3.1" : stripBadge(segWidget.value);
    const mat  = stripBadge(matWidget.value);
    const modeW = widgetMap.input_mode;
    const mode = modeW ? String(modeW.value || "auto").toLowerCase() : "auto";
    const sup  = onyx ? new Set(["points", "bbox", "text", "video", "auto"])
                      : (SEGMENTER_MODES[seg] || new Set(["auto"]));

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

    const ownFolding = !serverFoldingActive();
    for (const [name, pred] of Object.entries(PREDICATES)) {
        if (!ownFolding) break;
        const w = widgetMap[name];
        if (!w) continue;
        // Build a flat snapshot of widget values so predicates can
        // gate on sibling toggles (enable_luma_key, robust_propagation, …).
        const vals = {};
        for (const ww of node.widgets || []) vals[ww.name] = ww.value;
        const visible = !!pred(seg, mat, mode, sup, vals);
        setHidden(w, !visible);
    }

    // Live status header: at-a-glance pipeline summary + a warning when the
    // selected backend has NO local weights (the combo silently falls back
    // to showing only "(auto)" in that case — easy to miss in a 30+ param
    // node). Reuses the filtered lists applyFilteredChoices already built
    // above, so this costs nothing extra to compute.
    if (node._mecStatusPill) {
        const visibleCount = (node.widgets || [])
            .filter(w => !String(w.type).includes("hidden") && w.name !== "mec_status_header").length;
        const matLabel = (mat && mat !== "none") ? mat : "no matte";
        if (onyx) {
            const n = sceneObjectCount(widgetMap.scene_prompts?.value);
            const objects = n < 0 ? "scene JSON invalid"
                : n === 0 ? "1 object (prompt sockets)" : `${n} object${n === 1 ? "" : "s"}`;
            node._mecStatusPill.textContent = `SAM 3.1 video → ${matLabel} · ${objects} · ${visibleCount} params`;
        } else {
            node._mecStatusPill.textContent =
                `${seg || "?"} → ${matLabel} · ${mode} · ${visibleCount} params`;
        }
        node._mecStatusPill.title = node._mecStatusPill.textContent;
    }
    if (node._mecPipeBtns) {
        for (const [val, btn] of node._mecPipeBtns) {
            const on = (val === "onyx") === onyx;
            btn.setAttribute("aria-pressed", on ? "true" : "false");
            btn.style.background = on ? "color-mix(in srgb, #b494ff 30%, transparent)" : "transparent";
            btn.style.color = on ? "#f3f1ff" : "#8280ba";
        }
    }
    if (node._mecWarnPill) {
        // "auto"/"auto_best" are meta cascade-selectors, not concrete
        // backends — they route across whatever IS available at runtime, so
        // their model dropdown is legitimately "(auto)"-only. Only warn for
        // an actual named backend (present in *_TO_KEY) with no weights.
        const segIsConcrete = Object.prototype.hasOwnProperty.call(SEGMENTER_TO_KEY, seg);
        const onyxNoLocal = onyx && !(modelW?.options?.values || []).some((c) => c.startsWith("sam3.1/"));
        const matIsConcrete = !!mat && mat !== "none" && Object.prototype.hasOwnProperty.call(MATTER_TO_KEY, mat);
        const segChoices = modelW?.options?.values || [];
        const matChoices = matterModelW?.options?.values || [];
        const segMissing = segIsConcrete && segChoices.length > 0 && segChoices.every(c => c === "(auto)");
        const matMissing = matIsConcrete && matChoices.length > 0 && matChoices.every(c => c === "(auto)");
        if (onyxNoLocal) {
            const auto = !!widgetMap.auto_download?.value;
            node._mecWarnPill.textContent = auto ? "↓ SAM 3.1" : "⚠ SAM 3.1";
            node._mecWarnPill.title = auto
                ? "No SAM 3.1 checkpoint in models/sam3.1 yet - it downloads on the first run."
                : "No SAM 3.1 checkpoint in models/sam3.1. Tick auto_download, or place the weights there.";
            node._mecWarnPill.style.display = "inline-flex";
        } else if (segMissing || matMissing) {
            const missing = [segMissing && seg, matMissing && mat].filter(Boolean);
            node._mecWarnPill.textContent = `⚠ ${missing.join(", ")}`;
            node._mecWarnPill.title = `No local weights for ${missing.join(" and ")} - the dropdown only offers (auto).`;
            node._mecWarnPill.style.display = "inline-flex";
        } else {
            node._mecWarnPill.style.display = "none";
        }
    }

    vueSyncNodeWidgets(node);
    const sz = node.computeSize();
    // 30+ visible params: LiteGraph's title-derived default width (~253px)
    // truncates almost every label. 340px fits the longest label + value.
    node.setSize([Math.max(node.size[0], sz[0], 340), sz[1]]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "MEC.MaskMatting.DynamicWidgets",
    async beforeRegisterNodeDef(nodeType, nodeData, _appRef) {
        if (nodeData?.name !== "MaskOpsMEC") return;

        const _origIsWidgetVisible = nodeType.prototype.isWidgetVisible;
        nodeType.prototype.isWidgetVisible = function (widget) {
            if (widget?.__mec_origType === "hidden" || widget?.type === "hidden") return false;
            return _origIsWidgetVisible ? _origIsWidgetVisible.call(this, widget) : true;
        };

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;
            if (!node.properties) node.properties = {};
            node.properties.c2c_pipeline_v = 1;

            // Compact DOM status header — this node is a flat wall of 30+
            // native widgets (14 segmenter backends x 4 matting backends,
            // each with its own conditional sub-params); a one-line live
            // summary + a missing-weights warning gives at-a-glance context
            // that the native combo/slider rows can't.
            ensureC2CKit();
            const statusEl = document.createElement("div");
            statusEl.className = "c2ck";
            statusEl.style.cssText =
                "display:flex; align-items:center; gap:6px; flex-wrap:nowrap; overflow:hidden; " +
                "padding:0 2px; height:100%; box-sizing:border-box;";
            // Pipeline switch. The `pipeline` widget has to stay at the END of
            // the inputs (saved workflows are positional), which puts the
            // most important choice at the bottom of a tall node - so the
            // strip at the top carries it too.
            const seg = document.createElement("div");
            seg.setAttribute("role", "group");
            seg.setAttribute("aria-label", "Pipeline");
            seg.style.cssText = "display:inline-flex; flex:none; border:1px solid " +
                "color-mix(in srgb, #b494ff 45%, transparent); border-radius:5px; overflow:hidden;";
            const pipeBtns = [];
            for (const [val, label, tip] of [
                ["onyx", "ONYX", "One SAM 3.1 tracking session for the clip, up to 16 objects, tiled ViTMatte at full resolution."],
                ["cascade (legacy)", "Cascade", "The previous multi-backend auto_best path."],
            ]) {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.textContent = label;
                btn.title = tip;
                btn.style.cssText = "font:600 10px/1 system-ui,sans-serif; letter-spacing:.04em; " +
                    "padding:4px 8px; border:0; cursor:pointer; background:transparent; color:#8280ba;";
                btn.addEventListener("pointerdown", (e) => e.stopPropagation());
                btn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const w = (node.widgets || []).find((x) => x.name === "pipeline");
                    if (!w || w.value === val) return;
                    w.value = val;
                    w.callback?.(val);
                    node.setDirtyCanvas?.(true, true);
                });
                seg.appendChild(btn);
                pipeBtns.push([val, btn]);
            }
            node._mecPipeBtns = pipeBtns;
            const statusPill = document.createElement("span");
            statusPill.className = "c2ck-pill on";
            statusPill.style.cssText = "font-size:10px; white-space:nowrap; overflow:hidden; " +
                "text-overflow:ellipsis; min-width:0; flex:1 1 auto;";
            statusPill.textContent = "…";
            const warnPill = document.createElement("span");
            warnPill.className = "c2ck-pill off";
            warnPill.style.cssText = "font-size:10px; display:none; flex:none; white-space:nowrap;";
            statusEl.append(seg, statusPill, warnPill);
            // ComfyUI insets a DOM widget by `margin` (default 10) on every
            // side: a 22 px slot left the pill 2 px, so it spilled over the
            // first parameter row. 3 px margin in a 28 px slot fits it.
            const statusWidget = node.addDOMWidget("mec_status_header", "STATUS", statusEl, {
                serialize: false,
                margin: 3,
                getHeight: () => 28,
                getMinHeight: () => 28,
            });
            statusWidget.computeSize = () => [node.size?.[0] || 340, 28];
            // Move it to the front of the widget list so it reads as a
            // header above the parameter rows, not a footer beneath them.
            const _si = node.widgets.indexOf(statusWidget);
            if (_si > 0) { node.widgets.splice(_si, 1); node.widgets.unshift(statusWidget); }
            node._mecStatusPill = statusPill;
            node._mecWarnPill = warnPill;

            // Hook each control widget so any change refreshes visibility.
            const triggers = [
                "pipeline", "scene_prompts", "auto_download",
                "segmenter", "matter", "input_mode", "subject_preset",
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
                    // after the server-driven folding in the same chain, so
                    // the size fit sees the final widget set
                    setTimeout(() => refreshVisibility(node), 0);
                    return out;
                };
            }
            // Initial pass after the next tick (widgets fully populated), and
            // again once the folding spec has arrived from the server.
            setTimeout(() => refreshVisibility(node), 0);
            setTimeout(() => refreshVisibility(node), 400);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            if (!(info?.properties && "c2c_pipeline_v" in info.properties)) {
                const pipeW = (this.widgets || []).find((w) => w.name === "pipeline");
                if (pipeW) pipeW.value = "cascade (legacy)";
            }
            if (!this.properties) this.properties = {};
            this.properties.c2c_pipeline_v = 1;
            setTimeout(() => refreshVisibility(this), 0);
            setTimeout(() => refreshVisibility(this), 400);
            return r;
        };
    },
});
