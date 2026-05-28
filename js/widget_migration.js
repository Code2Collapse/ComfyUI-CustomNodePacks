// Generic widget-migration safety net for ALL CustomNodePacks (MEC) nodes.
//
// Problem: ComfyUI serializes node widget values positionally as
// `widgets_values: [v0, v1, …]`. When a node author appends a new widget
// to INPUT_TYPES, old saved workflows have a SHORTER array than the new
// widget count. ComfyUI's default behaviour leaves the new widgets at
// whatever value LiteGraph guesses — which can be wrong / out-of-range
// and triggers validation errors at queue time.
//
// This extension hooks `onConfigure` for every MEC node and pads short
// `widgets_values` arrays with each widget's default value pulled from
// `nodeData.input.required` / `nodeData.input.optional`. It is a no-op
// for arrays that already have the right length (or are longer — which
// is handled by per-node migrations like inpaint_crop_pro.js).
//
// Pattern: GENERIC TAIL-PAD ONLY. Middle-insert reorders need per-node
// logic (see inpaint_crop_pro.js for an example).

import { app } from "../../scripts/app.js";

// Match any node whose python_module is this pack OR whose category starts
// with one of these prefixes. Cheap & robust without a hard-coded list.
const PACK_PREFIXES = [
    "MEC", "MaskEditControl", "GLMImage", "NukeMax",
    "WanAnimal", "WanAnimate",
];

function isPackNode(nodeData) {
    if (!nodeData) return false;
    const py = nodeData.python_module || "";
    if (py.includes("ComfyUI-CustomNodePacks")) return true;
    if (py.includes("ComfyUI-GLM_Image")) return true;
    if (py.includes("ComfyUI-NukeMaxNodes")) return true;
    if (py.includes("ComfyUI-WanAnimalPreprocessor")) return true;
    if (py.includes("ComfyUI-WanAnimatePreprocessV2")) return true;
    const cat = nodeData.category || "";
    return PACK_PREFIXES.some(p => cat.startsWith(p));
}

// Resolve the default value for a single INPUT_TYPES entry.
// Spec format is [type, options?] where:
//   type = "INT" | "FLOAT" | "STRING" | "BOOLEAN" | list[options]
//   options.default → preferred. else fall back per type.
function widgetDefault(spec) {
    if (!Array.isArray(spec) || spec.length === 0) return null;
    const t = spec[0];
    const opts = spec.length > 1 && typeof spec[1] === "object" ? spec[1] : {};
    if ("default" in opts) return opts.default;
    if (Array.isArray(t)) return t[0];           // combo widget → first option
    if (t === "INT")     return 0;
    if (t === "FLOAT")   return 0.0;
    if (t === "STRING")  return "";
    if (t === "BOOLEAN") return false;
    return null;
}

// Order matters and must match what ComfyUI writes to widgets_values.
// ComfyUI iterates node.widgets which are added in the same order as
// INPUT_TYPES["required"] then INPUT_TYPES["optional"], skipping pure
// I/O slot inputs (IMAGE, MASK, MODEL, etc.). For a generic safety net
// we iterate required then optional and only count slots that produce
// a widget — i.e. types that are not a known socket-type string.
const SOCKET_TYPES = new Set([
    "IMAGE", "MASK", "LATENT", "MODEL", "CLIP", "VAE", "CONDITIONING",
    "CONTROL_NET", "STYLE_MODEL", "GLIGEN", "AUDIO", "VIDEO", "NOISE",
    "GUIDER", "SAMPLER", "SIGMAS", "PHOTOMAKER",
    // Pack-specific opaque types
    "STITCH_DATA", "CROP_INFO", "BBOX", "BBOX_LIST", "TCL_NODE", "MEC_INSIGHT",
    "ROTO_SHAPE", "TRACKING_DATA", "FFT_TENSOR", "MATERIAL_SET",
    "LIGHT_PROBE", "LIGHT_RIG", "AUDIO_FEATURES", "FLOW_FIELD", "DEEP_IMAGE",
    "ONNX_MODEL", "SAM_MODEL", "ANIMAL_DETECTION_MODEL",
]);

function isWidgetSpec(spec) {
    if (!Array.isArray(spec) || spec.length === 0) return false;
    const t = spec[0];
    if (Array.isArray(t)) return true;            // combo → widget
    if (typeof t !== "string") return false;
    if (SOCKET_TYPES.has(t.toUpperCase())) return false;
    return ["INT", "FLOAT", "STRING", "BOOLEAN"].includes(t.toUpperCase());
}

function expectedWidgetDefaults(nodeData) {
    const out = [];
    const inp = nodeData?.input || {};
    const order = nodeData?.input_order || {};
    for (const bucket of ["required", "optional"]) {
        const dict = inp[bucket] || {};
        const names = order[bucket] || Object.keys(dict);
        for (const name of names) {
            const spec = dict[name];
            if (!isWidgetSpec(spec)) continue;
            // Skip widgets marked forceInput (they become slot inputs).
            const opts = (Array.isArray(spec) && spec.length > 1 && typeof spec[1] === "object") ? spec[1] : {};
            if (opts.forceInput) continue;
            out.push({ name, def: widgetDefault(spec) });
        }
    }
    return out;
}

app.registerExtension({
    name: "MEC.GenericWidgetMigration",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isPackNode(nodeData)) return;
        const expected = expectedWidgetDefaults(nodeData);
        if (expected.length === 0) return;

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            try {
                const wv = info?.widgets_values;
                if (Array.isArray(wv) && wv.length < expected.length) {
                    const need = expected.length - wv.length;
                    const tail = expected.slice(wv.length).map(e => e.def);
                    info.widgets_values = wv.concat(tail);
                    console.log(
                        `[MEC migration] ${nodeData.name}: padded ${need} missing widget(s) ` +
                        `(${expected.slice(wv.length).map(e => e.name).join(", ")})`
                    );
                }
            } catch (e) {
                console.warn(`[MEC migration] ${nodeData?.name || "?"}: pad failed:`, e);
            }
            return onConfigure?.apply(this, arguments);
        };
    },
});
