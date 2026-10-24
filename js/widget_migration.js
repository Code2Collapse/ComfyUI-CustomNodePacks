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
// is handled by per-node migrations, e.g. c2c_auto_migrate.js).
//
// Pattern: GENERIC TAIL-PAD ONLY. Middle-insert reorders need per-node
// logic (c2c_auto_migrate.js rewrites removed / merged nodes on load).

import { app } from "../../scripts/app.js";

// Match any node whose python_module is this pack OR whose category starts
// with one of these prefixes. Cheap & robust without a hard-coded list.
const PACK_PREFIXES = [
    // Every Code2Collapse node now lives under one menu root (see
    // _c2c_menu.py). This fallback matters on installs whose folder names
    // differ from python_module's checks above - the Linux box uses the GitHub
    // names (ComfyUI-NukeNodePack, ...) - and without the root here the
    // safety net would have silently switched off for them.
    "\u{1F43A} C2C",
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

function specOf(nodeData, name) {
    const inp = nodeData?.input || {};
    return inp.required?.[name] ?? inp.optional?.[name];
}

// Does a saved value fit the Python widget it would land on?
function fits(spec, v) {
    if (!Array.isArray(spec)) return true;
    const t = spec[0];
    if (Array.isArray(t)) return t.includes(v);
    const u = String(t).toUpperCase();
    if (u === "INT" || u === "FLOAT") return typeof v === "number" && Number.isFinite(v);
    if (u === "BOOLEAN") return typeof v === "boolean";
    if (u === "STRING") return typeof v === "string";
    return true;
}

// Widgets in the order LiteGraph writes widgets_values (every widget whose `serialize` is not false).
function serializedWidgets(node) {
    return (node.widgets || []).filter((w) => w && w.serialize !== false);
}

// Front-end widgets saved AFTER the Python ones (an upload button, a hidden preview state, ...).
function trailingExtras(node, pyNames) {
    const sw = serializedWidgets(node);
    let last = -1;
    sw.forEach((w, i) => { if (pyNames.has(w.name)) last = i; });
    return sw.slice(last + 1);
}

// L9.38 (owner log 2026-10-24: LoadVideoC2C "handle_frames, , invalid literal for int()"): a save made before new
// Python widgets were appended ends with the front-end widgets' values (Load Video: upload "image", preview "").
// Loaded by position, those land on the new Python widgets. Insert the new widgets' defaults BEFORE that tail.
function insertBeforeExtras(node, nodeData, expected, wv) {
    const pyNames = new Set(expected.map((e) => e.name));
    const extras = trailingExtras(node, pyNames);
    const E = extras.length;
    if (!E || wv.length < E) return null;
    const pySaved = wv.length - E;
    if (pySaved >= expected.length) return null;
    const tailLooksLikeExtras = extras.every((w, k) => {
        const v = wv[pySaved + k];
        return v === null || v === undefined || typeof v === typeof w.value;
    });
    if (!tailLooksLikeExtras) return null;
    if (fits(specOf(nodeData, expected[pySaved].name), wv[pySaved])) return null;   // the old tail is not misplaced
    const missing = expected.slice(pySaved);
    return { values: wv.slice(0, pySaved).concat(missing.map((e) => e.def), wv.slice(pySaved)), missing };
}

// A workflow already re-saved after the shift carries the misplaced values at full length. Reset only values the
// widget can never accept: a non-number in INT / FLOAT, a non-boolean in BOOLEAN, or a combo holding a trailing
// front-end widget's value (e.g. "image" from the upload button). Anything else is left alone.
function sanitize(node, nodeData, expected) {
    const pyNames = new Set(expected.map((e) => e.name));
    const extraValues = new Set(trailingExtras(node, pyNames).map((w) => w.value)
        .filter((v) => typeof v === "string" && v !== ""));
    const fixed = [];
    for (const e of expected) {
        const w = node.widgets?.find((x) => x.name === e.name);
        if (!w) continue;
        const spec = specOf(nodeData, e.name);
        const t = Array.isArray(spec) ? spec[0] : null;
        const v = w.value;
        let bad = false;
        if (Array.isArray(t)) bad = !t.includes(v) && extraValues.has(v);
        else if (t === "INT" || t === "FLOAT") {
            if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) { w.value = Number(v); continue; }
            bad = !(typeof v === "number" && Number.isFinite(v));
        } else if (t === "BOOLEAN") bad = typeof v !== "boolean";
        if (bad && e.def !== null && e.def !== undefined) {
            fixed.push(`${e.name}=${JSON.stringify(v)}`);
            w.value = e.def;
        }
    }
    if (fixed.length) {
        console.warn(`[MEC migration] ${nodeData.name} #${node.id}: reset misplaced value(s) to the default: `
            + fixed.join(", ") + " (saved before new inputs were added; save the workflow again to keep the fix)");
    }
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
                if (Array.isArray(wv) && wv.length < serializedWidgets(this).length) {
                    const ins = insertBeforeExtras(this, nodeData, expected, wv);
                    if (ins) {
                        info.widgets_values = ins.values;
                        console.log(`[MEC migration] ${nodeData.name}: inserted ${ins.missing.length} new widget(s) `
                            + `before the front-end widgets (${ins.missing.map((e) => e.name).join(", ")})`);
                    }
                }
                const wv2 = info?.widgets_values;
                if (Array.isArray(wv2) && wv2.length < expected.length) {
                    const need = expected.length - wv2.length;
                    const tail = expected.slice(wv2.length).map(e => e.def);
                    info.widgets_values = wv2.concat(tail);
                    console.log(
                        `[MEC migration] ${nodeData.name}: padded ${need} missing widget(s) ` +
                        `(${expected.slice(wv2.length).map(e => e.name).join(", ")})`
                    );
                }
            } catch (e) {
                console.warn(`[MEC migration] ${nodeData?.name || "?"}: pad failed:`, e);
            }
            const r = onConfigure?.apply(this, arguments);
            // the front end writes the saved values into the widgets after this hook: check them on the next tick
            setTimeout(() => {
                try { sanitize(this, nodeData, expected); } catch (e) { /* node may be gone */ }
            }, 0);
            return r;
        };
    },
});
