// C2C Node Identity - deterministic naming engine (pure: no imports, no DOM, no time, no randomness).
// Design: docs/research/node_identity_design.md (workspace root), section 3. Runs in the browser and under Node
// (tests/test_node_identity_engine.mjs). Input is a plain SNAPSHOT built by adapter.js; output is a list of rows.
// The same snapshot always gives the same names, whatever the node order in it.
//
// Snapshot node: {key, id, type, title, defaultTitle, category, outputNode, isVirtual, isSubgraph,
//                 widgets:{name:value}, inputs:[{name,type,from}], outputs:[{name,type,to:[key]}], pos:[x,y],
//                 group, path, srName, identity}
// License: Apache-2.0

export const DEFAULT_RULES = Object.freeze({
    template: "{stage} · {function}",
    separator: " · ",
    use_prompt_words: true,
    keep_manual: true,
    excluded_types: [],
    max_len: 64,
});

// Title managed by its own extension, or a title that IS data (bookmark name): never renamed automatically.
export const ALWAYS_EXCLUDED = Object.freeze(["SetNode", "GetNode", "Reroute", "Note", "MarkdownNote",
    "PrimitiveNode", "Bookmark (rgthree)", "Label (rgthree)"]);

const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
const str = (v) => (v == null ? "" : String(v));

export function rulesOf(rules) {
    const r = { ...DEFAULT_RULES, ...(rules || {}) };
    if (!Array.isArray(r.excluded_types)) r.excluded_types = [];
    if (typeof r.separator !== "string" || !r.separator) r.separator = DEFAULT_RULES.separator;
    if (typeof r.template !== "string" || !r.template) r.template = DEFAULT_RULES.template;
    r.max_len = Number.isFinite(+r.max_len) && +r.max_len >= 8 ? Math.floor(+r.max_len) : DEFAULT_RULES.max_len;
    return r;
}

// ── snapshot helpers ────────────────────────────────────────────────────────────────────────────────────────
export function indexSnapshot(snap) {
    if (snap.byKey instanceof Map) return snap;
    const byKey = new Map();
    for (const n of snap.nodes || []) byKey.set(String(n.key), n);
    return { ...snap, byKey };
}

const nodeOf = (snap, key) => (key == null ? null : snap.byKey.get(String(key)) || null);

function upstream(snap, n, types) {
    const out = [];
    for (const i of n.inputs || []) {
        if (i.from == null) continue;
        if (types && !types.includes(str(i.type).toUpperCase())) continue;
        const u = nodeOf(snap, i.from);
        if (u) out.push({ node: u, input: i });
    }
    return out;
}

function downstream(snap, n, types) {
    const out = [];
    for (const o of n.outputs || []) {
        if (types && !types.includes(str(o.type).toUpperCase())) continue;
        (o.to || []).forEach((k, i) => {
            const d = nodeOf(snap, k);
            // toInputs: the target input's name, given by the adapter (needed across a subgraph boundary, where the
            // target's own input is fed by the subgraph node, not by this node)
            if (d) out.push({ node: d, output: o, inputName: o.toInputs?.[i] ?? null });
        });
    }
    return out;
}

/** The input of `target` that is fed by `source` (first match, by input order). */
function inputFedBy(target, sourceKey) {
    return (target.inputs || []).find((i) => i.from != null && String(i.from) === String(sourceKey)) || null;
}

// ── text helpers ────────────────────────────────────────────────────────────────────────────────────────────
const STOP = new Set(["a", "an", "the", "of", "and", "with", "in", "on", "at", "to", "for", "by", "is", "best",
    "quality", "masterpiece", "high", "highly", "detailed", "very", "photo", "image", "picture", "8k", "4k", "hd",
    "uhd", "raw", "realistic", "ultra", "score_9", "score_8_up", "score_7_up"]);

function titleCase(w) {
    if (!w) return w;
    if (/^[A-Z0-9]+$/.test(w) && w.length <= 5) return w;          // keep acronyms (VAE, XL, H3)
    return w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
}

export function words(text, max = 3, stop = true) {
    const parts = str(text).replace(/\([^)]*:[\d.]+\)/g, " ")          // (word:1.2) weights
        .replace(/<[^>]*>/g, " ")                                        // <lora:...>
        .split(/[^A-Za-z0-9]+/).filter(Boolean);
    const out = [];
    for (const p of parts) {
        if (stop && STOP.has(p.toLowerCase())) continue;
        if (/^\d+$/.test(p) && out.length === 0) continue;               // no leading bare numbers
        out.push(titleCase(p));
        if (out.length >= max) break;
    }
    return out.join(" ");
}

export function fileStem(value) {
    const s = str(value).split(/[\\/]/).pop();
    return s.replace(/\.[A-Za-z0-9]{1,5}$/, "");
}

// ── model family ────────────────────────────────────────────────────────────────────────────────────────────
const FAMILIES = [
    [/minimax|(^|[^a-z0-9])h3([^a-z0-9]|$)/i, "H3"],
    [/(^|[^a-z0-9])wan/i, "WAN"],
    [/flux/i, "FLUX"],
    [/ltx/i, "LTX"],
    [/hunyuan/i, "HUNYUAN"],
    [/qwen/i, "QWEN"],
    [/sd3|stable[-_ ]?diffusion[-_ ]?3/i, "SD3"],
    [/sdxl|sd_xl|[-_ ]xl[-_. ]|pony|illustrious/i, "SDXL"],
    [/sd15|sd_1\.?5|v1-5|sd1\.5/i, "SD15"],
];

export function familyOf(text) {
    const s = str(text);
    for (const [re, name] of FAMILIES) if (re.test(s)) return name;
    return "";
}

const MODEL_WIDGETS = ["ckpt_name", "unet_name", "model_name", "model", "model_path", "checkpoint", "gguf_name",
    "lora_name", "clip_name", "clip_name1", "vae_name"];

function modelFile(n) {
    const w = n.widgets || {};
    for (const k of MODEL_WIDGETS) if (typeof w[k] === "string" && w[k]) return w[k];
    return "";
}

/** Model family of a node: its own type / model file, else the first one found upstream along MODEL edges. */
function familyUpstream(snap, n, depth = 0, seen = new Set()) {
    if (!n || depth > 12 || seen.has(n.key)) return "";
    seen.add(n.key);
    const own = familyOf(modelFile(n)) || familyOf(n.type);
    if (own && depth > 0) return own;
    if (depth === 0 && familyOf(n.type)) return familyOf(n.type);
    for (const { node } of upstream(snap, n, ["MODEL"])) {
        const f = familyUpstream(snap, node, depth + 1, seen);
        if (f) return f;
    }
    if (depth === 0) {                                    // guider / custom sampler: any upstream edge (bounded)
        for (const { node } of upstream(snap, n, ["GUIDER", "SAMPLER", "SIGMAS"])) {
            const f = familyUpstream(snap, node, depth + 1, seen);
            if (f) return f;
        }
    }
    return own;
}

// ── role predicates ─────────────────────────────────────────────────────────────────────────────────────────
const outTypes = (n) => (n.outputs || []).map((o) => str(o.type).toUpperCase());
const inTypes = (n) => (n.inputs || []).map((i) => str(i.type).toUpperCase());
const cat0 = (n) => str(n.category).split("/")[0].trim().toLowerCase();

export function isSampler(n) {
    const t = str(n.type);
    return outTypes(n).includes("LATENT") && (/sampler/i.test(t) || cat0(n) === "sampling")
        && !/^(KSamplerSelect|SamplerSelect)/i.test(t) && !/select|scheduler/i.test(t);
}
const isUpscale = (n) => /upscal|latentupscale|imagescale|resize/i.test(str(n.type));
const isEncode = (n) => /vaeencode|encode.*vae|^vae.*encode/i.test(str(n.type)) && outTypes(n).includes("LATENT");
const isDecode = (n) => /vaedecode|decode/i.test(str(n.type)) && inTypes(n).includes("LATENT")
    && outTypes(n).some((t) => t === "IMAGE" || t === "VIDEO");
const isEmptyLatent = (n) => /empty.*latent|latent.*empty/i.test(str(n.type)) && outTypes(n).includes("LATENT");
const isPreview = (n) => /preview/i.test(str(n.type));
const isOutput = (n) => !!n.outputNode || /^save|save$|savevideo|videocombine|video_combine/i.test(str(n.type));
const isOutputish = (n) => /save|preview|combine|write|export/i.test(str(n.type))
    && (!!n.outputNode || !outTypes(n).length);
const FILE_RE = /\.(safetensors|ckpt|pt|pth|bin|gguf|onnx|png|jpe?g|webp|mp4|mov|webm|wav|mp3|flac)$/i;
const isTextEncoder = (n) => outTypes(n).includes("CONDITIONING")
    && Object.values(n.widgets || {}).some((v) => typeof v === "string" && !FILE_RE.test(v))
    && (/text|encode|prompt/i.test(str(n.type)) || cat0(n) === "conditioning");
const isMediaLoader = (n) => /^load|loader|load_?video|loadimage|loadaudio/i.test(str(n.type))
    && outTypes(n).some((t) => ["IMAGE", "VIDEO", "AUDIO", "MASK"].includes(t))
    && !outTypes(n).some((t) => ["MODEL", "CLIP", "VAE"].includes(t));

function promptText(n) {
    const w = n.widgets || {};
    for (const k of ["text", "prompt", "positive", "text_g", "text_l", "string", "value"]) {
        if (typeof w[k] === "string") return w[k];
    }
    let best = "";
    for (const v of Object.values(w)) if (typeof v === "string" && v.length > best.length) best = v;
    return best;
}

/** Positive / Negative by walking CONDITIONING edges downstream to an input named positive* / negative*. */
function polarity(snap, n) {
    const seen = new Set([n.key]);
    let frontier = [n];
    for (let depth = 0; depth < 8 && frontier.length; depth++) {
        const next = [];
        const sorted = frontier.slice().sort((a, b) => cmp(String(a.key), String(b.key)));
        for (const cur of sorted) {
            const outs = downstream(snap, cur).sort((a, b) => cmp(String(a.node.key), String(b.node.key)));
            for (const { node, inputName } of outs) {
                const name = str(inputName ?? inputFedBy(node, cur.key)?.name).toLowerCase();
                if (name.startsWith("positive") || name === "cond_pos" || name === "conditioning_pos") return "Positive";
                if (name.startsWith("negative") || name === "cond_neg" || name === "conditioning_neg") return "Negative";
                if (!seen.has(node.key) && outTypes(node).includes("CONDITIONING")) {
                    seen.add(node.key);
                    next.push(node);
                }
            }
        }
        frontier = next;
    }
    return "";
}

/** Walk LATENT edges upstream from a sampler: what feeds it? */
function latentOrigin(snap, n) {
    const seen = new Set([n.key]);
    let frontier = upstream(snap, n, ["LATENT"]).map((x) => x.node);
    let viaUpscale = false;
    for (let depth = 0; depth < 12 && frontier.length; depth++) {
        const next = [];
        for (const cur of frontier.slice().sort((a, b) => cmp(String(a.key), String(b.key)))) {
            if (seen.has(cur.key)) continue;
            seen.add(cur.key);
            if (isSampler(cur)) return viaUpscale ? "upscale" : "refine";
            if (isUpscale(cur)) viaUpscale = true;
            if (isEncode(cur)) {
                const img = upstream(snap, cur, ["IMAGE"]).map((x) => x.node);
                if (img.some((x) => isUpscale(x) || upstreamHasSampler(snap, x))) return "upscale";
                return "img2img";
            }
            if (isEmptyLatent(cur)) return "initial";
            for (const { node } of upstream(snap, cur, ["LATENT"])) next.push(node);
        }
        frontier = next;
    }
    return viaUpscale ? "upscale" : "initial";
}

function upstreamHasSampler(snap, n, depth = 0, seen = new Set()) {
    if (!n || depth > 12 || seen.has(n.key)) return false;
    seen.add(n.key);
    if (isSampler(n)) return true;
    return upstream(snap, n).some(({ node }) => upstreamHasSampler(snap, node, depth + 1, seen));
}

/** Where does a decoded image go? "final" (an output, no further sampling), "preview", or "". */
function decodeTarget(snap, n) {
    const seen = new Set([n.key]);
    let frontier = downstream(snap, n).map((x) => x.node);
    let final = false;
    let preview = false;
    for (let depth = 0; depth < 10 && frontier.length; depth++) {
        const next = [];
        for (const cur of frontier) {
            if (seen.has(cur.key)) continue;
            seen.add(cur.key);
            if (isSampler(cur) || isEncode(cur)) return "";
            if (isPreview(cur)) preview = true;
            else if (isOutput(cur)) final = true;
            for (const { node } of downstream(snap, cur)) next.push(node);
        }
        frontier = next;
    }
    return final ? "final" : preview ? "preview" : "";
}

// ── display-name cleanup for the generic fallback ──────────────────────────────────────────────────────────
const SUFFIXES = [/\s*\(rgthree\)\s*$/i, /\s*\(C2C\)\s*$/i, /\s*\(MEC\)\s*$/i, /\s+MEC$/, /\s*\|\s*pysssss\s*$/i,
    /\s+KJ$/, /\s*\(KJ\)\s*$/i, /\s*\[[^\]]*\]\s*$/, /\s*🔗\s*$/u];

export function cleanDisplayName(name) {
    let s = str(name).replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, "").trim();
    for (const re of SUFFIXES) s = s.replace(re, "").trim();
    if (!s) return "";
    if (!/\s/.test(s) && /[a-z][A-Z]/.test(s)) s = s.replace(/([a-z0-9])([A-Z])/g, "$1 $2");   // CamelCase -> words
    return s.replace(/_/g, " ").replace(/\s+/g, " ").trim();
}

const CATEGORY_STAGE = {
    loaders: "MODEL", conditioning: "PROMPT", sampling: "SAMPLE", latent: "LATENT", image: "IMAGE", mask: "MASK",
    video: "VIDEO", audio: "AUDIO", advanced: "UTIL", utils: "UTIL", _for_testing: "UTIL", api: "API",
};

export function stageFromCategory(category) {
    const parts = str(category).split("/").map((p) => p.trim()).filter(Boolean);
    if (!parts.length) return "NODE";
    const first = parts[0].replace(/[^\p{L}\p{N} _-]/gu, "").trim();
    if (/^c2c$/i.test(first) && parts.length >= 3) {
        const fam = parts[2].replace(/[^\p{L}\p{N} _-]/gu, "").trim().split(/[\s_-]+/)[0];
        if (fam) return fam.toUpperCase().slice(0, 10);
    }
    const low = first.toLowerCase();
    if (CATEGORY_STAGE[low]) return CATEGORY_STAGE[low];
    const w = first.split(/[\s_-]+/).filter(Boolean)[0] || "";
    return w ? w.toUpperCase().slice(0, 10) : "NODE";
}

// ── classification ──────────────────────────────────────────────────────────────────────────────────────────
export function classify(n, snapIn, rulesIn) {
    const snap = indexSnapshot(snapIn);
    const rules = rulesOf(rulesIn);
    const type = str(n.type);
    const outs = outTypes(n);
    const group = words(n.group, 3, false);

    if (isSampler(n)) {
        const fam = familyUpstream(snap, n) || "SAMPLE";
        const origin = latentOrigin(snap, n);
        const fn = { initial: "Initial Generation", refine: "Detail Refinement", upscale: "Upscale Refinement",
            img2img: "Image-to-Image" }[origin];
        return { stage: fam, function: fn, detail: group };
    }
    if (isEmptyLatent(n)) {
        const w = n.widgets || {};
        const size = Number.isFinite(+w.width) && Number.isFinite(+w.height) ? `${+w.width}x${+w.height}` : "";
        return { stage: "LATENT", function: "Canvas", detail: size || group };
    }
    if (isEncode(n)) return { stage: "LATENT", function: "Encode", detail: group };
    if (isDecode(n)) {
        const t = decodeTarget(snap, n);
        return { stage: "IMAGE", function: t === "final" ? "Final Decode" : t === "preview" ? "Preview Decode" : "Decode",
            detail: group };
    }
    if (isTextEncoder(n)) {
        const pol = polarity(snap, n);
        const detail = rules.use_prompt_words ? words(promptText(n), 3) : "";
        return { stage: "PROMPT", function: pol || "Conditioning", detail: detail || group };
    }
    if (isOutputish(n)) {
        if (isPreview(n)) return { stage: "PREVIEW", function: /video|anim/i.test(type) ? "Video" : "Image", detail: group };
        const video = /video|combine|anim|gif|mp4/i.test(type);
        const prefix = words(fileStem((n.widgets || {}).filename_prefix), 3, false);
        return { stage: "OUTPUT", function: video ? "Video Render" : "Final Render", detail: prefix || group };
    }
    if (isMediaLoader(n)) {
        const file = Object.entries(n.widgets || {}).find(([k, v]) => typeof v === "string"
            && /image|video|file|audio|path/i.test(k) && v);
        const kind = outs.includes("VIDEO") || /video/i.test(type) ? "Video" : outs.includes("AUDIO") ? "Audio" : "Image";
        const stem = file ? words(fileStem(file[1]), 3, false) : "";
        return { stage: "SOURCE", function: stem || kind, detail: stem ? kind : group };
    }
    if (outs.some((t) => ["MODEL", "CLIP", "VAE", "CONTROL_NET", "UPSCALE_MODEL", "CLIP_VISION"].includes(t))) {
        const file = modelFile(n);
        const fam = familyOf(file) || familyOf(type);
        const lora = /lora/i.test(type);
        const stage = lora ? "LORA" : outs.includes("CONTROL_NET") ? "CONTROL"
            : outs.includes("MODEL") ? "MODEL" : outs.includes("CLIP") ? "CLIP" : outs.includes("VAE") ? "VAE"
            : outs.includes("UPSCALE_MODEL") ? "UPSCALE" : "MODEL";
        const kind = lora ? "LoRA" : /checkpoint/i.test(type) ? "Checkpoint" : stage === "MODEL" ? "Model"
            : stage === "CONTROL" ? "ControlNet" : stage === "UPSCALE" ? "Upscale Model" : stage === "CLIP" ? "Text Encoder"
            : stage === "VAE" ? "VAE" : cleanDisplayName(n.defaultTitle || type);
        const fn = lora ? (words(fileStem(file), 3, false) || "LoRA") : [fam, kind].filter(Boolean).join(" ");
        return { stage, function: fn, detail: lora ? fam : words(fileStem(file), 3, false) };
    }
    // generic fallback: category stage + cleaned display name; a model family in the type / display name (e.g. a
    // MiniMax H3 sampler) is the stage instead, and is not repeated in the function
    const fam = familyOf(type) || familyOf(n.defaultTitle);
    let fn = cleanDisplayName(n.defaultTitle || type) || type || "Node";
    if (fam) {
        const lead = fn.match(/^(?:minimax\s+)?([A-Za-z0-9]+)\b[\s:._-]*/i);
        if (lead && familyOf(lead[0]) === fam) fn = fn.slice(lead[0].length).trim() || fn;
    }
    return { stage: fam || stageFromCategory(n.category), function: fn, detail: group };
}

// ── rendering ───────────────────────────────────────────────────────────────────────────────────────────────
export function render(template, parts, separator = DEFAULT_RULES.separator, maxLen = DEFAULT_RULES.max_len) {
    const tpl = str(template) || DEFAULT_RULES.template;
    const sep = str(separator) || DEFAULT_RULES.separator;
    const fill = (seg) => seg.replace(/\{(stage|function|detail|type)\}/g, (_m, k) => str(parts?.[k]).trim()).trim();
    const segs = tpl.split(sep).map(fill).filter((s) => s.length > 0);
    let out = segs.join(sep).replace(/\s+/g, " ").trim();
    if (out.length > maxLen) out = out.slice(0, Math.max(1, maxLen - 1)).trimEnd() + "…";
    return out;
}

// ── states ──────────────────────────────────────────────────────────────────────────────────────────────────
export function nodeState(n) {
    const title = str(n.title);
    const id = n.identity && typeof n.identity === "object" ? n.identity : null;
    if (id && id.source === "auto" && title === str(id.alias) && title) return "auto";
    if (id && id.source === "manual" && title === str(id.alias) && title) return "manual";
    if (!title || title === str(n.defaultTitle) || title === str(n.type)) return "default";
    return "manual";
}

export function isExcluded(n, rulesIn) {
    const rules = rulesOf(rulesIn);
    if (n.isVirtual || n.isSubgraph) return true;
    const t = str(n.type);
    if (ALWAYS_EXCLUDED.includes(t)) return true;
    return rules.excluded_types.map(str).includes(t);
}

/** Titles referenced by %Title.widget% in a string widget while no node carries that S&R name. */
export function referencedTitles(snapIn) {
    const snap = indexSnapshot(snapIn);
    const sr = new Set();
    for (const n of snap.nodes) if (n.srName) sr.add(str(n.srName));
    const refs = new Set();
    for (const n of snap.nodes) {
        for (const v of Object.values(n.widgets || {})) {
            if (typeof v !== "string" || v.indexOf("%") < 0) continue;
            for (const m of v.matchAll(/%([^%]+)%/g)) {
                const parts = m[1].split(".");
                if (parts.length === 2 && !sr.has(parts[0])) refs.add(parts[0]);
            }
        }
    }
    return refs;
}

// ── ordering: topological (upstream first), then pos x, pos y, key ─────────────────────────────────────────
function topoRank(snap) {
    const rank = new Map();
    const keys = snap.nodes.map((n) => String(n.key)).sort(cmp);
    const visiting = new Set();
    const depthOf = (k) => {
        if (rank.has(k)) return rank.get(k);
        if (visiting.has(k)) return 0;                     // cycle guard (graphs are DAGs; links may not be)
        visiting.add(k);
        const n = nodeOf(snap, k);
        let d = 0;
        for (const i of n?.inputs || []) {
            if (i.from == null || !snap.byKey.has(String(i.from))) continue;
            d = Math.max(d, depthOf(String(i.from)) + 1);
        }
        visiting.delete(k);
        rank.set(k, d);
        return d;
    };
    for (const k of keys) depthOf(k);
    return rank;
}

function stableOrder(snap, nodes) {
    const rank = topoRank(snap);
    return nodes.slice().sort((a, b) => (rank.get(String(a.key)) - rank.get(String(b.key)))
        || ((+a.pos?.[0] || 0) - (+b.pos?.[0] || 0)) || ((+a.pos?.[1] || 0) - (+b.pos?.[1] || 0))
        || cmp(str(a.path), str(b.path)) || cmp(String(a.key), String(b.key)));
}

const pad = (i) => String(i).padStart(2, "0");

function withNumber(base, i, maxLen) {
    const suffix = ` ${pad(i)}`;
    if (base.length + suffix.length <= maxLen) return base + suffix;
    return base.slice(0, Math.max(1, maxLen - suffix.length - 1)).trimEnd() + "…" + suffix;
}

function nextFree(base, taken, maxLen, start = 2) {
    for (let i = start; i < 10000; i++) {
        const cand = withNumber(base, i, maxLen);
        if (!taken.has(cand)) return cand;
    }
    return withNumber(base, 10000, maxLen);
}

// ── proposals ───────────────────────────────────────────────────────────────────────────────────────────────
/**
 * propose(snap, {scope, mode, rules}) -> rows sorted in stable order.
 * scope: Set/array of keys to rename, or null for every node. mode "rename" keeps existing auto aliases;
 * "regenerate" recomputes them. Manual aliases are kept unless rules.keep_manual is false.
 * Optional namer(node, parts) may return a string (AI hook); its answer goes through the same uniqueness rules.
 */
export function propose(snapIn, { scope = null, mode = "rename", rules: rulesIn = null, namer = null } = {}) {
    const snap = indexSnapshot(snapIn);
    const rules = rulesOf(rulesIn);
    const inScope = scope == null ? null : new Set([...scope].map(String));
    const refs = referencedTitles(snap);
    const rows = new Map();
    const candidates = [];

    for (const n of snap.nodes) {
        const key = String(n.key);
        const current = str(n.title);
        if (inScope && !inScope.has(key)) continue;
        const state = nodeState(n);
        if (isExcluded(n, rules)) { rows.set(key, { key, current, proposed: current, status: "excluded", reason: "excluded type" }); continue; }
        if (refs.has(current)) {
            rows.set(key, { key, current, proposed: current, status: "referenced",
                reason: `used as %${current}.…% in a text field` });
            continue;
        }
        if (state === "manual" && rules.keep_manual) { rows.set(key, { key, current, proposed: current, status: "keep-manual", reason: "your own name" }); continue; }
        if (state === "auto" && mode !== "regenerate") { rows.set(key, { key, current, proposed: current, status: "keep-auto", reason: "already named (use Regenerate)" }); continue; }
        candidates.push(n);
    }

    // names that stay: every node not being renamed (whole workflow)
    const renaming = new Set(candidates.map((n) => String(n.key)));
    const taken = new Set();
    for (const n of snap.nodes) if (!renaming.has(String(n.key)) && str(n.title)) taken.add(str(n.title));

    const info = new Map();
    for (const n of candidates) {
        const parts = classify(n, snap, rules);
        parts.type = cleanDisplayName(n.defaultTitle || n.type);
        let base = render(rules.template, parts, rules.separator, rules.max_len);
        if (typeof namer === "function") {
            try {
                const ai = namer(n, parts);
                if (typeof ai === "string" && ai.trim()) base = ai.trim().slice(0, rules.max_len);
            } catch (_e) { /* the deterministic name stays */ }
        }
        if (!base) base = cleanDisplayName(n.defaultTitle || n.type) || "Node";
        const withDetail = rules.template.includes("{detail}") ? base
            : render(rules.template + rules.separator + "{detail}", parts, rules.separator, rules.max_len);
        info.set(String(n.key), { base, withDetail, parts });
    }

    // group by base name, resolve collisions in stable order
    const groups = new Map();
    for (const n of stableOrder(snap, candidates)) {
        const b = info.get(String(n.key)).base;
        if (!groups.has(b)) groups.set(b, []);
        groups.get(b).push(n);
    }
    const final = new Map();
    const groupNames = [...groups.keys()].sort(cmp);
    const used = new Set(taken);
    // pass 1: lone names that are free claim them first (so numbering elsewhere never steals a plain name)
    for (const b of groupNames) {
        const members = groups.get(b);
        if (members.length === 1 && !used.has(b)) {
            final.set(String(members[0].key), b);
            used.add(b);
        }
    }
    for (const b of groupNames) {
        const members = groups.get(b);
        if (members.length === 1 && final.has(String(members[0].key))) continue;
        if (members.length === 1) {                                   // collides with a name that stays
            const name = nextFree(b, used, rules.max_len);
            final.set(String(members[0].key), name);
            used.add(name);
            continue;
        }
        // several: detail first, when it makes them all distinct and free
        const details = members.map((m) => info.get(String(m.key)).withDetail);
        const distinct = new Set(details).size === details.length && details.every((d) => d !== b && !used.has(d));
        if (distinct) {
            members.forEach((m, i) => { final.set(String(m.key), details[i]); used.add(details[i]); });
            continue;
        }
        let i = 1;
        for (const m of members) {
            let name = withNumber(b, i, rules.max_len);
            while (used.has(name)) name = withNumber(b, ++i, rules.max_len);
            final.set(String(m.key), name);
            used.add(name);
            i++;
        }
    }

    for (const n of candidates) {
        const key = String(n.key);
        const proposed = final.get(key);
        const current = str(n.title);
        const p = info.get(key).parts;
        rows.set(key, { key, current, proposed, status: proposed === current ? "unchanged" : "rename",
            reason: [p.stage, p.function, p.detail].filter(Boolean).join(" / ") });
    }
    const order = stableOrder(snap, snap.nodes.filter((n) => rows.has(String(n.key))));
    return order.map((n) => rows.get(String(n.key)));
}

/**
 * After a paste / duplicate: an AUTO alias on a new node that equals another node's title gets the next free
 * number. Manual aliases are reported (status "manual-duplicate") and left alone.
 */
export function resolvePasteCollisions(snapIn, newKeys, rulesIn = null) {
    const snap = indexSnapshot(snapIn);
    const rules = rulesOf(rulesIn);
    const fresh = new Set([...(newKeys || [])].map(String));
    const used = new Set();
    for (const n of snap.nodes) if (!fresh.has(String(n.key)) && str(n.title)) used.add(str(n.title));
    const out = [];
    for (const n of stableOrder(snap, snap.nodes.filter((x) => fresh.has(String(x.key))))) {
        const title = str(n.title);
        const state = nodeState(n);
        if (!title || !used.has(title)) { if (title) used.add(title); continue; }
        if (state !== "auto") {
            if (state === "manual") out.push({ key: String(n.key), current: title, proposed: title, status: "manual-duplicate" });
            continue;
        }
        const base = title.replace(/ \d{2,}$/, "");
        const name = nextFree(base, used, rules.max_len, 2);
        used.add(name);
        out.push({ key: String(n.key), current: title, proposed: name, status: "rename" });
    }
    return out;
}

/** Validate a manual alias: non-empty, at most max_len, unique in the workflow (other nodes' titles). */
export function checkManualAlias(snapIn, key, alias, rulesIn = null) {
    const snap = indexSnapshot(snapIn);
    const rules = rulesOf(rulesIn);
    const a = str(alias).replace(/\s+/g, " ").trim();
    if (!a) return { ok: false, reason: "The name is empty." };
    if (a.length > rules.max_len) return { ok: false, reason: `At most ${rules.max_len} characters.` };
    for (const n of snap.nodes) {
        if (String(n.key) !== String(key) && str(n.title) === a) return { ok: false, reason: `"${a}" is already used by another node.` };
    }
    return { ok: true, alias: a };
}
