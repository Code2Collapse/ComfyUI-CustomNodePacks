// c2c_node_taxonomy.js — Shared node capability + colour taxonomy (C2C)
// ─────────────────────────────────────────────────────────────────────
// The "core architecture" foundation, modelled on gregowahoo's
// comfyui-workflow-finder (MIT): a node-type → capability-keyword map plus
// category/data-type colour maps. Consumed by:
//   • c2c_workflow_find.js   — rank graph nodes by *what they do*, not just
//                              their name (capability search).
//   • c2c_workflow_library.js — semantic search across a library of saved
//                              workflow .json files.
//   • c2c_node_explain.js    — categorise + colour a node in the explainer.
//   • any graph-preview canvas — colour nodes by category, wires by type.
//
// All maps are plain data so they can be extended without code changes.
// Lookups degrade gracefully: unknown node types fall back to a CamelCase
// split for capability text and a prefix match (then a default) for colour.
//
// License: Apache-2.0  (capability map seeded from gregowahoo's NODE_CAPS,
// MIT-licensed, then extended for C2C + common custom packs.)
// ─────────────────────────────────────────────────────────────────────

import { C } from "./_c2c_theme.js";

// ── Node-type → capability keyword string ────────────────────────────
// Keyword text is intentionally redundant/natural-language so a tokenised
// query like "make a video from an image" intersects with the right nodes.
export const NODE_CAPS = {
    // Video I/O
    "VHS_LoadVideo":               "load video read video input file clip",
    "VHS_LoadVideoPath":           "load video path file",
    "VHS_LoadImages":              "load images batch video frames sequence",
    "VHS_LoadImagePath":           "load image path file",
    "VHS_VideoCombine":            "save video export combine frames output mp4 webm gif",
    "VHS_LoadAudio":               "load audio sound music track",
    "LoadVideo":                   "load video read video input",
    "VideoPathLoader":             "load video path file",
    "SaveVideo":                   "save video export output",
    "VideoToImages":               "video extract frames split decode",
    "ImagesToVideo":               "images to video encode frames combine",

    // Audio
    "LoadAudio":                   "load audio sound music input",
    "SaveAudio":                   "save audio export output wav",
    "LTXVAddAudio":                "ltx audio video sync attach",
    "EmptyAudioLatent":            "audio latent generate empty",

    // Image I/O
    "LoadImage":                   "load image input file picture photo",
    "LoadImageMask":               "load image mask alpha channel",
    "SaveImage":                   "save image export output png",
    "PreviewImage":                "preview image show display view",
    "ImageBatch":                  "batch multiple images combine",
    "ImageListToImageBatch":       "image list batch convert",
    "RepeatLatentBatch":           "repeat batch latent duplicate",

    // Captioning / Vision LLM
    "Florence2":                   "caption describe image vision generate prompt florence ocr detect",
    "Florence2toCoordinates":      "florence caption detect coordinates bbox",
    "WD14Tagger":                  "tag caption tagger wd14 booru danbooru",
    "BLIPCaption":                 "blip caption describe image",
    "LLaVALoader":                 "llava vision language model load",
    "JoyCaptionAlpha":             "joycaption caption describe generate prompt",
    "JoyCaption":                  "joycaption caption describe",
    "Moondream":                   "moondream vision caption describe",
    "MoondreamBatchQueries":       "moondream vision caption batch queries",
    "QwenVL":                      "qwen vision language caption describe image prompt",
    "GPT4Vision":                  "gpt4 vision describe caption openai",
    "CLIPInterrogator":            "clip interrogate caption reverse prompt",
    "DeepDanbooru":                "danbooru tag caption anime",
    "ImageToPrompt":               "image to prompt generate caption convert",
    "CaptionToPrompt":             "caption to prompt convert generate",

    // LLM / prompt
    "LLMChat":                     "llm language model chat generate text prompt",
    "OllamaGenerate":              "ollama llm generate text prompt local",
    "LLMPromptGenerator":          "llm generate prompt enhance",
    "WanVideoPromptGenerator":     "wan video prompt generate",

    // CLIP / text encode
    "CLIPTextEncode":              "text prompt encode clip conditioning positive negative",
    "CLIPTextEncodeFlux":          "flux text prompt encode conditioning",
    "CLIPTextEncodeSD3":           "sd3 text prompt encode conditioning",
    "CLIPTextEncodeHunyuan":       "hunyuan text prompt encode conditioning",
    "CLIPTextEncodeWan":           "wan text prompt encode conditioning",
    "CLIPTextEncodeLTXV":          "ltx text prompt encode conditioning",

    // Model loaders
    "CheckpointLoaderSimple":      "checkpoint model sd sdxl load base",
    "UNETLoader":                  "unet model flux diffusion load",
    "CLIPLoader":                  "clip text encoder load",
    "DualCLIPLoader":              "dual clip load flux sd3",
    "TripleCLIPLoader":            "triple clip load sd3",
    "VAELoader":                   "vae load decoder encoder",
    "LoraLoader":                  "lora style fine-tune adapter load",
    "LoraLoaderModelOnly":         "lora model adapter load",

    // Samplers / scheduling
    "KSampler":                    "sample generate diffusion denoise steps cfg seed",
    "KSamplerAdvanced":            "sample generate advanced diffusion denoise",
    "SamplerCustomAdvanced":       "sample generate custom guider sigmas",
    "FluxGuidance":                "flux cfg guidance distilled",
    "ModelSamplingFlux":           "flux model sampling shift",

    // LTX / Wan / Hunyuan video models
    "LTXVLoader":                  "ltx ltx-video video generate load model",
    "LTXVSampler":                 "ltx ltx-video video generate sample",
    "LTXVScheduler":               "ltx ltx-video schedule sigmas",
    "LTXVConditioning":            "ltx conditioning frame rate",
    "LTXVImgToVideo":              "ltx image to video i2v",
    "WanVideoSampler":             "wan video generate sample",
    "WanVideoLoader":              "wan video model load",
    "WanVideoEncode":              "wan video encode latent",
    "HunyuanVideoSampler":         "hunyuan video generate sample",
    "HunyuanVideoLoader":          "hunyuan video model load",

    // ControlNet / preprocessors
    "ControlNetLoader":            "controlnet control load model",
    "ControlNetApply":             "controlnet apply control conditioning",
    "ControlNetApplyAdvanced":     "controlnet apply advanced control conditioning",
    "DWPose_Preprocessor":         "pose dwpose controlnet skeleton openpose",
    "OpenposePreprocessor":        "openpose pose skeleton controlnet",
    "CannyEdgePreprocessor":       "canny edge lines controlnet",
    "DepthAnythingV2Preprocessor": "depth depthmap controlnet anything",
    "LineArtPreprocessor":         "lineart lines controlnet sketch",

    // IP-Adapter / Face
    "IPAdapter":                   "ip-adapter style transfer face reference image",
    "IPAdapterModelLoader":        "ip-adapter load model",
    "IPAdapterAdvanced":           "ip-adapter advanced style reference",
    "IPAdapterFaceID":             "ip-adapter face id identity reference",
    "FaceRestoreWithModel":        "face restore fix enhance gfpgan codeformer",
    "ReActorFaceSwap":             "face swap reactor identity",
    "PulidModelLoader":            "pulid face id consistency identity load",
    "InstantIDModelLoader":        "instantid face id consistency identity load",
    "ImpactFaceDetailer":          "face detail fix refine impact",

    // SAM / segmentation
    "SAMModelLoader":              "sam segment mask load model",
    "SAMPredictor":                "sam segment mask detect predict",
    "GroundingDinoSAMSegment":     "grounding dino segment detect object mask text",
    "SegmentAnything2":            "sam2 segment mask video tracking",

    // Mask / inpaint
    "InpaintModelConditioning":    "inpaint fill mask repair conditioning",
    "VAEEncodeForInpaint":         "inpaint vae encode latent mask",
    "GrowMask":                    "mask grow expand dilate",
    "GrowMaskWithBlur":            "mask grow blur feather expand",
    "MaskToImage":                 "mask image convert",
    "ImageToMask":                 "image mask convert channel",
    "LanPaintNode":                "lanpaint inpaint fill",

    // Upscale
    "ImageUpscaleWithModel":       "upscale super resolution enhance esrgan",
    "UpscaleModelLoader":          "upscale model load esrgan",
    "UltimateSDUpscale":           "upscale ultimate tile sd hires",
    "LatentUpscale":               "latent upscale resize hires",
    "ImageScale":                  "image resize scale resample",

    // Qwen / Joy image edit
    "QwenImageEditLoader":         "qwen image edit load model",
    "QwenImageEdit":               "qwen image edit modify transform instruction",
    "JoyAIImageEdit":              "joyai image edit modify",
    "JoyAILoader":                 "joyai load model",

    // Latent / VAE / conditioning utility
    "ReferenceLatent":             "reference latent conditioning consistent character",
    "VAEDecode":                   "vae decode image latent",
    "VAEEncode":                   "vae encode latent image",
    "EmptyLatentImage":            "empty latent image canvas size",
    "ImageCrop":                   "image crop cut region",
    "ConditioningConcat":          "conditioning combine concat merge",
    "ConditioningSetMask":         "conditioning mask region area",
    "ImpactWildcardProcessor":     "wildcard prompt dynamic random",
};

// ── Category → palette hue key (mirrors gregowahoo grouping) ───────────
// Keyed by node type. Prefix-match fallback then "default".
// Values are C2C palette hue keys — resolved at runtime via nodeColor().
export const NODE_HUES = {
    // Video I/O — teal
    "VHS_LoadVideo": "teal", "VHS_LoadVideoPath": "teal",
    "VHS_VideoCombine": "teal", "VHS_LoadImages": "teal",
    "LoadVideo": "teal", "SaveVideo": "teal", "VideoToImages": "teal",
    "ImagesToVideo": "teal",
    // Audio — sapphire
    "LoadAudio": "sapphire", "SaveAudio": "sapphire", "VHS_LoadAudio": "sapphire",
    "LTXVAddAudio": "sapphire", "EmptyAudioLatent": "sapphire",
    // Image I/O — blue
    "LoadImage": "blue", "SaveImage": "blue", "PreviewImage": "blue",
    "LoadImageMask": "blue",
    // Model loaders — mauve
    "CheckpointLoaderSimple": "mauve", "UNETLoader": "mauve",
    "CLIPLoader": "mauve", "VAELoader": "mauve", "LoraLoader": "mauve",
    "LoraLoaderModelOnly": "mauve", "DualCLIPLoader": "mauve",
    "TripleCLIPLoader": "mauve",
    // Samplers — peach
    "KSampler": "peach", "KSamplerAdvanced": "peach",
    "SamplerCustomAdvanced": "peach", "LTXVSampler": "peach",
    "WanVideoSampler": "peach", "HunyuanVideoSampler": "peach",
    // CLIP / text encode — green
    "CLIPTextEncode": "green", "CLIPTextEncodeFlux": "green",
    "CLIPTextEncodeWan": "green", "CLIPTextEncodeLTXV": "green",
    "CLIPTextEncodeSD3": "green", "CLIPTextEncodeHunyuan": "green",
    // VAE — sapphire
    "VAEDecode": "sapphire", "VAEEncode": "sapphire", "VAEEncodeForInpaint": "sapphire",
    // Captioning / LLM — yellow
    "Florence2": "yellow", "WD14Tagger": "yellow", "BLIPCaption": "yellow",
    "JoyCaptionAlpha": "yellow", "JoyCaption": "yellow", "Moondream": "yellow",
    "LLMChat": "yellow", "OllamaGenerate": "yellow", "QwenVL": "yellow",
    "ImageToPrompt": "yellow", "CaptionToPrompt": "yellow", "CLIPInterrogator": "yellow",
    // ControlNet — pink
    "ControlNetLoader": "pink", "ControlNetApply": "pink",
    "ControlNetApplyAdvanced": "pink", "DWPose_Preprocessor": "pink",
    "OpenposePreprocessor": "pink", "CannyEdgePreprocessor": "pink",
    "DepthAnythingV2Preprocessor": "pink", "LineArtPreprocessor": "pink",
    // SAM / segmentation — peach
    "SAMModelLoader": "peach", "SAMPredictor": "peach",
    "GroundingDinoSAMSegment": "peach", "SegmentAnything2": "peach",
    // Mask / inpaint — red
    "GrowMask": "red", "GrowMaskWithBlur": "red", "MaskToImage": "red",
    "ImageToMask": "red", "InpaintModelConditioning": "red", "LanPaintNode": "red",
    // LTX video — teal
    "LTXVLoader": "teal", "LTXVScheduler": "teal",
    "LTXVConditioning": "teal", "LTXVImgToVideo": "teal",
    // Wan / Hunyuan video — teal
    "WanVideoLoader": "teal", "WanVideoEncode": "teal", "HunyuanVideoLoader": "teal",
    // Flux — yellow
    "FluxGuidance": "yellow", "ModelSamplingFlux": "yellow",
    // Upscale — sky
    "ImageUpscaleWithModel": "sky", "UpscaleModelLoader": "sky",
    "UltimateSDUpscale": "sky",
    // Qwen / Joy image edit — lavender
    "QwenImageEditLoader": "lavender", "QwenImageEdit": "lavender",
    "JoyAIImageEdit": "lavender", "JoyAILoader": "lavender",
    // IP-Adapter / Face — pink
    "IPAdapter": "pink", "IPAdapterModelLoader": "pink", "IPAdapterAdvanced": "pink",
    "IPAdapterFaceID": "pink", "PulidModelLoader": "pink", "InstantIDModelLoader": "pink",
    "ReActorFaceSwap": "pink", "FaceRestoreWithModel": "pink", "ImpactFaceDetailer": "pink",
    // Reference / conditioning — blue
    "ReferenceLatent": "blue", "ConditioningConcat": "blue", "ConditioningSetMask": "blue",
    // Default
    "default": "lavender",
};

/** @deprecated Use NODE_HUES — kept for callers that import the old name. */
export const NODE_COLORS = NODE_HUES;

// ── Data-type → wire hue (substring match) ────────────────────────────
export const LINK_HUES = {
    "IMAGE": "green", "LATENT": "yellow", "MODEL": "mauve",
    "CLIP": "green", "VAE": "sapphire", "CONDITIONING": "blue",
    "VIDEO": "teal", "AUDIO": "lavender", "MASK": "peach",
    "CONTROL_NET": "pink", "STRING": "teal", "INT": "yellow",
    "FLOAT": "yellow",
};

/** @deprecated Use LINK_HUES */
export const LINK_COLORS = LINK_HUES;

/** Hue percent mixed into bg for category chips (canvas + CSS). */
const CHIP_MIX = 22;
const WIRE_MIX = 38;

// Human-readable category labels for legends, keyed by palette hue.
export const CATEGORY_LEGEND = [
    ["teal", "Video I/O"], ["sapphire", "Audio"], ["blue", "Image I/O"],
    ["mauve", "Model loaders"], ["peach", "Samplers"], ["green", "CLIP/Text"],
    ["sapphire", "VAE"], ["yellow", "Caption/LLM"], ["pink", "ControlNet"],
    ["peach", "SAM/Segment"], ["red", "Mask/Inpaint"], ["sky", "Upscale"],
    ["lavender", "Image edit"], ["pink", "IP-Adapter/Face"], ["lavender", "Utility"],
];

// ── Helpers ───────────────────────────────────────────────────────────

/** Split a CamelCase / snake_case node type into lowercase words. */
function splitTypeWords(type) {
    return String(type || "")
        .replace(/[_\-]+/g, " ")
        .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
        .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
        .toLowerCase()
        .trim();
}

/**
 * Capability keyword text for a node type. Falls back to a CamelCase split
 * of the type name when the type is not in NODE_CAPS, so every node is at
 * least searchable by the words in its class name.
 */
export function capabilityFor(type) {
    if (!type) return "";
    if (NODE_CAPS[type]) return NODE_CAPS[type];
    return splitTypeWords(type);
}

/** Resolve palette hue key for a node type. */
export function nodeHue(type) {
    if (!type) return NODE_HUES.default;
    if (NODE_HUES[type]) return NODE_HUES[type];
    for (const k of Object.keys(NODE_HUES)) {
        if (k === "default") continue;
        if (k.length >= 5 && type.startsWith(k.slice(0, Math.min(8, k.length)))) {
            return NODE_HUES[k];
        }
    }
    return NODE_HUES.default;
}

function _hex2rgb(h) {
    let s = String(h || "").replace("#", "");
    if (s.length === 3) s = s.split("").map((c) => c + c).join("");
    return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16) / 255);
}

function _rgb2hex(c) {
    return "#" + c.map((v) => {
        const n = Math.max(0, Math.min(255, Math.round(v * 255)));
        return n.toString(16).padStart(2, "0");
    }).join("");
}

/** Mix a palette hue into the active ground (canvas — read C at call time). */
function chipFill(hueKey, mix = CHIP_MIX) {
    const hue = C[hueKey];
    if (!hue) return C.panelBg;
    try {
        const t = mix / 100;
        const bg = _hex2rgb(C.bg);
        const fg = _hex2rgb(hue);
        return _rgb2hex(bg.map((b, i) => b * (1 - t) + fg[i] * t));
    } catch {
        return C.panelBg;
    }
}

/** CSS color-mix for category chips (live across variant switches). */
export function nodeColorCss(type, mix = CHIP_MIX) {
    const h = nodeHue(type);
    return `color-mix(in srgb, var(--c2c-${h}) ${mix}%, var(--c2c-bg) ${100 - mix}%)`;
}

/** Node fill colour for canvas: exact, prefix match, then default. */
export function nodeColor(type) {
    return chipFill(nodeHue(type));
}

/** Wire colour by data-type (substring, case-insensitive). */
export function linkColor(ltype) {
    const lt = String(ltype || "").toUpperCase();
    for (const k of Object.keys(LINK_HUES)) {
        if (lt.includes(k)) return chipFill(LINK_HUES[k], WIRE_MIX);
    }
    return chipFill("lavender", 18);
}

/** Lighten a #rrggbb hex by `amt` per channel (header/border tints). */
export function lighten(hx, amt = 30) {
    try {
        const r = Math.min(parseInt(hx.slice(1, 3), 16) + amt, 255);
        const g = Math.min(parseInt(hx.slice(3, 5), 16) + amt, 255);
        const b = Math.min(parseInt(hx.slice(5, 7), 16) + amt, 255);
        const h = (v) => v.toString(16).padStart(2, "0");
        return `#${h(r)}${h(g)}${h(b)}`;
    } catch {
        return hx;
    }
}

export default {
    NODE_CAPS, NODE_HUES, NODE_COLORS, LINK_HUES, LINK_COLORS, CATEGORY_LEGEND,
    capabilityFor, nodeHue, nodeColor, nodeColorCss, linkColor, lighten,
};
