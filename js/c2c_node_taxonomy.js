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

// ── Category → colour (dark theme hex; mirrors gregowahoo grouping) ───
// Keyed by node type. Prefix-match fallback then "default".
export const NODE_COLORS = {
    // Video I/O — teal
    "VHS_LoadVideo": "#0b3d36", "VHS_LoadVideoPath": "#0b3d36",
    "VHS_VideoCombine": "#0b3d36", "VHS_LoadImages": "#0b3d36",
    "LoadVideo": "#0b3d36", "SaveVideo": "#0b3d36", "VideoToImages": "#0b3d36",
    "ImagesToVideo": "#0b3d36",
    // Audio — slate blue
    "LoadAudio": "#1a2b3c", "SaveAudio": "#1a2b3c", "VHS_LoadAudio": "#1a2b3c",
    "LTXVAddAudio": "#1a2b3c", "EmptyAudioLatent": "#1a2b3c",
    // Image I/O — navy
    "LoadImage": "#0d2855", "SaveImage": "#0d2855", "PreviewImage": "#0d2855",
    "LoadImageMask": "#0d2855",
    // Model loaders — deep purple
    "CheckpointLoaderSimple": "#2d0d55", "UNETLoader": "#2d0d55",
    "CLIPLoader": "#2d0d55", "VAELoader": "#2d0d55", "LoraLoader": "#2d0d55",
    "LoraLoaderModelOnly": "#2d0d55", "DualCLIPLoader": "#2d0d55",
    "TripleCLIPLoader": "#2d0d55",
    // Samplers — burnt orange
    "KSampler": "#5c2800", "KSamplerAdvanced": "#5c2800",
    "SamplerCustomAdvanced": "#5c2800", "LTXVSampler": "#5c2800",
    "WanVideoSampler": "#5c2800", "HunyuanVideoSampler": "#5c2800",
    // CLIP / text encode — forest green
    "CLIPTextEncode": "#0d3d0d", "CLIPTextEncodeFlux": "#0d3d0d",
    "CLIPTextEncodeWan": "#0d3d0d", "CLIPTextEncodeLTXV": "#0d3d0d",
    "CLIPTextEncodeSD3": "#0d3d0d", "CLIPTextEncodeHunyuan": "#0d3d0d",
    // VAE — dark cyan
    "VAEDecode": "#003344", "VAEEncode": "#003344", "VAEEncodeForInpaint": "#003344",
    // Captioning / LLM — dark gold
    "Florence2": "#3d2e00", "WD14Tagger": "#3d2e00", "BLIPCaption": "#3d2e00",
    "JoyCaptionAlpha": "#3d2e00", "JoyCaption": "#3d2e00", "Moondream": "#3d2e00",
    "LLMChat": "#3d2e00", "OllamaGenerate": "#3d2e00", "QwenVL": "#3d2e00",
    "ImageToPrompt": "#3d2e00", "CaptionToPrompt": "#3d2e00", "CLIPInterrogator": "#3d2e00",
    // ControlNet — dark magenta
    "ControlNetLoader": "#3d003d", "ControlNetApply": "#3d003d",
    "ControlNetApplyAdvanced": "#3d003d", "DWPose_Preprocessor": "#3d003d",
    "OpenposePreprocessor": "#3d003d", "CannyEdgePreprocessor": "#3d003d",
    "DepthAnythingV2Preprocessor": "#3d003d", "LineArtPreprocessor": "#3d003d",
    // SAM / segmentation — dark rust
    "SAMModelLoader": "#3d1500", "SAMPredictor": "#3d1500",
    "GroundingDinoSAMSegment": "#3d1500", "SegmentAnything2": "#3d1500",
    // Mask / inpaint — dark maroon
    "GrowMask": "#2a1500", "GrowMaskWithBlur": "#2a1500", "MaskToImage": "#2a1500",
    "ImageToMask": "#2a1500", "InpaintModelConditioning": "#2a1500", "LanPaintNode": "#2a1500",
    // LTX video — dark teal-green
    "LTXVLoader": "#003333", "LTXVScheduler": "#003333",
    "LTXVConditioning": "#003333", "LTXVImgToVideo": "#003333",
    // Wan / Hunyuan video
    "WanVideoLoader": "#003028", "WanVideoEncode": "#003028", "HunyuanVideoLoader": "#002830",
    // Flux
    "FluxGuidance": "#1a1500", "ModelSamplingFlux": "#1a1500",
    // Upscale
    "ImageUpscaleWithModel": "#001a2a", "UpscaleModelLoader": "#001a2a",
    "UltimateSDUpscale": "#001a2a",
    // Qwen / Joy image edit — indigo
    "QwenImageEditLoader": "#1a0a3d", "QwenImageEdit": "#1a0a3d",
    "JoyAIImageEdit": "#1a0a3d", "JoyAILoader": "#1a0a3d",
    // IP-Adapter / Face — dark pink
    "IPAdapter": "#3d0025", "IPAdapterModelLoader": "#3d0025", "IPAdapterAdvanced": "#3d0025",
    "IPAdapterFaceID": "#3d0025", "PulidModelLoader": "#3d0025", "InstantIDModelLoader": "#3d0025",
    "ReActorFaceSwap": "#3d0025", "FaceRestoreWithModel": "#3d0025", "ImpactFaceDetailer": "#3d0025",
    // Reference / conditioning
    "ReferenceLatent": "#001840", "ConditioningConcat": "#001840", "ConditioningSetMask": "#001840",
    // Default
    "default": "#1a1a35",
};

// ── Data-type → wire colour (substring match) ─────────────────────────
export const LINK_COLORS = {
    "IMAGE": "#2a7a55", "LATENT": "#8a6a2a", "MODEL": "#6a3a9a",
    "CLIP": "#4a6a3a", "VAE": "#2a6a8a", "CONDITIONING": "#2a5a7a",
    "VIDEO": "#2a7a6a", "AUDIO": "#4a4a9a", "MASK": "#8a3a2a",
    "CONTROL_NET": "#7a3a7a", "STRING": "#3a6a6a", "INT": "#5a5a3a",
    "FLOAT": "#5a5a3a",
};

// Human-readable category labels for legends, keyed by representative hex.
export const CATEGORY_LEGEND = [
    ["#0b3d36", "Video I/O"], ["#1a2b3c", "Audio"], ["#0d2855", "Image I/O"],
    ["#2d0d55", "Model loaders"], ["#5c2800", "Samplers"], ["#0d3d0d", "CLIP/Text"],
    ["#003344", "VAE"], ["#3d2e00", "Caption/LLM"], ["#3d003d", "ControlNet"],
    ["#3d1500", "SAM/Segment"], ["#2a1500", "Mask/Inpaint"], ["#001a2a", "Upscale"],
    ["#1a0a3d", "Image edit"], ["#3d0025", "IP-Adapter/Face"], ["#1a1a35", "Utility"],
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

/** Node fill colour: exact, then prefix match, then default. */
export function nodeColor(type) {
    if (!type) return NODE_COLORS.default;
    if (NODE_COLORS[type]) return NODE_COLORS[type];
    for (const k of Object.keys(NODE_COLORS)) {
        if (k === "default") continue;
        if (k.length >= 5 && type.startsWith(k.slice(0, Math.min(8, k.length)))) {
            return NODE_COLORS[k];
        }
    }
    return NODE_COLORS.default;
}

/** Wire colour by data-type (substring, case-insensitive). */
export function linkColor(ltype) {
    const lt = String(ltype || "").toUpperCase();
    for (const k of Object.keys(LINK_COLORS)) {
        if (lt.includes(k)) return LINK_COLORS[k];
    }
    return "#3a3a6a";
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
    NODE_CAPS, NODE_COLORS, LINK_COLORS, CATEGORY_LEGEND,
    capabilityFor, nodeColor, linkColor, lighten,
};
