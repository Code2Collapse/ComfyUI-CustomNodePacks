/**
 * c2c_settings_recommender.js — rule-based (NO AI) recommended sampler /
 * scheduler / cfg / steps for the model(s) in the current workflow.
 *
 * Why: ComfyUI's inspector shows no guidance on what sampler/scheduler/cfg a
 * given checkpoint actually wants. This analyses the graph — finds the loaded
 * model file name(s) + the sampler node(s) — and recommends settings from a
 * curated rule table covering the common ComfyUI model families. Shown:
 *   • a sidebar tab "⚙ Recommended" (per detected model)
 *   • a right-click item on KSampler-type nodes: "⚙ Recommended settings"
 *     (lists the rec and offers a one-click Apply)
 *
 * Pure heuristics on the model FILENAME — zero network, zero AI.
 */

import { app } from "../../scripts/app.js";

// ── Rule table ───────────────────────────────────────────────────────────────
// Each rule: { test(name) -> bool, label, sampler, scheduler, cfg, steps, note }.
// First match wins; ordered specific → general. cfg/steps are typical ranges.
// "name" is the lower-cased model filename.
const RULES = [
    // --- speed-tuned variants (check BEFORE base families) ---
    { test: n => /lightning/.test(n),                 label: "Lightning (distilled)", sampler: "euler", scheduler: "sgm_uniform", cfg: "1.0–2.0", steps: "4–8",  note: "Distilled few-step. Low CFG, very low steps." },
    { test: n => /\bturbo\b|sdxl[-_]?turbo/.test(n),  label: "Turbo (distilled)",     sampler: "euler_ancestral", scheduler: "karras", cfg: "1.0–2.0", steps: "1–6", note: "Turbo: 1–6 steps, CFG ~1–2." },
    { test: n => /\blcm\b/.test(n),                   label: "LCM",                   sampler: "lcm", scheduler: "sgm_uniform", cfg: "1.0–2.0", steps: "4–8", note: "Use the LCM sampler; needs an LCM LoRA/model." },
    { test: n => /hyper[-_]?sd|hyper\b/.test(n),      label: "Hyper-SD",              sampler: "euler", scheduler: "sgm_uniform", cfg: "1.0", steps: "1–8", note: "Hyper-SD distilled. CFG 1, ultra-low steps." },

    // --- video models ---
    { test: n => /wan2\.?2|wan22/.test(n),            label: "Wan 2.2 (video)",       sampler: "uni_pc", scheduler: "simple", cfg: "5.0–6.0", steps: "20–30", note: "Wan 2.2. uni_pc/euler, shift≈5–8 on the model sampler." },
    { test: n => /wan2\.?1|wan21|\bwan\b/.test(n),    label: "Wan 2.1 (video)",       sampler: "uni_pc", scheduler: "simple", cfg: "5.0–6.0", steps: "20–30", note: "Wan 2.1. uni_pc/euler; tune shift on ModelSamplingSD3." },
    { test: n => /hunyuan.*video|hyvideo/.test(n),    label: "HunyuanVideo",          sampler: "euler", scheduler: "simple", cfg: "6.0", steps: "20–30", note: "HunyuanVideo: euler/simple." },
    { test: n => /ltx|ltxv/.test(n),                  label: "LTX-Video",             sampler: "euler", scheduler: "normal", cfg: "3.0", steps: "20–30", note: "LTXV: low CFG (~3)." },
    { test: n => /\bsvd\b|stable[-_]?video/.test(n),  label: "Stable Video Diffusion",sampler: "euler", scheduler: "karras", cfg: "2.5–3.0", steps: "25", note: "SVD: low CFG, motion via augmentation." },
    { test: n => /cogvideo/.test(n),                  label: "CogVideoX",             sampler: "euler", scheduler: "simple", cfg: "6.0", steps: "50", note: "CogVideoX." },

    // --- image models ---
    { test: n => /flux/.test(n),                      label: "FLUX",                  sampler: "euler", scheduler: "simple", cfg: "1.0", steps: "20–30", note: "FLUX: CFG 1.0 (use FluxGuidance ≈3.5); euler/simple or beta." },
    { test: n => /sd3\.?5|sd35/.test(n),              label: "SD 3.5",                sampler: "euler", scheduler: "sgm_uniform", cfg: "4.0–5.0", steps: "28–40", note: "SD3.5: sgm_uniform; CFG 4–5." },
    { test: n => /sd3|stable[-_]?diffusion[-_]?3/.test(n), label: "SD 3",            sampler: "euler", scheduler: "sgm_uniform", cfg: "4.0–5.0", steps: "28", note: "SD3: sgm_uniform; CFG 4–5." },
    { test: n => /pony/.test(n),                      label: "Pony (SDXL)",           sampler: "euler_ancestral", scheduler: "karras", cfg: "6.0–7.0", steps: "25–30", note: "Pony: needs score_9… tags; euler_a/karras." },
    { test: n => /playground/.test(n),                label: "Playground v2.5",       sampler: "dpmpp_2m", scheduler: "karras", cfg: "3.0", steps: "30", note: "Playground v2.5: low CFG (~3), EDM." },
    { test: n => /illustrious|noobai|animagine/.test(n), label: "Illustrious/NoobAI (SDXL anime)", sampler: "euler_ancestral", scheduler: "karras", cfg: "5.0–7.0", steps: "28", note: "SDXL anime checkpoints." },
    { test: n => /sdxl|xl[-_]|juggernaut.*xl|dreamshaper.*xl|_xl\b/.test(n), label: "SDXL", sampler: "dpmpp_2m", scheduler: "karras", cfg: "6.0–8.0", steps: "25–30", note: "SDXL base: dpmpp_2m/karras; CFG 6–8. Add a refiner if you have one." },
    { test: n => /sd[-_]?1\.?5|v1-5|dreamshaper(?!.*xl)|deliberate|realisticvision/.test(n), label: "SD 1.5", sampler: "dpmpp_2m", scheduler: "karras", cfg: "7.0–8.0", steps: "20–30", note: "SD1.5: dpmpp_2m or euler_a; CFG 7–8." },
    { test: n => /sd[-_]?2|768/.test(n),              label: "SD 2.x",                sampler: "dpmpp_2m", scheduler: "karras", cfg: "7.0", steps: "25–30", note: "SD2.x (768): v-prediction on some checkpoints." },
];

const FALLBACK = { label: "Unknown / generic", sampler: "dpmpp_2m", scheduler: "karras", cfg: "7.0", steps: "25", note: "Couldn't identify the model family from the filename — generic SD defaults. Rename the checkpoint or check the model card." };

function recFor(modelName) {
    const n = String(modelName || "").toLowerCase();
    for (const r of RULES) { try { if (r.test(n)) return r; } catch (_) {} }
    return FALLBACK;
}

// ── DEEP detail per family (the "deep as the ocean" layer) ───────────────────
// Keyed by rule.label. Resolution / VAE / CLIP / negative-prompt / refiner /
// alternatives / common pitfalls — everything the inspector should tell you.
const DEEP = {
    "SDXL": { res:"1024×1024 (or 896×1152, 832×1216, 1216×832…)", vae:"sdxl_vae.safetensors (fixes washed-out colors on some checkpoints)", clip:"—", neg:"short neg works; avoid huge neg lists", alts:"Alt samplers: dpmpp_2m_sde/karras, euler_a. Add a refiner pass (last ~20% steps) for crisp detail.", pitfalls:"Don't run at 512 (SDXL was trained at 1024 → bad anatomy). CFG >9 burns." },
    "SD 1.5": { res:"512×512 / 512×768 / 768×512", vae:"vae-ft-mse-840000 for realism", clip:"clip skip 1 (anime checkpoints often want 2)", neg:"benefits from a quality neg (worst quality, low quality…)", alts:"euler_a, dpmpp_2m_sde. Hires-fix ×1.5–2 for >512.", pitfalls:"Native 512 — go bigger only via hires-fix or you get duplicated faces/limbs." },
    "FLUX": { res:"1024×1024 (1MP class; 1344×768 etc.)", vae:"ae.safetensors (FLUX VAE)", clip:"needs t5xxl + clip_l (DualCLIPLoader)", neg:"NO negative prompt (CFG 1 = neg ignored). Use FluxGuidance node ≈3.0–4.0 instead.", alts:"Scheduler: simple or beta. dev=guidance-distilled; schnell=4 steps, guidance off.", pitfalls:"Real CFG must stay 1.0 (raising it doubles compute + breaks dev). Guidance ≠ CFG." },
    "SD 3.5": { res:"1024×1024", vae:"built-in (16-ch)", clip:"clip_g + clip_l + t5xxl (TripleCLIPLoader)", neg:"supports neg at CFG 4–5", alts:"sgm_uniform scheduler; dpmpp_2m also fine.", pitfalls:"Needs the 3 text encoders wired or output is garbage." },
    "SD 3": { res:"1024×1024", vae:"built-in (16-ch)", clip:"clip_g + clip_l + t5xxl", neg:"CFG 4–5", alts:"sgm_uniform.", pitfalls:"Triple CLIP required." },
    "Pony (SDXL)": { res:"1024×1024", vae:"sdxl_vae", clip:"clip skip 2", neg:"REQUIRES score tags: positive 'score_9, score_8_up, score_7_up…'; neg 'score_4, score_3…'", alts:"euler_a/karras; dpmpp_2m_sde.", pitfalls:"Without the score_ prefix the output looks broken. It's an SDXL finetune." },
    "Illustrious/NoobAI (SDXL anime)": { res:"1024×1024 / 832×1216", vae:"sdxl_vae", clip:"clip skip 2", neg:"booru-tag style; 'worst quality, lowres' neg", alts:"euler_a/karras.", pitfalls:"Danbooru tag prompting; quality tags matter." },
    "Playground v2.5": { res:"1024×1024", vae:"built-in EDM VAE", clip:"—", neg:"low CFG ~3", alts:"dpmpp_2m/karras.", pitfalls:"EDM model — keep CFG ~3, higher washes out." },
    "Wan 2.2 (video)": { res:"832×480 or 1280×720 (16:9); length 49/81 frames", vae:"wan_2.1_vae.safetensors", clip:"umt5_xxl text encoder", neg:"supports neg; keep concise", alts:"uni_pc/euler; set shift≈5–8 on ModelSamplingSD3. I2V vs T2V variants.", pitfalls:"VRAM heavy — use the fp8/GGUF + block-swap on <24GB. Match the VAE to the Wan version." },
    "Wan 2.1 (video)": { res:"832×480 / 1280×720", vae:"wan_2.1_vae", clip:"umt5_xxl", neg:"concise neg", alts:"uni_pc/euler; shift on ModelSamplingSD3.", pitfalls:"VRAM heavy; fp8/GGUF + block-swap for low VRAM." },
    "HunyuanVideo": { res:"720×1280 / 960×544; ~the model's native frame counts", vae:"hunyuan_video VAE", clip:"llava llama3 + clip_l", neg:"—", alts:"euler/simple.", pitfalls:"Very VRAM heavy; use fp8 + tiling." },
    "LTX-Video": { res:"768×512 class; 24–48fps", vae:"ltxv VAE", clip:"t5xxl", neg:"supports neg", alts:"euler/normal; low CFG (~3).", pitfalls:"Needs the LTXV-specific nodes/conditioning." },
    "Stable Video Diffusion": { res:"1024×576 (img2vid)", vae:"svd VAE", clip:"—", neg:"—", alts:"euler/karras.", pitfalls:"Motion via motion_bucket_id + augmentation_level, not CFG. Image-driven only." },
    "CogVideoX": { res:"720×480", vae:"cogvideox VAE", clip:"t5xxl", neg:"—", alts:"euler/simple.", pitfalls:"50 steps typical; VRAM heavy." },
    "Lightning (distilled)": { res:"match the base (SDXL→1024)", vae:"base VAE", clip:"—", neg:"keep minimal (low CFG ignores most neg)", alts:"euler/sgm_uniform; some want dpmpp_sde. 2/4/8-step variants exist.", pitfalls:"Use the LoRA/merge that MATCHES your step count. High CFG destroys distilled models." },
    "Turbo (distilled)": { res:"512–1024 (SDXL-Turbo → 512–1024)", vae:"base VAE", clip:"—", neg:"minimal", alts:"euler_a/karras.", pitfalls:"1–6 steps, CFG ~1. Not for high-detail finals." },
    "LCM": { res:"match base", vae:"base VAE", clip:"—", neg:"minimal", alts:"lcm sampler only.", pitfalls:"Requires the LCM LoRA or an LCM-baked checkpoint." },
    "Hyper-SD": { res:"match base", vae:"base VAE", clip:"—", neg:"minimal", alts:"euler/sgm_uniform.", pitfalls:"Match the Hyper LoRA to step count; CFG 1." },
    "SD 2.x": { res:"768×768", vae:"built-in", clip:"—", neg:"quality neg helps", alts:"dpmpp_2m/karras.", pitfalls:"Some 2.x are v-prediction → needs the v-pred config or output is gray." },
};
function deepFor(label) { return DEEP[label] || null; }

// Expose to the C2C node-info inspector (c2c_node_explain.js) so the
// recommendation renders INSIDE that panel — not the native inspector, not a
// separate sidebar-only widget.
try {
    window.__C2C_REC = {
        recFor, deepFor, detectedModels,
        modelForSampler: (node, models) => (models && models.length ? models[0].file : null),
        isSamplerNode: (node) => /KSampler|SamplerCustom|sampler/i.test(node?.comfyClass || node?.type || ""),
    };
} catch (_) {}

// ── Workflow analysis ────────────────────────────────────────────────────────
// Widget names that hold a model filename across the common loaders.
const MODEL_WIDGETS = ["ckpt_name", "unet_name", "model_name", "checkpoint", "model"];
const LOADER_HINT = /checkpoint|unet|loader|gguf|diffusion/i;

function detectedModels() {
    const out = [];
    const graph = app.graph;
    if (!graph || !graph._nodes) return out;
    for (const node of graph._nodes) {
        const cls = node.comfyClass || node.type || "";
        if (!LOADER_HINT.test(cls)) continue;
        for (const w of (node.widgets || [])) {
            if (MODEL_WIDGETS.includes(w.name) && typeof w.value === "string" && /\.(safetensors|ckpt|gguf|pt|pth|sft)$/i.test(w.value)) {
                out.push({ node, cls, file: w.value });
            }
        }
    }
    return out;
}

const SAMPLER_CLASS = /KSampler|SamplerCustom|sampler/i;
function isSamplerNode(node) {
    return SAMPLER_CLASS.test(node?.comfyClass || node?.type || "");
}

// Pick the model most likely feeding a given sampler (nearest upstream model),
// else the first detected model in the graph.
function modelForSampler(_node, models) {
    return models.length ? models[0].file : null;
}

// ── Sidebar panel ─────────────────────────────────────────────────────────────
function renderPanel(container) {
    const C = { bg: "#1e1e2e", card: "#181825", text: "#cdd6f4", dim: "#9399b2",
        accent: "#89b4fa", green: "#a6e3a1", border: "#313244", mono: "ui-monospace,monospace" };
    container.innerHTML = "";
    container.style.cssText = `padding:10px;color:${C.text};font:12px ui-sans-serif;overflow:auto;height:100%;background:${C.bg};`;
    const models = detectedModels();
    const h = document.createElement("div");
    h.style.cssText = `font-weight:600;font-size:13px;margin-bottom:8px;`;
    h.textContent = "⚙ Recommended settings";
    container.appendChild(h);
    const sub = document.createElement("div");
    sub.style.cssText = `color:${C.dim};margin-bottom:10px;`;
    sub.textContent = "Rule-based (no AI) — derived from the model filename(s) in this workflow.";
    container.appendChild(sub);

    if (!models.length) {
        const e = document.createElement("div");
        e.style.cssText = `color:${C.dim};`;
        e.textContent = "No checkpoint / UNET loader with a model selected yet. Add one and reopen this tab.";
        container.appendChild(e);
        return;
    }
    const seen = new Set();
    for (const m of models) {
        if (seen.has(m.file)) continue; seen.add(m.file);
        const r = recFor(m.file);
        const card = document.createElement("div");
        card.style.cssText = `background:${C.card};border:1px solid ${C.border};border-radius:6px;padding:8px 10px;margin-bottom:8px;`;
        const fam = document.createElement("div");
        fam.style.cssText = `font-weight:600;color:${C.accent};margin-bottom:2px;`;
        fam.textContent = r.label;
        const file = document.createElement("div");
        file.style.cssText = `color:${C.dim};font:11px ${C.mono};word-break:break-all;margin-bottom:6px;`;
        file.textContent = m.file;
        const rows = [
            ["sampler", r.sampler], ["scheduler", r.scheduler],
            ["cfg", r.cfg], ["steps", r.steps],
        ];
        const grid = document.createElement("div");
        grid.style.cssText = `display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin-bottom:6px;`;
        for (const [k, v] of rows) {
            const kk = document.createElement("div"); kk.style.cssText = `color:${C.dim};`; kk.textContent = k;
            const vv = document.createElement("div"); vv.style.cssText = `color:${C.green};font:12px ${C.mono};`; vv.textContent = v;
            grid.append(kk, vv);
        }
        const note = document.createElement("div");
        note.style.cssText = `color:${C.dim};font-size:11px;line-height:1.35;margin-bottom:6px;`;
        note.textContent = r.note;
        card.append(fam, file, grid, note);
        // Deep detail block
        const dd = deepFor(r.label);
        if (dd) {
            const deepRows = [
                ["resolution", dd.res], ["VAE", dd.vae], ["CLIP/encoder", dd.clip],
                ["negative", dd.neg], ["alternatives", dd.alts], ["⚠ pitfalls", dd.pitfalls],
            ].filter(([, v]) => v && v !== "—");
            const dgrid = document.createElement("div");
            dgrid.style.cssText = `display:grid;grid-template-columns:auto 1fr;gap:3px 10px;border-top:1px solid ${C.border};padding-top:6px;`;
            for (const [k, v] of deepRows) {
                const kk = document.createElement("div"); kk.style.cssText = `color:${C.dim};font-size:11px;white-space:nowrap;`; kk.textContent = k;
                const vv = document.createElement("div"); vv.style.cssText = `color:${C.text};font-size:11px;line-height:1.35;`; vv.textContent = v;
                dgrid.append(kk, vv);
            }
            card.appendChild(dgrid);
        }
        container.appendChild(card);
    }
}

// ── Apply to a sampler node ─────────────────────────────────────────────────
function applyTo(node, rec) {
    const setW = (name, val) => { const w = (node.widgets || []).find(w => w.name === name); if (w) { w.value = val; w.callback?.(val, app.canvas, node); } };
    // cfg/steps are ranges → take the lower bound as a safe starting point.
    const lo = (s) => { const m = String(s).match(/[\d.]+/); return m ? Number(m[0]) : undefined; };
    const samplerW = (node.widgets || []).find(w => w.name === "sampler_name");
    const schedW = (node.widgets || []).find(w => w.name === "scheduler");
    if (samplerW && (samplerW.options?.values || []).includes(rec.sampler)) setW("sampler_name", rec.sampler);
    if (schedW && (schedW.options?.values || []).includes(rec.scheduler)) setW("scheduler", rec.scheduler);
    const c = lo(rec.cfg); if (c !== undefined) setW("cfg", c);
    const st = lo(rec.steps); if (st !== undefined) setW("steps", st);
    node.setDirtyCanvas?.(true, true);
    try { app.extensionManager?.toast?.add({ severity: "success", summary: "Recommended settings applied", detail: `${rec.label}: ${rec.sampler}/${rec.scheduler}, cfg≈${c}, steps≈${st}`, life: 4000 }); } catch (_) {}
}

app.registerExtension({
    name: "C2C.SettingsRecommender",
    async setup() {
        try {
            app.extensionManager.registerSidebarTab({
                id: "c2c-recommended",
                title: "Recommended",
                icon: "pi pi-sliders-h",
                type: "custom",
                render: (el) => renderPanel(el),
            });
        } catch (_) { /* older frontend without sidebar API */ }
    },
    // Deep breakdown in the node's right-click menu (the in-node inspector).
    getNodeMenuItems(node) {
        if (!isSamplerNode(node)) return [];
        const rec = recFor(modelForSampler(node, detectedModels()));
        const dd = deepFor(rec.label);
        const items = [{
            content: `⚙ Recommended for ${rec.label}`,
            disabled: true,
        }, {
            content: `   sampler ${rec.sampler} · scheduler ${rec.scheduler} · cfg ${rec.cfg} · steps ${rec.steps}`,
            disabled: true,
        }];
        if (dd) {
            if (dd.res && dd.res !== "—") items.push({ content: `   resolution: ${dd.res}`, disabled: true });
            if (dd.vae && dd.vae !== "—") items.push({ content: `   VAE: ${dd.vae}`, disabled: true });
            if (dd.clip && dd.clip !== "—") items.push({ content: `   CLIP/encoder: ${dd.clip}`, disabled: true });
            if (dd.neg && dd.neg !== "—") items.push({ content: `   negative: ${dd.neg}`, disabled: true });
            if (dd.alts && dd.alts !== "—") items.push({ content: `   alts: ${dd.alts}`, disabled: true });
            if (dd.pitfalls && dd.pitfalls !== "—") items.push({ content: `   ⚠ ${dd.pitfalls}`, disabled: true });
        }
        items.push({ content: `   ↳ Apply ${rec.sampler}/${rec.scheduler}, cfg/steps`, callback: () => applyTo(node, rec) });
        return items;
    },
});
