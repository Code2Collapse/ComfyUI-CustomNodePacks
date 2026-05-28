/**
 * c2c_prompt_wizard.js — v2.0 P2: AI-assisted prompt builder.
 *
 * Sidebar tab "C2C Prompt" with a structured form:
 *   - Subject, Setting, Style, Mood, Camera/Lens, Lighting, Negative seeds
 *   - Model family selector (sdxl / sd1.5 / flux / any) — changes phrasing rules
 *   - Optional reference style preset (from /c2c/styles)
 *
 * Pressing "Generate" streams a structured response from /c2c/ai/stream
 * (feature=prompt_wizard) and splits it into positive / negative buckets.
 * "Apply to selected CLIPTextEncode" pushes the result into the currently
 * selected positive/negative pair (best-effort, see findTargetEncoder()).
 *
 * No stub behaviour — if no backend is configured the panel says so and
 * links to the AI Settings tab.
 */
import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { forAllNodes } from "./_subgraph_walk.js";
import { streamAI } from "./_c2c_ai_client.js";

const TAB_ID = "c2c.prompt";

// Theme colors come from `_c2c_theme.js` CSS variables (`--c2c-*`). They flip
// live across mocha/latte/oled when `setVariant()` rewrites :root vars, so the
// sidebar repaints without rebuild. No literal hex/rgb in this file.

// --------------------------------------------------------- prompt synthesis
const SYS_BY_MODEL = {
    sdxl:
        "You are a prompt engineer for SDXL. Output ONLY two blocks separated by an explicit marker line.\n" +
        "Use comma-separated tag style with quality boosters first, weighted parens for emphasis (e.g. (subject:1.2)). " +
        "Negative should be comma tokens. Keep each block under 600 tokens.\n\n" +
        "FORMAT:\n--- POSITIVE ---\n<tags>\n--- NEGATIVE ---\n<tags>\n",
    "sd1.5":
        "You are a prompt engineer for Stable Diffusion 1.5. Output ONLY two blocks separated by an explicit marker line.\n" +
        "Use weighted token style: (masterpiece:1.2), (best quality:1.2), ... Negative uses (lowres:1.4), (worst quality:1.4) " +
        "style. Keep each block compact.\n\n" +
        "FORMAT:\n--- POSITIVE ---\n<tokens>\n--- NEGATIVE ---\n<tokens>\n",
    flux:
        "You are a prompt engineer for FLUX. Output ONLY two blocks separated by an explicit marker line.\n" +
        "FLUX uses natural-language descriptions, full sentences, cinematographer/lens language. " +
        "FLUX largely ignores negative prompts; leave the NEGATIVE block empty or with one sentence describing what to avoid.\n\n" +
        "FORMAT:\n--- POSITIVE ---\n<paragraph>\n--- NEGATIVE ---\n<sentence or empty>\n",
    any:
        "You are a prompt engineer for diffusion models. Output ONLY two blocks separated by an explicit marker line.\n" +
        "Use a natural-language paragraph for the positive prompt and a comma-separated list of negatives.\n\n" +
        "FORMAT:\n--- POSITIVE ---\n<text>\n--- NEGATIVE ---\n<text>\n",
};

function buildUserPrompt(form, presetSummary) {
    const parts = [];
    if (form.subject)   parts.push(`Subject: ${form.subject}`);
    if (form.setting)   parts.push(`Setting: ${form.setting}`);
    if (form.style)     parts.push(`Style: ${form.style}`);
    if (form.mood)      parts.push(`Mood: ${form.mood}`);
    if (form.camera)    parts.push(`Camera/lens: ${form.camera}`);
    if (form.lighting)  parts.push(`Lighting: ${form.lighting}`);
    if (form.aspect)    parts.push(`Aspect: ${form.aspect}`);
    if (form.avoid)     parts.push(`Things to avoid: ${form.avoid}`);
    if (form.notes)     parts.push(`Other notes: ${form.notes}`);
    if (presetSummary)  parts.push(`Inspired by preset: ${presetSummary}`);
    return "Build positive/negative prompts from this brief:\n\n" + parts.join("\n");
}

function splitPositiveNegative(text) {
    // Tolerate variations: "--- POSITIVE ---", "POSITIVE:", etc.
    const norm = text.replace(/\r/g, "");
    const re = /(?:^|\n)\s*(?:[-—=]{1,5}\s*)?(POSITIVE|NEGATIVE)\s*(?:[-—=:]{1,5})?\s*\n/gi;
    const indices = [];
    let m;
    while ((m = re.exec(norm)) !== null) indices.push({ kind: m[1].toUpperCase(), at: m.index, end: re.lastIndex });
    if (indices.length === 0) return { positive: norm.trim(), negative: "" };
    const out = { positive: "", negative: "" };
    for (let i = 0; i < indices.length; i++) {
        const cur = indices[i];
        const next = indices[i + 1];
        const body = norm.slice(cur.end, next ? next.at : norm.length).trim();
        if (cur.kind === "POSITIVE" && !out.positive) out.positive = body;
        else if (cur.kind === "NEGATIVE" && !out.negative) out.negative = body;
    }
    return out;
}

// --------------------------------------------------------- selection helpers
function findTargetEncoders() {
    // Returns {positive: nodeOrNull, negative: nodeOrNull} based on current selection.
    // Heuristic: if exactly two CLIPTextEncode nodes are selected, the first by id
    // wins "positive" unless one is wired to KSampler.negative.
    const sel = Object.values(app.canvas?.selected_nodes || {});
    const encs = sel.filter(n => /CLIPTextEncode/.test(n.type || ""));
    if (encs.length === 0) {
        // No selection — scan root + every subgraph for plausible encoders.
        const all = [];
        forAllNodes((n) => { if (/CLIPTextEncode/.test(n.type || "")) all.push(n); });
        if (all.length === 0) return { positive: null, negative: null };
        if (all.length === 1) return { positive: all[0], negative: null };
        return _classifyByKSamplerWiring(all);
    }
    if (encs.length === 1) return { positive: encs[0], negative: null };
    return _classifyByKSamplerWiring(encs);
}

function _classifyByKSamplerWiring(encs) {
    // Find KSamplers anywhere (root + subgraphs) and see which encoder feeds .negative.
    // Track each sampler's owning graph so we resolve its links/nodes locally.
    const samplers = [];
    forAllNodes((n, g) => { if (/KSampler/.test(n.type || "")) samplers.push({ node: n, graph: g }); });
    for (const { node: s, graph: sg } of samplers) {
        const neg = (s.inputs || []).find(i => i.name === "negative");
        const pos = (s.inputs || []).find(i => i.name === "positive");
        if (neg?.link != null) {
            const ln = sg?.links?.[neg.link];
            const src = ln && sg?.getNodeById?.(ln.origin_id);
            if (encs.includes(src)) {
                const other = encs.find(n => n !== src) || null;
                return { positive: other, negative: src };
            }
        }
        if (pos?.link != null) {
            const ln = sg?.links?.[pos.link];
            const src = ln && sg?.getNodeById?.(ln.origin_id);
            if (encs.includes(src)) {
                const other = encs.find(n => n !== src) || null;
                return { positive: src, negative: other };
            }
        }
    }
    // Fallback: smallest id = positive (very common convention)
    const sorted = [...encs].sort((a, b) => a.id - b.id);
    return { positive: sorted[0], negative: sorted[1] || null };
}

function setEncoderText(node, text) {
    if (!node) return false;
    const widget = (node.widgets || []).find(w => w.name === "text");
    if (!widget) return false;
    widget.value = text;
    node.setDirtyCanvas?.(true, true);
    return true;
}

// --------------------------------------------------------- view
async function loadPresets() {
    try {
        const r = await fetch("/c2c/styles");
        const j = await r.json();
        return j.success ? j.data : [];
    } catch (_) { return []; }
}

async function loadAIStatus() {
    try {
        const r = await fetch("/c2c/ai/status");
        return await r.json();
    } catch (_) { return { backends: [] }; }
}

function buildView(root) {
    root.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.style.cssText = `padding:10px;background:var(--c2c-bg);color:var(--c2c-fg);
        font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;
        height:100%;overflow:auto;`;
    wrap.innerHTML =
        `<h3 style="margin:0 0 10px;color:var(--c2c-mauve);font-size:13px;letter-spacing:0.5px;text-transform:uppercase">Prompt Wizard</h3>
         <div id="pw-status" style="font-size:10px;color:var(--c2c-sub);margin-bottom:8px"></div>

         <label style="display:block;font-size:10px;color:var(--c2c-sub);margin-top:6px">Model family</label>
         <select id="pw-model" style="${selStyle()}">
           <option value="sdxl">SDXL (tag style)</option>
           <option value="sd1.5">SD 1.5 (weighted tokens)</option>
           <option value="flux">FLUX (natural language)</option>
           <option value="any">Any / generic</option>
         </select>

         <label style="display:block;font-size:10px;color:var(--c2c-sub);margin-top:6px">Style preset (optional)</label>
         <select id="pw-preset" style="${selStyle()}"><option value="">— none —</option></select>

         ${field("subject",  "Subject",          "e.g. a young woman with red hair, holding a lantern")}
         ${field("setting",  "Setting",          "e.g. moss-covered ruins at dusk")}
         ${field("style",    "Style",            "e.g. studio Ghibli, Moebius, oil painting")}
         ${field("mood",     "Mood",             "e.g. melancholic, hopeful, ominous")}
         ${field("camera",   "Camera / lens",    "e.g. 35mm anamorphic, shallow depth of field")}
         ${field("lighting", "Lighting",         "e.g. golden hour rim light, volumetric fog")}
         ${field("aspect",   "Aspect / framing", "e.g. wide cinematic, portrait, close-up")}
         ${field("avoid",    "Avoid",            "e.g. text, watermark, deformed hands")}
         ${field("notes",    "Other notes",      "")}

         <div style="display:flex;gap:6px;margin-top:10px">
           <button id="pw-go" style="${btnPrimary()}">✨ Generate</button>
           <button id="pw-clear" style="${btnGhost()}">Clear</button>
         </div>

         <h4 style="margin:14px 0 6px;color:var(--c2c-green);font-size:11px;letter-spacing:0.5px;text-transform:uppercase">Positive</h4>
         <textarea id="pw-pos" rows="5" style="${areaStyle()}"></textarea>

         <h4 style="margin:10px 0 6px;color:var(--c2c-red);font-size:11px;letter-spacing:0.5px;text-transform:uppercase">Negative</h4>
         <textarea id="pw-neg" rows="4" style="${areaStyle()}"></textarea>

         <div style="display:flex;gap:6px;margin-top:8px">
           <button id="pw-apply" style="${btnPrimary()}">Apply to selected encoders</button>
           <button id="pw-copy"  style="${btnGhost()}">Copy both</button>
         </div>
         <div id="pw-msg" style="margin-top:8px;font-size:10px;color:var(--c2c-sub)"></div>
         <div id="pw-stream" style="margin-top:8px;font-size:10px;color:var(--c2c-sub);max-height:80px;overflow:auto;white-space:pre-wrap;border:1px solid var(--c2c-border);border-radius:4px;padding:4px;display:none"></div>`;
    root.appendChild(wrap);

    // Populate presets + status
    loadPresets().then(list => {
        const sel = wrap.querySelector("#pw-preset");
        list.forEach(p => {
            const o = document.createElement("option");
            o.value = p.id; o.textContent = `${p.name} (${p.model_hint})`;
            sel.appendChild(o);
        });
    });
    loadAIStatus().then(s => {
        const ok = (s.backends || []).some(b => b.health?.ok);
        const el = wrap.querySelector("#pw-status");
        if (ok) {
            el.textContent = `${(s.backends || []).filter(b => b.health?.ok).length} backend(s) online · cost today $${(s.cost_today_usd ?? 0).toFixed(3)}`;
        } else {
            el.innerHTML = `<span style="color:var(--c2c-red)">No AI backend online.</span> Open the C2C AI sidebar tab to configure.`;
        }
    });

    // Wire actions
    wrap.querySelector("#pw-go").onclick = () => runGeneration(wrap);
    wrap.querySelector("#pw-clear").onclick = () => {
        wrap.querySelectorAll("input[data-field]").forEach(i => i.value = "");
        wrap.querySelector("#pw-pos").value = "";
        wrap.querySelector("#pw-neg").value = "";
        wrap.querySelector("#pw-msg").textContent = "";
    };
    wrap.querySelector("#pw-apply").onclick = () => applyToSelection(wrap);
    wrap.querySelector("#pw-copy").onclick = () => {
        const text = "POSITIVE:\n" + wrap.querySelector("#pw-pos").value +
                     "\n\nNEGATIVE:\n" + wrap.querySelector("#pw-neg").value;
        navigator.clipboard?.writeText(text);
        wrap.querySelector("#pw-msg").textContent = "Copied to clipboard.";
    };
}

function field(name, label, ph) {
    return `<label style="display:block;font-size:10px;color:var(--c2c-sub);margin-top:6px">${label}</label>
            <input type="text" data-field="${name}" placeholder="${ph}" style="${inputStyle()}"/>`;
}
function selStyle()   { return `width:100%;background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:4px;padding:4px;font-size:11px`; }
function inputStyle() { return selStyle(); }
function areaStyle()  { return `width:100%;background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:4px;padding:6px;font-size:11px;font-family:ui-monospace,Menlo,monospace`; }
// Primary button uses mauve bg + var(--c2c-bg) as text. Both flip across
// variants in opposite directions (mauve gets darker in latte, bg gets
// lighter), so contrast is preserved on every theme.
function btnPrimary() { return `background:var(--c2c-mauve);color:var(--c2c-bg);border:none;border-radius:4px;padding:5px 12px;cursor:pointer;font-size:11px`; }
function btnGhost()   { return `background:var(--c2c-bg2);color:var(--c2c-fg);border:1px solid var(--c2c-border);border-radius:4px;padding:5px 12px;cursor:pointer;font-size:11px`; }

async function runGeneration(wrap) {
    const form = {};
    wrap.querySelectorAll("input[data-field]").forEach(i => form[i.dataset.field] = i.value.trim());
    const model = wrap.querySelector("#pw-model").value;
    const presetId = wrap.querySelector("#pw-preset").value;
    let presetSummary = "";
    if (presetId) {
        try {
            const r = await fetch("/c2c/styles/" + encodeURIComponent(presetId));
            const j = await r.json();
            if (j.success) {
                presetSummary = j.data.name + " — " + (j.data.positive || "").slice(0, 200);
            }
        } catch (__c2cErr) { __c2cReport("c2c_prompt_wizard", __c2cErr); }
    }

    const sys = SYS_BY_MODEL[model] || SYS_BY_MODEL.any;
    const user = buildUserPrompt(form, presetSummary);

    const msg = wrap.querySelector("#pw-msg");
    const stream = wrap.querySelector("#pw-stream");
    msg.textContent = "Asking AI…"; stream.style.display = "block"; stream.textContent = "";
    wrap.querySelector("#pw-pos").value = "";
    wrap.querySelector("#pw-neg").value = "";

    let buffer = "";
    const result = await streamAI({
        feature: "prompt_wizard",
        sensitivity: "public",
        max_tokens: 900,
        temperature: 0.7,
        messages: [
            { role: "system", content: sys },
            { role: "user", content: user },
        ],
        onChunk: (_chunk, total) => {
            buffer = total;
            stream.textContent = buffer;
            stream.scrollTop = stream.scrollHeight;
        },
        onError: (err) => {
            msg.textContent = "AI " + (err.kind || "error") + ": " + (err.message || "");
        },
    });
    if (!result.ok) return;
    const split = splitPositiveNegative(buffer);
    wrap.querySelector("#pw-pos").value = split.positive;
    wrap.querySelector("#pw-neg").value = split.negative;
    msg.textContent = "Done. Review and apply.";
}

function applyToSelection(wrap) {
    const { positive, negative } = findTargetEncoders();
    const pos = wrap.querySelector("#pw-pos").value;
    const neg = wrap.querySelector("#pw-neg").value;
    let applied = [];
    if (positive && setEncoderText(positive, pos)) applied.push("positive#" + positive.id);
    if (negative && setEncoderText(negative, neg)) applied.push("negative#" + negative.id);
    const msg = wrap.querySelector("#pw-msg");
    if (applied.length === 0) {
        msg.textContent = "No CLIPTextEncode in selection or graph; select one positive + one negative encoder.";
    } else {
        msg.textContent = "Applied to " + applied.join(", ");
    }
}

// --------------------------------------------------------- register
app.registerExtension({
    name: "c2c.prompt.wizard",
    async setup() {
        try {
            app.extensionManager?.registerSidebarTab?.({
                id: TAB_ID,
                icon: "pi pi-pencil",
                title: "C2C Prompt",
                tooltip: "C2C Prompt Wizard — AI-assisted prompt builder",
                type: "custom",
                render: (el) => buildView(el),
            });
        } catch (exc) {
            console.warn("[c2c.prompt.wizard] sidebar tab registration failed:", exc);
        }
    },
});

window.__C2C_PROMPT_WIZARD__ = { buildView, splitPositiveNegative };
