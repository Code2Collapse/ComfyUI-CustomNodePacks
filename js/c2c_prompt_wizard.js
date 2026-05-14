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

const TAB_ID = "c2c.prompt";

const C = { bg:"#1e1e2e", bg2:"#181825", fg:"#cdd6f4", sub:"#a6adc8",
            border:"#313244", mauve:"#cba6f7", blue:"#89b4fa", green:"#a6e3a1",
            red:"#f38ba8", yellow:"#f9e2af", teal:"#94e2d5" };

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
        // No selection — scan whole graph for the two most plausible.
        const all = (app.graph._nodes || []).filter(n => /CLIPTextEncode/.test(n.type || ""));
        if (all.length === 0) return { positive: null, negative: null };
        if (all.length === 1) return { positive: all[0], negative: null };
        return _classifyByKSamplerWiring(all);
    }
    if (encs.length === 1) return { positive: encs[0], negative: null };
    return _classifyByKSamplerWiring(encs);
}

function _classifyByKSamplerWiring(encs) {
    // Find KSampler and see which encoder feeds .negative input
    const samplers = (app.graph._nodes || []).filter(n => /KSampler/.test(n.type || ""));
    for (const s of samplers) {
        const neg = (s.inputs || []).find(i => i.name === "negative");
        const pos = (s.inputs || []).find(i => i.name === "positive");
        if (neg?.link != null) {
            const ln = app.graph.links[neg.link];
            const src = ln && app.graph.getNodeById(ln.origin_id);
            if (encs.includes(src)) {
                const other = encs.find(n => n !== src) || null;
                return { positive: other, negative: src };
            }
        }
        if (pos?.link != null) {
            const ln = app.graph.links[pos.link];
            const src = ln && app.graph.getNodeById(ln.origin_id);
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
    wrap.style.cssText = `padding:10px;background:${C.bg};color:${C.fg};
        font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;
        height:100%;overflow:auto;`;
    wrap.innerHTML =
        `<h3 style="margin:0 0 10px;color:${C.mauve};font-size:13px;letter-spacing:0.5px;text-transform:uppercase">Prompt Wizard</h3>
         <div id="pw-status" style="font-size:10px;color:${C.sub};margin-bottom:8px"></div>

         <label style="display:block;font-size:10px;color:${C.sub};margin-top:6px">Model family</label>
         <select id="pw-model" style="${selStyle()}">
           <option value="sdxl">SDXL (tag style)</option>
           <option value="sd1.5">SD 1.5 (weighted tokens)</option>
           <option value="flux">FLUX (natural language)</option>
           <option value="any">Any / generic</option>
         </select>

         <label style="display:block;font-size:10px;color:${C.sub};margin-top:6px">Style preset (optional)</label>
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

         <h4 style="margin:14px 0 6px;color:${C.green};font-size:11px;letter-spacing:0.5px;text-transform:uppercase">Positive</h4>
         <textarea id="pw-pos" rows="5" style="${areaStyle()}"></textarea>

         <h4 style="margin:10px 0 6px;color:${C.red};font-size:11px;letter-spacing:0.5px;text-transform:uppercase">Negative</h4>
         <textarea id="pw-neg" rows="4" style="${areaStyle()}"></textarea>

         <div style="display:flex;gap:6px;margin-top:8px">
           <button id="pw-apply" style="${btnPrimary()}">Apply to selected encoders</button>
           <button id="pw-copy"  style="${btnGhost()}">Copy both</button>
         </div>
         <div id="pw-msg" style="margin-top:8px;font-size:10px;color:${C.sub}"></div>
         <div id="pw-stream" style="margin-top:8px;font-size:10px;color:${C.sub};max-height:80px;overflow:auto;white-space:pre-wrap;border:1px solid ${C.border};border-radius:4px;padding:4px;display:none"></div>`;
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
            el.innerHTML = `<span style="color:${C.red}">No AI backend online.</span> Open the C2C AI sidebar tab to configure.`;
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
    return `<label style="display:block;font-size:10px;color:${C.sub};margin-top:6px">${label}</label>
            <input type="text" data-field="${name}" placeholder="${ph}" style="${inputStyle()}"/>`;
}
function selStyle()   { return `width:100%;background:${C.bg2};color:${C.fg};border:1px solid ${C.border};border-radius:4px;padding:4px;font-size:11px`; }
function inputStyle() { return selStyle(); }
function areaStyle()  { return `width:100%;background:${C.bg2};color:${C.fg};border:1px solid ${C.border};border-radius:4px;padding:6px;font-size:11px;font-family:ui-monospace,Menlo,monospace`; }
function btnPrimary() { return `background:${C.mauve};color:${C.bg};border:none;border-radius:4px;padding:5px 12px;cursor:pointer;font-size:11px`; }
function btnGhost()   { return `background:${C.bg2};color:${C.fg};border:1px solid ${C.border};border-radius:4px;padding:5px 12px;cursor:pointer;font-size:11px`; }

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
        } catch (_) {}
    }

    const sys = SYS_BY_MODEL[model] || SYS_BY_MODEL.any;
    const user = buildUserPrompt(form, presetSummary);

    const msg = wrap.querySelector("#pw-msg");
    const stream = wrap.querySelector("#pw-stream");
    msg.textContent = "Asking AI…"; stream.style.display = "block"; stream.textContent = "";
    wrap.querySelector("#pw-pos").value = "";
    wrap.querySelector("#pw-neg").value = "";

    let buffer = "";
    try {
        const r = await fetch("/c2c/ai/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
            body: JSON.stringify({
                feature: "prompt_wizard",
                sensitivity: "public",
                max_tokens: 900,
                temperature: 0.7,
                messages: [
                    { role: "system", content: sys },
                    { role: "user", content: user },
                ],
            }),
        });
        if (!r.ok || !r.body) {
            const j = await r.json().catch(() => ({}));
            msg.textContent = "AI request failed: " + (j.message || ("HTTP " + r.status));
            return;
        }
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            let idx;
            while ((idx = buf.indexOf("\n\n")) !== -1) {
                const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
                const lines = frame.split("\n");
                let event = "message", data = "";
                lines.forEach(l => {
                    if (l.startsWith("event: ")) event = l.slice(7).trim();
                    else if (l.startsWith("data: ")) data += l.slice(6);
                });
                if (!data) continue;
                if (event === "error") { msg.textContent = "AI error: " + data; return; }
                if (event === "done")  { continue; }
                try {
                    const obj = JSON.parse(data);
                    if (obj.chunk) {
                        buffer += obj.chunk;
                        stream.textContent = buffer;
                        stream.scrollTop = stream.scrollHeight;
                    }
                } catch (_) {}
            }
        }
        // Split when stream is complete
        const split = splitPositiveNegative(buffer);
        wrap.querySelector("#pw-pos").value = split.positive;
        wrap.querySelector("#pw-neg").value = split.negative;
        msg.textContent = "Done. Review and apply.";
        window.__C2C_AI_HUD__?.refresh?.();
    } catch (exc) {
        msg.textContent = "Error: " + exc.message;
    }
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
