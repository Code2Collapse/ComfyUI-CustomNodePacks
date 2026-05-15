/**
 * c2c_ai_explainer.js — "Explain this node with AI" context menu action.
 *
 * Adds a right-click menu item on every node. When clicked:
 *   - Opens a docked panel (right side, resizable).
 *   - Streams an explanation from /c2c/ai/stream (feature=node_explainer).
 *   - Auto-redacts widget values via the cloud redactor on the backend.
 *
 * Coexists with the existing mec_node_explain Tier-1 hover tooltip:
 *   - Tier-1 is fast, offline, exact for known nodes.
 *   - This Tier-2 path calls AI for unknown / custom nodes or for deeper "why
 *     would I use this?" questions.
 */
import { app } from "../../scripts/app.js";

const PANEL_ID = "c2c-ai-explainer-panel";
const SETTING_ENABLE = "c2c.ai.explainer.enabled";

const C = { bg:"#1e1e2e", bg2:"#181825", fg:"#cdd6f4", sub:"#a6adc8",
            border:"#313244", mauve:"#cba6f7", blue:"#89b4fa", green:"#a6e3a1",
            red:"#f38ba8" };

function nodeSummary(node) {
    const lines = [];
    lines.push(`class_type: ${node.comfyClass || node.type}`);
    lines.push(`title: ${node.title || node.type}`);
    if (Array.isArray(node.inputs) && node.inputs.length) {
        lines.push("inputs:");
        node.inputs.forEach(i => lines.push(`  - ${i.name} (${i.type})${i.link ? " [linked]" : ""}`));
    }
    if (Array.isArray(node.outputs) && node.outputs.length) {
        lines.push("outputs:");
        node.outputs.forEach(o => lines.push(`  - ${o.name} (${o.type})`));
    }
    if (Array.isArray(node.widgets) && node.widgets.length) {
        lines.push("widgets:");
        node.widgets.forEach(w => {
            let v = w.value;
            if (typeof v === "string" && v.length > 80) v = v.slice(0, 80) + "…";
            lines.push(`  - ${w.name} = ${JSON.stringify(v)}`);
        });
    }
    return lines.join("\n");
}

function ensurePanel() {
    let p = document.getElementById(PANEL_ID);
    if (p) return p;
    p = document.createElement("div");
    p.id = PANEL_ID;
    p.style.cssText =
        `position:fixed;top:60px;right:12px;width:420px;max-height:75vh;
         z-index:9990;background:${C.bg};color:${C.fg};
         border:1px solid ${C.border};border-radius:8px;
         box-shadow:0 12px 40px rgba(0,0,0,0.55);overflow:hidden;
         display:flex;flex-direction:column;
         font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;`;
    p.innerHTML =
        `<div style="padding:8px 12px;background:${C.bg2};display:flex;align-items:center;gap:8px;border-bottom:1px solid ${C.border}">
           <span style="color:${C.mauve};font-weight:600;letter-spacing:0.5px;text-transform:uppercase;font-size:11px">AI Explainer</span>
           <span class="title" style="flex:1;color:${C.sub};font-size:11px"></span>
           <span class="meta" style="color:${C.sub};font-size:10px"></span>
           <button class="close" style="background:transparent;color:${C.sub};border:none;font-size:18px;cursor:pointer;line-height:1">×</button>
         </div>
         <div class="body" style="padding:12px;overflow:auto;white-space:pre-wrap;flex:1;line-height:1.5"></div>
         <div class="footer" style="padding:6px 12px;background:${C.bg2};border-top:1px solid ${C.border};color:${C.sub};font-size:10px;display:flex;justify-content:space-between"><span class="status"></span><span class="backend"></span></div>`;
    p.querySelector(".close").onclick = () => p.remove();
    document.body.appendChild(p);
    return p;
}

async function explain(node) {
    const panel = ensurePanel();
    panel.querySelector(".title").textContent = node.title || node.type;
    const body = panel.querySelector(".body");
    body.textContent = "";
    panel.querySelector(".status").textContent = "asking AI…";
    panel.querySelector(".backend").textContent = "";
    panel.querySelector(".meta").textContent = "";

    const sys = "You are a ComfyUI assistant. The user has a custom node and wants a concise explanation of what it does, when to use it, and the meaning of each widget. Be direct, no fluff. Use short paragraphs.";
    const user = "Explain this node:\n\n" + nodeSummary(node);

    try {
        const r = await fetch("/c2c/ai/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
            body: JSON.stringify({
                feature: "node_explainer",
                sensitivity: "normal",
                max_tokens: 600,
                temperature: 0.3,
                messages: [
                    { role: "system", content: sys },
                    { role: "user", content: user },
                ],
            }),
        });
        if (!r.ok || !r.body) {
            body.textContent = "AI request failed: HTTP " + r.status;
            panel.querySelector(".status").textContent = "error";
            return;
        }
        panel.querySelector(".status").textContent = "streaming…";
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
                if (event === "error") {
                    body.textContent += "\n[error] " + data;
                    panel.querySelector(".status").textContent = "error";
                } else if (event === "done") {
                    panel.querySelector(".status").textContent = "done";
                } else {
                    try {
                        const obj = JSON.parse(data);
                        if (obj.chunk) {
                            body.textContent += obj.chunk;
                            body.scrollTop = body.scrollHeight;
                        }
                    } catch (_) {}
                }
            }
        }
        // Refresh HUD after request so cost pill updates
        window.__C2C_AI_HUD__?.refresh?.();
    } catch (exc) {
        body.textContent = "Error: " + exc.message;
        panel.querySelector(".status").textContent = "error";
    }
}

app.registerExtension({
    name: "c2c.ai.explainer",
    settings: [
        { id: SETTING_ENABLE, name: "C2C ▸ AI ▸ Show 'Explain with AI' in node menu",
          type: "boolean", default: true },
    ],
    async beforeRegisterNodeDef(nodeType) {
        const orig = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            if (orig) orig.call(this, canvas, options);
            const enabled = app.ui?.settings?.getSettingValue(SETTING_ENABLE, true);
            if (!enabled) return;
            options.push(null);
            options.push({
                content: "🧠 Explain with AI",
                callback: () => explain(this),
            });
        };
    },
});

window.__C2C_AI_EXPLAINER__ = { explain };
