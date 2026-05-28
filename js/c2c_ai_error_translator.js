/**
 * c2c_ai_error_translator.js — AI fallback for the existing Error Translator.
 *
 * The Tier-1 pattern matcher (mec/error_translator backend) handles known
 * recurring errors. When it returns no match, OR when the user clicks
 * "Ask AI" inside an error toast, this module forwards the redacted
 * traceback to /c2c/ai/stream (feature=error_translator, sensitivity=sensitive).
 *
 * Sensitivity is intentionally "sensitive" — tracebacks routinely contain
 * absolute paths. The router's policy default for this feature is
 * PREFER_LOCAL; if no local backend is configured, the user gets a clear
 * "need a local AI to translate this safely" message instead of leaking
 * paths to the cloud.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const PANEL_ID = "c2c-ai-errtrans-panel";
const SETTING_AUTO = "c2c.ai.errorTranslator.auto";

// All colors source from CSS custom properties emitted by _c2c_theme.js
// (--c2c-bg, --c2c-fg, --c2c-red, etc.) so panel + toast repaint instantly
// when setVariant() flips mocha/latte/oled. No hardcoded palette.

function ensurePanel() {
    let p = document.getElementById(PANEL_ID);
    if (p) return p;
    p = document.createElement("div");
    p.id = PANEL_ID;
    p.style.cssText =
        `position:fixed;bottom:52px;right:18px;width:460px;max-height:60vh;
         z-index:var(--c2c-z-popover);background:var(--c2c-bg);color:var(--c2c-fg);
         border:1px solid var(--c2c-red);border-radius:8px;
         box-shadow:0 8px 32px color-mix(in srgb, var(--c2c-shadowBase) 60%, transparent);overflow:hidden;
         display:flex;flex-direction:column;
         font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;`;
    p.innerHTML =
        `<div style="padding:8px 12px;background:var(--c2c-bg2);display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--c2c-border)">
           <span style="color:var(--c2c-red);font-weight:600;letter-spacing:0.5px;text-transform:uppercase;font-size:11px">AI Error Helper</span>
           <span class="ctx" style="flex:1;color:var(--c2c-sub);font-size:11px"></span>
           <button class="close" style="background:transparent;color:var(--c2c-sub);border:none;font-size:18px;cursor:pointer;line-height:1">×</button>
         </div>
         <div class="body" style="padding:10px;overflow:auto;white-space:pre-wrap;flex:1;line-height:1.5"></div>
         <div class="footer" style="padding:6px 12px;background:var(--c2c-bg2);border-top:1px solid var(--c2c-border);color:var(--c2c-sub);font-size:10px;display:flex;justify-content:space-between"><span class="status"></span><span class="backend"></span></div>`;
    p.querySelector(".close").onclick = () => p.remove();
    document.body.appendChild(p);
    return p;
}

async function translate(errorText, context = "") {
    const panel = ensurePanel();
    panel.querySelector(".ctx").textContent = context || "execution error";
    const body = panel.querySelector(".body");
    body.textContent = "";
    panel.querySelector(".status").textContent = "asking local AI…";

    const sys = "You are a senior ComfyUI engineer. The user just hit a Python traceback. Translate it into a plain-English explanation that a non-developer can follow. Suggest concrete next steps. Be terse. Use bullet points. End with one sentence on the most likely fix.";
    const user = "Context: " + (context || "(none)") + "\n\nError:\n" + errorText;

    try {
        const r = await fetch("/c2c/ai/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
            body: JSON.stringify({
                feature: "error_translator",
                sensitivity: "sensitive",  // routes to local by default; redacted before any cloud send
                max_tokens: 500,
                temperature: 0.2,
                messages: [
                    { role: "system", content: sys },
                    { role: "user", content: user },
                ],
            }),
        });
        if (!r.ok || !r.body) {
            const j = await r.json().catch(() => ({}));
            body.textContent =
                "AI helper unavailable: " + (j.message || ("HTTP " + r.status)) +
                "\n\nTip: install Ollama and pull a Qwen3 GGUF, or set the policy for 'error_translator' to allow cloud in Settings.";
            panel.querySelector(".status").textContent = "no backend";
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
                    } catch (__c2cErr) { __c2cReport("c2c_ai_error_translator", __c2cErr); }
                }
            }
        }
        window.__C2C_AI_HUD__?.refresh?.();
    } catch (exc) {
        body.textContent = "Error: " + exc.message;
        panel.querySelector(".status").textContent = "error";
    }
}

app.registerExtension({
    name: "c2c.ai.errorTranslator",
    settings: [
        { id: SETTING_AUTO, name: "C2C ▸ AI ▸ Auto-open error helper on execution error",
          type: "boolean", default: false },
    ],
    async setup() {
        // Hook ComfyUI's execution error event
        api.addEventListener("execution_error", (ev) => {
            const auto = app.ui?.settings?.getSettingValue(SETTING_AUTO, false);
            const detail = ev?.detail || {};
            const text = [
                detail.exception_message || "",
                Array.isArray(detail.traceback) ? detail.traceback.join("") : (detail.traceback || ""),
            ].filter(Boolean).join("\n\n");
            if (!text) return;
            const ctx = detail.node_type ? `node ${detail.node_id} (${detail.node_type})` : "execution error";
            if (auto) {
                translate(text, ctx);
            } else {
                // Add a small toast button instead of auto-opening
                _showInlinePrompt(text, ctx);
            }
        });
    },
});

function _showInlinePrompt(text, ctx) {
    const id = "c2c-ai-errtrans-prompt";
    document.getElementById(id)?.remove();
    const div = document.createElement("div");
    div.id = id;
    div.style.cssText =
        `position:fixed;bottom:52px;right:18px;z-index:var(--c2c-z-toast);
         background:var(--c2c-bg);color:var(--c2c-fg);
         border:1px solid var(--c2c-red);border-radius:8px;padding:10px 14px;
         box-shadow:0 8px 32px color-mix(in srgb, var(--c2c-shadowBase) 60%, transparent);
         font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;
         display:flex;align-items:center;gap:10px;max-width:380px;`;
    div.innerHTML =
        `<span style="color:var(--c2c-red);font-weight:600">⚠</span>
         <span style="flex:1;color:var(--c2c-sub);font-size:11px">${ctx}: error caught.</span>
         <button class="ai" style="background:var(--c2c-mauve);color:var(--c2c-bg);border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px">Ask AI</button>
         <button class="dismiss" style="background:transparent;color:var(--c2c-sub);border:none;cursor:pointer;font-size:14px">×</button>`;
    div.querySelector(".ai").onclick = () => { div.remove(); translate(text, ctx); };
    div.querySelector(".dismiss").onclick = () => div.remove();
    document.body.appendChild(div);
    setTimeout(() => div.remove(), 30_000);
}

window.__C2C_AI_ERRTRANS__ = { translate };
