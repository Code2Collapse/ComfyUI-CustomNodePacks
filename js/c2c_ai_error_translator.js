/**
 * c2c_ai_error_translator.js — BIG RED plain-English error panel.
 *
 * Goal: the user does NOT know technical Python/torch terminology. When a
 * workflow errors, ComfyUI shows its native red toast/dialog (technical).
 * Immediately AFTER that, this module pops a large, bold-RED, dismissable
 * panel that explains the error in simple English: a headline, the cause,
 * and concrete fix steps.
 *
 * Pipeline (always-on, zero configuration required):
 *   1. On the `execution_error` API event we instantly show the panel with a
 *      built-in JS deterministic humaniser so SOMETHING simple always shows
 *      even if the server route is down.
 *   2. We then POST the error to /mec/translate_error which runs the backend
 *      3-tier explainer (error_assistant.explain). The route honours the
 *      user's saved tier mode (auto / deterministic_only / local_only /
 *      cloud_only) — Tier-1 (offline, no setup) ALWAYS fires first, so the
 *      panel upgrades to the curated cause + fix steps with no keys needed.
 *   3. "Ask AI" escalates to the streaming LLM endpoint (Tier 2/3) when the
 *      user wants a deeper, model-written explanation.
 *
 * Colours are LITERAL hex (bold red) so the panel is unmissable regardless of
 * the active theme; theme CSS vars are used only as soft fallbacks.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const PANEL_ID = "c2c-ai-errtrans-panel";
const SETTING_ENABLED = "c2c.ai.errorTranslator.enabled";
const SETTING_AUTO_AI = "c2c.ai.errorTranslator.autoAskAI";

// Soft, translucent "frosted garnet" palette — a calm advisory card, not an
// alarm. backdrop-filter does the depth; a single desaturated-rose accent is
// used sparingly (left stripe, glyph, labels, fix edges). All literal values
// (never a theme var() — Canvas/CSS safe).
const ACCENT      = "#e8788a";                    // soft desaturated rose — the ONE accent
const ACCENT_SOFT = "rgba(232,120,138,0.13)";     // translucent rose fill (Ask-AI / glow)
const ACCENT_HOV  = "rgba(232,120,138,0.22)";     // accent hover
const GLASS_BG    = "rgba(24, 18, 22, 0.58)";     // frosted warm-charcoal base
const GLASS_WASH  = "rgba(255, 255, 255, 0.04)";  // faint header/footer wash
const ROW_BG      = "rgba(255, 255, 255, 0.045)"; // fix-row / detail fill
const HAIR        = "rgba(255, 255, 255, 0.09)";  // hairline borders/dividers
const TEXT        = "#f3edf0";                     // warm near-white body
const TEXT_DIM    = "#b3a8af";                     // muted secondary

function _c2cHumanise(raw) {
    if (!raw || typeof raw !== "string") return "Something went wrong while running this workflow.";
    const rules = [
        // ── Validation errors (fire BEFORE the run — the most common failures) ──
        [/required input.*missing|required_input_missing|missing.*required input/i, "A required input on this node isn't connected or filled in. Wire something into the empty (highlighted) socket."],
        [/value.*not in.*list|value_not_in_list|not in \[/i, "The option chosen on the node no longer exists — a model/file was renamed, moved, or isn't installed. Re-open the dropdown and pick a valid one."],
        [/value.*smaller than.*min|value_smaller_than_min/i, "A number on the node is below its allowed minimum. Raise it."],
        [/value.*(bigger|larger) than.*max|value_(bigger|larger)_than_max/i, "A number on the node is above its allowed maximum. Lower it."],
        [/bad_linked_input|return type.*mismatch|does not match.*input type|type mismatch/i, "Two connected nodes don't speak the same type — a wire links an output to an input that expects something else. Connect compatible sockets (matching colours)."],
        [/invalid prompt|prompt has no outputs|does not have outputs|cannot execute because/i, "The workflow can't run as wired — usually nothing is connected to a Save/Preview output, or a node upstream is missing an input."],
        [/custom validation failed|VALIDATE_INPUTS|validation failed/i, "The node rejected your settings during its own check. Re-read the node's tooltips and fix the flagged field."],
        // ── Runtime (traceback) errors ──
        [/AttributeError.*NoneType|NoneType.*attribute/i, "A required input is not connected. Check the node that failed — one of its input sockets is empty."],
        [/CUDA out of memory|out of memory/i, "Your GPU ran out of memory. Lower the resolution or batch size, or restart ComfyUI with --lowvram."],
        [/FileNotFoundError|No such file|cannot find the file/i, "A model or file is missing. Check that the file the node points to actually exists."],
        [/KeyError/i, "A setting or key the node expected is missing. Check the node's parameters and inputs."],
        [/mat1 and mat2 shapes cannot be multiplied|size mismatch|RuntimeError.*size|shape/i, "Two things don't fit together (sizes don't match). The connected models or images are from different families — check they belong together."],
        [/expected .* channels.*but got/i, "Wrong latent type. You connected a latent from a different model family (e.g. Flux into SD1.5). Use the matching checkpoint and VAE."],
        [/ConnectionError|HTTPError|Max retries|Connection refused/i, "Could not reach a server or service. Make sure the service (Ollama, API, etc.) is running."],
        [/expected .* but got|dtype|scalar type/i, "Wrong data type between two nodes. The output type doesn't match what the next node expects."],
        [/ModuleNotFoundError|ImportError|No module named/i, "A required Python package is missing. The node pack needs something installed — check its requirements.txt."],
        [/ValueError/i, "A value passed into the node is not allowed. Check the numbers/text you entered on the node."],
        [/IndexError|out of range/i, "The node tried to read past the end of a list — usually an empty batch or a missing item upstream."],
    ];
    for (const [re, msg] of rules) if (re.test(raw)) return msg;
    const line = raw.split("\n").reverse().find(l => l.trim() && !l.startsWith(" ") && !l.startsWith("\t")) || raw.slice(0, 140);
    return "Error: " + line.trim();
}

function _esc(s) {
    return String(s == null ? "" : s)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Flatten a ComfyUI prompt-validation rejection (400 from POST /prompt) into a
 *  readable string. The rejection is `{ response: { error, node_errors } }`. */
function _formatValidation(err) {
    if (!err) return "";
    const r = err.response || err;
    if (typeof r === "string") return r;
    const parts = [];
    if (r.error) parts.push(r.error.message || r.error.details || r.error.type || "");
    if (r.node_errors && typeof r.node_errors === "object") {
        for (const [nid, ne] of Object.entries(r.node_errors)) {
            const cls = ne.class_type || ne.type || "?";
            const errs = (ne.errors || [])
                .map((e2) => `${e2.message || e2.type}${e2.details ? " — " + e2.details : ""}`)
                .join("; ");
            parts.push(`Node ${nid} (${cls}): ${errs || "input error"}`);
        }
    }
    if (!parts.length && err.message) parts.push(err.message);
    return parts.filter(Boolean).join("\n");
}

function ensurePanel() {
    let p = document.getElementById(PANEL_ID);
    if (p) return p;
    p = document.createElement("div");
    p.id = PANEL_ID;
    p.style.cssText =
        `position:fixed;top:64px;right:18px;width:416px;max-width:calc(100vw - 36px);
         max-height:74vh;z-index:2147483600;
         background:${GLASS_BG};color:${TEXT};
         -webkit-backdrop-filter:blur(24px) saturate(135%);backdrop-filter:blur(24px) saturate(135%);
         border:1px solid ${HAIR};border-left:3px solid ${ACCENT};border-radius:16px;
         box-shadow:0 24px 70px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.05) inset, 0 0 44px ${ACCENT_SOFT};
         display:flex;flex-direction:column;overflow:hidden;
         font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;`;
    p.innerHTML =
        `<div class="hdr" style="padding:13px 15px;background:${GLASS_WASH};
              display:flex;align-items:center;gap:11px;cursor:move;user-select:none;border-bottom:1px solid ${HAIR}">
           <span style="font-size:17px;line-height:1;color:${ACCENT};filter:drop-shadow(0 0 7px ${ACCENT_SOFT})">&#9888;</span>
           <span style="flex:1;color:${TEXT};font-weight:650;font-size:15px;letter-spacing:0.2px">
             What went wrong
             <span class="ctx" style="display:block;font-weight:500;font-size:11px;color:${TEXT_DIM};letter-spacing:0.2px;margin-top:1px"></span>
           </span>
           <button class="close" title="Dismiss (Esc)"
             style="background:transparent;color:${TEXT_DIM};border:1px solid ${HAIR};
                    border-radius:8px;width:28px;height:28px;font-size:16px;cursor:pointer;line-height:1;
                    display:flex;align-items:center;justify-content:center;transition:background .15s,color .15s">&times;</button>
         </div>
         <div class="body" style="padding:15px 16px;overflow:auto;flex:1;line-height:1.62;font-size:15px">
           <div class="friendly" style="color:${TEXT};font-size:15px;font-weight:500;white-space:pre-wrap"></div>
           <div class="cause" style="margin-top:9px;color:${TEXT_DIM};font-size:14px;line-height:1.6;white-space:pre-wrap"></div>
           <div class="fixes" style="margin-top:13px"></div>
         </div>
         <div class="footer" style="padding:10px 14px;background:${GLASS_WASH};border-top:1px solid ${HAIR};
              display:flex;align-items:center;gap:8px;font-size:12px">
           <span class="status" style="flex:1;color:${TEXT_DIM}"></span>
           <button class="askai" title="Get a deeper AI explanation"
             style="background:${ACCENT_SOFT};color:${ACCENT};border:1px solid ${ACCENT_SOFT};border-radius:8px;
                    padding:6px 13px;cursor:pointer;font-size:12px;font-weight:600;transition:background .15s">Ask AI for more</button>
           <button class="dismiss"
             style="background:transparent;color:${TEXT_DIM};border:1px solid ${HAIR};border-radius:8px;
                    padding:6px 13px;cursor:pointer;font-size:12px;font-weight:500;transition:color .15s">Dismiss</button>
         </div>`;
    document.body.appendChild(p);

    const close = () => p.remove();
    p.querySelector(".close").onclick = close;
    p.querySelector(".dismiss").onclick = close;

    // Quiet micro-interactions (the only motion — restrained per design).
    const _ax = p.querySelector(".askai");
    if (_ax) { _ax.onmouseenter = () => { _ax.style.background = ACCENT_HOV; }; _ax.onmouseleave = () => { _ax.style.background = ACCENT_SOFT; }; }
    const _cl = p.querySelector(".close");
    if (_cl) { _cl.onmouseenter = () => { _cl.style.background = ACCENT_SOFT; _cl.style.color = ACCENT; }; _cl.onmouseleave = () => { _cl.style.background = "transparent"; _cl.style.color = TEXT_DIM; }; }

    // Esc to dismiss while panel is open.
    const onKey = (e) => { if (e.key === "Escape" && document.getElementById(PANEL_ID)) { close(); } };
    document.addEventListener("keydown", onKey);
    const obs = new MutationObserver(() => {
        if (!document.body.contains(p)) {
            document.removeEventListener("keydown", onKey);
            obs.disconnect();
        }
    });
    obs.observe(document.body, { childList: true });

    // Draggable by the header.
    const hdr = p.querySelector(".hdr");
    let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;
    hdr.addEventListener("mousedown", (e) => {
        if (e.target.closest("button")) return;
        dragging = true;
        const r = p.getBoundingClientRect();
        sx = e.clientX; sy = e.clientY; ox = r.left; oy = r.top;
        p.style.right = "auto"; p.style.left = ox + "px"; p.style.top = oy + "px";
        e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        p.style.left = Math.max(0, ox + e.clientX - sx) + "px";
        p.style.top = Math.max(0, oy + e.clientY - sy) + "px";
    });
    window.addEventListener("mouseup", () => { dragging = false; });

    return p;
}

function _setFriendly(panel, text) {
    const el = panel.querySelector(".friendly");
    if (el) el.textContent = text || "";
}
function _setCause(panel, text) {
    const el = panel.querySelector(".cause");
    if (el) el.textContent = text || "";
}
function _setFixes(panel, fixes) {
    const wrap = panel.querySelector(".fixes");
    if (!wrap) return;
    if (!Array.isArray(fixes) || !fixes.length) { wrap.innerHTML = ""; return; }
    wrap.innerHTML =
        `<div style="font-weight:700;color:${ACCENT};font-size:11px;text-transform:uppercase;letter-spacing:0.9px;margin-bottom:8px;opacity:0.92">How to fix it</div>` +
        `<ol style="margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:7px">` +
        fixes.slice(0, 6).map((f, i) =>
            `<li style="background:${ROW_BG};border-left:2px solid ${ACCENT};border-radius:0 9px 9px 0;
                  padding:9px 11px;font-size:13.5px;color:${TEXT};display:flex;gap:9px;line-height:1.5">
                <span style="color:${ACCENT};font-weight:700;flex:0 0 auto;opacity:0.85">${i + 1}</span>
                <span style="flex:1">${_esc(f)}</span>
             </li>`).join("") +
        `</ol>`;
}
function _appendRaw(panel, raw) {
    const body = panel.querySelector(".body");
    if (!body) return;
    body.querySelector(".details")?.remove();
    const det = document.createElement("details");
    det.className = "details";
    det.style.cssText = `margin-top:14px;font-size:11px;color:${TEXT_DIM}`;
    det.innerHTML =
        `<summary style="cursor:pointer;color:${TEXT_DIM};font-weight:600;font-size:11.5px;letter-spacing:0.2px;opacity:0.85">Show technical details</summary>` +
        `<pre class="raw" style="margin:8px 0 0;padding:10px;background:rgba(0,0,0,0.3);border:1px solid ${HAIR};border-radius:9px;
              white-space:pre-wrap;word-break:break-word;max-height:160px;overflow:auto;
              font-family:ui-monospace,monospace;font-size:11px;color:${TEXT_DIM};line-height:1.5"></pre>`;
    body.appendChild(det);
    const pre = det.querySelector(".raw");
    if (pre) pre.textContent = raw || "";
}

/** Show the panel right away with the deterministic JS humaniser, then upgrade. */
function showError(errorText, context, nodeClass, nodeId) {
    const panel = ensurePanel();
    panel.querySelector(".ctx").textContent = context || "execution error";
    _setFriendly(panel, _c2cHumanise(errorText));
    _setCause(panel, "");
    _setFixes(panel, []);
    _appendRaw(panel, errorText);
    panel.querySelector(".status").textContent = "Reading the error…";

    // Remember the raw error for "Ask AI for more".
    panel._c2cRaw = errorText;
    panel._c2cCtx = context;
    panel.querySelector(".askai").onclick = () => askAI(errorText, context);

    _upgradeViaBackend(panel, errorText, context, nodeClass, nodeId);
}

/** POST to /mec/translate_error — runs the backend 3-tier explainer.
 *  Tier-1 always fires offline; the route honours the user's saved mode. */
async function _upgradeViaBackend(panel, errorText, context, nodeClass, nodeId) {
    try {
        const resp = await fetch("/mec/translate_error", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: errorText,
                node_class: nodeClass || undefined,
                node_id: nodeId != null ? nodeId : undefined,
            }),
        });
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const json = await resp.json().catch(() => null);
        const data = (json && (json.data || json)) || {};
        const cause = String(data.cause || "").trim();
        const headline = String(data.headline || "").trim();
        const fixes = Array.isArray(data.fixes) ? data.fixes : (Array.isArray(data.fix_steps) ? data.fix_steps : []);
        if (cause || fixes.length) {
            // Headline stays the simple humanised one; show the curated cause below it.
            if (cause) _setCause(panel, cause);
            _setFixes(panel, fixes);
            const tier = data.tier;
            const label = tier === 1 ? "Offline rule pack"
                        : tier === 2 ? "Local AI"
                        : tier >= 3 ? "Cloud AI"
                        : tier === "introspector" ? "Live tensor analysis"
                        : "Explained";
            panel.querySelector(".status").textContent = label;
            _appendRaw(panel, errorText);
            return;
        }
        // Route reachable but produced nothing useful → keep the JS humanise.
        panel.querySelector(".status").textContent = "Explained (basic)";
    } catch (err) {
        // Route unreachable → the JS humaniser already populated the panel.
        __c2cReport("c2c_ai_error_translator:backend", err, { level: "info" });
        panel.querySelector(".status").textContent = "Offline — basic explanation";
    }
}

/** Escalate to the streaming LLM endpoint for a deeper, model-written answer. */
async function askAI(errorText, context) {
    const panel = ensurePanel();
    const raw = errorText || panel._c2cRaw || "";
    const ctx = context || panel._c2cCtx || "";
    panel.querySelector(".status").textContent = "Asking AI…";
    _setCause(panel, "");
    _setFixes(panel, []);

    const sys = "You are a senior ComfyUI engineer. The user just hit a Python traceback and does NOT know technical terms. Explain in plain English what went wrong and exactly how to fix it. Be terse. Use short bullet points.";
    const user = "Context: " + (ctx || "(none)") + "\n\nError:\n" + raw;

    try {
        const r = await fetch("/c2c/ai/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
            body: JSON.stringify({
                feature: "error_translator",
                sensitivity: "sensitive",
                max_tokens: 500,
                temperature: 0.2,
                messages: [{ role: "system", content: sys }, { role: "user", content: user }],
            }),
        });
        if (!r.ok || !r.body) {
            const j = await r.json().catch(() => ({}));
            _setCause(panel, _c2cHumanise(raw) +
                "\n\nDeeper AI is unavailable: " + (j.message || ("HTTP " + r.status)) +
                ". The offline explanation above still applies. To enable AI, set up Ollama or a cloud key in C2C AI settings.");
            panel.querySelector(".status").textContent = "AI unavailable";
            return;
        }
        _setCause(panel, "");
        panel.querySelector(".status").textContent = "AI is explaining…";
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = "", out = "";
        const causeEl = panel.querySelector(".cause");
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            let idx;
            while ((idx = buf.indexOf("\n\n")) !== -1) {
                const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
                let event = "message", data = "";
                frame.split("\n").forEach(l => {
                    if (l.startsWith("event: ")) event = l.slice(7).trim();
                    else if (l.startsWith("data: ")) data += l.slice(6);
                });
                if (!data) continue;
                if (event === "error") {
                    out += "\n[error] " + data;
                    if (causeEl) causeEl.textContent = out;
                    panel.querySelector(".status").textContent = "AI error";
                } else if (event === "done") {
                    panel.querySelector(".status").textContent = "AI done";
                } else {
                    try {
                        const obj = JSON.parse(data);
                        if (obj.chunk) {
                            out += obj.chunk;
                            if (causeEl) causeEl.textContent = out;
                            panel.querySelector(".body").scrollTop = 0;
                        }
                    } catch (e) { __c2cReport("c2c_ai_error_translator:stream", e); }
                }
            }
        }
        window.__C2C_AI_HUD__?.refresh?.();
    } catch (exc) {
        _setCause(panel, _c2cHumanise(raw) + "\n\nCould not reach the AI helper: " + exc.message);
        panel.querySelector(".status").textContent = "AI error";
    }
}

app.registerExtension({
    name: "c2c.ai.errorTranslator",
    settings: [
        { id: SETTING_ENABLED,
          name: "C2C ▸ Errors ▸ Show big plain-English error panel",
          tooltip: "When a workflow fails, pop a large red panel that explains the error in simple English with fix steps.",
          type: "boolean", default: true },
        { id: SETTING_AUTO_AI,
          name: "C2C ▸ Errors ▸ Auto-ask AI for a deeper explanation",
          tooltip: "Also stream a model-written explanation automatically (needs Ollama or a cloud key). Off = offline rule pack only.",
          type: "boolean", default: false },
    ],
    async setup() {
        const _enabledNow = () => {
            try { return app.ui?.settings?.getSettingValue(SETTING_ENABLED, true) !== false; } catch { return true; }
        };
        const _maybeAutoAI = (text, ctx) => {
            let autoAI = false;
            try { autoAI = app.ui?.settings?.getSettingValue(SETTING_AUTO_AI, false); } catch {}
            if (autoAI) setTimeout(() => askAI(text, ctx), 400);
        };

        // 1) RUNTIME errors — exceptions raised while a node executes.
        api.addEventListener("execution_error", (ev) => {
            if (!_enabledNow()) return;
            const detail = ev?.detail || {};
            const text = [
                detail.exception_message || "",
                Array.isArray(detail.traceback) ? detail.traceback.join("") : (detail.traceback || ""),
            ].filter(Boolean).join("\n\n");
            if (!text) return;
            const nodeType = detail.node_type || "";
            const nodeId = detail.node_id;
            const ctx = nodeType ? `in node ${nodeId} (${nodeType})` : "during this run";
            // Defer so we appear AFTER ComfyUI's own native error toast/dialog.
            setTimeout(() => { showError(text, ctx, nodeType, nodeId); _maybeAutoAI(text, ctx); }, 60);
        });

        // 2) VALIDATION errors — missing input / bad value / wrong wiring. These
        //    come back from POST /prompt as a 400 {error, node_errors} and DO NOT
        //    fire execution_error, so without this the user only sees ComfyUI's
        //    raw technical toast. Wrap queuePrompt to surface the same red panel.
        try {
            const origQueue = api.queuePrompt && api.queuePrompt.bind(api);
            if (origQueue && !api.__c2cErrTransWrapped) {
                api.__c2cErrTransWrapped = true;
                api.queuePrompt = async function (number, prompt) {
                    try {
                        return await origQueue(number, prompt);
                    } catch (err) {
                        if (_enabledNow()) {
                            const msg = _formatValidation(err);
                            if (msg && msg.trim()) {
                                setTimeout(() => { showError(msg, "before the run started (validation)", "", null); _maybeAutoAI(msg, "validation"); }, 80);
                            }
                        }
                        throw err;   // rethrow so ComfyUI's native handling still runs
                    }
                };
            }
        } catch (e) { __c2cReport("c2c_ai_error_translator:queueWrap", e); }
    },
});

window.__C2C_AI_ERRTRANS__ = { showError, askAI, translate: showError };
