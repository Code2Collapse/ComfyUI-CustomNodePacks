/**
 * c2c_workflow_doctor.js — v2.0 P2: surface workflow_doctor static findings,
 * with an "Ask AI for deep review" handoff.
 *
 * UX:
 *   - Toolbar button (top-left, near auto-checkpoint history): "🩺 Doctor".
 *   - Click runs POST /c2c/doctor/analyze with the current serialized graph.
 *   - Panel lists findings grouped by severity. Each row links to the offending
 *     node (clicking centers and selects it).
 *   - "Ask AI for deep review" feeds the (redacted) graph + static findings
 *     into /c2c/ai/stream with feature="workflow_doctor".
 */
import { app } from "../../scripts/app.js";

const BTN_ID = "c2c-doctor-btn";
const PANEL_ID = "c2c-doctor-panel";
const SETTING_SHOW_BTN = "c2c.doctor.showButton";

const C = { bg:"#1e1e2e", bg2:"#181825", fg:"#cdd6f4", sub:"#a6adc8",
            border:"#313244", mauve:"#cba6f7", red:"#f38ba8",
            yellow:"#f9e2af", green:"#a6e3a1", blue:"#89b4fa" };

const SEV_COLOR = { error: C.red, warning: C.yellow, info: C.blue };
const SEV_ICON = { error: "✖", warning: "⚠", info: "ℹ" };

function focusNode(nid) {
    const n = app.graph?.getNodeById?.(nid);
    if (!n) return;
    app.canvas?.deselectAllNodes?.();
    app.canvas?.selectNode?.(n);
    if (app.canvas?.centerOnNode) {
        app.canvas.centerOnNode(n);
    } else if (app.canvas?.ds) {
        const ds = app.canvas.ds;
        const rect = app.canvas.canvas.getBoundingClientRect();
        ds.offset[0] = rect.width / 2 - (n.pos[0] + n.size[0] / 2) * ds.scale;
        ds.offset[1] = rect.height / 2 - (n.pos[1] + n.size[1] / 2) * ds.scale;
        app.canvas.setDirty(true, true);
    }
}

function severityChip(sev, count) {
    const col = SEV_COLOR[sev] || C.sub;
    return `<span style="background:${C.bg2};color:${col};border:1px solid ${col};
            border-radius:10px;padding:1px 8px;font-size:10px;margin-right:6px">
            ${SEV_ICON[sev]} ${count} ${sev}</span>`;
}

async function analyze() {
    const wf = app.graph?.serialize?.();
    if (!wf) return { success: false, error: "no_graph" };
    try {
        const r = await fetch("/c2c/doctor/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ workflow: wf }),
        });
        return await r.json();
    } catch (exc) {
        return { success: false, error: "fetch_failed", message: exc.message };
    }
}

function ensurePanel() {
    let p = document.getElementById(PANEL_ID);
    if (p) return p;
    p = document.createElement("div");
    p.id = PANEL_ID;
    // z-index 100000 keeps the × close button clickable above the
    // "What's Wired" chip (z=99995). top:96 leaves the chip uncovered.
    p.style.cssText =
        `position:fixed;top:96px;right:12px;width:480px;max-height:78vh;z-index:100000;
         background:${C.bg};color:${C.fg};border:1px solid ${C.border};border-radius:8px;
         box-shadow:0 12px 40px rgba(0,0,0,0.55);overflow:hidden;
         display:flex;flex-direction:column;
         font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;`;
    p.innerHTML =
        `<div style="padding:8px 12px;background:${C.bg2};display:flex;align-items:center;gap:8px;border-bottom:1px solid ${C.border}">
           <span style="color:${C.mauve};font-weight:600;letter-spacing:0.5px;text-transform:uppercase;font-size:11px">Workflow Doctor</span>
           <span class="summary" style="flex:1;font-size:11px"></span>
           <button class="rerun" style="background:${C.bg};color:${C.fg};border:1px solid ${C.border};border-radius:4px;padding:2px 8px;cursor:pointer;font-size:10px">↻</button>
           <button class="close" style="background:transparent;color:${C.sub};border:none;font-size:18px;cursor:pointer;line-height:1">×</button>
         </div>
         <div class="body" style="padding:8px;overflow:auto;flex:1"></div>
         <div class="footer" style="padding:6px 12px;background:${C.bg2};border-top:1px solid ${C.border};display:flex;gap:6px;align-items:center">
           <span class="stats" style="flex:1;color:${C.sub};font-size:10px"></span>
           <button class="ai" style="background:${C.mauve};color:${C.bg};border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px">🧠 Ask AI for deep review</button>
         </div>`;
    p.querySelector(".close").onclick = () => p.remove();
    p.querySelector(".rerun").onclick = () => refresh(p);
    p.querySelector(".ai").onclick = () => askAI(p);
    document.body.appendChild(p);
    return p;
}

let _lastStaticReport = null;

async function refresh(panel) {
    const body = panel.querySelector(".body");
    body.innerHTML = `<div style="color:${C.sub};padding:10px">Analysing…</div>`;
    panel.querySelector(".summary").textContent = "";
    panel.querySelector(".stats").textContent = "";
    const res = await analyze();
    if (!res.success) {
        body.innerHTML = `<div style="color:${C.red};padding:10px">Analyse failed: ${res.error || ""}</div>`;
        _lastStaticReport = null;
        return;
    }
    _lastStaticReport = res.data;
    const { summary, findings, stats } = res.data;
    panel.querySelector(".summary").innerHTML =
        severityChip("error", summary.errors) +
        severityChip("warning", summary.warnings) +
        severityChip("info", summary.infos);
    panel.querySelector(".stats").textContent =
        `${stats.nodes} nodes · ${stats.links} links · ${stats.samplers} sampler(s) · ${stats.checkpoints} ckpt · ${stats.loras} LoRA · ${stats.outputs} output(s)`;

    if (findings.length === 0) {
        body.innerHTML =
            `<div style="padding:20px;text-align:center;color:${C.green}">
               ✅ No issues detected.<br/>
               <span style="color:${C.sub};font-size:10px">
                 Static checks only — click "Ask AI for deep review" for a semantic pass.
               </span>
             </div>`;
        return;
    }

    // Group + sort: errors → warnings → infos
    const order = { error: 0, warning: 1, info: 2 };
    const sorted = [...findings].sort((a, b) => (order[a.severity] - order[b.severity]) || a.id.localeCompare(b.id));
    body.innerHTML = "";
    for (const f of sorted) {
        const row = document.createElement("div");
        row.style.cssText =
            `border-left:3px solid ${SEV_COLOR[f.severity]};background:${C.bg2};
             padding:6px 8px;margin-bottom:6px;border-radius:0 4px 4px 0`;
        const nodeLink = f.node_id != null
            ? `<a class="nodelink" data-id="${f.node_id}" style="color:${C.blue};cursor:pointer;text-decoration:underline">#${f.node_id} ${f.node_type || ""}</a>`
            : `<span style="color:${C.sub}">graph</span>`;
        row.innerHTML =
            `<div style="display:flex;gap:6px;align-items:center">
               <span style="color:${SEV_COLOR[f.severity]};font-weight:600;font-size:11px">${SEV_ICON[f.severity]} ${f.title}</span>
               <span style="flex:1"></span>
               ${nodeLink}
             </div>
             <div style="margin-top:3px;color:${C.fg};font-size:11px">${escapeHtml(f.detail || "")}</div>
             ${f.fix_hint ? `<div style="margin-top:2px;color:${C.sub};font-size:10px">💡 ${escapeHtml(f.fix_hint)}</div>` : ""}`;
        body.appendChild(row);
    }
    body.querySelectorAll(".nodelink").forEach(a => {
        a.onclick = () => focusNode(parseInt(a.dataset.id, 10));
    });
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
        ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

async function askAI(panel) {
    if (!_lastStaticReport) { await refresh(panel); }
    const wf = app.graph?.serialize?.();
    if (!wf) return;
    // Build a compact graph summary so we don't blow the context window.
    const summary = compactGraphSummary(wf);
    const sysPrompt =
        "You are a senior ComfyUI workflow reviewer. The user already ran the static checker. " +
        "Review the workflow for: model-family consistency, sampler/scheduler choice fit, " +
        "denoise/cfg/steps coherence, missing refiners or upscalers, redundant nodes, " +
        "smart-order issues, and high-value optimisations. Be terse, use bullet points. " +
        "End with the single highest-impact change.";
    const userPrompt =
        "Static findings:\n" + JSON.stringify(_lastStaticReport?.findings || [], null, 2) +
        "\n\nWorkflow summary:\n" + summary;

    // Open or reuse the AI Explainer panel-style modal here (inline below).
    const out = ensureAIPanel();
    out.querySelector(".body").textContent = "";
    out.querySelector(".status").textContent = "streaming…";
    try {
        const r = await fetch("/c2c/ai/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
            body: JSON.stringify({
                feature: "workflow_doctor",
                sensitivity: "normal",
                max_tokens: 800,
                temperature: 0.2,
                messages: [
                    { role: "system", content: sysPrompt },
                    { role: "user", content: userPrompt },
                ],
            }),
        });
        if (!r.ok || !r.body) {
            const j = await r.json().catch(() => ({}));
            out.querySelector(".body").textContent = "AI failed: " + (j.message || ("HTTP " + r.status));
            out.querySelector(".status").textContent = "error";
            return;
        }
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        const body = out.querySelector(".body");
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
                if (event === "error") { body.textContent += "\n[error] " + data; out.querySelector(".status").textContent = "error"; }
                else if (event === "done") { out.querySelector(".status").textContent = "done"; }
                else {
                    try {
                        const obj = JSON.parse(data);
                        if (obj.chunk) { body.textContent += obj.chunk; body.scrollTop = body.scrollHeight; }
                    } catch (_) {}
                }
            }
        }
        window.__C2C_AI_HUD__?.refresh?.();
    } catch (exc) {
        out.querySelector(".body").textContent = "Error: " + exc.message;
        out.querySelector(".status").textContent = "error";
    }
}

function ensureAIPanel() {
    const ID = "c2c-doctor-ai-panel";
    let p = document.getElementById(ID);
    if (p) return p;
    p = document.createElement("div");
    p.id = ID;
    p.style.cssText =
        `position:fixed;bottom:18px;right:18px;width:520px;max-height:60vh;z-index:9994;
         background:${C.bg};color:${C.fg};border:1px solid ${C.mauve};border-radius:8px;
         box-shadow:0 12px 40px rgba(0,0,0,0.55);overflow:hidden;
         display:flex;flex-direction:column;
         font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;`;
    p.innerHTML =
        `<div style="padding:8px 12px;background:${C.bg2};display:flex;align-items:center;gap:8px;border-bottom:1px solid ${C.border}">
           <span style="color:${C.mauve};font-weight:600;letter-spacing:0.5px;text-transform:uppercase;font-size:11px">AI Workflow Review</span>
           <span class="status" style="flex:1;color:${C.sub};font-size:11px"></span>
           <button class="close" style="background:transparent;color:${C.sub};border:none;font-size:18px;cursor:pointer;line-height:1">×</button>
         </div>
         <div class="body" style="padding:12px;overflow:auto;white-space:pre-wrap;flex:1;line-height:1.5"></div>`;
    p.querySelector(".close").onclick = () => p.remove();
    document.body.appendChild(p);
    return p;
}

function compactGraphSummary(wf) {
    const nodes = wf.nodes || [];
    const lines = [];
    lines.push(`nodes=${nodes.length} links=${(wf.links || []).length}`);
    const byType = new Map();
    for (const n of nodes) {
        const t = n.type || n.class_type || "?";
        byType.set(t, (byType.get(t) || 0) + 1);
    }
    lines.push("types: " + [...byType.entries()].map(([k, v]) => `${k}×${v}`).join(", "));
    // Sampler details
    for (const n of nodes) {
        const t = n.type || "";
        if (/KSampler/.test(t)) {
            lines.push(`sampler#${n.id} (${t}): widgets=${JSON.stringify(n.widgets_values || [])}`);
        }
    }
    return lines.join("\n");
}

function ensureButton() {
    const show = app.ui?.settings?.getSettingValue(SETTING_SHOW_BTN, true);
    let btn = document.getElementById(BTN_ID);
    if (!show) { btn?.remove(); return; }
    if (btn) return;
    btn = document.createElement("button");
    btn.id = BTN_ID;
    btn.title = "C2C — Workflow Doctor (Ctrl+Alt+D)";
    btn.textContent = "🩺 Doctor";
    btn.style.cssText =
        `position:fixed;top:6px;left:110px;z-index:9999;
         background:${C.bg};color:${C.fg};border:1px solid ${C.border};
         border-radius:10px;padding:3px 10px;cursor:pointer;font-size:11px;
         font-family:ui-sans-serif,system-ui,sans-serif;`;
    btn.onmouseenter = () => btn.style.borderColor = C.mauve;
    btn.onmouseleave = () => btn.style.borderColor = C.border;
    btn.onclick = () => { const p = ensurePanel(); refresh(p); };
    document.body.appendChild(btn);
}

app.registerExtension({
    name: "c2c.workflow.doctor",
    settings: [
        { id: SETTING_SHOW_BTN, name: "C2C ▸ Doctor ▸ Show top-left button",
          type: "boolean", default: true, onChange: ensureButton },
    ],
    commands: [
        { id: "c2c.doctor.open",
          label: "C2C: Open Workflow Doctor",
          function: () => { const p = ensurePanel(); refresh(p); } },
    ],
    keybindings: [
        { combo: { key: "d", ctrl: true, alt: true }, commandId: "c2c.doctor.open" },
    ],
    async setup() { ensureButton(); },
});

window.__C2C_DOCTOR__ = { analyze, refresh };
