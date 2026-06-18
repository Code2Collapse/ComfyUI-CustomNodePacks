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
import { findNodeAnywhere } from "./_subgraph_walk.js";
import { streamAI } from "./_c2c_ai_client.js";
import { buildPanel, esc, C } from "./_c2c_window.js";
import { getPrompt } from "./_c2c_prompts.js";
import { asOneUndoAsync } from "./_c2c_undo.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const BTN_ID = "c2c-doctor-btn";
const PANEL_ID = "c2c-doctor-panel";
const SETTING_SHOW_BTN = "c2c.doctor.showButton";

const SEV_COLOR = { error: C.red, warning: C.yellow, info: C.blue };
const SEV_ICON  = { error: "✖", warning: "⚠", info: "ℹ" };

function focusNode(nid) {
    // Subgraph-aware: nid may be inside a nested subgraph the root
    // graph cannot find directly.
    const resolved = findNodeAnywhere(nid);
    const n = resolved?.node || app.graph?.getNodeById?.(nid);
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
    const cls = sev === "error" ? "error" : sev === "warning" ? "warn" : "info";
    return `<span class="c2c-win-chip ${cls}" style="margin-right:6px">${SEV_ICON[sev]} ${count} ${sev}</span>`;
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
    let existing = document.getElementById(PANEL_ID);
    if (existing && existing._refs) return existing._refs;
    const refs = buildPanel({
        id: PANEL_ID,
        title: "Workflow Doctor",
        width: 540,
        height: 600,
        storageKey: "doctor",
        tabs: [
            { key: "findings", label: "🧩 Findings" },
            { key: "ai",       label: "🧠 AI Review" },
            { key: "stats",    label: "📊 Stats" },
        ],
        actions: [
            { label: "↻", className: "icon", id: "rerun", title: "Re-analyze",
              onClick: () => refresh(refs) },
            { label: "�", className: "icon", id: "browse", title: "Browse live GitHub issues for this workflow",
              onClick: () => window.__C2C_PRESET_HUB__?.open({ tab: "c2c_doctor" }) },
            { label: "�📋", className: "icon", id: "copy", title: "Copy findings JSON",
              onClick: () => {
                  navigator.clipboard?.writeText(JSON.stringify(_lastStaticReport, null, 2));
                  refs.setStatus("copied");
                  setTimeout(() => refs.setStatus(""), 1200);
              } },
        ],
    });
    refs.el._refs = refs;
    refs.el.addEventListener("c2c:tab", () => _renderActiveTab(refs));
    return refs;
}

let _lastStaticReport = null;
let _lastAiText = "";

function _renderActiveTab(refs) {
    const tab = refs.el.dataset.activeTab || "findings";
    if (tab === "findings") {
        _renderFindings(refs);
    } else if (tab === "ai") {
        _renderAITab(refs);
    } else if (tab === "stats") {
        _renderStats(refs);
    }
}

function _renderFindings(refs) {
    const body = refs.body;
    if (!_lastStaticReport) {
        body.innerHTML = `<div style="color:${C.sub};padding:10px">Click ↻ to analyse the current workflow.</div>`;
        return;
    }
    const { findings } = _lastStaticReport;
    if (!findings || findings.length === 0) {
        body.innerHTML =
            `<div style="padding:20px;text-align:center;color:${C.green}">
               ✅ No issues detected.<br/>
               <span style="color:${C.sub};font-size:10px">Click the AI Review tab for a semantic pass.</span>
             </div>`;
        return;
    }
    const order = { error: 0, warning: 1, info: 2 };
    const sorted = [...findings].sort((a, b) => (order[a.severity] - order[b.severity]) || a.id.localeCompare(b.id));
    body.innerHTML = "";
    sorted.forEach((f, idx) => {
        const row = document.createElement("div");
        row.style.cssText =
            `border-left:3px solid ${SEV_COLOR[f.severity]};background:${C.bg2};
             padding:6px 8px;margin-bottom:6px;border-radius:0 4px 4px 0`;
        const nodeLink = f.node_id != null
            ? `<a class="nodelink" data-id="${f.node_id}" style="color:${C.blue};cursor:pointer;text-decoration:underline">#${f.node_id} ${esc(f.node_type || "")}</a>`
            : `<span style="color:${C.sub}">graph</span>`;
        const fixBtn = f.fix
            ? `<button class="c2c-apply-fix" data-idx="${idx}" style="margin-top:4px;background:${C.bg};color:${C.fg};border:1px solid ${C.border};border-radius:3px;padding:2px 6px;font-size:10px;cursor:pointer">Apply fix: ${esc(f.fix.label || "auto")}</button>`
            : "";
        row.innerHTML =
            `<div style="display:flex;gap:6px;align-items:center">
               <span style="color:${SEV_COLOR[f.severity]};font-weight:600;font-size:11px">${SEV_ICON[f.severity]} ${esc(f.title)}</span>
               <span style="flex:1"></span>
               ${nodeLink}
             </div>
             <div style="margin-top:3px;color:${C.fg};font-size:11px">${esc(f.detail || "")}</div>
             ${f.fix_hint ? `<div style="margin-top:2px;color:${C.sub};font-size:10px">💡 ${esc(f.fix_hint)}</div>` : ""}
             ${fixBtn}`;
        body.appendChild(row);
    });
    body.querySelectorAll(".nodelink").forEach(a => {
        a.onclick = () => focusNode(parseInt(a.dataset.id, 10));
    });
    body.querySelectorAll(".c2c-apply-fix").forEach(btn => {
        btn.onclick = async () => {
            const i = parseInt(btn.dataset.idx, 10);
            const f = sorted[i];
            if (!f || !f.fix) return;
            btn.disabled = true;
            btn.textContent = "Applying…";
            try {
                const ok = await _applyDoctorFix(f.fix);
                if (ok) {
                    refs.setStatus("fix applied");
                    setTimeout(() => refresh(refs), 100);
                } else {
                    btn.disabled = false;
                    btn.textContent = "Apply fix failed — retry";
                    refs.setStatus("fix failed");
                }
            } catch (e) {
                btn.disabled = false;
                btn.textContent = "Apply fix error — retry";
                refs.setStatus("fix error: " + (e?.message || e));
            }
        };
    });
}

/**
 * Apply a server-emitted `fix` payload to the live LiteGraph, wrapped in a
 * single undo step. Supported kinds: set_widget, set_widget_many, set_mode.
 * Returns true on success, false on no-op / target missing.
 */
async function _applyDoctorFix(fix) {
    if (!fix || typeof fix !== "object") return false;
    const node = findNodeAnywhere(fix.node_id)?.node;
    if (!node) return false;
    let mutated = false;
    await asOneUndoAsync(`doctor: ${fix.label || fix.kind}`, () => {
        if (fix.kind === "set_widget") {
            const w = node.widgets?.find(w => w.name === fix.widget);
            if (!w) return;
            w.value = fix.value;
            try { w.callback?.(fix.value, app.canvas, node); } catch (__c2cErr) { __c2cReport("c2c_workflow_doctor", __c2cErr); }
            mutated = true;
        } else if (fix.kind === "set_widget_many") {
            for (const change of (fix.changes || [])) {
                const w = node.widgets?.find(w => w.name === change.widget);
                if (!w) continue;
                w.value = change.value;
                try { w.callback?.(change.value, app.canvas, node); } catch (__c2cErr) { __c2cReport("c2c_workflow_doctor", __c2cErr); }
                mutated = true;
            }
        } else if (fix.kind === "set_mode") {
            node.mode = fix.value;
            mutated = true;
        }
        node.setDirtyCanvas?.(true, true);
    });
    return mutated;
}

function _renderAITab(refs) {
    refs.body.innerHTML =
        `<div style="display:flex;gap:8px;margin-bottom:8px">
           <button class="c2c-win-btn primary" id="_ai-run">🧠 Ask AI for deep review</button>
         </div>
         <div class="body-ai" style="white-space:pre-wrap;line-height:1.55">${esc(_lastAiText || "")}</div>`;
    refs.body.querySelector("#_ai-run").onclick = () => askAI(refs);
}

function _renderStats(refs) {
    if (!_lastStaticReport) {
        refs.body.innerHTML = `<div style="color:${C.sub};padding:10px">Run an analysis first.</div>`;
        return;
    }
    const s = _lastStaticReport.stats || {};
    const rows = Object.entries(s).map(([k, v]) =>
        `<tr><td style="color:${C.sub};padding:4px 10px">${esc(k)}</td><td style="color:${C.green};font-weight:600;padding:4px 10px">${esc(String(v))}</td></tr>`).join("");
    refs.body.innerHTML = `<table style="width:100%;border-collapse:collapse"><tbody>${rows}</tbody></table>`;
}

async function refresh(refs) {
    refs.setStatus("analysing…");
    refs.setMeta("");
    refs.setSubtitle("");
    refs.body.innerHTML = `<div style="color:${C.sub};padding:10px">Analysing…</div>`;
    const res = await analyze();
    if (!res.success) {
        refs.body.innerHTML = `<div style="color:${C.red};padding:10px">Analyse failed: ${esc(res.error || "")}</div>`;
        _lastStaticReport = null;
        refs.setStatus("error");
        return;
    }
    _lastStaticReport = res.data;
    const { summary, stats } = res.data;
    refs.setSubtitle(
        severityChip("error", summary.errors) +
        severityChip("warning", summary.warnings) +
        severityChip("info", summary.infos));
    refs.setMeta(`${stats.nodes}n · ${stats.links}l · ${stats.samplers}s`);
    refs.setStatus("");
    _renderActiveTab(refs);
}

async function askAI(refs) {
    if (!_lastStaticReport) { await refresh(refs); }
    const wf = app.graph?.serialize?.();
    if (!wf) return;
    const summary = compactGraphSummary(wf);
    const sysPrompt = await getPrompt("workflow_doctor.system");
    const userPrompt =
        "Static findings:\n" + JSON.stringify(_lastStaticReport?.findings || [], null, 2) +
        "\n\nWorkflow summary:\n" + summary;

    refs.el.dataset.activeTab = "ai";
    refs.el.querySelectorAll(".c2c-win-tab").forEach(t => {
        t.classList.toggle("active", t.dataset.key === "ai");
    });
    _renderAITab(refs);
    const out = refs.body.querySelector(".body-ai");
    _lastAiText = "";
    refs.setStatus("streaming…");

    await streamAI({
        feature: "workflow_doctor",
        sensitivity: "normal",
        max_tokens: 800,
        temperature: 0.2,
        messages: [
            { role: "system", content: sysPrompt },
            { role: "user",   content: userPrompt },
        ],
        onStatus: (s) => { refs.setStatus(s === "streaming" ? "streaming…" : s); },
        onChunk: (chunk) => {
            _lastAiText += chunk;
            if (out) { out.textContent = _lastAiText; refs.body.scrollTop = refs.body.scrollHeight; }
        },
        onError: (err) => {
            _lastAiText += "\n[error] " + err.message;
            if (out) out.textContent = _lastAiText;
            refs.setStatus("error");
        },
    });
    refs.setStatus("done");
}

function ensureAIPanel() {
    // Legacy entry point retained for any external callers; just opens the
    // doctor panel and flips to the AI Review tab.
    const refs = ensurePanel();
    refs.el.dataset.activeTab = "ai";
    _renderActiveTab(refs);
    return refs.el;
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
    // DoctorV3 mega-panel (c2c_doctor.js) supersedes this lighter version.
    // If V3 is loaded, skip to avoid duplicate Doctor entries in the OmniBar.
    if (window.__C2C_DOCTOR_V3__) { document.getElementById(BTN_ID)?.remove(); return; }
    const raw = app.ui?.settings?.getSettingValue(SETTING_SHOW_BTN);
    // Newer ComfyUI Vue settings store ignores the default-arg of
    // getSettingValue; undefined => use declared default (true).
    const show = (raw === undefined || raw === null) ? true : !!raw;
    let btn = document.getElementById(BTN_ID);
    if (!show) { btn?.remove(); return; }
    if (btn) return;
    btn = document.createElement("button");
    btn.id = BTN_ID;
    btn.title = "C2C — Workflow Doctor (Ctrl+Alt+D)";
    btn.textContent = "🩺 Doctor";
    // Positioning is delegated to __c2cTopDock; the dock sits below the
    // ComfyUI top menu and reflows with sidebar / tab toggles.
    btn.style.cssText =
        `background:${C.bg};color:${C.fg};border:1px solid ${C.border};
         border-radius:10px;padding:3px 10px;cursor:pointer;font-size:11px;
         font-family:ui-sans-serif,system-ui,sans-serif;`;
    btn.onmouseenter = () => btn.style.borderColor = C.mauve;
    btn.onmouseleave = () => btn.style.borderColor = C.border;
    btn.onclick = () => { const refs = ensurePanel(); refresh(refs); };
    if (window.__c2cTopDock) {
        window.__c2cTopDock.register(btn, { side: "left", order: 20 });
    } else {
        // Fallback: pin top-left with safe fixed positioning so the
        // button never lays out as a full-width static block element
        // (button default is display:block → it would stretch to the
        // body's content box). Retry dock registration on the next
        // animation frame in case _c2c_top_dock.js loads after us.
        btn.style.position = "fixed";
        btn.style.top = "60px";
        btn.style.left = "82px";
        btn.style.setProperty("z-index", "var(--c2c-z-dock, 2500)");
        document.body.appendChild(btn);
        const tryRegister = () => {
            if (window.__c2cTopDock) {
                btn.style.position = "";
                btn.style.top = "";
                btn.style.left = "";
                btn.style.removeProperty("z-index");
                window.__c2cTopDock.register(btn, { side: "left", order: 20 });
            } else {
                requestAnimationFrame(tryRegister);
            }
        };
        requestAnimationFrame(tryRegister);
    }
}

app.registerExtension({
    name: "c2c.workflow.doctor",
    settings: [
        { id: SETTING_SHOW_BTN, name: "C2C â–¸ Doctor â–¸ Show top-left button",
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

window.__C2C_DOCTOR__ = { analyze, refresh, _applyDoctorFix };
