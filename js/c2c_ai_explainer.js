/**
 * c2c_ai_explainer.js — "Explain this node with AI" with rich tabbed UI.
 *
 * Behaviour upgrade 2026-05-19:
 *   - Built on the shared _c2c_window manager: draggable header, resizable
 *     bottom-right grip, pinnable, position+size persisted to localStorage.
 *   - Six tabs: Explain / Inputs / Outputs / Widgets / Show me / Raw.
 *   - Header actions: 📋 copy AI answer · ↻ regenerate · 📌 pin · × close.
 *   - Token counter in the footer.
 */
import { app } from "../../scripts/app.js";
import { streamAI } from "./_c2c_ai_client.js";
import { buildPanel, esc, nodeAnchor, clearConnector } from "./_c2c_window.js";
import { getPrompt } from "./_c2c_prompts.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";

const PANEL_ID = "c2c-ai-explainer-panel";
const SETTING_ENABLE = "c2c.ai.explainer.enabled";

function nodeSummary(node) {
    const lines = [];
    lines.push(`class_type: ${node.comfyClass || node.type}`);
    lines.push(`title: ${node.title || node.type}`);
    if (Array.isArray(node.inputs) && node.inputs.length) {
        lines.push("inputs:");
        node.inputs.forEach(i => lines.push(`  - ${i.name} (${i.type})${i.link != null ? " [linked]" : ""}`));
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
    return buildPanel({
        id: PANEL_ID,
        title: "AI Explainer",
        width: 480,
        height: 540,
        storageKey: "ai_explainer",
        tabs: [
            { key: "explain",  label: "🧠 Explain" },
            { key: "inputs",   label: "↓ Inputs" },
            { key: "outputs",  label: "↑ Outputs" },
            { key: "widgets",  label: "⚙ Widgets" },
            { key: "demo",     label: "🎬 Show me" },
            { key: "raw",      label: "{} Raw" },
        ],
        actions: [
            { label: "📋", className: "icon", id: "copy",  title: "Copy AI answer to clipboard",
              onClick: (el) => {
                  const txt = el.querySelector(".body-explain")?.innerText
                            || el.querySelector(".c2c-win-body")?.innerText || "";
                  navigator.clipboard?.writeText(txt);
                  el.querySelector(".status").textContent = "copied";
                  setTimeout(() => { el.querySelector(".status").textContent = ""; }, 1400);
              } },
            { label: "↻", className: "icon", id: "regen", title: "Regenerate the AI answer",
              onClick: (el) => { if (el._lastNode) explain(el._lastNode); } },
        ],
    });
}

function _renderTabs(refs, node, aiText) {
    const el = refs.el;
    const active = el.dataset.activeTab || "explain";
    const body = refs.body;
    body.innerHTML = "";

    if (active === "explain") {
        const d = document.createElement("div");
        d.className = "body-explain";
        d.style.cssText = "white-space:pre-wrap;line-height:1.55";
        d.textContent = aiText || "…";
        body.appendChild(d);
    } else if (active === "inputs") {
        const rows = (node.inputs || []).map(i => {
            const chip = i.link != null
                ? `<span class="c2c-win-chip ok">connected</span>`
                : `<span class="c2c-win-chip">free</span>`;
            return `<tr>
                <td style="color:var(--c2c-green);font-weight:600;padding:3px 8px;vertical-align:top;white-space:nowrap">${esc(i.name)}</td>
                <td style="color:var(--c2c-peach);padding:3px 8px;vertical-align:top;font-family:ui-monospace,monospace;font-size:11px">${esc(i.type)}</td>
                <td style="padding:3px 8px;vertical-align:top">${chip}</td>
            </tr>`;
        }).join("");
        body.innerHTML = rows
            ? `<table style="width:100%;border-collapse:collapse"><tbody>${rows}</tbody></table>`
            : `<div style="color:var(--c2c-sub)">No inputs declared on this node.</div>`;
    } else if (active === "outputs") {
        const rows = (node.outputs || []).map(o => {
            const n = Array.isArray(o.links) ? o.links.length : 0;
            const chip = n ? `<span class="c2c-win-chip ok">${n} link${n === 1 ? "" : "s"}</span>`
                           : `<span class="c2c-win-chip">unused</span>`;
            return `<tr>
                <td style="color:var(--c2c-green);font-weight:600;padding:3px 8px;white-space:nowrap">${esc(o.name)}</td>
                <td style="color:var(--c2c-peach);padding:3px 8px;font-family:ui-monospace,monospace;font-size:11px">${esc(o.type)}</td>
                <td style="padding:3px 8px">${chip}</td>
            </tr>`;
        }).join("");
        body.innerHTML = rows
            ? `<table style="width:100%;border-collapse:collapse"><tbody>${rows}</tbody></table>`
            : `<div style="color:var(--c2c-sub)">No outputs declared.</div>`;
    } else if (active === "widgets") {
        const rows = (node.widgets || []).map(w => {
            let v = w.value;
            const isLong = typeof v === "string" && v.length > 60;
            const vs = isLong ? v.slice(0, 60) + "…" : JSON.stringify(v);
            return `<tr>
                <td style="color:var(--c2c-green);font-weight:600;padding:3px 8px;white-space:nowrap;vertical-align:top">${esc(w.name)}</td>
                <td style="padding:3px 8px;vertical-align:top"><code>${esc(vs)}</code></td>
            </tr>`;
        }).join("");
        body.innerHTML = rows
            ? `<table style="width:100%;border-collapse:collapse"><tbody>${rows}</tbody></table>`
            : `<div style="color:var(--c2c-sub)">No widgets on this node.</div>`;
    } else if (active === "demo") {
        body.innerHTML =
            `<div style="color:var(--c2c-sub);margin-bottom:10px;line-height:1.5">Generate a minimal example workflow that uses this node and insert it next to your cursor. Ctrl+Z removes it.</div>
             <button class="c2c-win-btn success" id="_demo-go">🎬 Generate &amp; insert</button>
             <pre class="_demo-out" style="margin-top:12px;max-height:280px;display:none"></pre>`;
        body.querySelector("#_demo-go").onclick = () => showMe(node.comfyClass || node.type, refs);
    } else if (active === "raw") {
        body.innerHTML = `<pre style="font-size:11px;line-height:1.4">${esc(nodeSummary(node))}</pre>`;
    }
}

async function explain(node) {
    const refs = ensurePanel();
    refs.el._lastNode = node;
    refs.el.dataset.nodeClass = node.comfyClass || node.type;
    refs.setSubtitle(`${node.title || node.type}  ·  ${node.comfyClass || node.type}`);
    refs.setStatus("asking AI…");
    refs.setMeta("");

    // Anchor the panel beside the actual node the user right-clicked, so
    // the explanation card lives visually next to its subject (not in a
    // cascaded top-right corner where the user has to hunt for it).
    // Only auto-anchor if the user hasn't manually dragged the panel yet
    // for this session — otherwise their preferred position wins.
    if (!refs.el.dataset.userMoved) {
        if (refs.el._anchorHandle) { try { refs.el._anchorHandle.detach(); } catch (__c2cErr) { __c2cReport("c2c_ai_explainer", __c2cErr); } }
        refs.el._anchorHandle = nodeAnchor(refs.el, node, { gap: 14, prefer: "right" });
    }
    // Mark as user-moved on the first manual drag so re-explaining another
    // node doesn't yank the panel around.
    if (!refs.el._dragHookInstalled) {
        refs.el._dragHookInstalled = true;
        refs.el.querySelector(".c2c-win-header")?.addEventListener("mousedown", () => {
            refs.el.dataset.userMoved = "true";
            if (refs.el._anchorHandle) { try { refs.el._anchorHandle.detach(); } catch (__c2cErr) { __c2cReport("c2c_ai_explainer", __c2cErr); } refs.el._anchorHandle = null; }
            clearConnector();
        });
    }

    let aiText = "";
    _renderTabs(refs, node, aiText);

    // Hook tab switching to re-render with the current aiText snapshot.
    if (!refs.el._tabHookInstalled) {
        refs.el._tabHookInstalled = true;
        refs.el.addEventListener("c2c:tab", () => {
            const node2 = refs.el._lastNode;
            if (node2) _renderTabs(refs, node2, refs.el._aiText || "");
        });
    }

    const sys = await getPrompt("node_explainer.system");
    const user = "Explain this node:\n\n" + nodeSummary(node);

    let tokens = 0;
    await streamAI({
        feature: "node_explainer",
        sensitivity: "normal",
        max_tokens: 600,
        temperature: 0.3,
        messages: [
            { role: "system", content: sys },
            { role: "user",   content: user },
        ],
        onStatus: (s) => { refs.setStatus(s === "streaming" ? "streaming…" : s); },
        onChunk:  (chunk) => {
            aiText += chunk;
            refs.el._aiText = aiText;
            tokens += Math.max(1, Math.round(chunk.length / 4));
            refs.setMeta(`~${tokens} tok`);
            if ((refs.el.dataset.activeTab || "explain") === "explain") {
                const d = refs.body.querySelector(".body-explain");
                if (d) { d.textContent = aiText; refs.body.scrollTop = refs.body.scrollHeight; }
            }
        },
        onError:  (err) => {
            aiText += `\n[error] ${err.message}`;
            refs.el._aiText = aiText;
            _renderTabs(refs, node, aiText);
            refs.setStatus("error");
        },
    });
    if ((refs.el.dataset.activeTab || "explain") === "explain") {
        const d = refs.body.querySelector(".body-explain");
        if (d) d.textContent = aiText;
    }
    refs.setStatus("done");
}

// ---------------------------------------------------------------------------
//  Tier-3: "Show me" — ask AI for a minimal example workflow JSON and insert
//  it next to the cursor. Ctrl+Z to undo.
// ---------------------------------------------------------------------------
async function showMe(nodeClass, refs) {
    const out = refs.body.querySelector("._demo-out");
    const show = (txt) => {
        if (!out) return;
        out.style.display = "block";
        out.textContent = txt;
    };
    refs.setStatus("generating example…");
    show("Asking AI…");

    let knownTypes = [];
    try {
        knownTypes = Object.keys(app?.graph?.constructor?.registered_node_types || {}).slice(0, 200);
        if (!knownTypes.length) {
            knownTypes = Object.keys(window.LiteGraph?.registered_node_types || {}).slice(0, 200);
        }
    } catch (__c2cErr) { __c2cReport("c2c_ai_explainer", __c2cErr); }

    const sys =
        "You output ONE JSON object only — no prose, no markdown fences. " +
        "Schema: { nodes: [ { class_type: string, title?: string, widgets_values?: array, pos: [x,y] } ], links: [ [from_node_index, from_output_index, to_node_index, to_input_index] ] }. " +
        "Generate the smallest possible working example (2–4 nodes) that demonstrates the target node in a real pipeline. Use only class_types from the provided registry.";
    const user =
        `Target node class: ${nodeClass}\n\nAllowed class_types (pick from this list):\n${knownTypes.join(", ")}`;

    let chunks = "";
    await streamAI({
        feature: "node_explainer",
        sensitivity: "normal",
        max_tokens: 800,
        temperature: 0.1,
        messages: [
            { role: "system", content: sys },
            { role: "user",   content: user },
        ],
        onStatus: (s) => refs.setStatus(s === "streaming" ? "streaming demo…" : s),
        onChunk:  (c) => { chunks += c; show(chunks); },
        onError:  (err) => { show(`[show-me error] ${err.message}`); refs.setStatus("error"); },
    });

    let payload = null;
    try {
        let raw = chunks.trim();
        const m = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
        if (m) raw = m[1].trim();
        const j0 = raw.indexOf("{"); const j1 = raw.lastIndexOf("}");
        if (j0 !== -1 && j1 > j0) raw = raw.slice(j0, j1 + 1);
        payload = JSON.parse(raw);
    } catch (e) {
        show(`[show-me] couldn't parse AI output as JSON: ${e.message}\n\nRaw:\n${chunks}`);
        refs.setStatus("parse error");
        return;
    }
    if (!payload || !Array.isArray(payload.nodes) || !payload.nodes.length) {
        show("[show-me] AI returned an empty workflow.");
        refs.setStatus("empty");
        return;
    }

    const ds = app.canvas?.ds;
    const rect = app.canvas?.canvas?.getBoundingClientRect();
    const cx = ds && rect ? (rect.width  / 2 - ds.offset[0]) / ds.scale : 100;
    const cy = ds && rect ? (rect.height / 2 - ds.offset[1]) / ds.scale : 100;
    const created = [];
    const LG = window.LiteGraph;
    if (!LG) { show("[show-me] LiteGraph unavailable."); return; }
    payload.nodes.forEach((nd, i) => {
        const ct = nd.class_type || nd.type;
        const inst = LG.createNode(ct);
        if (!inst) { created.push(null); return; }
        if (nd.title) inst.title = nd.title;
        const px = Array.isArray(nd.pos) ? nd.pos[0] : (i * 240);
        const py = Array.isArray(nd.pos) ? nd.pos[1] : 0;
        inst.pos = [cx + px, cy + py];
        if (Array.isArray(nd.widgets_values) && inst.widgets) {
            nd.widgets_values.forEach((v, wi) => { if (inst.widgets[wi]) inst.widgets[wi].value = v; });
        }
        app.graph.add(inst);
        created.push(inst);
    });
    (payload.links || []).forEach((lk) => {
        try {
            const [fromI, fromS, toI, toS] = lk;
            const a = created[fromI]; const b = created[toI];
            if (a && b) a.connect(fromS, b, toS);
        } catch (__c2cErr) { __c2cReport("c2c_ai_explainer", __c2cErr); }
    });
    const okCount = created.filter(Boolean).length;
    show(`✅ Inserted ${okCount}/${payload.nodes.length} nodes near the canvas centre.\nPress Ctrl+Z to undo if you don't want them.`);
    refs.setStatus(`inserted ${okCount} nodes`);
    app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "c2c.ai.explainer",
    settings: [
        { id: SETTING_ENABLE, name: "C2C ▸ AI ▸ Show 'Explain with AI' in node menu",
          type: "boolean", default: true },
    ],
    async setup() {
        // Hook LGraphCanvas.getNodeMenuOptions instead of per-nodeType prototype.
        // The per-nodeType approach gets clobbered by other extensions that wrap
        // getExtraMenuOptions in a non-chaining way (observed: a wrapper that
        // captures `i` (prior) and `t` (its own) closure refs and replaces our
        // wrapped fn outright, dropping our pushed item). Hooking the canvas
        // method runs exactly once per right-click and is immune to that war.
        const LGC = window.LGraphCanvas;
        if (!LGC || !LGC.prototype || !LGC.prototype.getNodeMenuOptions) {
            console.warn("[c2c.ai.explainer] LGraphCanvas.getNodeMenuOptions not found; menu item will not be installed.");
            return;
        }
        if (LGC.prototype.__c2c_explainer_installed__) return;
        const orig = LGC.prototype.getNodeMenuOptions;
        LGC.prototype.getNodeMenuOptions = function (node) {
            const options = orig.apply(this, arguments);
            try {
                // NOTE: in newer ComfyUI front-ends the 2nd arg to
                // getSettingValue (default) is ignored and the function
                // returns `undefined` when the setting hasn't been touched
                // by the user. Treat undefined === "use the setting's
                // declared default", which for us is true.
                const v = app.ui?.settings?.getSettingValue(SETTING_ENABLE);
                const enabled = (v === undefined || v === null) ? true : !!v;
                if (!enabled) return options;
                if (!Array.isArray(options)) return options;
                // Avoid duplicate insertion if some other code already added it.
                if (options.some(o => o && /Explain with AI/.test(o.content || ""))) return options;
                options.push(null);
                options.push({
                    content: "🧠 Explain with AI",
                    callback: () => explain(node),
                });
            } catch (e) { console.error("[c2c.ai.explainer] menu hook error:", e); }
            return options;
        };
        LGC.prototype.__c2c_explainer_installed__ = true;
    },
});

window.__C2C_AI_EXPLAINER__ = { explain, showMe };
