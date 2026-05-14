/**
 * c2c_style_presets.js — v2.0 P2: Style Presets sidebar tab.
 *
 * Lists style presets from /c2c/styles, lets the user:
 *   - Browse + filter by category / model family.
 *   - Preview a preset (positive/negative + sampler defaults).
 *   - Apply a preset to the currently selected (or auto-detected)
 *     CLIPTextEncode pair + KSampler.
 *   - Save the currently-selected encoders as a new preset.
 *   - Edit name / category / model_hint of any non-built-in preset.
 *
 * Mirrors c2c_prompt_wizard's "findTargetEncoders" heuristic so the Apply
 * action does the obvious thing without the user having to wire anything up.
 */
import { app } from "../../scripts/app.js";

const TAB_ID = "c2c.styles";
const C = { bg:"#1e1e2e", bg2:"#181825", fg:"#cdd6f4", sub:"#a6adc8",
            border:"#313244", mauve:"#cba6f7", blue:"#89b4fa", green:"#a6e3a1",
            red:"#f38ba8", yellow:"#f9e2af", teal:"#94e2d5" };

// --------------------------------------------------------- API
async function apiList() {
    const r = await fetch("/c2c/styles"); const j = await r.json();
    return j.success ? j.data : [];
}
async function apiGet(id) {
    const r = await fetch("/c2c/styles/" + encodeURIComponent(id));
    const j = await r.json(); return j.success ? j.data : null;
}
async function apiSave(payload) {
    const r = await fetch("/c2c/styles", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    return await r.json();
}
async function apiDelete(id) {
    const r = await fetch("/c2c/styles/" + encodeURIComponent(id), { method: "DELETE" });
    return await r.json();
}

// --------------------------------------------------------- node helpers
function findTargetEncoders() {
    const sel = Object.values(app.canvas?.selected_nodes || {});
    const encs = sel.filter(n => /CLIPTextEncode/.test(n.type || ""));
    const all = (app.graph?._nodes || []).filter(n => /CLIPTextEncode/.test(n.type || ""));
    const samplers = (app.graph?._nodes || []).filter(n => /KSampler/.test(n.type || ""));
    const pool = encs.length >= 1 ? encs : all;
    if (pool.length === 0) return { positive: null, negative: null, sampler: samplers[0] || null };

    // Try to classify by KSampler wiring
    for (const s of samplers) {
        const neg = (s.inputs || []).find(i => i.name === "negative");
        const pos = (s.inputs || []).find(i => i.name === "positive");
        const negSrc = neg?.link != null ? app.graph.getNodeById(app.graph.links[neg.link]?.origin_id) : null;
        const posSrc = pos?.link != null ? app.graph.getNodeById(app.graph.links[pos.link]?.origin_id) : null;
        if (pool.includes(negSrc) || pool.includes(posSrc)) {
            return {
                positive: pool.includes(posSrc) ? posSrc : null,
                negative: pool.includes(negSrc) ? negSrc : null,
                sampler: s,
            };
        }
    }
    const sorted = [...pool].sort((a, b) => a.id - b.id);
    return { positive: sorted[0], negative: sorted[1] || null, sampler: samplers[0] || null };
}

function setEncoder(node, text) {
    if (!node) return false;
    const w = (node.widgets || []).find(w => w.name === "text");
    if (!w) return false;
    w.value = text;
    node.setDirtyCanvas?.(true, true);
    return true;
}

function applySamplerSettings(samplerNode, sampler) {
    if (!samplerNode || !sampler) return [];
    const applied = [];
    for (const [k, v] of Object.entries(sampler)) {
        const w = (samplerNode.widgets || []).find(w => w.name === k);
        if (!w) continue;
        w.value = v;
        applied.push(`${k}=${v}`);
    }
    if (applied.length) samplerNode.setDirtyCanvas?.(true, true);
    return applied;
}

// --------------------------------------------------------- view
let _state = { list: [], filterCat: "", filterModel: "", selectedId: null, detail: null };

async function buildView(root) {
    root.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.style.cssText = `padding:10px;background:${C.bg};color:${C.fg};
        font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;
        height:100%;overflow:auto;`;
    wrap.innerHTML =
        `<h3 style="margin:0 0 8px;color:${C.mauve};font-size:13px;letter-spacing:0.5px;text-transform:uppercase">Style Presets</h3>
         <div style="display:flex;gap:6px;margin-bottom:8px">
           <select id="sp-cat" style="${sel()}"><option value="">all categories</option><option>photoreal</option><option>anime</option><option>stylized</option><option>lighting</option><option>custom</option></select>
           <select id="sp-model" style="${sel()}"><option value="">all models</option><option>sdxl</option><option>sd1.5</option><option>flux</option><option>any</option></select>
           <button id="sp-refresh" style="${btnGhost()}">↻</button>
         </div>
         <div id="sp-list" style="border:1px solid ${C.border};border-radius:4px;max-height:280px;overflow:auto"></div>
         <h4 style="margin:12px 0 4px;color:${C.sub};font-size:10px;letter-spacing:0.5px;text-transform:uppercase">Preview / Edit</h4>
         <div id="sp-detail" style="background:${C.bg2};border:1px solid ${C.border};border-radius:4px;padding:8px;min-height:60px;font-size:11px;color:${C.sub}">Select a preset above.</div>
         <h4 style="margin:12px 0 4px;color:${C.sub};font-size:10px;letter-spacing:0.5px;text-transform:uppercase">Save current as preset</h4>
         <div id="sp-save" style="background:${C.bg2};border:1px solid ${C.border};border-radius:4px;padding:8px"></div>`;
    root.appendChild(wrap);

    wrap.querySelector("#sp-cat").onchange   = (e) => { _state.filterCat = e.target.value; renderList(wrap); };
    wrap.querySelector("#sp-model").onchange = (e) => { _state.filterModel = e.target.value; renderList(wrap); };
    wrap.querySelector("#sp-refresh").onclick = async () => { _state.list = await apiList(); renderList(wrap); };
    renderSaveBox(wrap);

    _state.list = await apiList();
    renderList(wrap);
}

function renderList(wrap) {
    const cont = wrap.querySelector("#sp-list");
    cont.innerHTML = "";
    let items = _state.list;
    if (_state.filterCat)   items = items.filter(p => p.category === _state.filterCat);
    if (_state.filterModel) items = items.filter(p => p.model_hint === _state.filterModel);
    if (items.length === 0) {
        cont.innerHTML = `<div style="padding:14px;color:${C.sub};text-align:center">No presets match.</div>`;
        return;
    }
    items.sort((a, b) => (a.builtin === b.builtin ? a.name.localeCompare(b.name) : (a.builtin ? -1 : 1)));
    for (const p of items) {
        const row = document.createElement("div");
        row.style.cssText =
            `padding:6px 8px;border-bottom:1px solid ${C.border};cursor:pointer;
             display:flex;align-items:center;gap:6px`;
        row.onmouseenter = () => row.style.background = C.bg2;
        row.onmouseleave = () => row.style.background = _state.selectedId === p.id ? C.bg2 : "transparent";
        if (_state.selectedId === p.id) row.style.background = C.bg2;
        row.innerHTML =
            `<div style="flex:1">
               <div style="font-weight:600">${escapeHtml(p.name)} ${p.builtin ? `<span style="color:${C.teal};font-size:9px;border:1px solid ${C.teal};border-radius:6px;padding:0 4px;margin-left:4px">built-in</span>` : ""}</div>
               <div style="color:${C.sub};font-size:10px">${p.category} · ${p.model_hint}${p.tags?.length ? ` · ${p.tags.join(", ")}` : ""}</div>
             </div>`;
        row.onclick = async () => {
            _state.selectedId = p.id;
            _state.detail = await apiGet(p.id);
            renderList(wrap); renderDetail(wrap);
        };
        cont.appendChild(row);
    }
}

function renderDetail(wrap) {
    const cont = wrap.querySelector("#sp-detail");
    const d = _state.detail;
    if (!d) { cont.textContent = "Select a preset above."; return; }
    const sampler = d.sampler || {};
    const sampLine = Object.entries(sampler).length
        ? Object.entries(sampler).map(([k,v]) => `<code style="color:${C.teal}">${k}=${escapeHtml(String(v))}</code>`).join(" · ")
        : `<span style="color:${C.sub}">no sampler defaults</span>`;
    cont.innerHTML =
        `<div style="display:flex;gap:6px;align-items:center;margin-bottom:6px">
           <strong style="color:${C.fg}">${escapeHtml(d.name)}</strong>
           <span style="color:${C.sub};font-size:10px">${d.category} · ${d.model_hint}</span>
           <span style="flex:1"></span>
           <button class="apply" style="${btnPrimary()}">Apply</button>
           ${d.builtin ? "" : `<button class="del" style="${btnDanger()}">Delete</button>`}
         </div>
         <div style="margin-bottom:4px;color:${C.green};font-size:10px;letter-spacing:0.5px;text-transform:uppercase">Positive</div>
         <div style="background:${C.bg};border:1px solid ${C.border};border-radius:3px;padding:5px;font-family:ui-monospace,Menlo,monospace;font-size:10px;white-space:pre-wrap;max-height:90px;overflow:auto">${escapeHtml(d.positive || "")}</div>
         <div style="margin-top:6px;margin-bottom:4px;color:${C.red};font-size:10px;letter-spacing:0.5px;text-transform:uppercase">Negative</div>
         <div style="background:${C.bg};border:1px solid ${C.border};border-radius:3px;padding:5px;font-family:ui-monospace,Menlo,monospace;font-size:10px;white-space:pre-wrap;max-height:80px;overflow:auto">${escapeHtml(d.negative || "")}</div>
         <div style="margin-top:6px;color:${C.sub};font-size:10px">Sampler defaults: ${sampLine}</div>
         <div class="msg" style="margin-top:6px;color:${C.sub};font-size:10px"></div>`;
    cont.querySelector(".apply").onclick = () => doApply(wrap, d);
    const delBtn = cont.querySelector(".del");
    if (delBtn) delBtn.onclick = async () => {
        if (!confirm(`Delete preset "${d.name}"?`)) return;
        const r = await apiDelete(d.id);
        if (r.success) {
            _state.list = await apiList(); _state.selectedId = null; _state.detail = null;
            renderList(wrap); renderDetail(wrap);
        } else {
            cont.querySelector(".msg").textContent = "Delete failed: " + (r.error || "");
        }
    };
}

function renderSaveBox(wrap) {
    const cont = wrap.querySelector("#sp-save");
    cont.innerHTML =
        `<label style="display:block;font-size:10px;color:${C.sub}">Name</label>
         <input type="text" id="sp-new-name" placeholder="My favourite look" style="${inp()}"/>
         <div style="display:flex;gap:6px;margin-top:4px">
           <select id="sp-new-cat" style="${sel()}">
             <option value="custom">custom</option><option>photoreal</option><option>anime</option><option>stylized</option><option>lighting</option>
           </select>
           <select id="sp-new-model" style="${sel()}">
             <option value="any">any</option><option>sdxl</option><option>sd1.5</option><option>flux</option><option>sd3</option><option>wan</option>
           </select>
           <button id="sp-save-btn" style="${btnPrimary()}">Save from current</button>
         </div>
         <div id="sp-save-msg" style="margin-top:6px;color:${C.sub};font-size:10px"></div>`;
    cont.querySelector("#sp-save-btn").onclick = () => doSave(wrap);
}

async function doApply(wrap, preset) {
    const { positive, negative, sampler: samplerNode } = findTargetEncoders();
    const applied = [];
    if (positive && setEncoder(positive, preset.positive || "")) applied.push("positive#" + positive.id);
    if (negative && setEncoder(negative, preset.negative || "")) applied.push("negative#" + negative.id);
    const samp = applySamplerSettings(samplerNode, preset.sampler || {});
    if (samp.length) applied.push(`sampler#${samplerNode.id}(${samp.join(",")})`);
    const msg = wrap.querySelector("#sp-detail .msg");
    if (applied.length === 0) {
        msg.style.color = C.red;
        msg.textContent = "Nothing applied — no CLIPTextEncode/KSampler found. Select a positive + negative encoder.";
    } else {
        msg.style.color = C.green;
        msg.textContent = "Applied to " + applied.join(", ");
    }
}

async function doSave(wrap) {
    const name = wrap.querySelector("#sp-new-name").value.trim();
    const cat  = wrap.querySelector("#sp-new-cat").value;
    const mh   = wrap.querySelector("#sp-new-model").value;
    const msg  = wrap.querySelector("#sp-save-msg");
    if (!name) { msg.style.color = C.red; msg.textContent = "Name is required."; return; }

    const { positive, negative, sampler: samplerNode } = findTargetEncoders();
    if (!positive && !negative) {
        msg.style.color = C.red;
        msg.textContent = "No CLIPTextEncode found. Select an encoder first.";
        return;
    }
    const posTxt = positive ? (positive.widgets?.find(w => w.name === "text")?.value || "") : "";
    const negTxt = negative ? (negative.widgets?.find(w => w.name === "text")?.value || "") : "";
    const samplerWidgets = {};
    if (samplerNode) {
        for (const w of (samplerNode.widgets || [])) {
            if (["steps", "cfg", "sampler_name", "scheduler", "denoise"].includes(w.name)) {
                samplerWidgets[w.name] = w.value;
            }
        }
    }
    const r = await apiSave({
        name, category: cat, model_hint: mh,
        positive: posTxt, negative: negTxt,
        sampler: samplerWidgets, tags: [],
    });
    if (!r.success) { msg.style.color = C.red; msg.textContent = "Save failed: " + (r.error || ""); return; }
    msg.style.color = C.green;
    msg.textContent = `Saved as ${r.data.name}.`;
    _state.list = await apiList(); renderList(wrap);
}

// --------------------------------------------------------- util
function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, c =>
        ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function sel()        { return `flex:1;background:${C.bg2};color:${C.fg};border:1px solid ${C.border};border-radius:4px;padding:3px;font-size:11px`; }
function inp()        { return `width:100%;background:${C.bg2};color:${C.fg};border:1px solid ${C.border};border-radius:4px;padding:4px;font-size:11px`; }
function btnPrimary() { return `background:${C.mauve};color:${C.bg};border:none;border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px`; }
function btnGhost()   { return `background:${C.bg2};color:${C.fg};border:1px solid ${C.border};border-radius:4px;padding:3px 10px;cursor:pointer;font-size:11px`; }
function btnDanger()  { return `background:${C.bg2};color:${C.red};border:1px solid ${C.red};border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px`; }

// --------------------------------------------------------- register
app.registerExtension({
    name: "c2c.style.presets",
    async setup() {
        try {
            app.extensionManager?.registerSidebarTab?.({
                id: TAB_ID,
                icon: "pi pi-palette",
                title: "C2C Styles",
                tooltip: "C2C Style Presets — reusable prompt + sampler bundles",
                type: "custom",
                render: (el) => buildView(el),
            });
        } catch (exc) {
            console.warn("[c2c.style.presets] sidebar tab registration failed:", exc);
        }
    },
});

window.__C2C_STYLE_PRESETS__ = { apiList, apiGet, apiSave, apiDelete };
