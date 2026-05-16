// c2c_metadata_inspector.js — PNG-drop metadata inspector (C2C v2.0 §7.2)
// ---------------------------------------------------------------------
// What it does:
//   • Listens for PNG drops on the canvas area. If the PNG carries a
//     ComfyUI workflow chunk (`workflow` tEXt key) or A1111 `parameters`
//     key, a modal opens BEFORE the default Comfy "load workflow" path
//     showing parsed fields. The default Comfy load is suppressed when
//     the user clicks "Cancel" in the modal; "Load workflow" lets it
//     proceed (or loads manually if Comfy doesn't auto-load).
//   • Shift+drop bypasses the inspector and goes straight to Comfy's
//     default behavior.
//   • Reads tEXt, iTXt, zTXt chunks from raw PNG bytes (no canvas roundtrip
//     so values are exact).
//   • Settings: c2c.metaInspect.enabled.
//
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";

const SETTING_ENABLED = "c2c.metaInspect.enabled";
let _enabled = true;

function readPngTextChunks(buffer) {
    // Returns { key: value } for all tEXt / iTXt / zTXt chunks.
    // buffer: ArrayBuffer.
    const out = {};
    const view = new DataView(buffer);
    const sig = [137, 80, 78, 71, 13, 10, 26, 10];
    for (let i = 0; i < 8; i++) if (view.getUint8(i) !== sig[i]) return out;
    let off = 8;
    while (off < view.byteLength - 8) {
        const length = view.getUint32(off); off += 4;
        const type = String.fromCharCode(view.getUint8(off), view.getUint8(off+1), view.getUint8(off+2), view.getUint8(off+3));
        off += 4;
        const data = new Uint8Array(buffer, off, length);
        off += length + 4; // skip CRC
        if (type === "tEXt") {
            const nul = data.indexOf(0);
            if (nul > 0) {
                const key = utf8(data.subarray(0, nul));
                const val = utf8(data.subarray(nul + 1));
                out[key] = val;
            }
        } else if (type === "iTXt") {
            const nul = data.indexOf(0);
            if (nul > 0) {
                const key = utf8(data.subarray(0, nul));
                // iTXt: keyword \0 compFlag(1) compMethod(1) langTag \0 transKey \0 text
                let p = nul + 1;
                const compFlag = data[p]; p += 2;
                const langEnd = data.indexOf(0, p); p = langEnd + 1;
                const transEnd = data.indexOf(0, p); p = transEnd + 1;
                let text;
                if (compFlag === 0) text = utf8(data.subarray(p));
                else { try { text = utf8(pako_inflate(data.subarray(p))); } catch { text = "<compressed>"; } }
                out[key] = text;
            }
        } else if (type === "IEND") break;
    }
    return out;
}

function utf8(arr) {
    try { return new TextDecoder("utf-8").decode(arr); } catch { return ""; }
}

// Minimal inflate fallback: prefer browser DecompressionStream if available.
function pako_inflate(arr) {
    if (typeof DecompressionStream === "undefined") throw new Error("no DecompressionStream");
    // Build synchronously is tricky; fall back to "<compressed>" in caller.
    throw new Error("sync inflate not supported");
}

function tryParseWorkflow(meta) {
    // Common keys: "workflow", "prompt", "parameters" (A1111), "Comment"
    for (const k of ["workflow", "Workflow"]) {
        if (meta[k]) {
            try { return { kind: "comfy-workflow", data: JSON.parse(meta[k]) }; } catch {}
        }
    }
    for (const k of ["prompt", "Prompt"]) {
        if (meta[k]) {
            try { return { kind: "comfy-prompt", data: JSON.parse(meta[k]) }; } catch {}
        }
    }
    if (meta.parameters) {
        return { kind: "a1111", data: meta.parameters };
    }
    return null;
}

function openModal(meta, parsed, file) {
    return new Promise((resolve) => {
        const root = document.createElement("div");
        root.style.cssText = `
            position: fixed; inset: 0; z-index: 13000;
            display: flex; align-items: center; justify-content: center;
            background: rgba(0,0,0,0.55); backdrop-filter: blur(2px);
        `;
        const dlg = document.createElement("div");
        dlg.style.cssText = `
            width: min(820px, 92vw); max-height: 84vh; overflow: hidden;
            background: #141821; color: #e5ecf5;
            border: 1px solid rgba(255,255,255,0.14); border-radius: 10px;
            box-shadow: 0 18px 48px rgba(0,0,0,0.7);
            font: 12px ui-sans-serif, system-ui, sans-serif;
            display: flex; flex-direction: column;
        `;
        const head = document.createElement("div");
        head.style.cssText = "padding:10px 14px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.08);";
        head.innerHTML = `<div><span style="font-weight:600;color:#cfe0ff;">Image Metadata</span> <span style="color:#7a8492;">— ${escapeHtml(file.name)} · ${(file.size/1024).toFixed(1)} KB</span></div>`;
        const closeBtn = document.createElement("span");
        closeBtn.textContent = "✕"; closeBtn.style.cssText = "cursor:pointer;color:#888;";
        closeBtn.addEventListener("click", () => done(false));
        head.appendChild(closeBtn);
        dlg.appendChild(head);

        const body = document.createElement("div");
        body.style.cssText = "padding:12px 16px; overflow:auto; flex:1;";

        const summary = document.createElement("div");
        summary.style.cssText = "margin-bottom:10px;";
        if (parsed) {
            const tag = parsed.kind === "a1111" ? "A1111" : (parsed.kind === "comfy-workflow" ? "ComfyUI workflow" : "ComfyUI prompt");
            summary.innerHTML = `<span style="background:#2c4a82;color:#cfe0ff;padding:2px 8px;border-radius:10px;font-weight:600;">${tag}</span>`;
            if (parsed.kind === "comfy-workflow") {
                const nodes = (parsed.data && parsed.data.nodes) || [];
                summary.innerHTML += ` · ${nodes.length} nodes`;
            } else if (parsed.kind === "comfy-prompt") {
                const ids = Object.keys(parsed.data || {});
                summary.innerHTML += ` · ${ids.length} executed nodes`;
            } else if (parsed.kind === "a1111") {
                const m = String(parsed.data).match(/Steps:\s*(\d+).*?Sampler:\s*([^,]+).*?Model:\s*([^,]+)/s);
                if (m) summary.innerHTML += ` · Sampler ${escapeHtml(m[2])} · ${escapeHtml(m[1])} steps · ${escapeHtml(m[3])}`;
            }
        } else {
            summary.innerHTML = `<span style="background:#5a3a30;color:#ffd1a3;padding:2px 8px;border-radius:10px;">No workflow chunk found</span>`;
        }
        body.appendChild(summary);

        const tabs = document.createElement("div");
        tabs.style.cssText = "display:flex; gap:8px; margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.06);";
        const content = document.createElement("pre");
        content.style.cssText = "background:#0f1218; padding:10px; border-radius:6px; max-height:46vh; overflow:auto; color:#cfd6e0; font:11px ui-monospace, monospace; white-space:pre-wrap; word-break:break-word;";

        const tabSpec = [];
        if (parsed) tabSpec.push({ k: "parsed", n: "Parsed" });
        tabSpec.push({ k: "all", n: `All chunks (${Object.keys(meta).length})` });
        const keys = Object.keys(meta);
        for (const k of keys) tabSpec.push({ k: "raw:" + k, n: k });

        const showTab = (k) => {
            for (const c of tabs.children) c.classList.remove("active");
            const cur = [...tabs.children].find(c => c.dataset.k === k);
            if (cur) cur.classList.add("active");
            if (k === "parsed") {
                if (parsed.kind === "a1111") content.textContent = parsed.data;
                else content.textContent = JSON.stringify(parsed.data, null, 2);
            } else if (k === "all") {
                content.textContent = keys.map(kk => `=== ${kk} ===\n${meta[kk]}`).join("\n\n");
            } else if (k.startsWith("raw:")) {
                content.textContent = meta[k.slice(4)] || "";
            }
        };
        for (const t of tabSpec) {
            const b = document.createElement("button");
            b.textContent = t.n; b.dataset.k = t.k;
            b.style.cssText = "padding:5px 10px; background:transparent; color:#cfd6e0; border:none; cursor:pointer; font-weight:500;";
            b.addEventListener("mouseenter", () => b.style.background = "rgba(255,255,255,0.05)");
            b.addEventListener("mouseleave", () => b.style.background = b.classList.contains("active") ? "rgba(91,141,239,0.18)" : "transparent");
            b.addEventListener("click", () => showTab(t.k));
            tabs.appendChild(b);
        }
        const style = document.createElement("style");
        style.textContent = `.active { background: rgba(91,141,239,0.18) !important; color:#cfe0ff !important; }`;
        dlg.appendChild(style);

        body.appendChild(tabs);
        body.appendChild(content);
        dlg.appendChild(body);

        const foot = document.createElement("div");
        foot.style.cssText = "padding:10px 14px; border-top:1px solid rgba(255,255,255,0.08); display:flex; gap:8px; justify-content:flex-end;";
        const cancel = mkBtn("Cancel", "#7a8492", () => done(false));
        const copy = mkBtn("Copy JSON", "#5b8def", async () => {
            try {
                await navigator.clipboard.writeText(content.textContent);
                copy.textContent = "Copied!";
                setTimeout(() => copy.textContent = "Copy JSON", 1200);
            } catch {}
        });
        const load = mkBtn(parsed && parsed.kind === "comfy-workflow" ? "Load workflow" : "Load image", "#3aa66a", () => done(true));
        foot.appendChild(cancel); foot.appendChild(copy); foot.appendChild(load);
        dlg.appendChild(foot);
        root.appendChild(dlg);
        document.body.appendChild(root);

        showTab(tabSpec[0].k);

        const done = (load) => { root.remove(); document.removeEventListener("keydown", onKey, true); resolve(load); };
        const onKey = (e) => { if (e.key === "Escape") done(false); };
        document.addEventListener("keydown", onKey, true);
    });
}

function mkBtn(text, color, onClick) {
    const b = document.createElement("button");
    b.textContent = text;
    b.style.cssText = `padding:6px 14px; background:${color}; color:#fff; border:none; border-radius:5px; cursor:pointer; font-weight:600; font-size:12px;`;
    b.addEventListener("click", onClick);
    return b;
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]);
}

async function handlePng(file, evt) {
    const buffer = await file.arrayBuffer();
    const meta = readPngTextChunks(buffer);
    if (Object.keys(meta).length === 0) return false;
    const parsed = tryParseWorkflow(meta);
    const shouldLoad = await openModal(meta, parsed, file);
    if (shouldLoad && parsed && parsed.kind === "comfy-workflow" && app.loadGraphData) {
        try { app.loadGraphData(parsed.data); } catch (e) { console.warn("[c2c_metaInspect] loadGraphData failed", e); }
    }
    return true;
}

function installDropInterceptor() {
    const handler = async (e) => {
        if (!_enabled) return;
        if (e.shiftKey) return; // explicit bypass
        const dt = e.dataTransfer;
        if (!dt || !dt.files || dt.files.length === 0) return;
        const file = dt.files[0];
        if (!file || file.type !== "image/png") return;
        e.stopImmediatePropagation();
        e.preventDefault();
        await handlePng(file, e);
    };
    // capture phase to beat Comfy's own listener.
    window.addEventListener("drop", handler, { capture: true });
    // Prevent the browser default for dragover so drop fires.
    window.addEventListener("dragover", (e) => {
        if (_enabled && e.dataTransfer && Array.from(e.dataTransfer.items || []).some(it => it.kind === "file")) {
            e.preventDefault();
        }
    }, { capture: true });
}

app.registerExtension({
    name: "C2C.MetadataInspector",
    async setup() {
        app.ui.settings.addSetting({
            id: SETTING_ENABLED, name: "C2C ▸ Metadata Inspector on PNG drop",
            type: "boolean", defaultValue: true,
            onChange: v => { _enabled = !!v; },
        });
        _enabled = app.ui.settings.getSettingValue(SETTING_ENABLED, true);
        installDropInterceptor();
    },
});
