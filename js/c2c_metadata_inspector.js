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
import { readPngTextChunks, assessWorkflowMetadata } from "./c2c_png_metadata.js";

const SETTING_ENABLED = "c2c.metaInspect.enabled";
let _enabled = true;

function tryParseWorkflow(meta) {
    const a = assessWorkflowMetadata(meta);
    if (a.status !== "valid") return null;
    if (a.kind === "a1111") return { kind: "a1111", data: a.data };
    return { kind: a.kind, data: a.data };
}

/** Legacy reader — delegates to shared safe parser. */
function readPngTextChunksLegacy(buffer) {
    return readPngTextChunks(buffer);
}

function openModal(meta, parsed, file) {
    return new Promise((resolve) => {
        const root = document.createElement("div");
        root.style.cssText = `
            position: fixed; inset: 0; z-index: var(--c2c-z-modal);
            display: flex; align-items: center; justify-content: center;
            background: color-mix(in srgb, var(--c2c-shadowBase) 55%, transparent); backdrop-filter: blur(2px);
        `;
        const dlg = document.createElement("div");
        dlg.style.cssText = `
            width: min(820px, 92vw); max-height: 84vh; overflow: hidden;
            background: var(--c2c-bg2); color: var(--c2c-accentText);
            border: 1px solid color-mix(in srgb, var(--c2c-highlightBase) 14%, transparent); border-radius: 10px;
            box-shadow: 0 18px 48px color-mix(in srgb, var(--c2c-shadowBase) 70%, transparent);
            font: 12px ui-sans-serif, system-ui, sans-serif;
            display: flex; flex-direction: column;
        `;
        const head = document.createElement("div");
        head.style.cssText = "padding:10px 14px; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid color-mix(in srgb, var(--c2c-highlightBase) 8%, transparent);";
        head.innerHTML = `<div><span style="font-weight:600;color:var(--c2c-accentLight);">Image Metadata</span> <span style="color:var(--c2c-sub);">— ${escapeHtml(file.name)} · ${(file.size/1024).toFixed(1)} KB</span></div>`;
        const closeBtn = document.createElement("span");
        closeBtn.textContent = "✕"; closeBtn.style.cssText = "cursor:pointer;color:var(--c2c-sub);";
        closeBtn.addEventListener("click", () => done(false));
        head.appendChild(closeBtn);
        dlg.appendChild(head);

        const body = document.createElement("div");
        body.style.cssText = "padding:12px 16px; overflow:auto; flex:1;";

        const summary = document.createElement("div");
        summary.style.cssText = "margin-bottom:10px;";
        if (parsed) {
            const tag = parsed.kind === "a1111" ? "A1111" : (parsed.kind === "comfy-workflow" ? "ComfyUI workflow" : "ComfyUI prompt");
            summary.innerHTML = `<span style="background:color-mix(in srgb, var(--c2c-blue) 40%, transparent);color:var(--c2c-accentLight);padding:2px 8px;border-radius:10px;font-weight:600;">${tag}</span>`;
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
            summary.innerHTML = `<span style="background:color-mix(in srgb, var(--c2c-peach) 30%, transparent);color:var(--c2c-peach);padding:2px 8px;border-radius:10px;">No workflow chunk found</span>`;
        }
        body.appendChild(summary);

        const tabs = document.createElement("div");
        tabs.style.cssText = "display:flex; gap:8px; margin-bottom:8px; border-bottom:1px solid color-mix(in srgb, var(--c2c-highlightBase) 6%, transparent);";
        const content = document.createElement("pre");
        content.style.cssText = "background:var(--c2c-bg); padding:10px; border-radius:6px; max-height:46vh; overflow:auto; color:var(--c2c-accentLight2); font:11px ui-monospace, monospace; white-space:pre-wrap; word-break:break-word;";

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
            b.style.cssText = "padding:5px 10px; background:transparent; color:var(--c2c-accentLight2); border:none; cursor:pointer; font-weight:500;";
            b.addEventListener("mouseenter", () => b.style.background = "color-mix(in srgb, var(--c2c-highlightBase) 5%, transparent)");
            b.addEventListener("mouseleave", () => b.style.background = b.classList.contains("active") ? "color-mix(in srgb, var(--c2c-blue) 18%, transparent)" : "transparent");
            b.addEventListener("click", () => showTab(t.k));
            tabs.appendChild(b);
        }
        const style = document.createElement("style");
        style.textContent = `.active { background: color-mix(in srgb, var(--c2c-blue) 18%, transparent) !important; color: var(--c2c-accentLight) !important; }`;
        dlg.appendChild(style);

        body.appendChild(tabs);
        body.appendChild(content);
        dlg.appendChild(body);

        const foot = document.createElement("div");
        foot.style.cssText = "padding:10px 14px; border-top:1px solid color-mix(in srgb, var(--c2c-highlightBase) 8%, transparent); display:flex; gap:8px; justify-content:flex-end;";
        const cancel = mkBtn("Cancel", "var(--c2c-sub)", () => done(false));
        const copy = mkBtn("Copy JSON", "var(--c2c-blue)", async () => {
            try {
                await navigator.clipboard.writeText(content.textContent);
                copy.textContent = "Copied!";
                setTimeout(() => copy.textContent = "Copy JSON", 1200);
            } catch {}
        });
        const load = mkBtn(parsed && parsed.kind === "comfy-workflow" ? "Load workflow" : "Load image", "var(--c2c-green)", () => done(true));
        const browse = mkBtn("🔍 Image → Prompt", "var(--c2c-mauve)", () => window.__C2C_PRESET_HUB__?.open({ tab: "image2prompt" }));
        foot.appendChild(cancel); foot.appendChild(copy); foot.appendChild(browse); foot.appendChild(load);
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
    // White-on-accent text is an intentional universal-contrast pattern for action buttons.
    // All --c2c-* accent tokens are saturated mid-tones across mocha/latte/oled.
    // --c2c-onAccent is the universal-contrast token (white by default) injected by _c2c_theme.
    b.style.cssText = `padding:6px 14px; background:${color}; color:var(--c2c-onAccent); border:none; border-radius:5px; cursor:pointer; font-weight:600; font-size:12px;`;
    b.addEventListener("click", onClick);
    return b;
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]);
}

async function handlePng(file, evt) {
    const buffer = await file.arrayBuffer();
    const meta = readPngTextChunksLegacy(buffer);
    if (Object.keys(meta).length === 0) return false;
    const parsed = tryParseWorkflow(meta);
    const shouldLoad = await openModal(meta, parsed, file);
    if (shouldLoad && parsed && parsed.kind === "comfy-workflow" && app.loadGraphData) {
        try { app.loadGraphData(parsed.data); } catch (e) { console.warn("[c2c_metaInspect] loadGraphData failed", e); }
    }
    return true;
}

/** Called from c2c_safe_image_drop when workflow metadata parses cleanly. */
async function metaInspectPrompt(meta, assessment, file) {
    if (!_enabled) return undefined;
    const parsed = tryParseWorkflow(meta);
    if (!parsed) return undefined;
    const shouldLoad = await openModal(meta, parsed, file);
    if (!shouldLoad) return false;
    if (parsed.kind === "comfy-workflow") return "workflow";
    return true;
}

function installDropInterceptor() {
    // Drop handling moved to c2c_safe_image_drop.js (app.handleFile wrap).
    // Legacy direct-drop path kept for environments without SafeImageDrop.
    if (app.__c2cSafeDropInstalled) return;
    const handler = async (e) => {
        if (!_enabled) return;
        if (e.shiftKey) return;
        const dt = e.dataTransfer;
        if (!dt || !dt.files || dt.files.length === 0) return;
        const file = dt.files[0];
        if (!file || file.type !== "image/png") return;
        e.stopImmediatePropagation();
        e.preventDefault();
        await handlePng(file, e);
    };
    window.addEventListener("drop", handler, { capture: true });
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
        app.__c2cMetaInspectPrompt = metaInspectPrompt;
        installDropInterceptor();
    },
});
