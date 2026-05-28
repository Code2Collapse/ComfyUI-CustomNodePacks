/**
 * c2c_whats_wired.js — Phase 8: "What's Wired" Workflow Legend
 *
 * Floating, draggable, resizable, minimisable window that summarises the
 * active pipeline at a glance. Pulls data live from the running graph and
 * every nested subgraph; refreshes on graph mutation or every 2 s.
 *
 * Fields (all 8 user-requested):
 *   - Pipeline       (auto-detected: txt2img / img2img / inpaint / upscale / video / audio)
 *   - Model + VAE    (checkpoint / unet + VAE filenames)
 *   - Sampler        (sampler_name + scheduler + steps + cfg + denoise)
 *   - Size           (width x height + batch)
 *   - LoRAs          (every active LoRA with strength_model + strength_clip)
 *   - Conditioning   (ControlNet / IP-Adapter / regional / style detected)
 *   - Prompts        (positive + negative char count + 40-char preview)
 *   - Outputs        (every Save / Preview / Output node that will fire)
 *   - Complexity     (node count + link count + VRAM tier guess)
 *
 * Window behaviour mirrors the Inspector panel (c2c_node_explain.js):
 *   drag header, 8 resize edges/corners, minimise button, close button,
 *   geom persisted in localStorage.
 */

import { app } from "../../scripts/app.js";
import { forAllNodes } from "./_subgraph_walk.js";

const HUD_ID    = "mec-whats-wired";
const STYLE_ID  = "mec-whats-wired-style";
const GEOM_KEY  = "c2c.whats_wired.geom.v2";

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${HUD_ID} {
    position: fixed;
    top: var(--c2c-ww-top, 96px);
    right: var(--c2c-ww-right, 16px);
    width: var(--c2c-ww-w, 320px);
    height: var(--c2c-ww-h, 360px);
    z-index: var(--c2c-z-hud, 1000);
    background: var(--c2c-bg);
    border: 1px solid var(--c2c-surface1);
    border-radius: 8px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.55);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 11px;
    color: var(--c2c-fg);
    display: none;
    flex-direction: column;
    overflow: hidden;
    user-select: none;
}
#${HUD_ID}.visible { display: flex; }
#${HUD_ID}[data-collapsed="1"] { height: auto !important; }
#${HUD_ID}[data-collapsed="1"] .ww-body { display: none; }
#${HUD_ID}[data-collapsed="1"] .ww-edge.s,
#${HUD_ID}[data-collapsed="1"] .ww-edge.sw,
#${HUD_ID}[data-collapsed="1"] .ww-edge.se,
#${HUD_ID}[data-collapsed="1"] .ww-edge.e,
#${HUD_ID}[data-collapsed="1"] .ww-edge.w { display: none; }

#${HUD_ID} .ww-head {
    display: flex; align-items: center; gap: 8px;
    height: 26px; padding: 0 8px;
    background: var(--c2c-bg2);
    border-bottom: 1px solid var(--c2c-surface0);
    cursor: move;
    flex-shrink: 0;
}
#${HUD_ID} .ww-grip { color: var(--c2c-overlay0); font-size: 10px; cursor: grab; line-height: 1; }
#${HUD_ID} .ww-title {
    font-weight: 700; color: var(--c2c-blue); font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.5px; flex: 1;
}
#${HUD_ID} .ww-pipeline-pill {
    background: var(--c2c-surface0); color: var(--c2c-sky); font-size: 9px;
    padding: 2px 6px; border-radius: 8px; font-weight: 600;
    text-transform: none; letter-spacing: 0;
}
#${HUD_ID} .ww-btn {
    background: transparent; border: none; color: var(--c2c-fg);
    font-size: 13px; line-height: 1; padding: 2px 6px; cursor: pointer;
    border-radius: 3px;
}
#${HUD_ID} .ww-btn:hover { background: var(--c2c-surface0); }
#${HUD_ID} .ww-body { flex: 1; overflow-y: auto; padding: 8px 10px 10px; user-select: text; }
#${HUD_ID} .ww-section { margin-bottom: 10px; }
#${HUD_ID} .ww-section:last-child { margin-bottom: 0; }
#${HUD_ID} .ww-section-title {
    color: var(--c2c-yellow); font-size: 9px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 3px;
}
#${HUD_ID} .ww-row { display: flex; gap: 6px; align-items: baseline; padding: 1px 0; line-height: 1.45; }
#${HUD_ID} .ww-key { color: var(--c2c-overlay0); min-width: 64px; font-size: 10px; flex-shrink: 0; }
#${HUD_ID} .ww-val {
    color: var(--c2c-fg); font-family: ui-monospace, Consolas, monospace;
    font-size: 10.5px; flex: 1; word-break: break-word;
}
#${HUD_ID} .ww-val.lora  { color: var(--c2c-okSoft); }
#${HUD_ID} .ww-val.warn  { color: var(--c2c-peach); }
#${HUD_ID} .ww-val.dim   { color: var(--c2c-overlay0); font-style: italic; }
#${HUD_ID} .ww-val.muted { color: var(--c2c-overlay2); }
#${HUD_ID} .ww-preview {
    color: var(--c2c-fg); font-family: ui-monospace, Consolas, monospace;
    font-size: 10px; background: var(--c2c-bg3); padding: 4px 6px;
    border-radius: 4px; margin-top: 2px;
    white-space: pre-wrap; word-break: break-word;
    border-left: 2px solid var(--c2c-okSoft);
}
#${HUD_ID} .ww-preview.neg { border-left-color: var(--c2c-red); }
#${HUD_ID} .ww-empty {
    color: var(--c2c-overlay0); font-style: italic; font-size: 10px;
    padding: 16px 8px; text-align: center;
}
#${HUD_ID} .ww-vram-pill {
    display: inline-block; padding: 1px 6px; border-radius: 8px;
    font-size: 9px; font-weight: 600;
}
#${HUD_ID} .ww-vram-low  { background: var(--c2c-bg); color: var(--c2c-okSoft); }
#${HUD_ID} .ww-vram-med  { background: var(--c2c-panelBg); color: var(--c2c-yellow); }
#${HUD_ID} .ww-vram-high { background: var(--c2c-bg); color: var(--c2c-peach); }
#${HUD_ID} .ww-vram-huge { background: var(--c2c-bg); color: var(--c2c-red); }

#${HUD_ID} .ww-edge { position: absolute; z-index: 5; }
#${HUD_ID} .ww-edge.n  { top:0;    left:8px;   right:8px;  height:5px; cursor: ns-resize; }
#${HUD_ID} .ww-edge.s  { bottom:0; left:8px;   right:8px;  height:5px; cursor: ns-resize; }
#${HUD_ID} .ww-edge.e  { right:0;  top:8px;    bottom:8px; width:5px;  cursor: ew-resize; }
#${HUD_ID} .ww-edge.w  { left:0;   top:8px;    bottom:8px; width:5px;  cursor: ew-resize; }
#${HUD_ID} .ww-edge.nw { top:0;    left:0;     width:10px; height:10px; cursor: nwse-resize; }
#${HUD_ID} .ww-edge.ne { top:0;    right:0;    width:10px; height:10px; cursor: nesw-resize; }
#${HUD_ID} .ww-edge.sw { bottom:0; left:0;     width:10px; height:10px; cursor: nesw-resize; }
#${HUD_ID} .ww-edge.se { bottom:0; right:0;    width:10px; height:10px; cursor: nwse-resize; }
#${HUD_ID} .ww-edge:hover { background: rgba(137,180,250,0.18); }
    `.trim();
    document.head.appendChild(style);
}

function _loadGeom() {
    try { return JSON.parse(localStorage.getItem(GEOM_KEY)) || {}; }
    catch (_) { return {}; }
}
function _saveGeom(g) {
    try { localStorage.setItem(GEOM_KEY, JSON.stringify(g)); } catch (_) {}
}
function _applyGeom(el) {
    const g = _loadGeom();
    const s = el.style;
    if (typeof g.top   === "number") s.setProperty("--c2c-ww-top",   g.top   + "px");
    if (typeof g.right === "number") s.setProperty("--c2c-ww-right", g.right + "px");
    if (typeof g.w === "number") s.setProperty("--c2c-ww-w", Math.max(240, g.w) + "px");
    if (typeof g.h === "number") s.setProperty("--c2c-ww-h", Math.max(120, g.h) + "px");
    el.dataset.collapsed = g.collapsed ? "1" : "0";
}

function _buildHud() {
    let hud = document.getElementById(HUD_ID);
    if (hud) return hud;
    hud = document.createElement("div");
    hud.id = HUD_ID;
    hud.innerHTML = `
        <header class="ww-head">
            <span class="ww-grip" title="Drag to move">\u22ee\u22ee</span>
            <span class="ww-title">What's Wired <span class="ww-shortcut">(Ctrl+Shift+W)</span></span>
            <span class="ww-pipeline-pill" style="display:none"></span>
            <button class="ww-btn ww-btn-min"   title="Minimise / restore">\u2013</button>
            <button class="ww-btn ww-btn-close" title="Hide (Ctrl+Shift+W to bring back)">\u00d7</button>
        </header>
        <div class="ww-body"><div class="ww-empty">No pipeline detected yet.</div></div>
        <div class="ww-edge n"  data-dir="N"></div>
        <div class="ww-edge s"  data-dir="S"></div>
        <div class="ww-edge e"  data-dir="E"></div>
        <div class="ww-edge w"  data-dir="W"></div>
        <div class="ww-edge nw" data-dir="NW"></div>
        <div class="ww-edge ne" data-dir="NE"></div>
        <div class="ww-edge sw" data-dir="SW"></div>
        <div class="ww-edge se" data-dir="SE"></div>
    `;
    document.body.appendChild(hud);
    _applyGeom(hud);

    const headEl = hud.querySelector(".ww-head");
    headEl.addEventListener("mousedown", (e) => {
        if (e.target.closest(".ww-btn") || e.target.closest(".ww-edge")) return;
        e.preventDefault();
        const startX = e.clientX, startY = e.clientY;
        const cs = getComputedStyle(hud);
        const startTop   = parseFloat(cs.top)   || 96;
        const startRight = parseFloat(cs.right) || 16;
        const onMove = (ev) => {
            const dx = ev.clientX - startX, dy = ev.clientY - startY;
            const w = hud.offsetWidth, h = hud.offsetHeight;
            let t = startTop + dy, r = startRight - dx;
            // TOOLBAR_GUTTER=60: never hide behind ComfyUI native top chrome.
            t = Math.max(60, Math.min(window.innerHeight - h - 4, t));
            r = Math.max(0, Math.min(window.innerWidth  - w,  r));
            hud.style.setProperty("--c2c-ww-top",   t + "px");
            hud.style.setProperty("--c2c-ww-right", r + "px");
        };
        const onUp = () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup",   onUp);
            const cs2 = getComputedStyle(hud);
            const g = _loadGeom();
            g.top = parseFloat(cs2.top); g.right = parseFloat(cs2.right);
            _saveGeom(g);
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup",   onUp);
    });

    const attachResize = (edgeEl, dir) => {
        const hasN = dir.includes("N"), hasS = dir.includes("S");
        const hasE = dir.includes("E"), hasW = dir.includes("W");
        edgeEl.addEventListener("mousedown", (e) => {
            e.preventDefault(); e.stopPropagation();
            const startX = e.clientX, startY = e.clientY;
            const cs = getComputedStyle(hud);
            const startW = hud.offsetWidth, startH = hud.offsetHeight;
            const startT = parseFloat(cs.top)   || 96;
            const startR = parseFloat(cs.right) || 16;
            const onMove = (ev) => {
                const dx = ev.clientX - startX, dy = ev.clientY - startY;
                let w = startW, h = startH, t = startT, r = startR;
                if (hasE) { w = startW + dx; r = startR - dx; }
                if (hasW) { w = startW - dx; }
                if (hasS) { h = startH + dy; }
                if (hasN) { h = startH - dy; t = startT + dy; }
                w = Math.max(240, Math.min(900, w));
                h = Math.max(120, Math.min(window.innerHeight - 64, h));
                r = Math.max(0,   Math.min(window.innerWidth  - w, r));
                t = Math.max(60,  Math.min(window.innerHeight - h - 4, t));
                hud.style.setProperty("--c2c-ww-w", w + "px");
                hud.style.setProperty("--c2c-ww-h", h + "px");
                hud.style.setProperty("--c2c-ww-top",   t + "px");
                hud.style.setProperty("--c2c-ww-right", r + "px");
            };
            const onUp = () => {
                window.removeEventListener("mousemove", onMove);
                window.removeEventListener("mouseup",   onUp);
                const cs2 = getComputedStyle(hud);
                const g = _loadGeom();
                g.w = hud.offsetWidth; g.h = hud.offsetHeight;
                g.top = parseFloat(cs2.top); g.right = parseFloat(cs2.right);
                _saveGeom(g);
            };
            window.addEventListener("mousemove", onMove);
            window.addEventListener("mouseup",   onUp);
        });
    };
    hud.querySelectorAll(".ww-edge").forEach(eg => attachResize(eg, eg.dataset.dir));

    hud.querySelector(".ww-btn-min").addEventListener("click", (e) => {
        e.stopPropagation();
        const collapsed = hud.dataset.collapsed === "1";
        hud.dataset.collapsed = collapsed ? "0" : "1";
        const g = _loadGeom();
        g.collapsed = !collapsed;
        _saveGeom(g);
        // EXPAND clamp: after restoring body, ensure the now-tall panel
        // doesn't run off the top OR the bottom of the viewport. Runs after
        // the next layout tick so offsetHeight reflects the expanded size.
        if (collapsed) {
            requestAnimationFrame(() => {
                const h = hud.offsetHeight, w = hud.offsetWidth;
                const cs = getComputedStyle(hud);
                let t = parseFloat(cs.top)   || 96;
                let r = parseFloat(cs.right) || 16;
                t = Math.max(60, Math.min(window.innerHeight - h - 4, t));
                r = Math.max(0,  Math.min(window.innerWidth  - w,     r));
                hud.style.setProperty("--c2c-ww-top",   t + "px");
                hud.style.setProperty("--c2c-ww-right", r + "px");
                const gg = _loadGeom(); gg.top = t; gg.right = r; _saveGeom(gg);
            });
        }
    });
    hud.querySelector(".ww-btn-close").addEventListener("click", (e) => {
        e.stopPropagation();
        hud.classList.remove("visible");
        const g = _loadGeom();
        g.hidden = true;
        _saveGeom(g);
    });

    return hud;
}

function _findWidget(node, names) {
    for (const w of (node.widgets || [])) {
        if (!w || !w.name) continue;
        for (const n of names) if (w.name.toLowerCase() === n.toLowerCase()) return w;
    }
    return null;
}
function _shortName(s) {
    if (!s) return "—";
    const parts = String(s).split(/[\\\/]/);
    return parts[parts.length - 1].replace(/\.(safetensors|ckpt|pt|bin|gguf|sft)$/i, "");
}

function _scanGraph() {
    const out = {
        model: null, vae: null,
        sampler: null, scheduler: null, steps: null, cfg: null, denoise: null,
        width: null, height: null, batch: null,
        loras: [], controlnets: [], ipadapters: [],
        regionals: 0, styles: 0,
        positives: [], negatives: [], outputs: [],
        nodeCount: 0, linkCount: 0,
        videoNodes: 0, audioNodes: 0, upscaleNodes: 0, inpaintNodes: 0, loadImageNodes: 0,
    };
    const g = app.graph;
    if (!g) return out;

    forAllNodes((node) => {
        out.nodeCount++;
        const cls = String(node.type || node.comfyClass || "");

        if (/checkpoint|unetloader|diffusionmodelloader|fluxloader|sd3loader/i.test(cls) &&
            !/lora|controlnet/i.test(cls)) {
            const w = _findWidget(node, ["ckpt_name","model_name","unet_name","model","diffusion_model"]);
            if (w && w.value) out.model = String(w.value);
        }
        if (/^vaeloader/i.test(cls)) {
            const w = _findWidget(node, ["vae_name","vae"]);
            if (w && w.value) out.vae = String(w.value);
        }
        if (/^k?sampler/i.test(cls) || /sampler(advanced|custom|select|gpu)?$/i.test(cls)) {
            const sw  = _findWidget(node, ["sampler_name","sampler"]);
            const sch = _findWidget(node, ["scheduler"]);
            const st  = _findWidget(node, ["steps"]);
            const cfg = _findWidget(node, ["cfg","cfg_scale","guidance"]);
            const dn  = _findWidget(node, ["denoise","denoising_strength"]);
            if (sw  && sw.value)  out.sampler   = String(sw.value);
            if (sch && sch.value) out.scheduler = String(sch.value);
            if (st  && typeof st.value === "number")  out.steps   = st.value;
            if (cfg && typeof cfg.value === "number") out.cfg     = cfg.value;
            if (dn  && typeof dn.value === "number")  out.denoise = dn.value;
        }
        if (/empty(latentimage|sd3latentimage|hunyuanlatentvideo|latentaudio|cosmoslatentvideo)/i.test(cls)) {
            const wW = _findWidget(node, ["width"]);
            const wH = _findWidget(node, ["height"]);
            const wB = _findWidget(node, ["batch_size","length","video_frames"]);
            if (wW && typeof wW.value === "number") out.width  = wW.value;
            if (wH && typeof wH.value === "number") out.height = wH.value;
            if (wB && typeof wB.value === "number") out.batch  = wB.value;
        }
        if (/loraloader|lora_loader|loralightningloader|powerloraloader/i.test(cls)) {
            const nameW = _findWidget(node, ["lora_name"]);
            const smW   = _findWidget(node, ["strength_model","strength"]);
            const scW   = _findWidget(node, ["strength_clip"]);
            if (nameW && nameW.value && nameW.value !== "None") {
                out.loras.push({
                    name: String(nameW.value),
                    sm: smW && typeof smW.value === "number" ? smW.value : 1.0,
                    sc: scW && typeof scW.value === "number" ? scW.value : 1.0,
                });
            }
            for (const w of (node.widgets || [])) {
                if (!w || !w.name) continue;
                const m = w.name.match(/^lora_(\d+)$/i);
                if (m && w.value && w.value !== "None") {
                    const idx = m[1];
                    const sm = _findWidget(node, ["strength_" + idx, "strength_model_" + idx]);
                    out.loras.push({
                        name: String(w.value),
                        sm: sm && typeof sm.value === "number" ? sm.value : 1.0,
                        sc: 1.0,
                    });
                }
            }
        }
        if (/controlnet(loader|applyadvanced)?$/i.test(cls) || /apply.*controlnet/i.test(cls)) {
            const nameW = _findWidget(node, ["control_net_name","controlnet_name"]);
            const strW  = _findWidget(node, ["strength"]);
            if (nameW && nameW.value) {
                out.controlnets.push({
                    name: String(nameW.value),
                    strength: strW && typeof strW.value === "number" ? strW.value : 1.0,
                });
            }
        }
        if (/ipadapter|ip_adapter/i.test(cls)) {
            const w = _findWidget(node, ["preset","weight","ipadapter_file"]);
            out.ipadapters.push({ name: w && w.value ? String(w.value) : cls });
        }
        if (/regional|^conditioningsetarea|^conditioningsetmask/i.test(cls)) out.regionals++;
        if (/style(model|loader|conditioning)/i.test(cls)) out.styles++;

        if (/^cliptextencode/i.test(cls) || /textencod/i.test(cls)) {
            const w = _findWidget(node, ["text","prompt"]);
            const txt = w && w.value ? String(w.value) : "";
            if (!txt) return;
            let isNegative = false;
            for (const out0 of (node.outputs || [])) {
                if (!out0 || !out0.links) continue;
                for (const lid of out0.links) {
                    const link = app.graph.links && app.graph.links[lid];
                    if (!link) continue;
                    const dst = app.graph.getNodeById(link.target_id);
                    if (!dst) continue;
                    const inp = dst.inputs && dst.inputs[link.target_slot];
                    if (inp && /negative/i.test(inp.name)) { isNegative = true; break; }
                }
                if (isNegative) break;
            }
            (isNegative ? out.negatives : out.positives).push(txt);
        }

        if (/^(saveimage|previewimage|saveanimated(png|webp)|savevideo|savewebm|saveaudio|previewaudio|previewvideo)/i.test(cls)) {
            const w = _findWidget(node, ["filename_prefix","prefix"]);
            out.outputs.push({ cls, prefix: w && w.value ? String(w.value) : "" });
        }

        if (/video|wan|hunyuan|cosmos|animatediff|svd/i.test(cls)) out.videoNodes++;
        if (/audio|voicecraft|chatterbox|f5tts|rvc/i.test(cls)) out.audioNodes++;
        if (/upscale|esrgan|swinir|realesrgan/i.test(cls) && !/latent/i.test(cls)) out.upscaleNodes++;
        if (/inpaint/i.test(cls)) out.inpaintNodes++;
        if (/^loadimage(mask)?$/i.test(cls)) out.loadImageNodes++;
    });

    const links = app.graph.links;
    if (links) {
        out.linkCount = Array.isArray(links) ? links.filter(Boolean).length : Object.keys(links).length;
    }
    return out;
}

function _detectPipeline(s) {
    if (s.videoNodes)                                    return "video";
    if (s.audioNodes)                                    return "audio";
    if (s.inpaintNodes)                                  return "inpaint";
    if (s.upscaleNodes && !s.sampler)                    return "upscale";
    if (s.upscaleNodes && s.sampler && s.loadImageNodes) return "hi-res fix";
    if (s.loadImageNodes && s.sampler)                   return "img2img";
    if (s.sampler)                                       return "txt2img";
    return "—";
}

function _vramTier(s) {
    const px = (s.width || 512) * (s.height || 512);
    const isFlux = s.model && /flux/i.test(s.model);
    const isSD3  = s.model && /sd3/i.test(s.model);
    const isXL   = s.model && /xl|sdxl/i.test(s.model);
    const heavy  = isFlux || isSD3 || s.videoNodes > 0;
    if (s.videoNodes >= 2 || (heavy && px >= 1024 * 1024)) return { tier: "huge", label: "≥ 16 GB" };
    if (heavy || px >= 1280 * 1280)                        return { tier: "high", label: "10-16 GB" };
    if (isXL || px >= 768 * 768)                           return { tier: "med",  label: "6-10 GB" };
    return { tier: "low", label: "≤ 6 GB" };
}

function _esc(s) {
    return String(s).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}
function _preview(s, n) {
    const t = (s || "").replace(/\s+/g, " ").trim();
    return t.length <= n ? t : t.slice(0, n).trim() + "…";
}
function _row(k, v) {
    return `<div class="ww-row"><span class="ww-key">${_esc(k)}</span><span class="ww-val">${v}</span></div>`;
}
function _section(title, rowsArr) {
    return `<div class="ww-section"><div class="ww-section-title">${_esc(title)}</div>${rowsArr.join("")}</div>`;
}
function _sectionRaw(title, htmlBody) {
    return `<div class="ww-section"><div class="ww-section-title">${_esc(title)}</div>${htmlBody}</div>`;
}

function _render() {
    const hud = document.getElementById(HUD_ID);
    if (!hud) return;
    let enabled = true;
    try { enabled = app.ui.settings.getSettingValue("c2c.whats_wired.enabled", true); } catch (_) {}
    const geom = _loadGeom();
    if (!enabled || geom.hidden) { hud.classList.remove("visible"); return; }

    const s = _scanGraph();
    const body = hud.querySelector(".ww-body");
    const pill = hud.querySelector(".ww-pipeline-pill");

    if (!s.nodeCount) {
        body.innerHTML = `<div class="ww-empty">Workflow is empty.<br>Drop a checkpoint loader to begin.</div>`;
        pill.style.display = "none";
        hud.classList.add("visible");
        return;
    }

    const pipeline = _detectPipeline(s);
    pill.textContent = pipeline;
    pill.style.display = pipeline === "—" ? "none" : "";

    const sections = [];

    const samplerRows = [];
    if (s.sampler || s.scheduler) {
        samplerRows.push(_row("Sampler", _esc(s.sampler || "—") + (s.scheduler ? " · " + _esc(s.scheduler) : "")));
    }
    const samplingBits = [];
    if (s.steps !== null)  samplingBits.push(`${s.steps} steps`);
    if (s.cfg   !== null)  samplingBits.push(`cfg ${s.cfg}`);
    if (s.denoise !== null && s.denoise !== 1) samplingBits.push(`denoise ${s.denoise}`);
    if (samplingBits.length) samplerRows.push(_row("Sampling", samplingBits.join(" · ")));
    if (samplerRows.length) sections.push(_section("Sampler", samplerRows));

    const modelRows = [];
    if (s.model) modelRows.push(_row("Model", _esc(_shortName(s.model))));
    if (s.vae)   modelRows.push(_row("VAE",   _esc(_shortName(s.vae))));
    if (modelRows.length) sections.push(_section("Model", modelRows));

    const sizeRows = [];
    if (s.width && s.height) {
        const sizeStr = `${s.width}×${s.height}` + (s.batch && s.batch > 1 ? ` · batch ${s.batch}` : "");
        sizeRows.push(_row("Size", sizeStr));
    }
    if (sizeRows.length) sections.push(_section("Output Size", sizeRows));

    if (s.loras.length) {
        const rows = s.loras.slice(0, 8).map(l => {
            const strs = [`m:${l.sm.toFixed(2)}`];
            if (l.sc !== l.sm) strs.push(`c:${l.sc.toFixed(2)}`);
            return `<div class="ww-row"><span class="ww-val lora">${_esc(_shortName(l.name))}</span><span class="ww-val muted">${strs.join(" ")}</span></div>`;
        });
        if (s.loras.length > 8) rows.push(`<div class="ww-row"><span class="ww-val dim">…and ${s.loras.length - 8} more</span></div>`);
        sections.push(_sectionRaw(`LoRAs (${s.loras.length})`, rows.join("")));
    }

    const condRows = [];
    if (s.controlnets.length) {
        for (const c of s.controlnets.slice(0, 4)) {
            condRows.push(_row("CNet", `${_esc(_shortName(c.name))} <span class="ww-val muted">@${c.strength.toFixed(2)}</span>`));
        }
        if (s.controlnets.length > 4) condRows.push(_row("CNet+", `…and ${s.controlnets.length - 4} more`));
    }
    if (s.ipadapters.length) condRows.push(_row("IP-Adp", `${s.ipadapters.length}×`));
    if (s.regionals)         condRows.push(_row("Region", `${s.regionals} area / mask conditioning`));
    if (s.styles)            condRows.push(_row("Style",  `${s.styles}× style model / cond`));
    if (condRows.length) sections.push(_section("Conditioning", condRows));

    const promptRows = [];
    if (s.positives.length) {
        const total = s.positives.reduce((a, t) => a + t.length, 0);
        promptRows.push(`<div class="ww-row"><span class="ww-key">+ chars</span><span class="ww-val">${total} (${s.positives.length} node${s.positives.length>1?"s":""})</span></div>`);
        promptRows.push(`<div class="ww-preview">${_esc(_preview(s.positives.join(" • "), 80))}</div>`);
    }
    if (s.negatives.length) {
        const total = s.negatives.reduce((a, t) => a + t.length, 0);
        promptRows.push(`<div class="ww-row"><span class="ww-key">− chars</span><span class="ww-val">${total} (${s.negatives.length} node${s.negatives.length>1?"s":""})</span></div>`);
        promptRows.push(`<div class="ww-preview neg">${_esc(_preview(s.negatives.join(" • "), 80))}</div>`);
    }
    if (promptRows.length) sections.push(_sectionRaw("Prompts", promptRows.join("")));

    if (s.outputs.length) {
        const rows = s.outputs.slice(0, 6).map(o => {
            const sub = o.prefix ? `<span class="ww-val muted">prefix: ${_esc(o.prefix)}</span>` : "";
            return `<div class="ww-row"><span class="ww-val">${_esc(o.cls)}</span>${sub}</div>`;
        });
        if (s.outputs.length > 6) rows.push(`<div class="ww-row"><span class="ww-val dim">…+${s.outputs.length - 6}</span></div>`);
        sections.push(_sectionRaw(`Outputs (${s.outputs.length})`, rows.join("")));
    }

    const v = _vramTier(s);
    sections.push(_section("Complexity", [
        _row("Nodes", `${s.nodeCount}`),
        _row("Links", `${s.linkCount}`),
        _row("VRAM~", `<span class="ww-vram-pill ww-vram-${v.tier}">${v.label}</span>`),
    ]));

    body.innerHTML = sections.join("");
    hud.classList.add("visible");
}

let _scheduled = false;
function _schedule() {
    if (_scheduled) return;
    _scheduled = true;
    requestAnimationFrame(() => { _scheduled = false; _render(); });
}
function _hookGraph() {
    const g = app.graph;
    if (!g) return;
    const wrap = (obj, prop) => {
        if (typeof obj[prop] !== "function") return;
        const orig = obj[prop];
        if (orig._mecWWWrapped) return;
        obj[prop] = function (...args) {
            const r = orig.apply(this, args);
            try { _schedule(); } catch (_) {}
            return r;
        };
        obj[prop]._mecWWWrapped = true;
    };
    wrap(g, "add"); wrap(g, "remove"); wrap(g, "configure"); wrap(g, "clear");
}

function _open() {
    const hud = document.getElementById(HUD_ID) || _buildHud();
    const g = _loadGeom();
    if (g.hidden) { g.hidden = false; _saveGeom(g); }
    hud.classList.add("visible");
    _render();
}

app.registerExtension({
    name: "C2C.WhatsWired",
    settings: [
        {
            id: "c2c.whats_wired.enabled",
            name: "What's Wired: workflow legend",
            tooltip: "Floating window summarising the active pipeline (model, sampler, size, LoRAs, conditioning, prompts, outputs).",
            type: "boolean",
            default: true,
            onChange: _render,
        },
    ],
    commands: [
        { id: "C2C.WhatsWired.open", label: "What's Wired: open / focus", function: _open },
    ],
    async setup() {
        _injectStyle();
        _buildHud();
        _hookGraph();
        _render();
        setInterval(_render, 2000);
        window.c2c_whats_wired = { open: _open, render: _render };
        console.log("[C2C.WhatsWired] window-mode loaded.");
    },
});
