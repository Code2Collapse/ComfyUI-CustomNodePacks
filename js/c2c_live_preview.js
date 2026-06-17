/**
 * c2c_live_preview.js — resilient live sampling/denoising preview HUD.
 *
 * Pairs with nodes/_c2c_preview_guard.py (which forces the server to emit
 * previews even when launched with --preview-method none). This renders a
 * floating, persistent preview from the server's `b_preview` frames + `progress`
 * events, and — by design — KEEPS THE LAST GOOD FRAME through errors,
 * interruptions, and node failures, so the preview never blanks mid-run.
 *
 * Robust / update-proof:
 *   - Listens ONLY to the long-stable public `api` websocket events
 *     (b_preview, progress, executing, execution_error/_interrupted, status).
 *   - Every handler is wrapped; if any event shape changes in a future ComfyUI
 *     it no-ops instead of throwing. Independent of ComfyUI's own node-preview UI.
 *   - Does not touch sampling or the graph — display only.
 *
 * Methods: the actual decode (auto / latent2rgb / taesd) happens server-side per
 * the preview method; latent2rgb (Auto's fallback) needs no model and cannot fail.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NS = "C2C.LivePreview";
const S = {
    enabled: "c2c.livePreview.enabled",
    size: "c2c.livePreview.size",
    opacity: "c2c.livePreview.opacity",
    keepLast: "c2c.livePreview.keepLast",
};

let _root = null, _img = null, _bar = null, _label = null;
let _lastUrl = null;
let _pos = JSON.parse(localStorage.getItem("c2c.livePreview.pos") || "null");

function _get(id, def) {
    try { const v = app.ui?.settings?.getSettingValue?.(id, def); return v === undefined ? def : v; }
    catch { return def; }
}

function _ensureHud() {
    if (_root) return _root;
    const root = document.createElement("div");
    root.id = "c2c-live-preview";
    const size = Number(_get(S.size, 256)) || 256;
    root.style.cssText = [
        "position:fixed", "z-index:1300", "width:" + size + "px",
        "background:var(--c2c-bg2,#1a1a22)", "border:1px solid var(--c2c-border,#333)",
        "border-radius:8px", "box-shadow:0 6px 24px rgba(0,0,0,.5)",
        "overflow:hidden", "user-select:none", "font:11px ui-sans-serif",
        "opacity:" + (Number(_get(S.opacity, 0.96)) || 0.96),
    ].join(";");
    const p = _pos || { right: 16, bottom: 64 };
    if (p.left != null) { root.style.left = p.left + "px"; } else { root.style.right = (p.right ?? 16) + "px"; }
    if (p.top != null) { root.style.top = p.top + "px"; } else { root.style.bottom = (p.bottom ?? 64) + "px"; }

    const bar = document.createElement("div");
    bar.style.cssText = "display:flex;align-items:center;gap:6px;padding:3px 6px;cursor:move;background:var(--c2c-bg,#12121a);color:var(--c2c-fg,#ddd)";
    bar.innerHTML = `<span style="font-weight:600">⚡ Live Preview</span><span data-role="lbl" style="margin-left:auto;opacity:.7"></span>
        <span data-role="close" title="Hide" style="cursor:pointer;padding:0 4px;opacity:.6">✕</span>`;

    const imgWrap = document.createElement("div");
    imgWrap.style.cssText = "position:relative;background:#000;line-height:0";
    const img = document.createElement("img");
    img.style.cssText = "display:block;width:100%;height:auto;image-rendering:auto";
    imgWrap.appendChild(img);

    const prog = document.createElement("div");
    prog.style.cssText = "height:3px;background:#000";
    const fill = document.createElement("div");
    fill.style.cssText = "height:100%;width:0%;background:var(--c2c-mauve,#89b4fa);transition:width .1s";
    prog.appendChild(fill);

    root.append(bar, imgWrap, prog);
    document.body.appendChild(root);

    // drag
    let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;
    bar.addEventListener("mousedown", (e) => {
        if (e.target?.dataset?.role === "close") return;
        dragging = true; sx = e.clientX; sy = e.clientY;
        const r = root.getBoundingClientRect(); ox = r.left; oy = r.top;
        root.style.left = ox + "px"; root.style.top = oy + "px"; root.style.right = ""; root.style.bottom = "";
        e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        root.style.left = (ox + e.clientX - sx) + "px";
        root.style.top = (oy + e.clientY - sy) + "px";
    });
    window.addEventListener("mouseup", () => {
        if (!dragging) return; dragging = false;
        const r = root.getBoundingClientRect();
        _pos = { left: Math.round(r.left), top: Math.round(r.top) };
        try { localStorage.setItem("c2c.livePreview.pos", JSON.stringify(_pos)); } catch {}
    });
    bar.querySelector('[data-role="close"]').addEventListener("click", () => { root.style.display = "none"; });

    _root = root; _img = img; _bar = bar; _label = bar.querySelector('[data-role="lbl"]'); _bar._fill = fill;
    return root;
}

function _show() { const r = _ensureHud(); r.style.display = ""; return r; }

function _setPreview(blob) {
    if (!(_get(S.enabled, true))) return;
    try {
        _show();
        const url = URL.createObjectURL(blob);
        _img.onload = () => { if (_lastUrl) URL.revokeObjectURL(_lastUrl); _lastUrl = url; };
        _img.src = url;
    } catch { /* keep last frame */ }
}

function _setProgress(value, max, node) {
    try {
        if (!_root) return;
        const pct = max ? Math.round((value / max) * 100) : 0;
        _root.querySelector("div > div")?.style && (_bar._fill.style.width = pct + "%");
        if (_label) _label.textContent = max ? `${value}/${max} (${pct}%)` : "";
    } catch {}
}

function _install() {
    // Binary preview frames (the decoded latent preview the server streams).
    api.addEventListener("b_preview", (e) => {
        try {
            const d = e?.detail;
            if (d instanceof Blob) _setPreview(d);
            else if (d?.image instanceof Blob) _setPreview(d.image);
        } catch {}
    });
    api.addEventListener("progress", (e) => {
        try { const d = e?.detail || {}; _setProgress(d.value, d.max, d.node); } catch {}
    });
    // RESILIENCE: on error / interruption, do NOT blank — keep the last frame,
    // just mark the state so the user still sees where sampling reached.
    const _mark = (txt) => { try { if (_label) _label.textContent = txt; } catch {} };
    api.addEventListener("execution_error", () => _mark("⚠ error — last frame kept"));
    api.addEventListener("execution_interrupted", () => _mark("⏹ interrupted — last frame kept"));
    api.addEventListener("execution_success", () => _mark("✓ done"));
    api.addEventListener("execution_start", () => { try { if (_bar?._fill) _bar._fill.style.width = "0%"; } catch {} });
}

app.registerExtension({
    name: NS,
    settings: [
        { id: S.enabled, name: "C2C ▸ Live Preview ▸ Enabled", type: "boolean", default: true,
          tooltip: "Floating, resilient sampling/denoising preview that keeps the last frame through errors." },
        { id: S.size, name: "C2C ▸ Live Preview ▸ Width (px)", type: "slider",
          attrs: { min: 128, max: 640, step: 16 }, default: 256,
          onChange: (v) => { if (_root) _root.style.width = (Number(v) || 256) + "px"; } },
        { id: S.opacity, name: "C2C ▸ Live Preview ▸ Opacity", type: "slider",
          attrs: { min: 0.3, max: 1, step: 0.02 }, default: 0.96,
          onChange: (v) => { if (_root) _root.style.opacity = String(v); } },
    ],
    async setup() {
        _install();
        console.log("[C2C.LivePreview] ready — resilient HUD on b_preview/progress (keeps last frame).");
    },
});
