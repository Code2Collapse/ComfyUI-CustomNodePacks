// c2c_mask_integrity_hud.js — v2.0 P5: Mask Integrity HUD sidebar.
//
// Polls /c2c/mask_integrity/recent every ~1.5 s and renders:
//   • Latest report header (source node, batch size, flagged count).
//   • Sparkline of per-frame area, centroid drift, IoU-prev.
//   • Click-to-jump list of flagged frames.
//
// Server-side feed: nodes/mask_matting/integrity_bridge.py — populated by
// MaskTemporalMEC and MaskRefineMEC after every run.

import { app } from "../../scripts/app.js";

const TAB_ID = "c2c.mask_integrity";
const C = {
    bg: "#1e1e2e", bg2: "#181825", fg: "#cdd6f4", sub: "#a6adc8",
    red: "#f38ba8", green: "#a6e3a1", yellow: "#f9e2af",
    blue: "#89b4fa", mauve: "#cba6f7", border: "#313244",
};

let _pollTimer = null;
let _latest = null;
let _root = null;

async function fetchRecent(limit = 8) {
    try {
        const r = await fetch(`/c2c/mask_integrity/recent?limit=${limit}`);
        const j = await r.json();
        return j && j.success ? j.data : [];
    } catch (_) { return []; }
}

function buildView(root) {
    _root = root;
    root.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.style.cssText = `padding:10px;background:${C.bg};color:${C.fg};
        font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;
        height:100%;overflow:auto;`;
    wrap.innerHTML = `
        <h3 style="margin:0 0 10px;color:${C.mauve};font-size:13px;letter-spacing:0.5px;text-transform:uppercase">
            Mask Integrity HUD
        </h3>
        <div id="mi-status" style="font-size:10px;color:${C.sub};margin-bottom:8px">
            Waiting for first MaskTemporalMEC / MaskRefineMEC report…
        </div>
        <div style="display:flex;gap:6px;margin-bottom:8px">
            <button id="mi-refresh" style="${btnGhost()}">↻ Refresh</button>
            <button id="mi-clear"   style="${btnDanger()}">Clear</button>
        </div>
        <div id="mi-report"></div>
        <h4 style="margin:14px 0 4px;color:${C.sub};font-size:10px;letter-spacing:0.5px;text-transform:uppercase">History</h4>
        <div id="mi-history" style="font-size:10px;color:${C.sub}"></div>
    `;
    root.appendChild(wrap);
    wrap.querySelector("#mi-refresh").onclick = poll;
    wrap.querySelector("#mi-clear").onclick = async () => {
        try { await fetch("/c2c/mask_integrity/clear", { method: "POST" }); } catch (_) {}
        _latest = null;
        renderReport();
    };
    startPolling();
}

function startPolling() {
    if (_pollTimer) clearInterval(_pollTimer);
    poll();
    _pollTimer = setInterval(poll, 1500);
}

async function poll() {
    const list = await fetchRecent(8);
    if (!list.length) return;
    const newest = list[list.length - 1];
    const changed = !_latest || _latest.ts !== newest.ts;
    _latest = newest;
    if (changed) renderReport(list);
}

function renderReport(history) {
    if (!_root) return;
    const status = _root.querySelector("#mi-status");
    const rep = _root.querySelector("#mi-report");
    const hist = _root.querySelector("#mi-history");
    if (!_latest) {
        status.textContent = "No reports.";
        rep.innerHTML = "";
        hist.innerHTML = "";
        return;
    }
    const r = _latest;
    const flagColor = r.flagged_count === 0 ? C.green
                      : r.flagged_count <= 2 ? C.yellow : C.red;
    const tsStr = new Date(r.ts * 1000).toLocaleTimeString();
    status.innerHTML = `<span style="color:${flagColor};font-weight:bold">●</span> `
        + `${r.source} • B=${r.B} • flagged=${r.flagged_count} • ${tsStr}`;

    let html = "";
    if (r.flagged_count > 0) {
        const links = r.flagged_frames
            .map(i => `<a href="#" data-frame="${i}" style="color:${C.yellow};margin-right:6px">#${i}</a>`)
            .join("");
        html += `<div style="background:${C.bg2};border:1px solid ${C.border};border-radius:4px;padding:6px;margin-bottom:6px;font-size:11px">
            <div style="color:${C.sub};margin-bottom:4px">Flagged frames (click to jump):</div>
            <div>${links}</div>
        </div>`;
    } else {
        html += `<div style="background:${C.bg2};border:1px solid ${C.border};border-radius:4px;padding:6px;margin-bottom:6px;color:${C.green};font-size:11px">
            All frames within thresholds.
        </div>`;
    }
    // Sparklines for area / centroid / iou.
    const triplets = [
        ["Area",         r.area,         C.blue],
        ["Centroid Δ",   r.centroid_dx,  C.mauve],
        ["IoU (prev)",   r.iou_prev,     C.green],
    ];
    for (const [label, arr, color] of triplets) {
        if (Array.isArray(arr) && arr.length > 1) {
            html += `<div style="margin-bottom:6px">
                <div style="font-size:10px;color:${C.sub};margin-bottom:2px">${label}</div>
                ${sparklineSVG(arr, color, r.flagged_frames || [])}
            </div>`;
        }
    }
    rep.innerHTML = html;

    // Frame jump handlers
    rep.querySelectorAll("a[data-frame]").forEach(a => {
        a.onclick = (ev) => {
            ev.preventDefault();
            const idx = parseInt(a.dataset.frame, 10);
            jumpToFrame(idx);
        };
    });

    // History list (older first)
    if (history && history.length > 1) {
        const older = history.slice(0, -1).reverse();
        hist.innerHTML = older.map(h => {
            const c = h.flagged_count === 0 ? C.green : C.red;
            const t = new Date(h.ts * 1000).toLocaleTimeString();
            return `<div style="padding:2px 0">
                <span style="color:${c}">●</span> ${h.source} B=${h.B} flagged=${h.flagged_count} <span style="color:${C.sub}">${t}</span>
            </div>`;
        }).join("");
    } else {
        hist.innerHTML = "";
    }
}

function sparklineSVG(values, color, flaggedIndices) {
    const W = 240, H = 32, pad = 2;
    if (!values.length) return "";
    let lo = Math.min(...values), hi = Math.max(...values);
    if (lo === hi) { lo -= 0.001; hi += 0.001; }
    const n = values.length;
    const xStep = (W - pad * 2) / Math.max(1, n - 1);
    const yScale = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - pad * 2);
    const pts = values.map((v, i) => `${pad + i * xStep},${yScale(v)}`).join(" ");
    const flagged = new Set(flaggedIndices || []);
    const markers = values
        .map((v, i) => flagged.has(i)
            ? `<circle cx="${pad + i * xStep}" cy="${yScale(v)}" r="2.5" fill="${C.red}"/>`
            : "")
        .join("");
    return `<svg width="${W}" height="${H}" style="background:${C.bg2};border:1px solid ${C.border};border-radius:3px;display:block">
        <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.5"/>
        ${markers}
    </svg>`;
}

function jumpToFrame(idx) {
    // Best-effort: find the most-recent MaskTemporal/MaskRefine node and
    // pan the canvas to it. We can't seek inside the batch, but at least
    // we highlight which node produced the integrity report.
    const candidates = ["MaskTemporalMEC", "MaskRefineMEC"];
    const node = (app.graph?._nodes || []).find(n => candidates.includes(n.type));
    if (!node) return;
    try {
        app.canvas.centerOnNode(node);
        node.color = "#cba6f7";
        node.setDirtyCanvas?.(true, true);
        setTimeout(() => { node.color = null; node.setDirtyCanvas?.(true, true); }, 1500);
    } catch (_) {}
    console.log(`[c2c.mask_integrity] Highlighted ${node.type} (frame #${idx})`);
}

function btnGhost()  { return `background:${C.bg2};color:${C.fg};border:1px solid ${C.border};border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px`; }
function btnDanger() { return `background:${C.bg2};color:${C.red};border:1px solid ${C.red};border-radius:4px;padding:4px 10px;cursor:pointer;font-size:11px`; }

app.registerExtension({
    name: "c2c.mask_integrity.hud",
    async setup() {
        try {
            app.extensionManager?.registerSidebarTab?.({
                id: TAB_ID,
                icon: "pi pi-chart-line",
                title: "Mask HUD",
                tooltip: "C2C Mask Integrity HUD — per-frame drift report",
                type: "custom",
                render: (el) => buildView(el),
            });
        } catch (exc) {
            console.warn("[c2c.mask_integrity.hud] sidebar registration failed:", exc);
        }
    },
});

window.__C2C_MASK_INTEGRITY__ = { buildView, fetchRecent };
