// FILE: web/extensions/nukenodemax/insight_overlay.js
// FEATURE: W1 — DOM heatmap overlay for per-node VRAM / timing / CPU / RAM
// INTEGRATES WITH: nodes/insight.py (event "nukenodemax.insight")
//
// Listens to the ComfyUI socket and paints a colored badge BELOW each node's
// LiteGraph shape (never over the title bar). Two compact lines:
//      line 1: <ms>ms · <vram>MB
//      line 2: cpu <ms>ms · ram <mb>MB
// Color = VRAM delta normalised against the heaviest node in the current run.
// Subgraph parent nodes show aggregated stats from their child executions.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { findNodeAnywhere } from "../_subgraph_walk.js";
import { reportFailure as __c2cReport } from "../_c2c_report.js";

const STATE = {
    nodes: new Map(),  // node_id -> {elapsed_ms, vram_delta_mb, cpu_ms, ram_delta_mb, error?}
    maxVram: 1,
    maxMs: 1,
};

function ensureBadge(nodeOrId) {
    // Accepts either a resolved LiteGraph node or a node id. When given an id,
    // walks subgraphs to locate the owner.
    let node = null;
    if (nodeOrId && typeof nodeOrId === "object") {
        node = nodeOrId;
    } else {
        node = findNodeAnywhere(nodeOrId)?.node || app.graph?._nodes_by_id?.[nodeOrId] || null;
    }
    if (!node) return null;
    let badge = node._mecInsightBadge;
    if (!badge) {
        badge = { line1: "", line2: "", color: "#444", tooltip: "" };
        node._mecInsightBadge = badge;
    }
    return badge;
}

function colorFor(deltaMb, maxMb, isError) {
    if (isError) return "#ff3a3a";
    const t = Math.max(0, Math.min(1, deltaMb / Math.max(maxMb, 1)));
    const r = Math.round(60 + 195 * t);
    const g = Math.round(200 - 140 * t);
    const b = 60;
    return `rgb(${r},${g},${b})`;
}

function fmtMs(v) {
    if (!isFinite(v)) return "0ms";
    if (v >= 10000) return `${(v / 1000).toFixed(1)}s`;
    return `${v.toFixed(0)}ms`;
}
function fmtMb(v) {
    if (!isFinite(v)) return "0MB";
    if (Math.abs(v) >= 1024) return `${(v / 1024).toFixed(1)}GB`;
    return `${v.toFixed(0)}MB`;
}

function refresh() {
    if (!app.graph) return;
    for (const [id, info] of STATE.nodes.entries()) {
        // Subgraph-aware: id may be composite ("7:5") or live inside a
        // SubgraphNode the root graph cannot see.
        const resolved = findNodeAnywhere(id);
        const node = resolved?.node || app.graph._nodes_by_id?.[id];
        if (!node) continue;
        const badge = ensureBadge(node);
        if (!badge) continue;
        if (info.error) {
            badge.line1 = "✕ " + (info.exc_type || "err");
            badge.line2 = "";
            badge.color = colorFor(0, STATE.maxVram, true);
            badge.tooltip = `${info.exc_type}: ${info.exc_msg || ""}\n${info.hint || ""}`;
        } else {
            // Single compact line: "12ms · 4MB · cpu 8ms · ram 1MB"
            badge.line1 = `${fmtMs(info.elapsed_ms || 0)} · ${fmtMb(info.vram_delta_mb || 0)} · cpu ${fmtMs(info.cpu_ms || 0)} · ram ${fmtMb(info.ram_delta_mb || 0)}`;
            badge.line2 = "";
            badge.color = colorFor(info.vram_delta_mb || 0, STATE.maxVram, false);
            const parts = [
                `elapsed=${(info.elapsed_ms || 0).toFixed(1)}ms`,
                `cpu=${(info.cpu_ms || 0).toFixed(1)}ms`,
                `vram_delta=${(info.vram_delta_mb || 0).toFixed(1)}MB`,
                `vram_peak=${(info.vram_peak_mb || 0).toFixed(1)}MB`,
                `ram_delta=${(info.ram_delta_mb || 0).toFixed(1)}MB`,
            ];
            if (info.subgraph_children) parts.push(`subgraph_children=${info.subgraph_children}`);
            badge.tooltip = parts.join("  ");
        }
    }
    app.graph.setDirtyCanvas(true, true);
}

function _aggregateInto(parentId, sample) {
    if (!parentId) return;
    let agg = STATE.nodes.get(parentId);
    if (!agg || !agg.subgraph_children) {
        agg = {
            elapsed_ms: 0,
            cpu_ms: 0,
            vram_delta_mb: 0,
            vram_peak_mb: 0,
            ram_delta_mb: 0,
            subgraph_children: 0,
        };
    }
    agg.elapsed_ms    += sample.elapsed_ms    || 0;
    agg.cpu_ms        += sample.cpu_ms        || 0;
    agg.vram_delta_mb += sample.vram_delta_mb || 0;
    agg.ram_delta_mb  += sample.ram_delta_mb  || 0;
    agg.vram_peak_mb   = Math.max(agg.vram_peak_mb || 0, sample.vram_peak_mb || 0);
    agg.subgraph_children += 1;
    STATE.nodes.set(parentId, agg);
    if (agg.vram_delta_mb > STATE.maxVram) STATE.maxVram = agg.vram_delta_mb;
}

function onInsight(ev) {
    const data = ev.detail || ev;
    if (!data || data.node_id === undefined || data.node_id === null) return;
    if (data.type === "node_done") {
        const sample = {
            elapsed_ms: data.elapsed_ms || 0,
            cpu_ms: data.cpu_ms || 0,
            vram_delta_mb: data.vram_delta_mb || 0,
            vram_peak_mb: data.vram_peak_mb || 0,
            ram_delta_mb: data.ram_delta_mb || 0,
        };
        STATE.nodes.set(data.node_id, sample);
        if (sample.vram_delta_mb > STATE.maxVram) STATE.maxVram = sample.vram_delta_mb;
        if (sample.elapsed_ms    > STATE.maxMs)   STATE.maxMs   = sample.elapsed_ms;
        if (data.parent_id) _aggregateInto(data.parent_id, sample);
    } else if (data.type === "node_error") {
        STATE.nodes.set(data.node_id, {
            error: true,
            elapsed_ms: data.elapsed_ms || 0,
            cpu_ms: 0,
            vram_delta_mb: 0,
            ram_delta_mb: 0,
            exc_type: data.exc_type,
            exc_msg: data.exc_msg,
            hint: data.hint,
            trace: data.trace,
        });
        if (data.parent_id) {
            const p = STATE.nodes.get(data.parent_id) || {};
            p.error = true;
            p.exc_type = data.exc_type;
            p.exc_msg = data.exc_msg;
            p.hint = data.hint;
            STATE.nodes.set(data.parent_id, p);
        }
    }
    refresh();
}

app.registerExtension({
    name: "nukenodemax.insight_overlay",
    settings: [
        {
            id: "mec.insight_overlay.always_on",
            name: "Insight Overlay: always-on badges",
            tooltip: "Paint VRAM/timing badges below every node continuously. When OFF (default), badges are still computed and visible on right-click → Insight, but the canvas stays clean. Disable this if the badges scale awkwardly when you zoom the canvas.",
            type: "boolean",
            default: false,
        },
    ],
    setup() {
        api.addEventListener("nukenodemax.insight", onInsight);

        // Paint badges BELOW the node body so the title bar stays clean.
        const origDraw = LGraphCanvas.prototype.drawNodeShape;
        LGraphCanvas.prototype.drawNodeShape = function (node, ctx, size, fgColor, bgColor, selected, mouseOver) {
            origDraw.apply(this, arguments);
            // Per user mandate 2026-05-19: do NOT always-paint. These
            // badges live in graph space and therefore scale with the
            // LiteGraph zoom — which the user described as "zooming in
            // and out when moving around the canvas". Gate behind a
            // setting (default OFF). Show on hover OR when enabled.
            let alwaysOn = false;
            try { alwaysOn = !!app.ui.settings.getSettingValue("mec.insight_overlay.always_on", false); } catch (__c2cErr) { __c2cReport("insight_overlay", __c2cErr); }
            if (!alwaysOn && !mouseOver) return;

            const badge = node._mecInsightBadge;
            if (!badge || (!badge.line1 && !badge.line2)) return;

            ctx.save();
            ctx.font = "10px monospace";
            const lines = badge.line2 ? [badge.line1, badge.line2] : [badge.line1];
            const lineH = 12;
            const h = lineH * lines.length + 4;

            // Full node-width strip just below the node, so it never
            // expands the visual footprint horizontally.
            const x = 0;
            const w = size[0];
            const y = size[1] + 4;

            // Auto-shrink the font if the text is wider than the node.
            let fontPx = 10;
            const maxTextWidth = w - 8;
            let textWidth = Math.max(...lines.map((s) => ctx.measureText(s).width));
            while (textWidth > maxTextWidth && fontPx > 7) {
                fontPx -= 1;
                ctx.font = `${fontPx}px monospace`;
                textWidth = Math.max(...lines.map((s) => ctx.measureText(s).width));
            }

            ctx.fillStyle = badge.color;
            ctx.fillRect(x, y, w, h);
            ctx.fillStyle = "#fff";
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            const cx = x + w / 2;
            for (let i = 0; i < lines.length; i++) {
                ctx.fillText(lines[i], cx, y + 2 + lineH * (i + 0.5));
            }
            ctx.restore();
        };

        // Right-click menu: show full hint / stats.
        const origMenu = LGraphCanvas.prototype.getNodeMenuOptions;
        LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
            const opts = origMenu ? origMenu.apply(this, arguments) : [];
            const badge = node._mecInsightBadge;
            if (badge && badge.tooltip) {
                const isErr = !!STATE.nodes.get(node.id)?.error;
                opts.unshift({
                    content: "Insight: " + (isErr ? "show error hint" : "show stats"),
                    callback: () => alert(badge.tooltip),
                });
                opts.unshift(null);  // separator
            }
            return opts;
        };
    },
});
