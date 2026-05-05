// FILE: web/extensions/nukenodemax/insight_overlay.js
// FEATURE: W1 — DOM heatmap overlay for per-node VRAM / timing
// INTEGRATES WITH: nodes/insight.py (event "nukenodemax.insight")
//
// Listens to the ComfyUI socket and paints a colored badge on each node's
// LiteGraph DOM container. Color = VRAM delta normalised against the heaviest
// node in the current run; tooltip shows ms + MB. NO console.log telemetry.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const STATE = {
    nodes: new Map(),  // node_id -> {elapsed_ms, vram_delta_mb, error?}
    maxVram: 1,
    maxMs: 1,
};

function ensureBadge(nodeId) {
    const node = app.graph?._nodes_by_id?.[nodeId];
    if (!node) return null;
    let badge = node._mecInsightBadge;
    if (!badge) {
        badge = {
            text: "",
            color: "#444",
            tooltip: "",
        };
        node._mecInsightBadge = badge;
    }
    return badge;
}

function colorFor(deltaMb, maxMb, isError) {
    if (isError) return "#ff3a3a";
    const t = Math.max(0, Math.min(1, deltaMb / Math.max(maxMb, 1)));
    // green -> yellow -> red
    const r = Math.round(60 + 195 * t);
    const g = Math.round(200 - 140 * t);
    const b = 60;
    return `rgb(${r},${g},${b})`;
}

function refresh() {
    if (!app.graph) return;
    for (const [id, info] of STATE.nodes.entries()) {
        const node = app.graph._nodes_by_id?.[id];
        if (!node) continue;
        const badge = ensureBadge(id);
        if (!badge) continue;
        if (info.error) {
            badge.text = "✕ " + (info.exc_type || "err");
            badge.color = colorFor(0, STATE.maxVram, true);
            badge.tooltip = `${info.exc_type}: ${info.exc_msg || ""}\n${info.hint || ""}`;
        } else {
            badge.text = `${info.elapsed_ms.toFixed(0)}ms · ${info.vram_delta_mb.toFixed(0)}MB`;
            badge.color = colorFor(info.vram_delta_mb, STATE.maxVram, false);
            badge.tooltip = `elapsed=${info.elapsed_ms.toFixed(1)}ms  vram_delta=${info.vram_delta_mb.toFixed(1)}MB  peak=${(info.vram_peak_mb || 0).toFixed(1)}MB`;
        }
    }
    app.graph.setDirtyCanvas(true, true);
}

function onInsight(ev) {
    const data = ev.detail || ev;
    if (!data || !data.node_id) return;
    if (data.type === "node_done") {
        STATE.nodes.set(data.node_id, {
            elapsed_ms: data.elapsed_ms || 0,
            vram_delta_mb: data.vram_delta_mb || 0,
            vram_peak_mb: data.vram_peak_mb || 0,
        });
        if ((data.vram_delta_mb || 0) > STATE.maxVram) STATE.maxVram = data.vram_delta_mb;
        if ((data.elapsed_ms || 0) > STATE.maxMs) STATE.maxMs = data.elapsed_ms;
    } else if (data.type === "node_error") {
        STATE.nodes.set(data.node_id, {
            error: true,
            elapsed_ms: data.elapsed_ms || 0,
            vram_delta_mb: 0,
            exc_type: data.exc_type,
            exc_msg: data.exc_msg,
            hint: data.hint,
            trace: data.trace,
        });
    }
    refresh();
}

app.registerExtension({
    name: "nukenodemax.insight_overlay",
    setup() {
        api.addEventListener("nukenodemax.insight", onInsight);

        // Hook draw to paint badges in-canvas.
        const origDraw = LGraphCanvas.prototype.drawNodeShape;
        LGraphCanvas.prototype.drawNodeShape = function (node, ctx, size, fgColor, bgColor, selected, mouseOver) {
            origDraw.apply(this, arguments);
            const badge = node._mecInsightBadge;
            if (!badge || !badge.text) return;
            ctx.save();
            ctx.fillStyle = badge.color;
            const w = ctx.measureText(badge.text).width + 12;
            const h = 16;
            const x = size[0] - w - 4;
            const y = -h - 2;
            ctx.fillRect(x, y, w, h);
            ctx.fillStyle = "#fff";
            ctx.font = "10px monospace";
            ctx.textBaseline = "middle";
            ctx.fillText(badge.text, x + 6, y + h / 2);
            ctx.restore();
        };

        // Tooltip on hover via title append.
        const origGetTitle = LGraphCanvas.prototype.processMouseMove;
        // Lightweight: rely on the badge tooltip stored on the node object;
        // user can right-click for full hint. (No DOM tooltip layer to avoid
        // fighting ComfyUI's own.)

        // Right-click menu: show hint for failing nodes.
        const origMenu = LGraphCanvas.prototype.getNodeMenuOptions;
        LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
            const opts = origMenu ? origMenu.apply(this, arguments) : [];
            const badge = node._mecInsightBadge;
            if (badge && badge.tooltip) {
                opts.unshift({
                    content: "Insight: " + (STATE.nodes.get(node.id)?.error ? "show error hint" : "show stats"),
                    callback: () => alert(badge.tooltip),
                });
                opts.unshift(null);  // separator
            }
            return opts;
        };
    },
});
