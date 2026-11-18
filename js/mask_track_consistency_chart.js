/**
 * Mask Track - consistency_check chart (L7.127).
 *
 * consistency_check scores how much the matte changes from one frame to the next; with metric edge_stability it is
 * the alpha change in a band around the matte EDGE, motion-compensated when the video is wired. The number alone
 * says "something flickers"; the curve says WHERE - which frames to look at or re-key. The node sends its per-pair
 * report as ui.mt_consistency (nodes/mask_tracker_mec.py) only in that mode, so the panel is created on the first
 * such run and hidden whenever another mode is selected (A28: only what the current mode uses is on screen).
 */
import { app } from "../../scripts/app.js";
import { lineChart, mountPanel } from "./c2c_ui/index.js";
import { setWidgetVisible, vueFitNodeHeight, vueNodesActive, vueSyncNodeWidgets } from "./_widget_visibility.js";

const NODE = "MaskTrackerMEC";
const MODE = "consistency_check";
const ST = "_c2cMtChart";
const PANEL_MIN = 200;

function modeOf(node) {
    return node.widgets?.find((w) => w.name === "mode")?.value;
}

function refit(node) {
    try {
        if (vueNodesActive()) { vueSyncNodeWidgets(node); vueFitNodeHeight(node); }
        else {
            const sz = node.computeSize();
            node.setSize([node.size[0], sz[1]]);
        }
        node.setDirtyCanvas?.(true, true);
    } catch (_e) { /* never break the graph */ }
}

function showForMode(node) {
    const st = node[ST];
    if (!st) return;
    const on = modeOf(node) === MODE;
    if (st.visible === on) return;
    st.visible = on;
    setWidgetVisible(st.widget, on);
    if (on) st.chart.redraw();
    refit(node);
}

function build(node) {
    if (node[ST]) return node[ST];
    const root = document.createElement("div");
    Object.assign(root.style, { display: "flex", flexDirection: "column", gap: "4px", width: "100%", height: "100%" });
    const header = document.createElement("div");
    header.className = "c2c-ui-chart__header";
    const pill = document.createElement("span");
    pill.className = "c2c-ui-chart__pill";
    pill.textContent = "—";
    const summary = document.createElement("span");
    summary.className = "c2c-ui-chart__header-text";
    header.append(pill, summary);
    root.appendChild(header);
    const chart = lineChart({
        minHeight: 120,
        xLabel: "frame pair (i -> i+1)",
        yLabel: "change",
        xFormat: (v) => String(Math.round(v)),
        yFormat: (v) => v.toFixed(3),
        series: [{ id: "score", label: "score", color: "--cu-series-2", axis: "y", points: true }],
        empty: { title: "Temporal consistency", hint: "Run consistency_check to see which frames change." },
    });
    chart.setState("empty");
    root.appendChild(chart.el);
    const widget = mountPanel(node, "mt_consistency_chart", root, { minHeight: PANEL_MIN });
    widget.serialize = false;                      // display only: never in widgets_values (L9.102 / L9.237)
    widget.onPanelResize = () => chart.redraw();
    const st = { widget, chart, pill, summary, visible: true };
    node[ST] = st;
    const origRemoved = node.onRemoved;
    node.onRemoved = function (...a) {
        try { chart.destroy(); } catch (_e) { /* ignore */ }
        delete node[ST];
        return origRemoved?.apply(this, a);
    };
    return st;
}

function apply(node, report) {
    const st = build(node);
    const pairs = Array.isArray(report.pairs) ? report.pairs : [];
    const xs = pairs.map((p) => Number(p.i));
    const ys = pairs.map((p) => Number(p.score));
    st.chart.setData(xs, { score: ys });
    const worst = Array.isArray(report.worst_pairs) ? report.worst_pairs : [];
    st.chart.setMarkers(worst.slice(0, 3).filter((i) => i < ys.length)
        .map((i) => ({ x: i, seriesId: "score", label: `f${i}->${i + 1} ${ys[i].toFixed(3)}` })));
    st.chart.setState(pairs.length ? "ready" : "empty");
    const mean = Number(report.mean ?? report.flicker_score ?? 0);
    const steady = mean < 0.02;
    st.pill.textContent = pairs.length ? (steady ? "STEADY" : "CHANGES") : "—";
    st.pill.classList.toggle("c2c-ui-chart__pill--ok", pairs.length > 0 && steady);
    st.pill.classList.toggle("c2c-ui-chart__pill--danger", pairs.length > 0 && !steady);
    const comp = report.motion_compensated === true ? ", motion followed" : "";
    st.summary.textContent = `${report.metric || ""}: mean ${mean.toFixed(4)}, max ${Number(report.max ?? 0).toFixed(4)}`
        + ` over ${report.n_pairs ?? pairs.length} pair(s)${comp}`;
    st.visible = null;
    showForMode(node);
}

app.registerExtension({
    name: "C2C.MaskTrack.ConsistencyChart",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const r = executed?.apply(this, arguments);
            try {
                const raw = output?.mt_consistency?.[0];
                if (typeof raw === "string") apply(this, JSON.parse(raw));
            } catch (_e) { /* never break the graph */ }
            return r;
        };
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created?.apply(this, arguments);
            const w = this.widgets?.find((x) => x.name === "mode");
            if (w) {
                const cb = w.callback;
                const node = this;
                w.callback = function (...a) { const out = cb?.apply(this, a); showForMode(node); return out; };
            }
            return r;
        };
    },
});
