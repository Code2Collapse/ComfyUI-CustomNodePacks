// FILE: js/mec_progress_hud.js
// FEATURE: Per-node progress HUD overlay.
//
// Why this exists:
//   ComfyUI core already emits a `progress` websocket event for every
//   node that calls `comfy.utils.ProgressBar.update_absolute()`. The
//   default frontend paints a thin green fill at the top of the running
//   node, but with no text — so on small / zoomed-out nodes the user
//   cannot tell whether the bar is at 1% or 99%.
//
//   This extension hooks every node's drawing pipeline (via the
//   global LiteGraph node prototype) and paints a centred
//   "NN/MM · PCT%" string on top of the green fill. The label is only
//   drawn while the node is actively executing (progress < max).
//
//   Works for ALL nodes in the graph — not only the MEC pack — because
//   every progress-aware ComfyUI node uses the same websocket channel.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// node_id (string) -> { value, max, ts }
const PROGRESS = new Map();
// node_id currently being executed (from 'executing' event).
let EXEC_NODE = null;

api.addEventListener("progress", (ev) => {
    const d = ev.detail || ev;
    if (!d || d.value == null || d.max == null) return;
    const id = String(d.node ?? d.node_id ?? EXEC_NODE ?? "");
    if (!id) return;
    PROGRESS.set(id, { value: +d.value, max: +d.max || 1, ts: Date.now() });
    if (app.graph) app.graph.setDirtyCanvas(true, true);
});

api.addEventListener("executing", (ev) => {
    const d = ev.detail || ev;
    const id = d?.node ?? d;
    EXEC_NODE = (id == null || id === "") ? null : String(id);
    if (EXEC_NODE === null) {
        // Workflow finished — clear all progress overlays.
        PROGRESS.clear();
        if (app.graph) app.graph.setDirtyCanvas(true, true);
    }
});

api.addEventListener("executed", (ev) => {
    const d = ev.detail || ev;
    const id = String(d?.node ?? "");
    if (id) PROGRESS.delete(id);
    if (app.graph) app.graph.setDirtyCanvas(true, true);
});

api.addEventListener("execution_error", () => {
    PROGRESS.clear();
    if (app.graph) app.graph.setDirtyCanvas(true, true);
});

app.registerExtension({
    name: "MEC.ProgressHUD",
    async beforeRegisterNodeDef(nodeType /*, nodeData, app */) {
        const origDrawFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origDrawFg) origDrawFg.apply(this, arguments);
            if (this.flags?.collapsed) return;
            const id = String(this.id);
            const p = PROGRESS.get(id);
            if (!p) return;
            const pct = Math.max(0, Math.min(100, (p.value / p.max) * 100));
            if (pct >= 100) return;        // node finished; clear handled on 'executed'
            // Draw centered label across the title bar.
            ctx.save();
            ctx.font = "bold 11px ui-monospace, 'Cascadia Mono', monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            const text = `${p.value} / ${p.max}  ·  ${pct.toFixed(0)}%`;
            const cx = this.size[0] / 2;
            const cy = -10;                // inside title bar (negative y)
            // Background pill for legibility.
            const w = ctx.measureText(text).width + 12;
            const h = 16;
            ctx.fillStyle = "rgba(0,0,0,0.55)";
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(cx - w / 2, cy - h / 2, w, h, 6);
            else ctx.rect(cx - w / 2, cy - h / 2, w, h);
            ctx.fill();
            // Fill colour: green→yellow→red gradient by percentage.
            ctx.fillStyle = pct < 50 ? "#a6e3a1" : pct < 85 ? "#f9e2af" : "#fab387";
            ctx.fillText(text, cx, cy);
            ctx.restore();
        };
    },
});
