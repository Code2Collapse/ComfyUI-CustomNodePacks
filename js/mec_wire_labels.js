/**
 * mec_wire_labels.js — Phase 12: Named Wire Labels
 *
 * Double-click on a connection to give it a name. Labels render at the
 * midpoint of the wire and persist in the workflow JSON via a graph
 * "extra" field — so they survive save/load.
 *
 * Setting:
 *   mec.wire_labels.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

const STORE_KEY = "mec_wire_labels"; // stored under graph.extra[STORE_KEY]
const STYLE_ID  = "mec-wire-labels-style";

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.mec-wire-label-dialog {
    position: fixed;
    z-index: 99997;
    background: #181825;
    border: 1px solid #89b4fa;
    border-radius: 6px;
    padding: 10px 12px;
    color: #cdd6f4;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px;
    width: 220px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.6);
}
.mec-wire-label-dialog h4 { margin: 0 0 6px 0; color: #89b4fa; font-size: 12px; }
.mec-wire-label-dialog input {
    width: 100%;
    background: #313244;
    border: 1px solid #45475a;
    color: #cdd6f4;
    border-radius: 3px;
    padding: 4px 6px;
    font-size: 12px;
}
.mec-wire-label-dialog .wl-actions {
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
    gap: 6px;
}
.mec-wire-label-dialog button {
    background: #313244;
    border: 1px solid #45475a;
    color: #cdd6f4;
    border-radius: 3px;
    padding: 3px 10px;
    cursor: pointer;
    font-size: 11px;
}
.mec-wire-label-dialog button.primary { border-color: #89b4fa; color: #89b4fa; }
.mec-wire-label-dialog button.danger  { border-color: #f38ba8; color: #f38ba8; }
    `.trim();
    document.head.appendChild(style);
}

function _labelStore() {
    const g = app.graph;
    if (!g) return {};
    g.extra = g.extra || {};
    g.extra[STORE_KEY] = g.extra[STORE_KEY] || {};
    return g.extra[STORE_KEY];
}

function _getLabel(linkId)        { return _labelStore()[linkId]; }
function _setLabel(linkId, text)  {
    const store = _labelStore();
    if (text && text.trim()) store[linkId] = text.trim();
    else delete store[linkId];
    app.canvas?.setDirty?.(true, true);
}

function _findLinkAt(canvas, e) {
    const offs = canvas.convertEventToCanvasOffset
        ? canvas.convertEventToCanvasOffset(e)
        : [e.canvasX, e.canvasY];
    const graph = canvas.graph;
    if (!graph) return null;
    const links = Array.isArray(graph.links) ? graph.links : Object.values(graph.links || {});
    let best = null, bestDist = 8;
    for (const link of links) {
        if (!link) continue;
        const src = graph.getNodeById(link.origin_id);
        const dst = graph.getNodeById(link.target_id);
        if (!src || !dst) continue;
        const a = src.getConnectionPos
            ? src.getConnectionPos(false, link.origin_slot)
            : null;
        const b = dst.getConnectionPos
            ? dst.getConnectionPos(true,  link.target_slot)
            : null;
        if (!a || !b) continue;
        const d = _distToSegment(offs[0], offs[1], a[0], a[1], b[0], b[1]);
        if (d < bestDist) { bestDist = d; best = link; }
    }
    return best;
}

function _distToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 === 0) return Math.hypot(px - ax, py - ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function _openDialog(linkId, screenX, screenY) {
    document.querySelectorAll(".mec-wire-label-dialog").forEach(e => e.remove());
    const existing = _getLabel(linkId) || "";
    const root = document.createElement("div");
    root.className = "mec-wire-label-dialog";
    root.innerHTML = `
        <h4>🏷 Wire label</h4>
        <input type="text" maxlength="40" placeholder="e.g. base latent">
        <div class="wl-actions">
            <button class="danger">Clear</button>
            <div>
                <button class="cancel">Cancel</button>
                <button class="primary">Save</button>
            </div>
        </div>
    `;
    document.body.appendChild(root);
    const W = 240, H = 110;
    root.style.left = Math.min(window.innerWidth - W - 8, screenX) + "px";
    root.style.top  = Math.min(window.innerHeight - H - 8, screenY) + "px";

    const input = root.querySelector("input");
    input.value = existing;
    input.focus();
    input.select();

    const finish = (commit, clear=false) => {
        if (commit && !clear) _setLabel(linkId, input.value);
        if (clear) _setLabel(linkId, "");
        root.remove();
    };
    root.querySelector(".primary").addEventListener("click", () => finish(true));
    root.querySelector(".cancel").addEventListener("click", () => finish(false));
    root.querySelector(".danger").addEventListener("click", () => finish(true, true));
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter")  finish(true);
        if (e.key === "Escape") finish(false);
    });
}

function _patchDblClick() {
    const orig = LGraphCanvas.prototype.processMouseDown;
    if (!orig || orig._mecWireLabelsPatched) return;
    LGraphCanvas.prototype.processMouseDown = function (e) {
        const r = orig.call(this, e);
        try {
            const enabled = (() => {
                try { return app.ui.settings.getSettingValue("mec.wire_labels.enabled", true); }
                catch { return true; }
            })();
            if (!enabled) return r;
            // Only react on left-button double-clicks that didn't hit a node.
            if (e.which !== 1 || !this.last_mouseclick) return r;
            const node = this.graph?.getNodeOnPos?.(e.canvasX, e.canvasY, this.visible_nodes);
            if (node) return r;
            const now = performance.now();
            if (!this._mecLastClickT) this._mecLastClickT = 0;
            const isDouble = now - this._mecLastClickT < 300;
            this._mecLastClickT = now;
            if (!isDouble) return r;
            const link = _findLinkAt(this, e);
            if (link) _openDialog(link.id, e.clientX, e.clientY);
        } catch (err) {
            console.warn("[MEC.WireLabels] dblclick handler error:", err);
        }
        return r;
    };
    LGraphCanvas.prototype.processMouseDown._mecWireLabelsPatched = true;
}

function _patchDrawConnections() {
    const orig = LGraphCanvas.prototype.drawConnections;
    if (!orig || orig._mecWireLabelsPatched) return;
    LGraphCanvas.prototype.drawConnections = function (ctx) {
        const r = orig.call(this, ctx);
        try {
            const enabled = (() => {
                try { return app.ui.settings.getSettingValue("mec.wire_labels.enabled", true); }
                catch { return true; }
            })();
            if (!enabled) return r;
            const graph = this.graph;
            if (!graph) return r;
            const store = (graph.extra && graph.extra[STORE_KEY]) || {};
            const linkIds = Object.keys(store);
            if (linkIds.length === 0) return r;
            const links = Array.isArray(graph.links) ? graph.links : Object.values(graph.links || {});
            ctx.save();
            ctx.font = "bold 11px sans-serif";
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            for (const link of links) {
                if (!link) continue;
                const label = store[link.id];
                if (!label) continue;
                const src = graph.getNodeById(link.origin_id);
                const dst = graph.getNodeById(link.target_id);
                if (!src || !dst) continue;
                const a = src.getConnectionPos(false, link.origin_slot);
                const b = dst.getConnectionPos(true,  link.target_slot);
                const mx = (a[0] + b[0]) / 2;
                const my = (a[1] + b[1]) / 2;
                const padX = 5;
                const w = ctx.measureText(label).width + padX * 2;
                ctx.fillStyle = "rgba(24, 24, 37, 0.95)";
                ctx.strokeStyle = "#89b4fa";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.roundRect
                    ? ctx.roundRect(mx - w / 2, my - 9, w, 18, 4)
                    : ctx.rect(mx - w / 2, my - 9, w, 18);
                ctx.fill();
                ctx.stroke();
                ctx.fillStyle = "#89b4fa";
                ctx.fillText(label, mx, my + 1);
            }
            ctx.restore();
        } catch (err) {
            console.warn("[MEC.WireLabels] draw error:", err);
        }
        return r;
    };
    LGraphCanvas.prototype.drawConnections._mecWireLabelsPatched = true;
}

app.registerExtension({
    name: "MEC.WireLabels",
    settings: [
        {
            id: "mec.wire_labels.enabled",
            name: "Wire Labels: double-click connection to name it",
            type: "boolean",
            defaultValue: true,
            onChange: () => app.canvas?.setDirty?.(true, true),
        },
    ],
    async setup() {
        _injectStyle();
        _patchDblClick();
        _patchDrawConnections();
        console.log("[MEC.WireLabels] Loaded.");
    },
});
