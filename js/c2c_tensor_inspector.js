/**
 * c2c_tensor_inspector.js — Phase 4: Live Tensor Inspector
 *
 * When the user clicks an output slot of any node, fetch
 * /mec/tensor_snapshot?node_id=<id>&slot=<n> and render a small floating
 * panel with shape / dtype / device / min / max / mean / nan-count / etc.
 *
 * Setting:
 *   mec.tensor_inspector.enabled — bool (default true)
 */

import { app } from "../../scripts/app.js";

const STYLE_ID = "mec-tensor-inspector-style";
const PANEL_ID = "mec-tensor-inspector-panel";

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${PANEL_ID} {
    position: fixed;
    z-index: var(--c2c-z-hud, 1000);
    width: 320px;
    max-height: 60vh;
    overflow-y: auto;
    background: var(--c2c-bg2);
    border: 1px solid var(--c2c-sapphire);
    border-radius: 6px;
    padding: 10px 12px;
    color: var(--c2c-fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 11px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.6);
    display: none;
}
#${PANEL_ID}.visible { display: block; }
#${PANEL_ID} .ti-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--c2c-surface0);
    padding-bottom: 4px;
    margin-bottom: 6px;
}
#${PANEL_ID} .ti-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--c2c-sapphire);
}
#${PANEL_ID} .ti-close {
    background: transparent;
    border: none;
    color: var(--c2c-overlay0);
    font-size: 14px;
    cursor: pointer;
}
#${PANEL_ID} .ti-empty {
    color: var(--c2c-overlay0);
    font-style: italic;
    text-align: center;
    padding: 8px 0;
}
#${PANEL_ID} table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
}
#${PANEL_ID} td {
    padding: 2px 4px;
    vertical-align: top;
}
#${PANEL_ID} td.k {
    color: var(--c2c-slate400);
    width: 32%;
    white-space: nowrap;
}
#${PANEL_ID} td.v {
    color: var(--c2c-fg);
    font-family: monospace;
    font-size: 11px;
    word-break: break-all;
}
#${PANEL_ID} .ti-slot {
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px dashed var(--c2c-surface1);
}
#${PANEL_ID} .ti-slot:first-child { border-top: none; margin-top: 0; padding-top: 0; }
#${PANEL_ID} .ti-slot-name {
    font-weight: 700;
    color: var(--c2c-okSoft);
    margin-bottom: 2px;
}
#${PANEL_ID} .ti-warn { color: var(--c2c-red); }
    `.trim();
    document.head.appendChild(style);
}

function _ensurePanel() {
    let panel = document.getElementById(PANEL_ID);
    if (!panel) {
        panel = document.createElement("div");
        panel.id = PANEL_ID;
        document.body.appendChild(panel);
    }
    return panel;
}

function _esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function _fmtNum(n) {
    if (n === null || n === undefined) return "—";
    if (typeof n !== "number") return String(n);
    if (!isFinite(n)) return String(n);
    const abs = Math.abs(n);
    if (abs !== 0 && (abs < 0.001 || abs >= 10000)) return n.toExponential(3);
    return n.toFixed(4);
}

function _rowsForValue(v) {
    const rows = [];
    const add = (k, val) => rows.push([k, val]);
    if (!v) { add("(none)", ""); return rows; }
    add("kind", v.kind);
    if (v.shape) add("shape", "[" + v.shape.join(", ") + "]");
    if (v.dtype) add("dtype", v.dtype);
    if (v.device) add("device", v.device);
    if (v.numel !== undefined) add("numel", v.numel.toLocaleString());
    if (v.batch_len !== undefined) add("batch", v.batch_len);
    if (v.min !== undefined) add("min", _fmtNum(v.min));
    if (v.max !== undefined) add("max", _fmtNum(v.max));
    if (v.mean !== undefined) add("mean", _fmtNum(v.mean));
    if (v.std !== undefined) add("std", _fmtNum(v.std));
    if (v.n_nan)   add("⚠ NaN count", `<span class="ti-warn">${v.n_nan}</span>`);
    if (v.n_inf)   add("⚠ Inf count", `<span class="ti-warn">${v.n_inf}</span>`);
    if (v.keys) add("keys", v.keys.map(_esc).join(", "));
    if (v.len !== undefined && v.kind !== "tensor" && v.kind !== "ndarray")
        add("len", v.len);
    if (v.preview !== undefined) add("preview", _esc(v.preview));
    if (v.value !== undefined) add("value", _esc(String(v.value)));
    if (v.repr) add("repr", _esc(v.repr));
    if (v.samples) {
        add("samples →", "");
        for (const [k, val] of _rowsForValue(v.samples))
            add("  " + k, val);
    }
    if (v.first) {
        add("first →", "");
        for (const [k, val] of _rowsForValue(v.first))
            add("  " + k, val);
    }
    if (v.error) add("error", `<span class="ti-warn">${_esc(v.error)}</span>`);
    if (v.stat_error) add("stat_error", `<span class="ti-warn">${_esc(v.stat_error)}</span>`);
    return rows;
}

function _renderPanel(panel, nodeTitle, nodeId, data) {
    if (!data || !data.available) {
        panel.innerHTML = `
            <div class="ti-header">
                <span class="ti-title">🔬 Tensor Inspector</span>
                <button class="ti-close">×</button>
            </div>
            <div class="ti-empty">No snapshot yet for ${_esc(nodeTitle)} (#${_esc(nodeId)}).<br>Run the prompt first.</div>
        `;
    } else {
        const slotsHtml = (data.slots || []).map(s => {
            const rowsHtml = _rowsForValue(s).map(([k, v]) =>
                `<tr><td class="k">${_esc(k)}</td><td class="v">${v}</td></tr>`
            ).join("");
            return `
                <div class="ti-slot">
                    <div class="ti-slot-name">slot ${s.slot}</div>
                    <table>${rowsHtml}</table>
                </div>`;
        }).join("");
        panel.innerHTML = `
            <div class="ti-header">
                <span class="ti-title">🔬 ${_esc(nodeTitle)} <span style="color:var(--c2c-overlay0);font-weight:400;">#${_esc(nodeId)}</span></span>
                <button class="ti-close">×</button>
            </div>
            ${slotsHtml || '<div class="ti-empty">No slot data.</div>'}
        `;
    }
    panel.querySelector(".ti-close")?.addEventListener("click", () => {
        panel.classList.remove("visible");
    });
}

function _positionPanel(panel, mouseX, mouseY) {
    const W = 320;
    const H = Math.min(window.innerHeight * 0.6, 400);
    let x = mouseX + 12;
    let y = mouseY + 12;
    if (x + W > window.innerWidth)  x = mouseX - W - 12;
    if (y + H > window.innerHeight) y = window.innerHeight - H - 8;
    panel.style.left = Math.max(8, x) + "px";
    panel.style.top  = Math.max(8, y) + "px";
}

async function _showFor(node, slotIndex, mouseX, mouseY) {
    const panel = _ensurePanel();
    _positionPanel(panel, mouseX, mouseY);
    panel.classList.add("visible");
    const title = node.title || node.type || "Node";
    const nodeId = String(node.id);

    // Show loading
    panel.innerHTML = `
        <div class="ti-header">
            <span class="ti-title">🔬 ${_esc(title)} <span style="color:var(--c2c-overlay0);font-weight:400;">#${_esc(nodeId)}</span></span>
            <button class="ti-close">×</button>
        </div>
        <div class="ti-empty">Loading snapshot…</div>
    `;
    panel.querySelector(".ti-close")?.addEventListener("click", () => {
        panel.classList.remove("visible");
    });

    try {
        const url = `/mec/tensor_snapshot?node_id=${encodeURIComponent(nodeId)}` +
                    (slotIndex !== null ? `&slot=${slotIndex}` : "");
        const resp = await fetch(url);
        const json = await resp.json();
        if (json.success) {
            _renderPanel(panel, title, nodeId, json.data);
        } else {
            _renderPanel(panel, title, nodeId, null);
        }
    } catch (e) {
        console.warn("[MEC.TensorInspector] fetch failed:", e);
        _renderPanel(panel, title, nodeId, null);
    }
}

function _hookCanvasContextMenu() {
    // Augment node context menu with an "Inspect tensor" item.
    const origGetMenu = LGraphCanvas.prototype.getCanvasMenuOptions;
    const origGetNodeMenu = LGraphCanvas.prototype.getNodeMenuOptions;
    if (origGetNodeMenu && !origGetNodeMenu._mecPatched) {
        LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
            const opts = origGetNodeMenu.call(this, node);
            opts.push(null); // separator
            opts.push({
                content: "🔬 Inspect tensor outputs",
                callback: () => {
                    const rect = this.canvas.getBoundingClientRect();
                    _showFor(
                        node,
                        null,
                        rect.left + rect.width / 2,
                        rect.top + rect.height / 2,
                    );
                },
            });
            return opts;
        };
        LGraphCanvas.prototype.getNodeMenuOptions._mecPatched = true;
    }
    void origGetMenu;  // keep reference, no-op
}

app.registerExtension({
    name: "C2C.TensorInspector",
    settings: [
        {
            id: "c2c.tensor_inspector.enabled",
            name: "Tensor Inspector: enabled",
            tooltip: "Right-click a node → 'Inspect tensor outputs' to see shape/dtype/stats.",
            type: "boolean",
            default: true,
        },
    ],
    async setup() {
        _injectStyle();
        _ensurePanel();

        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("c2c.tensor_inspector.enabled", true); }
            catch { return true; }
        })();
        if (!enabled) return;

        _hookCanvasContextMenu();
        console.log("[MEC.TensorInspector] Loaded — right-click a node to inspect outputs.");
    },
});
