/**
 * mec_flamegraph.js — Phase 3: Execution Flame Graph
 *
 * After a prompt execution completes, fetch /mec/diagnostics/flamegraph and
 * render a horizontal bar chart of per-node elapsed time. Surfaced as a
 * floating panel that the user can toggle on/off.
 *
 * Settings:
 *   mec.flamegraph.enabled       — bool (default true)
 *   mec.flamegraph.auto_show     — bool (default false) → auto-open after run
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE_ID  = "mec-flamegraph-style";
const PANEL_ID  = "mec-flamegraph-panel";
const BTN_ID    = "mec-flamegraph-btn";

let _isOpen = false;
let _lastData = null;

function _injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
#${BTN_ID} {
    position: fixed;
    bottom: 60px;
    right: 16px;
    z-index: 99998;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: #1e1e2e;
    border: 1px solid #45475a;
    color: #fab387;
    font-size: 18px;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
}
#${BTN_ID}:hover { border-color: #89b4fa; color: #89b4fa; }
#${PANEL_ID} {
    position: fixed;
    bottom: 60px;
    right: 64px;
    z-index: 99998;
    width: 480px;
    max-height: 60vh;
    overflow-y: auto;
    background: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 12px 14px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px;
    color: #cdd6f4;
    box-shadow: 0 6px 24px rgba(0,0,0,0.65);
    display: none;
}
#${PANEL_ID}.visible { display: block; }
#${PANEL_ID} .fg-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
#${PANEL_ID} .fg-title {
    font-size: 14px;
    font-weight: 700;
    color: #fab387;
}
#${PANEL_ID} .fg-meta {
    font-size: 11px;
    color: #6c7086;
}
#${PANEL_ID} .fg-close {
    background: transparent;
    border: none;
    color: #6c7086;
    cursor: pointer;
    font-size: 16px;
    padding: 0 4px;
}
#${PANEL_ID} .fg-row {
    display: grid;
    grid-template-columns: 1fr 60px;
    align-items: center;
    gap: 8px;
    padding: 3px 0;
    border-bottom: 1px solid #313244;
}
#${PANEL_ID} .fg-bar {
    position: relative;
    height: 20px;
    background: #313244;
    border-radius: 3px;
    overflow: hidden;
}
#${PANEL_ID} .fg-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}
#${PANEL_ID} .fg-bar-label {
    position: absolute;
    top: 0;
    left: 6px;
    line-height: 20px;
    font-size: 11px;
    color: #cdd6f4;
    white-space: nowrap;
    text-shadow: 0 0 3px rgba(0,0,0,0.7);
}
#${PANEL_ID} .fg-time {
    font-size: 11px;
    color: #a6e3a1;
    text-align: right;
    font-variant-numeric: tabular-nums;
}
#${PANEL_ID} .fg-row.fg-error .fg-bar-fill { background: #f38ba8 !important; }
#${PANEL_ID} .fg-row.fg-error .fg-time     { color: #f38ba8; }
#${PANEL_ID} .fg-empty {
    color: #6c7086;
    font-style: italic;
    padding: 12px 0;
    text-align: center;
}
#${PANEL_ID} .fg-actions {
    display: flex;
    gap: 6px;
    margin-top: 8px;
}
#${PANEL_ID} .fg-actions button {
    background: #313244;
    border: 1px solid #45475a;
    color: #cdd6f4;
    border-radius: 4px;
    padding: 4px 10px;
    cursor: pointer;
    font-size: 11px;
}
#${PANEL_ID} .fg-actions button:hover { border-color: #89b4fa; color: #89b4fa; }
    `.trim();
    document.head.appendChild(style);
}

function _heatColor(pct) {
    // pct in 0..1 → green to red
    const r = Math.round(166 + (243 - 166) * pct);  // a6 → f3
    const g = Math.round(227 + (139 - 227) * pct);  // e3 → 8b
    const b = Math.round(161 + (168 - 161) * pct);  // a1 → a8
    return `rgb(${r},${g},${b})`;
}

function _esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function _renderPanel(data) {
    const panel = document.getElementById(PANEL_ID);
    if (!panel) return;

    if (!data || !data.rows || data.rows.length === 0) {
        panel.innerHTML = `
            <div class="fg-header">
                <span class="fg-title">⏱ Execution Flame Graph</span>
                <button class="fg-close" title="Close">×</button>
            </div>
            <div class="fg-empty">No completed runs yet — queue a prompt first.</div>
            <div class="fg-actions"><button class="fg-refresh">Refresh</button></div>
        `;
    } else {
        const maxMs = data.rows[0].elapsed_ms || 1;
        const rowsHtml = data.rows.map(r => {
            const pct = Math.min(1, r.elapsed_ms / maxMs);
            const color = r.error ? "#f38ba8" : _heatColor(pct);
            const label = `${_esc(r.node_class)} (#${_esc(r.node_id)})`;
            return `
                <div class="fg-row ${r.error ? "fg-error" : ""}">
                    <div class="fg-bar" title="${r.error ? "Errored: " + _esc(r.exc_type || "") : ""}">
                        <div class="fg-bar-fill" style="width:${(pct * 100).toFixed(1)}%;background:${color};"></div>
                        <div class="fg-bar-label">${label}</div>
                    </div>
                    <div class="fg-time">${r.elapsed_ms.toFixed(0)} ms</div>
                </div>`;
        }).join("");

        panel.innerHTML = `
            <div class="fg-header">
                <div>
                    <div class="fg-title">⏱ Execution Flame Graph</div>
                    <div class="fg-meta">${data.node_count} nodes · total ${data.total_ms.toFixed(0)} ms${
                        data.prompt_id ? ` · prompt ${_esc(String(data.prompt_id).slice(0, 8))}…` : ""
                    }</div>
                </div>
                <button class="fg-close" title="Close">×</button>
            </div>
            <div class="fg-rows">${rowsHtml}</div>
            <div class="fg-actions">
                <button class="fg-refresh">Refresh</button>
                <button class="fg-copy">Copy CSV</button>
            </div>
        `;
    }

    panel.querySelector(".fg-close")?.addEventListener("click", _closePanel);
    panel.querySelector(".fg-refresh")?.addEventListener("click", _refresh);
    panel.querySelector(".fg-copy")?.addEventListener("click", _copyCsv);
}

function _copyCsv() {
    if (!_lastData || !_lastData.rows) return;
    const header = "node_id,node_class,elapsed_ms,cpu_ms,vram_delta_mb,vram_peak_mb,error";
    const lines = _lastData.rows.map(r =>
        `${r.node_id},${r.node_class},${r.elapsed_ms},${r.cpu_ms},${r.vram_delta_mb},${r.vram_peak_mb},${r.error}`
    );
    const csv = [header, ...lines].join("\n");
    navigator.clipboard?.writeText(csv).then(() => {
        console.log("[MEC.FlameGraph] CSV copied to clipboard.");
    });
}

async function _refresh() {
    try {
        const resp = await fetch("/mec/diagnostics/flamegraph?limit=50");
        const json = await resp.json();
        if (json.success) {
            _lastData = json.data;
            _renderPanel(json.data);
        }
    } catch (e) {
        console.warn("[MEC.FlameGraph] fetch failed:", e);
    }
}

function _openPanel() {
    _isOpen = true;
    const panel = document.getElementById(PANEL_ID);
    panel?.classList.add("visible");
    _refresh();
}

function _closePanel() {
    _isOpen = false;
    document.getElementById(PANEL_ID)?.classList.remove("visible");
}

function _togglePanel() {
    _isOpen ? _closePanel() : _openPanel();
}

function _ensureUi() {
    if (!document.getElementById(BTN_ID)) {
        const btn = document.createElement("button");
        btn.id = BTN_ID;
        btn.title = "Execution Flame Graph (per-node timing)";
        btn.textContent = "⏱";
        btn.addEventListener("click", _togglePanel);
        document.body.appendChild(btn);
    }
    if (!document.getElementById(PANEL_ID)) {
        const panel = document.createElement("div");
        panel.id = PANEL_ID;
        document.body.appendChild(panel);
    }
}

app.registerExtension({
    name: "MEC.FlameGraph",
    settings: [
        {
            id: "mec.flamegraph.enabled",
            name: "Flame Graph: enabled",
            tooltip: "Show the ⏱ button for the execution flame graph.",
            type: "boolean",
            defaultValue: true,
            onChange: (v) => {
                const btn = document.getElementById(BTN_ID);
                if (btn) btn.style.display = v ? "flex" : "none";
            },
        },
        {
            id: "mec.flamegraph.auto_show",
            name: "Flame Graph: auto-open after run",
            tooltip: "Automatically open the flame graph panel when a prompt finishes.",
            type: "boolean",
            defaultValue: false,
        },
    ],
    async setup() {
        _injectStyle();
        _ensureUi();

        const enabled = (() => {
            try { return app.ui.settings.getSettingValue("mec.flamegraph.enabled", true); }
            catch { return true; }
        })();
        const btn = document.getElementById(BTN_ID);
        if (btn) btn.style.display = enabled ? "flex" : "none";

        // Listen for prompt execution-done events
        api.addEventListener("execution_success", () => {
            const auto = (() => {
                try { return app.ui.settings.getSettingValue("mec.flamegraph.auto_show", false); }
                catch { return false; }
            })();
            if (_isOpen) {
                // Already open → silently refresh
                _refresh();
            } else if (auto) {
                _openPanel();
            }
        });

        // Also refresh on execution_error so user can see what crashed
        api.addEventListener("execution_error", () => {
            if (_isOpen) _refresh();
        });

        console.log("[MEC.FlameGraph] Loaded — click ⏱ button to view per-node timings.");
    },
});
