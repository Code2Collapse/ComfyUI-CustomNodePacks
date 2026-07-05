// c2c_connected_highlight.js — trace which slot connects to which.
// ---------------------------------------------------------------------------
// HOVER or SELECT a node and every wire running to/from it lights up, each in
// its OWN distinct colour, with a matching ring on BOTH of its slots. So you can
// follow "this slot → this coloured wire → that slot" at a glance — even while
// you drag the node around, and even when a fancy custom noodle (rainbow, tron…)
// hides the wire's normal type-colour. Hover is the key: no click needed.
//
// Hooks the low-level link renderer (`_renderAllLinkSegments`, the uncontested
// seam the floating-ports feature also uses) and draws a coloured underlay +
// slot rings beneath any "active" link, then calls through to the normal noodle
// render (composes with NoodleStyles). Idle cost is ~zero (nothing active).
import { app } from "/scripts/app.js";

const SETTING = "c2c.connectedHighlight.enabled";
let _enabled = true;
let _selIds = new Set();
let _lastCheck = 0;

const HALO = "rgba(120, 205, 255, 0.9)";   // fallback cyan

// Distinct, high-separation palette. Each wire gets a STABLE colour derived from
// its link id (O(1), no per-selection map to grow/evict) so a given wire keeps
// the same colour across hover/select and every frame.
const PALETTE = [
    "#ff4d6d", "#4dd2ff", "#ffd24d", "#7CFC00", "#c77dff",
    "#ff9f1c", "#2ec4b6", "#ff6ec7", "#9bf6ff", "#f15bb5",
    "#80ed99", "#fee440",
];
function _linkColor(link) {
    const id = (link && (link.id | 0)) || 0;
    return PALETTE[((id % PALETTE.length) + PALETTE.length) % PALETTE.length] || HALO;
}

// A link is "active" when either endpoint node is SELECTED or HOVERED.
function _linkActive(canvas, link) {
    if (_selIds.size && (_selIds.has(link.origin_id) || _selIds.has(link.target_id))) return true;
    const over = canvas && canvas.node_over;
    return !!(over && over.id != null && (over.id === link.origin_id || over.id === link.target_id));
}

function enabledNow() {
    const t = (window.performance && performance.now()) || Date.now();
    if (t - _lastCheck > 250) { _lastCheck = t; try { _enabled = app.ui.settings.getSettingValue(SETTING, true) !== false; } catch (_) {} }
    return _enabled;
}

function refreshSelection(canvas) {
    _selIds = new Set();
    const sel = canvas?.selected_nodes;
    if (sel) for (const k in sel) { const n = sel[k]; if (n && n.id != null) _selIds.add(n.id); }
    try { canvas?.setDirty(true, true); } catch (_) {}
}

function install() {
    const proto = window.LGraphCanvas && LGraphCanvas.prototype;
    if (!proto || proto.__c2cConnHighlight) return false;
    if (typeof proto._renderAllLinkSegments !== "function") return false;
    const orig = proto._renderAllLinkSegments;
    proto._renderAllLinkSegments = function (ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled) {
        if (enabledNow() && link && _linkActive(this, link)) {
            try {
                const a = startPt, b = endPt;
                const col = _linkColor(link);
                ctx.save();
                // 1) subtle coloured glow UNDER the wire (slimmed down — was too
                //    thick/big). Just enough to read, not a fat halo.
                ctx.strokeStyle = col;
                ctx.lineWidth = (this.connections_width || 3) + 3;
                ctx.lineCap = "round";
                ctx.shadowColor = col;
                ctx.shadowBlur = 7;
                ctx.globalAlpha = 0.8;
                const cx = Math.max(40, Math.abs(b[0] - a[0]) * 0.5);
                ctx.beginPath();
                ctx.moveTo(a[0], a[1]);
                ctx.bezierCurveTo(a[0] + cx, a[1], b[0] - cx, b[1], b[0], b[1]);
                ctx.stroke();
                // 2) small matching ring on BOTH slots so you can pair them by
                //    colour. Output = filled, input = hollow.
                ctx.shadowBlur = 5;
                ctx.globalAlpha = 1;
                ctx.lineWidth = 2;
                ctx.fillStyle = col;
                ctx.beginPath(); ctx.arc(a[0], a[1], 3.5, 0, Math.PI * 2); ctx.fill();
                ctx.beginPath(); ctx.arc(b[0], b[1], 4, 0, Math.PI * 2); ctx.stroke();
                ctx.restore();
            } catch (_) { /* never break link rendering */ }
        }
        return orig.call(this, ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled);
    };
    proto.__c2cConnHighlight = true;
    return true;
}

if (!(app.extensions || []).some((e) => e?.name === "C2C.ConnectedHighlight")) app.registerExtension({
    name: "C2C.ConnectedHighlight",
    settings: [
        {
            id: SETTING,
            name: "Trace connections on hover / select",
            tooltip: "Hover (or select) a node and the wires running to/from it light up, each in "
                   + "its own colour with a matching ring on both slots — so you can see exactly "
                   + "which slot connects to which, even under a custom noodle.",
            type: "boolean",
            defaultValue: true,
            category: ["c2c", "Canvas", "Connection tracing"],
            onChange: (v) => { _enabled = !!v; try { app.graph?.setDirtyCanvas(true, true); } catch (_) {} },
        },
    ],
    async setup() {
        try { _enabled = app.ui.settings.getSettingValue(SETTING, true) !== false; } catch (_) {}
        install();
        const canvas = app.canvas;
        if (canvas) {
            const orig = canvas.onSelectionChange;
            canvas.onSelectionChange = function (...a) {
                const r = orig ? orig.apply(this, a) : undefined;
                refreshSelection(this);
                return r;
            };
            // Some selection paths (marquee, deselect) don't fire onSelectionChange
            // reliably; refresh on each processMouseUp as a cheap safety net.
            const omu = canvas.processMouseUp;
            if (typeof omu === "function") {
                canvas.processMouseUp = function (e) {
                    const r = omu.apply(this, arguments);
                    try { refreshSelection(this); } catch (_) {}
                    return r;
                };
            }
            // HOVER tracing: when the node under the cursor changes, force ONE
            // redraw so its connections light up / clear immediately. Cheap — a
            // single dirty flag only on the frame the hovered node changes.
            const omm = canvas.processMouseMove;
            if (typeof omm === "function") {
                let _lastOver = null;
                canvas.processMouseMove = function (e) {
                    const r = omm.apply(this, arguments);
                    try {
                        const id = this.node_over ? this.node_over.id : null;
                        if (id !== _lastOver) { _lastOver = id; this.setDirty(false, true); }
                    } catch (_) {}
                    return r;
                };
            }
        }
    },
});
