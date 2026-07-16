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

// Highlight colour = the SLOT's data-type colour (IMAGE blue, MASK green,
// LATENT pink…) — the light that shines on the wire matches the slot it
// belongs to, exactly. (Was a random per-link palette; user spec 2026-07-15.)
function _linkColor(canvas, link) {
    const t = link && link.type;
    return (canvas && canvas.default_connection_color_byType && canvas.default_connection_color_byType[t])
        || (window.LGraphCanvas && LGraphCanvas.link_type_colors && LGraphCanvas.link_type_colors[t])
        || (canvas && canvas.default_link_color) || HALO;
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

// Re-entry guard: the glow pass re-invokes the FULL patched render chain (so
// the glow lands on the EXACT path every skin/shape/floating-exit draws) and
// this flag makes the inner invocation a plain draw.
let _rerender = false;

function install() {
    const proto = window.LGraphCanvas && LGraphCanvas.prototype;
    if (!proto || proto.__c2cConnHighlight) return false;
    if (typeof proto._renderAllLinkSegments !== "function") return false;
    const orig = proto._renderAllLinkSegments;
    proto._renderAllLinkSegments = function (ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled) {
        if (_rerender || !enabledNow() || !link) {
            return orig.call(this, ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled);
        }
        const anyActive = _selIds.size > 0 || !!(this.node_over);
        const active = anyActive && _linkActive(this, link);
        if (!anyActive) {
            return orig.call(this, ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled);
        }
        if (!active) {
            // DIM every wire not connected to the selected/hovered node, so the
            // lit ones pop. Alpha-wrap the normal draw — exact path, any skin.
            // finally-restore: an inner throw must never leak the 0.3 alpha.
            ctx.save();
            ctx.globalAlpha *= 0.3;
            try {
                return orig.call(this, ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled);
            } finally {
                ctx.restore();
            }
        }
        // ACTIVE wire: normal draw first…
        const r = orig.call(this, ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled);
        try {
            const col = _linkColor(this, link);
            // …then the LIGHT: re-render the very same wire (endpoints already
            // floated; core → NoodleStyles skin) with a slot-coloured shadow —
            // the glow hugs the exact noodle path, any skin or shape.
            _rerender = true;
            ctx.save();
            ctx.shadowColor = col;
            ctx.shadowBlur = 14;
            ctx.globalAlpha = 0.95;
            const _w = this.connections_width;
            this.connections_width = (_w || 3) + 1;   // slightly fatter re-stroke → halo shows
            try {
                // two stacked shadow passes = a clearly visible slot-coloured light
                orig.call(this, ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled);
                orig.call(this, ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled);
            } finally {
                // a throw must never leak shadow/alpha state, a fattened
                // connections_width, or a stuck _rerender flag
                this.connections_width = _w;
                ctx.restore();
                _rerender = false;
            }
            // slot pairing cue: slot-coloured dot on the output end, ring on the
            // input end — same colour as the slot itself.
            ctx.save();
            ctx.shadowColor = col; ctx.shadowBlur = 6;
            ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(startPt[0], startPt[1], 3.5, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.arc(endPt[0], endPt[1], 4, 0, Math.PI * 2); ctx.stroke();
            ctx.restore();
        } catch (_) { _rerender = false; /* never break link rendering */ }
        return r;
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
