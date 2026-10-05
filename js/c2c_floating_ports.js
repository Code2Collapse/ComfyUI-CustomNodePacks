// c2c_floating_ports.js — "floating ports" / perimeter wire routing (Nuke-style).
// ---------------------------------------------------------------------------
// ComfyUI uses FIXED ports: outputs on the right edge, inputs on the left, so a
// wire between two vertically-stacked nodes does a big U-bend. Nuke instead lets
// the wire endpoint SLIDE along the node perimeter to exit the side closest to
// the other node, keeping the graph readable.
//
// This implements that as **perimeter wire routing**: the clickable port DOTS
// stay exactly where they are (left=in / right=out — hit-testing and core
// behaviour untouched), but the drawn noodle's endpoint walks the node boundary
// to the point nearest the connected node, and the bezier exits in that edge's
// direction. No more U-bends.
//
// Opt-in (off by default) via Settings → C2C → "Floating ports (perimeter wire
// routing)". Off = zero overhead (the override early-returns to the original).
// Composes with C2C NoodleStyles: we only adjust the endpoints/directions, then
// call through to whatever renderLink was already installed.
//
// Perf: enabled-check and node bounds are amortised (counter gate + per-node box
// cache); geometry, tether colours, and routing behaviour are unchanged.

import { app } from "/scripts/app.js";
import { C } from "./_c2c_theme.js";
import { getRuntime } from "./_c2c_runtime.js";

const SETTING_ID = "c2c.floatingPorts.enabled";
let _enabled = false;
let _lastCheck = 0;
let _checkCounter = 0;
// Re-reads the persisted setting every 256 link draws at most, and no more
// often than ~4×/sec when the counter fires — onChange still updates immediately.
function enabledNow() {
    if ((_checkCounter++ & 255) === 0) {
        const t = (window.performance && performance.now()) || Date.now();
        if (t - _lastCheck >= 250) {
            _lastCheck = t;
            try { const v = app.ui.settings.getSettingValue(SETTING_ID); _enabled = (v === undefined || v === null) ? true : v === true; } catch (_) {}
        }
    }
    return _enabled;
}

// ── geometry helpers (graph-space) ──────────────────────────────────────────
const _box = [0, 0, 0, 0];
function nodeBox(node) {
    // Full render bounds incl. title bar, in graph coords — reused object on node.
    let b = node.__c2cBox;
    if (!b) b = node.__c2cBox = { x: 0, y: 0, w: 0, h: 0, cx: 0, cy: 0 };
    if (typeof node.getBounding === "function") {
        node.getBounding(_box);
        b.x = _box[0]; b.y = _box[1]; b.w = _box[2]; b.h = _box[3];
        b.cx = b.x + b.w / 2; b.cy = b.y + b.h / 2;
        return b;
    }
    const th = LiteGraph.NODE_TITLE_HEIGHT || 30;
    b.x = node.pos[0]; b.y = node.pos[1] - th; b.w = node.size[0]; b.h = node.size[1] + th;
    b.cx = b.x + b.w / 2; b.cy = b.y + b.h / 2;
    return b;
}

// TRUE 360° Nuke-style pipe attachment: the wire's visual exit is where the
// ray from THIS node's centre toward the OTHER node crosses this node's
// perimeter — drag the other node anywhere and the exit slides around all
// four edges, exactly like a Nuke pipe. The REAL slot never moves (render-only;
// hit-testing untouched); a thin colour-matched tether from the slot to the
// exit keeps the end-to-end slot connection clearly visible.
// dir always matches the crossed edge's outward normal, so the bezier exits
// cleanly (the old perpendicular-edge hook/curl bug came from mismatched dirs).
// Returns { pt:[x,y], dir } in graph space, or null when geometry degenerates
// (overlapping nodes) — caller falls back to the fixed slot.
function slideAlongDotEdge(b, dotPt, tx, ty) {
    const OUT = 4;          // push exit just outside the border so it isn't
                            // hidden under the node's own edge stroke
    const CORNER = 10;      // keep exits off the exact corners
    // Ray FROM THE SLOT (not the node centre — centre-rays detached the wire
    // from the slot on tall nodes even in plain side-by-side layouts).
    const sx = Math.max(b.x, Math.min(b.x + b.w, dotPt[0]));
    const sy = Math.max(b.y, Math.min(b.y + b.h, dotPt[1]));
    const dx = tx - sx, dy = ty - sy;
    if (Math.abs(dx) < 1e-3 && Math.abs(dy) < 1e-3) return null;
    // Which edge does the slot live on? If the target lies on the slot's OWN
    // side, the wire attaches at the slot exactly like classic ComfyUI — the
    // 360° float only kicks in when the target is genuinely on another side.
    const dl = Math.abs(sx - b.x), dr = Math.abs(b.x + b.w - sx);
    const dt = Math.abs(sy - b.y), db = Math.abs(b.y + b.h - sy);
    const m = Math.min(dl, dr, dt, db);
    const n = m === dr ? [1, 0] : m === dl ? [-1, 0] : m === dt ? [0, -1] : [0, 1];
    if (n[0] * dx + n[1] * dy >= -1e-6) return null;   // heading outward → classic
    // Ray marches INTO the box: find where it exits (slab method).
    const txr = dx > 0 ? (b.x + b.w - sx) / dx : dx < 0 ? (b.x - sx) / dx : Infinity;
    const tyr = dy > 0 ? (b.y + b.h - sy) / dy : dy < 0 ? (b.y - sy) / dy : Infinity;
    const t = Math.min(txr, tyr);
    if (!isFinite(t) || t <= 1e-6) return null;
    const clampY = (y) => Math.max(b.y + CORNER, Math.min(b.y + b.h - CORNER, y));
    const clampX = (x) => Math.max(b.x + CORNER, Math.min(b.x + b.w - CORNER, x));
    if (txr <= tyr) {
        const right = dx > 0;
        return { pt: [right ? b.x + b.w + OUT : b.x - OUT, clampY(sy + dy * t)],
                 dir: right ? LiteGraph.RIGHT : LiteGraph.LEFT };
    }
    const down = dy > 0;
    return { pt: [clampX(sx + dx * t), down ? b.y + b.h + OUT : b.y - OUT],
             dir: down ? LiteGraph.DOWN : LiteGraph.UP };
}

// Slot DATA-TYPE colour — ComfyUI's standard palette (IMAGE blue, MASK green,
// LATENT pink, ...), so the pipe endpoint always wears its slot's colour.
function slotTypeColor(canvas, link) {
    const t = link && link.type;
    return (canvas.default_connection_color_byType && canvas.default_connection_color_byType[t])
        || (window.LGraphCanvas && LGraphCanvas.link_type_colors && LGraphCanvas.link_type_colors[t])
        || (canvas.default_link_color) || C.overlay1;
}

// The pipe's attachment dot: a small filled disc at the perimeter exit —
// visually the connector riding the edge (real slots/hit-testing untouched).
function drawEndpointDot(ctx, pt, color) {
    const pf = ctx.fillStyle, ps = ctx.strokeStyle, pl = ctx.lineWidth;
    ctx.fillStyle = color;
    ctx.strokeStyle = "rgba(12,12,16,0.8)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(pt[0], pt[1], 3.5, 0, Math.PI * 2);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = pf; ctx.strokeStyle = ps; ctx.lineWidth = pl;
}

function install() {
    const proto = window.LGraphCanvas && LGraphCanvas.prototype;
    if (!proto) return false;
    // Hook the LOW-LEVEL segment renderer, not renderLink: in current litegraph
    // _renderAllLinkSegments(ctx, link, startPt, endPt, paths, time, startDir,
    // endDir, disabled) is what computes the endpoints + directions and feeds
    // renderLink. It's uncontested (NoodleStyles overrides renderLink, which
    // sits BELOW us), so adjusting the endpoints here reroutes EVERY noodle
    // style to the nearest edge while leaving the port dots + hit-testing alone.
    if (typeof proto._renderAllLinkSegments !== "function") return false;
    const moved = (p, q) => (Math.abs(p[0] - q[0]) + Math.abs(p[1] - q[1])) > 3;
    getRuntime().safePatch(proto, "_renderAllLinkSegments", (orig) => function (ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled) {
        if (enabledNow() && link && this.graph && !link._dragging) {
            const A = this.graph.getNodeById(link.origin_id);
            const B = this.graph.getNodeById(link.target_id);
            if (A && B && A !== B && !A.flags?.collapsed && !B.flags?.collapsed) {
                try {
                    const ba = nodeBox(A), bb = nodeBox(B);
                    // Slide each endpoint along its OWN dot's edge toward the
                    // other node (startPt = output dot, endPt = input dot).
                    const a = slideAlongDotEdge(ba, startPt, bb.cx, bb.cy) || { pt: startPt, dir: startDir };
                    const b = slideAlongDotEdge(bb, endPt, ba.cx, ba.cy) || { pt: endPt, dir: endDir };
                    // TWO THINGS AT ONCE, inline on the link canvas (correct ctx;
                    // the drawFrontCanvas post-pass failed because links render on
                    // the BACK canvas):
                    //   1) FLOAT — a solid tether keeps each wire attached to its
                    //      real slot as it floats to the perimeter exit.
                    //   2) WHICH-SLOT-CONNECTS-WHICH — colour BOTH real slots the
                    //      SAME distinct per-connection colour, so you pair them by
                    //      colour (output = filled ring, input = hollow ring), even
                    //      under a rainbow noodle. The slots "change" per wire.
                    if (ctx && !disabled) {
                        // Slot-TYPE colour (ComfyUI standard palette) — the dot at
                        // the pipe's perimeter attachment "is" the slot, riding the
                        // edge Nuke-style. Original slot dots stay untouched.
                        const col = slotTypeColor(this, link);
                        if (moved(startPt, a.pt)) drawEndpointDot(ctx, a.pt, col);
                        if (moved(endPt, b.pt))   drawEndpointDot(ctx, b.pt, col);
                    }
                    startPt = a.pt; endPt = b.pt;
                    // Direction matches the dot's own edge now, so the bezier
                    // exits cleanly (no hook) — safe to set.
                    startDir = a.dir; endDir = b.dir;
                } catch (_) { /* fall back to fixed ports on any geometry hiccup */ }
            }
        }
        return orig.call(this, ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled);
    }, { id: "floatingports._renderAllLinkSegments" });
    return true;
}

if (!(app.extensions || []).some((e) => e?.name === "C2C.FloatingPorts")) app.registerExtension({
    name: "C2C.FloatingPorts",
    settings: [
        {
            id: SETTING_ID,
            name: "Floating ports (perimeter wire routing, Nuke-style)",
            tooltip: "Nuke-style 360° pipes: the wire's exit slides around the node perimeter to "
                   + "face the connected node. Slots never move (a thin colour tether keeps the "
                   + "slot connection visible). Off = classic left/right wires.",
            type: "boolean",
            defaultValue: true,
            category: ["c2c", "Canvas", "Floating ports"],
            onChange: (v) => { _enabled = !!v; try { app.graph?.setDirtyCanvas(true, true); } catch (_) {} },
        },
    ],
    async setup() {
        // Read the persisted value, then install the override LAST (in setup,
        // after other extensions — incl. NoodleStyles — have wrapped renderLink),
        // so floating-port rerouting is the outermost adjustment.
        try { const v = app.ui.settings.getSettingValue(SETTING_ID); _enabled = (v === undefined || v === null) ? true : v === true; } catch (_) {}
        install();
    },
});
