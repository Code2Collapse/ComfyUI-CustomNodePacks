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

import { app } from "/scripts/app.js";

const SETTING_ID = "c2c.floatingPorts.enabled";
let _enabled = false;
let _lastCheck = 0;
// Robust enabled-check: re-reads the persisted setting at most ~4x/sec, so the
// toggle works even if a programmatic setSettingValue doesn't fire onChange.
function enabledNow() {
    const t = (window.performance && performance.now()) || Date.now();
    if (t - _lastCheck > 250) {
        _lastCheck = t;
        try { _enabled = app.ui.settings.getSettingValue(SETTING_ID, false) === true; } catch (_) {}
    }
    return _enabled;
}

// ── geometry helpers (graph-space) ──────────────────────────────────────────
const _box = [0, 0, 0, 0];
function nodeBox(node) {
    // Full render bounds incl. title bar, in graph coords.
    if (typeof node.getBounding === "function") {
        node.getBounding(_box);
        return { x: _box[0], y: _box[1], w: _box[2], h: _box[3],
                 cx: _box[0] + _box[2] / 2, cy: _box[1] + _box[3] / 2 };
    }
    const th = LiteGraph.NODE_TITLE_HEIGHT || 30;
    const x = node.pos[0], y = node.pos[1] - th, w = node.size[0], h = node.size[1] + th;
    return { x, y, w, h, cx: x + w / 2, cy: y + h / 2 };
}

// Float ONE endpoint by SLIDING its port dot along the SAME edge it already
// lives on, toward the other node. Keeping the endpoint on the dot's own edge
// (right for outputs, left for inputs) means its exit DIRECTION still matches
// the edge, so the noodle bezier stays clean — jumping to a *perpendicular*
// edge made the horizontal-offset bezier hook/curl. The float stays right next
// to its dot, so you can always see which dot a wire belongs to.
// Returns { pt:[x,y], dir } in graph space.
function slideAlongDotEdge(b, dotPt, tx, ty) {
    const margin = 12;
    // The exit floats toward the target but is CAPPED to stay right next to the
    // real slot, so the wire always reads as firmly attached (short connector)
    // while still nudging Nuke-style toward the connected node.
    const BIAS = 0.5;
    const MAXSLIDE = 12;  // keep the exit RIGHT ON the slot (tiny float only) so
                          // the wire stays visibly attached — "slot to slot does
                          // not change", it just floats a touch toward the target.
    const OUT = 4;        // push exit just OUTSIDE the edge so it isn't hidden
                          // under the node's own border.
    const cap = (v, c) => Math.max(-c, Math.min(c, v));
    const dl = Math.abs(dotPt[0] - b.x);            // dist to left edge
    const dr = Math.abs(dotPt[0] - (b.x + b.w));    // dist to right edge
    const dt = Math.abs(dotPt[1] - b.y);            // dist to top edge
    const db = Math.abs(dotPt[1] - (b.y + b.h));    // dist to bottom edge
    const nearestV = Math.min(dl, dr);              // nearest left/right edge
    const nearestH = Math.min(dt, db);              // nearest top/bottom edge
    if (nearestV <= nearestH) {
        // Dot is on a LEFT/RIGHT edge → slide vertically toward the target.
        const onLeft = dl <= dr;
        const x = onLeft ? b.x - OUT : b.x + b.w + OUT;
        const y = dotPt[1] + cap((ty - dotPt[1]) * BIAS, MAXSLIDE);
        return { pt: [x, y], dir: onLeft ? LiteGraph.LEFT : LiteGraph.RIGHT };
    }
    // Dot is on a TOP/BOTTOM edge → slide horizontally toward the target.
    const onTop = dt <= db;
    const y = onTop ? b.y - OUT : b.y + b.h + OUT;
    const x = dotPt[0] + cap((tx - dotPt[0]) * BIAS, MAXSLIDE);
    return { pt: [x, y], dir: onTop ? LiteGraph.UP : LiteGraph.DOWN };
}

// A DISTINCT colour per connection (stable hash of the link id). Both slots of a
// wire get this SAME colour, so you can pair them at a glance — "this slot
// connects to that slot" (match the colour) — even under a rainbow noodle. Two
// wires from the same output get DIFFERENT colours (different ids).
const PALETTE = [
    "#ff5d73", "#4dd2ff", "#ffd24d", "#8cff66", "#c77dff",
    "#ff9f1c", "#2ec4b6", "#ff77c8", "#7cc4ff", "#f15bb5",
    "#a0f0a0", "#ffe14d",
];
function distinctLinkColor(link) {
    const id = (link && (link.id | 0)) || 0;
    return PALETTE[((id % PALETTE.length) + PALETTE.length) % PALETTE.length];
}

// Draw a colour ring ON a real slot (hit-testing untouched). Output = filled,
// input = hollow, so direction reads too.
function drawSlotRing(ctx, pt, color, filled) {
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 9;
    ctx.globalAlpha = 1;
    // bright ring so the slot clearly "wears" its connection colour
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(pt[0], pt[1], filled ? 6 : 6.5, 0, Math.PI * 2); ctx.stroke();
    if (filled) {   // output = solid centre, input = hollow → direction reads too
        ctx.shadowBlur = 0;
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(pt[0], pt[1], 3, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
}

// Draw the "which-dot" leader: a short stub from the REAL port dot (where the
// slot actually is, hit-testing untouched) to the floating perimeter exit, with
// a filled dot at the slot end. This is the visual that answers the user's
// "show which dot it's connected to".
// Draw ONE "which-dot" leader: a short stub from the real port dot to the
// floating perimeter exit, a filled anchor dot on the real port, and a hollow
// ring at the floating exit. Drawn in a post-pass ON TOP of nodes (see the
// drawFrontCanvas hook) — LiteGraph renders links UNDER node bodies, so an
// inline leader from a right-edge dot to a bottom-edge exit got occluded by the
// node and you couldn't see which dot the floating wire belonged to.
// A SOLID, colour-matched connector from the real slot to the floating exit —
// so the pipe stays visibly ATTACHED to its slot while it floats (Nuke-style).
function drawTether(ctx, fromSlot, toExit, color) {
    ctx.save();
    ctx.lineCap = "round";
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 5;
    ctx.globalAlpha = 1;
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    ctx.moveTo(fromSlot[0], fromSlot[1]);
    ctx.lineTo(toExit[0], toExit[1]);
    ctx.stroke();
    ctx.restore();
}

function install() {
    const proto = window.LGraphCanvas && LGraphCanvas.prototype;
    if (!proto || proto.__c2cFloatingPorts) return false;
    // Hook the LOW-LEVEL segment renderer, not renderLink: in current litegraph
    // _renderAllLinkSegments(ctx, link, startPt, endPt, paths, time, startDir,
    // endDir, disabled) is what computes the endpoints + directions and feeds
    // renderLink. It's uncontested (NoodleStyles overrides renderLink, which
    // sits BELOW us), so adjusting the endpoints here reroutes EVERY noodle
    // style to the nearest edge while leaving the port dots + hit-testing alone.
    if (typeof proto._renderAllLinkSegments !== "function") return false;
    const orig = proto._renderAllLinkSegments;
    const moved = (p, q) => (Math.abs(p[0] - q[0]) + Math.abs(p[1] - q[1])) > 3;
    proto._renderAllLinkSegments = function (ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled) {
        if (enabledNow() && link && this.graph && !link._dragging) {
            const A = this.graph.getNodeById(link.origin_id);
            const B = this.graph.getNodeById(link.target_id);
            if (A && B && A !== B && !A.flags?.collapsed && !B.flags?.collapsed) {
                try {
                    const ba = nodeBox(A), bb = nodeBox(B);
                    // Slide each endpoint along its OWN dot's edge toward the
                    // other node (startPt = output dot, endPt = input dot).
                    const a = slideAlongDotEdge(ba, startPt, bb.cx, bb.cy);
                    const b = slideAlongDotEdge(bb, endPt, ba.cx, ba.cy);
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
                        const col = distinctLinkColor(link);
                        drawSlotRing(ctx, startPt, col, true);    // output slot
                        drawSlotRing(ctx, endPt, col, false);     // input slot
                        if (moved(startPt, a.pt)) drawTether(ctx, startPt, a.pt, col);
                        if (moved(endPt, b.pt))   drawTether(ctx, endPt, b.pt, col);
                    }
                    startPt = a.pt; endPt = b.pt;
                    // Direction matches the dot's own edge now, so the bezier
                    // exits cleanly (no hook) — safe to set.
                    startDir = a.dir; endDir = b.dir;
                } catch (_) { /* fall back to fixed ports on any geometry hiccup */ }
            }
        }
        return orig.call(this, ctx, link, startPt, endPt, paths, time, startDir, endDir, disabled);
    };

    proto.__c2cFloatingPorts = true;
    return true;
}

if (!(app.extensions || []).some((e) => e?.name === "C2C.FloatingPorts")) app.registerExtension({
    name: "C2C.FloatingPorts",
    settings: [
        {
            id: SETTING_ID,
            name: "Floating ports (perimeter wire routing, Nuke-style)",
            tooltip: "Wires exit the node edge nearest the connected node instead of always "
                   + "left/right. Port dots stay put; only the noodle reroutes. Off = no change.",
            type: "boolean",
            defaultValue: false,
            category: ["c2c", "Canvas", "Floating ports"],
            onChange: (v) => { _enabled = !!v; try { app.graph?.setDirtyCanvas(true, true); } catch (_) {} },
        },
    ],
    async setup() {
        // Read the persisted value, then install the override LAST (in setup,
        // after other extensions — incl. NoodleStyles — have wrapped renderLink),
        // so floating-port rerouting is the outermost adjustment.
        try { _enabled = app.ui.settings.getSettingValue(SETTING_ID, false) === true; } catch (_) {}
        install();
    },
});
