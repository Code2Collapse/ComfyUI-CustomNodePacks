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
        try { const v = app.ui.settings.getSettingValue(SETTING_ID); _enabled = (v === undefined || v === null) ? true : v === true; } catch (_) {}
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
    const dx = tx - b.cx, dy = ty - b.cy;
    if (Math.abs(dx) < 1e-3 && Math.abs(dy) < 1e-3) return null;
    const hw = b.w / 2, hh = b.h / 2;
    // Ray–box: which edge does (cx,cy)→(tx,ty) cross first?
    const txr = dx !== 0 ? hw / Math.abs(dx) : Infinity;
    const tyr = dy !== 0 ? hh / Math.abs(dy) : Infinity;
    const clampY = (y) => Math.max(b.y + CORNER, Math.min(b.y + b.h - CORNER, y));
    const clampX = (x) => Math.max(b.x + CORNER, Math.min(b.x + b.w - CORNER, x));
    if (txr <= tyr) {
        // crosses a LEFT/RIGHT edge
        const right = dx > 0;
        const x = right ? b.x + b.w + OUT : b.x - OUT;
        const y = clampY(b.cy + dy * txr);
        return { pt: [x, y], dir: right ? LiteGraph.RIGHT : LiteGraph.LEFT };
    }
    // crosses a TOP/BOTTOM edge
    const down = dy > 0;
    const y = down ? b.y + b.h + OUT : b.y - OUT;
    const x = clampX(b.cx + dx * tyr);
    return { pt: [x, y], dir: down ? LiteGraph.DOWN : LiteGraph.UP };
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
    ctx.shadowBlur = 4;
    ctx.globalAlpha = 1;
    // bright ring so the slot clearly "wears" its connection colour
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(pt[0], pt[1], filled ? 5 : 5.5, 0, Math.PI * 2); ctx.stroke();
    if (filled) {   // output = solid centre, input = hollow → direction reads too
        ctx.shadowBlur = 0;
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(pt[0], pt[1], 2.2, 0, Math.PI * 2); ctx.fill();
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
// The tether must never cross the node BODY — links render UNDER nodes, so a
// straight slot→exit line gets occluded (e.g. right-edge slot, bottom exit).
// Instead it hugs the perimeter: slot → its projection on the OUT-offset box →
// around the corner(s), shorter way → exit. Always outside the border, always
// visible, so the wire reads as firmly attached to its real slot.
function drawTether(ctx, b, fromSlot, toExit, color) {
    const OUT = 4;
    const L = b.x - OUT, R = b.x + b.w + OUT, T = b.y - OUT, Bo = b.y + b.h + OUT;
    const W = R - L, H = Bo - T, P = 2 * (W + H);
    // param s (clockwise from top-left) for a point ON the offset box
    const sOf = (x, y) => {
        const dT = Math.abs(y - T), dR = Math.abs(x - R), dB = Math.abs(y - Bo), dL = Math.abs(x - L);
        const m = Math.min(dT, dR, dB, dL);
        if (m === dT) return (Math.max(L, Math.min(R, x)) - L);
        if (m === dR) return W + (Math.max(T, Math.min(Bo, y)) - T);
        if (m === dB) return W + H + (R - Math.max(L, Math.min(R, x)));
        return W + H + W + (Bo - Math.max(T, Math.min(Bo, y)));
    };
    const xyOf = (s) => {
        s = ((s % P) + P) % P;
        if (s <= W) return [L + s, T];
        if (s <= W + H) return [R, T + (s - W)];
        if (s <= W + H + W) return [R - (s - W - H), Bo];
        return [L, Bo - (s - W - H - W)];
    };
    const s1 = sOf(fromSlot[0], fromSlot[1]), s2 = sOf(toExit[0], toExit[1]);
    let d = s2 - s1;
    if (d > P / 2) d -= P; else if (d < -P / 2) d += P;   // shorter way around
    const pts = [fromSlot, xyOf(s1)];
    // corner s-values passed while walking from s1 to s1+d, in walk order
    const hits = [];
    for (const c of [0, W, W + H, W + H + W]) {
        for (const k of [-1, 0, 1]) {                      // handle wrap
            const cs = c + k * P;
            if ((d >= 0 && cs > s1 + 0.5 && cs < s1 + d - 0.5) ||
                (d < 0 && cs < s1 - 0.5 && cs > s1 + d + 0.5)) hits.push(cs);
        }
    }
    hits.sort((a2, b2) => d >= 0 ? a2 - b2 : b2 - a2);
    for (const cs of hits) pts.push(xyOf(cs));
    pts.push(toExit);
    ctx.save();
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 2;
    ctx.globalAlpha = 0.95;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
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
                        const col = distinctLinkColor(link);
                        drawSlotRing(ctx, startPt, col, true);    // output slot
                        drawSlotRing(ctx, endPt, col, false);     // input slot
                        if (moved(startPt, a.pt)) drawTether(ctx, ba, startPt, a.pt, col);
                        if (moved(endPt, b.pt))   drawTether(ctx, bb, endPt, b.pt, col);
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
