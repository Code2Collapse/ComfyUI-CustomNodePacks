// roto_json <-> the spline editor's shape format.
//
// WHY AN ADAPTER RATHER THAN A SECOND EDITOR.
//
// VectorRotoMEC shipped its shapes as a `roto_json` TEXT BOX. Roto is closed
// cubic beziers with per-point tangent handles - nobody types one, so the node
// was unusable as shipped. But this pack already has a 1,428-line spline
// editor with drag, handles, undo, an image backdrop and a decade of small
// fixes in it. Writing a second one would mean two roto editors to keep
// working, and the newer one would be the worse of the two for a year.
//
// The geometry is the SAME - closed cubic beziers with in/out tangents. Only
// the serialisation differs, in three ways that all matter:
//
//   1. HANDLES ARE RELATIVE in the editor, ABSOLUTE in roto_json.
//      Editor: handles[i] = {in:{x:-40,y:0}, out:{x:40,y:0}}, an OFFSET from
//      the point. roto_json: {"in":[x,y], "out":[x,y]}, a position on the
//      canvas. Feeding one to the other without converting does not error -
//      it puts every tangent near the origin and the shape collapses toward
//      the top-left corner, which reads as "the editor mangled my roto".
//
//   2. roto_json HAS FRAMES. The backend holds a single frame across the whole
//      clip, and interpolates Catmull-Rom between several - so a one-frame
//      write is a valid static matte, and more frames is animated roto. The
//      editor has no frame concept, so the adapter carries the frame number.
//
//   3. roto_json CARRIES ITS CANVAS. The backend rescales by width/canvas.w,
//      so dropping the canvas silently rescales every shape.
//
// Point count per spline must match across keyframes - the backend
// interpolates point i against point i, so a keyframe with a different count
// interpolates against the wrong point and the shape tears. `keyframesAgree`
// checks that before anything is written.

export const ROTO_NODES = ["VectorRotoMEC"];

/** A fresh, empty roto document. */
export function emptyRoto(w = 1024, h = 1024) {
    return { canvas: { w, h }, frames: [{ frame: 0, splines: [] }] };
}

function num(v, fallback = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
}

/**
 * roto_json -> the editor's shapes.
 * Handles come back RELATIVE, which is what the editor expects.
 */
export function rotoToShapes(rotoJson, frame = 0) {
    let doc;
    try { doc = typeof rotoJson === "string" ? JSON.parse(rotoJson) : rotoJson; }
    catch (_e) { return { shapes: [], canvas: { w: 1024, h: 1024 }, frames: [0] }; }
    if (!doc || typeof doc !== "object") {
        return { shapes: [], canvas: { w: 1024, h: 1024 }, frames: [0] };
    }

    const canvas = {
        w: num(doc.canvas?.w, 1024) || 1024,
        h: num(doc.canvas?.h, 1024) || 1024,
    };
    const frames = Array.isArray(doc.frames) ? doc.frames : [];
    const frameNums = frames.map((f) => num(f.frame, 0));

    // The requested frame, or the nearest one at or before it - which is what
    // the backend's hold-then-interpolate does, so the editor shows what will
    // actually render rather than an empty canvas on an unkeyed frame.
    let entry = frames.find((f) => num(f.frame, 0) === frame);
    if (!entry && frames.length) {
        const before = frames.filter((f) => num(f.frame, 0) <= frame);
        entry = before.length
            ? before.reduce((a, b) => (num(a.frame) > num(b.frame) ? a : b))
            : frames[0];
    }

    const shapes = ((entry && entry.splines) || []).map((spline) => {
        const points = [], handles = [];
        for (const p of spline || []) {
            const x = num(p.x), y = num(p.y);
            points.push({ x, y });
            const i = Array.isArray(p.in) ? p.in : [x, y];
            const o = Array.isArray(p.out) ? p.out : [x, y];
            // absolute -> relative
            handles.push({
                in: { x: num(i[0]) - x, y: num(i[1]) - y },
                out: { x: num(o[0]) - x, y: num(o[1]) - y },
            });
        }
        return { points, handles, closed: true, type: "bezier" };
    });

    return { shapes, canvas, frames: frameNums.length ? frameNums : [0] };
}

/**
 * The editor's shapes -> roto_json, replacing just this frame's entry.
 * Handles go out ABSOLUTE.
 */
export function shapesToRoto(shapes, { previous, frame = 0, canvas } = {}) {
    let doc = emptyRoto(canvas?.w, canvas?.h);
    if (previous) {
        try {
            const p = typeof previous === "string" ? JSON.parse(previous) : previous;
            if (p && typeof p === "object") doc = p;
        } catch (_e) { /* a corrupt previous value must not lose this edit */ }
    }
    if (!doc.canvas) doc.canvas = { w: canvas?.w || 1024, h: canvas?.h || 1024 };
    if (canvas?.w) doc.canvas.w = canvas.w;
    if (canvas?.h) doc.canvas.h = canvas.h;
    if (!Array.isArray(doc.frames)) doc.frames = [];

    const splines = (shapes || []).map((sh) => {
        const pts = sh.points || [];
        const hs = sh.handles || [];
        return pts.map((p, i) => {
            const x = num(p.x), y = num(p.y);
            const h = hs[i] || {};
            const hin = h.in || { x: 0, y: 0 };
            const hout = h.out || { x: 0, y: 0 };
            // relative -> absolute
            return {
                x: +x.toFixed(3), y: +y.toFixed(3),
                in: [+(x + num(hin.x)).toFixed(3), +(y + num(hin.y)).toFixed(3)],
                out: [+(x + num(hout.x)).toFixed(3), +(y + num(hout.y)).toFixed(3)],
            };
        });
    }).filter((s) => s.length >= 2);   // a 1-point spline rasterises to nothing

    const at = doc.frames.findIndex((f) => num(f.frame, 0) === frame);
    if (at >= 0) doc.frames[at] = { frame, splines };
    else doc.frames.push({ frame, splines });
    doc.frames.sort((a, b) => num(a.frame) - num(b.frame));
    return doc;
}

/** Remove a keyframe. The last one is kept - zero frames rasterises nothing. */
export function removeKeyframe(rotoJson, frame) {
    let doc;
    try { doc = typeof rotoJson === "string" ? JSON.parse(rotoJson) : rotoJson; }
    catch (_e) { return emptyRoto(); }
    if (!doc || !Array.isArray(doc.frames) || doc.frames.length <= 1) return doc || emptyRoto();
    doc.frames = doc.frames.filter((f) => num(f.frame, 0) !== frame);
    if (!doc.frames.length) doc.frames = [{ frame: 0, splines: [] }];
    return doc;
}

/**
 * Do the keyframes agree about how many points each spline has?
 *
 * The backend interpolates point i against point i across keyframes. A
 * keyframe with a different count interpolates against the wrong point and
 * the shape TEARS between keys - and it does not raise, it just renders
 * wrongly, which is the hardest kind of roto bug to trace.
 */
export function keyframesAgree(rotoJson) {
    let doc;
    try { doc = typeof rotoJson === "string" ? JSON.parse(rotoJson) : rotoJson; }
    catch (_e) { return { ok: false, why: "roto_json is not valid JSON." }; }
    const frames = (doc && doc.frames) || [];
    if (frames.length <= 1) return { ok: true, why: "" };

    const counts = new Map();   // spline index -> {frame, n}[]
    for (const f of frames) {
        (f.splines || []).forEach((s, i) => {
            if (!counts.has(i)) counts.set(i, []);
            counts.get(i).push({ frame: num(f.frame, 0), n: (s || []).length });
        });
    }
    for (const [idx, seen] of counts) {
        const ns = new Set(seen.map((s) => s.n));
        if (ns.size > 1) {
            const detail = seen.map((s) => `f${s.frame}:${s.n}`).join(", ");
            return {
                ok: false,
                why: `Shape ${idx + 1} has a different number of points on `
                   + `different keyframes (${detail}). The renderer matches `
                   + `point 1 to point 1, so it will tear between keys. Add or `
                   + `remove points on EVERY keyframe, not just this one.`,
            };
        }
    }
    const lens = new Set(frames.map((f) => (f.splines || []).length));
    if (lens.size > 1) {
        return {
            ok: false,
            why: "Some keyframes have more shapes than others. A shape that "
               + "does not exist on every keyframe cannot be interpolated and "
               + "will pop in and out.",
        };
    }
    return { ok: true, why: "" };
}
