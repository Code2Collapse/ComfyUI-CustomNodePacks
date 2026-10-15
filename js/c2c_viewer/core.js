/**
 * Pure logic of the C2C viewer (L7.114, owner A14): no DOM, no ComfyUI imports, so it runs under Node in the tests
 * (tests/test_viewer_core_js.py). The DOM component is viewer.js; the transport is transport.js.
 *
 * Coordinates are NORMALISED to the plate (0..1, origin top left), as the node's `picks` input stores them
 * (nodes/mask_matting/_picks.py reads the same schema). Masks are Uint8Array hit maps (1 = inside) at the proxy
 * size Detect wrote them in.
 *
 * License: Apache-2.0
 */

export const PICKS_VERSION = 1;
/** A detection at least this much inside the clicked object is a part of it, not another subject. */
export const PART_INSIDE = 0.8;
/** An added part may not take in this much (or more) of an excluded object. */
export const SWALLOW = 0.5;
/** Ctrl-click removes the nearest marker within this distance (share of the plate's long side). */
export const REMOVE_RADIUS = 0.035;
export const DEFAULT_PREVIEW_FRAMES = 17;

// ── masks and hit maps ──────────────────────────────────────────────────────────────────────────────────────────

/** RGBA pixels (canvas ImageData.data) -> hit map. The mask lives in the alpha channel; a fully opaque image (a
 *  greyscale PNG) is read by its brightness instead. */
export function maskFromRGBA(data, w, h) {
    const n = w * h;
    const out = new Uint8Array(n);
    let opaque = true;
    for (let i = 0; i < n; i += 1) if (data[i * 4 + 3] !== 255) { opaque = false; break; }
    for (let i = 0; i < n; i += 1) {
        const v = opaque ? (data[i * 4] * 0.299 + data[i * 4 + 1] * 0.587 + data[i * 4 + 2] * 0.114) : data[i * 4 + 3];
        out[i] = v > 127 ? 1 : 0;
    }
    return out;
}

export function countOf(mask) {
    let c = 0;
    for (let i = 0; i < mask.length; i += 1) c += mask[i];
    return c;
}

/** |a AND b| / |b|: how much of b lies inside a (0 when b is empty or the sizes differ). */
export function containment(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let inter = 0, nb = 0;
    for (let i = 0; i < b.length; i += 1) {
        if (b[i]) { nb += 1; if (a[i]) inter += 1; }
    }
    return nb ? inter / nb : 0;
}

/** The object under a normalised point: the smallest one whose mask holds it (most specific), else null. */
export function hitTest(objects, x, y) {
    let best = null;
    for (const o of objects || []) {
        if (!o?.mask || !o.w || !o.h) continue;
        const px = Math.min(o.w - 1, Math.max(0, Math.floor(x * o.w)));
        const py = Math.min(o.h - 1, Math.max(0, Math.floor(y * o.h)));
        if (!o.mask[py * o.w + px]) continue;
        const area = o.count ?? countOf(o.mask);
        if (!best || area < best.area) best = { obj: o, area };
    }
    return best ? best.obj : null;
}

/** The point deepest inside a mask (largest chamfer distance to its edge), normalised; null for an empty mask.
 *  Excluded objects are sent to SAM as negative points: a point near their middle, never on their edge. */
export function interiorPoint(mask, w, h) {
    const BIG = 1e9;
    const d = new Float32Array(w * h);
    for (let i = 0; i < w * h; i += 1) d[i] = mask[i] ? BIG : 0;
    for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
            const i = y * w + x;
            if (!d[i]) continue;
            const up = y > 0 ? d[i - w] + 1 : 1, left = x > 0 ? d[i - 1] + 1 : 1;
            const ul = y > 0 && x > 0 ? d[i - w - 1] + 1.4142 : 1.4142;
            const ur = y > 0 && x < w - 1 ? d[i - w + 1] + 1.4142 : 1.4142;
            d[i] = Math.min(d[i], up, left, ul, ur);
        }
    }
    let best = -1, bi = -1;
    for (let y = h - 1; y >= 0; y -= 1) {
        for (let x = w - 1; x >= 0; x -= 1) {
            const i = y * w + x;
            if (!d[i]) continue;
            const down = y < h - 1 ? d[i + w] + 1 : 1, right = x < w - 1 ? d[i + 1] + 1 : 1;
            const dr = y < h - 1 && x < w - 1 ? d[i + w + 1] + 1.4142 : 1.4142;
            const dl = y < h - 1 && x > 0 ? d[i + w - 1] + 1.4142 : 1.4142;
            d[i] = Math.min(d[i], down, right, dr, dl);
            if (d[i] > best) { best = d[i]; bi = i; }
        }
    }
    if (bi < 0) return null;
    return { x: ((bi % w) + 0.5) / w, y: (Math.floor(bi / w) + 0.5) / h };
}

// ── the selection ───────────────────────────────────────────────────────────────────────────────────────────────
// sel = { frame, subject: {x, y, id?, box?, label?} | null, parts: [ids inside the subject], excluded: [{x, y, id,
//         box, label}], add: [{x, y, region?}], cut: [{x, y, region?}] }   (region = a hit map at proxy size)

export function emptySelection(frame = 0) {
    return { frame: frame | 0, subject: null, parts: [], excluded: [], add: [], cut: [] };
}

function objLabel(o) {
    return `${o.label || "object"} #${(o.id ?? 0) + 1}`;
}

function markerOf(o, x, y) {
    const m = { x, y, id: o?.id ?? null };
    if (o?.box) m.box = o.box.slice(0, 4);
    if (o?.label) m.label = o.label;
    return m;
}

/** Click (no modifier): the object under the point becomes the subject; every other detection is excluded, except
 *  those mostly inside the subject (they are parts of it). With no detection under the point the subject is the
 *  point alone (SAM decides its extent at run time). */
export function clickSubject(objects, x, y, frame) {
    const sel = emptySelection(frame);
    const hit = hitTest(objects, x, y);
    sel.subject = hit ? markerOf(hit, x, y) : { x, y, id: null };
    for (const o of objects || []) {
        if (hit && o.id === hit.id) continue;
        if (hit && containment(hit.mask, o.mask) >= PART_INSIDE) { sel.parts.push(o.id); continue; }
        const p = o.mask ? interiorPoint(o.mask, o.w, o.h) : null;
        if (p) sel.excluded.push(markerOf(o, p.x, p.y));
    }
    return { sel, message: hit ? `Subject: ${objLabel(hit)}. ${sel.excluded.length} other object(s) excluded.`
        : (objects?.length ? "Nothing was found at that point: the subject is the point itself. Click a tinted "
            + "object to use its outline." : "Subject point set.") };
}

/** Shift-click: add a part. Refused (sel unchanged, ok false, reason in words) when it would take in an excluded
 *  object - the point lies on one, or the part's region covers SWALLOW of one. ``region`` is the part's hit map
 *  when one is known (from Detect); without it only the point test applies. */
export function addPart(sel, objects, x, y, region = null) {
    if (!sel?.subject) return { ok: false, sel, reason: "Click the subject first (no modifier key), then Shift-click to add parts." };
    const hit = hitTest(objects, x, y);
    if (hit && (hit.id === sel.subject.id || sel.parts.includes(hit.id)) && !region) {
        return { ok: false, sel, reason: "That point is already part of the subject." };
    }
    const exByObj = new Map(objects.filter((o) => sel.excluded.some((e) => e.id === o.id)).map((o) => [o.id, o]));
    if (hit && exByObj.has(hit.id)) {
        return { ok: false, sel, reason: `That point is on ${objLabel(hit)}, another subject. Adding it would take that `
            + "object in. Click it without Shift to make it the subject, or Ctrl-click it to stop excluding it." };
    }
    if (region) {
        for (const o of exByObj.values()) {
            const share = containment(region, o.mask);
            if (share >= SWALLOW) {
                return { ok: false, sel, reason: `This part would take in ${Math.round(share * 100)}% of ${objLabel(o)}, `
                    + "another subject. Click closer to the part you want, or pick that object as the subject." };
            }
        }
    }
    const next = { ...sel, add: [...sel.add, region ? { x, y, region } : { x, y }] };
    return { ok: true, sel: next, reason: "" };
}

/** Alt-click: cut a part (a negative point; its region, when known, is taken out of the preview). */
export function cutPart(sel, x, y, region = null) {
    if (!sel?.subject) return { ok: false, sel, reason: "Click the subject first, then Alt-click to cut parts away." };
    return { ok: true, sel: { ...sel, cut: [...sel.cut, region ? { x, y, region } : { x, y }] }, reason: "" };
}

/** Ctrl-click: remove the nearest marker (subject, added, cut, excluded) within REMOVE_RADIUS. ``aspect`` = plate
 *  width / height, so the distance is round on screen. */
export function removeNearest(sel, x, y, aspect = 1) {
    if (!sel) return { sel, removed: null };
    const sx = aspect >= 1 ? 1 : aspect, sy = aspect >= 1 ? 1 / aspect : 1;
    const dist = (m) => Math.hypot((m.x - x) * sx, (m.y - y) * sy);
    let best = null;
    const consider = (kind, i, m) => { const d = dist(m); if (d <= REMOVE_RADIUS && (!best || d < best.d)) best = { kind, i, d }; };
    if (sel.subject) consider("subject", 0, sel.subject);
    sel.add.forEach((m, i) => consider("add", i, m));
    sel.cut.forEach((m, i) => consider("cut", i, m));
    sel.excluded.forEach((m, i) => consider("excluded", i, m));
    if (!best) return { sel, removed: null };
    const next = { ...sel, add: [...sel.add], cut: [...sel.cut], excluded: [...sel.excluded], parts: [...sel.parts] };
    if (best.kind === "subject") { next.subject = null; next.parts = []; next.add = []; next.cut = []; }
    else next[best.kind].splice(best.i, 1);
    return { sel: next, removed: best.kind };
}

/** The preview of what will be matted on the pick frame: subject + its parts + added regions - cut regions. */
export function selectionMask(objects, sel, w, h) {
    const out = new Uint8Array(w * h);
    if (!sel?.subject) return out;
    const byId = new Map((objects || []).map((o) => [o.id, o]));
    const orIn = (m) => { if (m && m.length === out.length) for (let i = 0; i < out.length; i += 1) if (m[i]) out[i] = 1; };
    orIn(byId.get(sel.subject.id)?.mask);
    for (const id of sel.parts) orIn(byId.get(id)?.mask);
    for (const a of sel.add) orIn(a.region);
    for (const c of sel.cut) {
        if (c.region && c.region.length === out.length) for (let i = 0; i < out.length; i += 1) if (c.region[i]) out[i] = 0;
    }
    return out;
}

// ── picks JSON (the node's `picks` input) ───────────────────────────────────────────────────────────────────────

const r5 = (v) => Math.round(v * 1e5) / 1e5;

/** The selection as the node reads it. ``plate`` = {frames, w, h}; ``range`` only for the viewer's own preview. */
export function toPicks(sel, { text = "", model = "", plate = null, range = null } = {}) {
    if (!sel?.subject) return "";
    const pt = (m, withBox) => {
        const o = { x: r5(m.x), y: r5(m.y) };
        if (withBox && m.box) o.box = m.box.map(r5);
        if (withBox && m.label) o.label = m.label;
        return o;
    };
    const p = {
        v: PICKS_VERSION, frame: sel.frame | 0, text: String(text || ""), model: String(model || ""),
        subject: pt(sel.subject, true),
        excluded: sel.excluded.map((m) => pt(m, true)),
        add: sel.add.map((m) => pt(m, false)),
        cut: sel.cut.map((m) => pt(m, false)),
    };
    if (plate) p.plate = { frames: plate.frames | 0, w: plate.w | 0, h: plate.h | 0 };
    if (range) p.range = { start: range.start | 0, count: range.count | 0 };
    return JSON.stringify(p);
}

/** The node's `picks` value back into a selection (markers only: masks come from a new Find). null = none. */
export function fromPicks(text) {
    if (!text || !String(text).trim()) return null;
    let d;
    try { d = JSON.parse(text); } catch (_e) { return null; }
    if (!d || typeof d !== "object" || !d.subject || d.v !== PICKS_VERSION) return null;
    const ok = (m) => m && Number.isFinite(m.x) && Number.isFinite(m.y) && m.x >= 0 && m.x <= 1 && m.y >= 0 && m.y <= 1;
    if (!ok(d.subject)) return null;
    const sel = emptySelection(d.frame | 0);
    sel.subject = { x: d.subject.x, y: d.subject.y, id: null, box: d.subject.box, label: d.subject.label };
    sel.excluded = (d.excluded || []).filter(ok).map((m) => ({ x: m.x, y: m.y, id: null, box: m.box, label: m.label }));
    sel.add = (d.add || []).filter(ok).map((m) => ({ x: m.x, y: m.y }));
    sel.cut = (d.cut || []).filter(ok).map((m) => ({ x: m.x, y: m.y }));
    return { sel, text: String(d.text || ""), model: String(d.model || ""), plate: d.plate || null };
}

// ── the stepper ─────────────────────────────────────────────────────────────────────────────────────────────────

/** Steps Plate -> Subject -> Render. A node without a pick step (remove background) has no Subject step.
 *  state = {plate: bool, subject: bool, preview: bool, result: bool, busy: bool, pickStep: bool}
 *  Returns {steps: [{id, label, state}], current, action: {id, label}} - the one prominent next action. */
export function stepper(state, previewFrames = DEFAULT_PREVIEW_FRAMES) {
    const pick = state.pickStep !== false;
    const ids = pick ? ["plate", "subject", "render"] : ["plate", "render"];
    const labels = { plate: "Plate", subject: "Subject", render: "Render" };
    const current = !state.plate ? "plate" : (pick && !state.subject) ? "subject" : "render";
    const at = ids.indexOf(current);
    const steps = ids.map((id, i) => ({ id, label: labels[id], state: i < at ? "done" : i === at ? "current" : "todo" }));
    let action;
    if (state.busy) action = { id: "wait", label: "Working…" };
    else if (current === "plate") action = { id: "load", label: "Load plate" };
    else if (current === "subject") action = { id: "find", label: "Find" };
    else if (!state.preview) action = { id: "preview", label: `Preview ${previewFrames} frames` };
    else action = { id: "full", label: "Render full shot" };
    return { steps, current, action };
}

/** The frames of a preview chunk: N frames from the pick frame, shifted back to fit at the end of the plate. */
export function previewRange(pickFrame, frames, n = DEFAULT_PREVIEW_FRAMES) {
    const total = Math.max(1, frames | 0);
    const count = Math.max(1, Math.min(total, n | 0));
    const start = Math.max(0, Math.min(pickFrame | 0, total - count));
    return { start, count };
}

// ── frames ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** The proxy entry to show for a plate frame: the nearest one written (a long clip has every k-th). */
export function nearestFrame(frames, index) {
    if (!frames?.length) return null;
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (frames[mid].index < index) lo = mid + 1; else hi = mid;
    }
    const a = frames[lo], b = frames[Math.max(0, lo - 1)];
    return Math.abs(b.index - index) < Math.abs(a.index - index) ? b : a;
}

/** "frame 12 (13 / 120)" - the plate frame number and its place in the clip. */
export function frameLabel(index, count) {
    return `frame ${index} (${index + 1} / ${count})`;
}

/** Where a client point falls on the plate drawn "contain" into a box: normalised {x, y, inside}. */
export function viewToPlate(box, plateW, plateH, clientX, clientY) {
    const s = Math.min(box.width / plateW, box.height / plateH);
    const dw = plateW * s, dh = plateH * s;
    const ox = box.left + (box.width - dw) / 2, oy = box.top + (box.height - dh) / 2;
    const x = (clientX - ox) / dw, y = (clientY - oy) / dh;
    return { x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)), inside: x >= 0 && x <= 1 && y >= 0 && y <= 1 };
}

/** The drawn rectangle of the plate inside a box (same "contain" fit), in box pixels. */
export function fitRect(boxW, boxH, plateW, plateH) {
    const s = Math.min(boxW / plateW, boxH / plateH);
    const w = plateW * s, h = plateH * s;
    return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h, scale: s };
}

// ── QC badges ───────────────────────────────────────────────────────────────────────────────────────────────────

export const QC_NAMES = { coverage: "Coverage", edge_detail: "Edge detail", flicker: "Flicker" };

/** green / amber / red / n/a for one number with the thresholds the server sent (nodes/c2c_viewer/qc.py). */
export function qcLevel(value, t) {
    if (value === null || value === undefined || !Number.isFinite(value) || !t) return "n/a";
    if (t.better === "higher") return value >= t.green ? "green" : value >= t.amber ? "amber" : "red";
    return value <= t.green ? "green" : value <= t.amber ? "amber" : "red";
}

export function qcValueText(name, value) {
    if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
    if (name === "coverage") return `${(value * 100).toFixed(value >= 0.995 ? 0 : 1)}%`;
    if (name === "edge_detail") return value.toFixed(2);
    return value.toFixed(3);
}

/** The tooltip: what the number is, the thresholds, and where they come from. */
export function qcTooltip(name, metric, t, qcSide = 640) {
    const what = {
        coverage: "Share of the selection (the segmenter's mask) that the matte covers.",
        edge_detail: "Gradient energy of the matte in the edge band, divided by the coarse mask's. A smeared edge drops toward 0; hair strands raise it.",
        flicker: "Motion-compensated change of the matte in the edge band from one frame to the next (optical flow on the plate), in alpha units.",
    }[name] || "";
    const lines = [`${QC_NAMES[name] || name}: ${qcValueText(name, metric?.value)}`, what];
    if (t) {
        const cmp = t.better === "higher" ? ["≥", "≥"] : ["≤", "≤"];
        const f = (v) => qcValueText(name, v);
        lines.push(`Green ${cmp[0]} ${f(t.green)}, amber ${cmp[1]} ${f(t.amber)}, red beyond.`);
    }
    if (metric?.why) lines.push(`n/a: ${metric.why}.`);
    if (name === "flicker" && metric?.worst?.frame != null) lines.push(`Worst pair ends on frame ${metric.worst.frame}.`);
    lines.push(`Thresholds measured on synthetic clips with a known matte at ≤ ${qcSide} px (docs/design/matte_viewer.md); not a claim about your footage.`);
    return lines.filter(Boolean).join("\n");
}
