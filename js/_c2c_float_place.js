/**
 * _c2c_float_place.js - where a floating control may sit (L9.71, owner A20 17.1).
 *
 * Pure geometry: no DOM, no app. Rectangles are {left, top, right, bottom} in viewport pixels; a box is
 * {x, y, w, h}; the viewport is {w, h, top} where `top` is the reserved toolbar gutter. Used by
 * js/c2c_workflow_issues.js and tested with Node.
 */

export const MARGIN = 8;
export const SNAP_PX = 24;

/** Keep the box inside the viewport, below the top gutter. */
export function clampRect(box, vp, margin = MARGIN) {
    const minX = margin, maxX = Math.max(margin, vp.w - box.w - margin);
    const minY = Math.max(margin, vp.top || 0), maxY = Math.max(minY, vp.h - box.h - margin);
    return { ...box, x: Math.min(Math.max(box.x, minX), maxX), y: Math.min(Math.max(box.y, minY), maxY) };
}

/** True when the box and the rectangle overlap (touching edges do not count). */
export function overlaps(box, r) {
    return box.x < r.right && box.x + box.w > r.left && box.y < r.bottom && box.y + box.h > r.top;
}

/**
 * Move the box out of every keep-out rectangle: above the one it overlaps when that fits under the gutter, else
 * below it when that fits in the viewport, else to its right. A few passes settle boxes pushed into a neighbour.
 */
export function avoidRects(box, rects, vp, gap = 6, margin = MARGIN) {
    let b = clampRect(box, vp, margin);
    for (let pass = 0; pass < 4; pass++) {
        const hit = rects.find((r) => r && overlaps(b, r));
        if (!hit) return b;
        const above = hit.top - gap - b.h;
        const below = hit.bottom + gap;
        if (above >= Math.max(margin, vp.top || 0)) b = { ...b, y: above };
        else if (below + b.h <= vp.h - margin) b = { ...b, y: below };
        else b = { ...b, x: hit.right + gap };
        b = clampRect(b, vp, margin);
    }
    return b;
}

/** Within `snap` px of an edge, land on the margin (the top edge is the gutter). */
export function snapToEdges(box, vp, snap = SNAP_PX, margin = MARGIN) {
    const top = Math.max(margin, vp.top || 0);
    let { x, y } = box;
    if (x - margin < snap) x = margin;
    else if (vp.w - margin - (x + box.w) < snap) x = vp.w - margin - box.w;
    if (y - top < snap) y = top;
    else if (vp.h - margin - (y + box.h) < snap) y = vp.h - margin - box.h;
    return { ...box, x, y };
}

/** The box's centre as a fraction of the viewport: survives resizes, zoom and other screens. */
export function toFraction(box, vp) {
    return { fx: (box.x + box.w / 2) / Math.max(1, vp.w), fy: (box.y + box.h / 2) / Math.max(1, vp.h) };
}

/** A saved fraction back to a box of size w x h (null when the saved value is not usable). */
export function fromFraction(pos, w, h, vp) {
    if (!pos || !Number.isFinite(pos.fx) || !Number.isFinite(pos.fy)) return null;
    return { x: pos.fx * vp.w - w / 2, y: pos.fy * vp.h - h / 2, w, h };
}
