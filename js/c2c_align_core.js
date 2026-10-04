/**
 * c2c_align_core.js — pure geometry for C2C Align (no DOM, no ComfyUI).
 * Rects: { id, x, y, w, h }
 */

function left(r) { return r.x; }
function right(r) { return r.x + r.w; }
function top(r) { return r.y; }
function bottom(r) { return r.y + r.h; }
function cx(r) { return r.x + r.w / 2; }
function cy(r) { return r.y + r.h / 2; }

function overlapY(a, b) {
    return top(a) < bottom(b) && top(b) < bottom(a);
}

function overlapX(a, b) {
    return left(a) < right(b) && left(b) < right(a);
}

export function unionRect(rects) {
    if (!rects.length) return { x: 0, y: 0, w: 0, h: 0 };
    const x0 = Math.min(...rects.map(left));
    const y0 = Math.min(...rects.map(top));
    const x1 = Math.max(...rects.map(right));
    const y1 = Math.max(...rects.map(bottom));
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

export function unitsFrom(rects) {
    return rects.map((r) => ({ id: r.id, x: r.x, y: r.y, w: r.w, h: r.h }));
}

/**
 * Live node bounds (title bar included). Pass LiteGraph constants via opts.
 * @param {object} node
 * @param {{noTitle?: number, titleHeight?: number}} opts
 */
export function nodeTitleHeight(node, opts = {}) {
    const noTitle = opts.noTitle ?? 1;
    const titleHeight = opts.titleHeight ?? 30;
    if (node?.constructor?.title_mode === noTitle || node?.title_mode === noTitle) return 0;
    return titleHeight;
}

/** @param {object} node @param {{noTitle?: number, titleHeight?: number}} opts */
export function nodeRect(node, opts = {}) {
    const titleH = nodeTitleHeight(node, opts);
    const collapsed = !!node?.flags?.collapsed;
    const size = node?.size ?? [200, 100];
    const w = collapsed ? (node._collapsed_width ?? size[0]) : size[0];
    const h = collapsed ? titleH : size[1] + titleH;
    const pos = node?.pos ?? [0, 0];
    return { x: pos[0], y: pos[1] - titleH, w, h };
}

/** Map a bounding rect back to node.pos (body origin below title). */
export function nodePosFromRect(rect, titleH) {
    return [rect.x, rect.y + titleH];
}

/** @param {object} group */
export function groupRect(group) {
    const size = group?.size ?? [200, 100];
    const pos = group?.pos ?? [0, 0];
    return { x: pos[0], y: pos[1], w: size[0], h: size[1] };
}

function selectionBounds(units) {
    const u = unionRect(units);
    return {
        left: u.x, right: u.x + u.w, top: u.y, bottom: u.y + u.h,
        cx: u.x + u.w / 2, cy: u.y + u.h / 2,
    };
}

/** @param {object[]} units @param {{minGap?: number}} opts */
export function smartAlign(units, opts = {}) {
    const minGap = opts.minGap ?? 0;
    if (!units || units.length < 2) return { axis: "row", moves: [] };

    const centres = units.map((u) => ({ x: cx(u), y: cy(u) }));
    const xSpread = Math.max(...centres.map((c) => c.x)) - Math.min(...centres.map((c) => c.x));
    const ySpread = Math.max(...centres.map((c) => c.y)) - Math.min(...centres.map((c) => c.y));
    const axis = xSpread >= ySpread ? "row" : "column";

    const sorted = [...units].sort((a, b) => (axis === "row" ? a.x - b.x : a.y - b.y));

    if (axis === "row") {
        const ref = sorted[0];
        const alignY = ref.y;
        const gaps = [];
        for (let i = 0; i < sorted.length - 1; i++) {
            gaps.push(sorted[i + 1].x - right(sorted[i]));
        }
        const avg = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : minGap;
        const gap = Math.max(minGap, avg);

        const moves = [];
        let nextX = ref.x + ref.w + gap;
        for (let i = 0; i < sorted.length; i++) {
            const u = sorted[i];
            const tx = i === 0 ? u.x : nextX;
            const ty = alignY;
            moves.push({ id: u.id, dx: tx - u.x, dy: ty - u.y });
            if (i > 0) nextX = tx + u.w + gap;
        }
        return { axis, moves };
    }

    const ref = sorted[0];
    const alignX = ref.x;
    const gaps = [];
    for (let i = 0; i < sorted.length - 1; i++) {
        gaps.push(sorted[i + 1].y - bottom(sorted[i]));
    }
    const avg = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : minGap;
    const gap = Math.max(minGap, avg);

    const moves = [];
    let nextY = ref.y + ref.h + gap;
    for (let i = 0; i < sorted.length; i++) {
        const u = sorted[i];
        const tx = alignX;
        const ty = i === 0 ? u.y : nextY;
        moves.push({ id: u.id, dx: tx - u.x, dy: ty - u.y });
        if (i > 0) nextY = ty + u.h + gap;
    }
    return { axis, moves };
}

/** @param {object[]} units @param {string} mode */
export function alignUnits(units, mode) {
    if (!units || units.length < 2) return { moves: [] };
    const b = selectionBounds(units);
    const moves = [];
    for (const u of units) {
        let dx = 0;
        let dy = 0;
        switch (mode) {
            case "left": dx = b.left - u.x; break;
            case "right": dx = b.right - right(u); break;
            case "top": dy = b.top - u.y; break;
            case "bottom": dy = b.bottom - bottom(u); break;
            case "centerX": dx = b.cx - cx(u); break;
            case "centerY": dy = b.cy - cy(u); break;
            default: break;
        }
        moves.push({ id: u.id, dx, dy });
    }
    return { moves };
}

/** @param {object[]} units @param {"x"|"y"} axis @param {{minGap?: number}} opts */
export function distribute(units, axis, opts = {}) {
    const minGap = opts.minGap ?? 0;
    if (!units || units.length < 2) return { moves: [] };

    const horiz = axis === "x";
    const sorted = [...units].sort((a, b) => (horiz ? a.x - b.x : a.y - b.y));
    const first = sorted[0];
    const last = sorted[sorted.length - 1];

    if (sorted.length === 2) return { moves: sorted.map((u) => ({ id: u.id, dx: 0, dy: 0 })) };

    const moves = new Map(sorted.map((u) => [u.id, { id: u.id, dx: 0, dy: 0 }]));
    const middle = sorted.slice(1, -1);

    const firstEdge = horiz ? right(first) : bottom(first);
    const lastEdge = horiz ? last.x : last.y;
    const middleSize = middle.reduce((s, u) => s + (horiz ? u.w : u.h), 0);
    const span = lastEdge - firstEdge - middleSize;
    const numGaps = middle.length + 1;
    let gap = span / numGaps;

    if (gap < minGap) {
        gap = minGap;
        let cursor = horiz ? firstEdge + gap : firstEdge + gap;
        for (const u of middle) {
            if (horiz) {
                moves.get(u.id).dx = cursor - u.x;
                cursor += u.w + gap;
            } else {
                moves.get(u.id).dy = cursor - u.y;
                cursor += u.h + gap;
            }
        }
        return { moves: Array.from(moves.values()) };
    }

    let cursor = firstEdge + gap;
    for (const u of middle) {
        if (horiz) {
            moves.get(u.id).dx = cursor - u.x;
            cursor += u.w + gap;
        } else {
            moves.get(u.id).dy = cursor - u.y;
            cursor += u.h + gap;
        }
    }
    return { moves: Array.from(moves.values()) };
}

/** @param {object[]} units @param {"width"|"height"} dim */
export function equalSize(units, dim) {
    if (!units || !units.length) return [];
    const key = dim === "width" ? "w" : "h";
    const target = Math.max(...units.map((u) => u[key]));
    return units.map((u) => ({
        id: u.id,
        w: dim === "width" ? target : u.w,
        h: dim === "height" ? target : u.h,
    }));
}

function pickBest(candidates, threshold) {
    let best = null;
    for (const c of candidates) {
        const abs = Math.abs(c.correction);
        if (abs > threshold) continue;
        if (!best || abs < Math.abs(best.correction)) best = c;
    }
    return best;
}

function alignCandidatesX(moving, targets) {
    const out = [];
    const anchors = [
        ["left", left(moving)], ["center", cx(moving)], ["right", right(moving)],
    ];
    for (const t of targets) {
        const tAnchors = [
            ["left", left(t)], ["center", cx(t)], ["right", right(t)],
        ];
        for (const [, ma] of anchors) {
            for (const [, ta] of tAnchors) {
                const dx = ta - ma;
                const at = ta;
                out.push({
                    correction: dx,
                    guide: {
                        kind: "align",
                        axis: "x",
                        at,
                        from: Math.min(top(moving), top(t)),
                        to: Math.max(bottom(moving), bottom(t)),
                    },
                });
            }
        }
    }
    return out;
}

function alignCandidatesY(moving, targets) {
    const out = [];
    const anchors = [
        ["top", top(moving)], ["center", cy(moving)], ["bottom", bottom(moving)],
    ];
    for (const t of targets) {
        const tAnchors = [
            ["top", top(t)], ["center", cy(t)], ["bottom", bottom(t)],
        ];
        for (const [, ma] of anchors) {
            for (const [, ta] of tAnchors) {
                const dy = ta - ma;
                out.push({
                    correction: dy,
                    guide: {
                        kind: "align",
                        axis: "y",
                        at: ta,
                        from: Math.min(left(moving), left(t)),
                        to: Math.max(right(moving), right(t)),
                    },
                });
            }
        }
    }
    return out;
}

function rowBandTargets(moving, targets) {
    const band = targets.filter((t) => overlapY(t, moving));
    band.sort((a, b) => a.x - b.x);
    return band;
}

function columnBandTargets(moving, targets) {
    const band = targets.filter((t) => overlapX(t, moving));
    band.sort((a, b) => a.y - b.y);
    return band;
}

function findLeftNeighbour(moving, band) {
    let best = null;
    let bestRight = -Infinity;
    for (const t of band) {
        const r = right(t);
        if (r <= left(moving) && r > bestRight) {
            bestRight = r;
            best = t;
        }
    }
    return best;
}

function findRightNeighbour(moving, band) {
    let best = null;
    let bestLeft = Infinity;
    for (const t of band) {
        const l = left(t);
        if (l >= right(moving) && l < bestLeft) {
            bestLeft = l;
            best = t;
        }
    }
    return best;
}

function findTopNeighbour(moving, band) {
    let best = null;
    let bestBottom = -Infinity;
    for (const t of band) {
        const b = bottom(t);
        if (b <= top(moving) && b > bestBottom) {
            bestBottom = b;
            best = t;
        }
    }
    return best;
}

function findBottomNeighbour(moving, band) {
    let best = null;
    let bestTop = Infinity;
    for (const t of band) {
        const t0 = top(t);
        if (t0 >= bottom(moving) && t0 < bestTop) {
            bestTop = t0;
            best = t;
        }
    }
    return best;
}

function spacingCandidatesX(moving, band) {
    const out = [];
    const L = findLeftNeighbour(moving, band);
    const R = findRightNeighbour(moving, band);
    if (L && R) {
        const gapL = left(moving) - right(L);
        const gapR = R.x - right(moving);
        const targetLeft = (right(L) + R.x - moving.w) / 2;
        const dx = targetLeft - moving.x;
        const midY = (top(moving) + bottom(moving)) / 2;
        const g = Math.round((gapL + gapR) / 2);
        out.push({
            correction: dx,
            guide: {
                kind: "spacing", axis: "x", at: midY,
                from: right(L), to: R.x, label: g,
            },
        });
    }

    const consecutiveGaps = [];
    for (let i = 0; i < band.length - 1; i++) {
        const a = band[i];
        const b = band[i + 1];
        const g = b.x - right(a);
        if (g >= 0) consecutiveGaps.push({ g, from: right(a), to: b.x, midY: (Math.max(top(a), top(b)) + Math.min(bottom(a), bottom(b))) / 2 });
    }

    for (const { g, from, to, midY } of consecutiveGaps) {
        if (L) {
            const dx = (right(L) + g) - moving.x;
            out.push({
                correction: dx,
                guide: { kind: "spacing", axis: "x", at: midY, from: right(L), to: left(moving) + moving.w, label: g },
            });
        }
        if (R) {
            const dx = (R.x - g - moving.w) - moving.x;
            out.push({
                correction: dx,
                guide: { kind: "spacing", axis: "x", at: midY, from: right(moving), to: R.x, label: g },
            });
        }
    }
    return out;
}

function spacingCandidatesY(moving, band) {
    const out = [];
    const T = findTopNeighbour(moving, band);
    const B = findBottomNeighbour(moving, band);
    if (T && B) {
        const gapT = top(moving) - bottom(T);
        const gapB = B.y - bottom(moving);
        const targetTop = (bottom(T) + B.y - moving.h) / 2;
        const dy = targetTop - moving.y;
        const midX = (left(moving) + right(moving)) / 2;
        const g = Math.round((gapT + gapB) / 2);
        out.push({
            correction: dy,
            guide: {
                kind: "spacing", axis: "y", at: midX,
                from: bottom(T), to: B.y, label: g,
            },
        });
    }

    const consecutiveGaps = [];
    for (let i = 0; i < band.length - 1; i++) {
        const a = band[i];
        const b = band[i + 1];
        const g = b.y - bottom(a);
        if (g >= 0) consecutiveGaps.push({ g, from: bottom(a), to: b.y, midX: (Math.max(left(a), left(b)) + Math.min(right(a), right(b))) / 2 });
    }

    for (const { g, from, to, midX } of consecutiveGaps) {
        if (T) {
            const dy = (bottom(T) + g) - moving.y;
            out.push({
                correction: dy,
                guide: { kind: "spacing", axis: "y", at: midX, from: bottom(T), to: top(moving) + moving.h, label: g },
            });
        }
        if (B) {
            const dy = (B.y - g - moving.h) - moving.y;
            out.push({
                correction: dy,
                guide: { kind: "spacing", axis: "y", at: midX, from: bottom(moving), to: B.y, label: g },
            });
        }
    }
    return out;
}

/**
 * @param {{x:number,y:number,w:number,h:number}} moving
 * @param {object[]} targets
 * @param {{threshold?: number}} opts
 */
export function snapMove(moving, targets, opts = {}) {
    const threshold = opts.threshold ?? 8;
    const rowBand = rowBandTargets(moving, targets);
    const colBand = columnBandTargets(moving, targets);

    const xCandidates = [
        ...alignCandidatesX(moving, targets),
        ...spacingCandidatesX(moving, rowBand),
    ];
    const yCandidates = [
        ...alignCandidatesY(moving, targets),
        ...spacingCandidatesY(moving, colBand),
    ];

    const bestX = pickBest(xCandidates, threshold);
    const bestY = pickBest(yCandidates, threshold);
    const guides = [];
    if (bestX?.guide) guides.push(bestX.guide);
    if (bestY?.guide) guides.push(bestY.guide);

    return {
        dx: bestX?.correction ?? 0,
        dy: bestY?.correction ?? 0,
        guides,
    };
}
