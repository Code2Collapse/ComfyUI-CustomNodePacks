// _editor_empty_state.js — shared empty-state painter for the MEC canvas
// editors (spline / points+bbox / tracker / paint).
//
// An empty editor used to render as a flat near-black void — no grid, no
// invitation, nothing that tells a first-time user what to do (the Director
// timeline's "+ add image — or drop a file" lanes set the quality bar).
// This paints, in DOC space (ctx already panned/zoomed):
//   1. a subtle dark checkerboard over the canvas rect (reads as "empty
//      artboard", not "broken node"),
//   2. a centered glyph + hint lines in muted slate.
//
// Canvas fillStyle cannot resolve var() — read C at draw time.

import { C } from "./_c2c_theme.js";

export function drawEditorEmptyState(ctx, w, h, z, glyph, lines) {
    const zz = Math.max(0.05, z || 1);
    ctx.save();
    // Checkerboard (doc-space 24px tiles, clipped to the canvas rect)
    ctx.beginPath();
    ctx.rect(0, 0, w, h);
    ctx.clip();
    const tile = 24;
    for (let y = 0; y < h; y += tile) {
        for (let x = 0; x < w; x += tile) {
            ctx.fillStyle = (((x + y) / tile) % 2 === 0) ? C.bg3 : C.bg2;
            ctx.fillRect(x, y, Math.min(tile, w - x), Math.min(tile, h - y));
        }
    }
    // Centered glyph + hint lines (sized in screen px via 1/z)
    const cx = w / 2, cy = h / 2;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(148,158,190,0.55)";
    ctx.font = `${28 / zz}px system-ui, sans-serif`;
    ctx.fillText(glyph || "✎", cx, cy - 26 / zz);
    // Hint lines are word-wrapped to the canvas: on a small editor (Vector
    // Roto's 1024px doc at 24% zoom is ~246px wide) they were clipped at both
    // edges. Spacing is unchanged for lines that fit: 18px between hints,
    // 14px for a wrapped continuation.
    const maxW = Math.max(40 / zz, w - 20 / zz);
    let y = cy + 4 / zz;
    (lines || []).forEach((line, i) => {
        if (i === 0) {
            ctx.font = `600 ${13 / zz}px system-ui, sans-serif`;
            ctx.fillStyle = "rgba(168,178,208,0.75)";
        } else if (i === 1) {
            ctx.font = `${11 / zz}px system-ui, sans-serif`;
            ctx.fillStyle = "rgba(128,138,166,0.55)";
        }
        _wrap(ctx, line, maxW).forEach((part, j) => {
            if (j > 0) y += 14 / zz;
            ctx.fillText(part, cx, y);
        });
        y += 18 / zz;
    });
    ctx.restore();
}

function _wrap(ctx, text, maxW) {
    const out = [];
    let line = "";
    for (const word of String(text || "").split(/\s+/).filter(Boolean)) {
        const next = line ? `${line} ${word}` : word;
        if (line && ctx.measureText(next).width > maxW) { out.push(line); line = word; }
        else line = next;
    }
    out.push(line);
    return out;
}
