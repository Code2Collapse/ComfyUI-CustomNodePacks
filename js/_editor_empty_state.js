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
// Colors are literal hex on purpose: canvas fillStyle cannot resolve
// var(--x) (the all-black-confetti bug class).

const CHECK_A = "#15151d";
const CHECK_B = "#1a1a24";

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
            ctx.fillStyle = (((x + y) / tile) % 2 === 0) ? CHECK_A : CHECK_B;
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
    ctx.font = `600 ${13 / zz}px system-ui, sans-serif`;
    ctx.fillStyle = "rgba(168,178,208,0.75)";
    if (lines && lines.length) ctx.fillText(lines[0], cx, cy + 4 / zz);
    ctx.font = `${11 / zz}px system-ui, sans-serif`;
    ctx.fillStyle = "rgba(128,138,166,0.55)";
    for (let i = 1; i < (lines?.length || 0); i++) {
        ctx.fillText(lines[i], cx, cy + (4 + i * 18) / zz);
    }
    ctx.restore();
}
