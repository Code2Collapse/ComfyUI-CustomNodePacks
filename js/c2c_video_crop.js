/**
 * Crop tool for the C2C video loaders (L2.53; owner 2026-10-15: "do whatever I want with the preview like
 * WhatDreamsCost"). Behaviour after WhatDreamsCost's Load Video UI crop mode (read for behaviour only, no code taken):
 * a rectangle over the preview with eight handles, aspect presets (Free, Original, 1:1 ... 1:2), reset, and the crop
 * size in pixels. It writes the node's crop_x / crop_y / crop_w / crop_h (fractions of the frame); the run then
 * decodes each frame, cuts the rectangle out and resizes it (custom_width / custom_height).
 *
 * The preview always shows the WHOLE frame; outside the crop is shaded. Nothing runs unless the pointer is on it.
 */
import {
    CROP_PRESETS, FULL_RECT, cropPixels, dragRect, fitAspect, isFullRect, presetRatio,
} from "./_c2c_player_core.js";

const STYLE_ID = "c2c-vcrop-v1";
const PARTS = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
.c2c-vcrop { position: absolute; inset: 0; pointer-events: none; overflow: hidden; z-index: 2; }
.c2c-vcrop[hidden] { display: none; }
.c2c-vcrop__rect { position: absolute; box-sizing: border-box; border: 1px solid var(--cu-accent);
  box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.55); }
.c2c-vcrop--edit .c2c-vcrop__rect { pointer-events: auto; cursor: move; border-width: 2px; }
.c2c-vcrop__third { position: absolute; background: rgba(255, 255, 255, 0.35); display: none; }
.c2c-vcrop--edit .c2c-vcrop__third { display: block; }
.c2c-vcrop__h { position: absolute; width: 10px; height: 10px; margin: -5px 0 0 -5px; border-radius: 2px;
  background: var(--cu-accent); box-shadow: 0 0 0 1px var(--cu-ground); display: none; pointer-events: auto; }
.c2c-vcrop--edit .c2c-vcrop__h { display: block; }
.c2c-vcrop__bar { display: flex; align-items: center; gap: 4px; flex-wrap: wrap; font-size: 10px; color: var(--cu-ink-soft); }
.c2c-vcrop__bar select { height: 20px; font-size: 11px; border-radius: 4px; border: 1px solid var(--cu-edge-strong);
  background: var(--cu-raised); color: var(--cu-ink); padding: 0 2px; }
.c2c-vcrop__label { font-variant-numeric: tabular-nums; white-space: nowrap; }
`;
    document.head.appendChild(s);
}

const HANDLE_POS = {
    nw: [0, 0], n: [50, 0], ne: [100, 0], e: [100, 50], se: [100, 100], s: [50, 100], sw: [0, 100], w: [0, 50],
};
const CURSORS = { nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
                  n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize" };

/**
 * viewport: the preview's box (position: relative). barHost: where the crop controls go.
 * opts.onCommit(rect) writes the widgets; opts.makeButton(text, title, onClick) builds a button in the player style.
 * Returns { setFrameSize(w, h), setRect(rect), destroy() }.
 */
export function attachCrop(viewport, barHost, opts) {
    ensureStyles();
    const state = { rect: { ...FULL_RECT }, frameW: 0, frameH: 0, editing: false, preset: "Free" };

    const layer = document.createElement("div");
    layer.className = "c2c-vcrop";
    const box = document.createElement("div");
    box.className = "c2c-vcrop__rect";
    for (const [k, v] of [["left", "33.333%"], ["left", "66.667%"]]) {
        const l = document.createElement("div");
        l.className = "c2c-vcrop__third";
        Object.assign(l.style, { [k]: v, top: "0", bottom: "0", width: "1px" });
        box.appendChild(l);
    }
    for (const v of ["33.333%", "66.667%"]) {
        const l = document.createElement("div");
        l.className = "c2c-vcrop__third";
        Object.assign(l.style, { top: v, left: "0", right: "0", height: "1px" });
        box.appendChild(l);
    }
    const handles = {};
    for (const p of PARTS) {
        const h = document.createElement("div");
        h.className = "c2c-vcrop__h";
        h.dataset.part = p;
        h.style.left = `${HANDLE_POS[p][0]}%`;
        h.style.top = `${HANDLE_POS[p][1]}%`;
        h.style.cursor = CURSORS[p];
        h.title = "Drag to resize the crop";
        handles[p] = h;
        box.appendChild(h);
    }
    layer.appendChild(box);
    viewport.appendChild(layer);

    // ── controls ──
    const bar = document.createElement("div");
    bar.className = "c2c-vcrop__bar";
    const bEdit = opts.makeButton("crop", "Crop the frames this node loads: drag the box or its handles on the preview", () => setEditing(!state.editing));
    const sel = document.createElement("select");
    sel.title = "Crop shape";
    sel.setAttribute("aria-label", "Crop shape");
    for (const p of CROP_PRESETS) {
        const o = document.createElement("option");
        o.value = p;
        o.textContent = p;
        sel.appendChild(o);
    }
    sel.addEventListener("pointerdown", (e) => e.stopPropagation());
    sel.addEventListener("change", (e) => {
        e.stopPropagation();
        state.preset = sel.value;
        const ratio = presetRatio(state.preset, state.frameW, state.frameH);
        if (ratio) {
            state.rect = fitAspect(ratio, state.frameW, state.frameH, isFullRect(state.rect) ? null : state.rect);
            render();
            commit();
        }
    });
    const bReset = opts.makeButton("no crop", "Remove the crop (load the whole frame)", () => {
        state.rect = { ...FULL_RECT };
        state.preset = "Free";
        sel.value = "Free";
        render();
        commit();
    });
    const label = document.createElement("span");
    label.className = "c2c-vcrop__label";
    bar.append(bEdit, sel, bReset, label);
    barHost.appendChild(bar);

    function render() {
        const r = state.rect;
        const full = isFullRect(r);
        layer.hidden = full && !state.editing;
        layer.classList.toggle("c2c-vcrop--edit", state.editing);
        box.style.left = `${r.x * 100}%`;
        box.style.top = `${r.y * 100}%`;
        box.style.width = `${r.w * 100}%`;
        box.style.height = `${r.h * 100}%`;
        bEdit.setAttribute("aria-pressed", String(state.editing));
        sel.hidden = !state.editing;
        bReset.disabled = full;
        if (state.frameW > 0 && state.frameH > 0) {
            const px = cropPixels(r, state.frameW, state.frameH);
            label.textContent = px.full
                ? `whole frame ${state.frameW}×${state.frameH}`
                : `crop ${px.w}×${px.h} at ${px.x}, ${px.y} (of ${state.frameW}×${state.frameH})`;
        } else {
            label.textContent = full ? "" : "crop set";
        }
    }

    function setEditing(on) {
        state.editing = !!on;
        if (state.editing && isFullRect(state.rect)) state.rect = { x: 0.1, y: 0.1, w: 0.8, h: 0.8 };
        render();
        if (!state.editing) commit();
    }

    function commit() {
        try { opts.onCommit?.(isFullRect(state.rect) ? { ...FULL_RECT } : { ...state.rect }); } catch (_e) { /* the widget write reports */ }
    }

    // ── pointer ──
    let drag = null;
    const onDown = (e) => {
        if (!state.editing) return;
        e.stopPropagation();
        e.preventDefault();
        const part = e.target?.dataset?.part || "move";
        const vr = layer.getBoundingClientRect();
        drag = { part, x0: e.clientX, y0: e.clientY, w: Math.max(1, vr.width), h: Math.max(1, vr.height),
                 start: { ...state.rect }, id: e.pointerId };
        try { box.setPointerCapture(e.pointerId); } catch (_e) { /* ignore */ }
    };
    const onMove = (e) => {
        if (!drag) return;
        e.stopPropagation();
        const dx = (e.clientX - drag.x0) / drag.w, dy = (e.clientY - drag.y0) / drag.h;
        const ratio = presetRatio(state.preset, state.frameW, state.frameH);
        state.rect = dragRect(drag.start, drag.part, dx, dy, ratio, state.frameW || 1, state.frameH || 1);
        render();
    };
    const onUp = (e) => {
        if (!drag) return;
        e.stopPropagation();
        try { box.releasePointerCapture(drag.id); } catch (_e) { /* ignore */ }
        drag = null;
        commit();
    };
    box.addEventListener("pointerdown", onDown);
    box.addEventListener("pointermove", onMove);
    box.addEventListener("pointerup", onUp);
    box.addEventListener("pointercancel", onUp);

    render();

    return {
        setFrameSize(w, h) {
            state.frameW = Math.max(0, w | 0);
            state.frameH = Math.max(0, h | 0);
            render();
        },
        setRect(r) {
            if (drag) return;                        // never fight the user's drag
            const x = Number(r?.x), y = Number(r?.y), w = Number(r?.w), h = Number(r?.h);
            state.rect = [x, y, w, h].every(Number.isFinite) ? { x, y, w, h } : { ...FULL_RECT };
            render();
        },
        get rect() { return { ...state.rect }; },
        destroy() {
            layer.remove();
            bar.remove();
        },
    };
}
