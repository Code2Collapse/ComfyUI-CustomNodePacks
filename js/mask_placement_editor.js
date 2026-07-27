/**
 * MaskPlacementMEC — placement quad editor widget.
 *
 * Drag the alpha into place on the anchor frame with a real 4-corner quad:
 *   drag inside      move
 *   drag a corner    free perspective warp
 *   drag an edge mid uniform-ish scale on that axis
 *   drag the rotate  handle above the quad -> rotate about the centre
 *   Shift + corner   uniform scale about the centre (keeps the shape square-ish)
 *   R                reset to a centred quad
 *   F                fit the quad to the frame
 *
 * Writes `placement_json` = {"corners":[[x,y] TL,TR,BR,BL], "feather":px}
 * in ANCHOR-FRAME PIXEL coordinates — exactly what mask_placement.py parses.
 *
 * The backdrop comes from the node's own `ui.mp_preview` payload (queue once
 * to populate it), same transport as pose_gaze_viewer.
 */
import { app } from "../../scripts/app.js";
import { reportFailure as __c2cReport } from "./_c2c_report.js";
import { ensureC2CKit } from "./_c2c_ui_kit.js";

const TARGET = "MaskPlacementMEC";
const HANDLE_R = 7;           // corner hit radius (screen px)
const ROT_OFFSET = 34;        // rotate handle distance above the top edge

// Canvas2D can't parse var() — resolve to literals once (see var-in-canvas-bug).
const COLOR = {
    line: "#5b9dd9",
    lineSoft: "rgba(91,157,217,0.55)",
    fill: "rgba(91,157,217,0.10)",
    corner: "#7fd1ff",
    mid: "#a6e3a1",
    rot: "#f9e2af",
    text: "#e6e6e6",
    sub: "#8b93a7",
    bg: "#141414",
};

function _clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function installEditor(node) {
    if (node._mpEditor) return;
    ensureC2CKit();

    const wPlacement = node.widgets?.find((w) => w.name === "placement_json");
    const wFeather = node.widgets?.find((w) => w.name === "feather_px");
    if (wPlacement) {
        // Editor owns it — hide the raw JSON textarea but keep the value live.
        wPlacement.type = "hidden";
        wPlacement.computeSize = () => [0, -4];
        if (wPlacement.element) wPlacement.element.style.display = "none";
    }

    const root = document.createElement("div");
    root.className = "c2ck-panel";
    root.style.cssText =
        "width:100%;height:100%;display:flex;flex-direction:column;gap:6px;" +
        "min-height:0;box-sizing:border-box;padding:6px;background:" + COLOR.bg + ";";

    // ── toolbar ──────────────────────────────────────────────────────
    const bar = document.createElement("div");
    bar.style.cssText = "display:flex;gap:6px;align-items:center;flex:0 0 auto;flex-wrap:wrap;";
    const mkBtn = (label, title, fn) => {
        const b = document.createElement("button");
        b.className = "c2ck-btn";
        b.textContent = label;
        b.title = title;
        b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); fn(); };
        bar.appendChild(b);
        return b;
    };
    mkBtn("Reset", "Reset the quad to a centred rectangle (R)", () => { resetQuad(); commit(); render(); });
    mkBtn("Fit", "Stretch the quad to the whole frame (F)", () => { fitQuad(); commit(); render(); });
    const status = document.createElement("span");
    status.className = "c2ck-pill";
    status.textContent = "queue once for the backdrop";
    bar.appendChild(status);
    root.appendChild(bar);

    // ── canvas ───────────────────────────────────────────────────────
    const stage = document.createElement("div");
    stage.style.cssText = "position:relative;flex:1 1 auto;min-height:120px;overflow:hidden;border-radius:5px;";
    const cvs = document.createElement("canvas");
    cvs.style.cssText = "width:100%;height:100%;display:block;touch-action:none;cursor:crosshair;";
    stage.appendChild(cvs);
    root.appendChild(stage);
    const ctx = cvs.getContext("2d");

    // ── state ────────────────────────────────────────────────────────
    let srcW = 512, srcH = 512;      // anchor-frame pixel dims
    let bdImg = null;                 // backdrop Image
    let corners = null;               // [[x,y] x4] in SOURCE pixel coords
    let drag = null;                  // {kind:'move'|'corner'|'mid'|'rot', idx, ...}
    let hover = -1;
    let raf = 0;

    function resetQuad() {
        const w = srcW * 0.3, h = srcH * 0.3;
        const cx = srcW * 0.5, cy = srcH * 0.5;
        corners = [[cx - w, cy - h], [cx + w, cy - h], [cx + w, cy + h], [cx - w, cy + h]];
    }
    function fitQuad() {
        corners = [[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]];
    }

    function load() {
        let ok = false;
        try {
            const raw = wPlacement?.value;
            if (raw && String(raw).trim()) {
                const d = JSON.parse(raw);
                if (Array.isArray(d?.corners) && d.corners.length === 4) {
                    corners = d.corners.map((p) => [Number(p[0]) || 0, Number(p[1]) || 0]);
                    ok = true;
                }
            }
        } catch (_) { /* fall through to default */ }
        if (!ok) resetQuad();
    }

    function commit() {
        if (!wPlacement || !corners) return;
        const payload = {
            corners: corners.map((p) => [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]),
            feather: Number(wFeather?.value ?? 6),
        };
        wPlacement.value = JSON.stringify(payload);
        // Mark the graph dirty so the value is serialized with the workflow.
        node.graph?.setDirtyCanvas(true, true);
    }

    // ── coordinate mapping (source px <-> canvas px, letterboxed) ────
    let map = { ox: 0, oy: 0, s: 1 };
    function computeMap() {
        const cw = cvs.width, ch = cvs.height;
        const s = Math.min(cw / srcW, ch / srcH);
        map = { ox: (cw - srcW * s) / 2, oy: (ch - srcH * s) / 2, s };
    }
    const toScreen = (p) => [map.ox + p[0] * map.s, map.oy + p[1] * map.s];
    const toSource = (x, y) => [(x - map.ox) / map.s, (y - map.oy) / map.s];

    function evtCanvas(e) {
        const r = cvs.getBoundingClientRect();
        return [(e.clientX - r.left) * (cvs.width / r.width),
                (e.clientY - r.top) * (cvs.height / r.height)];
    }

    function centroid() {
        return [(corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4,
                (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4];
    }
    function midpoints() {
        return corners.map((p, i) => {
            const q = corners[(i + 1) % 4];
            return [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
        });
    }
    function rotHandle() {
        const m = midpoints()[0];              // top edge midpoint
        const c = centroid();
        const dx = m[0] - c[0], dy = m[1] - c[1];
        const len = Math.hypot(dx, dy) || 1;
        const off = ROT_OFFSET / Math.max(map.s, 1e-3);
        return [m[0] + (dx / len) * off, m[1] + (dy / len) * off];
    }

    function pointInQuad(px, py) {
        let inside = false;
        for (let i = 0, j = 3; i < 4; j = i++) {
            const xi = corners[i][0], yi = corners[i][1];
            const xj = corners[j][0], yj = corners[j][1];
            if ((yi > py) !== (yj > py) &&
                px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-9) + xi) inside = !inside;
        }
        return inside;
    }

    function hitTest(cx, cy) {
        const near = (sp) => Math.hypot(sp[0] - cx, sp[1] - cy) <= HANDLE_R + 2;
        const rh = toScreen(rotHandle());
        if (near(rh)) return { kind: "rot" };
        for (let i = 0; i < 4; i++) if (near(toScreen(corners[i]))) return { kind: "corner", idx: i };
        const mids = midpoints();
        for (let i = 0; i < 4; i++) if (near(toScreen(mids[i]))) return { kind: "mid", idx: i };
        const sp = toSource(cx, cy);
        if (pointInQuad(sp[0], sp[1])) return { kind: "move" };
        return null;
    }

    // ── pointer handling ─────────────────────────────────────────────
    cvs.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.stopPropagation(); e.preventDefault();
        const [cx, cy] = evtCanvas(e);
        const hit = hitTest(cx, cy);
        if (!hit) return;
        try { cvs.setPointerCapture(e.pointerId); } catch (_) {}
        drag = {
            ...hit,
            start: toSource(cx, cy),
            orig: corners.map((p) => [p[0], p[1]]),
            origC: centroid(),
        };
    });

    cvs.addEventListener("pointermove", (e) => {
        const [cx, cy] = evtCanvas(e);
        if (!drag) {
            const h = hitTest(cx, cy);
            const nh = h ? (h.kind === "move" ? 100 : (h.idx ?? 0)) : -1;
            cvs.style.cursor = !h ? "crosshair"
                : h.kind === "move" ? "grab"
                : h.kind === "rot" ? "alias" : "pointer";
            if (nh !== hover) { hover = nh; render(); }
            return;
        }
        e.stopPropagation(); e.preventDefault();
        const now = toSource(cx, cy);
        const dx = now[0] - drag.start[0], dy = now[1] - drag.start[1];

        if (drag.kind === "move") {
            corners = drag.orig.map((p) => [p[0] + dx, p[1] + dy]);
        } else if (drag.kind === "corner") {
            if (e.shiftKey) {
                // Uniform scale about the centre.
                const c = drag.origC;
                const d0 = Math.hypot(drag.orig[drag.idx][0] - c[0], drag.orig[drag.idx][1] - c[1]) || 1;
                const d1 = Math.hypot(now[0] - c[0], now[1] - c[1]);
                const k = _clamp(d1 / d0, 0.05, 20);
                corners = drag.orig.map((p) => [c[0] + (p[0] - c[0]) * k, c[1] + (p[1] - c[1]) * k]);
            } else {
                corners = drag.orig.map((p, i) => (i === drag.idx ? [p[0] + dx, p[1] + dy] : [p[0], p[1]]));
            }
        } else if (drag.kind === "mid") {
            // Move the two corners of that edge together.
            const a = drag.idx, b = (drag.idx + 1) % 4;
            corners = drag.orig.map((p, i) =>
                (i === a || i === b) ? [p[0] + dx, p[1] + dy] : [p[0], p[1]]);
        } else if (drag.kind === "rot") {
            const c = drag.origC;
            const a0 = Math.atan2(drag.start[1] - c[1], drag.start[0] - c[0]);
            const a1 = Math.atan2(now[1] - c[1], now[0] - c[0]);
            const t = a1 - a0, cs = Math.cos(t), sn = Math.sin(t);
            corners = drag.orig.map((p) => {
                const px = p[0] - c[0], py = p[1] - c[1];
                return [c[0] + px * cs - py * sn, c[1] + px * sn + py * cs];
            });
        }
        render();
    });

    const endDrag = (e) => {
        if (!drag) return;
        drag = null;
        try { cvs.releasePointerCapture(e.pointerId); } catch (_) {}
        commit();
        render();
    };
    cvs.addEventListener("pointerup", endDrag);
    cvs.addEventListener("pointercancel", endDrag);

    cvs.addEventListener("keydown", (e) => {
        if (e.key === "r" || e.key === "R") { resetQuad(); commit(); render(); }
        else if (e.key === "f" || e.key === "F") { fitQuad(); commit(); render(); }
    });
    cvs.tabIndex = 0;

    // ── render ───────────────────────────────────────────────────────
    function render() {
        if (raf) return;
        raf = requestAnimationFrame(() => {
            raf = 0;
            try { draw(); } catch (err) { __c2cReport?.("MaskPlacement.render", err, "mask_placement_editor"); }
        });
    }

    function draw() {
        const r = stage.getBoundingClientRect();
        const w = Math.max(64, Math.floor(r.width)), h = Math.max(64, Math.floor(r.height));
        if (cvs.width !== w || cvs.height !== h) { cvs.width = w; cvs.height = h; }
        computeMap();
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = COLOR.bg;
        ctx.fillRect(0, 0, w, h);

        // backdrop
        if (bdImg && bdImg.complete && bdImg.naturalWidth) {
            ctx.drawImage(bdImg, map.ox, map.oy, srcW * map.s, srcH * map.s);
        } else {
            // checkerboard empty state + hint
            const t = 16;
            for (let y = 0; y < h; y += t) {
                for (let x = 0; x < w; x += t) {
                    ctx.fillStyle = ((x / t + y / t) & 1) ? "#1b1b1b" : "#202020";
                    ctx.fillRect(x, y, t, t);
                }
            }
            ctx.fillStyle = COLOR.sub;
            ctx.font = "12px ui-sans-serif,system-ui";
            ctx.textAlign = "center";
            ctx.fillText("Queue once to load the anchor frame", w / 2, h / 2 - 8);
            ctx.fillText("then drag the quad to place your mask", w / 2, h / 2 + 10);
            ctx.textAlign = "left";
        }
        if (!corners) return;

        // quad
        const sp = corners.map(toScreen);
        ctx.beginPath();
        ctx.moveTo(sp[0][0], sp[0][1]);
        for (let i = 1; i < 4; i++) ctx.lineTo(sp[i][0], sp[i][1]);
        ctx.closePath();
        ctx.fillStyle = COLOR.fill; ctx.fill();
        ctx.strokeStyle = COLOR.line; ctx.lineWidth = 2; ctx.stroke();

        // rotate handle
        const rh = toScreen(rotHandle());
        const tm = toScreen(midpoints()[0]);
        ctx.strokeStyle = COLOR.lineSoft; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(tm[0], tm[1]); ctx.lineTo(rh[0], rh[1]); ctx.stroke();
        ctx.beginPath(); ctx.arc(rh[0], rh[1], HANDLE_R - 1, 0, Math.PI * 2);
        ctx.fillStyle = COLOR.rot; ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.stroke();

        // edge midpoints
        ctx.fillStyle = COLOR.mid;
        for (const m of midpoints().map(toScreen)) {
            ctx.beginPath(); ctx.arc(m[0], m[1], HANDLE_R - 2, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.lineWidth = 1; ctx.stroke();
        }
        // corners
        for (const p of sp) {
            ctx.beginPath(); ctx.arc(p[0], p[1], HANDLE_R, 0, Math.PI * 2);
            ctx.fillStyle = COLOR.corner; ctx.fill();
            ctx.strokeStyle = "rgba(0,0,0,0.65)"; ctx.lineWidth = 1.5; ctx.stroke();
        }
    }

    // ── DOM widget + lifecycle ───────────────────────────────────────
    node.addDOMWidget("mask_placement_editor", "canvas", root, {
        getValue: () => "",
        setValue: () => {},
        serialize: false,
    });
    if (!node.size || node.size[0] < 380) node.size = [Math.max(node.size?.[0] || 0, 380), Math.max(node.size?.[1] || 0, 420)];

    const ro = new ResizeObserver(() => render());
    ro.observe(stage);

    const origExec = node.onExecuted;
    node.onExecuted = function (out) {
        origExec?.apply(this, arguments);
        try {
            const sz = out?.mp_size?.[0];
            if (Array.isArray(sz) && sz.length === 2) {
                const newW = Number(sz[0]) || srcW, newH = Number(sz[1]) || srcH;
                if (newW !== srcW || newH !== srcH) {
                    srcW = newW; srcH = newH;
                    if (!wPlacement?.value || !String(wPlacement.value).trim()) resetQuad();
                }
            }
            const b64 = out?.mp_preview?.[0];
            if (b64) {
                const im = new Image();
                im.onload = () => { bdImg = im; render(); };
                im.src = b64;
                status.textContent = `anchor frame ${out?.mp_anchor?.[0] ?? 0} · ${srcW}×${srcH}`;
                status.className = "c2ck-pill on";
            }
        } catch (err) {
            __c2cReport?.("MaskPlacement.onExecuted", err, "mask_placement_editor");
        }
        render();
    };

    const origRemoved = node.onRemoved;
    node.onRemoved = function () {
        origRemoved?.apply(this, arguments);
        try { ro.disconnect(); } catch (_) {}
        if (raf) cancelAnimationFrame(raf);
    };

    load();
    commit();
    node._mpEditor = { load, render, commit };
    setTimeout(render, 50);
}

// Guard so a duplicate copy of this file in another pack can't double-register.
if (!(app.extensions || []).some((e) => e?.name === "C2C.MaskPlacementEditor")) {
    app.registerExtension({
        name: "C2C.MaskPlacementEditor",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== TARGET) return;
            const orig = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                orig?.apply(this, arguments);
                if (this._mpEditor) { this._mpEditor.load(); this._mpEditor.render(); }
            };
        },
        async nodeCreated(node) {
            if (node.comfyClass !== TARGET) return;
            try { installEditor(node); }
            catch (err) { __c2cReport?.("MaskPlacement.install", err, "mask_placement_editor"); }
        },
    });
}
