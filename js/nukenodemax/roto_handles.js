// FILE: web/extensions/nukenodemax/roto_handles.js
// FEATURE: F1 (frontend) — Bezier handle overlay for VectorRotoMEC
// INTEGRATES WITH: nodes/roto.py (consumes the JSON we serialise here)
//
// Adds a custom widget that opens a fullscreen canvas modal where the user
// places / drags points and their cubic-bezier in/out handles. On close, the
// JSON is written into the node's `roto_json` STRING widget.

import { app } from "../../../scripts/app.js";

function newPoint(x, y) {
    return {
        x, y,
        in:  [x - 30, y],
        out: [x + 30, y],
    };
}

function openEditor(node, widget) {
    let data;
    try { data = JSON.parse(widget.value); } catch { data = { canvas: { w: 1024, h: 1024 }, frames: [{ frame: 0, splines: [] }] }; }
    if (!data.frames || !data.frames.length) data.frames = [{ frame: 0, splines: [] }];
    const splines = data.frames[0].splines;
    if (!splines.length) splines.push([]);
    const spline = splines[0];

    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", background: "rgba(0,0,0,0.85)",
        zIndex: 99999, display: "flex", flexDirection: "column",
    });
    const bar = document.createElement("div");
    bar.style.cssText = "padding:8px;color:#fff;font:12px monospace;display:flex;gap:8px;align-items:center;";
    bar.innerHTML = `
        <span>Click empty area to add point. Drag points / handles. Right-click point to delete.</span>
        <span style="flex:1"></span>
        <button id="mec-roto-cancel">Cancel</button>
        <button id="mec-roto-save">Save</button>
    `;
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "flex:1;background:#111;cursor:crosshair;";
    canvas.width = data.canvas.w;
    canvas.height = data.canvas.h;
    overlay.appendChild(bar);
    overlay.appendChild(canvas);
    document.body.appendChild(overlay);

    const ctx = canvas.getContext("2d");
    let dragging = null; // {pt, kind: "xy"|"in"|"out", idx}

    function pos(ev) {
        const r = canvas.getBoundingClientRect();
        return {
            x: (ev.clientX - r.left) * canvas.width / r.width,
            y: (ev.clientY - r.top) * canvas.height / r.height,
        };
    }
    function hit(p) {
        for (let i = 0; i < spline.length; i++) {
            const pt = spline[i];
            if (Math.hypot(p.x - pt.x, p.y - pt.y) < 8) return { pt, kind: "xy", idx: i };
            if (Math.hypot(p.x - pt.in[0], p.y - pt.in[1]) < 6) return { pt, kind: "in", idx: i };
            if (Math.hypot(p.x - pt.out[0], p.y - pt.out[1]) < 6) return { pt, kind: "out", idx: i };
        }
        return null;
    }
    function deCasteljau(p0, p1, p2, p3, n) {
        const out = [];
        for (let i = 0; i <= n; i++) {
            const t = i / n, u = 1 - t;
            const a0x = u*p0[0]+t*p1[0], a0y = u*p0[1]+t*p1[1];
            const a1x = u*p1[0]+t*p2[0], a1y = u*p1[1]+t*p2[1];
            const a2x = u*p2[0]+t*p3[0], a2y = u*p2[1]+t*p3[1];
            const b0x = u*a0x+t*a1x, b0y = u*a0y+t*a1y;
            const b1x = u*a1x+t*a2x, b1y = u*a1y+t*a2y;
            out.push([u*b0x+t*b1x, u*b0y+t*b1y]);
        }
        return out;
    }
    function draw() {
        ctx.fillStyle = "#111"; ctx.fillRect(0, 0, canvas.width, canvas.height);
        if (spline.length >= 2) {
            ctx.strokeStyle = "#5cf"; ctx.lineWidth = 2; ctx.beginPath();
            for (let i = 0; i < spline.length; i++) {
                const a = spline[i], b = spline[(i + 1) % spline.length];
                const seg = deCasteljau([a.x, a.y], a.out, b.in, [b.x, b.y], 24);
                seg.forEach(([x, y], k) => k === 0 && i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
            }
            ctx.closePath(); ctx.stroke();
        }
        // Handles + points
        for (const pt of spline) {
            ctx.strokeStyle = "#888"; ctx.beginPath();
            ctx.moveTo(pt.in[0], pt.in[1]); ctx.lineTo(pt.x, pt.y);
            ctx.lineTo(pt.out[0], pt.out[1]); ctx.stroke();
            ctx.fillStyle = "#fa0"; ctx.beginPath();
            ctx.arc(pt.in[0], pt.in[1], 4, 0, 6.283); ctx.fill();
            ctx.beginPath(); ctx.arc(pt.out[0], pt.out[1], 4, 0, 6.283); ctx.fill();
            ctx.fillStyle = "#fff"; ctx.beginPath();
            ctx.arc(pt.x, pt.y, 6, 0, 6.283); ctx.fill();
        }
    }
    canvas.addEventListener("mousedown", (e) => {
        const p = pos(e);
        const h = hit(p);
        if (e.button === 2 && h && h.kind === "xy") {
            spline.splice(h.idx, 1);
            draw();
            return;
        }
        if (h) { dragging = h; }
        else if (e.button === 0) { spline.push(newPoint(p.x, p.y)); draw(); }
    });
    canvas.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const p = pos(e);
        if (dragging.kind === "xy") {
            const dx = p.x - dragging.pt.x, dy = p.y - dragging.pt.y;
            dragging.pt.in[0] += dx; dragging.pt.in[1] += dy;
            dragging.pt.out[0] += dx; dragging.pt.out[1] += dy;
            dragging.pt.x = p.x; dragging.pt.y = p.y;
        } else {
            dragging.pt[dragging.kind][0] = p.x;
            dragging.pt[dragging.kind][1] = p.y;
        }
        draw();
    });
    canvas.addEventListener("mouseup", () => { dragging = null; });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    bar.querySelector("#mec-roto-cancel").onclick = () => document.body.removeChild(overlay);
    bar.querySelector("#mec-roto-save").onclick = () => {
        widget.value = JSON.stringify(data);
        node.setDirtyCanvas(true, true);
        document.body.removeChild(overlay);
    };
    draw();
}

app.registerExtension({
    name: "nukenodemax.roto_handles",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VectorRotoMEC") return;
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            orig?.apply(this, arguments);
            const widget = this.widgets?.find(w => w.name === "roto_json");
            this.addWidget("button", "✎ Edit Splines", "edit", () => {
                if (widget) openEditor(this, widget);
            });
        };
    },
});
