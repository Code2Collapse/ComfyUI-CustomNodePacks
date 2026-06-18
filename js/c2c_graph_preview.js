/**
 * c2c_graph_preview.js - Reusable workflow graph preview canvas (C2C).
 *
 * Renders a ComfyUI workflow JSON as a colour-coded node graph on a <canvas>:
 * nodes coloured by category (c2c_node_taxonomy.nodeColor), wires coloured by
 * data type (linkColor), with zoom / pan / fit-to-view / hover tooltip and a
 * category legend. Used by the Workflow Library detail pane and the Node
 * Explainer ("advanced, full of UI" surface). Inspired by
 * gregowahoo/comfyui-workflow-finder's parse_graph_data + canvas preview (MIT).
 *
 * Supports both workflow formats:
 *   • UI export  — { nodes:[{id,type,pos,size,title}], links:[[id,a,as,b,bs,type]] }
 *   • API/prompt — { "<id>": { class_type, inputs:{k:[srcId,slot]|value} } } (auto-laid-out)
 *
 * Public API:
 *   const ctrl = renderGraphPreview(containerEl, workflowJSON, { height });
 *   ctrl.fit();      // fit graph to view
 *   ctrl.destroy();  // remove listeners + canvas
 *
 * License: Apache-2.0
 */
import { nodeColor, linkColor, lighten, capabilityFor, CATEGORY_LEGEND } from "./c2c_node_taxonomy.js";

const DEF_W = 180, DEF_H = 70, COL_GAP = 90, ROW_GAP = 26;

// ── Parse either workflow format into {nodes, links} ──────────────────
function parseGraph(wf) {
    if (!wf || typeof wf !== "object") return { nodes: [], links: [], synthetic: false };

    // UI export format
    if (Array.isArray(wf.nodes)) {
        const nodes = wf.nodes.map((n) => {
            const pos = n.pos || [0, 0];
            const size = n.size || {};
            const w = size[0] ?? size.width ?? DEF_W;
            const h = size[1] ?? size.height ?? DEF_H;
            return {
                id: String(n.id),
                type: n.type || "Unknown",
                title: n.title || n.type || "Unknown",
                x: pos[0] ?? pos["0"] ?? 0,
                y: pos[1] ?? pos["1"] ?? 0,
                w: Math.max(120, w), h: Math.max(48, h),
            };
        });
        const links = (wf.links || []).map((l) => {
            // [linkId, originNode, originSlot, targetNode, targetSlot, type]
            if (Array.isArray(l)) {
                return { from: String(l[1]), to: String(l[3]), type: String(l[5] || "") };
            }
            return { from: String(l.origin_id), to: String(l.target_id), type: String(l.type || "") };
        }).filter((l) => l.from && l.to);
        const noPos = nodes.every((n) => n.x === 0 && n.y === 0);
        if (noPos) autoLayout(nodes, links);
        return { nodes, links, synthetic: noPos };
    }

    // API / prompt format: object keyed by id
    const nodes = [], links = [];
    for (const [id, def] of Object.entries(wf)) {
        if (!def || typeof def !== "object" || !def.class_type) continue;
        nodes.push({
            id: String(id), type: def.class_type,
            title: (def._meta && def._meta.title) || def.class_type,
            x: 0, y: 0, w: DEF_W, h: DEF_H,
        });
    }
    const ids = new Set(nodes.map((n) => n.id));
    for (const [id, def] of Object.entries(wf)) {
        if (!def || !def.inputs) continue;
        for (const v of Object.values(def.inputs)) {
            if (Array.isArray(v) && v.length === 2 && ids.has(String(v[0]))) {
                links.push({ from: String(v[0]), to: String(id), type: "" });
            }
        }
    }
    autoLayout(nodes, links);
    return { nodes, links, synthetic: true };
}

// ── Layered auto-layout for graphs without positions ──────────────────
function autoLayout(nodes, links) {
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const indeg = new Map(nodes.map((n) => [n.id, 0]));
    const outs = new Map(nodes.map((n) => [n.id, []]));
    for (const l of links) {
        if (byId.has(l.from) && byId.has(l.to)) {
            indeg.set(l.to, (indeg.get(l.to) || 0) + 1);
            outs.get(l.from).push(l.to);
        }
    }
    // Kahn-style longest-path layering.
    const depth = new Map(nodes.map((n) => [n.id, 0]));
    const queue = nodes.filter((n) => (indeg.get(n.id) || 0) === 0).map((n) => n.id);
    const rem = new Map(indeg);
    const seen = new Set();
    while (queue.length) {
        const id = queue.shift();
        if (seen.has(id)) continue;
        seen.add(id);
        for (const nx of outs.get(id) || []) {
            depth.set(nx, Math.max(depth.get(nx) || 0, (depth.get(id) || 0) + 1));
            rem.set(nx, (rem.get(nx) || 0) - 1);
            if (rem.get(nx) <= 0) queue.push(nx);
        }
    }
    // Any nodes left in cycles get depth 0.
    const cols = new Map();
    for (const n of nodes) {
        const d = depth.get(n.id) || 0;
        if (!cols.has(d)) cols.set(d, []);
        cols.get(d).push(n);
    }
    for (const [d, col] of cols) {
        col.forEach((n, i) => {
            n.x = d * (DEF_W + COL_GAP);
            n.y = i * (DEF_H + ROW_GAP);
        });
    }
}

// ── Render ────────────────────────────────────────────────────────────
export function renderGraphPreview(container, workflow, opts = {}) {
    const height = opts.height || 280;
    const { nodes, links, synthetic } = parseGraph(workflow);

    container.innerHTML = "";
    container.style.position = "relative";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = `width:100%;height:${height}px;display:block;background:#07071a;border-radius:6px;cursor:grab;`;
    container.appendChild(canvas);

    const bar = document.createElement("div");
    bar.style.cssText = "position:absolute;top:6px;right:8px;display:flex;gap:4px;z-index:2;";
    bar.innerHTML = `
<button data-a="fit" title="Fit (F)" style="background:#26264e;color:#d4d4f0;border:0;border-radius:4px;padding:2px 7px;cursor:pointer;font:11px monospace;">Fit</button>
<button data-a="in" style="background:#26264e;color:#d4d4f0;border:0;border-radius:4px;padding:2px 8px;cursor:pointer;font:11px monospace;">+</button>
<button data-a="out" style="background:#26264e;color:#d4d4f0;border:0;border-radius:4px;padding:2px 8px;cursor:pointer;font:11px monospace;">\u2212</button>`;
    container.appendChild(bar);

    const tip = document.createElement("div");
    tip.style.cssText = "position:absolute;pointer-events:none;display:none;background:#111128;color:#e0e0ff;border:1px solid #3a3a7e;border-radius:4px;padding:4px 7px;font:11px monospace;z-index:3;max-width:240px;";
    container.appendChild(tip);

    const ctx = canvas.getContext("2d");
    const view = { scale: 1, ox: 0, oy: 0 };
    let dpr = window.devicePixelRatio || 1;
    let raf = 0;

    function resize() {
        dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
        canvas.height = Math.max(1, Math.floor(height * dpr));
        draw();
    }

    function bounds() {
        if (!nodes.length) return { x: 0, y: 0, w: 100, h: 100 };
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const n of nodes) {
            x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y);
            x1 = Math.max(x1, n.x + n.w); y1 = Math.max(y1, n.y + n.h);
        }
        return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    }

    function fit() {
        const b = bounds();
        const cw = canvas.clientWidth || 1, ch = height;
        const pad = 30;
        const s = Math.min((cw - pad * 2) / b.w, (ch - pad * 2) / b.h, 1.5);
        view.scale = isFinite(s) && s > 0 ? s : 1;
        view.ox = (cw - b.w * view.scale) / 2 - b.x * view.scale;
        view.oy = (ch - b.h * view.scale) / 2 - b.y * view.scale;
        draw();
    }

    function toScreen(x, y) {
        return [x * view.scale + view.ox, y * view.scale + view.oy];
    }

    function draw() {
        ctx.save();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, canvas.clientWidth, height);
        ctx.fillStyle = "#07071a";
        ctx.fillRect(0, 0, canvas.clientWidth, height);

        const byId = new Map(nodes.map((n) => [n.id, n]));
        // wires
        ctx.lineWidth = Math.max(1, 1.6 * view.scale);
        for (const l of links) {
            const a = byId.get(l.from), b = byId.get(l.to);
            if (!a || !b) continue;
            const [sx, sy] = toScreen(a.x + a.w, a.y + a.h / 2);
            const [ex, ey] = toScreen(b.x, b.y + b.h / 2);
            const cdx = Math.max(30, Math.abs(ex - sx) * 0.5);
            ctx.strokeStyle = linkColor(l.type) || "#4a4a7a";
            ctx.globalAlpha = 0.85;
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.bezierCurveTo(sx + cdx, sy, ex - cdx, ey, ex, ey);
            ctx.stroke();
        }
        ctx.globalAlpha = 1;
        // nodes
        const fontPx = Math.max(8, Math.min(13, 11 * view.scale));
        for (const n of nodes) {
            const [x, y] = toScreen(n.x, n.y);
            const w = n.w * view.scale, h = n.h * view.scale;
            const base = nodeColor(n.type);
            ctx.fillStyle = base;
            roundRect(ctx, x, y, w, h, 5); ctx.fill();
            // header
            ctx.fillStyle = lighten(base, 34);
            roundRect(ctx, x, y, w, Math.min(20 * view.scale, h), 5); ctx.fill();
            ctx.strokeStyle = "#00000055"; ctx.lineWidth = 1;
            roundRect(ctx, x, y, w, h, 5); ctx.stroke();
            if (fontPx >= 8) {
                ctx.fillStyle = "#f0f0ff";
                ctx.font = `${fontPx}px Consolas,monospace`;
                ctx.save();
                ctx.beginPath(); ctx.rect(x + 4, y, w - 8, h); ctx.clip();
                ctx.fillText(n.title, x + 5, y + 14 * view.scale);
                ctx.restore();
            }
            n._screen = { x, y, w, h };
        }
        ctx.restore();
        if (synthetic) {
            ctx.save(); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.fillStyle = "#6a6a9a"; ctx.font = "10px monospace";
            ctx.fillText("auto-layout (no saved positions)", 8, height - 8);
            ctx.restore();
        }
    }

    function roundRect(c, x, y, w, h, r) {
        r = Math.min(r, w / 2, h / 2);
        c.beginPath();
        c.moveTo(x + r, y);
        c.arcTo(x + w, y, x + w, y + h, r);
        c.arcTo(x + w, y + h, x, y + h, r);
        c.arcTo(x, y + h, x, y, r);
        c.arcTo(x, y, x + w, y, r);
        c.closePath();
    }

    // ── interaction ──
    let dragging = false, lx = 0, ly = 0;
    const onDown = (e) => { dragging = true; lx = e.clientX; ly = e.clientY; canvas.style.cursor = "grabbing"; };
    const onUp = () => { dragging = false; canvas.style.cursor = "grab"; };
    const onMove = (e) => {
        if (dragging) {
            view.ox += e.clientX - lx; view.oy += e.clientY - ly;
            lx = e.clientX; ly = e.clientY; schedule();
            return;
        }
        // hover tooltip
        const r = canvas.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        let hit = null;
        for (const n of nodes) {
            const s = n._screen;
            if (s && mx >= s.x && mx <= s.x + s.w && my >= s.y && my <= s.y + s.h) { hit = n; break; }
        }
        if (hit) {
            tip.style.display = "block";
            tip.style.left = Math.min(mx + 12, r.width - 240) + "px";
            tip.style.top = (my + 12) + "px";
            tip.innerHTML = `<b>${escTip(hit.title)}</b><br><span style="color:#8a90ff">${escTip(hit.type)}</span><br><span style="color:#8888bb">${escTip(capabilityFor(hit.type))}</span>`;
        } else {
            tip.style.display = "none";
        }
    };
    const onWheel = (e) => {
        e.preventDefault();
        const r = canvas.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        const f = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        const ns = Math.max(0.1, Math.min(3, view.scale * f));
        view.ox = mx - (mx - view.ox) * (ns / view.scale);
        view.oy = my - (my - view.oy) * (ns / view.scale);
        view.scale = ns; schedule();
    };
    const onKey = (e) => { if (e.key === "f" || e.key === "F") fit(); };

    function schedule() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); }
    function escTip(s) { return String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; });
    canvas.addEventListener("wheel", onWheel, { passive: false });
    container.tabIndex = 0;
    container.addEventListener("keydown", onKey);
    bar.querySelector('[data-a="fit"]').addEventListener("click", fit);
    bar.querySelector('[data-a="in"]').addEventListener("click", () => { view.scale = Math.min(3, view.scale * 1.2); draw(); });
    bar.querySelector('[data-a="out"]').addEventListener("click", () => { view.scale = Math.max(0.1, view.scale / 1.2); draw(); });

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize(); fit();

    return {
        fit,
        nodeCount: nodes.length,
        linkCount: links.length,
        destroy() {
            ro.disconnect();
            window.removeEventListener("mouseup", onUp);
            container.removeEventListener("keydown", onKey);
            container.innerHTML = "";
        },
    };
}

// Small helper so callers can render a colour legend next to a preview.
export function legendHTML() {
    return CATEGORY_LEGEND.map(([hex, label]) =>
        `<span style="display:inline-flex;align-items:center;gap:4px;margin:2px 8px 2px 0;font-size:11px;color:#b0b0d0;">
<span style="width:10px;height:10px;border-radius:2px;background:${hex};display:inline-block;"></span>${label}</span>`).join("");
}

export default { renderGraphPreview, legendHTML };
