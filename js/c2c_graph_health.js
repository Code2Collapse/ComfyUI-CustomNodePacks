// c2c_graph_health.js — Dead-node + cycle + dangling-input detector (C2C)
// ---------------------------------------------------------------------
// What it does:
//   • Continuously analyses the graph (debounced ~400 ms after edits).
//   • Flags THREE classes of problem and paints them on the canvas:
//        ① Dead nodes      — no output is reachable from any terminal/
//                            output-marked node (won't execute).
//                            Painted with a translucent amber border.
//        ② Cycle members   — strongly-connected components of size>1.
//                            Painted with a translucent red border + ↻ glyph.
//        ③ Dangling inputs — required inputs (not optional) that are
//                            unconnected. Painted with a small red dot
//                            next to the slot in the minimap-paint hook.
//   • A small status pill (bottom-left) shows live counts:
//        "● 0 cycles · 3 dead · 1 dangling"
//     Click the pill → opens a list panel; clicking a row pans+pulses
//     the offending node.
//   • Toggle: c2c.graphHealth.enabled. Default ON.
//
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";

const PILL_ID  = "c2c-graph-health-pill";
const PANEL_ID = "c2c-graph-health-panel";
const SETTING_ID = "c2c.graphHealth.enabled";
const PULSE_MS = 1100;

let _result = { dead: [], cycles: [], dangling: [] };
let _scheduled = 0;

function injectStyle() {
    if (document.getElementById("c2c-graph-health-style")) return;
    const s = document.createElement("style");
    s.id = "c2c-graph-health-style";
    s.textContent = `
#${PILL_ID} {
    position: fixed; left: 14px; bottom: 14px;
    z-index: 9000;
    background: rgba(18,20,24,0.86);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 999px;
    padding: 5px 11px; cursor: pointer;
    font: 11px ui-sans-serif, system-ui, sans-serif;
    color: #cfd6e0; backdrop-filter: blur(6px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.40);
    display: flex; align-items: center; gap: 6px;
}
#${PILL_ID}.ok        { color: #7ee0a8; border-color: rgba(126,224,168,0.30); }
#${PILL_ID}.warn      { color: #ffd166; border-color: rgba(255,209,102,0.35); }
#${PILL_ID}.err       { color: #ff6b6b; border-color: rgba(255,107,107,0.40); }
#${PILL_ID} .dot      { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
#${PANEL_ID} {
    position: fixed; left: 14px; bottom: 50px;
    z-index: 9001;
    width: 360px; max-height: 360px;
    background: rgba(22,24,30,0.95);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 9px;
    color: #e7ecf3;
    font: 12px ui-sans-serif, system-ui, sans-serif;
    overflow: hidden;
    display: none; flex-direction: column;
    box-shadow: 0 10px 30px rgba(0,0,0,0.55);
    backdrop-filter: blur(8px);
}
#${PANEL_ID}.open { display: flex; }
#${PANEL_ID} .hdr  { padding: 8px 12px; background: rgba(91,141,239,0.10);
    border-bottom: 1px solid rgba(91,141,239,0.20); font-weight:600; color:#9ec1ff; }
#${PANEL_ID} .body { overflow-y: auto; padding: 4px 0; }
#${PANEL_ID} .row  { padding: 5px 12px; cursor: pointer;
    display:flex; gap: 8px; align-items: center;
    border-bottom: 1px solid rgba(255,255,255,0.04); }
#${PANEL_ID} .row:hover { background: rgba(91,141,239,0.10); }
#${PANEL_ID} .tag  { font-size: 9.5px; padding: 1px 6px; border-radius: 9px;
    background: rgba(255,255,255,0.06); }
#${PANEL_ID} .tag.dead  { color:#ffd166; }
#${PANEL_ID} .tag.cycle { color:#ff6b6b; }
#${PANEL_ID} .tag.dang  { color:#ff8e8e; }`;
    document.head.appendChild(s);
}

// ── Analysis ─────────────────────────────────────────────────────────
function analyze() {
    const g = app.graph;
    const nodes = g?._nodes || [];
    if (!nodes.length) { _result = { dead: [], cycles: [], dangling: [] }; return; }

    // Build adjacency.
    const succ = new Map(), pred = new Map();
    for (const n of nodes) { succ.set(n.id, new Set()); pred.set(n.id, new Set()); }
    for (const link of Object.values(g.links || {})) {
        if (!link) continue;
        const a = link.origin_id, b = link.target_id;
        if (succ.has(a) && pred.has(b)) {
            succ.get(a).add(b);
            pred.get(b).add(a);
        }
    }

    // ── ① Dead-node = STRICT: a node whose outputs are not connected
    // to ANY other node AND the node itself is not an OUTPUT_NODE/preview/save.
    // (Old reachability heuristic over-flagged useful upstream nodes when no
    //  downstream node matched the regex.)
    const isOutputish = (n) => {
        if (n.constructor?.nodeData?.output_node) return true;
        const type = (n.type || "").toLowerCase();
        return /save|preview|output|comparer|hud|toast|gallery|player|sidebar|view|combine|writer|exporter/.test(type);
    };
    const dead = nodes.filter((n) => {
        if ((n.mode || 0) === 4) return false;     // muted/bypass
        if (isOutputish(n)) return false;          // terminals are alive by definition
        if (!(n.outputs || []).length) return false; // pure-sink node — not dead, just terminal
        // Has outputs but NONE are connected anywhere → dead.
        const succs = succ.get(n.id);
        return !succs || succs.size === 0;
    }).map((n) => n.id);

    // ── ② Cycles via Tarjan SCC.
    let idx = 0;
    const indices = new Map(), lowlink = new Map(), onStack = new Map();
    const stk = [];
    const sccs = [];
    function strongconnect(v) {
        indices.set(v, idx); lowlink.set(v, idx); idx++;
        stk.push(v); onStack.set(v, true);
        for (const w of succ.get(v) || []) {
            if (!indices.has(w)) {
                strongconnect(w);
                lowlink.set(v, Math.min(lowlink.get(v), lowlink.get(w)));
            } else if (onStack.get(w)) {
                lowlink.set(v, Math.min(lowlink.get(v), indices.get(w)));
            }
        }
        if (lowlink.get(v) === indices.get(v)) {
            const comp = [];
            let w;
            do { w = stk.pop(); onStack.set(w, false); comp.push(w); } while (w !== v);
            if (comp.length > 1) sccs.push(comp);
        }
    }
    for (const n of nodes) if (!indices.has(n.id)) strongconnect(n.id);
    const cycleSet = new Set(sccs.flat());

    // ── ③ Dangling required inputs.
    // ComfyUI: only IMAGE/MASK/LATENT/MODEL/CLIP/VAE/CONDITIONING/AUDIO/etc. slots
    // are true wire-required. Primitive widget slots (INT/FLOAT/STRING/BOOLEAN/COMBO)
    // are widget-backed and never dangling. Also skip optional / converted-from-widget.
    const SLOT_TYPES = new Set([
        "IMAGE", "MASK", "LATENT", "MODEL", "CLIP", "VAE", "CONDITIONING",
        "AUDIO", "VIDEO", "CONTROL_NET", "STYLE_MODEL", "GLIGEN",
        "PHOTOMAKER", "UPSCALE_MODEL", "NOISE", "SAMPLER", "SIGMAS", "GUIDER",
        "FACE_ANALYSIS", "INSIGHTFACE", "BBOX", "SEGS",
    ]);
    const dangling = [];
    for (const n of nodes) {
        if ((n.mode || 0) === 4) continue;
        for (let i = 0; i < (n.inputs || []).length; i++) {
            const inp = n.inputs[i];
            if (!inp) continue;
            if (inp.link != null) continue;
            if (inp.optional || inp.widget) continue;
            const t = (inp.type || "").toString().toUpperCase();
            // Primitive widget types (despite being unconnected slots when shown
            // in some custom packs) are NEVER "dangling" — they are widgets.
            if (t === "INT" || t === "FLOAT" || t === "STRING" || t === "BOOLEAN" || t === "COMBO" || t === "*") continue;
            // Allow custom pack types only if they match a known data-slot type or
            // contain a comma (combo-as-slot pattern). Otherwise skip — too noisy.
            if (!SLOT_TYPES.has(t)) continue;
            dangling.push({ id: n.id, slot: i, name: inp.name });
        }
    }
    _result = { dead, cycles: sccs, dangling };
}

// ── Canvas paint hook ────────────────────────────────────────────────
function patchDraw() {
    const c = app.canvas;
    if (!c || c._c2c_health_painted) return;
    c._c2c_health_painted = true;
    const orig = c.drawNode;
    c.drawNode = function (node, ctx) {
        const r = orig.apply(this, arguments);
        if (!_result) return r;
        try {
            const isDead   = _result.dead.includes(node.id);
            const isCycle  = _result.cycles.some((cc) => cc.includes(node.id));
            const isDang   = _result.dangling.some((d) => d.id === node.id);
            if (!isDead && !isCycle && !isDang) return r;
            ctx.save();
            ctx.lineWidth = 2.0;
            if (isCycle) {
                ctx.strokeStyle = "rgba(255,107,107,0.85)";
                ctx.strokeRect(-3, -3, node.size[0] + 6, node.size[1] + 6);
                ctx.fillStyle = "rgba(255,107,107,0.85)";
                ctx.font = "12px sans-serif";
                ctx.fillText("↻", node.size[0] - 14, -8);
            } else if (isDead) {
                ctx.strokeStyle = "rgba(255,209,102,0.65)";
                ctx.setLineDash([6, 4]);
                ctx.strokeRect(-3, -3, node.size[0] + 6, node.size[1] + 6);
            }
            if (isDang) {
                const slots = _result.dangling.filter((d) => d.id === node.id);
                ctx.setLineDash([]);
                ctx.fillStyle = "rgba(255,107,107,0.95)";
                for (const d of slots) {
                    const sy = 14 + d.slot * 14 + 6;
                    ctx.beginPath(); ctx.arc(-6, sy, 3, 0, Math.PI * 2); ctx.fill();
                }
            }
            ctx.restore();
        } catch { /* */ }
        return r;
    };
}

// ── UI ───────────────────────────────────────────────────────────────
function ensurePill() {
    let p = document.getElementById(PILL_ID);
    if (p) return p;
    injectStyle();
    p = document.createElement("div");
    p.id = PILL_ID;
    p.innerHTML = `<span class="dot"></span><span class="lbl">…</span>`;
    document.body.appendChild(p);
    p.addEventListener("click", togglePanel);
    return p;
}
function ensurePanel() {
    let pn = document.getElementById(PANEL_ID);
    if (pn) return pn;
    pn = document.createElement("div");
    pn.id = PANEL_ID;
    pn.innerHTML = `<div class="hdr">Graph health</div><div class="body"></div>`;
    document.body.appendChild(pn);
    return pn;
}

function findNode(id) { return (app.graph?._nodes || []).find((n) => String(n.id) === String(id)); }
function pulse(node) {
    const c = app.canvas;
    if (!c || !node) return;
    c.selectNode?.(node, false);
    node._c2c_pulse_until = performance.now() + PULSE_MS;
    if (c.ds) {
        const cx = node.pos[0] + node.size[0] * 0.5;
        const cy = node.pos[1] + node.size[1] * 0.5;
        const dpr = window.devicePixelRatio || 1;
        c.ds.scale = Math.max(0.55, Math.min(1.2, c.ds.scale || 1));
        c.ds.offset[0] = -cx + (c.canvas.width  / dpr) / (2 * c.ds.scale);
        c.ds.offset[1] = -cy + (c.canvas.height / dpr) / (2 * c.ds.scale);
    }
    c.setDirty(true, true);
}

function renderUI() {
    let enabled = true;
    try { enabled = app.ui?.settings?.getSettingValue?.(SETTING_ID, true) !== false; } catch { /* */ }
    const pill = ensurePill();
    if (!enabled) { pill.style.display = "none"; const pn = document.getElementById(PANEL_ID); if (pn) pn.classList.remove("open"); return; }
    pill.style.display = "flex";

    const nDead = _result.dead.length, nCyc = _result.cycles.length, nDang = _result.dangling.length;
    pill.classList.remove("ok", "warn", "err");
    if (nCyc > 0)                pill.classList.add("err");
    else if (nDead + nDang > 0)  pill.classList.add("warn");
    else                         pill.classList.add("ok");
    pill.querySelector(".lbl").textContent = `${nCyc} cycle${nCyc === 1 ? "" : "s"} · ${nDead} dead · ${nDang} dangling`;

    const pn = document.getElementById(PANEL_ID);
    if (pn && pn.classList.contains("open")) renderPanel();
}

function renderPanel() {
    const pn = ensurePanel();
    const body = pn.querySelector(".body");
    const rows = [];
    for (const cc of _result.cycles) {
        rows.push({ ids: cc, tag: "cycle", text: `Cycle: ${cc.map((id) => findNode(id)?.title || findNode(id)?.type || "?").join(" ↻ ")}` });
    }
    for (const id of _result.dead) {
        const n = findNode(id);
        rows.push({ ids: [id], tag: "dead", text: `Dead: #${id} ${n?.title || n?.type || "?"}` });
    }
    for (const d of _result.dangling) {
        const n = findNode(d.id);
        rows.push({ ids: [d.id], tag: "dang", text: `Dangling input '${d.name}' on #${d.id} ${n?.title || n?.type || "?"}` });
    }
    if (!rows.length) {
        body.innerHTML = `<div style="padding:14px;color:#7ee0a8;text-align:center">All clear ✔</div>`;
        return;
    }
    body.innerHTML = rows.map((r) =>
        `<div class="row" data-id="${r.ids[0]}">
             <div class="tag ${r.tag}">${r.tag}</div>
             <div class="t">${r.text}</div>
         </div>`
    ).join("");
    body.querySelectorAll(".row").forEach((el) => {
        el.addEventListener("click", () => {
            const n = findNode(el.dataset.id);
            if (n) pulse(n);
        });
    });
}

function togglePanel() {
    const pn = ensurePanel();
    pn.classList.toggle("open");
    if (pn.classList.contains("open")) renderPanel();
}

function schedule() {
    if (_scheduled) return;
    _scheduled = setTimeout(() => {
        _scheduled = 0;
        try { analyze(); } catch (e) { console.warn("[C2C.GraphHealth] analyze failed:", e); }
        renderUI();
        app.canvas?.setDirty(true, true);
    }, 400);
}

app.registerExtension({
    name: "C2C.GraphHealth",
    async setup() {
        try {
            app.ui.settings.addSetting({
                id: SETTING_ID,
                name: "Graph health overlay (dead/cycle/dangling)",
                type: "boolean", defaultValue: true,
                category: ["c2c", "Diagnostics", "Graph Health"],
                onChange: renderUI,
            });
        } catch { /* */ }

        // Patch loadGraphData + node add/remove + link change for invalidation.
        const _origLoad = app.loadGraphData?.bind(app);
        if (typeof _origLoad === "function") {
            app.loadGraphData = function (...args) {
                const r = _origLoad(...args);
                schedule();
                return r;
            };
        }
        const G = window.LGraph?.prototype;
        if (G && !G._c2c_health_patched) {
            for (const m of ["add", "remove", "connect", "disconnect"]) {
                const o = G[m];
                if (typeof o === "function") {
                    G[m] = function (...args) {
                        const r = o.apply(this, args);
                        schedule();
                        return r;
                    };
                }
            }
            G._c2c_health_patched = true;
        }
        // Initial.
        setTimeout(() => { patchDraw(); schedule(); }, 600);
        // Safety re-paint hook (canvas may be reconstructed).
        setInterval(() => patchDraw(), 3000);
        console.log("[C2C.GraphHealth] ready.");
    },
});
