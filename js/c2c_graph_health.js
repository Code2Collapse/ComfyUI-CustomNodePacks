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
//   • Publishes counts + helpers on `window.__C2C_GRAPH_HEALTH__` and
//     dispatches a `c2c:graph-health-updated` event.
//     P0.2 Phase 3 migration (2026-05-25):
//        — Standalone status pill and panel have been retired. The INT
//          badge in the OmniBar stats section now hosts the breakdown.
//          This module remains the analyser + canvas painter only.
//   • Toggle: c2c.graphHealth.enabled. Default ON.
//
// License: Apache-2.0
// ---------------------------------------------------------------------

import { app } from "../../scripts/app.js";

const SETTING_ID = "c2c.graphHealth.enabled";
const PULSE_MS = 1100;

let _result = { dead: [], cycles: [], dangling: [] };
// Derived O(1) lookups, rebuilt ONLY when _result changes (in analyze()).
// drawNode runs per-node, per-frame during drag/pan; the old array
// .includes()/.some() there was O(nodes × cycles × len) and pegged the renderer
// (~151% CPU on a 130-node graph while moving the canvas). Sets/Map make each
// per-node test O(1).
let _deadSet = new Set();
let _cycleSet = new Set();
let _dangByNode = new Map();   // nodeId -> [{ id, slot, name }, ...]
let _scheduled = 0;
let _enabled = true;

function injectStyle() {
    if (document.getElementById("c2c-graph-health-style")) return;
    const s = document.createElement("style");
    s.id = "c2c-graph-health-style";
    s.textContent = "/* c2c_graph_health: canvas-only painter; no DOM pill */";
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

    // ── Virtual routing buses (KJNodes / rgthree Set→Get, Use-Everywhere) ──
    // These carry a value with NO literal link: a Set publishes a variable and a
    // matching Get pulls it by key. Without modelling that edge, every Set node
    // looks "dead" (its output connects to nothing) — a false positive the user hit.
    // Detect Set/Get by class/title + key widget and add Set→Get edges so the bus
    // is reachable exactly like a real wire. OS-agnostic (pure graph logic).
    const _BUS_KEY_WIDGETS = ["Constant", "constant", "value", "Key", "key",
                              "Name", "name", "variable", "Variable", "label", "Label"];
    const _busKey = (n) => {
        const w = (n.widgets || []).find((x) => _BUS_KEY_WIDGETS.includes(x.name));
        if (w && w.value != null && String(w.value).trim()) return String(w.value).trim().toLowerCase();
        const m = (n.title || "").match(/^(?:set|get)[\s_:\-]+(.+)$/i);
        return m ? m[1].trim().toLowerCase() : "";
    };
    const _isSet = (n) => {
        const c = (n.comfyClass || n.type || "").toLowerCase(), t = (n.title || "").toLowerCase();
        return c === "setnode" || c.includes("setnode") || t.startsWith("set ") || t.startsWith("set_") || t.startsWith("set:");
    };
    const _isGet = (n) => {
        const c = (n.comfyClass || n.type || "").toLowerCase(), t = (n.title || "").toLowerCase();
        return c === "getnode" || c.includes("getnode") || t.startsWith("get ") || t.startsWith("get_") || t.startsWith("get:");
    };
    // Routing/virtual nodes that are NEVER "dead" or "dangling" (they have no
    // literal wires by design): Set/Get buses, reroutes, Use-Everywhere, primitives, notes.
    const _isVirtual = (n) => {
        if (_isSet(n) || _isGet(n)) return true;
        const blob = ((n.comfyClass || n.type || "") + " " + (n.title || "")).toLowerCase();
        return /reroute|everywhere|primitivenode|^note$|\bnote\b|\bnotenode\b/.test(blob);
    };
    const _setsByKey = new Map();
    for (const n of nodes) {
        if (!_isSet(n)) continue;
        const k = _busKey(n);
        if (!k) continue;
        if (!_setsByKey.has(k)) _setsByKey.set(k, []);
        _setsByKey.get(k).push(n.id);
    }
    for (const n of nodes) {
        if (!_isGet(n)) continue;
        const srcs = _setsByKey.get(_busKey(n));
        if (!srcs) continue;
        for (const sid of srcs) {
            if (succ.has(sid) && pred.has(n.id)) { succ.get(sid).add(n.id); pred.get(n.id).add(sid); }
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
        if (_isVirtual(n)) return false;           // Set/Get/reroute/UE/primitive/note route via the bus, not links
        if (isOutputish(n)) return false;          // terminals are alive by definition
        if (!(n.outputs || []).length) return false; // pure-sink node — not dead, just terminal
        // Has outputs but NONE are connected anywhere (incl. virtual bus) → dead.
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
        if (_isVirtual(n)) continue;   // Get nodes are sources; Set inputs come via the bus — never dangling
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

    // Rebuild O(1) lookups ONCE here (on graph change), never per frame.
    _deadSet = new Set(dead);
    _cycleSet = new Set();
    for (const comp of sccs) for (const id of comp) _cycleSet.add(id);
    _dangByNode = new Map();
    for (const d of dangling) {
        let arr = _dangByNode.get(d.id);
        if (!arr) { arr = []; _dangByNode.set(d.id, arr); }
        arr.push(d);
    }
}
// ── Canvas paint helper: resolve CSS var so color-mix() works in canvas ──
function tok(name, pct) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "var(--c2c-black)";
    return `color-mix(in srgb, ${v} ${pct}%, transparent)`;
}
// Cache the three resolved overlay colors. tok() calls getComputedStyle (a
// forced style/layout read) — doing that per flagged node per frame was a second
// hot-path cost. Resolve once; invalidate on theme change.
let _colCache = null;
function _colors() {
    if (_colCache) return _colCache;
    _colCache = {
        cycle: tok("--c2c-danger", 85),
        dead:  tok("--c2c-warn", 65),
        dang:  tok("--c2c-danger", 95),
    };
    return _colCache;
}
try { window.addEventListener("c2c:theme-changed", () => { _colCache = null; }); } catch (_) { /* no-op */ }
// ── Canvas paint hook ────────────────────────────────────────────────
function patchDraw() {
    const c = app.canvas;
    if (!c || c._c2c_health_painted) return;
    c._c2c_health_painted = true;
    const orig = c.drawNode;
    c.drawNode = function (node, ctx) {
        const r = orig.apply(this, arguments);
        // O(1) fast bail: healthy graph → zero per-node work (the common case).
        if (_deadSet.size === 0 && _cycleSet.size === 0 && _dangByNode.size === 0) return r;
        try {
            const id = node.id;
            const isCycle = _cycleSet.has(id);     // O(1)  (was O(cycles × len))
            const isDead  = _deadSet.has(id);      // O(1)  (was O(dead))
            const dangArr = _dangByNode.get(id);   // O(1)  (was O(dangling) ×2)
            if (!isDead && !isCycle && !dangArr) return r;
            const COL = _colors();                 // colors resolved once, not per node
            ctx.save();
            ctx.lineWidth = 2.0;
            // Box must track the COLLAPSED pill, not the (still-full) node.size.
            const collapsed = !!node.flags?.collapsed;
            const TITLE_H = (window.LiteGraph?.NODE_TITLE_HEIGHT) || 30;
            const boxW = collapsed
                ? (node._collapsed_width || (window.LiteGraph?.NODE_COLLAPSED_WIDTH) || 80)
                : node.size[0];
            const boxY = collapsed ? -TITLE_H - 3 : -3;
            const boxH = collapsed ? TITLE_H : node.size[1];
            if (isCycle) {
                ctx.strokeStyle = COL.cycle;
                ctx.strokeRect(-3, boxY, boxW + 6, boxH + 6);
                ctx.fillStyle = COL.cycle;
                ctx.font = "12px sans-serif";
                ctx.fillText("↻", boxW - 14, collapsed ? -TITLE_H + 8 : -8);
            } else if (isDead) {
                ctx.strokeStyle = COL.dead;
                ctx.setLineDash([6, 4]);
                ctx.strokeRect(-3, boxY, boxW + 6, boxH + 6);
            }
            if (dangArr && !collapsed) {
                ctx.setLineDash([]);
                ctx.fillStyle = COL.dang;
                for (const d of dangArr) {
                    const sy = 14 + d.slot * 14 + 6;
                    ctx.beginPath(); ctx.arc(-6, sy, 3, 0, Math.PI * 2); ctx.fill();
                }
            }
            ctx.restore();
        } catch { /* */ }
        return r;
    };
}

// ── UI (retired — INT badge in OmniBar hosts the breakdown) ──────────
function ensurePill() { return null; }
function ensurePanel() { return null; }
function togglePanel() { /* no-op: INT popover replaces this */ }

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
    try { _enabled = app.ui?.settings?.getSettingValue?.(SETTING_ID, true) !== false; } catch { /* */ }
    // Always publish; consumers (INT badge popover) decide what to render.
    _publishGlobal();
    // Tear down any legacy DOM nodes if a previous boot of this module
    // left them around.
    const stalePill  = document.getElementById("c2c-graph-health-pill");
    if (stalePill) stalePill.remove();
    const stalePanel = document.getElementById("c2c-graph-health-panel");
    if (stalePanel) stalePanel.remove();
    app.canvas?.setDirty(true, true);
}

function renderPanel() { /* retired — INT popover renders the list */ }

function _publishGlobal() {
    const payload = {
        cycles: _result.cycles.map((cc) => cc.slice()),
        dead: _result.dead.slice(),
        dangling: _result.dangling.map((d) => ({ id: d.id, name: d.name })),
        enabled: _enabled,
        items: _buildRowItems(),
        focus: (id) => {
            const n = findNode(id);
            if (n) pulse(n);
        },
    };
    try { window.__C2C_GRAPH_HEALTH__ = payload; } catch { /* */ }
    try { window.dispatchEvent(new CustomEvent("c2c:graph-health-updated", { detail: payload })); } catch { /* */ }
}

function _buildRowItems() {
    const rows = [];
    for (const cc of _result.cycles) {
        rows.push({
            id: cc[0],
            ids: cc.slice(),
            tag: "cycle",
            text: `Cycle: ${cc.map((id) => findNode(id)?.title || findNode(id)?.type || "?").join(" \u21bb ")}`,
        });
    }
    for (const id of _result.dead) {
        const n = findNode(id);
        rows.push({ id, ids: [id], tag: "dead", text: `Dead: #${id} ${n?.title || n?.type || "?"}` });
    }
    for (const d of _result.dangling) {
        const n = findNode(d.id);
        rows.push({ id: d.id, ids: [d.id], tag: "dang", text: `Dangling input '${d.name}' on #${d.id} ${n?.title || n?.type || "?"}` });
    }
    return rows;
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
                let r;
                try { r = _origLoad(...args); }
                catch (err) {
                    log.warn?.("[graph_health] loadGraphData upstream threw", err);
                    throw err;
                }
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
